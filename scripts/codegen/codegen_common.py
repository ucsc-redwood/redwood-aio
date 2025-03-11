#!/usr/bin/env python3
import json
from pathlib import Path

###############################################################################
# Global map for CPU hardware -> ProcessorType
###############################################################################
HARDWARE_MAP = {
    "little": "ProcessorType::kLittleCore",
    "medium": "ProcessorType::kMediumCore",
    "big": "ProcessorType::kBigCore",
    "gpu": "GPU_PLACEHOLDER",  # We'll handle GPU specially
}


###############################################################################
# Parsing
###############################################################################
def parse_schedule_filename(filename: str):
    """
    Parse something like "3A021JEHN02756_CifarDense_schedule_001.json"
    into (device_id, application_name, schedule_id_stem).

    Returns (device_id, application_name, schedule_id).
    Raises ValueError if not matching.
    """
    stem = Path(filename).stem  # e.g. "3A021JEHN02756_CifarDense_schedule_001"
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Filename '{filename}' does not match <device>_<application>_schedule_<num>.json"
        )
    device_id = parts[0]
    application_name = parts[1]
    schedule_id = stem
    return device_id, application_name, schedule_id


def read_schedule_file(schedule_path: Path):
    """
    Reads the JSON from 'schedule_path' => (schedule_obj, total_stages).
    Raises ValueError if there's an issue (fewer than 2 total stages, etc.)
    """
    with open(schedule_path, "r") as f:
        data = json.load(f)

    if "schedule" not in data:
        raise ValueError(f"JSON file {schedule_path} missing 'schedule' key.")
    schedule_obj = data["schedule"]

    # Verify at least 2 total stages overall
    total_stages = sum(len(ch["stages"]) for ch in schedule_obj["chunks"])
    if total_stages < 2:
        raise ValueError(
            f"Schedule {schedule_path} has only {total_stages} stage(s); must have >= 2."
        )

    return schedule_obj, total_stages


###############################################################################
# Code Generation
###############################################################################
def generate_run_pipeline_code(schedule_obj: dict) -> str:
    """
    Returns the C++ code for:

        inline void run_pipeline(const int num_tasks) { ... }

    using the chunk template function with specific processor types and thread counts.
    """
    chunks = schedule_obj["chunks"]
    num_chunks = len(chunks)

    lines = []
    # Add inline to the function declaration
    lines.append(
        "inline void run_pipeline(const int num_tasks)"
    )
    lines.append("{")
    lines.append("  cuda::CudaManager mgr;")
    lines.append("")
    lines.append("  // Preallocate data for all tasks")
    lines.append("  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);")
    lines.append("")
    lines.append("  // Initialize input queue with tasks")
    lines.append("  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);")
    lines.append("")
    
    # If exactly one chunk => use single thread
    if num_chunks == 1:
        c = chunks[0]
        hw = c["hardware"].lower()
        start_stage = c["stages"][0]
        end_stage = c["stages"][-1]
        threads = c["threads"]

        if hw == "gpu":
            stage_func = f"cuda::run_multiple_stages<{start_stage}, {end_stage}>"
        else:
            pt_enum = HARDWARE_MAP.get(hw, "ProcessorType::kUnknown")
            stage_func = (
                f"omp::run_multiple_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>"
            )

        lines.append("  auto start = std::chrono::high_resolution_clock::now();")
        lines.append("")
        lines.append("  std::thread t_only([&]() {")
        lines.append(f"    chunk<Task, AppData>(q_input, nullptr, {stage_func}, mgr);")
        lines.append("  });")
        lines.append("")
        lines.append("  t_only.join();")
        
        lines.append("")
        lines.append("  auto end = std::chrono::high_resolution_clock::now();")
        lines.append("  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);")
        lines.append("  spdlog::info(\"Time taken per task: {} ms\", duration.count() / num_tasks);")
        
        lines.append("}")
        return "\n".join(lines)

    # Otherwise, multi-chunk
    # Create concurrency queues between chunk i and i+1
    for i in range(num_chunks - 1):
        lines.append(f"  moodycamel::ConcurrentQueue<Task *> q_{i}_{i+1};")
    lines.append("")
    
    lines.append("  auto start = std::chrono::high_resolution_clock::now();")
    lines.append("")

    thread_names = []
    for i, chunk in enumerate(chunks):
        name = chunk["name"] if "name" in chunk else f"t{i+1}"
        hw = chunk["hardware"].lower()
        threads = chunk["threads"]
        stages = chunk["stages"]
        start_stage = stages[0]
        end_stage = stages[-1]

        if hw == "gpu":
            stage_func = f"cuda::run_multiple_stages<{start_stage}, {end_stage}>"
        else:
            pt_enum = HARDWARE_MAP.get(hw, "ProcessorType::kUnknown")
            stage_func = (
                f"omp::run_multiple_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>"
            )

        tvar = f"t{i+1}"
        thread_names.append(tvar)

        if i == 0:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(f"    chunk<Task, AppData>(q_input, &q_{i}_{i+1}, {stage_func}, mgr);")
            lines.append("  });")
        elif i == num_chunks - 1:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(f"    chunk<Task, AppData>(q_{i-1}_{i}, nullptr, {stage_func}, mgr);")
            lines.append("  });")
        else:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(f"    chunk<Task, AppData>(q_{i-1}_{i}, &q_{i}_{i+1}, {stage_func}, mgr);")
            lines.append("  });")

    lines.append("")
    # Join them
    for tvar in thread_names:
        lines.append(f"  {tvar}.join();")

    lines.append("")
    lines.append("  auto end = std::chrono::high_resolution_clock::now();")
    lines.append("  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);")
    lines.append("  spdlog::info(\"Time taken per task: {} ms\", duration.count() / num_tasks);")

    lines.append("}")
    return "\n".join(lines)


def build_single_hpp_content(
    device_id: str, schedule_id: str, application_name: str, pipeline_code: str
) -> str:
    """
    Returns a single .hpp containing the includes plus:

    namespace device_<device_id> {
    namespace schedule_<schedule_id> {
      inline void run_pipeline(std::queue<Task>&, std::queue<Task>&);
    }
    }

    This is for a single schedule.
    """
    # Determine the app-specific include based on application_name
    app_include = ""
    app_data_typedef = ""
    
    # Convert application_name to lowercase for consistency
    app_name_lower = application_name.lower()
    
    if "cifardense" in app_name_lower:
        app_include = '#include "builtin-apps/cifar-dense/dense_appdata.hpp"'
        app_data_typedef = "using AppData = cifar_dense::AppData;"
    elif "cifarsparse" in app_name_lower:
        app_include = '#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"'
        app_data_typedef = "using AppData = cifar_sparse::AppData;"
    elif "tree" in app_name_lower:
        app_include = '#include "builtin-apps/tree/tree_appdata.hpp"'
        app_data_typedef = "using AppData = tree::AppData;"
    else:
        # Default case
        app_include = f'// Warning: Unknown application type: {application_name}'
        app_data_typedef = f'// Warning: No AppData typedef available for: {application_name}'
    
    code_lines = []
    code_lines.append(f"// Auto-generated header for schedule: {schedule_id}")
    code_lines.append(f"// Device: {device_id}, Application: {application_name}")
    code_lines.append("")
    code_lines.append("#pragma once")
    code_lines.append("")
    code_lines.append("#include <queue>")
    code_lines.append("#include <thread>")
    code_lines.append("#include <chrono>")
    code_lines.append("#include <concurrentqueue.h>")
    code_lines.append('#include <spdlog/spdlog.h>')
    code_lines.append("")
    code_lines.append('#include "../task.hpp"')
    code_lines.append('#include "../../templates.hpp"')
    code_lines.append('#include "../run_stages.hpp"')
    code_lines.append('#include "builtin-apps/common/cuda/manager.cuh"')
    code_lines.append(app_include)
    code_lines.append("")
    code_lines.append(f"namespace device_{device_id} {{")
    code_lines.append(f"namespace schedule_{schedule_id} {{")
    code_lines.append("")
    code_lines.append(app_data_typedef)
    code_lines.append("")
    code_lines.append(pipeline_code)
    code_lines.append("")
    code_lines.append(f"}}  // namespace schedule_{schedule_id}")
    code_lines.append(f"}}  // namespace device_{device_id}")
    code_lines.append("")
    return "\n".join(code_lines)
