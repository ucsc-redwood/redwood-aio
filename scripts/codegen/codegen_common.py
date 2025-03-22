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
    # GPU types are handled specially during code generation
    "gpu_cuda": "GPU_CUDA",
    "gpu_vulkan": "GPU_VULKAN",
    "gpu": "GPU_CUDA",  # Default to CUDA for backward compatibility
}


###############################################################################
# Parsing
###############################################################################


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

    # Determine if we need CUDA or Vulkan or both
    uses_cuda = any(
        chunk["hardware"].lower() == "gpu_cuda" or chunk["hardware"].lower() == "gpu"
        for chunk in chunks
    )
    uses_vulkan = any(chunk["hardware"].lower() == "gpu_vulkan" for chunk in chunks)

    lines = []
    # Add inline to the function declaration
    lines.append("inline void run_pipeline(const int num_tasks)")
    lines.append("{")

    # Add appropriate GPU manager based on what's used
    if uses_cuda:
        lines.append("  cuda::CudaManager cuda_mgr;")
    if uses_vulkan:
        lines.append("  vk::VulkanManager vk_mgr;")
    lines.append("")

    # Determine which manager to use for preallocation
    if uses_cuda:
        lines.append("  // Preallocate data for all tasks")
        lines.append(
            "  auto preallocated_data = init_appdata<AppData>(&cuda_mgr.get_mr(), num_tasks);"
        )
        lines.append("")
        lines.append("  // Initialize input queue with tasks")
        lines.append(
            "  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &cuda_mgr);"
        )
    elif uses_vulkan:
        lines.append("  // Preallocate data for all tasks")
        lines.append(
            "  auto preallocated_data = init_appdata<AppData>(&vk_mgr.get_mr(), num_tasks);"
        )
        lines.append("")
        lines.append("  // Initialize input queue with tasks")
        lines.append(
            "  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &vk_mgr);"
        )
    else:
        # CPU-only case
        lines.append("  // Preallocate data for all tasks (CPU only)")
        lines.append(
            "  auto preallocated_data = init_appdata<AppData>(nullptr, num_tasks);"
        )
        lines.append("")
        lines.append("  // Initialize input queue with tasks")
        lines.append(
            "  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, nullptr);"
        )

    lines.append("")

    # If exactly one chunk => use single thread
    if num_chunks == 1:
        c = chunks[0]
        hw = c["hardware"].lower()
        start_stage = c["stages"][0]
        end_stage = c["stages"][-1]
        threads = c["threads"]

        # Determine which GPU namespace to use based on hardware type
        if hw == "gpu_cuda" or hw == "gpu":
            stage_func = f"cuda::run_multiple_stages<{start_stage}, {end_stage}>"
            mgr_var = "cuda_mgr"
        elif hw == "gpu_vulkan":
            stage_func = f"vk::run_multiple_stages<{start_stage}, {end_stage}>"
            mgr_var = "vk_mgr"
        else:
            pt_enum = HARDWARE_MAP.get(hw, "ProcessorType::kUnknown")
            stage_func = f"omp::run_multiple_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>"
            mgr_var = (
                "cuda_mgr" if uses_cuda else "vk_mgr" if uses_vulkan else "nullptr"
            )

        lines.append("  auto start = std::chrono::high_resolution_clock::now();")
        lines.append("")
        lines.append("  std::thread t_only([&]() {")
        lines.append(
            f"    chunk<Task, AppData>(q_input, nullptr, {stage_func}, {mgr_var});"
        )
        lines.append("  });")
        lines.append("")
        lines.append("  t_only.join();")

        lines.append("")
        lines.append("  auto end = std::chrono::high_resolution_clock::now();")
        lines.append(
            "  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);"
        )
        lines.append(
            '  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);'
        )

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

        # Determine which GPU namespace to use based on hardware type
        if hw == "gpu_cuda" or hw == "gpu":
            stage_func = f"cuda::run_multiple_stages<{start_stage}, {end_stage}>"
            mgr_var = "cuda_mgr"
        elif hw == "gpu_vulkan":
            stage_func = f"vk::run_multiple_stages<{start_stage}, {end_stage}>"
            mgr_var = "vk_mgr"
        else:
            pt_enum = HARDWARE_MAP.get(hw, "ProcessorType::kUnknown")
            stage_func = f"omp::run_multiple_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>"
            mgr_var = (
                "cuda_mgr" if uses_cuda else "vk_mgr" if uses_vulkan else "nullptr"
            )

        tvar = f"t{i+1}"
        thread_names.append(tvar)

        if i == 0:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(
                f"    chunk<Task, AppData>(q_input, &q_{i}_{i+1}, {stage_func}, {mgr_var});"
            )
            lines.append("  });")
        elif i == num_chunks - 1:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(
                f"    chunk<Task, AppData>(q_{i-1}_{i}, nullptr, {stage_func}, {mgr_var});"
            )
            lines.append("  });")
        else:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(
                f"    chunk<Task, AppData>(q_{i-1}_{i}, &q_{i}_{i+1}, {stage_func}, {mgr_var});"
            )
            lines.append("  });")

    lines.append("")
    # Join them
    for tvar in thread_names:
        lines.append(f"  {tvar}.join();")

    lines.append("")
    lines.append("  auto end = std::chrono::high_resolution_clock::now();")
    lines.append(
        "  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);"
    )
    lines.append(
        '  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);'
    )

    lines.append("}")
    return "\n".join(lines)


def build_single_hpp_content(
    device_id: str,
    schedule_id: str,
    application_name: str,
    pipeline_code: str,
    schedule_obj: dict,
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
    # Determine if we need CUDA or Vulkan or both
    chunks = schedule_obj["chunks"]
    uses_cuda = any(
        chunk["hardware"].lower() == "gpu_cuda" or chunk["hardware"].lower() == "gpu"
        for chunk in chunks
    )
    uses_vulkan = any(chunk["hardware"].lower() == "gpu_vulkan" for chunk in chunks)

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
        app_include = f"// Warning: Unknown application type: {application_name}"
        app_data_typedef = (
            f"// Warning: No AppData typedef available for: {application_name}"
        )

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
    code_lines.append("#include <spdlog/spdlog.h>")
    code_lines.append("")
    code_lines.append('#include "../task.hpp"')
    code_lines.append('#include "../../templates.hpp"')
    code_lines.append('#include "../run_stages.hpp"')

    # Include GPU headers as needed
    if uses_cuda:
        code_lines.append('#include "builtin-apps/common/cuda/manager.cuh"')
    if uses_vulkan:
        code_lines.append('#include "builtin-apps/common/vulkan/manager.hpp"')

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
