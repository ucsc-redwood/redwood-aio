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

        inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) { ... }

    using chunk_first, chunk_middle, chunk_last (for multi-chunk) or
    chunk_single (if exactly one chunk).
    """
    chunks = schedule_obj["chunks"]
    num_chunks = len(chunks)

    lines = []
    # Add inline to the function declaration
    lines.append(
        "inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)"
    )
    lines.append("{")

    # If exactly one chunk => use chunk_single
    if num_chunks == 1:
        c = chunks[0]
        hw = c["hardware"].lower()
        start_stage = c["stages"][0]
        end_stage = c["stages"][-1]
        threads = c["threads"]

        if hw == "gpu":
            stage_func = f"run_gpu_stages<{start_stage}, {end_stage}>"
        else:
            pt_enum = HARDWARE_MAP.get(hw, "ProcessorType::kUnknown")
            stage_func = (
                f"run_cpu_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>"
            )

        lines.append("  std::thread t_only([&]() {")
        lines.append(f"    chunk_single(tasks, out_tasks, {stage_func});")
        lines.append("  });")
        lines.append("  t_only.join();")
        lines.append("}")
        return "\n".join(lines)

    # Otherwise, multi-chunk
    # Create concurrency queues between chunk i and i+1
    for i in range(num_chunks - 1):
        lines.append(f"  moodycamel::ConcurrentQueue<Task> q_{i}_{i+1};")
    lines.append("")

    thread_names = []
    for i, chunk in enumerate(chunks):
        name = chunk["name"]
        hw = chunk["hardware"].lower()
        threads = chunk["threads"]
        stages = chunk["stages"]
        start_stage = stages[0]
        end_stage = stages[-1]

        if hw == "gpu":
            stage_func = f"run_gpu_stages<{start_stage}, {end_stage}>"
        else:
            pt_enum = HARDWARE_MAP.get(hw, "ProcessorType::kUnknown")
            stage_func = (
                f"run_cpu_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>"
            )

        tvar = f"t_{name}"
        thread_names.append(tvar)

        if i == 0:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(f"    chunk_first(tasks, q_{i}_{i+1}, {stage_func});")
            lines.append("  });")
        elif i == num_chunks - 1:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(f"    chunk_last(q_{i-1}_{i}, out_tasks, {stage_func});")
            lines.append("  });")
        else:
            lines.append(f"  std::thread {tvar}([&]() {{")
            lines.append(f"    chunk_middle(q_{i-1}_{i}, q_{i}_{i+1}, {stage_func});")
            lines.append("  });")

    lines.append("")
    # Join them
    for tvar in thread_names:
        lines.append(f"  {tvar}.join();")

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
    code_lines = []
    code_lines.append(f"// Auto-generated header for schedule: {schedule_id}")
    code_lines.append(f"// Device: {device_id}, Application: {application_name}")
    code_lines.append("")
    code_lines.append("#pragma once")
    code_lines.append("")
    code_lines.append("#include <queue>")
    code_lines.append("#include <thread>")
    code_lines.append("#include <concurrentqueue.h>")
    code_lines.append('#include "../task.hpp"')
    # Use "../../templates.hpp" as requested
    code_lines.append(
        '#include "../../templates.hpp"  // chunk_first, chunk_middle, chunk_last, chunk_single'
    )
    code_lines.append('#include "../run_stages.hpp"')
    code_lines.append("")
    code_lines.append(f"namespace device_{device_id} {{")
    code_lines.append(f"namespace schedule_{schedule_id} {{")
    code_lines.append("")
    code_lines.append(pipeline_code)
    code_lines.append("")
    code_lines.append(f"}}  // namespace schedule_{schedule_id}")
    code_lines.append(f"}}  // namespace device_{device_id}")
    code_lines.append("")
    return "\n".join(code_lines)
