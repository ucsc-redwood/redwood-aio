#!/usr/bin/env python3
import json
from pathlib import Path

HARDWARE_MAP = {
    "little":  "ProcessorType::kLittleCore",
    "medium":  "ProcessorType::kMediumCore",
    "big":     "ProcessorType::kBigCore",
    "gpu":     "GPU_PLACEHOLDER"  # handle separately
}

def parse_schedule_filename(filename: str):
    """
    Parse something like "3A021JEHN02756_CifarDense_schedule_001.json"
    into (device_id, application_name, schedule_id_stem).

    Returns a tuple of (device_id, application_name, schedule_id),
    or raises ValueError if not matching.
    """
    stem = Path(filename).stem  # e.g. 3A021JEHN02756_CifarDense_schedule_001
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename '{filename}' does not match <device>_<application>_schedule_<num>.json")
    device_id = parts[0]
    application_name = parts[1]
    # entire stem is the schedule_id
    schedule_id = stem
    return device_id, application_name, schedule_id


def read_schedule_file(schedule_path: Path):
    """
    Reads the JSON from 'schedule_path' and returns (schedule_obj, total_stages).
    Raises exceptions if invalid.
    """
    with open(schedule_path, "r") as f:
        data = json.load(f)
    if "schedule" not in data:
        raise ValueError(f"JSON file {schedule_path} missing 'schedule' key.")
    schedule_obj = data["schedule"]

    # Verify at least 2 total stages
    total_stages = 0
    for ch in schedule_obj["chunks"]:
        total_stages += len(ch["stages"])
    if total_stages < 2:
        raise ValueError(
            f"Schedule {schedule_path} has only {total_stages} stages; must have >= 2."
        )

    return schedule_obj, total_stages


def generate_run_pipeline_code(schedule_obj: dict) -> str:
    """
    Returns the entire code for:

    void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) { ... }

    with chunk_first / chunk_middle / chunk_last usage.
    """
    chunks = schedule_obj["chunks"]
    lines = []
    lines.append("void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)")
    lines.append("{")

    num_chunks = len(chunks)
    # concurrency queues for chunk boundaries
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
        end_stage   = stages[-1]

        if hw == "gpu":
            stage_func = f"run_gpu_stages<{start_stage}, {end_stage}>"
        else:
            pt_enum = HARDWARE_MAP.get(hw, "ProcessorType::kUnknown")
            stage_func = f"run_cpu_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>"

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
    for tvar in thread_names:
        lines.append(f"  {tvar}.join();")

    lines.append("}")
    return "\n".join(lines)


def build_single_hpp_content(device_id: str,
                             schedule_id: str,
                             application_name: str,
                             pipeline_code: str) -> str:
    """
    Returns the entire .hpp file content with includes, namespace, etc.
    We wrap the pipeline_code in:

      namespace device_<device_id> {
      namespace schedule_<schedule_id> {
         ...
      }
      }

    """
    code_lines = []
    code_lines.append(f"// Auto-generated header for schedule: {schedule_id}")
    code_lines.append("//")
    code_lines.append(f"// Device: {device_id}, Application: {application_name}")
    code_lines.append("")
    code_lines.append("#pragma once")
    code_lines.append("")
    code_lines.append("#include <queue>")
    code_lines.append("#include <thread>")
    code_lines.append("#include <concurrentqueue.h>")
    code_lines.append('#include "../task.hpp"')
    code_lines.append('#include "../templates.hpp"')
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
