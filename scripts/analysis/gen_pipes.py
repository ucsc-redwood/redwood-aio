#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

HARDWARE_MAP = {
    "little": "ProcessorType::kLittleCore",
    "medium": "ProcessorType::kMediumCore",
    "big": "ProcessorType::kBigCore",
    "gpu": "ProcessorType::kGPU",
}

def generate_schedule_header(schedule_obj: dict) -> str:
    """
    Generate the header declarations for each schedule.
    We'll produce one chunk function for each chunk in the schedule,
    plus a run_pipeline function. Each schedule is wrapped in a
    sub-namespace inside the device namespace.
    """
    schedule_id = schedule_obj["schedule_id"]
    device_id = schedule_obj["device_id"]
    chunks = schedule_obj["chunks"]

    # Derive sub-schedule name (if schedule_id starts with device_id + "_", remove that prefix)
    if schedule_id.startswith(device_id + "_"):
        sub_schedule_id = schedule_id[len(device_id) + 1:]
    else:
        sub_schedule_id = schedule_id

    lines = []
    lines.append(f"namespace {sub_schedule_id} {{")
    lines.append("")
    lines.append(f'constexpr const char* kScheduleId = "{schedule_id}";')
    lines.append("")

    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        # Decide function signature based on position in pipeline
        if num_chunks == 1:
            # Single-chunk schedule => vector -> vector
            lines.append(
                f"void chunk_{chunk_name}(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks);"
            )
        elif i == 0:
            # First chunk => vector -> concurrent queue
            lines.append(
                f"void chunk_{chunk_name}(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);"
            )
        elif i == num_chunks - 1:
            # Last chunk => concurrent queue -> vector
            lines.append(
                f"void chunk_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);"
            )
        else:
            # Intermediate chunk => concurrent queue -> concurrent queue
            lines.append(
                f"void chunk_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);"
            )

    lines.append("")
    lines.append(
        "void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);"
    )
    lines.append("")
    lines.append(f"}}  // namespace {sub_schedule_id}")
    return "\n".join(lines)


def generate_schedule_source(schedule_obj: dict) -> str:
    """
    Generate the .cpp definitions for each chunk, plus the run_pipeline function,
    using std::vector for the first/last stage I/O and moodycamel::ConcurrentQueue
    with sentinel-based termination in between.
    """
    schedule_id = schedule_obj["schedule_id"]
    device_id = schedule_obj["device_id"]
    chunks = schedule_obj["chunks"]

    # Derive sub-schedule name
    if schedule_id.startswith(device_id + "_"):
        sub_schedule_id = schedule_id[len(device_id) + 1:]
    else:
        sub_schedule_id = schedule_id

    lines = []
    lines.append(f"namespace {sub_schedule_id} {{")
    lines.append("")

    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        hw_str = chunk["hardware"].lower()
        threads = chunk["threads"]
        stages = chunk["stages"]
        start_stage = stages[0]
        end_stage = stages[-1]

        # Which run_* call do we need?
        if hw_str == "gpu":
            run_call = f"run_gpu_stages<{start_stage}, {end_stage}>(task);"
        else:
            pt_enum = HARDWARE_MAP.get(hw_str, "ProcessorType::kUnknown")
            run_call = f"run_cpu_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>(task);"

        # Build function bodies
        if num_chunks == 1:
            # Single-chunk: vector->vector
            lines.append(
                f"void chunk_{chunk_name}(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks) {{"
            )
            lines.append("  for (auto& task : in_tasks) {")
            lines.append("    if (task.is_sentinel()) {")
            lines.append("      out_tasks.push_back(task);")
            lines.append("      continue;")
            lines.append("    }")
            lines.append("")
            lines.append(f"    {run_call}")
            lines.append("")
            lines.append("    out_tasks.push_back(task);")
            lines.append("  }")
            lines.append("}")
            lines.append("")
        elif i == 0:
            # First chunk: vector->moodycamel queue
            lines.append(
                f"void chunk_{chunk_name}(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {{"
            )
            lines.append("  for (auto& task : in_tasks) {")
            lines.append("    if (task.is_sentinel()) {")
            lines.append("      out_q.enqueue(task);")
            lines.append("      continue;")
            lines.append("    }")
            lines.append("")
            lines.append(f"    {run_call}")
            lines.append("")
            lines.append("    out_q.enqueue(task);")
            lines.append("  }")
            lines.append("}")
            lines.append("")
        elif i == num_chunks - 1:
            # Last chunk: moodycamel queue->vector
            lines.append(
                f"void chunk_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {{"
            )
            lines.append("  while (true) {")
            lines.append("    Task task;")
            lines.append("    if (in_q.try_dequeue(task)) {")
            lines.append("      if (task.is_sentinel()) {")
            lines.append("        out_tasks.push_back(task);")
            lines.append("        break;")
            lines.append("      }")
            lines.append("")
            lines.append(f"      {run_call}")
            lines.append("")
            lines.append("      out_tasks.push_back(task);")
            lines.append("    } else {")
            lines.append("      std::this_thread::yield();")
            lines.append("    }")
            lines.append("  }")
            lines.append("}")
            lines.append("")
        else:
            # Intermediate chunk: moodycamel queue->moodycamel queue
            lines.append(
                f"void chunk_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {{"
            )
            lines.append("  while (true) {")
            lines.append("    Task task;")
            lines.append("    if (in_q.try_dequeue(task)) {")
            lines.append("      if (task.is_sentinel()) {")
            lines.append("        out_q.enqueue(task);")
            lines.append("        break;")
            lines.append("      }")
            lines.append("")
            lines.append(f"      {run_call}")
            lines.append("")
            lines.append("      out_q.enqueue(task);")
            lines.append("    } else {")
            lines.append("      std::this_thread::yield();")
            lines.append("    }")
            lines.append("  }")
            lines.append("}")
            lines.append("")

    # run_pipeline
    lines.append(
        "void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {"
    )
    for i in range(num_chunks - 1):
        lines.append(f"  moodycamel::ConcurrentQueue<Task> q_{i}{i+1};")
    lines.append("")

    # Spawn threads
    thread_names = []
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        tvar = f"t_{chunk_name}"
        thread_names.append(tvar)

        if num_chunks == 1:
            # single chunk
            lines.append(
                f"  std::thread {tvar}([&]() {{ chunk_{chunk_name}(tasks, out_tasks); }});"
            )
        elif i == 0:
            lines.append(
                f"  std::thread {tvar}([&]() {{ chunk_{chunk_name}(tasks, q_0{1}); }});"
            )
        elif i == num_chunks - 1:
            lines.append(
                f"  std::thread {tvar}([&]() {{ chunk_{chunk_name}(q_{i-1}{i}, out_tasks); }});"
            )
        else:
            lines.append(
                f"  std::thread {tvar}([&]() {{ chunk_{chunk_name}(q_{i-1}{i}, q_{i}{i+1}); }});"
            )

    lines.append("")
    # Join threads
    for tvar in thread_names:
        lines.append(f"  {tvar}.join();")

    lines.append("}")
    lines.append("")
    lines.append(f"}}  // namespace {sub_schedule_id}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", required=True, help="Directory with .json schedule files"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Directory to output aggregated .hpp/.cpp"
    )
    parser.add_argument(
        "--application",
        required=True,
        choices=["Tree", "CifarDense", "CifarSparse"],
        help="Which application to generate code for.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        print(f"Error: input directory {in_dir} not found.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    schedules_by_device = defaultdict(list)

    # 1) Read all schedules and group by device_id, filtering by --application
    for json_file in in_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        if "schedule" not in data:
            print(f"Skipping {json_file}: no 'schedule' key found.")
            continue

        schedule_obj = data["schedule"]
        schedule_id = schedule_obj["schedule_id"]

        # Parse application from schedule_id
        # e.g. "3A021JEHN02756_CifarDense_schedule_001" => "CifarDense" is at index 1
        app_parts = schedule_id.split("_")
        if len(app_parts) < 2:
            print(f"Skipping {json_file}: invalid schedule_id format")
            continue

        app_name = app_parts[1]
        if app_name != args.application:
            continue

        device_id = schedule_obj["device_id"]
        schedules_by_device[device_id].append(schedule_obj)

    # 2) For each device, generate .hpp/.cpp with all schedules
    for device_id, schedule_list in schedules_by_device.items():
        hpp_name = f"device_{device_id}.hpp"
        cpp_name = f"device_{device_id}.cpp"

        # Build header file
        hpp_lines = []
        hpp_lines.append(f"// Auto-generated aggregated header for device: {device_id}")
        hpp_lines.append(f"// Contains all '{args.application}' schedules for device_{device_id}")
        hpp_lines.append("")
        hpp_lines.append("#pragma once")
        hpp_lines.append("")
        hpp_lines.append("#include <vector>")
        hpp_lines.append("#include <thread>")
        hpp_lines.append('#include "../task.hpp"')
        hpp_lines.append("#include <concurrentqueue.h>")
        hpp_lines.append("")
        hpp_lines.append(f"namespace device_{device_id} {{")
        hpp_lines.append("")
        for sch in schedule_list:
            hpp_lines.append(generate_schedule_header(sch))
        hpp_lines.append(f"}}  // namespace device_{device_id}")
        hpp_content = "\n".join(hpp_lines)

        # Build source file
        cpp_lines = []
        cpp_lines.append(f"// Auto-generated aggregated source for device: {device_id}")
        cpp_lines.append(f"// Contains all '{args.application}' schedules for device_{device_id}")
        cpp_lines.append(f'#include "{hpp_name}"')
        cpp_lines.append("")
        cpp_lines.append('#include "../run_stages.hpp"')
        cpp_lines.append("")
        cpp_lines.append(f"namespace device_{device_id} {{")
        cpp_lines.append("")
        for sch in schedule_list:
            cpp_lines.append(generate_schedule_source(sch))
        cpp_lines.append(f"}}  // namespace device_{device_id}")
        cpp_content = "\n".join(cpp_lines)

        # Write out the final files
        out_hpp_path = out_dir / hpp_name
        out_cpp_path = out_dir / cpp_name

        with open(out_hpp_path, "w") as hf:
            hf.write(hpp_content)

        with open(out_cpp_path, "w") as cf:
            cf.write(cpp_content)

        print(f"[+] Wrote {hpp_name} and {cpp_name} for device {device_id} (app={args.application})")


if __name__ == "__main__":
    main()
