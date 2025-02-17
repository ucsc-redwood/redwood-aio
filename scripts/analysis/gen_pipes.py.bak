#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

# Maps hardware string from JSON to the C++ ProcessorType enum.
HARDWARE_MAP = {
    "little":  "ProcessorType::kLittleCore",
    "medium":  "ProcessorType::kMediumCore",
    "big":     "ProcessorType::kBigCore",
    "gpu":     "ProcessorType::kGPU",
}


def generate_header(schedule_obj: dict) -> str:
    """
    Generate the .hpp content for a single schedule.
    The header declares:
      - Nested namespaces: device_{device_id}::{schedule_subname}
      - constexpr kScheduleId
      - function signatures for chunk stage groups
      - run_pipeline(...)
    """
    schedule_id = schedule_obj["schedule_id"]            # e.g. "3A021JEHN02756_CifarDense_schedule_001"
    device_id   = schedule_obj["device_id"]              # e.g. "3A021JEHN02756"
    chunks      = schedule_obj["chunks"]

    # We split schedule_id into two parts: the device prefix, and the remainder.
    # Example: "3A021JEHN02756_CifarDense_schedule_001" => device: "3A021JEHN02756", remainder: "CifarDense_schedule_001"
    # We'll only do this split if it starts with device_id.
    # Otherwise, we just use the full schedule_id for the second namespace.
    if schedule_id.startswith(device_id + "_"):
        sub_schedule_id = schedule_id[len(device_id)+1:]  # skip "3A021JEHN02756_"
    else:
        sub_schedule_id = schedule_id

    # Create namespace names
    device_ns = f"device_{device_id}"
    schedule_ns = sub_schedule_id

    lines = []
    lines.append(f"// Auto-generated code for schedule: {schedule_id}")
    lines.append(f"// Device ID: {device_id}")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <vector>")
    lines.append('#include "../task.hpp"')
    lines.append('#include <concurrentqueue.h>')
    lines.append("")
    lines.append(f"namespace {device_ns} {{")
    lines.append(f"namespace {schedule_ns} {{")
    lines.append("")
    lines.append(f'constexpr const char* kScheduleId = "{schedule_id}";')
    lines.append("")

    # For each chunk, declare the function signature
    # We'll follow your chunk naming convention: stage_group_<chunkName>
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]  # e.g. "chunk1"
        if i == 0:
            # first chunk
            lines.append(f"void stage_group_{chunk_name}(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);")
        elif i == num_chunks - 1:
            # last chunk
            lines.append(f"void stage_group_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);")
        else:
            # intermediate chunk
            lines.append(f"void stage_group_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);")

    lines.append("")
    lines.append("void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);")
    lines.append("")
    lines.append(f"}}  // namespace {schedule_ns}")
    lines.append(f"}}  // namespace {device_ns}")
    lines.append("")

    return "\n".join(lines)


def generate_source(schedule_obj: dict, header_filename: str) -> str:
    """
    Generate the .cpp content for a single schedule, implementing:
      - chunk functions
      - run_pipeline(...)
    The file #includes the generated header filename.
    """
    schedule_id = schedule_obj["schedule_id"]
    device_id   = schedule_obj["device_id"]
    chunks      = schedule_obj["chunks"]

    # Derive the same namespaces as in the header
    if schedule_id.startswith(device_id + "_"):
        sub_schedule_id = schedule_id[len(device_id)+1:]
    else:
        sub_schedule_id = schedule_id

    device_ns = f"device_{device_id}"
    schedule_ns = sub_schedule_id

    lines = []
    lines.append(f"// Auto-generated code for schedule: {schedule_id}")
    lines.append(f"// Device ID: {device_id}")
    lines.append("")
    # Include the generated header
    lines.append(f'#include "{header_filename}"')
    lines.append("")
    lines.append("#include <atomic>")
    lines.append("#include <thread>")
    lines.append('#include "../run_stages.hpp"')  # or your actual path
    lines.append("")
    lines.append(f"namespace {device_ns} {{")
    lines.append(f"namespace {schedule_ns} {{")
    lines.append("")
    lines.append("static std::atomic<bool> done(false);")
    lines.append("")

    # Generate chunk function definitions
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        hw_str     = chunk["hardware"].lower()
        threads    = chunk["threads"]
        stages     = chunk["stages"]
        start_stage = stages[0]
        end_stage   = stages[-1]

        # Decide run call
        if hw_str == "gpu":
            run_call = f"run_gpu_stages<{start_stage}, {end_stage}>(task.app_data);"
        else:
            pt_enum = HARDWARE_MAP.get(hw_str, "ProcessorType::kUnknown")
            run_call = f"run_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>(task.app_data);"

        if i == 0:
            # first chunk
            lines.append(f"void stage_group_{chunk_name}(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {{")
            lines.append("  for (auto& task : in_tasks) {")
            lines.append(f"    {run_call}")
            lines.append("    out_q.enqueue(task);")
            lines.append("  }")
            lines.append("  done.store(true, std::memory_order_release);")
            lines.append("}")
            lines.append("")
        elif i == num_chunks - 1:
            # last chunk
            lines.append(f"void stage_group_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {{")
            lines.append("  while (!done.load(std::memory_order_acquire)) {")
            lines.append("    Task task;")
            lines.append("    if (in_q.try_dequeue(task)) {")
            lines.append(f"      {run_call}")
            lines.append("      out_tasks.push_back(task);")
            lines.append("    } else {")
            lines.append("      std::this_thread::yield();")
            lines.append("    }")
            lines.append("  }")
            lines.append("  // Drain any remaining tasks if needed:")
            lines.append("  while (true) {")
            lines.append("    Task task;")
            lines.append("    if (!in_q.try_dequeue(task)) break;")
            lines.append(f"    {run_call}")
            lines.append("    out_tasks.push_back(task);")
            lines.append("  }")
            lines.append("}")
            lines.append("")
        else:
            # intermediate
            lines.append(f"void stage_group_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {{")
            lines.append("  while (!done.load(std::memory_order_acquire)) {")
            lines.append("    Task task;")
            lines.append("    if (in_q.try_dequeue(task)) {")
            lines.append(f"      {run_call}")
            lines.append("      out_q.enqueue(task);")
            lines.append("    } else {")
            lines.append("      std::this_thread::yield();")
            lines.append("    }")
            lines.append("  }")
            lines.append("  // Drain any remaining tasks if needed:")
            lines.append("  while (true) {")
            lines.append("    Task task;")
            lines.append("    if (!in_q.try_dequeue(task)) break;")
            lines.append(f"    {run_call}")
            lines.append("    out_q.enqueue(task);")
            lines.append("  }")
            lines.append("}")
            lines.append("")

    # run_pipeline
    lines.append("void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {")
    # We create concurrency queues: for N chunks, we have N-1 queues
    for i in range(num_chunks - 1):
        lines.append(f"  moodycamel::ConcurrentQueue<Task> q_{i}{i+1};")

    lines.append("")
    lines.append("  // Create threads")
    thread_names = []
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        tvar = f"t_{chunk_name}"
        thread_names.append(tvar)

        if i == 0:
            # first chunk -> tasks to q_01
            if num_chunks > 1:
                lines.append(f"  std::thread {tvar}(stage_group_{chunk_name}, std::ref(tasks), std::ref(q_0{1}));")
            else:
                # If there's only one chunk total, it goes directly tasks -> out_tasks
                lines.append(f"  std::thread {tvar}(stage_group_{chunk_name}, std::ref(tasks), std::ref(out_tasks));")
        elif i == num_chunks - 1:
            # last chunk -> q_{i-1}{i} to out_tasks
            lines.append(f"  std::thread {tvar}(stage_group_{chunk_name}, std::ref(q_{i-1}{i}), std::ref(out_tasks));")
        else:
            # intermediate chunk -> q_{i-1}{i} to q_{i}{i+1}
            lines.append(f"  std::thread {tvar}(stage_group_{chunk_name}, std::ref(q_{i-1}{i}), std::ref(q_{i}{i+1}));")

    lines.append("")
    lines.append("  // Join threads")
    for tvar in thread_names:
        lines.append(f"  {tvar}.join();")

    lines.append("}")
    lines.append("")

    lines.append(f"}}  // end namespace {schedule_ns}")
    lines.append(f"}}  // end namespace {device_ns}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="Directory with .json schedule files")
    parser.add_argument("--out_dir", required=True, help="Directory to output .hpp/.cpp files")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.is_dir():
        print(f"Error: input directory {in_dir} not found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Process all .json files in the input directory
    for json_file in in_dir.glob("*.json"):
        with open(json_file, "r") as f:
            schedule_data = json.load(f)

        # We expect a top-level structure with "schedule": {...}
        if "schedule" not in schedule_data:
            print(f"Skipping {json_file}: no 'schedule' key found.")
            continue
        schedule_obj = schedule_data["schedule"]

        # We'll generate .hpp + .cpp
        # We name them based on schedule_obj["schedule_id"]
        sid = schedule_obj["schedule_id"]  # e.g. "3A021JEHN02756_CifarDense_schedule_001"
        base_name = sid  # or a sanitized version

        hpp_name = f"{base_name}.hpp"
        cpp_name = f"{base_name}.cpp"

        # Generate content
        hpp_content = generate_header(schedule_obj)
        cpp_content = generate_source(schedule_obj, hpp_name)

        # Write them to disk
        with open(out_dir / hpp_name, "w") as hf:
            hf.write(hpp_content)
        with open(out_dir / cpp_name, "w") as cf:
            cf.write(cpp_content)

        print(f"Wrote {hpp_name} and {cpp_name} for schedule {sid}")


if __name__ == "__main__":
    main()
