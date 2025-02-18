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
    schedule_id = schedule_obj["schedule_id"]
    device_id = schedule_obj["device_id"]
    chunks = schedule_obj["chunks"]

    # Derive sub-schedule name
    if schedule_id.startswith(device_id + "_"):
        sub_schedule_id = schedule_id[len(device_id) + 1 :]
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
        if i == 0:
            lines.append(
                f"void stage_group_{schedule_id}_{chunk_name}(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);"
            )
        elif i == num_chunks - 1:
            lines.append(
                f"void stage_group_{schedule_id}_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);"
            )
        else:
            lines.append(
                f"void stage_group_{schedule_id}_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);"
            )

    lines.append("")
    lines.append(
        "void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);"
    )
    lines.append("")
    lines.append(f"}}  // namespace {sub_schedule_id}")
    return "\n".join(lines)


def generate_schedule_source(schedule_obj: dict) -> str:
    schedule_id = schedule_obj["schedule_id"]
    device_id = schedule_obj["device_id"]
    chunks = schedule_obj["chunks"]

    if schedule_id.startswith(device_id + "_"):
        sub_schedule_id = schedule_id[len(device_id) + 1 :]
    else:
        sub_schedule_id = schedule_id

    lines = []
    lines.append(f"namespace {sub_schedule_id} {{")
    lines.append("")

    lines.append("static std::atomic<int> tasks_in_flight{0};")
    lines.append("static std::atomic<bool> done(false);")
    lines.append("")

    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        hw_str = chunk["hardware"].lower()
        threads = chunk["threads"]
        stages = chunk["stages"]
        start_stage = stages[0]
        end_stage = stages[-1]

        if hw_str == "gpu":
            run_call = f"run_gpu_stages<{start_stage}, {end_stage}>(task.app_data);"
        else:
            pt_enum = HARDWARE_MAP.get(hw_str, "ProcessorType::kUnknown")
            run_call = f"run_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>(task.app_data);"

        if i == 0:
            # First chunk: increment tasks_in_flight
            lines.append(
                f"void stage_group_{schedule_id}_{chunk_name}(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {{"
            )
            lines.append("  for (auto& task : in_tasks) {")
            lines.append(f"    {run_call}")
            lines.append("    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);")
            lines.append("    out_q.enqueue(task);")
            lines.append("  }")
            lines.append("}")
            lines.append("")
        elif i == num_chunks - 1:
            # Last chunk: decrement tasks_in_flight. If 0 => done=true
            lines.append(
                f"void stage_group_{schedule_id}_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {{"
            )
            lines.append("  while (!done.load(std::memory_order_acquire)) {")
            lines.append("    Task task;")
            lines.append("    if (in_q.try_dequeue(task)) {")
            lines.append(f"      {run_call}")
            lines.append("      out_tasks.push_back(task);")
            lines.append(
                "      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;"
            )
            lines.append(
                "      if (r == 0) done.store(true, std::memory_order_release);"
            )
            lines.append("    } else {")
            lines.append("      std::this_thread::yield();")
            lines.append("    }")
            lines.append("  }")
            lines.append("  while (true) {")
            lines.append("    Task task;")
            lines.append("    if (!in_q.try_dequeue(task)) break;")
            lines.append(f"    {run_call}")
            lines.append("    out_tasks.push_back(task);")
            lines.append(
                "    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;"
            )
            lines.append("    if (r == 0) done.store(true, std::memory_order_release);")
            lines.append("  }")
            lines.append("}")
            lines.append("")
        else:
            # Intermediate chunks
            lines.append(
                f"void stage_group_{schedule_id}_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {{"
            )
            lines.append("  while (!done.load(std::memory_order_acquire)) {")
            lines.append("    Task task;")
            lines.append("    if (in_q.try_dequeue(task)) {")
            lines.append(f"      {run_call}")
            lines.append("      out_q.enqueue(task);")
            lines.append("    } else {")
            lines.append("      std::this_thread::yield();")
            lines.append("    }")
            lines.append("  }")
            lines.append("  while (true) {")
            lines.append("    Task task;")
            lines.append("    if (!in_q.try_dequeue(task)) break;")
            lines.append(f"    {run_call}")
            lines.append("    out_q.enqueue(task);")
            lines.append("  }")
            lines.append("}")
            lines.append("")

    # run_pipeline for this schedule
    lines.append(
        "void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {"
    )
    for i in range(num_chunks - 1):
        lines.append(f"  moodycamel::ConcurrentQueue<Task> q_{i}{i+1};")
    lines.append("")
    lines.append("  tasks_in_flight.store(0, std::memory_order_relaxed);")
    lines.append("  done.store(false, std::memory_order_relaxed);")
    lines.append("")

    thread_names = []
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        tvar = f"t_{chunk_name}"
        thread_names.append(tvar)
        if i == 0:
            if num_chunks > 1:
                lines.append(
                    f"  std::thread {tvar}(stage_group_{schedule_id}_{chunk_name}, std::ref(tasks), std::ref(q_0{1}));"
                )
            else:
                lines.append(
                    f"  std::thread {tvar}(stage_group_{schedule_id}_{chunk_name}, std::ref(tasks), std::ref(out_tasks));"
                )
        elif i == num_chunks - 1:
            lines.append(
                f"  std::thread {tvar}(stage_group_{schedule_id}_{chunk_name}, std::ref(q_{i-1}{i}), std::ref(out_tasks));"
            )
        else:
            lines.append(
                f"  std::thread {tvar}(stage_group_{schedule_id}_{chunk_name}, std::ref(q_{i-1}{i}), std::ref(q_{i}{i+1}));"
            )

    lines.append("")
    for tvar in thread_names:
        lines.append(f"  {tvar}.join();")
    lines.append("}")
    lines.append("")
    lines.append(f"}}  // end namespace {sub_schedule_id}")
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
    # Add --application argument
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

    # 1) Read all schedules and group them by device_id,
    # but only if schedule["application"] matches --application
    for json_file in in_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        if "schedule" not in data:
            print(f"Skipping {json_file}: no 'schedule' key found.")
            continue

        schedule_obj = data["schedule"]
        schedule_id = schedule_obj["schedule_id"]

        # Parse application from schedule_id (e.g. "3A021JEHN02756_CifarDense_schedule_001")
        app_parts = schedule_id.split("_")
        if len(app_parts) < 2:
            print(f"Skipping {json_file}: invalid schedule_id format")
            continue

        app_name = app_parts[1]
        if app_name != args.application:
            continue

        device_id = schedule_obj["device_id"]
        schedules_by_device[device_id].append(schedule_obj)

    # 2) For each device, generate a single .hpp / .cpp containing all schedules
    for device_id, schedule_list in schedules_by_device.items():
        # It's optional to sort if you want a consistent order:
        # schedule_list.sort(key=lambda s: s["schedule_id"])

        hpp_name = f"device_{device_id}.hpp"
        cpp_name = f"device_{device_id}.cpp"

        # Build up the HPP file
        hpp_lines = []
        hpp_lines.append(f"// Auto-generated aggregated header for device: {device_id}")
        hpp_lines.append(
            f"// Contains all '{args.application}' schedules for device_{device_id}"
        )
        hpp_lines.append("")
        hpp_lines.append("#pragma once")
        hpp_lines.append("")
        hpp_lines.append("#include <vector>")
        hpp_lines.append('#include "../task.hpp"')
        hpp_lines.append("#include <concurrentqueue.h>")
        hpp_lines.append("")
        hpp_lines.append(f"namespace device_{device_id} {{")
        hpp_lines.append("")
        for sch in schedule_list:
            hpp_lines.append(generate_schedule_header(sch))
        hpp_lines.append(f"}}  // namespace device_{device_id}")
        hpp_content = "\n".join(hpp_lines)

        # Build up the CPP file
        cpp_lines = []
        cpp_lines.append(f"// Auto-generated aggregated source for device: {device_id}")
        cpp_lines.append(
            f"// Contains all '{args.application}' schedules for device_{device_id}"
        )
        cpp_lines.append(f'#include "{hpp_name}"')
        cpp_lines.append("")
        cpp_lines.append("#include <atomic>")
        cpp_lines.append("#include <thread>")
        cpp_lines.append('#include "../run_stages.hpp"')
        cpp_lines.append("")
        cpp_lines.append(f"namespace device_{device_id} {{")
        cpp_lines.append("")
        for sch in schedule_list:
            cpp_lines.append(generate_schedule_source(sch))
        cpp_lines.append(f"}}  // namespace device_{device_id}")
        cpp_content = "\n".join(cpp_lines)

        # Write out the final aggregated files
        out_hpp_path = out_dir / hpp_name
        out_cpp_path = out_dir / cpp_name

        with open(out_hpp_path, "w") as hf:
            hf.write(hpp_content)

        with open(out_cpp_path, "w") as cf:
            cf.write(cpp_content)

        print(
            f"[+] Wrote {hpp_name} and {cpp_name} for device {device_id} (app={args.application})"
        )


if __name__ == "__main__":
    main()
