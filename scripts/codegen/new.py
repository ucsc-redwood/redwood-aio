#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path

# If your application names map to distinct C++ namespaces or data types,
# customize this dictionary:
APP_NAMESPACE_MAP = {
    "CifarDense": "cifar_dense",
    "CifarSparse": "cifar_sparse",
    "Tree": "tree",
}


def hardware_to_cpp(hardware, stages):
    """
    Return a string like:
        'vulkan::run_gpu_stages<3,7>'
    or
        'omp::run_multiple_stages<3,7, ProcessorType::kLittleCore, 4>'
    depending on the JSON chunk data.
    """
    stage_min = min(stages)
    stage_max = max(stages)

    if hardware == "gpu_vulkan":
        # GPU chunk
        return f"vulkan::run_gpu_stages<{stage_min}, {stage_max}>"
    else:
        # CPU chunk via OMP
        hw_map = {
            "little": "ProcessorType::kLittleCore",
            "medium": "ProcessorType::kMediumCore",
            "big":    "ProcessorType::kBigCore",
        }
        if hardware not in hw_map:
            raise ValueError(f"Unknown hardware type: {hardware}")
        proc_type = hw_map[hardware]

        # Example: run_multiple_stages<start,end,ProcessorType::kX,threads>
        return f"omp::run_multiple_stages<{stage_min}, {stage_max}, {proc_type}, {{threads}}>"


def generate_benchmark_code(schedule_json, app_namespace):
    """
    Given the parsed JSON for one schedule (dictionary) and the application
    namespace (string), emit a C++ function as a list of lines.
    """
    sched = schedule_json["schedule"]
    device_id = sched["device_id"]
    schedule_id = sched["schedule_id"]
    # e.g. "3A021JEHN02756_CifarDense_schedule_001"
    func_name = f"BM_schedule_{schedule_id}"

    # We'll assume it's something like "cifar_dense::AppData" for CifarDense, etc.
    app_data_type = f"{app_namespace}::AppData"

    lines = []
    lines.append(f"static void {func_name}(benchmark::State &state) {{")
    lines.append("    constexpr size_t num_tasks = 20;")
    lines.append("")
    lines.append(f"    auto mr = {app_namespace}::vulkan::Singleton::getInstance().get_mr();")
    lines.append("")
    lines.append("    // Preallocate data for all tasks")
    lines.append(f"    auto preallocated_data = init_appdata<{app_data_type}>(mr, num_tasks);")
    lines.append("")
    lines.append("    // Track individual task times")
    lines.append("    std::vector<double> task_times;")
    lines.append("    task_times.reserve(num_tasks);")
    lines.append("")
    lines.append("    for (auto _ : state) {")
    lines.append("        state.PauseTiming();")
    lines.append("        moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);")
    lines.append("")
    lines.append("        auto start_time = std::chrono::high_resolution_clock::now();")
    lines.append("        state.ResumeTiming();")
    lines.append("")
    lines.append("        // ---------------------------------------------------------------------")
    lines.append("        // Automatically generated from schedule JSON")
    lines.append("")

    chunks = sched["chunks"]
    n = len(chunks)
    # We'll define one queue per link in pipeline; i.e. for 4 chunks => 3 internal queues
    for i in range(n - 1):
        lines.append(f"        moodycamel::ConcurrentQueue<Task*> q_{i}_{i+1};")
    lines.append("")

    # Create each thread that processes one chunk
    for i, chunk in enumerate(chunks):
        if i == 0:
            inQ = "q_input"
        else:
            inQ = f"q_{i-1}_{i}"

        if i == n - 1:
            outQ = "nullptr"
        else:
            outQ = f"&q_{i}_{i+1}"

        hw = chunk["hardware"]
        threads = chunk["threads"]
        stages = chunk["stages"]

        if hw == "gpu_vulkan":
            stage_min = min(stages)
            stage_max = max(stages)
            run_fun = f"vulkan::run_gpu_stages<{stage_min}, {stage_max}>"
            thread_line = (
                f"std::thread t{i+1}([&]() {{ "
                f"chunk<Task, {app_data_type}>({inQ}, {outQ}, {run_fun}); }});"
            )
        else:
            # e.g. omp::run_multiple_stages<start,end,ProcessorType::kX,threads>
            stage_min = min(stages)
            stage_max = max(stages)

            proc_map = {
                "little": "ProcessorType::kLittleCore",
                "medium": "ProcessorType::kMediumCore",
                "big": "ProcessorType::kBigCore",
            }
            if hw not in proc_map:
                raise ValueError(f"Unknown hardware type: {hw}")

            proc_type = proc_map[hw]
            run_fun = f"omp::run_multiple_stages<{stage_min}, {stage_max}, {proc_type}, {threads}>"
            thread_line = (
                f"std::thread t{i+1}([&]() {{ "
                f"chunk<Task, {app_data_type}>({inQ}, {outQ}, {run_fun}); }});"
            )

        lines.append(f"        {thread_line}")

    # Join all threads
    lines.append("")
    for i in range(n):
        lines.append(f"        t{i+1}.join();")

    lines.append("")
    lines.append("        // ---------------------------------------------------------------------")
    lines.append("")
    lines.append("        state.PauseTiming();")
    lines.append("        auto end_time = std::chrono::high_resolution_clock::now();")
    lines.append(
        "        double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();"
    )
    lines.append("        task_times.push_back(elapsed / num_tasks);")
    lines.append("        state.ResumeTiming();")
    lines.append("    }  // for (auto _ : state)")
    lines.append("")
    lines.append("    // Calculate and report the actual average time per task")
    lines.append(
        "    double avg_task_time = "
        "std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();"
    )
    lines.append('    state.counters["avg_time_per_task"] = avg_task_time;')
    lines.append("}")
    lines.append("")

    return lines, func_name


def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_code.py <root_dir> <application> <output_file.hpp>")
        print("Example:")
        print("  python generate_code.py data/schedule_files/ CifarDense generated_code.hpp")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    application = sys.argv[2]
    output_path = Path(sys.argv[3])

    if application in APP_NAMESPACE_MAP:
        app_namespace = APP_NAMESPACE_MAP[application]
    else:
        app_namespace = "cifar_dense"  # fallback

    lines_out = []
    # Basic includes
    lines_out.append("#pragma once")
    lines_out.append("#include <benchmark/benchmark.h>")
    lines_out.append("#include <thread>")
    lines_out.append("#include <chrono>")
    lines_out.append("#include <numeric>")
    lines_out.append('#include "task.hpp"')
    lines_out.append('#include "run_stages.hpp"')
    lines_out.append('#include "../templates.hpp"')
    lines_out.append('#include "../templates_vk.hpp"')
    lines_out.append("")
    lines_out.append("// Automatically generated benchmark code")
    lines_out.append("")

    # We'll define a small struct in a common namespace, so each device can build a table of it:
    lines_out.append("namespace generated_schedules {")
    lines_out.append("using bench_func_t = void(*)(benchmark::State&);")
    lines_out.append("struct ScheduleRecord {")
    lines_out.append("    const char* name;")
    lines_out.append("    bench_func_t func;")
    lines_out.append("};")
    lines_out.append("} // namespace generated_schedules\n")

    # Now, for each device, we'll collect all schedule (id, function) pairs
    # and then produce a static array for that device only.
    for device_dir in sorted(root_dir.iterdir()):
        if not device_dir.is_dir():
            continue

        app_dir = device_dir / application
        if not app_dir.is_dir():
            # This device doesn't have the app subdir
            continue

        schedule_files = sorted(app_dir.glob("schedule_*.json"))
        if not schedule_files:
            continue

        # Start the device namespace
        device_ns = f"device_{device_dir.name}"
        lines_out.append(f"namespace {device_ns} {{")

        # We'll accumulate pairs for this device's table
        schedule_records = []

        for json_path in schedule_files:
            with open(json_path, "r") as f:
                schedule_json = json.load(f)

            sched = schedule_json["schedule"]
            schedule_id = sched["schedule_id"]
            chunks = sched["chunks"]

            # Skip unknown hardware
            valid_hardware = {"gpu_vulkan", "little", "medium", "big"}
            skip_this = False
            for ch in chunks:
                if ch["hardware"] not in valid_hardware:
                    print(f"Skipping {schedule_id} for device '{device_dir.name}' due to unknown hardware {ch['hardware']}")
                    skip_this = True
                    break
            if skip_this:
                continue

            # Generate the function code
            func_lines, func_name = generate_benchmark_code(schedule_json, app_namespace)
            lines_out.extend(func_lines)

            # We'll store it in the device's schedule table
            # function pointer is &BM_schedule_<schedule_id> (in this namespace)
            schedule_records.append((schedule_id, func_name))

        # Now emit the static table for this device
        lines_out.append("")
        lines_out.append("// Table of schedules for this device:")
        lines_out.append("static generated_schedules::ScheduleRecord schedule_table[] = {")
        for (sid, fname) in schedule_records:
            lines_out.append(f'    {{"{sid}", &{fname}}},')
        lines_out.append("};")
        lines_out.append(
            "static const size_t schedule_count = sizeof(schedule_table)/sizeof(schedule_table[0]);"
        )

        lines_out.append(f"}} // namespace {device_ns}\n")

    # Write out to file
    with open(output_path, "w") as outf:
        outf.write("\n".join(lines_out))
        outf.write("\n")

    print(f"Done. Generated code has been written to {output_path}")


if __name__ == "__main__":
    main()
