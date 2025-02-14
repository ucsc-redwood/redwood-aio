#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path

# This dictionary maps the hardware string from your JSON
# to the corresponding ProcessorType enum in C++.
HARDWARE_MAP = {
    "little": "ProcessorType::kLittleCore",
    "medium": "ProcessorType::kMediumCore",
    "big": "ProcessorType::kBigCore",
    "gpu": "ProcessorType::kGPU",
}


def generate_cpp_code(schedule_data: dict) -> str:
    """
    Generate the C++ code for a single schedule.

    schedule_data is expected to have:
      - "schedule": dictionary containing:
        - "schedule_id": unique string ID
        - "device_id": string with device name
        - "chunks": list of chunks, each chunk has:
             "name": chunk name (e.g. "chunk1")
             "hardware": "little"/"medium"/"big"/"gpu"
             "threads": integer
             "stages": list of integers
    """
    # Get the schedule dictionary
    schedule = schedule_data.get("schedule")
    if not schedule:
        raise ValueError("Missing 'schedule' field in input JSON")

    schedule_id = schedule["schedule_id"]
    device_id = schedule["device_id"]
    chunks = schedule["chunks"]

    # We create a unique namespace from the schedule_id by replacing non-alnum with underscores
    safe_ns = "".join(c if c.isalnum() else "_" for c in schedule_id)

    # Start building the code
    # Includes and forward declarations
    code_lines = []
    code_lines.append(f"// Auto-generated code for schedule: {schedule_id}\n")
    code_lines.append(f"// Device ID: {device_id}\n")
    code_lines.append("#include <vector>")
    code_lines.append("#include <thread>")
    code_lines.append("#include <atomic>")
    code_lines.append("#include <iostream>")
    code_lines.append('#include "moodycamel/ConcurrentQueue.h"')
    code_lines.append('#include "run_stages.hpp"')
    code_lines.append('#include "Task.hpp"')
    # If you have a registry, uncomment:
    # code_lines.append('#include "pipeline_registry.hpp"')
    code_lines.append("\n")

    # Open namespace
    code_lines.append(f"namespace schedule_{safe_ns} {{\n")

    # We'll have a local static atomic for controlling the pipeline loop exit
    code_lines.append("static std::atomic<bool> done(false);\n")

    # Generate each chunk function
    # We need to distinguish:
    #  - first chunk: (std::vector<Task>& in, moodycamel::ConcurrentQueue<Task>& out)
    #  - last chunk:  (moodycamel::ConcurrentQueue<Task>& in, std::vector<Task>& out)
    #  - intermediate chunk: (moodycamel::ConcurrentQueue<Task>& in, moodycamel::ConcurrentQueue<Task>& out)
    #
    # We'll do this by index:
    #   i=0 => first chunk
    #   i=last => final chunk
    #   else => intermediate

    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]  # e.g. "chunk1"
        hardware_str = chunk["hardware"]
        threads = chunk["threads"]
        stages = chunk["stages"]

        # Decide CPU or GPU
        if hardware_str.lower() == "gpu":
            # e.g. run_gpu_stages<start, end>(task.app_data);
            is_gpu = True
        else:
            is_gpu = False

        # For convenience, we define the range as from the first stage to the last stage in this chunk
        start_stage = stages[0]
        end_stage = stages[-1]

        # We'll build the line that calls either run_stages or run_gpu_stages
        if is_gpu:
            run_call = f"run_gpu_stages<{start_stage}, {end_stage}>(task.app_data);"
        else:
            pt_enum = HARDWARE_MAP.get(hardware_str.lower(), "ProcessorType::kUnknown")
            run_call = f"run_stages<{start_stage}, {end_stage}, {pt_enum}, {threads}>(task.app_data);"

        # Now define the function signature based on position
        if i == 0:
            # first chunk
            code_lines.append(
                f"void stage_group_{chunk_name}(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q)"
            )
            code_lines.append("{")
            code_lines.append("    for (auto& task : in_tasks) {")
            code_lines.append(f"        {run_call}")
            code_lines.append("        out_q.enqueue(task);")
            code_lines.append("    }")
            code_lines.append("    done.store(true, std::memory_order_release);")
            code_lines.append("}\n")
        elif i == len(chunks) - 1:
            # last chunk
            code_lines.append(
                f"void stage_group_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks)"
            )
            code_lines.append("{")
            code_lines.append("    while (!done.load(std::memory_order_acquire)) {")
            code_lines.append("        Task task;")
            code_lines.append("        if (in_q.try_dequeue(task)) {")
            code_lines.append(f"            {run_call}")
            code_lines.append("            out_tasks.push_back(task);")
            code_lines.append("        } else {")
            code_lines.append("            std::this_thread::yield();")
            code_lines.append("        }")
            code_lines.append("    }")
            code_lines.append("    // Drain any remaining tasks if needed:")
            code_lines.append("    while (true) {")
            code_lines.append("        Task task;")
            code_lines.append("        if (!in_q.try_dequeue(task)) break;")
            code_lines.append(f"        {run_call}")
            code_lines.append("        out_tasks.push_back(task);")
            code_lines.append("    }")
            code_lines.append("}\n")
        else:
            # intermediate
            code_lines.append(
                f"void stage_group_{chunk_name}(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q)"
            )
            code_lines.append("{")
            code_lines.append("    while (!done.load(std::memory_order_acquire)) {")
            code_lines.append("        Task task;")
            code_lines.append("        if (in_q.try_dequeue(task)) {")
            code_lines.append(f"            {run_call}")
            code_lines.append("            out_q.enqueue(task);")
            code_lines.append("        } else {")
            code_lines.append("            std::this_thread::yield();")
            code_lines.append("        }")
            code_lines.append("    }")
            code_lines.append("    // Drain any remaining tasks if needed:")
            code_lines.append("    while (true) {")
            code_lines.append("        Task task;")
            code_lines.append("        if (!in_q.try_dequeue(task)) break;")
            code_lines.append(f"        {run_call}")
            code_lines.append("        out_q.enqueue(task);")
            code_lines.append("    }")
            code_lines.append("}\n")

    # Now define the run_pipeline() function that sets up queues + threads
    code_lines.append(
        "void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks)"
    )
    code_lines.append("{")
    # We'll create one queue per boundary. If we have N chunks, we have N-1 "intermediate" queues
    # We'll store them in a small list here:
    num_chunks = len(chunks)
    for i in range(num_chunks - 1):
        code_lines.append(f"    moodycamel::ConcurrentQueue<Task> q_{i}{i+1};")
    code_lines.append("\n    // Create threads")

    thread_names = []
    for i, chunk in enumerate(chunks):
        chunk_name = chunk["name"]
        # We'll name the thread e.g. t_chunk1
        tvar = f"t_{chunk_name}"
        thread_names.append(tvar)
        if i == 0:
            # first chunk
            code_lines.append(
                f"    std::thread {tvar}(stage_group_{chunk_name}, std::ref(tasks), std::ref(q_0{1 if num_chunks>1 else ''}));"
            )
        elif i == num_chunks - 1:
            # last chunk
            code_lines.append(
                f"    std::thread {tvar}(stage_group_{chunk_name}, std::ref(q_{i-1}{i}), std::ref(out_tasks));"
            )
        else:
            code_lines.append(
                f"    std::thread {tvar}(stage_group_{chunk_name}, std::ref(q_{i-1}{i}), std::ref(q_{i}{i+1}));"
            )

    # Join threads
    code_lines.append("\n    // Join threads")
    for tvar in thread_names:
        code_lines.append(f"    {tvar}.join();")

    code_lines.append("}\n")

    # Optionally register with a pipeline registry
    # code_lines.append(f"static void register_pipeline_{safe_ns}() __attribute__((constructor));")
    # code_lines.append(f"static void register_pipeline_{safe_ns}() {{")
    # code_lines.append(f"    PipelineRegistry::instance().register_pipeline(\"{schedule_id}\", run_pipeline);")
    # code_lines.append("}\n")

    # Close namespace
    code_lines.append(f"}} // end namespace schedule_{safe_ns}\n")

    # Return the concatenated string
    return "\n".join(code_lines)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_schedule.json> <output.cpp>")
        sys.exit(1)

    in_json = Path(sys.argv[1])
    out_cpp = Path(sys.argv[2])

    if not in_json.is_file():
        print(f"Error: cannot find input file {in_json}")
        sys.exit(1)

    with open(in_json, "r") as f:
        schedule_data = json.load(f)

    # Generate code
    cpp_code = generate_cpp_code(schedule_data)

    # Write code to output file
    with open(out_cpp, "w") as fout:
        fout.write(cpp_code)

    print(f"Generated {out_cpp} from schedule {schedule_data['schedule']['schedule_id']}.")


if __name__ == "__main__":
    main()
