import json
from textwrap import indent


def hardware_to_processor_type(hardware):
    """
    Map hardware labels from JSON to ProcessorType enum values.
    Adjust as appropriate for your actual naming conventions.
    """
    hardware = hardware.lower()
    if hardware == "little":
        return "ProcessorType::kLittleCore"
    elif hardware == "medium":
        return "ProcessorType::kMediumCore"
    elif hardware == "big":
        return "ProcessorType::kBigCore"
    else:
        # If it's not one of the recognized CPU hardware names,
        # we'll treat it as a GPU path (e.g., "gpu_vulkan").
        return None


def generate_queue_definitions(num_chunks):
    """
    Generate definitions for moodycamel::ConcurrentQueue<Task*>.
    If there are N chunks, we need N-1 intermediate queues:
      q_0_1, q_1_2, ..., q_(N-2)_(N-1).
    Returns a list of C++ lines.
    """
    lines = []
    # For chunk indices i in [0..N-1], we produce the concurrency queues for pairs:
    #   q_0_1, q_1_2, ..., q_(N-2)_(N-1)
    for i in range(num_chunks - 1):
        lines.append(f"moodycamel::ConcurrentQueue<Task*> q_{i}_{i+1};")
    return lines


def generate_thread_call(i, chunk_info, total_chunks):
    """
    Generate the std::thread line for the i-th chunk in the schedule.
    chunk_info is a dictionary containing:
       {
         "name": "chunk1",
         "hardware": "little",
         "threads": 4,
         "stages": [1],
         "time": ...
       }

    total_chunks is the total number of chunks in the schedule.

    Returns a string with the C++ code for that thread.
    """
    hardware = chunk_info["hardware"]
    threads = chunk_info["threads"]
    stages = chunk_info["stages"]
    first_stage = stages[0]
    last_stage = stages[-1]

    # Determine the input and output queues for this chunk
    #  - The first chunk uses q_input as the input queue
    #  - The last chunk uses nullptr as the output queue
    #  - In between, they use q_{i-1}_{i} / q_{i}_{i+1}
    if i == 0:
        input_queue = "q_input"
    else:
        input_queue = f"q_{i-1}_{i}"

    if i == total_chunks - 1:
        output_queue = "nullptr"
    else:
        output_queue = f"&q_{i}_{i+1}"

    # Now generate the function call part for GPU or CPU
    processor_type = hardware_to_processor_type(hardware)

    if processor_type is None:
        # GPU path, e.g., "gpu_vulkan"
        # e.g., vulkan::run_gpu_stages<3,7>
        run_function = f"vulkan::run_gpu_stages<{first_stage}, {last_stage}>"
    else:
        # CPU path
        # e.g., omp::run_multiple_stages<3, 7, ProcessorType::kLittleCore, 4>
        run_function = (
            f"omp::run_multiple_stages<{first_stage}, {last_stage}, "
            f"{processor_type}, {threads}>"
        )

    thread_name = f"t{i+1}"
    call = (
        f"std::thread {thread_name}([&]() {{\n"
        f"  chunk<Task, cifar_dense::AppData>(\n"
        f"      {input_queue},\n"
        f"      {output_queue},\n"
        f"      {run_function});\n"
        f"}});"
    )
    return call


def generate_cpp_function_from_schedule(schedule_json):
    """
    Given the schedule JSON (parsed as a Python dict), generate
    the C++ code that uses std::threads, moodycamel queues, etc.
    Returns a string containing the C++ code snippet.
    """

    chunks = schedule_json["schedule"]["chunks"]
    num_chunks = len(chunks)

    # 1) Generate queue definitions
    queue_definitions = generate_queue_definitions(num_chunks)

    # 2) Generate thread calls
    thread_calls = []
    for i, chunk in enumerate(chunks):
        thread_calls.append(generate_thread_call(i, chunk, num_chunks))

    # 3) Generate the join calls
    #    We'll name each thread t1, t2, t3, ...
    join_calls = [f"t{i+1}.join();" for i in range(num_chunks)]

    # Put it all together
    # You can further wrap it in any additional code (like the "// ---------------------------------------------------------------------")
    # blocks, etc.) as needed.
    # We'll indent the lines to make it look nice.
    code_lines = []
    code_lines.append("// Automatically generated from schedule JSON\n")
    code_lines.append("// Queue definitions:")
    code_lines.extend(queue_definitions)
    code_lines.append("")
    code_lines.append("// Thread calls:")
    code_lines.extend(thread_calls)
    code_lines.append("")
    code_lines.append("// Thread joins:")
    code_lines.extend(join_calls)
    code_lines.append("")

    # Join them all into a single string. You can choose your own indentation style.
    return "\n".join(code_lines)


def generate_benchmark_function(schedule_json):
    benchmark_name = f"BM_schedule_{schedule_json['schedule']['schedule_id']}"

    template_code = """
static void <<<BENCHMARK_NAME>>>(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // ===== GENERATED CODE START =====
    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

    """

    cpp_code = generate_cpp_function_from_schedule(schedule_json)

    # Replace the template code with the generated code
    template_code = template_code.replace(
        "// ===== GENERATED CODE START =====", cpp_code
    )

    # Replace the <<<BENCHMARK_NAME>>> with the actual benchmark name
    template_code = template_code.replace("<<<BENCHMARK_NAME>>>", benchmark_name)

    return template_code


def read_schedules_from_dir(directory_path):
    """
    Read all JSON files from the given directory and return a list of schedule data.

    Args:
        directory_path (str): Path to directory containing JSON schedule files

    Returns:
        list: List of schedule dictionaries
    """
    import os
    import json

    schedules = []

    # Walk through all files in directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            try:
                # Read and parse JSON file
                with open(file_path, "r") as f:
                    schedule_data = json.load(f)

                # Validate that this is a schedule JSON
                if (
                    "schedule" in schedule_data
                    and "chunks" in schedule_data["schedule"]
                    and "max_chunk_time" in schedule_data
                ):
                    schedules.append(schedule_data)

            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON file {filename}")
            except Exception as e:
                print(f"Warning: Error reading file {filename}: {str(e)}")

    return schedules


def main():
    """
    Example usage of the generator.
    In practice, you might read the JSON from a file, e.g.:
        with open("some_schedule.json") as f:
            schedule_data = json.load(f)
    """
    # For demonstration, we'll just embed the schedule here:
    schedule_data = {
        "schedule": {
            "schedule_id": "3A021JEHN02756_CifarDense_schedule_001",
            "device_id": "3A021JEHN02756",
            "application": "CifarDense",
            "chunks": [
                {
                    "name": "chunk1",
                    "hardware": "little",
                    "threads": 4,
                    "stages": [1],
                    "time": 6.494148231768887,
                },
                {
                    "name": "chunk2",
                    "hardware": "medium",
                    "threads": 2,
                    "stages": [2],
                    "time": 0.3203897834324584,
                },
                {
                    "name": "chunk3",
                    "hardware": "gpu_vulkan",
                    "threads": 1,
                    "stages": [3, 4, 5, 6, 7],
                    "time": 27.287561781360886,
                },
                {
                    "name": "chunk4",
                    "hardware": "big",
                    "threads": 2,
                    "stages": [8, 9],
                    "time": 0.09971189033337916,
                },
            ],
        },
        "max_chunk_time": 27.287561781360886,
        "cpu_baseline_time": 252.87687740031592,
        "gpu_baseline_time": 37.74098821796361,
        "cpu_speedup": 9.267111493011686,
        "gpu_speedup": 1.3830839310730603,
    }

    generated_code = generate_benchmark_function(schedule_data)
    print(generated_code)


def generate_all_benchmarks_top_n(schedule_dir, num_schedules):
    schedules = read_schedules_from_dir(schedule_dir)
    # take only first num_schedules
    schedules = schedules[:num_schedules]

    print(
        "// ============================================================================="
    )
    print("// AUTOMATICALLY GENERATED BENCHMARK CODE")
    print(
        "// =============================================================================\n"
    )

    for i, schedule in enumerate(schedules, 1):
        print(
            f"// -----------------------------------------------------------------------------"
        )
        print(f"// Schedule {i:03d}: {schedule['schedule']['schedule_id']}")
        print(f"// Device: {schedule['schedule']['device_id']}")
        print(f"// Application: {schedule['schedule']['application']}")
        print(f"// Chunks: {len(schedule['schedule']['chunks'])}")
        print(
            f"// -----------------------------------------------------------------------------\n"
        )

        generated_code = generate_benchmark_function(schedule)
        print(generated_code)
        print("\n")  # Add extra newline between benchmarks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate benchmark code from schedule files"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing schedule files",
    )
    parser.add_argument(
        "--device-id", type=str, required=True, help="Device ID (e.g. 3A021JEHN02756)"
    )
    parser.add_argument(
        "--application",
        type=str,
        required=True,
        help="Application name (e.g. CifarDense)",
    )
    parser.add_argument(
        "--num-schedules",
        type=int,
        default=10,
        help="Number of schedules to generate (default: 10)",
    )

    args = parser.parse_args()

    schedule_dir = f"{args.root_dir}/{args.device_id}/{args.application}"
    generate_all_benchmarks_top_n(schedule_dir, args.num_schedules)
