import json
from itertools import product, permutations
import argparse
from typing import List, Tuple
import sqlite3
import os
import sys

# Define a type for a schedule entry: (core_type, num_threads)
Schedule = List[Tuple[str, int]]

DB_PATH = "data/benchmark_results.db"


def generate_schedules_with_chunks(
    device_key: str,
    app_key: str,
    hardware_path: str = "data/hardware_config.json",
    application_path: str = "data/application_config.json",
) -> List[Schedule]:
    """
    Generate all possible schedules subject to:
      1) Stages are grouped into contiguous chunks.
      2) Each chunk is assigned exactly one PU type.
      3) No PU type is used more than once across chunks in a schedule.
    """

    # ----------------------------
    # Load JSON files
    # ----------------------------
    with open(hardware_path, "r") as f:
        hardware_data = json.load(f)

    with open(application_path, "r") as f:
        application_data = json.load(f)

    # Get device & application info
    device_info = hardware_data[device_key]
    app_info = application_data[app_key]

    # Number of stages in the application
    num_stages = app_info["num_stages"]

    # Identify the PU types that are actually available (> 0 cores)
    all_pu_types = [
        core for core, count in device_info["pinnable_cores"].items() if count > 0
    ]

    # ----------------------------
    # 1) Partition stages into contiguous chunks
    # ----------------------------
    # We'll generate all partitions of [1..num_stages] into between 1 and len(all_pu_types) chunks.
    # Each partition is a list of (start_stage, end_stage) for each chunk.

    def generate_partitions(n_stages, max_chunks):
        """
        Generate all possible ways to partition the range [1..n_stages]
        into up to max_chunks contiguous segments.
        For example, if n_stages=3, possible partitions (K=1..3):
            K=1: [(1,3)]
            K=2: [(1,1), (2,3)], [(1,2), (3,3)]
            K=3: [(1,1), (2,2), (3,3)]
        """
        results = []

        def backtrack(start, chunks_left, current_partition):
            # If we've used up all stages, record the partition
            if start > n_stages:
                results.append(current_partition[:])
                return

            # If we can't exceed the number of chunks we said we would use
            if chunks_left == 0:
                return

            # Try all possible chunk endings for the current chunk
            for end in range(start, n_stages + 1):
                # Add [start..end] as a chunk
                current_partition.append((start, end))
                # Recurse from end+1
                backtrack(end + 1, chunks_left - 1, current_partition)
                current_partition.pop()

        # We want from 1 chunk up to max_chunks
        for k in range(1, max_chunks + 1):
            backtrack(start=1, chunks_left=k, current_partition=[])

        return results

    # We'll get all partitions with up to len(all_pu_types) chunks
    all_partitions = generate_partitions(num_stages, max_chunks=len(all_pu_types))

    # ----------------------------
    # 2) Assign each chunk a unique PU type
    # ----------------------------
    # For each partition with K chunks, we must choose exactly K distinct PUs
    # out of the available PUs, in order.

    all_schedules = []
    for partition in all_partitions:
        k = len(partition)  # number of chunks
        # If we have fewer PU types than chunks, skip
        if k > len(all_pu_types):
            continue

        # Generate all permutations of PU types of length k
        for pu_perm in permutations(all_pu_types, k):
            # Combine chunk info with the chosen PU type
            # We can store as a list of tuples: [((start, end), pu_type), ...]
            schedule = []
            for chunk_info, pu_type in zip(partition, pu_perm):
                schedule.append((chunk_info, pu_type))
            all_schedules.append(schedule)

    return all_schedules


def show_schedule_timing(
    schedule: Schedule,
    device_key: str,  # '3A021JEHN02756', '9b034f1b', 'ce0717178d7758b00b7e', 'jetson'
    app_key: str,  # 'Tree', 'CifarDense', 'CifarSparse', etc.
    hardware_path: str = "data/hardware_config.json",
):
    """
    Given a schedule of chunks for (device_key, app_key), show:
      - The total time for each chunk
      - The maximum chunk time
    A schedule is a list of ((start_stage, end_stage), pu_type).
    """

    # ----------------------------
    # Load JSON files
    # ----------------------------
    with open(hardware_path, "r") as f:
        hardware_data = json.load(f)

    # Info for the selected device & application
    device_info = hardware_data[device_key]

    # For CPU PUs, we’ll pick the max threads from "pinnable_cores"
    # e.g., device_info["pinnable_cores"]["little"] => 4
    pinnable_cores = device_info["pinnable_cores"]

    # -------------------------------------------------------------------
    # 2) Connect to the DB
    # -------------------------------------------------------------------
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # -------------------------------------------------------------------
    # 3) For each chunk in the schedule, sum up stage times
    # -------------------------------------------------------------------
    chunk_times = []

    for chunk_index, ((start_stage, end_stage), pu_type) in enumerate(
        schedule, start=1
    ):
        # Determine the backend, core_type, and num_threads
        if pu_type == "gpu":
            # GPU => use Vulkan, with no distinct core_type or num_threads in the DB
            backend = "VK"
            core_type = None
            num_threads = None
        else:
            # CPU => use OMP, with core_type = pu_type, and the maximum threads
            backend = "OMP"
            core_type = pu_type
            num_threads = pinnable_cores[pu_type]

        # Sum the times for each stage in [start_stage..end_stage]
        total_time_ms = 0.0
        for stage_id in range(start_stage, end_stage + 1):
            # Skip stage = 0 records if they exist (you mentioned they are baseline)
            if stage_id == 0:
                continue

            # Prepare a parameterized query to find the record
            # matching device_key, app_key, backend, stage, etc.
            query = """
                SELECT time_ms 
                FROM benchmark_result
                WHERE machine_name = ?
                  AND application = ?
                  AND backend = ?
                  AND stage = ?
            """
            params = [device_key, app_key, backend, stage_id]

            # core_type and num_threads are either both set or both None (GPU)
            if core_type is not None:
                query += " AND core_type = ?"
                params.append(core_type)
            else:
                query += " AND core_type IS NULL"

            if num_threads is not None:
                query += " AND num_threads = ?"
                params.append(num_threads)
            else:
                query += " AND num_threads IS NULL"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            if not rows:
                print(
                    f"Warning: No record found for device={device_key}, "
                    f"app={app_key}, stage={stage_id}, PU={pu_type}"
                )
                # We'll add 0.0 for missing data. Or you could do something else.
                chunk_time_for_stage = 0.0
            else:
                # Typically there should be one record, but if there's more, pick the first
                chunk_time_for_stage = rows[0][0]

            total_time_ms += chunk_time_for_stage

        # Store this chunk’s total time
        chunk_times.append(total_time_ms)

        # Print the chunk info
        print(
            f"Chunk {chunk_index} => stages {start_stage}-{end_stage}, "
            f"PU={pu_type}, total_time={total_time_ms:.2f} ms"
        )

    # -------------------------------------------------------------------
    # 4) Print the maximum chunk time
    # -------------------------------------------------------------------
    if chunk_times:
        max_chunk_time = max(chunk_times)
        print(f"Max chunk time: {max_chunk_time:.2f} ms")
    else:
        print("No valid chunks to measure.")

    # -------------------------------------------------------------------
    # 5) Clean up
    # -------------------------------------------------------------------
    conn.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate possible scheduling combinations."
    )
    parser.add_argument(
        "--machine_name", required=True, help="Device ID from hardware config"
    )
    parser.add_argument(
        "--app", required=True, help="Application name from application config"
    )

    args = parser.parse_args()

    device_key = args.machine_name
    app_key = args.app

    schedules = generate_schedules_with_chunks(device_key, app_key)

    print(f"Number of valid schedules: {len(schedules)}")

    # Filter only the schedules with 4 chunks
    schedules = [schedule for schedule in schedules if len(schedule) == 4]

    print(f"Number of valid schedules with 4 chunks: {len(schedules)}")

    # show the timing of first 5 schedules
    for schedule in schedules[:5]:
        show_schedule_timing(schedule, device_key, app_key)

    # Write all schedules to a log file
    with open("schedules.log", "w") as f:
        for idx, schedule in enumerate(schedules, 1):
            f.write(f"Schedule {idx}:\n")
            for stage_idx, core_type in enumerate(schedule, start=1):
                f.write(f"  Stage {stage_idx}: {core_type}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
