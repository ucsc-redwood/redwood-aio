import json
import argparse
import sqlite3
import os
import sys

from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class Schedule:
    chunks: List[Tuple[int, int]]
    pu_types: List[str]
    pu_threads: List[int]
    chunk_times: List[float]
    max_chunk_time: float


DB_PATH = "data/benchmark_results.db"


def load_configs(
    device_key: str,
    app_key: str,
    hardware_path: str = "data/hardware_config.json",
    application_path: str = "data/application_config.json",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load and return hardware and application configurations."""
    with open(hardware_path, "r") as f:
        hardware_data = json.load(f)

    with open(application_path, "r") as f:
        application_data = json.load(f)

    device_info = hardware_data[device_key]
    app_info = application_data[app_key]

    return hardware_data, application_data, device_info, app_info


def generate_schedules_with_chunks(
    device_info: Dict[str, Any],
    app_info: Dict[str, Any],
) -> List[Schedule]:
    """Generate all possible schedules with contiguous chunks."""
    num_stages = app_info["num_stages"]

    # Get available PU types and their core counts
    pu_configs = [
        (core_type, count)
        for core_type, count in device_info["pinnable_cores"].items()
        if count > 0
    ]

    # ----------------------------
    # 1) Partition stages into contiguous chunks
    # ----------------------------
    # We'll generate all partitions of [1..num_stages] into between 1 and len(pu_configs) chunks.
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

    # We'll get all partitions with up to len(pu_configs) chunks
    all_partitions = generate_partitions(num_stages, max_chunks=len(pu_configs))

    # ----------------------------
    # 2) Assign each chunk a unique PU type
    # ----------------------------
    # For each partition with K chunks, we must choose exactly K distinct PUs
    # out of the available PUs, in order.

    all_schedules = []
    for partition in all_partitions:
        k = len(partition)
        if k > len(pu_configs):
            continue

        # Generate all permutations of PU configs of length k
        for pu_perm in permutations(pu_configs, k):
            # Initialize lists for Schedule
            chunks = []
            pu_types = []
            pu_threads = []

            for chunk_info, (pu_type, num_threads) in zip(partition, pu_perm):
                chunks.append(chunk_info)
                pu_types.append(pu_type)
                # For GPU, set num_threads to 0 as it's not applicable
                threads = 0 if pu_type == "gpu" else num_threads
                pu_threads.append(threads)

            # Initialize with empty times - will be filled by timing function
            chunk_times = [0.0] * len(chunks)
            all_schedules.append(
                Schedule(
                    chunks=chunks,
                    pu_types=pu_types,
                    pu_threads=pu_threads,
                    chunk_times=chunk_times,
                    max_chunk_time=0.0,
                )
            )

    return all_schedules


def show_schedule_timing(
    schedule: Schedule,
    device_info: Dict[str, Any],
    cursor: sqlite3.Cursor,
    device_key: str,
    app_key: str,
) -> float:
    """Show timing for a schedule and return the max chunk time."""
    chunk_times = []

    for chunk_idx in range(len(schedule.chunks)):
        start_stage, end_stage = schedule.chunks[chunk_idx]
        pu_type = schedule.pu_types[chunk_idx]
        num_threads = schedule.pu_threads[chunk_idx]

        # Determine the backend and core_type
        if pu_type == "gpu":
            backend = "VK"
            core_type = None
            db_num_threads = None  # GPU doesn't use threads
        else:
            backend = "OMP"
            core_type = pu_type
            db_num_threads = num_threads

        # Sum the times for each stage in [start_stage..end_stage]
        total_time_ms = 0.0
        for stage_id in range(start_stage, end_stage + 1):
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

            if db_num_threads is not None:
                query += " AND num_threads = ?"
                params.append(db_num_threads)
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

        chunk_times.append(total_time_ms)
        schedule.chunk_times[chunk_idx] = total_time_ms

    if chunk_times:
        max_chunk_time = max(chunk_times)
        schedule.max_chunk_time = max_chunk_time
        return max_chunk_time
    else:
        schedule.max_chunk_time = 0.0
        return 0.0


def evaluate_and_sort_schedules(
    schedules: List[Schedule],
    device_info: Dict[str, Any],
    cursor: sqlite3.Cursor,
    device_key: str,
    app_key: str,
) -> List[Schedule]:
    """
    Evaluate each schedule's maximum chunk time and sort them by performance.
    Returns a list of schedules sorted by max_chunk_time.
    """
    # Evaluate each schedule
    for schedule in schedules:
        show_schedule_timing(schedule, device_info, cursor, device_key, app_key)

    # Sort by max_chunk_time
    return sorted(schedules, key=lambda x: x.max_chunk_time)


def remove_duplicate_schedules(schedules: List[Schedule]) -> List[Schedule]:
    """Remove duplicate schedules from a list of schedules."""
    # Convert each schedule to a tuple for hashing
    seen = set()
    unique_schedules = []

    for schedule in schedules:
        # Create a hashable representation of the schedule
        schedule_key = (
            tuple(schedule.chunks),
            tuple(schedule.pu_types),
            tuple(schedule.pu_threads),
        )

        if schedule_key not in seen:
            seen.add(schedule_key)
            unique_schedules.append(schedule)

    return unique_schedules


def query_baseline(
    device_key: str,
    app_key: str,
    conn: sqlite3.Connection,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Query for baseline records (stage=0). Return (num_threads, best_time) with the lowest time.
    """
    query = """
    SELECT num_threads, time_ms FROM benchmark_result
    WHERE machine_name = ?
      AND application = ?
      AND stage = 0
      AND backend = 'OMP'
      AND core_type IS NULL
    """
    cur = conn.cursor()
    cur.execute(query, (device_key, app_key))
    rows = cur.fetchall()
    if not rows:
        return None, None
    best = min(rows, key=lambda r: float(r[1]))
    return best[0], float(best[1])


def main():
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

    # Load configurations once
    hardware_data, application_data, device_info, app_info = load_configs(
        args.machine_name, args.app
    )

    # Connect to database once
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Generate schedules using the loaded configs
        schedules = generate_schedules_with_chunks(device_info, app_info)
        print(f"Number of valid schedules: {len(schedules)}")

        # Remove duplicates
        schedules = remove_duplicate_schedules(schedules)
        print(f"Number of unique schedules: {len(schedules)}")

        # Evaluate and sort all schedules
        print("\nEvaluating all schedules...")
        schedules = evaluate_and_sort_schedules(
            schedules, device_info, cursor, args.machine_name, args.app
        )

        # Query baseline
        baseline_threads, baseline_time = query_baseline(
            args.machine_name, args.app, conn
        )

        # filter out schedules whose max chunk time is greater than baseline time
        schedules = [
            schedule
            for schedule in schedules
            if schedule.max_chunk_time <= baseline_time
        ]

        print(
            f"Number of schedules after filtering <= {baseline_time:.2f}ms: {len(schedules)}"
        )

        # Write all schedules to a log file, now sorted by performance
        with open("schedules.log", "w") as f:
            for idx, schedule in enumerate(schedules, 1):
                f.write(
                    f"Schedule {idx} (Max chunk time: {schedule.max_chunk_time:.2f} ms):\n"
                )
                for i in range(len(schedule.chunks)):
                    start, end = schedule.chunks[i]
                    f.write(
                        f"  Stages {start}-{end}: {schedule.pu_types[i]} "
                        f"(threads={schedule.pu_threads[i]}, "
                        f"time={schedule.chunk_times[i]:.2f} ms)\n"
                    )
                f.write("\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
