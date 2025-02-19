import json
import argparse
import sqlite3
import os
import sys
from pathlib import Path

from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple, Dict, Any, Optional

HARDWARE_PATH = "data/hardware_config.json"
APPLICATION_PATH = "data/application_config.json"
DB_PATH = "data/benchmark_results.db"


@dataclass
class Schedule:
    chunks: List[Tuple[int, int]]
    pu_types: List[str]
    pu_threads: List[int]
    chunk_times: List[float]
    max_chunk_time: float


def load_configs(
    device_key: str,
    app_key: str,
    hardware_path: str = HARDWARE_PATH,
    application_path: str = APPLICATION_PATH,
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

    def generate_partitions(n_stages, max_chunks):
        """Generate all possible ways to partition [1..n_stages] into up to max_chunks contiguous segments."""
        results = []

        def backtrack(start, chunks_left, current_partition):
            if start > n_stages:
                results.append(current_partition[:])
                return
            if chunks_left == 0:
                return

            for end in range(start, n_stages + 1):
                current_partition.append((start, end))
                backtrack(end + 1, chunks_left - 1, current_partition)
                current_partition.pop()

        for k in range(1, max_chunks + 1):
            backtrack(start=1, chunks_left=k, current_partition=[])

        return results

    all_partitions = generate_partitions(num_stages, max_chunks=len(pu_configs))

    all_schedules = []
    for partition in all_partitions:
        k = len(partition)
        if k > len(pu_configs):
            continue

        # Generate all permutations of PU configs of length k
        for pu_perm in permutations(pu_configs, k):
            chunks = []
            pu_types = []
            pu_threads = []

            for chunk_info, (pu_type, num_threads) in zip(partition, pu_perm):
                chunks.append(chunk_info)
                pu_types.append(pu_type)
                threads = 0 if pu_type == "gpu" else num_threads
                pu_threads.append(threads)

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
    """Look up timings for each chunk using rows where aggregate_name='mean'."""
    chunk_times = []

    for chunk_idx in range(len(schedule.chunks)):
        start_stage, end_stage = schedule.chunks[chunk_idx]
        pu_type = schedule.pu_types[chunk_idx]
        num_threads = schedule.pu_threads[chunk_idx]

        # Determine the backend and core_type
        if pu_type == "gpu":
            backend = "VK"
            core_type = None
            db_num_threads = None
        else:
            backend = "OMP"
            core_type = pu_type
            db_num_threads = num_threads

        total_time_ms = 0.0
        for stage_id in range(start_stage, end_stage + 1):
            if stage_id == 0:
                continue  # Defensive check

            # Only look up rows with aggregate_name='mean'
            query = """
                SELECT real_time
                FROM benchmarks
                WHERE device = ?
                  AND application = ?
                  AND backend = ?
                  AND stage = ?
                  AND aggregate_name = 'mean'
            """
            params = [device_key, app_key, backend, stage_id]

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
            row = cursor.fetchone()
            if row and row[0] is not None:
                chunk_time_for_stage = float(row[0])
            else:
                print(
                    f"Warning: No 'mean' record found for device={device_key}, "
                    f"app={app_key}, stage={stage_id}, PU={pu_type}."
                )
                chunk_time_for_stage = 0.0

            total_time_ms += chunk_time_for_stage

        chunk_times.append(total_time_ms)
        schedule.chunk_times[chunk_idx] = total_time_ms

    schedule.max_chunk_time = max(chunk_times) if chunk_times else 0.0
    return schedule.max_chunk_time


def evaluate_and_sort_schedules(
    schedules: List[Schedule],
    device_info: Dict[str, Any],
    cursor: sqlite3.Cursor,
    device_key: str,
    app_key: str,
) -> List[Schedule]:
    for schedule in schedules:
        show_schedule_timing(schedule, device_info, cursor, device_key, app_key)
    return sorted(schedules, key=lambda x: x.max_chunk_time)


def remove_duplicate_schedules(schedules: List[Schedule]) -> List[Schedule]:
    seen = set()
    unique_schedules = []
    for schedule in schedules:
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
    Query for baseline (stage=0) using only aggregate_name='mean', then pick the best row.
    """
    query = """
        SELECT num_threads, real_time
        FROM benchmarks
        WHERE device = ?
          AND application = ?
          AND stage = 0
          AND backend = 'OMP'
          AND core_type IS NULL
          AND aggregate_name = 'mean'
        ORDER BY real_time ASC
        LIMIT 1
    """
    cur = conn.cursor()
    cur.execute(query, (device_key, app_key))
    row = cur.fetchone()
    if not row:
        return None, None
    return row[0], float(row[1])  # (num_threads, real_time)


def schedule_to_json(
    schedule: Schedule,
    schedule_id: str,
    device_id: str,
) -> dict:
    chunks_json = []
    for i, (start, end) in enumerate(schedule.chunks):
        stages = list(range(start, end + 1))
        chunk_json = {
            "name": f"chunk{i+1}",
            "hardware": schedule.pu_types[i],
            "threads": schedule.pu_threads[i],
            "stages": stages,
            "time": schedule.chunk_times[i],
        }
        chunks_json.append(chunk_json)

    return {
        "schedule": {
            "schedule_id": schedule_id,
            "device_id": device_id,
            "chunks": chunks_json,
        },
        "max_chunk_time": schedule.max_chunk_time,
    }


def write_schedules_to_json(
    schedules: List[Schedule],
    device_id: str,
    app_name: str,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, schedule in enumerate(schedules, 1):
        schedule_id = f"{device_id}_{app_name}_schedule_{idx:03d}"
        schedule_json = schedule_to_json(schedule, schedule_id, device_id)
        filename = f"{schedule_id}.json"
        file_path = output_path / filename
        with open(file_path, "w") as f:
            json.dump(schedule_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate possible scheduling combinations."
    )
    parser.add_argument(
        "--device", required=True, help="Device ID from hardware config"
    )
    parser.add_argument("--app", required=True, help="Application name from app config")
    parser.add_argument(
        "--output_dir",
        default="data/generated-schedules",
        help="Directory for output JSON files",
    )
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)

    # Load hardware and application configs
    hardware_data, application_data, device_info, app_info = load_configs(
        args.device, args.app
    )

    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1) Generate all possible schedules
        schedules = generate_schedules_with_chunks(device_info, app_info)
        print(f"Number of valid schedules: {len(schedules)}")

        # 2) Remove duplicates
        schedules = remove_duplicate_schedules(schedules)
        print(f"Number of unique schedules: {len(schedules)}")

        # 3) Evaluate and sort
        print("\nEvaluating all schedules (using aggregate_name='mean')...")
        schedules = evaluate_and_sort_schedules(
            schedules, device_info, cursor, args.device, args.app
        )

        # 4) Compare to baseline
        baseline_threads, baseline_time = query_baseline(args.device, args.app, conn)
        if baseline_time is not None:
            schedules = [s for s in schedules if s.max_chunk_time <= baseline_time]
            print(
                f"Number of schedules after filtering <= {baseline_time:.2f}ms: "
                f"{len(schedules)}"
            )
        else:
            print("No 'mean' baseline found; skipping baseline filter.")

        # 5) Take top 50
        schedules = schedules[:50]

        # 6) Write schedules out
        write_schedules_to_json(schedules, args.device, args.app, args.output_dir)
        print(f"Wrote {len(schedules)} schedules to {args.output_dir}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
