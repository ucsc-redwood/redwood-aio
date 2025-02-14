import sqlite3
import argparse
import random
from typing import Dict, List, Tuple, Optional, Any, Set
import json
from pathlib import Path

DB_NAME: str = "benchmark_results.db"

# Devices dictionary.
# For each machine we list available (usable) processing unit types and the number of threads to use.
devices: Dict[str, Dict[str, int]] = {
    "3A021JEHN02756": {"little": 4, "medium": 2, "big": 2, "gpu": 1},
    # Additional devices can be added here.
}


def get_stage_time(
    conn: sqlite3.Connection,
    machine_name: str,
    application: str,
    stage: int,
    hardware: str,
    threads: int,
) -> float:
    """
    Query the database for the execution time (time_ms) of a given stage.

    For CPU-based scheduling:
      - backend is assumed to be 'OMP'
      - core_type is set to the hardware (e.g. "little", "big", "medium")
      - num_threads is the provided threads value.

    For GPU:
      - if hardware.lower() == "gpu", backend is set to 'VK'
      - core_type and num_threads are assumed to be NULL.

    Returns the execution time in ms (or 9999.9 if not found).
    """
    if hardware.lower() == "gpu":
        backend = "VK"
        core_type = None
        num_threads = None
    else:
        backend = "OMP"
        core_type = hardware
        num_threads = threads

    query = """
    SELECT time_ms FROM benchmark_result
    WHERE machine_name = ?
      AND application = ?
      AND backend = ?
      AND stage = ?
    """
    params = [machine_name, application, backend, stage]

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

    cur = conn.cursor()
    cur.execute(query, params)
    row = cur.fetchone()
    if row:
        return float(row[0])
    else:
        # Return a high value if no data is found.
        return 9999.9


def estimate_schedule_time(
    conn: sqlite3.Connection,
    machine_name: str,
    application: str,
    schedule: List[Dict[str, Any]],
) -> float:
    """
    For the given execution schedule (a list of chunk dictionaries),
    query the DB for each stage's time. For each chunk, sum the stage times,
    and then compute the overall pipeline time.

    Each chunk is updated with:
      - a 'stage_times' list (each entry is a tuple: (stage, time_ms))
      - a 'chunk_time' field (sum of stage times)

    Returns the total pipeline time.
    """
    total_time = 0.0
    for chunk in schedule:
        chunk_time = 0.0
        chunk["stage_times"] = []
        for stage in chunk["stages"]:
            stage_time = get_stage_time(
                conn,
                machine_name=machine_name,
                application=application,
                stage=stage,
                hardware=chunk["hardware"],
                threads=chunk["threads"],
            )
            chunk["stage_times"].append((stage, stage_time))
            chunk_time += stage_time
        chunk["chunk_time"] = chunk_time
        total_time += chunk_time
    return total_time


def print_schedule_report(schedule: List[Dict[str, Any]], total_time: float) -> None:
    """
    Print a report for one schedule:
      - Print timing for each stage in each chunk.
      - Print the total time per chunk.
      - Print the overall pipeline time.
      - Print the maximum (slowest) chunk time.
    """
    max_chunk_time = max(chunk["chunk_time"] for chunk in schedule)
    print("Schedule Report:")
    for chunk in schedule:
        print(
            f"  Chunk {chunk['chunk_id']}: Hardware = {chunk['hardware']}, Threads = {chunk['threads']}"
        )
        for stage, t in chunk["stage_times"]:
            print(f"    Stage {stage}: {t} ms")
        print(f"    Chunk Total Time: {chunk['chunk_time']} ms")
    print(f"Pipeline Total Time: {total_time} ms")
    print(f"Max (Slowest) Chunk Time: {max_chunk_time} ms")
    print("-" * 50)


def random_fixed_partition(stages: List[int], num_chunks: int) -> List[List[int]]:
    """
    Partition the list 'stages' (must have length >= num_chunks) into exactly 'num_chunks'
    contiguous groups by randomly choosing num_chunks-1 breakpoints.
    """
    n = len(stages)
    if num_chunks > n:
        raise ValueError(
            "Not enough stages to partition into the requested number of chunks."
        )
    # Choose num_chunks - 1 unique breakpoints from 1 to n-1
    breakpoints = sorted(random.sample(range(1, n), num_chunks - 1))
    partition = []
    prev = 0
    for bp in breakpoints:
        partition.append(stages[prev:bp])
        prev = bp
    partition.append(stages[prev:])
    return partition


def random_fixed_schedule(
    stages: List[int], hw_specs: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Generate a random schedule subject to the constraint that each hardware type appears exactly once.

    - The schedule will have exactly num_chunks = len(hw_specs).
    - Stages is a list of stage numbers (for example, stages 1 to 9).
    - hw_specs is a dictionary mapping hardware type -> threads.

    Returns a schedule: a list of chunks, where each chunk is a dictionary with:
        'chunk_id', 'stages', 'hardware', and 'threads'.
    """
    num_chunks = len(hw_specs)
    partition = random_fixed_partition(stages, num_chunks)
    # Randomly choose a permutation of the available hardware types.
    hw_types = list(hw_specs.keys())
    chosen_perm = random.sample(hw_types, len(hw_types))
    schedule = []
    for i, chunk in enumerate(partition, start=1):
        schedule.append(
            {
                "chunk_id": i,
                "stages": chunk,
                "hardware": chosen_perm[i - 1],
                "threads": hw_specs[chosen_perm[i - 1]],
            }
        )
    return schedule


def query_baseline(
    conn: sqlite3.Connection, machine_name: str, application: str
) -> Tuple[Optional[int], Optional[float]]:
    """
    Query the database for baseline records (stage 0) and return the best (lowest time)
    along with its number of threads and time.

    Assumes baseline records use backend 'OMP', stage=0, and core_type IS NULL.
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
    cur.execute(query, (machine_name, application))
    rows = cur.fetchall()
    if not rows:
        print("No baseline records found.")
        return None, None
    best = min(rows, key=lambda r: float(r[1]))
    return best[0], float(best[1])


def schedule_to_json(
    machine_name: str, schedule: List[Dict[str, Any]], schedule_id: str
) -> Dict[str, Any]:
    """Convert a schedule to a JSON-compatible dictionary format."""
    return {
        "schedule_id": schedule_id,
        "device_id": machine_name,
        "chunks": [
            {
                "name": f"chunk{chunk['chunk_id']}",
                "hardware": chunk["hardware"],
                "threads": chunk["threads"],
                "stages": chunk["stages"],
            }
            for chunk in schedule
        ],
    }


def save_schedules_to_json(
    machine_name: str,
    valid_schedules: List[Tuple[List[Dict[str, Any]], float, float]],
    application: str,
) -> None:
    """Save all valid schedules to a JSON file."""
    schedules_json = [
        {
            "schedule": schedule_to_json(
                machine_name, sched, f"{machine_name}_{application}_schedule_{idx:03d}"
            ),
            "total_time": total_time,
            "max_chunk_time": max_chunk_time,
        }
        for idx, (sched, total_time, max_chunk_time) in enumerate(valid_schedules, 1)
    ]

    output_dir = Path("schedules")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{machine_name}_{application}_schedules.json"

    with open(output_file, "w") as f:
        json.dump(schedules_json, f, indent=2)
    print(f"\nSaved {len(valid_schedules)} schedules to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine_name", required=True, help="Machine name (e.g., 3A021JEHN02756)"
    )
    parser.add_argument(
        "--application", required=True, help="Application name (e.g., CifarDense)"
    )
    parser.add_argument("--db_name", default=DB_NAME, help="SQLite database file")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of valid schedules to sample",
    )
    args = parser.parse_args()

    machine_name = args.machine_name
    application = args.application
    num_samples_required = args.num_samples

    # Connect to the database.
    conn = sqlite3.connect(args.db_name)

    # Query the baseline (stage 0) records and find the best baseline.
    baseline_threads, best_baseline_time = query_baseline(
        conn, machine_name, application
    )
    if best_baseline_time is None:
        print("Cannot determine baseline. Exiting.")
        return
    print(f"Best baseline for machine {machine_name} and application {application}:")
    print(f"  OMP with {baseline_threads} threads, time = {best_baseline_time} ms.\n")

    # For scheduling, we use pipeline stages 1 to 9 (stage 0 is baseline).
    stages = list(range(1, 10))

    # Get hardware specs for the chosen machine.
    if machine_name not in devices:
        print(
            f"Device info for machine {machine_name} not found in devices dictionary."
        )
        return
    hw_specs = devices[machine_name]
    # Number of chunks will be exactly the number of hardware types available.
    num_chunks = len(hw_specs)
    print(f"Generating schedules with {num_chunks} chunks (one per hardware type).")

    # Randomly sample execution schedules that meet the criterion:
    # maximum chunk time (slowest chunk) is faster than the best baseline time.
    valid_schedules = []
    max_iterations = 10000
    iterations = 0

    while len(valid_schedules) < num_samples_required and iterations < max_iterations:
        iterations += 1
        sched = random_fixed_schedule(stages, hw_specs)
        total_time = estimate_schedule_time(conn, machine_name, application, sched)
        max_chunk_time = max(chunk["chunk_time"] for chunk in sched)
        if max_chunk_time < best_baseline_time:
            valid_schedules.append((sched, total_time, max_chunk_time))

    print(
        f"\nGenerated {len(valid_schedules)} valid schedules (after {iterations} iterations).\n"
    )

    # Remove duplicates by converting schedules to a hashable format and using a set
    unique_schedules = []
    seen = set()

    for sched, total_time, max_chunk_time in valid_schedules:
        # Create a tuple representation of the schedule that can be hashed
        schedule_key = tuple(
            (
                chunk["chunk_id"],
                tuple(chunk["stages"]),
                chunk["hardware"],
                chunk["threads"],
            )
            for chunk in sched
        )

        if schedule_key not in seen:
            seen.add(schedule_key)
            unique_schedules.append((sched, total_time, max_chunk_time))

    valid_schedules = unique_schedules

    # Sort valid schedules by max_chunk_time
    valid_schedules.sort(key=lambda x: x[2])  # x[2] is max_chunk_time

    # reverse the list
    valid_schedules.reverse()

    print(f"Found {len(valid_schedules)} unique schedules.\n")

    # Save schedules to JSON file
    save_schedules_to_json(machine_name, valid_schedules, application)

    # Report the schedules.
    for idx, (sched, total_time, max_chunk_time) in enumerate(valid_schedules, start=1):
        schedule_id = f"{machine_name}_{application}_schedule_{idx:03d}"
        print(f"--- Valid Execution Schedule #{idx} (ID: {schedule_id}) ---")
        print_schedule_report(sched, total_time)
        # Print JSON representation
        json_repr = schedule_to_json(machine_name, sched, schedule_id)
        print("JSON representation:")
        print(json.dumps(json_repr, indent=2))
        print("-" * 50)

    conn.close()


if __name__ == "__main__":
    main()
