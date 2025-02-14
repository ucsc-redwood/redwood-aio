import sqlite3
import argparse
import random
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import uuid

DB_NAME: str = "benchmark_results.db"

# Devices dictionary.
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
    return float(row[0]) if row else 9999.9


def estimate_schedule_time(
    conn: sqlite3.Connection,
    machine_name: str,
    application: str,
    schedule: List[Dict[str, Any]],
) -> float:
    """
    For the given execution schedule (a list of chunk dictionaries), query the DB for each stage's time.
    For each chunk, sum the stage times, and update the chunk with 'stage_times' and 'chunk_time'.
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
    Print a human-readable report for a schedule.
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
    Partition 'stages' (length >= num_chunks) into exactly 'num_chunks' contiguous groups
    by randomly choosing num_chunks-1 breakpoints.
    """
    n = len(stages)
    if num_chunks > n:
        raise ValueError(
            "Not enough stages to partition into the requested number of chunks."
        )
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
    Generate a schedule with one chunk per hardware type in hw_specs.
    Partition the stages contiguously and assign each partition to one hardware type
    in a random permutation.
    """
    num_chunks = len(hw_specs)
    partition = random_fixed_partition(stages, num_chunks)
    hw_types = list(hw_specs.keys())
    chosen_perm = random.sample(hw_types, len(hw_types))

    schedule = []
    for i, chunk_stages in enumerate(partition, start=1):
        hw = chosen_perm[i - 1]
        schedule.append(
            {
                "chunk_id": i,
                "stages": chunk_stages,
                "hardware": hw,
                "threads": hw_specs[hw],
            }
        )
    return schedule


def query_baseline(
    conn: sqlite3.Connection, machine_name: str, application: str
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
    cur.execute(query, (machine_name, application))
    rows = cur.fetchall()
    if not rows:
        return None, None
    best = min(rows, key=lambda r: float(r[1]))
    return best[0], float(best[1])


def schedule_to_json(
    machine_name: str, schedule: List[Dict[str, Any]], schedule_id: str
) -> Dict[str, Any]:
    """
    Convert an internal schedule structure to a JSON-friendly dictionary.
    """
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


def save_schedules_to_individual_json(
    machine_name: str,
    valid_schedules: List[Tuple[List[Dict[str, Any]], float, float]],
    application: str,
) -> None:
    """
    Write each valid schedule to its own JSON file in 'schedules/' directory.
    The filename will be something like: {schedule_id}_{uuid}.json
    """
    output_dir = Path("schedules")
    output_dir.mkdir(exist_ok=True)

    for idx, (sched, total_time, max_chunk_time) in enumerate(valid_schedules, start=1):
        schedule_id = f"{machine_name}_{application}_schedule_{idx:03d}"
        schedule_dict = schedule_to_json(machine_name, sched, schedule_id)

        # We'll embed total_time and max_chunk_time into the top-level dictionary:
        data_to_save = {
            "schedule": schedule_dict,
            "total_time": total_time,
            "max_chunk_time": max_chunk_time,
        }

        # Create a unique filename with a short UUID suffix
        short_uid = str(uuid.uuid4())[:8]
        file_name = f"{schedule_id}_{short_uid}.json"
        output_file = output_dir / file_name

        with open(output_file, "w") as f:
            json.dump(data_to_save, f, indent=2)

    print(
        f"\nSaved {len(valid_schedules)} schedules as individual JSON files in '{output_dir}/'.\n"
    )


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

    # Connect to the database
    conn = sqlite3.connect(args.db_name)

    # Query baseline (stage 0)
    baseline_threads, best_baseline_time = query_baseline(
        conn, machine_name, application
    )
    if best_baseline_time is None:
        print("No baseline found. Exiting.")
        return

    print(f"Best baseline for machine {machine_name} / {application}:")
    print(f"  OMP with {baseline_threads} threads, time = {best_baseline_time} ms.\n")

    # We'll schedule pipeline stages 1..9
    stages = list(range(1, 10))

    # Check device specs
    if machine_name not in devices:
        print(f"Device info for machine {machine_name} not found.")
        return
    hw_specs = devices[machine_name]
    num_chunks = len(hw_specs)
    print(f"Generating schedules with {num_chunks} chunks (one per hardware type).\n")

    valid_schedules = []
    max_iterations = 10000
    iterations = 0

    # Collect schedules whose max chunk time < baseline
    while len(valid_schedules) < num_samples_required and iterations < max_iterations:
        iterations += 1
        sched = random_fixed_schedule(stages, hw_specs)
        total_time = estimate_schedule_time(conn, machine_name, application, sched)
        max_chunk_time = max(chunk["chunk_time"] for chunk in sched)
        if max_chunk_time < best_baseline_time:
            valid_schedules.append((sched, total_time, max_chunk_time))

    print(
        f"Generated {len(valid_schedules)} valid schedules after {iterations} iterations.\n"
    )

    # Deduplicate schedules
    unique_schedules = []
    seen = set()
    for sched, total_time, max_chunk_time in valid_schedules:
        # Turn chunk info into a tuple for hashing
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
    # Sort by max_chunk_time (descending)
    valid_schedules.sort(key=lambda x: x[2], reverse=True)

    print(f"Found {len(valid_schedules)} unique schedules.\n")

    # Save each schedule to an individual JSON file
    save_schedules_to_individual_json(machine_name, valid_schedules, application)

    # Optionally also print them out
    for idx, (sched, total_time, max_chunk_time) in enumerate(valid_schedules, start=1):
        schedule_id = f"{machine_name}_{application}_schedule_{idx:03d}"
        print(f"--- Valid Execution Schedule #{idx} (ID: {schedule_id}) ---")
        print_schedule_report(sched, total_time)
        # Show JSON representation in console
        json_repr = schedule_to_json(machine_name, sched, schedule_id)
        print("JSON representation:")
        print(json.dumps(json_repr, indent=2))
        print("-" * 50)

    conn.close()


if __name__ == "__main__":
    main()
