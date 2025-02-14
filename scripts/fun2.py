import sqlite3

DB_NAME = "benchmark_results.db"


def get_stage_time(conn, machine_name, application, stage, hardware, threads):
    """
    Query the database for the execution time (time_ms) of a given stage.

    For CPU-based scheduling:
      - backend is assumed to be 'OMP'
      - core_type is the hardware (e.g., "little", "big", etc.)
      - num_threads is the given threads value.

    For GPU:
      - hardware is "gpu" (case-insensitive), backend is set to 'VK'
      - core_type and num_threads are assumed to be NULL.

    Returns the time in ms (or 9999.9 if no matching record is found).
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
        # Return a high value to indicate missing data.
        return 9999.9


def estimate_schedule_time(conn, machine_name, application, schedule):
    """
    Given an execution schedule (a list of chunks), query the DB to get the time
    for each stage in each chunk. Compute each chunk's total time and the overall
    pipeline time. The schedule is updated to include per-stage timings and a
    'chunk_time' field.
    """
    total_time = 0.0
    for chunk in schedule:
        chunk_time = 0.0
        # Prepare to store per-stage timing details in the chunk.
        chunk["stage_times"] = []
        for stage in chunk["stages"]:
            # We skip stage 0 per the requirement; assume schedules already have stages >= 1.
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


def print_schedule_report(schedule, total_time):
    """
    Print a nicely formatted report for a given schedule.
    """
    print("Schedule Report:")
    for chunk in schedule:
        print(
            f"  Chunk {chunk['chunk_id']}: Hardware = {chunk['hardware']}, Threads = {chunk['threads']}"
        )
        for stage, stage_time in chunk["stage_times"]:
            print(f"    Stage {stage}: {stage_time} ms")
        print(f"    Chunk Total Time: {chunk['chunk_time']} ms")
    print(f"Total Pipeline Time: {total_time} ms")
    print("-" * 40)


def main():
    # For demonstration, we hard-code the machine and application.
    # (These could also be provided via input or command-line arguments.)
    machine_name = "3A021JEHN02756"
    application = "CifarDense"  # Alternatively, "Tree" or "CifarSparse"

    # Define an array of execution schedules.
    # Each schedule is a list of chunks.
    # Each chunk is a dict with:
    #   - chunk_id: an identifier,
    #   - stages: a list of stage numbers (stage 0 is not used),
    #   - hardware: the processing unit type ("little", "medium", "big", "gpu", etc.),
    #   - threads: the number of threads (or cores) to use.
    schedules = [
        [
            {"chunk_id": 1, "stages": [1, 2, 3], "hardware": "little", "threads": 4},
            {"chunk_id": 2, "stages": [4, 5], "hardware": "big", "threads": 2},
            {"chunk_id": 3, "stages": [6, 7, 8, 9], "hardware": "gpu", "threads": 1},
        ],
        [
            {"chunk_id": 1, "stages": [1, 2], "hardware": "medium", "threads": 2},
            {"chunk_id": 2, "stages": [3, 4, 5], "hardware": "little", "threads": 4},
            {"chunk_id": 3, "stages": [6, 7], "hardware": "big", "threads": 2},
            {"chunk_id": 4, "stages": [8, 9], "hardware": "gpu", "threads": 1},
        ],
    ]
    # Note: stage 0 is not used.

    # Connect to the SQLite database.
    conn = sqlite3.connect(DB_NAME)

    # Process each schedule.
    for idx, schedule in enumerate(schedules, start=1):
        print(f"--- Execution Schedule #{idx} ---")
        total_time = estimate_schedule_time(conn, machine_name, application, schedule)
        print_schedule_report(schedule, total_time)

    conn.close()


if __name__ == "__main__":
    main()
