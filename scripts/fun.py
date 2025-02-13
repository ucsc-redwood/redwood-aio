import sqlite3
import itertools

DB_NAME = "benchmark_results.db"

# --------------------------------------------------------------------
# 1) Device dictionary: total vs. usable cores
#    (same structure as before; adjust as needed)
devices = {
    "3A021JEHN02756": {
        "little": {"total": 4, "usable": 4},
        "mid":    {"total": 2, "usable": 2},
        "big":    {"total": 2, "usable": 2},
        "gpu":    {"total": 1, "usable": 1}   # We'll treat 'gpu' => backend=VK, core_type=None
    },
    "9b034f1b": {
        "little": {"total": 3, "usable": 3},
        "mid":    {"total": 4, "usable": 2},  # 2 mid cores are not pinable => 2 usable
        "big":    {"total": 1, "usable": 0},  # big core not pinable => 0 usable
        "gpu":    {"total": 1, "usable": 1}   # We'll treat 'gpu' => backend=VK, core_type=None
    },
    "ce0717178d7758b00b7e": {
        "small":  {"total": 4, "usable": 4},
        "big":    {"total": 4, "usable": 4},
        "gpu":    {"total": 1, "usable": 1}
    },
    "jetson": {
        "core":   {"total": 6, "usable": 6},
        "gpu":    {"total": 1, "usable": 1}
    }
}

# --------------------------------------------------------------------
# 2) Partition function (same as before)
def partition_stages(stages):
    """
    Generate all ways to partition a list of stages into consecutive chunks.
    E.g. for [1,2,3], we get:
      [[1,2,3]],
      [[1],[2,3]],
      [[1,2],[3]],
      [[1],[2],[3]]
    """
    n = len(stages)
    if n <= 1:
        yield [stages]
        return

    for cut_pattern in range(1 << (n - 1)):
        chunk = []
        partition = []
        for i in range(n):
            chunk.append(stages[i])
            if (i == n - 1) or (cut_pattern & (1 << i)):
                partition.append(chunk)
                chunk = []
        yield partition

# --------------------------------------------------------------------
# 3) Schedule generation: chunk-based, each chunk picks a single hardware
#    using all of its usable threads.
def generate_execution_schedules(stages, hw_specs):
    """
    For demonstration, we'll just create partitions with 1..(len(stages)) chunks,
    and for each chunk, pick one hardware type from 'hw_specs'.
    
    This can create many schedules. You could refine to exactly N chunks, etc.
    """
    # Filter out HW with 0 usable cores
    hw_specs_filtered = {k: v for k, v in hw_specs.items() if v > 0}
    hw_types = list(hw_specs_filtered.keys())
    
    for partition in partition_stages(stages):
        # For each partition (chunks), we do a cartesian product of hardware types
        # across the number of chunks.
        combos = itertools.product(hw_types, repeat=len(partition))
        for hw_combo in combos:
            schedule = []
            for chunk_id, (chunk_stages, hw) in enumerate(zip(partition, hw_combo), start=1):
                schedule.append({
                    'chunk_id': chunk_id,
                    'stages': chunk_stages,
                    'hardware': hw,              # e.g. 'little', 'big', 'gpu', ...
                    'threads': hw_specs_filtered[hw]  # e.g. 4, 2, 1, etc.
                })
            yield schedule

# --------------------------------------------------------------------
# 4) Database query function to get time for a single stage's assignment
def get_stage_time(conn, machine_name, application, stage, hardware, threads):
    """
    Query the database for the time (time_ms) given the combination of:
    machine_name, application, stage, backend, core_type, num_threads.
    
    We'll assume:
       - If `hardware` == "gpu", then we do backend='VK', core_type=None, num_threads=None
       - Else backend='OMP', core_type=hardware, num_threads=threads
    Adjust logic if your real DB uses different labels or if you want to handle e.g. 'small' vs 'little'.
    """
    # Simple mapping for demonstration:
    if hardware.lower() == "gpu":
        backend = "VK"
        core_type = None
        num_threads = None
    else:
        backend = "OMP"
        # Some devices have "small" or "core" etc. Instead of direct hardware,
        # you might need further logic. For example:
        #   hardware = "small" => core_type="little"
        #   hardware = "core"  => core_type="little"
        # For simplicity, here we treat hardware as the DB's core_type directly.
        core_type = hardware
        num_threads = threads
    
    query = """
        SELECT time_ms
        FROM benchmark_result
        WHERE machine_name = ?
          AND application = ?
          AND backend = ?
          AND stage = ?
    """
    params = [machine_name, application, backend, stage]

    # If core_type is not None, add that filter
    if core_type is not None:
        query += " AND core_type = ?"
        params.append(core_type)
    else:
        query += " AND core_type IS NULL"

    # If num_threads is not None, add that filter
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
        # If not found, you can return some sentinel or raise an exception
        # We'll just return 9999.9 to indicate "no data"
        return 9999.9

# --------------------------------------------------------------------
# 5) Helper: estimate schedule times from DB
def estimate_schedule_time(conn, machine_name, application, schedule):
    """
    Given a schedule (list of chunk dicts), look up the time for each stage's assignment
    in the DB. Return total pipeline time. Also attach per-stage and per-chunk times.
    
    We'll treat the pipeline as sequential: chunk1, chunk2, ...
    So total pipeline time = sum of chunk times.
    Each chunk's time = sum of stage times in that chunk.
    """
    total_time = 0.0

    for chunk in schedule:
        chunk_time = 0.0
        # For reporting, store times per stage
        chunk['stage_times'] = []
        for stage_id in chunk['stages']:
            stage_time = get_stage_time(
                conn,
                machine_name=machine_name,
                application=application,
                stage=stage_id,
                hardware=chunk['hardware'],
                threads=chunk['threads']
            )
            chunk['stage_times'].append( (stage_id, stage_time) )
            chunk_time += stage_time
        chunk['chunk_time'] = chunk_time
        total_time += chunk_time

    return total_time

# --------------------------------------------------------------------
# 6) Main script to tie it all together
def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", default=DB_NAME, help="Path to the SQLite database")
    args = parser.parse_args()

    # Ask user which device
    print("Available devices:")
    for d in devices.keys():
        print("  ", d)
    device_id = input("\nWhich device would you like to use? ")
    if device_id not in devices:
        print("Error: invalid device ID.")
        return
    
    # Ask user which application
    # We'll assume three possible apps: "CifarDense", "CifarSparse", "Tree"
    print("\nApplications available: Tree, CifarDense, CifarSparse")
    application = input("Which application would you like to test? ")
    if application not in ["Tree", "CifarDense", "CifarSparse"]:
        print("Error: invalid application.")
        return
    
    # Connect to DB
    conn = sqlite3.connect(args.db_name)
    
    # For demonstration, let's define a pipeline with 10 stages: [0..9].
    # Or you can do [1..9]. Adapt as needed to match your DB stage indexing.
    stages = list(range(0, 10))

    # Build {hw_type: usable_cores} from device specs
    hw_specs = {}
    for hw_type, info in devices[device_id].items():
        hw_specs[hw_type] = info["usable"]
    
    print(f"\nGenerating schedules for device={device_id}, application={application}.")
    all_schedules = generate_execution_schedules(stages, hw_specs)

    # We'll just iterate over all schedules (could be huge). Print a few examples.
    # Or keep them all in a list.
    schedules_list = list(all_schedules)
    print(f"Total schedules generated: {len(schedules_list)}")

    # Now pick some schedules to estimate times for:
    # For example, the first 5.
    schedules_to_print = schedules_list[:5]
    
    for idx, schedule in enumerate(schedules_to_print, start=1):
        print("\n------------------------------------")
        print(f"Schedule #{idx} (out of {len(schedules_list)})")
        total_time = estimate_schedule_time(conn, device_id, application, schedule)
        
        # Print details
        for chunk in schedule:
            hw     = chunk['hardware']
            thr    = chunk['threads']
            c_time = chunk['chunk_time']
            st_list= chunk['stage_times']  # [(stage_id, time_ms), ...]
            
            print(f"  Chunk {chunk['chunk_id']}: HW={hw}, threads={thr}")
            for (stg, st_time) in st_list:
                print(f"    Stage {stg}: {st_time} ms")
            print(f"  -> Chunk total time = {c_time:.3f} ms")
        
        print(f"==> Pipeline total time = {total_time:.3f} ms")

    conn.close()

if __name__ == "__main__":
    main()
