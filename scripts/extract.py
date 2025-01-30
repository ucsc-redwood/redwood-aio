import sqlite3
import random


def create_database(db_name="benchmark_results.db"):
    """
    Create the SQLite database with a single table (Option A design).
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the table with a unique constraint to avoid duplicates.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_name TEXT,
            application TEXT,
            backend TEXT,
            stage INTEGER,
            core_type TEXT,
            num_threads INTEGER,
            time_ms REAL,
            UNIQUE (machine_name, application, backend, stage, core_type, num_threads)
        )
    """
    )
    conn.commit()
    conn.close()


def insert_benchmark_result(
    db_name, machine_name, application, backend, stage, core_type, num_threads, time_ms
):
    """
    Insert a single row into the benchmark_result table.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO benchmark_result
                (machine_name, application, backend, stage,
                 core_type, num_threads, time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                machine_name,
                application,
                backend,
                stage,
                core_type,
                num_threads,
                time_ms,
            ),
        )
    except sqlite3.IntegrityError:
        # If there's a unique constraint violation, handle it here.
        # For this demo, we'll just ignore duplicates.
        pass

    conn.commit()
    conn.close()


def populate_fake_data(db_name="benchmark_results.db"):
    """
    Populate the database with fake benchmark data:
      - Multiple machines
      - 3 applications (Tree, CifarSparse, CifarDense) with different stage counts
      - 3 backends (OMP, CUDA, VK)
      - OMP variants with different core_types and thread counts
      - A 'baseline' stage (stage=0) plus normal stages
    """
    machine_names = ["machine-01", "machine-02", "machine-03"]

    # Map each application to its number of stages
    # (Tree has 7; CifarSparse and CifarDense have 9)
    apps_and_stages = {"Tree": 7, "CifarSparse": 9, "CifarDense": 9}

    backends = ["OMP", "CUDA", "VK"]
    core_types = ["little", "medium", "big"]  # For OMP only
    max_threads = 3  # Or however many you like

    for machine_name in machine_names:
        for application, stage_count in apps_and_stages.items():
            for backend in backends:
                # Insert a baseline row as stage=0
                time_ms = round(random.uniform(1.0, 100.0), 2)
                if backend == "OMP":
                    # For baseline under OMP, we can pick e.g. 'little' & 1 thread,
                    # or just set them to None to indicate no meaning.
                    # But let's keep them as None for a general baseline row.
                    insert_benchmark_result(
                        db_name,
                        machine_name,
                        application,
                        backend,
                        0,
                        None,
                        None,
                        time_ms,
                    )
                else:
                    # For CUDA/VK baseline, just store them as None
                    insert_benchmark_result(
                        db_name,
                        machine_name,
                        application,
                        backend,
                        0,
                        None,
                        None,
                        time_ms,
                    )

                # Insert each stage
                for stage in range(1, stage_count + 1):
                    if backend == "OMP":
                        # For OMP, vary over core_types and thread counts
                        for ctype in core_types:
                            for tcount in range(1, max_threads + 1):
                                time_ms = round(random.uniform(1.0, 50.0), 2)
                                insert_benchmark_result(
                                    db_name,
                                    machine_name,
                                    application,
                                    backend,
                                    stage,
                                    ctype,
                                    tcount,
                                    time_ms,
                                )
                    else:
                        # For CUDA / VK, no core_type or num_threads
                        time_ms = round(random.uniform(1.0, 50.0), 2)
                        insert_benchmark_result(
                            db_name,
                            machine_name,
                            application,
                            backend,
                            stage,
                            None,
                            None,
                            time_ms,
                        )


def print_all_data(db_name="benchmark_results.db"):
    """
    Print out all rows in the benchmark_result table, ordered by machine, app, stage, etc.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT machine_name, application, backend, stage,
               core_type, num_threads, time_ms
        FROM benchmark_result
        ORDER BY machine_name, application, backend, stage, core_type, num_threads
    """
    )
    rows = cursor.fetchall()
    conn.close()

    print("All Benchmark Results:")
    for row in rows:
        (machine_name, application, backend, stage, core_type, num_threads, time_ms) = (
            row
        )
        print(
            f"Machine: {machine_name}, App: {application}, Backend: {backend}, "
            f"Stage: {stage}, CoreType: {core_type}, Threads: {num_threads}, "
            f"Time: {time_ms} ms"
        )


# import sqlite3

def compare_custom_pipeline_to_baseline(machine_name, application, db_name="benchmark_results.db"):
    """
    Compares the total time of a custom pipeline:
      - stages 1-2 => OMP (little cores, 2 threads)
      - stages 3-4 => OMP (big cores, 2 threads)
      - stages 5-6 => OMP (little cores, 2 threads)
      - stage 7    => CUDA (stage=7, no core_type, no num_threads)
    to a baseline time (stage=0) for the same machine/application.
    
    Prints out the total pipeline time, the baseline time, and the ratio.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 1) Query the custom pipeline times in a single SUM:
    #    We sum time_ms for each specific combination of (backend, core_type, num_threads, stage).
    #    Stages 1-2 => OMP + little + 2 threads
    #    Stages 3-4 => OMP + big    + 2 threads
    #    Stages 5-6 => OMP + little + 2 threads
    #    Stage  7   => CUDA + None  + None
    custom_pipeline_query = """
        SELECT SUM(time_ms)
        FROM benchmark_result
        WHERE machine_name = :machine
          AND application = :app
          AND (
               (backend='OMP'
                AND core_type='little'
                AND num_threads=2
                AND stage IN (1,2))
            OR (backend='OMP'
                AND core_type='big'
                AND num_threads=2
                AND stage IN (3,4))
            OR (backend='OMP'
                AND core_type='little'
                AND num_threads=2
                AND stage IN (5,6))
            OR (backend='CUDA'
                AND stage=7)
          )
    """
    
    cursor.execute(custom_pipeline_query, {"machine": machine_name, "app": application})
    custom_time_result = cursor.fetchone()
    custom_time = custom_time_result[0] if custom_time_result and custom_time_result[0] else 0.0
    
    # 2) Query the baseline time:
    #    We'll assume that your baseline is stored as stage=0. 
    #    If you want to specify a particular backend for baseline, do it here.
    baseline_query = """
        SELECT time_ms
        FROM benchmark_result
        WHERE machine_name = :machine
          AND application = :app
          AND stage = 0
          -- AND backend = ?  # optional if you want a specific backend's baseline
        LIMIT 1
    """
    cursor.execute(baseline_query, {"machine": machine_name, "app": application})
    baseline_result = cursor.fetchone()
    baseline_time = baseline_result[0] if baseline_result else 0.0
    
    conn.close()
    
    if baseline_time == 0:
        print(f"No baseline found for machine={machine_name}, application={application}")
        return
    
    # 3) Compute difference / ratio
    difference = custom_time - baseline_time
    ratio = custom_time / baseline_time if baseline_time != 0 else 0
    
    # 4) Print summary
    print(f"Machine: {machine_name}, Application: {application}")
    print(f"  Baseline Time (stage=0): {baseline_time} ms")
    print(f"  Custom Pipeline Time   : {custom_time} ms")
    print(f"  Difference             : {difference} ms")
    print(f"  Ratio (custom / base)  : {ratio:.2f}x")


if __name__ == "__main__":
    # 1. Create / initialize the database
    create_database()

    # 2. Populate with fake data
    populate_fake_data()

    # 3. Print everything to verify
    # print_all_data()

    # 4. Compare custom pipeline to baseline
    compare_custom_pipeline_to_baseline("machine-01", "Tree")
