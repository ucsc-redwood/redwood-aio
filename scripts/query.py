import sqlite3


def compare_custom_pipeline_to_baseline_verbose(
    machine_name, application, db_name="benchmark_results.db"
):
    """
    Compares the total time of a custom pipeline to the baseline (stage=0),
    but also prints individual stage rows that sum to the final pipeline time.

    Custom Pipeline (example):
      - stages 1-2 => OMP (little cores, 2 threads)
      - stages 3-4 => OMP (big cores, 2 threads)
      - stages 5-6 => OMP (little cores, 2 threads)
      - stage 7    => CUDA (stage=7, no core_type, no num_threads)
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 1) Query all relevant rows for the pipeline
    custom_pipeline_query = """
        SELECT stage, backend, core_type, num_threads, time_ms
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
        ORDER BY stage
    """
    cursor.execute(custom_pipeline_query, {"machine": machine_name, "app": application})
    pipeline_rows = cursor.fetchall()

    # 2) Query the baseline time (assuming baseline is stage=0)
    baseline_query = """
        SELECT time_ms
        FROM benchmark_result
        WHERE machine_name = :machine
          AND application = :app
          AND stage = 0
        LIMIT 1
    """
    cursor.execute(baseline_query, {"machine": machine_name, "app": application})
    baseline_result = cursor.fetchone()
    baseline_time = baseline_result[0] if baseline_result else 0.0

    conn.close()

    # 3) Print baseline info (check if baseline is missing)
    print(f"Machine: {machine_name}, Application: {application}")
    if baseline_time <= 0:
        print("  (No baseline found or baseline time=0. Can't compute ratio.)")
        print(f"  Baseline time: {baseline_time} ms")
    else:
        print(f"  Baseline time (stage=0): {baseline_time} ms")

    # 4) Print each pipeline stage row, sum up the total
    total_custom_time = 0.0
    if pipeline_rows:
        print("\nCustom Pipeline Stages:")
        for stage, backend, core_type, num_threads, time_ms in pipeline_rows:
            total_custom_time += time_ms
            print(
                f"  Stage={stage}, Backend={backend}, core_type={core_type}, "
                f"threads={num_threads}, time={time_ms} ms"
            )

    else:
        print("\nNo rows found for the specified pipeline.")

    # 5) Print the total pipeline time and compare to baseline
    print(f"\nTotal Custom Pipeline Time: {total_custom_time} ms")

    if baseline_time > 0:
        difference = total_custom_time - baseline_time
        ratio = total_custom_time / baseline_time
        print(f"Difference vs baseline: {difference:.2f} ms")
        print(f"Ratio (custom / baseline): {ratio:.2f}x")


# Example usage:
if __name__ == "__main__":
    # benchmark_results.db

    compare_custom_pipeline_to_baseline_verbose(
        machine_name="machine-01", application="Tree", db_name="benchmark_results.db"
    )
