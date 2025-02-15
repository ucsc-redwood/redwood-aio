# adb -s 3A021JEHN02756 shell /data/local/tmp/pipe-cifar-dense-vk -l info --device=3A021JEHN02756 -s 1


NUM_SCHEDULES = 76

import subprocess
import re
from typing import Dict, Optional
import time


def run_command(schedule_num: int) -> Optional[float]:
    """
    Run the command for a given schedule number and return the average time if successful.
    Returns None if the execution failed.
    """
    cmd = [
        "adb",
        "-s",
        "3A021JEHN02756",
        "shell",
        "/data/local/tmp/pipe-cifar-dense-vk",
        "-l",
        "info",
        "--device=3A021JEHN02756",
        "-s",
        str(schedule_num),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr

        # Look for the average time pattern
        time_match = re.search(r"Average time per iteration: (\d+\.?\d*)", output)
        if time_match:
            return float(time_match.group(1))
        return None
    except subprocess.TimeoutExpired:
        print(f"Schedule {schedule_num} timed out")
        return None
    except Exception as e:
        print(f"Schedule {schedule_num} failed with error: {str(e)}")
        return None


def main():
    results: Dict[int, Optional[float]] = {}
    successful_runs = []
    failed_runs = []

    print("Starting test runs...")
    print("-" * 50)

    for schedule_num in range(1, NUM_SCHEDULES + 1):
        print(f"Running schedule {schedule_num}/{NUM_SCHEDULES}...")
        avg_time = run_command(schedule_num)
        results[schedule_num] = avg_time

        if avg_time is not None:
            successful_runs.append(schedule_num)
            print(f"Schedule {schedule_num} completed: {avg_time:.2f} ms")
        else:
            failed_runs.append(schedule_num)
            print(f"Schedule {schedule_num} failed")

        # Small delay between runs to avoid overwhelming the device
        time.sleep(1)

    # Generate report
    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)

    print("\nSuccessful runs:")
    for schedule in successful_runs:
        print(f"Schedule {schedule}: {results[schedule]:.2f} ms")

    print("\nFailed runs:")
    for schedule in failed_runs:
        print(f"Schedule {schedule}")

    if successful_runs:
        best_schedule = min(successful_runs, key=lambda x: results[x])
        worst_schedule = max(successful_runs, key=lambda x: results[x])
        avg_time = sum(results[s] for s in successful_runs) / len(successful_runs)

        print("\nSummary:")
        print(f"Total schedules tested: {NUM_SCHEDULES}")
        print(f"Successful runs: {len(successful_runs)}")
        print(f"Failed runs: {len(failed_runs)}")
        print(
            f"Best performing schedule: {best_schedule} ({results[best_schedule]:.2f} ms)"
        )
        print(
            f"Worst performing schedule: {worst_schedule} ({results[worst_schedule]:.2f} ms)"
        )
        print(f"Average execution time: {avg_time:.2f} ms")


if __name__ == "__main__":
    main()
