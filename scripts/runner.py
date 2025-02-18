# adb -s 3A021JEHN02756 shell /data/local/tmp/pipe-cifar-dense-vk -l info --device=3A021JEHN02756 -s 1


NUM_SCHEDULES = 50

import subprocess
import re
from typing import Dict, Optional, Tuple
import time
import json
import glob
import os
import numpy as np
from scipy import stats
import argparse
from colorama import init, Fore, Style

# Initialize colorama
init()


def build_binary():
    """Build the binary using xmake"""
    print("Building binary with xmake...")
    try:
        subprocess.run(["xmake", "b", "pipe-cifar-dense-vk"], check=True)
        print("Build successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False


def load_mathematical_predictions(device_id: str) -> Dict[int, float]:
    """Load all mathematical predictions from schedule JSON files"""
    predictions = {}

    # Update glob pattern to use device_id
    schedule_files = glob.glob(
        f"./data/generated-schedules/{device_id}_CifarDense_schedule_*.json"
    )

    for file_path in schedule_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Extract schedule number from filename, accounting for the hash
                schedule_num = int(
                    re.search(r"schedule_(\d+)\.json", file_path).group(1)
                )
                predictions[schedule_num] = data["max_chunk_time"]
        except Exception as e:
            print(f"Warning: Could not load predictions from {file_path}: {e}")

    return predictions


def run_command(device_id: str, schedule_num: int) -> Optional[float]:
    """
    Run the command for a given schedule number and return the average time if successful.
    Returns None if the execution failed.
    """
    # Push the executable to the device
    try:
        subprocess.run(
            [
                "adb",
                "-s",
                device_id,
                "push",
                "./build/android/arm64-v8a/release/pipe-cifar-dense-vk",
                "/data/local/tmp/pipe-cifar-dense-vk",
            ],
            check=True,
            stdout=subprocess.DEVNULL,  # Hide stdout
            stderr=subprocess.DEVNULL,  # Hide stderr
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to push executable to device: {e}")
        return None

    # Run the executable
    cmd = [
        "adb",
        "-s",
        device_id,
        "shell",
        "/data/local/tmp/pipe-cifar-dense-vk",
        "-l",
        "info",
        f"--device={device_id}",
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
        print(f"{Fore.RED}Schedule {schedule_num} timed out{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(
            f"{Fore.RED}Schedule {schedule_num} failed with error: {str(e)}{Style.RESET_ALL}"
        )
        return None


def calculate_prediction_accuracy(actual: float, predicted: float) -> Tuple[float, str]:
    """Calculate the percentage difference between actual and predicted times"""
    diff_percent = ((actual - predicted) / predicted) * 100
    if diff_percent > 0:
        status = "slower"
    else:
        status = "faster"
    return diff_percent, status


def create_console_histogram(data, bins=10, width=50):
    """Create a simple console-based histogram"""
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)

    histogram = []
    for count, edge in zip(hist, bin_edges[:-1]):
        bar = "#" * int((count / max_count) * width)
        histogram.append(f"{edge:6.2f}% | {bar} ({count})")

    return "\n".join(histogram)


def main():
    parser = argparse.ArgumentParser(
        description="Run performance tests on a specific device"
    )
    parser.add_argument("device_id", help="Device ID (e.g., 3A021JEHN02756)")
    parser.add_argument(
        "--num-schedules",
        type=int,
        default=50,
        help="Number of schedules to test (default: 50)",
    )
    args = parser.parse_args()

    # Build the binary first
    if not build_binary():
        print("Exiting due to build failure")
        return

    results: Dict[int, Optional[float]] = {}
    successful_runs = []
    failed_runs = []

    # Load mathematical predictions with device ID
    predictions = load_mathematical_predictions(args.device_id)

    print(f"Starting test runs on device {args.device_id}...")
    print("-" * 50)

    for schedule_num in range(1, args.num_schedules + 1):
        print(f"Running schedule {schedule_num}/{args.num_schedules}...")
        avg_time = run_command(args.device_id, schedule_num)
        results[schedule_num] = avg_time

        if avg_time is not None:
            successful_runs.append(schedule_num)
            predicted = predictions.get(schedule_num)
            if predicted:
                diff = calculate_prediction_accuracy(avg_time, predicted)[0]
                print(f"\nSchedule {schedule_num}:")
                print(f"  → Measured: {avg_time:.2f} ms")
                print(f"  → Predicted: {predicted:.2f} ms")
                if diff > 0:
                    print(
                        f"  → {Fore.RED}{abs(diff):.1f}% slower than predicted{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"  → {Fore.GREEN}{abs(diff):.1f}% faster than predicted{Style.RESET_ALL}"
                    )
            else:
                print(f"\nSchedule {schedule_num}:")
                print(f"  → Measured: {avg_time:.2f} ms")
                print("  → No prediction available")
        else:
            failed_runs.append(schedule_num)
            print(f"{Fore.RED}Schedule {schedule_num} failed{Style.RESET_ALL}")

        # Small delay between runs to avoid overwhelming the device
        time.sleep(1)

    # Generate report
    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)

    print("\nSuccessful runs with prediction comparison:")
    prediction_accuracy = []
    for schedule in successful_runs:
        actual = results[schedule]
        predicted = predictions.get(schedule)
        if predicted:
            diff = calculate_prediction_accuracy(actual, predicted)[0]
            prediction_accuracy.append(diff)
            print(f"Schedule {schedule}:")
            print(f"  Actual: {actual:.2f} ms")
            print(f"  Predicted: {predicted:.2f} ms")
            print(
                f"  {abs(diff):.1f}% {'slower' if diff > 0 else 'faster'} than predicted"
            )

    print("\nFailed runs:")
    for schedule in failed_runs:
        print(f"Schedule {schedule}")

    if successful_runs:
        actual_times = [results[s] for s in successful_runs]
        prediction_differences = []
        faster_than_predicted = []
        slower_than_predicted = []

        for schedule in successful_runs:
            actual = results[schedule]
            predicted = predictions.get(schedule)
            if predicted:
                diff = calculate_prediction_accuracy(actual, predicted)[0]
                prediction_differences.append(diff)
                if diff > 0:
                    slower_than_predicted.append(diff)
                else:
                    faster_than_predicted.append(abs(diff))

        # Basic Performance Statistics
        print("\nPerformance Summary:")
        print(f"Total schedules tested: {args.num_schedules}")
        print(f"Successful runs: {len(successful_runs)}")
        print(f"Failed runs: {len(failed_runs)}")

        best_schedule = min(successful_runs, key=lambda x: results[x])
        worst_schedule = max(successful_runs, key=lambda x: results[x])

        print(f"\nExecution Time Statistics:")
        print(
            f"Best performing schedule: {best_schedule} ({results[best_schedule]:.2f} ms)"
        )
        print(
            f"Worst performing schedule: {worst_schedule} ({results[worst_schedule]:.2f} ms)"
        )
        print(f"Mean execution time: {np.mean(actual_times):.2f} ms")
        print(f"Median execution time: {np.median(actual_times):.2f} ms")
        print(f"Standard deviation: {np.std(actual_times):.2f} ms")
        print(f"95th percentile: {np.percentile(actual_times, 95):.2f} ms")
        print(f"5th percentile: {np.percentile(actual_times, 5):.2f} ms")

        if prediction_differences:
            print("\nPrediction Accuracy Summary:")
            print("Faster than predicted:")
            if faster_than_predicted:
                print(f"  Count: {len(faster_than_predicted)}")
                print(f"  Mean error: {np.mean(faster_than_predicted):.1f}%")
                print(f"  Max error: {max(faster_than_predicted):.1f}%")
                print(f"  Min error: {min(faster_than_predicted):.1f}%")
            else:
                print("  No schedules were faster than predicted")

            print("\nSlower than predicted:")
            if slower_than_predicted:
                print(f"  Count: {len(slower_than_predicted)}")
                print(f"  Mean error: {np.mean(slower_than_predicted):.1f}%")
                print(f"  Max error: {max(slower_than_predicted):.1f}%")
                print(f"  Min error: {min(slower_than_predicted):.1f}%")
            else:
                print("  No schedules were slower than predicted")

            print("\nOverall Prediction Statistics:")
            print(f"Mean prediction error: {np.mean(prediction_differences):.1f}%")
            print(f"Median prediction error: {np.median(prediction_differences):.1f}%")
            print(f"Standard deviation: {np.std(prediction_differences):.1f}%")
            print(f"Skewness: {stats.skew(prediction_differences):.2f}")
            print(f"Kurtosis: {stats.kurtosis(prediction_differences):.2f}")

            # Histogram of prediction errors
            print("\nHistogram of Prediction Errors:")
            print(
                "(Negative % = faster than predicted, Positive % = slower than predicted)"
            )
            print(create_console_histogram(prediction_differences))

            # Additional analysis
            print("\nPrediction Reliability:")
            within_5_percent = sum(abs(x) <= 5 for x in prediction_differences)
            within_10_percent = sum(abs(x) <= 10 for x in prediction_differences)
            within_20_percent = sum(abs(x) <= 20 for x in prediction_differences)

            total_predictions = len(prediction_differences)
            print(
                f"Predictions within ±5%: {within_5_percent/total_predictions*100:.1f}%"
            )
            print(
                f"Predictions within ±10%: {within_10_percent/total_predictions*100:.1f}%"
            )
            print(
                f"Predictions within ±20%: {within_20_percent/total_predictions*100:.1f}%"
            )


if __name__ == "__main__":
    main()
