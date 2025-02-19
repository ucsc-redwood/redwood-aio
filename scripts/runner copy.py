import subprocess
import re
from typing import Dict, Optional, Tuple
import time
import json
import glob
import numpy as np
from scipy import stats
import argparse
from colorama import init, Fore, Style
from tabulate import tabulate

NUM_SCHEDULES = 50

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
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
    print(f"\n{Style.BRIGHT}{'='*50}")
    print(f"{Fore.CYAN}FINAL REPORT{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{'='*50}{Style.RESET_ALL}\n")

    if successful_runs:
        # Summary Table
        summary_data = [
            ["Total schedules", args.num_schedules],
            ["Successful runs", f"{Fore.GREEN}{len(successful_runs)}{Style.RESET_ALL}"],
            ["Failed runs", f"{Fore.RED}{len(failed_runs)}{Style.RESET_ALL}"],
        ]
        print(f"{Style.BRIGHT}Test Summary:{Style.RESET_ALL}")
        print(tabulate(summary_data, tablefmt="simple"))
        print()

        # Performance Statistics Table
        actual_times = [results[s] for s in successful_runs]
        best_schedule = min(successful_runs, key=lambda x: results[x])
        worst_schedule = max(successful_runs, key=lambda x: results[x])

        stats_data = [
            ["Best Schedule", f"{best_schedule} ({results[best_schedule]:.2f} ms)"],
            ["Worst Schedule", f"{worst_schedule} ({results[worst_schedule]:.2f} ms)"],
            ["Mean Time", f"{np.mean(actual_times):.2f} ms"],
            ["Median Time", f"{np.median(actual_times):.2f} ms"],
            ["Standard Deviation", f"{np.std(actual_times):.2f} ms"],
            ["95th Percentile", f"{np.percentile(actual_times, 95):.2f} ms"],
            ["5th Percentile", f"{np.percentile(actual_times, 5):.2f} ms"],
        ]
        print(f"{Style.BRIGHT}Performance Statistics:{Style.RESET_ALL}")
        print(tabulate(stats_data, tablefmt="simple"))
        print()

        # Add new section for detailed performance analysis
        print(f"\n{Style.BRIGHT}Detailed Performance Analysis:{Style.RESET_ALL}")

        # Performance Distribution
        perf_data = []
        time_ranges = [(0, 25), (25, 30), (30, 35), (35, float("inf"))]

        for min_t, max_t in time_ranges:
            count = sum(1 for t in actual_times if min_t <= t < max_t)
            percentage = (count / len(actual_times)) * 100
            range_str = f"{min_t}-{max_t if max_t != float('inf') else '+'}"
            perf_data.append([f"{range_str} ms", count, f"{percentage:.1f}%"])

        print("\nPerformance Distribution:")
        print(
            tabulate(
                perf_data,
                headers=["Time Range", "Count", "Percentage"],
                tablefmt="simple",
            )
        )

        # Stability Analysis
        print(f"\n{Style.BRIGHT}Stability Analysis:{Style.RESET_ALL}")

        # Calculate coefficient of variation (CV)
        cv = (np.std(actual_times) / np.mean(actual_times)) * 100

        # Calculate quartiles and IQR
        q1 = np.percentile(actual_times, 25)
        q3 = np.percentile(actual_times, 75)
        iqr = q3 - q1

        # Calculate outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [t for t in actual_times if t < lower_bound or t > upper_bound]

        stability_data = [
            ["Coefficient of Variation", f"{cv:.1f}%"],
            ["Interquartile Range", f"{iqr:.2f} ms"],
            ["Number of Outliers", len(outliers)],
            ["Outlier Percentage", f"{(len(outliers)/len(actual_times))*100:.1f}%"],
        ]
        print(tabulate(stability_data, tablefmt="simple"))

        # Prediction Analysis
        prediction_differences = []
        faster_than_predicted = []
        slower_than_predicted = []

        prediction_data = []
        for schedule in successful_runs:
            actual = results[schedule]
            predicted = predictions.get(schedule)
            if predicted:
                diff = calculate_prediction_accuracy(actual, predicted)[0]
                prediction_differences.append(diff)
                status = (
                    f"{Fore.RED}+{diff:.1f}%"
                    if diff > 0
                    else f"{Fore.GREEN}{diff:.1f}%{Style.RESET_ALL}"
                )

                prediction_data.append(
                    [schedule, f"{actual:.2f}", f"{predicted:.2f}", status]
                )

                if diff > 0:
                    slower_than_predicted.append(diff)
                else:
                    faster_than_predicted.append(abs(diff))

        if prediction_differences:
            print(f"\n{Style.BRIGHT}Prediction Results:{Style.RESET_ALL}")
            headers = ["Schedule", "Measured (ms)", "Predicted (ms)", "Difference"]
            print(tabulate(prediction_data, headers=headers, tablefmt="simple"))

            print(f"\n{Style.BRIGHT}Prediction Accuracy Distribution:{Style.RESET_ALL}")
            within_5_percent = sum(abs(x) <= 5 for x in prediction_differences)
            within_10_percent = sum(abs(x) <= 10 for x in prediction_differences)
            within_20_percent = sum(abs(x) <= 20 for x in prediction_differences)
            total_predictions = len(prediction_differences)

            accuracy_data = [
                ["Within ±5%", f"{within_5_percent/total_predictions*100:.1f}%"],
                ["Within ±10%", f"{within_10_percent/total_predictions*100:.1f}%"],
                ["Within ±20%", f"{within_20_percent/total_predictions*100:.1f}%"],
            ]
            print(tabulate(accuracy_data, tablefmt="simple"))

            # Prediction Error Analysis
            print(f"\n{Style.BRIGHT}Prediction Error Analysis:{Style.RESET_ALL}")

            # Calculate error statistics
            abs_errors = [abs(diff) for diff in prediction_differences]
            mean_abs_error = np.mean(abs_errors)
            rmse = np.sqrt(np.mean(np.square(prediction_differences)))

            error_data = [
                ["Mean Absolute Error", f"{mean_abs_error:.1f}%"],
                ["Root Mean Square Error", f"{rmse:.1f}%"],
                ["Error Standard Deviation", f"{np.std(prediction_differences):.1f}%"],
                ["Maximum Overprediction", f"{min(prediction_differences):.1f}%"],
                ["Maximum Underprediction", f"{max(prediction_differences):.1f}%"],
            ]
            print(tabulate(error_data, tablefmt="simple"))

            # Correlation Analysis
            print(f"\n{Style.BRIGHT}Correlation Analysis:{Style.RESET_ALL}")

            # Calculate correlation between predicted and actual times
            predicted_times = [
                predictions[s] for s in successful_runs if s in predictions
            ]
            actual_times_corr = [
                results[s] for s in successful_runs if s in predictions
            ]

            correlation = np.corrcoef(predicted_times, actual_times_corr)[0, 1]

            corr_data = [
                ["Prediction-Actual Correlation", f"{correlation:.3f}"],
                [
                    "Correlation Strength",
                    (
                        "Strong"
                        if abs(correlation) > 0.7
                        else "Moderate" if abs(correlation) > 0.4 else "Weak"
                    ),
                ],
            ]
            print(tabulate(corr_data, tablefmt="simple"))

    if failed_runs:
        print(f"\n{Style.BRIGHT}{Fore.RED}Failed Runs:{Style.RESET_ALL}")
        for schedule in failed_runs:
            print(f"  • Schedule {schedule}")


if __name__ == "__main__":
    main()
