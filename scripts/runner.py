#!/usr/bin/env python3
from pathlib import Path
import re
import time
import json
from typing import Dict, Optional, List, NamedTuple
import argparse
import numpy as np
from colorama import init, Fore, Style
from tabulate import tabulate
import subprocess

from helpers import (
    run_command,
    ALL_DEVICES,
    ALL_APPLICATIONS,
    APPLICATION_NAME_MAP,
    GENERATED_SCHEDULES_PATH,
    interactive_select,
    parse_schedule_range,
    select_schedules,
)

# Initialize colorama for colored output
init()


class RunResult(NamedTuple):
    """Store results of a benchmark run."""

    time: Optional[float]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.time is not None


def load_predictions(device_id: str, app: str) -> Dict[int, float]:
    """Load mathematical predictions from schedule JSON files."""
    predictions = {}
    pattern = Path(GENERATED_SCHEDULES_PATH) / f"{device_id}_{app}_schedule_*.json"

    for file_path in pattern.parent.glob(pattern.name):
        try:
            schedule_num = int(
                re.search(r"schedule_(\d+)\.json", file_path.name).group(1)
            )
            with open(file_path) as f:
                predictions[schedule_num] = json.load(f)["max_chunk_time"]
        except Exception as e:
            print(f"Warning: Could not load predictions from {file_path}: {e}")

    return predictions


def run_schedule(device_id: str, app: str, schedule_num: int) -> RunResult:
    """Run a single schedule and return its execution time."""
    app_name = app.lower()
    binary_name = f"pipe-{app_name}-vk"
    device_path = f"/data/local/tmp/{binary_name}"

    # Push executable to device
    try:
        push_cmd = (
            f"adb -s {device_id} push "
            f"./build/android/arm64-v8a/release/{binary_name} "
            f"{device_path}"
        )
        run_command(push_cmd, hide_output=True)
    except Exception as e:
        return RunResult(None, f"Failed to push executable: {e}")

    # Run the executable
    try:
        run_cmd = (
            f"adb -s {device_id} shell {device_path} "
            f"-l info --device={device_id} -s {schedule_num}"
        )
        # Capture output using subprocess directly for this command
        result = subprocess.run(
            run_cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        output = result.stdout + result.stderr

        # Extract average time from output
        if match := re.search(r"Average time per iteration: (\d+\.?\d*)", output):
            return RunResult(float(match.group(1)))
        return RunResult(None, "Could not find timing in output")

    except subprocess.TimeoutExpired:
        return RunResult(None, "Execution timed out")
    except Exception as e:
        return RunResult(None, str(e))


def print_schedule_result(
    schedule_num: int, result: RunResult, predicted: Optional[float] = None
) -> None:
    """Print the result of a single schedule run."""
    print(f"\nSchedule {schedule_num}:")

    if not result.success:
        print(f"  {Fore.RED}→ Failed: {result.error}{Style.RESET_ALL}")
        return

    print(f"  → Measured: {result.time:.2f} ms")

    if predicted:
        diff = ((result.time - predicted) / predicted) * 100
        status = "slower" if diff > 0 else "faster"
        color = Fore.RED if status == "slower" else Fore.GREEN
        print(f"  → Predicted: {predicted:.2f} ms")
        print(f"  → {color}{abs(diff):.1f}% {status} than predicted{Style.RESET_ALL}")
    else:
        print("  → No prediction available")


def generate_performance_report(
    results: Dict[int, RunResult], predictions: Dict[int, float]
) -> None:
    """Generate and print detailed performance analysis report."""
    successful_results = {k: r.time for k, r in results.items() if r.success}
    if not successful_results:
        print(f"\n{Fore.RED}No successful runs to report{Style.RESET_ALL}")
        return

    successful_times = list(successful_results.values())

    # Basic statistics
    stats_data = [
        ["Total Runs", len(results)],
        ["Successful Runs", len(successful_results)],
        ["Failed Runs", len(results) - len(successful_results)],
        ["Best Time", f"{min(successful_times):.2f} ms"],
        ["Worst Time", f"{max(successful_times):.2f} ms"],
        ["Mean Time", f"{np.mean(successful_times):.2f} ms"],
        ["Median Time", f"{np.median(successful_times):.2f} ms"],
        ["Std Dev", f"{np.std(successful_times):.2f} ms"],
    ]

    print(f"\n{Style.BRIGHT}Performance Statistics:{Style.RESET_ALL}")
    print(tabulate(stats_data, tablefmt="simple"))

    # Prediction accuracy
    if predictions:
        diffs = []
        for schedule, time in successful_results.items():
            if predicted := predictions.get(schedule):
                diff = ((time - predicted) / predicted) * 100
                diffs.append(diff)

        if diffs:
            accuracy_data = [
                [
                    "Within ±5%",
                    f"{sum(abs(d) <= 5 for d in diffs)/len(diffs)*100:.1f}%",
                ],
                [
                    "Within ±10%",
                    f"{sum(abs(d) <= 10 for d in diffs)/len(diffs)*100:.1f}%",
                ],
                ["Mean Error", f"{np.mean(np.abs(diffs)):.1f}%"],
                ["Max Error", f"{np.max(np.abs(diffs)):.1f}%"],
            ]

            print(f"\n{Style.BRIGHT}Prediction Accuracy:{Style.RESET_ALL}")
            print(tabulate(accuracy_data, tablefmt="simple"))

    # Print failed runs
    failed = [(k, r.error) for k, r in results.items() if not r.success]
    if failed:
        print(f"\n{Style.BRIGHT}{Fore.RED}Failed Runs:{Style.RESET_ALL}")
        for schedule, error in failed:
            print(f"  • Schedule {schedule}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Run performance tests on Android devices"
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device ID (if not provided, interactive selection will be used)",
    )
    parser.add_argument(
        "--application",
        "-a",
        help="Application to test (if not provided, interactive selection will be used)",
        choices=ALL_APPLICATIONS,
    )
    parser.add_argument(
        "--schedules",
        "-s",
        help="Schedule range to test (e.g. '1-5' or '1,3,5' or '1-3,5,7-9')",
    )
    args = parser.parse_args()

    # Select single device and application
    device = args.device if args.device else interactive_select(ALL_DEVICES, "device")
    app = (
        args.application
        if args.application
        else interactive_select(ALL_APPLICATIONS, "application")
    )

    print(f"\nRunning tests for {app} on device {device}")

    # Build executable if needed
    binary_name = f"pipe-{app.lower()}-vk"
    binary_path = Path(f"./build/android/arm64-v8a/release/{binary_name}")
    if not binary_path.exists():
        print(f"Building {binary_name}...")
        run_command(f"xmake b {binary_name}")

    # Load predictions using canonical name
    print("Loading predictions...")
    app_canonical = APPLICATION_NAME_MAP[app]
    predictions = load_predictions(device, app_canonical)
    if not predictions:
        print(f"{Fore.YELLOW}Warning: No predictions found{Style.RESET_ALL}")
        return

    # Get schedules for this device-app pair
    schedule_ids = select_schedules(device, app, args.schedules)
    available_schedules = sorted(predictions.keys())

    # Validate that selected schedules have predictions
    invalid_ids = [sid for sid in schedule_ids if sid not in available_schedules]
    if invalid_ids:
        print(
            f"{Fore.RED}No predictions available for schedules {invalid_ids}{Style.RESET_ALL}"
        )
        return

    print(f"Found {len(available_schedules)} schedule files")
    print(f"Will test {len(schedule_ids)} schedules")

    # Run schedules
    results: Dict[int, RunResult] = {}
    for schedule_num in sorted(schedule_ids):
        print(f"\nRunning schedule {schedule_num}...")
        result = run_schedule(device, app, schedule_num)
        results[schedule_num] = result
        print_schedule_result(schedule_num, result, predictions.get(schedule_num))
        time.sleep(1)

    # Generate final report
    print(f"\n{Style.BRIGHT}Results for {app} on device {device}:{Style.RESET_ALL}")
    generate_performance_report(results, predictions)


if __name__ == "__main__":
    main()
