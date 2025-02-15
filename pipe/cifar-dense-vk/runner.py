# adb -s 3A021JEHN02756 shell /data/local/tmp/pipe-cifar-dense-vk -l info --device=3A021JEHN02756 -s 1


NUM_SCHEDULES = 76

import subprocess
import re
from typing import Dict, Optional, Tuple
import time
import json
import glob
import os


def load_mathematical_predictions() -> Dict[int, float]:
    """Load all mathematical predictions from schedule JSON files"""
    predictions = {}
    schedule_files = glob.glob("./scripts/schedules/3A021JEHN02756_CifarDense_schedule_*_*.json")
    
    for file_path in schedule_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract schedule number from filename
                schedule_num = int(re.search(r'schedule_(\d+)_', file_path).group(1))
                predictions[schedule_num] = data['max_chunk_time']
        except Exception as e:
            print(f"Warning: Could not load predictions from {file_path}: {e}")
    
    return predictions


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


def calculate_prediction_accuracy(actual: float, predicted: float) -> Tuple[float, str]:
    """Calculate the percentage difference between actual and predicted times"""
    diff_percent = ((actual - predicted) / predicted) * 100
    if diff_percent > 0:
        status = "slower"
    else:
        status = "faster"
        diff_percent = abs(diff_percent)
    return diff_percent, status


def main():
    results: Dict[int, Optional[float]] = {}
    successful_runs = []
    failed_runs = []
    
    # Load mathematical predictions
    predictions = load_mathematical_predictions()
    
    print("Starting test runs...")
    print("-" * 50)
    
    for schedule_num in range(1, NUM_SCHEDULES + 1):
        print(f"Running schedule {schedule_num}/{NUM_SCHEDULES}...")
        avg_time = run_command(schedule_num)
        results[schedule_num] = avg_time
        
        if avg_time is not None:
            successful_runs.append(schedule_num)
            predicted = predictions.get(schedule_num)
            if predicted:
                diff_percent, status = calculate_prediction_accuracy(avg_time, predicted)
                print(f"Schedule {schedule_num} completed: {avg_time:.2f} ms")
                print(f"  → Predicted: {predicted:.2f} ms")
                print(f"  → {diff_percent:.1f}% {status} than predicted")
            else:
                print(f"Schedule {schedule_num} completed: {avg_time:.2f} ms (no prediction available)")
        else:
            failed_runs.append(schedule_num)
            print(f"Schedule {schedule_num} failed")
        
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
            diff_percent, status = calculate_prediction_accuracy(actual, predicted)
            prediction_accuracy.append(diff_percent)
            print(f"Schedule {schedule}:")
            print(f"  Actual: {actual:.2f} ms")
            print(f"  Predicted: {predicted:.2f} ms")
            print(f"  {diff_percent:.1f}% {status} than predicted")
    
    print("\nFailed runs:")
    for schedule in failed_runs:
        print(f"Schedule {schedule}")
    
    if successful_runs:
        best_schedule = min(successful_runs, key=lambda x: results[x])
        worst_schedule = max(successful_runs, key=lambda x: results[x])
        avg_time = sum(results[s] for s in successful_runs) / len(successful_runs)
        
        print("\nPerformance Summary:")
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
        
        if prediction_accuracy:
            avg_prediction_error = sum(prediction_accuracy) / len(prediction_accuracy)
            max_prediction_error = max(prediction_accuracy)
            min_prediction_error = min(prediction_accuracy)
            print("\nPrediction Accuracy Summary:")
            print(f"Average prediction error: {avg_prediction_error:.1f}%")
            print(f"Maximum prediction error: {max_prediction_error:.1f}%")
            print(f"Minimum prediction error: {min_prediction_error:.1f}%")


if __name__ == "__main__":
    main()
