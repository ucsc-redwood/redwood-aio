#!/usr/bin/env python3

import re
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import argparse


@dataclass
class ScheduleResult:
    device: str
    app_name: str
    schedule_id: str
    max_chunk_time: float
    avg_time_per_task: float
    difference_percentage: float


def parse_benchmark_output(output_text: str) -> List[Tuple[str, str, str, float]]:
    """
    Parse benchmark output and extract device, app name, schedule ID and average time per task.

    Args:
        output_text (str): The benchmark output text

    Returns:
        List[Tuple[str, str, str, float]]: List of tuples containing (device, app_name, schedule_id, avg_time_per_task)
    """
    results = []

    # Skip the header lines
    lines = output_text.strip().split("\n")

    for line in lines:
        # Extract device, app name, schedule ID and avg_time_per_task using regex
        match = re.match(
            r"([A-Z0-9]+)_(\w+)_schedule_(\d+)/iterations:\d+\s+\d+\.\d+\s+ms\s+\d+\.\d+\s+ms\s+\d+\s+avg_time_per_task=(\d+\.\d+)",
            line,
        )
        if match:
            device = match.group(1)
            app_name = match.group(2)
            schedule_id = match.group(3)
            avg_time = float(match.group(4))
            results.append((device, app_name, schedule_id, avg_time))

    return results


def read_schedule_file(schedule_path: str) -> Dict:
    """
    Read a schedule file and return its contents.

    Args:
        schedule_path (str): Path to the schedule file

    Returns:
        Dict: Schedule file contents
    """
    with open(schedule_path, "r") as f:
        return json.load(f)


def find_schedule_file(
    device: str, app_name: str, schedule_id: str, schedule_root: str
) -> str:
    """
    Find the path to a schedule file based on device, app name, and schedule ID.

    Args:
        device (str): Device name (e.g., "jetson")
        app_name (str): Application name (e.g., "Tree")
        schedule_id (str): Schedule ID (e.g., "001")
        schedule_root (str): Root directory for schedule files

    Returns:
        str: Path to the schedule file, or None if not found
    """
    # Construct the expected path pattern
    schedule_path = (
        Path(schedule_root) / device / app_name / f"schedule_{schedule_id}.json"
    )

    # Check if the file exists
    if schedule_path.exists():
        return str(schedule_path)

    # If the exact path doesn't exist, try to find it with a glob pattern
    potential_files = list(
        Path(schedule_root).glob(
            f"**/*{device}*{app_name}*schedule_{schedule_id}*.json"
        )
    )
    if potential_files:
        return str(potential_files[0])

    return None


def compare_benchmark_with_schedule(
    device: str, app_name: str, schedule_id: str, avg_time: float, schedule_root: str
) -> ScheduleResult:
    """
    Compare benchmark result with schedule file data.

    Args:
        device (str): Device name
        app_name (str): Application name
        schedule_id (str): Schedule ID
        avg_time (float): Average time per task from benchmark
        schedule_root (str): Root directory for schedule files

    Returns:
        ScheduleResult: Comparison result, or None if schedule file not found
    """
    # Find the schedule file
    schedule_file_path = find_schedule_file(
        device, app_name, schedule_id, schedule_root
    )

    if not schedule_file_path:
        print(
            f"Warning: Schedule file not found for {device}_{app_name}_schedule_{schedule_id}"
        )
        return None

    # Read the schedule file
    schedule_data = read_schedule_file(schedule_file_path)

    # Extract the max_chunk_time
    max_chunk_time = schedule_data.get("max_chunk_time")
    if max_chunk_time is None:
        print(
            f"Warning: max_chunk_time not found in schedule file {schedule_file_path}"
        )
        return None

    # Calculate difference
    difference = abs(max_chunk_time - avg_time)
    difference_percentage = (difference / max_chunk_time) * 100

    return ScheduleResult(
        device=device,
        app_name=app_name,
        schedule_id=schedule_id,
        max_chunk_time=max_chunk_time,
        avg_time_per_task=avg_time,
        difference_percentage=difference_percentage,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark output and compare with schedule files"
    )
    parser.add_argument("benchmark_file", help="Path to the benchmark output file")
    parser.add_argument(
        "--schedule-root",
        default="data/schedule_files_v2",
        help="Root directory for schedule files",
    )
    args = parser.parse_args()

    # Read benchmark output from file
    with open(args.benchmark_file, "r") as f:
        benchmark_output = f.read()

    # Parse benchmark results
    benchmark_results = parse_benchmark_output(benchmark_output)

    if not benchmark_results:
        print(f"Warning: No benchmark results found in {args.benchmark_file}")
        return

    print(f"Found {len(benchmark_results)} benchmark results")

    # Compare results for each schedule
    print("\nSchedule Comparison Results:")
    print("-" * 100)
    print(
        f"{'Device':<10} {'App':<10} {'Schedule ID':<12} {'Max Chunk Time':<15} {'Avg Time/Task':<15} {'Difference %':<15}"
    )
    print("-" * 100)

    comparison_results = []
    for device, app_name, schedule_id, avg_time in benchmark_results:
        result = compare_benchmark_with_schedule(
            device, app_name, schedule_id, avg_time, args.schedule_root
        )
        if result:
            comparison_results.append(result)

    if not comparison_results:
        print("Warning: No schedule files found for comparison")
        return

    # Sort results by difference percentage (ascending)
    sorted_results = sorted(comparison_results, key=lambda r: r.difference_percentage)

    for result in sorted_results:
        print(
            f"{result.device:<10} {result.app_name:<10} {result.schedule_id:<12} "
            f"{result.max_chunk_time:<15.2f} {result.avg_time_per_task:<15.2f} {result.difference_percentage:<15.2f}%"
        )

    print("-" * 100)


if __name__ == "__main__":
    main()
