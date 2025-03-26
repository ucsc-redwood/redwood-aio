#!/usr/bin/env python3

import re
import json
import os
import glob
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
    is_better: bool  # True if real measure is better than expected


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
        if not line.strip():
            continue

        # Extract components from the beginning of the line
        parts = line.split("/")
        if len(parts) < 2:
            continue

        full_id = parts[0]
        id_components = full_id.split("_")

        if len(id_components) < 4:
            print(f"Warning: Could not parse components from line: {line}")
            continue

        device = id_components[0]
        app_name = id_components[1]
        # Assuming 'schedule' is always part of the component
        schedule_id = id_components[3]

        # Extract avg_time_per_task from the end of the line
        avg_time_match = re.search(r"avg_time_per_task=(\d+\.\d+)", line)
        if avg_time_match:
            avg_time = float(avg_time_match.group(1))
            results.append((device, app_name, schedule_id, avg_time))
        else:
            print(f"Warning: Could not find avg_time_per_task in line: {line}")

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
    # Lower time is better, so if avg_time < max_chunk_time, performance is better
    is_better = avg_time < max_chunk_time

    # Calculate signed percentage difference
    # Positive percentage means faster than expected (better)
    # Negative percentage means slower than expected (worse)
    difference = max_chunk_time - avg_time
    difference_percentage = (difference / max_chunk_time) * 100

    return ScheduleResult(
        device=device,
        app_name=app_name,
        schedule_id=schedule_id,
        max_chunk_time=max_chunk_time,
        avg_time_per_task=avg_time,
        difference_percentage=difference_percentage,
        is_better=is_better,
    )


def process_benchmark_file(
    benchmark_file: str, schedule_root: str
) -> List[ScheduleResult]:
    """
    Process a single benchmark file and return comparison results.

    Args:
        benchmark_file (str): Path to the benchmark file
        schedule_root (str): Root directory for schedule files

    Returns:
        List[ScheduleResult]: List of comparison results
    """
    print(f"\nProcessing file: {benchmark_file}")

    # Read benchmark output from file
    with open(benchmark_file, "r") as f:
        benchmark_output = f.read()

    # Parse benchmark results
    benchmark_results = parse_benchmark_output(benchmark_output)

    if not benchmark_results:
        print(f"Warning: No benchmark results found in {benchmark_file}")
        return []

    print(f"Found {len(benchmark_results)} benchmark results")

    comparison_results = []
    for device, app_name, schedule_id, avg_time in benchmark_results:
        result = compare_benchmark_with_schedule(
            device, app_name, schedule_id, avg_time, schedule_root
        )
        if result:
            comparison_results.append(result)

    if not comparison_results:
        print(f"Warning: No schedule files found for comparison in {benchmark_file}")

    return comparison_results


def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark output and compare with schedule files"
    )
    parser.add_argument(
        "benchmark_path",
        help="Path to a benchmark file or directory containing benchmark files",
    )
    parser.add_argument(
        "--schedule-root",
        default="data/schedule_files_v2",
        help="Root directory for schedule files",
    )
    parser.add_argument(
        "--sort-by",
        choices=["difference", "max_chunk_time", "avg_time"],
        default="difference",
        help="Sort results by this metric",
    )
    args = parser.parse_args()

    # Check if the path is a directory or a file
    if os.path.isdir(args.benchmark_path):
        # Process all .txt files in the directory
        benchmark_files = glob.glob(os.path.join(args.benchmark_path, "*.txt"))
        if not benchmark_files:
            print(f"No .txt files found in directory: {args.benchmark_path}")
            return
        print(f"Found {len(benchmark_files)} benchmark files to process")
    else:
        # Process a single file
        if not args.benchmark_path.endswith(".txt"):
            print("Warning: Benchmark file should be a .txt file")
        benchmark_files = [args.benchmark_path]

    all_results = []
    for benchmark_file in benchmark_files:
        results = process_benchmark_file(benchmark_file, args.schedule_root)
        all_results.extend(results)

    if not all_results:
        print("No comparison results found across all files")
        return

    # Sort results based on the specified criterion
    if args.sort_by == "difference":
        sorted_results = sorted(
            all_results, key=lambda r: r.difference_percentage, reverse=True
        )
    elif args.sort_by == "max_chunk_time":
        sorted_results = sorted(all_results, key=lambda r: r.max_chunk_time)
    elif args.sort_by == "avg_time":
        sorted_results = sorted(all_results, key=lambda r: r.avg_time_per_task)

    # Print the combined results
    print("\nCombined Schedule Comparison Results:")
    print("-" * 130)
    print(
        f"{'Device':<15} {'App':<15} {'Schedule ID':<12} {'Max Chunk Time':<15} {'Avg Time/Task':<15} {'Difference %':<15} {'Status':<10}"
    )
    print("-" * 130)

    for result in sorted_results:
        # Format the difference percentage with sign (+ for better, - for worse)
        sign = "+" if result.is_better else "-"
        formatted_diff = f"{sign}{abs(result.difference_percentage):.2f}%"

        # Determine status based on performance
        status = "BETTER" if result.is_better else "WORSE"

        print(
            f"{result.device:<15} {result.app_name:<15} {result.schedule_id:<12} "
            f"{result.max_chunk_time:<15.2f} {result.avg_time_per_task:<15.2f} "
            f"{formatted_diff:<15} {status:<10}"
        )

    print("-" * 130)

    # Print summary statistics
    better_count = sum(1 for r in all_results if r.is_better)
    total_count = len(all_results)
    if total_count > 0:
        better_percentage = (better_count / total_count) * 100
        print(
            f"\nSummary: {better_count}/{total_count} ({better_percentage:.2f}%) of results are better than expected"
        )


if __name__ == "__main__":
    main()
