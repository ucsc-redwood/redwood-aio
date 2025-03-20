#!/usr/bin/env python3

import json
import sys
import os
from typing import Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class ParsedRunName:
    backend: str
    application: str
    stage: int
    core_type: Optional[str]
    num_threads: Optional[int]


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse benchmark filename to extract metadata.

    Args:
        filename: Filename like 'BM_CifarDense_OMP_3A021JEHN02756.json'

    Returns:
        Tuple of (application_name, backend, device_name)

    Raises:
        ValueError: If filename doesn't match expected format
    """
    base = os.path.basename(filename)
    root, _ = os.path.splitext(base)
    parts = root.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    return parts[1], parts[2], parts[3]


def parse_run_name(line: str) -> Optional[ParsedRunName]:
    """
    Parses a single line of the form:
        {Backend}_{Application}/{StageInfo}[/NumThreads]

    Where:
      - Backend ∈ {OMP, CUDA, VK}
      - Application ∈ {CifarDense, CifarSparse, Tree}
      - StageInfo is either 'Baseline' or 'StageN' (N in 1..9),
        optionally with '_little', '_medium', or '_big'.
      - num_threads is an integer in the third segment (if present).
      - Any line containing 'std' is ignored (returns None).

    Returns:
        ParsedRunName if the line is valid, otherwise None.
    """
    # If line contains 'std', ignore it
    if "std" in line:
        return None

    segments = line.split("/")
    if len(segments) < 2:
        return None  # Invalid format

    # The first segment should be "Backend_Application"
    first_segment = segments[0].split("_", 1)
    if len(first_segment) != 2:
        return None
    backend, application = first_segment

    second_segment = segments[1]
    stage = None
    core_type = None

    # Handle Baseline => stage = 0
    if second_segment == "Baseline":
        stage = 0
    else:
        # Should match "Stage{N}" or "Stage{N}_{core_type}"
        # N must be a digit 1..9, and core_type ∈ {little, medium, big}
        match = re.match(r"Stage(\d)(?:_(little|medium|big))?$", second_segment)
        if not match:
            return None
        stage = int(match.group(1))
        if match.group(2):
            core_type = match.group(2)

    num_threads = None
    # If there's a third segment, parse out the leading integer
    if len(segments) > 2:
        third_segment = segments[2]
        m = re.match(r"(\d+)", third_segment)
        if m:
            num_threads = int(m.group(1))

    return ParsedRunName(
        backend=backend,
        application=application,
        stage=stage,
        core_type=core_type,
        num_threads=num_threads,
    )


def pritty_print_entry(entry):
    """Pretty print a benchmark entry showing just run_name and real_time."""
    # print(f"Run: {entry['run_name']}")
    # print(f"Real time: {entry['real_time']:.2f} {entry['time_unit']}\n")
    run_name = parse_run_name(entry["run_name"])
    print(run_name)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <path_to_google_benchmark_json>")
        sys.exit(1)

    json_file = sys.argv[1]
    device_name = sys.argv[2]

    with open(json_file, "r") as f:
        data = json.load(f)

    # Ensure the JSON has the "benchmarks" key
    if "benchmarks" not in data:
        print("Error: Invalid Google Benchmark JSON file (no 'benchmarks' key).")
        sys.exit(1)

    # Filter and print only those entries whose aggregate_name is "mean"
    for benchmark in data["benchmarks"]:
        if benchmark.get("aggregate_name") == "mean":
            pritty_print_entry(benchmark)

if __name__ == "__main__":
    main()
