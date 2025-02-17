#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse

# Full list of benchmarks and devices
ALL_BENCHMARKS = [
    "bm-cifar-dense-omp",
    "bm-cifar-dense-vk",
    "bm-cifar-sparse-omp",
    "bm-cifar-sparse-vk",
    "bm-tree-omp",
    "bm-tree-vk",
]

ALL_DEVICES = ["3A021JEHN02756", "9b034f1b", "ce0717178d7758b00b7e"]

# Directory to save pulled results
OUTPUT_DIR = "./data/raw_bm_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_command(cmd):
    """Run a shell command and exit if it fails."""
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: Command failed: {cmd}", file=sys.stderr)
        sys.exit(1)


def get_json_filename(benchmark, device):
    """
    Compute the expected JSON file name.

    For benchmarks with four parts (e.g. bm-cifar-dense-omp), the file name is:
      BM_<Part2><Part3>_<Part4>_<device>.json
    For benchmarks with three parts (e.g. bm-tree-omp), the file name is:
      BM_<Part2>_<Part3>_<device>.json
    """
    parts = benchmark.split("-")
    if len(parts) == 4:
        return f"BM_{parts[1].capitalize()}{parts[2].capitalize()}_{parts[3].upper()}_{device}.json"
    elif len(parts) == 3:
        return f"BM_{parts[1].capitalize()}_{parts[2].upper()}_{device}.json"
    else:
        return f"{benchmark}_{device}.json"


def interactive_select(options, option_type):
    """
    Prompt the user to select options by number.

    options: list of strings (benchmarks or devices)
    option_type: string to display (e.g. "benchmarks" or "devices")
    Returns the list of selected options.
    """
    print(
        f"Select {option_type} to run (enter numbers separated by commas, or press Enter for all):"
    )
    for i, option in enumerate(options):
        print(f"{i+1}: {option}")
    selection = input("Enter your choices: ").strip()
    if not selection:
        return options
    try:
        indices = [int(x) - 1 for x in selection.split(",") if x.strip().isdigit()]
        selected = [options[i] for i in indices if 0 <= i < len(options)]
        return selected
    except Exception:
        print("Invalid selection, using all options.")
        return options


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on specified devices. Use interactive selection if not provided."
    )
    parser.add_argument(
        "--benchmarks",
        "-b",
        type=str,
        help="Comma-separated list of benchmarks to run (if not provided, interactive selection will be used).",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        help="Comma-separated list of device IDs to run (if not provided, interactive selection will be used).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use command-line arguments if provided; otherwise prompt interactively.
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    else:
        benchmarks = interactive_select(ALL_BENCHMARKS, "benchmarks")

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",")]
    else:
        devices = interactive_select(ALL_DEVICES, "devices")

    print(f"\nBenchmarks to run: {benchmarks}")
    print(f"Devices to run on: {devices}\n")

    for benchmark in benchmarks:
        print(f"\n=== Processing benchmark: {benchmark} ===")
        # Step 1: Build the executable.
        build_cmd = f"xmake b {benchmark}"
        run_command(build_cmd)

        for device in devices:
            print(f"\n--- Running on device: {device} ---")
            # Step 2: Push the executable to the device.
            push_cmd = (
                f"adb -s {device} push "
                f"./build/android/arm64-v8a/release/{benchmark} "
                f"/data/local/tmp/{benchmark}"
            )
            run_command(push_cmd)

            # Step 3: Run the executable on the device.
            run_exe_cmd = (
                f"adb -s {device} shell /data/local/tmp/{benchmark} "
                f"--device {device}"
            )
            run_command(run_exe_cmd)

            # Step 4: Pull the output JSON file.
            json_filename = get_json_filename(benchmark, device)
            pull_cmd = (
                f"adb -s {device} pull "
                f"/data/local/tmp/{json_filename} "
                f"{OUTPUT_DIR}/"
            )
            run_command(pull_cmd)


if __name__ == "__main__":
    main()
