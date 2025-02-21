#!/usr/bin/env python3
from pathlib import Path
import argparse

from helpers import (
    ALL_DEVICES,
    ALL_BENCHMARKS,
    interactive_select,
    RAW_BENCHMARK_PATH,
    run_command,
)


def get_json_filename(benchmark: str, device: str) -> str:
    """Generate benchmark JSON filename based on benchmark name and device."""
    parts = benchmark.split("-")
    if len(parts) == 4:
        return f"BM_{parts[1].capitalize()}{parts[2].capitalize()}_{parts[3].upper()}_{device}.json"
    return f"BM_{parts[1].capitalize()}_{parts[2].upper()}_{device}.json"


def run_benchmark(benchmark: str, device: str, output_dir: Path) -> None:
    """Run a single benchmark on a device and collect results."""
    print(f"\n--- Running {benchmark} on device: {device} ---")

    # Push executable to device
    exe_path = f"./build/android/arm64-v8a/release/{benchmark}"
    device_path = f"/data/local/tmp/{benchmark}"
    run_command(f"adb -s {device} push {exe_path} {device_path}")

    # Run benchmark
    run_command(f"adb -s {device} shell {device_path} --device {device}")

    # Pull results
    json_file = get_json_filename(benchmark, device)
    run_command(f"adb -s {device} pull /data/local/tmp/{json_file} {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run Android benchmarks")
    parser.add_argument("--benchmarks", "-b", help="Comma-separated benchmarks")
    parser.add_argument("--devices", "-d", help="Comma-separated device IDs")
    args = parser.parse_args()

    # Select benchmarks and devices
    benchmarks = (
        args.benchmarks.split(",")
        if args.benchmarks
        else interactive_select(ALL_BENCHMARKS, "benchmarks")
    )
    devices = (
        args.devices.split(",")
        if args.devices
        else interactive_select(ALL_DEVICES, "devices")
    )

    print(f"\nRunning benchmarks: {benchmarks}")
    print(f"On devices: {devices}\n")

    output_dir = Path(RAW_BENCHMARK_PATH)
    output_dir.mkdir(exist_ok=True)

    for benchmark in benchmarks:
        print(f"\n=== Processing benchmark: {benchmark} ===")
        run_command(f"xmake b {benchmark}")

        for device in devices:
            run_benchmark(benchmark, device, output_dir)


if __name__ == "__main__":
    main()
