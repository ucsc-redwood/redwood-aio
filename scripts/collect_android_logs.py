#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

from helpers import (
    ALL_DEVICES,
    ALL_APPLICATIONS,
    run_command,
    interactive_select,
    parse_schedule_range,
    get_num_schedules,
    RAW_LOGS_PATH,
    select_schedules,
)


def get_log_filename(device: str, app: str, schedule_id: int) -> str:
    """Generate log filename for given parameters."""
    app_name = app[0] if isinstance(app, list) else app
    return f"logs-{device}-{app_name}-schedule-{schedule_id}.txt"


def collect_log(device: str, app: str, schedule_id: int, output_dir: Path) -> None:
    """Collect a single log file from device."""
    exe_name = f"pipe-{app.lower()}-vk"
    device_log = "/data/local/tmp/logs.txt"

    print(f"\n--- Collecting schedule {schedule_id} from {device} ---")

    # Run executable and collect logs
    run_command(
        f"adb -s {device} shell /data/local/tmp/{exe_name} "
        f"--device {device} --schedule {schedule_id} --log-level debug"
    )

    # Pull and cleanup logs
    output_file = output_dir / get_log_filename(device, app, schedule_id)
    run_command(f"adb -s {device} pull {device_log} {output_file}", hide_output=True)
    run_command(f"adb -s {device} shell rm {device_log}", hide_output=True)


def main():
    parser = argparse.ArgumentParser(description="Collect logs from Android devices")
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

    print(f"\nCollecting logs for {app} on device {device}")

    # Setup output directory
    output_dir = Path(RAW_LOGS_PATH)
    output_dir.mkdir(exist_ok=True)

    # Build executable if needed
    exe_name = f"pipe-{app.lower()}-vk"
    exe_path = Path(f"./build/android/arm64-v8a/release/{exe_name}")
    if not exe_path.exists():
        print(f"Building {exe_name}...")
        run_command(f"xmake b {exe_name}")

    # Get schedules for this device-app pair
    schedule_ids = select_schedules(device, app, args.schedules)
    print(f"Will collect {len(schedule_ids)} schedules")

    # Push executable
    run_command(
        f"adb -s {device} push {exe_path} /data/local/tmp/{exe_name}",
        hide_output=True,
    )

    # Collect logs
    for schedule_id in sorted(schedule_ids):
        collect_log(device, app, schedule_id, output_dir)


if __name__ == "__main__":
    main()
