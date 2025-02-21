#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse

from helpers import (
    ALL_DEVICES,
    ALL_APPLICATIONS,
    run_command,
    interactive_select,
    parse_schedule_range,
    get_num_schedules,
    RAW_LOGS_PATH,
)

# Directory to save pulled results
OUTPUT_DIR = RAW_LOGS_PATH
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect logs from specified devices.")
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        help="Comma-separated list of device IDs to collect logs from (if not provided, interactive selection will be used).",
    )
    parser.add_argument(
        "--application",
        "-a",
        type=str,
        help="Name of the application to collect logs from.",
    )
    parser.add_argument(
        "--schedules",
        "-s",
        type=str,
        help="Schedule IDs to collect (e.g., '1-5' or '1,3,5' or '1-3,5,7-9')",
    )
    return parser.parse_args()


def gen_log_filename(device: str, application_name: str, schedule_id: int) -> str:
    # Ensure application_name is a single string, not a list
    app_name = (
        application_name[0] if isinstance(application_name, list) else application_name
    )
    return f"logs-{device}-{app_name}-schedule-{schedule_id}.txt"


def obtain_a_single_log(device: str, application_name: str, schedule_id: int) -> None:
    executable_name = f"pipe-{application_name.lower()}-vk"

    # Step 2: Run the executable on the device
    run_exe_cmd = (
        f"adb -s {device} shell /data/local/tmp/{executable_name} "
        f"--device {device} "
        f"--schedule {schedule_id} "
        f"--log-level debug"
    )
    run_command(run_exe_cmd)

    # Step 3: Pull the logs from the device
    print(f"\n--- Collecting logs from device: {device} ---")
    filename = gen_log_filename(device, application_name, schedule_id)
    pull_cmd = f"adb -s {device} pull /data/local/tmp/logs.txt {OUTPUT_DIR}/{filename}"
    run_command(pull_cmd)

    # Step 4: Remove the logs from the device
    rm_cmd = f"adb -s {device} shell rm /data/local/tmp/logs.txt"
    run_command(rm_cmd)


def main():
    args = parse_args()

    # Use command-line arguments if provided; otherwise prompt interactively.
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",")]
    else:
        devices = interactive_select(ALL_DEVICES, "devices")

    if args.application:
        application_name = args.application
    else:
        # Take first selected application when using interactive selection
        application_name = interactive_select(ALL_APPLICATIONS, "applications")[0]

    print(f"Devices to collect logs from: {devices}")
    print(f"Application: {application_name}\n")

    # Get the number of schedules
    num_schedules = get_num_schedules(devices[0], application_name)
    print(f"Number of schedules: {num_schedules}")

    # Parse schedule range if provided
    schedule_ids = parse_schedule_range(args.schedules)
    if not schedule_ids:
        schedule_ids = set(range(1, num_schedules + 1))
    else:
        # Validate schedule IDs are within range
        invalid_ids = [sid for sid in schedule_ids if sid < 1 or sid > num_schedules]
        if invalid_ids:
            print(
                f"Error: Invalid schedule IDs {invalid_ids}. Must be between 1 and {num_schedules}"
            )
            sys.exit(1)

    # Step 0: Build the executable
    build_cmd = f"xmake b pipe-{application_name.lower()}-vk"
    run_command(build_cmd)

    for device in devices:
        # Step 1: Push the executable to the device
        executable_name = f"pipe-{application_name.lower()}-vk"
        push_cmd = (
            f"adb -s {device} push "
            f"./build/android/arm64-v8a/release/{executable_name} "
            f"/data/local/tmp/{executable_name}"
        )
        run_command(push_cmd)

        # Step 2: Obtain the logs
        for schedule_id in sorted(schedule_ids):
            obtain_a_single_log(device, application_name, schedule_id)


if __name__ == "__main__":
    main()
