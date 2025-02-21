#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse

ALL_DEVICES = ["3A021JEHN02756", "9b034f1b", "ce0717178d7758b00b7e"]
ALL_APPLICATIONS = ["tree", "cifar-dense", "cifar-sparse"]

# Directory to save pulled results
OUTPUT_DIR = "./data/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_command(cmd):
    """Run a shell command and exit if it fails."""
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: Command failed: {cmd}", file=sys.stderr)
        sys.exit(1)


def get_num_schedules(device, application_name):
    schedules_dir = "data/generated-schedules"
    num_schedules = 0

    # Map application names to their canonical form
    APPLICATION_NAME_MAP = {
        "tree": "Tree",
        "cifar-sparse": "CifarSparse",
        "cifar-dense": "CifarDense",
    }

    for each_file in os.listdir(schedules_dir):
        # Split filename into parts
        parts = each_file.split("_")

        # Check if this is a valid schedule file for the given device and app
        if (
            len(parts) == 4
            and parts[0] == device
            and parts[1] == APPLICATION_NAME_MAP[application_name]
            and parts[2] == "schedule"
            and parts[3].endswith(".json")
        ):
            num_schedules += 1

    return num_schedules


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
    parser = argparse.ArgumentParser(description="Collect logs from specified devices.")
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        help="Comma-separated list of device IDs to collect logs from (if not provided, interactive selection will be used).",
    )
    parser.add_argument(
        "--application-name",
        "-a",
        type=str,
        help="Name of the application to collect logs from.",
    )
    return parser.parse_args()


def log_filename(device, application_name, schedule_id):
    # Ensure application_name is a single string, not a list
    app_name = (
        application_name[0] if isinstance(application_name, list) else application_name
    )
    return f"logs-{device}-{app_name}-schedule-{schedule_id}.txt"


def obtain_a_single_log(device, application_name, schedule_id):
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
    filename = log_filename(device, application_name, schedule_id)
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

    if args.application_name:
        application_name = args.application_name
    else:
        # Take first selected application when using interactive selection
        application_name = interactive_select(ALL_APPLICATIONS, "applications")[0]

    print(f"Devices to collect logs from: {devices}")
    print(f"Application: {application_name}\n")

    num_schedules = get_num_schedules(devices[0], application_name)
    print(f"Number of schedules: {num_schedules}")

    for device in devices:
        # Step 1: Push the executable to the device
        executable_name = f"pipe-{application_name.lower()}-vk"
        push_cmd = (
            f"adb -s {device} push "
            f"./build/android/arm64-v8a/release/{executable_name} "
            f"/data/local/tmp/{executable_name}"
        )
        run_command(push_cmd)

        for schedule_id in range(1, num_schedules + 1):
            obtain_a_single_log(device, application_name, schedule_id)


if __name__ == "__main__":
    main()
