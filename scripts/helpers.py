import argparse
import subprocess
import sys
import os
from typing import List, Dict

# hardcoded paths
RAW_BENCHMARK_PATH = "data/raw-benchmarks"
RAW_LOGS_PATH = "data/logs"
GENERATED_SCHEDULES_PATH = "data/generated-schedules"

DB_PATH = "data/benchmark_results.json"


ALL_DEVICES: List[str] = ["3A021JEHN02756", "9b034f1b", "ce0717178d7758b00b7e"]
ALL_APPLICATIONS: List[str] = ["tree", "cifar-dense", "cifar-sparse"]

# Map application names to their canonical form
APPLICATION_NAME_MAP: Dict[str, str] = {
    "tree": "Tree",
    "cifar-sparse": "CifarSparse",
    "cifar-dense": "CifarDense",
}


def run_command(cmd: str) -> None:
    """Run a shell command and exit if it fails."""
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: Command failed: {cmd}", file=sys.stderr)
        sys.exit(1)


def interactive_select(options: List[str], option_type: str) -> List[str]:
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


def parse_schedule_range(range_str: str) -> set:
    """Parse schedule range string into a set of schedule IDs.

    Accepts formats:
    - Single number: "1"
    - Comma-separated: "1,3,5"
    - Range: "1-5"
    - Mixed: "1-3,5,7-9"
    """
    if not range_str:
        return set()

    schedule_ids = set()
    parts = range_str.split(",")

    for part in parts:
        if "-" in part:
            # Handle range
            try:
                start, end = map(int, part.split("-"))
                schedule_ids.update(range(start, end + 1))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid range format: {part}. Expected format: start-end"
                )
        else:
            # Handle single number
            try:
                schedule_ids.add(int(part))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid schedule ID: {part}. Must be an integer"
                )

    return schedule_ids


def get_num_schedules(device: str, application_name: str) -> int:
    """Get the number of schedules for a given device and application.

    By looking at the files in the `GENERATED_SCHEDULES_PATH` directory.
    We expect the files to be in the format `<device>_<application>_schedule_<id>.json`

    """
    schedules_dir = GENERATED_SCHEDULES_PATH
    num_schedules = 0

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
