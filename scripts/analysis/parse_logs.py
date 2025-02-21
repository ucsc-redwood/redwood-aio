import re
from collections import defaultdict
import json
import os
import glob
import argparse
from typing import Dict, List, Set
from colorama import init, Fore, Style

# Initialize colorama
init()

# Color constants
COLOR_ERROR = Fore.RED
COLOR_WARNING = Fore.YELLOW
COLOR_PASS = Fore.GREEN
COLOR_INFO = Fore.CYAN
COLOR_HEADER = Fore.BLUE + Style.BRIGHT
COLOR_RESET = Style.RESET_ALL


def format_status(status: str) -> str:
    """Format a status string with appropriate color."""
    if status.lower() == "pass":
        return f"{COLOR_PASS}{status}{COLOR_RESET}"
    elif status.lower() == "fail":
        return f"{COLOR_ERROR}{status}{COLOR_RESET}"
    elif status.lower() == "error":
        return f"{COLOR_ERROR}{status}{COLOR_RESET}"
    return status


def print_header(text: str):
    """Print a section header."""
    print(f"\n{COLOR_HEADER}=== {text} ==={COLOR_RESET}")


def print_error(text: str):
    """Print an error message."""
    print(f"{COLOR_ERROR}ERROR: {text}{COLOR_RESET}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{COLOR_WARNING}WARNING: {text}{COLOR_RESET}")


def print_pass(text: str):
    """Print a pass message."""
    print(f"{COLOR_PASS}PASS: {text}{COLOR_RESET}")


def print_info(text: str):
    """Print an info message."""
    print(f"{COLOR_INFO}{text}{COLOR_RESET}")


def parse_log_lines(lines):
    """
    Parse lines from the custom log format and return a list of dictionaries.

    Each dictionary contains:
      - backend        (str, e.g. 'omp', 'vk', 'cuda')
      - core_id        (str or None)
      - thread_idx     (str or None)
      - num_threads    (str or None)
      - stage          (str)
      - app_address    (str, e.g. '0xb4000078accf2810')
    """
    pattern = re.compile(
        r"\[.*?\] \[.*?\] \[debug\] \[(?P<backend>\w+)\]"
        r"(?:\[Core:\s*(?P<core_id>\d+)\])?"
        r"(?:\[Thread:\s*(?P<thread_idx>\d+)\/(?P<num_threads>\d+)\])?"
        r"\s*\[Stage:\s*(?P<stage>\d+)\]"
        r"\s*\[App:\s*(?P<app_address>0x[0-9a-fA-F]+)\]"
    )

    results = []
    for line in lines:
        match = pattern.search(line)
        if match:
            results.append(
                {
                    "backend": match.group("backend"),
                    "core_id": match.group("core_id"),
                    "thread_idx": match.group("thread_idx"),
                    "num_threads": match.group("num_threads"),
                    "stage": match.group("stage"),
                    "app_address": match.group("app_address"),
                }
            )

    return results


def verify_log(parsed_data):
    """
    1) Check there are exactly 20 unique App addresses.
    2) For each App address, check that Stage goes 1..9 in ascending order
       without skipping or going out of order.
    """
    errors_found = 0

    from collections import defaultdict

    last_stage_for_app = defaultdict(lambda: 0)
    stages_for_app = defaultdict(list)

    # Process each line
    for i, row in enumerate(parsed_data):
        app = row["app_address"]
        stage_str = row["stage"]

        if not stage_str.isdigit():
            print_error(f"Stage is not numeric on line {i}: {stage_str}")
            errors_found += 1
            continue

        stage = int(stage_str)
        if stage < last_stage_for_app[app]:
            print_error(
                f"Out-of-order stage for App {app} at line {i}. "
                f"Got stage {stage} after {last_stage_for_app[app]}."
            )
            errors_found += 1
        last_stage_for_app[app] = stage

        # Keep them for final checks
        stages_for_app[app].append((stage, row["backend"]))

    # Check #1: Exactly 20 distinct apps
    all_apps = list(stages_for_app.keys())
    if len(all_apps) != 20:
        print_error(f"Expected 20 distinct app addresses, but found {len(all_apps)}.")
        print_error(f"Apps found: {all_apps}")
        errors_found += 1
    else:
        print_pass("Exactly 20 distinct app addresses found.")

    # Check #2: Each app must get from 1..9
    for app, stage_tuples in stages_for_app.items():
        # Already in order, but let's confirm the final stage is >=9
        stage_numbers = [s for (s, _) in stage_tuples]
        if stage_numbers[-1] < 9:
            print(f"ERROR: App {app} only reached stage {stage_numbers[-1]}, not 9.")
            errors_found += 1
        else:
            needed = set(range(1, 10))
            have = set(stage_numbers)
            missing = needed - have
            if missing:
                print(f"ERROR: App {app} is missing stage(s) {sorted(missing)}.")
                errors_found += 1
            else:
                # Double-check ascending
                if any(
                    stage_numbers[i] > stage_numbers[i + 1]
                    for i in range(len(stage_numbers) - 1)
                ):
                    print(f"ERROR: App {app} has out-of-order stages: {stage_numbers}")
                    errors_found += 1
                else:
                    print(f"PASS: App {app} has stages 1..9 in ascending order.")

    print("\n--- FINAL REPORT ---")
    for app in sorted(stages_for_app.keys()):
        desc = ", ".join(f"{s}({b})" for (s, b) in stages_for_app[app])
        print(f"App: {app} => {desc}")

    if errors_found == 0:
        print("\nPASS: All app-stage checks completed successfully.")
    else:
        print(f"\nFAIL: Found {errors_found} errors in stage ordering or app count.")


def load_schedule_from_json(json_path):
    """
    Load schedule config from JSON file.
    Return a dict with 'chunks', plus schedule/device metadata.
    Each chunk will have a new 'stage_set' field (set of ints).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    schedule = data["schedule"]
    for chunk in schedule["chunks"]:
        chunk["stage_set"] = set(chunk["stages"])
    return schedule


def verify_log_against_schedule(parsed_logs, schedule):
    """
    1) Determine which core_ids (for 'omp') map to which chunk (CPU).
    2) Check GPU usage if hardware='gpu' (using 'vk'/'cuda').
    3) Print final chunk->core mapping.
    4) Report overall pass/fail based on errors found.
    """
    errors_found = 0
    warnings_found = 0

    # Gather usage by core
    stages_used_by_core = defaultdict(set)  # core_id -> set of stages
    backends_used_by_core = defaultdict(set)  # core_id -> set of backends
    gpu_stages_encountered = set()  # For lines with vk/cuda & no core

    for row in parsed_logs:
        stage = int(row["stage"])
        backend = row["backend"]  # 'omp', 'vk', 'cuda'
        core_id = row["core_id"]  # may be None if not present

        if backend in ("vk", "cuda"):
            # GPU usage
            if core_id is None:
                gpu_stages_encountered.add(stage)
            else:
                stages_used_by_core[core_id].add(stage)
                backends_used_by_core[core_id].add(backend)
        else:
            # 'omp' => CPU usage
            if core_id is not None:
                stages_used_by_core[core_id].add(stage)
                backends_used_by_core[core_id].add(backend)
            else:
                print(f"WARNING: OMP line but no core_id? {row}")
                warnings_found += 1

    # Build chunk map for quick reference
    chunk_map = {}
    for chunk in schedule["chunks"]:
        chunk_name = chunk["name"]
        chunk_map[chunk_name] = {
            "hardware": chunk["hardware"],
            "threads": chunk["threads"],
            "stage_set": chunk["stage_set"],
        }

    # Attempt to assign CPU cores to CPU chunks
    chunk_cores = defaultdict(set)
    unmatched_cores = set(stages_used_by_core.keys())

    for chunk_name, info in chunk_map.items():
        hardware = info["hardware"]
        threads_req = info["threads"]
        chunk_stages = info["stage_set"]

        # GPU chunk => skip CPU matching
        if hardware.lower() == "gpu":
            continue

        # For CPU chunk, find all cores that exactly match the chunk's stage set.
        candidate_cores = []
        for cid in unmatched_cores:
            s_used = stages_used_by_core[cid]
            if s_used == chunk_stages:  # or some subset logic if needed
                candidate_cores.append(cid)

        if len(candidate_cores) == threads_req:
            # Perfect match: assign them
            chunk_cores[chunk_name] = set(candidate_cores)
            for cid in candidate_cores:
                unmatched_cores.remove(cid)
        else:
            print(
                f"WARNING: For chunk '{chunk_name}' (hardware={hardware}), "
                f"expected {threads_req} cores with stage set={chunk_stages}, "
                f"but found {len(candidate_cores)} matches."
            )
            warnings_found += 1

    # Check GPU chunks
    for chunk_name, info in chunk_map.items():
        if info["hardware"].lower() == "gpu":
            chunk_stages = info["stage_set"]
            # We expect these stages to be done by 'vk'/ 'cuda' lines
            # If no explicit core ID for GPU usage, we check `gpu_stages_encountered`
            missing_stages = chunk_stages - gpu_stages_encountered

            # Also account for GPU-labeled cores:
            # any core that used 'vk'/'cuda' might have done these stages
            # Combine them if needed:
            all_gpu_stages = set(gpu_stages_encountered)
            for cid in stages_used_by_core:
                if any(b in ("vk", "cuda") for b in backends_used_by_core[cid]):
                    all_gpu_stages |= stages_used_by_core[cid]

            # Now see if chunk_stages is fully in all_gpu_stages
            missing_stages = chunk_stages - all_gpu_stages
            if missing_stages:
                print(
                    f"ERROR: GPU chunk '{chunk_name}' has stage(s) {missing_stages} "
                    f"not found in GPU usage logs."
                )
                errors_found += 1
            else:
                print(
                    f"PASS: GPU chunk '{chunk_name}' usage looks correct for stages {chunk_stages}"
                )

    # Print final chunk->core mapping
    print("\n--- Final Chunk-Core Mapping ---")
    for chunk_name, info in chunk_map.items():
        hw = info["hardware"]
        cids = chunk_cores.get(chunk_name, set())
        if hw.lower() == "gpu":
            # GPU chunk => typically no CPU core assignment
            if cids:
                # This might be unusual if you truly expect zero CPU usage
                print(f"{chunk_name} (GPU) => Assigned cores? {cids} (Check logs!)")
            else:
                print(f"{chunk_name} (GPU) => No cores assigned (expected for GPU).")
        else:
            if cids:
                print(f"{chunk_name} => {cids}")
            else:
                print(f"{chunk_name} => No cores assigned (WARNING?)")

    if unmatched_cores:
        print(f"\nWARNING: Some cores not matched to any CPU chunk: {unmatched_cores}")
        warnings_found += 1

    # Final pass/fail reporting
    print("\n--- SCHEDULE VERIFICATION RESULTS ---")
    if errors_found == 0:
        print(f"PASS: No errors found. (Warnings: {warnings_found})")
    else:
        print(f"FAIL: Found {errors_found} error(s). (Warnings: {warnings_found})")


def map_application_name(app_name):
    """Map the application name from log filename to schedule filename format."""
    mapping = {
        "cifar-dense": "CifarDense",
        "cifar-sparse": "CifarSparse",
        "tree": "Tree",
    }
    return mapping.get(app_name, app_name)


def get_verification_result(log_path: str, verbose: bool = False) -> Dict:
    """Process a single log file and return its verification result."""
    result = {
        "log_path": log_path,
        "status": "unknown",
        "errors": 0,
        "warnings": 0,
        "schedule_path": None,
        "device": None,
        "app_name": None,
        "schedule_id": None,
        "details": "",  # Store detailed output
    }

    # Parse the log filename
    log_filename = os.path.basename(log_path)
    match = re.match(
        r"logs-([^-]+)-([^-]+(?:-[^-]+)*)-schedule-(\d+)\.txt", log_filename
    )

    if not match:
        result["status"] = "error"
        result["errors"] = 1
        return result

    device, app_name, schedule_id = match.groups()
    result.update(
        {"device": device, "app_name": app_name, "schedule_id": int(schedule_id)}
    )

    mapped_app_name = map_application_name(app_name)
    schedule_filename = (
        f"{device}_{mapped_app_name}_schedule_{int(schedule_id):03d}.json"
    )
    schedule_path = os.path.join("data/generated-schedules", schedule_filename)
    result["schedule_path"] = schedule_path

    if not os.path.exists(schedule_path):
        result["status"] = "error"
        result["errors"] = 1
        return result

    # Capture stdout to avoid cluttering during processing
    import io
    import sys

    stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        with open(log_path, "r") as f:
            lines = f.read().splitlines()

        parsed_data = parse_log_lines(lines)
        verify_log(parsed_data)
        schedule = load_schedule_from_json(schedule_path)
        verify_log_against_schedule(parsed_data, schedule)

        # Extract error and warning counts from output
        output = captured_output.getvalue()
        result["details"] = output  # Store the full output
        result["errors"] = output.count("ERROR:")
        result["warnings"] = output.count("WARNING:")
        result["status"] = "pass" if result["errors"] == 0 else "fail"

    except Exception as e:
        result["status"] = "error"
        result["errors"] = 1
        result["details"] = f"Exception occurred: {str(e)}"
    finally:
        sys.stdout = stdout

        # If verbose and failed/error, print details immediately
        if verbose and result["status"] != "pass":
            print("\nDetailed verification output:")
            print(result["details"])

    return result


def print_summary(results: List[Dict], verbose: bool = False):
    """Print a summary of all verification results."""
    print_header("FINAL VERIFICATION SUMMARY")

    # Group results by device and application
    by_device: Dict[str, Dict[str, List[Dict]]] = {}
    for r in results:
        device = r["device"]
        app = r["app_name"]
        if device not in by_device:
            by_device[device] = {}
        if app not in by_device[device]:
            by_device[device][app] = []
        by_device[device][app].append(r)

    # Print summary by device and application
    total_passed = 0
    total_failed = 0
    total_error = 0

    for device in sorted(by_device.keys()):
        print(f"\n{COLOR_HEADER}Device: {device}{COLOR_RESET}")
        for app in sorted(by_device[device].keys()):
            results = by_device[device][app]
            passed = sum(1 for r in results if r["status"] == "pass")
            failed = sum(1 for r in results if r["status"] == "fail")
            error = sum(1 for r in results if r["status"] == "error")

            total_passed += passed
            total_failed += failed
            total_error += error

            print(f"\n  {COLOR_INFO}{app}:{COLOR_RESET}")
            print(
                f"    Passed: {COLOR_PASS}{passed}{COLOR_RESET}, "
                f"Failed: {COLOR_ERROR}{failed}{COLOR_RESET}, "
                f"Errors: {COLOR_ERROR}{error}{COLOR_RESET}"
            )

            # List failed/error schedules
            if failed > 0 or error > 0:
                print(f"\n    {COLOR_ERROR}Failed/Error schedules:{COLOR_RESET}")
                for r in results:
                    if r["status"] != "pass":
                        status_color = (
                            COLOR_ERROR if r["status"] == "error" else COLOR_WARNING
                        )
                        print(
                            f"      Schedule {r['schedule_id']}: "
                            f"{status_color}{r['status'].upper()}{COLOR_RESET}"
                        )
                        print(
                            f"        {r['errors']} errors, "
                            f"{COLOR_WARNING}{r['warnings']} warnings{COLOR_RESET}"
                        )

                        # In verbose mode, show details for failed verifications
                        if verbose:
                            print(
                                f"\n      {COLOR_INFO}Detailed verification output:{COLOR_RESET}"
                            )
                            details = r["details"].replace("\n", "\n        ")
                            # Color the details output
                            details = (
                                details.replace(
                                    "ERROR:", f"{COLOR_ERROR}ERROR:{COLOR_RESET}"
                                )
                                .replace(
                                    "WARNING:", f"{COLOR_WARNING}WARNING:{COLOR_RESET}"
                                )
                                .replace("PASS:", f"{COLOR_PASS}PASS:{COLOR_RESET}")
                            )
                            print(f"        {details}")
                            print()

    print_header("OVERALL RESULTS")
    print(
        f"Total Passed: {COLOR_PASS}{total_passed}{COLOR_RESET}\n"
        f"Total Failed: {COLOR_ERROR}{total_failed}{COLOR_RESET}\n"
        f"Total Errors: {COLOR_ERROR}{total_error}{COLOR_RESET}"
    )


def main():
    """Process log files based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify execution logs against schedules"
    )
    parser.add_argument(
        "--device", help="Device ID to verify (default: all)", default=None
    )
    parser.add_argument(
        "--app", help="Application name to verify (default: all)", default=None
    )
    parser.add_argument(
        "-v", "--verbose", help="Show detailed output for failures", action="store_true"
    )
    args = parser.parse_args()

    # Find matching log files
    log_pattern = "data/logs/logs-*.txt"
    if args.device:
        log_pattern = f"data/logs/logs-{args.device}-*.txt"
    log_files = glob.glob(log_pattern)

    if not log_files:
        print_error("No log files found matching criteria")
        return

    # Filter by application if specified
    if args.app:
        log_files = [f for f in log_files if f"-{args.app}-schedule-" in f]

    if not log_files:
        print_error("No log files found matching criteria after filtering")
        return

    print_info(f"Found {len(log_files)} log files to process")

    # Process all matching files
    results = []
    for log_path in sorted(log_files):
        print_info(f"\nProcessing: {log_path}")
        result = get_verification_result(log_path, verbose=args.verbose)
        results.append(result)

    # Print final summary
    print_summary(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
