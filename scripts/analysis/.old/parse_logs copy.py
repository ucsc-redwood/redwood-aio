import re
from collections import defaultdict
import json


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
    # This pattern:
    #  - Captures the backend in a group named 'backend'   => ([vk|omp|cuda])
    #  - Optionally captures "[Core: <number>]"            => (?:\[Core:\s*(?P<core_id>\d+)\])?
    #  - Optionally captures "[Thread: <idx>/<num>]"       => (?:\[Thread:\s*(?P<thread_idx>\d+)\/(?P<num_threads>\d+)\])?
    #  - Captures "Stage: <number>" in group 'stage'       => \[Stage:\s*(?P<stage>\d+)\]
    #  - Captures "App: 0x..." in group 'app_address'      => \[App:\s*(?P<app_address>0x[0-9a-fA-F]+)\]
    #
    # We make use of non-capturing groups (?: ...) to handle the optional parts gracefully.

    pattern = re.compile(
        r"\[.*?\] \[.*?\] \[debug\] \[(?P<backend>\w+)\]"  # e.g. [omp] or [vk] or [cuda]
        r"(?:\[Core:\s*(?P<core_id>\d+)\])?"  # optionally match [Core: X]
        r"(?:\[Thread:\s*(?P<thread_idx>\d+)\/(?P<num_threads>\d+)\])?"  # optionally match [Thread: x/y]
        r"\s*\[Stage:\s*(?P<stage>\d+)\]"  # match [Stage: x]
        r"\s*\[App:\s*(?P<app_address>0x[0-9a-fA-F]+)\]"  # match [App: 0x...]
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
    Given a list of parsed log lines (each a dict containing at least:
      {
        'backend':    str,
        'core_id':    str or None,
        'thread_idx': str or None,
        'num_threads':str or None,
        'stage':      str (digit),
        'app_address':str (e.g. '0xb4000078accf2810')
      }
    in chronological order),
    this function will:

    1) Check there are exactly 20 unique App addresses.
    2) For each App address, check that Stage goes from 1..9 in ascending order
       (no stage out-of-order).

    It prints a final report, showing:
        - whether the 20-app check passed
        - stage-order checks per app
        - summary for each app

    """

    # Keep track of the highest stage encountered for each app
    # so we can validate that each new line is >= the last stage
    # (and hopefully that they eventually reach stage 9).
    last_stage_for_app = defaultdict(lambda: 0)

    # To verify "Stage 1..9 in ascending order" strictly, we also track
    # the distinct set of stages we've actually seen for each app.
    stages_for_app = defaultdict(list)

    # We'll process each parsed line in the order it appears
    for i, row in enumerate(parsed_data):
        app = row["app_address"]
        backend = row["backend"]  # 'omp', 'vk', or 'cuda'
        stage_str = row["stage"]

        if not stage_str.isdigit():
            print(f"WARNING: Stage is not numeric on line {i}: {stage_str}")
            continue

        stage = int(stage_str)

        # Check ordering: The new stage must be >= the last stage we saw,
        # otherwise it's out of order for that app.
        # If you want a strictly increasing rule (1 < 2 < 3 ...),
        # use  stage <= last_stage_for_app[app] as a failure condition.
        if stage < last_stage_for_app[app]:
            print(
                f"ERROR: Out-of-order stage for App {app} at line {i}. "
                f"Got stage {stage} after having stage {last_stage_for_app[app]}."
            )
        else:
            # Update last stage
            last_stage_for_app[app] = stage

        # Record it in a list (so we can see exactly which stages we got)
        stages_for_app[app].append((stage, backend))

    # ----------------------------------------------------------------------
    # Final check #1: Exactly 20 distinct App addresses
    # ----------------------------------------------------------------------
    all_apps = list(stages_for_app.keys())
    if len(all_apps) != 20:
        print(f"ERROR: Expected 20 distinct app addresses, but found {len(all_apps)}.")
        print("Apps found:", all_apps)
    else:
        print("PASS: Exactly 20 distinct app addresses found.")

    # ----------------------------------------------------------------------
    # Final check #2: Each app must have stages 1..9 in ascending order
    # ----------------------------------------------------------------------
    # We do a stricter check that each app has each stage from 1 to 9 at least once
    # and in ascending sequence. That means:
    #   The final stage seen must be at least 9
    #   We never jumped backwards
    #   (Optionally, check if they might have repeated the same stage multiple times)
    #
    # Implementation detail:
    # We'll gather the unique stage numbers for each app in chronological order
    # and verify it contains at least 1..9 in ascending order.
    #
    for app, stage_tuples in stages_for_app.items():
        # Sort by the order in which they appeared.
        # If you truly read them in chronological order from the logs,
        # stage_tuples is already in that order. But let's be safe.
        # (If the input is guaranteed chronological, you can skip sorting.)
        # stage_tuples = sorted(stage_tuples, key=lambda x: x[0])  # Not correct, we want log order
        # Actually, we assume the original parse was in the exact log order, so no re-sorting:
        pass

        # Extract just the stage numbers in the order they appeared
        stage_numbers = [s for (s, b) in stage_tuples]

        # Quick check that final stage is 9
        # and that we never jumped backward
        # (We partially did the check in the loop, but let's do a final summary.)
        if stage_numbers[-1] < 9:
            print(f"ERROR: App {app} only reached stage {stage_numbers[-1]}, not 9.")
        else:
            # If you want to ensure it hits *every* stage from 1 to 9, you can do:
            needed = set(range(1, 10))
            have = set(stage_numbers)
            missing = needed - have
            if missing:
                print(
                    f"ERROR: App {app} is missing stage(s) {sorted(missing)}. "
                    f"Stages seen: {stage_numbers}"
                )
            else:
                # We can do an additional check: are they in ascending order?
                # (We mostly enforced that during the parse, but let's do a final pass.)
                is_ascending = all(
                    stage_numbers[i] <= stage_numbers[i + 1]
                    for i in range(len(stage_numbers) - 1)
                )
                if not is_ascending:
                    print(f"ERROR: App {app} has out-of-order stages: {stage_numbers}")
                else:
                    print(f"PASS: App {app} has stages 1..9 in ascending order.")

    # ----------------------------------------------------------------------
    # Print a final summary for each app (helpful for debugging/trace)
    # ----------------------------------------------------------------------
    print("\n--- FINAL REPORT ---")
    for app in sorted(stages_for_app.keys()):
        stage_str = ", ".join(f"{s}({b})" for (s, b) in stages_for_app[app])
        print(f"App: {app} => {stage_str}")


def load_schedule_from_json(json_path):
    """
    Load schedule config from JSON file.
    Return a dictionary:
        {
          'schedule_id': str,
          'device_id': str,
          'chunks': [
              {
                 'name': str,
                 'hardware': str,
                 'threads': int,
                 'stage_set': set of ints
              },
              ...
          ]
        }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    schedule = data["schedule"]
    # Convert stage lists to sets
    for chunk in schedule["chunks"]:
        chunk["stage_set"] = set(chunk["stages"])

    return schedule


def verify_log_against_schedule(parsed_logs, schedule):
    """
    Given parsed log lines and the loaded schedule data,
    attempt to map each core_id to a chunk, and check GPU usage etc.
    """

    # 1) Gather usage by core
    #    and track GPU usage if backend is 'vk' or 'cuda'
    stages_used_by_core = defaultdict(set)  # core_id -> set of stages
    backends_used_by_core = defaultdict(set)  # core_id -> set of backends
    gpu_stages_encountered = set()  # For lines with vk/cuda & no core

    for row in parsed_logs:
        stage = int(row["stage"])
        backend = row["backend"]  # 'omp', 'vk', or 'cuda'
        core_id = row["core_id"]  # might be None if not in the log line

        if backend in ("vk", "cuda"):
            # GPU usage
            if core_id is None:
                # purely GPU lines with no [Core: X]
                gpu_stages_encountered.add(stage)
            else:
                # Possibly your system logs a 'Core' ID for GPU in a different way
                stages_used_by_core[core_id].add(stage)
                backends_used_by_core[core_id].add(backend)
        else:
            # 'omp' => CPU usage
            if core_id is not None:
                stages_used_by_core[core_id].add(stage)
                backends_used_by_core[core_id].add(backend)
            else:
                # If there's an OMP line but no core ID, that is unusual or an error
                print(f"WARNING: OMP line but no core_id? {row}")

    # 2) Build an index of chunk_name -> {stage_set}, threads, hardware
    chunk_map = {}
    for chunk in schedule["chunks"]:
        chunk_name = chunk["name"]
        chunk_map[chunk_name] = {
            "hardware": chunk["hardware"],
            "threads": chunk["threads"],
            "stage_set": chunk["stage_set"],
        }

    # 3) Attempt to find which core_ids belong to which chunk
    #    We'll track a reverse mapping: chunk_name -> set of core_ids
    chunk_cores = defaultdict(set)

    # We'll also keep track of any "unmatched" cores or conflicting usage
    unmatched_cores = set(stages_used_by_core.keys())

    for chunk_name, info in chunk_map.items():
        # The chunk's official stage set
        chunk_stages = info["stage_set"]
        hardware = info["hardware"]
        threads_req = info["threads"]

        # We find all cores that have EXACTly the same stage set
        # or at least a subset that matches chunk's stage set
        # In real usage, you might refine the logic:
        #   (stages_used_by_core[core] <= chunk_stages)
        #   and also no overlap with other chunk stage sets
        # But here's a simple approach for demonstration:
        candidate_cores = []
        for c in unmatched_cores:
            # e.g. check if the set of stages is exactly equal to chunk_stages
            # or is a sub/sup set depending on your scenario
            s_used = stages_used_by_core[c]

            if s_used == chunk_stages:
                candidate_cores.append(c)

        # If we found exactly `threads_req` cores that match, we assume they belong here
        if len(candidate_cores) == threads_req:
            chunk_cores[chunk_name] = set(candidate_cores)
            # remove them from unmatched
            for c in candidate_cores:
                unmatched_cores.remove(c)
        else:
            print(
                f"WARNING: For chunk '{chunk_name}', expected {threads_req} cores "
                f"with stage set={chunk_stages}, but found {len(candidate_cores)} matches."
            )

    # 4) Check GPU usage chunk
    #    If there's a chunk with hardware='gpu', threads=0 => we expect that chunk's stage set
    #    to appear in the `gpu_stages_encountered` or in a GPU-labeled core.
    for chunk_name, info in chunk_map.items():
        if info["hardware"] == "gpu":
            chunk_stages = info["stage_set"]
            # Check if chunk_stages is included in the union of all GPU stages actually encountered:
            #   union of (gpu_stages_encountered) + any stages used by a core that had 'cuda' or 'vk'
            #   for demonstration, let's just check the first part
            #   (In reality, if you log a "Core" ID for GPU threads, you'd also gather that.)
            #
            missing_stages = chunk_stages - gpu_stages_encountered
            if missing_stages:
                print(
                    f"ERROR: GPU chunk '{chunk_name}' has stage(s) {missing_stages} "
                    f"not found in GPU usage logs."
                )
            else:
                print(
                    f"PASS: GPU chunk '{chunk_name}' usage looks correct for stages {chunk_stages}"
                )

    # 5) Print out final assignment and mismatches
    print("\n--- Final Chunk-Core Mapping ---")
    for chunk_name in chunk_map:
        cset = chunk_cores[chunk_name]
        print(f"{chunk_name} => {cset if cset else 'No cores assigned'}")

    if unmatched_cores:
        print(f"\nWARNING: Some cores not matched to any chunk: {unmatched_cores}")


if __name__ == "__main__":
    log_filename = "data/logs/logs-3A021JEHN02756-cifar-dense-schedule-1.txt"
    schedule_filename = (
        "data/generated-schedules/3A021JEHN02756_CifarDense_schedule_001.json"
    )

    with open(log_filename, "r") as f:
        log_text = f.read()

        # Convert the multiline string to a list of lines
        lines = log_text.splitlines()

        # Parse
        parsed_data = parse_log_lines(lines)

        # # Print each result
        # for entry in parsed_data:
        #     print(entry)

        # Verify
        verify_log(parsed_data)

        # Load schedule
        schedule = load_schedule_from_json(schedule_filename)

        # Verify
        verify_log_against_schedule(parsed_data, schedule)
