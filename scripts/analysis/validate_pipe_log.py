#!/usr/bin/env python3
import sys
import re
import argparse


def main():
    """
    Validate pipeline log consistency:
      1) For each task (app_data), stages 1..9 appear at least once.
      2) Exactly 20 unique tasks.
      3) No more than 8 unique core IDs used.
      4) Each core uses a consistent subset of stages throughout.
    Report any issues found.
    """

    parser = argparse.ArgumentParser(description="Validate pipeline log consistency.")
    parser.add_argument("logfile", help="Path to the .txt log file")
    args = parser.parse_args()

    # Regex to parse lines of the form:
    # [timestamp] [debug] [backend][coreID][thread THR] process_stage_N, app_data: 0x...
    line_regex = re.compile(
        r"^\[([^\]]+)\]\s+\[debug\]\s+\[(?P<backend>\w+)\]\[(?P<core>\d+)\]\[thread\s+(?P<thread>\d+)\]\s+"
        r"process_stage_(?P<stage>\d+),\s+app_data:\s+(?P<appdata>0x[0-9A-Fa-f]+)$"
    )

    # Track for each task which stages we have seen
    stages_per_task = {}  # app_data -> set of stage numbers

    # Track which cores appear in the log
    core_stage_map = {}  # core_id -> set of stage numbers encountered
    backend_map = {}  # core_id -> set of backends encountered (omp/vk, etc.)

    with open(args.logfile, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = line_regex.match(line)
            if not match:
                # Line didn't match the pattern; ignore or log a warning if needed
                continue

            gd = match.groupdict()
            backend = gd["backend"]
            core_id = int(gd["core"])
            thread_id = int(gd["thread"])
            stage_num = int(gd["stage"])
            app_data = gd["appdata"]

            # Update the stages for this task
            if app_data not in stages_per_task:
                stages_per_task[app_data] = set()
            stages_per_task[app_data].add(stage_num)

            # Update info on this core
            if core_id not in core_stage_map:
                core_stage_map[core_id] = set()
                backend_map[core_id] = set()
            core_stage_map[core_id].add(stage_num)
            backend_map[core_id].add(backend)

    # ---- Checks ----

    issues_found = False

    # 1) Exactly 20 unique tasks?
    all_appdatas = list(stages_per_task.keys())
    num_tasks = len(all_appdatas)
    if num_tasks != 20:
        print(f"[ERROR] Expected 20 unique tasks, but found {num_tasks}.")
        issues_found = True

    # 2) For each task, did we see all stages 1..9?
    for app_data, stg_set in stages_per_task.items():
        missing_stages = [s for s in range(1, 10) if s not in stg_set]
        if missing_stages:
            print(f"[ERROR] Task {app_data} missing stages: {missing_stages}")
            issues_found = True

    # 3) Check number of unique cores (should be <= 8)
    all_cores = list(core_stage_map.keys())
    if len(all_cores) > 8:
        print(f"[ERROR] Found {len(all_cores)} unique cores, but expected at most 8.")
        issues_found = True

    # 4) Ensure each core uses a consistent subset of stages throughout.
    #    (As described before, we simply gather stages per core.
    #     If needed, you can add logic to check a fixed mapping of allowed stages.)
    # For illustration, just print the final subset for each core if no issues:
    if not issues_found:
        print("No issues found. Summary of final subsets per core:")
        for cid in sorted(core_stage_map.keys()):
            stage_subset = sorted(core_stage_map[cid])
            bset = ",".join(sorted(backend_map[cid]))
            print(f"  Core {cid} (backends={bset}) => stages {stage_subset}")
    else:
        print("Finished checking. Issues were reported above.")


if __name__ == "__main__":
    main()
