import re
from collections import defaultdict


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


if __name__ == "__main__":
    filename = "data/logs/logs-3A021JEHN02756-cifar-dense-schedule-1.txt"

    with open(filename, "r") as f:
        log_text = f.read()

        # Convert the multiline string to a list of lines
        lines = log_text.splitlines()

        # Parse
        parsed_data = parse_log_lines(lines)

        # Print each result
        for entry in parsed_data:
            print(entry)

        # Verify
        verify_log(parsed_data)
