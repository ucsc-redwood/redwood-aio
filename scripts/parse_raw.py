import re
import sys

# Regex that matches lines of the form:
# OMP_CifarDense/Baseline/1             223 ms         221 ms            3
# VK_CifarSparse/Stage1/iterations:100  2.19 ms        0.309 ms         2868
# capturing:
#  - benchmark_name => "OMP_CifarDense/Baseline/1" etc.
#  - real_time => first number (223, 2.19, etc.)
#  - cpu_time => second number (221, 0.309, etc.) [we won't necessarily use it]
#  - iterations => final integer
timing_line_pattern = re.compile(
    r"""
    ^
    (?P<benchmark_name>(?:OMP|CUDA|VK)\S*)   # "OMP_CifarDense/Stage1_little/..."
    \s+
    (?P<real_time>[0-9]+\.[0-9]+|[0-9]+)     # "223" or "223.5"
    \s+ms\s+
    (?P<cpu_time>[0-9]+\.[0-9]+|[0-9]+)
    \s+ms\s+
    (?P<iterations>\d+)
    """,
    re.VERBOSE,
)


def parse_benchmark_name(benchmark_name):
    """
    Given something like:
      "OMP_CifarDense/Baseline/1"
      "OMP_CifarDense/Baseline_Pinned_Little/2"
      "OMP_CifarDense/Stage1_big/4/iterations:100"
      "VK_CifarSparse/Stage9/iterations:100"
      "CUDA_Tree/Baseline"
    return (backend, application, stage, core_type, num_threads).

    - stage=0 if 'Baseline' (unless we find 'StageX' explicitly).
    - If "StageX" => stage = X.
    - If "little"/"medium"/"big" => core_type
    - If final slash part is purely digits => num_threads
    """
    backend = None
    application = None
    stage = None
    core_type = None
    num_threads = None

    # Example: "OMP_CifarDense/Stage1_little/2/iterations:100"
    # Split by '/'
    parts = benchmark_name.split("/")

    # The first part typically has <Backend>_<Application>, e.g. "OMP_CifarDense"
    first_part = parts[0]  # "OMP_CifarDense"
    subparts = first_part.split("_", maxsplit=1)
    # e.g. ["OMP", "CifarDense"] or ["VK", "CifarSparse"] or ["OMP","Tree"]

    if len(subparts) == 2:
        backend, possible_app = subparts
    else:
        # If somehow there's no underscore
        backend = subparts[0]
        possible_app = "UnknownApp"

    # Recognize known apps
    known_apps = ["CifarDense", "CifarSparse", "Tree"]
    application = None
    for app in known_apps:
        if possible_app.startswith(app):
            application = app
            break
    if not application:
        application = possible_app  # fallback if we didn't match

    # We'll look for "Baseline" => stage=0, or "Stage(\d+)" => stage = that number
    stage_pattern = re.compile(r"Stage(\d+)", re.IGNORECASE)
    # We'll detect "little", "medium", "big", or pinned, etc.

    for p in parts[1:]:
        p_lower = p.lower()
        # Baseline => stage=0
        if "baseline" in p_lower:
            stage = 0

        # Check if we have StageX
        stage_match = stage_pattern.search(p)
        if stage_match:
            stage = int(stage_match.group(1))

        # Check for core_type
        if "little" in p_lower:
            core_type = "little"
        elif "medium" in p_lower:
            core_type = "medium"
        elif "big" in p_lower:
            core_type = "big"

        # If p is purely digits => num_threads
        if p.isdigit():
            num_threads = int(p)

    # If stage is still None, default to 0 (some logs won't have "Stage" or "Baseline")
    if stage is None:
        stage = 0

    return backend, application, stage, core_type, num_threads


def parse_benchmark_log(raw_text, machine_name="MyDevice"):
    """
    Parse a single device's raw log text.
    Return a list of dicts:
      [
        {
          "machine_name": ...,
          "backend": "OMP"/"CUDA"/"VK",
          "application": "CifarDense"/"CifarSparse"/"Tree"/...,
          "stage": 0 or 1..N,
          "core_type": "little"/"medium"/"big"/None,
          "num_threads": int or None,
          "time_ms": float
        },
        ...
      ]
    """
    results = []
    lines = raw_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = timing_line_pattern.match(line)
        if match:
            benchmark_name = match.group("benchmark_name")
            real_time_str = match.group("real_time")
            real_time = float(real_time_str)

            backend, application, stage, core_type, num_threads = parse_benchmark_name(
                benchmark_name
            )

            record = {
                "machine_name": machine_name,
                "backend": backend,
                "application": application,
                "stage": stage,
                "core_type": core_type,
                "num_threads": num_threads,
                "time_ms": real_time,
            }
            results.append(record)
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_raw.py <benchmark_log_file>")
        sys.exit(1)

    filename = sys.argv[1]
    machine_name = filename.replace(".txt", "")

    with open(filename) as f:
        raw_log = f.read()

        parsed = parse_benchmark_log(raw_log, machine_name=machine_name)
        for r in parsed:
            print(r)
