# parser.py

import re

# The regex to identify benchmark lines
timing_line_pattern = re.compile(
    r"""
    ^
    (?P<benchmark_name>(?:OMP|CUDA|VK)\S*)
    \s+
    (?P<real_time>[0-9]+\.[0-9]+|[0-9]+)
    \s+ms\s+
    (?P<cpu_time>[0-9]+\.[0-9]+|[0-9]+)
    \s+ms\s+
    (?P<iterations>\d+)
    """,
    re.VERBOSE,
)


def parse_benchmark_name(benchmark_name):
    backend = None
    application = None
    stage = None
    core_type = None
    num_threads = None

    parts = benchmark_name.split("/")
    first_part = parts[0]  # e.g., "OMP_CifarDense"
    subparts = first_part.split("_", maxsplit=1)

    if len(subparts) == 2:
        backend, possible_app = subparts
    else:
        backend = subparts[0]
        possible_app = "UnknownApp"

    known_apps = ["CifarDense", "CifarSparse", "Tree"]
    for app in known_apps:
        if possible_app.startswith(app):
            application = app
            break
    if not application:
        application = possible_app

    import re

    stage_pattern = re.compile(r"Stage(\d+)", re.IGNORECASE)
    for p in parts[1:]:
        plow = p.lower()
        if "baseline" in plow:
            stage = 0
        m = stage_pattern.search(p)
        if m:
            stage = int(m.group(1))
        if "little" in plow:
            core_type = "little"
        elif "medium" in plow:
            core_type = "medium"
        elif "big" in plow:
            core_type = "big"
        if p.isdigit():
            num_threads = int(p)

    if stage is None:
        stage = 0

    return backend, application, stage, core_type, num_threads


def parse_benchmark_log(raw_text, machine_name="Unknown"):
    """
    Parse the raw log text for a single device, returning a list of records:
    [
      {
        "machine_name": machine_name,
        "backend": ...,
        "application": ...,
        "stage": ...,
        "core_type": ...,
        "num_threads": ...,
        "time_ms": ...
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
            results.append(
                {
                    "machine_name": machine_name,
                    "backend": backend,
                    "application": application,
                    "stage": stage,
                    "core_type": core_type,
                    "num_threads": num_threads,
                    "time_ms": real_time,
                }
            )

    return results
