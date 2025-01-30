import re

line_pattern = re.compile(
    r"""
    ^(?P<benchmark_name>(?:OMP|CUDA|VK)\S*)      # e.g. OMP_CifarDense/Stage1_little/1/iterations:100
    \s+                                         # some whitespace
    (?P<real_time>[0-9]+\.[0-9]+|[0-9]+)         # e.g. 58.1 or 216
    \s+ms\s+
    (?P<cpu_time>[0-9]+\.[0-9]+|[0-9]+)
    \s+ms\s+
    (?P<iterations>\d+)
    """,
    re.VERBOSE
)

def parse_benchmark_name(benchmark_name):
    """
    Parse something like:
      'OMP_CifarDense/Baseline_Pinned_Little/4/iterations:100'
      'VK_CifarSparse_Baseline/iterations:100'
      'OMP_CifarDense/Stage3_big/2'
    Return (backend, application, stage, core_type, num_threads).
    """
    # Defaults
    backend = None
    application = None
    stage = None
    core_type = None
    num_threads = None

    # 1) Split on '/'
    parts = benchmark_name.split('/')
    # e.g. ["OMP_CifarDense", "Baseline_Pinned_Little", "4", "iterations:100"]

    # 2) Parse the first part to get backend, application, maybe "Baseline"/"Stage"
    first_part = parts[0]  # e.g. "OMP_CifarDense" or "VK_CifarSparse_Baseline"
    
    # We'll find the backend by splitting on '_'
    # e.g. "OMP_CifarDense" => ["OMP", "CifarDense"]
    # or   "VK_CifarSparse_Baseline" => ["VK", "CifarSparse", "Baseline"]
    subparts = first_part.split('_')
    backend = subparts[0]  # "OMP", "VK", or "CUDA"
    
    # We’ll accumulate the rest for figuring out the application and possibly "Baseline"
    # e.g. subparts[1:] => ["CifarDense"] or ["CifarSparse","Baseline"] etc.
    # A simple approach is to join them back and see if "Baseline" or "Stage" is in there.
    remainder = "_".join(subparts[1:])  # e.g. "CifarDense", "CifarSparse_Baseline"

    # We can check if it starts with "CifarDense", "CifarSparse", "Tree", etc.
    # Just do a small check:
    possible_apps = ["CifarDense", "CifarSparse", "Tree"]
    found_app = None
    for app_name in possible_apps:
        if remainder.startswith(app_name):
            found_app = app_name
            break

    if found_app:
        application = found_app
        # remove that from remainder
        # e.g. remainder="CifarDense_Baseline"
        remainder = remainder[len(found_app):]  # => "_Baseline"
        if remainder.startswith("_"):
            remainder = remainder[1:]  # => "Baseline"
    else:
        # If we can't parse it, fallback or set unknown
        application = "UnknownApp"

    # If remainder contains "Baseline", we set stage=0
    # e.g. remainder might be "Baseline" or "Baseline_Pinned_Little"
    if "Baseline" in remainder:
        stage = 0
        # Check if there's "little"/"medium"/"big" in remainder
        # or "Pinned_Little" etc.
        if "little" in remainder.lower():
            core_type = "little"
        elif "medium" in remainder.lower():
            core_type = "medium"
        elif "big" in remainder.lower():
            core_type = "big"
        # num_threads might appear in the next parts (parts[1:], parts[2:], etc.)

    # Alternatively, if "Stage" is in remainder, parse stage number
    # e.g. remainder="Stage1_little"
    import re
    stage_match = re.search(r"Stage(\d+)", remainder)
    if stage_match:
        stage = int(stage_match.group(1))
        # Also see if "little", "medium", or "big" is in there
        if "little" in remainder.lower():
            core_type = "little"
        elif "medium" in remainder.lower():
            core_type = "medium"
        elif "big" in remainder.lower():
            core_type = "big"

    # 3) Now parse the slash-separated parts beyond [1].
    #    For example, if parts[1] = "Stage1_little",
    #    parts[2] might be "4" => num_threads=4.
    #    or if parts[1] = "Baseline_Pinned_Little", parts[2] might be "4".
    for p in parts[1:]:
        # If p is purely digits => threads
        if p.isdigit():
            num_threads = int(p)
        # If "little"/"medium"/"big" is found, set core_type (in case it's here)
        # If "Stage(\d+)" is found here, we can parse that too
        # If "Baseline" is found here, stage=0, etc.
        # If "iterations:" is found => ignore

        # you can do a quick check:
        p_lower = p.lower()
        if "little" in p_lower:
            core_type = "little"
        elif "medium" in p_lower:
            core_type = "medium"
        elif "big" in p_lower:
            core_type = "big"
        elif "baseline" in p_lower:
            stage = 0

        # If it matches something like "Stage(\d+)"
        s_match = re.match(r"Stage(\d+)", p)
        if s_match:
            stage = int(s_match.group(1))

    # If stage is still None => fallback?
    if stage is None:
        stage = 0  # or set to -1 if you prefer

    return backend, application, stage, core_type, num_threads

def parse_raw_benchmark_lines(lines, machine_name="GooglePixel"):
    """
    Parse the raw lines from the logs, returning a list of dict with:
      backend, application, stage, core_type, num_threads, time_ms, machine_name
    """
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = line_pattern.match(line)
        if match:
            benchmark_name = match.group("benchmark_name")
            real_time = float(match.group("real_time"))
            # cpu_time = float(match.group("cpu_time"))  # if needed
            # iters = int(match.group("iters"))          # if needed

            backend, application, stage, core_type, num_threads = parse_benchmark_name(benchmark_name)

            record = {
                "machine_name": machine_name,
                "backend": backend,
                "application": application,
                "stage": stage,
                "core_type": core_type,
                "num_threads": num_threads,
                "time_ms": real_time
            }
            results.append(record)
        else:
            # either doesn't start with OMP/CUDA/VK or doesn't fit the pattern
            continue
    return results


if __name__ == "__main__":
    # Example usage:

    # # Pretend we read from a file or have the lines in a list:
    # sample_lines = [
    #   "OMP_CifarDense/Baseline/1                      216 ms          214 ms            3",
    #   "OMP_CifarDense/Baseline_Pinned_Little/2        656 ms          649 ms            1",
    #   "OMP_CifarDense/Stage1_little/4/iterations:100  4.46 ms         4.38 ms          100",
    #   "VK_CifarDense_Baseline/iterations:100          58.1 ms         3.61 ms          100",
    #   "VK_CifarSparse/Stage1/iterations:100           1.75 ms         0.101 ms         100"
    # ]

    # Read lines from parse.raw file
    with open("3A021JEHN02756.raw", "r") as f:
        sample_lines = f.readlines()

        parsed = parse_raw_benchmark_lines(sample_lines, machine_name="GooglePixel")
        for r in parsed:
            print(r)
