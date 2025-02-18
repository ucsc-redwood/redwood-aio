import json
import sqlite3
import os
import glob
import re
from typing import Optional


def parse_filename(filename):
    """
    Given a filename like: BM_CifarDense_OMP_3A021JEHN02756.json
    extract the application name, backend, and device name.

    Returns (application_name, backend, device_name).
    """
    base = os.path.basename(filename)  # BM_CifarDense_OMP_3A021JEHN02756.json
    root, _ = os.path.splitext(base)  # BM_CifarDense_OMP_3A021JEHN02756
    parts = root.split("_")  # ["BM", "CifarDense", "OMP", "3A021JEHN02756"]

    if len(parts) < 4:
        # Adjust as needed if your filenames have a different pattern
        raise ValueError(f"Unexpected filename format: {filename}")

    application_name = parts[1]  # e.g. CifarDense
    backend = parts[2]  # e.g. OMP
    device_name = parts[3]  # e.g. 3A021JEHN02756

    return application_name, backend, device_name


def read_benchmarks(folder="data/raw_bm_results"):
    """
    Read and parse all JSON benchmark files under `folder`.
    Returns a list of dictionaries, each containing:
      - application
      - backend
      - device
      - data (the JSON content)
    """
    results = []

    # Use glob to find all .json files in the folder
    pattern = os.path.join(folder, "*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"Warning: No JSON files found in {folder}")
        return results

    for file_path in json_files:
        # Parse the filename for application, backend, and device
        try:
            application, backend, device = parse_filename(file_path)
        except ValueError as e:
            print(f"Error parsing filename {file_path}: {e}")
            continue

        # Load the JSON benchmark data
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from {file_path}: {e}")
            print(f"Line {e.lineno}, column {e.colno}: {e.msg}")
            continue
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {e}")
            continue

        # Store the data along with metadata
        results.append(
            {
                "application": application,
                "backend": backend,
                "device": device,
                "data": data,
            }
        )

    return results


def parse_run_name(input_str):
    """
    Parse an input string with the following possible format:

      {Backend}_{Application}/{StageInfo}[/{NumThreads}]

    Where:
      - Backend is one of {OMP, CUDA, VK}
      - Application is one of {CifarDense, CifarSparse, Tree}
      - StageInfo is either:
          "Baseline" optionally followed by a tail (e.g., "_median", "_cv", "_stddev")
          OR
          "StageX" optionally with an underscore and a core type (e.g., "Stage4_big")
          In the case of "Baseline", stage is 0.
      - NumThreads is an integer, optionally followed by tailing data (which are ignored).

    Examples:
      - "OMP_CifarDense/Baseline/2"               -> backend: "OMP",  application: "CifarDense", stage: 0, core_type: None, num_threads: 2
      - "OMP_CifarSparse/Baseline/1_median"         -> backend: "OMP",  application: "CifarSparse", stage: 0, core_type: None, num_threads: 1
      - "OMP_Tree/Stage2_big/1_cv"                  -> backend: "OMP",  application: "Tree", stage: 2, core_type: "big", num_threads: 1
      - "VK_Tree/Stage6_cv"                         -> backend: "VK",   application: "Tree", stage: 6, core_type: None, num_threads: None

    Returns:
        dict: A dictionary with keys:
            - 'backend'
            - 'application'
            - 'stage'
            - 'core_type'
            - 'num_threads'
    """
    # Split by '/'
    segments = input_str.split("/")
    if len(segments) < 2:
        raise ValueError("Input must have at least two segments separated by '/'")

    # Parse the first segment: "Backend_Application"
    try:
        backend, application = segments[0].split("_", 1)
    except ValueError:
        raise ValueError("First segment must be in the format 'Backend_Application'")

    # Initialize defaults
    stage = None
    core_type = None
    num_threads = None

    # Parse the stage segment (second segment)
    stage_segment = segments[1]
    if stage_segment.startswith("Baseline"):
        stage = 0
    elif stage_segment.startswith("Stage"):
        # Remove "Stage" and capture the stage number and optional core type.
        # This regex captures one or more digits and an optional underscore followed by word characters.
        m = re.match(r"Stage(\d+)(?:_(\w+))?", stage_segment)
        if m:
            stage = int(m.group(1))
            core_candidate = m.group(2)
            # Only assign core_type if it is one of the allowed types.
            if core_candidate in {"little", "small", "big"}:
                core_type = core_candidate
        else:
            raise ValueError(
                "Stage segment does not match expected format (e.g., 'Stage4_big')"
            )
    else:
        raise ValueError("Stage segment must start with 'Baseline' or 'Stage'")

    # Parse the number of threads if the third segment exists.
    if len(segments) > 2:
        thread_segment = segments[2]
        # Extract the leading number ignoring any tailing data (like _median, _cv, _stddev, etc.)
        m = re.match(r"(\d+)", thread_segment)
        if m:
            num_threads = int(m.group(1))

    return {
        "backend": backend,
        "application": application,
        "stage": stage,
        "core_type": core_type,
        "num_threads": num_threads,
    }


def main():

    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect("./data/tmp.db")
    cursor = conn.cursor()

    # Create a table to store the benchmark results
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        run_name TEXT,
        run_type TEXT,
        backend TEXT,
        application TEXT,
        device TEXT,
        stage INTEGER,
        num_threads INTEGER,
        core_type TEXT,
        repetitions INTEGER,
        iterations INTEGER,
        real_time REAL,
        time_unit TEXT,
        aggregate_name TEXT NULL
    )
    """
    )

    benchmarks = read_benchmarks("data/raw_bm_results")

    for bm in benchmarks:
        for result in bm["data"]["benchmarks"]:
            try:
                parsed_run_name = parse_run_name(result["run_name"])
            except ValueError as e:
                print(f"Warning: {e}")
                continue

            # Get aggregate_name with .get() to handle missing key
            aggregate_name = result.get("aggregate_name")

            cursor.execute(
                """
            INSERT INTO benchmarks (
                name, run_name, run_type, backend, application, device, stage, num_threads, core_type, repetitions, iterations, real_time, time_unit, aggregate_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result["name"],
                    result["run_name"],
                    result["run_type"],
                    bm["backend"],
                    bm["application"],
                    bm["device"],
                    parsed_run_name["stage"],
                    parsed_run_name["num_threads"],
                    parsed_run_name["core_type"],
                    result["repetitions"],
                    result["iterations"],
                    result["real_time"],
                    result["time_unit"],
                    aggregate_name,  # Using the .get() result from above
                ),
            )

        # exit(0)

    conn.commit()
    conn.close()

    print("Benchmark data has been written to tmp.db")


if __name__ == "__main__":
    main()
