import json
import sqlite3
import os
import glob


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


def main():

    benchmarks = read_benchmarks("data/raw_bm_results")

    for bm in benchmarks:
        print(
            f"\nResults for {bm['application']} using {bm['backend']} on device {bm['device']}:"
        )

        exit(0)

    # # Connect to the SQLite database (or create it if it doesn't exist)
    # conn = sqlite3.connect("./data/tmp.db")
    # cursor = conn.cursor()

    # # Create a table to store the benchmark results
    # cursor.execute(
    #     """
    # CREATE TABLE IF NOT EXISTS benchmarks (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     name TEXT,
    #     run_name TEXT,
    #     run_type TEXT,
    #     repetitions INTEGER,
    #     iterations INTEGER,
    #     real_time REAL,
    #     time_unit TEXT,
    #     aggregate_name TEXT
    # )
    # """
    # )

    # # Insert benchmark data into the table
    # for benchmark in data.get("benchmarks", []):
    #     cursor.execute(
    #         """
    #     INSERT INTO benchmarks (
    #         name, run_name, run_type, repetitions, iterations, real_time,
    #         time_unit, aggregate_name
    #     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    #     """,
    #         (
    #             benchmark.get("name"),
    #             benchmark.get("run_name"),
    #             benchmark.get("run_type"),
    #             benchmark.get("repetitions"),
    #             benchmark.get("iterations"),
    #             benchmark.get("real_time"),
    #             benchmark.get("time_unit"),
    #             benchmark.get("aggregate_name"),
    #         ),
    #     )

    # # Commit and close the connection
    # conn.commit()
    # conn.close()

    # print("Benchmark data has been written to tmp.db")


if __name__ == "__main__":
    main()
