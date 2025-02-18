import json
import sqlite3
import os
import glob
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ParsedRunName:
    backend: str
    application: str
    stage: int
    core_type: Optional[str]
    num_threads: Optional[int]


@dataclass
class BenchmarkResult:
    application: str
    backend: str
    device: str
    data: Dict[str, Any]


@dataclass
class DatabaseStats:
    new_entries: int = 0
    updated_entries: int = 0


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse benchmark filename to extract metadata.

    Args:
        filename: Filename like 'BM_CifarDense_OMP_3A021JEHN02756.json'

    Returns:
        Tuple of (application_name, backend, device_name)

    Raises:
        ValueError: If filename doesn't match expected format
    """
    base = os.path.basename(filename)
    root, _ = os.path.splitext(base)
    parts = root.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    return parts[1], parts[2], parts[3]


def read_benchmarks(folder: str = "data/raw_bm_results") -> List[BenchmarkResult]:
    """
    Read and parse all JSON benchmark files under specified folder.

    Args:
        folder: Directory containing benchmark JSON files

    Returns:
        List of BenchmarkResult objects containing parsed data
    """
    results = []
    pattern = os.path.join(folder, "*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"Warning: No JSON files found in {folder}")
        return results

    for file_path in json_files:
        try:
            application, backend, device = parse_filename(file_path)
            with open(file_path, "r") as f:
                data = json.load(f)
            results.append(
                BenchmarkResult(
                    application=application, backend=backend, device=device, data=data
                )
            )
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error processing {file_path}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {e}")
            continue

    return results


def parse_run_name(input_str: str) -> ParsedRunName:
    """
    Parse benchmark run name string into components.

    Args:
        input_str: String like "Backend_Application/StageInfo[/NumThreads]"

    Returns:
        ParsedRunName object containing extracted components

    Raises:
        ValueError: If input string doesn't match expected format
    """
    segments = input_str.split("/")
    if len(segments) < 2:
        raise ValueError("Input must have at least two segments separated by '/'")

    try:
        backend, application = segments[0].split("_", 1)
    except ValueError:
        raise ValueError("First segment must be in the format 'Backend_Application'")

    stage = None
    core_type = None
    num_threads = None

    stage_segment = segments[1]
    if stage_segment.startswith("Baseline"):
        stage = 0
    elif stage_segment.startswith("Stage"):
        m = re.match(r"Stage(\d+)(?:_(\w+))?", stage_segment)
        if m:
            stage = int(m.group(1))
            core_candidate = m.group(2)
            if core_candidate in {"little", "small", "big"}:
                core_type = core_candidate
        else:
            raise ValueError("Stage segment does not match expected format")
    else:
        raise ValueError("Stage segment must start with 'Baseline' or 'Stage'")

    if len(segments) > 2:
        thread_segment = segments[2]
        m = re.match(r"(\d+)", thread_segment)
        if m:
            num_threads = int(m.group(1))

    return ParsedRunName(
        backend=backend,
        application=application,
        stage=stage,
        core_type=core_type,
        num_threads=num_threads,
    )


def create_database_schema(cursor: sqlite3.Cursor) -> None:
    """Create the database schema if it doesn't exist."""
    # First drop the existing table if we're recreating the schema
    cursor.execute("DROP TABLE IF EXISTS benchmarks")

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device TEXT,
        application TEXT,
        backend TEXT,
        stage INTEGER,
        core_type TEXT,
        num_threads INTEGER,
        name TEXT,
        run_name TEXT,
        run_type TEXT,
        repetitions INTEGER,
        iterations INTEGER,
        real_time REAL,
        time_unit TEXT,
        aggregate_name TEXT NULL,
        UNIQUE(device, application, backend, stage, core_type, num_threads)
    )
    """
    )


def insert_benchmark_data(
    cursor: sqlite3.Cursor,
    benchmark: BenchmarkResult,
    parsed_run: ParsedRunName,
    result: Dict[str, Any],
) -> bool:
    """
    Insert or update a single benchmark result in the database.

    Returns:
        bool: True if new entry was inserted, False if existing entry was updated
    """
    # First check if entry exists
    cursor.execute(
        """
        SELECT COUNT(*) FROM benchmarks 
        WHERE device = ? 
        AND application = ? 
        AND backend = ? 
        AND stage = ? 
        AND core_type = ? 
        AND num_threads = ?
        """,
        (
            benchmark.device,
            benchmark.application,
            benchmark.backend,
            parsed_run.stage,
            parsed_run.core_type,
            parsed_run.num_threads,
        ),
    )
    exists = cursor.fetchone()[0] > 0

    cursor.execute(
        """
        INSERT INTO benchmarks (
            device, application, backend, stage, core_type, num_threads,
            name, run_name, run_type, repetitions, iterations, real_time, time_unit, aggregate_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(device, application, backend, stage, core_type, num_threads) 
        DO UPDATE SET
            name = excluded.name,
            run_name = excluded.run_name,
            run_type = excluded.run_type,
            repetitions = excluded.repetitions,
            iterations = excluded.iterations,
            real_time = excluded.real_time,
            time_unit = excluded.time_unit,
            aggregate_name = excluded.aggregate_name
        """,
        (
            benchmark.device,
            benchmark.application,
            benchmark.backend,
            parsed_run.stage,
            parsed_run.core_type,
            parsed_run.num_threads,
            result["name"],
            result["run_name"],
            result["run_type"],
            result["repetitions"],
            result["iterations"],
            result["real_time"],
            result["time_unit"],
            result.get("aggregate_name"),
        ),
    )

    return not exists


def process_benchmarks(
    benchmarks: List[BenchmarkResult], db_path: str = "./data/tmp.db"
) -> None:
    """
    Process benchmark results and store them in the database.

    Args:
        benchmarks: List of benchmark results to process
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_database_schema(cursor)

    # Track stats for each file
    file_stats: Dict[str, DatabaseStats] = {}

    for bm in benchmarks:
        print(f"Processing {bm.application} {bm.backend} {bm.device}...")

        file_key = f"{bm.application}_{bm.backend}_{bm.device}"
        stats = DatabaseStats()

        for result in bm.data["benchmarks"]:
            try:
                parsed_run = parse_run_name(result["run_name"])
                is_new = insert_benchmark_data(cursor, bm, parsed_run, result)
                if is_new:
                    stats.new_entries += 1
                else:
                    stats.updated_entries += 1
            except ValueError as e:
                print(f"Warning: {e}")
                continue

        file_stats[file_key] = stats

    conn.commit()
    conn.close()

    # Print final report
    print("\nDatabase Update Report:")
    print("-" * 60)
    total_new = 0
    total_updated = 0

    for file_key, stats in file_stats.items():
        print(f"\n{file_key}:")
        print(f"  New entries: {stats.new_entries}")
        print(f"  Updated entries: {stats.updated_entries}")
        total_new += stats.new_entries
        total_updated += stats.updated_entries

    print("\nTotal Summary:")
    print(f"Total new entries: {total_new}")
    print(f"Total updated entries: {total_updated}")
    print(f"Total processed: {total_new + total_updated}")


def main():
    benchmarks = read_benchmarks("data/raw_bm_results")
    process_benchmarks(benchmarks)


if __name__ == "__main__":
    main()
