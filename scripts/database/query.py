import sqlite3
import argparse
import os
import sys
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass

DB_PATH = "data/benchmark_results.db"


@dataclass
class QueryFilters:
    device: Optional[str] = None
    application: Optional[str] = None
    backend: Optional[str] = None
    stage: Optional[int] = None
    core_type: Optional[str] = None
    num_threads: Optional[int] = None


def build_query(filters: QueryFilters) -> Tuple[str, List[Any]]:
    """
    Build SQL query string and parameters based on provided filters.

    Args:
        filters: QueryFilters object containing filter criteria

    Returns:
        Tuple of (query_string, parameters)
    """
    query = "SELECT * FROM benchmarks WHERE 1=1"
    query += " AND run_type = 'aggregate'"
    query += " AND aggregate_name = 'mean'"
    params: List[Any] = []

    # Apply filters if provided
    if filters.device:
        query += " AND device = ?"
        params.append(filters.device)
    if filters.application:
        query += " AND application = ?"
        params.append(filters.application)
    if filters.backend:
        query += " AND backend = ?"
        params.append(filters.backend)
    if filters.stage is not None:
        query += " AND stage = ?"
        params.append(filters.stage)
    if filters.core_type:
        query += " AND core_type = ?"
        params.append(filters.core_type)
    if filters.num_threads is not None:
        query += " AND num_threads = ?"
        params.append(filters.num_threads)

    query += " ORDER BY device, application, backend, core_type, stage, num_threads"

    return query, params


def print_verbose_results(rows: List[Tuple], column_names: List[str]) -> None:
    """
    Print all columns of query results.

    Args:
        rows: List of database rows
        column_names: List of column names
    """
    if not rows:
        print("No matching records found.")
        return

    print(f"{' | '.join(column_names)}")
    print("-" * 100)
    for row in rows:
        print(" | ".join(str(item) for item in row))


def print_summary_results(rows: List[Tuple], column_names: List[str]) -> None:
    """
    Print summarized version of query results.

    Args:
        rows: List of database rows
        column_names: List of column names
    """
    if not rows:
        print("No matching records found.")
        return

    headers = [
        "device",
        "application",
        "backend",
        "stage",
        "core_type",
        "num_threads",
        "real_time",
    ]
    col_indices = [column_names.index(col) for col in headers]

    print(f"{' | '.join(headers)}")
    print("-" * 50)
    for row in rows:
        values = [
            str(row[i]) if i != col_indices[-1] else f"{float(row[i]):.4f}"
            for i in col_indices
        ]
        print(" | ".join(values))


def query_database(
    filters: QueryFilters,
    verbose: bool = False,
) -> None:
    """
    Query the database with optional filters and display the results.

    Args:
        filters: QueryFilters object containing filter criteria
        verbose: Whether to show all columns in output
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query, params = build_query(filters)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]

    if verbose:
        print_verbose_results(rows, column_names)
    else:
        print_summary_results(rows, column_names)

    conn.close()


def parse_arguments() -> Tuple[QueryFilters, bool]:
    """
    Parse command line arguments.

    Returns:
        Tuple of (QueryFilters, verbose_flag)
    """
    parser = argparse.ArgumentParser(
        description="Query benchmark database with optional filters."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device", type=str, help="Filter by machine name")
    parser.add_argument("--application", type=str, help="Filter by application name")
    parser.add_argument(
        "--backend", type=str, help="Filter by backend type (e.g., OMP, CUDA, Vulkan)"
    )
    parser.add_argument("--stage", type=int, help="Filter by stage number")
    parser.add_argument(
        "--core_type", type=str, help="Filter by core type (e.g., little, medium, big)"
    )
    parser.add_argument("--num_threads", type=int, help="Filter by number of threads")

    args = parser.parse_args()

    filters = QueryFilters(
        device=args.device,
        application=args.application,
        backend=args.backend,
        stage=args.stage,
        core_type=args.core_type,
        num_threads=args.num_threads,
    )

    return filters, args.verbose


def main():
    filters, verbose = parse_arguments()
    query_database(filters, verbose)


if __name__ == "__main__":
    main()
