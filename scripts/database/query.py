import sqlite3
import argparse
import os
import sys

DB_PATH = "data/tmp.db"


def query_database(
    device=None,
    application=None,
    backend=None,
    stage=None,
    core_type=None,
    num_threads=None,
    verbose=False,
):
    """
    Query the database with optional filters and display the results.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Base query
    query = "SELECT * FROM benchmarks WHERE 1=1"
    query += " AND run_type = 'aggregate'"
    query += " AND aggregate_name = 'mean'"
    params = []

    # Apply filters if provided
    if device:
        query += " AND device = ?"
        params.append(device)
    if application:
        query += " AND application = ?"
        params.append(application)
    if backend:
        query += " AND backend = ?"
        params.append(backend)
    if stage is not None:
        query += " AND stage = ?"
        params.append(stage)
    if core_type:
        query += " AND core_type = ?"
        params.append(core_type)
    if num_threads is not None:
        query += " AND num_threads = ?"
        params.append(num_threads)

    # Order results
    query += " ORDER BY device, application, backend, core_type, stage, num_threads"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Get column names
    column_names = [description[0] for description in cursor.description]

    if verbose:
        # Print the results
        if rows:
            print(f"{' | '.join(column_names)}")
            print("-" * 100)
            for row in rows:
                print(" | ".join(str(item) for item in row))
        else:
            print("No matching records found.")
    else:
        # Print just core_type, num_threads and real_time
        if rows:
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
        else:
            print("No matching records found.")

    conn.close()


if __name__ == "__main__":
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

    query_database(
        device=args.device,
        application=args.application,
        backend=args.backend,
        stage=args.stage,
        core_type=args.core_type,
        num_threads=args.num_threads,
        verbose=args.verbose,
    )
