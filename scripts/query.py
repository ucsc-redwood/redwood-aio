import sqlite3
import argparse

DB_NAME = "benchmark_results.db"


def query_database(
    machine_name=None,
    application=None,
    backend=None,
    stage=None,
    core_type=None,
    num_threads=None,
):
    """
    Query the database with optional filters and display the results.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Base query
    query = "SELECT * FROM benchmark_result WHERE 1=1"
    params = []

    # Apply filters if provided
    if machine_name:
        query += " AND machine_name = ?"
        params.append(machine_name)
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

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Get column names
    column_names = [description[0] for description in cursor.description]

    # Print the results
    if rows:
        print(f"{' | '.join(column_names)}")
        print("-" * 100)
        for row in rows:
            print(" | ".join(str(item) for item in row))
    else:
        print("No matching records found.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query benchmark database with optional filters."
    )
    parser.add_argument("--machine_name", type=str, help="Filter by machine name")
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
        machine_name=args.machine_name,
        application=args.application,
        backend=args.backend,
        stage=args.stage,
        core_type=args.core_type,
        num_threads=args.num_threads,
    )
