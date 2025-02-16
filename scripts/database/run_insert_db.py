# insert_db.py

import sys
import os
from parser import parse_benchmark_log
from db import create_database, insert_benchmark_result

DB_PATH = "scripts/benchmark_results.db"

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/database/run_insert_db.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    if not os.path.isabs(logfile):
        logfile = os.path.abspath(logfile)
    machine_name = os.path.splitext(os.path.basename(logfile))[0]

    # Create DB if needed
    create_database(DB_PATH)

    # Parse the log
    with open(logfile, "r") as f:
        raw_text = f.read()
    records = parse_benchmark_log(raw_text, machine_name)

    # Insert each record
    for rec in records:
        insert_benchmark_result(DB_PATH, rec)

    print(f"Inserted {len(records)} records into {DB_PATH} for machine '{machine_name}'")

if __name__ == "__main__":
    main()
