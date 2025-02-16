# insert_db.py

import sys
import os
from parser import parse_benchmark_log
from db import create_database, insert_benchmark_result


def main():
    if len(sys.argv) < 2:
        print("Usage: python insert_db.py <logfile> [db_file]")
        sys.exit(1)

    logfile = sys.argv[1]
    # Extract machine name from filename without .txt extension
    machine_name = os.path.splitext(os.path.basename(logfile))[0]
    db_file = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results.db"

    # 1) Create DB if needed
    create_database(db_file)

    # 2) Parse the log
    with open(logfile, "r") as f:
        raw_text = f.read()
    records = parse_benchmark_log(raw_text, machine_name)

    # 3) Insert each record
    for rec in records:
        insert_benchmark_result(db_file, rec)

    print(
        f"Inserted {len(records)} records into {db_file} for machine '{machine_name}'"
    )


if __name__ == "__main__":
    main()
