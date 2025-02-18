import sqlite3
import argparse
import os
import sys
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass

DB_PATH = "data/benchmark_results.db"

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT * FROM benchmarks WHERE 1=1"
    query += " AND run_type = 'aggregate'"

    cursor.execute(query)
    rows = cursor.fetchall()

    # Print header
    print(f"{'Device':<15} {'Benchmark':<30} {'Time':>12} {'Unit':<8}")
    print("-" * 65)
    for row in rows:
        device = row[1]
        run_name = row[7]
        aggregate_name = row[14]
        if aggregate_name == "cv":
            real_time = f"{float(row[12]):.2f}"
            time_unit = "%"
        else:
            real_time = f"{float(row[12]):.4f}"
            time_unit = row[13]
        print(f"{device:<15} {run_name:<40} {real_time:>12} {time_unit:<8}")

    conn.close()
