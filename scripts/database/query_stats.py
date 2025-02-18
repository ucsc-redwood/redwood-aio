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
    query += " AND aggregate_name = 'mean'"


    cursor.execute(query)
    rows = cursor.fetchall()

    # print the "device" "run_name" and "mean"
    for row in rows:
        print(row[0], row[7], row[3])

    conn.close()

