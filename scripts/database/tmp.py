import sqlite3
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass

DB_PATH = "data/benchmark_results.db"

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT * FROM benchmarks"

    cursor.execute(query)
    rows = cursor.fetchall()

    for row in rows:
        run_name = row[8]
        print(run_name)

    conn.close()
