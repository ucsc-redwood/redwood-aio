# db.py

import sqlite3


def create_database(db_name="benchmark_results.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_name TEXT,
            application TEXT,
            backend TEXT,
            stage INTEGER,
            core_type TEXT,
            num_threads INTEGER,
            time_ms REAL,
            UNIQUE (machine_name, application, backend, stage, core_type, num_threads)
        )
    """
    )
    conn.commit()
    conn.close()


def insert_benchmark_result(db_name, record):
    """
    record: A dict with keys:
      machine_name, application, backend, stage, core_type, num_threads, time_ms
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO benchmark_result
                (machine_name, application, backend, stage, core_type, num_threads, time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record["machine_name"],
                record["application"],
                record["backend"],
                record["stage"],
                record["core_type"],
                record["num_threads"],
                record["time_ms"],
            ),
        )
    except sqlite3.IntegrityError:
        # Ignore duplicates in this example
        pass
    conn.commit()
    conn.close()
