import sqlite3

DB_NAME = "benchmark_results.db"

def dump_database(db_name=DB_NAME):
    """
    Dumps all the records from the benchmark_result table.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Fetch all records from the benchmark_result table
    cursor.execute("SELECT * FROM benchmark_result")
    rows = cursor.fetchall()

    # Get column names
    column_names = [description[0] for description in cursor.description]

    # Print the records in a readable format
    print(f"{' | '.join(column_names)}")
    print("-" * 100)
    
    for row in rows:
        print(" | ".join(str(item) for item in row))

    conn.close()

if __name__ == "__main__":
    dump_database()
