import sqlite3

DB_PATH = "data/tmp.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM benchmarks")
count = cursor.fetchone()[0]
print(f"Number of entries in benchmarks table: {count}")

conn.close()
