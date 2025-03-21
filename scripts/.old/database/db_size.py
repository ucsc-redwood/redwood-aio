import sqlite3
import sys
from pathlib import Path

# Add scripts directory to Python path
scripts_dir = str(Path(__file__).parent.parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from helpers import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM benchmarks")
count = cursor.fetchone()[0]
print(f"Number of entries in benchmarks table: {count}")

conn.close()
