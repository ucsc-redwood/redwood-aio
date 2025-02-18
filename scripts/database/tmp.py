import json
import sqlite3

# Load the JSON file
data_file = './data/raw_bm_results/BM_CifarDense_OMP_3A021JEHN02756.json'
with open(data_file, 'r') as f:
    data = json.load(f)

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('./data/tmp.db')
cursor = conn.cursor()

# Create a table to store the benchmark results
cursor.execute('''
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    family_index INTEGER,
    instance_index INTEGER,
    run_name TEXT,
    run_type TEXT,
    repetitions INTEGER,
    repetition_index INTEGER,
    threads INTEGER,
    iterations INTEGER,
    real_time REAL,
    cpu_time REAL,
    time_unit TEXT,
    aggregate_name TEXT,
    aggregate_unit TEXT
)
''')

# Insert benchmark data into the table
for benchmark in data.get('benchmarks', []):
    cursor.execute('''
    INSERT INTO benchmarks (
        name, family_index, instance_index, run_name, run_type, repetitions,
        repetition_index, threads, iterations, real_time, cpu_time, time_unit,
        aggregate_name, aggregate_unit
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        benchmark.get('name'),
        benchmark.get('family_index'),
        benchmark.get('per_family_instance_index'),
        benchmark.get('run_name'),
        benchmark.get('run_type'),
        benchmark.get('repetitions'),
        benchmark.get('repetition_index', None),
        benchmark.get('threads'),
        benchmark.get('iterations'),
        benchmark.get('real_time'),
        benchmark.get('cpu_time'),
        benchmark.get('time_unit'),
        benchmark.get('aggregate_name'),
        benchmark.get('aggregate_unit')
    ))

# Commit and close the connection
conn.commit()
conn.close()

print("Benchmark data has been written to benchmark_results.db")
