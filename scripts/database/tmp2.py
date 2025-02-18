import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect("./data/tmp.db")
cursor = conn.cursor()


# make a query to get all the data from the table where run_type is aggregate
cursor.execute(
    """
               SELECT run_name, real_time FROM benchmarks 
               WHERE run_type = 'aggregate'
               AND aggregate_name = 'mean'
               ORDER BY real_time DESC
    """
)
data = cursor.fetchall()

# print the data
for row in data:
    print(row)

# Commit and close the connection
conn.commit()
conn.close()
