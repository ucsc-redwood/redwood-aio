# Scripts

## Database

scripts are used to parse the raw logs and insert them into the database.

```bash
    rm -f scripts/benchmark_results.db
    py scripts/database/run_splite_raw.py scripts/database/raw_logs/02_16_2025.txt
    py scripts/database/run_insert_db.py scripts/database/raw_logs/3A021JEHN02756.txt
    py scripts/database/run_insert_db.py scripts/database/raw_logs/ce0717178d7758b00b7e.txt
    py scripts/database/run_insert_db.py scripts/database/raw_logs/9b034f1b.txt
```

or just 

```bash
    just bm-to-db
```

### Query

Example run:

```bash
    py scripts/database/run_query.py --stage 0 --application CifarDense
```


