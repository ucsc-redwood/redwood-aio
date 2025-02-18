from pathlib import Path
import sqlite3

def main():
    in_dir = Path("./data/raw_bm_results")
    db_path = "./data/benchmark_results.db"

    # check if the input directory exists
    if not in_dir.is_dir():
        print(f"Error: input directory {in_dir} not found.")
        return
    
    # print out all .json files in the input directory
    for file in in_dir.glob("*.json"):
        print(file)



if __name__ == "__main__":
    main()
