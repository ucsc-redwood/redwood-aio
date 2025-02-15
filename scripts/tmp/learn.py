import json
from pathlib import Path
import argparse


def parse_schedule(file: Path):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Python CLI Example")
    parser.add_argument("path", help="Path to the input file")
    args = parser.parse_args()

    path = Path(args.path)
    for file in path.iterdir():
        if file.is_file() and file.suffix == ".json":
            data = parse_schedule(file)
            print(data)

            
    threads = [1, 2, 3, 4]

    print(json.dumps(threads))


if __name__ == "__main__":
    main()
