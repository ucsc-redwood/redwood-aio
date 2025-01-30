# parse_only.py

import sys
from parser import parse_benchmark_log


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_only.py <logfile> [machine_name]")
        sys.exit(1)

    logfile = sys.argv[1]
    machine_name = sys.argv[2] if len(sys.argv) > 2 else "Unknown"

    with open(logfile, "r") as f:
        raw_text = f.read()

    records = parse_benchmark_log(raw_text, machine_name)
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
