# parse_only.py

import sys
import os
from parser import parse_benchmark_log


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/database/run_parse_only.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    if not os.path.isabs(logfile):
        logfile = os.path.abspath(logfile)
    machine_name = os.path.splitext(os.path.basename(logfile))[0]

    with open(logfile, "r") as f:
        raw_text = f.read()

    records = parse_benchmark_log(raw_text, machine_name)
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
