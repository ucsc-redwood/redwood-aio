import csv


def parse_csv(csv_file):
    results = {}

    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)

        # (Optional) Skip the header line if it exists:
        header = next(reader, None)

        for row in reader:
            if not row:
                continue

            name = row[0]
            # Only process if the name starts with our target prefix
            if name.startswith("OMP_CifarDense/Baseline"):
                parts = name.split("/")
                core_count_str = parts[-1]

                time_str = row[2]

                core_count = int(core_count_str)
                time = float(time_str)

                results[core_count] = time
    return results


if __name__ == "__main__":
    csv_file = "data.csv"
    results = parse_csv(csv_file)
    print(results)
