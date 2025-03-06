#!/usr/bin/env python3
import argparse
from pathlib import Path

from codegen_common import (
    parse_schedule_filename,
    read_schedule_file,
    generate_run_pipeline_code,
    build_single_hpp_content,  # We can adapt from it
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True,
                        help="Directory with multiple .json schedule files.")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to place the single aggregated .hpp file.")
    parser.add_argument("--device", required=True,
                        help="Device ID to filter on, e.g. '3A021JEHN02756'.")
    parser.add_argument("--application", required=True,
                        choices=["Tree", "CifarDense", "CifarSparse"],  # or any
                        help="Application name to filter on.")
    parser.add_argument("--out_name", default="multi_schedules.hpp",
                        help="Name of the final aggregated .hpp.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(f"Error: input directory not found: {in_dir}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_id_filter = args.device
    app_filter       = args.application
    out_hpp_name     = args.out_name
    out_hpp_path     = out_dir / out_hpp_name

    # We'll gather schedule_id -> pipeline_code
    schedule_map = {}  # { schedule_id: pipeline_code }

    # Find all relevant .json files
    for json_file in in_dir.glob("*.json"):
        # parse device/app from filename
        try:
            d_id, a_name, sch_id = parse_schedule_filename(json_file.name)
        except ValueError:
            # skip files that don't match pattern
            continue

        if d_id != device_id_filter:
            continue
        if a_name != app_filter:
            continue

        # Now we read the file to confirm it's valid and build code
        try:
            schedule_obj, _ = read_schedule_file(json_file)
        except ValueError as e:
            print(f"Skipping {json_file}: {e}")
            continue

        pipeline_code = generate_run_pipeline_code(schedule_obj)
        schedule_map[sch_id] = pipeline_code

    if not schedule_map:
        print(f"No schedules found in {in_dir} for device={device_id_filter} and application={app_filter}.")
        return

    # Now let's build a single .hpp
    # We'll do something like:
    #
    # #pragma once
    # #includes ...
    # namespace device_<device_id> {
    #   namespace schedule_<sch_id> {
    #       void run_pipeline(...) { ... }
    #   }
    #   namespace schedule_<sch_id2> {
    #       ...
    #   }
    # }

    final_lines = []
    final_lines.append(f"// Aggregated schedules for device: {device_id_filter}, application: {app_filter}")
    final_lines.append("#pragma once")
    final_lines.append("")
    final_lines.append("#include <queue>")
    final_lines.append("#include <thread>")
    final_lines.append("#include <concurrentqueue.h>")
    final_lines.append('#include "../task.hpp"')
    final_lines.append('#include "../templates.hpp"')
    final_lines.append('#include "../run_stages.hpp"')
    final_lines.append("")
    final_lines.append(f"namespace device_{device_id_filter} {{")
    final_lines.append("")

    # Insert each schedule as a sub-namespace
    for sch_id, code in schedule_map.items():
        final_lines.append(f"namespace schedule_{sch_id} {{")
        final_lines.append("")
        final_lines.append(code)
        final_lines.append("")
        final_lines.append(f"}}  // namespace schedule_{sch_id}")
        final_lines.append("")

    final_lines.append(f"}}  // namespace device_{device_id_filter}")
    final_lines.append("")

    # Write out
    with open(out_hpp_path, "w") as f:
        f.write("\n".join(final_lines))

    print(f"[+] Wrote aggregated file: {out_hpp_path}")


if __name__ == "__main__":
    main()
