#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

from codegen_common import (
    read_schedule_file,
    generate_run_pipeline_code,
    build_single_hpp_content,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule_file", required=True, help="Path to a single .json schedule file."
    )
    parser.add_argument(
        "--out_dir", required=True, help="Directory to place the generated .hpp file."
    )
    args = parser.parse_args()

    schedule_path = Path(args.schedule_file)
    if not schedule_path.exists():
        print(f"Error: schedule file not found: {schedule_path}")
        return

    # Extract device and application from path components
    try:
        device_id = schedule_path.parent.parent.name
        application_name = schedule_path.parent.name
        schedule_num = int(re.search(r"schedule_(\d+)", schedule_path.stem).group(1))
        schedule_id = f"{device_id}_{application_name}_schedule_{schedule_num:03d}"
    except (IndexError, AttributeError) as e:
        print(f"Error: Invalid path structure: {e}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read and validate the JSON
    try:
        schedule_obj, _ = read_schedule_file(schedule_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Generate code
    pipeline_code = generate_run_pipeline_code(schedule_obj)

    # Build final .hpp content
    hpp_content = build_single_hpp_content(
        device_id=device_id,
        schedule_id=schedule_id,
        application_name=application_name,
        pipeline_code=pipeline_code,
        schedule_obj=schedule_obj,
    )

    # Write out
    out_hpp_name = f"{schedule_id}.hpp"
    out_hpp_path = out_dir / out_hpp_name
    with open(out_hpp_path, "w") as f:
        f.write(hpp_content)

    print(f"[+] Wrote {out_hpp_path}")


if __name__ == "__main__":
    main()
