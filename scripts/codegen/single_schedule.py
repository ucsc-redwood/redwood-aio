#!/usr/bin/env python3
import argparse
from pathlib import Path

from codegen_common import (
    parse_schedule_filename,
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse from filename
    try:
        device_id, application_name, schedule_id = parse_schedule_filename(
            schedule_path.name
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

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
    )

    # Write out
    out_hpp_name = f"{schedule_id}.hpp"
    out_hpp_path = out_dir / out_hpp_name
    with open(out_hpp_path, "w") as f:
        f.write(hpp_content)

    print(f"[+] Wrote {out_hpp_path}")


if __name__ == "__main__":
    main()
