#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

from codegen_common import (
    read_schedule_file,
    generate_run_pipeline_code,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", required=True, help="Directory with multiple .json schedule files."
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to place the single aggregated .hpp file.",
    )
    parser.add_argument(
        "--device", required=True, help="Device ID to filter, e.g. 'jetson'."
    )
    parser.add_argument(
        "--application",
        required=True,
        choices=["Tree", "CifarDense", "CifarSparse"],
        help="App name to filter on.",
    )
    parser.add_argument(
        "--out_name",
        default="all_schedules.hpp",
        help="Name of the final aggregated .hpp output.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir) / args.device / args.application
    if not in_dir.is_dir():
        print(f"Error: input directory not found: {in_dir}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_filter = args.device
    app_filter = args.application
    out_hpp_name = args.out_name
    out_hpp_path = out_dir / out_hpp_name

    # We'll store a dict: { schedule_id : generated_code }
    schedules_code = {}

    # Collect matching schedules - simplified since we're already in device/app directory
    for json_file in in_dir.glob("schedule_*.json"):
        try:
            schedule_obj, _ = read_schedule_file(json_file)
        except ValueError as e:
            print(f"Skipping {json_file}: {e}")
            continue

        code = generate_run_pipeline_code(schedule_obj)
        # Extract schedule number from filename for consistent ordering
        schedule_num = int(re.search(r"schedule_(\d+)", json_file.stem).group(1))
        schedule_id = f"{args.device}_{args.application}_schedule_{schedule_num:03d}"
        schedules_code[schedule_id] = code

    if not schedules_code:
        print(
            f"No schedules found for device='{device_filter}', app='{app_filter}' in {in_dir}"
        )
        return

    # Sort schedule IDs for consistent ordering
    sorted_schedule_ids = sorted(schedules_code.keys())

    # Build the final aggregated .hpp
    lines = []
    lines.append(
        f"// Aggregated schedules for device: {device_filter}, application: {app_filter}"
    )
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <thread>")
    lines.append("#include <chrono>")
    lines.append("#include <concurrentqueue.h>")
    lines.append('#include <spdlog/spdlog.h>')
    lines.append("")
    lines.append('#include "../task.hpp"')
    lines.append('#include "../../templates.hpp"')
    lines.append('#include "../run_stages.hpp"')
    lines.append('#include "builtin-apps/common/cuda/manager.cuh"')
    
    # Determine the app-specific include based on app_filter
    app_name_lower = app_filter.lower()
    if "cifardense" in app_name_lower:
        lines.append('#include "builtin-apps/cifar-dense/dense_appdata.hpp"')
    elif "cifarsparse" in app_name_lower:
        lines.append('#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"')
    elif "tree" in app_name_lower:
        lines.append('#include "builtin-apps/tree/tree_appdata.hpp"')
    else:
        lines.append(f'// Warning: Unknown application type: {app_filter}')
    
    lines.append("")
    lines.append(f"namespace device_{device_filter} {{")
    lines.append("")
    
    # Add the AppData typedef based on app_filter
    if "cifardense" in app_name_lower:
        lines.append("using AppData = cifar_dense::AppData;")
    elif "cifarsparse" in app_name_lower:
        lines.append("using AppData = cifar_sparse::AppData;")
    elif "tree" in app_name_lower:
        lines.append("using AppData = tree::AppData;")
    else:
        lines.append(f'// Warning: No AppData typedef available for: {app_filter}')
    
    lines.append("")

    # Insert each schedule as a sub-namespace
    for sch_id in sorted_schedule_ids:
        code = schedules_code[sch_id]
        lines.append(f"namespace schedule_{sch_id} {{")
        lines.append("")
        lines.append(code)
        lines.append("")
        lines.append(f"}}  // namespace schedule_{sch_id}")
        lines.append("")

    # Define a function pointer table and get_num_schedules
    lines.append(
        "// --------------------------------------------------------------------------"
    )
    lines.append("// Define function pointer type for run_pipeline")
    lines.append(
        "using RunPipelineFunc = void (*)(int);"
    )
    lines.append("")
    lines.append("// Array of function pointers to all run_pipeline implementations")
    lines.append("static const RunPipelineFunc run_pipeline_table[] = {")
    for sch_id in sorted_schedule_ids:
        lines.append(f"    schedule_{sch_id}::run_pipeline,")
    lines.append("};")
    lines.append("")
    lines.append("[[nodiscard]] constexpr int get_num_schedules() {")
    lines.append(
        "    return sizeof(run_pipeline_table) / sizeof(run_pipeline_table[0]);"
    )
    lines.append("}")
    lines.append("")

  # Add the additional get_run_pipeline_func function
    lines.append("[[nodiscard]] inline RunPipelineFunc get_run_pipeline_func(const int schedule_id) {")
    lines.append("    if (schedule_id < 1 || schedule_id > get_num_schedules()) {")
    lines.append('        spdlog::error("Invalid schedule ID: {}", schedule_id);')
    lines.append('        throw std::invalid_argument("Invalid schedule ID");')
    lines.append("    }")
    lines.append("    return run_pipeline_table[schedule_id - 1];")
    lines.append("}")
    lines.append("")

    lines.append(f"}}  // namespace device_{device_filter}")
    lines.append("")

    with open(out_hpp_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[+] Wrote aggregated file: {out_hpp_path}")


if __name__ == "__main__":
    main()
