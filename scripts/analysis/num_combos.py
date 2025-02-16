import json
import sys
from typing import Dict, Any, List, Tuple
import argparse
from collections import defaultdict
import os

def load_config(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_unrestricted_combinations(device_config: Dict[str, Any], app_config: Dict[str, Any]) -> int:
    """Calculate combinations with no restrictions."""
    num_stages = app_config["num_stages"]
    cores = device_config["pinnable_cores"]
    
    options_per_stage = sum(count for count in cores.values() if count > 0)
    return options_per_stage ** num_stages

def calculate_restricted_combinations(device_config: Dict[str, Any], app_config: Dict[str, Any]) -> List[List[Tuple[str, int]]]:
    """
    Calculate and return all valid combinations where:
    1. All processing units must be used at least once
    2. Each stage can use up to max threads available for that core type
    3. Total threads used across stages cannot exceed available cores
    
    Returns a list of schedules, where each schedule is a list of (core_type, thread_count) tuples
    """
    num_stages = app_config["num_stages"]
    cores = device_config["pinnable_cores"]
    valid_schedules = []
    
    def generate_valid_assignments(stage: int, remaining_cores: Dict[str, int], current_schedule: List[Tuple[str, int]]):
        # Base case: all stages assigned
        if stage == num_stages:
            # Check if all core types have been used at least once
            if all(used > 0 for used in core_usage.values()):
                valid_schedules.append(current_schedule[:])
            return
        
        # Try each core type for this stage
        for core_type, max_threads in cores.items():
            if max_threads == 0:
                continue
                
            # For each core type, try using all possible thread counts up to either:
            # 1. The maximum threads available for this core type
            # 2. The remaining threads we have available
            max_possible = min(max_threads, remaining_cores[core_type])
            for threads in range(1, max_possible + 1):
                # Try this assignment
                remaining_cores[core_type] -= threads
                core_usage[core_type] += 1
                current_schedule.append((core_type, threads))
                
                generate_valid_assignments(stage + 1, remaining_cores, current_schedule)
                
                # Backtrack
                remaining_cores[core_type] += threads
                core_usage[core_type] -= 1
                current_schedule.pop()
    
    # Track how many times each core type has been used
    core_usage = defaultdict(int)
    # Track remaining cores of each type
    remaining = {k: v for k, v in cores.items()}
    
    generate_valid_assignments(0, remaining, [])
    return valid_schedules

def format_schedule(schedule: List[Tuple[str, int]], num_stages: int) -> str:
    """Format a schedule for pretty printing."""
    result = []
    for stage_num, (core_type, threads) in enumerate(schedule):
        result.append(f"Stage {stage_num + 1}: {threads} {core_type} core{'s' if threads > 1 else ''}")
    return "\n  ".join(result)

def main():
    parser = argparse.ArgumentParser(description='Calculate possible scheduling combinations.')
    parser.add_argument('--device', required=True, help='Device ID from hardware config')
    parser.add_argument('--app', required=True, help='Application name from application config')
    parser.add_argument('--print-all', action='store_true', help='Print all valid combinations')
    parser.add_argument('--max-print', type=int, default=10, help='Maximum number of combinations to print')
    parser.add_argument('--output', help='Output file path for full schedule list')
    
    args = parser.parse_args()
    
    # Load configurations
    try:
        hw_config = load_config('data/hardware_config.json')
        app_config = load_config('data/application_config.json')
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    
    # Validate inputs
    if args.device not in hw_config:
        print(f"Error: Device '{args.device}' not found in hardware config")
        print(f"Available devices: {', '.join(hw_config.keys())}")
        sys.exit(1)
    
    if args.app not in app_config:
        print(f"Error: Application '{args.app}' not found in application config")
        print(f"Available applications: {', '.join(app_config.keys())}")
        sys.exit(1)
    
    # Get configurations
    device_config = hw_config[args.device]
    app_config = app_config[args.app]
    
    # Calculate combinations
    unrestricted = calculate_unrestricted_combinations(device_config, app_config)
    valid_schedules = calculate_restricted_combinations(device_config, app_config)
    
    # Print results to console
    print(f"\nAnalysis for {args.app} on {device_config['name']} ({args.device}):")
    print(f"Number of stages: {app_config['num_stages']}")
    print("\nAvailable processing units:")
    for core_type, count in device_config['pinnable_cores'].items():
        if count > 0:
            print(f"  - {core_type}: {count} cores (can use 1 to {count} threads)")
    
    print(f"\nUnrestricted combinations: {unrestricted:,}")
    print(f"Restricted combinations: {len(valid_schedules):,}")
    print("\nRestriction rules:")
    print("1. All processing unit types must be used at least once")
    print("2. Each stage can use up to max threads available for that core type")
    print("3. Total threads used across stages cannot exceed available cores")
    
    # Print sample schedules to console
    if args.print_all or len(valid_schedules) <= args.max_print:
        print("\nAll valid schedules:")
        for i, schedule in enumerate(valid_schedules, 1):
            print(f"\nSchedule {i}:")
            print("  " + format_schedule(schedule, app_config["num_stages"]))
    elif len(valid_schedules) > 0:
        print(f"\nFirst {args.max_print} valid schedules:")
        for i, schedule in enumerate(valid_schedules[:args.max_print], 1):
            print(f"\nSchedule {i}:")
            print("  " + format_schedule(schedule, app_config["num_stages"]))
        print(f"\n... and {len(valid_schedules) - args.max_print:,} more combinations")
    
    # Save full schedule list to file if output path is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(args.output, 'w') as f:
            f.write(f"Analysis for {args.app} on {device_config['name']} ({args.device})\n")
            f.write(f"Number of stages: {app_config['num_stages']}\n")
            f.write("\nAvailable processing units:\n")
            for core_type, count in device_config['pinnable_cores'].items():
                if count > 0:
                    f.write(f"  - {core_type}: {count} cores (can use 1 to {count} threads)\n")
            
            f.write(f"\nTotal valid schedules: {len(valid_schedules):,}\n")
            
            for i, schedule in enumerate(valid_schedules, 1):
                f.write(f"\nSchedule {i}:\n")
                f.write("  " + format_schedule(schedule, app_config["num_stages"]) + "\n")
            
        print(f"\nFull schedule list saved to: {args.output}")

if __name__ == "__main__":
    main()
