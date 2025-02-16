import json
import sys
from typing import Dict, Any
import argparse
from collections import defaultdict

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

def calculate_restricted_combinations(device_config: Dict[str, Any], app_config: Dict[str, Any]) -> int:
    """
    Calculate combinations where:
    1. All processing units must be used
    2. Total threads used across all stages cannot exceed available cores
    3. Same processing unit can be used for multiple stages
    """
    num_stages = app_config["num_stages"]
    cores = device_config["pinnable_cores"]
    
    def count_valid_assignments(stage: int, remaining_cores: Dict[str, int]) -> int:
        # Base case: all stages assigned
        if stage == num_stages:
            # Check if all core types have been used
            return 1 if all(used > 0 for used in core_usage.values()) else 0
        
        total = 0
        # Try each core type and thread count for this stage
        for core_type, max_threads in cores.items():
            if max_threads == 0:
                continue
                
            # Try different thread counts for this stage
            for threads in range(1, max_threads + 1):
                if remaining_cores[core_type] >= threads:
                    # Try this assignment
                    remaining_cores[core_type] -= threads
                    core_usage[core_type] += 1
                    
                    total += count_valid_assignments(stage + 1, remaining_cores)
                    
                    # Backtrack
                    remaining_cores[core_type] += threads
                    core_usage[core_type] -= 1
                    
        return total
    
    # Track how many times each core type has been used
    core_usage = defaultdict(int)
    # Track remaining cores of each type
    remaining = {k: v for k, v in cores.items()}
    
    return count_valid_assignments(0, remaining)

def main():
    parser = argparse.ArgumentParser(description='Calculate possible scheduling combinations.')
    parser.add_argument('--device', required=True, help='Device ID from hardware config')
    parser.add_argument('--app', required=True, help='Application name from application config')
    
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
    
    # Calculate both types of combinations
    unrestricted = calculate_unrestricted_combinations(device_config, app_config)
    restricted = calculate_restricted_combinations(device_config, app_config)
    
    # Print results
    print(f"\nAnalysis for {args.app} on {device_config['name']} ({args.device}):")
    print(f"Number of stages: {app_config['num_stages']}")
    print("\nAvailable processing units:")
    for core_type, count in device_config['pinnable_cores'].items():
        if count > 0:
            print(f"  - {core_type}: {count} cores (can use 1 to {count} threads)")
    
    print(f"\nUnrestricted combinations: {unrestricted:,}")
    print(f"Restricted combinations: {restricted:,}")
    print("\nRestriction rules:")
    print("1. All processing unit types must be used")
    print("2. Total threads used across stages cannot exceed available cores")
    print("3. Same processing unit can be used for multiple stages")

if __name__ == "__main__":
    main()
