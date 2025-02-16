import json
import sys
from typing import Dict, Any
import argparse

def load_config(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_combinations(device_config: Dict[str, Any], app_config: Dict[str, Any]) -> int:
    """
    Calculate total number of possible combinations for running an app on a device.
    
    For each stage:
    - Can run on any core type (little/medium/big/gpu)
    - For each core type, can use 1 to N threads where N is the number of cores of that type
    - Can mix and match different configurations for different stages
    """
    num_stages = app_config["num_stages"]
    cores = device_config["pinnable_cores"]
    
    # For each core type, calculate how many thread options we have
    # If we have N cores, we can use 1 to N threads
    options_per_core_type = {}
    for core_type, count in cores.items():
        if count > 0:
            # Can use 1 to count threads
            options_per_core_type[core_type] = count
    
    # For each stage, calculate how many options we have
    options_per_stage = 0
    for core_type, max_threads in options_per_core_type.items():
        # For each core type, we can use 1 to max_threads threads
        options_per_stage += max_threads
    
    # Total combinations is this number raised to the power of stages
    # (because each stage can independently use any option)
    total_combinations = options_per_stage ** num_stages
    
    return total_combinations

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
    
    # Calculate combinations
    device_config = hw_config[args.device]
    app_config = app_config[args.app]
    total_combinations = calculate_combinations(device_config, app_config)
    
    # Print results
    print(f"\nAnalysis for {args.app} on {device_config['name']} ({args.device}):")
    print(f"Number of stages: {app_config['num_stages']}")
    print("Available core types and counts:")
    for core_type, count in device_config['pinnable_cores'].items():
        if count > 0:
            print(f"  - {core_type}: {count} cores (can use 1 to {count} threads)")
    print(f"\nTotal possible combinations: {total_combinations:,}")

if __name__ == "__main__":
    main()
