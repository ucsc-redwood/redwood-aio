import sys
from pathlib import Path
import importlib


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts <script_name> [args...]")
        sys.exit(1)

    script_name = sys.argv[1]
    script_args = sys.argv[2:]

    # Update sys.argv for the script
    sys.argv = [script_name] + script_args

    # Import and run the script
    try:
        if script_name.endswith(".py"):
            script_name = script_name[:-3]
        module_path = script_name.replace("/", ".")
        if module_path.startswith("scripts."):
            module_path = module_path[8:]
        module = importlib.import_module(f"scripts.{module_path}")
        if hasattr(module, "main"):
            module.main()
        else:
            print(f"Error: {script_name} has no main() function")
            sys.exit(1)
    except ImportError as e:
        print(f"Error: Could not import {script_name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
