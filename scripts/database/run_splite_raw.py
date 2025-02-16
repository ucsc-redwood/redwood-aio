import re
import sys
import os
from collections import defaultdict

def split_logs(log_text):
    """
    Splits the log_text into chunks based on headers that indicate the device.
    Each header is expected to match the pattern:
       "[<any>/<any>] Running <program> on device: <device>"
    """
    # This regex looks for lines starting with [x/y] Running ... on device: <device>
    header_pattern = re.compile(r"\[\d+/\d+\]\s+Running\s+.*?\s+on device:\s+(\S+)")
    
    # Find all header matches in the text
    matches = list(header_pattern.finditer(log_text))
    
    # Dictionary to hold chunks by device name
    logs_by_device = defaultdict(list)
    
    # Iterate over each header match
    for i, match in enumerate(matches):
        device = match.group(1)
        start_index = match.start()
        # Determine the end index: either the start of the next match or the end of the file
        end_index = matches[i+1].start() if i + 1 < len(matches) else len(log_text)
        chunk = log_text[start_index:end_index].strip()
        logs_by_device[device].append(chunk)
    
    return logs_by_device

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_splite_raw.py <path_to_log_file>")
        sys.exit(1)
        
    log_file_path = sys.argv[1]
    output_dir = os.path.dirname(log_file_path)
    
    try:
        with open(log_file_path, "r") as infile:
            log_text = infile.read()
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Split the logs into chunks per device
    device_logs = split_logs(log_text)
    
    # Write each device's logs to a separate file named <device>.txt in the same directory
    for device, chunks in device_logs.items():
        output_filename = os.path.join(output_dir, f"{device}.txt")
        with open(output_filename, "w") as outfile:
            outfile.write("\n\n".join(chunks))
        print(f"Wrote logs for device {device} to {output_filename}")
