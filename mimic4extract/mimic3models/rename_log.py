from __future__ import absolute_import
from __future__ import print_function

import os
import re
import argparse

def main():
    """
    Main function to process log files:
    - Skips files that are already renamed or directories.
    - Renames log files based on the `model.final_name` found in the log content.
    """
    parser = argparse.ArgumentParser(description="Rename log files based on their content.")
    parser.add_argument('log', type=str, nargs='+', help="List of log files to process.")
    args = parser.parse_args()

    if not isinstance(args.log, list):
        args.log = [args.log]

    for log in args.log:
        if "renamed" in log:
            print(f"{log} is already renamed by hand, skipping...")
            continue
        
        if os.path.isdir(log):
            print(f"{log} is a directory, skipping...")
            continue
        
        try:
            with open(log, 'r') as logfile:
                text = logfile.read()
                match = re.search("==> model.final_name: (.*)\n", text)
                if match is None:
                    print(f"No model.final_name in log file: {log}. Skipping...")
                    continue
                name = match.group(1)
        except IOError as e:
            print(f"Error reading file {log}: {e}")
            continue
        
        dirname = os.path.dirname(log)
        new_path = os.path.join(dirname, f"{name}.log")
        try:
            os.rename(log, new_path)
            print(f"Renamed {log} to {new_path}")
        except OSError as e:
            print(f"Error renaming file {log} to {new_path}: {e}")

if __name__ == '__main__':
    main()