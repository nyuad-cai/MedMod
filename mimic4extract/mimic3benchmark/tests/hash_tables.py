from __future__ import absolute_import
from __future__ import print_function

from tqdm import tqdm

import pickle as pkl
import hashlib
import os
import argparse
import pandas as pd

def formatter(x):
    """
    Formats a value by converting it to a float with one decimal place, if possible.

    Args:
        x: The value to format.

    Returns:
        str: The formatted value as a string. Returns the original value if it cannot be converted to float.
    """
    try:
        x = float(x)
        return '{:.1f}'.format(x)
    except:
        return x

def main():
    """
    Recursively hashes all CSV files in the specified directory and saves the hashes to a file.

    Command line arguments:
    - `--directory`, `-d`: The directory to hash.
    - `--output_file`, `-o`: The file to save the hashes (default is 'hashes.pkl').
    """
    parser = argparse.ArgumentParser(description='Recursively produces hashes for all tables inside this directory')
    parser.add_argument('--directory', '-d', type=str, required=True, help='The directory to hash.')
    parser.add_argument('--output_file', '-o', type=str, default='hashes.pkl', help='File to save the hashes.')
    args = parser.parse_args()
    print(args)

    # Count the total number of files in the directory
    total = 0
    for subdir, dirs, files in tqdm(os.walk(args.directory), desc='Counting directories'):
        total += len(files)

    # Change to the target directory
    initial_dir = os.getcwd()
    os.chdir(args.directory)

    # Initialize a dictionary to store file hashes
    hashes = {}
    pbar = tqdm(total=total, desc='Iterating over files')
    
    for subdir, dirs, files in os.walk('.'):
        for file in files:
            pbar.update(1)
            # Process only CSV files
            extension = file.split('.')[-1]
            if extension != 'csv':
                continue

            full_path = os.path.join(subdir, file)
            df = pd.read_csv(full_path, index_col=False)

            # Convert all numerical values to floats with fixed precision
            for col in df.columns:
                df[col] = df[col].apply(formatter)

            # Sort by the first column with unique values
            for col in df.columns:
                if len(df[col].unique()) == len(df):
                    df = df.sort_values(by=col).reset_index(drop=True)
                    break

            # Convert the DataFrame to string and compute its hash
            df_str = df.to_string().encode()
            hashcode = hashlib.md5(df_str).hexdigest()
            hashes[full_path] = hashcode
    
    pbar.close()

    # Return to the initial directory and save the hash results
    os.chdir(initial_dir)
    with open(args.output_file, 'wb') as f:
        pkl.dump(hashes, f)

if __name__ == "__main__":
    main()