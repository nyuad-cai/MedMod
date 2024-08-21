from __future__ import absolute_import
from __future__ import print_function

import shutil
import argparse
import os

def main():
    """
    Split the dataset into training and validation sets based on a predefined set of validation patients.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Split train data into train and validation sets.")
    parser.add_argument('dataset_dir', type=str, help='Path to the directory containing the dataset')
    args = parser.parse_args()

    # Load validation patients from a file
    val_patients = set()
    valset_path = os.path.join(os.path.dirname(__file__), 'resources/valset_iv.csv')
    with open(valset_path, 'r') as valset_file:
        for line in valset_file:
            x, y = line.strip().split(',')
            if int(y) == 1:
                val_patients.add(x)

    # Read the listfile and separate into train and validation sets
    listfile_path = os.path.join(args.dataset_dir, 'train/listfile.csv')
    with open(listfile_path) as listfile:
        lines = listfile.readlines()
        header = lines[0]
        data_lines = lines[1:]

    # Split lines into train and validation sets
    train_lines = [line for line in data_lines if line.split('_')[0] not in val_patients]
    val_lines = [line for line in data_lines if line.split('_')[0] in val_patients]
    assert len(train_lines) + len(val_lines) == len(data_lines), "Mismatch in the number of lines"

    # Write the split listfiles
    train_listfile_path = os.path.join(args.dataset_dir, 'train_listfile.csv')
    val_listfile_path = os.path.join(args.dataset_dir, 'val_listfile.csv')

    with open(train_listfile_path, 'w') as train_listfile:
        train_listfile.write(header)
        train_listfile.writelines(train_lines)

    with open(val_listfile_path, 'w') as val_listfile:
        val_listfile.write(header)
        val_listfile.writelines(val_lines)

    # Copy the test listfile to a new location
    test_listfile_src = os.path.join(args.dataset_dir, 'test/listfile.csv')
    test_listfile_dst = os.path.join(args.dataset_dir, 'test_listfile.csv')
    shutil.copy(test_listfile_src, test_listfile_dst)

if __name__ == '__main__':
    main()