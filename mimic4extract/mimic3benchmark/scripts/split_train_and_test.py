from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse

def move_to_partition(args, patients, partition):
    """
    Moves a list of patient directories to a specified partition directory.

    Args:
        args: Namespace object containing the paths for subjects.
        patients (list of str): List of patient directories to move.
        partition (str): Target partition directory (e.g., 'train', 'test').
    """
    partition_path = os.path.join(args.subjects_root_path, partition)
    os.makedirs(partition_path, exist_ok=True)
    
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(partition_path, patient)
        shutil.move(src, dest)

def main():
    """
    Splits patient data into train and test sets based on a predefined test set.

    The script reads a CSV file that defines which patients belong to the test set. 
    Patients not listed in the test set are assigned to the train set. It then moves 
    patient directories to the appropriate sub-directories ('train' or 'test').

    Arguments:
    - `subjects_root_path`: Directory containing patient sub-directories.
    """
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    test_set = set()

    # Read the test set definitions from a CSV file
    with open(os.path.join(os.path.dirname(__file__), '../resources/testset_iv.csv'), "r") as test_set_file:
        for line in test_set_file:
            patient_id, label = line.strip().split(',')
            if int(label) == 1:
                test_set.add(patient_id)

    # List patient directories
    folders = os.listdir(args.subjects_root_path)
    folders = list(filter(str.isdigit, folders))
    
    # Divide patients into train and test sets
    train_patients = [x for x in folders if x not in test_set]
    test_patients = [x for x in folders if x in test_set]
    
    # Ensure there is no overlap between train and test sets
    assert len(set(train_patients) & set(test_patients)) == 0

    # Move patient directories to the respective partitions
    move_to_partition(args, train_patients, "train")
    move_to_partition(args, test_patients, "test")

if __name__ == '__main__':
    main()

# import pandas as pd
# import numpy as np
# # # initialise data of lists.
# split = np.zeros(len(folders))
# test_inds = np.random.permutation(len(folders))[:int((20*len(folders)) / 100)]
# split[test_inds] = 1
# data = {'subject_id':folders, 'test':split.astype(int).tolist()}
 
# # # Create DataFrame
# df = pd.DataFrame(data)

# path = os.path.join(os.path.dirname(__file__), '../resources/testset_iv.csv')

# # df.to_csv(path)
# df.to_csv(path, header=False, index=False)