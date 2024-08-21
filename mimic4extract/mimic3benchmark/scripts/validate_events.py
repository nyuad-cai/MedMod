
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
from tqdm import tqdm

def is_subject_folder(x):
    """
    Checks if a given directory name is a numeric subject folder.

    Args:
        x (str): Directory name.

    Returns:
        bool: True if the directory name is numeric, False otherwise.
    """
    return str.isdigit(x)

def main():
    """
    Processes and cleans event data for each subject. This script ensures consistency 
    between 'events.csv' and 'stays.csv' files for each subject. It performs the following checks:
    - Ensures no empty or duplicated IDs.
    - Validates and attempts to recover missing stay IDs.
    - Removes invalid events based on hospital admission and stay IDs.

    Command line arguments:
    - `subjects_root_path`: Directory containing subject subdirectories with 'stays.csv' and 'events.csv' files.
    """
    # Initialize counters for various data issues
    n_events = 0                   # Total number of events processed
    empty_hadm = 0                 # Number of events with empty hadm_id
    no_hadm_in_stay = 0            # Number of events with hadm_id not in stays.csv
    no_icustay = 0                 # Number of events with empty stay_id in events.csv
    recovered = 0                  # Number of empty stay_ids recovered from stays.csv
    could_not_recover = 0          # Number of empty stay_ids that could not be recovered
    icustay_missing_in_stays = 0   # Number of events with stay_id not in stays.csv

    parser = argparse.ArgumentParser(description='Process and clean event data for each subject.')
    parser.add_argument('subjects_root_path', type=str,
                        help='Directory containing subject subdirectories.')
    args = parser.parse_args()
    print(args)

    # List subject directories and filter numeric ones
    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))

    for subject in tqdm(subjects, desc='Iterating over subjects'):
        stays_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'stays.csv'))
        
        # Ensure no empty or duplicate stay_id or hadm_id in stays.csv
        assert(not stays_df['stay_id'].isnull().any())
        assert(not stays_df['hadm_id'].isnull().any())
        assert(len(stays_df['stay_id'].unique()) == len(stays_df['stay_id']))
        assert(len(stays_df['hadm_id'].unique()) == len(stays_df['hadm_id']))

        events_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'))
        n_events += events_df.shape[0]

        # Drop events with empty hadm_id
        empty_hadm += events_df['hadm_id'].isnull().sum()
        events_df = events_df.dropna(subset=['hadm_id'])

        # Merge events with stays to check for valid hadm_id and stay_id
        merged_df = events_df.merge(stays_df, left_on=['hadm_id'], right_on=['hadm_id'],
                                    how='left', suffixes=['', '_r'], indicator=True)

        # Drop events where hadm_id is not in stays.csv
        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']

        # Attempt to recover missing stay_id from stays.csv
        cur_no_icustay = merged_df['stay_id'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'stay_id'] = merged_df['stay_id'].fillna(merged_df['stay_id_r'])
        recovered += cur_no_icustay - merged_df['stay_id'].isnull().sum()
        could_not_recover += merged_df['stay_id'].isnull().sum()
        merged_df = merged_df.dropna(subset=['stay_id'])

        # Drop events where stay_id does not match stays.csv
        icustay_missing_in_stays += (merged_df['stay_id'] != merged_df['stay_id_r']).sum()
        merged_df = merged_df[(merged_df['stay_id'] == merged_df['stay_id_r'])]

        # Write cleaned events to output CSV
        to_write = merged_df[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']]
        to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)

    # Output summary of data issues
    assert(could_not_recover == 0)
    print('n_events: {}'.format(n_events))
    print('empty_hadm: {}'.format(empty_hadm))
    print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
    print('no_icustay: {}'.format(no_icustay))
    print('recovered: {}'.format(recovered))
    print('could_not_recover: {}'.format(could_not_recover))
    print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))

if __name__ == "__main__":
    main()