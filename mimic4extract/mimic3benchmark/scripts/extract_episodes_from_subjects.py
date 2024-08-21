from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
from tqdm import tqdm

from mimic3benchmark.subject import (read_stays, read_diagnoses, read_events, get_events_for_stay,
    add_hours_elpased_to_events)
from mimic3benchmark.subject import (convert_events_to_timeseries, get_first_valid_from_timeseries)
from mimic3benchmark.preprocessing import (read_itemid_to_variable_map, map_itemids_to_variables,
    clean_events)
from mimic3benchmark.preprocessing import assemble_episodic_data


def main():
    """
    Extracts and processes episodic data for each subject in a specified directory.
    
    The script iterates through each subject directory, reads the relevant data files, 
    cleans and converts event data to time series, and then extracts and saves episodic data 
    and time series for each stay within the subject.
    """
    parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    parser.add_argument('--variable_map_file', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                        help='CSV file containing ITEMID-to-variable mapping.')
    parser.add_argument('--reference_range_file', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                        help='CSV file containing reference ranges for variables.')
    args, _ = parser.parse_known_args()

    # Load the variable map from the CSV file
    var_map = read_itemid_to_variable_map(args.variable_map_file)
    variables = var_map.variable.unique()

    # Iterate over each subject directory
    for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
        dn = os.path.join(args.subjects_root_path, subject_dir)
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue

        try:
            # Read tables for the current subject
            stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
            diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
            events = read_events(os.path.join(args.subjects_root_path, subject_dir))
        except:
            sys.stderr.write(f'Error reading from disk for subject: {subject_id}\n')
            continue

        # Assemble episodic data
        episodic_data = assemble_episodic_data(stays, diagnoses)

        # Clean and convert event data to time series
        events = map_itemids_to_variables(events, var_map)
        events = clean_events(events)

        if events.shape[0] == 0:
            # Skip if no valid events for this subject
            continue

        timeseries = convert_events_to_timeseries(events, variables=variables)

        # Extract episodes for each stay
        for i in range(stays.shape[0]):
            stay_id = stays.stay_id.iloc[i]
            intime = stays.intime.iloc[i]
            outtime = stays.outtime.iloc[i]

            episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
            if episode.shape[0] == 0:
                # Skip if no data for this episode
                continue

            episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)

            # Update episodic data with weight and height
            if stay_id in episodic_data.index:
                episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
                episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')

            # Save episodic data
            episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                                  f'episode{i+1}.csv'),
                                                                     index_label='Icustay')

            # Save time series data
            columns = list(episode.columns)
            columns_sorted = sorted(columns, key=lambda x: "" if x == "Hours" else x)
            episode = episode[columns_sorted]
            episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, f'episode{i+1}_timeseries.csv'),
                           index_label='Hours')