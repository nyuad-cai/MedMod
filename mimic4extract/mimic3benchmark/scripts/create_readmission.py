from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(49297)  # Set seed for reproducibility
from tqdm import tqdm
from datetime import datetime, timedelta


def process_partition(args, partition, eps=1e-6, time_limit=30):
    """
    Process a specific data partition (train or test) to create data for re-admission prediction.

    Args:
        args (argparse.Namespace): Command-line arguments containing paths for root and output directories.
        partition (str): The data partition to process ('train' or 'test').
        eps (float): Small epsilon value to handle edge cases in time comparisons.
        time_limit (int): Number of days after which a patient is considered not to be readmitted.

    Outputs:
        Creates a CSV file for each partition containing filenames, length of stay, stay IDs, and readmission labels.
    """
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []  # List to store tuples of (filename, length of stay, stay ID, readmission label)

    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                stay_filename = 'stays.csv'
                stay_df = pd.read_csv(os.path.join(patient_folder, stay_filename))
                stay_df['intime'] = pd.to_datetime(stay_df['intime'], format='%Y-%m-%d %H:%M:%S')
                stay_df = stay_df.sort_values(by='intime')

                # Skip if the label file is empty
                if label_df.shape[0] == 0:
                    continue
                
                icustay = label_df['Icustay'].iloc[0]
                # Get the current stay row 
                curr_stay = stay_df.loc[stay_df['stay_id'] == icustay]
                
                # Calculate length of stay
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                # Read and filter time-series data
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                            

                # Skip if no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                # Write filtered time-series data
                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)
                
                # Determine readmission label
                if stay_df.shape[0] == 1:
                    # If there's only one stay, we automatically assign readmission label 0
                    readmit = 0
                elif stay_df.shape[0] == curr_stay.index[0] + 1:
                    # If this is the last stay, we automatically assign readmission label 0
                    readmit = 0
                else:
                    # Calculate readmission based on the next stay's intime
                    out_time = curr_stay['outtime'].tolist()[0]
                    date_format = "%Y-%m-%d %H:%M:%S"
                    date_time = datetime.strptime(out_time, date_format)
                    end_time = date_time + timedelta(days=time_limit)
                    intime = stay_df['intime'].loc[curr_stay.index[0] + 1]
                    if intime < end_time:
                        readmit = 1
                    else:
                        readmit = 0

                # Append the tuple to xy_pairs
                xy_pairs.append((output_ts_filename, los, icustay, readmit))

    print("Number of created samples:", len(xy_pairs))
    
    # Shuffle or sort based on partition type
    if partition == "train":
        random.shuffle(xy_pairs)
    elif partition == "test":
        xy_pairs = sorted(xy_pairs)

    # Prepare and write the listfile
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for (x, t, icustay, y) in xy_pairs:
            listfile.write('{},0,{},{:d}\n'.format(x, t, icustay, y))


def main():
    """
    Main function to set up command-line arguments and process both train and test partitions.
    """
    parser = argparse.ArgumentParser(description="Create data for re-admission prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()