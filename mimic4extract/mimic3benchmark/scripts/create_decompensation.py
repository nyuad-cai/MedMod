from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
random.seed(49297)  # Set seed for reproducibility
from tqdm import tqdm


def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0,
                      eps=1e-6, future_time_interval=24.0):
    """
    Process a specific data partition (train or test) to create time-series data samples.

    Args:
        args (argparse.Namespace): Command-line arguments containing paths for root and output directories.
        partition (str): The data partition to process ('train' or 'test').
        sample_rate (float): Interval (in hours) at which samples are taken from the time series.
        shortest_length (float): Minimum length (in hours) of time series to consider.
        eps (float): Small epsilon value to handle edge cases in time comparisons.
        future_time_interval (float): Time window (in hours) to determine mortality prediction.

    Outputs:
        Creates CSV files with time-series data and a listfile for each partition.
    """

    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []  # List to hold (filename, time, icustay, mortality) tuples
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # Skip empty label files
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # Length of stay in hours
                if pd.isnull(los):
                    print("(length of stay is missing)", patient, ts_filename)
                    continue
                
                # Get stay information
                stay = stays_df[stays_df.stay_id == label_df.iloc[0]['Icustay']]
                icustay = label_df['Icustay'].iloc[0]
                deathtime = stay['deathtime'].iloc[0]
                intime = stay['intime'].iloc[0]
                if pd.isnull(deathtime):
                    lived_time = 1e18  # Indicate patient is alive
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0

                # Read and filter time-series data
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # Skip if no events in the ICU
                if len(ts_lines) == 0:
                    print("(no events in ICU) ", patient, ts_filename)
                    continue

                # Define sample times
                sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate)
                sample_times = list(filter(lambda x: x > shortest_length, sample_times))
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))

                # Write filtered time-series data to output file
                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                # Append (filename, time, icustay, mortality) tuples
                for t in sample_times:
                    if mortality == 0:
                        cur_mortality = 0
                    else:
                        cur_mortality = int(lived_time - t < future_time_interval)
                    xty_triples.append((output_ts_filename, t, icustay, cur_mortality))

    print("Number of created samples:", len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)  # Shuffle training data
    if partition == "test":
        xty_triples = sorted(xty_triples)  # Sort test data

    # Write listfile with sample information
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for (x, t, icustay, y) in xty_triples:
            listfile.write('{},{:.6f},{},{:d}\n'.format(x, t, icustay, y))


def main():
    """
    Main function to set up command-line arguments and process both train and test partitions.
    """
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()