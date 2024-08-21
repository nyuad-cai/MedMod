from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(49297)  # Set seed for reproducibility
from tqdm import tqdm


def process_partition(args, partition, eps=1e-6, n_hours=48):
    """
    Process a specific data partition (train or test) to create time-series data samples for mortality prediction.

    Args:
        args (argparse.Namespace): Command-line arguments containing paths for root and output directories.
        partition (str): The data partition to process ('train' or 'test').
        eps (float): Small epsilon value to handle edge cases in time comparisons.
        n_hours (float): Minimum length (in hours) of time series to consider.

    Outputs:
        Creates CSV files with time-series data and a listfile for each partition.
    """
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []  # List to hold (filename, icustay, mortality) tuples
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # Skip empty label files
                if label_df.shape[0] == 0:
                    continue
                icustay = label_df['Icustay'].iloc[0]
                
                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # Length of stay in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                # Skip if length of stay is shorter than the minimum required
                if los < n_hours - eps:
                    continue

                # Read and filter time-series data
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # Skip if no events in the time window
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                # Write filtered time-series data to output file
                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                # Append (filename, icustay, mortality) tuples
                xy_pairs.append((output_ts_filename, icustay, mortality))

    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)  # Shuffle training data
    if partition == "test":
        xy_pairs = sorted(xy_pairs)  # Sort test data

    # Write listfile with sample information
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for (x, icustay, y) in xy_pairs:
            listfile.write('{},0,{},{:d}\n'.format(x, icustay, y))


def main():
    """
    Main function to set up command-line arguments and process both train and test partitions.
    """
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()