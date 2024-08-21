from __future__ import absolute_import
from __future__ import print_function

from mimic3models.metrics import print_metrics_regression
import sklearn.utils as sk_utils
import numpy as np
import pandas as pd
import argparse
import json
import os

def main():
    """Main function to evaluate predictions and compute regression metrics for length of stay."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', type=str, 
                        help='Path to the CSV file containing the predictions.')
    parser.add_argument('--test_listfile', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/test/listfile.csv'),
                        help='Path to the CSV file containing the test data list.')
    parser.add_argument('--n_iters', type=int, default=1000,
                        help='Number of iterations for resampling to compute metrics.')
    parser.add_argument('--save_file', type=str, default='los_results.json',
                        help='Path to the JSON file where results will be saved.')
    args = parser.parse_args()

    # Read prediction and test data from CSV files
    pred_df = pd.read_csv(args.prediction, index_col=False, dtype={'period_length': np.float32,
                                                                   'y_true': np.float32})
    test_df = pd.read_csv(args.test_listfile, index_col=False, dtype={'period_length': np.float32,
                                                                   'y_true': np.float32})

    # Merge prediction data with test data on 'stay' and 'period_length'
    df = test_df.merge(pred_df, left_on=['stay', 'period_length'], right_on=['stay', 'period_length'],
                       how='left', suffixes=['_l', '_r'])
    
    # Assert that there are no missing predictions and that true labels match
    assert (df['prediction'].isnull().sum() == 0)
    assert (df['y_true_l'].equals(df['y_true_r']))

    # Metrics to compute
    metrics = [('Kappa', 'kappa'),
               ('MAD', 'mad'),
               ('MSE', 'mse'),
               ('MAPE', 'mape')]

    # Prepare data for metric computation
    data = np.zeros((df.shape[0], 2))
    data[:, 0] = np.array(df['prediction'])
    data[:, 1] = np.array(df['y_true_l'])

    # Initialize results dictionary
    results = dict()
    results['n_iters'] = args.n_iters
    # Compute metrics on the entire dataset
    ret = print_metrics_regression(data[:, 1], data[:, 0], verbose=0)
    for (m, k) in metrics:
        results[m] = dict()
        results[m]['value'] = ret[k]
        results[m]['runs'] = []

    # Compute metrics for resampled datasets
    for i in range(args.n_iters):
        cur_data = sk_utils.resample(data, n_samples=len(data))
        ret = print_metrics_regression(cur_data[:, 1], cur_data[:, 0], verbose=0)
        for (m, k) in metrics:
            results[m]['runs'].append(ret[k])

    # Calculate statistics (mean, median, std, percentiles) for each metric
    for (m, k) in metrics:
        runs = results[m]['runs']
        results[m]['mean'] = np.mean(runs)
        results[m]['median'] = np.median(runs)
        results[m]['std'] = np.std(runs)
        results[m]['2.5% percentile'] = np.percentile(runs, 2.5)
        results[m]['97.5% percentile'] = np.percentile(runs, 97.5)
        del results[m]['runs']  # Remove the raw runs data

    # Save the results to a JSON file
    print("Saving the results in {} ...".format(args.save_file))
    with open(args.save_file, 'w') as f:
        json.dump(results, f)

    # Print the results
    print(results)

if __name__ == "__main__":
    main()