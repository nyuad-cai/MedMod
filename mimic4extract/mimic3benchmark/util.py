from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np

def dataframe_from_csv(path, header=0, index_col=0):
    """Read a CSV file into a DataFrame.
    Args:
        path: Path to the CSV file.
        header: Row number(s) to use as the column names, default is 0.
        index_col: Column to set as the index, default is 0.
    Returns:
        DataFrame with data from the CSV file.
    """
    return pd.read_csv(path, header=header, index_col=index_col)


def count_class(values, class_label):
    """Count occurrences of a specific class label in an array.
    Args:
        values: Array of values.
        class_label: The class label to count.
    Returns:
        The number of occurrences of the class label.
    """
    return np.where(values == class_label)[0].shape[0]


def get_data_stats(data_root, dataset='mimic-iii'):
    """Print statistics for the data in training, validation, and test sets.
    Args:
        data_root: Root directory containing the dataset files.
        dataset: Name of the dataset to print statistics for (default is 'mimic-iii').
    Prints:
        Statistics for the overall dataset and for each subset (train, val, test).
    """
    # Read y_true values from CSV files for train, validation, and test sets
    train = pd.read_csv(f'{data_root}/train_listfile.csv').y_true.values
    val = pd.read_csv(f'{data_root}/val_listfile.csv').y_true.values
    test = pd.read_csv(f'{data_root}/test_listfile.csv').y_true.values
    
    # Initialize counters for class occurrences
    total_0 = 0
    total_1 = 0
    
    # Count occurrences of class labels (0 and 1) across all datasets
    total_0 = count_class(train, 0) + count_class(val, 0) + count_class(test, 0)
    total_1 = count_class(train, 1) + count_class(val, 1) + count_class(test, 1)
    
    # Print statistics for the dataset
    print(f'{dataset}')
    print(f'overall 0s {total_0}  1s {total_1}')
    print(f'train  0s {count_class(train, 0)}  1s {count_class(train, 1)}')
    print(f'val  0s {count_class(val, 0)}  1s {count_class(val, 1)}')
    print(f'test  0s {count_class(test, 0)}  1s {count_class(test, 1)}')

# Print statistics for MIMIC-III and MIMIC-IV datasets
get_data_stats('data/decompensation', dataset='mimic-iii')
get_data_stats('dataiv/decompensation', dataset='mimic-iv')