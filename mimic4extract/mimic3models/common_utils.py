from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import json
import random
from mimic3models.feature_extractor import extract_features


def convert_to_dict(data, header, channel_info):
    """
    Converts raw data into a dictionary format for feature extraction.

    Args:
        data (np.ndarray): 2D array where each row is a timestamp and each column is a feature.
        header (list): List of feature names corresponding to the columns in `data`.
        channel_info (dict): Dictionary containing metadata for each channel (feature).

    Returns:
        list: A list of lists where each inner list contains tuples of (time, value) for a feature.
    """
    ret = [[] for _ in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        # Filter out empty values and pair timestamps with values
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        # Map possible categorical values to their numerical counterparts
        if len(channel_info[channel]['possible_values']) != 0:
            ret[i-1] = [(x[0], channel_info[channel]['values'][x[1]]) for x in ret[i-1]]
        # Convert to float for numerical consistency
        ret[i-1] = [(float(x[0]), float(x[1])) for x in ret[i-1]]
    return ret


def extract_features_from_rawdata(chunk, header, period, features):
    """
    Extracts features from raw data using the specified period and feature set.

    Args:
        chunk (list): List of 2D arrays containing the raw data.
        header (list): List of feature names corresponding to the columns in `chunk`.
        period (float): The period over which to aggregate features.
        features (list): List of features to extract.

    Returns:
        list: A list of extracted features for each input data point.
    """
    channel_info_path = os.path.join(os.path.dirname(__file__), "resources/channel_info.json")
    with open(channel_info_path) as channel_info_file:
        channel_info = json.load(channel_info_file)
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def read_chunk(reader, chunk_size):
    """
    Reads a chunk of data from a reader.

    Args:
        reader: An object that reads data in chunks.
        chunk_size (int): The number of data points to read.

    Returns:
        dict: A dictionary where each key is a feature and each value is a list of values for that feature.
    """
    data = {}
    for _ in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]  # Keep only the first header
    return data

def sort_and_shuffle(data, batch_size):
    """
    Sorts and shuffles data by sequence length, then divides it into batches.

    Args:
        data (tuple): A tuple of arrays or lists to be sorted and shuffled.
        batch_size (int): The number of data points per batch.

    Returns:
        tuple: A tuple of shuffled and batched data arrays or lists.
    """
    assert len(data) >= 2, "Data must have at least two elements."
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=lambda x: x[0].shape[0])  # Sort by the length of the first element

    mas = [head[i: i + batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    return list(zip(*data))

def add_common_arguments(parser):
    """
    Adds common command-line arguments for various tasks.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which common arguments are added.
    """
    parser.add_argument('--network', type=str)
    parser.add_argument('--dim', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--depth', type=int, default=1, help='Number of bi-LSTMs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of chunks to train')
    parser.add_argument('--load_state', type=str, default="", help='Path to load a saved state file')
    parser.add_argument('--mode', type=str, default="train", help='Mode: train or test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
    parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
    parser.add_argument('--save_every', type=int, default=1, help='Save state every x epoch')
    parser.add_argument('--prefix', type=str, default="", help='Optional prefix for network name')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rec_dropout', type=float, default=0.0, help="Dropout rate for recurrent connections")
    parser.add_argument('--batch_norm', type=bool, default=False, help='Batch normalization')
    parser.add_argument('--timestep', type=float, default=1.0, help="Fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--small_part', dest='small_part', action='store_true')
    parser.add_argument('--whole_data', dest='small_part', action='store_false')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='Beta_1 parameter for Adam optimizer')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--size_coef', type=float, default=4.0)
    parser.add_argument('--normalizer_state', type=str, default=None, help='Path to a normalizer state file')

class DeepSupervisionDataLoader:
    """
    Data loader for deep supervision tasks such as decompensation and length of stay prediction.
    Reads all the data for one patient at once.

    Parameters:
        dataset_dir (str): Directory where timeseries files are stored.
        listfile (str, optional): Path to a listfile. Defaults to `dataset_dir/listfile.csv`.
        small_part (bool, optional): If True, loads only a small part of the dataset. Defaults to False.
    """

    def __init__(self, dataset_dir, listfile=None, small_part=False):
        self._dataset_dir = dataset_dir
        listfile_path = listfile or os.path.join(dataset_dir, "listfile.csv")
        
        with open(listfile_path, "r") as lfile:
            self._data = [line.split(',') for line in lfile.readlines()[1:]]  # Skip header

        self._data = [(x, float(t), y) for (x, t, y) in self._data]
        self._data = sorted(self._data)

        mas = {"X": [], "ts": [], "ys": [], "name": []}
        i = 0
        while i < len(self._data):
            j = i
            cur_stay = self._data[i][0]
            cur_ts = []
            cur_labels = []
            while j < len(self._data) and self._data[j][0] == cur_stay:
                cur_ts.append(self._data[j][1])
                cur_labels.append(self._data[j][2])
                j += 1

            cur_X, header = self._read_timeseries(cur_stay)
            mas["X"].append(cur_X)
            mas["ts"].append(cur_ts)
            mas["ys"].append(cur_labels)
            mas["name"].append(cur_stay)

            i = j
            if small_part and len(mas["name"]) == 256:
                break

        self._data = mas

    def _read_timeseries(self, ts_filename):
        """
        Reads the timeseries data for a single patient.

        Args:
            ts_filename (str): Filename of the timeseries data.

        Returns:
            tuple: A tuple containing the timeseries data (np.ndarray) and the header (list).
        """
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                ret.append(np.array(line.strip().split(',')))
        return np.stack(ret), header

def create_directory(directory):
    """
    Creates a directory if it doesn't already exist.

    Args:
        directory (str): The path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def pad_zeros(arr, min_length=None):
    """
    Pads sequences of varying lengths with zeros to equalize their lengths.

    Args:
        arr (list of np.ndarray): A list of arrays to be padded.
        min_length (int, optional): The minimum length to pad the arrays to. Defaults to None.

    Returns:
        tuple: A tuple containing the padded arrays and their original sequence lengths.
    """
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if min_length is not None and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length