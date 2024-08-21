from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.stats import skew

# List of statistical functions to be applied to data
all_functions = [min, max, np.mean, np.std, skew, len]

# Mapping of function sets based on user preferences
functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

# Mapping of different time periods within the data to specific ranges
periods_map = {
    "all": (0, 0, 1, 0),  # Full period
    "first4days": (0, 0, 0, 4 * 24),  # First 4 days
    "first8days": (0, 0, 0, 8 * 24),  # First 8 days
    "last12hours": (1, -12, 1, 0),  # Last 12 hours
    "first25percent": (2, 25),  # First 25% of the time series
    "first50percent": (2, 50)  # First 50% of the time series
}

# Sub-periods used within the full period for feature extraction
sub_periods = [
    (2, 100),  # Full period
    (2, 10),  # First 10%
    (2, 25),  # First 25%
    (2, 50),  # First 50%
    (3, 10),  # Last 10%
    (3, 25),  # Last 25%
    (3, 50)  # Last 50%
]


def get_range(begin, end, period):
    """
    Calculate the start and end times of a given period within the time series.
    :param begin: The start time of the full time series.
    :param end: The end time of the full time series.
    :param period: A tuple representing the period type and range.
    :return: A tuple representing the start and end times of the specified period.
    """
    # Handle percentage-based periods
    if period[0] == 2:  # First p%
        return (begin, begin + (end - begin) * period[1] / 100.0)
    if period[0] == 3:  # Last p%
        return (end - (end - begin) * period[1] / 100.0, end)

    # Handle fixed periods
    L = begin + period[1] if period[0] == 0 else end + period[1]
    R = begin + period[3] if period[2] == 0 else end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    """
    Calculate statistical features for a specified period and sub-period within the time series.
    :param channel_data: The time series data for a specific channel.
    :param period: The full period over which to calculate features.
    :param sub_period: The sub-period within the full period to focus on.
    :param functions: The list of statistical functions to apply.
    :return: An array of calculated features.
    """
    if len(channel_data) == 0:
        return np.full((len(functions),), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions),), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions):
    """
    Extract features for a single episode of time series data.
    :param data_raw: The raw time series data for a single episode.
    :param period: The full period over which to calculate features.
    :param functions: The list of statistical functions to apply.
    :return: A concatenated array of features for all channels and sub-periods.
    """
    global sub_periods
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)


def extract_features(data_raw, period, features):
    """
    Extract features for multiple episodes of time series data.
    :param data_raw: A list of time series data, one for each episode.
    :param period: The period within the time series to focus on.
    :param features: The set of statistical functions to apply.
    :return: A numpy array of extracted features for each episode.
    """
    period = periods_map[period]
    functions = functions_map[features]
    return np.array([extract_features_single_episode(x, period, functions)
                     for x in data_raw])