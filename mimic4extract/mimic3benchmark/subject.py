from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import pandas as pd

from mimic3benchmark.util import dataframe_from_csv


def read_stays(subject_path):
    """Read and preprocess the stays CSV file.
    Args:
        subject_path: Directory path containing the 'stays.csv' file.
    Returns:
        DataFrame with stay information, including datetime conversions and sorting.
    """
    stays = dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.intime = pd.to_datetime(stays.intime)  # Convert 'intime' to datetime
    stays.outtime = pd.to_datetime(stays.outtime)  # Convert 'outtime' to datetime
    # stays.dob = pd.to_datetime(stays.dob) # 'dob' is missing in MIMIC-IV
    stays.dod = pd.to_datetime(stays.dod)  # Convert 'dod' to datetime
    stays.deathtime = pd.to_datetime(stays.deathtime)  # Convert 'deathtime' to datetime
    stays.sort_values(by=['intime', 'outtime'], inplace=True)  # Sort stays by 'intime' and 'outtime'
    return stays


def read_diagnoses(subject_path):
    """Read the diagnoses CSV file.
    Args:
        subject_path: Directory path containing the 'diagnoses.csv' file.
    Returns:
        DataFrame with diagnosis information.
    """
    return dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)


def read_events(subject_path, remove_null=True):
    """Read and preprocess the events CSV file.
    Args:
        subject_path: Directory path containing the 'events.csv' file.
        remove_null: Boolean to indicate if rows with null 'value' should be removed.
    Returns:
        DataFrame with event information, including datetime conversion and type casting.
    """
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events[events.value.notnull()]  # Remove rows with null 'value' if specified
    events.charttime = pd.to_datetime(events.charttime)  # Convert 'charttime' to datetime
    events.hadm_id = events.hadm_id.fillna(value=-1).astype(int)  # Fill null 'hadm_id' with -1 and convert to int
    events.stay_id = events.stay_id.fillna(value=-1).astype(int)  # Fill null 'stay_id' with -1 and convert to int
    events.valuenum = events.valuenum.fillna('').astype(str)  # Fill null 'valuenum' with empty string and convert to str
    # events.sort_values(by=['charttime', 'ITEMID', 'stay_id'], inplace=True) # Optional sorting
    return events


def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    """Filter events for a specific stay and time range.
    Args:
        events: DataFrame with event data.
        icustayid: ICU stay ID to filter events by.
        intime: Optional start time for filtering events.
        outtime: Optional end time for filtering events.
    Returns:
        DataFrame with events for the specified stay and time range.
    """
    idx = (events.stay_id == icustayid)  # Filter by stay ID
    if intime is not None and outtime is not None:
        idx = idx | ((events.charttime >= intime) & (events.charttime <= outtime))  # Filter by time range if provided
    events = events[idx]
    del events['stay_id']  # Remove 'stay_id' column
    return events


def add_hours_elapsed_to_events(events, dt, remove_charttime=True):
    """Add a column for hours elapsed from a given datetime to each event.
    Args:
        events: DataFrame with event data.
        dt: Datetime from which to calculate elapsed hours.
        remove_charttime: Boolean to indicate if 'charttime' column should be removed.
    Returns:
        DataFrame with 'HOURS' column added and optionally 'charttime' removed.
    """
    events = events.copy()
    events['HOURS'] = (events.charttime - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60  # Calculate hours elapsed
    if remove_charttime:
        del events['charttime']  # Remove 'charttime' column if specified
    return events


def convert_events_to_timeseries(events, variable_column='variable', variables=[]):
    """Convert event data to a time series format.
    Args:
        events: DataFrame with event data.
        variable_column: Column name for variables in the events DataFrame.
        variables: List of variables to ensure are in the output DataFrame.
    Returns:
        DataFrame in time series format with variables as columns.
    """
    # Extract and sort metadata for timeseries
    metadata = events[['charttime', 'stay_id']].sort_values(by=['charttime', 'stay_id'])\
                    .drop_duplicates(keep='first').set_index('charttime')
    # Prepare timeseries by pivoting event data
    timeseries = events[['charttime', variable_column, 'value']]\
                    .sort_values(by=['charttime', variable_column, 'value'], axis=0)\
                    .drop_duplicates(subset=['charttime', variable_column], keep='last')
    timeseries = timeseries.pivot(index='charttime', columns=variable_column, values='value')\
                    .merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan  # Ensure specified variables are included in the output DataFrame
    return timeseries


def get_first_valid_from_timeseries(timeseries, variable):
    """Get the first valid value for a specified variable from the timeseries DataFrame.
    Args:
        timeseries: DataFrame in time series format.
        variable: Variable name to retrieve the first valid value for.
    Returns:
        First valid value for the specified variable, or NaN if not found.
    """
    if variable in timeseries:
        idx = timeseries[variable].notnull()  # Find non-null values for the variable
        if idx.any():
            loc = np.where(idx)[0][0]  # Get the location of the first non-null value
            return timeseries[variable].iloc[loc]
    return np.nan