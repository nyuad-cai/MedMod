from __future__ import absolute_import
from __future__ import print_function

import csv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from mimic3benchmark.util import dataframe_from_csv

def read_patients_table(path):
    """Read and process the patients table.

    Args:
        path (str): File path to the CSV file containing patient data.

    Returns:
        pd.DataFrame: DataFrame with columns ['subject_id', 'gender', 'anchor_age', 'dod'].
                      'dod' (date of death) is converted to datetime format.
    """
    pats = pd.read_csv(path)  # Read the CSV file into a DataFrame
    columns = ['subject_id', 'gender', 'anchor_age', 'dod']  # Columns to keep
    pats = pats[columns]  # Filter DataFrame to include only the specified columns
    pats.dod = pd.to_datetime(pats.dod)  # Convert 'dod' column to datetime format
    return pats  # Return the processed DataFrame


def read_admissions_table(path):
    """Read and process the admissions table.

    Args:
        path (str): File path to the CSV file containing admissions data.

    Returns:
        pd.DataFrame: DataFrame with columns ['subject_id', 'hadm_id', 'admittime', 
                      'dischtime', 'deathtime', 'ethnicity']. All time-related columns
                      are converted to datetime format.
    """
    admits = pd.read_csv(path)  # Read the CSV file into a DataFrame
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'ethnicity']]  # Select relevant columns
    admits.admittime = pd.to_datetime(admits.admittime)  # Convert 'admittime' to datetime
    admits.dischtime = pd.to_datetime(admits.dischtime)  # Convert 'dischtime' to datetime
    admits.deathtime = pd.to_datetime(admits.deathtime)  # Convert 'deathtime' to datetime
    return admits  # Return the processed DataFrame


def read_icustays_table(path):
    """Read and process the ICU stays table.

    Args:
        path (str): File path to the CSV file containing ICU stays data.

    Returns:
        pd.DataFrame: DataFrame with ICU stay data where 'intime' and 'outtime'
                      columns are converted to datetime format.
    """
    stays = pd.read_csv(path)  # Read the CSV file into a DataFrame
    stays.intime = pd.to_datetime(stays.intime)  # Convert 'intime' to datetime
    stays.outtime = pd.to_datetime(stays.outtime)  # Convert 'outtime' to datetime
    return stays  # Return the processed DataFrame


def read_icd_diagnoses_table(path):
    """Read and process the ICD diagnoses table.

    Args:
        path (str): Directory path containing 'd_icd_diagnoses.csv' and 'diagnoses_icd.csv'.

    Returns:
        pd.DataFrame: Merged DataFrame of diagnosis codes and their descriptions,
                      along with patient-specific diagnosis information.
    """
    codes = pd.read_csv(f'{path}/d_icd_diagnoses.csv')  # Read ICD code descriptions
    codes = codes[['icd_code', 'long_title']]  # Keep relevant columns
    diagnoses = pd.read_csv(f'{path}/diagnoses_icd.csv')  # Read patient diagnoses data
    diagnoses = diagnoses.merge(codes, how='inner', left_on='icd_code', right_on='icd_code')  # Merge on 'icd_code'
    diagnoses[['subject_id', 'hadm_id', 'seq_num']] = diagnoses[['subject_id', 'hadm_id', 'seq_num']].astype(int)  # Ensure integer types
    return diagnoses  # Return the processed DataFrame


def read_events_table_by_row(mimic3_path, table):
    """Generator function to read events table row by row.

    Args:
        mimic3_path (str): Base directory path to the MIMIC-III dataset.
        table (str): Name of the table to read ('chartevents', 'labevents', 'outputevents').

    Yields:
        tuple: A tuple containing the row (dict), row number (int), and total number of rows (int).
    """
    nb_rows = {'chartevents': 329499788, 'labevents': 122103667, 'outputevents': 4457381}  # Row counts for each table
    csv_files = {'chartevents': 'icu/chartevents.csv', 'labevents': 'hosp/labevents.csv', 'outputevents': 'icu/outputevents.csv'}  # File paths
    reader = csv.DictReader(open(os.path.join(mimic3_path, csv_files[table.lower()]), 'r'))  # CSV reader

    for i, row in enumerate(reader):
        if 'stay_id' not in row:  # Check if 'stay_id' is missing in the row
            row['stay_id'] = ''  # Assign an empty string if missing
        yield row, i, nb_rows[table.lower()]  # Yield the row, index, and total number of rows


def count_icd_codes(diagnoses, output_path=None):
    """Count the occurrence of each ICD code.

    Args:
        diagnoses (pd.DataFrame): DataFrame containing ICD diagnoses data.
        output_path (str, optional): If provided, the results will be saved to a CSV file.

    Returns:
        pd.DataFrame: DataFrame with ICD codes, their descriptions, and their count, 
                      sorted by the count in descending order.
    """
    codes = diagnoses[['icd_code', 'long_title']].drop_duplicates().set_index('icd_code')  # Remove duplicates and set 'icd_code' as index
    codes['COUNT'] = diagnoses.groupby('icd_code')['stay_id'].count()  # Count occurrences of each ICD code
    codes.COUNT = codes.COUNT.fillna(0).astype(int)  # Fill missing counts with 0 and convert to integer
    codes = codes[codes.COUNT > 0]  # Filter out codes with zero count

    if output_path:  # If an output path is provided
        codes.to_csv(output_path, index_label='icd_code')  # Save the results to a CSV file

    return codes.sort_values('COUNT', ascending=False).reset_index()  # Return the sorted DataFrame


def remove_icustays_with_transfers(stays):
    """Remove ICU stays with patient transfers between care units.

    Args:
        stays (pd.DataFrame): DataFrame containing ICU stays data.

    Returns:
        pd.DataFrame: Filtered DataFrame where stays with transfers are removed.
    """
    stays = stays[(stays.first_careunit == stays.last_careunit)]  # Keep stays where first and last care units are the same
    return stays[['subject_id', 'hadm_id', 'stay_id', 'last_careunit', 'intime', 'outtime', 'los']]  # Return relevant columns


def merge_on_subject(table1, table2):
    """Merge two tables on 'subject_id'.

    Args:
        table1 (pd.DataFrame): First DataFrame to merge.
        table2 (pd.DataFrame): Second DataFrame to merge.

    Returns:
        pd.DataFrame: Merged DataFrame based on 'subject_id'.
    """
    return table1.merge(table2, how='inner', left_on=['subject_id'], right_on=['subject_id'])  # Inner merge on 'subject_id'


def merge_on_subject_admission(table1, table2):
    """Merge two tables on 'subject_id' and 'hadm_id'.

    Args:
        table1 (pd.DataFrame): First DataFrame to merge.
        table2 (pd.DataFrame): Second DataFrame to merge.

    Returns:
        pd.DataFrame: Merged DataFrame based on 'subject_id' and 'hadm_id'.
    """
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])  # Inner merge on 'subject_id' and 'hadm_id'


def add_age_to_icustays(stays):
    """Add patient age to ICU stays.

    Args:
        stays (pd.DataFrame): DataFrame containing ICU stays data.

    Returns:
        pd.DataFrame: Updated DataFrame with an added 'age' column. If age is negative, it is set to 90.
    """
    stays['age'] = stays.anchor_age  # Assign 'anchor_age' to the new 'age' column
    stays.loc[stays.age < 0, 'age'] = 90  # Set negative ages to 90 (age estimation for patients > 89 years)
    return stays  # Return the updated DataFrame

# ----------------------------------------------
def add_inhospital_mortality_to_icustays(stays):

    mortality = stays.dod.notnull() & ((stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime)))
    stays['mortality'] = mortality.astype(int)
    stays['mortality_inhospital'] = stays['mortality']
    # INHOSPITAL
    return stays


def add_inunit_mortality_to_icustays(stays):

    mortality = stays.dod.notnull() & ((stays.intime <= stays.dod) & (stays.outtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime)))
    stays['mortality_inunit'] = mortality.astype(int)
    # mortality_INUNIT
    return stays


def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):

    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    to_keep = to_keep[(to_keep.stay_id >= min_nb_stays) & (to_keep.stay_id <= max_nb_stays)][['hadm_id']]
    stays = stays.merge(to_keep, how='inner', left_on='hadm_id', right_on='hadm_id')
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):

    stays = stays[(stays.age >= min_age) & (stays.age <= max_age)]
    return stays

    # import pdb; pdb.set_trace()

def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates(), how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays[stays.subject_id == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses[diagnoses.subject_id == subject_id].sort_values(by=['stay_id', 'seq_num'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)


def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    obs_header = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    # nb_rows_dict = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    nb_rows_dict = {'chartevents': 329499788, 'labevents': 122103667, 'outputevents': 4457381}
    
    nb_rows = nb_rows_dict[table.lower()]
    # import pdb;pdb.set_trace()

    for row, row_no, _ in tqdm(read_events_table_by_row(mimic3_path, table), total=nb_rows,
                                                        desc='Processing {} table'.format(table)):

        if (subjects_to_keep is not None) and (row['subject_id'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['itemid'] not in items_to_keep):
            continue
        
        # import pdb; pdb.set_trace()
        # value = row['valueuom'] if table=='OUTPUTEVENTS' else row['valuenum']
        row_out = {'subject_id': row['subject_id'],
                   'hadm_id': row['hadm_id'],
                   'stay_id': '' if 'stay_id' not in row else row['stay_id'],
                   'charttime': row['charttime'],
                   'itemid': row['itemid'],
                   'value': row['value'],
                   'valuenum': row['valueuom'] if table=='OUTPUTEVENTS' else row['valuenum']}
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['subject_id']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['subject_id']

    if data_stats.curr_subject_id != '':
        write_current_observations()
