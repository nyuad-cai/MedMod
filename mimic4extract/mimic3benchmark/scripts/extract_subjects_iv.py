from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import yaml

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import (add_hcup_ccs_2015_groups, make_phenotype_label_matrix)
from mimic3benchmark.util import dataframe_from_csv

def main():
    """
    Extracts and processes per-subject data from MIMIC-III CSV files.

    The script reads MIMIC-III data files, processes the data to filter and clean records, 
    and generates per-subject files for stays, diagnoses, and event data. It also creates 
    phenotype labels and saves them to specified output directories.

    Arguments:
    - `mimic3_path`: Directory containing the MIMIC-III CSV files.
    - `output_path`: Directory where the processed data will be written.
    - `--event_tables`: List of event tables to read from.
    - `--phenotype_definitions`: YAML file with phenotype definitions.
    - `--itemids_file`: CSV containing ITEMIDs to keep.
    - `--verbose`: Print detailed information about processing.
    - `--quiet`: Suppress printing of details.
    - `--test`: Process a limited subset of data for testing.
    """
    parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
    parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
    parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
    parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                        default=['OUTPUTEVENTS', 'CHARTEVENTS', 'LABEVENTS'])
    parser.add_argument('--phenotype_definitions', '-p', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/icd_9_10_definitions_2.yaml'),
                        help='YAML file with phenotype definitions.')
    parser.add_argument('--itemids_file', '-i', type=str, default=os.path.join(os.path.dirname(__file__),
                                                                                '../resources/itemid_to_variable_map.csv'),
                        help='CSV containing list of ITEMIDs to keep.')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Print detailed information.')
    parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suppress printing of details.')
    parser.set_defaults(verbose=True)
    parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1,000,000 events.')
    args, _ = parser.parse_known_args()

    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Read core tables
    patients = read_patients_table(f'{args.mimic3_path}/core/patients.csv')
    admits = read_admissions_table(f'{args.mimic3_path}/core/admissions.csv')
    stays = read_icustays_table(f'{args.mimic3_path}/icu/icustays.csv')

    if args.verbose:
        print(f'START:\n\tstay_ids: {stays.stay_id.unique().shape[0]}\n\thadm_ids: {stays.hadm_id.unique().shape[0]}\n\tsubject_ids: {stays.subject_id.unique().shape[0]}')

    # Process and filter stays
    stays = remove_icustays_with_transfers(stays)
    if args.verbose:
        print(f'REMOVE ICU TRANSFERS:\n\tstay_ids: {stays.stay_id.unique().shape[0]}\n\thadm_ids: {stays.hadm_id.unique().shape[0]}\n\tsubject_ids: {stays.subject_id.unique().shape[0]}')

    stays = merge_on_subject_admission(stays, admits)
    stays = merge_on_subject(stays, patients)
    stays = filter_admissions_on_nb_icustays(stays)
    if args.verbose:
        print(f'REMOVE MULTIPLE STAYS PER ADMIT:\n\tstay_ids: {stays.stay_id.unique().shape[0]}\n\thadm_ids: {stays.hadm_id.unique().shape[0]}\n\tsubject_ids: {stays.subject_id.unique().shape[0]}')

    stays = add_age_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = filter_icustays_on_age(stays)
    if args.verbose:
        print(f'REMOVE PATIENTS AGE < 18:\n\tstay_ids: {stays.stay_id.unique().shape[0]}\n\thadm_ids: {stays.hadm_id.unique().shape[0]}\n\tsubject_ids: {stays.subject_id.unique().shape[0]}')

    stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)

    # Read and process diagnoses
    diagnoses = read_icd_diagnoses_table(f'{args.mimic3_path}/hosp')
    diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
    diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)

    # Count ICD codes and add phenotype groups
    count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))
    phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.safe_load(open(args.phenotype_definitions, 'r')))
    make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                          index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Testing mode: subset data
    if args.test:
        pat_idx = np.random.choice(patients.shape[0], size=1000, replace=False)
        patients = patients.iloc[pat_idx]
        stays = stays.merge(patients[['subject_id']], left_on='subject_id', right_on='subject_id')
        args.event_tables = [args.event_tables[0]]
        print(f'Using only {stays.shape[0]} stays and only {args.event_tables[0]} table')

    # Process and save per-subject data
    subjects = stays.subject_id.unique()
    break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
    break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)
    items_to_keep = set([int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None

    for table in args.event_tables:
        read_events_table_and_break_up_by_subject(f'{args.mimic3_path}', table, args.output_path, items_to_keep=items_to_keep,
                                                  subjects_to_keep=subjects)

if __name__ == '__main__':
    main()