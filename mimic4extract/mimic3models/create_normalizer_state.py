from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader, DecompensationReader, LengthOfStayReader, PhenotypingReader, MultitaskReader
from mimic3models.preprocessing import Discretizer, Normalizer

import os
import argparse


def main():
    """
    Main function for creating a normalizer state file. This file stores the means and standard deviations of columns
    after they have been processed by a discretizer. These statistics are later used to standardize the input of neural models.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Script for creating a normalizer state, which standardizes the input of neural models.')
    
    parser.add_argument('--task', type=str, required=True, choices=['ihm', 'decomp', 'los', 'pheno', 'multi'],
                        help="Specify the task: in-hospital mortality (ihm), decompensation (decomp), length of stay (los), phenotyping (pheno), or multitask (multi).")
    
    parser.add_argument('--timestep', type=float, default=1.0, help="Rate of re-sampling to discretize time-series data.")
    
    parser.add_argument('--impute_strategy', type=str, default='previous', choices=['zero', 'next', 'previous', 'normal_value'],
                        help='Strategy for imputing missing values: zero, next, previous, or a normal value.')
    
    parser.add_argument('--start_time', type=str, choices=['zero', 'relative'],
                        help='Start time for discretization: "zero" uses the beginning of the ICU stay, "relative" uses the first ICU event time.')
    
    parser.add_argument('--store_masks', dest='store_masks', action='store_true', help='Store masks indicating observed/imputed values.')
    parser.add_argument('--no-masks', dest='store_masks', action='store_false', help='Do not store masks indicating observed/imputed values.')
    
    parser.add_argument('--n_samples', type=int, default=-1, help='Number of samples to use for estimating means and standard deviations. Use -1 to process all training samples.')
    
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output normalizer file.')
    
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset.')
    
    parser.set_defaults(store_masks=True)
    
    args = parser.parse_args()
    print(args)

    # Select the appropriate reader based on the task
    reader = None
    dataset_dir = os.path.join(args.data, 'train')
    if args.task == 'ihm':
        reader = InHospitalMortalityReader(dataset_dir=dataset_dir, period_length=48.0)
    elif args.task == 'decomp':
        reader = DecompensationReader(dataset_dir=dataset_dir)
    elif args.task == 'los':
        reader = LengthOfStayReader(dataset_dir=dataset_dir)
    elif args.task == 'pheno':
        reader = PhenotypingReader(dataset_dir=dataset_dir)
    elif args.task == 'multi':
        reader = MultitaskReader(dataset_dir=dataset_dir)

    # Initialize the discretizer
    discretizer = Discretizer(timestep=args.timestep,
                              store_masks=args.store_masks,
                              impute_strategy=args.impute_strategy,
                              start_time=args.start_time)
    
    # Obtain the header from the first example to identify continuous channels
    discretizer_header = reader.read_example(0)['header']
    continuous_channels = [i for (i, x) in enumerate(discretizer_header) if "->" not in x]

    # Initialize the normalizer with the continuous channels
    normalizer = Normalizer(fields=continuous_channels)

    # Determine the number of samples to process
    n_samples = args.n_samples
    if n_samples == -1:
        n_samples = reader.get_number_of_examples()

    # Feed data into the normalizer to compute means and standard deviations
    for i in range(n_samples):
        if i % 1000 == 0:
            print(f'Processed {i} / {n_samples} samples', end='\r')
        ret = reader.read_example(i)
        data, new_header = discretizer.transform(ret['X'], end=ret['t'])
        normalizer._feed_data(data)
    print('\n')

    # Save the computed normalizer state
    file_name = f'{args.task}_ts:{args.timestep:.2f}_impute:{args.impute_strategy}_start:{args.start_time}_masks:{args.store_masks}_n:{n_samples}.normalizer'
    file_name = os.path.join(args.output_dir, file_name)
    print(f'Saving the state in {file_name} ...')
    normalizer._save_params(file_name)


if __name__ == '__main__':
    main()