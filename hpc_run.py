"""Run the models in HPC."""


import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
import main
import models
import tests
import pdb


# If we are inside an HPC parallel run, let PRN (parallel run number)
# be the index corresponding to the parallel run. If we are not, we set
# PRN = 0, and code is run in sequence. Note that parallel runs should
# start at 1 and not 0 (NOT via pythonic numbering)
if 'PBS_ARRAY_INDEX' in os.environ:    # i.e. if we are in HPC parallel
    PRN = int(os.environ['PBS_ARRAY_INDEX'])
    RUN_ID = os.environ['PBS_JOBNAME'] + '_' + str(PRN)
else:     # i.e. we are running on laptop or regular bash shell
    PRN = 0
    RUN_ID = 'LAPTOP'


def parse_args():
    """Read in model run arguments from bash command."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str,
                        choices=['1_region', '6_region'],
                        help='1_region or 6_region')
    parser.add_argument('--ts_data_left_index', required=True, type=str,
                        help='Start of time series data, pandas dt index')
    parser.add_argument('--ts_data_right_index', required=True, type=str,
                        help='End of time series data, pandas dt index')
    parser.add_argument('--run_mode', required=True, type=str,
                        choices=['plan', 'operate'], help='plan or operate')
    parser.add_argument('--fixed_caps_file', required=False, type=str,
                        help='path to file containing fixed caps')
    parser.add_argument('--baseload_integer', required=True, type=str,
                        choices=['False', 'True'],
                        help='Baseload integer constraint for plan mode')
    parser.add_argument('--baseload_ramping', required=True, type=str,
                        choices=['False', 'True'],
                        help='Baseload ramping constraint')
    parser.add_argument('--ts_subsampling', required=False, type=str,
                        choices=['None', 'random',
                                 'importance', 'clustering'],
                        default=None, help='Time series subsampling')
    parser.add_argument('--num_days_subsample', required=False, type=int,
                        default=0, help='Number of days in subsample')
    parser.add_argument('--num_days_high', required=False, type=int,
                        default=0,
                        help='Numer of high generation cost in subsample, '
                             'applies only in importance subsampling mode')
    parser.add_argument('--subsample_blocks', required=False, type=str,
                        default='None', choices=['hours', 'days'],
                        help='Length of blocks to sample')
    parser.add_argument('--num_iterations', required=True, type=int,
                        help='Number of times to repeat experiment')
    parser.add_argument('--logging_level', required=False, type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'], default='WARNING',
                        help='Python logging module verbosity level')
    args = parser.parse_args()

    def str_to_bool(string):
        if string == 'False':
            boolean = False
        elif string == 'True':
            boolean = True
        else:
            raise ValueError()
        return boolean

    args.baseload_integer = str_to_bool(args.baseload_integer)
    args.baseload_ramping = str_to_bool(args.baseload_ramping)

    if args.ts_subsampling == 'None':
        args.ts_subsampling = None

    return args


def get_output_directory(create_directory=False):
    """Get the directory path for run outputs."""
    args = parse_args()
    baseload_integer_path = {False: 'cont',
                             True: 'int'}
    baseload_ramping_path = {False: 'noramp',
                             True: 'ramp'}

    base_directory = 'outputs_hpc'
    model_directory = args.model_name + '_' + args.run_mode
    baseload_directory = 'baseload_{}_{}'.format(
        baseload_integer_path[args.baseload_integer],
        baseload_ramping_path[args.baseload_ramping]
    )
    time_directory = args.ts_data_left_index+'_'+args.ts_data_right_index
    if args.run_mode == 'plan':
        sample_directory = 'sampling-{}_days-{}_high-{}'.format(
            str(args.ts_subsampling),
            args.num_days_subsample,
            args.num_days_high
        )
    elif args.run_mode == 'operate':
        sample_directory = '{}-{}'.format(
            args.fixed_caps_file.split('/')[-1].split('.')[0].split('_')[1],
            args.fixed_caps_file.split('/')[-2]
        )
    output_directory = os.path.join(base_directory,
                                    model_directory,
                                    baseload_directory,
                                    time_directory,
                                    sample_directory)

    # Create the right output directory if it does not exist
    if create_directory:
        sub_directory_path = ''
        for sub_directory in [base_directory,
                              model_directory,
                              baseload_directory,
                              time_directory,
                              sample_directory]:
            sub_directory_path = os.path.join(sub_directory_path,
                                              sub_directory)
            os.makedirs(sub_directory_path, exist_ok=True)

    return output_directory


def conduct_model_run(iteration):
    """Conduct a single model run in planning mode.

    Parameters:
    -----------
    iteration (int) : iteration number of multiple runs, used only if
        args.run_mode is operate
    """

    # Read in command line arguments
    args = parse_args()

    # Get correct time series data
    ts_data = models.load_time_series_data(model_name=args.model_name)
    ts_data = ts_data.loc[args.ts_data_left_index:args.ts_data_right_index]

    # Run the model, with subsampling scheme is applicable
    run_characteristics = {'model_name': args.model_name,
                           'ts_data': ts_data,
                           'run_mode': args.run_mode,
                           'baseload_integer': args.baseload_integer,
                           'baseload_ramping': args.baseload_ramping,
                           'run_id': str(RUN_ID) + '_' + str(iteration)}
    if args.run_mode == 'plan':
        if args.ts_subsampling is None:
            solved_model = main.run_model(**run_characteristics)
        elif args.ts_subsampling == 'random':
            solved_model = main.run_model_with_random_subsample(
                **run_characteristics,
                num_days_sample=args.num_days_subsample,
                subsample_blocks=args.subsample_blocks
            )
        elif args.ts_subsampling == 'clustering':
            if args.subsample_blocks == 'hours':
                raise ValueError('Cluster subsample blocks must be days.')
            solved_model = main.run_model_with_clustered_subsample(
                **run_characteristics,
                num_days_sample=args.num_days_subsample
            )
        elif args.ts_subsampling == 'importance':
            solved_model = main.run_model_with_importance_subsample(
                **run_characteristics,
                num_days_sample=args.num_days_subsample,
                num_days_high=args.num_days_high,
                subsample_blocks=args.subsample_blocks
            )
        else:
            raise ValueError('Invalid subsampling scheme')
    elif args.run_mode == 'operate':
        # Load the fixed caps and run the model forward
        assert args.ts_subsampling is None, \
            'Do not use subsampling in forward runs (operate mode).'
        fixed_caps = pd.read_csv(args.fixed_caps_file,
                                 index_col=0).loc[iteration]
        solved_model = main.run_model_forward_with_fixed_caps(
            **run_characteristics, fixed_caps=fixed_caps
        )

    return solved_model


def conduct_model_runs():
    """Conduct the model runs in HPC."""

    # Read in command line arguments
    args = parse_args()

    # Create output directory if it does not exist yet
    output_directory = get_output_directory(create_directory=True)

    # Run the model and save results to CSV
    iteration_index = np.arange(args.num_iterations)
    for iteration in iteration_index:
        if (PRN == 0) or (iteration == PRN - 1):
            np.random.seed(iteration)
            output_path = os.path.join(output_directory,
                                       'iter_{:04d}.csv'.format(iteration))

            # Read in command line arguments and log run info
            args = parse_args()
            logging.basicConfig(
                format='[%(asctime)s] %(levelname)s: %(message)s',
                level=getattr(logging, args.logging_level),
                datefmt='%Y-%m-%d,%H:%M:%S'
            )
            logging.info('\n\n\n\n\nIteration: %s', iteration)
            logging.info('%s', args)

            # Run only if file doesn't already exist
            if os.path.isfile(output_path):
                logging.info('Output file already exists.')
                continue

            # Run the simulation, conduct tests and save
            solved_model = conduct_model_run(iteration)
            if args.model_name == '1_region':
                summary_outputs = solved_model.get_summary_outputs()
                tests.test_output_consistency_1_region(solved_model,
                                                       args.run_mode)
            elif args.model_name == '6_region':
                summary_outputs = solved_model.get_summary_outputs()
                tests.test_output_consistency_6_region(solved_model,
                                                       args.run_mode)
            summary_outputs.to_csv(output_path)


if __name__ == '__main__':
    conduct_model_runs()
