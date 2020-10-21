"""Run the models."""


import os
import argparse
import logging
import models
import iss


def parse_args():
    """Read in model run arguments from bash command."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--logging_level', required=False, type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'], default='INFO',
                        help='Python logging module verbosity level')
    args = parser.parse_args()

    return args


def conduct_model_run(model_name_in_paper, ts_data, ts_subsampling,
                      subsample_blocks, num_days_subsample, num_days_high):
    """Conduct a model run using subsampled data.

    Parameters:
    -----------
    model_name_in_paper (str) : which model to run. Either 'LP' or 'MILP',
        corresponding to the models in the paper. The only difference
        is that, in the 'MILP' model, baseload can only be installed
        in units of 3GW and has a ramping constraint of 20%/hr.
    ts_data (pandas DataFrame) : the time series to run the model across
        (possibly after subsampling)
    ts_subsampling (str or None) : how to subsample the time series data.
        Either None (no subsampling, run across full time series),
        'random' (random sampling of days or hours), 'clustering' (k-medoids
        (clustering into days) or 'importance' (importance subsampling)
    subsample_blocks (str) : the subsample blocks. If 'days', subsampling
        is used to create a set of contiguous days. If 'hours', subsamples
        are hours. 'Hours' are not allowed if ts_subsampling='clustering',
        and has no effect if ts_subsampling=None.
    num_days_subsample (int) : number of days in the subsample.
        If subsample_blocks='hours', subsample length (in hours)
        is 24*num_days_subsample
    num_days_high (int) : number of "extreme" days with high cost to
        subsample if ts_subsampling='importance'. Not used otherwise.

    Returns:
    --------
    Nothing, but saves model outputs to CSV: 'summary_outputs.csv' and
        many more in a directory called 'outputs'
    """

    run_characteristics = {'ts_data': ts_data,
                           'baseload_integer': model_name_in_paper == 'MILP',
                           'baseload_ramping': model_name_in_paper == 'MILP'}
    if ts_subsampling is None:
        solved_model = iss.run_model(**run_characteristics)
    elif ts_subsampling == 'random':
        solved_model = iss.run_model_with_random_subsample(
            **run_characteristics,
            num_days_sample=num_days_subsample,
            subsample_blocks=subsample_blocks
        )
    elif ts_subsampling == 'clustering':
        if subsample_blocks == 'hours':
            raise ValueError('Cluster subsample blocks must be days.')
        solved_model = iss.run_model_with_clustered_subsample(
            **run_characteristics,
            num_days_sample=num_days_subsample
        )
    elif ts_subsampling == 'importance':
        solved_model = iss.run_model_with_importance_subsample(
            **run_characteristics,
            num_days_sample=num_days_subsample,
            num_days_high=num_days_high,
            subsample_blocks=subsample_blocks
        )
    else:
        raise ValueError('Invalid subsampling scheme')

    # Save summary outputs and a directory of the full range of outputs
    solved_model.get_summary_outputs().to_csv('summary_outputs.csv')
    solved_model.to_csv('full_outputs')


def run_example():
    """Run an example application of importance subsampling.

    The default settings are importance subsampling being applied to
    the 'LP' model to estimate the optimal system design across 2017
    using a 48-day subsample. The number of "extreme" days (n_d_e in
    the paper) is 16. It's easy to customise this function to use
    different subsample length, subsample scheme (e.g. random
    subsampling or regular k-medoids representative days), and to
    change to the 'MILP' model. See the docstring for conduct_model_run
    above for more details.
    """

    # Read in command line arguments and log run info
    args = parse_args()
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=getattr(logging, args.logging_level),
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    if os.path.exists('summary_outputs.csv'):
        raise ValueError(
            'Example script creates file `summary_outputs.csv`, but this '
            'already exists. Delete or rename this file before continuing')
    if os.path.exists('full_outputs'):
        raise ValueError(
            'Example script creates directory `full_outputs`, but this '
            'already exists. Delete or rename it before continuing')

    # Load the full time series that we will sample from
    ts_data = models.load_time_series_data(model_name='6_region')
    ts_data = ts_data.loc['2017']

    conduct_model_run(model_name_in_paper='LP',
                      ts_data=ts_data,
                      ts_subsampling='importance',
                      subsample_blocks='days',
                      num_days_subsample=48,
                      num_days_high=16)


if __name__ == '__main__':
    run_example()
