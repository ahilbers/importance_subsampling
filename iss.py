"""Example model runs."""


import logging
import numpy as np
import pandas as pd
import models
import tests


def calculate_generation_costs(ts_data, fixed_caps, baseload_ramping=False):
    """Calculate each time step's generation cost in some demand & weather
    time series data.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to run the model across,
        after importance subsampling from it
    fixed_caps (dict) : the fixed capacities used to calculate the
        generation costs
    baseload_ramping (bool) : if True, baseload subject to ramping
        constraint of 20% of installed capacity per hour.

    Returns:
    --------
    generation_costs (pandas DataFrame) : time series of generation costs
        for each time step in ts_data

    """

    logging.info('Calculating generation costs. '
                 'Handing over to Calliope now.')
    model = models.SixRegionModel(ts_data=ts_data,
                                  run_mode='operate',
                                  baseload_integer=False,
                                  baseload_ramping=baseload_ramping,
                                  fixed_caps=fixed_caps)
    model.run()
    logging.info('Summary outputs from operational run:%s',
                 model.get_summary_outputs())
    logging.info('Model run complete.\n\n')
    generation_costs = model.results.cost_var.values.sum(axis=(0, 1))
    generation_costs = pd.DataFrame(generation_costs, index=ts_data.index,
                                    columns=['generation_cost'])

    return generation_costs


def get_day_sample(ts_data, sample_days):
    """Get a sample from a DataFrame of days.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : demand and wind data to sample from.
    sample_days (pandas DataFrame) : the sample days: dataFrame with 3
        columns: 'year', 'month' and 'day'

    Returns:
    --------
    sample: the demand and wind time series on the sampled days.
    """

    # Construct the sample by concatenating the sampled days
    sample = pd.concat([ts_data.loc[
        ts_data.index.year.isin([sample_day.year]) &
        ts_data.index.month.isin([sample_day.month]) &
        ts_data.index.day.isin([sample_day.day])
    ] for sample_day in sample_days.itertuples()])

    return sample


def create_random_subsample(ts_data, num_days_sample, blocks):
    """Create random subsample.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to create sample from.
    num_days_sample (int) : number of days in the subsample.
        If subsample_blocks='hours', subsample length (in hours)
        is 24*num_days_subsample
    blocks (str) : the subsample blocks. If 'days', subsampling
        is used to create a set of contiguous days. If 'hours', subsamples
        are hours. 'Hours' are not allowed if ts_subsampling='clustering',
        and has no effect if ts_subsampling=None.

    Returns:
    --------
    sample (pandas DataFrame) : (weighted) sample.
    """

    if blocks == 'hours':
        sample = ts_data.iloc[np.sort(np.random.choice(
            ts_data.shape[0], size=24*num_days_sample, replace=False
        ))]
    elif blocks == 'days':
        unique_days = pd.DataFrame(
            list(dict.fromkeys(zip(ts_data.index.year,
                                   ts_data.index.month,
                                   ts_data.index.day))),
            columns=['year', 'month', 'day']
        )
        sample_days = unique_days.iloc[np.sort(np.random.choice(
            len(unique_days), size=num_days_sample, replace=False
        ))]
        sample = get_day_sample(ts_data, sample_days)
    else:
        raise ValueError('Valid subsample blocks: hours or days.')

    return sample


def create_daily_vectors(ts_data):
    """Create day vectors of time series data."""

    # Reshape data into daily vectors
    sample_index = ts_data.resample('24h').mean().dropna().index
    sample_columns = sum([
        ['{}_{}'.format(input_column, hour) for hour in range(24)]
        for input_column in ts_data.columns
    ], [])
    daily_vecs = pd.DataFrame(index=sample_index, columns=sample_columns)
    for i, input_column in enumerate(ts_data.columns):
        column_data = ts_data.loc[:, input_column].values
        daily_vecs.iloc[:, 24*i:24*(i+1)] = np.reshape(
            column_data, newshape=(round(ts_data.shape[0]/24), -1)
        )

    return daily_vecs


def create_clustered_sample(ts_data, num_clusters):
    """Create weighted subsample by k-medoid clustering daily ts data.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to create sample from.
    num_clusters (int) : number of clusters (representative days) to
        sample into.

    Returns:
    --------
    sample (pandas DataFrame) : (weighted) sample.
    """

    # Obtain daily vectors and rescale them to lie between 0 and 1
    daily_vecs = create_daily_vectors(ts_data)
    daily_vecs_rescaled = daily_vecs.copy()
    for i, input_column in enumerate(ts_data.columns):
        min_val = ts_data.loc[:, input_column].min()
        max_val = ts_data.loc[:, input_column].max()
        daily_vecs_rescaled.iloc[:, 24*i:24*(i+1)] = (
            (daily_vecs.iloc[:, 24*i:24*(i+1)] - min_val)
            / (max_val - min_val)
        )

    # Cluster the days and create a sample of them
    from sklearn.cluster import k_means
    means, labels, _ = k_means(daily_vecs_rescaled.values,
                               n_clusters=num_clusters)
    medoids = pd.DataFrame(index=np.arange(num_clusters),
                           columns=['year', 'month', 'day'])
    for cluster_num in range(num_clusters):
        closest_day = (daily_vecs_rescaled.iloc[labels == cluster_num]
                       - means[cluster_num]).pow(2).sum(axis=1).idxmin()
        medoids.loc[cluster_num, :] = [closest_day.year,
                                       closest_day.month,
                                       closest_day.day]
    sample = get_day_sample(ts_data, medoids)

    # Adjust the weights to account for cluster sizes
    weights = pd.DataFrame(index=sample.index, columns=['weight'],
                           dtype=float)
    for cluster_num in range(num_clusters):
        weights.iloc[24*cluster_num:24*(cluster_num+1), 0] = float(
            num_clusters * len(labels[labels == cluster_num])
            / round(ts_data.shape[0] / 24)
        )

    # Merge sample and weights to create weighted sample
    sample = pd.merge(left=sample, right=weights,
                      left_index=True, right_index=True)

    return sample


def create_importance_subsample(ts_data, generation_costs,
                                num_days_sample, num_days_high, blocks):
    """Create importance subsample.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to create sample from.
    generation_costs (pandas DataFrame) : the generation costs for
        each time step in ts_data
    num_days_sample (int) : number of days in the subsample.
        If subsample_blocks='hours', subsample length (in hours)
        is 24*num_days_subsample
    num_days_high (int) : number of "extreme" days with high cost to
        subsample if ts_subsampling='importance'. Not used otherwise.
    blocks (str) : the subsample blocks. If 'days', subsampling
        is used to create a set of contiguous days. If 'hours', subsamples
        are hours. 'Hours' are not allowed if ts_subsampling='clustering',
        and has no effect if ts_subsampling=None.

    Returns:
    --------
    sample (pandas DataFrame) : (weighted) sample.
    """

    if not all(ts_data.index == generation_costs.index):
        raise ValueError('Time series data and generation costs '
                         'should have same index.')

    # Add generation costs to time series data
    ts_data_gc = pd.merge(left=ts_data, right=generation_costs,
                          left_index=True, right_index=True)

    num_days_input = round(ts_data.shape[0] / 24)
    num_days_low = num_days_sample - num_days_high
    num_ts_input, num_ts_sample = 24*num_days_input, 24*num_days_sample
    num_ts_high, num_ts_low = 24*num_days_high, 24*num_days_low

    if blocks == 'hours':
        # Sample 24*num_days_high time steps with highest generation cost
        # and a random selection of those remaining
        ts_data_sorted = ts_data_gc.sort_values(by='generation_cost',
                                                ascending=False)
        sample_high = ts_data_sorted.iloc[:num_ts_high]
        sample_low = ts_data_sorted.iloc[
            num_ts_high + np.random.choice(num_ts_input - num_ts_high,
                                           num_ts_low, replace=False)
        ]
        sample = pd.concat((sample_high, sample_low), axis=0)
        sample.loc[:, 'cluster_weight'] = 1    # No clustering
    elif blocks == 'days':
        # Sample num_days_high days with highest peak generation cost
        # and cluster the remaining ones
        days_sorted = ts_data_gc.resample('24h').max().dropna().sort_values(
            by='generation_cost', ascending=False
        ).index
        days_sorted = pd.DataFrame(zip(days_sorted.year,
                                       days_sorted.month,
                                       days_sorted.day),
                                   columns=['year', 'month', 'day'])
        sample_high = get_day_sample(ts_data,
                                     days_sorted.iloc[:num_days_high])
        sample_high.loc[:, 'cluster_weight'] = 1    # No clustering
        ts_data_low = ts_data.loc[~ts_data.index.isin(sample_high.index)]
        sample_low = create_clustered_sample(ts_data_low,
                                             num_clusters=num_days_low)
        sample_low = sample_low.rename(columns={'weight':'cluster_weight'})
        sample = pd.concat([sample_high, sample_low], sort=False)

    # Calculate weights (summing to num_ts_total). These are a product
    # of the cluster weights and the importance weights
    weights = np.zeros(shape=num_ts_sample)
    cluster_weights = sample.loc[:, 'cluster_weight'].copy()
    weights[:num_ts_high] = (cluster_weights[:num_ts_high]
                             * num_ts_sample / num_ts_input)
    weights[num_ts_high:] = (cluster_weights[num_ts_high:]
                             * num_ts_sample
                             * (num_ts_input - num_ts_high)
                             / (num_ts_low * num_ts_input))
    sample.loc[:, 'weight'] = weights

    # Remove generation cost column and reset index
    sample = sample.drop(['cluster_weight'], axis=1)

    logging.info('Sampled days: \n%s',
                 sample.resample('24h').mean().dropna().index)

    return sample


def run_model(ts_data, baseload_integer=False, baseload_ramping=False):
    """Run model with some time series data.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to run the model across.
    baseload_integer (bool) : if True, baseload capacity may be built
        in units of 3GW ('MILP' model). If False, any positive value.
    baseload_ramping (bool) : if True, baseload subject to ramping
        constraint of 20% of installed capacity per hour.

    Returns:
    --------
    model (calliope.Model) : solved model.
    """

    # Create and run model in Calliope
    logging.info('Creating instance of Calliope model.')
    model = models.SixRegionModel(ts_data=ts_data,
                                  run_mode='plan',
                                  baseload_integer=baseload_integer,
                                  baseload_ramping=baseload_ramping)
    logging.info('Model instance created.\n')
    logging.info('Running model.')
    model.run()
    tests.test_output_consistency_6_region(model, run_mode='plan')
    logging.info('Model run complete.\n\n')

    return model


def run_model_with_random_subsample(ts_data,
                                    baseload_integer, baseload_ramping,
                                    num_days_sample, subsample_blocks):
    """Run model on a random selection of hours or days.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to run the model across,
        after sampling from it.
    baseload_integer (bool) : if True, baseload capacity may be built
        in units of 3GW ('MILP' model). If False, any positive value.
    baseload_ramping (bool) : if True, baseload subject to ramping
        constraint of 20% of installed capacity per hour.
    num_days_sample (int) : number of days in the subsample.
        If subsample_blocks='hours', subsample length (in hours)
        is 24*num_days_subsample
    subsample_blocks (str) : the subsample blocks. If 'days', subsampling
        is used to create a set of contiguous days. If 'hours', subsamples
        are hours.

    Returns:
    --------
    model (calliope.Model) : solved model.
    """

    # Create random subsample and reset index
    subsample = create_random_subsample(ts_data=ts_data,
                                        num_days_sample=num_days_sample,
                                        blocks=subsample_blocks)
    subsample.index = pd.to_datetime(np.arange(subsample.shape[0]),
                                     unit='h', origin='2020')

    logging.info('Running model with random subsample.')
    solved_model = run_model(ts_data=subsample,
                             baseload_integer=baseload_integer,
                             baseload_ramping=baseload_ramping)

    return solved_model


def run_model_with_clustered_subsample(ts_data,
                                       baseload_integer, baseload_ramping,
                                       num_days_sample):
    """Run model on a time series clustered into representative days.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to run the model across,
        after taking a clustered sample from it.
    baseload_integer (bool) : if True, baseload capacity may be built
        in units of 3GW ('MILP' model). If False, any positive value.
    baseload_ramping (bool) : if True, baseload subject to ramping
        constraint of 20% of installed capacity per hour.
    num_days_sample (int) : number of days in the subsample.
        If subsample_blocks='hours', subsample length (in hours)
        is 24*num_days_subsample.

    Returns:
    --------
    model (calliope.Model) : solved model.
    """

    # Create random subsample and reset index
    subsample = create_clustered_sample(ts_data=ts_data,
                                        num_clusters=num_days_sample)
    subsample.index = pd.to_datetime(np.arange(subsample.shape[0]),
                                     unit='h', origin='2020')

    logging.info('Running model with k-medoids clustered subsample.')
    solved_model = run_model(ts_data=subsample,
                             baseload_integer=baseload_integer,
                             baseload_ramping=baseload_ramping)

    return solved_model


def run_model_with_importance_subsample(ts_data,
                                        baseload_integer, baseload_ramping,
                                        num_days_sample, num_days_high,
                                        subsample_blocks):
    """Run model on time series subsampled using importance subsampling.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : the time series to run the model across,
        after importance subsampling from it
    baseload_integer (bool) : if True, baseload capacity may be built
        in units of 3GW ('MILP' model). If False, any positive value.
    baseload_ramping (bool) : if True, baseload subject to ramping
        constraint of 20% of installed capacity per hour.
    num_days_sample (int) : number of days in the subsample.
        If subsample_blocks='hours', subsample length (in hours)
        is 24*num_days_subsample
    num_days_high (int) : number of "extreme" days with high cost to
        subsample if ts_subsampling='importance'. Not used otherwise.
    subsample_blocks (str) : the subsample blocks. If 'days', subsampling
        is used to create a set of contiguous days. If 'hours', subsamples
        are hours and any clustering is replaced with random selection of
        time steps.

    Returns:
    --------
    model (calliope.Model) : solved model.
    """

    # Stage 1 model run with clustered subsample
    logging.info('Starting stage 1: run model with clustered subsample.')
    subsample_s1 = create_clustered_sample(ts_data=ts_data,
                                           num_clusters=num_days_sample)
    subsample_s1.index = pd.to_datetime(np.arange(subsample_s1.shape[0]),
                                        unit='h', origin='2020')
    solved_model_s1 = run_model(ts_data=subsample_s1,
                                baseload_integer=baseload_integer,
                                baseload_ramping=baseload_ramping)
    summary_outputs = solved_model_s1.get_summary_outputs()
    logging.info('Finished stage 1 model run.\n\n')

    caps_s1 = summary_outputs.loc[:, 'output']
    logging.info('Stage 1 estimated capacities:\n %s\n\n', caps_s1)

    logging.info('Calculating generation costs across full time series.')
    generation_costs = calculate_generation_costs(
        ts_data=ts_data,
        fixed_caps=caps_s1,
        baseload_ramping=baseload_ramping
    )
    logging.info('Finished calculating generation costs.')
    logging.info('Generation costs:\n %s\n\n', generation_costs)

    importance_subsample_s2 = create_importance_subsample(
        ts_data=ts_data,
        generation_costs=generation_costs,
        num_days_sample=num_days_sample,
        num_days_high=num_days_high,
        blocks=subsample_blocks
    )
    importance_subsample_s2.index = pd.to_datetime(
        np.arange(importance_subsample_s2.shape[0]), unit='h', origin='2020'
    )    # Reset index

    # Stage 2 model run with importance subsample
    logging.info('\n\n')
    logging.info('Starting stage 2: run model with importance subsample.')
    solved_model_s2 = run_model(ts_data=importance_subsample_s2,
                                baseload_integer=baseload_integer,
                                baseload_ramping=baseload_ramping)
    logging.info('Finished stage 2 model run.\n\n')

    return solved_model_s2
