import numpy as np
import pandas as pd
import os
import calliope


def imports():
    demand_wind = pd.read_csv('data/demand_wind.csv', index_col=0)
    demand_wind.index = pd.to_datetime(demand_wind.index)
    series_index = ['baseload', 'midmerit', 'peaking', 'wind']
    caps = pd.Series([10, 10, 10, 30], index=series_index)
    tech_varcosts = pd.Series([0.005, 0.035, 0.100, 0.000],
                              index=series_index)

    return demand_wind, caps, tech_varcosts


def convert_numpy_to_pandas(data, columns=['demand', 'wind'],
                            index_start='2015-01-01 00:00:00'):
    """Convert numpy array to pandas DataFrame time series for Calliope.

    Parameters:
    -----------
    data: data to be converted, as numpy array.
    columns: columns for export pandas DataFrame
    index: pandas DataFrame index start.

    Returns:
    --------
    data_pd: pandas DataFrame of data input
    """

    index = pd.date_range(start=index_start, periods=data.shape[0],
                          freq='h')
    data_pd = pd.DataFrame(data, index=index, columns=columns)
    return data_pd


def calculate_varcosts(dem_wind, caps, tech_varcosts):
    """Calculate each timestep's variable cost.

    Parameters:
    -----------
    dem_wind: pandas DataFrame of demand and wind time series data
    caps: pandas Series with installed capacities of each generation
        technology (baseload, midmerit, peaking and wind).
    tech_varcosts: pandas Series with generation costs of each technology

    Returns:
    --------
    time_step_varcosts: pandas DataFrame with variable cost time series
    """

    demand = dem_wind.loc[:, 'demand']
    wind_cf = dem_wind.loc[:, 'wind']

    # Calculate merit-order stacking generation levels of each technology
    gens_wd = np.minimum(caps['wind'] * wind_cf, demand)
    gens_bl = np.minimum(caps['baseload'], demand - gens_wd)
    gens_mm = np.minimum(caps['midmerit'], demand - gens_wd - gens_bl)
    gens_pk = demand - gens_wd - gens_bl - gens_mm    # No unmet demand

    # Calculate variable (generation) cost at each time step
    time_step_varcosts = tech_varcosts['baseload'] * gens_bl + \
        tech_varcosts['midmerit'] * gens_mm + \
        tech_varcosts['peaking'] * gens_pk + \
        tech_varcosts['wind'] * gens_wd

    return time_step_varcosts


def importance_subsampling(dem_wind, caps, tech_varcosts,
                           num_ts_total, num_ts_highbin):
    """Create demand and wind timeseries via importance subsampling
    using a timestep's variable cost as the importance function.

    Parameters:
    -----------
    dem_wind: pandas DataFrame of demand and wind time series data to
        sample from
    caps: list with capacities used to calculate variable cost
    num_ts_total: total number of sampled timesteps
    num_ts_highbin: number of timesteps in bin with high variable cost

    Returns:
    --------
    sampled_demand: numpy array of sampled demand values
    sampled_wind: numpy array of sampled wind values
    weights: numpy array of timestep weights
    """

    if num_ts_total < num_ts_highbin:
        raise ValueError('Number of time steps in high bin cannot exceed '
                         'total desired sample size.')

    # Create importance subsample
    num_ts_input = dem_wind.shape[0]
    num_ts_lowbin = num_ts_total - num_ts_highbin
    time_step_varcosts = \
        np.array(calculate_varcosts(dem_wind, caps, tech_varcosts))
    dem_wind_sorted = \
        np.array(dem_wind.iloc[np.argsort(-time_step_varcosts)])
    sampled_highbin = dem_wind_sorted[:num_ts_highbin]
    sampled_lowbin_index = \
        num_ts_highbin + np.random.choice(num_ts_input - num_ts_highbin,
                                          num_ts_lowbin, replace=False)
    sampled_lowbin = dem_wind_sorted[sampled_lowbin_index]
    sampled_data = np.concatenate((sampled_highbin, sampled_lowbin),
                                  axis=0)
    sampled_demand, sampled_wind = sampled_data[:, 0], sampled_data[:, 1]

    # Calculate weights (summing to 8760)
    weight_highbin = 1 / num_ts_input
    weight_lowbin = \
        (num_ts_input - num_ts_highbin) / (num_ts_lowbin * num_ts_input)
    weights_highbin = weight_highbin * 8760 * np.ones(shape=(num_ts_highbin))
    weights_lowbin = weight_lowbin * 8760 * np.ones(shape=(num_ts_lowbin))
    weights = np.concatenate((weights_highbin, weights_lowbin), axis=0)

    return sampled_demand, sampled_wind, weights


def run_calliope_model(dem_wind, weights=None, save_csv=True,
                       return_model=False):
    """Run Calliope model with some demand & wind data.

    Parameters:
    -----------
    dem_wind: pandas DataFrame of demand and wind time series data
    weights: numpy array or pandas DataFrame with time step weights
    save_csv: save model outputs as CSV (True or False)
    return_model: return the solved Calliope model (True or False)

    Returns:
    --------
    model: solved Calliope model (if return_model is True)
    """

    if weights is not None:
        if dem_wind.shape[0] != weights.shape[0]:
            raise ValueError('demand/wind data and weights must have same '
                             'number of time steps.')

    num_ts = dem_wind.shape[0]

    # Calliope requires a CSV file for time series data. We create a blank
    # one to initialize the model, then add in the correct data manually.
    dem_wind_placeholder = pd.DataFrame(data=np.zeros(dem_wind.shape),
                                        index=dem_wind.index,
                                        columns=dem_wind.columns)
    dem_wind_placeholder.to_csv('model_files/demand_wind_placeholder.csv',
                                float_format='%.4f')
    model = calliope.Model('model_files/model.yaml')
    os.remove('model_files/demand_wind_placeholder.csv')

    # Input demand, wind and weights
    model.inputs.resource.loc['region1::demand_power'].values[:] = \
        -np.array(dem_wind.loc[:, 'demand'])
    model.inputs.resource.loc['region1::wind'].values[:] = \
        np.array(dem_wind.loc[:, 'wind'])
    if weights is not None:
        model.inputs.timestep_weights.values = np.array(weights)

    model.run()
    if save_csv:
        model.to_csv('results')    # Output directory
    if return_model:
        return model


def run_calliope_model_importance_subsampling(dem_wind_full,
                                              num_ts_total,
                                              num_ts_highbin,
                                              save_csv=True,
                                              return_model=False):
    """Run Calliope with demand and wind subsample generated using
    importance subsampling method.

    Parameters:
    -----------
    dem_wind_full: pandas DataFrame of full demand and wind time series
        to be sampled from.
    num_ts_total: total number of sampled timesteps
    num_ts_highbin: number of timesteps in bin with high variable cost
    save_csv: save model outputs as CSV (True or False)

    Returns:
    --------
    model: solved Calliope model (if return_model is True)
    """

    # Stage 1: run model with random subsample of time steps
    dem_wind_s1 = \
        np.array(dem_wind_full)[np.random.choice(dem_wind_full.shape[0],
                                                 size=num_ts_total,
                                                 replace=False)]
    dem_wind_s1 = convert_numpy_to_pandas(dem_wind_s1)
    model = run_calliope_model(dem_wind_s1, weights=None,
                               save_csv=False, return_model=True)

    # Variable costs of the technologies
    vc_bl = float(model.inputs.cost_om_con.loc[:, 'region1::baseload'])
    vc_mm = float(model.inputs.cost_om_con.loc[:, 'region1::midmerit'])
    vc_pk = float(model.inputs.cost_om_con.loc[:, 'region1::peaking'])
    vc_wd = float(model.inputs.cost_om_con.loc[:, 'region1::wind'])
    tech_varcosts = \
        pd.Series([vc_bl, vc_mm, vc_pk, vc_wd],
                  index=['baseload', 'midmerit', 'peaking', 'wind'])

    # Stage 1 capacities used to calculate variable cost
    res = model.results
    cap_bl = float(res.energy_cap.loc['region1::baseload'].values)
    cap_mm = float(res.energy_cap.loc['region1::midmerit'].values)
    cap_pk = float(res.energy_cap.loc['region1::peaking'].values)
    cap_wd = float(res.resource_area.loc['region1::wind'].values)
    caps_s1 = pd.Series([cap_bl, cap_mm, cap_pk, cap_wd],
                        index=['baseload', 'midmerit', 'peaking', 'wind'])

    # Stage 2: run model with importance subsample of time steps
    dem_s2, wind_s2, weights_s2 = \
        importance_subsampling(dem_wind_full, caps_s1, tech_varcosts,
                               num_ts_total, num_ts_highbin)
    dem_wind_s2 = np.vstack((dem_s2, wind_s2)).T
    dem_wind_s2 = convert_numpy_to_pandas(dem_wind_s2)
    run_calliope_model(dem_wind_s2, weights=weights_s2,
                       save_csv=save_csv, return_model=return_model)


def run_importance_subsampling_example():
    """Run Calliope model with time series created using importance
    subsampling methodology, as an example."""

    dem_wind_full = pd.read_csv('data/demand_wind.csv', index_col=0)
    dem_wind_full.index = pd.to_datetime(dem_wind_full.index)

    run_calliope_model_importance_subsampling(dem_wind_full,
                                              num_ts_total=8760,
                                              num_ts_highbin=60,
                                              save_csv=True,
                                              return_model=False)


if __name__ == '__main__':
    run_importance_subsampling_example()
