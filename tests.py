"""Tests to check whether the models behave as required."""


import os
import logging
import numpy as np
import pandas as pd
import models
import main


# Install costs and generation costs. These should match the information
# provided in the model.yaml and techs.yaml files in the model definition
COSTS = pd.DataFrame(columns=['install', 'generation'])

# Costs for 1 region model
COSTS.loc['baseload'] = [300., 0.005]
COSTS.loc['peaking']  = [100., 0.035]
COSTS.loc['wind']     = [100., 0.000]
COSTS.loc['unmet']    = [  0., 6.000]

# Heterogenised costs for 6 region model
COSTS.loc['baseload_region1'] = [300.1, 0.005001]
COSTS.loc['baseload_region3'] = [300.3, 0.005003]
COSTS.loc['baseload_region6'] = [300.6, 0.005006]
COSTS.loc['peaking_region1']  = [100.1, 0.035001]
COSTS.loc['peaking_region3']  = [100.3, 0.035003]
COSTS.loc['peaking_region6']  = [100.6, 0.035006]
COSTS.loc['wind_region2']     = [100.2, 0.000002]
COSTS.loc['wind_region5']     = [100.5, 0.000005]
COSTS.loc['wind_region6']     = [100.6, 0.000006]
COSTS.loc['unmet_region2']    = [  0.0, 6.000002]
COSTS.loc['unmet_region4']    = [  0.0, 6.000004]
COSTS.loc['unmet_region5']    = [  0.0, 6.000005]
COSTS.loc['transmission_region1_region2'] = [100.12, 0]
COSTS.loc['transmission_region1_region5'] = [150.15, 0]
COSTS.loc['transmission_region1_region6'] = [100.16, 0]
COSTS.loc['transmission_region2_region3'] = [100.23, 0]
COSTS.loc['transmission_region3_region4'] = [100.34, 0]
COSTS.loc['transmission_region4_region5'] = [100.45, 0]
COSTS.loc['transmission_region5_region6'] = [100.56, 0]


# Topology of 6 region model. These should match the information provided
# in the locations.yaml files in the model definition
BASELOAD_TOP, PEAKING_TOP, WIND_TOP, UNMET_TOP, DEMAND_TOP = (
    [('baseload', i) for i in ['region1', 'region3', 'region6']],
    [('peaking', i) for i in ['region1', 'region3', 'region6']],
    [('wind', i) for i in ['region2', 'region5', 'region6']],
    [('unmet', i) for i in ['region2', 'region4', 'region5']],
    [('demand', i) for i in ['region2', 'region4', 'region5']]
)
TRANSMISSION_TOP = [('transmission', *i)
                    for i in [('region1', 'region2'),
                              ('region1', 'region5'),
                              ('region1', 'region6'),
                              ('region2', 'region3'),
                              ('region3', 'region4'),
                              ('region4', 'region5'),
                              ('region5', 'region6')]]


def test_output_consistency_1_region(model, run_mode):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
   -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    run_mode (str) : 'plan' or 'operate'

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True
    cost_total_method1 = 0

    out = model.get_summary_outputs()
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

    # Test if generation technology installation costs are consistent
    if run_mode == 'plan':
        for tech in ['baseload', 'peaking', 'wind']:
            cost_method1 = float(COSTS.loc[tech, 'install']
                                 * out.loc['cap_{}_total'.format(tech)])
            cost_method2 = corrfac * float(
                res.cost_investment[0].loc['region1::{}'.format(tech)]
            )
            if abs(cost_method1 - cost_method2) > 0.1:
                logging.error('FAIL: %s install costs do not match!\n'
                              '    manual: %s, model: %s',
                              tech, cost_method1, cost_method2)
                passing = False
            cost_total_method1 += cost_method1

    # Test if generation costs are consistent
    for tech in ['baseload', 'peaking', 'wind', 'unmet']:
        cost_method1 = float(COSTS.loc[tech, 'generation']
                             * out.loc['gen_{}_total'.format(tech)])
        cost_method2 = corrfac * float(res.cost_var[0].loc[
            'region1::{}'.format(tech)
        ].sum())
        if abs(cost_method1 - cost_method2) > 0.1:
            logging.error('FAIL: %s generation costs do not match!\n'
                          '    manual: %s, model: %s',
                          tech, cost_method1, cost_method2)
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
    if run_mode == 'plan':
        cost_total_method2 = corrfac * float(res.cost.sum())
        if abs(cost_total_method1 - cost_total_method2) > 0.1:
            logging.error('FAIL: total system costs do not match!\n'
                          '    manual: %s, model: %s',
                          cost_total_method1, cost_total_method2)
            passing = False

    # Test if supply matches demand
    generation_total = float(out.loc[['gen_baseload_total',
                                      'gen_peaking_total',
                                      'gen_wind_total',
                                      'gen_unmet_total']].sum())
    demand_total = float(out.loc['demand_total'])
    if abs(generation_total - demand_total) > 0.1:
        logging.error('FAIL: generation does not match demand!\n'
                      '    generation: {}, demand: {}'.format(generation_total,
                                                              demand_total))
        passing = False

    return passing


def test_output_consistency_6_region(model, run_mode):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    run_mode (str) : 'plan' or 'operate'

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True
    cost_total_method1 = 0

    out = model.get_summary_outputs()
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

    # Test if generation technology installation costs are consistent
    if run_mode == 'plan':
        for tech, region in BASELOAD_TOP + PEAKING_TOP + WIND_TOP:
            cost_method1 = float(
                COSTS.loc['{}_{}'.format(tech, region), 'install'] *
                out.loc['cap_{}_{}'.format(tech, region)]
            )
            cost_method2 = corrfac * float(
                res.cost_investment[0].loc['{}::{}_{}'.format(region,
                                                              tech,
                                                              region)]
            )
            if abs(cost_method1 - cost_method2) > 0.1:
                logging.error('FAIL: %s install costs in %s do not match!\n'
                              '    manual: %s, model: %s',
                              tech, region, cost_method1, cost_method2)
                passing = False
            cost_total_method1 += cost_method1

    # Test if transmission installation costs are consistent
    if run_mode == 'plan':
        for tech, region_a, region_b in TRANSMISSION_TOP:
            cost_method1 = float(
                COSTS.loc[
                    '{}_{}_{}'.format(tech, region_a, region_b), 'install'
                ] * out.loc[
                    'cap_transmission_{}_{}'.format(region_a, region_b)
                ]
            )
            cost_method2 = 2 * corrfac * \
                float(res.cost_investment[0].loc[
                    '{}::{}_{}_{}:{}'.format(region_a,
                                             tech,
                                             region_a,
                                             region_b,
                                             region_b)
                ])
            if abs(cost_method1 - cost_method2) > 0.1:
                logging.error('FAIL: %s install costs from %s to %s do '
                              'not match!\n    manual: %s, model: %s',
                              tech, region_a, region_b,
                              cost_method1, cost_method2)
                passing = False
            cost_total_method1 += cost_method1

    # Test if generation costs are consistent
    for tech, region in BASELOAD_TOP + PEAKING_TOP + WIND_TOP + UNMET_TOP:
        cost_method1 = float(
            COSTS.loc['{}_{}'.format(tech, region), 'generation']
            * out.loc['gen_{}_{}'.format(tech, region)]
        )
        cost_method2 = corrfac * float(
            res.cost_var[0].loc['{}::{}_{}'.format(region, tech, region)].sum()
        )
        if abs(cost_method1 - cost_method2) > 0.1:
            logging.error('FAIL: %s generation costs in %s do not match!\n'
                          '    manual: %s, model: %s',
                          tech, region, cost_method1, cost_method2)
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
    if run_mode == 'plan':
        cost_total_method2 = corrfac * float(res.cost.sum())
        if abs(cost_total_method1 - cost_total_method2) > 0.1:
            logging.error('FAIL: total system costs do not match!\n'
                          '    manual: %s, model: %s',
                          cost_total_method1, cost_total_method2)
            passing = False

    # Test if supply matches demand
    generation_total = float(out.loc[['gen_baseload_total',
                                      'gen_peaking_total',
                                      'gen_wind_total',
                                      'gen_unmet_total']].sum())
    demand_total = float(out.loc['demand_total'])
    if abs(generation_total - demand_total) > 0.1:
        logging.error('FAIL: generation does not match demand!\n'
                      '    generation: %s, demand: %s',
                      generation_total, demand_total)
        passing = False

    return passing


def test_override_dict(model_name):
    """Test if the override dictionary is working properly"""

    passing = True
    fixed_caps, o_dict_1 = main.get_test_fixed_caps_override_dict(model_name)

    # Test if override dictionary created by function is correct
    o_dict_2 = models.get_cap_override_dict(model_name, fixed_caps)
    if o_dict_1 != o_dict_2:
        print('FAIL: Override dictionary does not match!\n'
              '    Problem keys:')
        for key in o_dict_1:
            try:
                if o_dict_1[key] != o_dict_2[key]:
                    print(key)
            except KeyError:
                print(key)
        passing = False

    if passing:
        print('PASS: override_dictionary is created properly.')

    return passing


def test_get_sample_days():
    """Test the get_sample_days function by comparing results with benchmark."""
    passing = True

    np.random.seed(0)
    ts_data = models.load_time_series_data(model_name='1_region')
    sample = main.get_sample_days(ts_data, num_days=10)
    benchmark_sample = pd.read_csv('benchmarks/sample_days.csv', index_col=0)
    if np.abs(sample - benchmark_sample).max().max() > 1e-10:
        passing = False

    return passing


def test_calculate_generation_costs():

    passing = True

    ts_data = models.load_time_series_data(model_name='1_region').loc['2017-01']
    fixed_caps = {'cap_baseload_total': 14.8159205143,
                  'cap_peaking_total': 33.9451081239,
                  'cap_wind_total': 32.0864991233}

    generation_costs, model = main.calculate_generation_costs(
        model_name='1_region', ts_data=ts_data, fixed_caps=fixed_caps,
        baseload_ramping=True, test_mode=True)
    generation_costs = generation_costs.values
    gen_bl = model.results.carrier_prod.loc['region1::baseload::power'].values
    gen_pk = model.results.carrier_prod.loc['region1::peaking::power'].values
    gen_wd = model.results.carrier_prod.loc['region1::wind::power'].values
    gen_um = model.results.carrier_prod.loc['region1::unmet::power'].values

    if np.abs(COSTS.loc['baseload', 'generation'] * gen_bl
              + COSTS.loc['peaking', 'generation' ] * gen_pk
              + COSTS.loc['wind', 'generation'] * gen_wd
              + COSTS.loc['unmet', 'generation'] * gen_um
              - generation_costs).max() > 1e-10:
        passing = False

    return passing


def test_get_sample_days():

    passing = True

    ts_data = models.load_time_series_data(model_name='1_region')
    sample_days = np.array([[1993, 9, 18],
                            [1994, 7, 9]])
    sample = get_sample_days(ts_data, sample_days)
    sample_days_alt = np.array(list(dict.fromkeys(
        zip(sample.index.year, sample.index.month, sample.index.day)
    )))
    if not np.array_equal(sample_days, sample_days_alt):
        passing = False

    return passing


def test_create_daily_vectors():
    passing = True

    test_day = '1993-09-18'
    ts_data = models.load_time_series_data(model_name='6_region')
    daily_vectors = create_daily_vectors(ts_data)
    ts_data_td = ts_data.loc[test_day]
    daily_vectors_td = daily_vectors.loc[test_day]

    for i, column in enumerate(ts_data.columns):
        if not all(ts_data_td.loc[:, column].values
                   == daily_vectors_td[24*i:24*(i+1)].values):
            print('Error in column {}'.format(column))
            passing = False

    return passing


def test_create_clustered_sample():
    passing = True

    ts_data = models.load_time_series_data(model_name='6_region')
    sample = create_clustered_sample(ts_data, num_clusters=10)

    if sample.shape != (240, 7):
        print('Wrong shape: {} (should be (240, 7))'.format(sample.shape))
        passing = False
    if abs(sample.loc[:, 'weight'].mean() - 1) > 1e-10:
        print('Wrong weights average {} (should be 1)'.format(
            sample.loc[:, 'weight'].mean()
        ))
        passing = False

    return passing


def test_create_importance_subsample():
    """THIS FUNCTION IS NOT FINISHED YET!"""
    passing = True

    def test_sample(run_dict):
        num_samples = 10
        days_mult = np.empty(shape=(
            num_samples,
            run_dict['num_days_sample'],
            3 + create_importance_subsample(**run_dict).shape[1])
        )
        for sample_num in range(num_samples):
            iss = create_importance_subsample(**run_dict)
            days_high = iss.iloc[:24*run_dict['num_days_high']]
            days_low = iss.iloc[24*run_dict['num_days_high']:]
            days_high = days_high.resample('24h').mean().dropna()
            days_low = days_low.resample('24h').mean().dropna()
            days = pd.concat([days_high, days_low])
            print(days)
            print('')

            days_mult[sample_num, :, :3] = list(dict.fromkeys(
                zip(days.index.year, days.index.month, days.index.day)
            ))
            days_mult[sample_num, :, 3:] = days.values

        return days_mult

    ts_data = models.load_time_series_data(model_name='1_region')
    generation_costs = pd.read_csv('benchmarks/1_region_cont/'
                                   'generation_costs.csv',
                                   index_col=0)
    generation_costs.index = pd.to_datetime(generation_costs.index)


    # Test the function for hourly blocks
    run_dict = {'ts_data': ts_data,
                'generation_costs': generation_costs,
                'num_days_sample': 3,
                'num_days_high': 2,
                'blocks': 'days'}

    days_mult = test_sample(run_dict)

    values_equal = np.empty(shape=days_mult.shape[1:], dtype=bool)
    for i in range(days_mult.shape[1]):
        for j in range(days_mult.shape[2]):
            values = days_mult[:, i, j]
            values_equal[i, j] = (values[0] == values[:]).all()

    # if not (values_equal[:, 2]).all():
    #     print('FAIL: weights are not equal.')
    #     passing = False
    # if not (values_equal[:run_dict['num_days_high']]).all():
    #     print('FAIL: high bins is not equal')
    #     passing = False
