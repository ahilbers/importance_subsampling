import numpy as np
import pandas as pd
import calliope


class Importance_Subsampling:
    def __init__(self, data_type='regression'):
        """Import demand and wind timeseries."""
        self.data = pd.read_csv('data/demand_wind.csv', index_col=0)
        self.data.index = pd.to_datetime(self.data.index)

    def calculate_varcosts(self, caps):
        """Calculate each timestep's variable cost and add to demand and
        wind timeseries.

        Parameters:
        -----------
        caps: list with installed capacities of baseload, mid-merit, peaking and
            wind technologies (in GW) used to calculate variable cost

        Returns:
        --------
        data_w_varcosts_pd: pandas DataFrame with 3 columns: 'demand', 'wind'
            and 'varcost' for each timestep
        """

        # Capacities of baseload, mid-merit, peaking and wind
        cap_bl, cap_mm, cap_pk, cap_wd = caps

        # Calculate variable costs, assuming infinite peaking capacity
        net_demand = self.data['demand'] - cap_wd * self.data['wind']
        net_demand[net_demand < 0] = 0
        gens_bl = np.minimum(net_demand, cap_bl)
        gens_mm = np.minimum(net_demand - gens_bl, cap_mm)
        gens_pk = net_demand - gens_bl - gens_mm
        varcosts = pd.DataFrame(0.005*gens_bl + 0.035*gens_mm + 0.1*gens_pk,
                                columns=['varcost'])
        data_w_varcosts = self.data.join(varcosts)

        return data_w_varcosts

    def importance_subsampling(self, caps, num_ts, num_ts_h):
        """Create demand and wind timeseries via importance subsampling,
        using a timestep's variable cost as a timestep's importance.

        Parameters:
        -----------
        caps: list with capacities used to calculate variable cost
        num_ts: total number of sampled timesteps
        num_ts_h: number of timesteps in bin with high variable cost

        Returns:
        --------
        sampled_demand: numpy array of sampled demand values
        sampled_wind: numpy array of sampled wind values
        weights: numpy array of timestep weights

        Raises:
        -------
        AssertionError: when num_ts < num_ts_h
        """

        assert (num_ts >= num_ts_h), 'num_ts_h cannot be greater than num_ts'

        num_ts_in = self.data.shape[0]    # length of input timeseries
        num_ts_l = num_ts - num_ts_h     # number of periods in lower bin
        data_vc = self.calculate_varcosts(caps)    # data with variable cost
        data_sorted = data_vc.sort_values(by='varcost', ascending=False)

        # Sample num_ts_h timesteps with highest variable cost and random
        # selection of size num_ts_l from those remaining
        sampled_data_high_vc = data_sorted.values[:num_ts_h]
        data_low_vc = data_sorted.values[num_ts_h:]
        index_low_vc = np.random.choice(np.arange(num_ts_in - num_ts_h),
                                        num_ts_l, replace=False)
        sampled_data_low_vc = data_low_vc[index_low_vc]
        sampled_data = np.concatenate((sampled_data_high_vc,
                                       sampled_data_low_vc))

        # Calculate weights and add them to sampled demand and wind data
        h_weight = 1 / num_ts_in
        l_weight = (num_ts_in - num_ts_h) / (num_ts_l * num_ts_in)
        # Ensure weights sum to 8760 (one year's worth of hourly timesteps)
        h_weights = np.ones(shape=(num_ts_h)) * h_weight * 8760
        l_weights = np.ones(shape=(num_ts_l)) * l_weight * 8760
        weights = np.concatenate((h_weights, l_weights))

        sampled_demand, sampled_wind = sampled_data[:, 0], sampled_data[:, 1]

        return sampled_demand, sampled_wind, weights

    def PSM_importance_subsampling(self, num_ts, num_ts_h):
        """Run PSM with demand and wind timeseries inputs generated using
        importance subsampling method.

        Parameters:
        -----------
        num_ts: total number of sampled timesteps. If a number besides
            120, 240, 480, 960, 1920, 3840 or 8760 is desired, the user
            should modify model_files/overrides.yaml to make a new entry
        num_ts_h: number of timesteps in bin with high variable cost

        Returns: none, but exports model outputs to CSV files
        """

        # Create Calliope model with correct number of timesteps
        groupname = str(num_ts) + 'ts'
        override_file = 'model_files/overrides.yaml:' + groupname
        model = calliope.Model('model_files/model.yaml',
                               override_file=override_file)

        # Stage 1: random subsampling of timesteps
        timesteps = np.random.choice(np.arange(self.data.shape[0]), num_ts,
                                     replace=False)
        data_sample_s1 = self.data.values[timesteps]
        demand_s1, wind_s1 = data_sample_s1[:, 0], data_sample_s1[:, 1]

        # Create model, adjust timesteps and weights, and run
        model.inputs.resource.loc['region1::demand_power'].values[:] = \
            -demand_s1[:]
        model.inputs.resource.loc['region1::wind'].values[:] = wind_s1[:]
        model.run()

        # Stage 1 capacities used to calculate variable cost
        res = model.results
        cap_bl = float(res.energy_cap.loc['region1::baseload'].values)
        cap_mm = float(res.energy_cap.loc['region1::midmerit'].values)
        cap_pk = float(res.energy_cap.loc['region1::peaking'].values)
        cap_wd = float(res.resource_area.loc['region1::wind'].values)
        caps = [cap_bl, cap_mm, cap_pk, cap_wd]

        # Stage 2 demand, wind and weights
        demand_s2, wind_s2, weights_s2 = \
            self.importance_subsampling(caps, num_ts, num_ts_h)

        # Create the model, input demand, wind and weights, run, and
        # export outputs to CSV
        model = calliope.Model('model_files/model.yaml',
                               override_file=override_file)
        model.inputs.resource.loc['region1::demand_power'].values[:] = \
            -demand_s2[:]
        model.inputs.resource.loc['region1::wind'].values[:] = wind_s2[:]
        model.inputs.timestep_weights.values = weights_s2
        model.run()
        model.to_csv('outputs')
