import scripts as sc

# Run PSM once with subsample generated using importance subsampling
# PSM outputs are exported to CSV files in new directory 'outputs'
ISS = sc.Importance_Subsampling()
ISS.PSM_importance_subsampling(num_ts=8760, num_ts_h=60)

# Note: when running model, a warning of the type
#
#  * `monetary` interest rate of zero for technology XX, setting depreciation
#  rate as 1/lifetime
#
# may appear. This happens since the depreciation rate is set to 0 and
# annualised installation costs are used. This is an intentional choice.
