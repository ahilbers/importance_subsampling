# 2019_importance_subsampling
Data and model files for preprint "Importance Subsampling: Improving Power System Planning Under Climate-based Uncertainty (2019)"



## Notes 

- Before running the model, the data/demand_wind.csv should be split into two different .csv files, called data/demand.csv and data/wind.csv, with the correct demand and wind capacity factors. All demand values should be multipled by -1. The files data/demand_ex.csv and data/wind_ex.csv are provided as an example of the correct format.
- Model files are used in model generation Calliope (see acknowledgements). Model files are for Calliope version 0.6.2.



## Contains

- data/demand_wind.csv: dataset with estimates of hourly UK-wide demand levels and wind capacity factors over the period 1980-2015. All leap days (29-Feb) are removed.
- data/demand.csv: example of file format for the correct data/demand.csv file. The values from the 'demand' column in data/demand_wind.csv should be inserted here. All values should be negative.
- data/wind.csv: example of file format for the correct data/wind.csv file. The values from the 'wind' column in data/demand_wind.csv should be inserted here.
- model_files/model.yaml: main file used to create Calliope model
- model_files/locations.yaml: locations file used to create Calliope model. 
- model_files/techs.yaml: technology characteristics file used to create Calliope model.  


## Contact

Adriaan Hilbers. Department of Mathematics, Imperial College London. aph416@ic.ac.uk.



## Acknowledgements

Models are constructed in the modelling framework Calliope, created by Stefan Pfenninger and Bryn Pickering. See callio.pe or the following paper for details:

Pfenninger et al., (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:10.21105/joss.00825

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following papers:

Bloomfield, H. C., Brayshaw, D. J., Shaffrey, L. C., Coker, P. J. and Thornton, H. E. (2016) Quantifying the increasing sensitivity of power systems to climate variability. Environmental Research Letters, 11 (12). 124025. ISSN 1748­ 9326 doi: 10.1088/1748­9326/11/12/124025

Cannon, D. J., Brayshaw, D. J., Methven, J., Coker, P. J., & Lenaghan, D. (2015). Using reanalysis data to quantify extreme wind power generation statistics: A 33 year case study in Great Britain. Renewable Energy, 75 , 767 – 778. doi:10.1016/j.renene.2014.10.024

Drew, D. R., Cannon, D. J., Brayshaw, D. J., Barlow, J. F., & Coker, P. J. (2015). The impact of future offshore wind farms on wind power generation in Great Britain. Resources, 4 , 155–171. doi:10.3390/resources4010155.
