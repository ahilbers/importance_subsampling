# 2019_importance_subsampling
Data, model files, example code and supplementary material related to the paper _Importance subsampling: improving power system planning under climate-based uncertainty (2019)_.




## Contains

### Modelling & data files

All files required to create the PSM. 

- `model_files/`: repository containing all model files for the employed PSM, in open source model generator _Calliope_ (see acknowledgements), version 0.6.2
- `data/`: repository containing data files. These can be substituted into the `model_files/demand.csv` and `model_files/wind_r.csv` (but be careful to respect the style in those files to ensure the PSM is built correctly, e.g. multiply demand by -1!)
  - `demand_wind.csv`: demand and wind timeseries used in paper, across 36-year period 1980-2015
  - `demand_wind_national_grid.csv`: demand and wind timeseries used in supplementary material, across 8-year period 2008-2015


### Sample code

Sample code that builds the PSM and runs it using timeseries data subsampled using the _importance subsampling_ methodology. Relevant files:

- `main.py`: main python script
- `scripts.py`: relevant definitions and scripts

Designed to run with `Python` 3.6 with `numpy` and `pandas` and _Calliope_ version 0.6.2. It is run as follows from a unix command line. Note that you need to install _Calliope_ first!

    $ source activate calliope      # start Calliope environment
    (calliope) $ python3 main.py    # run code

A directory `outputs` is created with all relevant PSM outputs.


### Supplementary material

- `supplementary_material.pdf`: PDF file with results of applying _importance subsampling_ to the demand_wind_national_grid.csv timeseries data instead of demand_wind.csv as in the paper.





## Contact

Adriaan Hilbers. Department of Mathematics, Imperial College London. aph416@ic.ac.uk.




## Acknowledgements

Models are constructed in the modelling framework _Calliope_, created by Stefan Pfenninger and Bryn Pickering. See callio.pe or the following paper for details:

- Pfenninger et al., (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:10.21105/joss.00825

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following papers:

- Bloomfield, H. C., Brayshaw, D. J., Shaffrey, L. C., Coker, P. J. and Thornton, H. E. (2016) Quantifying the increasing sensitivity of power systems to climate variability. Environmental Research Letters, 11 (12). 124025. ISSN 1748­ 9326 doi: 10.1088/1748­9326/11/12/124025

- Cannon, D. J., Brayshaw, D. J., Methven, J., Coker, P. J., & Lenaghan, D. (2015). Using reanalysis data to quantify extreme wind power generation statistics: A 33 year case study in Great Britain. Renewable Energy, 75 , 767 – 778. doi:10.1016/j.renene.2014.10.024

- Drew, D. R., Cannon, D. J., Brayshaw, D. J., Barlow, J. F., & Coker, P. J. (2015). The impact of future offshore wind farms on wind power generation in Great Britain. Resources, 4 , 155–171. doi:10.3390/resources4010155.
