# 2019_importance_subsampling
Data, model files, example code and supplementary material related to the paper [Importance subsampling: improving power system planning under climate-based uncertainty](https://www.sciencedirect.com/science/article/pii/S0306261919307639) (2019). A publicly available preprint can be found [here](https://arxiv.org/pdf/1903.10916.pdf). 




## Contains

### Modelling & data files

- `model_files/`: model files for employed PSM in open source model generator `Calliope` (see acknowledgements), version 0.6.4
- `data/`: demand and weather time series data
  - `demand_wind.csv`: demand and wind time series used in paper
  - `demand_wind_national_grid.csv`: demand and wind time series used in supplementary material


### Code

`main.py` contains a simple implementation of _importance subsampling_ on the power system model introduced in the paper. This model is created and solved in the open-source energy system modelling framework `Calliope` (see acknowledgements). See [requirements & installation](##Requirements & Installation) for setup details. The following assumes that `Calliope` is installed in a virtual environment called `calliope`. 

It can be run from a unix command line as follows: 

    $ conda activate calliope       # activate virtual environment
    (calliope) $ python3 main.py    # run code

A directory `results` is created with all relevant PSM outputs.


### Supplementary material

- `supplementary_material.pdf`: PDF file with results of applying _importance subsampling_ to the `demand_wind_national_grid.csv` time series data instead of `demand_wind.csv` as in the original paper




## Requirements & Installation

Since `main.py`, containing all code, is a short file with only a few functions, it's probably easier to directly copy-paste any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` requires:
- Python modules:
  - `Calliope 0.64`:  see [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation. By default, `Calliope` is installed in a virtual environment, which we assume is called `calliope`.
  - `numpy 1.62.2`
  - `pandas 0.24.2`
- Other:
  - `cbc`: open-source optimiser: see [this link](https://projects.coin-or.org/Cbc) for installation. Other solvers (e.g. `gurobi`) are also possible -- the solver can be specified in `model_files/model.yaml`.





## Contact

Adriaan Hilbers. Department of Mathematics, Imperial College London. [aph416@ic.ac.uk](mailto:aph416@ic.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](callio.pe) or the following paper for details:

- Pfenninger et al., (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:10.21105/joss.00825.

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following papers:

- Bloomfield, H. C., Brayshaw, D. J., Shaffrey, L. C., Coker, P. J. and Thornton, H. E. (2016) Quantifying the increasing sensitivity of power systems to climate variability. Environmental Research Letters, 11 (12). 124025. ISSN 1748­ 9326 doi: 10.1088/1748­9326/11/12/124025

- Cannon, D. J., Brayshaw, D. J., Methven, J., Coker, P. J., & Lenaghan, D. (2015). Using reanalysis data to quantify extreme wind power generation statistics: A 33 year case study in Great Britain. Renewable Energy, 75 , 767 – 778. doi:10.1016/j.renene.2014.10.024

- Drew, D. R., Cannon, D. J., Brayshaw, D. J., Barlow, J. F., & Coker, P. J. (2015). The impact of future offshore wind farms on wind power generation in Great Britain. Resources, 4 , 155–171. doi:10.3390/resources4010155.
