# 2019_importance_subsampling
Data, model files, example code and supplementary material related to the paper [Importance subsampling: improving power system planning under climate-based uncertainty](https://www.sciencedirect.com/science/article/pii/S0306261919307639) (2019). A publicly available preprint can be found [here](https://arxiv.org/pdf/1903.10916.pdf). 




## Contains

### Modelling & data files

- `model_files/`: power system model generating files, for `Calliope` (see acknowledgements)
- `data/`: demand and weather time series data
  - `demand_wind.csv`: demand and wind time series used in paper


### Code

- `main.py`: simple implementation of _importance subsampling_ on the power system model introduced in the paper. This model is created and solved in the open-source energy system modelling framework `Calliope` (see acknowledgements). See _requirements & installation_ for setup details. `main.py` can be run directly from a command line. A directory `results` is created with the relevant model outputs.


### Supplementary material

- `supplementary_material.pdf`: PDF file with results of applying _importance subsampling_ to the `demand_wind_national_grid.csv` time series data instead of `demand_wind.csv` as in the original paper




## Requirements & Installation

Since `main.py`, containing all code, is a short file with only a few functions, it's probably easier to directly copy-paste any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` requires:
- Python modules:
  - `Calliope 0.6.4`:  see [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation.
  - `numpy 1.62.2`
  - `pandas 0.24.2`
- Other:
  - `cbc`: open-source optimiser: see [this link](https://projects.coin-or.org/Cbc) for installation. Other solvers (e.g. `gurobi`) are also possible -- the solver can be specified in `model_files/model.yaml`.




## How to cite

If you use this repository for further research, please cite the following paper:

- AP Hilbers, DJ Brayshaw, A Gandy (2019). Importance subsampling: improving power system planning under climate-based uncertainty. Applied Energy, 251 (113114), doi:[10.1016/j.apenergy.2019.04.110](https://doi.org/10.1016/j.apenergy.2019.04.110).



## Contact

Adriaan Hilbers. Department of Mathematics, Imperial College London. [aph416@ic.ac.uk](mailto:aph416@ic.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](hptts://callio.pe) or the following paper for details:

- S Pfenninger and B Pickering (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](https://doi.org/10.21105/joss.00825).

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following papers:

- HC Bloomfield, DJ Brayshaw, LC Shaffrey, PJ Coker and HE Thornton (2016). Quantifying the increasing sensitivity of power systems to climate variability. Environmental Research Letters, 11 (12). 124025. ISSN 1748­ 9326 doi:[10.1088/1748­9326/11/12/124025](https://doi.org/10.1088/1748­9326/11/12/124025)

- DJ Cannon, DJ Brayshaw, J Methven, PJ Coker, D Lenaghan (2015). Using reanalysis data to quantify extreme wind power generation statistics: A 33 year case study in Great Britain. Renewable Energy, 75 , 767 – 778. doi:[10.1016/j.renene.2014.10.024](https://doi.org/10.1016/j.renene.2014.10.024)

- DR Drew, DJ Cannon, DJ Brayshaw, JF Barlow, PJ Coker (2015). The impact of future offshore wind farms on wind power generation in Great Britain. Resources, 4 , 155–171. doi:[10.3390/resources4010155](https://doi.org/10.3390/resources4010155).
