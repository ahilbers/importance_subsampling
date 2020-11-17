[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# importance_subsampling




## Summary

This repository contains data, model files and example code for the paper [*Importance subsampling for power system planning under multi-year demand and weather variability*](https://ieeexplore.ieee.org/abstract/document/9183591) (2020) (publicly available version [here](https://arxiv.org/abs/2008.10300)).

**Note**: This paper is a generalisation of an older paper: [*Importance subsampling: improving power system planning under climate-based uncertainty*](https://doi.org/10.1016/j.apenergy.2019.04.110) that appeared in Applied Energy (publicly available version [here](https://arxiv.org/abs/1903.10916)). If you've come to this repository from that paper, check out the new paper, as it generalises the method significantly. If you want to see the code for the older paper, check out the branch `2019_applied_energy_paper` of this repository.

The test-case models used in the paper are specific instances of a more general class of test power system models, available open-source in [this repository](https://github.com/ahilbers/renewable_test_PSMs), where they are documented and available in a more general form. If you want to use these models for your own research, its easier to use that respository instead of this one.




## Usage

The easiest way to get started is by running

```
python3 main.py
```

from a command line. This runs a simple application of *importance subsampling* on the *LP* model. A directory `results` is created with the relevant model outputs. Arguments can be specified direclty in `main.py`, which contains extensive documentation. This function creates a single file `summary_outputs.csv` with a high-level overview of relevant model outputs, and a directory called `full_outputs` that contains the full set of model outputs.

In the default settings, calling `main.py` gives a lot of logging statements. If you want to get more concise print statements, run instead `python3 main.py  --logging_level ERROR`




## Contains

### Modelling & data files

- `models/`: power system model generating files, for `Calliope` (see acknowledgements)
- `data/`: demand and weather time series data.


### Code

- `main.py`: simple implementation of _importance subsampling_ on the power system model introduced in the paper. This model is created and solved in the open-source energy system modelling framework `Calliope` (see acknowledgements). See _requirements & installation_ for setup details. `main.py` can be run directly from a command line. A directory `results` is created with the relevant model outputs.
- `iss.py`: the code for the *importance_subsampling* methodology.
- `models.py`: some utility code for the models.
- `tests.py`: some tests to check if the models are behaving as expected.


### Supplementary material

- Additional model output plots, see README inside the directory `supplementary material`.




## Requirements & Installation

Since `main.py`, containing all code, is a short file with only a few functions, and is tailored to the specific power system model syntax, it's probably easier to fork/copy-paste and edit any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` requires:
- Python modules:
  - `Calliope 0.6.6`: A (fully open-source) energy system model generator. See [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation. Everything also works with `0.6.5`. Other versions have not bee checked, but may still work.
  - basic: `numpy`, `pandas`.
- Other:
  - `cbc`: open-source optimiser: see [this link](https://projects.coin-or.org/Cbc) for installation. Other solvers (e.g. `gurobi`) are also possible -- the solver can be specified in `model_files/model.yaml`.




## How to cite

If you use this repository for further research, please cite the following papers:

- AP Hilbers, DJ Brayshaw, A Gandy (2019). Importance subsampling: improving power system planning under climate-based uncertainty. Applied Energy, 251 (113114), doi:[10.1016/j.apenergy.2019.04.110](https://doi.org/10.1016/j.apenergy.2019.04.110).

- AP Hilbers, DJ Brayshaw, A Gandy (2020). Importance subsampling for power system planning under multi-year demand and weather uncertainty. In proceedings of the 16th International Conference on Probabilistic Methods Applied to Power Systems.



## Contact

[Adriaan Hilbers](https://ahilbers.github.io/). Department of Mathematics, Imperial College London. [a.hilbers17@imperial.ac.uk](mailto:a.hilbers17@imperial.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](https://callio.pe) or the following paper for details:

- Pfenninger, S. and Pickering, B. (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](https://doi.org/10.21105/joss.00825).

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following paper and dataset:

- Bloomfield, H. C., Brayshaw, D. J. and Charlton-Perez, A. (2019) Characterising the winter meteorological drivers of the European electricity system using Targeted Circulation Types. Meteorological Applications. ISSN 1469-8080 (In Press). doi:[10.1002/met.1858](https://doi.org/10.1002/met.1858)

- HC Bloomfield, DJ Brayshaw, A Charlton-Perez (2020). MERRA2 derived time series of European country-aggregate electricity demand, wind power generation and solar power generation. University of Reading. Dataset. doi:[10.17864/1947.239](https://doi.org/10.17864/1947.239)
