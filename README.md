# chemical-emissions
Repository for code relating to paper entitled "Greenhouse gas emissions from global petrochemical production" output from the [C-THRU](https://www.c-thru.org/) project.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## About

This repository contains the open‑source code, notebooks, and curated auxiliary data that underpin the study **“Greenhouse‑gas emissions from global petrochemical production”** by Meng *et al.* (in review, 2024).  The workflow harmonises several commercial and open datasets to quantify historical, present, and scenario‑based future emissions from **37 379** production facilities worldwide.

The project is part of the [C‑THRU](https://www.c‑thru.org/) initiative on resource efficiency in the petrochemical sector.

## Workflow

```
raw proprietary spreadsheets → **data_extraction** → **data_processing** → **data_combination** → **analysis**
```

1. **data_extraction** – scripts/notebooks to parse commercial datasets (ICIS Supply & Demand, IHS PEP, EcoInvent, CarbonMinds) into clean *feather* tables.  
2. **data_processing** – unit harmonisation, energy/feedstock balancing, and calculation of production volumes by facility.  
3. **data_combination** – merges processed streams into a single facility‑level panel dataset, attaches emission‑factor libraries, and performs uncertainty propagation.  
4. **analysis** – Jupyter notebooks that reproduce the figures and tables of the paper, including scenario modelling of efficiency, carbon capture, and alternative feedstocks.

> ⚠️ The repository does **not** ship the raw commercial spreadsheets; you will need your own licensed copies and must update the file paths in the extraction notebooks before running the pipeline.

## Repository layout

| Path | What’s inside                                                                                          |
|------|--------------------------------------------------------------------------------------------------------|
| `data/extra_inputs/` | Curated conversion factors, emission factors, matching tables, etc.                                    |
| `data_extraction/`   | Python / Jupyter scripts to ingest ICIS, IHS, EcoInvent and CarbonMinds files.                         |
| `data_processing/`   | Jupyter notebooks that harmonise units and calculate facility production.                              |
| `data_combination/`  | Notebook `process_iterations.ipynb` that assembles the master dataset.                                 |
| `analysis/`          | Notebooks that generate paper figures (`fig1_overview.ipynb`, …) and scenario analysis (`scenarios/`). |
| `LICENSE`            | MIT License.                                                                                           |

## Quick start

```bash
# clone the repo
git clone https://github.com/Resource-Efficiency-Collective/chemical-emissions.git
cd chemical-emissions

# create and activate a fresh environment (conda recommended)
conda create -n chememis python=3.10
conda activate chememis

# install dependencies
pip install -r requirements.txt   # see below
```

### Minimum dependencies

| Package | Tested version |
|---------|----------------|
| Python  | ≥ 3.9 |
| pandas  | ≥ 2.0 |
| numpy   | ≥ 1.24 |
| pyarrow | ≥ 14.0 *(for Feather I/O)* |
| jupyterlab | ≥ 4.0 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |
| plotly | ≥ 5.18 |

Install with:

```bash
pip install pandas numpy pyarrow jupyterlab matplotlib seaborn plotly
```

## Running the pipeline

1. Place the raw proprietary spreadsheets in a folder of your choice.
2. Update the constants at the top of each `data_extraction/*` script.
3. Execute the notebooks in the order shown in the **Workflow** section.
4. Open the notebooks in `analysis/` to reproduce the figures.  Generated outputs are saved to `output/` (created automatically).

All notebooks are parameterised; you can pass alternative scenario parameters through the `analysis/scenarios/*` notebooks to explore decarbonisation pathways.

## Citation

If you use this code or derived data in your work, please cite:

```
Meng F., Cullen L., Lupton R., & Cullen J. M.  
“Greenhouse‑gas emissions from global petrochemical production (1978‑2050)”  
In review, 2024.  Pre‑print available from the C‑THRU project website.
```


## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
