## Data Access:

- The Sixth Coupled Model Intercomparison Project (CMIP6) data, contributed by research labs worldwide, is publicly available (All ~ 30 terrabytes):
  - [Primary Access Point](https://esgf-index1.ceda.ac.uk/projects/esgf-ceda/)
  - [Cloud Archive on AWS](https://registry.opendata.aws/cmip6/)
- We accessed the data using [Casper](https://arc.ucar.edu/docs), a data analysis cluster at the National Center for Atmospheric Research (NCAR), which also served as our coding environment.
- Data exploration and processing are documented in the following notebooks under `/processing`:
  - `explore_data.ipynb`: Guidance on navigating the data, exploratory visuals, and models.
  - `data_process.ipynb`: Scripts for processing the output data.
- For baseline models, we used data from [ClimateBench](https://zenodo.org/records/7064308), which includes validation and test datasets.

## Dependencies

To run notebooks, ensure you have the following dependencies installed:

- Python 3.8 or higher
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `netCDF4`
  - `xarray`
  - `eofs`
  - `matplotlib`
  - `cartopy`
  - `scikit-learn`
  - `joblib`
  - `esem`

Dependencies can be installed via console:
```
pip install -r requirements.txt
```
Or inside a notebook:
```
!pip install -r requirements.txt
```

## Baseline Replication
1. **Download Data**:
   - Download `test.tar.gz` and `train_val.tar.gz` from [ClimateBench](https://zenodo.org/records/7064308) (~1gb).
   - Decompress the files into a single folder (e.g. `./dataset`).
   - Upload the files to Casper.

2. **Download Utilities**:
   - Download `utils.py` from this repository and place it in the same directory as the model notebooks.

3. **Set Data Path**:
   - Ensure that the `data_path` variable within `utils.py` is set to the directory where your data is stored (e.g. `./dataset`).

4. **Models**:
    - Pattern Scaling: [`Pattern_Scaling.ipynb`](https://github.com/lemoncastle/ClimateEmulation/blob/main/models/Pattern_Scaling.ipynb)
    - Gaussian Process: [`Gaussian_Process.ipynb`](https://github.com/lemoncastle/ClimateEmulation/blob/main/models/Gaussian_Process.ipynb)
    - Random Forests: [`Random_Forest.ipynb`](https://github.com/lemoncastle/ClimateEmulation/blob/main/models/Random_Forest.ipynb)

## Notebook Outputs

- At the bottom of each notebook, you will find functions to generate the images used in our report.
- This is found for:
  - Exploratory data visualizations in `explore_data.ipynb`.
  - Model predictions and comparisons in the respective modeling notebooks (`Pattern_Scaling.ipynb`, `Gaussian_Process.ipynb`, `Random_Forest.ipynb`).
    - The user can set different ```cmap``` colors, and save visuals as png if they wish.

## Checkpoints
### Checkpoint 11/7
- Paper Replication Progress
    - Data Exploration and visualizations - ```explore_data.ipynb```
    - Processing output data - ```data_process.ipynb```
    - Models - TBD
    - Report - Abstract, introduction (Overview, prior works, description of data).

### Checkpoint 11/21
- Code progress and checkpoint for Q2 project.

Watson-Parris, D., Rao, Y., Olivié, D., Seland, Ø., … “ClimateBench v1.0: A benchmark for data-driven climate projections”. Journal of Advances in Modeling Earth Systems 14, e2021MS002954: https://doi.org/10.1029/2021MS002954