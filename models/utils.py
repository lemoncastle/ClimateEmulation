import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors

### Set data path
data_path = "./dataset/"

min_co2 = 0.
max_co2 = 9500
def normalize_co2(data):
    return data / max_co2

def un_normalize_co2(data):
    return data * max_co2

min_ch4 = 0.
max_ch4 = 0.8
def normalize_ch4(data):
    return data / max_ch4

def un_normalize_ch4(data):
    return data * max_ch4


def create_predictor_data(data_sets, n_eofs=5):
    """  
    Loads emissions input datasets, computes aerosol EOFs,
    normalizes greenhouse gases, and returns a full predictor table.
    
    Args:
        data_sets list(str): names of datasets
        n_eofs (int): number of eofs to create for aerosol variables
    """
        
    # Create training and testing arrays
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    X = xr.concat([xr.open_dataset(data_path + f"inputs_{file}.nc") for file in data_sets], dim='time')
    X = X.assign_coords(time=np.arange(len(X.time)))

    # Compute EOFs for BC
    bc_solver = Eof(X['BC'])
    bc_eofs = bc_solver.eofsAsCorrelation(neofs=n_eofs)
    bc_pcs = bc_solver.pcs(npcs=n_eofs, pcscaling=1)

    # Compute EOFs for SO2
    so2_solver = Eof(X['SO2'])
    so2_eofs = so2_solver.eofsAsCorrelation(neofs=n_eofs)
    so2_pcs = so2_solver.pcs(npcs=n_eofs, pcscaling=1)

    # Convert to pandas
    bc_df = bc_pcs.to_dataframe().unstack('mode')
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    so2_df = so2_pcs.to_dataframe().unstack('mode')
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame({
        "CO2": normalize_co2(X["CO2"].data),
        "CH4": normalize_ch4(X["CH4"].data)
    }, index=X["CO2"].coords['time'].data)

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs, (so2_solver, bc_solver)


def get_test_data(file, eof_solvers, n_eofs=5):
    """
    Loads a test input dataset and projects aerosol fields onto the
    previously fitted EOFs to generate test-time predictor features.
    
    Args:
        file str: name of datasets
        n_eofs (int): number of eofs to create for aerosol variables
        eof_solvers (Eof_so2, Eof_bc): Fitted Eof objects to use for projection
    """
        
    # Create training and testing arrays
    X = xr.open_dataset(data_path + f"inputs_{file}.nc")
        
    so2_pcs = eof_solvers[0].projectField(X["SO2"], neofs=5, eofscaling=1)
    so2_df = so2_pcs.to_dataframe().unstack('mode')
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    bc_pcs = eof_solvers[1].projectField(X["BC"], neofs=5, eofscaling=1)
    bc_df = bc_pcs.to_dataframe().unstack('mode')
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame({
        "CO2": normalize_co2(X["CO2"].data),
        "CH4": normalize_ch4(X["CH4"].data)
    }, index=X["CO2"].coords['time'].data)

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs


def create_predictdand_data(data_sets):
    """
    Loads climate output datasets (tas, pr, pr90, dtr), averages ensemble
    members, converts precipitation units to mm/day, and returns dataset
    """
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    Y = xr.concat([xr.open_dataset(data_path + f"outputs_{file}.nc") for file in data_sets], dim='time').mean("member")
    # Convert the precip values to mm/day
    Y["pr"] *= 86400
    Y["pr90"] *= 86400
    return Y


def get_rmse(truth, pred):
    """
    Computes area-weighted RMSE over the spatial grid.
    """
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon'])).data

def nrmse_spatial(truth, pred):
    """
    Computes spatially averaged NRMSE across the globe using
    weighted RMSE divided by weighted truth mean.
    """
    pred = pred.assign_coords(time=truth.time.values)
    weights = np.cos(np.deg2rad(truth.lat))
    
    # weighted RMSE across lat/lon
    rmse = np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon']))
    
    # weighted mean of truth (normalization)
    truth_mean = truth.weighted(weights).mean(['lat', 'lon'])
    
    return (rmse / np.abs(truth_mean)).data

def nrmse_global(truth, pred):
    """
    Computes NRMSE based on the error between global-mean time series.
    """
    pred = pred.assign_coords(time=truth.time.values)
    weights = np.cos(np.deg2rad(truth.lat))

    # compute global means
    truth_mean = truth.weighted(weights).mean(['lat', 'lon'])
    pred_mean  = pred.weighted(weights).mean(['lat', 'lon'])

    return (np.abs(pred_mean - truth_mean) / np.abs(truth_mean)).data

def nrmse_total(truth, pred, alpha=5):
    """
    Combines spatial NRMSE and global NRMSE into a weighted total score.
    """
    pred = pred.assign_coords(time=truth.time.values)
    s = nrmse_spatial(truth, pred)
    g = nrmse_global(truth, pred)
    return s + alpha * g

def plot_mean_2080_2100(truth, emulated, varname, cmap="coolwarm",
                           diff_cmap="PRGn", savefig=False, Model=None):
    """
    Computes and visualizes the 2080–2100 mean climate fields for truth,
    model projections, and their spatial difference on Robinson projection.
    """

    # --- Compute mean 2080-2100 fields ---
    emulated = emulated.assign_coords(time=truth.time.values)
    truth_mean = truth.sel(time=slice(2080, 2100)).mean("time")
    pred_mean  = emulated.sel(time=slice(2080, 2100)).mean("time")
    
    diff = truth_mean - pred_mean

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6),
                             subplot_kw={"projection": ccrs.Robinson()})
    
    divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)

    # === TRUE FIELD MEAN ===
    ax = axes[0]
    truth_mean.plot(
        ax=ax, cmap=cmap, norm=divnorm, transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": f"{varname} (2080–2100 Mean)"}
    )
    ax.set_title(f"{varname} True Mean (2080–2100)")
    ax.coastlines()

    # === EMULATED FIELD MEAN ===
    ax = axes[1]
    pred_mean.plot(
        ax=ax, cmap=cmap, norm=divnorm, transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": f"{varname} (2080–2100 Mean)"}
    )
    ax.set_title(f"{varname} {Model} Emulated Mean (2080–2100)")
    ax.coastlines()

    # === DIFFERENCE ===
    ax = axes[2]
    diff.plot(
        ax=ax, cmap=diff_cmap, transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": f"{varname} Difference (2080–2100 Mean)"}
    )
    ax.set_title(f"{varname} Difference\n(Truth − {Model} Mean)")
    ax.coastlines()

    fig.suptitle(f"{varname} 2080–2100 Mean Climate Comparison", fontsize=18)
    fig.tight_layout()

    if savefig:
        fig.savefig(f"{Model}_{varname}_mean_2080_2100.png",
                    dpi=250, bbox_inches="tight")

    plt.show()