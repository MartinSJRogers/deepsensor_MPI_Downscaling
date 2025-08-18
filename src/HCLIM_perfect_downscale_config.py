import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # I had to add this to get the code to see the GPU, but may not be necessary

# Now safe to import torch
import torch
import xarray as xr
import deepsensor.torch
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device
from deepsensor.data import DataProcessor, TaskLoader

import pandas as pd
import numpy as np

import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os, sys
import config_cmip


if len(sys.argv) != 2:
        print("Usage: Needs a config json file as as the second argument.")
        sys.exit(1)

json_file = sys.argv[1]

config = config_cmip.set_variables(json_file)

# model_run_type = config.run_type
rcm_directory = config["BASE_DIR"]
train_range = config["train_range"]
val_range = config["val_range"]
# Every other how many items in train and validation sets
date_subsample_factor = config["date_subsample_factor"]
batch_size = config["batch_size"]

# Open all files in directory corresponding to surface temperature
# List all files that contain 'tas_ANT' and end with '.nc'
rcm_files = [os.path.join(rcm_directory, f) for f in os.listdir(rcm_directory) if "tas_ANT" in f and f.endswith(".nc")]
rcm_opened = xr.open_mfdataset(rcm_files, concat_dim='time', combine='nested', parallel=True) ## To generalise

# Open auxillary data
elev_opened= xr.open_dataset(config["ELEV_FN"])
land_mask_opened = xr.open_dataset(config["LAND_MASK_FN"])


def preprocess_hclim_data(hclim_opened, hclim_elev_opened, hclim_lm_opened):
    """
    Steps specific to HCLIM data that are applied to convert initial netcdfs into format that 
    enables them to be passed to the dataprocessor() without generating an error:
    Steps:
        - Removing unneccesary vars

    Inputs:
        - hclim_raw : uploaded hclim data
        - hclim_elev_raw: hclim elevation data
        - hclim_lm_raw: hclim land mask
    """
    # Drop unneccessary vars. 
    hclim_raw = hclim_opened.drop_vars(["crs", "time_bnds"], errors="ignore")
    hclim_elev_raw = hclim_elev_opened.drop_vars(['Polar_Stereographic', 'lon_bnds', 'lat_bnds'], errors="ignore")
    hclim_lm_raw = hclim_lm_opened.drop_vars(['Polar_Stereographic', 'lon_bnds', 'lat_bnds'], errors="ignore")

    return hclim_raw, hclim_elev_raw, hclim_lm_raw

def preprocess_MetUM_data(metum_opened, metum_elev_opened, metum_lm_opened):
    """
    TO DO- add in processing steps for metum data
    """
    # metum_raw, metum_elev_raw, metum_lm_raw = metum_opened 
    metum_raw = metum_opened.drop_vars(["time_bnds","longitude_bnds","latitude_bnds","grid_longitude_bnds","grid_latitude_bnds","rotated_latitude_longitude"], errors="ignore")
    metum_elev_raw = metum_elev_opened.drop_vars(["rotated_latitude_longitude"], errors="ignore")
    metum_lm_raw = metum_lm_opened.drop_vars(["rotated_latitude_longitude"], errors="ignore")

    # # Rename coordinates
    # y_lat = "grid_latitude"
    # x_lon = "grid_longitude"

    return metum_raw, metum_elev_raw, metum_lm_raw #, y_lat, x_lon

if config["rcm_model"] == "HCLIM":
    rcm_raw, elev_raw, land_mask_raw = preprocess_hclim_data(rcm_opened, elev_opened, land_mask_opened)
    print (rcm_raw)
elif config["rcm_model"] == "MetUM":
    rcm_raw, elev_raw, land_mask_raw = preprocess_MetUM_data(rcm_opened, elev_opened, land_mask_opened)
    # rcm_raw = rcm_raw.rename({config["var_tas"]:"tas", config["y_coord"]:"y", config["x_coord"]:"x"})
    print (rcm_raw)


def upscale_rcm(rcm_processed, var_tas, y_lat, x_lon):
    """
    Upscale (reduce resolution) of HCLIM or METUM data when running perfect models. 
    """
    # TO DO check for correct grid spacing
    y_spacing = 138000  # Grid spacing for 'y'
    x_spacing = 138000  # Grid spacing for 'x'

    new_lat = np.arange(rcm_processed[y_lat].min(), rcm_processed[y_lat].max(), y_spacing)
    new_lon = np.arange(rcm_processed[x_lon].min(), rcm_processed[x_lon].max(), x_spacing)
    
    tas_coarse = rcm_processed[var_tas].interp({y_lat:new_lat, x_lon:new_lon}, method="linear")
    
    rcm_upscaled = xr.Dataset(
        {
            var_tas: tas_coarse
        },
        coords={
            "time": rcm_processed["time"],
            y_lat: new_lat,
            x_lon: new_lon
        }
    )
    
    return rcm_upscaled
    
if config["model_type"] == "perfect":
    rcm_coarse_raw = upscale_rcm(rcm_raw, config["var_tas"], config["y_coord"], config["x_coord"])


def mask_pressure_levels(gcm, var_names):
    """
    Apply mask so only data below each pressure level is retained. 
    For example, if surface level pressure = 750hPa in one pixel, the varianle variables will be masked in the 
    layer for 800, 900 and 1000. 
    This for imperfect applications that we are not running yet. 

    Inputs:
        - gcm: Variable data at each pressure level
        - Surface pressure level data

    Returns:
        - masked_gcm: GCM data masked by the pressure level on that date and location.
    
    """
    # Extract the surface pressure DataArray (assuming it's named 'sp' for surface pressure)
    sp = gcm['sp']  # Shape: (time, lat, lon)
    
    # Create a masked temperature dataset
    masked_gcm = gcm.copy()
    
    # Iterate through each temperature variable
    for var in gcm.data_vars:
        if var.startswith("ta"):  # Ensure it's a temperature variable
            # Extract pressure level from variable name (e.g., "ta500" → 500)
            pressure_level = int(var[2:])
            # Mask values where surface pressure is less than the pressure level
            masked_gcm[var] = gcm[var].where(sp >= pressure_level, np.nan)
    
    return masked_gcm

# data_processor = DataProcessor(x1_name="y", x2_name="x")
data_processor = DataProcessor(x1_name=config["y_coord"],x2_name = config["x_coord"])
rcm_coarse, rcm, elevation, land_mask = data_processor([rcm_coarse_raw, rcm_raw, elev_raw, land_mask_raw])

task_loader = TaskLoader(
    context = [rcm_coarse, elevation, land_mask] * config.context_set_day_count,
    target = rcm,
    context_delta_t = config.delta_t, 
    target_delta_t = 0,
    aux_at_targets=elevation
)

def generate_tasks(dates,progress=True, **kwargs):
    train_tasks = []
    for date in tqdm(dates, disable=not progress):
            task = task_loader(date, context_sampling=config.context_sampling_strategy,
                            target_sampling="all")
            task.remove_context_nans().remove_target_nans()
            train_tasks.append(task)
    return train_tasks

def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]  # assume 1st target set and 1D

    for task in val_tasks:  
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        if mean.ndim > 2:
            n = mean.shape[1] * mean.shape[2]
            mean = mean.reshape(1,n)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        if true.ndim > 2:
            m = true.shape[1] * true.shape[2]
            true = true.reshape(1,m)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))

def train_model_mixed_precision(data_processor, task_loader, train_range, date_subsample_factor, batch_size):

    set_gpu_default_device()
    
    model = ConvNP(data_processor, task_loader)
    opt = optim.Adam(model.model.parameters(), lr=config.l_rate)
    grad_scaler = GradScaler()  # Initialize GradScaler
    
    val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]
    val_tasks = generate_tasks(val_dates, progress=False)
    
    loss_records = []
    trainer = Trainer(model, lr=config.l_rate)

    for epoch in tqdm(range(config.num_epochs)):

        train_dates = pd.date_range(train_range[0], train_range[1])[::date_subsample_factor]
        train_tasks = generate_tasks(train_dates, progress=False)
    
        # Call the trainer with the tasks and mixed precision
        batch_losses = trainer(train_tasks, scaler=grad_scaler)
        
        # Calculate and record losses
        train_loss = np.mean(batch_losses)  # You can compute the loss from batch_losses

        # Compute and store validation loss (RMSE)
        val_loss = compute_val_rmse(model, val_tasks)
        loss_records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    
        # Optionally save the model
        if epoch % 5 == 0:
            model.save(f"{config.WEIGHT_DIR}/{config.run_type}/{config.run_identifier_fn}_{epoch}.json")

          # Save losses to CSV
        df_losses = pd.DataFrame(loss_records)
        loss_log_path = f"{config.LOSS_DIR}/{config.run_type}/{config.run_identifier_fn}_losses.csv"
        df_losses.to_csv(loss_log_path, index=False)
        
    return model

trained_model = train_model_mixed_precision(data_processor, task_loader, train_range, date_subsample_factor, batch_size)
