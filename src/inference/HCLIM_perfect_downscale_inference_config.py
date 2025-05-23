import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Now safe to import torch
import torch
import xarray as xr
import deepsensor.torch
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device
from deepsensor.data import DataProcessor, TaskLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
import config_cmip_inference as config

rcm_directory = config.BASE_DIR
test_dates = pd.date_range(config.test_range[0], config.test_range[1])
plot_decision = config.plot

# Open all files in directory corresponding to surface temperature
# List all files that contain 'tas_ANT' and end with '.nc'

rcm_files = [os.path.join(rcm_directory, f) for f in os.listdir(rcm_directory) if "tas_ANT" in f and f.endswith(".nc")]
for f in rcm_files:
    if not os.path.isfile(f):
        print("Missing or invalid file:", f)
rcm_opened = xr.open_mfdataset(rcm_files, concat_dim='time', combine='nested', parallel=False) ## To generalise

# Open auxillary data
elev_opened= xr.open_dataset(config.ELEV_FN)
land_mask_opened = xr.open_dataset(config.LAND_MASK_FN)


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
    metum_raw, metum_elev_raw, metum_lm_raw = metum_opened 
    return metum_raw, metum_elev_raw, metum_lm_raw

if config.rcm_model == "HCLIM":
    rcm_raw, elev_raw, land_mask_raw = preprocess_hclim_data(rcm_opened, elev_opened, land_mask_opened)    
elif config.rcm_model == "MetUM":
    rcm_raw, elev_raw, land_mask_raw = preprocess_MetUM_data(rcm_opened, elev_opened, land_mask_opened) 


def upscale_rcm(rcm_processed):
    """
    Upscale (reduce resolution) of HCLIM or METUM data when running perfect models. 
    """
    # TO DO check for correct grid spacing
    y_spacing = 138000  # Grid spacing for 'y'
    x_spacing = 138000  # Grid spacing for 'x'

    new_lat = np.arange(rcm_processed['y'].min(), rcm_processed['y'].max(), y_spacing)
    new_lon = np.arange(rcm_processed['x'].min(), rcm_processed['x'].max(), x_spacing)
    
    tas_coarse = rcm_processed['tas'].interp(y=new_lat, x=new_lon, method="linear")
    
    rcm_upscaled = xr.Dataset(
        {
            "tas": tas_coarse
        },
        coords={
            "time": rcm_processed["time"],
            "y": new_lat,
            "x": new_lon
        }
    )
    
    return rcm_upscaled
    
if config.model_type == "perfect":
    rcm_coarse_raw = upscale_rcm(rcm_raw)


def generate_tasks(date,progress=True, **kwargs):
   
    task = task_loader(date, context_sampling=config.context_sampling_strategy,
                    target_sampling="all")
    task.remove_context_nans().remove_target_nans()

    return task

data_processor = DataProcessor(x1_name="y", x2_name="x")
rcm_coarse, rcm, elevation, land_mask = data_processor([rcm_coarse_raw, rcm_raw, elev_raw, land_mask_raw])

task_loader = TaskLoader(
    context = [rcm_coarse, elevation, land_mask] * config.context_set_day_count,
    target = rcm,
    context_delta_t = config.delta_t, 
    target_delta_t = 0,
    aux_at_targets=elevation
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded = ConvNP(data_processor, task_loader, config.MODEL_FN)

predictions = []
for test_date in tqdm(test_dates):
    test_task = generate_tasks(test_date)
    prediction = model_loaded.predict(test_task, X_t=rcm_raw)
    predictions.append(prediction)



def calculate_mae(predictions_to_val, rcm_raw_val, test_date_range, plot=False):
    mae_per_date = []
    
    for prediction, test_date in zip(predictions_to_val, test_date_range):
        real = rcm_raw_val['tas'].sel(time=test_date)
        mean_pred = prediction['tas']['mean']
    
        # Compute absolute error and take mean (ignoring NaNs)
        mae = np.nanmean(np.abs(mean_pred - real))/2
        mae_per_date.append(mae)
    
    if plot:
        # Plot the MAE over time
        plt.figure(figsize=(10, 5))
        plt.plot(test_dates, mae_per_date, marker='o', linestyle='-')
        plt.xlabel('Date')
        plt.ylabel('Mean Absolute Error')
        plt.title('Mean Absolute Error per Date')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        plt.grid()
        save_path = f"{config.PLOT_DIR}/{config.run_type}/{config.plot_fn_mae}"
        plt.savefig(save_path, format='png')

calculate_mae(predictions, rcm_raw, test_dates, plot=plot_decision)
    
def get_power_spectrum(image):
    # Remove NaNs (required for FFT)
    image = np.nan_to_num(image)

    # Subtract mean to avoid spike at zero frequency
    image = image - np.mean(image)

    # 2D FFT
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)

    # Power spectrum
    power = np.abs(Fshift)**2

    return power

def get_radial_profile(data):
    y, x = np.indices(data.shape)
    center = np.array(data.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)

    return radial_profile

def calculate_power_spectrum(predictions_to_val, rcm_raw_val, test_date_range, plot=False):

    for prediction, test_date in zip(predictions_to_val, test_date_range[:10]):
        # Extract predicted and real images
        mean_pred = prediction['tas']['mean'][0].values  # Predicted: shape (lat, lon)
        real = rcm_raw['tas'].sel(time=test_date).values  # Real: convert xarray to numpy
    
        # Ensure 2D
        if real.ndim > 2:
            real = real[0]
    
        # Compute power spectra
        pred_power2d = get_power_spectrum(mean_pred)
        real_power2d = get_power_spectrum(real)
    
        pred_power1d = get_radial_profile(pred_power2d)
        real_power1d = get_radial_profile(real_power2d)
    
        if plot:
            # Plot both on the same log-log plot
            plt.figure(figsize=(10,5))
            plt.loglog(pred_power1d, label='Prediction', color='blue')
            plt.loglog(real_power1d, label='Real', color='red')
            plt.title(f"Power Spectrum - {test_date.strftime('%Y-%m-%d')}")
            plt.xlabel("Spatial Frequency (log scale)")
            plt.ylabel("Power (log scale)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the plot as a PNG file
            save_path = f"{config.PLOT_DIR}/{config.run_type}/{config.plot_fn_psd}"
            plt.savefig(save_path, format='png')

calculate_power_spectrum(predictions, rcm_raw, test_dates, plot=False)