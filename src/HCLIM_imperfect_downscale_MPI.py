import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # I had to add this to get the code to see the GPU, but may not be necessary

# Now safe to import torch
import deepsensor.torch
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device
from deepsensor.data import DataProcessor, TaskLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import os
import interpolate_to_ease2sh
import iris
from iris.cube import CubeList
import torch.optim as optim
import config_cmip as config
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import multiprocessing
import time
from functools import partial



import os
os.environ["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")


model_run_type = config.run_type
rcm_directory = config.BASE_DIR
gcm_directory = config.GCM_DIR
data_range = config.data_range
train_range = config.train_range
val_range = config.val_range
# Every other how many items in train and validation sets
date_subsample_factor = config.date_subsample_factor
batch_size = config.batch_size
selected_vars = config.selected_vars
p_levels = config.p_levels
land_mask_iris = iris.load_cube(config.LAND_MASK_FN)
elevation_iris = iris.load_cube(config.ELEV_FN)
workers = 8
task_loader = None

def process_gcm_mpi_data(gcm_directory, selected_vars, p_levels):
    """
    Process MPI GCM data: Only open datasets pertaining to variables, dates and pressure levels defined in config.
    Current input is masked data, so only additional step is to reproject to EASE2.

    Input:
        - gcm_directory: directory containing masked MPI data

    Ouput:
        - gcm_var_dict: All masked GCM data in in EASE2 projection in dictionary where each key corresponds to a different 
        variable selected in the config file. 
    
    """
    gcm_files = [os.path.join(gcm_directory, f) for f in os.listdir(gcm_directory) if f.endswith(".nc")]
    # Create a dictionary containing all variable xarrays for the defined date range.
    gcm_var_dict = {}
    
    for var in selected_vars:
        # Filter only files that correspond to the current variable
        var_files = [f for f in gcm_files if os.path.basename(f).startswith(f"{var}_")]
    
        if not var_files:
            print(f"No files found for variable: {var}")
            continue
    
        # Open all matching files as a single multi-file dataset
        var_ds = xr.open_mfdataset(var_files, combine='by_coords')
    
        # Check that the variable exists in the dataset
        if var not in var_ds:
            print(f"Variable {var} not found in dataset files.")
            continue
    
        # Slice the dataset by the desired time range
        sliced_var_ds = var_ds[var].sel(time=slice(*data_range))
    
        # Convert to Iris cube and remove extra attributes prior to reprojection
        gcm_var_cube = sliced_var_ds.to_iris()  # Or use a sorted list of filenames
        for key in ['tracking_id', 'history', "creation_date", "comment"]:
            gcm_var_cube.attributes.pop(key, None)  # Remove if it exists
    
        gcm_var_ease2 = interpolate_to_ease2sh.reproject_mpi(gcm_var_cube)
        
        gcm_var_dict[var] = gcm_var_ease2

    return gcm_var_dict


def preprocess_hclim_rcm_data(rcm_directory):
    """
    Process HCLIM RCM data: open relevant files, convert to iris and reproject to EASE2 grid. 
    Remove unnecessary variables which can cause errors and merge.

    Inputs:
        - rcm_directory : directory of RCM HCLIM data

    Output:
        - merged_hclim_cube: iris cube of rcm data.
    """
    rcm_files = [os.path.join(rcm_directory, f) for f in os.listdir(rcm_directory) if "tas_ANT" in f and f.endswith(".nc")]
    hclim_cube_list = iris.load(rcm_files)  # Or use a sorted list of filenames
    for cube in hclim_cube_list:
        for key in ['tracking_id', 'creation_date']:
            cube.attributes.pop(key, None)  # Remove if it exists
    
    # Merge into a single cube along time
    merged_hclim_cube = CubeList(hclim_cube_list).concatenate_cube()
    rcm_ease2 = interpolate_to_ease2sh.reproject_hclim(merged_hclim_cube).fillna(270)
    rcm_raw = rcm_ease2.sel(time=slice(*config.data_range))

    return rcm_raw

def preprocess_MetUM_data(metum_opened, metum_elev_opened, metum_lm_opened):
    """
    TO DO- add in processing steps for metum data
    """
    metum_raw, metum_elev_raw, metum_lm_raw = metum_opened 
    return metum_raw, metum_elev_raw, metum_lm_raw

def generate_single_task(dates, progress=False, desc="Generating tasks", **kwargs):
    train_tasks = []
    for date in tqdm(dates, disable=not progress):
            task = task_loader(date, context_sampling=config.context_sampling_strategy,
                            target_sampling="all")
            task.remove_context_nans().remove_target_nans()
            train_tasks.append(task)
    return train_tasks



def init_task_loader(context_set, rcm, elevation):
    global task_loader
    task_loader = TaskLoader(
        context=context_set * config.context_set_day_count,
        target=rcm,
        context_delta_t=config.delta_t,
        target_delta_t=0,
        aux_at_targets=elevation
    )


# 1. Worker function
def load_and_clean_no_pickle(date, context_set, rcm, elevation):
    pid = os.getpid()
    print(f"[Worker {pid}] Loading date: {date}")
    t0 = time.time()

    try:
        # Each process creates its own TaskLoader instance
        task_loader = TaskLoader(
            context=context_set * config.context_set_day_count,
            target=rcm,
            context_delta_t=config.delta_t,
            target_delta_t=0,
            aux_at_targets=elevation
        )

        task = task_loader(
            date,
            context_sampling=config.context_sampling_strategy,
            target_sampling="all"
        )
        task.remove_context_nans().remove_target_nans()

        print(f"[Worker {pid}] Done {date} in {time.time() - t0:.2f}s")
        return date, task

    except Exception as e:
        print(f"[Worker {pid}] Exception on date {date}: {e}")
        traceback.print_exc()
        raise e  # Surface the error


# 2. Parallel task generator
def generate_tasks_parallel_no_pickle(dates, context_set, rcm, elevation, progress=False, max_workers=None):
    print(f"[Main PID {os.getpid()}] Launching parallel pool with {max_workers or os.cpu_count()} workers...")

    # Partial func to bind shared args to each worker
    taskloader_func = partial(load_and_clean, context_set=context_set, rcm=rcm, elevation=elevation)

    tasks_by_date = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(taskloader_func, date): date for date in dates}
        for future in tqdm(as_completed(futures), total=len(dates), disable=not progress, desc="Generating tasks"):
            date, task = future.result()
            tasks_by_date[date] = task

    # Return tasks in the same order as input dates
    return [tasks_by_date[date] for date in dates]


def load_and_clean(date, context_set, rcm, elevation):
    import traceback
    pid = os.getpid()
    print(f"[Worker {pid}] Loading date: {date}")
    t0 = time.time()

    try:
        task_loader = TaskLoader(
            context=context_set * config.context_set_day_count,
            target=rcm,
            context_delta_t=config.delta_t,
            target_delta_t=0,
            aux_at_targets=elevation
        )

        task = task_loader(date, context_sampling=config.context_sampling_strategy, target_sampling="all")
        task.remove_context_nans().remove_target_nans()
        print(f"[Worker {pid}] Done {date} in {time.time() - t0:.2f}s")
        return date, task

    except Exception as e:
        print(f"[Worker {pid}] Exception on date {date}: {e}")
        traceback.print_exc()
        raise e  # Re-raise so it fails visibly


def generate_tasks_parallel(dates, context_set, rcm, elevation, progress=False, max_workers=None, cache_path=None):

    print(f"[Main PID {os.getpid()}] Launching parallel pool with {max_workers} workers...")

    if cache_path and os.path.exists(cache_path):
        print(f"Loading tasks from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Bind all shared args except date so the taskloader can efficiently be passed to all workers.
    taskloader_func = partial(load_and_clean, context_set=context_set, rcm=rcm, elevation=elevation)

    tasks = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(taskloader_func, date): date for date in dates}
        for future in tqdm(as_completed(futures), total=len(dates), disable=not progress):
            date, task = future.result()
            tasks[date] = task

    ordered_tasks = [tasks[date] for date in dates]

    if cache_path:
        print(f"Saving tasks to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(ordered_tasks, f)

    return ordered_tasks



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

def train_model_mixed_precision(data_processor, task_loader, train_range, date_subsample_factor, 
                                batch_size, all_dates=None, all_tasks=None,
                               context_set=None, rcm=None, elevation=None):

    set_gpu_default_device()
    
    model = ConvNP(data_processor, task_loader)
    opt = optim.Adam(model.model.parameters(), lr=config.l_rate)
    #grad_scaler = GradScaler()  # Initialize GradScaler
    
    val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]
    val_tasks = generate_tasks_parallel_no_pickle(val_dates, context_set, rcm, elevation, progress=False)
    #val_tasks = generate_tasks(val_dates, progress=False)
    
    loss_records = []
    trainer = Trainer(model, lr=config.l_rate)
    
    if all_dates is not None and all_tasks is not None:
        # Mapping from date to task for fast lookup
        date_to_task = {date: task for date, task in zip(all_dates, all_tasks)}
        # Training loop
        total_days = len(all_dates)
        n_train_dates = total_days // date_subsample_factor
    
    for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
        
        if all_dates is not None and all_tasks is not None:
            #Generate tasks as subset from cached tasks
            train_dates = np.random.choice(all_dates, size=n_train_dates, replace=False)        
            train_tasks = [date_to_task[pd.Timestamp(date)] for date in train_dates]

        elif all_dates is None and all_tasks is None:
            # Generate tasks from original generate_task() mnethod. 
            train_dates = pd.date_range(train_range[0], train_range[1])[::date_subsample_factor]
            train_tasks = generate_tasks_parallel_no_pickle(train_dates, context_set, rcm, elevation, progress=False)
    
        # Call the trainer with the tasks and mixed precision
        batch_losses = trainer(train_tasks)#, scaler=grad_scaler)
        
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



def main():
    t10 = time.time()
    gcm_var_datasets = process_gcm_mpi_data(gcm_directory, selected_vars, p_levels)
    print(f"processed gcm{time.time() - t10:.2f}s")
    if config.model_type == "perfect":
        print("TODO")
    elif config.model_type == "imperfect":
        rcm_raw = preprocess_hclim_rcm_data(rcm_directory)
        print(f"processed hclim{time.time() - t10:.2f}s")
        elev_ease2 = interpolate_to_ease2sh.reproject_hclim(elevation_iris).fillna(0)
        lm_ease2 = interpolate_to_ease2sh.reproject_hclim(land_mask_iris).fillna(0)

        input_data = [rcm_raw, elev_ease2, lm_ease2]
        for var in selected_vars:
            input_data.extend(gcm_var_datasets[var].sel(plev=p_level) for p_level in p_levels)

        print(f"generate gcm tasks{time.time() - t10:.2f}s")
        data_processor = DataProcessor(x1_name="projection_y_coordinate", x2_name="projection_x_coordinate")
        print(f"dataloader created{time.time() - t10:.2f}s")
        outputs = data_processor(input_data)
        print(f"dataloader called{time.time() - t10:.2f}s")
        rcm = outputs[0]
        elevation = outputs[1]
        land_mask = outputs[2]
        gcm = outputs[3:]

        context_set = list(gcm) + [elevation, land_mask]


    generate_tasks_at_once = False

    if generate_tasks_at_once == True:
        all_dates = pd.date_range(train_range[0], train_range[1])
        print(f'generating all tasks{time.time() - t10:.2f}s')
        # Now call generate_tasks_parallel() which uses task_loader
        all_tasks = generate_tasks_parallel(all_dates, context_set, rcm, elevation, progress=False, max_workers=workers, cache_path="cache/all_train_tasks198.pkl")
        
        print(f'tasks now generated{time.time() - t10:.2f}s')
    
        trained_model = train_model_mixed_precision(
            data_processor, task_loader,
            train_range, date_subsample_factor,
            batch_size, all_dates, all_tasks
        )
    elif generate_tasks_at_once == False:
        init_task_loader(context_set, rcm, elevation)
        trained_model = train_model_mixed_precision(
                    data_processor, task_loader,
                    train_range, date_subsample_factor,
                    batch_size, 
                    context_set=context_set, rcm=rcm, elevation=elevation)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()