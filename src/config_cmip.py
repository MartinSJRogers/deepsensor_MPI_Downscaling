from pathlib import Path
<<<<<<< HEAD
import os
=======
import os, sys
import json
>>>>>>> 837bbe2 (Added the use of json file for some parameters in config_cmip.py. Changes done only to HCLIM_perfect_downscale.py file)


# Date range of data
"""
# Consider using this smaller train/val range in the first instance to check it works
train_range = ("1985-01-06T12:00:00.000000000", "1985-01-25T12:00:00.000000000")
val_range = ("1985-02-06T12:00:00.000000000", "1985-02-25T12:00:00.000000000")
"""
<<<<<<< HEAD
data_range = ("2009-01-01T00:00:00.000000000", "2012-01-31T00:00:00.000000000")
train_range = ("2009-01-06T00:00:00.000000000", "2012-01-21T00:00:00.000000000")
val_range = ("2012-01-21T00:00:00.000000000", "2012-01-23T00:00:00.000000000")
=======
data_range = ("2009-01-01T00:00:00.000000000", "2009-01-31T00:00:00.000000000")
train_range = ("2009-01-06T00:00:00.000000000", "2009-01-21T00:00:00.000000000")
val_range = ("2009-01-21T00:00:00.000000000", "2009-01-23T00:00:00.000000000")
>>>>>>> 837bbe2 (Added the use of json file for some parameters in config_cmip.py. Changes done only to HCLIM_perfect_downscale.py file)


### Training paramaters used irrespective of experiment ####
## Training parameters ##
<<<<<<< HEAD
num_epochs = 151
### Change to 1e-5 and 5e-6 ###
l_rate = 1e-5
batch_size = 2
date_subsample_factor = 5

##### Manually defined variables #####
# options : ["MetUM_perfect", "MetUM_imperfect", "HCLIM_perfect", "HCLIM_imperfect"].  
run_type = "HCLIM_imperfect"

# Imperfect pressure level and variable selection - possible options
selected_vars = ["ua", "va", "ta"] # More to come
p_levels = [100000.,  85000.,  70000., 50000.]

# Convert vars and levels to compact strings
vars_str = "_".join(selected_vars)
plevs_str = "_".join([str(int(p/1000)) for p in p_levels])  # e.g., 100000 -> "100"

# Generate run idenetifier to use in output weight files
run_identifier_fn = f"decades_training_stack_{vars_str}_{plevs_str}_{l_rate}_{num_epochs}_{batch_size}_N2_C16_G8"
print (run_identifier_fn)

##### End of manually defined variables #####

rcm_model = run_type.split('_')[0] # Returns HCLIM or MetUM
model_type = run_type.split('_')[1] # Returns perfect or imperfect

### Check these directories are correct and that you have created a weight and loss directory. 
if rcm_model == "HCLIM":
    BASE_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/HCLIM/MPI-ESM1-2-LR/hist/"
    ELEV_FN = "/home/users/marrog/palaeoclim/surfeit/MetUM/MetUM_PolarRES_Antarctic_11km_surface_altitude.nc"
    LAND_MASK_FN = "/home/users/marrog/palaeoclim/surfeit/MetUM/MetUM_PolarRES_Antarctic_11km_land_binary_mask.nc"
    #ELEV_FN = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/orog_clim_ANT11_ANT11_eval_ERA5_fx.nc"
    #LAND_MASK_FN = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/lsm_clim_ANT11_ANT11_eval_ERA5_fx.nc"
    GCM_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/CMIP6/MPI-ESM1-2-LR/masks"
elif rcm_model == "MetUM":
    BASE_DIR = '/data/hpcdata/users/marrog/DeepSensor_code/cmip_data' ## To update

WEIGHT_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_output/model_weights"
LOSS_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_output/model_losses"

# Create the folders if they don't exist
weight_dir_path = os.path.join(WEIGHT_DIR,run_type)
Path(weight_dir_path).mkdir(parents=True, exist_ok=True)
loss_dir_path = os.path.join(LOSS_DIR,run_type)
Path(loss_dir_path).mkdir(parents=True, exist_ok=True)

## Taskloader parameters ##
# How many days +/- time t=0 do is contained within context set
days_either_side = 5
# number of context variables. Plus 2 for elevation and temperature
context_var_count = (len(p_levels)*len(selected_vars)) + 2 


# Number of days in context set e.g. [-1, 0, 1] = 3.
context_set_day_count = (days_either_side * 2) + 1 
# Explicitly writing out delta_t argument, e.g. [-2,-2,-1-1,0,0,1,1,2,2]
delta_t = [i for i in range (-days_either_side, days_either_side + 1) for day in range(context_var_count)]

# Calculate number of entries of 'all' to contain in context sampling argument e.g. ['all', 'all', 'all'].
context_sample_entries = (days_either_side * context_var_count * 2) + context_var_count
value = 'all'
context_sampling_strategy = []
for i in range(context_sample_entries):
    context_sampling_strategy.append(value)
=======
num_epochs = 11
### Change to 1e-5 and 5e-6 ###
l_rate = 1e-5
batch_size = 5
date_subsample_factor = 5

##### Manually defined variables #####
# # options : ["MetUM_perfect", "MetUM_imperfect", "HCLIM_perfect", "HCLIM_imperfect"].  
# run_type = "MetUM_perfect"
# # run_type = "HCLIM_perfect"
# ### Change to something you can recognise when training e.g. contain l_rate value###
#  # To identify output weight files
# run_identifier_fn = "decades_training_stack_"+str(l_rate)+"_"+str(num_epochs)+"_"+str(batch_size)+"_N2_C16_G8"
# print (run_identifier_fn)

# # Imperfect pressure level and variable selection - possible options
# #selected_vars = ["ua", "va", "ta"]
# #p_levels = [100000.,  85000.,  70000.,  50000.]
selected_vars = ["ta"]
p_levels = [50000.]

##### End of manually defined variables #####

def set_variables (json_file):
    try:
        with open(json_file, 'r') as f:
            config_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON - {e}")
        sys.exit(1)

    

    # Extract values from JSON
    run_type = config_json.get("run_type")
    l_rate = config_json.get("l_rate")
    batch_size = config_json.get("batch_size")
    data_range = config_json.get("data_range")
    train_range = config_json.get("train_range")
    val_range = config_json.get("val_range")
    date_subsample_factor = config_json.get("date_subsample_factor")
    run_identifier_fn_prefix = config_json.get("run_identifier_fn_prefix")
    num_epochs = config_json.get("num_epochs")
    run_identifier_fn = run_identifier_fn_prefix+str(l_rate)+"_"+str(num_epochs)+"_"+str(batch_size)+"_N2_C16_G8"

    config_json["rcm_model"] = run_type.split('_')[0] # Returns HCLIM or MetUM
    config_json["model_type"] = run_type.split('_')[1] # Returns perfect or imperfect

    ### Check these directories are correct and that you have created a weight and loss directory. 
    if config_json["rcm_model"] == "HCLIM":
        config_json["BASE_DIR"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/HCLIM/MPI-ESM1-2-LR/hist/"
        config_json["ELEV_FN"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/orog_clim_ANT11_ANT11_eval_ERA5_fx.nc"
        config_json["LAND_MASK_FN"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/lsm_clim_ANT11_ANT11_eval_ERA5_fx.nc"
        config_json["GCM_DIR"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/CMIP6/MPI-ESM1-2-LR/masks"
        config_json["var_tas"] = "tas"
        config_json["y_coord"] = "y"
        config_json["x_coord"] = "x"
    elif config_json["rcm_model"] == "MetUM":
        # BASE_DIR = '/data/hpcdata/users/marrog/DeepSensor_code/cmip_data' ## To update
        config_json["BASE_DIR"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/MetUM/daily/"
        config_json["ELEV_FN"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/MetUM/MetUM_PolarRES_Antarctic_11km_surface_altitude.nc"
        config_json["LAND_MASK_FN"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/MetUM/MetUM_PolarRES_Antarctic_11km_land_binary_mask.nc"
        config_json["GCM_DIR"] = "/gws/nopw/j04/bas_palaeoclim/surfeit/CMIP6/MPI-ESM1-2-LR/masks"
        config_json["var_tas"] = "near_surface_air_temperature"
        config_json["y_coord"] = "grid_latitude"
        config_json["x_coord"] = "grid_longitude"

    WEIGHT_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_output/model_weights"
    LOSS_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_output/model_losses"

    # Create the folders if they don't exist
    config_json["weight_dir_path"] = os.path.join(WEIGHT_DIR,run_type)
    Path(config_json["weight_dir_path"]).mkdir(parents=True, exist_ok=True)
    config_json["loss_dir_path"] = os.path.join(LOSS_DIR,run_type)
    Path(config_json["loss_dir_path"]).mkdir(parents=True, exist_ok=True)

    ## Taskloader parameters ##
    # How many days +/- time t=0 do is contained within context set
    days_either_side = 5
    # number of context variables. Plus 2 for elevation and temperature
    context_var_count = (len(p_levels)*len(selected_vars)) + 2 


    # Number of days in context set e.g. [-1, 0, 1] = 3.
    context_set_day_count = (days_either_side * 2) + 1 
    # Explicitly writing out delta_t argument, e.g. [-2,-2,-1-1,0,0,1,1,2,2]
    delta_t = [i for i in range (-days_either_side, days_either_side + 1) for day in range(context_var_count)]

    # Calculate number of entries of 'all' to contain in context sampling argument e.g. ['all', 'all', 'all'].
    context_sample_entries = (days_either_side * context_var_count * 2) + context_var_count
    value = 'all'
    context_sampling_strategy = []
    for i in range(context_sample_entries):
        context_sampling_strategy.append(value)
    
    config_json["context_sampling_strategy"] = context_sampling_strategy

    return config_json
>>>>>>> 837bbe2 (Added the use of json file for some parameters in config_cmip.py. Changes done only to HCLIM_perfect_downscale.py file)


