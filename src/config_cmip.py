from pathlib import Path
import os


# Date range of data
"""
# Consider using this smaller train/val range in the first instance to check it works
train_range = ("1985-01-06T12:00:00.000000000", "1985-01-25T12:00:00.000000000")
val_range = ("1985-02-06T12:00:00.000000000", "1985-02-25T12:00:00.000000000")
"""
data_range = ("2009-01-01T00:00:00.000000000", "2010-12-31T00:00:00.000000000")
train_range = ("2010-01-06T00:00:00.000000000", "2010-06-21T00:00:00.000000000")
val_range = ("2010-06-21T00:00:00.000000000", "2010-12-23T00:00:00.000000000")

# Extract years for file names
start_year = data_range[0][:4]
end_year = data_range[1][:4]

### Training paramaters used irrespective of experiment ####
## Training parameters ##
num_epochs = 151
### Change to 1e-5 and 5e-6 ###
l_rate = 1e-5
batch_size = 1
date_subsample_factor = 5

##### Manually defined variables #####
# options : ["MetUM_perfect", "MetUM_imperfect", "HCLIM_perfect", "HCLIM_imperfect"].  
run_type = "HCLIM_imperfect"

# Imperfect pressure level and variable selection - possible options
selected_vars = ["ta"] # More to come "zg", , "hus"
p_levels = [50000.]# 100000.,  85000.,  70000., 50000.

# Convert vars and levels to compact strings
#Seperate 
vars_str = "_".join(selected_vars)
plevs_str = "-".join([str(int(p/1000)) for p in p_levels])  # e.g., 100000 -> "100"

# Generate run idenetifier to use in output weight files
run_identifier_fn = f"decades_training_stack_unmasked__{vars_str}__{plevs_str}__{l_rate}__{num_epochs}__{batch_size}_Y{start_year}{end_year}"
print (run_identifier_fn)

##### End of manually defined variables #####

rcm_model = run_type.split('_')[0] # Returns HCLIM or MetUM
model_type = run_type.split('_')[1] # Returns perfect or imperfect

### Check these directories are correct and that you have created a weight and loss directory. 
if rcm_model == "HCLIM":
    BASE_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/HCLIM/MPI-ESM1-2-LR/hist/" # 1984 - 2014
    ELEV_FN = "/home/users/marrog/palaeoclim/surfeit/MetUM/MetUM_PolarRES_Antarctic_11km_surface_altitude.nc"
    LAND_MASK_FN = "/home/users/marrog/palaeoclim/surfeit/MetUM/MetUM_PolarRES_Antarctic_11km_land_binary_mask.nc"
    #ELEV_FN = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/orog_clim_ANT11_ANT11_eval_ERA5_fx.nc"
    #LAND_MASK_FN = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/lsm_clim_ANT11_ANT11_eval_ERA5_fx.nc"
    GCM_DIR = "/home/users/marrog/palaeoclim/mrogers/historical_unmasked_mpi" # 19790 - 2014
    #GCM_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/CMIP6/MPI-ESM1-2-LR/masks"
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


