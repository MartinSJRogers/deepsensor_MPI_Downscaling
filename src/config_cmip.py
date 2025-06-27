from pathlib import Path
import os


# Date range of data
"""
# Consider using this smaller train/val range in the first instance to check it works
train_range = ("1985-01-06T12:00:00.000000000", "1985-01-25T12:00:00.000000000")
val_range = ("1985-02-06T12:00:00.000000000", "1985-02-25T12:00:00.000000000")
"""
# train_range = ("1985-01-06T12:00:00.000000000", "1985-01-15T12:00:00.000000000")
# val_range = ("1985-02-06T12:00:00.000000000", "1985-02-15T12:00:00.000000000")

train_range = ("1985-01-06T12:00:00.000000000", "2011-02-25T12:00:00.000000000")
val_range = ("2012-02-06T12:00:00.000000000", "2014-08-25T12:00:00.000000000")


### Training paramaters used irrespective of experiment ####
## Training parameters ##
num_epochs = 500
### Change to 1e-5 and 5e-6 ###
l_rate = 1e-5
batch_size = 5
date_subsample_factor = 5

##### Manually defined variables #####
# options : ["MetUM_perfect", "MetUM_imperfect", "HCLIM_perfect", "HCLIM_imperfect"].  
run_type = "HCLIM_perfect"
### Change to something you can recognise when training e.g. contain l_rate value###
 # To identify output weight files
run_identifier_fn = "decades_training_stack_"+str(l_rate)+"_"+str(num_epochs)+"_"+str(batch_size)+"_N2_C16_G8"
print (run_identifier_fn)

##### End of manually defined variables #####

rcm_model = run_type.split('_')[0] # Returns HCLIM or MetUM
model_type = run_type.split('_')[1] # Returns perfect or imperfect

### Check these directories are correct and that you have created a weight and loss directory. 
if rcm_model == "HCLIM":
    BASE_DIR = "/gws/nopw/j04/bas_palaeoclim/surfeit/HCLIM/MPI-ESM1-2-LR/hist/"
    ELEV_FN = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/orog_clim_ANT11_ANT11_eval_ERA5_fx.nc"
    LAND_MASK_FN = "/gws/nopw/j04/bas_palaeoclim/surfeit/ds_runs/data_input/auxillary_files/lsm_clim_ANT11_ANT11_eval_ERA5_fx.nc"
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
context_var_count = 3 # number of context variables e.g. elevation and temperature

# Number of days in context set e.g. [-1, 0, 1] = 3.
context_set_day_count = (days_either_side * 2) + 1 
# Explicitly writing out delta_t argument, e.g. [-2,-1,0,1,2]
delta_t = [i for i in range (-days_either_side, days_either_side + 1) for day in range(context_var_count)]

# Calculate number of entries of 'all' to contain in context sampling argument e.g. ['all', 'all', 'all'].
context_sample_entries = (days_either_side * context_var_count * 2) + context_var_count
value = 'all'
context_sampling_strategy = []
for i in range(context_sample_entries):
    context_sampling_strategy.append(value)


