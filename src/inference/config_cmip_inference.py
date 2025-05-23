##### Manually defined variables #####
# options : ["MetUM_perfect", "MetUM_imperfect", "HCLIM_perfect", "HCLIM_imperfect"].  
run_type = "HCLIM_perfect"
plot_fn_mae = 'hclim_temp_century_mae.png'
plot_fn_psd = 'hclim_temp_example_psd.png'
plot = True
#Choose any file name
MODEL_FN = "/home/users/marrog/palaeoclim/mrogers/model_weights/HCLIM_perfect/decade_training_stack_140.json"

# Date range of data
test_range = ("2030-01-06T12:00:00.000000000", "2030-01-20T12:00:00.000000000")

##### End of manually defined variables #####

rcm_model = run_type.split('_')[0] # Returns HCLIM or MetUM
model_type = run_type.split('_')[1] # Returns perfect or imperfect

if rcm_model == "HCLIM":
    BASE_DIR = "/home/users/marrog/palaeoclim/surfeit/HCLIM/MPI-ESM1-2-LR/ssp370/"
    ELEV_FN = "/home/users/marrog/palaeoclim/mrogers/auxillary_files/orog_clim_ANT11_ANT11_eval_ERA5_fx.nc"
    LAND_MASK_FN = "/home/users/marrog/palaeoclim/mrogers/auxillary_files/lsm_clim_ANT11_ANT11_eval_ERA5_fx.nc"
elif rcm_model == "MetUM":
    BASE_DIR = '/data/hpcdata/users/marrog/DeepSensor_code/cmip_data' ## To update

PLOT_DIR = "/home/users/marrog/palaeoclim/mrogers/evaluation_plots"

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


