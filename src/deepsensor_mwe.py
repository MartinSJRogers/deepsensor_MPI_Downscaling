# @martin rogers marrog@bas.ac.uk
from deepsensor.data import DataProcessor 
from deepsensor.data.sources import get_era5_reanalysis_data
from deepsensor.data import TaskLoader
from deepsensor.train import Trainer, set_gpu_default_device
from deepsensor.model import ConvNP
import deepsensor.torch
import pandas as pd
from tqdm import tqdm

data_range=("2010-06-25", "2010-06-30") 
bounding_box_antarctica=(-180, 180, -90, -60) 
extent=bounding_box_antarctica
era5_var_IDs=["2m_temperature"]
cache_dir=".datacache"

print('Retrieving ERA5 data...')
era5_raw_ds=get_era5_reanalysis_data(era5_var_IDs, extent=bounding_box_antarctica, date_range=data_range, cache=False, cache_dir=cache_dir)
print('ERA5 data retrieved')

data_processor=DataProcessor(x1_name="lat", x2_name="lon")

era5_ds =data_processor(era5_raw_ds)

task_loader = TaskLoader(
    context=[era5_ds]*2,
    target=[era5_ds],
    context_delta_t=[-1, 0],
    target_delta_t=1,
)
set_gpu_default_device()
model = ConvNP(data_processor, task_loader)

train_tasks = []
for date in pd.date_range("2010-06-27", "2010-06-28"):
    task = task_loader(date, context_sampling="all", target_sampling = "all")
    train_tasks.append(task)

# Train model
trainer = Trainer(model, lr=5e-5)

for epoch in tqdm(range(2)):
    batch_losses = trainer(train_tasks)

test_task = task_loader("2010-06-29", context_sampling=["all", "all"], target_sampling="all")
predictions = model.predict(test_task, X_t=era5_raw_ds)

