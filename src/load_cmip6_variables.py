import xarray as xr
import os
import re

def load_cmip6_variable(cmip6_root, variable_name, start_year, end_year, pressure_level):
    """
    Load a CMIP6 4D variable from the archive, filtered by time range and pressure level.

    Args:
        cmip6_root (str): Root path to the CMIP6 archive (e.g., /home/users/sg240/arcv_cmip6)
        variable_name (str): Name of the variable (e.g., 'ua', 'va', 'ta', 'hus')
        start_year (int): Start year of interest
        end_year (int): End year of interest
        pressure_level (int): Pressure level in Pa (e.g., 50000 for 500 hPa)

    Returns:
        xarray.Dataset: Dataset filtered by pressure level and time
    """

    # Path to the folder where files for this variable should be found
    if variable_name == "ps":
        path = os.path.join(
            cmip6_root,
            "CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/CFday",
            variable_name,
            "gn/latest"
        )
        
    else:
        path = os.path.join(
            cmip6_root,
            "CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/day",
            variable_name,
            "gn/latest"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    # Find file(s) for the variable
    if variable_name == "ps":
        pattern = re.compile(rf"{variable_name}_CFday_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_(\d{{8}})-(\d{{8}})\.nc")
    else:
        pattern = re.compile(rf"{variable_name}_day_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_(\d{{8}})-(\d{{8}})\.nc")
    selected_files = []

    for fname in os.listdir(path):
        match = pattern.match(fname)
        if match:
            file_start = int(match.group(1)[:4])
            file_end = int(match.group(2)[:4])
            if file_end >= start_year and file_start <= end_year:
                selected_files.append(os.path.join(path, fname))

    if not selected_files:
        raise FileNotFoundError(f"No files found for variable {variable_name} between {start_year} and {end_year}")

    # Load files into a single dataset
    ds = xr.open_mfdataset(selected_files, combine='by_coords')

    # Filter time
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Filter pressure level (assumes variable is 4D: time, lev, lat, lon)
    if 'plev' in ds.dims:
        closest_lev = ds['plev'].sel(plev=pressure_level, method='nearest')
        ds = ds.sel(plev=closest_lev)

    return ds
