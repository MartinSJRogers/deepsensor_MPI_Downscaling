#!/usr/bin/env python

from load_cmip6_variables import load_cmip6_variable

cmip6_ds = load_cmip6_variable(
    cmip6_root="/home/users/sg240/arcv_cmip6",
    variable_name="ua",
    start_year=2010,
    end_year=2011,
    pressure_level=50000  # 500 hPa
)

print(cmip6_ds)
