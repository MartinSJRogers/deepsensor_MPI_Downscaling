import iris
import numpy as np
import xarray as xr
import pandas as pd

def create_empty_ease2_grid(x_bounds, y_bounds, x_grid_spacing, y_grid_spacing, epsg6932):
    """
    Create empty iris cube with defined coordinate reference system, bounds and resolution.
    """
    x_points = np.arange(x_bounds[0]+x_grid_spacing/2, x_bounds[1], x_grid_spacing)
    y_points = np.arange(y_bounds[0]+y_grid_spacing/2, y_bounds[1], y_grid_spacing)

    xc = iris.coords.DimCoord(x_points, standard_name='projection_x_coordinate', units='m', coord_system=epsg6932)
    yc = iris.coords.DimCoord(y_points, standard_name='projection_y_coordinate', units='m', coord_system=epsg6932)
    for coord in [xc, yc]:
        coord.guess_bounds()

    ease2sh_template = iris.cube.Cube(np.empty((len(yc.points), len(xc.points))),
                                          dim_coords_and_dims=[(yc, 0), (xc, 1)])
    return ease2sh_template    


def reformat_to_xarray(iris_datacube):
    """
    Convert from iris data cube to xarray. 
    Convert time coordinates so that there are no errors in DeepSensor's dataloader and Taskloader. 
    """
    xr_ds = xr.DataArray.from_iris(iris_datacube)

    # Convert the time coordinate format to YYYY-MM-DD. Normalize adds the time 00:00:00
    xr_ds = xr_ds.assign_coords(time=pd.to_datetime(xr_ds.time.values).normalize())
    
    return xr_ds


    
def reproject_era5(era5):
    """
    Reproject ERA5 and MetUM to EASE2 grid and subset using small x and y bounds so there are no NaNs in the MetUM data.
    Ensure the names of lon_coord and lat_coord match with the original ERA5 coordinate names. 
    """
    lon_coord = era5.coord(var_name='longitude')
    for attr, value in {'units': 'degrees_east', 'standard_name': 'longitude'}.items():
        setattr(lon_coord, attr, value)
    lat_coord = era5.coord(var_name='latitude')
    for attr, value in {'units': 'degrees', 'standard_name': 'latitude'}.items():
        setattr(lat_coord, attr, value)
    cs = iris.coord_systems.GeogCS(6367470.0)
    for axis in ['x', 'y']:
        era5.coord(axis=axis, dim_coords=True).coord_system = cs
        
    # create an EPSG:6932 coordinate system for the EASE 2.0 grid
    # see: https://epsg.io/6932
    wgs84 = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563)
    epsg6932 = iris.coord_systems.LambertAzimuthalEqualArea(latitude_of_projection_origin=-90.0,
                                                            longitude_of_projection_origin=0.0,
                                                            ellipsoid=wgs84)

    # Small bonds
    x_bounds_small = [-2.6e6, 2.6e6]
    y_bounds_small = [2.55e6, -1.85e6]

    # Grid spacing ERA5
    x_grid_spacing_era5 = 25000.0
    y_grid_spacing_era5 = -25000.0


    ease2sh_template_era5 = create_empty_ease2_grid(x_bounds_small, y_bounds_small, x_grid_spacing_era5, y_grid_spacing_era5, epsg6932)
    # regrid the ERA5 data onto the EASE 2.0 SH grid
    era5_on_ease2sh_grid = era5.regrid(ease2sh_template_era5, iris.analysis.Linear(extrapolation_mode='mask'))
    
    era5_xr = reformat_to_xarray(era5_on_ease2sh_grid)
    
    return era5_xr


def reproject_mpi(era5):
    """
    Reproject ERA5 and MetUM to EASE2 grid and subset using small x and y bounds so there are no NaNs in the MetUM data.
    Ensure lon_coord and lat_Coord are the correct names of the coordinates for the original ERA5 data
    """
    lon_coord = era5.coord(var_name='lon')
    for attr, value in {'units': 'degrees_east', 'standard_name': 'longitude'}.items():
        setattr(lon_coord, attr, value)
    lat_coord = era5.coord(var_name='lat')
    for attr, value in {'units': 'degrees', 'standard_name': 'latitude'}.items():
        setattr(lat_coord, attr, value)
    cs = iris.coord_systems.GeogCS(6367470.0)
    for axis in ['x', 'y']:
        era5.coord(axis=axis, dim_coords=True).coord_system = cs
        
    # create an EPSG:6932 coordinate system for the EASE 2.0 grid
    # see: https://epsg.io/6932
    wgs84 = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563)
    epsg6932 = iris.coord_systems.LambertAzimuthalEqualArea(latitude_of_projection_origin=-90.0,
                                                            longitude_of_projection_origin=0.0,
                                                            ellipsoid=wgs84)

    # Small bonds
    x_bounds_small = [-2.6e6, 2.6e6]
    y_bounds_small = [2.55e6, -2.55e6]

    # Grid spacing MPI - using values from Van de Meer paper.
    x_grid_spacing_era5 = 68000.0
    y_grid_spacing_era5 = -208000.0


    ease2sh_template_era5 = create_empty_ease2_grid(x_bounds_small, y_bounds_small, x_grid_spacing_era5, y_grid_spacing_era5, epsg6932)
    # regrid the ERA5 data onto the EASE 2.0 SH grid
    era5_on_ease2sh_grid = era5.regrid(ease2sh_template_era5, iris.analysis.Linear(extrapolation_mode='mask'))
    
    era5_xr = reformat_to_xarray(era5_on_ease2sh_grid)
    
    return era5_xr

def reproject_hclim(metum):
    """
    Reproject single array e.g. for elevation
    """
    cs = iris.coord_systems.GeogCS(6367470.0)


    # create an EPSG:6932 coordinate system for the EASE 2.0 grid
    # see: https://epsg.io/6932
    wgs84 = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563)
    epsg6932 = iris.coord_systems.LambertAzimuthalEqualArea(latitude_of_projection_origin=-90.0,
                                                            longitude_of_projection_origin=0.0,
                                                            ellipsoid=wgs84)

    # Small bonds
    x_bounds_small = [-2.6e6, 2.6e6]
    y_bounds_small = [2.55e6, -2.55e6]

    # Grid spacing MetUM
    x_grid_spacing_metum = 11000.0
    y_grid_spacing_metum = -11000.0

    ease2sh_template_metum = create_empty_ease2_grid(x_bounds_small, y_bounds_small, x_grid_spacing_metum, y_grid_spacing_metum, epsg6932)
    # regrid the ERA5 data onto the EASE 2.0 SH grid
    metum_on_ease2sh_grid = metum.regrid(ease2sh_template_metum, iris.analysis.Linear(extrapolation_mode='mask'))

    metum_xr = reformat_to_xarray(metum_on_ease2sh_grid)
    
    return metum_xr

def reproject_metum(metum):
    """
    Reproject single array e.g. for elevation
    """
    cs = iris.coord_systems.GeogCS(6367470.0)


    # create an EPSG:6932 coordinate system for the EASE 2.0 grid
    # see: https://epsg.io/6932
    wgs84 = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563)
    epsg6932 = iris.coord_systems.LambertAzimuthalEqualArea(latitude_of_projection_origin=-90.0,
                                                            longitude_of_projection_origin=0.0,
                                                            ellipsoid=wgs84)

    # Small bonds
    #x_bounds_small = [-2.6e6, 2.6e6]
    #y_bounds_small = [2.55e6, -1.85e6]

    # trialing for south polar stereographic
    x_bounds_small = [-3e6, 3e6]
    y_bounds_small = [3e6, -3e6]

    # Grid spacing MetUM
    x_grid_spacing_metum = 11000.0
    y_grid_spacing_metum = -11000.0

    ease2sh_template_metum = create_empty_ease2_grid(x_bounds_small, y_bounds_small, x_grid_spacing_metum, y_grid_spacing_metum, epsg6932)
    # regrid the ERA5 data onto the EASE 2.0 SH grid
    metum_on_ease2sh_grid = metum.regrid(ease2sh_template_metum, iris.analysis.Linear(extrapolation_mode='mask'))

    metum_xr = reformat_to_xarray(metum_on_ease2sh_grid)
    
    return metum_xr