"""
Module storing utilities for converting between coordinate systems. 
"""
import numpy as np
from sarpy.geometry.geocoords import (
    enu_to_ecf, 
    geodetic_to_ecf,
    ecf_to_enu,  
)

def build_enu_image_grid(
        x_size: int, 
        y_size: int, 
        spacing: float, 
        reference_point: np.ndarray, 
        target = None
    ):
    
    """
    Build an ENU image grid.

    Parameters
    ----------
    x_size : int
        Number of pixels in the x direction.
    
    y_size : int
        Number of pixels in the y direction.
    
    reference_point: array 
        reference point in ECF format

    target_point: array 
        target point to center the grid on.
    
    spacing : float
        Spacing between pixels.

    Returns
    -------
    np.ndarray
        A 3D array of shape (x_size, y_size, 3) where each pixel contains [East, North, Up] coordinates in meters.
    """

    center_x = 0 
    center_y = 0 

    if target is not None: 
        targ_enu = ecf_to_enu(geodetic_to_ecf(target), reference_point)
        center_x = targ_enu[0]
        center_y = targ_enu[1]
        print(f'centering enu grid on {targ_enu}')


    x_width_m = x_size * spacing
    y_height_m = y_size * spacing
    x_axis = np.linspace((-x_width_m / 2) + center_x, (x_width_m / 2) + center_x, x_size)
    y_axis = np.linspace((-y_height_m / 2) + center_y, (y_height_m / 2) + center_y, y_size)
    east_grid, north_grid = np.meshgrid(x_axis, y_axis)
    east_grid = east_grid.flatten()
    north_grid = north_grid.flatten()

    up_grid = np.zeros(east_grid.shape)
    enu_grid = np.column_stack((east_grid, north_grid, up_grid))
    return enu_grid

 
def convert_enu_grid_to_ecef(enu_grid, origin_ecef):
    """
    Convert an ENU grid to ECEF coordinates.
    
    Parameters
    ----------
    enu_grid : np.ndarray
        Array of shape (N, 3) containing [East, North, Up] coordinates
    origin_ecef : list or np.ndarray
        Origin point in ecef format
    
    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing ECEF coordinates
    """
    ecef_grid = np.zeros_like(enu_grid)
    
    for i in range(len(enu_grid)):
        enu_point = enu_grid[i]
        ecef_grid[i] = enu_to_ecf(enu_point, origin_ecef)
    
    return ecef_grid