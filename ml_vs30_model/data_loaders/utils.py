from pathlib import Path
from typing import Callable
import logging

import einops
import rasterio
import numpy as np
from pyproj import Geod

from .. import utils
from .. import constants

logger = logging.getLogger(__name__)

def convert_dtype_and_handle_nodata(values: np.ndarray, no_data_value: float | int | None, variable: constants.InputVariable) -> np.ndarray:
    """
    Converts the values to appropriate dtype 
    and sets no-data values to standard constants.
    """
    try:
        if np.issubdtype(values.dtype, np.integer):
            if no_data_value is not None and int(no_data_value) != constants.INTEGER_NO_DATA_VALUE:
                values[values == no_data_value] = constants.INTEGER_NO_DATA_VALUE
            values = utils.safe_cast(values, np.int32)
        elif np.issubdtype(values.dtype, np.floating):
            if no_data_value is not None and not np.isnan(no_data_value):
                values[values == no_data_value] = np.nan
            values = utils.safe_cast(values, np.float32)
    except OverflowError as e:
        utils.raise_log(
            OverflowError,
            f"Values for variable {variable} contain values out of bounds for target dtype: {values.dtype}",
            logger,
        )

    return values

def find_nearest_valid_wgs84(
    tif_ffp: Path,
    invalid_coords: np.ndarray,
    valid_check: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray
) -> np.ndarray:
    """
    Find nearest valid values for given coordinates in a TIFF file.

    Parameters
    ----------
    tif_ffp : Path
        File path to the TIFF file.
    invalid_coords : np.ndarray
        Array of shape (N, 2) containing coordinates (lon, lat) with invalid values.
    valid_check : Callable[[np.ndarray], np.ndarray]
        Function that takes an array of values and returns a boolean array indicating valid values.
    dtype : np.dtype
        Data type of the values to be returned.

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing the nearest valid values for the given coordinates.
    """
    assert values.shape[0] == invalid_coords.shape[0], "Values and coordinates must have the same length."

    geod = Geod(ellps="WGS84")
    updated_values = np.empty(invalid_coords.shape[0], dtype=values.dtype)
    for i in range(invalid_coords.shape[0]):
        # Create small meshgrid around the point
        lon, lat = invalid_coords[i]
        lat_grid = np.linspace(lat - (0.0001 * 50 / 2), lat + (0.0001 * 50 / 2), 50)
        lon_grid = np.linspace(lon - (0.0001 * 50 / 2), lon + (0.0001 * 50 / 2), 50)
        meshgrid = np.array(np.meshgrid(lon_grid, lat_grid))

        meshgrid_coords = einops.rearrange(meshgrid, "d h w -> (h w) d", d=2)
        meshgrid_dist = geod.inv(
            np.full(meshgrid_coords.shape[0], lon),
            np.full(meshgrid_coords.shape[0], lat),
            meshgrid_coords[:, 0],
            meshgrid_coords[:, 1],
        )[2]
        with rasterio.open(tif_ffp) as dataset:
            meshgrid_values = np.concatenate(list(dataset.sample(meshgrid_coords)))

        # Mask out fill values
        valid_mask = valid_check(meshgrid_values)
        if np.any(valid_mask):
            nearest_idx = np.argmin(meshgrid_dist[valid_mask])
            assert meshgrid_dist[valid_mask][nearest_idx] < 500, (
                f"Nearest valid value for coordinate {invalid_coords[i]} is too far away."
            )
            logger.debug(
                f"Replacing value at coords {invalid_coords[i]} with nearest valid value "
                f"at distance {meshgrid_dist[valid_mask][nearest_idx]:.2f} m."
            )
            updated_values[i] = meshgrid_values[valid_mask][nearest_idx]
        else:
            logger.warning(
                f"No valid values found within search radius for coordinate {invalid_coords[i]}. "
                f"Setting to no-data value."
            )
            updated_values[i] = values[i]

    return updated_values
