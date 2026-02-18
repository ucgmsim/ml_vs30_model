import logging
from pathlib import Path
from typing import Callable

import einops
import numpy as np
from pyproj import Geod

import rasterio

from .. import constants
from .. import utils

logger = logging.getLogger(__name__)

class TIFLoader:
    """Class for retrieving data from downloaded TIFF files."""

    SUPPORTED_VARIABLES = {
        constants.InputVariable.LandformShannonIndex,
        constants.InputVariable.LandformEntropy,
        constants.InputVariable.LandformUniformity,
        constants.InputVariable.AbsoluteDepthToBedrock,
    }

    VAR_TO_FILENAME_MAP = {
        constants.InputVariable.LandformShannonIndex: "geom_1KMsha_GMTEDmd.tif",
        constants.InputVariable.LandformEntropy: "geom_1KMent_GMTEDmd.tif",
        constants.InputVariable.LandformUniformity: "geom_1KMuni_GMTEDmd.tif",
        constants.InputVariable.AbsoluteDepthToBedrock: "absolute_depth_to_bedrock/BDTICM_M_250m_ll.tif",
    }

    def __init__(
        self, base_raw_data_dir: Path = constants.BASE_DATA_DIR / "input_data" / "raw"
    ) -> None:
        self.base_raw_data_dir = base_raw_data_dir

    def get_values(self, coords: np.ndarray, variable: constants.InputVariable):
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by TIFLoader.",
                logger,
            )

        tif_ffp = self.base_raw_data_dir / self.VAR_TO_FILENAME_MAP[variable]
        if not tif_ffp.exists():
            utils.raise_log(
                FileNotFoundError,
                f"TIF file for variable {variable} not found at {tif_ffp}.",
                logger,
            )

        with rasterio.open(tif_ffp) as dataset:
            assert (
                dataset.crs.to_epsg() == constants.WGS84_EPSG
            ), "Dataset CRS is not WGS84."

            values = np.concatenate(list(dataset.sample(coords)))

        # Deal with negative absolute depth to bedrock values
        if variable == constants.InputVariable.AbsoluteDepthToBedrock:
            mask = values < 0
            if np.any(mask):
                logger.info(
                    f"Found {np.sum(mask)} negative values for Absolute Depth to Bedrock. "
                    f"Using nearest non-zero value."
                )

                values[mask] = find_nearest_valid(
                    tif_ffp, coords[mask], lambda v: v >= 0, values.dtype
                )

        return values

def find_nearest_valid(
    tif_ffp: Path,
    invalid_coords: np.ndarray,
    valid_check: Callable[[np.ndarray], np.ndarray],
    dtype: np.dtype,
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
    geod = Geod(ellps="WGS84")
    updated_values = np.empty(invalid_coords.shape[0], dtype=dtype)
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

        # Mask out negative values
        valid_mask = valid_check(meshgrid_values)
        if np.any(valid_mask):
            nearest_idx = np.argmin(meshgrid_dist[valid_mask])
            logger.debug(
                f"Replacing value at coords {invalid_coords[i]} with nearest valid value "
                f"at distance {meshgrid_dist[valid_mask][nearest_idx]:.2f} m."
            )
            updated_values[i] = meshgrid_values[valid_mask][nearest_idx]

    return updated_values
