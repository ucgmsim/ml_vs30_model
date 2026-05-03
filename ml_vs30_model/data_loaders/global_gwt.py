import logging
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from .. import constants
from .. import utils
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)

class GlobalGWT(BaseLoader):
    """
    Class for accessing global groundwater table data.
    Dataset source: https://www.science.org/doi/10.1126/science.1229881
    """

    SUPPORTED_VARIABLES = [
        constants.InputVariable.DepthToGroundwater,
    ]

    DB_COORD_LIMITS = {
        "AFRICA": {
            "lat_min": -34.99583435058594,
            "lat_max": 37.995540618896484,
            "lon_min": -18.995832443237305,
            "lon_max": 54.995540618896484,
        },
        "EURASIA": {
            "lat_min": 0.004166666883975267,
            "lat_max": 82.9955062866211,
            "lon_min": -13.995833396911621,
            "lon_max": 179.99505615234375,
        },
        "NAMERICA": {
            "lat_min": 5.004166603088379,
            "lat_max": 83.99552154541016,
            "lon_min": -179.99583435058594,
            "lon_max": -52.004676818847656,
        },
        "SAMERICA": {
            "lat_min": -55.99583435058594,
            "lat_max": 14.995550155639648,
            "lon_min": -92.99583435058594,
            "lon_max": -32.00440979003906,
        },
        "OCEANIA": {
            "lat_min": -47.662498474121094,
            "lat_max": 7.4956159591674805,
            "lon_min": 95.00416564941406,
            "lon_max": 179.99549865722656,  
        }
    }
    DB_COORD_LIMITS_DF = pd.DataFrame(DB_COORD_LIMITS).T

    def __init__(
        self, base_raw_data_dir: Path = constants.BASE_DATA_DIR / "input_data" / "raw"
    ):
        self.base_raw_data_dir = base_raw_data_dir

    def get_values(
        self, coords: np.ndarray, variable: constants.InputVariable
    ) -> np.ndarray:
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by GeoMorpho90.",
                logger,
            )

        lon_mask = coords[:, 0][:, None] >= self.DB_COORD_LIMITS_DF["lon_min"].values[None, :]
        lon_mask &= coords[:, 0][:, None] <= self.DB_COORD_LIMITS_DF["lon_max"].values[None, :]
        lat_mask = coords[:, 1][:, None] >= self.DB_COORD_LIMITS_DF["lat_min"].values[None, :]
        lat_mask &= coords[:, 1][:, None] <= self.DB_COORD_LIMITS_DF["lat_max"].values[None, :]
        mask = lon_mask & lat_mask
        # Ensure there is exactly one match per coordinate
        assert np.all(mask.sum(axis=1) == 1)

        # Get the region for each coordinate
        region_ind = mask.argmax(axis=1)
        regions = self.DB_COORD_LIMITS_DF.index[region_ind].values.astype(str)
        unqiue_region = np.unique(regions)

        values = np.full(coords.shape[0], np.nan)
        for region in unqiue_region:
            coords_mask = regions == region

            # Load data and interpolate
            data_ffp = self.base_raw_data_dir / f"depth_to_groundwater/{region}_WTD_annualmean.nc" 
            da = xr.open_dataset(data_ffp).sel(time=1).WTD

            interp = RegularGridInterpolator((da.lat.values, da.lon.values), da.values)
            values[coords_mask] = interp(coords[coords_mask][:, ::-1])  

        return values


