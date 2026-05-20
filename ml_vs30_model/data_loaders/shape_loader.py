import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from .. import constants
from .. import utils
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ShapeLoader(BaseLoader):
    """Class for retrieving data from downloaded shapefiles."""

    SUPPORTED_VARIABLES = {
        constants.InputVariable.NZGeologyCategory,
        constants.InputVariable.NZGeologyAgeMin,
        constants.InputVariable.NZGeologyAgeMax,
        constants.InputVariable.NZLithologyCategory,
        constants.InputVariable.NZGeologicalUnit,
    }

    VAR_TO_FILENAME_MAP = {
        constants.InputVariable.NZGeologyCategory: "foster_geological_category/qmap.shp",
        constants.InputVariable.NZGeologyAgeMin: "nz_geology/ShapeFiles/NZL_GNS_250K_geological_units.shp",
        constants.InputVariable.NZGeologyAgeMax: "nz_geology/ShapeFiles/NZL_GNS_250K_geological_units.shp",
        constants.InputVariable.NZLithologyCategory: "nz_geology/ShapeFiles/NZL_GNS_250K_geological_units.shp",
        constants.InputVariable.NZGeologicalUnit: "nz_geology/ShapeFiles/NZL_GNS_250K_geological_units.shp",
    }

    def __init__(
        self, base_data_dir: Path = constants.BASE_DATA_DIR / "input_data"
    ) -> None:
        self.base_data_dir = base_data_dir

    def get_values(self, coords: np.ndarray, variable: constants.InputVariable, address_missing: bool = True) -> np.ndarray:
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by ShapeLoader.",
                logger,
            )

        shp_ffp = self.base_data_dir / self.VAR_TO_FILENAME_MAP[variable]
        if not shp_ffp.exists():
            utils.raise_log(
                FileNotFoundError,
                f"Shapefile for variable {variable} not found at {shp_ffp}.",
                logger,
            )

        point_df = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
            crs=constants.WGS84_EPSG,
        ).to_crs(epsg=constants.NZTM2000_EPSG)   

        shape_df = gpd.read_file(shp_ffp)
        
        if variable == constants.InputVariable.NZGeologyCategory:
            shape_df = shape_df.rename(columns={"gid": "value"})
            shape_df = shape_df.set_crs(epsg=constants.NZTM2000_EPSG, allow_override=True)
        elif variable in constants.InputVariable.NZGeologyAgeMin:
            shape_df = shape_df.rename(columns={"ABSMIN_MA": "value"})
        elif variable in constants.InputVariable.NZGeologyAgeMax:
            shape_df = shape_df.rename(columns={"ABSMAX_MA": "value"})
        elif variable == constants.InputVariable.NZLithologyCategory:
            shape_df = shape_df.rename(columns={"LITHO2014": "value"})
        elif variable == constants.InputVariable.NZGeologicalUnit:
            shape_df = shape_df.rename(columns={"MAPSYMBOL": "value"})
        else:
            utils.raise_log(
                NotImplementedError,
                f"Implementation missing for variable {variable} in ShapeLoader.",
                logger,
            )

        assert shape_df.crs.to_epsg() == constants.NZTM2000_EPSG, "Shape dataframe CRS is not NZTM2000."
        merged_df = gpd.sjoin(point_df, shape_df, how="left", predicate="intersects")

        if address_missing:
            logger.info(f"Addressing missing values for variable {variable} using nearest neighbor approach.")
            missing_mask = merged_df["value"].isna() | (merged_df["value"] == -9999)  
            nearest_df = gpd.sjoin_nearest(
                point_df[missing_mask], shape_df, max_distance=250, distance_col="dist_to_shape")
            merged_df.loc[missing_mask, "value"] = nearest_df["value"].values
        else:
            if variable == constants.InputVariable.NZGeologyCategory:
                merged_df["value"] = merged_df["value"].fillna(-9999).astype(int)

        return merged_df["value"].to_numpy()