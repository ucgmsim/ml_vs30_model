import time
import logging
from pathlib import Path


import matplotlib.pyplot as plt
import geopandas as gpd

import pandas as pd
import numpy as np
import shapely


from .. import constants
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)

class NZDistanceToCoast(BaseLoader):
    """
    Class for computing distance to coast for New Zealand locations.
    """

    SUPPORTED_VARIABLES = [
        constants.InputVariable.NZDistanceToCoast,
    ]

    def __init__(
        self, coastline_ffp: Path = constants.BASE_DATA_DIR / "input_data/nz_coastline/nz_coastline.parquet"
    ) -> None:
        self.coastline_df = gpd.read_file(coastline_ffp).set_crs(epsg=constants.WGS84_EPSG).to_crs(epsg=constants.NZTM2000_EPSG)

        assert self.coastline_df.geometry.apply(lambda g: g.is_ring).all()
        self.coastline_df["geometry"] = self.coastline_df.geometry.apply(lambda g: shapely.Polygon(g))
        
        # Extract all vertex coordinates to bypass massive bounding boxes
        points = shapely.get_coordinates(self.coastline_df.geometry.values)
        self.coastline_points = shapely.points(points)
        self.tree = shapely.STRtree(self.coastline_points)


    def get_values(self, coords: np.ndarray, variable: constants.InputVariable):
        if variable not in self.SUPPORTED_VARIABLES:
            raise ValueError(f"Variable {variable} is not supported by NZDistanceToCoast.")
        
        points = gpd.points_from_xy(coords[:, 0], coords[:, 1], crs=constants.WGS84_EPSG).to_crs(epsg=constants.NZTM2000_EPSG)
        _, distances = self.tree.query_nearest(points, return_distance=True, all_matches=False)

        return distances

class NZDistanceToRiver(BaseLoader):
    """Class for computing distance to nearest river for New Zealand locations."""

    SUPPORTED_VARIABLES = [
        constants.InputVariable.NZDistanceToRiver_ST1,
        constants.InputVariable.NZDistanceToRiver_ST2,
        constants.InputVariable.NZDistanceToRiver_ST3,
        constants.InputVariable.NZDistanceToRiver_ST4,
        constants.InputVariable.NZDistanceToRiver_ST5,
        constants.InputVariable.NZDistanceToRiver_ST6,
        constants.InputVariable.NZDistanceToRiver_ST7,
        constants.InputVariable.NZDistanceToRiver_ST8,
    ]

    def __init__(
        self, river_ffp: Path = constants.BASE_DATA_DIR / "input_data/nz_rivers/riverline.shp"
    ) -> None:
        self.river_df = gpd.read_file(river_ffp)
        assert self.river_df.crs.to_epsg() == constants.NZTM2000_EPSG, "River shapefile must be in NZTM2000 projection"
        
        # Extract all vertex coordinates to bypass massive bounding boxes

    def get_values(self, coords: np.ndarray, variable: constants.InputVariable):
        if variable not in self.SUPPORTED_VARIABLES:
            raise ValueError(f"Variable {variable} is not supported by NZDistanceToRiver.")
        
        # Get the Strahler order from the variable name
        strahler_order = int(variable.value.split("_st")[-1])
        # Filter river segments by Strahler order
        df = self.river_df[self.river_df.HIERARCHY == strahler_order]

        # Build SRT tree
        tree = shapely.STRtree(shapely.points(shapely.get_coordinates(df.geometry.values)))
        
        points = gpd.points_from_xy(coords[:, 0], coords[:, 1], crs=constants.WGS84_EPSG).to_crs(epsg=constants.NZTM2000_EPSG)
        _, distances = tree.query_nearest(points, return_distance=True, all_matches=False)

        return distances