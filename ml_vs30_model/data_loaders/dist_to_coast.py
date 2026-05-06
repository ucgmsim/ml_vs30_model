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
        
        # Pre-build STRtree on boundaries for fast nearest-neighbour lookup
        self.boundaries = self.coastline_df.geometry.boundary.values
        
        # Extract all vertex coordinates to bypass massive bounding boxes
        points = shapely.get_coordinates(self.boundaries)
        self.coastline_points = shapely.points(points)
        self.tree = shapely.STRtree(self.coastline_points)


    def get_values(self, coords: np.ndarray, variable: constants.InputVariable):
        if variable not in self.SUPPORTED_VARIABLES:
            raise ValueError(f"Variable {variable} is not supported by NZDistanceToCoast.")
        
        points = gpd.points_from_xy(coords[:, 0], coords[:, 1], crs=constants.WGS84_EPSG).to_crs(epsg=constants.NZTM2000_EPSG)
        _, distances = self.tree.query_nearest(points, return_distance=True, all_matches=False)

        return distances

