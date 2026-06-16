import logging
from pathlib import Path


import geopandas as gpd

import pandas as pd
import numpy as np
import shapely

import ml_tools as mlt

from .. import constants
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class NZQuaternaryRegionLoader(BaseLoader):
    """
    Class for computing quaternary region for New Zealand locations.
    """

    SUPPORTED_VARIABLES = [
        constants.InputVariable.NZQuaternaryRegion,
    ]

    def __init__(
        self,
        grouping_polygons_dir: Path = constants.BASE_DATA_DIR
        / "other/quaternary_clustering",
        grouping_boundaries_dir: Path = constants.BASE_DATA_DIR
        / "other/quaternary_clustering/grouping_boundaries",
    ):
        logger.info("Preparing quaternary region loader...")

        # Load filtered grouping polygons
        ind_group_polygons = mlt.utils.load_pickle(
            grouping_polygons_dir / "filtered_ind_group_polygons.pkl"
        )
        group_polygons_df = gpd.read_file(
            grouping_polygons_dir / "filtered_groups.parquet"
        ).set_index("region", drop=True)

        # Load manually defined grouping boundaries
        group_boundary_files = list(grouping_boundaries_dir.glob("*.geojson"))
        self.group_boundaries = gpd.GeoDataFrame(
            pd.concat(
                [gpd.read_file(ffp) for ffp in group_boundary_files],
                ignore_index=True,
            )
        ).drop(columns=["style"])
        self.group_boundaries.index = [f.stem for f in group_boundary_files]
        self.group_boundaries = self.group_boundaries.to_crs(
            epsg=constants.NZTM2000_EPSG
        )

        self.group_polygons = {}
        for group_name, polygons in ind_group_polygons.items():
            if group_name in self.group_boundaries.index:
                boundary = self.group_boundaries.loc[group_name, "geometry"]
                intersections = [
                    shapely.intersection(boundary, poly) for poly in polygons
                ]
                self.group_polygons[group_name] = shapely.unary_union(intersections)
            else:
                self.group_polygons[group_name] = group_polygons_df.loc[
                    group_name, "geometry"
                ]

        self.group_polygons = gpd.GeoDataFrame.from_dict(
            self.group_polygons, orient="index", columns=["geometry"]
        ).set_crs(epsg=constants.NZTM2000_EPSG)

    def get_values(self, coords: np.ndarray, variable: constants.InputVariable):
        if variable not in self.SUPPORTED_VARIABLES:
            raise ValueError(
                f"Variable {variable} is not supported by NZQuaternaryRegionLoader."
            )

        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
            crs=constants.WGS84_EPSG,
        ).to_crs(epsg=constants.NZTM2000_EPSG)
        merged_df = gpd.sjoin(
            points, self.group_polygons, how="left", predicate="intersects"
        )

        merged_df["index_right"] = (
            merged_df["index_right"].fillna("no_region").astype(str)
        )

        assert (
            merged_df["index_right"].isin(constants.QUATERNARY_REGION_TO_ID_MAPPING.index).all()
        ), "Merged dataframe contains unknown regions."
        return merged_df["index_right"].map(constants.QUATERNARY_REGION_TO_ID_MAPPING).values

    @staticmethod
    def id_to_region(region_ids: np.ndarray) -> str:
        assert np.isin(region_ids, constants.QUATERNARY_ID_TO_REGION_MAPPING.index).all(), "Region IDs contain unknown values."
        return constants.QUATERNARY_ID_TO_REGION_MAPPING[region_ids]