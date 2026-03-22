import time
import logging
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import rasterio
from rasterio.transform import Affine
import xarray as xr
import numpy as np
import pandas as pd
import ml_tools as mlt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pygmt_helper import plotting as gmt_plotting

from . import constants
from . import data_loaders
from . import utils

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:

    rel_vs30_values_ffp: str
    index_col: str | None

    # apply_vs30_weighting: bool
    # max_vs30_weight: int
    # total_max_weight: int

    input_variables: list[constants.InputVariable]

    def __post_init__(self):

        self._vs30_values_ffp = None

    @property
    def vs30_values_ffp(self) -> Path:
        if self._vs30_values_ffp is None:
            self._vs30_values_ffp = constants.BASE_DATA_DIR / self.rel_vs30_values_ffp

        return self._vs30_values_ffp

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DataConfig":
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, config_ffp: Path) -> "DataConfig":
        config_dict = mlt.utils.load_yaml(config_ffp)
        config_dict["input_variables"] = [
            constants.InputVariable(var_str)
            for var_str in config_dict["input_variables"]
        ]

        return cls.from_dict(config_dict)


def gen_dataset(data_config: DataConfig, out_ffp: Path) -> None:
    """
    Generates a dataset containing Vs30 values and additional input variables,
    based on the given data configuration.
    """
    if out_ffp.exists():
        utils.raise_log(
            FileExistsError, f"Output file already exists: {out_ffp}", logger
        )

    vs30_values_df = pd.read_csv(data_config.vs30_values_ffp)
    vs30_values_df = vs30_values_df.astype({"vs30": float})
    assert np.all(np.isin(["lon", "lat", "vs30"], vs30_values_df.columns))

    df = vs30_values_df[["lon", "lat", "vs30"]].copy()
    if data_config.index_col is not None:
        df["index"] = vs30_values_df[data_config.index_col]
        df = df.set_index("index")

    # Add input variable values
    for variable in data_config.input_variables:
        df[variable.value] = get_input_values(df[["lon", "lat"]].to_numpy(), variable)

    if (df["vs30"] < 0.0).any():
        raise ValueError("Vs30 values must be non-negative.")
    if (zero_mask := (df["vs30"].values == 0.0)).any():
        df = df.loc[~zero_mask, :]
        logger.warning(
            f"{zero_mask.sum()} Vs30 values of 0.0 found, dropped from dataset."
        )

    df["vs30_bin"] = pd.cut(
        df.vs30,
        constants.VS30_WEIGHTING_BINS,
        labels=constants.VS30_WEIGHTING_BIN_NAMES,
    )

    df["dense_vs30_bin"] = pd.cut(
        df.vs30,
        constants.DENSE_VS30_BINS,
        labels=constants.DENSE_VS30_BIN_NAMES,
    )

    df.to_parquet(out_ffp)
    logger.info(f"Dataset saved to {out_ffp}")


def get_input_values(points: np.ndarray, variable: constants.InputVariable):
    """
    Gets the input variable values for the given points and variables.

    Params
    ------
    points: np.ndarray
        An array of shape (n_points, 2) containing the longitude and latitude of the points.
    variable: constants.InputVariable
        The input variable to retrieve values for.
    """
    logger.info(f"Processing variable: {variable.value}")
    data_source = constants.INPUT_VARIABLE_SOURCE_MAPPING.get(variable)

    if data_source == constants.DataSource.GeoMorpho90:
        logger.info(f"Using GeoMorpho90 data source for variable: {variable.value}")
        geomorpho90 = data_loaders.GeoMorpho90()
        values = geomorpho90.get_values(points, variable)
    elif data_source == constants.DataSource.TIFLoader:
        logger.info(f"Using TIFLoader data source for variable: {variable.value}")
        tif_loader = data_loaders.TIFLoader()
        values = tif_loader.get_values(points, variable)
    elif data_source == constants.DataSource.GlobalGWT:
        logger.info(f"Using GlobalGWT data source for variable: {variable.value}")
        global_gwt = data_loaders.GlobalGWT()
        values = global_gwt.get_values(points, variable)
    elif data_source == constants.DataSource.ShapeLoader:
        logger.info(f"Using ShapeLoader data source for variable: {variable.value}")
        shape_loader = data_loaders.ShapeLoader()
        values = shape_loader.get_values(points, variable)
    elif data_source == constants.DataSource.NZDistanceToCoast:
        logger.info(f"Using NZDistanceToCoast data source for variable: {variable.value}")
        dist_to_coast_loader = data_loaders.NZDistanceToCoast()
        values = dist_to_coast_loader.get_values(points, variable)
    else:
        error_msg = f"Data source for variable {variable} not implemented."
        logger.error(error_msg)
        raise NotImplementedError(error_msg)

    assert np.any(
        values != -9999
    ), f"Variable {variable} contains missing values after processing."
    return values


def _get_variable_da(
    land_points: np.ndarray,
    land_mask: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    variable: constants.InputVariable,
) -> xr.DataArray:
    """MP helper function"""
    variable_values = get_input_values(land_points, variable)

    if np.issubdtype(variable_values.dtype, np.floating):
        variable_da = xr.DataArray(
            np.full(land_mask.shape, np.nan, dtype=np.float64),
            coords=[lat_coords, lon_coords],
            dims=["lat", "lon"],
        )
    elif np.issubdtype(variable_values.dtype, np.integer):
        variable_da = xr.DataArray(
            np.full(land_mask.shape, -9999, dtype=np.int32),
            coords=[lat_coords, lon_coords],
            dims=["lat", "lon"],
        )
    else:
        raise ValueError(
            f"Unsupported data type for variable {variable}: {variable_values.dtype}"
        )

    variable_da.values[land_mask] = variable_values

    return variable, variable_da


def create_nz_input_grid(
    bounding_box: tuple[float, float, float, float],
    resolution: float,
    output_dir: Path,
    variables: list[constants.InputVariable],
    tolerance: int | None = None,
    min_area: int | None = None,
    n_procs: int = 1,
) -> pd.DataFrame:
    """Creates a grid of points within the specified bounding box and resolution."""
    assert (
        resolution > 1e-6
    ), "Resolution must be greater than 1e-6 degrees, as float32 is used."
    min_lon, max_lon, min_lat, max_lat = bounding_box
    # lon_coords = np.arange(min_lon, max_lon + resolution, resolution, dtype=np.float32)
    # lat_coords = np.arange(min_lat, max_lat + resolution, resolution, dtype=np.float32)

    width = int((max_lon - min_lon) / resolution)
    height = int((max_lat - min_lat) / resolution)

    lon_coords = np.linspace(min_lon, max_lon, width)
    lat_coords = np.linspace(max_lat, min_lat, height)  # Note: top to bottom

    lon_points, lat_points = np.meshgrid(lon_coords, lat_coords)
    points = np.stack([lon_points.ravel(), lat_points.ravel()], axis=-1)

    logger.info(
        f"Generating land mask for {len(points)} points using a tolerance of {tolerance}."
    )
    start = time.time()
    map_data = gmt_plotting.NZMapData.load()
    land_mask = gmt_plotting.on_land_via_join(
        map_data, points[:, ::-1], tolerance=tolerance, min_area=min_area
    )
    land_mask = land_mask.reshape(lon_points.shape)
    print(f"Took: {time.time() - start} get land mask")

    on_land_da = xr.DataArray(
        land_mask, coords=[lat_coords, lon_coords], dims=["lat", "lon"]
    )
    grid_dataset = xr.Dataset({"on_land": on_land_da})

    logger.info("Extracting variable values for land points.")
    land_points = points[land_mask.ravel()]
    if n_procs == 1:
        for variable in variables:
            logger.info(f"Processing variable: {variable.value}")
            _, variable_da = _get_variable_da(
                land_points, land_mask, lat_coords, lon_coords, variable
            )
            grid_dataset[variable.value] = variable_da
    else:
        with mp.Pool(processes=n_procs) as p:
            results = p.starmap(
                _get_variable_da,
                [
                    (land_points, land_mask, lat_coords, lon_coords, variable)
                    for variable in variables
                ],
            )

            for variable, variable_da in results:
                grid_dataset[variable.value] = variable_da

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving grid dataset to {output_dir}")
    grid_dataset.to_netcdf(output_dir / "input_grid.nc")

    logger.info("Saving variable TIFF files for tiling.")
    transform = rasterio.transform.from_bounds(
        min_lon, min_lat, max_lon, max_lat, width, height
    )

    for variable in variables:
        cur_data = grid_dataset[variable.value].values

        # Create a mask for valid data (not NaN)
        if np.issubdtype(cur_data.dtype, np.floating):
            valid_mask = ~np.isnan(cur_data)
        elif np.issubdtype(cur_data.dtype, np.integer):
            valid_mask = cur_data != -9999
        else:
            raise ValueError(
                f"Unsupported data type for variable {variable}: {cur_data.dtype}"
            )

        # Normalize & convert to RGBA
        norm = Normalize(
            vmin=np.min(cur_data[valid_mask]), vmax=np.max(cur_data[valid_mask])
        )
        normalized_data = norm(cur_data)
        rgba_data = plt.get_cmap("viridis")(normalized_data)
        rgba_uint8 = (rgba_data * 255).astype(np.uint8)
        rgba_uint8[~valid_mask, 3] = 0

        with rasterio.open(
            output_dir / f"{variable.value}.tif",
            "w",
            driver="GTiff",
            height=cur_data.shape[0],
            width=cur_data.shape[1],
            count=4,
            dtype=rasterio.uint8,
            transform=transform,
            crs="EPSG:4326",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(rgba_uint8[:, :, 0], 1)  # Red channel
            dst.write(rgba_uint8[:, :, 1], 2)  # Green channel
            dst.write(rgba_uint8[:, :, 2], 3)  # Blue channel
            dst.write(rgba_uint8[:, :, 3], 4)  # Alpha channel
