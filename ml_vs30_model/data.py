import time
import logging
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from functools import partial

import xarray as xr
import numpy as np
import pandas as pd
import shapely
import networkx as nx
import geopandas as gpd
import plotly.graph_objects as go
from pyproj import Transformer
from tqdm import tqdm

import ml_tools as mlt
from pygmt_helper import plotting as gmt_plotting
from qcore import coordinates

from . import constants
from . import data_loaders
from . import utils
from .feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:

    rel_vs30_values_ffp: str
    index_col: str | None

    drop_sites: list[str] | None
    drop_quality_score: list[str] | None

    input_variables: list[constants.InputVariable]
    derived_variables: list[constants.InputVariable] | None

    def __post_init__(self):
        self._vs30_values_ffp = None

        self._derived_variables_check()

    @property
    def vs30_values_ffp(self) -> Path:
        if self._vs30_values_ffp is None:
            self._vs30_values_ffp = constants.BASE_DATA_DIR / self.rel_vs30_values_ffp

        return self._vs30_values_ffp

    def _derived_variables_check(self):
        """Checks that derived variables have their dependencies met."""
        if self.derived_variables is not None:
            for derived_var in self.derived_variables:
                if derived_var not in constants.DERIVED_VARIABLES_DEPENDENCIES:
                    utils.raise_log(
                        ValueError,
                        f"Derived variable {derived_var}"
                        " does not have defined dependencies.",
                        logger,
                    )

                dependencies = constants.DERIVED_VARIABLES_DEPENDENCIES[derived_var]
                for dep in dependencies:
                    if dep not in self.input_variables:
                        utils.raise_log(
                            ValueError,
                            f"Derived variable {derived_var} depends on {dep}, "
                            "which is not in input variables.",
                            logger,
                        )

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
        if config_dict.get("derived_variables") is not None:
            config_dict["derived_variables"] = [
                constants.InputVariable(var_str)
                for var_str in config_dict["derived_variables"]
            ]

        return cls.from_dict(config_dict)


def gen_dataset(
    data_config: DataConfig, out_ffp: Path, address_missing: bool = True
) -> None:
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

    if data_config.drop_sites is not None:
        drop_mask = vs30_values_df["sta"].isin(data_config.drop_sites)
        if drop_mask.any():
            vs30_values_df = vs30_values_df.loc[~drop_mask, :]
            logger.info(
                f"Dropped {drop_mask.sum()} sites based on drop_sites list in config."
            )
        else:
            logger.warning(
                "No sites were dropped based on drop_sites list in config. "
                "Check that site identifiers in drop_sites match those in dataset."
            )

    df = vs30_values_df[["lon", "lat", "vs30"]].copy()
    if data_config.index_col is not None:
        df["index"] = vs30_values_df[data_config.index_col]
        df = df.set_index("index")


    if "std" in vs30_values_df.columns:
        assert np.all(df.index.values == vs30_values_df.sta.values)
        df["ln_vs30_std"] = vs30_values_df["std"].values

    # Include quality score if available
    if "quality_score" in vs30_values_df.columns:
        assert np.all(df.index.values == vs30_values_df.sta.values)
        df["quality_score"] = vs30_values_df["quality_score"].values

        # Always use 0.3 as ln_vs30_std for Q3 sites
        df["ln_vs30_std"] = np.where(df.quality_score == "Q3", 0.3, df.ln_vs30_std.values)

        if data_config.drop_quality_score is not None:
            drop_mask = df["quality_score"].isin(data_config.drop_quality_score)
            if drop_mask.any():
                df = df.loc[~drop_mask, :]
                logger.info(
                    f"Dropped {drop_mask.sum()} sites based on drop_quality_score list in config."
                )
            else:
                logger.warning(
                    "No sites were dropped based on drop_quality_score list in config. "
                    "Check that quality score values in drop_quality_score match those in dataset."
                )
                
    # Add input variable values
    logger.info(f"Retrieving input variable values for {len(df)} sites.")
    for variable in data_config.input_variables:
        df[variable.value] = get_input_values(
            df[["lon", "lat"]].to_numpy(), variable, address_missing=address_missing
        )

    # Add derived variables
    logger.info("Computing derived variable values.")
    if data_config.derived_variables is not None:
        feature_engineer = FeatureEngineer(df)
        df[data_config.derived_variables] = feature_engineer.compute_features(
            data_config.derived_variables
        )[data_config.derived_variables]

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


def get_input_values(
    points: np.ndarray, variable: constants.InputVariable, address_missing: bool = True
):
    """
    Gets the input variable values for the given points and variables.

    Params
    ------
    points: np.ndarray
        An array of shape (n_points, 2) containing the longitude and latitude of the points.
    variable: constants.InputVariable
        The input variable to retrieve values for.
    address_missing: bool
        Whether to address missing values (e.g. negative depth to bedrock)
        by finding nearest valid value. Can be slow for large number of points.
    """
    logger.info(f"Processing variable: {variable.value}")
    data_source = constants.INPUT_VARIABLE_SOURCE_MAPPING.get(variable)

    if data_source == constants.DataSource.GeoMorpho90:
        logger.info(f"Using GeoMorpho90 data source for variable: {variable.value}")
        geomorpho90 = data_loaders.GeoMorpho90()
        values = geomorpho90.get_values(
            points, variable, address_missing=address_missing
        )
    elif data_source == constants.DataSource.SRTMGL1:
        logger.info(f"Using SRTMGL1 data source for variable: {variable.value}")
        srtm_loader = data_loaders.SRTMGL1()
        values = srtm_loader.get_values(
            points, variable, address_missing=address_missing
        )
    elif data_source == constants.DataSource.TIFLoader:
        logger.info(f"Using TIFLoader data source for variable: {variable.value}")
        tif_loader = data_loaders.TIFLoader()
        values = tif_loader.get_values(
            points, variable, address_missing=address_missing
        )
    elif data_source == constants.DataSource.NZTMTIFLoader:
        logger.info(f"Using NZTMTIFLoader data source for variable: {variable.value}")
        nztm_loader = data_loaders.NZTMTIFLoader()
        values = nztm_loader.get_values(points, variable, address_missing=address_missing)
    elif data_source == constants.DataSource.GlobalGWT:
        logger.info(f"Using GlobalGWT data source for variable: {variable.value}")
        global_gwt = data_loaders.GlobalGWT()
        values = global_gwt.get_values(points, variable)
    elif data_source == constants.DataSource.ShapeLoader:
        logger.info(f"Using ShapeLoader data source for variable: {variable.value}")
        shape_loader = data_loaders.ShapeLoader()
        values = shape_loader.get_values(
            points, variable, address_missing=address_missing
        )
    elif data_source == constants.DataSource.NZDistanceToCoast:
        logger.info(
            f"Using NZDistanceToCoast data source for variable: {variable.value}"
        )
        dist_to_coast_loader = data_loaders.NZDistanceToCoast()
        values = dist_to_coast_loader.get_values(points, variable)
    elif data_source == constants.DataSource.NZDistanceToRiver:
        logger.info(
            f"Using NZDistanceToRiver data source for variable: {variable.value}"
        )
        dist_to_river_loader = data_loaders.NZDistanceToRiver()
        values = dist_to_river_loader.get_values(points, variable)
    elif data_source == constants.DataSource.NZQuaternaryRegion:
        logger.info(
            f"Using NZQuaternaryRegion data source for variable: {variable.value}"
        )
        quaternary_region_loader = data_loaders.NZQuaternaryRegionLoader()
        values = quaternary_region_loader.get_values(points, variable)
    else:
        error_msg = f"Data source for variable {variable} not implemented."
        logger.error(error_msg)
        raise NotImplementedError(error_msg)

    if (
        address_missing
        # Skip as categorical variable of type str (object)
        and variable
        not in [
            constants.InputVariable.NZLithologyCategory,
            constants.InputVariable.NZGeologicalUnit,
            constants.InputVariable.NZQuaternaryRegion,
        ]
        and (np.any(values == constants.INTEGER_NO_DATA_VALUE) or np.any(np.isnan(values)))
        and variable
    ):
        if variable in [
            constants.InputVariable.NZNLMGroundwaterDepth,
            constants.InputVariable.NZNWTGroundwaterDepth,
            constants.InputVariable.AbsoluteDepthToBedrock,
        ]:
            logger.info(
                f"Found missing values for variable {variable}. "
                f"However, this is expected, therefore not addressing missing values."
            )
        else:
            utils.raise_log(
                ValueError,
                f"Variable {variable} contains missing values after processing.",
                logger,
            )

    logger.info(f"Completed processing for variable: {variable.value}")
    return values


def __get_variable_da(
    variable_values: np.ndarray,
    land_mask: np.ndarray,
    nztm_y_coords: np.ndarray,
    nztm_x_coords: np.ndarray,
    variable: constants.InputVariable,
) -> xr.DataArray:
    """
    Helper function to create a DataArray for a variable,
    with values filled in for land points and NaN/-9999 for non-land points.
    """
    if np.issubdtype(variable_values.dtype, np.floating):
        variable_da = xr.DataArray(
            np.full(land_mask.shape, np.nan, dtype=np.float32),
            coords=[nztm_y_coords, nztm_x_coords],
            dims=["y", "x"],
        )
        variable_da = variable_da.rio.write_crs(
            constants.NZTM2000_EPSG_STR
        ).rio.write_nodata(np.nan)
    elif np.issubdtype(variable_values.dtype, np.integer):
        variable_da = xr.DataArray(
            np.full(land_mask.shape, -9999, dtype=np.int32),
            coords=[nztm_y_coords, nztm_x_coords],
            dims=["y", "x"],
        )
        variable_da = variable_da.rio.write_crs(
            constants.NZTM2000_EPSG_STR
        ).rio.write_nodata(-9999)
    else:
        raise ValueError(
            f"Unsupported data type for variable {variable}: {variable_values.dtype}"
        )

    variable_da.values[land_mask] = variable_values
    return variable_da


def _get_variable_nztm_da(
    land_points: np.ndarray,
    land_mask: np.ndarray,
    nztm_y_coords: np.ndarray,
    nztm_x_coords: np.ndarray,
    variable: constants.InputVariable,
) -> xr.DataArray:
    """
    MP helper function

    Creates a DataArray for the given variable, with values filled in for land points
    and NaN or -9999 for non-land points.

    Parameter land_points needs to be shape[:, 2], in lon/lat order,
    and the resulting DataArray uses nztm_x/y coordinates.
    """
    variable_values = get_input_values(land_points, variable, address_missing=False)
    variable_da = __get_variable_da(
        variable_values, land_mask, nztm_y_coords, nztm_x_coords, variable
    )
    return variable, variable_da


def _compute_derived_variables_nztm_da(
    dataset_ffp: Path,
    land_mask: np.ndarray,
    nztm_y_coords: np.ndarray,
    nztm_x_coords: np.ndarray,
    variable: constants.InputVariable,
):
    """
    MP helper function

    Creates a DataArray for the given derived variable, with values filled in for land points
    and NaN or -9999 for non-land points.

    Given dataset must contain the required dependencies for the derived variable.

    Parameter land_points needs to be shape[:, 2], in lon/lat order,
    and the resulting DataArray uses nztm_x/y coordinates.
    """
    dependencies = constants.DERIVED_VARIABLES_DEPENDENCIES[variable]

    # Load required data
    with xr.open_dataset(dataset_ffp, mode="r") as ds:
        data_df = (
            ds[dependencies]
            .to_dataframe()
            .loc[land_mask.ravel()]
            .reset_index(drop=True)
        )

    # Compute derived variable values
    feature_engineer = FeatureEngineer(data_df)
    variable_values = feature_engineer.compute_features([variable])[variable].values

    variable_da = __get_variable_da(
        variable_values, land_mask, nztm_y_coords, nztm_x_coords, variable
    )
    return variable, variable_da


def create_nz_nztm_input_grid(
    dx: float,
    dy: float,
    output_dir: Path,
    variables: list[constants.InputVariable],
    n_procs: int = 1,
):
    """
    Creates a grid of input variable values for New Zealand in NZTM coordinates,
    based on the provided grid spacing (in meters).

    The grid is created within the NZTM bounding box, and values are extracted for land points only.

    Parameters
    ----------
    dx: float
        Grid spacing in the x-direction (longitude) in meters.
    dy: float
        Grid spacing in the y-direction (latitude) in meters.
    output_dir: Path
        Directory to save the output grid dataset.
    variables: list[constants.InputVariable]
        List of input variables to include in the grid dataset.
        Can include both original and derived variables.
    n_procs: int
        Number of processes to use for parallel processing.
    """
    min_x, max_x, min_y, max_y = constants.NZTM_BOUNDING_BOX
    # Coordinates are pixel centers, and y is descending (north-up), matching
    # standard GDAL/rasterio raster conventions.
    nztm_x = np.arange(min_x + dx / 2, max_x, dx)
    nztm_y = np.arange(max_y - dy / 2, min_y, -dy)

    grid_x, grid_y = np.meshgrid(nztm_x, nztm_y)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_points_latlon = coordinates.nztm_to_wgs_depth(grid_points[:, ::-1])

    logger.info(f"Generating land mask for {len(grid_points)} points.")
    map_data = gmt_plotting.NZMapData.load()

    start = time.time()
    coast_mask, water_mask = gmt_plotting.get_coast_water_mask(
        map_data,
        grid_points_latlon,
    )
    logger.info(f"Took: {time.time() - start} for optimized coast/water mask")

    land_mask = coast_mask & ~water_mask
    land_mask = land_mask.reshape(grid_x.shape)

    on_land_da = xr.DataArray(
        land_mask.astype(np.int16), coords=[nztm_y, nztm_x], dims=["y", "x"]
    )



    # on_land_da.attrs["_FillValue"] = np.int16(-9999)
    grid_dataset = xr.Dataset({"on_land": on_land_da})
    grid_dataset = grid_dataset.rio.write_crs(constants.NZTM2000_EPSG_STR)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    out_ffp = output_dir / "input_grid.nc"

    grid_dataset = grid_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y")
    grid_dataset = grid_dataset.rio.write_grid_mapping()
    grid_dataset = grid_dataset.rio.write_crs("EPSG:2193")

    enc = dict(grid_dataset["on_land"].encoding)
    enc.update({"dtype": np.int16, "_FillValue": -9999})

    grid_dataset.to_netcdf(out_ffp, encoding={"on_land": enc})
    del grid_dataset

    og_variables = [
        var
        for var in variables
        if var not in constants.DERIVED_VARIABLES_DEPENDENCIES.keys()
    ]
    derived_variables = [
        var
        for var in variables
        if var in constants.DERIVED_VARIABLES_DEPENDENCIES.keys()
    ]

    logger.info("Extracting variable values for land points.")
    land_points_lonlat = grid_points_latlon[land_mask.ravel()][:, ::-1]
    if n_procs == 1:
        for variable in og_variables:
            _, variable_da = _get_variable_nztm_da(
                land_points_lonlat, land_mask, nztm_y, nztm_x, variable
            )
            _write_variable_to_netcdf(variable_da, variable, out_ffp)
            # xr.Dataset({variable.value: variable_da}).to_netcdf(out_ffp, mode="a")
            del variable_da
    else:
        initializer_fn = partial(mlt.utils.setup_logging)
        fn_call = partial(
            _get_variable_nztm_da, land_points_lonlat, land_mask, nztm_y, nztm_x
        )
        with mp.Pool(processes=n_procs, initializer=initializer_fn) as p:
            results = p.imap_unordered(fn_call, og_variables)

            for variable, variable_da in results:
                _write_variable_to_netcdf(variable_da, variable, out_ffp)
                # xr.Dataset({variable.value: variable_da}).to_netcdf(out_ffp, mode="a")
                del variable_da

    logger.info("Computing derived variable values.")
    for variable in derived_variables:
        _, variable_da = _compute_derived_variables_nztm_da(
            out_ffp, land_mask, nztm_y, nztm_x, variable
        )
        _write_variable_to_netcdf(variable_da, variable, out_ffp)
        # xr.Dataset({variable.value: variable_da}).to_netcdf(out_ffp, mode="a")
        del variable_da


def _write_variable_to_netcdf(variable_da: xr.DataArray, variable: constants.InputVariable, out_ffp: Path) -> None:
    """Writes a variable DataArray to a NetCDF file, preserving dtype."""
    variable_da.attrs.pop("_FillValue", None)
    ds = xr.Dataset({variable.value: variable_da})

    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_grid_mapping()
    ds = ds.rio.write_crs("EPSG:2193")

    enc = dict(ds[variable.value].encoding)
    if np.issubdtype(variable_da.dtype, np.integer):
        enc.update({"dtype": variable_da.dtype, "_FillValue": -9999})
    else:
        enc.update({"dtype": variable_da.dtype, "_FillValue": None})

    # if np.issubdtype(variable_da.dtype, np.integer):
    #     encoding = {variable.value: {"dtype": variable_da.dtype, "_FillValue": -9999}}
    # else:
    #     encoding = {variable.value: {"dtype": variable_da.dtype, "_FillValue": None}}

    # ds.to_netcdf(out_ffp, mode="a", encoding=encoding)
    ds.to_netcdf(out_ffp, mode="a", encoding={variable.value: enc})

def select_test_sites(dataset_ffp: Path, output_dir: Path, seed: int):
    """
    Selects test sites from the dataset, stratified by Vs30 bins.
    """
    np.random.seed(seed)
    dataset_df = pd.read_parquet(dataset_ffp)

    sites_to_sample = {
        "0-180": 5,
        "180-360": 15,
        "360-760": 15,
        "760-10000": 10,
    }

    test_sites = []
    for vs30_bin in constants.VS30_WEIGHTING_BIN_NAMES:
        n_sites_to_sample = sites_to_sample[vs30_bin]
        bin_sites = dataset_df.loc[dataset_df["vs30_bin"] == vs30_bin].index.values

        assert (
            len(bin_sites) >= n_sites_to_sample
        ), f"Not enough sites in Vs30 bin {vs30_bin} to sample {n_sites_to_sample} test sites."

        sites = np.random.choice(bin_sites, size=n_sites_to_sample, replace=False)
        test_sites.extend(sites)
    np.save(output_dir / "test_sites.npy", np.array(test_sites))

    # Create spatial plot of test sites to check distribution
    test_df = dataset_df.loc[test_sites]
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        map=dict(zoom=3, center=dict(lat=test_df.lat.mean(), lon=test_df.lon.mean())),
        showlegend=False,
    )
    fig.add_trace(
        go.Scattermap(
            lon=test_df["lon"],
            lat=test_df["lat"],
            name="Measurements",
            mode="markers",
            marker=dict(size=4, color="blue"),
        )
    )
    fig.write_html(output_dir / "test_sites_map.html")


def compute_nz_quaternary_polygon_groups(geo_df: gpd.GeoDataFrame, dataset_df: gpd.GeoDataFrame, q1_count_threshold: int = 3, min_group_area: float = 1e6):
    """Computes connected polygon groups for quaternary regions in New Zealand."""
    GROUPING_SITE_MAPPING = {
        "ICCS": "invercargill",
        "ROLC": "canterbury",
        "WEMS": "wellington",
        "SOCS": "wellington_hutt",
        "PNRS": "palmerston_north",
        "OPSS": "taranaki",
        "GWTS": "gisborne",
        "NCHS": "napier",
        "RPCS": "taupo",
    }

    polygons = geo_df["geometry"].values
    tree = shapely.strtree.STRtree(polygons)

    # Create groupings
    G = nx.Graph()
    for i, poly in tqdm(enumerate(polygons), total=len(polygons)):
        G.add_node(i)

        # candidate neighbors via spatial index
        candidates = tree.query(poly)

        for j in candidates:
            # Ensure we only check each pair once
            if i >= j:
                continue

            other = polygons[j]
            if poly.intersects(other) and (
                shapely.intersection(poly, other).length > 1_000
                or (poly.contains(other) or other.contains(poly))
            ):
                G.add_edge(i, j)
    groups = list(nx.connected_components(G))

    raw_comb_group_polygons = {}
    raw_ind_group_polygons = {}
    for i, group in enumerate(groups):
        if len(group) <= 1:
            continue

        cur_group_polygon = shapely.unary_union([polygons[j] for j in group])
        if cur_group_polygon.area < min_group_area:
            continue

        raw_comb_group_polygons[i] = cur_group_polygon
        raw_ind_group_polygons[i] = [polygons[j] for j in group]

    raw_comb_group_polygons_df = gpd.GeoDataFrame(
        geometry=list(raw_comb_group_polygons.values()),
        index=list(raw_comb_group_polygons.keys()),
        columns=["geometry"],
    ).set_crs(constants.NZTM2000_EPSG_STR)

    # Convert dataset_df points to NZTM for spatial operations
    to_nztm_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    dataset_df["geometry"] = dataset_df.apply(
        lambda row: shapely.Point(
            to_nztm_transformer.transform(row["lon"], row["lat"])
        ),
        axis=1,
    )

    # Apply filtering
    q1_points = dataset_df[dataset_df.quality_score == "Q1"]["geometry"].values
    filtered_groups = {i:p for i, p in raw_comb_group_polygons.items() if p.contains(q1_points).sum() > q1_count_threshold}

    filtered_groups_df = gpd.GeoDataFrame(filtered_groups.values(), index=filtered_groups.keys(), columns=["geometry"]).set_crs(constants.NZTM2000_EPSG_STR)
    filtered_groups_df["group_id"] = filtered_groups_df.index

    mapping_sites = list(GROUPING_SITE_MAPPING.keys())
    for i in filtered_groups_df.index:
        cur_site = filtered_groups_df.loc[i].geometry.contains(dataset_df.loc[mapping_sites, "geometry"]).idxmax()
        filtered_groups_df.loc[i, "region"] = GROUPING_SITE_MAPPING.get(cur_site, "Unknown")
    filtered_groups_df = filtered_groups_df.set_index("region", drop=True)

    filtered_ind_group_polygons = {k: raw_ind_group_polygons[row.group_id] for k, row in filtered_groups_df.iterrows()}

    return filtered_groups_df, filtered_ind_group_polygons, raw_comb_group_polygons_df

