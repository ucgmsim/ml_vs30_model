import logging
from pathlib import Path

import rasterio
from rasterio import transform
import seaborn as sns
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import typer

import ml_tools as mlt
import ml_vs30_model as vs30

app = typer.Typer(
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@app.command("gen-dataset")
def gen_dataset(
    config_ffp: Path,
    out_ffp: Path,
    log_ffp: Path | None = None,
    address_missing: bool = True,
):
    """
    Creates a dataset for training a VS30 model,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging(log_file=log_ffp)
    logging.getLogger("rclone").setLevel(logging.WARNING)

    config = vs30.DataConfig.from_yaml(config_ffp)
    vs30.data.gen_dataset(config, out_ffp, address_missing=address_missing)


@app.command("create-nz-nztm-input-grid")
def create_nz_nztm_input_grid(
    dx: float,
    dy: float,
    output_dir: Path,
    variables: list[vs30.constants.InputVariable],
    n_procs: int = 1,
):
    """
    Creates a grid of input variable values for New Zealand in NZTM coordinates,
    based on the provided grid spacing (in meters).
    """
    mlt.utils.setup_logging()
    vs30.data.create_nz_nztm_input_grid(dx, dy, output_dir, variables, n_procs=n_procs)


@app.command("get-foster-residuals")
def get_foster_residuals(dataset_ffp: Path, foster_tif_ffp: Path, output_ffp: Path):
    dataset_df = pd.read_parquet(dataset_ffp)

    coords = np.stack(vs30.constants.WGS84_TO_NZTM_TRANSFORMER.transform(
        dataset_df["lon"].values, dataset_df["lat"].values
    ), axis=1)

    with rasterio.open(foster_tif_ffp) as ds:
        assert ds.crs.to_epsg() == vs30.constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"
        foster_original_data = ds.read(1)

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
        foster_original_vs30_mean = foster_original_data[rows, cols]

    foster_results = dataset_df[["lon", "lat", "vs30", "quality_score"]].copy()
    foster_results["pred_vs30"] = foster_original_vs30_mean
    foster_results["ln_residual"] = np.log(foster_results["vs30"]) - np.log(foster_results["pred_vs30"])
    foster_results["mae"] = np.abs(foster_results["vs30"] - foster_results["pred_vs30"])

    nan_mask = foster_results["pred_vs30"] < 0
    foster_results.loc[nan_mask, ["pred_vs30", "ln_residual", "mae"]] = np.nan

    foster_results.to_parquet(output_ffp)



@app.command("select-test-sites")
def select_test_sites(dataset_ffp: Path, output_dir: Path, seed: int):
    """
    Selects test sites from the dataset, stratified by Vs30 bins.
    """
    mlt.utils.setup_logging()
    vs30.data.select_test_sites(dataset_ffp, output_dir, seed)


@app.command("create-nz-quaternary-polygon-groups")
def create_nz_quaternary_polygon_groups(
    dataset_ffp: Path,
    output_dir: Path,
):
    """
    Creates connected polygon groups for the quaternary region polygons.
    Filtered to those that contain at least 3 Q1 points.
    """
    logger = mlt.utils.setup_logging()
    dataset_df = pd.read_parquet(dataset_ffp)

    geo_df = gpd.read_file(vs30.constants.QMAP_FFP)
    geo_df = geo_df[geo_df["ABSMAX_MA"] < 2.58]

    logger.info("Computing quaternary polygon groups...")
    filtered_groups_df, filtered_ind_group_polygons, raw_comb_group_polygons_df = (
        vs30.data.compute_nz_quaternary_polygon_groups(
            geo_df, dataset_df, q1_count_threshold=3, min_group_area=5e6)
    )

    filtered_groups_df.to_parquet(output_dir / "filtered_groups.parquet")
    mlt.utils.write_pickle(
        filtered_ind_group_polygons,
        output_dir / "filtered_ind_group_polygons.pkl",
        clobber=True,
    )
    raw_comb_group_polygons_df.to_parquet(
        output_dir / "raw_comb_group_polygons.parquet"
    )

    ## Create maps
    logger.info("Creating maps...")
    # All quaternary polygons
    m = folium.Map(location=[-43.5, 172.6], zoom_start=6, tiles="cartodb positron")
    for k, row in geo_df.to_crs(vs30.constants.WGS84_EPSG_STR).iterrows():
        folium.GeoJson(row.geometry, color="blue", opacity=0.7).add_to(m)
    m.save(output_dir / "quaternary_regions.html")

    # All quaternary groupings
    m = folium.Map(location=[-43.5, 172.6], zoom_start=6, tiles="cartodb positron")
    for k, row in raw_comb_group_polygons_df.to_crs(
        vs30.constants.WGS84_EPSG_STR
    ).iterrows():
        folium.GeoJson(row.geometry, color="blue", opacity=0.7).add_to(m)
    m.save(output_dir / "quaternary_groupings.html")

    # Filtered quaternary groupings
    m = folium.Map(location=[-43.5, 172.6], zoom_start=6, tiles="cartodb positron")
    colors = sns.color_palette("tab10", n_colors=len(filtered_groups_df)).as_hex()
    for i, (k, row) in enumerate(
        filtered_groups_df.to_crs(vs30.constants.WGS84_EPSG_STR).iterrows()
    ):
        folium.GeoJson(row.geometry, color=colors[i], opacity=0.7).add_to(m)
    m.save(output_dir / "filtered_quaternary_groupings.html")

    # Create final groupings with boundaries
    loader = vs30.data_loaders.NZQuaternaryRegionLoader()
    loader.group_polygons.to_parquet(output_dir / "final_group_polygons.parquet")

    m = folium.Map(location=[-43.5, 172.6], zoom_start=6, tiles="cartodb positron")
    for i, cluster in enumerate(loader.group_polygons.to_crs(vs30.constants.WGS84_EPSG_STR).geometry.values):
        folium.GeoJson(cluster, color="blue").add_to(m)

    for boundary in loader.group_boundaries.to_crs(vs30.constants.WGS84_EPSG_STR).geometry.values:
        folium.GeoJson(boundary, color="red").add_to(m)
    m.save("/Users/claudy/dev/work/data/vs30/other/quaternary_clustering/final_grouping.html")


if __name__ == "__main__":
    app()
