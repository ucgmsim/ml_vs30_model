import logging
import os
import time
import io
from pathlib import Path

import rasterio
from rasterio import transform
import geopandas as gpd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from scipy.stats import norm
import xarray as xr
import typer
import pygmt

from qcore import coordinates
from pygmt_helper import plotting
import ml_tools as mlt


import ml_vs30_model as vs30

app = typer.Typer()


def _fig_settings(logger: logging.Logger):
    for cur_env_key in os.environ.keys():
        if cur_env_key.startswith("fig_"):
            logger.info(
                f"Using figure parameter: {cur_env_key} = {os.environ[cur_env_key]}"
            )

    # Update font size
    if vs30.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": vs30.constants.FIG_FONT_SIZE,
            }
        )


@app.command("gen-site-map")
def gen_site_map(dataset_ffp: Path, output_dir: Path):
    mlt.utils.setup_logging()

    main_projection = "M15.5c"
    inset_projection = "M7.5c"

    df = pd.read_parquet(dataset_ffp)

    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=True,
        plot_highways=True,
        region=vs30.constants.NZ_BOUNDING_BOX,
        projection=main_projection,
    )

    # Set the extreme colors
    pygmt.config(COLOR_BACKGROUND="black", COLOR_FOREGROUND="white")

    cmap_min, cmap_max = 0, 1000
    pygmt.makecpt(cmap="inferno", series=[cmap_min, cmap_max], reverse=True)

    q3_mask = df.quality_score == "Q3"
    spatial_plot.plot_sites(
        df.loc[q3_mask],
        # label="Q3",
        style="t0.25c",
        fill=df.loc[q3_mask, "vs30"].values,
        cmap=True,
        pen="0.5p,black",
    )

    q2_mask = df.quality_score == "Q2"
    spatial_plot.plot_sites(
        df.loc[q2_mask],
        # label="Q2",
        style="d0.25c",
        fill=df.loc[q2_mask, "vs30"].values,
        cmap=True,
        pen="0.5p,black",
    )

    q1_mask = df.quality_score == "Q1"
    spatial_plot.plot_sites(
        df.loc[q1_mask],
        # label="Q1",
        style="c0.25c",
        fill=df.loc[q1_mask, "vs30"].values,
        cmap=True,
        pen="0.4p,black",
    )

    # Add legend
    legend_spec = io.StringIO()
    legend_spec.write("H 12p,Helvetica-Bold Site Quality\n")
    legend_spec.write("D 0.1i 1p\n")
    legend_spec.write("S 0.1i c 0.25c black 0.4p,black 0.4i Q1 (N=115)\n")
    legend_spec.write("S 0.1i d 0.25c black 0.4p,black 0.4i Q2 (N=40)\n")
    legend_spec.write("S 0.1i t 0.25c black 0.5p,black 0.4i Q3 (N=717)\n")
    spatial_plot.fig.legend(
        spec=legend_spec, position="JTR+jTR+o0.2c", box="+gwhite+p1p"
    )

    # Canterbury inset
    with spatial_plot.fig.inset(
        position="jBR+o0.2c",
        box="+gwhite+p1p",
        projection=inset_projection,
        region=vs30.constants.CANTERBURY_BOUNDING_BOX,
    ):
        try:
            plotting.gen_region_fig(
                projection=inset_projection,
                region=vs30.constants.CANTERBURY_BOUNDING_BOX,
                plot_kwargs=vs30.plotting.spatial.SpatialPlot.DEFAULT_PLT_KWARGS
                | {"highway_pen_width": 1.0},
                config_options=vs30.plotting.spatial.SpatialPlot.DEFAULT_CONFIG_OPTIONS,
                fig=spatial_plot.fig,
            )

            pygmt.makecpt(cmap="inferno", series=[cmap_min, cmap_max], reverse=True)
            spatial_plot.plot_sites(
                df.loc[q3_mask],
                style="t0.25c",
                fill=df.loc[q3_mask, "vs30"].values,
                cmap=True,
                pen="0.5p,black",
            )
            spatial_plot.plot_sites(
                df.loc[q2_mask],
                style="d0.25c",
                fill=df.loc[q2_mask, "vs30"].values,
                cmap=True,
                pen="0.5p,black",
            )
            spatial_plot.plot_sites(
                df.loc[q1_mask],
                style="c0.25c",
                fill=df.loc[q1_mask, "vs30"].values,
                cmap=True,
                pen="0.4p,black",
            )
        except Exception as e:
            print(f"Inset error: {e}")
            raise

    x_min, x_max, y_min, y_max = vs30.constants.CANTERBURY_BOUNDING_BOX
    spatial_plot.fig.plot(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        pen="1.5p,red",
    )

    # Wellington inset
    with spatial_plot.fig.inset(
        position="jTL+o0.2c",
        box="+gwhite+p1p",
        projection=inset_projection,
        region=vs30.constants.WELLINGTON_BOUNDING_BOX,
    ):
        try:
            plotting.gen_region_fig(
                projection=inset_projection,
                region=vs30.constants.WELLINGTON_BOUNDING_BOX,
                plot_kwargs=vs30.plotting.spatial.SpatialPlot.DEFAULT_PLT_KWARGS
                | {"highway_pen_width": 1.0},
                config_options=vs30.plotting.spatial.SpatialPlot.DEFAULT_CONFIG_OPTIONS,
                fig=spatial_plot.fig,
            )

            pygmt.makecpt(cmap="inferno", series=[cmap_min, cmap_max], reverse=True)
            spatial_plot.plot_sites(
                df.loc[q3_mask],
                style="t0.25c",
                fill=df.loc[q3_mask, "vs30"].values,
                cmap=True,
                pen="0.5p,black",
            )
            spatial_plot.plot_sites(
                df.loc[q2_mask],
                style="d0.25c",
                fill=df.loc[q2_mask, "vs30"].values,
                cmap=True,
                pen="0.5p,black",
            )
            spatial_plot.plot_sites(
                df.loc[q1_mask],
                style="c0.25c",
                fill=df.loc[q1_mask, "vs30"].values,
                cmap=True,
                pen="0.4p,black",
            )
        except Exception as e:
            print(f"Inset error: {e}")
            raise

    x_min, x_max, y_min, y_max = vs30.constants.WELLINGTON_BOUNDING_BOX
    spatial_plot.fig.plot(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        pen="1.5p,red",
    )

    spatial_plot.fig.colorbar(
        position=spatial_plot.CB_POSITION, frame=["x+lVs30", "y+lm/s"]
    )
    spatial_plot.save(output_dir / f"site_map.{vs30.constants.FIG_FORMAT}")


@app.command("gen-vs30-map")
def gen_vs30_map(
    dataset_ffp: Path,
    output_dir: Path,
    region_key: str,
    grid_spacing: str = "250e/250e",
    show_highways: bool = False,
    show_cities: bool = True,
    show_towns: bool = False,
):
    logger = mlt.utils.setup_logging()
    region = vs30.constants.REGION_MAPPING[region_key]

    # Load vs30 values
    with xr.open_dataset(dataset_ffp) as ds:
        vs30_da = ds["vs30"]

    nan_mask = vs30_da.isnull().values
    mesh_x, mesh_y = np.meshgrid(vs30_da.coords["x"].values, vs30_da.coords["y"].values)
    vs30_df = pd.DataFrame(
        {
            "nztm_x": mesh_x[~nan_mask],
            "nztm_y": mesh_y[~nan_mask],
            "vs30": vs30_da.values[~nan_mask],
        }
    )

    latlon_coords = coordinates.nztm_to_wgs_depth(vs30_df[["nztm_y", "nztm_x"]].values)
    vs30_df["lat"], vs30_df["lon"] = latlon_coords[:, 0], latlon_coords[:, 1]

    if region_key != "nz":
        vs30_df = vs30_df.loc[
            (vs30_df["lon"] >= region[0])
            & (vs30_df["lon"] <= region[1])
            & (vs30_df["lat"] >= region[2])
            & (vs30_df["lat"] <= region[3])
        ]

    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=False,
        plot_highways=False,
        region=region,
        projection="M8.5c",
    )

    logger.info("Plotting Vs30 values...")
    start = time.time()
    spatial_plot.plot_vs30_values(vs30_df, region=region, grid_spacing=grid_spacing)
    logger.info(f"Took: {time.time() - start} to plot Vs30 values")

    if show_highways:
        spatial_plot.add_highways()
    if show_cities:
        spatial_plot.add_city_labels()
    if show_towns:
        spatial_plot.add_town_labels()

    spatial_plot.save(output_dir / f"vs30_{region_key}.{vs30.constants.FIG_FORMAT}")


@app.command("gen-residual-map")
def gen_residual_map(
    dataset_ffp: Path,
    output_dir: Path,
    region_key: str,
    grid_spacing: str = "250e/250e",
    show_highways: bool = False,
    show_cities: bool = True,
    show_towns: bool = False,
):
    logger = mlt.utils.setup_logging()
    region = vs30.constants.REGION_MAPPING[region_key]

    # Configure residual key and colorbar label
    residual_key = "foster_original_vs30_ln_res"
    cb_label = "ln(Foster) - ln(ML)"

    # Load residual values
    with xr.open_dataset(dataset_ffp) as ds:
        residual_da = ds[residual_key]

    nan_mask = residual_da.isnull().values
    mesh_x, mesh_y = np.meshgrid(
        residual_da.coords["x"].values, residual_da.coords["y"].values
    )
    residual_df = pd.DataFrame(
        {
            "nztm_x": mesh_x[~nan_mask],
            "nztm_y": mesh_y[~nan_mask],
            "ln_residual": residual_da.values[~nan_mask],
        }
    )

    latlon_coords = coordinates.nztm_to_wgs_depth(
        residual_df[["nztm_y", "nztm_x"]].values
    )
    residual_df["lat"], residual_df["lon"] = latlon_coords[:, 0], latlon_coords[:, 1]

    if region_key != "nz":
        residual_df = residual_df.loc[
            (residual_df["lon"] >= region[0])
            & (residual_df["lon"] <= region[1])
            & (residual_df["lat"] >= region[2])
            & (residual_df["lat"] <= region[3])
        ]

    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=False,
        plot_highways=False,
        region=region,
        projection="M8.5c",
    )

    logger.info("Plotting residual values...")
    start = time.time()
    spatial_plot.plot_ratio(
        residual_df,
        cmap_limits=(-1.0, 1.0, 2.0 / 16),
        region=region,
        cb_label=cb_label,
        grid_spacing=grid_spacing,
    )
    logger.info(f"Took: {time.time() - start} to plot residual values")

    if show_highways:
        spatial_plot.add_highways()
    if show_cities:
        spatial_plot.add_city_labels()
    if show_towns:
        spatial_plot.add_town_labels()

    spatial_plot.save(
        output_dir / f"{residual_key}_{region_key}.{vs30.constants.FIG_FORMAT}"
    )


@app.command("input-variable-map")
def input_variable_map(
    dataset_ffp: Path,
    variable: vs30.constants.InputVariable,
    region_key: str,
    output_dir: Path,
):
    logger = mlt.utils.setup_logging()
    region = vs30.constants.REGION_MAPPING[region_key]

    if variable not in vs30.constants.InputVariable:
        vs30.utils.raise_log(
            ValueError, f"Variable {variable} is not a valid input variable."
        )

    # Load input variable values
    with xr.open_dataset(dataset_ffp) as ds:
        variable_da = ds[variable]
    nan_mask = variable_da.isnull().values
    mesh_x, mesh_y = np.meshgrid(
        variable_da.coords["x"].values, variable_da.coords["y"].values
    )
    variable_df = pd.DataFrame(
        {
            "nztm_x": mesh_x[~nan_mask],
            "nztm_y": mesh_y[~nan_mask],
            variable: variable_da.values[~nan_mask],
        }
    )

    # Convert to lat/lon and filter to region
    latlon_coords = coordinates.nztm_to_wgs_depth(
        variable_df[["nztm_y", "nztm_x"]].values
    )
    variable_df["lat"], variable_df["lon"] = latlon_coords[:, 0], latlon_coords[:, 1]
    if region_key != "nz":
        variable_df = variable_df.loc[
            (variable_df["lon"] >= region[0])
            & (variable_df["lon"] <= region[1])
            & (variable_df["lat"] >= region[2])
            & (variable_df["lat"] <= region[3])
        ]

    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=False,
        plot_highways=False,
        region=region,
        projection="M8.5c",
    )

    logger.info(f"Plotting {variable} values...")
    start = time.time()
    spatial_plot.plot_input_variable_values(variable_df, variable, region=region)
    logger.info(f"Took: {time.time() - start} to plot {variable} values")

    spatial_plot.save(
        output_dir / f"input_var_{variable}_{region_key}.{vs30.constants.FIG_FORMAT}"
    )


@app.command("input-variable-kde-distribution")
def input_variable_kde_distribution(
    dataset_ffp: Path,
    nz_dataset_ffp: Path,
    variable: vs30.constants.InputVariable,
    output_dir: Path,
    show_legend: bool = True,
    show_rug: bool = False,
):
    logger = mlt.utils.setup_logging()

    _fig_settings(logger)

    # Load input variable values
    dataset_df = pd.read_parquet(dataset_ffp)
    with xr.open_dataset(nz_dataset_ffp) as ds:
        variable_da = ds[variable]

    min_val, max_val = vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[variable]

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        np.clip(dataset_df[variable].values, min_val, max_val),
        fill=True,
        ax=ax,
        color="tab:blue",
        label="NZ Site Database",
    )
    sns.kdeplot(
        np.clip(variable_da.values.flatten(), min_val, max_val),
        fill=True,
        ax=ax,
        color="tab:red",
        label="NZ Input Grid",
    )
    if show_rug:
        sns.rugplot(
            np.clip(dataset_df[variable].values, min_val, max_val),
            ax=ax,
            color="tab:blue",
            height=0.02,
        )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel(vs30.constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING[variable])
    ax.set_ylabel("Density")
    ax.set_xlim(min_val, max_val)
    ax.yaxis.set_ticklabels([])
    if show_legend:
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / f"input_var_kde_{variable}.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("foster-dataset-comparison")
def foster_dataset_comparison(
    dataset_ffp: Path,
    foster_dataset_ffp: Path,
    output_dir: Path,
):
    dataset_df = pd.read_parquet(dataset_ffp)
    dataset_df["source"] = "nz_all"

    qual_df = dataset_df[dataset_df["quality_score"] != "Q3"].copy()
    qual_df["source"] = "nz_qual"

    foster_df = pd.read_parquet(foster_dataset_ffp)
    foster_df["source"] = "foster"

    comb_df = pd.concat(
        [
            dataset_df[["vs30", "nzenvds_slope_deg", "nz_geology_age_mid", "source"]],
            foster_df[["vs30", "nzenvds_slope_deg", "nz_geology_age_mid", "source"]],
            qual_df[["vs30", "nzenvds_slope_deg", "nz_geology_age_mid", "source"]],
        ],
        ignore_index=True,
    )
    comb_df["vs30"] = comb_df["vs30"].clip(0, 1600)
    comb_df["nzenvds_slope_deg"] = comb_df["nzenvds_slope_deg"].clip(0, 20)
    comb_df["nz_geology_age_mid"] = comb_df["nz_geology_age_mid"].clip(0.1, None)

    color_map = {
        "nz_all": "tab:red",
        "nz_qual": "tab:orange",
        "foster": "tab:blue",
    }

    ### Vs30
    fig, ax = plt.subplots(
        figsize=(vs30.constants.FIG_SIZE[0] * 2, vs30.constants.FIG_SIZE[1])
    )

    sns.kdeplot(
        comb_df,
        x="vs30",
        hue="source",
        fill=True,
        ax=ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Vs30 (m/s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1600)
    ax.yaxis.set_ticklabels([])

    # Legend
    label_map = {
        "nz_all": f"NZ - All (N = {len(dataset_df)})",
        "nz_qual": f"NZ Q1 & Q2 (N = {len(qual_df)})",
        "foster": f"Foster et al. (N = {len(foster_df)})",
    }
    handles = [
        mpatches.Patch(color=color_map[k], label=label_map[k])
        for k in ["nz_all", "nz_qual", "foster"]
    ]

    ax.legend(
        handles=handles,
        # title="Dataset",
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / f"foster_comparison.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)

    ### Slope
    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        comb_df,
        x="nzenvds_slope_deg",
        hue="source",
        fill=True,
        ax=ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
    )
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Slope (degrees)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, comb_df["nzenvds_slope_deg"].max())
    ax.yaxis.set_ticklabels([])

    fig.tight_layout()
    fig.savefig(
        output_dir / f"foster_comparison_slope.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)

    ### Geological Age
    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        comb_df,
        x="nz_geology_age_mid",
        hue="source",
        fill=True,
        ax=ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
        log_scale=(True, False),
    )
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Geological Age")
    ax.set_ylabel("Density")
    ax.set_xlim(0.1, comb_df["nz_geology_age_mid"].max())
    ax.yaxis.set_ticklabels([])

    fig.tight_layout()
    fig.savefig(
        output_dir / f"foster_comparison_geological_age.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("geyin-dataset-comparison")
def geyin_dataset_comparison(
    dataset_ffp: Path,
    geyin_dataset_ffp: Path,
    output_dir: Path,
):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    dataset_df = pd.read_parquet(dataset_ffp)
    dataset_df["slope"] = dataset_df["nzenvds_slope_deg"]
    dataset_df["source"] = "nz_all"

    qual_df = dataset_df[dataset_df["quality_score"] != "Q3"].copy()
    qual_df["source"] = "nz_qual"

    geyin_dataset_df = pd.read_parquet(geyin_dataset_ffp)
    geyin_dataset_df["source"] = "geyin"
    geyin_dataset_df["slope"] = geyin_dataset_df["topographic_slope"]

    comb_df = pd.concat(
        [
            dataset_df[["vs30", "slope", "source"]],
            geyin_dataset_df[["vs30", "slope", "source"]],
            qual_df[["vs30", "slope", "source"]],
        ],
        ignore_index=True,
    )
    comb_df["vs30"] = comb_df["vs30"].clip(0, 1600)
    comb_df["slope"] = comb_df["slope"].clip(0, 20)

    color_map = {
        "nz_all": "tab:red",
        "nz_qual": "tab:orange",
        "geyin": "tab:blue",
    }

    ### Vs30
    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        comb_df,
        x="vs30",
        hue="source",
        fill=True,
        ax=ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Vs30 (m/s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1600)
    ax.yaxis.set_ticklabels([])

    # Legend
    label_map = {
        "nz_all": f"NZ - All (N = {len(dataset_df)})",
        "nz_qual": f"NZ Q1 & Q2 (N = {len(qual_df)})",
        "geyin": f"Geyin et al. (N = {len(geyin_dataset_df)})",
    }
    handles = [
        mpatches.Patch(color=color_map[k], label=label_map[k])
        for k in ["nz_all", "nz_qual", "geyin"]
    ]

    ax.legend(
        handles=handles,
        # title="Dataset",
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / f"geyin_comparison.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)

    ### Slope
    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        comb_df,
        x="slope",
        hue="source",
        fill=True,
        ax=ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
    )
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Slope (degrees)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, comb_df["slope"].max())
    ax.yaxis.set_ticklabels([])

    fig.tight_layout()
    fig.savefig(
        output_dir / f"geyin_comparison_slope.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("create-nz-vs30-histogram")
def create_nz_vs30_histogram(
    full_model_results_dir: Path,
    foster_tif: Path,
    population_density_ffp: Path,
    output_dir: Path,
):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    # Load vs30 values
    with xr.open_dataset(full_model_results_dir / "nz_vs30_results.nc") as ds:
        vs30_da = ds["vs30"]
        y, x = vs30_da.y.values, vs30_da.x.values
        nan_mask = vs30_da.isnull().values

    grid_x, grid_y = np.meshgrid(x, y)
    coords = np.column_stack((grid_x[~nan_mask], grid_y[~nan_mask]))
    resolution = grid_x[0, 1] - grid_x[0, 0]

    estimates_df = gpd.GeoDataFrame(
        {
            "nztm_x": coords[:, 0],
            "nztm_y": coords[:, 1],
            "pred_vs30": vs30_da.values[~nan_mask],
        },
        geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
        crs=vs30.constants.NZTM2000_EPSG
    )

    logger.info("Extracting original Foster et al. estimates...")
    with rasterio.open(foster_tif) as ds:
        assert ds.crs.to_epsg() == vs30.constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"
        foster_original_data = ds.read(1, masked=True).filled(np.nan)

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])

        estimates_df["pred_vs30_foster"] = foster_original_data[rows, cols]

    # Drop missing values
    assert estimates_df["pred_vs30"].isna().sum() == 0 
    nan_mask = estimates_df["pred_vs30_foster"].isna()
    if nan_mask.sum() > 0:
        logger.warning(f"Dropping {nan_mask.sum()} points with missing Foster estimates")
        estimates_df = estimates_df.loc[~nan_mask]

    # Load population density data
    population_gdf = gpd.read_file(population_density_ffp)
    assert population_gdf.area.unique().shape[0] == 1, "Population grid cells have varying areas, which is not supported"
    population_cell_area = population_gdf.area.unique()[0]

    # Compute intersection of population grid with vs30 estimate points
    intersection_df = estimates_df.copy()
    intersection_df.geometry = estimates_df.buffer(resolution / 2, cap_style="square")
    intersection_df["vs30_point_index"] = intersection_df.index 
    intersection_df = gpd.overlay(intersection_df, population_gdf, how="intersection")
    
    # Weight vs30 estimates by population proportion
    intersection_df["population_proportion"] = intersection_df["PopEst2023"] * (intersection_df.geometry.area / population_cell_area)
    estimates_df["population_count"] = intersection_df.groupby("vs30_point_index")["population_proportion"].sum()
    estimates_df["population_weight"] = estimates_df["population_count"] / estimates_df["population_count"].sum()
    estimates_df["population_weight"] = estimates_df["population_weight"].fillna(0)
    
    bins = list(np.linspace(100, 1500, 40))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=vs30.constants.FIG_SIZE)

    # Unweighted histogram
    sns.histplot(
        estimates_df,
        x="pred_vs30_foster",
        bins=bins,
        ax=ax1,
        color="tab:red",
        label="Foster et al.",
        stat="density",
        edgecolor="black",
        alpha=0.5,
    )
    sns.histplot(
        estimates_df,
        x="pred_vs30",
        bins=bins,
        ax=ax1,
        color="tab:blue",
        label="ML Model",
        stat="density",
        edgecolor="black",
        alpha=0.5,
    )

    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xlabel("Vs30 (m/s)")
    ax1.set_ylabel("Density")
    ax1.set_xlim(min(bins), max(bins))
    ax1.yaxis.set_ticklabels([])
    ax1.legend()

    # Population-weighted histogram
    sns.histplot(
        estimates_df,
        x="pred_vs30_foster",
        bins=bins,
        ax=ax2,
        color="tab:red",
        label="Foster et al.",
        stat="density",
        weights="population_weight",
        edgecolor="black",
        alpha=0.5,
    )
    sns.histplot(
        estimates_df,
        x="pred_vs30",
        bins=bins,
        ax=ax2,
        color="tab:blue",
        label="ML Model",
        stat="density",
        weights="population_weight",
        edgecolor="black",
        alpha=0.5,
    )
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlabel("Vs30 (m/s)")
    ax2.set_ylabel("Population-Weighted Density")
    ax2.set_xlim(min(bins), max(bins))
    ax2.yaxis.set_ticklabels([])

    # Ensure same y-axis limits for both subplots
    ax1.set_ylim(0, max(ax1.get_ylim()[1], ax2.get_ylim()[1]))

    fig.tight_layout()
    fig.savefig(
        output_dir / f"nz_vs30_histogram.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("gen-vs30-hist")
def gen_vs30_hist(dataset_ffp: Path, output_dir: Path):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    dataset_df = pd.read_parquet(dataset_ffp)

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE, dpi=vs30.constants.FIG_DPI)

    sns.histplot(
        dataset_df,
        x="vs30",
        bins=vs30.constants.DENSE_VS30_BINS[:-2],
        ax=ax,
        hue="quality_score",
        palette=vs30.constants.QUALITY_SCORE_COLORS,
        # edgecolor="black",
        multiple="stack",
        hue_order=["Q3", "Q2", "Q1"],
    )
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Vs30 (m/s)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1600)
    ax.get_legend().set_title("Quality Score")

    fig.tight_layout()
    fig.savefig(output_dir / f"vs30_hist.{vs30.constants.FIG_FORMAT}")
    plt.close(fig)


def _make_scatter_proxy(scatter):
    """Creates a proxy artist for a scatter plot to be used in legends."""
    return mlines.Line2D(
        [],
        [],
        color=scatter.get_facecolors()[0],
        marker=scatter.get_paths()[0],
        linestyle="None",
        markersize=np.sqrt(scatter.get_sizes()[0]),
    )


@app.command("gen-residual-scatter-plot")
def gen_residual_scatter_plot(
    results_ffp: Path,
    output_dir: Path,
    is_foster: bool = False,
    full_model_dir: Path = None,
):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    results_df = pd.read_parquet(results_ffp)
    test_results_df = (
        pd.read_parquet(full_model_dir / "test_results.parquet")
        if full_model_dir
        else None
    )

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    test_p_cols = []
    for i, (k, color) in enumerate(vs30.constants.QUALITY_SCORE_COLORS.items()):
        mask = results_df["quality_score"] == k
        ax.scatter(
            results_df.loc[mask, "vs30"],
            results_df.loc[mask, "ln_residual"],
            label=rf"{k} (N={mask.sum()})",
            # rf"$\mu$={results_df.loc[mask, 'ln_residual'].mean():.3f}, "
            # rf"$\sigma$={results_df.loc[mask, 'ln_residual'].std():.3f})",
            # alpha=0.5,
            color=color,
            zorder=10 - i,
            s=vs30.constants.QUALITY_SCORE_MARKER_SIZE[k] * 0.25,
            marker=vs30.constants.QUALITY_SCORE_MARKERS[k],
        )
        if test_results_df is not None:
            test_mask = test_results_df["quality_score"] == k
            cur_p_col = ax.scatter(
                test_results_df.loc[test_mask, "vs30"],
                test_results_df.loc[test_mask, "ln_residual"],
                color="red",
                zorder=10 - i,
                s=vs30.constants.QUALITY_SCORE_MARKER_SIZE[k] * 0.25,
                marker=vs30.constants.QUALITY_SCORE_MARKERS[k],
            )
            test_p_cols.append(cur_p_col)

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel(r"$Vs30$ (m/s)")
    ax.set_ylabel(r"Residual, $\ln(\text{True}) - \ln(\text{Predicted})$")
    ax.set_ylim(-1.75, 1.75)

    if test_results_df is not None:
        test_legend = ax.legend(
            handles=[
                (
                    _make_scatter_proxy(test_p_cols[0]),
                    _make_scatter_proxy(test_p_cols[1]),
                    _make_scatter_proxy(test_p_cols[2]),
                )
            ],
            labels=[
                # f"Test Set (N={(test_results_df.quality_score == "Q1").sum()}/{(test_results_df.quality_score == "Q2").sum()}/{(test_results_df.quality_score == "Q3").sum()})"
                "Test Set"
            ],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
            handlelength=2.5,
        )
        ax.add_artist(test_legend)

    ax.legend(title="Quality Score")

    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"{'foster' if is_foster else 'model'}_residual_scatter.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("gen-residual-histogram")
def gen_residual_histogram(
    results_ffp: Path,
    foster_results_ffp: Path,
    output_dir: Path,
    quality_score: str = None,
    hide_y_label: bool = False,
    show_legend: bool = True,
    y_max_limit: float = None,
):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    results_df = pd.read_parquet(results_ffp)
    foster_results_df = pd.read_parquet(foster_results_ffp)

    if quality_score is not None:
        results_df = results_df[results_df["quality_score"] == quality_score]
        foster_results_df = foster_results_df[
            foster_results_df["quality_score"] == quality_score
        ]

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        results_df["ln_residual"],
        fill=True,
        ax=ax,
        color="tab:blue",
        # label=rf"Model - $\mu$ = {results_df['ln_residual'].mean():.3f}, $\sigma$ = {results_df['ln_residual'].std():.3f}",
        label="Model",
        # cut=1.0,
        legend=False,
    )

    sns.kdeplot(
        foster_results_df["ln_residual"],
        fill=True,
        ax=ax,
        color="tab:orange",
        label="Foster",
        # cut=1.0,
        legend=False,
        # label=rf"Foster - $\mu$ = {foster_results_df['ln_residual'].mean():.3f}, $\sigma$ = {foster_results_df['ln_residual'].std():.3f}",
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel(r"Residual, $\ln(\text{True}) - \ln(\text{Predicted})$")
    ax.set_xlim(-1.75, 1.75)
    if y_max_limit is not None:
        ax.set_ylim(0, y_max_limit)
    ax.yaxis.set_ticklabels([])
    ax.set_ylabel("" if hide_y_label else "Density")
    if show_legend:
        ax.legend()

    if quality_score is not None:
        ax.text(
            0.025,
            0.98,
            f"{quality_score} (N={len(results_df)})",
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            fontweight="bold",
        )

    fig.tight_layout()
    filename = (
        f"residual_histogram_{quality_score}"
        if quality_score is not None
        else "residual_histogram"
    )
    fig.savefig(
        output_dir / f"{filename}.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("gen-one-to-one-plot")
def gen_one_to_one_plot(
    results_ffp: Path,
    output_dir: Path,
    is_foster: bool = False,
    show_legend: bool = True,
    full_model_dir: Path = None,
):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    results_df = pd.read_parquet(results_ffp)
    test_results_df = (
        pd.read_parquet(full_model_dir / "test_results.parquet")
        if full_model_dir
        else None
    )

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE, dpi=vs30.constants.FIG_DPI)

    test_p_cols = []
    for i, (k, color) in enumerate(vs30.constants.QUALITY_SCORE_COLORS.items()):
        mask = results_df["quality_score"] == k
        ax.scatter(
            results_df.loc[mask, "vs30"],
            results_df.loc[mask, "pred_vs30"],
            label=rf"{k} (N={mask.sum()})",
            # label=rf"{k}",
            color=color,
            zorder=10 - i,
            s=vs30.constants.QUALITY_SCORE_MARKER_SIZE[k] * 0.25,
            marker=vs30.constants.QUALITY_SCORE_MARKERS[k],
        )
        if test_results_df is not None:
            test_mask = test_results_df["quality_score"] == k
            cur_p_col = ax.scatter(
                test_results_df.loc[test_mask, "vs30"],
                test_results_df.loc[test_mask, "pred_vs30"],
                color="red",
                zorder=10 - i,
                s=vs30.constants.QUALITY_SCORE_MARKER_SIZE[k] * 0.25,
                marker=vs30.constants.QUALITY_SCORE_MARKERS[k],
            )
            test_p_cols.append(cur_p_col)

    ax.plot(
        [100, 1600],
        [100, 1600],
        "k",
    )

    ax.set_xlim(100, 1600)
    ax.set_ylim(100, 1600)
    ax.set_xlabel(r"$V_{S30}$ (m/s)", labelpad=-3)
    ax.set_ylabel(r"Predicted $V_{S30}$ (m/s)", labelpad=-5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(
        0.025,
        0.98,
        "Foster Model" if is_foster else "ML Model",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontweight="bold",
    )

    if test_results_df is not None:
        test_legend = ax.legend(
            handles=[
                (
                    _make_scatter_proxy(test_p_cols[0]),
                    _make_scatter_proxy(test_p_cols[1]),
                    _make_scatter_proxy(test_p_cols[2]),
                )
            ],
            labels=[
                # f"Test Set (N={(test_results_df.quality_score == "Q1").sum()}/{(test_results_df.quality_score == "Q2").sum()}/{(test_results_df.quality_score == "Q3").sum()})"
                "Test Set"
            ],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
            handlelength=2.5,
            loc="lower left",
            bbox_to_anchor=(0.075, 0.002),
        )
        ax.add_artist(test_legend)

    if show_legend:
        ax.legend()

    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    # fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08)
    filename = f"{'foster' if is_foster else 'model'}_one_to_one_plot"
    fig.savefig(
        output_dir / f"{filename}.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("gen-PIT-plot")
def gen_PIT_plot(results_ffp: Path, output_dir: Path):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    results_df = pd.read_parquet(results_ffp)

    std_res = (
        np.log(results_df["pred_vs30"]) - np.log(results_df["vs30"])
    ) / results_df["pred_vs30_std"]
    pit_values = norm.cdf(std_res)

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)
    sns.histplot(
        pit_values,
        bins=15,
        ax=ax,
        color="tab:blue",
        edgecolor="black",
        stat="density",
        label="PIT Values",
    )
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.axhline(1.0, color="red", linestyle="--", label="Uniform Distribution Density")
    ax.set_xlabel("PIT Values")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / f"pit_plot.{vs30.constants.FIG_FORMAT}", dpi=vs30.constants.FIG_DPI
    )
    plt.close(fig)


# @app.command("gen-global-feature-importance")
# def gen_global_feature_importance(cv_model_results_dir: Path, output_dir: Path):
#     # TODO: I don't using shap plotting funcationality will work here, if we decide to include, create manually.
#     logger = mlt.utils.setup_logging()
#     _fig_settings(logger)

#     shap_values = pd.read_pickle(cv_model_results_dir / "shap_values.pkl")

#     shap_values.feature_names = [
#         (
#             feat
#             if feat not in vs30.constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING
#             else vs30.constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING[feat]
#         )
#         for feat in shap_values.feature_names
#     ]

#     fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)
#     shap.plots.bar(shap_values, show=False, ax=ax)

#     fig.tight_layout()
#     fig.savefig(output_dir / f"global_feature_importance.{vs30.constants.FIG_FORMAT}", dpi=vs30.constants.FIG_DPI)

#     plt.close(fig)


if __name__ == "__main__":
    app()


############## Archive

# @app.command("input-variable-distribution")
# def input_variable_distribution(
#     dataset_ffp: Path,
#     nz_dataset_ffp: Path,
#     variable: vs30.constants.InputVariable,
#     output_dir: Path,
# ):
#     logger = mlt.utils.setup_logging()

#     _fig_settings(logger)

#     # Load input variable values
#     dataset_df = pd.read_parquet(dataset_ffp)
#     with xr.open_dataset(nz_dataset_ffp) as ds:
#         variable_da = ds[variable]

#     min_val, max_val = vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[variable]
#     bins = np.linspace(min_val, max_val, 25)

#     fig, (ax1, ax2) = mlt.plotting.get_fig_axes(
#         2, 1, 2, ind_figsize=vs30.constants.FIG_SIZE, dpi=vs30.constants.FIG_DPI
#     )

#     hist_values_1, *_ = ax1.hist(
#         np.clip(dataset_df[variable], min_val, max_val),
#         bins=bins,
#         color="tab:blue",
#         edgecolor="black",
#         density=True,
#     )
#     ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
#     ax1.set_ylabel("Frequency")
#     ax1.set_xlim(min_val, max_val)
#     ax1.set_xticklabels([])

#     ax1.text(
#         0.025,
#         0.97,
#         "NZ Site Database",
#         transform=ax1.transAxes,
#         horizontalalignment="left",
#         verticalalignment="top",
#         fontweight="bold",
#     )

#     hist_values_2, *_ = ax2.hist(
#         np.clip(variable_da.values.flatten(), min_val, max_val),
#         bins=bins,
#         color="#911eb4",
#         edgecolor="black",
#         density=True,
#     )
#     ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
#     ax2.set_xlabel(vs30.constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING[variable])
#     ax2.set_ylabel("Frequency")
#     ax2.set_xlim(min_val, max_val)

#     y_max = max(hist_values_1.max(), hist_values_2.max()) * 1.2
#     ax1.set_ylim(0, y_max)
#     ax2.set_ylim(0, y_max)

#     ax2.text(
#         0.025,
#         0.97,
#         "NZ Input Grid",
#         transform=ax2.transAxes,
#         horizontalalignment="left",
#         verticalalignment="top",
#         fontweight="bold",
#     )

#     fig.tight_layout()
#     fig.savefig(
#         output_dir / f"input_var_dist_{variable}.{vs30.constants.FIG_FORMAT}",
#         dpi=vs30.constants.FIG_DPI,
#     )
#     plt.close(fig)
