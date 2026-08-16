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
import matplotlib.ticker as mticker
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from scipy import stats
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
        region=plotting.ProjectedRegion.from_box(
            *vs30.constants.NZ_BOUNDING_BOX, main_projection
        ),
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
                region=plotting.ProjectedRegion.from_box(
                    *vs30.constants.CANTERBURY_BOUNDING_BOX, inset_projection
                ),
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

            spatial_plot.fig.text(
                x=vs30.constants.CANTERBURY_BOUNDING_BOX[1],
                y=vs30.constants.CANTERBURY_BOUNDING_BOX[3],
                text="Canterbury",
                justify="TR",
                offset="-0.05c/-0.075c",
                font=vs30.constants.GMT_FIG_FONT_LABEL,
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
                region=plotting.ProjectedRegion.from_box(
                    *vs30.constants.WELLINGTON_BOUNDING_BOX, inset_projection
                ),
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

            spatial_plot.fig.text(
                x=vs30.constants.WELLINGTON_BOUNDING_BOX[0],
                y=vs30.constants.WELLINGTON_BOUNDING_BOX[3],
                text="Wellington",
                justify="TL",
                offset="0.05c/-0.075c",
                font=vs30.constants.GMT_FIG_FONT_LABEL,
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

@app.command("gen-pred-std-map")
def gen_pred_std_map(
    full_model_dir: Path,
    output_dir: Path,
    region_key: str,
    grid_spacing: str = "250e/250e",
    label: str = None,
):
    logger = mlt.utils.setup_logging()
    region_coords = vs30.constants.REGION_MAPPING[region_key]

    # Load vs30 values
    with xr.open_dataset(full_model_dir / "nz_vs30_results.nc") as ds:
        pred_std_da = ds["lnVs30_std"]

    nan_mask = pred_std_da.isnull().values
    mesh_x, mesh_y = np.meshgrid(pred_std_da.coords["x"].values, pred_std_da.coords["y"].values)
    pred_std_df = pd.DataFrame(
        {
            "nztm_x": mesh_x[~nan_mask],
            "nztm_y": mesh_y[~nan_mask],
            "pred_std_vs30": pred_std_da.values[~nan_mask],
        }
    )

    latlon_coords = coordinates.nztm_to_wgs_depth(pred_std_df[["nztm_y", "nztm_x"]].values)
    pred_std_df["lat"], pred_std_df["lon"] = latlon_coords[:, 0], latlon_coords[:, 1]

    if region_key != "nz":
        pred_std_df = pred_std_df.loc[
            (pred_std_df["lon"] >= region_coords[0])
            & (pred_std_df["lon"] <= region_coords[1])
            & (pred_std_df["lat"] >= region_coords[2])
            & (pred_std_df["lat"] <= region_coords[3])
        ]

    plot_region = plotting.ProjectedRegion.from_box(*region_coords, "M8.5c")
    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=False,
        plot_highways=False,
        region=plot_region,
    )

    logger.info("Plotting Pred Std values...")
    start = time.time()
    spatial_plot.plot_pred_std_vs30(
        pred_std_df,
        region=plot_region,
        grid_spacing=grid_spacing,
        std_limits=(0.0, 0.5)
    )
    logger.info(f"Took: {time.time() - start} to plot Pred Std values")

    spatial_plot.save(
        output_dir / f"pred_std_{region_key}.{vs30.constants.FIG_FORMAT}"
    )


@app.command("gen-vs30-map")
def gen_vs30_map(
    full_model_dir: Path,
    output_dir: Path,
    region_key: str,
    grid_spacing: str = "250e/250e",
    show_highways: bool = False,
    show_cities: bool = True,
    plot_foster: bool = False,
    plot_kriged: bool = False,
    plot_sites: bool = False,
    town: list[str] = None,
    show_towns: bool = False,
    region: list[str] = None,
    show_colorbar: bool = True,
    label: str = None,
    projection: str = "M8.5c",
    azimuth: float = None,
    width: str = "18c",
    vertical: bool = False,
):
    logger = mlt.utils.setup_logging()
    region_coords = vs30.constants.REGION_MAPPING[region_key]
    # With --azimuth, the map is rotated by that angle (degrees clockwise from
    # north) so a long, tilted region like NZ fits one page. `region_coords`'s
    # SW/NE corners then span the tilted rectangle to plot, and the oblique
    # projection is built around its centre; --projection is unused in that
    # case. --vertical lays it up the page instead of across (and so wants a
    # much smaller --width). Otherwise it is a plain north-up box.
    plot_region = (
        plotting.ProjectedRegion.from_rotated_corners(
            *region_coords, azimuth=azimuth, width=width, vertical=vertical
        )
        if azimuth is not None
        else plotting.ProjectedRegion.from_box(*region_coords, projection)
    )

    # Load vs30 values
    with xr.open_dataset(full_model_dir / "nz_vs30_results.nc") as ds:
        if plot_foster:
            vs30_da = ds["foster_original_vs30_mean"]
            fn_prefix = "foster_vs30"
        elif plot_kriged:
            vs30_da = ds["kriged_vs30_mean"]
            fn_prefix = "kriged_model_vs30"
        else:
            vs30_da = ds["vs30"]
            fn_prefix = "model_vs30"

    if plot_foster:
        site_df = pd.read_parquet(
            vs30.constants.BASE_DATA_DIR / "datasets/foster.parquet"
        )
        test_site_df = None
    else:
        site_df = pd.read_parquet(full_model_dir / "train_results.parquet")
        if (test_site_ffp := full_model_dir / "test_results.parquet").exists():
            test_site_df = pd.read_parquet(test_site_ffp)

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

    # Only apply the region filter if the map is not rotated
    if region_key != "nz" and azimuth is None:
        vs30_df = vs30_df.loc[
            (vs30_df["lon"] >= region_coords[0])
            & (vs30_df["lon"] <= region_coords[1])
            & (vs30_df["lat"] >= region_coords[2])
            & (vs30_df["lat"] <= region_coords[3])
        ]

    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=False,
        plot_highways=False,
        region=plot_region,
    )

    logger.info("Plotting Vs30 values...")
    start = time.time()
    spatial_plot.plot_vs30_values(
        vs30_df,
        region=plot_region,
        grid_spacing=grid_spacing,
        show_colorbar=show_colorbar,
        show_colorbar_label=False,
    )
    logger.info(f"Took: {time.time() - start} to plot Vs30 values")

    if show_highways:
        spatial_plot.add_highways()

    if plot_sites:
        logger.info("Plotting site locations...")
        site_style = "d0.075c"
        spatial_plot.plot_sites(
            site_df,
            cmap=True,
            fill=site_df["vs30"].values,
            style=site_style,
            pen="0.1p,black",
        )

        if test_site_df is not None:
            spatial_plot.plot_sites(
                test_site_df,
                cmap=True,
                fill=test_site_df["vs30"].values,
                style=site_style,
                pen="0.1p,red",
            )

    if show_cities:
        spatial_plot.add_city_labels()

    if town is not None or show_towns:
        spatial_plot.add_town_labels(towns=town if town is not None else None)

    if region is not None:
        spatial_plot.add_region_labels(regions=region)

    if label is not None:
        spatial_plot.fig.text(
            x=region_coords[0],
            y=region_coords[3],
            text=label,
            justify="TL",
            offset="0.05c/-0.1c",
            font=vs30.constants.GMT_FIG_FONT_LABEL.replace("Helvetica", "Helvetica-Bold")
        )

    spatial_plot.save(
        output_dir / f"{fn_prefix}_{region_key}.{vs30.constants.FIG_FORMAT}"
    )


@app.command("gen-residual-map")
def gen_residual_map(
    dataset_ffp: Path,
    output_dir: Path,
    region_key: str,
    grid_spacing: str = "250e/250e",
    show_highways: bool = False,
    show_cities: bool = True,
    show_towns: bool = False,
    use_kriged: bool = False,
    show_colorbar: bool = True,
    label: str = None,
):
    logger = mlt.utils.setup_logging()
    region = vs30.constants.REGION_MAPPING[region_key]

    # Configure residual key and colorbar label
    residual_key = (
        "kriged_vs30_foster_original_vs30_ln_res"
        if use_kriged
        else "foster_original_vs30_ln_res"
    )
    cb_label = "ln(F19) - ln(ML)"

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

    plot_region = plotting.ProjectedRegion.from_box(*region, "M8.5c")
    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=False,
        plot_highways=False,
        region=plot_region,
    )

    logger.info("Plotting residual values...")
    start = time.time()
    spatial_plot.plot_ratio(
        residual_df,
        cmap_limits=(-1.0, 1.0, 2.0 / 16),
        region=plot_region,
        cb_label=cb_label,
        grid_spacing=grid_spacing,
        show_colorbar=show_colorbar,
    )
    logger.info(f"Took: {time.time() - start} to plot residual values")

    if show_highways:
        spatial_plot.add_highways()
    if show_cities:
        spatial_plot.add_city_labels()
    if show_towns:
        spatial_plot.add_town_labels()

    if label is not None:
        spatial_plot.fig.text(
            x=region[0],
            y=region[3],
            text=label,
            justify="TL",
            offset="0.05c/-0.1c",
            font=vs30.constants.GMT_FIG_FONT_LABEL.replace("Helvetica", "Helvetica-Bold")
        )

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

    plot_region = plotting.ProjectedRegion.from_box(*region, "M8.5c")
    spatial_plot = vs30.plotting.spatial.SpatialPlot(
        plot_topo=False,
        plot_highways=False,
        region=plot_region,
    )

    logger.info(f"Plotting {variable} values...")
    start = time.time()
    spatial_plot.plot_input_variable_values(
        variable_df, variable, region=plot_region
    )
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
    x_label: str = None,
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
        fill=False,
        ax=ax,
        color="blue",
        label="NZ Site Database",
        linewidth=vs30.constants.FIG_LINEWIDTH,
    )
    sns.kdeplot(
        np.clip(variable_da.values.flatten(), min_val, max_val),
        fill=False,
        ax=ax,
        color="tab:red",
        label="NZ Input Grid",
        linewidth=vs30.constants.FIG_LINEWIDTH,
    )
    if show_rug:
        sns.rugplot(
            np.clip(dataset_df[variable].values, min_val, max_val),
            ax=ax,
            color="tab:blue",
            height=0.02,
        )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel(x_label or vs30.constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING[variable])
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
    vs30_fig, vs30_ax = plt.subplots(
        figsize=(vs30.constants.FIG_SIZE[0] * 2, vs30.constants.FIG_SIZE[1])
    )

    sns.kdeplot(
        comb_df,
        x="vs30",
        hue="source",
        fill=True,
        ax=vs30_ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
    )

    vs30_ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    vs30_ax.set_xlabel("Vs30 (m/s)")
    vs30_ax.set_ylabel("Density")
    vs30_ax.set_xlim(0, 1600)
    vs30_ax.yaxis.set_ticklabels([])

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

    vs30_ax.legend(
        handles=handles,
        # title="Dataset",
    )

    vs30_fig.tight_layout()
    vs30_fig.savefig(
        output_dir / f"foster_comparison.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(vs30_fig)

    ### Slope
    slope_fig, slope_ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        comb_df,
        x="nzenvds_slope_deg",
        hue="source",
        fill=True,
        ax=slope_ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
    )
    slope_ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    slope_ax.set_xlabel("Slope (degrees)")
    slope_ax.set_ylabel("Density")
    slope_ax.set_xlim(0, comb_df["nzenvds_slope_deg"].max())
    slope_ax.yaxis.set_ticklabels([])

    slope_fig.tight_layout()
    slope_fig.savefig(
        output_dir / f"foster_comparison_slope.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(slope_fig)

    ### Geological Age
    age_fig, age_ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    sns.kdeplot(
        comb_df,
        x="nz_geology_age_mid",
        hue="source",
        fill=True,
        ax=age_ax,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=0,
        bw_adjust=0.75,
        legend=False,
        log_scale=(True, False),
    )
    age_ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    age_ax.set_xlabel("Geological Age")
    age_ax.set_ylabel("Density")
    age_ax.set_xlim(0.1, comb_df["nz_geology_age_mid"].max())
    age_ax.yaxis.set_ticklabels([])

    age_fig.tight_layout()
    age_fig.savefig(
        output_dir / f"foster_comparison_geological_age.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(age_fig)


@app.command("combined-dataset-comparison")
def combined_dataset_comparison(
    dataset_ffp: Path,
    foster_dataset_ffp: Path,
    geyin_dataset_ffp: Path,
    output_dir: Path,
):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    dataset_df = pd.read_parquet(dataset_ffp)
    dataset_df["source"] = "nz_all"

    qual_df = dataset_df[dataset_df["quality_score"] != "Q3"].copy()
    qual_df["source"] = "nz_qual"

    foster_df = pd.read_parquet(foster_dataset_ffp)
    foster_df["source"] = "foster"

    geyin_dataset_df = pd.read_parquet(geyin_dataset_ffp)
    geyin_dataset_df["source"] = "geyin"
    # geyin_dataset_df["slope"] = geyin_dataset_df["topographic_slope"]
    geyin_dataset_df["nz_geology_age_mid"] = np.nan

    comb_df = pd.concat(
        [
            dataset_df[["vs30", "nzenvds_topo_roughness", "nz_geology_age_mid", "source"]],
            foster_df[["vs30", "nzenvds_topo_roughness", "nz_geology_age_mid", "source"]],
            qual_df[["vs30", "nzenvds_topo_roughness", "nz_geology_age_mid", "source"]],
        ],
        ignore_index=True,
    )
    comb_df = comb_df.rename(columns={"nzenvds_topo_roughness": "roughness"})
    comb_df = pd.concat(
        [
            comb_df,
            geyin_dataset_df[["vs30", "roughness", "nz_geology_age_mid", "source"]],
        ],
        ignore_index=True,
    )

    comb_df["vs30"] = comb_df["vs30"].clip(0, 1600)
    # comb_df["slope"] = comb_df["slope"].clip(0, 20)
    comb_df["nz_geology_age_mid"] = comb_df["nz_geology_age_mid"].clip(0.1, None)

    fig = plt.figure(figsize=vs30.constants.FIG_SIZE, layout="constrained")
    axd = fig.subplot_mosaic([["A", "A"], ["B", "C"]])

    color_map = {
        "nz_all": "blue",
        "nz_qual": "purple",
        "geyin": "green",
        "foster": "tab:red",
    }

    ax_vs30, ax_roughness, ax_age = axd["A"], axd["B"], axd["C"]

    fill, cut = False, 1
    linewidth = vs30.constants.FIG_LINEWIDTH

    ### Vs30
    sns.kdeplot(
        comb_df,
        x="vs30",
        hue="source",
        fill=fill,
        ax=ax_vs30,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=cut,
        bw_adjust=0.75,
        legend=False,
        linewidth=linewidth,
        log_scale=(False, False),
    )

    ax_vs30.text(
        -0.02,
        0.99,
        "a)",
        transform=ax_vs30.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontweight="bold",
    )

    ax_vs30.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax_vs30.set_xlabel("Vs30 (m/s)")
    ax_vs30.set_ylabel("Density")
    ax_vs30.set_xlim(0, 1600)
    ax_vs30.yaxis.set_ticklabels([])

    # Legend
    label_map = {
        "nz_all": f"NZ - All (N = {len(dataset_df)})",
        "nz_qual": f"NZ Q1 & Q2 (N = {len(qual_df)})",
        "foster": f"Foster et al. (N = {len(foster_df)})",
        "geyin": f"Geyin et al. (N = {len(geyin_dataset_df)})",
    }
    handles = [
        mpatches.Patch(color=color_map[k], label=label_map[k])
        for k in ["nz_all", "nz_qual", "foster", "geyin"]
    ]

    ax_vs30.legend(
        handles=handles,
    )

    ### Slope
    sns.kdeplot(
        comb_df,
        x="roughness",
        hue="source",
        fill=fill,
        ax=ax_roughness,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=cut,
        bw_adjust=0.75,
        legend=False,
        linewidth=linewidth,
    )

    ax_roughness.text(
        -0.04,
        0.99,
        "b)",
        transform=ax_roughness.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontweight="bold",
    )

    ax_roughness.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    # ax_roughness.set_xlabel("Slope (degrees)")
    ax_roughness.set_xlabel("Topographic Roughness")
    ax_roughness.set_ylabel("Density")
    ax_roughness.set_xlim(0, comb_df["roughness"].max())
    ax_roughness.yaxis.set_ticklabels([])

    ### Geological Age
    sns.kdeplot(
        comb_df,
        x="nz_geology_age_mid",
        hue="source",
        fill=fill,
        ax=ax_age,
        palette=color_map,
        hue_order=color_map.keys(),
        common_norm=False,
        cut=cut,
        bw_adjust=0.75,
        legend=False,
        log_scale=(True, False),
        linewidth=linewidth,
    )

    ax_age.text(
        -0.04,
        0.99,
        "c)",
        transform=ax_age.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontweight="bold",
    )

    ax_age.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax_age.set_xlabel("Geological Age (Ma)")
    ax_age.set_ylabel("Density")
    ax_age.set_xlim(0.1, comb_df["nz_geology_age_mid"].max())
    ax_age.yaxis.set_ticklabels([])

    fig.savefig(
        output_dir / f"combined_dataset_comparison.{vs30.constants.FIG_FORMAT}",
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
        crs=vs30.constants.NZTM2000_EPSG,
    )

    logger.info("Extracting original Foster et al. estimates...")
    with rasterio.open(foster_tif) as ds:
        assert (
            ds.crs.to_epsg() == vs30.constants.NZTM2000_EPSG
        ), "Dataset CRS is not NZTM"
        foster_original_data = ds.read(1, masked=True).filled(np.nan)

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])

        estimates_df["pred_vs30_foster"] = foster_original_data[rows, cols]

    # Drop missing values
    assert estimates_df["pred_vs30"].isna().sum() == 0
    nan_mask = estimates_df["pred_vs30_foster"].isna()
    if nan_mask.sum() > 0:
        logger.warning(
            f"Dropping {nan_mask.sum()} points with missing Foster estimates"
        )
        estimates_df = estimates_df.loc[~nan_mask]

    # Load population density data
    population_gdf = gpd.read_file(population_density_ffp)
    assert (
        population_gdf.area.unique().shape[0] == 1
    ), "Population grid cells have varying areas, which is not supported"
    population_cell_area = population_gdf.area.unique()[0]

    # Compute intersection of population grid with vs30 estimate points
    intersection_df = estimates_df.copy()
    intersection_df.geometry = estimates_df.buffer(resolution / 2, cap_style="square")
    intersection_df["vs30_point_index"] = intersection_df.index
    intersection_df = gpd.overlay(
        intersection_df, population_gdf, how="intersection", keep_geom_type=True
    )

    # Weight vs30 estimates by population proportion
    intersection_df["population_proportion"] = intersection_df["PopEst2023"] * (
        intersection_df.geometry.area / population_cell_area
    )
    estimates_df["population_count"] = intersection_df.groupby("vs30_point_index")[
        "population_proportion"
    ].sum()
    estimates_df["population_weight"] = (
        estimates_df["population_count"] / estimates_df["population_count"].sum()
    )
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
        label="Foster et al. (2019)",
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
        label="ML Model (This Study)",
        stat="density",
        edgecolor="black",
        alpha=0.5,
    )

    ax1.text(
        -0.05,
        1.0,
        "a)",
        transform=ax1.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontweight="bold",
    )

    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xlabel("Vs30 (m/s)")
    ax1.set_ylabel("Spatial Density")
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

    ax2.text(
        -0.05,
        1.0,
        "b)",
        transform=ax2.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontweight="bold",
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

    bins = np.arange(100, 1500 + 100, 100)

    sns.histplot(
        dataset_df,
        x="vs30",
        # bins=vs30.constants.DENSE_VS30_BINS[:-2],
        bins=bins,
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
    ax.set_xlim(bins.min(), bins.max())
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


def _make_line_proxy(line):
    """Creates a proxy artist for a line plot to be used in legends."""
    return mlines.Line2D(
        [],
        [],
        color=line.get_color(),
        linestyle=line.get_linestyle(),
        linewidth=1.0,
    )


@app.command("gen-residual-scatter-plot")
def gen_residual_scatter_plot(
    results_ffp: Path,
    output_dir: Path,
    is_foster: bool = False,
    full_model_dir: Path = None,
    hide_x_label: bool = False,
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

    ax.axhline(0, color="black", linewidth=0.75, zorder=5)

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

    vs30.plotting.model_perf_plots.plot_lowess_line(
        results_df["vs30"].values,
        results_df["ln_residual"].values,
        ax,
        "gray",
        vs30.constants.FIG_LINEWIDTH,
        frac=0.7,
        alpha=0.75,
        zorder=15,
        label="LOESS Line",
        quantiles=None,
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--", zorder=0)
    ax.set_xlabel(r"$Vs30$ (m/s)")
    if hide_x_label:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    # ax.set_ylabel(r"Residual, $\ln(\text{True}) - \ln(\text{Predicted})$")
    ax.set_ylabel(r"Residual, $r_i$")
    ax.set_ylim(-1.75, 1.75)
    ax.set_xlim(0, 1550)

    if test_results_df is not None:
        test_legend = ax.legend(
            handles=[
                (
                    _make_scatter_proxy(test_p_cols[0]),
                    _make_scatter_proxy(test_p_cols[1]),
                    _make_scatter_proxy(test_p_cols[2]),
                ),
            ],
            labels=["Test Set"],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
            handlelength=2.5,
        )
        ax.add_artist(test_legend)

    # ax.text(
    #     0.015,
    #     0.975,
    #     "Foster et al. (2019)" if is_foster else "ML Model (This Study)",
    #     transform=ax.transAxes,
    #     horizontalalignment="left",
    #     verticalalignment="top",
    #     fontweight="bold",
    # )

    ax.text(
        0.5,
        0.975,
        "Foster et al. (2019)" if is_foster else "ML Model (This Study)",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        fontweight="bold",
    )

    ax.text(
        0.015,
        0.925,
        "Overprediction",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.text(
        0.015,
        0.075,
        "Underprediction",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    ax.legend(loc="lower right")

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
        cur_results_df = results_df.loc[results_df["quality_score"] == k]
        ax.scatter(
            cur_results_df["vs30"],
            cur_results_df["pred_vs30"],
            label=rf"{k} (N={len(cur_results_df)})",
            # label=rf"{k}",
            color=color,
            zorder=10 - i,
            s=vs30.constants.QUALITY_SCORE_MARKER_SIZE[k] * 0.25,
            marker=vs30.constants.QUALITY_SCORE_MARKERS[k],
        )
        if not is_foster:
            ax.errorbar(
                cur_results_df["vs30"],
                cur_results_df["pred_vs30"],
                yerr=np.stack(
                    (
                        cur_results_df["pred_vs30"]
                        - np.exp(
                            np.log(cur_results_df["pred_vs30"])
                            - cur_results_df["pred_vs30_std"]
                        ),
                        np.exp(
                            np.log(cur_results_df["pred_vs30"])
                            + cur_results_df["pred_vs30_std"]
                        )
                        - cur_results_df["pred_vs30"],
                    )
                ),
                elinewidth=0.5,
                fmt="none",
                ecolor=color,
                alpha=0.25,
                zorder=10 - 3.5 - i,
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
        linewidth=0.75,
        linestyle="--",
    )

    vs30.plotting.model_perf_plots.plot_lowess_line(
        results_df["vs30"].values,
        results_df["pred_vs30"].values,
        ax,
        "gray",
        vs30.constants.FIG_LINEWIDTH,
        frac=0.7,
        alpha=0.75,
        zorder=15,
        label="LOESS Line",
        quantiles=None,
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
        "Foster et al. (2019)" if is_foster else "ML Model (This Study)",
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

    vs30.plotting.model_perf_plots.pit_plot(
        results_df,
        output_dir / f"pit_plot.{vs30.constants.FIG_FORMAT}",
        write_yaml=False,
    )


@app.command("gen-std-res-cdf-plot")
def gen_std_res_cdf_plot(results_ffp: Path, output_dir: Path):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    results_df = pd.read_parquet(results_ffp)

    std_res = (
        np.log(results_df["pred_vs30"]) - np.log(results_df["vs30"])
    ) / results_df["pred_vs30_std"]
    assert std_res.notnull().all(), "Standardized residuals contain NaN values"

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    # Standard Normal CDF and KS Critical Values
    x = np.linspace(std_res.min(), std_res.max(), 1000)
    norm_cdf = stats.norm.cdf(x)
    ks_critical = stats.ksone.ppf(1 - 0.05 / 2, std_res.shape[0])
    ax.plot(
        x,
        norm_cdf + ks_critical,
        color="tab:red",
        linestyle="--",
        linewidth=vs30.constants.FIG_GROUP_LINEWIDTH,
        label="KS Critical Value",
    )
    ax.plot(
        x,
        norm_cdf - ks_critical,
        color="tab:red",
        linestyle="--",
        linewidth=vs30.constants.FIG_GROUP_LINEWIDTH,
    )
    ax.plot(
        x,
        norm_cdf,
        color="tab:red",
        label="Standard Normal",
        linestyle="-",
        linewidth=vs30.constants.FIG_LINEWIDTH,
    )

    # ECDF
    sns.ecdfplot(
        std_res,
        ax=ax,
        color="tab:blue",
        label="ML Model",
        linewidth=vs30.constants.FIG_LINEWIDTH,
    )

    ks_stat = stats.kstest(std_res, "norm").statistic
    ax.text(
        0.025,
        0.975,
        f"KS statistic = {ks_stat:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    ax.text(
        -0.125,
        1.018,
        "a)",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontweight="bold",
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Standardized Residual")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 1)
    ax.legend()

    # fig.tight_layout()
    fig.subplots_adjust(left=0.15, right=0.99, top=0.9825, bottom=0.15)
    fig.savefig(
        output_dir / f"std_res_cdf_plot.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)


@app.command("gen-global-feature-importance")
def gen_global_feature_importance(cv_model_results_dir: Path, output_dir: Path):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    shap_values = pd.read_pickle(cv_model_results_dir / "shap_values.pkl")

    feature_names = [
        vs30.constants.INPUT_VAR_TO_PAPER_NICE_NAME_MAPPING[feat]
        for feat in shap_values.feature_names
    ]

    global_shap_values = np.abs(shap_values.values).mean(axis=0)
    
    sort_ind = np.argsort(global_shap_values)
    feature_names = np.array(feature_names)[sort_ind]
    global_shap_values = global_shap_values[sort_ind]

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    ax.barh(
        feature_names,
        global_shap_values,
        color="tab:blue",
        edgecolor="black",
    )

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"global_feature_importance.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close(fig)

@app.command("gen-feature-trend-plots")
def gen_feature_trend_plots(cv_model_results_dir: Path, output_dir: Path, features: list[str]):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)
    assert len(features) == 4

    shap_values = pd.read_pickle(cv_model_results_dir / "shap_values.pkl")
    results_df = pd.read_parquet(cv_model_results_dir / "val_results.parquet")
    
    run_config = vs30.RunConfig.from_yaml(cv_model_results_dir / "run_config.yaml")
    dataset_df = pd.read_parquet(run_config.dataset_ffp)

    results_df = results_df.join(dataset_df, how="left", validate="1:1", rsuffix="_dataset")

    # feature_names = [
    #     vs30.constants.INPUT_VAR_TO_PAPER_NICE_NAME_MAPPING[feat]
    #     for feat in shap_values.feature_names
    # ]

    x_lim_dict = {
        "nzenvds_topo_roughness": (-5, 150),
        "nz_combined_groundwater_depth": (-0.25, 10.25),
        "nzenvds_topo_normalised_height": (0, 1),
        "nz_geology_age_ln_mid": (0.005, None),
    }

    fig, axs = mlt.plotting.get_fig_axes(4, 2, 2, ind_figsize=vs30.constants.FIG_SIZE)
    labels = ["a)", "b)", "c)", "d)"]


    for i, feat in enumerate(features):
        cur_ax = axs[i]
        cur_ax.axhline(0, color="black", linewidth=0.75, zorder=5)
        shap_ix = shap_values.feature_names.index(feat)

        x_values = results_df[feat].values

        nice_feature_name = vs30.constants.INPUT_VAR_TO_PAPER_NICE_NAME_MAPPING[feat]
        if feat == "nz_geology_age_ln_mid":
            x_values = np.exp(x_values)
            cur_ax.set_xscale("log")
        if feat == "nz_combined_groundwater_depth":
            x_values = np.clip(x_values, 0, 10)

        scatter = cur_ax.scatter(
            x_values,
            shap_values.values[:, shap_ix],
            c=results_df["vs30"].values,
            cmap="viridis",
            vmin=0,
            vmax=1000,
            s=1.0,
            zorder=10,
        )

        if feat in x_lim_dict:
            cur_ax.set_xlim(x_lim_dict[feat])


        cur_ax.grid(linewidth=0.5, alpha=0.5, linestyle="--", zorder=0)
        cur_ax.set_xlabel(nice_feature_name)
        cur_ax.set_ylabel("SHAP Value")

        cur_ax.text(
            0.025,
            0.975,
            labels[i],
            transform=cur_ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            fontweight="bold",
        )   

        if i % 2 == 1:
            cur_ax.set_ylabel("")
            cur_ax.set_yticklabels([])

    y_limits = max(abs(min(ax.get_ylim()[0] for ax in axs)), abs(max(ax.get_ylim()[1] for ax in axs)))
    for ax in axs:
        ax.set_ylim(-y_limits, y_limits)

    # Leave space on the right for the shared colorbar
    fig.subplots_adjust(left=0.09, right=0.88, top=0.99, bottom=0.08, hspace=0.2, wspace=0.05)

    cbar = fig.colorbar(
        scatter,
        ax=axs,
        location="right",
        fraction=0.03,
        pad=0.02,
    )
    cbar.set_label("Vs30 (m/s)")

    fig.savefig(
        output_dir / f"feature_trend_plots.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )
    plt.close()


@app.command("predicted-std-vs30")
def predicted_std_vs30(cv_model_results_dir: Path, output_dir: Path):
    logger = mlt.utils.setup_logging()
    _fig_settings(logger)

    results_df = pd.read_parquet(cv_model_results_dir / "val_results.parquet")

    fig, ax = plt.subplots(figsize=vs30.constants.FIG_SIZE)

    vs30.plotting.model_perf_plots.plot_lowess_line(
        results_df["pred_vs30"].values,
        results_df["pred_vs30_std"].values,
        ax,
        "gray",
        vs30.constants.FIG_LINEWIDTH,
        zorder=15,
        label="LOESS Line",
        quantiles=None,
        frac=0.7,
        alpha=0.75,
    )

    for i, (k, color) in enumerate(vs30.constants.QUALITY_SCORE_COLORS.items()):
        cur_results_df = results_df.loc[results_df["quality_score"] == k]
        ax.scatter(
            cur_results_df["pred_vs30"],
            cur_results_df["pred_vs30_std"],
            label=rf"{k} (N={len(cur_results_df)})",
            # label=rf"{k}",
            color=color,
            zorder=10 - i,
            s=vs30.constants.QUALITY_SCORE_MARKER_SIZE[k] * 0.25,
            marker=vs30.constants.QUALITY_SCORE_MARKERS[k],
        )

    ax.text(
        -0.125,
        1.018,
        "b)",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontweight="bold",
    )

    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel(r"Predicted $V_{S30}$ (m/s)", labelpad=-5)
    ax.set_ylabel(r"Predicted $ln_{V_{S30}}$ Standard Deviation")
    ax.set_xlim(
        results_df["pred_vs30"].values.min() - 10,
        results_df["pred_vs30"].values.max() + 10,
    )

    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(0, 0.7)
    ax.legend(loc="lower right")

    fig.subplots_adjust(left=0.15, right=0.99, top=0.9825, bottom=0.12)
    # fig.tight_layout()
    fig.savefig(
        output_dir / f"predicted_std_vs30.{vs30.constants.FIG_FORMAT}",
        dpi=vs30.constants.FIG_DPI,
    )


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
