import time
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import typer
import pygmt

from qcore import coordinates
from pygmt_helper import plotting
import ml_tools as mlt


import ml_vs30_model as vs30

app = typer.Typer()


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

    spatial_plot.fig.colorbar(position=spatial_plot.CB_POSITION, frame=["x+lVs30", "y+lm/s"])
    spatial_plot.save(output_dir / f"site_map.{vs30.constants.FIG_FORMAT}")


@app.command("gen-vs30-map")
def gen_vs30_map(dataset_ffp: Path, output_dir: Path, region_key: str):
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
    spatial_plot.plot_vs30_values(vs30_df, region=region)
    logger.info(f"Took: {time.time() - start} to plot Vs30 values")

    spatial_plot.save(output_dir / f"vs30_{region_key}.{vs30.constants.FIG_FORMAT}")


@app.command("gen-residual-map")
def gen_residual_map(
    dataset_ffp: Path, output_dir: Path, region_key: str
):
    logger = mlt.utils.setup_logging()
    region = vs30.constants.REGION_MAPPING[region_key]

    # Confi
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
        residual_df, cmap_limits=(-1.0, 1.0, 2.0 / 16), region=region, 
        cb_label=cb_label
    )
    logger.info(f"Took: {time.time() - start} to plot residual values")

    spatial_plot.save(
        output_dir / f"{residual_key}_{region_key}.{vs30.constants.FIG_FORMAT}"
    )


if __name__ == "__main__":
    app()
