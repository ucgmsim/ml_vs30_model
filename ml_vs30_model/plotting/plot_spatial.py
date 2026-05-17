import time
import logging
from pathlib import Path

import xarray as xr
import pygmt
import pandas as pd
import plotly.graph_objects as go
from pygmt_helper import plotting

from .. import constants

logger = logging.getLogger(__name__)


### PyGMT
class SpatialPlot:

    DEFAULT_PLT_KWARGS = {
        "topo_cmap_min": -350,
        "topo_cmap_max": 6000,
        "topo_cmap_inc": 10,
        "highway_pen_width": 0.3,
        "highway_pen_color": "orange",
        "frame_args": ["f"],
        "coastline_pen_width": 0.1,
        "coastline_pen_color": "black",
        "water_color": "white",
    }

    DEFAULT_CONFIG_OPTIONS = dict(
        MAP_FRAME_TYPE="plain",
        # FORMAT_GEO_MAP="ddd.xx",
        MAP_TICK_PEN_PRIMARY="0.5p,black",
        MAP_FRAME_PEN="0.5p,black",
        MAP_FRAME_AXES="wsne",
        FONT_ANNOT_PRIMARY=constants.GMT_FIG_FONT_ANNOT_PRIMARY,
        FONT_LABEL=constants.GMT_FIG_FONT_LABEL,
    )

    CB_POSITION = "JBC+o0c/0.1c+h"

    def __init__(
        self, plot_kwargs: dict = None, config_options: dict = None, **fig_kwargs
    ):
        plot_kwargs = (
            self.DEFAULT_PLT_KWARGS
            if plot_kwargs is None
            else self.DEFAULT_PLT_KWARGS | plot_kwargs
        )
        config_options = (
            self.DEFAULT_CONFIG_OPTIONS
            if config_options is None
            else self.DEFAULT_CONFIG_OPTIONS | config_options
        )

        if "region" not in fig_kwargs:
            fig_kwargs["region"] = constants.NZ_BOUNDING_BOX

        self.fig = plotting.gen_region_fig(
            **fig_kwargs,
            config_options=config_options,
            plot_kwargs=plot_kwargs,
        )

    def plot_sites(self, site_df: pd.DataFrame, **plot_kwargs):
        """Adds the specified sites to the existing figure."""
        plot_kwargs = {"style": "p0.035c", "fill": "black"} | plot_kwargs

        self.fig.plot(
            x=site_df["lon"].values,
            y=site_df["lat"].values,
            **plot_kwargs,
        )

        return self

    def plot_ratio(
        self,
        ratio_df: pd.DataFrame,
        cb_label: str | None = None,
        data_key: str = "ln_residual",
        grid_spacing: str = "250e/250e",
        region: tuple[float, float, float, float] | None = None,
        cmap_limits: tuple[float, float, float] = None,
        **plot_grid_kwargs,
    ):
        """Adds a ratio grid to the existing figure."""
        plot_grid_kwargs = {
            "cb_label": cb_label or data_key,
            "plot_contours": False,
            "reverse_cmap": True,
            "cmap": "polar",
            "cmap_limits": (
                (-0.5, 0.5, 1.0 / 17) if cmap_limits is None else cmap_limits
            ),
            "cmap_limit_colors": ("darkred", "darkblue"),
            "cb_position": self.CB_POSITION
        } | plot_grid_kwargs

        grid = plotting.create_grid(
            ratio_df,
            data_key,
            grid_spacing=grid_spacing,
            region=region,
            interp_method="nearest",
        )

        plotting.plot_grid(self.fig, grid, **plot_grid_kwargs)

        return self

    def plot_vs30_values(
        self,
        vs30_df: pd.DataFrame,
        transparency: float | None = None,
        region: tuple[float, float, float, float] | None = None,
        vs_30_cmap_limits: tuple[float, float] = (0, 1000),
        grid_spacing: str = "250e/250e",
    ):
        start = time.time()
        grid = plotting.create_grid(
            vs30_df,
            "vs30",
            region=region,
            grid_spacing=grid_spacing,
            interp_method="nearest",
        )
        logger.info(f"Took: {time.time() - start} to create grid.")

        # Set the extreme colors
        pygmt.config(COLOR_BACKGROUND="black", COLOR_FOREGROUND="yellow")

        # Plot the grid
        pygmt.makecpt(
            cmap="viridis", series=[vs_30_cmap_limits[0], vs_30_cmap_limits[1]]
        )
        self.fig.grdimage(
            grid,
            cmap=True,
            transparency=transparency,
            interpolation="c",
            nan_transparent=True,
        )

        self.fig.colorbar(position=self.CB_POSITION, frame=["x+lVs30", "y+lm/s"])

        return self

    def save(self, output_ffp: Path, dpi: int = 900):
        self.fig.savefig(output_ffp, dpi=dpi, anti_alias=True)


### Plotly


def dataset_locations_map(df: pd.DataFrame, output_ffp: Path) -> None:
    """
    Generates a map showing the locations of measurements in the dataset,
    and saves it to the specified output file path.

    Hovering shows all input variable values for each measurement.
    """
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        map=dict(zoom=3, center=dict(lat=df.lat.mean(), lon=df.lon.mean())),
        showlegend=False,
    )

    hover_template = [
        f"{constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING.get(col, col)}: %{{customdata[{i}]}}"
        for i, col in enumerate(df.columns)
    ]
    hover_template = "<br>".join(hover_template) + "<extra></extra>"

    fig.add_trace(
        go.Scattermap(
            lon=df["lon"],
            lat=df["lat"],
            name="Measurements",
            mode="markers",
            marker=dict(size=4, color="blue"),
            hovertemplate=hover_template,
            customdata=df.values,
        )
    )

    fig.write_html(output_ffp)


def nz_site_database_map(
    df: pd.DataFrame, output_ffp: Path, markersize: int = 8
) -> None:
    """
    Generates a map showing the locations of sites in the NZ Site Database,
    colored by quality score, and saves it to the specified output file path.
    """
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        map=dict(zoom=3, center=dict(lat=df.lat.mean(), lon=df.lon.mean())),
        showlegend=True,
    )

    hover_template = [
        f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(df.columns)
    ]
    hover_template = "<br>".join(hover_template) + "<extra></extra>"

    if "Q3" in df["quality_score"].unique():
        q3_df = df[df["quality_score"] == "Q3"]
        fig.add_trace(
            go.Scattermap(
                lon=q3_df["lon"],
                lat=q3_df["lat"],
                name="Q3 Sites",
                mode="markers",
                marker=dict(size=markersize, color="red"),
                hovertemplate=hover_template,
                customdata=q3_df.values,
            )
        )

    if "Q2" in df["quality_score"].unique():
        q2_df = df[df["quality_score"] == "Q2"]
        fig.add_trace(
            go.Scattermap(
                lon=q2_df["lon"],
                lat=q2_df["lat"],
                name="Q2 Sites",
                mode="markers",
                marker=dict(size=markersize, color="#f032e6"),
                hovertemplate=hover_template,
                customdata=q2_df.values,
            )
        )

    q1_df = df[df["quality_score"] == "Q1"]
    fig.add_trace(
        go.Scattermap(
            lon=q1_df["lon"],
            lat=q1_df["lat"],
            name="Q1 Sites",
            mode="markers",
            marker=dict(size=markersize, color="blue"),
            hovertemplate=hover_template,
            customdata=q1_df.values,
        )
    )

    fig.write_html(output_ffp)


def nz_site_residuals_map(results_df: pd.DataFrame, output_ffp: Path) -> None:
    """
    Generates a map showing the locations of sites in the NZ Site Database,
    colored by ln(vs30) - ln(pred_vs30), and saves it to the specified output file path.
    """
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        map=dict(
            zoom=3, center=dict(lat=results_df.lat.mean(), lon=results_df.lon.mean())
        ),
        showlegend=False,
    )

    hover_columns = [
        "station",
        "lon",
        "lat",
        "vs30",
        "pred_vs30",
        "ln_residual",
        "quality_score",
    ]
    hover_template = [
        f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(hover_columns)
    ]
    hover_template = "<br>".join(hover_template) + "<extra></extra>"

    fig.add_trace(
        go.Scattermap(
            lon=results_df["lon"],
            lat=results_df["lat"],
            name="Sites",
            mode="markers",
            marker=dict(
                size=8,
                color=results_df["ln_residual"],
                colorscale="RdBu",
                colorbar_title="ln(vs30) - ln(pred_vs30)",
                cmin=-1.0,
                cmax=1.0,
            ),
            hovertemplate=hover_template,
            customdata=results_df[hover_columns].values,
        )
    )

    fig.write_html(output_ffp)
