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
        self.region = fig_kwargs["region"]

        self.fig = plotting.gen_region_fig(
            **fig_kwargs,
            config_options=config_options,
            plot_kwargs=plot_kwargs,
        )

    def add_town_labels(self, towns: list[str] = None, **plot_kwargs):
        """Adds town labels"""
        plot_kwargs = {
            "style": "c0.15c",
            "pen": "0.5p,black",
            "fill": None
        }

        towns = towns or constants.TOWN_COORDS.keys()

        # for town_name, town_coords in constants.TOWN_COORDS.items():
        for town in towns:
            town_coords = constants.TOWN_COORDS[town]
            self.fig.plot(x=town_coords[0], y=town_coords[1], **plot_kwargs)

            offset = "0.0c/0.12c"
            if town == "Blenheim":
                offset = "-0.45c/-0.3c"
                
            self.fig.text(
                x=town_coords[0],
                y=town_coords[1],
                text=town,
                font=constants.GMT_FIG_MINOR_FONT_LABEL,
                offset=offset,
                justify="CB"
            )

    def add_region_labels(self, regions: list[str] = None):
        """Adds region labels"""
        regions = regions or constants.REGION_COORDS.keys()

        for region in regions:
            region_coords = constants.REGION_COORDS[region]
            # self.fig.plot(x=region_coords[0], y=region_coords[1], **plot_kwargs)
            self.fig.text(
                x=region_coords[0],
                y=region_coords[1],
                text=region,
                font=constants.GMT_FIG_MINOR_FONT_LABEL,
                # offset="0.0c/0.12c",
                justify="CB"
            )


    def add_city_labels(self, **plot_kwargs):
        """Adds city labels"""
        plot_kwargs = {
            "style": "c0.3c",
            "pen": "0.75p,black",
            "fill": None
        }
        for city_name, city_coords in constants.CITY_COORDS.items():
            self.fig.plot(x=city_coords[0], y=city_coords[1], **plot_kwargs)
            self.fig.text(
                x=city_coords[0],
                y=city_coords[1],
                text=city_name,
                font=constants.GMT_FIG_FONT_LABEL,
                offset="0.0c/0.25c",
                justify="CB"
            )

    def add_highways(self, pen_width: float = 0.3, pen_color: str = "orange"):
        map_data = plotting.NZMapData.load(region=self.region, high_res_topo=False)
        self.fig.plot(
            data=map_data.highway_df,
            pen=f"{pen_width}p,{pen_color}",
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
        show_colorbar: bool = True,
        **plot_grid_kwargs,
    ):
        """Adds a ratio grid to the existing figure."""
        plot_grid_kwargs = {
            # "cb_label": cb_label or data_key,
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

        plotting.plot_grid(self.fig, grid, show_cb=show_colorbar, **plot_grid_kwargs)

        return self
    
    def plot_input_variable_values(
            self, 
            input_variable_df: pd.DataFrame,
            variable: constants.InputVariable,
            transparency: float | None = None,
            region: tuple[float, float, float, float] | None = None,
            cmap_limits: tuple[float, float] | None = None,
            grid_spacing: str = "250e/250e",
    ):
        """Adds a grid of input variable values to the existing figure."""
        start = time.time()
        grid = plotting.create_grid(
            input_variable_df,
            variable.value,
            region=region,
            grid_spacing=grid_spacing,
            interp_method="nearest",
        )
        logger.info(f"Took: {time.time() - start} to create grid.")

        cmap_limits = constants.INPUT_VARIABLE_CMAP_LIMITS[variable] if cmap_limits is None else cmap_limits

        # Set the extreme colors
        pygmt.config(COLOR_BACKGROUND="black", COLOR_FOREGROUND="yellow")

        # Plot the grid
        pygmt.makecpt(
            cmap="viridis", series=[cmap_limits[0], cmap_limits[1]]
        )
        self.fig.grdimage(
            grid,
            cmap=True,
            transparency=transparency,
            interpolation="c",
            nan_transparent=True,
        )

        self.fig.colorbar(position=self.CB_POSITION, frame=[f"x+l{constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING[variable]}"])

        return self

    def plot_pred_std_vs30(
        self,
        pred_std_df: pd.DataFrame,
        transparency: float | None = None,
        region: tuple[float, float, float, float] | None = None,
        std_limits: tuple[float, float] = (0, 1),
        grid_spacing: str = "250e/250e",
        interp_method: str = "nearest",
    ):
        start = time.time()
        grid = plotting.create_grid(
            pred_std_df,
            "pred_std_vs30",
            region=region,
            grid_spacing=grid_spacing,
            interp_method=interp_method,
        )
        logger.info(f"Took: {time.time() - start} to create grid.")

        # Set the extreme colors
        pygmt.config(COLOR_BACKGROUND="white", COLOR_FOREGROUND="black")

        # Plot the grid
        pygmt.makecpt(
            cmap="hot", series=[std_limits[0], std_limits[1]], reverse=True
        )
        self.fig.grdimage(
            grid,
            cmap=True,
            transparency=transparency,
            interpolation="c",
            nan_transparent=True,
        )

        self.fig.colorbar(position=self.CB_POSITION, frame=["x+lPredicted Standard Deviation"])

        return self

    def plot_vs30_values(
        self,
        vs30_df: pd.DataFrame,
        transparency: float | None = None,
        region: tuple[float, float, float, float] | None = None,
        vs_30_cmap_limits: tuple[float, float] = (0, 1000),
        grid_spacing: str = "250e/250e",
        interp_method: str = "nearest",
        show_colorbar: bool = True,
        show_colorbar_label: bool = True
    ):
        start = time.time()
        grid = plotting.create_grid(
            vs30_df,
            "vs30",
            region=region,
            grid_spacing=grid_spacing,
            interp_method=interp_method,
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

        if show_colorbar:
            cb_frame = ["x+lVs30 (m/s)", "y+lm/s"]
            if not show_colorbar_label:
                cb_frame[0] = "x"
            self.fig.colorbar(position=self.CB_POSITION, frame=cb_frame)

        return self

    def save(self, output_ffp: Path, dpi: int = constants.FIG_DPI):
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
