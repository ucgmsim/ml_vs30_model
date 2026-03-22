from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .. import constants


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

def nz_site_database_map(df: pd.DataFrame, output_ffp: Path, markersize: int =8) -> None:
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
        f"{col}: %{{customdata[{i}]}}"
        for i, col in enumerate(df.columns)
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
        map=dict(zoom=3, center=dict(lat=results_df.lat.mean(), lon=results_df.lon.mean())),
        showlegend=False,
    )

    hover_columns = ["station", "lon", "lat", "vs30", "pred_vs30", "ln_residual"]
    hover_template = [
        f"{col}: %{{customdata[{i}]}}"
        for i, col in enumerate(hover_columns)
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


