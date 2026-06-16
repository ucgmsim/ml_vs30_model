from PIL import Image
from pathlib import Path
import json
import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import folium
import requests


import ml_vs30_model as vs30
from ml_vs30_model.constants import InputVariable
from qcore import coordinates

BASE_URL = "http://127.0.0.1:8000/xr"
MODEL_BASE_URL = "http://127.0.0.1:8000/model"

# variables = list(vs30.constants.INPUT_VAR_TO_FFP_MAP.keys())
datasets = [f.name for f in vs30.constants.BASE_DATA_DIR.glob("grids/*") if f.is_dir()]

models = [
    f.name
    for f in vs30.constants.BASE_DATA_DIR.glob("results/ind_results/*")
    if f.is_dir() and "full" in f.name and not f.name.startswith("_")
]

CMAP_LIMITS = {
    InputVariable.NZEnvDSSlopeDeg: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSlopeDeg][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSlopeDeg][1],
        60,
    ),
    InputVariable.NZNWTGroundwaterDepth: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZNWTGroundwaterDepth][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZNWTGroundwaterDepth][
            1
        ],
        1000,
    ),
    InputVariable.NZNLMGroundwaterDepth: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZNLMGroundwaterDepth][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZNLMGroundwaterDepth][
            1
        ],
        25,
    ),
    InputVariable.NZEnvDSDistanceRivers: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSDistanceRivers][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSDistanceRivers][
            1
        ],
        100_000,
    ),
    InputVariable.NZEnvDSDistanceRiversVertical: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZEnvDSDistanceRiversVertical
        ][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZEnvDSDistanceRiversVertical
        ][1],
        3500,
    ),
    InputVariable.NZEnvDSPrecipAnn: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSPrecipAnn][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSPrecipAnn][1],
        10_000,
    ),
    InputVariable.NZEnvDSSoilAcidP: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilAcidP][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilAcidP][1],
    ),
    InputVariable.NZEnvDSSoilAge: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilAge][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilAge][1],
    ),
    InputVariable.NZEnvDSSoilDrainage: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilDrainage][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilDrainage][1],
    ),
    InputVariable.NZEnvDSSoilInduration: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilInduration][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSSoilInduration][
            1
        ],
    ),
    InputVariable.NZEnvDSTopoGeomorphons: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoGeomorphons][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoGeomorphons][
            1
        ],
    ),
    InputVariable.NZEnvDSSoilParticleSize: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZEnvDSSoilParticleSize
        ][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZEnvDSSoilParticleSize
        ][1],
    ),
    InputVariable.NZEnvDSTopoNormalisedHeight: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZEnvDSTopoNormalisedHeight
        ][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZEnvDSTopoNormalisedHeight
        ][1],
    ),
    InputVariable.NZEnvDSTopoPosition: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoPosition][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoPosition][1],
        -50,
        -50,
    ),
    InputVariable.NZEnvDSTopoRoughness: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoRoughness][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoRoughness][
            1
        ],
        500,
    ),
    InputVariable.NZEnvDSTopoRuggedness: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoRuggedness][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoRuggedness][
            1
        ],
        100,
    ),
    InputVariable.NZEnvDSTopoValleyDepth: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoValleyDepth][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoValleyDepth][
            1
        ],
        750,
    ),
    InputVariable.NZEnvDSTopoWetness: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoWetness][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZEnvDSTopoWetness][1],
        14,
    ),
    InputVariable.DepthToGroundwater: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.DepthToGroundwater][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.DepthToGroundwater][1],
        -500,
        0,
    ),
    InputVariable.NZDistanceToCoast: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToCoast][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToCoast][1],
        150_000,
    ),
    InputVariable.NZDistanceToRiver_ST1: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST1][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST1][
            1
        ],
        200_000,
    ),
    InputVariable.NZDistanceToRiver_ST2: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST2][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST2][
            1
        ],
        200_000,
    ),
    InputVariable.NZDistanceToRiver_ST3: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST3][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST3][
            1
        ],
        200_000,
    ),
    InputVariable.NZDistanceToRiver_ST4: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST4][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST4][
            1
        ],
        200_000,
    ),
    InputVariable.NZDistanceToRiver_ST5: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST5][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST5][
            1
        ],
        200_000,
    ),
    InputVariable.NZDistanceToRiver_ST6: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST6][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST6][
            1
        ],
        200_000,
    ),
    InputVariable.NZDistanceToRiver_ST7: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST7][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST7][
            1
        ],
        200_000,
    ),
    InputVariable.NZDistanceToRiver_ST8: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST8][
            0
        ],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZDistanceToRiver_ST8][
            1
        ],
        200_000,
    ),
    InputVariable.NZGeologyCategory: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyCategory][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyCategory][1],
    ),
    InputVariable.NZGeologyAgeMin: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeMin][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeMin][1],
    ),
    InputVariable.NZGeologyAgeMax: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeMax][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeMax][1],
    ),
    InputVariable.CompoundTopgraphicIndex: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.CompoundTopgraphicIndex
        ][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.CompoundTopgraphicIndex
        ][1],
        -10,
        10,
    ),
    InputVariable.Elevation: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.Elevation][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.Elevation][1],
        3500,
    ),
    InputVariable.NZGeologyAgeMid: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeMid][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeMid][1],
    ),
    InputVariable.NZGeologyAgeLnMid: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeLnMid][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[InputVariable.NZGeologyAgeLnMid][1],
    ),
    InputVariable.NZCombinedGroundwaterDepth: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZCombinedGroundwaterDepth
        ][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZCombinedGroundwaterDepth
        ][1],
        1000,
    ),
    InputVariable.NZCombinedGroundwaterDepthLn: (
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZCombinedGroundwaterDepthLn
        ][0],
        vs30.constants.INPUT_VARIABLE_CMAP_LIMITS[
            InputVariable.NZCombinedGroundwaterDepthLn
        ][1],
    ),
}

QUALITY_COLOR_MAPPTING = vs30.constants.QUALITY_SCORE_COLORS

VS30_CMAP_MIN, VS30_CMAP_MAX = 0, 1200
VS30_CMAP_RES_MIN, VS30_CMAP_RES_MAX = -250, 250
VS30_CMAP_LN_RES_MIN, VS30_CMAP_LN_RES_MAX = -1, 1
VS30_CMAP_STD_MIN, VS30_CMAP_STD_MAX = 0, 1

DEFAULT_CMAP = "viridis"
STD_CMAP = "Reds"
RES_CMAP = "bwr_r"


def _get_model_dir(model_name: str) -> Path:
    return vs30.constants.BASE_DATA_DIR / "results/ind_results" / model_name


def make_cbar_html(cmap_name: str, cmap_min: float, cmap_max: float, label: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 0.4))
    norm = plt.Normalize(vmin=cmap_min, vmax=cmap_max)
    cmap = plt.get_cmap(cmap_name)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        label=label,
    )
    fig.patch.set_alpha(0.7)
    fig.tight_layout(pad=0.2)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    return f'<img src="data:image/png;base64,{b64}" style="position:absolute; bottom:30px; left:50%; transform:translateX(-50%); z-index:1000; width:400px;">'


def get_input_variable_stats(dataset_name: str, variable: str):
    if variable in CMAP_LIMITS:
        limits = CMAP_LIMITS[variable]
        if len(limits) == 2:
            min_val, max_val = limits
            allowed_min, allowed_max = limits
        elif len(limits) == 3:
            min_val, max_val, allowed_max = limits
            allowed_min = min_val
        elif len(limits) == 4:
            min_val, max_val, allowed_min, allowed_max = limits
        else:
            raise ValueError(
                f"Invalid CMAP_LIMITS entry for variable {variable}: {limits}"
            )
    else:
        params = {
            "url": dataset_name,
            "variable": variable,
        }

        # Get full dataset bounds
        info = requests.get(
            f"{BASE_URL}/info",
            params=params,
        ).json()
        bounds = info["bounds"]  # [minx, miny, maxx, maxy]

        # Build a GeoJSON polygon from bounds
        minx, miny, maxx, maxy = bounds
        min_lat, min_lon = coordinates.nztm_to_wgs_depth(np.array([miny, minx]))
        max_lat, max_lon = coordinates.nztm_to_wgs_depth(np.array([maxy, maxx]))

        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]
                ],
            },
            "properties": {},
        }

        response = requests.post(f"{BASE_URL}/statistics", params=params, json=geojson)

        if response.status_code != 200:
            raise ValueError(f"Error fetching variable stats: {response.text}")

        stats_dict = json.loads(response.text)["properties"]["statistics"]["b1"]
        counts, bin_edges = stats_dict["histogram"]
        stats = {
            "histogram": (counts, bin_edges),
            "min": stats_dict["min"],
            "max": stats_dict["max"],
        }

        min_val, max_val = float(np.round((stats["min"]), 3)), float(
            np.round(stats["max"], 3)
        )
        allowed_min, allowed_max = min_val, max_val

    step_size = (max_val - min_val) / 100
    return (
        gr.Slider(
            minimum=allowed_min, maximum=allowed_max, value=min_val, step=step_size
        ),
        gr.Slider(
            minimum=allowed_min, maximum=allowed_max, value=max_val, step=step_size
        ),
    )


def inputs_supported_variables(dataset_name: str):
    """Gets the list of variables available in the dataset for tiling."""
    variables_tiler_url = f"{BASE_URL}/variables?url={dataset_name}"

    vars = json.loads(requests.get(variables_tiler_url).text)
    if "spatial_ref" in vars:
        vars.remove("spatial_ref")  # Remove the spatial_ref

    return gr.Dropdown(choices=vars)


def create_inputs_map(
    dataset_name: str,
    variable: str,
    model_name: str,
    cmap: str,
    cmap_min: float,
    cmap_max: float,
    site_ln_res_min: float,
    site_ln_res_max: float,
):
    tiler_url = (
        BASE_URL
        + "/tiles/WebMercatorQuad/{z}/{x}/{y}"
        + f"?url={dataset_name}&variable={variable}"
        + f"&colormap_name={cmap}&rescale={cmap_min},{cmap_max}"
    )

    # Map
    m = folium.Map(location=[-42, 172.6], zoom_start=6, tiles="cartodb positron")
    folium.TileLayer(
        tiles=tiler_url,
        attr="My Data",
        name=variable,
        overlay=True,
        control=True,
    ).add_to(m)

    if model_name is not None:
        marker_group = get_model_markers(model_name, site_ln_res_min, site_ln_res_max)
        marker_group.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Colorbar
    fig, ax = plt.subplots(figsize=(16, 1))
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=cmap_min, vmax=cmap_max)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        label=variable,
    )
    fig.subplots_adjust(bottom=0.5)
    plt.close(fig)

    return (
        m._repr_html_(),
        fig,
    )


def get_model_markers(model_name: str, site_ln_res_min: float, site_ln_res_max: float):
    marker_group = folium.FeatureGroup(name="Sites", show=True)
    model_dir = _get_model_dir(model_name)
    train_df = pd.read_parquet(model_dir / "train_results.parquet")
    run_config = vs30.RunConfig.from_yaml(model_dir / "run_config.yaml")
    dataset_df = pd.read_parquet(run_config.dataset_ffp)

    for k, row in train_df.iterrows():
        if (site_ln_res_min is not None and row.ln_residual < site_ln_res_min) or (
            site_ln_res_max is not None and row.ln_residual > site_ln_res_max
        ):
            continue

        cur_quality_score = dataset_df.loc[k, "quality_score"]

        if (plot_ffp := model_dir / f"plots/waterfall_plots/{k}.png").exists():
            with open(plot_ffp, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            popup_html = f"Name: {k}, Vs30: {row['vs30']:.1f}, Pred: {row['pred_vs30']:.1f}, Quality: {cur_quality_score}"
            popup_html += (
                f'<br><img src="data:image/png;base64,{encoded}" width="800"><br>'
            )
        else:
            popup_html = f"Name: {k}<br>Vs30: {row['vs30']:.1f}<br>Pred: {row['pred_vs30']:.1f}<br>Quality: {cur_quality_score}"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            popup=folium.Popup(
                popup_html,
                max_width=800,
            ),
            color=QUALITY_COLOR_MAPPTING[cur_quality_score],
            fill=True,
        ).add_to(marker_group)

    return marker_group


def create_model_map(
    model_name: str,
    variable: str,
    site_ln_res_min: float | None,
    site_ln_res_max: float | None,
):
    if variable.endswith("_ln_res"):
        cmap_min, cmap_max = VS30_CMAP_LN_RES_MIN, VS30_CMAP_LN_RES_MAX
        cmap = RES_CMAP
    elif variable.endswith("_res"):
        cmap_min, cmap_max = VS30_CMAP_RES_MIN, VS30_CMAP_RES_MAX
        cmap = RES_CMAP
    elif variable.endswith("_std"):
        cmap_min, cmap_max = VS30_CMAP_STD_MIN, VS30_CMAP_STD_MAX
        cmap = STD_CMAP
    else:
        cmap_min, cmap_max = VS30_CMAP_MIN, VS30_CMAP_MAX
        cmap = DEFAULT_CMAP

    tiler_url = (
        MODEL_BASE_URL
        + "/tiles/WebMercatorQuad/{z}/{x}/{y}"
        + f"?url={model_name}&variable={variable}"
        + f"&colormap_name={cmap.lower()}&rescale={cmap_min},{cmap_max}"
    )

    # Map
    m = folium.Map(location=[-42, 172.6], zoom_start=6, tiles="cartodb positron")
    folium.TileLayer(
        tiles=tiler_url,
        attr="My Data",
        name=variable,
        overlay=True,
        control=True,
    ).add_to(m)

    # Add markers
    marker_group = get_model_markers(model_name, site_ln_res_min, site_ln_res_max)
    marker_group.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    map_html = m._repr_html_()
    cbar_html = make_cbar_html(cmap, cmap_min, cmap_max, variable)
    combined = f'<div style="position:relative">{map_html}{cbar_html}</div>'

    return combined


def models_supported_variables(model_name: str):
    model_variables_url = f"{MODEL_BASE_URL}/variables?url={model_name}"
    vars = json.loads(requests.get(model_variables_url).text)
    if "spatial_ref" in vars:
        vars.remove("spatial_ref")  # Remove the spatial_ref
    return gr.Dropdown(choices=vars)


def update_site_selection(model_name: str):
    model_dir = _get_model_dir(model_name)
    train_df = pd.read_parquet(model_dir / "train_results.parquet")
    site_names = train_df.index.astype(str).tolist()
    return gr.Dropdown(choices=site_names)


def update_site_plot(model_name: str, site_name: str):
    model_dir = _get_model_dir(model_name)

    if (ffp := model_dir / f"plots/waterfall_plots/{site_name}.png").exists():
        return gr.Image(value=Image.open(ffp))

    return None


with gr.Blocks() as demo:

    with gr.Tab("Model"):
        model_map = gr.HTML("<h1>Model predictions...</h1>")

        # gr.Markdown("----------------------")
        # gr.Markdown("# Site Analysis")

        # site_selection = gr.Dropdown(interactive=True, label="Select a site")
        # site_plot = gr.Image(interactive=False, container=False)

    with gr.Tab("Inputs"):
        inputs_map = gr.HTML("<h1>Select a dataset and variable...</h1>")
        inputs_cbar_plot = gr.Plot(container=False)

        # gr.Markdown("# Variable histogram")
        # inputs_hist_plot = gr.Plot(container=False)

        # inputs_var_stats = gr.State()

    with gr.Sidebar(open=True):
        with gr.Accordion("Model Options"):
            with gr.Group():
                model_selection = gr.Dropdown(
                    choices=models,
                    label="Model",
                    value=None,
                    show_label=True,
                    interactive=True,
                )
                model_variable_dropdown = gr.Dropdown(
                    label="Model Variable",
                    show_label=True,
                    value=None,
                    interactive=True,
                )
                model_update_btn = gr.Button("Update", variant="primary")

        with gr.Accordion(
            "Input Variable Options", open=True
        ) as input_options_accordion:
            with gr.Group():
                gr.Markdown("### Dataset/Variable")
                dataset_dropdown = gr.Dropdown(
                    interactive=True,
                    choices=datasets,
                    value=None,
                    label="Dataset",
                    show_label=True,
                )
                var_dropdown = gr.Dropdown(
                    label="Input Variable", show_label=True, value=None
                )
                data_update_btn = gr.Button("Update", variant="primary")

            gr.Markdown("----------------------")

            # Colormap
            with gr.Accordion("Colormap Options", open=False):
                with gr.Group():
                    input_cmap_dropdown = gr.Dropdown(
                        choices=["viridis", "plasma", "inferno", "magma", "greys"],
                        label="Colormap",
                        value="viridis",
                        show_label=True,
                    )

                    input_cmap_min_slider = gr.Slider(
                        label="Colormap min",
                        interactive=True,
                        precision=3,
                    )
                    input_cmap_max_slider = gr.Slider(
                        label="Colormap max", interactive=True, precision=3
                    )

                    input_vis_update_btn = gr.Button("Update", variant="primary")

        with gr.Accordion("Site Options", open=False):
            with gr.Group():
                site_ln_residual_min_slider = gr.Slider(
                    label="Ln Residual Min",
                    interactive=True,
                    precision=3,
                    value=-1.5,
                    minimum=-3,
                    maximum=3,
                )
                site_ln_residual_max_slider = gr.Slider(
                    label="Ln Residual Max",
                    interactive=True,
                    precision=3,
                    value=1.5,
                    minimum=-3,
                    maximum=3,
                )
                site_update_btn = gr.Button("Update", variant="primary")

    # On dataset change, retrieve supported variables
    dataset_dropdown.change(
        inputs_supported_variables, inputs=dataset_dropdown, outputs=var_dropdown
    )

    # Dataset update button
    data_update_btn.click(
        get_input_variable_stats,
        inputs=[dataset_dropdown, var_dropdown],
        outputs=[input_cmap_min_slider, input_cmap_max_slider],
    ).then(
        create_inputs_map,
        inputs=[
            dataset_dropdown,
            var_dropdown,
            model_selection,
            input_cmap_dropdown,
            input_cmap_min_slider,
            input_cmap_max_slider,
            site_ln_residual_min_slider,
            site_ln_residual_max_slider,
        ],
        outputs=[inputs_map, inputs_cbar_plot],
    )
    # .then(
    #     gen_inputs_hist_plot,
    #     inputs=[var_dropdown, inputs_var_stats],
    #     outputs=inputs_hist_plot,
    # )

    # Visualization update button
    input_vis_update_btn.click(
        create_inputs_map,
        inputs=[
            dataset_dropdown,
            var_dropdown,
            model_selection,
            input_cmap_dropdown,
            input_cmap_min_slider,
            input_cmap_max_slider,
            site_ln_residual_min_slider,
            site_ln_residual_max_slider,
        ],
        outputs=[inputs_map, inputs_cbar_plot],
    )

    # On model change, retrieve supported variables
    model_selection.change(
        models_supported_variables,
        inputs=model_selection,
        outputs=model_variable_dropdown,
    )

    # Model update button
    model_update_btn.click(
        create_model_map,
        inputs=[
            model_selection,
            model_variable_dropdown,
            site_ln_residual_min_slider,
            site_ln_residual_max_slider,
        ],
        # inputs=[model_selection, model_variable_dropdown, model_cmap_dropdown],
        outputs=model_map,
    )

    # .then(
    #     update_site_selection,
    #     inputs=model_selection,
    #     outputs=site_selection,
    # )

    site_update_btn.click(
        create_model_map,
        inputs=[
            model_selection,
            model_variable_dropdown,
            site_ln_residual_min_slider,
            site_ln_residual_max_slider,
        ],
        outputs=model_map,
    ).then(
        create_inputs_map,
        inputs=[
            dataset_dropdown,
            var_dropdown,
            model_selection,
            input_cmap_dropdown,
            input_cmap_min_slider,
            input_cmap_max_slider,
            site_ln_residual_min_slider,
            site_ln_residual_max_slider,
        ],
        outputs=[inputs_map, inputs_cbar_plot],
    )

    # site_selection.change(
    #     update_site_plot,
    #     inputs=[model_selection, site_selection],
    #     outputs=site_plot,
    # )

    # model_cmap_update_btn.click(
    #     create_model_map,
    #     inputs=[model_selection, model_variable_dropdown, model_cmap_dropdown],
    #     outputs=model_map,
    # )

    demo.load(lambda: gr.Accordion(open=False), outputs=input_options_accordion)
demo.launch(css="""
    .gradio-container {padding: 0 !important; margin: 0 !important;}
    .main {padding: 0 !important;}
    #component-0 {height: 100vh !important;}
    footer {display: none !important;}
""")
