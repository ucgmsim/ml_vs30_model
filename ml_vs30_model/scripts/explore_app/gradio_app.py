from functools import lru_cache
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import folium
import requests

import ml_vs30_model as vs30
from qcore import coordinates

BASE_URL = "http://127.0.0.1:8000/xr"

# variables = list(vs30.constants.INPUT_VAR_TO_FFP_MAP.keys())
datasets = [f.name for f in vs30.constants.BASE_DATA_DIR.glob("grids/*") if f.is_dir()]


def get_variable_stats(dataset_name: str, variable: str):
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

    min_val, max_val = float(np.round((stats_dict["min"]), 3)), float(
        np.round(stats_dict["max"], 3)
    )
    step_size = (max_val - min_val) / 100

    return (
        stats,
        gr.Slider(minimum=min_val, maximum=max_val, value=min_val, step=step_size),
        gr.Slider(minimum=min_val, maximum=max_val, value=max_val, step=step_size),
    )


def supported_variables(dataset_name: str):
    """Gets the list of variables available in the dataset for tiling."""
    variables_tiler_url = f"{BASE_URL}/variables?url={dataset_name}"

    vars = json.loads(requests.get(variables_tiler_url).text)
    vars.remove("spatial_ref")  # Remove the spatial_ref

    return gr.Dropdown(choices=vars)


def create_map(
    dataset_name: str, variable: str, cmap: str, cmap_min: float, cmap_max: float
):
    # tiler_url = f"http://127.0.0.1:8000/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}"
    # tiler_url += f"?url={variable}&colormap_name={cmap}&rescale={cmap_min},{cmap_max}"

    tiler_url = (
        BASE_URL
        + "/tiles/WebMercatorQuad/{z}/{x}/{y}"
        + f"?url={dataset_name}&variable={variable}"
        + f"&colormap_name={cmap}&rescale={cmap_min},{cmap_max}"
    )

    m = folium.Map(location=[-43.5, 172.6], zoom_start=6, tiles="cartodb positron")

    folium.TileLayer(
        tiles=tiler_url,
        attr="My Data",
        name=variable,
        overlay=True,
        control=True,
    ).add_to(m)

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


def gen_hist_plot(variable: str, variable_stats: dict):
    min_val, max_val = variable_stats["min"], variable_stats["max"]
    counts, bin_edges = variable_stats["histogram"]

    bin_width = np.diff(bin_edges)[0]
    bin_centres = bin_edges[:-1] + (bin_width / 2)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(bin_centres, counts, width=bin_width, align="center")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    ax.set_xlim(min_val, max_val)
    ax.set_xlabel(variable)

    fig.tight_layout()
    plt.close()
    return fig


with gr.Blocks() as demo:

    map = gr.HTML("<h1>Select a variable...</h1>")
    cbar_plot = gr.Plot(container=False)

    gr.Markdown("# Variable histogram")
    hist_plot = gr.Plot(container=False)

    var_stats = gr.State()

    with gr.Sidebar(open=True):
        with gr.Group():
            dataset_dropdown = gr.Dropdown(
                interactive=True, choices=datasets, label="Dataset", show_label=True
            )
            var_dropdown = gr.Dropdown(
                label="Input Variable",
                show_label=True,
            )
            data_update_btn = gr.Button("Update", variant="primary")

        gr.HTML("<hr>")

        # Colormap
        with gr.Group():
            cmap_dropdown = gr.Dropdown(
                choices=["viridis", "plasma", "inferno", "magma", "greys"],
                label="Colormap",
                value="viridis",
                show_label=True,
            )

            cmap_min_slider = gr.Slider(
                label="Colormap min", interactive=True, precision=3
            )
            cmap_max_slider = gr.Slider(
                label="Colormap max", interactive=True, precision=3
            )

            vis_update_btn = gr.Button("Update", variant="primary")

    dataset_dropdown.change(
        supported_variables, inputs=dataset_dropdown, outputs=var_dropdown
    )

    data_update_btn.click(
        get_variable_stats,
        inputs=[dataset_dropdown, var_dropdown],
        outputs=[var_stats, cmap_min_slider, cmap_max_slider],
    ).then(
        create_map,
        inputs=[
            dataset_dropdown,
            var_dropdown,
            cmap_dropdown,
            cmap_min_slider,
            cmap_max_slider,
        ],
        outputs=[map, cbar_plot],
    )

    # # Input variable changed
    # var_dropdown.change(
    #     get_variable_stats, inputs=var_dropdown, outputs=[var_stats, cmap_min_slider, cmap_max_slider]
    # ).then(
    #     create_map,
    #     inputs=[var_dropdown, cmap_dropdown, cmap_min_slider, cmap_max_slider],
    #     outputs=[map, cbar_plot],
    # ).then(
    #     gen_hist_plot,
    #     inputs=[var_dropdown, var_stats],
    #     outputs=hist_plot,
    # )

    # # Colormap settings changed
    # cmap_dropdown.change(
    #     create_map,
    #     inputs=[var_dropdown, cmap_dropdown, cmap_min_slider, cmap_max_slider],
    #     outputs=[map, cbar_plot],
    # )
    # cmap_min_slider.change(
    #     create_map,
    #     inputs=[var_dropdown, cmap_dropdown, cmap_min_slider, cmap_max_slider],
    #     outputs=[map, cbar_plot],
    # )
    # cmap_max_slider.change(
    #     create_map,
    #     inputs=[var_dropdown, cmap_dropdown, cmap_min_slider, cmap_max_slider],
    #     outputs=[map, cbar_plot],
    # )


demo.launch(css="""
    .gradio-container {padding: 0 !important; margin: 0 !important;}
    .main {padding: 0 !important;}
    #component-0 {height: 100vh !important;}
    footer {display: none !important;}
""")
