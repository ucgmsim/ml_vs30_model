from pathlib import Path

import numpy as np
import pandas as pd
import typer
import rasterio
import geopandas as gpd
import seaborn as sns
import folium

import ml_tools as mlt

import ml_vs30_model as vs30

app = typer.Typer(
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@app.command("dataset-locations-map")
def dataset_locations_map(dataset_ffp: Path, output_ffp: Path):
    dataset_df = pd.read_parquet(dataset_ffp)

    vs30.plotting.spatial.dataset_locations_map(dataset_df, output_ffp)


@app.command("nz-site-database-map")
def nz_site_database_map(dataset_ffp: Path, output_ffp: Path):
    dataset_df = pd.read_csv(dataset_ffp)

    vs30.plotting.spatial.nz_site_database_map(dataset_df, output_ffp)


@app.command("gen-feature-importance-plots")
def gen_feature_importance_plots(
    model_dir: Path,
    gen_waterfall_plots: bool = False,
):
    """
    Generates feature importance plots for the model in the provided directory.
    """
    mlt.utils.setup_logging()

    vs30.post_processing.gen_feature_importance_plots(
        model_dir, gen_waterfall_plots=gen_waterfall_plots
    )


@app.command("gen-nz-vs30-hist")
def gen_nz_vs30_hist(tif_ffp: Path, band_ix: int, output_ffp: Path, n_bins: int = 50):
    with rasterio.open(tif_ffp) as ds:
        assert (
            ds.crs.to_epsg() == vs30.constants.NZTM2000_EPSG
        ), "Dataset CRS is not NZTM"
        vs30_values = ds.read(band_ix, masked=True).astype(float).filled(np.nan)

    # Drop nan values
    vs30_values = vs30_values[~np.isnan(vs30_values)]

    vs30.plotting.other.plot_nz_vs30_hist(vs30_values, output_ffp, n_bins=n_bins)


@app.command("gen-lithology-map")
def gen_lithology_map(
    dataset_ffp: Path, output_ffp: Path, min_count: int = 5, max_count: int = 100
):
    """
    Generates a map of lithology categories for the categories
    in the dataset with counts between min_count and max_count.
    """
    # Load data
    geo_df = gpd.read_file(
        vs30.constants.BASE_DATA_DIR
        / "input_data"
        / vs30.data_loaders.ShapeLoader.VAR_TO_FILENAME_MAP[
            vs30.constants.InputVariable.NZLithologyCategory
        ]
    ).to_crs(epsg=vs30.constants.WGS84_EPSG)
    dataset_df = pd.read_parquet(dataset_ffp)

    # Select relevant categories
    cat_count = dataset_df[
        vs30.constants.InputVariable.NZLithologyCategory
    ].value_counts()
    map_cats = cat_count.loc[
        (cat_count >= min_count) & (cat_count < max_count)
    ].sort_values(ascending=False)

    m = folium.Map(location=[-40.9006, 174.8860], zoom_start=5, tiles="cartodbpositron")

    colors = sns.color_palette("tab20", n_colors=len(map_cats)).as_hex()
    for cur_cat, cur_color in zip(map_cats.index, colors):
        color_swatch = f'<span style="background:{cur_color};width:12px;height:12px;display:inline-block;margin-right:4px;border-radius:2px;"></span>'
        layer = folium.FeatureGroup(
            name=f"{color_swatch}{cur_cat} (N={cat_count[cur_cat]})"
        )
        cur_cat_df = geo_df[geo_df.LITHO2014 == cur_cat]

        for ix, cur_row in cur_cat_df.iterrows():
            folium.GeoJson(
                cur_row.geometry,
                style_function=lambda x, color=cur_color: {"color": color},
            ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_ffp)


@app.command("gen-geological-unit-map")
def gen_geological_unit_map(
    dataset_ffp: Path, output_ffp: Path, min_count: int = 10, max_count: int = 100
):
    """
    Generates a map of geological units for the categories 
    in the dataset with counts between min_count and max_count.
    """
    # Load data
    geo_df = (
        gpd.read_file(
            vs30.constants.BASE_DATA_DIR
            / "input_data"
            / vs30.data_loaders.ShapeLoader.VAR_TO_FILENAME_MAP[
                vs30.constants.InputVariable.NZGeologicalUnit
            ]
        )
        .to_crs(epsg=vs30.constants.WGS84_EPSG)
    )
    dataset_df = pd.read_parquet(dataset_ffp)

    # Select relevant categories
    cat_count = dataset_df[vs30.constants.InputVariable.NZGeologicalUnit].value_counts()
    map_cats = cat_count.loc[
        (cat_count >= min_count) & (cat_count < max_count)
    ].sort_values(ascending=False)

    m = folium.Map(location=[-40.9006, 174.8860], zoom_start=5, tiles="cartodbpositron")

    colors = sns.color_palette("tab20", n_colors=len(map_cats)).as_hex()
    for cur_cat, cur_color in zip(map_cats.index, colors):
        cur_cat_df = geo_df[geo_df["MAPSYMBOL"] == cur_cat]

        color_swatch = f'<span style="background:{cur_color};width:12px;height:12px;display:inline-block;margin-right:4px;border-radius:2px;"></span>'
        layer = folium.FeatureGroup(
            name=f"{color_swatch}{cur_cat} ({cur_cat_df['MAPNAME'].iloc[0]}) (N={cat_count[cur_cat]})"
        )

        for ix, cur_row in cur_cat_df.iterrows():
            folium.GeoJson(
                cur_row.geometry,
                style_function=lambda x, color=cur_color: {"color": color},
            ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_ffp)

@app.command("gen-geological-map")
def gen_geological_map(output_ffp: Path, key: str):
    
    geo_df = gpd.read_file(
        vs30.constants.BASE_DATA_DIR
        / "input_data"
        / vs30.data_loaders.ShapeLoader.VAR_TO_FILENAME_MAP[
            vs30.constants.InputVariable.NZGeologicalUnit
        ]
    ).to_crs(epsg=vs30.constants.WGS84_EPSG)

    categories = geo_df[key].unique()

    m = folium.Map(location=[-40.9006, 174.8860], zoom_start=5, tiles="cartodbpositron")

    colors = sns.color_palette("tab20", n_colors=len(categories)).as_hex()
    for cur_cat, cur_color in zip(categories, colors):
        cur_geo_df = geo_df[geo_df[key] == cur_cat]

        layer = folium.FeatureGroup()
        for ix, cur_row in cur_geo_df.iterrows():
            folium.GeoJson(
                cur_row.geometry,
                style_function=lambda x, color=cur_color: {"color": color},
                popup=folium.Popup(f"{key}: {cur_cat}", max_width=300),
            ).add_to(layer)

        layer.add_to(m)
    
    m.save(output_ffp)


if __name__ == "__main__":
    app()
