from pathlib import Path

import numpy as np
import pandas as pd
import typer
import rasterio

import ml_tools as mlt

import ml_vs30_model as vs30

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False, add_completion=False)


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

    vs30.post_processing.gen_feature_importance_plots(model_dir, gen_waterfall_plots=gen_waterfall_plots)

@app.command("gen-nz-vs30-hist")
def gen_nz_vs30_hist(tif_ffp: Path, band_ix: int, output_ffp: Path, n_bins: int = 50):
    with rasterio.open(tif_ffp) as ds:
        assert ds.crs.to_epsg() == vs30.constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"
        vs30_values = ds.read(band_ix, masked=True).astype(float).filled(np.nan)

    # Drop nan values
    vs30_values = vs30_values[~np.isnan(vs30_values)]

    vs30.plotting.other.plot_nz_vs30_hist(vs30_values, output_ffp, n_bins=n_bins)


if __name__ == "__main__":
    app()
