from pathlib import Path

import pandas as pd
import typer

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


if __name__ == "__main__":
    app()
