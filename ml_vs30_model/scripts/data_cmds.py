import logging
from pathlib import Path

import typer

import ml_tools as mlt
import ml_vs30_model as vs30

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False, add_completion=False)


@app.command("gen-dataset")
def gen_dataset(config_ffp: Path, out_ffp: Path, log_ffp: Path | None = None):
    """
    Creates a dataset for training a VS30 model,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging(log_file=log_ffp)
    logging.getLogger("rclone").setLevel(logging.WARNING)

    config = vs30.DataConfig.from_yaml(config_ffp)
    vs30.data.gen_dataset(config, out_ffp)


@app.command("create-nz-input-grid")
def create_nz_input_grid(
    resolution: float,
    output_dir: Path,
    variables: list[vs30.constants.InputVariable],
    tolerance: int | None = None,
    min_area: int | None = None,
    n_procs: int = 1,
):
    """
    Creates a grid of input variable values for New Zealand,
    based on the provided resolution (in degrees).
    """
    mlt.utils.setup_logging()

    vs30.data.create_nz_input_grid(
        vs30.constants.NZ_BOUNDING_BOX,
        resolution,
        output_dir,
        variables,
        tolerance=tolerance,
        min_area=min_area,
        n_procs=n_procs,
    )


if __name__ == "__main__":
    app()
