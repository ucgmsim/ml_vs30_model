import logging
from pathlib import Path

import typer

import ml_tools as mlt
import ml_vs30_model as vs30

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False)


@app.command("gen-dataset")
def gen_dataset(config_ffp: Path, out_ffp: Path, log_ffp: Path | None = None):
    logger = mlt.utils.setup_logging(log_file=log_ffp)
    logging.getLogger("rclone").setLevel(logging.WARNING)

    if out_ffp.exists():
        vs30.utils.raise_log(
            FileExistsError, f"Output file already exists: {out_ffp}", logger
        )

    config = vs30.DataConfig.from_yaml(config_ffp)
    vs30.data.gen_dataset(config, out_ffp)


@app.command("placeholder")
def placeholder(config_ffp: Path):
    print("wtf")


if __name__ == "__main__":
    app()
