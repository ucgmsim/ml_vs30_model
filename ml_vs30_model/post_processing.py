import logging
from pathlib import Path

import pandas as pd

from .plotting import model_perf_plots

logger = logging.getLogger(__name__)

def gen_model_perfomance_plots(results_dir: Path) -> None:
    """
    Generates model performance plots for the provided results directory.
    """
    val_results_df = pd.read_parquet(results_dir / "val_results.parquet")

    logger.info("Generating model performance plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    model_perf_plots.one_to_one_plot(val_results_df, outdir / "one_to_one_plot.png")
