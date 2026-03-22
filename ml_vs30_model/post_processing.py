import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import model_perf_plots
from .plotting import spatial

logger = logging.getLogger(__name__)


def add_residuals(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds residuals to the provided results dataframe.
    """
    results_df["ln_residual"] = np.log(results_df["vs30"]) - np.log(
        results_df["pred_vs30"]
    )
    return results_df


def add_mae(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds mean absolute error (MAE) to the provided results dataframe.
    """
    results_df["mae"] = np.abs(results_df["vs30"] - results_df["pred_vs30"])
    return results_df

def add_lnVs30_mse(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds mean squared error (MSE) to the provided results dataframe.
    """
    results_df["lnVs30_mse"] = (np.log(results_df["vs30"]) - np.log(results_df["pred_vs30"])) ** 2
    return results_df


def gen_model_perfomance_plots(
    results_dir: Path, val_results_df: pd.DataFrame | None = None
) -> None:
    """
    Generates model performance plots for the provided results directory.
    """
    val_results_df = (
        pd.read_parquet(results_dir / "val_results.parquet")
        if val_results_df is None
        else val_results_df
    )

    logger.info("Generating model performance plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    model_perf_plots.one_to_one_plot(val_results_df, outdir / "one_to_one_plot.png")
    model_perf_plots.residuals_histogram(
        val_results_df, outdir / "residuals_histogram.png"
    )
    model_perf_plots.residual_kde(val_results_df, outdir / "residuals_kde.png")
    model_perf_plots.metric_scatter_plot(
        val_results_df,
        outdir / "mae_scatter_plot.png",
        metric_name="mae",
        x_limits=(0, 2000),
        y_limits=(0, 1000),
        show_geyin_maurer_model=True,
    )
    model_perf_plots.metric_scatter_plot(
        val_results_df,
        outdir / "lnVs30_mse_scatter_plot.png",
        metric_name="lnVs30_mse",
        x_limits=(0, 2000),
        # y_limits=(0, 1000),
    )
    model_perf_plots.metric_scatter_plot(
        val_results_df,
        outdir / "ln_residual_scatter_plot.png",
        metric_name="ln_residual",
        x_limits=(0, 2000),
        y_limits=(-1.5, 1.5),
    )


def gen_spatial_plots(
    results_dir: Path, val_results_df: pd.DataFrame | None = None
) -> None:
    """
    Generates spatial plots for the provided results directory.
    """
    val_results_df = (
        pd.read_parquet(results_dir / "val_results.parquet")
        if val_results_df is None
        else val_results_df
    )

    logger.info("Generating spatial plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    spatial.nz_site_residuals_map(val_results_df, outdir / "site_residual.html")
