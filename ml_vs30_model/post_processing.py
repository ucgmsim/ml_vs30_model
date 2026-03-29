import logging
from pathlib import Path

import xarray as xr
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
    results_df["lnVs30_mse"] = (
        np.log(results_df["vs30"]) - np.log(results_df["pred_vs30"])
    ) ** 2
    return results_df


def gen_model_perfomance_plots(
    results_dir: Path, results_df: pd.DataFrame | None = None
) -> None:
    """
    Generates model performance plots for the provided results directory.
    """
    if results_df is None:
        logger.info("Reading results dataframe from parquet file val_results.parquet.")
        results_df = pd.read_parquet(results_dir / "val_results.parquet")

    logger.info("Generating model performance plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    model_perf_plots.one_to_one_plot(results_df, outdir / "one_to_one_plot.png")
    model_perf_plots.residuals_histogram(results_df, outdir / "residuals_histogram.png")
    model_perf_plots.residual_kde(results_df, outdir / "residuals_kde.png")
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "mae_scatter_plot.png",
        metric_name="mae",
        x_limits=(0, 2000),
        y_limits=(0, 1000),
        show_geyin_maurer_model=True,
    )
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "lnVs30_mse_scatter_plot.png",
        metric_name="lnVs30_mse",
        x_limits=(0, 2000),
        # y_limits=(0, 1000),
    )
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "ln_residual_scatter_plot.png",
        metric_name="ln_residual",
        x_limits=(0, 2000),
        y_limits=(-1.5, 1.5),
    )


def gen_cv_iteration_metric_plots(
    results_dir: Path,
    train_metrics: xr.DataArray | None = None,
    val_metrics: xr.DataArray | None = None,
    metrics: list[str] | None = None,
):
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)

    # Load the data if not provided
    if train_metrics is None:
        logger.info("Reading train metrics from netcdf file train_metrics.nc.")
        train_metrics = xr.open_dataarray(results_dir / "train_metrics.nc")
    if val_metrics is None:
        if (results_dir / "val_metrics.nc").exists():
            logger.info("Reading validation metrics from netcdf file val_metrics.nc.")
            val_metrics = xr.open_dataarray(results_dir / "val_metrics.nc")
        else:
            logger.warning(
                "Validation metrics file val_metrics.nc not found. Validation metrics will be set to None."
            )
            val_metrics = None

    if metrics is None:
        metrics = train_metrics.metric.values

    for metric in metrics:
        model_perf_plots.cv_iteration_metric_plot(
            train_metrics.sel(metric=metric).to_pandas().T,
            val_metrics_df=(
                val_metrics.sel(metric=metric).to_pandas().T
                if val_metrics is not None
                else None
            ),
            out_fp=outdir / f"{metric}_iteration_plot.png",
            metric=metric,
        )


def gen_spatial_plots(
    results_dir: Path, results_df: pd.DataFrame | None = None
) -> None:
    """
    Generates spatial plots for the provided results directory.
    """
    if results_df is None:
        logger.info("Reading results dataframe from parquet file val_results.parquet.")
        results_df = pd.read_parquet(results_dir / "val_results.parquet")

    logger.info("Generating spatial plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    spatial.nz_site_residuals_map(results_df, outdir / "site_residual.html")
