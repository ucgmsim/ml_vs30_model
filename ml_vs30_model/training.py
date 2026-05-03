from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import model_selection as ms

import ml_tools as mlt
import shap

from .configs import RunConfig
from . import post_processing

logger = logging.getLogger(__name__)


def cv_train(
    model_train_fn: callable, run_config: RunConfig, base_out_dir: Path, run_post_processing: bool = True, **model_fn_kwargs: dict
) -> None:
    """
    Runs cross-validation training and saves results per fold.

    Parameters
    ----------
    model_train_fn : callable
        The function that runs the training for a single fold. It should have the signature:
        model_train_fn(dataset_df, train_sites, val_sites, run_config, fold_out_dir, cv_ix)
    run_config : RunConfig
        The configuration for the run.
    base_out_dir : Path
        The base directory where results should be saved.
    run_post_processing : bool, optional
        Whether to run post-processing after training, by default True.
    **model_fn_kwargs : dict
        Additional keyword arguments to pass to the model_train_fn.
    """
    logger.info(f"Loading dataset from {run_config.dataset_ffp}")
    dataset_df = pd.read_parquet(run_config.dataset_ffp)
    logger.info(f"Dataset loaded with {len(dataset_df)} samples")

    kf = ms.KFold(
        n_splits=run_config.n_cv_folds, shuffle=True, random_state=run_config.seed
    )
    logger.info(f"Starting {run_config.n_cv_folds}-fold cross-validation")

    for i, (train, val) in enumerate(kf.split(dataset_df)):
        logger.info(f"Processing fold {i+1}/{run_config.n_cv_folds}")
        train_sites, val_sites = (
            dataset_df.iloc[train].index.values,
            dataset_df.iloc[val].index.values,
        )
        # train_y, val_y = dataset_df.iloc[train]["vs30"], dataset_df.iloc[val]["vs30"]
        logger.info(
            f"Train size: {train_sites.size}, Validation size: {val_sites.size}"
        )

        model_train_fn(
            dataset_df,
            train_sites,
            val_sites,
            run_config,
            base_out_dir / f"cv_{i:02d}",
            cv_ix=i,
            **model_fn_kwargs
        )
        logger.info(f"Completed fold {i+1}/{run_config.n_cv_folds}")

    logger.info("Cross-validation training completed")

    # Combine model validation results into a single file
    cv_ids = [f"cv_{i:02d}" for i in range(run_config.n_cv_folds)]
    val_results_dfs = []
    for cv_id in cv_ids:
        fold_val_results_ffp = base_out_dir / cv_id / "val_results.parquet"
        fold_val_results_df = pd.read_parquet(fold_val_results_ffp)
        val_results_dfs.append(fold_val_results_df)
    val_results_df = pd.concat(val_results_dfs)

    # Add extra information
    val_results_df = post_processing.add_residuals(val_results_df)
    val_results_df = post_processing.add_mae(val_results_df)
    val_results_df = post_processing.add_lnVs30_mse(val_results_df)
    val_results_df.to_parquet(base_out_dir / "val_results.parquet")

    # Combine metrics into a single file
    train_metric_dfs = [
        pd.read_parquet(base_out_dir / cv_id / "train_metrics.parquet")
        for cv_id in cv_ids
    ]
    val_metric_dfs = [
        pd.read_parquet(base_out_dir / cv_id / "val_metrics.parquet")
        for cv_id in cv_ids
    ]

    # Save metrics
    train_metrics = xr.DataArray(
        np.stack([df.values for df in train_metric_dfs], axis=0),
        dims=["cv_fold", "iteration", "metric"],
        coords={
            "cv_fold": cv_ids,
            "iteration": train_metric_dfs[0].index.values,
            "metric": train_metric_dfs[0].columns.values,
        },
    )
    train_metrics.to_netcdf(base_out_dir / "train_metrics.nc")
    val_metrics = xr.DataArray(
        np.stack([df.values for df in val_metric_dfs], axis=0),
        dims=["cv_fold", "iteration", "metric"],
        coords={
            "cv_fold": cv_ids,
            "iteration": val_metric_dfs[0].index.values,
            "metric": val_metric_dfs[0].columns.values,
        },
    )
    val_metrics.to_netcdf(base_out_dir / "val_metrics.nc")

    # Compute combined shap values across folds
    explanations = [
        pd.read_pickle(base_out_dir / cv_id / "shap_values.pkl") for cv_id in cv_ids
    ]
    shap_values = shap.Explanation(
        values=np.concatenate([e.values for e in explanations], axis=0),
        base_values=np.concatenate([e.base_values for e in explanations], axis=0),
        data=np.concatenate([e.data for e in explanations], axis=0),
        feature_names=explanations[0].feature_names,
    )
    mlt.utils.write_pickle(shap_values, base_out_dir / "shap_values.pkl")

    # Generate post-processing plots
    if run_post_processing:
        logger.info("Running post-processing...")
        post_processing.gen_cv_iteration_metric_plots(
            base_out_dir, train_metrics, val_metrics=val_metrics
        )
        post_processing.gen_model_perfomance_plots(
            base_out_dir, results_df=val_results_df
        )
        post_processing.gen_spatial_plots(
            base_out_dir, results_df=val_results_df, run_config=run_config
        )
        post_processing.gen_feature_importance_plots(
            base_out_dir, shap_values, results_df=val_results_df
        )

    run_config.to_yaml(base_out_dir / "run_config.yaml")


# def full_train(run_config: RunConfig, out_dir: Path, run_post_processing: bool = True):
#     """Runs training on the full dataset and saves results."""
#     logger.info(f"Loading dataset from {run_config.dataset_ffp}")
#     dataset_df = pd.read_parquet(run_config.dataset_ffp)
#     logger.info(f"Dataset loaded with {len(dataset_df)} samples")

#     run_model_training(
#         dataset_df,
#         dataset_df.index.values,
#         None,
#         run_config,
#         out_dir,
#         save_train_results=True,
#     )

#     if run_post_processing:
#         logger.info("Running post-processing...")
#         train_results_df = pd.read_parquet(out_dir / "train_results.parquet")

#         # Quantities
#         train_results_df = post_processing.add_residuals(train_results_df)
#         train_results_df = post_processing.add_mae(train_results_df)
#         train_results_df = post_processing.add_lnVs30_mse(train_results_df)
#         shap_values = post_processing.compute_shap_feature_importance(out_dir)

#         # Plots
#         post_processing.gen_model_perfomance_plots(out_dir, results_df=train_results_df)
#         post_processing.gen_spatial_plots(out_dir, results_df=train_results_df)
#         post_processing.gen_feature_importance_plots(
#             out_dir, results_df=train_results_df, shap_values=shap_values
#         )
