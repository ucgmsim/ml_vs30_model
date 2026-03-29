from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import model_selection as ms

from catboost import CatBoostRegressor
from .configs import RunConfig
from . import pre_processing
from . import post_processing

logger = logging.getLogger(__name__)


def cv_train(
    run_config: RunConfig, base_out_dir: Path, run_post_processing: bool = True
) -> None:
    """Runs cross-validation training and saves results per fold."""
    logger.info(f"Loading dataset from {run_config.dataset_ffp}")
    dataset_df = pd.read_parquet(run_config.dataset_ffp)
    logger.info(f"Dataset loaded with {len(dataset_df)} samples")

    kf = ms.KFold(
        n_splits=run_config.n_cv_folds, shuffle=True, random_state=run_config.seed
    )
    logger.info(f"Starting {run_config.n_cv_folds}-fold cross-validation")

    for i, (train, val) in enumerate(kf.split(dataset_df)):
        logger.info(f"Processing fold {i+1}/{run_config.n_cv_folds}")
        train_df, val_df = dataset_df.iloc[train], dataset_df.iloc[val]
        # train_y, val_y = dataset_df.iloc[train]["vs30"], dataset_df.iloc[val]["vs30"]
        logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

        run_model_training(
            dataset_df,
            train_df,
            val_df,
            run_config,
            base_out_dir / f"cv_{i:02d}",
            cv_ix=i,
        )
        logger.info(f"Completed fold {i+1}/{run_config.n_cv_folds}")

    logger.info("Cross-validation training completed")

    # Combine model results into a single file
    val_results_dfs = []
    for i in range(run_config.n_cv_folds):
        fold_val_results_ffp = base_out_dir / f"cv_{i:02d}" / "val_results.parquet"
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
        pd.read_parquet(base_out_dir / f"cv_{i:02d}" / "train_metrics.parquet")
        for i in range(run_config.n_cv_folds)
    ]
    val_metric_dfs = [
        pd.read_parquet(base_out_dir / f"cv_{i:02d}" / "val_metrics.parquet")
        for i in range(run_config.n_cv_folds)
    ]

    # Save metrics
    train_metrics = xr.DataArray(
        np.stack([df.values for df in train_metric_dfs], axis=0),
        dims=["cv_fold", "iteration", "metric"],
        coords={
            "cv_fold": [f"cv_{i:02d}" for i in range(run_config.n_cv_folds)],
            "iteration": train_metric_dfs[0].index.values,
            "metric": train_metric_dfs[0].columns.values,
        },
    )
    train_metrics.to_netcdf(base_out_dir / "train_metrics.nc")
    val_metrics = xr.DataArray(
        np.stack([df.values for df in val_metric_dfs], axis=0),
        dims=["cv_fold", "iteration", "metric"],
        coords={
            "cv_fold": [f"cv_{i:02d}" for i in range(run_config.n_cv_folds)],
            "iteration": val_metric_dfs[0].index.values,
            "metric": val_metric_dfs[0].columns.values,
        },
    )
    val_metrics.to_netcdf(base_out_dir / "val_metrics.nc")

    # Generate post-processing plots
    if run_post_processing:
        logger.info("Running post-processing...")
        post_processing.gen_cv_iteration_metric_plots(
            base_out_dir, train_metrics, val_metrics=val_metrics
        )
        post_processing.gen_model_perfomance_plots(
            base_out_dir, results_df=val_results_df
        )
        post_processing.gen_spatial_plots(base_out_dir, results_df=val_results_df)

    run_config.to_yaml(base_out_dir / "run_config.yaml")


def full_train(run_config: RunConfig, out_dir: Path, run_post_processing: bool = True):
    """Runs training on the full dataset and saves results."""
    logger.info(f"Loading dataset from {run_config.dataset_ffp}")
    dataset_df = pd.read_parquet(run_config.dataset_ffp)
    logger.info(f"Dataset loaded with {len(dataset_df)} samples")

    run_model_training(
        dataset_df, dataset_df, None, run_config, out_dir, save_train_results=True
    )

    if run_post_processing:
        logger.info("Running post-processing...")
        train_results_df = pd.read_parquet(out_dir / "train_results.parquet")
        train_results_df = post_processing.add_residuals(train_results_df)
        train_results_df = post_processing.add_mae(train_results_df)
        train_results_df = post_processing.add_lnVs30_mse(train_results_df)
        post_processing.gen_model_perfomance_plots(out_dir, results_df=train_results_df)
        post_processing.gen_spatial_plots(out_dir, results_df=train_results_df)


def run_model_training(
    dataset_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    run_config: RunConfig,
    out_dir: Path,
    cv_ix: int | None = None,
    verbose: bool = False,
    save_train_results: bool = False,
) -> None:
    run_config = run_config.copy()

    # Don't modify the original dataframes
    train_df = train_df.copy()
    val_df = val_df.copy() if val_df is not None else None

    # Compute sample weighting
    train_df = pre_processing.add_sample_weights(train_df, run_config)

    # Pre-process
    train_X, scale_params = pre_processing.pre_process_features(
        train_df, run_config, pre_process_categorial=False
    )
    train_y = pre_processing.pre_process_vs30(train_df["vs30"])
    run_config.scale_params = scale_params
    assert train_df.index.equals(train_X.index)

    val_data = None
    if val_df is not None:
        val_X, _ = pre_processing.pre_process_features(
            val_df, run_config, pre_process_categorial=False
        )
        val_y = pre_processing.pre_process_vs30(val_df["vs30"])
        assert val_df.index.equals(val_X.index)
        val_data = (val_X, val_y)

    logger.info("Running model training")
    model = CatBoostRegressor(
        random_seed=run_config.seed,
        cat_features=run_config.categorial_variables,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        use_best_model=False,
        iterations=run_config.model_config.iterations,
    )
    model.fit(
        train_X,
        train_y,
        eval_set=val_data,
        verbose=verbose,
        sample_weight=train_df["sample_weight"].values,
    )

    out_dir.mkdir(parents=True, exist_ok=False)

    # Save iteration metrics
    eval_results = model.get_evals_result()
    pd.DataFrame(eval_results["learn"]).to_parquet(out_dir / "train_metrics.parquet")

    # Validation results
    if val_df is not None:
        # Get validation predictions and save results
        val_result_df = pd.DataFrame(
            index=val_y.index,
            data=dataset_df.loc[
                val_y.index, ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
            ],
        )
        val_result_df["station"] = val_result_df.index.astype(str)
        val_result_df["cv_ix"] = cv_ix
        val_result_df["pred_vs30"] = np.exp(model.predict(val_X))
        val_result_df.to_parquet(out_dir / "val_results.parquet")

        pd.DataFrame(eval_results["validation"]).to_parquet(
            out_dir / "val_metrics.parquet"
        )

    # Training results
    if save_train_results:
        train_result_df = pd.DataFrame(
            index=train_y.index,
            data=dataset_df.loc[
                train_y.index, ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
            ],
        )
        train_result_df["station"] = train_result_df.index.astype(str)
        train_result_df["cv_ix"] = cv_ix

        # Get training predictions
        train_result_df["pred_vs30"] = np.exp(model.predict(train_X))

        train_result_df.to_parquet(out_dir / "train_results.parquet")

    # Save results
    model.save_model(out_dir / "model.cbm")
    run_config.to_yaml(out_dir / "run_config.yaml")
