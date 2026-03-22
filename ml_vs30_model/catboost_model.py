from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn import model_selection as ms

from catboost import CatBoostRegressor
from .configs import RunConfig
from . import pre_processing
from . import post_processing

logger = logging.getLogger(__name__)


def cv_train(
    run_config: RunConfig, base_out_dir: Path, run_post_processing: bool = True
) -> None:
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
            i,
            base_out_dir / f"cv_{i:02d}",
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

    if run_post_processing:
        logger.info("Running post-processing...")
        val_results_df = post_processing.add_residuals(val_results_df)
        val_results_df = post_processing.add_mae(val_results_df)
        val_results_df = post_processing.add_lnVs30_mse(val_results_df)
        post_processing.gen_model_perfomance_plots(
            base_out_dir, val_results_df=val_results_df
        )
        post_processing.gen_spatial_plots(base_out_dir, val_results_df=val_results_df)

    val_results_df.to_parquet(base_out_dir / "val_results.parquet")
    run_config.to_yaml(base_out_dir / "run_config.yaml")


def run_model_training(
    dataset_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_config: RunConfig,
    cv_ix: int,
    out_dir: Path,
    verbose: bool = False,
) -> None:
    run_config = run_config.copy()

    # Don't modify the original dataframes
    train_df = train_df.copy()
    val_df = val_df.copy()

    # Compute sample weighting
    train_df = pre_processing.add_sample_weights(train_df, run_config)

    # Pre-process
    train_X, scale_params = pre_processing.pre_process_features(
        train_df, run_config, pre_process_categorial=False
    )
    train_y = pre_processing.pre_process_vs30(train_df["vs30"])
    run_config.scale_params = scale_params
    assert train_df.index.equals(train_X.index)

    val_X, _ = pre_processing.pre_process_features(
        val_df, run_config, pre_process_categorial=False
    )
    val_y = pre_processing.pre_process_vs30(val_df["vs30"])
    assert val_df.index.equals(val_X.index)

    logger.info("Running model training")
    model = CatBoostRegressor(
        random_seed=run_config.seed,
        cat_features=run_config.categorial_variables,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    model.fit(
        train_X,
        train_y,
        eval_set=(val_X, val_y),
        verbose=verbose,
        sample_weight=train_df["sample_weight"].values,
    )

    val_result_df = pd.DataFrame(
        index=val_y.index,
        data=dataset_df.loc[
            val_y.index, ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
        ],
    )
    val_result_df["station"] = val_result_df.index.astype(str)
    val_result_df["cv_ix"] = cv_ix

    # Get validation predictions
    val_result_df["pred_vs30"] = np.exp(model.predict(val_X))

    # Save results
    out_dir.mkdir(parents=True, exist_ok=False)
    val_result_df.to_parquet(out_dir / "val_results.parquet")
    model.save_model(out_dir / "model.cbm")
    run_config.to_yaml(out_dir / "run_config.yaml")
