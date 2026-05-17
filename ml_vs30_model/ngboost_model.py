import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor

import ml_tools as mlt

from .configs import RunConfig
from . import pre_processing
from . import training
from . import post_processing

logger = logging.getLogger(__name__)


def cv_train(
    run_config: RunConfig, base_out_dir: Path, run_post_processing: bool = True
) -> None:
    """Runs cross-validation training of the ngboost model."""
    training.cv_train(
        run_model_training,
        run_config,
        base_out_dir,
        run_post_processing=run_post_processing,
        compute_shap=True,
    )


def run_model_training(
    dataset_df: pd.DataFrame,
    train_sites: np.ndarray,
    val_sites: np.ndarray,
    run_config: RunConfig,
    out_dir: Path,
    cv_ix: int | None = None,
    verbose: bool = False,
    save_train_results: bool = False,
    compute_shap: bool = False,
) -> None:
    """Runs training of the ngboost model on the provided dataset."""
    assert (
        np.isin(run_config.test_sites, dataset_df.index.values.astype(str)).sum() == 0
    ), "Test sites must not be present in the training dataset"

    if len(run_config.categorial_variables) > 0:
        raise NotImplementedError(
            "Pre-processing of categorial variables is not implemented yet"
        )

    run_config, train_X, train_y, train_df, val_X, val_y, val_df = (
        pre_processing.get_pre_processed_train_val_df(
            dataset_df,
            train_sites,
            run_config,
            val_sites=val_sites,
        )
    )

    logger.info("Running model training")
    ngb = NGBRegressor(
        Dist=Normal,
        Base=DecisionTreeRegressor(max_depth=5, min_samples_leaf=5),
        random_state=run_config.seed,
        n_estimators=run_config.model_config.iterations,
        verbose=verbose,
    )
    ngb.fit(
        train_X,
        train_y,
        X_val=val_X if val_X is not None else None,
        Y_val=val_y if val_y is not None else None,
        sample_weight=train_df["sample_weight"].values,
    )

    out_dir.mkdir(parents=True, exist_ok=False)

    # Iteration metrics
    pd.DataFrame(ngb.evals_result["train"]).to_parquet(
        out_dir / "train_metrics.parquet"
    )

    # Validation results
    val_results_df = None
    if val_df is not None:
        val_result_df = pd.DataFrame(
            index=val_y.index,
            data=dataset_df.loc[
                val_y.index, ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
            ],
        )
        val_result_df["station"] = val_result_df.index.astype(str)
        val_result_df["cv_ix"] = cv_ix

        val_pred = ngb.pred_dist(val_X).params
        val_result_df["pred_vs30"] = np.exp(val_pred["loc"])
        val_result_df["pred_vs30_std"] = val_pred["scale"]
        val_result_df.to_parquet(out_dir / "val_results.parquet")

        pd.DataFrame(ngb.evals_result["val"]).to_parquet(
            out_dir / "val_metrics.parquet"
        )

    # Training results
    train_result_df = pd.DataFrame(
        index=train_y.index,
        data=dataset_df.loc[
            train_y.index, ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
        ],
    )
    if save_train_results:
        train_result_df["station"] = train_result_df.index.astype(str)
        train_result_df["cv_ix"] = cv_ix

        # Get training predictions
        train_pred = ngb.pred_dist(train_X).params
        train_result_df["pred_vs30"] = np.exp(train_pred["loc"])
        train_result_df["pred_vs30_std"] = train_pred["scale"]

        train_result_df.to_parquet(out_dir / "train_results.parquet")

    # Compute SHAP values
    if compute_shap:
        post_processing.compute_shap_feature_importance(
            out_dir,
            run_config=run_config,
            train_results=train_result_df,
            val_results=val_results_df,
            model=ngb,
        )

    # Save model and config
    mlt.utils.write_pickle(ngb, out_dir / "model.pkl")
    run_config.to_yaml(out_dir / "run_config.yaml")
