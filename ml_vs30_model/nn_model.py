from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import model_selection as ms

from catboost import CatBoostRegressor
import ml_tools as mlt
import shap

from .configs import RunConfig
from . import pre_processing
from . import post_processing
from . import training

logger = logging.getLogger(__name__)


def cv_train(
    run_config: RunConfig, base_out_dir: Path, run_post_processing: bool = True
) -> None:
    """Runs cross-validation training of the neural network model."""
    training.cv_train(
        run_model_training,
        run_config,
        base_out_dir,
        run_post_processing=run_post_processing,
    )


def run_model_training(
    dataset_df: pd.DataFrame,
    train_sites: np.ndarray,
    val_sites: np.ndarray,
    run_config: RunConfig,
    fold_out_dir: Path,
    cv_ix: int,
    save_train_results: bool = False,
) -> None:
    """Runs training of the neural network model."""
    assert (
        np.isin(run_config.test_sites, dataset_df.index.values.astype(str)).sum() == 0
    ), "Test sites must not be present in the training dataset"

    run_config, train_X, train_y, train_df, val_X, val_y, val_df = (
        pre_processing.get_pre_processed_train_val_df(
            dataset_df,
            train_sites,
            run_config,
            val_sites=val_sites,
        )
    )

    print("wtf")
