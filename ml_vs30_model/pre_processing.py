"""Module that contains functions for pre-processing of the features."""

import logging

import pandas as pd
import numpy as np

from .configs import RunConfig
from . import constants

logger = logging.getLogger(__name__)

def normalize(
    series: pd.Series, mean: float | None = None, std: float | None = None
) -> pd.Series:
    """
    Normalizes the given series using the
    provided mean and std, or computes them if not provided.
    """
    assert (mean is None) == (std is None)
    if mean is None:
        mean, std = float(series.mean()), float(series.std())

    return (series - mean) / std, (mean, std)


def min_max_scale(series: pd.Series, var: str | constants.InputVariable) -> pd.Series:
    """Scales the given series to the range [-1, 1] using min-max scaling."""
    cur_min, cur_max = constants.MIN_MAX_SCALE_PARAMS[var]
    return 2 * (series - cur_min) / (cur_max - cur_min) - 1


def pre_process_features(
    df: pd.DataFrame, run_config: RunConfig
) -> pd.DataFrame:
    """Performs pre-processing on the features in the given DataFrame."""
    scaled_input_df = pd.DataFrame(index=df.index)
    
    scale_params = run_config.scale_params
    if run_config.scale_params is None:
        scale_params = {} 

    for var in run_config.input_variables:
        if var in constants.LN_NORM_VARS:
            scaled_input_df[var], scale_params[var] = normalize(
                np.log1p(df[var]), *run_config.get_scale_params(var)
            )
        elif var in constants.NORM_VARS:
            scaled_input_df[var], scale_params[var] = normalize(
                df[var], *run_config.get_scale_params(var)
            )
        elif var in constants.MIN_MAX_SCALE_PARAMS:
            scaled_input_df[var] = min_max_scale(df[var], var)
        elif var in constants.CATEGORIAL_VARIABLES:
            if run_config.pre_process_categorial:
                one_hot_df = pd.get_dummies(df[var], prefix=var)
                scaled_input_df = pd.concat([scaled_input_df, one_hot_df], axis=1)
            else:
                scaled_input_df[var] = df[var]
        else:
            raise ValueError(f"Variable {var} is not categorized for pre-processing.")

    return scaled_input_df, scale_params


def pre_process_vs30(values: np.ndarray | pd.Series):
    """Pre-processes the target variable Vs30."""
    return np.log(values)


def add_sample_weights(train_df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
    """Computes sample weights based on the Vs30 values in the training DataFrame."""
    train_df.loc[:, "sample_weight"] = 1.0

    if run_config.apply_vs30_sample_weights:
        vs30_bin_counts = train_df["vs30_bin"].value_counts().sort_index()
        vs30_bin_weights = np.clip(
            (vs30_bin_counts.max() / vs30_bin_counts) - 1,
            0.0,
            run_config.max_vs30_weight,
        )

        train_df.loc[:, "vs30_weight"] = (
            train_df["vs30_bin"].map(vs30_bin_weights).astype(float)
        )

        train_df.loc[:, "sample_weight"] += train_df["vs30_weight"]
        
    return train_df


def get_pre_processed_train_val_df(
    dataset_df: pd.DataFrame,
    train_sites: np.ndarray,
    run_config: RunConfig,
    val_sites: np.ndarray | None = None,
):
    train_df = dataset_df.loc[train_sites].copy()
    val_df = dataset_df.loc[val_sites].copy() if val_sites is not None else None

    train_missing_mask = (train_df[run_config.input_variables].isna() | (train_df[run_config.input_variables] == -9999)).any(axis=1)
    if train_missing_mask.any():
        logger.warning(f"Missing values found in training data, dropping {train_missing_mask.sum()} rows.")
        train_df = train_df.loc[~train_missing_mask]

    if val_df is not None :
        val_missing_mask = (val_df[run_config.input_variables].isna() | (val_df[run_config.input_variables] == -9999)).any(axis=1)
        if val_missing_mask.any():
            logger.warning(f"Missing values found in validation data, dropping {val_missing_mask.sum()} rows.")
            val_df = val_df.loc[~val_missing_mask]

    train_df, val_df = train_df.copy(), val_df.copy() if val_df is not None else None

    # Compute sample weighting
    train_df = add_sample_weights(train_df, run_config)

    # Pre-process
    train_X, scale_params = pre_process_features(
        train_df, run_config
    )
    train_y = pre_process_vs30(train_df["vs30"])
    if run_config.scale_params is None:
        run_config = run_config.copy()
        run_config.scale_params = scale_params
    assert train_df.index.equals(train_X.index)

    val_X, val_y= None, None
    if val_df is not None:
        val_X, _ = pre_process_features(
            val_df, run_config
        )
        val_y = pre_process_vs30(val_df["vs30"])
        assert val_df.index.equals(val_X.index)

    return run_config, train_X, train_y, train_df, val_X, val_y, val_df
