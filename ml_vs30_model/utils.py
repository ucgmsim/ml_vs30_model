import logging

import numpy as np
import pandas as pd
from pyproj import Geod

from . import constants
from .configs import RunConfig
from . import pre_processing


def raise_log(ex_type: Exception, error_msg: str, logger: logging.Logger) -> None:
    logger.error(error_msg)
    raise ex_type(error_msg)


def safe_cast(arr, dtype):
    arr = np.asarray(arr)
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)

    if np.any(arr < info.min) or np.any(arr > info.max):
        raise OverflowError("Value out of bounds for dtype")

    return arr.astype(dtype)



def get_vs30_weights(df: pd.DataFrame, max_weight: int) -> pd.DataFrame:
    """
    Computes the additional sample weight due to Vs30,
    to be added to the base weight of one.
    """
    if "vs30_bin" not in df.columns:
        df["vs30_bin"] = pd.cut(
            df.vs30,
            constants.VS30_WEIGHTING_BINS,
            labels=constants.VS30_WEIGHTING_BIN_NAMES,
        )

    vs30_bin_counts = df.vs30_bin.value_counts().sort_index()

    vs30_bin_weights = np.clip(
        (vs30_bin_counts.sum() / vs30_bin_counts) - 1, 0.0, max_weight
    )
    df["vs30_weight"] = df.vs30_bin.map(vs30_bin_weights).astype(np.float16)

    return df


def get_bounding_box_corners(
    lons: np.ndarray, lats: np.ndarray, width_m: float, height_m: float
) -> np.ndarray:
    """
    Vectorized version of get_bounding_box_corners for multiple centre points.

    Parameters
    ----------
    lons : ndarray
        Centre longitudes in decimal degrees, shape (N,).
    lats : ndarray
        Centre latitudes in decimal degrees, shape (N,).
    width_m : float
        Width of the bounding box in metres (same for all points).
    height_m : float
        Height of the bounding box in metres (same for all points).

    Returns
    -------
    ndarray
        A 3D numpy array with shape (N, 4, 2) containing [lon, lat] coordinates
        of the four corners per point:
            - upper left
            - lower left
            - upper right
            - lower right
    """
    geod = Geod(ellps="WGS84")
    n_points = lons.shape[0]

    half_w = np.ones(n_points) * width_m / 2
    half_h = np.ones(n_points) * height_m / 2

    top_lons, top_lats, _ = geod.fwd(lons, lats, np.zeros(n_points), half_h)
    bottom_lons, bottom_lats, _ = geod.fwd(lons, lats, np.ones(n_points) * 180, half_h)
    right_lons, right_lats, _ = geod.fwd(lons, lats, np.ones(n_points) * 90, half_w)
    left_lons, left_lats, _ = geod.fwd(lons, lats, np.ones(n_points) * 270, half_w)

    # Stack into (N, 4, 2)
    return np.stack(
        [
            np.column_stack([left_lons, top_lats]),  # upper left
            np.column_stack([left_lons, bottom_lats]),  # lower left
            np.column_stack([right_lons, top_lats]),  # upper right
            np.column_stack([right_lons, bottom_lats]),  # lower right
        ],
        axis=1,
    )


