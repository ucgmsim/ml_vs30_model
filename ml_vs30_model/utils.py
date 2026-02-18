import logging

import numpy as np
import pandas as pd

from . import constants

def raise_log(ex_type: Exception, error_msg: str, logger: logging.Logger) -> None:
    logger.error(error_msg)
    raise ex_type(error_msg)


def get_vs30_weights(df: pd.DataFrame, max_weight: int) -> pd.DataFrame:
    """
    Computes the additional sample weight due to Vs30,
    to be added to the base weight of one.
    """
    df["vs30_bin"] = pd.cut(
        df.vs30,
        constants.VS30_WEIGHTING_BINS,
        labels=constants.VS30_WEIGHTING_BIN_NAMES,
    )

    vs30_bin_counts = df.vs30_bin.value_counts().sort_index()

    vs30_bin_weights = np.clip(
        (vs30_bin_counts.sum() / vs30_bin_counts) - 1, 0.0, max_weight
    )
    df["vs30_weight"] = df.vs30_bin.map(
        vs30_bin_weights
    ).astype(np.float16)

    return df