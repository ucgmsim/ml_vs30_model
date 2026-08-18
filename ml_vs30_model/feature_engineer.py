"""Class for creating new input variables from existing raw data variables."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from . import constants
from . import utils

logger = logging.getLogger(__name__)


class FeatureEngineer:

    SUPPORTED_VARIABLES = {
        constants.InputVariable.NZGeologyAgeMid,
        constants.InputVariable.NZGeologyAgeLnMid,
        constants.InputVariable.NZCombinedGroundwaterDepth,
        constants.InputVariable.NZCombinedGroundwaterDepthLn,
        constants.InputVariable.MainrockProxy,
    }

    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df.copy()

    def compute_features(
        self, variables: list[constants.InputVariable]
    ) -> pd.DataFrame:
        """Computes the specified derived variables and adds them to the DataFrame."""
        for variable in variables:
            if variable not in self.SUPPORTED_VARIABLES:
                utils.raise_log(
                    ValueError,
                    f"Variable {variable} is not supported by FeatureEngineer.",
                    logger,
                )

            self._compute_values(variable)

        return self.data_df

    def _compute_values(self, variable: constants.InputVariable) -> pd.DataFrame:
        """Create new features from existing raw data variables."""
        if variable == constants.InputVariable.NZGeologyAgeMid:
            self.data_df[variable] = (
                self.data_df[constants.InputVariable.NZGeologyAgeMin]
                + self.data_df[constants.InputVariable.NZGeologyAgeMax]
            ) / 2
        elif variable == constants.InputVariable.NZGeologyAgeLnMid:
            # Compute mid-age if not already computed
            if (
                constants.InputVariable.NZGeologyAgeMid
                not in self.data_df.columns
            ):
                self._compute_values(constants.InputVariable.NZGeologyAgeMid)

            self.data_df[variable] = np.log(
                self.data_df[constants.InputVariable.NZGeologyAgeMid]
            )
        elif variable == constants.InputVariable.NZCombinedGroundwaterDepth:
            # Fill NLM nan values with NWT values
            self.data_df[variable] = self.data_df[
                constants.InputVariable.NZNLMGroundwaterDepth
            ].combine_first(
                self.data_df[constants.InputVariable.NZNWTGroundwaterDepth]
            )

            # Deal with any remaining nan values (if any)
            if self.data_df[variable].isna().any():
                logger.info(
                    f"{self.data_df[variable].isna().sum()} NaN values remain in {variable} "
                    "after combining NLM and NWT values. Filling remaining with global data."
                )
                self.data_df[variable] = self.data_df[variable].combine_first(
                    self.data_df[constants.InputVariable.DepthToGroundwater].abs()
                )

        elif variable == constants.InputVariable.NZCombinedGroundwaterDepthLn:
            # Compute combined groundwater depth if not already computed
            if (
                constants.InputVariable.NZCombinedGroundwaterDepth
                not in self.data_df.columns
            ):
                self._compute_values(constants.InputVariable.NZCombinedGroundwaterDepth)

            self.data_df[variable] = np.log(
                self.data_df[constants.InputVariable.NZCombinedGroundwaterDepth].clip(lower=np.exp(-2))
            )
        elif variable == constants.InputVariable.MainrockProxy:
            self.data_df[variable] = (
                self.data_df[constants.InputVariable.NZMainRock]
                .str.strip()
                .str.lower()
                .map(constants.MAINROCK_GRAIN_RANK)
            )
        else:
            utils.raise_log(
                NotImplementedError,
                f"Implementation missing for variable {variable} in FeatureEngineer.",
                logger,
            )
