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
        constants.InputVariable.SubrockMinProxy,
        constants.InputVariable.SubrockMeanProxy,
        constants.InputVariable.SubrockMedianProxy,
        constants.InputVariable.SubrockMaxProxy,
    }

    def __init__(self, data_df: pd.DataFrame, allow_missing: bool = False):
        self.data_df = data_df.copy()
        self._subrock_ranks = None
        self._allow_missing = allow_missing

    def _get_subrock_ranks(self) -> pd.Series:
        """Per-row list of matched grain-rank scores parsed from NZSubRocks.

        Cached on the instance (not written to data_df) so repeated calls for
        the different subrock proxy aggregates don't re-parse SUBROCKS text.
        """

        def _parse(value):
            return (
                [
                    constants.ROCK_GRAIN_RANK[cur_subrock]
                    for cur_subrock in (t.strip().lower() for t in value.split(","))
                    if cur_subrock in constants.ROCK_GRAIN_RANK
                ]
                if isinstance(value, str)
                else []
            )

        if self._subrock_ranks is None:
            self._subrock_ranks = self.data_df[
                constants.InputVariable.NZSubRocks
            ].apply(_parse)

        return self._subrock_ranks

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

            self.data_df = self._compute_values(variable)
            if not self._allow_missing:
                assert (
                    self.data_df[variable].notna().all()
                ), f"NaN values found in computed variable {variable}."

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
            if constants.InputVariable.NZGeologyAgeMid not in self.data_df.columns:
                self._compute_values(constants.InputVariable.NZGeologyAgeMid)

            self.data_df[variable] = np.log(
                self.data_df[constants.InputVariable.NZGeologyAgeMid]
            )
        elif variable == constants.InputVariable.NZCombinedGroundwaterDepth:
            # Fill NLM nan values with NWT values
            self.data_df[variable] = self.data_df[
                constants.InputVariable.NZNLMGroundwaterDepth
            ].combine_first(self.data_df[constants.InputVariable.NZNWTGroundwaterDepth])

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
                self.data_df[constants.InputVariable.NZCombinedGroundwaterDepth].clip(
                    lower=np.exp(-2)
                )
            )
        elif variable == constants.InputVariable.MainrockProxy:
            self.data_df[variable] = (
                self.data_df[constants.InputVariable.NZMainRock]
                .str.strip()
                .str.lower()
                .map(constants.ROCK_GRAIN_RANK)
            )
        elif variable in [
            constants.InputVariable.SubrockMinProxy,
            constants.InputVariable.SubrockMeanProxy,
            constants.InputVariable.SubrockMedianProxy,
            constants.InputVariable.SubrockMaxProxy,
        ]:
            self.data_df[variable] = self._subrock_helper(variable)
        else:
            utils.raise_log(
                NotImplementedError,
                f"Implementation missing for variable {variable} in FeatureEngineer.",
                logger,
            )

        return self.data_df

    def _subrock_helper(self, subrock_variable: constants.InputVariable):
        subrock_proxy_agg_fn_mapping = {
            constants.InputVariable.SubrockMinProxy: min,
            constants.InputVariable.SubrockMeanProxy: np.mean,
            constants.InputVariable.SubrockMedianProxy: np.median,
            constants.InputVariable.SubrockMaxProxy: max,
        }
        assert (
            subrock_variable in subrock_proxy_agg_fn_mapping.keys()
        ), f"Invalid subrock variable: {subrock_variable}"

        # Get mainrock proxy values if not already
        if constants.InputVariable.MainrockProxy not in self.data_df.columns:
            self._compute_values(constants.InputVariable.MainrockProxy)

        return [
            subrock_proxy_agg_fn_mapping[subrock_variable](ranks) if ranks else mainrock
            for ranks, mainrock in zip(
                self._get_subrock_ranks(),
                self.data_df[constants.InputVariable.MainrockProxy],
            )
        ]
