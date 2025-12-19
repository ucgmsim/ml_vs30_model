import logging
from pathlib import Path

import numpy as np
import pandas as pd

import rasterio

logger = logging.getLogger(__name__)

from .. import constants
from .. import utils

class TIFLoader:
    """Class for retrieving data from downloaded TIFF files."""

    SUPPORTED_VARIABLES = {
        constants.InputVariable.LandformShannonIndex,
        constants.InputVariable.LandformEntropy,
        constants.InputVariable.LandformUniformity,
        constants.InputVariable.AbsoluteDepthToBedrock,
    }

    VAR_TO_FILENAME_MAP = {
        constants.InputVariable.LandformShannonIndex: "geom_1KMsha_GMTEDmd.tif",
        constants.InputVariable.LandformEntropy: "geom_1KMent_GMTEDmd.tif",
        constants.InputVariable.LandformUniformity: "geom_1KMuni_GMTEDmd.tif",
        constants.InputVariable.AbsoluteDepthToBedrock: "absolute_depth_to_bedrock/BDTICM_M_250m_ll.tif"
    }

    def __init__(self, base_raw_data_dir: Path = constants.BASE_DATA_DIR / "input_data" / "raw") -> None:
        self.base_raw_data_dir = base_raw_data_dir

    def get_values(
        self, coords: np.ndarray, variable: constants.InputVariable
    ):
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by TIFLoader.",
                logger,
            )

        tif_ffp = self.base_raw_data_dir / self.VAR_TO_FILENAME_MAP[variable]
        if not tif_ffp.exists():
            utils.raise_log(
                FileNotFoundError,
                f"TIF file for variable {variable} not found at {tif_ffp}.",
                logger,
            )

        with rasterio.open(tif_ffp) as dataset:
            assert (
                dataset.crs.to_epsg() == constants.WGS84_EPSG
            ), "Dataset CRS is not WGS84."

            return np.concatenate(list(dataset.sample(coords)))

        



    

    