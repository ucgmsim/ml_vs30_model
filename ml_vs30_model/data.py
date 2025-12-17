import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ml_tools as mlt

from . import constants
from .geomorpho90 import GeoMorpho90

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:

    rel_vs30_values_ffp: str
    input_variables: list[constants.InputVariable]

    def __post_init__(self):
        
        self._vs30_values_ffp = None

    @property
    def vs30_values_ffp(self) -> Path:
        if self._vs30_values_ffp is None:
            self._vs30_values_ffp = constants.BASE_DATA_DIR / self.rel_vs30_values_ffp

        return self._vs30_values_ffp

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DataConfig":
        return cls(
            rel_vs30_values_ffp=config_dict["vs30_values_ffp"],
            input_variables=config_dict["input_variables"],
        )
    
    @classmethod
    def from_yaml(cls, config_ffp: Path) -> "DataConfig":
        config_dict = mlt.utils.load_yaml(config_ffp)
        config_dict["input_variables"] = [
            constants.InputVariable(var_str) for var_str in config_dict["input_variables"]
        ]

        return cls.from_dict(config_dict)
    

def gen_dataset(data_config: DataConfig, out_ffp: Path) -> None:
    vs30_values_df = pd.read_csv(data_config.vs30_values_ffp)
    assert np.all(np.isin(["lon", "lat", "vs30"], vs30_values_df.columns))

    df = vs30_values_df[["lon", "lat", "vs30"]].copy()
    for variable in data_config.input_variables:
        logger.info(f"Processing variable: {variable.value}")
        data_source = constants.INPUT_VARIABLE_SOURCE_MAPPING.get(variable)

        if data_source == constants.DataSource.GeoMorpho90:
            logger.info(f"Using GeoMorpho90 data source for variable: {variable.value}")
            geomorpho90 = GeoMorpho90()
            values = geomorpho90.get_values(
                vs30_values_df[["lon", "lat"]].to_numpy(), variable
            )
            df[variable.value] = values
        else:
            error_msg = f"Data source for variable {variable} not implemented."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        
    df.to_parquet(out_ffp)
    logger.info(f"Dataset saved to {out_ffp}")

