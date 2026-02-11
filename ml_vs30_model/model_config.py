from pathlib import Path
from dataclasses import dataclass
from enum import StrEnum

import ml_tools as mlt

from . import constants


class ModelType(StrEnum):
    NGBoost = "ngboost"


@dataclass
class ModelConfig:

    model_type: ModelType

    rel_dataset_ffp: str

    input_variables: list[str]  

    n_cv_folds: int 

    def __post_init__(self):
        self._dataset_ffp = None

    @property
    def dataset_ffp(self) -> Path:
        if self._dataset_ffp is None:
            self._dataset_ffp = constants.BASE_DATA_DIR / self.rel_dataset_ffp

        return self._dataset_ffp
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":

        config_dict["model_type"] = ModelType(config_dict["model_type"])

        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, config_ffp: Path) -> "ModelConfig":
        config_dict = mlt.utils.load_yaml(config_ffp)
        return cls.from_dict(config_dict)
    
    

    
