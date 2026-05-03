import copy
from pathlib import Path
import dataclasses
from enum import StrEnum

import ml_tools as mlt

from . import constants


class ModelType(StrEnum):
    MLP = "mlp"
    NGBoost = "ngboost"
    CatBoost = "catboost"

@dataclasses.dataclass  
class CatboostModelConfig:
    iterations: int

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CatboostModelConfig":
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        return {
            "iterations": int(self.iterations),
        }

@dataclasses.dataclass
class RunConfig:

    seed: int

    model_type: ModelType | str

    rel_dataset_ffp: str
    input_variables: list[str]  

    apply_vs30_sample_weights: bool
    max_vs30_weight: float

    n_cv_folds: int 
    """Number of CV folds to use. Only applicable when using CV."""

    rel_results_dir: str

    model_config: CatboostModelConfig
    pre_process_categorial: bool

    _scale_params: dict = None

    def __post_init__(self):
        self._dataset_ffp = None

    @property
    def dataset_ffp(self) -> Path:
        if self._dataset_ffp is None:
            self._dataset_ffp = constants.BASE_DATA_DIR / self.rel_dataset_ffp

        return self._dataset_ffp

    @property  
    def results_dir(self) -> Path:
        return constants.BASE_DATA_DIR / self.rel_results_dir    
    
    @property
    def categorial_variables(self) -> list[constants.InputVariable]:
        return [var for var in self.input_variables if var in constants.CATEGORIAL_VARIABLES]

    @property
    def scale_params(self):
        return self._scale_params

    @scale_params.setter
    def scale_params(self, value):
        if self._scale_params is None:
            self._scale_params = value
        else:
            raise ValueError("Scale parameters are already set")
        
    def get_scale_params(self, var: str | constants.InputVariable):
        if self._scale_params is None:
            return None, None
        elif var not in self._scale_params:
            raise ValueError(f"Scale parameters for variable {var} are not set")
        else:
            return self._scale_params[var]

    def copy(self):
        return copy.deepcopy(self)  

    @classmethod
    def from_config_kwargs(cls, config_ffp: Path, **kwargs):
        """
        Creates an instance from the given config.
        If kwargs are set then they overwrite the values
        specified in the config.
        """
        config_dict = mlt.utils.load_yaml(config_ffp)

        for cur_key, cur_val in kwargs.items():
            if cur_val is None: 
                continue

            if cur_key == "iterations":
                config_dict["model_config"]["iterations"] = cur_val
            else:
                config_dict[cur_key] = cur_val

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RunConfig":
        """Creates an instance from the given config dictionary."""
        config_dict["model_type"] = ModelType(config_dict["model_type"])
        config_dict["model_config"] = CatboostModelConfig.from_dict(config_dict["model_config"])

        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, config_ffp: Path) -> "RunConfig":
        """Creates an instance from the given YAML config file."""
        config_dict = mlt.utils.load_yaml(config_ffp)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """Converts the RunConfig instance to a dictionary."""
        config_dict = {
            "seed": int(self.seed),
            "model_type": str(self.model_type),
            "rel_dataset_ffp": str(self.rel_dataset_ffp),
            "input_variables": list(self.input_variables),
            "n_cv_folds": int(self.n_cv_folds),
            "rel_results_dir": str(self.rel_results_dir),
            "model_config": self.model_config.to_dict(),
            "apply_vs30_sample_weights": bool(self.apply_vs30_sample_weights),
            "max_vs30_weight": float(self.max_vs30_weight),
            "pre_process_categorial": bool(self.pre_process_categorial),
        }

        if self._scale_params is not None:
            config_dict["_scale_params"] = self._scale_params

        return config_dict

    def to_yaml(self, ffp: Path):
        """Save the RunConfig to a YAML file."""
        mlt.utils.write_to_yaml(self.to_dict(), ffp)
    
    

    
