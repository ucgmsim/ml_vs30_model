from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np







from .model_config import ModelConfig


def cv_train(config: ModelConfig) -> None:
    df = pd.read_parquet(config.dataset_ffp)


    print("wtf")