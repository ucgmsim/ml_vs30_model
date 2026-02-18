from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np







from .configs import RunConfig


def cv_train(config: RunConfig) -> None:
    df = pd.read_parquet(config.dataset_ffp)


    print("wtf")