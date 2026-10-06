from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml_vs30_model.pre_processing import add_sample_weights


def test_bin_totals_equalised_and_q3_relative_weight():
    df = pd.DataFrame(
        {
            "vs30_bin": ["a"] * 6 + ["b"] * 4 + ["c"],
            "quality_score": ["Q1", "Q1"] + ["Q3"] * 8 + ["Q1"],
        }
    )
    cfg = SimpleNamespace(
        apply_quality_sample_weight_factor=True,
        q3_weight_factor=0.5,
        apply_vs30_sample_weights=True,
        max_vs30_weight=2.5,
    )
    w = add_sample_weights(df, cfg)["sample_weight"]

    # mixed bin: Q3 at half the Q1 weight; bin a is the largest (total 4.0)
    assert np.isclose(w[2] / w[0], 0.5)
    # all-Q3 bin b (raw total 2.0) is rescaled to match bin a, so Q3 is not penalised there
    assert np.isclose(w[df.vs30_bin == "b"].sum(), 4.0)
    # single-site bin c hits the cap
    assert np.isclose(w[10], 2.5)
