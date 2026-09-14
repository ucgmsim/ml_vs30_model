"""Correctness checks for the NormalLabelNoise distribution's gradients and
Fisher information (ngboost_model.NormalLabelNoiseLogScore)."""

import numpy as np
from ngboost.distns.normal import NormalLogScore

from ml_vs30_model.ngboost_model import (
    NormalLabelNoise,
    NormalLabelNoiseLogScore,
    build_label_noise_y,
)

RNG = np.random.default_rng(0)
EPS = 1e-6


def _make_dist(loc, log_scale):
    return NormalLabelNoise(np.stack([loc, log_scale]))


def test_d_score_matches_finite_difference():
    n = 50
    loc = RNG.normal(size=n)
    log_scale = RNG.normal(scale=0.3, size=n)
    D = loc + RNG.normal(scale=0.5, size=n)
    sigma_L = np.abs(RNG.normal(scale=0.2, size=n)) + 1e-3
    Y = build_label_noise_y(D, sigma_L)
    score = NormalLabelNoiseLogScore()
    score.__dict__.update(_make_dist(loc, log_scale).__dict__)

    analytic = score.d_score(Y)

    for param_ix in range(2):
        params = [loc.copy(), log_scale.copy()]
        params[param_ix] = params[param_ix] + EPS
        plus = NormalLabelNoiseLogScore()
        plus.__dict__.update(_make_dist(*params).__dict__)

        params = [loc.copy(), log_scale.copy()]
        params[param_ix] = params[param_ix] - EPS
        minus = NormalLabelNoiseLogScore()
        minus.__dict__.update(_make_dist(*params).__dict__)

        numeric = (plus.score(Y) - minus.score(Y)) / (2 * EPS)
        assert np.allclose(numeric, analytic[:, param_ix], atol=1e-3)


def test_metric_matches_plain_normal_regardless_of_sigma_l():
    n = 20
    loc = RNG.normal(size=n)
    log_scale = RNG.normal(scale=0.3, size=n)
    dist = _make_dist(loc, log_scale)

    label_noise_score = NormalLabelNoiseLogScore()
    label_noise_score.__dict__.update(dist.__dict__)

    plain_score = NormalLogScore()
    plain_score.__dict__.update(dist.__dict__)

    assert np.allclose(label_noise_score.metric(), plain_score.metric())
