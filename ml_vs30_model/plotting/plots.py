import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from .. import constants

logger = logging.getLogger(__name__)

def plot_nz_vs30_hist(vs30_values: np.ndarray, out_ffp: Path, n_bins: int = 50):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=constants.FIG_DPI)
    ax.hist(vs30_values, bins=n_bins, color="tab:blue", edgecolor="black", density=False)

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("Vs30 (m/s)")
    ax.set_ylabel("Count")
    # ax.set_ylabel("Density")
    ax.set_xlim(0, 1600)

    ax.text(
        0.02,
        0.98,
        r"NZ $V_{S30}$ Estimate" + f"\n{vs30_values.size:,} Sites",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontweight="bold",  
    )

    fig.tight_layout()
    fig.savefig(out_ffp, dpi=constants.FIG_DPI)
    plt.close(fig)
