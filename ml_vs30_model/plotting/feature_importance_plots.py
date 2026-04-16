import logging
from pathlib import Path

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import ml_tools as mlt


from .. import constants

logger = logging.getLogger(__name__)


def shap_global(
    shap_values: shap.Explanation, results_df: pd.DataFrame, out_dir: Path
) -> None:
    """
    Generates SHAP global feature importance plots
    """
    # Global feature importance
    out_fp = out_dir / "shap_global.png"
    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)
    shap.plots.bar(shap_values, show=False, ax=ax)
    fig.tight_layout()
    fig.savefig(out_fp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        {"type": "feature-importance", "method": "shap-global"},
        out_fp.with_suffix(".yaml"),
        clobber=True,
    )

    # Feature importance per vs30 bin
    order = np.argsort(np.abs(shap_values.values).mean(axis=0))[::-1]
    out_fp = out_dir / "shap_global_vs30Bin_per_bin.png"
    fig, axs = mlt.plotting.get_fig_axes(
        len(results_df.vs30_bin.cat.categories),
        2,
        -1,
        ind_figsize=(8, 5),
        dpi=constants.FIG_DPI,
    )
    x_max = -np.inf
    for i, (vs30_bin, ax) in enumerate(zip(results_df.vs30_bin.cat.categories, axs)):
        cur_shap_values = shap_values[(results_df.vs30_bin == vs30_bin).values]
        shap.plots.bar(cur_shap_values, show=False, ax=ax, order=order)
        x_max = max(x_max, ax.get_xlim()[1])

        if i % 2 != 0:
            ax.set_yticklabels([])
        if i < len(results_df.vs30_bin.cat.categories) - 2:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        ax.text(
            0.98,
            0.02,
            f"Vs30 bin: {vs30_bin}",
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
        )

    [ax.set_xlim(0, x_max) for ax in axs]

    fig.subplots_adjust(
        wspace=0.075, hspace=0.05, right=0.98, top=0.98, bottom=0.03, left=0.2
    )
    fig.savefig(out_fp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        {"type": "feature-importance", "method": "shap-global-vs30Bin-per-bin"},
        out_fp.with_suffix(".yaml"),
        clobber=True,
    )

    # Feature importance per vs30 bin (combined)
    out_fp = out_dir / "shap_global_vs30Bin.png"
    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)
    shap.plots.bar(
        shap_values.cohorts(results_df.vs30_bin.values.astype(str)), show=False, ax=ax
    )
    fig.tight_layout()
    fig.savefig(out_fp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        {"type": "feature-importance", "method": "shap-global-vs30Bin"},
        out_fp.with_suffix(".yaml"),
        clobber=True,
    )


def shap_beeswarm(
    shap_values: shap.Explanation, results_df: pd.DataFrame, out_dir: Path
) -> None:
    """
    Generates SHAP beeswarm plot
    """
    out_fp = out_dir / "shap_beeswarm.png"
    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)
    shap.plots.beeswarm(shap_values, show=False, ax=ax, plot_size=None)
    fig.tight_layout()
    fig.savefig(out_fp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        {"type": "feature-importance", "method": "shap-beeswarm"},
        out_fp.with_suffix(".yaml"),
        clobber=True,
    )


def shap_waterfall(
    shap_values: shap.Explanation, results_df: pd.DataFrame, out_dir: Path
):
    """
    Generates SHAP waterfall plots for each sample, grouped by Vs30 bin.
    """
    logger.info("Generating SHAP waterfall plots...")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(shap_values.shape[0])):
        out_fp = out_dir / f"{results_df.index[i]}.png"

        ax = shap.plots.waterfall(shap_values[i], show=False)
        ax.axvline(
            np.log(results_df.iloc[i]["vs30"]), color="red", linestyle="--"
        )
        ax.text(
            -0.0,
            1.04,
            f"{results_df.index[i]} - "
            f"Vs30: {results_df.iloc[i]['vs30']:.1f}, "
            f"Predicted: {results_df.iloc[i]['pred_vs30']:.1f}",
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=12,
            fontweight="bold",
        )

        fig = plt.gcf()
        fig_size = fig.get_size_inches()
        fig.set_size_inches(10, fig_size[1])

        fig.tight_layout()
        plt.savefig(out_fp)
        plt.close()

        mlt.utils.write_to_yaml(
            {
                "type": "feature-importance",
                "method": "shap-waterfall",
                "vs30_bin": results_df.iloc[i]["vs30_bin"],
                "ln_residual": float(results_df.iloc[i]["ln_residual"]),
                "abs_ln_residual": float(abs(results_df.iloc[i]["ln_residual"])),
            },
            out_fp.with_suffix(".yaml"),
            clobber=True,
        )
