from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ml_tools as mlt

from .. import constants


def one_to_one_plot(results_df: pd.DataFrame, output_ffp: Path):
    """
    Generates a one-to-one plot comparing true vs30 values to predicted vs30 values,
    and saves it to the specified output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    for vs30_bin in constants.VS30_WEIGHTING_BIN_NAMES:
        mask = results_df["vs30_bin"] == vs30_bin
        ax.scatter(
            results_df.loc[mask, "vs30"],
            results_df.loc[mask, "pred_vs30"],
            label=rf"{vs30_bin}, $\mu$={results_df.loc[mask, 'ln_residual'].mean():.3f}, "
            rf"$\sigma$={results_df.loc[mask, 'ln_residual'].std():.3f} (N={mask.sum()})",
            alpha=0.5,
        )

    ax.plot(
        [0, 1550],
        [0, 1550],
        "k",
    )

    ax.set_xlim(0, 1550)
    ax.set_ylim(0, 1550)
    ax.set_xlabel("True vs30")
    ax.set_ylabel("Predicted vs30")
    ax.legend(title="Vs30 Bin")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(type="one-to-one"),
        output_ffp.with_name(output_ffp.stem + ".yaml"),
        clobber=True,
    )


def residual_kde(results_df: pd.DataFrame, output_ffp: Path):
    """
    Generates a kernel density estimate (KDE) plot of the residuals (ln(vs30) - ln(predicted_vs30)),
    and saves it to the specified output file path.
    """
    x_limits = (-1.5, 1.5)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)

    sns.kdeplot(
        data=results_df,
        x="ln_residual",
        hue="vs30_bin",
        ax=ax,
        fill=True,
        common_norm=False,
        alpha=0.5,
        clip=x_limits,
    )

    # Legend
    bin_groups = results_df.groupby("vs30_bin", observed=True)
    bias_by_bin = bin_groups["ln_residual"].mean()
    res_std_by_bin = bin_groups["ln_residual"].std()
    legend = ax.get_legend()
    for text in legend.get_texts():
        label = text.get_text()
        if label in bias_by_bin.index:
            text.set_text(
                rf"{label}: $\mu$={bias_by_bin[label]:.3f}, "
                rf"$\sigma$={res_std_by_bin[label]:.3f} (N={bin_groups.size()[label]})"
            )
    legend.set_title("Vs30 Bin")

    ax.text(
        0.99,
        0.5,
        "Underprediction",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="center",
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.5,
        "Overprediction",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="center",
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.99,
        rf"Total - $\mu$={results_df['ln_residual'].mean():.3f}, $\sigma$={results_df['ln_residual'].std():.3f}",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=constants.FIG_FONT_SIZE,
    )

    ax.set_xlim(x_limits)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(type="residual-kde"),
        output_ffp.with_name(output_ffp.stem + ".yaml"),
        clobber=True,
    )


def residuals_histogram(results_df: pd.DataFrame, output_ffp: Path):
    """
    Generates a histogram of the residuals (ln(vs30) - ln(predicted_vs30)),
    and saves it to the specified output file path.
    """
    x_limits = (-1.5, 1.5)
    bins = np.linspace(x_limits[0], x_limits[1], 30)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)

    sns.histplot(
        data=results_df,
        x="ln_residual",
        bins=bins,
        ax=ax,
        color="skyblue",
        edgecolor="k",
        hue="vs30_bin",
        multiple="stack",
    )

    # Legend
    bin_groups = results_df.groupby("vs30_bin", observed=True)
    bias_by_bin = bin_groups["ln_residual"].mean()
    res_std_by_bin = bin_groups["ln_residual"].std()
    legend = ax.get_legend()
    for text in legend.get_texts():
        label = text.get_text()
        if label in bias_by_bin.index:
            text.set_text(
                rf"{label}: $\mu$={bias_by_bin[label]:.3f}, "
                rf"$\sigma$={res_std_by_bin[label]:.3f} (N={bin_groups.size()[label]})"
            )
    legend.set_title("Vs30 Bin")

    ax.text(
        0.99,
        0.5,
        "Underprediction",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="center",
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.5,
        "Overprediction",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="center",
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.99,
        rf"Total - $\mu$={results_df['ln_residual'].mean():.3f}, $\sigma$={results_df['ln_residual'].std():.3f}",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=constants.FIG_FONT_SIZE,
    )

    ax.set_xlim(x_limits)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(type="residual-histogram"),
        output_ffp.with_name(output_ffp.stem + ".yaml"),
        clobber=True,
    )


def metric_scatter_plot(
    results_df: pd.DataFrame,
    output_ffp: Path,
    metric_name: str,
    y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
    show_geyin_maurer_model: bool = False,
):
    """
    Generates a scatter plot of the specified metric (e.g., MAE) vs true vs30 values,
    and saves it to the specified output file path.

    A trend line is added to the plot to show how the metric varies with vs30.
    """
    # Limits
    scatter_options = mlt.plotting.ScatterOptions(
        "vs30",
        metric_name,
        binning_method=mlt.plotting.BinningMethod.EqualCount,
        trend_n_data_points=50,
        trend_n_bins=None,
        alpha=0.25,
        color="blue",
        trend_color="red",
    )

    fig, ax = mlt.plotting.gen_scatter_trend_plot(
        results_df, scatter_options, dpi=constants.FIG_DPI
    )

    ax.set_xlabel("True vs30")
    if y_limits is not None:
        ax.set_ylim(y_limits)
    if x_limits is not None:
        ax.set_xlim(x_limits)

    if metric_name == "ln_residual":
        ax.axhline(0, color="k", linewidth=1, zorder=0)

    if metric_name == "mae":
        ax.set_ylabel("Mean Absolute Error (MAE)")
        if show_geyin_maurer_model:
            ax.plot(
                constants.GEYIN_MAURER_MODEL_MAE.keys(),
                constants.GEYIN_MAURER_MODEL_MAE.values(),
                marker="x",
                color="green",
                label="G&M Model MAE (Test)",
            )

    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(type="metric-scatter", metric=metric_name),
        output_ffp.with_name(output_ffp.stem + ".yaml"),
        clobber=True,
    )
