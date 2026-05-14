from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ml_tools as mlt

from .. import constants

METRIC_Y_LIMITS = {
    "RMSE": (0, 0.8),
}


def one_to_one_plot(
    results_df: pd.DataFrame, output_ffp: Path, quality_score: str | None = None
):
    """
    Generates a one-to-one plot comparing true vs30 values to predicted vs30 values,
    and saves it to the specified output file path.
    """
    # Apply quality score filter
    if quality_score is not None:
        results_df = results_df.loc[results_df["quality_score"] == quality_score]

    # Vs30
    fig, ax = plt.subplots(figsize=(10, 10))
    for vs30_bin, color in zip(
        constants.VS30_WEIGHTING_BIN_NAMES, constants.V30_BIN_COLORS
    ):
        mask = results_df["vs30_bin"] == vs30_bin
        ax.scatter(
            results_df.loc[mask, "vs30"],
            results_df.loc[mask, "pred_vs30"],
            label=rf"{vs30_bin}, $\mu$={results_df.loc[mask, 'ln_residual'].mean():.3f}, "
            rf"$\sigma$={results_df.loc[mask, 'ln_residual'].std():.3f} (N={mask.sum()})",
            alpha=0.5,
            color=color,
        )
    ax.plot(
        [0, 1550],
        [0, 1550],
        "k",
    )

    ax.text(
        0.5,
        0.98,
        (
            f"Vs30 - Quality Score: {quality_score}"
            if quality_score is not None
            else "Vs30"
        ),
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        fontweight="bold",
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

    # Metadata
    meta_dict = dict(type="one-to-one", variable="vs30")
    if quality_score is not None:
        meta_dict["quality_score"] = quality_score
    mlt.utils.write_to_yaml(
        meta_dict,
        output_ffp.with_name(output_ffp.stem + ".yaml"),
        clobber=True,
    )

    # Ln(Vs30)
    ln_min, ln_max = 4.5, 7.5
    fig, ax = plt.subplots(figsize=(10, 10))
    for vs30_bin, color in zip(
        constants.VS30_WEIGHTING_BIN_NAMES, constants.V30_BIN_COLORS
    ):
        mask = results_df["vs30_bin"] == vs30_bin
        ax.scatter(
            np.log(results_df.loc[mask, "vs30"]),
            np.log(results_df.loc[mask, "pred_vs30"]),
            label=rf"{vs30_bin}, $\mu$={results_df.loc[mask, 'ln_residual'].mean():.3f}, "
            rf"$\sigma$={results_df.loc[mask, 'ln_residual'].std():.3f} (N={mask.sum()})",
            alpha=0.5,
            color=color,
        )
    ax.plot(
        [ln_min, ln_max],
        [ln_min, ln_max],
        "k",
    )

    ax.text(
        0.5,
        0.98,
        (
            f"ln(Vs30) - Quality Score: {quality_score}"
            if quality_score is not None
            else "ln(Vs30)"
        ),
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        fontweight="bold",
    )

    ax.set_xlim(ln_min, ln_max)
    ax.set_ylim(ln_min, ln_max)
    ax.set_xlabel("True ln(vs30)")
    ax.set_ylabel("Predicted ln(vs30)")
    ax.legend(title="Vs30 Bin")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_ffp.with_name(output_ffp.stem + "_lnVs30.png"))
    plt.close(fig)

    # Metadata
    meta_dict = dict(type="one-to-one", variable="lnVs30")
    if quality_score is not None:
        meta_dict["quality_score"] = quality_score
    mlt.utils.write_to_yaml(
        meta_dict,
        output_ffp.with_name(output_ffp.stem + "_lnVs30.yaml"),
        clobber=True,
    )


def pred_vs30_variable_scatter_plot(
    results_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    variable_name: str,
    output_ffp: Path,
    y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
):
    """
    Generates a scatter plot comparing the specified input 
    variable to predicted vs30 values,
    and saves it to the specified output file path.
    """
    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)

    for vs30_bin, color in zip(
        constants.VS30_WEIGHTING_BIN_NAMES, constants.V30_BIN_COLORS
    ):
        cur_sites = results_df.loc[
            results_df["vs30_bin"] == vs30_bin
        ].index.values.astype(str)
        ax.scatter(
            dataset_df.loc[cur_sites, variable_name],
            results_df.loc[cur_sites, "pred_vs30"],
            label=vs30_bin,
            alpha=0.5,
            color=color,
        )

    ax.set_xlabel(variable_name)
    ax.set_ylabel("Predicted vs30")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend(title="Vs30 Bin")

    if y_limits is not None:
        ax.set_ylim(y_limits)
    if x_limits is not None:
        ax.set_xlim(x_limits)

    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(type="variable-scatter", variable=variable_name),
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
    show_trend_line: bool = True,
):
    """
    Generates a scatter plot of the specified metric (e.g., MAE) vs true vs30 values,
    and saves it to the specified output file path.

    A trend line is added to the plot to show how the metric varies with vs30.
    """
    # scatter_options = mlt.plotting.ScatterOptions(
    #     "vs30",
    #     metric_name,
    #     binning_method=mlt.plotting.BinningMethod.EqualCount,
    #     trend_n_data_points=50,
    #     trend_n_bins=None,
    #     alpha=0.25,
    #     color="blue",
    #     trend_color="red",
    #     use_fixed_color=False,
    # )

    # fig, ax = mlt.plotting.gen_scatter_trend_plot(
    #     results_df, scatter_options, dpi=constants.FIG_DPI,
    # )

    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)
    marker_size = {
        "Q1": 40,
        "Q2": 30,
        "Q3": 15,
    }
    for i, (k, color) in enumerate(constants.QUALITY_SCORE_COLORS.items()):
        mask = results_df["quality_score"] == k
        ax.scatter(
            results_df.loc[mask, "vs30"],
            results_df.loc[mask, metric_name],
            label=f"Quality Score {k}",
            # alpha=0.5,
            color=color,
            zorder=10 - i,
            s=marker_size[k],
        )

    bin_centers, bin_means, bin_stds = mlt.utils.compute_count_binned_trend(
        results_df["vs30"].values,
        results_df[metric_name].values,
        n_points_per_bin=100,
        n_bins=None,
    )

    if show_trend_line:
        ax.plot(
            bin_centers,
            bin_means,
            color="red",
            linewidth=2,
            label="Trend Line",
            zorder=15,
        )
        ax.fill_between(
            bin_centers,
            bin_means - bin_stds,
            bin_means + bin_stds,
            color="red",
            alpha=0.2,
            label="Trend ± 1 Std Dev",
            zorder=14,
        )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlabel("True vs30")
    ax.set_ylabel(metric_name)
    ax.legend(title="Quality Score")
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
                linewidth=2,
                label="G&M Model MAE (Test)",
            )

    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(type="metric-scatter", metric=metric_name),
        output_ffp.with_name(output_ffp.stem + ".yaml"),
        clobber=True,
    )


def cv_iteration_metric_plot(
    train_metrics_df: pd.DataFrame,
    val_metrics_df: pd.DataFrame | None,
    out_fp: Path,
    metric: str,
):
    fig, ax = plt.subplots(figsize=(16, 10), dpi=constants.FIG_DPI)

    # Training
    for cv_fold in train_metrics_df.columns:
        ax.plot(
            train_metrics_df.index.values,
            train_metrics_df[cv_fold].values,
            c="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
    # Mean line
    ax.plot(
        train_metrics_df.index.values,
        train_metrics_df.mean(axis=1).values,
        c="blue",
        linestyle="-",
        linewidth=2,
    )

    # Validation
    if val_metrics_df is not None:
        for cv_fold in val_metrics_df.columns:
            ax.plot(
                val_metrics_df.index.values,
                val_metrics_df[cv_fold].values,
                c="red",
                linestyle="-",
                linewidth=1,
                alpha=0.5,
            )
            plt.scatter(
                val_metrics_df.idxmin().values,
                val_metrics_df.min().values,
                c="red",
                s=50,
                alpha=0.5,
            )
        # Mean line
        ax.plot(
            val_metrics_df.index.values,
            val_metrics_df.mean(axis=1).values,
            c="red",
            linestyle="-",
            linewidth=2,
        )

    ax.set_xlim(
        train_metrics_df.index.values.min(), train_metrics_df.index.values.max()
    )
    if metric in METRIC_Y_LIMITS:
        ax.set_ylim(METRIC_Y_LIMITS[metric])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_fp)
    plt.close(fig)

    mlt.utils.write_to_yaml(
        dict(type="cv-iteration-metric", metric=str(metric)),
        out_fp.with_name(out_fp.stem + ".yaml"),
        clobber=True,
    )
