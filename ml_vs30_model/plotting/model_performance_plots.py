from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def one_to_one_plot(results_df: pd.DataFrame, output_ffp: Path):
    """
    Generates a one-to-one plot comparing true vs30 values to predicted vs30 values,
    and saves it to the specified output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        [results_df["vs30"].min(), results_df["vs30"].max()],
        [results_df["vs30"].min(), results_df["vs30"].max()],
        "k",
    )
    ax.scatter(results_df["vs30"], results_df["pred_vs30"])

    ax.set_xlabel("True vs30")
    ax.set_ylabel("Predicted vs30")

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_ffp)
