from pathlib import Path
import typer

import ml_tools as mlt
import ml_vs30_model as vs30

app = typer.Typer(
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@app.command("cv-train-nn")
def cv_train_nn(
    run_config_ffp: Path,
    rel_dataset_ffp: Path,
    n_cv_folds: int | None = None,
    n_iterations: int | None = None,
    apply_vs30_sample_weights: bool | None = None,
    id_suffix: str | None = None,
    run_post_processing: bool = True,
):
    """
    Runs cross-validation training of the neural network model,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging()
    run_config = vs30.RunConfig.from_config_kwargs(
        run_config_ffp,
        rel_dataset_ffp=rel_dataset_ffp,
        n_cv_folds=n_cv_folds,
        apply_vs30_sample_weights=apply_vs30_sample_weights,
        iterations=n_iterations,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"
    )

    vs30.nn_model.cv_train(run_config, base_out_dir, run_post_processing=run_post_processing)


@app.command("cv-train-ngboost")
def cv_train_ngboost(config_ffp: Path):
    """
    Runs cross-validation training of the NGBoost model,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging()
    run_config = vs30.RunConfig.from_yaml(config_ffp)

    vs30.ngboost_model.cv_train(run_config)


@app.command("cv-train-catboost")
def cv_train_catboost(
    run_config_ffp: Path,
    rel_dataset_ffp: Path,
    n_cv_folds: int | None = None,
    n_iterations: int | None = None,
    apply_vs30_sample_weights: bool | None = None,
    id_suffix: str | None = None,
    run_post_processing: bool = True,
):
    """
    Runs cross-validation training of the CatBoost model,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging()

    run_config = vs30.RunConfig.from_config_kwargs(
        run_config_ffp,
        rel_dataset_ffp=rel_dataset_ffp,
        n_cv_folds=n_cv_folds,
        apply_vs30_sample_weights=apply_vs30_sample_weights,
        iterations=n_iterations,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"
    )

    vs30.catboost_model.cv_train(run_config, base_out_dir, run_post_processing)


@app.command("full-train-catboost")
def full_train_catboost(
    run_config_ffp: Path,
    rel_dataset_ffp: Path,
    n_iterations: int | None = None,
    apply_vs30_sample_weights: bool | None = None,
    id_suffix: str | None = None,
    run_post_processing: bool = True,
):
    """
    Runs training of the CatBoost model on the full dataset,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging()

    run_config = vs30.RunConfig.from_config_kwargs(
        run_config_ffp,
        rel_dataset_ffp=rel_dataset_ffp,
        apply_vs30_sample_weights=apply_vs30_sample_weights,
        iterations=n_iterations,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    out_dir = run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"

    vs30.catboost_model.full_train(run_config, out_dir, run_post_processing)


@app.command("run-cv-post-processing")
def run_post_processing(results_dir: Path, gen_waterfall_plots: bool = False):
    """Runs post-processing on the results of a cross-validation training run"""
    logger = mlt.utils.setup_logging()

    logger.info("Running post-processing...")
    vs30.post_processing.gen_cv_iteration_metric_plots(results_dir)

    vs30.post_processing.gen_model_perfomance_plots(results_dir)
    vs30.post_processing.gen_spatial_plots(results_dir)
    vs30.post_processing.gen_feature_importance_plots(
        results_dir, gen_waterfall_plots=gen_waterfall_plots
    )


@app.command("estimate-vs30-nz")
def estimate_vs30_nz(
    model_dir: Path, grid_dx: float, grid_dy: float):
    """
    Estimates Vs30 across New Zealand using the trained model.

    Parameters
    ----------
    model_dir : Path
        Directory containing the trained model and run configuration.
    grid_dx : float
        Grid spacing in the x-direction (longitude) in meters.
    grid_dy : float
        Grid spacing in the y-direction (latitude) in meters.
    """
    vs30.catboost_model.estimate_vs30_nz(model_dir, grid_dx, grid_dy)

if __name__ == "__main__":
    app()
