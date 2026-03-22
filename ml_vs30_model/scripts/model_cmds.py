from pathlib import Path
import typer

import ml_tools as mlt
import ml_vs30_model as vs30

app = typer.Typer(
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


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
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"
    )

    vs30.catboost_model.cv_train(run_config, base_out_dir, run_post_processing)


@app.command("run-post-processing")
def run_post_processing(results_dir: Path):
    logger = mlt.utils.setup_logging()

    logger.info("Running post-processing...")
    vs30.post_processing.gen_model_perfomance_plots(results_dir)
    vs30.post_processing.gen_spatial_plots(results_dir)


if __name__ == "__main__":
    app()
