from pathlib import Path
import typer
import xarray as xr

import pandas as pd
import numpy as np
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

    vs30.nn_model.cv_train(
        run_config, base_out_dir, run_post_processing=run_post_processing
    )


@app.command("full-train-ngboost")
def full_train_ngboost(
    run_config_ffp: Path,
    rel_dataset_ffp: Path | None = None,
    n_iterations: int | None = None,
    apply_vs30_sample_weights: bool | None = None,
    id_suffix: str | None = None,
    run_post_processing: bool = True,
    extra_input_variables: list[str] | None = None,
):
    """
    Runs training of the NGBoost model on the full dataset,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging()

    run_config = vs30.RunConfig.from_config_kwargs(
        run_config_ffp,
        rel_dataset_ffp=rel_dataset_ffp,
        apply_vs30_sample_weights=apply_vs30_sample_weights,
        iterations=n_iterations,
        extra_input_variables=extra_input_variables,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    out_dir = run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"

    vs30.ngboost_model.full_train(run_config, out_dir, run_post_processing)


@app.command("cv-train-ngboost")
def cv_train_ngboost(
    run_config_ffp: Path,
    rel_dataset_ffp: Path | None = None,
    n_cv_folds: int | None = None,
    n_iterations: int | None = None,
    apply_vs30_sample_weights: bool | None = None,
    apply_quality_sample_weights: bool | None = None,
    id_suffix: str | None = None,
    run_post_processing: bool = True,
    extra_input_variables: list[str] | None = None,
    n_procs: int = 1,
    rel_results_dir: Path | None = None,
):
    """
    Runs cross-validation training of the NGBoost model,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging()

    run_config = vs30.RunConfig.from_config_kwargs(
        run_config_ffp,
        rel_dataset_ffp=rel_dataset_ffp,
        n_cv_folds=n_cv_folds,
        apply_vs30_sample_weights=apply_vs30_sample_weights,
        apply_quality_sample_weights=apply_quality_sample_weights,
        iterations=n_iterations,
        extra_input_variables=extra_input_variables,
        rel_results_dir=rel_results_dir,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"
    )

    vs30.ngboost_model.cv_train(
        run_config,
        base_out_dir,
        run_post_processing=run_post_processing,
        n_procs=n_procs,
    )


@app.command("hp-opt-ngboost")
def hp_opt_ngboost(
    hp_config_ffp: Path,
    base_run_config_ffp: Path,
    n_trials: int,
    n_procs: int = 1,
    suffix: str = "",
    n_startup_trials: int = 25,
    n_iterations: int | None = None,
):
    """
    Runs hyperparameter optimization for the NGBoost model,
    based on the provided configuration file.
    """
    mlt.utils.setup_logging()

    hp_config = vs30.ngboost_model.NGBoostHPOptConfig.from_config(
        hp_config_ffp, base_run_config_ffp, n_iterations=n_iterations
    )
    vs30.ngboost_model.run_ngboost_hp_opt(
        hp_config, n_startup_trials, n_trials, n_procs, suffix
    )


@app.command("cv-train-catboost")
def cv_train_catboost(
    run_config_ffp: Path,
    rel_dataset_ffp: Path | None = None,
    n_cv_folds: int | None = None,
    n_iterations: int | None = None,
    apply_vs30_sample_weights: bool | None = None,
    apply_quality_sample_weights: bool | None = None,
    id_suffix: str | None = None,
    run_post_processing: bool = True,
    extra_input_variables: list[str] | None = None,
    n_procs: int = 1,
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
        extra_input_variables=extra_input_variables,
        apply_quality_sample_weights=apply_quality_sample_weights,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"
    )

    vs30.catboost_model.cv_train(
        run_config,
        base_out_dir,
        run_post_processing=run_post_processing,
        n_procs=n_procs,
    )


@app.command("full-train-catboost")
def full_train_catboost(
    run_config_ffp: Path,
    rel_dataset_ffp: Path | None = None,
    n_iterations: int | None = None,
    apply_vs30_sample_weights: bool | None = None,
    id_suffix: str | None = None,
    run_post_processing: bool = True,
    extra_input_variables: list[str] | None = None,
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
        extra_input_variables=extra_input_variables,
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
def estimate_vs30_nz(model_dir: Path, input_dataset_ffp: Path):
    """
    Estimates Vs30 across New Zealand using the trained model.
    """
    mlt.utils.setup_logging()

    run_config = vs30.RunConfig.from_yaml(model_dir / "run_config.yaml")

    if run_config.model_type == vs30.configs.ModelType.CatBoost:
        ffp = vs30.catboost_model.estimate_vs30_nz(model_dir, input_dataset_ffp)
    elif run_config.model_type == vs30.configs.ModelType.NGBoost:
        ffp = vs30.ngboost_model.estimate_vs30_nz(model_dir, input_dataset_ffp)
    else:
        vs30.utils.raise_log(
            NotImplementedError,
            f"Model type {run_config.model_type} not supported for NZ-wide estimation.",
        )

    # Create histogram
    ds = xr.open_dataset(ffp)
    vs30_values = ds["vs30"].values[~ds["vs30"].isnull()]
    vs30.plotting.other.plot_nz_vs30_hist(
        vs30_values, model_dir / "nz_vs30_histogram.png"
    )
    
@app.command("test-predictions")
def test_predictions(dataset_ffp: Path, full_model_dir: Path, test_sites_ffp: Path):
    test_sites = np.load(test_sites_ffp)
    dataset_df = pd.read_parquet(dataset_ffp)

    results_df = vs30.ngboost_model.estimate_vs30(full_model_dir, dataset_df.loc[test_sites])
    results_df.to_parquet(full_model_dir / "test_results.parquet")

@app.command("add-other-nz-estimates")
def add_other_nz_estimates(dataset_ffp: Path):
    """
    Adds other Vs30 estimates for New Zealand to the provided dataset.
    """
    mlt.utils.setup_logging()

    foster_data_dir = vs30.constants.BASE_DATA_DIR / "nz_estimates/vs30map_data_2023"
    vs30.post_processing.add_foster_nz_estimates(dataset_ffp, foster_data_dir)

    foster_original_ffp = (
        vs30.constants.BASE_DATA_DIR
        / "nz_estimates/foster_original/foster_paper_original.tif"
    )
    vs30.post_processing.add_foster_original_nz_estimates(
        dataset_ffp, foster_original_ffp
    )

    jaehwi_v1p0_ffp = (
        vs30.constants.BASE_DATA_DIR / "nz_estimates/jaehwi_v1p0_26March/v1p0_26Mar.tif"
    )
    vs30.post_processing.add_jaehwi_nz_estimates(
        dataset_ffp, jaehwi_v1p0_ffp, prefix="jw_v1p0"
    )


@app.command("add-ml-model-residuals")
def add_ml_model_residuals(dataset_ffp: Path, other_dataset_ffp: Path):
    """
    Adds residuals with respect to the other model
    """
    mlt.utils.setup_logging()
    vs30.post_processing.add_ml_model_residuals(dataset_ffp, other_dataset_ffp)

@app.command("add-krigged-vs30")
def add_krigged_vs30(full_model_dir: Path):
    """
    Adds krigged Vs30 estimates to the dataset used for the full model.
    """
    mlt.utils.setup_logging()
    vs30.post_processing.add_krigged_vs30(full_model_dir)


    print("wtf")


@app.command("run-feature-selection")
def run_feature_selection(
    base_run_config_ffp: Path,
    variables: list[vs30.constants.InputVariable],
    base_out_dir: Path,
    n_iterations: int | None = None,
    id_suffix: str | None = None,
    n_procs: int = 1,
):
    """
    Runs feature selection
    """
    mlt.utils.setup_logging()

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = base_out_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"
    base_out_dir.mkdir(parents=True, exist_ok=False)

    base_run_config = vs30.RunConfig.from_config_kwargs(
        base_run_config_ffp, iterations=n_iterations
    )
    vs30.feature_selection.run_feature_selection(
        base_run_config, variables, base_out_dir, n_procs=n_procs
    )


if __name__ == "__main__":
    app()
