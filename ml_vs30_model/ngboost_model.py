import copy
import logging
from pathlib import Path
from dataclasses import dataclass
import functools

import optuna as opt
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from scipy import stats
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor

import ml_tools as mlt

from .configs import RunConfig
from . import pre_processing
from . import training
from . import post_processing
from . import constants
from . import utils

# Opt-in to the future pandas behavior to prevent downcasting warnings during .ffill
pd.set_option("future.no_silent_downcasting", True)

logger = logging.getLogger(__name__)


def cv_train(
    run_config: RunConfig,
    base_out_dir: Path,
    run_post_processing: bool = True,
    n_procs: int = 1,
    save_model: bool = True,
    compute_shap: bool = True,
) -> None:
    """Runs cross-validation training of the ngboost model."""
    training.cv_train(
        run_model_training,
        run_config,
        base_out_dir,
        run_post_processing=run_post_processing,
        compute_shap=compute_shap,
        n_procs=n_procs,
        save_model=save_model,
    )


def full_train(run_config: RunConfig, out_dir: Path, run_post_processing: bool = True):
    """Runs training on the full dataset and saves results."""
    logger.info(f"Loading dataset from {run_config.dataset_ffp}")
    dataset_df = pd.read_parquet(run_config.dataset_ffp)
    logger.info(f"Dataset loaded with {len(dataset_df)} samples")

    # Drop test sites
    dataset_df = dataset_df[~dataset_df.index.isin(run_config.test_sites)]
    logger.info(f"Dataset size after dropping test sites: {len(dataset_df)} samples")

    run_model_training(
        dataset_df,
        dataset_df.index.values,
        None,
        run_config,
        out_dir,
        save_train_results=True,
    )

    if run_post_processing:
        logger.info("Running post-processing...")
        train_results_df = pd.read_parquet(out_dir / "train_results.parquet")

        # Quantities
        train_results_df = post_processing.add_residuals(train_results_df)
        train_results_df = post_processing.add_mae(train_results_df)
        train_results_df = post_processing.add_lnVs30_mse(train_results_df)
        train_results_df.to_parquet(out_dir / "train_results.parquet")
        shap_values = post_processing.compute_shap_values(out_dir)

        # Plots
        post_processing.gen_model_perfomance_plots(out_dir, results_df=train_results_df)
        post_processing.gen_spatial_plots(out_dir, results_df=train_results_df)
        post_processing.gen_feature_importance_plots(
            out_dir, results_df=train_results_df, shap_values=shap_values
        )


def estimate_vs30_nz(model_dir: Path, input_dataset_ffp: Path) -> pd.DataFrame:
    """Estimates Vs30 for New Zealand using the trained model."""
    run_config = RunConfig.from_yaml(model_dir / "run_config.yaml")

    with xr.open_dataset(input_dataset_ffp, mode="r", mask_and_scale=False) as ds:
        logger.info("Loading input dataset for Vs30 estimation across New Zealand")
        land_mask = ds["on_land"].values.astype(bool)
        input_ds = ds[run_config.input_variables]

        # NaN values in numerical variables
        null_mask = np.any(
            np.isnan(input_ds[run_config.numerical_variables].to_array().values),
            axis=0,
        )
        # -9999 values in categorial variables
        if len(run_config.categorial_variables) > 0:
            null_mask |= np.any(
                input_ds[run_config.categorial_variables].to_array().values == -9999,
                axis=0,
            )
        logger.info(
            f"Input dataset contains {null_mask.sum() - (~land_mask).sum()} NaN/-9999 values. Dropping these for prediction."
        )

        # Get predictions
        logger.info("Running Vs30 estimation across New Zealand...")
        input_df = input_ds.to_dataframe().loc[(~null_mask).ravel()].reset_index()
        pre_input_df, _ = pre_processing.pre_process_features(input_df, run_config)
        model = mlt.utils.load_pickle(model_dir / "model.pkl")
        pred_dist = model.pred_dist(pre_input_df)
        pred_lnVs30_mean, pred_lnVs30_std = (
            pred_dist.params["loc"],
            pred_dist.params["scale"],
        )

        # Create data arrays
        pred_lnVs30_mean_da = xr.DataArray(
            data=np.full(land_mask.shape, np.nan),
            coords={"y": ds.y, "x": ds.x},
            dims=["y", "x"],
        )
        pred_lnVs30_mean_da.values[~null_mask] = pred_lnVs30_mean

        pred_lnVs30_std_da = xr.DataArray(
            data=np.full(land_mask.shape, np.nan),
            coords={"y": ds.y, "x": ds.x},
            dims=["y", "x"],
        )
        pred_lnVs30_std_da.values[~null_mask] = pred_lnVs30_std
        pred_vs30 = np.exp(pred_lnVs30_mean_da)

    # Save
    out_ffp = model_dir / "nz_vs30_results.nc"
    grid_dataset = xr.Dataset(
        {
            "vs30": pred_vs30,
            "lnVs30_mean": pred_lnVs30_mean_da,
            "lnVs30_std": pred_lnVs30_std_da,
        },
        attrs=ds.attrs,
    )

    grid_dataset = grid_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y")
    grid_dataset = grid_dataset.rio.write_crs(constants.NZTM2000_EPSG_STR)
    grid_dataset.to_netcdf(out_ffp)
    logger.info(f"Saved Vs30 estimates across New Zealand to {out_ffp}")

    return out_ffp


def vs30_nz_geotiff(model_dir: Path) -> Path:
    """
    Saves the kriged Vs30 (mean) + ln(Vs30) std across NZ as a 2-band GeoTIFF.
    Requires nz_vs30_results.nc (from estimate_vs30_nz) to already contain
    kriged Vs30 estimates (from post_processing.add_krigged_vs30).
    """
    nc_ffp = model_dir / "nz_vs30_results.nc"
    if not nc_ffp.exists():
        utils.raise_log(
            FileNotFoundError,
            f"{nc_ffp} not found. Run estimate_vs30_nz first.",
            logger,
        )

    with xr.open_dataset(nc_ffp) as ds:
        if "kriged_vs30_mean" not in ds.variables or "lnVs30_std" not in ds.variables:
            utils.raise_log(
                ValueError,
                f"{nc_ffp} is missing required variables.",
                logger,
            )

        kriged_vs30_da = ds["kriged_vs30_mean"].drop_encoding()
        lnvs30_std_da = ds["lnVs30_std"].drop_encoding()
        stacked = xr.concat([kriged_vs30_da, lnvs30_std_da], dim="band").assign_coords(
            band=[1, 2]
        )
        stacked = stacked.rio.write_crs(constants.NZTM2000_EPSG_STR)
        stacked = stacked.rio.write_nodata(np.nan)

        out_ffp = model_dir / "nz_vs30_results.tif"
        stacked.rio.to_raster(out_ffp)

    with rasterio.open(out_ffp, "r+") as dst:
        dst.set_band_description(1, "kriged_vs30")
        dst.set_band_description(2, "lnVs30_std")

    logger.info(f"Saved kriged Vs30 + std GeoTIFF across New Zealand to {out_ffp}")
    return out_ffp


def estimate_vs30(model_dir: Path, input_df: pd.DataFrame):
    run_config = RunConfig.from_yaml(model_dir / "run_config.yaml")
    pre_input_df, _ = pre_processing.pre_process_features(input_df, run_config)

    model = mlt.utils.load_pickle(model_dir / "model.pkl")
    pred_dist = model.pred_dist(pre_input_df)
    pred_lnVs30_mean, pred_lnVs30_std = (
        pred_dist.params["loc"],
        pred_dist.params["scale"],
    )

    results_df = input_df[["lon", "lat"]].copy()
    if "vs30" in input_df.columns:
        results_df["vs30"] = input_df["vs30"]
        results_df["quality_score"] = input_df["quality_score"]

    results_df["pred_vs30"] = np.exp(pred_lnVs30_mean)
    results_df["pred_vs30_std"] = pred_lnVs30_std

    results_df = post_processing.add_residuals(results_df)
    return results_df


def run_model_training(
    dataset_df: pd.DataFrame,
    train_sites: np.ndarray,
    val_sites: np.ndarray,
    run_config: RunConfig,
    out_dir: Path,
    cv_ix: int | None = None,
    verbose: bool = False,
    save_train_results: bool = False,
    compute_shap: bool = False,
    save_model: bool = True,
) -> None:
    """Runs training of the ngboost model on the provided dataset."""
    assert (
        np.isin(run_config.test_sites, dataset_df.index.values.astype(str)).sum() == 0
    ), "Test sites must not be present in the training dataset"

    run_config, train_X, train_y, train_df, val_X, val_y, val_df = (
        pre_processing.get_pre_processed_train_val_df(
            dataset_df,
            train_sites,
            run_config,
            val_sites=val_sites,
        )
    )
    sample_weights = train_df["sample_weight"].values

    assert (
        train_X.isna().any(axis=0).sum() == 0
    ), f"Training features contain NaN values in columns: {train_X.columns[train_X.isna().any(axis=0)].tolist()}"
    assert val_df is None or (
        val_X.isna().any(axis=0).sum() == 0
    ), f"Validation features contain NaN values in columns: {val_X.columns[val_X.isna().any(axis=0)].tolist()}"

    # Label MC sampling
    mc_train_X, mc_train_y, mc_sample_weights = None, None, None
    if run_config.apply_mc_label_sampling:
        logger.info(
            f"Applying Monte Carlo sampling of the labels with {run_config.mc_label_sampling_n} samples per site."
        )
        sampled_labels = stats.norm.rvs(
            loc=train_y.values[:, None],
            scale=dataset_df.loc[train_y.index, "ln_vs30_std"].values[:, None],
            size=(len(train_y), run_config.mc_label_sampling_n),
        )

        mc_train_X = np.repeat(
            train_X.values[:, None, :], run_config.mc_label_sampling_n, axis=1
        ).reshape(-1, train_X.shape[-1])
        mc_train_y = sampled_labels.ravel()
        mc_sample_weights = np.repeat(
            sample_weights[:, None], run_config.mc_label_sampling_n, axis=1
        ).ravel()

    logger.info("Running model training")
    ngb = NGBRegressor(
        Dist=Normal,
        learning_rate=run_config.model_config.learning_rate,
        minibatch_frac=run_config.model_config.minibatch_frac,
        col_sample=run_config.model_config.col_sample,
        Base=DecisionTreeRegressor(
            max_depth=run_config.model_config.base_max_depth,
            min_samples_leaf=run_config.model_config.base_min_samples_leaf,
        ),
        random_state=run_config.seed,
        n_estimators=run_config.model_config.iterations,
        verbose=verbose,
    )
    ngb.fit(
        mc_train_X if mc_train_X is not None else train_X,
        mc_train_y if mc_train_y is not None else train_y,
        X_val=val_X if val_X is not None else None,
        Y_val=val_y if val_y is not None else None,
        sample_weight=(
            mc_sample_weights if mc_sample_weights is not None else sample_weights
        ),
        val_sample_weight=(
            val_df["sample_weight"].values if val_df is not None else None
        ),
        early_stopping_rounds=run_config.model_config.early_stopping_rounds,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Iteration metrics
    tmp_metrics = pd.DataFrame(ngb.evals_result["train"])
    train_metrics_df = pd.DataFrame(
        index=np.arange(run_config.model_config.iterations), columns=tmp_metrics.columns
    )
    train_metrics_df.loc[tmp_metrics.index, tmp_metrics.columns] = tmp_metrics.values
    train_metrics_df = train_metrics_df.ffill().infer_objects(copy=False)
    train_metrics_df.to_parquet(out_dir / "train_metrics.parquet")

    # Validation results
    val_results_df = None
    if val_df is not None:
        val_result_df = pd.DataFrame(
            index=val_y.index,
            data=dataset_df.loc[
                val_y.index, ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
            ],
        )
        val_result_df["station"] = val_result_df.index.astype(str)
        val_result_df["cv_ix"] = cv_ix

        val_pred = ngb.pred_dist(val_X).params
        val_result_df["pred_vs30"] = np.exp(val_pred["loc"])
        val_result_df["pred_vs30_std"] = val_pred["scale"]
        val_result_df.to_parquet(out_dir / "val_results.parquet")

        tmp_metrics = pd.DataFrame(ngb.evals_result["val"])
        val_metrics_df = pd.DataFrame(
            index=np.arange(run_config.model_config.iterations),
            columns=tmp_metrics.columns,
        )
        val_metrics_df.loc[tmp_metrics.index, tmp_metrics.columns] = tmp_metrics.values
        val_metrics_df = val_metrics_df.ffill().infer_objects(copy=False)
        val_metrics_df.to_parquet(out_dir / "val_metrics.parquet")

    # Training results
    train_result_df = train_df[
        ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
    ].copy()
    if save_train_results:
        train_result_df["station"] = train_result_df.index.astype(str)
        train_result_df["cv_ix"] = cv_ix

        # Get training predictions
        train_pred = ngb.pred_dist(train_X).params
        train_result_df["pred_vs30"] = np.exp(train_pred["loc"])
        train_result_df["pred_vs30_std"] = train_pred["scale"]

        train_result_df.to_parquet(out_dir / "train_results.parquet")

    # Compute SHAP values
    if compute_shap:
        post_processing.compute_shap_values(
            out_dir,
            run_config=run_config,
            train_results=train_result_df,
            val_results=val_results_df,
            model=ngb,
        )

    # Save model and config
    if save_model:
        mlt.utils.write_pickle(ngb, out_dir / "model.pkl")
    run_config.to_yaml(out_dir / "run_config.yaml")


@dataclass
class HPParamConfig:
    """Configuration for a single hyperparameter to be optimized."""

    name: str

    min: float | int
    max: float | int


@dataclass
class NGBoostHPOptConfig:
    """Configuration for hyperparameter optimization of the NGBoost model."""

    rel_base_out_dir: Path
    base_run_config: RunConfig

    n_iterations: int
    hp_params: dict[str, HPParamConfig]

    def __post_init__(self):
        self._study_dir = None

    @property
    def base_out_dir(self) -> Path:
        return constants.BASE_DATA_DIR / self.rel_base_out_dir

    @property
    def study_dir(self) -> Path:
        return self._study_dir

    @study_dir.setter
    def study_dir(self, value: Path):
        if self._study_dir is not None:
            raise ValueError("study_dir has already been set and cannot be modified.")
        self._study_dir = value

    def from_config(
        config_ffp: Path, run_config_ffp: Path, n_iterations: int | None = None
    ) -> "NGBoostHPOptConfig":
        config_dict = mlt.utils.load_yaml(config_ffp)
        base_run_config = RunConfig.from_config_kwargs(run_config_ffp)

        return NGBoostHPOptConfig(
            rel_base_out_dir=config_dict["rel_base_out_dir"],
            base_run_config=base_run_config,
            n_iterations=(
                config_dict["n_iterations"] if n_iterations is None else n_iterations
            ),
            hp_params={
                name: HPParamConfig(name=name, min=param["min"], max=param["max"])
                for name, param in config_dict["hp_params"].items()
            },
        )

    def to_dict(self):
        return {
            "rel_base_out_dir": self.rel_base_out_dir,
            "base_run_config": self.base_run_config.to_dict(),
            "n_iterations": self.n_iterations,
            "hp_params": {
                name: {"min": param.min, "max": param.max}
                for name, param in self.hp_params.items()
            },
        }

    def to_yaml(self, ffp: Path):
        mlt.utils.write_to_yaml(self.to_dict(), ffp)


def run_ngboost_hp_opt(
    hp_config: NGBoostHPOptConfig,
    n_startup_trials: int,
    n_trials: int,
    n_procs: int,
    suffix: str = "",
):
    """Runs hyperparameter optimization for the NGBoost model."""
    objective_fn_call = functools.partial(
        _model_objective_fn, hp_config=hp_config, n_procs=n_procs
    )

    study_id = mlt.utils.create_run_id()
    study_name = f"{study_id}{f'_{suffix}' if suffix else ''}"
    hp_config.study_dir = hp_config.base_out_dir / study_name
    hp_config.study_dir.mkdir(parents=False, exist_ok=False)
    hp_config.to_yaml(hp_config.study_dir / "hp_config.yaml")

    # Create the study & start optimizing
    study = opt.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=opt.samplers.TPESampler(
            n_startup_trials=n_startup_trials, n_ei_candidates=1000
        ),
        storage="sqlite:///{}.db".format(hp_config.study_dir / study_name),
    )
    study.optimize(objective_fn_call, n_trials=n_trials)


def _model_objective_fn(trial: opt.Trial, hp_config: NGBoostHPOptConfig, n_procs: int):
    """Objective function for hyperparameter optimization of the NGBoost model."""
    run_config = _get_trial_run_config(trial, hp_config)

    output_dir = hp_config.study_dir / f"trial_{trial.number:03d}"
    output_dir.mkdir(parents=False, exist_ok=False)

    cv_train(
        run_config,
        output_dir,
        run_post_processing=False,
        n_procs=n_procs,
        save_model=False,
        compute_shap=False,
    )

    val_logscore = xr.open_dataarray(output_dir / "val_metrics.nc").sel(
        metric="LOGSCORE"
    )
    best_iters = val_logscore.argmin(dim="iteration")
    best_iter_mean = best_iters.mean(dim="cv_fold").item()
    best_iters_std = best_iters.std(dim="cv_fold").item()

    best_logscore_mean = val_logscore.sel(iteration=best_iters).mean().item()
    best_logscore_std = val_logscore.sel(iteration=best_iters).std().item()

    trial.set_user_attr("best_iteration_mean", int(best_iter_mean))
    trial.set_user_attr("best_iteration_std", float(best_iters_std))
    trial.set_user_attr("best_logscore_mean", float(best_logscore_mean))
    trial.set_user_attr("best_logscore_std", float(best_logscore_std))

    return best_logscore_mean


def _get_trial_run_config(trial: opt.Trial, hp_config: NGBoostHPOptConfig):
    """Creates the RunConfig for the given trial"""
    run_config = copy.deepcopy(hp_config.base_run_config)
    run_config.model_config.iterations = hp_config.n_iterations

    run_config.model_config.learning_rate = trial.suggest_float(
        "learning_rate",
        low=hp_config.hp_params["learning_rate"].min,
        high=hp_config.hp_params["learning_rate"].max,
        step=0.001,
    )
    run_config.model_config.minibatch_frac = trial.suggest_float(
        "minibatch_frac",
        low=hp_config.hp_params["minibatch_frac"].min,
        high=hp_config.hp_params["minibatch_frac"].max,
        step=0.01,
    )
    run_config.model_config.col_sample = trial.suggest_float(
        "col_sample",
        low=hp_config.hp_params["col_sample"].min,
        high=hp_config.hp_params["col_sample"].max,
        step=0.01,
    )
    run_config.model_config.base_max_depth = trial.suggest_int(
        "base_max_depth",
        low=hp_config.hp_params["base_max_depth"].min,
        high=hp_config.hp_params["base_max_depth"].max,
    )
    run_config.model_config.base_min_samples_leaf = trial.suggest_int(
        "base_min_samples_leaf",
        low=hp_config.hp_params["base_min_samples_leaf"].min,
        high=hp_config.hp_params["base_min_samples_leaf"].max,
    )

    return run_config
