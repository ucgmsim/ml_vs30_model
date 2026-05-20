import copy
import logging
from pathlib import Path
from dataclasses import dataclass
import functools

import optuna as opt
import pandas as pd
import numpy as np
import xarray as xr
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor

import ml_tools as mlt

from .configs import RunConfig
from . import pre_processing
from . import training
from . import post_processing
from . import constants

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

    if len(run_config.categorial_variables) > 0:
        raise NotImplementedError(
            "Pre-processing of categorial variables is not implemented yet"
        )

    run_config, train_X, train_y, train_df, val_X, val_y, val_df = (
        pre_processing.get_pre_processed_train_val_df(
            dataset_df,
            train_sites,
            run_config,
            val_sites=val_sites,
        )
    )

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
        train_X,
        train_y,
        X_val=val_X if val_X is not None else None,
        Y_val=val_y if val_y is not None else None,
        sample_weight=train_df["sample_weight"].values,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Iteration metrics
    pd.DataFrame(ngb.evals_result["train"]).to_parquet(
        out_dir / "train_metrics.parquet"
    )

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

        pd.DataFrame(ngb.evals_result["val"]).to_parquet(
            out_dir / "val_metrics.parquet"
        )

    # Training results
    train_result_df = pd.DataFrame(
        index=train_y.index,
        data=dataset_df.loc[
            train_y.index, ["lon", "lat", "vs30", "vs30_bin", "dense_vs30_bin"]
        ],
    )
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
        post_processing.compute_shap_feature_importance(
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
    best_iter = val_logscore.mean(dim="cv_fold").argmin(dim="iteration").item()

    best_logscore = val_logscore.sel(iteration=best_iter).mean().item()
    best_logscore_std = val_logscore.sel(iteration=best_iter).std().item()

    trial.set_user_attr("best_iteration", int(best_iter))
    trial.set_user_attr("best_logscore", float(best_logscore))
    trial.set_user_attr("best_logscore_std", float(best_logscore_std))

    return best_logscore


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
