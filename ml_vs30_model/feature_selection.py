import logging
import copy
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import ml_tools as mlt

from .configs import RunConfig, ModelType
from . import utils
from . import catboost_model
from . import ngboost_model
from . import constants

logger = logging.getLogger(__name__)


def run_feature_selection(
    base_run_config: RunConfig, variables: list, base_out_dir: Path, n_procs: int = 1
):
    """
    Runs feature selection by epxloring the impact of each variable with
    respect to the base run configuration.
    """
    if base_run_config.model_type is ModelType.CatBoost:
        run_fn = catboost_model.cv_train
    elif base_run_config.model_type == ModelType.NGBoost:
        run_fn = ngboost_model.cv_train
    else:
        utils.raise_log(
            ValueError, f"Unsupported model type: {base_run_config.model_type}"
        )

    # Run base
    _run_helper(base_run_config, "base", base_out_dir, run_fn)

    # Run variable variations
    if n_procs == 1:
        logger.info("Running feature selection using a single process")
        for variable in variables:
            _run_helper(base_run_config, variable, base_out_dir, run_fn)
    else:
        logger.info(f"Running feature selection using {n_procs} processes")
        with mp.Pool(processes=n_procs) as pool:
            pool.starmap(
                _run_helper,
                [
                    (base_run_config, variable, base_out_dir, run_fn, p_ix)
                    for p_ix, variable in enumerate(variables)
                ],
            )
    
    # Summarise results
    logger.info("Summarising feature selection results")
    results = {}
    run_dir_names = [cur_dir.name for cur_dir in base_out_dir.glob("*/") if cur_dir.is_dir()]
    for cur_dir_name in run_dir_names:
        variable = cur_dir_name.split("_var_")[-1]
        results_df = pd.read_parquet(base_out_dir / cur_dir_name / "val_results.parquet")

        bias = np.mean(results_df.ln_residual.values)
        res_std = np.std(results_df.ln_residual.values)
        cur_results = {
            "bias": bias,
            "res_std": res_std,
        }

        metric_da = xr.load_dataarray(base_out_dir / cur_dir_name / "val_metrics.nc")

        if base_run_config.model_type == ModelType.CatBoost:
            best_iter = metric_da.mean(dim="cv_fold").sel(metric="RMSE").argmin(dim="iteration").item()
            cur_results |= {
                "best_iter": best_iter,
                "best_rmse": metric_da.sel(iteration=best_iter, metric="RMSE").mean().item(),
                "best_rmse_std": metric_da.sel(iteration=best_iter, metric="RMSE").mean().item(),
            }
        elif base_run_config.model_type == ModelType.NGBoost:
            best_iter = metric_da.mean(dim="cv_fold").sel(metric="LOGSCORE").argmin(dim="iteration").item()
            cur_results |= {
                "best_iter": best_iter,
                "best_logscore": metric_da.sel(iteration=best_iter, metric="LOGSCORE").mean().item(),
                "best_logscore_std": metric_da.sel(iteration=best_iter, metric="LOGSCORE").std().item(),
            }

        results[variable] = cur_results

    results_df = pd.DataFrame(results).T
    results_df.to_parquet(base_out_dir / "feature_selection_results.parquet")

def _run_helper(
    base_run_config: RunConfig,
    variable: str,
    base_out_dir: Path,
    run_fn: callable,
    p_ix: int | None = None,
):
    """Helper function to run a single feature selection run for a given variable."""
    if variable in base_run_config.input_variables:
        logger.warning(f"Variable {variable} is already in the base run configuration. Skipping run for this variable.")
        return

    run_id = f"{mlt.utils.create_run_id(True)}_var_{variable}"
    (run_dir := base_out_dir / run_id).mkdir(parents=True, exist_ok=False)

    # Setup logging for the run
    log_ffp = run_dir / "run.log"
    if p_ix is None:
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(log_ffp)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        run_logger = logging.getLogger(__name__)
        run_logger.info(f"Running feature selection for variable {variable}, run ID: {run_id}")
    else:
        run_logger = mlt.utils.setup_logging(log_ffp, enable_console=False)
        run_logger.info(
            f"Running feature selection for variable {variable}, run ID: {run_id}, on process {p_ix}."
        )

    run_config = copy.deepcopy(base_run_config)
    if variable != "base":
        run_config.input_variables.append(str(variable))

    run_fn(run_config, run_dir, n_procs=1)
