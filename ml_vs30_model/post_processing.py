import logging
from pathlib import Path
import time

import rasterio
from rasterio import transform
import shap
import xarray as xr
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
import ml_tools as mlt


from .plotting import model_perf_plots, spatial, feature_importance_plots
from .configs import RunConfig, ModelType
from . import pre_processing
from . import constants
from . import utils

logger = logging.getLogger(__name__)


def add_residuals(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds residuals to the provided results dataframe.
    """
    results_df["ln_residual"] = np.log(results_df["vs30"]) - np.log(
        results_df["pred_vs30"]
    )
    return results_df


def add_mae(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds absolute error (MAE) to the provided results dataframe.
    """
    results_df["mae"] = np.abs(results_df["vs30"] - results_df["pred_vs30"])
    return results_df


def add_lnVs30_mse(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds mean squared error (MSE) to the provided results dataframe.
    """
    results_df["lnVs30_mse"] = (
        np.log(results_df["vs30"]) - np.log(results_df["pred_vs30"])
    ) ** 2
    return results_df


def compute_shap_values(
    model_dir: Path,
    run_config: RunConfig | None = None,
    train_results: pd.DataFrame | None = None,
    val_results: pd.DataFrame | None = None,
    model: CatBoostRegressor | NGBRegressor | None = None,
) -> None:
    """
    Computes SHAP values
    for the model in the provided directory.
    """
    logger.info("Computing SHAP values...")
    run_config = (
        RunConfig.from_yaml(model_dir / "run_config.yaml")
        if run_config is None
        else run_config
    )
    assert run_config.scale_params is not None

    if run_config.model_type not in [ModelType.CatBoost, ModelType.NGBoost]:
        utils.raise_log(
            NotImplementedError,
            f"SHAP values computation not implemented for model type {run_config.model_type}",
            logger,
        )

    # Load data
    dataset_df = pd.read_parquet(run_config.dataset_ffp)
    train_results = (
        pd.read_parquet(model_dir / "train_results.parquet")
        if train_results is None
        else train_results
    )
    if val_results or (val_results_ffp := model_dir / "val_results.parquet").exists():
        val_results = (
            pd.read_parquet(val_results_ffp) if val_results is None else val_results
        )

    _, train_X, __, ___, val_X, *_ = pre_processing.get_pre_processed_train_val_df(
        dataset_df,
        train_results.index.values,
        run_config,
        val_sites=val_results.index.values if val_results is not None else None,
    )
    assert train_results.index.equals(train_X.index)
    assert val_results is None or val_results.index.equals(val_X.index)

    if model is None:
        if run_config.model_type == ModelType.CatBoost:
            model = CatBoostRegressor()
            model.load_model(model_dir / "model.cbm")
        elif run_config.model_type == ModelType.NGBoost:
            model = mlt.utils.load_pickle(model_dir / "model.pkl")
        else:
            raise ValueError(
                f"Unsupported model type {run_config.model_type} "
                "for SHAP feature importance computation."
            )

    explainer = shap.TreeExplainer(
        model,
        train_X,
        model_output="raw" if run_config.model_type == ModelType.CatBoost else 0,
    )
    explainer_values = explainer(
        train_X if val_results is None else val_X, check_additivity=False
    )
    mlt.utils.write_pickle(explainer_values, model_dir / "shap_values.pkl")

    return explainer_values


def gen_model_perfomance_plots(
    results_dir: Path,
    results_df: pd.DataFrame | None = None,
    run_config: RunConfig | None = None,
) -> None:
    """
    Generates model performance plots for the provided results directory.
    """
    if results_df is None:
        logger.info("Reading results dataframe from parquet file val_results.parquet.")
        results_df = pd.read_parquet(results_dir / "val_results.parquet")

    # Add quality score to results_df
    run_config = (
        RunConfig.from_yaml(results_dir / "run_config.yaml")
        if run_config is None
        else run_config
    )
    dataset_df = pd.read_parquet(run_config.dataset_ffp)
    results_df["quality_score"] = dataset_df.loc[results_df.index, "quality_score"]

    logger.info("Generating model performance plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    model_perf_plots.one_to_one_plot(results_df, outdir / "one_to_one_plot.png")
    # model_perf_plots.residuals_histogram(results_df, outdir / "residuals_histogram.png")
    model_perf_plots.residual_kde(results_df, outdir / "residuals_kde.png")
    # model_perf_plots.quaternary_region_residual(
    # results_df, dataset_df, outdir / "quaternary_region_residuals.png"
    # )
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "mae_scatter_plot.png",
        metric_name="mae",
        x_limits=(0, 1550),
        y_limits=(0, 1000),
        show_geyin_maurer_model=True,
    )
    # model_perf_plots.metric_scatter_plot(
    #     results_df,
    #     outdir / "lnVs30_mse_scatter_plot.png",
    #     metric_name="lnVs30_mse",
    #     x_limits=(0, 2000),
    #     # y_limits=(0, 1000),
    # )
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "ln_residual_scatter_plot.png",
        metric_name="ln_residual",
        x_limits=(100, 1550),
        y_limits=(-1.5, 1.5),
        show_quality_trend_lines=True,
    )

    # model_perf_plots.pit_plot(results_df, outdir)


def gen_cv_iteration_metric_plots(
    results_dir: Path,
    train_metrics: xr.DataArray | None = None,
    val_metrics: xr.DataArray | None = None,
    metrics: list[str] | None = None,
):
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)

    # Load the data if not provided
    if train_metrics is None:
        logger.info("Reading train metrics from netcdf file train_metrics.nc.")
        train_metrics = xr.open_dataarray(results_dir / "train_metrics.nc")
    if val_metrics is None:
        if (results_dir / "val_metrics.nc").exists():
            logger.info("Reading validation metrics from netcdf file val_metrics.nc.")
            val_metrics = xr.open_dataarray(results_dir / "val_metrics.nc")
        else:
            logger.warning(
                "Validation metrics file val_metrics.nc not found. Validation metrics will be set to None."
            )
            val_metrics = None

    if metrics is None:
        metrics = train_metrics.metric.values

    for metric in metrics:
        model_perf_plots.cv_iteration_metric_plot(
            train_metrics.sel(metric=metric).to_pandas().T,
            val_metrics_df=(
                val_metrics.sel(metric=metric).to_pandas().T
                if val_metrics is not None
                else None
            ),
            out_fp=outdir / f"{metric}_iteration_plot.png",
            metric=metric,
        )


def gen_spatial_plots(
    results_dir: Path,
    results_df: pd.DataFrame | None = None,
    run_config: RunConfig | None = None,
) -> None:
    """
    Generates spatial plots for the provided results directory.
    """
    if results_df is None:
        logger.info("Reading results dataframe from parquet file val_results.parquet.")
        results_df = pd.read_parquet(results_dir / "val_results.parquet")

    # Add quality score to results_df
    if "quality_score" not in results_df.columns:
        run_config = (
            RunConfig.from_yaml(results_dir / "run_config.yaml")
            if run_config is None
            else run_config
        )
        dataset_df = pd.read_parquet(run_config.dataset_ffp)
        if "quality_score" in dataset_df.columns:
            results_df["quality_score"] = dataset_df.loc[
                results_df.index, "quality_score"
            ]

    logger.info("Generating spatial plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    spatial.nz_site_residuals_map(results_df, outdir / "site_residual.html")


def gen_feature_importance_plots(
    results_dir: Path,
    shap_values: shap.Explanation | None = None,
    results_df: pd.DataFrame | None = None,
    gen_waterfall_plots: bool = False,
) -> None:
    """
    Generates feature importance plots
    for the provided results directory.
    """
    if shap_values is None:
        logger.info("Loading SHAP values from pickle file shap_values.pkl.")
        shap_values = pd.read_pickle(results_dir / "shap_values.pkl")

    # Use nice feature names
    shap_features = shap_values.feature_names
    shap_values.feature_names = [
        (
            feat
            if feat not in constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING
            else constants.INPUT_VARIABLE_TO_NICE_NAME_MAPPING[feat]
        )
        for feat in shap_values.feature_names
    ]

    if results_df is None:
        if (results_dir / "val_results.parquet").exists():
            logger.info(
                "Reading results dataframe from parquet file val_results.parquet."
            )
            results_df = pd.read_parquet(results_dir / "val_results.parquet")
        elif (results_dir / "train_results.parquet").exists():
            logger.info(
                "Reading results dataframe from parquet file train_results.parquet."
            )
            results_df = pd.read_parquet(results_dir / "train_results.parquet")

    logger.info("Generating feature importance plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    feature_importance_plots.shap_global(shap_values, results_df, outdir)
    feature_importance_plots.shap_beeswarm(shap_values, results_df, outdir)
    feature_importance_plots.shap_feature_trends(shap_values, shap_features, outdir)
    if gen_waterfall_plots:
        feature_importance_plots.shap_waterfall(
            shap_values, results_df, outdir / "waterfall_plots"
        )


def _get_dataset_values(dataset_ffp: Path, extract_kriged: bool = False):
    """
    Helper function to get the
    coordinates, nan mask, and vs30 data array.
    """
    with xr.open_dataset(dataset_ffp, mode="r") as ds:
        if extract_kriged:
            vs30_da = ds["kriged_vs30_mean"]
        else:
            vs30_da = ds["vs30"]
        y, x = vs30_da.y.values, vs30_da.x.values
        nan_mask = vs30_da.isnull().values

    grid_x, grid_y = np.meshgrid(x, y)
    coords = np.column_stack((grid_x[~nan_mask], grid_y[~nan_mask]))

    return coords, nan_mask, vs30_da


def _create_dataset_from_estimates(
    ml_vs30_da: xr.DataArray,
    nan_mask: np.ndarray,
    mean_estimate: np.ndarray,
    prefix: str,
    std_estimate: np.ndarray | None = None,
    res_only: bool = False,
) -> xr.Dataset:
    y, x = ml_vs30_da.y.values, ml_vs30_da.x.values

    # Compute residual
    res = mean_estimate - ml_vs30_da.values[~nan_mask]
    ln_res = np.log(mean_estimate) - np.log(ml_vs30_da.values[~nan_mask])

    # Create data arrays
    vs30_mean_da = xr.DataArray(
        data=np.full(ml_vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    )
    vs30_mean_da.values[~nan_mask] = mean_estimate

    if std_estimate is not None:
        vs30_std_da = xr.DataArray(
            data=np.full(ml_vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
        )
        vs30_std_da.values[~nan_mask] = std_estimate

    res_da = xr.DataArray(
        data=np.full(ml_vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    )
    res_da.values[~nan_mask] = res

    ln_res_da = xr.DataArray(
        data=np.full(ml_vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    )
    ln_res_da.values[~nan_mask] = ln_res

    ds_dict = {
        f"{prefix}_vs30_res": res_da,
        f"{prefix}_vs30_ln_res": ln_res_da,
    }
    if not res_only:
        ds_dict[f"{prefix}_vs30_mean"] = vs30_mean_da
        if std_estimate is not None:
            ds_dict[f"{prefix}_vs30_std"] = vs30_std_da

    ds = xr.Dataset(ds_dict)
    # if std_estimate is not None:
    # ds[f"{prefix}_vs30_std"] = vs30_std_da

    return ds


def add_foster_nz_estimates(dataset_ffp: Path, foster_data_dir: Path):
    """
    Adds Foster et al. (2016) Vs30 estimates for New Zealand to the provided dataset.
    """
    logger.info(f"Adding modified Foster et al. result to database {dataset_ffp}...")
    coords, nan_mask, vs30_da = _get_dataset_values(dataset_ffp)

    ### Combined MVN
    logger.info("Extracting Foster et al. combined MVN estimates...")
    with rasterio.open(foster_data_dir / "combined_mvn.tif") as ds:
        assert ds.crs.to_epsg() == constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"
        foster_data = ds.read([1, 2], masked=True).filled(np.nan)

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
        foster_combined_mvn_vs30_mean = foster_data[0, rows, cols]
        foster_combined_mvn_vs30_std = foster_data[1, rows, cols]

    if (foster_nan_count := np.isnan(foster_combined_mvn_vs30_mean).sum()) > 0:
        logger.warning(
            f"Foster has {foster_nan_count} NaN estimates out of {len(foster_combined_mvn_vs30_mean)} total estimates."
        )

    ds = _create_dataset_from_estimates(
        vs30_da,
        nan_mask,
        foster_combined_mvn_vs30_mean,
        "foster_combined_mvn",
        std_estimate=foster_combined_mvn_vs30_std,
    )

    # Save
    logger.info("Saving modified Foster et al. estimates to dataset...")
    ds.to_netcdf(dataset_ffp, mode="a")


def add_jaehwi_nz_estimates(
    dataset_ffp: Path, jaehwi_data_ffp: Path, prefix: str = "jw"
):
    """
    Adds Jaehwi's Vs30 estimates for New Zealand to the provided dataset.
    """
    logger.info(
        f"Adding Jaehwi's estimates to result ({jaehwi_data_ffp.parent.name}) database {dataset_ffp}..."
    )

    coords, nan_mask, vs30_da = _get_dataset_values(dataset_ffp)

    logger.info("Extracting Jaehwi's estimates...")
    with rasterio.open(jaehwi_data_ffp) as ds:
        assert ds.crs.to_epsg() == constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"

        jw_data = ds.read([1, 2], masked=True).filled(np.nan)

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
        jw_vs30_mean, jw_vs30_std = jw_data[0, rows, cols], jw_data[1, rows, cols]

    if (jw_nan_count := np.isnan(jw_vs30_mean).sum()) > 0:
        logger.warning(
            f"Jaehwi's estimates have {jw_nan_count} NaN estimates out of {len(jw_vs30_mean)} total estimates."
        )

    ds = _create_dataset_from_estimates(
        vs30_da,
        nan_mask,
        jw_vs30_mean,
        prefix,
        std_estimate=jw_vs30_std,
    )

    # Save
    logger.info("Saving Jaehwi's estimates to dataset...")
    ds.to_netcdf(dataset_ffp, mode="a")


def add_foster_original_nz_estimates(
    dataset_ffp: Path, foster_original_ffp: Path, use_kriged: bool = False
):
    logger.info(f"Adding original Foster et al. result to database {dataset_ffp}...")
    coords, nan_mask, vs30_da = _get_dataset_values(
        dataset_ffp, extract_kriged=use_kriged
    )

    logger.info("Extracting original Foster et al. estimates...")
    with rasterio.open(foster_original_ffp) as ds:
        assert ds.crs.to_epsg() == constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"
        foster_original_data = ds.read(1, masked=True).filled(np.nan)

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
        foster_original_vs30_mean = foster_original_data[rows, cols]

    if (foster_original_nan_count := np.isnan(foster_original_vs30_mean).sum()) > 0:
        logger.warning(
            f"Original Foster et al. estimates have {foster_original_nan_count} "
            f"NaN estimates out of {len(foster_original_vs30_mean)} total estimates."
        )

    ds = _create_dataset_from_estimates(
        vs30_da,
        nan_mask,
        foster_original_vs30_mean,
        "kriged_vs30_foster_original" if use_kriged else "foster_original",
        res_only=use_kriged,
    )

    # Save
    logger.info("Saving original Foster et al. estimates to dataset...")
    ds.to_netcdf(dataset_ffp, mode="a")


def add_ml_model_residuals(dataset_ffp: Path, other_dataset_ffp: Path):
    logger.info(
        f"Adding residuals with respect to {other_dataset_ffp.parent.name} to {dataset_ffp.parent.name}"
    )
    coords, nan_mask, vs30_da = _get_dataset_values(dataset_ffp)
    other_coords, other_nan_mask, other_vs30_da = _get_dataset_values(other_dataset_ffp)

    assert np.array_equal(
        coords, other_coords
    ), "Coordinates of the two datasets do not match. Cannot compute residuals."
    assert np.array_equal(
        nan_mask, other_nan_mask
    ), "NaN masks of the two datasets do not match. Cannot compute residuals."

    res = other_vs30_da.values[~nan_mask] - vs30_da.values[~nan_mask]
    ln_res = np.log(other_vs30_da.values[~nan_mask]) - np.log(vs30_da.values[~nan_mask])

    res_da = xr.DataArray(
        data=np.full(vs30_da.shape, np.nan),
        coords=[vs30_da.y.values, vs30_da.x.values],
        dims=["y", "x"],
    )
    res_da.values[~nan_mask] = res

    ln_res_da = xr.DataArray(
        data=np.full(vs30_da.shape, np.nan),
        coords=[vs30_da.y.values, vs30_da.x.values],
        dims=["y", "x"],
    )
    ln_res_da.values[~nan_mask] = ln_res

    ds = xr.Dataset(
        {
            f"{other_dataset_ffp.parent.name}_vs30_res": res_da,
            f"{other_dataset_ffp.parent.name}_vs30_ln_res": ln_res_da,
        }
    )

    # Save
    ds.to_netcdf(dataset_ffp, mode="a")


def add_krigged_vs30(full_model_dir: Path):
    """
    Post-proccessing step for the final model!
    Updates the Vs30 estimates with krigged estimates
    based on the residuals at measured sites.
    """
    from VarioCorreKrigE.variofit import variofit
    from VarioCorreKrigE.okrig import ordinary_kriging

    assert (
        full_model_dir / "test_results.parquet"
    ).exists(), "Test results not found. Please run test_predictions command first."

    # Load residuals (including test, since we want to use all sites for kriging)
    logger.info("Loading data for kriging...")
    test_results_df = pd.read_parquet(full_model_dir / "test_results.parquet")
    train_results_df = pd.read_parquet(full_model_dir / "train_results.parquet")
    results_df = pd.concat([train_results_df, test_results_df])

    results_df["nztm_x"], results_df["nztm_y"] = (
        constants.WGS84_TO_NZTM_TRANSFORMER.transform(
            results_df["lon"].values, results_df["lat"].values
        )
    )

    target_locs, nan_mask, nz_vs30_da = _get_dataset_values(
        full_model_dir / "nz_vs30_results.nc"
    )

    # with xr.open_dataset(full_model_dir / "nz_vs30_results.nc") as nz_ds:
    # nz_vs30_da = nz_ds.vs30

    # grid_x, grid_y = np.meshgrid(nz_vs30_da.x.values, nz_vs30_da.y.values)
    # nan_mask = nz_vs30_da.isnull().values
    # target_locs = np.column_stack((grid_x[~nan_mask], grid_y[~nan_mask]))

    # target_locs = coords[~nan_mask]

    # Compute standardized residuals
    ln_residual_mean = results_df["ln_residual"].mean()
    ln_residual_std = results_df["ln_residual"].std()
    results_df["ln_residual_z"] = (
        results_df["ln_residual"] - ln_residual_mean
    ) / ln_residual_std

    # Fit the variogram model to the standardized residuals
    logger.info("Fitting variogram model to standardized residuals...")
    model_type = "powered_exponential"
    distance_type = "cartesian"
    h_lag, n_obs, gamma, params_nug, r2_wls, r2_ols = variofit(
        values=results_df["ln_residual_z"],
        coordinates=results_df[["nztm_x", "nztm_y"]] / 1000,
        distance_type=distance_type,
        max_distance=10,
        bin_size=1.5,
        estimator_type="Matheron",
        model_type=model_type,
        weight_fn=None,
        weight_params=None,
        xmax_factor=100,
        fix_sill=True,
        fix_nugget=True,
        plot=True,
        plot_path=full_model_dir / "kriging_variogram.png",
        transform="correlation",
    )
    logger.info(
        "Variogram model fitted with parameters: "
        + ", ".join([f"{k}={v:.4f}" for k, v in params_nug.items()])
    )

    # Ordinary Kriging
    logger.info(
        "Performing ordinary kriging to estimate residuals at target locations..."
    )
    est, std = ordinary_kriging(
        values=results_df["ln_residual_z"].values,
        coords=results_df[["nztm_x", "nztm_y"]] / 1000,
        targets=target_locs / 1000,
        model_family="correlation",
        model_type=model_type,
        params=params_nug,
        distance_type=distance_type,
        jitter=1e-10,
        return_weights=False,
        max_neighbors=10,
    )

    est = est * ln_residual_std + ln_residual_mean
    std = np.sqrt(std) * ln_residual_std

    kriging_df = pd.DataFrame(
        {
            "nztm_x": target_locs[:, 0],
            "nztm_y": target_locs[:, 1],
            "ln_residual_est": est,
            "ln_residual_std": std,
        }
    )
    kriging_df.to_parquet(full_model_dir / "kriging_estimates.parquet", index=False)

    kriged_vs30 = np.exp(np.log(nz_vs30_da.values[~nan_mask]) + est)

    kriged_vs30_da = xr.DataArray(
        data=np.full(nz_vs30_da.shape, np.nan),
        coords=[nz_vs30_da.y.values, nz_vs30_da.x.values],
        dims=["y", "x"],
    )
    kriged_vs30_da.values[~nan_mask] = kriged_vs30

    kriged_vs30_std_da = xr.DataArray(
        data=np.full(nz_vs30_da.shape, np.nan),
        coords=[nz_vs30_da.y.values, nz_vs30_da.x.values],
        dims=["y", "x"],
    )
    kriged_vs30_std_da.values[~nan_mask] = std

    ds = _create_dataset_from_estimates(
        nz_vs30_da, nan_mask, kriged_vs30, "kriged", std_estimate=std
    )

    ds.to_netcdf(full_model_dir / "nz_vs30_results.nc", mode="a")


def add_grid_SHAP_values(
    full_model_dir: Path,
    input_grid_ffp: Path,
):
    run_config = RunConfig.from_yaml(full_model_dir / "run_config.yaml")
    dataset_ffp = full_model_dir / "nz_vs30_results.nc"
    coords, nan_mask, vs30_da = _get_dataset_values(dataset_ffp)

    logger.info("Loading input grid dataset...")
    with xr.open_dataset(input_grid_ffp, mode="r", mask_and_scale=False) as ds:
        land_mask = ds["on_land"].values.astype(bool)
        input_ds = ds[run_config.input_variables]

        # NaN values in numerical variables
        null_mask = np.any(
            np.isnan(input_ds[run_config.numerical_variables].to_array().values),
            axis=0,
        )
        assert (
            len(run_config.categorial_variables) == 0
        ), "Categorical variables not supported for SHAP value computation on grid."
        logger.info(
            f"Input dataset contains {null_mask.sum() - (~land_mask).sum()} NaN/-9999 values. Dropping these for prediction."
        )

    logger.info("Pre-processing input grid dataset...")
    input_df = input_ds.to_dataframe().loc[(~null_mask).ravel()].reset_index()
    assert np.all(input_df.x == coords[:, 0]) and np.all(
        input_df.y == coords[:, 1]
    ), "Input grid coordinates do not match dataset coordinates."
    pre_input_df, _ = pre_processing.pre_process_features(input_df, run_config)

    model = mlt.utils.load_pickle(full_model_dir / "model.pkl")
    _, train_X, *_ = pre_processing.get_pre_processed_train_val_df(
        pd.read_parquet(run_config.dataset_ffp),
        pd.read_parquet(full_model_dir / "train_results.parquet").index.values,
        run_config,
    )

    logger.info("Computing SHAP values for input grid dataset...")
    start = time.time()
    explainer = shap.TreeExplainer(model, train_X, model_output=0)
    explainer_values = explainer(pre_input_df, check_additivity=False)
    logger.info(
        f"Took: {time.time() - start} to compute SHAP values for {len(pre_input_df)} grid points."
    )

    print("wtf")


def print_vs30_bin_metrics(
    results_df: pd.DataFrame,
    foster_results_df: pd.DataFrame | None = None,
    bin_set: tuple[list[float], list[str]] | None = None,
):
    metrics = ["mae"]
    if bin_set is None:
        bin_set = (constants.GEYIN_VS30_BINS, constants.GEYIN_VS30_BIN_NAMES)
    bins, bin_names = bin_set

    print("------------------------------------------")
    print(f"Metrics for Vs30 bins: {bin_names}")

    cur_df = results_df.copy()
    cur_df["cur_vs30_bin"] = pd.cut(cur_df["vs30"], bins=bins, labels=bin_names)

    grouped = cur_df.groupby("cur_vs30_bin", observed=True)[metrics].agg(
        ["mean", "std"]
    )
    counts = cur_df.groupby("cur_vs30_bin", observed=True)[metrics[0]].count()

    display_df = pd.DataFrame(index=grouped.index)
    display_df["n"] = counts
    for metric in metrics:
        display_df[metric] = (
            grouped[(metric, "mean")]
            .map("{:.3f}".format)
            .str.cat(grouped[(metric, "std")].map(" \u00b1 {:.3f}".format))
        )

    if foster_results_df is not None:
        assert results_df.index.isin(
            foster_results_df.index
        ).all(), "Not all sites in results_df are present in foster_results_df"
        foster_df = foster_results_df.loc[
            foster_results_df.index.intersection(results_df.index)
        ].copy()

        # Load data points Foster was developed on
        foster_data = pd.read_csv(constants.FOSTER_DATA_FFP)
        foster_data["location_id"] = foster_data["location_id"].astype(str).str.strip()
        foster_data = foster_data.loc[foster_data.location_id != "NA"]

        # Get number of Foster data points in each bin
        foster_df["in_foster_data"] = foster_df.index.isin(
            foster_data.location_id.values
        )

        # Compute number of points in Chch region
        cur_df["in_chch_region"] = (
            (cur_df["lon"] >= constants.CHCH_REGION_BOUNDING_BOX[0])
            & (cur_df["lon"] <= constants.CHCH_REGION_BOUNDING_BOX[1])
            & (cur_df["lat"] >= constants.CHCH_REGION_BOUNDING_BOX[2])
            & (cur_df["lat"] <= constants.CHCH_REGION_BOUNDING_BOX[3])
        )

        foster_df["cur_vs30_bin"] = pd.cut(
            foster_df["vs30"], bins=bins, labels=bin_names
        )
        foster_mae = foster_df.groupby("cur_vs30_bin", observed=True)["mae"].agg(
            ["mean", "std"]
        )
        model_mae = grouped[("mae", "mean")]

        display_df["foster_mae"] = (
            foster_mae["mean"]
            .map("{:.3f}".format)
            .str.cat(foster_mae["std"].map(" \u00b1 {:.3f}".format))
        )
        display_df["pct_improvement"] = (
            (foster_mae["mean"] - model_mae) / foster_mae["mean"] * 100
        ).map("{:+.1f}%".format)
        display_df["mae_improvement"] = (foster_mae["mean"] - model_mae).map(
            "{:+.3f}".format
        )

        display_df["n_foster_points"] = (
            foster_df.groupby("cur_vs30_bin", observed=True)["in_foster_data"]
            .sum()
            .map("{:d}".format)
        )

        display_df["n_chch_points"] = (
            cur_df.groupby("cur_vs30_bin", observed=True)["in_chch_region"]
            .sum()
            .map("{:d}".format)
        )

    print(display_df.to_string())
