import logging
from pathlib import Path

import rasterio
from rasterio import transform
import shap
import xarray as xr
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import ml_tools as mlt


from .plotting import model_perf_plots, spatial, feature_importance_plots
from .configs import RunConfig
from . import pre_processing
from . import constants

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
    Adds mean absolute error (MAE) to the provided results dataframe.
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


def compute_shap_feature_importance(
    model_dir: Path,
    run_config: RunConfig | None = None,
    train_results: pd.DataFrame | None = None,
    val_results: pd.DataFrame | None = None,
    model: CatBoostRegressor | None = None,
) -> None:
    """
    Computes SHAP feature importance
    for the model in the provided directory.
    """
    logger.info("Computing SHAP feature importance...")
    run_config = (
        RunConfig.from_yaml(model_dir / "run_config.yaml")
        if run_config is None
        else run_config
    )
    dataset_df = pd.read_parquet(run_config.dataset_ffp)

    # Load training results
    train_results = (
        pd.read_parquet(model_dir / "train_results.parquet")
        if train_results is None
        else train_results
    )

    # Load validation results if available
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
        if run_config.model_type == "catboost":
            model = CatBoostRegressor()
            model.load_model(model_dir / "model.cbm")
        else:
            raise ValueError(
                f"Unsupported model type {run_config.model_type} "
                "for SHAP feature importance computation."
            )

    explainer = shap.TreeExplainer(model, train_X)
    # explainer = shap.TreeExplainer(model)
    explainer_values = explainer(train_X if val_results is None else val_X)
    mlt.utils.write_pickle(explainer_values, model_dir / "shap_values.pkl")

    return explainer_values


def gen_model_perfomance_plots(
    results_dir: Path, results_df: pd.DataFrame | None = None
) -> None:
    """
    Generates model performance plots for the provided results directory.
    """
    if results_df is None:
        logger.info("Reading results dataframe from parquet file val_results.parquet.")
        results_df = pd.read_parquet(results_dir / "val_results.parquet")

    logger.info("Generating model performance plots...")
    (outdir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    model_perf_plots.one_to_one_plot(results_df, outdir / "one_to_one_plot.png")
    model_perf_plots.residuals_histogram(results_df, outdir / "residuals_histogram.png")
    model_perf_plots.residual_kde(results_df, outdir / "residuals_kde.png")
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "mae_scatter_plot.png",
        metric_name="mae",
        x_limits=(0, 2000),
        y_limits=(0, 1000),
        show_geyin_maurer_model=True,
    )
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "lnVs30_mse_scatter_plot.png",
        metric_name="lnVs30_mse",
        x_limits=(0, 2000),
        # y_limits=(0, 1000),
    )
    model_perf_plots.metric_scatter_plot(
        results_df,
        outdir / "ln_residual_scatter_plot.png",
        metric_name="ln_residual",
        x_limits=(0, 2000),
        y_limits=(-1.5, 1.5),
    )


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
    if gen_waterfall_plots:
        feature_importance_plots.shap_waterfall(
            shap_values, results_df, outdir / "waterfall_plots"
        )


def _get_dataset_values(dataset_ffp: Path):
    """
    Helper function to get the
    coordinates, nan mask, and vs30 data array.
    """
    with xr.open_dataset(dataset_ffp, mode="r") as ds:
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

    ds = xr.Dataset(
        {
            f"{prefix}_vs30_mean": vs30_mean_da,
            f"{prefix}_vs30_res": res_da,
            f"{prefix}_vs30_ln_res": ln_res_da,
        }
    )
    if std_estimate is not None:
        ds[f"{prefix}_vs30_std"] = vs30_std_da

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

        foster_data = ds.read([1, 2])

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
        foster_combined_mvn_vs30_mean = foster_data[0, rows, cols]
        foster_combined_mvn_vs30_std = foster_data[1, rows, cols]

    # # Compute residual
    # res = foster_combined_mvn_vs30_mean - vs30_da.values[~nan_mask]
    # ln_res = np.log(foster_combined_mvn_vs30_mean) - np.log(vs30_da.values[~nan_mask])

    # # Create data arrays
    # foster_combined_mvn_vs30_mean_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # foster_combined_mvn_vs30_mean_da.values[~nan_mask] = foster_combined_mvn_vs30_mean
    # foster_combined_mvn_vs30_std_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # foster_combined_mvn_vs30_std_da.values[~nan_mask] = foster_combined_mvn_vs30_std
    # res_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # res_da.values[~nan_mask] = res
    # ln_res_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # ln_res_da.values[~nan_mask] = ln_res

    ds = _create_dataset_from_estimates(
        vs30_da,
        nan_mask,
        foster_combined_mvn_vs30_mean,
        "foster_combined_mvn",
        std_estimate=foster_combined_mvn_vs30_std,
    )

    # Save
    logger.info("Saving modified Foster et al. estimates to dataset...")
    # xr.Dataset(
    #     {
    #         "foster_combined_mvn_vs30_mean": foster_combined_mvn_vs30_mean_da,
    #         "foster_combined_mvn_vs30_std": foster_combined_mvn_vs30_std_da,
    #         "foster_combined_mvn_vs30_res": res_da,
    #         "foster_combined_mvn_vs30_ln_res": ln_res_da,
    #     }
    # )
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

    # with xr.open_dataset(dataset_ffp, mode="r") as ds:
    #     vs30_da = ds["vs30"]
    #     y, x = vs30_da.y.values, vs30_da.x.values
    #     nan_mask = vs30_da.isnull().values

    # grid_x, grid_y = np.meshgrid(x, y)
    # coords = np.column_stack((grid_x[~nan_mask], grid_y[~nan_mask]))

    coords, nan_mask, vs30_da = _get_dataset_values(dataset_ffp)

    logger.info("Extracting Jaehwi's estimates...")
    with rasterio.open(jaehwi_data_ffp) as ds:
        assert ds.crs.to_epsg() == constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"

        jw_data = ds.read([1, 2])

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
        jw_vs30_mean, jw_vs30_std = jw_data[0, rows, cols], jw_data[1, rows, cols]

    # # Compute residual
    # res = jw_vs30_mean - vs30_da.values[~nan_mask]
    # ln_res = np.log(jw_vs30_mean) - np.log(vs30_da.values[~nan_mask])

    # # Create data arrays
    # jw_vs30_mean_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # jw_vs30_mean_da.values[~nan_mask] = jw_vs30_mean
    # jw_vs30_std_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # jw_vs30_std_da.values[~nan_mask] = jw_vs30_std
    # res_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # res_da.values[~nan_mask] = res
    # ln_res_da = xr.DataArray(
    #     data=np.full(vs30_da.shape, np.nan), coords=[y, x], dims=["y", "x"]
    # )
    # ln_res_da.values[~nan_mask] = ln_res
    ds = _create_dataset_from_estimates(
        vs30_da,
        nan_mask,
        jw_vs30_mean,
        prefix,
        std_estimate=jw_vs30_std,
    )

    # Save
    logger.info("Saving Jaehwi's estimates to dataset...")
    # xr.Dataset(
    #     {
    #         f"{prefix}_vs30_mean": jw_vs30_mean_da,
    #         f"{prefix}_vs30_std": jw_vs30_std_da,
    #         f"{prefix}_vs30_res": res_da,
    #         f"{prefix}_vs30_ln_res": ln_res_da,
    #     }
    # )
    ds.to_netcdf(dataset_ffp, mode="a")


def add_foster_original_nz_estimates(dataset_ffp: Path, foster_original_ffp: Path):
    logger.info(f"Adding original Foster et al. result to database {dataset_ffp}...")
    coords, nan_mask, vs30_da = _get_dataset_values(dataset_ffp)

    logger.info("Extracting original Foster et al. estimates...")
    with rasterio.open(foster_original_ffp) as ds:
        assert ds.crs.to_epsg() == constants.NZTM2000_EPSG, "Dataset CRS is not NZTM"

        foster_original_data = ds.read(1)

        rows, cols = transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
        foster_original_vs30_mean = foster_original_data[rows, cols]

    ds = _create_dataset_from_estimates(
        vs30_da,
        nan_mask,
        foster_original_vs30_mean,
        "foster_original",
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
        data=np.full(vs30_da.shape, np.nan), coords=[vs30_da.y.values, vs30_da.x.values], dims=["y", "x"]
    )
    res_da.values[~nan_mask] = res

    ln_res_da = xr.DataArray(
        data=np.full(vs30_da.shape, np.nan), coords=[vs30_da.y.values, vs30_da.x.values], dims=["y", "x"]
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

    print("wtf")
