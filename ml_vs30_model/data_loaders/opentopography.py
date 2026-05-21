from pathlib import Path
import re
import tarfile
import logging

import pandas as pd
import numpy as np
import rasterio
from rasterio import transform
from rclone_python import rclone

from .. import constants
from .. import utils
from . import utils as data_loader_utils
from .base_loader import BaseLoader


logger = logging.getLogger(__name__)


class OpenTopographyS3Loader(BaseLoader):
    """Class for accessing geospatial data from OpenTopography S3 bucket."""

    RCLONE_CONFIG_FFP = (
        Path(__file__).parent.parent.parent / "resources" / "rclone.conf"
    )

    def __init__(self, base_input_data_dir: Path):
        self.base_base_input_data_dir = base_input_data_dir

        rclone.set_config_file(self.RCLONE_CONFIG_FFP)

        # Cache variables
        self.variable_file_dfs_cache: dict[constants.InputVariable, list[str]] = {}

    def get_values(
        self, coords: np.ndarray, variable: constants.InputVariable, address_missing: bool = True
    ) -> np.ndarray:
        """Get the values of the specified variable at the given lat/lon coordinates."""
        data_dir = self.base_base_input_data_dir / variable

        values = None
        filenames = self._get_tif_filenames(coords, variable)

        unique_filenames = np.unique(filenames)

        for filename in unique_filenames:
            # Download files if not already present
            if not (data_dir / filename).exists():
                self._download_tif_file(str(filename), variable)

            # Get values
            mask = filenames == filename
            cur_coords = coords[mask]
            cur_values = self._get_values(cur_coords, data_dir / filename, variable)

            if values is None:
                values = np.empty(coords.shape[0], dtype=cur_values.dtype)

            missing_mask = (cur_values == constants.INTEGER_NO_DATA_VALUE) | np.isnan(cur_values)
            if address_missing and np.any(missing_mask):
                logger.info(
                    f"Found {np.sum(missing_mask)}/{cur_values.shape[0]} missing values for variable {variable} "
                    f"for filename {filename}. Using nearest non-missing value."
                )
                cur_values[missing_mask] = data_loader_utils.find_nearest_valid_wgs84(
                    data_dir / filename,
                    cur_coords[missing_mask],
                    lambda v: (v != -9999) & ~np.isnan(v),
                    cur_values.dtype,
                )

            values[mask] = cur_values

        return values

    def get_region_values(
        self,
        lon_values: np.ndarray,
        lat_values: np.ndarray,
        width_m: float,
        height_m: float,
        variable: constants.InputVariable,
    ) -> np.ndarray:
        """
        Get the values of the specified variable for a region defined by a centre point and bounding box dimensions.
        """
        n_coords = lon_values.shape[0]
        logger.info(
            f"Getting region values for variable {variable} for {n_coords} coordinates"
            f" with width {width_m}m and height {height_m}m."
        )
        corner_coords = utils.get_bounding_box_corners(
            lon_values, lat_values, width_m, height_m
        )

        results = []
        for i in range(n_coords):
            # Bounds
            cur_corner_coords = corner_coords[i]
            x_min, y_min = cur_corner_coords[:, 0].min(), cur_corner_coords[:, 1].min()
            x_max, y_max = cur_corner_coords[:, 0].max(), cur_corner_coords[:, 1].max()

            # Get filenames & download files if needed
            tif_files = self._get_tif_filenames(cur_corner_coords, variable)
            for tif_file in tif_files:
                if not (self.base_base_input_data_dir / variable / tif_file).exists():
                    logger.info(
                        f"TIFF file {tif_file} not found locally. Downloading..."
                    )
                    if failed_download := self._download_tif_file(
                        str(tif_file), variable
                    ):
                        break
            if failed_download:
                logger.warning(
                    f"Failed to download all required TIFF files for coordinate {i}. Skipping."
                )
                continue

            # Sanity check
            if i == 0:
                with rasterio.open(
                    self.base_base_input_data_dir / variable / tif_files[0]
                ) as dataset:
                    assert (
                        dataset.crs.to_epsg() == constants.WGS84_EPSG
                    ), "Dataset CRS is not WGS84"

            # Region is across multiple tiles
            if len(set(tif_files)) > 1:
                cur_results, _ = rasterio.merge.merge(
                    [
                        rasterio.open(self.base_base_input_data_dir / variable / tif)
                        for tif in tif_files
                    ],
                    bounds=[x_min, y_min, x_max, y_max],
                )
                # Close the files
                [src.close() for src in tif_files]
            else:
                with rasterio.open(
                    self.base_base_input_data_dir / variable / tif_files[0]
                ) as dataset:
                    window = rasterio.windows.from_bounds(
                        x_min, y_min, x_max, y_max, transform=dataset.transform
                    )
                    cur_results = dataset.read(1, window=window)

            results.append(cur_results)

        return results

    def _get_tif_filenames(
        self, coords: np.ndarray, variable: constants.InputVariable
    ) -> np.ndarray:
        """Get the filename of the TIFF file for the given lat/lon and variable."""
        raise NotImplementedError(
            "Subclasses must implement _get_tif_filenames method."
        )

    def _download_tif_file(
        self, tif_filename: str, variable: constants.InputVariable
    ) -> bool:
        raise NotImplementedError(
            "Subclasses must implement _download_tif_file method."
        )

    def _get_values(self, coords: np.ndarray, tif_ffp: Path, variable: constants.InputVariable) -> np.ndarray:
        """Get the values from the TIFF file at the specified coordinates."""
        if not tif_ffp.exists():
            utils.raise_log(
                FileNotFoundError,
                f"TIF file not found at {tif_ffp}.",
                logger,
            )

        with rasterio.open(tif_ffp) as dataset:
            assert (
                dataset.crs.to_epsg() == constants.WGS84_EPSG
            ), "Dataset CRS is not WGS84"

            data = dataset.read()
            assert data.shape[0] == 1, "Expected single-band TIFF file."
            rows, cols = transform.rowcol(dataset.transform, coords[:, 0], coords[:, 1])
            values = data[0, rows, cols]

            values = data_loader_utils.convert_dtype_and_handle_nodata(values, dataset.nodatavals[0], variable)

        return values


class SRTMGL1(OpenTopographyS3Loader):
    """Class for accessing SRTM GL1 data."""

    SUPPORTED_VARIABLES = [
        constants.InputVariable.Elevation,
    ]

    S3_BASE_PATH = "opentopo:raster/SRTM_GL1/SRTM_GL1_srtm"

    def __init__(
        self, base_input_data_dir: Path = constants.BASE_DATA_DIR / "input_data" / "raw"
    ):
        super().__init__(base_input_data_dir)

    def get_values(
        self, coords: np.ndarray, variable: constants.InputVariable, address_missing: bool = True
    ) -> np.ndarray:
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by SRTMGL1.",
                logger,
            )

        return super().get_values(coords, variable, address_missing=address_missing)

    def _download_tif_file(
        self, tif_filename: str, variable: constants.InputVariable
    ) -> bool:
        """Download the specified TIFF file for SRTM GL1."""
        assert (
            variable == constants.InputVariable.Elevation
        ), "SRTMGL1 only supports Elevation variable."

        # Ensure output directory exists
        (out_dir := self.base_base_input_data_dir / variable).mkdir(exist_ok=True)
        logger.info(f"Downloading {tif_filename} to {out_dir}")
        rclone.copy(
            self.S3_BASE_PATH + f"/{tif_filename}",
            out_dir,
            show_progress=False,
        )

        if not (out_dir / tif_filename).exists():
            logger.warning(f"Failed to download TIFF file {tif_filename}.")
            return False
        return True

    def _get_tif_filenames(
        self, coords: np.ndarray, variable: constants.InputVariable
    ) -> np.ndarray:
        """Get the filename of the SRTM GL1 TIFF file for the given lat/lon."""
        assert (
            variable == constants.InputVariable.Elevation
        ), "SRTMGL1 only supports Elevation variable."
        lon_values, lat_values = coords[:, 0], coords[:, 1]

        filenames = []
        for lat, lon in zip(lat_values, lon_values):
            lat_prefix = "N" if lat >= 0 else "S"
            lon_prefix = "E" if lon >= 0 else "W"

            lat_idx = int(abs(np.floor(lat)))
            lon_idx = int(abs(np.floor(lon)))

            filenames.append(f"{lat_prefix}{lat_idx:02d}{lon_prefix}{lon_idx:03d}.tif")

        return np.array(filenames)


class GeoMorpho90(OpenTopographyS3Loader):
    """Class for accessing GeoMorpho90 data."""

    SUPPORTED_VARIABLES = [
        constants.InputVariable.Roughness,
        constants.InputVariable.TopographicSlope,
        constants.InputVariable.CompoundTopgraphicIndex,
        constants.InputVariable.Geomorphon,
        constants.InputVariable.ProfileCurvature,
        constants.InputVariable.TangentialCurvature,
        constants.InputVariable.TerrainRuggednessIndex,
        constants.InputVariable.TopographicPositionIndex,
        constants.InputVariable.VectorRuggednessMeasure,
    ]

    S3_VAR_NAME_MAP = {
        constants.InputVariable.Roughness: "roughness",
        constants.InputVariable.TopographicSlope: "slope",
        constants.InputVariable.CompoundTopgraphicIndex: "cti",
        constants.InputVariable.Geomorphon: "geom",
        constants.InputVariable.ProfileCurvature: "pcurv",
        constants.InputVariable.TangentialCurvature: "tcurv",
        constants.InputVariable.TerrainRuggednessIndex: "tri",
        constants.InputVariable.TopographicPositionIndex: "tpi",
        constants.InputVariable.VectorRuggednessMeasure: "vrm",
    }

    S3_BASE_PATH = "opentopo:dataspace/OTDS.012020.4326.1/raster"
    REGEX_PATTERN = r"\w+_\d+M_([ns])(\d+)([ew])(\d+)\.(?:tif|tar\.gz)"

    def __init__(
        self, base_input_data_dir: Path = constants.BASE_DATA_DIR / "input_data" / "raw"
    ):
        super().__init__(base_input_data_dir)

    def get_values(
        self, coords: np.ndarray, variable: constants.InputVariable, address_missing: bool = True
    ) -> np.ndarray:
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by GeoMorpho90.",
                logger,
            )

        return super().get_values(coords, variable, address_missing=address_missing)

    def _get_tif_filenames(
        self, coords: np.ndarray, variable: constants.InputVariable
    ) -> np.ndarray:
        """Get the filename of the GeoMorpho90 TIFF file for the given lat/lon and variable."""
        # Convert lat/lon to filename using MERIT-DEM tiling system
        # Each tile is 5x5 degrees, named by lower left corner
        lon_lower_values = (coords[:, 0] // 5).astype(int) * 5
        lat_lower_values = (coords[:, 1] // 5).astype(int) * 5

        filenames = []
        for lon_lower, lat_lower in zip(lon_lower_values, lat_lower_values):
            # Format longitude component (e/w)
            if lon_lower >= 0:
                lon_str = f"e{abs(lon_lower):03d}"
            else:
                lon_str = f"w{abs(lon_lower):03d}"

            # Format latitude component (n/s)
            if lat_lower >= 0:
                lat_str = f"n{abs(lat_lower):02d}"
            else:
                lat_str = f"s{abs(lat_lower):02d}"

            filenames.append(
                f"{self.S3_VAR_NAME_MAP[variable]}_90M_{lat_str}{lon_str}.tif"
            )

        return np.array(filenames)

    def _download_tif_file(
        self, tif_filename: str, variable: constants.InputVariable
    ) -> bool:
        """Download and extract the correct gzip file for the specified TIFF filename and variable."""
        files_df = self._get_files_df(variable)

        # Find the correct gzip filename
        # File name coordinates are the bottom left corner of the tile
        query = self._nsew_lat_lon_from_filename(tif_filename)

        query_dlat = query[1] if query[0] == "n" else -query[1]
        query_dlon = query[3] if query[2] == "e" else -query[3]
        mask = (
            (files_df.ns == query[0])
            & (files_df.ew == query[2])
            & (files_df.dlat <= query_dlat)
            & (files_df.dlon <= query_dlon)
        )

        gzip_filename = (
            (query_dlat - files_df.loc[mask].dlat)
            + (query_dlon - files_df.loc[mask].dlon)
        ).idxmin()

        # Ensure output directory exists
        (out_dir := self.base_base_input_data_dir / variable).mkdir(exist_ok=True)
        logger.info(f"Downloading {gzip_filename} to {out_dir}")
        rclone.copy(
            self.S3_BASE_PATH + f"/{self.S3_VAR_NAME_MAP[variable]}/" + gzip_filename,
            out_dir,
            show_progress=False,
        )

        if not (gzip_ffp := out_dir / gzip_filename).exists():
            error_msg = f"Failed to download file: {gzip_filename}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Extract the TIFF from the gzip file
        logger.info(f"Extracting {gzip_ffp} to {out_dir}")
        with tarfile.open(gzip_ffp, "r") as tar:
            tar.extractall(path=out_dir)

        # Remove the gzip file after extraction
        gzip_ffp.unlink()

        return True

    def _nsew_lat_lon_from_filename(self, filename: str) -> tuple[str, int, str, int]:
        """Extract the n/s, lat, e/w, lon components from the filename."""
        if match := re.search(self.REGEX_PATTERN, filename):
            ns = match.group(1)  # "n" or "s"
            lat_value = int(match.group(2))
            ew = match.group(3)  # "e" or "w"
            lon_value = int(match.group(4))
        else:
            error_msg = f"Filename does not match expected pattern: {filename}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return ns, lat_value, ew, lon_value

    def _get_files_df(self, variable: constants.InputVariable) -> list[str]:
        """List all available GeoMorpho90 files for the specified variable."""
        if variable not in self.variable_file_dfs_cache:
            dir_content = rclone.ls(
                self.S3_BASE_PATH + f"/{self.S3_VAR_NAME_MAP[variable]}"
            )

            file_dict = {
                item["Name"]: self._nsew_lat_lon_from_filename(item["Name"])
                for item in dir_content
                if item["MimeType"] == "application/gzip"
            }
            files_df = pd.DataFrame.from_dict(
                file_dict, orient="index", columns=["ns", "lat", "ew", "lon"]
            )
            files_df["dlon"] = np.where(
                files_df["ew"] == "w", files_df["lon"] * -1, files_df["lon"]
            )
            files_df["dlat"] = np.where(
                files_df["ns"] == "s", files_df["lat"] * -1, files_df["lat"]
            )
            self.variable_file_dfs_cache[variable] = files_df

        return self.variable_file_dfs_cache[variable]
