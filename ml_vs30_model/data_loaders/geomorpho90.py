from pathlib import Path
from enum import StrEnum
import re
import tarfile
import logging

import pandas as pd
import numpy as np
import rasterio
from rclone_python import rclone
from tqdm import tqdm

from .. import constants
from .. import utils
from .tif_loader import find_nearest_valid


logger = logging.getLogger(__name__)


class GeoMorpho90:
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

    RCLONE_CONFIG_FFP = Path(__file__).parent.parent.parent / "resources" / "rclone.conf"
    S3_BASE_PATH = "opentopo:dataspace/OTDS.012020.4326.1/raster"
    REGEX_PATTERN = r"\w+_\d+M_([ns])(\d+)([ew])(\d+)\.(?:tif|tar\.gz)"

    def __init__(
        self, base_raw_data_dir: Path = constants.BASE_DATA_DIR / "input_data" / "raw"
    ):
        self.base_raw_data_dir = base_raw_data_dir

        rclone.set_config_file(self.RCLONE_CONFIG_FFP)

        # Cache variables
        self.variable_file_dfs_cache: dict[constants.InputVariable, list[str]] = {}

    def get_values(
        self, coords: np.ndarray, variable: constants.InputVariable
    ) -> np.ndarray:
        """Get the values of the specified variable at the given lat/lon coordinates."""
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by GeoMorpho90.",
                logger,
            )

        data_dir = self.base_raw_data_dir / variable

        values = None
        filenames = self._get_tif_filenames(coords, variable)

        unique_filenames = np.unique(filenames)

        for filename in unique_filenames:
            # Download files if not already present
            if not (data_dir / filename).exists():
                self._download_tif_file(str(filename), variable)

            # Get values
            mask = filenames == filename
            cur_values = self._get_values(coords[mask], data_dir / filename)
            if values is None:
                values = np.empty(coords.shape[0], dtype=cur_values.dtype)
            values[mask] = cur_values

            missing_mask = values == -9999    
            if np.any(missing_mask):
                logger.info(
                    f"Found {np.sum(missing_mask)}/{coords.shape[0]} missing values for variable {variable}. "
                    f"Using nearest non-missing value."
                )
                values[missing_mask] = find_nearest_valid(
                    data_dir / filename, coords[missing_mask], lambda v: v != -9999, values.dtype
                )

        return values

    def _get_tif_filenames(
        self, coords: np.ndarray, variable: constants.InputVariable
    ) -> np.ndarray:
        """Get the filename of the GeoMorpho90 TIFF file for the given lat/lon and variable."""
        # Convert lat/lon to filename using MERIT-DEM tiling system
        # Each tile is 5x5 degrees, named by lower left corner
        lon_values = coords[:, 0]
        lat_values = coords[:, 1]

        lon_lower_values = (lon_values // 5).astype(int) * 5
        lat_lower_values = (lat_values // 5).astype(int) * 5

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

    def _get_values(self, coords: np.ndarray, tif_ffp: Path) -> np.ndarray:
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

            return np.concatenate(list(dataset.sample(coords)))

    def _download_tif_file(
        self, tif_filename: str, variable: constants.InputVariable
    ) -> None:
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
        (out_dir := self.base_raw_data_dir / variable).mkdir(exist_ok=True)
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
