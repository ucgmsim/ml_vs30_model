import logging
from pathlib import Path
from typing import Callable

import einops
import numpy as np
from pyproj import Geod, Transformer

import rasterio
from rasterio import transform

from .. import constants
from .. import utils
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)

class TIFLoader(BaseLoader):
    """Class for retrieving data from downloaded TIFF files."""

    SUPPORTED_VARIABLES = {
        constants.InputVariable.LandformShannonIndex,
        constants.InputVariable.LandformEntropy,
        constants.InputVariable.LandformUniformity,
        constants.InputVariable.AbsoluteDepthToBedrock,
    }

    VAR_TO_FILENAME_MAP = {
        constants.InputVariable.LandformShannonIndex: "raw/geom_1KMsha_GMTEDmd.tif",
        constants.InputVariable.LandformEntropy: "raw/geom_1KMent_GMTEDmd.tif",
        constants.InputVariable.LandformUniformity: "raw/geom_1KMuni_GMTEDmd.tif",
        constants.InputVariable.AbsoluteDepthToBedrock: "raw/absolute_depth_to_bedrock/BDTICM_M_250m_ll.tif",
    }

    def __init__(
        self, base_input_data_dir: Path = constants.BASE_DATA_DIR / "input_data"
    ) -> None:
        self.base_input_data_dir = base_input_data_dir

    def get_values(self, coords: np.ndarray, variable: constants.InputVariable, address_missing: bool = True):
        """
        Get values for the given WGS84 (lon, lat) coordinates.
        
        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 2) containing coordinates as (lon, lat) in WGS84.
        variable : constants.InputVariable
            The variable to retrieve values for.
        address_missing : bool
            Whether to address missing values (e.g. negative depth to bedrock) 
            by finding nearest valid value. Can be slow for large number of points.
        """
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by TIFLoader.",
                logger,
            )

        tif_ffp = self.base_input_data_dir / self.VAR_TO_FILENAME_MAP[variable]
        if not tif_ffp.exists():
            utils.raise_log(
                FileNotFoundError,
                f"TIF file for variable {variable} not found at {tif_ffp}.",
                logger,
            )

        with rasterio.open(tif_ffp) as dataset:
            assert (
                dataset.crs.to_epsg() == constants.WGS84_EPSG
            ), "Dataset CRS is not WGS84."
            data = dataset.read()
            assert data.shape[0] == 1, "Expected single-band TIFF file."
            rows, cols = transform.rowcol(dataset.transform, coords[:, 0], coords[:, 1])
            values = data[0, rows, cols]

        # Deal with negative absolute depth to bedrock values
        if address_missing and variable == constants.InputVariable.AbsoluteDepthToBedrock:
            mask = values < 0
            if np.any(mask):
                logger.info(
                    f"Found {np.sum(mask)} negative values for Absolute Depth to Bedrock. "
                    f"Using nearest non-zero value. This can be slow for large number of points."
                )

                values[mask] = find_nearest_valid(
                    tif_ffp, coords[mask], lambda v: v >= 0, values.dtype
                )

        return values

class NZTMTIFLoader:
    """Class for retrieving data from downloaded TIFF files in NZTM projection.
    
    Accepts WGS84 (lon/lat) coordinates and converts them to NZTM before querying.
    """

    SUPPORTED_VARIABLES = {
        constants.InputVariable.NZEnvDSDistanceRivers,
        constants.InputVariable.NZEnvDSDistanceRiversVertical,
        constants.InputVariable.NZEnvDSPrecipAnn,
        constants.InputVariable.NZEnvDSSlopeDeg,
        constants.InputVariable.NZEnvDSSoilAcidP,
        constants.InputVariable.NZEnvDSSoilAge,
        constants.InputVariable.NZEnvDSSoilDrainage,
        constants.InputVariable.NZEnvDSSoilInduration,
        constants.InputVariable.NZEnvDSSoilParticleSize,
        constants.InputVariable.NZEnvDSTopoGeomorphons,
        constants.InputVariable.NZEnvDSTopoNormalisedHeight,
        constants.InputVariable.NZEnvDSTopoPosition,
        constants.InputVariable.NZEnvDSTopoRoughness,
        constants.InputVariable.NZEnvDSTopoRuggedness,
        constants.InputVariable.NZEnvDSTopoValleyDepth,
        constants.InputVariable.NZEnvDSTopoWetness,
        constants.InputVariable.NZNLMGroundwaterDepth,
        constants.InputVariable.NZNWTGroundwaterDepth,
    }

    VAR_TO_FILENAME_MAP = {
        # constants.InputVariable.NZEnvDSDistanceRivers: "nzenvds_v1p1_nztm/final_layers_nztm/distance_rivers.tif",
        # constants.InputVariable.NZEnvDSDistanceRiversVertical: "nzenvds_v1p1_nztm/final_layers_nztm/distance_riversVertical.tif",
        # constants.InputVariable.NZEnvDSPrecipAnn: "nzenvds_v1p1_nztm/final_layers_nztm/precip_ann.tif",
        # constants.InputVariable.NZEnvDSSlopeDeg: "nzenvds_v1p1_nztm/final_layers_nztm/slope_deg.tif",
        constants.InputVariable.NZEnvDSSoilAcidP: "nzenvds_v1p1_nztm/final_layers_nztm/soil_acidP.tif",
        constants.InputVariable.NZEnvDSSoilAge: "nzenvds_v1p1_nztm/final_layers_nztm/soil_age.tif",
        # constants.InputVariable.NZEnvDSSoilDrainage: "nzenvds_v1p1_nztm/final_layers_nztm/soil_drainage.tif",
        constants.InputVariable.NZEnvDSSoilInduration: "nzenvds_v1p1_nztm/final_layers_nztm/soil_induration.tif",
        # constants.InputVariable.NZEnvDSSoilParticleSize: "nzenvds_v1p1_nztm/final_layers_nztm/soil_particleSize.tif",
        constants.InputVariable.NZEnvDSTopoGeomorphons: "nzenvds_v1p1_nztm/final_layers_nztm/topo_geomorphons.tif",
        # constants.InputVariable.NZEnvDSTopoNormalisedHeight: "nzenvds_v1p1_nztm/final_layers_nztm/topo_normalisedHeight.tif",
        # constants.InputVariable.NZEnvDSTopoPosition: "nzenvds_v1p1_nztm/final_layers_nztm/topo_position.tif",
        # constants.InputVariable.NZEnvDSTopoRoughness: "nzenvds_v1p1_nztm/final_layers_nztm/topo_roughness.tif",
        constants.InputVariable.NZEnvDSTopoRuggedness: "nzenvds_v1p1_nztm/final_layers_nztm/topo_ruggedness.tif",
        # constants.InputVariable.NZEnvDSTopoValleyDepth: "nzenvds_v1p1_nztm/final_layers_nztm/topo_valleyDepth.tif",
        # constants.InputVariable.NZEnvDSTopoWetness: "nzenvds_v1p1_nztm/final_layers_nztm/topo_wetness.tif",
        # constants.InputVariable.NZNLMGroundwaterDepth: "nz_nlm/NLM_gwd.tif",
        # constants.InputVariable.NZNWTGroundwaterDepth: "nz_nwt/nwt_wtd_NZ_20220825.tif",
    }

    _WGS84_TO_NZTM = Transformer.from_crs(
        constants.WGS84_EPSG, constants.NZTM2000_EPSG, always_xy=True
    )

    def __init__(
        self, base_input_data_dir: Path = constants.BASE_DATA_DIR / "input_data"
    ) -> None:
        self.base_input_data_dir = base_input_data_dir

    def get_values(self, coords: np.ndarray, variable: constants.InputVariable):
        """
        Get values for the given WGS84 (lon, lat) coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 2) containing coordinates as (lon, lat) in WGS84.
        variable : constants.InputVariable
            The variable to retrieve values for.
        """
        if variable not in self.SUPPORTED_VARIABLES:
            utils.raise_log(
                ValueError,
                f"Variable {variable} is not supported by NZTMTIFLoader.",
                logger,
            )

        tif_ffp = constants.INPUT_VAR_TO_FFP_MAP.get(variable)
        if not tif_ffp:
            tif_ffp = self.base_input_data_dir / self.VAR_TO_FILENAME_MAP[variable]
        if not tif_ffp.exists():
            utils.raise_log(
                FileNotFoundError,
                f"TIF file for variable {variable} not found at {tif_ffp}.",
                logger,
            )

        nztm_coords = self._to_nztm(coords)
        with rasterio.open(tif_ffp) as dataset:
            assert (
                dataset.crs.to_epsg() == constants.NZTM2000_EPSG
            ), "Dataset CRS is not NZTM2000."
            data = dataset.read()
            assert data.shape[0] == 1, "Expected single-band TIFF file."
            rows, cols = transform.rowcol(dataset.transform, nztm_coords[:, 0], nztm_coords[:, 1])
            values = data[0, rows, cols]

        return values

    @classmethod
    def _to_nztm(cls, coords: np.ndarray) -> np.ndarray:
        """Convert WGS84 (lon, lat) coordinates to NZTM (easting, northing).

        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 2) containing coordinates as (lon, lat).

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) containing NZTM (easting, northing) coordinates.
        """
        easting, northing = cls._WGS84_TO_NZTM.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([easting, northing])


def find_nearest_valid(
    tif_ffp: Path,
    invalid_coords: np.ndarray,
    valid_check: Callable[[np.ndarray], np.ndarray],
    dtype: np.dtype,
) -> np.ndarray:
    """
    Find nearest valid values for given coordinates in a TIFF file.

    Parameters
    ----------
    tif_ffp : Path
        File path to the TIFF file.
    invalid_coords : np.ndarray
        Array of shape (N, 2) containing coordinates (lon, lat) with invalid values.
    valid_check : Callable[[np.ndarray], np.ndarray]
        Function that takes an array of values and returns a boolean array indicating valid values.
    dtype : np.dtype
        Data type of the values to be returned.

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing the nearest valid values for the given coordinates.
    """
    geod = Geod(ellps="WGS84")
    updated_values = np.empty(invalid_coords.shape[0], dtype=dtype)
    for i in range(invalid_coords.shape[0]):
        # Create small meshgrid around the point
        lon, lat = invalid_coords[i]
        lat_grid = np.linspace(lat - (0.0001 * 50 / 2), lat + (0.0001 * 50 / 2), 50)
        lon_grid = np.linspace(lon - (0.0001 * 50 / 2), lon + (0.0001 * 50 / 2), 50)
        meshgrid = np.array(np.meshgrid(lon_grid, lat_grid))

        meshgrid_coords = einops.rearrange(meshgrid, "d h w -> (h w) d", d=2)
        meshgrid_dist = geod.inv(
            np.full(meshgrid_coords.shape[0], lon),
            np.full(meshgrid_coords.shape[0], lat),
            meshgrid_coords[:, 0],
            meshgrid_coords[:, 1],
        )[2]
        with rasterio.open(tif_ffp) as dataset:
            meshgrid_values = np.concatenate(list(dataset.sample(meshgrid_coords)))

        # Mask out negative values
        valid_mask = valid_check(meshgrid_values)
        if np.any(valid_mask):
            nearest_idx = np.argmin(meshgrid_dist[valid_mask])
            logger.debug(
                f"Replacing value at coords {invalid_coords[i]} with nearest valid value "
                f"at distance {meshgrid_dist[valid_mask][nearest_idx]:.2f} m."
            )
            updated_values[i] = meshgrid_values[valid_mask][nearest_idx]

    return updated_values