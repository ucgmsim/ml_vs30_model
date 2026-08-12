from enum import StrEnum
from pathlib import Path
import os

from pyproj import Transformer
import pandas as pd
import numpy as np

if (BASE_DATA_DIR := os.getenv("VS30_MODEL_BASE_DATA_DIR")) is None:
    raise EnvironmentError("Environment variable VS30_MODEL_BASE_DATA_DIR is not set.")
BASE_DATA_DIR = Path(BASE_DATA_DIR)

INTEGER_NO_DATA_VALUE = -9999

QMAP_FFP = (
    BASE_DATA_DIR / "input_data/nz_geology/ShapeFiles/NZL_GNS_250K_geological_units.shp"
)


class DataSource(StrEnum):
    GeoMorpho90 = "geomorpho90"
    SRTMGL1 = "srtmgl1"
    TIFLoader = "tif_loader"
    NZTMTIFLoader = "nztm_tif_loader"
    GlobalGWT = "global_gwt"  # Global Groundwater Table
    ShapeLoader = "shape_loader"
    NZDistanceToCoast = "nz_distance_to_coast"
    NZDistanceToRiver = "nz_distance_to_river"
    NZQuaternaryRegion = "nz_quaternary_region"


class InputVariable(StrEnum):
    # Global variables
    Roughness = "roughness"
    TopographicSlope = "topographic_slope"
    CompoundTopgraphicIndex = "compound_topographic_index"
    Geomorphon = "geomorphon"
    ProfileCurvature = "profile_curvature"
    TangentialCurvature = "tangential_curvature"
    TerrainRuggednessIndex = "terrain_ruggedness_index"
    TopographicPositionIndex = "topographic_position_index"
    VectorRuggednessMeasure = "vector_ruggedness_measure"
    LandformEntropy = "landform_entropy"
    LandformShannonIndex = "landform_shannon_index"
    LandformUniformity = "landform_uniformity"
    AbsoluteDepthToBedrock = "absolute_depth_to_bedrock"
    DepthToGroundwater = "depth_to_groundwater"
    Elevation = "elevation"  # SRTMGL1 elevation data (30m)
    # NZ
    NZGeologyCategory = "nz_geology_category"  # Foster et al.
    NZDistanceToCoast = "nz_distance_to_coast"  # Manually computed
    NZDistanceToRiver_ST1 = "nz_distance_to_river_st1"  # Manually computed distance to river (Strahler 1st order)
    NZDistanceToRiver_ST2 = "nz_distance_to_river_st2"  # Manually computed distance to river (Strahler 2nd order)
    NZDistanceToRiver_ST3 = "nz_distance_to_river_st3"  # Manually computed distance to river (Strahler 3rd order)
    NZDistanceToRiver_ST4 = "nz_distance_to_river_st4"  # Manually computed distance to river (Strahler 4th order)
    NZDistanceToRiver_ST5 = "nz_distance_to_river_st5"  # Manually computed distance to river (Strahler 5th order)
    NZDistanceToRiver_ST6 = "nz_distance_to_river_st6"  # Manually computed distance to river (Strahler 6th order)
    NZDistanceToRiver_ST7 = "nz_distance_to_river_st7"  # Manually computed distance to river (Strahler 7th order)
    NZDistanceToRiver_ST8 = "nz_distance_to_river_st8"  # Manually computed distance to river (Strahler 8th order)
    NZDistanceToRiver_ST1_Greater = "nz_distance_to_river_st1_greater"  # Strahler 1st and greater order rivers
    NZDistanceToRiver_ST2_Greater = "nz_distance_to_river_st2_greater"  # Strahler 2nd and greater order rivers
    NZDistanceToRiver_ST3_Greater = "nz_distance_to_river_st3_greater"  # Strahler 3rd and greater order rivers
    NZDistanceToRiver_ST4_Greater = "nz_distance_to_river_st4_greater"  # Strahler 4th and greater order rivers
    NZDistanceToRiver_ST5_Greater = "nz_distance_to_river_st5_greater"  # Strahler 5th and greater order rivers
    NZDistanceToRiver_ST6_Greater = "nz_distance_to_river_st6_greater"  # Strahler 6th and greater order rivers
    NZDistanceToRiver_ST7_Greater = "nz_distance_to_river_st7_greater"  # Strahler 7th and greater order rivers
    NZGeologyAgeMin = "nz_geology_age_min"  # GNS Geology Units
    NZGeologyAgeMax = "nz_geology_age_max"  # GNS Geology Units
    NZGeologyAgeMid = "nz_geology_age_mid"
    NZGeologyAgeLnMid = "nz_geology_age_ln_mid"
    NZLithologyCategory = "nz_lithology_category"  # GNS Lithology Units
    NZGeologicalUnit = "nz_geological_unit"  # GNS Geological Unit ID
    NZNLMGroundwaterDepth = "nz_nlm_groundwater_depth"  # NLM Groundwater Depth Model
    NZNWTGroundwaterDepth = "nz_nwt_groundwater_depth"  # National Water Table
    NZCombinedGroundwaterDepth = (
        "nz_combined_groundwater_depth"  # Combined groundwater depth from NLM and NWT
    )
    NZCombinedGroundwaterDepthLn = "nz_combined_groundwater_depth_ln"
    NZQuaternaryRegion = "nz_quaternary_region"
    # NZEnvDS
    NZEnvDSDistanceRivers = "nzenvds_distance_rivers"
    NZEnvDSDistanceRiversVertical = "nzenvds_distance_rivers_vertical"
    NZEnvDSPrecipAnn = "nzenvds_precip_ann"
    NZEnvDSSlopeDeg = "nzenvds_slope_deg"
    NZEnvDSSoilAcidP = "nzenvds_soil_acid_p"
    NZEnvDSSoilAge = "nzenvds_soil_age"
    NZEnvDSSoilDrainage = "nzenvds_soil_drainage"
    NZEnvDSSoilInduration = "nzenvds_soil_induration"
    NZEnvDSSoilParticleSize = "nzenvds_soil_particle_size"
    NZEnvDSTopoGeomorphons = "nzenvds_topo_geomorphons"
    NZEnvDSTopoNormalisedHeight = "nzenvds_topo_normalised_height"
    NZEnvDSTopoPosition = "nzenvds_topo_position"
    NZEnvDSTopoRoughness = "nzenvds_topo_roughness"
    NZEnvDSTopoRuggedness = "nzenvds_topo_ruggedness"
    NZEnvDSTopoValleyDepth = "nzenvds_topo_valley_depth"
    NZEnvDSTopoWetness = "nzenvds_topo_wetness"


INPUT_VARIABLE_SOURCE_MAPPING = {
    InputVariable.Roughness: DataSource.GeoMorpho90,
    InputVariable.TopographicSlope: DataSource.GeoMorpho90,
    InputVariable.CompoundTopgraphicIndex: DataSource.GeoMorpho90,
    InputVariable.Geomorphon: DataSource.GeoMorpho90,
    InputVariable.ProfileCurvature: DataSource.GeoMorpho90,
    InputVariable.TangentialCurvature: DataSource.GeoMorpho90,
    InputVariable.TerrainRuggednessIndex: DataSource.GeoMorpho90,
    InputVariable.TopographicPositionIndex: DataSource.GeoMorpho90,
    InputVariable.VectorRuggednessMeasure: DataSource.GeoMorpho90,
    InputVariable.LandformEntropy: DataSource.TIFLoader,
    InputVariable.LandformShannonIndex: DataSource.TIFLoader,
    InputVariable.LandformUniformity: DataSource.TIFLoader,
    InputVariable.AbsoluteDepthToBedrock: DataSource.TIFLoader,
    InputVariable.DepthToGroundwater: DataSource.GlobalGWT,
    InputVariable.Elevation: DataSource.SRTMGL1,
    InputVariable.NZGeologyCategory: DataSource.ShapeLoader,
    InputVariable.NZDistanceToCoast: DataSource.NZDistanceToCoast,
    InputVariable.NZDistanceToRiver_ST1: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST2: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST3: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST4: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST5: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST6: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST7: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST8: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST1_Greater: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST2_Greater: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST3_Greater: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST4_Greater: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST5_Greater: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST6_Greater: DataSource.NZDistanceToRiver,
    InputVariable.NZDistanceToRiver_ST7_Greater: DataSource.NZDistanceToRiver,
    InputVariable.NZGeologyAgeMin: DataSource.ShapeLoader,
    InputVariable.NZGeologyAgeMax: DataSource.ShapeLoader,
    InputVariable.NZLithologyCategory: DataSource.ShapeLoader,
    InputVariable.NZGeologicalUnit: DataSource.ShapeLoader,
    InputVariable.NZQuaternaryRegion: DataSource.NZQuaternaryRegion,
    InputVariable.NZNLMGroundwaterDepth: DataSource.NZTMTIFLoader,
    InputVariable.NZNWTGroundwaterDepth: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSDistanceRivers: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSDistanceRiversVertical: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSPrecipAnn: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSSlopeDeg: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSSoilAcidP: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSSoilAge: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSSoilDrainage: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSSoilInduration: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSSoilParticleSize: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSTopoGeomorphons: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSTopoNormalisedHeight: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSTopoPosition: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSTopoRoughness: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSTopoRuggedness: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSTopoValleyDepth: DataSource.NZTMTIFLoader,
    InputVariable.NZEnvDSTopoWetness: DataSource.NZTMTIFLoader,
}

INPUT_VARIABLE_TO_NICE_NAME_MAPPING = {
    InputVariable.Elevation: "Elevation",
    InputVariable.Roughness: "Roughness",
    InputVariable.TopographicSlope: "Topographic Slope",
    InputVariable.CompoundTopgraphicIndex: "Compound Topographic Index",
    InputVariable.Geomorphon: "Geomorphon",
    InputVariable.ProfileCurvature: "Profile Curvature",
    InputVariable.TangentialCurvature: "Tangential Curvature",
    InputVariable.TerrainRuggednessIndex: "Terrain Ruggedness Index",
    InputVariable.TopographicPositionIndex: "Topographic Position Index",
    InputVariable.VectorRuggednessMeasure: "Vector Ruggedness Measure",
    InputVariable.LandformEntropy: "Landform Entropy",
    InputVariable.LandformShannonIndex: "Landform Shannon Index",
    InputVariable.LandformUniformity: "Landform Uniformity",
    InputVariable.AbsoluteDepthToBedrock: "Absolute Depth To Bedrock",
    InputVariable.DepthToGroundwater: "Depth To Groundwater",
    InputVariable.NZGeologyCategory: "NZ Geology Category",
    InputVariable.NZDistanceToCoast: "NZ Distance To Coast",
    InputVariable.NZGeologyAgeMin: "NZ Geology Age (Min)",
    InputVariable.NZGeologyAgeMax: "NZ Geology Age (Max)",
    InputVariable.NZGeologyAgeMid: "NZ Geology Age (Mid)",
    InputVariable.NZGeologyAgeLnMid: "NZ Geology Age (Ln Mid)",
    InputVariable.NZNLMGroundwaterDepth: "NZ NLM Groundwater Depth",
    InputVariable.NZNWTGroundwaterDepth: "NZ NWT Groundwater Depth",
    InputVariable.NZCombinedGroundwaterDepth: "NZ Groundwater Depth (Comb)",
    InputVariable.NZCombinedGroundwaterDepthLn: "NZ Groundwater Depth (Comb, Ln)",
    InputVariable.NZEnvDSDistanceRivers: "NZEnvDS Distance To Rivers",
    InputVariable.NZEnvDSDistanceRiversVertical: "NZEnvDS Vertical Distance To Rivers",
    InputVariable.NZEnvDSPrecipAnn: "NZEnvDS Annual Precipitation",
    InputVariable.NZEnvDSSlopeDeg: "NZEnvDS Slope (Degrees)",
    InputVariable.NZEnvDSSoilAcidP: "NZEnvDS Soil phosphorus",
    InputVariable.NZEnvDSSoilAge: "NZEnvDS Soil Age",
    InputVariable.NZEnvDSSoilDrainage: "NZEnvDS Soil Drainage",
    InputVariable.NZEnvDSSoilInduration: "NZEnvDS Soil Induration",
    InputVariable.NZEnvDSSoilParticleSize: "NZEnvDS Soil Particle Size",
    InputVariable.NZEnvDSTopoGeomorphons: "NZEnvDS Topo Geomorphons",
    InputVariable.NZEnvDSTopoNormalisedHeight: "NZEnvDS Topo Normalised Height",
    InputVariable.NZEnvDSTopoPosition: "NZEnvDS Topo Position",
    InputVariable.NZEnvDSTopoRoughness: "NZEnvDS Topo Roughness",
    InputVariable.NZEnvDSTopoRuggedness: "NZEnvDS Topo Ruggedness",
    InputVariable.NZEnvDSTopoValleyDepth: "NZEnvDS Topo Valley Depth",
    InputVariable.NZEnvDSTopoWetness: "NZEnvDS Topo Wetness",
}

INPUT_VAR_TO_PAPER_NICE_NAME_MAPPING = {
    InputVariable.NZEnvDSTopoRoughness: "Topographic Roughness (index)",
    InputVariable.NZGeologyAgeLnMid: "Geological Age (Ma)",
    InputVariable.NZCombinedGroundwaterDepth: "Groundwater Depth (m)",
    InputVariable.NZEnvDSTopoNormalisedHeight: "Topographic Normalised Height (index)",
    InputVariable.NZEnvDSDistanceRiversVertical: "Distance to Rivers (Vertical) (m)",
}

REVERSE_NICE_NAME_TO_INPUT_VARIABLE_MAPPING = {
    nice_name: var for var, nice_name in INPUT_VARIABLE_TO_NICE_NAME_MAPPING.items()
}

CATEGORIAL_VARIABLES = [
    InputVariable.Geomorphon,
    InputVariable.NZEnvDSTopoGeomorphons,
    InputVariable.NZGeologicalUnit,
    InputVariable.NZLithologyCategory,
    InputVariable.NZQuaternaryRegion,
]

ORDINAL_VARIABLES = [
    InputVariable.NZGeologyCategory,
    InputVariable.NZEnvDSSoilAcidP,
    InputVariable.NZEnvDSSoilAge,
    InputVariable.NZEnvDSSoilDrainage,
    InputVariable.NZEnvDSSoilInduration,
    InputVariable.NZEnvDSSoilParticleSize,
]


GLOBAL_INPUT_VARS = np.array(
    [
        InputVariable.Roughness,
        InputVariable.TopographicSlope,
        InputVariable.CompoundTopgraphicIndex,
        InputVariable.Geomorphon,
        InputVariable.ProfileCurvature,
        InputVariable.TangentialCurvature,
        InputVariable.TerrainRuggednessIndex,
        InputVariable.TopographicPositionIndex,
        InputVariable.VectorRuggednessMeasure,
        InputVariable.LandformEntropy,
        InputVariable.LandformShannonIndex,
        InputVariable.LandformUniformity,
        InputVariable.AbsoluteDepthToBedrock,
        InputVariable.DepthToGroundwater,
        InputVariable.Elevation,
    ]
)

NZ_INPUT_VARS = np.array(
    [
        InputVariable.NZGeologyCategory,
        InputVariable.NZDistanceToCoast,
        InputVariable.NZGeologyAgeLnMid,
        InputVariable.NZCombinedGroundwaterDepth,
        InputVariable.NZCombinedGroundwaterDepthLn,
        InputVariable.NZEnvDSDistanceRivers,
        InputVariable.NZEnvDSDistanceRiversVertical,
        InputVariable.NZEnvDSPrecipAnn,
        InputVariable.NZEnvDSSlopeDeg,
        InputVariable.NZEnvDSSoilAcidP,
        InputVariable.NZEnvDSSoilAge,
        InputVariable.NZEnvDSSoilDrainage,
        InputVariable.NZEnvDSSoilInduration,
        InputVariable.NZEnvDSSoilParticleSize,
        InputVariable.NZEnvDSTopoGeomorphons,
        InputVariable.NZEnvDSTopoNormalisedHeight,
        InputVariable.NZEnvDSTopoPosition,
        InputVariable.NZEnvDSTopoRoughness,
        InputVariable.NZEnvDSTopoRuggedness,
        InputVariable.NZEnvDSTopoValleyDepth,
        InputVariable.NZEnvDSTopoWetness,
    ]
)

DERIVED_VARIABLES_DEPENDENCIES = {
    InputVariable.NZGeologyAgeMid: [
        InputVariable.NZGeologyAgeMin,
        InputVariable.NZGeologyAgeMax,
    ],
    InputVariable.NZGeologyAgeLnMid: [
        InputVariable.NZGeologyAgeMin,
        InputVariable.NZGeologyAgeMax,
    ],
    InputVariable.NZCombinedGroundwaterDepth: [
        InputVariable.NZNLMGroundwaterDepth,
        InputVariable.NZNWTGroundwaterDepth,
        InputVariable.DepthToGroundwater,
    ],
    InputVariable.NZCombinedGroundwaterDepthLn: [
        InputVariable.NZNLMGroundwaterDepth,
        InputVariable.NZNWTGroundwaterDepth,
        InputVariable.DepthToGroundwater,
    ],
}

# Input variables locations
INPUT_VAR_TO_FFP_MAP = {
    InputVariable.NZEnvDSSlopeDeg: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/slope_deg.tif",
    InputVariable.NZEnvDSTopoNormalisedHeight: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/topo_normalisedHeight.tif",
    InputVariable.NZEnvDSTopoPosition: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/topo_position.tif",
    InputVariable.NZEnvDSTopoRoughness: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/topo_roughness.tif",
    InputVariable.NZEnvDSTopoWetness: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/topo_wetness.tif",
    InputVariable.NZEnvDSDistanceRivers: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/distance_rivers.tif",
    InputVariable.NZEnvDSDistanceRiversVertical: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/distance_riversVertical.tif",
    InputVariable.NZEnvDSPrecipAnn: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/precip_ann.tif",
    InputVariable.NZEnvDSSoilDrainage: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/soil_drainage.tif",
    InputVariable.NZEnvDSSoilParticleSize: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/soil_particleSize.tif",
    InputVariable.NZEnvDSTopoValleyDepth: BASE_DATA_DIR
    / "input_data/nzenvds_v1p1_nztm/final_layers_nztm/topo_valleyDepth.tif",
    InputVariable.NZNLMGroundwaterDepth: BASE_DATA_DIR
    / "input_data/nz_nlm/NLM_gwd.tif",
    InputVariable.NZNWTGroundwaterDepth: BASE_DATA_DIR
    / "input_data/nz_nwt/nwt_wtd_NZ_20220825.tif",
}


WGS84_EPSG_STR = "EPSG:4326"
WGS84_EPSG = 4326
NZTM2000_EPSG_STR = "EPSG:2193"
NZTM2000_EPSG = 2193

NZTM_TO_WGS84_TRANSFORMER = Transformer.from_crs(
    NZTM2000_EPSG, WGS84_EPSG, always_xy=True
)
WGS84_TO_NZTM_TRANSFORMER = Transformer.from_crs(
    WGS84_EPSG, NZTM2000_EPSG, always_xy=True
)


LN_NORM_VARS = [
    InputVariable.AbsoluteDepthToBedrock,
    InputVariable.TopographicSlope,
    InputVariable.Roughness,
    InputVariable.VectorRuggednessMeasure,
    InputVariable.TerrainRuggednessIndex,
    InputVariable.NZEnvDSSlopeDeg,
    InputVariable.NZEnvDSTopoRoughness,
    InputVariable.NZEnvDSTopoRuggedness,
    InputVariable.NZEnvDSDistanceRivers,
    InputVariable.NZEnvDSPrecipAnn,
    InputVariable.NZEnvDSTopoValleyDepth,
    InputVariable.NZEnvDSTopoWetness,
]
NORM_VARS = [
    InputVariable.TopographicPositionIndex,
    InputVariable.ProfileCurvature,
    InputVariable.TangentialCurvature,
]

MIN_MAX_CLIP_SCALE_PARAMS = {
    InputVariable.Elevation: (0, 1500, True),
    InputVariable.NZCombinedGroundwaterDepthLn: (
        -2,
        3.2188758249,
        True,
    ),  # (0.002478752177, 25) (m)
    # InputVariable.NZCombinedGroundwaterDepth: (0, 25, True),
    InputVariable.NZCombinedGroundwaterDepth: (0, 10, True),
    InputVariable.CompoundTopgraphicIndex: (-4.0, 10.0, False),
    InputVariable.LandformEntropy: (0, 3.0, False),
    InputVariable.LandformUniformity: (0, 1.0, False),
    InputVariable.LandformShannonIndex: (0, 3.0, False),
    InputVariable.NZDistanceToCoast: (0, 100_000, True),
    InputVariable.NZEnvDSTopoNormalisedHeight: (0, 1, False),
    InputVariable.NZEnvDSTopoPosition: (-20, 20, True),
    InputVariable.NZGeologyAgeLnMid: (-6, 6, False),
    InputVariable.NZDistanceToRiver_ST1: (0, 1_000, True),
    InputVariable.NZDistanceToRiver_ST2: (0, 1_000, True),
    InputVariable.NZDistanceToRiver_ST3: (0, 5_000, True),
    InputVariable.NZDistanceToRiver_ST4: (0, 5_000, True),
    InputVariable.NZDistanceToRiver_ST5: (0, 10_000, True),
    InputVariable.NZDistanceToRiver_ST6: (0, 10_000, True),
    InputVariable.NZDistanceToRiver_ST7: (0, 15_000, True),
    InputVariable.NZDistanceToRiver_ST8: (0, 30_000, True),
    InputVariable.NZDistanceToRiver_ST1_Greater: (0, 30_000, True),
    InputVariable.NZDistanceToRiver_ST2_Greater: (0, 30_000, True),
    InputVariable.NZDistanceToRiver_ST3_Greater: (0, 30_000, True),
    InputVariable.NZDistanceToRiver_ST4_Greater: (0, 30_000, True),
    InputVariable.NZDistanceToRiver_ST5_Greater: (0, 30_000, True),
    InputVariable.NZDistanceToRiver_ST6_Greater: (0, 30_000, True),
    InputVariable.NZDistanceToRiver_ST7_Greater: (0, 30_000, True),
    InputVariable.NZEnvDSDistanceRiversVertical: (0, 300, True),
}

INPUT_VARIABLE_CMAP_LIMITS = {
    InputVariable.NZEnvDSSlopeDeg: (0, 40),
    InputVariable.NZNWTGroundwaterDepth: (0, 400),
    InputVariable.NZNLMGroundwaterDepth: (0, 10),
    InputVariable.NZEnvDSDistanceRivers: (0, 10_000),
    InputVariable.NZEnvDSDistanceRiversVertical: (0, 1500),
    InputVariable.NZEnvDSPrecipAnn: (0, 5000),
    InputVariable.NZEnvDSSoilAcidP: (0, 5),
    InputVariable.NZEnvDSSoilAge: (0, 2),
    InputVariable.NZEnvDSSoilDrainage: (0, 5),
    InputVariable.NZEnvDSSoilInduration: (0, 5),
    InputVariable.NZEnvDSTopoGeomorphons: (0, 10),
    InputVariable.NZEnvDSSoilParticleSize: (0, 5),
    InputVariable.NZEnvDSTopoNormalisedHeight: (0, 1),
    # InputVariable.NZEnvDSTopoPosition: (-25, 25),
    InputVariable.NZEnvDSTopoPosition: (-15, 15),
    InputVariable.NZEnvDSTopoRoughness: (0, 200),
    InputVariable.NZEnvDSTopoRuggedness: (0, 50),
    InputVariable.NZEnvDSTopoValleyDepth: (0, 250),
    InputVariable.NZEnvDSTopoWetness: (2, 12),
    InputVariable.DepthToGroundwater: (-200, 0),
    InputVariable.NZDistanceToCoast: (0, 80_000),
    InputVariable.NZDistanceToRiver_ST1: (0, 1_000),
    InputVariable.NZDistanceToRiver_ST2: (0, 1_000),
    InputVariable.NZDistanceToRiver_ST3: (0, 5_000),
    InputVariable.NZDistanceToRiver_ST4: (0, 15_000),
    InputVariable.NZDistanceToRiver_ST5: (0, 30_000),
    InputVariable.NZDistanceToRiver_ST6: (0, 60_000),
    InputVariable.NZDistanceToRiver_ST7: (0, 80_000),
    InputVariable.NZDistanceToRiver_ST8: (0, 100_000),
    InputVariable.NZGeologyCategory: (0, 15),
    InputVariable.NZGeologyAgeMin: (0, 500),
    InputVariable.NZGeologyAgeMax: (0, 500),
    InputVariable.CompoundTopgraphicIndex: (-5.0, 5.0),
    InputVariable.Elevation: (0, 1500),
    InputVariable.NZGeologyAgeMid: (0, 500),
    InputVariable.NZGeologyAgeLnMid: (-5, 6),
    InputVariable.NZCombinedGroundwaterDepth: (0, 25),
    InputVariable.NZCombinedGroundwaterDepthLn: (-2, 6),
}

QUATERNARY_REGION_TO_ID_MAPPING = pd.Series(
    {
        "no_region": 0,
        "canterbury": 1,
        "invercargill": 2,
        "napier": 3,
        "palmerston_north": 4,
        "taranaki": 5,
        "taupo": 6,
        "wellington": 7,
        "wellington_hutt": 8,
    }
)
QUATERNARY_ID_TO_REGION_MAPPING = pd.Series(
    {v: k for k, v in QUATERNARY_REGION_TO_ID_MAPPING.items()}
)

VS30_WEIGHTING_BINS = np.asarray([0, 180, 360, 760, 10_000])
VS30_WEIGHTING_BIN_NAMES = [
    f"{VS30_WEIGHTING_BINS[i]}-{VS30_WEIGHTING_BINS[i + 1]}"
    for i in range(len(VS30_WEIGHTING_BINS) - 1)
]
V30_BIN_COLORS = ["blue", "green", "orange", "red"]

DENSE_VS30_BINS = np.asarray([0, 180, 260, 360, 540, 760, 1000, 1600, 3000, 10_000])
DENSE_VS30_BIN_NAMES = [
    f"{DENSE_VS30_BINS[i]}-{DENSE_VS30_BINS[i + 1]}"
    for i in range(len(DENSE_VS30_BINS) - 1)
]

GEYIN_VS30_BINS = np.asarray([0, 180, 259,360, 537, 760, 1150, 2000])
GEYIN_VS30_BIN_NAMES = [
    f"{GEYIN_VS30_BINS[i]}-{GEYIN_VS30_BINS[i + 1]}"
    for i in range(len(GEYIN_VS30_BINS) - 1)
]

QUALITY_SCORE_COLORS = {
    # "Q1": "tab:green",
    "Q1": "blue",
    # "Q2": "tab:blue",
    "Q2": "purple",
    "Q3": "black",
}
QUALITY_SCORE_MARKER_SIZE = {
    "Q1": 40,
    "Q2": 30,
    "Q3": 15,
}
QUALITY_SCORE_MARKERS = {
    "Q1": "o",
    "Q2": "D",
    "Q3": "X",
}

NZ_FULL_BOUNDING_BOX = [166.3, 178.65, -47.1, -34.25]
NZ_BOUNDING_BOX = [166.3, 178.65, -47.05, -35.5]
CANTERBURY_BOUNDING_BOX = [171.54, 173.15, -43.96, -43.2025]
CANTERBURY_NARROW_BOUNDING_BOX = [171.96, 173.15, -43.98, -43.04]
CROMWELL_REGION = [168.448, 169.625, -45.388, -44.455]
WELLINGTON_BOUNDING_BOX = [174.67, 175.1, -41.42, -41.08]
WELLINGTON_LARGE_BOUNDING_BOX = [174.567, 175.744, -41.676, -40.742]
WELLINGTON_MODERATE_BOUNDING_BOX = [174.554, 175.345, -41.515, -40.842]
NORTH_ISLAND_BOUNDING_BOX = [172.55, 178.625, -41.65, -34.4]
SOUTH_ISLAND_BOUNDING_BOX = [166.3, 174.4, -47.3, -40.3]
AUCKLAND_HAMILTON_REGION_BOUNDING_BOX = [174.348, 175.726, -38.056, -36.612]
TAURANGE_ROTORUA_REGION_BOUNDING_BOX = [175.805, 176.982, -38.368, -37.434]

REGION_MAPPING = {
    "nz_full": NZ_FULL_BOUNDING_BOX,
    "nz": NZ_BOUNDING_BOX,
    "ni": NORTH_ISLAND_BOUNDING_BOX,
    "si": SOUTH_ISLAND_BOUNDING_BOX,
    "canterbury": CANTERBURY_BOUNDING_BOX,
    "wellington": WELLINGTON_BOUNDING_BOX,
    "canterbury_narrow": CANTERBURY_NARROW_BOUNDING_BOX,
    "cromwell_region": CROMWELL_REGION,
    "auckland_hamilton_region": AUCKLAND_HAMILTON_REGION_BOUNDING_BOX,
    "taurange_rotorua_region": TAURANGE_ROTORUA_REGION_BOUNDING_BOX,
    "wellington_large": WELLINGTON_LARGE_BOUNDING_BOX,
    "wellington_moderate": WELLINGTON_MODERATE_BOUNDING_BOX,
}

CITY_COORDS = {
    "Christchurch": (172.63669300877544, -43.531923487539935),
    "Wellington": (174.77791888634852, -41.28387793785542),
    "Auckland": (174.76555503318232, -36.850282550438685),
}

TOWN_COORDS = {
    "Tauranga": (176.165822426251, -37.68682531361862),
    "Hamilton": (175.25243436298052, -37.782667727885766),
    "Rangiora": (172.59675756330435, -43.30336314429278),
    "Kaiapoi": (172.66230663063166, -43.3787869396717),
    "Amberley": (172.7304373827998, -43.15761096839294),
    "Rolleston": (172.38365123459576, -43.59673064857969),
    "Cromwell": (169.19552400667155, -45.04608046452913),
    "Queenstown": (168.6620834594349, -45.030224565662294),
    "Arrowtown": (168.82795518682846, -44.94257606312187),
    "Wanaka": (169.14217408524496, -44.694323126053014),
    "Lower Hutt": (174.89950374781867, -41.21267461217909),
    "Upper Hutt": (175.0657471968382, -41.12496597728303),
    "Johnsonville": (174.8079760665923, -41.22056779631806),
    "Porirua": (174.84752179010113, -41.138059203524115),
    "Rotorua": (176.23749341152626, -38.144628098911454),
    # "Papamoa": (176.28481293859434, -37.69727293258872),
    # "Mount Maunganui": (176.20818224666397, -37.6645311303861),
    "Raglan": (174.87117463105753, -37.800733659242745),
    "Huntly": (175.15937169874206, -37.55706591572003),
    # "Manakau": (174.8740306499414, -36.99174153203661),
    # "Glenfield": (174.7211172686564, -36.78178846008382)
    "Taupo": (176.07099197440127, -38.684391122710124),
    "Palmerston North": (175.60998552418505, -40.354237801149175),
    "Napier": (176.91908274377244, -39.489327066422106),
    "Nelson": (173.246953726745, -41.29896111466095),
    "Blenheim": (173.96041433170313, -41.51421741822313),
    "Greymouth": (171.19716741172354, -42.461954708045816),
    "Haast": (169.0434759605885, -43.87930449185373)
}

REGION_COORDS = {
    "Taranaki": (174.07362998031326, -39.29826300458175),
    "Tasman Range": (172.47053573567698, -41.284313712045034),
    "Raukumara Range": (177.81863814835225, -37.917050464248874),
    "East Cape": (178.0272757557843, -37.88741822489124),
}

NZTM_BOUNDING_BOX = [1079100.000, 2100800.000, 4736600.000, 6229700.000]

# Geyin & Maurer model MAE values for Vs30 bins
# Table 2 of Geyin & Maurer (2023)
GEYIN_MAURER_MODEL_MAE = {
    90: 55,
    220: 55,
    310: 77,
    448.5: 98,
    648.5: 148,
    955: 296,
    1575: 531,
}


# Default figure settings
FIG_SIZE = (16, 10)
if (env_figsize := os.environ.get("fig_size")) is not None:
    FIG_SIZE = [float(x) for x in env_figsize.split(",")]

FIG_FORMAT = "png"
if (env_fig_format := os.environ.get("fig_format")) is not None:
    FIG_FORMAT = env_fig_format

FIG_DPI = 300
if (env_fig_dpi := os.environ.get("fig_dpi")) is not None:
    FIG_DPI = int(env_fig_dpi)

FIG_FONT_SIZE = None
if (env_fig_font_size := os.environ.get("fig_font_size")) is not None:
    FIG_FONT_SIZE = int(env_fig_font_size)

FIG_LINEWIDTH = 2.0
if (env_fig_linewidth := os.environ.get("fig_linewidth")) is not None:
    FIG_LINEWIDTH = float(env_fig_linewidth)

FIG_GROUP_LINEWIDTH = 1.5
if (env_fig_group_linewidth := os.environ.get("fig_group_linewidth")) is not None:
    FIG_GROUP_LINEWIDTH = float(env_fig_group_linewidth)

GMT_FIG_FONT_LABEL = "14p,Helvetica,black"
if (env_gmt_fig_font_label := os.environ.get("gmt_fig_font_label")) is not None:
    GMT_FIG_FONT_LABEL = env_gmt_fig_font_label

GMT_FIG_MINOR_FONT_LABEL = "10p,Helvetica,black"
if (
    env_gmt_fig_minor_font_label := os.environ.get("gmt_fig_minor_font_label")
) is not None:
    GMT_FIG_MINOR_FONT_LABEL = env_gmt_fig_minor_font_label

GMT_FIG_BOLD_FONT_LABEL = "14p,Helvetica-Bold,black"
if (
    env_gmt_fig_bold_font_label := os.environ.get("gmt_fig_bold_font_label")
) is not None:
    GMT_FIG_BOLD_FONT_LABEL = env_gmt_fig_bold_font_label

GMT_FIG_FONT_ANNOT_PRIMARY = "11p,Helvetica,black"
if (
    env_gmt_fig_font_annot_primary := os.environ.get("gmt_fig_font_annot_primary")
) is not None:
    GMT_FIG_FONT_ANNOT_PRIMARY = env_gmt_fig_font_annot_primary

GMT_SHOW_CB_LABEL = True
if (env_gmt_show_cb_label := os.environ.get("gmt_show_cb_label")) is not None:
    GMT_SHOW_CB_LABEL = env_gmt_show_cb_label.lower() in ("1", "true", "yes")
