from enum import StrEnum
from pathlib import Path
import os

import numpy as np


class DataSource(StrEnum):
    GeoMorpho90 = "geomorpho90"
    SRTMGL1 = "srtmgl1"
    TIFLoader = "tif_loader"
    NZTMTIFLoader = "nztm_tif_loader"
    GlobalGWT = "global_gwt" # Global Groundwater Table
    ShapeLoader = "shape_loader"
    NZDistanceToCoast = "nz_distance_to_coast"

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
    Elevation = "elevation"                     # SRTMGL1 elevation data (30m)
    # NZ 
    NZGeologyCategory = "nz_geology_category"   # Foster et al.
    NZDistanceToCoast = "nz_distance_to_coast"  # Manually computed
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

PRETTY_INPUT_VARIABLE_NAMES = {
    InputVariable.NZGeologyCategory: "NZ Geology Category",
    InputVariable.CompoundTopgraphicIndex: "Compound Topographic Index",    
    InputVariable.DepthToGroundwater: "Groundwater Depth",
    InputVariable.NZEnvDSSlopeDeg: "NZEnvDS Slope (Degrees)",
    InputVariable.NZEnvDSTopoNormalisedHeight: "NZEnvDS Norm. Topo Height",
    InputVariable.NZEnvDSTopoPosition: "NZEnvDS Topo Position",
    InputVariable.NZEnvDSTopoWetness: "NZEnvDS Wetness",
    InputVariable.NZEnvDSTopoRoughness: "NZEnvDS Roughness",
}

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

CATEGORIAL_VARIABLES = [
    InputVariable.Geomorphon,
    InputVariable.NZGeologyCategory,
    InputVariable.NZEnvDSTopoGeomorphons,
    InputVariable.NZEnvDSSoilParticleSize,
    InputVariable.NZEnvDSSoilInduration,
    InputVariable.NZEnvDSSoilDrainage,
    InputVariable.NZEnvDSSoilAge,
    InputVariable.NZEnvDSSoilAcidP,
]

GLOBAL_INPUT_VARS = np.array([
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
])

NZ_INPUT_VARS = np.array([
    InputVariable.NZGeologyCategory,    
    InputVariable.NZDistanceToCoast,
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
])

WGS84_EPSG_STR = "EPSG:4326"    
WGS84_EPSG = 4326    
NZTM2000_EPSG_STR = "EPSG:2193" 
NZTM2000_EPSG = 2193

if (BASE_DATA_DIR := os.getenv("VS30_MODEL_BASE_DATA_DIR")) is None:
    raise EnvironmentError("Environment variable VS30_MODEL_BASE_DATA_DIR is not set.")
BASE_DATA_DIR  = Path(BASE_DATA_DIR)


LN_NORM_VARS = [
    InputVariable.AbsoluteDepthToBedrock,
    InputVariable.TopographicSlope,
    InputVariable.NZEnvDSSlopeDeg,
    InputVariable.Roughness,
    InputVariable.NZEnvDSTopoRoughness,
    InputVariable.TerrainRuggednessIndex,
    InputVariable.VectorRuggednessMeasure,
]
NORM_VARS = [
    InputVariable.TopographicPositionIndex,
    InputVariable.ProfileCurvature,
    InputVariable.TangentialCurvature,

]

MIN_MAX_SCALE_PARAMS = {
    InputVariable.DepthToGroundwater: (-5, 5),
    InputVariable.CompoundTopgraphicIndex: (-4.0, 10.0),
    InputVariable.LandformEntropy: (0, 3.0),
    InputVariable.LandformUniformity: (0, 1.0),
    InputVariable.LandformShannonIndex: (0, 3.0),
    InputVariable.NZDistanceToCoast: (0, 100_000),  
    InputVariable.NZEnvDSTopoNormalisedHeight: (0, 1),
    InputVariable.NZEnvDSTopoPosition: (-30, 30),
    InputVariable.NZEnvDSTopoWetness: (2, 15),
}


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

NZ_BOUNDING_BOX = [166.3, 179, -47.4, -36.0]

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

FIG_LINEWIDTH = None
if (env_fig_linewidth := os.environ.get("fig_linewidth")) is not None:
    FIG_LINEWIDTH = float(env_fig_linewidth)

FIG_GROUP_LINEWIDTH = None
if (env_fig_group_linewidth := os.environ.get("fig_group_linewidth")) is not None:
    FIG_GROUP_LINEWIDTH = float(env_fig_group_linewidth)

GMT_FIG_FONT_LABEL = "14p,Helvetica,black"
if (env_gmt_fig_font_label := os.environ.get("gmt_fig_font_label")) is not None:
    GMT_FIG_FONT_LABEL = env_gmt_fig_font_label

GMT_FIG_BOLD_FONT_LABEL = "14p,Helvetica-Bold,black"
if (env_gmt_fig_bold_font_label := os.environ.get("gmt_fig_bold_font_label")) is not None:
    GMT_FIG_BOLD_FONT_LABEL = env_gmt_fig_bold_font_label

GMT_FIG_FONT_ANNOT_PRIMARY = "11p,Helvetica,black"
if (env_gmt_fig_font_annot_primary := os.environ.get("gmt_fig_font_annot_primary")) is not None:
    GMT_FIG_FONT_ANNOT_PRIMARY = env_gmt_fig_font_annot_primary

GMT_SHOW_CB_LABEL = True
if (env_gmt_show_cb_label := os.environ.get("gmt_show_cb_label")) is not None:
    GMT_SHOW_CB_LABEL = env_gmt_show_cb_label.lower() in ("1", "true", "yes")