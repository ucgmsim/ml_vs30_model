from enum import StrEnum
from pathlib import Path
import os

import numpy as np


class DataSource(StrEnum):
    GeoMorpho90 = "geomorpho90"
    TIFLoader = "tif_loader"
    GlobalGWT = "global_gwt" # Global Groundwater Table
    ShapeLoader = "shape_loader"
    NZDistanceToCoast = "nz_distance_to_coast"

class InputVariable(StrEnum):
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
    NZGeologyCategory = "nz_geology_category"
    NZDistanceToCoast = "nz_distance_to_coast"


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
    InputVariable.NZGeologyCategory: DataSource.ShapeLoader,
    InputVariable.NZDistanceToCoast: DataSource.NZDistanceToCoast,
}

INPUT_VARIABLE_TO_NICE_NAME_MAPPING = {
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
}

CATEGORIAL_VARIABLES = [
    InputVariable.Geomorphon,
    InputVariable.NZGeologyCategory,
]

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
    InputVariable.Roughness,
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
}


VS30_WEIGHTING_BINS = np.asarray([0, 180, 360, 760, 1600])
VS30_WEIGHTING_BIN_NAMES = [
    f"{VS30_WEIGHTING_BINS[i]}-{VS30_WEIGHTING_BINS[i + 1]}"
    for i in range(len(VS30_WEIGHTING_BINS) - 1)
]
V30_BIN_COLORS = ["blue", "green", "orange", "red"]

DENSE_VS30_BINS = np.asarray([0, 180, 260, 360, 540, 760, 1000, 1600, 3000])
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