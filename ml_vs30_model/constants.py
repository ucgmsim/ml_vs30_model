from enum import StrEnum
from pathlib import Path
import os

import numpy as np


class DataSource(StrEnum):
    GeoMorpho90 = "geomorpho90"
    TIFLoader = "tif_loader"
    GlobalGWT = "global_gwt" # Global Groundwater Table
    ShapeLoader = "shape_loader"

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
}


VS30_WEIGHTING_BINS = np.asarray([0, 180, 360, 760, 1600])
VS30_WEIGHTING_BIN_NAMES = [
    f"{VS30_WEIGHTING_BINS[i]}_{VS30_WEIGHTING_BINS[i + 1]}"
    for i in range(len(VS30_WEIGHTING_BINS) - 1)
]


NZ_BOUNDING_BOX = [166.3, 179, -47.4, -36.0]