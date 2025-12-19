from enum import StrEnum
from pathlib import Path
import os

class DataSource(StrEnum):
    GeoMorpho90 = "geomorpho90"
    TIFLoader = "tif_loader"
    GlobalGWT = "global_gwt" # Global Groundwater Table

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
}

WGS84_EPSG_STR = "EPSG:4326"    
WGS84_EPSG = 4326    

if (BASE_DATA_DIR := os.getenv("VS30_MODEL_BASE_DATA_DIR")) is None:
    raise EnvironmentError("Environment variable VS30_MODEL_BASE_DATA_DIR is not set.")
BASE_DATA_DIR  = Path(BASE_DATA_DIR)