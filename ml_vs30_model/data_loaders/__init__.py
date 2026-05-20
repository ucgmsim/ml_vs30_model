"""Module of classes for the loading of different types of geospatial data."""
from .base_loader import BaseLoader
from .opentopography import GeoMorpho90, SRTMGL1
from .tif_loader import TIFLoader, NZTMTIFLoader
from .global_gwt import GlobalGWT
from .shape_loader import ShapeLoader
from .distance_to_shape import NZDistanceToCoast, NZDistanceToRiver

__all__ = [
    "BaseLoader",
    "GeoMorpho90",
    "SRTMGL1",
    "TIFLoader",
    "NZTMTIFLoader",
    "GlobalGWT",
    "ShapeLoader",
    "NZDistanceToCoast",
    "NZDistanceToRiver",
]
