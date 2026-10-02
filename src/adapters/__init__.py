"""
Adapters Package
Contains decoupled model wrappers for third-party ML architectures.
"""
from .base import BaseModelAdapter
from .object_detector import GenericObjectDetectorAdapter
from .road_damage import RoadDamageDetectorAdapter
from .road_segmenter import RoadSegmenterAdapter
from .traffic_sign import TrafficSignDetectorAdapter

__all__ = [
    "BaseModelAdapter",
    "GenericObjectDetectorAdapter",
    "RoadDamageDetectorAdapter",
    "RoadSegmenterAdapter",
    "TrafficSignDetectorAdapter",
]
