"""Object detection modules - classical and modern approaches."""

from .base import BaseDetector, DetectionResult
from .haar_cascade import HaarCascadeDetector
from .yolo import YOLODetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "HaarCascadeDetector",
    "YOLODetector",
]
