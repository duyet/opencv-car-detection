"""
OpenCV Car Detection - Computer Vision Evolution
Compare classical Haar Cascades with modern deep learning for vehicle detection.

This package demonstrates the evolution of computer vision object detection
from classical methods (Haar Cascades, 2001) to modern deep learning approaches (YOLO, 2023).
"""

__version__ = "2.0.0"
__author__ = "Van-Duyet Le"
__email__ = "lvduit08@gmail.com"

from .detectors.base import BaseDetector
from .detectors.haar_cascade import HaarCascadeDetector
from .detectors.yolo import YOLODetector

__all__ = [
    "BaseDetector",
    "HaarCascadeDetector",
    "YOLODetector",
]
