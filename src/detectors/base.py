"""
Base abstract class for all object detectors.

This module defines the interface that all detectors (Haar Cascade, YOLO, etc.)
must implement, ensuring consistency and enabling easy comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path


@dataclass
class DetectionResult:
    """
    Result from object detection.

    Attributes:
        boxes: List of bounding boxes (x, y, width, height)
        confidences: Confidence scores for each detection (0-1)
        class_ids: Class IDs for each detection
        class_names: Human-readable class names
        inference_time: Time taken for detection in seconds
        image_shape: Original image shape (height, width, channels)
    """
    boxes: List[Tuple[int, int, int, int]]
    confidences: List[float]
    class_ids: List[int]
    class_names: List[str]
    inference_time: float
    image_shape: Tuple[int, int, int]

    @property
    def num_detections(self) -> int:
        """Number of objects detected."""
        return len(self.boxes)

    def filter_by_confidence(self, threshold: float = 0.5) -> 'DetectionResult':
        """
        Filter detections by confidence threshold.

        Args:
            threshold: Minimum confidence score (0-1)

        Returns:
            New DetectionResult with filtered detections
        """
        indices = [i for i, conf in enumerate(self.confidences) if conf >= threshold]
        return DetectionResult(
            boxes=[self.boxes[i] for i in indices],
            confidences=[self.confidences[i] for i in indices],
            class_ids=[self.class_ids[i] for i in indices],
            class_names=[self.class_names[i] for i in indices],
            inference_time=self.inference_time,
            image_shape=self.image_shape,
        )


class BaseDetector(ABC):
    """
    Abstract base class for all object detectors.

    This class defines the interface that all concrete detector implementations
    must follow. It ensures consistency and enables polymorphic usage.
    """

    def __init__(self, model_path: Optional[Path] = None, **kwargs):
        """
        Initialize the detector.

        Args:
            model_path: Path to the model file
            **kwargs: Additional detector-specific parameters
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the detection model.

        This method must be implemented by all subclasses to load their
        respective models (Haar Cascade XML, YOLO weights, etc.).

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """
        Detect objects in an image.

        Args:
            image: Input image as numpy array (BGR format)
            **kwargs: Detector-specific parameters

        Returns:
            DetectionResult containing bounding boxes, confidences, etc.

        Raises:
            ValueError: If image is invalid
            RuntimeError: If model is not loaded
        """
        pass

    def detect_batch(self, images: List[np.ndarray], **kwargs) -> List[DetectionResult]:
        """
        Detect objects in multiple images.

        Default implementation processes images sequentially.
        Subclasses can override for batch optimization.

        Args:
            images: List of input images
            **kwargs: Detector-specific parameters

        Returns:
            List of DetectionResult objects
        """
        return [self.detect(img, **kwargs) for img in images]

    @abstractmethod
    def get_detector_name(self) -> str:
        """
        Get the human-readable name of this detector.

        Returns:
            Detector name (e.g., "Haar Cascade", "YOLOv8")
        """
        pass

    @abstractmethod
    def get_detector_info(self) -> dict:
        """
        Get detailed information about this detector.

        Returns:
            Dictionary with detector metadata (year, paper, performance, etc.)
        """
        pass

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"{self.get_detector_name()}(loaded={self.is_loaded})"
