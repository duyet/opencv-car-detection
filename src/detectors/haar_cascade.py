"""
Haar Cascade Detector - Classical Computer Vision (2001).

This module implements vehicle detection using Viola-Jones Haar Cascade classifier,
a classical computer vision approach that was state-of-the-art in the early 2000s.

Reference:
    Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade
    of simple features. CVPR 2001.
"""

import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from loguru import logger

from .base import BaseDetector, DetectionResult


class HaarCascadeDetector(BaseDetector):
    """
    Haar Cascade-based vehicle detector.

    This detector uses OpenCV's pre-trained Haar Cascade classifier for car detection.
    While outdated compared to deep learning methods, it's lightweight, fast on CPU,
    and requires no GPU.

    Advantages:
        - Very fast on CPU
        - Low memory footprint
        - No training required (uses pre-trained cascades)
        - Simple to understand and deploy

    Limitations:
        - Lower accuracy than deep learning methods
        - Sensitive to object scale and orientation
        - High false positive rate
        - Limited to frontal/rear views
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
        min_size: Tuple[int, int] = (30, 30),
        **kwargs
    ):
        """
        Initialize Haar Cascade detector.

        Args:
            model_path: Path to Haar Cascade XML file
            scale_factor: Image pyramid scale factor (1.1 = 10% reduction each level)
            min_neighbors: Minimum neighbors for detection (higher = fewer false positives)
            min_size: Minimum object size (width, height)
            **kwargs: Additional parameters
        """
        super().__init__(model_path, **kwargs)

        # Default to bundled cascade if no path provided
        if model_path is None:
            self.model_path = Path(__file__).parent.parent.parent / "data" / "models" / "cars.xml"
        else:
            self.model_path = Path(model_path)

        # Detection parameters
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        logger.info(f"Initialized {self.get_detector_name()} with cascade: {self.model_path}")

    def load_model(self) -> None:
        """
        Load the Haar Cascade classifier.

        Raises:
            FileNotFoundError: If cascade XML file doesn't exist
            ValueError: If cascade file is invalid
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Haar Cascade file not found: {self.model_path}\n"
                f"Please ensure the cars.xml file is in data/models/"
            )

        try:
            self.model = cv2.CascadeClassifier(str(self.model_path))

            # Verify the cascade loaded correctly
            if self.model.empty():
                raise ValueError(
                    f"Failed to load Haar Cascade from {self.model_path}\n"
                    f"The XML file may be corrupted or invalid."
                )

            self.is_loaded = True
            logger.success(f"Loaded Haar Cascade from {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading Haar Cascade: {e}")
            raise

    def detect(
        self,
        image: np.ndarray,
        scale_factor: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        min_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> DetectionResult:
        """
        Detect vehicles in an image using Haar Cascade.

        Args:
            image: Input image (BGR format)
            scale_factor: Override default scale factor
            min_neighbors: Override default min_neighbors
            min_size: Override default min_size
            **kwargs: Additional parameters

        Returns:
            DetectionResult with bounding boxes and metadata

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If image is invalid
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, loading now...")
            self.load_model()

        if image is None or image.size == 0:
            raise ValueError("Invalid image: empty or None")

        # Use provided parameters or defaults
        scale_factor = scale_factor or self.scale_factor
        min_neighbors = min_neighbors or self.min_neighbors
        min_size = min_size or self.min_size

        # Convert to grayscale (Haar Cascades work on grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Measure inference time
        start_time = time.time()

        # Detect vehicles
        # NOTE: This is where the original code had a bug (ncars not initialized)
        # We properly handle the results here
        detections = self.model.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        inference_time = time.time() - start_time

        # Convert detections to our format
        boxes = []
        confidences = []
        class_ids = []
        class_names = []

        for (x, y, w, h) in detections:
            boxes.append((int(x), int(y), int(w), int(h)))
            # Haar Cascades don't provide confidence scores,
            # so we use a proxy based on detection quality
            confidence = 1.0  # Or could be based on neighbors/scale
            confidences.append(confidence)
            class_ids.append(0)  # Single class: "car"
            class_names.append("car")

        logger.debug(
            f"Detected {len(boxes)} vehicles in {inference_time*1000:.2f}ms "
            f"(scale_factor={scale_factor}, min_neighbors={min_neighbors})"
        )

        return DetectionResult(
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names,
            inference_time=inference_time,
            image_shape=image.shape,
        )

    def get_detector_name(self) -> str:
        """Get detector name."""
        return "Haar Cascade (Viola-Jones 2001)"

    def get_detector_info(self) -> dict:
        """Get detailed detector information."""
        return {
            "name": "Haar Cascade",
            "full_name": "Viola-Jones Haar Cascade Classifier",
            "year": 2001,
            "authors": "Paul Viola & Michael Jones",
            "paper": "Rapid Object Detection using a Boosted Cascade of Simple Features",
            "approach": "Classical Computer Vision",
            "algorithm": "AdaBoost with Haar-like features",
            "training": "Pre-trained on vehicle dataset",
            "speed": "Very Fast (CPU-only)",
            "accuracy": "Moderate (outdated by modern standards)",
            "advantages": [
                "Extremely fast on CPU",
                "Low memory footprint",
                "No GPU required",
                "Simple deployment"
            ],
            "limitations": [
                "Lower accuracy than deep learning",
                "High false positive rate",
                "Sensitive to orientation",
                "Limited to specific views"
            ],
            "parameters": {
                "scale_factor": self.scale_factor,
                "min_neighbors": self.min_neighbors,
                "min_size": self.min_size,
            }
        }
