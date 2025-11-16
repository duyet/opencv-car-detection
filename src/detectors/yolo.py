"""
YOLOv8 Detector - Modern Deep Learning (2023).

This module implements vehicle detection using Ultralytics YOLOv8, representing
the state-of-the-art in real-time object detection as of 2023.

Reference:
    Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
    Original YOLO: Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection
"""

import time
from pathlib import Path
from typing import Optional, List
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not available. Install with: pip install ultralytics")

from .base import BaseDetector, DetectionResult


class YOLODetector(BaseDetector):
    """
    YOLOv8-based vehicle detector.

    This detector uses Ultralytics YOLOv8, a modern deep learning approach that
    represents the cutting edge of object detection as of 2023.

    Advantages:
        - State-of-the-art accuracy
        - Real-time inference speed
        - Handles multiple scales and orientations
        - Low false positive rate
        - Detects multiple object classes
        - Provides confidence scores

    Limitations:
        - Requires more computational resources
        - Larger model size
        - May need GPU for real-time performance
        - More complex deployment
    """

    # Vehicle-related COCO class IDs
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_size: str = "n",  # n, s, m, l, x
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",  # 'cpu' or 'cuda'
        **kwargs
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            model_path: Path to custom YOLO model (if None, uses pre-trained)
            model_size: Model size - n(ano), s(mall), m(edium), l(arge), x(large)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLOv8 is not installed. Install with: pip install ultralytics torch torchvision"
            )

        super().__init__(model_path, **kwargs)

        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Use pre-trained COCO model if no custom path provided
        if model_path is None:
            self.model_name = f"yolov8{model_size}.pt"
        else:
            self.model_name = str(model_path)

        logger.info(
            f"Initialized {self.get_detector_name()} "
            f"(model={self.model_name}, device={device})"
        )

    def load_model(self) -> None:
        """
        Load the YOLOv8 model.

        The model will be automatically downloaded if using pre-trained weights.

        Raises:
            FileNotFoundError: If custom model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_name}")
            self.model = YOLO(self.model_name)

            # Move to specified device
            self.model.to(self.device)

            self.is_loaded = True
            logger.success(
                f"Loaded YOLOv8 model: {self.model_name} on {self.device}"
            )

        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise RuntimeError(f"Failed to load YOLOv8: {e}")

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        vehicle_only: bool = True,
        **kwargs
    ) -> DetectionResult:
        """
        Detect vehicles in an image using YOLOv8.

        Args:
            image: Input image (BGR format)
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            vehicle_only: If True, filter for vehicles only
            **kwargs: Additional parameters

        Returns:
            DetectionResult with bounding boxes, confidences, and class info

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
        conf_thresh = confidence_threshold or self.confidence_threshold
        iou_thresh = iou_threshold or self.iou_threshold

        # Measure inference time
        start_time = time.time()

        # Run inference
        results = self.model.predict(
            image,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
            device=self.device,
        )

        inference_time = time.time() - start_time

        # Parse results
        boxes = []
        confidences = []
        class_ids = []
        class_names = []

        if results and len(results) > 0:
            result = results[0]  # First image

            # Extract detections
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Convert to xywh format
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Filter for vehicles if requested
                    if vehicle_only and class_id not in self.VEHICLE_CLASSES:
                        continue

                    # Get class name
                    class_name = self.VEHICLE_CLASSES.get(
                        class_id,
                        result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    )

                    boxes.append((x, y, w, h))
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    class_names.append(class_name)

        logger.debug(
            f"YOLOv8 detected {len(boxes)} objects in {inference_time*1000:.2f}ms "
            f"(conf={conf_thresh}, iou={iou_thresh})"
        )

        return DetectionResult(
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names,
            inference_time=inference_time,
            image_shape=image.shape,
        )

    def detect_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8,
        **kwargs
    ) -> List[DetectionResult]:
        """
        Detect objects in multiple images with batching optimization.

        Args:
            images: List of input images
            batch_size: Number of images per batch
            **kwargs: Additional parameters

        Returns:
            List of DetectionResult objects
        """
        if not self.is_loaded:
            self.load_model()

        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = [self.detect(img, **kwargs) for img in batch]
            results.extend(batch_results)

        return results

    def get_detector_name(self) -> str:
        """Get detector name."""
        return f"YOLOv8-{self.model_size.upper()}"

    def get_detector_info(self) -> dict:
        """Get detailed detector information."""
        model_sizes = {
            "n": "Nano (fastest, least accurate)",
            "s": "Small (balanced)",
            "m": "Medium (good tradeoff)",
            "l": "Large (more accurate)",
            "x": "Extra Large (most accurate, slowest)",
        }

        return {
            "name": f"YOLOv8-{self.model_size.upper()}",
            "full_name": "You Only Look Once v8",
            "year": 2023,
            "organization": "Ultralytics",
            "paper": "YOLOv8: State-of-the-art Object Detection",
            "approach": "Deep Learning (CNN)",
            "architecture": "CSPDarknet + PANet + YOLOv8 Head",
            "training": "Pre-trained on COCO dataset",
            "speed": "Real-time (GPU recommended)",
            "accuracy": "State-of-the-art",
            "model_size": model_sizes.get(self.model_size, "Unknown"),
            "advantages": [
                "State-of-the-art accuracy",
                "Real-time inference",
                "Multi-scale detection",
                "Confidence scores",
                "Multiple object classes"
            ],
            "limitations": [
                "Requires more compute resources",
                "Larger model size",
                "GPU recommended for real-time",
                "More complex deployment"
            ],
            "parameters": {
                "model_size": self.model_size,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "device": self.device,
            },
            "detected_classes": list(self.VEHICLE_CLASSES.values()),
        }
