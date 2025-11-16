"""Tests for image utilities."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.utils.image import (
    validate_image,
    validate_image_path,
    draw_detections,
    resize_image,
)


def test_validate_image_valid():
    """Test validation of valid image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    validate_image(img)  # Should not raise


def test_validate_image_invalid():
    """Test validation of invalid images."""
    with pytest.raises(ValueError):
        validate_image(None)

    with pytest.raises(ValueError):
        validate_image(np.array([]))

    # Too large
    with pytest.raises(ValueError):
        huge_img = np.zeros((20000, 20000, 3), dtype=np.uint8)
        validate_image(huge_img)


def test_draw_detections():
    """Test drawing detections on image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = [(100, 100, 50, 50), (200, 200, 60, 60)]
    labels = ["car", "car"]
    confidences = [0.9, 0.8]

    result = draw_detections(img, boxes, labels, confidences)

    assert result.shape == img.shape
    assert not np.array_equal(result, img)  # Image should be modified


def test_resize_image():
    """Test image resizing."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Resize with max width
    resized = resize_image(img, max_width=320)
    assert resized.shape[1] <= 320

    # Resize with max height
    resized = resize_image(img, max_height=240)
    assert resized.shape[0] <= 240

    # No resize needed
    resized = resize_image(img, max_width=1000, max_height=1000)
    assert resized.shape == img.shape
