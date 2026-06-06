"""Tests for Haar Cascade detector."""

import pytest
import numpy as np
from pathlib import Path

from src.detectors.haar_cascade import HaarCascadeDetector
from src.detectors.base import DetectionResult


@pytest.fixture
def test_image():
    """Create a simple test image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def detector():
    """Create Haar Cascade detector instance."""
    return HaarCascadeDetector()


def test_detector_initialization(detector):
    """Test detector can be initialized."""
    assert detector is not None
    assert not detector.is_loaded


def test_detector_load_model(detector):
    """Test model loading."""
    detector.load_model()
    assert detector.is_loaded
    assert detector.model is not None


def test_detector_info(detector):
    """Test detector info."""
    info = detector.get_detector_info()
    assert info['year'] == 2001
    assert info['name'] == 'Haar Cascade'
    assert 'advantages' in info
    assert 'limitations' in info


def test_detect_on_image(detector, test_image):
    """Test detection on image."""
    detector.load_model()
    result = detector.detect(test_image)

    assert isinstance(result, DetectionResult)
    assert isinstance(result.boxes, list)
    assert isinstance(result.confidences, list)
    assert isinstance(result.class_names, list)
    assert result.inference_time > 0


def test_detect_invalid_image(detector):
    """Test detection with invalid image."""
    detector.load_model()

    with pytest.raises(ValueError):
        detector.detect(None)

    with pytest.raises(ValueError):
        detector.detect(np.array([]))


def test_detector_name(detector):
    """Test detector name."""
    name = detector.get_detector_name()
    assert "Haar Cascade" in name
    assert "2001" in name
