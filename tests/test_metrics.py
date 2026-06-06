"""Tests for metrics module."""

import pytest
from src.utils.metrics import calculate_iou, calculate_fps, FPSCounter


def test_calculate_iou_perfect_overlap():
    """Test IoU with perfect overlap."""
    box1 = (0, 0, 100, 100)
    box2 = (0, 0, 100, 100)
    iou = calculate_iou(box1, box2)
    assert iou == pytest.approx(1.0)


def test_calculate_iou_no_overlap():
    """Test IoU with no overlap."""
    box1 = (0, 0, 100, 100)
    box2 = (200, 200, 100, 100)
    iou = calculate_iou(box1, box2)
    assert iou == 0.0


def test_calculate_iou_partial_overlap():
    """Test IoU with partial overlap."""
    box1 = (0, 0, 100, 100)
    box2 = (50, 50, 100, 100)
    iou = calculate_iou(box1, box2)
    assert 0.0 < iou < 1.0


def test_calculate_fps():
    """Test FPS calculation."""
    fps = calculate_fps(0.1)  # 100ms
    assert fps == pytest.approx(10.0)

    fps = calculate_fps(0.01)  # 10ms
    assert fps == pytest.approx(100.0)


def test_fps_counter():
    """Test FPS counter."""
    counter = FPSCounter(window_size=5)

    # Simulate frames
    for _ in range(10):
        counter.start()
        # Simulate processing
        counter.stop()

    fps = counter.fps()
    assert fps > 0

    counter.reset()
    assert counter.fps() == 0.0
