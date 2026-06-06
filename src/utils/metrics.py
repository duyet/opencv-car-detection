"""
Performance metrics for object detection evaluation.

This module provides functions to calculate standard metrics like
IoU, precision, recall, F1-score, and FPS.
"""

from typing import List, Tuple, Dict
import time
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""
    precision: float
    recall: float
    f1_score: float
    average_iou: float
    true_positives: int
    false_positives: int
    false_negatives: int


def calculate_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box1: First box (x, y, w, h)
        box2: Second box (x, y, w, h)

    Returns:
        IoU score (0-1)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_precision_recall(
    predicted_boxes: List[Tuple[int, int, int, int]],
    ground_truth_boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5
) -> DetectionMetrics:
    """
    Calculate precision, recall, and F1-score for detections.

    Args:
        predicted_boxes: List of predicted bounding boxes
        ground_truth_boxes: List of ground truth bounding boxes
        iou_threshold: IoU threshold for matching (default: 0.5)

    Returns:
        DetectionMetrics object with calculated metrics
    """
    if len(ground_truth_boxes) == 0:
        # No ground truth - all predictions are false positives
        return DetectionMetrics(
            precision=0.0 if len(predicted_boxes) > 0 else 1.0,
            recall=1.0,  # Technically undefined, but we use 1.0
            f1_score=0.0,
            average_iou=0.0,
            true_positives=0,
            false_positives=len(predicted_boxes),
            false_negatives=0
        )

    if len(predicted_boxes) == 0:
        # No predictions - all ground truth are false negatives
        return DetectionMetrics(
            precision=1.0,  # Technically undefined, but we use 1.0
            recall=0.0,
            f1_score=0.0,
            average_iou=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=len(ground_truth_boxes)
        )

    # Match predictions to ground truth using Hungarian algorithm (greedy approximation)
    matched_gt = set()
    true_positives = 0
    iou_scores = []

    # For each prediction, find best matching ground truth
    for pred_box in predicted_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # If IoU exceeds threshold, count as true positive
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            iou_scores.append(best_iou)

    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(ground_truth_boxes) - true_positives

    # Calculate precision and recall
    precision = true_positives / len(predicted_boxes) if len(predicted_boxes) > 0 else 0.0
    recall = true_positives / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0.0

    # Calculate F1-score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Calculate average IoU for matched boxes
    average_iou = np.mean(iou_scores) if iou_scores else 0.0

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        average_iou=average_iou,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


def calculate_fps(inference_time: float) -> float:
    """
    Calculate frames per second from inference time.

    Args:
        inference_time: Time in seconds

    Returns:
        FPS (frames per second)
    """
    if inference_time <= 0:
        return 0.0
    return 1.0 / inference_time


class FPSCounter:
    """
    Real-time FPS counter for video processing.

    Example:
        counter = FPSCounter()
        for frame in video:
            counter.start()
            process(frame)
            counter.stop()
            fps = counter.fps()
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.start_time = None

    def start(self) -> None:
        """Start timing a frame."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timing a frame and record duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.frame_times.append(duration)

            # Keep only recent frames
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)

            self.start_time = None

    def fps(self) -> float:
        """
        Get current FPS.

        Returns:
            Average FPS over recent frames
        """
        if not self.frame_times:
            return 0.0

        avg_time = np.mean(self.frame_times)
        return calculate_fps(avg_time)

    def reset(self) -> None:
        """Reset the counter."""
        self.frame_times = []
        self.start_time = None
