"""
Beautiful visualization tools for comparing detection methods.

This module creates publication-quality comparison charts and visualizations
to showcase the evolution from classical to modern computer vision.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from ..detectors.base import DetectionResult
from ..utils.image import draw_detections


def create_comparison_image(
    original_image: np.ndarray,
    detections: Dict[str, DetectionResult],
    colors: Dict[str, Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Create a side-by-side comparison of different detection methods.

    Args:
        original_image: Original input image
        detections: Dictionary mapping detector names to DetectionResult
        colors: Optional dictionary mapping detector names to BGR colors

    Returns:
        Combined comparison image
    """
    if colors is None:
        # Default colors
        colors = {
            "Haar Cascade": (0, 0, 255),  # Red
            "YOLO": (0, 255, 0),  # Green
        }

    # Create output images for each detector
    detector_images = []
    detector_names = []

    for name, result in detections.items():
        color = colors.get(name, (255, 255, 0))  # Default cyan
        img_with_boxes = draw_detections(
            original_image,
            result.boxes,
            labels=[f"{cn}" for cn in result.class_names],
            confidences=result.confidences,
            color=color,
            thickness=2,
        )

        # Add detector name and stats at the top
        stats_text = f"{name}: {result.num_detections} detections ({result.inference_time*1000:.1f}ms)"
        cv2.putText(
            img_with_boxes,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        detector_images.append(img_with_boxes)
        detector_names.append(name)

    # Combine images horizontally
    combined = np.hstack(detector_images)
    return combined


def create_side_by_side_comparison(
    image: np.ndarray,
    result1: DetectionResult,
    result2: DetectionResult,
    title1: str = "Method 1",
    title2: str = "Method 2",
    color1: Tuple[int, int, int] = (0, 0, 255),
    color2: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Create side-by-side comparison of two detection methods.

    Args:
        image: Original image
        result1: First detection result
        result2: Second detection result
        title1: Title for first method
        title2: Title for second method
        color1: Color for first method boxes (BGR)
        color2: Color for second method boxes (BGR)

    Returns:
        Side-by-side comparison image
    """
    # Draw detections
    img1 = draw_detections(image, result1.boxes, confidences=result1.confidences, color=color1)
    img2 = draw_detections(image, result2.boxes, confidences=result2.confidences, color=color2)

    # Add titles and stats
    stats1 = f"{title1}: {result1.num_detections} ({result1.inference_time*1000:.1f}ms)"
    stats2 = f"{title2}: {result2.num_detections} ({result2.inference_time*1000:.1f}ms)"

    cv2.putText(img1, stats1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img2, stats2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine horizontally
    combined = np.hstack([img1, img2])
    return combined


def create_performance_chart(
    results: Dict[str, Dict[str, Any]],
    output_path: Path = None,
) -> None:
    """
    Create a beautiful performance comparison chart.

    Args:
        results: Dictionary mapping detector names to their metrics
                 e.g., {"Haar": {"fps": 45, "detections": 10}, "YOLO": {...}}
        output_path: Optional path to save chart
    """
    detector_names = list(results.keys())
    metrics = ["fps", "detections", "inference_time_ms"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Detection Performance Comparison: Classical vs Modern', fontsize=16, fontweight='bold')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # FPS chart
    fps_values = [results[name].get("fps", 0) for name in detector_names]
    axes[0].bar(detector_names, fps_values, color=colors[:len(detector_names)])
    axes[0].set_title('Speed (FPS)', fontweight='bold')
    axes[0].set_ylabel('Frames Per Second')
    axes[0].grid(axis='y', alpha=0.3)

    # Detections chart
    detection_counts = [results[name].get("detections", 0) for name in detector_names]
    axes[1].bar(detector_names, detection_counts, color=colors[:len(detector_names)])
    axes[1].set_title('Detection Count', fontweight='bold')
    axes[1].set_ylabel('Number of Detections')
    axes[1].grid(axis='y', alpha=0.3)

    # Inference time chart
    inference_times = [results[name].get("inference_time_ms", 0) for name in detector_names]
    axes[2].bar(detector_names, inference_times, color=colors[:len(detector_names)])
    axes[2].set_title('Inference Time', fontweight='bold')
    axes[2].set_ylabel('Time (ms)')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_evolution_timeline() -> None:
    """
    Create a visual timeline showing the evolution of object detection.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    milestones = [
        (2001, "Haar Cascades\n(Viola-Jones)", "#FF6B6B"),
        (2012, "AlexNet\n(Deep Learning Era)", "#FFD93D"),
        (2014, "R-CNN", "#6BCF7F"),
        (2016, "YOLO v1\n(Real-time Detection)", "#4ECDC4"),
        (2020, "YOLOv5", "#45B7D1"),
        (2023, "YOLOv8\n(State-of-the-art)", "#A8E6CF"),
    ]

    years = [m[0] for m in milestones]
    labels = [m[1] for m in milestones]
    colors = [m[2] for m in milestones]

    # Plot timeline
    ax.scatter(years, [1]*len(years), s=500, c=colors, alpha=0.6, edgecolors='black', linewidth=2)

    # Add labels
    for year, label, color in milestones:
        ax.text(year, 1.15, label, ha='center', fontsize=10, fontweight='bold')

    # Styling
    ax.set_ylim(0.8, 1.4)
    ax.set_xlim(1998, 2025)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_title('Evolution of Object Detection: 2001 → 2023', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('data/results/detection_evolution_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
