#!/usr/bin/env python3
"""
Generate demo screenshots for README and documentation.

This script uses the ACTUAL production codebase to generate demo images:
- Imports detectors from src/detectors/
- Uses production image utilities from src/utils/
- Creates professional overlays with performance metrics
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path to import production code
sys.path.insert(0, str(Path(__file__).parent))

# Import from PRODUCTION codebase
from src.detectors.haar_cascade import HaarCascadeDetector
from src.utils.image import load_image, save_image
from src.utils.metrics import calculate_fps

print("🎨 Generating Professional Demo Screenshots")
print("=" * 70)
print("📦 Using PRODUCTION codebase from src/")
print("   - Detectors: src/detectors/haar_cascade.py")
print("   - Utils: src/utils/image.py, src/utils/metrics.py")
print("=" * 70)

# Initialize detector using production code
print("\n⚙️  Initializing Haar Cascade detector...")
haar = HaarCascadeDetector()
haar.load_model()
detector_info = haar.get_detector_info()
print(f"✅ Loaded: {detector_info['name']} ({detector_info['year']})")
print(f"   Algorithm: {detector_info['algorithm']}")
print()

def draw_detections_with_boxes(img, boxes, color=(0, 0, 255), thickness=3):
    """Draw detection boxes on image using production code."""
    img_copy = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
    return img_copy


def create_professional_overlay(img, detector_name, num_detections, inference_time_ms, fps):
    """Create a professional overlay with gradient background and metrics."""
    height, width = img.shape[:2]
    overlay = img.copy()

    # Create gradient background for overlay (top section)
    overlay_height = 100
    gradient = np.zeros((overlay_height, width, 3), dtype=np.uint8)

    # Create blue-to-dark gradient
    for i in range(overlay_height):
        alpha = i / overlay_height
        # Dark blue to black gradient
        color = (int(40 * (1 - alpha)), int(20 * (1 - alpha)), int(80 * (1 - alpha)))
        gradient[i, :] = color

    # Blend gradient onto image
    overlay[0:overlay_height, 0:width] = cv2.addWeighted(
        overlay[0:overlay_height, 0:width], 0.4,
        gradient, 0.6,
        0
    )

    # Add detector name (large, bold)
    cv2.putText(
        overlay,
        detector_name,
        (20, 35),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Add metrics in a cleaner format
    metrics_y = 70

    # Detections
    cv2.putText(
        overlay,
        f"Detections: {num_detections}",
        (20, metrics_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 255, 100),  # Green
        2,
        cv2.LINE_AA
    )

    # Inference time
    cv2.putText(
        overlay,
        f"Time: {inference_time_ms:.1f}ms",
        (220, metrics_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 200, 255),  # Light blue
        2,
        cv2.LINE_AA
    )

    # FPS
    cv2.putText(
        overlay,
        f"FPS: {fps:.1f}",
        (420, metrics_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 100),  # Orange
        2,
        cv2.LINE_AA
    )

    # Add subtle "LIVE DEMO" badge
    badge_text = "LIVE DEMO"
    badge_x = width - 150
    cv2.rectangle(overlay, (badge_x - 10, 10), (width - 10, 45), (255, 0, 0), -1)
    cv2.putText(
        overlay,
        badge_text,
        (badge_x, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return overlay


# Process multiple images for comprehensive demo
test_images = [
    ("car2.jpg", "Traffic scene - multiple vehicles"),
    ("car1.jpg", "Highway scene"),
]

for img_name, description in test_images:
    img_path = f"data/samples/{img_name}"
    if not Path(img_path).exists():
        print(f"⚠️  Skipping {img_name} - file not found")
        continue

    print(f"📸 Processing {img_name} ({description})...")

    # Load image using production utility
    img = load_image(img_path)

    # Detect using production detector
    haar_result = haar.detect(img)

    # Calculate metrics using production code
    fps = calculate_fps(haar_result.inference_time)
    inference_ms = haar_result.inference_time * 1000

    # Draw detection boxes using production approach
    img_with_boxes = draw_detections_with_boxes(
        img,
        haar_result.boxes,
        color=(0, 0, 255),  # Red for Haar
        thickness=3
    )

    # Add professional overlay
    final_img = create_professional_overlay(
        img_with_boxes,
        "Haar Cascade (2001)",
        haar_result.num_detections,
        inference_ms,
        fps
    )

    # Save with descriptive name
    base_name = Path(img_name).stem
    output_path = f"data/results/demo_{base_name}.jpg"
    save_image(final_img, output_path)

    print(f"✅ Saved: {output_path}")
    print(f"   📊 {haar_result.num_detections} vehicles | {inference_ms:.1f}ms | {fps:.1f} FPS")
    print(f"   🎯 Using production detector: {detector_info['full_name']}")
    print()

# Use car2 as the main demo image for README
print("=" * 70)
main_demo_path = "data/results/demo_car2.jpg"
if Path(main_demo_path).exists():
    import shutil
    shutil.copy(main_demo_path, "data/results/demo_detection.jpg")
    print("📋 Copied demo_car2.jpg → demo_detection.jpg (main README image)")
else:
    print("⚠️  Main demo image not found - using existing demo_detection.jpg")

# Summary
print("\n" + "=" * 70)
print("🎉 Professional Demo Screenshots Generated!")
print("=" * 70)
print("📂 Output files:")
print("   - data/results/demo_car1.jpg (Highway scene)")
print("   - data/results/demo_car2.jpg (Traffic scene)")
print("   - data/results/demo_detection.jpg (Main README image)")
print("\n✨ Features:")
print("   - Gradient overlay backgrounds")
print("   - Color-coded metrics (Detections, Time, FPS)")
print("   - LIVE DEMO badge")
print("   - Using PRODUCTION codebase from src/")
print("\n📊 All images use actual detection results from:")
print(f"   {detector_info['full_name']}")
print("=" * 70)
