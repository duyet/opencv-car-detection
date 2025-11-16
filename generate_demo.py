#!/usr/bin/env python3
"""
Generate demo screenshots for README and documentation.

This script creates professional demo images showing:
- Individual detector results
- Side-by-side comparisons
- Multiple test images
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detectors.haar_cascade import HaarCascadeDetector
from src.utils.image import load_image, save_image, draw_detections
from src.visualization.visualizer import create_side_by_side_comparison

print("🎨 Generating demo screenshots for README...")
print("=" * 70)

# Load Haar detector (YOLO optional for now)
haar = HaarCascadeDetector()
haar.load_model()
print("✅ Haar Cascade loaded\n")

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
    img = load_image(img_path)
    haar_result = haar.detect(img)

    # Create annotated image with better styling
    img_with_boxes = draw_detections(
        img,
        haar_result.boxes,
        confidences=haar_result.confidences,
        color=(0, 0, 255),  # Red for Haar
        thickness=3
    )

    # Add info overlay
    fps = 1.0 / haar_result.inference_time if haar_result.inference_time > 0 else 0
    info_text = f"Haar Cascade: {haar_result.num_detections} vehicles | {haar_result.inference_time*1000:.1f}ms | {fps:.1f} FPS"

    # Add semi-transparent background for text
    overlay = img_with_boxes.copy()
    cv2.rectangle(overlay, (10, 10), (len(info_text) * 12, 50), (0, 0, 0), -1)
    img_with_boxes = cv2.addWeighted(overlay, 0.3, img_with_boxes, 0.7, 0)

    # Add text
    cv2.putText(
        img_with_boxes,
        info_text,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Save with descriptive name
    base_name = Path(img_name).stem
    output_path = f"data/results/demo_{base_name}.jpg"
    save_image(img_with_boxes, output_path)
    print(f"✅ Saved: {output_path}")
    print(f"   {haar_result.num_detections} vehicles | {haar_result.inference_time*1000:.1f}ms | {fps:.1f} FPS\n")

# Use car2 as the main demo image
main_demo_path = "data/results/demo_car2.jpg"
if Path(main_demo_path).exists():
    import shutil
    shutil.copy(main_demo_path, "data/results/demo_detection.jpg")
    print("📋 Copied demo_car2.jpg → demo_detection.jpg (main README image)")
else:
    print("⚠️  Main demo image not found - using existing demo_detection.jpg")

# Try YOLO if available
try:
    from src.detectors.yolo import YOLODetector
    print("\n📸 Creating comparison with YOLOv8...")

    yolo = YOLODetector(device="cpu")
    yolo.load_model()
    yolo_result = yolo.detect(img)

    # Create side-by-side comparison
    comparison = create_side_by_side_comparison(
        img,
        haar_result,
        yolo_result,
        title1='Haar Cascade (2001)',
        title2='YOLOv8 (2023)',
        color1=(0, 0, 255),
        color2=(0, 255, 0)
    )

    comparison_path = "data/results/comparison_demo.jpg"
    save_image(comparison, comparison_path)
    print(f"✅ Saved: {comparison_path}")
    print(f"   YOLO detected {yolo_result.num_detections} vehicles in {yolo_result.inference_time*1000:.1f}ms")

except Exception as e:
    print(f"⚠️  YOLO comparison skipped: {e}")
    print("   (This is OK - Haar Cascade demo still works)")

print("\n🎉 Demo screenshots generated!")
print("   Check data/results/ for output images")
