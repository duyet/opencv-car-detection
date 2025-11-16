#!/usr/bin/env python3
"""
Generate demo screenshots for README.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detectors.haar_cascade import HaarCascadeDetector
from src.utils.image import load_image, save_image, draw_detections
from src.visualization.visualizer import create_side_by_side_comparison

print("🎨 Generating demo screenshots for README...")

# Load Haar detector (YOLO optional for now)
haar = HaarCascadeDetector()
haar.load_model()
print("✅ Haar Cascade loaded")

# Process car2.jpg (nice traffic scene)
print("\n📸 Processing car2.jpg...")
img = load_image("data/samples/car2.jpg")
haar_result = haar.detect(img)

# Create annotated image
img_with_boxes = draw_detections(
    img,
    haar_result.boxes,
    confidences=haar_result.confidences,
    color=(0, 0, 255),  # Red
    thickness=3
)

# Save for README
output_path = "data/results/demo_detection.jpg"
save_image(img_with_boxes, output_path)
print(f"✅ Saved: {output_path}")
print(f"   Detected {haar_result.num_detections} vehicles in {haar_result.inference_time*1000:.1f}ms")

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
