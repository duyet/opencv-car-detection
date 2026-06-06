#!/usr/bin/env python3
"""
Benchmark script to compare detector performance.

Runs both Haar Cascade and YOLO on sample images and reports metrics.
"""

import sys
import time
from pathlib import Path
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detectors.haar_cascade import HaarCascadeDetector
from src.detectors.yolo import YOLODetector
from src.utils.image import load_image
from src.utils.metrics import calculate_fps


def benchmark():
    """Run benchmark on sample images."""
    print("🏁 Starting Detection Benchmark\n")

    # Find sample images
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return

    image_files = list(samples_dir.glob("car*.jpg"))
    if not image_files:
        print(f"❌ No sample images found in {samples_dir}")
        return

    print(f"📊 Testing on {len(image_files)} images\n")

    # Initialize detectors
    print("Loading detectors...")
    haar = HaarCascadeDetector()
    haar.load_model()
    print("✅ Haar Cascade loaded")

    try:
        yolo = YOLODetector(device="cpu")
        yolo.load_model()
        print("✅ YOLOv8 loaded\n")
        has_yolo = True
    except Exception as e:
        print(f"⚠️  YOLOv8 not available: {e}\n")
        has_yolo = False

    # Run benchmarks
    results = []

    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        img = load_image(img_path)

        # Haar Cascade
        haar_result = haar.detect(img)
        haar_fps = calculate_fps(haar_result.inference_time)

        row = [
            img_path.name,
            "Haar",
            haar_result.num_detections,
            f"{haar_result.inference_time*1000:.1f}",
            f"{haar_fps:.1f}"
        ]
        results.append(row)

        # YOLO
        if has_yolo:
            yolo_result = yolo.detect(img)
            yolo_fps = calculate_fps(yolo_result.inference_time)

            row = [
                img_path.name,
                "YOLO",
                yolo_result.num_detections,
                f"{yolo_result.inference_time*1000:.1f}",
                f"{yolo_fps:.1f}"
            ]
            results.append(row)

    # Print results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70 + "\n")

    headers = ["Image", "Detector", "Detections", "Time (ms)", "FPS"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

    print("\n✅ Benchmark complete!\n")


if __name__ == "__main__":
    try:
        benchmark()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
