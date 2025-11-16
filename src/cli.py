#!/usr/bin/env python3
"""
Command-line interface for vehicle detection.

This CLI provides an easy-to-use interface for running vehicle detection
on images, videos, and webcams using either Haar Cascades or YOLO.
"""

import sys
import click
from pathlib import Path
from loguru import logger
import cv2

from .detectors.haar_cascade import HaarCascadeDetector
from .detectors.yolo import YOLODetector
from .utils.image import load_image, save_image, draw_detections
from .utils.config import load_config, get_default_config
from .utils.metrics import calculate_fps
from .visualization.visualizer import create_side_by_side_comparison, create_performance_chart


# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """
    OpenCV Car Detection - Computer Vision Evolution

    Compare classical Haar Cascades (2001) with modern YOLOv8 (2023)
    for vehicle detection.
    """
    pass


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option(
    '--detector',
    '-d',
    type=click.Choice(['haar', 'yolo', 'both'], case_sensitive=False),
    default='both',
    help='Which detector to use'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    default=None,
    help='Output image path'
)
@click.option(
    '--show/--no-show',
    default=True,
    help='Display result image'
)
@click.option(
    '--confidence',
    '-c',
    type=float,
    default=0.25,
    help='Confidence threshold for YOLO (0-1)'
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'cuda']),
    default='cpu',
    help='Device for YOLO inference'
)
def detect(image_path, detector, output, show, confidence, device):
    """
    Detect vehicles in an image.

    Example:
        car-detect detect car.jpg
        car-detect detect car.jpg --detector yolo --output result.jpg
    """
    logger.info(f"Loading image: {image_path}")
    image = load_image(image_path)

    results = {}

    # Run Haar Cascade
    if detector in ['haar', 'both']:
        logger.info("Running Haar Cascade detector...")
        haar_detector = HaarCascadeDetector()
        haar_detector.load_model()
        haar_result = haar_detector.detect(image)
        results['Haar Cascade'] = haar_result
        logger.success(
            f"Haar Cascade: {haar_result.num_detections} detections "
            f"({haar_result.inference_time*1000:.1f}ms, "
            f"{calculate_fps(haar_result.inference_time):.1f} FPS)"
        )

    # Run YOLO
    if detector in ['yolo', 'both']:
        logger.info(f"Running YOLOv8 detector (device={device})...")
        try:
            yolo_detector = YOLODetector(confidence_threshold=confidence, device=device)
            yolo_detector.load_model()
            yolo_result = yolo_detector.detect(image)
            results['YOLOv8'] = yolo_result
            logger.success(
                f"YOLOv8: {yolo_result.num_detections} detections "
                f"({yolo_result.inference_time*1000:.1f}ms, "
                f"{calculate_fps(yolo_result.inference_time):.1f} FPS)"
            )
        except ImportError as e:
            logger.error("YOLOv8 not available. Install with: pip install ultralytics torch")
            if detector == 'yolo':
                return

    # Create visualization
    if len(results) == 2:
        # Side-by-side comparison
        output_image = create_side_by_side_comparison(
            image,
            results['Haar Cascade'],
            results['YOLOv8'],
            title1='Haar Cascade (2001)',
            title2='YOLOv8 (2023)',
            color1=(0, 0, 255),
            color2=(0, 255, 0)
        )
    elif 'Haar Cascade' in results:
        output_image = draw_detections(
            image,
            results['Haar Cascade'].boxes,
            confidences=results['Haar Cascade'].confidences,
            color=(0, 0, 255)
        )
    else:
        output_image = draw_detections(
            image,
            results['YOLOv8'].boxes,
            confidences=results['YOLOv8'].confidences,
            labels=results['YOLOv8'].class_names,
            color=(0, 255, 0)
        )

    # Save output
    if output:
        save_image(output_image, output)
        logger.success(f"Saved result to: {output}")

    # Display
    if show:
        cv2.imshow('Vehicle Detection', output_image)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option(
    '--detector',
    '-d',
    type=click.Choice(['haar', 'yolo'], case_sensitive=False),
    default='yolo',
    help='Which detector to use'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    default=None,
    help='Output video path'
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'cuda']),
    default='cpu',
    help='Device for inference'
)
def video(video_path, detector, output, device):
    """
    Detect vehicles in a video file.

    Example:
        car-detect video traffic.mp4 --detector yolo
    """
    logger.info(f"Processing video: {video_path}")

    # Initialize detector
    if detector == 'haar':
        det = HaarCascadeDetector()
    else:
        det = YOLODetector(device=device)

    det.load_model()
    logger.success(f"Loaded {det.get_detector_name()}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Output writer
    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    from tqdm import tqdm
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            result = det.detect(frame)

            # Draw
            frame_with_boxes = draw_detections(
                frame,
                result.boxes,
                labels=result.class_names,
                confidences=result.confidences
            )

            # Write
            if writer:
                writer.write(frame_with_boxes)

            # Show
            cv2.imshow('Detection', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            pbar.update(1)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.success(f"Processed {frame_count} frames")
    if output:
        logger.success(f"Saved to: {output}")


@cli.command()
@click.option(
    '--detector',
    '-d',
    type=click.Choice(['haar', 'yolo'], case_sensitive=False),
    default='yolo',
    help='Which detector to use'
)
@click.option(
    '--camera',
    type=int,
    default=0,
    help='Camera index (default: 0)'
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'cuda']),
    default='cpu',
    help='Device for inference'
)
def webcam(detector, camera, device):
    """
    Run real-time detection on webcam.

    Example:
        car-detect webcam --detector yolo

    Press 'q' to quit.
    """
    logger.info(f"Starting webcam detection (camera {camera})...")

    # Initialize detector
    if detector == 'haar':
        det = HaarCascadeDetector()
    else:
        det = YOLODetector(device=device)

    det.load_model()
    logger.success(f"Loaded {det.get_detector_name()}")

    # Open webcam
    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera}")
        return

    logger.info("Press 'q' to quit")

    from .utils.metrics import FPSCounter
    fps_counter = FPSCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter.start()

        # Detect
        result = det.detect(frame)

        # Draw
        frame_with_boxes = draw_detections(
            frame,
            result.boxes,
            labels=result.class_names,
            confidences=result.confidences
        )

        fps_counter.stop()

        # Add FPS overlay
        fps_text = f"FPS: {fps_counter.fps():.1f} | Detections: {result.num_detections}"
        cv2.putText(
            frame_with_boxes,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Show
        cv2.imshow('Webcam Detection', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam detection stopped")


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    default='data/results/comparison.jpg',
    help='Output image path'
)
def compare(image_path, output):
    """
    Run both detectors and create visual comparison.

    Example:
        car-detect compare traffic.jpg
    """
    logger.info("Running comprehensive comparison...")

    image = load_image(image_path)

    # Haar Cascade
    logger.info("Running Haar Cascade...")
    haar = HaarCascadeDetector()
    haar.load_model()
    haar_result = haar.detect(image)

    # YOLO
    logger.info("Running YOLOv8...")
    try:
        yolo = YOLODetector()
        yolo.load_model()
        yolo_result = yolo.detect(image)
    except ImportError:
        logger.error("YOLOv8 not available")
        return

    # Create comparison
    comparison = create_side_by_side_comparison(
        image,
        haar_result,
        yolo_result,
        title1='Haar Cascade (2001)',
        title2='YOLOv8 (2023)'
    )

    save_image(comparison, output)
    logger.success(f"Comparison saved to: {output}")

    # Print stats
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Haar Cascade: {haar_result.num_detections} detections, "
                f"{haar_result.inference_time*1000:.1f}ms, "
                f"{calculate_fps(haar_result.inference_time):.1f} FPS")
    logger.info(f"YOLOv8:       {yolo_result.num_detections} detections, "
                f"{yolo_result.inference_time*1000:.1f}ms, "
                f"{calculate_fps(yolo_result.inference_time):.1f} FPS")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
