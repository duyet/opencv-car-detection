#!/usr/bin/env python3
"""
Flask web application for vehicle detection.

This provides a beautiful browser-based interface for comparing
Haar Cascade and YOLO detectors.
"""

import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from loguru import logger
import cv2
import numpy as np
import base64

# Add parent directory to path to import src modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors.haar_cascade import HaarCascadeDetector
from src.detectors.yolo import YOLODetector
from src.utils.image import load_image, save_image, validate_image_path, ALLOWED_EXTENSIONS
from src.utils.config import get_default_config
from src.visualization.visualizer import create_side_by_side_comparison

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = get_default_config()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'opencv-car-detection-secret-key')
app.config['MAX_CONTENT_LENGTH'] = config['web']['max_file_size']
app.config['UPLOAD_FOLDER'] = Path(config['web']['upload_folder'])
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

# Rate limiting for security
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[config['security']['rate_limit']]
)

# Initialize detectors (lazy loading)
haar_detector = None
yolo_detector = None


def get_haar_detector():
    """Get or initialize Haar Cascade detector."""
    global haar_detector
    if haar_detector is None:
        haar_detector = HaarCascadeDetector()
        haar_detector.load_model()
        logger.info("Haar Cascade detector loaded")
    return haar_detector


def get_yolo_detector():
    """Get or initialize YOLO detector."""
    global yolo_detector
    if yolo_detector is None:
        try:
            device = config['detectors']['yolo']['device']
            yolo_detector = YOLODetector(device=device)
            yolo_detector.load_model()
            logger.info(f"YOLO detector loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            raise
    return yolo_detector


def allowed_file(filename):
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in {ext.lower() for ext in ALLOWED_EXTENSIONS}


def image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    _, buffer = cv2.imencode('.jpg', image_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
@limiter.limit("10 per minute")
def detect():
    """
    API endpoint for vehicle detection.

    Accepts an uploaded image and runs detection with selected detector(s).
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        detector_type = request.form.get('detector', 'both')

        # Save uploaded file
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)

        logger.info(f"Processing {filename} with detector={detector_type}")

        # Load image
        image = load_image(filepath)

        results = {}
        images = {}

        # Run Haar Cascade
        if detector_type in ['haar', 'both']:
            haar = get_haar_detector()
            haar_result = haar.detect(image)
            results['haar'] = {
                'name': 'Haar Cascade (2001)',
                'detections': haar_result.num_detections,
                'inference_time_ms': round(haar_result.inference_time * 1000, 2),
                'fps': round(1.0 / haar_result.inference_time if haar_result.inference_time > 0 else 0, 1),
                'boxes': [(int(x), int(y), int(w), int(h)) for x, y, w, h in haar_result.boxes],
            }

            # Create annotated image
            from src.utils.image import draw_detections
            haar_img = draw_detections(image, haar_result.boxes, color=(0, 0, 255))
            images['haar'] = image_to_base64(haar_img)

        # Run YOLO
        if detector_type in ['yolo', 'both']:
            try:
                yolo = get_yolo_detector()
                yolo_result = yolo.detect(image)
                results['yolo'] = {
                    'name': 'YOLOv8 (2023)',
                    'detections': yolo_result.num_detections,
                    'inference_time_ms': round(yolo_result.inference_time * 1000, 2),
                    'fps': round(1.0 / yolo_result.inference_time if yolo_result.inference_time > 0 else 0, 1),
                    'boxes': [(int(x), int(y), int(w), int(h)) for x, y, w, h in yolo_result.boxes],
                    'classes': yolo_result.class_names,
                }

                # Create annotated image
                from src.utils.image import draw_detections
                yolo_img = draw_detections(
                    image,
                    yolo_result.boxes,
                    labels=yolo_result.class_names,
                    confidences=yolo_result.confidences,
                    color=(0, 255, 0)
                )
                images['yolo'] = image_to_base64(yolo_img)

            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
                results['yolo'] = {'error': str(e)}

        # Create comparison if both
        if detector_type == 'both' and 'haar' in results and 'yolo' in results:
            haar_res = haar.detect(image)
            yolo_res = yolo.detect(image)
            comparison = create_side_by_side_comparison(
                image, haar_res, yolo_res,
                title1='Haar Cascade (2001)',
                title2='YOLOv8 (2023)'
            )
            images['comparison'] = image_to_base64(comparison)

        # Cleanup
        os.remove(filepath)

        return jsonify({
            'success': True,
            'results': results,
            'images': images,
        })

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/info', methods=['GET'])
def info():
    """Get information about available detectors."""
    info_data = {
        'detectors': {
            'haar': {
                'available': True,
                'name': 'Haar Cascade',
                'year': 2001,
                'description': 'Classical computer vision using Viola-Jones algorithm',
            },
            'yolo': {
                'available': True,
                'name': 'YOLOv8',
                'year': 2023,
                'description': 'Modern deep learning for real-time object detection',
            },
        },
        'version': '2.0.0',
    }

    try:
        # Try to get YOLO info
        yolo = get_yolo_detector()
        info_data['detectors']['yolo']['info'] = yolo.get_detector_info()
    except Exception as e:
        info_data['detectors']['yolo']['available'] = False
        info_data['detectors']['yolo']['error'] = str(e)

    return jsonify(info_data)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'version': '2.0.0'})


def main():
    """Run the Flask application."""
    host = config['web']['host']
    port = config['web']['port']
    debug = config['web']['debug']

    logger.info(f"Starting web server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
