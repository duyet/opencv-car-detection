"""
Image processing utilities with security and validation.

This module provides safe image loading, validation, and visualization functions
with proper error handling and security measures against malicious files.
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from loguru import logger

# Security: Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Security: Maximum image dimensions (prevent DoS via huge images)
MAX_IMAGE_DIMENSION = 10000  # 10k pixels
MAX_IMAGE_PIXELS = 100_000_000  # 100 megapixels


def validate_image_path(image_path: Union[str, Path]) -> Path:
    """
    Validate image file path for security.

    Args:
        image_path: Path to image file

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file extension is not allowed
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Security: Check file extension
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Invalid image extension: {path.suffix}\n"
            f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    return path


def validate_image(image: np.ndarray) -> None:
    """
    Validate image array for security and correctness.

    Args:
        image: Image as numpy array

    Raises:
        ValueError: If image is invalid or exceeds size limits
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or None")

    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D array.")

    # Security: Check image dimensions
    height, width = image.shape[:2]

    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions too large: {width}x{height}\n"
            f"Maximum dimension: {MAX_IMAGE_DIMENSION} pixels"
        )

    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image too large: {width * height} pixels\n"
            f"Maximum pixels: {MAX_IMAGE_PIXELS}"
        )


def load_image(image_path: Union[str, Path], validate: bool = True) -> np.ndarray:
    """
    Safely load an image from file with validation.

    Args:
        image_path: Path to image file
        validate: Whether to perform security validation

    Returns:
        Image as numpy array (BGR format)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid or too large
        IOError: If image cannot be read
    """
    # Validate path
    path = validate_image_path(image_path)

    try:
        # Load using OpenCV
        image = cv2.imread(str(path))

        if image is None:
            # Fallback to PIL for better format support
            logger.debug(f"OpenCV failed, trying PIL for {path}")
            pil_image = Image.open(path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if image is None:
            raise IOError(f"Failed to load image: {path}")

        # Validate image
        if validate:
            validate_image(image)

        logger.debug(f"Loaded image: {path} (shape={image.shape})")
        return image

    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        raise


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    create_dirs: bool = True,
    quality: int = 95
) -> Path:
    """
    Save image to file with proper error handling.

    Args:
        image: Image as numpy array (BGR format)
        output_path: Path to save image
        create_dirs: Create parent directories if they don't exist
        quality: JPEG quality (1-100)

    Returns:
        Path to saved image

    Raises:
        ValueError: If image is invalid
        IOError: If image cannot be saved
    """
    validate_image(image)

    output_path = Path(output_path)

    # Create parent directories
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save with quality settings
        params = []
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif output_path.suffix.lower() == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9]

        success = cv2.imwrite(str(output_path), image, params)

        if not success:
            raise IOError(f"Failed to save image: {output_path}")

        logger.debug(f"Saved image: {output_path} (shape={image.shape})")
        return output_path

    except Exception as e:
        logger.error(f"Error saving image {output_path}: {e}")
        raise


def draw_detections(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),  # Green
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes (x, y, w, h)
        labels: Optional list of labels for each box
        confidences: Optional list of confidence scores
        color: BGR color for boxes and text
        thickness: Line thickness
        font_scale: Font scale for text

    Returns:
        Image with drawn detections
    """
    # Create a copy to avoid modifying original
    output = image.copy()

    for i, (x, y, w, h) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        # Prepare label text
        label_parts = []
        if labels and i < len(labels):
            label_parts.append(labels[i])
        if confidences and i < len(confidences):
            label_parts.append(f"{confidences[i]:.2f}")

        if label_parts:
            label = " ".join(label_parts)

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                output,
                (x, y - text_height - baseline - 5),
                (x + text_width, y),
                color,
                -1  # Filled
            )

            # Draw text
            cv2.putText(
                output,
                label,
                (x, y - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                thickness - 1,
                cv2.LINE_AA
            )

    return output


def resize_image(
    image: np.ndarray,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize image while optionally maintaining aspect ratio.

    Args:
        image: Input image
        max_width: Maximum width (None = no limit)
        max_height: Maximum height (None = no limit)
        maintain_aspect: Maintain aspect ratio

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    if max_width is None and max_height is None:
        return image

    if maintain_aspect:
        # Calculate scale factor
        scale = 1.0
        if max_width and width > max_width:
            scale = min(scale, max_width / width)
        if max_height and height > max_height:
            scale = min(scale, max_height / height)

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        new_width = max_width or width
        new_height = max_height or height
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image
