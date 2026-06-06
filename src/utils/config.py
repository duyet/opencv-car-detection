"""
Configuration management for the application.

This module handles loading configuration from YAML files and environment variables.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml
from dotenv import load_dotenv
from loguru import logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file (if None, uses default)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(".config/config.yaml"),
        ]
        for path in default_paths:
            if path.exists():
                config_path = path
                break

    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Return default config if no file found
    logger.info("Using default configuration")
    return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    # Load environment variables
    load_dotenv()

    return {
        "detectors": {
            "haar_cascade": {
                "enabled": True,
                "model_path": "data/models/cars.xml",
                "scale_factor": float(os.getenv("HAAR_SCALE_FACTOR", "1.1")),
                "min_neighbors": int(os.getenv("HAAR_MIN_NEIGHBORS", "3")),
                "min_size": [30, 30],
            },
            "yolo": {
                "enabled": True,
                "model_size": os.getenv("YOLO_MODEL_SIZE", "n"),
                "confidence_threshold": float(os.getenv("YOLO_CONF_THRESHOLD", "0.25")),
                "iou_threshold": float(os.getenv("YOLO_IOU_THRESHOLD", "0.45")),
                "device": os.getenv("YOLO_DEVICE", "cpu"),
            },
        },
        "visualization": {
            "box_thickness": 2,
            "font_scale": 0.5,
            "colors": {
                "haar": [0, 0, 255],  # Red (BGR)
                "yolo": [0, 255, 0],  # Green (BGR)
            },
        },
        "web": {
            "host": os.getenv("WEB_HOST", "0.0.0.0"),
            "port": int(os.getenv("WEB_PORT", "5000")),
            "debug": os.getenv("WEB_DEBUG", "False").lower() == "true",
            "upload_folder": os.getenv("UPLOAD_FOLDER", "data/uploads"),
            "max_file_size": int(os.getenv("MAX_FILE_SIZE", "10485760")),  # 10MB
        },
        "security": {
            "rate_limit": os.getenv("RATE_LIMIT", "10 per minute"),
            "allowed_extensions": [".jpg", ".jpeg", ".png"],
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        },
    }


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise
