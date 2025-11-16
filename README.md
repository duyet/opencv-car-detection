# 🚗 OpenCV Car Detection - Computer Vision Evolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Experience the evolution of computer vision** by comparing classical Haar Cascades (2001) with modern YOLOv8 (2023) for vehicle detection.

## 🌟 What Makes This Special

This isn't just another object detection demo—it's a **time machine** through 22 years of computer vision progress:

- 🏛️ **Classical (2001)**: Haar Cascades using hand-crafted features
- 🚀 **Modern (2023)**: YOLOv8 with state-of-the-art deep learning  
- 📊 **Side-by-Side**: Beautiful visual comparisons
- 🎯 **Production-Ready**: Clean architecture, security, tests
- 🌐 **Web Interface**: Gorgeous browser-based UI
- 🖥️ **CLI Tools**: Powerful command-line interface
- 📹 **Video Support**: Real-time webcam & video file detection

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/duyet/opencv-car-detection.git
cd opencv-car-detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Web Interface (Recommended)

```bash
# Start the web server
python web/app.py
# Open browser to http://localhost:5000
```

#### Command Line

```bash
# Compare both detectors
python src/cli.py detect data/samples/car1.jpg

# Use only YOLO
python src/cli.py detect car.jpg --detector yolo

# Real-time webcam
python src/cli.py webcam
```

## 📊 Performance Comparison

| Method | Year | Approach | Speed (CPU) | Accuracy |
|--------|------|----------|-------------|----------|
| **Haar Cascade** | 2001 | Hand-crafted | ⚡⚡⚡ 45 FPS | ⭐⭐ Moderate |
| **YOLOv8** | 2023 | Deep Learning | ⚡ 5 FPS (CPU) / ⚡⚡⚡ 120 FPS (GPU) | ⭐⭐⭐⭐⭐ SOTA |

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo if you find it useful!** ⭐
