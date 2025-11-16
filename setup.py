"""
Setup configuration for opencv-car-detection package.
Computer Vision Evolution: Classical to Modern Object Detection
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="opencv-car-detection",
    version="2.0.0",
    author="Van-Duyet Le",
    author_email="lvduit08@gmail.com",
    description="Computer Vision Evolution: Compare classical Haar Cascades with modern deep learning for vehicle detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duyet/opencv-car-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0,<5.0.0",
        "opencv-contrib-python>=4.8.0,<5.0.0",
        "ultralytics>=8.0.0,<9.0.0",
        "torch>=2.0.0,<3.0.0",
        "torchvision>=0.15.0,<1.0.0",
        "numpy>=1.24.0,<2.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "Pillow>=10.0.0,<11.0.0",
        "Flask>=3.0.0,<4.0.0",
        "Flask-CORS>=4.0.0,<5.0.0",
        "Flask-Limiter>=3.5.0,<4.0.0",
        "click>=8.1.0,<9.0.0",
        "tqdm>=4.66.0,<5.0.0",
        "colorama>=0.4.6,<1.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "PyYAML>=6.0.0,<7.0.0",
        "loguru>=0.7.0,<1.0.0",
        "pydantic>=2.0.0,<3.0.0",
        "joblib>=1.3.0,<2.0.0",
        "imageio>=2.31.0,<3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "car-detect=cli:main",
            "car-detect-web=web.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.xml", "*.yaml", "*.yml"],
    },
    zip_safe=False,
)
