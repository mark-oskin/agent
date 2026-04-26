#!/usr/bin/env python3
"""Test script for repo_dependency_analyzer."""

import shutil
import sys
from pathlib import Path

# Add current directory to path to import the analyzer
sys.path.insert(0, str(Path(__file__).resolve().parent))

import repo_dependency_analyzer as analyzer


def create_mock_repo(base_dir: Path) -> None:
    """Create a mock repository with sample Python files."""
    src_dir = base_dir / "src"
    src_dir.mkdir(parents=True)

    # File 1: Imports requests and os
    (src_dir / "main.py").write_text(
        """import requests
import os

import numpy as np

from collections import OrderedDict

def fetch_data():
    return requests.get('http://example.com')
"""
    )

    # File 2: Imports PIL (Pillow) and cv2
    (src_dir / "image_processor.py").write_text(
        """from PIL import Image
import cv2

import json

def process_image(path):
    img = Image.open(path)
    return img
"""
    )

    return src_dir


def test_mock_repo():
    """Test the analyzer against the mock repository."""
    # Create a temporary directory
    tmp_dir = Path("/tmp/test_mock_repo")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    try:
        src_dir = create_mock_repo(tmp_dir)

        # Run the analyzer
        packages, constraints = analyzer.analyze_repo(src_dir)

        print(f"Found packages: {list(packages.keys())}")
        print(f"Constraints: {constraints}")

        # Check for expected packages
        expected_packages = ["requests", "numpy", "pillow", "opencv-python"]
        found_packages = set(packages.keys())

        missing = [pkg for pkg in expected_packages if pkg not in found_packages]
        if missing:
            print(f"ERROR: Missing expected packages: {missing}")
            return False

        print("SUCCESS: All expected packages found!")
        return True

    finally:
        # Clean up
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    success = test_mock_repo()
    sys.exit(0 if success else 1)
