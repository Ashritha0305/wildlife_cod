"""
Unit Tests for Robust Error Handling and Input Boundary Checking
"""

import pytest
import numpy as np
from yolov8.src.detect import WildlifeDetector


@pytest.fixture
def detector():
    return WildlifeDetector(model_path="yolov8n.pt", conf_threshold=0.25)


def test_missing_model_file():
    """Verify detector raises FileNotFoundError on missing weights path."""
    with pytest.raises(FileNotFoundError):
        _ = WildlifeDetector(model_path="non_existent_weights_12345.pt")


def test_none_frame_input(detector):
    """Verify detector raises ValueError on None input."""
    with pytest.raises(ValueError):
        _ = detector.detect(None)


def test_invalid_type_input(detector):
    """Verify detector raises TypeError on non-numpy input."""
    with pytest.raises(TypeError):
        _ = detector.detect("not_an_image_array")


def test_zero_dimension_frame(detector):
    """Verify detector handles empty zero-dimension arrays cleanly."""
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = detector.detect(empty_frame)


def test_sub_resolution_frame(detector):
    """Verify detector rejects extremely small non-image arrays."""
    tiny_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = detector.detect(tiny_frame)
