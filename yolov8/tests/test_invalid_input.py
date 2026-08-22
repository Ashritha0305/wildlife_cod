"""
Unit Tests for Robust Error Handling and Input Boundary Checking
"""

import pytest
import numpy as np
from yolov8.src.detect import WildlifeDetector, DEFAULT_YOLOV8S_CHECKPOINT
from yolov8.src.utils import validate_image_frame


@pytest.fixture(scope="module")
def detector():
    model_path = DEFAULT_YOLOV8S_CHECKPOINT if os.path.exists(DEFAULT_YOLOV8S_CHECKPOINT) else "yolov8s.pt"
    return WildlifeDetector(model_path=model_path, conf_threshold=0.25)


import os


def test_missing_model_file():
    """Verify detector raises FileNotFoundError on missing weights path."""
    with pytest.raises(FileNotFoundError):
        _ = WildlifeDetector(model_path="non_existent_weights_path_99999.pt")


def test_none_frame_input(detector):
    """Verify detector raises ValueError on None input."""
    with pytest.raises(ValueError):
        _ = detector.detect(None)


def test_invalid_type_input(detector):
    """Verify detector raises TypeError on non-numpy input."""
    with pytest.raises(TypeError):
        _ = detector.detect("not_an_image_array")
    with pytest.raises(TypeError):
        _ = detector.detect([1, 2, 3])


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


def test_validate_image_frame_utility():
    """Verify standalone validate_image_frame function."""
    valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = validate_image_frame(valid_frame)
    assert h == 480
    assert w == 640

    with pytest.raises(ValueError):
        validate_image_frame(None)
    with pytest.raises(TypeError):
        validate_image_frame(12345)
