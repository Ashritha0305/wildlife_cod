"""
Unit Tests for Deep SORT Interface Contract Conformance
"""

import pytest
import numpy as np
from yolov8.src.utils import format_detection_for_deepsort, validate_detection_structure


def test_deep_sort_format_structure():
    """Verify format_detection_for_deepsort produces strict expected schema."""
    formatted = format_detection_for_deepsort(
        class_id=1,
        class_name="elephant",
        confidence=0.923456,
        bbox=(50, 100, 450, 500)
    )
    
    assert formatted == {
        "class_id": 1,
        "class_name": "elephant",
        "confidence": 0.9235,
        "bbox": [50, 100, 450, 500]
    }


def test_deep_sort_format_validation():
    """Test validation logic against edge cases and invalid coordinates."""
    valid_det = {
        "class_id": 0,
        "class_name": "buffalo",
        "confidence": 0.85,
        "bbox": [10, 20, 200, 300]
    }
    assert validate_detection_structure(valid_det, (640, 640)) is True
    
    # Inverted bounding box (x1 >= x2)
    invalid_box = valid_det.copy()
    invalid_box["bbox"] = [200, 20, 100, 300]
    assert validate_detection_structure(invalid_box, (640, 640)) is False
    
    # Negative confidence
    invalid_conf = valid_det.copy()
    invalid_conf["confidence"] = -0.1
    assert validate_detection_structure(invalid_conf, (640, 640)) is False
    
    # Missing key
    invalid_key = {"class_id": 0, "bbox": [10, 20, 200, 300]}
    assert validate_detection_structure(invalid_key, (640, 640)) is False
