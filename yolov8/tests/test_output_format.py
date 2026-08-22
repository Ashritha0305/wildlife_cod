"""
Unit Tests for Deep SORT Interface Contract Conformance & Format Conversion
"""

import pytest
import numpy as np
from yolov8.src.utils import (
    format_detection_for_deepsort,
    validate_detection_structure,
    convert_to_deepsort_input_format,
    clip_bbox,
    SUPPORTED_SPECIES
)


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
    """Test validation logic against valid structures and edge cases."""
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

    # Out-of-bounds confidence (> 1.0)
    invalid_conf_high = valid_det.copy()
    invalid_conf_high["confidence"] = 1.05
    assert validate_detection_structure(invalid_conf_high, (640, 640)) is False
    
    # Missing key
    invalid_key = {"class_id": 0, "bbox": [10, 20, 200, 300]}
    assert validate_detection_structure(invalid_key, (640, 640)) is False

    # Non-integer coordinates
    invalid_coord_type = valid_det.copy()
    invalid_coord_type["bbox"] = [10.5, 20.0, 200.0, 300.0]
    assert validate_detection_structure(invalid_coord_type, (640, 640)) is False


def test_bbox_clipping_utility():
    """Verify clip_bbox clamps out-of-boundary predictions safely."""
    shape = (480, 640)
    # Box extending beyond frame edges
    raw_box = (-15.0, -10.0, 700.0, 500.0)
    clipped = clip_bbox(raw_box, shape)
    
    assert clipped[0] >= 0
    assert clipped[1] >= 0
    assert clipped[2] <= 640
    assert clipped[3] <= 480
    assert clipped[0] < clipped[2]
    assert clipped[1] < clipped[3]


def test_deep_sort_adapter_converter():
    """Verify convert_to_deepsort_input_format adapts detections for tracker ingest."""
    detections = [
        {"class_id": 0, "class_name": "buffalo", "confidence": 0.95, "bbox": [100, 150, 300, 400]},
        {"class_id": 3, "class_name": "zebra", "confidence": 0.88, "bbox": [350, 200, 550, 450]}
    ]
    
    # Test xyxy mode
    xyxy_inputs = convert_to_deepsort_input_format(detections, bbox_format="xyxy")
    assert len(xyxy_inputs) == 2
    assert xyxy_inputs[0] == ([100, 150, 300, 400], 0.95, "buffalo")
    assert xyxy_inputs[1] == ([350, 200, 550, 450], 0.88, "zebra")
    
    # Test tlwh mode (top-left x, top-left y, width, height)
    tlwh_inputs = convert_to_deepsort_input_format(detections, bbox_format="tlwh")
    assert len(tlwh_inputs) == 2
    assert tlwh_inputs[0] == ([100, 150, 200, 250], 0.95, "buffalo")
    assert tlwh_inputs[1] == ([350, 200, 200, 250], 0.88, "zebra")
