"""
Unit Tests for Detection Execution and Coordinate Bounds
"""

import os
import glob
import cv2
import pytest
import numpy as np
from yolov8.src.detect import WildlifeDetector


@pytest.fixture
def detector():
    return WildlifeDetector(model_path="yolov8n.pt", conf_threshold=0.25)


def test_detection_on_synthetic_frame(detector):
    """Verify detection runs on valid numpy array without error."""
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
    
    dets = detector.detect(frame)
    assert isinstance(dets, list)


def test_detection_on_real_image(detector):
    """Verify detection runs on real dataset image if available."""
    test_imgs = glob.glob(r"C:\wildlife_COD\yolov8\data\baseline\test\images\*.*")
    if not test_imgs:
        pytest.skip("No dataset images available for real detection test.")
        
    sample_img = test_imgs[0]
    frame = cv2.imread(sample_img)
    assert frame is not None
    
    dets = detector.detect(frame)
    assert isinstance(dets, list)
    
    h, w = frame.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
        assert 0.0 <= d["confidence"] <= 1.0
        assert isinstance(d["class_id"], int)
        assert isinstance(d["class_name"], str)
