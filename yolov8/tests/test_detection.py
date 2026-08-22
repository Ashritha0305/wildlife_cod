"""
Unit Tests for Detection Execution, Coordinate Bounds, and Multi-Species Detection
"""

import os
import glob
import cv2
import pytest
import numpy as np
from yolov8.src.detect import WildlifeDetector, DEFAULT_YOLOV8S_CHECKPOINT
from yolov8.src.utils import SUPPORTED_SPECIES


@pytest.fixture(scope="module")
def detector():
    model_path = DEFAULT_YOLOV8S_CHECKPOINT if os.path.exists(DEFAULT_YOLOV8S_CHECKPOINT) else "yolov8s.pt"
    return WildlifeDetector(model_path=model_path, conf_threshold=0.25, iou_threshold=0.60)


def test_detection_on_synthetic_frame(detector):
    """Verify detection runs on valid numpy array without error."""
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
    
    dets = detector.detect(frame)
    assert isinstance(dets, list)


def test_detection_on_real_dataset_images(detector):
    """Verify detection runs on real baseline test images and returns valid structured output."""
    test_imgs = glob.glob(r"yolov8/data/baseline/test/images/*.*")
    if not test_imgs:
        pytest.skip("No baseline test images found.")
        
    for sample_img in test_imgs[:5]:
        frame = cv2.imread(sample_img)
        assert frame is not None
        
        h, w = frame.shape[:2]
        dets = detector.detect(frame)
        assert isinstance(dets, list)
        
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            # Strict boundary assertion
            assert 0 <= x1 < x2 <= w, f"Invalid bbox x-coords: [{x1}, {x2}] for width {w}"
            assert 0 <= y1 < y2 <= h, f"Invalid bbox y-coords: [{y1}, {y2}] for height {h}"
            
            # Confidence bounds
            assert 0.0 <= d["confidence"] <= 1.0
            
            # Species & class validity
            assert isinstance(d["class_id"], int)
            assert d["class_id"] in SUPPORTED_SPECIES
            assert isinstance(d["class_name"], str)
            assert d["class_name"] == SUPPORTED_SPECIES[d["class_id"]]


def test_multiple_detections_support(detector):
    """Verify detector handles scenes with multiple animals correctly."""
    multi_img = r"yolov8/data/baseline/test/images/1 (192).jpg"
    if not os.path.exists(multi_img):
        pytest.skip("Multi-animal test image not found.")
        
    frame = cv2.imread(multi_img)
    dets = detector.detect(frame, conf=0.15)
    assert isinstance(dets, list)
    # Should detect multiple animals
    assert len(dets) >= 1
    for d in dets:
        assert isinstance(d["bbox"], list)
        assert len(d["bbox"]) == 4


def test_convenience_methods(detector):
    """Verify detect_image, detect_video_frame, predict, and draw_detections work identically."""
    test_imgs = glob.glob(r"yolov8/data/baseline/test/images/*.*")
    if not test_imgs:
        pytest.skip("No baseline test images found.")
        
    img_path = test_imgs[0]
    frame = cv2.imread(img_path)
    
    dets_frame = detector.detect(frame)
    dets_path = detector.detect_image(img_path)
    dets_pred = detector.predict(frame)
    dets_vid = detector.detect_video_frame(frame)
    
    assert dets_frame == dets_path == dets_pred == dets_vid
    
    # Verify draw_detections
    annotated = detector.draw_detections(frame, dets_frame)
    assert annotated is not None
    assert annotated.shape == frame.shape
