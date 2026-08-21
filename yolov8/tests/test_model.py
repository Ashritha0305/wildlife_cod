"""
Unit Tests for YOLOv8 Model Initialization and Architecture
"""

import os
import pytest
import torch
from ultralytics import YOLO
from yolov8.src.detect import WildlifeDetector


def test_yolov8_model_loading():
    """Verify official YOLOv8 model initializes cleanly."""
    model = YOLO("yolov8n.pt")
    assert model is not None
    assert model.task == "detect"
    assert len(model.names) > 0


def test_wildlife_detector_initialization():
    """Verify WildlifeDetector class loads model and configures target device."""
    detector = WildlifeDetector(
        model_path="yolov8n.pt",
        conf_threshold=0.35,
        iou_threshold=0.45
    )
    assert detector.model is not None
    assert detector.conf_threshold == 0.35
    assert detector.iou_threshold == 0.45
    if torch.cuda.is_available():
        assert "cuda" in str(detector.device)
    else:
        assert detector.device == "cpu"
