"""
Unit Tests for YOLOv8 Model Initialization, Device Mapping, and Metadata
"""

import os
import pytest
import torch
from ultralytics import YOLO
from yolov8.src.detect import WildlifeDetector, DEFAULT_YOLOV8S_CHECKPOINT
from yolov8.src.utils import SUPPORTED_SPECIES


def test_yolov8s_model_loading_official():
    """Verify official YOLOv8s pretrained weights load without error."""
    model_path = "yolov8s.pt"
    if not os.path.exists(model_path):
        pytest.skip("yolov8s.pt not found in root directory.")
    model = YOLO(model_path)
    assert model is not None
    assert model.task == "detect"
    assert len(model.names) > 0


def test_yolov8s_baseline_checkpoint_loading():
    """Verify trained baseline YOLOv8s checkpoint loads cleanly."""
    if not os.path.exists(DEFAULT_YOLOV8S_CHECKPOINT):
        pytest.skip(f"Baseline checkpoint not found at {DEFAULT_YOLOV8S_CHECKPOINT}")
        
    detector = WildlifeDetector(
        model_path=DEFAULT_YOLOV8S_CHECKPOINT,
        conf_threshold=0.25,
        iou_threshold=0.60
    )
    assert detector.model is not None
    assert detector.conf_threshold == 0.25
    assert detector.iou_threshold == 0.60
    
    info = detector.get_model_info()
    assert "model_path" in info
    assert "device" in info
    assert "supported_classes" in info
    assert len(info["supported_classes"]) >= 4


def test_wildlife_detector_device_mapping():
    """Verify WildlifeDetector class maps to requested device correctly."""
    # Test CPU device mapping
    cpu_detector = WildlifeDetector(
        model_path=DEFAULT_YOLOV8S_CHECKPOINT if os.path.exists(DEFAULT_YOLOV8S_CHECKPOINT) else "yolov8s.pt",
        device="cpu"
    )
    assert cpu_detector.device == "cpu"

    # Test CUDA device mapping if available
    if torch.cuda.is_available():
        cuda_detector = WildlifeDetector(
            model_path=DEFAULT_YOLOV8S_CHECKPOINT if os.path.exists(DEFAULT_YOLOV8S_CHECKPOINT) else "yolov8s.pt",
            device=0
        )
        assert "0" in str(cuda_detector.device) or "cuda" in str(cuda_detector.device)
