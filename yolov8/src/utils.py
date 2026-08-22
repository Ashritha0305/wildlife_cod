"""
Utility functions for YOLOv8 Wildlife Detection Subsystem
Includes coordinate transformations, bounding box validation & clipping,
downstream Deep SORT interface structuring, and contract verification.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import os
import cv2
import numpy as np


SUPPORTED_SPECIES: Dict[int, str] = {
    0: "buffalo",
    1: "elephant",
    2: "rhino",
    3: "zebra"
}


def validate_image_frame(frame: Any) -> Tuple[int, int]:
    """
    Validate that an input frame is a non-empty, valid numpy array image.
    
    Args:
        frame: Input object to validate.
        
    Returns:
        (height, width) of the validated image.
        
    Raises:
        ValueError: If frame is None, empty, or below minimum dimensions.
        TypeError: If frame is not a numpy.ndarray.
    """
    if frame is None:
        raise ValueError("Input frame cannot be None.")
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Input frame must be a numpy.ndarray, got {type(frame).__name__}")
    if frame.size == 0 or len(frame.shape) < 2:
        raise ValueError(f"Invalid frame dimensions or empty frame: {frame.shape}")
        
    h, w = frame.shape[:2]
    if h < 10 or w < 10:
        raise ValueError(f"Frame resolution too small: {w}x{h} (minimum 10x10 required)")
        
    return (h, w)


def clip_bbox(
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Clamp and integerize bounding box coordinates strictly within image boundaries.
    
    Args:
        bbox: (x1, y1, x2, y2) in pixel space.
        frame_shape: (height, width) of the image frame.
        
    Returns:
        (x1, y1, x2, y2) integers satisfying:
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height.
    """
    h_frame, w_frame = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    
    x1_clamped = int(round(max(0.0, min(float(x1), float(w_frame - 1)))))
    y1_clamped = int(round(max(0.0, min(float(y1), float(h_frame - 1)))))
    x2_clamped = int(round(max(float(x1_clamped + 1), min(float(x2), float(w_frame)))))
    y2_clamped = int(round(max(float(y1_clamped + 1), min(float(y2), float(h_frame)))))
    
    return (x1_clamped, y1_clamped, x2_clamped, y2_clamped)


def xyxy_to_yolo(
    box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert absolute pixel coordinates [x1, y1, x2, y2] to normalized YOLO [x_center, y_center, width, height].
    """
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(img_width)))
    y1 = max(0.0, min(float(y1), float(img_height)))
    x2 = max(0.0, min(float(x2), float(img_width)))
    y2 = max(0.0, min(float(y2), float(img_height)))
    
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    x_center = x1 + (box_w / 2.0)
    y_center = y1 + (box_h / 2.0)
    
    return (
        x_center / max(float(img_width), 1.0),
        y_center / max(float(img_height), 1.0),
        box_w / max(float(img_width), 1.0),
        box_h / max(float(img_height), 1.0)
    )


def yolo_to_xyxy(
    yolo_box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert normalized YOLO [x_center, y_center, width, height] to absolute pixel [x1, y1, x2, y2].
    """
    x_c, y_c, w, h = yolo_box
    x1 = int(round((x_c - (w / 2.0)) * img_width))
    y1 = int(round((y_c - (h / 2.0)) * img_height))
    x2 = int(round((x_c + (w / 2.0)) * img_width))
    y2 = int(round((y_c + (h / 2.0)) * img_height))
    
    return clip_bbox((x1, y1, x2, y2), (img_height, img_width))


def coco_to_yolo(
    coco_box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """Convert COCO bounding box [xmin, ymin, width, height] in pixels to normalized YOLO format."""
    xmin, ymin, w, h = coco_box
    xmax = xmin + w
    ymax = ymin + h
    return xyxy_to_yolo((xmin, ymin, xmax, ymax), img_width, img_height)


def format_detection_for_deepsort(
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: Tuple[int, int, int, int]
) -> Dict[str, Any]:
    """
    Construct standardized dictionary conforming strictly to the downstream Deep SORT interface contract.
    
    Contract Schema:
    {
        "class_id": int,
        "class_name": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2]
    }
    """
    x1, y1, x2, y2 = bbox
    return {
        "class_id": int(class_id),
        "class_name": str(class_name),
        "confidence": float(round(confidence, 4)),
        "bbox": [int(x1), int(y1), int(x2), int(y2)]
    }


def validate_detection_structure(detection: Dict[str, Any], frame_shape: Tuple[int, int]) -> bool:
    """
    Validate that a single detection dictionary conforms strictly to the Deep SORT interface contract.
    """
    if not isinstance(detection, dict):
        return False
        
    required_keys = {"class_id", "class_name", "confidence", "bbox"}
    if not required_keys.issubset(detection.keys()):
        return False
        
    if not isinstance(detection["class_id"], int) or detection["class_id"] < 0:
        return False
        
    if not isinstance(detection["class_name"], str) or len(detection["class_name"]) == 0:
        return False
        
    conf = detection["confidence"]
    if not isinstance(conf, (float, int)) or conf < 0.0 or conf > 1.0:
        return False
        
    bbox = detection["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
        
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame_shape[:2]
    
    if not all(isinstance(c, (int, np.integer)) for c in [x1, y1, x2, y2]):
        return False
        
    if x1 >= x2 or y1 >= y2:
        return False
        
    if x1 < 0 or y1 < 0 or x2 > w_frame or y2 > h_frame:
        return False
        
    return True


def convert_to_deepsort_input_format(
    detections: List[Dict[str, Any]],
    bbox_format: str = "xyxy"
) -> List[Tuple[List[int], float, str]]:
    """
    Adapter demonstrating downstream consumption by Deep SORT tracker.
    
    Args:
        detections: List of structured detection dicts from WildlifeDetector.
        bbox_format: 'xyxy' for [x1, y1, x2, y2] or 'tlwh' for [top_left_x, top_left_y, width, height].
        
    Returns:
        List of tuples: (bbox, confidence, class_name) ready for Deep SORT Tracker update.
    """
    deepsort_inputs = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        if bbox_format == "tlwh":
            box = [x1, y1, x2 - x1, y2 - y1]
        else:
            box = [x1, y1, x2, y2]
        deepsort_inputs.append((box, d["confidence"], d["class_name"]))
    return deepsort_inputs
