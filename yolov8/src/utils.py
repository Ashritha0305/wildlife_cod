"""
Utility functions for YOLOv8 Wildlife Detection Subsystem
Includes coordinate transformations, bounding box validation,
downstream interface structuring, and annotation conversion tools.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np


def xyxy_to_yolo(
    box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert absolute pixel coordinates [x1, y1, x2, y2] to normalized YOLO [x_center, y_center, width, height].
    
    Args:
        box: (x1, y1, x2, y2) in pixel space.
        img_width: Width of image in pixels.
        img_height: Height of image in pixels.
        
    Returns:
        (x_center, y_center, width, height) normalized to [0.0, 1.0].
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
        x_center / img_width,
        y_center / img_height,
        box_w / img_width,
        box_h / img_height
    )


def yolo_to_xyxy(
    yolo_box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert normalized YOLO [x_center, y_center, width, height] to absolute pixel [x1, y1, x2, y2].
    
    Args:
        yolo_box: (x_center, y_center, width, height) in [0.0, 1.0].
        img_width: Width of image in pixels.
        img_height: Height of image in pixels.
        
    Returns:
        (x1, y1, x2, y2) integers in pixel space clamped to frame bounds.
    """
    x_c, y_c, w, h = yolo_box
    x1 = int(round((x_c - (w / 2.0)) * img_width))
    y1 = int(round((y_c - (h / 2.0)) * img_height))
    x2 = int(round((x_c + (w / 2.0)) * img_width))
    y2 = int(round((y_c + (h / 2.0)) * img_height))
    
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    return (x1, y1, x2, y2)


def coco_to_yolo(
    coco_box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert COCO bounding box [xmin, ymin, width, height] in pixels to normalized YOLO format.
    """
    xmin, ymin, w, h = coco_box
    xmax = xmin + w
    ymax = ymin + h
    return xyxy_to_yolo((xmin, ymin, xmax, ymax), img_width, img_height)


def mask_to_yolo_bbox(
    binary_mask: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """
    Derive normalized YOLO bounding box [x_center, y_center, width, height] from binary segmentation mask.
    
    Args:
        binary_mask: 2D numpy array (uint8 or bool) where foreground > 0.
        
    Returns:
        (x_center, y_center, width, height) normalized, or None if mask is empty.
    """
    if binary_mask is None or not np.any(binary_mask > 0):
        return None
        
    h_img, w_img = binary_mask.shape[:2]
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # xmax, ymax inclusive boundary adjustment
    return xyxy_to_yolo((float(xmin), float(ymin), float(xmax + 1), float(ymax + 1)), w_img, h_img)


def format_detection_for_deepsort(
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: Tuple[int, int, int, int]
) -> Dict[str, Any]:
    """
    Construct standardized dictionary for downstream Deep SORT integration.
    
    Contract:
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
    
    if x1 >= x2 or y1 >= y2:
        return False
        
    if x1 < 0 or y1 < 0 or x2 > w_frame or y2 > h_frame:
        return False
        
    return True
