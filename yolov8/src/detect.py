"""
Unified Wildlife Object Detection API
Module: YOLOv8 Wildlife Object Detection Subsystem
Downstream Target: Deep SORT Multi-Object Tracking & ZoeDepth Depth Estimation
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from yolov8.src.utils import format_detection_for_deepsort, validate_detection_structure


class WildlifeDetector:
    """
    Production-grade YOLOv8 Wildlife Detection Engine.
    Exposes unified detect() API returning structured detection dictionaries
    strictly conforming to the downstream Deep SORT interface contract.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[Union[str, int]] = None,
        imgsz: int = 640
    ):
        """
        Initialize the detector and load weights once onto the target device.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)

        # Device determination
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        # Load model once
        self.model = YOLO(model_path)
        self.class_names = self.model.names if hasattr(self.model, "names") else {}

        # Warm up GPU
        self._warmup()

    def _warmup(self, num_runs: int = 3):
        """Run dummy inference to initialize CUDA kernels and eliminate startup latency."""
        if "cuda" in str(self.device) and torch.cuda.is_available():
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(num_runs):
                _ = self.model.predict(
                    source=dummy,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    imgsz=self.imgsz,
                    verbose=False
                )
            torch.cuda.synchronize()

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute wildlife object detection on a single frame.

        Args:
            frame: Input image as 2D/3D numpy array (BGR format).
            conf: Optional runtime confidence threshold override.
            iou: Optional runtime NMS IoU threshold override.

        Returns:
            List of structured detection dictionaries:
            [
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2]  # pixel coordinates
                }
            ]
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Input frame must be a numpy.ndarray, got {type(frame).__name__}")
        if frame.size == 0 or len(frame.shape) < 2:
            raise ValueError(f"Invalid frame dimensions: {frame.shape}")

        h_frame, w_frame = frame.shape[:2]
        if h_frame < 10 or w_frame < 10:
            raise ValueError(f"Frame resolution too small: {w_frame}x{h_frame}")

        conf_thresh = conf if conf is not None else self.conf_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold

        # Execute prediction
        results = self.model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
        confs = boxes.conf.cpu().numpy()  # [N]
        clss = boxes.cls.cpu().numpy()    # [N]

        for i in range(len(xyxy)):
            box = xyxy[i]
            x1 = int(round(max(0, min(float(box[0]), w_frame - 1))))
            y1 = int(round(max(0, min(float(box[1]), h_frame - 1))))
            x2 = int(round(max(x1 + 1, min(float(box[2]), w_frame))))
            y2 = int(round(max(y1 + 1, min(float(box[3]), h_frame))))

            cls_id = int(clss[i])
            cls_name = self.class_names.get(cls_id, f"wildlife_{cls_id}")
            score = float(confs[i])

            det_dict = format_detection_for_deepsort(
                class_id=cls_id,
                class_name=cls_name,
                confidence=score,
                bbox=(x1, y1, x2, y2)
            )

            if validate_detection_structure(det_dict, (h_frame, w_frame)):
                detections.append(det_dict)

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Helper to render bounding boxes and class names onto a frame for visualization."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - h - 4)), (x1 + w + 4, max(0, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(h + 2, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        return annotated
