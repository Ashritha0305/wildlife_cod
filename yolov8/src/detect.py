"""
Production Wildlife Object Detection API
Module: YOLOv8 Wildlife Object Detection Subsystem
Downstream Target: Deep SORT Multi-Object Tracking & ZoeDepth Depth Estimation

Selected Baseline Architecture: YOLOv8s (Small)
Supported Classes:
  0: buffalo
  1: elephant
  2: rhino
  3: zebra
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from yolov8.src.utils import (
    validate_image_frame,
    clip_bbox,
    format_detection_for_deepsort,
    validate_detection_structure,
    SUPPORTED_SPECIES
)

# Standard default checkpoint location for production YOLOv8s baseline model
DEFAULT_YOLOV8S_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "baseline_yolov8s",
    "weights",
    "best.pt"
)


class WildlifeDetector:
    """
    Production-grade YOLOv8 Wildlife Detection Engine.
    Exposes unified detect(), detect_image(), detect_video_frame(), and predict() APIs
    returning deterministic structured detection dictionaries strictly conforming
    to the downstream Deep SORT interface contract.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.60,
        device: Optional[Union[str, int]] = None,
        imgsz: int = 640
    ):
        """
        Initialize the detector and load weights once onto the target device.
        
        Args:
            model_path: Path to YOLOv8 weights (.pt). Defaults to baseline_yolov8s/weights/best.pt.
            conf_threshold: Default confidence threshold for object filtering (default 0.25).
            iou_threshold: Default IoU threshold for Non-Maximum Suppression (default 0.60).
            device: Target device ('cuda:0', 0, 'cpu', etc.). Auto-resolves if None.
            imgsz: Input resolution square size (default 640).
        """
        # Resolve model path
        if model_path is None:
            if os.path.exists(DEFAULT_YOLOV8S_CHECKPOINT):
                model_path = DEFAULT_YOLOV8S_CHECKPOINT
            elif os.path.exists("yolov8s.pt"):
                model_path = "yolov8s.pt"
            else:
                model_path = "yolov8n.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        self.model_path = os.path.abspath(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)

        # Device determination
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        # Load model once
        self.model = YOLO(self.model_path)
        
        # Load class mappings
        if hasattr(self.model, "names") and isinstance(self.model.names, dict) and len(self.model.names) > 0:
            self.class_names = self.model.names
        else:
            self.class_names = SUPPORTED_SPECIES

        # Warm up GPU to eliminate cold-start CUDA initialization latency
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
        Execute wildlife object detection on a single frame numpy array.

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
                    "bbox": [x1, y1, x2, y2]  # absolute pixel coordinates clamped to frame
                }
            ]
        """
        # Validate input frame
        h_frame, w_frame = validate_image_frame(frame)

        conf_thresh = float(conf) if conf is not None else self.conf_threshold
        iou_thresh = float(iou) if iou is not None else self.iou_threshold

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
            x1, y1, x2, y2 = clip_bbox(
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                (h_frame, w_frame)
            )

            cls_id = int(clss[i])
            cls_name = self.class_names.get(cls_id, SUPPORTED_SPECIES.get(cls_id, f"wildlife_{cls_id}"))
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

    def detect_image(
        self,
        image_input: Union[str, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper for single image input (file path or numpy array).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path not found: {image_input}")
            frame = cv2.imread(image_input)
            if frame is None:
                raise ValueError(f"Could not decode image at: {image_input}")
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            raise TypeError(f"image_input must be a filepath string or numpy.ndarray, got {type(image_input).__name__}")

        return self.detect(frame, conf=conf, iou=iou)

    def detect_video_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper for real-time video frame inference.
        """
        return self.detect(frame, conf=conf, iou=iou)

    def predict(
        self,
        source: Union[str, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Unified alias matching standard predictor interface.
        """
        return self.detect_image(source, conf=conf, iou=iou)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Helper to render bounding boxes and class labels onto a frame for visualization."""
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid frame provided to draw_detections")

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

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and runtime configuration."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "supported_classes": self.class_names
        }
