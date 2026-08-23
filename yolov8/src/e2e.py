"""Minimal end-to-end wildlife inference orchestration."""

from typing import Any, Dict, List, Optional, Protocol, Tuple
import os
import time

import cv2
import numpy as np

from yolov8.src.threat import ThreatAnalyzer
from yolov8.src.tracking import ByteTrackTracker


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        ...


class DepthEstimator(Protocol):
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        ...


class EndToEndInference:
    """Compose detection, tracking, depth sampling, and threat analysis."""

    def __init__(
        self,
        detector: Detector,
        tracker: ByteTrackTracker,
        depth_estimator: DepthEstimator,
        threat_analyzer: ThreatAnalyzer,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.depth_estimator = depth_estimator
        self.threat_analyzer = threat_analyzer

    @staticmethod
    def _distance_from_bbox(depth_map: np.ndarray, bbox: List[int]) -> Optional[float]:
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
            raise ValueError("depth estimator must return a 2D numpy array")
        x1, y1, x2, y2 = bbox
        height, width = depth_map.shape
        x1 = max(0, min(int(x1), width))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height))
        y2 = max(0, min(int(y2), height))
        if x1 >= x2 or y1 >= y2:
            return None
        values = depth_map[y1:y2, x1:x2].astype(np.float32).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return float(np.median(values))

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run the complete pipeline and return one final result per track."""
        detections = self.detector.detect(frame)
        tracked = self.tracker.update(detections, frame_id=frame_id, timestamp=timestamp)
        depth_map = self.depth_estimator.estimate(frame)
        results: List[Dict[str, Any]] = []
        for detection in tracked:
            distance = self._distance_from_bbox(depth_map, detection["bbox"])
            threat = self.threat_analyzer.analyze(detection, distance)
            results.append({
                "track_id": detection["track_id"],
                "class_name": threat["class_name"],
                "confidence": threat["confidence"],
                "bbox": detection["bbox"],
                "distance": threat["distance"],
                "threat_level": threat["threat_level"],
                "threat_score": threat["threat_score"],
                "reason": threat["reason"],
            })
        return results

    def process_image(self, image_path: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Load an image from disk and run the same frame pipeline."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not decode image at: {image_path}")
        return self.process_frame(frame, **kwargs)

    @staticmethod
    def render_results(frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Render boxes, distances, and threat levels without changing results."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a color numpy image")
        rendered = frame.copy()
        colors = {"SAFE": (0, 180, 0), "WARNING": (0, 180, 255), "CRITICAL": (0, 0, 255)}
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            level = result["threat_level"]
            color = colors.get(level, (255, 255, 255))
            distance = result["distance"]
            distance_text = "unknown" if distance is None else f"{distance:.2f}"
            label = f"{result['class_name']} {level} {distance_text}"
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            cv2.putText(rendered, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return rendered


def create_default_inference(**kwargs: Any) -> EndToEndInference:
    """Create the production composition; model weights load in constructors."""
    from yolov8.src.depth import DepthEstimator as ZoeDepthEstimator
    from yolov8.src.detect import WildlifeDetector

    return EndToEndInference(
        detector=WildlifeDetector(**kwargs.pop("detector", {})),
        tracker=kwargs.pop("tracker", ByteTrackTracker()),
        depth_estimator=ZoeDepthEstimator(**kwargs.pop("depth", {})),
        threat_analyzer=kwargs.pop("threat", ThreatAnalyzer()),
    )