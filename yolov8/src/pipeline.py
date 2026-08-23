"""Orchestration boundary between YOLO inference and downstream modules."""

from typing import Any, Dict, Iterable, List, Optional, Protocol
import numpy as np

from yolov8.src.threat import ThreatAnalyzer
from yolov8.src.tracking import ByteTrackTracker


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        ...


class WildlifeTrackingPipeline:
    """Run unchanged YOLO inference and add tracking metadata at the boundary."""

    def __init__(
        self,
        detector: Detector,
        tracker: Optional[ByteTrackTracker] = None,
        threat_analyzer: Optional[ThreatAnalyzer] = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or ByteTrackTracker()
        self.threat_analyzer = threat_analyzer

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        distances: Optional[Iterable[Any]] = None,
    ) -> List[Dict[str, Any]]:
        detections = self.detector.detect(frame)
        tracked = self.tracker.update(detections, frame_id=frame_id, timestamp=timestamp)
        if self.threat_analyzer is None or distances is None:
            return tracked
        threat_results = self.threat_analyzer.analyze_detections(tracked, distances)
        return [dict(detection, **threat_result)
                for detection, threat_result in zip(tracked, threat_results)]