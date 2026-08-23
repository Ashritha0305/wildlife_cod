"""ByteTrack-style multi-object tracking for YOLO detection dictionaries."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time


Detection = Dict[str, Any]


def _iou(first: List[int], second: Tuple[int, int, int, int]) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    missed_frames: int = 0


class ByteTrackTracker:
    """Track YOLO detections with high/low-confidence IoU association."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        high_confidence: float = 0.5,
        low_confidence: float = 0.1,
        max_age: int = 30,
        start_frame_id: int = 0,
    ) -> None:
        if not 0.0 <= low_confidence <= high_confidence <= 1.0:
            raise ValueError("confidence thresholds must satisfy 0 <= low <= high <= 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        self.iou_threshold = float(iou_threshold)
        self.high_confidence = float(high_confidence)
        self.low_confidence = float(low_confidence)
        self.max_age = int(max_age)
        self._next_track_id = 1
        self._frame_id = int(start_frame_id)
        self._tracks: Dict[int, _Track] = {}

    @property
    def active_track_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._next_track_id = 1
        self._frame_id = 0
        self._tracks.clear()

    def update(
        self,
        detections: List[Detection],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[Detection]:
        """Assign track IDs and frame metadata to detections for one frame."""
        if frame_id is None:
            self._frame_id += 1
            current_frame_id = self._frame_id
        else:
            current_frame_id = int(frame_id)
            self._frame_id = current_frame_id
        current_timestamp = time.time() if timestamp is None else float(timestamp)

        candidates = [
            (index, detection)
            for index, detection in enumerate(detections)
            if self.low_confidence <= float(detection["confidence"]) <= 1.0
        ]
        high = [(index, detection) for index, detection in candidates
                if float(detection["confidence"]) >= self.high_confidence]
        low = [(index, detection) for index, detection in candidates
               if float(detection["confidence"]) < self.high_confidence]
        assignments: Dict[int, int] = {}
        unmatched_tracks = set(self._tracks)
        unmatched_detections = {index for index, _ in candidates}

        for group in (high, low):
            matches = sorted(
                (
                    _iou(self._tracks[track_id].bbox, tuple(detection["bbox"])),
                    track_id,
                    index,
                )
                for track_id in unmatched_tracks
                for index, detection in group
                if detection["class_id"] == self._tracks[track_id].class_id
            )
            used_tracks = set()
            used_detections = set()
            for score, track_id, index in reversed(matches):
                if score < self.iou_threshold or track_id in used_tracks or index in used_detections:
                    continue
                assignments[index] = track_id
                used_tracks.add(track_id)
                used_detections.add(index)
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(index)

        for track_id in unmatched_tracks:
            self._tracks[track_id].missed_frames += 1
        for track_id in [track_id for track_id, track in self._tracks.items()
                         if track.missed_frames > self.max_age]:
            del self._tracks[track_id]

        tracked_detections: List[Detection] = []
        for index, detection in candidates:
            track_id = assignments.get(index)
            if track_id is None:
                if float(detection["confidence"]) < self.high_confidence:
                    continue
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    bbox=tuple(detection["bbox"]),
                    class_id=int(detection["class_id"]),
                )
            else:
                track = self._tracks[track_id]
                track.bbox = tuple(detection["bbox"])
                track.missed_frames = 0
            enriched = dict(detection)
            enriched["track_id"] = track_id
            enriched["frame_id"] = current_frame_id
            enriched["timestamp"] = current_timestamp
            tracked_detections.append(enriched)
        return tracked_detections