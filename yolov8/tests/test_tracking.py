import numpy as np

from yolov8.src.pipeline import WildlifeTrackingPipeline
from yolov8.src.tracking import ByteTrackTracker


def detection(bbox, confidence=0.9, class_id=1):
    return {
        "class_id": class_id,
        "class_name": "elephant",
        "confidence": confidence,
        "bbox": bbox,
    }


def test_tracker_preserves_contract_and_adds_metadata():
    tracker = ByteTrackTracker()
    result = tracker.update([detection([10, 10, 50, 50])], frame_id=7, timestamp=12.5)

    assert result == [{
        "class_id": 1,
        "class_name": "elephant",
        "confidence": 0.9,
        "bbox": [10, 10, 50, 50],
        "track_id": 1,
        "frame_id": 7,
        "timestamp": 12.5,
    }]


def test_tracker_keeps_id_for_matching_detection():
    tracker = ByteTrackTracker(iou_threshold=0.3)
    first = tracker.update([detection([10, 10, 50, 50])], timestamp=1.0)
    second = tracker.update([detection([12, 12, 52, 52])], timestamp=2.0)

    assert first[0]["track_id"] == second[0]["track_id"]
    assert second[0]["frame_id"] == 2


def test_tracker_creates_new_id_after_expiry():
    tracker = ByteTrackTracker(max_age=1)
    first = tracker.update([detection([10, 10, 50, 50])])
    tracker.update([])
    tracker.update([])
    new = tracker.update([detection([10, 10, 50, 50])])

    assert new[0]["track_id"] != first[0]["track_id"]


def test_pipeline_passes_frame_to_detector_and_tracks_output():
    class FakeDetector:
        def detect(self, frame):
            assert frame.shape == (20, 20, 3)
            return [detection([1, 1, 10, 10])]

    pipeline = WildlifeTrackingPipeline(FakeDetector())
    result = pipeline.process_frame(np.zeros((20, 20, 3), dtype=np.uint8), frame_id=3, timestamp=4.0)

    assert result[0]["track_id"] == 1
    assert result[0]["frame_id"] == 3
    assert result[0]["timestamp"] == 4.0