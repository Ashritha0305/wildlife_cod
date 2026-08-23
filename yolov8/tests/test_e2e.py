import numpy as np

from yolov8.src.e2e import EndToEndInference
from yolov8.src.threat import ThreatAnalyzer
from yolov8.src.tracking import ByteTrackTracker


class FakeDetector:
    def detect(self, frame):
        return [{
            "class_id": 1,
            "class_name": "elephant",
            "confidence": 0.9,
            "bbox": [2, 2, 8, 8],
        }]


class FakeDepthEstimator:
    def estimate(self, frame):
        depth = np.full(frame.shape[:2], 25.0, dtype=np.float32)
        depth[2:8, 2:8] = 5.0
        return depth


def make_inference():
    return EndToEndInference(
        detector=FakeDetector(),
        tracker=ByteTrackTracker(),
        depth_estimator=FakeDepthEstimator(),
        threat_analyzer=ThreatAnalyzer(),
    )


def test_end_to_end_returns_requested_final_schema():
    results = make_inference().process_frame(
        np.zeros((10, 10, 3), dtype=np.uint8), frame_id=4, timestamp=8.0
    )

    assert results == [{
        "track_id": 1,
        "class_name": "elephant",
        "confidence": 0.9,
        "bbox": [2, 2, 8, 8],
        "distance": 5.0,
        "threat_level": "CRITICAL",
        "threat_score": 0.9,
        "reason": "elephant is within the critical distance threshold.",
    }]


def test_rendering_returns_annotated_copy():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    results = make_inference().process_frame(frame)
    rendered = make_inference().render_results(frame, results)

    assert rendered.shape == frame.shape
    assert not np.array_equal(rendered, frame)


def test_invalid_depth_values_produce_safe_result():
    class InvalidDepth:
        def estimate(self, frame):
            return np.full(frame.shape[:2], np.nan, dtype=np.float32)

    inference = EndToEndInference(
        FakeDetector(), ByteTrackTracker(), InvalidDepth(), ThreatAnalyzer()
    )
    result = inference.process_frame(np.zeros((10, 10, 3), dtype=np.uint8))[0]

    assert result["distance"] is None
    assert result["threat_level"] == "SAFE"
    assert result["threat_score"] == 0.0