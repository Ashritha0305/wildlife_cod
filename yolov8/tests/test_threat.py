import numpy as np
import pytest

from yolov8.src.pipeline import WildlifeTrackingPipeline
from yolov8.src.threat import ThreatAnalyzer


def make_detection(class_name="elephant", confidence=0.9):
    return {"class_name": class_name, "confidence": confidence}


def test_safe_warning_and_critical_levels():
    analyzer = ThreatAnalyzer(warning_distance=20, critical_distance=10)

    assert analyzer.analyze(make_detection(), 25)["threat_level"] == "SAFE"
    assert analyzer.analyze(make_detection(), 15)["threat_level"] == "WARNING"
    assert analyzer.analyze(make_detection(), 10)["threat_level"] == "CRITICAL"


def test_result_contains_required_fields_and_score():
    result = ThreatAnalyzer().analyze(make_detection(confidence=0.8), 15)

    assert set(result) == {
        "class_name", "confidence", "distance", "threat_level", "threat_score", "reason"
    }
    assert result["class_name"] == "elephant"
    assert result["confidence"] == 0.8
    assert result["distance"] == 15.0
    assert 0.0 <= result["threat_score"] <= 1.0
    assert result["reason"]


@pytest.mark.parametrize("distance", [None, -1, float("nan"), float("inf"), "near"])
def test_invalid_distance_defaults_safely(distance):
    result = ThreatAnalyzer().analyze(make_detection(), distance)

    assert result["distance"] is None
    assert result["threat_level"] == "SAFE"
    assert result["threat_score"] == 0.0


def test_species_weight_is_configurable():
    result = ThreatAnalyzer(species_weights={"elephant": 2.0}).analyze(
        make_detection(confidence=0.5), 15
    )

    assert result["threat_score"] == 0.5


def test_threshold_validation():
    with pytest.raises(ValueError):
        ThreatAnalyzer(warning_distance=10, critical_distance=10)
    with pytest.raises(ValueError):
        ThreatAnalyzer(warning_distance=float("nan"))


def test_pipeline_adds_threat_fields_without_changing_tracking_metadata():
    class FakeDetector:
        def detect(self, frame):
            return [{
                "class_id": 1,
                "class_name": "elephant",
                "confidence": 0.9,
                "bbox": [1, 1, 10, 10],
            }]

    pipeline = WildlifeTrackingPipeline(
        FakeDetector(), threat_analyzer=ThreatAnalyzer()
    )
    result = pipeline.process_frame(
        np.zeros((20, 20, 3), dtype=np.uint8),
        frame_id=3,
        timestamp=4.0,
        distances=[5.0],
    )[0]

    assert result["track_id"] == 1
    assert result["frame_id"] == 3
    assert result["timestamp"] == 4.0
    assert result["threat_level"] == "CRITICAL"


def test_pipeline_keeps_detections_when_distance_is_missing():
    class FakeDetector:
        def detect(self, frame):
            return [
                {"class_id": 1, "class_name": "elephant", "confidence": 0.9,
                 "bbox": [1, 1, 10, 10]},
                {"class_id": 0, "class_name": "buffalo", "confidence": 0.8,
                 "bbox": [10, 10, 19, 19]},
            ]

    pipeline = WildlifeTrackingPipeline(
        FakeDetector(), threat_analyzer=ThreatAnalyzer()
    )
    results = pipeline.process_frame(
        np.zeros((20, 20, 3), dtype=np.uint8), distances=[5.0]
    )

    assert len(results) == 2
    assert results[1]["distance"] is None
    assert results[1]["threat_level"] == "SAFE"