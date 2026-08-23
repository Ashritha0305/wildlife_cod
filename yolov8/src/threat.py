"""Configurable distance-based threat analysis for tracked wildlife."""

from math import isfinite
from itertools import zip_longest
from typing import Any, Dict, Iterable, List, Mapping, Optional


ThreatResult = Dict[str, Any]


class ThreatAnalyzer:
    """Classify wildlife detections as SAFE, WARNING, or CRITICAL.

    Distances use the same unit as the caller, normally metres. A detection at
    or below ``critical_distance`` is CRITICAL; one above that and at or below
    ``warning_distance`` is WARNING; farther detections are SAFE.
    """

    def __init__(
        self,
        warning_distance: float = 20.0,
        critical_distance: float = 10.0,
        species_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not isfinite(warning_distance) or not isfinite(critical_distance):
            raise ValueError("distance thresholds must be finite")
        if critical_distance < 0 or warning_distance <= critical_distance:
            raise ValueError("warning_distance must be greater than critical_distance >= 0")
        self.warning_distance = float(warning_distance)
        self.critical_distance = float(critical_distance)
        self.species_weights = dict(species_weights or {})
        for species, weight in self.species_weights.items():
            if not isinstance(species, str) or not isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError("species weights must be finite positive numbers")

    @staticmethod
    def _valid_distance(distance: Any) -> Optional[float]:
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            return None
        value = float(distance)
        return value if isfinite(value) and value >= 0 else None

    @staticmethod
    def _valid_confidence(confidence: Any) -> float:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return 0.0
        value = float(confidence)
        return min(1.0, max(0.0, value)) if isfinite(value) else 0.0

    def analyze(self, detection: Mapping[str, Any], distance: Any) -> ThreatResult:
        """Return one structured threat result without mutating ``detection``."""
        class_name = str(detection.get("class_name", "unknown"))
        confidence = self._valid_confidence(detection.get("confidence"))
        valid_distance = self._valid_distance(distance)

        if valid_distance is None:
            return {
                "class_name": class_name,
                "confidence": confidence,
                "distance": None,
                "threat_level": "SAFE",
                "threat_score": 0.0,
                "reason": "Distance unavailable or invalid; defaulted to SAFE.",
            }

        if valid_distance <= self.critical_distance:
            threat_level = "CRITICAL"
            reason = f"{class_name} is within the critical distance threshold."
            urgency = 1.0
        elif valid_distance <= self.warning_distance:
            threat_level = "WARNING"
            reason = f"{class_name} is within the warning distance threshold."
            urgency = (self.warning_distance - valid_distance) / (
                self.warning_distance - self.critical_distance
            )
        else:
            threat_level = "SAFE"
            reason = f"{class_name} is beyond the warning distance threshold."
            urgency = 0.0

        species_weight = float(self.species_weights.get(class_name, 1.0))
        threat_score = min(1.0, max(0.0, urgency * confidence * species_weight))
        return {
            "class_name": class_name,
            "confidence": confidence,
            "distance": valid_distance,
            "threat_level": threat_level,
            "threat_score": round(threat_score, 4),
            "reason": reason,
        }

    def analyze_detections(
        self,
        detections: Iterable[Mapping[str, Any]],
        distances: Iterable[Any],
    ) -> List[ThreatResult]:
        """Analyze detections using distances in the same order."""
        return [self.analyze(detection, distance)
            for detection, distance in zip_longest(detections, distances, fillvalue=None)
            if detection is not None]