from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.types import Anchor, Path, Segment, VectorDocument
from services.edge_error import EdgeErrorResult


@dataclass(frozen=True, slots=True)
class ScorerConfig:
    line_complexity_weight: float = 1.0
    arc_complexity_weight: float = 1.5
    circle_complexity_weight: float = 1.5
    ellipse_complexity_weight: float = 2.0
    bezier_complexity_weight: float = 2.5
    bspline_complexity_weight: float = 3.0
    polyline_complexity_weight: float = 1.5
    control_point_penalty: float = 0.5
    topology_error_penalty: float = 25.0
    max_gap_weight: float = 1.0
    self_intersection_penalty: float = 20.0
    coordinate_space_penalty: float = 10.0
    non_vector_metadata_penalty: float = 2.0


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    edge_error_score: float
    geometry_complexity_score: float
    topology_error_score: float
    self_intersection_score: float
    coordinate_consistency_score: float


@dataclass(frozen=True, slots=True)
class ScoreResult:
    total_score: float
    breakdown: ScoreBreakdown


class Scorer:
    def __init__(self, config: ScorerConfig | None = None) -> None:
        self.config = config or ScorerConfig()

    def score_document(
        self,
        document: VectorDocument,
        *,
        edge_error: EdgeErrorResult | None = None,
    ) -> ScoreResult:
        edge_error_score = self._edge_error_score(edge_error)
        geometry_complexity_score = self._geometry_complexity_score(document)
        topology_error_score = self._topology_error_score(document)
        self_intersection_score = self._self_intersection_score(document)
        coordinate_consistency_score = self._coordinate_consistency_score(document)

        breakdown = ScoreBreakdown(
            edge_error_score=edge_error_score,
            geometry_complexity_score=geometry_complexity_score,
            topology_error_score=topology_error_score,
            self_intersection_score=self_intersection_score,
            coordinate_consistency_score=coordinate_consistency_score,
        )
        total_score = (
            breakdown.edge_error_score
            + breakdown.geometry_complexity_score
            + breakdown.topology_error_score
            + breakdown.self_intersection_score
            + breakdown.coordinate_consistency_score
        )
        return ScoreResult(total_score=total_score, breakdown=breakdown)

    def _edge_error_score(self, edge_error: EdgeErrorResult | None) -> float:
        if edge_error is None:
            return 0.0
        return float(edge_error.chamfer_error)

    def _geometry_complexity_score(self, document: VectorDocument) -> float:
        return sum(self._segment_complexity(segment) for segment in document.segments)

    def _segment_complexity(self, segment: Segment) -> float:
        weight_by_type = {
            "line": self.config.line_complexity_weight,
            "arc": self.config.arc_complexity_weight,
            "circle": self.config.circle_complexity_weight,
            "ellipse": self.config.ellipse_complexity_weight,
            "bezier": self.config.bezier_complexity_weight,
            "bspline": self.config.bspline_complexity_weight,
            "polyline": self.config.polyline_complexity_weight,
        }
        complexity = weight_by_type[segment.type]
        if segment.type == "bezier":
            complexity += self.config.control_point_penalty * self._bezier_control_point_count(segment)
        if segment.type == "bspline":
            complexity += self.config.control_point_penalty * len(segment.params.get("points", ()))
        if segment.complexity_score is not None:
            complexity += float(segment.complexity_score)
        return complexity

    def _bezier_control_point_count(self, segment: Segment) -> int:
        return int("control1" in segment.params) + int("control2" in segment.params)

    def _topology_error_score(self, document: VectorDocument) -> float:
        score = 0.0
        for path in document.paths:
            if path.topology_status == "topology_error":
                score += self.config.topology_error_penalty
            score += max(0.0, float(path.max_gap)) * self.config.max_gap_weight
        return score

    def _self_intersection_score(self, document: VectorDocument) -> float:
        return sum(max(0, int(path.self_intersection_count)) * self.config.self_intersection_penalty for path in document.paths)

    def _coordinate_consistency_score(self, document: VectorDocument) -> float:
        score = 0.0
        if document.coordinate_system.internal_space != "vector":
            score += self.config.coordinate_space_penalty

        score += self._coordinate_space_penalty(document.metadata)
        for path in document.paths:
            score += self._coordinate_space_penalty(path.metadata)
        for segment in document.segments:
            score += self._coordinate_space_penalty(segment.metadata)
        for anchor in document.anchors:
            score += self._coordinate_space_penalty(anchor.metadata)
        return score

    def _coordinate_space_penalty(self, value: Any) -> float:
        if isinstance(value, dict):
            score = 0.0
            for key, nested_value in value.items():
                if key == "coordinate_space":
                    score += 0.0 if nested_value == "vector" else self.config.non_vector_metadata_penalty
                else:
                    score += self._coordinate_space_penalty(nested_value)
            return score
        if isinstance(value, (list, tuple)):
            return sum(self._coordinate_space_penalty(item) for item in value)
        return 0.0


__all__ = [
    "ScoreBreakdown",
    "ScoreResult",
    "Scorer",
    "ScorerConfig",
]
