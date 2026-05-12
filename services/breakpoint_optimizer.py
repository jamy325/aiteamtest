from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from core.precision import PrecisionUtility, Vector2
from core.types import Point

TargetType = Literal["line", "arc", "circle", "ellipse", "bezier", "bspline", "polyline", "unknown"]


@dataclass(frozen=True, slots=True)
class BreakPointOptimizerConfig:
    min_confidence_threshold: float = 0.05
    angle_weight: float = 1.0
    curvature_weight: float = 0.9
    residual_weight: float = 1.1
    tangent_weight: float = 0.7
    user_hint_weight: float = 1.4
    adjacent_endpoint_weight: float = 0.9
    ai_region_weight: float = 0.35
    range_padding: int = 1


@dataclass(frozen=True, slots=True)
class BreakPointRequest:
    points: tuple[Point, ...]
    rough_range: tuple[int, int]
    target_type: TargetType | str
    residuals: tuple[float, ...] = ()
    tangent_vectors: tuple[Vector2 | None, ...] = ()
    user_breakpoints: tuple[int, ...] = ()
    adjacent_endpoints: tuple[int, ...] = ()
    ai_marked_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple((float(x), float(y)) for x, y in self.points))
        object.__setattr__(self, "rough_range", (int(self.rough_range[0]), int(self.rough_range[1])))
        object.__setattr__(self, "residuals", tuple(float(value) for value in self.residuals))
        object.__setattr__(
            self,
            "tangent_vectors",
            tuple(None if vector is None else (float(vector[0]), float(vector[1])) for vector in self.tangent_vectors),
        )
        object.__setattr__(self, "user_breakpoints", tuple(int(index) for index in self.user_breakpoints))
        object.__setattr__(self, "adjacent_endpoints", tuple(int(index) for index in self.adjacent_endpoints))
        if self.ai_marked_range is not None:
            object.__setattr__(self, "ai_marked_range", (int(self.ai_marked_range[0]), int(self.ai_marked_range[1])))


@dataclass(frozen=True, slots=True)
class BreakPointResult:
    optimized_range: tuple[int, int]
    breakpoints: tuple[int, ...]
    confidence: float
    reason: str


class BreakPointOptimizer:
    def __init__(self, config: BreakPointOptimizerConfig | None = None) -> None:
        self.config = config or BreakPointOptimizerConfig()

    def optimize(self, request: BreakPointRequest) -> BreakPointResult:
        if len(request.points) < 2:
            raise ValueError("at least two Vector Space points are required")

        point_count = len(request.points)
        rough_start, rough_end = self._clamp_range(request.rough_range, point_count)
        candidate_indexes = tuple(range(max(1, rough_start), min(point_count - 2, rough_end) + 1))

        if not candidate_indexes:
            return BreakPointResult(
                optimized_range=(rough_start, rough_end),
                breakpoints=(),
                confidence=0.0,
                reason="fallback_to_rough_range",
            )

        residual_scale = max((abs(request.residuals[index]) for index in candidate_indexes if index < len(request.residuals)), default=0.0)
        scored_candidates = tuple(
            self._score_candidate(
                request=request,
                index=index,
                rough_range=(rough_start, rough_end),
                residual_scale=residual_scale,
            )
            for index in candidate_indexes
        )
        best_candidate = max(scored_candidates, key=lambda item: (item.score, -abs(item.index - ((rough_start + rough_end) / 2.0))))

        if best_candidate.score < self.config.min_confidence_threshold:
            return BreakPointResult(
                optimized_range=(rough_start, rough_end),
                breakpoints=(),
                confidence=max(0.0, min(1.0, best_candidate.score)),
                reason="fallback_to_rough_range",
            )

        support_indexes = [best_candidate.index]
        support_indexes.extend(index for index in request.user_breakpoints if rough_start <= index <= rough_end)
        support_indexes.extend(index for index in request.adjacent_endpoints if rough_start <= index <= rough_end)
        if request.ai_marked_range is not None:
            ai_start, ai_end = self._clamp_range(request.ai_marked_range, point_count)
            if ai_start <= rough_end and ai_end >= rough_start:
                support_indexes.extend((max(rough_start, ai_start), min(rough_end, ai_end)))

        optimized_start = max(rough_start, min(support_indexes) - self.config.range_padding)
        optimized_end = min(rough_end, max(support_indexes) + self.config.range_padding)
        confidence = max(0.0, min(1.0, best_candidate.score / self._normalization_denominator()))
        reason = ",".join(best_candidate.reasons) if best_candidate.reasons else "score_peak"

        return BreakPointResult(
            optimized_range=(optimized_start, optimized_end),
            breakpoints=(best_candidate.index,),
            confidence=confidence,
            reason=reason,
        )

    def _score_candidate(
        self,
        *,
        request: BreakPointRequest,
        index: int,
        rough_range: tuple[int, int],
        residual_scale: float,
    ) -> "_ScoredCandidate":
        reasons: list[str] = []
        score = 0.0

        angle_score = self._angle_jump_score(request.points, index)
        if angle_score > 0.0:
            score += angle_score * self.config.angle_weight
            reasons.append("angle_jump")

        curvature_score = self._curvature_jump_score(request.points, index)
        if curvature_score > 0.0:
            score += curvature_score * self.config.curvature_weight
            reasons.append("curvature_jump")

        tangent_score = self._tangent_jump_score(request, index)
        if tangent_score > 0.0:
            score += tangent_score * self.config.tangent_weight
            reasons.append("tangent_direction_jump")

        residual_score = self._residual_peak_score(request, index, residual_scale)
        if residual_score > 0.0:
            score += residual_score * self.config.residual_weight
            reasons.append("residual_peak")

        if index in request.user_breakpoints:
            score += self.config.user_hint_weight
            reasons.append("user_breakpoint")

        if index in request.adjacent_endpoints:
            score += self.config.adjacent_endpoint_weight
            reasons.append("adjacent_endpoint")

        if request.ai_marked_range is not None:
            ai_start, ai_end = self._clamp_range(request.ai_marked_range, len(request.points))
            if ai_start <= index <= ai_end:
                score += self.config.ai_region_weight
                reasons.append("ai_marked_region")

        if request.target_type in {"arc", "circle", "ellipse"} and curvature_score > 0.0:
            score += 0.25
        if request.target_type == "line" and angle_score > 0.0:
            score += 0.1

        deduped_reasons = tuple(dict.fromkeys(reasons))
        return _ScoredCandidate(index=index, score=score, reasons=deduped_reasons)

    def _angle_jump_score(self, points: tuple[Point, ...], index: int) -> float:
        left = self._vector(points[index - 1], points[index])
        right = self._vector(points[index], points[index + 1])
        left_unit = PrecisionUtility.normalize_vector(left)
        right_unit = PrecisionUtility.normalize_vector(right)
        if left_unit is None or right_unit is None:
            return 0.0
        dot = max(-1.0, min(1.0, (left_unit[0] * right_unit[0]) + (left_unit[1] * right_unit[1])))
        turn_angle = math.acos(dot)
        return turn_angle / math.pi

    def _curvature_jump_score(self, points: tuple[Point, ...], index: int) -> float:
        current_curvature = self._local_curvature(points, index)
        previous_curvature = self._local_curvature(points, index - 1) if index - 1 >= 1 else 0.0
        next_curvature = self._local_curvature(points, index + 1) if index + 1 <= len(points) - 2 else current_curvature
        return min(1.0, max(abs(current_curvature - previous_curvature), abs(next_curvature - current_curvature)))

    def _local_curvature(self, points: tuple[Point, ...], index: int) -> float:
        if index < 1 or index > len(points) - 2:
            return 0.0
        left_length = PrecisionUtility.distance_between_points(points[index - 1], points[index])
        right_length = PrecisionUtility.distance_between_points(points[index], points[index + 1])
        average_length = (left_length + right_length) / 2.0
        if PrecisionUtility.near_zero(average_length):
            return 0.0
        return min(1.0, self._angle_jump_score(points, index) / average_length)

    def _tangent_jump_score(self, request: BreakPointRequest, index: int) -> float:
        if len(request.tangent_vectors) > index and request.tangent_vectors[index] is not None and request.tangent_vectors[index - 1] is not None:
            left = PrecisionUtility.normalize_vector(request.tangent_vectors[index - 1])
            right = PrecisionUtility.normalize_vector(request.tangent_vectors[index])
            if left is None or right is None:
                return 0.0
            dot = max(-1.0, min(1.0, (left[0] * right[0]) + (left[1] * right[1])))
            return math.acos(dot) / math.pi
        return 0.0

    def _residual_peak_score(self, request: BreakPointRequest, index: int, residual_scale: float) -> float:
        if index >= len(request.residuals) or PrecisionUtility.near_zero(residual_scale):
            return 0.0
        return min(1.0, abs(request.residuals[index]) / residual_scale)

    def _clamp_range(self, rough_range: tuple[int, int], point_count: int) -> tuple[int, int]:
        start, end = sorted((int(rough_range[0]), int(rough_range[1])))
        return (max(0, min(start, point_count - 1)), max(0, min(end, point_count - 1)))

    def _vector(self, start: Point, end: Point) -> Vector2:
        return (end[0] - start[0], end[1] - start[1])

    def _normalization_denominator(self) -> float:
        return (
            self.config.angle_weight
            + self.config.curvature_weight
            + self.config.residual_weight
            + self.config.tangent_weight
            + self.config.user_hint_weight
            + self.config.adjacent_endpoint_weight
            + self.config.ai_region_weight
            + 0.25
        )


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    index: int
    score: float
    reasons: tuple[str, ...]


__all__ = [
    "BreakPointOptimizer",
    "BreakPointOptimizerConfig",
    "BreakPointRequest",
    "BreakPointResult",
    "TargetType",
]
