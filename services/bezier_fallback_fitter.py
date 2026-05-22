from __future__ import annotations

from dataclasses import dataclass
import math

from core.precision import PrecisionUtility
from core.types import Anchor, Point, Segment
from services.scorer import ScorerConfig


@dataclass(frozen=True, slots=True)
class BezierFallbackFitterConfig:
    max_error: float = 2.5
    max_segments: int = 8
    smoothness: float = 0.75
    min_rdp_epsilon: float = 0.25
    rdp_epsilon_ratio: float = 0.5
    min_segment_point_count: int = 4
    confidence_fit_weight: float = 0.65
    confidence_segment_weight: float = 0.2
    confidence_reduction_weight: float = 0.15


@dataclass(frozen=True, slots=True)
class BezierFallbackFitResult:
    closed: bool
    anchors: tuple[Anchor, ...]
    segments: tuple[Segment, ...]
    fit_error: float
    max_error: float
    complexity_score: float
    confidence: float
    source_point_count: int


@dataclass(frozen=True, slots=True)
class _BezierCurve:
    start: Point
    control1: Point
    control2: Point
    end: Point


@dataclass(frozen=True, slots=True)
class _ChunkFit:
    start_index: int
    end_index: int
    curve: _BezierCurve
    rmse: float
    max_error: float
    worst_index: int


class BezierFallbackFitter:
    def __init__(self, config: BezierFallbackFitterConfig | None = None) -> None:
        self.config = config or BezierFallbackFitterConfig()
        scorer_config = ScorerConfig()
        self._segment_complexity = (
            scorer_config.bezier_complexity_weight + (2.0 * scorer_config.control_point_penalty)
        )

    def fit_contour(
        self,
        points: tuple[Point, ...] | list[Point],
        *,
        path_id: str,
        closed: bool = False,
        max_error: float | None = None,
        max_segments: int | None = None,
        smoothness: float | None = None,
    ) -> BezierFallbackFitResult:
        normalized = self._normalize_points(points, closed=closed)
        minimum_points = 3 if closed else 2
        if len(normalized) < minimum_points:
            raise ValueError("not enough points for bezier fallback fitting")

        effective_max_error = float(max_error if max_error is not None else self.config.max_error)
        effective_max_segments = max(1, int(max_segments if max_segments is not None else self.config.max_segments))
        effective_smoothness = max(
            0.0,
            min(1.0, float(smoothness if smoothness is not None else self.config.smoothness)),
        )

        ordered_points = normalized
        if closed:
            ordered_points = self._rotate_closed_points(normalized)
            ordered_points = ordered_points + (ordered_points[0],)

        breakpoints = self._adaptive_breakpoints(
            ordered_points,
            closed=closed,
            max_error=effective_max_error,
            max_segments=effective_max_segments,
        )
        chunk_fits = self._fit_chunks(
            ordered_points,
            breakpoints,
            smoothness=effective_smoothness,
            max_error=effective_max_error,
            max_segments=effective_max_segments,
        )
        anchors, segments = self._build_geometry(
            path_id=path_id,
            ordered_points=ordered_points,
            chunk_fits=chunk_fits,
            closed=closed,
        )
        fit_error = math.sqrt(
            sum((chunk.rmse ** 2) * max(chunk.end_index - chunk.start_index, 1) for chunk in chunk_fits)
            / max(sum(max(chunk.end_index - chunk.start_index, 1) for chunk in chunk_fits), 1)
        )
        observed_max_error = max(chunk.max_error for chunk in chunk_fits) if chunk_fits else 0.0
        complexity_score = len(segments) * self._segment_complexity
        confidence = self._confidence(
            fit_error=fit_error,
            max_error=observed_max_error,
            segment_count=len(segments),
            max_segments=effective_max_segments,
            source_point_count=len(normalized),
        )
        return BezierFallbackFitResult(
            closed=closed,
            anchors=anchors,
            segments=segments,
            fit_error=fit_error,
            max_error=observed_max_error,
            complexity_score=complexity_score,
            confidence=confidence,
            source_point_count=len(normalized),
        )

    def _adaptive_breakpoints(
        self,
        points: tuple[Point, ...],
        *,
        closed: bool,
        max_error: float,
        max_segments: int,
    ) -> list[int]:
        if len(points) <= 2:
            return [0, len(points) - 1]

        epsilon = max(self.config.min_rdp_epsilon, max_error * self.config.rdp_epsilon_ratio)
        breakpoints = self._rdp_indices(points, epsilon)
        while (len(breakpoints) - 1) > max_segments:
            epsilon *= 1.35
            breakpoints = self._rdp_indices(points, epsilon)

        if closed and breakpoints[0] != 0:
            breakpoints = [0] + [index for index in breakpoints if index != 0]
        if breakpoints[-1] != len(points) - 1:
            breakpoints.append(len(points) - 1)
        return sorted(set(breakpoints))

    def _fit_chunks(
        self,
        points: tuple[Point, ...],
        breakpoints: list[int],
        *,
        smoothness: float,
        max_error: float,
        max_segments: int,
    ) -> tuple[_ChunkFit, ...]:
        current_breakpoints = list(breakpoints)
        while True:
            chunk_fits = [
                self._fit_chunk(
                    points,
                    start_index=current_breakpoints[index],
                    end_index=current_breakpoints[index + 1],
                    smoothness=smoothness,
                )
                for index in range(len(current_breakpoints) - 1)
            ]
            worst = max(chunk_fits, key=lambda item: item.max_error)
            if worst.max_error <= max_error or (len(current_breakpoints) - 1) >= max_segments:
                return tuple(chunk_fits)
            if worst.worst_index <= worst.start_index or worst.worst_index >= worst.end_index:
                return tuple(chunk_fits)
            current_breakpoints.append(worst.worst_index)
            current_breakpoints = sorted(set(current_breakpoints))

    def _fit_chunk(
        self,
        points: tuple[Point, ...],
        *,
        start_index: int,
        end_index: int,
        smoothness: float,
    ) -> _ChunkFit:
        chunk = points[start_index : end_index + 1]
        if len(chunk) < 2:
            raise ValueError("chunk must contain at least two points")

        t_values = self._chord_length_parameters(chunk)
        start = chunk[0]
        end = chunk[-1]
        fallback_alpha = PrecisionUtility.distance_between_points(start, end) / 3.0
        tangent_start = self._normalized_direction(chunk[0], chunk[1])
        tangent_end = self._normalized_direction(chunk[-1], chunk[-2])
        alpha1, alpha2 = self._solve_alpha_parameters(chunk, t_values, tangent_start, tangent_end, fallback_alpha)
        alpha1 = (smoothness * alpha1) + ((1.0 - smoothness) * fallback_alpha)
        alpha2 = (smoothness * alpha2) + ((1.0 - smoothness) * fallback_alpha)

        curve = _BezierCurve(
            start=start,
            control1=(start[0] + (tangent_start[0] * alpha1), start[1] + (tangent_start[1] * alpha1)),
            control2=(end[0] + (tangent_end[0] * alpha2), end[1] + (tangent_end[1] * alpha2)),
            end=end,
        )

        squared_errors: list[float] = []
        max_error = 0.0
        worst_index = start_index
        for local_index, (point, t_value) in enumerate(zip(chunk, t_values)):
            bezier_point = self._cubic_bezier_point(curve, t_value)
            error = PrecisionUtility.distance_between_points(point, bezier_point)
            squared_errors.append(error * error)
            if error > max_error:
                max_error = error
                worst_index = start_index + local_index

        rmse = math.sqrt(sum(squared_errors) / max(len(squared_errors), 1))
        return _ChunkFit(
            start_index=start_index,
            end_index=end_index,
            curve=curve,
            rmse=rmse,
            max_error=max_error,
            worst_index=worst_index,
        )

    def _solve_alpha_parameters(
        self,
        points: tuple[Point, ...],
        t_values: tuple[float, ...],
        tangent_start: Point,
        tangent_end: Point,
        fallback_alpha: float,
    ) -> tuple[float, float]:
        c00 = 0.0
        c01 = 0.0
        c11 = 0.0
        x0 = 0.0
        x1 = 0.0
        start = points[0]
        end = points[-1]

        for point, t_value in zip(points, t_values):
            one_minus_t = 1.0 - t_value
            b0 = one_minus_t ** 3
            b1 = 3.0 * (one_minus_t ** 2) * t_value
            b2 = 3.0 * one_minus_t * (t_value ** 2)
            b3 = t_value ** 3

            a1 = (tangent_start[0] * b1, tangent_start[1] * b1)
            a2 = (tangent_end[0] * b2, tangent_end[1] * b2)
            base = (
                (start[0] * (b0 + b1)) + (end[0] * (b2 + b3)),
                (start[1] * (b0 + b1)) + (end[1] * (b2 + b3)),
            )
            rhs = (point[0] - base[0], point[1] - base[1])

            c00 += (a1[0] * a1[0]) + (a1[1] * a1[1])
            c01 += (a1[0] * a2[0]) + (a1[1] * a2[1])
            c11 += (a2[0] * a2[0]) + (a2[1] * a2[1])
            x0 += (a1[0] * rhs[0]) + (a1[1] * rhs[1])
            x1 += (a2[0] * rhs[0]) + (a2[1] * rhs[1])

        determinant = (c00 * c11) - (c01 * c01)
        if abs(determinant) <= 1e-9:
            return (fallback_alpha, fallback_alpha)

        alpha1 = ((x0 * c11) - (x1 * c01)) / determinant
        alpha2 = ((c00 * x1) - (c01 * x0)) / determinant
        if not math.isfinite(alpha1) or not math.isfinite(alpha2):
            return (fallback_alpha, fallback_alpha)
        if alpha1 <= 1e-6 or alpha2 <= 1e-6:
            return (fallback_alpha, fallback_alpha)
        return (alpha1, alpha2)

    def _build_geometry(
        self,
        *,
        path_id: str,
        ordered_points: tuple[Point, ...],
        chunk_fits: tuple[_ChunkFit, ...],
        closed: bool,
    ) -> tuple[tuple[Anchor, ...], tuple[Segment, ...]]:
        if not chunk_fits:
            raise ValueError("bezier fallback produced no segments")

        if closed:
            anchors: list[Anchor] = []
            segment_ids = tuple(f"{path_id}_bezier_fallback_segment_{index}" for index in range(len(chunk_fits)))
            for index, chunk in enumerate(chunk_fits):
                previous_chunk = chunk_fits[index - 1]
                anchors.append(
                    self._anchor(
                        anchor_id=f"{path_id}_bezier_fallback_anchor_{index}",
                        path_id=path_id,
                        position=chunk.curve.start,
                        in_handle=previous_chunk.curve.control2,
                        out_handle=chunk.curve.control1,
                    )
                )
            segments = tuple(
                Segment(
                    segment_id=segment_ids[index],
                    path_id=path_id,
                    type="bezier",
                    params={
                        "start": list(chunk.curve.start),
                        "control1": list(chunk.curve.control1),
                        "control2": list(chunk.curve.control2),
                        "end": list(chunk.curve.end),
                    },
                    anchors=(
                        anchors[index].anchor_id,
                        anchors[(index + 1) % len(anchors)].anchor_id,
                    ),
                    fit_error=chunk.rmse,
                    confidence=None,
                    complexity_score=self._segment_complexity,
                    rigidity="medium",
                    metadata={"coordinate_space": "vector", "fallback": "bezier"},
                )
                for index, chunk in enumerate(chunk_fits)
            )
            return (tuple(anchors), segments)

        anchors: list[Anchor] = []
        for index, chunk in enumerate(chunk_fits):
            if index == 0:
                anchors.append(
                    self._anchor(
                        anchor_id=f"{path_id}_bezier_fallback_anchor_0",
                        path_id=path_id,
                        position=chunk.curve.start,
                        in_handle=None,
                        out_handle=chunk.curve.control1,
                    )
                )
            anchors.append(
                self._anchor(
                    anchor_id=f"{path_id}_bezier_fallback_anchor_{index + 1}",
                    path_id=path_id,
                    position=chunk.curve.end,
                    in_handle=chunk.curve.control2,
                    out_handle=None,
                )
            )
        anchors = self._merge_open_anchor_handles(anchors, chunk_fits)
        segments = tuple(
            Segment(
                segment_id=f"{path_id}_bezier_fallback_segment_{index}",
                path_id=path_id,
                type="bezier",
                params={
                    "start": list(chunk.curve.start),
                    "control1": list(chunk.curve.control1),
                    "control2": list(chunk.curve.control2),
                    "end": list(chunk.curve.end),
                },
                anchors=(anchors[index].anchor_id, anchors[index + 1].anchor_id),
                fit_error=chunk.rmse,
                confidence=None,
                complexity_score=self._segment_complexity,
                rigidity="medium",
                metadata={"coordinate_space": "vector", "fallback": "bezier"},
            )
            for index, chunk in enumerate(chunk_fits)
        )
        return (tuple(anchors), segments)

    def _merge_open_anchor_handles(
        self,
        anchors: list[Anchor],
        chunk_fits: tuple[_ChunkFit, ...],
    ) -> list[Anchor]:
        merged = list(anchors)
        for index in range(1, len(anchors) - 1):
            previous_chunk = chunk_fits[index - 1]
            next_chunk = chunk_fits[index]
            merged[index] = self._anchor(
                anchor_id=anchors[index].anchor_id,
                path_id=anchors[index].path_id,
                position=anchors[index].position,
                in_handle=previous_chunk.curve.control2,
                out_handle=next_chunk.curve.control1,
            )
        return merged

    def _anchor(
        self,
        *,
        anchor_id: str,
        path_id: str,
        position: Point,
        in_handle: Point | None,
        out_handle: Point | None,
    ) -> Anchor:
        shared_tangent = None
        if in_handle is not None and out_handle is not None:
            shared_tangent = self._normalized_direction(in_handle, out_handle)
        elif out_handle is not None:
            shared_tangent = self._normalized_direction(position, out_handle)
        elif in_handle is not None:
            shared_tangent = self._normalized_direction(in_handle, position)
        return Anchor(
            anchor_id=anchor_id,
            path_id=path_id,
            position=position,
            continuity="smooth",
            shared_tangent=shared_tangent,
            in_handle=in_handle,
            out_handle=out_handle,
            metadata={"coordinate_space": "vector", "fallback": "bezier"},
        )

    def _normalize_points(
        self,
        points: tuple[Point, ...] | list[Point],
        *,
        closed: bool,
    ) -> tuple[Point, ...]:
        normalized: list[Point] = []
        for point in points:
            candidate = (float(point[0]), float(point[1]))
            if normalized and PrecisionUtility.points_close(normalized[-1], candidate):
                continue
            normalized.append(candidate)
        if closed and len(normalized) > 1 and PrecisionUtility.points_close(normalized[0], normalized[-1]):
            normalized.pop()
        return tuple(normalized)

    def _rotate_closed_points(self, points: tuple[Point, ...]) -> tuple[Point, ...]:
        start_index = min(range(len(points)), key=lambda index: (points[index][0], points[index][1]))
        return points[start_index:] + points[:start_index]

    def _rdp_indices(self, points: tuple[Point, ...], epsilon: float) -> list[int]:
        if len(points) <= 2:
            return [0, len(points) - 1]

        def recurse(start_index: int, end_index: int) -> list[int]:
            start = points[start_index]
            end = points[end_index]
            best_distance = -1.0
            best_index = -1
            for index in range(start_index + 1, end_index):
                distance = self._point_to_line_distance(points[index], start, end)
                if distance > best_distance:
                    best_distance = distance
                    best_index = index
            if best_distance <= epsilon or best_index < 0:
                return [start_index, end_index]
            left = recurse(start_index, best_index)
            right = recurse(best_index, end_index)
            return left[:-1] + right

        indices = recurse(0, len(points) - 1)
        deduped: list[int] = []
        for index in indices:
            if deduped and deduped[-1] == index:
                continue
            deduped.append(index)
        return deduped

    def _chord_length_parameters(self, points: tuple[Point, ...]) -> tuple[float, ...]:
        if len(points) == 1:
            return (0.0,)
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(
                distances[-1] + PrecisionUtility.distance_between_points(points[index - 1], points[index])
            )
        total = distances[-1]
        if PrecisionUtility.near_zero(total):
            count = len(points) - 1
            return tuple(index / max(count, 1) for index in range(len(points)))
        return tuple(distance / total for distance in distances)

    def _confidence(
        self,
        *,
        fit_error: float,
        max_error: float,
        segment_count: int,
        max_segments: int,
        source_point_count: int,
    ) -> float:
        fit_budget = max(self.config.max_error, max_error, 1e-6)
        fit_score = 1.0 if fit_error <= fit_budget else max(0.0, 1.0 - ((fit_error - fit_budget) / fit_budget))
        segment_score = max(0.0, 1.0 - ((segment_count - 1) / max(max_segments - 1, 1)))
        reduction_ratio = source_point_count / max(segment_count * 6, 1)
        reduction_score = max(0.0, min(1.0, reduction_ratio / 3.0))
        weighted = (
            (fit_score * self.config.confidence_fit_weight)
            + (segment_score * self.config.confidence_segment_weight)
            + (reduction_score * self.config.confidence_reduction_weight)
        )
        return max(0.0, min(1.0, weighted))

    def _normalized_direction(self, start: Point, end: Point) -> Point:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if PrecisionUtility.near_zero(length):
            return (1.0, 0.0)
        return (dx / length, dy / length)

    def _point_to_line_distance(self, point: Point, start: Point, end: Point) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if PrecisionUtility.near_zero(dx) and PrecisionUtility.near_zero(dy):
            return PrecisionUtility.distance_between_points(point, start)
        numerator = abs((dy * point[0]) - (dx * point[1]) + (end[0] * start[1]) - (end[1] * start[0]))
        denominator = math.hypot(dx, dy)
        return numerator / denominator

    def _cubic_bezier_point(self, curve: _BezierCurve, t_value: float) -> Point:
        one_minus_t = 1.0 - t_value
        x = (
            (one_minus_t ** 3) * curve.start[0]
            + 3.0 * (one_minus_t ** 2) * t_value * curve.control1[0]
            + 3.0 * one_minus_t * (t_value ** 2) * curve.control2[0]
            + (t_value ** 3) * curve.end[0]
        )
        y = (
            (one_minus_t ** 3) * curve.start[1]
            + 3.0 * (one_minus_t ** 2) * t_value * curve.control1[1]
            + 3.0 * one_minus_t * (t_value ** 2) * curve.control2[1]
            + (t_value ** 3) * curve.end[1]
        )
        return (x, y)


__all__ = [
    "BezierFallbackFitResult",
    "BezierFallbackFitter",
    "BezierFallbackFitterConfig",
]
