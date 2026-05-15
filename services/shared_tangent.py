from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from core.precision import PrecisionUtility
from core.types import Anchor, Constraint, Point, Segment, VectorDocument, updated


def segment_outward_tangent(segment: Segment, anchor: Anchor) -> Point | None:
    if segment.type == "line":
        return _line_outward_tangent(segment, anchor)
    if segment.type == "arc":
        return _arc_outward_tangent(segment, anchor)
    if segment.type == "bezier":
        return _bezier_outward_tangent(segment, anchor)
    return None


def shared_tangent_violation(
    segment_a: Segment,
    segment_b: Segment,
    anchor: Anchor,
    *,
    shared_tangent: Point | None = None,
) -> float | None:
    tangent_a = segment_outward_tangent(segment_a, anchor)
    tangent_b = segment_outward_tangent(segment_b, anchor)
    if tangent_a is None or tangent_b is None:
        return None

    pair_mismatch = _tangent_mismatch(tangent_a, tangent_b)
    if shared_tangent is None:
        return pair_mismatch

    shared_unit = PrecisionUtility.normalize_vector(shared_tangent)
    if shared_unit is None:
        return pair_mismatch

    error_a = _vector_to_shared_tangent_error(tangent_a, shared_unit)
    error_b = _vector_to_shared_tangent_error(tangent_b, shared_unit)
    return max(pair_mismatch, (error_a + error_b) / 2.0)


@dataclass(frozen=True, slots=True)
class SharedTangentOptimizationResult:
    success: bool
    segment_a: Segment
    segment_b: Segment
    shared_tangent: Point | None
    violation: float
    confidence: float
    fit_error: float
    tangent_mismatch: float
    movement_penalty: float
    reason: str
    constraint_id: str | None = None


class SharedTangentOptimizer:
    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        max_tangent_mismatch: float = 0.2,
        movement_penalty_scale: float = 5.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_tangent_mismatch = max_tangent_mismatch
        self.movement_penalty_scale = movement_penalty_scale

    def optimize_document(
        self,
        document: VectorDocument,
        *,
        points_by_constraint: Mapping[str, Sequence[Point]] | None = None,
    ) -> tuple[SharedTangentOptimizationResult, ...]:
        segment_by_id = {segment.segment_id: segment for segment in document.segments}
        anchor_by_id = {anchor.anchor_id: anchor for anchor in document.anchors}
        results: list[SharedTangentOptimizationResult] = []

        for constraint in document.constraints:
            if constraint.type not in {"g1_continuity", "shared_tangent"}:
                continue
            if len(constraint.targets) < 3:
                continue

            segment_a = segment_by_id.get(constraint.targets[0])
            segment_b = segment_by_id.get(constraint.targets[1])
            anchor = anchor_by_id.get(constraint.targets[2])
            if segment_a is None or segment_b is None or anchor is None:
                continue

            support_points = tuple(points_by_constraint.get(constraint.constraint_id, ())) if points_by_constraint else ()
            results.append(
                self.optimize_pair(
                    segment_a,
                    segment_b,
                    anchor,
                    support_points,
                    constraint=constraint,
                )
            )

        return tuple(results)

    def optimize_pair(
        self,
        segment_a: Segment,
        segment_b: Segment,
        anchor: Anchor,
        points: Sequence[Point],
        *,
        constraint: Constraint | None = None,
    ) -> SharedTangentOptimizationResult:
        if constraint is not None and constraint.locked:
            return self._failure(segment_a, segment_b, "locked constraint", constraint=constraint)
        if segment_a.locked or segment_b.locked or anchor.locked:
            return self._failure(segment_a, segment_b, "locked segment or anchor", constraint=constraint)

        confidence_hint = self._constraint_confidence(constraint)
        if confidence_hint < self.min_confidence:
            return self._failure(segment_a, segment_b, "low confidence constraint", constraint=constraint)

        pair_types = {segment_a.type, segment_b.type}
        if pair_types == {"line", "arc"}:
            return self._optimize_line_arc_pair(
                segment_a,
                segment_b,
                anchor,
                tuple(points),
                confidence_hint=confidence_hint,
                constraint=constraint,
            )
        if pair_types == {"arc", "bezier"}:
            return self._optimize_arc_bezier_pair(
                segment_a,
                segment_b,
                anchor,
                tuple(points),
                confidence_hint=confidence_hint,
                constraint=constraint,
            )
        return self._failure(segment_a, segment_b, "unsupported segment pair", constraint=constraint)

    def _optimize_line_arc_pair(
        self,
        segment_a: Segment,
        segment_b: Segment,
        anchor: Anchor,
        points: tuple[Point, ...],
        *,
        confidence_hint: float,
        constraint: Constraint | None = None,
    ) -> SharedTangentOptimizationResult:
        if segment_a.type == "line":
            line_segment = segment_a
            arc_segment = segment_b
            line_is_first = True
        else:
            line_segment = segment_b
            arc_segment = segment_a
            line_is_first = False

        line_outward = self._line_outward_tangent(line_segment, anchor)
        if line_outward is None:
            return self._failure(segment_a, segment_b, "invalid line tangent", constraint=constraint)

        desired_arc_outward = PrecisionUtility.normalize_vector((-line_outward[0], -line_outward[1]))
        current_arc_outward = self._arc_outward_tangent(arc_segment, anchor)
        if desired_arc_outward is None or current_arc_outward is None:
            return self._failure(segment_a, segment_b, "invalid arc tangent", constraint=constraint)

        shared_point = self._shared_anchor_point(line_segment, arc_segment, anchor)
        if shared_point is None:
            return self._failure(segment_a, segment_b, "missing shared anchor point", constraint=constraint)

        optimized_point = self._point_for_arc_tangent(arc_segment, anchor, desired_arc_outward)
        if optimized_point is None:
            return self._failure(segment_a, segment_b, "unable to derive shared tangent point", constraint=constraint)

        optimized_line = self._updated_line(line_segment, anchor, optimized_point)
        optimized_arc = self._updated_arc(arc_segment, anchor, optimized_point)
        optimized_line_outward = self._line_outward_tangent(optimized_line, anchor)
        optimized_arc_outward = self._arc_outward_tangent(optimized_arc, anchor)
        if optimized_line_outward is None or optimized_arc_outward is None:
            return self._failure(segment_a, segment_b, "optimized tangents are invalid", constraint=constraint)

        tangent_mismatch = self._tangent_mismatch(optimized_line_outward, optimized_arc_outward)
        movement_penalty = PrecisionUtility.distance_between_points(shared_point, optimized_point)
        fit_error = self._fit_error(points, optimized_line, optimized_arc)
        violation = tangent_mismatch + (movement_penalty * 0.05)
        confidence = self._score_confidence(
            confidence_hint=confidence_hint,
            tangent_mismatch=tangent_mismatch,
            movement_penalty=movement_penalty,
            fit_error=fit_error,
        )

        if tangent_mismatch > self.max_tangent_mismatch or confidence < self.min_confidence:
            return self._failure(
                segment_a,
                segment_b,
                "low confidence optimization",
                constraint=constraint,
                violation=violation,
                confidence=confidence,
                fit_error=fit_error,
                tangent_mismatch=tangent_mismatch,
                movement_penalty=movement_penalty,
            )

        if line_is_first:
            segment_a_result = optimized_line
            segment_b_result = optimized_arc
        else:
            segment_a_result = optimized_arc
            segment_b_result = optimized_line

        return SharedTangentOptimizationResult(
            success=True,
            segment_a=segment_a_result,
            segment_b=segment_b_result,
            shared_tangent=desired_arc_outward,
            violation=violation,
            confidence=confidence,
            fit_error=fit_error,
            tangent_mismatch=tangent_mismatch,
            movement_penalty=movement_penalty,
            reason="optimized shared tangent",
            constraint_id=constraint.constraint_id if constraint is not None else None,
        )

    def _optimize_arc_bezier_pair(
        self,
        segment_a: Segment,
        segment_b: Segment,
        anchor: Anchor,
        points: tuple[Point, ...],
        *,
        confidence_hint: float,
        constraint: Constraint | None = None,
    ) -> SharedTangentOptimizationResult:
        if segment_a.type == "arc":
            arc_segment = segment_a
            bezier_segment = segment_b
            arc_is_first = True
        else:
            arc_segment = segment_b
            bezier_segment = segment_a
            arc_is_first = False

        current_arc_outward = self._arc_outward_tangent(arc_segment, anchor)
        current_bezier_outward = self._bezier_outward_tangent(bezier_segment, anchor)
        if current_arc_outward is None or current_bezier_outward is None:
            return self._failure(segment_a, segment_b, "invalid bezier or arc tangent", constraint=constraint)

        desired_bezier_outward = PrecisionUtility.normalize_vector((-current_arc_outward[0], -current_arc_outward[1]))
        if desired_bezier_outward is None:
            return self._failure(segment_a, segment_b, "invalid desired bezier tangent", constraint=constraint)

        optimized_bezier = self._updated_bezier(bezier_segment, anchor, desired_bezier_outward)
        optimized_arc = arc_segment
        optimized_bezier_outward = self._bezier_outward_tangent(optimized_bezier, anchor)
        optimized_arc_outward = self._arc_outward_tangent(optimized_arc, anchor)
        if optimized_bezier_outward is None or optimized_arc_outward is None:
            return self._failure(segment_a, segment_b, "optimized tangents are invalid", constraint=constraint)

        tangent_mismatch = self._tangent_mismatch(optimized_bezier_outward, optimized_arc_outward)
        movement_penalty = self._bezier_handle_movement_penalty(bezier_segment, optimized_bezier, anchor)
        fit_error = self._fit_error(points, optimized_arc, optimized_bezier)
        violation = tangent_mismatch + (movement_penalty * 0.05)
        confidence = self._score_confidence(
            confidence_hint=confidence_hint,
            tangent_mismatch=tangent_mismatch,
            movement_penalty=movement_penalty,
            fit_error=fit_error,
        )

        if tangent_mismatch > self.max_tangent_mismatch or confidence < self.min_confidence:
            return self._failure(
                segment_a,
                segment_b,
                "low confidence optimization",
                constraint=constraint,
                violation=violation,
                confidence=confidence,
                fit_error=fit_error,
                tangent_mismatch=tangent_mismatch,
                movement_penalty=movement_penalty,
            )

        if arc_is_first:
            segment_a_result = optimized_arc
            segment_b_result = optimized_bezier
        else:
            segment_a_result = optimized_bezier
            segment_b_result = optimized_arc

        return SharedTangentOptimizationResult(
            success=True,
            segment_a=segment_a_result,
            segment_b=segment_b_result,
            shared_tangent=desired_bezier_outward,
            violation=violation,
            confidence=confidence,
            fit_error=fit_error,
            tangent_mismatch=tangent_mismatch,
            movement_penalty=movement_penalty,
            reason="optimized shared tangent",
            constraint_id=constraint.constraint_id if constraint is not None else None,
        )

    def _constraint_confidence(self, constraint: Constraint | None) -> float:
        if constraint is None:
            return 1.0
        if constraint.confidence is not None:
            return float(constraint.confidence)
        return float(constraint.strength)

    def _failure(
        self,
        segment_a: Segment,
        segment_b: Segment,
        reason: str,
        *,
        constraint: Constraint | None = None,
        violation: float = math.inf,
        confidence: float = 0.0,
        fit_error: float = math.inf,
        tangent_mismatch: float = math.inf,
        movement_penalty: float = math.inf,
    ) -> SharedTangentOptimizationResult:
        return SharedTangentOptimizationResult(
            success=False,
            segment_a=segment_a,
            segment_b=segment_b,
            shared_tangent=None,
            violation=violation,
            confidence=confidence,
            fit_error=fit_error,
            tangent_mismatch=tangent_mismatch,
            movement_penalty=movement_penalty,
            reason=reason,
            constraint_id=constraint.constraint_id if constraint is not None else None,
        )

    def _line_outward_tangent(self, segment: Segment, anchor: Anchor) -> Point | None:
        return _line_outward_tangent(segment, anchor)

    def _arc_outward_tangent(self, segment: Segment, anchor: Anchor) -> Point | None:
        return _arc_outward_tangent(segment, anchor)

    def _bezier_outward_tangent(self, segment: Segment, anchor: Anchor) -> Point | None:
        return _bezier_outward_tangent(segment, anchor)

    def _shared_anchor_point(self, line_segment: Segment, arc_segment: Segment, anchor: Anchor) -> Point | None:
        line_point = self._segment_anchor_point(line_segment, anchor)
        if line_point is not None:
            return line_point
        return self._segment_anchor_point(arc_segment, anchor)

    def _segment_anchor_point(self, segment: Segment, anchor: Anchor) -> Point | None:
        if segment.type == "line":
            if segment.anchors and segment.anchors[0] == anchor.anchor_id:
                return self._coerce_point(segment.params.get("start"))
            if segment.anchors and segment.anchors[-1] == anchor.anchor_id:
                return self._coerce_point(segment.params.get("end"))
        if segment.type == "arc":
            center = (
                self._coerce_float(segment.params.get("cx")),
                self._coerce_float(segment.params.get("cy")),
            )
            radius = self._coerce_positive_float(segment.params.get("r"))
            if center[0] is None or center[1] is None or radius is None:
                return None
            if segment.anchors and segment.anchors[0] == anchor.anchor_id:
                angle = self._coerce_float(segment.params.get("start_angle"))
            elif segment.anchors and segment.anchors[-1] == anchor.anchor_id:
                angle = self._coerce_float(segment.params.get("end_angle"))
            else:
                return None
            if angle is None:
                return None
            return (
                float(center[0]) + (radius * math.cos(angle)),
                float(center[1]) + (radius * math.sin(angle)),
            )
        return None

    def _point_for_arc_tangent(self, segment: Segment, anchor: Anchor, desired_tangent: Point) -> Point | None:
        center_x = self._coerce_float(segment.params.get("cx"))
        center_y = self._coerce_float(segment.params.get("cy"))
        radius = self._coerce_positive_float(segment.params.get("r"))
        if center_x is None or center_y is None or radius is None:
            return None

        tangent = PrecisionUtility.normalize_vector(desired_tangent)
        if tangent is None:
            return None

        direction = str(segment.params.get("direction", "ccw")).lower()
        at_start = bool(segment.anchors and segment.anchors[0] == anchor.anchor_id)
        if (at_start and direction == "ccw") or ((not at_start) and direction == "cw"):
            radial = self._rotate_cw(tangent)
        else:
            radial = self._rotate_ccw(tangent)

        return (
            center_x + (radius * radial[0]),
            center_y + (radius * radial[1]),
        )

    def _updated_line(self, segment: Segment, anchor: Anchor, shared_point: Point) -> Segment:
        params = dict(segment.params)
        if segment.anchors and segment.anchors[0] == anchor.anchor_id:
            params["start"] = [shared_point[0], shared_point[1]]
        else:
            params["end"] = [shared_point[0], shared_point[1]]
        return updated(segment, params=params)

    def _updated_arc(self, segment: Segment, anchor: Anchor, shared_point: Point) -> Segment:
        params = dict(segment.params)
        center_x = float(params["cx"])
        center_y = float(params["cy"])
        angle = math.atan2(shared_point[1] - center_y, shared_point[0] - center_x)

        if segment.anchors and segment.anchors[0] == anchor.anchor_id:
            params["start_angle"] = angle
            params["start"] = [shared_point[0], shared_point[1]]
        else:
            params["end_angle"] = angle
            params["end"] = [shared_point[0], shared_point[1]]
        return updated(segment, params=params)

    def _updated_bezier(self, segment: Segment, anchor: Anchor, desired_outward: Point) -> Segment:
        params = dict(segment.params)
        start = self._coerce_point(params.get("start"))
        end = self._coerce_point(params.get("end"))
        control1 = self._coerce_point(params.get("control1"))
        control2 = self._coerce_point(params.get("control2"))
        if start is None or end is None or control1 is None or control2 is None:
            return segment

        desired_unit = PrecisionUtility.normalize_vector(desired_outward)
        if desired_unit is None:
            return segment

        if segment.anchors and segment.anchors[0] == anchor.anchor_id:
            handle_length = PrecisionUtility.distance_between_points(start, control1)
            if PrecisionUtility.near_zero(handle_length):
                handle_length = max(
                    PrecisionUtility.distance_between_points(start, end) * 0.25,
                    1.0,
                )
            params["control1"] = [
                start[0] + (desired_unit[0] * handle_length),
                start[1] + (desired_unit[1] * handle_length),
            ]
        elif segment.anchors and segment.anchors[-1] == anchor.anchor_id:
            handle_length = PrecisionUtility.distance_between_points(end, control2)
            if PrecisionUtility.near_zero(handle_length):
                handle_length = max(
                    PrecisionUtility.distance_between_points(start, end) * 0.25,
                    1.0,
                )
            params["control2"] = [
                end[0] + (desired_unit[0] * handle_length),
                end[1] + (desired_unit[1] * handle_length),
            ]
        return updated(segment, params=params)

    def _tangent_mismatch(self, line_outward: Point, arc_outward: Point) -> float:
        return _tangent_mismatch(line_outward, arc_outward)

    def _bezier_handle_movement_penalty(self, before: Segment, after: Segment, anchor: Anchor) -> float:
        if before.anchors and before.anchors[0] == anchor.anchor_id:
            anchor_point = self._coerce_point(before.params.get("start"))
            opposite_point = self._coerce_point(before.params.get("end"))
            old_handle = self._coerce_point(before.params.get("control1"))
            new_handle = self._coerce_point(after.params.get("control1"))
        else:
            anchor_point = self._coerce_point(before.params.get("end"))
            opposite_point = self._coerce_point(before.params.get("start"))
            old_handle = self._coerce_point(before.params.get("control2"))
            new_handle = self._coerce_point(after.params.get("control2"))
        if old_handle is None or new_handle is None or anchor_point is None or opposite_point is None:
            return math.inf

        handle_move = PrecisionUtility.distance_between_points(old_handle, new_handle)
        chord_length = PrecisionUtility.distance_between_points(anchor_point, opposite_point)
        if PrecisionUtility.near_zero(chord_length):
            return handle_move
        return handle_move / chord_length

    def _fit_error(self, points: tuple[Point, ...], first_segment: Segment, second_segment: Segment) -> float:
        if not points:
            return 0.0

        samplers = (self._sample_segment(first_segment), self._sample_segment(second_segment))
        if samplers[0] is None or samplers[1] is None:
            return math.inf

        errors = []
        for point in points:
            errors.append(
                min(
                    self._point_to_polyline_distance(point, samplers[0]),
                    self._point_to_polyline_distance(point, samplers[1]),
                )
            )
        return sum(errors) / len(errors)

    def _score_confidence(
        self,
        *,
        confidence_hint: float,
        tangent_mismatch: float,
        movement_penalty: float,
        fit_error: float,
    ) -> float:
        mismatch_factor = max(0.0, 1.0 - (tangent_mismatch / max(self.max_tangent_mismatch, 1e-9)))
        movement_factor = max(0.0, 1.0 - (movement_penalty / max(self.movement_penalty_scale, 1e-9)))
        fit_factor = 1.0 / (1.0 + max(0.0, fit_error))
        return max(0.0, min(1.0, confidence_hint * mismatch_factor * movement_factor * fit_factor))

    def _line_coefficients(self, segment: Segment) -> tuple[float, float, float] | None:
        start = self._coerce_point(segment.params.get("start"))
        end = self._coerce_point(segment.params.get("end"))
        if start is None or end is None:
            return None
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if PrecisionUtility.near_zero(length):
            return None
        a = dy / length
        b = -dx / length
        c = -((a * start[0]) + (b * start[1]))
        return (a, b, c)

    def _circle_params(self, segment: Segment) -> tuple[float, float, float] | None:
        center_x = self._coerce_float(segment.params.get("cx"))
        center_y = self._coerce_float(segment.params.get("cy"))
        radius = self._coerce_positive_float(segment.params.get("r"))
        if center_x is None or center_y is None or radius is None:
            return None
        return (center_x, center_y, radius)

    def _sample_segment(self, segment: Segment) -> tuple[Point, ...] | None:
        if segment.type == "line":
            start = self._coerce_point(segment.params.get("start"))
            end = self._coerce_point(segment.params.get("end"))
            if start is None or end is None:
                return None
            return (start, end)
        if segment.type == "arc":
            center = self._circle_params(segment)
            start_angle = self._coerce_float(segment.params.get("start_angle"))
            end_angle = self._coerce_float(segment.params.get("end_angle"))
            direction = str(segment.params.get("direction", "ccw")).lower()
            if center is None or start_angle is None or end_angle is None:
                return None
            span = (end_angle - start_angle) % (2.0 * math.pi)
            if direction == "cw":
                span = -((start_angle - end_angle) % (2.0 * math.pi))
            steps = 12
            samples = []
            for index in range(steps + 1):
                t = index / steps
                angle = start_angle + (span * t)
                samples.append(
                    (
                        center[0] + (center[2] * math.cos(angle)),
                        center[1] + (center[2] * math.sin(angle)),
                    )
                )
            return tuple(samples)
        if segment.type == "bezier":
            start = self._coerce_point(segment.params.get("start"))
            control1 = self._coerce_point(segment.params.get("control1"))
            control2 = self._coerce_point(segment.params.get("control2"))
            end = self._coerce_point(segment.params.get("end"))
            if start is None or control1 is None or control2 is None or end is None:
                return None
            samples = []
            steps = 12
            for index in range(steps + 1):
                t = index / steps
                one_minus_t = 1.0 - t
                x = (
                    (one_minus_t ** 3) * start[0]
                    + 3.0 * (one_minus_t ** 2) * t * control1[0]
                    + 3.0 * one_minus_t * (t ** 2) * control2[0]
                    + (t ** 3) * end[0]
                )
                y = (
                    (one_minus_t ** 3) * start[1]
                    + 3.0 * (one_minus_t ** 2) * t * control1[1]
                    + 3.0 * one_minus_t * (t ** 2) * control2[1]
                    + (t ** 3) * end[1]
                )
                samples.append((x, y))
            return tuple(samples)
        return None

    def _point_to_polyline_distance(self, point: Point, polyline: tuple[Point, ...]) -> float:
        if len(polyline) < 2:
            return math.inf
        return min(
            self._point_to_segment_distance(point, polyline[index - 1], polyline[index])
            for index in range(1, len(polyline))
        )

    def _point_to_segment_distance(self, point: Point, start: Point, end: Point) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length_sq = (dx * dx) + (dy * dy)
        if PrecisionUtility.near_zero(segment_length_sq):
            return PrecisionUtility.distance_between_points(point, start)
        projection = (((point[0] - start[0]) * dx) + ((point[1] - start[1]) * dy)) / segment_length_sq
        clamped = max(0.0, min(1.0, projection))
        projected_point = (start[0] + (clamped * dx), start[1] + (clamped * dy))
        return PrecisionUtility.distance_between_points(point, projected_point)

    def _coerce_point(self, value: object) -> Point | None:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_positive_float(self, value: object) -> float | None:
        result = self._coerce_float(value)
        if result is None or result <= 0.0:
            return None
        return result

    def _rotate_cw(self, vector: Point) -> Point:
        return (vector[1], -vector[0])

    def _rotate_ccw(self, vector: Point) -> Point:
        return (-vector[1], vector[0])


def _line_outward_tangent(segment: Segment, anchor: Anchor) -> Point | None:
    start = _coerce_point(segment.params.get("start"))
    end = _coerce_point(segment.params.get("end"))
    if start is None or end is None:
        return None

    if segment.anchors and segment.anchors[0] == anchor.anchor_id:
        return (end[0] - start[0], end[1] - start[1])
    if segment.anchors and segment.anchors[-1] == anchor.anchor_id:
        return (start[0] - end[0], start[1] - end[1])
    return None


def _arc_outward_tangent(segment: Segment, anchor: Anchor) -> Point | None:
    direction = str(segment.params.get("direction", "ccw")).lower()
    angle_value = None
    if segment.anchors and segment.anchors[0] == anchor.anchor_id:
        angle_value = segment.params.get("start_angle")
        at_start = True
    elif segment.anchors and segment.anchors[-1] == anchor.anchor_id:
        angle_value = segment.params.get("end_angle")
        at_start = False
    else:
        return None

    angle = _coerce_float(angle_value)
    radius = _coerce_positive_float(segment.params.get("r"))
    if angle is None or radius is None:
        return None

    if at_start:
        return (math.sin(angle), -math.cos(angle)) if direction == "cw" else (-math.sin(angle), math.cos(angle))
    return (-math.sin(angle), math.cos(angle)) if direction == "cw" else (math.sin(angle), -math.cos(angle))


def _bezier_outward_tangent(segment: Segment, anchor: Anchor) -> Point | None:
    start = _coerce_point(segment.params.get("start"))
    end = _coerce_point(segment.params.get("end"))
    control1 = _coerce_point(segment.params.get("control1"))
    control2 = _coerce_point(segment.params.get("control2"))
    if start is None or end is None or control1 is None or control2 is None:
        return None

    if segment.anchors and segment.anchors[0] == anchor.anchor_id:
        return (control1[0] - start[0], control1[1] - start[1])
    if segment.anchors and segment.anchors[-1] == anchor.anchor_id:
        return (control2[0] - end[0], control2[1] - end[1])
    return None


def _tangent_mismatch(first_outward: Point, second_outward: Point) -> float:
    first_unit = PrecisionUtility.normalize_vector(first_outward)
    second_unit = PrecisionUtility.normalize_vector(second_outward)
    if first_unit is None or second_unit is None:
        return math.inf
    first_angle = math.atan2(first_unit[1], first_unit[0])
    second_angle = math.atan2(second_unit[1], second_unit[0]) + math.pi
    delta = (first_angle - second_angle + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def _vector_to_shared_tangent_error(vector: Point, shared_tangent: Point) -> float:
    vector_unit = PrecisionUtility.normalize_vector(vector)
    if vector_unit is None:
        return math.inf

    vector_angle = math.atan2(vector_unit[1], vector_unit[0])
    shared_angle = math.atan2(shared_tangent[1], shared_tangent[0])
    opposite_angle = shared_angle + math.pi
    delta_direct = abs((vector_angle - shared_angle + math.pi) % (2.0 * math.pi) - math.pi)
    delta_opposite = abs((vector_angle - opposite_angle + math.pi) % (2.0 * math.pi) - math.pi)
    return min(delta_direct, delta_opposite)


def _coerce_point(value: object) -> Point | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_float(value: object) -> float | None:
    result = _coerce_float(value)
    if result is None or result <= 0.0:
        return None
    return result


__all__ = [
    "SharedTangentOptimizationResult",
    "SharedTangentOptimizer",
    "segment_outward_tangent",
    "shared_tangent_violation",
]
