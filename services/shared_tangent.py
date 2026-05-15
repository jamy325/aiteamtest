from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from core.precision import PrecisionUtility
from core.types import Anchor, Constraint, Point, Segment, VectorDocument, updated


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
        if pair_types != {"line", "arc"}:
            return self._failure(segment_a, segment_b, "unsupported segment pair", constraint=constraint)

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
        fit_error = self._fit_error(tuple(points), optimized_line, optimized_arc)
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
        start = self._coerce_point(segment.params.get("start"))
        end = self._coerce_point(segment.params.get("end"))
        if start is None or end is None:
            return None

        if segment.anchors and segment.anchors[0] == anchor.anchor_id:
            return (end[0] - start[0], end[1] - start[1])
        if segment.anchors and segment.anchors[-1] == anchor.anchor_id:
            return (start[0] - end[0], start[1] - end[1])
        return None

    def _arc_outward_tangent(self, segment: Segment, anchor: Anchor) -> Point | None:
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

        angle = self._coerce_float(angle_value)
        radius = self._coerce_positive_float(segment.params.get("r"))
        if angle is None or radius is None:
            return None

        if at_start:
            return (math.sin(angle), -math.cos(angle)) if direction == "cw" else (-math.sin(angle), math.cos(angle))
        return (-math.sin(angle), math.cos(angle)) if direction == "cw" else (math.sin(angle), -math.cos(angle))

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

    def _tangent_mismatch(self, line_outward: Point, arc_outward: Point) -> float:
        line_unit = PrecisionUtility.normalize_vector(line_outward)
        arc_unit = PrecisionUtility.normalize_vector(arc_outward)
        if line_unit is None or arc_unit is None:
            return math.inf
        line_angle = math.atan2(line_unit[1], line_unit[0])
        arc_angle = math.atan2(arc_unit[1], arc_unit[0]) + math.pi
        delta = (line_angle - arc_angle + math.pi) % (2.0 * math.pi) - math.pi
        return abs(delta)

    def _fit_error(self, points: tuple[Point, ...], line_segment: Segment, arc_segment: Segment) -> float:
        if not points:
            return 0.0

        line = self._line_coefficients(line_segment)
        circle = self._circle_params(arc_segment)
        if line is None or circle is None:
            return math.inf

        errors = []
        for point in points:
            line_error = abs((line[0] * point[0]) + (line[1] * point[1]) + line[2])
            circle_error = abs(math.hypot(point[0] - circle[0], point[1] - circle[1]) - circle[2])
            errors.append(min(line_error, circle_error))
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


__all__ = ["SharedTangentOptimizationResult", "SharedTangentOptimizer"]
