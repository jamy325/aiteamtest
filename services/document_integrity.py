from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from core.precision import PrecisionUtility
from core.types import Point, Segment, VectorDocument


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    code: str
    message: str
    affected_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    success: bool
    errors: tuple[IntegrityIssue, ...]
    warnings: tuple[IntegrityIssue, ...]
    affected_ids: tuple[str, ...]


class DocumentIntegrityValidator:
    def __init__(self, *, epsilon: float | None = None) -> None:
        self.epsilon = PrecisionUtility.EPSILON if epsilon is None else float(epsilon)

    def validate(self, document: VectorDocument) -> IntegrityReport:
        errors: list[IntegrityIssue] = []
        warnings: list[IntegrityIssue] = []

        path_ids = {path.path_id for path in document.paths}
        object_ids = {obj.object_id for obj in document.objects}
        segment_ids = {segment.segment_id for segment in document.segments}
        anchor_ids = {anchor.anchor_id for anchor in document.anchors}
        referenced_segment_ids = {segment_id for path in document.paths for segment_id in path.segments}
        referenced_anchor_ids = {anchor_id for segment in document.segments for anchor_id in segment.anchors}

        if document.coordinate_system.internal_space != "vector":
            errors.append(
                IntegrityIssue(
                    code="NON_VECTOR_COORDINATE_SPACE",
                    message="document.coordinate_system.internal_space must be vector",
                    affected_ids=(document.document_id,),
                )
            )

        segment_by_id = {segment.segment_id: segment for segment in document.segments}

        for path in document.paths:
            missing_segments = tuple(segment_id for segment_id in path.segments if segment_id not in segment_by_id)
            if missing_segments:
                errors.append(
                    IntegrityIssue(
                        code="MISSING_PATH_SEGMENT_REFERENCE",
                        message=f"path {path.path_id} references missing segment(s)",
                        affected_ids=(path.path_id,) + missing_segments,
                    )
                )
            if path.closed:
                closed_error = self._closed_path_issue(path, segment_by_id)
                if closed_error is not None:
                    errors.append(closed_error)

        for segment in document.segments:
            missing_anchors = tuple(anchor_id for anchor_id in segment.anchors if anchor_id not in anchor_ids)
            if missing_anchors:
                errors.append(
                    IntegrityIssue(
                        code="MISSING_SEGMENT_ANCHOR_REFERENCE",
                        message=f"segment {segment.segment_id} references missing anchor(s)",
                        affected_ids=(segment.segment_id,) + missing_anchors,
                    )
                )

            angle_issue = self._angle_contract_issue(segment)
            if angle_issue is not None:
                errors.append(angle_issue)

        for segment in document.segments:
            if segment.segment_id not in referenced_segment_ids and not bool(segment.metadata.get("orphan_allowed")):
                errors.append(
                    IntegrityIssue(
                        code="DANGLING_SEGMENT",
                        message=f"segment {segment.segment_id} is not referenced by any path",
                        affected_ids=(segment.segment_id,),
                    )
                )

        for anchor in document.anchors:
            if anchor.anchor_id not in referenced_anchor_ids and not bool(anchor.metadata.get("orphan_allowed")):
                errors.append(
                    IntegrityIssue(
                        code="DANGLING_ANCHOR",
                        message=f"anchor {anchor.anchor_id} is not referenced by any segment",
                        affected_ids=(anchor.anchor_id,),
                    )
                )

        valid_target_ids = object_ids | path_ids | segment_ids | anchor_ids
        for constraint in document.constraints:
            external_targets = {str(item) for item in constraint.metadata.get("external_targets", ())}
            invalid_targets = {str(item) for item in constraint.metadata.get("invalid_targets", ())}
            missing_targets = tuple(
                target
                for target in constraint.targets
                if target not in valid_target_ids and target not in external_targets and target not in invalid_targets
            )
            if missing_targets:
                errors.append(
                    IntegrityIssue(
                        code="DANGLING_CONSTRAINT_TARGET",
                        message=f"constraint {constraint.constraint_id} references missing target(s)",
                        affected_ids=(constraint.constraint_id,) + missing_targets,
                    )
                )

        affected_ids = tuple(
            dict.fromkeys(
                affected_id
                for issue in errors + warnings
                for affected_id in issue.affected_ids
            )
        )
        return IntegrityReport(
            success=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            affected_ids=affected_ids,
        )

    def _closed_path_issue(
        self,
        path: Any,
        segment_by_id: dict[str, Segment],
    ) -> IntegrityIssue | None:
        if not path.segments:
            return IntegrityIssue(
                code="CLOSED_PATH_EMPTY",
                message=f"closed path {path.path_id} has no segments",
                affected_ids=(path.path_id,),
            )

        segments = [segment_by_id[segment_id] for segment_id in path.segments if segment_id in segment_by_id]
        if not segments:
            return IntegrityIssue(
                code="CLOSED_PATH_MISSING_SEGMENTS",
                message=f"closed path {path.path_id} has no resolvable segments",
                affected_ids=(path.path_id,),
            )

        endpoints_connected = self._path_endpoints_connected(segments)
        gap_is_small = float(path.max_gap) <= self.epsilon
        topology_is_closed = path.topology_status == "closed"

        if endpoints_connected and gap_is_small and not topology_is_closed:
            return IntegrityIssue(
                code="CLOSED_PATH_STATUS_MISMATCH",
                message=f"closed path {path.path_id} is geometrically closed but topology_status is not closed",
                affected_ids=(path.path_id,),
            )
        if (not endpoints_connected or not gap_is_small) and topology_is_closed:
            return IntegrityIssue(
                code="CLOSED_PATH_GAP_MISMATCH",
                message=f"closed path {path.path_id} has a closing gap inconsistent with topology_status=closed",
                affected_ids=(path.path_id,),
            )
        return None

    def _path_endpoints_connected(self, segments: list[Segment]) -> bool:
        if len(segments) == 1 and segments[0].type in {"circle", "ellipse"}:
            return True
        first_start = self._segment_start(segments[0])
        last_end = self._segment_end(segments[-1])
        if first_start is None or last_end is None:
            return False
        return PrecisionUtility.points_close(first_start, last_end, epsilon=self.epsilon)

    def _segment_start(self, segment: Segment) -> Point | None:
        if "start" in segment.params:
            return self._coerce_point(segment.params.get("start"))
        if segment.type == "polyline":
            points = segment.params.get("points")
            if isinstance(points, list) and points:
                return self._coerce_point(points[0])
        if segment.type == "arc":
            return self._arc_endpoint(segment, start=True)
        if segment.type in {"circle", "ellipse"}:
            return (0.0, 0.0)
        return None

    def _segment_end(self, segment: Segment) -> Point | None:
        if "end" in segment.params:
            return self._coerce_point(segment.params.get("end"))
        if segment.type == "polyline":
            points = segment.params.get("points")
            if isinstance(points, list) and points:
                return self._coerce_point(points[-1])
        if segment.type == "arc":
            return self._arc_endpoint(segment, start=False)
        if segment.type in {"circle", "ellipse"}:
            return (0.0, 0.0)
        return None

    def _arc_endpoint(self, segment: Segment, *, start: bool) -> Point | None:
        try:
            cx = float(segment.params["cx"])
            cy = float(segment.params["cy"])
            r = abs(float(segment.params["r"]))
            angle = float(segment.params["start_angle" if start else "end_angle"])
        except (KeyError, TypeError, ValueError):
            return None
        return (cx + r * math.cos(angle), cy + r * math.sin(angle))

    def _angle_contract_issue(self, segment: Segment) -> IntegrityIssue | None:
        angle_unit = str(segment.params.get("angle_unit", segment.metadata.get("angle_unit", "radian"))).strip().lower()
        if angle_unit in {"degree", "degrees"}:
            return IntegrityIssue(
                code="ANGLE_UNIT_NOT_RADIANS",
                message=f"segment {segment.segment_id} uses degree angles; internal contract requires radians",
                affected_ids=(segment.segment_id,),
            )

        if segment.type == "arc":
            if "start_angle" not in segment.params or "end_angle" not in segment.params:
                return IntegrityIssue(
                    code="ARC_ANGLE_FIELDS_MISSING",
                    message=f"arc segment {segment.segment_id} is missing start_angle/end_angle",
                    affected_ids=(segment.segment_id,),
                )
            for field_name in ("start_angle", "end_angle"):
                issue = self._angle_range_issue(segment, field_name)
                if issue is not None:
                    return issue

        if segment.type == "ellipse" and "rotation" in segment.params:
            return self._angle_range_issue(segment, "rotation")
        return None

    def _angle_range_issue(self, segment: Segment, field_name: str) -> IntegrityIssue | None:
        try:
            angle_value = float(segment.params[field_name])
        except (KeyError, TypeError, ValueError):
            return IntegrityIssue(
                code="ANGLE_FIELD_INVALID",
                message=f"segment {segment.segment_id} has invalid {field_name}",
                affected_ids=(segment.segment_id,),
            )
        if not math.isfinite(angle_value) or abs(angle_value) > (math.tau * 8.0):
            return IntegrityIssue(
                code="ANGLE_RANGE_IMPLAUSIBLE",
                message=f"segment {segment.segment_id} has implausible radian value for {field_name}",
                affected_ids=(segment.segment_id,),
            )
        return None

    def _coerce_point(self, value: object) -> Point | None:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None


__all__ = [
    "DocumentIntegrityValidator",
    "IntegrityIssue",
    "IntegrityReport",
]
