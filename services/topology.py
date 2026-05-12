from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.precision import PrecisionUtility
from core.types import Anchor, Path, Point, Segment, VectorDocument, updated
from services.segment_rigidity import MovementStrategy, SegmentRigidityPolicy

TopologyStatus = Literal["open", "closed", "topology_error"]


@dataclass(frozen=True, slots=True)
class PathClosingConfig:
    gap_epsilon: float = 1e-6
    auto_snap_distance: float = 0.5


@dataclass(frozen=True, slots=True)
class PathGapCorrection:
    left_segment_id: str
    right_segment_id: str
    distance: float
    closing_gap: bool
    corrected: bool
    topology_error: bool
    strategy: MovementStrategy | None
    reason: str
    moved_anchor_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PathTopologyResult:
    document: VectorDocument
    path_id: str
    topology_status: TopologyStatus
    max_gap: float
    corrections: tuple[PathGapCorrection, ...]
    topology_error: bool


class TopologyEngine:
    def __init__(
        self,
        config: PathClosingConfig | None = None,
        rigidity_policy: SegmentRigidityPolicy | None = None,
    ) -> None:
        self.config = config or PathClosingConfig()
        self.rigidity_policy = rigidity_policy or SegmentRigidityPolicy()

    def enforce_path_topology(self, document: VectorDocument, path_id: str) -> PathTopologyResult:
        path_index = self._find_path_index(document, path_id)
        if path_index is None:
            raise ValueError(f"unknown path_id: {path_id}")

        path = document.paths[path_index]
        segment_ids = tuple(path.segments)
        if len(segment_ids) < 2:
            topology_status: TopologyStatus = "closed" if path.closed else "open"
            updated_document = self._replace_path(document, path_index, updated(path, topology_status=topology_status, max_gap=0.0))
            return PathTopologyResult(
                document=updated_document,
                path_id=path_id,
                topology_status=topology_status,
                max_gap=0.0,
                corrections=(),
                topology_error=False,
            )

        current_document = document
        corrections: list[PathGapCorrection] = []
        max_gap = 0.0
        topology_error = False

        pair_count = len(segment_ids) if path.closed else len(segment_ids) - 1
        for pair_index in range(pair_count):
            current_path = self._path_by_id(current_document, path_id)
            left_segment = self._segment_by_id(current_document, current_path.segments[pair_index])
            right_segment = self._segment_by_id(current_document, current_path.segments[(pair_index + 1) % len(current_path.segments)])
            closing_gap = path.closed and pair_index == len(segment_ids) - 1
            current_document, correction = self._enforce_gap(current_document, left_segment, right_segment, closing_gap=closing_gap)
            max_gap = max(max_gap, correction.distance)
            topology_error = topology_error or correction.topology_error
            if correction.distance > self.config.gap_epsilon:
                corrections.append(correction)

        topology_status = "topology_error" if topology_error else ("closed" if path.closed else "open")
        final_path = updated(self._path_by_id(current_document, path_id), topology_status=topology_status, max_gap=max_gap)
        current_document = self._replace_path(current_document, self._find_path_index(current_document, path_id), final_path)
        return PathTopologyResult(
            document=current_document,
            path_id=path_id,
            topology_status=topology_status,
            max_gap=max_gap,
            corrections=tuple(corrections),
            topology_error=topology_error,
        )

    def _enforce_gap(
        self,
        document: VectorDocument,
        left_segment: Segment,
        right_segment: Segment,
        *,
        closing_gap: bool,
    ) -> tuple[VectorDocument, PathGapCorrection]:
        left_end = self._segment_endpoint(left_segment, "end")
        right_start = self._segment_endpoint(right_segment, "start")
        distance = PrecisionUtility.distance_between_points(left_end, right_start)

        if PrecisionUtility.points_close(left_end, right_start, epsilon=self.config.gap_epsilon):
            return (
                document,
                PathGapCorrection(
                    left_segment_id=left_segment.segment_id,
                    right_segment_id=right_segment.segment_id,
                    distance=distance,
                    closing_gap=closing_gap,
                    corrected=False,
                    topology_error=False,
                    strategy=None,
                    reason="within_gap_epsilon",
                ),
            )

        if PrecisionUtility.compare(distance, self.config.auto_snap_distance, epsilon=self.config.gap_epsilon) == 1:
            return (
                document,
                PathGapCorrection(
                    left_segment_id=left_segment.segment_id,
                    right_segment_id=right_segment.segment_id,
                    distance=distance,
                    closing_gap=closing_gap,
                    corrected=False,
                    topology_error=True,
                    strategy=None,
                    reason="gap_exceeds_auto_snap_distance",
                ),
            )

        decision = self.rigidity_policy.choose_segment_to_move(left_segment, right_segment)
        if decision.blocked:
            return (
                document,
                PathGapCorrection(
                    left_segment_id=left_segment.segment_id,
                    right_segment_id=right_segment.segment_id,
                    distance=distance,
                    closing_gap=closing_gap,
                    corrected=False,
                    topology_error=True,
                    strategy=decision.strategy,
                    reason=decision.reason,
                ),
            )

        midpoint = ((left_end[0] + right_start[0]) / 2.0, (left_end[1] + right_start[1]) / 2.0)
        if decision.strategy == "move_left":
            updated_document, moved_anchor_ids = self._move_segment_end_anchor(document, left_segment, left_end, right_start)
        elif decision.strategy == "move_right":
            updated_document, moved_anchor_ids = self._move_segment_start_anchor(document, right_segment, right_start, left_end)
        elif decision.strategy in {"move_both_midpoint", "move_both_minimal"}:
            updated_document, moved_anchor_ids = self._move_both_gap_anchors(document, left_segment, right_segment, left_end, right_start, midpoint)
        else:
            updated_document = document
            moved_anchor_ids = ()

        return (
            updated_document,
            PathGapCorrection(
                left_segment_id=left_segment.segment_id,
                right_segment_id=right_segment.segment_id,
                distance=distance,
                closing_gap=closing_gap,
                corrected=True,
                topology_error=False,
                strategy=decision.strategy,
                reason=decision.reason,
                moved_anchor_ids=moved_anchor_ids,
            ),
        )

    def _move_segment_end_anchor(
        self,
        document: VectorDocument,
        segment: Segment,
        current_point: Point,
        target_point: Point,
    ) -> tuple[VectorDocument, tuple[str, ...]]:
        return self._move_anchor_or_endpoint(document, segment, len(segment.anchors) - 1, "end", current_point, target_point)

    def _move_segment_start_anchor(
        self,
        document: VectorDocument,
        segment: Segment,
        current_point: Point,
        target_point: Point,
    ) -> tuple[VectorDocument, tuple[str, ...]]:
        return self._move_anchor_or_endpoint(document, segment, 0, "start", current_point, target_point)

    def _move_both_gap_anchors(
        self,
        document: VectorDocument,
        left_segment: Segment,
        right_segment: Segment,
        left_end: Point,
        right_start: Point,
        midpoint: Point,
    ) -> tuple[VectorDocument, tuple[str, ...]]:
        current_document, moved_left = self._move_segment_end_anchor(document, left_segment, left_end, midpoint)
        current_document, moved_right = self._move_segment_start_anchor(current_document, right_segment, right_start, midpoint)
        moved_anchor_ids = tuple(dict.fromkeys(moved_left + moved_right))
        return current_document, moved_anchor_ids

    def _move_anchor_or_endpoint(
        self,
        document: VectorDocument,
        segment: Segment,
        anchor_position: int,
        endpoint: Literal["start", "end"],
        current_point: Point,
        target_point: Point,
    ) -> tuple[VectorDocument, tuple[str, ...]]:
        if len(segment.anchors) > anchor_position:
            anchor_id = segment.anchors[anchor_position]
            anchor_index = self._find_anchor_index(document, anchor_id)
            if anchor_index is not None:
                return self._move_anchor(document, anchor_index, target_point), (anchor_id,)
        return self._move_segment_endpoint_only(document, segment.segment_id, endpoint, current_point, target_point), ()

    def _move_anchor(self, document: VectorDocument, anchor_index: int, target_point: Point) -> VectorDocument:
        anchor = document.anchors[anchor_index]
        delta = (target_point[0] - anchor.position[0], target_point[1] - anchor.position[1])
        moved_anchor = updated(
            anchor,
            position=target_point,
            in_handle=self._shift_optional_point(anchor.in_handle, delta),
            out_handle=self._shift_optional_point(anchor.out_handle, delta),
        )

        anchors = list(document.anchors)
        anchors[anchor_index] = moved_anchor

        segments = list(document.segments)
        for index, candidate in enumerate(document.segments):
            updated_segment = candidate
            if candidate.anchors and candidate.anchors[0] == anchor.anchor_id:
                updated_segment = self._move_segment_endpoint(updated_segment, "start", target_point)
            if candidate.anchors and candidate.anchors[-1] == anchor.anchor_id:
                updated_segment = self._move_segment_endpoint(updated_segment, "end", target_point)
            segments[index] = updated_segment

        return updated(document, anchors=tuple(anchors), segments=tuple(segments))

    def _move_segment_endpoint_only(
        self,
        document: VectorDocument,
        segment_id: str,
        endpoint: Literal["start", "end"],
        current_point: Point,
        target_point: Point,
    ) -> VectorDocument:
        segment_index = self._find_segment_index(document, segment_id)
        if segment_index is None:
            raise ValueError(f"unknown segment_id: {segment_id}")
        segments = list(document.segments)
        segments[segment_index] = self._move_segment_endpoint(segments[segment_index], endpoint, target_point)
        return updated(document, segments=tuple(segments))

    def _move_segment_endpoint(self, segment: Segment, endpoint: Literal["start", "end"], target_point: Point) -> Segment:
        current_point = self._segment_endpoint(segment, endpoint)
        delta = (target_point[0] - current_point[0], target_point[1] - current_point[1])
        params = dict(segment.params)
        params[endpoint] = [target_point[0], target_point[1]]
        if endpoint == "start" and "control1" in params:
            params["control1"] = list(self._shift_point(self._coerce_point(params["control1"]), delta))
        if endpoint == "end" and "control2" in params:
            params["control2"] = list(self._shift_point(self._coerce_point(params["control2"]), delta))
        return updated(segment, params=params)

    def _segment_endpoint(self, segment: Segment, endpoint: Literal["start", "end"]) -> Point:
        if endpoint not in segment.params:
            raise ValueError(f"segment {segment.segment_id} is missing {endpoint} point")
        return self._coerce_point(segment.params[endpoint])

    def _coerce_point(self, value: Point | list[float]) -> Point:
        return (float(value[0]), float(value[1]))

    def _shift_optional_point(self, point: Point | None, delta: Point) -> Point | None:
        if point is None:
            return None
        return self._shift_point(point, delta)

    def _shift_point(self, point: Point, delta: Point) -> Point:
        return (point[0] + delta[0], point[1] + delta[1])

    def _replace_path(self, document: VectorDocument, path_index: int | None, path: Path) -> VectorDocument:
        if path_index is None:
            raise ValueError(f"unknown path_id: {path.path_id}")
        paths = list(document.paths)
        paths[path_index] = path
        return updated(document, paths=tuple(paths))

    def _path_by_id(self, document: VectorDocument, path_id: str) -> Path:
        path_index = self._find_path_index(document, path_id)
        if path_index is None:
            raise ValueError(f"unknown path_id: {path_id}")
        return document.paths[path_index]

    def _segment_by_id(self, document: VectorDocument, segment_id: str) -> Segment:
        segment_index = self._find_segment_index(document, segment_id)
        if segment_index is None:
            raise ValueError(f"unknown segment_id: {segment_id}")
        return document.segments[segment_index]

    def _find_path_index(self, document: VectorDocument, path_id: str) -> int | None:
        for index, path in enumerate(document.paths):
            if path.path_id == path_id:
                return index
        return None

    def _find_segment_index(self, document: VectorDocument, segment_id: str) -> int | None:
        for index, segment in enumerate(document.segments):
            if segment.segment_id == segment_id:
                return index
        return None

    def _find_anchor_index(self, document: VectorDocument, anchor_id: str) -> int | None:
        for index, anchor in enumerate(document.anchors):
            if anchor.anchor_id == anchor_id:
                return index
        return None


__all__ = [
    "PathClosingConfig",
    "PathGapCorrection",
    "PathTopologyResult",
    "TopologyEngine",
    "TopologyStatus",
]
