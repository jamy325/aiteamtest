from __future__ import annotations

from dataclasses import dataclass

from core.precision import PrecisionUtility
from core.types import Path, Point, Segment, VectorDocument, updated
from services.segment_sampler import SegmentSampler, SegmentSamplerConfig


@dataclass(frozen=True, slots=True)
class SelfIntersectionConfig:
    epsilon: float = 1e-6
    max_chord_error: float = 0.25
    min_segments_per_arc: int = 8
    max_segments_per_arc: int = 128
    bezier_segments: int = 24


@dataclass(frozen=True, slots=True)
class SelfIntersectionResult:
    document: VectorDocument
    path_id: str
    self_intersection_count: int
    self_intersection_points: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class _SegmentPolyline:
    segment_id: str
    path_segment_index: int
    points: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class _EdgeFragment:
    start: Point
    end: Point
    path_segment_index: int
    fragment_index: int
    fragment_count: int


class SelfIntersectionDetector:
    def __init__(self, config: SelfIntersectionConfig | None = None) -> None:
        self.config = config or SelfIntersectionConfig()
        self._sampler = SegmentSampler(
            SegmentSamplerConfig(
                max_chord_error=self.config.max_chord_error,
                min_segments_per_arc=self.config.min_segments_per_arc,
                max_segments_per_arc=self.config.max_segments_per_arc,
                bezier_segments=self.config.bezier_segments,
            )
        )

    def detect_path_self_intersections(self, document: VectorDocument, path_id: str) -> SelfIntersectionResult:
        path_index = self._find_path_index(document, path_id)
        if path_index is None:
            raise ValueError(f"unknown path_id: {path_id}")

        path = document.paths[path_index]
        segments = self._path_segments(document, path)
        polylines = tuple(
            _SegmentPolyline(
                segment_id=segment.segment_id,
                path_segment_index=index,
                points=self._sample_segment(segment),
            )
            for index, segment in enumerate(segments)
        )
        intersections = self._detect_polyline_intersections(polylines, closed=path.closed)
        updated_document = self._with_intersection_state(document, path_index, path, intersections)
        return SelfIntersectionResult(
            document=updated_document,
            path_id=path_id,
            self_intersection_count=len(intersections),
            self_intersection_points=intersections,
        )

    def _with_intersection_state(
        self,
        document: VectorDocument,
        path_index: int,
        path: Path,
        intersections: tuple[Point, ...],
    ) -> VectorDocument:
        metadata = dict(path.metadata)
        metadata["self_intersection_points"] = [[point[0], point[1]] for point in intersections]
        topology_status = "self_intersected" if intersections else path.topology_status
        updated_path = updated(
            path,
            topology_status=topology_status,
            self_intersection_count=len(intersections),
            metadata=metadata,
        )
        paths = list(document.paths)
        paths[path_index] = updated_path
        return updated(document, paths=tuple(paths))

    def _detect_polyline_intersections(
        self,
        polylines: tuple[_SegmentPolyline, ...],
        *,
        closed: bool,
    ) -> tuple[Point, ...]:
        fragments = self._edge_fragments(polylines)
        intersections: list[Point] = []

        for left_index, left in enumerate(fragments):
            for right in fragments[left_index + 1 :]:
                point = self._line_line_intersection(left.start, left.end, right.start, right.end)
                if point is None:
                    continue
                if self._is_ignored_adjacent_touch(left, right, point, len(polylines), closed=closed):
                    continue
                if self._contains_point(intersections, point):
                    continue
                intersections.append(point)

        intersections.sort(key=lambda point: (point[0], point[1]))
        return tuple(intersections)

    def _edge_fragments(self, polylines: tuple[_SegmentPolyline, ...]) -> tuple[_EdgeFragment, ...]:
        fragments: list[_EdgeFragment] = []
        for polyline in polylines:
            for index in range(len(polyline.points) - 1):
                start = polyline.points[index]
                end = polyline.points[index + 1]
                if PrecisionUtility.points_close(start, end, epsilon=self.config.epsilon):
                    continue
                fragments.append(
                    _EdgeFragment(
                        start=start,
                        end=end,
                        path_segment_index=polyline.path_segment_index,
                        fragment_index=index,
                        fragment_count=len(polyline.points) - 1,
                    )
                )
        return tuple(fragments)

    def _is_ignored_adjacent_touch(
        self,
        left: _EdgeFragment,
        right: _EdgeFragment,
        point: Point,
        path_segment_count: int,
        *,
        closed: bool,
    ) -> bool:
        if not self._touches_at_endpoint(left, point) or not self._touches_at_endpoint(right, point):
            return False

        if left.path_segment_index == right.path_segment_index:
            if {left.fragment_index, right.fragment_index} == {0, left.fragment_count - 1}:
                return True
            return abs(left.fragment_index - right.fragment_index) <= 1

        if left.path_segment_index + 1 == right.path_segment_index:
            return left.fragment_index == left.fragment_count - 1 and right.fragment_index == 0

        if right.path_segment_index + 1 == left.path_segment_index:
            return right.fragment_index == right.fragment_count - 1 and left.fragment_index == 0

        if closed and {left.path_segment_index, right.path_segment_index} == {0, path_segment_count - 1}:
            if left.path_segment_index == path_segment_count - 1:
                return left.fragment_index == left.fragment_count - 1 and right.fragment_index == 0
            return right.fragment_index == right.fragment_count - 1 and left.fragment_index == 0

        return False

    def _touches_at_endpoint(self, fragment: _EdgeFragment, point: Point) -> bool:
        return PrecisionUtility.points_close(fragment.start, point, epsilon=self.config.epsilon) or PrecisionUtility.points_close(
            fragment.end,
            point,
            epsilon=self.config.epsilon,
        )

    def _contains_point(self, points: list[Point], candidate: Point) -> bool:
        return any(PrecisionUtility.points_close(point, candidate, epsilon=self.config.epsilon) for point in points)

    def _path_segments(self, document: VectorDocument, path: Path) -> tuple[Segment, ...]:
        by_id = {segment.segment_id: segment for segment in document.segments}
        return tuple(by_id[segment_id] for segment_id in path.segments)

    def _sample_segment(self, segment: Segment) -> tuple[Point, ...]:
        return self._sampler.sample_segment(segment)

    def _line_line_intersection(self, p1: Point, p2: Point, q1: Point, q2: Point) -> Point | None:
        r = (p2[0] - p1[0], p2[1] - p1[1])
        s = (q2[0] - q1[0], q2[1] - q1[1])
        r_cross_s = self._cross(r, s)
        q_minus_p = (q1[0] - p1[0], q1[1] - p1[1])

        if PrecisionUtility.near_zero(r_cross_s, epsilon=self.config.epsilon):
            return self._collinear_overlap_point(p1, q1, q2, r, q_minus_p)

        t = self._cross(q_minus_p, s) / r_cross_s
        u = self._cross(q_minus_p, r) / r_cross_s

        if not self._within_segment(t) or not self._within_segment(u):
            return None

        return (p1[0] + t * r[0], p1[1] + t * r[1])

    def _collinear_overlap_point(
        self,
        p1: Point,
        q1: Point,
        q2: Point,
        r: Point,
        q_minus_p: Point,
    ) -> Point | None:
        if not PrecisionUtility.near_zero(self._cross(q_minus_p, r), epsilon=self.config.epsilon):
            return None

        r_dot_r = self._dot(r, r)
        if PrecisionUtility.near_zero(r_dot_r, epsilon=self.config.epsilon):
            return None

        t0 = self._dot(q_minus_p, r) / r_dot_r
        t1 = self._dot((q2[0] - p1[0], q2[1] - p1[1]), r) / r_dot_r
        overlap_start = max(0.0, min(t0, t1))
        overlap_end = min(1.0, max(t0, t1))
        if overlap_end < overlap_start - self.config.epsilon:
            return None

        if overlap_end - overlap_start <= self.config.epsilon:
            representative_t = overlap_start
        else:
            representative_t = (overlap_start + overlap_end) / 2.0

        clamped_t = min(max(representative_t, 0.0), 1.0)
        return (p1[0] + clamped_t * r[0], p1[1] + clamped_t * r[1])

    def _within_segment(self, value: float) -> bool:
        return -self.config.epsilon <= value <= 1.0 + self.config.epsilon

    def _cross(self, left: Point, right: Point) -> float:
        return left[0] * right[1] - left[1] * right[0]

    def _dot(self, left: Point, right: Point) -> float:
        return left[0] * right[0] + left[1] * right[1]

    def _coerce_point(self, value: Point | list[float]) -> Point:
        return (float(value[0]), float(value[1]))

    def _find_path_index(self, document: VectorDocument, path_id: str) -> int | None:
        for index, path in enumerate(document.paths):
            if path.path_id == path_id:
                return index
        return None


__all__ = [
    "SelfIntersectionConfig",
    "SelfIntersectionDetector",
    "SelfIntersectionResult",
]
