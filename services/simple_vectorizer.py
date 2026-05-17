from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from typing import Literal

from core.types import Anchor, Path, Point, Segment

InitialSegmentType = Literal["line", "bezier"]


@dataclass(frozen=True, slots=True)
class VectorizationResult:
    path: Path
    segments: tuple[Segment, ...]
    anchors: tuple[Anchor, ...]


class SimpleVectorizer:
    def __init__(self, segment_type: InitialSegmentType = "line") -> None:
        if segment_type not in {"line", "bezier"}:
            raise ValueError(f"unsupported initial segment type: {segment_type}")
        self.segment_type = segment_type

    def vectorize_contour(
        self,
        points: tuple[Point, ...] | list[Point],
        *,
        path_id: str,
        object_id: str | None = None,
        closed: bool = False,
        source: str = "resampled_contour",
        fill_role: str = "unknown",
        path_metadata: dict[str, Any] | None = None,
    ) -> VectorizationResult:
        normalized_points = self._normalize_points(points, closed=closed)
        minimum_points = 3 if closed else 2
        if len(normalized_points) < minimum_points:
            raise ValueError("not enough vector-space points to create a path")

        anchor_handles = self._build_anchor_handles(normalized_points, closed=closed)
        anchors = tuple(
            Anchor(
                anchor_id=f"{path_id}_anchor_{index}",
                path_id=path_id,
                position=point,
                continuity="smooth" if self.segment_type == "bezier" else "corner",
                shared_tangent=self._shared_tangent(normalized_points, index, closed=closed),
                in_handle=handles[0],
                out_handle=handles[1],
                metadata={"coordinate_space": "vector"},
            )
            for index, (point, handles) in enumerate(zip(normalized_points, anchor_handles))
        )

        segments = self._build_segments(path_id, anchors, closed=closed)
        path = Path(
            path_id=path_id,
            object_id=object_id,
            closed=closed,
            source=source,
            fill_role=fill_role,
            segments=tuple(segment.segment_id for segment in segments),
            topology_status="closed" if closed else "open",
            metadata={
                "coordinate_space": "vector",
                "initial_segment_type": self.segment_type,
                **(path_metadata or {}),
            },
        )
        return VectorizationResult(path=path, segments=segments, anchors=anchors)

    def _normalize_points(self, points: tuple[Point, ...] | list[Point], closed: bool) -> tuple[Point, ...]:
        normalized: list[Point] = []
        for point in points:
            candidate = (float(point[0]), float(point[1]))
            if normalized and self._points_close(normalized[-1], candidate):
                continue
            normalized.append(candidate)

        if closed and len(normalized) > 1 and self._points_close(normalized[0], normalized[-1]):
            normalized.pop()

        return tuple(normalized)

    def _build_anchor_handles(
        self,
        points: tuple[Point, ...],
        *,
        closed: bool,
    ) -> tuple[tuple[Point | None, Point | None], ...]:
        if self.segment_type == "line":
            return tuple((None, None) for _ in points)

        handles: list[tuple[Point | None, Point | None]] = []
        last_index = len(points) - 1

        for index, point in enumerate(points):
            if not closed and index == 0:
                next_point = points[1]
                delta = ((next_point[0] - point[0]) / 3.0, (next_point[1] - point[1]) / 3.0)
                handles.append((None, (point[0] + delta[0], point[1] + delta[1])))
                continue

            if not closed and index == last_index:
                prev_point = points[last_index - 1]
                delta = ((point[0] - prev_point[0]) / 3.0, (point[1] - prev_point[1]) / 3.0)
                handles.append(((point[0] - delta[0], point[1] - delta[1]), None))
                continue

            prev_point = points[(index - 1) % len(points)]
            next_point = points[(index + 1) % len(points)]
            tangent = ((next_point[0] - prev_point[0]) / 6.0, (next_point[1] - prev_point[1]) / 6.0)
            handles.append(
                (
                    (point[0] - tangent[0], point[1] - tangent[1]),
                    (point[0] + tangent[0], point[1] + tangent[1]),
                )
            )

        return tuple(handles)

    def _build_segments(self, path_id: str, anchors: tuple[Anchor, ...], *, closed: bool) -> tuple[Segment, ...]:
        segments: list[Segment] = []
        segment_count = len(anchors) if closed else len(anchors) - 1

        for index in range(segment_count):
            start_anchor = anchors[index]
            end_anchor = anchors[(index + 1) % len(anchors)]
            segments.append(
                Segment(
                    segment_id=f"{path_id}_segment_{index}",
                    path_id=path_id,
                    type=self.segment_type,
                    params=self._segment_params(start_anchor, end_anchor),
                    anchors=(start_anchor.anchor_id, end_anchor.anchor_id),
                    rigidity="high" if self.segment_type == "line" else "medium",
                    metadata={"coordinate_space": "vector"},
                )
            )

        return tuple(segments)

    def _segment_params(self, start_anchor: Anchor, end_anchor: Anchor) -> dict[str, Point]:
        if self.segment_type == "line":
            return {
                "start": start_anchor.position,
                "end": end_anchor.position,
            }

        return {
            "start": start_anchor.position,
            "control1": start_anchor.out_handle or start_anchor.position,
            "control2": end_anchor.in_handle or end_anchor.position,
            "end": end_anchor.position,
        }

    def _shared_tangent(self, points: tuple[Point, ...], index: int, *, closed: bool) -> Point | None:
        if self.segment_type != "bezier":
            return None

        point = points[index]
        if closed:
            prev_point = points[index - 1]
            next_point = points[(index + 1) % len(points)]
        elif index == 0:
            prev_point = point
            next_point = points[1]
        elif index == len(points) - 1:
            prev_point = points[index - 1]
            next_point = point
        else:
            prev_point = points[index - 1]
            next_point = points[index + 1]

        tangent = (next_point[0] - prev_point[0], next_point[1] - prev_point[1])
        length = math.hypot(tangent[0], tangent[1])
        if math.isclose(length, 0.0, abs_tol=1e-9):
            return None
        return (tangent[0] / length, tangent[1] / length)

    def _points_close(self, left: Point, right: Point) -> bool:
        return math.isclose(left[0], right[0], abs_tol=1e-9) and math.isclose(left[1], right[1], abs_tol=1e-9)


__all__ = ["InitialSegmentType", "SimpleVectorizer", "VectorizationResult"]
