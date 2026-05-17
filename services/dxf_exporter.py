from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from core.coordinate import CoordinateTransformer
from core.precision import PrecisionUtility
from core.types import CoordinateSystem, Path, Point, Segment, VectorDocument
from services.segment_sampler import SegmentSampler

ExportMode = Literal["outline", "centerline", "all_debug"]

@dataclass(frozen=True, slots=True)
class DxfExporter:
    """Export VectorDocument to ASCII DXF.

    Internal segment angles stay in radians. DXF ARC entities require degrees,
    so conversion only occurs during export.
    """

    _segment_sampler: SegmentSampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_segment_sampler", SegmentSampler())

    def export_document(self, document: VectorDocument, *, export_mode: ExportMode = "all_debug") -> str:
        transformer = self._transformer_for_document(document)
        lines: list[str] = [
            "0",
            "SECTION",
            "2",
            "HEADER",
            "9",
            "$INSUNITS",
            "70",
            "4",
            "0",
            "ENDSEC",
            "0",
            "SECTION",
            "2",
            "ENTITIES",
        ]

        selected_paths = self._selected_paths(document, export_mode)
        if not selected_paths:
            lines.extend(["999", f"warning: no paths selected for export_mode={export_mode}"])
        segment_lookup = {segment.segment_id: segment for segment in document.segments}
        warnings: list[str] = []
        for path in selected_paths:
            lines.extend(self._path_entity_lines(path, segment_lookup, transformer, warnings))

        for warning in warnings:
            lines.extend(["999", warning])

        lines.extend(["0", "ENDSEC", "0", "EOF"])
        return "\n".join(lines) + "\n"

    def export_report(self, document: VectorDocument, *, export_mode: ExportMode = "all_debug") -> dict[str, object]:
        selected_paths = self._selected_paths(document, export_mode)
        skipped_paths = tuple(path for path in document.paths if path not in selected_paths)
        entity_counts = self._entity_counts(document, selected_paths)
        warnings = self._collect_warnings(document, selected_paths)
        return {
            "export_mode": export_mode,
            "exported_path_count": len(selected_paths),
            "skipped_path_count": len(skipped_paths),
            "exported_by_source": self._count_sources(selected_paths),
            "skipped_by_source": self._count_sources(skipped_paths),
            "entity_counts": entity_counts,
            "warning": None if selected_paths else f"no paths selected for export_mode={export_mode}",
            "warnings": warnings,
        }

    def _selected_paths(self, document: VectorDocument, export_mode: ExportMode) -> tuple[Path, ...]:
        if export_mode == "all_debug":
            return document.paths
        if export_mode == "outline":
            return tuple(path for path in document.paths if path.source == "binary_contour")
        if export_mode == "centerline":
            return tuple(path for path in document.paths if path.source == "skeleton_contour")
        raise ValueError(f"unsupported export_mode: {export_mode}")

    def _count_sources(self, paths: tuple[Path, ...] | list[Path]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in paths:
            counts[path.source] = counts.get(path.source, 0) + 1
        return counts

    def _entity_counts(self, document: VectorDocument, paths: tuple[Path, ...]) -> dict[str, int]:
        segment_lookup = {segment.segment_id: segment for segment in document.segments}
        counts: dict[str, int] = {"LWPOLYLINE": 0, "LINE": 0, "ARC": 0, "CIRCLE": 0}
        for path in paths:
            polyline_points = self._polyline_vertices(path, segment_lookup)
            if polyline_points is not None:
                counts["LWPOLYLINE"] += 1
                continue
            for segment_id in path.segments:
                segment = segment_lookup.get(segment_id)
                if segment is None:
                    continue
                if segment.type == "line":
                    counts["LINE"] += 1
                elif segment.type == "arc":
                    counts["ARC"] += 1
                elif segment.type == "circle":
                    counts["CIRCLE"] += 1
                else:
                    counts["LWPOLYLINE"] += 1
        return counts

    def _collect_warnings(self, document: VectorDocument, paths: tuple[Path, ...]) -> tuple[str, ...]:
        segment_lookup = {segment.segment_id: segment for segment in document.segments}
        warnings: list[str] = []
        for path in paths:
            if path.closed and path.topology_status == "closed":
                segments = [segment_lookup.get(segment_id) for segment_id in path.segments]
                segments = [segment for segment in segments if segment is not None]
                if segments and all(segment.type == "line" for segment in segments):
                    if self._polyline_vertices(path, segment_lookup) is None:
                        warnings.append(f"path {path.path_id} is not continuous enough for closed LWPOLYLINE fallback")
        return tuple(warnings)

    def _path_entity_lines(
        self,
        path: Path,
        segment_lookup: dict[str, Segment],
        transformer: CoordinateTransformer,
        warnings: list[str],
    ) -> list[str]:
        polyline_points = self._polyline_vertices(path, segment_lookup)
        if polyline_points is not None:
            return self._path_polyline_lines(path, polyline_points, transformer)

        if path.closed and path.topology_status == "closed":
            segments = [segment_lookup.get(segment_id) for segment_id in path.segments]
            segments = [segment for segment in segments if segment is not None]
            if segments and all(segment.type == "line" for segment in segments):
                warnings.append(f"path {path.path_id} is not continuous enough for closed LWPOLYLINE fallback")

        lines: list[str] = []
        for segment_id in path.segments:
            segment = segment_lookup.get(segment_id)
            if segment is not None:
                lines.extend(self._segment_entity_lines(segment, transformer))
        return lines

    def _path_polyline_lines(
        self,
        path: Path,
        points: tuple[Point, ...],
        transformer: CoordinateTransformer,
    ) -> list[str]:
        lines = [
            "0",
            "LWPOLYLINE",
            "8",
            path.path_id,
            "90",
            str(len(points)),
            "70",
            "1" if path.closed and path.topology_status == "closed" else "0",
        ]
        for point in points:
            dxf_point = transformer.vector_to_dxf(point)
            lines.extend(["10", self._fmt(dxf_point[0]), "20", self._fmt(dxf_point[1])])
        return lines

    def _polyline_vertices(self, path: Path, segment_lookup: dict[str, Segment]) -> tuple[Point, ...] | None:
        if not path.segments:
            return None
        segments = [segment_lookup.get(segment_id) for segment_id in path.segments]
        if any(segment is None for segment in segments):
            return None
        typed_segments = [segment for segment in segments if segment is not None]
        if len(typed_segments) < 2 or not all(segment.type == "line" for segment in typed_segments):
            return None

        vertices = [self._point_from_params(typed_segments[0].params, "start")]
        previous_end = None
        for segment in typed_segments:
            start = self._point_from_params(segment.params, "start")
            end = self._point_from_params(segment.params, "end")
            if previous_end is not None and not PrecisionUtility.points_close(previous_end, start):
                return None
            vertices.append(end)
            previous_end = end

        if path.closed and path.topology_status == "closed":
            if not PrecisionUtility.points_close(vertices[0], vertices[-1]):
                return None
            vertices.pop()
            if len(vertices) < 3:
                return None

        return tuple(vertices)

    def _segment_entity_lines(self, segment: Segment, transformer: CoordinateTransformer) -> list[str]:
        if segment.type == "line":
            return self._line_entity_lines(segment, transformer)
        if segment.type == "circle":
            return self._circle_entity_lines(segment, transformer)
        if segment.type == "arc":
            return self._arc_entity_lines(segment, transformer)
        if segment.type == "ellipse":
            return self._polyline_fallback_lines(segment, transformer, closed=True)
        return self._polyline_fallback_lines(segment, transformer, closed=False)

    def _line_entity_lines(self, segment: Segment, transformer: CoordinateTransformer) -> list[str]:
        start = transformer.vector_to_dxf(self._point_from_params(segment.params, "start"))
        end = transformer.vector_to_dxf(self._point_from_params(segment.params, "end"))
        return [
            "0",
            "LINE",
            "8",
            segment.path_id,
            "10",
            self._fmt(start[0]),
            "20",
            self._fmt(start[1]),
            "11",
            self._fmt(end[0]),
            "21",
            self._fmt(end[1]),
        ]

    def _circle_entity_lines(self, segment: Segment, transformer: CoordinateTransformer) -> list[str]:
        center_source = (float(segment.params["cx"]), float(segment.params["cy"]))
        center = transformer.vector_to_dxf(center_source)
        start = transformer.vector_to_dxf((center_source[0] + abs(float(segment.params["r"])), center_source[1]))
        radius = math.dist(center, start)
        return [
            "0",
            "CIRCLE",
            "8",
            segment.path_id,
            "10",
            self._fmt(center[0]),
            "20",
            self._fmt(center[1]),
            "40",
            self._fmt(radius),
        ]

    def _arc_entity_lines(self, segment: Segment, transformer: CoordinateTransformer) -> list[str]:
        cx = float(segment.params["cx"])
        cy = float(segment.params["cy"])
        radius = abs(float(segment.params["r"]))
        start_angle = float(segment.params["start_angle"])
        end_angle = float(segment.params["end_angle"])
        direction = str(segment.params.get("direction", "ccw")).lower()
        center = transformer.vector_to_dxf((cx, cy))
        start_point = transformer.vector_to_dxf((cx + (radius * math.cos(start_angle)), cy + (radius * math.sin(start_angle))))
        end_point = transformer.vector_to_dxf((cx + (radius * math.cos(end_angle)), cy + (radius * math.sin(end_angle))))
        dxf_radius = math.dist(center, start_point)
        start_degrees = self._angle_degrees(center, start_point)
        end_degrees = self._angle_degrees(center, end_point)
        if self._swap_arc_angles(direction=direction, transformer=transformer):
            start_degrees, end_degrees = end_degrees, start_degrees
        return [
            "0",
            "ARC",
            "8",
            segment.path_id,
            "10",
            self._fmt(center[0]),
            "20",
            self._fmt(center[1]),
            "40",
            self._fmt(dxf_radius),
            "50",
            self._fmt(start_degrees),
            "51",
            self._fmt(end_degrees),
        ]

    def _polyline_fallback_lines(
        self,
        segment: Segment,
        transformer: CoordinateTransformer,
        *,
        closed: bool,
    ) -> list[str]:
        sampled = self._segment_sampler.sample_segment(segment)
        if len(sampled) < 2:
            raise ValueError(f"segment {segment.segment_id} has insufficient points for DXF fallback")

        lines = [
            "0",
            "LWPOLYLINE",
            "8",
            segment.path_id,
            "90",
            str(len(sampled)),
            "70",
            "1" if closed else "0",
        ]
        for point in sampled:
            dxf_point = transformer.vector_to_dxf(point)
            lines.extend(["10", self._fmt(dxf_point[0]), "20", self._fmt(dxf_point[1])])
        return lines

    def _transformer_for_document(self, document: VectorDocument) -> CoordinateTransformer:
        coordinate_system = document.coordinate_system
        if coordinate_system.view_box is not None:
            return CoordinateTransformer(coordinate_system)
        fallback_view_box = (0.0, 0.0, float(document.width), float(document.height))
        return CoordinateTransformer(
            CoordinateSystem(
                internal_space=coordinate_system.internal_space,
                source_space=coordinate_system.source_space,
                origin=coordinate_system.origin,
                y_axis=coordinate_system.y_axis,
                unit=coordinate_system.unit,
                precision=coordinate_system.precision,
                view_box=fallback_view_box,
                scale=coordinate_system.scale,
                metadata=coordinate_system.metadata,
            )
        )

    @staticmethod
    def _point_from_params(params: dict[str, object], key: str) -> Point:
        value = params[key]
        return (float(value[0]), float(value[1]))  # type: ignore[index]

    @staticmethod
    def _angle_degrees(center: Point, point: Point) -> float:
        angle = math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))
        if angle < 0.0:
            angle += 360.0
        return angle

    @staticmethod
    def _swap_arc_angles(*, direction: str, transformer: CoordinateTransformer) -> bool:
        y_flip_reverses_orientation = transformer.coordinate_system.y_axis != "up"
        return (direction == "cw") != y_flip_reverses_orientation

    @staticmethod
    def _fmt(value: float) -> str:
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text if text else "0"


__all__ = ["DxfExporter", "ExportMode"]
