from __future__ import annotations

from dataclasses import dataclass, field
import math

from core.coordinate import CoordinateTransformer
from core.types import CoordinateSystem, Point, Segment, VectorDocument
from services.segment_sampler import SegmentSampler


@dataclass(frozen=True, slots=True)
class DxfExporter:
    _segment_sampler: SegmentSampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_segment_sampler", SegmentSampler())

    def export_document(self, document: VectorDocument) -> str:
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

        for segment in document.segments:
            lines.extend(self._segment_entity_lines(segment, transformer))

        lines.extend(["0", "ENDSEC", "0", "EOF"])
        return "\n".join(lines) + "\n"

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


__all__ = ["DxfExporter"]
