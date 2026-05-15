from __future__ import annotations

from dataclasses import dataclass, field
import math
from xml.etree import ElementTree as ET

from core.coordinate import CoordinateTransformer
from core.types import Path, Point, Segment, Style, VectorDocument
from services.segment_sampler import SegmentSampler


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass(frozen=True, slots=True)
class SvgExporter:
    """Export VectorDocument to SVG.

    Internal segment angles stay in radians. SVG-specific degree conversion only
    happens at export boundaries where the target format requires it, such as
    ellipse ``rotate(...)`` transforms.
    """

    pretty: bool = False
    _segment_sampler: SegmentSampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_segment_sampler", SegmentSampler())

    def export_document(self, document: VectorDocument) -> str:
        transformer = CoordinateTransformer(document.coordinate_system)
        root = ET.Element(self._qualified("svg"))
        root.set("version", "1.1")
        root.set("width", self._fmt(document.width))
        root.set("height", self._fmt(document.height))
        root.set("viewBox", self._view_box(document, transformer))

        paths_by_id = {path.path_id: path for path in document.paths}
        for path in document.paths:
            if path.parent_path is not None:
                continue
            element = self._path_element(document, path, paths_by_id, transformer)
            root.append(element)

        self._indent(root)
        return ET.tostring(root, encoding="unicode")

    def _path_element(
        self,
        document: VectorDocument,
        path: Path,
        paths_by_id: dict[str, Path],
        transformer: CoordinateTransformer,
    ) -> ET.Element:
        compound_paths = (path,) + tuple(paths_by_id[child_id] for child_id in path.child_paths if child_id in paths_by_id)
        style = path.style or Style()

        if len(compound_paths) == 1:
            native = self._native_shape_element(document, path, transformer, style)
            if native is not None:
                return native

        element = ET.Element(self._qualified("path"))
        element.set("d", " ".join(self._path_d(document, item, transformer) for item in compound_paths))
        if len(compound_paths) > 1:
            element.set("fill-rule", "evenodd")
        self._apply_style(element, style)
        element.set("id", path.path_id)
        return element

    def _native_shape_element(
        self,
        document: VectorDocument,
        path: Path,
        transformer: CoordinateTransformer,
        style: Style,
    ) -> ET.Element | None:
        if len(path.segments) != 1:
            return None
        segment = self._segment_by_id(document, path.segments[0])
        if segment is None:
            return None

        if segment.type == "circle":
            element = ET.Element(self._qualified("circle"))
            center = transformer.vector_to_svg((float(segment.params["cx"]), float(segment.params["cy"])))
            radius = transformer.precision_rounding(float(segment.params["r"]))
            element.set("cx", self._fmt(center[0]))
            element.set("cy", self._fmt(center[1]))
            element.set("r", self._fmt(radius))
        elif segment.type == "ellipse":
            element = ET.Element(self._qualified("ellipse"))
            center = transformer.vector_to_svg((float(segment.params["cx"]), float(segment.params["cy"])))
            rx = transformer.precision_rounding(float(segment.params["rx"]))
            ry = transformer.precision_rounding(float(segment.params["ry"]))
            element.set("cx", self._fmt(center[0]))
            element.set("cy", self._fmt(center[1]))
            element.set("rx", self._fmt(rx))
            element.set("ry", self._fmt(ry))
            rotation = float(segment.params.get("rotation", 0.0))
            if not math.isclose(rotation, 0.0, abs_tol=1e-9):
                angle_degrees = math.degrees(rotation)
                element.set("transform", f"rotate({self._fmt(angle_degrees)} {self._fmt(center[0])} {self._fmt(center[1])})")
        else:
            return None

        self._apply_style(element, style)
        element.set("id", path.path_id)
        return element

    def _path_d(self, document: VectorDocument, path: Path, transformer: CoordinateTransformer) -> str:
        segments = [self._segment_by_id(document, segment_id) for segment_id in path.segments]
        segments = [segment for segment in segments if segment is not None]
        if not segments:
            raise ValueError(f"path {path.path_id} has no segments")

        commands: list[str] = []
        first_start = self._segment_start(segments[0], transformer)
        commands.append(f"M {self._fmt(first_start[0])} {self._fmt(first_start[1])}")
        for segment in segments:
            commands.extend(self._segment_commands(segment, transformer))
        if path.closed:
            commands.append("Z")
        return " ".join(commands)

    def _segment_commands(self, segment: Segment, transformer: CoordinateTransformer) -> list[str]:
        if segment.type == "line":
            end = transformer.vector_to_svg(self._point_from_params(segment.params, "end"))
            return [f"L {self._fmt(end[0])} {self._fmt(end[1])}"]
        if segment.type == "bezier":
            control1 = transformer.vector_to_svg(self._point_from_params(segment.params, "control1"))
            control2 = transformer.vector_to_svg(self._point_from_params(segment.params, "control2"))
            end = transformer.vector_to_svg(self._point_from_params(segment.params, "end"))
            return [
                "C "
                f"{self._fmt(control1[0])} {self._fmt(control1[1])} "
                f"{self._fmt(control2[0])} {self._fmt(control2[1])} "
                f"{self._fmt(end[0])} {self._fmt(end[1])}"
            ]
        if segment.type == "arc":
            return [self._arc_command(segment, transformer)]
        if segment.type == "circle":
            return self._circle_path_commands(segment, transformer)
        if segment.type == "ellipse":
            return self._ellipse_path_commands(segment, transformer)
        sampled = self._segment_sampler.sample_segment(segment)
        if len(sampled) < 2:
            return []
        commands: list[str] = []
        for point in sampled[1:]:
            svg_point = transformer.vector_to_svg(point)
            commands.append(f"L {self._fmt(svg_point[0])} {self._fmt(svg_point[1])}")
        return commands

    def _arc_command(self, segment: Segment, transformer: CoordinateTransformer) -> str:
        radius = abs(float(segment.params["r"]))
        start_angle = float(segment.params["start_angle"])
        end_angle = float(segment.params["end_angle"])
        direction = str(segment.params.get("direction", "ccw")).lower()
        end = transformer.vector_to_svg(self._point_from_params(segment.params, "end", fallback=self._arc_endpoint(segment, end_angle)))
        sweep = self._signed_arc_sweep(start_angle, end_angle, direction)
        large_arc_flag = 1 if abs(sweep) > math.pi else 0
        sweep_flag = 0 if direction == "cw" else 1
        rx = transformer.precision_rounding(radius)
        ry = transformer.precision_rounding(radius)
        return (
            "A "
            f"{self._fmt(rx)} {self._fmt(ry)} 0 {large_arc_flag} {sweep_flag} "
            f"{self._fmt(end[0])} {self._fmt(end[1])}"
        )

    def _circle_path_commands(self, segment: Segment, transformer: CoordinateTransformer) -> list[str]:
        cx = float(segment.params["cx"])
        cy = float(segment.params["cy"])
        r = abs(float(segment.params["r"]))
        start = transformer.vector_to_svg((cx + r, cy))
        rx = self._fmt(transformer.precision_rounding(r))
        ry = self._fmt(transformer.precision_rounding(r))
        return [
            f"L {self._fmt(start[0])} {self._fmt(start[1])}",
            f"A {rx} {ry} 0 1 1 {self._fmt(transformer.vector_to_svg((cx - r, cy))[0])} {self._fmt(transformer.vector_to_svg((cx - r, cy))[1])}",
            f"A {rx} {ry} 0 1 1 {self._fmt(start[0])} {self._fmt(start[1])}",
        ]

    def _ellipse_path_commands(self, segment: Segment, transformer: CoordinateTransformer) -> list[str]:
        cx = float(segment.params["cx"])
        cy = float(segment.params["cy"])
        rx = abs(float(segment.params["rx"]))
        ry = abs(float(segment.params["ry"]))
        rotation = math.degrees(float(segment.params.get("rotation", 0.0)))
        start = transformer.vector_to_svg(self._ellipse_point(cx, cy, rx, ry, math.radians(rotation), 0.0))
        opposite = transformer.vector_to_svg(self._ellipse_point(cx, cy, rx, ry, math.radians(rotation), math.pi))
        return [
            f"L {self._fmt(start[0])} {self._fmt(start[1])}",
            f"A {self._fmt(transformer.precision_rounding(rx))} {self._fmt(transformer.precision_rounding(ry))} {self._fmt(rotation)} 1 1 {self._fmt(opposite[0])} {self._fmt(opposite[1])}",
            f"A {self._fmt(transformer.precision_rounding(rx))} {self._fmt(transformer.precision_rounding(ry))} {self._fmt(rotation)} 1 1 {self._fmt(start[0])} {self._fmt(start[1])}",
        ]

    def _segment_start(self, segment: Segment, transformer: CoordinateTransformer) -> Point:
        if segment.type == "circle":
            return transformer.vector_to_svg((float(segment.params["cx"]) + abs(float(segment.params["r"])), float(segment.params["cy"])))
        if segment.type == "ellipse":
            cx = float(segment.params["cx"])
            cy = float(segment.params["cy"])
            rx = abs(float(segment.params["rx"]))
            ry = abs(float(segment.params["ry"]))
            rotation = float(segment.params.get("rotation", 0.0))
            return transformer.vector_to_svg(self._ellipse_point(cx, cy, rx, ry, rotation, 0.0))
        return transformer.vector_to_svg(self._point_from_params(segment.params, "start", fallback=self._arc_endpoint(segment, float(segment.params["start_angle"])) if segment.type == "arc" else None))

    def _segment_by_id(self, document: VectorDocument, segment_id: str) -> Segment | None:
        for segment in document.segments:
            if segment.segment_id == segment_id:
                return segment
        return None

    def _point_from_params(self, params: dict[str, object], key: str, fallback: Point | None = None) -> Point:
        value = params.get(key)
        if value is None:
            if fallback is None:
                raise ValueError(f"missing segment param: {key}")
            return fallback
        return (float(value[0]), float(value[1]))  # type: ignore[index]

    def _arc_endpoint(self, segment: Segment, angle: float) -> Point:
        cx = float(segment.params["cx"])
        cy = float(segment.params["cy"])
        r = abs(float(segment.params["r"]))
        return (cx + (r * math.cos(angle)), cy + (r * math.sin(angle)))

    def _ellipse_point(self, cx: float, cy: float, rx: float, ry: float, rotation: float, angle: float) -> Point:
        cos_theta = math.cos(rotation)
        sin_theta = math.sin(rotation)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        return (
            cx + (rx * cos_angle * cos_theta) - (ry * sin_angle * sin_theta),
            cy + (rx * cos_angle * sin_theta) + (ry * sin_angle * cos_theta),
        )

    def _signed_arc_sweep(self, start_angle: float, end_angle: float, direction: str) -> float:
        if math.isclose(start_angle, end_angle, abs_tol=1e-9):
            return -math.tau if direction == "cw" else math.tau
        if direction == "cw":
            sweep = end_angle - start_angle
            if sweep >= 0.0:
                sweep -= math.tau
            return sweep
        sweep = end_angle - start_angle
        if sweep <= 0.0:
            sweep += math.tau
        return sweep

    def _apply_style(self, element: ET.Element, style: Style) -> None:
        if style.fill_color is None:
            element.set("fill", "none")
        else:
            element.set("fill", self._color(style.fill_color))
        if style.fill_alpha is not None:
            element.set("fill-opacity", self._fmt(style.fill_alpha))
        if style.stroke_color is None or style.stroke_width <= 0.0:
            element.set("stroke", "none")
        else:
            element.set("stroke", self._color(style.stroke_color))
            element.set("stroke-width", self._fmt(style.stroke_width))
        if style.stroke_alpha is not None and style.stroke_color is not None:
            element.set("stroke-opacity", self._fmt(style.stroke_alpha))
        if not math.isclose(style.opacity, 1.0, abs_tol=1e-9):
            element.set("opacity", self._fmt(style.opacity))

    def _view_box(self, document: VectorDocument, transformer: CoordinateTransformer) -> str:
        if document.coordinate_system.view_box is not None:
            x, y, width, height = document.coordinate_system.view_box
            p0 = transformer.vector_to_svg((float(x), float(y)))
            p1 = transformer.vector_to_svg((float(x) + float(width), float(y) + float(height)))
            min_x = min(p0[0], p1[0])
            min_y = min(p0[1], p1[1])
            view_width = abs(p1[0] - p0[0])
            view_height = abs(p1[1] - p0[1])
            return f"{self._fmt(min_x)} {self._fmt(min_y)} {self._fmt(view_width)} {self._fmt(view_height)}"
        return f"0 0 {self._fmt(document.width)} {self._fmt(document.height)}"

    def _qualified(self, name: str) -> str:
        return f"{{{SVG_NS}}}{name}"

    @staticmethod
    def _fmt(value: float) -> str:
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text if text else "0"

    @staticmethod
    def _color(color: tuple[int, int, int]) -> str:
        return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"

    def _indent(self, element: ET.Element, level: int = 0) -> None:
        if not self.pretty:
            return
        indent = "\n" + ("  " * level)
        if len(element):
            if not element.text or not element.text.strip():
                element.text = indent + "  "
            for child in element:
                self._indent(child, level + 1)
            if not element[-1].tail or not element[-1].tail.strip():
                element[-1].tail = indent
        if level and (not element.tail or not element.tail.strip()):
            element.tail = indent


__all__ = ["SvgExporter"]
