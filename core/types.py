from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping, TypeVar

Point = tuple[float, float]
Color = tuple[int, int, int]
SegmentType = Literal["line", "arc", "circle", "ellipse", "bezier", "bspline", "polyline"]
ShapeCandidateTargetType = Literal["circle", "rectangle", "ellipse", "arc", "line"]
ContinuityType = Literal["corner", "smooth", "symmetric", "curvature"]
SegmentTypes = ("line", "arc", "circle", "ellipse", "bezier", "bspline", "polyline")

T = TypeVar("T")


def updated(obj: T, **changes: Any) -> T:
    return replace(obj, **changes)


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _canonicalize_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_canonicalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonicalize_json_value(item) for key, item in value.items()}
    return value


def _copy_float_mapping(value: Mapping[str, float] | None) -> dict[str, float]:
    return {str(key): float(item) for key, item in (value or {}).items()}


def _tuple_of_strings(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(values or ())


def _point(value: Iterable[float] | None) -> Point | None:
    if value is None:
        return None
    x, y = value
    return (float(x), float(y))


def _required_point(value: Iterable[float]) -> Point:
    point = _point(value)
    if point is None:
        raise ValueError("point value is required")
    return point


def _color(value: Iterable[int] | None) -> Color | None:
    if value is None:
        return None
    red, green, blue = value
    return (int(red), int(green), int(blue))


def _view_box(value: Iterable[float] | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    min_x, min_y, width, height = value
    return (float(min_x), float(min_y), float(width), float(height))


@dataclass(frozen=True, slots=True)
class Anchor:
    anchor_id: str
    path_id: str
    position: Point
    continuity: ContinuityType = "corner"
    shared_tangent: Point | None = None
    locked: bool = False
    in_handle: Point | None = None
    out_handle: Point | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _required_point(self.position))
        object.__setattr__(self, "shared_tangent", _point(self.shared_tangent))
        object.__setattr__(self, "in_handle", _point(self.in_handle))
        object.__setattr__(self, "out_handle", _point(self.out_handle))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Segment:
    """Pure data segment.

    Angle-bearing params use radians internally:
    - arc.start_angle
    - arc.end_angle
    - ellipse.rotation

    External degree-based imports must convert explicitly before core algorithms
    consume the segment, or carry an explicit adapter hint such as
    ``angle_unit='degree'`` for importer-facing utilities.
    """

    segment_id: str
    path_id: str
    type: SegmentType
    params: dict[str, Any] = field(default_factory=dict)
    anchors: tuple[str, ...] = ()
    fit_error: float | None = None
    complexity_score: float | None = None
    confidence: float | None = None
    rigidity: str | None = None
    locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in SegmentTypes:
            raise ValueError(f"unsupported segment type: {self.type}")
        object.__setattr__(self, "params", _canonicalize_json_value(_copy_mapping(self.params)))
        object.__setattr__(self, "anchors", _tuple_of_strings(self.anchors))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Style:
    fill_color: Color | None = None
    fill_alpha: float | None = None
    stroke_color: Color | None = None
    stroke_alpha: float | None = None
    stroke_width: float = 0.0
    opacity: float = 1.0
    color_confidence: float | None = None
    color_variance: float | None = None
    alpha_variance: float | None = None
    paint_type: str = "solid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_color", _color(self.fill_color))
        object.__setattr__(self, "stroke_color", _color(self.stroke_color))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Path:
    path_id: str
    object_id: str | None = None
    closed: bool = False
    source: str = "unknown"
    fill_role: str = "unknown"
    parent_path: str | None = None
    child_paths: tuple[str, ...] = ()
    segments: tuple[str, ...] = ()
    style: Style | None = None
    topology_status: str = "open"
    max_gap: float = 0.0
    self_intersection_count: int = 0
    locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_paths", _tuple_of_strings(self.child_paths))
        object.__setattr__(self, "segments", _tuple_of_strings(self.segments))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Object:
    object_id: str
    type: str
    semantic_label: str | None = None
    paths: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    confidence: float | None = None
    locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _tuple_of_strings(self.paths))
        object.__setattr__(self, "constraints", _tuple_of_strings(self.constraints))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Constraint:
    constraint_id: str
    type: str
    targets: tuple[str, ...]
    strength: float = 1.0
    source: str = "system"
    confidence: float | None = None
    locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", _tuple_of_strings(self.targets))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CoordinateSystem:
    internal_space: str = "vector"
    source_space: str = "pixel"
    origin: str = "top_left"
    y_axis: str = "down"
    unit: str = "px"
    precision: int = 4
    view_box: tuple[float, float, float, float] | None = None
    scale: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "view_box", _view_box(self.view_box))
        object.__setattr__(self, "scale", _copy_float_mapping(self.scale))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class VectorDocument:
    document_id: str
    width: float
    height: float
    coordinate_system: CoordinateSystem
    objects: tuple[Object, ...] = ()
    paths: tuple[Path, ...] = ()
    segments: tuple[Segment, ...] = ()
    anchors: tuple[Anchor, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ShapeCandidate:
    candidate_id: str
    target_type: ShapeCandidateTargetType
    path_id: str
    segment_range: tuple[int, int]
    source: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_range", (int(self.segment_range[0]), int(self.segment_range[1])))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "evidence", _copy_mapping(self.evidence))


__all__ = [
    "Anchor",
    "Color",
    "Constraint",
    "ContinuityType",
    "CoordinateSystem",
    "Object",
    "Path",
    "Point",
    "ShapeCandidate",
    "ShapeCandidateTargetType",
    "Segment",
    "SegmentType",
    "SegmentTypes",
    "Style",
    "VectorDocument",
    "updated",
]
