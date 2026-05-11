from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterable

from core.types import Anchor, Constraint, CoordinateSystem, Object, Path, Segment, Style, VectorDocument, updated


def create_document(
    document_id: str,
    width: float,
    height: float,
    coordinate_system: CoordinateSystem,
    metadata: dict[str, Any] | None = None,
) -> VectorDocument:
    return VectorDocument(
        document_id=document_id,
        width=width,
        height=height,
        coordinate_system=coordinate_system,
        metadata=metadata or {},
    )


def add_object(document: VectorDocument, obj: Object) -> VectorDocument:
    _ensure_unique_ids(document.objects, obj.object_id, "object_id")
    return updated(document, objects=document.objects + (obj,))


def add_path(document: VectorDocument, path: Path) -> VectorDocument:
    _ensure_unique_ids(document.paths, path.path_id, "path_id")

    objects = document.objects
    if path.object_id is not None:
        object_index = _find_index(document.objects, "object_id", path.object_id)
        if object_index is None:
            raise ValueError(f"unknown object_id: {path.object_id}")
        owner = document.objects[object_index]
        objects = _replace_at(
            document.objects,
            object_index,
            updated(owner, paths=_append_unique(owner.paths, path.path_id)),
        )

    paths = document.paths
    if path.parent_path is not None:
        parent_index = _find_index(document.paths, "path_id", path.parent_path)
        if parent_index is None:
            raise ValueError(f"unknown parent_path: {path.parent_path}")
        parent = document.paths[parent_index]
        paths = _replace_at(
            document.paths,
            parent_index,
            updated(parent, child_paths=_append_unique(parent.child_paths, path.path_id)),
        )

    return updated(document, objects=objects, paths=paths + (path,))


def add_segment(document: VectorDocument, segment: Segment) -> VectorDocument:
    _ensure_unique_ids(document.segments, segment.segment_id, "segment_id")
    path_index = _find_index(document.paths, "path_id", segment.path_id)
    if path_index is None:
        raise ValueError(f"unknown path_id: {segment.path_id}")

    owner_path = document.paths[path_index]
    paths = _replace_at(
        document.paths,
        path_index,
        updated(owner_path, segments=_append_unique(owner_path.segments, segment.segment_id)),
    )
    return updated(document, paths=paths, segments=document.segments + (segment,))


def add_anchor(document: VectorDocument, anchor: Anchor) -> VectorDocument:
    _ensure_unique_ids(document.anchors, anchor.anchor_id, "anchor_id")
    if _find_index(document.paths, "path_id", anchor.path_id) is None:
        raise ValueError(f"unknown path_id: {anchor.path_id}")
    return updated(document, anchors=document.anchors + (anchor,))


def add_constraint(document: VectorDocument, constraint: Constraint) -> VectorDocument:
    _ensure_unique_ids(document.constraints, constraint.constraint_id, "constraint_id")

    objects = document.objects
    for index, obj in enumerate(document.objects):
        if obj.object_id in constraint.targets:
            objects = _replace_at(
                objects,
                index,
                updated(obj, constraints=_append_unique(obj.constraints, constraint.constraint_id)),
            )

    return updated(document, objects=objects, constraints=document.constraints + (constraint,))


def to_json(document: VectorDocument) -> str:
    return json.dumps(asdict(document), sort_keys=True)


def from_json(payload: str) -> VectorDocument:
    return from_dict(json.loads(payload))


def from_dict(data: dict[str, Any]) -> VectorDocument:
    coordinate_system = _coordinate_system_from_dict(data["coordinate_system"])
    objects = tuple(_object_from_dict(item) for item in data.get("objects", ()))
    paths = tuple(_path_from_dict(item) for item in data.get("paths", ()))
    segments = tuple(_segment_from_dict(item) for item in data.get("segments", ()))
    anchors = tuple(_anchor_from_dict(item) for item in data.get("anchors", ()))
    constraints = tuple(_constraint_from_dict(item) for item in data.get("constraints", ()))

    return VectorDocument(
        document_id=data["document_id"],
        width=float(data["width"]),
        height=float(data["height"]),
        coordinate_system=coordinate_system,
        objects=objects,
        paths=paths,
        segments=segments,
        anchors=anchors,
        constraints=constraints,
        metadata=dict(data.get("metadata", {})),
    )


def _anchor_from_dict(data: dict[str, Any]) -> Anchor:
    return Anchor(
        anchor_id=data["anchor_id"],
        path_id=data["path_id"],
        position=tuple(data["position"]),
        continuity=data.get("continuity", "corner"),
        shared_tangent=data.get("shared_tangent"),
        locked=bool(data.get("locked", False)),
        in_handle=data.get("in_handle"),
        out_handle=data.get("out_handle"),
        metadata=dict(data.get("metadata", {})),
    )


def _segment_from_dict(data: dict[str, Any]) -> Segment:
    return Segment(
        segment_id=data["segment_id"],
        path_id=data["path_id"],
        type=data["type"],
        params=dict(data.get("params", {})),
        anchors=tuple(data.get("anchors", ())),
        fit_error=data.get("fit_error"),
        complexity_score=data.get("complexity_score"),
        confidence=data.get("confidence"),
        rigidity=data.get("rigidity"),
        locked=bool(data.get("locked", False)),
        metadata=dict(data.get("metadata", {})),
    )


def _style_from_dict(data: dict[str, Any] | None) -> Style | None:
    if data is None:
        return None
    return Style(
        fill_color=data.get("fill_color"),
        fill_alpha=data.get("fill_alpha"),
        stroke_color=data.get("stroke_color"),
        stroke_alpha=data.get("stroke_alpha"),
        stroke_width=float(data.get("stroke_width", 0.0)),
        opacity=float(data.get("opacity", 1.0)),
        color_confidence=data.get("color_confidence"),
        color_variance=data.get("color_variance"),
        alpha_variance=data.get("alpha_variance"),
        paint_type=data.get("paint_type", "solid"),
        metadata=dict(data.get("metadata", {})),
    )


def _path_from_dict(data: dict[str, Any]) -> Path:
    return Path(
        path_id=data["path_id"],
        object_id=data.get("object_id"),
        closed=bool(data.get("closed", False)),
        source=data.get("source", "unknown"),
        fill_role=data.get("fill_role", "unknown"),
        parent_path=data.get("parent_path"),
        child_paths=tuple(data.get("child_paths", ())),
        segments=tuple(data.get("segments", ())),
        style=_style_from_dict(data.get("style")),
        topology_status=data.get("topology_status", "open"),
        max_gap=float(data.get("max_gap", 0.0)),
        self_intersection_count=int(data.get("self_intersection_count", 0)),
        locked=bool(data.get("locked", False)),
        metadata=dict(data.get("metadata", {})),
    )


def _object_from_dict(data: dict[str, Any]) -> Object:
    return Object(
        object_id=data["object_id"],
        type=data["type"],
        semantic_label=data.get("semantic_label"),
        paths=tuple(data.get("paths", ())),
        constraints=tuple(data.get("constraints", ())),
        confidence=data.get("confidence"),
        locked=bool(data.get("locked", False)),
        metadata=dict(data.get("metadata", {})),
    )


def _constraint_from_dict(data: dict[str, Any]) -> Constraint:
    return Constraint(
        constraint_id=data["constraint_id"],
        type=data["type"],
        targets=tuple(data.get("targets", ())),
        strength=float(data.get("strength", 1.0)),
        source=data.get("source", "system"),
        confidence=data.get("confidence"),
        locked=bool(data.get("locked", False)),
        metadata=dict(data.get("metadata", {})),
    )


def _coordinate_system_from_dict(data: dict[str, Any]) -> CoordinateSystem:
    return CoordinateSystem(
        internal_space=data.get("internal_space", "vector"),
        source_space=data.get("source_space", "pixel"),
        origin=data.get("origin", "top_left"),
        y_axis=data.get("y_axis", "down"),
        unit=data.get("unit", "px"),
        precision=int(data.get("precision", 4)),
        view_box=data.get("view_box"),
        scale=dict(data.get("scale", {})),
        metadata=dict(data.get("metadata", {})),
    )


def _append_unique(values: Iterable[str], item: str) -> tuple[str, ...]:
    current = tuple(values)
    if item in current:
        return current
    return current + (item,)


def _replace_at(values: tuple[Any, ...], index: int, replacement: Any) -> tuple[Any, ...]:
    items = list(values)
    items[index] = replacement
    return tuple(items)


def _find_index(values: Iterable[Any], attribute: str, expected: str) -> int | None:
    for index, value in enumerate(values):
        if getattr(value, attribute) == expected:
            return index
    return None


def _ensure_unique_ids(values: Iterable[Any], expected: str, attribute: str) -> None:
    if _find_index(values, attribute, expected) is not None:
        raise ValueError(f"duplicate {attribute}: {expected}")


__all__ = [
    "add_anchor",
    "add_constraint",
    "add_object",
    "add_path",
    "add_segment",
    "create_document",
    "from_dict",
    "from_json",
    "to_json",
]
