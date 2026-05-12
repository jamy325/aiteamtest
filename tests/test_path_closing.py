import ast
from pathlib import Path

import pytest

from core.document import add_anchor, add_path, add_segment, create_document
from core.types import Anchor, CoordinateSystem, Path as VectorPath, Segment
from services.topology import PathClosingConfig, TopologyEngine


def _build_document(
    *,
    path: VectorPath,
    anchors: tuple[Anchor, ...],
    segments: tuple[Segment, ...],
):
    document = create_document(
        document_id=f"doc_{path.path_id}",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(),
    )
    document = add_path(document, path)
    for anchor in anchors:
        document = add_anchor(document, anchor)
    for segment in segments:
        document = add_segment(document, segment)
    return document


def test_topology_engine_leaves_open_contiguous_path_unchanged() -> None:
    path = VectorPath(path_id="path_open", closed=False, topology_status="open")
    anchors = (
        Anchor(anchor_id="a1", path_id="path_open", position=(0.0, 0.0)),
        Anchor(anchor_id="a2", path_id="path_open", position=(10.0, 0.0)),
        Anchor(anchor_id="a3", path_id="path_open", position=(20.0, 0.0)),
    )
    segments = (
        Segment("seg_1", "path_open", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a1", "a2")),
        Segment("seg_2", "path_open", "line", params={"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a2", "a3")),
    )
    document = _build_document(path=path, anchors=anchors, segments=segments)
    engine = TopologyEngine(PathClosingConfig(auto_snap_distance=0.5))

    result = engine.enforce_path_topology(document, "path_open")

    assert result.topology_status == "open"
    assert result.topology_error is False
    assert result.max_gap == pytest.approx(0.0)
    assert result.corrections == ()
    assert result.document == document


def test_topology_engine_auto_snaps_small_gap_in_open_path() -> None:
    path = VectorPath(path_id="path_gap", closed=False, topology_status="open")
    anchors = (
        Anchor(anchor_id="a1", path_id="path_gap", position=(0.0, 0.0)),
        Anchor(anchor_id="a2", path_id="path_gap", position=(10.0, 0.0)),
        Anchor(anchor_id="a3", path_id="path_gap", position=(10.2, 0.0)),
        Anchor(anchor_id="a4", path_id="path_gap", position=(20.0, 0.0)),
    )
    segments = (
        Segment("seg_1", "path_gap", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a1", "a2")),
        Segment("seg_2", "path_gap", "polyline", params={"start": [10.2, 0.0], "end": [20.0, 0.0]}, anchors=("a3", "a4")),
    )
    document = _build_document(path=path, anchors=anchors, segments=segments)
    engine = TopologyEngine(PathClosingConfig(auto_snap_distance=0.5))

    result = engine.enforce_path_topology(document, "path_gap")

    assert result.topology_status == "open"
    assert result.topology_error is False
    assert result.max_gap == pytest.approx(0.2)
    assert len(result.corrections) == 1
    assert result.corrections[0].strategy == "move_right"
    assert result.corrections[0].corrected is True
    assert result.corrections[0].moved_anchor_ids == ("a3",)
    assert result.document.segments[1].params["start"] == [10.0, 0.0]
    assert result.document.anchors[2].position == (10.0, 0.0)
    assert document.segments[1].params["start"] == [10.2, 0.0]


def test_topology_engine_corrects_closed_path_last_to_first_gap() -> None:
    path = VectorPath(path_id="path_closed", closed=True, topology_status="open")
    anchors = (
        Anchor(anchor_id="a1", path_id="path_closed", position=(0.0, 0.0)),
        Anchor(anchor_id="a2", path_id="path_closed", position=(10.0, 0.0)),
        Anchor(anchor_id="a3", path_id="path_closed", position=(10.0, 10.0)),
        Anchor(anchor_id="a4", path_id="path_closed", position=(0.2, 0.0)),
    )
    segments = (
        Segment("seg_1", "path_closed", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a1", "a2")),
        Segment("seg_2", "path_closed", "line", params={"start": [10.0, 0.0], "end": [10.0, 10.0]}, anchors=("a2", "a3")),
        Segment("seg_3", "path_closed", "line", params={"start": [10.0, 10.0], "end": [0.2, 0.0]}, anchors=("a3", "a4")),
    )
    document = _build_document(path=path, anchors=anchors, segments=segments)
    engine = TopologyEngine(PathClosingConfig(auto_snap_distance=0.5))

    result = engine.enforce_path_topology(document, "path_closed")

    assert result.topology_status == "closed"
    assert result.topology_error is False
    assert result.max_gap == pytest.approx(0.2)
    assert len(result.corrections) == 1
    assert result.corrections[0].closing_gap is True
    assert result.corrections[0].strategy == "move_both_minimal"
    assert result.document.segments[0].params["start"] == [0.1, 0.0]
    assert result.document.segments[2].params["end"] == [0.1, 0.0]
    assert result.document.anchors[0].position == (0.1, 0.0)
    assert result.document.anchors[3].position == (0.1, 0.0)


def test_topology_engine_marks_large_gap_as_topology_error() -> None:
    path = VectorPath(path_id="path_error", closed=False, topology_status="open")
    anchors = (
        Anchor(anchor_id="a1", path_id="path_error", position=(0.0, 0.0)),
        Anchor(anchor_id="a2", path_id="path_error", position=(10.0, 0.0)),
        Anchor(anchor_id="a3", path_id="path_error", position=(14.0, 0.0)),
        Anchor(anchor_id="a4", path_id="path_error", position=(20.0, 0.0)),
    )
    segments = (
        Segment("seg_1", "path_error", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a1", "a2")),
        Segment("seg_2", "path_error", "line", params={"start": [14.0, 0.0], "end": [20.0, 0.0]}, anchors=("a3", "a4")),
    )
    document = _build_document(path=path, anchors=anchors, segments=segments)
    engine = TopologyEngine(PathClosingConfig(auto_snap_distance=0.5))

    result = engine.enforce_path_topology(document, "path_error")

    assert result.topology_status == "topology_error"
    assert result.topology_error is True
    assert result.max_gap == pytest.approx(4.0)
    assert len(result.corrections) == 1
    assert result.corrections[0].corrected is False
    assert result.corrections[0].topology_error is True
    assert result.corrections[0].reason == "gap_exceeds_auto_snap_distance"
    assert result.document.segments[1].params["start"] == [14.0, 0.0]


def test_topology_engine_does_not_move_locked_segment() -> None:
    path = VectorPath(path_id="path_locked", closed=False, topology_status="open")
    anchors = (
        Anchor(anchor_id="a1", path_id="path_locked", position=(0.0, 0.0)),
        Anchor(anchor_id="a2", path_id="path_locked", position=(10.0, 0.0)),
        Anchor(anchor_id="a3", path_id="path_locked", position=(10.2, 0.0)),
        Anchor(anchor_id="a4", path_id="path_locked", position=(20.0, 0.0)),
    )
    segments = (
        Segment(
            "seg_1",
            "path_locked",
            "line",
            params={"start": [0.0, 0.0], "end": [10.0, 0.0]},
            anchors=("a1", "a2"),
            locked=True,
        ),
        Segment(
            "seg_2",
            "path_locked",
            "polyline",
            params={"start": [10.2, 0.0], "end": [20.0, 0.0]},
            anchors=("a3", "a4"),
        ),
    )
    document = _build_document(path=path, anchors=anchors, segments=segments)
    engine = TopologyEngine(PathClosingConfig(auto_snap_distance=0.5))

    result = engine.enforce_path_topology(document, "path_locked")

    assert result.topology_status == "open"
    assert result.topology_error is False
    assert len(result.corrections) == 1
    assert result.corrections[0].strategy == "move_right"
    assert result.document.segments[0].params["end"] == [10.0, 0.0]
    assert result.document.segments[1].params["start"] == [10.0, 0.0]


def test_topology_engine_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/topology.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
