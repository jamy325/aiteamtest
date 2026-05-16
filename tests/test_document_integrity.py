from __future__ import annotations

import math

from core.document import add_anchor, add_constraint, add_path, add_segment, create_document
from core.types import Anchor, Constraint, CoordinateSystem, Path as VectorPath, Segment, updated
from services.command_executor import CommandExecutor
from services.document_integrity import DocumentIntegrityValidator


def _build_document(
    *,
    path_id: str,
    closed: bool,
    segments: tuple[Segment, ...],
    anchors: tuple[Anchor, ...] = (),
    constraints: tuple[Constraint, ...] = (),
):
    document = create_document(
        document_id=f"doc_{path_id}",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id=path_id,
            closed=closed,
            segments=tuple(segment.segment_id for segment in segments),
        ),
    )
    for segment in segments:
        document = add_segment(document, segment)
    for anchor in anchors:
        document = add_anchor(document, anchor)
    for constraint in constraints:
        document = add_constraint(document, constraint)
    return document


def _circle_points(*, cx: float, cy: float, radius: float, count: int) -> tuple[tuple[float, float], ...]:
    points = [
        (
            cx + radius * math.cos(math.tau * index / count),
            cy + radius * math.sin(math.tau * index / count),
        )
        for index in range(count)
    ]
    points.append(points[0])
    return tuple(points)


def _ellipse_points(
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rotation: float,
    count: int,
) -> tuple[tuple[float, float], ...]:
    cos_theta = math.cos(rotation)
    sin_theta = math.sin(rotation)
    points = []
    for index in range(count):
        angle = math.tau * index / count
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        points.append(
            (
                cx + (x * cos_theta) - (y * sin_theta),
                cy + (x * sin_theta) + (y * cos_theta),
            )
        )
    points.append(points[0])
    return tuple(points)


def test_replace_path_with_circle_cleans_dangling_segments_anchors_and_constraints() -> None:
    full_points = _circle_points(cx=30.0, cy=25.0, radius=8.0, count=12)
    first_half = full_points[:7]
    second_half = full_points[6:]
    anchors = (
        Anchor("a0", "circle_path", first_half[0]),
        Anchor("a1", "circle_path", first_half[-1]),
    )
    segments = (
        Segment("circle_seg_1", "circle_path", "polyline", {"points": [[x, y] for x, y in first_half]}, anchors=("a0", "a1")),
        Segment("circle_seg_2", "circle_path", "polyline", {"points": [[x, y] for x, y in second_half]}, anchors=("a1", "a0")),
    )
    constraints = (
        Constraint("c_remove", "coincident", targets=("circle_seg_2", "a1")),
    )
    document = _build_document(
        path_id="circle_path",
        closed=True,
        segments=segments,
        anchors=anchors,
        constraints=constraints,
    )

    result = CommandExecutor().execute(
        {
            "command_id": "replace_circle",
            "tool": "propose_replace_path_with_circle",
            "path_id": "circle_path",
            "reason": "intent only",
            "confidence": 0.85,
            "requires_user_confirmation": True,
        },
        document,
    )

    assert result.success is True
    assert result.document.paths[0].segments == ("circle_seg_1",)
    assert tuple(segment.segment_id for segment in result.document.segments) == ("circle_seg_1",)
    assert tuple(anchor.anchor_id for anchor in result.document.anchors) == ("a0",)
    assert result.document.constraints == ()
    assert result.document.metadata["command_executor_cleanup"]["removed_segment_ids"] == ["circle_seg_2"]
    assert result.document.metadata["command_executor_cleanup"]["removed_anchor_ids"] == ["a1"]
    assert result.document.metadata["command_executor_cleanup"]["removed_constraint_ids"] == ["c_remove"]
    assert tuple(segment.segment_id for segment in document.segments) == ("circle_seg_1", "circle_seg_2")
    assert tuple(anchor.anchor_id for anchor in document.anchors) == ("a0", "a1")


def test_replace_path_with_ellipse_cleans_dangling_segments() -> None:
    full_points = _ellipse_points(
        cx=40.0,
        cy=18.0,
        rx=12.0,
        ry=6.0,
        rotation=math.pi / 6.0,
        count=24,
    )
    first_half = full_points[:13]
    second_half = full_points[12:]
    anchors = (
        Anchor("a0", "ellipse_path", first_half[0]),
        Anchor("a1", "ellipse_path", first_half[-1]),
    )
    segments = (
        Segment("ellipse_seg_1", "ellipse_path", "polyline", {"points": [[x, y] for x, y in first_half]}, anchors=("a0", "a1")),
        Segment("ellipse_seg_2", "ellipse_path", "polyline", {"points": [[x, y] for x, y in second_half]}, anchors=("a1", "a0")),
    )
    document = _build_document(
        path_id="ellipse_path",
        closed=True,
        segments=segments,
        anchors=anchors,
    )

    result = CommandExecutor().execute(
        {
            "command_id": "replace_ellipse",
            "tool": "propose_replace_path_with_ellipse",
            "path_id": "ellipse_path",
            "reason": "intent only",
            "confidence": 0.85,
            "requires_user_confirmation": True,
        },
        document,
    )

    assert result.success is True
    assert result.document.paths[0].segments == ("ellipse_seg_1",)
    assert tuple(segment.segment_id for segment in result.document.segments) == ("ellipse_seg_1",)
    assert tuple(anchor.anchor_id for anchor in result.document.anchors) == ("a0",)
    assert result.document.metadata["command_executor_cleanup"]["removed_segment_ids"] == ["ellipse_seg_2"]
    assert result.document.metadata["command_executor_cleanup"]["removed_anchor_ids"] == ["a1"]


def test_replace_segment_range_cleans_removed_segment_anchor_and_constraint_targets() -> None:
    segment_1_points = ((0.0, 0.0), (1.0, 0.0))
    segment_2_points = ((1.0, 0.0), (2.0, 0.04), (3.0, 0.0))
    segment_3_points = ((3.0, 0.0), (4.0, -0.04), (5.0, 0.0))
    anchors = (
        Anchor("a0", "line_path", segment_1_points[0]),
        Anchor("a1", "line_path", segment_1_points[-1]),
        Anchor("a2", "line_path", segment_2_points[-1]),
        Anchor("a3", "line_path", segment_3_points[-1]),
    )
    segments = (
        Segment("line_seg_1", "line_path", "line", {"start": [segment_1_points[0][0], segment_1_points[0][1]], "end": [segment_1_points[-1][0], segment_1_points[-1][1]]}, anchors=("a0", "a1")),
        Segment("line_seg_2", "line_path", "polyline", {"points": [[x, y] for x, y in segment_2_points]}, anchors=("a1", "a2")),
        Segment("line_seg_3", "line_path", "polyline", {"points": [[x, y] for x, y in segment_3_points]}, anchors=("a2", "a3")),
    )
    constraints = (
        Constraint("c_drop", "coincident", targets=("line_seg_3", "a2")),
    )
    document = _build_document(
        path_id="line_path",
        closed=False,
        segments=segments,
        anchors=anchors,
        constraints=constraints,
    )

    result = CommandExecutor().execute(
        {
            "command_id": "replace_line_range",
            "tool": "propose_replace_segment_with_line",
            "path_id": "line_path",
            "segment_range": [1, 2],
            "reason": "intent only",
            "confidence": 0.85,
            "requires_user_confirmation": True,
        },
        document,
    )

    assert result.success is True
    assert result.document.paths[0].segments == ("line_seg_1", "line_seg_2")
    assert tuple(segment.segment_id for segment in result.document.segments) == ("line_seg_1", "line_seg_2")
    assert tuple(anchor.anchor_id for anchor in result.document.anchors) == ("a0", "a1", "a3")
    assert result.document.constraints == ()
    assert result.document.metadata["command_executor_cleanup"]["removed_segment_ids"] == ["line_seg_3"]
    assert result.document.metadata["command_executor_cleanup"]["removed_anchor_ids"] == ["a2"]
    assert result.document.metadata["command_executor_cleanup"]["removed_constraint_ids"] == ["c_drop"]


def test_document_integrity_validator_accepts_normal_document() -> None:
    segments = (
        Segment("line_seg_1", "normal_path", "line", {"start": [0.0, 0.0], "end": [4.0, 0.0]}, anchors=("a0", "a1")),
        Segment("line_seg_2", "normal_path", "line", {"start": [4.0, 0.0], "end": [8.0, 0.0]}, anchors=("a1", "a2")),
    )
    anchors = (
        Anchor("a0", "normal_path", (0.0, 0.0)),
        Anchor("a1", "normal_path", (4.0, 0.0)),
        Anchor("a2", "normal_path", (8.0, 0.0)),
    )
    document = _build_document(path_id="normal_path", closed=False, segments=segments, anchors=anchors)

    report = DocumentIntegrityValidator().validate(document)

    assert report.success is True
    assert report.errors == ()
    assert report.warnings == ()
    assert report.affected_ids == ()


def test_document_integrity_validator_reports_dangling_segment() -> None:
    document = create_document(
        document_id="doc_dangling_segment",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_ok", closed=False, segments=("seg_used",)))
    document = add_segment(document, Segment("seg_used", "path_ok", "line", {"start": [0.0, 0.0], "end": [1.0, 0.0]}))
    document = updated(
        document,
        segments=document.segments
        + (Segment("seg_dangling", "path_ok", "line", {"start": [1.0, 0.0], "end": [2.0, 0.0]}),),
    )

    report = DocumentIntegrityValidator().validate(document)

    assert report.success is False
    assert any(issue.code == "DANGLING_SEGMENT" for issue in report.errors)
    assert "seg_dangling" in report.affected_ids


def test_document_integrity_validator_reports_dangling_anchor() -> None:
    document = create_document(
        document_id="doc_dangling_anchor",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_ok", closed=False, segments=("seg_used",)))
    document = add_segment(document, Segment("seg_used", "path_ok", "line", {"start": [0.0, 0.0], "end": [1.0, 0.0]}, anchors=("a0",)))
    document = add_anchor(document, Anchor("a0", "path_ok", (0.0, 0.0)))
    document = add_anchor(document, Anchor("a_orphan", "path_ok", (3.0, 0.0)))

    report = DocumentIntegrityValidator().validate(document)

    assert report.success is False
    assert any(issue.code == "DANGLING_ANCHOR" for issue in report.errors)
    assert "a_orphan" in report.affected_ids


def test_document_integrity_validator_reports_dangling_constraint_target() -> None:
    document = create_document(
        document_id="doc_dangling_constraint",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_ok", closed=False, segments=("seg_used",)))
    document = add_segment(document, Segment("seg_used", "path_ok", "line", {"start": [0.0, 0.0], "end": [1.0, 0.0]}))
    document = add_constraint(document, Constraint("c_bad", "coincident", targets=("missing_seg",)))

    report = DocumentIntegrityValidator().validate(document)

    assert report.success is False
    assert any(issue.code == "DANGLING_CONSTRAINT_TARGET" for issue in report.errors)
    assert "c_bad" in report.affected_ids
    assert "missing_seg" in report.affected_ids


def test_document_integrity_validator_reports_non_vector_coordinate_space() -> None:
    document = create_document(
        document_id="doc_non_vector",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="pixel"),
    )

    report = DocumentIntegrityValidator().validate(document)

    assert report.success is False
    assert any(issue.code == "NON_VECTOR_COORDINATE_SPACE" for issue in report.errors)


def test_document_integrity_validator_reports_closed_path_gap_mismatch() -> None:
    segments = (
        Segment("line_seg_1", "closed_gap_path", "line", {"start": [0.0, 0.0], "end": [4.0, 0.0]}),
        Segment("line_seg_2", "closed_gap_path", "line", {"start": [4.0, 0.0], "end": [4.0, 4.0]}),
        Segment("line_seg_3", "closed_gap_path", "line", {"start": [4.0, 4.0], "end": [0.0, 4.0]}),
        Segment("line_seg_4", "closed_gap_path", "line", {"start": [0.0, 4.0], "end": [0.5, 0.5]}),
    )
    document = _build_document(path_id="closed_gap_path", closed=True, segments=segments)
    document = add_path(
        create_document(
            document_id="doc_closed_gap",
            width=200.0,
            height=200.0,
            coordinate_system=CoordinateSystem(internal_space="vector"),
        ),
        VectorPath(
            path_id="closed_gap_path",
            closed=True,
            topology_status="closed",
            max_gap=0.0,
            segments=tuple(segment.segment_id for segment in segments),
        ),
    )
    for segment in segments:
        document = add_segment(document, segment)

    report = DocumentIntegrityValidator().validate(document)

    assert report.success is False
    assert any(issue.code == "CLOSED_PATH_GAP_MISMATCH" for issue in report.errors)
    assert "closed_gap_path" in report.affected_ids


def test_document_integrity_validator_reports_degree_angle_contract_violation() -> None:
    document = create_document(
        document_id="doc_degree_arc",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="arc_path", closed=False, segments=("arc_seg",)))
    document = add_segment(
        document,
        Segment(
            "arc_seg",
            "arc_path",
            "arc",
            {
                "cx": 10.0,
                "cy": 10.0,
                "r": 5.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
                "direction": "ccw",
                "angle_unit": "degree",
            },
        ),
    )

    report = DocumentIntegrityValidator().validate(document)

    assert report.success is False
    assert any(issue.code == "ANGLE_UNIT_NOT_RADIANS" for issue in report.errors)
    assert "arc_seg" in report.affected_ids
