from __future__ import annotations

import math

import pytest

from core.document import add_anchor, add_constraint, add_path, add_segment, create_document
from core.types import Anchor, Constraint, CoordinateSystem, Path as VectorPath, Segment
from services.scorer import Scorer


def _build_document(*, anchors: tuple[Anchor, ...], segments: tuple[Segment, ...], constraints: tuple[Constraint, ...] = ()):
    document = create_document("doc_shared_score", 100.0, 100.0, CoordinateSystem())
    document = add_path(document, VectorPath("p1", segments=tuple(segment.segment_id for segment in segments)))
    for anchor in anchors:
        document = add_anchor(document, anchor)
    for segment in segments:
        document = add_segment(document, segment)
    for constraint in constraints:
        document = add_constraint(document, constraint)
    return document


def test_shared_tangent_score_is_zero_without_constraints() -> None:
    document = _build_document(
        anchors=(Anchor("a1", "p1", position=(10.0, 0.0), shared_tangent=(1.0, 0.0)),),
        segments=(
            Segment("seg_1", "p1", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")),
            Segment("seg_2", "p1", "line", params={"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a2")),
        ),
    )

    result = Scorer().score_document(document)

    assert result.breakdown.shared_tangent_violation_score == pytest.approx(0.0)


def test_shared_tangent_score_is_zero_when_constraint_is_satisfied() -> None:
    document = _build_document(
        anchors=(Anchor("a1", "p1", position=(10.0, 0.0), shared_tangent=(1.0, 0.0)),),
        segments=(
            Segment("seg_1", "p1", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")),
            Segment("seg_2", "p1", "line", params={"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a2")),
        ),
        constraints=(
            Constraint("g1_ok", "shared_tangent", targets=("seg_1", "seg_2", "a1")),
        ),
    )

    result = Scorer().score_document(document)

    assert result.breakdown.shared_tangent_violation_score == pytest.approx(0.0)


def test_shared_tangent_score_penalizes_violation() -> None:
    document = _build_document(
        anchors=(Anchor("a1", "p1", position=(10.0, 0.0), shared_tangent=(1.0, 0.0)),),
        segments=(
            Segment("seg_1", "p1", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")),
            Segment("seg_2", "p1", "line", params={"start": [10.0, 0.0], "end": [20.0, 5.0]}, anchors=("a1", "a2")),
        ),
        constraints=(
            Constraint("g1_bad", "g1_continuity", targets=("seg_1", "seg_2", "a1")),
        ),
    )

    result = Scorer().score_document(document)

    assert result.breakdown.shared_tangent_violation_score > 0.0


def test_shared_tangent_score_includes_locked_constraint() -> None:
    document = _build_document(
        anchors=(Anchor("a1", "p1", position=(10.0, 0.0), shared_tangent=(1.0, 0.0)),),
        segments=(
            Segment(
                "seg_1",
                "p1",
                "arc",
                params={"cx": 10.0, "cy": 10.0, "r": 10.0, "start_angle": -math.pi / 2.0, "end_angle": -0.2, "direction": "ccw"},
                anchors=("a1", "a2"),
            ),
            Segment(
                "seg_2",
                "p1",
                "bezier",
                params={
                    "start": [10.0, 0.0],
                    "control1": [8.8, 1.6],
                    "control2": [12.0, 3.0],
                    "end": [14.0, 4.0],
                },
                anchors=("a1", "a3"),
            ),
        ),
        constraints=(
            Constraint("shared_locked", "shared_tangent", targets=("seg_1", "seg_2", "a1"), locked=True),
        ),
    )

    result = Scorer().score_document(document)

    assert result.breakdown.shared_tangent_violation_score > 0.0
