from __future__ import annotations

import math

from core.document import add_constraint, add_path, add_segment, create_document
from core.types import Constraint, CoordinateSystem, Path as VectorPath, Segment
from services.g1_candidate_detector import G1Candidate, G1CandidateDetector


def test_line_line_smooth() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    doc = add_segment(doc, Segment("s1", "p1", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")))
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a2")))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 1
    assert candidates[0].anchor_id == "a1"
    assert candidates[0].confidence == 1.0


def test_line_line_sharp() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    doc = add_segment(doc, Segment("s1", "p1", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")))
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [10.0, 10.0]}, anchors=("a1", "a2")))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 0


def test_line_arc_smooth() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    doc = add_segment(doc, Segment("s1", "p1", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")))
    doc = add_segment(
        doc,
        Segment(
            "s2",
            "p1",
            "arc",
            {
                "cx": 10.0,
                "cy": 10.0,
                "r": 10.0,
                "start_angle": -math.pi / 2.0,
                "end_angle": 0.0,
                "direction": "ccw",
            },
            anchors=("a1", "a2"),
        ),
    )

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 1
    assert candidates[0].anchor_id == "a1"


def test_arc_line_smooth() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    doc = add_segment(
        doc,
        Segment(
            "s1",
            "p1",
            "arc",
            {
                "cx": 10.0,
                "cy": 10.0,
                "r": 10.0,
                "start_angle": -math.pi,
                "end_angle": -math.pi / 2.0,
                "direction": "ccw",
            },
            anchors=("a0", "a1"),
        ),
    )
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a2")))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 1
    assert candidates[0].anchor_id == "a1"


def test_bezier_line_smooth() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    doc = add_segment(
        doc,
        Segment(
            "s1",
            "p1",
            "bezier",
            {
                "start": [0.0, 0.0],
                "control1": [5.0, 0.0],
                "control2": [8.0, 0.0],
                "end": [10.0, 0.0],
            },
            anchors=("a0", "a1"),
        ),
    )
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a2")))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 1


def test_closed_path_adjacency() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2"), closed=True))
    doc = add_segment(doc, Segment("s1", "p1", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")))
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a0")))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 2
    anchors = {c.anchor_id for c in candidates}
    assert anchors == {"a0", "a1"}


def test_locked_constraint_no_override() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    doc = add_segment(doc, Segment("s1", "p1", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")))
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a2")))
    doc = add_constraint(doc, Constraint("c1", "shared_tangent", targets=("a1",), locked=True))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 1

    constraints = detector.generate_constraints(doc, candidates)
    assert len(constraints) == 0


def test_generate_constraints() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    doc = add_segment(doc, Segment("s1", "p1", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")))
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [20.0, 0.0]}, anchors=("a1", "a2")))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    constraints = detector.generate_constraints(doc, candidates)

    assert len(constraints) == 1
    assert constraints[0].type == "g1_continuity"
    assert constraints[0].strength < 1.0
    assert not constraints[0].locked

def test_misjudged_filtered() -> None:
    # ensure angles above tolerance are filtered
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("s1", "s2")))
    # Angle is slightly > 0.1 tolerance (0.15 rad ~ 8.5 degrees)
    doc = add_segment(doc, Segment("s1", "p1", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")))
    doc = add_segment(doc, Segment("s2", "p1", "line", {"start": [10.0, 0.0], "end": [20.0, 1.5]}, anchors=("a1", "a2")))

    detector = G1CandidateDetector(angle_tolerance=0.1)
    candidates = detector.detect_candidates(doc)
    assert len(candidates) == 0
