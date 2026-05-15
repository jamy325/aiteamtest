from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.document import add_anchor, add_constraint, add_path, add_segment, create_document
from core.types import Anchor, Constraint, CoordinateSystem, Path as VectorPath, Segment
from services.shared_tangent import SharedTangentOptimizer


def _line_segment(*, segment_id: str, path_id: str, start: tuple[float, float], end: tuple[float, float], anchors: tuple[str, str], locked: bool = False) -> Segment:
    return Segment(
        segment_id=segment_id,
        path_id=path_id,
        type="line",
        params={"start": [start[0], start[1]], "end": [end[0], end[1]]},
        anchors=anchors,
        locked=locked,
    )


def _arc_segment(
    *,
    segment_id: str,
    path_id: str,
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    anchors: tuple[str, str],
    direction: str = "ccw",
    locked: bool = False,
) -> Segment:
    return Segment(
        segment_id=segment_id,
        path_id=path_id,
        type="arc",
        params={
            "cx": center[0],
            "cy": center[1],
            "r": radius,
            "start_angle": start_angle,
            "end_angle": end_angle,
            "direction": direction,
        },
        anchors=anchors,
        locked=locked,
    )


def test_shared_tangent_optimizer_improves_line_arc_g1_pair() -> None:
    optimizer = SharedTangentOptimizer()
    anchor = Anchor("a1", "p1", position=(10.2, 0.15))
    line = _line_segment(segment_id="line", path_id="p1", start=(0.0, 0.0), end=(10.2, 0.15), anchors=("a0", "a1"))
    arc = _arc_segment(
        segment_id="arc",
        path_id="p1",
        center=(10.0, 10.0),
        radius=10.0,
        start_angle=-1.52,
        end_angle=-0.35,
        anchors=("a1", "a2"),
    )
    support_points = (
        (2.0, 0.0),
        (6.0, 0.02),
        (10.0, 0.0),
        (11.8, 0.18),
        (13.2, 0.58),
    )
    before_mismatch = optimizer._tangent_mismatch(
        optimizer._line_outward_tangent(line, anchor),
        optimizer._arc_outward_tangent(arc, anchor),
    )

    result = optimizer.optimize_pair(line, arc, anchor, support_points)

    assert result.success is True
    assert result.shared_tangent is not None
    assert result.violation < 0.05
    assert result.tangent_mismatch < before_mismatch
    assert result.tangent_mismatch < 0.05
    assert result.confidence >= 0.5
    assert result.segment_a.segment_id == "line"
    assert result.segment_b.segment_id == "arc"
    assert result.segment_a.params["end"] != line.params["end"]
    optimized_end = tuple(result.segment_a.params["end"])
    optimized_start_angle = float(result.segment_b.params["start_angle"])
    derived_arc_point = (
        float(result.segment_b.params["cx"]) + (float(result.segment_b.params["r"]) * math.cos(optimized_start_angle)),
        float(result.segment_b.params["cy"]) + (float(result.segment_b.params["r"]) * math.sin(optimized_start_angle)),
    )
    assert optimized_end == pytest.approx(derived_arc_point, abs=1e-6)
    assert line.params["end"] == [10.2, 0.15]
    assert arc.params["start_angle"] == pytest.approx(-1.52)


def test_shared_tangent_optimizer_skips_locked_segment() -> None:
    optimizer = SharedTangentOptimizer()
    anchor = Anchor("a1", "p1", position=(10.0, 0.0))
    line = _line_segment(
        segment_id="line",
        path_id="p1",
        start=(0.0, 0.0),
        end=(10.0, 0.0),
        anchors=("a0", "a1"),
        locked=True,
    )
    arc = _arc_segment(
        segment_id="arc",
        path_id="p1",
        center=(10.0, 10.0),
        radius=10.0,
        start_angle=-1.50,
        end_angle=-0.40,
        anchors=("a1", "a2"),
    )

    result = optimizer.optimize_pair(line, arc, anchor, ())

    assert result.success is False
    assert "locked" in result.reason
    assert result.segment_a == line
    assert result.segment_b == arc


def test_shared_tangent_optimizer_reads_constraints_and_rejects_low_confidence() -> None:
    doc = create_document("doc", 100.0, 100.0, CoordinateSystem())
    doc = add_path(doc, VectorPath("p1", segments=("line", "arc")))
    doc = add_anchor(doc, Anchor("a1", "p1", position=(10.0, 0.0)))
    doc = add_segment(doc, _line_segment(segment_id="line", path_id="p1", start=(0.0, 0.0), end=(10.0, 0.0), anchors=("a0", "a1")))
    doc = add_segment(
        doc,
        _arc_segment(
            segment_id="arc",
            path_id="p1",
            center=(10.0, 10.0),
            radius=10.0,
            start_angle=-1.52,
            end_angle=-0.35,
            anchors=("a1", "a2"),
        ),
    )
    doc = add_constraint(
        doc,
        Constraint(
            constraint_id="g1_low",
            type="g1_continuity",
            targets=("line", "arc", "a1"),
            confidence=0.2,
            locked=False,
        ),
    )

    optimizer = SharedTangentOptimizer(min_confidence=0.5)
    results = optimizer.optimize_document(doc)

    assert len(results) == 1
    assert results[0].success is False
    assert "low confidence" in results[0].reason
    assert results[0].constraint_id == "g1_low"


def test_shared_tangent_optimizer_has_no_forbidden_dependencies() -> None:
    source = Path("services/shared_tangent.py").read_text(encoding="utf-8")
    assert "cv2" not in source
    assert "PyQt" not in source
    assert "openai" not in source
