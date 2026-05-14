import math

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.command_executor import CommandExecutor


def _document_with_segments(
    segments: tuple[Segment, ...],
    *,
    path_id: str = "path_line",
    path_locked: bool = False,
):
    document = create_document(
        document_id=f"doc_{path_id}",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id=path_id, locked=path_locked, segments=tuple(segment.segment_id for segment in segments)))
    for segment in segments:
        document = add_segment(document, segment)
    return document


def _line_command(*, path_id: str = "path_line") -> dict[str, object]:
    return {
        "command_id": "cmd_line",
        "tool": "propose_replace_segment_with_line",
        "path_id": path_id,
        "segment_range": [0, 0],
        "reason": "This region should be a straight line.",
        "confidence": 0.84,
        "requires_user_confirmation": True,
    }


def test_command_executor_executes_line_replacement_successfully() -> None:
    document = _document_with_segments(
        (
            Segment(
                segment_id="seg_polyline",
                path_id="path_line",
                type="polyline",
                params={"points": [[0.0, 0.0], [4.0, 0.1], [8.0, -0.08], [12.0, 0.0]]},
            ),
        )
    )
    executor = CommandExecutor()

    result = executor.execute(_line_command(), document)

    assert result.success is True
    assert result.reason is None
    assert result.topology_status == "open"
    assert result.self_intersection_count == 0
    assert result.document.segments[0].type == "line"
    assert result.document.segments[0].fit_error is not None
    assert result.document.segments[0].confidence is not None
    assert result.new_score is not None
    assert result.old_score is not None
    assert result.new_score < result.old_score


def test_command_executor_rejects_locked_target_segment_for_line_replacement() -> None:
    document = _document_with_segments(
        (
            Segment(
                segment_id="seg_locked_polyline",
                path_id="path_line",
                type="polyline",
                params={"points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]},
                locked=True,
            ),
        )
    )
    executor = CommandExecutor()

    result = executor.execute(_line_command(), document)

    assert result.success is False
    assert "locked segment" in (result.reason or "")
    assert result.document == document
    assert result.requires_rerender is False


def test_command_executor_returns_refinement_feedback_failure_for_low_confidence_line() -> None:
    points = tuple(
        (
            8.0 * math.cos(angle),
            8.0 * math.sin(angle),
        )
        for angle in (0.0, 0.35, 0.7, 1.05, 1.4, 1.75, 2.1, 2.45)
    )
    document = _document_with_segments(
        (
            Segment(
                segment_id="seg_curved_polyline",
                path_id="path_line",
                type="polyline",
                params={"points": [[x, y] for x, y in points]},
            ),
        )
    )
    executor = CommandExecutor()

    result = executor.execute(_line_command(), document)

    assert result.success is False
    assert result.document == document
    assert result.new_score is None
    assert result.requires_rerender is False
    assert result.reason in {"low_inlier_ratio", "high_fit_error", "low_confidence", "unstable_params"}


def test_command_executor_runs_topology_postprocessing_after_line_replacement() -> None:
    document = _document_with_segments(
        (
            Segment(
                segment_id="seg_polyline",
                path_id="path_line",
                type="polyline",
                params={"points": [[0.0, 0.0], [4.0, 0.05], [8.0, -0.05], [12.0, 0.0]]},
            ),
            Segment(
                segment_id="seg_line_2",
                path_id="path_line",
                type="polyline",
                params={"points": [[12.2, 0.0], [16.0, 0.0], [20.0, 0.0]], "start": [12.2, 0.0], "end": [20.0, 0.0]},
            ),
        )
    )
    executor = CommandExecutor()

    result = executor.execute(_line_command(), document)

    assert result.success is True
    assert result.topology_status == "open"
    assert 0.0 < result.document.paths[0].max_gap < 0.5
    first_segment = result.document.segments[0]
    second_segment = result.document.segments[1]
    assert first_segment.params["end"][0] == pytest.approx(11.999687438960075, abs=1e-6)
    assert second_segment.params["start"][0] == pytest.approx(11.999687438960075, abs=1e-6)
