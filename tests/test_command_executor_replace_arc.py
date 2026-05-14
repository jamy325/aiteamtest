import math

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.breakpoint_optimizer import BreakPointResult
from services.command_executor import CommandExecutor


def _build_document_with_segments(
    segment_point_sets: tuple[tuple[tuple[float, float], ...], ...],
    *,
    path_id: str = "path_1",
    closed: bool = False,
    segment_locked: tuple[bool, ...] | None = None,
) -> object:
    document = create_document(
        document_id=f"doc_{path_id}",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    segment_ids = tuple(f"{path_id}_seg_{index + 1}" for index in range(len(segment_point_sets)))
    document = add_path(document, VectorPath(path_id=path_id, closed=closed, segments=segment_ids))
    locked_flags = segment_locked or tuple(False for _ in segment_point_sets)

    for index, points in enumerate(segment_point_sets):
        document = add_segment(
            document,
            Segment(
                segment_id=segment_ids[index],
                path_id=path_id,
                type="polyline",
                params={"points": [[float(x), float(y)] for x, y in points]},
                locked=locked_flags[index],
            ),
        )
    return document


def _command(
    *,
    path_id: str = "path_1",
    segment_range: tuple[int, int] = (0, 0),
    command_id: str = "replace_arc",
) -> dict[str, object]:
    return {
        "command_id": command_id,
        "tool": "propose_replace_segment_with_arc",
        "path_id": path_id,
        "segment_range": [segment_range[0], segment_range[1]],
        "reason": "intent only",
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }


def _arc_points(
    *,
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    count: int,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            cx + radius * math.cos(start_angle + ((end_angle - start_angle) * index / (count - 1))),
            cy + radius * math.sin(start_angle + ((end_angle - start_angle) * index / (count - 1))),
        )
        for index in range(count)
    )


class _FullRangeBreakPointOptimizer:
    def optimize(self, request: object) -> BreakPointResult:
        return BreakPointResult(
            optimized_range=request.rough_range,
            breakpoints=(),
            confidence=0.0,
            reason="full_range",
        )


def test_command_executor_replaces_polyline_with_arc() -> None:
    document = _build_document_with_segments(
        (
            _arc_points(
                cx=20.0,
                cy=10.0,
                radius=6.0,
                start_angle=0.25,
                end_angle=1.4,
                count=10,
            ),
        )
    )
    executor = CommandExecutor()

    result = executor.execute(_command(), document)

    assert result.success is True
    assert result.topology_status == "open"
    assert result.self_intersection_count == 0
    assert result.old_score is not None
    assert result.new_score is not None
    assert result.requires_rerender is True
    segment = result.document.segments[0]
    assert segment.type == "arc"
    assert segment.params["cx"] == pytest.approx(20.0, abs=0.3)
    assert segment.params["cy"] == pytest.approx(10.0, abs=0.3)
    assert segment.params["r"] == pytest.approx(6.0, abs=0.3)


def test_command_executor_rejects_locked_arc_target_without_mutation() -> None:
    document = _build_document_with_segments(
        (
            _arc_points(
                cx=12.0,
                cy=8.0,
                radius=5.0,
                start_angle=0.1,
                end_angle=1.1,
                count=8,
            ),
        ),
        segment_locked=(True,),
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="locked_arc"), document)

    assert result.success is False
    assert "locked" in (result.reason or "")
    assert result.document == document
    assert result.requires_rerender is False


def test_command_executor_rejects_short_arc_with_low_coverage_feedback() -> None:
    document = _build_document_with_segments(
        (
            _arc_points(
                cx=20.0,
                cy=10.0,
                radius=12.0,
                start_angle=0.2,
                end_angle=0.45,
                count=8,
            ),
        )
    )
    executor = CommandExecutor(breakpoint_optimizer=_FullRangeBreakPointOptimizer())

    result = executor.execute(_command(command_id="short_arc"), document)

    assert result.success is False
    assert "coverage" in (result.reason or "")
    assert result.document == document
    assert result.new_score is None
    assert result.requires_rerender is False


def test_command_executor_replaces_segment_range_with_arc_and_keeps_path_consistent() -> None:
    first = _arc_points(
        cx=20.0,
        cy=10.0,
        radius=6.0,
        start_angle=0.25,
        end_angle=0.8,
        count=6,
    )
    second = _arc_points(
        cx=20.0,
        cy=10.0,
        radius=6.0,
        start_angle=0.8,
        end_angle=1.4,
        count=6,
    )
    document = _build_document_with_segments((first, second))
    executor = CommandExecutor()

    result = executor.execute(_command(segment_range=(0, 1), command_id="range_arc"), document)

    assert result.success is True
    assert result.topology_status == "open"
    assert result.affected_segments == ("path_1_seg_1", "path_1_seg_2")
    assert result.document.paths[0].segments == ("path_1_seg_1",)
    assert len(result.document.segments) == 1
    assert result.document.segments[0].type == "arc"
