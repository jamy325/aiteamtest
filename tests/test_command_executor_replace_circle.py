import math

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.command_executor import CommandExecutor


def _build_document_with_segments(
    segment_point_sets: tuple[tuple[tuple[float, float], ...], ...],
    *,
    path_id: str = "path_1",
    closed: bool = True,
    path_locked: bool = False,
    segment_locked: tuple[bool, ...] | None = None,
    segment_anchors: tuple[tuple[str, ...], ...] | None = None,
) -> object:
    document = create_document(
        document_id=f"doc_{path_id}",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    segment_ids = tuple(f"{path_id}_seg_{index + 1}" for index in range(len(segment_point_sets)))
    document = add_path(
        document,
        VectorPath(path_id=path_id, closed=closed, locked=path_locked, segments=segment_ids),
    )
    locked_flags = segment_locked or tuple(False for _ in segment_point_sets)
    anchor_sets = segment_anchors or tuple(() for _ in segment_point_sets)

    for index, points in enumerate(segment_point_sets):
        document = add_segment(
            document,
            Segment(
                segment_id=segment_ids[index],
                path_id=path_id,
                type="polyline",
                params={"points": [[float(x), float(y)] for x, y in points]},
                locked=locked_flags[index],
                anchors=anchor_sets[index],
            ),
        )
    return document


def _command(*, path_id: str = "path_1", command_id: str = "replace_circle") -> dict[str, object]:
    return {
        "command_id": command_id,
        "tool": "propose_replace_path_with_circle",
        "path_id": path_id,
        "reason": "intent only",
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }


def _circle_points(
    *,
    cx: float,
    cy: float,
    radius: float,
    count: int,
    noise: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    points = []
    for index in range(count):
        angle = math.tau * index / count
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        if noise:
            x += noise * math.cos(3.0 * angle)
            y += noise * math.sin(5.0 * angle)
        points.append((x, y))
    points.append(points[0])
    return tuple(points)


def _ellipse_points(
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    count: int,
) -> tuple[tuple[float, float], ...]:
    points = []
    for index in range(count):
        angle = math.tau * index / count
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    points.append(points[0])
    return tuple(points)


def test_command_executor_replaces_closed_path_with_circle() -> None:
    first_half = _circle_points(cx=30.0, cy=25.0, radius=8.0, count=12)[:7]
    second_half = _circle_points(cx=30.0, cy=25.0, radius=8.0, count=12)[6:]
    document = _build_document_with_segments(
        (first_half, second_half),
        segment_anchors=(("anchor_0", "anchor_1"), ("anchor_1", "anchor_0")),
    )
    original_document = document
    executor = CommandExecutor()

    result = executor.execute(_command(), document)

    assert result.success is True
    assert result.topology_status == "closed"
    assert result.self_intersection_count == 0
    assert result.affected_paths == ("path_1",)
    assert result.affected_segments == ("path_1_seg_1", "path_1_seg_2")
    assert result.document.paths[0].segments == ("path_1_seg_1",)
    assert len(result.document.segments) == 1
    replacement = result.document.segments[0]
    assert replacement.type == "circle"
    assert replacement.params["cx"] == pytest.approx(30.0, abs=0.3)
    assert replacement.params["cy"] == pytest.approx(25.0, abs=0.3)
    assert replacement.params["r"] == pytest.approx(8.0, abs=0.3)
    assert replacement.anchors == ("anchor_0",)
    assert document == original_document


def test_command_executor_rejects_open_path_circle_replacement() -> None:
    document = _build_document_with_segments(
        (_circle_points(cx=30.0, cy=25.0, radius=8.0, count=16)[:-1],),
        closed=False,
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="open_circle"), document)

    assert result.success is False
    assert "closed" in (result.reason or "")
    assert result.document == document
    assert result.requires_rerender is False


def test_command_executor_rejects_locked_path_circle_replacement() -> None:
    document = _build_document_with_segments(
        (_circle_points(cx=18.0, cy=12.0, radius=5.0, count=16),),
        path_locked=True,
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="locked_circle"), document)

    assert result.success is False
    assert "locked path" in (result.reason or "")
    assert result.document == document


def test_command_executor_rejects_low_confidence_circle_replacement() -> None:
    document = _build_document_with_segments(
        (_circle_points(cx=30.0, cy=25.0, radius=8.0, count=24, noise=2.4),),
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="noisy_circle"), document)

    assert result.success is False
    assert result.document == document
    assert result.requires_rerender is False
    assert (
        "radial" in (result.reason or "")
        or "fit_error" in (result.reason or "")
        or "inlier" in (result.reason or "")
    )


def test_command_executor_accepts_stable_circle_low_confidence_gate() -> None:
    executor = CommandExecutor()

    assert (
        executor._should_reject_circle_feedback(
            "low_confidence",
            rmse=0.15,
            radial_error=0.09,
            inlier_ratio=0.94,
        )
        is False
    )
    assert (
        executor._should_reject_circle_feedback(
            "low_confidence",
            rmse=0.22,
            radial_error=0.09,
            inlier_ratio=0.94,
        )
        is True
    )
