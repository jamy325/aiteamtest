import math
import random

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


def _command(*, path_id: str = "path_1", command_id: str = "replace_ellipse") -> dict[str, object]:
    return {
        "command_id": command_id,
        "tool": "propose_replace_path_with_ellipse",
        "path_id": path_id,
        "reason": "intent only",
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }


def _ellipse_points(
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rotation: float,
    count: int,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    rng = random.Random(seed)
    cos_theta = math.cos(rotation)
    sin_theta = math.sin(rotation)
    points = []
    for index in range(count):
        angle = math.tau * index / count
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        px = cx + (x * cos_theta) - (y * sin_theta) + rng.gauss(0.0, noise)
        py = cy + (x * sin_theta) + (y * cos_theta) + rng.gauss(0.0, noise)
        points.append((px, py))
    points.append(points[0])
    return tuple(points)


def test_command_executor_replaces_closed_path_with_ellipse() -> None:
    full_points = _ellipse_points(
        cx=40.0,
        cy=18.0,
        rx=12.0,
        ry=6.0,
        rotation=math.pi / 6.0,
        count=28,
    )
    first_half = full_points[:15]
    second_half = full_points[14:]
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
    assert replacement.type == "ellipse"
    assert replacement.params["cx"] == pytest.approx(40.0, abs=0.5)
    assert replacement.params["cy"] == pytest.approx(18.0, abs=0.5)
    assert replacement.params["rx"] == pytest.approx(12.0, abs=0.8)
    assert replacement.params["ry"] == pytest.approx(6.0, abs=0.8)
    assert replacement.anchors == ("anchor_0",)
    assert document == original_document


def test_command_executor_rejects_open_path_ellipse_replacement() -> None:
    document = _build_document_with_segments(
        (_ellipse_points(cx=30.0, cy=20.0, rx=10.0, ry=6.0, rotation=0.3, count=24)[:-1],),
        closed=False,
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="open_ellipse"), document)

    assert result.success is False
    assert "closed" in (result.reason or "")
    assert result.document == document
    assert result.requires_rerender is False


def test_command_executor_rejects_segment_id_for_path_ellipse_replacement() -> None:
    document = _build_document_with_segments(
        (_ellipse_points(cx=30.0, cy=20.0, rx=10.0, ry=6.0, rotation=0.3, count=24),),
        path_id="strange_ellipse_path_999",
    )
    original_document = document
    executor = CommandExecutor()
    command = _command(path_id="strange_ellipse_path_999", command_id="ellipse_with_segment_id")
    command["segment_id"] = "strange_ellipse_path_999_seg_1"

    result = executor.execute(command, document)

    assert result.success is False
    assert "segment_id" in (result.reason or "")
    assert result.document == original_document
    assert result.requires_rerender is False


def test_command_executor_rejects_locked_path_ellipse_replacement() -> None:
    document = _build_document_with_segments(
        (_ellipse_points(cx=30.0, cy=20.0, rx=10.0, ry=6.0, rotation=0.3, count=24),),
        path_locked=True,
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="locked_ellipse"), document)

    assert result.success is False
    assert "locked path" in (result.reason or "")
    assert result.document == document


def test_command_executor_propagates_ellipse_fitter_failure() -> None:
    document = _build_document_with_segments(
        (_ellipse_points(cx=40.0, cy=18.0, rx=10.0, ry=9.8, rotation=math.pi / 8.0, count=28, noise=0.02, seed=3),),
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="near_circle_ellipse"), document)

    assert result.success is False
    assert result.document == document
    assert (
        "ellipse" in (result.reason or "")
        or "unstable" in (result.reason or "")
        or "similar" in (result.reason or "")
    )


def test_command_executor_rejects_low_confidence_ellipse_replacement() -> None:
    document = _build_document_with_segments(
        (_ellipse_points(cx=40.0, cy=18.0, rx=12.0, ry=6.0, rotation=math.pi / 6.0, count=28, noise=0.5, seed=3),),
    )
    executor = CommandExecutor()

    result = executor.execute(_command(command_id="noisy_ellipse"), document)

    assert result.success is False
    assert result.document == document
    assert result.requires_rerender is False
    assert (
        "confidence" in (result.reason or "")
        or "fit_error" in (result.reason or "")
        or "inlier" in (result.reason or "")
    )
