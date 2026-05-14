import ast
import math
from pathlib import Path

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.command_executor import CommandExecutor


def _build_polyline_document(
    points: tuple[tuple[float, float], ...],
    *,
    path_id: str = "path_1",
    closed: bool = False,
    locked: bool = False,
) -> object:
    document = create_document(
        document_id=f"doc_{path_id}",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id=path_id, closed=closed, locked=locked))
    document = add_segment(
        document,
        Segment(
            segment_id=f"{path_id}_seg_1",
            path_id=path_id,
            type="polyline",
            params={"points": [[float(x), float(y)] for x, y in points]},
        ),
    )
    return document


def _command(tool: str, *, path_id: str = "path_1", command_id: str | None = None) -> dict[str, object]:
    return {
        "command_id": command_id or tool,
        "tool": tool,
        "path_id": path_id,
        "segment_range": [0, 0],
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


def test_command_executor_replaces_polyline_with_line_and_updates_scores() -> None:
    document = _build_polyline_document(((0.0, 0.0), (2.0, 0.15), (4.0, -0.1), (6.0, 0.0)))
    executor = CommandExecutor()

    result = executor.execute(_command("propose_replace_segment_with_line"), document)

    assert result.success is True
    assert result.command_id == "propose_replace_segment_with_line"
    assert result.affected_paths == ("path_1",)
    assert result.affected_segments == ("path_1_seg_1",)
    assert result.requires_rerender is True
    assert result.old_score is not None
    assert result.new_score is not None
    assert result.new_score < result.old_score
    assert result.topology_status == "open"
    assert result.self_intersection_count == 0
    assert result.document != document
    assert result.document.segments[0].type == "line"


def test_command_executor_replaces_polyline_with_arc() -> None:
    document = _build_polyline_document(
        _arc_points(
            cx=20.0,
            cy=10.0,
            radius=6.0,
            start_angle=0.25,
            end_angle=1.4,
            count=10,
        )
    )
    executor = CommandExecutor()

    result = executor.execute(_command("propose_replace_segment_with_arc"), document)

    assert result.success is True
    assert result.topology_status == "open"
    segment = result.document.segments[0]
    assert segment.type == "arc"
    assert segment.params["cx"] == pytest.approx(20.0, abs=0.3)
    assert segment.params["cy"] == pytest.approx(10.0, abs=0.3)
    assert segment.params["r"] == pytest.approx(6.0, abs=0.3)


def test_command_executor_replaces_closed_circle_like_polyline_with_circle() -> None:
    document = _build_polyline_document(_circle_points(cx=30.0, cy=25.0, radius=8.0, count=24), closed=True)
    executor = CommandExecutor()

    result = executor.execute(_command("propose_replace_segment_with_circle"), document)

    assert result.success is True
    assert result.topology_status == "closed"
    assert result.self_intersection_count == 0
    segment = result.document.segments[0]
    assert segment.type == "circle"
    assert segment.params["cx"] == pytest.approx(30.0, abs=0.4)
    assert segment.params["cy"] == pytest.approx(25.0, abs=0.4)
    assert segment.params["r"] == pytest.approx(8.0, abs=0.4)


def test_command_executor_replaces_closed_ellipse_like_polyline_with_ellipse() -> None:
    document = _build_polyline_document(
        _ellipse_points(
            cx=40.0,
            cy=18.0,
            rx=12.0,
            ry=6.0,
            rotation=math.pi / 6.0,
            count=28,
        ),
        closed=True,
    )
    executor = CommandExecutor()

    result = executor.execute(_command("propose_replace_segment_with_ellipse"), document)

    assert result.success is True
    assert result.topology_status == "closed"
    segment = result.document.segments[0]
    assert segment.type == "ellipse"
    assert segment.params["cx"] == pytest.approx(40.0, abs=0.5)
    assert segment.params["cy"] == pytest.approx(18.0, abs=0.5)
    assert segment.params["rx"] == pytest.approx(12.0, abs=0.8)
    assert segment.params["ry"] == pytest.approx(6.0, abs=0.8)


def test_command_executor_rejects_batch_execution_without_mutation() -> None:
    document = _build_polyline_document(((0.0, 0.0), (2.0, 0.0), (4.0, 0.0)))
    original_document = document
    executor = CommandExecutor()
    command = {
        "command_id": "batch_cmd",
        "tool": "propose_batch_refinement",
        "summary": "group review",
        "commands": [_command("propose_replace_segment_with_line")],
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }

    result = executor.execute(command, document)

    assert result.success is False
    assert "batch command execution" in (result.reason or "")
    assert result.document == original_document
    assert result.requires_rerender is False


def test_command_executor_returns_failure_without_partial_write_for_unfit_shape() -> None:
    document = _build_polyline_document(((0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0)))
    original_document = document
    executor = CommandExecutor()

    result = executor.execute(_command("propose_replace_segment_with_circle"), document)

    assert result.success is False
    assert result.document == original_document
    assert result.new_score is None
    assert result.requires_rerender is False
    assert "circle" in (result.reason or "") or "robust" in (result.reason or "")


def test_command_executor_surfaces_validation_failure() -> None:
    document = _build_polyline_document(((0.0, 0.0), (2.0, 0.0), (4.0, 0.0)), path_id="locked_path", locked=True)
    executor = CommandExecutor()

    result = executor.execute(_command("propose_replace_segment_with_line", path_id="locked_path"), document)

    assert result.success is False
    assert "locked path" in (result.reason or "")
    assert result.document == document


def test_command_executor_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/command_executor.py")
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
    assert "exec(" not in source
