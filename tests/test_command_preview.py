from __future__ import annotations

import ast
import math
from pathlib import Path

from core.document import add_anchor, add_constraint, add_path, add_segment, create_document
from core.types import Anchor, Constraint, CoordinateSystem, Path as VectorPath, Segment
from services.command_preview import CommandPreviewService


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


def test_command_preview_returns_success_summary_without_mutating_document() -> None:
    document = _build_polyline_document(((0.0, 0.0), (2.0, 0.15), (4.0, -0.1), (6.0, 0.0)))
    original_document = document
    preview = CommandPreviewService().preview(_command("propose_replace_segment_with_line"), document)

    assert preview.success is True
    assert preview.command_id == "propose_replace_segment_with_line"
    assert preview.old_score is not None
    assert preview.predicted_new_score is not None
    assert preview.score_delta is not None
    assert preview.affected_paths == ("path_1",)
    assert preview.affected_segments == ("path_1_seg_1",)
    assert preview.topology_status_before["path_1"] == "open"
    assert preview.topology_status_after["path_1"] == "open"
    assert preview.self_intersection_count_before["path_1"] == 0
    assert preview.self_intersection_count_after["path_1"] == 0
    assert preview.segment_type_summary["before"]["polyline"] == 1
    assert preview.segment_type_summary["after"]["line"] == 1
    assert preview.segment_type_summary["delta"]["polyline"] == -1
    assert preview.segment_type_summary["delta"]["line"] == 1
    assert preview.export_impact_summary.before["json_char_count"] > 0
    assert document == original_document


def test_command_preview_returns_failure_reason_without_mutating_document() -> None:
    document = _build_polyline_document(((0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0)))
    original_document = document
    preview = CommandPreviewService().preview(_command("propose_replace_segment_with_circle"), document)

    assert preview.success is False
    assert preview.reason is not None
    assert preview.predicted_new_score is None
    assert preview.score_delta is None
    assert preview.segment_type_summary["before"] == preview.segment_type_summary["after"]
    assert preview.export_impact_summary.before == preview.export_impact_summary.after
    assert document == original_document


def test_command_preview_batch_continues_after_partial_failure() -> None:
    document = create_document(
        document_id="doc_batch_preview",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_1", segments=("path_1_seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="path_1_seg_1",
            path_id="path_1",
            type="polyline",
            params={"points": [[0.0, 0.0], [2.0, 0.1], [4.0, -0.1], [6.0, 0.0]]},
        ),
    )
    document = add_path(document, VectorPath(path_id="locked_path", locked=True, segments=("locked_seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="locked_seg_1",
            path_id="locked_path",
            type="polyline",
            params={"points": [[10.0, 0.0], [12.0, 0.0], [14.0, 0.0]]},
        ),
    )
    document = add_path(document, VectorPath(path_id="path_2", segments=("path_2_seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="path_2_seg_1",
            path_id="path_2",
            type="polyline",
            params={"points": [[20.0, 0.0], [22.0, 0.1], [24.0, -0.1], [26.0, 0.0]]},
        ),
    )
    original_document = document

    result = CommandPreviewService().preview_batch(
        [
            _command("propose_replace_segment_with_line", path_id="path_1", command_id="cmd_ok_1"),
            _command("propose_replace_segment_with_line", path_id="locked_path", command_id="cmd_fail"),
            _command("propose_replace_segment_with_line", path_id="path_2", command_id="cmd_ok_2"),
        ],
        document,
    )

    assert result.success_count == 2
    assert result.failure_count == 1
    assert tuple(item.success for item in result.previews) == (True, False, True)
    assert "locked path" in (result.previews[1].reason or "")
    assert result.previews[2].segment_type_summary["after"]["line"] >= 2
    assert document == original_document


def test_command_preview_batch_handles_dirty_command_without_crashing() -> None:
    document = _build_polyline_document(
        _arc_points(cx=20.0, cy=10.0, radius=6.0, start_angle=0.25, end_angle=1.4, count=10)
    )
    original_document = document
    result = CommandPreviewService().preview_batch(
        [
            None,
            _command("propose_replace_segment_with_arc", command_id="arc_after_none"),
        ],
        document,
        continue_on_failure=True,
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    assert tuple(item.success for item in result.previews) == (False, True)
    assert "command must be a dictionary" in (result.previews[0].reason or "")
    assert document == original_document


def test_command_preview_reports_removed_constraint_ids_and_type_delta() -> None:
    full_points = _circle_points(cx=30.0, cy=25.0, radius=8.0, count=12)
    first_half = full_points[:7]
    second_half = full_points[6:]
    document = create_document(
        document_id="doc_circle_preview",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="circle_path",
            closed=True,
            segments=("circle_seg_1", "circle_seg_2"),
        ),
    )
    document = add_segment(
        document,
        Segment(
            "circle_seg_1",
            "circle_path",
            "polyline",
            {"points": [[x, y] for x, y in first_half]},
            anchors=("a0", "a1"),
        ),
    )
    document = add_segment(
        document,
        Segment(
            "circle_seg_2",
            "circle_path",
            "polyline",
            {"points": [[x, y] for x, y in second_half]},
            anchors=("a1", "a0"),
        ),
    )
    document = add_anchor(document, Anchor("a0", "circle_path", first_half[0]))
    document = add_anchor(document, Anchor("a1", "circle_path", first_half[-1]))
    document = add_constraint(document, Constraint("c_drop", "coincident", targets=("circle_seg_2", "a1")))
    original_document = document

    preview = CommandPreviewService().preview(
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

    assert preview.success is True
    assert preview.constraint_change_summary.before["coincident"] == 1
    assert preview.constraint_change_summary.after.get("coincident", 0) == 0
    assert preview.constraint_change_summary.delta["coincident"] == -1
    assert preview.constraint_change_summary.removed_constraint_ids == ("c_drop",)
    assert preview.constraint_change_summary.added_constraint_ids == ()
    assert preview.constraint_change_summary.changed_constraint_ids == ()
    assert document == original_document


def test_command_preview_failure_keeps_constraint_summary_unchanged() -> None:
    document = create_document(
        document_id="doc_locked_constraint_preview",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(path_id="locked_path", closed=False, locked=True, segments=("locked_seg_1",)),
    )
    document = add_segment(
        document,
        Segment(
            "locked_seg_1",
            "locked_path",
            "polyline",
            {"points": [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]},
        ),
    )
    document = add_constraint(
        document,
        Constraint("c_keep", "coincident", targets=("locked_seg_1",), locked=True),
    )
    original_document = document

    preview = CommandPreviewService().preview(
        _command("propose_replace_segment_with_line", path_id="locked_path", command_id="locked_fail"),
        document,
    )

    assert preview.success is False
    assert "locked path" in (preview.reason or "")
    assert preview.constraint_change_summary.before == {"coincident": 1}
    assert preview.constraint_change_summary.after == {"coincident": 1}
    assert preview.constraint_change_summary.delta == {"coincident": 0}
    assert preview.constraint_change_summary.added_constraint_ids == ()
    assert preview.constraint_change_summary.removed_constraint_ids == ()
    assert preview.constraint_change_summary.changed_constraint_ids == ()
    assert document == original_document


def test_command_preview_has_no_forbidden_dependencies() -> None:
    source = Path("services/command_preview.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic"}
    assert imports.isdisjoint(forbidden_imports)
