from __future__ import annotations

import ast
from pathlib import Path

from services.ai_agent import AIReviewInput, AIReviewService
from ui.canvas_widget import CanvasWidget
from ui.main_window import MainWindow
from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment


def _document():
    document = create_document(
        document_id="doc_overlay",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(
            unit="px",
            y_axis="down",
            precision=4,
            view_box=(0.0, 0.0, 100.0, 100.0),
        ),
    )
    path = VectorPath(path_id="path_1", segments=("seg_1", "seg_2"))
    document = add_path(document, path)
    document = add_segment(document, Segment("seg_1", "path_1", "line", {"start": [10.0, 10.0], "end": [40.0, 10.0]}))
    document = add_segment(document, Segment("seg_2", "path_1", "line", {"start": [40.0, 10.0], "end": [70.0, 20.0]}))
    return document


def test_canvas_widget_builds_issue_and_command_suggestion_overlays() -> None:
    canvas = CanvasWidget()
    canvas.set_document(_document())

    overlays = canvas.set_review_display(
        summary="overlay",
        issues=(
            {
                "issue_id": "issue_1",
                "category": "topology",
                "severity": "medium",
                "summary": "A topology issue",
                "path_id": "path_1",
                "segment_range": [0, 1],
            },
        ),
        proposed_commands=(
            {
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_1",
                "segment_range": [1, 1],
                "reason": "replace segment",
                "confidence": 0.85,
                "requires_user_confirmation": True,
                "bbox": [35.0, 5.0, 40.0, 20.0],
            },
        ),
    )

    assert len(overlays) == 2
    issue_overlay = overlays[0]
    command_overlay = overlays[1]
    assert issue_overlay.source_type == "issue"
    assert issue_overlay.path_id == "path_1"
    assert issue_overlay.segment_ids == ("seg_1", "seg_2")
    assert issue_overlay.unknown_target_ids == ()
    assert command_overlay.source_type == "command"
    assert command_overlay.tool == "propose_replace_segment_with_line"
    assert command_overlay.confidence == 0.85
    assert command_overlay.segment_ids == ("seg_2",)
    assert command_overlay.bbox == (35.0, 5.0, 40.0, 20.0)


def test_canvas_widget_clears_suggestion_overlays() -> None:
    canvas = CanvasWidget()
    canvas.set_document(_document())
    canvas.set_review_display(
        summary="overlay",
        issues=(
            {
                "issue_id": "issue_1",
                "category": "topology",
                "severity": "medium",
                "summary": "A topology issue",
            },
        ),
        proposed_commands=(),
    )

    overlays = canvas.set_review_display(summary="", issues=(), proposed_commands=())

    assert overlays == ()
    assert canvas.suggestion_overlays == ()
    assert canvas.review_issues == ()
    assert canvas.review_commands == ()


def test_canvas_widget_marks_locked_targets_in_suggestion_overlay() -> None:
    canvas = CanvasWidget(locked_ids=("path_1", "seg_2", "anchor_7"))
    canvas.set_document(_document())

    overlays = canvas.set_review_display(
        summary="overlay",
        issues=(),
        proposed_commands=(
            {
                "tool": "propose_replace_segment_with_arc",
                "path_id": "path_1",
                "segment_range": [1, 1],
                "reason": "arc candidate",
                "confidence": 0.77,
                "requires_user_confirmation": True,
                "locked_anchor_ids": ["anchor_7", "anchor_999"],
            },
        ),
    )

    overlay = overlays[0]
    assert overlay.locked_target_ids == ("path_1", "seg_2", "anchor_7")


def test_canvas_widget_tolerates_unknown_targets_without_crashing() -> None:
    canvas = CanvasWidget()
    canvas.set_document(_document())

    overlays = canvas.set_review_display(
        summary="overlay",
        issues=(
            {
                "issue_id": "issue_missing",
                "category": "geometry",
                "severity": "low",
                "summary": "Unknown target should not crash",
                "path_id": "missing_path",
                "segment_id": "missing_seg",
            },
        ),
        proposed_commands=(),
    )

    overlay = overlays[0]
    assert overlay.unknown_target_ids == ("missing_path", "missing_seg")
    assert [target.exists for target in overlay.targets if target.target_id in {"missing_path", "missing_seg"}] == [False, False]


def test_main_window_exposes_suggestion_overlays_without_command_execution() -> None:
    canvas = CanvasWidget(locked_ids=("seg_2",))
    canvas.set_document(_document())

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        return {
            "summary": "One candidate is visible.",
            "issues": [
                {
                    "issue_id": "issue_1",
                    "category": "geometry",
                    "severity": "low",
                    "summary": "Geometry looks slightly off.",
                    "path_id": "path_1",
                    "segment_range": [1, 1],
                }
            ],
            "proposed_commands": [
                {
                    "tool": "propose_replace_segment_with_line",
                    "path_id": "path_1",
                    "segment_range": [1, 1],
                    "reason": "Line candidate.",
                    "confidence": 0.92,
                    "requires_user_confirmation": True,
                }
            ],
        }

    window = MainWindow(ai_review_service=AIReviewService(responder=responder), canvas_widget=canvas)
    display_state = window.trigger_ai_review(
        original_image="raw.png",
        overlay_image="overlay.png",
        distance_field_diff_image="diff.png",
        vector_document_json={"document_id": "doc_1"},
        fit_error=0.12,
        complexity_score=0.45,
        topology_status="closed",
        self_intersection_count=0,
        coordinate_system={"unit": "px", "view_box": [0, 0, 100, 100]},
        available_tools=("propose_replace_segment_with_line",),
    )

    assert len(display_state.suggestion_overlays) == 2
    assert display_state.suggestion_overlays[1].locked_target_ids == ("seg_2",)
    assert window.executed_commands == ()


def test_ai_suggestion_visualization_has_no_forbidden_dependencies() -> None:
    source_paths = (
        Path("ui/canvas_widget.py"),
        Path("ui/main_window.py"),
        Path("services/ai_agent.py"),
    )
    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic"}

    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(name.name.split(".")[0] for name in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

        assert imports.isdisjoint(forbidden_imports)
        assert ".execute(" not in source
