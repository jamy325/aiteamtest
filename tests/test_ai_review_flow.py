import ast
from pathlib import Path

import pytest
from jsonschema import ValidationError

from services.ai_agent import AIReviewInput, AIReviewService
from ui.canvas_widget import CanvasWidget
from ui.main_window import MainWindow


def test_ai_review_flow_displays_summary_issues_and_proposed_commands_without_execution() -> None:
    captured: dict[str, object] = {}

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        captured["prompt"] = prompt
        captured["review_input"] = review_input
        return {
            "summary": "One region looks circular and another has a topology warning.",
            "issues": [
                {
                    "issue_id": "issue_1",
                    "category": "topology",
                    "severity": "medium",
                    "summary": "Path closure may drift after refinement.",
                    "path_id": "path_1",
                    "object_id": "object_1",
                    "segment_range": [2, 5],
                    "topology_hint": "Preserve closure after refinement.",
                    "self_intersection_hint": None,
                    "alpha_hint": None,
                    "color_hint": None,
                }
            ],
            "proposed_commands": [
                {
                    "command_type": "propose_replace_segment_with_circle",
                    "path_id": "path_1",
                    "segment_range": [2, 5],
                    "reason": "This region visually reads as a full circular feature.",
                    "confidence": 0.83,
                    "requires_user_confirmation": True,
                    "locked_anchor_ids": ["anchor_2"],
                    "topology_hint": "Check closure after replacement.",
                    "self_intersection_hint": None,
                    "alpha_hint": "Ignore transparent fringe.",
                    "color_hint": "Preserve stroke grouping.",
                }
            ],
        }

    canvas_widget = CanvasWidget(locked_ids=("anchor_2", "anchor_5"))
    window = MainWindow(
        ai_review_service=AIReviewService(responder=responder),
        canvas_widget=canvas_widget,
    )

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
        available_tools=("propose_replace_segment_with_circle", "propose_batch_refinement"),
        alpha_notes="minor fringe",
        color_notes="verify highlight grouping",
    )

    assert display_state.summary == "One region looks circular and another has a topology warning."
    assert len(display_state.issues) == 1
    assert len(display_state.proposed_commands) == 1
    assert window.executed_commands == ()
    assert window.last_review_input is not None
    assert window.last_review_input.user_locked_ids == ("anchor_2", "anchor_5")
    assert window.last_review_input.available_tools == (
        "propose_replace_segment_with_circle",
        "propose_batch_refinement",
    )
    assert window.last_review_output is not None
    assert "Do not output precise geometry parameters" in window.last_review_output.prompt
    assert captured["review_input"] == window.last_review_input


def test_ai_review_flow_rejects_invalid_schema_response() -> None:
    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        return {
            "summary": "Invalid because it leaks geometry parameters.",
            "issues": [],
            "proposed_commands": [
                {
                    "command_type": "propose_replace_segment_with_arc",
                    "path_id": "path_1",
                    "segment_range": [1, 3],
                    "reason": "The region reads as an arc.",
                    "confidence": 0.9,
                    "requires_user_confirmation": True,
                    "cx": 12.0,
                }
            ],
        }

    window = MainWindow(ai_review_service=AIReviewService(responder=responder))

    with pytest.raises(ValidationError):
        window.trigger_ai_review(
            original_image=None,
            overlay_image=None,
            distance_field_diff_image=None,
            vector_document_json={"document_id": "doc_2"},
            fit_error=0.2,
            complexity_score=0.2,
            topology_status="open",
            self_intersection_count=1,
            coordinate_system={"unit": "px"},
        )

    assert window.executed_commands == ()
    assert window.review_display_state.summary == ""


def test_canvas_widget_tracks_locked_ids_for_ai_review_input() -> None:
    canvas_widget = CanvasWidget()

    canvas_widget.lock_id("anchor_1")
    canvas_widget.lock_id("anchor_3")
    canvas_widget.unlock_id("anchor_1")

    assert canvas_widget.locked_ids == ("anchor_3",)


def test_ai_review_flow_has_no_forbidden_dependencies() -> None:
    source_paths = (
        Path("services/ai_agent.py"),
        Path("ui/main_window.py"),
        Path("ui/canvas_widget.py"),
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
