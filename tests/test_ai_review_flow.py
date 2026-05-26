import ast
from pathlib import Path

import pytest
from jsonschema import ValidationError

from services.ai_agent import (
    AIReviewInput,
    AIReviewInputTooLarge,
    AIReviewService,
    ProviderContextLimitExceeded,
)
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


def test_ai_review_service_supports_adapter_and_legacy_responder_paths() -> None:
    captured: dict[str, object] = {}

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        captured["prompt"] = prompt
        captured["review_input"] = review_input
        return {
            "summary": "Algorithm candidate looks valid.",
            "issues": [],
            "proposed_commands": [
                {
                    "tool": "propose_replace_path_with_circle",
                    "path_id": "path_circle",
                    "reason": "The candidate reads as a circle.",
                    "confidence": 0.82,
                    "requires_user_confirmation": True,
                    "candidate_id": "cand_circle_1",
                    "semantic_source": "legacy_responder",
                    "semantic_confidence": 0.9,
                    "topology_hint": None,
                    "self_intersection_hint": None,
                    "alpha_hint": None,
                    "color_hint": None,
                }
            ],
        }

    review_input = AIReviewInput(
        original_image="raw.png",
        overlay_image="overlay.png",
        distance_field_diff_image="diff.png",
        vector_document_json={"document_id": "doc_compat"},
        candidates=(
            {"candidate_id": "cand_circle_1", "shape_type": "circle", "path_id": "path_circle", "confidence": 0.91},
        ),
        proposed_commands_from_algorithm=(
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_circle",
                "reason": "Algorithm candidate already suggests a circle.",
                "confidence": 0.8,
                "requires_user_confirmation": True,
                "candidate_id": "cand_circle_1",
            },
        ),
        preview_summary={"accepted_count": 1, "rejected_count": 0},
        fit_error=0.1,
        complexity_score=0.25,
        topology_status="closed",
        self_intersection_count=0,
        coordinate_system={"unit": "px", "view_box": [0, 0, 100, 100]},
    )

    output = AIReviewService(responder=responder).run_review(review_input)

    assert output.summary == "Algorithm candidate looks valid."
    assert output.proposed_commands[0]["candidate_id"] == "cand_circle_1"
    assert output.proposed_commands[0]["proposal_source"] == "ai_review"
    assert captured["review_input"] == review_input


def test_ai_review_input_payload_includes_rejection_feedback_context() -> None:
    captured: dict[str, object] = {}

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        captured["prompt"] = prompt
        captured["review_input"] = review_input
        return {
            "summary": "Use rejection feedback to avoid repeating invalid commands.",
            "issues": [],
            "proposed_commands": [],
        }

    review_input = AIReviewInput(
        original_image=None,
        overlay_image=None,
        distance_field_diff_image=None,
        vector_document_json={"document_id": "doc_feedback"},
        fit_error=0.2,
        complexity_score=0.2,
        topology_status="open",
        self_intersection_count=1,
        coordinate_system={"unit": "px"},
        preview_summary={"iteration": 2},
        policy_feedback=(
            {
                "reason_code": "self_intersection_increase",
                "message": "Previous proposal introduced self intersections.",
                "metrics_delta": {"self_intersection_delta": 1},
                "policy_hint": "avoid this replacement pattern",
                "retry_allowed": True,
                "retry_constraints": {"max_retry_per_target": 2},
                "forbidden_repeated_commands": ["propose_replace_path_with_circle:path_1"],
            },
        ),
        rejection_memory=(
            {
                "target": "path_1",
                "tool": "propose_replace_path_with_circle",
                "reason_code": "self_intersection_increase",
                "retry_count": 1,
                "last_metrics_delta": {"self_intersection_delta": 1},
            },
        ),
        forbidden_repeated_commands=("propose_replace_path_with_circle:path_1",),
        retry_budget={"max_iterations": 3, "max_retry_per_target": 2},
    )

    AIReviewService(responder=responder).run_review(review_input)

    assert captured["review_input"] == review_input
    assert "policy_feedback" in str(captured["prompt"])
    assert "rejection_memory" in str(captured["prompt"])
    assert "forbidden_repeated_commands" in str(captured["prompt"])


def test_ai_review_input_supports_local_visual_context_budget() -> None:
    captured: dict[str, object] = {}

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        captured["review_input"] = review_input
        captured["prompt"] = prompt
        return {"summary": "local visual review", "issues": [], "proposed_commands": []}

    review_input = AIReviewInput(
        original_image="original.png",
        overlay_image="overlay.png",
        distance_field_diff_image="diff.png",
        vector_document_json={"document_id": "doc_local", "path_count": 1},
        document_summary={"document_id": "doc_local", "path_count": 1},
        review_jobs=(
            {
                "job_id": "job_1",
                "path_id": "path_1",
                "window_id": "path_1:window_1",
                "crop_bbox": [0, 0, 64, 64],
                "image_count": 3,
                "truncated": False,
            },
        ),
        prompt_budget={"max_prompt_chars": 4000, "max_crop_size_px": 512},
        ai_input_mode="local_visual_context",
        fit_error=0.2,
        complexity_score=0.2,
        topology_status="open",
        self_intersection_count=1,
        coordinate_system={"unit": "px"},
    )

    output = AIReviewService(responder=responder).run_review(review_input)

    assert output.summary == "local visual review"
    assert captured["review_input"] == review_input
    assert "review_jobs" in str(captured["prompt"])
    assert "local visual context" in str(captured["prompt"]).lower()


def test_ai_review_service_rejects_oversized_prompt_before_adapter_call() -> None:
    called = {"value": False}

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        called["value"] = True
        return {"summary": "should not run", "issues": [], "proposed_commands": []}

    review_input = AIReviewInput(
        original_image=None,
        overlay_image=None,
        distance_field_diff_image=None,
        vector_document_json={"document_id": "doc_large", "payload": "x" * 4000},
        prompt_budget={"max_prompt_chars": 256},
        fit_error=0.2,
        complexity_score=0.2,
        topology_status="open",
        self_intersection_count=1,
        coordinate_system={"unit": "px"},
    )

    with pytest.raises(AIReviewInputTooLarge):
        AIReviewService(responder=responder).run_review(review_input)

    assert called["value"] is False


def test_ai_review_service_wraps_provider_context_overflow_errors() -> None:
    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        raise RuntimeError("context length exceeded for this model")

    review_input = AIReviewInput(
        original_image=None,
        overlay_image=None,
        distance_field_diff_image=None,
        vector_document_json={"document_id": "doc_provider"},
        fit_error=0.2,
        complexity_score=0.2,
        topology_status="open",
        self_intersection_count=1,
        coordinate_system={"unit": "px"},
    )

    with pytest.raises(ProviderContextLimitExceeded):
        AIReviewService(responder=responder).run_review(review_input)


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


@pytest.mark.parametrize(
    "bad_response",
    (
        {"summary": "bad", "issues": 123, "proposed_commands": []},
        {"summary": "bad", "issues": [], "proposed_commands": 123},
        {
            "summary": "bad",
            "issues": [],
            "proposed_commands": [
                {
                    "tool": "propose_batch_refinement",
                    "summary": "nested bad payload",
                    "commands": 123,
                    "confidence": 0.5,
                    "requires_user_confirmation": True,
                }
            ],
        },
        {"summary": "bad", "issues": [], "proposed_commands": [123]},
    ),
)
def test_ai_review_flow_rejects_structurally_invalid_response_with_value_error(bad_response: dict[str, object]) -> None:
    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        return bad_response

    service = AIReviewService(responder=responder)
    review_input = AIReviewInput(
        original_image=None,
        overlay_image=None,
        distance_field_diff_image=None,
        vector_document_json={"document_id": "doc_bad"},
        fit_error=0.2,
        complexity_score=0.2,
        topology_status="open",
        self_intersection_count=1,
        coordinate_system={"unit": "px"},
    )

    with pytest.raises(ValueError):
        service.run_review(review_input)


def test_ai_review_flow_rejects_excessive_batch_nesting_with_value_error() -> None:
    nested_command: dict[str, object] = {
        "command_type": "propose_replace_segment_with_line",
        "path_id": "path_1",
        "segment_range": [0, 1],
        "reason": "base command",
        "confidence": 0.7,
        "requires_user_confirmation": True,
    }
    for depth in range(11):
        nested_command = {
            "tool": "propose_batch_refinement",
            "summary": f"batch depth {depth}",
            "commands": [nested_command],
            "confidence": 0.6,
            "requires_user_confirmation": True,
        }

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        return {"summary": "too deep", "issues": [], "proposed_commands": [nested_command]}

    service = AIReviewService(responder=responder)
    review_input = AIReviewInput(
        original_image=None,
        overlay_image=None,
        distance_field_diff_image=None,
        vector_document_json={"document_id": "doc_deep"},
        fit_error=0.2,
        complexity_score=0.2,
        topology_status="open",
        self_intersection_count=1,
        coordinate_system={"unit": "px"},
    )

    with pytest.raises(ValueError, match="max depth"):
        service.run_review(review_input)


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
        if source_path.name != "main_window.py":
            assert ".execute(" not in source


def test_ai_review_service_rejects_simultaneous_adapter_and_responder_configuration() -> None:
    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        return {"summary": "unused", "issues": [], "proposed_commands": []}

    with pytest.raises(ValueError):
        AIReviewService(adapter=object(), responder=responder)  # type: ignore[arg-type]
