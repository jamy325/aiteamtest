import ast
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from services.ai_adapters import FileResponseVisionAdapter, MockVisionAdapter
from services.ai_agent import AIReviewInput, AIReviewService


def _review_input() -> AIReviewInput:
    return AIReviewInput(
        original_image="original.png",
        overlay_image="overlay.png",
        distance_field_diff_image="diff.png",
        vector_document_json={"document_id": "doc_adapter"},
        candidates=(
            {
                "candidate_id": "candidate_circle_1",
                "shape_type": "circle",
                "path_id": "path_1",
                "confidence": 0.91,
            },
        ),
        proposed_commands_from_algorithm=(
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "Algorithm candidate already suggests a circle replacement.",
                "confidence": 0.86,
                "requires_user_confirmation": True,
                "candidate_id": "candidate_circle_1",
            },
        ),
        preview_summary={"accepted_count": 1, "rejected_count": 0},
        fit_error=0.08,
        complexity_score=0.22,
        topology_status="closed",
        self_intersection_count=0,
        coordinate_system={"unit": "px"},
        available_tools=("propose_replace_path_with_circle", "propose_batch_refinement"),
    )


def _valid_response() -> dict[str, object]:
    return {
        "summary": "The loop candidate looks valid after visual review.",
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "The loop visually reads as a circle and matches the algorithm candidate.",
                "confidence": 0.9,
                "requires_user_confirmation": True,
                "candidate_id": "candidate_circle_1",
                "semantic_source": "mock_adapter",
                "semantic_confidence": 0.94,
                "topology_hint": "Keep the loop closed after replacement.",
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None,
            }
        ],
    }


def test_mock_vision_adapter_returns_valid_review_output() -> None:
    review_service = AIReviewService(adapter=MockVisionAdapter(response=_valid_response()))

    review_output = review_service.run_review(_review_input())

    assert review_output.summary == "The loop candidate looks valid after visual review."
    assert review_output.proposed_commands[0]["tool"] == "propose_replace_path_with_circle"
    assert "candidates" in review_output.prompt
    assert "proposed_commands_from_algorithm" in review_output.prompt
    assert "preview_summary" in review_output.prompt
    assert "Review algorithm candidates" in review_output.prompt


def test_file_response_vision_adapter_reads_response_from_json_file(tmp_path: Path) -> None:
    response_path = tmp_path / "ai_response.json"
    response_path.write_text(json.dumps(_valid_response(), indent=2), encoding="utf-8")

    review_service = AIReviewService(adapter=FileResponseVisionAdapter(response_path=response_path))
    review_output = review_service.run_review(_review_input())

    assert review_output.raw_response["summary"] == "The loop candidate looks valid after visual review."
    assert review_output.proposed_commands[0]["candidate_id"] == "candidate_circle_1"


def test_file_response_vision_adapter_rejects_invalid_schema_response(tmp_path: Path) -> None:
    response_path = tmp_path / "invalid_ai_response.json"
    response_path.write_text(
        json.dumps(
            {
                "summary": "This response leaks exact geometry.",
                "issues": [],
                "proposed_commands": [
                    {
                        "tool": "propose_replace_segment_with_arc",
                        "path_id": "path_1",
                        "segment_range": [1, 3],
                        "reason": "This region reads as an arc.",
                        "confidence": 0.88,
                        "requires_user_confirmation": True,
                        "start_angle": 1.57,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    review_service = AIReviewService(adapter=FileResponseVisionAdapter(response_path=response_path))

    with pytest.raises(ValidationError):
        review_service.run_review(_review_input())


def test_ai_adapters_have_no_forbidden_dependencies() -> None:
    source_paths = (
        Path("services/ai_adapters/base.py"),
        Path("services/ai_adapters/mock.py"),
        Path("services/ai_adapters/file_response.py"),
    )
    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "google", "requests"}

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
