import ast
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from services.ai_agent import (
    AIReviewInput,
    AI_REVIEW_PROMPT,
    build_review_prompt,
    load_ai_command_schema,
    validate_ai_review_response,
)


def test_ai_review_prompt_enforces_intent_only_output() -> None:
    review_input = AIReviewInput(
        original_image="raw.png",
        overlay_image="overlay.png",
        distance_field_diff_image="diff.png",
        vector_document_json={"document_id": "doc_1"},
        fit_error=0.12,
        complexity_score=0.34,
        topology_status="closed",
        self_intersection_count=0,
        coordinate_system={"unit": "px"},
        user_locked_ids=("anchor_1",),
        available_tools=("propose_replace_segment_with_arc", "propose_batch_refinement"),
        alpha_notes="alpha fringe near upper edge",
        color_notes="color grouping may be wrong",
    )

    prompt = build_review_prompt(review_input)

    assert "only output modification intent" in prompt
    assert "Do not output precise geometry parameters" in prompt
    assert "Do not mutate the VectorDocument directly" in prompt
    assert "use the `tool` field" in prompt
    assert "original_image" in prompt
    assert "available_tools" in prompt
    assert AI_REVIEW_PROMPT in prompt


def test_ai_command_schema_accepts_valid_replace_command_and_batch() -> None:
    response = {
        "summary": "Several primitive replacements should be reviewed.",
        "issues": [
            {
                "issue_id": "issue_1",
                "category": "topology",
                "severity": "medium",
                "summary": "Closure may drift after replacement.",
                "path_id": "path_1",
                "object_id": "object_1",
                "segment_range": [0, 2],
                "topology_hint": "Check closure after replacement.",
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None
            }
        ],
        "proposed_commands": [
            {
                "tool": "propose_replace_segment_with_arc",
                "path_id": "path_1",
                "segment_range": [0, 2],
                "reason": "This region reads as a circular arc.",
                "confidence": 0.8,
                "requires_user_confirmation": True,
                "locked_anchor_ids": ["anchor_1"],
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None
            },
            {
                "tool": "propose_batch_refinement",
                "summary": "Review both primitive replacements together.",
                "commands": [
                    {
                        "tool": "propose_replace_segment_with_line",
                        "path_id": "path_2",
                        "segment_range": [1, 3],
                        "reason": "This edge should be straight.",
                        "confidence": 0.77,
                        "requires_user_confirmation": True,
                        "locked_anchor_ids": [],
                        "topology_hint": None,
                        "self_intersection_hint": None,
                        "alpha_hint": None,
                        "color_hint": None
                    }
                ],
                "confidence": 0.75,
                "requires_user_confirmation": True,
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None
            }
        ]
    }

    validate_ai_review_response(response)


def test_ai_command_schema_accepts_path_level_commands_and_mixed_batch() -> None:
    response = {
        "summary": "Primitive replacements are available at both path and segment scope.",
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_circle",
                "reason": "The whole loop reads as a clean circle.",
                "confidence": 0.87,
                "requires_user_confirmation": True,
                "candidate_id": "cand_circle_1",
                "semantic_source": "ai_review",
                "semantic_confidence": 0.92,
                "topology_hint": "Keep the loop closed after replacement.",
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None
            },
            {
                "tool": "propose_replace_path_with_ellipse",
                "path_id": "path_ellipse",
                "reason": "The silhouette is better explained by an ellipse.",
                "confidence": 0.83,
                "requires_user_confirmation": True,
                "candidate_id": "cand_ellipse_1",
                "semantic_source": "planner",
                "semantic_confidence": 0.89,
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None
            },
            {
                "tool": "propose_batch_refinement",
                "summary": "Apply a loop replacement and a local edge cleanup together.",
                "commands": [
                    {
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_circle",
                        "reason": "Keep the batch aligned with the circle interpretation.",
                        "confidence": 0.84,
                        "requires_user_confirmation": True,
                        "candidate_id": "cand_circle_batch",
                        "semantic_source": "batch_planner",
                        "semantic_confidence": 0.9,
                        "topology_hint": None,
                        "self_intersection_hint": None,
                        "alpha_hint": None,
                        "color_hint": None
                    },
                    {
                        "tool": "propose_replace_segment_with_line",
                        "path_id": "path_line",
                        "segment_range": [0, 1],
                        "reason": "This local edge should remain straight.",
                        "confidence": 0.79,
                        "requires_user_confirmation": True,
                        "locked_anchor_ids": [],
                        "topology_hint": None,
                        "self_intersection_hint": None,
                        "alpha_hint": None,
                        "color_hint": None
                    }
                ],
                "confidence": 0.8,
                "requires_user_confirmation": True,
                "candidate_id": "cand_batch_1",
                "semantic_source": "planner",
                "semantic_confidence": 0.86,
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None
            }
        ]
    }

    validate_ai_review_response(response)


def test_ai_command_schema_rejects_precise_geometry_parameters() -> None:
    response = {
        "summary": "This region looks circular.",
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_segment_with_circle",
                "path_id": "path_9",
                "segment_range": [1, 4],
                "reason": "The region appears to be a circle.",
                "confidence": 0.88,
                "requires_user_confirmation": True,
                "locked_anchor_ids": [],
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None,
                "cx": 10.0
            }
        ]
    }

    with pytest.raises(ValidationError):
        validate_ai_review_response(response)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("start_angle", 1.57079632679),
        ("end_angle", 3.14159265359),
        ("rotation", 0.78539816339),
    ),
)
def test_ai_command_schema_rejects_precise_angle_parameters(field_name: str, field_value: float) -> None:
    response = {
        "summary": "This region looks like a primitive.",
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_segment_with_arc",
                "path_id": "path_9",
                "segment_range": [1, 4],
                "reason": "The region appears to be an arc.",
                "confidence": 0.88,
                "requires_user_confirmation": True,
                "locked_anchor_ids": [],
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None,
                field_name: field_value,
            }
        ],
    }

    with pytest.raises(ValidationError):
        validate_ai_review_response(response)


def test_ai_command_schema_rejects_invalid_path_level_semantic_confidence() -> None:
    response = {
        "summary": "This loop looks circular.",
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_circle",
                "reason": "The whole loop reads as a circle.",
                "confidence": 0.88,
                "requires_user_confirmation": True,
                "semantic_confidence": 1.2
            }
        ]
    }

    with pytest.raises(ValidationError):
        validate_ai_review_response(response)


def test_ai_command_schema_rejects_missing_summary_and_unknown_tool() -> None:
    response = {
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_segment_with_bezier",
                "path_id": "path_1",
                "segment_range": [0, 2],
                "reason": "Unsupported command type for this schema.",
                "confidence": 0.5,
                "requires_user_confirmation": True
            }
        ]
    }

    with pytest.raises(ValidationError):
        validate_ai_review_response(response)


def test_ai_command_schema_file_loads_as_json_schema_document() -> None:
    schema = load_ai_command_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "proposed_commands" in schema["properties"]
    assert "command" in schema["$defs"]
    json.dumps(schema)


def test_ai_agent_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/ai_agent.py")
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
    assert "execute(" not in source
