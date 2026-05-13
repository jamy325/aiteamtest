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
    assert "original_image" in prompt
    assert "overlay_image" in prompt
    assert "distance_field_diff_image" in prompt
    assert "vector_document_json" in prompt
    assert "self_intersection_count" in prompt
    assert "available_tools" in prompt
    assert AI_REVIEW_PROMPT in prompt


def test_ai_command_schema_accepts_line_command_and_issue_hints() -> None:
    response = {
        "summary": "Line segment looks bowed and should be simplified.",
        "issues": [
            {
                "issue_id": "issue_topology_1",
                "category": "topology",
                "severity": "medium",
                "summary": "Closing continuity looks fragile near the replacement range.",
                "path_id": "path_1",
                "object_id": "object_1",
                "segment_range": [3, 6],
                "topology_hint": "Preserve closure after replacement.",
                "self_intersection_hint": None,
                "alpha_hint": "Ignore matte fringe near the source edge.",
                "color_hint": "Keep original stroke grouping."
            }
        ],
        "proposed_commands": [
            {
                "command_type": "propose_replace_segment_with_line",
                "path_id": "path_1",
                "segment_range": [3, 6],
                "reason": "This region reads as a clean straight edge.",
                "confidence": 0.86,
                "requires_user_confirmation": True,
                "locked_anchor_ids": ["anchor_10"],
                "topology_hint": "Re-check closure at both endpoints.",
                "self_intersection_hint": None,
                "alpha_hint": "Do not chase transparent fringe pixels.",
                "color_hint": "Keep the current stroke family."
            }
        ]
    }

    validate_ai_review_response(response)


def test_ai_command_schema_accepts_arc_circle_ellipse_and_batch_commands() -> None:
    response = {
        "summary": "Multiple regions appear to need primitive replacement.",
        "issues": [
            {
                "issue_id": "issue_self_intersection_1",
                "category": "self_intersection",
                "severity": "high",
                "summary": "One overlap likely comes from the current arc region.",
                "path_id": "path_2",
                "object_id": None,
                "segment_range": [5, 8],
                "topology_hint": None,
                "self_intersection_hint": "Prefer the simpler arc replacement first.",
                "alpha_hint": None,
                "color_hint": None
            },
            {
                "issue_id": "issue_color_1",
                "category": "color",
                "severity": "low",
                "summary": "The fill edge may be grouped with the wrong style family.",
                "path_id": None,
                "object_id": "object_7",
                "segment_range": None,
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": "Consider separating highlight from body contour."
            }
        ],
        "proposed_commands": [
            {
                "command_type": "propose_batch_refinement",
                "summary": "Replace the noisy primitive regions as a coordinated batch.",
                "commands": [
                    {
                        "command_type": "propose_replace_segment_with_arc",
                        "path_id": "path_2",
                        "segment_range": [5, 8],
                        "reason": "The curvature appears circular but partial.",
                        "confidence": 0.74,
                        "requires_user_confirmation": True,
                        "locked_anchor_ids": [],
                        "topology_hint": "Keep endpoint continuity stable.",
                        "self_intersection_hint": "Check overlap after replacement.",
                        "alpha_hint": None,
                        "color_hint": None
                    },
                    {
                        "command_type": "propose_replace_segment_with_circle",
                        "path_id": "path_3",
                        "segment_range": [0, 11],
                        "reason": "The region appears to be a full circular feature.",
                        "confidence": 0.9,
                        "requires_user_confirmation": True,
                        "locked_anchor_ids": [],
                        "topology_hint": None,
                        "self_intersection_hint": None,
                        "alpha_hint": None,
                        "color_hint": None
                    },
                    {
                        "command_type": "propose_replace_segment_with_ellipse",
                        "path_id": "path_4",
                        "segment_range": [2, 15],
                        "reason": "The outline looks elongated rather than circular.",
                        "confidence": 0.68,
                        "requires_user_confirmation": True,
                        "locked_anchor_ids": [],
                        "topology_hint": None,
                        "self_intersection_hint": None,
                        "alpha_hint": "Do not overfit semi-transparent edge bloom.",
                        "color_hint": "Preserve distinct interior highlight."
                    }
                ],
                "confidence": 0.72,
                "requires_user_confirmation": True,
                "topology_hint": "Review topology after the batch.",
                "self_intersection_hint": "Re-check all crossings after replacement.",
                "alpha_hint": "Keep alpha-related judgments conservative.",
                "color_hint": "Maintain style grouping consistency."
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
                "command_type": "propose_replace_segment_with_circle",
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


def test_ai_command_schema_rejects_missing_summary_and_unknown_command_type() -> None:
    response = {
        "issues": [],
        "proposed_commands": [
            {
                "command_type": "propose_replace_segment_with_bezier",
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
