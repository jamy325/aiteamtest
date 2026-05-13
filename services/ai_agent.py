from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "ai_commands.schema.json"

AI_REVIEW_PROMPT = """You are the AI review planner for an AI tracing vector reconstruction system.

Your role is limited to visual review, semantic judgment, and modification-intent planning.
You must only output modification intent through the approved JSON schema.

Hard rules:
- Do not output precise geometry parameters.
- Do not output exact centers, radii, control points, line equations, start angles, end angles, or tangent vectors.
- Do not execute tools or proposed commands.
- Do not mutate the VectorDocument directly.
- Proposed commands must stay at the intent-planning level and must require deterministic algorithm refinement later.

Required output shape:
- summary
- issues
- proposed_commands

Inputs available to you:
- original_image
- overlay_image
- distance_field_diff_image
- vector_document_json
- fit_error
- complexity_score
- topology_status
- self_intersection_count
- coordinate_system
- user_locked_ids
- available_tools
- alpha_notes
- color_notes

When describing issues or commands:
- include topology guidance when path closure, gap, or continuity is suspicious
- include self_intersection guidance when paths cross or overlap incorrectly
- include alpha guidance when transparency or matte pollution affects interpretation
- include color guidance when style or color grouping appears wrong

Return JSON only and ensure it validates against the proposed_commands schema.
"""


@dataclass(frozen=True, slots=True)
class AIReviewInput:
    original_image: str | None
    overlay_image: str | None
    distance_field_diff_image: str | None
    vector_document_json: dict[str, Any]
    fit_error: float
    complexity_score: float
    topology_status: str
    self_intersection_count: int
    coordinate_system: dict[str, Any]
    user_locked_ids: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    alpha_notes: str | None = None
    color_notes: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_review_prompt(review_input: AIReviewInput) -> str:
    payload = json.dumps(review_input.to_payload(), ensure_ascii=True, sort_keys=True, indent=2)
    return f"{AI_REVIEW_PROMPT}\n\nReview input:\n{payload}"


def load_ai_command_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_ai_review_response(response: dict[str, Any]) -> None:
    validator = Draft202012Validator(load_ai_command_schema())
    validator.validate(response)


__all__ = [
    "AIReviewInput",
    "AI_REVIEW_PROMPT",
    "SCHEMA_PATH",
    "build_review_prompt",
    "load_ai_command_schema",
    "validate_ai_review_response",
]
