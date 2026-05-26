from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator
from services.ai_adapters import ResponderVisionAdapter, VisionReviewAdapter

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
- Review algorithm candidates and existing intent commands; do not replace them with precise fitted geometry.

Required output shape:
- summary
- issues
- proposed_commands

Inputs available to you:
- original_image
- overlay_image
- distance_field_diff_image
- vector_document_json
- candidates
- proposed_commands_from_algorithm
- preview_summary
- fit_error
- complexity_score
- topology_status
- self_intersection_count
- coordinate_system
- user_locked_ids
- available_tools
- alpha_notes
- color_notes
- policy_feedback
- rejection_memory
- forbidden_repeated_commands
- retry_budget

When describing issues or commands:
- inspect algorithm candidates first and explain why a candidate should or should not be trusted
- keep any replacement proposal at semantic intent level so later deterministic refinement can solve the exact geometry
- include topology guidance when path closure, gap, or continuity is suspicious
- include self_intersection guidance when paths cross or overlap incorrectly
- include alpha guidance when transparency or matte pollution affects interpretation
- include color guidance when style or color grouping appears wrong
- use the `tool` field for proposed commands, not `command_type`

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
    candidates: tuple[dict[str, Any], ...] = ()
    proposed_commands_from_algorithm: tuple[dict[str, Any], ...] = ()
    preview_summary: dict[str, Any] | None = None
    policy_feedback: tuple[dict[str, Any], ...] = ()
    rejection_memory: tuple[dict[str, Any], ...] = ()
    forbidden_repeated_commands: tuple[str, ...] = ()
    retry_budget: dict[str, Any] | None = None
    user_locked_ids: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    alpha_notes: str | None = None
    color_notes: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AIReviewOutput:
    summary: str
    issues: tuple[dict[str, Any], ...]
    proposed_commands: tuple[dict[str, Any], ...]
    prompt: str
    review_input: AIReviewInput
    raw_response: dict[str, Any]


def build_review_prompt(review_input: AIReviewInput) -> str:
    payload = json.dumps(review_input.to_payload(), ensure_ascii=True, sort_keys=True, indent=2)
    return f"{AI_REVIEW_PROMPT}\n\nReview input:\n{payload}"


def load_ai_command_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def normalize_ai_review_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("AI review response must be a dict")

    normalized = dict(response)
    normalized["issues"] = [_normalize_issue(issue) for issue in _coerce_sequence(response.get("issues", ()), field_name="issues")]
    normalized["proposed_commands"] = [
        _normalize_command(command)
        for command in _coerce_sequence(response.get("proposed_commands", ()), field_name="proposed_commands")
    ]
    return normalized


def validate_ai_review_response(response: dict[str, Any]) -> None:
    validator = Draft202012Validator(load_ai_command_schema())
    validator.validate(normalize_ai_review_response(response))


def _normalize_issue(issue: Any) -> dict[str, Any]:
    if not isinstance(issue, dict):
        raise ValueError("each issue must be a dict")
    return dict(issue)


def _normalize_command(command: Any, *, depth: int = 0, max_depth: int = 10) -> dict[str, Any]:
    if not isinstance(command, dict):
        raise ValueError("each proposed command must be a dict")
    if depth > max_depth:
        raise ValueError(f"AI review command nesting exceeds max depth {max_depth}")

    normalized = dict(command)
    if "tool" not in normalized and "command_type" in normalized:
        normalized["tool"] = normalized.pop("command_type")
    if normalized.get("tool") == "propose_batch_refinement":
        normalized["commands"] = [
            _normalize_command(item, depth=depth + 1, max_depth=max_depth)
            for item in _coerce_sequence(normalized.get("commands", ()), field_name="propose_batch_refinement.commands")
        ]
    return normalized


def _coerce_sequence(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    return list(value)


class AIReviewService:
    def __init__(
        self,
        adapter: VisionReviewAdapter | None = None,
        responder: Callable[[str, AIReviewInput], dict[str, Any]] | None = None,
    ) -> None:
        if adapter is not None and responder is not None:
            raise ValueError("configure either adapter or responder, not both")
        self.adapter = adapter if adapter is not None else (
            ResponderVisionAdapter(responder) if responder is not None else None
        )
        self.responder = responder

    def run_review(self, review_input: AIReviewInput) -> AIReviewOutput:
        if self.adapter is None:
            raise RuntimeError("AI review adapter is not configured")

        prompt = build_review_prompt(review_input)
        response = normalize_ai_review_response(self.adapter.review(prompt, review_input))
        validate_ai_review_response(response)
        normalized_commands = []
        for command in response["proposed_commands"]:
            normalized_command = dict(command)
            normalized_command.setdefault("proposal_source", "ai_review")
            normalized_commands.append(normalized_command)
        response["proposed_commands"] = normalized_commands
        return AIReviewOutput(
            summary=str(response["summary"]),
            issues=tuple(dict(issue) for issue in response["issues"]),
            proposed_commands=tuple(dict(command) for command in response["proposed_commands"]),
            prompt=prompt,
            review_input=review_input,
            raw_response=dict(response),
        )


__all__ = [
    "AIReviewInput",
    "AIReviewOutput",
    "AIReviewService",
    "AI_REVIEW_PROMPT",
    "SCHEMA_PATH",
    "build_review_prompt",
    "load_ai_command_schema",
    "normalize_ai_review_response",
    "validate_ai_review_response",
]
