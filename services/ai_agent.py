from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

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
- document_summary
- review_jobs
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
- ai_input_mode
- prompt_budget

When describing issues or commands:
- inspect algorithm candidates first and explain why a candidate should or should not be trusted
- use local review_jobs and the crop panels from the three images as the primary local visual context
- treat vector_document_json as a compact summary, not a full geometric document dump
- keep any replacement proposal at semantic intent level so later deterministic refinement can solve the exact geometry
- include topology guidance when path closure, gap, or continuity is suspicious
- include self_intersection guidance when paths cross or overlap incorrectly
- include alpha guidance when transparency or matte pollution affects interpretation
- include color guidance when style or color grouping appears wrong
- if an algorithm candidate primitive looks wrong, describe the mismatch in issues or semantic guidance instead of directly retyping a line candidate into an arc/circle/ellipse replacement command
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
    ai_input_mode: str = "legacy_document_context"
    document_summary: dict[str, Any] | None = None
    review_jobs: tuple[dict[str, Any], ...] = ()
    prompt_budget: dict[str, Any] | None = None
    ai_input_truncated: bool = False

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


class AIReviewInputTooLarge(ValueError):
    pass


class ProviderContextLimitExceeded(RuntimeError):
    pass


class AIReviewProviderTimeout(RuntimeError):
    pass


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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        interaction_logger: Callable[[dict[str, Any]], None] | None = None,
        timeout_seconds: float | None = None,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        if adapter is not None and responder is not None:
            raise ValueError("configure either adapter or responder, not both")
        self.adapter = adapter if adapter is not None else (
            ResponderVisionAdapter(responder) if responder is not None else None
        )
        self.responder = responder
        self.progress_callback = progress_callback
        self.interaction_logger = interaction_logger
        self.provider_name = ""
        self.provider_model = ""
        self.provider_status = ""
        self.timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)

    def set_progress_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.progress_callback = callback

    def set_interaction_logger(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.interaction_logger = callback

    def set_runtime_metadata(self, *, provider: str = "", model: str = "", status: str = "") -> None:
        self.provider_name = str(provider).strip().lower()
        self.provider_model = str(model).strip()
        self.provider_status = str(status).strip()

    def run_review(self, review_input: AIReviewInput) -> AIReviewOutput:
        if self.adapter is None:
            raise RuntimeError("AI review adapter is not configured")

        prompt = build_review_prompt(review_input)
        max_prompt_chars = _max_prompt_chars(review_input.prompt_budget)
        if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
            raise AIReviewInputTooLarge(
                f"AI review input exceeds max prompt chars: {len(prompt)} > {max_prompt_chars}"
            )
        provider_metadata = self._provider_metadata()
        image_paths = tuple(
            item
            for item in (
                review_input.original_image,
                review_input.overlay_image,
                review_input.distance_field_diff_image,
            )
            if item
        )
        panel_count = sum(int(job.get("image_count", 0)) for job in review_input.review_jobs)
        interaction_id = f"ai_review_{uuid4().hex}"
        request_payload = {
            "interaction_id": interaction_id,
            "provider": provider_metadata["provider"],
            "model": provider_metadata["model"],
            "status": "request_sent",
            "prompt": prompt,
            "prompt_char_count": len(prompt),
            "image_paths": list(image_paths),
            "image_file_count": len(image_paths),
            "panel_count": panel_count,
            "review_job_count": len(review_input.review_jobs),
            "review_input_summary": _review_input_summary(review_input),
        }
        self._record_interaction(request_payload)
        self._emit_progress(
            "ai_provider_call_start",
            message="Calling AI review provider.",
            provider=provider_metadata["provider"],
            model=provider_metadata["model"],
            status=provider_metadata["status"],
            prompt_char_count=len(prompt),
            image_file_count=len(image_paths),
            panel_count=panel_count,
            review_job_count=len(review_input.review_jobs),
        )
        try:
            raw_response, duration_ms = self._call_provider_with_monitoring(
                prompt=prompt,
                review_input=review_input,
                provider_metadata=provider_metadata,
                prompt_char_count=len(prompt),
                image_file_count=len(image_paths),
                panel_count=panel_count,
                review_job_count=len(review_input.review_jobs),
            )
        except Exception as exc:
            timeout_error = isinstance(exc, AIReviewProviderTimeout) or _looks_like_timeout(exc)
            duration_ms = getattr(exc, "duration_ms", None)
            self._emit_progress(
                "ai_provider_call_done",
                message="AI review provider call timed out." if timeout_error else "AI review provider call failed.",
                provider=provider_metadata["provider"],
                model=provider_metadata["model"],
                status="timeout" if timeout_error else "error",
                duration_ms=0.0 if duration_ms is None else duration_ms,
                error_type=type(exc).__name__,
            )
            self._record_interaction(
                {
                    "interaction_id": interaction_id,
                    "status": "timeout" if timeout_error else "failed",
                    "error_type": "AIReviewProviderTimeout" if timeout_error else type(exc).__name__,
                    "message": str(exc),
                    "duration_ms": None if duration_ms is None else round(float(duration_ms), 3),
                }
            )
            if _looks_like_context_overflow(exc):
                raise ProviderContextLimitExceeded(str(exc)) from exc
            if timeout_error and not isinstance(exc, AIReviewProviderTimeout):
                raise AIReviewProviderTimeout(str(exc)) from exc
            raise
        response = normalize_ai_review_response(raw_response)
        validate_ai_review_response(response)
        self._emit_progress(
            "ai_provider_call_done",
            message="AI review provider call completed.",
            provider=provider_metadata["provider"],
            model=provider_metadata["model"],
            status=provider_metadata["status"],
            duration_ms=duration_ms,
            prompt_char_count=len(prompt),
            image_file_count=len(image_paths),
            panel_count=panel_count,
            review_job_count=len(review_input.review_jobs),
        )
        normalized_commands = []
        for command in response["proposed_commands"]:
            normalized_command = dict(command)
            normalized_command.setdefault("proposal_source", "ai_review")
            normalized_commands.append(normalized_command)
        response["proposed_commands"] = normalized_commands
        primitive_mismatch_warnings = _primitive_mismatch_warnings(response["proposed_commands"])
        self._record_interaction(
            {
                "interaction_id": interaction_id,
                "provider": provider_metadata["provider"],
                "model": provider_metadata["model"],
                "status": "completed",
                "raw_response": json.loads(json.dumps(raw_response)),
                "normalized_response": json.loads(json.dumps(response)),
                "primitive_mismatch_warnings": primitive_mismatch_warnings,
                "duration_ms": round(duration_ms, 3),
            }
        )
        return AIReviewOutput(
            summary=str(response["summary"]),
            issues=tuple(dict(issue) for issue in response["issues"]),
            proposed_commands=tuple(dict(command) for command in response["proposed_commands"]),
            prompt=prompt,
            review_input=review_input,
            raw_response=dict(response),
        )

    def _call_provider_with_monitoring(
        self,
        *,
        prompt: str,
        review_input: AIReviewInput,
        provider_metadata: dict[str, str],
        prompt_char_count: int,
        image_file_count: int,
        panel_count: int,
        review_job_count: int,
    ) -> tuple[dict[str, Any], float]:
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        provider_start = perf_counter()

        def _worker() -> None:
            try:
                result_queue.put(("result", self.adapter.review(prompt, review_input)))
            except BaseException as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=_worker, name="ai-review-provider-call", daemon=True)
        thread.start()
        heartbeat_interval = self.heartbeat_interval_seconds if self.heartbeat_interval_seconds > 0 else None

        while True:
            provider_elapsed_seconds = perf_counter() - provider_start
            remaining_timeout = None if self.timeout_seconds is None else max(self.timeout_seconds - provider_elapsed_seconds, 0.0)
            if remaining_timeout is not None and remaining_timeout <= 0.0:
                error = AIReviewProviderTimeout(
                    "AI review provider timed out after "
                    f"{self.timeout_seconds:.1f}s for provider={provider_metadata['provider']} "
                    f"model={provider_metadata['model']}. Adjust --ai-review-timeout-seconds and retry."
                )
                setattr(error, "duration_ms", provider_elapsed_seconds * 1000.0)
                raise error

            wait_seconds = heartbeat_interval
            if wait_seconds is None:
                wait_seconds = remaining_timeout
            elif remaining_timeout is not None:
                wait_seconds = min(wait_seconds, remaining_timeout)
            thread.join(timeout=wait_seconds)

            if not result_queue.empty():
                result_type, payload = result_queue.get_nowait()
                duration_ms = (perf_counter() - provider_start) * 1000.0
                if result_type == "error":
                    setattr(payload, "duration_ms", duration_ms)
                    raise payload
                return payload, duration_ms

            if thread.is_alive() and heartbeat_interval is not None:
                provider_elapsed_ms = round((perf_counter() - provider_start) * 1000.0, 3)
                self._emit_progress(
                    "ai_provider_call_waiting",
                    message="Waiting for AI review provider response.",
                    provider=provider_metadata["provider"],
                    model=provider_metadata["model"],
                    status=provider_metadata["status"],
                    provider_elapsed_ms=provider_elapsed_ms,
                    prompt_char_count=prompt_char_count,
                    image_file_count=image_file_count,
                    panel_count=panel_count,
                    review_job_count=review_job_count,
                )

    def _provider_metadata(self) -> dict[str, str]:
        provider = self.provider_name
        model = self.provider_model
        status = self.provider_status or "enabled"
        if not provider and self.adapter is not None:
            adapter_name = type(self.adapter).__name__.lower()
            if "openai" in adapter_name:
                provider = "openai"
            elif "gemini" in adapter_name:
                provider = "gemini"
            elif "siliconflow" in adapter_name:
                provider = "siliconflow"
            elif "file" in adapter_name:
                provider = "file"
            elif "recorded" in adapter_name:
                provider = str(getattr(self.adapter, "provider_name", "")).strip().lower() or "recorded"
            else:
                provider = adapter_name
        if not model and self.adapter is not None:
            model = str(getattr(self.adapter, "model", "")).strip()
        return {"provider": provider, "model": model, "status": status}

    def _emit_progress(self, stage: str, *, message: str, **fields: Any) -> None:
        if self.progress_callback is None:
            return
        event = {"stage": stage, "message": message}
        event.update(fields)
        self.progress_callback(event)

    def _record_interaction(self, payload: dict[str, Any]) -> None:
        if self.interaction_logger is None:
            return
        self.interaction_logger(payload)


def _max_prompt_chars(prompt_budget: dict[str, Any] | None) -> int | None:
    if not isinstance(prompt_budget, dict):
        return None
    value = prompt_budget.get("max_prompt_chars")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return int(value)


def _looks_like_context_overflow(exc: BaseException) -> bool:
    message = str(exc).lower()
    patterns = (
        "context length",
        "too many tokens",
        "max_seq_len",
        "maximum context",
        "input tokens",
        "prompt is too long",
        "context window",
    )
    return any(pattern in message for pattern in patterns)


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def _review_input_summary(review_input: AIReviewInput) -> dict[str, Any]:
    return {
        "ai_input_mode": review_input.ai_input_mode,
        "ai_input_truncated": bool(review_input.ai_input_truncated),
        "fit_error": float(review_input.fit_error),
        "complexity_score": float(review_input.complexity_score),
        "topology_status": review_input.topology_status,
        "self_intersection_count": int(review_input.self_intersection_count),
        "document_summary": json.loads(json.dumps(review_input.document_summary)),
        "preview_summary": json.loads(json.dumps(review_input.preview_summary)),
        "prompt_budget": json.loads(json.dumps(review_input.prompt_budget)),
        "review_job_count": len(review_input.review_jobs),
        "candidate_count": len(review_input.candidates),
        "algorithm_command_count": len(review_input.proposed_commands_from_algorithm),
        "policy_feedback_count": len(review_input.policy_feedback),
        "rejection_memory_count": len(review_input.rejection_memory),
        "forbidden_repeated_command_count": len(review_input.forbidden_repeated_commands),
        "image_paths": [
            item
            for item in (
                review_input.original_image,
                review_input.overlay_image,
                review_input.distance_field_diff_image,
            )
            if item
        ],
        "review_jobs": json.loads(json.dumps(review_input.review_jobs)),
    }


def _primitive_mismatch_warnings(commands: list[dict[str, Any]]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for command in commands:
        candidate_target_type = _candidate_target_type(command.get("candidate_id"))
        command_target_type = _command_target_type(command.get("tool"))
        if candidate_target_type and command_target_type and candidate_target_type != command_target_type:
            warnings.append(
                {
                    "candidate_id": str(command.get("candidate_id")),
                    "candidate_target_type": candidate_target_type,
                    "command_tool": str(command.get("tool")),
                    "command_target_type": command_target_type,
                    "warning": "candidate_target_type_mismatch",
                }
            )
    return warnings


def _candidate_target_type(candidate_id: Any) -> str | None:
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return None
    known_targets = {"circle", "ellipse", "rectangle", "line", "arc", "bezier"}
    for part in candidate_id.split(":"):
        normalized = part.strip().lower()
        if normalized in known_targets:
            return normalized
    return None


def _command_target_type(tool: Any) -> str | None:
    if not isinstance(tool, str) or not tool.strip():
        return None
    normalized = tool.strip().lower()
    suffix_map = {
        "circle": "circle",
        "ellipse": "ellipse",
        "rectangle": "rectangle",
        "line": "line",
        "arc": "arc",
        "bezier": "bezier",
    }
    for suffix, target_type in suffix_map.items():
        if normalized.endswith(f"_with_{suffix}"):
            return target_type
    return None


__all__ = [
    "AIReviewInput",
    "AIReviewOutput",
    "AIReviewService",
    "AIReviewInputTooLarge",
    "AIReviewProviderTimeout",
    "AI_REVIEW_PROMPT",
    "ProviderContextLimitExceeded",
    "SCHEMA_PATH",
    "build_review_prompt",
    "load_ai_command_schema",
    "normalize_ai_review_response",
    "validate_ai_review_response",
]
