from __future__ import annotations

import json
import hashlib
import math
import mimetypes
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.common import encode_image_as_data_url
from services.free_pen_canvas import FreePenCanvasError, FreePenCanvasState
from services.free_pen_conversation import FreePenConversationMemory
from services.free_pen_native_tools import build_free_pen_native_tools_schema, parse_native_tool_call
from services.free_pen_prompt import (
    FreePenPromptInput,
    FreePenToolPromptInput,
    build_free_pen_prompt,
    build_free_pen_tool_state_text,
    build_free_pen_tool_system_prompt,
)
from services.public_image_resolver import PublicImageResolver


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "ai_free_pen.schema.json"
TOOL_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "ai_free_pen_tool.schema.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class FreePenReviewInput:
    original_image: str
    overlay_image: str | None
    distance_field_diff_image: str | None
    canvas_width: int
    canvas_height: int
    round_index: int
    max_rounds: int
    coordinate_space: str = "image_px"
    previous_overlay_available: bool = False

    def prompt_input(self) -> FreePenPromptInput:
        return FreePenPromptInput(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            round_index=self.round_index,
            max_rounds=self.max_rounds,
            coordinate_space=self.coordinate_space,
            previous_overlay_available=self.previous_overlay_available,
        )


@dataclass(frozen=True, slots=True)
class FreePenRunResult:
    status: str
    final_overlay_path: Path
    rounds_executed: int
    final_decision: str | None
    final_reason: str | None
    error_message: str | None = None
    response_files: tuple[Path, ...] = ()
    round_summaries: tuple[dict[str, Any], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_overlay_path"] = str(self.final_overlay_path)
        payload["response_files"] = [str(path) for path in self.response_files]
        return payload


def load_free_pen_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_free_pen_response(response: dict[str, Any]) -> None:
    Draft202012Validator(load_free_pen_schema()).validate(normalize_free_pen_response(response))


def normalize_free_pen_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("Free pen response must be a dict")

    normalized = dict(response)
    decision = str(normalized.get("decision", "")).strip().lower()
    normalized["decision"] = decision
    if decision == "draw":
        normalized["coordinate_space"] = str(normalized.get("coordinate_space", "")).strip()
        segments = normalized.get("segments", ())
        if not isinstance(segments, (list, tuple)):
            raise ValueError("draw decision requires a list of segments")
        normalized["segments"] = [_normalize_segment(segment) for segment in segments]
    return normalized


def _normalize_segment(segment: Any) -> dict[str, list[float]]:
    if not isinstance(segment, dict):
        raise ValueError("Bezier segment must be a dict")
    normalized: dict[str, list[float]] = {}
    for key in ("p0", "c1", "c2", "p1"):
        point = segment.get(key)
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Bezier segment point `{key}` must be a 2-item list")
        normalized[key] = [float(point[0]), float(point[1])]
    return normalized


@dataclass(slots=True)
class FreePenRuntime:
    adapter: VisionReviewAdapter
    max_rounds: int = 1
    stroke_width: int = 3
    stroke_rgba: tuple[int, int, int, int] = (0, 255, 0, 255)
    sample_count_per_segment: int = 64
    interaction_logger: Callable[[dict[str, Any]], None] | None = None
    raw_response_logger: Callable[[int, Any], None] | None = None
    provider_name: str = ""
    provider_model: str = ""

    def run(self, source_image_path: Path, output_dir: Path) -> FreePenRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_image = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise ValueError(f"failed to load source image: {source_image_path}")

        height, width = source_image.shape[:2]
        current_overlay = self._empty_overlay(width=width, height=height)
        response_files: list[Path] = []
        round_summaries: list[dict[str, Any]] = []
        final_decision: str | None = None
        final_reason: str | None = None
        error_message: str | None = None

        for round_index in range(1, max(1, int(self.max_rounds)) + 1):
            review_input = FreePenReviewInput(
                original_image=str(source_image_path),
                overlay_image=None,
                distance_field_diff_image=None,
                canvas_width=width,
                canvas_height=height,
                round_index=round_index,
                max_rounds=max(1, int(self.max_rounds)),
            )
            prompt = build_free_pen_prompt(review_input.prompt_input())
            interaction_id = f"free_pen_{uuid4().hex}"
            raw_response: Any = None
            normalized_response: dict[str, Any] | None = None
            response_error: str | None = None
            status = "received"
            self._record_interaction(
                {
                    "interaction_id": interaction_id,
                    "provider": self.provider_name,
                    "model": self.provider_model,
                    "status": "request_sent",
                    "round_index": round_index,
                    "prompt": prompt,
                    "prompt_char_count": len(prompt),
                    "image_paths": [str(source_image_path)],
                    "image_file_count": 1,
                    "canvas_width": width,
                    "canvas_height": height,
                    "source_image_size_bytes": int(source_image_path.stat().st_size),
                    "image_upload_summary": self._image_upload_summary(source_image_path),
                    "provider_request_content_summary": self._provider_request_content_summary(
                        prompt=prompt,
                        source_image_path=source_image_path,
                    ),
                }
            )
            try:
                raw_response = self.adapter.review(prompt, review_input)
                self._record_raw_response(round_index, raw_response)
                normalized_response = normalize_free_pen_response(raw_response)
                validate_free_pen_response(normalized_response)
                final_decision = str(normalized_response["decision"])
                final_reason = str(normalized_response.get("reason") or "").strip() or None
                self._record_interaction(
                    {
                        "interaction_id": interaction_id,
                        "provider": self.provider_name,
                        "model": self.provider_model,
                        "status": "completed",
                        "round_index": round_index,
                        "raw_response": raw_response,
                        "normalized_response": normalized_response,
                        "final_decision": final_decision,
                    }
                )
                if final_decision == "draw":
                    current_overlay = self._render_overlay(
                        width=width,
                        height=height,
                        segments=tuple(normalized_response["segments"]),
                    )
                    status = "drawn"
                elif final_decision == "accept":
                    status = "accepted"
                    response_files.append(
                        self._write_round_response(
                            output_dir=output_dir,
                            round_index=round_index,
                            prompt=prompt,
                            raw_response=raw_response,
                            normalized_response=normalized_response,
                            status=status,
                            error=None,
                        )
                    )
                    round_summaries.append(
                        {
                            "round_index": round_index,
                            "decision": final_decision,
                            "status": status,
                            "reason": final_reason,
                        }
                    )
                    break
                elif final_decision == "stalled":
                    status = "stalled"
                    response_files.append(
                        self._write_round_response(
                            output_dir=output_dir,
                            round_index=round_index,
                            prompt=prompt,
                            raw_response=raw_response,
                            normalized_response=normalized_response,
                            status=status,
                            error=None,
                        )
                    )
                    round_summaries.append(
                        {
                            "round_index": round_index,
                            "decision": final_decision,
                            "status": status,
                            "reason": final_reason,
                        }
                    )
                    break
                else:
                    raise ValueError(f"unsupported decision: {final_decision}")
            except Exception as exc:
                response_error = str(exc)
                error_message = response_error
                status = "invalid_response"
                self._record_interaction(
                    {
                        "interaction_id": interaction_id,
                        "provider": self.provider_name,
                        "model": self.provider_model,
                        "status": "failed",
                        "round_index": round_index,
                        "raw_response": raw_response,
                        "normalized_response": normalized_response,
                        "error": response_error,
                    }
                )

            response_files.append(
                self._write_round_response(
                    output_dir=output_dir,
                    round_index=round_index,
                    prompt=prompt,
                    raw_response=raw_response,
                    normalized_response=normalized_response,
                    status=status,
                    error=response_error,
                )
            )
            round_summaries.append(
                {
                    "round_index": round_index,
                    "decision": final_decision,
                    "status": status,
                    "reason": final_reason,
                    "error": response_error,
                }
            )

            if response_error is not None:
                break

        final_overlay_path = output_dir / "final_overlay.png"
        cv2.imwrite(str(final_overlay_path), current_overlay)

        status = "max_rounds_reached"
        if error_message is not None:
            status = "invalid_response"
        elif final_decision == "accept":
            status = "accepted"
        elif final_decision == "stalled":
            status = "stalled"
        elif final_decision == "draw":
            status = "drawn"

        return FreePenRunResult(
            status=status,
            final_overlay_path=final_overlay_path,
            rounds_executed=len(round_summaries),
            final_decision=final_decision,
            final_reason=final_reason,
            error_message=error_message,
            response_files=tuple(response_files),
            round_summaries=tuple(round_summaries),
        )

    def _render_overlay(self, *, width: int, height: int, segments: tuple[dict[str, list[float]], ...]) -> np.ndarray:
        overlay = self._empty_overlay(width=width, height=height)
        color = tuple(int(channel) for channel in self.stroke_rgba)
        for segment in segments:
            sampled = self._sample_cubic_segment(segment)
            if len(sampled) < 2:
                continue
            cv2.polylines(
                overlay,
                [sampled],
                isClosed=False,
                color=color,
                thickness=max(1, int(self.stroke_width)),
                lineType=cv2.LINE_AA,
            )
        return overlay

    def _sample_cubic_segment(self, segment: dict[str, list[float]]) -> np.ndarray:
        p0 = np.array(segment["p0"], dtype=np.float64)
        c1 = np.array(segment["c1"], dtype=np.float64)
        c2 = np.array(segment["c2"], dtype=np.float64)
        p1 = np.array(segment["p1"], dtype=np.float64)
        samples: list[list[int]] = []
        sample_count = max(8, int(self.sample_count_per_segment))
        for index in range(sample_count):
            t = index / float(sample_count - 1)
            point = (
                ((1.0 - t) ** 3) * p0
                + 3.0 * ((1.0 - t) ** 2) * t * c1
                + 3.0 * (1.0 - t) * (t**2) * c2
                + (t**3) * p1
            )
            samples.append([int(round(point[0])), int(round(point[1]))])
        return np.asarray(samples, dtype=np.int32)

    @staticmethod
    def _empty_overlay(*, width: int, height: int) -> np.ndarray:
        return np.zeros((int(height), int(width), 4), dtype=np.uint8)

    def _record_interaction(self, payload: dict[str, Any]) -> None:
        if self.interaction_logger is not None:
            self.interaction_logger(payload)

    def _record_raw_response(self, round_index: int, raw_response: Any) -> None:
        if self.raw_response_logger is not None:
            self.raw_response_logger(round_index, raw_response)

    def _image_upload_summary(self, source_image_path: Path) -> dict[str, Any]:
        raw_bytes = source_image_path.read_bytes()
        mime_type = mimetypes.guess_type(source_image_path.name)[0] or "application/octet-stream"
        data_url = self._data_url_from_bytes(raw_bytes=raw_bytes, mime_type=mime_type)
        return {
            "path": str(source_image_path),
            "mime_type": mime_type,
            "file_size_bytes": len(raw_bytes),
            "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "data_url_header": data_url.split(",", 1)[0],
            "data_url_char_count": len(data_url),
            "data_url_sha256": hashlib.sha256(data_url.encode("utf-8")).hexdigest(),
        }

    def _provider_request_content_summary(self, *, prompt: str, source_image_path: Path) -> dict[str, Any]:
        image_summary = self._image_upload_summary(source_image_path)
        provider = self.provider_name.strip().lower()
        if provider == "openai":
            return {
                "provider_format": "openai.responses",
                "content": [
                    {
                        "type": "input_image",
                        "image_url_header": image_summary["data_url_header"],
                        "image_url_char_count": image_summary["data_url_char_count"],
                        "image_url_sha256": image_summary["data_url_sha256"],
                    },
                    {"type": "input_text", "text_char_count": len(prompt)},
                ],
            }
        if provider == "siliconflow":
            return {
                "provider_format": "openai.chat.completions",
                "content": [
                    {
                        "type": "image_url",
                        "image_url_header": image_summary["data_url_header"],
                        "image_url_char_count": image_summary["data_url_char_count"],
                        "image_url_sha256": image_summary["data_url_sha256"],
                        "detail": "auto",
                    },
                    {"type": "text", "text_char_count": len(prompt)},
                ],
            }
        if provider == "gemini":
            return {
                "provider_format": "gemini.generate_content",
                "content": [
                    {
                        "type": "image_file",
                        "mime_type": image_summary["mime_type"],
                        "file_size_bytes": image_summary["file_size_bytes"],
                        "file_sha256": image_summary["file_sha256"],
                    },
                    {"type": "text", "text_char_count": len(prompt)},
                ],
            }
        return {
            "provider_format": provider or type(self.adapter).__name__,
            "content": [
                {
                    "type": "image_file",
                    "mime_type": image_summary["mime_type"],
                    "file_size_bytes": image_summary["file_size_bytes"],
                    "file_sha256": image_summary["file_sha256"],
                },
                {"type": "text", "text_char_count": len(prompt)},
            ],
        }

    @staticmethod
    def _data_url_from_bytes(*, raw_bytes: bytes, mime_type: str) -> str:
        import base64

        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _write_round_response(
        *,
        output_dir: Path,
        round_index: int,
        prompt: str,
        raw_response: Any,
        normalized_response: dict[str, Any] | None,
        status: str,
        error: str | None,
    ) -> Path:
        response_path = output_dir / f"round_{round_index:03d}_response.json"
        payload = {
            "round_index": round_index,
            "status": status,
            "prompt": prompt,
            "raw_response": raw_response,
            "normalized_response": normalized_response,
            "error": error,
        }
        response_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return response_path


@dataclass(frozen=True, slots=True)
class FreePenToolReviewInput:
    original_image: str
    overlay_image: str | None
    distance_field_diff_image: str | None
    canvas_width: int
    canvas_height: int
    step_index: int
    max_steps: int
    coordinate_space: str = "image_px"
    previous_overlay_available: bool = False
    path_open: bool = False
    current_point: list[float] | None = None
    current_subpath_start: list[float] | None = None
    path_count: int = 0
    closed_path_count: int = 0
    successful_step_count: int = 0
    invalid_step_count: int = 0
    recent_history: tuple[str, ...] = ()
    current_feedback: tuple[str, ...] = ()
    current_goal: str = ""
    last_action: str = ""
    allowed_next_actions: tuple[str, ...] = ()
    forbidden_next_actions: tuple[str, ...] = ()
    messages: tuple[dict[str, Any], ...] = ()
    session_state: dict[str, Any] | None = None
    image_transport: str = "base64"
    public_image_base_url: str | None = None
    tool_mode: str = "native_tools"
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | None = None

    def prompt_input(self) -> FreePenToolPromptInput:
        return FreePenToolPromptInput(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            step_index=self.step_index,
            max_steps=self.max_steps,
            coordinate_space=self.coordinate_space,
            previous_overlay_available=self.previous_overlay_available,
            path_open=self.path_open,
            current_point=self.current_point,
            current_subpath_start=self.current_subpath_start,
            path_count=self.path_count,
            closed_path_count=self.closed_path_count,
            successful_step_count=self.successful_step_count,
            invalid_step_count=self.invalid_step_count,
            recent_history=self.recent_history,
            current_feedback=self.current_feedback,
            current_goal=self.current_goal,
            last_action=self.last_action,
            allowed_next_actions=self.allowed_next_actions,
            forbidden_next_actions=self.forbidden_next_actions,
        )


@dataclass(frozen=True, slots=True)
class FreePenImageTransportConfig:
    mode: str = "base64"
    public_image_base_url: str | None = None
    public_image_root: Path = PROJECT_ROOT
    conversation_max_turns: int = 30


@dataclass(frozen=True, slots=True)
class FreePenToolRunResult:
    status: str
    final_overlay_path: Path
    final_composite_path: Path
    paths_json_path: Path
    tool_trace_path: Path
    rounds_executed: int
    successful_step_count: int
    invalid_step_count: int
    rejected_step_count: int
    rollback_count: int
    final_decision: str | None
    final_reason: str | None
    error_message: str | None = None
    error_type: str | None = None
    response_files: tuple[Path, ...] = ()
    overlay_files: tuple[Path, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_overlay_path"] = str(self.final_overlay_path)
        payload["final_composite_path"] = str(self.final_composite_path)
        payload["paths_json_path"] = str(self.paths_json_path)
        payload["tool_trace_path"] = str(self.tool_trace_path)
        payload["response_files"] = [str(path) for path in self.response_files]
        payload["overlay_files"] = [str(path) for path in self.overlay_files]
        return payload


def load_free_pen_tool_schema() -> dict[str, Any]:
    schema = json.loads(TOOL_SCHEMA_PATH.read_text(encoding="utf-8"))
    tool_one_of = schema.setdefault("$defs", {}).setdefault("tool_call", {}).setdefault("oneOf", [])
    if not any(
        isinstance(entry, dict)
        and isinstance(entry.get("properties"), dict)
        and entry["properties"].get("tool", {}).get("const") == "convert_line_to_curve"
        for entry in tool_one_of
    ):
        tool_one_of.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["tool", "segment_id", "c1", "c2"],
                "properties": {
                    "tool": {"const": "convert_line_to_curve"},
                    "segment_id": {"type": "string", "minLength": 1},
                    "c1": {"$ref": "#/$defs/point2d"},
                    "c2": {"$ref": "#/$defs/point2d"},
                },
            }
        )
    return schema


def validate_free_pen_tool_response(response: dict[str, Any]) -> None:
    Draft202012Validator(load_free_pen_tool_schema()).validate(normalize_free_pen_tool_response(response))


def normalize_free_pen_tool_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("Free pen tool response must be a dict")
    normalized = dict(response)
    decision = str(normalized.get("decision", "")).strip().lower()
    normalized["decision"] = decision
    if decision in {"finish", "stalled"}:
        reason = normalized.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{decision} decision requires a non-empty reason")
        normalized["reason"] = reason.strip()
        return normalized
    if decision == "tool_call":
        tool_call = normalized.get("tool_call")
        if not isinstance(tool_call, dict):
            raise ValueError("tool_call decision requires a tool_call object")
        reason = normalized.get("reason")
        nested_reason = tool_call.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            if isinstance(nested_reason, str) and nested_reason.strip():
                normalized["reason"] = nested_reason.strip()
            else:
                raise ValueError("tool_call decision requires a non-empty reason")
        normalized["tool_call"] = _normalize_tool_call(tool_call)
    return normalized


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ValueError("tool_call must be a dict")
    normalized = dict(tool_call)
    tool = str(normalized.get("tool", "")).strip().lower()
    normalized["tool"] = tool
    if tool in {"start_path", "line_to"}:
        normalized["x"] = float(normalized["x"])
        normalized["y"] = float(normalized["y"])
        return normalized
    if tool == "curve_to":
        normalized["c1"] = _normalize_point(normalized.get("c1"), key="c1")
        normalized["c2"] = _normalize_point(normalized.get("c2"), key="c2")
        normalized["p"] = _normalize_point(normalized.get("p"), key="p")
        return normalized
    if tool == "convert_line_to_curve":
        normalized["segment_id"] = str(normalized["segment_id"])
        normalized["c1"] = _normalize_point(normalized.get("c1"), key="c1")
        normalized["c2"] = _normalize_point(normalized.get("c2"), key="c2")
        return normalized
    if tool == "move_anchor":
        normalized["anchor_id"] = str(normalized["anchor_id"])
        normalized["x"] = float(normalized["x"])
        normalized["y"] = float(normalized["y"])
        return normalized
    if tool == "move_handle":
        normalized["segment_id"] = str(normalized["segment_id"])
        normalized["handle"] = str(normalized["handle"]).strip().lower()
        normalized["x"] = float(normalized["x"])
        normalized["y"] = float(normalized["y"])
        return normalized
    if tool == "set_segment_handles":
        normalized["segment_id"] = str(normalized["segment_id"])
        normalized["c1"] = _normalize_point(normalized.get("c1"), key="c1")
        normalized["c2"] = _normalize_point(normalized.get("c2"), key="c2")
        return normalized
    if tool == "close_path":
        return {"tool": "close_path"}
    if tool == "undo_last":
        return {"tool": "undo_last"}
    if tool == "rollback_to_step":
        normalized["step"] = int(normalized["step"])
        return normalized
    if tool == "inspect_history":
        last_n = normalized.get("last_n", 8)
        normalized["last_n"] = int(last_n)
        return normalized
    if tool == "restart_path":
        normalized["x"] = float(normalized["x"])
        normalized["y"] = float(normalized["y"])
        return normalized
    raise ValueError(f"unsupported free-pen tool: {tool}")


def _normalize_point(point: Any, *, key: str) -> list[float]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"{key} must be a 2-item list")
    return [float(point[0]), float(point[1])]


@dataclass(slots=True)
class FileSequenceFreePenAdapter(VisionReviewAdapter):
    response_path: Path
    _response_index: int = 0

    def review(self, prompt: str, review_input: object) -> dict[str, Any]:
        payload = json.loads(Path(self.response_path).read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            if self._response_index >= len(payload):
                raise ValueError("free pen file response sequence exhausted")
            entry = payload[self._response_index]
            self._response_index += 1
        else:
            entry = payload
        if not isinstance(entry, dict):
            raise ValueError("free pen file response adapter expects object entries")
        return dict(entry)


@dataclass(slots=True)
class NativeToolCallSequenceAdapter(VisionReviewAdapter):
    response_path: Path
    _response_index: int = 0

    def review(self, prompt: str, review_input: object) -> dict[str, Any]:
        payload = json.loads(Path(self.response_path).read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            if self._response_index >= len(payload):
                raise ValueError("free pen native tool response sequence exhausted")
            entry = payload[self._response_index]
            self._response_index += 1
        else:
            entry = payload
        if not isinstance(entry, dict):
            raise ValueError("free pen native tool response adapter expects object entries")
        tool_call_payload = entry.get("tool_call")
        if not isinstance(tool_call_payload, dict):
            raise ValueError("native tool sequence entry requires a tool_call object")
        call_id = str(tool_call_payload.get("id") or f"call_{self._response_index:03d}")
        name = str(tool_call_payload.get("name") or "").strip()
        arguments = tool_call_payload.get("arguments")
        canonical = parse_native_tool_call(name, arguments)
        raw_tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments),
            },
        }
        canonical["_parsed_from"] = "native_tool_call"
        canonical["_raw_tool_calls"] = [raw_tool_call]
        canonical["_assistant_message"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [raw_tool_call],
        }
        canonical["_raw_response"] = {
            "tool_calls": [raw_tool_call],
            "content": None,
        }
        return canonical


@dataclass(slots=True)
class FreePenToolRuntime:
    adapter: VisionReviewAdapter
    max_steps: int = 16
    stroke_width: int = 2
    stroke_rgba: tuple[int, int, int, int] = (0, 128, 255, 255)
    sample_count_per_segment: int = 64
    interaction_logger: Callable[[dict[str, Any]], None] | None = None
    raw_response_logger: Callable[[int, Any], None] | None = None
    provider_name: str = ""
    provider_model: str = ""
    history_summary_steps: int = 8
    max_rollbacks: int = 3
    closed_contour_mode: bool = True
    image_transport_config: FreePenImageTransportConfig = FreePenImageTransportConfig()

    _SMOOTH_REASON_HINTS = ("curve", "curved", "oval", "ellipse", "circle", "arc", "smooth")
    _OVERLAY_CONFUSION_HINTS = ("orange line", "overlay", "previous line", "current drawing", "draft line")
    _ADVANCING_TOOLS = {"curve_to", "line_to", "close_path"}
    _SEGMENT_ACCEPTABLE_P90_PX = 4.0
    _SEGMENT_ACCEPTABLE_MEAN_PX = 2.0
    _SEGMENT_ACCEPTABLE_MAX_PX = 8.0

    def run(self, source_image_path: Path, output_dir: Path) -> FreePenToolRunResult:
        import os

        # 1. 优先级：显式参数 > 环境变量
        output_dir.mkdir(parents=True, exist_ok=True)
        source_image = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise ValueError(f"failed to load source image: {source_image_path}")
        height, width = source_image.shape[:2]
        source_distance_map = self._build_source_distance_map(source_image)
        canvas = FreePenCanvasState(width=int(width), height=int(height))
        response_files: list[Path] = []
        overlay_files: list[Path] = []
        request_snapshot_paths: list[str] = []
        trace_rounds: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        successful_drawing_tool_calls: list[dict[str, Any]] = []
        final_decision: str | None = None
        final_reason: str | None = None
        error_message: str | None = None
        error_type: str | None = None
        previous_overlay_path: Path | None = None
        current_feedback: list[str] = []
        rejected_step_count = 0
        invalid_step_count = 0
        rollback_count = 0
        resolver = self._build_public_image_resolver()
        conversation = FreePenConversationMemory(
            system_message={"role": "system", "content": build_free_pen_tool_system_prompt()},
            max_turns=max(1, int(self.image_transport_config.conversation_max_turns)),
        )
        conversation_history_path = output_dir / "conversation_messages.json"

        for step_index in range(1, max(1, int(self.max_steps)) + 1):
            current_composite_path = self._write_round_composite_context(
                output_dir=output_dir,
                step_index=step_index,
                canvas=canvas,
                source_image=source_image,
            )
            current_segment_context = self._build_current_segment_context(
                canvas=canvas,
                source_distance_map=source_distance_map,
                focus_tool_call=None,
            )
            session_state = self._build_session_state(
                canvas=canvas,
                history=history,
                current_feedback=current_feedback,
                current_segment_context=current_segment_context,
            )

            if step_index == 1:
                image_message_content = self._build_round_image_message_content(
                    source_image_path=source_image_path,
                    overlay_image_path=None,
                    composite_image_path=None,
                    resolver=resolver,
                    include_source_image=True,
                )
                conversation.append_user_message(image_message_content)

            current_tools = tuple(build_free_pen_native_tools_schema())
            current_tool_choice = os.environ.get("AI_FREE_PEN_TOOL_CHOICE", "none")

            review_input = FreePenToolReviewInput(
                original_image=str(source_image_path),
                overlay_image=None if previous_overlay_path is None else str(previous_overlay_path),
                distance_field_diff_image=None,
                canvas_width=width,
                canvas_height=height,
                step_index=step_index,
                max_steps=max(1, int(self.max_steps)),
                previous_overlay_available=previous_overlay_path is not None,
                path_open=canvas.path_open,
                current_point=None
                if canvas.current_point is None
                else [canvas.current_point[0], canvas.current_point[1]],
                current_subpath_start=None
                if canvas.current_subpath_start is None
                else [canvas.current_subpath_start[0], canvas.current_subpath_start[1]],
                path_count=len(canvas.paths),
                closed_path_count=sum(1 for path in canvas.paths if path.closed),
                successful_step_count=len(successful_drawing_tool_calls),
                invalid_step_count=invalid_step_count,
                recent_history=tuple(self._recent_history_summary(history)),
                current_feedback=tuple(current_feedback),
                current_goal=str(session_state["current_goal"]),
                last_action=str(session_state["last_action"]),
                allowed_next_actions=tuple(session_state["allowed_next_actions"]),
                forbidden_next_actions=tuple(session_state["forbidden_next_actions"]),
                session_state=session_state,
                image_transport=self.image_transport_config.mode,
                public_image_base_url=self.image_transport_config.public_image_base_url,
                tool_mode="native_tools",
                tools=current_tools,
                tool_choice=current_tool_choice,
            )
            state_text = build_free_pen_tool_state_text(review_input.prompt_input())
            conversation.append_user_message(state_text)
            request_messages = conversation.build_messages_for_request()
            request_snapshot_path = output_dir / f"round_{step_index:03d}_request.json"
            self._write_request_snapshot(
                request_snapshot_path=request_snapshot_path,
                provider=self.provider_name,
                model=self.provider_model,
                image_transport=self.image_transport_config.mode,
                public_image_base_url=self.image_transport_config.public_image_base_url,
                messages=request_messages,
                session_state=session_state,
                tool_mode="native_tools",
                tools=current_tools,
                tool_choice=current_tool_choice,
            )
            request_snapshot_paths.append(str(request_snapshot_path))
            prompt = build_free_pen_tool_system_prompt()
            interaction_id = f"free_pen_tool_{uuid4().hex}"
            raw_response: Any = None
            normalized_response: dict[str, Any] | None = None
            validation_error: str | None = None
            preflight_result: dict[str, Any] | None = None
            executed_tool_call: dict[str, Any] | None = None
            rejected_tool_call: dict[str, Any] | None = None
            runtime_description = ""
            quality = "good"
            quality_summary = ""
            warnings: list[dict[str, Any]] = []
            round_status = "received"
            execution_result: dict[str, Any] | None = None
            tool_result_message: dict[str, Any] | None = None
            self._record_interaction(
                {
                    "interaction_id": interaction_id,
                    "provider": self.provider_name,
                    "model": self.provider_model,
                    "status": "request_sent",
                    "step_index": step_index,
                    "prompt": prompt,
                    "prompt_char_count": len(prompt),
                    "image_transport": self.image_transport_config.mode,
                    "public_image_base_url": self.image_transport_config.public_image_base_url,
                    "messages": self._sanitize_messages_for_snapshot(request_messages),
                    "image_urls": self._collect_image_urls_from_messages(request_messages),
                    "image_file_count": len(self._collect_image_urls_from_messages(request_messages)),
                    "canvas_width": width,
                    "canvas_height": height,
                    "review_input_summary": review_input.prompt_input().to_payload(),
                    "session_state": session_state,
                }
            )
            parsed_from = "native_tool_call"
            raw_tool_calls = None
            raw_response_for_save = None

            try:
                review_input = FreePenToolReviewInput(
                    **{**asdict(review_input), "messages": tuple(request_messages)}
                )
                raw_response = self.adapter.review(prompt, review_input)
                self._record_raw_response(step_index, raw_response)

                assistant_message = raw_response.get("_assistant_message")
                if assistant_message:
                    conversation.append_assistant_message(assistant_message)
                
                raw_response_for_save = raw_response.get("_raw_response", raw_response)
                parsed_from = raw_response.get("_parsed_from", "native_tool_call")
                raw_tool_calls = raw_response.get("_raw_tool_calls")
                
                clean_response = {k: v for k, v in raw_response.items() if not k.startswith("_")}

                normalized_response = normalize_free_pen_tool_response(clean_response)
                validate_free_pen_tool_response(normalized_response)
                
                final_decision = str(normalized_response["decision"])
                final_reason = str(normalized_response.get("reason") or "").strip() or None

                execution_result = {}
                if final_decision == "tool_call":
                    tool_call = dict(normalized_response["tool_call"])
                    preflight_result = self._preflight_tool_call(
                        tool_call=tool_call,
                        ai_reason=final_reason or "",
                        canvas=canvas,
                        successful_drawing_step_count=len(successful_drawing_tool_calls),
                        rollback_count=rollback_count,
                        current_segment_context=current_segment_context,
                    )
                    warnings = list(preflight_result["warnings"])
                    if not preflight_result["success"]:
                        rejected_step_count += 1
                        rejected_tool_call = tool_call
                        round_status = "rejected_action"
                        runtime_description = str(preflight_result["runtime_description"])
                        quality = "bad"
                        quality_summary = str(preflight_result["quality_summary"])
                        current_feedback = self._feedback_from_rejected_action(warnings=warnings, quality_summary=quality_summary)
                        current_segment_context = self._build_current_segment_context(
                            canvas=canvas,
                            source_distance_map=source_distance_map,
                            focus_tool_call=tool_call,
                        )
                        execution_result = {
                            "ok": False,
                            "accepted": False,
                            "runtime_description": runtime_description,
                            "quality_summary": quality_summary,
                            "warnings": warnings,
                        }
                    else:
                        (
                            executed_tool_call,
                            runtime_description,
                            quality,
                            quality_summary,
                            warnings,
                            rollback_applied,
                            current_feedback,
                        ) = self._execute_tool_call(
                            tool_call=tool_call,
                            ai_reason=final_reason or "",
                            canvas=canvas,
                            successful_drawing_tool_calls=successful_drawing_tool_calls,
                            history=history,
                        )
                        rollback_count += rollback_applied
                        current_segment_context = self._build_current_segment_context(
                            canvas=canvas,
                            source_distance_map=source_distance_map,
                            focus_tool_call=tool_call,
                        )
                        round_status = "tool_applied"
                        execution_result = {
                            "ok": True,
                            "accepted": True,
                            "runtime_description": runtime_description,
                            "quality_summary": quality_summary,
                            "warnings": warnings,
                        }
                elif final_decision == "finish":
                    finish_blocked_by_quality = bool(
                        current_segment_context["status"].get("may_advance_to_next_segment") is False
                    )
                    finish_warning = self._warning(
                        code="finish_with_open_path",
                        message="The current path is still open. Continue drawing, close_path when appropriate, or rollback.",
                    )
                    if finish_blocked_by_quality:
                        warnings = [
                            self._warning(
                                "current_segment_needs_refinement",
                                "The newest segment is not acceptable yet. Refine it before drawing the next segment.",
                            )
                        ]
                        rejected_step_count += 1
                        round_status = "rejected_action"
                        canvas.final_status = "invalid_finish"
                        quality = "bad"
                        quality_summary = warnings[0]["message"]
                        runtime_description = "Rejected finish because the newest segment still needs refinement."
                        current_feedback = self._feedback_from_rejected_action(warnings=warnings, quality_summary=quality_summary)
                        execution_result = {
                            "ok": False,
                            "accepted": False,
                            "runtime_description": runtime_description,
                            "quality_summary": quality_summary,
                            "warnings": warnings,
                        }
                    elif self.closed_contour_mode and canvas.path_open:
                        warnings = [finish_warning]
                        rejected_step_count += 1
                        round_status = "rejected_action"
                        canvas.final_status = "invalid_finish"
                        quality = "bad"
                        quality_summary = finish_warning["message"]
                        runtime_description = "Rejected finish because the current path is still open."
                        current_feedback = self._feedback_from_rejected_action(warnings=warnings, quality_summary=quality_summary)
                        execution_result = {
                            "ok": False,
                            "accepted": False,
                            "runtime_description": runtime_description,
                            "quality_summary": quality_summary,
                            "warnings": warnings,
                        }
                    else:
                        round_status = "finished"
                        canvas.final_status = "finish"
                        quality = "good"
                        quality_summary = "Finished tracing without validation warnings."
                        runtime_description = "Finished the tracing loop."
                        execution_result = {
                            "ok": True,
                            "accepted": True,
                            "runtime_description": runtime_description,
                            "quality_summary": quality_summary,
                            "warnings": warnings,
                        }
                elif final_decision == "stalled":
                    round_status = "stalled"
                    canvas.final_status = "stalled"
                    quality = "warning"
                    quality_summary = "The model reported that it could not continue reliably."
                    runtime_description = "Stopped the tracing loop because the model returned stalled."
                    execution_result = {
                        "ok": True,
                        "accepted": True,
                        "runtime_description": runtime_description,
                        "quality_summary": quality_summary,
                        "warnings": warnings,
                    }
                else:
                    raise ValueError(f"unsupported free-pen tool decision: {final_decision}")
                
                if raw_tool_calls:
                    tool_call_id = raw_tool_calls[0].get("id")
                    if tool_call_id:
                        session_state_after = self._build_session_state(
                            canvas=canvas,
                            history=history,
                            current_feedback=current_feedback,
                            current_segment_context=current_segment_context,
                        )
                        tool_result_content = {
                            **execution_result,
                            "state_after": session_state_after,
                            "runtime_description": runtime_description,
                            "quality_summary": quality_summary,
                            "warnings": warnings,
                            "editable_geometry": current_segment_context["editable_geometry"],
                            "geometry_hint": "Blue points are anchors. Green points and lines are control handles. Use move_anchor, move_handle, or set_segment_handles to refine cubic segments. If a line segment needs handle editing, use convert_line_to_curve first.",
                            "current_segment_focus": current_segment_context["focus"],
                            "current_segment_status": current_segment_context["status"],
                            "quality_metrics": current_segment_context["quality_metrics"],
                            "allowed_next_actions": session_state_after["allowed_next_actions"],
                            "forbidden_next_actions": session_state_after["forbidden_next_actions"],
                            "next_hint": self._next_hint(
                                final_decision=final_decision,
                                round_status=round_status,
                                warnings=warnings,
                                session_state=session_state_after,
                                current_segment_context=current_segment_context,
                            ),
                            "visual_feedback_hint": (
                                "A visual feedback user message with overlay/composite images will follow this tool result. "
                                "Inspect it before choosing the next tool. BLACK is target; ORANGE is your drawing; BLUE are anchors; GREEN are control handles."
                            ),
                        }
                        conversation.append_tool_result(tool_call_id, tool_result_content)
                        tool_result_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(tool_result_content, ensure_ascii=False),
                        }

                conversation.save(
                    conversation_history_path,
                    message_sanitizer=self._sanitize_messages_for_snapshot,
                )
            except Exception as exc:
                validation_error = str(exc)
                error_message = validation_error
                error_type = "ProviderTimeout" if self._is_timeout_error(exc) else type(exc).__name__
                round_status = "provider_timeout" if error_type == "ProviderTimeout" else "invalid_response"
                canvas.final_status = round_status
                invalid_step_count += 1
                quality = "bad"
                quality_summary = validation_error
                runtime_description = (
                    f"Provider call timed out at step {step_index}."
                    if round_status == "provider_timeout"
                    else f"Failed to parse or validate the model response at step {step_index}."
                )
                current_feedback = []
                conversation.save(
                    conversation_history_path,
                    message_sanitizer=self._sanitize_messages_for_snapshot,
                )

            overlay = canvas.render_overlay(
                stroke_width=max(1, int(self.stroke_width)),
                stroke_rgba=self.stroke_rgba,
                sample_count_per_segment=max(8, int(self.sample_count_per_segment)),
            )
            overlay_path = output_dir / f"round_{step_index:03d}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            overlay_files.append(overlay_path)
            previous_overlay_path = overlay_path

            post_tool_composite_path = self._write_post_tool_composite_feedback(
                output_dir=output_dir,
                step_index=step_index,
                canvas=canvas,
                source_image=source_image,
            )

            if validation_error is None and final_decision not in {"finish", "stalled"}:
                visual_feedback_content = self._build_post_tool_visual_feedback_message_content(
                    overlay_image_path=overlay_path,
                    composite_image_path=post_tool_composite_path,
                    resolver=resolver,
                    step_index=step_index,
                )
                conversation.append_user_message(visual_feedback_content)
                conversation.save(
                    conversation_history_path,
                    message_sanitizer=self._sanitize_messages_for_snapshot,
                )

            response_files.append(
                FreePenRuntime._write_round_response(
                    output_dir=output_dir,
                    round_index=step_index,
                    prompt=prompt,
                    raw_response=raw_response_for_save,
                    normalized_response=normalized_response,
                    status=round_status,
                    error=validation_error,
                )
            )
            history_entry = {
                "step": step_index,
                "tool": None if normalized_response is None or final_decision != "tool_call" else normalized_response["tool_call"]["tool"],
                "args": {} if normalized_response is None or final_decision != "tool_call" else dict(normalized_response["tool_call"]),
                "ai_reason": final_reason,
                "runtime_description": runtime_description,
                "quality": quality,
                "quality_summary": quality_summary,
                "warnings": warnings,
                "state_after": canvas.state_summary(),
                "overlay_path": overlay_path.name,
                "round_status": round_status,
            }
            history.append(history_entry)
            trace_rounds.append(
                {
                    "step_index": step_index,
                    "raw_response": raw_response_for_save,
                    "parsed_decision": final_decision,
                    "validation_result": {
                        "success": validation_error is None,
                        "error": validation_error,
                    },
                    "preflight_result": preflight_result,
                    "executed_tool_call": executed_tool_call,
                    "rejected_tool_call": rejected_tool_call,
                    "warnings": warnings,
                    "runtime_description": runtime_description,
                    "quality_summary": quality_summary,
                    "canvas_state_summary": canvas.state_summary(),
                    "current_segment_focus": current_segment_context["focus"],
                    "current_segment_status": current_segment_context["status"],
                    "quality_metrics": current_segment_context["quality_metrics"],
                    "reason": final_reason,
                    "output_overlay_path": str(overlay_path),
                    "request_snapshot_path": str(request_snapshot_path),
                    "image_urls": self._collect_image_urls_from_messages(request_messages),
                    "parsed_from": parsed_from,
                    "raw_tool_calls": raw_tool_calls,
                    "normalized_response": normalized_response,
                    "provider_parse_error": validation_error if validation_error else None,
                    "execution_result": execution_result,
                    "tool_result_message": tool_result_message,
                }
            )
            self._record_interaction(
                {
                    "interaction_id": interaction_id,
                    "provider": self.provider_name,
                    "model": self.provider_model,
                    "status": "completed" if validation_error is None else "failed",
                    "step_index": step_index,
                    "raw_response": raw_response,
                    "normalized_response": normalized_response,
                    "validation_error": validation_error,
                    "preflight_result": preflight_result,
                    "executed_tool_call": executed_tool_call,
                    "rejected_tool_call": rejected_tool_call,
                    "warnings": warnings,
                    "runtime_description": runtime_description,
                    "quality_summary": quality_summary,
                    "final_decision": final_decision,
                    "request_snapshot_path": str(request_snapshot_path),
                    "conversation_history_path": str(conversation_history_path),
                }
            )

            if round_status in {"provider_timeout", "invalid_response"}:
                break
            if final_decision in {"finish", "stalled"} and round_status != "rejected_action":
                break

        if canvas.final_status == "initialized":
            canvas.final_status = "max_steps_reached" if final_decision == "tool_call" else (final_decision or "initialized")

        final_overlay = canvas.render_overlay(
            stroke_width=max(1, int(self.stroke_width)),
            stroke_rgba=self.stroke_rgba,
            sample_count_per_segment=max(8, int(self.sample_count_per_segment)),
        )
        final_overlay_path = output_dir / "final_overlay.png"
        cv2.imwrite(str(final_overlay_path), final_overlay)
        final_composite_path = output_dir / "final_composite.png"
        composite = canvas.render_composite(
            source_image,
            stroke_width=max(1, int(self.stroke_width)),
            stroke_rgba=self.stroke_rgba,
            sample_count_per_segment=max(8, int(self.sample_count_per_segment)),
        )
        cv2.imwrite(str(final_composite_path), composite)

        paths_json_path = output_dir / "free_pen_paths.json"
        paths_json_path.write_text(json.dumps(canvas.paths_payload(), indent=2, ensure_ascii=False), encoding="utf-8")

        tool_trace_path = output_dir / "tool_trace.json"
        tool_trace_path.write_text(
            json.dumps(
                {
                    "successful_step_count": len(successful_drawing_tool_calls),
                    "invalid_step_count": invalid_step_count,
                    "rejected_step_count": rejected_step_count,
                    "rollback_count": rollback_count,
                    "final_status": canvas.final_status,
                    "conversation_history_path": str(conversation_history_path),
                    "request_snapshot_paths": request_snapshot_paths,
                    "image_transport": self.image_transport_config.mode,
                    "history": history,
                    "rounds": trace_rounds,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        status = canvas.final_status
        if final_decision == "finish" and canvas.final_status != "invalid_finish":
            status = "finished"
        elif final_decision == "stalled":
            status = "stalled"
        elif error_type == "ProviderTimeout":
            status = "provider_timeout"
        elif error_message is not None:
            status = "invalid_response"
        elif final_decision == "tool_call":
            status = "max_steps_reached"

        return FreePenToolRunResult(
            status=status,
            final_overlay_path=final_overlay_path,
            final_composite_path=final_composite_path,
            paths_json_path=paths_json_path,
            tool_trace_path=tool_trace_path,
            rounds_executed=len(trace_rounds),
            successful_step_count=len(successful_drawing_tool_calls),
            invalid_step_count=invalid_step_count,
            rejected_step_count=rejected_step_count,
            rollback_count=rollback_count,
            final_decision=final_decision,
            final_reason=final_reason,
            error_message=error_message,
            error_type=error_type,
            response_files=tuple(response_files),
            overlay_files=tuple(overlay_files),
        )
    
    def _write_post_tool_composite_feedback(
        self,
        *,
        output_dir: Path,
        step_index: int,
        canvas: FreePenCanvasState,
        source_image: np.ndarray,
    ) -> Path:
        composite = canvas.render_composite(
            source_image,
            stroke_width=max(1, int(self.stroke_width)),
            stroke_rgba=self.stroke_rgba,
            sample_count_per_segment=max(8, int(self.sample_count_per_segment)),
            show_handles=True,
        )
        composite_path = output_dir / f"round_{step_index:03d}_post_tool_composite.png"
        cv2.imwrite(str(composite_path), composite)
        return composite_path
    
    def _preflight_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        ai_reason: str,
        canvas: FreePenCanvasState,
        successful_drawing_step_count: int,
        rollback_count: int,
        current_segment_context: dict[str, Any],
    ) -> dict[str, Any]:
        tool = str(tool_call["tool"])
        warnings = self._reason_based_warnings(ai_reason=ai_reason, tool=tool)
        if tool in {"start_path", "line_to", "restart_path", "move_anchor"}:
            warnings.extend(self._coordinate_warnings(x=tool_call["x"], y=tool_call["y"], width=canvas.width, height=canvas.height))
        elif tool == "curve_to":
            for key in ("c1", "c2", "p"):
                warnings.extend(
                    self._coordinate_warnings(
                        x=tool_call[key][0],
                        y=tool_call[key][1],
                        width=canvas.width,
                        height=canvas.height,
                        point_name=key,
                    )
                )
        elif tool == "move_handle":
            warnings.extend(self._coordinate_warnings(x=tool_call["x"], y=tool_call["y"], width=canvas.width, height=canvas.height, point_name=tool_call["handle"]))
        elif tool == "set_segment_handles":
            for key in ("c1", "c2"):
                warnings.extend(
                    self._coordinate_warnings(
                        x=tool_call[key][0],
                        y=tool_call[key][1],
                        width=canvas.width,
                        height=canvas.height,
                        point_name=key,
                    )
                )
        elif tool == "convert_line_to_curve":
            for key in ("c1", "c2"):
                warnings.extend(
                    self._coordinate_warnings(
                        x=tool_call[key][0],
                        y=tool_call[key][1],
                        width=canvas.width,
                        height=canvas.height,
                        point_name=key,
                    )
                )
        reject_codes = {warning["code"] for warning in warnings if warning["code"] in {"non_finite_coordinate", "out_of_bounds_coordinate"}}
        if tool in {"line_to", "curve_to", "close_path", "move_anchor", "move_handle", "set_segment_handles", "convert_line_to_curve"} and not canvas.path_open:
            reject_codes.add("path_not_open")
            warnings.append(self._warning("path_not_open", f"{tool} requires an open path started by start_path."))
        if tool == "start_path" and canvas.path_open:
            reject_codes.add("path_already_open")
            warnings.append(self._warning("path_already_open", "A path is already open. Use restart_path, rollback_to_step, or close_path first."))
        if tool == "line_to" and any(warning["code"] == "line_to_used_on_smooth_curve_hint" for warning in warnings):
            reject_codes.add("line_to_used_on_smooth_curve_hint")
        if (
            current_segment_context["status"].get("may_advance_to_next_segment") is False
            and tool in self._ADVANCING_TOOLS
        ):
            reject_codes.add("current_segment_needs_refinement")
            warnings.append(
                self._warning(
                    "current_segment_needs_refinement",
                    "The newest segment is not acceptable yet. Refine it before drawing the next segment.",
                )
            )
        if tool == "close_path" and canvas.path_open:
            if canvas.current_path_drawable_segment_count() < 2 or not canvas.current_path_has_cubic_segment():
                reject_codes.add("close_path_used_too_early")
                warnings.append(
                    self._warning(
                        "close_path_used_too_early",
                        "close_path was called when the path had too few smooth drawable segments.",
                    )
                )
            distance_to_start = canvas.distance_to_start()
            if distance_to_start is not None:
                threshold = max(8.0, canvas.current_path_bbox_diagonal() * 0.10)
                if distance_to_start > threshold:
                    reject_codes.add("long_chord_if_closed")
                    warnings.append(
                        self._warning(
                            "long_chord_if_closed",
                            f"Closing now would create a long straight chord of length {distance_to_start:.2f} back to the start point.",
                        )
                    )
        if tool == "undo_last" and successful_drawing_step_count == 0:
            reject_codes.add("undo_without_history")
            warnings.append(self._warning("undo_without_history", "There is no successful drawing step to undo."))
        if tool == "rollback_to_step":
            target_step = int(tool_call["step"])
            if target_step < 0 or target_step > successful_drawing_step_count:
                reject_codes.add("rollback_step_out_of_range")
                warnings.append(
                    self._warning(
                        "rollback_step_out_of_range",
                        f"rollback_to_step must target an existing successful drawing step between 0 and {successful_drawing_step_count}.",
                    )
                )
        if tool == "inspect_history" and rollback_count > self.max_rollbacks:
            warnings.append(self._warning("rollback_budget_exceeded", "Rollback budget has already been exceeded. Prefer a direct correction."))
        if tool == "move_anchor" and canvas.path_open:
            geometry = canvas.editable_geometry()
            anchors = {anchor["id"] for path in geometry["paths"] for anchor in path["anchors"]}
            if str(tool_call["anchor_id"]) not in anchors:
                reject_codes.add("unknown_anchor_id")
                warnings.append(self._warning("unknown_anchor_id", f"Unknown anchor_id: {tool_call['anchor_id']}"))
        editable_geometry = current_segment_context["editable_geometry"]
        if tool in {"move_handle", "set_segment_handles", "convert_line_to_curve"} and canvas.path_open:
            geometry = editable_geometry
            segments = {
                segment["id"]: segment["type"]
                for path in geometry["paths"]
                for segment in path["segments"]
            }
            segment_id = str(tool_call["segment_id"])
            if segment_id not in segments:
                reject_codes.add("unknown_segment_id")
                warnings.append(self._warning("unknown_segment_id", f"Unknown segment_id: {segment_id}"))
            elif tool in {"move_handle", "set_segment_handles"} and segments[segment_id] != "cubic":
                reject_codes.add("segment_not_cubic")
                warnings.append(self._warning("segment_not_cubic", f"{segment_id} is not a cubic segment."))
            elif tool == "convert_line_to_curve" and segments[segment_id] != "line":
                reject_codes.add("segment_not_line")
                warnings.append(self._warning("segment_not_line", f"{segment_id} is not a line segment."))

        if reject_codes:
            runtime_description = f"Rejected {tool} during preflight validation."
            quality_summary = " ; ".join(warning["message"] for warning in warnings)
            if tool == "line_to" and "line_to_used_on_smooth_curve_hint" in reject_codes:
                runtime_description = "Rejected line_to because the model described a smooth curve but used a straight line tool."
                quality_summary = "line_to draws a straight segment and is not appropriate for the described smooth curve."
            elif "current_segment_needs_refinement" in reject_codes:
                runtime_description = "Rejected the next drawing step because the newest segment still needs refinement."
                quality_summary = "The newest segment is not acceptable yet. Refine it before drawing the next segment."
            return {
                "success": False,
                "rejected": True,
                "warnings": warnings,
                "runtime_description": runtime_description,
                "quality_summary": quality_summary,
            }
        return {
            "success": True,
            "rejected": False,
            "warnings": warnings,
            "runtime_description": f"Accepted {tool} for execution.",
            "quality_summary": "",
        }

    def _execute_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        ai_reason: str,
        canvas: FreePenCanvasState,
        successful_drawing_tool_calls: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str, str, str, list[dict[str, Any]], int, list[str]]:
        tool = str(tool_call["tool"])
        warnings = self._reason_based_warnings(ai_reason=ai_reason, tool=tool)
        rollback_applied = 0
        current_feedback: list[str] = []
        if tool == "inspect_history":
            last_n = max(1, min(20, int(tool_call.get("last_n", 8))))
            summary = self._recent_history_summary(history, last_n=last_n)
            runtime_description = f"Reviewed the most recent {last_n} history entries without changing the canvas."
            quality_summary = "Recent history was inspected successfully."
            current_feedback = summary if summary else ["No prior history is available yet."]
            return (
                {"tool": "inspect_history", "last_n": last_n},
                runtime_description,
                "good",
                quality_summary,
                warnings,
                rollback_applied,
                current_feedback,
            )

        if tool == "undo_last":
            target_step = max(0, len(successful_drawing_tool_calls) - 1)
            removed_steps = len(successful_drawing_tool_calls) - target_step
            del successful_drawing_tool_calls[target_step:]
            self._replay_successful_tool_calls(canvas=canvas, tool_calls=successful_drawing_tool_calls)
            rollback_applied = 1
            runtime_description = f"Rolled back the canvas by undoing the most recent successful drawing step to step {target_step}."
            quality_summary = f"Undo removed {removed_steps} previously successful drawing step(s)."
            current_feedback = ["The previous drawing step was removed. Continue from the corrected canvas state."]
            return (
                {"tool": "undo_last", "target_step": target_step},
                runtime_description,
                "good",
                quality_summary,
                warnings,
                rollback_applied,
                current_feedback,
            )

        if tool == "rollback_to_step":
            target_step = int(tool_call["step"])
            removed_steps = len(successful_drawing_tool_calls) - target_step
            del successful_drawing_tool_calls[target_step:]
            self._replay_successful_tool_calls(canvas=canvas, tool_calls=successful_drawing_tool_calls)
            rollback_applied = 1
            runtime_description = f"Rolled back the canvas to successful drawing step {target_step}."
            quality_summary = f"Rollback removed {removed_steps} drawing step(s) after step {target_step}."
            current_feedback = [
                f"The canvas was restored to drawing step {target_step}.",
                "Retry the next action from the corrected canvas state.",
            ]
            return (
                {"tool": "rollback_to_step", "step": target_step, "removed_steps": removed_steps},
                runtime_description,
                "good",
                quality_summary,
                warnings,
                rollback_applied,
                current_feedback,
            )

        if tool == "restart_path":
            successful_drawing_tool_calls.clear()
            self._replay_successful_tool_calls(canvas=canvas, tool_calls=())
            restart_call = {"tool": "start_path", "x": tool_call["x"], "y": tool_call["y"]}
            executed = canvas.apply_tool_call(restart_call)
            successful_drawing_tool_calls.append(restart_call)
            rollback_applied = 1
            runtime_description = f"Cleared the canvas and restarted a path at [{tool_call['x']:.2f}, {tool_call['y']:.2f}]."
            quality_summary = "Restarted the drawing from a new anchor point."
            current_feedback = ["The previous path was discarded. Continue tracing the black source stroke from the new start point."]
            return (
                {"tool": "restart_path", "x": tool_call["x"], "y": tool_call["y"], "executed_start_path": executed},
                runtime_description,
                "good",
                quality_summary,
                warnings,
                rollback_applied,
                current_feedback,
            )

        if tool in {"move_anchor", "move_handle", "set_segment_handles", "convert_line_to_curve"}:
            if tool == "convert_line_to_curve":
                executed = self._convert_line_to_curve(canvas=canvas, tool_call=tool_call)
                canvas.successful_step_count += 1
                canvas.step_count += 1
            else:
                executed = canvas.apply_tool_call(tool_call)
            successful_drawing_tool_calls.append(dict(tool_call))
            if tool == "move_anchor":
                runtime_description = (
                    f"Moved anchor {tool_call['anchor_id']} from "
                    f"[{executed['old_point'][0]:.2f},{executed['old_point'][1]:.2f}] to "
                    f"[{executed['point'][0]:.2f},{executed['point'][1]:.2f}]."
                )
            elif tool == "move_handle":
                runtime_description = (
                    f"Moved handle {tool_call['handle']} of {tool_call['segment_id']} from "
                    f"[{executed['old_point'][0]:.2f},{executed['old_point'][1]:.2f}] to "
                    f"[{executed['point'][0]:.2f},{executed['point'][1]:.2f}]."
                )
            elif tool == "convert_line_to_curve":
                runtime_description = (
                    f"Converted {tool_call['segment_id']} from a line segment into a cubic segment with "
                    f"c1=[{executed['c1'][0]:.2f},{executed['c1'][1]:.2f}] and "
                    f"c2=[{executed['c2'][0]:.2f},{executed['c2'][1]:.2f}]."
                )
            else:
                runtime_description = (
                    f"Set handles of {tool_call['segment_id']} to "
                    f"c1=[{executed['c1'][0]:.2f},{executed['c1'][1]:.2f}] and "
                    f"c2=[{executed['c2'][0]:.2f},{executed['c2'][1]:.2f}]."
                )
            quality_summary = "Updated the editable Bezier geometry."
            current_feedback = [
                "Inspect the BLUE anchors and GREEN handles before continuing.",
                "If the current segment is still misaligned, adjust handles or anchors before adding new geometry.",
            ]
            return (executed, runtime_description, "good", quality_summary, warnings, rollback_applied, current_feedback)

        before_current_point = canvas.current_point
        executed = canvas.apply_tool_call(tool_call)
        successful_drawing_tool_calls.append(dict(tool_call))
        warnings.extend(self._post_execution_warnings(tool_call=tool_call, ai_reason=ai_reason, canvas=canvas, before_current_point=before_current_point))
        quality = "good" if not warnings else "warning"
        quality_summary = (
            "Executed the requested tool call without validation warnings."
            if not warnings
            else " ; ".join(warning["message"] for warning in warnings)
        )
        runtime_description = self._runtime_description_for_tool(tool_call=tool_call, executed_tool_call=executed, before_current_point=before_current_point)
        current_feedback = self._feedback_from_executed_action(warnings=warnings, quality_summary=quality_summary, canvas=canvas)
        return (executed, runtime_description, quality, quality_summary, warnings, rollback_applied, current_feedback)

    def _post_execution_warnings(
        self,
        *,
        tool_call: dict[str, Any],
        ai_reason: str,
        canvas: FreePenCanvasState,
        before_current_point: tuple[float, float] | None,
    ) -> list[dict[str, Any]]:
        tool = str(tool_call["tool"])
        warnings: list[dict[str, Any]] = []
        if tool in {"line_to", "curve_to"} and before_current_point is not None and canvas.current_point is not None:
            segment_length = math.hypot(
                canvas.current_point[0] - before_current_point[0],
                canvas.current_point[1] - before_current_point[1],
            )
            if segment_length < 1.0:
                warnings.append(self._warning("duplicate_point", "The new segment endpoint is almost identical to the previous current point."))
                warnings.append(self._warning("zero_length_segment", "The drawn segment has near-zero length."))
        if tool == "line_to":
            if canvas.line_to_streak() >= 2:
                warnings.append(
                    self._warning(
                        "line_to_streak_too_long",
                        f"The current path now contains a line_to streak of {canvas.line_to_streak()} consecutive straight segments.",
                    )
                )
            if any(token in ai_reason.lower() for token in self._SMOOTH_REASON_HINTS):
                warnings.append(
                    self._warning(
                        "line_to_used_on_smooth_curve_hint",
                        "The model described a smooth curve but used line_to, which draws a straight segment.",
                    )
                )
        return warnings

    def _runtime_description_for_tool(
        self,
        *,
        tool_call: dict[str, Any],
        executed_tool_call: dict[str, Any],
        before_current_point: tuple[float, float] | None,
    ) -> str:
        tool = str(tool_call["tool"])
        if tool == "start_path":
            return f"Started a new path at [{tool_call['x']:.2f}, {tool_call['y']:.2f}]."
        if tool == "line_to":
            if before_current_point is None:
                return f"Drew a straight line to [{tool_call['x']:.2f}, {tool_call['y']:.2f}]."
            return (
                f"Drew a straight line from [{before_current_point[0]:.2f},{before_current_point[1]:.2f}] "
                f"to [{tool_call['x']:.2f},{tool_call['y']:.2f}]."
            )
        if tool == "curve_to":
            start = before_current_point if before_current_point is not None else (tool_call["p"][0], tool_call["p"][1])
            return (
                f"Drew a cubic curve from [{start[0]:.2f},{start[1]:.2f}] to [{tool_call['p'][0]:.2f},{tool_call['p'][1]:.2f}] "
                f"using c1=[{tool_call['c1'][0]:.2f},{tool_call['c1'][1]:.2f}] and c2=[{tool_call['c2'][0]:.2f},{tool_call['c2'][1]:.2f}]."
            )
        if tool == "move_anchor":
            return f"Moved anchor {tool_call['anchor_id']}."
        if tool == "move_handle":
            return f"Moved handle {tool_call['handle']} of {tool_call['segment_id']}."
        if tool == "set_segment_handles":
            return f"Set both handles of {tool_call['segment_id']}."
        if tool == "convert_line_to_curve":
            return f"Converted line segment {tool_call['segment_id']} into an editable cubic segment."
        if tool == "close_path":
            return "Closed the current path by drawing a straight chord from the current point back to the path start."
        return f"Executed {tool}."

    def _recent_history_summary(self, history: list[dict[str, Any]], *, last_n: int | None = None) -> list[str]:
        effective_last_n = max(1, min(20, int(last_n if last_n is not None else self.history_summary_steps)))
        lines: list[str] = []
        for entry in history[-effective_last_n:]:
            prefix = f"{entry['step']}. "
            description = str(entry.get("runtime_description") or "").strip()
            quality_summary = str(entry.get("quality_summary") or "").strip()
            if quality_summary:
                lines.append(f"{prefix}{description} {quality_summary}")
            else:
                lines.append(f"{prefix}{description}")
        return lines

    def _feedback_from_executed_action(
        self,
        *,
        warnings: list[dict[str, Any]],
        quality_summary: str,
        canvas: FreePenCanvasState,
    ) -> list[str]:
        feedback: list[str] = []
        if warnings:
            feedback.extend(warning["message"] for warning in warnings)
        else:
            feedback.append(quality_summary)
        if canvas.path_open:
            feedback.append("The current path is open.")
        return feedback

    @staticmethod
    def _feedback_from_rejected_action(*, warnings: list[dict[str, Any]], quality_summary: str) -> list[str]:
        feedback = [warning["message"] for warning in warnings]
        if any(warning["code"] == "line_to_used_on_smooth_curve_hint" for warning in warnings):
            feedback.append("You described a smooth curve but used line_to.")
            feedback.append("line_to draws a straight segment and is rejected for this smooth curve.")
            feedback.append("Use curve_to with c1, c2, and p.")
        if any(warning["code"] == "current_segment_needs_refinement" for warning in warnings):
            feedback.append("Do not draw the next segment yet.")
            feedback.append("Refine the current segment first using set_segment_handles, move_handle, move_anchor, or convert_line_to_curve if it is a line segment.")
        if any(warning["code"] == "out_of_bounds_coordinate" for warning in warnings):
            feedback.append("The previous tool call was rejected because one or more coordinates were outside the canvas bounds.")
            feedback.append("Do not continue to the next segment. Retry the same segment with in-bounds coordinates.")
        if not feedback and quality_summary:
            feedback.append(quality_summary)
        return feedback

    def _reason_based_warnings(self, *, ai_reason: str, tool: str) -> list[dict[str, Any]]:
        reason_lower = ai_reason.lower()
        warnings: list[dict[str, Any]] = []
        if tool == "line_to" and any(token in reason_lower for token in self._SMOOTH_REASON_HINTS):
            warnings.append(
                self._warning(
                    "line_to_used_on_smooth_curve_hint",
                    "The model described a smooth curve but used line_to, which draws a straight segment. Use curve_to instead.",
                )
            )
        if any(token in reason_lower for token in self._OVERLAY_CONFUSION_HINTS):
            warnings.append(
                self._warning(
                    "overlay_target_confusion_hint",
                    "The overlay is your previous drawing, not the target. Trace the black source stroke, not the overlay.",
                )
            )
        return warnings

    def _coordinate_warnings(
        self,
        *,
        x: Any,
        y: Any,
        width: int,
        height: int,
        point_name: str | None = None,
    ) -> list[dict[str, Any]]:
        label = point_name or "point"
        warnings: list[dict[str, Any]] = []
        numeric_x = self._finite_number_or_none(x)
        numeric_y = self._finite_number_or_none(y)
        if numeric_x is None or numeric_y is None:
            warnings.append(self._warning("non_finite_coordinate", f"{label} contains NaN, Infinity, or another non-finite coordinate."))
            return warnings
        if numeric_x < 0.0 or numeric_x >= float(width) or numeric_y < 0.0 or numeric_y >= float(height):
            warnings.append(self._warning("out_of_bounds_coordinate", f"{label} lies outside the canvas bounds."))
        return warnings

    @staticmethod
    def _finite_number_or_none(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    @staticmethod
    def _warning(code: str, message: str) -> dict[str, Any]:
        return {"code": code, "message": message}

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        return "timeout" in name or "timed out" in message or isinstance(exc, TimeoutError)

    def _build_public_image_resolver(self) -> PublicImageResolver | None:
        if self.image_transport_config.mode != "url" or not self.image_transport_config.public_image_base_url:
            return None
        return PublicImageResolver(
            project_root=self.image_transport_config.public_image_root,
            base_url=self.image_transport_config.public_image_base_url,
        )
    
    def _build_post_tool_visual_feedback_message_content(
        self,
        *,
        overlay_image_path: Path | None,
        composite_image_path: Path | None,
        resolver: PublicImageResolver | None,
        step_index: int,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        header = (
            "Visual feedback after your previous tool call.\n"
            "BLACK = source target.\n"
            "ORANGE = your drawing.\n"
            "BLUE = anchors.\n"
            "GREEN = control handles.\n"
            "Inspect the newest segment before choosing exactly one next tool.\n"
            "If the newest segment is misaligned, adjust handles or anchors before drawing the next segment."
        )

        content.append({"type": "text", "text": header})

        if overlay_image_path is not None:
            content.extend(
                self._build_image_parts(
                    semantic_text=(
                        "Overlay image after the previous tool call. "
                        "It shows your current drawing only; it is not the target."
                    ),
                    image_path=overlay_image_path,
                    resolver=resolver,
                )
            )

        if composite_image_path is not None:
            content.extend(
                self._build_image_parts(
                    semantic_text=(
                        "Composite image after the previous tool call. "
                        "BLACK is the source target contour. ORANGE is your current drawing. "
                        "Compare them carefully before deciding the next tool call."
                    ),
                    image_path=composite_image_path,
                    resolver=resolver,
                )
            )

        return content

    def _build_round_image_message_content(
        self,
        *,
        source_image_path: Path,
        overlay_image_path: Path | None,
        composite_image_path: Path | None,
        resolver: PublicImageResolver | None,
        include_source_image: bool,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if include_source_image:
            content.extend(
                self._build_image_parts(
                    semantic_text="This is the target source image. Trace the single black contour in this image.",
                    image_path=source_image_path,
                    resolver=resolver,
                )
            )
        if overlay_image_path is not None:
            content.extend(
                self._build_image_parts(
                    semantic_text="This is the overlay after the previous accepted tool call. It is your previous drawing, not the target.",
                    image_path=overlay_image_path,
                    resolver=resolver,
                )
            )
        if composite_image_path is not None:
            content.extend(
                self._build_image_parts(
                    semantic_text=(
                        "Current composite preview. "
                        "BLACK = source target. ORANGE = your drawing. BLUE = anchors. GREEN = control handles. "
                        "Inspect the newest segment before choosing exactly one next tool. "
                        "If the newest segment is misaligned, adjust handles or anchors before drawing the next segment."
                    ),
                    image_path=composite_image_path,
                    resolver=resolver,
                )
            )
        return content

    def _build_editable_geometry(self, canvas: FreePenCanvasState) -> dict[str, Any]:
        geometry = canvas.editable_geometry()
        for path in geometry["paths"]:
            for segment in path["segments"]:
                if segment["type"] == "line":
                    segment["editable_with_handles"] = False
                    segment["conversion_tool"] = "convert_line_to_curve"
                elif segment["type"] == "cubic":
                    segment["editable_with_handles"] = True
        return geometry

    def _build_source_distance_map(self, source_image: np.ndarray) -> np.ndarray | None:
        try:
            if source_image.ndim == 2:
                gray = source_image
            elif source_image.shape[2] == 4:
                gray = cv2.cvtColor(source_image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            source_mask = (gray < 128).astype(np.uint8)
            if int(np.count_nonzero(source_mask)) == 0:
                return None
            distance_input = np.where(source_mask > 0, 0, 255).astype(np.uint8)
            return cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
        except Exception:
            return None

    def _build_current_segment_context(
        self,
        *,
        canvas: FreePenCanvasState,
        source_distance_map: np.ndarray | None,
        focus_tool_call: dict[str, Any] | None,
    ) -> dict[str, Any]:
        editable_geometry = self._build_editable_geometry(canvas)
        focus = self._determine_current_segment_focus(
            editable_geometry=editable_geometry,
            focus_tool_call=focus_tool_call,
        )
        quality_metrics = self._build_quality_metrics(
            canvas=canvas,
            editable_geometry=editable_geometry,
            source_distance_map=source_distance_map,
            focus=focus,
        )
        status = self._build_current_segment_status(focus=focus, metrics=quality_metrics)
        return {
            "editable_geometry": editable_geometry,
            "focus": focus,
            "status": status,
            "quality_metrics": quality_metrics,
        }

    def _build_image_parts(
        self,
        *,
        semantic_text: str,
        image_path: Path,
        resolver: PublicImageResolver | None,
    ) -> list[dict[str, Any]]:
        image_url = self._image_url_for_request(image_path=image_path, resolver=resolver)
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "auto",
                    "mime_type": mime_type,
                },
            },
            {"type": "text", "text": semantic_text},
        ]

    def _image_url_for_request(self, *, image_path: Path, resolver: PublicImageResolver | None) -> str:
        if self.image_transport_config.mode == "url":
            if resolver is None:
                raise ValueError("URL image transport requires a configured PublicImageResolver")
            return resolver.to_public_url(image_path)
        return encode_image_as_data_url(image_path, max_image_bytes=20 * 1024 * 1024)

    def _determine_current_segment_focus(
        self,
        *,
        editable_geometry: dict[str, Any],
        focus_tool_call: dict[str, Any] | None,
    ) -> dict[str, Any]:
        all_segments = [segment for path in editable_geometry["paths"] for segment in path["segments"]]
        if not all_segments:
            return {
                "segment_id": None,
                "type": None,
                "from_anchor": None,
                "to_anchor": None,
                "recommended_action": "draw_first_segment",
                "hint": "No drawable segment exists yet. Start the first local segment on the black contour.",
            }
        segment_lookup = {segment["id"]: segment for segment in all_segments}
        tool = str(focus_tool_call.get("tool")) if focus_tool_call else ""
        target_segment = all_segments[-1]
        if tool in {"move_handle", "set_segment_handles", "convert_line_to_curve"}:
            target_segment = segment_lookup.get(str(focus_tool_call.get("segment_id")), target_segment)
        elif tool == "move_anchor":
            anchor_id = str(focus_tool_call.get("anchor_id"))
            target_segment = next(
                (segment for segment in all_segments if segment["from_anchor"] == anchor_id or segment["to_anchor"] == anchor_id),
                target_segment,
            )
        if target_segment["type"] == "line":
            hint = (
                f"Inspect {target_segment['id']} first. If it needs local curvature or handle-based editing, use convert_line_to_curve before moving on."
            )
        else:
            hint = (
                f"Inspect {target_segment['id']} first. If it is misaligned, use set_segment_handles or move_handle before drawing the next segment."
            )
        return {
            "segment_id": target_segment["id"],
            "type": target_segment["type"],
            "from_anchor": target_segment["from_anchor"],
            "to_anchor": target_segment["to_anchor"],
            "recommended_action": "inspect_or_refine",
            "hint": hint,
        }

    def _build_quality_metrics(
        self,
        *,
        canvas: FreePenCanvasState,
        editable_geometry: dict[str, Any],
        source_distance_map: np.ndarray | None,
        focus: dict[str, Any],
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        if segment_id is None:
            return {"current_segment": {"segment_id": None, "unavailable_reason": "no_current_segment"}}
        if source_distance_map is None:
            return {"current_segment": {"segment_id": segment_id, "unavailable_reason": "source_mask_unavailable"}}
        segment_record = self._segment_record_from_canvas(
            canvas=canvas,
            editable_geometry=editable_geometry,
            segment_id=str(segment_id),
        )
        if segment_record is None:
            return {"current_segment": {"segment_id": segment_id, "unavailable_reason": "segment_not_found"}}
        sampled_points = self._sample_segment_points(segment_record=segment_record)
        if sampled_points.size == 0:
            return {"current_segment": {"segment_id": segment_id, "unavailable_reason": "empty_segment_samples"}}
        xs = np.clip(sampled_points[:, 0].round().astype(np.int32), 0, source_distance_map.shape[1] - 1)
        ys = np.clip(sampled_points[:, 1].round().astype(np.int32), 0, source_distance_map.shape[0] - 1)
        distances = source_distance_map[ys, xs].astype(np.float64)
        return {
            "current_segment": {
                "segment_id": segment_id,
                "path_to_source_mean_px": float(np.mean(distances)),
                "path_to_source_max_px": float(np.max(distances)),
                "path_to_source_p90_px": float(np.percentile(distances, 90)),
                "sample_count": int(len(distances)),
                "within_2px_ratio": float(np.mean(distances <= 2.0)),
                "within_4px_ratio": float(np.mean(distances <= 4.0)),
            }
        }

    def _build_current_segment_status(
        self,
        *,
        focus: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        metric_payload = metrics.get("current_segment", {})
        unavailable_reason = metric_payload.get("unavailable_reason")
        if segment_id is None:
            return {
                "segment_id": None,
                "status": "unknown",
                "reason": "No current drawable segment exists yet.",
                "recommended_next_tools": ["start_path", "curve_to"],
                "may_advance_to_next_segment": True,
            }
        if unavailable_reason:
            return {
                "segment_id": segment_id,
                "status": "unknown",
                "reason": str(unavailable_reason),
                "recommended_next_tools": ["inspect_history", "set_segment_handles", "move_handle", "move_anchor", "stalled"],
                "may_advance_to_next_segment": False,
            }
        p90 = float(metric_payload["path_to_source_p90_px"])
        mean = float(metric_payload["path_to_source_mean_px"])
        max_dist = float(metric_payload["path_to_source_max_px"])
        acceptable = p90 <= self._SEGMENT_ACCEPTABLE_P90_PX or (
            mean <= self._SEGMENT_ACCEPTABLE_MEAN_PX and max_dist <= self._SEGMENT_ACCEPTABLE_MAX_PX
        )
        if acceptable:
            return {
                "segment_id": segment_id,
                "status": "acceptable",
                "reason": "The newest segment is locally acceptable against the black source contour.",
                "recommended_next_tools": ["curve_to", "line_to", "close_path"],
                "may_advance_to_next_segment": True,
            }
        recommended = ["set_segment_handles", "move_handle", "move_anchor", "undo_last", "rollback_to_step", "inspect_history", "stalled"]
        if focus.get("type") == "line":
            recommended.insert(0, "convert_line_to_curve")
        return {
            "segment_id": segment_id,
            "status": "needs_refinement",
            "reason": "The newest segment is not acceptable yet and should be refined before advancing.",
            "recommended_next_tools": recommended,
            "may_advance_to_next_segment": False,
        }

    def _segment_record_from_canvas(
        self,
        *,
        canvas: FreePenCanvasState,
        editable_geometry: dict[str, Any],
        segment_id: str,
    ) -> dict[str, Any] | None:
        path = canvas.current_path() or (canvas.paths[-1] if canvas.paths else None)
        if path is None or not editable_geometry["paths"]:
            return None
        geometry_path = editable_geometry["paths"][0]
        anchors = {anchor["id"]: anchor["p"] for anchor in geometry_path["anchors"]}
        drawable_segments = [segment for segment in path.segments if segment["type"] in {"line", "cubic"}]
        for index, segment in enumerate(geometry_path["segments"]):
            if segment["id"] != segment_id:
                continue
            raw_segment = drawable_segments[index]
            return {
                "id": segment_id,
                "type": raw_segment["type"],
                "from_point": anchors.get(segment["from_anchor"]),
                "to_point": anchors.get(segment["to_anchor"]),
                "raw": raw_segment,
            }
        return None

    def _sample_segment_points(self, *, segment_record: dict[str, Any]) -> np.ndarray:
        from_point = segment_record.get("from_point")
        to_point = segment_record.get("to_point")
        raw_segment = segment_record["raw"]
        if not from_point or not to_point:
            return np.asarray([], dtype=np.float64)
        if raw_segment["type"] == "line":
            sample_count = max(8, int(self.sample_count_per_segment))
            p0 = np.array(from_point, dtype=np.float64)
            p1 = np.array(to_point, dtype=np.float64)
            samples = []
            for index in range(sample_count):
                t = index / float(sample_count - 1)
                point = ((1.0 - t) * p0) + (t * p1)
                samples.append(point.tolist())
            return np.asarray(samples, dtype=np.float64)
        return np.asarray(
            FreePenCanvasState._sample_cubic_segment(
                p0=(float(from_point[0]), float(from_point[1])),
                c1=(float(raw_segment["c1"][0]), float(raw_segment["c1"][1])),
                c2=(float(raw_segment["c2"][0]), float(raw_segment["c2"][1])),
                p1=(float(to_point[0]), float(to_point[1])),
                sample_count=max(8, int(self.sample_count_per_segment)),
            ),
            dtype=np.float64,
        )

    def _convert_line_to_curve(self, *, canvas: FreePenCanvasState, tool_call: dict[str, Any]) -> dict[str, Any]:
        path = canvas.current_path()
        if path is None:
            raise FreePenCanvasError("convert_line_to_curve requires an open path")
        editable_geometry = self._build_editable_geometry(canvas)
        geometry_path = editable_geometry["paths"][0] if editable_geometry["paths"] else {"segments": []}
        drawable_segments = [segment for segment in path.segments if segment["type"] in {"line", "cubic"}]
        for index, segment in enumerate(geometry_path["segments"]):
            if segment["id"] != str(tool_call["segment_id"]):
                continue
            raw_segment = drawable_segments[index]
            if raw_segment["type"] != "line":
                raise FreePenCanvasError(f"{tool_call['segment_id']} is not a line segment")
            endpoint = [float(raw_segment["p"][0]), float(raw_segment["p"][1])]
            raw_segment["type"] = "cubic"
            raw_segment["c1"] = [float(tool_call["c1"][0]), float(tool_call["c1"][1])]
            raw_segment["c2"] = [float(tool_call["c2"][0]), float(tool_call["c2"][1])]
            return {
                "tool": "convert_line_to_curve",
                "path_id": path.path_id,
                "segment_id": str(tool_call["segment_id"]),
                "p": endpoint,
                "c1": [float(tool_call["c1"][0]), float(tool_call["c1"][1])],
                "c2": [float(tool_call["c2"][0]), float(tool_call["c2"][1])],
            }
        raise FreePenCanvasError(f"unknown segment_id: {tool_call['segment_id']}")

    def _replay_successful_tool_calls(self, *, canvas: FreePenCanvasState, tool_calls: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
        canvas.clear()
        canvas.step_count = 0
        canvas.successful_step_count = 0
        canvas.invalid_step_count = 0
        for tool_call in tool_calls:
            if str(tool_call.get("tool")) == "convert_line_to_curve":
                self._convert_line_to_curve(canvas=canvas, tool_call=tool_call)
                canvas.successful_step_count += 1
            else:
                canvas.apply_tool_call(tool_call, count_step=False)

    def _build_session_state(
        self,
        *,
        canvas: FreePenCanvasState,
        history: list[dict[str, Any]],
        current_feedback: list[str],
        current_segment_context: dict[str, Any],
    ) -> dict[str, Any]:
        closed_path_count = sum(1 for path in canvas.paths if path.closed)
        last_action = str(history[-1]["runtime_description"]) if history else "none"
        current_segment_focus = current_segment_context["focus"]
        current_segment_status = current_segment_context["status"]
        if canvas.path_open:
            near_start = canvas.distance_to_start()
            if current_segment_status.get("may_advance_to_next_segment") is False and current_segment_focus.get("segment_id"):
                current_goal = "Refine the newest segment before drawing the next segment."
                allowed_next_actions = tuple(current_segment_status.get("recommended_next_tools") or ())
                forbidden_next_actions = ("curve_to", "line_to", "close_path", "finish")
            elif (
                near_start is not None
                and near_start <= max(8.0, canvas.current_path_bbox_diagonal() * 0.10)
                and canvas.current_path_drawable_segment_count() >= 2
            ):
                current_goal = "The path has returned to the start point. Inspect the final visual feedback. If the orange path matches the black contour, call close_path as the next single tool call."
                allowed_next_actions = ("close_path", "set_segment_handles", "move_handle", "move_anchor", "convert_line_to_curve", "undo_last", "rollback_to_step", "inspect_history", "stalled")
                forbidden_next_actions = ("finish", "start_path", "curve_to", "line_to")
            else:
                current_goal = "Continue tracing the current open target contour. Prefer curve_to over line_to unless the source contour is clearly straight."
                allowed_next_actions = ("curve_to", "line_to", "close_path", "undo_last", "rollback_to_step", "inspect_history", "restart_path", "move_anchor", "move_handle", "set_segment_handles", "convert_line_to_curve")
                forbidden_next_actions = ("finish", "start_path")
        elif closed_path_count >= 1:
            current_goal = "Return finish if the closed path matches the source; otherwise use rollback_to_step or restart_path."
            allowed_next_actions = ("finish", "rollback_to_step", "restart_path", "inspect_history", "undo_last")
            forbidden_next_actions = ("start_path", "line_to", "curve_to")
        else:
            current_goal = "Start tracing the single black target contour."
            allowed_next_actions = ("start_path", "inspect_history", "stalled")
            forbidden_next_actions = ("line_to", "curve_to", "close_path", "finish")
        return {
            "task": "continue tracing the same single black target contour",
            "mode": "single_contour_pen_tracing",
            "path_count": len(canvas.paths),
            "closed_path_count": closed_path_count,
            "path_open": canvas.path_open,
            "current_point": None if canvas.current_point is None else [canvas.current_point[0], canvas.current_point[1]],
            "current_subpath_start": None
            if canvas.current_subpath_start is None
            else [canvas.current_subpath_start[0], canvas.current_subpath_start[1]],
            "last_action": last_action,
            "current_goal": current_goal,
            "allowed_next_actions": list(allowed_next_actions),
            "forbidden_next_actions": list(forbidden_next_actions),
            "current_feedback": list(current_feedback),
            "current_segment_focus": current_segment_focus,
            "current_segment_status": current_segment_status,
        }

    def _write_round_composite_context(
        self,
        *,
        output_dir: Path,
        step_index: int,
        canvas: FreePenCanvasState,
        source_image: np.ndarray,
    ) -> Path | None:
        if not canvas.paths:
            return None
        composite = canvas.render_composite(
            source_image,
            stroke_width=max(1, int(self.stroke_width)),
            stroke_rgba=self.stroke_rgba,
            sample_count_per_segment=max(8, int(self.sample_count_per_segment)),
        )
        composite_path = output_dir / f"round_{step_index:03d}_composite_context.png"
        cv2.imwrite(str(composite_path), composite)
        return composite_path

    def _write_request_snapshot(
        self,
        *,
        request_snapshot_path: Path,
        provider: str,
        model: str,
        image_transport: str,
        public_image_base_url: str | None,
        messages: list[dict[str, Any]],
        session_state: dict[str, Any],
        tool_mode: str = "native_tools",
        tools: tuple[dict[str, Any], ...] = (),
        tool_choice: str | None = None,
    ) -> None:
        sanitized_messages = self._sanitize_messages_for_snapshot(messages)
        payload = {
            "provider": provider,
            "model": model,
            "image_transport": image_transport,
            "public_image_base_url": public_image_base_url,
            "messages": sanitized_messages,
            "image_urls": self._collect_image_urls_from_messages(sanitized_messages),
            "session_state": session_state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_mode": tool_mode,
            "tools": list(tools),
        }
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        request_snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _sanitize_messages_for_snapshot(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized_messages: list[dict[str, Any]] = []
        for message in messages:
            sanitized_message: dict[str, Any] = {"role": message.get("role"), "content": message.get("content")}
            if "tool_calls" in message:
                sanitized_message["tool_calls"] = message.get("tool_calls")
            if "tool_call_id" in message:
                sanitized_message["tool_call_id"] = message.get("tool_call_id")
            content = sanitized_message["content"]
            if isinstance(content, list):
                sanitized_parts: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, dict):
                        sanitized_parts.append(part)
                        continue
                    if part.get("type") != "image_url":
                        sanitized_parts.append(dict(part))
                        continue
                    image_url = dict(part.get("image_url", {}))
                    if isinstance(image_url.get("url"), str):
                        image_url["url"] = self._sanitize_image_url(str(image_url["url"]))
                    sanitized_parts.append({"type": "image_url", "image_url": image_url})
                sanitized_message["content"] = sanitized_parts
            sanitized_messages.append(sanitized_message)
        return sanitized_messages

    @staticmethod
    def _sanitize_image_url(url: str) -> str:
        if url.startswith("data:"):
            header = url.split(",", 1)[0]
            return f"{header},<base64 data omitted>"
        return url

    def _collect_image_urls_from_messages(self, messages: list[dict[str, Any]]) -> list[str]:
        image_urls: list[str] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    image_urls.append(str(image_url["url"]))
        return image_urls

    def _next_hint(
        self,
        *,
        final_decision: str | None,
        round_status: str,
        warnings: list[dict[str, Any]],
        session_state: dict[str, Any],
        current_segment_context: dict[str, Any],
    ) -> str:
        if round_status == "rejected_action" and any(warning["code"] == "line_to_used_on_smooth_curve_hint" for warning in warnings):
            return "Use curve_to with c1, c2, and p."
        if round_status == "rejected_action" and any(warning["code"] == "current_segment_needs_refinement" for warning in warnings):
            return "The newest segment is not acceptable yet. Do not draw the next segment. Refine the current segment first using set_segment_handles, move_handle, move_anchor, or convert_line_to_curve if it is a line segment."
        if round_status == "rejected_action" and any(warning["code"] == "out_of_bounds_coordinate" for warning in warnings):
            return "The previous tool call was rejected because a coordinate was outside the canvas bounds. Do not continue to the next segment. Retry the same segment with in-bounds coordinates, or refine the latest existing segment with set_segment_handles."
        if round_status == "rejected_action":
            return "Revise the action instead of repeating the rejected call."
        if final_decision == "finish":
            return "Tracing is complete."
        if final_decision == "stalled":
            return "Stop the tracing loop."
        if current_segment_context["status"].get("may_advance_to_next_segment") is False and current_segment_context["focus"].get("segment_id"):
            return "The newest segment is not acceptable yet. Do not draw the next segment. Refine the current segment first."
        allowed = session_state.get("allowed_next_actions") or []
        if allowed:
            return f"Choose one of the allowed next actions: {', '.join(str(item) for item in allowed)}."
        return "Continue with the next step."

    def _record_interaction(self, payload: dict[str, Any]) -> None:
        if self.interaction_logger is not None:
            self.interaction_logger(payload)

    def _record_raw_response(self, step_index: int, raw_response: Any) -> None:
        if self.raw_response_logger is not None:
            self.raw_response_logger(step_index, raw_response)


__all__ = [
    "FileSequenceFreePenAdapter",
    "NativeToolCallSequenceAdapter",
    "FreePenReviewInput",
    "FreePenRunResult",
    "FreePenRuntime",
    "FreePenToolReviewInput",
    "FreePenToolRunResult",
    "FreePenToolRuntime",
    "SCHEMA_PATH",
    "TOOL_SCHEMA_PATH",
    "load_free_pen_schema",
    "load_free_pen_tool_schema",
    "normalize_free_pen_response",
    "normalize_free_pen_tool_response",
    "validate_free_pen_response",
    "validate_free_pen_tool_response",
]
