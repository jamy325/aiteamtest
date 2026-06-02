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
    source_contour_summary: dict[str, Any] | None = None
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
            source_contour_summary=self.source_contour_summary,
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
    if not any(
        isinstance(entry, dict)
        and isinstance(entry.get("properties"), dict)
        and entry["properties"].get("tool", {}).get("const") == "restore_best_segment"
        for entry in tool_one_of
    ):
        tool_one_of.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["tool", "segment_id"],
                "properties": {
                    "tool": {"const": "restore_best_segment"},
                    "segment_id": {"type": "string", "minLength": 1},
                },
            }
        )
    if not any(
        isinstance(entry, dict)
        and isinstance(entry.get("properties"), dict)
        and entry["properties"].get("tool", {}).get("const") == "request_segment_zoom"
        for entry in tool_one_of
    ):
        tool_one_of.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["tool", "segment_id", "zoom_scale", "padding_px"],
                "properties": {
                    "tool": {"const": "request_segment_zoom"},
                    "segment_id": {"type": "string", "minLength": 1},
                    "zoom_scale": {"type": "number", "minimum": 2, "maximum": 6},
                    "padding_px": {"type": "number", "minimum": 20, "maximum": 200},
                },
            }
        )
    if not any(
        isinstance(entry, dict)
        and isinstance(entry.get("properties"), dict)
        and entry["properties"].get("tool", {}).get("const") == "request_zoom_window"
        for entry in tool_one_of
    ):
        tool_one_of.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["tool", "x", "y", "width", "height", "zoom_scale"],
                "properties": {
                    "tool": {"const": "request_zoom_window"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number", "minimum": 80, "maximum": 600},
                    "height": {"type": "number", "minimum": 80, "maximum": 600},
                    "zoom_scale": {"type": "number", "minimum": 2, "maximum": 6},
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
    if tool == "restore_best_segment":
        normalized["segment_id"] = str(normalized["segment_id"])
        return normalized
    if tool == "request_segment_zoom":
        normalized["segment_id"] = str(normalized["segment_id"])
        normalized["zoom_scale"] = float(normalized["zoom_scale"])
        normalized["padding_px"] = float(normalized["padding_px"])
        return normalized
    if tool == "request_zoom_window":
        normalized["x"] = float(normalized["x"])
        normalized["y"] = float(normalized["y"])
        normalized["width"] = float(normalized["width"])
        normalized["height"] = float(normalized["height"])
        normalized["zoom_scale"] = float(normalized["zoom_scale"])
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
    _START_ANCHOR_TOLERANCE_PX = 10.0
    _ENDPOINT_TOLERANCE_PX = 12.0
    _MOVE_ANCHOR_TOLERANCE_PX = 10.0
    _MAX_REFINES_PER_SEGMENT = 3
    _MAX_WORSE_STREAK = 2
    _QUALITY_DELTA_EPSILON_PX = 0.5
    _MIN_ZOOM_SCALE = 2.0
    _MAX_ZOOM_SCALE = 6.0
    _MIN_ZOOM_PADDING_PX = 20.0
    _MAX_ZOOM_PADDING_PX = 200.0
    _MIN_ZOOM_WINDOW_WIDTH_PX = 80.0
    _MIN_ZOOM_WINDOW_HEIGHT_PX = 80.0
    _MAX_ZOOM_WINDOW_WIDTH_PX = 600.0
    _MAX_ZOOM_WINDOW_HEIGHT_PX = 600.0
    _MIN_SEGMENT_ZOOM_WIDTH_PX = 180.0
    _MIN_SEGMENT_ZOOM_HEIGHT_PX = 140.0
    _MAX_SEGMENT_ZOOM_WIDTH_PX = 360.0
    _MAX_SEGMENT_ZOOM_HEIGHT_PX = 240.0
    _MIN_SEGMENT_ZOOM_PADDING_PX = 16.0
    _BASE_ZOOM_SAMPLE_COUNT = 128
    _MAX_ZOOM_SAMPLE_COUNT = 1024
    _ZOOM_SAMPLE_SPACING_PX = 2.0
    _MAX_REQUESTED_ZOOMS_PER_SEGMENT = 2
    _MAX_REQUESTED_ZOOMS_TOTAL = 8
    _ZOOM_EDITOR_LEFT_RULER_WIDTH = 56
    _ZOOM_EDITOR_TOP_RULER_HEIGHT = 36

    def run(self, source_image_path: Path, output_dir: Path) -> FreePenToolRunResult:
        import os

        # 1. 优先级：显式参数 > 环境变量
        output_dir.mkdir(parents=True, exist_ok=True)
        source_image = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise ValueError(f"failed to load source image: {source_image_path}")
        height, width = source_image.shape[:2]
        source_mask = self._build_source_mask(source_image)
        source_distance_map = self._build_source_distance_map(source_mask)
        source_contour_summary = self._build_source_contour_summary(
            source_mask=source_mask,
            width=width,
            height=height,
        )
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
        segment_refinement: dict[str, dict[str, Any]] = {}
        requested_zoom_total_count = 0
        requested_zoom_by_segment: dict[str, int] = {}
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
                segment_refinement=segment_refinement,
            )
            session_state = self._build_session_state(
                canvas=canvas,
                history=history,
                current_feedback=current_feedback,
                current_segment_context=current_segment_context,
                source_contour_summary=source_contour_summary,
                requested_zoom_total_count=requested_zoom_total_count,
                requested_zoom_by_segment=requested_zoom_by_segment,
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
                source_contour_summary=session_state.get("source_contour_summary"),
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
            requested_zoom_windows: list[dict[str, Any]] = []
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
                        source_distance_map=source_distance_map,
                        segment_refinement=segment_refinement,
                        requested_zoom_total_count=requested_zoom_total_count,
                        requested_zoom_by_segment=requested_zoom_by_segment,
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
                            segment_refinement=segment_refinement,
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
                            execution_accepted,
                        ) = self._execute_tool_call(
                            tool_call=tool_call,
                            ai_reason=final_reason or "",
                            canvas=canvas,
                            successful_drawing_tool_calls=successful_drawing_tool_calls,
                            history=history,
                            segment_refinement=segment_refinement,
                        )
                        if not execution_accepted:
                            rejected_step_count += 1
                            rejected_tool_call = tool_call
                            current_segment_context = self._build_current_segment_context(
                                canvas=canvas,
                                source_distance_map=source_distance_map,
                                focus_tool_call=tool_call,
                                segment_refinement=segment_refinement,
                            )
                            round_status = "rejected_action"
                            execution_result = {
                                "ok": False,
                                "accepted": False,
                                "runtime_description": runtime_description,
                                "quality_summary": quality_summary,
                                "warnings": warnings,
                            }
                        else:
                            rollback_count += rollback_applied
                            if tool_call["tool"] in {"undo_last", "rollback_to_step", "restart_path"}:
                                segment_refinement = self._prune_segment_refinement_state(
                                    segment_refinement=segment_refinement,
                                    canvas=canvas,
                                )
                            current_segment_context = self._build_current_segment_context(
                                canvas=canvas,
                                source_distance_map=source_distance_map,
                                focus_tool_call=tool_call,
                                segment_refinement=segment_refinement,
                            )
                            segment_refinement = self._update_segment_refinement_state(
                                segment_refinement=segment_refinement,
                                tool_call=tool_call,
                                current_segment_context=current_segment_context,
                                canvas=canvas,
                                successful_step_count=len(successful_drawing_tool_calls),
                            )
                            current_segment_context = self._build_current_segment_context(
                                canvas=canvas,
                                source_distance_map=source_distance_map,
                                focus_tool_call=tool_call,
                                segment_refinement=segment_refinement,
                            )
                            round_status = "tool_applied"
                            execution_result = {
                                "ok": True,
                                "accepted": True,
                                "runtime_description": runtime_description,
                                "quality_summary": quality_summary,
                                "warnings": warnings,
                            }
                            if tool_call["tool"] in {"request_segment_zoom", "request_zoom_window"}:
                                execution_result["inspection_only"] = True
                                execution_result["state_changed"] = False
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

                if (
                    validation_error is None
                    and final_decision == "tool_call"
                    and round_status == "tool_applied"
                    and str(normalized_response.get("tool_call", {}).get("tool")) in {"request_segment_zoom", "request_zoom_window"}
                ):
                    requested_zoom_windows = self._build_requested_zoom_windows(
                        output_dir=output_dir,
                        step_index=step_index,
                        tool_call=dict(normalized_response["tool_call"]),
                        canvas=canvas,
                        current_segment_context=current_segment_context,
                        source_image=source_image,
                    )
                    if requested_zoom_windows:
                        requested_zoom_total_count += 1
                        for requested_zoom in requested_zoom_windows:
                            segment_id = requested_zoom.get("segment_id")
                            if segment_id is not None:
                                requested_zoom_by_segment[str(segment_id)] = int(requested_zoom_by_segment.get(str(segment_id), 0)) + 1
                        execution_result["inspection_only"] = True
                        execution_result["state_changed"] = False
                        execution_result["visual_feedback_metadata"] = {
                            "requested_zoom_windows": requested_zoom_windows,
                        }
                
                if raw_tool_calls:
                    tool_call_id = raw_tool_calls[0].get("id")
                    if tool_call_id:
                        session_state_after = self._build_session_state(
                            canvas=canvas,
                            history=history,
                            current_feedback=current_feedback,
                            current_segment_context=current_segment_context,
                            source_contour_summary=source_contour_summary,
                            requested_zoom_total_count=requested_zoom_total_count,
                            requested_zoom_by_segment=requested_zoom_by_segment,
                        )
                        next_hint_text = self._next_hint(
                            final_decision=final_decision,
                            round_status=round_status,
                            warnings=warnings,
                            session_state=session_state_after,
                            current_segment_context=current_segment_context,
                        )
                        if (
                            final_decision == "tool_call"
                            and isinstance(normalized_response, dict)
                            and str(normalized_response.get("tool_call", {}).get("tool")) in {"request_segment_zoom", "request_zoom_window"}
                            and round_status == "tool_applied"
                        ):
                            next_hint_text = "Inspect the requested zoom image before choosing exactly one next drawing or editing tool."
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
                            "anchor_quality": current_segment_context["anchor_quality"],
                            "segment_split_hint": current_segment_context["segment_split_hint"],
                            "quality_delta": current_segment_context["quality_delta"],
                            "refinement_summary": current_segment_context["refinement_summary"],
                            "best_candidate_hint": current_segment_context["best_candidate_hint"],
                            "visual_feedback_metadata": {"requested_zoom_windows": requested_zoom_windows},
                            "allowed_next_actions": session_state_after["allowed_next_actions"],
                            "forbidden_next_actions": session_state_after["forbidden_next_actions"],
                            "next_hint": next_hint_text,
                            "visual_feedback_hint": (
                                "A visual feedback user message with overlay/composite images will follow this tool result. "
                                "If a requested zoom image is provided, inspect it before choosing the next tool. "
                                "BLACK is target; ORANGE is your drawing; BLUE are anchors; GREEN are control handles."
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
                    requested_zoom_windows=requested_zoom_windows,
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
                "anchor_quality": current_segment_context["anchor_quality"],
                "segment_split_hint": current_segment_context["segment_split_hint"],
                "quality_delta": current_segment_context["quality_delta"],
                "refinement_summary": current_segment_context["refinement_summary"],
                    "best_candidate_hint": current_segment_context["best_candidate_hint"],
                    "reason": final_reason,
                    "output_overlay_path": str(overlay_path),
                    "requested_zoom_windows": requested_zoom_windows,
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
                    "requested_zoom_windows": requested_zoom_windows,
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
                    "segment_refinement": segment_refinement,
                    "requested_zoom_total_count": requested_zoom_total_count,
                    "requested_zoom_by_segment": requested_zoom_by_segment,
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

    def _build_requested_zoom_windows(
        self,
        *,
        output_dir: Path,
        step_index: int,
        tool_call: dict[str, Any],
        canvas: FreePenCanvasState,
        current_segment_context: dict[str, Any],
        source_image: np.ndarray,
    ) -> list[dict[str, Any]]:
        tool = str(tool_call.get("tool"))
        if tool == "request_segment_zoom":
            metadata = self._build_requested_segment_zoom_metadata(
                tool_call=tool_call,
                canvas=canvas,
                current_segment_context=current_segment_context,
                source_image=source_image,
                output_dir=output_dir,
                step_index=step_index,
            )
            return [] if metadata is None else [metadata]
        if tool == "request_zoom_window":
            metadata = self._build_requested_window_zoom_metadata(
                tool_call=tool_call,
                canvas=canvas,
                current_segment_context=current_segment_context,
                source_image=source_image,
                output_dir=output_dir,
                step_index=step_index,
            )
            return [] if metadata is None else [metadata]
        return []

    def _build_requested_segment_zoom_metadata(
        self,
        *,
        tool_call: dict[str, Any],
        canvas: FreePenCanvasState,
        current_segment_context: dict[str, Any],
        source_image: np.ndarray,
        output_dir: Path,
        step_index: int,
    ) -> dict[str, Any] | None:
        segment_id = str(tool_call["segment_id"])
        editable_geometry = current_segment_context["editable_geometry"]
        segment_record = self._segment_record_from_canvas(
            canvas=canvas,
            editable_geometry=editable_geometry,
            segment_id=segment_id,
        )
        if segment_record is None:
            return None
        zoom_scale = float(tool_call["zoom_scale"])
        sampled_points, sampling_info = self._build_zoom_segment_samples(
            segment_record=segment_record,
            zoom_scale=zoom_scale,
        )
        zoom_window = self._build_requested_segment_zoom_window(
            segment_record=segment_record,
            sampled_points=sampled_points,
            padding_px=float(tool_call["padding_px"]),
            canvas_width=canvas.width,
            canvas_height=canvas.height,
        )
        if zoom_window is None:
            return None
        crop_origin = zoom_window["crop_origin"]
        crop_size = zoom_window["crop_size"]
        output_path = output_dir / f"round_{step_index:03d}_requested_segment_zoom_{segment_id}.png"
        self._write_zoom_editor_view(
            source_image=source_image,
            canvas=canvas,
            editable_geometry=editable_geometry,
            crop_origin=crop_origin,
            crop_size=crop_size,
            zoom_scale=zoom_scale,
            output_path=output_path,
            highlighted_segment_id=segment_id,
            segment_sampling_overrides={segment_id: sampled_points},
        )
        return {
            "type": "requested_segment_zoom",
            "segment_id": segment_id,
            "path": str(output_path),
            "crop_origin": [int(crop_origin[0]), int(crop_origin[1])],
            "crop_size": [int(crop_size[0]), int(crop_size[1])],
            "zoom_scale": zoom_scale,
            "requested_zoom_scale": zoom_scale,
            "effective_zoom_scale": zoom_scale,
            "coordinate_space": "original_image_px",
            "render_mode": "zoom_editor_view",
            "highlighted_segment_id": segment_id,
            "required_bbox": zoom_window["required_bbox"],
            "anchors_visible": bool(zoom_window["anchors_visible"]),
            "curve_visible": bool(zoom_window["curve_visible"]),
            "handles_visible": bool(zoom_window["handles_visible"]),
            "clipped": bool(zoom_window["clipped"]),
            "clamped": bool(zoom_window["clamped"]),
            "ruler": {
                "top": True,
                "left": True,
                "labels_are_original_coordinates": True,
                "minor_tick_step_px": 10,
                "major_tick_step_px": 50,
            },
            "grid": {
                "minor_step_px": 10,
                "major_step_px": 50,
                "labels_are_original_coordinates": True,
            },
            "sampling": sampling_info,
        }

    def _build_requested_window_zoom_metadata(
        self,
        *,
        tool_call: dict[str, Any],
        canvas: FreePenCanvasState,
        current_segment_context: dict[str, Any],
        source_image: np.ndarray,
        output_dir: Path,
        step_index: int,
    ) -> dict[str, Any] | None:
        crop_origin, crop_size, clipped, clamped = self._normalize_zoom_crop_window(
            x=float(tool_call["x"]),
            y=float(tool_call["y"]),
            width=float(tool_call["width"]),
            height=float(tool_call["height"]),
            canvas_width=int(canvas.width),
            canvas_height=int(canvas.height),
            min_width=self._MIN_ZOOM_WINDOW_WIDTH_PX,
            min_height=self._MIN_ZOOM_WINDOW_HEIGHT_PX,
            max_width=self._MAX_ZOOM_WINDOW_WIDTH_PX,
            max_height=self._MAX_ZOOM_WINDOW_HEIGHT_PX,
        )
        output_path = output_dir / f"round_{step_index:03d}_requested_zoom_window_001.png"
        highlighted_segment_id = current_segment_context.get("focus", {}).get("segment_id")
        self._write_zoom_editor_view(
            source_image=source_image,
            canvas=canvas,
            editable_geometry=current_segment_context["editable_geometry"],
            crop_origin=crop_origin,
            crop_size=crop_size,
            zoom_scale=float(tool_call["zoom_scale"]),
            output_path=output_path,
            highlighted_segment_id=str(highlighted_segment_id) if highlighted_segment_id is not None else None,
        )
        return {
            "type": "requested_zoom_window",
            "path": str(output_path),
            "crop_origin": [int(crop_origin[0]), int(crop_origin[1])],
            "crop_size": [int(crop_size[0]), int(crop_size[1])],
            "zoom_scale": float(tool_call["zoom_scale"]),
            "coordinate_space": "original_image_px",
            "render_mode": "zoom_editor_view",
            "highlighted_segment_id": highlighted_segment_id,
            "clipped": bool(clipped),
            "clamped": bool(clamped),
            "ruler": {
                "top": True,
                "left": True,
                "labels_are_original_coordinates": True,
                "minor_tick_step_px": 10,
                "major_tick_step_px": 50,
            },
            "grid": {
                "minor_step_px": 10,
                "major_step_px": 50,
                "labels_are_original_coordinates": True,
            },
        }

    def _normalize_zoom_crop_window(
        self,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        canvas_width: int,
        canvas_height: int,
        min_width: float,
        min_height: float,
        max_width: float,
        max_height: float,
    ) -> tuple[tuple[int, int], tuple[int, int], bool, bool]:
        requested_x0 = float(x)
        requested_y0 = float(y)
        requested_x1 = float(x + width)
        requested_y1 = float(y + height)
        x0 = max(0.0, requested_x0)
        y0 = max(0.0, requested_y0)
        x1 = min(float(canvas_width), requested_x1)
        y1 = min(float(canvas_height), requested_y1)
        bounded_width = max(0.0, x1 - x0)
        bounded_height = max(0.0, y1 - y0)
        target_width = min(max(bounded_width, float(min_width)), float(max_width), float(canvas_width))
        target_height = min(max(bounded_height, float(min_height)), float(max_height), float(canvas_height))
        clamped = abs(target_width - bounded_width) >= 0.5 or abs(target_height - bounded_height) >= 0.5
        if x1 <= x0:
            center_x = min(max((requested_x0 + requested_x1) * 0.5, 0.0), float(canvas_width))
            half = max(1.0, min_width * 0.5)
            x0 = max(0.0, center_x - half)
            x1 = min(float(canvas_width), center_x + half)
        if y1 <= y0:
            center_y = min(max((requested_y0 + requested_y1) * 0.5, 0.0), float(canvas_height))
            half = max(1.0, min_height * 0.5)
            y0 = max(0.0, center_y - half)
            y1 = min(float(canvas_height), center_y + half)
        center_x = (x0 + x1) * 0.5
        center_y = (y0 + y1) * 0.5
        final_width = target_width
        final_height = target_height
        x0 = max(0.0, min(float(canvas_width) - final_width, center_x - (final_width * 0.5)))
        y0 = max(0.0, min(float(canvas_height) - final_height, center_y - (final_height * 0.5)))
        x1 = x0 + final_width
        y1 = y0 + final_height
        crop_origin = (int(round(x0)), int(round(y0)))
        crop_size = (
            max(1, min(int(round(final_width)), canvas_width - crop_origin[0])),
            max(1, min(int(round(final_height)), canvas_height - crop_origin[1])),
        )
        clipped = not (
            abs(crop_origin[0] - requested_x0) < 0.5
            and abs(crop_origin[1] - requested_y0) < 0.5
            and abs(crop_size[0] - width) < 0.5
            and abs(crop_size[1] - height) < 0.5
        )
        return crop_origin, crop_size, clipped, clamped

    def _write_zoom_editor_view(
        self,
        *,
        source_image: np.ndarray,
        canvas: FreePenCanvasState,
        editable_geometry: dict[str, Any],
        crop_origin: tuple[int, int],
        crop_size: tuple[int, int],
        zoom_scale: float,
        output_path: Path,
        highlighted_segment_id: str | None,
        segment_sampling_overrides: dict[str, np.ndarray] | None = None,
    ) -> None:
        x0, y0 = crop_origin
        width, height = crop_size
        source_bgr = self._to_bgr_image(source_image)
        source_crop = source_bgr[y0 : y0 + height, x0 : x0 + width]
        scaled_width = max(1, int(round(width * zoom_scale)))
        scaled_height = max(1, int(round(height * zoom_scale)))
        zoomed_source = cv2.resize(source_crop, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        left_ruler_width = self._ZOOM_EDITOR_LEFT_RULER_WIDTH
        top_ruler_height = self._ZOOM_EDITOR_TOP_RULER_HEIGHT
        image_origin_x = left_ruler_width
        image_origin_y = top_ruler_height
        editor = np.full((top_ruler_height + scaled_height, left_ruler_width + scaled_width, 3), 255, dtype=np.uint8)
        editor[image_origin_y : image_origin_y + scaled_height, image_origin_x : image_origin_x + scaled_width] = zoomed_source
        self._draw_zoom_editor_grid(
            editor=editor,
            image_origin=(image_origin_x, image_origin_y),
            image_size=(scaled_width, scaled_height),
            crop_origin=crop_origin,
            zoom_scale=zoom_scale,
        )
        self._draw_zoom_editor_geometry(
            editor=editor,
            canvas=canvas,
            editable_geometry=editable_geometry,
            crop_origin=crop_origin,
            zoom_scale=zoom_scale,
            image_origin=(image_origin_x, image_origin_y),
            highlighted_segment_id=highlighted_segment_id,
            segment_sampling_overrides=segment_sampling_overrides,
        )
        self._draw_zoom_editor_rulers(
            editor=editor,
            crop_origin=crop_origin,
            crop_size=crop_size,
            zoom_scale=zoom_scale,
            left_ruler_width=left_ruler_width,
            top_ruler_height=top_ruler_height,
        )
        self._draw_zoom_editor_overlay_info(
            editor=editor,
            crop_origin=crop_origin,
            crop_size=crop_size,
            zoom_scale=zoom_scale,
            image_origin=(image_origin_x, image_origin_y),
            highlighted_segment_id=highlighted_segment_id,
        )
        cv2.imwrite(str(output_path), editor)

    def _draw_zoom_editor_grid(
        self,
        *,
        editor: np.ndarray,
        image_origin: tuple[int, int],
        image_size: tuple[int, int],
        crop_origin: tuple[int, int],
        zoom_scale: float,
    ) -> None:
        image_origin_x, image_origin_y = image_origin
        image_width, image_height = image_size
        minor_step = max(1, int(round(10 * zoom_scale)))
        major_step = max(1, int(round(50 * zoom_scale)))
        x0, y0 = crop_origin
        minor_color = (236, 236, 236)
        major_color = (206, 206, 206)
        first_x = ((x0 + 9) // 10) * 10
        for original_x in range(first_x, x0 + int(round(image_width / zoom_scale)) + 1, 10):
            scaled_x = int(round((original_x - x0) * zoom_scale))
            color = major_color if original_x % 50 == 0 else minor_color
            x = image_origin_x + min(max(scaled_x, 0), image_width - 1)
            cv2.line(editor, (x, image_origin_y), (x, image_origin_y + image_height - 1), color, 1, cv2.LINE_AA)
        first_y = ((y0 + 9) // 10) * 10
        for original_y in range(first_y, y0 + int(round(image_height / zoom_scale)) + 1, 10):
            scaled_y = int(round((original_y - y0) * zoom_scale))
            color = major_color if original_y % 50 == 0 else minor_color
            y = image_origin_y + min(max(scaled_y, 0), image_height - 1)
            cv2.line(editor, (image_origin_x, y), (image_origin_x + image_width - 1, y), color, 1, cv2.LINE_AA)

    def _draw_zoom_editor_rulers(
        self,
        *,
        editor: np.ndarray,
        crop_origin: tuple[int, int],
        crop_size: tuple[int, int],
        zoom_scale: float,
        left_ruler_width: int,
        top_ruler_height: int,
    ) -> None:
        x0, y0 = crop_origin
        width, height = crop_size
        scaled_width = max(1, int(round(width * zoom_scale)))
        scaled_height = max(1, int(round(height * zoom_scale)))
        ruler_bg = (238, 238, 238)
        ruler_tick = (96, 96, 96)
        ruler_text = (56, 56, 56)
        border_color = (160, 160, 160)
        axis_font_scale = 0.56
        minor_step = 10
        major_step = 50
        cv2.rectangle(editor, (0, 0), (editor.shape[1] - 1, top_ruler_height - 1), ruler_bg, -1)
        cv2.rectangle(editor, (0, top_ruler_height), (left_ruler_width - 1, editor.shape[0] - 1), ruler_bg, -1)
        cv2.rectangle(editor, (0, 0), (left_ruler_width - 1, top_ruler_height - 1), ruler_bg, -1)
        cv2.line(editor, (left_ruler_width, 0), (left_ruler_width, editor.shape[0] - 1), border_color, 1, cv2.LINE_AA)
        cv2.line(editor, (left_ruler_width, top_ruler_height), (editor.shape[1] - 1, top_ruler_height), border_color, 1, cv2.LINE_AA)
        cv2.line(editor, (0, top_ruler_height - 1), (editor.shape[1] - 1, top_ruler_height - 1), border_color, 1, cv2.LINE_AA)
        cv2.line(editor, (left_ruler_width - 1, 0), (left_ruler_width - 1, editor.shape[0] - 1), border_color, 1, cv2.LINE_AA)
        first_x = ((x0 + minor_step - 1) // minor_step) * minor_step
        for original_x in range(first_x, x0 + width, minor_step):
            tick_x = left_ruler_width + int(round((original_x - x0) * zoom_scale))
            tick_len = 14 if original_x % major_step == 0 else 7
            cv2.line(editor, (tick_x, top_ruler_height - tick_len), (tick_x, top_ruler_height - 1), ruler_tick, 1, cv2.LINE_AA)
            if original_x % major_step == 0:
                label = str(original_x)
                (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, axis_font_scale, 1)
                text_x = max(left_ruler_width + 2, min(tick_x - text_width // 2, editor.shape[1] - text_width - 2))
                cv2.putText(editor, label, (text_x, 16), cv2.FONT_HERSHEY_SIMPLEX, axis_font_scale, ruler_text, 1, cv2.LINE_AA)
        first_y = ((y0 + minor_step - 1) // minor_step) * minor_step
        for original_y in range(first_y, y0 + height, minor_step):
            tick_y = top_ruler_height + int(round((original_y - y0) * zoom_scale))
            tick_len = 14 if original_y % major_step == 0 else 7
            cv2.line(editor, (left_ruler_width - tick_len, tick_y), (left_ruler_width - 1, tick_y), ruler_tick, 1, cv2.LINE_AA)
            if original_y % major_step == 0:
                label = str(original_y)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, axis_font_scale, 1)
                text_x = max(2, left_ruler_width - text_width - 4)
                text_y = max(top_ruler_height + text_height, min(tick_y + text_height // 2, editor.shape[0] - 4))
                cv2.putText(editor, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, axis_font_scale, ruler_text, 1, cv2.LINE_AA)

    def _draw_zoom_editor_overlay_info(
        self,
        *,
        editor: np.ndarray,
        crop_origin: tuple[int, int],
        crop_size: tuple[int, int],
        zoom_scale: float,
        image_origin: tuple[int, int],
        highlighted_segment_id: str | None,
    ) -> None:
        image_origin_x, image_origin_y = image_origin
        info_lines = [
            f"{highlighted_segment_id or 'Zoom view'}  {int(round(zoom_scale))}x",
            f"origin=({crop_origin[0]},{crop_origin[1]}) size={crop_size[0]}x{crop_size[1]} original image_px",
        ]
        padding = 8
        line_height = 15
        box_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for line in info_lines) + (padding * 2)
        box_height = len(info_lines) * line_height + padding
        x = image_origin_x + 8
        y = image_origin_y + 8
        cv2.rectangle(editor, (x, y), (x + box_width, y + box_height), (255, 255, 255), -1)
        cv2.rectangle(editor, (x, y), (x + box_width, y + box_height), (170, 170, 170), 1)
        for index, line in enumerate(info_lines):
            cv2.putText(
                editor,
                line,
                (x + padding, y + padding + 10 + (index * line_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

    def _draw_zoom_editor_geometry(
        self,
        *,
        editor: np.ndarray,
        canvas: FreePenCanvasState,
        editable_geometry: dict[str, Any],
        crop_origin: tuple[int, int],
        zoom_scale: float,
        image_origin: tuple[int, int],
        highlighted_segment_id: str | None,
        segment_sampling_overrides: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not editable_geometry["paths"]:
            return
        geometry_path = editable_geometry["paths"][0]
        segment_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for segment in geometry_path["segments"]:
            record = self._segment_record_from_canvas(
                canvas=canvas,
                editable_geometry=editable_geometry,
                segment_id=str(segment["id"]),
            )
            if record is not None:
                segment_payloads.append((segment, record))
        normal_segments = [payload for payload in segment_payloads if str(payload[0]["id"]) != str(highlighted_segment_id)]
        highlighted_segments = [payload for payload in segment_payloads if str(payload[0]["id"]) == str(highlighted_segment_id)]
        for segment, record in [*normal_segments, *highlighted_segments]:
            self._draw_zoom_editor_segment(
                editor=editor,
                segment=segment,
                segment_record=record,
                crop_origin=crop_origin,
                zoom_scale=zoom_scale,
                image_origin=image_origin,
                highlighted=str(segment["id"]) == str(highlighted_segment_id),
                sampled_override=(segment_sampling_overrides or {}).get(str(segment["id"])),
            )
        for anchor in geometry_path["anchors"]:
            self._draw_zoom_editor_anchor(
                editor=editor,
                anchor=anchor,
                crop_origin=crop_origin,
                zoom_scale=zoom_scale,
                image_origin=image_origin,
            )

    def _draw_zoom_editor_segment(
        self,
        *,
        editor: np.ndarray,
        segment: dict[str, Any],
        segment_record: dict[str, Any],
        crop_origin: tuple[int, int],
        zoom_scale: float,
        image_origin: tuple[int, int],
        highlighted: bool,
        sampled_override: np.ndarray | None = None,
    ) -> None:
        stroke_color = (0, 170, 255) if not highlighted else (0, 110, 255)
        stroke_width = 2 if not highlighted else 4
        sampled = sampled_override if sampled_override is not None else self._sample_segment_points(segment_record=segment_record)
        if sampled.size == 0:
            return
        display_points = np.asarray(
            [
                self._map_original_point_to_zoom(
                    point=(float(point[0]), float(point[1])),
                    crop_origin=crop_origin,
                    zoom_scale=zoom_scale,
                    image_origin=image_origin,
                )
                for point in sampled
            ],
            dtype=np.int32,
        )
        cv2.polylines(editor, [display_points], isClosed=False, color=stroke_color, thickness=stroke_width, lineType=cv2.LINE_AA)
        mid_point = display_points[len(display_points) // 2]
        self._draw_outlined_text(
            image=editor,
            text=str(segment["id"]),
            origin=(int(mid_point[0] + 6), int(mid_point[1] - 6)),
            font_scale=0.7 if highlighted else 0.58,
            fill_color=(18, 18, 18),
            outline_color=(250, 250, 250),
        )
        if segment_record["raw"]["type"] != "cubic":
            return
        from_point = self._map_original_point_to_zoom(
            point=(float(segment_record["from_point"][0]), float(segment_record["from_point"][1])),
            crop_origin=crop_origin,
            zoom_scale=zoom_scale,
            image_origin=image_origin,
        )
        to_point = self._map_original_point_to_zoom(
            point=(float(segment_record["to_point"][0]), float(segment_record["to_point"][1])),
            crop_origin=crop_origin,
            zoom_scale=zoom_scale,
            image_origin=image_origin,
        )
        c1_point = self._map_original_point_to_zoom(
            point=(float(segment_record["raw"]["c1"][0]), float(segment_record["raw"]["c1"][1])),
            crop_origin=crop_origin,
            zoom_scale=zoom_scale,
            image_origin=image_origin,
        )
        c2_point = self._map_original_point_to_zoom(
            point=(float(segment_record["raw"]["c2"][0]), float(segment_record["raw"]["c2"][1])),
            crop_origin=crop_origin,
            zoom_scale=zoom_scale,
            image_origin=image_origin,
        )
        handle_color = (0, 210, 0)
        handle_radius = 5 if highlighted else 4
        cv2.line(editor, from_point, c1_point, handle_color, 1, cv2.LINE_AA)
        cv2.line(editor, to_point, c2_point, handle_color, 1, cv2.LINE_AA)
        cv2.circle(editor, c1_point, handle_radius, handle_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(editor, c2_point, handle_radius, handle_color, thickness=-1, lineType=cv2.LINE_AA)
        if highlighted:
            self._draw_outlined_text(editor, "c1", (c1_point[0] + 5, c1_point[1] - 5), 0.5, (18, 18, 18), (250, 250, 250))
            self._draw_outlined_text(editor, "c2", (c2_point[0] + 5, c2_point[1] - 5), 0.5, (18, 18, 18), (250, 250, 250))

    def _draw_zoom_editor_anchor(
        self,
        *,
        editor: np.ndarray,
        anchor: dict[str, Any],
        crop_origin: tuple[int, int],
        zoom_scale: float,
        image_origin: tuple[int, int],
    ) -> None:
        point = self._map_original_point_to_zoom(
            point=(float(anchor["p"][0]), float(anchor["p"][1])),
            crop_origin=crop_origin,
            zoom_scale=zoom_scale,
            image_origin=image_origin,
        )
        cv2.circle(editor, point, 7, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        self._draw_outlined_text(
            image=editor,
            text=str(anchor["id"]),
            origin=(point[0] + 6, point[1] - 6),
            font_scale=0.72,
            fill_color=(18, 18, 18),
            outline_color=(250, 250, 250),
        )

    def _map_original_point_to_zoom(
        self,
        *,
        point: tuple[float, float],
        crop_origin: tuple[int, int],
        zoom_scale: float,
        image_origin: tuple[int, int],
    ) -> tuple[int, int]:
        image_origin_x, image_origin_y = image_origin
        x0, y0 = crop_origin
        return (
            int(round(image_origin_x + ((float(point[0]) - float(x0)) * float(zoom_scale)))),
            int(round(image_origin_y + ((float(point[1]) - float(y0)) * float(zoom_scale)))),
        )

    @staticmethod
    def _draw_outlined_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font_scale: float,
        fill_color: tuple[int, int, int],
        outline_color: tuple[int, int, int],
    ) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, fill_color, 1, cv2.LINE_AA)

    @staticmethod
    def _to_bgr_image(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = (image[:, :, :3].astype(np.float32) * alpha) + (255.0 * (1.0 - alpha))
            return np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        if image.shape[2] == 3:
            return image.copy()
        raise ValueError(f"unsupported source image shape: {image.shape}")
    
    def _preflight_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        ai_reason: str,
        canvas: FreePenCanvasState,
        successful_drawing_step_count: int,
        rollback_count: int,
        current_segment_context: dict[str, Any],
        source_distance_map: np.ndarray | None,
        segment_refinement: dict[str, dict[str, Any]] | None = None,
        requested_zoom_total_count: int = 0,
        requested_zoom_by_segment: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        tool = str(tool_call["tool"])
        warnings = self._reason_based_warnings(ai_reason=ai_reason, tool=tool)
        requested_zoom_by_segment = requested_zoom_by_segment or {}
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
        elif tool == "request_zoom_window":
            if self._finite_number_or_none(tool_call.get("x")) is None or self._finite_number_or_none(tool_call.get("y")) is None:
                warnings.append(
                    self._warning(
                        "non_finite_coordinate",
                        "zoom_window_origin contains NaN, Infinity, or another non-finite coordinate.",
                    )
                )
        reject_codes = {warning["code"] for warning in warnings if warning["code"] in {"non_finite_coordinate", "out_of_bounds_coordinate"}}
        if tool in {"line_to", "curve_to", "close_path", "move_anchor", "move_handle", "set_segment_handles", "convert_line_to_curve", "restore_best_segment"} and not canvas.path_open:
            reject_codes.add("path_not_open")
            warnings.append(self._warning("path_not_open", f"{tool} requires an open path started by start_path."))
        if tool == "start_path" and canvas.path_open:
            reject_codes.add("path_already_open")
            warnings.append(self._warning("path_already_open", "A path is already open. Use restart_path, rollback_to_step, or close_path first."))
        if tool == "line_to" and any(warning["code"] == "line_to_used_on_smooth_curve_hint" for warning in warnings):
            reject_codes.add("line_to_used_on_smooth_curve_hint")
        contour_distance_checks: list[tuple[str, float, float, float]] = []
        if tool in {"start_path", "restart_path"}:
            distance = self._distance_to_source_contour(
                source_distance_map=source_distance_map,
                x=tool_call["x"],
                y=tool_call["y"],
                width=canvas.width,
                height=canvas.height,
            )
            if distance is None:
                warnings.append(self._warning("source_contour_validation_unavailable", "Source contour distance validation is unavailable for the proposed start anchor."))
            else:
                contour_distance_checks.append(("anchor_start", float(tool_call["x"]), float(tool_call["y"]), distance))
                if distance > self._START_ANCHOR_TOLERANCE_PX:
                    reject_codes.add("anchor_not_on_source_contour")
                    warnings.append(
                        self._warning(
                            "anchor_not_on_source_contour",
                            f"{tool} point [{float(tool_call['x']):.2f},{float(tool_call['y']):.2f}] is {distance:.2f}px away from the black source contour. Choose a point on the black contour.",
                        )
                    )
        elif tool == "curve_to":
            distance = self._distance_to_source_contour(
                source_distance_map=source_distance_map,
                x=tool_call["p"][0],
                y=tool_call["p"][1],
                width=canvas.width,
                height=canvas.height,
            )
            if distance is None:
                warnings.append(self._warning("source_contour_validation_unavailable", "Source contour distance validation is unavailable for the proposed endpoint anchor."))
            else:
                contour_distance_checks.append(("endpoint", float(tool_call["p"][0]), float(tool_call["p"][1]), distance))
                if distance > self._ENDPOINT_TOLERANCE_PX:
                    reject_codes.add("endpoint_not_on_source_contour")
                    warnings.append(
                        self._warning(
                            "endpoint_not_on_source_contour",
                            f"curve_to endpoint p=[{float(tool_call['p'][0]):.2f},{float(tool_call['p'][1]):.2f}] is {distance:.2f}px away from the black source contour. The endpoint anchor must lie on or near the contour.",
                        )
                    )
        elif tool == "line_to":
            distance = self._distance_to_source_contour(
                source_distance_map=source_distance_map,
                x=tool_call["x"],
                y=tool_call["y"],
                width=canvas.width,
                height=canvas.height,
            )
            if distance is None:
                warnings.append(self._warning("source_contour_validation_unavailable", "Source contour distance validation is unavailable for the proposed endpoint anchor."))
            else:
                contour_distance_checks.append(("endpoint", float(tool_call["x"]), float(tool_call["y"]), distance))
                if distance > self._ENDPOINT_TOLERANCE_PX:
                    reject_codes.add("endpoint_not_on_source_contour")
                    warnings.append(
                        self._warning(
                            "endpoint_not_on_source_contour",
                            f"line_to endpoint [{float(tool_call['x']):.2f},{float(tool_call['y']):.2f}] is {distance:.2f}px away from the black source contour. The endpoint anchor must lie on or near the contour.",
                        )
                    )
        elif tool == "move_anchor":
            distance = self._distance_to_source_contour(
                source_distance_map=source_distance_map,
                x=tool_call["x"],
                y=tool_call["y"],
                width=canvas.width,
                height=canvas.height,
            )
            if distance is None:
                warnings.append(self._warning("source_contour_validation_unavailable", "Source contour distance validation is unavailable for the proposed anchor move."))
            else:
                contour_distance_checks.append(("anchor_move", float(tool_call["x"]), float(tool_call["y"]), distance))
                if distance > self._MOVE_ANCHOR_TOLERANCE_PX:
                    reject_codes.add("anchor_not_on_source_contour")
                    warnings.append(
                        self._warning(
                            "anchor_not_on_source_contour",
                            f"move_anchor target [{float(tool_call['x']):.2f},{float(tool_call['y']):.2f}] is {distance:.2f}px away from the black source contour. Move the anchor onto the black contour.",
                        )
                    )
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
        if (
            current_segment_context["status"].get("refinement_limit_reached")
            and tool in {"set_segment_handles", "move_handle"}
        ):
            reject_codes.add("refinement_limit_reached")
            warnings.append(
                self._warning(
                    "refinement_limit_reached",
                    "The current segment did not become acceptable after repeated handle edits. Change strategy instead of continuing handle edits.",
                )
            )
        if (
            current_segment_context["status"].get("status") == "needs_anchor_correction"
            and tool in {"set_segment_handles", "move_handle"}
        ):
            reject_codes.add("anchor_needs_correction")
            warnings.append(
                self._warning(
                    "anchor_needs_correction",
                    "One or more anchors are far from the black source contour. Do not adjust handles until the anchor is corrected.",
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
        if tool in {"request_segment_zoom", "request_zoom_window"} and requested_zoom_total_count >= self._MAX_REQUESTED_ZOOMS_TOTAL:
            reject_codes.add("zoom_budget_exceeded")
            warnings.append(
                self._warning(
                    "zoom_budget_exceeded",
                    "Zoom budget reached. Use the current visual feedback to choose a drawing or editing tool.",
                )
            )
        if tool == "move_anchor" and canvas.path_open:
            geometry = canvas.editable_geometry()
            anchors = {anchor["id"] for path in geometry["paths"] for anchor in path["anchors"]}
            if str(tool_call["anchor_id"]) not in anchors:
                reject_codes.add("unknown_anchor_id")
                warnings.append(self._warning("unknown_anchor_id", f"Unknown anchor_id: {tool_call['anchor_id']}"))
        editable_geometry = current_segment_context["editable_geometry"]
        if tool in {"move_handle", "set_segment_handles", "convert_line_to_curve", "restore_best_segment", "request_segment_zoom"}:
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
            elif tool == "restore_best_segment":
                best_geometry = self._get_best_segment_geometry(
                    segment_refinement=segment_refinement,
                    segment_id=segment_id,
                )
                if best_geometry is None:
                    reject_codes.add("best_segment_not_available")
                    warnings.append(self._warning("best_segment_not_available", f"No best segment geometry is available for {segment_id}.")) 
            elif tool == "request_segment_zoom":
                if int(requested_zoom_by_segment.get(segment_id, 0)) >= self._MAX_REQUESTED_ZOOMS_PER_SEGMENT:
                    reject_codes.add("zoom_budget_exceeded")
                    warnings.append(
                        self._warning(
                            "zoom_budget_exceeded",
                            f"Zoom budget for {segment_id} has been exhausted. Use the current visual feedback to choose the next tool.",
                        )
                    )
                zoom_scale = self._finite_number_or_none(tool_call.get("zoom_scale"))
                padding_px = self._finite_number_or_none(tool_call.get("padding_px"))
                if zoom_scale is None or zoom_scale < self._MIN_ZOOM_SCALE or zoom_scale > self._MAX_ZOOM_SCALE:
                    reject_codes.add("zoom_scale_out_of_range")
                    warnings.append(
                        self._warning(
                            "zoom_scale_out_of_range",
                            f"zoom_scale must be between {self._MIN_ZOOM_SCALE:.0f} and {self._MAX_ZOOM_SCALE:.0f}.",
                        )
                    )
                if padding_px is None or padding_px < self._MIN_ZOOM_PADDING_PX or padding_px > self._MAX_ZOOM_PADDING_PX:
                    reject_codes.add("zoom_padding_out_of_range")
                    warnings.append(
                        self._warning(
                            "zoom_padding_out_of_range",
                            f"padding_px must be between {self._MIN_ZOOM_PADDING_PX:.0f} and {self._MAX_ZOOM_PADDING_PX:.0f}.",
                        )
                    )
        if tool == "request_zoom_window":
            zoom_scale = self._finite_number_or_none(tool_call.get("zoom_scale"))
            window_width = self._finite_number_or_none(tool_call.get("width"))
            window_height = self._finite_number_or_none(tool_call.get("height"))
            origin_x = self._finite_number_or_none(tool_call.get("x"))
            origin_y = self._finite_number_or_none(tool_call.get("y"))
            if zoom_scale is None or zoom_scale < self._MIN_ZOOM_SCALE or zoom_scale > self._MAX_ZOOM_SCALE:
                reject_codes.add("zoom_scale_out_of_range")
                warnings.append(
                    self._warning(
                        "zoom_scale_out_of_range",
                        f"zoom_scale must be between {self._MIN_ZOOM_SCALE:.0f} and {self._MAX_ZOOM_SCALE:.0f}.",
                    )
                )
            if (
                window_width is None
                or window_height is None
                or window_width < self._MIN_ZOOM_WINDOW_WIDTH_PX
                or window_width > self._MAX_ZOOM_WINDOW_WIDTH_PX
                or window_height < self._MIN_ZOOM_WINDOW_HEIGHT_PX
                or window_height > self._MAX_ZOOM_WINDOW_HEIGHT_PX
            ):
                reject_codes.add("zoom_window_size_out_of_range")
                warnings.append(
                    self._warning(
                        "zoom_window_size_out_of_range",
                        f"Zoom window width/height must stay within [{self._MIN_ZOOM_WINDOW_WIDTH_PX:.0f},{self._MAX_ZOOM_WINDOW_WIDTH_PX:.0f}] and [{self._MIN_ZOOM_WINDOW_HEIGHT_PX:.0f},{self._MAX_ZOOM_WINDOW_HEIGHT_PX:.0f}] pixels.",
                    )
                )
            if (
                origin_x is not None
                and origin_y is not None
                and window_width is not None
                and window_height is not None
                and (origin_x + window_width <= 0.0 or origin_y + window_height <= 0.0 or origin_x >= float(canvas.width) or origin_y >= float(canvas.height))
            ):
                reject_codes.add("zoom_window_outside_canvas")
                warnings.append(
                    self._warning(
                        "zoom_window_outside_canvas",
                        "The requested zoom window does not intersect the canvas.",
                    )
                )

        if reject_codes:
            runtime_description = f"Rejected {tool} during preflight validation."
            quality_summary = " ; ".join(warning["message"] for warning in warnings)
            if tool == "line_to" and "line_to_used_on_smooth_curve_hint" in reject_codes:
                runtime_description = "Rejected line_to because the model described a smooth curve but used a straight line tool."
                quality_summary = "line_to draws a straight segment and is not appropriate for the described smooth curve."
            elif "current_segment_needs_refinement" in reject_codes:
                runtime_description = "Rejected the next drawing step because the newest segment still needs refinement."
                quality_summary = "The newest segment is not acceptable yet. Refine it before drawing the next segment."
            elif "refinement_limit_reached" in reject_codes:
                runtime_description = "Rejected repeated handle refinement because the current segment did not improve after several attempts."
                quality_summary = "The current segment did not become acceptable after repeated handle edits. Change strategy instead of continuing handle edits."
            elif "anchor_needs_correction" in reject_codes:
                runtime_description = "Rejected handle editing because one or more anchors are too far from the black source contour."
                quality_summary = "An anchor is far from the black source contour. Move the anchor or restart the path before adjusting handles."
            elif "best_segment_not_available" in reject_codes:
                runtime_description = f"Rejected restore_best_segment because no best recorded geometry is available for {tool_call.get('segment_id')}."
                quality_summary = "No best segment candidate is currently available to restore."
            elif "zoom_budget_exceeded" in reject_codes:
                runtime_description = f"Rejected {tool} because the zoom inspection budget has been exhausted."
                quality_summary = "Zoom budget reached. Use the current visual feedback to choose a drawing or editing tool."
            elif "anchor_not_on_source_contour" in reject_codes:
                runtime_description = f"Rejected {tool} because the proposed anchor is too far from the measured black source contour."
                quality_summary = "Anchor points must lie on or very near the black source contour."
            elif "endpoint_not_on_source_contour" in reject_codes:
                runtime_description = f"Rejected {tool} because the proposed endpoint anchor is too far from the measured black source contour."
                quality_summary = "Endpoint anchors must lie on or very near the black source contour."
            elif "zoom_window_outside_canvas" in reject_codes:
                runtime_description = "Rejected request_zoom_window because the requested crop does not intersect the canvas."
                quality_summary = "Requested zoom windows must overlap the canvas."
            return {
                "success": False,
                "rejected": True,
                "warnings": warnings,
                "runtime_description": runtime_description,
                "quality_summary": quality_summary,
                "contour_distance_checks": contour_distance_checks,
            }
        return {
            "success": True,
            "rejected": False,
            "warnings": warnings,
            "runtime_description": f"Accepted {tool} for execution.",
            "quality_summary": "",
            "contour_distance_checks": contour_distance_checks,
        }

    def _execute_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        ai_reason: str,
        canvas: FreePenCanvasState,
        successful_drawing_tool_calls: list[dict[str, Any]],
        history: list[dict[str, Any]],
        segment_refinement: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, str, str, str, list[dict[str, Any]], int, list[str], bool]:
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
                True,
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
                True,
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
                True,
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
                True,
            )

        if tool in {"request_segment_zoom", "request_zoom_window"}:
            if tool == "request_segment_zoom":
                runtime_description = (
                    f"Generated an inspection-only requested zoom for {tool_call['segment_id']} at {float(tool_call['zoom_scale']):.1f}x."
                )
            else:
                runtime_description = (
                    "Generated an inspection-only requested zoom window for the requested original-image region."
                )
            quality_summary = "Inspection-only zoom request executed. The path was not modified."
            current_feedback = [
                "Inspect the requested zoom image before choosing exactly one next drawing or editing tool.",
                "Tool calls must still use original image_px coordinates. Do not use zoomed display pixels as tool coordinates.",
            ]
            return (
                dict(tool_call),
                runtime_description,
                "good",
                quality_summary,
                warnings,
                rollback_applied,
                current_feedback,
                True,
            )

        if tool in {"move_anchor", "move_handle", "set_segment_handles", "convert_line_to_curve", "restore_best_segment"}:
            if tool == "convert_line_to_curve":
                executed = self._convert_line_to_curve(canvas=canvas, tool_call=tool_call)
                canvas.successful_step_count += 1
                canvas.step_count += 1
            elif tool == "restore_best_segment":
                best_geometry = self._get_best_segment_geometry(
                    segment_refinement=segment_refinement,
                    segment_id=str(tool_call["segment_id"]),
                )
                executed = self._restore_best_segment(
                    canvas=canvas,
                    tool_call=tool_call,
                    best_geometry=best_geometry,
                )
                if executed is None:
                    warnings.append(
                        self._warning(
                            "best_segment_not_available",
                            f"No best segment geometry is available for {tool_call['segment_id']}.",
                        )
                    )
                    runtime_description = (
                        f"Rejected restore_best_segment because no best recorded geometry is available for {tool_call['segment_id']}."
                    )
                    quality_summary = "restore_best_segment requires a previously recorded best geometry."
                    current_feedback = [
                        "No best segment geometry is available to restore.",
                        "Use undo_last, rollback_to_step, restart_path, or continue local refinement if no best candidate is available.",
                    ]
                    return (
                        None,
                        runtime_description,
                        "bad",
                        quality_summary,
                        warnings,
                        rollback_applied,
                        current_feedback,
                        False,
                    )
                canvas.successful_step_count += 1
                canvas.step_count += 1
            else:
                executed = canvas.apply_tool_call(tool_call)
            stored_tool_call = dict(tool_call)
            if tool == "restore_best_segment":
                stored_tool_call["best_segment_geometry"] = dict(executed["restored_geometry"])
            successful_drawing_tool_calls.append(stored_tool_call)
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
            elif tool == "restore_best_segment":
                runtime_description = f"Restored {tool_call['segment_id']} to its best recorded geometry."
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
            return (executed, runtime_description, "good", quality_summary, warnings, rollback_applied, current_feedback, True)

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
        return (executed, runtime_description, quality, quality_summary, warnings, rollback_applied, current_feedback, True)

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
        if any(warning["code"] == "refinement_limit_reached" for warning in warnings):
            feedback.append("Do not keep adjusting the same handles.")
            feedback.append("Use move_anchor if an anchor is wrong, or use undo_last, rollback_to_step, or restart_path if the current segment cannot be repaired.")
        if any(warning["code"] == "anchor_not_on_source_contour" for warning in warnings):
            feedback.append("The proposed anchor is not on the black contour.")
            feedback.append("Do not use empty background coordinates. Choose a start_path or move_anchor point on the measured black source contour.")
        if any(warning["code"] == "endpoint_not_on_source_contour" for warning in warnings):
            feedback.append("The proposed endpoint is not on the black contour.")
            feedback.append("Control handles may leave the contour, but endpoints must stay on it.")
        if any(warning["code"] == "anchor_needs_correction" for warning in warnings):
            feedback.append("An anchor is far from the black contour.")
            feedback.append("Do not adjust handles. Use move_anchor, undo_last, rollback_to_step, or restart_path.")
        if any(warning["code"] == "out_of_bounds_coordinate" for warning in warnings):
            feedback.append("The previous tool call was rejected because one or more coordinates were outside the canvas bounds.")
            feedback.append("Do not continue to the next segment. Retry the same segment with in-bounds coordinates.")
        if any(warning["code"] == "zoom_budget_exceeded" for warning in warnings):
            feedback.append("Zoom budget reached. Use the current visual feedback to choose a drawing or editing tool.")
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
        requested_zoom_windows: list[dict[str, Any]] | None = None,
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

        for requested_zoom in requested_zoom_windows or []:
            image_path_value = requested_zoom.get("path")
            if not isinstance(image_path_value, str):
                continue
            clipped_note = " The requested crop was clipped to the image bounds." if requested_zoom.get("clipped") else ""
            clamped_note = " The requested crop was clamped to the configured zoom window limits." if requested_zoom.get("clamped") else ""
            content.extend(
                self._build_image_parts(
                    semantic_text=(
                        "Requested zoom feedback after your previous inspection tool call. "
                        "This image is a local editor view, not a new coordinate system. "
                        "Top and left rulers show original image_px coordinates. "
                        "Grid labels use original image_px coordinates. "
                        "Tool calls must still use original image_px coordinates. "
                        "Do not use zoomed display pixels as tool coordinates. "
                        "Inspect the highlighted current segment before choosing the next tool."
                        + clipped_note
                        + clamped_note
                    ),
                    image_path=Path(image_path_value),
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

    def _build_source_mask(self, source_image: np.ndarray) -> np.ndarray | None:
        try:
            if source_image.ndim == 2:
                gray = source_image
            elif source_image.shape[2] == 4:
                gray = cv2.cvtColor(source_image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            source_mask = (gray < 128)
            if int(np.count_nonzero(source_mask)) == 0:
                return None
            return source_mask.astype(np.uint8)
        except Exception:
            return None

    def _build_source_distance_map(self, source_mask: np.ndarray | None) -> np.ndarray | None:
        try:
            if source_mask is None or int(np.count_nonzero(source_mask)) == 0:
                return None
            distance_input = np.where(source_mask > 0, 0, 255).astype(np.uint8)
            return cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
        except Exception:
            return None

    def _build_source_contour_summary(
        self,
        *,
        source_mask: np.ndarray | None,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        if source_mask is None or int(np.count_nonzero(source_mask)) == 0:
            return {
                "canvas_width": int(width),
                "canvas_height": int(height),
                "unavailable_reason": "source_mask_unavailable",
            }
        ys, xs = np.nonzero(source_mask)
        left_index = int(np.argmin(xs))
        right_index = int(np.argmax(xs))
        top_index = int(np.argmin(ys))
        bottom_index = int(np.argmax(ys))
        return {
            "canvas_width": int(width),
            "canvas_height": int(height),
            "bbox": {
                "x_min": int(np.min(xs)),
                "y_min": int(np.min(ys)),
                "x_max": int(np.max(xs)),
                "y_max": int(np.max(ys)),
            },
            "anchors": {
                "leftmost": [int(xs[left_index]), int(ys[left_index])],
                "topmost": [int(xs[top_index]), int(ys[top_index])],
                "rightmost": [int(xs[right_index]), int(ys[right_index])],
                "bottommost": [int(xs[bottom_index]), int(ys[bottom_index])],
            },
            "note": "These coordinates are measured from black source pixels in original image_px coordinates. Prefer these anchors over visual guessing.",
        }

    def _distance_to_source_contour(
        self,
        *,
        source_distance_map: np.ndarray | None,
        x: Any,
        y: Any,
        width: int,
        height: int,
    ) -> float | None:
        if source_distance_map is None:
            return None
        numeric_x = self._finite_number_or_none(x)
        numeric_y = self._finite_number_or_none(y)
        if numeric_x is None or numeric_y is None:
            return None
        if numeric_x < 0.0 or numeric_x >= float(width) or numeric_y < 0.0 or numeric_y >= float(height):
            return None
        sample_x = int(np.clip(round(numeric_x), 0, width - 1))
        sample_y = int(np.clip(round(numeric_y), 0, height - 1))
        return float(source_distance_map[sample_y, sample_x])

    def _build_current_segment_context(
        self,
        *,
        canvas: FreePenCanvasState,
        source_distance_map: np.ndarray | None,
        focus_tool_call: dict[str, Any] | None,
        segment_refinement: dict[str, dict[str, Any]],
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
        anchor_quality = self._build_anchor_quality(
            editable_geometry=editable_geometry,
            source_distance_map=source_distance_map,
            focus=focus,
            width=canvas.width,
            height=canvas.height,
        )
        refinement_state = self._segment_refinement_state_for_focus(
            focus=focus,
            segment_refinement=segment_refinement,
        )
        segment_split_hint = self._build_segment_split_hint(
            focus=focus,
            anchor_quality=anchor_quality,
            refinement_state=refinement_state,
            current_segment_metrics=quality_metrics.get("current_segment", {}),
        )
        quality_delta = self._build_quality_delta(
            focus=focus,
            metrics=quality_metrics,
            refinement_state=refinement_state,
        )
        status = self._build_current_segment_status(
            focus=focus,
            metrics=quality_metrics,
            refinement_state=refinement_state,
            anchor_quality=anchor_quality,
        )
        refinement_summary = self._build_refinement_summary(
            focus=focus,
            refinement_state=refinement_state,
            current_segment_status=status,
        )
        best_candidate_hint = self._build_best_candidate_hint(
            focus=focus,
            refinement_state=refinement_state,
        )
        return {
            "editable_geometry": editable_geometry,
            "focus": focus,
            "status": status,
            "quality_metrics": quality_metrics,
            "anchor_quality": anchor_quality,
            "segment_split_hint": segment_split_hint,
            "quality_delta": quality_delta,
            "refinement_summary": refinement_summary,
            "best_candidate_hint": best_candidate_hint,
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
        if tool in {"move_handle", "set_segment_handles", "convert_line_to_curve", "restore_best_segment", "request_segment_zoom"}:
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

    def _build_anchor_quality(
        self,
        *,
        editable_geometry: dict[str, Any],
        source_distance_map: np.ndarray | None,
        focus: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        if segment_id is None:
            return {"current_segment": {"segment_id": None, "unavailable_reason": "no_current_segment"}}
        segments = {
            str(segment["id"]): segment
            for path in editable_geometry["paths"]
            for segment in path["segments"]
        }
        anchors = {
            str(anchor["id"]): anchor
            for path in editable_geometry["paths"]
            for anchor in path["anchors"]
        }
        segment = segments.get(str(segment_id))
        if segment is None:
            return {"current_segment": {"segment_id": segment_id, "unavailable_reason": "segment_not_found"}}
        from_anchor = anchors.get(str(segment["from_anchor"]))
        to_anchor = anchors.get(str(segment["to_anchor"]))
        if from_anchor is None or to_anchor is None:
            return {"current_segment": {"segment_id": segment_id, "unavailable_reason": "anchor_not_found"}}
        return {
            "current_segment": {
                "segment_id": segment_id,
                "from_anchor": self._anchor_quality_payload(
                    anchor=from_anchor,
                    source_distance_map=source_distance_map,
                    width=width,
                    height=height,
                ),
                "to_anchor": self._anchor_quality_payload(
                    anchor=to_anchor,
                    source_distance_map=source_distance_map,
                    width=width,
                    height=height,
                ),
            }
        }

    def _anchor_quality_payload(
        self,
        *,
        anchor: dict[str, Any],
        source_distance_map: np.ndarray | None,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        distance = self._distance_to_source_contour(
            source_distance_map=source_distance_map,
            x=anchor["p"][0],
            y=anchor["p"][1],
            width=width,
            height=height,
        )
        if distance is None:
            status = "unknown"
        elif distance <= 10.0:
            status = "ok"
        elif distance <= 20.0:
            status = "warning"
        else:
            status = "bad"
        payload = {
            "id": str(anchor["id"]),
            "p": list(anchor["p"]),
            "status": status,
        }
        if distance is None:
            payload["unavailable_reason"] = "source_contour_validation_unavailable"
        else:
            payload["distance_to_source_px"] = float(distance)
        return payload

    def _build_current_segment_status(
        self,
        *,
        focus: dict[str, Any],
        metrics: dict[str, Any],
        refinement_state: dict[str, Any] | None,
        anchor_quality: dict[str, Any],
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        metric_payload = metrics.get("current_segment", {})
        unavailable_reason = metric_payload.get("unavailable_reason")
        refine_count = int((refinement_state or {}).get("refine_count", 0))
        worse_streak = int((refinement_state or {}).get("worse_streak", 0))
        anchor_payload = anchor_quality.get("current_segment", {})
        from_anchor_status = (anchor_payload.get("from_anchor") or {}).get("status")
        to_anchor_status = (anchor_payload.get("to_anchor") or {}).get("status")
        refinement_limit_reached = (
            refine_count >= self._MAX_REFINES_PER_SEGMENT
            or worse_streak >= self._MAX_WORSE_STREAK
        )
        if segment_id is None:
            return {
                "segment_id": None,
                "status": "unknown",
                "reason": "No current drawable segment exists yet.",
                "recommended_next_tools": ["start_path", "curve_to"],
                "may_advance_to_next_segment": True,
                "refinement_limit_reached": False,
                "may_continue_handle_refinement": False,
            }
        if from_anchor_status == "bad" or to_anchor_status == "bad":
            return {
                "segment_id": segment_id,
                "status": "needs_anchor_correction",
                "reason": "One or more anchors are far from the black source contour. This segment cannot be fixed reliably by handle edits.",
                "recommended_next_tools": ["move_anchor", "undo_last", "rollback_to_step", "restart_path", "inspect_history", "stalled"],
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": False,
                "may_continue_handle_refinement": False,
            }
        if unavailable_reason:
            return {
                "segment_id": segment_id,
                "status": "unknown",
                "reason": str(unavailable_reason),
                "recommended_next_tools": ["inspect_history", "set_segment_handles", "move_handle", "move_anchor", "restart_path", "stalled"],
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": False,
                "may_continue_handle_refinement": True,
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
                "refinement_limit_reached": False,
                "may_continue_handle_refinement": True,
            }
        if refinement_limit_reached:
            recommended = ["move_anchor", "undo_last", "rollback_to_step", "restart_path", "inspect_history", "stalled"]
            if self._normalize_best_segment_geometry(
                segment_id=str(segment_id),
                geometry=(refinement_state or {}).get("best_segment_geometry"),
            ) is not None:
                recommended.insert(0, "restore_best_segment")
            if focus.get("type") == "line":
                recommended.insert(0, "convert_line_to_curve")
            return {
                "segment_id": segment_id,
                "status": "needs_refinement",
                "reason": "The current segment did not become acceptable after repeated handle edits.",
                "recommended_next_tools": recommended,
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": True,
                "may_continue_handle_refinement": False,
            }
        recommended = ["set_segment_handles", "move_handle", "move_anchor", "undo_last", "rollback_to_step", "inspect_history", "restart_path", "stalled"]
        if focus.get("type") == "line":
            recommended.insert(0, "convert_line_to_curve")
        return {
            "segment_id": segment_id,
            "status": "needs_refinement",
            "reason": "The newest segment is not acceptable yet and should be refined before advancing.",
            "recommended_next_tools": recommended,
            "may_advance_to_next_segment": False,
            "refinement_limit_reached": False,
            "may_continue_handle_refinement": True,
        }

    def _build_segment_split_hint(
        self,
        *,
        focus: dict[str, Any],
        anchor_quality: dict[str, Any],
        refinement_state: dict[str, Any] | None,
        current_segment_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        if segment_id is None:
            return {"segment_id": None, "should_consider_split": False, "unavailable_reason": "no_current_segment"}
        anchor_payload = anchor_quality.get("current_segment", {})
        from_status = (anchor_payload.get("from_anchor") or {}).get("status")
        to_status = (anchor_payload.get("to_anchor") or {}).get("status")
        best_quality = (refinement_state or {}).get("best_quality") or {}
        best_p90 = best_quality.get("path_to_source_p90_px")
        refinement_limit_reached = bool(
            (refinement_state or {}).get("refine_count", 0) >= self._MAX_REFINES_PER_SEGMENT
            or (refinement_state or {}).get("worse_streak", 0) >= self._MAX_WORSE_STREAK
        )
        should_consider_split = bool(
            focus.get("type") == "cubic"
            and refinement_limit_reached
            and from_status == "ok"
            and to_status == "ok"
            and best_p90 is not None
            and float(best_p90) > self._SEGMENT_ACCEPTABLE_P90_PX
        )
        result = {
            "segment_id": segment_id,
            "should_consider_split": should_consider_split,
        }
        if should_consider_split:
            rollback_before = max(0, int((refinement_state or {}).get("created_at_successful_step", 0)) - 1)
            result.update(
                {
                    "reason": "Both anchors are on the black contour, but the cubic segment still does not fit after repeated handle edits. This segment may be too long or has high curvature variation.",
                    "recommended_strategy": "rollback_to_before_segment_and_redraw_as_two_shorter_curves",
                    "rollback_before_segment_step": rollback_before,
                }
            )
        else:
            result["recommended_strategy"] = "continue_current_strategy"
        return result

    @staticmethod
    def _segment_refinement_state_for_focus(
        *,
        focus: dict[str, Any],
        segment_refinement: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        segment_id = focus.get("segment_id")
        if segment_id is None:
            return None
        return segment_refinement.get(str(segment_id))

    @classmethod
    def _normalize_best_segment_geometry(
        cls,
        *,
        segment_id: str,
        geometry: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(geometry, dict):
            return None
        segment_type = str(geometry.get("type") or "").strip().lower()
        if segment_type not in {"line", "cubic"}:
            return None
        point = geometry.get("p")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        point_x = cls._finite_number_or_none(point[0])
        point_y = cls._finite_number_or_none(point[1])
        if point_x is None or point_y is None:
            return None
        normalized: dict[str, Any] = {
            "segment_id": str(segment_id),
            "type": segment_type,
            "p": [point_x, point_y],
        }
        if segment_type == "cubic":
            c1 = geometry.get("c1")
            c2 = geometry.get("c2")
            if not isinstance(c1, (list, tuple)) or len(c1) != 2 or not isinstance(c2, (list, tuple)) or len(c2) != 2:
                return None
            c1x = cls._finite_number_or_none(c1[0])
            c1y = cls._finite_number_or_none(c1[1])
            c2x = cls._finite_number_or_none(c2[0])
            c2y = cls._finite_number_or_none(c2[1])
            if c1x is None or c1y is None or c2x is None or c2y is None:
                return None
            normalized["c1"] = [c1x, c1y]
            normalized["c2"] = [c2x, c2y]
        return normalized

    def _get_best_segment_geometry(
        self,
        *,
        segment_refinement: dict[str, dict[str, Any]] | None,
        segment_id: str | None,
    ) -> dict[str, Any] | None:
        if segment_id is None or segment_refinement is None:
            return None
        payload = segment_refinement.get(str(segment_id)) or {}
        return self._normalize_best_segment_geometry(
            segment_id=str(segment_id),
            geometry=payload.get("best_segment_geometry"),
        )

    def _build_quality_delta(
        self,
        *,
        focus: dict[str, Any],
        metrics: dict[str, Any],
        refinement_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        metric_payload = metrics.get("current_segment", {})
        unavailable_reason = metric_payload.get("unavailable_reason")
        if segment_id is None:
            return {"segment_id": None, "status": "unavailable", "message": "No current segment exists yet."}
        if unavailable_reason:
            return {
                "segment_id": segment_id,
                "status": "unavailable",
                "unavailable_reason": str(unavailable_reason),
                "message": "Current segment quality delta is unavailable.",
            }
        previous_quality = (refinement_state or {}).get("previous_quality")
        best_quality = (refinement_state or {}).get("best_quality")
        previous_p90 = None if not previous_quality else previous_quality.get("path_to_source_p90_px")
        current_p90 = float(metric_payload["path_to_source_p90_px"])
        best_p90 = None if not best_quality else best_quality.get("path_to_source_p90_px")
        improved_vs_previous = (
            previous_p90 is not None and current_p90 < float(previous_p90) - self._QUALITY_DELTA_EPSILON_PX
        )
        improved_vs_best = (
            best_p90 is not None and current_p90 < float(best_p90) - self._QUALITY_DELTA_EPSILON_PX
        )
        worse_streak = int((refinement_state or {}).get("worse_streak", 0))
        message = "No previous local quality is available yet for this segment."
        if previous_p90 is not None and current_p90 > float(previous_p90) + self._QUALITY_DELTA_EPSILON_PX:
            message = "This edit made the current segment worse. Do not keep moving handles in the same direction."
        elif improved_vs_previous:
            message = "This edit improved the current segment compared with the previous attempt."
        elif previous_p90 is not None:
            message = "This edit did not meaningfully improve the current segment. Change strategy if repeated handle edits keep failing."
        return {
            "segment_id": segment_id,
            "previous_p90_px": previous_p90,
            "current_p90_px": current_p90,
            "best_p90_px": best_p90,
            "improved_vs_previous": bool(improved_vs_previous),
            "improved_vs_best": bool(improved_vs_best),
            "worse_streak": worse_streak,
            "message": message,
        }

    def _build_refinement_summary(
        self,
        *,
        focus: dict[str, Any],
        refinement_state: dict[str, Any] | None,
        current_segment_status: dict[str, Any],
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        if segment_id is None:
            return {
                "segment_id": None,
                "refine_count": 0,
                "max_refines": self._MAX_REFINES_PER_SEGMENT,
                "worse_streak": 0,
                "refinement_limit_reached": False,
                "recommended_strategy": "draw_first_segment",
            }
        best_quality = (refinement_state or {}).get("best_quality") or {}
        recommended_strategy = "continue_local_refinement"
        if current_segment_status.get("refinement_limit_reached"):
            recommended_strategy = (
                "restore_best_or_split_segment"
                if (refinement_state or {}).get("best_segment_geometry") is not None
                else "stop_handle_refinement_and_move_anchor_or_rollback"
            )
        elif current_segment_status.get("may_advance_to_next_segment"):
            recommended_strategy = "advance_to_next_segment_when_ready"
        return {
            "segment_id": segment_id,
            "refine_count": int((refinement_state or {}).get("refine_count", 0)),
            "max_refines": self._MAX_REFINES_PER_SEGMENT,
            "worse_streak": int((refinement_state or {}).get("worse_streak", 0)),
            "refinement_limit_reached": bool(current_segment_status.get("refinement_limit_reached")),
            "best_quality": {
                "path_to_source_p90_px": best_quality.get("path_to_source_p90_px"),
            }
            if best_quality
            else {},
            "best_tool_call": (refinement_state or {}).get("best_tool_call"),
            "best_segment_geometry": self._normalize_best_segment_geometry(
                segment_id=str(segment_id),
                geometry=(refinement_state or {}).get("best_segment_geometry"),
            ),
            "best_restore_available": self._normalize_best_segment_geometry(
                segment_id=str(segment_id),
                geometry=(refinement_state or {}).get("best_segment_geometry"),
            )
            is not None,
            "created_at_successful_step": (refinement_state or {}).get("created_at_successful_step"),
            "rollback_before_segment_step": (
                None
                if (refinement_state or {}).get("created_at_successful_step") is None
                else max(0, int((refinement_state or {}).get("created_at_successful_step")) - 1)
            ),
            "recommended_strategy": recommended_strategy,
        }

    def _build_best_candidate_hint(
        self,
        *,
        focus: dict[str, Any],
        refinement_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        segment_id = focus.get("segment_id")
        if segment_id is None:
            return {
                "segment_id": None,
                "can_restore": False,
                "message": "No current segment exists yet.",
            }
        best_quality = (refinement_state or {}).get("best_quality") or {}
        can_restore = self._normalize_best_segment_geometry(
            segment_id=str(segment_id),
            geometry=(refinement_state or {}).get("best_segment_geometry"),
        ) is not None
        message = "No best candidate is currently recorded for this segment."
        if can_restore:
            message = "The best known version of this segment is available. Use restore_best_segment before trying rollback or split."
        return {
            "segment_id": segment_id,
            "best_p90_px": best_quality.get("path_to_source_p90_px"),
            "can_restore": can_restore,
            "restore_tool": "restore_best_segment" if can_restore else None,
            "message": message,
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
        return self._sample_segment_points_with_count(segment_record=segment_record, sample_count=None)

    def _sample_segment_points_with_count(
        self,
        *,
        segment_record: dict[str, Any],
        sample_count: int | None,
    ) -> np.ndarray:
        from_point = segment_record.get("from_point")
        to_point = segment_record.get("to_point")
        raw_segment = segment_record["raw"]
        if not from_point or not to_point:
            return np.asarray([], dtype=np.float64)
        effective_sample_count = max(8, int(self.sample_count_per_segment if sample_count is None else sample_count))
        if raw_segment["type"] == "line":
            p0 = np.array(from_point, dtype=np.float64)
            p1 = np.array(to_point, dtype=np.float64)
            samples = []
            for index in range(effective_sample_count):
                t = index / float(effective_sample_count - 1)
                point = ((1.0 - t) * p0) + (t * p1)
                samples.append(point.tolist())
            return np.asarray(samples, dtype=np.float64)
        return np.asarray(
            FreePenCanvasState._sample_cubic_segment(
                p0=(float(from_point[0]), float(from_point[1])),
                c1=(float(raw_segment["c1"][0]), float(raw_segment["c1"][1])),
                c2=(float(raw_segment["c2"][0]), float(raw_segment["c2"][1])),
                p1=(float(to_point[0]), float(to_point[1])),
                sample_count=effective_sample_count,
            ),
            dtype=np.float64,
        )

    def _build_zoom_segment_samples(
        self,
        *,
        segment_record: dict[str, Any],
        zoom_scale: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        estimated_length = self._estimate_segment_length(segment_record=segment_record)
        display_length_px = max(0.0, estimated_length * float(zoom_scale))
        sample_count = max(
            int(self._BASE_ZOOM_SAMPLE_COUNT),
            int(math.ceil(display_length_px / self._ZOOM_SAMPLE_SPACING_PX)) + 1,
        )
        sample_count = min(int(self._MAX_ZOOM_SAMPLE_COUNT), sample_count)
        samples = self._sample_segment_points_with_count(
            segment_record=segment_record,
            sample_count=sample_count,
        )
        return (
            samples,
            {
                "mode": "dynamic_zoom_polyline",
                "sample_count": int(sample_count),
                "base_zoom_sample_count": int(self._BASE_ZOOM_SAMPLE_COUNT),
                "sample_spacing_px": float(self._ZOOM_SAMPLE_SPACING_PX),
            },
        )

    def _estimate_segment_length(self, *, segment_record: dict[str, Any]) -> float:
        raw_segment = segment_record["raw"]
        from_point = segment_record.get("from_point")
        to_point = segment_record.get("to_point")
        if not from_point or not to_point:
            return 0.0
        if raw_segment["type"] == "line":
            return float(math.hypot(float(to_point[0]) - float(from_point[0]), float(to_point[1]) - float(from_point[1])))
        coarse_samples = self._sample_segment_points_with_count(segment_record=segment_record, sample_count=64)
        if coarse_samples.shape[0] < 2:
            return 0.0
        deltas = np.diff(coarse_samples, axis=0)
        return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))

    def _build_requested_segment_zoom_window(
        self,
        *,
        segment_record: dict[str, Any],
        sampled_points: np.ndarray,
        padding_px: float,
        canvas_width: int,
        canvas_height: int,
    ) -> dict[str, Any] | None:
        required_points: list[list[float]] = []
        from_point = segment_record.get("from_point")
        to_point = segment_record.get("to_point")
        if from_point is not None:
            required_points.append([float(from_point[0]), float(from_point[1])])
        if to_point is not None:
            required_points.append([float(to_point[0]), float(to_point[1])])
        if sampled_points.size:
            required_points.extend(sampled_points.tolist())
        if not required_points:
            return None
        required_array = np.asarray(required_points, dtype=np.float64)
        required_bbox = self._bbox_from_points(required_array)
        requested_padding = max(float(self._MIN_SEGMENT_ZOOM_PADDING_PX), float(padding_px))
        min_padding = float(self._MIN_SEGMENT_ZOOM_PADDING_PX)
        required_width = float(required_bbox["x_max"] - required_bbox["x_min"])
        required_height = float(required_bbox["y_max"] - required_bbox["y_min"])

        def _effective_padding(required_span: float, soft_limit: float) -> float:
            if required_span + (2.0 * requested_padding) <= soft_limit:
                return requested_padding
            if required_span + (2.0 * min_padding) <= soft_limit:
                return max(min_padding, (soft_limit - required_span) * 0.5)
            return min_padding

        pad_x = _effective_padding(required_width, float(self._MAX_SEGMENT_ZOOM_WIDTH_PX))
        pad_y = _effective_padding(required_height, float(self._MAX_SEGMENT_ZOOM_HEIGHT_PX))
        target_width = max(float(self._MIN_SEGMENT_ZOOM_WIDTH_PX), required_width + (2.0 * pad_x))
        target_height = max(float(self._MIN_SEGMENT_ZOOM_HEIGHT_PX), required_height + (2.0 * pad_y))
        requested_crop_x0 = float(required_bbox["x_min"]) - requested_padding
        requested_crop_y0 = float(required_bbox["y_min"]) - requested_padding
        requested_crop_x1 = float(required_bbox["x_max"]) + requested_padding
        requested_crop_y1 = float(required_bbox["y_max"]) + requested_padding
        crop_x0 = max(0.0, min(float(required_bbox["x_min"] - pad_x), float(canvas_width) - target_width))
        crop_y0 = max(0.0, min(float(required_bbox["y_min"] - pad_y), float(canvas_height) - target_height))
        crop_x1 = min(float(canvas_width), crop_x0 + target_width)
        crop_y1 = min(float(canvas_height), crop_y0 + target_height)
        crop_origin = (int(math.floor(crop_x0)), int(math.floor(crop_y0)))
        crop_size = (
            max(1, int(math.ceil(crop_x1 - crop_x0))),
            max(1, int(math.ceil(crop_y1 - crop_y0))),
        )
        crop_bounds = (
            float(crop_origin[0]),
            float(crop_origin[1]),
            float(crop_origin[0] + crop_size[0]),
            float(crop_origin[1] + crop_size[1]),
        )
        anchors_visible = all(
            self._point_in_crop(point=(float(point[0]), float(point[1])), crop_bounds=crop_bounds)
            for point in (from_point, to_point)
            if point is not None
        )
        curve_visible = self._points_within_crop(points=required_array, crop_bounds=crop_bounds)
        raw_segment = segment_record["raw"]
        handle_points = []
        if raw_segment["type"] == "cubic":
            handle_points = [
                (float(raw_segment["c1"][0]), float(raw_segment["c1"][1])),
                (float(raw_segment["c2"][0]), float(raw_segment["c2"][1])),
            ]
        handles_visible = all(self._point_in_crop(point=point, crop_bounds=crop_bounds) for point in handle_points)
        clamped = bool(
            target_width > float(self._MAX_SEGMENT_ZOOM_WIDTH_PX)
            or target_height > float(self._MAX_SEGMENT_ZOOM_HEIGHT_PX)
            or pad_x < requested_padding - 0.5
            or pad_y < requested_padding - 0.5
        )
        clipped = bool(
            requested_crop_x0 < 0.0
            or requested_crop_y0 < 0.0
            or requested_crop_x1 > float(canvas_width)
            or requested_crop_y1 > float(canvas_height)
        )
        return {
            "crop_origin": crop_origin,
            "crop_size": crop_size,
            "required_bbox": required_bbox,
            "anchors_visible": bool(anchors_visible),
            "curve_visible": bool(curve_visible),
            "handles_visible": bool(handles_visible),
            "clipped": clipped,
            "clamped": clamped,
        }

    @staticmethod
    def _bbox_from_points(points: np.ndarray) -> dict[str, int]:
        return {
            "x_min": int(math.floor(float(np.min(points[:, 0])))),
            "y_min": int(math.floor(float(np.min(points[:, 1])))),
            "x_max": int(math.ceil(float(np.max(points[:, 0])))),
            "y_max": int(math.ceil(float(np.max(points[:, 1])))),
        }

    @staticmethod
    def _point_in_crop(
        *,
        point: tuple[float, float],
        crop_bounds: tuple[float, float, float, float],
    ) -> bool:
        x0, y0, x1, y1 = crop_bounds
        return x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1

    def _points_within_crop(
        self,
        *,
        points: np.ndarray,
        crop_bounds: tuple[float, float, float, float],
    ) -> bool:
        if points.size == 0:
            return False
        x0, y0, x1, y1 = crop_bounds
        return bool(
            np.all(points[:, 0] >= x0)
            and np.all(points[:, 0] <= x1)
            and np.all(points[:, 1] >= y0)
            and np.all(points[:, 1] <= y1)
        )

    def _segment_geometry_from_canvas(
        self,
        *,
        canvas: FreePenCanvasState,
        segment_id: str,
    ) -> dict[str, Any] | None:
        editable_geometry = self._build_editable_geometry(canvas)
        record = self._segment_record_from_canvas(
            canvas=canvas,
            editable_geometry=editable_geometry,
            segment_id=segment_id,
        )
        if record is None:
            return None
        raw_segment = record["raw"]
        payload: dict[str, Any] = {
            "segment_id": str(segment_id),
            "type": str(raw_segment["type"]),
        }
        if raw_segment["type"] == "line":
            payload["p"] = [float(raw_segment["p"][0]), float(raw_segment["p"][1])]
        else:
            payload["c1"] = [float(raw_segment["c1"][0]), float(raw_segment["c1"][1])]
            payload["c2"] = [float(raw_segment["c2"][0]), float(raw_segment["c2"][1])]
            payload["p"] = [float(raw_segment["p"][0]), float(raw_segment["p"][1])]
        return payload

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

    def _restore_best_segment(
        self,
        *,
        canvas: FreePenCanvasState,
        tool_call: dict[str, Any],
        best_geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        path = canvas.current_path()
        if path is None:
            raise FreePenCanvasError("restore_best_segment requires an open path")
        geometry = best_geometry or self._normalize_best_segment_geometry(
            segment_id=str(tool_call["segment_id"]),
            geometry=tool_call.get("best_segment_geometry"),
        )
        if geometry is None:
            return None
        editable_geometry = self._build_editable_geometry(canvas)
        geometry_path = editable_geometry["paths"][0] if editable_geometry["paths"] else {"segments": []}
        drawable_segments = [segment for segment in path.segments if segment["type"] in {"line", "cubic"}]
        for index, segment in enumerate(geometry_path["segments"]):
            if segment["id"] != str(tool_call["segment_id"]):
                continue
            raw_segment = drawable_segments[index]
            old_geometry = self._segment_geometry_from_canvas(canvas=canvas, segment_id=str(tool_call["segment_id"]))
            restored_type = str(geometry["type"])
            raw_segment["type"] = restored_type
            raw_segment["p"] = [float(geometry["p"][0]), float(geometry["p"][1])]
            if restored_type == "cubic":
                raw_segment["c1"] = [float(geometry["c1"][0]), float(geometry["c1"][1])]
                raw_segment["c2"] = [float(geometry["c2"][0]), float(geometry["c2"][1])]
            else:
                raw_segment.pop("c1", None)
                raw_segment.pop("c2", None)
            return {
                "tool": "restore_best_segment",
                "segment_id": str(tool_call["segment_id"]),
                "old_geometry": old_geometry,
                "restored_geometry": dict(geometry),
            }
        raise FreePenCanvasError(f"unknown segment_id: {tool_call['segment_id']}")

    def _update_segment_refinement_state(
        self,
        *,
        segment_refinement: dict[str, dict[str, Any]],
        tool_call: dict[str, Any],
        current_segment_context: dict[str, Any],
        canvas: FreePenCanvasState,
        successful_step_count: int,
    ) -> dict[str, dict[str, Any]]:
        updated = {
            segment_id: dict(payload)
            for segment_id, payload in segment_refinement.items()
        }
        focus = current_segment_context["focus"]
        metrics = current_segment_context["quality_metrics"].get("current_segment", {})
        segment_id = focus.get("segment_id")
        if segment_id is None:
            return updated
        tool = str(tool_call.get("tool"))
        relevant_tools = {
            "curve_to",
            "line_to",
            "convert_line_to_curve",
            "restore_best_segment",
            "set_segment_handles",
            "move_handle",
            "move_anchor",
        }
        if tool not in relevant_tools:
            return updated
        current_quality = None if metrics.get("unavailable_reason") else {
            "path_to_source_mean_px": float(metrics["path_to_source_mean_px"]),
            "path_to_source_max_px": float(metrics["path_to_source_max_px"]),
            "path_to_source_p90_px": float(metrics["path_to_source_p90_px"]),
        }
        existing = dict(updated.get(str(segment_id), {}))
        previous_quality = existing.get("last_quality")
        best_quality = existing.get("best_quality")
        best_tool_call = existing.get("best_tool_call")
        best_segment_geometry = self._normalize_best_segment_geometry(
            segment_id=str(segment_id),
            geometry=existing.get("best_segment_geometry"),
        )
        refine_count = int(existing.get("refine_count", 0))
        worse_streak = int(existing.get("worse_streak", 0))

        if tool in {"curve_to", "line_to"}:
            refine_count = 0
            worse_streak = 0
            created_at_successful_step = int(successful_step_count)
        elif tool == "convert_line_to_curve":
            refine_count = 0
            worse_streak = 0
            created_at_successful_step = existing.get("created_at_successful_step", int(successful_step_count))
        elif tool == "move_anchor":
            refine_count = 1
            worse_streak = 0
            created_at_successful_step = existing.get("created_at_successful_step")
        elif tool == "restore_best_segment":
            refine_count = int(existing.get("refine_count", 0))
            worse_streak = 0
            created_at_successful_step = existing.get("created_at_successful_step")
            if tool_call.get("best_segment_geometry"):
                best_segment_geometry = self._normalize_best_segment_geometry(
                    segment_id=str(segment_id),
                    geometry=tool_call.get("best_segment_geometry"),
                )
        else:
            refine_count += 1
            created_at_successful_step = existing.get("created_at_successful_step")
            if current_quality is not None and previous_quality is not None:
                current_p90 = float(current_quality["path_to_source_p90_px"])
                previous_p90 = float(previous_quality["path_to_source_p90_px"])
                if current_p90 > previous_p90 + self._QUALITY_DELTA_EPSILON_PX:
                    worse_streak += 1
                elif current_p90 < previous_p90 - self._QUALITY_DELTA_EPSILON_PX:
                    worse_streak = 0

        if current_quality is not None:
            if best_quality is None or (
                current_quality["path_to_source_p90_px"]
                < float(best_quality["path_to_source_p90_px"]) - self._QUALITY_DELTA_EPSILON_PX
            ):
                best_quality = dict(current_quality)
                best_tool_call = dict(tool_call)
                best_segment_geometry = self._normalize_best_segment_geometry(
                    segment_id=str(segment_id),
                    geometry=self._segment_geometry_from_canvas(canvas=canvas, segment_id=str(segment_id)),
                )
            elif best_quality is None:
                best_quality = dict(current_quality)
                best_tool_call = dict(tool_call)
                best_segment_geometry = self._normalize_best_segment_geometry(
                    segment_id=str(segment_id),
                    geometry=self._segment_geometry_from_canvas(canvas=canvas, segment_id=str(segment_id)),
                )

        updated[str(segment_id)] = {
            "refine_count": refine_count,
            "previous_quality": previous_quality,
            "last_quality": current_quality,
            "best_quality": best_quality,
            "best_tool_call": best_tool_call,
            "best_segment_geometry": best_segment_geometry,
            "last_tool_call": dict(tool_call),
            "worse_streak": worse_streak,
            "created_at_successful_step": created_at_successful_step,
        }
        return updated

    def _prune_segment_refinement_state(
        self,
        *,
        segment_refinement: dict[str, dict[str, Any]],
        canvas: FreePenCanvasState,
    ) -> dict[str, dict[str, Any]]:
        editable_geometry = self._build_editable_geometry(canvas)
        valid_segment_ids = {
            str(segment["id"])
            for path in editable_geometry["paths"]
            for segment in path["segments"]
        }
        return {
            segment_id: payload
            for segment_id, payload in segment_refinement.items()
            if segment_id in valid_segment_ids
        }

    def _replay_successful_tool_calls(self, *, canvas: FreePenCanvasState, tool_calls: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
        canvas.clear()
        canvas.step_count = 0
        canvas.successful_step_count = 0
        canvas.invalid_step_count = 0
        for tool_call in tool_calls:
            if str(tool_call.get("tool")) == "convert_line_to_curve":
                self._convert_line_to_curve(canvas=canvas, tool_call=tool_call)
                canvas.successful_step_count += 1
            elif str(tool_call.get("tool")) == "restore_best_segment":
                self._restore_best_segment(canvas=canvas, tool_call=tool_call)
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
        source_contour_summary: dict[str, Any],
        requested_zoom_total_count: int,
        requested_zoom_by_segment: dict[str, int],
    ) -> dict[str, Any]:
        closed_path_count = sum(1 for path in canvas.paths if path.closed)
        last_action = str(history[-1]["runtime_description"]) if history else "none"
        current_segment_focus = current_segment_context["focus"]
        current_segment_status = current_segment_context["status"]
        if canvas.path_open:
            near_start = canvas.distance_to_start()
            if current_segment_status.get("may_advance_to_next_segment") is False and current_segment_focus.get("segment_id"):
                if current_segment_status.get("refinement_limit_reached"):
                    if current_segment_context.get("segment_split_hint", {}).get("should_consider_split"):
                        current_goal = "The newest segment still cannot be repaired with repeated handle edits. Restore the best version or roll back before this segment and redraw the region with shorter curves."
                    else:
                        current_goal = "The newest segment still cannot be repaired with repeated handle edits. Change strategy before continuing."
                elif current_segment_status.get("status") == "needs_anchor_correction":
                    current_goal = "The newest segment has an anchor far from the source contour. Correct the anchor before continuing."
                else:
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
        zoom_actions = self._build_zoom_allowed_actions(
            current_segment_context=current_segment_context,
            requested_zoom_total_count=requested_zoom_total_count,
            requested_zoom_by_segment=requested_zoom_by_segment,
        )
        allowed_next_actions = tuple(dict.fromkeys([*allowed_next_actions, *zoom_actions]))
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
            "source_contour_summary": source_contour_summary,
            "requested_zoom_total_count": int(requested_zoom_total_count),
            "requested_zoom_total_budget": int(self._MAX_REQUESTED_ZOOMS_TOTAL),
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

    def _build_zoom_allowed_actions(
        self,
        *,
        current_segment_context: dict[str, Any],
        requested_zoom_total_count: int,
        requested_zoom_by_segment: dict[str, int],
    ) -> tuple[str, ...]:
        if requested_zoom_total_count >= self._MAX_REQUESTED_ZOOMS_TOTAL:
            return ()
        allowed: list[str] = ["request_zoom_window"]
        segment_id = current_segment_context.get("focus", {}).get("segment_id")
        if segment_id is not None and int(requested_zoom_by_segment.get(str(segment_id), 0)) < self._MAX_REQUESTED_ZOOMS_PER_SEGMENT:
            allowed.append("request_segment_zoom")
        return tuple(allowed)

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
        if round_status == "rejected_action" and any(warning["code"] == "anchor_not_on_source_contour" for warning in warnings):
            return "The proposed start point is not on the black contour. Choose a new start_path or restart_path point on the measured black source contour. Do not use empty background coordinates."
        if round_status == "rejected_action" and any(warning["code"] == "endpoint_not_on_source_contour" for warning in warnings):
            return "The proposed endpoint is not on the black contour. Pick an endpoint anchor on the black source contour. Control handles may leave the contour, but endpoints must stay on it."
        if round_status == "rejected_action" and any(warning["code"] == "anchor_needs_correction" for warning in warnings):
            return "The current segment has an anchor far from the black contour. Do not adjust handles. Use move_anchor to place the bad anchor on the black contour, or use undo_last, rollback_to_step, or restart_path if the path started wrong."
        if round_status == "rejected_action" and any(warning["code"] == "refinement_limit_reached" for warning in warnings):
            best_p90 = current_segment_context.get("refinement_summary", {}).get("best_quality", {}).get("path_to_source_p90_px")
            if current_segment_context.get("best_candidate_hint", {}).get("can_restore"):
                return (
                    "The current segment is still not acceptable after repeated handle edits. "
                    "Do not keep adjusting the same handles. "
                    "Prefer restore_best_segment before trying a different strategy. "
                    f"The best previous candidate for this segment had p90={best_p90:.1f}. "
                    "Restore the best previous candidate with restore_best_segment, then consider rollback_to_step and redraw this segment as two shorter curve_to segments if one cubic cannot fit the contour."
                )
            return "The current segment is still not acceptable after repeated handle edits. Do not keep adjusting the same handles. Use move_anchor if an anchor is wrong, or use undo_last, rollback_to_step, or restart_path if this segment cannot be repaired."
        if round_status == "rejected_action" and any(warning["code"] == "current_segment_needs_refinement" for warning in warnings):
            return "The newest segment is not acceptable yet. Do not draw the next segment. Refine the current segment first using set_segment_handles, move_handle, move_anchor, or convert_line_to_curve if it is a line segment."
        if round_status == "rejected_action" and any(warning["code"] == "out_of_bounds_coordinate" for warning in warnings):
            return "The previous tool call was rejected because a coordinate was outside the canvas bounds. Do not continue to the next segment. Retry the same segment with in-bounds coordinates, or refine the latest existing segment with set_segment_handles."
        if round_status == "rejected_action" and any(warning["code"] == "zoom_budget_exceeded" for warning in warnings):
            return "Zoom budget reached. Use the current visual feedback to choose a drawing or editing tool."
        if round_status == "rejected_action":
            return "Revise the action instead of repeating the rejected call."
        if final_decision == "finish":
            return "Tracing is complete."
        if final_decision == "stalled":
            return "Stop the tracing loop."
        if current_segment_context["status"].get("status") == "needs_anchor_correction":
            return "The current segment has an anchor far from the black contour. Do not adjust handles. Use move_anchor to place the bad anchor on the black contour, or use undo_last, rollback_to_step, or restart_path if the path started wrong."
        if current_segment_context["status"].get("may_advance_to_next_segment") is False and current_segment_context["focus"].get("segment_id"):
            if current_segment_context["status"].get("refinement_limit_reached"):
                best_p90 = current_segment_context.get("refinement_summary", {}).get("best_quality", {}).get("path_to_source_p90_px")
                if current_segment_context.get("best_candidate_hint", {}).get("can_restore"):
                    rollback_before = current_segment_context.get("segment_split_hint", {}).get("rollback_before_segment_step")
                    return (
                        "The current segment is still not acceptable after repeated handle edits. "
                        "Do not keep adjusting the same handles. "
                        f"The best previous candidate for this segment had p90={best_p90:.1f}. "
                        "Prefer restore_best_segment before trying a different strategy. "
                        + (
                            f"Consider rollback_to_step({rollback_before}) to remove this segment, then redraw this region as two shorter curve_to segments with an intermediate anchor on the black contour. "
                            if rollback_before is not None and current_segment_context.get("segment_split_hint", {}).get("should_consider_split")
                            else ""
                        )
                        + "Do not draw the next outer segment until the replacement segment is acceptable."
                    )
                return "The current segment is still not acceptable after repeated handle edits. Do not keep adjusting the same handles. Use move_anchor if an anchor is wrong, or use undo_last, rollback_to_step, or restart_path if this segment cannot be repaired."
            delta = current_segment_context.get("quality_delta", {})
            if delta.get("message") and delta.get("improved_vs_previous") is False and delta.get("previous_p90_px") is not None:
                restore_text = (
                    " Prefer restore_best_segment before trying a different strategy."
                    if current_segment_context.get("best_candidate_hint", {}).get("can_restore")
                    else ""
                )
                return (
                    f"The newest segment is not acceptable yet. Do not draw the next segment. "
                    f"{delta['message']}{restore_text} Refine the current segment first."
                )
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
