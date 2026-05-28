from __future__ import annotations

import json
import hashlib
import math
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from services.ai_adapters.base import VisionReviewAdapter
from services.free_pen_canvas import FreePenCanvasError, FreePenCanvasState
from services.free_pen_prompt import (
    FreePenPromptInput,
    FreePenToolPromptInput,
    build_free_pen_prompt,
    build_free_pen_tool_prompt,
)


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "ai_free_pen.schema.json"
TOOL_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "ai_free_pen_tool.schema.json"


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
    successful_step_count: int = 0
    invalid_step_count: int = 0
    recent_history: tuple[str, ...] = ()
    current_feedback: tuple[str, ...] = ()

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
            successful_step_count=self.successful_step_count,
            invalid_step_count=self.invalid_step_count,
            recent_history=self.recent_history,
            current_feedback=self.current_feedback,
        )


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
    return json.loads(TOOL_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_free_pen_tool_response(response: dict[str, Any]) -> None:
    Draft202012Validator(load_free_pen_tool_schema()).validate(normalize_free_pen_tool_response(response))


def normalize_free_pen_tool_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("Free pen tool response must be a dict")
    normalized = dict(response)
    decision = str(normalized.get("decision", "")).strip().lower()
    normalized["decision"] = decision
    if decision == "tool_call":
        tool_call = normalized.get("tool_call")
        if not isinstance(tool_call, dict):
            raise ValueError("tool_call decision requires a tool_call object")
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

    _SMOOTH_REASON_HINTS = ("curve", "curved", "oval", "ellipse", "circle", "arc", "smooth")
    _OVERLAY_CONFUSION_HINTS = ("orange line", "overlay", "previous line", "current drawing", "draft line")

    def run(self, source_image_path: Path, output_dir: Path) -> FreePenToolRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_image = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise ValueError(f"failed to load source image: {source_image_path}")
        height, width = source_image.shape[:2]
        canvas = FreePenCanvasState(width=int(width), height=int(height))
        response_files: list[Path] = []
        overlay_files: list[Path] = []
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

        for step_index in range(1, max(1, int(self.max_steps)) + 1):
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
                successful_step_count=len(successful_drawing_tool_calls),
                invalid_step_count=invalid_step_count,
                recent_history=tuple(self._recent_history_summary(history)),
                current_feedback=tuple(current_feedback),
            )
            prompt = build_free_pen_tool_prompt(review_input.prompt_input())
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
            self._record_interaction(
                {
                    "interaction_id": interaction_id,
                    "provider": self.provider_name,
                    "model": self.provider_model,
                    "status": "request_sent",
                    "step_index": step_index,
                    "prompt": prompt,
                    "prompt_char_count": len(prompt),
                    "image_paths": [str(source_image_path)] + ([] if previous_overlay_path is None else [str(previous_overlay_path)]),
                    "image_file_count": 1 if previous_overlay_path is None else 2,
                    "canvas_width": width,
                    "canvas_height": height,
                    "review_input_summary": review_input.prompt_input().to_payload(),
                }
            )
            try:
                raw_response = self.adapter.review(prompt, review_input)
                self._record_raw_response(step_index, raw_response)
                normalized_response = normalize_free_pen_tool_response(raw_response)
                validate_free_pen_tool_response(normalized_response)
                final_decision = str(normalized_response["decision"])
                final_reason = str(normalized_response.get("reason") or "").strip() or None
                if final_decision == "tool_call":
                    tool_call = dict(normalized_response["tool_call"])
                    preflight_result = self._preflight_tool_call(
                        tool_call=tool_call,
                        ai_reason=final_reason or "",
                        canvas=canvas,
                        successful_drawing_step_count=len(successful_drawing_tool_calls),
                        rollback_count=rollback_count,
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
                        round_status = "tool_applied"
                elif final_decision == "finish":
                    finish_warning = self._warning(
                        code="finish_with_open_path",
                        message="The current path is still open. Continue drawing, close_path when appropriate, or rollback.",
                    )
                    if self.closed_contour_mode and canvas.path_open:
                        warnings = [finish_warning]
                        rejected_step_count += 1
                        round_status = "rejected_action"
                        canvas.final_status = "invalid_finish"
                        quality = "bad"
                        quality_summary = finish_warning["message"]
                        runtime_description = "Rejected finish because the current path is still open."
                        current_feedback = self._feedback_from_rejected_action(warnings=warnings, quality_summary=quality_summary)
                    else:
                        round_status = "finished"
                        canvas.final_status = "finish"
                        quality = "good"
                        quality_summary = "Finished tracing without validation warnings."
                        runtime_description = "Finished the tracing loop."
                elif final_decision == "stalled":
                    round_status = "stalled"
                    canvas.final_status = "stalled"
                    quality = "warning"
                    quality_summary = "The model reported that it could not continue reliably."
                    runtime_description = "Stopped the tracing loop because the model returned stalled."
                else:
                    raise ValueError(f"unsupported free-pen tool decision: {final_decision}")
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

            overlay = canvas.render_overlay(
                stroke_width=max(1, int(self.stroke_width)),
                stroke_rgba=self.stroke_rgba,
                sample_count_per_segment=max(8, int(self.sample_count_per_segment)),
            )
            overlay_path = output_dir / f"round_{step_index:03d}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            overlay_files.append(overlay_path)
            previous_overlay_path = overlay_path

            response_files.append(
                FreePenRuntime._write_round_response(
                    output_dir=output_dir,
                    round_index=step_index,
                    prompt=prompt,
                    raw_response=raw_response,
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
                    "raw_response": raw_response,
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
                    "reason": final_reason,
                    "output_overlay_path": str(overlay_path),
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

    def _preflight_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        ai_reason: str,
        canvas: FreePenCanvasState,
        successful_drawing_step_count: int,
        rollback_count: int,
    ) -> dict[str, Any]:
        tool = str(tool_call["tool"])
        warnings = self._reason_based_warnings(ai_reason=ai_reason, tool=tool)
        if tool in {"start_path", "line_to", "restart_path"}:
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
        reject_codes = {warning["code"] for warning in warnings if warning["code"] in {"non_finite_coordinate", "out_of_bounds_coordinate"}}
        if tool in {"line_to", "curve_to", "close_path"} and not canvas.path_open:
            reject_codes.add("path_not_open")
            warnings.append(self._warning("path_not_open", f"{tool} requires an open path started by start_path."))
        if tool == "start_path" and canvas.path_open:
            reject_codes.add("path_already_open")
            warnings.append(self._warning("path_already_open", "A path is already open. Use restart_path, rollback_to_step, or close_path first."))
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

        if reject_codes:
            return {
                "success": False,
                "rejected": True,
                "warnings": warnings,
                "runtime_description": f"Rejected {tool} during preflight validation.",
                "quality_summary": " ; ".join(warning["message"] for warning in warnings),
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
            canvas.replay_tool_calls(successful_drawing_tool_calls)
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
            canvas.replay_tool_calls(successful_drawing_tool_calls)
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
            canvas.replay_tool_calls(())
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
                    "The model described a smooth curve but used line_to, which draws a straight segment.",
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

    def _record_interaction(self, payload: dict[str, Any]) -> None:
        if self.interaction_logger is not None:
            self.interaction_logger(payload)

    def _record_raw_response(self, step_index: int, raw_response: Any) -> None:
        if self.raw_response_logger is not None:
            self.raw_response_logger(step_index, raw_response)


__all__ = [
    "FileSequenceFreePenAdapter",
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
