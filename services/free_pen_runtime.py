from __future__ import annotations

import json
import hashlib
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
    final_decision: str | None
    final_reason: str | None
    error_message: str | None = None
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
        final_decision: str | None = None
        final_reason: str | None = None
        error_message: str | None = None
        previous_overlay_path: Path | None = None

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
                successful_step_count=canvas.successful_step_count,
                invalid_step_count=canvas.invalid_step_count,
            )
            prompt = build_free_pen_tool_prompt(review_input.prompt_input())
            interaction_id = f"free_pen_tool_{uuid4().hex}"
            raw_response: Any = None
            normalized_response: dict[str, Any] | None = None
            validation_error: str | None = None
            executed_tool_call: dict[str, Any] | None = None
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
                    executed_tool_call = canvas.apply_tool_call(dict(normalized_response["tool_call"]))
                    round_status = "tool_applied"
                elif final_decision == "finish":
                    round_status = "finished"
                    canvas.final_status = "finish"
                elif final_decision == "stalled":
                    round_status = "stalled"
                    canvas.final_status = "stalled"
                else:
                    raise ValueError(f"unsupported free-pen tool decision: {final_decision}")
            except Exception as exc:
                validation_error = str(exc)
                error_message = validation_error
                round_status = "invalid_response"
                canvas.final_status = "invalid_response"

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
            trace_rounds.append(
                {
                    "step_index": step_index,
                    "raw_response": raw_response,
                    "parsed_decision": final_decision,
                    "validation_result": {
                        "success": validation_error is None,
                        "error": validation_error,
                    },
                    "executed_tool_call": executed_tool_call,
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
                    "executed_tool_call": executed_tool_call,
                    "final_decision": final_decision,
                }
            )

            if validation_error is not None or final_decision in {"finish", "stalled"}:
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
                    "successful_step_count": canvas.successful_step_count,
                    "invalid_step_count": canvas.invalid_step_count,
                    "final_status": canvas.final_status,
                    "rounds": trace_rounds,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        status = canvas.final_status
        if final_decision == "finish":
            status = "finished"
        elif final_decision == "stalled":
            status = "stalled"
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
            successful_step_count=canvas.successful_step_count,
            invalid_step_count=canvas.invalid_step_count,
            final_decision=final_decision,
            final_reason=final_reason,
            error_message=error_message,
            response_files=tuple(response_files),
            overlay_files=tuple(overlay_files),
        )

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
