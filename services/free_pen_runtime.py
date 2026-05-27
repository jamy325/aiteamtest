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
from services.free_pen_prompt import FreePenPromptInput, build_free_pen_prompt


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "ai_free_pen.schema.json"


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


__all__ = [
    "FreePenReviewInput",
    "FreePenRunResult",
    "FreePenRuntime",
    "SCHEMA_PATH",
    "load_free_pen_schema",
    "normalize_free_pen_response",
    "validate_free_pen_response",
]
