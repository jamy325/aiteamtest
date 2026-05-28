from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "ai_commands.schema.json"
MAX_REVIEW_IMAGE_BYTES = 20 * 1024 * 1024


class ProviderConfigurationError(RuntimeError):
    pass


def load_response_schema(schema_path: Path | None = None) -> dict[str, Any]:
    resolved_path = SCHEMA_PATH if schema_path is None else Path(schema_path)
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def resolve_api_key(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def collect_image_paths(
    review_input: AIReviewInput,
    *,
    max_image_bytes: int = MAX_REVIEW_IMAGE_BYTES,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw_path in (
        review_input.original_image,
        review_input.overlay_image,
        review_input.distance_field_diff_image,
    ):
        if raw_path is None:
            continue
        image_path = Path(raw_path)
        ensure_review_image_path(image_path, max_image_bytes=max_image_bytes)
        paths.append(image_path)
    return tuple(paths)


def ensure_review_image_path(image_path: Path, *, max_image_bytes: int = MAX_REVIEW_IMAGE_BYTES) -> Path:
    if not image_path.exists():
        raise ValueError(f"review image does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"review image is not a file: {image_path}")
    file_size = image_path.stat().st_size
    if file_size > int(max_image_bytes):
        raise ValueError(
            f"review image exceeds size limit: {image_path} ({file_size} bytes > {int(max_image_bytes)} bytes)"
        )
    return image_path


def encode_image_as_data_url(image_path: Path, *, max_image_bytes: int = MAX_REVIEW_IMAGE_BYTES) -> str:
    ensure_review_image_path(image_path, max_image_bytes=max_image_bytes)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def parse_json_response_text(text: str, *, provider_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{provider_name} provider did not return valid JSON text") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{provider_name} provider must return a top-level JSON object")
    return payload


def get_review_messages(review_input: object) -> tuple[dict[str, Any], ...] | None:
    messages = getattr(review_input, "messages", None)
    if messages is None:
        return None
    if not isinstance(messages, (list, tuple)):
        raise ValueError("review_input.messages must be a list or tuple")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each review_input.messages entry must be a dict")
        normalized.append(dict(message))
    return tuple(normalized)


def get_message_image_urls(messages: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    image_urls: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str) and url:
                image_urls.append(url)
    return tuple(image_urls)


def extract_text_value(response: Any, *, provider_name: str, attr_names: tuple[str, ...]) -> str:
    for attr_name in attr_names:
        if isinstance(response, dict):
            value = response.get(attr_name)
        else:
            value = getattr(response, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            for attr_name in attr_names:
                value = dumped.get(attr_name)
                if isinstance(value, str) and value.strip():
                    return value

    raise ValueError(f"{provider_name} provider response does not contain text output")


__all__ = [
    "ProviderConfigurationError",
    "MAX_REVIEW_IMAGE_BYTES",
    "SCHEMA_PATH",
    "collect_image_paths",
    "ensure_review_image_path",
    "encode_image_as_data_url",
    "extract_text_value",
    "get_message_image_urls",
    "get_review_messages",
    "load_response_schema",
    "parse_json_response_text",
    "resolve_api_key",
]
