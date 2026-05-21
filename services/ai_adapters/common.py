from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "ai_commands.schema.json"


class ProviderConfigurationError(RuntimeError):
    pass


def load_response_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def resolve_api_key(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def collect_image_paths(review_input: AIReviewInput) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw_path in (
        review_input.original_image,
        review_input.overlay_image,
        review_input.distance_field_diff_image,
    ):
        if raw_path is None:
            continue
        image_path = Path(raw_path)
        if not image_path.exists():
            raise ValueError(f"review image does not exist: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"review image is not a file: {image_path}")
        paths.append(image_path)
    return tuple(paths)


def encode_image_as_data_url(image_path: Path) -> str:
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
    "SCHEMA_PATH",
    "collect_image_paths",
    "encode_image_as_data_url",
    "extract_text_value",
    "load_response_schema",
    "parse_json_response_text",
    "resolve_api_key",
]
