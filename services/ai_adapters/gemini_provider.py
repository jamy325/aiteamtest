from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.common import (
    MAX_REVIEW_IMAGE_BYTES,
    ProviderConfigurationError,
    collect_image_paths,
    ensure_review_image_path,
    extract_text_value,
    get_review_messages,
    load_response_schema,
    parse_json_response_text,
    resolve_api_key,
)

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


@dataclass(slots=True)
class GeminiVisionAdapter(VisionReviewAdapter):
    model: str = "gemini-3.5-flash"
    api_key: str | None = None
    client: Any | None = None
    image_loader: Callable[[Path], Any] | None = None
    max_image_bytes: int = MAX_REVIEW_IMAGE_BYTES
    timeout_seconds: float | None = None
    response_schema: dict[str, Any] | None = None
    response_schema_path: Path | None = None

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        client = self._resolve_client()
        review_messages = get_review_messages(review_input)
        if review_messages is not None:
            contents = [self._convert_message(message) for message in review_messages]
        else:
            contents: list[Any] = []
            for image_path in collect_image_paths(review_input, max_image_bytes=self.max_image_bytes):
                contents.append(self._load_image(image_path))
            contents.append(prompt)
        response_schema = self.response_schema or load_response_schema(self.response_schema_path)
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": response_schema,
            },
        )
        response_text = extract_text_value(response, provider_name="gemini", attr_names=("text", "output_text"))
        return parse_json_response_text(response_text, provider_name="gemini")

    def _convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = message.get("content")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"text": content})
            return {"role": role, "parts": parts}
        if not isinstance(content, list):
            raise ValueError("Gemini review_input.messages content must be a string or list")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Gemini review_input.messages parts must be dicts")
            part_type = str(part.get("type", "")).strip().lower()
            if part_type == "text":
                parts.append({"text": str(part.get("text", ""))})
                continue
            if part_type == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
                    raise ValueError("Gemini image_url parts must contain image_url.url")
                parts.append(
                    {
                        "file_data": {
                            "file_uri": str(image_url["url"]),
                            "mime_type": str(image_url.get("mime_type", "image/png")),
                        }
                    }
                )
                continue
            raise ValueError(f"unsupported Gemini message part type: {part_type}")
        return {"role": role, "parts": parts}

    def _resolve_client(self) -> Any:
        if self.client is not None:
            return self.client

        api_key = self.api_key or resolve_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "Gemini provider requires GEMINI_API_KEY, GOOGLE_API_KEY, or an explicit api_key"
            )

        try:
            module = importlib.import_module("google.genai")
            client_class = getattr(module, "Client")
        except (ImportError, AttributeError) as exc:
            raise ProviderConfigurationError("Gemini provider requires the optional `google-genai` package") from exc

        return client_class(api_key=api_key)

    def _load_image(self, image_path: Path) -> Any:
        ensure_review_image_path(image_path, max_image_bytes=self.max_image_bytes)
        if self.image_loader is not None:
            return self.image_loader(image_path)

        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise ProviderConfigurationError("Gemini provider image inputs require the optional `pillow` package") from exc

        return image_module.open(image_path)


__all__ = ["GeminiVisionAdapter"]
