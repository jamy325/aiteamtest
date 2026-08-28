from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.common import (
    MAX_REVIEW_IMAGE_BYTES,
    ProviderConfigurationError,
    collect_image_paths,
    encode_image_as_data_url,
    extract_text_value,
    get_review_messages,
    load_response_schema,
    parse_json_response_text,
    resolve_api_key,
)

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


@dataclass(slots=True)
class OpenAIVisionAdapter(VisionReviewAdapter):
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    client: Any | None = None
    image_detail: str = "auto"
    max_image_bytes: int = MAX_REVIEW_IMAGE_BYTES
    timeout_seconds: float | None = None
    response_schema: dict[str, Any] | None = None
    response_schema_path: Path | None = None

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        client = self._resolve_client()
        review_messages = get_review_messages(review_input)
        if review_messages is not None:
            request_messages = [self._convert_message(message) for message in review_messages]
        else:
            content: list[dict[str, Any]] = []
            for image_path in collect_image_paths(review_input, max_image_bytes=self.max_image_bytes):
                content.append(
                    {
                        "type": "input_image",
                        "image_url": encode_image_as_data_url(image_path, max_image_bytes=self.max_image_bytes),
                        "detail": self.image_detail,
                    }
                )
            content.append({"type": "input_text", "text": prompt})
            request_messages = [{"role": "user", "content": content}]
        response_schema = self.response_schema or load_response_schema(self.response_schema_path)
        response = client.responses.create(
            model=self.model,
            input=request_messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ai_review_response",
                    "schema": response_schema,
                    "strict": True,
                }
            },
            timeout=self.timeout_seconds,
        )
        response_text = extract_text_value(response, provider_name="openai", attr_names=("output_text", "text"))
        return parse_json_response_text(response_text, provider_name="openai")

    def _convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = message.get("content")
        if isinstance(content, str):
            return {"role": role, "content": [{"type": "input_text", "text": content}]}
        if not isinstance(content, list):
            raise ValueError("OpenAI review_input.messages content must be a string or list")
        converted_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("OpenAI review_input.messages parts must be dicts")
            part_type = str(part.get("type", "")).strip().lower()
            if part_type == "text":
                converted_parts.append({"type": "input_text", "text": str(part.get("text", ""))})
                continue
            if part_type == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
                    raise ValueError("OpenAI image_url parts must contain image_url.url")
                converted_parts.append(
                    {
                        "type": "input_image",
                        "image_url": str(image_url["url"]),
                        "detail": str(image_url.get("detail", self.image_detail)),
                    }
                )
                continue
            raise ValueError(f"unsupported OpenAI message part type: {part_type}")
        return {"role": role, "content": converted_parts}

    def _resolve_client(self) -> Any:
        if self.client is not None:
            return self.client

        api_key = self.api_key or resolve_api_key("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("OpenAI provider requires OPENAI_API_KEY or an explicit api_key")

        try:
            module = importlib.import_module("openai")
            client_class = getattr(module, "OpenAI")
        except (ImportError, AttributeError) as exc:
            raise ProviderConfigurationError("OpenAI provider requires the optional `openai` package") from exc

        return client_class(api_key=api_key)


__all__ = ["OpenAIVisionAdapter"]
