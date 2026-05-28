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
    get_review_messages,
    load_response_schema,
    parse_json_response_text,
    resolve_api_key,
)

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


@dataclass(slots=True)
class SiliconFlowVisionAdapter(VisionReviewAdapter):
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    api_key: str | None = None
    base_url: str = "https://api.siliconflow.cn/v1"
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
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image_as_data_url(image_path, max_image_bytes=self.max_image_bytes),
                            "detail": self.image_detail,
                        },
                    }
                )
            content.append({"type": "text", "text": prompt})
            request_messages = [{"role": "user", "content": content}]
        response_schema = self.response_schema or load_response_schema(self.response_schema_path)
        response = client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ai_review_response",
                    "schema": response_schema,
                    "strict": True,
                },
            },
            timeout=self.timeout_seconds,
        )
        response_text = self._extract_message_content(response)
        return parse_json_response_text(response_text, provider_name="siliconflow")

    def _convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = message.get("content")
        if isinstance(content, str):
            return {"role": role, "content": [{"type": "text", "text": content}]}
        if not isinstance(content, list):
            raise ValueError("SiliconFlow review_input.messages content must be a string or list")
        converted_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("SiliconFlow review_input.messages parts must be dicts")
            part_type = str(part.get("type", "")).strip().lower()
            if part_type == "text":
                converted_parts.append({"type": "text", "text": str(part.get("text", ""))})
                continue
            if part_type == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
                    raise ValueError("SiliconFlow image_url parts must contain image_url.url")
                converted_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(image_url["url"]),
                            "detail": str(image_url.get("detail", self.image_detail)),
                        },
                    }
                )
                continue
            raise ValueError(f"unsupported SiliconFlow message part type: {part_type}")
        return {"role": role, "content": converted_parts}

    def _resolve_client(self) -> Any:
        if self.client is not None:
            return self.client

        api_key = self.api_key or resolve_api_key("SILICONFLOW_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "SiliconFlow provider requires SILICONFLOW_API_KEY or an explicit api_key"
            )

        try:
            module = importlib.import_module("openai")
            client_class = getattr(module, "OpenAI")
        except (ImportError, AttributeError) as exc:
            raise ProviderConfigurationError(
                "SiliconFlow provider requires the optional `openai` package"
            ) from exc

        return client_class(api_key=api_key, base_url=self.base_url)

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("siliconflow provider response does not contain choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError("siliconflow provider response does not contain a message")

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        raise ValueError("siliconflow provider response does not contain text content")


__all__ = ["SiliconFlowVisionAdapter"]
