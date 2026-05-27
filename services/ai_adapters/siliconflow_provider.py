from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.common import (
    MAX_REVIEW_IMAGE_BYTES,
    ProviderConfigurationError,
    collect_image_paths,
    encode_image_as_data_url,
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

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        client = self._resolve_client()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
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
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ai_review_response",
                    "schema": load_response_schema(),
                    "strict": True,
                },
            },
            timeout=self.timeout_seconds,
        )
        response_text = self._extract_message_content(response)
        return parse_json_response_text(response_text, provider_name="siliconflow")

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
