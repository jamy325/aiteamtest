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
    extract_text_value,
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

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        client = self._resolve_client()
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_path in collect_image_paths(review_input, max_image_bytes=self.max_image_bytes):
            content.append(
                {
                    "type": "input_image",
                    "image_url": encode_image_as_data_url(image_path, max_image_bytes=self.max_image_bytes),
                    "detail": self.image_detail,
                }
            )
        response = client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ai_review_response",
                    "schema": load_response_schema(),
                    "strict": True,
                }
            },
        )
        response_text = extract_text_value(response, provider_name="openai", attr_names=("output_text", "text"))
        return parse_json_response_text(response_text, provider_name="openai")

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
