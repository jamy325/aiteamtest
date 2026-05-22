from __future__ import annotations

from pathlib import Path
from typing import Any

from services.ai_adapters.base import ResponderVisionAdapter, VisionReviewAdapter
from services.ai_adapters.file_response import FileResponseVisionAdapter
from services.ai_adapters.gemini_provider import GeminiVisionAdapter
from services.ai_adapters.mock import MockVisionAdapter
from services.ai_adapters.openai_provider import OpenAIVisionAdapter
from services.ai_adapters.siliconflow_provider import SiliconFlowVisionAdapter


def create_vision_adapter(provider: str, **kwargs: Any) -> VisionReviewAdapter:
    normalized_provider = str(provider).strip().lower()

    if normalized_provider == "mock":
        if "response" not in kwargs:
            raise ValueError("mock provider requires `response`")
        return MockVisionAdapter(response=dict(kwargs["response"]))

    if normalized_provider == "file":
        if "response_path" not in kwargs:
            raise ValueError("file provider requires `response_path`")
        return FileResponseVisionAdapter(response_path=Path(kwargs["response_path"]))

    if normalized_provider == "openai":
        return OpenAIVisionAdapter(
            model=str(kwargs.get("model", "gpt-4.1-mini")),
            api_key=kwargs.get("api_key"),
            client=kwargs.get("client"),
            image_detail=str(kwargs.get("image_detail", "auto")),
        )

    if normalized_provider == "gemini":
        return GeminiVisionAdapter(
            model=str(kwargs.get("model", "gemini-3.5-flash")),
            api_key=kwargs.get("api_key"),
            client=kwargs.get("client"),
            image_loader=kwargs.get("image_loader"),
        )

    if normalized_provider == "siliconflow":
        return SiliconFlowVisionAdapter(
            model=str(kwargs.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")),
            api_key=kwargs.get("api_key"),
            base_url=str(kwargs.get("base_url", "https://api.siliconflow.cn/v1")),
            client=kwargs.get("client"),
            image_detail=str(kwargs.get("image_detail", "auto")),
        )

    if normalized_provider == "responder":
        responder = kwargs.get("responder")
        if responder is None:
            raise ValueError("responder provider requires `responder`")
        return ResponderVisionAdapter(responder)

    raise ValueError(f"unsupported vision adapter provider: {provider}")


__all__ = ["create_vision_adapter"]
