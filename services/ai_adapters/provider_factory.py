from __future__ import annotations

from pathlib import Path
from typing import Any

from services.ai_adapters.base import ResponderVisionAdapter, VisionReviewAdapter
from services.ai_adapters.common import MAX_REVIEW_IMAGE_BYTES
from services.ai_adapters.file_response import FileResponseVisionAdapter
from services.ai_adapters.gemini_provider import GeminiVisionAdapter
from services.ai_adapters.mock import MockVisionAdapter
from services.ai_adapters.openai_provider import OpenAIVisionAdapter
from services.ai_adapters.siliconflow_provider import SiliconFlowVisionAdapter
from services.ai_recorded_provider import DEFAULT_RECORDED_FIXTURE_DIR, RecordedVisionProvider


def create_vision_adapter(provider: str, **kwargs: Any) -> VisionReviewAdapter:
    normalized_provider = str(provider).strip().lower()
    recorded_mode = kwargs.pop("recorded_mode", None)

    if recorded_mode is not None:
        normalized_mode = str(recorded_mode).strip().lower()
        if normalized_mode not in {"record", "replay"}:
            raise ValueError("recorded_mode must be `record` or `replay`")

        fixture_path = kwargs.pop("fixture_path", None)
        fixtures_dir = Path(kwargs.pop("fixtures_dir", DEFAULT_RECORDED_FIXTURE_DIR))
        allow_live = bool(kwargs.pop("allow_live", False))
        live_adapter = kwargs.pop("live_adapter", None)
        model = str(kwargs.get("model") or "")

        if normalized_mode == "record" and live_adapter is None:
            live_adapter = create_vision_adapter(provider, **kwargs)

        return RecordedVisionProvider(
            provider_name=provider,
            model=model or getattr(live_adapter, "model", "unknown-model"),
            mode=normalized_mode,
            live_adapter=live_adapter,
            fixtures_dir=fixtures_dir,
            fixture_path=fixture_path,
            allow_live=allow_live,
        )

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
            max_image_bytes=int(kwargs.get("max_image_bytes", MAX_REVIEW_IMAGE_BYTES)),
            timeout_seconds=None if kwargs.get("timeout_seconds") is None else float(kwargs.get("timeout_seconds")),
        )

    if normalized_provider == "gemini":
        return GeminiVisionAdapter(
            model=str(kwargs.get("model", "gemini-3.5-flash")),
            api_key=kwargs.get("api_key"),
            client=kwargs.get("client"),
            image_loader=kwargs.get("image_loader"),
            max_image_bytes=int(kwargs.get("max_image_bytes", MAX_REVIEW_IMAGE_BYTES)),
            timeout_seconds=None if kwargs.get("timeout_seconds") is None else float(kwargs.get("timeout_seconds")),
        )

    if normalized_provider == "siliconflow":
        return SiliconFlowVisionAdapter(
            model=str(kwargs.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")),
            api_key=kwargs.get("api_key"),
            base_url=str(kwargs.get("base_url", "https://api.siliconflow.cn/v1")),
            client=kwargs.get("client"),
            image_detail=str(kwargs.get("image_detail", "auto")),
            max_image_bytes=int(kwargs.get("max_image_bytes", MAX_REVIEW_IMAGE_BYTES)),
            timeout_seconds=None if kwargs.get("timeout_seconds") is None else float(kwargs.get("timeout_seconds")),
        )

    if normalized_provider == "responder":
        responder = kwargs.get("responder")
        if responder is None:
            raise ValueError("responder provider requires `responder`")
        return ResponderVisionAdapter(responder)

    raise ValueError(f"unsupported vision adapter provider: {provider}")


__all__ = ["create_vision_adapter"]
