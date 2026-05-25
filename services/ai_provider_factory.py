from __future__ import annotations

from pathlib import Path
from typing import Any

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.provider_factory import create_vision_adapter as create_live_vision_adapter
from services.ai_recorded_provider import DEFAULT_RECORDED_FIXTURE_DIR, RecordedVisionProvider


def create_vision_adapter(provider: str, **kwargs: Any) -> VisionReviewAdapter:
    recorded_mode = kwargs.pop("recorded_mode", None)
    if recorded_mode is None:
        return create_live_vision_adapter(provider, **kwargs)

    normalized_mode = str(recorded_mode).strip().lower()
    if normalized_mode not in {"record", "replay"}:
        raise ValueError("recorded_mode must be `record` or `replay`")

    fixture_path = kwargs.pop("fixture_path", None)
    fixtures_dir = Path(kwargs.pop("fixtures_dir", DEFAULT_RECORDED_FIXTURE_DIR))
    allow_live = bool(kwargs.pop("allow_live", False))
    live_adapter = kwargs.pop("live_adapter", None)
    model = str(kwargs.get("model") or "")

    if normalized_mode == "record" and live_adapter is None:
        live_adapter = create_live_vision_adapter(provider, **kwargs)

    return RecordedVisionProvider(
        provider_name=provider,
        model=model or getattr(live_adapter, "model", "unknown-model"),
        mode=normalized_mode,
        live_adapter=live_adapter,
        fixtures_dir=fixtures_dir,
        fixture_path=fixture_path,
        allow_live=allow_live,
    )


__all__ = ["create_vision_adapter"]
