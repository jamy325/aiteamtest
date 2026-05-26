from __future__ import annotations

from typing import Any

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.provider_factory import create_vision_adapter as _canonical_create_vision_adapter


def create_vision_adapter(provider: str, **kwargs: Any) -> VisionReviewAdapter:
    return _canonical_create_vision_adapter(provider, **kwargs)


__all__ = ["create_vision_adapter"]
