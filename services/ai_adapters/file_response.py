from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.ai_adapters.base import VisionReviewAdapter


@dataclass(frozen=True, slots=True)
class FileResponseVisionAdapter(VisionReviewAdapter):
    response_path: Path

    def review(self, prompt: str, review_input: object) -> dict[str, Any]:
        payload = json.loads(Path(self.response_path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("file response adapter expects a top-level JSON object")
        return dict(payload)


__all__ = ["FileResponseVisionAdapter"]
