from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from services.ai_adapters.base import VisionReviewAdapter


@dataclass(frozen=True, slots=True)
class MockVisionAdapter(VisionReviewAdapter):
    response: dict[str, Any]

    def review(self, prompt: str, review_input: object) -> dict[str, Any]:
        return json.loads(json.dumps(self.response))


__all__ = ["MockVisionAdapter"]
