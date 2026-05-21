from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


class VisionReviewAdapter(Protocol):
    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        ...


class ResponderVisionAdapter:
    def __init__(self, responder: Callable[[str, AIReviewInput], dict[str, Any]]) -> None:
        self._responder = responder

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        return dict(self._responder(prompt, review_input))


__all__ = ["ResponderVisionAdapter", "VisionReviewAdapter"]
