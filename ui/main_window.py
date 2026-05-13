from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.ai_agent import AIReviewInput, AIReviewOutput, AIReviewService
from ui.canvas_widget import CanvasWidget


@dataclass(frozen=True, slots=True)
class AIReviewDisplayState:
    summary: str = ""
    issues: tuple[dict[str, Any], ...] = ()
    proposed_commands: tuple[dict[str, Any], ...] = ()


class MainWindow:
    def __init__(
        self,
        *,
        ai_review_service: AIReviewService,
        canvas_widget: CanvasWidget | None = None,
    ) -> None:
        self.ai_review_service = ai_review_service
        self.canvas_widget = canvas_widget or CanvasWidget()
        self.review_display_state = AIReviewDisplayState()
        self.last_review_input: AIReviewInput | None = None
        self.last_review_output: AIReviewOutput | None = None
        self.executed_commands: tuple[dict[str, Any], ...] = ()

    def trigger_ai_review(
        self,
        *,
        original_image: str | None,
        overlay_image: str | None,
        distance_field_diff_image: str | None,
        vector_document_json: dict[str, Any],
        fit_error: float,
        complexity_score: float,
        topology_status: str,
        self_intersection_count: int,
        coordinate_system: dict[str, Any],
        available_tools: tuple[str, ...] = (),
        alpha_notes: str | None = None,
        color_notes: str | None = None,
    ) -> AIReviewDisplayState:
        review_input = AIReviewInput(
            original_image=original_image,
            overlay_image=overlay_image,
            distance_field_diff_image=distance_field_diff_image,
            vector_document_json=vector_document_json,
            fit_error=float(fit_error),
            complexity_score=float(complexity_score),
            topology_status=str(topology_status),
            self_intersection_count=int(self_intersection_count),
            coordinate_system=dict(coordinate_system),
            user_locked_ids=self.canvas_widget.locked_ids,
            available_tools=tuple(str(tool) for tool in available_tools),
            alpha_notes=alpha_notes,
            color_notes=color_notes,
        )
        review_output = self.ai_review_service.run_review(review_input)
        self.last_review_input = review_input
        self.last_review_output = review_output
        self.review_display_state = AIReviewDisplayState(
            summary=review_output.summary,
            issues=review_output.issues,
            proposed_commands=review_output.proposed_commands,
        )
        return self.review_display_state
