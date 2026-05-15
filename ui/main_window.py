from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.types import VectorDocument
from services.ai_agent import AIReviewInput, AIReviewOutput, AIReviewService
from services.minimal_pipeline import MinimalPipeline, MinimalPipelineResult
from ui.canvas_widget import CanvasWidget
from ui.canvas_widget import AISuggestionOverlay


@dataclass(frozen=True, slots=True)
class AIReviewDisplayState:
    summary: str = ""
    issues: tuple[dict[str, Any], ...] = ()
    proposed_commands: tuple[dict[str, Any], ...] = ()
    suggestion_overlays: tuple[AISuggestionOverlay, ...] = ()


class MainWindow:
    def __init__(
        self,
        *,
        ai_review_service: AIReviewService | None = None,
        pipeline: MinimalPipeline | None = None,
        canvas_widget: CanvasWidget | None = None,
    ) -> None:
        self.ai_review_service = ai_review_service
        self.pipeline = pipeline or MinimalPipeline()
        self.canvas_widget = canvas_widget or CanvasWidget()
        self.review_display_state = AIReviewDisplayState()
        self.selected_image_path: str | None = None
        self.pipeline_result: MinimalPipelineResult | None = None
        self.last_error: str | None = None
        self.last_json_export_path: str | None = None
        self.last_overlay_export_path: str | None = None
        self.last_distance_field_diff_export_path: str | None = None
        self.last_review_input: AIReviewInput | None = None
        self.last_review_output: AIReviewOutput | None = None
        self.executed_commands: tuple[dict[str, Any], ...] = ()

    def select_image(self, image_path: str | Path) -> str:
        try:
            self.pipeline.load_image(image_path)
            self.selected_image_path = str(image_path)
            self.pipeline_result = None
            self.review_display_state = AIReviewDisplayState()
            self.canvas_widget.set_image_path(self.selected_image_path)
            self.canvas_widget.set_document(None)
            self.canvas_widget.set_review_display(summary="", issues=(), proposed_commands=())
            self._clear_error()
            return self.selected_image_path
        except Exception as exc:
            self._set_error(exc)
            raise

    def trigger_auto_fit(self, *, document_id: str = "document_1") -> MinimalPipelineResult:
        if self.selected_image_path is None:
            exc = ValueError("no image selected")
            self._set_error(exc)
            raise exc

        try:
            result = self.pipeline.run_from_file(self.selected_image_path, document_id=document_id)
            self.pipeline_result = result
            self.review_display_state = AIReviewDisplayState()
            self.canvas_widget.set_document(result.document)
            self.canvas_widget.set_review_display(summary="", issues=(), proposed_commands=())
            self._clear_error()
            return result
        except Exception as exc:
            self._set_error(exc)
            raise

    def trigger_ai_review_for_current_document(
        self,
        *,
        available_tools: tuple[str, ...] = (
            "propose_replace_segment_with_line",
            "propose_replace_segment_with_arc",
            "propose_replace_segment_with_circle",
            "propose_replace_segment_with_ellipse",
            "propose_batch_refinement",
        ),
        alpha_notes: str | None = None,
        color_notes: str | None = None,
    ) -> AIReviewDisplayState:
        if self.ai_review_service is None:
            exc = RuntimeError("AI review service is not configured")
            self._set_error(exc)
            raise exc
        if self.pipeline_result is None or self.selected_image_path is None:
            exc = ValueError("auto fit must run before AI review")
            self._set_error(exc)
            raise exc

        try:
            review_artifacts = self._ensure_review_artifacts()
            display_state = self.trigger_ai_review(
                original_image=self.selected_image_path,
                overlay_image=review_artifacts["overlay"],
                distance_field_diff_image=review_artifacts["distance_field_diff"],
                vector_document_json=json.loads(review_artifacts["json_payload"]),
                fit_error=self._aggregate_fit_error(self.pipeline_result.document),
                complexity_score=self._aggregate_complexity(self.pipeline_result.document),
                topology_status=self._aggregate_topology_status(self.pipeline_result.document),
                self_intersection_count=self._aggregate_self_intersections(self.pipeline_result.document),
                coordinate_system=asdict(self.pipeline_result.document.coordinate_system),
                available_tools=available_tools,
                alpha_notes=alpha_notes,
                color_notes=color_notes,
            )
            self._clear_error()
            return display_state
        except Exception as exc:
            self._set_error(exc)
            raise

    def export_json(self, output_path: str | Path) -> str:
        document = self._require_document()
        try:
            payload = self.pipeline.export_json(document, output_path)
            self.last_json_export_path = str(output_path)
            self._clear_error()
            return payload
        except Exception as exc:
            self._set_error(exc)
            raise

    def export_overlay(self, output_path: str | Path) -> str:
        document = self._require_document()
        image = self._require_source_image()
        try:
            self.pipeline.export_overlay(document, image, output_path)
            self.last_overlay_export_path = str(output_path)
            self._clear_error()
            return self.last_overlay_export_path
        except Exception as exc:
            self._set_error(exc)
            raise

    def export_distance_field_diff(self, output_path: str | Path) -> str:
        document = self._require_document()
        try:
            self.pipeline.export_distance_field_diff(document, output_path)
            self.last_distance_field_diff_export_path = str(output_path)
            self._clear_error()
            return self.last_distance_field_diff_export_path
        except Exception as exc:
            self._set_error(exc)
            raise

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
        if self.ai_review_service is None:
            raise RuntimeError("AI review service is not configured")

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
        suggestion_overlays = self.canvas_widget.set_review_display(
            summary=review_output.summary,
            issues=review_output.issues,
            proposed_commands=review_output.proposed_commands,
        )
        self.review_display_state = AIReviewDisplayState(
            summary=review_output.summary,
            issues=review_output.issues,
            proposed_commands=review_output.proposed_commands,
            suggestion_overlays=suggestion_overlays,
        )
        return self.review_display_state

    def _require_document(self) -> VectorDocument:
        if self.pipeline_result is None:
            exc = ValueError("auto fit must run before export")
            self._set_error(exc)
            raise exc
        return self.pipeline_result.document

    def _require_source_image(self) -> np.ndarray:
        if self.pipeline_result is not None and self.pipeline_result.source_image is not None:
            return self.pipeline_result.source_image
        if self.selected_image_path is None:
            exc = ValueError("no image selected")
            self._set_error(exc)
            raise exc
        return self.pipeline.load_image(self.selected_image_path)

    def _ensure_review_artifacts(self) -> dict[str, str]:
        document = self._require_document()
        image = self._require_source_image()
        artifact_dir = Path(tempfile.gettempdir()) / "aiteamtest_ui_review"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        document_id = document.document_id
        json_path = artifact_dir / f"{document_id}.json"
        overlay_path = artifact_dir / f"{document_id}_overlay.png"
        diff_path = artifact_dir / f"{document_id}_diff.png"
        json_payload = self.pipeline.export_json(document, json_path)
        self.pipeline.export_overlay(document, image, overlay_path)
        self.pipeline.export_distance_field_diff(document, diff_path)
        self.last_json_export_path = str(json_path)
        self.last_overlay_export_path = str(overlay_path)
        self.last_distance_field_diff_export_path = str(diff_path)
        return {
            "json": str(json_path),
            "overlay": str(overlay_path),
            "distance_field_diff": str(diff_path),
            "json_payload": json_payload,
        }

    def _aggregate_fit_error(self, document: VectorDocument) -> float:
        errors = [float(segment.fit_error) for segment in document.segments if segment.fit_error is not None]
        if not errors:
            return 0.0
        return sum(errors) / len(errors)

    def _aggregate_complexity(self, document: VectorDocument) -> float:
        scores = [
            float(segment.complexity_score)
            for segment in document.segments
            if segment.complexity_score is not None
        ]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _aggregate_topology_status(self, document: VectorDocument) -> str:
        statuses = {path.topology_status for path in document.paths}
        if "topology_error" in statuses:
            return "topology_error"
        if statuses == {"closed"}:
            return "closed"
        if "open" in statuses:
            return "open"
        return next(iter(statuses), "open")

    def _aggregate_self_intersections(self, document: VectorDocument) -> int:
        return sum(int(path.self_intersection_count) for path in document.paths)

    def _set_error(self, exc: Exception) -> None:
        self.last_error = str(exc)

    def _clear_error(self) -> None:
        self.last_error = None
