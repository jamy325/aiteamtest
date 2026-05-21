from __future__ import annotations

import json
from pathlib import Path

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.auto_refinement_pipeline import AutoRefinementPipeline, AutoRefinementPipelineConfig
from services.minimal_pipeline import MinimalPipeline


def _rectangle_document():
    document = create_document(
        document_id="doc_rectangle_pipeline",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="rect_path",
            closed=True,
            segments=("rect_0", "rect_1", "rect_2", "rect_3"),
        ),
    )
    corners = ((20.0, 20.0), (120.0, 20.0), (120.0, 80.0), (20.0, 80.0))
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        document = add_segment(
            document,
            Segment(
                segment_id=f"rect_{index}",
                path_id="rect_path",
                type="line",
                params={"start": [start[0], start[1]], "end": [end[0], end[1]]},
            ),
        )
    return document


def test_auto_refinement_pipeline_circle_fixture_auto_accepts_circle_command() -> None:
    pipeline_result = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/circle/test_input_circle.png"),
        document_id="circle_auto_pipeline",
    )

    result = AutoRefinementPipeline().run_from_pipeline_result(
        pipeline_result,
        target_types=("circle",),
    )

    assert any(candidate.target_type == "circle" for candidate in result.candidates)
    assert any(command["tool"] == "propose_replace_path_with_circle" for command in result.proposed_commands)
    assert any(decision.decision == "auto_accept" for decision in result.preview_decisions)
    assert result.report.candidate_stats["by_target_type"]["circle"] >= 1
    assert result.report.decision_stats["auto_accept"] >= 1
    assert result.refined_document != pipeline_result.document


def test_auto_refinement_pipeline_ellipse_fixture_produces_reviewable_decision() -> None:
    pipeline_result = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/ellipse/test_input_ellipse.png"),
        document_id="ellipse_auto_pipeline",
    )

    result = AutoRefinementPipeline().run_from_pipeline_result(
        pipeline_result,
        target_types=("ellipse",),
    )

    assert any(candidate.target_type == "ellipse" for candidate in result.candidates)
    assert any(command["tool"] == "propose_replace_path_with_ellipse" for command in result.proposed_commands)
    assert result.preview_decisions
    assert {decision.decision for decision in result.preview_decisions} <= {"auto_accept", "user_confirm", "reject"}
    assert any(decision.decision in {"auto_accept", "user_confirm"} for decision in result.preview_decisions)


def test_auto_refinement_pipeline_rectangle_document_generates_line_or_batch_commands() -> None:
    document = _rectangle_document()

    result = AutoRefinementPipeline().run(
        document,
        target_types=("rectangle", "line"),
    )

    tools = {command["tool"] for command in result.proposed_commands}
    assert "propose_replace_segment_with_line" in tools or "propose_batch_refinement" in tools
    assert result.preview_decisions
    assert result.report.command_stats["total"] >= 1
    assert result.report.decision_stats["auto_accept"] + result.report.decision_stats["user_confirm"] + result.report.decision_stats["reject"] == len(result.preview_decisions)


def test_auto_refinement_pipeline_dry_run_only_keeps_original_document_and_serializes() -> None:
    pipeline_result = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/circle/test_input_circle.png"),
        document_id="circle_auto_dry_run",
    )

    result = AutoRefinementPipeline(
        config=AutoRefinementPipelineConfig(dry_run_only=True),
    ).run_from_pipeline_result(
        pipeline_result,
        target_types=("circle",),
    )

    payload = json.loads(result.to_json())

    assert result.refined_document == pipeline_result.document
    assert result.report.dry_run_only is True
    assert payload["report"]["dry_run_only"] is True
    assert payload["report"]["score_before"] >= 0.0
    assert payload["report"]["score_after"] >= 0.0
    assert payload["refined_document"]["document_id"] == "circle_auto_dry_run"
