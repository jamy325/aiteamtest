from __future__ import annotations

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment, ShapeCandidate
from services.ai_agent import validate_ai_review_response
from services.command_schema import validate_command
from services.proposed_command_planner import ProposedCommandPlanner, ProposedCommandPlannerConfig


def _planner_document() -> object:
    document = create_document(
        document_id="doc_planner",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    for path in (
        VectorPath(path_id="circle_path", closed=True, segments=("circle_0", "circle_1", "circle_2", "circle_3")),
        VectorPath(path_id="ellipse_path", closed=True, segments=("ellipse_0", "ellipse_1")),
        VectorPath(path_id="rect_path", closed=True, segments=("rect_0", "rect_1", "rect_2", "rect_3")),
        VectorPath(path_id="line_path", closed=False, segments=("line_0", "line_1")),
        VectorPath(path_id="arc_path", closed=False, segments=("arc_0",)),
    ):
        document = add_path(document, path)

    for segment in (
        Segment(segment_id="circle_0", path_id="circle_path", type="line", params={"start": [0.0, 0.0], "end": [1.0, 0.0]}),
        Segment(segment_id="circle_1", path_id="circle_path", type="line", params={"start": [1.0, 0.0], "end": [1.0, 1.0]}),
        Segment(segment_id="circle_2", path_id="circle_path", type="line", params={"start": [1.0, 1.0], "end": [0.0, 1.0]}),
        Segment(segment_id="circle_3", path_id="circle_path", type="line", params={"start": [0.0, 1.0], "end": [0.0, 0.0]}),
        Segment(segment_id="ellipse_0", path_id="ellipse_path", type="polyline", params={"points": [[0.0, 0.0], [2.0, 1.0], [4.0, 0.0]]}),
        Segment(segment_id="ellipse_1", path_id="ellipse_path", type="polyline", params={"points": [[4.0, 0.0], [2.0, -1.0], [0.0, 0.0]]}),
        Segment(segment_id="rect_0", path_id="rect_path", type="line", params={"start": [0.0, 0.0], "end": [4.0, 0.0]}),
        Segment(segment_id="rect_1", path_id="rect_path", type="line", params={"start": [4.0, 0.0], "end": [4.0, 3.0]}),
        Segment(segment_id="rect_2", path_id="rect_path", type="line", params={"start": [4.0, 3.0], "end": [0.0, 3.0]}),
        Segment(segment_id="rect_3", path_id="rect_path", type="line", params={"start": [0.0, 3.0], "end": [0.0, 0.0]}),
        Segment(segment_id="line_0", path_id="line_path", type="polyline", params={"points": [[0.0, 0.0], [3.0, 0.1], [6.0, 0.0]]}),
        Segment(segment_id="line_1", path_id="line_path", type="polyline", params={"points": [[6.0, 0.0], [9.0, -0.1], [12.0, 0.0]]}),
        Segment(segment_id="arc_0", path_id="arc_path", type="polyline", params={"points": [[0.0, 0.0], [2.0, 1.0], [4.0, 0.0]]}),
    ):
        document = add_segment(document, segment)
    return document


def test_proposed_command_planner_maps_candidates_to_valid_commands() -> None:
    document = _planner_document()
    candidates = (
        ShapeCandidate(
            candidate_id="cand_circle",
            target_type="circle",
            path_id="circle_path",
            segment_range=(0, 3),
            source="raw_contour_points",
            confidence=0.92,
            evidence={"model_complexity_delta": 3.0},
            reason="The loop reads as a clean circle.",
        ),
        ShapeCandidate(
            candidate_id="cand_ellipse",
            target_type="ellipse",
            path_id="ellipse_path",
            segment_range=(0, 1),
            source="raw_contour_points",
            confidence=0.88,
            evidence={"model_complexity_delta": 2.0},
            reason="The loop is better explained by an ellipse.",
        ),
        ShapeCandidate(
            candidate_id="cand_rectangle",
            target_type="rectangle",
            path_id="rect_path",
            segment_range=(0, 3),
            source="segment_samples_fallback",
            confidence=0.84,
            evidence={"model_complexity_delta": 2.0},
        ),
        ShapeCandidate(
            candidate_id="cand_line",
            target_type="line",
            path_id="line_path",
            segment_range=(0, 1),
            source="segment_samples_fallback",
            confidence=0.8,
            evidence={"model_complexity_delta": 1.0},
        ),
        ShapeCandidate(
            candidate_id="cand_arc",
            target_type="arc",
            path_id="arc_path",
            segment_range=(0, 0),
            source="raw_contour_points",
            confidence=0.79,
            evidence={"model_complexity_delta": 1.0},
        ),
    )

    commands = ProposedCommandPlanner().plan_commands(document, candidates)

    tools = [command["tool"] for command in commands]
    assert "propose_replace_path_with_circle" in tools
    assert "propose_replace_path_with_ellipse" in tools
    assert "propose_replace_segment_with_arc" in tools
    assert "propose_batch_refinement" in tools
    assert tools.count("propose_replace_segment_with_line") >= 5

    circle_command = next(command for command in commands if command["tool"] == "propose_replace_path_with_circle")
    assert circle_command["candidate_id"] == "cand_circle"
    assert circle_command["semantic_source"] == "shape_candidate_detector:raw_contour_points"

    rectangle_ranges = {
        tuple(command["segment_range"])
        for command in commands
        if command["tool"] == "propose_replace_segment_with_line" and command["path_id"] == "rect_path"
    }
    assert rectangle_ranges == {(0, 0), (1, 1), (2, 2), (3, 3)}

    for command in commands:
        validate_command(command, document)

    validate_ai_review_response(
        {
            "summary": "Planner output should be schema-valid.",
            "issues": [],
            "proposed_commands": list(commands),
        }
    )


def test_proposed_command_planner_dedupes_by_target_scope_and_uses_benefit_score() -> None:
    document = _planner_document()
    candidates = (
        ShapeCandidate(
            candidate_id="cand_circle",
            target_type="circle",
            path_id="circle_path",
            segment_range=(0, 3),
            source="raw_contour_points",
            confidence=0.82,
            evidence={"model_complexity_delta": 4.0},
        ),
        ShapeCandidate(
            candidate_id="cand_ellipse",
            target_type="ellipse",
            path_id="circle_path",
            segment_range=(0, 3),
            source="raw_contour_points",
            confidence=0.9,
            evidence={"model_complexity_delta": 0.0},
        ),
    )

    commands = ProposedCommandPlanner().plan_commands(document, candidates)
    non_batch_commands = [command for command in commands if command["tool"] != "propose_batch_refinement"]

    assert len(non_batch_commands) == 1
    assert non_batch_commands[0]["tool"] == "propose_replace_path_with_circle"
    assert non_batch_commands[0]["candidate_id"] == "cand_circle"


def test_proposed_command_planner_skips_very_low_confidence_candidates_and_keeps_reviewable_ones() -> None:
    document = _planner_document()
    candidates = (
        ShapeCandidate(
            candidate_id="cand_skip",
            target_type="line",
            path_id="line_path",
            segment_range=(0, 1),
            source="segment_samples_fallback",
            confidence=0.24,
            evidence={"model_complexity_delta": 1.0},
        ),
        ShapeCandidate(
            candidate_id="cand_keep",
            target_type="line",
            path_id="line_path",
            segment_range=(0, 1),
            source="segment_samples_fallback",
            confidence=0.55,
            evidence={"model_complexity_delta": 1.0},
        ),
    )

    commands = ProposedCommandPlanner(
        ProposedCommandPlannerConfig(
            min_candidate_confidence=0.3,
            include_batch_command=False,
        )
    ).plan_commands(document, candidates)

    assert len(commands) == 1
    assert commands[0]["candidate_id"] == "cand_keep"
    assert commands[0]["requires_user_confirmation"] is True
