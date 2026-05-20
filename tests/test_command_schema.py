import ast
from pathlib import Path

import pytest

from core.document import add_anchor, add_constraint, add_path, add_segment, create_document
from core.types import Anchor, Constraint, CoordinateSystem, Path as VectorPath, Segment
from services.command_schema import CommandValidationError, validate_command


def _document_for_command_validation() -> object:
    document = create_document(
        document_id="doc_command_schema",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    path = VectorPath(path_id="path_1")
    locked_path = VectorPath(path_id="path_locked", locked=True)
    document = add_path(document, path)
    document = add_path(document, locked_path)
    for anchor in (
        Anchor(anchor_id="anchor_1", path_id="path_1", position=(0.0, 0.0)),
        Anchor(anchor_id="anchor_2", path_id="path_1", position=(5.0, 0.0)),
        Anchor(anchor_id="anchor_3", path_id="path_1", position=(10.0, 0.0), locked=True),
    ):
        document = add_anchor(document, anchor)
    for segment in (
        Segment(
            segment_id="segment_1",
            path_id="path_1",
            type="line",
            params={"start": [0.0, 0.0], "end": [5.0, 0.0]},
            anchors=("anchor_1", "anchor_2"),
        ),
        Segment(
            segment_id="segment_2",
            path_id="path_1",
            type="line",
            params={"start": [5.0, 0.0], "end": [10.0, 0.0]},
            anchors=("anchor_2", "anchor_3"),
        ),
    ):
        document = add_segment(document, segment)
    document = add_constraint(
        document,
        Constraint(
            constraint_id="constraint_1",
            type="coincident",
            targets=("segment_2",),
            locked=True,
        ),
    )
    return document


def _closed_path_document_for_command_validation() -> object:
    document = create_document(
        document_id="doc_path_command_schema",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="closed_path",
            closed=True,
            segments=("closed_segment_1", "closed_segment_2"),
        ),
    )
    for anchor in (
        Anchor(anchor_id="closed_anchor_1", path_id="closed_path", position=(0.0, 0.0)),
        Anchor(anchor_id="closed_anchor_2", path_id="closed_path", position=(5.0, 5.0)),
    ):
        document = add_anchor(document, anchor)
    for segment in (
        Segment(
            segment_id="closed_segment_1",
            path_id="closed_path",
            type="polyline",
            params={"points": [[0.0, 0.0], [4.0, 1.0], [5.0, 5.0]]},
            anchors=("closed_anchor_1", "closed_anchor_2"),
        ),
        Segment(
            segment_id="closed_segment_2",
            path_id="closed_path",
            type="polyline",
            params={"points": [[5.0, 5.0], [1.0, 4.0], [0.0, 0.0]]},
            anchors=("closed_anchor_2", "closed_anchor_1"),
        ),
    ):
        document = add_segment(document, segment)
    return document


def test_validate_command_accepts_valid_line_intent() -> None:
    document = _document_for_command_validation()
    command = {
        "tool": "propose_replace_segment_with_line",
        "path_id": "path_1",
        "segment_range": [0, 0],
        "reason": "This edge should be straight.",
        "confidence": 0.8,
        "requires_user_confirmation": True,
        "locked_anchor_ids": ["anchor_1"],
    }

    result = validate_command(command, document)

    assert result.tool == "propose_replace_segment_with_line"
    assert result.target_path_id == "path_1"
    assert result.target_segment_ids == ("segment_1",)


def test_validate_command_accepts_path_level_circle_intent() -> None:
    document = _closed_path_document_for_command_validation()
    command = {
        "tool": "propose_replace_path_with_circle",
        "path_id": "closed_path",
        "reason": "The entire loop reads as a circle.",
        "confidence": 0.86,
        "requires_user_confirmation": True,
        "candidate_id": "cand_circle_1",
        "semantic_source": "ai_review",
        "semantic_confidence": 0.91,
    }

    result = validate_command(command, document)

    assert result.tool == "propose_replace_path_with_circle"
    assert result.target_path_id == "closed_path"
    assert result.target_segment_ids == ("closed_segment_1", "closed_segment_2")


def test_validate_command_rejects_unknown_tool() -> None:
    document = _document_for_command_validation()
    command = {
        "tool": "propose_replace_segment_with_bezier",
        "path_id": "path_1",
        "segment_range": [0, 0],
        "reason": "Unsupported tool.",
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }

    with pytest.raises(CommandValidationError, match="unknown tool"):
        validate_command(command, document)


def test_validate_command_rejects_missing_required_fields() -> None:
    document = _document_for_command_validation()
    command = {
        "tool": "propose_replace_segment_with_arc",
        "path_id": "path_1",
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }

    with pytest.raises(CommandValidationError, match="missing required fields"):
        validate_command(command, document)


def test_validate_command_rejects_locked_path_segment_anchor_and_constraint() -> None:
    document = _document_for_command_validation()

    with pytest.raises(CommandValidationError, match="locked path"):
        validate_command(
            {
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_locked",
                "segment_range": [0, 0],
                "reason": "Try replacing locked path.",
                "confidence": 0.8,
                "requires_user_confirmation": True,
            },
            document,
        )

    with pytest.raises(CommandValidationError, match="locked anchor|locked segment|locked constraint"):
        validate_command(
            {
                "tool": "propose_replace_segment_with_arc",
                "path_id": "path_1",
                "segment_range": [1, 1],
                "reason": "Touches locked anchor / constraint.",
                "confidence": 0.8,
                "requires_user_confirmation": True,
            },
            document,
        )

    with pytest.raises(CommandValidationError, match="locked anchor|locked constraint"):
        validate_command(
            {
                "tool": "propose_replace_segment_with_circle",
                "path_id": "path_1",
                "segment_range": [0, 1],
                "reason": "Touches locked segment.",
                "confidence": 0.8,
                "requires_user_confirmation": True,
            },
            document,
        )


def test_validate_command_rejects_precise_geometry_parameters() -> None:
    document = _document_for_command_validation()
    command = {
        "tool": "propose_replace_segment_with_ellipse",
        "path_id": "path_1",
        "segment_range": [0, 0],
        "reason": "This region appears elliptical.",
        "confidence": 0.7,
        "requires_user_confirmation": True,
        "cx": 12.0,
    }

    with pytest.raises(CommandValidationError, match="precise geometry parameter"):
        validate_command(command, document)


def test_validate_command_rejects_non_vector_coordinate_system() -> None:
    document = create_document(
        document_id="doc_bad_space",
        width=10.0,
        height=10.0,
        coordinate_system=CoordinateSystem(internal_space="pixel"),
    )
    document = add_path(document, VectorPath(path_id="path_1"))
    command = {
        "tool": "propose_replace_segment_with_line",
        "path_id": "path_1",
        "segment_range": [0, 0],
        "reason": "Should fail before execution.",
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }

    with pytest.raises(CommandValidationError, match="Vector Space"):
        validate_command(command, document)


def test_validate_command_accepts_batch_command() -> None:
    document = create_document(
        document_id="doc_batch",
        width=10.0,
        height=10.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_batch", segments=("seg_a",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_a",
            path_id="path_batch",
            type="line",
            params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
        ),
    )
    command = {
        "tool": "propose_batch_refinement",
        "summary": "Review primitive replacements together.",
        "commands": [
            {
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_batch",
                "segment_range": [0, 0],
                "reason": "Keep this straight.",
                "confidence": 0.9,
                "requires_user_confirmation": True,
            }
        ],
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }

    result = validate_command(command, document)

    assert result.tool == "propose_batch_refinement"
    assert result.target_segment_ids == ("seg_a",)


def test_validate_command_accepts_mixed_path_and_segment_batch_command() -> None:
    document = _closed_path_document_for_command_validation()
    document = add_path(document, VectorPath(path_id="path_batch", segments=("seg_a",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_a",
            path_id="path_batch",
            type="line",
            params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
        ),
    )
    command = {
        "tool": "propose_batch_refinement",
        "summary": "Review full-loop and local edge replacements together.",
        "commands": [
            {
                "tool": "propose_replace_path_with_ellipse",
                "path_id": "closed_path",
                "reason": "The loop reads as an ellipse.",
                "confidence": 0.82,
                "requires_user_confirmation": True,
                "candidate_id": "cand_ellipse_1",
                "semantic_source": "planner",
                "semantic_confidence": 0.88,
            },
            {
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_batch",
                "segment_range": [0, 0],
                "reason": "Keep this edge straight.",
                "confidence": 0.9,
                "requires_user_confirmation": True,
            },
        ],
        "confidence": 0.84,
        "requires_user_confirmation": True,
    }

    result = validate_command(command, document)

    assert result.tool == "propose_batch_refinement"
    assert result.target_path_id == "closed_path"
    assert result.target_segment_ids == ("closed_segment_1", "closed_segment_2", "seg_a")


def test_validate_command_rejects_batch_nested_unknown_tool() -> None:
    document = create_document(
        document_id="doc_batch_bad_tool",
        width=10.0,
        height=10.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_batch", segments=("seg_a",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_a",
            path_id="path_batch",
            type="line",
            params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
        ),
    )
    command = {
        "tool": "propose_batch_refinement",
        "summary": "batch",
        "commands": [
            {
                "tool": "propose_replace_segment_with_bezier",
                "path_id": "path_batch",
                "segment_range": [0, 0],
                "reason": "intent only",
                "confidence": 0.8,
                "requires_user_confirmation": True,
            }
        ],
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }

    with pytest.raises(CommandValidationError, match="unknown tool"):
        validate_command(command, document)


@pytest.mark.parametrize(
    ("segment_range", "message"),
    (
        ([0.5, 0.5], "segment_range indices must be integers"),
        (["0", "0"], "segment_range indices must be integers"),
        ([True, False], "segment_range indices must be integers"),
    ),
)
def test_validate_command_rejects_invalid_segment_range_types(segment_range: list[object], message: str) -> None:
    document = create_document(
        document_id="doc_bad_range",
        width=10.0,
        height=10.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_1", segments=("seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_1",
            path_id="path_1",
            type="line",
            params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
        ),
    )

    with pytest.raises(CommandValidationError, match=message):
        validate_command(
            {
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_1",
                "segment_range": segment_range,
                "reason": "invalid range type",
                "confidence": 0.8,
                "requires_user_confirmation": True,
            },
            document,
        )


def test_validate_command_rejects_boolean_confidence() -> None:
    document = create_document(
        document_id="doc_bool_confidence",
        width=10.0,
        height=10.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_1", segments=("seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_1",
            path_id="path_1",
            type="line",
            params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
        ),
    )

    with pytest.raises(CommandValidationError, match="confidence must be within \\[0, 1\\]"):
        validate_command(
            {
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_1",
                "segment_range": [0, 0],
                "reason": "bool confidence",
                "confidence": True,
                "requires_user_confirmation": True,
            },
            document,
        )


def test_validate_command_rejects_non_dict_command() -> None:
    document = create_document(
        document_id="doc_non_dict",
        width=10.0,
        height=10.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )

    with pytest.raises(CommandValidationError, match="command must be a dictionary"):
        validate_command([], document)


def test_validate_command_rejects_path_level_segment_range_and_bad_semantic_confidence() -> None:
    document = _closed_path_document_for_command_validation()

    with pytest.raises(CommandValidationError, match="path replacement does not accept segment_range"):
        validate_command(
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "closed_path",
                "segment_range": [0, 1],
                "reason": "path-level command should not carry a range",
                "confidence": 0.8,
                "requires_user_confirmation": True,
            },
            document,
        )

    with pytest.raises(CommandValidationError, match="semantic_confidence must be within \\[0, 1\\]"):
        validate_command(
            {
                "tool": "propose_replace_path_with_ellipse",
                "path_id": "closed_path",
                "reason": "path-level confidence metadata is invalid",
                "confidence": 0.8,
                "requires_user_confirmation": True,
                "semantic_confidence": 1.4,
            },
            document,
        )


def test_command_schema_service_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/command_schema.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "execute(" not in source
