from __future__ import annotations

from pathlib import Path

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment, updated
from services.history_manager import HistoryManager


def _build_document(
    *,
    document_id: str = "doc_history",
    path_id: str = "path_1",
    segment_id: str = "path_1_seg_1",
    segment_type: str = "polyline",
    points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0)),
) -> object:
    document = create_document(
        document_id=document_id,
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id=path_id, segments=(segment_id,)))
    document = add_segment(
        document,
        Segment(
            segment_id=segment_id,
            path_id=path_id,
            type=segment_type,
            params={"points": [[float(x), float(y)] for x, y in points]},
        ),
    )
    return document


def _replace_first_segment(document: object, *, segment_type: str, params: dict[str, object] | None = None) -> object:
    replacement = updated(
        document.segments[0],
        type=segment_type,
        params=params or document.segments[0].params,
    )
    segments = (replacement,) + document.segments[1:]
    return updated(document, segments=segments)


def test_history_manager_supports_undo_and_redo() -> None:
    before = _build_document()
    after = _replace_first_segment(
        before,
        segment_type="line",
        params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
    )
    manager = HistoryManager()

    item = manager.record(
        command={"command_id": "cmd_1", "tool": "propose_replace_segment_with_line"},
        before_document=before,
        after_document=after,
        old_score=10.0,
        new_score=5.0,
    )

    assert item.version == 1
    assert item.old_score == 10.0
    assert item.new_score == 5.0
    assert manager.undo() == before
    assert manager.redo() == after


def test_history_manager_supports_consecutive_undo_and_redo() -> None:
    document_a = _build_document()
    document_b = _replace_first_segment(
        document_a,
        segment_type="line",
        params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
    )
    document_c = _replace_first_segment(
        document_b,
        segment_type="arc",
        params={
            "cx": 0.5,
            "cy": 0.5,
            "r": 1.0,
            "start_angle": 0.0,
            "end_angle": 1.0,
            "direction": "ccw",
        },
    )
    manager = HistoryManager()
    manager.record(command={"command_id": "cmd_1"}, before_document=document_a, after_document=document_b)
    manager.record(command={"command_id": "cmd_2"}, before_document=document_b, after_document=document_c)

    assert manager.undo() == document_b
    assert manager.undo() == document_a
    with pytest.raises(ValueError, match="undo"):
        manager.undo()
    assert manager.redo() == document_b
    assert manager.redo() == document_c
    with pytest.raises(ValueError, match="redo"):
        manager.redo()


def test_history_manager_truncates_redo_stack_after_new_record() -> None:
    document_a = _build_document()
    document_b = _replace_first_segment(
        document_a,
        segment_type="line",
        params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
    )
    document_c = _replace_first_segment(
        document_b,
        segment_type="arc",
        params={
            "cx": 0.5,
            "cy": 0.5,
            "r": 1.0,
            "start_angle": 0.0,
            "end_angle": 1.0,
            "direction": "ccw",
        },
    )
    document_d = _replace_first_segment(
        document_b,
        segment_type="circle",
        params={"cx": 0.5, "cy": 0.0, "r": 0.5},
    )
    manager = HistoryManager()
    manager.record(command={"command_id": "cmd_1"}, before_document=document_a, after_document=document_b)
    manager.record(command={"command_id": "cmd_2"}, before_document=document_b, after_document=document_c)

    assert manager.undo() == document_b
    manager.record(command={"command_id": "cmd_3"}, before_document=document_b, after_document=document_d)

    assert tuple(item.version for item in manager.items) == (1, 2)
    assert tuple(item.command["command_id"] for item in manager.items) == ("cmd_1", "cmd_3")
    with pytest.raises(ValueError, match="redo"):
        manager.redo()


def test_history_manager_supports_batch_history_and_query_by_command_id() -> None:
    before = _build_document()
    after = _replace_first_segment(
        before,
        segment_type="line",
        params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
    )
    manager = HistoryManager()
    batch_command = {
        "command_id": "batch_001",
        "tool": "propose_batch_refinement",
        "commands": [
            {"command_id": "cmd_1", "tool": "propose_replace_segment_with_line"},
        ],
    }

    manager.record(command=batch_command, before_document=before, after_document=after)
    batch_items = manager.get_by_command_id("batch_001")

    assert len(batch_items) == 1
    assert batch_items[0].command == batch_command
    assert manager.undo() == before


def test_history_manager_snapshots_are_isolated_from_later_mutation() -> None:
    before = _build_document(points=((0.0, 0.0), (1.0, 0.0)))
    after = _replace_first_segment(
        before,
        segment_type="line",
        params={"start": [0.0, 0.0], "end": [1.0, 0.0]},
    )
    manager = HistoryManager()
    manager.record(command={"command_id": "cmd_1"}, before_document=before, after_document=after)

    before.segments[0].params["points"][0][0] = 999.0
    after.segments[0].params["end"][0] = 999.0

    restored_before = manager.undo()
    restored_after = manager.redo()

    assert restored_before.segments[0].params["points"] == [[0.0, 0.0], [1.0, 0.0]]
    assert restored_after.segments[0].params["end"] == [1.0, 0.0]


def test_history_manager_has_no_forbidden_dependencies() -> None:
    source = Path("services/history_manager.py").read_text(encoding="utf-8")
    assert "cv2" not in source
    assert "PyQt" not in source
    assert "openai" not in source
