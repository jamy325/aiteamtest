import ast
from pathlib import Path as FilePath

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path, Segment
from services.lock_manager import LockManager, lock_segment


def _document_with_segments() -> tuple:
    document = create_document(
        document_id="doc_lock",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(),
    )
    document = add_path(document, Path(path_id="path_1"))
    document = add_segment(
        document,
        Segment(
            segment_id="segment_1",
            path_id="path_1",
            type="line",
            params={"start": (0.0, 0.0), "end": (10.0, 0.0)},
        ),
    )
    document = add_segment(
        document,
        Segment(
            segment_id="segment_2",
            path_id="path_1",
            type="line",
            params={"start": (10.0, 0.0), "end": (20.0, 0.0)},
        ),
    )
    return document, document.segments[0], document.segments[1]


def test_lock_segment_sets_segment_locked_without_mutating_original_document() -> None:
    document, first_segment, second_segment = _document_with_segments()

    locked_document = lock_segment(document, "segment_1")

    assert document.segments[0].locked is False
    assert locked_document.segments[0].locked is True
    assert locked_document.segments[0].segment_id == first_segment.segment_id
    assert locked_document.segments[0].path_id == first_segment.path_id
    assert locked_document.segments[0].params == first_segment.params
    assert locked_document.segments[1] == second_segment


def test_lock_manager_repeated_lock_is_idempotent() -> None:
    document, _, _ = _document_with_segments()
    manager = LockManager()

    locked_once = manager.lock_segment(document, "segment_1")
    locked_twice = manager.lock_segment(locked_once, "segment_1")

    assert locked_once.segments[0].locked is True
    assert locked_twice is locked_once


def test_lock_segment_rejects_unknown_segment_id() -> None:
    document, _, _ = _document_with_segments()

    with pytest.raises(ValueError, match="unknown segment_id: missing_segment"):
        lock_segment(document, "missing_segment")


def test_lock_manager_has_no_forbidden_dependencies() -> None:
    source_path = FilePath("services/lock_manager.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"openai", "anthropic", "ui", "cv2", "matplotlib", "PyQt5", "PyQt6"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
