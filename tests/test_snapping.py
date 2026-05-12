import ast
from pathlib import Path

from core.document import add_anchor, add_object, add_path, create_document
from core.types import Anchor, CoordinateSystem, Object, Path as VectorPath
from services.snapping import BruteForceAnchorIndex, GlobalSnappingEngine, SnappingConfig


def _document_with_anchors() -> tuple:
    document = create_document(
        document_id="doc_snapping",
        width=200.0,
        height=100.0,
        coordinate_system=CoordinateSystem(),
    )
    object_1 = Object(object_id="object_1", type="shape")
    object_2 = Object(object_id="object_2", type="shape")
    path_a = VectorPath(path_id="path_a", object_id="object_1")
    path_b = VectorPath(path_id="path_b", object_id="object_1")
    path_c = VectorPath(path_id="path_c", object_id="object_1")
    path_d = VectorPath(path_id="path_d", object_id="object_2")

    document = add_object(document, object_1)
    document = add_object(document, object_2)
    document = add_path(document, path_a)
    document = add_path(document, path_b)
    document = add_path(document, path_c)
    document = add_path(document, path_d)

    anchors = (
        Anchor(anchor_id="a1", path_id="path_a", position=(0.0, 0.0)),
        Anchor(anchor_id="a2", path_id="path_a", position=(0.1, 0.0)),
        Anchor(anchor_id="b1", path_id="path_b", position=(1.0, 0.0)),
        Anchor(anchor_id="c1", path_id="path_c", position=(1.1, 0.0), locked=True),
        Anchor(anchor_id="d1", path_id="path_b", position=(2.0, 0.0)),
        Anchor(anchor_id="e1", path_id="path_d", position=(2.1, 0.0)),
    )
    for anchor in anchors:
        document = add_anchor(document, anchor)

    return document, anchors


def test_bruteforce_anchor_index_supports_radius_queries() -> None:
    anchors = (
        Anchor(anchor_id="a1", path_id="path_1", position=(0.0, 0.0)),
        Anchor(anchor_id="a2", path_id="path_1", position=(0.15, 0.0)),
        Anchor(anchor_id="a3", path_id="path_1", position=(5.0, 5.0)),
    )
    index = BruteForceAnchorIndex(anchors)

    matches = index.query_radius(anchors[0], 0.2)

    assert tuple(anchor.anchor_id for anchor in matches) == ("a2",)


def test_global_snapping_engine_finds_same_path_cross_path_and_cross_object_candidates() -> None:
    document, _ = _document_with_anchors()
    engine = GlobalSnappingEngine(SnappingConfig(epsilon=0.15))

    candidates = engine.find_candidates(document)
    by_pair = {tuple(sorted(candidate.anchor_ids)): candidate for candidate in candidates}

    assert by_pair[("a1", "a2")].relation == "same_path"
    assert by_pair[("b1", "c1")].relation == "cross_path_same_object"
    assert by_pair[("d1", "e1")].relation == "cross_object"


def test_global_snapping_engine_marks_locked_candidates_without_moving_locked_anchor() -> None:
    document, _ = _document_with_anchors()
    engine = GlobalSnappingEngine(SnappingConfig(epsilon=0.15))

    candidates = engine.find_candidates(document)
    by_pair = {tuple(sorted(candidate.anchor_ids)): candidate for candidate in candidates}
    candidate = by_pair[("b1", "c1")]

    assert candidate.locked_involved is True
    assert candidate.movable_anchor_ids == ("b1",)


def test_global_snapping_engine_uses_spatial_index_interface() -> None:
    document, anchors = _document_with_anchors()

    class RecordingIndex:
        def __init__(self, indexed_anchors: tuple[Anchor, ...]) -> None:
            self.indexed_anchor_ids = tuple(anchor.anchor_id for anchor in indexed_anchors)
            self.query_count = 0

        def query_radius(self, anchor: Anchor, radius: float) -> tuple[Anchor, ...]:
            self.query_count += 1
            return tuple(
                candidate
                for candidate in anchors
                if candidate.anchor_id != anchor.anchor_id and abs(candidate.position[0] - anchor.position[0]) <= radius
            )

    created_indexes: list[RecordingIndex] = []

    def build_index(indexed_anchors: tuple[Anchor, ...]) -> RecordingIndex:
        index = RecordingIndex(indexed_anchors)
        created_indexes.append(index)
        return index

    engine = GlobalSnappingEngine(SnappingConfig(epsilon=0.15), index_builder=build_index)
    candidates = engine.find_candidates(document)

    assert candidates
    assert len(created_indexes) == 1
    assert created_indexes[0].indexed_anchor_ids == tuple(anchor.anchor_id for anchor in document.anchors)
    assert created_indexes[0].query_count == len(document.anchors)


def test_global_snapping_engine_does_not_mutate_document_or_anchors() -> None:
    document, _ = _document_with_anchors()
    original_document = document
    original_anchors = document.anchors
    engine = GlobalSnappingEngine(SnappingConfig(epsilon=0.15))

    _ = engine.find_candidates(document)

    assert document == original_document
    assert document.anchors == original_anchors


def test_snapping_service_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/snapping.py")
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
    assert "open(" not in source
