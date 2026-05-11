import ast
from pathlib import Path

from core.document import add_anchor, add_path, add_segment, create_document
from core.types import CoordinateSystem
from services.simple_vectorizer import SimpleVectorizer


def test_simple_vectorizer_creates_closed_line_path_segments_and_anchors() -> None:
    points = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
        (0.0, 0.0),
    ]
    vectorizer = SimpleVectorizer(segment_type="line")

    result = vectorizer.vectorize_contour(points, path_id="path_line", closed=True, source="binary_contour")

    assert result.path.closed is True
    assert result.path.topology_status == "closed"
    assert result.path.source == "binary_contour"
    assert result.path.metadata["coordinate_space"] == "vector"
    assert len(result.anchors) == 4
    assert len(result.segments) == 4
    assert result.path.segments == tuple(segment.segment_id for segment in result.segments)
    assert all(anchor.in_handle is None and anchor.out_handle is None for anchor in result.anchors)
    assert all(anchor.continuity == "corner" for anchor in result.anchors)
    assert all(segment.type == "line" for segment in result.segments)
    assert result.segments[0].anchors == ("path_line_anchor_0", "path_line_anchor_1")
    assert result.segments[-1].anchors == ("path_line_anchor_3", "path_line_anchor_0")
    assert result.segments[0].params["start"] == (0.0, 0.0)
    assert result.segments[0].params["end"] == (10.0, 0.0)


def test_simple_vectorizer_creates_closed_bezier_segments_with_handles() -> None:
    points = [
        (0.0, 0.0),
        (6.0, 1.0),
        (10.0, 6.0),
        (5.0, 11.0),
        (0.0, 7.0),
        (0.0, 0.0),
    ]
    vectorizer = SimpleVectorizer(segment_type="bezier")

    result = vectorizer.vectorize_contour(points, path_id="path_bezier", closed=True, source="skeleton_contour")

    assert result.path.closed is True
    assert result.path.metadata["initial_segment_type"] == "bezier"
    assert len(result.anchors) == 5
    assert len(result.segments) == 5
    assert all(anchor.continuity == "smooth" for anchor in result.anchors)
    assert all(anchor.in_handle is not None and anchor.out_handle is not None for anchor in result.anchors)
    assert all(anchor.shared_tangent is not None for anchor in result.anchors)
    assert all(segment.type == "bezier" for segment in result.segments)
    assert all(segment.anchors[0] != segment.anchors[1] for segment in result.segments)
    assert set(result.segments[0].params) == {"start", "control1", "control2", "end"}
    assert result.segments[-1].anchors == ("path_bezier_anchor_4", "path_bezier_anchor_0")


def test_simple_vectorizer_result_is_writable_to_vector_document() -> None:
    document = create_document(
        document_id="doc_vectorizer",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(),
    )
    points = [
        (0.0, 0.0),
        (8.0, 0.0),
        (8.0, 8.0),
        (0.0, 8.0),
        (0.0, 0.0),
    ]
    result = SimpleVectorizer(segment_type="line").vectorize_contour(points, path_id="path_doc", closed=True)

    document = add_path(document, result.path)
    for anchor in result.anchors:
        document = add_anchor(document, anchor)
    for segment in result.segments:
        document = add_segment(document, segment)

    assert len(document.paths) == 1
    assert len(document.anchors) == 4
    assert len(document.segments) == 4
    assert document.paths[0].segments == tuple(segment.segment_id for segment in result.segments)
    assert document.segments[0].anchors == ("path_doc_anchor_0", "path_doc_anchor_1")


def test_simple_vectorizer_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/simple_vectorizer.py")
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
