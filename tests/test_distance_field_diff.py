import ast
from pathlib import Path

import numpy as np

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.distance_field_diff import DistanceFieldDiffOptions, DistanceFieldDiffRenderer


def _diff_document() -> object:
    document = create_document(
        document_id="doc_diff",
        width=64.0,
        height=64.0,
        coordinate_system=CoordinateSystem(
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, 64.0, 64.0),
        ),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": "binary_1",
                            "points": [[5.0, 10.0], [25.0, 10.0]],
                            "coordinate_space": "vector",
                            "closed": False,
                        }
                    ],
                    "skeleton_contours": [],
                }
            }
        },
    )
    path = VectorPath(path_id="path_1")
    segment = Segment(
        segment_id="segment_1",
        path_id="path_1",
        type="line",
        params={"start": [5.0, 12.0], "end": [25.0, 12.0]},
    )
    document = add_path(document, path)
    document = add_segment(document, segment)
    return document


def _scaled_diff_document() -> object:
    document = create_document(
        document_id="doc_diff_scaled",
        width=64.0,
        height=64.0,
        coordinate_system=CoordinateSystem(
            unit="mm",
            precision=4,
            view_box=(0.0, 0.0, 32.0, 32.0),
            scale={"px_to_mm": 0.5},
        ),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": "binary_scaled",
                            "points": [[2.5, 5.0], [12.5, 5.0]],
                            "coordinate_space": "vector",
                            "closed": False,
                        }
                    ],
                    "skeleton_contours": [],
                }
            }
        },
    )
    path = VectorPath(path_id="path_scaled")
    segment = Segment(
        segment_id="segment_scaled",
        path_id="path_scaled",
        type="line",
        params={"start": [2.5, 6.0], "end": [12.5, 6.0]},
    )
    document = add_path(document, path)
    document = add_segment(document, segment)
    return document


def _closed_source_rectangle_document() -> object:
    return create_document(
        document_id="doc_closed_source",
        width=64.0,
        height=64.0,
        coordinate_system=CoordinateSystem(
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, 64.0, 64.0),
        ),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": "binary_closed",
                            "points": [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
                            "coordinate_space": "vector",
                            "closed": True,
                        }
                    ],
                    "skeleton_contours": [],
                }
            }
        },
    )


def test_distance_field_diff_renderer_generates_missing_and_overdraw_image() -> None:
    document = _diff_document()
    renderer = DistanceFieldDiffRenderer()

    result = renderer.render_diff(document)
    encoded = renderer.export_diff_png(document)

    assert result.image.shape == (64, 64, 3)
    assert result.missing_edge_error > 0.0
    assert result.overdraw_error > 0.0
    assert result.chamfer_error == result.missing_edge_error + result.overdraw_error
    assert result.source_point_count == 2
    assert result.vector_point_count >= 2
    assert result.image[10, 5, 2] > 0
    assert result.image[12, 5, 0] > 0
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_distance_field_diff_renderer_sampling_density_affects_vector_sample_count() -> None:
    document = _diff_document()
    sparse_renderer = DistanceFieldDiffRenderer(DistanceFieldDiffOptions(sample_step=5.0))
    dense_renderer = DistanceFieldDiffRenderer(DistanceFieldDiffOptions(sample_step=1.0))

    sparse_result = sparse_renderer.render_diff(document)
    dense_result = dense_renderer.render_diff(document)

    assert sparse_result.vector_point_count < dense_result.vector_point_count
    assert sparse_result.source_point_count == dense_result.source_point_count


def test_distance_field_diff_renderer_uses_document_pixel_dimensions_for_scaled_coordinate_system() -> None:
    document = _scaled_diff_document()
    renderer = DistanceFieldDiffRenderer()

    result = renderer.render_diff(document)

    assert result.image.shape == (64, 64, 3)
    assert result.image[10, 5, 2] > 0
    assert result.image[12, 5, 0] > 0


def test_distance_field_diff_renderer_preserves_closed_source_contour_edges() -> None:
    document = _closed_source_rectangle_document()
    renderer = DistanceFieldDiffRenderer()

    result = renderer.render_diff(document)

    assert result.image[20, 10, 2] > 0


def test_distance_field_diff_renderer_does_not_mutate_document() -> None:
    document = _diff_document()
    original_document = document
    renderer = DistanceFieldDiffRenderer()

    _ = renderer.render_diff(document)

    assert document == original_document


def test_distance_field_diff_renderer_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/distance_field_diff.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
