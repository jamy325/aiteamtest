import ast
from pathlib import Path

import cv2
import numpy as np

from core.document import from_json
from core.types import CoordinateSystem
from services.minimal_pipeline import MinimalPipeline


def _test_image() -> np.ndarray:
    image = np.zeros((120, 140), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (70, 90), 255, thickness=-1)
    cv2.line(image, (85, 20), (125, 60), 255, thickness=1)
    return image


def test_minimal_pipeline_runs_end_to_end_from_image_file(tmp_path: Path) -> None:
    image_path = tmp_path / "pipeline_input.png"
    cv2.imwrite(str(image_path), _test_image())

    pipeline = MinimalPipeline(
        coordinate_system=CoordinateSystem(
            view_box=(0.0, 0.0, 140.0, 120.0),
            precision=4,
        )
    )
    result = pipeline.run_from_file(image_path, document_id="doc_pipeline")
    restored = from_json(result.json_payload)

    assert result.document.document_id == "doc_pipeline"
    assert restored == result.document
    assert result.extracted_contours.binary_contours
    assert result.extracted_contours.skeleton_contours
    assert len(result.document.paths) >= 2
    assert len(result.document.segments) > 0
    assert len(result.document.anchors) > 0
    assert {path.source for path in result.document.paths}.issubset({"binary_contour", "skeleton_contour"})
    assert all(contour.coordinate_space == "vector" for contour in result.extracted_contours.binary_contours)
    assert all(contour.coordinate_space == "vector" for contour in result.extracted_contours.skeleton_contours)
    assert all(isinstance(anchor.position[0], float) and isinstance(anchor.position[1], float) for anchor in result.document.anchors)

    pipeline_metadata = restored.metadata["pipeline"]
    assert pipeline_metadata["segment_type"] == "line"
    assert pipeline_metadata["source_contours"]["binary_contours"]
    assert pipeline_metadata["source_contours"]["skeleton_contours"]
    assert all(item["coordinate_space"] == "vector" for item in pipeline_metadata["source_contours"]["binary_contours"])
    assert all(item["coordinate_space"] == "vector" for item in pipeline_metadata["source_contours"]["skeleton_contours"])
    assert all(item["coordinate_space"] == "vector" for item in pipeline_metadata["resampled_contours"]["binary_contours"])
    assert all(item["coordinate_space"] == "vector" for item in pipeline_metadata["resampled_contours"]["skeleton_contours"])


def test_minimal_pipeline_supports_bezier_segment_generation() -> None:
    pipeline = MinimalPipeline(
        coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 140.0, 120.0)),
        segment_type="bezier",
    )

    result = pipeline.run(_test_image(), document_id="doc_bezier")

    assert result.document.document_id == "doc_bezier"
    assert result.document.paths
    assert result.document.segments
    assert all(segment.type == "bezier" for segment in result.document.segments)
    assert result.document.metadata["pipeline"]["segment_type"] == "bezier"


def test_minimal_pipeline_has_no_ui_or_ai_dependencies() -> None:
    source_path = Path("services/minimal_pipeline.py")
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
