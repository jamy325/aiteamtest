import ast
from pathlib import Path

import cv2
import numpy as np

from services.contour_extractor import ContourExtractor


def test_extract_binary_contours_preserves_hierarchy_and_fields() -> None:
    image = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (110, 110), 255, thickness=-1)
    cv2.rectangle(image, (40, 40), (80, 80), 0, thickness=-1)

    extractor = ContourExtractor()
    contours = extractor.extract_binary_contours(image)

    assert len(contours) == 2

    roots = [contour for contour in contours if contour.parent_contour is None]
    children = [contour for contour in contours if contour.parent_contour is not None]

    assert len(roots) == 1
    assert len(children) == 1

    root = roots[0]
    child = children[0]

    assert root.source == "binary_contour"
    assert root.contour_id.startswith("binary_contour_")
    assert root.closed is True
    assert root.area > child.area > 0.0
    assert root.depth == 0
    assert child.depth == 1
    assert child.parent_contour == root.contour_id
    assert root.children == (child.contour_id,)
    assert len(root.points) >= 4
    assert len(child.points) >= 4


def test_extract_binary_contours_handles_color_images() -> None:
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    cv2.circle(image, (30, 30), 15, (255, 255, 255), thickness=-1)

    contours = ContourExtractor().extract_binary_contours(image)

    assert len(contours) == 1
    assert contours[0].source == "binary_contour"
    assert contours[0].parent_contour is None
    assert contours[0].children == ()


def test_extract_contours_keeps_binary_and_skeleton_sets() -> None:
    image = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (110, 110), 255, thickness=-1)
    cv2.rectangle(image, (40, 40), (80, 80), 0, thickness=-1)

    extracted = ContourExtractor().extract_contours(image)

    assert len(extracted.binary_contours) == 2
    assert len(extracted.skeleton_contours) >= 1
    assert all(contour.source == "binary_contour" for contour in extracted.binary_contours)
    assert all(contour.source == "skeleton_contour" for contour in extracted.skeleton_contours)


def test_extract_skeleton_contours_from_thin_line_input() -> None:
    image = np.zeros((80, 80), dtype=np.uint8)
    cv2.line(image, (10, 40), (70, 40), 255, thickness=1)

    skeleton_contours = ContourExtractor().extract_skeleton_contours(image)

    assert len(skeleton_contours) == 1
    assert skeleton_contours[0].source == "skeleton_contour"
    assert skeleton_contours[0].closed is False
    assert len(skeleton_contours[0].points) >= 20
    assert all(point[1] == 40 for point in skeleton_contours[0].points)


def test_contour_extractor_has_no_forbidden_dependencies() -> None:
    source = Path("services/contour_extractor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"openai", "anthropic", "PyQt5", "PyQt6", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "VectorDocument" not in source
