import ast
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.coordinate import CoordinateTransformer
from core.types import CoordinateSystem
from services.contour_extractor import ContourExtractor


FIXTURE_ROOT = Path("tests/fixtures/debug_artifacts")


def _chebyshev_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def _assert_continuous_points(points: tuple[tuple[float, float], ...], *, closed: bool) -> None:
    assert points
    for left, right in zip(points, points[1:]):
        assert _chebyshev_distance(left, right) <= 1.0
    if closed and len(points) > 1:
        assert _chebyshev_distance(points[-1], points[0]) <= 1.0


def _contour_touches_page_border(points: tuple[tuple[float, float], ...], image_shape: tuple[int, int]) -> bool:
    height, width = image_shape
    if not points:
        return False
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs) <= 0.0 or min(ys) <= 0.0 or max(xs) >= float(width - 1) or max(ys) >= float(height - 1)


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


def test_extract_binary_contours_area_matches_pixel_area_in_px_unit() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    top_left = (15, 20)
    bottom_right = (70, 65)
    cv2.rectangle(image, top_left, bottom_right, 255, thickness=-1)

    extractor = ContourExtractor()
    contours = extractor.extract_binary_contours(image)

    assert len(contours) == 1
    contour = contours[0]
    processed_mask = extractor._preprocess_binary_mask(image)
    processed_contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    expected_area = float(cv2.contourArea(processed_contours[0]))
    assert contour.coordinate_space == "vector"
    assert contour.area == pytest.approx(expected_area)


def test_extract_binary_contours_area_scales_with_non_unit_px_to_mm_factor() -> None:
    image = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(image, (60, 60), 18, 255, thickness=-1)

    scale = 0.2
    pixel_extractor = ContourExtractor()
    extractor = ContourExtractor(
        coordinate_transformer=CoordinateTransformer(
            CoordinateSystem(
                unit="mm",
                scale={"px_to_mm": scale},
            )
        )
    )

    pixel_contours = pixel_extractor.extract_binary_contours(image)
    contours = extractor.extract_binary_contours(image)

    assert len(pixel_contours) == 1
    assert len(contours) == 1
    assert contours[0].area == pytest.approx(pixel_contours[0].area * scale * scale)


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
    assert all(contour.coordinate_space == "vector" for contour in extracted.binary_contours)
    assert all(contour.coordinate_space == "vector" for contour in extracted.skeleton_contours)


def test_extract_skeleton_contours_from_thin_line_input() -> None:
    image = np.zeros((80, 80), dtype=np.uint8)
    cv2.line(image, (10, 40), (70, 40), 255, thickness=1)

    skeleton_contours = ContourExtractor().extract_skeleton_contours(image)

    assert len(skeleton_contours) == 1
    assert skeleton_contours[0].source == "skeleton_contour"
    assert skeleton_contours[0].closed is False
    assert len(skeleton_contours[0].points) >= 20
    assert skeleton_contours[0].area == float(len(skeleton_contours[0].points))
    _assert_continuous_points(skeleton_contours[0].points, closed=False)
    assert skeleton_contours[0].points[0] == (10.0, 40.0)
    assert skeleton_contours[0].points[-1] == (70.0, 40.0)


def test_extract_skeleton_contours_marks_closed_loops() -> None:
    image = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(image, (15, 15), (60, 55), 255, thickness=1)

    skeleton_contours = ContourExtractor().extract_skeleton_contours(image)

    assert len(skeleton_contours) == 1
    assert skeleton_contours[0].source == "skeleton_contour"
    assert skeleton_contours[0].closed is True
    assert len(skeleton_contours[0].points) >= 20
    _assert_continuous_points(skeleton_contours[0].points, closed=True)


def test_extract_skeleton_contours_from_thick_open_stroke_keeps_one_ordered_path() -> None:
    image = np.zeros((120, 120), dtype=np.uint8)
    start = (10, 60)
    end = (110, 60)
    cv2.line(image, start, end, 255, thickness=3)

    skeleton_contours = ContourExtractor().extract_skeleton_contours(image)

    assert len(skeleton_contours) == 1
    contour = skeleton_contours[0]
    assert contour.closed is False
    _assert_continuous_points(contour.points, closed=False)
    assert _chebyshev_distance(contour.points[0], (float(start[0]), float(start[1]))) <= 2.0
    assert _chebyshev_distance(contour.points[-1], (float(end[0]), float(end[1]))) <= 2.0


def test_extract_skeleton_contours_from_thick_circle_stroke_has_major_closed_loop() -> None:
    image = np.zeros((140, 140), dtype=np.uint8)
    center = (70, 70)
    radius = 40
    cv2.circle(image, center, radius, 255, thickness=3)

    skeleton_contours = ContourExtractor().extract_skeleton_contours(image)

    assert skeleton_contours
    major = max(skeleton_contours, key=lambda contour: len(contour.points))
    assert major.closed is True
    _assert_continuous_points(major.points, closed=True)
    assert len(major.points) == max(len(contour.points) for contour in skeleton_contours)
    assert all(len(contour.points) >= 2 for contour in skeleton_contours)


def test_extract_skeleton_contours_from_thick_rectangle_stroke_has_major_closed_loop() -> None:
    image = np.zeros((140, 140), dtype=np.uint8)
    top_left = (20, 25)
    bottom_right = (115, 105)
    cv2.rectangle(image, top_left, bottom_right, 255, thickness=3)

    skeleton_contours = ContourExtractor().extract_skeleton_contours(image)

    assert skeleton_contours
    major = max(skeleton_contours, key=lambda contour: len(contour.points))
    assert major.closed is True
    _assert_continuous_points(major.points, closed=True)
    assert len(major.points) == max(len(contour.points) for contour in skeleton_contours)
    assert all(len(contour.points) >= 2 for contour in skeleton_contours)


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


def test_auto_foreground_polarity_rejects_page_border_for_black_square_fixture() -> None:
    image = cv2.imread(str(FIXTURE_ROOT / "black_square_on_white.png"), cv2.IMREAD_COLOR)
    extracted, debug = ContourExtractor().extract_contours_with_debug(image)

    assert debug.foreground_mode == "dark_on_light"
    assert debug.threshold_polarity == "dark_on_light"
    assert "border_foreground_ratio" in debug.foreground_reason
    assert all(not (item.get("touches_border") and item.get("bbox_coverage", 0.0) >= 0.98 and not item.get("filtered")) for item in debug.binary_contours_hierarchy)
    assert all(not _contour_touches_page_border(contour.points, image.shape[:2]) for contour in extracted.binary_contours)
    assert all(not _contour_touches_page_border(contour.points, image.shape[:2]) for contour in extracted.skeleton_contours)


def test_auto_foreground_polarity_rejects_page_border_for_blue_circle_fixture() -> None:
    image = cv2.imread(str(FIXTURE_ROOT / "blue_circle_on_white.png"), cv2.IMREAD_COLOR)
    extracted, debug = ContourExtractor().extract_contours_with_debug(image)

    assert debug.foreground_mode == "dark_on_light"
    assert all(not (item.get("touches_border") and item.get("bbox_coverage", 0.0) >= 0.98 and not item.get("filtered")) for item in debug.binary_contours_hierarchy)
    assert all(not _contour_touches_page_border(contour.points, image.shape[:2]) for contour in extracted.binary_contours)
    assert all(not _contour_touches_page_border(contour.points, image.shape[:2]) for contour in extracted.skeleton_contours)
