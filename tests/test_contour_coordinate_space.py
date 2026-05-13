import pytest
import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import CoordinateSystem
from services.contour_extractor import ContourExtractor


def test_extracted_binary_and_skeleton_points_are_in_vector_space() -> None:
    image = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (30, 30), 255, thickness=-1)
    cv2.line(image, (40, 50), (70, 50), 255, thickness=1)

    transformer = CoordinateTransformer(
        CoordinateSystem(
            unit="mm",
            y_axis="up",
            precision=4,
            view_box=(0.0, 0.0, 40.0, 40.0),
            scale={"px_to_mm": 0.5},
        )
    )
    extracted = ContourExtractor(coordinate_transformer=transformer).extract_contours(image)

    assert len(extracted.binary_contours) >= 1
    assert len(extracted.skeleton_contours) >= 1
    assert all(contour.coordinate_space == "vector" for contour in extracted.binary_contours)
    assert all(contour.coordinate_space == "vector" for contour in extracted.skeleton_contours)
    assert all(isinstance(point[0], float) and isinstance(point[1], float) for contour in extracted.binary_contours for point in contour.points)
    assert all(isinstance(point[0], float) and isinstance(point[1], float) for contour in extracted.skeleton_contours for point in contour.points)

    binary_points = [point for contour in extracted.binary_contours for point in contour.points]
    xs = [point[0] for point in binary_points]
    ys = [point[1] for point in binary_points]

    assert min(xs) == pytest.approx(5.0, abs=0.5)
    assert max(xs) == pytest.approx(15.0, abs=0.5)
    assert min(ys) == pytest.approx(25.0, abs=0.5)
    assert max(ys) == pytest.approx(35.0, abs=0.5)

    expected_skeleton_point = transformer.pixel_to_vector((40.0, 50.0))
    assert any(point == pytest.approx(expected_skeleton_point) for contour in extracted.skeleton_contours for point in contour.points)
    assert all(contour.area >= 0.0 for contour in extracted.binary_contours)
    assert all(contour.area >= 0.0 for contour in extracted.skeleton_contours)


def test_binary_contour_area_is_converted_to_vector_space_mm2() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (60, 50), 255, thickness=-1)

    scale = 0.5
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
    assert contours[0].coordinate_space == "vector"
    assert contours[0].area == pytest.approx(pixel_contours[0].area * scale * scale)
