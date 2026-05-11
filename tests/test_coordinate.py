import ast
from pathlib import Path

import pytest

from core.coordinate import CoordinateTransformer
from core.types import CoordinateSystem


def test_pixel_and_vector_round_trip_in_pixels() -> None:
    transformer = CoordinateTransformer(
        CoordinateSystem(unit="px", y_axis="down", precision=4, view_box=(0.0, 0.0, 800.0, 600.0))
    )

    point = (123.25, 456.75)
    vector = transformer.pixel_to_vector(point)
    pixel = transformer.vector_to_pixel(vector)

    assert vector == pytest.approx(point)
    assert pixel == pytest.approx(point)


def test_pixel_and_vector_round_trip_in_millimeters_with_y_flip() -> None:
    transformer = CoordinateTransformer(
        CoordinateSystem(
            unit="mm",
            y_axis="up",
            precision=4,
            view_box=(0.0, 0.0, 80.0, 60.0),
            scale={"px_to_mm": 0.1},
        )
    )

    vector = transformer.pixel_to_vector((100.0, 200.0))

    assert vector == pytest.approx((10.0, 40.0))
    assert transformer.vector_to_pixel(vector) == pytest.approx((100.0, 200.0))


def test_svg_and_dxf_conversions_follow_target_coordinate_conventions() -> None:
    transformer = CoordinateTransformer(
        CoordinateSystem(unit="px", y_axis="down", precision=3, view_box=(0.0, 0.0, 800.0, 600.0), scale={"px_to_mm": 0.2})
    )

    assert transformer.vector_to_svg((12.34567, 89.12345)) == pytest.approx((12.346, 89.123))
    assert transformer.vector_to_dxf((10.0, 20.0)) == pytest.approx((2.0, 116.0))


def test_unit_conversion_and_precision_helpers() -> None:
    transformer = CoordinateTransformer(
        CoordinateSystem(unit="mm", y_axis="down", precision=2, view_box=(0.0, 0.0, 100.0, 50.0), scale={"px_to_mm": 0.5})
    )

    assert transformer.px_to_mm(10.0) == pytest.approx(5.0)
    assert transformer.mm_to_px(5.0) == pytest.approx(10.0)
    assert transformer.y_axis_flip((3.0, 10.0)) == pytest.approx((3.0, 40.0))
    assert transformer.precision_rounding(3.14159) == pytest.approx(3.14)
    assert transformer.precision_rounding((3.14159, 2.71828), precision=3) == pytest.approx((3.142, 2.718))


def test_zero_scale_rejected_for_mm_to_px() -> None:
    transformer = CoordinateTransformer(
        CoordinateSystem(unit="mm", y_axis="down", precision=4, view_box=(0.0, 0.0, 100.0, 100.0), scale={"px_to_mm": 0.0})
    )

    with pytest.raises(ValueError):
        transformer.mm_to_px(1.0)


def test_coordinate_modules_has_no_forbidden_dependencies() -> None:
    for path in (Path("core/coordinate.py"), Path("core/precision.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(name.name.split(".")[0] for name in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

        forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui", "services"}

        assert imports.isdisjoint(forbidden_imports)
