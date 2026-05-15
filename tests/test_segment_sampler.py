import ast
import math
from pathlib import Path

import pytest

from core.types import Segment
from services.segment_sampler import SegmentSampler, SegmentSamplerConfig


def test_segment_sampler_samples_line_with_optional_density() -> None:
    sparse_sampler = SegmentSampler()
    dense_sampler = SegmentSampler(SegmentSamplerConfig(line_sample_step=1.0))
    segment = Segment(
        segment_id="line_1",
        path_id="path_1",
        type="line",
        params={"start": [0.0, 0.0], "end": [4.0, 0.0]},
    )

    sparse = sparse_sampler.sample_segment(segment)
    dense = dense_sampler.sample_segment(segment)

    assert sparse == ((0.0, 0.0), (4.0, 0.0))
    assert len(dense) == 5
    assert dense[2] == pytest.approx((2.0, 0.0))


def test_segment_sampler_samples_bezier_arc_circle_and_ellipse_in_vector_space() -> None:
    sampler = SegmentSampler(
        SegmentSamplerConfig(
            max_chord_error=0.1,
            min_segments_per_arc=8,
            max_segments_per_arc=128,
            circle_segments=24,
            ellipse_segments=24,
            bezier_segments=12,
        )
    )
    bezier = Segment(
        segment_id="bezier_1",
        path_id="path_1",
        type="bezier",
        params={
            "start": [0.0, 0.0],
            "control1": [1.0, 2.0],
            "control2": [3.0, 2.0],
            "end": [4.0, 0.0],
        },
    )
    arc = Segment(
        segment_id="arc_1",
        path_id="path_1",
        type="arc",
        params={
            "cx": 10.0,
            "cy": 10.0,
            "r": 5.0,
            "start_angle": 0.0,
            "end_angle": math.pi / 2.0,
            "direction": "ccw",
        },
    )
    circle = Segment(
        segment_id="circle_1",
        path_id="path_1",
        type="circle",
        params={"cx": 5.0, "cy": -2.0, "r": 3.0},
    )
    ellipse = Segment(
        segment_id="ellipse_1",
        path_id="path_1",
        type="ellipse",
        params={"cx": 4.0, "cy": 6.0, "rx": 5.0, "ry": 2.0, "rotation": math.pi / 6.0},
    )

    bezier_points = sampler.sample_segment(bezier)
    arc_points = sampler.sample_segment(arc)
    circle_points = sampler.sample_segment(circle)
    ellipse_points = sampler.sample_segment(ellipse)

    assert len(bezier_points) == 13
    assert bezier_points[0] == pytest.approx((0.0, 0.0))
    assert bezier_points[-1] == pytest.approx((4.0, 0.0))

    assert len(arc_points) >= 9
    assert arc_points[0] == pytest.approx((15.0, 10.0))
    assert arc_points[-1] == pytest.approx((10.0, 15.0))

    assert len(circle_points) == 25
    assert sampler.is_closed(circle) is True
    assert circle_points[0] == pytest.approx(circle_points[-1])

    assert len(ellipse_points) == 25
    assert sampler.is_closed(ellipse) is True
    assert ellipse_points[0] == pytest.approx(ellipse_points[-1])
    assert max(point[1] for point in ellipse_points) > min(point[1] for point in ellipse_points)


def test_segment_sampler_samples_half_circle_when_arc_angles_are_radians() -> None:
    sampler = SegmentSampler(SegmentSamplerConfig(min_segments_per_arc=8, max_segments_per_arc=64))
    segment = Segment(
        segment_id="arc_half_circle",
        path_id="path_1",
        type="arc",
        params={
            "cx": 5.0,
            "cy": 5.0,
            "r": 3.0,
            "start_angle": 0.0,
            "end_angle": math.pi,
            "direction": "ccw",
        },
    )

    points = sampler.sample_segment(segment)

    assert points[0] == pytest.approx((8.0, 5.0))
    assert points[-1] == pytest.approx((2.0, 5.0), abs=1e-6)


def test_segment_sampler_does_not_implicitly_treat_180_as_degrees() -> None:
    sampler = SegmentSampler(SegmentSamplerConfig(min_segments_per_arc=8, max_segments_per_arc=64))
    segment = Segment(
        segment_id="arc_180_raw",
        path_id="path_1",
        type="arc",
        params={
            "cx": 5.0,
            "cy": 5.0,
            "r": 3.0,
            "start_angle": 0.0,
            "end_angle": 180.0,
            "direction": "ccw",
        },
    )

    points = sampler.sample_segment(segment)

    assert points[-1] != pytest.approx((2.0, 5.0), abs=1e-3)


def test_segment_sampler_supports_explicit_degree_angle_unit_for_import_adapters() -> None:
    sampler = SegmentSampler(SegmentSamplerConfig(min_segments_per_arc=8, max_segments_per_arc=64))
    segment = Segment(
        segment_id="arc_180_degree",
        path_id="path_1",
        type="arc",
        params={
            "cx": 5.0,
            "cy": 5.0,
            "r": 3.0,
            "start_angle": 0.0,
            "end_angle": 180.0,
            "direction": "ccw",
            "angle_unit": "degree",
        },
    )

    points = sampler.sample_segment(segment)

    assert points[-1] == pytest.approx((2.0, 5.0), abs=1e-6)


def test_segment_sampler_rejects_unknown_segment_type() -> None:
    sampler = SegmentSampler()
    segment = Segment(
        segment_id="polyline_1",
        path_id="path_1",
        type="polyline",
        params={"points": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]},
    )

    assert sampler.sample_segment(segment) == ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0))

    segment = segment.__class__(
        segment_id="bad_1",
        path_id="path_1",
        type="line",
        params={"foo": [0.0, 0.0]},
    )
    with pytest.raises(KeyError):
        sampler.sample_segment(segment)


def test_segment_sampler_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/segment_sampler.py")
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
