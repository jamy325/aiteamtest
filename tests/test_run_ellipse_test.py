from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_ellipse_test import create_synthetic_ellipse_images, run_ellipse_tests


def test_create_synthetic_ellipse_images_writes_defaults(tmp_path: Path) -> None:
    image_paths = create_synthetic_ellipse_images(tmp_path)

    assert {path.name for path in image_paths} == {"black_ellipse_on_white.png", "blue_ellipse_on_white.png"}
    assert all(path.exists() for path in image_paths)


def test_run_ellipse_test_generates_summary_and_records_fitting_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "ellipse_inputs"
    output_dir = tmp_path / "ellipse_outputs"
    create_synthetic_ellipse_images(input_dir)

    results = run_ellipse_tests(
        input_dir=input_dir,
        output_dir=output_dir,
        image="black_ellipse_on_white.png",
        debug=True,
        export_mode="centerline",
        min_path_area=128.0,
    )

    assert len(results) == 1
    payload = json.loads(Path(results[0]["output_files"]["summary"]).read_text(encoding="utf-8"))

    assert payload["selected_path_id"] is not None
    assert payload["path_candidates"]
    assert "has_ellipse_segment" in payload
    assert "dxf_entity_counts" in payload
    assert "fitting_source" in payload
    assert payload["integrity_success"] is True
    assert set(payload["dxf_entity_counts"]).issuperset({"CIRCLE", "ARC", "LINE", "LWPOLYLINE", "POLYLINE"})

    if payload["execute_success"]:
        assert payload["has_ellipse_segment"] is True
        assert payload["fitting_source"] == "raw_contour_points"
        assert payload["ellipse_segments"]
    else:
        pytest.fail(f"synthetic ellipse replacement did not succeed: {payload['execute_reason']}")


@pytest.mark.skipif(
    not Path("test_images/ellipse/test_input_ellipse.png").exists(),
    reason="manual ellipse fixture is not present in the repository checkout",
)
def test_run_ellipse_test_handles_manual_ellipse_fixture_when_available(tmp_path: Path) -> None:
    output_dir = tmp_path / "manual_ellipse_outputs"

    results = run_ellipse_tests(
        input_dir=Path("test_images/ellipse"),
        output_dir=output_dir,
        image="test_input_ellipse.png",
        debug=True,
        export_mode="centerline",
        min_path_area=128.0,
    )

    assert len(results) == 1
    payload = json.loads(Path(results[0]["output_files"]["summary"]).read_text(encoding="utf-8"))
    assert payload["selected_path_id"] is not None
    assert payload["path_candidates"]
    if payload["execute_success"]:
        assert payload["has_ellipse_segment"] is True
        assert payload["fitting_source"] in {"raw_contour_points", "segment_samples_fallback"}
        assert payload["integrity_success"] is True
    else:
        pytest.fail(f"manual ellipse replacement did not succeed: {payload['execute_reason']}")
