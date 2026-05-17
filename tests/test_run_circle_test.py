from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_circle_test import create_synthetic_circle_images, run_circle_tests


def test_create_synthetic_circle_images_writes_defaults(tmp_path: Path) -> None:
    image_paths = create_synthetic_circle_images(tmp_path)

    assert {path.name for path in image_paths} == {"black_circle_on_white.png", "blue_circle_on_white.png"}
    assert all(path.exists() for path in image_paths)


def test_run_circle_test_generates_summary_and_records_fitting_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "circle_inputs"
    output_dir = tmp_path / "circle_outputs"
    create_synthetic_circle_images(input_dir)

    results = run_circle_tests(
        input_dir=input_dir,
        output_dir=output_dir,
        image="black_circle_on_white.png",
        debug=True,
        export_mode="centerline",
        min_path_area=128.0,
    )

    assert len(results) == 1
    summary = results[0]

    summary_path = Path(summary["output_files"]["summary"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["selected_path_id"] is not None
    assert payload["path_candidates"]
    assert "has_circle_segment" in payload
    assert "dxf_entity_counts" in payload
    assert "fitting_source" in payload
    assert set(payload["dxf_entity_counts"]).issuperset({"CIRCLE", "ARC", "LINE", "LWPOLYLINE", "POLYLINE"})

    if payload["execute_success"]:
        assert payload["has_circle_segment"] is True
        assert payload["fitting_source"] == "raw_contour_points"
        assert payload["circle_segments"]
        svg_payload = Path(payload["output_files"]["svg"]).read_text(encoding="utf-8")
        assert "<circle" in svg_payload or "<path" in svg_payload
        assert 'stroke="none"' not in svg_payload or 'fill="none" stroke="none"' not in svg_payload
    else:
        assert payload["execute_reason"]


@pytest.mark.skipif(
    not Path("test_images/circle/test_input_circle.png").exists(),
    reason="manual circle fixture is not present in the repository checkout",
)
def test_run_circle_test_handles_manual_circle_fixture_when_available(tmp_path: Path) -> None:
    output_dir = tmp_path / "manual_circle_outputs"

    results = run_circle_tests(
        input_dir=Path("test_images/circle"),
        output_dir=output_dir,
        image="test_input_circle.png",
        debug=True,
        export_mode="centerline",
        min_path_area=128.0,
    )

    assert len(results) == 1
    payload = json.loads(Path(results[0]["output_files"]["summary"]).read_text(encoding="utf-8"))
    assert payload["selected_path_id"] is not None
    assert payload["path_candidates"]
    assert payload["fitting_source"] == "raw_contour_points"
    if payload["execute_success"]:
        assert payload["has_circle_segment"] is True
        assert payload["dxf_entity_counts"]["CIRCLE"] >= 1
    else:
        pytest.fail(f"manual circle replacement did not succeed: {payload['execute_reason']}")
