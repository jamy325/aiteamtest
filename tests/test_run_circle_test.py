from __future__ import annotations

import json
from pathlib import Path

from scripts.run_circle_test import create_synthetic_circle_images, run_circle_tests


def test_create_synthetic_circle_images_writes_defaults(tmp_path: Path) -> None:
    image_paths = create_synthetic_circle_images(tmp_path)

    names = {path.name for path in image_paths}
    assert names == {"black_circle_on_white.png", "blue_circle_on_white.png"}
    assert all(path.exists() for path in image_paths)


def test_run_circle_test_generates_summary_and_exports(tmp_path: Path) -> None:
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
    result = results[0]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.summary_path.exists()
    assert (result.output_dir / "vector_document_before.json").exists()
    assert (result.output_dir / "vector_document_after.json").exists()
    assert (result.output_dir / "vector_result.svg").exists()
    assert (result.output_dir / "vector_result.dxf").exists()
    assert (result.output_dir / "overlay_before.png").exists()
    assert (result.output_dir / "overlay_after.png").exists()
    assert (result.output_dir / "vision_manifest.json").exists()

    assert summary["selected_path_id"] is not None
    assert summary["path_candidates"]
    assert "has_circle_segment" in summary
    assert "dxf_entity_counts" in summary
    assert "execute_reason" in summary
    assert set(summary["dxf_entity_counts"]).issuperset({"CIRCLE", "ARC", "LINE", "LWPOLYLINE", "POLYLINE"})

    if summary["execute_success"]:
        assert summary["has_circle_segment"] is True
        assert summary["circle_segments"]
        assert any(segment["params"].get("r", 0.0) for segment in summary["circle_segments"])
    else:
        assert summary["execute_reason"]

