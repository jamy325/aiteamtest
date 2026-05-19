from __future__ import annotations

import json
from pathlib import Path

from scripts.run_bezier_test import build_arg_parser, create_synthetic_bezier_images, main, run_bezier_tests


def test_create_synthetic_bezier_images_writes_defaults(tmp_path: Path) -> None:
    image_paths = create_synthetic_bezier_images(tmp_path)

    assert {path.name for path in image_paths} == {"black_heart_on_white.png", "blue_blob_on_white.png"}
    assert all(path.exists() for path in image_paths)


def test_run_bezier_test_generates_summary_and_detects_bezier_segments(tmp_path: Path) -> None:
    input_dir = tmp_path / "bezier_inputs"
    output_dir = tmp_path / "bezier_outputs"
    create_synthetic_bezier_images(input_dir)

    results = run_bezier_tests(
        input_dir=input_dir,
        output_dir=output_dir,
        image="black_heart_on_white.png",
        debug=True,
        export_mode="centerline",
        fail_on_no_bezier=True,
    )

    assert len(results) == 1
    payload = json.loads(Path(results[0]["output_files"]["summary"]).read_text(encoding="utf-8"))

    assert payload["path_count"] >= 1
    assert payload["segment_count"] >= 1
    assert payload["bezier_segment_count"] >= 1
    assert payload["has_bezier_segment"] is True
    assert payload["integrity_success"] is True
    assert payload["closed_path_count"] >= 1
    assert payload["svg_contains_cubic_command"] is True
    assert payload["dxf_bezier_export_mode"] == "polyline_fallback"
    assert set(payload["dxf_entity_counts"]).issuperset({"CIRCLE", "ARC", "LINE", "LWPOLYLINE", "POLYLINE"})


def test_run_bezier_test_cli_smoke_and_help_mentions_fallback(tmp_path: Path) -> None:
    input_dir = tmp_path / "bezier_inputs"
    output_dir = tmp_path / "bezier_outputs"
    create_synthetic_bezier_images(input_dir)

    exit_code = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--image",
            "blue_blob_on_white.png",
            "--fail-on-no-bezier",
        ]
    )

    assert exit_code == 0

    payload = json.loads((output_dir / "blue_blob_on_white" / "summary.json").read_text(encoding="utf-8"))
    assert payload["bezier_segment_count"] >= 1
    assert payload["has_bezier_segment"] is True

    svg_payload = Path(payload["output_files"]["svg"]).read_text(encoding="utf-8")
    assert "C " in svg_payload or " C " in svg_payload

    help_text = build_arg_parser().format_help()
    assert "Bezier fallback" in help_text or "BezierOptimizer" in help_text
