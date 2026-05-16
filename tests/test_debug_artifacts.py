from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.types import CoordinateSystem
from services.minimal_pipeline import MinimalPipeline


FIXTURE_ROOT = Path("tests/fixtures/debug_artifacts")


def _artifact_dir(root: Path) -> Path:
    directories = [path for path in root.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


def test_minimal_pipeline_debug_exports_black_square_fixture_artifacts(tmp_path: Path) -> None:
    pipeline = MinimalPipeline(coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 120.0, 120.0)))

    result = pipeline.run_from_file(
        FIXTURE_ROOT / "black_square_on_white.png",
        document_id="black_square_debug",
        debug=True,
        debug_output_dir=tmp_path,
    )

    assert result.debug_artifacts is not None
    output_dir = _artifact_dir(tmp_path)
    assert output_dir == result.debug_artifacts.output_dir

    expected_files = {
        "original.png",
        "grayscale.png",
        "threshold_binary.png",
        "denoised.png",
        "morphology_closed.png",
        "binary_contours_overlay.png",
        "binary_contours_hierarchy.json",
        "skeleton_mask.png",
        "skeleton_contours_overlay.png",
        "resampled_contours_overlay.png",
        "vector_overlay_debug.png",
        "debug_summary.json",
    }
    assert expected_files.issubset(set(result.debug_artifacts.exported_files))
    assert not (output_dir / "alpha.png").exists()
    assert not (output_dir / "alpha_mask.png").exists()

    summary = json.loads((output_dir / "debug_summary.json").read_text(encoding="utf-8"))
    assert summary["document_id"] == "black_square_debug"
    assert summary["image_size"] == {"width": 120, "height": 120, "channels": 3}
    assert set(summary["contour_counts"]) == {"binary", "skeleton"}
    assert isinstance(summary["segment_count"], int)
    assert set(summary["filter_counts"]) == {
        "binary_contours_skipped_for_vectorization",
        "skeleton_contours_skipped_for_vectorization",
    }
    assert set(summary["timings_ms"]).issuperset(
        {"grayscale", "binary_preprocess", "binary_contours", "skeleton", "resampling", "vectorization"}
    )
    assert summary["threshold_polarity"] == "foreground_white"


def test_minimal_pipeline_debug_exports_blue_circle_fixture_hierarchy_metadata(tmp_path: Path) -> None:
    pipeline = MinimalPipeline(coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 128.0, 128.0)))

    result = pipeline.run_from_file(
        FIXTURE_ROOT / "blue_circle_on_white.png",
        document_id="blue_circle_debug",
        debug=True,
        debug_output_dir=tmp_path,
    )

    assert result.debug_artifacts is not None
    output_dir = _artifact_dir(tmp_path)
    hierarchy = json.loads((output_dir / "binary_contours_hierarchy.json").read_text(encoding="utf-8"))

    assert hierarchy
    required_fields = {"contour_id", "area", "bbox", "depth", "parent", "children", "touches_border", "bbox_coverage"}
    assert all(required_fields.issubset(item) for item in hierarchy)
    assert (output_dir / "binary_contours_overlay.png").exists()
    assert (output_dir / "skeleton_contours_overlay.png").exists()
    assert (output_dir / "vector_overlay_debug.png").exists()


def test_minimal_pipeline_debug_exports_alpha_masks_when_present(tmp_path: Path) -> None:
    image = np.zeros((80, 80, 4), dtype=np.uint8)
    cv2.circle(image, (40, 40), 20, (255, 0, 0, 255), thickness=-1)

    pipeline = MinimalPipeline(coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 80.0, 80.0)))
    result = pipeline.run(
        image,
        document_id="alpha_debug",
        debug=True,
        debug_output_dir=tmp_path,
    )

    assert result.debug_artifacts is not None
    output_dir = _artifact_dir(tmp_path)
    assert (output_dir / "alpha.png").exists()
    assert (output_dir / "alpha_mask.png").exists()


def test_minimal_pipeline_debug_stage_filter_limits_output_files(tmp_path: Path) -> None:
    pipeline = MinimalPipeline(coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 120.0, 120.0)))

    result = pipeline.run_from_file(
        FIXTURE_ROOT / "black_square_on_white.png",
        document_id="filtered_debug",
        debug=True,
        debug_output_dir=tmp_path,
        debug_stages=("binary_contours_hierarchy", "debug_summary"),
    )

    assert result.debug_artifacts is not None
    output_dir = _artifact_dir(tmp_path)
    produced = {path.name for path in output_dir.iterdir()}
    assert produced == {"binary_contours_hierarchy.json", "debug_summary.json"}


def test_minimal_pipeline_debug_false_has_no_file_side_effects(tmp_path: Path) -> None:
    pipeline = MinimalPipeline(coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 120.0, 120.0)))

    result = pipeline.run_from_file(
        FIXTURE_ROOT / "black_square_on_white.png",
        document_id="no_debug",
    )

    assert result.debug_artifacts is None
    assert list(tmp_path.iterdir()) == []
