from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.dxf_exporter import DxfExporter
from services.document_integrity import DocumentIntegrityValidator
from services.minimal_pipeline import MinimalPipeline
from services.svg_exporter import SvgExporter
from scripts.run_circle_test import IMAGE_EXTENSIONS, count_dxf_entities, _resolve_images


DEFAULT_INPUT_DIR = Path("test_images/bezier")
DEFAULT_OUTPUT_DIR = Path("out/bezier_test")


def create_synthetic_bezier_images(input_dir: str | Path) -> tuple[Path, ...]:
    directory = Path(input_dir)
    directory.mkdir(parents=True, exist_ok=True)
    heart_path = directory / "black_heart_on_white.png"
    blob_path = directory / "blue_blob_on_white.png"
    if not heart_path.exists():
        _write_freeform_curve_ring_image(heart_path, curve="heart", stroke_bgr=(0, 0, 0))
    if not blob_path.exists():
        _write_freeform_curve_ring_image(blob_path, curve="blob", stroke_bgr=(255, 0, 0))
    return tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))


def run_bezier_tests(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    image: str | None = None,
    export_mode: str = "centerline",
    debug: bool = True,
    fail_on_no_bezier: bool = False,
) -> tuple[dict[str, object], ...]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    create_synthetic_bezier_images(input_root)
    images = _resolve_images(input_root, image)
    if not images:
        raise ValueError(f"no input images found in {input_root}")

    pipeline = MinimalPipeline(segment_type="bezier")
    integrity_validator = DocumentIntegrityValidator()
    svg_exporter = SvgExporter()
    dxf_exporter = DxfExporter()

    results: list[dict[str, object]] = []
    for image_path in images:
        results.append(
            run_single_bezier_test(
                image_path=image_path,
                output_root=output_root,
                pipeline=pipeline,
                integrity_validator=integrity_validator,
                svg_exporter=svg_exporter,
                dxf_exporter=dxf_exporter,
                export_mode=export_mode,
                debug=debug,
            )
        )

    if fail_on_no_bezier and any(not bool(result["has_bezier_segment"]) for result in results):
        raise SystemExit(1)
    return tuple(results)


def run_single_bezier_test(
    *,
    image_path: str | Path,
    output_root: str | Path,
    pipeline: MinimalPipeline,
    integrity_validator: DocumentIntegrityValidator,
    svg_exporter: SvgExporter,
    dxf_exporter: DxfExporter,
    export_mode: str,
    debug: bool,
) -> dict[str, object]:
    image_path = Path(image_path)
    output_dir = Path(output_root) / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_result = pipeline.run_from_file(
        image_path,
        document_id=f"manual_bezier_test_{image_path.stem}",
        debug=debug,
        debug_output_dir=output_dir / "debug" if debug else None,
    )
    document = pipeline_result.document
    integrity = integrity_validator.validate(document)

    json_path = output_dir / "vector_document.json"
    overlay_path = output_dir / "overlay.png"
    svg_path = output_dir / "vector_result.svg"
    dxf_path = output_dir / "vector_result.dxf"
    summary_path = output_dir / "summary.json"
    vision_manifest_path = output_dir / "vision_manifest.json"

    pipeline.export_json(document, json_path)
    pipeline.export_overlay(document, pipeline_result.source_image, overlay_path)
    svg_payload = svg_exporter.export_document(document, export_mode=export_mode)
    svg_path.write_text(svg_payload, encoding="utf-8")
    dxf_payload = dxf_exporter.export_document(document, export_mode=export_mode)
    dxf_path.write_text(dxf_payload, encoding="utf-8")

    bezier_segments = [
        {
            "segment_id": segment.segment_id,
            "path_id": segment.path_id,
            "params": _json_safe(segment.params),
        }
        for segment in document.segments
        if segment.type == "bezier"
    ]
    closed_paths = [path.path_id for path in document.paths if path.closed]
    non_closed_paths = [path.path_id for path in document.paths if not path.closed]

    summary = {
        "input_image": str(image_path),
        "path_count": len(document.paths),
        "segment_count": len(document.segments),
        "bezier_segment_count": len(bezier_segments),
        "has_bezier_segment": bool(bezier_segments),
        "bezier_segments": bezier_segments,
        "closed_path_count": len(closed_paths),
        "closed_path_ids": closed_paths,
        "non_closed_path_ids": non_closed_paths,
        "integrity_success": integrity.success,
        "integrity_errors": [issue.code for issue in integrity.errors],
        "export_mode": export_mode,
        "svg_contains_cubic_command": " C " in svg_payload or "C " in svg_payload,
        "dxf_entity_counts": count_dxf_entities(dxf_payload),
        "dxf_bezier_export_mode": "polyline_fallback",
        "output_files": {
            "vector_document": str(json_path),
            "overlay": str(overlay_path),
            "svg": str(svg_path),
            "dxf": str(dxf_path),
            "summary": str(summary_path),
            "vision_manifest": str(vision_manifest_path),
        },
        "debug_output_dir": None if pipeline_result.debug_artifacts is None else str(pipeline_result.debug_artifacts.output_dir),
        "notes": "This script validates current Bezier fallback expression only; it does not imply a P2 BezierOptimizer exists.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    vision_manifest = {
        "input_image": str(image_path),
        "overlay": str(overlay_path),
        "svg_file": str(svg_path),
        "dxf_file": str(dxf_path),
        "summary_file": str(summary_path),
        "expected_shape": "freeform_bezier_closed_curve",
        "expected_result": "Final vector document should contain at least one bezier segment, and DXF may use polyline fallback instead of native spline.",
        "notes": "Current validation covers Bezier fallback expression only, not a future P2 BezierOptimizer.",
        "svg_preview.png": None,
        "dxf_preview.png": None,
    }
    vision_manifest_path.write_text(json.dumps(vision_manifest, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate current closed freeform Bezier fallback expression. This does not imply a P2 BezierOptimizer already exists."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image")
    parser.add_argument("--export-mode", choices=("outline", "centerline", "all_debug"), default="centerline")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.add_argument("--fail-on-no-bezier", action="store_true")
    parser.set_defaults(debug=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summaries = run_bezier_tests(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image=args.image,
            export_mode=args.export_mode,
            debug=args.debug,
            fail_on_no_bezier=args.fail_on_no_bezier,
        )
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for summary in summaries:
        print(json.dumps(summary, indent=2))
    return 0


def _write_freeform_curve_ring_image(
    output_path: Path,
    *,
    curve: str,
    stroke_bgr: tuple[int, int, int],
    size: tuple[int, int] = (320, 280),
    thickness: int = 12,
) -> None:
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)

    if curve == "heart":
        points = _heart_curve_points(center=center, scale=7.0, count=240)
    elif curve == "blob":
        points = _blob_curve_points(center=center, base_radius=76.0, count=280)
    else:
        raise ValueError(f"unsupported synthetic bezier curve: {curve}")

    cv2.polylines(image, [points], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"unable to write synthetic test image: {output_path}")


def _heart_curve_points(*, center: np.ndarray, scale: float, count: int) -> np.ndarray:
    points: list[list[int]] = []
    for index in range(count):
        t = (2.0 * np.pi * index) / count
        x = 16.0 * np.sin(t) ** 3
        y = 13.0 * np.cos(t) - 5.0 * np.cos(2.0 * t) - 2.0 * np.cos(3.0 * t) - np.cos(4.0 * t)
        px = center[0] + (x * scale)
        py = center[1] - (y * scale)
        points.append([int(round(px)), int(round(py))])
    return np.asarray(points, dtype=np.int32)


def _blob_curve_points(*, center: np.ndarray, base_radius: float, count: int) -> np.ndarray:
    points: list[list[int]] = []
    for index in range(count):
        t = (2.0 * np.pi * index) / count
        radius = base_radius + 16.0 * np.sin(3.0 * t) + 8.0 * np.cos(5.0 * t)
        px = center[0] + radius * np.cos(t)
        py = center[1] + radius * np.sin(t)
        points.append([int(round(px)), int(round(py))])
    return np.asarray(points, dtype=np.int32)


def _json_safe(value: object) -> object:
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    raise SystemExit(main())
