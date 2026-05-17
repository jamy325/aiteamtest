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

from core.types import Path as VectorPath
from services.command_executor import CommandExecutionResult, CommandExecutor
from services.command_preview import CommandPreviewResult, CommandPreviewService
from services.document_integrity import DocumentIntegrityValidator
from services.dxf_exporter import DxfExporter
from services.minimal_pipeline import MinimalPipeline
from services.segment_sampler import SegmentSampler
from services.svg_exporter import SvgExporter
from scripts.run_circle_test import (
    IMAGE_EXTENSIONS,
    PathCandidate,
    _json_safe,
    _path_by_id,
    _resolve_images,
    _segment_type_counts,
    bbox_touches_border,
    count_dxf_entities,
    path_bbox,
    sample_path_points,
)


DEFAULT_INPUT_DIR = Path("test_images/ellipse")
DEFAULT_OUTPUT_DIR = Path("out/ellipse_test")


def create_synthetic_ellipse_images(input_dir: str | Path) -> tuple[Path, ...]:
    directory = Path(input_dir)
    directory.mkdir(parents=True, exist_ok=True)
    black_path = directory / "black_ellipse_on_white.png"
    blue_path = directory / "blue_ellipse_on_white.png"
    if not black_path.exists():
        _write_ellipse_ring_image(black_path, ring_bgr=(0, 0, 0))
    if not blue_path.exists():
        _write_ellipse_ring_image(blue_path, ring_bgr=(255, 0, 0))
    return tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))


def run_ellipse_tests(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    image: str | None = None,
    export_mode: str = "centerline",
    debug: bool = True,
    min_path_area: float = 256.0,
    prefer_source: str = "skeleton_contour",
    fail_on_no_ellipse: bool = False,
) -> tuple[dict[str, object], ...]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    create_synthetic_ellipse_images(input_root)
    images = _resolve_images(input_root, image)
    if not images:
        raise ValueError(f"no input images found in {input_root}")

    pipeline = MinimalPipeline(segment_type="line")
    preview_service = CommandPreviewService()
    command_executor = CommandExecutor()
    integrity_validator = DocumentIntegrityValidator()
    svg_exporter = SvgExporter()
    dxf_exporter = DxfExporter()
    sampler = SegmentSampler()

    results: list[dict[str, object]] = []
    for image_path in images:
        results.append(
            run_single_ellipse_test(
                image_path=image_path,
                output_root=output_root,
                pipeline=pipeline,
                preview_service=preview_service,
                command_executor=command_executor,
                integrity_validator=integrity_validator,
                svg_exporter=svg_exporter,
                dxf_exporter=dxf_exporter,
                sampler=sampler,
                export_mode=export_mode,
                debug=debug,
                min_path_area=min_path_area,
                prefer_source=prefer_source,
            )
        )

    if fail_on_no_ellipse and any(not bool(result["has_ellipse_segment"]) for result in results):
        raise SystemExit(1)
    return tuple(results)


def run_single_ellipse_test(
    *,
    image_path: str | Path,
    output_root: str | Path,
    pipeline: MinimalPipeline,
    preview_service: CommandPreviewService,
    command_executor: CommandExecutor,
    integrity_validator: DocumentIntegrityValidator,
    svg_exporter: SvgExporter,
    dxf_exporter: DxfExporter,
    sampler: SegmentSampler,
    export_mode: str,
    debug: bool,
    min_path_area: float,
    prefer_source: str,
) -> dict[str, object]:
    image_path = Path(image_path)
    output_dir = Path(output_root) / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_result = pipeline.run_from_file(
        image_path,
        document_id=f"manual_ellipse_test_{image_path.stem}",
        debug=debug,
        debug_output_dir=output_dir / "debug" if debug else None,
    )
    before_document = pipeline_result.document
    candidates = rank_ellipse_path_candidates(
        before_document,
        sampler=sampler,
        min_path_area=min_path_area,
        prefer_source=prefer_source,
    )
    selected = select_ellipse_path_candidate(candidates, prefer_source=prefer_source)

    preview_result: CommandPreviewResult | None = None
    execute_result: CommandExecutionResult | None = None
    if selected is not None:
        command = {
            "command_id": f"manual_ellipse_replace_{image_path.stem}",
            "tool": "propose_replace_path_with_ellipse",
            "path_id": str(selected["path_id"]),
            "reason": "manual ellipse test: replace selected closed path with ellipse",
            "confidence": 0.95,
            "requires_user_confirmation": True,
        }
        preview_result = preview_service.preview(command, before_document)
        execute_result = command_executor.execute(command, before_document)

    after_document = before_document if execute_result is None or not execute_result.success else execute_result.document
    integrity = integrity_validator.validate(after_document)

    before_json_path = output_dir / "vector_document_before.json"
    after_json_path = output_dir / "vector_document_after.json"
    overlay_before_path = output_dir / "overlay_before.png"
    overlay_after_path = output_dir / "overlay_after.png"
    svg_path = output_dir / "vector_result.svg"
    dxf_path = output_dir / "vector_result.dxf"
    summary_path = output_dir / "summary.json"
    vision_manifest_path = output_dir / "vision_manifest.json"

    pipeline.export_json(before_document, before_json_path)
    pipeline.export_json(after_document, after_json_path)
    pipeline.export_overlay(before_document, pipeline_result.source_image, overlay_before_path)
    pipeline.export_overlay(after_document, pipeline_result.source_image, overlay_after_path)
    svg_path.write_text(svg_exporter.export_document(after_document, export_mode=export_mode), encoding="utf-8")
    dxf_payload = dxf_exporter.export_document(after_document, export_mode=export_mode)
    dxf_path.write_text(dxf_payload, encoding="utf-8")

    ellipse_segments = [
        {
            "segment_id": segment.segment_id,
            "path_id": segment.path_id,
            "params": _json_safe(segment.params),
        }
        for segment in after_document.segments
        if segment.type == "ellipse"
    ]

    selected_path = None if selected is None else _path_by_id(before_document, str(selected["path_id"]))
    summary = {
        "input_image": str(image_path),
        "selected_path_id": None if selected is None else selected["path_id"],
        "selected_path_source": None if selected is None else selected["source"],
        "selected_path_closed": None if selected is None else selected["closed"],
        "selected_path_segment_count_before": None if selected_path is None else len(selected_path.segments),
        "path_candidates": candidates,
        "preview_success": None if preview_result is None else preview_result.success,
        "preview_score_delta": None if preview_result is None else preview_result.score_delta,
        "execute_success": False if execute_result is None else execute_result.success,
        "execute_reason": "no suitable closed path candidate found" if execute_result is None else execute_result.reason,
        "fitting_source": None if execute_result is None else execute_result.fitting_source,
        "integrity_success": integrity.success,
        "segment_type_counts_before": _segment_type_counts(before_document),
        "segment_type_counts_after": _segment_type_counts(after_document),
        "has_ellipse_segment": bool(ellipse_segments),
        "ellipse_segments": ellipse_segments,
        "export_mode": export_mode,
        "dxf_entity_counts": count_dxf_entities(dxf_payload),
        "output_files": {
            "vector_document_before_json": str(before_json_path),
            "vector_document_after_json": str(after_json_path),
            "overlay_before": str(overlay_before_path),
            "overlay_after": str(overlay_after_path),
            "svg": str(svg_path),
            "dxf": str(dxf_path),
            "summary": str(summary_path),
            "vision_manifest": str(vision_manifest_path),
        },
        "debug_output_dir": None if pipeline_result.debug_artifacts is None else str(pipeline_result.debug_artifacts.output_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    vision_manifest = {
        "input_image": str(image_path),
        "overlay_before": str(overlay_before_path),
        "overlay_after": str(overlay_after_path),
        "svg_file": str(svg_path),
        "dxf_file": str(dxf_path),
        "summary_file": str(summary_path),
        "expected_shape": "ellipse",
        "expected_result": "final vector document should contain one ellipse segment",
        "notes": "Manual ellipse replacement verification manifest.",
        "svg_preview.png": None,
        "dxf_preview.png": None,
    }
    vision_manifest_path.write_text(json.dumps(vision_manifest, indent=2), encoding="utf-8")
    return summary


def rank_ellipse_path_candidates(
    document,
    *,
    sampler: SegmentSampler,
    min_path_area: float,
    prefer_source: str,
) -> list[PathCandidate]:
    candidates: list[PathCandidate] = []
    allowed_sources = {prefer_source, "skeleton_contour", "binary_contour"}
    for path in document.paths:
        points = sample_path_points(document, path, sampler=sampler)
        bbox = path_bbox(points)
        bbox_area = None if bbox is None else bbox[2] * bbox[3]
        aspect_ratio = None if bbox is None or bbox[3] <= 0.0 else bbox[2] / bbox[3]
        touches_border = False if bbox is None else bbox_touches_border(bbox, document.width, document.height)
        reject_reason: str | None = None
        score: float | None = None

        if path.source not in allowed_sources:
            reject_reason = "unsupported_source"
        elif not path.closed:
            reject_reason = "path_not_closed"
        elif len(path.segments) < 4:
            reject_reason = "segment_count_too_small"
        elif bbox is None:
            reject_reason = "path_has_no_geometry"
        elif bbox_area is None or bbox_area < float(min_path_area):
            reject_reason = "bbox_area_too_small"
        elif touches_border:
            reject_reason = "touches_page_border"
        else:
            score = _ellipse_candidate_score(
                path=path,
                bbox_area=bbox_area,
                aspect_ratio=aspect_ratio,
                document=document,
                prefer_source=prefer_source,
            )

        candidates.append(
            PathCandidate(
                path_id=path.path_id,
                source=path.source,
                closed=path.closed,
                segment_count=len(path.segments),
                bbox=None if bbox is None else list(bbox),
                score=score,
                reject_reason=reject_reason,
                touches_border=touches_border,
                bbox_area=bbox_area,
                aspect_ratio=aspect_ratio,
            )
        )
    return candidates


def select_ellipse_path_candidate(candidates: list[PathCandidate], *, prefer_source: str) -> PathCandidate | None:
    viable = [item for item in candidates if item.get("reject_reason") is None and item.get("score") is not None]
    if not viable:
        return None
    preferred = [item for item in viable if item.get("source") == prefer_source]
    pool = preferred if preferred else viable
    return max(pool, key=lambda item: float(item["score"]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image")
    parser.add_argument("--export-mode", choices=("outline", "centerline", "all_debug"), default="centerline")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.add_argument("--min-path-area", type=float, default=256.0)
    parser.add_argument("--prefer-source", default="skeleton_contour")
    parser.add_argument("--fail-on-no-ellipse", action="store_true")
    parser.set_defaults(debug=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summaries = run_ellipse_tests(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image=args.image,
            export_mode=args.export_mode,
            debug=args.debug,
            min_path_area=args.min_path_area,
            prefer_source=args.prefer_source,
            fail_on_no_ellipse=args.fail_on_no_ellipse,
        )
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for summary in summaries:
        print(json.dumps(summary, indent=2))
    return 0


def _write_ellipse_ring_image(
    output_path: Path,
    *,
    ring_bgr: tuple[int, int, int],
    size: tuple[int, int] = (320, 240),
    axes: tuple[int, int] = (92, 56),
    thickness: int = 10,
    angle_degrees: float = 18.0,
) -> None:
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.ellipse(
        image,
        center,
        axes,
        angle_degrees,
        0.0,
        360.0,
        ring_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"unable to write synthetic test image: {output_path}")


def _ellipse_candidate_score(
    *,
    path: VectorPath,
    bbox_area: float,
    aspect_ratio: float | None,
    document,
    prefer_source: str,
) -> float:
    source_bonus = 1000.0 if path.source == prefer_source else 500.0
    document_area = max(float(document.width) * float(document.height), 1.0)
    area_score = min(1.0, bbox_area / document_area)
    segment_score = min(1.0, len(path.segments) / 96.0)
    aspect_bonus = 0.0
    if aspect_ratio is not None and aspect_ratio > 0.0:
        normalized_ratio = aspect_ratio if aspect_ratio >= 1.0 else 1.0 / aspect_ratio
        if normalized_ratio <= 4.0:
            aspect_bonus = 20.0 / normalized_ratio
    return source_bonus + (area_score * 100.0) + (segment_score * 10.0) + aspect_bonus


if __name__ == "__main__":
    raise SystemExit(main())
