from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.precision import PrecisionUtility
from core.types import Path as VectorPath
from core.types import Point, VectorDocument
from services.command_executor import CommandExecutionResult, CommandExecutor
from services.command_preview import CommandPreviewResult, CommandPreviewService
from services.document_integrity import DocumentIntegrityValidator
from services.dxf_exporter import DxfExporter
from services.minimal_pipeline import MinimalPipeline
from services.segment_sampler import SegmentSampler
from services.svg_exporter import SvgExporter


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_INPUT_DIR = Path("test_images/circle")
DEFAULT_OUTPUT_DIR = Path("out/circle_test")


class PathCandidate(dict[str, object]):
    pass


def create_synthetic_circle_images(input_dir: str | Path) -> tuple[Path, ...]:
    directory = Path(input_dir)
    directory.mkdir(parents=True, exist_ok=True)
    black_path = directory / "black_circle_on_white.png"
    blue_path = directory / "blue_circle_on_white.png"
    if not black_path.exists():
        _write_circle_ring_image(black_path, ring_bgr=(0, 0, 0))
    if not blue_path.exists():
        _write_circle_ring_image(blue_path, ring_bgr=(255, 0, 0))
    return _image_files(directory)


def run_circle_tests(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    image: str | None = None,
    export_mode: str = "centerline",
    debug: bool = True,
    min_path_area: float = 256.0,
    prefer_source: str = "skeleton_contour",
    fail_on_no_circle: bool = False,
) -> tuple[dict[str, object], ...]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    create_synthetic_circle_images(input_root)
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
        summary = run_single_circle_test(
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
        results.append(summary)

    if fail_on_no_circle and any(not bool(result["has_circle_segment"]) for result in results):
        raise SystemExit(1)
    return tuple(results)


def run_single_circle_test(
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
        document_id=f"manual_circle_test_{image_path.stem}",
        debug=debug,
        debug_output_dir=output_dir / "debug" if debug else None,
    )
    before_document = pipeline_result.document
    candidates = rank_circle_path_candidates(
        before_document,
        sampler=sampler,
        min_path_area=min_path_area,
        prefer_source=prefer_source,
    )
    selected = select_circle_path_candidate(candidates, prefer_source=prefer_source)

    preview_result: CommandPreviewResult | None = None
    execute_result: CommandExecutionResult | None = None
    command: dict[str, object] | None = None
    if selected is not None:
        command = {
            "command_id": f"manual_circle_replace_{image_path.stem}",
            "tool": "propose_replace_path_with_circle",
            "path_id": str(selected["path_id"]),
            "reason": "manual circle test: replace selected closed path with circle",
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

    circle_segments = [
        {
            "segment_id": segment.segment_id,
            "path_id": segment.path_id,
            "params": _json_safe(segment.params),
        }
        for segment in after_document.segments
        if segment.type == "circle"
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
        "has_circle_segment": bool(circle_segments),
        "circle_segments": circle_segments,
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
        "expected_shape": "circle",
        "expected_result": "DXF should contain one CIRCLE entity or final vector document should contain one circle segment",
        "notes": "Manual circle replacement verification manifest.",
        "svg_preview.png": None,
        "dxf_preview.png": None,
    }
    vision_manifest_path.write_text(json.dumps(vision_manifest, indent=2), encoding="utf-8")
    return summary


def rank_circle_path_candidates(
    document: VectorDocument,
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
        elif len(path.segments) < 3:
            reject_reason = "segment_count_too_small"
        elif bbox is None:
            reject_reason = "path_has_no_geometry"
        elif bbox_area is None or bbox_area < float(min_path_area):
            reject_reason = "bbox_area_too_small"
        elif touches_border:
            reject_reason = "touches_page_border"
        else:
            score = _candidate_score(
                path=path,
                bbox=bbox,
                bbox_area=bbox_area,
                aspect_ratio=1.0 if aspect_ratio is None else aspect_ratio,
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


def select_circle_path_candidate(candidates: Iterable[PathCandidate], *, prefer_source: str) -> PathCandidate | None:
    viable = [item for item in candidates if item.get("reject_reason") is None and item.get("score") is not None]
    if not viable:
        return None
    preferred = [item for item in viable if item.get("source") == prefer_source]
    pool = preferred if preferred else viable
    return max(pool, key=lambda item: float(item["score"]))


def sample_path_points(document: VectorDocument, path: VectorPath, *, sampler: SegmentSampler) -> tuple[Point, ...]:
    lookup = {segment.segment_id: segment for segment in document.segments}
    sampled: list[Point] = []
    for segment_id in path.segments:
        segment = lookup.get(segment_id)
        if segment is None:
            continue
        current = tuple(sampler.sample_segment(segment))
        if not current:
            continue
        if sampled and PrecisionUtility.points_close(sampled[-1], current[0]):
            sampled.extend(current[1:])
        else:
            sampled.extend(current)
    return tuple(sampled)


def path_bbox(points: tuple[Point, ...]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return (float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y))


def bbox_touches_border(bbox: tuple[float, float, float, float], width: float, height: float) -> bool:
    min_x, min_y, bbox_width, bbox_height = bbox
    max_x = min_x + bbox_width
    max_y = min_y + bbox_height
    epsilon = 1.0
    return min_x <= epsilon or min_y <= epsilon or max_x >= float(width) - epsilon or max_y >= float(height) - epsilon


def count_dxf_entities(dxf_payload: str) -> dict[str, int]:
    counts = {"CIRCLE": 0, "ARC": 0, "LINE": 0, "LWPOLYLINE": 0, "POLYLINE": 0}
    lines = dxf_payload.splitlines()
    for index in range(len(lines) - 1):
        if lines[index].strip() != "0":
            continue
        entity_type = lines[index + 1].strip()
        if entity_type in counts:
            counts[entity_type] += 1
    return counts


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
    parser.add_argument("--fail-on-no-circle", action="store_true")
    parser.set_defaults(debug=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summaries = run_circle_tests(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image=args.image,
            export_mode=args.export_mode,
            debug=args.debug,
            min_path_area=args.min_path_area,
            prefer_source=args.prefer_source,
            fail_on_no_circle=args.fail_on_no_circle,
        )
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for summary in summaries:
        print(json.dumps(summary, indent=2))
    return 0


def _write_circle_ring_image(
    output_path: Path,
    *,
    ring_bgr: tuple[int, int, int],
    size: int = 256,
    radius: int = 72,
    thickness: int = 10,
) -> None:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(image, center, radius, ring_bgr, thickness=thickness, lineType=cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"unable to write synthetic test image: {output_path}")


def _image_files(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))


def _resolve_images(input_dir: Path, image: str | None) -> tuple[Path, ...]:
    images = _image_files(input_dir)
    if image is None:
        return images
    selected = [path for path in images if path.name == image or path.stem == image]
    if selected:
        return tuple(selected)
    raise ValueError(f"image not found in {input_dir}: {image}")


def _candidate_score(
    *,
    path: VectorPath,
    bbox: tuple[float, float, float, float],
    bbox_area: float,
    aspect_ratio: float,
    document: VectorDocument,
    prefer_source: str,
) -> float:
    source_bonus = 1000.0 if path.source == prefer_source else 500.0
    circle_ratio_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio))
    document_area = max(float(document.width) * float(document.height), 1.0)
    area_score = min(1.0, bbox_area / document_area)
    segment_score = min(1.0, len(path.segments) / 64.0)
    bbox_square_bonus = min(bbox[2], bbox[3]) / max(max(bbox[2], bbox[3]), 1e-9)
    return source_bonus + (circle_ratio_score * 100.0) + (bbox_square_bonus * 20.0) + (area_score * 10.0) + segment_score


def _segment_type_counts(document: VectorDocument) -> dict[str, int]:
    counts: dict[str, int] = {}
    for segment in document.segments:
        counts[segment.type] = counts.get(segment.type, 0) + 1
    return counts


def _path_by_id(document: VectorDocument, path_id: str) -> VectorPath | None:
    for path in document.paths:
        if path.path_id == path_id:
            return path
    return None


def _json_safe(value: object) -> object:
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    raise SystemExit(main())
