from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import VectorDocument
from services.contour_extractor import BinaryContour, ContourExtractionDebugArtifacts
from services.renderer import Renderer


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class DebugArtifactExportResult:
    output_dir: Path
    exported_files: tuple[str, ...]
    summary: dict[str, Any]


class DebugArtifactExporter:
    _STAGE_ALIASES = {
        "original": {"original"},
        "grayscale": {"grayscale"},
        "alpha": {"alpha", "alpha_mask"},
        "binary": {"threshold_binary", "denoised", "morphology_closed"},
        "contours": {
            "binary_contours_overlay",
            "binary_contours_hierarchy",
            "skeleton_contours_overlay",
        },
        "skeleton": {"skeleton_mask", "skeleton_contours_overlay"},
        "resampling": {"resampled_contours_overlay"},
        "vector": {"vector_overlay_debug"},
        "summary": {"debug_summary"},
    }

    def __init__(
        self,
        output_root: str | Path | None = None,
        *,
        debug_stages: Iterable[str] | None = None,
        renderer: Renderer | None = None,
    ) -> None:
        if output_root is None:
            output_root = Path(tempfile.gettempdir()) / "curve_fitting_ai_agent_debug"
        self.output_root = Path(output_root)
        self.debug_stages = self._normalize_debug_stages(debug_stages)
        self.renderer = renderer or Renderer()

    def export_pipeline_debug(
        self,
        *,
        document_id: str,
        source_image: np.ndarray,
        document: VectorDocument,
        contour_debug: ContourExtractionDebugArtifacts,
        resampled_binary: tuple[tuple[str, tuple[Point, ...]], ...],
        resampled_skeleton: tuple[tuple[str, tuple[Point, ...]], ...],
        skipped_binary_count: int,
        skipped_skeleton_count: int,
        vectorized_path_count: int,
        vectorized_segment_count: int,
    ) -> DebugArtifactExportResult:
        output_dir = self._create_run_directory(document_id)
        exported_files: list[str] = []

        self._maybe_write_image(output_dir, "original", source_image, exported_files)
        self._maybe_write_image(output_dir, "grayscale", contour_debug.grayscale, exported_files)
        if contour_debug.alpha is not None:
            self._maybe_write_image(output_dir, "alpha", contour_debug.alpha, exported_files)
        if contour_debug.alpha_mask is not None:
            self._maybe_write_image(output_dir, "alpha_mask", contour_debug.alpha_mask, exported_files)
        self._maybe_write_image(output_dir, "threshold_binary", contour_debug.threshold_binary, exported_files)
        self._maybe_write_image(output_dir, "denoised", contour_debug.denoised, exported_files)
        self._maybe_write_image(output_dir, "morphology_closed", contour_debug.morphology_closed, exported_files)
        self._maybe_write_image(output_dir, "binary_contours_overlay", contour_debug.binary_contours_overlay, exported_files)
        self._maybe_write_json(
            output_dir,
            "binary_contours_hierarchy",
            contour_debug.binary_contours_hierarchy,
            exported_files,
        )
        self._maybe_write_image(output_dir, "skeleton_mask", contour_debug.skeleton_mask, exported_files)
        self._maybe_write_image(
            output_dir,
            "skeleton_contours_overlay",
            contour_debug.skeleton_contours_overlay,
            exported_files,
        )

        resampled_overlay = self._draw_resampled_overlay(
            source_image,
            document.coordinate_system,
            contour_debug.binary_contours,
            contour_debug.skeleton_contours,
            resampled_binary,
            resampled_skeleton,
        )
        self._maybe_write_image(output_dir, "resampled_contours_overlay", resampled_overlay, exported_files)

        vector_overlay = self._draw_vector_overlay_debug(document, source_image)
        self._maybe_write_image(output_dir, "vector_overlay_debug", vector_overlay, exported_files)

        summary = {
            "document_id": document.document_id,
            "image_size": {
                "width": int(source_image.shape[1]),
                "height": int(source_image.shape[0]),
                "channels": int(source_image.shape[2]) if source_image.ndim == 3 else 1,
            },
            "contour_counts": {
                "binary": len(contour_debug.binary_contours),
                "skeleton": len(contour_debug.skeleton_contours),
            },
            "segment_count": len(document.segments),
            "path_count": len(document.paths),
            "anchor_count": len(document.anchors),
            "vectorized_path_count": int(vectorized_path_count),
            "vectorized_segment_count": int(vectorized_segment_count),
            "filter_counts": {
                "binary_contours_skipped_for_vectorization": int(skipped_binary_count),
                "skeleton_contours_skipped_for_vectorization": int(skipped_skeleton_count),
            },
            "skeleton_simplification": self._summarize_skeleton_simplification(
                contour_debug.skeleton_contours,
                resampled_skeleton,
            ),
            "filtered_binary_contours": list(contour_debug.filtered_binary_contours),
            "filtered_skeleton_contours": list(contour_debug.filtered_skeleton_contours),
            "timings_ms": dict(contour_debug.timings_ms),
            "threshold_polarity": contour_debug.threshold_polarity,
            "foreground_mode": contour_debug.foreground_mode,
            "foreground_reason": contour_debug.foreground_reason,
            "exported_files": sorted(exported_files),
        }
        self._maybe_write_json(output_dir, "debug_summary", summary, exported_files)
        summary["exported_files"] = sorted(exported_files)

        return DebugArtifactExportResult(
            output_dir=output_dir,
            exported_files=tuple(sorted(exported_files)),
            summary=summary,
        )

    def _summarize_skeleton_simplification(
        self,
        original_contours: tuple[BinaryContour, ...],
        simplified_contours: tuple[tuple[str, tuple[Point, ...]], ...],
    ) -> dict[str, Any]:
        original_lookup = {contour.contour_id: contour for contour in original_contours}
        entries: list[dict[str, Any]] = []
        total_original_points = 0
        total_simplified_points = 0
        total_original_segments = 0
        total_simplified_segments = 0

        for contour_id, simplified_points in simplified_contours:
            original = original_lookup.get(contour_id)
            if original is None:
                continue
            original_point_count = len(original.points)
            simplified_point_count = len(simplified_points)
            original_segment_count = self._segment_count_for_points(original_point_count, original.closed)
            simplified_segment_count = self._segment_count_for_points(simplified_point_count, original.closed)
            total_original_points += original_point_count
            total_simplified_points += simplified_point_count
            total_original_segments += original_segment_count
            total_simplified_segments += simplified_segment_count
            entries.append(
                {
                    "contour_id": contour_id,
                    "closed": original.closed,
                    "original_point_count": original_point_count,
                    "simplified_point_count": simplified_point_count,
                    "original_segment_count": original_segment_count,
                    "simplified_segment_count": simplified_segment_count,
                }
            )

        return {
            "original_point_count": total_original_points,
            "simplified_point_count": total_simplified_points,
            "original_segment_count": total_original_segments,
            "simplified_segment_count": total_simplified_segments,
            "per_contour": entries,
        }

    def _create_run_directory(self, document_id: str) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        base = self.output_root / str(document_id)
        candidate = base
        suffix = 2
        while candidate.exists():
            candidate = self.output_root / f"{document_id}_{suffix}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _should_export(self, stage_name: str) -> bool:
        return self.debug_stages is None or stage_name in self.debug_stages

    def _normalize_debug_stages(self, debug_stages: Iterable[str] | None) -> set[str] | None:
        if debug_stages is None:
            return None
        normalized: set[str] = set()
        for stage in debug_stages:
            stage_name = str(stage)
            normalized.add(stage_name)
            normalized.update(self._STAGE_ALIASES.get(stage_name, ()))
        return normalized

    def _maybe_write_image(
        self,
        output_dir: Path,
        stage_name: str,
        image: np.ndarray,
        exported_files: list[str],
    ) -> None:
        if not self._should_export(stage_name):
            return
        image_path = output_dir / f"{stage_name}.png"
        if not cv2.imwrite(str(image_path), image):
            raise ValueError(f"failed to write debug artifact: {image_path}")
        exported_files.append(image_path.name)

    def _maybe_write_json(
        self,
        output_dir: Path,
        stage_name: str,
        payload: Any,
        exported_files: list[str],
    ) -> None:
        if not self._should_export(stage_name):
            return
        json_path = output_dir / f"{stage_name}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        exported_files.append(json_path.name)

    def _draw_resampled_overlay(
        self,
        source_image: np.ndarray,
        coordinate_system: Any,
        binary_contours: tuple[BinaryContour, ...],
        skeleton_contours: tuple[BinaryContour, ...],
        resampled_binary: tuple[tuple[str, tuple[Point, ...]], ...],
        resampled_skeleton: tuple[tuple[str, tuple[Point, ...]], ...],
    ) -> np.ndarray:
        canvas = self._to_bgr(source_image)
        transformer = CoordinateTransformer(coordinate_system)
        original_lookup = {contour.contour_id: contour for contour in binary_contours + skeleton_contours}
        color_lookup = {
            "binary": ((100, 100, 255), (0, 0, 255)),
            "skeleton": ((100, 255, 100), (0, 180, 0)),
        }

        for group_name, contours in (("binary", resampled_binary), ("skeleton", resampled_skeleton)):
            polyline_color, point_color = color_lookup[group_name]
            for contour_id, points in contours:
                original = original_lookup.get(contour_id)
                if original is not None and len(original.points) >= 2:
                    original_pixels = np.array(
                        [self._point_to_pixel(transformer, point) for point in original.points],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        canvas,
                        [original_pixels],
                        bool(original.closed),
                        polyline_color,
                        1,
                        cv2.LINE_AA,
                    )
                for point in points:
                    cv2.circle(canvas, self._point_to_pixel(transformer, point), 3, point_color, -1)
                if points:
                    cv2.putText(
                        canvas,
                        contour_id,
                        self._point_to_pixel(transformer, points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        point_color,
                        1,
                        cv2.LINE_AA,
                    )
        return canvas

    def _draw_vector_overlay_debug(self, document: VectorDocument, source_image: np.ndarray) -> np.ndarray:
        canvas = self.renderer.render_overlay(document, self._to_bgr(source_image))
        transformer = CoordinateTransformer(document.coordinate_system)
        segment_lookup = {segment.segment_id: segment for segment in document.segments}

        for path in document.paths:
            label_point: Point | None = None
            for segment_id in path.segments:
                segment = segment_lookup.get(segment_id)
                if segment is None:
                    continue
                label_point = self._segment_label_point(segment)
                if label_point is not None:
                    break
            if label_point is None:
                continue
            cv2.putText(
                canvas,
                path.path_id,
                self._point_to_pixel(transformer, label_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _segment_label_point(self, segment: Any) -> Point | None:
        for key in ("start", "end", "center"):
            value = segment.params.get(key)
            point = self._coerce_point(value)
            if point is not None:
                return point
        points = segment.params.get("points")
        if isinstance(points, list) and points:
            return self._coerce_point(points[0])
        return None

    def _coerce_point(self, value: object) -> Point | None:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None

    def _segment_count_for_points(self, point_count: int, closed: bool) -> int:
        if point_count <= 1:
            return 0
        if closed:
            return point_count - 1
        return max(0, point_count - 1)

    def _point_to_pixel(self, transformer: CoordinateTransformer, point: Point) -> tuple[int, int]:
        pixel = transformer.vector_to_pixel(point)
        return (int(round(pixel[0])), int(round(pixel[1])))

    def _to_bgr(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError("image must be grayscale, BGR, or BGRA")


__all__ = ["DebugArtifactExportResult", "DebugArtifactExporter"]
