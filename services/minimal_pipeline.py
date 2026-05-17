from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.document import add_anchor, add_path, add_segment, create_document, to_json
from core.types import CoordinateSystem, VectorDocument, updated
from services.contour_extractor import BinaryContour, ContourExtractor, ExtractedContours
from services.debug_artifacts import DebugArtifactExportResult, DebugArtifactExporter
from services.distance_field_diff import DistanceFieldDiffRenderer
from services.json_exporter import JsonExporter
from services.renderer import Renderer
from services.resampler import Resampler
from services.simple_vectorizer import InitialSegmentType, SimpleVectorizer
from services.skeleton_graph import SkeletonJunction


@dataclass(frozen=True, slots=True)
class MinimalPipelineResult:
    document: VectorDocument
    json_payload: str
    extracted_contours: ExtractedContours
    source_image: np.ndarray | None = None
    debug_artifacts: DebugArtifactExportResult | None = None


class MinimalPipeline:
    def __init__(
        self,
        coordinate_system: CoordinateSystem | None = None,
        *,
        contour_extractor: ContourExtractor | None = None,
        resampler: Resampler | None = None,
        segment_type: InitialSegmentType = "line",
        json_exporter: JsonExporter | None = None,
        renderer: Renderer | None = None,
        distance_field_diff_renderer: DistanceFieldDiffRenderer | None = None,
    ) -> None:
        self.coordinate_system = coordinate_system or CoordinateSystem()
        self.contour_extractor = contour_extractor or ContourExtractor(
            coordinate_transformer=CoordinateTransformer(self.coordinate_system)
        )
        self.resampler = resampler or Resampler()
        self.segment_type = segment_type
        self.json_exporter = json_exporter or JsonExporter()
        self.renderer = renderer or Renderer()
        self.distance_field_diff_renderer = distance_field_diff_renderer or DistanceFieldDiffRenderer()

    def run(
        self,
        image: np.ndarray,
        *,
        document_id: str = "document_1",
        debug: bool = False,
        debug_output_dir: str | Path | None = None,
        debug_stages: tuple[str, ...] | list[str] | None = None,
    ) -> MinimalPipelineResult:
        debug_enabled = bool(debug or debug_output_dir is not None)
        if debug_enabled:
            extracted_contours, contour_debug = self.contour_extractor.extract_contours_with_debug(image)
        else:
            extracted_contours = self.contour_extractor.extract_contours(image)
            contour_debug = None
        height, width = image.shape[:2]
        document = create_document(
            document_id=document_id,
            width=float(width),
            height=float(height),
            coordinate_system=self.coordinate_system,
        )

        resample_start = perf_counter()
        resampled_binary = tuple(self._resample_contour(contour) for contour in extracted_contours.binary_contours)
        resampled_skeleton = tuple(
            self._resample_contour(contour, simplify_linear=True)
            for contour in extracted_contours.skeleton_contours
        )
        resample_elapsed_ms = (perf_counter() - resample_start) * 1000.0

        document = updated(
            document,
            metadata={
                "pipeline": {
                    "segment_type": self.segment_type,
                    "source_contours": {
                        "binary_contours": [
                            self._serialize_contour(contour) for contour in extracted_contours.binary_contours
                        ],
                        "skeleton_contours": [
                            self._serialize_contour(contour) for contour in extracted_contours.skeleton_contours
                        ],
                    },
                    "skeleton_junctions": [
                        self._serialize_junction(junction) for junction in extracted_contours.skeleton_junctions
                    ],
                    "resampled_contours": {
                        "binary_contours": [
                            self._serialize_resampled(contour_id, points)
                            for contour_id, points in resampled_binary
                        ],
                        "skeleton_contours": [
                            self._serialize_resampled(contour_id, points)
                            for contour_id, points in resampled_skeleton
                        ],
                    },
                }
            },
        )

        vectorizer = SimpleVectorizer(segment_type=self.segment_type)
        skipped_binary_count = 0
        skipped_skeleton_count = 0
        vectorize_start = perf_counter()
        for prefix, contours, resampled in (
            ("binary", extracted_contours.binary_contours, dict(resampled_binary)),
            ("skeleton", extracted_contours.skeleton_contours, dict(resampled_skeleton)),
        ):
            for index, contour in enumerate(contours):
                resampled_points = resampled[contour.contour_id]
                minimum_points = 3 if contour.closed else 2
                if len(resampled_points) < minimum_points:
                    if prefix == "binary":
                        skipped_binary_count += 1
                    else:
                        skipped_skeleton_count += 1
                    continue

                vectorized = vectorizer.vectorize_contour(
                    resampled_points,
                    path_id=f"{prefix}_path_{index}",
                    closed=contour.closed,
                    source=contour.source,
                )
                document = add_path(document, vectorized.path)
                for anchor in vectorized.anchors:
                    document = add_anchor(document, anchor)
                for segment in vectorized.segments:
                    document = add_segment(document, segment)
        vectorize_elapsed_ms = (perf_counter() - vectorize_start) * 1000.0

        debug_artifacts: DebugArtifactExportResult | None = None
        if debug_enabled and contour_debug is not None:
            timings_ms = dict(contour_debug.timings_ms)
            timings_ms["resampling"] = resample_elapsed_ms
            timings_ms["vectorization"] = vectorize_elapsed_ms
            contour_debug = type(contour_debug)(
                grayscale=contour_debug.grayscale,
                alpha=contour_debug.alpha,
                alpha_mask=contour_debug.alpha_mask,
                threshold_binary=contour_debug.threshold_binary,
                denoised=contour_debug.denoised,
                morphology_closed=contour_debug.morphology_closed,
                binary_contours_overlay=contour_debug.binary_contours_overlay,
                binary_contours_hierarchy=contour_debug.binary_contours_hierarchy,
                skeleton_mask=contour_debug.skeleton_mask,
                skeleton_contours_overlay=contour_debug.skeleton_contours_overlay,
                binary_contours=contour_debug.binary_contours,
                skeleton_contours=contour_debug.skeleton_contours,
                timings_ms=timings_ms,
                threshold_polarity=contour_debug.threshold_polarity,
                foreground_mode=contour_debug.foreground_mode,
                foreground_reason=contour_debug.foreground_reason,
                filtered_binary_contours=contour_debug.filtered_binary_contours,
                filtered_skeleton_contours=contour_debug.filtered_skeleton_contours,
            )
            debug_artifacts = DebugArtifactExporter(
                output_root=debug_output_dir,
                debug_stages=debug_stages,
                renderer=self.renderer,
            ).export_pipeline_debug(
                document_id=document_id,
                source_image=image,
                document=document,
                contour_debug=contour_debug,
                resampled_binary=resampled_binary,
                resampled_skeleton=resampled_skeleton,
                skipped_binary_count=skipped_binary_count,
                skipped_skeleton_count=skipped_skeleton_count,
                vectorized_path_count=len(document.paths),
                vectorized_segment_count=len(document.segments),
            )

        return MinimalPipelineResult(
            document=document,
            json_payload=to_json(document),
            extracted_contours=extracted_contours,
            source_image=image.copy(),
            debug_artifacts=debug_artifacts,
        )

    def run_from_file(
        self,
        image_path: str | Path,
        *,
        document_id: str = "document_1",
        debug: bool = False,
        debug_output_dir: str | Path | None = None,
        debug_stages: tuple[str, ...] | list[str] | None = None,
    ) -> MinimalPipelineResult:
        image = self.load_image(image_path)
        return self.run(
            image,
            document_id=document_id,
            debug=debug,
            debug_output_dir=debug_output_dir,
            debug_stages=debug_stages,
        )

    def load_image(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"unable to load image: {image_path}")
        return image

    def export_json(self, document: VectorDocument, output_path: str | Path | None = None) -> str:
        payload = self.json_exporter.export_document(document)
        if output_path is not None:
            Path(output_path).write_text(payload, encoding="utf-8")
        return payload

    def render_overlay(self, document: VectorDocument, image: np.ndarray) -> np.ndarray:
        return self.renderer.render_overlay(document, image)

    def export_overlay(
        self,
        document: VectorDocument,
        image: np.ndarray,
        output_path: str | Path | None = None,
    ) -> bytes:
        encoded = self.renderer.export_overlay_png(document, image)
        if output_path is not None:
            Path(output_path).write_bytes(encoded)
        return encoded

    def render_distance_field_diff(self, document: VectorDocument) -> np.ndarray:
        return self.distance_field_diff_renderer.render_diff(document).image

    def export_distance_field_diff(
        self,
        document: VectorDocument,
        output_path: str | Path | None = None,
    ) -> bytes:
        encoded = self.distance_field_diff_renderer.export_diff_png(document)
        if output_path is not None:
            Path(output_path).write_bytes(encoded)
        return encoded

    def _resample_contour(
        self,
        contour: BinaryContour,
        *,
        simplify_linear: bool = False,
    ) -> tuple[str, tuple[tuple[float, float], ...]]:
        if contour.coordinate_space != "vector":
            raise ValueError("minimal pipeline expects Vector Space contours")
        points = self.resampler.resample(contour.points, closed=contour.closed)
        if simplify_linear:
            points = self.resampler.simplify_linear_contour(points, closed=contour.closed)
        return (contour.contour_id, points)

    def _serialize_contour(self, contour: BinaryContour) -> dict[str, Any]:
        return {
            "contour_id": contour.contour_id,
            "source": contour.source,
            "points": [list(point) for point in contour.points],
            "coordinate_space": contour.coordinate_space,
            "closed": contour.closed,
            "area": contour.area,
            "depth": contour.depth,
            "parent_contour": contour.parent_contour,
            "children": list(contour.children),
        }

    def _serialize_resampled(
        self,
        contour_id: str,
        points: tuple[tuple[float, float], ...],
    ) -> dict[str, Any]:
        return {
            "contour_id": contour_id,
            "points": [list(point) for point in points],
            "coordinate_space": "vector",
        }

    def _serialize_junction(self, junction: SkeletonJunction) -> dict[str, Any]:
        return {
            "pixel": list(junction.pixel),
            "junction_id": junction.junction_id,
            "degree": junction.degree,
            "endpoints": [
                {
                    "path_index": endpoint.path_index,
                    "is_start": endpoint.is_start,
                }
                for endpoint in junction.endpoints
            ],
        }


__all__ = ["MinimalPipeline", "MinimalPipelineResult"]
