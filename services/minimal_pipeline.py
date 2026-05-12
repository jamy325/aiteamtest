from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.document import add_anchor, add_path, add_segment, create_document, to_json
from core.types import CoordinateSystem, VectorDocument, updated
from services.contour_extractor import BinaryContour, ContourExtractor, ExtractedContours
from services.resampler import Resampler
from services.simple_vectorizer import InitialSegmentType, SimpleVectorizer


@dataclass(frozen=True, slots=True)
class MinimalPipelineResult:
    document: VectorDocument
    json_payload: str
    extracted_contours: ExtractedContours


class MinimalPipeline:
    def __init__(
        self,
        coordinate_system: CoordinateSystem | None = None,
        *,
        contour_extractor: ContourExtractor | None = None,
        resampler: Resampler | None = None,
        segment_type: InitialSegmentType = "line",
    ) -> None:
        self.coordinate_system = coordinate_system or CoordinateSystem()
        self.contour_extractor = contour_extractor or ContourExtractor(
            coordinate_transformer=CoordinateTransformer(self.coordinate_system)
        )
        self.resampler = resampler or Resampler()
        self.segment_type = segment_type

    def run(self, image: np.ndarray, *, document_id: str = "document_1") -> MinimalPipelineResult:
        extracted_contours = self.contour_extractor.extract_contours(image)
        height, width = image.shape[:2]
        document = create_document(
            document_id=document_id,
            width=float(width),
            height=float(height),
            coordinate_system=self.coordinate_system,
        )

        resampled_binary = tuple(self._resample_contour(contour) for contour in extracted_contours.binary_contours)
        resampled_skeleton = tuple(self._resample_contour(contour) for contour in extracted_contours.skeleton_contours)

        document = updated(
            document,
            metadata={
                "pipeline": {
                    "segment_type": self.segment_type,
                    "source_contours": {
                        "binary_contours": [self._serialize_contour(contour) for contour in extracted_contours.binary_contours],
                        "skeleton_contours": [self._serialize_contour(contour) for contour in extracted_contours.skeleton_contours],
                    },
                    "resampled_contours": {
                        "binary_contours": [self._serialize_resampled(contour_id, points) for contour_id, points in resampled_binary],
                        "skeleton_contours": [self._serialize_resampled(contour_id, points) for contour_id, points in resampled_skeleton],
                    },
                }
            },
        )

        vectorizer = SimpleVectorizer(segment_type=self.segment_type)
        for prefix, contours in (
            ("binary", extracted_contours.binary_contours),
            ("skeleton", extracted_contours.skeleton_contours),
        ):
            for index, contour in enumerate(contours):
                resampled_points = dict(resampled_binary if prefix == "binary" else resampled_skeleton)[contour.contour_id]
                minimum_points = 3 if contour.closed else 2
                if len(resampled_points) < minimum_points:
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
                    document = add_segment(document, updated(segment, params=self._json_ready_value(segment.params)))

        return MinimalPipelineResult(
            document=document,
            json_payload=to_json(document),
            extracted_contours=extracted_contours,
        )

    def run_from_file(self, image_path: str | Path, *, document_id: str = "document_1") -> MinimalPipelineResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to load image: {image_path}")
        return self.run(image, document_id=document_id)

    def _resample_contour(self, contour: BinaryContour) -> tuple[str, tuple[tuple[float, float], ...]]:
        if contour.coordinate_space != "vector":
            raise ValueError("minimal pipeline expects Vector Space contours")
        points = self.resampler.resample(contour.points, closed=contour.closed)
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

    def _json_ready_value(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return [self._json_ready_value(item) for item in value]
        if isinstance(value, list):
            return [self._json_ready_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._json_ready_value(item) for key, item in value.items()}
        return value


__all__ = ["MinimalPipeline", "MinimalPipelineResult"]
