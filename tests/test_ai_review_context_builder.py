from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.ai_review_context_builder import AIReviewContextBuilder


def _document():
    document = create_document(
        document_id="context_doc",
        width=96.0,
        height=96.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": "binary_1",
                            "points": [[24.0, 24.0], [72.0, 24.0], [72.0, 72.0], [24.0, 72.0]],
                            "coordinate_space": "vector",
                            "closed": True,
                        }
                    ],
                    "skeleton_contours": [],
                },
                "resampled_contours": {
                    "binary_contours": [{"contour_id": "binary_1", "points": [[24.0, 24.0], [72.0, 72.0]]}],
                    "skeleton_contours": [],
                },
            }
        },
    )
    document = add_path(
        document,
        VectorPath(
            path_id="path_1",
            source="binary_contour",
            closed=True,
            segments=("seg_1",),
            metadata={"source_contour_id": "binary_1"},
        ),
    )
    document = add_segment(
        document,
        Segment(
            segment_id="seg_1",
            path_id="path_1",
            type="line",
            params={"start": [24.0, 24.0], "end": [72.0, 72.0]},
        ),
    )
    return document


def _image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (72, 72), (0, 0, 0), thickness=2)
    return image


def _multi_path_document(path_count: int):
    binary_contours = []
    document = create_document(
        document_id="context_doc_multi",
        width=2048.0,
        height=2048.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
        metadata={"pipeline": {"source_contours": {"binary_contours": binary_contours, "skeleton_contours": []}}},
    )
    for index in range(path_count):
        left = 40.0 + ((index % 3) * 640.0)
        top = 40.0 + ((index // 3) * 320.0)
        contour_id = f"binary_{index}"
        binary_contours.append(
            {
                "contour_id": contour_id,
                "points": [[left, top], [left + 220.0, top], [left + 220.0, top + 220.0], [left, top + 220.0]],
                "coordinate_space": "vector",
                "closed": True,
            }
        )
        path_id = f"path_{index}"
        segment_id = f"seg_{index}"
        document = add_path(
            document,
            VectorPath(
                path_id=path_id,
                source="binary_contour",
                closed=True,
                segments=(segment_id,),
                metadata={"source_contour_id": contour_id},
            ),
        )
        document = add_segment(
            document,
            Segment(
                segment_id=segment_id,
                path_id=path_id,
                type="line",
                params={"start": [left, top], "end": [left + 220.0, top + 220.0]},
            ),
        )
    return document


def test_ai_review_context_builder_uses_local_visual_context_without_full_document_dump(tmp_path: Path) -> None:
    builder = AIReviewContextBuilder()
    result = builder.build(
        document=_document(),
        source_image=_image(),
        candidates=(),
        algorithm_commands=(),
        fit_error=12.0,
        complexity_score=5.0,
        topology_status="closed",
        self_intersection_count=0,
        processing_summary={"processing_contour_source": "binary"},
    )

    assert result.original_image_path is not None
    assert result.overlay_image_path is not None
    assert result.diff_image_path is not None
    assert Path(result.original_image_path).is_file()
    assert Path(result.overlay_image_path).is_file()
    assert Path(result.diff_image_path).is_file()
    assert len(result.review_jobs) == 1
    assert result.image_count == 3
    assert result.image_file_count == 3
    assert result.panel_count == 3
    job = result.review_jobs[0]
    assert job["image_count"] == 3
    assert job["crop_size"][0] <= 512
    assert job["crop_size"][1] <= 512
    original_sheet = cv2.imread(str(result.original_image_path), cv2.IMREAD_UNCHANGED)
    assert original_sheet is not None
    assert max(original_sheet.shape[0], original_sheet.shape[1]) <= 512

    payload = json.dumps(
        {
            "document_summary": result.document_summary,
            "review_jobs": result.review_jobs,
            "candidates": result.candidates,
            "algorithm_commands": result.algorithm_commands,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    for forbidden in ("source_contours", "resampled_contours", "\"paths\"", "\"segments\"", "\"anchors\""):
        assert forbidden not in payload


def test_ai_review_context_builder_limits_candidate_count_to_budget() -> None:
    from core.types import ShapeCandidate

    candidates = tuple(
        ShapeCandidate(
            candidate_id=f"cand_{index}",
            target_type="circle",
            path_id="path_1",
            segment_range=(0, 0),
            source="raw_contour_points",
            confidence=1.0 - (index * 0.01),
            evidence={"fit_error": 0.1 + index},
            reason="candidate",
        )
        for index in range(30)
    )
    builder = AIReviewContextBuilder()
    result = builder.build(
        document=_document(),
        source_image=_image(),
        candidates=candidates,
        algorithm_commands=(),
        fit_error=12.0,
        complexity_score=5.0,
        topology_status="closed",
        self_intersection_count=0,
        processing_summary={"processing_contour_source": "binary"},
    )

    assert len(result.candidates) == 20
    assert result.candidate_count == 20


def test_ai_review_context_builder_resizes_contact_sheet_to_budget() -> None:
    from core.types import ShapeCandidate

    document = _multi_path_document(10)
    image = np.full((2048, 2048, 3), 255, dtype=np.uint8)
    candidates = tuple(
        ShapeCandidate(
            candidate_id=f"cand_{index}",
            target_type="circle",
            path_id=f"path_{index}",
            segment_range=(0, 0),
            source="raw_contour_points",
            confidence=1.0 - (index * 0.01),
            evidence={},
            reason="candidate",
        )
        for index in range(12)
    )
    builder = AIReviewContextBuilder()
    result = builder.build(
        document=document,
        source_image=image,
        candidates=candidates,
        algorithm_commands=(),
        fit_error=1.0,
        complexity_score=1.0,
        topology_status="closed",
        self_intersection_count=0,
        processing_summary={"processing_contour_source": "binary"},
    )

    for path in (result.original_image_path, result.overlay_image_path, result.diff_image_path):
        assert path is not None
        sheet = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert sheet is not None
        assert max(sheet.shape[0], sheet.shape[1]) <= 512
