from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from core.types import ShapeCandidate, VectorDocument
from services.distance_field_diff import DistanceFieldDiffRenderer
from services.renderer import Renderer


@dataclass(frozen=True, slots=True)
class AIReviewContextBudget:
    max_prompt_chars: int = 120_000
    max_review_jobs: int = 10
    max_candidate_count: int = 20
    max_images_per_job: int = 3
    max_crop_size_px: int = 512
    crop_padding_px: int = 16
    enable_sampled_points: bool = False
    max_sampled_points: int = 0


@dataclass(frozen=True, slots=True)
class AIReviewContextResult:
    original_image_path: str | None
    overlay_image_path: str | None
    diff_image_path: str | None
    review_jobs: tuple[dict[str, Any], ...]
    document_summary: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    algorithm_commands: tuple[dict[str, Any], ...]
    prompt_budget: dict[str, Any]
    truncated: bool
    prompt_char_count: int
    image_count: int
    image_file_count: int
    panel_count: int
    candidate_count: int
    sampled_point_count: int


class AIReviewContextBuilder:
    def __init__(
        self,
        *,
        renderer: Renderer | None = None,
        distance_field_diff_renderer: DistanceFieldDiffRenderer | None = None,
        budget: AIReviewContextBudget | None = None,
    ) -> None:
        self.renderer = renderer or Renderer()
        self.distance_field_diff_renderer = distance_field_diff_renderer or DistanceFieldDiffRenderer()
        self.budget = budget or AIReviewContextBudget()

    def build(
        self,
        *,
        document: VectorDocument,
        source_image: np.ndarray | None,
        candidates: tuple[ShapeCandidate, ...],
        algorithm_commands: tuple[dict[str, Any], ...],
        fit_error: float,
        complexity_score: float,
        topology_status: str,
        self_intersection_count: int,
        processing_summary: dict[str, Any] | None = None,
        previous_issues: tuple[dict[str, Any], ...] = (),
    ) -> AIReviewContextResult:
        if source_image is None:
            source_image = np.full(
                (
                    max(1, int(round(document.height))),
                    max(1, int(round(document.width))),
                    3,
                ),
                255,
                dtype=np.uint8,
            )
        overlay = self.renderer.render_overlay(document, source_image)
        diff_result = self.distance_field_diff_renderer.render_diff(document)
        diff_image = diff_result.image

        sorted_candidates = tuple(sorted(candidates, key=lambda item: item.confidence, reverse=True))
        candidate_summaries = tuple(
            _candidate_summary(candidate) for candidate in sorted_candidates[: self.budget.max_candidate_count]
        )
        algorithm_summaries = tuple(
            _command_summary(command) for command in algorithm_commands[: self.budget.max_candidate_count]
        )
        selected_path_ids = self._select_path_ids(
            document=document,
            candidates=sorted_candidates,
            algorithm_commands=algorithm_commands,
        )
        contour_by_id = _source_contour_by_id(document)
        path_by_id = {path.path_id: path for path in document.paths}

        original_crops: list[np.ndarray] = []
        overlay_crops: list[np.ndarray] = []
        diff_crops: list[np.ndarray] = []
        review_jobs: list[dict[str, Any]] = []
        any_truncated = False

        for index, path_id in enumerate(selected_path_ids, start=1):
            path = path_by_id.get(path_id)
            if path is None:
                continue
            bbox = _path_bbox(document, path, contour_by_id)
            crop_bbox = _expand_and_limit_bbox(
                bbox,
                width=source_image.shape[1],
                height=source_image.shape[0],
                padding=self.budget.crop_padding_px,
                max_size=self.budget.max_crop_size_px,
            )
            original_crops.append(_crop_image(source_image, crop_bbox))
            overlay_crops.append(_crop_image(overlay, crop_bbox))
            diff_crops.append(_crop_image(diff_image, crop_bbox))

            path_candidates = [_candidate_summary(candidate) for candidate in sorted_candidates if candidate.path_id == path_id]
            path_commands = [_command_summary(command) for command in algorithm_commands if str(command.get("path_id", "")) == path_id]
            candidate_limit = min(3, len(path_candidates))
            command_limit = min(3, len(path_commands))
            job_truncated = len(path_candidates) > candidate_limit or len(path_commands) > command_limit
            any_truncated = any_truncated or job_truncated
            review_jobs.append(
                {
                    "job_id": f"review_job_{index}",
                    "path_id": path_id,
                    "window_id": f"{path_id}:window_1",
                    "panel_index": index,
                    "contour_source": path.source,
                    "crop_bbox": [crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]],
                    "crop_size": [crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]],
                    "image_count": self.budget.max_images_per_job,
                    "truncated": job_truncated,
                    "path_summary": {
                        "closed": bool(path.closed),
                        "topology_status": str(path.topology_status),
                        "self_intersection_count": int(path.self_intersection_count),
                        "locked": bool(path.locked),
                        "stroke_semantic": str(path.metadata.get("stroke_semantic", "")),
                        "stroke_width": _finite_or_none(path.metadata.get("stroke_width")),
                        "stroke_width_confidence": _finite_or_none(path.metadata.get("stroke_width_confidence")),
                    },
                    "candidate_summaries": path_candidates[:candidate_limit],
                    "algorithm_command_summaries": path_commands[:command_limit],
                }
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="ai-review-context-"))
        original_path = _write_sheet(
            temp_dir / "original_crops.png",
            original_crops,
            "original",
            max_edge=self.budget.max_crop_size_px,
        )
        overlay_path = _write_sheet(
            temp_dir / "overlay_crops.png",
            overlay_crops,
            "overlay",
            max_edge=self.budget.max_crop_size_px,
        )
        diff_path = _write_sheet(
            temp_dir / "diff_crops.png",
            diff_crops,
            "diff",
            max_edge=self.budget.max_crop_size_px,
        )
        document_summary = {
            "document_id": document.document_id,
            "width": float(document.width),
            "height": float(document.height),
            "path_count": len(document.paths),
            "segment_count": len(document.segments),
            "constraint_count": len(document.constraints),
            "fit_error": float(fit_error),
            "complexity_score": float(complexity_score),
            "topology_status": topology_status,
            "self_intersection_count": int(self_intersection_count),
            "processing_summary": dict(processing_summary or {}),
            "selected_path_ids": list(selected_path_ids[: len(review_jobs)]),
            "previous_issue_count": len(previous_issues),
            "stroke_mask_error": _finite_or_none(diff_result.stroke_mask_error),
        }
        prompt_budget = {
            "max_prompt_chars": self.budget.max_prompt_chars,
            "max_review_jobs": self.budget.max_review_jobs,
            "max_candidate_count": self.budget.max_candidate_count,
            "max_images_per_job": self.budget.max_images_per_job,
            "max_crop_size_px": self.budget.max_crop_size_px,
            "enable_sampled_points": self.budget.enable_sampled_points,
            "max_sampled_points": self.budget.max_sampled_points,
        }
        return AIReviewContextResult(
            original_image_path=None if original_path is None else str(original_path),
            overlay_image_path=None if overlay_path is None else str(overlay_path),
            diff_image_path=None if diff_path is None else str(diff_path),
            review_jobs=tuple(review_jobs),
            document_summary=document_summary,
            candidates=candidate_summaries,
            algorithm_commands=algorithm_summaries,
            prompt_budget=prompt_budget,
            truncated=any_truncated,
            prompt_char_count=0,
            image_count=sum(1 for item in (original_path, overlay_path, diff_path) if item is not None),
            image_file_count=sum(1 for item in (original_path, overlay_path, diff_path) if item is not None),
            panel_count=len(review_jobs) * self.budget.max_images_per_job,
            candidate_count=len(candidate_summaries),
            sampled_point_count=0,
        )

    def _select_path_ids(
        self,
        *,
        document: VectorDocument,
        candidates: tuple[ShapeCandidate, ...],
        algorithm_commands: tuple[dict[str, Any], ...],
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()

        def remember(path_id: str | None) -> None:
            normalized = str(path_id or "").strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            ordered.append(normalized)

        for candidate in candidates:
            remember(candidate.path_id)
        for command in algorithm_commands:
            remember(command.get("path_id"))
        for path in document.paths:
            remember(path.path_id)
        return tuple(ordered[: self.budget.max_review_jobs])


def _candidate_summary(candidate: ShapeCandidate) -> dict[str, Any]:
    evidence = dict(candidate.evidence)
    return {
        "candidate_id": candidate.candidate_id,
        "target_type": candidate.target_type,
        "path_id": candidate.path_id,
        "segment_range": [candidate.segment_range[0], candidate.segment_range[1]],
        "source": candidate.source,
        "confidence": round(float(candidate.confidence), 4),
        "reason": candidate.reason,
        "model_complexity_delta": _finite_or_none(evidence.get("model_complexity_delta")),
        "fit_error": _finite_or_none(evidence.get("fit_error")),
        "inlier_ratio": _finite_or_none(evidence.get("inlier_ratio")),
        "aspect_ratio": _finite_or_none(evidence.get("aspect_ratio")),
        "raw_point_count": _int_or_none(evidence.get("raw_point_count")),
        "segment_count": _int_or_none(evidence.get("segment_count")),
    }


def _command_summary(command: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "tool": str(command.get("tool", command.get("command_type", ""))),
        "path_id": str(command.get("path_id", "")),
        "reason": str(command.get("reason", "")),
        "confidence": _finite_or_none(command.get("confidence")),
        "candidate_id": command.get("candidate_id"),
        "proposal_source": str(command.get("proposal_source", "algorithm")),
    }
    if command.get("segment_range") is not None:
        summary["segment_range"] = list(command["segment_range"])
    return summary


def _source_contour_by_id(document: VectorDocument) -> dict[str, dict[str, Any]]:
    contour_by_id: dict[str, dict[str, Any]] = {}
    pipeline = document.metadata.get("pipeline", {})
    source_contours = pipeline.get("source_contours", {})
    for group in ("binary_contours", "skeleton_contours"):
        for contour in source_contours.get(group, ()):
            contour_id = str(contour.get("contour_id", "")).strip()
            if contour_id:
                contour_by_id[contour_id] = dict(contour)
    return contour_by_id


def _path_bbox(
    document: VectorDocument,
    path,
    contour_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int, int, int]:
    contour_id = str(path.metadata.get("source_contour_id", "")).strip()
    contour = contour_by_id.get(contour_id)
    if contour is not None:
        points = contour.get("points", ())
        if points:
            return _bbox_from_points(points)
    segment_points: list[tuple[float, float]] = []
    segment_ids = set(path.segments)
    for segment in document.segments:
        if segment.segment_id not in segment_ids:
            continue
        segment_points.extend(_segment_points(segment))
    if segment_points:
        return _bbox_from_points(segment_points)
    return (0, 0, max(1, int(round(document.width))), max(1, int(round(document.height))))


def _segment_points(segment) -> list[tuple[float, float]]:
    params = dict(segment.params)
    points: list[tuple[float, float]] = []
    for key in ("start", "end", "center", "p0", "p1", "p2", "p3"):
        value = params.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            points.append((float(value[0]), float(value[1])))
    for key in ("control_points", "points"):
        value = params.get(key)
        if isinstance(value, (list, tuple)):
            for point in value:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    points.append((float(point[0]), float(point[1])))
    radius = params.get("radius")
    if points and isinstance(radius, (int, float)):
        center = points[0]
        radius_value = float(radius)
        points.extend(
            [
                (center[0] - radius_value, center[1] - radius_value),
                (center[0] + radius_value, center[1] + radius_value),
            ]
        )
    return points


def _bbox_from_points(points: Any) -> tuple[int, int, int, int]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x = int(math.floor(min(xs)))
    min_y = int(math.floor(min(ys)))
    max_x = int(math.ceil(max(xs)))
    max_y = int(math.ceil(max(ys)))
    if max_x <= min_x:
        max_x = min_x + 1
    if max_y <= min_y:
        max_y = min_y + 1
    return (min_x, min_y, max_x, max_y)


def _expand_and_limit_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding: int,
    max_size: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_width = min(max_size, max(1, right - left))
    crop_height = min(max_size, max(1, bottom - top))
    crop_width = min(width, crop_width)
    crop_height = min(height, crop_height)
    left = int(round(center_x - (crop_width / 2.0)))
    top = int(round(center_y - (crop_height / 2.0)))
    left = min(max(0, left), max(0, width - crop_width))
    top = min(max(0, top), max(0, height - crop_height))
    right = left + crop_width
    bottom = top + crop_height
    return (left, top, right, bottom)


def _crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bbox
    return image[top:bottom, left:right].copy()


def _write_sheet(output_path: Path, images: list[np.ndarray], prefix: str, *, max_edge: int) -> Path | None:
    if not images:
        return None
    sheet = _contact_sheet(images, prefix)
    sheet = _resize_max_edge(sheet, max_edge=max_edge)
    success, encoded = cv2.imencode(".png", sheet)
    if not success:
        raise ValueError(f"failed to encode {prefix} contact sheet")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def _contact_sheet(images: list[np.ndarray], prefix: str) -> np.ndarray:
    normalized = [_to_bgr(image) for image in images]
    tile_height = max(image.shape[0] for image in normalized) + 24
    tile_width = max(image.shape[1] for image in normalized)
    columns = 2 if len(normalized) > 1 else 1
    rows = int(math.ceil(len(normalized) / columns))
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    for index, image in enumerate(normalized, start=1):
        row = (index - 1) // columns
        column = (index - 1) % columns
        y = row * tile_height
        x = column * tile_width
        canvas[y + 20 : y + 20 + image.shape[0], x : x + image.shape[1]] = image
        cv2.putText(
            canvas,
            f"{prefix}:{index}",
            (x + 4, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _resize_max_edge(image: np.ndarray, *, max_edge: int) -> np.ndarray:
    current_max_edge = max(int(image.shape[0]), int(image.shape[1]))
    if current_max_edge <= max_edge:
        return image
    scale = float(max_edge) / float(current_max_edge)
    target_width = max(1, int(round(image.shape[1] * scale)))
    target_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _finite_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return int(value)


__all__ = [
    "AIReviewContextBudget",
    "AIReviewContextBuilder",
    "AIReviewContextResult",
]
