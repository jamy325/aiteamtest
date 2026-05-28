from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np


class FreePenCanvasError(ValueError):
    pass


@dataclass(slots=True)
class FreePenPathRecord:
    path_id: str
    closed: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "closed": self.closed,
            "segments": [dict(segment) for segment in self.segments],
        }

    def drawable_segment_count(self) -> int:
        return sum(1 for segment in self.segments if segment["type"] in {"line", "cubic"})

    def has_cubic_segment(self) -> bool:
        return any(segment["type"] == "cubic" for segment in self.segments)

    def points(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        current_point: tuple[float, float] | None = None
        subpath_start: tuple[float, float] | None = None
        for segment in self.segments:
            segment_type = segment["type"]
            if segment_type == "move":
                current_point = (float(segment["p"][0]), float(segment["p"][1]))
                subpath_start = current_point
                points.append(current_point)
            elif segment_type == "line":
                current_point = (float(segment["p"][0]), float(segment["p"][1]))
                points.append(current_point)
            elif segment_type == "cubic":
                current_point = (float(segment["p"][0]), float(segment["p"][1]))
                points.append(current_point)
                points.append((float(segment["c1"][0]), float(segment["c1"][1])))
                points.append((float(segment["c2"][0]), float(segment["c2"][1])))
            elif segment_type == "close" and subpath_start is not None:
                points.append(subpath_start)
        return points


@dataclass(slots=True)
class FreePenCanvasState:
    width: int
    height: int
    coordinate_space: str = "image_px"
    path_open: bool = False
    current_point: tuple[float, float] | None = None
    current_subpath_start: tuple[float, float] | None = None
    current_path_id: str | None = None
    paths: list[FreePenPathRecord] = field(default_factory=list)
    step_count: int = 0
    successful_step_count: int = 0
    invalid_step_count: int = 0
    final_status: str = "initialized"

    def clear(self) -> None:
        self.path_open = False
        self.current_point = None
        self.current_subpath_start = None
        self.current_path_id = None
        self.paths = []

    def replay_tool_calls(self, tool_calls: Iterable[dict[str, Any]]) -> None:
        self.clear()
        self.step_count = 0
        self.successful_step_count = 0
        self.invalid_step_count = 0
        for tool_call in tool_calls:
            self.apply_tool_call(tool_call, count_step=False)

    def apply_tool_call(self, tool_call: dict[str, Any], *, count_step: bool = True) -> dict[str, Any]:
        if count_step:
            self.step_count += 1
        tool = str(tool_call.get("tool", "")).strip()
        if tool == "start_path":
            executed = self.start_path(x=tool_call.get("x"), y=tool_call.get("y"))
        elif tool == "line_to":
            executed = self.line_to(x=tool_call.get("x"), y=tool_call.get("y"))
        elif tool == "curve_to":
            executed = self.curve_to(
                c1=tool_call.get("c1"),
                c2=tool_call.get("c2"),
                p=tool_call.get("p"),
            )
        elif tool == "close_path":
            executed = self.close_path()
        else:
            self.invalid_step_count += 1
            raise FreePenCanvasError(f"unsupported free-pen tool: {tool}")
        self.successful_step_count += 1
        return executed

    def start_path(self, *, x: Any, y: Any) -> dict[str, Any]:
        if self.path_open:
            self.invalid_step_count += 1
            raise FreePenCanvasError("cannot start_path while another path is open")
        point = self._validated_point(x=x, y=y, label="start_path")
        path_id = f"path_{len(self.paths) + 1:03d}"
        path = FreePenPathRecord(
            path_id=path_id,
            segments=[{"type": "move", "p": [point[0], point[1]]}],
        )
        self.paths.append(path)
        self.path_open = True
        self.current_point = point
        self.current_subpath_start = point
        self.current_path_id = path_id
        return {
            "tool": "start_path",
            "path_id": path_id,
            "point": [point[0], point[1]],
        }

    def line_to(self, *, x: Any, y: Any) -> dict[str, Any]:
        path = self._require_open_path(tool="line_to")
        point = self._validated_point(x=x, y=y, label="line_to")
        path.segments.append({"type": "line", "p": [point[0], point[1]]})
        self.current_point = point
        return {
            "tool": "line_to",
            "path_id": path.path_id,
            "point": [point[0], point[1]],
        }

    def curve_to(self, *, c1: Any, c2: Any, p: Any) -> dict[str, Any]:
        path = self._require_open_path(tool="curve_to")
        control_1 = self._validated_point_like(c1, label="curve_to.c1")
        control_2 = self._validated_point_like(c2, label="curve_to.c2")
        end_point = self._validated_point_like(p, label="curve_to.p")
        path.segments.append(
            {
                "type": "cubic",
                "c1": [control_1[0], control_1[1]],
                "c2": [control_2[0], control_2[1]],
                "p": [end_point[0], end_point[1]],
            }
        )
        self.current_point = end_point
        return {
            "tool": "curve_to",
            "path_id": path.path_id,
            "c1": [control_1[0], control_1[1]],
            "c2": [control_2[0], control_2[1]],
            "p": [end_point[0], end_point[1]],
        }

    def close_path(self) -> dict[str, Any]:
        path = self._require_open_path(tool="close_path")
        path.segments.append({"type": "close"})
        path.closed = True
        self.path_open = False
        self.current_point = None
        self.current_subpath_start = None
        closed_path_id = path.path_id
        self.current_path_id = None
        return {
            "tool": "close_path",
            "path_id": closed_path_id,
        }

    def current_path(self) -> FreePenPathRecord | None:
        if self.current_path_id is None:
            return None
        for path in self.paths:
            if path.path_id == self.current_path_id:
                return path
        return None

    def current_path_drawable_segment_count(self) -> int:
        path = self.current_path()
        return 0 if path is None else path.drawable_segment_count()

    def current_path_has_cubic_segment(self) -> bool:
        path = self.current_path()
        return False if path is None else path.has_cubic_segment()

    def current_path_bbox_diagonal(self) -> float:
        path = self.current_path()
        if path is None:
            return 0.0
        points = path.points()
        if not points:
            return 0.0
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def distance_to_start(self) -> float | None:
        if self.current_point is None or self.current_subpath_start is None:
            return None
        return math.hypot(
            self.current_point[0] - self.current_subpath_start[0],
            self.current_point[1] - self.current_subpath_start[1],
        )

    def line_to_streak(self) -> int:
        path = self.current_path()
        if path is None:
            return 0
        streak = 0
        for segment in reversed(path.segments):
            segment_type = segment["type"]
            if segment_type == "line":
                streak += 1
                continue
            if segment_type in {"cubic", "move", "close"}:
                break
        return streak

    def render_overlay(
        self,
        *,
        stroke_width: int = 2,
        stroke_rgba: tuple[int, int, int, int] = (0, 128, 255, 255),
        sample_count_per_segment: int = 64,
    ) -> np.ndarray:
        overlay = np.zeros((int(self.height), int(self.width), 4), dtype=np.uint8)
        color = tuple(int(channel) for channel in stroke_rgba)
        for path in self.paths:
            current_point: tuple[float, float] | None = None
            subpath_start: tuple[float, float] | None = None
            for segment in path.segments:
                segment_type = segment["type"]
                if segment_type == "move":
                    current_point = (float(segment["p"][0]), float(segment["p"][1]))
                    subpath_start = current_point
                    continue
                if segment_type == "line":
                    if current_point is None:
                        continue
                    next_point = (float(segment["p"][0]), float(segment["p"][1]))
                    poly = np.asarray(
                        [
                            [int(round(current_point[0])), int(round(current_point[1]))],
                            [int(round(next_point[0])), int(round(next_point[1]))],
                        ],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        overlay,
                        [poly],
                        isClosed=False,
                        color=color,
                        thickness=max(1, int(stroke_width)),
                        lineType=cv2.LINE_AA,
                    )
                    current_point = next_point
                    continue
                if segment_type == "cubic":
                    if current_point is None:
                        continue
                    sampled = self._sample_cubic_segment(
                        p0=current_point,
                        c1=(float(segment["c1"][0]), float(segment["c1"][1])),
                        c2=(float(segment["c2"][0]), float(segment["c2"][1])),
                        p1=(float(segment["p"][0]), float(segment["p"][1])),
                        sample_count=sample_count_per_segment,
                    )
                    if len(sampled) >= 2:
                        cv2.polylines(
                            overlay,
                            [sampled],
                            isClosed=False,
                            color=color,
                            thickness=max(1, int(stroke_width)),
                            lineType=cv2.LINE_AA,
                        )
                    current_point = (float(segment["p"][0]), float(segment["p"][1]))
                    continue
                if segment_type == "close" and current_point is not None and subpath_start is not None:
                    poly = np.asarray(
                        [
                            [int(round(current_point[0])), int(round(current_point[1]))],
                            [int(round(subpath_start[0])), int(round(subpath_start[1]))],
                        ],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        overlay,
                        [poly],
                        isClosed=False,
                        color=color,
                        thickness=max(1, int(stroke_width)),
                        lineType=cv2.LINE_AA,
                    )
                    current_point = subpath_start
        return overlay

    def render_composite(
        self,
        source_image: np.ndarray,
        *,
        stroke_width: int = 2,
        stroke_rgba: tuple[int, int, int, int] = (0, 128, 255, 255),
        sample_count_per_segment: int = 64,
    ) -> np.ndarray:
        source_rgba = self._to_rgba(source_image)
        overlay = self.render_overlay(
            stroke_width=stroke_width,
            stroke_rgba=stroke_rgba,
            sample_count_per_segment=sample_count_per_segment,
        )
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        composite_rgb = (
            source_rgba[:, :, :3].astype(np.float32) * (1.0 - alpha)
            + overlay[:, :, :3].astype(np.float32) * alpha
        )
        composite = source_rgba.copy()
        composite[:, :, :3] = np.clip(composite_rgb, 0.0, 255.0).astype(np.uint8)
        composite[:, :, 3] = 255
        return composite

    def paths_payload(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "coordinate_space": self.coordinate_space,
            "paths": [path.to_dict() for path in self.paths],
        }

    def state_summary(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "coordinate_space": self.coordinate_space,
            "path_open": bool(self.path_open),
            "current_point": None if self.current_point is None else [self.current_point[0], self.current_point[1]],
            "current_subpath_start": None
            if self.current_subpath_start is None
            else [self.current_subpath_start[0], self.current_subpath_start[1]],
            "path_count": len(self.paths),
            "step_count": int(self.step_count),
            "successful_step_count": int(self.successful_step_count),
            "invalid_step_count": int(self.invalid_step_count),
            "distance_to_start": self.distance_to_start(),
            "line_to_streak": self.line_to_streak(),
            "final_status": self.final_status,
        }

    def _require_open_path(self, *, tool: str) -> FreePenPathRecord:
        if not self.path_open or self.current_path_id is None:
            self.invalid_step_count += 1
            raise FreePenCanvasError(f"{tool} requires an open path")
        for path in self.paths:
            if path.path_id == self.current_path_id:
                return path
        self.invalid_step_count += 1
        raise FreePenCanvasError(f"{tool} requires an existing current path")

    def _validated_point(self, *, x: Any, y: Any, label: str) -> tuple[float, float]:
        return self._validated_point_like([x, y], label=label)

    def _validated_point_like(self, point: Any, *, label: str) -> tuple[float, float]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            self.invalid_step_count += 1
            raise FreePenCanvasError(f"{label} must be a 2-item point")
        try:
            x = self._validated_scalar(point[0], label=f"{label}.x")
            y = self._validated_scalar(point[1], label=f"{label}.y")
        except FreePenCanvasError:
            self.invalid_step_count += 1
            raise
        if x < 0.0 or x >= float(self.width):
            self.invalid_step_count += 1
            raise FreePenCanvasError(f"{label}.x must satisfy 0 <= x < width")
        if y < 0.0 or y >= float(self.height):
            self.invalid_step_count += 1
            raise FreePenCanvasError(f"{label}.y must satisfy 0 <= y < height")
        return (x, y)

    @staticmethod
    def _validated_scalar(value: Any, *, label: str) -> float:
        if isinstance(value, bool):
            raise FreePenCanvasError(f"{label} must be a finite number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise FreePenCanvasError(f"{label} must be a finite number") from exc
        if not math.isfinite(numeric):
            raise FreePenCanvasError(f"{label} must be finite")
        return numeric

    @staticmethod
    def _sample_cubic_segment(
        *,
        p0: tuple[float, float],
        c1: tuple[float, float],
        c2: tuple[float, float],
        p1: tuple[float, float],
        sample_count: int,
    ) -> np.ndarray:
        sample_total = max(8, int(sample_count))
        p0_arr = np.array(p0, dtype=np.float64)
        c1_arr = np.array(c1, dtype=np.float64)
        c2_arr = np.array(c2, dtype=np.float64)
        p1_arr = np.array(p1, dtype=np.float64)
        samples: list[list[int]] = []
        for index in range(sample_total):
            t = index / float(sample_total - 1)
            point = (
                ((1.0 - t) ** 3) * p0_arr
                + 3.0 * ((1.0 - t) ** 2) * t * c1_arr
                + 3.0 * (1.0 - t) * (t**2) * c2_arr
                + (t**3) * p1_arr
            )
            samples.append([int(round(point[0])), int(round(point[1]))])
        return np.asarray(samples, dtype=np.int32)

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        if image.shape[2] == 4:
            return image.copy()
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        raise FreePenCanvasError(f"unsupported source image shape: {image.shape}")
