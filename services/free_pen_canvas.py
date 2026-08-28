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
        elif tool == "move_anchor":
            executed = self.move_anchor(anchor_id=tool_call.get("anchor_id"), x=tool_call.get("x"), y=tool_call.get("y"))
        elif tool == "move_handle":
            executed = self.move_handle(
                segment_id=tool_call.get("segment_id"),
                handle=tool_call.get("handle"),
                x=tool_call.get("x"),
                y=tool_call.get("y"),
            )
        elif tool == "set_segment_handles":
            executed = self.set_segment_handles(
                segment_id=tool_call.get("segment_id"),
                c1=tool_call.get("c1"),
                c2=tool_call.get("c2"),
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
        distance_to_start_before_close = self.distance_to_start()
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
            "distance_to_start_before_close": distance_to_start_before_close,
        }

    def move_anchor(self, *, anchor_id: Any, x: Any, y: Any) -> dict[str, Any]:
        path = self._require_open_path(tool="move_anchor")
        anchor_name = str(anchor_id or "").strip()
        anchor_index = self._anchor_index_for_id(anchor_name, path=path)
        point = self._validated_point(x=x, y=y, label="move_anchor")
        old_point = self._anchor_point(path=path, anchor_index=anchor_index)
        self._set_anchor_point(path=path, anchor_index=anchor_index, point=point)
        return {
            "tool": "move_anchor",
            "path_id": path.path_id,
            "anchor_id": anchor_name,
            "old_point": [old_point[0], old_point[1]],
            "point": [point[0], point[1]],
        }

    def move_handle(self, *, segment_id: Any, handle: Any, x: Any, y: Any) -> dict[str, Any]:
        path = self._require_open_path(tool="move_handle")
        segment_name = str(segment_id or "").strip()
        handle_name = str(handle or "").strip().lower()
        if handle_name not in {"c1", "c2"}:
            self.invalid_step_count += 1
            raise FreePenCanvasError("move_handle.handle must be c1 or c2")
        segment = self._cubic_segment_for_id(segment_name, path=path)
        point = self._validated_point(x=x, y=y, label=f"move_handle.{handle_name}")
        old_point = (float(segment[handle_name][0]), float(segment[handle_name][1]))
        segment[handle_name] = [point[0], point[1]]
        return {
            "tool": "move_handle",
            "path_id": path.path_id,
            "segment_id": segment_name,
            "handle": handle_name,
            "old_point": [old_point[0], old_point[1]],
            "point": [point[0], point[1]],
        }

    def set_segment_handles(self, *, segment_id: Any, c1: Any, c2: Any) -> dict[str, Any]:
        path = self._require_open_path(tool="set_segment_handles")
        segment_name = str(segment_id or "").strip()
        segment = self._cubic_segment_for_id(segment_name, path=path)
        new_c1 = self._validated_point_like(c1, label="set_segment_handles.c1")
        new_c2 = self._validated_point_like(c2, label="set_segment_handles.c2")
        old_c1 = (float(segment["c1"][0]), float(segment["c1"][1]))
        old_c2 = (float(segment["c2"][0]), float(segment["c2"][1]))
        segment["c1"] = [new_c1[0], new_c1[1]]
        segment["c2"] = [new_c2[0], new_c2[1]]
        return {
            "tool": "set_segment_handles",
            "path_id": path.path_id,
            "segment_id": segment_name,
            "old_c1": [old_c1[0], old_c1[1]],
            "old_c2": [old_c2[0], old_c2[1]],
            "c1": [new_c1[0], new_c1[1]],
            "c2": [new_c2[0], new_c2[1]],
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
        show_handles: bool = False,
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
        if show_handles:
            self._draw_editable_geometry_annotations(composite)
        return composite

    def paths_payload(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "coordinate_space": self.coordinate_space,
            "paths": [path.to_dict() for path in self.paths],
        }

    def editable_geometry(self) -> dict[str, Any]:
        active_paths = [self.current_path()] if self.current_path() is not None else ([self.paths[-1]] if self.paths else [])
        payload_paths: list[dict[str, Any]] = []
        for path in active_paths:
            if path is None:
                continue
            payload_paths.append(self._editable_geometry_for_path(path))
        return {"paths": payload_paths}

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

    def _editable_geometry_for_path(self, path: FreePenPathRecord) -> dict[str, Any]:
        anchors: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        anchor_index = 0
        drawable_index = 0
        for segment in path.segments:
            segment_type = segment["type"]
            if segment_type == "move":
                anchor_index += 1
                anchors.append({"id": f"A{anchor_index}", "p": [float(segment["p"][0]), float(segment["p"][1])]})
                continue
            if segment_type not in {"line", "cubic"}:
                continue
            drawable_index += 1
            from_anchor = f"A{anchor_index}"
            anchor_index += 1
            to_anchor = f"A{anchor_index}"
            anchors.append({"id": to_anchor, "p": [float(segment["p"][0]), float(segment["p"][1])]})
            segment_payload = {
                "id": f"S{drawable_index}",
                "type": segment_type,
                "from_anchor": from_anchor,
                "to_anchor": to_anchor,
            }
            if segment_type == "line":
                segment_payload["p"] = [float(segment["p"][0]), float(segment["p"][1])]
            else:
                segment_payload["c1"] = [float(segment["c1"][0]), float(segment["c1"][1])]
                segment_payload["c2"] = [float(segment["c2"][0]), float(segment["c2"][1])]
            segments.append(segment_payload)
        return {
            "path_id": path.path_id,
            "anchors": anchors,
            "segments": segments,
        }

    def _anchor_index_for_id(self, anchor_id: str, *, path: FreePenPathRecord) -> int:
        geometry = self._editable_geometry_for_path(path)
        for index, anchor in enumerate(geometry["anchors"], start=1):
            if anchor["id"] == anchor_id:
                return index
        self.invalid_step_count += 1
        raise FreePenCanvasError(f"unknown anchor_id: {anchor_id}")

    def _anchor_point(self, *, path: FreePenPathRecord, anchor_index: int) -> tuple[float, float]:
        if anchor_index == 1:
            move_segment = path.segments[0]
            return (float(move_segment["p"][0]), float(move_segment["p"][1]))
        drawable_segments = [segment for segment in path.segments if segment["type"] in {"line", "cubic"}]
        segment = drawable_segments[anchor_index - 2]
        return (float(segment["p"][0]), float(segment["p"][1]))

    def _set_anchor_point(self, *, path: FreePenPathRecord, anchor_index: int, point: tuple[float, float]) -> None:
        if anchor_index == 1:
            path.segments[0]["p"] = [point[0], point[1]]
            if self.current_subpath_start is not None:
                self.current_subpath_start = point
        else:
            drawable_segments = [segment for segment in path.segments if segment["type"] in {"line", "cubic"}]
            segment = drawable_segments[anchor_index - 2]
            segment["p"] = [point[0], point[1]]
        if self.current_point is not None:
            anchor_count = len(self._editable_geometry_for_path(path)["anchors"])
            if anchor_index == anchor_count:
                self.current_point = point

    def _cubic_segment_for_id(self, segment_id: str, *, path: FreePenPathRecord) -> dict[str, Any]:
        geometry = self._editable_geometry_for_path(path)
        drawable_segments = [segment for segment in path.segments if segment["type"] in {"line", "cubic"}]
        for index, segment in enumerate(geometry["segments"]):
            if segment["id"] == segment_id:
                target = drawable_segments[index]
                if target["type"] != "cubic":
                    self.invalid_step_count += 1
                    raise FreePenCanvasError(f"{segment_id} is not a cubic segment")
                return target
        self.invalid_step_count += 1
        raise FreePenCanvasError(f"unknown segment_id: {segment_id}")

    def _draw_editable_geometry_annotations(self, image: np.ndarray) -> None:
        blue = (255, 0, 0, 255)
        green = (0, 200, 0, 255)
        black = (0, 0, 0, 255)
        white = (255, 255, 255, 255)
        for path in self.editable_geometry()["paths"]:
            anchors = {anchor["id"]: anchor["p"] for anchor in path["anchors"]}
            for anchor_id, point in anchors.items():
                center = (int(round(point[0])), int(round(point[1])))
                cv2.circle(image, center, 4, blue, thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(image, anchor_id, (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 3, cv2.LINE_AA)
                cv2.putText(image, anchor_id, (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, black, 1, cv2.LINE_AA)
            for segment in path["segments"]:
                if segment["type"] != "cubic":
                    continue
                from_point = anchors[segment["from_anchor"]]
                to_point = anchors[segment["to_anchor"]]
                c1 = segment["c1"]
                c2 = segment["c2"]
                c1_center = (int(round(c1[0])), int(round(c1[1])))
                c2_center = (int(round(c2[0])), int(round(c2[1])))
                cv2.line(image, (int(round(from_point[0])), int(round(from_point[1]))), c1_center, green, 1, cv2.LINE_AA)
                cv2.line(image, (int(round(to_point[0])), int(round(to_point[1]))), c2_center, green, 1, cv2.LINE_AA)
                cv2.circle(image, c1_center, 3, green, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(image, c2_center, 3, green, thickness=-1, lineType=cv2.LINE_AA)
                mid_x = int(round((from_point[0] + to_point[0]) / 2.0))
                mid_y = int(round((from_point[1] + to_point[1]) / 2.0))
                cv2.putText(image, segment["id"], (mid_x + 4, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 3, cv2.LINE_AA)
                cv2.putText(image, segment["id"], (mid_x + 4, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, black, 1, cv2.LINE_AA)
