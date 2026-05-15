from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from core.document import VectorDocument
from core.precision import PrecisionUtility
from core.types import Constraint, Segment


@dataclass(frozen=True, slots=True)
class G1Candidate:
    anchor_id: str
    segments: tuple[str, str]
    angle_error: float
    confidence: float
    reason: str


class G1CandidateDetector:
    def __init__(self, angle_tolerance: float = 0.1) -> None:
        self.angle_tolerance = angle_tolerance

    def detect_candidates(self, document: VectorDocument) -> tuple[G1Candidate, ...]:
        candidates: list[G1Candidate] = []

        for path in document.paths:
            if not path.segments:
                continue

            segments = [
                seg for seg_id in path.segments
                for seg in document.segments if seg.segment_id == seg_id
            ]

            for i in range(len(segments) - 1):
                c = self._check_pair(segments[i], segments[i+1], is_adjacent=True)
                if c:
                    candidates.append(c)

            if path.closed and len(segments) > 1:
                c = self._check_pair(segments[-1], segments[0], is_adjacent=True)
                if c:
                    candidates.append(c)

        visited = set()
        unique_candidates: list[G1Candidate] = []
        for c in candidates:
            key = (c.anchor_id, frozenset(c.segments))
            if key not in visited:
                visited.add(key)
                unique_candidates.append(c)

        return tuple(unique_candidates)

    def _check_pair(self, seg1: Segment, seg2: Segment, is_adjacent: bool) -> G1Candidate | None:
        if seg1.segment_id == seg2.segment_id:
            return None

        anchor_id = self._find_shared_anchor(seg1, seg2, is_adjacent)
        if not anchor_id:
            return None

        t1 = self._get_outward_tangent(seg1, anchor_id)
        t2 = self._get_outward_tangent(seg2, anchor_id)

        if not t1 or not t2:
            return None

        if (PrecisionUtility.near_zero(t1[0]) and PrecisionUtility.near_zero(t1[1])) or \
           (PrecisionUtility.near_zero(t2[0]) and PrecisionUtility.near_zero(t2[1])):
            return None

        angle1 = math.atan2(t1[1], t1[0])
        angle2 = math.atan2(t2[1], t2[0])
        expected_angle = angle2 + math.pi

        if PrecisionUtility.angle_close(angle1, expected_angle, epsilon=self.angle_tolerance):
            delta = (angle1 - expected_angle + math.pi) % (2.0 * math.pi) - math.pi
            angle_error = abs(delta)
            confidence = max(0.0, 1.0 - (angle_error / self.angle_tolerance)) if self.angle_tolerance > 0 else 1.0
            return G1Candidate(
                anchor_id=anchor_id,
                segments=(seg1.segment_id, seg2.segment_id),
                angle_error=angle_error,
                confidence=confidence,
                reason=f"{seg1.type}-{seg2.type} smooth transition"
            )
        return None

    def _find_shared_anchor(self, seg1: Segment, seg2: Segment, is_adjacent: bool) -> str | None:
        if is_adjacent and seg1.anchors and seg2.anchors:
            if seg1.anchors[-1] == seg2.anchors[0]:
                return seg1.anchors[-1]

        if not seg1.anchors or not seg2.anchors:
            return None

        for a1 in [seg1.anchors[-1], seg1.anchors[0]]:
            for a2 in [seg2.anchors[0], seg2.anchors[-1]]:
                if a1 == a2:
                    return a1
        return None

    def _get_outward_tangent(self, segment: Segment, anchor_id: str) -> tuple[float, float] | None:
        if not segment.anchors:
            return None

        if segment.anchors[0] == anchor_id:
            if segment.type == "line":
                p1 = segment.params["start"]
                p2 = segment.params["end"]
                return (p2[0] - p1[0], p2[1] - p1[1])
            elif segment.type == "bezier":
                p1 = segment.params["start"]
                p2 = segment.params["control1"]
                return (p2[0] - p1[0], p2[1] - p1[1])
            elif segment.type == "arc":
                direction = str(segment.params.get("direction", "ccw")).lower()
                a = float(segment.params["start_angle"])
                if direction == "cw":
                    return (math.sin(a), -math.cos(a))
                else:
                    return (-math.sin(a), math.cos(a))

        elif segment.anchors[-1] == anchor_id:
            if segment.type == "line":
                p1 = segment.params["end"]
                p2 = segment.params["start"]
                return (p2[0] - p1[0], p2[1] - p1[1])
            elif segment.type == "bezier":
                p1 = segment.params["end"]
                p2 = segment.params["control2"]
                return (p2[0] - p1[0], p2[1] - p1[1])
            elif segment.type == "arc":
                direction = str(segment.params.get("direction", "ccw")).lower()
                a = float(segment.params["end_angle"])
                if direction == "cw":
                    return (-math.sin(a), math.cos(a))
                else:
                    return (math.sin(a), -math.cos(a))

        return None

    def generate_constraints(
        self, document: VectorDocument, candidates: Sequence[G1Candidate]
    ) -> tuple[Constraint, ...]:
        existing_targets: set[str] = set()
        for c in document.constraints:
            if c.type in {"shared_tangent", "g1_continuity"} and c.locked:
                existing_targets.update(c.targets)

        results: list[Constraint] = []
        for idx, candidate in enumerate(candidates):
            if candidate.confidence < 0.5:
                continue

            if candidate.anchor_id in existing_targets:
                continue
            if candidate.segments[0] in existing_targets and candidate.segments[1] in existing_targets:
                continue

            constraint = Constraint(
                constraint_id=f"g1_cand_{candidate.anchor_id}_{idx}",
                type="g1_continuity",
                targets=(candidate.segments[0], candidate.segments[1], candidate.anchor_id),
                strength=candidate.confidence * 0.8,
                source="system",
                confidence=candidate.confidence,
                locked=False,
                metadata={
                    "reason": candidate.reason,
                    "angle_error": candidate.angle_error
                }
            )
            results.append(constraint)

        return tuple(results)

__all__ = ["G1Candidate", "G1CandidateDetector"]
