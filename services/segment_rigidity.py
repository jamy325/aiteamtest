from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.types import Segment, SegmentType

RigidityLevel = Literal["high", "medium_high", "medium", "low"]

RIGIDITY_BY_TYPE: dict[SegmentType, RigidityLevel] = {
    "line": "high",
    "circle": "high",
    "arc": "high",
    "ellipse": "medium_high",
    "bezier": "medium",
    "bspline": "low",
    "polyline": "low",
}

RIGIDITY_RANK: dict[RigidityLevel, int] = {
    "low": 0,
    "medium": 1,
    "medium_high": 2,
    "high": 3,
}


@dataclass(frozen=True, slots=True)
class SegmentMovementDecision:
    move_segment_id: str | None
    reference_segment_id: str | None
    move_rigidity: RigidityLevel | None
    reference_rigidity: RigidityLevel | None
    reason: str
    blocked: bool = False


class SegmentRigidityPolicy:
    def rigidity_for_type(self, segment_type: SegmentType) -> RigidityLevel:
        return RIGIDITY_BY_TYPE[segment_type]

    def rigidity_for_segment(self, segment: Segment) -> RigidityLevel:
        return self.rigidity_for_type(segment.type)

    def choose_segment_to_move(self, left: Segment, right: Segment) -> SegmentMovementDecision:
        left_rigidity = self.rigidity_for_segment(left)
        right_rigidity = self.rigidity_for_segment(right)

        if left.locked and right.locked:
            return SegmentMovementDecision(
                move_segment_id=None,
                reference_segment_id=None,
                move_rigidity=None,
                reference_rigidity=None,
                reason="both_locked",
                blocked=True,
            )

        if left.locked:
            return SegmentMovementDecision(
                move_segment_id=right.segment_id,
                reference_segment_id=left.segment_id,
                move_rigidity=right_rigidity,
                reference_rigidity=left_rigidity,
                reason="left_locked_move_right",
            )

        if right.locked:
            return SegmentMovementDecision(
                move_segment_id=left.segment_id,
                reference_segment_id=right.segment_id,
                move_rigidity=left_rigidity,
                reference_rigidity=right_rigidity,
                reason="right_locked_move_left",
            )

        if RIGIDITY_RANK[left_rigidity] < RIGIDITY_RANK[right_rigidity]:
            return SegmentMovementDecision(
                move_segment_id=left.segment_id,
                reference_segment_id=right.segment_id,
                move_rigidity=left_rigidity,
                reference_rigidity=right_rigidity,
                reason="move_less_rigid_left",
            )

        if RIGIDITY_RANK[right_rigidity] < RIGIDITY_RANK[left_rigidity]:
            return SegmentMovementDecision(
                move_segment_id=right.segment_id,
                reference_segment_id=left.segment_id,
                move_rigidity=right_rigidity,
                reference_rigidity=left_rigidity,
                reason="move_less_rigid_right",
            )

        return SegmentMovementDecision(
            move_segment_id=right.segment_id,
            reference_segment_id=left.segment_id,
            move_rigidity=right_rigidity,
            reference_rigidity=left_rigidity,
            reason="equal_rigidity_prefer_trailing_segment",
        )


__all__ = [
    "RIGIDITY_BY_TYPE",
    "RIGIDITY_RANK",
    "RigidityLevel",
    "SegmentMovementDecision",
    "SegmentRigidityPolicy",
]
