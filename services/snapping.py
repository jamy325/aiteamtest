from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from core.document import add_constraint
from core.precision import PrecisionUtility
from core.types import Anchor, Constraint, VectorDocument

AnchorRelation = Literal["same_path", "cross_path_same_object", "cross_object", "cross_path_unknown_object"]
CoincidentConstraintMode = Literal["soft", "hard"]


@dataclass(frozen=True, slots=True)
class SnappingConfig:
    epsilon: float = PrecisionUtility.EPSILON


@dataclass(frozen=True, slots=True)
class SnappingCandidate:
    anchor_ids: tuple[str, str]
    path_ids: tuple[str, str]
    object_ids: tuple[str | None, str | None]
    relation: AnchorRelation
    distance: float
    locked_involved: bool
    movable_anchor_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoincidentConstraintConfig:
    soft_confidence_threshold: float = 0.5
    hard_confidence_threshold: float = 0.9
    soft_strength: float = 0.5
    hard_strength: float = 1.0
    source: str = "global_snapping"


class AnchorSpatialIndex(Protocol):
    def query_radius(self, anchor: Anchor, radius: float) -> tuple[Anchor, ...]:
        ...


class BruteForceAnchorIndex:
    def __init__(self, anchors: tuple[Anchor, ...] | list[Anchor]) -> None:
        self.anchors = tuple(anchors)

    def query_radius(self, anchor: Anchor, radius: float) -> tuple[Anchor, ...]:
        return tuple(
            candidate
            for candidate in self.anchors
            if candidate.anchor_id != anchor.anchor_id
            and PrecisionUtility.distance_between_points(anchor.position, candidate.position) <= radius
        )


class GlobalSnappingEngine:
    def __init__(
        self,
        config: SnappingConfig | None = None,
        index_builder: Callable[[tuple[Anchor, ...]], AnchorSpatialIndex] | None = None,
    ) -> None:
        self.config = config or SnappingConfig()
        self.index_builder = index_builder or BruteForceAnchorIndex

    def find_candidates(self, document: VectorDocument) -> tuple[SnappingCandidate, ...]:
        anchors = tuple(document.anchors)
        if len(anchors) < 2:
            return ()

        index = self.index_builder(anchors)
        path_to_object = {path.path_id: path.object_id for path in document.paths}
        candidates: list[SnappingCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()

        for anchor in anchors:
            for neighbor in index.query_radius(anchor, self.config.epsilon):
                distance = PrecisionUtility.distance_between_points(anchor.position, neighbor.position)
                if distance > self.config.epsilon:
                    continue
                pair = tuple(sorted((anchor.anchor_id, neighbor.anchor_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                candidates.append(self._candidate(anchor, neighbor, path_to_object, distance))

        candidates.sort(key=lambda candidate: (candidate.distance, candidate.anchor_ids))
        return tuple(candidates)

    def _candidate(
        self,
        left: Anchor,
        right: Anchor,
        path_to_object: dict[str, str | None],
        distance: float,
    ) -> SnappingCandidate:
        relation = self._relation(left, right, path_to_object)
        locked_involved = left.locked or right.locked
        movable_anchor_ids = tuple(anchor.anchor_id for anchor in (left, right) if not anchor.locked)

        return SnappingCandidate(
            anchor_ids=(left.anchor_id, right.anchor_id),
            path_ids=(left.path_id, right.path_id),
            object_ids=(path_to_object.get(left.path_id), path_to_object.get(right.path_id)),
            relation=relation,
            distance=distance,
            locked_involved=locked_involved,
            movable_anchor_ids=movable_anchor_ids,
        )

    def _relation(
        self,
        left: Anchor,
        right: Anchor,
        path_to_object: dict[str, str | None],
    ) -> AnchorRelation:
        if left.path_id == right.path_id:
            return "same_path"

        left_object_id = path_to_object.get(left.path_id)
        right_object_id = path_to_object.get(right.path_id)
        if left_object_id is None or right_object_id is None:
            return "cross_path_unknown_object"
        if left_object_id == right_object_id:
            return "cross_path_same_object"
        return "cross_object"


class CoincidentConstraintEngine:
    def __init__(
        self,
        snapping_engine: GlobalSnappingEngine | None = None,
        config: CoincidentConstraintConfig | None = None,
    ) -> None:
        self.snapping_engine = snapping_engine or GlobalSnappingEngine()
        self.config = config or CoincidentConstraintConfig()

    def generate_constraints(self, document: VectorDocument) -> tuple[Constraint, ...]:
        constraints: list[Constraint] = []
        seen_target_pairs = {
            tuple(sorted(constraint.targets))
            for constraint in document.constraints
            if constraint.type == "coincident"
        }

        for candidate in self.snapping_engine.find_candidates(document):
            constraint = self._constraint_from_candidate(candidate)
            if constraint is None:
                continue
            target_pair = tuple(sorted(constraint.targets))
            if target_pair in seen_target_pairs:
                continue
            seen_target_pairs.add(target_pair)
            constraints.append(constraint)

        return tuple(constraints)

    def apply_constraints(self, document: VectorDocument) -> VectorDocument:
        updated_document = document
        for constraint in self.generate_constraints(document):
            if any(
                existing.constraint_id == constraint.constraint_id
                or (existing.type == "coincident" and tuple(sorted(existing.targets)) == tuple(sorted(constraint.targets)))
                for existing in updated_document.constraints
            ):
                continue
            updated_document = add_constraint(updated_document, constraint)
        return updated_document

    def _constraint_from_candidate(self, candidate: SnappingCandidate) -> Constraint | None:
        confidence = self._confidence_for_candidate(candidate)
        if confidence < self.config.soft_confidence_threshold:
            return None

        mode: CoincidentConstraintMode = "soft"
        strength = self.config.soft_strength
        if not candidate.locked_involved and confidence >= self.config.hard_confidence_threshold:
            mode = "hard"
            strength = self.config.hard_strength

        anchor_ids = tuple(sorted(candidate.anchor_ids))
        return Constraint(
            constraint_id=f"constraint_coincident_{anchor_ids[0]}_{anchor_ids[1]}",
            type="coincident",
            targets=anchor_ids,
            strength=strength,
            source=self.config.source,
            confidence=confidence,
            metadata={
                "mode": mode,
                "relation": candidate.relation,
                "distance": candidate.distance,
                "locked_involved": candidate.locked_involved,
                # Preserve the candidate's non-forcible movement information for downstream topology work.
                "movable_anchor_ids": list(candidate.movable_anchor_ids),
            },
        )

    def _confidence_for_candidate(self, candidate: SnappingCandidate) -> float:
        epsilon = self.snapping_engine.config.epsilon
        if PrecisionUtility.near_zero(epsilon):
            return 1.0 if PrecisionUtility.near_zero(candidate.distance) else 0.0
        return max(0.0, min(1.0, 1.0 - (candidate.distance / epsilon)))


__all__ = [
    "AnchorRelation",
    "AnchorSpatialIndex",
    "BruteForceAnchorIndex",
    "CoincidentConstraintConfig",
    "CoincidentConstraintEngine",
    "CoincidentConstraintMode",
    "GlobalSnappingEngine",
    "SnappingCandidate",
    "SnappingConfig",
]
