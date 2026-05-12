from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from core.precision import PrecisionUtility
from core.types import Anchor, VectorDocument

AnchorRelation = Literal["same_path", "cross_path_same_object", "cross_object", "cross_path_unknown_object"]


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
                pair = tuple(sorted((anchor.anchor_id, neighbor.anchor_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                candidates.append(self._candidate(anchor, neighbor, path_to_object))

        candidates.sort(key=lambda candidate: (candidate.distance, candidate.anchor_ids))
        return tuple(candidates)

    def _candidate(
        self,
        left: Anchor,
        right: Anchor,
        path_to_object: dict[str, str | None],
    ) -> SnappingCandidate:
        relation = self._relation(left, right, path_to_object)
        locked_involved = left.locked or right.locked
        movable_anchor_ids = tuple(anchor.anchor_id for anchor in (left, right) if not anchor.locked)

        return SnappingCandidate(
            anchor_ids=(left.anchor_id, right.anchor_id),
            path_ids=(left.path_id, right.path_id),
            object_ids=(path_to_object.get(left.path_id), path_to_object.get(right.path_id)),
            relation=relation,
            distance=PrecisionUtility.distance_between_points(left.position, right.position),
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


__all__ = [
    "AnchorRelation",
    "AnchorSpatialIndex",
    "BruteForceAnchorIndex",
    "GlobalSnappingEngine",
    "SnappingCandidate",
    "SnappingConfig",
]
