from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from core.types import Constraint

ConstraintType = Literal[
    "horizontal",
    "vertical",
    "concentric",
    "tangent",
    "coincident",
    "shared_tangent",
    "g1_continuity",
]

SUPPORTED_CONSTRAINT_TYPES: tuple[ConstraintType, ...] = (
    "horizontal",
    "vertical",
    "concentric",
    "tangent",
    "coincident",
    "shared_tangent",
    "g1_continuity",
)


@dataclass(frozen=True, slots=True)
class ConstraintGraph:
    constraints: tuple[Constraint, ...] = ()
    _by_constraint_id: dict[str, Constraint] = field(init=False, repr=False)
    _by_target_id: dict[str, tuple[Constraint, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        constraints = tuple(self.constraints)
        by_constraint_id: dict[str, Constraint] = {}
        by_target_id: dict[str, list[Constraint]] = {}

        for constraint in constraints:
            _validate_constraint(constraint)
            if constraint.constraint_id in by_constraint_id:
                raise ValueError(f"duplicate constraint_id: {constraint.constraint_id}")
            by_constraint_id[constraint.constraint_id] = constraint
            for target_id in constraint.targets:
                by_target_id.setdefault(target_id, []).append(constraint)

        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "_by_constraint_id", by_constraint_id)
        object.__setattr__(
            self,
            "_by_target_id",
            {target_id: tuple(items) for target_id, items in by_target_id.items()},
        )

    def add_constraint(self, constraint: Constraint) -> ConstraintGraph:
        if constraint.constraint_id in self._by_constraint_id:
            raise ValueError(f"duplicate constraint_id: {constraint.constraint_id}")
        _validate_constraint(constraint)
        return ConstraintGraph(self.constraints + (constraint,))

    def remove_constraint(self, constraint_id: str) -> ConstraintGraph:
        if constraint_id not in self._by_constraint_id:
            raise KeyError(f"unknown constraint_id: {constraint_id}")
        return ConstraintGraph(
            tuple(constraint for constraint in self.constraints if constraint.constraint_id != constraint_id)
        )

    def get_constraint(self, constraint_id: str) -> Constraint | None:
        return self._by_constraint_id.get(constraint_id)

    def constraints_for_target(self, target_id: str) -> tuple[Constraint, ...]:
        return self._by_target_id.get(target_id, ())

    def has_constraint(self, constraint_id: str) -> bool:
        return constraint_id in self._by_constraint_id

    def list_constraints(self) -> tuple[Constraint, ...]:
        return self.constraints


def _validate_constraint_type(constraint_type: str) -> None:
    if constraint_type not in SUPPORTED_CONSTRAINT_TYPES:
        raise ValueError(f"unsupported constraint type: {constraint_type}")


def _validate_constraint(constraint: Constraint) -> None:
    _validate_constraint_type(constraint.type)
    if len(set(constraint.targets)) != len(constraint.targets):
        raise ValueError(f"duplicate target ids in constraint: {constraint.constraint_id}")


def supports_constraint_type(constraint_type: str) -> bool:
    return constraint_type in SUPPORTED_CONSTRAINT_TYPES


def build_constraint_graph(constraints: Iterable[Constraint]) -> ConstraintGraph:
    return ConstraintGraph(tuple(constraints))


__all__ = [
    "ConstraintGraph",
    "ConstraintType",
    "SUPPORTED_CONSTRAINT_TYPES",
    "build_constraint_graph",
    "supports_constraint_type",
]
