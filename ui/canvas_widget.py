from __future__ import annotations


class CanvasWidget:
    def __init__(self, *, locked_ids: tuple[str, ...] = ()) -> None:
        self._locked_ids = tuple(str(item) for item in locked_ids)

    @property
    def locked_ids(self) -> tuple[str, ...]:
        return self._locked_ids

    def set_locked_ids(self, locked_ids: tuple[str, ...] | list[str]) -> None:
        self._locked_ids = tuple(str(item) for item in locked_ids)

    def lock_id(self, item_id: str) -> None:
        if item_id in self._locked_ids:
            return
        self._locked_ids = self._locked_ids + (str(item_id),)

    def unlock_id(self, item_id: str) -> None:
        self._locked_ids = tuple(existing for existing in self._locked_ids if existing != item_id)
