from __future__ import annotations

from difflib import unified_diff
from pathlib import Path


def normalize_snapshot_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def assert_text_snapshot(*, actual: str, snapshot_path: str | Path) -> None:
    path = Path(snapshot_path)
    expected = normalize_snapshot_text(path.read_text(encoding="utf-8"))
    normalized_actual = normalize_snapshot_text(actual)
    if normalized_actual == expected:
        return

    diff = "".join(
        unified_diff(
            expected.splitlines(keepends=True),
            normalized_actual.splitlines(keepends=True),
            fromfile=str(path),
            tofile="actual",
        )
    )
    raise AssertionError(f"snapshot mismatch for {path}\n{diff}")


__all__ = ["assert_text_snapshot", "normalize_snapshot_text"]
