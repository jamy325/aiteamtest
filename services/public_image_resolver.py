from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
from urllib.parse import quote


_SENSITIVE_NAMES = {".env", ".git", ".github", ".roo"}
_SENSITIVE_SUFFIXES = (".pem", ".key")


@dataclass(frozen=True)
class PublicImageResolver:
    project_root: Path
    base_url: str
    allowed_dirs: tuple[str, ...] = ("out", "samples")

    def to_public_url(self, local_path: Path) -> str:
        root = self.project_root.resolve()
        target = Path(local_path).resolve()
        try:
            relative_path = target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path is outside project root: {local_path}") from exc
        self._validate_relative_path(relative_path)
        normalized_base_url = self.base_url.rstrip("/")
        normalized_path = "/".join(quote(part) for part in relative_path.parts)
        return f"{normalized_base_url}/{normalized_path}"

    def _validate_relative_path(self, relative_path: PurePath) -> None:
        if not relative_path.parts:
            raise ValueError("path must not resolve to project root")
        top_level = relative_path.parts[0]
        if top_level not in self.allowed_dirs:
            raise ValueError(f"path is outside allowed public directories: {relative_path}")
        for part in relative_path.parts:
            lowered = part.lower()
            if part in {"..", "."}:
                raise ValueError(f"path must not contain traversal segments: {relative_path}")
            if lowered in _SENSITIVE_NAMES:
                raise ValueError(f"sensitive path is not allowed for public URL exposure: {relative_path}")
            if lowered.endswith(_SENSITIVE_SUFFIXES):
                raise ValueError(f"sensitive file type is not allowed for public URL exposure: {relative_path}")


__all__ = ["PublicImageResolver"]
