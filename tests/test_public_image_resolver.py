from __future__ import annotations

from pathlib import Path

import pytest

from services.public_image_resolver import PublicImageResolver


def test_public_image_resolver_windows_path_to_url() -> None:
    resolver = PublicImageResolver(
        project_root=Path("D:/works/curve-fitting-ai-agent"),
        base_url="https://img.jinyao.qzz.io/",
    )
    local_path = Path("D:/works/curve-fitting-ai-agent/out/free_pen_real07/final_composite.png")

    public_url = resolver.to_public_url(local_path)

    assert public_url == "https://img.jinyao.qzz.io/out/free_pen_real07/final_composite.png"


@pytest.mark.parametrize(
    ("relative_path",),
    [
        (".env",),
        (".git/config",),
        ("samples/private.key",),
        ("out/../../secret.pem",),
    ],
)
def test_public_image_resolver_blocks_sensitive_paths(relative_path: str) -> None:
    resolver = PublicImageResolver(
        project_root=Path("D:/works/curve-fitting-ai-agent"),
        base_url="https://img.jinyao.qzz.io/",
    )

    with pytest.raises(ValueError):
        resolver.to_public_url(Path("D:/works/curve-fitting-ai-agent") / relative_path)
