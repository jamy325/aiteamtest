from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_core_package_is_importable() -> None:
    module = import_module("core")
    assert module.__name__ == "core"


def test_required_project_files_exist() -> None:
    required_files = [
        ROOT / "AGENTS.md",
        ROOT / "pyproject.toml",
        ROOT / "docs" / "design" / "vector-reconstruction.md",
        ROOT / "docs" / "process" / "codex-workflow.md",
    ]
    assert all(path.is_file() for path in required_files)


def test_forbidden_secret_like_files_are_not_present() -> None:
    forbidden_paths = [
        ROOT / ".env",
    ]
    secret_patterns = ("*.pem", "*.key")

    assert all(not path.exists() for path in forbidden_paths)
    for pattern in secret_patterns:
        assert not any(ROOT.rglob(pattern))
