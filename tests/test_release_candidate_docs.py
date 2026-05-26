from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_candidate_docs_reference_existing_samples_and_configs() -> None:
    readme = _read_text(ROOT / "README.md")
    release_doc = _read_text(ROOT / "docs" / "release_candidate.md")
    expected_artifacts = _read_text(ROOT / "samples" / "expected_artifacts.md")

    assert "samples/inputs/circle_quickstart.png" in readme
    assert "samples/inputs/ellipse_quickstart.png" in readme
    assert "python -m vector_reconstruction run" in readme
    assert "autonomous_safe" in readme
    assert "recorded_mode" in readme

    assert "Known Limitations" in release_doc
    assert "Troubleshooting" in release_doc
    assert "manual_only" in release_doc
    assert "assisted" in release_doc
    assert "autonomous_safe" in release_doc
    assert "autonomous_full" in release_doc
    assert "python scripts/run_acceptance_benchmark.py" in release_doc
    assert "python scripts/run_real_world_regression.py" in release_doc

    assert "circle_quickstart.png" in expected_artifacts
    assert "ellipse_quickstart.png" in expected_artifacts
    assert "document.json" in expected_artifacts
    assert "output.svg" in expected_artifacts

    assert (ROOT / "samples" / "inputs" / "circle_quickstart.png").is_file()
    assert (ROOT / "samples" / "inputs" / "ellipse_quickstart.png").is_file()


def test_release_candidate_sample_configs_are_valid_json() -> None:
    quality_profile = json.loads(_read_text(ROOT / "configs" / "release_candidate_quality_profile.sample.json"))
    ai_provider = json.loads(_read_text(ROOT / "configs" / "release_candidate_ai_provider.sample.json"))
    benchmark = json.loads(_read_text(ROOT / "configs" / "release_candidate_benchmark.sample.json"))

    assert quality_profile["profile_name"] == "release_candidate_circle"
    assert quality_profile["recommended_thresholds"]["min_algorithm_confidence"] == 0.55

    assert ai_provider["provider"] == "openai"
    assert ai_provider["recorded_mode"] == "replay"
    assert ai_provider["allow_live"] is False

    assert benchmark["acceptance"]["manifest"] == "benchmarks/acceptance_manifest.json"
    assert benchmark["real_world_regression"]["baseline"] == "benchmarks/baselines/real_world_regression_baseline.json"


def test_release_candidate_quickstart_cli_smoke_runs_offline_for_bundled_samples(tmp_path: Path) -> None:
    sample_specs = (
        ("circle_quickstart.png", "circle"),
        ("ellipse_quickstart.png", "ellipse"),
    )

    for sample_name, target_type in sample_specs:
        input_path = ROOT / "samples" / "inputs" / sample_name
        output_dir = tmp_path / sample_name.replace(".png", "")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "vector_reconstruction",
                "run",
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
                "--dry-run-only",
                "--max-iterations",
                "1",
                "--target-types",
                target_type,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout.strip())
        assert payload["ok"] is True
        assert payload["status"] in {
            "completed",
            "completed_with_unresolved_regions",
            "requires_external_decision",
        }
        for name in (
            "document.json",
            "output.svg",
            "output.dxf",
            "overlay.png",
            "diff.png",
            "decision_report.json",
            "metrics.json",
        ):
            assert (output_dir / name).exists(), f"{sample_name}:{name}"
