from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from core.document import create_document
from core.types import CoordinateSystem
from services.ai_agent import AIReviewService
from services.engine_protocol import EngineResult, EngineStatus
from services.minimal_pipeline import MinimalPipelineResult
from services.vector_reconstruction_engine import VectorReconstructionArtifactBundle
from vector_reconstruction.cli import main


def _write_circle_image(image_path: Path) -> None:
    image = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(image, (48, 48), 24, 255, thickness=2)
    assert cv2.imwrite(str(image_path), image)


def _write_rgba_black_line_image(image_path: Path) -> None:
    image = np.full((96, 96, 4), 255, dtype=np.uint8)
    cv2.line(image, (16, 48), (80, 48), (0, 0, 0, 255), thickness=5)
    assert cv2.imwrite(str(image_path), image)


def _bundle(document_id: str = "cli_doc") -> VectorReconstructionArtifactBundle:
    document = create_document(
        document_id=document_id,
        width=96.0,
        height=96.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    pipeline_result = MinimalPipelineResult(
        document=document,
        json_payload="{}",
        extracted_contours=None,  # type: ignore[arg-type]
        source_image=np.zeros((8, 8, 3), dtype=np.uint8),
        debug_artifacts=None,
    )
    return VectorReconstructionArtifactBundle(
        engine_result=EngineResult(
            status=EngineStatus.COMPLETED,
            document=document,
            report={
                "score_before": 10.0,
                "score_after": 8.0,
                "decision_stats": {"auto_apply": 1, "auto_reject": 0, "requires_external_decision": 0},
                "integrity": {"success": True},
                "iteration_count": 1,
                "unresolved_targets": [],
            },
            metadata={"dry_run_only": False},
        ),
        pipeline_result=pipeline_result,
        document_json='{"document_id":"cli_doc"}',
        output_svg="<svg xmlns=\"http://www.w3.org/2000/svg\"/>",
        output_dxf="0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
        overlay_png=b"overlay",
        diff_png=b"diff",
        decision_report={"status": "completed"},
        metrics={"status": "completed", "dry_run_only": False, "export_mode": "all_debug"},
    )


class _FakeEngine:
    last_call: dict[str, object] | None = None
    last_init: dict[str, object] | None = None

    def __init__(self, *, ai_review_service=None, config=None):
        type(self).last_init = {
            "ai_review_service": ai_review_service,
            "config": config,
        }

    def run_artifact_bundle(self, image_path, **kwargs):
        type(self).last_call = {
            "image_path": str(image_path),
            **kwargs,
        }
        return _bundle()


def test_vector_reconstruction_cli_writes_artifact_bundle_and_passes_parameters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"not-an-image-but-exists")
    output_dir = tmp_path / "out"
    monkeypatch.setattr("vector_reconstruction.cli.VectorReconstructionEngine", _FakeEngine)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--autonomy",
            "autonomous_safe",
            "--max-iterations",
            "4",
            "--target-types",
            "circle,arc",
        ]
    )

    assert exit_code == 0
    assert _FakeEngine.last_call is not None
    assert _FakeEngine.last_init is not None
    assert _FakeEngine.last_call["image_path"] == str(input_path)
    assert _FakeEngine.last_call["max_iterations"] == 4
    assert _FakeEngine.last_call["target_types"] == ("circle", "arc")
    assert _FakeEngine.last_call["dry_run_only"] is False
    for name in (
        "document.json",
        "output.svg",
        "output.dxf",
        "overlay.png",
        "diff.png",
        "decision_report.json",
        "metrics.json",
    ):
        assert (output_dir / name).exists()


def test_vector_reconstruction_cli_returns_structured_error_for_missing_input(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "run",
            "--input",
            str(tmp_path / "missing.png"),
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip())
    assert payload["ok"] is False
    assert payload["error_type"] == "InputImageNotFound"
    assert "missing.png" in payload["message"]


def test_vector_reconstruction_cli_passes_dry_run_only_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"exists")
    output_dir = tmp_path / "out"
    monkeypatch.setattr("vector_reconstruction.cli.VectorReconstructionEngine", _FakeEngine)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--dry-run-only",
        ]
    )

    assert exit_code == 0
    assert _FakeEngine.last_call is not None
    assert _FakeEngine.last_call["dry_run_only"] is True


def test_vector_reconstruction_cli_passes_export_mode_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"exists")
    output_dir = tmp_path / "out"
    monkeypatch.setattr("vector_reconstruction.cli.VectorReconstructionEngine", _FakeEngine)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--export-mode",
            "centerline",
        ]
    )

    assert exit_code == 0
    assert _FakeEngine.last_call is not None
    assert _FakeEngine.last_call["export_mode"] == "centerline"


def test_vector_reconstruction_cli_does_not_enable_ai_review_without_flag_even_if_env_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"exists")
    output_dir = tmp_path / "out"
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("AI_PROVIDER_MODEL", "env-model")
    monkeypatch.setattr("vector_reconstruction.cli.VectorReconstructionEngine", _FakeEngine)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert _FakeEngine.last_init is not None
    assert _FakeEngine.last_init["ai_review_service"] is None
    config = _FakeEngine.last_init["config"]
    assert getattr(config, "enable_ai_review") is False
    assert getattr(config, "ai_status") == "disabled"


def test_vector_reconstruction_cli_module_run_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "circle.png"
    _write_circle_image(input_path)
    output_dir = tmp_path / "bundle"

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
            "circle",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["ok"] is True
    assert payload["status"] in {"completed", "completed_with_unresolved_regions", "requires_external_decision"}
    for name in (
        "document.json",
        "output.svg",
        "output.dxf",
        "overlay.png",
        "diff.png",
        "decision_report.json",
        "metrics.json",
    ):
        assert (output_dir / name).exists(), name


def test_vector_reconstruction_cli_file_provider_ai_review_uses_local_visual_context(tmp_path: Path) -> None:
    input_path = tmp_path / "circle.png"
    _write_circle_image(input_path)
    output_dir = tmp_path / "bundle_ai"
    response_path = tmp_path / "ai_response.json"
    response_path.write_text(
        json.dumps(
            {
                "summary": "Local visual review completed.",
                "issues": [],
                "proposed_commands": [],
            }
        ),
        encoding="utf-8",
    )
    env = dict(**os.environ)
    env["AI_PROVIDER"] = "file"
    env["AI_FILE_RESPONSE_PATH"] = str(response_path)

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
            "--max-iterations",
            "1",
            "--export-mode",
            "centerline",
            "--enable-ai-review",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    decision_report = json.loads((output_dir / "decision_report.json").read_text(encoding="utf-8"))
    assert metrics["enable_ai_review"] is True
    assert metrics["ai_provider"] == "file"
    assert metrics["ai_status"] == "enabled"
    assert metrics["ai_input_mode"] == "local_visual_context"
    assert metrics["ai_prompt_char_count"] > 0
    assert metrics["ai_review_job_count"] >= 1
    assert metrics["ai_review_image_count"] == 3
    assert metrics["ai_review_image_file_count"] == 3
    assert metrics["ai_review_panel_count"] >= 3
    assert metrics["ai_review_crop_max_size_px"] == 512
    assert "ai_review_summary" in decision_report


def test_vector_reconstruction_cli_emits_jsonl_progress_and_writes_ai_review_log(tmp_path: Path) -> None:
    input_path = tmp_path / "circle.png"
    _write_circle_image(input_path)
    output_dir = tmp_path / "bundle_ai_log"
    log_path = tmp_path / "ai_review_interaction.json"
    response_path = tmp_path / "ai_response.json"
    response_path.write_text(
        json.dumps(
            {
                "summary": "Local visual review completed.",
                "issues": [],
                "proposed_commands": [],
            }
        ),
        encoding="utf-8",
    )
    env = dict(**os.environ)
    env["AI_PROVIDER"] = "file"
    env["AI_FILE_RESPONSE_PATH"] = str(response_path)
    env["OPENAI_API_KEY"] = "super-secret-key"

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
            "--max-iterations",
            "1",
            "--export-mode",
            "centerline",
            "--enable-ai-review",
            "--ai-review-log-path",
            str(log_path),
            "--progress-format",
            "jsonl",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_payload = json.loads(completed.stdout.strip())
    assert stdout_payload["ok"] is True
    progress_events = [json.loads(line) for line in completed.stderr.splitlines() if line.strip()]
    stages = {event["stage"] for event in progress_events}
    for required in (
        "load_input_image_start",
        "load_input_image_done",
        "minimal_pipeline_start",
        "minimal_pipeline_done",
        "contour_extraction_start",
        "contour_extraction_done",
        "skeleton_processing_start",
        "skeleton_processing_done",
        "auto_refinement_start",
        "auto_refinement_done",
        "ai_review_context_build_start",
        "ai_review_context_build_done",
        "ai_provider_call_start",
        "ai_provider_call_done",
        "artifact_export_start",
        "artifact_export_done",
        "total_done",
    ):
        assert required in stages

    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "super-secret-key" not in log_text
    for forbidden in ("source_contours", "resampled_contours", "\"paths\"", "\"segments\"", "\"anchors\""):
        assert forbidden not in log_text
    interaction_log = json.loads(log_text)
    assert interaction_log["interaction_count"] == 1
    interaction = interaction_log["interactions"][0]
    assert interaction["provider"] == "file"
    assert interaction["prompt_char_count"] > 0
    assert interaction["review_input_summary"]["review_job_count"] >= 1
    assert interaction["normalized_response"]["summary"] == "Local visual review completed."


def test_vector_reconstruction_cli_file_provider_accepts_utf8_bom_response(tmp_path: Path) -> None:
    input_path = tmp_path / "circle.png"
    _write_circle_image(input_path)
    output_dir = tmp_path / "bundle_bom"
    response_path = tmp_path / "ai_response_bom.json"
    response_path.write_text(
        json.dumps(
            {
                "summary": "BOM response completed.",
                "issues": [],
                "proposed_commands": [],
            }
        ),
        encoding="utf-8-sig",
    )
    env = dict(**os.environ)
    env["AI_PROVIDER"] = "file"
    env["AI_FILE_RESPONSE_PATH"] = str(response_path)

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
            "--max-iterations",
            "1",
            "--export-mode",
            "centerline",
            "--enable-ai-review",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["ok"] is True
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["ai_provider"] == "file"


def test_vector_reconstruction_cli_quiet_suppresses_progress_logs(tmp_path: Path) -> None:
    input_path = tmp_path / "circle.png"
    _write_circle_image(input_path)
    output_dir = tmp_path / "bundle_quiet"

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
            "--quiet",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.strip() == ""


def test_vector_reconstruction_cli_centerline_rgba_output_has_no_non_finite_tokens(tmp_path: Path) -> None:
    input_path = tmp_path / "rgba_line.png"
    _write_rgba_black_line_image(input_path)
    output_dir = tmp_path / "bundle"

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
            "--autonomy",
            "autonomous_safe",
            "--max-iterations",
            "1",
            "--export-mode",
            "centerline",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    forbidden_tokens = ("NaN", "Infinity", "inf")
    for name in ("document.json", "metrics.json", "decision_report.json", "output.svg", "output.dxf"):
        payload = (output_dir / name).read_text(encoding="utf-8")
        assert all(token not in payload for token in forbidden_tokens), name


def test_vector_reconstruction_cli_enable_ai_review_requires_ai_provider(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"exists")
    output_dir = tmp_path / "out"
    monkeypatch.setattr("vector_reconstruction.cli._ENV_FILE_CANDIDATES", ())
    monkeypatch.delenv("AI_PROVIDER", raising=False)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--enable-ai-review",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["error_type"] == "AIProviderNotConfigured"
    assert "AI_PROVIDER" in payload["message"]


def test_vector_reconstruction_cli_enable_ai_review_surfaces_missing_provider_key(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"exists")
    output_dir = tmp_path / "out"
    monkeypatch.setattr("vector_reconstruction.cli._ENV_FILE_CANDIDATES", ())
    monkeypatch.setenv("AI_PROVIDER", "siliconflow")
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--enable-ai-review",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["error_type"] == "AIProviderConfigurationError"
    assert "SILICONFLOW_API_KEY" in payload["message"]


def test_vector_reconstruction_cli_builds_ai_review_service_from_env_for_recorded_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"exists")
    output_dir = tmp_path / "out"
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "provider_name": "openai",
                "model": "gpt-4.1-mini",
                "request_fingerprint": "unused-in-cli-test",
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "prompt_instructions_sha256": "abc",
                "response": {
                    "summary": "ok",
                    "issues": [],
                    "proposed_commands": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("vector_reconstruction.cli._ENV_FILE_CANDIDATES", ())
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("AI_PROVIDER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("AI_RECORDED_MODE", "replay")
    monkeypatch.setenv("AI_RECORDED_FIXTURE_PATH", str(fixture_path))
    monkeypatch.setattr("vector_reconstruction.cli.VectorReconstructionEngine", _FakeEngine)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--enable-ai-review",
        ]
    )

    assert exit_code == 0
    assert _FakeEngine.last_init is not None
    ai_review_service = _FakeEngine.last_init["ai_review_service"]
    config = _FakeEngine.last_init["config"]
    assert isinstance(ai_review_service, AIReviewService)
    assert getattr(config, "enable_ai_review") is True
    assert getattr(config, "ai_provider") == "openai"
    assert getattr(config, "ai_model") == "gpt-4.1-mini"
    assert getattr(config, "ai_status") == "recorded_replay"
    assert _FakeEngine.last_call is not None
    assert _FakeEngine.last_call["enable_ai_review"] is True


def test_vector_reconstruction_cli_prefers_system_env_over_dotenv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"exists")
    output_dir = tmp_path / "out"
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "provider_name": "openai",
                "model": "env-model",
                "request_fingerprint": "unused-in-cli-test",
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "prompt_instructions_sha256": "abc",
                "response": {
                    "summary": "ok",
                    "issues": [],
                    "proposed_commands": [],
                },
            }
        ),
        encoding="utf-8",
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "AI_PROVIDER=openai",
                "AI_PROVIDER_MODEL=dotenv-model",
                "AI_RECORDED_MODE=replay",
                f"AI_RECORDED_FIXTURE_PATH={fixture_path}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("vector_reconstruction.cli._ENV_FILE_CANDIDATES", (dotenv_path,))
    monkeypatch.setenv("AI_PROVIDER_MODEL", "env-model")
    monkeypatch.setattr("vector_reconstruction.cli.VectorReconstructionEngine", _FakeEngine)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--enable-ai-review",
        ]
    )

    assert exit_code == 0
    assert _FakeEngine.last_init is not None
    config = _FakeEngine.last_init["config"]
    assert getattr(config, "ai_model") == "env-model"
