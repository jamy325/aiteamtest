from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from core.document import create_document
from core.types import CoordinateSystem
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
