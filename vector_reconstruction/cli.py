from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from core.types import ShapeCandidateTargetType
from services.engine_protocol import AutonomyLevel, EngineStatus
from services.vector_reconstruction_engine import VectorReconstructionEngine


_TARGET_TYPE_CHOICES: tuple[ShapeCandidateTargetType, ...] = ("circle", "rectangle", "ellipse", "arc", "line")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run autonomous vector reconstruction from an input image.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run vector reconstruction and write an artifact bundle.")
    run_parser.add_argument("--input", required=True, help="Input image path.")
    run_parser.add_argument("--output", required=True, help="Output artifact directory.")
    run_parser.add_argument(
        "--autonomy",
        default=AutonomyLevel.AUTONOMOUS_SAFE.value,
        choices=[item.value for item in AutonomyLevel],
        help="Autonomy level for DecisionPolicy.",
    )
    run_parser.add_argument("--max-iterations", type=int, default=None, help="Max AI/policy refinement iterations.")
    run_parser.add_argument(
        "--target-types",
        nargs="*",
        default=None,
        help="Optional target types, e.g. --target-types circle arc or --target-types circle,arc",
    )
    run_parser.add_argument("--dry-run-only", action="store_true", help="Only preview decisions; do not commit final modifications.")
    run_parser.add_argument("--enable-ai-review", action="store_true", help="Enable the AI review loop if configured.")
    run_parser.add_argument("--document-id", default=None, help="Optional document_id override.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run_command(args)
    raise SystemExit(f"unsupported command: {args.command}")


def _run_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.is_file():
        return _emit_error(
            "InputImageNotFound",
            f"input image not found: {input_path}",
            {
                "input": str(input_path),
                "output": str(output_dir),
            },
        )

    try:
        target_types = _parse_target_types(args.target_types)
        engine = VectorReconstructionEngine()
        bundle = engine.run_artifact_bundle(
            input_path,
            document_id=args.document_id,
            target_types=target_types,
            autonomy_level=AutonomyLevel(args.autonomy),
            max_iterations=args.max_iterations,
            dry_run_only=args.dry_run_only,
            enable_ai_review=args.enable_ai_review,
        )
        _write_bundle(output_dir, bundle)
        if bundle.engine_result.status == EngineStatus.FAILED:
            return _emit_error(
                "EngineFailed",
                "vector reconstruction engine returned failed status",
                {
                    "status": bundle.engine_result.status.value,
                    "output": str(output_dir),
                },
            )
        print(json.dumps({"ok": True, "status": bundle.engine_result.status.value, "output": str(output_dir)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        return _emit_error(
            type(exc).__name__,
            str(exc),
            {
                "input": str(input_path),
                "output": str(output_dir),
            },
        )


def _parse_target_types(raw_values: Sequence[str] | None) -> tuple[ShapeCandidateTargetType, ...]:
    if not raw_values:
        return ()
    parsed: list[ShapeCandidateTargetType] = []
    for raw_value in raw_values:
        for part in str(raw_value).split(","):
            value = part.strip().lower()
            if not value:
                continue
            if value not in _TARGET_TYPE_CHOICES:
                choices = ", ".join(_TARGET_TYPE_CHOICES)
                raise ValueError(f"unsupported target type: {value}. expected one of: {choices}")
            parsed.append(value)  # type: ignore[arg-type]
    return tuple(parsed)


def _write_bundle(output_dir: Path, bundle) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "document.json").write_text(bundle.document_json, encoding="utf-8")
    (output_dir / "output.svg").write_text(bundle.output_svg, encoding="utf-8")
    (output_dir / "output.dxf").write_text(bundle.output_dxf, encoding="utf-8")
    (output_dir / "overlay.png").write_bytes(bundle.overlay_png)
    (output_dir / "diff.png").write_bytes(bundle.diff_png)
    (output_dir / "decision_report.json").write_text(
        json.dumps(bundle.decision_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(bundle.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _emit_error(error_type: str, message: str, details: dict[str, object] | None = None) -> int:
    payload = {
        "ok": False,
        "error_type": error_type,
        "message": message,
        "details": details or {},
    }
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


__all__ = ["build_parser", "main"]
