from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from core.types import ShapeCandidateTargetType
from services.ai_adapters import ProviderConfigurationError, create_vision_adapter
from services.ai_agent import AIReviewService
from services.engine_protocol import AutonomyLevel, EngineStatus
from services.vector_reconstruction_engine import VectorReconstructionEngine, VectorReconstructionEngineConfig


_TARGET_TYPE_CHOICES: tuple[ShapeCandidateTargetType, ...] = ("circle", "rectangle", "ellipse", "arc", "line")
_EXPORT_MODE_CHOICES: tuple[str, ...] = ("outline", "centerline", "all_debug")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE_CANDIDATES: tuple[Path, ...] = (
    _REPO_ROOT / ".env",
    _REPO_ROOT / ".github" / ".env",
)


class AIProviderNotConfigured(RuntimeError):
    pass


class AIProviderConfigurationError(RuntimeError):
    pass


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
    run_parser.add_argument(
        "--export-mode",
        default="all_debug",
        choices=list(_EXPORT_MODE_CHOICES),
        help="Exporter path selection mode for SVG/DXF output.",
    )
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
        ai_review_service, engine_config = _build_engine_runtime(enable_ai_review=bool(args.enable_ai_review))
        engine = VectorReconstructionEngine(
            ai_review_service=ai_review_service,
            config=engine_config,
        )
        bundle = engine.run_artifact_bundle(
            input_path,
            document_id=args.document_id,
            target_types=target_types,
            autonomy_level=AutonomyLevel(args.autonomy),
            max_iterations=args.max_iterations,
            dry_run_only=args.dry_run_only,
            enable_ai_review=args.enable_ai_review,
            export_mode=str(args.export_mode),
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


def _build_engine_runtime(*, enable_ai_review: bool) -> tuple[AIReviewService | None, VectorReconstructionEngineConfig]:
    if not enable_ai_review:
        return (
            None,
            VectorReconstructionEngineConfig(
                enable_ai_review=False,
                ai_status="disabled",
            ),
        )

    runtime_env = _merged_ai_environment()
    provider = str(runtime_env.get("AI_PROVIDER", "")).strip().lower()
    if not provider:
        raise AIProviderNotConfigured("AI_PROVIDER is required when --enable-ai-review is set")

    model = str(runtime_env.get("AI_PROVIDER_MODEL", "")).strip() or None
    adapter_kwargs: dict[str, object] = {}
    if model:
        adapter_kwargs["model"] = model

    openai_key = str(runtime_env.get("OPENAI_API_KEY", "")).strip()
    gemini_key = str(runtime_env.get("GEMINI_API_KEY", "")).strip()
    google_key = str(runtime_env.get("GOOGLE_API_KEY", "")).strip()
    siliconflow_key = str(runtime_env.get("SILICONFLOW_API_KEY", "")).strip()

    if provider == "openai" and openai_key:
        adapter_kwargs["api_key"] = openai_key
    elif provider == "gemini":
        resolved_gemini_key = gemini_key or google_key
        if resolved_gemini_key:
            adapter_kwargs["api_key"] = resolved_gemini_key
    elif provider == "siliconflow" and siliconflow_key:
        adapter_kwargs["api_key"] = siliconflow_key
    elif provider == "file":
        response_path = str(runtime_env.get("AI_FILE_RESPONSE_PATH", "")).strip()
        if not response_path:
            raise AIProviderConfigurationError("file provider requires AI_FILE_RESPONSE_PATH")
        adapter_kwargs["response_path"] = _resolve_config_path(response_path)

    recorded_mode = str(runtime_env.get("AI_RECORDED_MODE", "")).strip().lower()
    ai_status = "enabled"
    if recorded_mode:
        adapter_kwargs["recorded_mode"] = recorded_mode
        fixture_path = str(runtime_env.get("AI_RECORDED_FIXTURE_PATH", "")).strip()
        fixtures_dir = str(runtime_env.get("AI_RECORDED_FIXTURES_DIR", "")).strip()
        if fixture_path:
            adapter_kwargs["fixture_path"] = _resolve_config_path(fixture_path)
        if fixtures_dir:
            adapter_kwargs["fixtures_dir"] = _resolve_config_path(fixtures_dir)
        adapter_kwargs["allow_live"] = _env_truthy(runtime_env.get("ENABLE_LIVE_AI_PROVIDER_RECORD")) or _env_truthy(
            runtime_env.get("ENABLE_LIVE_AI_PROVIDER_TESTS")
        )
        ai_status = f"recorded_{recorded_mode}"

    _validate_provider_configuration(
        provider=provider,
        recorded_mode=recorded_mode,
        openai_key=openai_key,
        gemini_key=gemini_key,
        google_key=google_key,
        siliconflow_key=siliconflow_key,
    )

    try:
        adapter = create_vision_adapter(provider, **adapter_kwargs)
    except ProviderConfigurationError as exc:
        raise AIProviderConfigurationError(str(exc)) from exc
    except ValueError as exc:
        raise AIProviderConfigurationError(str(exc)) from exc

    return (
        AIReviewService(adapter=adapter),
        VectorReconstructionEngineConfig(
            enable_ai_review=True,
            ai_provider=provider,
            ai_model=str(model or getattr(adapter, "model", "") or ""),
            ai_status=ai_status,
        ),
    )


def _merged_ai_environment() -> dict[str, str]:
    merged: dict[str, str] = {}
    for env_path in _ENV_FILE_CANDIDATES:
        if env_path.exists() and env_path.is_file():
            merged.update(_parse_env_file(env_path))
    for key, value in os.environ.items():
        merged[key] = value
    return merged


def _parse_env_file(env_path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        parsed[normalized_key] = _strip_env_value(value.strip())
    return parsed


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _env_truthy(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_provider_configuration(
    *,
    provider: str,
    recorded_mode: str,
    openai_key: str,
    gemini_key: str,
    google_key: str,
    siliconflow_key: str,
) -> None:
    if provider == "file" or recorded_mode == "replay":
        return
    if provider == "openai" and not openai_key:
        raise AIProviderConfigurationError("OpenAI provider requires OPENAI_API_KEY")
    if provider == "gemini" and not (gemini_key or google_key):
        raise AIProviderConfigurationError("Gemini provider requires GEMINI_API_KEY or GOOGLE_API_KEY")
    if provider == "siliconflow" and not siliconflow_key:
        raise AIProviderConfigurationError("SiliconFlow provider requires SILICONFLOW_API_KEY")


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
