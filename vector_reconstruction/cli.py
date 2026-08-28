from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

from core.types import ShapeCandidateTargetType
from services.ai_adapters import ProviderConfigurationError, create_vision_adapter
from services.ai_agent import AIReviewService
from services.free_pen_runtime import (
    FileSequenceFreePenAdapter,
    FreePenImageTransportConfig,
    FreePenRuntime,
    NativeToolCallSequenceAdapter,
    FreePenToolRuntime,
    load_free_pen_schema,
)
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


class AIReviewLogWriteError(RuntimeError):
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
    run_parser.add_argument("--quiet", action="store_true", help="Disable progress logs on stderr.")
    run_parser.add_argument(
        "--progress-format",
        default="text",
        choices=("text", "jsonl"),
        help="Progress log format written to stderr.",
    )
    run_parser.add_argument(
        "--ai-review-log-path",
        default=None,
        help="Optional JSON path for sanitized AI review interaction logs.",
    )
    run_parser.add_argument(
        "--ai-review-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for a single AI review provider call.",
    )

    free_pen_parser = subparsers.add_parser("free-pen", help="Run the experimental FreePenRuntime and write final_overlay.png.")
    free_pen_parser.add_argument("--input", required=True, help="Input source image path.")
    free_pen_parser.add_argument("--output", required=True, help="Output directory for final_overlay.png and round responses.")
    free_pen_parser.add_argument("--max-rounds", type=int, default=1, help="Maximum FreePen AI rounds.")
    free_pen_parser.add_argument("--stroke-width", type=int, default=3, help="Rendered stroke width in pixels.")
    free_pen_parser.add_argument(
        "--ai-review-log-path",
        default=None,
        help="Optional JSON path for FreePen AI request/response logs.",
    )
    free_pen_parser.add_argument(
        "--ai-review-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for a single FreePen provider call.",
    )

    free_pen_tool_parser = subparsers.add_parser(
        "free-pen-tool",
        help="Run the experimental FreePenRuntime V1 tool loop and write overlay/composite/path artifacts.",
    )
    free_pen_tool_parser.add_argument("--input", required=True, help="Input source image path.")
    free_pen_tool_parser.add_argument("--output", required=True, help="Output directory for FreePen tool artifacts.")
    free_pen_tool_parser.add_argument("--max-steps", type=int, default=16, help="Maximum drawing tool-call steps.")
    free_pen_tool_parser.add_argument("--stroke-width", type=int, default=2, help="Rendered stroke width in pixels.")
    free_pen_tool_parser.add_argument(
        "--ai-review-log-path",
        default=None,
        help="Optional JSON path for FreePen tool AI request/response logs.",
    )
    free_pen_tool_parser.add_argument(
        "--ai-review-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for a single FreePen tool provider call.",
    )
    free_pen_tool_parser.add_argument(
        "--image-transport",
        default=None,
        choices=("url", "base64"),
        help="Optional image transport override for FreePen tool requests.",
    )
    free_pen_tool_parser.add_argument(
        "--public-image-base-url",
        default=None,
        help="Optional public image base URL for URL-based FreePen tool image transport.",
    )
    free_pen_tool_parser.add_argument(
        "--conversation-max-turns",
        type=int,
        default=None,
        help="Optional maximum number of assistant turns to retain in FreePen tool conversation history.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run_command(args)
    if args.command == "free-pen":
        return _free_pen_command(args)
    if args.command == "free-pen-tool":
        return _free_pen_tool_command(args)
    raise SystemExit(f"unsupported command: {args.command}")


def _run_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    progress_reporter = _ProgressReporter(quiet=bool(args.quiet), format=str(args.progress_format))
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
        total_start = perf_counter()
        target_types = _parse_target_types(args.target_types)
        ai_review_service, engine_config = _build_engine_runtime(
            enable_ai_review=bool(args.enable_ai_review),
            ai_review_timeout_seconds=args.ai_review_timeout_seconds,
        )
        interaction_logger = _AIReviewInteractionLogWriter(Path(args.ai_review_log_path)) if args.ai_review_log_path else None
        if ai_review_service is not None:
            ai_review_service.set_progress_callback(progress_reporter.emit_event)
            ai_review_service.set_interaction_logger(None if interaction_logger is None else interaction_logger.record)
            ai_review_service.set_runtime_metadata(
                provider=engine_config.ai_provider,
                model=engine_config.ai_model,
                status=engine_config.ai_status,
            )
        engine = VectorReconstructionEngine(
            ai_review_service=ai_review_service,
            config=engine_config,
        )
        if hasattr(engine, "set_progress_callback"):
            engine.set_progress_callback(progress_reporter.emit_event)
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
        progress_reporter.emit_event(
            {
                "stage": "total_done",
                "message": "CLI run completed.",
                "duration_ms": (perf_counter() - total_start) * 1000.0,
                "status": bundle.engine_result.status.value,
                "output": str(output_dir),
            }
        )
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


def _free_pen_command(args: argparse.Namespace) -> int:
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
        runtime_env = _merged_ai_environment()
        adapter = _build_free_pen_adapter(ai_review_timeout_seconds=args.ai_review_timeout_seconds)
        interaction_logger = _AIReviewInteractionLogWriter(Path(args.ai_review_log_path)) if args.ai_review_log_path else None
        runtime = FreePenRuntime(
            adapter=adapter,
            max_rounds=max(1, int(args.max_rounds)),
            stroke_width=max(1, int(args.stroke_width)),
            interaction_logger=None if interaction_logger is None else interaction_logger.record,
            raw_response_logger=_print_free_pen_raw_response,
            provider_name=str(getattr(adapter, "provider_name", "") or runtime_env.get("AI_PROVIDER", "")).strip().lower(),
            provider_model=str(getattr(adapter, "model", "") or runtime_env.get("AI_PROVIDER_MODEL", "")).strip(),
        )
        result = runtime.run(input_path, output_dir)
        print(
            json.dumps(
                {
                    "ok": result.error_message is None,
                    "status": result.status,
                    "output": str(output_dir),
                    "final_overlay": str(result.final_overlay_path),
                    "rounds_executed": result.rounds_executed,
                    "final_decision": result.final_decision,
                    "error_message": result.error_message,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.error_message is None else 1
    except Exception as exc:
        return _emit_error(
            type(exc).__name__,
            str(exc),
            {
                "input": str(input_path),
                "output": str(output_dir),
            },
        )


def _free_pen_tool_command(args: argparse.Namespace) -> int:
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
        runtime_env = _merged_ai_environment()
        adapter = _build_free_pen_tool_adapter(ai_review_timeout_seconds=args.ai_review_timeout_seconds)
        interaction_logger = _AIReviewInteractionLogWriter(Path(args.ai_review_log_path)) if args.ai_review_log_path else None
        public_image_base_url = str(
            args.public_image_base_url or runtime_env.get("AI_PUBLIC_IMAGE_BASE_URL", "")
        ).strip() or None
        raw_image_transport = str(args.image_transport or runtime_env.get("AI_IMAGE_TRANSPORT", "")).strip().lower()
        image_transport = raw_image_transport or ("url" if public_image_base_url else "base64")
        conversation_max_turns_raw = args.conversation_max_turns
        if conversation_max_turns_raw is None:
            conversation_max_turns_raw = runtime_env.get("AI_CONVERSATION_MAX_TURNS", "30")
        conversation_max_turns = max(1, int(conversation_max_turns_raw))
        runtime = FreePenToolRuntime(
            adapter=adapter,
            max_steps=max(1, int(args.max_steps)),
            stroke_width=max(1, int(args.stroke_width)),
            interaction_logger=None if interaction_logger is None else interaction_logger.record,
            raw_response_logger=_print_free_pen_raw_response,
            provider_name=str(getattr(adapter, "provider_name", "") or runtime_env.get("AI_PROVIDER", "")).strip().lower(),
            provider_model=str(getattr(adapter, "model", "") or runtime_env.get("AI_PROVIDER_MODEL", "")).strip(),
            image_transport_config=FreePenImageTransportConfig(
                mode=image_transport,
                public_image_base_url=public_image_base_url,
                public_image_root=_REPO_ROOT,
                conversation_max_turns=conversation_max_turns,
            ),
        )
        result = runtime.run(input_path, output_dir)
        print(
            json.dumps(
                {
                    "ok": result.error_message is None,
                    "status": result.status,
                    "output": str(output_dir),
                    "final_overlay": str(result.final_overlay_path),
                    "final_composite": str(result.final_composite_path),
                    "free_pen_paths": str(result.paths_json_path),
                    "tool_trace": str(result.tool_trace_path),
                    "rounds_executed": result.rounds_executed,
                    "successful_step_count": result.successful_step_count,
                    "invalid_step_count": result.invalid_step_count,
                    "rejected_step_count": result.rejected_step_count,
                    "rollback_count": result.rollback_count,
                    "final_decision": result.final_decision,
                    "error_message": result.error_message,
                    "error_type": result.error_type,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.error_message is None else 1
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


def _build_engine_runtime(
    *,
    enable_ai_review: bool,
    ai_review_timeout_seconds: float | None = None,
) -> tuple[AIReviewService | None, VectorReconstructionEngineConfig]:
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
    if ai_review_timeout_seconds is not None:
        adapter_kwargs["timeout_seconds"] = float(ai_review_timeout_seconds)

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
        AIReviewService(adapter=adapter, timeout_seconds=ai_review_timeout_seconds),
        VectorReconstructionEngineConfig(
            enable_ai_review=True,
            ai_provider=provider,
            ai_model=str(model or getattr(adapter, "model", "") or ""),
            ai_status=ai_status,
            ai_review_timeout_seconds=None if ai_review_timeout_seconds is None else float(ai_review_timeout_seconds),
        ),
    )


def _build_free_pen_adapter(*, ai_review_timeout_seconds: float | None = None):
    runtime_env = _merged_ai_environment()
    provider = str(runtime_env.get("AI_PROVIDER", "")).strip().lower()
    if not provider:
        raise AIProviderNotConfigured("AI_PROVIDER is required for `vector_reconstruction free-pen`")

    model = str(runtime_env.get("AI_PROVIDER_MODEL", "")).strip() or None
    adapter_kwargs: dict[str, object] = {
        "response_schema": load_free_pen_schema(),
    }
    if model:
        adapter_kwargs["model"] = model
    if ai_review_timeout_seconds is not None:
        adapter_kwargs["timeout_seconds"] = float(ai_review_timeout_seconds)

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

    _validate_provider_configuration(
        provider=provider,
        recorded_mode="",
        openai_key=openai_key,
        gemini_key=gemini_key,
        google_key=google_key,
        siliconflow_key=siliconflow_key,
    )

    try:
        return create_vision_adapter(provider, **adapter_kwargs)
    except ProviderConfigurationError as exc:
        raise AIProviderConfigurationError(str(exc)) from exc
    except ValueError as exc:
        raise AIProviderConfigurationError(str(exc)) from exc


def _build_free_pen_tool_adapter(*, ai_review_timeout_seconds: float | None = None):
    runtime_env = _merged_ai_environment()
    provider = str(runtime_env.get("AI_PROVIDER", "")).strip().lower()
    if not provider:
        raise AIProviderNotConfigured("AI_PROVIDER is required for `vector_reconstruction free-pen-tool`")

    model = str(runtime_env.get("AI_PROVIDER_MODEL", "")).strip() or None
    adapter_kwargs: dict[str, object] = {}
    if model:
        adapter_kwargs["model"] = model
    if ai_review_timeout_seconds is not None:
        adapter_kwargs["timeout_seconds"] = float(ai_review_timeout_seconds)

    openai_key = str(runtime_env.get("OPENAI_API_KEY", "")).strip()
    gemini_key = str(runtime_env.get("GEMINI_API_KEY", "")).strip()
    google_key = str(runtime_env.get("GOOGLE_API_KEY", "")).strip()
    siliconflow_key = str(runtime_env.get("SILICONFLOW_API_KEY", "")).strip()

    if provider == "file":
        response_path = str(runtime_env.get("AI_FILE_RESPONSE_PATH", "")).strip()
        if not response_path:
            raise AIProviderConfigurationError("file provider requires AI_FILE_RESPONSE_PATH")
        return NativeToolCallSequenceAdapter(response_path=_resolve_config_path(response_path))

    if provider == "openai" and openai_key:
        adapter_kwargs["api_key"] = openai_key
    elif provider in {"gemini", "gemini_openai", "gemini-openai"}:
        resolved_gemini_key = gemini_key or google_key
        if resolved_gemini_key:
            adapter_kwargs["api_key"] = resolved_gemini_key
    elif provider == "siliconflow" and siliconflow_key:
        adapter_kwargs["api_key"] = siliconflow_key

    _validate_provider_configuration(
        provider=provider,
        recorded_mode="",
        openai_key=openai_key,
        gemini_key=gemini_key,
        google_key=google_key,
        siliconflow_key=siliconflow_key,
    )

    try:
        return create_vision_adapter(provider, **adapter_kwargs)
    except ProviderConfigurationError as exc:
        raise AIProviderConfigurationError(str(exc)) from exc
    except ValueError as exc:
        raise AIProviderConfigurationError(str(exc)) from exc


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
    if provider in {"gemini", "gemini_openai", "gemini-openai"}and not (gemini_key or google_key):
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


def _print_free_pen_raw_response(round_index: int, raw_response: object, provider_duration_ms: float | None) -> None:
    try:
        rendered = json.dumps(raw_response, ensure_ascii=False)
    except TypeError:
        rendered = repr(raw_response)
    duration_suffix = (
        f"[provider_duration_ms={round(float(provider_duration_ms), 3)}]"
        if provider_duration_ms is not None
        else ""
    )
    print(f"[free_pen_raw_response][round={round_index}]{duration_suffix} {rendered}", file=sys.stderr)


class _ProgressReporter:
    def __init__(self, *, quiet: bool, format: str) -> None:
        self.quiet = quiet
        self.format = format
        self.started_at = perf_counter()

    def emit_event(self, event: dict[str, object]) -> None:
        if self.quiet:
            return
        payload = dict(event)
        payload.setdefault("elapsed_ms", round((perf_counter() - self.started_at) * 1000.0, 3))
        if self.format == "jsonl":
            print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
            return
        stage = str(payload.get("stage", "unknown_stage"))
        message = str(payload.get("message", ""))
        details = ", ".join(
            f"{key}={value}"
            for key, value in payload.items()
            if key not in {"stage", "message"} and value not in (None, "", (), [], {})
        )
        suffix = f" ({details})" if details else ""
        print(f"[{stage}] {message}{suffix}", file=sys.stderr)


class _AIReviewInteractionLogWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, object]] = []
        self.record_index_by_id: dict[str, int] = {}

    def record(self, payload: dict[str, object]) -> None:
        sanitized = _sanitize_for_ai_review_log(payload)
        interaction_id = sanitized.get("interaction_id")
        if isinstance(interaction_id, str) and interaction_id in self.record_index_by_id:
            index = self.record_index_by_id[interaction_id]
            merged = dict(self.records[index])
            merged.update(sanitized)
            self.records[index] = merged
        else:
            if isinstance(interaction_id, str):
                self.record_index_by_id[interaction_id] = len(self.records)
            self.records.append(sanitized)
        document = {
            "interaction_count": len(self.records),
            "interactions": self.records,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            raise AIReviewLogWriteError(f"failed to write AI review interaction log: {self.path}") from exc


def _sanitize_for_ai_review_log(value: object) -> object:
    forbidden_keys = {"source_contours", "resampled_contours", "paths", "segments", "anchors", "document_json"}
    secret_markers = ("api_key", "authorization", "secret", "token")
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = str(key)
            lowered = normalized_key.lower()
            if normalized_key in forbidden_keys:
                continue
            if any(marker in lowered for marker in secret_markers):
                sanitized[normalized_key] = "[redacted]"
                continue
            sanitized[normalized_key] = _sanitize_for_ai_review_log(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_ai_review_log(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_ai_review_log(item) for item in value]
    return value


__all__ = ["build_parser", "main"]
