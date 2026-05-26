from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.common import ProviderConfigurationError

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


DEFAULT_RECORDED_FIXTURE_DIR = Path("tests/fixtures/ai_responses")
LIVE_RECORD_ENV_VARS = ("ENABLE_LIVE_AI_PROVIDER_RECORD", "ENABLE_LIVE_AI_PROVIDER_TESTS")


class MissingRecordedFixtureError(FileNotFoundError):
    pass


@dataclass(frozen=True, slots=True)
class RecordedFixtureMetadata:
    provider_name: str
    model: str
    request_fingerprint: str
    recorded_at: str
    prompt_instructions_sha256: str
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model": self.model,
            "request_fingerprint": self.request_fingerprint,
            "recorded_at": self.recorded_at,
            "prompt_instructions_sha256": self.prompt_instructions_sha256,
            "response": self.response,
        }


def build_recorded_request_fingerprint(prompt: str, review_input: AIReviewInput) -> str:
    payload = {
        "prompt_instructions_sha256": _sha256_text(_prompt_instructions(prompt)),
        "review_input": _canonical_review_input(review_input),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prompt_instructions(prompt: str) -> str:
    marker = "\n\nReview input:\n"
    if marker in prompt:
        return prompt.split(marker, 1)[0]
    return prompt


class RecordedVisionProvider(VisionReviewAdapter):
    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        mode: str,
        live_adapter: VisionReviewAdapter | None = None,
        fixtures_dir: Path | str = DEFAULT_RECORDED_FIXTURE_DIR,
        fixture_path: Path | str | None = None,
        allow_live: bool = False,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"record", "replay"}:
            raise ValueError("recorded provider mode must be `record` or `replay`")
        if normalized_mode == "record" and live_adapter is None:
            raise ValueError("record mode requires a live_adapter")
        self.provider_name = str(provider_name).strip().lower()
        self.model = str(model).strip() or "unknown-model"
        self.mode = normalized_mode
        self.live_adapter = live_adapter
        self.fixtures_dir = Path(fixtures_dir)
        self.fixture_path = Path(fixture_path) if fixture_path is not None else None
        self.allow_live = bool(allow_live)

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        request_fingerprint = build_recorded_request_fingerprint(prompt, review_input)
        prompt_hash = _sha256_text(_prompt_instructions(prompt))
        fixture_path = self._fixture_path_for(request_fingerprint)

        if self.mode == "replay":
            fixture = self._load_fixture(fixture_path, expected_fingerprint=request_fingerprint)
            return dict(fixture.response)

        if not _live_record_enabled(self.allow_live):
            raise ProviderConfigurationError(
                "Recorded provider live record mode is disabled. "
                "Pass allow_live=True or set ENABLE_LIVE_AI_PROVIDER_RECORD=1."
            )
        if self.live_adapter is None:
            raise ProviderConfigurationError("Recorded provider record mode requires a configured live adapter")

        response = dict(self.live_adapter.review(prompt, review_input))
        fixture = RecordedFixtureMetadata(
            provider_name=self.provider_name,
            model=self.model,
            request_fingerprint=request_fingerprint,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            prompt_instructions_sha256=prompt_hash,
            response=response,
        )
        self._write_fixture(fixture_path, fixture)
        return response

    def _fixture_path_for(self, request_fingerprint: str) -> Path:
        if self.fixture_path is not None:
            return self.fixture_path
        return (
            self.fixtures_dir
            / _sanitize_path_component(self.provider_name)
            / _sanitize_path_component(self.model)
            / f"{request_fingerprint}.json"
        )

    def _load_fixture(self, fixture_path: Path, *, expected_fingerprint: str) -> RecordedFixtureMetadata:
        if not fixture_path.exists():
            raise MissingRecordedFixtureError(
                "Missing recorded AI response fixture for "
                f"provider={self.provider_name} model={self.model} fingerprint={expected_fingerprint}. "
                "Generate it with record mode and explicit live enable."
            )
        try:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Recorded AI fixture is not valid JSON: {fixture_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Recorded AI fixture must contain a JSON object: {fixture_path}")
        response = payload.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"Recorded AI fixture response must be a JSON object: {fixture_path}")
        recorded_provider = str(payload.get("provider_name", "")).strip().lower()
        if recorded_provider != self.provider_name:
            raise ValueError(
                "Recorded AI fixture provider mismatch: "
                f"expected {self.provider_name}, found {recorded_provider}"
            )
        recorded_model = str(payload.get("model", ""))
        if recorded_model != self.model:
            raise ValueError(
                "Recorded AI fixture model mismatch: "
                f"expected {self.model}, found {recorded_model}"
            )
        recorded_fingerprint = payload.get("request_fingerprint")
        if recorded_fingerprint != expected_fingerprint:
            raise ValueError(
                "Recorded AI fixture fingerprint mismatch: "
                f"expected {expected_fingerprint}, found {recorded_fingerprint}"
            )
        return RecordedFixtureMetadata(
            provider_name=recorded_provider,
            model=recorded_model,
            request_fingerprint=str(recorded_fingerprint),
            recorded_at=str(payload.get("recorded_at", "")),
            prompt_instructions_sha256=str(payload.get("prompt_instructions_sha256", "")),
            response=dict(response),
        )

    def _write_fixture(self, fixture_path: Path, fixture: RecordedFixtureMetadata) -> None:
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(fixture.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")


def _live_record_enabled(allow_live: bool) -> bool:
    if allow_live:
        return True
    return any(_env_truthy(name) for name in LIVE_RECORD_ENV_VARS)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_review_input(review_input: AIReviewInput) -> dict[str, Any]:
    payload = review_input.to_payload()
    for field_name in ("original_image", "overlay_image", "distance_field_diff_image"):
        payload[field_name] = _canonical_image_value(payload.get(field_name))
    return payload


def _canonical_image_value(raw_value: Any) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    image_path = Path(str(raw_value))
    if not image_path.exists() or not image_path.is_file():
        return {"name": image_path.name, "missing": True}
    image_bytes = image_path.read_bytes()
    return {
        "name": image_path.name,
        "size": len(image_bytes),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }


def _sanitize_path_component(value: str) -> str:
    return value.replace("\\", "__").replace("/", "__").replace(":", "_").replace(" ", "_")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_RECORDED_FIXTURE_DIR",
    "LIVE_RECORD_ENV_VARS",
    "MissingRecordedFixtureError",
    "RecordedFixtureMetadata",
    "RecordedVisionProvider",
    "build_recorded_request_fingerprint",
]
