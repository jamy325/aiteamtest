import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from services.ai_agent import AIReviewInput, AIReviewService, build_review_prompt
from services.ai_provider_factory import create_vision_adapter
from services.ai_recorded_provider import (
    MissingRecordedFixtureError,
    RecordedVisionProvider,
    build_recorded_request_fingerprint,
)


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\xd9\x8f\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _StubLiveAdapter:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = dict(response)
        self.calls = 0

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        self.calls += 1
        return dict(self.response)


def _review_input_with_images(tmp_path: Path) -> AIReviewInput:
    original = tmp_path / "original.png"
    overlay = tmp_path / "overlay.png"
    diff = tmp_path / "diff.png"
    for path in (original, overlay, diff):
        path.write_bytes(PNG_BYTES)
    return AIReviewInput(
        original_image=str(original),
        overlay_image=str(overlay),
        distance_field_diff_image=str(diff),
        vector_document_json={"document_id": "recorded_doc"},
        fit_error=0.1,
        complexity_score=0.2,
        topology_status="closed",
        self_intersection_count=0,
        coordinate_system={"unit": "px"},
        proposed_commands_from_algorithm=(
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "Algorithm suggests a circle.",
                "confidence": 0.87,
                "requires_user_confirmation": True,
            },
        ),
    )


def _valid_response() -> dict[str, object]:
    return {
        "summary": "Recorded provider review succeeded.",
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "Replay agrees with the circle intent.",
                "confidence": 0.9,
                "requires_user_confirmation": True,
                "semantic_source": "recorded_fixture",
                "semantic_confidence": 0.93,
                "topology_hint": None,
                "self_intersection_hint": None,
                "alpha_hint": None,
                "color_hint": None,
            }
        ],
    }


def _invalid_schema_response() -> dict[str, object]:
    return {
        "summary": "Invalid because it leaks geometry parameters.",
        "issues": [],
        "proposed_commands": [
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "Leaked geometry parameters.",
                "confidence": 0.9,
                "requires_user_confirmation": True,
                "cx": 12.0,
            }
        ],
    }


def _write_recorded_fixture(
    fixture_path: Path,
    *,
    provider_name: str,
    model: str,
    request_fingerprint: str,
    response: dict[str, object],
) -> None:
    fixture_path.write_text(
        json.dumps(
            {
                "provider_name": provider_name,
                "model": model,
                "request_fingerprint": request_fingerprint,
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "prompt_instructions_sha256": "abc",
                "response": response,
            }
        ),
        encoding="utf-8",
    )


def test_recorded_provider_record_writes_fixture_and_replay_hits(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    prompt = "Test prompt\n\nReview input:\n{}"
    live_adapter = _StubLiveAdapter(_valid_response())
    fixtures_dir = tmp_path / "fixtures"

    recorder = RecordedVisionProvider(
        provider_name="openai",
        model="gpt-4.1-mini",
        mode="record",
        live_adapter=live_adapter,
        fixtures_dir=fixtures_dir,
        allow_live=True,
    )
    recorded_response = recorder.review(prompt, review_input)

    assert recorded_response["summary"] == "Recorded provider review succeeded."
    assert live_adapter.calls == 1

    fingerprint = build_recorded_request_fingerprint(prompt, review_input)
    fixture_path = fixtures_dir / "openai" / "gpt-4.1-mini" / f"{fingerprint}.json"
    fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture_payload["provider_name"] == "openai"
    assert fixture_payload["model"] == "gpt-4.1-mini"
    assert fixture_payload["request_fingerprint"] == fingerprint
    assert fixture_payload["response"]["summary"] == "Recorded provider review succeeded."

    replay_live_adapter = _StubLiveAdapter({"summary": "should not run", "issues": [], "proposed_commands": []})
    replayer = RecordedVisionProvider(
        provider_name="openai",
        model="gpt-4.1-mini",
        mode="replay",
        live_adapter=replay_live_adapter,
        fixtures_dir=fixtures_dir,
    )
    replayed_response = replayer.review(prompt, review_input)

    assert replayed_response["summary"] == "Recorded provider review succeeded."
    assert replay_live_adapter.calls == 0


def test_recorded_provider_replay_missing_fixture_raises_guidance(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    provider = RecordedVisionProvider(
        provider_name="gemini",
        model="gemini-3.5-flash",
        mode="replay",
        fixtures_dir=tmp_path / "missing-fixtures",
    )

    with pytest.raises(MissingRecordedFixtureError, match="Generate it with record mode"):
        provider.review("prompt", review_input)


def test_recorded_provider_detects_fingerprint_mismatch(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    prompt = "prompt"
    expected = build_recorded_request_fingerprint(prompt, review_input)
    fixture_path = tmp_path / "mismatch.json"
    fixture_path.write_text(
        json.dumps(
            {
                "provider_name": "openai",
                "model": "gpt-4.1-mini",
                "request_fingerprint": "wrong-fingerprint",
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "prompt_instructions_sha256": "abc",
                "response": _valid_response(),
            }
        ),
        encoding="utf-8",
    )
    provider = RecordedVisionProvider(
        provider_name="openai",
        model="gpt-4.1-mini",
        mode="replay",
        fixture_path=fixture_path,
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        provider.review(prompt, review_input)

    assert expected != "wrong-fingerprint"


def test_recorded_provider_replay_uses_existing_schema_validation(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    prompt = build_review_prompt(review_input)
    fingerprint = build_recorded_request_fingerprint(prompt, review_input)
    fixture_path = tmp_path / "invalid.json"
    _write_recorded_fixture(
        fixture_path,
        provider_name="siliconflow",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        request_fingerprint=fingerprint,
        response=_invalid_schema_response(),
    )
    adapter = RecordedVisionProvider(
        provider_name="siliconflow",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        mode="replay",
        fixture_path=fixture_path,
    )

    with pytest.raises(ValidationError):
        AIReviewService(adapter=adapter).run_review(review_input)


def test_ai_provider_factory_supports_recorded_replay_without_live_api_key(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    prompt = build_review_prompt(review_input)
    fingerprint = build_recorded_request_fingerprint(prompt, review_input)
    fixture_path = tmp_path / "openai-recorded.json"
    _write_recorded_fixture(
        fixture_path,
        provider_name="openai",
        model="gpt-4.1-mini",
        request_fingerprint=fingerprint,
        response=_valid_response(),
    )

    adapter = create_vision_adapter(
        "openai",
        recorded_mode="replay",
        model="gpt-4.1-mini",
        fixture_path=fixture_path,
    )
    output = AIReviewService(adapter=adapter).run_review(review_input)

    assert output.summary == "Recorded provider review succeeded."


def test_recorded_provider_detects_provider_mismatch_with_explicit_fixture_path(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    prompt = build_review_prompt(review_input)
    fingerprint = build_recorded_request_fingerprint(prompt, review_input)
    fixture_path = tmp_path / "provider-mismatch.json"
    _write_recorded_fixture(
        fixture_path,
        provider_name="openai",
        model="gpt-4.1-mini",
        request_fingerprint=fingerprint,
        response=_valid_response(),
    )
    provider = RecordedVisionProvider(
        provider_name="gemini",
        model="gpt-4.1-mini",
        mode="replay",
        fixture_path=fixture_path,
    )

    with pytest.raises(ValueError, match="provider mismatch"):
        provider.review(prompt, review_input)


def test_recorded_provider_detects_model_mismatch_with_explicit_fixture_path(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    prompt = build_review_prompt(review_input)
    fingerprint = build_recorded_request_fingerprint(prompt, review_input)
    fixture_path = tmp_path / "model-mismatch.json"
    _write_recorded_fixture(
        fixture_path,
        provider_name="openai",
        model="gpt-4.1-mini",
        request_fingerprint=fingerprint,
        response=_valid_response(),
    )
    provider = RecordedVisionProvider(
        provider_name="openai",
        model="gpt-4.1",
        mode="replay",
        fixture_path=fixture_path,
    )

    with pytest.raises(ValueError, match="model mismatch"):
        provider.review(prompt, review_input)


def test_ai_provider_factory_record_mode_requires_explicit_live_enable(tmp_path: Path) -> None:
    live_adapter = _StubLiveAdapter(_valid_response())
    adapter = create_vision_adapter(
        "openai",
        recorded_mode="record",
        model="gpt-4.1-mini",
        live_adapter=live_adapter,
        fixtures_dir=tmp_path / "fixtures",
    )

    with pytest.raises(Exception, match="live record mode is disabled"):
        adapter.review("prompt", _review_input_with_images(tmp_path))


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "gpt-4.1-mini"),
        ("gemini", "gemini-3.5-flash"),
        ("siliconflow", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ],
)
def test_ai_provider_factory_supports_unified_record_mode_with_stub_live_adapters(
    tmp_path: Path,
    provider: str,
    model: str,
) -> None:
    fixtures_dir = tmp_path / "fixtures"
    live_adapter = _StubLiveAdapter(_valid_response())
    adapter = create_vision_adapter(
        provider,
        recorded_mode="record",
        model=model,
        live_adapter=live_adapter,
        fixtures_dir=fixtures_dir,
        allow_live=True,
    )
    review_input = _review_input_with_images(tmp_path)
    output = AIReviewService(adapter=adapter).run_review(review_input)

    assert output.summary == "Recorded provider review succeeded."
    assert live_adapter.calls == 1
