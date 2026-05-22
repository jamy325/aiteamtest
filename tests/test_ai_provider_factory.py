import ast
import json
import os
from pathlib import Path

import pytest
from jsonschema import ValidationError

from services.ai_adapters import (
    FileResponseVisionAdapter,
    GeminiVisionAdapter,
    MockVisionAdapter,
    OpenAIVisionAdapter,
    ProviderConfigurationError,
    SiliconFlowVisionAdapter,
    create_vision_adapter,
)
from services.ai_agent import AIReviewInput, AIReviewService


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\xd9\x8f\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _review_input_with_images(tmp_path: Path) -> AIReviewInput:
    image_paths = []
    for name in ("original.png", "overlay.png", "diff.png"):
        image_path = tmp_path / name
        image_path.write_bytes(PNG_BYTES)
        image_paths.append(image_path)
    return AIReviewInput(
        original_image=str(image_paths[0]),
        overlay_image=str(image_paths[1]),
        distance_field_diff_image=str(image_paths[2]),
        vector_document_json={"document_id": "provider_doc"},
        candidates=(
            {"candidate_id": "cand_1", "shape_type": "circle", "path_id": "path_1", "confidence": 0.91},
        ),
        proposed_commands_from_algorithm=(
            {
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "Algorithm already suggests a circle.",
                "confidence": 0.88,
                "requires_user_confirmation": True,
                "candidate_id": "cand_1",
            },
        ),
        preview_summary={"accepted_count": 1, "rejected_count": 0},
        fit_error=0.1,
        complexity_score=0.2,
        topology_status="closed",
        self_intersection_count=0,
        coordinate_system={"unit": "px"},
    )


def _valid_response_text() -> str:
    return json.dumps(
        {
            "summary": "Provider review succeeded.",
            "issues": [],
            "proposed_commands": [
                {
                    "tool": "propose_replace_path_with_circle",
                    "path_id": "path_1",
                    "reason": "The provider agrees with the algorithm candidate.",
                    "confidence": 0.9,
                    "requires_user_confirmation": True,
                    "candidate_id": "cand_1",
                    "semantic_source": "provider",
                    "semantic_confidence": 0.94,
                    "topology_hint": None,
                    "self_intersection_hint": None,
                    "alpha_hint": None,
                    "color_hint": None,
                }
            ],
        }
    )


class _OpenAIResponsesStub:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        return type("OpenAIResponse", (), {"output_text": _valid_response_text()})()


class _OpenAIClientStub:
    def __init__(self) -> None:
        self.responses = _OpenAIResponsesStub()


class _GeminiModelsStub:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def generate_content(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        return type("GeminiResponse", (), {"text": _valid_response_text()})()


class _GeminiClientStub:
    def __init__(self) -> None:
        self.models = _GeminiModelsStub()


class _SiliconFlowCompletionsStub:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        message = type("SiliconFlowMessage", (), {"content": _valid_response_text()})()
        choice = type("SiliconFlowChoice", (), {"message": message})()
        return type("SiliconFlowResponse", (), {"choices": [choice]})()


class _SiliconFlowClientStub:
    def __init__(self) -> None:
        self.chat = type("SiliconFlowChat", (), {"completions": _SiliconFlowCompletionsStub()})()


def test_create_vision_adapter_supports_mock_and_file(tmp_path: Path) -> None:
    response = json.loads(_valid_response_text())
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")

    assert isinstance(create_vision_adapter("mock", response=response), MockVisionAdapter)
    assert isinstance(create_vision_adapter("file", response_path=response_path), FileResponseVisionAdapter)


def test_create_vision_adapter_supports_openai_and_gemini_stubs(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)

    openai_client = _OpenAIClientStub()
    openai_adapter = create_vision_adapter("openai", client=openai_client)
    openai_output = AIReviewService(adapter=openai_adapter).run_review(review_input)
    assert isinstance(openai_adapter, OpenAIVisionAdapter)
    assert openai_output.summary == "Provider review succeeded."
    assert openai_client.responses.last_kwargs is not None
    openai_input = openai_client.responses.last_kwargs["input"][0]["content"]  # type: ignore[index]
    assert any(item["type"] == "input_image" for item in openai_input)  # type: ignore[index]

    gemini_client = _GeminiClientStub()
    gemini_adapter = create_vision_adapter(
        "gemini",
        client=gemini_client,
        image_loader=lambda path: {"loaded_path": str(path)},
    )
    gemini_output = AIReviewService(adapter=gemini_adapter).run_review(review_input)
    assert isinstance(gemini_adapter, GeminiVisionAdapter)
    assert gemini_output.summary == "Provider review succeeded."
    assert gemini_client.models.last_kwargs is not None
    gemini_contents = gemini_client.models.last_kwargs["contents"]  # type: ignore[index]
    assert any(isinstance(item, dict) and "loaded_path" in item for item in gemini_contents)  # type: ignore[arg-type]


def test_create_vision_adapter_supports_siliconflow_stub(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)
    siliconflow_client = _SiliconFlowClientStub()
    adapter = create_vision_adapter(
        "siliconflow",
        client=siliconflow_client,
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        base_url="https://api.siliconflow.cn/v1",
    )

    review_output = AIReviewService(adapter=adapter).run_review(review_input)

    assert isinstance(adapter, SiliconFlowVisionAdapter)
    assert adapter.base_url == "https://api.siliconflow.cn/v1"
    assert review_output.summary == "Provider review succeeded."
    last_kwargs = siliconflow_client.chat.completions.last_kwargs
    assert last_kwargs is not None
    content = last_kwargs["messages"][0]["content"]  # type: ignore[index]
    assert any(item["type"] == "image_url" for item in content)  # type: ignore[index]
    assert content[0]["type"] == "text"  # type: ignore[index]


def test_openai_provider_requires_api_key_when_client_is_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = create_vision_adapter("openai")

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        adapter.review("prompt", AIReviewInput(None, None, None, {}, 0.0, 0.0, "open", 0, {}))


def test_gemini_provider_requires_api_key_when_client_is_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    adapter = create_vision_adapter("gemini")

    with pytest.raises(ProviderConfigurationError, match="GEMINI_API_KEY"):
        adapter.review("prompt", AIReviewInput(None, None, None, {}, 0.0, 0.0, "open", 0, {}))


def test_siliconflow_provider_requires_api_key_when_client_is_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    adapter = create_vision_adapter("siliconflow")

    with pytest.raises(ProviderConfigurationError, match="SILICONFLOW_API_KEY"):
        adapter.review("prompt", AIReviewInput(None, None, None, {}, 0.0, 0.0, "open", 0, {}))


def test_openai_provider_missing_sdk_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_import(name: str) -> object:
        if name == "openai":
            raise ImportError("missing openai")
        return __import__(name)

    monkeypatch.setattr("services.ai_adapters.openai_provider.importlib.import_module", fake_import)
    adapter = create_vision_adapter("openai")

    with pytest.raises(ProviderConfigurationError, match="openai"):
        adapter.review("prompt", AIReviewInput(None, None, None, {}, 0.0, 0.0, "open", 0, {}))


def test_gemini_provider_missing_sdk_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_import(name: str) -> object:
        if name == "google.genai":
            raise ImportError("missing google.genai")
        return __import__(name)

    monkeypatch.setattr("services.ai_adapters.gemini_provider.importlib.import_module", fake_import)
    adapter = create_vision_adapter("gemini")

    with pytest.raises(ProviderConfigurationError, match="google-genai"):
        adapter.review("prompt", AIReviewInput(None, None, None, {}, 0.0, 0.0, "open", 0, {}))


def test_siliconflow_provider_missing_sdk_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    def fake_import(name: str) -> object:
        if name == "openai":
            raise ImportError("missing openai")
        return __import__(name)

    monkeypatch.setattr("services.ai_adapters.siliconflow_provider.importlib.import_module", fake_import)
    adapter = create_vision_adapter("siliconflow")

    with pytest.raises(ProviderConfigurationError, match="openai"):
        adapter.review("prompt", AIReviewInput(None, None, None, {}, 0.0, 0.0, "open", 0, {}))


def test_provider_factory_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        create_vision_adapter("unknown")


def test_openai_provider_invalid_json_is_rejected_by_review_service(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)

    class _BadResponses:
        def create(self, **kwargs: object) -> object:
            return type("BadResponse", (), {"output_text": "{\"summary\": 1}"})()

    bad_client = type("BadOpenAIClient", (), {"responses": _BadResponses()})()
    adapter = create_vision_adapter("openai", client=bad_client)

    with pytest.raises(ValidationError):
        AIReviewService(adapter=adapter).run_review(review_input)


def test_siliconflow_provider_invalid_precise_geometry_is_rejected_by_review_service(tmp_path: Path) -> None:
    review_input = _review_input_with_images(tmp_path)

    class _BadSiliconFlowCompletions:
        def create(self, **kwargs: object) -> object:
            payload = json.dumps(
                {
                    "summary": "This leaks exact geometry.",
                    "issues": [],
                    "proposed_commands": [
                        {
                            "tool": "propose_replace_segment_with_arc",
                            "path_id": "path_1",
                            "segment_range": [0, 1],
                            "reason": "bad",
                            "confidence": 0.8,
                            "requires_user_confirmation": True,
                            "cx": 12.0,
                        }
                    ],
                }
            )
            message = type("SiliconFlowMessage", (), {"content": payload})()
            choice = type("SiliconFlowChoice", (), {"message": message})()
            return type("SiliconFlowResponse", (), {"choices": [choice]})()

    bad_client = type(
        "BadSiliconFlowClient",
        (),
        {"chat": type("BadSiliconFlowChat", (), {"completions": _BadSiliconFlowCompletions()})()},
    )()
    adapter = create_vision_adapter("siliconflow", client=bad_client)

    with pytest.raises(ValidationError):
        AIReviewService(adapter=adapter).run_review(review_input)


@pytest.mark.skipif(
    os.environ.get("ENABLE_LIVE_AI_PROVIDER_TESTS") != "1",
    reason="live provider integration test is disabled by default",
)
def test_live_provider_integration_when_explicitly_enabled(tmp_path: Path) -> None:
    provider = os.environ.get("AI_PROVIDER", "openai")
    model = os.environ.get("AI_PROVIDER_MODEL")
    review_input = _review_input_with_images(tmp_path)
    adapter = create_vision_adapter(provider, model=model) if model else create_vision_adapter(provider)
    review_output = AIReviewService(adapter=adapter).run_review(review_input)

    assert review_output.summary
    assert isinstance(review_output.raw_response, dict)


def test_provider_files_have_no_forbidden_network_shortcuts() -> None:
    source_paths = (
        Path("services/ai_adapters/common.py"),
        Path("services/ai_adapters/provider_factory.py"),
        Path("services/ai_adapters/openai_provider.py"),
        Path("services/ai_adapters/gemini_provider.py"),
        Path("services/ai_adapters/siliconflow_provider.py"),
    )
    forbidden_imports = {"requests", "urllib3", "httpx"}

    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(name.name.split(".")[0] for name in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

        assert imports.isdisjoint(forbidden_imports)
