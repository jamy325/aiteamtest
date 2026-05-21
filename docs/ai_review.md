# AI Review

`AIReviewService` keeps the AI review layer provider-neutral.

## Adapter model

- `VisionReviewAdapter` defines the minimal `review(prompt, review_input)` protocol.
- `MockVisionAdapter` returns a fixed JSON payload for deterministic tests.
- `FileResponseVisionAdapter` reads a JSON response from disk for offline review playback.
- `create_vision_adapter(...)` is the provider factory for `mock`, `file`, `openai`, and `gemini`.
- `OpenAIVisionAdapter` and `GeminiVisionAdapter` are optional runtime adapters with lazy SDK loading.
- Legacy `responder(prompt, review_input)` callables still work through `ResponderVisionAdapter`.

## Input contract

`AIReviewInput` carries review artifacts and algorithm context:

- `original_image`
- `overlay_image`
- `distance_field_diff_image`
- `vector_document_json`
- `candidates`
- `proposed_commands_from_algorithm`
- `preview_summary`

The review model is expected to inspect algorithm candidates and existing intent proposals, then return only validated intent-level commands.

## Output contract

- Every adapter response is validated through `validate_ai_review_response`.
- Responses must stay at semantic intent level.
- Exact centers, radii, rotations, control points, and other precise geometry parameters are forbidden.
- The adapter layer must not mutate `VectorDocument` and must not execute commands.

## Provider configuration

- OpenAI provider:
  - Optional package: `pip install openai`
  - Environment variable: `OPENAI_API_KEY`
  - Default model in this repo: `gpt-4.1-mini`
- Gemini provider:
  - Optional packages: `pip install google-genai pillow`
  - Environment variables: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - Default model in this repo: `gemini-3.5-flash`

If a provider SDK is not installed, or the API key is missing, the adapter raises a clear `ProviderConfigurationError`.

## Testing policy

- Default unit tests only use `mock` and `file` providers, or injected stub clients.
- Default test runs must not access the network.
- A live provider integration test exists behind `ENABLE_LIVE_AI_PROVIDER_TESTS=1`; keep it disabled in normal CI and local validation.
