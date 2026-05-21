# AI Review

`AIReviewService` keeps the AI review layer provider-neutral.

## Adapter model

- `VisionReviewAdapter` defines the minimal `review(prompt, review_input)` protocol.
- `MockVisionAdapter` returns a fixed JSON payload for deterministic tests.
- `FileResponseVisionAdapter` reads a JSON response from disk for offline review playback.
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
