# Provider behavior fixtures

Frozen wire captures for the provider adapters' mock tests. Every file is
consumed by (at least) the named adapter unit test; the golden test
`tests/integration/tests/provider_fixtures.rs` parses all of them with
`serde_json` only (no live adapter code) and asserts the shapes below.

## Ollama (crates/ollama/src/lib.rs)

- `ollama-api-tags.json` — `GET /api/tags` discovery response. Shape
  consumed by `discovery_via_api_tags`: `{"models":[{"name": ...}]}`.
  Optional per-model fields `model`, `size`, `modified_at`, `details` may
  be present or absent.
- `ollama-api-show-qwen3.8.json` — `POST /api/show` capability probe.
  Shape consumed by `capability_probe_maps_metadata`:
  `model_info.context_length` (262144) and the `capabilities` list
  `["tools", "vision", "embeddings", "reasoning"]`. The adapter maps the
  list to `ModelCapabilities` flags.
- `ollama-api-chat-stream.json` — native `/api/chat` NDJSON stream, one
  JSON object per line. Frames: text chunks, a `message.tool_calls` frame
  (`{"function":{"name","arguments"}}`), and a terminal `{"done":true}`
  frame. Shapes consumed by `native_tool_call_parsed`,
  `wire_shape_is_native_and_clean`, and `malformed_ndjson_is_malformed_error`.

## OpenAI (crates/openai/src/lib.rs)

- `openai-chat-stream.json` — `/chat/completions` SSE stream:
  `data: {json}` frames separated by blank lines, terminated by
  `data: [DONE]`. Includes `delta.tool_calls` frames whose `function.arguments`
  arrive as string fragments, and a `finish_reason: "tool_calls"` frame.
  Shape consumed by `tool_call_accumulates_and_completes` and
  `wire_body_has_no_internal_leakage`.

## Anthropic (crates/anthropic/src/lib.rs)

- `anthropic-messages-stream.json` — `/v1/messages` SSE stream with typed
  events (`content_block_start`, `content_block_delta` with
  `input_json_delta.partial_json` fragments, `message_stop`), terminated by
  `data: [DONE]`. Shape consumed by `text_and_tool_use_stream` and
  `wire_shape_is_clean`.

## Google Gemini (crates/google/src/lib.rs)

- `gemini-stream.json` — `streamGenerateContent` SSE stream:
  `data: {json}` frames with `candidates[0].content.parts[]`, one `text`
  part and one `functionCall` part (`{"name","args","id"}`), terminated by
  `data: [DONE]`. Shape consumed by `text_and_function_call`.
