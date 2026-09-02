//! Golden checks for `fixtures/providers`: the frozen wire shapes the
//! adapters' mock tests consume must parse and carry their required keys.
//! Everything is parsed with serde_json only — no live adapter code runs.

use std::path::Path;

fn providers_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/providers")
}

fn load_json(name: &str) -> serde_json::Value {
    let path = providers_dir().join(name);
    let raw =
        std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_slice(&raw)
        .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", path.display()))
}

/// Wire hygiene shared with the adapters' own assertions: internal runtime
/// fields must never appear on the wire.
fn assert_no_leakage(value: &serde_json::Value) {
    for leaked in [
        "operation_id",
        "session_id",
        "attempt",
        "deadline_ms",
        "cancellation",
    ] {
        assert!(
            !value.as_object().unwrap().contains_key(leaked),
            "wire frame leaks {leaked}: {value}"
        );
    }
}

/// Stream fixtures (NDJSON/SSE) are multi-frame documents: the whole file
/// must not parse as a single JSON value. Every non-empty, non-`data:`
/// line must parse as a standalone JSON document, and every `data:` line
/// must carry JSON (or the `[DONE]` sentinel).
fn assert_parseable(raw: &str, path: &std::path::Path) {
    let name = path.display();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
        assert!(!value.is_null(), "{name} must be a real JSON document");
        return;
    }
    let mut saw_frame = false;
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        saw_frame = true;
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            assert!(
                data == "[DONE]" || serde_json::from_str::<serde_json::Value>(data).is_ok(),
                "{name}: line {} is a data: frame without JSON payload",
                i + 1
            );
        } else {
            serde_json::from_str::<serde_json::Value>(line)
                .unwrap_or_else(|e| panic!("{name}: line {} is not valid JSON: {e}", i + 1));
        }
    }
    assert!(saw_frame, "{name} must contain at least one frame");
}

#[test]
fn provider_fixtures_all_files_parse() {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(providers_dir()).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let raw = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            assert_parseable(&raw, &path);
            files.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    files.sort();
    assert_eq!(
        files,
        [
            "anthropic-messages-stream.json",
            "gemini-stream.json",
            "ollama-api-chat-stream.json",
            "ollama-api-show-qwen3.8.json",
            "ollama-api-tags.json",
            "openai-chat-stream.json",
        ],
        "the provider fixture corpus is frozen"
    );
}

#[test]
fn provider_fixtures_ollama_tags_shape() {
    let doc = load_json("ollama-api-tags.json");
    let models = doc["models"].as_array().expect("models array");
    assert_eq!(models.len(), 2);
    let names: Vec<&str> = models
        .iter()
        .map(|m| m["name"].as_str().expect("every model has a name"))
        .collect();
    assert!(names.contains(&"qwen3.8:latest"));
    assert!(names.contains(&"llama3.2:3b"));
    // The wire order matches the adapter's mock byte-for-byte: the server
    // returns discovery unsorted, and the adapter sorts the result. The
    // fixture locks the wire shape; the sorted contract is the adapter's.
    assert_eq!(names, ["qwen3.8:latest", "llama3.2:3b"]);
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(sorted, ["llama3.2:3b", "qwen3.8:latest"]);
    // Optional detail fields are allowed on the wire but never required.
    for m in models {
        let obj = m.as_object().unwrap();
        for optional in ["model", "size", "modified_at", "details"] {
            if obj.contains_key(optional) {
                assert!(!obj[optional].is_null(), "{optional} must not be null");
            }
        }
    }
}

#[test]
fn provider_fixtures_ollama_show_shape() {
    let doc = load_json("ollama-api-show-qwen3.8.json");
    // The adapter maps model_info.context_length into ModelCapabilities.
    assert_eq!(doc["model_info"]["context_length"], 262_144);
    let caps = doc["capabilities"].as_array().expect("capabilities array");
    for required in ["tools", "vision", "embeddings", "reasoning"] {
        assert!(
            caps.iter().any(|c| c.as_str() == Some(required)),
            "missing capability {required}: {caps:?}"
        );
    }
    assert_no_leakage(&doc);
}

#[test]
fn provider_fixtures_ollama_chat_stream_frames() {
    let raw = std::fs::read_to_string(providers_dir().join("ollama-api-chat-stream.json"))
        .unwrap_or_else(|e| panic!("ollama-api-chat-stream.json: {e}"));
    let mut frames = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        assert!(!line.is_empty(), "NDJSON frame {} is blank", i + 1);
        let value: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("NDJSON frame {} is not valid JSON: {e}", i + 1));
        assert_no_leakage(&value);
        frames.push(value);
    }
    // Text frames exist before the tool call.
    assert!(frames.iter().any(|f| f["message"]["content"]
        .as_str()
        .map(|s| !s.is_empty())
        .unwrap_or(false)));
    // Exactly one tool_calls frame, in the native shape.
    let tool_frames: Vec<_> = frames
        .iter()
        .filter(|f| {
            f["message"]["tool_calls"]
                .as_array()
                .is_some_and(|a| !a.is_empty())
        })
        .collect();
    assert_eq!(tool_frames.len(), 1, "exactly one tool_calls frame");
    let tc = &tool_frames[0]["message"]["tool_calls"][0]["function"];
    assert_eq!(tc["name"], "read_file");
    assert_eq!(tc["arguments"]["path"], "a.rs");
    // The terminal frame carries done=true, like the adapter's mock.
    let last = frames.last().unwrap();
    assert_eq!(last["done"], true, "stream must terminate with done=true");
    // Malformed NDJSON must be absent: every frame is a JSON object.
    assert!(frames.iter().all(|f| f.is_object()));
}

#[test]
fn provider_fixtures_openai_sse_shape() {
    let raw = std::fs::read_to_string(providers_dir().join("openai-chat-stream.json"))
        .unwrap_or_else(|e| panic!("openai-chat-stream.json: {e}"));
    let mut payloads = Vec::new();
    let mut done = false;
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue; // SSE frame separator
        }
        assert!(
            line.starts_with("data:"),
            "line {} is not an SSE data frame",
            i + 1
        );
        let data = line.trim_start_matches("data:").trim();
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(data)
            .unwrap_or_else(|e| panic!("SSE payload on line {} is not JSON: {e}", i + 1));
        assert_no_leakage(&value);
        payloads.push(value);
    }
    assert!(done, "stream must terminate with [DONE]");
    // Tool-call deltas accumulate across frames into the frozen arguments.
    let deltas: Vec<&serde_json::Value> = payloads
        .iter()
        .flat_map(|p| {
            p["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .map(|a| a.as_slice())
                .unwrap_or(&[])
        })
        .collect();
    assert!(!deltas.is_empty(), "stream must carry tool_call deltas");
    assert_eq!(deltas[0]["id"], "c1");
    assert_eq!(deltas[0]["function"]["name"], "read_file");
    let args: String = deltas
        .iter()
        .filter_map(|d| d["function"]["arguments"].as_str())
        .collect();
    assert_eq!(args, "{\"path\":\"a.rs\"}");
    // The accumulator flushes on finish_reason == "tool_calls".
    assert!(payloads
        .iter()
        .any(|p| p["choices"][0]["finish_reason"] == "tool_calls"));
    // A text delta precedes the tool call.
    assert!(payloads
        .iter()
        .any(|p| p["choices"][0]["delta"]["content"].as_str().is_some()));
}

#[test]
fn provider_fixtures_anthropic_sse_shape() {
    let raw = std::fs::read_to_string(providers_dir().join("anthropic-messages-stream.json"))
        .unwrap_or_else(|e| panic!("anthropic-messages-stream.json: {e}"));
    let mut events: Vec<(String, serde_json::Value)> = Vec::new();
    let mut done = false;
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        assert!(
            line.starts_with("data:"),
            "line {} is not an SSE data frame",
            i + 1
        );
        let data = line.trim_start_matches("data:").trim();
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(data)
            .unwrap_or_else(|e| panic!("SSE payload on line {} is not JSON: {e}", i + 1));
        assert_no_leakage(&value);
        let ty = value["type"]
            .as_str()
            .unwrap_or_else(|| panic!("anthropic event on line {} has no type", i + 1));
        events.push((ty.to_string(), value));
    }
    assert!(done, "stream must terminate with [DONE]");
    for required in [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_stop",
    ] {
        assert!(
            events.iter().any(|(t, _)| t == required),
            "missing {required} event: {:?}",
            events.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>()
        );
    }
    // The tool_use block carries the frozen id/name.
    let tool_use = events
        .iter()
        .find(|(t, v)| t == "content_block_start" && v["content_block"]["type"] == "tool_use")
        .expect("tool_use block start");
    assert_eq!(tool_use.1["content_block"]["id"], "toolu_1");
    assert_eq!(tool_use.1["content_block"]["name"], "read_file");
    // input_json_delta fragments concatenate into the frozen arguments.
    let mut input_json = String::new();
    for (t, v) in &events {
        if t == "content_block_delta" && v["delta"]["type"] == "input_json_delta" {
            input_json.push_str(v["delta"]["partial_json"].as_str().unwrap_or_default());
        }
    }
    assert_eq!(input_json, "{\"path\":\"a.rs\"}");
}

#[test]
fn provider_fixtures_gemini_sse_shape() {
    let raw = std::fs::read_to_string(providers_dir().join("gemini-stream.json"))
        .unwrap_or_else(|e| panic!("gemini-stream.json: {e}"));
    let mut text = String::new();
    let mut call: Option<(String, String)> = None;
    let mut done = false;
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        assert!(
            line.starts_with("data:"),
            "line {} is not an SSE data frame",
            i + 1
        );
        let data = line.trim_start_matches("data:").trim();
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(data)
            .unwrap_or_else(|e| panic!("SSE payload on line {} is not JSON: {e}", i + 1));
        assert_no_leakage(&value);
        let parts = value["candidates"][0]["content"]["parts"]
            .as_array()
            .expect("candidates[0].content.parts");
        for part in parts {
            if let Some(t) = part["text"].as_str() {
                text.push_str(t);
            }
            if part["functionCall"].is_object() {
                let fc = &part["functionCall"];
                call = Some((
                    fc["name"].as_str().expect("functionCall name").to_string(),
                    fc["args"]["path"]
                        .as_str()
                        .expect("functionCall args.path")
                        .to_string(),
                ));
            }
        }
    }
    assert!(done, "stream must terminate with [DONE]");
    assert_eq!(text, "let me check");
    assert_eq!(call, Some(("read_file".to_string(), "a.rs".to_string())));
}
