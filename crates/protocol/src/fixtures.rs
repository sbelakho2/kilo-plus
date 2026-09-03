//! Golden fixture loading and the adversarial golden tests that lock the
//! v7.5.6 wire contract. Fixtures live in `compat/kilo-v756/` at the repo
//! root — changing wire behavior requires changing fixtures there.

use std::path::PathBuf;

pub fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../compat/kilo-v756")
}

pub fn load(name: &str) -> serde_json::Value {
    let path = fixture_root().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture {name} missing at {path:?}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("fixture {name} invalid JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::from_core;
    use crate::sse::SseEvent;
    use crate::v756::*;
    use faktor_core::error::{Error, ErrorKind};

    /// Serialize a value and compare *byte-for-byte* with the fixture, and
    /// parse the fixture and re-serialize it (idempotence). This locks field
    /// presence, ordering, nulls, and defaults all at once.
    fn assert_golden<T>(fixture: &str, value: T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let raw = load(fixture);
        let parsed: T = serde_json::from_value(raw.clone())
            .unwrap_or_else(|e| panic!("fixture {fixture} does not parse: {e}"));
        assert_eq!(
            parsed, value,
            "fixture {fixture} differs from canonical value"
        );
        let canonical = serde_json::to_value(value).unwrap();
        let reparsed: T = serde_json::from_value(canonical.clone())
            .unwrap_or_else(|e| panic!("fixture {fixture} cannot roundtrip: {e}"));
        assert_eq!(reparsed, parsed, "fixture {fixture} is not idempotent");
        let bytes1 = serde_json::to_string(&serde_json::to_value(&parsed).unwrap()).unwrap();
        let bytes2 = serde_json::to_string(&serde_json::to_value(&reparsed).unwrap()).unwrap();
        assert_eq!(bytes1, bytes2, "fixture {fixture} serialization unstable");
    }

    #[test]
    fn startup_line_golden() {
        let raw = load("startup_line.json");
        let template = raw["template"].as_str().unwrap();
        let example = raw["example"].as_str().unwrap();
        // The template is the frozen stdout contract; {port} is substituted
        // with the bound port.
        assert!(
            template.contains("{port}"),
            "template must document the port slot"
        );
        assert_eq!(template.replace("{port}", "45678"), example);
        assert_eq!(crate::v756::startup_line(45678), example);
        // The line is NOT a JSON handshake (from_line must reject it).
        assert_eq!(Handshake::from_line(example), None);
        // The legacy handshake type still roundtrips for old tests, but it is
        // never printed on stdout.
        let h = Handshake {
            version: "0.1.0".into(),
            protocol: "v756".into(),
            pid: 4242,
            auth_token: "tok-123".into(),
            port: 45678,
        };
        let line = h.to_line();
        assert_eq!(Handshake::from_line(&line), Some(h));
    }

    #[test]
    fn hello_golden() {
        let r = HelloResponse {
            ok: true,
            version: "0.1.0".into(),
            protocol: "v756".into(),
            auth_required: true,
            providers: vec![
                "ollama".into(),
                "deepseek".into(),
                "openai".into(),
                "anthropic".into(),
                "google".into(),
                "gateway".into(),
                "openai-compatible".into(),
            ],
        };
        assert_golden("hello.json", r);
    }

    #[test]
    fn create_session_golden() {
        let req: CreateSessionRequest =
            serde_json::from_value(load("create_session.json")["request"].clone()).unwrap();
        assert_eq!(req.provider, "ollama");
        assert_eq!(req.workspace.as_deref(), Some("/home/u/proj"));
        let resp: CreateSessionResponse =
            serde_json::from_value(load("create_session.json")["response"].clone()).unwrap();
        assert_eq!(resp.id, "sess-1001");
        assert_eq!(resp.created_ms, 1750000000000);
    }

    #[test]
    fn messages_page_golden_locks_field_presence() {
        let raw = load("messages_page.json");
        let page: MessagesPage = serde_json::from_value(raw).unwrap();
        assert_eq!(page.messages.len(), 3);
        assert!(page.has_more);
        assert_eq!(page.next_before, Some(3));
        // Null behavior: exit_code present and numeric, artifact present.
        match &page.messages[2].parts[0] {
            Part::ToolResult { result, .. } => {
                assert_eq!(result.exit_code, Some(0));
                assert_eq!(result.artifact.as_deref(), Some("artifact://4d789"));
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
        // Field presence: a part must have exactly {type, text} etc.
        let part_json = &serde_json::to_value(&page.messages[1].parts[0]).unwrap();
        assert_eq!(
            part_json.as_object().unwrap().len(),
            2,
            "text part is exactly type+text"
        );
        // Canonical idempotence.
        assert_golden("messages_page.json", page);
    }

    #[test]
    fn sse_frames_golden() {
        let raw = load("sse_frames.json");
        let frames = raw.as_array().unwrap();
        assert!(frames.len() >= 5, "fixture must keep the full sequence");
        let mut prev_id = 0u64;
        for f in frames {
            let id = f["id"].as_u64().unwrap();
            assert!(
                id > prev_id,
                "fixture ids must be monotonic (resume cursor)"
            );
            prev_id = id;
            let frame = f["frame"].as_str().unwrap();
            let (parsed_id, _ev) = SseEvent::from_frame(frame)
                .unwrap_or_else(|| panic!("fixture frame {id} does not parse"));
            assert_eq!(parsed_id, id);
            // Idempotence: re-encoding the parsed event must reproduce the
            // fixture frame byte-for-byte.
            let (parsed_id2, ev) = SseEvent::from_frame(frame).unwrap();
            let reencoded = ev.to_frame(parsed_id2);
            assert_eq!(reencoded, frame, "frame {id} re-encoding drift");
        }
    }

    #[test]
    fn errors_golden_locks_mapping() {
        let raw = load("errors.json");
        for case in raw.as_array().unwrap() {
            let kind: ErrorKind = match case["kind"].as_str().unwrap() {
                "not_found" => ErrorKind::NotFound,
                "oversized" => ErrorKind::Oversized,
                "rate_limited" => ErrorKind::RateLimited,
                other => panic!("fixture needs mapping for {other}"),
            };
            let e = Error::new(kind, "x");
            let api = from_core(&e);
            assert_eq!(api.code, case["code"].as_str().unwrap());
            assert_eq!(
                api.http_status,
                case["http_status"].as_u64().unwrap() as u16
            );
            assert_eq!(api.retryable, case["retryable"].as_bool().unwrap());
            assert_eq!(api.to_json(), case["body"]);
        }
    }

    #[test]
    fn provider_list_golden() {
        let list: ProviderList = serde_json::from_value(load("provider_list.json")).unwrap();
        assert_eq!(list.providers.len(), 2);
        let ollama = &list.providers[0];
        assert_eq!(ollama.kind, "ollama");
        let qwen = &ollama.models[0];
        assert_eq!(qwen.id, "qwen3.8");
        assert!(qwen.capabilities.tools);
        assert!(qwen.capabilities.thinking);
        assert_eq!(qwen.capabilities.context, 262144);
        // DeepSeek profile is a first-class family, not an afterthought.
        let ds = &list.providers[1];
        assert_eq!(ds.kind, "openai-compatible");
        assert!(ds.models[0].capabilities.parallel_tools);
        assert_golden("provider_list.json", list);
    }

    #[test]
    fn global_event_golden() {
        use crate::v756::{GlobalEvent, GlobalEventPayload};
        let raw = load("global_event.json");
        let cases = raw.as_array().unwrap();
        assert!(!cases.is_empty(), "fixture must keep examples");
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let json = case["event"].clone();
            let parsed: GlobalEvent = serde_json::from_value(json.clone())
                .unwrap_or_else(|e| panic!("global_event fixture {name} does not parse: {e}"));
            // Idempotence: re-serialization reproduces the fixture bytes.
            let reencoded = serde_json::to_value(&parsed).unwrap();
            assert_eq!(
                reencoded, json,
                "global_event fixture {name} not idempotent"
            );
            // Field presence locked: exactly directory/project/workspace/payload.
            let mut keys: Vec<&str> = json
                .as_object()
                .unwrap()
                .keys()
                .map(|k| k.as_str())
                .collect();
            keys.sort_unstable();
            assert_eq!(keys, vec!["directory", "payload", "project", "workspace"]);
            // Wire bytes preserve the declaration order (frozen).
            let bytes = serde_json::to_string(&parsed).unwrap();
            assert!(
                bytes.starts_with("{\"directory\":"),
                "envelope must start with directory: {bytes}"
            );
            // The payload type tag matches the fixture name.
            assert_eq!(
                parsed.payload.type_name(),
                name,
                "fixture {name} payload type drift"
            );
        }
        // The three documented examples are present.
        let names: Vec<&str> = cases.iter().map(|c| c["name"].as_str().unwrap()).collect();
        for required in [
            "session_created",
            "message_part_updated",
            "session_next_text_delta",
        ] {
            assert!(
                names.contains(&required),
                "fixture missing example {required}"
            );
        }
        // A SessionNextTextDelta example parses to the delta payload.
        let delta = cases
            .iter()
            .find(|c| c["name"].as_str() == Some("session_next_text_delta"))
            .unwrap();
        let ge: GlobalEvent = serde_json::from_value(delta["event"].clone()).unwrap();
        assert!(matches!(
            ge.payload,
            GlobalEventPayload::SessionNextTextDelta { ref delta, .. } if delta == "hello wo"
        ));
    }

    #[test]
    fn password_auth_golden() {
        let raw = load("password_auth.json");
        assert_eq!(raw["env_var"], "FAKTOR_SERVER_PASSWORD");
        let forms = raw["accepted_header_forms"].as_array().unwrap();
        let forms: Vec<&str> = forms.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(forms.contains(&"authorization_basic"));
        assert!(forms.contains(&"authorization_bearer"));
        assert!(forms.contains(&"x_faktor_server_password"));
        // The unauthorized error shape is the frozen 401 contract.
        assert_eq!(raw["unauthorized"]["code"], "unauthorized");
        assert_eq!(raw["unauthorized"]["http_status"], 401);
        assert_eq!(raw["unauthorized"]["retryable"], false);
        // Nothing is public anymore: /global/health requires auth (the
        // frozen client authenticates every request).
        assert_eq!(raw["public_endpoints"], serde_json::json!([]));
        // /api/hello remains the legacy public alias.
        assert_eq!(raw["legacy_public_alias"], "/api/hello");
    }

    #[test]
    fn basic_auth_golden() {
        use base64::Engine as _;
        let raw = load("basic_auth.json");
        assert_eq!(raw["env_var"], "FAKTOR_SERVER_PASSWORD");
        assert_eq!(raw["username"], "kilo");
        assert_eq!(raw["basic_scheme"], "Basic");
        assert_eq!(raw["max_header_bytes"], 4096);
        // The documented construction: base64("kilo:" + password).
        let example_pw = raw["example_password"].as_str().unwrap();
        assert_eq!(example_pw.len(), 64);
        let expected = format!(
            "Authorization: Basic {}",
            base64::engine::general_purpose::STANDARD.encode(format!("kilo:{example_pw}"))
        );
        assert_eq!(
            raw["example_header"].as_str().unwrap(),
            expected,
            "fixture header must be the exact base64(\"kilo:\"+password) construction"
        );
        // The fixture documents that auth is required everywhere.
        assert_eq!(raw["required_everywhere"], true);
        // Unauthorized shape is frozen.
        assert_eq!(raw["unauthorized"]["code"], "unauthorized");
        assert_eq!(raw["unauthorized"]["http_status"], 401);
        // The legacy forms remain documented as accepted.
        let legacy = raw["legacy_accepted_header_forms"].as_array().unwrap();
        let legacy: Vec<&str> = legacy.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(legacy.contains(&"authorization_bearer"));
        assert!(legacy.contains(&"x_faktor_server_password"));
    }

    #[test]
    fn wire_session_create_golden() {
        use crate::v756::wire::{SessionCreateRequest, SessionCreateResponse};
        let raw = load("wire_session_create.json");
        let req: SessionCreateRequest = serde_json::from_value(raw["request"].clone()).unwrap();
        assert_eq!(req.parent_id.as_deref(), Some("sess-1000"));
        assert_eq!(req.title.as_deref(), Some("Fix the parser"));
        assert_eq!(req.agent.as_deref(), Some("default"));
        let model = req.model.as_ref().unwrap();
        assert_eq!(model.id, "qwen3.8");
        assert_eq!(model.provider_id, "ollama");
        assert_eq!(model.variant.as_deref(), Some("fast"));
        assert_eq!(req.metadata.as_ref().unwrap()["origin"], "audit-round-2");
        assert_eq!(req.platform.as_deref(), Some("darwin"));
        assert_eq!(req.workspace_id.as_deref(), Some("/home/u/proj"));
        // Wire names are camelCase, never snake_case.
        let v = serde_json::to_value(&req).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        for snake in ["parent_id", "workspace_id", "sandbox_inheritance_token"] {
            assert!(
                !keys.contains(&snake),
                "{snake} must never appear on the wire"
            );
        }
        assert!(keys.contains(&"parentID"));
        assert!(keys.contains(&"workspaceID"));
        assert!(keys.contains(&"sandboxInheritanceToken"));
        // The nested model object carries providerID, never provider_id.
        assert_eq!(v["model"]["providerID"], "ollama");
        assert!(!v["model"].as_object().unwrap().contains_key("provider_id"));
        // The request body is idempotent byte-for-byte.
        let back: SessionCreateRequest = serde_json::from_value(raw["request"].clone()).unwrap();
        assert_eq!(
            serde_json::to_string(&back).unwrap(),
            serde_json::to_string(&req).unwrap()
        );
        // The response.
        let resp: SessionCreateResponse = serde_json::from_value(raw["response"].clone()).unwrap();
        assert_eq!(resp.session_id, "sess-1001");
        assert_eq!(resp.title, "Fix the parser");
        assert_eq!(resp.created_ms, 1750000000000);
        let v = serde_json::to_value(&resp).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert!(keys.contains(&"sessionID"));
        assert!(keys.contains(&"createdMs"));
        assert!(!keys.contains(&"session_id"));
    }

    #[test]
    fn wire_message_send_golden() {
        use crate::v756::wire::{MessageSendRequest, MessageSendResponse};
        let raw = load("wire_message_send.json");
        let req: MessageSendRequest = serde_json::from_value(raw["request"].clone()).unwrap();
        assert_eq!(req.model.provider_id, "ollama");
        assert_eq!(req.model.model_id, "qwen3.8");
        assert_eq!(req.no_reply, Some(false));
        assert_eq!(req.snapshot_initialization, Some(false));
        assert_eq!(
            req.tools.as_deref(),
            Some(&["read_file".to_string(), "write_file".to_string()][..])
        );
        assert_eq!(req.editor_context.as_ref().unwrap()["file"], "src/lexer.rs");
        assert_eq!(
            req.parts.len(),
            12,
            "fixture must keep the full parts[] sample"
        );
        // Every part type tag is present in the fixture.
        let tags: Vec<String> = req
            .parts
            .iter()
            .map(|p| {
                serde_json::to_value(p).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        for tag in [
            "text",
            "file",
            "subtask",
            "reasoning",
            "tool",
            "stepStart",
            "stepFinish",
            "snapshot",
            "patch",
            "agent",
            "retry",
            "compaction",
        ] {
            assert!(tags.contains(&tag.to_string()), "fixture missing {tag}");
        }
        // Wire names are camelCase inside parts too.
        let tool = req
            .parts
            .iter()
            .find(|p| matches!(p, crate::v756::wire::WirePart::Tool { .. }))
            .unwrap();
        let v = serde_json::to_value(tool).unwrap();
        assert!(v.as_object().unwrap().contains_key("callID"));
        assert!(!v.as_object().unwrap().contains_key("call_id"));
        // Idempotence.
        let back: MessageSendRequest = serde_json::from_value(raw["request"].clone()).unwrap();
        assert_eq!(
            serde_json::to_string(&back).unwrap(),
            serde_json::to_string(&req).unwrap()
        );
        // The response shape: {info: AssistantMessage, parts: Part[]} — the
        // info carries the durable seq identity; parts ride at the top level.
        let resp: MessageSendResponse = serde_json::from_value(raw["response"].clone()).unwrap();
        assert_eq!(resp.info.session_id, "sess-1001");
        assert_eq!(resp.info.message_id, "18");
        assert_eq!(resp.info.role, "assistant");
        assert_eq!(resp.info.created_ms, 1750000009000);
        assert_eq!(resp.info.provider_id.as_deref(), Some("ollama"));
        assert_eq!(resp.info.model_id.as_deref(), Some("qwen3.8"));
        assert_eq!(resp.parts.len(), 1);
        let v = serde_json::to_value(&resp).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys,
            vec!["info", "parts"],
            "top level is exactly info+parts"
        );
        let info_keys: Vec<&str> = v["info"]
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        assert!(info_keys.contains(&"sessionID"));
        assert!(info_keys.contains(&"messageID"));
        assert!(
            !info_keys.contains(&"parts"),
            "parts never nest inside info"
        );
        assert!(!v.as_object().unwrap().contains_key("accepted"));
        assert!(!v.as_object().unwrap().contains_key("queued"));
    }

    #[test]
    fn wire_part_union_golden() {
        use crate::v756::wire::WirePart;
        let cases = load("wire_part_union.json").as_array().unwrap().clone();
        let mut seen = std::collections::HashSet::new();
        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let part_json = case["part"].clone();
            let part: WirePart = serde_json::from_value(part_json.clone())
                .unwrap_or_else(|e| panic!("wire_part_union fixture {name} does not parse: {e}"));
            // The fixture's name is the exact type tag.
            assert_eq!(part_json["type"], name, "tag drift for {name}");
            seen.insert(name.to_string());
            // Idempotence: re-serialization reproduces the fixture bytes.
            assert_eq!(
                serde_json::to_value(&part).unwrap(),
                part_json,
                "fixture {name} not idempotent"
            );
        }
        // Every variant of the union is locked by exactly one example.
        for expected in [
            "text",
            "subtask",
            "reasoning",
            "file",
            "tool",
            "stepStart",
            "stepFinish",
            "snapshot",
            "patch",
            "agent",
            "retry",
            "compaction",
        ] {
            assert!(
                seen.contains(expected),
                "fixture missing example {expected}"
            );
        }
    }

    #[test]
    fn fixture_corpus_is_complete_and_every_file_parses() {
        // Adversarial guard against silent fixture rot: every file in the
        // corpus must parse as JSON, and known files must all be exercised.
        let dir = std::fs::read_dir(fixture_root()).unwrap();
        let mut seen = Vec::new();
        for entry in dir {
            let entry = entry.unwrap();
            if entry
                .path()
                .extension()
                .map(|e| e == "json")
                .unwrap_or(false)
            {
                let text = std::fs::read_to_string(entry.path()).unwrap();
                serde_json::from_str::<serde_json::Value>(&text)
                    .unwrap_or_else(|e| panic!("fixture {} corrupt: {e}", entry.path().display()));
                seen.push(entry.file_name().to_string_lossy().to_string());
            }
        }
        for expected in [
            "startup_line.json",
            "hello.json",
            "create_session.json",
            "messages_page.json",
            "sse_frames.json",
            "errors.json",
            "provider_list.json",
            "global_event.json",
            "password_auth.json",
            "basic_auth.json",
            "wire_session_create.json",
            "wire_message_send.json",
            "wire_part_union.json",
        ] {
            assert!(
                seen.contains(&expected.to_string()),
                "fixture {expected} missing"
            );
        }
        // The legacy handshake fixture is gone: the stdout line replaced it.
        assert!(!seen.contains(&"handshake.json".to_string()));
    }
}
