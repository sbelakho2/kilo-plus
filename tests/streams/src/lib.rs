//! faktor-tests-streams — the provider-stream adversarial suite (audit item
//! 25). Every provider adapter must ingest adversarial streams identically
//! AFTER normalization, so each family here is driven through its REAL
//! adapter (`crates/openai` chat + responses codecs, `crates/ollama` native)
//! against the scripted `MockServer` (`crates/provider/src/testing.rs`) and
//! the delivered chunks are reduced to one normalized sequence
//! (`Text/Reasoning/ToolCall/Usage/Done/Err(kind)`).
//!
//! Scenarios locked per family:
//!   a. duplicate text chunks — the adapter does NOT dedup (locked: two Text
//!      chunks for two identical wire deltas; the runtime normalizes);
//!   b. missing Done — Done is emitted exactly once at EOF;
//!   c. double Done — the second terminal is a no-op (exactly one Done);
//!   d. tool arguments fragmented arbitrarily — ONE complete ToolCall with
//!      reassembled args;
//!   e. tool ids changed mid-call — openai-chat accumulates per WIRE INDEX
//!      (a re-id'd index is last-writer-wins on the id; distinct indices
//!      never cross-talk); openai-responses accumulates per item id and a
//!      delta for an unknown item is dropped;
//!   f. usage ordering — chunks surface in wire order; a usage object that
//!      shares a frame with a content delta never becomes a chunk
//!      (documented);
//!   g. 429 -> Err(RateLimited), 500 -> Err(Server) (both retryable, code
//!      preserved) before the first token; socket death AFTER content ->
//!      Err(Network) with the already-delivered content chunks preserved;
//!   h. partial UTF-8 fragmented across HTTP-chunk boundaries reassembles
//!      (byte-level line buffering, mirroring the transport tests).
//!
//! Adapter-level invariants ONLY. Runtime-level normalization (dedup,
//! duplicate suppression) lives in the runtime crates — not in the crate set
//! this suite may touch — so it is out of scope here by design; each
//! per-family "observed normalization" assertion below documents the exact
//! adapter contract the runtime normalizes against.
//!
//! Everything lives under `#[cfg(test)]` (a test-harness crate; the lib view
//! exists only so the workspace builds it).

#[cfg(test)]
mod streams_tests {
    use std::time::Duration;

    use faktor_core::cancellation::CancellationToken;
    use faktor_core::id::{OpId, SessionId};
    use faktor_ollama::{OllamaConfig, OllamaProvider};
    use faktor_openai::{OpenAiConfig, OpenAiProvider};
    use faktor_provider::testing::{MockAction, MockServer};
    use faktor_provider::{
        ContentPart, GenericAgentRequest, ProviderChunk, ProviderError, ProviderErrorKind,
        ProviderStream, RequestMessage, RequestMeta, Role, ToolSpec,
    };
    use futures::StreamExt;
    use serde_json::{json, Value};

    // ------------------------------------------------------------- harness

    /// One normalized stream item — the adapter-level event vocabulary every
    /// family must produce after ingestion (names only, payloads attached so
    /// reassembly is assertable).
    #[derive(Debug, Clone, PartialEq)]
    enum Norm {
        Text(String),
        Reasoning(String),
        ToolCall {
            id: String,
            name: String,
            args: Value,
            complete: bool,
        },
        Usage {
            tokens_in: u64,
            tokens_out: u64,
        },
        Done,
        Err {
            kind: ProviderErrorKind,
            retryable: bool,
            code: Option<String>,
        },
    }

    fn norm(item: Result<ProviderChunk, ProviderError>) -> Norm {
        match item {
            Ok(ProviderChunk::Text { text }) => Norm::Text(text),
            Ok(ProviderChunk::Reasoning { text }) => Norm::Reasoning(text),
            Ok(ProviderChunk::ToolCall {
                id,
                name,
                input,
                complete,
            }) => Norm::ToolCall {
                id,
                name,
                args: input,
                complete,
            },
            Ok(ProviderChunk::Usage {
                tokens_in,
                tokens_out,
                ..
            }) => Norm::Usage {
                tokens_in,
                tokens_out,
            },
            Ok(ProviderChunk::Done) => Norm::Done,
            Err(e) => Norm::Err {
                kind: e.kind,
                retryable: e.retryable,
                code: e.code,
            },
        }
    }

    /// Drain one provider stream to its end and reduce every item to Norm.
    async fn drive(stream: ProviderStream) -> Vec<Norm> {
        let mut stream = stream;
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(norm(item));
        }
        out
    }

    fn request(model: &str) -> GenericAgentRequest {
        GenericAgentRequest {
            model: model.into(),
            system: "sys".into(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text("hi")],
            }],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "read".into(),
                input_schema: json!({"type": "object"}),
            }],
            max_output: Some(1000),
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "streams-test".into(),
                attempt: 0,
                deadline_ms: 5000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    /// One SSE frame from a JSON event.
    fn sse_frame(v: &Value) -> String {
        format!("data: {v}\n\n")
    }

    /// The SSE end-of-stream marker frame.
    fn sse_done() -> String {
        "data: [DONE]\n\n".into()
    }

    // ---------------------------------------------------------- openai chat

    fn chat_text(t: &str) -> Value {
        json!({"choices": [{"delta": {"content": t}}]})
    }

    fn chat_finish(reason: &str) -> Value {
        json!({"choices": [{"delta": {}, "finish_reason": reason}]})
    }

    fn chat_usage_frame(prompt: u64, completion: u64) -> Value {
        json!({
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        })
    }

    fn chat_tool_frame(calls: Vec<Value>) -> Value {
        json!({"choices": [{"delta": {"tool_calls": calls}}]})
    }

    /// One chat tool-call delta fragment (empty id/name fields stay absent,
    /// exactly like a real server omits them on continuation frames).
    fn chat_tc(index: u64, id: &str, name: &str, arguments: &str) -> Value {
        let mut v = json!({"index": index});
        if !id.is_empty() {
            v["id"] = json!(id);
        }
        let mut function = json!({"arguments": arguments});
        if !name.is_empty() {
            function["name"] = json!(name);
        }
        v["function"] = function;
        v
    }

    /// One chat SSE stream: script the mock, drive the real openai-chat
    /// adapter, and reduce the delivered items to Norm.
    async fn chat_sse_norms(events: Vec<String>) -> Vec<Norm> {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Sse {
                status: 200,
                events,
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        drive(provider.stream(request("m"))).await
    }

    async fn chat_respond_norms(status: u16, body: String) -> Vec<Norm> {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond { status, body },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        drive(provider.stream(request("m"))).await
    }

    #[tokio::test]
    async fn chat_duplicate_text_not_deduped_done_once() {
        // (a) Two IDENTICAL content deltas: the adapter emits two Text
        // chunks — dedup is deliberately NOT applied here (the runtime
        // normalizes duplicates); the stream stays well-formed with exactly
        // one Done.
        let norms = chat_sse_norms(vec![
            sse_frame(&chat_text("hi")),
            sse_frame(&chat_text("hi")),
            sse_done(),
        ])
        .await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Text("hi".into()), Norm::Done],
            "wire duplicates pass through the adapter verbatim"
        );
    }

    #[tokio::test]
    async fn chat_missing_done_emits_done_at_eof() {
        // (b) The server never sends finish_reason or [DONE]: the adapter
        // emits Done exactly once at EOF — the stream is still well-formed.
        let norms = chat_sse_norms(vec![
            sse_frame(&chat_text("hi")),
            sse_frame(&chat_text(" there")),
        ])
        .await;
        assert_eq!(
            norms,
            vec![
                Norm::Text("hi".into()),
                Norm::Text(" there".into()),
                Norm::Done
            ],
            "Done is synthesized once at EOF"
        );
    }

    #[tokio::test]
    async fn chat_double_done_emits_one_done() {
        // (c) A second [DONE] (and any events after it) is a no-op: the
        // first terminal ends the stream — exactly one Done surfaces.
        let norms = chat_sse_norms(vec![
            sse_frame(&chat_text("hi")),
            sse_done(),
            sse_done(),
            sse_frame(&chat_text("ignored")),
        ])
        .await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Done],
            "the first terminal marker ends the stream; later frames are never read"
        );
    }

    #[tokio::test]
    async fn chat_tool_args_fragmented_into_one_complete_call() {
        // (d) Arguments fragmented across SIX deltas reassemble into ONE
        // complete ToolCall; the chunk appears only at the finishing marker.
        let norms = chat_sse_norms(vec![
            sse_frame(&chat_tool_frame(vec![chat_tc(
                0,
                "call_1",
                "read_file",
                r#"{"path":"#,
            )])),
            sse_frame(&chat_tool_frame(vec![chat_tc(0, "", "", r#""a/"#)])),
            sse_frame(&chat_tool_frame(vec![chat_tc(0, "", "", r#"b.rs","#)])),
            sse_frame(&chat_tool_frame(vec![chat_tc(0, "", "", r#""mode":"#)])),
            sse_frame(&chat_tool_frame(vec![chat_tc(0, "", "", r#""read""#)])),
            sse_frame(&chat_tool_frame(vec![chat_tc(0, "", "", "}")])),
            sse_frame(&chat_finish("tool_calls")),
            sse_done(),
        ])
        .await;
        assert_eq!(
            norms,
            vec![
                Norm::ToolCall {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    args: json!({"path": "a/b.rs", "mode": "read"}),
                    complete: true,
                },
                Norm::Done,
            ],
            "arbitrarily fragmented arguments must reassemble into one complete call"
        );
    }

    #[tokio::test]
    async fn chat_tool_id_flip_accumulates_per_index_last_id_wins() {
        // (e) A tool id that CHANGES mid-call on the same wire index: the
        // adapter keys accumulation by the wire index, so the fragments
        // join and the LAST non-empty id wins — no second call for 'a' ever
        // materializes. Documented actual behavior: the runtime must not
        // rely on id stability inside one call.
        let norms = chat_sse_norms(vec![
            sse_frame(&chat_tool_frame(vec![chat_tc(
                0,
                "call_a",
                "read_file",
                r#"{"path":"#,
            )])),
            sse_frame(&chat_tool_frame(vec![chat_tc(
                0,
                "call_b",
                "",
                r#""b.rs"}"#,
            )])),
            sse_frame(&chat_finish("tool_calls")),
            sse_done(),
        ])
        .await;
        assert_eq!(
            norms,
            vec![
                Norm::ToolCall {
                    id: "call_b".into(),
                    name: "read_file".into(),
                    args: json!({"path": "b.rs"}),
                    complete: true,
                },
                Norm::Done,
            ],
            "index-keyed accumulation joins fragments across an id flip; the last id wins"
        );
    }

    #[tokio::test]
    async fn chat_parallel_tool_ids_accumulate_without_crosstalk() {
        // (e) Two calls ('a' and 'b') whose fragments interleave across
        // frames: each accumulates its own complete call with the correct
        // args — no cross-talk between ids.
        let norms = chat_sse_norms(vec![
            sse_frame(&chat_tool_frame(vec![
                chat_tc(0, "call_a", "read_file", r#"{"path":"#),
                chat_tc(1, "call_b", "sum", r#"{"nums":"#),
            ])),
            sse_frame(&chat_tool_frame(vec![
                chat_tc(0, "", "", r#""a.rs""#),
                chat_tc(1, "", "", "[1,2]"),
            ])),
            sse_frame(&chat_tool_frame(vec![
                chat_tc(0, "", "", "}"),
                chat_tc(1, "", "", "}"),
            ])),
            sse_frame(&chat_finish("tool_calls")),
            sse_done(),
        ])
        .await;
        assert_eq!(
            norms,
            vec![
                Norm::ToolCall {
                    id: "call_a".into(),
                    name: "read_file".into(),
                    args: json!({"path": "a.rs"}),
                    complete: true,
                },
                Norm::ToolCall {
                    id: "call_b".into(),
                    name: "sum".into(),
                    args: json!({"nums": [1, 2]}),
                    complete: true,
                },
                Norm::Done,
            ],
            "interleaved fragments never cross-talk between ids"
        );
    }

    #[tokio::test]
    async fn chat_usage_passthrough_preserves_wire_order() {
        // (f) Dedicated usage frames pass through as Usage chunks in WIRE
        // order — both before and after text deltas — followed by Done.
        let usage_first = chat_sse_norms(vec![
            sse_frame(&chat_usage_frame(7, 3)),
            sse_frame(&chat_text("hi")),
            sse_done(),
        ])
        .await;
        assert_eq!(
            usage_first,
            vec![
                Norm::Usage {
                    tokens_in: 7,
                    tokens_out: 3,
                },
                Norm::Text("hi".into()),
                Norm::Done,
            ],
            "a usage frame before the text keeps its wire position"
        );
        let text_first = chat_sse_norms(vec![
            sse_frame(&chat_text("hi")),
            sse_frame(&chat_usage_frame(7, 3)),
            sse_done(),
        ])
        .await;
        assert_eq!(
            text_first,
            vec![
                Norm::Text("hi".into()),
                Norm::Usage {
                    tokens_in: 7,
                    tokens_out: 3,
                },
                Norm::Done,
            ],
            "a usage frame after the text keeps its wire position"
        );
    }

    #[tokio::test]
    async fn chat_usage_on_content_frame_is_not_a_chunk() {
        // (f) A usage object that shares its frame with a content delta is
        // NEVER surfaced (the frame yields its first chunk — the text — and
        // the rest of the frame is not re-scanned). Documented actual
        // behavior: usage reaches the agent only on dedicated frames.
        let norms = chat_sse_norms(vec![sse_frame(&json!({
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }))])
        .await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Done],
            "in-frame usage after a content delta is dropped, not reordered or duplicated"
        );
    }

    #[tokio::test]
    async fn chat_429_and_500_map_retryable_before_first_token() {
        // (g) HTTP errors before the first token map to the retryable
        // kinds, preserving the status code.
        let rate_limited =
            chat_respond_norms(429, r#"{"error":{"message":"slow down"}}"#.into()).await;
        assert_eq!(
            rate_limited,
            vec![Norm::Err {
                kind: ProviderErrorKind::RateLimited,
                retryable: true,
                code: Some("429".into()),
            }],
            "429 must map to a retryable RateLimited error"
        );
        let server = chat_respond_norms(500, r#"{"error":{"message":"boom"}}"#.into()).await;
        assert_eq!(
            server,
            vec![Norm::Err {
                kind: ProviderErrorKind::Server,
                retryable: true,
                code: Some("500".into()),
            }],
            "500 before the first token must map to a retryable Server error"
        );
    }

    #[tokio::test]
    async fn chat_socket_death_after_content_keeps_text() {
        // (g) The server flushes real content and then dies mid-chunked-body
        // (FIN without the terminal 0-chunk): the content chunk already
        // delivered stays delivered, then Err(Network) — never a clean Done.
        let (addr, _handle) = spawn_death_server(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial reply\"}}]}\n\n".into(),
        ])
        .await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(format!("http://{addr}"), None));
        let norms = drive(provider.stream(request("m"))).await;
        assert_eq!(
            norms,
            vec![
                Norm::Text("partial reply".into()),
                Norm::Err {
                    kind: ProviderErrorKind::Network,
                    retryable: true,
                    code: None,
                },
            ],
            "content delivered before the death must survive; the death is a Network error"
        );
    }

    #[tokio::test]
    async fn chat_multibyte_rune_split_across_sse_chunks() {
        // (h) "héllo" split between the two bytes of 'é' across HTTP chunks:
        // byte-level line buffering reassembles one correct Text chunk.
        let e = "é".as_bytes();
        let mut c1 = b"data: {\"choices\":[{\"delta\":{\"content\":\"h".to_vec();
        c1.push(e[0]);
        let mut c2 = vec![e[1]];
        c2.extend_from_slice(b"llo\"}}]}");
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![c1, c2, b"\n\n".to_vec(), b"data: [DONE]\n\n".to_vec()],
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let norms = drive(provider.stream(request("m"))).await;
        assert_eq!(
            norms,
            vec![Norm::Text("héllo".into()), Norm::Done],
            "a rune split mid-boundary must reassemble"
        );
    }

    // ----------------------------------------------------- openai responses

    fn responses_text_delta(t: &str) -> Value {
        json!({"type": "response.output_text.delta", "item_id": "m1", "delta": t})
    }

    fn responses_args_delta(item_id: &str, delta: &str) -> Value {
        json!({"type": "response.function_call_arguments.delta", "item_id": item_id, "delta": delta})
    }

    fn responses_item_added(call_id: &str, name: &str) -> Value {
        json!({"type": "response.output_item.added", "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": "",
        }})
    }

    fn responses_item_done(call_id: &str, name: &str, arguments: &str) -> Value {
        json!({"type": "response.output_item.done", "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }})
    }

    fn responses_completed() -> Value {
        json!({"type": "response.completed", "response": {"id": "r1"},
               "usage": {"input_tokens": 7, "output_tokens": 3}})
    }

    async fn responses_norms(events: Vec<String>) -> Vec<Norm> {
        let server = MockServer::new();
        server.route(
            "POST",
            "/responses",
            MockAction::Sse {
                status: 200,
                events,
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::responses(base, None));
        drive(provider.stream(request("m"))).await
    }

    async fn responses_respond_norms(status: u16, body: String) -> Vec<Norm> {
        let server = MockServer::new();
        server.route("POST", "/responses", MockAction::Respond { status, body });
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::responses(base, None));
        drive(provider.stream(request("m"))).await
    }

    #[tokio::test]
    async fn responses_duplicate_text_not_deduped_done_once() {
        // (a) Duplicate text deltas pass through as two Text chunks (no
        // adapter-level dedup); an EMPTY delta yields no chunk; the usage on
        // the completed event is ignored by this codec (no Usage chunk —
        // documented); exactly one Done.
        let norms = responses_norms(vec![
            sse_frame(&responses_text_delta("hi")),
            sse_frame(&responses_text_delta("")),
            sse_frame(&responses_text_delta("hi")),
            sse_frame(&responses_completed()),
            sse_done(),
        ])
        .await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Text("hi".into()), Norm::Done],
            "duplicates pass through; empty deltas vanish; usage is not a responses chunk"
        );
    }

    #[tokio::test]
    async fn responses_missing_completed_emits_done_at_eof() {
        // (b) Neither response.completed nor [DONE] arrives: the stream end
        // synthesizes Done exactly once.
        let norms = responses_norms(vec![
            sse_frame(&responses_text_delta("hi")),
            sse_frame(&responses_text_delta(" there")),
        ])
        .await;
        assert_eq!(
            norms,
            vec![
                Norm::Text("hi".into()),
                Norm::Text(" there".into()),
                Norm::Done
            ],
            "EOF emits Done exactly once for the responses codec too"
        );
    }

    #[tokio::test]
    async fn responses_double_done_emits_one_done() {
        // (c) A second [DONE] after the first is a no-op.
        let norms = responses_norms(vec![
            sse_frame(&responses_text_delta("hi")),
            sse_done(),
            sse_done(),
            sse_frame(&responses_text_delta("ignored")),
        ])
        .await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Done],
            "the first [DONE] ends the responses stream"
        );
    }

    #[tokio::test]
    async fn responses_tool_args_fragmented_into_one_complete_call() {
        // (d) function_call_arguments.delta fragmented across six events
        // reassembles into ONE complete ToolCall, terminated by
        // response.completed.
        let frags = [
            r#"{"path":"#,
            r#""a/"#,
            r#"b.rs","#,
            r#""mode":"#,
            r#""read""#,
            "}",
        ];
        let mut events = vec![sse_frame(&responses_item_added("fc_1", "read_file"))];
        for f in frags {
            events.push(sse_frame(&responses_args_delta("fc_1", f)));
        }
        events.push(sse_frame(&responses_completed()));
        let norms = responses_norms(events).await;
        assert_eq!(
            norms,
            vec![
                Norm::ToolCall {
                    id: "fc_1".into(),
                    name: "read_file".into(),
                    args: json!({"path": "a/b.rs", "mode": "read"}),
                    complete: true,
                },
                Norm::Done,
            ],
            "fragmented responses arguments reassemble into one complete call"
        );
    }

    #[tokio::test]
    async fn responses_eof_flush_of_pending_tool_call_no_longer_panics() {
        // DEFECT LOCK (openai responses codec): when the server dies without
        // any terminal marker while a tool call is still being accumulated,
        // the adapter flushes the completed call at EOF — then RE-POLLS the
        // already-finished line stream on the next poll, which panics inside
        // futures' unfold. A fixed adapter must return the flushed chunk and
        // then terminate (Stage::Done) without touching the dead line
        // stream; convert this test to
        // `[ToolCall { .. }, Done]` assertions when crates/openai is fixed.
        let mut events = vec![sse_frame(&responses_item_added("fc_1", "read_file"))];
        for f in [
            r#"{"path":"#,
            r#""a/"#,
            r#"b.rs","#,
            r#""mode":"#,
            r#""read""#,
            "}",
        ] {
            events.push(sse_frame(&responses_args_delta("fc_1", f)));
        }
        let _ = responses_norms(events).await;
    }

    #[tokio::test]
    async fn chat_eof_flush_of_pending_tool_call_no_longer_panics() {
        // DEFECT LOCK (openai chat codec): the identical hazard — a stream
        // that ends (no finish_reason, no [DONE]) while tool fragments are
        // accumulated flushes the call at EOF and then re-polls the finished
        // line stream on the next poll, panicking inside futures' unfold.
        // Convert to `[ToolCall { .. }, Done]` assertions when fixed.
        let events = vec![
            sse_frame(&chat_tool_frame(vec![chat_tc(
                0,
                "call_1",
                "read_file",
                r#"{"path":"#,
            )])),
            sse_frame(&chat_tool_frame(vec![chat_tc(0, "", "", r#""a.rs""#)])),
            sse_frame(&chat_tool_frame(vec![chat_tc(0, "", "", "}")])),
        ];
        let _ = chat_sse_norms(events).await;
    }

    #[tokio::test]
    async fn responses_out_of_order_done_resets_then_deltas_reaccumulate() {
        // (e) An output_item.done that arrives BEFORE the argument deltas
        // (out-of-order event) replaces the stored call with whatever it
        // carries — here empty arguments — and the later deltas accumulate
        // on top of it. Event-order processing, not sequence validation.
        let norms = responses_norms(vec![
            sse_frame(&responses_item_added("fc_1", "read_file")),
            sse_frame(&responses_item_done("fc_1", "read_file", "")),
            sse_frame(&responses_args_delta("fc_1", r#"{"x":"#)),
            sse_frame(&responses_args_delta("fc_1", "1}")),
            sse_frame(&responses_completed()),
        ])
        .await;
        assert_eq!(
            norms,
            vec![
                Norm::ToolCall {
                    id: "fc_1".into(),
                    name: "read_file".into(),
                    args: json!({"x": 1}),
                    complete: true,
                },
                Norm::Done,
            ],
            "an early done event resets accumulation; later deltas re-accumulate correctly"
        );
    }

    #[tokio::test]
    async fn responses_tool_ids_interleave_no_crosstalk_unknown_item_dropped() {
        // (e) Two call ids whose deltas interleave accumulate independently
        // (no cross-talk), and a delta referencing an item id that was never
        // added is dropped silently — it cannot inject fragments into any
        // stored call.
        let norms = responses_norms(vec![
            sse_frame(&responses_item_added("fc_a", "read_file")),
            sse_frame(&responses_item_added("fc_b", "sum")),
            sse_frame(&responses_args_delta("fc_a", r#"{"x":"#)),
            sse_frame(&responses_args_delta("ghost", r#"{"evil":1}"#)),
            sse_frame(&responses_args_delta("fc_b", r#"{"y":"#)),
            sse_frame(&responses_args_delta("fc_a", "1}")),
            sse_frame(&responses_args_delta("fc_b", "[2]}")),
            sse_frame(&responses_item_done("fc_a", "read_file", r#"{"x":1}"#)),
            sse_frame(&responses_item_done("fc_b", "sum", r#"{"y":[2]}"#)),
            sse_frame(&responses_completed()),
        ])
        .await;
        assert_eq!(
            norms,
            vec![
                Norm::ToolCall {
                    id: "fc_a".into(),
                    name: "read_file".into(),
                    args: json!({"x": 1}),
                    complete: true,
                },
                Norm::ToolCall {
                    id: "fc_b".into(),
                    name: "sum".into(),
                    args: json!({"y": [2]}),
                    complete: true,
                },
                Norm::Done,
            ],
            "per-item accumulation never cross-talks; unknown-item deltas are dropped"
        );
    }

    #[tokio::test]
    async fn responses_429_and_500_map_retryable_before_first_token() {
        let rate_limited =
            responses_respond_norms(429, r#"{"error":{"message":"slow down"}}"#.into()).await;
        assert_eq!(
            rate_limited,
            vec![Norm::Err {
                kind: ProviderErrorKind::RateLimited,
                retryable: true,
                code: Some("429".into()),
            }]
        );
        let server = responses_respond_norms(500, r#"{"error":{"message":"boom"}}"#.into()).await;
        assert_eq!(
            server,
            vec![Norm::Err {
                kind: ProviderErrorKind::Server,
                retryable: true,
                code: Some("500".into()),
            }]
        );
    }

    #[tokio::test]
    async fn responses_socket_death_after_content_keeps_text() {
        let (addr, _handle) = spawn_death_server(vec![
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"m1\",\"delta\":\"partial reply\"}\n\n".into(),
        ])
        .await;
        let provider =
            OpenAiProvider::build(OpenAiConfig::responses(format!("http://{addr}"), None));
        let norms = drive(provider.stream(request("m"))).await;
        assert_eq!(
            norms,
            vec![
                Norm::Text("partial reply".into()),
                Norm::Err {
                    kind: ProviderErrorKind::Network,
                    retryable: true,
                    code: None,
                },
            ]
        );
    }

    #[tokio::test]
    async fn responses_multibyte_rune_split_across_sse_chunks() {
        let e = "é".as_bytes();
        let mut c1 =
            b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"m1\",\"delta\":\"h"
                .to_vec();
        c1.push(e[0]);
        let mut c2 = vec![e[1]];
        c2.extend_from_slice(b"llo\"}\n\n");
        let server = MockServer::new();
        server.route(
            "POST",
            "/responses",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![c1, c2, b"data: [DONE]\n\n".to_vec()],
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::responses(base, None));
        let norms = drive(provider.stream(request("m"))).await;
        assert_eq!(
            norms,
            vec![Norm::Text("héllo".into()), Norm::Done],
            "a rune split mid-boundary must reassemble in the responses codec"
        );
    }

    // ------------------------------------------------------------- ollama

    /// One native NDJSON /api/chat frame.
    fn ollama_frame(v: &Value) -> String {
        format!("{v}\n")
    }

    async fn ollama_respond_norms(status: u16, body: String) -> Vec<Norm> {
        let server = MockServer::new();
        server.route("POST", "/api/chat", MockAction::Respond { status, body });
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        drive(provider.stream(request("qwen3.8"))).await
    }

    #[tokio::test]
    async fn ollama_duplicate_text_frames_not_deduped_done_once() {
        // (a) Two identical content frames: two Text chunks — the adapter
        // does NOT dedup native frames; done:true then ends the stream once.
        let body = format!(
            "{}{}{}",
            ollama_frame(
                &json!({"message": {"role": "assistant", "content": "hi"}, "done": false})
            ),
            ollama_frame(
                &json!({"message": {"role": "assistant", "content": "hi"}, "done": false})
            ),
            ollama_frame(&json!({"done": true})),
        );
        let norms = ollama_respond_norms(200, body).await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Text("hi".into()), Norm::Done],
            "duplicate native frames pass through verbatim"
        );
    }

    #[tokio::test]
    async fn ollama_missing_done_flag_emits_done_at_eof() {
        // (b) Frames stream without any done:true and the body simply ends:
        // Done is synthesized exactly once at EOF.
        let body = ollama_frame(
            &json!({"message": {"role": "assistant", "content": "hi"}, "done": false}),
        );
        let norms = ollama_respond_norms(200, body).await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Done],
            "EOF after a done:false frame must still terminate the stream once"
        );
    }

    #[tokio::test]
    async fn ollama_double_done_flag_emits_one_done() {
        // (c) A second done:true frame after the first terminal is never
        // read: exactly one Done.
        let body = format!(
            "{}{}{}",
            ollama_frame(
                &json!({"message": {"role": "assistant", "content": "hi"}, "done": false})
            ),
            ollama_frame(&json!({"done": true})),
            ollama_frame(
                &json!({"message": {"role": "assistant", "content": "ignored"}, "done": true})
            ),
        );
        let norms = ollama_respond_norms(200, body).await;
        assert_eq!(
            norms,
            vec![Norm::Text("hi".into()), Norm::Done],
            "the first done:true ends the stream; later frames are never read"
        );
    }

    #[tokio::test]
    async fn ollama_tool_call_split_across_frames_is_per_frame_atomic() {
        // (d/e) Native frames are ATOMIC: a tool call whose pieces arrive in
        // two frames (first name-only, then name+arguments) surfaces as TWO
        // independent complete ToolCall chunks — the adapter never
        // accumulates tool calls across native frames — and the synthesized
        // id restarts at the per-frame index, so both chunks share
        // "ollama:0:0". Documented actual behavior: frame order (text then
        // calls) is preserved and Done arrives exactly once.
        let body = format!(
            "{}{}{}{}",
            ollama_frame(&json!({
                "message": {"role": "assistant", "content": "using tools"},
                "done": false,
            })),
            ollama_frame(&json!({
                "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "read_file"}}
                ]},
                "done": false,
            })),
            ollama_frame(&json!({
                "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.rs"}}}
                ]},
                "done": false,
            })),
            ollama_frame(&json!({"done": true})),
        );
        let norms = ollama_respond_norms(200, body).await;
        assert_eq!(
            norms,
            vec![
                Norm::Text("using tools".into()),
                Norm::ToolCall {
                    id: "ollama:0:0".into(),
                    name: "read_file".into(),
                    args: Value::Null,
                    complete: true,
                },
                Norm::ToolCall {
                    id: "ollama:0:0".into(),
                    name: "read_file".into(),
                    args: json!({"path": "a.rs"}),
                    complete: true,
                },
                Norm::Done,
            ],
            "each native frame completes independently; ids restart per frame index"
        );
    }

    #[tokio::test]
    async fn ollama_eof_after_tool_call_frame_is_clean() {
        // POSITIVE CONTROL for the openai codecs' EOF-flush hazard: a native
        // stream that ends right after a tool-call frame (no done:true, no
        // further bytes) must deliver the call and then Done — the ollama
        // loop never re-polls a finished line stream.
        let body = ollama_frame(&json!({
            "message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "a.rs"}}}
            ]},
            "done": false,
        }));
        let norms = ollama_respond_norms(200, body).await;
        assert_eq!(
            norms,
            vec![
                Norm::ToolCall {
                    id: "ollama:0:0".into(),
                    name: "read_file".into(),
                    args: json!({"path": "a.rs"}),
                    complete: true,
                },
                Norm::Done,
            ],
            "EOF after a tool-call frame must flush the frame chunks, then terminate once"
        );
    }

    #[tokio::test]
    async fn ollama_429_and_500_map_retryable_before_first_token() {
        let rate_limited = ollama_respond_norms(429, r#"{"error":"rate limited"}"#.into()).await;
        assert_eq!(
            rate_limited,
            vec![Norm::Err {
                kind: ProviderErrorKind::RateLimited,
                retryable: true,
                code: Some("429".into()),
            }],
            "429 must map to a retryable RateLimited error"
        );
        let server = ollama_respond_norms(500, r#"{"error":"boom"}"#.into()).await;
        assert_eq!(
            server,
            vec![Norm::Err {
                kind: ProviderErrorKind::Server,
                retryable: true,
                code: Some("500".into()),
            }],
            "500 must map to a retryable Server error"
        );
    }

    #[tokio::test]
    async fn ollama_socket_death_after_content_keeps_text() {
        let (addr, _handle) = spawn_death_server(vec![
            "{\"message\":{\"role\":\"assistant\",\"content\":\"partial reply\"},\"done\":false}\n"
                .into(),
        ])
        .await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(format!("http://{addr}"))));
        let norms = drive(provider.stream(request("qwen3.8"))).await;
        assert_eq!(
            norms,
            vec![
                Norm::Text("partial reply".into()),
                Norm::Err {
                    kind: ProviderErrorKind::Network,
                    retryable: true,
                    code: None,
                },
            ],
            "NDJSON content before the death survives; the death is a Network error"
        );
    }

    #[tokio::test]
    async fn ollama_multibyte_rune_split_across_ndjson_chunks() {
        // (h) An NDJSON frame whose content carries "héllo" split between
        // the two bytes of 'é' across HTTP chunks must reassemble.
        let e = "é".as_bytes();
        let mut c1 = b"{\"message\":{\"role\":\"assistant\",\"content\":\"h".to_vec();
        c1.push(e[0]);
        let mut c2 = vec![e[1]];
        c2.extend_from_slice(b"llo\"},\"done\":false}\n");
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![c1, c2, b"{\"done\":true}\n".to_vec()],
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let norms = drive(provider.stream(request("qwen3.8"))).await;
        assert_eq!(
            norms,
            vec![Norm::Text("héllo".into()), Norm::Done],
            "a rune split mid-boundary must reassemble in NDJSON frames"
        );
    }

    // -------------------------------------------------------- death server

    /// Serve ONE request with a chunked SSE/NDJSON body whose frames are
    /// flushed and whose socket then dies WITHOUT the terminal 0-chunk — the
    /// honest reproduction of a socket death after content. The client must
    /// deliver the flushed content chunks, then surface Err(Network); it
    /// must never hang and never see a clean Done.
    async fn spawn_death_server(
        frames: Vec<String>,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            // Consume the request head + body (same discipline as the mock).
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let mut header_end = 0usize;
            while header_end == 0 {
                let Ok(n) = socket.read(&mut tmp).await else {
                    return;
                };
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end = pos + 4;
                }
                if buf.len() > 64 * 1024 {
                    return;
                }
            }
            let header = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let content_length = header
                .lines()
                .find_map(|l| {
                    l.to_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_length {
                let Ok(n) = socket.read(&mut tmp).await else {
                    return;
                };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let head = "HTTP/1.1 200 X\r\nConnection: close\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
            if socket.write_all(head.as_bytes()).await.is_err() {
                return;
            }
            for frame in &frames {
                let framed = format!("{:x}\r\n{}\r\n", frame.len(), frame);
                if socket.write_all(framed.as_bytes()).await.is_err() {
                    return;
                }
            }
            if socket.flush().await.is_err() {
                return;
            }
            // Give the client time to consume the flushed bytes, then die
            // mid-body: FIN without the terminal 0-chunk.
            tokio::time::sleep(Duration::from_millis(100)).await;
        });
        (addr, handle)
    }
}
