//! The OFFICIAL `agent-client-protocol` client drives the `faktor-acp`
//! server over the official NDJSON transport, unmodified: no transport
//! translation, no id remapping, no frame rewriting. Every scenario in this
//! file goes through the official SDK's connection, JSON-RPC layer and
//! typed schema; the harness only records the raw bytes each side wrote
//! (`common::WireTrace`) so conformance facts can be asserted directly.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_client_protocol::schema::v1::{
    AuthMethodId, AuthenticateRequest, ClientCapabilities, ContentBlock, FileSystemCapabilities,
    ImageContent, InitializeResponse, LoadSessionResponse, McpServer, McpServerStdio,
    NewSessionRequest, PromptRequest, PromptResponse, SessionNotification, SessionUpdate,
    StopReason, TextContent, ToolCallStatus,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{ConnectionTo, ErrorCode, UntypedMessage};
use faktor_acp::AcpServer;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use common::*;

/// Every scenario is bounded: a hang fails the test instead of blocking CI.
async fn run<F>(
    harness: &mut ServerHarness,
    log: &Arc<NotificationLog>,
    main: F,
) -> Result<(), String>
where
    F: AsyncFnOnce(
        ConnectionTo<agent_client_protocol::Agent>,
    ) -> Result<(), agent_client_protocol::Error>,
{
    let run = connect_official(harness, Arc::clone(log), main);
    tokio::time::timeout(WAIT, run)
        .await
        .map_err(|_| "official client run timed out".to_string())?
        .map_err(|error| format!("official client run failed: {error:?}"))
}

#[tokio::test]
async fn official_initialize_negotiates_v1_and_honest_capabilities() {
    let backend = FakeBackend::new(&["hi"]);
    let mut harness = ServerHarness::start(backend);
    let log = Arc::new(NotificationLog::default());
    let adapter = Arc::clone(&harness.trace);

    let capabilities = ClientCapabilities::new().fs(FileSystemCapabilities::new()
        .read_text_file(true)
        .write_text_file(false));

    run(&mut harness, &log, async move |cx| {
        let response: InitializeResponse = initialize(&cx, capabilities).await?;
        assert_eq!(response.protocol_version, ProtocolVersion::V1);
        assert!(!response.agent_capabilities.load_session);
        assert!(!response.agent_capabilities.prompt_capabilities.image);
        assert!(!response.agent_capabilities.prompt_capabilities.audio);
        assert!(
            !response
                .agent_capabilities
                .prompt_capabilities
                .embedded_context,
            "no content surface is advertised that this agent cannot honor"
        );
        assert!(response.auth_methods.is_empty());
        assert!(response.agent_info.is_none());

        // The typed request cannot declare extensions, so the server must not
        // echo any. (Raw declaration is covered separately.)
        let raw_init = adapter
            .server_frames()
            .into_iter()
            .filter(|frame| frame["result"]["protocolVersion"] == json!(1))
            .collect::<Vec<Value>>();
        assert_eq!(raw_init.len(), 1);
        assert!(raw_init[0]["result"].get("extensions").is_none());
        Ok(())
    })
    .await
    .expect("initialize negotiation");
}

#[tokio::test]
async fn official_raw_extension_declaration_echoes_only_accepted_names() {
    let backend = FakeBackend::new(&["hi"]);
    let mut harness = ServerHarness::start(backend);
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        // The official typed `InitializeRequest` has no `extensions` member
        // (only `_meta`); the declaration this server accepts is therefore
        // only reachable through the official connection's raw request
        // surface, still driven by the official client's JSON-RPC layer.
        let raw = UntypedMessage::new(
            "initialize",
            json!({
                "protocolVersion": 1,
                "extensions": ["faktor.agentStateChanged", "unknown.extension"],
            }),
        )?;
        let response = cx.send_request(raw).block_task().await?;
        assert_eq!(response["protocolVersion"], 1);
        assert_eq!(response["extensions"], json!(["faktor.agentStateChanged"]));

        // Malformed declaration: official invalid-params error, never a
        // silent ignore (extensions must be an array of strings).
        let malformed = UntypedMessage::new(
            "initialize",
            json!({"protocolVersion": 1, "extensions": "x"}),
        )?;
        let error = cx
            .send_request(malformed)
            .block_task()
            .await
            .expect_err("malformed extensions must be refused");
        assert_error_code(&error, ErrorCode::InvalidParams);
        assert_eq!(error.message, "\"extensions\" must be an array of strings");
        Ok(())
    })
    .await
    .expect("extension negotiation");
}

#[tokio::test]
async fn official_unsupported_protocol_version_is_a_typed_error() {
    let backend = FakeBackend::new(&["hi"]);
    let mut harness = ServerHarness::start(backend);
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        let raw = UntypedMessage::new("initialize", json!({"protocolVersion": 2}))?;
        let error = cx
            .send_request(raw)
            .block_task()
            .await
            .expect_err("protocol v2 must be refused");
        assert_error_code(&error, ErrorCode::InvalidParams);
        let data = error.data.expect("typed version error carries data");
        assert_eq!(data["protocolVersion"], 2);
        assert_eq!(data["supportedProtocolVersion"], 1);
        Ok(())
    })
    .await
    .expect("version negotiation");
}

#[tokio::test]
async fn official_prompt_streams_to_completion_with_typed_updates() {
    let backend = FakeBackend::new(&["alpha", "beta"]);
    let mut harness = ServerHarness::start(backend.clone());
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        let _ = initialize(&cx, ClientCapabilities::default()).await?;
        let session_id = new_session(&cx).await?;
        assert_eq!(session_id.to_string(), "sess-1");

        let response: PromptResponse =
            prompt(&cx, session_id.clone(), "hello official client").await?;
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        let meta = response.meta.expect("backend report rides _meta");
        assert_eq!(meta["turns"], 1);
        Ok(())
    })
    .await
    .expect("prompt streaming");

    let updates = log.update_params();
    let texts: Vec<String> = updates
        .iter()
        .filter_map(|params| text_of(&params["update"]))
        .collect();
    assert_eq!(texts, vec!["alpha".to_string(), "beta".to_string()]);

    // Every recorded update must parse through the OFFICIAL typed schema.
    for params in &updates {
        let typed = typed_update(params);
        assert_eq!(typed.session_id.to_string(), "sess-1");
        match typed.update {
            SessionUpdate::AgentMessageChunk(chunk) => match chunk.content {
                ContentBlock::Text(text) => assert!(texts.contains(&text.text)),
                other => panic!("unexpected typed chunk: {other:?}"),
            },
            other => panic!("unexpected typed update: {other:?}"),
        }
    }

    assert_notifications_omit_id(&harness);
    assert_ndjson_wire(&harness);
    assert_eq!(harness.backend.emitted(), vec!["alpha", "beta"]);
}

#[tokio::test]
async fn official_typed_client_cannot_declare_extension_and_server_gates_frames() {
    let backend = FakeBackend::new(&["plain"]).with_state_frames(true);
    let mut harness = ServerHarness::start(backend.clone());
    let log = Arc::new(NotificationLog::default());
    let log_main = Arc::clone(&log);

    run(&mut harness, &log, async move |cx| {
        let response = initialize(&cx, ClientCapabilities::default()).await?;
        assert_eq!(response.protocol_version, ProtocolVersion::V1);

        let session_id = new_session(&cx).await?;
        let response = prompt(&cx, session_id, "gated").await?;
        assert_eq!(response.stop_reason, StopReason::EndTurn);

        // First (typed) turn: no extension frame may reach the client.
        let gated = log_main.update_params();
        assert_eq!(gated.len(), 1);
        assert!(
            gated
                .iter()
                .all(|params| params["update"].get("kind").is_none()),
            "unnegotiated extension frames must be suppressed: {gated:?}"
        );

        // Same connection, now declaring the extension through the raw
        // surface: it is echoed, and extension frames become visible.
        let raw = UntypedMessage::new(
            "initialize",
            json!({
                "protocolVersion": 1,
                "extensions": ["faktor.agentStateChanged"],
            }),
        )?;
        let response = cx.send_request(raw).block_task().await?;
        assert_eq!(response["extensions"], json!(["faktor.agentStateChanged"]));

        let session_id = new_session(&cx).await?;
        let response = prompt(&cx, session_id, "declared").await?;
        assert_eq!(response.stop_reason, StopReason::EndTurn);

        // Second turn: busy + idle extension frames are delivered. The
        // official typed `SessionUpdate` schema has no variant for them, so
        // the raw collector sees them and the typed parse explicitly fails
        // (asserted, not skipped).
        let all = log_main.update_params();
        let extensions = all
            .iter()
            .filter(|params| params["update"].get("kind").is_some())
            .collect::<Vec<_>>();
        assert_eq!(extensions.len(), 2, "busy + idle frames: {extensions:?}");
        assert_eq!(extensions[0]["update"]["kind"], "agentStateChanged");
        assert_eq!(extensions[0]["update"]["agentState"]["status"], "busy");
        assert_eq!(extensions[1]["update"]["agentState"]["status"], "idle");
        assert!(
            serde_json::from_value::<SessionNotification>(extensions[0].clone()).is_err(),
            "the official typed schema cannot represent Faktor extension frames"
        );
        Ok(())
    })
    .await
    .expect("extension gating");

    assert_eq!(harness.backend.emitted(), vec!["plain", "plain"]);
}

#[tokio::test]
async fn official_cancel_while_streaming_yields_cancelled_stop_reason() {
    let backend = FakeBackend::new(&["chunk-a"]).with_hold_ms(5_000);
    let mut harness = ServerHarness::start(backend.clone());
    let log = Arc::new(NotificationLog::default());
    let log_main = Arc::clone(&log);

    run(&mut harness, &log, async move |cx| {
        let _ = initialize(&cx, ClientCapabilities::default()).await?;
        let session_id = new_session(&cx).await?;

        let request = cx.send_request(PromptRequest::new(
            session_id.clone(),
            vec![ContentBlock::Text(TextContent::new("hold"))],
        ));
        log_main.await_first_update().await;
        cancel(&cx, session_id).await?;

        let response: PromptResponse = request.block_task().await?;
        assert_eq!(response.stop_reason, StopReason::Cancelled);
        Ok(())
    })
    .await
    .expect("cancel while streaming");

    // The turn really was mid-stream (first chunk delivered) and the server
    // refused further frames after cancellation was observed.
    assert!(harness.backend.started());
    assert!(!harness.backend.completed());
    assert_eq!(harness.backend.post_cancel_emit_attempts(), 1);
    let texts: Vec<String> = log
        .update_params()
        .iter()
        .filter_map(|params| text_of(&params["update"]))
        .collect();
    assert_eq!(texts, vec!["chunk-a".to_string()]);
    assert!(!texts.iter().any(|text| text == "post-cancel"));
}

#[tokio::test]
async fn official_session_load_replays_history_when_capability_is_on() {
    let backend = FakeBackend::new(&["unused"]).with_load_page(history_page("sess-1"));
    let mut harness = ServerHarness::start(backend.clone());
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        let response = initialize(&cx, ClientCapabilities::default()).await?;
        assert!(response.agent_capabilities.load_session);

        let session_id = new_session(&cx).await?;
        let response = load_session(&cx, session_id).await?;
        assert_eq!(response, LoadSessionResponse::default());
        Ok(())
    })
    .await
    .expect("session load");

    let updates = log.update_params();
    let kinds: Vec<String> = updates
        .iter()
        .map(|params| {
            params["update"]["sessionUpdate"]
                .as_str()
                .expect("sessionUpdate discriminator")
                .to_string()
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            "user_message_chunk",
            "agent_thought_chunk",
            "tool_call",
            "agent_message_chunk"
        ]
    );

    // Typed official parse of the replay, in order.
    match typed_update(&updates[0]).update {
        SessionUpdate::UserMessageChunk(chunk) => match chunk.content {
            ContentBlock::Text(text) => assert_eq!(text.text, "question"),
            other => panic!("unexpected user chunk: {other:?}"),
        },
        other => panic!("unexpected replay frame: {other:?}"),
    }
    match typed_update(&updates[1]).update {
        SessionUpdate::AgentThoughtChunk(chunk) => match chunk.content {
            ContentBlock::Text(text) => assert_eq!(text.text, "thought"),
            other => panic!("unexpected thought chunk: {other:?}"),
        },
        other => panic!("unexpected replay frame: {other:?}"),
    }
    match typed_update(&updates[2]).update {
        SessionUpdate::ToolCall(tool_call) => {
            assert_eq!(tool_call.tool_call_id.0.as_ref(), "call-1");
            assert_eq!(tool_call.title, "echo");
            assert_eq!(tool_call.status, ToolCallStatus::InProgress);
        }
        other => panic!("unexpected replay frame: {other:?}"),
    }
    match typed_update(&updates[3]).update {
        SessionUpdate::AgentMessageChunk(chunk) => match chunk.content {
            ContentBlock::Text(text) => assert_eq!(text.text, "answer"),
            other => panic!("unexpected answer chunk: {other:?}"),
        },
        other => panic!("unexpected replay frame: {other:?}"),
    }

    assert_notifications_omit_id(&harness);
    assert_ndjson_wire(&harness);
}

#[tokio::test]
async fn official_load_without_capability_surfaces_method_not_found() {
    let backend = FakeBackend::new(&["unused"]);
    let mut harness = ServerHarness::start(backend);
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        let response = initialize(&cx, ClientCapabilities::default()).await?;
        assert!(
            !response.agent_capabilities.load_session,
            "backend hook is off, loadSession must be false"
        );

        // The official SDK does not gate callers on the advertised
        // capability; it surfaces the server's official error. The error
        // must therefore be the loud typed refusal, never an empty replay.
        let session_id = new_session(&cx).await?;
        let error = load_session(&cx, session_id)
            .await
            .expect_err("session/load must be refused when loadSession=false");
        assert_error_code(&error, ErrorCode::MethodNotFound);
        assert_eq!(error.message, "Method not found");
        Ok(())
    })
    .await
    .expect("load capability gate");
}

#[tokio::test]
async fn official_bad_params_and_unknown_methods_are_typed_official_errors() {
    let backend = FakeBackend::new(&["ok"]);
    let mut harness = ServerHarness::start(backend);
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        let _ = initialize(&cx, ClientCapabilities::default()).await?;
        let session_id = new_session(&cx).await?;

        // Empty prompt block list.
        let error = cx
            .send_request(PromptRequest::new(session_id.clone(), vec![]))
            .block_task()
            .await
            .expect_err("empty prompt must be refused");
        assert_error_code(&error, ErrorCode::InvalidParams);
        assert_eq!(
            error.message,
            "\"prompt\" must contain at least one content block"
        );

        // Non-text prompt content (image is not advertised).
        let error = cx
            .send_request(PromptRequest::new(
                session_id.clone(),
                vec![ContentBlock::Image(ImageContent::new("aGk=", "image/png"))],
            ))
            .block_task()
            .await
            .expect_err("image prompt must be refused");
        assert_error_code(&error, ErrorCode::InvalidParams);
        assert!(error.message.contains("unsupported content block type"));

        // MCP servers are refused while the backend advertises no MCP.
        let error = cx
            .send_request(
                NewSessionRequest::new("/work").mcp_servers(vec![McpServer::Stdio(
                    McpServerStdio::new("mcp", "/bin/false"),
                )]),
            )
            .block_task()
            .await
            .expect_err("mcp servers must be refused");
        assert_error_code(&error, ErrorCode::InvalidParams);
        assert_eq!(error.message, "this agent does not support MCP servers");

        // Wrong-typed params only reachable through the raw surface.
        let raw = UntypedMessage::new("session/new", json!({"cwd": "/work", "mcpServers": 5}))?;
        let error = cx
            .send_request(raw)
            .block_task()
            .await
            .expect_err("bad mcpServers type must be refused");
        assert_error_code(&error, ErrorCode::InvalidParams);
        assert_eq!(error.message, "\"mcpServers\" must be an array");

        // authenticate is a typed refusal while authMethods is empty.
        let error = cx
            .send_request(AuthenticateRequest::new(AuthMethodId::new("api-key")))
            .block_task()
            .await
            .expect_err("authenticate must be refused");
        assert_error_code(&error, ErrorCode::InvalidParams);
        assert_eq!(error.message, "no authentication methods are available");
        assert_eq!(
            error.data.expect("refusal carries the method id")["methodId"],
            "api-key"
        );

        // Unknown method: the official method-not-found error.
        let raw = UntypedMessage::new(
            "session/terminate",
            json!({ "sessionId": session_id.to_string() }),
        )?;
        let error = cx
            .send_request(raw)
            .block_task()
            .await
            .expect_err("unknown method must be refused");
        assert_error_code(&error, ErrorCode::MethodNotFound);
        assert_eq!(error.message, "Method not found");
        Ok(())
    })
    .await
    .expect("official error surface");
}

#[tokio::test]
async fn official_connection_close_ends_the_server_cleanly() {
    let backend = FakeBackend::new(&["unused"]);
    let mut harness = ServerHarness::start(backend);
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        let _ = initialize(&cx, ClientCapabilities::default()).await?;
        let _ = new_session(&cx).await?;
        // Main returns; the official client drops the transport. The server
        // must observe EOF and wind down.
        Ok(())
    })
    .await
    .expect("client session");

    let server_task = harness.server_task;
    let outcome = tokio::time::timeout(WAIT, server_task)
        .await
        .expect("server must stop after client close")
        .expect("server task joins");
    assert_eq!(outcome, Ok(()), "clean EOF is not a server error");
}

/// The official SDK allocates string/UUID request ids (JSON-RPC 2.0 allows
/// them). With the harness shims deleted, every request the SDK sends over
/// the wire must round-trip verbatim: the server echoes the UUID, never a
/// remapped integer, and never answers `-32600` with a null id.
#[tokio::test]
async fn official_uuid_request_ids_round_trip_verbatim() {
    let backend = FakeBackend::new(&["alpha"]);
    let mut harness = ServerHarness::start(backend);
    let log = Arc::new(NotificationLog::default());

    run(&mut harness, &log, async move |cx| {
        let _ = initialize(&cx, ClientCapabilities::default()).await?;
        let session_id = new_session(&cx).await?;
        let response = prompt(&cx, session_id, "uuid round trip").await?;
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        Ok(())
    })
    .await
    .expect("uuid request ids");

    let client_ids: Vec<Value> = harness
        .trace
        .client_frames()
        .into_iter()
        .filter(|frame| frame.get("method").is_some())
        .filter_map(|frame| frame.get("id").cloned())
        .collect();
    assert!(
        !client_ids.is_empty(),
        "the official client must have sent requests"
    );
    for id in &client_ids {
        assert!(
            id.is_string(),
            "official SDK request ids are UUID strings, got {id}"
        );
    }
    let responses: Vec<Value> = harness
        .trace
        .server_frames()
        .into_iter()
        .filter(|frame| {
            frame.get("method").is_none()
                && (frame.get("result").is_some() || frame.get("error").is_some())
        })
        .collect();
    assert_eq!(
        responses.len(),
        client_ids.len(),
        "one response per request, no remap: {responses:?}"
    );
    for response in &responses {
        let id = response.get("id").expect("response id");
        assert!(
            client_ids.contains(id),
            "server echoed {id}, which matches no client request id"
        );
    }
    assert_notifications_omit_id(&harness);
    assert_ndjson_wire(&harness);
}

/// Raw NDJSON probe without any SDK help: a string id must be accepted and
/// echoed byte-for-byte, and the reply must be one NDJSON line (the official
/// transport), never a Content-Length frame.
#[tokio::test]
async fn raw_ndjson_string_id_is_accepted_and_echoed() {
    let (mut client_io, server_io) = tokio::io::duplex(PIPE);
    let server = AcpServer::new_streaming(FakeBackend::new(&["x"]));
    let task = tokio::spawn(async move {
        let (reader, writer) = tokio::io::split(server_io);
        server.serve_connection(reader, writer).await
    });

    let uuid = "e70f649f-bb05-42b2-9b08-380299012ea8";
    let frame = faktor_acp::protocol::encode_line(&json!({
        "jsonrpc": "2.0",
        "id": uuid,
        "method": "initialize",
        "params": { "protocolVersion": 1 },
    }))
    .expect("frame encodes");
    client_io.write_all(&frame).await.expect("write");

    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let (consumed, response) = loop {
        if let Ok(Some(found)) = faktor_acp::protocol::parse_ndjson(&buf) {
            break found;
        }
        let n = client_io.read(&mut chunk).await.expect("read");
        assert!(
            n > 0,
            "server closed without answering the string-id request"
        );
        buf.extend_from_slice(&chunk[..n]);
    };
    assert_eq!(buf.first(), Some(&b'{'), "NDJSON reply expected");
    assert_eq!(
        buf[consumed - 1],
        b'\n',
        "NDJSON line must be newline-terminated"
    );
    assert_eq!(response["id"], uuid, "string id must be echoed verbatim");
    assert_eq!(response["result"]["protocolVersion"], 1);

    drop(client_io);
    let _ = tokio::time::timeout(WAIT, task).await;
}

/// The legacy Content-Length path stays served for frozen pre-conformance
/// peers: a peer that opens with a Content-Length header is answered with
/// Content-Length framing (dual-mode read autodetect, per-connection mode).
#[tokio::test]
async fn legacy_content_length_peer_is_still_served() {
    let (mut client_io, server_io) = tokio::io::duplex(PIPE);
    let server = AcpServer::new_streaming(FakeBackend::new(&["x"]));
    let task = tokio::spawn(async move {
        let (reader, writer) = tokio::io::split(server_io);
        server.serve_connection(reader, writer).await
    });

    let frame = faktor_acp::protocol::encode(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": { "protocolVersion": 1 },
    }))
    .expect("frame encodes");
    client_io.write_all(&frame).await.expect("write");

    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let response = loop {
        if let Ok(Some((_consumed, value))) = faktor_acp::protocol::parse_frame(&buf) {
            break value;
        }
        let n = client_io.read(&mut chunk).await.expect("read");
        assert!(n > 0, "server closed without answering the legacy request");
        buf.extend_from_slice(&chunk[..n]);
    };
    assert!(
        buf.starts_with(b"Content-Length: "),
        "legacy peer must stay on Content-Length framing: {:?}",
        String::from_utf8_lossy(&buf)
    );
    assert_eq!(response["id"], 1);
    assert_eq!(response["result"]["protocolVersion"], 1);

    drop(client_io);
    let _ = tokio::time::timeout(WAIT, task).await;
}

/// The official SDK cancels a dropped request with a `$/cancel_request`
/// notification (ACP request-level cancellation), NOT with
/// `session/cancel`. The matching running prompt must be cancelled with
/// exactly one terminal `stopReason: cancelled` response, and the turn must
/// not complete.
#[tokio::test]
async fn official_dropped_prompt_is_cancelled_by_request_level_cancel() {
    let backend = FakeBackend::new(&["only-chunk"]).with_hold_ms(5_000);
    let mut harness = ServerHarness::start(backend.clone());
    let log = Arc::new(NotificationLog::default());
    let log_main = Arc::clone(&log);
    let trace = Arc::clone(&harness.trace);
    let trace_main = Arc::clone(&trace);

    run(&mut harness, &log, async move |cx| {
        let _ = initialize(&cx, ClientCapabilities::default()).await?;
        let session_id = new_session(&cx).await?;

        let request = cx.send_request(PromptRequest::new(
            session_id,
            vec![ContentBlock::Text(TextContent::new("drop me"))],
        ));
        log_main.await_first_update().await;

        // Dropping selects the SDK's request-level cancellation.
        drop(request);

        // Wait until the server wrote the cancelled terminal, so the raw
        // trace assertions below are races-free.
        let deadline = Instant::now() + WAIT;
        loop {
            let terminal = trace_main.server_frames().iter().any(|frame| {
                frame
                    .get("result")
                    .and_then(|result| result.get("stopReason"))
                    == Some(&json!("cancelled"))
            });
            if backend.post_cancel_emit_attempts() > 0 && terminal {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "turn neither cancelled nor completed"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok(())
    })
    .await
    .expect("dropped request cancellation");

    let methods = harness.trace.client_methods();
    assert!(
        methods.iter().any(|method| method == "$/cancel_request"),
        "the official SDK must send $/cancel_request for a dropped request: {methods:?}"
    );
    assert!(
        !methods.iter().any(|method| method == "session/cancel"),
        "request-level cancellation must not be replaced by session/cancel"
    );
    assert!(
        !harness.backend.completed(),
        "a cancelled turn must not complete"
    );
    assert_eq!(
        harness.backend.post_cancel_emit_attempts(),
        1,
        "no frame may be emitted after cancellation"
    );

    // Exactly one terminal response for the dropped prompt id.
    let prompt_id = harness
        .trace
        .client_frames()
        .into_iter()
        .find(|frame| frame.get("method").and_then(Value::as_str) == Some("session/prompt"))
        .and_then(|frame| frame.get("id").cloned())
        .expect("the dropped prompt request id");
    assert!(
        prompt_id.is_string(),
        "official prompt id is a UUID string: {prompt_id}"
    );
    let terminals: Vec<Value> = harness
        .trace
        .server_frames()
        .into_iter()
        .filter(|frame| {
            frame.get("method").is_none()
                && (frame.get("result").is_some() || frame.get("error").is_some())
                && frame.get("id") == Some(&prompt_id)
        })
        .collect();
    assert_eq!(terminals.len(), 1, "exactly one terminal: {terminals:?}");
    assert_eq!(terminals[0]["result"]["stopReason"], "cancelled");
}
