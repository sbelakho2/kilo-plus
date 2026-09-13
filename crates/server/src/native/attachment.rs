//! Native binary-attachment surface (additive, strict): upload ONE
//! `data_base64` payload into the session's durable CAS-backed attachment
//! store, resolve it by digest (metadata and bytes), and the ONE wire
//! admission rule every task-start DTO uses.
//!
//! - Bytes are validated (mime/filename/size bounds), written to the CAS,
//!   and persisted as a typed `AttachmentId { digest, mime, filename, size }`
//!   row BEFORE any task admission; a repeated upload of identical bytes is
//!   a dedupe hit returning the FIRST durable row byte-identically.
//! - The upload body is bounded by [`MAX_ATTACHMENT_UPLOAD_BYTES`] so its
//!   base64 form plus the JSON envelope stays under the daemon's 10 MiB
//!   request cap (`crate::api::MAX_BODY_BYTES`). The session/CAS ceiling
//!   remains `MAX_ATTACHMENT_BYTES` for programmatic callers.
//! - IMAGES ARE REFUSED LOUDLY (code `unsupported`, 400): provider
//!   media/content parts are NOT wired. The provider layer has only a
//!   URL-shaped `ContentKind::Image { url }` whose adapter encodings
//!   disagree (OpenAI takes a data URL, Anthropic emits a `type:"url"`
//!   source its API rejects, Google hardcodes `image/png` and expects raw
//!   base64) and the agent has no path from a durable row to a request
//!   part. The follow-up is precise: (1) add a binary part carrying
//!   `{mime, bytes}`, (2) encode it per adapter wire, (3) gate on
//!   `ModelCapabilities::vision`, (4) replace this refusal with delivery.
//!   Until then the refusal keeps the client draft/images intact instead of
//!   pretending the model saw them.
//! - Hostile DTOs (unknown fields, non-string members, malformed base64,
//!   traversal filenames, hostile mimes, oversized payloads) are typed
//!   400/413s; an unknown digest resolves to a typed 404, never a phantom.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use faktor_core::attachment::{validate_mime, AttachmentId, MAX_ATTACHMENT_BYTES};
use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;
use faktor_protocol::error::ApiError;

use super::*;
use crate::api::AppState;

/// Decoded-byte ceiling of one HTTP upload. Base64 inflates by 4/3, so
/// 7 MiB decodes to ~9.33 MiB encoded — with the JSON envelope this stays
/// under the daemon's 10 MiB `MAX_BODY_BYTES` cap. Larger payloads are a
/// typed 413 before any decode; the session/CAS ceiling
/// (`MAX_ATTACHMENT_BYTES`) is unchanged for programmatic callers.
pub(crate) const MAX_ATTACHMENT_UPLOAD_BYTES: usize = 7 * 1024 * 1024;

/// Strict request DTO of one native attachment upload. `data_base64` is the
/// standard-alphabet base64 of the raw bytes; unknown members, missing
/// members and non-string values are plain 400s (`deny_unknown_fields`).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeAttachmentUpload {
    mime: String,
    #[serde(default)]
    filename: Option<String>,
    data_base64: String,
}

/// The typed wire refusal for image submission while provider
/// media/content parts are not wired. Code `unsupported` so clients can
/// distinguish it from a malformed body and restore the draft loudly.
fn media_unsupported(mime: &str) -> ApiError {
    ApiError {
        code: "unsupported",
        message: format!(
            "image attachment ({mime}) cannot be submitted: provider media/content parts are not wired, so the bytes can never reach a model; the draft and attachments were kept — retry without the image or wait for the media-part follow-up"
        ),
        http_status: 400,
        retryable: false,
    }
}

/// Map one session-layer attachment refusal onto the wire: the DTO
/// admission class is a client 400 (oversized keeps its distinct code),
/// never a 5xx.
fn admission_error(e: &Error) -> ApiError {
    let code = match e.kind {
        ErrorKind::Oversized => "oversized",
        _ => "malformed",
    };
    ApiError {
        code,
        message: e.message.clone(),
        http_status: 400,
        retryable: false,
    }
}

/// The ONE wire admission rule for a task's binary attachment set: bounded
/// count/structural validity/durable byte-identical resolution via the
/// session layer, plus the loud image refusal. Runs BEFORE any run/task row
/// so a refused start leaves no partial durable admission.
pub(crate) fn validate_wire_attachments(
    handle: &faktor_session::SessionHandle,
    ids: &[AttachmentId],
) -> Result<(), ApiError> {
    for id in ids {
        if id.is_image() {
            return Err(media_unsupported(&id.mime));
        }
    }
    handle
        .resolve_attachments(ids)
        .map_err(|e| admission_error(&e))
}

/// Parse one digest path segment strictly: 64 lowercase/uppercase hex chars.
fn parse_digest(raw: &str) -> Result<FileHash, ApiError> {
    FileHash::from_hex(raw).ok_or_else(|| ApiError {
        code: "malformed",
        message: format!("{raw:?} is not a 64-char hex BLAKE3 digest"),
        http_status: 400,
        retryable: false,
    })
}

/// `POST /native/session/{id}/attachments` — upload ONE bounded attachment.
/// Returns the durable typed [`AttachmentId`] (same bytes → same id).
pub(crate) async fn native_attachment_upload(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Result<Json<NativeAttachmentUpload>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(upload) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native attachment upload body")),
    };
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    // Canonicalize exactly like the store write: a hostile or non-canonical
    // mime is a typed 400 before any decode/CAS write; an image is the loud
    // unsupported refusal.
    let mime = upload.mime.trim().to_ascii_lowercase();
    if let Err(e) = validate_mime(&mime) {
        return wire_status(admission_error(&e));
    }
    if mime.starts_with("image/") {
        return wire_status(media_unsupported(&mime));
    }
    if let Some(name) = upload.filename.as_deref() {
        if let Err(e) = faktor_core::attachment::validate_filename(name) {
            return wire_status(admission_error(&e));
        }
    }
    // Decode with an explicit ceiling BEFORE materializing: the base64 form
    // is pre-checked (4/3 + padding) and the decoded length is checked
    // twice (bounded form + absolute attachment ceiling).
    let encoded = upload.data_base64.as_bytes();
    let max_encoded = MAX_ATTACHMENT_UPLOAD_BYTES.div_ceil(3) * 4 + 4;
    if encoded.len() > max_encoded {
        return wire_status(ApiError {
            code: "oversized",
            message: format!(
                "attachment upload of {} base64 bytes exceeds the {} byte bound",
                encoded.len(),
                max_encoded
            ),
            http_status: 413,
            retryable: false,
        });
    }
    let compact: Vec<u8> = encoded
        .iter()
        .copied()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    let bytes = match base64::engine::general_purpose::STANDARD.decode(&compact) {
        Ok(b) => b,
        Err(e) => {
            return wire_status(ApiError {
                code: "malformed",
                message: format!("attachment data_base64 is not valid base64: {e}"),
                http_status: 400,
                retryable: false,
            })
        }
    };
    if bytes.len() > MAX_ATTACHMENT_UPLOAD_BYTES || bytes.len() as u64 > MAX_ATTACHMENT_BYTES {
        return wire_status(ApiError {
            code: "oversized",
            message: format!(
                "attachment of {} bytes exceeds the upload bound ({MAX_ATTACHMENT_UPLOAD_BYTES})",
                bytes.len()
            ),
            http_status: 413,
            retryable: false,
        });
    }
    match handle.put_attachment(&mime, upload.filename.as_deref(), &bytes) {
        Ok(stored) => {
            Json(serde_json::to_value(&stored).unwrap_or(serde_json::Value::Null)).into_response()
        }
        Err(e) => api_err(&e),
    }
}

/// The metadata projection of one resolved attachment (no bytes).
fn attachment_meta(id: &AttachmentId) -> serde_json::Value {
    serde_json::json!({
        "digest": id.digest.to_hex(),
        "mime": id.mime,
        "filename": id.filename,
        "size": id.size,
    })
}

/// `GET /native/session/{id}/attachments/{digest}` — resolve ONE durable
/// attachment row by digest (restart-safe). Unknown digests are typed 404s.
pub(crate) async fn native_attachment_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, digest)): Path<(String, String)>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let hash = match parse_digest(&digest) {
        Ok(h) => h,
        Err(e) => return wire_status(e),
    };
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match handle.attachment(hash) {
        Ok(Some(stored)) => Json(attachment_meta(&stored)).into_response(),
        Ok(None) => wire_status(not_found(&format!(
            "attachment {digest} in session {}",
            handle.id()
        ))),
        Err(e) => api_err(&e),
    }
}

/// `GET /native/session/{id}/attachments/{digest}/bytes` — the verified
/// bytes of ONE durable attachment (CAS re-hash; corruption is loud).
pub(crate) async fn native_attachment_bytes(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, digest)): Path<(String, String)>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let hash = match parse_digest(&digest) {
        Ok(h) => h,
        Err(e) => return wire_status(e),
    };
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let stored = match handle.attachment(hash) {
        Ok(Some(stored)) => stored,
        Ok(None) => {
            return wire_status(not_found(&format!(
                "attachment {digest} in session {}",
                handle.id()
            )))
        }
        Err(e) => return api_err(&e),
    };
    if stored.is_image() {
        // Defensive symmetry with the upload refusal: an image row can only
        // exist if written programmatically; delivering its bytes over the
        // wire would imply the media path exists.
        return wire_status(media_unsupported(&stored.mime));
    }
    match handle.attachment_bytes(&stored, MAX_ATTACHMENT_UPLOAD_BYTES) {
        Ok(bytes) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, stored.mime.as_str())],
            bytes,
        )
            .into_response(),
        Err(e) => api_err(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ServerDeps;
    use faktor_agent::{AgentDeps, AgentRuntime, NoEvidence, ToolCallMode, ToolRegistry};
    use faktor_core::model::ModelCapabilities;
    use faktor_core::time::SystemClock;
    use faktor_provider::{FakeProvider, ProviderRegistry};
    use faktor_session::SessionManager;
    use std::sync::Arc;

    fn test_state(root: &std::path::Path) -> (AppState, faktor_session::SessionHandle) {
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions =
            crate::permission::ChannelPermissionRequester::new(std::time::Duration::from_secs(5));
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::new(
                "fake",
                ModelCapabilities::default(),
            )))
            .unwrap();
        let mut deps = ServerDeps::new(
            session.clone(),
            {
                AgentRuntime::new(AgentDeps {
                    session: session.clone(),
                    providers: Arc::new(registry),
                    chunk_sink: None,
                    permission_requester: permissions.clone(),
                    evidence: Arc::new(NoEvidence),
                    tools: Arc::new(ToolRegistry::new()),
                    cas: None,
                    workspaces: faktor_fs::WorkspaceFileService::new(),
                    edit: None,
                    snapshots: None,
                    sandbox: None,
                    supervisor: None,
                    verification: faktor_agent::VerificationService::disabled(),
                    model: "m".into(),
                    compaction_model: None,
                    compact_at_usage: 0.65,
                    instructions: "i".into(),
                    hooks: None,
                    instructions_resolver: faktor_instructions::no_roots_resolver(),
                    routing: faktor_agent::FixedRoutingPolicy::passthrough(),
                    budgets: Arc::new(faktor_session::NoopBudget),
                    clock: Arc::new(SystemClock),
                    tool_call_mode: ToolCallMode::Native,
                    tool_deadline_ms: 1000,
                    retry_policy: faktor_core::retry::RetryPolicy::default(),
                    semantic: faktor_agent::fallback_semantic_registry(),
                    context_prior: None,
                    efficiency: Default::default(),
                })
                .unwrap()
            },
            permissions,
        );
        deps.directory = Some(root.to_string_lossy().into_owned());
        let ws = session.create_workspace(root.to_str().unwrap()).unwrap();
        let created = session
            .create_session(ws, "attachment-test", "fake", "m")
            .unwrap();
        let handle = session.get_session(created.id()).unwrap().unwrap();
        let state = AppState {
            bus: Arc::new(crate::global::GlobalEventBus::new(session.clone(), None)),
            deps: Arc::new(deps),
            config: Arc::new(std::sync::RwLock::new(serde_json::Value::Object(
                Default::default(),
            ))),
            auth: Arc::new(std::sync::RwLock::new(None)),
            ptys: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            next_pty_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            terminal_owners: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            terminal_events: Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())),
            next_terminal_event_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };
        (state, handle)
    }

    fn auth_headers(state: &AppState) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-faktor-server-password",
            state
                .deps
                .server_password
                .as_str()
                .parse::<axum::http::HeaderValue>()
                .unwrap(),
        );
        headers
    }

    fn upload(mime: &str, filename: Option<&str>, bytes: &[u8]) -> NativeAttachmentUpload {
        NativeAttachmentUpload {
            mime: mime.into(),
            filename: filename.map(str::to_string),
            data_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        }
    }

    #[tokio::test]
    async fn upload_resolve_and_dedupe_roundtrip_over_the_wire() {
        let dir = tempfile::tempdir().unwrap();
        let (state, handle) = test_state(dir.path());
        let sid = handle.id().to_string();
        let headers = auth_headers(&state);
        let first = native_attachment_upload(
            State(state.clone()),
            headers.clone(),
            Path(sid.clone()),
            Ok(Json(upload(
                "application/pdf",
                Some("spec.pdf"),
                b"%PDF-1.4",
            ))),
        )
        .await;
        assert_eq!(first.status(), StatusCode::OK);
        let body = axum::body::to_bytes(first.into_body(), 1 << 20)
            .await
            .unwrap();
        let id: AttachmentId = serde_json::from_slice(&body).unwrap();
        assert_eq!(id.mime, "application/pdf");
        assert_eq!(id.size, 8);
        // Dedupe by digest: identical bytes return the byte-identical row.
        let again = native_attachment_upload(
            State(state.clone()),
            headers.clone(),
            Path(sid.clone()),
            Ok(Json(upload(
                "application/pdf",
                Some("other.pdf"),
                b"%PDF-1.4",
            ))),
        )
        .await;
        let body = axum::body::to_bytes(again.into_body(), 1 << 20)
            .await
            .unwrap();
        let deduped: AttachmentId = serde_json::from_slice(&body).unwrap();
        assert_eq!(deduped, id);
        // Resolve metadata and bytes by digest.
        let meta = native_attachment_get(
            State(state.clone()),
            headers.clone(),
            Path((sid.clone(), id.digest.to_hex())),
        )
        .await;
        assert_eq!(meta.status(), StatusCode::OK);
        let bytes = native_attachment_bytes(
            State(state.clone()),
            headers.clone(),
            Path((sid.clone(), id.digest.to_hex())),
        )
        .await;
        assert_eq!(bytes.status(), StatusCode::OK);
        let body = axum::body::to_bytes(bytes.into_body(), 1 << 20)
            .await
            .unwrap();
        assert_eq!(&body[..], b"%PDF-1.4");
        // Unknown digest: typed 404, never a phantom.
        let missing = native_attachment_get(
            State(state.clone()),
            headers.clone(),
            Path((sid.clone(), "0".repeat(64))),
        )
        .await;
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn hostile_uploads_are_typed_refusals() {
        let dir = tempfile::tempdir().unwrap();
        let (state, handle) = test_state(dir.path());
        let sid = handle.id().to_string();
        let headers = auth_headers(&state);
        let expect_status = |response: Response, status: StatusCode| async move {
            assert_eq!(response.status(), status);
            axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap()
        };
        // An image is the LOUD unsupported refusal (media parts not wired).
        let image = native_attachment_upload(
            State(state.clone()),
            headers.clone(),
            Path(sid.clone()),
            Ok(Json(upload("image/png", Some("shot.png"), b"\x89PNG"))),
        )
        .await;
        let body = expect_status(image, StatusCode::BAD_REQUEST).await;
        let err: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(err["error"]["code"], "unsupported");
        assert!(
            err["error"]["message"]
                .as_str()
                .unwrap()
                .contains("provider media"),
            "{err}"
        );
        // Hostile mime, traversal filename and malformed base64 are 400s.
        // (Uppercase is NOT hostile: it is canonicalized to lowercase before
        // validation, exactly like the session write path.)
        for hostile in [
            upload("", None, b"x"),
            upload("text", None, b"x"),
            upload("text/", None, b"x"),
            upload("text/plain/extra", None, b"x"),
            upload("bad mime/type", None, b"x"),
        ] {
            let response = native_attachment_upload(
                State(state.clone()),
                headers.clone(),
                Path(sid.clone()),
                Ok(Json(hostile)),
            )
            .await;
            expect_status(response, StatusCode::BAD_REQUEST).await;
        }
        let traversal = native_attachment_upload(
            State(state.clone()),
            headers.clone(),
            Path(sid.clone()),
            Ok(Json(upload("text/plain", Some("../secrets"), b"x"))),
        )
        .await;
        expect_status(traversal, StatusCode::BAD_REQUEST).await;
        let mut bad_b64 = upload("text/plain", None, b"x");
        bad_b64.data_base64 = "@@@@".into();
        let response = native_attachment_upload(
            State(state.clone()),
            headers.clone(),
            Path(sid.clone()),
            Ok(Json(bad_b64)),
        )
        .await;
        expect_status(response, StatusCode::BAD_REQUEST).await;
        // Oversized payloads are typed 413s BEFORE any CAS write.
        let mut big = upload("application/octet-stream", None, b"");
        big.data_base64 = "A".repeat(MAX_ATTACHMENT_UPLOAD_BYTES.div_ceil(3) * 4 + 8);
        let response = native_attachment_upload(
            State(state.clone()),
            headers.clone(),
            Path(sid.clone()),
            Ok(Json(big)),
        )
        .await;
        expect_status(response, StatusCode::PAYLOAD_TOO_LARGE).await;
        // Nothing durable was left behind by any hostile attempt.
        assert!(handle.list_attachments(16).unwrap().is_empty());
    }

    #[test]
    fn upload_dto_is_strict() {
        // Unknown and missing members are rejected by serde itself.
        assert!(
            serde_json::from_value::<NativeAttachmentUpload>(serde_json::json!({
                "mime": "text/plain",
                "data_base64": "eA==",
                "extra": 1
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<NativeAttachmentUpload>(serde_json::json!({
                "mime": "text/plain"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<NativeAttachmentUpload>(serde_json::json!({
                "mime": "text/plain",
                "filename": 7,
                "data_base64": "eA=="
            }))
            .is_err()
        );
        let ok: NativeAttachmentUpload = serde_json::from_value(serde_json::json!({
            "mime": "text/plain",
            "data_base64": "eA=="
        }))
        .unwrap();
        assert!(ok.filename.is_none());
    }

    #[test]
    fn wire_admission_requires_durable_rows_and_refuses_images() {
        let dir = tempfile::tempdir().unwrap();
        let (_state, handle) = test_state(dir.path());
        let stored = handle
            .put_attachment("application/pdf", Some("spec.pdf"), b"%PDF")
            .unwrap();
        // A durable byte-identical id admits; an unknown digest is a 400
        // (client error) with the not-found cause preserved.
        validate_wire_attachments(&handle, std::slice::from_ref(&stored))
            .expect("durable id admits");
        let unknown = AttachmentId {
            digest: FileHash::from([8; 32]),
            ..stored.clone()
        };
        let err = validate_wire_attachments(&handle, &[unknown]).expect_err("unknown digest");
        assert_eq!(err.http_status, 400);
        assert!(err.message.contains("not stored"), "{err:?}");
        // A programmatically stored image is refused at admission with the
        // typed `unsupported` code (provider media parts are not wired).
        let image = handle
            .put_attachment("image/png", None, b"\x89PNG")
            .unwrap();
        let err = validate_wire_attachments(&handle, &[image, stored]).expect_err("image");
        assert_eq!(err.code, "unsupported");
        assert!(err.message.contains("provider media"), "{err:?}");
    }
}
