//! Strict external semantic-provider configuration and clients (audit
//! 48-54/58/59/83).
//!
//! A host may configure one or more EXTERNAL semantic providers in the
//! additive `[semantic]` section. Exactly two transports exist, and both are
//! built on the daemon's own authorities — never on a private process or
//! HTTP layer:
//!
//! - [`SemanticProviderConfig::Process`] runs a child through the ONE
//!   [`ProcessSupervisor`] with a sanitized environment (the supervisor's
//!   cleared-env base) and a typed, bounded, `Content-Length`-framed JSON
//!   protocol on stdin/stdout. The configured `timeout_ms` bounds the child
//!   and caller cancellation kills the supervised group.
//! - [`SemanticProviderConfig::Http`] POSTs the same typed frame through the
//!   injected [`HttpTransport`] — production wires the policy-checked +
//!   secret-scanned egress transport, so destination policy and whole-payload
//!   secret scanning run before any connect — under a `timeout_ms` deadline.
//!
//! Every response is validated before it can become provider DATA: frame
//! schema version, echoed operation, provider identity, and then the full
//! envelope validation (schema, workspace, snapshot, payload bounds, entity
//! ids, provenance). A failure is a typed error; the registry degrades it to
//! the generic fallback unless the call carries `require_provider`.

use std::future::Future;
use std::io::{BufReader, Read};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use faktor_core::{CommandSpec, EnvSpec};
use faktor_provider::egress::{execute_post_json, HttpTransport};
use faktor_terminal::{ProcessOwner, ProcessSupervisor, SpawnConfig};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::types::{
    AffectedRequest, AffectedSet, BoxFuture, SemanticCall, SemanticCapabilities,
    SemanticContextPack, SemanticContextRequest, SemanticDelta, SemanticDeltaRequest,
    SemanticEnvelope, SemanticError, SemanticExplainRequest, SemanticExplanation, SemanticPayload,
    SemanticProvider, SemanticProviderId, SemanticResponseCaps, SemanticSnapshot,
    SemanticSnapshotRequest, SemanticVerification, SemanticVerifyRequest, SEMANTIC_SCHEMA_VERSION,
};

/// Bounds of the external-provider config surface (hostile configs are
/// rejected, never spawned or dialed).
pub const MAX_EXTERNAL_COMMAND_BYTES: usize = 512;
pub const MAX_EXTERNAL_ARGS: usize = 32;
pub const MAX_EXTERNAL_ARG_BYTES: usize = 512;
pub const MAX_EXTERNAL_ENDPOINT_BYTES: usize = 2048;
pub const MAX_EXTERNAL_TIMEOUT_MS: u64 = 300_000;
pub const MAX_EXTERNAL_AUTH_ENV_BYTES: usize = 128;

/// The frame schema version of the external wire protocol. Frames carry
/// explicit typed fields and unknown schema versions are refused.
pub const WIRE_SCHEMA_VERSION: u32 = SEMANTIC_SCHEMA_VERSION;

/// Slack above the configured payload cap for headers/framing.
const WIRE_FRAME_SLACK_BYTES: usize = 64 * 1024;
/// Absolute hard ceiling on one wire body (defense in depth).
const MAX_WIRE_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Maximum bytes of one frame header block.
const MAX_WIRE_HEADER_BYTES: usize = 16 * 1024;

/// One external semantic provider entry, parsed strictly (all fields
/// explicit; unknown keys and invalid values are parse errors).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SemanticProviderConfig {
    /// A supervised local child speaking the typed stdin/stdout frame
    /// protocol.
    Process {
        id: SemanticProviderId,
        command: String,
        #[serde(default)]
        args: Vec<String>,
        timeout_ms: u64,
    },
    /// A remote endpoint reached through the injected checked transport.
    Http {
        id: SemanticProviderId,
        endpoint: String,
        #[serde(default)]
        auth_env: Option<String>,
        timeout_ms: u64,
    },
}

impl SemanticProviderConfig {
    pub fn id(&self) -> &SemanticProviderId {
        match self {
            Self::Process { id, .. } | Self::Http { id, .. } => id,
        }
    }

    pub fn timeout_ms(&self) -> u64 {
        match self {
            Self::Process { timeout_ms, .. } | Self::Http { timeout_ms, .. } => *timeout_ms,
        }
    }

    /// Typed validation: bounded command/args/endpoint/timeout, an http(s)
    /// endpoint, and a syntactically safe auth env var name.
    pub fn validate(&self) -> Result<(), SemanticError> {
        let timeout_ms = self.timeout_ms();
        if timeout_ms == 0 || timeout_ms > MAX_EXTERNAL_TIMEOUT_MS {
            return Err(SemanticError::Oversized {
                max: MAX_EXTERNAL_TIMEOUT_MS as usize,
                actual: timeout_ms as usize,
            });
        }
        match self {
            Self::Process { command, args, .. } => {
                if command.is_empty() || command.len() > MAX_EXTERNAL_COMMAND_BYTES {
                    return Err(SemanticError::Malformed(
                        "process semantic provider command is empty or oversized".to_string(),
                    ));
                }
                if args.len() > MAX_EXTERNAL_ARGS {
                    return Err(SemanticError::Oversized {
                        max: MAX_EXTERNAL_ARGS,
                        actual: args.len(),
                    });
                }
                for arg in args {
                    if arg.len() > MAX_EXTERNAL_ARG_BYTES {
                        return Err(SemanticError::Oversized {
                            max: MAX_EXTERNAL_ARG_BYTES,
                            actual: arg.len(),
                        });
                    }
                }
            }
            Self::Http {
                endpoint, auth_env, ..
            } => {
                if endpoint.len() > MAX_EXTERNAL_ENDPOINT_BYTES {
                    return Err(SemanticError::Oversized {
                        max: MAX_EXTERNAL_ENDPOINT_BYTES,
                        actual: endpoint.len(),
                    });
                }
                if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
                    return Err(SemanticError::Malformed(format!(
                        "semantic provider endpoint {endpoint:?} must be an http(s) URL"
                    )));
                }
                if let Some(var) = auth_env {
                    if var.is_empty()
                        || var.len() > MAX_EXTERNAL_AUTH_ENV_BYTES
                        || !var.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    {
                        return Err(SemanticError::Malformed(format!(
                            "semantic provider auth_env {var:?} must match [A-Za-z0-9_]+"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Build the client over the daemon's own authorities. Every response
    /// the client returns has already been identity-checked; the registry
    /// re-validates the full envelope.
    pub fn build(
        &self,
        env: &SemanticClientEnv,
    ) -> Result<Arc<dyn SemanticProvider>, SemanticError> {
        self.validate()?;
        let max_response_bytes = env
            .caps
            .max_payload_bytes
            .saturating_add(WIRE_FRAME_SLACK_BYTES)
            .min(MAX_WIRE_BODY_BYTES);
        let timeout = Duration::from_millis(self.timeout_ms());
        let provider: Arc<dyn SemanticProvider> = match self {
            Self::Process {
                id, command, args, ..
            } => Arc::new(ExternalProvider {
                inner: ProcessSemanticClient {
                    id: id.clone(),
                    command: command.clone(),
                    args: args.clone(),
                    timeout,
                    supervisor: env.supervisor.clone(),
                    max_response_bytes,
                },
            }),
            Self::Http {
                id,
                endpoint,
                auth_env,
                ..
            } => Arc::new(ExternalProvider {
                inner: HttpSemanticClient {
                    id: id.clone(),
                    endpoint: endpoint.clone(),
                    auth_env: auth_env.clone(),
                    timeout,
                    transport: env.transport.clone(),
                    max_response_bytes,
                },
            }),
        };
        Ok(provider)
    }
}

/// The daemon authorities an external client executes through.
#[derive(Clone)]
pub struct SemanticClientEnv {
    pub supervisor: Arc<ProcessSupervisor>,
    pub transport: Arc<dyn HttpTransport>,
    pub caps: SemanticResponseCaps,
}

/// The single external client contract: one typed JSON call per operation.
trait ExternalTransport: Send + Sync {
    fn provider_id(&self) -> SemanticProviderId;
    fn provider_version(&self) -> u32;
    fn call_json<T>(
        &self,
        op: crate::types::SemanticOp,
        call: SemanticCall,
        body: serde_json::Value,
    ) -> BoxFuture<'static, Result<SemanticEnvelope<T>, SemanticError>>
    where
        T: SemanticPayload + DeserializeOwned + Send + 'static;
}

/// The [`SemanticProvider`] adapter over one external transport.
struct ExternalProvider<C: ExternalTransport> {
    inner: C,
}

impl<C: ExternalTransport> SemanticProvider for ExternalProvider<C> {
    fn id(&self) -> SemanticProviderId {
        self.inner.provider_id()
    }

    fn version(&self) -> u32 {
        self.inner.provider_version()
    }

    fn capabilities(&self) -> SemanticCapabilities {
        // External providers advertise every operation; an operation they do
        // not implement fails typed and degrades to the generic fallback
        // (unless the call required provider proof).
        SemanticCapabilities::ALL
    }

    fn snapshot(
        &self,
        request: SemanticSnapshotRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticSnapshot>, SemanticError>> {
        let body = serde_json::json!({
            "workspace": request.workspace,
            "source_revision": request.source_revision,
        });
        self.inner
            .call_json(crate::types::SemanticOp::Snapshot, request.call, body)
    }

    fn context(
        &self,
        request: SemanticContextRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticContextPack>, SemanticError>> {
        let body = serde_json::json!({
            "workspace": request.workspace,
            "source_revision": request.source_revision,
            "snapshot_id": request.snapshot_id,
            "query": request.query,
            "max_items": request.max_items,
            "max_bytes": request.max_bytes,
        });
        self.inner
            .call_json(crate::types::SemanticOp::Context, request.call, body)
    }

    fn delta(
        &self,
        request: SemanticDeltaRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticDelta>, SemanticError>> {
        let body = serde_json::json!({
            "workspace": request.workspace,
            "from_snapshot": request.from_snapshot,
            "from_source_revision": request.from_source_revision,
            "to_source_revision": request.to_source_revision,
        });
        self.inner
            .call_json(crate::types::SemanticOp::Delta, request.call, body)
    }

    fn affected(
        &self,
        request: AffectedRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<AffectedSet>, SemanticError>> {
        let body = serde_json::json!({
            "workspace": request.workspace,
            "snapshot_id": request.snapshot_id,
            "changed": request.changed,
            "max_depth": request.max_depth,
        });
        self.inner
            .call_json(crate::types::SemanticOp::Affected, request.call, body)
    }

    fn verify(
        &self,
        request: SemanticVerifyRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticVerification>, SemanticError>> {
        let body = serde_json::json!({
            "workspace": request.workspace,
            "snapshot_id": request.snapshot_id,
            "entity": request.entity,
            "claim": request.claim,
        });
        self.inner
            .call_json(crate::types::SemanticOp::Verify, request.call, body)
    }

    fn explain(
        &self,
        request: SemanticExplainRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticExplanation>, SemanticError>> {
        let body = serde_json::json!({
            "workspace": request.workspace,
            "snapshot_id": request.snapshot_id,
            "entity": request.entity,
            "question": request.question,
        });
        self.inner
            .call_json(crate::types::SemanticOp::Explain, request.call, body)
    }
}

/// A supervised-child client (see the module docs).
pub struct ProcessSemanticClient {
    id: SemanticProviderId,
    command: String,
    args: Vec<String>,
    timeout: Duration,
    supervisor: Arc<ProcessSupervisor>,
    max_response_bytes: usize,
}

/// An HTTP client over the injected checked transport (see the module docs).
pub struct HttpSemanticClient {
    id: SemanticProviderId,
    endpoint: String,
    auth_env: Option<String>,
    timeout: Duration,
    transport: Arc<dyn HttpTransport>,
    max_response_bytes: usize,
}

// ---------------------------------------------------------------------------
// wire framing
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct WireRequest<'a> {
    v: u32,
    op: &'a str,
    provider: &'a SemanticProviderId,
    request: &'a serde_json::Value,
}

#[derive(Deserialize)]
struct WireResponse {
    v: u32,
    op: String,
    envelope: serde_json::Value,
}

fn request_frame(
    op: crate::types::SemanticOp,
    provider: &SemanticProviderId,
    body: &serde_json::Value,
) -> Result<serde_json::Value, SemanticError> {
    serde_json::to_value(WireRequest {
        v: WIRE_SCHEMA_VERSION,
        op: op.as_str(),
        provider,
        request: body,
    })
    .map_err(|e| SemanticError::Malformed(format!("semantic request is not serializable: {e}")))
}

fn encode_frame(value: &serde_json::Value) -> Result<Vec<u8>, SemanticError> {
    let body = serde_json::to_vec(value).map_err(|e| {
        SemanticError::Malformed(format!("semantic frame is not serializable: {e}"))
    })?;
    let mut frame = Vec::with_capacity(body.len() + 32);
    frame.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}

fn parse_wire_response<T>(
    payload: &[u8],
    op: crate::types::SemanticOp,
    provider: &SemanticProviderId,
) -> Result<SemanticEnvelope<T>, SemanticError>
where
    T: SemanticPayload + DeserializeOwned,
{
    let frame: WireResponse = serde_json::from_slice(payload)
        .map_err(|e| SemanticError::Malformed(format!("malformed semantic wire response: {e}")))?;
    if frame.v != WIRE_SCHEMA_VERSION {
        return Err(SemanticError::UnsupportedSchema {
            supported: WIRE_SCHEMA_VERSION,
            got: frame.v,
        });
    }
    if frame.op != op.as_str() {
        return Err(SemanticError::Malformed(format!(
            "semantic wire response op {:?} does not match the request op {:?}",
            frame.op,
            op.as_str()
        )));
    }
    let envelope: SemanticEnvelope<T> = serde_json::from_value(frame.envelope).map_err(|e| {
        SemanticError::Malformed(format!("malformed semantic response envelope: {e}"))
    })?;
    envelope.validate_provider_id(provider)?;
    // The envelope's own schema and provenance are validated HERE too, so a
    // direct client can never return an envelope the registry would reject
    // (workspace/snapshot/entity checks need the caller's expectation and
    // stay in `SemanticEnvelope::validate`).
    if envelope.schema_version != WIRE_SCHEMA_VERSION {
        return Err(SemanticError::UnsupportedSchema {
            supported: WIRE_SCHEMA_VERSION,
            got: envelope.schema_version,
        });
    }
    envelope.assert_data_only()?;
    Ok(envelope)
}

/// Read one `Content-Length` framed body with a hard byte bound. Header
/// bytes, duplicate/missing/invalid lengths and oversized bodies are typed
/// refusals; the reader never buffers past `max_body`.
fn read_frame(reader: &mut impl Read, max_body: usize) -> Result<Vec<u8>, SemanticError> {
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    while !header.ends_with(b"\r\n\r\n") {
        if header.len() >= MAX_WIRE_HEADER_BYTES {
            return Err(SemanticError::Oversized {
                max: MAX_WIRE_HEADER_BYTES,
                actual: header.len(),
            });
        }
        reader
            .read_exact(&mut byte)
            .map_err(|e| SemanticError::Malformed(format!("semantic frame header read: {e}")))?;
        header.push(byte[0]);
    }
    let text = std::str::from_utf8(&header)
        .map_err(|_| SemanticError::Malformed("semantic frame header is not UTF-8".to_string()))?;
    let mut length: Option<usize> = None;
    for line in text.split("\r\n") {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(SemanticError::Malformed(format!(
                "semantic frame header line {line:?} has no value separator"
            )));
        };
        if !name.trim().eq_ignore_ascii_case("content-length") {
            continue;
        }
        if length.is_some() {
            return Err(SemanticError::Malformed(
                "semantic frame carries a duplicate Content-Length".to_string(),
            ));
        }
        length = Some(value.trim().parse::<usize>().map_err(|_| {
            SemanticError::Malformed(format!(
                "semantic frame Content-Length {value:?} is invalid"
            ))
        })?);
    }
    let length = length.ok_or_else(|| {
        SemanticError::Malformed("semantic frame is missing Content-Length".to_string())
    })?;
    if length > max_body {
        return Err(SemanticError::Oversized {
            max: max_body,
            actual: length,
        });
    }
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .map_err(|e| SemanticError::Malformed(format!("semantic frame body read: {e}")))?;
    Ok(body)
}

// ---------------------------------------------------------------------------
// process transport
// ---------------------------------------------------------------------------

/// The child slot shared with the async waiter: the pid is published once
/// the child exists, and a cancellation that races the publish still kills
/// the child (the publish re-checks the flag).
#[derive(Default)]
struct ChildSlot {
    pid: Mutex<Option<u32>>,
    cancelled: AtomicBool,
}

impl ChildSlot {
    fn publish(&self, supervisor: &ProcessSupervisor, pid: u32) {
        let mut guard = self.pid.lock().unwrap();
        *guard = Some(pid);
        if self.cancelled.load(Ordering::SeqCst) {
            drop(guard);
            let _ = supervisor.kill_child_pid(pid, 200);
        }
    }

    fn cancel(&self, supervisor: &ProcessSupervisor) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Some(pid) = *self.pid.lock().unwrap() {
            let _ = supervisor.kill_child_pid(pid, 200);
        }
    }
}

/// Run one framed call to completion in a blocking thread: lower the typed
/// [`CommandSpec`] first (empty/NUL programs are refused before any process
/// exists), spawn through the supervisor with the cleared-env platform
/// baseline ([`EnvSpec::default_baseline`] — PATH/HOME and the platform bits,
/// never the daemon's full environment), write the frame to stdin, read the
/// bounded response frame from stdout, and always retire the child. The
/// watchdog kills the group when `timeout` expires without a response.
#[allow(clippy::too_many_arguments)]
fn run_child_frame(
    supervisor: &Arc<ProcessSupervisor>,
    command: &str,
    args: &[String],
    frame: &[u8],
    max_body: usize,
    timeout: Duration,
    slot: &Arc<ChildSlot>,
    provider: &SemanticProviderId,
) -> Result<Vec<u8>, SemanticError> {
    let failed = |detail: String| SemanticError::ProviderFailed {
        provider: provider.to_string(),
        detail,
    };
    let resolved = CommandSpec::program(command, args.iter().cloned())
        .lower()
        .map_err(|e| failed(format!("provider command refused: {e}")))?;
    let cfg = SpawnConfig {
        cmd: resolved.program.to_string_lossy().into_owned(),
        args: resolved
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect(),
        env: EnvSpec::default_baseline(),
        capture: false,
        artifact_max: 0,
        owner: ProcessOwner::Daemon,
        ..SpawnConfig::default()
    };
    let child = supervisor
        .spawn_detached_with_pipes(cfg)
        .map_err(|e| failed(format!("spawn failed: {e}")))?;
    slot.publish(supervisor, child.child_pid);
    let done = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    {
        let done = done.clone();
        let timed_out = timed_out.clone();
        let supervisor = supervisor.clone();
        let pid = child.child_pid;
        std::thread::spawn(move || {
            let deadline = Instant::now() + timeout;
            while Instant::now() < deadline {
                if done.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            if !done.load(Ordering::SeqCst) {
                timed_out.store(true, Ordering::SeqCst);
                let _ = supervisor.kill_child_pid(pid, 300);
            }
        });
    }
    {
        let mut stdin = child.stdin;
        let mut source = frame;
        std::io::copy(&mut source, &mut stdin)
            .and_then(|_| {
                use std::io::Write as _;
                stdin.flush()
            })
            .map_err(|e| failed(format!("request write failed: {e}")))?;
    }
    let read = read_frame(&mut BufReader::new(child.stdout), max_body);
    done.store(true, Ordering::SeqCst);
    // Always retire the supervised child; the reader/reaper owns the wait.
    let _ = supervisor.kill_child_pid(child.child_pid, 100);
    if timed_out.load(Ordering::SeqCst) {
        return Err(SemanticError::DeadlineExceeded {
            provider: provider.to_string(),
        });
    }
    read
}

/// A blocking thread's answer slot; executor-agnostic and std-only.
mod oneshot {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    struct State<T> {
        value: Option<T>,
        waker: Option<Waker>,
        closed: bool,
    }

    pub(super) fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let inner = Arc::new(Mutex::new(State {
            value: None,
            waker: None,
            closed: false,
        }));
        (Sender(inner.clone()), Receiver(inner))
    }

    pub(super) struct Sender<T>(Arc<Mutex<State<T>>>);

    impl<T> Sender<T> {
        pub(super) fn send(self, value: T) {
            let waker = {
                let mut state = self.0.lock().unwrap();
                if state.closed {
                    return;
                }
                state.value = Some(value);
                state.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        }
    }

    pub(super) struct Receiver<T>(Arc<Mutex<State<T>>>);

    impl<T> Drop for Receiver<T> {
        fn drop(&mut self) {
            if let Ok(mut state) = self.0.lock() {
                state.closed = true;
            }
        }
    }

    impl<T> Unpin for Receiver<T> {}

    impl<T> Future for Receiver<T> {
        type Output = Option<T>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
            let mut state = self.0.lock().unwrap();
            if let Some(value) = state.value.take() {
                return Poll::Ready(Some(value));
            }
            if state.closed {
                return Poll::Ready(None);
            }
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Wait for the blocking thread's frame while observing caller cancellation.
struct ProcessWait {
    rx: oneshot::Receiver<Result<Vec<u8>, SemanticError>>,
    cancel: Pin<Box<dyn Future<Output = ()> + Send>>,
    slot: Arc<ChildSlot>,
    supervisor: Arc<ProcessSupervisor>,
    provider: SemanticProviderId,
}

impl Future for ProcessWait {
    type Output = Result<Vec<u8>, SemanticError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.cancel.as_mut().poll(cx).is_ready() {
            this.slot.cancel(&this.supervisor);
            return Poll::Ready(Err(SemanticError::Cancelled {
                provider: this.provider.to_string(),
            }));
        }
        match Pin::new(&mut this.rx).poll(cx) {
            Poll::Ready(Some(result)) => Poll::Ready(result),
            Poll::Ready(None) => Poll::Ready(Err(SemanticError::ProviderFailed {
                provider: this.provider.to_string(),
                detail: "provider worker ended without a response".to_string(),
            })),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl ExternalTransport for ProcessSemanticClient {
    fn provider_id(&self) -> SemanticProviderId {
        self.id.clone()
    }

    fn provider_version(&self) -> u32 {
        1
    }

    fn call_json<T>(
        &self,
        op: crate::types::SemanticOp,
        call: SemanticCall,
        body: serde_json::Value,
    ) -> BoxFuture<'static, Result<SemanticEnvelope<T>, SemanticError>>
    where
        T: SemanticPayload + DeserializeOwned + Send + 'static,
    {
        let provider = self.id.clone();
        if call.cancellation.is_cancelled() {
            return Box::pin(async move {
                Err(SemanticError::Cancelled {
                    provider: provider.to_string(),
                })
            });
        }
        let value = match request_frame(op, &provider, &body) {
            Ok(value) => value,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let frame = match encode_frame(&value) {
            Ok(frame) => frame,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let (tx, rx) = oneshot::channel();
        let slot = Arc::new(ChildSlot::default());
        let supervisor = self.supervisor.clone();
        let wait_supervisor = self.supervisor.clone();
        let command = self.command.clone();
        let args = self.args.clone();
        let timeout = self.timeout;
        let max_body = self.max_response_bytes;
        let worker_provider = provider.clone();
        let worker_slot = slot.clone();
        std::thread::spawn(move || {
            let result = run_child_frame(
                &supervisor,
                &command,
                &args,
                &frame,
                max_body,
                timeout,
                &worker_slot,
                &worker_provider,
            );
            tx.send(result);
        });
        let token = call.cancellation.clone();
        let cancel = Box::pin(async move { token.cancelled().await });
        Box::pin(async move {
            let bytes = ProcessWait {
                rx,
                cancel,
                slot,
                supervisor: wait_supervisor,
                provider: provider.clone(),
            }
            .await?;
            parse_wire_response::<T>(&bytes, op, &provider)
        })
    }
}

// ---------------------------------------------------------------------------
// http transport
// ---------------------------------------------------------------------------

impl ExternalTransport for HttpSemanticClient {
    fn provider_id(&self) -> SemanticProviderId {
        self.id.clone()
    }

    fn provider_version(&self) -> u32 {
        1
    }

    fn call_json<T>(
        &self,
        op: crate::types::SemanticOp,
        call: SemanticCall,
        body: serde_json::Value,
    ) -> BoxFuture<'static, Result<SemanticEnvelope<T>, SemanticError>>
    where
        T: SemanticPayload + DeserializeOwned + Send + 'static,
    {
        let provider = self.id.clone();
        if call.cancellation.is_cancelled() {
            return Box::pin(async move {
                Err(SemanticError::Cancelled {
                    provider: provider.to_string(),
                })
            });
        }
        let value = match request_frame(op, &provider, &body) {
            Ok(value) => value,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let auth_env = self.auth_env.clone();
        let timeout = self.timeout;
        let max_body = self.max_response_bytes;
        let token = call.cancellation.clone();
        Box::pin(async move {
            let mut headers = http::HeaderMap::new();
            if let Some(var) = &auth_env {
                let secret = std::env::var(var).map_err(|_| SemanticError::ProviderFailed {
                    provider: provider.to_string(),
                    detail: format!("auth env {var} is not set"),
                })?;
                let value =
                    http::HeaderValue::from_str(&format!("Bearer {secret}")).map_err(|_| {
                        SemanticError::ProviderFailed {
                            provider: provider.to_string(),
                            detail: format!("auth env {var} is not a valid header value"),
                        }
                    })?;
                headers.insert(http::header::AUTHORIZATION, value);
            }
            let send = async {
                let response = execute_post_json(transport.as_ref(), &endpoint, headers, &value)
                    .await
                    .map_err(|e| SemanticError::ProviderFailed {
                        provider: provider.to_string(),
                        detail: format!("egress refused: {e}"),
                    })?;
                let status = response.status();
                if !status.is_success() {
                    return Err(SemanticError::ProviderFailed {
                        provider: provider.to_string(),
                        detail: format!("provider endpoint answered HTTP {status}"),
                    });
                }
                let mut response = response;
                let mut bytes: Vec<u8> = Vec::new();
                while let Some(chunk) =
                    response
                        .chunk()
                        .await
                        .map_err(|e| SemanticError::ProviderFailed {
                            provider: provider.to_string(),
                            detail: format!("provider response read failed: {e}"),
                        })?
                {
                    if bytes.len().saturating_add(chunk.len()) > max_body {
                        return Err(SemanticError::Oversized {
                            max: max_body,
                            actual: bytes.len().saturating_add(chunk.len()),
                        });
                    }
                    bytes.extend_from_slice(&chunk);
                }
                Ok::<Vec<u8>, SemanticError>(bytes)
            };
            let bytes = tokio::select! {
                _ = token.cancelled() => {
                    return Err(SemanticError::Cancelled { provider: provider.to_string() });
                }
                result = tokio::time::timeout(timeout, send) => match result {
                    Ok(result) => result?,
                    Err(_) => {
                        return Err(SemanticError::DeadlineExceeded {
                            provider: provider.to_string(),
                        });
                    }
                },
            };
            parse_wire_response::<T>(&bytes, op, &provider)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fallback::{GenericSemanticFallback, GENERIC_FALLBACK_ID};
    use crate::registry::SemanticProviderRegistry;
    use crate::test_support::{call, provider_id, snapshot};
    use crate::types::{
        SemanticContextPack, SemanticContextRequest, SemanticOp, SemanticResponseCaps,
        SemanticSnapshotId, SEMANTIC_SCHEMA_VERSION,
    };
    use faktor_core::{CancellationToken, WorkspaceId};
    use faktor_provider::egress::{MockHttpTransport, PolicyCheckedHttpTransport};
    use std::fs;
    use std::path::{Path, PathBuf};

    fn workspace() -> WorkspaceId {
        WorkspaceId::new(1)
    }

    fn expected_snapshot() -> SemanticSnapshotId {
        snapshot(workspace(), "rev-1")
    }

    fn context_request() -> SemanticContextRequest {
        SemanticContextRequest {
            call: call(),
            workspace: workspace(),
            source_revision: "rev-1".to_string(),
            snapshot_id: expected_snapshot(),
            query: "where is the scheduler".to_string(),
            max_items: 8,
            max_bytes: 4096,
        }
    }

    fn context_envelope(provider: &SemanticProviderId) -> serde_json::Value {
        let envelope = SemanticEnvelope::new(
            provider.clone(),
            1,
            workspace(),
            expected_snapshot(),
            7,
            SemanticContextPack {
                items: Vec::new(),
                truncated: false,
                total_bytes: 0,
                degraded: false,
            },
        );
        serde_json::to_value(envelope).unwrap()
    }

    /// A valid context envelope with one field mutated (hostile shapes).
    fn mutated_envelope(
        provider: &SemanticProviderId,
        mutate: impl FnOnce(&mut serde_json::Value),
    ) -> serde_json::Value {
        let mut envelope = context_envelope(provider);
        mutate(&mut envelope);
        envelope
    }

    fn frame(
        op: SemanticOp,
        provider: &SemanticProviderId,
        envelope: serde_json::Value,
    ) -> Vec<u8> {
        typed_frame(&serde_json::json!({
            "v": WIRE_SCHEMA_VERSION,
            "op": op.as_str(),
            "provider": provider,
            "envelope": envelope,
        }))
    }

    fn typed_frame(value: &serde_json::Value) -> Vec<u8> {
        let body = serde_json::to_vec(value).unwrap();
        let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        out.extend_from_slice(&body);
        out
    }

    /// The HTTP wire body: the transport carries its own framing, so the
    /// response body is the bare typed wire JSON object.
    fn wire_body(op: SemanticOp, envelope: serde_json::Value) -> String {
        serde_json::to_string(&serde_json::json!({
            "v": WIRE_SCHEMA_VERSION,
            "op": op.as_str(),
            "envelope": envelope,
        }))
        .unwrap()
    }

    fn raw_frame(body: &[u8]) -> Vec<u8> {
        let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        out.extend_from_slice(body);
        out
    }

    fn supervisor(dir: &Path) -> Arc<ProcessSupervisor> {
        ProcessSupervisor::new(Arc::new(faktor_cas::Cas::new(dir.join("cas"))))
    }

    fn client_env(dir: &Path) -> SemanticClientEnv {
        SemanticClientEnv {
            supervisor: supervisor(dir),
            transport: Arc::new(MockHttpTransport::new(200, "")),
            caps: SemanticResponseCaps::default(),
        }
    }

    // ------------------------------------------------------------------
    // process client over a fake provider program
    // ------------------------------------------------------------------

    #[cfg(unix)]
    struct FakeProcess {
        dir: tempfile::TempDir,
        script: PathBuf,
        capture: PathBuf,
        response: PathBuf,
    }

    #[cfg(unix)]
    impl FakeProcess {
        /// `tail` runs after the request frame was drained to `$1`; `$2` is
        /// the canned response path.
        fn new(tail: &str) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let script = dir.path().join("provider.sh");
            let capture = dir.path().join("request.frame");
            let response = dir.path().join("response.frame");
            fs::write(
                &script,
                format!("#!/bin/sh\ncat > \"$1\" || exit 4\n{tail}\n"),
            )
            .unwrap();
            Self {
                dir,
                script,
                capture,
                response,
            }
        }

        fn config(&self, timeout_ms: u64) -> SemanticProviderConfig {
            SemanticProviderConfig::Process {
                id: provider_id("fake-proc"),
                command: "/bin/sh".to_string(),
                args: vec![
                    self.script.to_string_lossy().into_owned(),
                    self.capture.to_string_lossy().into_owned(),
                    self.response.to_string_lossy().into_owned(),
                ],
                timeout_ms,
            }
        }

        fn respond(&self, frame: &[u8]) {
            fs::write(&self.response, frame).unwrap();
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn process_client_round_trips_the_typed_frame() {
        let fake = FakeProcess::new("cat \"$2\"");
        let env = client_env(fake.dir.path());
        let provider = fake.config(5_000).build(&env).unwrap();
        fake.respond(&frame(
            SemanticOp::Context,
            &provider_id("fake-proc"),
            context_envelope(&provider_id("fake-proc")),
        ));
        let out = provider.context(context_request()).await.unwrap();
        assert_eq!(out.provider_id.as_str(), "fake-proc");
        assert!(!out.payload.degraded);
        // The child received the typed, bounded request frame.
        let captured = fs::read(&fake.capture).unwrap();
        let text = String::from_utf8(captured).unwrap();
        let (header, body) = text.split_once("\r\n\r\n").unwrap();
        assert_eq!(header, format!("Content-Length: {}", body.len()));
        let request: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(request["v"], WIRE_SCHEMA_VERSION);
        assert_eq!(request["op"], "context");
        assert_eq!(request["provider"], "fake-proc");
        assert_eq!(request["request"]["query"], "where is the scheduler");
        assert_eq!(request["request"]["workspace"], 1);
    }

    /// One adversarial response case: a label, the raw child output, and the
    /// typed error predicate it must produce.
    type HostileCase = (&'static str, Vec<u8>, fn(&SemanticError) -> bool);

    #[cfg(unix)]
    #[tokio::test]
    async fn process_client_refuses_hostile_responses_typed() {
        let fake = FakeProcess::new("cat \"$2\"");
        let env = client_env(fake.dir.path());
        let provider = fake.config(5_000).build(&env).unwrap();
        let expected = provider_id("fake-proc");

        let cases: Vec<HostileCase> = vec![
            (
                "wrong wire schema",
                typed_frame(&serde_json::json!({
                    "v": WIRE_SCHEMA_VERSION + 1,
                    "op": "context",
                    "provider": expected,
                    "envelope": context_envelope(&expected),
                })),
                |e| matches!(e, SemanticError::UnsupportedSchema { .. }),
            ),
            (
                "wrong op echo",
                typed_frame(&serde_json::json!({
                    "v": WIRE_SCHEMA_VERSION,
                    "op": "delta",
                    "envelope": context_envelope(&expected),
                })),
                |e| matches!(e, SemanticError::Malformed(_)),
            ),
            (
                "wrong provider identity",
                frame(
                    SemanticOp::Context,
                    &expected,
                    context_envelope(&provider_id("other")),
                ),
                |e| matches!(e, SemanticError::ProviderMismatch { .. }),
            ),
            (
                "wrong envelope schema",
                typed_frame(&serde_json::json!({
                    "v": WIRE_SCHEMA_VERSION,
                    "op": "context",
                    "envelope": mutated_envelope(&expected, |envelope| {
                        envelope["schema_version"] = serde_json::json!(WIRE_SCHEMA_VERSION + 1);
                    }),
                })),
                |e| matches!(e, SemanticError::UnsupportedSchema { .. }),
            ),
            (
                "user-policy provenance laundering",
                typed_frame(&serde_json::json!({
                    "v": WIRE_SCHEMA_VERSION,
                    "op": "context",
                    "envelope": mutated_envelope(&expected, |envelope| {
                        envelope["provenance"] =
                            serde_json::json!({"entries": ["user_policy"]});
                    }),
                })),
                |e| matches!(e, SemanticError::Refused(_)),
            ),
            ("malformed json", raw_frame(b"{not json"), |e| {
                matches!(e, SemanticError::Malformed(_))
            }),
            (
                "missing content-length",
                b"X-Other: 1\r\n\r\n{}".to_vec(),
                |e| matches!(e, SemanticError::Malformed(_)),
            ),
            (
                "duplicate content-length",
                b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}".to_vec(),
                |e| matches!(e, SemanticError::Malformed(_)),
            ),
            (
                "oversized declared length",
                b"Content-Length: 99999999\r\n\r\n".to_vec(),
                |e| matches!(e, SemanticError::Oversized { .. }),
            ),
            (
                "truncated body",
                b"Content-Length: 99\r\n\r\n{}".to_vec(),
                |e| matches!(e, SemanticError::Malformed(_)),
            ),
        ];
        for (name, response, matches) in cases {
            fake.respond(&response);
            match provider.context(context_request()).await {
                Err(err) => assert!(matches(&err), "{name}: unexpected typed error {err:?}"),
                Ok(other) => panic!("{name}: hostile response was accepted: {other:?}"),
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn process_client_deadline_is_typed_and_retires_the_child() {
        let fake = FakeProcess::new("sleep 30");
        let env = client_env(fake.dir.path());
        let provider = fake.config(150).build(&env).unwrap();
        match provider.context(context_request()).await {
            Err(SemanticError::DeadlineExceeded { provider }) => assert_eq!(provider, "fake-proc"),
            other => panic!("expected DeadlineExceeded, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelled_process_call_is_typed_and_never_spawns() {
        let fake = FakeProcess::new("cat \"$2\"");
        let env = client_env(fake.dir.path());
        let provider = fake.config(5_000).build(&env).unwrap();
        let mut request = context_request();
        let token = CancellationToken::new();
        token.cancel();
        request.call.cancellation = token;
        assert!(matches!(
            provider.context(request).await,
            Err(SemanticError::Cancelled { .. })
        ));
        assert!(
            !fake.capture.exists(),
            "a pre-cancelled call must never spawn the provider child"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn in_flight_cancellation_kills_the_running_child() {
        let fake = FakeProcess::new("cat \"$2\"; sleep 30");
        let env = client_env(fake.dir.path());
        let provider: Arc<dyn SemanticProvider> = fake.config(30_000).build(&env).unwrap();
        let token = CancellationToken::new();
        let mut request = context_request();
        request.call.cancellation = token.clone();
        let handle = tokio::spawn({
            let provider = provider.clone();
            async move { provider.context(request).await }
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        token.cancel();
        match handle.await.unwrap() {
            Err(SemanticError::Cancelled { provider }) => assert_eq!(provider, "fake-proc"),
            other => panic!("expected in-flight Cancelled, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn provider_failure_degrades_to_fallback_and_require_provider_is_typed() {
        let fake = FakeProcess::new("cat \"$2\"");
        let env = client_env(fake.dir.path());
        let provider = fake.config(5_000).build(&env).unwrap();
        fake.respond(b"Content-Length: 5\r\n\r\n{not json");
        let mut registry = SemanticProviderRegistry::new(GenericSemanticFallback::default());
        registry.register(provider);
        // Ordinary consult: the typed failure degrades to the generic
        // fallback, exactly like an absent provider.
        let out = registry.context(context_request()).await.unwrap();
        assert!(out.payload.degraded);
        assert_eq!(out.provider_id.as_str(), GENERIC_FALLBACK_ID);
        // require_provider: the failure is terminal, never laundered.
        let mut request = context_request();
        request.call = request.call.requiring_provider();
        match registry.context(request).await {
            Err(SemanticError::Malformed(detail)) => assert!(detail.contains("wire response")),
            other => panic!("expected the typed provider failure, got {other:?}"),
        }
        // Absence + require_provider is ProviderRequired, never a fallback.
        let empty = SemanticProviderRegistry::new(GenericSemanticFallback::default());
        let mut request = context_request();
        request.call = request.call.requiring_provider();
        match empty.context(request).await {
            Err(SemanticError::ProviderRequired { op }) => assert_eq!(op, "context"),
            other => panic!("expected ProviderRequired, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // http client over the injected checked transport
    // ------------------------------------------------------------------

    fn http_config(endpoint: &str, auth_env: Option<&str>) -> SemanticProviderConfig {
        SemanticProviderConfig::Http {
            id: provider_id("fake-http"),
            endpoint: endpoint.to_string(),
            auth_env: auth_env.map(str::to_string),
            timeout_ms: 5_000,
        }
    }

    #[tokio::test]
    async fn http_client_validates_frames_over_the_mock_transport() {
        let dir = tempfile::tempdir().unwrap();
        let expected = provider_id("fake-http");
        let mock = Arc::new(MockHttpTransport::new(
            200,
            wire_body(SemanticOp::Context, context_envelope(&expected)),
        ));
        let env = SemanticClientEnv {
            supervisor: supervisor(dir.path()),
            transport: mock.clone(),
            caps: SemanticResponseCaps::default(),
        };
        let provider = http_config("http://provider.example/context", None)
            .build(&env)
            .unwrap();
        let out = provider.context(context_request()).await.unwrap();
        assert_eq!(out.provider_id.as_str(), "fake-http");
        assert_eq!(
            mock.requests(),
            vec![(
                "POST".to_string(),
                "http://provider.example/context".to_string()
            )]
        );

        // Wrong identity is a typed refusal even through the mock.
        let mock = Arc::new(MockHttpTransport::new(
            200,
            wire_body(SemanticOp::Context, context_envelope(&provider_id("other"))),
        ));
        let env = SemanticClientEnv {
            supervisor: supervisor(dir.path()),
            transport: mock,
            caps: SemanticResponseCaps::default(),
        };
        let provider = http_config("http://provider.example/context", None)
            .build(&env)
            .unwrap();
        assert!(matches!(
            provider.context(context_request()).await,
            Err(SemanticError::ProviderMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn http_client_refuses_denied_destinations_before_connect() {
        use faktor_security::destination::{DeniedReason, RuleMatch};

        let dir = tempfile::tempdir().unwrap();
        // A real policy-checked transport with an EMPTY allowlist denies the
        // destination before any connect: the refusal is typed provider
        // failure, and no frame is inspected.
        let env = SemanticClientEnv {
            supervisor: supervisor(dir.path()),
            transport: Arc::new(PolicyCheckedHttpTransport::with_policy(Some(
                faktor_security::destination::DestinationPolicy::empty(),
            ))),
            caps: SemanticResponseCaps::default(),
        };
        let provider = http_config("http://provider.example/context", None)
            .build(&env)
            .unwrap();
        match provider.context(context_request()).await {
            Err(SemanticError::ProviderFailed { detail, .. }) => {
                assert!(detail.contains("egress refused"), "{detail}");
            }
            other => panic!("expected the egress refusal, got {other:?}"),
        }

        // A mock denial (the transport seam's own typed refusal) maps the
        // same way.
        let mock = MockHttpTransport::denying(faktor_provider::egress::EgressError::Denied {
            url: "http://provider.example/context".to_string(),
            reason: DeniedReason {
                rule_fired: None,
                matched: RuleMatch::None,
            },
        });
        let env = SemanticClientEnv {
            supervisor: supervisor(dir.path()),
            transport: Arc::new(mock),
            caps: SemanticResponseCaps::default(),
        };
        let provider = http_config("http://provider.example/context", None)
            .build(&env)
            .unwrap();
        assert!(matches!(
            provider.context(context_request()).await,
            Err(SemanticError::ProviderFailed { .. })
        ));
        // Missing auth env is a typed refusal before any request exists.
        let mock = Arc::new(MockHttpTransport::new(200, ""));
        let env = SemanticClientEnv {
            supervisor: supervisor(dir.path()),
            transport: mock.clone(),
            caps: SemanticResponseCaps::default(),
        };
        let provider = http_config(
            "http://provider.example/context",
            Some("KP_SEMANTIC_MISSING_TEST_TOKEN"),
        )
        .build(&env)
        .unwrap();
        assert!(matches!(
            provider.context(context_request()).await,
            Err(SemanticError::ProviderFailed { .. })
        ));
        assert_eq!(mock.request_count(), 0, "no request may be attempted");
    }

    #[tokio::test]
    async fn http_client_enforces_the_response_byte_bound() {
        let dir = tempfile::tempdir().unwrap();
        let mock = Arc::new(MockHttpTransport::new(200, "x".repeat(70_000)));
        let env = SemanticClientEnv {
            supervisor: supervisor(dir.path()),
            transport: mock,
            caps: SemanticResponseCaps::new(SEMANTIC_SCHEMA_VERSION, 1, 1),
        };
        let provider = http_config("http://provider.example/context", None)
            .build(&env)
            .unwrap();
        assert!(matches!(
            provider.context(context_request()).await,
            Err(SemanticError::Oversized { .. })
        ));

        // Non-success status is a typed provider failure.
        let mock = Arc::new(MockHttpTransport::new(500, "boom"));
        let env = SemanticClientEnv {
            supervisor: supervisor(dir.path()),
            transport: mock,
            caps: SemanticResponseCaps::default(),
        };
        let provider = http_config("http://provider.example/context", None)
            .build(&env)
            .unwrap();
        assert!(matches!(
            provider.context(context_request()).await,
            Err(SemanticError::ProviderFailed { .. })
        ));
    }

    // ------------------------------------------------------------------
    // hostile config surface
    // ------------------------------------------------------------------

    #[test]
    fn hostile_external_configs_are_refused_typed() {
        let valid = SemanticProviderConfig::Process {
            id: provider_id("ok"),
            command: "/bin/true".to_string(),
            args: vec!["a".to_string()],
            timeout_ms: 1_000,
        };
        assert!(valid.validate().is_ok());
        for bad in [
            SemanticProviderConfig::Process {
                id: provider_id("ok"),
                command: String::new(),
                args: vec![],
                timeout_ms: 1_000,
            },
            SemanticProviderConfig::Process {
                id: provider_id("ok"),
                command: "/bin/true".to_string(),
                args: vec!["a".to_string(); MAX_EXTERNAL_ARGS + 1],
                timeout_ms: 1_000,
            },
            SemanticProviderConfig::Process {
                id: provider_id("ok"),
                command: "/bin/true".to_string(),
                args: vec!["x".repeat(MAX_EXTERNAL_ARG_BYTES + 1)],
                timeout_ms: 1_000,
            },
            SemanticProviderConfig::Process {
                id: provider_id("ok"),
                command: "x".repeat(MAX_EXTERNAL_COMMAND_BYTES + 1),
                args: vec![],
                timeout_ms: 1_000,
            },
            SemanticProviderConfig::Process {
                id: provider_id("ok"),
                command: "/bin/true".to_string(),
                args: vec![],
                timeout_ms: 0,
            },
            SemanticProviderConfig::Process {
                id: provider_id("ok"),
                command: "/bin/true".to_string(),
                args: vec![],
                timeout_ms: MAX_EXTERNAL_TIMEOUT_MS + 1,
            },
            http_config("ftp://provider.example", None),
            http_config("provider.example", None),
            http_config("http://provider.example", Some("BAD-NAME")),
            SemanticProviderConfig::Http {
                id: provider_id("ok"),
                endpoint: "http://provider.example".to_string(),
                auth_env: Some("x".repeat(MAX_EXTERNAL_AUTH_ENV_BYTES + 1)),
                timeout_ms: 1_000,
            },
        ] {
            assert!(
                bad.validate().is_err(),
                "hostile config must be refused: {bad:?}"
            );
        }
    }
}
