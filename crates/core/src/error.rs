//! Typed errors. Every error carries an explicit retryability class so that
//! state-aware retries (never blind replays) can be decided structurally.

use crate::state::AgentState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Resource (session, message, artifact, model) does not exist.
    NotFound,
    /// Optimistic-concurrency conflict (hash mismatch, stale patch, ...).
    Conflict,
    /// An illegal state-machine transition was attempted.
    InvalidState {
        from: AgentState,
        to: AgentState,
    },
    /// Denied by the sandbox / permission engine.
    Permission,
    /// Deadline exceeded.
    Timeout,
    /// Operation cancelled.
    Cancelled,
    /// Storage layer failure (SQLite, CAS).
    Store,
    /// Transport-level network failure; safe to retry.
    Network,
    /// Provider returned an error; retryable iff `code` says so.
    Provider { code: String, retryable: bool },
    /// Malformed payload (bad JSON, bad protocol, bad tool call).
    Malformed,
    /// Input exceeded a configured bound (bytes, tokens, depth).
    Oversized,
    /// Provider rate limiting; retry after backoff.
    RateLimited,
    /// Scheduler detected a deadlock/starvation cycle.
    Deadlock,
    /// Anything else.
    Internal,
}

impl ErrorKind {
    pub fn is_retryable(&self) -> bool {
        match self {
            ErrorKind::Network => true,
            ErrorKind::Timeout => true,
            ErrorKind::RateLimited => true,
            ErrorKind::Provider { retryable, .. } => *retryable,
            ErrorKind::Store => true, // transient storage failures are retryable
            ErrorKind::Deadlock => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub retryable: bool,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        let retryable = kind.is_retryable();
        Self {
            kind,
            message: message.into(),
            retryable,
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Conflict, message)
    }

    pub fn permission(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Permission, message)
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Timeout, message)
    }

    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled, "operation cancelled")
    }

    pub fn malformed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Malformed, message)
    }

    pub fn oversized(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Oversized, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::internal(format!("io error: {e}"))
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::malformed(format!("json error: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryability_is_explicit_and_correct() {
        assert!(ErrorKind::Network.is_retryable());
        assert!(ErrorKind::Timeout.is_retryable());
        assert!(ErrorKind::RateLimited.is_retryable());
        assert!(ErrorKind::Provider { code: "429".into(), retryable: true }.is_retryable());
        assert!(!ErrorKind::Provider { code: "400".into(), retryable: false }.is_retryable());
        assert!(!ErrorKind::Conflict.is_retryable());
        assert!(!ErrorKind::Permission.is_retryable());
        assert!(!ErrorKind::Malformed.is_retryable());
        assert!(!ErrorKind::Oversized.is_retryable());
        assert!(!ErrorKind::NotFound.is_retryable());
    }

    #[test]
    fn error_carries_retryable_flag_consistent_with_kind() {
        let e = Error::new(ErrorKind::Network, "boom");
        assert!(e.retryable);
        let e = Error::conflict("stale");
        assert!(!e.retryable);
    }

    #[test]
    fn malformed_json_forwarded() {
        let e: Error = serde_json::from_str::<serde_json::Value>("{").unwrap_err().into();
        assert_eq!(e.kind, ErrorKind::Malformed);
    }

    #[test]
    fn invalid_state_error_carries_both_ends() {
        let e = Error::new(
            ErrorKind::InvalidState {
                from: AgentState::Completed,
                to: AgentState::Preparing,
            },
            "illegal",
        );
        assert!(e.message.contains("illegal"));
    }
}
