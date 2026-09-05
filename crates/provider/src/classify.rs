//! Single HTTP error classifier for every adapter transport (audit waves
//! 93-96 subset). One classification policy, one code table — adapters must
//! never re-derive status→kind chains locally, or the taxonomies drift.
//!
//! Classification contract:
//!
//! | HTTP status | hint (`error.type`/`error.code`/`error.status`) | kind |
//! |---|---|---|
//! | 401, 407 | ANY (even a `rate_limit`-looking body) | `Auth` |
//! | 403 | `permission_denied`-family (Google `PERMISSION_DENIED`) | `Permission` |
//! | 403 | `rate_limit`-family (`RESOURCE_EXHAUSTED`, quota) | `RateLimited` |
//! | 403 | auth-family or none | `Auth` |
//! | 400 | auth/permission-family (OpenAI `code: invalid_api_key`) | `Auth` |
//! | 400 | none | `Malformed` |
//! | 404 | — | `NotFound` |
//! | 409 | — | `Conflict` |
//! | 422, 501 | — | `Unsupported` |
//! | 429 | ANY | `RateLimited` |
//! | 408, 425, 504 | — | `Timeout` |
//! | 5xx (rest) | — | `Provider { retryable: true }` |
//! | other 4xx | — | `Provider { retryable: false }` |
//!
//! Why status beats hint on 401/407/429: a body token is server-authored
//! data on a channel that already denied credentials or throttled us. A 401
//! whose body says `rate_limit` is a lying or misconfigured proxy — retrying
//! it would re-send secrets over an unauthenticated channel and burn
//! backoff sleeps for nothing. A 429 whose body says `invalid_api_key`
//! still got a genuine rate-limit response from an endpoint we reached with
//! valid-enough credentials to be throttled; rate limits stay retryable
//! with backoff. Hints only override where providers genuinely differ:
//! 400 and 403 (OpenAI 400s bad keys with `code: invalid_api_key`; Google
//! 403s permission boundaries with `PERMISSION_DENIED` — permission is NOT
//! auth, and quota 403s with `RESOURCE_EXHAUSTED` are rate-limit class).
//!
//! Hint scanning is STRUCTURED ONLY: JSON fields `error.type`,
//! `error.code` and `error.status` (OpenAI/Anthropic/Google shapes), key
//! and value matching case-insensitive, numeric codes stringified. Message
//! text is NEVER scanned — a message saying "invalid api key" proves
//! nothing.

use faktor_core::error::{Error, ErrorKind};
use serde_json::Value;

use crate::{ProviderError, ProviderErrorKind};

/// A hint token classified into its semantic family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorHint {
    Auth,
    Permission,
    RateLimited,
}

/// Fold a wire token for membership: lowercase, alphanumerics only, so
/// `PERMISSION_DENIED`, `permission_denied`, `Permission-Denied` and
/// `permission denied` all compare equal.
fn fold(w: &str) -> String {
    w.trim()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The auth-family token set: credential rejection, never message text.
fn is_auth_token(folded: &str) -> bool {
    matches!(
        folded,
        "authenticationerror"
            | "invalidapikey"
            | "invalidkey"
            | "unauthorized"
            | "authentication"
            | "auth"
            | "autherror"
            | "apikeyinvalid"
            | "apikeynotfound"
            | "apikeyrequired"
            | "missingapikey"
            | "unauthenticated"
            | "invalidcredentials"
            | "accesstokeninvalid"
            | "authenticationfailed"
    ) || folded.starts_with("authentication_")
        || folded.starts_with("apikey")
}

/// The permission-family token set (Google `PERMISSION_DENIED`, Anthropic
/// `permission_error`, ...). The caller was authenticated but not allowed.
fn is_permission_token(folded: &str) -> bool {
    folded == "forbidden"
        || folded == "accessdenied"
        || folded == "notallowed"
        || folded.starts_with("permission")
}

/// The rate-limit-family token set (incl. Google quota denials that ride
/// 403: `RESOURCE_EXHAUSTED`, `quota_exceeded`, ...).
fn is_rate_limit_token(folded: &str) -> bool {
    folded.starts_with("ratelimit")
        || folded == "toomanyrequests"
        || folded.starts_with("toomany")
        || folded.starts_with("quota")
        || folded.starts_with("resourceexhausted")
        || folded.starts_with("throttl")
        || folded == "requestslimitexceeded"
}

/// Classify one normalized wire token into its family.
pub fn hint_kind(hint: &str) -> Option<ErrorHint> {
    let folded = fold(hint);
    if folded.is_empty() {
        return None;
    }
    // Precedence when a body carries several structured fields: an auth
    // token beats a permission token beats a rate-limit token (a request
    // whose code says the credentials are invalid is an auth failure no
    // matter what the status field claims).
    if is_auth_token(&folded) {
        Some(ErrorHint::Auth)
    } else if is_permission_token(&folded) {
        Some(ErrorHint::Permission)
    } else if is_rate_limit_token(&folded) {
        Some(ErrorHint::RateLimited)
    } else {
        None
    }
}

/// Extract the strongest structured hint from an error body. Returns the
/// normalized token (`invalid_api_key`, `PERMISSION_DENIED`, ...) of the
/// highest-precedence classified field, or `None`.
///
/// Recognized shapes (never the message text):
/// - OpenAI/Anthropic: `{"error": {"type": "authentication_error" | "code": "invalid_api_key"}}`
/// - Google: `{"error": {"status": "PERMISSION_DENIED" | "code": 7}}`
/// - Google ErrorInfo detail rows (400-level auth/quota): `{"error": {"details": [{"reason": "API_KEY_INVALID"}]}}`
/// - SSE error events: `{"type": "error", "error": {"code": ...}}`
pub fn body_error_hint(body: &str) -> Option<String> {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return None;
    };
    let error = find_ci(&value, "error")?;
    let error = error.as_object()?;
    let mut best: Option<(u8, String)> = None;
    let mut push = |rank: u8, token: &str| {
        if best.as_ref().is_none_or(|(r, _)| rank > *r) {
            best = Some((rank, fold(token)));
        }
    };
    // Direct structured fields: error.type / error.code / error.status.
    for field in ["code", "status", "type"] {
        let Some(raw) = error
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(field))
            .map(|(_, v)| v)
        else {
            continue;
        };
        let text = match raw {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            _ => continue,
        };
        if let Some(kind) = hint_kind(&text) {
            push(rank(kind), &text);
        }
    }
    // Google ErrorInfo rows: error.details[].reason (structured, never the
    // message text) — a 400 carrying API_KEY_INVALID there is an auth
    // failure, a RATE_LIMIT_EXCEEDED row is rate-limit class.
    if let Some(details) = error
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("details"))
        .map(|(_, v)| v)
    {
        if let Some(rows) = details.as_array() {
            for row in rows {
                let Some(inner) = row.as_object() else {
                    continue;
                };
                let Some(Value::String(reason)) = inner
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case("reason"))
                    .map(|(_, v)| v)
                else {
                    continue;
                };
                if let Some(kind) = hint_kind(reason) {
                    push(rank(kind), reason);
                }
            }
        }
    }
    best.map(|(_, token)| token)
}

fn rank(kind: ErrorHint) -> u8 {
    match kind {
        ErrorHint::Auth => 3,
        ErrorHint::Permission => 2,
        ErrorHint::RateLimited => 1,
    }
}

fn find_ci<'v>(value: &'v Value, key: &str) -> Option<&'v Value> {
    let obj = value.as_object()?;
    obj.iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v)
}

/// THE classifier: HTTP status → typed core kind, with the documented
/// 400/403 hint overrides. Every adapter transport error path must call
/// this — no locally duplicated status chains anywhere.
pub fn classify_http(status: u16, body_hint: Option<&str>) -> ErrorKind {
    let hint = body_hint.and_then(hint_kind);
    match status {
        // Credentials were rejected: NEVER retryable. Auth wins over any
        // body hint — see the module docs.
        401 | 407 => ErrorKind::Auth,
        // 403: the server authenticated the request but refused it. The
        // hint decides the family — permission is NOT auth (Google
        // PERMISSION_DENIED), quota exhaustion is rate-limit class; a bare
        // 403 (or an auth token) is treated as an auth denial exactly like
        // the legacy adapters did.
        403 => match hint {
            Some(ErrorHint::Permission) => ErrorKind::Permission,
            Some(ErrorHint::RateLimited) => ErrorKind::RateLimited,
            _ => ErrorKind::Auth,
        },
        // 400 with an auth/permission token: some providers 400 instead of
        // 401 on bad keys (OpenAI-style `code: invalid_api_key`); the hint
        // overrides the status. No hint: malformed request.
        400 => match hint {
            Some(ErrorHint::Auth) | Some(ErrorHint::Permission) => ErrorKind::Auth,
            _ => ErrorKind::Malformed,
        },
        404 => ErrorKind::NotFound,
        409 => ErrorKind::Conflict,
        422 | 501 => ErrorKind::Unsupported,
        // Rate limits stay retryable with backoff; the status wins over any
        // hint (a throttled channel proved it reached a live endpoint).
        429 => ErrorKind::RateLimited,
        408 | 425 | 504 => ErrorKind::Timeout,
        // 5xx are infrastructure failures: retryable (legacy semantics).
        500..=599 => ErrorKind::Provider {
            code: status.to_string(),
            retryable: true,
        },
        // Every other non-success status is a client-class failure: typed
        // kinds above where the contract has one, otherwise a non-retryable
        // provider error.
        401..=499 => ErrorKind::Provider {
            code: status.to_string(),
            retryable: false,
        },
        _ => ErrorKind::Provider {
            code: status.to_string(),
            retryable: false,
        },
    }
}

/// Classify a body hint with NO status context (SSE `error` events on an
/// otherwise-2xx stream, proxies that carry auth errors inside the stream).
/// `None` when the hint is not a typed family — callers keep their generic
/// non-retryable kind then.
pub fn classify_hint_only(body_hint: Option<&str>) -> Option<ErrorKind> {
    match body_hint.and_then(hint_kind) {
        Some(ErrorHint::Auth) => Some(ErrorKind::Auth),
        Some(ErrorHint::Permission) => Some(ErrorKind::Permission),
        Some(ErrorHint::RateLimited) => Some(ErrorKind::RateLimited),
        None => None,
    }
}

/// Build the transport error envelope for a classified kind. The envelope
/// mirrors the kind's retryability exactly, so a state-aware retry loop
/// that consults `ProviderError::retryable` inherits the taxonomy.
pub fn provider_error_for(
    kind: ErrorKind,
    code: impl Into<String>,
    message: impl Into<String>,
) -> ProviderError {
    let envelope = match &kind {
        ErrorKind::Auth => ProviderErrorKind::Auth,
        ErrorKind::Permission => ProviderErrorKind::Permission,
        ErrorKind::RateLimited => ProviderErrorKind::RateLimited,
        ErrorKind::Timeout => ProviderErrorKind::Timeout,
        ErrorKind::Network => ProviderErrorKind::Network,
        ErrorKind::NotFound => ProviderErrorKind::NotFound,
        ErrorKind::Conflict => ProviderErrorKind::Conflict,
        ErrorKind::Unsupported => ProviderErrorKind::Unsupported,
        ErrorKind::Malformed => ProviderErrorKind::Malformed,
        ErrorKind::Provider {
            retryable: true, ..
        } => ProviderErrorKind::Server,
        // Oversized/Cancelled/Store/Internal/Deadlock/InvalidState and
        // non-retryable Provider kinds all fold to the non-retryable
        // BadRequest envelope (the classifier never emits them).
        _ => ProviderErrorKind::BadRequest,
    };
    ProviderError::with_code(envelope, code, message)
}

/// Build a typed core error from a classified kind (the non-streaming
/// adapter paths — ollama discovery/probing — return `faktor_core::Error`
/// directly).
pub fn core_error_for(kind: ErrorKind, message: impl Into<String>) -> Error {
    Error::new(kind, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind_for(status: u16, hint: Option<&str>) -> ErrorKind {
        classify_http(status, hint)
    }

    #[test]
    fn classifier_matrix_status_x_hint() {
        use ErrorKind as K;
        // Auth rows: 401/407 win over EVERY hint, even a rate-limit body.
        assert_eq!(kind_for(401, None), K::Auth);
        assert_eq!(kind_for(401, Some("rate_limit")), K::Auth);
        assert_eq!(kind_for(401, Some("invalid_api_key")), K::Auth);
        assert_eq!(kind_for(407, None), K::Auth);
        assert_eq!(kind_for(407, Some("rate_limit_exceeded")), K::Auth);
        // 403: hint decides; bare 403 is auth (legacy semantics).
        assert_eq!(kind_for(403, None), K::Auth);
        assert_eq!(kind_for(403, Some("invalid_api_key")), K::Auth);
        assert_eq!(kind_for(403, Some("authentication_error")), K::Auth);
        // Permission is NOT auth (Google PERMISSION_DENIED boundary).
        assert_eq!(kind_for(403, Some("PERMISSION_DENIED")), K::Permission);
        assert_eq!(kind_for(403, Some("permission_denied")), K::Permission);
        assert_eq!(kind_for(403, Some("permission_error")), K::Permission);
        assert_eq!(kind_for(403, Some("forbidden")), K::Permission);
        assert_eq!(kind_for(403, Some("access_denied")), K::Permission);
        // Google quota denials ride 403 with a rate-limit code.
        assert_eq!(kind_for(403, Some("RESOURCE_EXHAUSTED")), K::RateLimited);
        assert_eq!(kind_for(403, Some("quota_exceeded")), K::RateLimited);
        // 400: auth hint overrides the status (OpenAI code: invalid_api_key
        // on a 400 body); no hint is a malformed request.
        assert_eq!(kind_for(400, None), K::Malformed);
        assert_eq!(kind_for(400, Some("invalid_api_key")), K::Auth);
        assert_eq!(kind_for(400, Some("authentication_error")), K::Auth);
        assert_eq!(kind_for(400, Some("unauthorized")), K::Auth);
        assert_eq!(kind_for(400, Some("permission")), K::Auth);
        assert_eq!(kind_for(400, Some("rate_limit")), K::Malformed);
        // Typed 4xx.
        assert_eq!(kind_for(404, None), K::NotFound);
        assert_eq!(kind_for(409, None), K::Conflict);
        assert_eq!(kind_for(422, None), K::Unsupported);
        assert_eq!(kind_for(501, None), K::Unsupported);
        // Rate limits: status wins over every hint.
        assert_eq!(kind_for(429, None), K::RateLimited);
        assert_eq!(kind_for(429, Some("invalid_api_key")), K::RateLimited);
        // Timeout-ish retryable rows (408/425 stay, 504 keeps legacy).
        assert_eq!(kind_for(408, None), K::Timeout);
        assert_eq!(kind_for(425, None), K::Timeout);
        assert_eq!(kind_for(504, None), K::Timeout);
        // 5xx retryable, other 4xx not.
        assert!(kind_for(500, None).is_retryable());
        assert!(kind_for(503, None).is_retryable());
        assert!(kind_for(529, None).is_retryable());
        assert_eq!(
            kind_for(500, Some("invalid_api_key")),
            K::Provider {
                code: "500".into(),
                retryable: true
            }
        );
        assert_eq!(
            kind_for(405, None),
            K::Provider {
                code: "405".into(),
                retryable: false
            }
        );
        assert_eq!(
            kind_for(418, None),
            K::Provider {
                code: "418".into(),
                retryable: false
            }
        );
        // Retryability follows the kind: Auth/typed 4xx are terminal.
        assert!(!K::Auth.is_retryable());
        assert!(!K::Permission.is_retryable());
        assert!(!K::Unsupported.is_retryable());
        assert!(!K::NotFound.is_retryable());
        assert!(!K::Conflict.is_retryable());
        assert!(!K::Malformed.is_retryable());
    }

    #[test]
    fn hint_scanning_is_structured_and_never_reads_message_text() {
        // OpenAI 401/400 shapes.
        assert_eq!(
            body_error_hint(r#"{"error":{"type":"authentication_error","message":"bad key"}}"#)
                .as_deref(),
            Some("authenticationerror")
        );
        assert_eq!(
            body_error_hint(
                r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error","code":"invalid_api_key"}}"#
            )
            .as_deref(),
            Some("invalidapikey"),
            "error.code wins over an unclassified error.type"
        );
        // Anthropic 403.
        assert_eq!(
            body_error_hint(r#"{"type":"error","error":{"type":"permission_error"}}"#).as_deref(),
            Some("permissionerror")
        );
        // Google shapes: status field, and numeric code 7 stringified.
        assert_eq!(
            body_error_hint(
                r#"{"error":{"code":7,"message":"nope","status":"PERMISSION_DENIED"}}"#
            )
            .as_deref(),
            Some("permissiondenied")
        );
        // A message that LOOKS like an auth error must NOT be scanned.
        assert_eq!(
            body_error_hint(r#"{"error":{"message":"invalid api key supplied"}}"#),
            None,
            "message text is never a hint"
        );
        // Non-JSON bodies, garbage and plain strings yield no hint.
        assert_eq!(body_error_hint("401 Unauthorized"), None);
        assert_eq!(body_error_hint(""), None);
        assert_eq!(body_error_hint(r#"{"error":"model not found"}"#), None);
        assert_eq!(body_error_hint("{not json"), None);
        // Case-insensitive keys.
        assert_eq!(
            body_error_hint(r#"{"ERROR":{"CODE":"INVALID_API_KEY"}}"#).as_deref(),
            Some("invalidapikey")
        );
        // Hostile mixed keys with conflicting tokens: auth wins.
        assert_eq!(
            body_error_hint(
                r#"{"error":{"status":"RESOURCE_EXHAUSTED","type":"authentication_error"}}"#
            )
            .as_deref(),
            Some("authenticationerror")
        );
    }

    #[test]
    fn envelope_maps_kind_retryability_exactly() {
        for (status, hint) in [
            (401, None),
            (403, None),
            (403, Some("PERMISSION_DENIED")),
            (400, Some("invalid_api_key")),
            (404, None),
            (409, None),
            (422, None),
            (405, None),
        ] {
            let kind = classify_http(status, hint);
            let err = provider_error_for(kind.clone(), status.to_string(), "boom");
            assert_eq!(
                err.retryable,
                kind.is_retryable(),
                "{status} {hint:?}: envelope retryability must mirror the kind"
            );
            assert!(!err.retryable, "{status} {hint:?} must be terminal");
            let code = status.to_string();
            assert_eq!(err.code.as_deref(), Some(code.as_str()));
        }
        // Retryable rows keep the envelope retryable.
        for (status, hint) in [(429, None), (500, None), (503, None), (408, None)] {
            let kind = classify_http(status, hint);
            let err = provider_error_for(kind.clone(), status.to_string(), "boom");
            assert!(err.retryable, "{status} {hint:?} must stay retryable");
            let code = status.to_string();
            assert_eq!(err.code.as_deref(), Some(code.as_str()));
        }
        // Kinds surface typed through the envelope.
        let err = provider_error_for(ErrorKind::Auth, "401", "k");
        assert_eq!(err.kind, ProviderErrorKind::Auth);
        let err = provider_error_for(ErrorKind::Permission, "403", "k");
        assert_eq!(err.kind, ProviderErrorKind::Permission);
        let err = provider_error_for(ErrorKind::Unsupported, "422", "k");
        assert_eq!(err.kind, ProviderErrorKind::Unsupported);
        let err = provider_error_for(ErrorKind::NotFound, "404", "k");
        assert_eq!(err.kind, ProviderErrorKind::NotFound);
        let err = provider_error_for(ErrorKind::Conflict, "409", "k");
        assert_eq!(err.kind, ProviderErrorKind::Conflict);
        let err = provider_error_for(ErrorKind::RateLimited, "429", "k");
        assert_eq!(err.kind, ProviderErrorKind::RateLimited);
    }

    #[test]
    fn core_error_carries_classified_kind_and_code() {
        let kind = classify_http(401, None);
        let err = core_error_for(kind, "invalid api key");
        assert_eq!(err.kind, ErrorKind::Auth);
        assert_eq!(err.code(), "auth_required");
        assert!(!err.retryable);
        let err = core_error_for(classify_http(422, None), "unsupported");
        assert_eq!(err.code(), "unsupported");
        assert!(!err.retryable);
    }

    #[test]
    fn hint_kind_token_families() {
        assert_eq!(hint_kind("authentication_error"), Some(ErrorHint::Auth));
        assert_eq!(hint_kind("Invalid_API_Key"), Some(ErrorHint::Auth));
        assert_eq!(hint_kind("unauthorized"), Some(ErrorHint::Auth));
        assert_eq!(hint_kind("PERMISSION_DENIED"), Some(ErrorHint::Permission));
        assert_eq!(hint_kind("forbidden"), Some(ErrorHint::Permission));
        assert_eq!(hint_kind("rate_limit"), Some(ErrorHint::RateLimited));
        assert_eq!(
            hint_kind("RESOURCE_EXHAUSTED"),
            Some(ErrorHint::RateLimited)
        );
        assert_eq!(
            hint_kind("rate_limit_exceeded"),
            Some(ErrorHint::RateLimited)
        );
        assert_eq!(hint_kind("too_many_requests"), Some(ErrorHint::RateLimited));
        assert_eq!(hint_kind("internal_error"), None);
        assert_eq!(hint_kind("invalid_request_error"), None);
        assert_eq!(hint_kind(""), None);
        assert_eq!(hint_kind("  "), None);
    }

    #[test]
    fn hint_only_classification_for_sse_error_events() {
        assert_eq!(
            classify_hint_only(Some("invalid_api_key")),
            Some(ErrorKind::Auth)
        );
        assert_eq!(
            classify_hint_only(Some("PERMISSION_DENIED")),
            Some(ErrorKind::Permission)
        );
        assert_eq!(
            classify_hint_only(Some("rate_limit_exceeded")),
            Some(ErrorKind::RateLimited)
        );
        assert_eq!(classify_hint_only(None), None);
        assert_eq!(classify_hint_only(Some("internal_error")), None);
        assert_eq!(classify_hint_only(Some("")), None);
    }
}
