//! API errors: the frozen error-code contract. `code` strings and their HTTP
//! status mappings are locked by golden tests.

use kilop_core::error::ErrorKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    /// Frozen machine-readable code.
    pub code: &'static str,
    pub message: String,
    pub http_status: u16,
    pub retryable: bool,
}

impl ApiError {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }
        })
    }
}

/// Map core errors onto the frozen API error surface.
pub fn from_core(e: &kilop_core::error::Error) -> ApiError {
    match &e.kind {
        ErrorKind::NotFound => ApiError {
            code: "not_found",
            message: e.message.clone(),
            http_status: 404,
            retryable: false,
        },
        ErrorKind::Conflict => ApiError {
            code: "conflict",
            message: e.message.clone(),
            http_status: 409,
            retryable: false,
        },
        ErrorKind::InvalidState { .. } => ApiError {
            code: "invalid_state",
            message: e.message.clone(),
            http_status: 409,
            retryable: false,
        },
        ErrorKind::Permission => ApiError {
            code: "permission_denied",
            message: e.message.clone(),
            http_status: 403,
            retryable: false,
        },
        ErrorKind::Timeout => ApiError {
            code: "timeout",
            message: e.message.clone(),
            http_status: 504,
            retryable: true,
        },
        ErrorKind::Cancelled => ApiError {
            code: "cancelled",
            message: e.message.clone(),
            http_status: 499,
            retryable: false,
        },
        ErrorKind::Store => ApiError {
            code: "store_error",
            message: e.message.clone(),
            http_status: 500,
            retryable: true,
        },
        ErrorKind::Network => ApiError {
            code: "network_error",
            message: e.message.clone(),
            http_status: 502,
            retryable: true,
        },
        ErrorKind::Provider { code, .. } => ApiError {
            code: "provider_error",
            message: format!("provider {code}: {}", e.message),
            http_status: 502,
            retryable: e.retryable,
        },
        ErrorKind::Malformed => ApiError {
            code: "malformed",
            message: e.message.clone(),
            http_status: 400,
            retryable: false,
        },
        ErrorKind::Oversized => ApiError {
            code: "oversized",
            message: e.message.clone(),
            http_status: 413,
            retryable: false,
        },
        ErrorKind::RateLimited => ApiError {
            code: "rate_limited",
            message: e.message.clone(),
            http_status: 429,
            retryable: true,
        },
        ErrorKind::Deadlock => ApiError {
            code: "deadlock",
            message: e.message.clone(),
            http_status: 409,
            retryable: true,
        },
        ErrorKind::Internal => ApiError {
            code: "internal_error",
            message: e.message.clone(),
            http_status: 500,
            retryable: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_mapping_is_frozen() {
        let cases: &[(ErrorKind, &str, u16, bool)] = &[
            (ErrorKind::NotFound, "not_found", 404, false),
            (ErrorKind::Conflict, "conflict", 409, false),
            (
                ErrorKind::InvalidState {
                    from: kilop_core::state::AgentState::Idle,
                    to: kilop_core::state::AgentState::Completed,
                },
                "invalid_state",
                409,
                false,
            ),
            (ErrorKind::Permission, "permission_denied", 403, false),
            (ErrorKind::Timeout, "timeout", 504, true),
            (ErrorKind::Cancelled, "cancelled", 499, false),
            (ErrorKind::Store, "store_error", 500, true),
            (ErrorKind::Network, "network_error", 502, true),
            (
                ErrorKind::Provider { code: "5xx".into(), retryable: true },
                "provider_error",
                502,
                true,
            ),
            (ErrorKind::Malformed, "malformed", 400, false),
            (ErrorKind::Oversized, "oversized", 413, false),
            (ErrorKind::RateLimited, "rate_limited", 429, true),
            (ErrorKind::Deadlock, "deadlock", 409, true),
            (ErrorKind::Internal, "internal_error", 500, false),
        ];
        for (kind, code, status, retryable) in cases {
            let e = kilop_core::error::Error::new(kind.clone(), "x");
            let api = from_core(&e);
            assert_eq!(api.code, *code, "{kind:?}");
            assert_eq!(api.http_status, *status, "{kind:?}");
            assert_eq!(api.retryable, *retryable, "{kind:?}");
            let json = api.to_json();
            assert_eq!(json["error"]["code"], *code);
            assert_eq!(json["error"]["retryable"], *retryable);
        }
    }

    #[test]
    fn json_shape_has_no_extra_fields() {
        let e = kilop_core::error::Error::new(ErrorKind::NotFound, "gone");
        let api = from_core(&e);
        let json = api.to_json();
        let error = json["error"].as_object().unwrap();
        assert_eq!(error.len(), 3, "frozen: only code, message, retryable");
    }
}
