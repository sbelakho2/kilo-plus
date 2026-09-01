//! Local auth: the frontend generates a 64-hex `KILO_SERVER_PASSWORD` and
//! passes it to the daemon via env; the daemon never prints it. The frozen
//! v7.5.6 extension authenticates every request (including `/global/health`)
//! with `Authorization: Basic base64("kilo:" + KILO_SERVER_PASSWORD)`
//! (`ServerPassword::check_authorization`). The Kilo+-native forms
//! (`Authorization: Bearer <password>` and `x-kilo-server-password:
//! <password>`) remain accepted, and the legacy per-start `AuthToken` keeps
//! the old tests working.

use base64::Engine as _;
use rand::Rng;

/// The server password: read from `KILO_SERVER_PASSWORD` or generated.
/// Comparisons are constant-time; the value is never serialized or logged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerPassword(String);

impl ServerPassword {
    /// Read `KILO_SERVER_PASSWORD` from the environment. When missing (or
    /// empty/whitespace), generate a random 64-hex secret and log it at
    /// DEBUG — never stdout.
    pub fn from_env() -> Self {
        match std::env::var("KILO_SERVER_PASSWORD") {
            Ok(v) if !v.trim().is_empty() => Self(v),
            _ => {
                let pw = Self::generate();
                tracing::debug!(
                    "KILO_SERVER_PASSWORD not set; generated ephemeral server password"
                );
                pw
            }
        }
    }

    /// A random 64-hex secret.
    pub fn generate() -> Self {
        let mut rng = rand::rng();
        let bytes: [u8; 32] = rng.random();
        Self(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check one `Authorization` header value. Accepted, in order:
    ///
    /// 1. `Basic <base64>` — the frozen v7.5.6 form. The payload must decode
    ///    (headers longer than [`MAX_AUTH_HEADER_BYTES`] are rejected before
    ///    decoding), split at the *first* `:`, the username must be exactly
    ///    `kilo`, and the password is compared constant-time against the
    ///    secret. Trailing junk is rejected: the whole payload must decode.
    /// 2. `Bearer <password>` — Kilo+-native, retained.
    ///
    /// Anything else (missing header, garbage scheme, malformed base64,
    /// wrong username/password) is rejected.
    pub fn check_authorization(&self, header: Option<&str>) -> bool {
        let Some(header) = header else {
            return false;
        };
        if header.len() > MAX_AUTH_HEADER_BYTES {
            return false;
        }
        if let Some(encoded) = header.strip_prefix("Basic ") {
            let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
                return false;
            };
            let Some(sep) = decoded.iter().position(|&b| b == b':') else {
                return false;
            };
            if &decoded[..sep] != b"kilo" {
                return false;
            }
            return ct_eq(self.as_str().as_bytes(), &decoded[sep + 1..]);
        }
        if let Some(bearer) = header.strip_prefix("Bearer ") {
            return ct_eq(self.as_str().as_bytes(), bearer.as_bytes());
        }
        false
    }
}

/// The bound on one `Authorization` header value (bounded everything): a
/// header larger than this is rejected without even attempting a decode.
pub const MAX_AUTH_HEADER_BYTES: usize = 4096;

/// Constant-time comparison of two fixed-length byte strings.
fn ct_eq(expected: &[u8], actual: &[u8]) -> bool {
    if expected.len() != actual.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in expected.iter().zip(actual.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// Accepts the password from either `Authorization: Bearer <pw>` or
/// `x-kilo-server-password: <pw>`. No header → false; any mismatch → false.
pub fn check_password(
    password: &ServerPassword,
    authorization: Option<&str>,
    x_kilo_server_password: Option<&str>,
) -> bool {
    let expected = password.as_str().as_bytes();
    if let Some(bearer) = authorization.and_then(|h| h.strip_prefix("Bearer ")) {
        if ct_eq(expected, bearer.as_bytes()) {
            return true;
        }
    }
    if let Some(header) = x_kilo_server_password {
        if ct_eq(expected, header.as_bytes()) {
            return true;
        }
    }
    false
}

/// Random per-start bearer token. Never persisted (a restart invalidates
/// stale UI connections, which is the point of local auth).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthToken(String);

impl AuthToken {
    pub fn generate() -> Self {
        let mut rng = rand::rng();
        let bytes: [u8; 32] = rng.random();
        Self(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Constant-time comparison (no length leakage beyond the token's own fixed
/// length).
pub fn check_bearer(token: &AuthToken, header: Option<&str>) -> bool {
    let Some(header) = header else {
        return false;
    };
    let Some(bearer) = header.strip_prefix("Bearer ") else {
        return false;
    };
    ct_eq(token.as_str().as_bytes(), bearer.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn tokens_are_random_and_hex() {
        let a = AuthToken::generate();
        let b = AuthToken::generate();
        assert_ne!(a, b);
        assert_eq!(a.as_str().len(), 64);
        assert!(a.as_str().bytes().all(|c| c.is_ascii_hexdigit()));
        let p = ServerPassword::generate();
        assert_eq!(p.as_str().len(), 64);
        assert!(p.as_str().bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(p, ServerPassword::generate());
    }

    #[test]
    fn bearer_matching_is_exact_and_strict() {
        let t = AuthToken::generate();
        assert!(check_bearer(&t, Some(&format!("Bearer {}", t.as_str()))));
        assert!(!check_bearer(&t, None));
        assert!(!check_bearer(&t, Some("")));
        assert!(!check_bearer(&t, Some(t.as_str())), "missing Bearer prefix");
        assert!(!check_bearer(&t, Some("bearer x")), "case-sensitive scheme");
        assert!(!check_bearer(&t, Some("Bearer x")), "wrong token");
        assert!(!check_bearer(&t, Some(&format!("Bearer {}/1", t.as_str()))));
        // Prefix-only headers must not match.
        assert!(!check_bearer(
            &t,
            Some(&format!("Bearer {}", &t.as_str()[..32]))
        ));
        // Trailing garbage must not match.
        assert!(!check_bearer(
            &t,
            Some(&format!("Bearer {} extra", t.as_str()))
        ));
    }

    #[test]
    fn token_survives_clone_and_display() {
        let t = AuthToken::generate();
        let t2 = t.clone();
        assert_eq!(t, t2);
        assert_eq!(t.as_str().to_string(), t.as_str());
    }

    #[test]
    fn server_password_reads_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("KILO_SERVER_PASSWORD", "a".repeat(64));
        let pw = ServerPassword::from_env();
        assert_eq!(pw.as_str(), "a".repeat(64));
        std::env::remove_var("KILO_SERVER_PASSWORD");
    }

    #[test]
    fn server_password_generates_when_env_missing_or_empty() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("KILO_SERVER_PASSWORD");
        let pw = ServerPassword::from_env();
        assert_eq!(pw.as_str().len(), 64);
        assert!(pw.as_str().bytes().all(|c| c.is_ascii_hexdigit()));
        // Empty and whitespace-only values are treated as missing (a secret
        // that is empty is no secret at all).
        std::env::set_var("KILO_SERVER_PASSWORD", "");
        let pw = ServerPassword::from_env();
        assert_eq!(pw.as_str().len(), 64);
        std::env::set_var("KILO_SERVER_PASSWORD", "   ");
        let pw = ServerPassword::from_env();
        assert_eq!(pw.as_str().len(), 64);
        std::env::remove_var("KILO_SERVER_PASSWORD");
    }

    #[test]
    fn password_accepted_in_both_header_forms() {
        let pw = ServerPassword::generate();
        // Bearer form.
        let auth = Some(&format!("Bearer {}", pw.as_str())[..]);
        assert!(check_password(&pw, auth, None));
        // x-kilo-server-password form.
        assert!(check_password(&pw, None, Some(pw.as_str())));
        // Both forms can even be sent together.
        assert!(check_password(&pw, auth, Some(pw.as_str())));
    }

    #[test]
    fn password_rejects_missing_wrong_and_partial() {
        let pw = ServerPassword::generate();
        assert!(!check_password(&pw, None, None), "no header");
        assert!(!check_password(&pw, Some(""), None));
        assert!(!check_password(&pw, None, Some("")));
        assert!(!check_password(&pw, Some("Bearer wrong"), None));
        assert!(!check_password(
            &pw,
            Some("Bearer wrong"),
            Some("also wrong")
        ));
        // Missing Bearer prefix on the auth header.
        assert!(!check_password(&pw, Some(pw.as_str()), None));
        // Scheme is case-sensitive.
        assert!(!check_password(
            &pw,
            Some(&format!("bearer {}", pw.as_str())),
            None
        ));
        // Prefix of the real password must not match.
        assert!(!check_password(
            &pw,
            Some(&format!("Bearer {}", &pw.as_str()[..32])),
            None
        ));
        // Trailing garbage must not match.
        assert!(!check_password(
            &pw,
            Some(&format!("Bearer {} extra", pw.as_str())),
            None
        ));
        // Wrong x-kilo form.
        assert!(!check_password(&pw, None, Some("wrong")));
        // Header containing the password with surrounding whitespace is
        // rejected (headers are exact).
        assert!(!check_password(
            &pw,
            None,
            Some(&format!(" {}", pw.as_str()))
        ));
    }

    #[test]
    fn password_and_legacy_token_are_independent() {
        // The old token flow must keep working alongside the password flow.
        let token = AuthToken::generate();
        let pw = ServerPassword::generate();
        assert!(check_bearer(
            &token,
            Some(&format!("Bearer {}", token.as_str()))
        ));
        assert!(!check_password(
            &pw,
            Some(&format!("Bearer {}", token.as_str())),
            None
        ));
        assert!(!check_bearer(
            &token,
            Some(&format!("Bearer {}", pw.as_str()))
        ));
    }

    fn basic(username: &str, password: &str) -> String {
        use base64::Engine as _;
        format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode(format!("{username}:{password}"))
        )
    }

    #[test]
    fn basic_auth_valid_credentials_accepted() {
        let pw = ServerPassword::generate();
        let header = basic("kilo", pw.as_str());
        assert!(pw.check_authorization(Some(&header)), "{header}");
        // The password may itself contain a colon: only the FIRST colon
        // splits the credentials.
        let tricky = ServerPassword("kilo:with:colons".into());
        let header = basic("kilo", tricky.as_str());
        assert!(tricky.check_authorization(Some(&header)));
    }

    #[test]
    fn basic_auth_wrong_password_rejected() {
        let pw = ServerPassword::generate();
        let wrong = ServerPassword::generate();
        let header = basic("kilo", wrong.as_str());
        assert!(!pw.check_authorization(Some(&header)));
        // A prefix of the real password must not match.
        let header = basic("kilo", &pw.as_str()[..32]);
        assert!(!pw.check_authorization(Some(&header)));
        // Empty password credential.
        let header = basic("kilo", "");
        assert!(!pw.check_authorization(Some(&header)));
    }

    #[test]
    fn basic_auth_wrong_username_rejected() {
        let pw = ServerPassword::generate();
        for user in ["admin", "root", "kil", "kilo-", "Kilo", "kilop", ""] {
            let header = basic(user, pw.as_str());
            assert!(
                !pw.check_authorization(Some(&header)),
                "username {user:?} must be rejected"
            );
        }
        // No colon at all: nothing to split, rejected.
        let header = format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode(pw.as_str())
        );
        assert!(!pw.check_authorization(Some(&header)));
    }

    #[test]
    fn basic_auth_malformed_base64_rejected() {
        let pw = ServerPassword::generate();
        for bad in [
            "Basic !!!not-base64!!!",
            "Basic a2lsbzpwYXNz",  // truncated (missing padding)
            "Basic a2lsbzpwYXNz=", // wrong padding
            "Basic \u{00a0}\u{00a0}",
            "Basic -----",
        ] {
            assert!(
                !pw.check_authorization(Some(bad)),
                "{bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn basic_auth_garbage_and_missing_headers_rejected() {
        let pw = ServerPassword::generate();
        assert!(!pw.check_authorization(None));
        assert!(!pw.check_authorization(Some("")));
        assert!(!pw.check_authorization(Some("Basic")));
        assert!(!pw.check_authorization(Some("basic a2lsbzp4")));
        assert!(!pw.check_authorization(Some("Digest a2lsbzp4")));
        assert!(!pw.check_authorization(Some(pw.as_str())));
        assert!(!pw.check_authorization(Some("Token 12345")));
        assert!(!pw.check_authorization(Some("Bearer ")));
    }

    #[test]
    fn basic_auth_huge_header_rejected() {
        let pw = ServerPassword::generate();
        let huge = format!("Basic {}", "A".repeat(MAX_AUTH_HEADER_BYTES));
        assert!(!pw.check_authorization(Some(&huge)));
        // Exactly at the bound is still rejected only when the payload is
        // invalid; a legitimately sized header is fine.
        let ok = basic("kilo", pw.as_str());
        assert!(ok.len() < MAX_AUTH_HEADER_BYTES);
        assert!(pw.check_authorization(Some(&ok)));
    }

    #[test]
    fn basic_auth_trailing_junk_rejected() {
        let pw = ServerPassword::generate();
        let good = basic("kilo", pw.as_str());
        let good_encoded = good.strip_prefix("Basic ").unwrap();
        for junk in [" extra", "==", "!!", " x", "\n", "\t"] {
            let header = format!("Basic {good_encoded}{junk}");
            assert!(
                !pw.check_authorization(Some(&header)),
                "trailing junk {junk:?} must be rejected"
            );
        }
        // Whitespace INSIDE the base64 (not trailing junk) also fails.
        let header = format!("Basic {} {}", &good_encoded[..10], &good_encoded[10..]);
        assert!(!pw.check_authorization(Some(&header)));
    }

    #[test]
    fn bearer_and_x_kilo_forms_still_work() {
        let pw = ServerPassword::generate();
        // Bearer through the same entry point.
        let header = format!("Bearer {}", pw.as_str());
        assert!(pw.check_authorization(Some(&header)));
        assert!(!pw.check_authorization(Some("Bearer wrong")));
        // x-kilo-server-password is a separate header; check_password covers
        // it (the Authorization entry point must NOT accept it as Basic).
        assert!(check_password(&pw, None, Some(pw.as_str())));
        assert!(!pw.check_authorization(Some(pw.as_str())));
    }

    #[test]
    fn basic_and_bearer_and_legacy_token_are_independent() {
        let pw = ServerPassword::generate();
        let token = AuthToken::generate();
        // A legacy token is NOT the password: it fails every password path.
        let header = format!("Bearer {}", token.as_str());
        assert!(!pw.check_authorization(Some(&header)));
        let header = basic("kilo", token.as_str());
        assert!(!pw.check_authorization(Some(&header)));
        // And the token path still accepts its own bearer.
        let header = format!("Bearer {}", token.as_str());
        assert!(check_bearer(&token, Some(&header)));
    }
}
