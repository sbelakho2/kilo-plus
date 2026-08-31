//! Local auth token: random per start, constant-time comparison.

use rand::Rng;

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
    let expected = token.as_str().as_bytes();
    let actual = bearer.as_bytes();
    if expected.len() != actual.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in expected.iter().zip(actual.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_random_and_hex() {
        let a = AuthToken::generate();
        let b = AuthToken::generate();
        assert_ne!(a, b);
        assert_eq!(a.as_str().len(), 64);
        assert!(a.as_str().bytes().all(|c| c.is_ascii_hexdigit()));
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
        assert!(!check_bearer(&t, Some(&format!("Bearer {}", &t.as_str()[..32]))));
        // Trailing garbage must not match.
        assert!(!check_bearer(&t, Some(&format!("Bearer {} extra", t.as_str()))));
    }

    #[test]
    fn token_survives_clone_and_display() {
        let t = AuthToken::generate();
        let t2 = t.clone();
        assert_eq!(t, t2);
        assert_eq!(format!("{}", t.as_str()), t.as_str());
    }
}
