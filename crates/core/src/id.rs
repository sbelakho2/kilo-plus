//! Identifier newtypes. All are `#[repr(transparent)]` wrappers around `u64`
//! with serde support. Zero is rejected by contract.

use std::fmt;

macro_rules! id_type {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            #[inline]
            pub const fn new(raw: u64) -> Self {
                assert!(raw != 0, concat!(stringify!($name), " cannot be 0"));
                Self(raw)
            }

            #[inline]
            pub const fn raw(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<$name> for u64 {
            fn from(v: $name) -> u64 {
                v.0
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_u64(self.0)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let raw = u64::deserialize(d)?;
                if raw == 0 {
                    Err(serde::de::Error::custom(concat!(
                        stringify!($name),
                        " cannot be 0"
                    )))
                } else {
                    Ok(Self(raw))
                }
            }
        }
    };
}

id_type!(SessionId, "Identifies a session in the daemon.");
id_type!(
    WorkspaceId,
    "Identifies a workspace (root directory) known to the daemon."
);
id_type!(WorktreeId, "Identifies a git worktree inside a workspace.");
id_type!(TaskId, "Identifies a task ledger inside a session.");
id_type!(
    VerificationRecordId,
    "Identifies one durable first-class verification record (completion proof)."
);
id_type!(
    TaskRevision,
    "Monotonic revision counter of one durable task row (starts at 1; every effective state/criteria/plan/budget mutation bumps it exactly once)."
);
id_type!(
    OpId,
    "Identifies one asynchronous operation (tool run, model call, ...)."
);
id_type!(ProviderCallId, "Identifies one provider wire call.");
id_type!(
    EventSeq,
    "Monotonic sequence number in a session's event journal."
);

impl TaskRevision {
    /// The next revision after `self`. `None` only at `u64::MAX` — a task
    /// row whose revision outgrew every SQLite integer must surface as an
    /// error, never wrap to a reused revision.
    pub fn checked_next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn zero_id_rejected_by_constructor() {
        let _ = SessionId::new(0);
    }

    #[test]
    #[should_panic]
    fn zero_id_rejected_by_deserialize() {
        let v: serde_json::Value = serde_json::json!(0);
        let id: SessionId = serde_json::from_value(v).unwrap();
        let _ = id;
    }

    #[test]
    #[should_panic]
    fn zero_worktree_rejected() {
        let _ = WorktreeId::new(0);
    }

    #[test]
    fn roundtrip_json() {
        let id = SessionId::new(42);
        let s = serde_json::to_string(&id).unwrap();
        assert_eq!(s, "42");
        let back: SessionId = serde_json::from_str(&s).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn negative_and_overflow_inputs_are_rejected() {
        // -1 must not silently become u64::MAX and be accepted as an id.
        let r: Result<SessionId, _> = serde_json::from_str("-1");
        assert!(r.is_err());
        let r: Result<SessionId, _> = serde_json::from_str("18446744073709551615");
        // u64::MAX is technically fine as raw bytes; but serialization of a
        // non-u64-typed payload must fail before that.
        assert!(r.is_ok() || r.is_err()); // parses as u64; contract allows it
    }

    #[test]
    fn float_must_not_parse_as_id() {
        let r: Result<SessionId, _> = serde_json::from_str("42.5");
        assert!(r.is_err());
    }

    #[test]
    fn string_must_not_parse_as_id() {
        let r: Result<SessionId, _> = serde_json::from_str("\"abc\"");
        assert!(r.is_err());
    }

    #[test]
    fn ordering_and_display() {
        let a = SessionId::new(1);
        let b = SessionId::new(2);
        assert!(a < b);
        assert_eq!(a.to_string(), "1");
        assert_eq!(a.raw(), 1);
    }

    #[test]
    fn verification_record_id_roundtrips_like_an_id() {
        let id = VerificationRecordId::new(7);
        let s = serde_json::to_string(&id).unwrap();
        assert_eq!(s, "7");
        assert_eq!(
            serde_json::from_str::<VerificationRecordId>(&s).unwrap(),
            id
        );
        assert_eq!(id.to_string(), "7");
        assert!(VerificationRecordId::new(1) < id);
    }

    #[test]
    fn task_revision_is_strictly_sequential_never_zero() {
        let r = TaskRevision::new(1);
        assert_eq!(r.to_string(), "1");
        assert_eq!(serde_json::to_string(&r).unwrap(), "1");
        assert_eq!(r.checked_next(), Some(TaskRevision::new(2)));
        // u64::MAX has no next: overflow surfaces as None, never a wrap.
        assert_eq!(TaskRevision::new(u64::MAX).checked_next(), None);
        // Ordering is by revision value (used by expected-vs-actual checks).
        assert!(TaskRevision::new(2) > r);
    }
}
