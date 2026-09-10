//! Typed, bounded evidence retrieval.
//!
//! Every read is explicit: a selector is mandatory, it is validated before any
//! bytes are produced, and there is never an implicit fallback to the whole
//! backing blob ("auto-dump") when the requested selector cannot be honored.
//! `All` is itself a selector, and it refuses unless the envelope's
//! [`RetrievalPolicy::allow_ranges`](crate::types::RetrievalPolicy) permits it.
//!
//! Retrieval reads retained backing bytes only. If the store dropped backing
//! (cap, incomplete capture, non-compactable kind), retrieval fails with
//! [`EvidenceError::Refused`] instead of substituting the compact body, so a
//! caller can always tell which bytes it actually holds.
//!
//! [`retrieve`] is scope-checked through
//! [`EvidenceStore::get_scoped`](crate::store::EvidenceStore::get_scoped):
//! `Items` never silently drops ids outside the context; any foreign id denies
//! the whole request.

use crate::store::{EvidenceAccessContext, EvidenceStore, StoredEvidence};
use crate::types::{EvidenceError, EvidenceId, RetrievalPolicy};

/// Hard cap on search hits per request, independent of the caller-supplied
/// `max_hits`. The envelope's `max_bytes` still bounds the response size.
pub const MAX_SEARCH_HITS: u32 = 256;
/// Hard cap on the number of ids one `Items` request may name.
pub const MAX_ITEMS: usize = 256;
/// Hard cap on search query length in bytes.
pub const MAX_QUERY_BYTES: usize = 4096;

/// How much of one evidence envelope to read. There is no default: a caller
/// that wants bytes must say which bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalSelector {
    /// The whole retained backing. Refused unless the envelope allows ranges;
    /// still bounded by `max_bytes`.
    All,
    /// Half-open byte range `[start, end)` into the backing. `end == len` is
    /// the last valid edge; `start == end` is an empty read.
    ByteRange { start: u64, end: u64 },
    /// Inclusive, 1-based line range `start..=end`. Lines are split on `\n`
    /// and the slices include their terminating newline when present.
    LineRange { start: u64, end: u64 },
    /// Case-sensitive substring search. Returns only matching line slices,
    /// never surrounding context, bounded by `max_hits` and the envelope's
    /// `max_bytes`.
    Search { query: String, max_hits: u32 },
    /// Explicitly enumerated evidence ids, concatenated in request order and
    /// bounded by the policy of the governing `id`.
    Items { ids: Vec<EvidenceId> },
}

/// The result of one retrieval. `bytes` is exactly what the selector selected;
/// `truncated_by_policy` is set only when the policy byte cap or the hit cap
/// dropped further matches (ranges are refused when out of bounds, never
/// silently shortened).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievedEvidence {
    pub id: EvidenceId,
    pub selector_echo: String,
    pub bytes: Vec<u8>,
    pub truncated_by_policy: bool,
}

impl RetrievedEvidence {
    /// Byte length of the retrieved payload.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Read evidence through a typed selector. The primary `id` governs the
/// response budget and must be accessible from `ctx`; for `Items`, every
/// requested id is scope-checked as well.
pub fn retrieve<S: EvidenceStore + ?Sized>(
    store: &S,
    ctx: &EvidenceAccessContext,
    id: EvidenceId,
    selector: RetrievalSelector,
) -> Result<RetrievedEvidence, EvidenceError> {
    let stored = &store.get_scoped(id, ctx)?;
    match selector {
        RetrievalSelector::All => retrieve_all(id, stored),
        RetrievalSelector::ByteRange { start, end } => retrieve_byte_range(id, stored, start, end),
        RetrievalSelector::LineRange { start, end } => retrieve_line_range(id, stored, start, end),
        RetrievalSelector::Search { query, max_hits } => {
            retrieve_search(id, stored, &query, max_hits)
        }
        RetrievalSelector::Items { ids } => retrieve_items(store, ctx, id, stored, ids),
    }
}

fn backing_of(id: EvidenceId, stored: &StoredEvidence) -> Result<&[u8], EvidenceError> {
    stored.backing.as_deref().ok_or_else(|| {
        EvidenceError::Refused(format!(
            "backing bytes for evidence {id} are not retained by the store"
        ))
    })
}

fn retrieve_all(
    id: EvidenceId,
    stored: &StoredEvidence,
) -> Result<RetrievedEvidence, EvidenceError> {
    if !stored.envelope.retrieval.allow_ranges {
        return Err(EvidenceError::Refused(format!(
            "evidence {id}: `All` requires retrieval.allow_ranges; refusing whole-blob auto-dump"
        )));
    }
    let backing = backing_of(id, stored)?;
    stored.envelope.retrieval.check_range(backing.len())?;
    Ok(RetrievedEvidence {
        id,
        selector_echo: "all".to_string(),
        bytes: backing.to_vec(),
        truncated_by_policy: false,
    })
}

fn retrieve_byte_range(
    id: EvidenceId,
    stored: &StoredEvidence,
    start: u64,
    end: u64,
) -> Result<RetrievedEvidence, EvidenceError> {
    if start > end {
        return Err(EvidenceError::Malformed(format!(
            "byte range {start}..{end} has start > end"
        )));
    }
    let backing = backing_of(id, stored)?;
    if end > backing.len() as u64 {
        return Err(EvidenceError::Oversized {
            max: backing.len(),
            actual: usize::try_from(end).unwrap_or(usize::MAX),
        });
    }
    let len = (end - start) as usize;
    stored.envelope.retrieval.check_range(len)?;
    Ok(RetrievedEvidence {
        id,
        selector_echo: format!("bytes:{start}..{end}"),
        bytes: backing[start as usize..end as usize].to_vec(),
        truncated_by_policy: false,
    })
}

fn retrieve_line_range(
    id: EvidenceId,
    stored: &StoredEvidence,
    start: u64,
    end: u64,
) -> Result<RetrievedEvidence, EvidenceError> {
    if start == 0 || end == 0 {
        return Err(EvidenceError::Malformed(
            "line ranges are 1-based; line 0 does not exist".to_string(),
        ));
    }
    if start > end {
        return Err(EvidenceError::Malformed(format!(
            "line range {start}..={end} has start > end"
        )));
    }
    let backing = backing_of(id, stored)?;
    let lines = line_slices(backing);
    if end > lines.len() as u64 {
        return Err(EvidenceError::Oversized {
            max: lines.len(),
            actual: usize::try_from(end).unwrap_or(usize::MAX),
        });
    }
    let selected = &lines[(start - 1) as usize..end as usize];
    let total: usize = selected.iter().map(|line| line.len()).sum();
    stored.envelope.retrieval.check_range(total)?;
    let mut bytes = Vec::with_capacity(total);
    for line in selected {
        bytes.extend_from_slice(line);
    }
    Ok(RetrievedEvidence {
        id,
        selector_echo: format!("lines:{start}..={end}"),
        bytes,
        truncated_by_policy: false,
    })
}

fn retrieve_search(
    id: EvidenceId,
    stored: &StoredEvidence,
    query: &str,
    max_hits: u32,
) -> Result<RetrievedEvidence, EvidenceError> {
    if query.is_empty() {
        return Err(EvidenceError::Malformed(
            "search query must not be empty".to_string(),
        ));
    }
    if query.len() > MAX_QUERY_BYTES {
        return Err(EvidenceError::Oversized {
            max: MAX_QUERY_BYTES,
            actual: query.len(),
        });
    }
    if max_hits == 0 {
        return Err(EvidenceError::Malformed(
            "search max_hits must be at least 1".to_string(),
        ));
    }
    stored.envelope.retrieval.check_search()?;
    let backing = backing_of(id, stored)?;
    let policy = stored.envelope.retrieval;
    let hit_cap = max_hits.min(MAX_SEARCH_HITS) as usize;
    let needle = query.as_bytes();
    let mut bytes = Vec::new();
    let mut hits = 0usize;
    let mut truncated_by_policy = false;
    for line in line_slices(backing) {
        if !line_contains(line, needle) {
            continue;
        }
        if hits >= hit_cap {
            truncated_by_policy = true;
            break;
        }
        let hit = strip_trailing_newline(line);
        let separator = usize::from(hits > 0);
        let projected = bytes
            .len()
            .saturating_add(separator)
            .saturating_add(hit.len());
        if projected > policy.max_bytes {
            truncated_by_policy = true;
            break;
        }
        if hits > 0 {
            bytes.push(b'\n');
        }
        bytes.extend_from_slice(hit);
        hits += 1;
    }
    Ok(RetrievedEvidence {
        id,
        selector_echo: format!("search:{query:?};max_hits={max_hits}"),
        bytes,
        truncated_by_policy,
    })
}

fn retrieve_items<S: EvidenceStore + ?Sized>(
    store: &S,
    ctx: &EvidenceAccessContext,
    id: EvidenceId,
    governing: &StoredEvidence,
    ids: Vec<EvidenceId>,
) -> Result<RetrievedEvidence, EvidenceError> {
    if ids.is_empty() {
        return Err(EvidenceError::Malformed(
            "Items selector requires at least one evidence id".to_string(),
        ));
    }
    if ids.len() > MAX_ITEMS {
        return Err(EvidenceError::Oversized {
            max: MAX_ITEMS,
            actual: ids.len(),
        });
    }
    let echo = format!(
        "items:[{}]",
        ids.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let budget = governing.envelope.retrieval.max_bytes;
    let mut bytes = Vec::new();
    for item_id in ids {
        let stored = store.get_scoped(item_id, ctx)?;
        let item = backing_of(item_id, &stored)?;
        let projected = bytes
            .len()
            .checked_add(item.len())
            .ok_or(EvidenceError::Oversized {
                max: budget,
                actual: usize::MAX,
            })?;
        if projected > budget {
            return Err(EvidenceError::Oversized {
                max: budget,
                actual: projected,
            });
        }
        bytes.extend_from_slice(item);
    }
    Ok(RetrievedEvidence {
        id,
        selector_echo: echo,
        bytes,
        truncated_by_policy: false,
    })
}

/// Split `backing` into line slices terminated by `\n` where present. A
/// trailing newline does not create an extra empty line; empty input has no
/// lines.
fn line_slices(backing: &[u8]) -> Vec<&[u8]> {
    if backing.is_empty() {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let mut start = 0usize;
    for (i, byte) in backing.iter().enumerate() {
        if *byte == b'\n' {
            lines.push(&backing[start..=i]);
            start = i + 1;
        }
    }
    if start < backing.len() {
        lines.push(&backing[start..]);
    }
    lines
}

fn line_contains(line: &[u8], needle: &[u8]) -> bool {
    needle.len() <= line.len() && line.windows(needle.len()).any(|window| window == needle)
}

fn strip_trailing_newline(line: &[u8]) -> &[u8] {
    match line.last() {
        Some(b'\n') => &line[..line.len() - 1],
        _ => line,
    }
}

/// The policy of one envelope, exposed for tests and callers that want to
/// pre-check a selector without reading.
pub fn policy_of(stored: &StoredEvidence) -> RetrievalPolicy {
    stored.envelope.retrieval
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryEvidenceStore;
    use crate::types::{
        BackingCompleteness, CompactRepresentation, Compressibility, CompressionRecord,
        EvidenceEnvelope, EvidenceKind, ProvenanceSet, ProvenanceSource,
    };
    use faktor_core::{SessionId, WorkspaceId};

    fn open(max_bytes: usize) -> RetrievalPolicy {
        RetrievalPolicy::new(true, true, max_bytes)
    }

    fn ctx() -> EvidenceAccessContext {
        EvidenceAccessContext::new(1, 2, Some(3))
    }

    fn envelope(
        id: u64,
        policy: RetrievalPolicy,
        session: u64,
        task: Option<u64>,
    ) -> EvidenceEnvelope {
        EvidenceEnvelope::new(
            EvidenceId(id),
            EvidenceKind::ProcessLog,
            SessionId::new(session),
            WorkspaceId::new(2),
            task,
            None,
            ProvenanceSet::new([ProvenanceSource::Tool]),
            Compressibility::Aggressive,
            CompactRepresentation {
                grammar: "log-v1".to_string(),
                body: "COMPACT-BODY-MUST-NEVER-LEAK-WHOLE".to_string(),
            },
            Some([5u8; 32]),
            BackingCompleteness::Complete,
            CompressionRecord::identity(0),
            policy,
        )
        .unwrap()
    }

    fn store_with(id: u64, backing: &[u8], policy: RetrievalPolicy) -> MemoryEvidenceStore {
        let mut store = MemoryEvidenceStore::new(4096);
        store
            .insert(envelope(id, policy, 1, Some(3)), Some(backing.to_vec()))
            .unwrap();
        store
    }

    fn read(
        store: &MemoryEvidenceStore,
        id: u64,
        selector: RetrievalSelector,
    ) -> Result<RetrievedEvidence, EvidenceError> {
        retrieve(store, &ctx(), EvidenceId(id), selector)
    }

    #[test]
    fn scope_isolation_denies_foreign_readers_even_knowing_the_id() {
        let store = store_with(1, b"secret", open(64));
        let foreign = EvidenceAccessContext::new(9, 2, Some(3));
        let selectors = [
            RetrievalSelector::All,
            RetrievalSelector::ByteRange { start: 0, end: 3 },
            RetrievalSelector::LineRange { start: 1, end: 1 },
            RetrievalSelector::Search {
                query: "secret".to_string(),
                max_hits: 1,
            },
            RetrievalSelector::Items {
                ids: vec![EvidenceId(1)],
            },
        ];
        for selector in selectors {
            let err = retrieve(&store, &foreign, EvidenceId(1), selector).unwrap_err();
            assert!(
                matches!(err, EvidenceError::AccessDenied(_)),
                "foreign scope must deny, got {err:?}"
            );
        }
        // The owner still reads fine.
        assert_eq!(
            read(&store, 1, RetrievalSelector::All).unwrap().bytes,
            b"secret"
        );
    }

    #[test]
    fn unknown_id_is_typed_malformed() {
        let store = store_with(1, b"abc", open(64));
        let err = read(&store, 99, RetrievalSelector::All).unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn dropped_backing_is_refused_not_substituted_with_compact_body() {
        // Cap 2 drops the 6-byte backing but keeps the envelope.
        let mut store = MemoryEvidenceStore::new(2);
        store
            .insert(envelope(1, open(64), 1, Some(3)), Some(b"abcdef".to_vec()))
            .unwrap();
        for selector in [
            RetrievalSelector::All,
            RetrievalSelector::ByteRange { start: 0, end: 1 },
            RetrievalSelector::LineRange { start: 1, end: 1 },
            RetrievalSelector::Search {
                query: "a".to_string(),
                max_hits: 1,
            },
        ] {
            let err = retrieve(&store, &ctx(), EvidenceId(1), selector).unwrap_err();
            assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
        }
    }

    #[test]
    fn byte_range_exactness_including_empty_one_byte_and_end_eq_len() {
        let store = store_with(1, b"abcdef", open(64));
        let cases: &[((u64, u64), &[u8])] = &[
            ((0, 0), b""),       // empty at the front
            ((3, 3), b""),       // empty in the middle
            ((6, 6), b""),       // empty at end == len
            ((0, 1), b"a"),      // one byte
            ((2, 3), b"c"),      // interior one byte
            ((5, 6), b"f"),      // last byte
            ((0, 6), b"abcdef"), // end == len is the full backing
            ((2, 6), b"cdef"),
        ];
        for ((start, end), expected) in cases {
            let got = read(
                &store,
                1,
                RetrievalSelector::ByteRange {
                    start: *start,
                    end: *end,
                },
            )
            .unwrap();
            assert_eq!(&got.bytes, expected, "bytes:{start}..{end}");
            assert!(!got.truncated_by_policy);
            assert_eq!(got.selector_echo, format!("bytes:{start}..{end}"));
            assert_eq!(got.id, EvidenceId(1));
        }
        assert!(
            read(&store, 1, RetrievalSelector::ByteRange { start: 0, end: 0 })
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn byte_range_hostile_edges_are_typed() {
        let store = store_with(1, b"abcdef", open(3));
        // start > end.
        let err = read(&store, 1, RetrievalSelector::ByteRange { start: 2, end: 1 }).unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");
        // end beyond the backing (end == len is fine; len + 1 is not).
        let err = read(&store, 1, RetrievalSelector::ByteRange { start: 0, end: 7 }).unwrap_err();
        match err {
            EvidenceError::Oversized { max, actual } => assert_eq!((max, actual), (6, 7)),
            other => panic!("expected Oversized, got {other:?}"),
        }
        // start beyond the backing.
        let err = read(
            &store,
            1,
            RetrievalSelector::ByteRange {
                start: u64::MAX,
                end: u64::MAX,
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Oversized { .. }), "{err:?}");
        // Length above the envelope's own byte budget.
        let err = read(&store, 1, RetrievalSelector::ByteRange { start: 0, end: 4 }).unwrap_err();
        match err {
            EvidenceError::Oversized { max, actual } => assert_eq!((max, actual), (3, 4)),
            other => panic!("expected Oversized, got {other:?}"),
        }
        // Ranges forbidden entirely => AccessDenied.
        let closed = store_with(2, b"abcdef", RetrievalPolicy::new(false, true, 64));
        let err = read(
            &closed,
            2,
            RetrievalSelector::ByteRange { start: 0, end: 1 },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");
    }

    #[test]
    fn line_range_is_inclusive_one_based_and_exact() {
        let store = store_with(1, b"one\ntwo\nthree\n", open(64));
        let cases: &[((u64, u64), &[u8])] = &[
            ((1, 1), b"one\n"),
            ((2, 2), b"two\n"),
            ((3, 3), b"three\n"),
            ((2, 3), b"two\nthree\n"),
            ((1, 3), b"one\ntwo\nthree\n"),
        ];
        for ((start, end), expected) in cases {
            let got = read(
                &store,
                1,
                RetrievalSelector::LineRange {
                    start: *start,
                    end: *end,
                },
            )
            .unwrap();
            assert_eq!(&got.bytes, expected, "lines:{start}..={end}");
            assert_eq!(got.selector_echo, format!("lines:{start}..={end}"));
        }

        // No trailing newline: the last line is still a line.
        let store2 = store_with(2, b"a\nb", open(64));
        assert_eq!(
            read(
                &store2,
                2,
                RetrievalSelector::LineRange { start: 2, end: 2 }
            )
            .unwrap()
            .bytes,
            b"b"
        );
    }

    #[test]
    fn line_range_hostile_edges_are_typed() {
        let store = store_with(1, b"one\ntwo\nthree\n", open(64));
        for (start, end) in [(0, 1), (1, 0), (2, 1)] {
            let err = read(&store, 1, RetrievalSelector::LineRange { start, end }).unwrap_err();
            assert!(
                matches!(err, EvidenceError::Malformed(_)),
                "{start}..={end}: {err:?}"
            );
        }
        let err = read(&store, 1, RetrievalSelector::LineRange { start: 1, end: 4 }).unwrap_err();
        match err {
            EvidenceError::Oversized { max, actual } => assert_eq!((max, actual), (3, 4)),
            other => panic!("expected Oversized, got {other:?}"),
        }
        // A huge line number never allocates: typed Oversized.
        let err = read(
            &store,
            1,
            RetrievalSelector::LineRange {
                start: 1,
                end: u64::MAX,
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Oversized { .. }), "{err:?}");
    }

    #[test]
    fn search_returns_hit_only_slices_and_respects_max_hits() {
        let store = store_with(1, b"alpha\nbeta\ngamma\nbeta\n", open(64));
        assert_eq!(
            read(&store, 1, RetrievalSelector::LineRange { start: 1, end: 1 })
                .unwrap()
                .bytes,
            b"alpha\n"
        );

        let one = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: "beta".to_string(),
                max_hits: 1,
            },
        )
        .unwrap();
        assert_eq!(one.bytes, b"beta", "hit-only slice, no context or newline");
        assert!(one.truncated_by_policy, "a second match was dropped");

        let two = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: "beta".to_string(),
                max_hits: 2,
            },
        )
        .unwrap();
        assert_eq!(two.bytes, b"beta\nbeta");
        assert!(!two.truncated_by_policy);

        // No match: empty result, never a fallback to the backing.
        let none = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: "zeta".to_string(),
                max_hits: 8,
            },
        )
        .unwrap();
        assert!(none.bytes.is_empty());
        assert!(!none.truncated_by_policy);
        assert_ne!(none.bytes, b"alpha\nbeta\ngamma\nbeta\n");

        // Matching is case-sensitive byte matching.
        let upper = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: "BETA".to_string(),
                max_hits: 8,
            },
        )
        .unwrap();
        assert!(upper.bytes.is_empty());
    }

    #[test]
    fn search_byte_budget_truncates_not_overshoots() {
        let store = store_with(1, b"alpha\nbeta\ngamma\nbeta\n", open(6));
        let got = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: "beta".to_string(),
                max_hits: 8,
            },
        )
        .unwrap();
        // First hit fits (4 <= 6); the second would need 4 + 1 + 4 = 9.
        assert_eq!(got.bytes, b"beta");
        assert!(got.truncated_by_policy);

        // Budget zero: nothing can be returned, but the request stays bounded
        // and typed.
        let zero = store_with(2, b"beta\nbeta\n", open(0));
        let got = read(
            &zero,
            2,
            RetrievalSelector::Search {
                query: "beta".to_string(),
                max_hits: 8,
            },
        )
        .unwrap();
        assert!(got.bytes.is_empty());
        assert!(got.truncated_by_policy);
    }

    #[test]
    fn search_hit_cap_is_hard_bounded() {
        let mut backing = Vec::new();
        for _ in 0..(MAX_SEARCH_HITS + 10) {
            backing.extend_from_slice(b"hit\n");
        }
        let store = store_with(1, &backing, open(64 * 1024));
        let got = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: "hit".to_string(),
                max_hits: u32::MAX,
            },
        )
        .unwrap();
        let hits = got
            .bytes
            .windows(3)
            .filter(|window| *window == b"hit")
            .count();
        assert_eq!(hits, MAX_SEARCH_HITS as usize);
        assert!(got.truncated_by_policy);
    }

    #[test]
    fn search_hostile_selectors_are_typed() {
        let store = store_with(1, b"alpha\n", open(64));
        let err = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: String::new(),
                max_hits: 1,
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");

        let err = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: "a".to_string(),
                max_hits: 0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");

        let huge = "x".repeat(MAX_QUERY_BYTES + 1);
        let err = read(
            &store,
            1,
            RetrievalSelector::Search {
                query: huge,
                max_hits: 1,
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Oversized { .. }), "{err:?}");

        // Search forbidden => AccessDenied, even with a valid query.
        let closed = store_with(2, b"alpha\n", RetrievalPolicy::new(true, false, 64));
        let err = read(
            &closed,
            2,
            RetrievalSelector::Search {
                query: "alpha".to_string(),
                max_hits: 1,
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Allow_search without allow_ranges is still a valid search.
        let search_only = store_with(3, b"alpha\n", RetrievalPolicy::new(false, true, 64));
        let ok = read(
            &search_only,
            3,
            RetrievalSelector::Search {
                query: "alpha".to_string(),
                max_hits: 1,
            },
        )
        .unwrap();
        assert_eq!(ok.bytes, b"alpha");
    }

    #[test]
    fn all_is_refused_without_allow_ranges_and_bounded_by_max_bytes() {
        let closed = store_with(1, b"abcdef", RetrievalPolicy::new(false, true, 64));
        let err = read(&closed, 1, RetrievalSelector::All).unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
        assert!(err.to_string().contains("allow_ranges"), "{err}");

        let open_but_small = store_with(2, b"abcdef", open(5));
        let err = read(&open_but_small, 2, RetrievalSelector::All).unwrap_err();
        match err {
            EvidenceError::Oversized { max, actual } => assert_eq!((max, actual), (5, 6)),
            other => panic!("expected Oversized, got {other:?}"),
        }

        let exact = store_with(3, b"abcde", open(5));
        let got = read(&exact, 3, RetrievalSelector::All).unwrap();
        assert_eq!(got.bytes, b"abcde");
        assert_eq!(got.selector_echo, "all");
        assert!(!got.truncated_by_policy);
    }

    #[test]
    fn items_concatenates_in_request_order_and_is_bounded() {
        let mut store = MemoryEvidenceStore::new(4096);
        store
            .insert(envelope(1, open(64), 1, Some(3)), Some(b"aaa".to_vec()))
            .unwrap();
        store
            .insert(envelope(2, open(64), 1, Some(3)), Some(b"bb".to_vec()))
            .unwrap();
        store
            .insert(envelope(3, open(64), 1, Some(3)), Some(b"c".to_vec()))
            .unwrap();

        let got = read(
            &store,
            1,
            RetrievalSelector::Items {
                ids: vec![EvidenceId(2), EvidenceId(1), EvidenceId(3)],
            },
        )
        .unwrap();
        assert_eq!(got.bytes, b"bbaaac", "request order is preserved");
        assert_eq!(got.selector_echo, "items:[2,1,3]");
        assert!(!got.truncated_by_policy);

        // Duplicates duplicate bytes, still bounded.
        let dup = read(
            &store,
            1,
            RetrievalSelector::Items {
                ids: vec![EvidenceId(1), EvidenceId(1)],
            },
        )
        .unwrap();
        assert_eq!(dup.bytes, b"aaaaaa");
    }

    #[test]
    fn items_hostile_selectors_are_typed_and_never_silently_ignore_scope() {
        let mut store = MemoryEvidenceStore::new(4096);
        store
            .insert(envelope(1, open(64), 1, Some(3)), Some(b"aaa".to_vec()))
            .unwrap();
        // A foreign envelope that exists in another session.
        store
            .insert(envelope(2, open(64), 9, Some(3)), Some(b"zzz".to_vec()))
            .unwrap();

        // Empty list.
        let err = read(&store, 1, RetrievalSelector::Items { ids: vec![] }).unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");

        // Unknown item.
        let err = read(
            &store,
            1,
            RetrievalSelector::Items {
                ids: vec![EvidenceId(42)],
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");

        // Out-of-scope item denies the whole request; it is never skipped.
        let err = read(
            &store,
            1,
            RetrievalSelector::Items {
                ids: vec![EvidenceId(1), EvidenceId(2)],
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Too many ids.
        let err = read(
            &store,
            1,
            RetrievalSelector::Items {
                ids: vec![EvidenceId(1); MAX_ITEMS + 1],
            },
        )
        .unwrap_err();
        match err {
            EvidenceError::Oversized { max, actual } => {
                assert_eq!((max, actual), (MAX_ITEMS, MAX_ITEMS + 1));
            }
            other => panic!("expected Oversized, got {other:?}"),
        }

        // Aggregate over the governing budget => Oversized, no partial bytes.
        let mut tiny = MemoryEvidenceStore::new(4096);
        tiny.insert(envelope(1, open(5), 1, Some(3)), Some(b"aaa".to_vec()))
            .unwrap();
        tiny.insert(envelope(2, open(5), 1, Some(3)), Some(b"bbb".to_vec()))
            .unwrap();
        let err = read(
            &tiny,
            1,
            RetrievalSelector::Items {
                ids: vec![EvidenceId(1), EvidenceId(2)],
            },
        )
        .unwrap_err();
        match err {
            EvidenceError::Oversized { max, actual } => assert_eq!((max, actual), (5, 6)),
            other => panic!("expected Oversized, got {other:?}"),
        }

        // An item whose backing was dropped is refused.
        let mut dropped = MemoryEvidenceStore::new(1);
        dropped
            .insert(envelope(1, open(64), 1, Some(3)), Some(b"aaa".to_vec()))
            .unwrap();
        let err = read(
            &dropped,
            1,
            RetrievalSelector::Items {
                ids: vec![EvidenceId(1)],
            },
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
    }

    #[test]
    fn policy_of_exposes_the_envelope_policy() {
        let store = store_with(1, b"abc", open(9));
        let stored = store.get(EvidenceId(1)).unwrap();
        assert_eq!(policy_of(&stored), open(9));
    }
}
