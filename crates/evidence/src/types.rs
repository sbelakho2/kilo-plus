//! Core evidence types: identity, compression policy, provenance, compact
//! representations and the validated envelope everything else in this crate
//! flows through.
//!
//! Two invariants are structural here, not advisory:
//!
//! 1. **Reversibility requires complete backing.** A representation may only
//!    claim [`Compressibility::Reversible`] when a backing hash exists AND the
//!    backing capture is [`BackingCompleteness::Complete`]. Truncated backing
//!    can never be un-compacted, so claiming reversibility over it is refused
//!    at construction.
//! 2. **Content loss requires a digest.** A [`CompressionRecord`] that admits
//!    [`Lossiness::Content`] must carry a `backing_digest`, otherwise there is
//!    no way to prove which backing bytes the lossy summary came from.
//!
//! Nothing in this module performs I/O, and provenance never silently gains
//! instruction authority: only [`ProvenanceSource::UserPolicy`] is an
//! instruction authority.

use std::fmt;
use std::str::FromStr;

use faktor_core::{SessionId, WorkspaceId};
use serde::{Deserialize, Serialize};

/// Identifies one evidence envelope. Serialized/parsed as a plain `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EvidenceId(pub u64);

impl fmt::Display for EvidenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for EvidenceId {
    type Err = EvidenceError;

    /// Strict u64 parse: no sign but `+` (stdlib semantics), no whitespace,
    /// no radix prefixes, no trailing garbage, no overflow. Every rejection
    /// is a typed [`EvidenceError::Malformed`], never a silent default.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u64>()
            .map(Self)
            .map_err(|_| EvidenceError::Malformed(format!("{s:?} is not a plain u64 evidence id")))
    }
}

/// Typed evidence error. Distinct variants exist so callers can react
/// structurally (a refused invariant is not a malformed input, an oversized
/// request is not an access denial).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceError {
    /// Input could not be parsed into evidence state (ids, payload shapes).
    Malformed(String),
    /// A bounded request or payload exceeded its explicit byte bound.
    Oversized { max: usize, actual: usize },
    /// The retrieval policy forbids the requested access mode.
    AccessDenied(String),
    /// The requested claim violates a hard evidence invariant and is refused.
    Refused(String),
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvidenceError::Malformed(m) => write!(f, "malformed evidence: {m}"),
            EvidenceError::Oversized { max, actual } => {
                write!(
                    f,
                    "evidence oversized: {actual} bytes exceeds the {max} byte bound"
                )
            }
            EvidenceError::AccessDenied(m) => write!(f, "evidence access denied: {m}"),
            EvidenceError::Refused(m) => write!(f, "evidence refused: {m}"),
        }
    }
}

impl std::error::Error for EvidenceError {}

/// How aggressively the compact representation for one evidence kind may
/// compress. Hardcoded by [`EvidenceKind::default_compressibility`] — never
/// chosen per call site by mood.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Compressibility {
    /// No compaction at all: bytes are carried verbatim.
    Never,
    /// Compaction must be bit-exact (formatting/whitespace preserved).
    LosslessOnly,
    /// Structure survives and can be re-expanded from backing; rendering
    /// details (whitespace, ordering) may differ.
    Reversible,
    /// Lossy structural compaction is permitted (never for policy inputs).
    Aggressive,
}

impl Compressibility {
    /// The compressibility every caller MUST use for goal/criteria/policy/
    /// instructions payloads.
    ///
    /// Those inputs carry instruction authority (see
    /// [`ProvenanceSource::UserPolicy`]); compacting them would let a
    /// compressor silently rewrite what the user asked for. The constructor
    /// name is deliberately policy-shaped rather than a bare `Never` so call
    /// sites read `Compressibility::for_policy_inputs()` and reviewers can
    /// see the authority rule at the use site.
    pub const fn for_policy_inputs() -> Self {
        Self::Never
    }
}

/// What a compact representation loses relative to its backing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lossiness {
    /// Bit-exact: the compact body re-expands to the backing bytes.
    None,
    /// Structure preserved, incidental formatting/ordering dropped.
    Structural,
    /// Information was summarized or dropped (requires a backing digest).
    Content,
}

/// Whether the captured backing is the whole original or a bounded prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackingCompleteness {
    /// The full original payload was captured in CAS.
    Complete,
    /// Capture was bounded: the backing is a truncated view of the original.
    /// Truncated backing can never claim [`Compressibility::Reversible`].
    Truncated,
}

/// Severity of one diagnostic entry carried inside evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Build/test/verification failure.
    Error,
    /// Non-fatal problem worth surfacing.
    Warning,
    /// Informational entry.
    Info,
}

/// The shape of the raw output the envelope was normalized from. Compression
/// policy is hardcoded per kind so callers never hand-pick it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// Compiler/linter diagnostics (file, line, severity, message).
    DiagnosticSet,
    /// Test run report (suite, case, status, duration).
    TestReport,
    /// Retrieval/search hits (path, snippet, score).
    SearchResults,
    /// File map / repo outline entries.
    FileMap,
    /// Unified diff hunks.
    Diff,
    /// Process or tool log lines.
    ProcessLog,
    /// Symbol definitions/references.
    SymbolSet,
    /// Rows of structured records (JSON/CSV-shaped).
    StructuredRows,
    /// Handoff payload from a child agent/task.
    ChildHandoff,
    /// Semantically retrieved context (embeddings provider output).
    SemanticContext,
    /// Anything unstructured: opaque text without a known grammar.
    GenericText,
}

impl EvidenceKind {
    /// The compression policy hardcoded for this kind.
    ///
    /// Structured, machine-generated output (diagnostics, tests, search,
    /// logs, symbols, rows, file maps) may be compacted aggressively. Diffs
    /// and handoff/semantic context stay reversible so they can be
    /// re-expanded from backing. Arbitrary text is only ever compacted
    /// losslessly — a generic text compressor is not allowed to summarize.
    pub fn default_compressibility(&self) -> Compressibility {
        match self {
            EvidenceKind::DiagnosticSet
            | EvidenceKind::TestReport
            | EvidenceKind::SearchResults
            | EvidenceKind::ProcessLog
            | EvidenceKind::SymbolSet
            | EvidenceKind::StructuredRows
            | EvidenceKind::FileMap => Compressibility::Aggressive,
            EvidenceKind::Diff | EvidenceKind::ChildHandoff | EvidenceKind::SemanticContext => {
                Compressibility::Reversible
            }
            EvidenceKind::GenericText => Compressibility::LosslessOnly,
        }
    }
}

/// Where evidence came from. Only user-authored policy may instruct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSource {
    /// User-authored goal/criteria/policy/instructions. The ONLY source with
    /// instruction authority.
    UserPolicy,
    /// Output of a tool run (shell, editor, language server, ...).
    Tool,
    /// Repository content (files, git objects).
    Repository,
    /// Verification records / completion proof machinery.
    Verification,
    /// Model-generated content (plans, summaries).
    Model,
    /// External semantic provider output (embeddings, rerankers).
    SemanticProvider,
    /// Agent-coordination board content (another agent's post, receipt or
    /// task update). Peer-agent DATA: like tool/model output it can never
    /// instruct — a sibling's post is evidence to weigh, never policy.
    AgentCoordination,
}

impl ProvenanceSource {
    /// True only for [`ProvenanceSource::UserPolicy`]. Tool output, repo
    /// content, verification records, model output, provider output and
    /// agent-coordination board content can never acquire instruction
    /// authority by flowing through evidence.
    pub const fn is_instruction_authority(&self) -> bool {
        matches!(self, ProvenanceSource::UserPolicy)
    }
}

/// The provenance attached to one envelope: where every contributing fragment
/// came from. An empty set means unknown provenance, never user authority.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ProvenanceSet {
    pub entries: Vec<ProvenanceSource>,
}

impl ProvenanceSet {
    pub fn new(entries: impl IntoIterator<Item = ProvenanceSource>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    pub fn push(&mut self, source: ProvenanceSource) {
        self.entries.push(source);
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when any entry is an instruction authority
    /// ([`ProvenanceSource::UserPolicy`]). Evidence carrying this is
    /// user-policy input and MUST stay lossless via
    /// [`Compressibility::for_policy_inputs`].
    pub fn has_instruction_authority(&self) -> bool {
        self.entries
            .iter()
            .any(ProvenanceSource::is_instruction_authority)
    }
}

/// One compression step applied to raw backing bytes to produce a compact
/// representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressionRecord {
    /// Algorithm name, e.g. `"zstd"`, `"identity"`.
    pub algorithm: String,
    /// Algorithm grammar/version, so compact bodies stay decodable.
    pub version: u32,
    /// Size of the original backing payload in bytes.
    pub original_bytes: u64,
    /// Size of the compact representation in bytes.
    pub compact_bytes: u64,
    /// What this step lost relative to the backing.
    pub lossiness: Lossiness,
    /// Digest of the backing bytes this record was produced from. REQUIRED
    /// when `lossiness == Lossiness::Content` (enforced by
    /// [`EvidenceEnvelope::new`]): without it a lossy summary cannot be tied
    /// back to specific backing bytes.
    pub backing_digest: Option<[u8; 32]>,
}

impl CompressionRecord {
    /// The no-compaction record: bytes are carried verbatim and losslessly.
    /// Use for [`Compressibility::Never`] / [`Compressibility::LosslessOnly`]
    /// payloads and for policy inputs.
    pub fn identity(original_bytes: u64) -> Self {
        Self {
            algorithm: "identity".to_string(),
            version: 1,
            original_bytes,
            compact_bytes: original_bytes,
            lossiness: Lossiness::None,
            backing_digest: None,
        }
    }
}

/// A compact, grammar-tagged rendering of evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactRepresentation {
    /// Name/version of the grammar `body` is written in (e.g. `"diag-v1"`).
    pub grammar: String,
    /// The compact body itself. Bounded elsewhere; never auto-dumped whole.
    pub body: String,
}

/// How a compact representation may be accessed. Retrieval is explicit and
/// bounded: whole-blob auto-dumps are never permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalPolicy {
    /// Byte-range reads are permitted.
    pub allow_ranges: bool,
    /// Search/query retrieval is permitted.
    pub allow_search: bool,
    /// Hard ceiling for any single retrieval response, in bytes.
    pub max_bytes: usize,
}

impl RetrievalPolicy {
    pub const fn new(allow_ranges: bool, allow_search: bool, max_bytes: usize) -> Self {
        Self {
            allow_ranges,
            allow_search,
            max_bytes,
        }
    }

    /// Refuse search retrieval unless explicitly permitted.
    pub fn check_search(&self) -> Result<(), EvidenceError> {
        if self.allow_search {
            Ok(())
        } else {
            Err(EvidenceError::AccessDenied(
                "search retrieval is not permitted for this envelope".to_string(),
            ))
        }
    }

    /// Refuse ranged retrieval unless permitted, and refuse any requested
    /// length above `max_bytes` with a structured `Oversized` error — the
    /// bound is enforced before the read, never after.
    pub fn check_range(&self, requested_bytes: usize) -> Result<(), EvidenceError> {
        if !self.allow_ranges {
            return Err(EvidenceError::AccessDenied(
                "ranged retrieval is not permitted for this envelope".to_string(),
            ));
        }
        if requested_bytes > self.max_bytes {
            return Err(EvidenceError::Oversized {
                max: self.max_bytes,
                actual: requested_bytes,
            });
        }
        Ok(())
    }
}

/// A fully-typed evidence package: what it is, who it belongs to, where it
/// came from, how it was compacted, and how it may be retrieved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceEnvelope {
    pub id: EvidenceId,
    pub kind: EvidenceKind,
    pub session_id: SessionId,
    pub workspace_id: WorkspaceId,
    pub task_id: Option<u64>,
    pub source_revision: Option<String>,
    pub provenance: ProvenanceSet,
    pub compressibility: Compressibility,
    pub compact: CompactRepresentation,
    pub backing_hash: Option<[u8; 32]>,
    pub backing_completeness: BackingCompleteness,
    pub compression: CompressionRecord,
    pub retrieval: RetrievalPolicy,
}

impl EvidenceEnvelope {
    /// Construct an envelope, enforcing the hard evidence invariants:
    ///
    /// - `compressibility == Reversible` requires `backing_hash` to be `Some`
    ///   AND `backing_completeness == Complete`. A truncated backing cannot be
    ///   re-expanded, so reversibility over it is refused — never downgraded
    ///   silently.
    /// - `compression.lossiness == Content` requires
    ///   `compression.backing_digest` to be `Some`, so a lossy summary is
    ///   always tied to identifiable backing bytes.
    ///
    /// Violations are returned as [`EvidenceError::Refused`]; the caller must
    /// either capture complete backing or pick an honest compressibility.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: EvidenceId,
        kind: EvidenceKind,
        session_id: SessionId,
        workspace_id: WorkspaceId,
        task_id: Option<u64>,
        source_revision: Option<String>,
        provenance: ProvenanceSet,
        compressibility: Compressibility,
        compact: CompactRepresentation,
        backing_hash: Option<[u8; 32]>,
        backing_completeness: BackingCompleteness,
        compression: CompressionRecord,
        retrieval: RetrievalPolicy,
    ) -> Result<Self, EvidenceError> {
        if compressibility == Compressibility::Reversible {
            if backing_hash.is_none() {
                return Err(EvidenceError::Refused(
                    "Reversible evidence requires a backing hash".to_string(),
                ));
            }
            if backing_completeness != BackingCompleteness::Complete {
                return Err(EvidenceError::Refused(
                    "truncated backing can never claim Reversible".to_string(),
                ));
            }
        }
        if compression.lossiness == Lossiness::Content && compression.backing_digest.is_none() {
            return Err(EvidenceError::Refused(
                "Content lossiness requires a backing digest".to_string(),
            ));
        }
        Ok(Self {
            id,
            kind,
            session_id,
            workspace_id,
            task_id,
            source_revision,
            provenance,
            compressibility,
            compact,
            backing_hash,
            backing_completeness,
            compression,
            retrieval,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type MakeResult = Result<EvidenceEnvelope, EvidenceError>;

    fn make(
        compressibility: Compressibility,
        backing_hash: Option<[u8; 32]>,
        backing_completeness: BackingCompleteness,
        lossiness: Lossiness,
        backing_digest: Option<[u8; 32]>,
    ) -> MakeResult {
        EvidenceEnvelope::new(
            EvidenceId(7),
            EvidenceKind::TestReport,
            SessionId::new(1),
            WorkspaceId::new(2),
            Some(3),
            Some("rev-abc".to_string()),
            ProvenanceSet::new([ProvenanceSource::Tool]),
            compressibility,
            CompactRepresentation {
                grammar: "test-v1".to_string(),
                body: "PASS tests=3 failed=0".to_string(),
            },
            backing_hash,
            backing_completeness,
            CompressionRecord {
                algorithm: "zstd".to_string(),
                version: 1,
                original_bytes: 100,
                compact_bytes: 20,
                lossiness,
                backing_digest,
            },
            RetrievalPolicy::new(true, true, 4096),
        )
    }

    #[test]
    fn default_compressibility_table_is_exact() {
        let table = [
            (EvidenceKind::DiagnosticSet, Compressibility::Aggressive),
            (EvidenceKind::TestReport, Compressibility::Aggressive),
            (EvidenceKind::SearchResults, Compressibility::Aggressive),
            (EvidenceKind::FileMap, Compressibility::Aggressive),
            (EvidenceKind::ProcessLog, Compressibility::Aggressive),
            (EvidenceKind::SymbolSet, Compressibility::Aggressive),
            (EvidenceKind::StructuredRows, Compressibility::Aggressive),
            (EvidenceKind::Diff, Compressibility::Reversible),
            (EvidenceKind::ChildHandoff, Compressibility::Reversible),
            (EvidenceKind::SemanticContext, Compressibility::Reversible),
            (EvidenceKind::GenericText, Compressibility::LosslessOnly),
        ];
        assert_eq!(table.len(), 11, "every kind must be covered exactly once");
        for (kind, expected) in table {
            assert_eq!(
                kind.default_compressibility(),
                expected,
                "wrong policy for {kind:?}"
            );
        }
        // Policy inputs are NEVER compacted, regardless of kind.
        assert_eq!(Compressibility::for_policy_inputs(), Compressibility::Never);
    }

    #[test]
    fn reversible_requires_complete_backing_rows() {
        let hash = [7u8; 32];
        let digest = [8u8; 32];

        // The only admissible reversible row: complete backing + hash.
        assert!(make(
            Compressibility::Reversible,
            Some(hash),
            BackingCompleteness::Complete,
            Lossiness::None,
            None
        )
        .is_ok());

        // Reversible without a backing hash: refused.
        let err = make(
            Compressibility::Reversible,
            None,
            BackingCompleteness::Complete,
            Lossiness::None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");

        // Reversible over truncated backing, even with a hash: refused.
        let err = make(
            Compressibility::Reversible,
            Some(hash),
            BackingCompleteness::Truncated,
            Lossiness::None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");

        // Both missing: still refused, never a silent downgrade.
        let err = make(
            Compressibility::Reversible,
            None,
            BackingCompleteness::Truncated,
            Lossiness::None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");

        // Non-reversible kinds may legitimately carry truncated backing.
        assert!(make(
            Compressibility::Aggressive,
            None,
            BackingCompleteness::Truncated,
            Lossiness::Structural,
            None
        )
        .is_ok());
        assert!(make(
            Compressibility::LosslessOnly,
            None,
            BackingCompleteness::Truncated,
            Lossiness::None,
            None
        )
        .is_ok());
        assert!(make(
            Compressibility::Never,
            None,
            BackingCompleteness::Truncated,
            Lossiness::None,
            None
        )
        .is_ok());
        // A complete backing with content loss and a digest is admissible.
        assert!(make(
            Compressibility::Aggressive,
            Some(hash),
            BackingCompleteness::Complete,
            Lossiness::Content,
            Some(digest)
        )
        .is_ok());
    }

    #[test]
    fn lossy_content_without_backing_digest_is_refused() {
        let err = make(
            Compressibility::Aggressive,
            Some([1u8; 32]),
            BackingCompleteness::Complete,
            Lossiness::Content,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
        assert!(err.to_string().contains("digest"), "{err}");

        // Structural loss does not demand a digest (only Content does).
        assert!(make(
            Compressibility::Aggressive,
            Some([1u8; 32]),
            BackingCompleteness::Complete,
            Lossiness::Structural,
            None
        )
        .is_ok());
    }

    #[test]
    fn envelope_serde_round_trip_preserves_shape_and_nulls() {
        let env = make(
            Compressibility::Reversible,
            Some([9u8; 32]),
            BackingCompleteness::Complete,
            Lossiness::Structural,
            Some([8u8; 32]),
        )
        .unwrap();
        let json = serde_json::to_string(&env).unwrap();
        let back: EvidenceEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(env, back);

        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value["task_id"].is_number());
        assert!(value["source_revision"].is_string());
        assert!(value["backing_hash"].is_array());
        assert_eq!(
            value["backing_hash"].as_array().map(Vec::len),
            Some(32),
            "backing hash must round-trip as 32 bytes"
        );

        // Absent optional fields serialize as explicit nulls, not omissions
        // (a reader must be able to tell "no revision" from "field dropped").
        let mut bare = env.clone();
        bare.task_id = None;
        bare.source_revision = None;
        bare.backing_hash = None;
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&bare).unwrap()).unwrap();
        assert!(value["task_id"].is_null());
        assert!(value["source_revision"].is_null());
        assert!(value["backing_hash"].is_null());
        let back: EvidenceEnvelope = serde_json::from_value(value).unwrap();
        assert_eq!(bare, back);

        // Enum tags are locked to snake_case strings.
        assert_eq!(
            serde_json::to_string(&EvidenceKind::DiagnosticSet).unwrap(),
            "\"diagnostic_set\""
        );
        assert_eq!(
            serde_json::to_string(&Compressibility::LosslessOnly).unwrap(),
            "\"lossless_only\""
        );
        assert_eq!(serde_json::to_string(&Lossiness::None).unwrap(), "\"none\"");
        assert_eq!(
            serde_json::to_string(&BackingCompleteness::Truncated).unwrap(),
            "\"truncated\""
        );
        assert_eq!(
            serde_json::to_string(&ProvenanceSource::SemanticProvider).unwrap(),
            "\"semantic_provider\""
        );
        assert_eq!(
            serde_json::to_string(&ProvenanceSource::AgentCoordination).unwrap(),
            "\"agent_coordination\""
        );
        assert_eq!(
            serde_json::to_string(&Severity::Error).unwrap(),
            "\"error\""
        );
        // A wrong-cased tag must not deserialize.
        assert!(serde_json::from_str::<Compressibility>("\"Aggressive\"").is_err());
    }

    #[test]
    fn evidence_id_display_and_from_str_hostile_inputs() {
        assert_eq!(EvidenceId(42).to_string(), "42");
        assert_eq!("42".parse::<EvidenceId>().unwrap(), EvidenceId(42));
        assert_eq!(serde_json::to_string(&EvidenceId(42)).unwrap(), "42");
        assert_eq!(
            serde_json::from_str::<EvidenceId>("42").unwrap(),
            EvidenceId(42)
        );

        let hostile = [
            "",                     // empty
            " ",                    // whitespace only
            " 42",                  // leading whitespace
            "42 ",                  // trailing whitespace
            "-1",                   // negative
            "-0",                   // negative zero
            "4 2",                  // embedded whitespace
            "0x2a",                 // radix prefix
            "42abc",                // trailing garbage
            "4.2",                  // float
            "1e3",                  // exponent
            "١٢٣",                  // non-ascii digits
            "18446744073709551616", // u64::MAX + 1 overflow
            "null",                 // json literal as text
        ];
        for raw in hostile {
            let err = raw.parse::<EvidenceId>().unwrap_err();
            assert!(
                matches!(err, EvidenceError::Malformed(_)),
                "{raw:?} must be a typed Malformed error, got {err:?}"
            );
            assert!(!err.to_string().is_empty());
        }

        // Serde is equally strict: no string coercion, no floats, no
        // negatives, no overflow.
        for raw in ["\"42\"", "42.0", "-1", "18446744073709551616", "null"] {
            assert!(
                serde_json::from_str::<EvidenceId>(raw).is_err(),
                "{raw} must not deserialize as EvidenceId"
            );
        }
        // u64::MAX itself is a valid raw id.
        assert_eq!(
            serde_json::from_str::<EvidenceId>("18446744073709551615").unwrap(),
            EvidenceId(u64::MAX)
        );
    }

    #[test]
    fn provenance_only_user_policy_has_instruction_authority() {
        assert!(ProvenanceSource::UserPolicy.is_instruction_authority());
        for source in [
            ProvenanceSource::Tool,
            ProvenanceSource::Repository,
            ProvenanceSource::Verification,
            ProvenanceSource::Model,
            ProvenanceSource::SemanticProvider,
            ProvenanceSource::AgentCoordination,
        ] {
            assert!(
                !source.is_instruction_authority(),
                "{source:?} must never carry instruction authority"
            );
        }

        let tool_only = ProvenanceSet::new([
            ProvenanceSource::Tool,
            ProvenanceSource::Repository,
            ProvenanceSource::Model,
        ]);
        assert!(!tool_only.has_instruction_authority());
        assert_eq!(tool_only.len(), 3);
        assert!(!tool_only.is_empty());

        let with_user = ProvenanceSet::new([ProvenanceSource::Tool, ProvenanceSource::UserPolicy]);
        assert!(with_user.has_instruction_authority());

        // Unknown provenance is empty, never implicitly trusted.
        let empty = ProvenanceSet::default();
        assert!(empty.is_empty());
        assert!(!empty.has_instruction_authority());
    }

    #[test]
    fn retrieval_policy_decisions_are_typed() {
        let open = RetrievalPolicy::new(true, true, 10);
        assert!(open.check_search().is_ok());
        assert!(open.check_range(0).is_ok());
        assert!(open.check_range(10).is_ok(), "bound is inclusive");
        match open.check_range(11) {
            Err(EvidenceError::Oversized { max, actual }) => {
                assert_eq!((max, actual), (10, 11));
            }
            other => panic!("expected Oversized, got {other:?}"),
        }

        let closed = RetrievalPolicy::new(false, false, 10);
        match closed.check_search() {
            Err(EvidenceError::AccessDenied(_)) => {}
            other => panic!("expected AccessDenied, got {other:?}"),
        }
        match closed.check_range(1) {
            Err(EvidenceError::AccessDenied(_)) => {}
            other => panic!("expected AccessDenied, got {other:?}"),
        }
        // A zero-byte ceiling refuses every non-empty read before it happens.
        let zero = RetrievalPolicy::new(true, true, 0);
        assert!(zero.check_range(0).is_ok());
        assert!(matches!(
            zero.check_range(1),
            Err(EvidenceError::Oversized { max: 0, actual: 1 })
        ));
    }

    #[test]
    fn identity_compression_record_is_lossless_verbatim() {
        let rec = CompressionRecord::identity(1234);
        assert_eq!(rec.original_bytes, rec.compact_bytes);
        assert_eq!(rec.lossiness, Lossiness::None);
        assert_eq!(rec.backing_digest, None);
        assert_eq!(rec.version, 1);

        // All four error variants render non-empty, distinct Display text.
        let rendered = [
            EvidenceError::Malformed("x".to_string()).to_string(),
            EvidenceError::Oversized { max: 1, actual: 2 }.to_string(),
            EvidenceError::AccessDenied("x".to_string()).to_string(),
            EvidenceError::Refused("x".to_string()).to_string(),
        ];
        for text in &rendered {
            assert!(!text.is_empty());
        }
        let unique: std::collections::BTreeSet<_> = rendered.iter().collect();
        assert_eq!(
            unique.len(),
            rendered.len(),
            "variants must be distinguishable"
        );
    }
}
