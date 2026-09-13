//! Durable binary/image attachments: bytes in the CAS, typed metadata rows
//! in the store (`AttachmentId { digest, mime, filename, size }`, schema
//! v24).
//!
//! - `put_attachment` validates the mime/filename/size bounds BEFORE hashing
//!   or writing anything, streams the bytes into the CAS (content-addressed,
//!   dedupe + corruption detection come from the CAS), then persists one
//!   typed metadata row keyed `(session_id, digest)`.
//! - Dedupe by digest: an identical payload resolves to the FIRST-written
//!   metadata row — repeated uploads are idempotent and return a
//!   byte-identical [`AttachmentId`].
//! - `attachment`/`list_attachments` resolve the durable rows after a
//!   restart (never a process-local map); `attachment_bytes` re-verifies the
//!   CAS blob and refuses a metadata/blob size mismatch.
//!
//! The attachment is deliberately SEPARATE from the workspace-relative
//! `files` vocabulary: an attachment is bytes addressed by digest, never a
//! path. Provider delivery of attachment bytes is NOT wired: the provider
//! layer has only a URL-shaped `ContentKind::Image { url }` part whose
//! adapter encodings are inconsistent (OpenAI accepts a data URL, Anthropic
//! emits a `type:"url"` source the API rejects, Google hardcodes
//! `image/png` and expects raw base64), and the agent has no path from a
//! durable attachment row to a request part. The honest contract is
//! therefore a LOUD typed refusal of `is_image()` submission at the wire
//! boundary (`POST .../attachments` and task admission return code
//! `unsupported`), with the composer draft/images restored by the client.
//! The media-part follow-up is: (1) add a binary `ContentKind` that carries
//! `{mime, bytes}` (not a URL), (2) teach every adapter to encode it for its
//! own wire (base64 source for Anthropic/Google, data URL for
//! OpenAI/Ollama), (3) gate it on `ModelCapabilities::vision`, and (4) turn
//! the refusal into delivery once (1)-(3) exist. Until then no code path
//! may claim images reached a provider.

use faktor_core::attachment::{
    validate_filename, validate_mime, AttachmentId, MAX_ATTACHMENTS_PER_TASK, MAX_ATTACHMENT_BYTES,
};
use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;

use crate::handle::SessionHandle;
use crate::SessionError;

impl SessionHandle {
    /// Store a bounded attachment: validate bounds/shape, put the bytes into
    /// the CAS, persist the typed metadata row. Returns the (possibly
    /// pre-existing) canonical [`AttachmentId`]. Hostile mime/filename shapes
    /// and oversized payloads are typed refusals before any write.
    pub fn put_attachment(
        &self,
        mime: &str,
        filename: Option<&str>,
        bytes: &[u8],
    ) -> faktor_core::Result<AttachmentId> {
        let canonical_mime = mime.trim().to_ascii_lowercase();
        validate_mime(&canonical_mime)?;
        if let Some(name) = filename {
            validate_filename(name)?;
        }
        if bytes.len() as u64 > MAX_ATTACHMENT_BYTES {
            return Err(SessionError::Oversized(format!(
                "attachment of {} bytes exceeds MAX_ATTACHMENT_BYTES ({MAX_ATTACHMENT_BYTES})",
                bytes.len()
            ))
            .into());
        }
        let digest = self.manager.cas().put(bytes).map_err(SessionError::from)?;
        let computed = AttachmentId {
            digest,
            mime: canonical_mime,
            filename: filename.map(str::to_string),
            size: bytes.len() as u64,
        };
        let stored = self
            .manager
            .store()
            .put_attachment(self.id, &computed)
            .map_err(crate::map_store_err)?;
        if stored.size != computed.size {
            return Err(SessionError::Malformed(format!(
                "attachment {digest} row size {} does not match the {} uploaded bytes (durable row corruption)",
                stored.size,
                computed.size
            ))
            .into());
        }
        Ok(stored)
    }

    /// Resolve one durable attachment row by its digest (restart-safe).
    pub fn attachment(&self, digest: FileHash) -> faktor_core::Result<Option<AttachmentId>> {
        self.manager
            .store()
            .attachment(self.id, digest)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The session's durable attachment rows, bounded page.
    pub fn list_attachments(&self, limit: usize) -> faktor_core::Result<Vec<AttachmentId>> {
        self.manager
            .store()
            .list_attachments(self.id, limit)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Read the attachment bytes back, verifying the CAS blob against the
    /// durable metadata: the tracked size is checked before any I/O and the
    /// decoded byte count must equal the row's `size` (a mismatch is loud,
    /// never served as the attachment).
    pub fn attachment_bytes(
        &self,
        id: &AttachmentId,
        max_bytes: usize,
    ) -> faktor_core::Result<Vec<u8>> {
        if id.size > max_bytes as u64 {
            return Err(SessionError::Oversized(format!(
                "attachment {} is {} bytes, limit {max_bytes}",
                id.digest, id.size
            ))
            .into());
        }
        let bytes = self
            .manager
            .cas()
            .get_verified_now(id.digest)
            .map_err(SessionError::from)?;
        if bytes.len() as u64 != id.size {
            return Err(SessionError::Malformed(format!(
                "attachment {} decoded to {} bytes, metadata says {}",
                id.digest,
                bytes.len(),
                id.size
            ))
            .into());
        }
        Ok(bytes)
    }

    /// Resolve and validate one attachment SET at admission time: the count
    /// is bounded by [`MAX_ATTACHMENTS_PER_TASK`], every id is structurally
    /// validated (hostile mime/filename/size are typed refusals), and every
    /// digest must resolve to a byte-identical durable row of THIS session.
    /// Admission never fabricates an attachment: an unknown digest is a
    /// typed `NotFound`, a mismatched durable row a typed `Malformed`.
    ///
    /// Media policy lives ABOVE this layer: provider media/content parts are
    /// not wired, so the server refuses `is_image()` ids loudly before this
    /// call (see `crates/server/src/native/attachment.rs` for the precise
    /// follow-up contract).
    pub fn resolve_attachments(&self, ids: &[AttachmentId]) -> faktor_core::Result<()> {
        if ids.len() > MAX_ATTACHMENTS_PER_TASK {
            return Err(Error::new(
                ErrorKind::Oversized,
                format!(
                    "{} attachments exceed MAX_ATTACHMENTS_PER_TASK ({MAX_ATTACHMENTS_PER_TASK})",
                    ids.len()
                ),
            ));
        }
        for id in ids {
            id.validate()?;
            match self.attachment(id.digest)? {
                Some(stored) if stored == *id => {}
                Some(stored) => {
                    return Err(Error::new(
                        ErrorKind::Malformed,
                        format!(
                            "attachment {} does not match the durable row (requested {} bytes / {} mime, stored {} bytes / {} mime)",
                            id.digest, id.size, id.mime, stored.size, stored.mime
                        ),
                    ));
                }
                None => {
                    return Err(Error::new(
                        ErrorKind::NotFound,
                        format!(
                            "attachment {} is not stored in session {}",
                            id.digest, self.id
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use crate::SessionManager;

    #[test]
    fn upload_is_durable_typed_deduped_and_survives_reopen() {
        let (dir, m) = test_manager();
        let s = session(&m);
        let bytes = b"durable attachment bytes".to_vec();
        let first = s
            .put_attachment("Image/PNG", Some("shot.png"), &bytes)
            .unwrap();
        assert_eq!(first.mime, "image/png", "mime is canonicalized");
        assert_eq!(first.size, bytes.len() as u64);
        assert!(first.is_image());
        // Dedupe by digest: same bytes (even with different metadata) return
        // the FIRST durable row byte-identically.
        let again = s
            .put_attachment("application/pdf", Some("other.pdf"), &bytes)
            .unwrap();
        assert_eq!(again, first, "dedupe by digest returns the stored id");
        assert_eq!(s.list_attachments(16).unwrap(), vec![first.clone()]);
        // Bytes resolve from the CAS.
        assert_eq!(s.attachment_bytes(&first, 1 << 20).unwrap(), bytes);
        // Bounded reads refuse before I/O.
        let err = s.attachment_bytes(&first, 4).unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Oversized);

        // Reopen the REAL store/cas from disk: the typed row resolves
        // identically and the bytes still verify.
        drop(s);
        drop(m);
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let h = m2
            .get_session(faktor_core::id::SessionId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(h.attachment(first.digest).unwrap(), Some(first.clone()));
        assert_eq!(h.list_attachments(16).unwrap(), vec![first.clone()]);
        assert_eq!(h.attachment_bytes(&first, 1 << 20).unwrap(), bytes);
        // An unknown digest is an honest absence, never a phantom.
        assert_eq!(h.attachment(FileHash::from([9; 32])).unwrap(), None);
    }

    #[test]
    fn hostile_attachments_are_typed_refusals_before_any_write() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let payload = b"payload".to_vec();
        for mime in [
            "",
            "image",
            "image/",
            "/png",
            "image/png/extra",
            "bad mime/type",
        ] {
            let err = s.put_attachment(mime, None, &payload).unwrap_err();
            assert!(
                matches!(
                    err.kind,
                    faktor_core::ErrorKind::Malformed | faktor_core::ErrorKind::Oversized
                ),
                "mime {mime:?} => {:?}",
                err.kind
            );
        }
        for name in ["../secrets", "a/b.png", "a\\b.png", ".", "..", "bad\0name"] {
            let err = s
                .put_attachment("image/png", Some(name), &payload)
                .unwrap_err();
            assert!(
                matches!(
                    err.kind,
                    faktor_core::ErrorKind::Malformed | faktor_core::ErrorKind::Oversized
                ),
                "filename {name:?} => {:?}",
                err.kind
            );
        }
        // Oversized payloads are refused BEFORE hashing/writing anything.
        let big = vec![0u8; MAX_ATTACHMENT_BYTES as usize + 1];
        let err = s.put_attachment("image/png", None, &big).unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Oversized);
        // Nothing durable was left behind by any hostile attempt.
        assert!(s.list_attachments(16).unwrap().is_empty());
    }

    #[test]
    fn corrupted_digest_metadata_mismatch_is_loud() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let id = s
            .put_attachment("application/octet-stream", None, b"1234")
            .unwrap();
        // A forged id naming the real digest but a different size can never
        // be served as the attachment.
        let forged = AttachmentId {
            size: id.size + 1,
            ..id.clone()
        };
        let err = s.attachment_bytes(&forged, 1 << 20).unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Malformed);
    }

    #[test]
    fn admission_resolution_requires_durable_byte_identical_rows() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let stored = s
            .put_attachment("application/pdf", Some("spec.pdf"), b"%PDF-1.4")
            .unwrap();
        // A stored, byte-identical set resolves cleanly (dedupe included).
        s.resolve_attachments(std::slice::from_ref(&stored))
            .unwrap();
        s.resolve_attachments(&[]).unwrap();
        // An unknown digest is a typed absence, never a phantom admission.
        let unknown = AttachmentId {
            digest: FileHash::from([7; 32]),
            ..stored.clone()
        };
        let err = s.resolve_attachments(&[unknown]).unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::NotFound);
        // A mismatched durable row (same digest, different metadata) is a
        // typed refusal: admission trusts the durable row, not the request.
        let mismatched = AttachmentId {
            mime: "text/plain".into(),
            ..stored.clone()
        };
        let err = s.resolve_attachments(&[mismatched]).unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Malformed);
        // Structurally hostile ids are refused before any store read.
        let hostile = AttachmentId {
            filename: Some("../secrets".into()),
            ..stored.clone()
        };
        let err = s.resolve_attachments(&[hostile]).unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Malformed);
        // The count bound is typed and inclusive.
        let many = vec![stored; MAX_ATTACHMENTS_PER_TASK + 1];
        let err = s.resolve_attachments(&many).unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Oversized);
    }
}
