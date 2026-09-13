//! Binary attachment identity: content-addressed bytes plus their declared
//! media type and optional display filename.
//!
//! This is a PURE value type (no I/O): the bytes live in the CAS, the
//! metadata row lives in the store, and every consumer validates through
//! [`AttachmentId::validate`]. Canonical on write (lowercase mime), strict on
//! read: a hostile mime shape, a traversal filename or an oversized size is a
//! typed [`crate::Error`], never a silent normalization.

use crate::error::{Error, ErrorKind};
use crate::hash::FileHash;

/// Hard ceiling of one attachment payload (decompressed bytes). Mirrors the
/// session layer's `MAX_ATTACHMENT_BYTES`; the CAS ceiling is higher, so an
/// attachment is always a valid CAS blob.
pub const MAX_ATTACHMENT_BYTES: u64 = 32 * 1024 * 1024;
/// Maximum length of one mime type string.
pub const MAX_ATTACHMENT_MIME_BYTES: usize = 128;
/// Maximum length of one attachment filename (bytes).
pub const MAX_ATTACHMENT_FILENAME_BYTES: usize = 255;
/// Maximum number of attachments on one task/child spec.
pub const MAX_ATTACHMENTS_PER_TASK: usize = 64;

/// One durable binary attachment identity:
/// `{ digest, mime, filename, size }`. `digest` is the BLAKE3 CAS address of
/// the bytes; `size` is the decompressed byte count; `mime` is the canonical
/// lowercase media type; `filename` is an optional DISPLAY label (never a
/// path — separators and traversal are refused).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttachmentId {
    pub digest: FileHash,
    pub mime: String,
    pub filename: Option<String>,
    pub size: u64,
}

impl AttachmentId {
    /// Build one attachment identity, canonicalizing the mime to lowercase
    /// and validating bounds/shape. Every failure is a typed error before
    /// any store/CAS write.
    pub fn new(
        digest: FileHash,
        mime: impl AsRef<str>,
        filename: Option<&str>,
        size: u64,
    ) -> Result<Self, Error> {
        let id = Self {
            digest,
            mime: mime.as_ref().trim().to_ascii_lowercase(),
            filename: filename.map(str::to_string),
            size,
        };
        id.validate()?;
        Ok(id)
    }

    /// Structural validation (bounds + hostile shapes). The digest is typed
    /// by construction; existence in the CAS/store is a separate resolution
    /// step the caller owns.
    pub fn validate(&self) -> Result<(), Error> {
        validate_mime(&self.mime)?;
        if let Some(name) = &self.filename {
            validate_filename(name)?;
        }
        if self.size > MAX_ATTACHMENT_BYTES {
            return Err(Error::new(
                ErrorKind::Oversized,
                format!(
                    "attachment of {} bytes exceeds MAX_ATTACHMENT_BYTES ({MAX_ATTACHMENT_BYTES})",
                    self.size
                ),
            ));
        }
        Ok(())
    }

    /// TRUE when the declared media type is an image (`image/*`). Image
    /// bytes can only reach a provider through a provider media/content
    /// part; until the adapters carry attachment bytes, admission refuses
    /// these loudly (the composer draft is retained by the client).
    pub fn is_image(&self) -> bool {
        self.mime.starts_with("image/")
    }
}

/// Validate one canonical mime type: lowercase `type/subtype`, RFC-2045-ish
/// token characters only, bounded length. Uppercase is refused (the write
/// path canonicalizes; a stored/decoded uppercase value is corruption or a
/// hostile DTO, never silently accepted).
pub fn validate_mime(mime: &str) -> Result<(), Error> {
    if mime.is_empty() {
        return Err(Error::new(ErrorKind::Malformed, "attachment mime is empty"));
    }
    if mime.len() > MAX_ATTACHMENT_MIME_BYTES {
        return Err(Error::new(
            ErrorKind::Oversized,
            format!(
                "attachment mime of {} bytes exceeds MAX_ATTACHMENT_MIME_BYTES ({MAX_ATTACHMENT_MIME_BYTES})",
                mime.len()
            ),
        ));
    }
    if mime.trim() != mime || mime != mime.to_ascii_lowercase() {
        return Err(Error::new(
            ErrorKind::Malformed,
            format!("attachment mime {mime:?} is not canonical lowercase without surrounding whitespace"),
        ));
    }
    let Some((kind, subtype)) = mime.split_once('/') else {
        return Err(Error::new(
            ErrorKind::Malformed,
            format!("attachment mime {mime:?} is not type/subtype"),
        ));
    };
    if kind.is_empty() || subtype.is_empty() || subtype.contains('/') {
        return Err(Error::new(
            ErrorKind::Malformed,
            format!("attachment mime {mime:?} is not type/subtype"),
        ));
    }
    for token in [kind, subtype] {
        let mut chars = token.chars();
        let first = chars.next().expect("non-empty token");
        if !first.is_ascii_alphanumeric() {
            return Err(Error::new(
                ErrorKind::Malformed,
                format!("attachment mime {mime:?} starts a token with a non-alphanumeric byte"),
            ));
        }
        if !chars.all(is_token_char) {
            return Err(Error::new(
                ErrorKind::Malformed,
                format!("attachment mime {mime:?} carries a non-token byte"),
            ));
        }
    }
    Ok(())
}

fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '!' | '#' | '$' | '&' | '^' | '_' | '.' | '+' | '-')
}

/// Validate one attachment filename: a bounded DISPLAY label, never a path.
/// Empty/whitespace-padded names, control characters, any path separator,
/// `.`/`..` and over-long names are typed refusals.
pub fn validate_filename(name: &str) -> Result<(), Error> {
    if name.is_empty() || name.trim() != name {
        return Err(Error::new(
            ErrorKind::Malformed,
            format!("attachment filename {name:?} is empty or padded with whitespace"),
        ));
    }
    if name.len() > MAX_ATTACHMENT_FILENAME_BYTES {
        return Err(Error::new(
            ErrorKind::Oversized,
            format!(
                "attachment filename of {} bytes exceeds MAX_ATTACHMENT_FILENAME_BYTES ({MAX_ATTACHMENT_FILENAME_BYTES})",
                name.len()
            ),
        ));
    }
    if name == "." || name == ".." {
        return Err(Error::new(
            ErrorKind::Malformed,
            format!("attachment filename {name:?} is a traversal segment"),
        ));
    }
    if name.chars().any(char::is_control) {
        return Err(Error::new(
            ErrorKind::Malformed,
            format!("attachment filename {name:?} carries control characters"),
        ));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(Error::new(
            ErrorKind::Malformed,
            format!("attachment filename {name:?} contains a path separator; filenames are labels, not paths"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> FileHash {
        FileHash::from([byte; 32])
    }

    #[test]
    fn canonical_attachment_roundtrips_and_lowercases_mime() {
        let id = AttachmentId::new(hash(1), "Image/PNG", Some("shot.png"), 17).unwrap();
        assert_eq!(id.mime, "image/png");
        assert!(id.is_image());
        assert_eq!(id.filename.as_deref(), Some("shot.png"));
        let json = serde_json::to_string(&id).unwrap();
        let back: AttachmentId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
        // Unknown fields are a hostile DTO, never ignored.
        let err = serde_json::from_str::<AttachmentId>(
            r#"{"digest":"0000000000000000000000000000000000000000000000000000000000000000","mime":"text/plain","filename":null,"size":1,"evil":1}"#,
        );
        assert!(err.is_err());
    }

    #[test]
    fn hostile_mimes_are_typed_refusals() {
        for mime in [
            "",
            "   ",
            "text",
            "text/",
            "/plain",
            "text/plain/extra",
            "text /plain",
            "TEXT/PLAIN",
            "text/pla in",
            "te\u{7}xt/plain",
        ] {
            let err = validate_mime(mime).expect_err(mime);
            assert!(
                matches!(err.kind, ErrorKind::Malformed | ErrorKind::Oversized),
                "mime {mime:?} => {err}"
            );
        }
        let long = format!("text/{}", "a".repeat(MAX_ATTACHMENT_MIME_BYTES));
        assert_eq!(validate_mime(&long).unwrap_err().kind, ErrorKind::Oversized);
        assert!(validate_mime("application/pdf").is_ok());
        assert!(validate_mime("text/x-rust+md").is_ok());
    }

    #[test]
    fn hostile_filenames_are_typed_refusals() {
        for name in [
            "",
            " padded.png",
            "padded.png ",
            ".",
            "..",
            "a/b.png",
            "a\\b.png",
            "..\\secrets",
            "bad\0name",
            "bad\nname",
        ] {
            let err = validate_filename(name).expect_err(name);
            assert!(
                matches!(err.kind, ErrorKind::Malformed | ErrorKind::Oversized),
                "filename {name:?} => {err}"
            );
        }
        let long = "a".repeat(MAX_ATTACHMENT_FILENAME_BYTES + 1);
        assert_eq!(
            validate_filename(&long).unwrap_err().kind,
            ErrorKind::Oversized
        );
    }

    #[test]
    fn oversized_size_is_refused() {
        let id = AttachmentId {
            digest: hash(2),
            mime: "application/pdf".into(),
            filename: None,
            size: MAX_ATTACHMENT_BYTES + 1,
        };
        assert_eq!(id.validate().unwrap_err().kind, ErrorKind::Oversized);
        let ok = AttachmentId {
            size: MAX_ATTACHMENT_BYTES,
            ..id.clone()
        };
        assert!(ok.validate().is_ok());
    }
}
