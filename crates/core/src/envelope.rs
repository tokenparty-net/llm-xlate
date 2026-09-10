//! The `rtr1.` opaque-reasoning envelope (plan §7.2).
//!
//! Provider-bound opaque blobs (Anthropic signatures / redacted thinking, OpenAI encrypted
//! reasoning) are wrapped so they can cross the client-facing boundary of a *different*
//! provider and be replayed later. The envelope is deterministic (no nonce): identical input
//! plus identical key yields a byte-identical string.
//!
//! Wire form: `"rtr1." + base64url_nopad(canonical_json)` where `canonical_json` is a compact
//! JSON object with keys in the fixed order `{v, p, m, k, d, h}`:
//! `v` = version (1), `p` = producing family label, `m` = producing model (may be empty),
//! `k` = [`OpaqueKind`] slug, `d` = the opaque data, `h` = the base64url MAC.
//!
//! **MAC framing:** `h = HMAC-SHA256(key, frame)` where `frame` is the length-prefixed
//! concatenation of `p`, `m`, `k`, `d` — each field is a 4-byte big-endian byte length
//! followed by the field's UTF-8 bytes. Length-prefixing makes the MAC unambiguous (no
//! delimiter can be forged across field boundaries). `h` itself is excluded from the frame.

use base64::Engine;
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::Sha256;

use crate::error::XlateError;
use crate::ir::{OpaqueBlob, OpaqueKind, ProviderFamily};

type HmacSha256 = Hmac<Sha256>;

const PREFIX: &str = "rtr1.";
const VERSION: u64 = 1;

/// Seals and opens [`OpaqueBlob`]s. Cheap to clone. Its [`std::fmt::Debug`] redacts the key.
#[derive(Clone)]
pub struct Sealer {
    key: Vec<u8>,
}

impl std::fmt::Debug for Sealer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sealer")
            .field("key", &format_args!("<redacted, {} bytes>", self.key.len()))
            .finish()
    }
}

/// Errors from [`Sealer::open`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EnvelopeError {
    /// The string does not start with the `rtr1.` prefix.
    #[error("not an rtr1 envelope")]
    NotEnvelope,
    /// The envelope body could not be base64-decoded or JSON-parsed, or is missing fields.
    #[error("malformed envelope: {0}")]
    Malformed(String),
    /// The envelope version is not supported.
    #[error("unsupported envelope version: {0}")]
    UnsupportedVersion(u64),
    /// The opaque-kind slug was not recognized.
    #[error("unknown opaque kind: {0}")]
    UnknownKind(String),
    /// The HMAC did not verify (wrong key or tampered payload).
    #[error("envelope signature verification failed")]
    BadSignature,
}

impl From<EnvelopeError> for XlateError {
    fn from(e: EnvelopeError) -> Self {
        XlateError::upstream_malformed(format!("reasoning envelope: {e}"))
    }
}

fn kind_slug(k: OpaqueKind) -> &'static str {
    match k {
        OpaqueKind::Signature => "signature",
        OpaqueKind::Redacted => "redacted",
        OpaqueKind::Encrypted => "encrypted",
        OpaqueKind::Compaction => "compaction",
    }
}

fn kind_from_slug(s: &str) -> Option<OpaqueKind> {
    match s {
        "signature" => Some(OpaqueKind::Signature),
        "redacted" => Some(OpaqueKind::Redacted),
        "encrypted" => Some(OpaqueKind::Encrypted),
        "compaction" => Some(OpaqueKind::Compaction),
        _ => None,
    }
}

fn family_from_label(s: &str) -> ProviderFamily {
    match s {
        "anthropic" => ProviderFamily::Anthropic,
        "openai" => ProviderFamily::OpenAI,
        other => ProviderFamily::Other(other.to_string()),
    }
}

fn frame(p: &str, m: &str, k: &str, d: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    for s in [p, m, k, d] {
        buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    buf
}

impl Sealer {
    /// Construct a sealer from a key of any length.
    pub fn new(key: &[u8]) -> Self {
        Self { key: key.to_vec() }
    }

    /// Whether a string is an `rtr1.` envelope (cheap prefix check).
    pub fn is_envelope(s: &str) -> bool {
        s.starts_with(PREFIX)
    }

    fn mac(&self, frame: &[u8]) -> Vec<u8> {
        let mut mac =
            HmacSha256::new_from_slice(&self.key).expect("HMAC-SHA256 accepts any key length");
        mac.update(frame);
        mac.finalize().into_bytes().to_vec()
    }

    /// Seal an [`OpaqueBlob`] into a deterministic `rtr1.` envelope string.
    pub fn seal(&self, blob: &OpaqueBlob) -> String {
        let p = blob.family.label();
        let m = blob.model.as_deref().unwrap_or("");
        let k = kind_slug(blob.kind);
        let d = blob.data.as_str();

        let mac = self.mac(&frame(p, m, k, d));
        let h = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(mac);

        let mut map = serde_json::Map::new();
        map.insert("v".to_string(), Value::from(VERSION));
        map.insert("p".to_string(), Value::from(p));
        map.insert("m".to_string(), Value::from(m));
        map.insert("k".to_string(), Value::from(k));
        map.insert("d".to_string(), Value::from(d));
        map.insert("h".to_string(), Value::from(h));
        let json = serde_json::to_string(&Value::Object(map)).unwrap_or_default();

        let body = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json.as_bytes());
        format!("{PREFIX}{body}")
    }

    /// Open an `rtr1.` envelope, verifying the HMAC.
    pub fn open(&self, s: &str) -> Result<OpaqueBlob, EnvelopeError> {
        let body = s.strip_prefix(PREFIX).ok_or(EnvelopeError::NotEnvelope)?;
        let json = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(body.as_bytes())
            .map_err(|e| EnvelopeError::Malformed(format!("base64: {e}")))?;
        let value: Value = serde_json::from_slice(&json)
            .map_err(|e| EnvelopeError::Malformed(format!("json: {e}")))?;

        let get_str = |key: &str| -> Result<String, EnvelopeError> {
            value
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| EnvelopeError::Malformed(format!("missing field {key}")))
        };

        let v = value
            .get("v")
            .and_then(Value::as_u64)
            .ok_or_else(|| EnvelopeError::Malformed("missing field v".to_string()))?;
        if v != VERSION {
            return Err(EnvelopeError::UnsupportedVersion(v));
        }
        let p = get_str("p")?;
        let m = get_str("m")?;
        let k = get_str("k")?;
        let d = get_str("d")?;
        let h = get_str("h")?;

        let expected = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(h.as_bytes())
            .map_err(|e| EnvelopeError::Malformed(format!("mac base64: {e}")))?;

        let mut mac =
            HmacSha256::new_from_slice(&self.key).expect("HMAC-SHA256 accepts any key length");
        mac.update(&frame(&p, &m, &k, &d));
        mac.verify_slice(&expected).map_err(|_| EnvelopeError::BadSignature)?;

        let kind = kind_from_slug(&k).ok_or_else(|| EnvelopeError::UnknownKind(k.clone()))?;
        Ok(OpaqueBlob {
            family: family_from_label(&p),
            kind,
            data: d,
            model: if m.is_empty() { None } else { Some(m) },
        })
    }

    /// Open `s` if it is a valid envelope; otherwise treat it as a **native** blob for the
    /// given family/kind (used at the provider-facing boundary). Never fails: a string that
    /// looks like an envelope but fails to open falls back to native with the raw string.
    pub fn open_or_native(
        &self,
        s: &str,
        native_family: ProviderFamily,
        native_kind: OpaqueKind,
    ) -> OpaqueBlob {
        if Self::is_envelope(s) {
            if let Ok(blob) = self.open(s) {
                return blob;
            }
        }
        OpaqueBlob { family: native_family, kind: native_kind, data: s.to_string(), model: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn blob() -> OpaqueBlob {
        OpaqueBlob {
            family: ProviderFamily::Anthropic,
            kind: OpaqueKind::Signature,
            data: "sig-data-123".to_string(),
            model: Some("claude-opus-4-8".to_string()),
        }
    }

    #[test]
    fn round_trip() {
        let s = Sealer::new(b"key");
        let sealed = s.seal(&blob());
        assert!(Sealer::is_envelope(&sealed));
        assert_eq!(s.open(&sealed).unwrap(), blob());
    }

    #[test]
    fn round_trip_no_model() {
        let s = Sealer::new(b"key");
        let b = OpaqueBlob {
            family: ProviderFamily::OpenAI,
            kind: OpaqueKind::Encrypted,
            data: "ct".into(),
            model: None,
        };
        assert_eq!(s.open(&s.seal(&b)).unwrap(), b);
    }

    #[test]
    fn deterministic() {
        let s = Sealer::new(b"key");
        assert_eq!(s.seal(&blob()), s.seal(&blob()));
    }

    #[test]
    fn wrong_key_fails() {
        let sealed = Sealer::new(b"key-a").seal(&blob());
        assert_eq!(Sealer::new(b"key-b").open(&sealed), Err(EnvelopeError::BadSignature));
    }

    #[test]
    fn tamper_detection() {
        let s = Sealer::new(b"key");
        let sealed = s.seal(&blob());
        // Flip a character in the base64 body.
        let mut chars: Vec<char> = sealed.chars().collect();
        let last = chars.len() - 1;
        chars[last] = if chars[last] == 'A' { 'B' } else { 'A' };
        let tampered: String = chars.into_iter().collect();
        assert!(s.open(&tampered).is_err());
    }

    #[test]
    fn not_envelope() {
        assert_eq!(Sealer::new(b"k").open("native-sig"), Err(EnvelopeError::NotEnvelope));
        assert!(!Sealer::is_envelope("native-sig"));
    }

    #[test]
    fn open_or_native_falls_back() {
        let s = Sealer::new(b"k");
        let native = s.open_or_native("native-sig", ProviderFamily::Anthropic, OpaqueKind::Signature);
        assert_eq!(native.data, "native-sig");
        assert_eq!(native.family, ProviderFamily::Anthropic);

        let sealed = s.seal(&blob());
        let opened = s.open_or_native(&sealed, ProviderFamily::OpenAI, OpaqueKind::Encrypted);
        assert_eq!(opened, blob());
    }

    #[test]
    fn debug_redacts_key() {
        let dbg = format!("{:?}", Sealer::new(b"supersecret"));
        assert!(!dbg.contains("supersecret"));
        assert!(dbg.contains("redacted"));
    }
}
