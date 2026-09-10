//! Bundled deterministic media assets referenced by probes via `$ASSET:<name>` and
//! `$ASSET_DATA_URL:<name>`.
//!
//! The bytes are generated once (not downloaded) and checked into `crates/e2e/assets/`; they are
//! embedded here with [`include_bytes!`] so the runtime [`AssetSource`] needs no file I/O and is
//! fully reproducible. The [`redact_assets`] pass rewrites any base64 (or data-URL) asset payload
//! back to its `$ASSET:<name>` placeholder before a request body is written to disk, so captures
//! stay small and reconstructible (plan §4).

use crate::probe::{b64encode, AssetSource};
use anyhow::{anyhow, Result};

/// A valid 1×1 opaque-red PNG.
pub const PNG_1PX: &[u8] = include_bytes!("../assets/png_1px.png");
/// A valid 1×1 baseline JPEG.
pub const JPEG_1PX: &[u8] = include_bytes!("../assets/jpeg_1px.jpg");
/// A valid one-page PDF containing the text `llm-xlate e2e`.
pub const PDF_1PAGE: &[u8] = include_bytes!("../assets/pdf_1page.pdf");
/// A tiny plain-text document.
pub const DOC_TXT: &[u8] = include_bytes!("../assets/doc.txt");

/// Every bundled asset as `(name, bytes, mime)`, in a stable order.
pub const ALL: [(&str, &[u8], &str); 4] = [
    ("png_1px", PNG_1PX, "image/png"),
    ("jpeg_1px", JPEG_1PX, "image/jpeg"),
    ("pdf_1page", PDF_1PAGE, "application/pdf"),
    ("doc", DOC_TXT, "text/plain"),
];

/// Look up an asset by name (without file extension).
fn find(name: &str) -> Option<(&'static [u8], &'static str)> {
    ALL.iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, b, m)| (*b, *m))
}

/// The bundled asset provider used at run time.
#[derive(Debug, Clone, Copy, Default)]
pub struct Assets;

impl AssetSource for Assets {
    fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        find(name)
            .map(|(b, _)| b.to_vec())
            .ok_or_else(|| anyhow!("unknown asset `{name}`"))
    }

    fn mime(&self, name: &str) -> Result<String> {
        find(name)
            .map(|(_, m)| m.to_string())
            .ok_or_else(|| anyhow!("unknown asset `{name}`"))
    }
}

/// The base64 (and data-URL) forms of every asset, paired with the placeholder they came from.
/// Longest payload first, so the data-URL form (which contains the base64 form) is replaced before
/// the bare base64 form.
fn reverse_table() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (name, bytes, mime) in ALL {
        let b64 = b64encode(bytes);
        let data_url = format!("data:{mime};base64,{b64}");
        out.push((data_url, format!("$ASSET_DATA_URL:{name}")));
        out.push((b64, format!("$ASSET:{name}")));
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.0.len()));
    out
}

/// Rewrite any embedded asset base64 / data-URL string back to its `$ASSET:<name>` placeholder,
/// throughout a JSON value. Used by the capture writer so on-disk requests stay small and never
/// carry a large payload verbatim.
pub fn redact_assets(value: &serde_json::Value) -> serde_json::Value {
    let table = reverse_table();
    redact_with(value, &table)
}

fn redact_with(value: &serde_json::Value, table: &[(String, String)]) -> serde_json::Value {
    match value {
        serde_json::Value::String(s) => {
            let mut cur = s.clone();
            for (payload, placeholder) in table {
                if cur.contains(payload.as_str()) {
                    cur = cur.replace(payload.as_str(), placeholder);
                }
            }
            serde_json::Value::String(cur)
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.iter().map(|e| redact_with(e, table)).collect())
        }
        serde_json::Value::Object(o) => {
            let mut m = serde_json::Map::new();
            for (k, v) in o {
                m.insert(k.clone(), redact_with(v, table));
            }
            serde_json::Value::Object(m)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha16(b: &[u8]) -> String {
        let d = Sha256::digest(b);
        hex(&d[..8])
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn png_is_valid_and_stable() {
        assert_eq!(&PNG_1PX[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(PNG_1PX.len(), 69);
        assert_eq!(sha16(PNG_1PX), "2e9b06dc65a4dec8");
    }

    #[test]
    fn jpeg_is_valid_and_stable() {
        assert_eq!(&JPEG_1PX[..2], b"\xff\xd8"); // SOI
        assert_eq!(&JPEG_1PX[JPEG_1PX.len() - 2..], b"\xff\xd9"); // EOI
        assert_eq!(sha16(JPEG_1PX), "64612ed9c33b31d0");
    }

    #[test]
    fn pdf_is_valid_and_stable() {
        assert_eq!(&PDF_1PAGE[..5], b"%PDF-");
        assert!(PDF_1PAGE.windows(4).any(|w| w == b"%%EO"));
        // Contains the required visible text.
        assert!(PDF_1PAGE
            .windows(b"llm-xlate e2e".len())
            .any(|w| w == b"llm-xlate e2e"));
        assert_eq!(sha16(PDF_1PAGE), "e880b282bb17b600");
    }

    #[test]
    fn doc_txt_is_text() {
        assert!(DOC_TXT.starts_with(b"llm-xlate e2e"));
    }

    #[test]
    fn asset_source_reads_bytes_and_mime() {
        let a = Assets;
        assert_eq!(a.bytes("png_1px").unwrap(), PNG_1PX);
        assert_eq!(a.mime("pdf_1page").unwrap(), "application/pdf");
        assert!(a.bytes("nope").is_err());
    }

    #[test]
    fn redact_replaces_base64_and_data_url() {
        let b64 = b64encode(PNG_1PX);
        let data_url = format!("data:image/png;base64,{b64}");
        let body = serde_json::json!({
            "a": b64,
            "b": data_url,
            "c": "unrelated string",
        });
        let out = redact_assets(&body);
        assert_eq!(out["a"], serde_json::json!("$ASSET:png_1px"));
        assert_eq!(out["b"], serde_json::json!("$ASSET_DATA_URL:png_1px"));
        assert_eq!(out["c"], serde_json::json!("unrelated string"));
    }
}
