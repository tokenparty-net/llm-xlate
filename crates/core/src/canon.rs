//! Canonical JSON helpers: compact serialization with **no key sorting** (user key order is
//! preserved via `serde_json`'s `preserve_order`), parsing that maps failures into
//! [`XlateError`], order-preserving JSON-schema keyword stripping, and `data:` URL codecs.

use base64::Engine;
use bytes::Bytes;
use serde::Serialize;
use serde_json::Value;

use crate::error::XlateError;
use crate::ir::JsonText;

/// Serialize compactly with no key sorting. Well-formed IR never fails to serialize; on the
/// impossible failure an empty buffer is returned rather than panicking.
pub fn to_bytes<T: Serialize>(value: &T) -> Bytes {
    serde_json::to_vec(value).map(Bytes::from).unwrap_or_default()
}

/// Serialize compactly to a `String` (no key sorting).
pub fn to_string<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// Parse client-supplied bytes into a [`Value`]; failures are [`XlateError::invalid_request`].
pub fn parse(body: &[u8]) -> Result<Value, XlateError> {
    serde_json::from_slice(body)
        .map_err(|e| XlateError::invalid_request(format!("invalid JSON: {e}")))
}

/// Parse upstream (provider) bytes into a [`Value`]; failures are
/// [`XlateError::upstream_malformed`].
pub fn parse_upstream(body: &[u8]) -> Result<Value, XlateError> {
    serde_json::from_slice(body)
        .map_err(|e| XlateError::upstream_malformed(format!("upstream returned invalid JSON: {e}")))
}

/// Remove the given JSON-schema keyword keys from `value`, recursively and **order-
/// preserving**.
///
/// Keys are removed at any depth, **except** keys that sit directly under a `properties`,
/// `$defs`, or `definitions` map — those are property/definition *names*, not keywords, so
/// they are kept (their schema values are still recursed into). Array elements are recursed
/// into as schemas.
pub fn remove_keywords_recursive(value: &mut Value, keywords: &[String]) {
    fn go(v: &mut Value, keywords: &[String], keys_are_names: bool) {
        match v {
            Value::Object(map) => {
                // Rebuild to guarantee order preservation regardless of the backing map's
                // removal semantics.
                let old = std::mem::replace(map, serde_json::Map::new());
                for (k, mut child) in old {
                    if !keys_are_names && keywords.iter().any(|kw| kw == &k) {
                        continue;
                    }
                    let child_names = !keys_are_names
                        && matches!(k.as_str(), "properties" | "$defs" | "definitions");
                    go(&mut child, keywords, child_names);
                    map.insert(k, child);
                }
            }
            Value::Array(arr) => {
                for e in arr.iter_mut() {
                    go(e, keywords, false);
                }
            }
            _ => {}
        }
    }
    go(value, keywords, false);
}

/// Recursively set `additionalProperties: false` on every JSON-Schema *object* node that does not
/// already specify it. Required by OpenAI strict mode ("'additionalProperties' is required to be
/// supplied and to be false") and by current Anthropic structured output (a missing one 400s) —
/// live-verified 2026-09-10. Applied when a schema is lowered with `strict` still on.
///
/// A node is treated as an object schema when it has `"type":"object"` or a `properties` map. The
/// walk descends into `properties`/`$defs`/`definitions` values (which are name→schema maps) as
/// schemas, and into arrays (`anyOf`/`allOf`/`oneOf`/`prefixItems`) and `items` as schemas.
pub fn ensure_additional_properties_false_recursive(value: &mut Value) {
    fn go(v: &mut Value, keys_are_names: bool) {
        match v {
            Value::Object(map) => {
                if !keys_are_names {
                    let is_object_schema = map.get("type").and_then(|t| t.as_str()) == Some("object")
                        || map.contains_key("properties");
                    if is_object_schema && !map.contains_key("additionalProperties") {
                        map.insert("additionalProperties".to_string(), Value::Bool(false));
                    }
                }
                for (k, child) in map.iter_mut() {
                    let child_names =
                        !keys_are_names && matches!(k.as_str(), "properties" | "$defs" | "definitions");
                    go(child, child_names);
                }
            }
            Value::Array(arr) => {
                for e in arr.iter_mut() {
                    go(e, false);
                }
            }
            _ => {}
        }
    }
    go(value, false);
}

/// Whether every JSON-Schema *object* node with a `properties` map lists **all** of its property
/// names in `required` — the extra constraint OpenAI strict mode enforces beyond
/// `additionalProperties: false` ("'required' must contain every key in 'properties'"). Returns
/// `false` as soon as one object node has a property missing from its `required` array (or has
/// `properties` but no `required` at all).
///
/// Used to decide whether a schema can *stay* strict: a strict schema with optional properties is
/// invalid to OpenAI, so rather than silently promote those properties to required (a semantic
/// change) or emit an invalid strict schema, the lowering passes downgrade `strict` to `false`.
///
/// The walk mirrors [`ensure_additional_properties_false_recursive`]: it descends into
/// `properties`/`$defs`/`definitions` values as schemas, and into arrays (`anyOf`/`allOf`/`oneOf`/
/// `prefixItems`) and `items` as schemas.
pub fn strict_schema_required_covers_properties(value: &Value) -> bool {
    fn go(v: &Value, keys_are_names: bool) -> bool {
        match v {
            Value::Object(map) => {
                if !keys_are_names {
                    if let Some(Value::Object(props)) = map.get("properties") {
                        let required: std::collections::BTreeSet<&str> = map
                            .get("required")
                            .and_then(|r| r.as_array())
                            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                            .unwrap_or_default();
                        if props.keys().any(|k| !required.contains(k.as_str())) {
                            return false;
                        }
                    }
                }
                map.iter().all(|(k, child)| {
                    let child_names =
                        !keys_are_names && matches!(k.as_str(), "properties" | "$defs" | "definitions");
                    go(child, child_names)
                })
            }
            Value::Array(arr) => arr.iter().all(|e| go(e, false)),
            _ => true,
        }
    }
    go(value, false)
}

/// `data:` URL parsing and building.
pub mod data_url {
    use super::*;

    /// Parse a `data:` URL into `(media_type, bytes)`. Handles both `;base64,` payloads and
    /// (percent-decoded) plain payloads. Returns `None` if the input is not a `data:` URL.
    pub fn parse(s: &str) -> Option<(String, Bytes)> {
        let rest = s.strip_prefix("data:")?;
        let comma = rest.find(',')?;
        let meta = &rest[..comma];
        let data = &rest[comma + 1..];

        let mut media_type = String::new();
        let mut is_base64 = false;
        for (i, part) in meta.split(';').enumerate() {
            if part.eq_ignore_ascii_case("base64") {
                is_base64 = true;
            } else if i == 0 {
                media_type = part.to_string();
            }
        }
        if media_type.is_empty() {
            media_type = "text/plain".to_string();
        }

        let bytes = if is_base64 {
            base64::engine::general_purpose::STANDARD
                .decode(data.as_bytes())
                .ok()?
        } else {
            percent_decode(data)
        };
        Some((media_type, Bytes::from(bytes)))
    }

    /// Build a base64 `data:` URL for the given media type and bytes.
    pub fn build(media_type: &str, bytes: &[u8]) -> String {
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        format!("data:{media_type};base64,{b64}")
    }

    fn percent_decode(s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(h), Some(l)) = (hi, lo) {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        out
    }
}

/// Default filenames for documents that carry no client-supplied title.
pub mod filename {
    /// Synthesize a default document filename from its media type. OpenAI's Chat `file` and
    /// Responses `input_file` parts require a `filename` alongside base64 `file_data` (the
    /// extension is how OpenAI infers the file type) and reject the request otherwise. An
    /// Anthropic PDF has no filename concept and often no title, so a cross-family document may
    /// arrive title-less; give it a name derived from the media type (e.g. `application/pdf`
    /// → `document.pdf`, anything unrecognized → `document.bin`).
    pub fn for_media_type(media_type: &str) -> String {
        let ext = match media_type.trim().to_ascii_lowercase().as_str() {
            "application/pdf" => "pdf",
            "text/plain" => "txt",
            "text/markdown" => "md",
            "text/html" => "html",
            "text/csv" => "csv",
            "application/json" => "json",
            "application/xml" | "text/xml" => "xml",
            "application/msword" => "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "docx",
            "application/vnd.ms-excel" => "xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx",
            _ => "bin",
        };
        format!("document.{ext}")
    }
}

/// [`JsonText`] validation and conversion.
pub mod json_text {
    use super::*;

    /// Validate that a [`JsonText`] holds well-formed JSON.
    pub fn validate(jt: &JsonText) -> Result<(), XlateError> {
        to_value(jt).map(|_| ())
    }

    /// Parse a [`JsonText`] into a [`Value`] (object key order preserved). Failures are
    /// [`XlateError::invalid_request`].
    pub fn to_value(jt: &JsonText) -> Result<Value, XlateError> {
        serde_json::from_str(jt.as_str())
            .map_err(|e| XlateError::invalid_request(format!("invalid tool arguments JSON: {e}")))
    }

    /// Render a [`Value`] as compact [`JsonText`] (no key sorting).
    pub fn from_value(v: &Value) -> JsonText {
        JsonText::new(v.to_string())
    }
}

/// Provider limits on end-user identifiers and a deterministic way to fit a longer one.
///
/// OpenAI rejects `user` / `safety_identifier` values longer than 64 characters; Anthropic
/// rejects `metadata.user_id` longer than 256. Clients such as Claude Code send long composite
/// identifiers, so a cross-surface hop needs a stable replacement: the SHA-256 hex digest
/// (64 ASCII chars) truncated to the target's limit. Same input ⇒ same output, distinct users
/// stay distinct, and the rewrite is reported as a [`Degradation`](crate::degrade::Degradation).
pub mod identifier {
    use sha2::{Digest, Sha256};

    /// Maximum length of OpenAI `user` / `safety_identifier`.
    pub const OPENAI_MAX: usize = 64;
    /// Maximum length of Anthropic `metadata.user_id`.
    pub const ANTHROPIC_MAX: usize = 256;

    /// Returns `Some(replacement)` when `value` exceeds `max` characters, else `None` (keep verbatim).
    pub fn fit(value: &str, max: usize) -> Option<String> {
        if value.chars().count() <= max {
            return None;
        }
        let digest = Sha256::digest(value.as_bytes());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        Some(hex.chars().take(max).collect())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn short_values_are_kept() {
            assert_eq!(fit("user-123", OPENAI_MAX), None);
            assert_eq!(fit(&"x".repeat(64), OPENAI_MAX), None);
        }

        #[test]
        fn long_values_become_a_deterministic_digest_within_the_limit() {
            let long = "user_".to_string() + &"a".repeat(150);
            let a = fit(&long, OPENAI_MAX).unwrap();
            let b = fit(&long, OPENAI_MAX).unwrap();
            assert_eq!(a, b);
            assert_eq!(a.len(), 64);
            assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
            let other = fit(&(long.clone() + "b"), OPENAI_MAX).unwrap();
            assert_ne!(a, other);
            assert_eq!(fit(&"y".repeat(300), 32).unwrap().len(), 32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn to_bytes_no_key_sorting() {
        let v = json!({"b": 1, "a": 2, "c": 3});
        assert_eq!(&to_bytes(&v)[..], br#"{"b":1,"a":2,"c":3}"#);
    }

    #[test]
    fn parse_ok_and_err() {
        assert!(parse(b"{\"x\":1}").is_ok());
        let e = parse(b"{bad").unwrap_err();
        assert_eq!(e.kind, crate::error::ErrorKind::InvalidRequest);
        let e = parse_upstream(b"nope").unwrap_err();
        assert_eq!(e.kind, crate::error::ErrorKind::UpstreamMalformed);
    }

    #[test]
    fn remove_keywords_top_level_and_nested() {
        let mut v = json!({
            "type": "object",
            "minLength": 3,
            "properties": {
                "minLength": { "type": "string", "pattern": "x", "minLength": 1 },
                "n": { "type": "integer", "format": "int64" }
            }
        });
        remove_keywords_recursive(&mut v, &["minLength".into(), "pattern".into(), "format".into()]);
        // Top-level `minLength` keyword removed; the property *named* `minLength` kept, but
        // the `pattern`/`minLength` keywords inside its schema removed; `format` inside `n` removed.
        assert_eq!(
            v,
            json!({
                "type": "object",
                "properties": {
                    "minLength": { "type": "string" },
                    "n": { "type": "integer" }
                }
            })
        );
    }

    #[test]
    fn remove_keywords_preserves_order() {
        let mut v = json!({"z": 1, "minLength": 2, "a": 3});
        remove_keywords_recursive(&mut v, &["minLength".into()]);
        let s = to_string(&v);
        assert_eq!(s, r#"{"z":1,"a":3}"#);
    }

    #[test]
    fn remove_keywords_recurses_defs_values() {
        let mut v = json!({
            "$defs": { "Foo": { "type": "string", "pattern": "^x$" } },
            "allOf": [ { "pattern": "y" } ]
        });
        remove_keywords_recursive(&mut v, &["pattern".into()]);
        assert_eq!(
            v,
            json!({ "$defs": { "Foo": { "type": "string" } }, "allOf": [ {} ] })
        );
    }

    #[test]
    fn additional_properties_false_injected_recursively() {
        let mut v = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" },
                "meta": { "type": "object", "properties": { "k": { "type": "string" } } }
            },
            "required": ["answer"],
            "$defs": { "Nested": { "type": "object", "properties": { "x": { "type": "number" } } } }
        });
        ensure_additional_properties_false_recursive(&mut v);
        assert_eq!(v["additionalProperties"], json!(false));
        assert_eq!(v["properties"]["meta"]["additionalProperties"], json!(false));
        assert_eq!(v["$defs"]["Nested"]["additionalProperties"], json!(false));
        // A leaf string schema is untouched.
        assert!(v["properties"]["answer"].get("additionalProperties").is_none());
    }

    #[test]
    fn additional_properties_existing_value_preserved() {
        let mut v = json!({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true});
        ensure_additional_properties_false_recursive(&mut v);
        // Do not clobber an explicit author choice.
        assert_eq!(v["additionalProperties"], json!(true));
    }

    #[test]
    fn strict_required_coverage() {
        // Complete: every property is required, at every level.
        let complete = json!({
            "type": "object",
            "properties": { "a": { "type": "string" }, "b": { "type": "object", "properties": { "x": { "type": "number" } }, "required": ["x"] } },
            "required": ["a", "b"]
        });
        assert!(strict_schema_required_covers_properties(&complete));
        // Top-level optional property.
        let missing_top = json!({
            "type": "object",
            "properties": { "a": { "type": "string" }, "b": { "type": "string" } },
            "required": ["a"]
        });
        assert!(!strict_schema_required_covers_properties(&missing_top));
        // A nested object with an uncovered property.
        let missing_nested = json!({
            "type": "object",
            "properties": { "a": { "type": "object", "properties": { "x": { "type": "number" }, "y": { "type": "number" } }, "required": ["x"] } },
            "required": ["a"]
        });
        assert!(!strict_schema_required_covers_properties(&missing_nested));
        // `properties` with no `required` at all is incomplete.
        assert!(!strict_schema_required_covers_properties(&json!({
            "type": "object", "properties": { "a": { "type": "string" } }
        })));
        // A property literally named `required` is not confused with the keyword.
        let named = json!({
            "type": "object",
            "properties": { "required": { "type": "string" } },
            "required": ["required"]
        });
        assert!(strict_schema_required_covers_properties(&named));
    }

    #[test]
    fn data_url_round_trip_base64() {
        let url = data_url::build("image/png", &[1, 2, 3, 4]);
        assert_eq!(url, "data:image/png;base64,AQIDBA==");
        let (mt, bytes) = data_url::parse(&url).unwrap();
        assert_eq!(mt, "image/png");
        assert_eq!(&bytes[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn data_url_plain_and_non_data() {
        let (mt, bytes) = data_url::parse("data:text/plain,Hello%20World").unwrap();
        assert_eq!(mt, "text/plain");
        assert_eq!(&bytes[..], b"Hello World");
        assert!(data_url::parse("https://example.com/x.png").is_none());
    }

    #[test]
    fn json_text_helpers() {
        let jt = JsonText::new(r#"{"b":1,"a":2}"#);
        assert!(json_text::validate(&jt).is_ok());
        let v = json_text::to_value(&jt).unwrap();
        assert_eq!(json_text::from_value(&v), JsonText::new(r#"{"b":1,"a":2}"#));
        assert!(json_text::validate(&JsonText::new("{oops")).is_err());
    }
}
