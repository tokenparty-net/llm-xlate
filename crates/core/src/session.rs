//! Session-affinity capture and emission (protocol-neutral).
//!
//! A **session id** identifies a logical client session so the router can route it with
//! affinity (sticky routing, prompt-cache locality). Different clients surface it in different
//! places; [`capture_session`] reads a request's candidate slots in a fixed priority order and
//! folds them into a single [`SessionConfig`], noting any conflicting duplicates. On the way
//! out, [`session_emit`] renders the captured id into the first place the backend accepts one
//! ([`SessionCap`](crate::caps::SessionCap)).
//!
//! ## Incoming priority (highest first)
//!
//! 1. `prompt_cache_key` body field
//! 2. `session_id` body field
//! 3. `x-session-affinity` header
//! 4. `x-opencode-session` header
//! 5. `x-claude-code-session-id` header
//! 6. `x-session-id` header
//!
//! The first present, non-empty candidate wins. Any *other* present candidate whose value
//! differs from the winner is recorded in [`SessionConfig::conflicting_sources`] so lowering
//! can emit one degradation.

use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value};

use crate::caps::{Capabilities, SessionSink};
use crate::degrade::Degradations;
use crate::ir::SessionConfig;

/// The session-id request headers, in descending priority order.
///
/// Kept as a public constant so codecs treat these headers as *consumed* (they must not also
/// be forwarded verbatim) and so the list has a single source of truth.
pub const SESSION_HEADERS: [&str; 4] =
    ["x-session-affinity", "x-opencode-session", "x-claude-code-session-id", "x-session-id"];

/// Capture a session id from a request's candidate slots (see the module docs for priority).
///
/// `prompt_cache_key` and `session_id_field` are the two body-field candidates each codec
/// extracts from its own dialect (either may be `None`); `hdrs` supplies the three header
/// candidates. Empty (or whitespace-only) values are treated as absent.
pub fn capture_session(
    prompt_cache_key: Option<&str>,
    session_id_field: Option<&str>,
    hdrs: &HeaderMap,
) -> SessionConfig {
    // Collect (source-label, value) for every present, non-empty candidate, in priority order.
    let mut candidates: Vec<(&str, String)> = Vec::new();
    let mut push = |label: &'static str, value: Option<&str>| {
        if let Some(v) = value {
            let t = v.trim();
            if !t.is_empty() {
                candidates.push((label, t.to_string()));
            }
        }
    };
    push("prompt_cache_key", prompt_cache_key);
    push("session_id", session_id_field);
    for name in SESSION_HEADERS {
        let v = hdrs.get(name).and_then(|h| h.to_str().ok());
        push(name, v);
    }

    let mut cfg = SessionConfig::default();
    if let Some((_, chosen)) = candidates.first().cloned() {
        // Any lower-priority candidate that carries a *different* value is a conflict.
        for (label, value) in candidates.iter().skip(1) {
            if *value != chosen {
                cfg.conflicting_sources.push((*label).to_string());
            }
        }
        cfg.id = Some(chosen);
    }
    cfg
}

/// Where a session id should be written on an outgoing request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEmit {
    /// Set a top-level request body field.
    Field {
        /// Field name.
        name: String,
        /// Session id value.
        value: String,
    },
    /// Set a request header.
    Header {
        /// Header name.
        name: String,
        /// Session id value.
        value: String,
    },
}

/// Decide where to emit the captured session `id` for a backend with the given `caps`.
///
/// Returns the first place [`SessionCap::accepts`](crate::caps::SessionCap::accepts) lists, or
/// `None` when there is nothing to emit. When `id` is present but the backend accepts no
/// session id, a single `Dropped` degradation is recorded (the id is best-effort affinity data,
/// so this never fails the request — the [required](crate::caps::SessionCap::required) check is
/// a separate lowering step).
///
/// `cache_key` is the request's `prompt_cache_key` value, if any: when the session id is
/// exactly that value the id is already carried (or already reported as dropped) by the
/// prompt-cache path, so the redundant `session_id=dropped` degradation is suppressed.
pub fn session_emit(
    caps: &Capabilities,
    id: Option<&str>,
    cache_key: Option<&str>,
    degr: &mut Degradations,
) -> Option<SessionEmit> {
    let id = id?;
    match caps.session.primary_sink() {
        Some(SessionSink::Field(name)) => {
            Some(SessionEmit::Field { name: name.clone(), value: id.to_string() })
        }
        Some(SessionSink::Header(name)) => {
            Some(SessionEmit::Header { name: name.clone(), value: id.to_string() })
        }
        None => {
            if Some(id) != cache_key {
                degr.dropped("session_id", "target accepts no session id");
            }
            None
        }
    }
}

/// Emit the captured session `id` into an outgoing request's `body` map and `headers`, gated
/// on `caps`. A convenience wrapper over [`session_emit`] for codecs whose request body is a
/// [`serde_json::Map`]: a [`SessionEmit::Field`] is inserted into `body` (never clobbering an
/// existing entry — e.g. a `prompt_cache_key` already written by the cache path), and a
/// [`SessionEmit::Header`] into `headers`. A header value that is not a valid HTTP header value
/// is skipped. Records the same degradations as [`session_emit`].
pub fn apply_session(
    caps: &Capabilities,
    id: Option<&str>,
    cache_key: Option<&str>,
    body: &mut Map<String, Value>,
    headers: &mut HeaderMap,
    degr: &mut Degradations,
) {
    match session_emit(caps, id, cache_key, degr) {
        Some(SessionEmit::Field { name, value }) => {
            body.entry(name).or_insert_with(|| Value::from(value));
        }
        Some(SessionEmit::Header { name, value }) => {
            if let (Ok(n), Ok(v)) =
                (HeaderName::from_bytes(name.as_bytes()), HeaderValue::from_str(&value))
            {
                headers.insert(n, v);
            }
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{SessionCap, Tri};
    use http::{HeaderName, HeaderValue};

    fn hdrs(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn prompt_cache_key_wins_over_everything() {
        let cfg = capture_session(
            Some("pck"),
            Some("sid"),
            &hdrs(&[("x-session-id", "hdr")]),
        );
        assert_eq!(cfg.id.as_deref(), Some("pck"));
        // session_id and x-session-id both differ from the winner.
        assert_eq!(cfg.conflicting_sources, vec!["session_id", "x-session-id"]);
    }

    #[test]
    fn header_priority_order() {
        let cfg = capture_session(
            None,
            None,
            &hdrs(&[("x-session-id", "c"), ("x-opencode-session", "b"), ("x-session-affinity", "a")]),
        );
        assert_eq!(cfg.id.as_deref(), Some("a"));
        assert_eq!(cfg.conflicting_sources, vec!["x-opencode-session", "x-session-id"]);
    }

    #[test]
    fn claude_code_header_is_captured() {
        let cfg = capture_session(None, None, &hdrs(&[("x-claude-code-session-id", "cc-1")]));
        assert_eq!(cfg.id.as_deref(), Some("cc-1"));
        assert!(cfg.conflicting_sources.is_empty());
    }

    #[test]
    fn claude_code_header_priority() {
        // Ranks below the explicit affinity override but above the generic x-session-id.
        let cfg = capture_session(
            None,
            None,
            &hdrs(&[
                ("x-session-affinity", "aff"),
                ("x-claude-code-session-id", "cc"),
                ("x-session-id", "generic"),
            ]),
        );
        assert_eq!(cfg.id.as_deref(), Some("aff"));
        assert_eq!(cfg.conflicting_sources, vec!["x-claude-code-session-id", "x-session-id"]);

        // Without the affinity header, the Claude Code id wins over the generic one.
        let cfg = capture_session(
            None,
            None,
            &hdrs(&[("x-claude-code-session-id", "cc"), ("x-session-id", "generic")]),
        );
        assert_eq!(cfg.id.as_deref(), Some("cc"));
        assert_eq!(cfg.conflicting_sources, vec!["x-session-id"]);
    }

    #[test]
    fn equal_duplicates_are_not_conflicts() {
        let cfg = capture_session(Some("same"), Some("same"), &hdrs(&[("x-session-id", "same")]));
        assert_eq!(cfg.id.as_deref(), Some("same"));
        assert!(cfg.conflicting_sources.is_empty());
    }

    #[test]
    fn empty_values_are_absent() {
        let cfg = capture_session(Some("   "), Some("sid"), &HeaderMap::new());
        assert_eq!(cfg.id.as_deref(), Some("sid"));
        assert!(cfg.conflicting_sources.is_empty());
    }

    #[test]
    fn no_candidates_is_empty() {
        let cfg = capture_session(None, None, &HeaderMap::new());
        assert_eq!(cfg, SessionConfig::default());
    }

    fn caps_with(sink: Option<SessionSink>) -> Capabilities {
        let mut c = Capabilities::unknown();
        c.session = SessionCap { accepts: sink.map(|s| vec![s]), required: Tri::Unknown };
        c
    }

    #[test]
    fn emit_field_and_header() {
        let mut d = Degradations::new();
        let f = session_emit(&caps_with(Some(SessionSink::Field("prompt_cache_key".into()))), Some("x"), None, &mut d);
        assert_eq!(f, Some(SessionEmit::Field { name: "prompt_cache_key".into(), value: "x".into() }));
        assert!(d.is_empty());

        let h = session_emit(&caps_with(Some(SessionSink::Header("x-session-affinity".into()))), Some("x"), None, &mut d);
        assert_eq!(h, Some(SessionEmit::Header { name: "x-session-affinity".into(), value: "x".into() }));
        assert!(d.is_empty());
    }

    #[test]
    fn emit_drops_and_degrades_when_no_sink() {
        let mut d = Degradations::new();
        let out = session_emit(&caps_with(None), Some("x"), None, &mut d);
        assert!(out.is_none());
        assert_eq!(d.len(), 1);
        assert_eq!(d.iter().next().unwrap().field, "session_id");
    }

    #[test]
    fn emit_no_sink_degradation_suppressed_when_id_is_cache_key() {
        let mut d = Degradations::new();
        // The session id equals the prompt_cache_key, whose loss the cache path already reports.
        let out = session_emit(&caps_with(None), Some("x"), Some("x"), &mut d);
        assert!(out.is_none());
        assert!(d.is_empty());
    }

    #[test]
    fn emit_none_when_no_id() {
        let mut d = Degradations::new();
        let out = session_emit(&caps_with(None), None, None, &mut d);
        assert!(out.is_none());
        assert!(d.is_empty());
    }
}
