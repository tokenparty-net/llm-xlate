//! Session-affinity capture, mapping, and emission through the full [`Translator`] pipeline.
//!
//! Capture priority + conflict detection are unit-tested in `llm-xlate-core`; these tests cover
//! the end-to-end behaviour the router relies on: an incoming session id (from any slot) is
//! emitted into the backend's configured place (body field or header), a differing duplicate is
//! surfaced as a degradation, and a backend that requires a session id rejects a request that
//! lacks one.

use llm_xlate::caps::{preset, Capabilities, SessionCap, SessionSink, Tri};
use llm_xlate::codec::TranslatorConfig;
use llm_xlate::degrade::DegradationKind;
use llm_xlate::error::ErrorKind;
use llm_xlate::ir::Protocol;
use llm_xlate::requirements::Resolutions;
use llm_xlate::{HeaderMap, Translator};

fn xl() -> Translator {
    Translator::new(TranslatorConfig::default())
}

/// gpt-4o (chat + responses) caps with a session sink overlaid.
fn gpt4o_session(accepts: Vec<SessionSink>, required: Tri) -> Capabilities {
    let mut c = preset::gpt4o();
    c.session = SessionCap { accepts: Some(accepts), required };
    c
}

fn hdrs(pairs: &[(&str, &str)]) -> HeaderMap {
    use llm_xlate::{HeaderName, HeaderValue};
    let mut h = HeaderMap::new();
    for (k, v) in pairs {
        h.insert(
            HeaderName::from_bytes(k.as_bytes()).unwrap(),
            HeaderValue::from_str(v).unwrap(),
        );
    }
    h
}

fn chat_body() -> &'static [u8] {
    br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#
}

/// An `x-session-id` header maps into the backend's `prompt_cache_key` body field.
#[test]
fn header_session_id_emitted_to_body_field() {
    let x = xl();
    let caps = gpt4o_session(vec![SessionSink::Field("prompt_cache_key".into())], Tri::Unknown);
    let (enc, _) = x
        .translate_request(
            Protocol::OaiChat,
            chat_body(),
            &hdrs(&[("x-session-id", "sess-1")]),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .unwrap();

    let body: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(body["prompt_cache_key"], "sess-1", "session id → prompt_cache_key: {body}");
    assert!(enc.degradations.is_empty(), "no degradations expected: {:?}", enc.degradations);
}

/// A `prompt_cache_key` field maps into the backend's affinity **header**.
#[test]
fn prompt_cache_key_emitted_to_header() {
    let x = xl();
    let caps = gpt4o_session(vec![SessionSink::Header("x-session-affinity".into())], Tri::Unknown);
    let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"prompt_cache_key":"pck-9"}"#;
    let (enc, _) = x
        .translate_request(
            Protocol::OaiChat,
            body,
            &HeaderMap::new(),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .unwrap();

    assert_eq!(
        enc.headers.get("x-session-affinity").map(|v| v.to_str().unwrap()),
        Some("pck-9"),
        "session id → affinity header",
    );
}

/// The first sink in `accepts` is the one used (the rest are reserved for future expansion).
#[test]
fn first_sink_wins() {
    let x = xl();
    let caps = gpt4o_session(
        vec![
            SessionSink::Header("x-session-affinity".into()),
            SessionSink::Field("prompt_cache_key".into()),
        ],
        Tri::Unknown,
    );
    let (enc, _) = x
        .translate_request(
            Protocol::OaiChat,
            chat_body(),
            &hdrs(&[("x-session-id", "s")]),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert!(enc.headers.contains_key("x-session-affinity"), "header sink used");
    assert!(body.get("prompt_cache_key").is_none(), "second sink not used: {body}");
}

/// A session id present with no backend sink is dropped with a degradation (and the request
/// still succeeds — affinity is best-effort). The id here is header-only, so it is not shadowed
/// by the prompt-cache path's own reporting.
#[test]
fn no_sink_drops_and_degrades() {
    let x = xl();
    let caps = preset::gpt4o(); // no session sink declared
    let (enc, _) = x
        .translate_request(
            Protocol::OaiChat,
            chat_body(),
            &hdrs(&[("x-session-id", "sess-1")]),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .unwrap();
    let drop = enc
        .degradations
        .iter()
        .find(|d| d.field == "session_id")
        .expect("a session_id degradation");
    assert_eq!(drop.kind, DegradationKind::Dropped);
}

/// Differing session ids in two slots produce a single conflict degradation naming the loser.
#[test]
fn conflicting_session_ids_degrade() {
    let x = xl();
    // prompt_cache_key = "a" (winner) and x-session-id = "b" (loser). gpt-4o accepts
    // prompt_cache_key as a cache key, so the winning value is still carried; only the
    // conflict is reported.
    let caps = preset::gpt4o();
    let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"prompt_cache_key":"a"}"#;
    let (enc, _) = x
        .translate_request(
            Protocol::OaiChat,
            body,
            &hdrs(&[("x-session-id", "b")]),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .unwrap();

    let conflict = enc
        .degradations
        .iter()
        .find(|d| d.field == "session_id")
        .expect("a session_id conflict degradation");
    assert_eq!(conflict.kind, DegradationKind::Dropped);
    assert!(conflict.detail.contains("x-session-id"), "names the loser: {}", conflict.detail);
}

/// A backend that *requires* a session id rejects a request that supplies none.
#[test]
fn required_session_id_missing_is_rejected() {
    let x = xl();
    let caps = gpt4o_session(vec![SessionSink::Field("prompt_cache_key".into())], Tri::Yes);
    let err = x
        .translate_request(
            Protocol::OaiChat,
            chat_body(),
            &HeaderMap::new(),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .expect_err("must reject a missing required session id");
    assert_eq!(err.kind, ErrorKind::InvalidRequest);
}

/// The same required backend accepts a request that supplies a session id.
#[test]
fn required_session_id_present_is_ok() {
    let x = xl();
    let caps = gpt4o_session(vec![SessionSink::Field("prompt_cache_key".into())], Tri::Yes);
    let (enc, _) = x
        .translate_request(
            Protocol::OaiChat,
            chat_body(),
            &hdrs(&[("x-session-affinity", "ok")]),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .expect("present session id satisfies the requirement");
    let body: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(body["prompt_cache_key"], "ok");
}

/// Cross-protocol: an Anthropic client's `x-opencode-session` header maps onto an OpenAI
/// Responses backend's `prompt_cache_key` field.
#[test]
fn cross_protocol_header_to_field() {
    let x = xl();
    let mut caps = preset::gpt5_responses();
    caps.session = SessionCap {
        accepts: Some(vec![SessionSink::Field("prompt_cache_key".into())]),
        required: Tri::Unknown,
    };
    let body = br#"{"model":"claude-opus-5","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
    let (enc, _) = x
        .translate_request(
            Protocol::Anthropic,
            body,
            &hdrs(&[("x-opencode-session", "oc-7")]),
            Protocol::OaiResponses,
            &caps,
            &Resolutions::new(),
        )
        .unwrap();
    let out: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(out["prompt_cache_key"], "oc-7", "cross-protocol affinity: {out}");
}
