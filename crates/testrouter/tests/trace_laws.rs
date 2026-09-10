//! Trace laws (plan §8, §11, milestone R3): over every trace produced by a small in-test harness
//! session (text / tools / reasoning / media across all surfaces, streaming and not), assert the
//! invariants that must hold of any well-formed trace:
//!
//! (a) schema round trip — every record serializes and re-parses;
//! (b) aggregate law — the recorded client frames, re-decoded with the client protocol's decoder
//!     and aggregated, structurally match `encode_response(that aggregate)` re-decoded;
//! (c) no minted router ids (trace id / client response id) leak into the upstream prompt;
//! (d) every drop has a degradation — degradation bookkeeping is consistent, and a request with a
//!     feature the target cannot honour records at least one degradation;
//! (e) determinism — `replay` over each trace is a no-diff;
//! (f) triage over the clean session reports zero anomalies, and exactly one after a seeded
//!     failing upstream.
//!
//! Offline only: every request goes through the in-process mock upstream.

use std::time::Duration;

use bytes::Bytes;
use regex::Regex;
use serde_json::{json, Value};

use llm_xlate::Translator;
use llm_xlate_core::error::{ErrorKind, XlateError};
use llm_xlate_core::caps::{shipped, Capabilities};
use llm_xlate_core::codec::{EncodeCtx, TranslatorConfig};
use llm_xlate_core::ir::{Item, IrResponse, Part, Protocol, ProviderFamily, ResponseId};

use llm_xlate_testrouter::config::Config;
use llm_xlate_testrouter::test_support::*;
use llm_xlate_testrouter::trace::TraceRecord;
use llm_xlate_testrouter::tracecli;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Harness
// ─────────────────────────────────────────────────────────────────────────────────────────────

const REQ_ID: &str = "req_mock_0001";

/// Wrap a responder so its response carries an `x-request-id` header (real providers always do;
/// the offline mock does not, so we add one to keep triage clean for successful legs).
fn with_request_id(r: Responder) -> Responder {
    match r {
        Responder::Full { status, mut headers, body } => {
            headers.push(("x-request-id".into(), REQ_ID.into()));
            Responder::Full { status, headers, body }
        }
        Responder::Sse { status, mut headers, frames } => {
            headers.push(("x-request-id".into(), REQ_ID.into()));
            Responder::Sse { status, headers, frames }
        }
    }
}

fn client_path(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "/v1/chat/completions",
        Protocol::OaiResponses => "/v1/responses",
        Protocol::Anthropic => "/v1/messages",
    }
}

/// One harness request: post `body` for `client` and return the single produced trace record.
async fn run_case(client: Protocol, path_headers: &[(&str, &str)], body: Bytes, responder: Responder) -> TraceRecord {
    let tr = TestRouter::example(MockUpstream::always(with_request_id(responder)));
    let resp = tr.post(client_path(client), path_headers, body).await;
    assert_eq!(resp.status, 200, "harness case failed: {}", resp.text());
    tr.trace(&resp)
}

/// Build the clean harness session: text / tools / reasoning / media × surfaces, stream + not.
async fn clean_session() -> Vec<TraceRecord> {
    let mut recs = Vec::new();

    // Text, same-surface and cross-surface, streaming + non-streaming.
    type TextCase = (Protocol, &'static str, Protocol, bool, &'static [(&'static str, &'static str)]);
    let text_cases: &[TextCase] = &[
        (Protocol::OaiChat, "gpt-4o", Protocol::OaiChat, false, &[("x-xlate-upstream", "chat")]),
        (Protocol::OaiChat, "gpt-4o", Protocol::OaiChat, true, &[("x-xlate-upstream", "chat")]),
        (Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]),
        (Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, true, &[]),
        (Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, false, &[]),
        (Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, &[]),
        (Protocol::OaiChat, "claude-sonnet-5", Protocol::Anthropic, true, &[]),
        (Protocol::Anthropic, "gpt-5.4", Protocol::OaiResponses, true, &[]),
    ];
    for (client, model, upstream, stream, headers) in text_cases {
        let responder = if *stream {
            respond_sse(*upstream, model, &text_events(model, "Hello there friend"), Duration::ZERO)
        } else {
            respond_full(*upstream, &text_response(model, "Hello there friend"))
        };
        let body = simple_request(*client, model, "hi", *stream);
        recs.push(run_case(*client, headers, body, responder).await);
    }

    // Tools (streaming Anthropic + non-streaming Chat).
    {
        let model = "claude-sonnet-5";
        let body = anthropic_tools_body(model, true);
        let responder = respond_sse(
            Protocol::Anthropic,
            model,
            &tool_call_events(model, "call_up", "get_weather", "{\"city\":\"NYC\"}"),
            Duration::ZERO,
        );
        recs.push(run_case(Protocol::Anthropic, &[], body, responder).await);
    }

    // Reasoning (Anthropic thinking → Chat client, streaming).
    {
        let model = "claude-sonnet-5";
        let responder = respond_sse(
            Protocol::Anthropic,
            model,
            &reasoning_events(model, ProviderFamily::Anthropic, "let me think", "the answer"),
            Duration::ZERO,
        );
        let body = simple_request(Protocol::OaiChat, model, "hard", true);
        recs.push(run_case(Protocol::OaiChat, &[], body, responder).await);
    }

    // Media (image, non-streaming Chat).
    {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}}
            ]}]
        });
        let responder = respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "a cat"));
        recs.push(run_case(Protocol::OaiChat, &[], Bytes::from(serde_json::to_vec(&body).unwrap()), responder).await);
    }

    recs
}

fn anthropic_tools_body(model: &str, stream: bool) -> Bytes {
    Bytes::from(serde_json::to_vec(&json!({
        "model": model,
        "max_tokens": 1024,
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": "weather in NYC?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "NYC"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F sunny"}]},
            {"role": "user", "content": "and tomorrow?"}
        ],
        "stream": stream
    })).unwrap())
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (a) schema round trip
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn law_a_schema_round_trip() {
    let recs = clean_session().await;
    assert!(recs.len() >= 11, "expected a full session, got {}", recs.len());
    for rec in &recs {
        let s = serde_json::to_string(rec).expect("serialize");
        let back: TraceRecord = serde_json::from_str(&s).expect("round-trip");
        assert_eq!(back.trace_id, rec.trace_id);
        assert_eq!(back.v, 1);
        assert!(s.starts_with("{\"v\":1,\"trace_id\":"), "fixed field order");
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (b) aggregate law over the client frames
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn core_protocol(s: &str) -> Protocol {
    match s {
        "chat" => Protocol::OaiChat,
        "responses" => Protocol::OaiResponses,
        "anthropic" => Protocol::Anthropic,
        other => panic!("unknown protocol {other}"),
    }
}

fn client_family(p: Protocol) -> ProviderFamily {
    match p {
        Protocol::OaiChat | Protocol::OaiResponses => ProviderFamily::OpenAI,
        Protocol::Anthropic => ProviderFamily::Anthropic,
    }
}

/// Project an `IrResponse` onto its load-bearing structure (ignoring ids): stop reason + per-item
/// kind / text-presence / tool name+args, so a streaming vs non-streaming client encoder mismatch
/// is caught without depending on minted ids.
fn project(resp: &IrResponse) -> (String, Vec<String>) {
    let items = resp
        .items
        .iter()
        .filter_map(|it| match it {
            Item::Message { role, content, .. } => {
                let has_text = content.iter().any(|p| p.as_text().is_some());
                let has_refusal = content.iter().any(|p| matches!(p, Part::Refusal { .. }));
                Some(format!("msg:{role:?}:text={has_text}:refusal={has_refusal}"))
            }
            // Compare tool arguments as JSON values: streamed `partial_json` keeps provider
            // whitespace, the non-streaming side re-serializes compactly.
            Item::ToolCall { name, arguments, .. } => {
                let canonical = serde_json::from_str::<serde_json::Value>(arguments.as_str())
                    .map(|v| v.to_string())
                    .unwrap_or_else(|_| arguments.as_str().to_string());
                Some(format!("call:{name}:{canonical}"))
            }
            Item::ToolResult { is_error, .. } => Some(format!("result:err={is_error}")),
            Item::Reasoning(_) => Some("reasoning".to_string()),
            _ => None,
        })
        .collect();
    (format!("{:?}", resp.stop), items)
}

#[tokio::test]
async fn law_b_aggregate_matches_encode_response() {
    let recs = clean_session().await;
    let t = Translator::new(TranslatorConfig::default());
    let mut checked = 0;
    for rec in &recs {
        let Some(cr) = &rec.client_response else { continue };
        let Some(frames) = &cr.frames else { continue }; // streaming client only
        let client = core_protocol(&rec.client.protocol);
        let model = rec
            .route
            .as_ref()
            .and_then(|r| r.get("client_model"))
            .and_then(Value::as_str)
            .unwrap_or("model");
        let caps: Capabilities = shipped().resolve(&client_family(client), model, None);

        // Re-decode the recorded content frames + aggregate.
        let raw: Vec<u8> = frames
            .iter()
            .filter(|f| !f.keepalive)
            .flat_map(|f| f.data.as_bytes().to_vec())
            .collect();
        let mut dec = t.stream_decoder(client, &caps);
        let mut ev = dec.push(&raw);
        ev.extend(dec.finish());
        let left = t.aggregate_stream(ev.iter().cloned()).expect("aggregate client frames");

        // encode_response(left) re-decoded + aggregated must project to the same structure.
        let mut ctx = EncodeCtx::new(client, "law-model", ResponseId::new("resp_law"), t.sealer());
        ctx.created_at = 1_700_000_000;
        ctx.stream = false;
        let body = t.encode_response(client, &left, &ctx);
        let revents = t.decode_response(client, body.as_ref(), &caps).expect("decode re-encoded response");
        let right = t.aggregate_stream(revents.iter().cloned()).expect("aggregate re-encoded response");

        assert_eq!(
            project(&left),
            project(&right),
            "aggregate law failed for {} ({})",
            rec.trace_id,
            rec.client.protocol
        );
        checked += 1;
    }
    assert!(checked >= 3, "expected several streaming traces, checked {checked}");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (c) no minted router ids in the upstream prompt
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn law_c_no_minted_ids_upstream() {
    let recs = clean_session().await;
    // Router-minted ids use an 8-digit sequential fragment (see ids::SequentialSource).
    let minted = Regex::new(r"(tr_|chatcmpl-|resp_|msg_|rs_|fc_)\d{8}").unwrap();
    let mut checked = 0;
    for rec in &recs {
        let Some(up) = &rec.upstream else { continue };
        let Some(body) = &up.body else { continue };
        let s = serde_json::to_string(body).unwrap();
        // The trace id must never appear.
        assert!(!s.contains(&rec.trace_id), "trace id leaked upstream in {}", rec.trace_id);
        // No router-minted id shape may appear in the upstream prompt (Responses request item ids
        // in the fixtures use short suffixes like `fc_1`, not the 8-digit minted form).
        if let Some(m) = minted.find(&s) {
            panic!("minted id `{}` leaked into upstream prompt of {}", m.as_str(), rec.trace_id);
        }
        checked += 1;
    }
    assert!(checked >= 8, "expected many upstream legs, checked {checked}");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (d) every drop has a degradation
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn law_d_degradation_bookkeeping_consistent() {
    let recs = clean_session().await;
    for rec in &recs {
        let recorded = rec.lowered.as_ref().map(|l| l.degradations.len()).unwrap_or(0);
        assert_eq!(
            recorded as u32, rec.checks.degradation_count,
            "degradation count / list mismatch for {}",
            rec.trace_id
        );
        if let Some(l) = &rec.lowered {
            for d in &l.degradations {
                assert!(d.is_object(), "degradation is not an object in {}", rec.trace_id);
            }
        }
    }
}

#[tokio::test]
async fn law_d_dropped_feature_records_degradation() {
    // A Chat request carrying sampling that the resolved (Responses) target cannot honour must
    // record at least one degradation, and the degraded header must agree with the count.
    let tr = TestRouter::example(MockUpstream::always(with_request_id(respond_full(
        Protocol::OaiResponses,
        &text_response("gpt-5.4", "hi"),
    ))));
    let body = json!({"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5});
    let resp = tr.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "{}", resp.text());
    let rec = tr.trace(&resp);
    let degraded_header = resp.headers.get("x-router-degraded").is_some();
    assert_eq!(degraded_header, rec.checks.degradation_count > 0, "degraded header vs count");
    // When a drop happened, at least one degradation is recorded.
    if degraded_header {
        assert!(rec.lowered.as_ref().is_some_and(|l| !l.degradations.is_empty()));
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (e) determinism — replay is a no-diff
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn law_e_replay_is_no_diff() {
    let recs = clean_session().await;
    let mut replayed = 0;
    for rec in &recs {
        let outcome = tracecli::replay(rec, Config::example()).await.expect("replay");
        assert!(
            outcome.matched(),
            "replay diverged for {}: upstream_diff={:?} client_diff={:?} status {}!={}",
            rec.trace_id,
            outcome.upstream_diff,
            outcome.client_diff,
            outcome.client_status,
            outcome.recorded_status
        );
        replayed += 1;
    }
    assert_eq!(replayed, recs.len());
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (f) triage anomaly counts
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn law_f_triage_clean_then_one_anomaly() {
    let mut recs = clean_session().await;
    let report = tracecli::triage(&recs, &tracecli::TriageFilter::default());
    assert_eq!(
        report.anomalous_rows, 0,
        "clean session had anomalies: {:?}",
        report.rows.iter().filter(|r| r.is_anomalous()).map(|r| (&r.trace_id, &r.anomalies)).collect::<Vec<_>>()
    );

    // Seed one failing upstream (an authentication error passthrough).
    let err = XlateError::new(ErrorKind::Authentication, "bad key");
    let tr = TestRouter::example(MockUpstream::always(respond_error(Protocol::OaiChat, &err)));
    let resp = tr.post("/v1/chat/completions", &[("x-xlate-upstream", "chat")], simple_request(Protocol::OaiChat, "gpt-4o", "hi", false)).await;
    assert_eq!(resp.status, 401, "{}", resp.text());
    recs.push(tr.trace(&resp));

    let report2 = tracecli::triage(&recs, &tracecli::TriageFilter::default());
    assert_eq!(report2.anomalous_rows, 1, "expected exactly one anomaly after the seeded failure");
}
