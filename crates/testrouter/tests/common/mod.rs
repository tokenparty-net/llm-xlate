//! Shared helpers for the `cli_*` integration tests (a `tests/common/` module is compiled into
//! each test binary that declares `mod common;`, not as a standalone test target).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use bytes::Bytes;
use serde_json::json;

use llm_xlate_core::ir::{Protocol, ProviderFamily};

use llm_xlate_testrouter::test_support::*;
use llm_xlate_testrouter::trace::TraceRecord;

/// A stable mock upstream request id, so successful legs are not flagged as anomalies.
pub const REQ_ID: &str = "req_mock_0001";

/// Add an `x-request-id` header to a responder (real providers always send one).
pub fn with_request_id(r: Responder) -> Responder {
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

/// The client POST path for a protocol.
pub fn client_path(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "/v1/chat/completions",
        Protocol::OaiResponses => "/v1/responses",
        Protocol::Anthropic => "/v1/messages",
    }
}

/// Produce one successful text trace for `(client, model, upstream, stream)`.
pub async fn text_trace(
    client: Protocol,
    model: &str,
    upstream: Protocol,
    stream: bool,
    headers: &[(&str, &str)],
) -> TraceRecord {
    let responder = if stream {
        respond_sse(upstream, model, &text_events(model, "Hello there friend"), Duration::ZERO)
    } else {
        respond_full(upstream, &text_response(model, "Hello there friend"))
    };
    let tr = TestRouter::example(MockUpstream::always(with_request_id(responder)));
    let body = simple_request(client, model, "hi", stream);
    let resp = tr.post(client_path(client), headers, body).await;
    assert_eq!(resp.status, 200, "text_trace failed: {}", resp.text());
    tr.trace(&resp)
}

/// Produce one reasoning trace (Anthropic thinking → Chat client, streaming).
pub async fn reasoning_trace() -> TraceRecord {
    let model = "claude-sonnet-5";
    let responder = respond_sse(
        Protocol::Anthropic,
        model,
        &reasoning_events(model, ProviderFamily::Anthropic, "let me think", "the answer"),
        Duration::ZERO,
    );
    let tr = TestRouter::example(MockUpstream::always(with_request_id(responder)));
    let body = simple_request(Protocol::OaiChat, model, "hard", true);
    let resp = tr.post("/v1/chat/completions", &[], body).await;
    assert_eq!(resp.status, 200, "{}", resp.text());
    tr.trace(&resp)
}

/// Produce one failing-upstream trace (authentication passthrough).
pub async fn failing_trace() -> TraceRecord {
    use llm_xlate_core::error::{ErrorKind, XlateError};
    let err = XlateError::new(ErrorKind::Authentication, "bad key");
    let tr = TestRouter::example(MockUpstream::always(respond_error(Protocol::OaiChat, &err)));
    let resp = tr
        .post(
            "/v1/chat/completions",
            &[("x-xlate-upstream", "chat")],
            simple_request(Protocol::OaiChat, "gpt-4o", "hi", false),
        )
        .await;
    assert_eq!(resp.status, 401, "{}", resp.text());
    tr.trace(&resp)
}

/// A unique temp directory for a test.
pub fn tmp_dir(tag: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("xlate-testrouter-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Write records into `<data_dir>/traces/session.jsonl` (no index — `find_by_id` scans) and return
/// the traces dir.
pub fn write_session(data_dir: &Path, records: &[TraceRecord]) -> PathBuf {
    let traces = data_dir.join("traces");
    std::fs::create_dir_all(&traces).unwrap();
    let mut out = String::new();
    for r in records {
        out.push_str(&serde_json::to_string(r).unwrap());
        out.push('\n');
    }
    std::fs::write(traces.join("session.jsonl"), out).unwrap();
    traces
}

/// A minimal Chat tools request body (with a prior tool turn).
pub fn chat_tools_body(model: &str, stream: bool) -> Bytes {
    Bytes::from(
        serde_json::to_vec(&json!({
            "model": model,
            "messages": [
                {"role": "user", "content": "weather in NYC?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "72F sunny"},
                {"role": "user", "content": "and tomorrow?"}
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "w",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            "stream": stream
        }))
        .unwrap(),
    )
}
