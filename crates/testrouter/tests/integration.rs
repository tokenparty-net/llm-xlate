//! Hermetic integration tests (plan §8). Every request goes through the mock upstream — no
//! network, no secrets. Each success test asserts the client bytes, a present `x-xlate-trace-id`,
//! a schema-valid trace record, and that the inline aggregate check holds.

use std::time::Duration;

use bytes::Bytes;
use serde_json::{json, Value};

use llm_xlate_core::error::{ErrorKind, XlateError};
use llm_xlate_core::ir::{Protocol, ProviderFamily};

use llm_xlate_testrouter::config::Config;
use llm_xlate_testrouter::test_support::*;
use llm_xlate_testrouter::trace::TraceRecord;

// ── helpers ────────────────────────────────────────────────────────────────────────────────

fn client_path(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "/v1/chat/completions",
        Protocol::OaiResponses => "/v1/responses",
        Protocol::Anthropic => "/v1/messages",
    }
}

/// Assert the response carries a trace id, its record is schema-valid, and the aggregate check
/// holds. Returns the record.
fn assert_trace_ok(tr: &TestRouter, resp: &TestResponse) -> TraceRecord {
    assert!(!resp.trace_id().is_empty(), "missing x-xlate-trace-id header");
    let rec = tr.trace(resp);
    // Schema-valid: serializes and round-trips.
    let s = serde_json::to_string(&rec).expect("trace serializes");
    let back: TraceRecord = serde_json::from_str(&s).expect("trace round-trips");
    assert_eq!(back.trace_id, rec.trace_id);
    assert_eq!(back.v, 1);
    assert!(
        rec.checks.aggregate_matches_encode_response,
        "aggregate check failed for {}",
        rec.trace_id
    );
    // Fixed field order: `v` first.
    assert!(s.starts_with("{\"v\":1,\"trace_id\":"), "fixed field order");
    rec
}

async fn text_case(client: Protocol, model: &str, upstream: Protocol, stream: bool, headers: &[(&str, &str)]) {
    let responder = if stream {
        respond_sse(upstream, model, &text_events(model, "Hello there"), Duration::ZERO)
    } else {
        respond_full(upstream, &text_response(model, "Hello there"))
    };
    let tr = TestRouter::example(MockUpstream::always(responder));
    let body = simple_request(client, model, "Hi there", stream);
    let resp = tr.post(client_path(client), headers, body).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    assert_eq!(rec.client.protocol, protocol_token(client));
    if stream {
        assert!(resp.text().contains("data:"), "streamed body: {}", resp.text());
    } else {
        assert!(resp.json().is_object());
    }
    // The trace records the resolved upstream protocol.
    let up = rec.route.as_ref().unwrap().get("upstream_protocol").and_then(Value::as_str).unwrap();
    assert_eq!(up, protocol_token(upstream));
}

fn protocol_token(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "chat",
        Protocol::OaiResponses => "responses",
        Protocol::Anthropic => "anthropic",
    }
}

const CHAT_UP: &[(&str, &str)] = &[("x-xlate-upstream", "chat")];

// ── Group A: same-surface text ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn a1_chat_text_nonstream() {
    text_case(Protocol::OaiChat, "gpt-4o", Protocol::OaiChat, false, CHAT_UP).await;
}
#[tokio::test]
async fn a2_chat_text_stream() {
    text_case(Protocol::OaiChat, "gpt-4o", Protocol::OaiChat, true, CHAT_UP).await;
}
#[tokio::test]
async fn a3_responses_text_nonstream() {
    text_case(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
}
#[tokio::test]
async fn a4_responses_text_stream() {
    text_case(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, true, &[]).await;
}
#[tokio::test]
async fn a5_anthropic_text_nonstream() {
    text_case(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, false, &[]).await;
}
#[tokio::test]
async fn a6_anthropic_text_stream() {
    text_case(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
}

// ── Group B: cross-surface text ────────────────────────────────────────────────────────────

#[tokio::test]
async fn b1_chat_client_responses_upstream_nonstream() {
    text_case(Protocol::OaiChat, "gpt-5.4", Protocol::OaiResponses, false, &[]).await;
}
#[tokio::test]
async fn b2_chat_client_responses_upstream_stream() {
    text_case(Protocol::OaiChat, "gpt-5.4", Protocol::OaiResponses, true, &[]).await;
}
#[tokio::test]
async fn b3_chat_client_anthropic_upstream_nonstream() {
    text_case(Protocol::OaiChat, "claude-sonnet-5", Protocol::Anthropic, false, &[]).await;
}
#[tokio::test]
async fn b4_chat_client_anthropic_upstream_stream() {
    text_case(Protocol::OaiChat, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
}
#[tokio::test]
async fn b5_anthropic_client_responses_upstream_nonstream() {
    text_case(Protocol::Anthropic, "gpt-5.4", Protocol::OaiResponses, false, &[]).await;
}
#[tokio::test]
async fn b6_anthropic_client_responses_upstream_stream() {
    text_case(Protocol::Anthropic, "gpt-5.4", Protocol::OaiResponses, true, &[]).await;
}
#[tokio::test]
async fn b7_responses_client_anthropic_upstream_nonstream() {
    text_case(Protocol::OaiResponses, "claude-sonnet-5", Protocol::Anthropic, false, &[]).await;
}
#[tokio::test]
async fn b8_responses_client_anthropic_upstream_stream() {
    text_case(Protocol::OaiResponses, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
}

// ── Group C: tools (with a tool-result turn) ───────────────────────────────────────────────

fn chat_tools_body(model: &str, stream: bool) -> Bytes {
    Bytes::from(serde_json::to_vec(&json!({
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
    })).unwrap())
}

fn responses_tools_body(model: &str, stream: bool) -> Bytes {
    Bytes::from(serde_json::to_vec(&json!({
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "weather in NYC?"}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}", "id": "fc_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": "72F sunny"},
            {"role": "user", "content": [{"type": "input_text", "text": "and tomorrow?"}]}
        ],
        "tools": [{"type": "function", "name": "get_weather", "description": "w",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "stream": stream
    })).unwrap())
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

async fn tools_case(client: Protocol, model: &str, upstream: Protocol, stream: bool, body: Bytes) {
    let responder = if stream {
        respond_sse(upstream, model, &tool_call_events(model, "call_up", "get_weather", "{\"city\":\"NYC\"}"), Duration::ZERO)
    } else {
        respond_full(upstream, &tool_call_response(model, "call_up", "get_weather", "{\"city\":\"NYC\"}"))
    };
    let tr = TestRouter::example(MockUpstream::always(responder));
    let resp = tr.post(client_path(client), &[], body).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}

#[tokio::test]
async fn c1_chat_tools_nonstream() {
    tools_case(Protocol::OaiChat, "gpt-4o", Protocol::OaiResponses, false, chat_tools_body("gpt-4o", false)).await;
}
#[tokio::test]
async fn c2_chat_tools_stream() {
    tools_case(Protocol::OaiChat, "gpt-4o", Protocol::OaiResponses, true, chat_tools_body("gpt-4o", true)).await;
}
#[tokio::test]
async fn c3_responses_tools_nonstream() {
    tools_case(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, responses_tools_body("gpt-4o", false)).await;
}
#[tokio::test]
async fn c4_responses_tools_stream() {
    tools_case(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, true, responses_tools_body("gpt-4o", true)).await;
}
#[tokio::test]
async fn c5_anthropic_tools_nonstream() {
    tools_case(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, false, anthropic_tools_body("claude-sonnet-5", false)).await;
}
#[tokio::test]
async fn c6_anthropic_tools_stream() {
    tools_case(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, anthropic_tools_body("claude-sonnet-5", true)).await;
}

// ── Group D: reasoning ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn d1_anthropic_thinking_to_chat_client_nonstream() {
    let model = "claude-sonnet-5";
    let responder = respond_full(Protocol::Anthropic, &reasoning_response(model, ProviderFamily::Anthropic, "let me think", "the answer"));
    let tr = TestRouter::example(MockUpstream::always(responder));
    let body = simple_request(Protocol::OaiChat, model, "hard question", false);
    let resp = tr.post("/v1/chat/completions", &[], body).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}
#[tokio::test]
async fn d2_anthropic_thinking_to_chat_client_stream() {
    let model = "claude-sonnet-5";
    let responder = respond_sse(Protocol::Anthropic, model, &reasoning_events(model, ProviderFamily::Anthropic, "let me think", "the answer"), Duration::ZERO);
    let tr = TestRouter::example(MockUpstream::always(responder));
    let body = simple_request(Protocol::OaiChat, model, "hard question", true);
    let resp = tr.post("/v1/chat/completions", &[], body).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}
#[tokio::test]
async fn d3_responses_reasoning_to_anthropic_client_nonstream() {
    let model = "gpt-5.4";
    let responder = respond_full(Protocol::OaiResponses, &reasoning_response(model, ProviderFamily::OpenAI, "thinking", "answer"));
    let tr = TestRouter::example(MockUpstream::always(responder));
    let body = Bytes::from(serde_json::to_vec(&json!({
        "model": model, "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [{"role": "user", "content": "hard question"}]
    })).unwrap());
    let resp = tr.post("/v1/messages", &[], body).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}
#[tokio::test]
async fn d4_responses_reasoning_to_anthropic_client_stream() {
    let model = "gpt-5.4";
    let responder = respond_sse(Protocol::OaiResponses, model, &reasoning_events(model, ProviderFamily::OpenAI, "thinking", "answer"), Duration::ZERO);
    let tr = TestRouter::example(MockUpstream::always(responder));
    let body = Bytes::from(serde_json::to_vec(&json!({
        "model": model, "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [{"role": "user", "content": "hard question"}], "stream": true
    })).unwrap());
    let resp = tr.post("/v1/messages", &[], body).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}

// ── Group E: image input ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn e1_chat_image_nonstream() {
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "compare"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}}
        ]}]
    });
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "two cats"))));
    let resp = tr.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}
#[tokio::test]
async fn e2_responses_image_nonstream() {
    let body = json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
        ]}]
    });
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "a cat"))));
    let resp = tr.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}
#[tokio::test]
async fn e3_anthropic_image_nonstream() {
    let body = json!({
        "model": "claude-sonnet-5", "max_tokens": 1024,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRg=="}}
        ]}]
    });
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::Anthropic, &text_response("claude-sonnet-5", "a cat"))));
    let resp = tr.post("/v1/messages", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}

// ── Group F: structured output ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn f1_chat_structured_nonstream() {
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "extract"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "person", "strict": true,
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": false}
        }}
    });
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "{\"name\":\"Ada\"}"))));
    let resp = tr.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}
#[tokio::test]
async fn f2_responses_structured_nonstream() {
    let body = json!({
        "model": "gpt-4o",
        "input": "extract",
        "text": {"format": {"type": "json_schema", "name": "person", "strict": true,
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": false}}}
    });
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "{\"name\":\"Ada\"}"))));
    let resp = tr.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}

// ── Group G: mid-context system ────────────────────────────────────────────────────────────

#[tokio::test]
async fn g1_chat_midcontext_system_nonstream() {
    let body = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "again"}
        ]
    });
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "ok"))));
    let resp = tr.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}
#[tokio::test]
async fn g2_responses_midcontext_system_nonstream() {
    let body = json!({
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"role": "system", "content": [{"type": "input_text", "text": "be terse"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "again"}]}
        ]
    });
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "ok"))));
    let resp = tr.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}

// ── Group H: error passthrough ─────────────────────────────────────────────────────────────

async fn error_case(client: Protocol, model: &str, headers: &[(&str, &str)], upstream: Protocol, kind: ErrorKind, expect_status: u16) {
    let err = XlateError::new(kind, "boom");
    let tr = TestRouter::example(MockUpstream::always(respond_error(upstream, &err)));
    let body = simple_request(client, model, "hi", false);
    let resp = tr.post(client_path(client), headers, body).await;
    assert_eq!(resp.status, expect_status, "kind={kind:?} body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    assert!(!rec.errors.is_empty(), "error not recorded in trace");
    assert!(resp.json().get("error").is_some() || resp.json().get("type").is_some(), "dialect error body: {}", resp.text());
}

#[tokio::test]
async fn h1_chat_invalid_request() {
    error_case(Protocol::OaiChat, "gpt-4o", CHAT_UP, Protocol::OaiChat, ErrorKind::InvalidRequest, 400).await;
}
#[tokio::test]
async fn h2_chat_authentication() {
    error_case(Protocol::OaiChat, "gpt-4o", CHAT_UP, Protocol::OaiChat, ErrorKind::Authentication, 401).await;
}
#[tokio::test]
async fn h3_chat_rate_limited() {
    error_case(Protocol::OaiChat, "gpt-4o", CHAT_UP, Protocol::OaiChat, ErrorKind::RateLimited, 429).await;
}
#[tokio::test]
async fn h4_responses_context_length() {
    error_case(Protocol::OaiResponses, "gpt-4o", &[], Protocol::OaiResponses, ErrorKind::ContextLengthExceeded, 400).await;
}
#[tokio::test]
async fn h5_anthropic_overloaded_529() {
    error_case(Protocol::Anthropic, "claude-sonnet-5", &[], Protocol::Anthropic, ErrorKind::Overloaded, 529).await;
}
#[tokio::test]
async fn h6_error_stream_client() {
    // A streaming client whose upstream returns an error before any frame.
    let err = XlateError::new(ErrorKind::RateLimited, "slow down");
    let tr = TestRouter::example(MockUpstream::always(respond_error(Protocol::OaiResponses, &err)));
    let body = simple_request(Protocol::OaiChat, "gpt-5.4", "hi", true);
    let resp = tr.post("/v1/chat/completions", &[], body).await;
    assert_eq!(resp.status, 429, "body={}", resp.text());
    // A pre-stream upstream error must be a plain JSON body under the error status — never an SSE
    // `data:{error}` + `[DONE]` shape (that is only valid mid-stream, after 200 is committed).
    let ct = resp.headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    assert_eq!(ct, "application/json", "pre-stream error must be JSON, not SSE; body={}", resp.text());
    assert!(resp.json().get("error").is_some(), "expected a JSON error body: {}", resp.text());
    assert_trace_ok(&tr, &resp);
}

// ── Group I: cross-cutting ─────────────────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn i1_keepalive_on_stalled_stream() {
    let model = "claude-sonnet-5";
    let mut cfg = Config::example();
    cfg.server.keepalive_secs = 1;
    let responder = respond_sse(Protocol::Anthropic, model, &text_events(model, "eventually"), Duration::from_secs(3));
    let tr = TestRouter::build(cfg, MockUpstream::always(responder));
    let body = simple_request(Protocol::Anthropic, model, "hi", true);
    let resp = tr.post("/v1/messages", &[], body).await;
    assert_eq!(resp.status, 200);
    let rec = assert_trace_ok(&tr, &resp);
    let frames = rec.client_response.as_ref().unwrap().frames.as_ref().unwrap();
    assert!(frames.iter().any(|f| f.keepalive), "expected at least one keepalive frame");
}

#[tokio::test]
async fn i2_client_disconnect_cancels() {
    use http_body_util::BodyExt;
    let model = "claude-sonnet-5";
    let responder = respond_sse(Protocol::Anthropic, model, &text_events(model, "hello world here"), Duration::ZERO);
    let tr = TestRouter::example(MockUpstream::always(responder));
    let body = simple_request(Protocol::Anthropic, model, "hi", true);
    let resp = tr.post_raw("/v1/messages", &[], body).await;
    let mut body = resp.into_body();
    // Pull a single frame, then drop the body to simulate a client disconnect.
    let _ = body.frame().await;
    drop(body);
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
    let recs = tr.sink.all();
    assert!(
        recs.iter().any(|r| r.errors.iter().any(|e| e.stage == "cancelled")),
        "expected a cancelled trace, got {} records",
        recs.len()
    );
}

#[tokio::test]
async fn i3_oversize_body_413() {
    let mut cfg = Config::example();
    cfg.server.max_body_bytes = 32;
    let tr = TestRouter::build(cfg, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi"))));
    let big = simple_request(Protocol::OaiChat, "gpt-4o", &"x".repeat(1000), false);
    let resp = tr.post("/v1/chat/completions", CHAT_UP, big).await;
    assert_eq!(resp.status, 413);
}

#[tokio::test]
async fn i4_token_auth_required() {
    let mut cfg = Config::example();
    cfg.server.token = "s3cret".to_string();
    let tr = TestRouter::build(cfg, MockUpstream::always(respond_full(Protocol::OaiChat, &text_response("gpt-4o", "hi"))));
    // Missing token → 401.
    let resp = tr.post("/v1/chat/completions", CHAT_UP, simple_request(Protocol::OaiChat, "gpt-4o", "hi", false)).await;
    assert_eq!(resp.status, 401);
    // Correct token → 200.
    let hdrs: &[(&str, &str)] = &[("x-xlate-upstream", "chat"), ("authorization", "Bearer s3cret")];
    let resp = tr.post("/v1/chat/completions", hdrs, simple_request(Protocol::OaiChat, "gpt-4o", "hi", false)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
}

#[tokio::test]
async fn i5_xlate_overrides_recorded() {
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiChat, &text_response("gpt-4o", "hi"))));
    let hdrs: &[(&str, &str)] = &[
        ("x-xlate-upstream", "chat"),
        ("x-xlate-model", "gpt-4o-2024"),
        ("x-xlate-tag", "sess-42"),
    ];
    let resp = tr.post("/v1/chat/completions", hdrs, simple_request(Protocol::OaiChat, "gpt-4o", "hi", false)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    assert_eq!(rec.tag.as_deref(), Some("sess-42"));
    let route = rec.route.as_ref().unwrap();
    assert_eq!(route.get("upstream_protocol").and_then(Value::as_str), Some("chat"));
    assert_eq!(route.get("upstream_model").and_then(Value::as_str), Some("gpt-4o-2024"));
}

#[tokio::test]
async fn i5b_alias_resolves_upstream_model_and_echoes_client_name() {
    // Regression (found in the first live smoke): the pipeline resolved the route but never wrote
    // the upstream model into the IR, so an alias such as `claude` was sent upstream verbatim and
    // the provider answered 404. The upstream body must carry the resolved model while the client
    // keeps seeing the name it asked for.
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::Anthropic, &text_response("claude-sonnet-5", "hi"))));
    let resp = tr.post("/v1/chat/completions", &[], simple_request(Protocol::OaiChat, "claude", "hi", false)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    let route = rec.route.as_ref().unwrap();
    assert_eq!(route.get("upstream_model").and_then(Value::as_str), Some("claude-sonnet-5"));
    let up = rec.upstream.as_ref().expect("upstream leg recorded");
    let up_model = up.body.as_ref().and_then(|b| b.get("model")).and_then(Value::as_str);
    assert_eq!(up_model, Some("claude-sonnet-5"), "alias must be resolved before encoding");
    assert_eq!(resp.json().get("model").and_then(Value::as_str), Some("claude"), "client name is echoed back");
}

#[tokio::test]
async fn i6_caps_preset_override() {
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiChat, &text_response("gpt-5.4", "hi"))));
    let hdrs: &[(&str, &str)] = &[("x-xlate-upstream", "chat"), ("x-xlate-caps", "gpt5_chat")];
    let resp = tr.post("/v1/chat/completions", hdrs, simple_request(Protocol::OaiChat, "gpt-5.4", "hi", false)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    assert_eq!(rec.route.as_ref().unwrap().get("caps_source").and_then(Value::as_str), Some("preset"));
}

#[tokio::test]
async fn i7_models_endpoint_offline() {
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiChat, &text_response("gpt-4o", "hi"))));
    let resp = tr.get("/v1/models").await;
    assert_eq!(resp.status, 200);
    let v = resp.json();
    assert_eq!(v.get("object").and_then(Value::as_str), Some("list"));
    assert!(v.get("data").and_then(Value::as_array).map(|a| !a.is_empty()).unwrap_or(false));
    // Every response carries a trace id, and `/v1/models` (a real client API surface) is recorded.
    let id = resp.trace_id();
    assert!(!id.is_empty(), "/v1/models missing x-xlate-trace-id");
    let rec = tr.sink.get(&id).expect("/v1/models trace not recorded");
    assert_eq!(rec.client.path, "/v1/models");
    // The upstream `x-request-id` id is exposed on the success path too.
    assert!(resp.headers.get("x-request-id").is_some(), "/v1/models missing x-request-id");
}

#[tokio::test]
async fn i8_healthz() {
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiChat, &text_response("gpt-4o", "hi"))));
    let resp = tr.get("/healthz").await;
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json().get("status").and_then(Value::as_str), Some("ok"));
    // A pure probe, but it still carries a trace id (the every-response invariant).
    assert!(!resp.trace_id().is_empty(), "/healthz missing x-xlate-trace-id");
}

#[tokio::test]
async fn i9_store_404_and_count_tokens_passthrough() {
    // The Responses store endpoints are implemented (R2): an unknown id is a 404 in the OpenAI
    // dialect. `count_tokens` is an untranslated proxy passthrough (plan §7): the upstream JSON is
    // returned verbatim and the whole leg is recorded (visible to `trace triage`/`show`).
    let ct_body = br#"{"input_tokens":42}"#.to_vec();
    let mock = MockUpstream::new(vec![
        Rule::path("/v1/messages/count_tokens", respond_raw(200, ct_body)),
        Rule::any(respond_full(Protocol::OaiChat, &text_response("gpt-4o", "hi"))),
    ]);
    let tr = TestRouter::example(mock);

    let resp = tr.get("/v1/responses/resp_x").await;
    assert_eq!(resp.status, 404, "unknown stored id: {}", resp.text());
    assert!(resp.json().get("error").is_some(), "dialect error body: {}", resp.text());

    let ct = tr
        .post("/v1/messages/count_tokens", &[], simple_request(Protocol::Anthropic, "claude-sonnet-5", "hi", false))
        .await;
    assert_eq!(ct.status, 200, "count_tokens passthrough: {}", ct.text());
    assert_eq!(ct.json().get("input_tokens").and_then(Value::as_u64), Some(42), "verbatim upstream body");
    assert!(!ct.trace_id().is_empty(), "count_tokens missing x-xlate-trace-id");
    let rec = tr.trace(&ct);
    assert!(rec.upstream.is_some(), "count_tokens upstream leg not recorded");
    assert_eq!(rec.client.path, "/v1/messages/count_tokens");
    assert_eq!(rec.client.protocol, "anthropic");
}

#[tokio::test]
async fn i9b_count_tokens_auth_is_anthropic_dialect() {
    // A `count_tokens` 401 must be the Anthropic error envelope (`{"type":"error",...}`), not the
    // OpenAI Responses shape the shared store-endpoint auth path uses.
    let mut cfg = Config::example();
    cfg.server.token = "s3cret".to_string();
    let tr = TestRouter::build(cfg, MockUpstream::always(respond_raw(200, br#"{"input_tokens":1}"#.to_vec())));
    let ct = tr
        .post("/v1/messages/count_tokens", &[], simple_request(Protocol::Anthropic, "claude-sonnet-5", "hi", false))
        .await;
    assert_eq!(ct.status, 401, "missing token → 401: {}", ct.text());
    let v = ct.json();
    assert_eq!(v.get("type").and_then(Value::as_str), Some("error"), "Anthropic error envelope: {}", ct.text());
    assert!(v.get("error").and_then(|e| e.get("type")).is_some(), "Anthropic error.type: {}", ct.text());
    assert!(!ct.trace_id().is_empty(), "count_tokens 401 missing x-xlate-trace-id");
}

#[tokio::test]
async fn i3b_oversize_body_413_is_dialect_and_traced() {
    // An over-limit body is rejected before the handler by the body-limit layer, but the router
    // still renders it in the client dialect with a trace id and records it.
    let mut cfg = Config::example();
    cfg.server.max_body_bytes = 32;
    let tr = TestRouter::build(cfg, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi"))));
    let big = simple_request(Protocol::OaiChat, "gpt-4o", &"x".repeat(1000), false);
    let resp = tr.post("/v1/chat/completions", CHAT_UP, big).await;
    assert_eq!(resp.status, 413);
    let ct = resp.headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    assert_eq!(ct, "application/json", "413 must be JSON, not text/plain; body={}", resp.text());
    assert!(resp.json().get("error").is_some(), "413 must carry a dialect error body: {}", resp.text());
    let id = resp.trace_id();
    assert!(!id.is_empty(), "413 missing x-xlate-trace-id");
    assert!(tr.sink.get(&id).is_some(), "413 not recorded as a trace");
}

#[tokio::test]
async fn i10_redaction_secret_headers() {
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi"))));
    let hdrs: &[(&str, &str)] = &[("authorization", "Bearer sk-shouldnotappear12345678")];
    let resp = tr.post("/v1/responses", hdrs, simple_request(Protocol::OaiResponses, "gpt-4o", "hi", false)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    let auth = rec.client.headers.get("authorization").and_then(Value::as_str).unwrap();
    assert_eq!(auth, "<redacted>");
    let s = serde_json::to_string(&rec).unwrap();
    assert!(!s.contains("sk-shouldnotappear"), "secret leaked into trace");
}

#[tokio::test]
async fn i11_degraded_header_and_count() {
    // A Chat client with sampling/features that lower to Responses may degrade; assert the header
    // and trace count agree. Use a temperature on a reasoning-only path to force at least a record.
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-5.4", "hi"))));
    let body = json!({"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5});
    let resp = tr.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    let degraded_header = resp.headers.get("x-router-degraded").is_some();
    assert_eq!(degraded_header, rec.checks.degradation_count > 0);
}

#[tokio::test]
async fn i11c_responses_hosted_tools_are_dropped_for_a_chat_upstream() {
    // Regression (Codex → `deepseek-v4-flash` on the local vLLM chat backend, 2026-09-10, trace
    // `tr_dc34ef14…`): Codex declares OpenAI **Responses** hosted tools — `{"type":"namespace"}`
    // groups and `{"type":"web_search"}` — beside plain functions. They decode to provider tools
    // of family `OpenAI`, which the Chat encoder used to pass through raw because the family
    // matched the target. Chat Completions has no carrier for a hosted tool, so vLLM rejected the
    // whole request with `400 … Input should be 'function'` at `body.tools.7.type`.
    //
    // Every entry the router now puts on a Chat wire must be a `type:"function"` tool, and each
    // dropped hosted tool must be reported as a degradation in the trace + `x-router-degraded`.
    // The tool JSON is the captured Codex request (sub-tool list trimmed to one).
    let tr = TestRouter::example(MockUpstream::always(respond_full(
        Protocol::OaiChat,
        &text_response("gpt-4o", "ok"),
    )));
    let body = json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "tools": [
            {"type": "function", "name": "exec_command", "description": "Runs a command in a PTY.",
             "strict": false, "parameters": {"type": "object", "properties": {}, "additionalProperties": false}},
            {"type": "namespace", "name": "multi_agent_v1",
             "description": "Tools for spawning and managing sub-agents.",
             "tools": [{"type": "function", "name": "close_agent", "description": "Close an agent.",
                        "strict": false,
                        "parameters": {"type": "object", "properties": {"target": {"type": "string"}},
                                       "required": ["target"], "additionalProperties": false}}]},
            {"type": "web_search", "external_web_access": false}
        ]
    });
    let resp = tr.post("/v1/responses", CHAT_UP, Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "hosted tools must not 400 the turn; body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);

    let up = rec.upstream.as_ref().expect("upstream leg");
    let tools = up
        .body
        .as_ref()
        .and_then(|b| b.get("tools"))
        .and_then(Value::as_array)
        .expect("upstream tools array");
    assert_eq!(tools.len(), 1, "only the plain function survives: {tools:?}");
    for t in tools {
        assert_eq!(t.get("type").and_then(Value::as_str), Some("function"), "bad wire tool: {t}");
        assert!(t.get("function").is_some(), "Chat tool needs a `function` object: {t}");
    }

    let degs = rec.lowered.as_ref().map(|l| l.degradations.clone()).unwrap_or_default();
    let details: Vec<String> = degs
        .iter()
        .filter(|d| d.get("field").and_then(Value::as_str) == Some("tools.provider"))
        .filter_map(|d| d.get("detail").and_then(Value::as_str).map(str::to_string))
        .collect();
    assert_eq!(details.len(), 2, "both hosted tools must be reported: {degs:?}");
    assert!(details.iter().any(|d| d.contains("namespace multi_agent_v1")), "{details:?}");
    assert!(details.iter().any(|d| d.contains("web_search")), "{details:?}");
    assert!(resp.headers.get("x-router-degraded").is_some(), "dropped tools must set the header");
}

// ── Group J: golden captures (loaders) ─────────────────────────────────────────────────────

#[tokio::test]
async fn j1_golden_responses_full() {
    let tr = TestRouter::example(MockUpstream::always(golden_resp("responses/completed.resp.json")));
    let resp = tr.post("/v1/responses", &[], simple_request(Protocol::OaiResponses, "gpt-4o", "hi", false)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert_trace_ok(&tr, &resp);
}

#[tokio::test]
async fn j2_golden_responses_stream() {
    let tr = TestRouter::example(MockUpstream::always(golden_sse("responses/message.stream.sse")));
    let resp = tr.post("/v1/responses", &[], simple_request(Protocol::OaiResponses, "gpt-4o", "hi", true)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert!(resp.text().contains("data:"));
    assert_trace_ok(&tr, &resp);
}

#[tokio::test]
async fn j3_golden_chat_error() {
    let tr = TestRouter::example(MockUpstream::always(golden_err("chat/error_auth.err.json")));
    let resp = tr.post("/v1/chat/completions", CHAT_UP, simple_request(Protocol::OaiChat, "gpt-4o", "hi", false)).await;
    assert_eq!(resp.status, 401, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    assert!(!rec.errors.is_empty());
}

// ── Group K: trace content ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn k1_trace_has_all_sections() {
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hello"))));
    let resp = tr.post("/v1/responses", &[], simple_request(Protocol::OaiResponses, "gpt-4o", "hi", false)).await;
    let rec = assert_trace_ok(&tr, &resp);
    assert!(rec.ir_request.is_some(), "ir_request");
    assert!(rec.route.is_some(), "route");
    assert!(rec.lowered.is_some(), "lowered");
    assert!(rec.upstream.is_some(), "upstream");
    assert!(rec.ir_response.is_some(), "ir_response");
    assert!(rec.client_response.is_some(), "client_response");
    assert!(rec.trace_id.starts_with("tr_"));
    assert_eq!(rec.client_response.unwrap().status, 200);
    assert!(rec.upstream.unwrap().raw.is_some(), "raw upstream captured");
}

#[tokio::test]
async fn k2_deterministic_ids() {
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi"))));
    let resp = tr.post("/v1/responses", &[], simple_request(Protocol::OaiResponses, "gpt-4o", "hi", false)).await;
    // Sequential id source: the first minted id is the trace id `tr_00000001`.
    assert_eq!(resp.trace_id(), "tr_00000001");
}

#[tokio::test]
async fn k3_media_redaction_in_trace() {
    let mut cfg = Config::example();
    cfg.trace.redact_media_over_bytes = 64;
    let big = "/9j/".to_string() + &"A".repeat(500);
    let body = json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "x"},
            {"type": "input_image", "image_url": format!("data:image/jpeg;base64,{big}")}
        ]}]
    });
    let tr = TestRouter::build(cfg, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "ok"))));
    let resp = tr.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    let s = serde_json::to_string(&rec.client.body).unwrap();
    assert!(s.contains("$redacted"), "large media not redacted: {s}");
}

#[tokio::test]
async fn k4_routes_resolution_unknown_family() {
    // A provider the registry does not know → unknown caps source.
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiChat, &text_response("zzz", "hi"))));
    let hdrs: &[(&str, &str)] = &[("x-xlate-provider", "mystery"), ("x-xlate-upstream", "chat")];
    let resp = tr.post("/v1/chat/completions", hdrs, simple_request(Protocol::OaiChat, "zzz-model", "hi", false)).await;
    // No provider config for "mystery" → base URL empty; the mock still answers. Assert routing.
    let rec = assert_trace_ok(&tr, &resp);
    assert_eq!(rec.route.as_ref().unwrap().get("caps_source").and_then(Value::as_str), Some("unknown"));
}

#[tokio::test]
async fn i11b_long_client_user_id_is_rewritten_and_traced() {
    // Regression (Claude Code on an OpenAI model, 2026-09-10): Claude Code sends a ~150-char
    // `metadata.user_id`; OpenAI caps `user` at 64 chars. The encoder now rewrites it to a digest,
    // and that encoder-side degradation must reach the trace and the `x-router-degraded` header
    // (previously only lowering's degradations were recorded).
    let tr = TestRouter::example(MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-5.4", "pong"))));
    let long_id = format!("user_{}", "a".repeat(145));
    let body = json!({
        "model": "gpt-5.4", "max_tokens": 16,
        "metadata": {"user_id": long_id},
        "messages": [{"role": "user", "content": "Say exactly: pong"}]
    });
    let hdrs: &[(&str, &str)] = &[("anthropic-version", "2023-06-01")];
    let resp = tr.post("/v1/messages", hdrs, Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let rec = assert_trace_ok(&tr, &resp);
    let up = rec.upstream.as_ref().expect("upstream leg");
    let user = up.body.as_ref().and_then(|b| b.get("user")).and_then(Value::as_str).expect("user sent upstream");
    assert_eq!(user.len(), 64, "over-long user id must be digested to OpenAI's limit");
    let degs = rec.lowered.as_ref().map(|l| l.degradations.clone()).unwrap_or_default();
    assert!(
        degs.iter().any(|d| d.get("field").and_then(Value::as_str) == Some("user")),
        "encoder degradation must be in the trace: {degs:?}"
    );
    assert!(rec.checks.degradation_count >= 1);
    assert!(resp.headers.get("x-router-degraded").is_some(), "header must reflect encoder degradations");
}
