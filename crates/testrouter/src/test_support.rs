//! In-process test scaffolding (plan §11, §8): a [`MockUpstream`] whose responses are chosen by a
//! `Vec<`[`Rule`]`>`, fixture builders that produce provider bodies/SSE through the real codecs (so
//! the mock serves bytes the router then decodes), golden-capture loaders, and a [`TestRouter`]
//! harness that drives the app with deterministic ids + clock and an in-memory trace sink.
//!
//! `MockUpstream` implements the [`Upstream`] trait directly rather than binding a TCP port. This
//! keeps the whole suite hermetic and offline (no sockets, no flakiness) while satisfying the
//! brief's intent — responses chosen per request by a predicate over `(path, body)` and served as
//! a status + headers + full body or SSE frames with optional per-frame delays.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{Body, Bytes};
use serde_json::{json, Value};

use llm_xlate::Translator;
use llm_xlate_core::codec::{EncodeCtx, TranslatorConfig};
use llm_xlate_core::ir::{
    IrEvent, IrResponse, Item, ItemId, ItemKind, Part, Protocol, ResponseId, Role, StopReason,
    Usage,
};

use crate::config::Config;
use crate::ids::{FixedClock, SequentialSource};
use crate::sidecar::MemorySidecar;
use crate::store::MemoryStore;
use crate::trace::{TraceRecord, TraceSink};
use crate::upstream::{
    upstream_path, BoxFuture, ByteStream, Upstream, UpstreamBody, UpstreamError, UpstreamRequest,
    UpstreamResponse,
};
use crate::{App, Deps};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Mock upstream
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// What the mock knows about a request when choosing a rule.
pub struct MockReq {
    /// Upstream path (e.g. `/v1/responses`).
    pub path: String,
    /// Upstream protocol.
    pub protocol: Protocol,
    /// The parsed request body (or `Null` if not JSON).
    pub body: Value,
}

/// One SSE frame to emit, with a delay before it is produced.
pub struct SseFrame {
    /// Delay before this frame is yielded.
    pub delay: Duration,
    /// The exact frame bytes.
    pub bytes: Bytes,
}

/// How a matched rule responds.
pub enum Responder {
    /// A fully-buffered body.
    Full {
        /// HTTP status.
        status: u16,
        /// Extra response headers.
        headers: Vec<(String, String)>,
        /// The body bytes.
        body: Bytes,
    },
    /// A streaming SSE body.
    Sse {
        /// HTTP status.
        status: u16,
        /// Extra response headers.
        headers: Vec<(String, String)>,
        /// The frames to emit, in order.
        frames: Vec<SseFrame>,
    },
}

type Matcher = Box<dyn Fn(&MockReq) -> bool + Send + Sync>;

/// A mock upstream rule: a predicate over the request and the response to serve.
pub struct Rule {
    matcher: Matcher,
    responder: Responder,
}

impl Rule {
    /// A rule that matches every request.
    pub fn any(responder: Responder) -> Rule {
        Rule { matcher: Box::new(|_| true), responder }
    }
    /// A rule that matches a given upstream path.
    pub fn path(path: impl Into<String>, responder: Responder) -> Rule {
        let p = path.into();
        Rule { matcher: Box::new(move |r| r.path == p), responder }
    }
    /// A rule with a custom predicate.
    pub fn when(matcher: impl Fn(&MockReq) -> bool + Send + Sync + 'static, responder: Responder) -> Rule {
        Rule { matcher: Box::new(matcher), responder }
    }
}

/// An in-process [`Upstream`] mock (plan §8/§11). Rules are tried in order; the first match wins.
pub struct MockUpstream {
    rules: Vec<Rule>,
    /// Requests the mock actually received (for assertions).
    pub seen: Arc<Mutex<Vec<Value>>>,
}

impl MockUpstream {
    /// A mock with the given rules.
    pub fn new(rules: Vec<Rule>) -> Self {
        Self { rules, seen: Arc::new(Mutex::new(Vec::new())) }
    }
    /// A mock that answers every request with `responder`.
    pub fn always(responder: Responder) -> Self {
        Self::new(vec![Rule::any(responder)])
    }
}

impl Upstream for MockUpstream {
    fn send<'a>(&'a self, req: UpstreamRequest) -> BoxFuture<'a, Result<UpstreamResponse, UpstreamError>> {
        Box::pin(async move {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            self.seen.lock().unwrap().push(body.clone());
            let path = if req.url.is_empty() {
                upstream_path(req.protocol).to_string()
            } else {
                // Extract the path portion of the URL.
                req.url
                    .find("/v1/")
                    .map(|i| req.url[i..].to_string())
                    .unwrap_or_else(|| upstream_path(req.protocol).to_string())
            };
            let mr = MockReq { path, protocol: req.protocol, body };
            let responder = self
                .rules
                .iter()
                .find(|r| (r.matcher)(&mr))
                .map(|r| &r.responder)
                .ok_or_else(|| UpstreamError::new("mock upstream: no rule matched"))?;
            Ok(build_response(responder))
        })
    }
}

fn header_map(pairs: &[(String, String)]) -> llm_xlate_core::HeaderMap {
    let mut h = llm_xlate_core::HeaderMap::new();
    for (k, v) in pairs {
        if let (Ok(name), Ok(val)) = (
            k.parse::<llm_xlate_core::HeaderName>(),
            v.parse::<llm_xlate_core::HeaderValue>(),
        ) {
            h.insert(name, val);
        }
    }
    h
}

fn build_response(responder: &Responder) -> UpstreamResponse {
    match responder {
        Responder::Full { status, headers, body } => UpstreamResponse {
            status: *status,
            headers: header_map(headers),
            body: UpstreamBody::Full(body.clone()),
            ttfb: Duration::from_millis(0),
        },
        Responder::Sse { status, headers, frames } => {
            let frames: Vec<(Duration, Bytes)> = frames.iter().map(|f| (f.delay, f.bytes.clone())).collect();
            let ttfb = frames.first().map(|(d, _)| *d).unwrap_or_default();
            let stream: ByteStream = Box::pin(async_stream::stream! {
                for (delay, bytes) in frames {
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    yield Ok::<Bytes, UpstreamError>(bytes);
                }
            });
            UpstreamResponse {
                status: *status,
                headers: header_map(headers),
                body: UpstreamBody::Stream(stream),
                ttfb,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Fixture builders — provider bytes through the real codecs
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn fixture_translator() -> Translator {
    Translator::new(TranslatorConfig::default())
}

fn fixture_ctx(protocol: Protocol, model: &str) -> EncodeCtx {
    let t = fixture_translator();
    let mut ctx = EncodeCtx::new(protocol, model.to_string(), ResponseId::new("resp_upstream"), t.sealer());
    // Fixtures expose reasoning fully and opt into encrypted content so reasoning-bearing
    // fixtures actually carry their reasoning through the provider bytes.
    ctx.expose = llm_xlate_core::ir::ReasoningExposure::Full;
    ctx.include = vec!["reasoning.encrypted_content".to_string()];
    ctx
}

/// Encode an [`IrResponse`] as a provider (non-streaming) body for `protocol`.
pub fn full_body_for(protocol: Protocol, resp: &IrResponse) -> Bytes {
    let t = fixture_translator();
    let ctx = fixture_ctx(protocol, &resp.model);
    t.encode_response(protocol, resp, &ctx)
}

/// Encode a sequence of [`IrEvent`]s as provider SSE frames for `protocol`.
pub fn sse_frames_for(protocol: Protocol, model: &str, events: &[IrEvent]) -> Vec<Bytes> {
    let t = fixture_translator();
    let ctx = fixture_ctx(protocol, model);
    let mut enc = t.stream_encoder(protocol, ctx);
    let mut out = Vec::new();
    for ev in events {
        out.extend(enc.push(ev.clone()));
    }
    out.extend(enc.finish());
    out
}

/// A [`Responder::Full`] for a provider (non-streaming) body.
pub fn respond_full(protocol: Protocol, resp: &IrResponse) -> Responder {
    Responder::Full { status: 200, headers: vec![], body: full_body_for(protocol, resp) }
}

/// A [`Responder::Sse`] for provider SSE, with `delay` before the first frame and no delay after.
pub fn respond_sse(protocol: Protocol, model: &str, events: &[IrEvent], first_delay: Duration) -> Responder {
    let frames = sse_frames_for(protocol, model, events);
    let sse = frames
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| SseFrame { delay: if i == 0 { first_delay } else { Duration::ZERO }, bytes })
        .collect();
    Responder::Sse { status: 200, headers: vec![], frames: sse }
}

/// A [`Responder::Full`] error body for `protocol` at `status`.
pub fn respond_error(protocol: Protocol, err: &llm_xlate_core::error::XlateError) -> Responder {
    let t = fixture_translator();
    let enc = t.encode_error(protocol, err, false, false);
    let headers = enc
        .headers
        .iter()
        .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.as_str().to_string(), s.to_string())))
        .collect();
    Responder::Full { status: enc.status, headers, body: enc.body }
}

/// A raw [`Responder::Full`] from status + JSON bytes.
pub fn respond_raw(status: u16, body: impl Into<Bytes>) -> Responder {
    Responder::Full { status, headers: vec![], body: body.into() }
}

// ── IR builders for the common cases ─────────────────────────────────────────────────────────

/// An assistant text [`IrResponse`].
pub fn text_response(model: &str, text: &str) -> IrResponse {
    IrResponse {
        id: ResponseId::new("resp_upstream"),
        model: model.to_string(),
        items: vec![Item::Message {
            role: Role::Assistant,
            content: vec![Part::text(text)],
            id: Some(ItemId::new("msg_up_1")),
        }],
        stop: StopReason::EndTurn,
        usage: Usage::new(11, 7),
        ext: Default::default(),
    }
}

/// The streaming events for an assistant text turn.
pub fn text_events(model: &str, text: &str) -> Vec<IrEvent> {
    vec![
        IrEvent::Start { response_id: ResponseId::new("resp_upstream"), model: model.to_string(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: Some(ItemId::new("msg_up_1")), call: None },
        IrEvent::Delta { index: 0, delta: llm_xlate_core::ir::Delta::Text(text.to_string()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(11, 7), ext: Default::default() },
    ]
}

/// A tool-call [`IrResponse`].
pub fn tool_call_response(model: &str, call_id: &str, name: &str, args: &str) -> IrResponse {
    IrResponse {
        id: ResponseId::new("resp_upstream"),
        model: model.to_string(),
        items: vec![Item::ToolCall {
            call_id: llm_xlate_core::ir::CallId::new(call_id),
            name: name.to_string(),
            arguments: llm_xlate_core::ir::JsonText::new(args),
            id: Some(ItemId::new("fc_up_1")),
        }],
        stop: StopReason::ToolUse,
        usage: Usage::new(20, 9),
        ext: Default::default(),
    }
}

/// The streaming events for a tool-call turn.
pub fn tool_call_events(model: &str, call_id: &str, name: &str, args: &str) -> Vec<IrEvent> {
    vec![
        IrEvent::Start { response_id: ResponseId::new("resp_upstream"), model: model.to_string(), usage_prefill: None },
        IrEvent::ItemStart {
            index: 0,
            kind: ItemKind::ToolCall,
            id: Some(ItemId::new("fc_up_1")),
            call: Some((llm_xlate_core::ir::CallId::new(call_id), name.to_string())),
        },
        IrEvent::Delta { index: 0, delta: llm_xlate_core::ir::Delta::ToolArgs(args.to_string()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::ToolUse, usage: Usage::new(20, 9), ext: Default::default() },
    ]
}

/// A reasoning + answer [`IrResponse`] for `family` (Anthropic signature / OpenAI encrypted).
pub fn reasoning_response(model: &str, family: llm_xlate_core::ir::ProviderFamily, think: &str, answer: &str) -> IrResponse {
    use llm_xlate_core::ir::{OpaqueBlob, OpaqueKind, ReasoningItem};
    let kind = match family {
        llm_xlate_core::ir::ProviderFamily::Anthropic => OpaqueKind::Signature,
        _ => OpaqueKind::Encrypted,
    };
    IrResponse {
        id: ResponseId::new("resp_upstream"),
        model: model.to_string(),
        items: vec![
            Item::Reasoning(ReasoningItem {
                text: Some(think.to_string()),
                summary: vec![think.to_string()],
                opaque: Some(OpaqueBlob::new(family, kind, "UkVBU09OQkxPQg==")),
                id: Some(ItemId::new("rs_up_1")),
            }),
            Item::Message {
                role: Role::Assistant,
                content: vec![Part::text(answer)],
                id: Some(ItemId::new("msg_up_1")),
            },
        ],
        stop: StopReason::EndTurn,
        usage: Usage { input: 30, output: 15, reasoning: Some(8), ..Default::default() },
        ext: Default::default(),
    }
}

/// The streaming events for a reasoning + answer turn.
pub fn reasoning_events(model: &str, family: llm_xlate_core::ir::ProviderFamily, think: &str, answer: &str) -> Vec<IrEvent> {
    use llm_xlate_core::ir::{Delta, OpaqueBlob, OpaqueKind};
    let kind = match family {
        llm_xlate_core::ir::ProviderFamily::Anthropic => OpaqueKind::Signature,
        _ => OpaqueKind::Encrypted,
    };
    vec![
        IrEvent::Start { response_id: ResponseId::new("resp_upstream"), model: model.to_string(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: Some(ItemId::new("rs_up_1")), call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningText(think.to_string()) },
        IrEvent::Delta { index: 0, delta: Delta::Opaque(OpaqueBlob::new(family, kind, "UkVBU09OQkxPQg==")) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::ItemStart { index: 1, kind: ItemKind::Message, id: Some(ItemId::new("msg_up_1")), call: None },
        IrEvent::Delta { index: 1, delta: Delta::Text(answer.to_string()) },
        IrEvent::ItemStop { index: 1 },
        IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage { input: 30, output: 15, reasoning: Some(8), ..Default::default() }, ext: Default::default() },
    ]
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Golden-capture loaders
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The workspace `crates/xlate/tests/golden` directory.
pub fn golden_dir() -> std::path::PathBuf {
    llm_xlate_e2e::keys::workspace_root()
        .expect("workspace root")
        .join("crates/xlate/tests/golden")
}

/// Load a golden `.resp.json` (a provider response body) as a [`Responder::Full`].
pub fn golden_resp(rel: &str) -> Responder {
    let bytes = std::fs::read(golden_dir().join(rel)).unwrap_or_else(|e| panic!("reading golden {rel}: {e}"));
    Responder::Full { status: 200, headers: vec![], body: Bytes::from(bytes) }
}

/// Load a golden `.stream.sse` as a [`Responder::Sse`] (all frames emitted at once).
pub fn golden_sse(rel: &str) -> Responder {
    let text = std::fs::read_to_string(golden_dir().join(rel)).unwrap_or_else(|e| panic!("reading golden {rel}: {e}"));
    // Serve the whole SSE document as a single frame; the SSE parser is byte-boundary-safe.
    Responder::Sse {
        status: 200,
        headers: vec![],
        frames: vec![SseFrame { delay: Duration::ZERO, bytes: Bytes::from(text) }],
    }
}

/// Load a golden `.err.json` (`{status, headers, body}`) as a [`Responder::Full`].
pub fn golden_err(rel: &str) -> Responder {
    let text = std::fs::read_to_string(golden_dir().join(rel)).unwrap_or_else(|e| panic!("reading golden {rel}: {e}"));
    let v: Value = serde_json::from_str(&text).expect("golden err json");
    let status = v.get("status").and_then(Value::as_u64).unwrap_or(500) as u16;
    let headers = v
        .get("headers")
        .and_then(Value::as_object)
        .map(|o| o.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
        .unwrap_or_default();
    let body = serde_json::to_vec(v.get("body").unwrap_or(&Value::Null)).unwrap();
    Responder::Full { status, headers, body: Bytes::from(body) }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// In-memory trace sink
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A [`TraceSink`] that keeps every record in memory for assertions.
#[derive(Default)]
pub struct MemoryTraceSink {
    records: Mutex<Vec<TraceRecord>>,
}

impl MemoryTraceSink {
    /// A fresh sink.
    pub fn new() -> Self {
        Self::default()
    }
    /// All records so far, in submission order.
    pub fn all(&self) -> Vec<TraceRecord> {
        self.records.lock().unwrap().clone()
    }
    /// The record for a given trace id, if present.
    pub fn get(&self, trace_id: &str) -> Option<TraceRecord> {
        self.records.lock().unwrap().iter().find(|r| r.trace_id == trace_id).cloned()
    }
    /// How many records have been submitted.
    pub fn len(&self) -> usize {
        self.records.lock().unwrap().len()
    }
    /// Whether no records have been submitted.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl TraceSink for MemoryTraceSink {
    fn submit(&self, record: TraceRecord) {
        self.records.lock().unwrap().push(record);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Test harness
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A response captured from the router.
pub struct TestResponse {
    /// HTTP status.
    pub status: u16,
    /// Response headers.
    pub headers: llm_xlate_core::HeaderMap,
    /// The (fully-collected) body bytes.
    pub body: Bytes,
}

impl TestResponse {
    /// The `x-xlate-trace-id` header.
    pub fn trace_id(&self) -> String {
        self.headers.get("x-xlate-trace-id").and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
    }
    /// The body parsed as JSON.
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    /// The body as a UTF-8 string (for SSE).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// A deterministic in-process router for integration tests.
pub struct TestRouter {
    app: axum::Router,
    /// The in-memory trace sink.
    pub sink: Arc<MemoryTraceSink>,
    /// The in-memory response store.
    pub store: Arc<MemoryStore>,
    /// The in-memory sidecar.
    pub sidecar: Arc<MemorySidecar>,
}

impl TestRouter {
    /// Build a router from a config + mock upstream with deterministic ids + clock.
    pub fn build(config: Config, mock: MockUpstream) -> TestRouter {
        Self::build_with(config, Arc::new(mock))
    }

    /// Build from a shared upstream (e.g. one whose `seen` list the test inspects).
    pub fn build_with(config: Config, upstream: Arc<dyn Upstream>) -> TestRouter {
        let sink = Arc::new(MemoryTraceSink::new());
        let store = Arc::new(MemoryStore::new());
        let sidecar = Arc::new(MemorySidecar::new());
        let deps = Deps {
            translator: Translator::new(TranslatorConfig::default()),
            id_source: Arc::new(SequentialSource::new()),
            clock: Arc::new(FixedClock::default()),
            store: store.clone(),
            sidecar: sidecar.clone(),
            trace_sink: sink.clone(),
            upstream,
        };
        let app = App::build(config, deps).expect("build app");
        TestRouter { app, sink, store, sidecar }
    }

    /// The default example config.
    pub fn example(mock: MockUpstream) -> TestRouter {
        TestRouter::build(Config::example(), mock)
    }

    /// POST a request and collect the full response (drives any stream to completion).
    pub async fn post(&self, path: &str, headers: &[(&str, &str)], body: impl Into<Bytes>) -> TestResponse {
        use tower::ServiceExt;
        let mut builder = axum::http::Request::builder().method("POST").uri(path);
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        let req = builder.body(Body::from(body.into())).unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        TestResponse { status, headers, body }
    }

    /// GET a path and collect the full response.
    pub async fn get(&self, path: &str) -> TestResponse {
        use tower::ServiceExt;
        let req = axum::http::Request::builder().method("GET").uri(path).body(Body::empty()).unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        TestResponse { status, headers, body }
    }

    /// Return the streaming response object without collecting its body (for disconnect tests).
    pub async fn post_raw(&self, path: &str, headers: &[(&str, &str)], body: impl Into<Bytes>) -> axum::response::Response {
        use tower::ServiceExt;
        let mut builder = axum::http::Request::builder().method("POST").uri(path);
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        let req = builder.body(Body::from(body.into())).unwrap();
        self.app.clone().oneshot(req).await.unwrap()
    }

    /// The trace record for a captured response's trace id.
    pub fn trace(&self, resp: &TestResponse) -> TraceRecord {
        let id = resp.trace_id();
        self.sink.get(&id).unwrap_or_else(|| panic!("no trace for {id}"))
    }
}

/// A minimal request body for a given client protocol.
pub fn simple_request(protocol: Protocol, model: &str, user_text: &str, stream: bool) -> Bytes {
    let v = match protocol {
        Protocol::OaiChat => json!({
            "model": model,
            "messages": [{"role": "user", "content": user_text}],
            "stream": stream,
        }),
        Protocol::OaiResponses => json!({
            "model": model,
            "input": user_text,
            "stream": stream,
        }),
        Protocol::Anthropic => json!({
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_text}],
            "stream": stream,
        }),
    };
    Bytes::from(serde_json::to_vec(&v).unwrap())
}
