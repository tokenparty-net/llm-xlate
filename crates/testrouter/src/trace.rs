//! The trace record (plan §5): one JSON object per request with a **fixed field order**, a
//! builder that every pipeline stage feeds, inline invariant [`Checks`], header + media
//! redaction, and a non-blocking [`FileTraceWriter`].
//!
//! Field order is the struct declaration order (serde serializes struct fields in order), so the
//! on-disk schema is stable regardless of `serde_json`'s `preserve_order` feature.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use llm_xlate_core::aggregate::Aggregator;
use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::codec::EncodeCtx;
use llm_xlate_core::ir::{IrEvent, IrResponse, Item, Part, Protocol};
use llm_xlate_core::HeaderMap;

use llm_xlate::Translator;

use llm_xlate_e2e::keys::{is_secret_header, looks_like_key};

/// One trace record (plan §5). Field order is fixed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    /// Schema version.
    pub v: u32,
    /// Router trace id (`tr_…`).
    pub trace_id: String,
    /// Request start timestamp (RFC-3339).
    pub ts_start: String,
    /// Request end timestamp (RFC-3339).
    pub ts_end: String,
    /// Session tag (`x-xlate-tag`), if any.
    pub tag: Option<String>,
    /// What the client sent.
    pub client: ClientReq,
    /// The decoded IR request.
    pub ir_request: Option<Value>,
    /// The resolved route.
    pub route: Option<Value>,
    /// Chain (`previous_response_id`) info.
    pub chain: ChainTrace,
    /// The requirements pre-pass result.
    pub requirements: Option<Value>,
    /// A summary of the resolutions supplied to lowering.
    pub resolutions_summary: Option<Value>,
    /// The lowered IR + degradations.
    pub lowered: Option<LoweredTrace>,
    /// What was sent upstream and what came back.
    pub upstream: Option<UpstreamTrace>,
    /// The IR events decoded from the upstream response, with arrival offsets.
    pub ir_events: Vec<EventTrace>,
    /// The aggregated IR response.
    pub ir_response: Option<Value>,
    /// What the client actually received.
    pub client_response: Option<ClientResponse>,
    /// Errors recorded along the way.
    pub errors: Vec<ErrorTrace>,
    /// Stored-response bookkeeping.
    pub store: Option<StoreTrace>,
    /// Timings.
    pub timing: Timing,
    /// Inline invariant checks (plan §11).
    pub checks: Checks,
}

/// The client request block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientReq {
    /// Client protocol token.
    pub protocol: String,
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Redacted request headers.
    pub headers: Map<String, Value>,
    /// The parsed request body (JSON), media-redacted.
    pub body: Option<Value>,
    /// SHA-256 of the raw request bytes.
    pub body_raw_sha256: String,
    /// Whether the client asked for a stream.
    pub stream: bool,
}

/// The chain block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChainTrace {
    /// The `previous_response_id` this request chained from, if any.
    pub previous_response_id: Option<String>,
    /// The materialized chain ids (oldest → newest).
    pub materialized_ids: Vec<String>,
}

/// The lowered-request block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoweredTrace {
    /// The lowered IR.
    pub ir: Value,
    /// The degradations recorded while lowering + wiring.
    pub degradations: Vec<Value>,
}

/// The upstream block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpstreamTrace {
    /// The request URL.
    pub url: String,
    /// Redacted request headers.
    pub headers: Map<String, Value>,
    /// The upstream request body (JSON), media-redacted.
    pub body: Option<Value>,
    /// Whether the upstream request asked for a stream.
    pub stream: bool,
    /// HTTP status.
    pub status: u16,
    /// Redacted response headers.
    pub resp_headers: Map<String, Value>,
    /// Upstream request id, if any.
    pub request_id: Option<String>,
    /// The exact upstream bytes (SSE or JSON), when `include_raw_sse`.
    pub raw: Option<String>,
    /// Total upstream elapsed time.
    pub elapsed_ms: u64,
    /// Time to first byte / frame.
    pub ttfb_ms: u64,
}

/// One recorded IR event with an arrival offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTrace {
    /// Milliseconds since request start.
    pub t_ms: u64,
    /// The IR event.
    pub event: Value,
}

/// The client-response block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientResponse {
    /// HTTP status.
    pub status: u16,
    /// Response headers.
    pub headers: Map<String, Value>,
    /// Streamed frames, when the client streamed.
    pub frames: Option<Vec<FrameTrace>>,
    /// The non-streaming body, when the client did not stream.
    pub body: Option<Value>,
}

/// One recorded client frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTrace {
    /// Milliseconds since request start.
    pub t_ms: u64,
    /// Whether the frame is a keepalive comment.
    pub keepalive: bool,
    /// The exact frame bytes as a (lossy) string.
    pub data: String,
}

/// One recorded error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrace {
    /// The pipeline stage the error occurred in.
    pub stage: String,
    /// The error kind slug.
    pub kind: String,
    /// HTTP status rendered.
    pub status: u16,
    /// Error message.
    pub message: String,
    /// The rendered client-dialect error body, if any.
    pub rendered: Option<Value>,
}

/// Stored-response bookkeeping.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoreTrace {
    /// The id stored, if the response was persisted.
    pub stored_id: Option<String>,
    /// The previous id in the chain, if any.
    pub previous: Option<String>,
}

/// Timings (plan §5).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timing {
    /// Decode microseconds.
    pub decode_us: u64,
    /// Lower microseconds.
    pub lower_us: u64,
    /// Encode microseconds.
    pub encode_us: u64,
    /// Upstream milliseconds.
    pub upstream_ms: u64,
    /// Total milliseconds.
    pub total_ms: u64,
}

/// Inline invariant checks (plan §11). A failing check never affects the client response — it is
/// the whole point of the trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checks {
    /// The upstream IR events re-aggregated equal the recorded `ir_response` (plan §11.8).
    pub aggregate_matches_encode_response: bool,
    /// Fields that differ when the client request is decoded, re-encoded in its own protocol, and
    /// decoded again (empty = a clean round trip).
    pub reencode_client_request_diff: Vec<String>,
    /// How many degradations lowering recorded.
    pub degradation_count: u32,
}

impl Default for Checks {
    fn default() -> Self {
        Self {
            aggregate_matches_encode_response: true,
            reencode_client_request_diff: Vec::new(),
            degradation_count: 0,
        }
    }
}

/// A compact index line for triage (plan §5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceIndexEntry {
    /// Trace id.
    pub trace_id: String,
    /// Start timestamp.
    pub ts: String,
    /// Session tag.
    pub tag: Option<String>,
    /// Client protocol.
    pub client_protocol: String,
    /// Provider name.
    pub provider: Option<String>,
    /// Upstream protocol.
    pub upstream_protocol: Option<String>,
    /// Final client status.
    pub status: u16,
    /// The first error kind, if any.
    pub error_kind: Option<String>,
    /// Degradation count.
    pub degradation_count: u32,
    /// Whether any inline check failed.
    pub check_failures: bool,
    /// Whether the request was cancelled.
    pub cancelled: bool,
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Accumulates a [`TraceRecord`] across pipeline stages, applying redaction as data arrives.
pub struct TraceBuilder {
    record: TraceRecord,
    /// Threshold above which a string payload is media-redacted.
    redact_over: usize,
    /// Keep raw upstream/frame bytes.
    include_raw: bool,
    /// The upstream events collected, for the aggregate check.
    events: Vec<IrEvent>,
    /// The recorded aggregated response, for the aggregate check.
    ir_response: Option<IrResponse>,
    /// Whether the request was cancelled mid-stream.
    cancelled: bool,
    /// The plan-§5/§11 client-frame law result, computed by the pipeline/stream from the actual
    /// client-facing output and set via [`set_aggregate_check`](TraceBuilder::set_aggregate_check).
    /// When present it wins over the (weaker) upstream re-aggregation fallback.
    aggregate_check_override: Option<bool>,
}

impl TraceBuilder {
    /// Start a record.
    pub fn new(
        trace_id: String,
        ts_start: String,
        tag: Option<String>,
        redact_over: usize,
        include_raw: bool,
    ) -> Self {
        let record = TraceRecord {
            v: 1,
            trace_id,
            ts_start,
            ts_end: String::new(),
            tag,
            client: ClientReq {
                protocol: String::new(),
                method: String::new(),
                path: String::new(),
                headers: Map::new(),
                body: None,
                body_raw_sha256: String::new(),
                stream: false,
            },
            ir_request: None,
            route: None,
            chain: ChainTrace::default(),
            requirements: None,
            resolutions_summary: None,
            lowered: None,
            upstream: None,
            ir_events: Vec::new(),
            ir_response: None,
            client_response: None,
            errors: Vec::new(),
            store: None,
            timing: Timing::default(),
            checks: Checks::default(),
        };
        Self {
            record,
            redact_over,
            include_raw,
            events: Vec::new(),
            ir_response: None,
            cancelled: false,
            aggregate_check_override: None,
        }
    }

    /// The trace id.
    pub fn trace_id(&self) -> &str {
        &self.record.trace_id
    }

    /// Record the client request.
    pub fn set_client(
        &mut self,
        protocol: &str,
        method: &str,
        path: &str,
        headers: &HeaderMap,
        raw_body: &[u8],
        stream: bool,
    ) {
        self.record.client.protocol = protocol.to_string();
        self.record.client.method = method.to_string();
        self.record.client.path = path.to_string();
        self.record.client.headers = redact_headers(headers);
        self.record.client.body = self.parse_and_redact(raw_body);
        self.record.client.body_raw_sha256 = sha256_hex(raw_body);
        self.record.client.stream = stream;
    }

    /// Record the decoded IR request.
    pub fn set_ir_request(&mut self, ir: &Value) {
        self.record.ir_request = Some(self.redact_value(ir.clone()));
    }

    /// Record the resolved route (any serializable summary).
    pub fn set_route(&mut self, route: Value) {
        self.record.route = Some(route);
    }

    /// Record the chain block.
    pub fn set_chain(&mut self, previous: Option<String>, materialized: Vec<String>) {
        self.record.chain = ChainTrace { previous_response_id: previous, materialized_ids: materialized };
    }

    /// Record the requirements pre-pass.
    pub fn set_requirements(&mut self, requirements: Value, resolutions_summary: Value) {
        self.record.requirements = Some(requirements);
        self.record.resolutions_summary = Some(resolutions_summary);
    }

    /// Record the lowered IR + degradations.
    pub fn set_lowered(&mut self, ir: Value, degradations: Vec<Value>) {
        self.record.checks.degradation_count = degradations.len() as u32;
        self.record.lowered = Some(LoweredTrace { ir: self.redact_value(ir), degradations });
    }

    /// Append degradations recorded by a later stage (the upstream encoder reports its own lossy
    /// steps, e.g. an over-long `user` identifier rewritten to fit the provider's limit) so the
    /// trace and the `x-router-degraded` header cover the whole request path, not just lowering.
    pub fn add_degradations(&mut self, more: Vec<Value>) {
        if more.is_empty() {
            return;
        }
        match self.record.lowered.as_mut() {
            Some(l) => l.degradations.extend(more),
            None => self.record.lowered = Some(LoweredTrace { ir: Value::Null, degradations: more }),
        }
        self.record.checks.degradation_count =
            self.record.lowered.as_ref().map(|l| l.degradations.len() as u32).unwrap_or(0);
    }

    /// Record the upstream request side.
    pub fn set_upstream_request(&mut self, url: &str, headers: &HeaderMap, body: &[u8], stream: bool) {
        let up = self.record.upstream.get_or_insert_with(UpstreamTrace::default);
        up.url = url.to_string();
        up.headers = redact_headers(headers);
        up.stream = stream;
        up.body = parse_and_redact_impl(body, self.redact_over);
    }

    /// Record the upstream response side.
    pub fn set_upstream_response(
        &mut self,
        status: u16,
        headers: &HeaderMap,
        raw: Option<&[u8]>,
        elapsed_ms: u64,
        ttfb_ms: u64,
    ) {
        let request_id = headers
            .get("x-request-id")
            .or_else(|| headers.get("request-id"))
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let up = self.record.upstream.get_or_insert_with(UpstreamTrace::default);
        up.status = status;
        up.resp_headers = redact_headers(headers);
        up.request_id = request_id;
        up.elapsed_ms = elapsed_ms;
        up.ttfb_ms = ttfb_ms;
        if self.include_raw {
            if let Some(raw) = raw {
                up.raw = Some(String::from_utf8_lossy(raw).into_owned());
            }
        }
    }

    /// Record one decoded IR event.
    pub fn push_event(&mut self, t_ms: u64, event: &IrEvent) {
        self.events.push(event.clone());
        if let Ok(v) = serde_json::to_value(event) {
            self.record.ir_events.push(EventTrace { t_ms, event: self.redact_value(v) });
        }
    }

    /// Record the aggregated IR response.
    pub fn set_ir_response(&mut self, resp: &IrResponse) {
        self.ir_response = Some(resp.clone());
        if let Ok(v) = serde_json::to_value(resp) {
            self.record.ir_response = Some(self.redact_value(v));
        }
    }

    /// Record a non-streaming client response body.
    pub fn set_client_response_body(&mut self, status: u16, headers: &HeaderMap, body: &[u8]) {
        self.record.client_response = Some(ClientResponse {
            status,
            headers: redact_headers(headers),
            frames: None,
            body: self.parse_and_redact(body),
        });
    }

    /// Start a streaming client response (frames are pushed as they are sent).
    pub fn start_client_response_stream(&mut self, status: u16, headers: &HeaderMap) {
        self.record.client_response = Some(ClientResponse {
            status,
            headers: redact_headers(headers),
            frames: Some(Vec::new()),
            body: None,
        });
    }

    /// Record one client frame.
    pub fn push_frame(&mut self, t_ms: u64, keepalive: bool, data: &[u8]) {
        if !self.include_raw {
            return;
        }
        if let Some(cr) = &mut self.record.client_response {
            if let Some(frames) = &mut cr.frames {
                frames.push(FrameTrace { t_ms, keepalive, data: String::from_utf8_lossy(data).into_owned() });
            }
        }
    }

    /// Record an error.
    pub fn push_error(&mut self, stage: &str, kind: &str, status: u16, message: &str, rendered: Option<Value>) {
        self.record.errors.push(ErrorTrace {
            stage: stage.to_string(),
            kind: kind.to_string(),
            status,
            message: message.to_string(),
            rendered,
        });
    }

    /// Record stored-response bookkeeping.
    pub fn set_store(&mut self, stored_id: Option<String>, previous: Option<String>) {
        self.record.store = Some(StoreTrace { stored_id, previous });
    }

    /// Record timings.
    pub fn set_timing(&mut self, timing: Timing) {
        self.record.timing = timing;
    }

    /// Set only the total-milliseconds timing field (at finalize time).
    pub fn set_total_ms(&mut self, total_ms: u64) {
        self.record.timing.total_ms = total_ms;
    }

    /// Set the exact upstream bytes, if raw capture is enabled.
    pub fn set_upstream_raw(&mut self, raw: &[u8]) {
        if !self.include_raw {
            return;
        }
        let up = self.record.upstream.get_or_insert_with(UpstreamTrace::default);
        up.raw = Some(String::from_utf8_lossy(raw).into_owned());
    }

    /// The number of degradations recorded by lowering.
    pub fn degradation_count(&self) -> u32 {
        self.record.checks.degradation_count
    }

    /// Record the client request re-encode diff check (computed by the pipeline).
    pub fn set_reencode_diff(&mut self, diff: Vec<String>) {
        self.record.checks.reencode_client_request_diff = diff;
    }

    /// Record the plan-§5/§11 client-frame law result, computed by the pipeline/stream from the
    /// actual client-facing output (see [`client_frame_law`]). Wins over the fallback check.
    pub fn set_aggregate_check(&mut self, ok: bool) {
        self.aggregate_check_override = Some(ok);
    }

    /// Mark the request cancelled.
    pub fn set_cancelled(&mut self) {
        self.cancelled = true;
    }

    /// Whether the request was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Finalize: settle the aggregate check, stamp `ts_end`, and return the record.
    ///
    /// The client-frame law (the real plan-§5/§11 invariant) is computed by the pipeline/stream
    /// from the client-facing output and stored via [`set_aggregate_check`](Self::set_aggregate_check);
    /// when it was computed it wins. The upstream re-aggregation fallback below only applies to
    /// paths that never produced client frames (store ops, pre-send errors), where it is vacuously
    /// true, so it never overrides a real result.
    pub fn finish(mut self, ts_end: String) -> TraceRecord {
        self.record.ts_end = ts_end;
        self.record.checks.aggregate_matches_encode_response = self
            .aggregate_check_override
            .unwrap_or_else(|| self.compute_aggregate_check());
        self.record
    }

    fn compute_aggregate_check(&self) -> bool {
        match &self.ir_response {
            None => true,
            Some(expected) => {
                if self.events.is_empty() {
                    return true;
                }
                let mut agg = Aggregator::new();
                for ev in &self.events {
                    agg.push(ev.clone());
                }
                match agg.finish() {
                    Ok(got) => &got == expected,
                    Err(_) => false,
                }
            }
        }
    }

    fn parse_and_redact(&self, raw: &[u8]) -> Option<Value> {
        parse_and_redact_impl(raw, self.redact_over)
    }

    fn redact_value(&self, v: Value) -> Value {
        redact_media(v, self.redact_over)
    }

    /// The compact index entry for this (in-progress or finished) record.
    pub fn index_entry(&self) -> TraceIndexEntry {
        let (provider, upstream_protocol) = self
            .record
            .route
            .as_ref()
            .map(|r| {
                (
                    r.get("provider").and_then(Value::as_str).map(str::to_string),
                    r.get("upstream_protocol").and_then(Value::as_str).map(str::to_string),
                )
            })
            .unwrap_or((None, None));
        let status = self.record.client_response.as_ref().map(|c| c.status).unwrap_or(0);
        TraceIndexEntry {
            trace_id: self.record.trace_id.clone(),
            ts: self.record.ts_start.clone(),
            tag: self.record.tag.clone(),
            client_protocol: self.record.client.protocol.clone(),
            provider,
            upstream_protocol,
            status,
            error_kind: self.record.errors.first().map(|e| e.kind.clone()),
            degradation_count: self.record.checks.degradation_count,
            check_failures: !self.record.checks.aggregate_matches_encode_response,
            cancelled: self.cancelled,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The client-frame law (plan §5/§11)
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The real plan-§5/§11 invariant: the bytes the client actually received, re-decoded with the
/// client protocol's decoder and aggregated, must project to the same structure as
/// `encode_response(aggregate)` re-decoded and aggregated. This exercises the *client* encoder
/// (streaming and non-streaming), unlike the trivial upstream re-aggregation fallback.
///
/// `client_output` is the concatenated content-frame bytes (streaming) or the full response body
/// (non-streaming). `ir_response` is the aggregate of the upstream events. `resp_ctx` supplies the
/// response id / created-at / echo the non-streaming re-encode needs. Any decode/aggregate failure
/// is a law failure (`false`).
pub fn client_frame_law(
    translator: &Translator,
    client_protocol: Protocol,
    client_output: &[u8],
    streaming: bool,
    ir_response: &IrResponse,
    resp_ctx: &EncodeCtx,
) -> bool {
    let mut caps = Capabilities::unknown();
    caps.transport.protocols = Some(vec![client_protocol]);

    // Left: what the client actually got, re-parsed.
    let left_events = if streaming {
        let mut dec = translator.stream_decoder(client_protocol, &caps);
        let mut ev = dec.push(client_output);
        ev.extend(dec.finish());
        ev
    } else {
        match translator.decode_response(client_protocol, client_output, &caps) {
            Ok(ev) => ev,
            Err(_) => return false,
        }
    };
    let Ok(left) = translator.aggregate_stream(left_events) else {
        return false;
    };

    // Right: canonical `encode_response(aggregate)` re-decoded (always as a non-streaming body).
    let mut ctx = resp_ctx.clone();
    ctx.stream = false;
    let body = translator.encode_response(client_protocol, ir_response, &ctx);
    let Ok(right_events) = translator.decode_response(client_protocol, body.as_ref(), &caps) else {
        return false;
    };
    let Ok(right) = translator.aggregate_stream(right_events) else {
        return false;
    };

    project(&left) == project(&right)
}

/// Project an `IrResponse` onto its load-bearing structure (ignoring minted ids / usage / timing):
/// stop reason + per-item kind / text-presence / tool name+args. Mirrors `trace_laws::law_b`.
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
            // Tool arguments arrive as concatenated `partial_json` deltas on the streamed side
            // (whitespace as the provider emitted it) and as a re-serialized compact object on
            // the non-streaming side, so compare them as JSON values, not as strings.
            Item::ToolCall { name, arguments, .. } => {
                let canonical = serde_json::from_str::<Value>(arguments.as_str())
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

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Redaction helpers
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(out.len() * 2);
    for b in out {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Redact secret / key-shaped header values.
pub fn redact_headers(headers: &HeaderMap) -> Map<String, Value> {
    let mut out = Map::new();
    for (name, value) in headers.iter() {
        let n = name.as_str();
        let v = value.to_str().unwrap_or("<non-utf8>");
        let rendered = if is_secret_header(n) || looks_like_key(v) {
            "<redacted>".to_string()
        } else {
            v.to_string()
        };
        out.insert(n.to_string(), Value::String(rendered));
    }
    out
}

fn parse_and_redact_impl(raw: &[u8], redact_over: usize) -> Option<Value> {
    if raw.is_empty() {
        return None;
    }
    match serde_json::from_slice::<Value>(raw) {
        Ok(v) => Some(redact_media(v, redact_over)),
        Err(_) => Some(Value::String(String::from_utf8_lossy(raw).into_owned())),
    }
}

/// Replace any string longer than `over` bytes with `{"$redacted":{sha256,len}}` (plan §11.2).
pub fn redact_media(v: Value, over: usize) -> Value {
    match v {
        Value::String(s) if over > 0 && s.len() > over => {
            json!({"$redacted": {"sha256": sha256_hex(s.as_bytes()), "len": s.len()}})
        }
        Value::Array(a) => Value::Array(a.into_iter().map(|x| redact_media(x, over)).collect()),
        Value::Object(o) => {
            Value::Object(o.into_iter().map(|(k, x)| (k, redact_media(x, over))).collect())
        }
        other => other,
    }
}

/// Base64-encode bytes (used by the store engineer's persistence layer).
pub fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Sinks
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Where finished trace records go. The pipeline submits a record and never blocks on disk.
pub trait TraceSink: Send + Sync {
    /// Submit a finished record.
    fn submit(&self, record: TraceRecord);
}

/// A sink that drops everything (tracing disabled).
#[derive(Debug, Default)]
pub struct NullTraceSink;

impl TraceSink for NullTraceSink {
    fn submit(&self, _record: TraceRecord) {}
}

/// The non-blocking file writer (plan §5): a tokio task with a channel, appending + flushing per
/// record to `traces/{date}.jsonl` and `traces/index.jsonl`.
pub struct FileTraceWriter {
    tx: tokio::sync::mpsc::UnboundedSender<TraceRecord>,
}

impl FileTraceWriter {
    /// Spawn the writer task rooted at `data_dir`, honouring the `[trace].file` template (plan §3;
    /// default `traces/{date}.jsonl`). The template's directory portion (relative to `data_dir`)
    /// holds both the dated file and the compact `index.jsonl`; its file portion is the file name
    /// with `{date}` substituted per record.
    pub fn spawn(data_dir: impl Into<PathBuf>, file_template: &str) -> Arc<FileTraceWriter> {
        let (subdir, name_template) = split_file_template(file_template);
        let dir = data_dir.into().join(subdir);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<TraceRecord>();
        tokio::spawn(async move {
            let _ = std::fs::create_dir_all(&dir);
            while let Some(record) = rx.recv().await {
                write_record(&dir, &name_template, &record);
            }
        });
        Arc::new(FileTraceWriter { tx })
    }
}

/// Split a `[trace].file` template into `(subdir relative to data_dir, file-name template)`. The
/// last `/`-separated segment is the file-name template (carrying `{date}`); everything before it
/// is the subdirectory. A template with no `/` writes into `data_dir` itself.
fn split_file_template(template: &str) -> (String, String) {
    let t = template.trim().trim_end_matches('/');
    match t.rsplit_once('/') {
        Some((dir, name)) if !name.is_empty() => (dir.to_string(), name.to_string()),
        _ if !t.is_empty() => (String::new(), t.to_string()),
        _ => ("traces".to_string(), "{date}.jsonl".to_string()),
    }
}

impl TraceSink for FileTraceWriter {
    fn submit(&self, record: TraceRecord) {
        let _ = self.tx.send(record);
    }
}

fn write_record(dir: &Path, name_template: &str, record: &TraceRecord) {
    use std::io::Write;
    let date = record.ts_start.get(0..10).unwrap_or("unknown");
    let path = dir.join(name_template.replace("{date}", date));
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        if let Ok(line) = serde_json::to_string(record) {
            let _ = writeln!(f, "{line}");
            // `File::flush` is a no-op (no user-space buffer); `sync_data` fsyncs so a crash
            // after this point cannot lose the record (plan §5 durability).
            let _ = f.sync_data();
        }
    }
    // The compact index.
    let entry = index_entry_of(record);
    let ipath = dir.join("index.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&ipath) {
        if let Ok(line) = serde_json::to_string(&entry) {
            let _ = writeln!(f, "{line}");
            let _ = f.sync_data();
        }
    }
}

fn index_entry_of(record: &TraceRecord) -> TraceIndexEntry {
    let (provider, upstream_protocol) = record
        .route
        .as_ref()
        .map(|r| {
            (
                r.get("provider").and_then(Value::as_str).map(str::to_string),
                r.get("upstream_protocol").and_then(Value::as_str).map(str::to_string),
            )
        })
        .unwrap_or((None, None));
    TraceIndexEntry {
        trace_id: record.trace_id.clone(),
        ts: record.ts_start.clone(),
        tag: record.tag.clone(),
        client_protocol: record.client.protocol.clone(),
        provider,
        upstream_protocol,
        status: record.client_response.as_ref().map(|c| c.status).unwrap_or(0),
        error_kind: record.errors.first().map(|e| e.kind.clone()),
        degradation_count: record.checks.degradation_count,
        check_failures: !record.checks.aggregate_matches_encode_response,
        // Derived from the serialized record the same way `tracecli::anomalies` does — the
        // in-memory `TraceBuilder.cancelled` flag is not on `TraceRecord`, so a cancellation is
        // recognised by its recorded error stage. (Fast-triage reads only the index.)
        cancelled: record.errors.iter().any(|e| e.stage == "cancelled"),
    }
}

impl TraceRecord {
    /// Read every record from a `.jsonl` trace file (for the CLI engineer / tests).
    pub fn read_all(path: impl AsRef<Path>) -> std::io::Result<Vec<TraceRecord>> {
        let text = std::fs::read_to_string(path)?;
        Ok(text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect())
    }
}

/// A reader over a `traces/index.jsonl` file, for fast triage (the CLI engineer builds on this).
pub struct TraceIndex {
    /// The parsed index entries, in file order.
    pub entries: Vec<TraceIndexEntry>,
}

impl TraceIndex {
    /// Load an index file.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<TraceIndex> {
        let text = std::fs::read_to_string(path)?;
        let entries = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        Ok(TraceIndex { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_large_media() {
        let big = "A".repeat(200);
        let v = json!({"image": big, "small": "hi"});
        let r = redact_media(v, 64);
        assert!(r["image"]["$redacted"]["len"].as_u64() == Some(200));
        assert_eq!(r["small"], json!("hi"));
    }

    #[test]
    fn redacts_secret_headers() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "Bearer sk-secret".parse().unwrap());
        h.insert("anthropic-version", "2023-06-01".parse().unwrap());
        let out = redact_headers(&h);
        assert_eq!(out["authorization"], json!("<redacted>"));
        assert_eq!(out["anthropic-version"], json!("2023-06-01"));
    }

    #[test]
    fn record_is_schema_stable_and_roundtrips() {
        let b = TraceBuilder::new("tr_1".into(), "2024-01-01T00:00:00.000Z".into(), None, 64, true);
        let rec = b.finish("2024-01-01T00:00:01.000Z".into());
        let s = serde_json::to_string(&rec).unwrap();
        // Fixed field order: `v` then `trace_id` then `ts_start`.
        assert!(s.starts_with("{\"v\":1,\"trace_id\":\"tr_1\",\"ts_start\":"));
        let back: TraceRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(back.trace_id, "tr_1");
        assert!(back.checks.aggregate_matches_encode_response);
    }
}
