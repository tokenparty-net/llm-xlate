//! Client SSE plumbing (plan §4.9, §8): the client response body as a frame stream, a keepalive
//! ticker while the upstream is quiet, cancellation on client disconnect, and the two mismatch
//! paths (non-streaming upstream → streaming client via `synthesize`, and the reverse handled by
//! [`crate::pipeline`]).
//!
//! ## Why concrete codec enums instead of `Box<dyn StreamEncoder>`
//! `axum`'s response body must be `Send`, but `Box<dyn StreamEncoder>` erases the `Send` marker of
//! the concrete codec state machines. Rather than reach for `unsafe`, we hold the concrete
//! (`Send`) encoder/decoder types in [`AnyEncoder`] / [`AnyDecoder`] enums, which the whole body
//! generator can then carry across `.await` points.

use axum::body::Body;
use axum::http::header::CONTENT_TYPE;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;

use llm_xlate::{
    AnthropicStreamDecoder, AnthropicStreamEncoder, ChatStreamDecoder, ChatStreamEncoder,
    ResponsesStreamDecoder, ResponsesStreamEncoder, Translator,
};
use llm_xlate_core::aggregate::Aggregator;
use llm_xlate_core::codec::EncodeCtx;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrEvent, Protocol};

use crate::pipeline::{base_response_headers, build_response, RespondCtx};
use crate::trace::TraceBuilder;
use crate::upstream::{UpstreamBody, UpstreamResponse};
use crate::AppState;

/// A `Send` streaming encoder over the three concrete codec state machines.
pub enum AnyEncoder {
    /// OpenAI Chat.
    Chat(ChatStreamEncoder),
    /// OpenAI Responses.
    Responses(ResponsesStreamEncoder),
    /// Anthropic Messages.
    Anthropic(AnthropicStreamEncoder),
}

impl AnyEncoder {
    /// Construct the client encoder for `protocol`.
    pub fn new(protocol: Protocol, ctx: EncodeCtx) -> Self {
        match protocol {
            Protocol::OaiChat => AnyEncoder::Chat(ChatStreamEncoder::new(ctx)),
            Protocol::OaiResponses => AnyEncoder::Responses(ResponsesStreamEncoder::new(ctx)),
            Protocol::Anthropic => AnyEncoder::Anthropic(AnthropicStreamEncoder::new(ctx)),
        }
    }
    fn push(&mut self, ev: IrEvent) -> Vec<Bytes> {
        use llm_xlate_core::codec::StreamEncoder;
        match self {
            AnyEncoder::Chat(e) => e.push(ev),
            AnyEncoder::Responses(e) => e.push(ev),
            AnyEncoder::Anthropic(e) => e.push(ev),
        }
    }
    fn finish(&mut self) -> Vec<Bytes> {
        use llm_xlate_core::codec::StreamEncoder;
        match self {
            AnyEncoder::Chat(e) => e.finish(),
            AnyEncoder::Responses(e) => e.finish(),
            AnyEncoder::Anthropic(e) => e.finish(),
        }
    }
    fn keepalive(&mut self) -> Option<Bytes> {
        use llm_xlate_core::codec::StreamEncoder;
        match self {
            AnyEncoder::Chat(e) => e.keepalive(),
            AnyEncoder::Responses(e) => e.keepalive(),
            AnyEncoder::Anthropic(e) => e.keepalive(),
        }
    }
}

/// A `Send` streaming decoder over the three concrete codec state machines.
pub enum AnyDecoder {
    /// OpenAI Chat.
    Chat(ChatStreamDecoder),
    /// OpenAI Responses.
    Responses(ResponsesStreamDecoder),
    /// Anthropic Messages.
    Anthropic(AnthropicStreamDecoder),
}

impl AnyDecoder {
    /// Construct the upstream decoder for `protocol`.
    pub fn new(protocol: Protocol) -> Self {
        match protocol {
            Protocol::OaiChat => AnyDecoder::Chat(ChatStreamDecoder::new()),
            Protocol::OaiResponses => AnyDecoder::Responses(ResponsesStreamDecoder::new()),
            Protocol::Anthropic => AnyDecoder::Anthropic(AnthropicStreamDecoder::new()),
        }
    }
    fn push(&mut self, bytes: &[u8]) -> Vec<IrEvent> {
        use llm_xlate_core::codec::StreamDecoder;
        match self {
            AnyDecoder::Chat(d) => d.push(bytes),
            AnyDecoder::Responses(d) => d.push(bytes),
            AnyDecoder::Anthropic(d) => d.push(bytes),
        }
    }
    fn finish(&mut self) -> Vec<IrEvent> {
        use llm_xlate_core::codec::StreamDecoder;
        match self {
            AnyDecoder::Chat(d) => d.finish(),
            AnyDecoder::Responses(d) => d.finish(),
            AnyDecoder::Anthropic(d) => d.finish(),
        }
    }
}

/// Everything the stream needs to persist/sidecar-learn the aggregated response once it completes
/// (a `store:true` streaming Responses turn must be as durable as a non-streaming one — plan §4.4).
struct StreamPersist {
    state: std::sync::Arc<AppState>,
    route: crate::route::Route,
    response_id: llm_xlate_core::ir::ResponseId,
    created_at: u64,
    store_ir: Option<llm_xlate_core::ir::IrRequest>,
}

/// Encapsulates the client-encoder + aggregator state that survives across `.await` points.
struct Encoding {
    encoder: AnyEncoder,
    agg: Aggregator,
    translator: Translator,
    client_protocol: Protocol,
    resp_ctx: EncodeCtx,
    started: bool,
    error_seen: bool,
    /// Concatenated non-keepalive client frame bytes, for the plan-§5/§11 client-frame law.
    content: Vec<u8>,
    /// Persist/sidecar context (always present on the real response path).
    persist: Option<StreamPersist>,
}

impl Encoding {
    /// Process one IR event: record it, aggregate it, and return the client frames to send. A
    /// mid-stream [`IrEvent::Error`] is encoded as a client error (with `started`) and stops the
    /// stream.
    fn on_event(&mut self, ev: IrEvent, builder: &mut TraceBuilder, now: u64) -> Vec<Bytes> {
        if let IrEvent::Error(err) = &ev {
            let enc = self.translator.encode_error(self.client_protocol, err, true, self.started);
            builder.push_error("stream", err.kind.slug(), enc.status, &err.message, None);
            for f in &enc.frames {
                builder.push_frame(now, false, f);
            }
            self.error_seen = true;
            return enc.frames;
        }
        builder.push_event(now, &ev);
        self.agg.push(ev.clone());
        let frames = self.encoder.push(ev);
        for f in &frames {
            builder.push_frame(now, false, f);
            self.content.extend_from_slice(f);
            self.started = true;
        }
        frames
    }

    /// Flush the client encoder at end of stream.
    fn on_finish(&mut self, builder: &mut TraceBuilder, now: u64) -> Vec<Bytes> {
        let frames = self.encoder.finish();
        for f in &frames {
            builder.push_frame(now, false, f);
            self.content.extend_from_slice(f);
        }
        frames
    }

    /// Finalize the aggregate into an `ir_response` (unless a mid-stream error stopped us), compute
    /// the plan-§5/§11 client-frame law from the emitted frames, and persist / sidecar-learn.
    fn finalize_aggregate(self, builder: &mut TraceBuilder) {
        if self.error_seen {
            return;
        }
        if let Ok(resp) = self.agg.finish() {
            builder.set_ir_response(&resp);
            if !self.content.is_empty() {
                let law = crate::trace::client_frame_law(
                    &self.translator,
                    self.client_protocol,
                    &self.content,
                    true,
                    &resp,
                    &self.resp_ctx,
                );
                builder.set_aggregate_check(law);
            }
            if let Some(p) = &self.persist {
                crate::pipeline::persist_aggregate(
                    &p.state.deps,
                    crate::pipeline::PersistInputs {
                        route: &p.route,
                        response_id: &p.response_id,
                        created_at: p.created_at,
                        resp_ctx: &self.resp_ctx,
                        store_ir: p.store_ir.as_ref(),
                        ir_response: &resp,
                    },
                    builder,
                );
            }
        }
    }
}

/// A drop-guard that owns the [`TraceBuilder`] during streaming and submits the record when the
/// stream completes normally ([`TraceGuard::complete`]) or when the client disconnects (the
/// generator future is dropped, running [`Drop`]).
struct TraceGuard {
    state: std::sync::Arc<AppState>,
    builder: Option<TraceBuilder>,
    start: std::time::Instant,
}

impl TraceGuard {
    fn builder_mut(&mut self) -> &mut TraceBuilder {
        self.builder.as_mut().expect("builder present until complete/drop")
    }

    fn complete(mut self) {
        if let Some(mut b) = self.builder.take() {
            let total = self.start.elapsed().as_millis() as u64;
            b.set_total_ms(total);
            self.submit(b);
        }
    }

    fn submit(&self, mut b: TraceBuilder) {
        if self.state.trace_enabled {
            let ts_end = self.state.deps.clock.now_rfc3339();
            let total = self.start.elapsed().as_millis() as u64;
            b.set_total_ms(total);
            let record = b.finish(ts_end);
            self.state.deps.trace_sink.submit(record);
        }
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        if let Some(mut b) = self.builder.take() {
            // Reached only when the stream was dropped before normal completion → cancellation.
            b.set_cancelled();
            b.push_error("cancelled", "cancelled", 499, "client disconnected mid-stream", None);
            self.submit(b);
        }
    }
}

/// Build the streaming client [`Response`] (plan §4.9). Consumes the [`RespondCtx`] and the
/// upstream response; the returned body owns the trace builder and submits the record when it
/// finishes or is dropped.
pub(crate) fn streaming_response(
    mut c: RespondCtx,
    up: UpstreamResponse,
    upstream_request_id: Option<String>,
) -> Response {
    let degraded = c.builder.degradation_count() > 0;
    let mut headers = base_response_headers(&c.trace_id, degraded, upstream_request_id.as_deref());
    headers.insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
    c.builder.start_client_response_stream(200, &headers);

    // Record the upstream request-id passthrough on a possible mid-stream error too.
    let _ = &upstream_request_id;

    let state = c.state.clone();
    let translator = state.deps.translator.clone();
    let client_protocol = c.client_protocol;
    let upstream_protocol = c.upstream_protocol;
    let caps = c.caps.clone();
    let resp_ctx = c.resp_ctx.clone();
    let keepalive = state.keepalive;
    let start = c.start;

    let persist = StreamPersist {
        state: state.clone(),
        route: c.route.clone(),
        response_id: c.response_id.clone(),
        created_at: c.created_at,
        store_ir: c.store_ir.take(),
    };

    let mut enc = Encoding {
        encoder: AnyEncoder::new(client_protocol, resp_ctx.clone()),
        agg: Aggregator::new(),
        translator: translator.clone(),
        client_protocol,
        resp_ctx,
        started: false,
        error_seen: false,
        content: Vec::new(),
        persist: Some(persist),
    };

    let mut guard = TraceGuard { state: state.clone(), builder: Some(c.builder), start };

    let body_stream = async_stream::stream! {
        let now0 = start.elapsed().as_millis() as u64;
        match up.body {
            UpstreamBody::Full(bytes) => {
                // Non-streaming upstream → streaming client (synthesize, plan §8).
                guard.builder_mut().set_upstream_raw(&bytes);
                match translator.decode_response(upstream_protocol, &bytes, &caps) {
                    Ok(events) => {
                        for ev in events {
                            let now = start.elapsed().as_millis() as u64;
                            let frames = enc.on_event(ev, guard.builder_mut(), now);
                            for f in frames {
                                yield Ok::<Bytes, std::io::Error>(f);
                            }
                            if enc.error_seen { break; }
                        }
                        if !enc.error_seen {
                            let now = start.elapsed().as_millis() as u64;
                            for f in enc.on_finish(guard.builder_mut(), now) {
                                yield Ok::<Bytes, std::io::Error>(f);
                            }
                        }
                    }
                    Err(err) => {
                        emit_prestart_error(&translator, client_protocol, &err, guard.builder_mut());
                        let ee = translator.encode_error(client_protocol, &err, true, false);
                        for f in ee.frames {
                            yield Ok::<Bytes, std::io::Error>(f);
                        }
                    }
                }
                enc.finalize_aggregate(guard.builder_mut());
            }
            UpstreamBody::Stream(mut s) => {
                let mut dec = AnyDecoder::new(upstream_protocol);
                let mut raw: Vec<u8> = Vec::new();
                let mut interval = tokio::time::interval(keepalive);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                interval.tick().await; // consume the immediate first tick

                let _ = now0;
                'pump: loop {
                    tokio::select! {
                        biased;
                        item = s.next() => {
                            match item {
                                Some(Ok(chunk)) => {
                                    raw.extend_from_slice(&chunk);
                                    let events = dec.push(&chunk);
                                    for ev in events {
                                        let now = start.elapsed().as_millis() as u64;
                                        let frames = enc.on_event(ev, guard.builder_mut(), now);
                                        for f in frames {
                                            yield Ok::<Bytes, std::io::Error>(f);
                                        }
                                        if enc.error_seen { break 'pump; }
                                    }
                                }
                                Some(Err(transport)) => {
                                    let err = XlateError::new(
                                        llm_xlate_core::error::ErrorKind::ServerError,
                                        format!("upstream stream error: {transport}"),
                                    );
                                    let now = start.elapsed().as_millis() as u64;
                                    let frames = enc.on_event(IrEvent::Error(err), guard.builder_mut(), now);
                                    for f in frames {
                                        yield Ok::<Bytes, std::io::Error>(f);
                                    }
                                    break 'pump;
                                }
                                None => {
                                    let tail = dec.finish();
                                    for ev in tail {
                                        let now = start.elapsed().as_millis() as u64;
                                        let frames = enc.on_event(ev, guard.builder_mut(), now);
                                        for f in frames {
                                            yield Ok::<Bytes, std::io::Error>(f);
                                        }
                                        if enc.error_seen { break; }
                                    }
                                    if !enc.error_seen {
                                        let now = start.elapsed().as_millis() as u64;
                                        for f in enc.on_finish(guard.builder_mut(), now) {
                                            yield Ok::<Bytes, std::io::Error>(f);
                                        }
                                    }
                                    break 'pump;
                                }
                            }
                        }
                        _ = interval.tick() => {
                            if let Some(k) = enc.encoder.keepalive() {
                                let now = start.elapsed().as_millis() as u64;
                                guard.builder_mut().push_frame(now, true, &k);
                                yield Ok::<Bytes, std::io::Error>(k);
                            }
                        }
                    }
                }
                guard.builder_mut().set_upstream_raw(&raw);
                enc.finalize_aggregate(guard.builder_mut());
            }
        }
        guard.complete();
    };

    build_response(200, headers, Body::from_stream(body_stream))
}

/// Record a decode-response error that occurred before any client frame was sent.
fn emit_prestart_error(
    translator: &Translator,
    client_protocol: Protocol,
    err: &XlateError,
    builder: &mut TraceBuilder,
) {
    let enc = translator.encode_error(client_protocol, err, true, false);
    builder.push_error("decode_response", err.kind.slug(), enc.status, &err.message, None);
}
