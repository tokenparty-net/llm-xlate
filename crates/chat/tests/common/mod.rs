//! Shared helpers for the `llm-xlate-chat` integration tests.
#![allow(dead_code)]

use llm_xlate_core::aggregate::Aggregator;
use llm_xlate_core::caps::preset;
use llm_xlate_core::caps::{ReasoningMode, ReplayMode};
use llm_xlate_core::codec::{Codec, DecodeCtx, EncodeCtx};
use llm_xlate_core::ir::{Effort, Protocol, ResponseId};
use llm_xlate_core::{Capabilities, IrEvent, IrRequest, IrResponse, Sealer};

use llm_xlate_chat::ChatCodec;

/// The dev envelope key shared by decode and encode contexts.
pub const KEY: &[u8] = b"llm-xlate-dev-key";

pub fn codec() -> ChatCodec {
    ChatCodec
}

pub fn sealer() -> Sealer {
    Sealer::new(KEY)
}

pub fn decode_ctx() -> DecodeCtx {
    DecodeCtx::default()
}

/// A response/encode context for the Chat client.
pub fn encode_ctx(model: &str, include_usage: bool) -> EncodeCtx {
    let mut ctx = EncodeCtx::new(Protocol::OaiChat, model, ResponseId::new("chatcmpl-test"), sealer());
    ctx.created_at = 1_700_000_000;
    ctx.include_usage = include_usage;
    ctx.stream = include_usage;
    ctx
}

pub fn decode_request(body: &str) -> IrRequest {
    codec().decode_request(body.as_bytes(), &Default::default(), &decode_ctx()).expect("decode_request")
}

pub fn try_decode_request(body: &str) -> Result<IrRequest, llm_xlate_core::XlateError> {
    codec().decode_request(body.as_bytes(), &Default::default(), &decode_ctx())
}

pub fn encode_request(ir: &IrRequest, caps: &Capabilities) -> llm_xlate_core::EncodedRequest {
    codec().encode_request(ir, caps, &encode_ctx("gpt-4o", false)).expect("encode_request")
}

pub fn try_encode_request(
    ir: &IrRequest,
    caps: &Capabilities,
) -> Result<llm_xlate_core::EncodedRequest, llm_xlate_core::XlateError> {
    codec().encode_request(ir, caps, &encode_ctx("gpt-4o", false))
}

pub fn encode_request_str(ir: &IrRequest, caps: &Capabilities) -> String {
    String::from_utf8(encode_request(ir, caps).body.to_vec()).unwrap()
}

/// gpt-4o preset (no reasoning, tools/strict/media/sampling all supported).
pub fn gpt4o() -> Capabilities {
    preset::gpt4o()
}

/// gpt-5.4 accessed as a Chat backend (`tools_with_reasoning = false`).
pub fn gpt5_chat() -> Capabilities {
    preset::gpt5_chat()
}

/// Conservative generic openai-compatible backend (most fields Unknown).
pub fn openai_compatible() -> Capabilities {
    preset::openai_compatible()
}

/// A permissive "everything supported" Chat target used for lossless round-trip tests: gpt-4o
/// plus reasoning (effort + `TextField` replay), verbosity, and audio input.
pub fn full_caps() -> Capabilities {
    let mut c = preset::gpt4o();
    c.reasoning.mode = Some(ReasoningMode::EffortOnly);
    c.reasoning.effort_levels =
        Some(vec![Effort::Minimal, Effort::Low, Effort::Medium, Effort::High, Effort::XHigh]);
    c.reasoning.replay = Some(ReplayMode::TextField);
    c.reasoning.tools_with_reasoning = llm_xlate_core::Tri::Yes;
    c.output.verbosity = llm_xlate_core::Tri::Yes;
    c.media.audio.sources = Some(vec![llm_xlate_core::caps::MediaSourceKind::Base64]);
    c
}

/// Decode a full SSE stream (single push) into events.
pub fn decode_stream(sse: &str, caps: &Capabilities) -> Vec<IrEvent> {
    let mut dec = codec().stream_decoder(caps);
    let mut events = dec.push(sse.as_bytes());
    events.extend(dec.finish());
    events
}

/// Decode an SSE stream split at every byte boundary.
pub fn decode_stream_split(sse: &str, caps: &Capabilities) -> Vec<IrEvent> {
    let mut dec = codec().stream_decoder(caps);
    let mut events = Vec::new();
    for b in sse.as_bytes() {
        events.extend(dec.push(&[*b]));
    }
    events.extend(dec.finish());
    events
}

/// Encode an event sequence through the Chat stream encoder into a single SSE string.
pub fn encode_stream(events: Vec<IrEvent>, ctx: EncodeCtx) -> String {
    let mut enc = codec().stream_encoder(ctx);
    let mut buf = Vec::new();
    for ev in events {
        for frame in enc.push(ev) {
            buf.extend_from_slice(&frame);
        }
    }
    for frame in enc.finish() {
        buf.extend_from_slice(&frame);
    }
    String::from_utf8(buf).unwrap()
}

/// Decode a non-streaming response body into events.
pub fn decode_response(body: &str, caps: &Capabilities) -> Vec<IrEvent> {
    codec().decode_response(body.as_bytes(), caps).expect("decode_response")
}

/// Encode an aggregated response into `chat.completion` bytes with the given context.
pub fn encode_response(r: &IrResponse, ctx: &EncodeCtx) -> String {
    String::from_utf8(codec().encode_response(r, ctx).to_vec()).unwrap()
}

/// Aggregate an event stream into an [`IrResponse`].
pub fn aggregate(events: Vec<IrEvent>) -> IrResponse {
    let mut agg = Aggregator::new();
    for ev in events {
        agg.push(ev);
    }
    agg.finish().expect("aggregate")
}

/// The SSE `data:` payloads of a stream string, in order (drops comments/blank).
pub fn sse_data_lines(s: &str) -> Vec<String> {
    s.lines()
        .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
        .collect()
}
