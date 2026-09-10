//! Shared test helpers for the `llm-xlate-responses` integration tests.
#![allow(dead_code)]

use llm_xlate_core::{
    Capabilities, Codec, DecodeCtx, EncodeCtx, EncodedRequest, HeaderMap, IrRequest, IrResponse,
    Protocol, ResponseId, Sealer,
};
use llm_xlate_responses::ResponsesCodec;
use serde_json::Value;

/// The codec under test.
pub fn codec() -> ResponsesCodec {
    ResponsesCodec::new()
}

/// The dev envelope key (matches `TranslatorConfig::default`).
pub const KEY: &[u8] = b"llm-xlate-dev-key";

/// A default decode context.
pub fn dctx() -> DecodeCtx {
    DecodeCtx::default()
}

/// Decode a request body string into the IR (panics on error — test helper).
pub fn decode(body: &str) -> IrRequest {
    codec().decode_request(body.as_bytes(), &HeaderMap::new(), &dctx()).unwrap()
}

/// A deterministic encode context (fixed `created_at`, response id `resp_test`).
pub fn ectx() -> EncodeCtx {
    let mut c = EncodeCtx::new(
        Protocol::OaiResponses,
        "gpt-5.4",
        ResponseId::new("resp_test"),
        Sealer::new(KEY),
    );
    c.created_at = 1_726_000_000;
    c
}

/// An encode context whose `request_echo` is built from `req` (as the facade would).
pub fn ectx_echo(req: &IrRequest) -> EncodeCtx {
    let mut c = ectx();
    c.request_echo = codec().request_echo(req, &c);
    c
}

/// Encode a request against `caps` (panics on error — test helper).
pub fn encode(req: &IrRequest, caps: &Capabilities) -> EncodedRequest {
    codec().encode_request(req, caps, &ectx()).unwrap()
}

/// The GPT-5.4 Responses capability preset (the common backend for these tests).
pub fn caps() -> Capabilities {
    llm_xlate_core::caps::preset::gpt5_responses()
}

/// Parse bytes into a JSON value.
pub fn json(b: &[u8]) -> Value {
    serde_json::from_slice(b).unwrap()
}

/// Aggregate an event slice into an [`IrResponse`].
pub fn aggregate(events: &[llm_xlate_core::IrEvent]) -> IrResponse {
    let mut agg = llm_xlate_core::Aggregator::new();
    for e in events {
        agg.push(e.clone());
    }
    agg.finish().unwrap()
}
