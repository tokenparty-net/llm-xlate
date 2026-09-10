//! `llm-xlate-anthropic` — the Anthropic Messages API codec.
//!
//! Implements [`llm_xlate_core::Codec`] for [`Protocol::Anthropic`]: request decode/encode,
//! streaming decode/encode, non-streaming response bridging, and error mapping. The public
//! surface is the thin [`AnthropicCodec`] plus the streaming state machines
//! ([`AnthropicStreamDecoder`], [`AnthropicStreamEncoder`]); every method delegates to the
//! module that owns that boundary.
//!
//! * [`decode`] — Anthropic request JSON -> [`IrRequest`] (client-facing; opens `rtr1.`
//!   envelopes).
//! * [`encode`] — [`IrRequest`] -> Anthropic request bytes + headers (provider-facing;
//!   capability-gated, every lossy step recorded as a [`Degradation`]).
//! * [`stream_dec`] / [`stream_enc`] — SSE <-> [`IrEvent`].
//! * [`response`] — non-streaming body <-> events / [`IrResponse`].
//! * [`errors`] — Anthropic error dialect <-> [`XlateError`].
//!
//! [`Degradation`]: llm_xlate_core::Degradation
//! [`IrRequest`]: llm_xlate_core::IrRequest
//! [`IrEvent`]: llm_xlate_core::IrEvent
//! [`IrResponse`]: llm_xlate_core::IrResponse
//! [`XlateError`]: llm_xlate_core::XlateError

// The codec API mandates `Result<_, XlateError>` signatures; `XlateError` is a large struct,
// so the large-`Err` lint fires everywhere. Allowed crate-wide, matching `llm-xlate-core`.
#![allow(clippy::result_large_err)]
#![allow(rustdoc::private_intra_doc_links)]

mod decode;
mod encode;
mod errors;
mod response;
mod shared;
mod stream_dec;
mod stream_enc;
mod wire;

pub use stream_dec::AnthropicStreamDecoder;
pub use stream_enc::AnthropicStreamEncoder;


use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::{
    Capabilities, Codec, DecodeCtx, EncodeCtx, EncodedError, EncodedRequest, HeaderMap, IrRequest,
    IrResponse, Protocol, StreamDecoder, StreamEncoder, XlateError,
};

/// The Anthropic Messages API codec.
///
/// Zero-sized and stateless; construct one with [`AnthropicCodec`] and share it freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicCodec;

impl Codec for AnthropicCodec {
    fn protocol(&self) -> Protocol {
        Protocol::Anthropic
    }

    fn decode_request(
        &self,
        body: &[u8],
        hdrs: &HeaderMap,
        ctx: &DecodeCtx,
    ) -> Result<IrRequest, XlateError> {
        decode::decode_request(body, hdrs, ctx)
    }

    fn encode_request(
        &self,
        req: &IrRequest,
        caps: &Capabilities,
        ctx: &EncodeCtx,
    ) -> Result<EncodedRequest, XlateError> {
        encode::encode_request(req, caps, ctx)
    }

    fn stream_decoder(&self, _caps: &Capabilities) -> Box<dyn StreamDecoder + Send> {
        Box::new(AnthropicStreamDecoder::new())
    }

    fn stream_encoder(&self, ctx: EncodeCtx) -> Box<dyn StreamEncoder + Send> {
        Box::new(AnthropicStreamEncoder::new(ctx))
    }

    fn decode_response(
        &self,
        body: &[u8],
        caps: &Capabilities,
    ) -> Result<Vec<llm_xlate_core::IrEvent>, XlateError> {
        response::decode_response(body, caps)
    }

    fn encode_response(&self, r: &IrResponse, ctx: &EncodeCtx) -> Bytes {
        response::encode_response(r, ctx)
    }

    /// Anthropic response envelopes (`message_start` / `message_delta`) echo nothing beyond
    /// the model, which the encoder already reads from [`EncodeCtx::client_model`]. So the
    /// request echo is empty.
    fn request_echo(&self, _req: &IrRequest, _ctx: &EncodeCtx) -> Map<String, Value> {
        Map::new()
    }

    fn decode_error(
        &self,
        status: u16,
        body: &[u8],
        hdrs: &HeaderMap,
        caps: &Capabilities,
    ) -> XlateError {
        errors::decode_error(status, body, hdrs, caps)
    }

    fn encode_error(&self, e: &XlateError, streaming: bool, started: bool) -> EncodedError {
        errors::encode_error(e, streaming, started)
    }
}
