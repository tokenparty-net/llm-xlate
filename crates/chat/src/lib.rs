//! # llm-xlate-chat
//!
//! The OpenAI **Chat Completions** codec for `llm-xlate`. Implements
//! [`llm_xlate_core::Codec`] for [`Protocol::OaiChat`] (family
//! [`ProviderFamily::OpenAI`](llm_xlate_core::ir::ProviderFamily::OpenAI)), owning the wire
//! shape of the `chat.completion` request/response/stream dialect (plus the de-facto gateway
//! extensions such as `reasoning_content` / `reasoning_details`, legacy `functions` /
//! `function_call`, and the `chat.*` ext namespace for verbatim passthrough).
//!
//! ## Modules
//! * [`decode`] — `decode_request`: a Chat request body → [`IrRequest`]
//!   (client-facing, lossless; unknown fields captured under `ext["chat.<field>"]`).
//! * [`encode`] — `encode_request`: an `IrRequest` → Chat request bytes (provider-facing;
//!   capability-gated, every drop a [`Degradation`](llm_xlate_core::Degradation)).
//! * [`stream_dec`] / [`stream_enc`] — the streaming decoder / encoder ([`ChatStreamDecoder`],
//!   [`ChatStreamEncoder`]).
//! * [`response`] — non-streaming `decode_response` / `encode_response`.
//! * [`errors`] — error-body ⇄ [`XlateError`] mapping.
//! * [`common`] — the `chat.` namespace, stop-reason / usage mapping, and small helpers.
//!
//! ## Envelope boundary
//! `decode_request` opens `rtr1.` reasoning envelopes with the client-facing sealer (native
//! family = OpenAI for bare blobs). `stream_encoder` / `encode_response` seal every opaque
//! reasoning blob into an `rtr1.` envelope carried in `reasoning_details`. The provider-facing
//! `encode_request` / `stream_decoder` / `decode_response` deal in native OpenAI blobs.

#![allow(clippy::result_large_err)]

pub mod common;
pub mod decode;
pub mod encode;
pub mod errors;
pub mod response;
pub mod stream_dec;
pub mod stream_enc;
pub mod wire;

pub use stream_dec::ChatStreamDecoder;
pub use stream_enc::ChatStreamEncoder;

use llm_xlate_core::codec::{Codec, DecodeCtx, EncodeCtx, EncodedError, EncodedRequest, HeaderMap};
use llm_xlate_core::ir::{IrRequest, IrResponse, Protocol};
use llm_xlate_core::{Capabilities, IrEvent, StreamDecoder, StreamEncoder, XlateError};

/// The OpenAI Chat Completions codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatCodec;

impl Codec for ChatCodec {
    fn protocol(&self) -> Protocol {
        Protocol::OaiChat
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
        Box::new(ChatStreamDecoder::new())
    }

    fn stream_encoder(&self, ctx: EncodeCtx) -> Box<dyn StreamEncoder + Send> {
        Box::new(ChatStreamEncoder::new(ctx))
    }

    fn decode_response(
        &self,
        body: &[u8],
        caps: &Capabilities,
    ) -> Result<Vec<IrEvent>, XlateError> {
        response::decode_response(body, caps)
    }

    fn encode_response(&self, r: &IrResponse, ctx: &EncodeCtx) -> bytes::Bytes {
        response::encode_response(r, ctx)
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
