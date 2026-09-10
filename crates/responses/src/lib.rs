//! # llm-xlate-responses — the OpenAI **Responses** (`/v1/responses`) codec.
//!
//! Implements [`llm_xlate_core::Codec`] for [`Protocol::OaiResponses`]: it decodes a Responses
//! request body into the shared [`IrRequest`], encodes an [`IrRequest`] back into a Responses
//! request body (capability-gated, every lossy step recorded as a
//! [`llm_xlate_core::Degradation`]), and translates the streaming / non-streaming response and
//! error paths in both directions. Item ids minted when acting as a Responses server follow the
//! plan §5 rule `"{kind}_{response_id}_{index}"`.
//!
//! ## Boundaries (see [`llm_xlate_core::codec`])
//! * **Client-facing** ([`ResponsesCodec::decode_request`], [`ResponsesCodec::stream_encoder`] /
//!   [`ResponsesCodec::encode_response`]): opaque reasoning/compaction blobs are opened with the
//!   [`llm_xlate_core::Sealer`] (`rtr1.` → envelope family; anything else native OpenAI) on the
//!   way in, and sealed into `rtr1.` on the way out when they belong to a foreign family.
//! * **Provider-facing** ([`ResponsesCodec::encode_request`], [`ResponsesCodec::stream_decoder`] /
//!   [`ResponsesCodec::decode_response`]): blobs are native OpenAI; a foreign blob that slips
//!   through is dropped with a degradation (it is `lower()`'s job to have removed it).
//!
//! ## Module map
//! * [`decode`] — request body → IR (lossless; unknown fields land in `ext["responses.*"]`).
//! * [`encode`] — IR → request body + `request_echo` (capability-gated).
//! * [`render`] — shared IR-item → wire-item rendering used by the response, stream, and stored
//!   encoders so all three agree byte-for-byte.
//! * [`response`] — non-streaming `decode_response` / `encode_response`.
//! * `stream_dec` / `stream_enc` — the SSE push state machines.
//! * [`stored`] — the store-support helpers the facade calls for `GET /v1/responses/{id}` and
//!   `…/input_items`.
//! * [`errors`] — error dialect mapping (plan §10).

#![allow(clippy::result_large_err)]
#![allow(rustdoc::private_intra_doc_links)]

mod decode;
mod encode;
mod errors;
mod render;
mod response;
mod stream_dec;
mod stream_enc;
pub mod stored;
mod util;

pub use stored::{encode_input_items, encode_stored_response, StoredStatus, StoredView};
pub use stream_dec::ResponsesStreamDecoder;
pub use stream_enc::ResponsesStreamEncoder;

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::{
    Capabilities, Codec, DecodeCtx, EncodeCtx, EncodedError, EncodedRequest, HeaderMap, IrEvent,
    IrRequest, IrResponse, Protocol, StreamDecoder, StreamEncoder, XlateError,
};

/// The OpenAI Responses codec. A zero-sized type; all state lives in the arguments and the
/// returned push state machines.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResponsesCodec;

impl ResponsesCodec {
    /// Construct the codec.
    pub fn new() -> Self {
        Self
    }
}

impl Codec for ResponsesCodec {
    fn protocol(&self) -> Protocol {
        Protocol::OaiResponses
    }

    fn decode_request(
        &self,
        body: &[u8],
        _hdrs: &HeaderMap,
        ctx: &DecodeCtx,
    ) -> Result<IrRequest, XlateError> {
        decode::decode_request(body, ctx)
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
        Box::new(ResponsesStreamDecoder::new())
    }

    fn stream_encoder(&self, ctx: EncodeCtx) -> Box<dyn StreamEncoder + Send> {
        Box::new(ResponsesStreamEncoder::new(ctx))
    }

    fn decode_response(
        &self,
        body: &[u8],
        caps: &Capabilities,
    ) -> Result<Vec<IrEvent>, XlateError> {
        response::decode_response(body, caps)
    }

    fn encode_response(&self, r: &IrResponse, ctx: &EncodeCtx) -> Bytes {
        response::encode_response(r, ctx)
    }

    fn request_echo(&self, req: &IrRequest, _ctx: &EncodeCtx) -> Map<String, Value> {
        encode::request_echo(req)
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
