//! The [`Translator`] façade (plan §4).
//!
//! [`Translator`] is the single object the router holds. It wires the three protocol codec
//! crates ([`llm_xlate_chat`], [`llm_xlate_responses`], [`llm_xlate_anthropic`]) onto the
//! sans-IO primitives of [`llm_xlate_core`] and this crate's [`lower`](mod@crate::lower) /
//! [`requirements`](mod@crate::requirements) / [`store`](crate::store) building blocks, exposing the
//! whole two-phase request path as a set of pure methods:
//!
//! ```text
//!  decode_request → requirements → [router resolves I/O] → materialize_chain → lower → encode_request
//!  stream_decoder / decode_response  → aggregate / stream_encoder / encode_response
//! ```
//!
//! Every method is a thin, deterministic delegation onto a codec or a lowering pass; the
//! façade owns the *registry* ([`codec_for`](Translator::codec_for)), the [`EncodeCtx`]
//! construction policy ([`encode_ctx`](Translator::encode_ctx)), the stored-response mapping,
//! and the two streaming *mismatch* helpers of plan §8
//! ([`synthesize_stream`](Translator::synthesize_stream) and
//! [`aggregate_stream`](Translator::aggregate_stream)).

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::aggregate::Aggregator;
use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::codec::{
    Codec, DecodeCtx, EncodeCtx, EncodedError, EncodedRequest, StreamDecoder, StreamEncoder,
    TranslatorConfig,
};
use llm_xlate_core::envelope::Sealer;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrEvent, IrRequest, IrResponse, Protocol, ResponseId};
use llm_xlate_core::HeaderMap;

use llm_xlate_chat::ChatCodec;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_responses::{stored as resp_stored, ResponsesCodec};

use crate::lower::{lower, Lowered};
use crate::requirements::{requirements, Requirements, Resolutions};
use crate::store::{
    materialize_chain, to_stored, BackendBinding, StoredResponse, StoredStatus,
};

/// The cross-API translation façade (plan §4).
///
/// Construct once per router with [`Translator::new`]; it is cheap to clone (the codecs are
/// zero-sized and the config is a small owned struct). All methods are pure: identical
/// `(inputs, capabilities, config)` always produce identical output.
#[derive(Debug, Clone)]
pub struct Translator {
    config: TranslatorConfig,
    chat: ChatCodec,
    responses: ResponsesCodec,
    anthropic: AnthropicCodec,
}

impl Translator {
    /// Build a translator from a [`TranslatorConfig`] (envelope key, mid-instruction fallback,
    /// foreign-provider-tool policy, unresolved-reasoning policy, wrap version).
    pub fn new(config: TranslatorConfig) -> Self {
        Self {
            config,
            chat: ChatCodec,
            responses: ResponsesCodec,
            anthropic: AnthropicCodec,
        }
    }

    /// The active configuration.
    pub fn config(&self) -> &TranslatorConfig {
        &self.config
    }

    /// The client-facing [`Sealer`] derived from the configured envelope key.
    pub fn sealer(&self) -> Sealer {
        Sealer::new(&self.config.envelope_key)
    }

    /// The codec registry: the [`Codec`] implementation for a protocol.
    pub fn codec_for(&self, p: Protocol) -> &dyn Codec {
        match p {
            Protocol::OaiChat => &self.chat,
            Protocol::OaiResponses => &self.responses,
            Protocol::Anthropic => &self.anthropic,
        }
    }

    // ---- request path (pure) -------------------------------------------------------------

    /// Decode a client request body into the IR (plan §4). Opens any `rtr1.` reasoning
    /// envelopes with the configured sealer.
    pub fn decode_request(
        &self,
        p: Protocol,
        body: &[u8],
        hdrs: &HeaderMap,
    ) -> Result<IrRequest, XlateError> {
        let ctx = DecodeCtx::from_config(self.config.clone());
        self.codec_for(p).decode_request(body, hdrs, &ctx)
    }

    /// The read-only requirements pre-pass (plan §4): foreign file ids to bridge, missing
    /// reasoning blobs to fetch, and a `previous_response_id` chain to materialize.
    pub fn requirements(
        &self,
        req: &IrRequest,
        caps: &Capabilities,
        target: Protocol,
    ) -> Requirements {
        requirements(req, caps, target)
    }

    /// Run the deterministic lowering passes (plan §7) for `target`.
    pub fn lower(
        &self,
        req: IrRequest,
        caps: &Capabilities,
        target: Protocol,
        res: &Resolutions,
    ) -> Result<Lowered, XlateError> {
        lower(req, caps, target, res, &self.config)
    }

    /// Encode a lowered request for the target protocol (plan §4). `ctx_seed` is the
    /// [`EncodeCtx`] the response encoders will use; `encode_request` may adjust it (e.g.
    /// `include_usage`) and returns it on [`EncodedRequest::ctx`].
    pub fn encode_request(
        &self,
        p: Protocol,
        req: &IrRequest,
        caps: &Capabilities,
        ctx_seed: &EncodeCtx,
    ) -> Result<EncodedRequest, XlateError> {
        self.codec_for(p).encode_request(req, caps, ctx_seed)
    }

    /// Construct the [`EncodeCtx`] the response path needs from a **decoded client request**
    /// (plan §4, §8). Fills `request_echo` (via the *client* codec's
    /// [`Codec::request_echo`]), `include_usage` (Chat `stream_options.include_usage`),
    /// `expose`, `store`, `include`, and `stream` from the request, and stamps the router-
    /// supplied `response_id` and `created_at`.
    pub fn encode_ctx(
        &self,
        client_protocol: Protocol,
        req: &IrRequest,
        response_id: ResponseId,
        created_at: u64,
    ) -> EncodeCtx {
        let sealer = self.sealer();
        let mut ctx = EncodeCtx::new(
            client_protocol,
            req.model.client_name.clone(),
            response_id,
            sealer,
        );
        ctx.expose = req.reasoning.expose.clone();
        ctx.store = req.state.store;
        ctx.include = req.state.include.clone();
        ctx.stream = req.stream;
        ctx.include_usage = include_usage_requested(req);
        ctx.created_at = created_at;
        // The client codec shapes its own dialect echo from the pre-lowering request.
        ctx.request_echo = self.codec_for(client_protocol).request_echo(req, &ctx);
        ctx
    }

    /// Convenience: `decode → requirements → lower → encode` in one call (plan §4). Returns the
    /// encoded upstream request (with lowering + wiring degradations merged into
    /// [`EncodedRequest::degradations`]) and the [`Requirements`] computed from the decoded
    /// request.
    ///
    /// The router normally calls the steps *separately* because the requirements must be
    /// resolved asynchronously (sidecar lookups, file bridging, chain load) between
    /// [`requirements`](Translator::requirements) and [`lower`](Translator::lower); this helper
    /// is for the common case where nothing needs resolving (an empty [`Resolutions`]), e.g.
    /// tests and same-family passthrough.
    pub fn translate_request(
        &self,
        client_p: Protocol,
        body: &[u8],
        hdrs: &HeaderMap,
        target_p: Protocol,
        caps: &Capabilities,
        res: &Resolutions,
    ) -> Result<(EncodedRequest, Requirements), XlateError> {
        let ir = self.decode_request(client_p, body, hdrs)?;
        let reqs = self.requirements(&ir, caps, target_p);
        // Build the response ctx from the pre-lowering client request.
        let ctx = self.encode_ctx(client_p, &ir, ResponseId::new(""), 0);
        let low = self.lower(ir, caps, target_p, res)?;
        let mut enc = self.encode_request(target_p, &low.req, caps, &ctx)?;
        // Merge lowering degradations (first) ahead of the codec's wiring degradations.
        let mut merged = low.degradations;
        merged.extend(std::mem::take(&mut enc.degradations));
        enc.degradations = merged;
        Ok((enc, reqs))
    }

    // ---- response path (push state machines) --------------------------------------------

    /// A provider → IR streaming decoder for `p`.
    pub fn stream_decoder(&self, p: Protocol, caps: &Capabilities) -> Box<dyn StreamDecoder + Send> {
        self.codec_for(p).stream_decoder(caps)
    }

    /// An IR → client streaming encoder for `p`, driven by `ctx`.
    pub fn stream_encoder(&self, p: Protocol, ctx: EncodeCtx) -> Box<dyn StreamEncoder + Send> {
        self.codec_for(p).stream_encoder(ctx)
    }

    /// Decode a non-streaming provider response body into IR events (plan §4).
    pub fn decode_response(
        &self,
        p: Protocol,
        body: &[u8],
        caps: &Capabilities,
    ) -> Result<Vec<IrEvent>, XlateError> {
        self.codec_for(p).decode_response(body, caps)
    }

    /// Encode a full [`IrResponse`] as a non-streaming client response body (plan §4).
    pub fn encode_response(&self, p: Protocol, r: &IrResponse, ctx: &EncodeCtx) -> Bytes {
        self.codec_for(p).encode_response(r, ctx)
    }

    /// Decode a provider error (status + body + headers) into an [`XlateError`] (plan §10).
    pub fn decode_error(
        &self,
        p: Protocol,
        status: u16,
        body: &[u8],
        hdrs: &HeaderMap,
        caps: &Capabilities,
    ) -> XlateError {
        self.codec_for(p).decode_error(status, body, hdrs, caps)
    }

    /// Encode an [`XlateError`] as a client error response (plan §10). `streaming` selects the
    /// SSE frame form; `started` says whether a `Start` was already emitted (Responses:
    /// `response.failed` vs a bare `error` event).
    pub fn encode_error(
        &self,
        p: Protocol,
        e: &XlateError,
        streaming: bool,
        started: bool,
    ) -> EncodedError {
        self.codec_for(p).encode_error(e, streaming, started)
    }

    // ---- streaming mismatch helpers (plan §8 last paragraph) ----------------------------

    /// Bridge a **non-streaming** provider response to a **streaming** client (plan §8): decode
    /// the provider body into IR events, then run them through the client stream encoder in a
    /// single burst. `p` is the *provider* protocol of `body`; `ctx.client_protocol` selects
    /// the client encoder.
    pub fn synthesize_stream(
        &self,
        p: Protocol,
        body: &[u8],
        caps: &Capabilities,
        ctx: EncodeCtx,
    ) -> Result<Vec<Bytes>, XlateError> {
        let events = self.decode_response(p, body, caps)?;
        let client_p = ctx.client_protocol;
        let mut enc = self.stream_encoder(client_p, ctx);
        let mut out = Vec::new();
        for ev in events {
            out.extend(enc.push(ev));
        }
        out.extend(enc.finish());
        Ok(out)
    }

    /// Bridge a **streaming** provider to a **non-streaming** client (plan §8): fold a decoded
    /// IR event sequence into a single [`IrResponse`] via the [`Aggregator`]. Returns the
    /// aggregator's error if the stream carried an [`IrEvent::Error`].
    pub fn aggregate_stream(
        &self,
        events: impl IntoIterator<Item = IrEvent>,
    ) -> Result<IrResponse, XlateError> {
        let mut agg = Aggregator::new();
        for ev in events {
            agg.push(ev);
        }
        agg.finish()
    }

    // ---- store support (pure; router persists) ------------------------------------------

    /// Replay a stored `previous_response_id` chain plus a new request into one stateless
    /// [`IrRequest`] (plan §9).
    pub fn materialize_chain(
        &self,
        chain: &[StoredResponse],
        new: IrRequest,
    ) -> Result<IrRequest, XlateError> {
        materialize_chain(chain, new)
    }

    /// Build a [`StoredResponse`] persistence record from a completed turn (plan §9).
    pub fn to_stored(
        &self,
        req: &IrRequest,
        out: &IrResponse,
        binding: BackendBinding,
        id: ResponseId,
        created_at: u64,
        request_echo: Map<String, Value>,
    ) -> StoredResponse {
        to_stored(req, out, binding, id, created_at, request_echo)
    }

    /// Encode a stored response as a `GET /v1/responses/{id}` body (plan §9). **Responses
    /// only** — other client protocols have no stored-response endpoint and return
    /// [`ErrorKind::Unsupported`](llm_xlate_core::error::ErrorKind::Unsupported).
    pub fn encode_stored_response(
        &self,
        p: Protocol,
        s: &StoredResponse,
    ) -> Result<Bytes, XlateError> {
        if p != Protocol::OaiResponses {
            return Err(XlateError::unsupported(
                "protocol",
                "stored-response retrieval is a Responses-only endpoint",
            ));
        }
        let view = stored_view(s);
        Ok(resp_stored::encode_stored_response(&view, &self.sealer()))
    }

    /// Encode a stored response's input items as a `GET /v1/responses/{id}/input_items` body
    /// (plan §9). Responses shape; protocol-agnostic (the router only exposes it on the
    /// Responses surface).
    pub fn encode_input_items(&self, s: &StoredResponse) -> Bytes {
        let view = stored_view(s);
        resp_stored::encode_input_items(&view)
    }
}

/// Whether the client asked for streamed usage (Chat `stream_options.include_usage: true`),
/// carried by the Chat decoder into `ext["chat.include_usage"]`.
fn include_usage_requested(req: &IrRequest) -> bool {
    req.ext
        .get("chat.include_usage")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Map the façade's [`StoredStatus`] onto the Responses codec's status enum.
fn map_status(s: StoredStatus) -> resp_stored::StoredStatus {
    match s {
        StoredStatus::Queued => resp_stored::StoredStatus::Queued,
        StoredStatus::InProgress => resp_stored::StoredStatus::InProgress,
        StoredStatus::Completed => resp_stored::StoredStatus::Completed,
        StoredStatus::Incomplete => resp_stored::StoredStatus::Incomplete,
        StoredStatus::Failed => resp_stored::StoredStatus::Failed,
        StoredStatus::Cancelled => resp_stored::StoredStatus::Cancelled,
    }
}

/// Borrow a [`StoredResponse`] as the Responses codec's [`resp_stored::StoredView`].
fn stored_view(s: &StoredResponse) -> resp_stored::StoredView<'_> {
    resp_stored::StoredView {
        id: &s.id,
        previous_id: s.previous_id.as_ref(),
        status: map_status(s.status),
        created_at: s.created_at,
        model: &s.binding.model,
        output_items: &s.output_items,
        usage: &s.usage,
        stop: &s.stop,
        request_echo: &s.request_echo,
        error: s.error.as_ref(),
        request_items: &s.request_items,
        instructions: &s.instructions,
    }
}
