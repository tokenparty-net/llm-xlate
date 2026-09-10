//! Shared codec plumbing every protocol crate (`chat`, `responses`, `anthropic`) implements.
//!
//! # The two boundaries (codecs rely on this)
//!
//! Opaque reasoning blobs live in two forms depending on which side of a translation they
//! are on:
//!
//! * **Client-facing boundary** — [`Codec::decode_request`], [`Codec::stream_encoder`] /
//!   [`Codec::encode_response`]: blobs are **envelope-sealed**. On the way in
//!   ([`Codec::decode_request`]) a blob that starts with `rtr1.` is opened with the
//!   [`Sealer`] and its family comes from the envelope; anything else is treated as
//!   **native** with `family = self.protocol().family()`. Use
//!   [`Sealer::open_or_native`] for exactly this. On the way out
//!   ([`Codec::stream_encoder`] / [`Codec::encode_response`]) blobs bound to a *foreign*
//!   family are sealed into an `rtr1.` envelope so the client can replay them later.
//! * **Provider-facing boundary** — [`Codec::encode_request`], [`Codec::stream_decoder`] /
//!   [`Codec::decode_response`]: blobs are **native** (verbatim signatures / encrypted
//!   content the upstream produced or expects). A same-family blob is replayed natively; a
//!   foreign one is dropped (never replayed as text).
//!
//! Every lossy step in [`Codec::encode_request`] records a [`Degradation`] into
//! [`EncodedRequest::degradations`]; nothing is dropped silently.

pub use http::{HeaderMap, HeaderName, HeaderValue};

use bytes::Bytes;
use serde_json::{Map, Value};

use crate::caps::Capabilities;
use crate::degrade::{Degradation, Degradations};
use crate::envelope::Sealer;
use crate::error::XlateError;
use crate::ir::{IrEvent, IrRequest, IrResponse, Protocol, ReasoningExposure, ResponseId};
use crate::wrap::WRAP_VERSION;

/// Everything a response encoder needs to echo back to the client. Built by the facade after
/// `decode_request`, carried through `encode_request`, and handed to `stream_encoder` /
/// `encode_response`.
#[derive(Debug, Clone)]
pub struct EncodeCtx {
    /// The protocol the client speaks (what we render back to).
    pub client_protocol: Protocol,
    /// The model string to echo to the client.
    pub client_model: String,
    /// Router-minted client-facing response id.
    pub response_id: ResponseId,
    /// Chat `stream_options.include_usage` was requested by the client.
    pub include_usage: bool,
    /// How the client wants reasoning exposed.
    pub expose: ReasoningExposure,
    /// The client's `store` flag (Responses).
    pub store: Option<bool>,
    /// The client's `include` list (Responses).
    pub include: Vec<String>,
    /// Whether the client asked for a stream.
    pub stream: bool,
    /// Request params the Resp/Ant response envelopes must echo (instructions, tools,
    /// `text.format`, temperature, …). Insertion order preserved. Built by the facade via
    /// [`Codec::request_echo`] on the **client** codec, so each dialect shapes its own echo.
    pub request_echo: Map<String, Value>,
    /// The client-facing sealer (seals foreign opaque blobs into `rtr1.` envelopes).
    pub sealer: Sealer,
    /// Router-supplied unix seconds for `created`/`created_at`; `0` when unknown. Codecs
    /// **never** read a clock — determinism requires this be supplied.
    pub created_at: u64,
}

impl EncodeCtx {
    /// A context with defaults for the optional fields (no usage echo, exposure hidden, not
    /// streaming, empty echo, `created_at = 0`).
    pub fn new(
        client_protocol: Protocol,
        client_model: impl Into<String>,
        response_id: ResponseId,
        sealer: Sealer,
    ) -> Self {
        Self {
            client_protocol,
            client_model: client_model.into(),
            response_id,
            include_usage: false,
            expose: ReasoningExposure::None,
            store: None,
            include: Vec::new(),
            stream: false,
            request_echo: Map::new(),
            sealer,
            created_at: 0,
        }
    }
}

/// A wire request ready to send upstream, plus the response context and any degradations.
#[derive(Debug, Clone)]
pub struct EncodedRequest {
    /// The serialized request body.
    pub body: Bytes,
    /// Headers to send (auth is the router's job; these are protocol/beta headers).
    pub headers: HeaderMap,
    /// Whether the upstream request asks for a streaming response.
    pub upstream_streams: bool,
    /// The response context to hand to the stream encoder / response encoder.
    pub ctx: EncodeCtx,
    /// Lossy steps taken while lowering + wiring this request.
    pub degradations: Degradations,
}

/// A rendered error: status, headers, and either a non-streaming `body` or streaming
/// `frames` (empty when not streaming).
#[derive(Debug, Clone)]
pub struct EncodedError {
    /// HTTP status.
    pub status: u16,
    /// Response headers (e.g. `retry-after`, `x-router-upstream-request-id`).
    pub headers: HeaderMap,
    /// Non-streaming error body.
    pub body: Bytes,
    /// Streaming error frames (SSE). Empty when the error is rendered non-streaming.
    pub frames: Vec<Bytes>,
}

/// What a request decoder needs: the client-facing sealer and the translator config.
#[derive(Debug, Clone)]
pub struct DecodeCtx {
    /// The client-facing sealer (opens `rtr1.` envelopes on the way in).
    pub sealer: Sealer,
    /// Translator configuration.
    pub config: TranslatorConfig,
}

impl DecodeCtx {
    /// Build a decode context whose sealer is derived from `config.envelope_key`.
    pub fn from_config(config: TranslatorConfig) -> Self {
        let sealer = Sealer::new(&config.envelope_key);
        Self { sealer, config }
    }
}

impl Default for DecodeCtx {
    fn default() -> Self {
        Self::from_config(TranslatorConfig::default())
    }
}

/// A provider-SSE → [`IrEvent`] push state machine (provider-facing).
pub trait StreamDecoder {
    /// Feed upstream bytes; return any events that completed.
    fn push(&mut self, bytes: &[u8]) -> Vec<IrEvent>;
    /// Flush at end of the upstream stream.
    fn finish(&mut self) -> Vec<IrEvent>;
}

/// An [`IrEvent`] → client-SSE push state machine (client-facing).
pub trait StreamEncoder {
    /// Feed the next event; return any client frames to send.
    fn push(&mut self, ev: IrEvent) -> Vec<Bytes>;
    /// Flush at end of stream (final frames, e.g. `data: [DONE]`).
    fn finish(&mut self) -> Vec<Bytes>;
    /// A keepalive frame to emit while waiting, if the protocol has one.
    fn keepalive(&mut self) -> Option<Bytes>;
}

/// One protocol dialect's full codec. Implemented by each protocol crate against this API.
pub trait Codec {
    /// The protocol this codec speaks.
    fn protocol(&self) -> Protocol;

    /// Decode a client request body into the IR (client-facing: opens `rtr1.` envelopes).
    fn decode_request(
        &self,
        body: &[u8],
        hdrs: &HeaderMap,
        ctx: &DecodeCtx,
    ) -> Result<IrRequest, XlateError>;

    /// Regroup + wire an IR request for this protocol as an upstream target. Lossy steps are
    /// recorded in [`EncodedRequest::degradations`] (provider-facing: native blobs).
    fn encode_request(
        &self,
        req: &IrRequest,
        caps: &Capabilities,
        ctx: &EncodeCtx,
    ) -> Result<EncodedRequest, XlateError>;

    /// A streaming decoder for this protocol as an upstream source (provider-facing).
    fn stream_decoder(&self, caps: &Capabilities) -> Box<dyn StreamDecoder + Send>;

    /// A streaming encoder for this protocol as the client target (client-facing).
    fn stream_encoder(&self, ctx: EncodeCtx) -> Box<dyn StreamEncoder + Send>;

    /// Decode a non-streaming provider response body into events (provider-facing).
    fn decode_response(&self, body: &[u8], caps: &Capabilities)
        -> Result<Vec<IrEvent>, XlateError>;

    /// Encode an aggregated response for a non-streaming client (client-facing).
    fn encode_response(&self, r: &IrResponse, ctx: &EncodeCtx) -> Bytes;

    /// Shape this **client** dialect's response-envelope echo from the decoded client request.
    ///
    /// A spec-correct Responses `response` object — and Anthropic `message_start` /
    /// `message_delta` — must echo the request's parameters (instructions, tools,
    /// `tool_choice`, `temperature` / `top_p`, `max_output_tokens`, `parallel_tool_calls`,
    /// reasoning, `text.format`, `metadata`, `store`, `previous_response_id`, …). Only the
    /// codec that owns a dialect knows how to render those, but the codec running
    /// [`Codec::encode_request`] is the *backend* codec, not the client one — so the client
    /// codec never gets to build its own echo there.
    ///
    /// This hook closes that gap: the facade calls it on the **client** codec with the
    /// client's decoded [`IrRequest`] (pre-lowering, as the client sent it) and stashes the
    /// result into the [`EncodeCtx`] `request_echo` field, which [`Codec::encode_response`]
    /// and [`Codec::stream_encoder`] then read. `ctx` is the response context being assembled
    /// (`response_id`, `client_model`, `store`, `include`, `created_at`, …) so the echo can
    /// reference it.
    ///
    /// The default returns an empty map (Chat, whose streamed chunks echo nothing, and any
    /// dialect that needs no echo). Responses and Anthropic codecs **override** it.
    fn request_echo(&self, req: &IrRequest, ctx: &EncodeCtx) -> Map<String, Value> {
        let _ = (req, ctx);
        Map::new()
    }

    /// Decode a provider error body into the unified [`XlateError`].
    fn decode_error(
        &self,
        status: u16,
        body: &[u8],
        hdrs: &HeaderMap,
        caps: &Capabilities,
    ) -> XlateError;

    /// Render a unified error into this protocol's client dialect. `streaming` selects
    /// body-vs-frames; `started` says whether a `Start` event was already emitted (Resp:
    /// `response.failed` vs `error`).
    fn encode_error(&self, e: &XlateError, streaming: bool, started: bool) -> EncodedError;
}

/// Fallback for a mid-context instruction the target cannot place natively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MidInstructionFallback {
    /// Inline-wrap the instruction into the next user-side message (`wrap_v1`).
    #[default]
    InlineWrap,
    /// Fail with [`XlateError::incompatible_history`].
    Fail,
}

/// Policy for a foreign provider tool in the *current* request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForeignProviderTool {
    /// Return [`crate::error::ErrorKind::Unsupported`] (default; history folding still applies).
    #[default]
    Unsupported,
    /// Drop the tool + degrade.
    Drop,
}

/// Policy when required reasoning on the last tool turn cannot be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnresolvedReasoning {
    /// Return [`XlateError::incompatible_history`] (default).
    #[default]
    Fail,
    /// Strip the reasoning and record a degradation.
    StripAndDegrade,
}

/// Global translator configuration. The envelope key is per-deployment; rotating it
/// invalidates in-flight reasoning envelopes (plan §14.5).
#[derive(Clone)]
pub struct TranslatorConfig {
    /// HMAC key for the `rtr1.` envelope.
    pub envelope_key: Vec<u8>,
    /// Mid-context instruction fallback.
    pub mid_instruction_fallback: MidInstructionFallback,
    /// Foreign-provider-tool policy.
    pub foreign_provider_tool: ForeignProviderTool,
    /// Unresolved-required-reasoning policy.
    pub unresolved_reasoning: UnresolvedReasoning,
    /// The wrapper-string version in force.
    pub wrap_version: String,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            envelope_key: b"llm-xlate-dev-key".to_vec(),
            mid_instruction_fallback: MidInstructionFallback::default(),
            foreign_provider_tool: ForeignProviderTool::default(),
            unresolved_reasoning: UnresolvedReasoning::default(),
            wrap_version: WRAP_VERSION.to_string(),
        }
    }
}

impl std::fmt::Debug for TranslatorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranslatorConfig")
            .field("envelope_key", &format_args!("<redacted, {} bytes>", self.envelope_key.len()))
            .field("mid_instruction_fallback", &self.mid_instruction_fallback)
            .field("foreign_provider_tool", &self.foreign_provider_tool)
            .field("unresolved_reasoning", &self.unresolved_reasoning)
            .field("wrap_version", &self.wrap_version)
            .finish()
    }
}

/// Convenience: build a [`Degradation`] list from a single entry (used by codecs that want to
/// start a chain). Re-exported for ergonomics.
pub fn one_degradation(d: Degradation) -> Degradations {
    let mut ds = Degradations::new();
    ds.push(d);
    ds
}
