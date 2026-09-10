//! Serde structs for the pieces of the Chat Completions wire format we *read back* from a
//! provider: the non-streaming `chat.completion` response and the streaming
//! `chat.completion.chunk`. These are deliberately tolerant (every field `#[serde(default)]`,
//! unknown fields ignored) because the response path translates into the client dialect rather
//! than round-tripping the provider body — the byte-exact round-trip laws live on the
//! *client* boundary (`decode_request` / `encode_response` / the stream encoder), which build
//! their JSON with full control over field order.
//!
//! The request is parsed from a [`serde_json::Value`] in [`crate::decode`] instead of a struct,
//! so that every unknown top-level field can be captured verbatim into `ext["chat.<field>"]`.

use serde::Deserialize;
use serde_json::Value;

/// A non-streaming `chat.completion` response body.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireResponse {
    /// Response id.
    #[serde(default)]
    pub id: String,
    /// Model string.
    #[serde(default)]
    pub model: String,
    /// Choices (we consume `choices[0]`).
    #[serde(default)]
    pub choices: Vec<WireChoice>,
    /// Usage object (mapped by [`crate::common::decode_usage`]).
    #[serde(default)]
    pub usage: Option<Value>,
    /// System fingerprint.
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    /// Echoed service tier.
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// One choice in a `chat.completion` response.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireChoice {
    /// Choice index.
    #[serde(default)]
    pub index: u32,
    /// The assistant message.
    #[serde(default)]
    pub message: WireRespMessage,
    /// Finish reason.
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Logprobs payload (passed through into `ext` when present).
    #[serde(default)]
    pub logprobs: Option<Value>,
}

/// The assistant message inside a response choice.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireRespMessage {
    /// `content`: string | array | null.
    #[serde(default)]
    pub content: Option<Value>,
    /// Refusal string.
    #[serde(default)]
    pub refusal: Option<String>,
    /// Tool calls.
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCall>>,
    /// Annotations / citations.
    #[serde(default)]
    pub annotations: Option<Vec<Value>>,
    /// Gateway reasoning text (primary field).
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Gateway reasoning text (alternate field on some gateways).
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Structured reasoning detail entries.
    #[serde(default)]
    pub reasoning_details: Option<Vec<Value>>,
    /// Legacy `function_call` object.
    #[serde(default)]
    pub function_call: Option<WireFunctionCall>,
}

/// A `tool_calls[]` entry (also used for streaming deltas, where `index`/`id`/`function`
/// fields may all be partial).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireToolCall {
    /// Streaming delta index (absent in non-streaming responses).
    #[serde(default)]
    pub index: Option<u32>,
    /// Tool-call id.
    #[serde(default)]
    pub id: Option<String>,
    /// Tool type (`function`).
    #[serde(default, rename = "type")]
    pub type_: Option<String>,
    /// The function payload.
    #[serde(default)]
    pub function: WireFn,
}

/// The `function` payload of a tool call.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireFn {
    /// Function name.
    #[serde(default)]
    pub name: Option<String>,
    /// Argument JSON text (kept verbatim as a string).
    #[serde(default)]
    pub arguments: Option<String>,
}

/// A legacy `function_call` object on an assistant message.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireFunctionCall {
    /// Function name.
    #[serde(default)]
    pub name: Option<String>,
    /// Argument JSON text.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// A streaming `chat.completion.chunk`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireChunk {
    /// Response id.
    #[serde(default)]
    pub id: String,
    /// Model string.
    #[serde(default)]
    pub model: String,
    /// Choices (we consume `choices[0]`; a usage-only final chunk has none).
    #[serde(default)]
    pub choices: Vec<WireChunkChoice>,
    /// Usage on the final chunk (when `stream_options.include_usage`).
    #[serde(default)]
    pub usage: Option<Value>,
    /// Echoed service tier.
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// One streaming choice.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireChunkChoice {
    /// Choice index (only 0 is handled; `n>1` is unsupported).
    #[serde(default)]
    pub index: u32,
    /// The incremental delta.
    #[serde(default)]
    pub delta: WireDelta,
    /// Finish reason (on the closing chunk for this choice).
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// A streaming delta.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireDelta {
    /// Role (only on the first delta).
    #[serde(default)]
    pub role: Option<String>,
    /// Content text chunk.
    #[serde(default)]
    pub content: Option<String>,
    /// Refusal text chunk.
    #[serde(default)]
    pub refusal: Option<String>,
    /// Tool-call deltas.
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCall>>,
    /// Reasoning text chunk (primary field).
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Reasoning text chunk (alternate field).
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Structured reasoning detail entries.
    #[serde(default)]
    pub reasoning_details: Option<Vec<Value>>,
}
