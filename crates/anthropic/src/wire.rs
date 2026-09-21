//! Wire (JSON) types for the Anthropic Messages API (Sept 2026), plus small shared
//! constants and JSON helpers.
//!
//! Request/response *decoding* uses these permissive structs (unknown fields are captured
//! with `#[serde(flatten)]` into a `serde_json::Map` so nothing is lost). Request/response
//! *encoding* builds ordered `serde_json::Map`s directly (via the helpers here), which keeps
//! full control over field order and conditional omission and lets provider-hosted blocks
//! pass through verbatim. `serde_json`'s `preserve_order` makes insertion order the
//! serialization order, so the output is canonical and deterministic.

use serde::Deserialize;
use serde_json::{Map, Value};

use llm_xlate_core::ProviderFamily;

/// The provider family this codec speaks.
pub const FAMILY: ProviderFamily = ProviderFamily::Anthropic;

/// The `ext` namespace prefix for Anthropic-specific passthrough (`anthropic.<field>`).
pub const EXT_PREFIX: &str = "anthropic.";

/// Default `anthropic-version` header when caps do not supply one.
pub const DEFAULT_API_VERSION: &str = "2023-06-01";

/// Build an `ext` key for a top-level Anthropic field.
pub fn ext_key(field: &str) -> String {
    format!("{EXT_PREFIX}{field}")
}

/// The `ext` field (`anthropic.system_headers`) carrying captured client `x-anthropic-<name>:`
/// system header blocks, in order: `[{"name", "text", "cache_control"?}]`. The encoder re-emits
/// a block only when `instructions.system_headers` lists its name.
pub const SYSTEM_HEADERS_EXT: &str = "system_headers";

/// The `<name>` of a system text block that is a client header line
/// (`x-anthropic-<name>: <value>`, a single line, `<name>` in `[a-z0-9-]`), else `None`. Only a
/// whole single-line block qualifies, so a real prompt that merely starts with the prefix is
/// never mistaken for one.
pub fn system_header_name(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("x-anthropic-")?;
    let (name, _) = rest.split_once(':')?;
    let valid = !name.is_empty()
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    (valid && !text.contains('\n')).then_some(name)
}

/// A fresh ordered JSON object.
pub fn omap() -> Map<String, Value> {
    Map::new()
}

/// Provider-hosted block/tool `type` prefixes and names that must be passed through
/// verbatim (as [`llm_xlate_core::OpaqueItem`]s) rather than modelled.
pub fn is_provider_tool_result_type(t: &str) -> bool {
    t.ends_with("_tool_result")
}

/// Whether a block `type` is a server (provider-hosted) tool call.
pub fn is_server_tool_use_type(t: &str) -> bool {
    t == "server_tool_use"
}

/// Whether a request `tool` entry (from `tools[]`) is a provider-hosted tool (has a `type`),
/// as opposed to a client function tool (which carries `input_schema` and no `type`).
pub fn tool_is_provider(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str).is_some()
}

// ---------------------------------------------------------------------------
// Request (decode)
// ---------------------------------------------------------------------------

/// A decoded Anthropic Messages request. Known fields are named; anything else lands in
/// `extra` and becomes `ext["anthropic.<field>"]`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct AntRequestWire {
    /// Target model.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<AntMessageWire>,
    /// Maximum output tokens (required by Anthropic; validated in decode).
    pub max_tokens: Option<u32>,
    /// Top-level system prompt (`string` or `[text blocks]`).
    pub system: Option<Value>,
    /// Declared tools (function + provider-hosted).
    pub tools: Option<Vec<Value>>,
    /// Tool-choice policy.
    pub tool_choice: Option<Value>,
    /// Thinking configuration.
    pub thinking: Option<Value>,
    /// Output configuration (effort + structured-output format).
    pub output_config: Option<Value>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    pub top_p: Option<f64>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Streaming flag.
    pub stream: Option<bool>,
    /// Request metadata (`{user_id}`).
    pub metadata: Option<Value>,
    /// Service tier.
    pub service_tier: Option<String>,
    /// Request-level cache control.
    pub cache_control: Option<Value>,
    /// Container (opaque passthrough).
    pub container: Option<Value>,
    /// Context-management config (opaque passthrough).
    pub context_management: Option<Value>,
    /// MCP servers (opaque passthrough).
    pub mcp_servers: Option<Value>,
    /// Any unmodelled top-level field.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One request message (`role` + `content`). Content is kept as a raw [`Value`] because it is
/// `string | [blocks]` and blocks include provider-hosted shapes we pass through verbatim.
#[derive(Debug, Deserialize)]
pub struct AntMessageWire {
    /// `user` | `assistant` | `system`.
    pub role: String,
    /// String content or an array of content blocks.
    #[serde(default)]
    pub content: Value,
}
