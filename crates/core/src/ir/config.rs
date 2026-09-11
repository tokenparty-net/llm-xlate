//! Request-level IR: [`IrRequest`], [`Instruction`], and every configuration struct
//! (reasoning, output, sampling, limits, cache, state, meta, tools).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::ResponseId;
use super::item::{OpaqueItem, Part};

/// The full decoded request in item-based, protocol-neutral form (plan §5).
///
/// Field order is fixed (declaration order == serialization order) to keep output
/// canonical. Every field has a sensible [`Default`], so partial construction in tests is
/// `IrRequest { model, items, ..Default::default() }`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct IrRequest {
    /// Requested model (client name + optional router-resolved backend name).
    pub model: ModelRef,
    /// Leading and mid-context system/developer instructions.
    pub instructions: Vec<Instruction>,
    /// The transcript, as a flat list of items.
    pub items: Vec<super::item::Item>,
    /// Declared tools.
    pub tools: Vec<ToolDef>,
    /// Tool-choice policy.
    pub tool_choice: ToolChoice,
    /// Whether parallel tool calls are allowed (`None` = provider default).
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning / effort configuration.
    pub reasoning: ReasoningConfig,
    /// Structured-output configuration.
    pub output: OutputConfig,
    /// Token / stop-sequence limits.
    pub limits: Limits,
    /// Sampling parameters.
    pub sampling: Sampling,
    /// Prompt-cache hints.
    pub cache: CacheHints,
    /// Session-affinity information captured from the client request.
    pub session: SessionConfig,
    /// Server-state configuration (store, chaining, background, include list).
    pub state: StateConfig,
    /// Whether the client asked for a streaming response.
    pub stream: bool,
    /// Request metadata (user/safety identifiers, metadata map, service tier).
    pub meta: RequestMeta,
    /// Namespaced passthrough extensions.
    pub ext: Extensions,
}

/// Requested model. `resolved` is filled by the router once a backend is chosen.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ModelRef {
    /// Model string as the client sent it (echoed back to the client).
    pub client_name: String,
    /// Backend model string chosen by the router, if any.
    pub resolved: Option<String>,
}

impl ModelRef {
    /// Construct from a client-facing model name.
    pub fn new(client_name: impl Into<String>) -> Self {
        Self { client_name: client_name.into(), resolved: None }
    }
    /// The model string to send upstream (`resolved` if present, else `client_name`).
    pub fn upstream(&self) -> &str {
        self.resolved.as_deref().unwrap_or(&self.client_name)
    }
}

/// A system or developer instruction, with a placement in the transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    /// System vs developer.
    pub role: InstructionRole,
    /// Where this instruction sits relative to the items.
    pub position: Position,
    /// Content parts (text only in practice).
    pub content: Vec<Part>,
    /// Optional cache breakpoint on this instruction.
    pub cache_control: Option<CacheControl>,
    /// Anthropic mid-conversation effort override carried by a `role:"system"` message.
    pub effort: Option<Effort>,
    /// Anthropic `system_clear_at` marker (schema permissive).
    pub clear_at: Option<ClearAt>,
}

impl Instruction {
    /// A leading system instruction from plain text.
    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: InstructionRole::System,
            position: Position::Leading,
            content: vec![Part::text(text)],
            cache_control: None,
            effort: None,
            clear_at: None,
        }
    }
    /// A leading developer instruction from plain text.
    pub fn developer_text(text: impl Into<String>) -> Self {
        Self { role: InstructionRole::Developer, ..Self::system_text(text) }
    }
    /// The concatenated plain text of this instruction's text parts.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for p in &self.content {
            if let Part::Text { text, .. } = p {
                out.push_str(text);
            }
        }
        out
    }
}

/// Role of an [`Instruction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstructionRole {
    /// A `system` instruction.
    #[default]
    System,
    /// A `developer` instruction.
    Developer,
}

/// Placement of an [`Instruction`] relative to [`IrRequest::items`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Position {
    /// Before the first item (a leading instruction).
    #[default]
    Leading,
    /// Immediately before the item at this index (a mid-context instruction).
    Before(usize),
}

/// Reasoning / thinking configuration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Requested effort level.
    pub effort: Option<Effort>,
    /// Explicit Anthropic-style thinking budget in tokens.
    pub budget_tokens: Option<u32>,
    /// Explicit enable/disable (`None` = infer from `effort`).
    pub enabled: Option<bool>,
    /// How reasoning is exposed back to the client.
    pub expose: ReasoningExposure,
}

/// Effort levels, ordered `None < Minimal < Low < Medium < High < XHigh < Max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    /// No reasoning.
    #[default]
    None,
    /// Minimal reasoning (OpenAI `minimal`; Anthropic budget 1024).
    Minimal,
    /// Low effort.
    Low,
    /// Medium effort.
    Medium,
    /// High effort.
    High,
    /// Extra-high effort.
    #[serde(rename = "xhigh")]
    XHigh,
    /// Maximum effort (Anthropic `max`, Opus 4.6+).
    Max,
}

impl Effort {
    /// Map an Anthropic `budget_tokens` value to an effort bucket (plan §7.2):
    /// `<2048` → Low, `<8192` → Medium, `<24576` → High, else XHigh.
    pub fn from_budget_tokens(budget: u32) -> Effort {
        if budget < 2048 {
            Effort::Low
        } else if budget < 8192 {
            Effort::Medium
        } else if budget < 24576 {
            Effort::High
        } else {
            Effort::XHigh
        }
    }

    /// Map this effort to an Anthropic budget-mode `budget_tokens` value, capped at
    /// `max_tokens - 1` (plan §7.2). `None` yields 0 (caller gates on `enabled`).
    pub fn ant_budget(self, max_tokens: u32) -> u32 {
        let cap = max_tokens.saturating_sub(1);
        let base = match self {
            Effort::None => 0,
            Effort::Minimal => 1024,
            Effort::Low => 2048,
            Effort::Medium => 8192,
            Effort::High => 16384,
            Effort::XHigh => 32768,
            Effort::Max => cap,
        };
        base.min(cap)
    }

    /// The lowercase wire token used by OpenAI (`reasoning_effort`) for this level,
    /// mapping `Max` → `"xhigh"` (OpenAI has no `max`).
    ///
    /// **OpenAI only.** Do **not** reuse this for Anthropic adaptive effort: Anthropic has no
    /// `"minimal"` token and *does* accept `"max"` (Opus 4.6+), so this mapping would emit an
    /// invalid `"minimal"` and silently downgrade `Max`. Use [`Effort::ant_effort_token`] for
    /// Anthropic.
    pub fn openai_token(self) -> &'static str {
        match self {
            Effort::None => "none",
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh | Effort::Max => "xhigh",
        }
    }

    /// The Anthropic adaptive-effort `output_config.effort` token for this level (plan §7.2):
    /// `Minimal` → `"low"` (Anthropic has no `"minimal"`), `Low`/`Medium`/`High` → same,
    /// `XHigh` → `"xhigh"`, `Max` → `"max"`. `None` → `"low"` (callers gate reasoning on
    /// [`ReasoningConfig`]'s `enabled` / effort presence and never emit a token for disabled
    /// thinking).
    ///
    /// `Max` maps to `"max"` here **unconditionally**; the "`max` only if it appears in the
    /// model's `effort_levels`, else fall back to `"xhigh"`" rule (plan §7.2) is
    /// capability-dependent and must be applied by the codec against the model's
    /// [`crate::caps::ReasoningCap`] `effort_levels` before calling this — or by preferring
    /// [`Effort::openai_token`]'s `"xhigh"` when `max` is unlisted.
    pub fn ant_effort_token(self) -> &'static str {
        match self {
            Effort::None | Effort::Minimal | Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }
}

/// How reasoning is surfaced back to the client.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningExposure {
    /// Hide reasoning text; only opaque carriers (encrypted/signature) are round-tripped.
    #[default]
    None,
    /// Expose a summary at the given detail level.
    Summary(SummaryLevel),
    /// Expose full reasoning text.
    Full,
}

/// Summary detail level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryLevel {
    /// Provider-chosen detail.
    #[default]
    Auto,
    /// Concise summary.
    Concise,
    /// Detailed summary.
    Detailed,
}

/// Structured-output configuration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct OutputConfig {
    /// The requested output format.
    pub format: OutputFormat,
    /// Verbosity hint (OpenAI `verbosity`).
    pub verbosity: Option<Verbosity>,
}

/// Output format requested by the client.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// Free-form text.
    #[default]
    Text,
    /// Any JSON object (`json_object`).
    JsonObject,
    /// JSON constrained by a schema.
    JsonSchema {
        /// Schema name.
        name: String,
        /// The JSON schema (kept verbatim; never re-sorted).
        schema: Value,
        /// Whether strict adherence is requested.
        strict: bool,
        /// Optional schema description.
        description: Option<String>,
    },
}

/// Output verbosity hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verbosity {
    /// Terse output.
    Low,
    /// Balanced output.
    Medium,
    /// Verbose output.
    High,
}

/// A declared tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDef {
    /// A client-defined function tool.
    Function {
        /// Tool name.
        name: String,
        /// Optional description.
        description: Option<String>,
        /// JSON-schema parameters (kept verbatim).
        parameters: Value,
        /// Whether strict schema adherence is requested (`None` = provider default).
        strict: Option<bool>,
        /// Optional cache breakpoint on this tool.
        cache_control: Option<CacheControl>,
    },
    /// A provider-hosted tool, passed through opaquely.
    Provider(OpaqueItem),
}

/// Tool-choice policy.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model decides (default).
    #[default]
    Auto,
    /// No tools may be called.
    None,
    /// A tool call is required.
    Required,
    /// A specific named tool must be called.
    Named(String),
}

/// Token and stop-sequence limits.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Limits {
    /// Maximum output tokens (`max_tokens` / `max_completion_tokens` / `max_output_tokens`).
    pub max_output_tokens: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Vec<String>,
}

/// Sampling parameters. All optional; absent means "provider default".
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Sampling {
    /// Temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling top-p.
    pub top_p: Option<f64>,
    /// Top-k.
    pub top_k: Option<u32>,
    /// PRNG seed.
    pub seed: Option<i64>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Logit bias map (kept verbatim).
    pub logit_bias: Option<Value>,
    /// Whether logprobs are requested.
    pub logprobs: Option<bool>,
    /// Number of top logprobs requested.
    pub top_logprobs: Option<u32>,
    /// Number of completions requested.
    pub n: Option<u32>,
}

/// Prompt-cache hints.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheHints {
    /// Request-level automatic caching was requested (Anthropic `cache_control` at the
    /// request level, or inferred).
    pub request_level: bool,
    /// Explicit prompt cache key (OpenAI `prompt_cache_key`).
    pub prompt_cache_key: Option<String>,
}

/// Session-affinity information captured from the client request.
///
/// `id` is the session id the router uses for sticky routing / affinity, captured from the
/// first present of (in priority order) the `prompt_cache_key` body field, the `session_id`
/// body field, the `x-session-affinity` header, the `x-opencode-session` header, and the
/// `x-session-id` header (see [`crate::session::capture_session`]). On the way out it is
/// emitted to the first place the backend accepts a session id
/// ([`crate::caps::SessionCap`]).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SessionConfig {
    /// The captured session id, if the client supplied one.
    pub id: Option<String>,
    /// Labels of the other sources that also carried a session id whose value **differed**
    /// from [`SessionConfig::id`]. Populated at decode time so lowering can emit a single
    /// degradation; empty when the client supplied a consistent (or single) session id.
    pub conflicting_sources: Vec<String>,
}

/// Server-state configuration (Responses store / chaining / background / include).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StateConfig {
    /// Whether to store this response server-side (`None` = provider default).
    pub store: Option<bool>,
    /// Previous response id to chain from.
    pub previous_response_id: Option<ResponseId>,
    /// Conversation id (Responses conversation API).
    pub conversation: Option<String>,
    /// Whether to run in the background.
    pub background: Option<bool>,
    /// `include` list (e.g. `reasoning.encrypted_content`).
    pub include: Vec<String>,
}

/// Request metadata.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RequestMeta {
    /// End-user identifier (OpenAI `user`).
    pub user: Option<String>,
    /// Safety identifier (OpenAI `safety_identifier`).
    pub safety_identifier: Option<String>,
    /// Free-form metadata map (order preserved).
    pub metadata: serde_json::Map<String, Value>,
    /// Requested service tier.
    pub service_tier: Option<String>,
}

/// Namespaced passthrough extensions.
///
/// A `BTreeMap` (not `serde_json::RawValue`): the map gives deterministic key ordering and
/// `PartialEq`, and `serde_json::Value` under the `preserve_order` feature already keeps the
/// key order of each contained object. `RawValue` is not `PartialEq` and cannot be a map
/// value we compare in round-trip laws, so it is deliberately avoided here.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Extensions(pub BTreeMap<String, Value>);

impl Extensions {
    /// An empty extension map.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }
    /// Insert a namespaced value.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.0.insert(key.into(), value);
    }
    /// Look up a namespaced value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }
    /// Whether there are no extensions.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Iterate over the entries in key order.
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, String, Value> {
        self.0.iter()
    }
}

impl core::ops::Deref for Extensions {
    type Target = BTreeMap<String, Value>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl core::ops::DerefMut for Extensions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl From<BTreeMap<String, Value>> for Extensions {
    fn from(m: BTreeMap<String, Value>) -> Self {
        Self(m)
    }
}

/// A cache breakpoint with a TTL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    /// Time-to-live of the cache breakpoint.
    pub ttl: CacheTtl,
}

impl CacheControl {
    /// A 5-minute ephemeral breakpoint.
    pub fn ephemeral_5m() -> Self {
        Self { ttl: CacheTtl::FiveMinutes }
    }
    /// A 1-hour ephemeral breakpoint.
    pub fn ephemeral_1h() -> Self {
        Self { ttl: CacheTtl::OneHour }
    }
}

/// Cache time-to-live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CacheTtl {
    /// 5 minutes (`ephemeral`).
    #[default]
    #[serde(rename = "5m")]
    FiveMinutes,
    /// 1 hour.
    #[serde(rename = "1h")]
    OneHour,
}

/// Anthropic `system_clear_at` marker. The precise schema is not yet frozen, so the payload
/// is kept as a permissive raw value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClearAt {
    /// Raw provider payload (kept verbatim).
    pub raw: Value,
}

impl From<Value> for ClearAt {
    fn from(raw: Value) -> Self {
        Self { raw }
    }
}
