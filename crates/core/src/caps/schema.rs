//! The capability schema (plan §6): the resolved [`Capabilities`] and every sub-table.
//!
//! One type serves two roles. As a *resolved* capability set it is the answer from
//! [`crate::caps::Registry::resolve`]; as a *sparse overlay* (`[defaults]`, a `[[model]]`
//! entry, or a [`BackendOverrides`]) it carries only the fields a layer sets. Overlaying is
//! uniform: `Option` fields replace when `Some`, [`Tri`] fields replace when not
//! [`Tri::Unknown`], and sub-tables recurse. Unset fields stay conservative
//! ([`Tri::Unknown`], `None`, empty), which the accessor helpers treat as "not supported".

use serde::{Deserialize, Serialize};

use crate::ir::{CacheTtl, Effort, Protocol};

/// A tri-state boolean. `Unknown` behaves as `No` for lossy features (plan §6) and is the
/// default for any capability a layer does not set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tri {
    /// Supported.
    Yes,
    /// Not supported.
    No,
    /// Not known — treated conservatively as "no" for lossy features.
    #[default]
    Unknown,
}

impl Tri {
    /// Whether this is definitely `Yes`.
    pub fn is_yes(self) -> bool {
        self == Tri::Yes
    }
    /// Whether this is `No` or `Unknown` (the conservative complement of [`Tri::is_yes`]).
    pub fn is_no_or_unknown(self) -> bool {
        self != Tri::Yes
    }
    /// Whether this is definitely `No`.
    pub fn is_no(self) -> bool {
        self == Tri::No
    }
    /// Whether this is `Unknown`.
    pub fn is_unknown(self) -> bool {
        self == Tri::Unknown
    }
}

impl Serialize for Tri {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Tri::Yes => s.serialize_bool(true),
            Tri::No => s.serialize_bool(false),
            Tri::Unknown => s.serialize_str("unknown"),
        }
    }
}

impl<'de> Deserialize<'de> for Tri {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bool(bool),
            Str(String),
        }
        Ok(match Raw::deserialize(d)? {
            Raw::Bool(true) => Tri::Yes,
            Raw::Bool(false) => Tri::No,
            Raw::Str(s) => match s.as_str() {
                "yes" | "true" => Tri::Yes,
                "no" | "false" => Tri::No,
                _ => Tri::Unknown,
            },
        })
    }
}

/// A sampling-parameter rule (plan §6 `[model.sampling]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingRule {
    /// Accepted within the inclusive `[min, max]` range.
    Range([f64; 2]),
    /// Rejected: sending a non-default value is a 400, so it must be dropped + degraded.
    Rejected,
    /// Ignored by the backend: still dropped + degraded (never silently forwarded).
    Ignored,
}

impl SamplingRule {
    /// Whether a value is inside an accepted range (always `false` for `Rejected`/`Ignored`).
    pub fn accepts(self, value: f64) -> bool {
        match self {
            SamplingRule::Range([lo, hi]) => value >= lo && value <= hi,
            _ => false,
        }
    }
}

impl Serialize for SamplingRule {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            SamplingRule::Range(r) => {
                use serde::ser::SerializeStruct;
                let mut st = s.serialize_struct("SamplingRule", 1)?;
                st.serialize_field("range", r)?;
                st.end()
            }
            SamplingRule::Rejected => s.serialize_str("rejected"),
            SamplingRule::Ignored => s.serialize_str("ignored"),
        }
    }
}

impl<'de> Deserialize<'de> for SamplingRule {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Str(String),
            Range { range: [f64; 2] },
        }
        match Raw::deserialize(d)? {
            Raw::Range { range } => Ok(SamplingRule::Range(range)),
            Raw::Str(s) => match s.as_str() {
                "rejected" => Ok(SamplingRule::Rejected),
                "ignored" => Ok(SamplingRule::Ignored),
                other => Err(serde::de::Error::custom(format!("invalid sampling rule: {other}"))),
            },
        }
    }
}

/// A `{ max = N }` rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MaxRule {
    /// The maximum, if bounded.
    pub max: Option<u32>,
}

macro_rules! overlay_fields {
    ($self:ident, $other:ident; opt: $($o:ident),* ; tri: $($t:ident),* $(;)?) => {{
        $( if $other.$o.is_some() { $self.$o = $other.$o.clone(); } )*
        $( if !matches!($other.$t, Tri::Unknown) { $self.$t = $other.$t; } )*
    }};
}

// ---------- transport ----------

/// Allowed streaming modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Streaming {
    /// Both streaming and non-streaming.
    Both,
    /// Streaming only.
    StreamOnly,
    /// Non-streaming only.
    NonStreamOnly,
}

/// When usage is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageTiming {
    /// At message start and end.
    StartAndEnd,
    /// Only at the end.
    EndOnly,
}

/// Transport capabilities (`[model.transport]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TransportCap {
    /// Allowed wire protocols (TOML tokens `chat` / `responses` / `anthropic`).
    #[serde(with = "opt_protocol_vec")]
    pub protocols: Option<Vec<Protocol>>,
    /// Streaming mode.
    pub streaming: Option<Streaming>,
    /// Usage-report timing.
    pub usage_timing: Option<UsageTiming>,
    /// Chat needs `stream_options.include_usage` to receive stream usage.
    pub stream_usage_opt_in: Tri,
    /// Suggested keepalive interval in seconds.
    pub keepalive_interval_secs: Option<u32>,
    /// API version header value.
    pub api_version: Option<String>,
    /// Beta-header feature → header value map.
    pub beta_headers: Option<std::collections::BTreeMap<String, String>>,
    /// `max_output_tokens` must be supplied (Anthropic).
    pub max_output_tokens_required: Tri,
    /// Default `max_output_tokens` to inject when absent.
    pub default_max_output_tokens: Option<u32>,
    /// The backend rejects any output-token limit (`max_tokens` /
    /// `max_completion_tokens` / `max_output_tokens`), e.g. the ChatGPT-subscription
    /// backend behind a Codex connector. `Yes` ⇒ lowering drops the client's limit with a
    /// `Dropped` degradation; `No`/`Unknown` ⇒ the limit is forwarded as usual (this is a
    /// deliberate exception to "unknown ⇒ unsupported": a limit is a hint, not a lossy
    /// feature, and stripping it by default would silently change every backend).
    pub max_output_tokens_rejected: Tri,
}

impl TransportCap {
    fn overlay(&mut self, o: &TransportCap) {
        overlay_fields!(self, o;
            opt: protocols, streaming, usage_timing, keepalive_interval_secs, api_version,
                 beta_headers, default_max_output_tokens;
            tri: stream_usage_opt_in, max_output_tokens_required, max_output_tokens_rejected);
    }
}

// ---------- instructions ----------

/// How a top-level system prompt is represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopLevelSystem {
    /// A single `system` string.
    #[serde(rename = "string")]
    Plain,
    /// An array of text blocks (Anthropic).
    TextBlocks,
    /// Not supported.
    None,
}

/// How mid-conversation system messages are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MidConversationSystem {
    /// Not supported (fall back to inline-wrap or fail).
    None,
    /// Native `role:"system"` messages mid-array.
    Native,
    /// Only via inline-wrap (never native).
    InlineWrapOnly,
}

/// Instruction capabilities (`[model.instructions]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InstructionsCap {
    /// Top-level system representation.
    pub top_level_system: Option<TopLevelSystem>,
    /// Mid-conversation system handling.
    pub mid_conversation_system: Option<MidConversationSystem>,
    /// Placement rule string (e.g. `follows_user_or_server_tool_and_precedes_assistant_or_ends_array`).
    pub mid_system_placement: Option<String>,
    /// Whether a distinct `developer` role exists.
    pub developer_role: Tri,
    /// Whether `system_clear_at` is supported.
    pub system_clear_at: Tri,
    /// Whether a mid-conversation effort override on a system message is supported.
    pub system_effort_override: Tri,
    /// Client `x-anthropic-<name>:` system header blocks (e.g. Claude Code's
    /// `x-anthropic-billing-header`) the backend accepts, by `<name>` (`["billing-header"]`).
    /// A captured header whose name is not listed is dropped rather than forwarded as prompt
    /// text. `None` forwards none.
    pub system_headers: Option<Vec<String>>,
}

impl InstructionsCap {
    fn overlay(&mut self, o: &InstructionsCap) {
        overlay_fields!(self, o;
            opt: top_level_system, mid_conversation_system, mid_system_placement, system_headers;
            tri: developer_role, system_clear_at, system_effort_override);
    }
}

// ---------- reasoning ----------

/// Reasoning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// No reasoning.
    None,
    /// Explicit `budget_tokens` mode.
    Budget,
    /// Adaptive thinking + effort.
    Adaptive,
    /// Effort levels only (no budget).
    EffortOnly,
}

/// Whether an explicit thinking budget is accepted, ignored, or rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetStatus {
    /// Accepted.
    Accepted,
    /// Silently ignored (still dropped + degraded on our side).
    Ignored,
    /// Rejected (400 if sent).
    Rejected,
}

/// Reasoning-budget rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BudgetRule {
    /// Minimum budget.
    pub min: Option<u32>,
    /// Budget status.
    pub status: Option<BudgetStatus>,
}

/// How reasoning is exposed by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureMode {
    /// Full reasoning text.
    FullText,
    /// Summary only.
    Summary,
    /// Nothing exposed.
    None,
}

/// How reasoning is replayed on tool loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    /// Verbatim signature replay (Anthropic).
    Signature,
    /// Encrypted reasoning item (OpenAI stateless).
    EncryptedItem,
    /// A text field carrier.
    TextField,
    /// No replay slot.
    None,
}

/// Reasoning capabilities (`[model.reasoning]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReasoningCap {
    /// Reasoning mode.
    pub mode: Option<ReasoningMode>,
    /// Supported effort levels.
    pub effort_levels: Option<Vec<Effort>>,
    /// Budget rule.
    pub budget: Option<BudgetRule>,
    /// Exposure mode.
    pub exposure: Option<ExposureMode>,
    /// Replay mode.
    pub replay: Option<ReplayMode>,
    /// Reasoning must be replayed on the last tool-bearing assistant turn.
    pub required_on_last_tool_turn: Tri,
    /// Tool calling works while reasoning is enabled.
    pub tools_with_reasoning: Tri,
    /// A fixed temperature is forced when thinking is on.
    pub forced_temperature_with_thinking: Tri,
}

impl ReasoningCap {
    fn overlay(&mut self, o: &ReasoningCap) {
        overlay_fields!(self, o;
            opt: mode, effort_levels, budget, exposure, replay;
            tri: required_on_last_tool_turn, tools_with_reasoning, forced_temperature_with_thinking);
    }
}

// ---------- tools ----------

/// Default strict-mode behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrictDefault {
    /// Strict off by default.
    Off,
    /// Strict attempted by default.
    Attempted,
}

/// Strict-schema tool support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StrictRule {
    /// Whether strict is supported at all.
    pub supported: Tri,
    /// Default strict behavior.
    pub default: Option<StrictDefault>,
}

/// A tool-choice mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceKind {
    /// `auto`.
    Auto,
    /// `none`.
    None,
    /// `required` / `any`.
    Required,
    /// A named tool.
    Named,
}

/// A tool-result content kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultContentKind {
    /// Text.
    Text,
    /// Image.
    Image,
    /// Document.
    Document,
    /// Audio.
    Audio,
}

/// Tool capabilities (`[model.tools]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsCap {
    /// Function tools supported.
    pub function_tools: Tri,
    /// Strict-schema support.
    pub strict: StrictRule,
    /// Supported tool-choice modes.
    pub tool_choice: Option<Vec<ToolChoiceKind>>,
    /// `parallel_tool_calls` control supported.
    pub parallel_control: Tri,
    /// Supported tool-result content kinds.
    pub result_content: Option<Vec<ResultContentKind>>,
    /// Tool-call id validation pattern.
    pub id_pattern: Option<String>,
    /// Provider-hosted tool type names.
    pub hosted: Option<Vec<String>>,
    /// JSON-schema keywords the tool parameter schema does not support.
    pub schema_unsupported_keywords: Option<Vec<String>>,
    /// Maximum number of tools.
    pub max_tools: Option<u32>,
    /// Whether tools may change mid-conversation.
    pub mid_conversation_tool_changes: Tri,
    /// Whether a tool-result message should carry the tool's `name` alongside its call id.
    ///
    /// Some OpenAI-compatible backends resolve a tool result's name positionally — by pairing
    /// each `tool` message with the preceding assistant `tool_calls` entry in order — and reject
    /// the request outright when that pairing is ambiguous. A client is free to return results
    /// in a different order from the calls, so ordering alone is not always enough; declaring
    /// this makes the Chat encoder name each result explicitly instead. Left `Unknown` (the
    /// conservative default) the field is omitted, since a strict server may reject an
    /// unexpected key.
    pub result_name: Tri,
}

impl ToolsCap {
    fn overlay(&mut self, o: &ToolsCap) {
        if !o.strict.supported.is_unknown() {
            self.strict.supported = o.strict.supported;
        }
        if o.strict.default.is_some() {
            self.strict.default = o.strict.default;
        }
        overlay_fields!(self, o;
            opt: tool_choice, result_content, id_pattern, hosted, schema_unsupported_keywords,
                 max_tools;
            tri: function_tools, parallel_control, mid_conversation_tool_changes, result_name);
    }
}

// ---------- output ----------

/// Structured-output format support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormatCap {
    /// None.
    None,
    /// `json_object` only.
    JsonObject,
    /// `json_schema` only.
    JsonSchema,
    /// Both `json_object` and `json_schema`.
    Both,
}

/// Output capabilities (`[model.output]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputCap {
    /// Supported output format.
    pub format: Option<OutputFormatCap>,
    /// Strict structured output supported.
    pub strict_supported: Tri,
    /// JSON-schema keywords not supported (stripped, recursively, order-preserving).
    pub schema_unsupported_keywords: Option<Vec<String>>,
    /// Structured output works together with tools.
    pub format_with_tools: Tri,
    /// Assistant-message prefill allowed.
    pub prefill_allowed: Tri,
    /// `verbosity` supported.
    pub verbosity: Tri,
}

impl OutputCap {
    fn overlay(&mut self, o: &OutputCap) {
        overlay_fields!(self, o;
            opt: format, schema_unsupported_keywords;
            tri: strict_supported, format_with_tools, prefill_allowed, verbosity);
    }
}

// ---------- media ----------

/// A media source kind (matches [`crate::ir::MediaSource`] shapes plus `file_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaSourceKind {
    /// Inline base64.
    Base64,
    /// A URL.
    Url,
    /// A provider file id.
    FileId,
    /// Inline text.
    Text,
}

/// Per-media-type capability.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MediaKind {
    /// Supported source kinds.
    pub sources: Option<Vec<MediaSourceKind>>,
    /// Supported MIME types.
    pub mime: Option<Vec<String>>,
    /// Maximum bytes.
    pub max_bytes: Option<u64>,
    /// Maximum count.
    pub max_count: Option<u32>,
}

impl MediaKind {
    fn overlay(&mut self, o: &MediaKind) {
        overlay_fields!(self, o; opt: sources, mime, max_bytes, max_count; tri:);
    }
    /// Whether a given source kind is supported.
    pub fn allows(&self, kind: MediaSourceKind) -> bool {
        self.sources.as_ref().is_some_and(|s| s.contains(&kind))
    }
}

/// Media capabilities (`[model.media]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MediaCap {
    /// Image support.
    pub image: MediaKind,
    /// PDF support.
    pub pdf: MediaKind,
    /// Text-document support.
    pub text_doc: MediaKind,
    /// Audio support.
    pub audio: MediaKind,
    /// The family namespace of file ids this backend mints.
    pub file_id_namespace: Option<String>,
    /// Container upload supported.
    pub container_upload: Tri,
    /// `image_url.detail` parameter supported.
    pub image_detail_param: Tri,
}

impl MediaCap {
    fn overlay(&mut self, o: &MediaCap) {
        self.image.overlay(&o.image);
        self.pdf.overlay(&o.pdf);
        self.text_doc.overlay(&o.text_doc);
        self.audio.overlay(&o.audio);
        overlay_fields!(self, o; opt: file_id_namespace; tri: container_upload, image_detail_param);
    }
}

// ---------- sampling ----------

/// Sampling capabilities (`[model.sampling]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingCap {
    /// Temperature rule.
    pub temperature: Option<SamplingRule>,
    /// top-p rule.
    pub top_p: Option<SamplingRule>,
    /// top-k rule.
    pub top_k: Option<SamplingRule>,
    /// Seed rule.
    pub seed: Option<SamplingRule>,
    /// Stop-sequence count limit.
    pub stop_sequences: Option<MaxRule>,
    /// Frequency-penalty rule.
    pub frequency_penalty: Option<SamplingRule>,
    /// Presence-penalty rule.
    pub presence_penalty: Option<SamplingRule>,
    /// Logit-bias rule.
    pub logit_bias: Option<SamplingRule>,
    /// Logprobs rule.
    pub logprobs: Option<SamplingRule>,
    /// Completion-count (`n`) limit.
    pub n: Option<MaxRule>,
    /// Supported service tiers.
    pub service_tier: Option<Vec<String>>,
}

impl SamplingCap {
    fn overlay(&mut self, o: &SamplingCap) {
        overlay_fields!(self, o;
            opt: temperature, top_p, top_k, seed, stop_sequences, frequency_penalty,
                 presence_penalty, logit_bias, logprobs, n, service_tier;
            tri:);
    }
}

// ---------- state ----------

/// Compaction mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionMode {
    /// Server-side compaction.
    Server,
    /// A dedicated endpoint.
    Endpoint,
    /// Anthropic `context_management` blocks.
    ContextManagement,
    /// None.
    None,
}

/// Server-state capabilities (`[model.state]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StateCap {
    /// `store` supported.
    pub store: Tri,
    /// `previous_response_id` chaining supported.
    pub previous_response_id: Tri,
    /// Conversation API supported.
    pub conversation_api: Tri,
    /// Background mode supported.
    pub background: Tri,
    /// `include: ["reasoning.encrypted_content"]` supported.
    pub encrypted_reasoning_include: Tri,
    /// Compaction mechanism.
    pub compaction: Option<CompactionMode>,
    /// Zero-data-retention: forces `store=false` upstream (backend-supplied).
    pub zdr: Tri,
}

impl StateCap {
    fn overlay(&mut self, o: &StateCap) {
        overlay_fields!(self, o;
            opt: compaction;
            tri: store, previous_response_id, conversation_api, background,
                 encrypted_reasoning_include, zdr);
    }
}

// ---------- cache ----------

/// Explicit cache-breakpoint rule.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BreakpointRule {
    /// Maximum explicit breakpoints.
    pub max: Option<u32>,
    /// Supported TTLs.
    pub ttl: Option<Vec<CacheTtl>>,
}

/// Cache capabilities (`[model.cache]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheCap {
    /// Explicit breakpoint rule.
    pub explicit_breakpoints: Option<BreakpointRule>,
    /// Automatic request-level caching (Anthropic).
    pub auto_request_level: Tri,
    /// `prompt_cache_key` supported (OpenAI).
    pub prompt_cache_key: Tri,
    /// Usage fields reported.
    pub usage_fields: Option<Vec<String>>,
}

impl CacheCap {
    fn overlay(&mut self, o: &CacheCap) {
        overlay_fields!(self, o;
            opt: explicit_breakpoints, usage_fields;
            tri: auto_request_level, prompt_cache_key);
    }
}

// ---------- session ----------

/// A place a backend accepts a session-affinity id.
///
/// TOML/JSON shape is a single-key table: `{ field = "prompt_cache_key" }` for a top-level
/// request body field, or `{ header = "x-session-affinity" }` for a request header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionSink {
    /// A top-level request body field with this name.
    Field(String),
    /// A request header with this name.
    Header(String),
}

impl SessionSink {
    /// The place's name (the field or header name), regardless of kind.
    pub fn name(&self) -> &str {
        match self {
            SessionSink::Field(n) | SessionSink::Header(n) => n,
        }
    }
}

impl Serialize for SessionSink {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("SessionSink", 1)?;
        match self {
            SessionSink::Field(n) => st.serialize_field("field", n)?,
            SessionSink::Header(n) => st.serialize_field("header", n)?,
        }
        st.end()
    }
}

impl<'de> Deserialize<'de> for SessionSink {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            field: Option<String>,
            #[serde(default)]
            header: Option<String>,
        }
        let raw = Raw::deserialize(d)?;
        match (raw.field, raw.header) {
            (Some(f), None) => Ok(SessionSink::Field(f)),
            (None, Some(h)) => Ok(SessionSink::Header(h)),
            (Some(_), Some(_)) => Err(serde::de::Error::custom(
                "session sink must set exactly one of `field` / `header`, not both",
            )),
            (None, None) => Err(serde::de::Error::custom(
                "session sink must set one of `field` / `header`",
            )),
        }
    }
}

/// Session-affinity capabilities (`[model.session]`).
///
/// Describes where, if anywhere, a backend accepts a client session id, and whether one is
/// mandatory. See [`crate::ir::SessionConfig`] and [`crate::session`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionCap {
    /// Ordered list of places this backend accepts a session id, most-preferred first. The
    /// first entry is the one used when emitting; the list exists for future expansion.
    /// `None` / empty ⇒ the backend accepts no session id (nothing is emitted).
    pub accepts: Option<Vec<SessionSink>>,
    /// Whether the backend **requires** a session id to function. `Yes` ⇒ a request with no
    /// captured session id is rejected during lowering.
    pub required: Tri,
}

impl SessionCap {
    fn overlay(&mut self, o: &SessionCap) {
        overlay_fields!(self, o; opt: accepts; tri: required);
    }
    /// The place a session id should be emitted, if any (the first accepted sink).
    pub fn primary_sink(&self) -> Option<&SessionSink> {
        self.accepts.as_ref().and_then(|v| v.first())
    }
}

// ---------- limits ----------

/// Numeric limits (`[model.limits]`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LimitsCap {
    /// Context window in tokens.
    pub context_window: Option<u32>,
    /// Max output tokens.
    pub max_output_tokens: Option<u32>,
    /// Max messages.
    pub max_messages: Option<u32>,
    /// Max request bytes.
    pub max_request_bytes: Option<u64>,
}

impl LimitsCap {
    fn overlay(&mut self, o: &LimitsCap) {
        overlay_fields!(self, o;
            opt: context_window, max_output_tokens, max_messages, max_request_bytes; tri:);
    }
}

// ---------- errors ----------

/// Error-mapping data (`[model.errors]`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ErrorsCap {
    /// Substrings that mark a context-length error in a 400 body.
    pub context_length_patterns: Option<Vec<String>>,
    /// HTTP statuses that are retryable.
    pub retryable_status: Option<Vec<u16>>,
    /// The "overloaded" status for this backend.
    pub overloaded_status: Option<u16>,
    /// Valid stop-reason strings.
    pub stop_reasons: Option<Vec<String>>,
}

impl ErrorsCap {
    fn overlay(&mut self, o: &ErrorsCap) {
        overlay_fields!(self, o;
            opt: context_length_patterns, retryable_status, overloaded_status, stop_reasons; tri:);
    }
}

// ---------- Capabilities ----------

/// A resolved capability set, or (used sparsely) an overlay layer. See the module docs.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Capabilities {
    /// The `verified_at` date of the layer that last set this (informational).
    pub verified_at: Option<String>,
    /// Transport.
    pub transport: TransportCap,
    /// Instructions.
    pub instructions: InstructionsCap,
    /// Reasoning.
    pub reasoning: ReasoningCap,
    /// Tools.
    pub tools: ToolsCap,
    /// Output.
    pub output: OutputCap,
    /// Media.
    pub media: MediaCap,
    /// Sampling.
    pub sampling: SamplingCap,
    /// State.
    pub state: StateCap,
    /// Session affinity.
    pub session: SessionCap,
    /// Cache.
    pub cache: CacheCap,
    /// Limits.
    pub limits: LimitsCap,
    /// Errors.
    pub errors: ErrorsCap,
}

/// A backend/deployment override layer — a sparse [`Capabilities`] applied last in
/// [`crate::caps::Registry::resolve`] (streaming mode, ZDR, beta-header gaps, …).
pub type BackendOverrides = Capabilities;

impl Capabilities {
    /// The fully-conservative capability set: every boolean [`Tri::Unknown`], every option
    /// `None`, every list empty. Unknown behaves as "not supported".
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Overlay `other` onto `self` in place (later layer wins per field).
    pub fn overlay(&mut self, other: &Capabilities) {
        if other.verified_at.is_some() {
            self.verified_at = other.verified_at.clone();
        }
        self.transport.overlay(&other.transport);
        self.instructions.overlay(&other.instructions);
        self.reasoning.overlay(&other.reasoning);
        self.tools.overlay(&other.tools);
        self.output.overlay(&other.output);
        self.media.overlay(&other.media);
        self.sampling.overlay(&other.sampling);
        self.state.overlay(&other.state);
        self.session.overlay(&other.session);
        self.cache.overlay(&other.cache);
        self.limits.overlay(&other.limits);
        self.errors.overlay(&other.errors);
    }

    /// Whether a wire protocol is allowed (conservative: `false` if unset).
    pub fn protocol_allowed(&self, p: Protocol) -> bool {
        self.transport.protocols.as_ref().is_some_and(|ps| ps.contains(&p))
    }

    /// Whether an effort level is supported. [`Effort::None`] (no thinking) is always
    /// supported; otherwise the reasoning mode must be enabled and, when `effort_levels` is
    /// set, must list the level.
    pub fn effort_supported(&self, e: Effort) -> bool {
        if e == Effort::None {
            return true;
        }
        match self.reasoning.mode {
            Some(ReasoningMode::None) | None => false,
            Some(_) => match &self.reasoning.effort_levels {
                Some(levels) => levels.contains(&e),
                None => true,
            },
        }
    }

    /// Whether a captured client `x-anthropic-<name>:` system header block may be forwarded
    /// (conservative: `false` unless `instructions.system_headers` lists `name`).
    pub fn forwards_system_header(&self, name: &str) -> bool {
        self.instructions.system_headers.as_ref().is_some_and(|l| l.iter().any(|n| n == name))
    }

    /// The effective streaming mode (default [`Streaming::Both`] when unset).
    pub fn streaming(&self) -> Streaming {
        self.transport.streaming.unwrap_or(Streaming::Both)
    }

    /// Whether the request should be sent to the backend as a stream, given what the client
    /// asked for.
    ///
    /// When the backend does not accept both modes, the backend decides and the router bridges
    /// the difference: a [`Streaming::StreamOnly`] backend is sent a streaming request even for
    /// a non-streaming client (the response is aggregated back into one body), and a
    /// [`Streaming::NonStreamOnly`] backend is sent a plain request even for a streaming client
    /// (the client's stream is synthesized from the response). Only under [`Streaming::Both`]
    /// does the client's own choice carry through.
    ///
    /// Encoders must derive the request's `stream` flag from this rather than from the client's
    /// intent alone, or a stream-only backend receives a non-streaming request and rejects it.
    pub fn upstream_streams(&self, client_wants_stream: bool) -> bool {
        match self.streaming() {
            Streaming::StreamOnly => true,
            Streaming::NonStreamOnly => false,
            Streaming::Both => client_wants_stream,
        }
    }
}

/// serde (de)serialization for the `protocols` field: TOML short tokens
/// `chat` / `responses` / `anthropic` <-> [`Protocol`].
mod opt_protocol_vec {
    use super::Protocol;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    fn token(p: Protocol) -> &'static str {
        match p {
            Protocol::OaiChat => "chat",
            Protocol::OaiResponses => "responses",
            Protocol::Anthropic => "anthropic",
        }
    }

    pub fn serialize<S: Serializer>(v: &Option<Vec<Protocol>>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            None => s.serialize_none(),
            Some(list) => {
                let toks: Vec<&str> = list.iter().map(|p| token(*p)).collect();
                toks.serialize(s)
            }
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<Protocol>>, D::Error> {
        let opt: Option<Vec<String>> = Option::deserialize(d)?;
        match opt {
            None => Ok(None),
            Some(list) => {
                let mut out = Vec::with_capacity(list.len());
                for s in list {
                    out.push(match s.as_str() {
                        "chat" => Protocol::OaiChat,
                        "responses" => Protocol::OaiResponses,
                        "anthropic" => Protocol::Anthropic,
                        other => {
                            return Err(serde::de::Error::custom(format!(
                                "unknown protocol token: {other}"
                            )))
                        }
                    });
                }
                Ok(Some(out))
            }
        }
    }
}
