//! Streaming events and the aggregated response: [`IrEvent`], [`Delta`], [`Usage`],
//! [`StopReason`], and [`IrResponse`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::{CallId, ItemId, ProviderFamily, ResponseId};
use super::config::Extensions;
use super::item::{Annotation, Item, OpaqueBlob};
use crate::error::XlateError;

/// Token accounting, normalized to **disjoint** billing categories.
///
/// The prompt-side categories never overlap: `input` is the *fresh* (uncached) prompt count,
/// and the cached and written portions sit beside it rather than inside it. The gross prompt a
/// provider would call `prompt_tokens` / `input_tokens` is therefore [`Usage::gross_prompt`] =
/// `input + cache_read + cache_write`.
///
/// This is the Anthropic wire convention. The OpenAI dialects report a *gross* prompt with the
/// cached portion nested as a subset of it, so their codecs normalize on decode (via
/// [`Usage::from_gross`]) and re-gross on encode (via [`Usage::gross_prompt`]). Every codec
/// must hold to this: a codec that stores a gross count in `input` double-counts the cached
/// tokens on any cross-protocol route, and the categories bill at different rates.
///
/// `cache_write` counts cache writes at **every** TTL. `cache_write_1h` is the subset of them
/// written with a 1-hour TTL, which some providers price differently; it is always
/// `<= cache_write` (see [`Usage::enforce_invariants`]). A provider that reports one unsplit
/// write total leaves `cache_write_1h` at `None`, which means "TTL not reported", not "no
/// 1-hour writes".
///
/// `input`/`output` are always present (default 0); the cache and reasoning counters are
/// optional because not every provider reports them.
///
/// Not `Eq`: the `ext` map holds `serde_json::Value`, which is not `Eq` (floats).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Fresh (uncached) prompt tokens. **Not** the gross prompt — see [`Usage::gross_prompt`].
    pub input: u32,
    /// Output (completion) tokens.
    pub output: u32,
    /// Tokens read from cache.
    pub cache_read: Option<u32>,
    /// Tokens written to cache, at every TTL.
    pub cache_write: Option<u32>,
    /// The subset of `cache_write` written with a 1-hour TTL. `None` means the provider did not
    /// report a TTL split, not that there were no 1-hour writes.
    pub cache_write_1h: Option<u32>,
    /// Reasoning tokens (a subset of `output`, never added on top).
    pub reasoning: Option<u32>,
    /// Provider-specific extra usage fields (kept verbatim).
    pub ext: Extensions,
}

fn opt_add(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        (x, y) => Some(x.unwrap_or(0) + y.unwrap_or(0)),
    }
}

impl Usage {
    /// A usage with only input/output set. `input` is the **fresh** prompt count.
    pub fn new(input: u32, output: u32) -> Self {
        Self { input, output, ..Default::default() }
    }

    /// Build a usage from a **gross** prompt count — the OpenAI dialects' `prompt_tokens` /
    /// `input_tokens`, which counts the cached and written portions inside it — reducing it to
    /// the fresh remainder this type stores.
    ///
    /// Invariants are enforced *before* the subtraction, so a write total recovered from a
    /// 1-hour figure is never left billed as fresh input. Saturating: a provider whose subset
    /// counters exceed its own gross total yields `input = 0` rather than wrapping.
    pub fn from_gross(
        gross: u32,
        output: u32,
        cache_read: Option<u32>,
        cache_write: Option<u32>,
        cache_write_1h: Option<u32>,
    ) -> Self {
        let mut u = Usage {
            input: 0,
            output,
            cache_read,
            cache_write,
            cache_write_1h,
            reasoning: None,
            ext: Extensions::default(),
        };
        u.enforce_invariants();
        u.input = gross
            .saturating_sub(u.cache_read.unwrap_or(0))
            .saturating_sub(u.cache_write.unwrap_or(0));
        u
    }

    /// The gross prompt count: `input + cache_read + cache_write`. This is what the OpenAI
    /// dialects spell `prompt_tokens` / `input_tokens`, and what their `total_tokens` adds
    /// `output` to.
    pub fn gross_prompt(&self) -> u32 {
        self.input
            .saturating_add(self.cache_read.unwrap_or(0))
            .saturating_add(self.cache_write.unwrap_or(0))
    }

    /// The portion of `cache_write` **not** reported as 1-hour, for encoders that need an
    /// explicit short-TTL figure (Anthropic's `cache_creation.ephemeral_5m_input_tokens`).
    /// `None` when no write total is known. An unsplit total reports itself here, matching the
    /// convention that writes of unreported TTL bill at the short rate.
    pub fn cache_write_5m(&self) -> Option<u32> {
        self.cache_write.map(|w| w.saturating_sub(self.cache_write_1h.unwrap_or(0)))
    }

    /// Enforce `cache_write_1h <= cache_write`. A shape carrying a 1-hour figure without — or
    /// above — a write total (a `cache_creation` object with no flat total, or two conventions
    /// mixed into one object) would otherwise flow downstream as a contradiction that a
    /// consumer clamps into zero-billed tokens. Raising the total to the known 1-hour floor
    /// keeps every write token counted.
    pub fn enforce_invariants(&mut self) {
        if let Some(w1h) = self.cache_write_1h {
            if self.cache_write.is_none_or(|w| w < w1h) {
                self.cache_write = Some(w1h);
            }
        }
    }

    /// Accumulate `other` into `self` field-by-field (used when totalling usage across several
    /// requests). Optional counters combine as "present if either is present"; `ext` entries
    /// from `other` overwrite on key collision.
    pub fn add(&mut self, other: &Usage) {
        self.input += other.input;
        self.output += other.output;
        self.cache_read = opt_add(self.cache_read, other.cache_read);
        self.cache_write = opt_add(self.cache_write, other.cache_write);
        self.cache_write_1h = opt_add(self.cache_write_1h, other.cache_write_1h);
        self.reasoning = opt_add(self.reasoning, other.reasoning);
        for (k, v) in other.ext.iter() {
            self.ext.insert(k.clone(), v.clone());
        }
        self.enforce_invariants();
    }
}

/// The reason generation stopped.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end of turn.
    #[default]
    EndTurn,
    /// Hit the output-token limit.
    MaxTokens,
    /// Emitted a stop sequence (carried verbatim).
    StopSequence(String),
    /// Stopped to make tool calls.
    ToolUse,
    /// Stopped by a content filter.
    ContentFilter,
    /// Produced a refusal.
    Refusal,
    /// Paused mid-turn (Anthropic `pause_turn`).
    PauseTurn,
    /// Cancelled (background / cancel).
    Cancelled,
}

/// Which kind of item a streaming [`IrEvent::ItemStart`] opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    /// A message item (refusal is a [`Delta::Refusal`] on a message item).
    Message,
    /// A reasoning item. Its opaque replay carrier arrives as [`Delta::Opaque`]; text as
    /// [`Delta::ReasoningText`] / [`Delta::ReasoningSummary`].
    Reasoning,
    /// A function tool-call item.
    ToolCall,
    /// A provider-hosted tool-call item. Its payload arrives as [`Delta::ProviderRaw`].
    ProviderToolCall,
    /// A provider-hosted tool-result item. Its payload arrives as [`Delta::ProviderRaw`].
    ProviderToolResult,
    /// A provider compaction / context-management block (e.g. Anthropic `context_management`),
    /// aggregated back into [`Item::Compaction`]. Its opaque payload arrives as
    /// [`Delta::Opaque`] (`kind = Compaction`).
    Compaction,
}

/// An event in the internal streaming model. The response path is streaming-only inside the
/// crate; non-streaming providers become event lists, non-streaming clients use the
/// [`crate::aggregate::Aggregator`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IrEvent {
    /// Response opened. `usage_prefill` carries input usage if the provider reports it early.
    Start {
        /// Client-facing response id.
        response_id: ResponseId,
        /// Model string.
        model: String,
        /// Prefill (input) usage, if known at start.
        usage_prefill: Option<Usage>,
    },
    /// An item opened at `index`.
    ItemStart {
        /// Regrouped item index.
        index: u32,
        /// The kind of item.
        kind: ItemKind,
        /// Preserved / minted item id.
        id: Option<ItemId>,
        /// For a function [`ItemKind::ToolCall`] — and a [`ItemKind::ProviderToolCall`] that
        /// exposes one — the `(call_id, tool_name)` head. A provider item's full payload still
        /// arrives as [`Delta::ProviderRaw`]; this is just the correlation head when known.
        call: Option<(CallId, String)>,
    },
    /// A delta on the item at `index`.
    Delta {
        /// Item index this delta applies to.
        index: u32,
        /// The delta payload.
        delta: Delta,
    },
    /// The item at `index` closed.
    ItemStop {
        /// Item index that closed.
        index: u32,
    },
    /// Generation stopped, with final usage and response-level extensions.
    Stop {
        /// Why generation stopped.
        reason: StopReason,
        /// Final usage.
        usage: Usage,
        /// Response-level passthrough extensions that must reach the [`IrResponse`] `ext`
        /// unchanged: `stop_details` for a [`StopReason::Refusal`] (plan §5/§7 stop-reason
        /// table), an echoed `service_tier`, provider response metadata, and the like. A
        /// [`crate::codec::StreamDecoder`] emits it here so the streaming and non-streaming
        /// paths agree (`aggregate(stream) == aggregate(response)`, plan §11.8). Empty when
        /// there is none.
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        ext: Extensions,
    },
    /// A mid-stream error.
    Error(XlateError),
}

/// A streaming delta payload.
///
/// Externally tagged: several variants carry a bare `String`, which internal tagging cannot
/// represent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Delta {
    /// Assistant message text.
    Text(String),
    /// Tool-call argument text.
    ToolArgs(String),
    /// Full reasoning text.
    ReasoningText(String),
    /// A reasoning summary chunk on a given summary `part` index.
    ReasoningSummary {
        /// Summary part index (deltas with the same index concatenate).
        part: u32,
        /// The summary text chunk.
        text: String,
    },
    /// A refusal chunk (on a message item).
    Refusal(String),
    /// An opaque replay-carrier chunk for a **reasoning** item (`OpaqueKind::Signature` /
    /// `Redacted` / `Encrypted`) or a **compaction** item (`OpaqueKind::Compaction`). If a
    /// provider chunks a single carrier across several `Opaque` deltas, the
    /// [`crate::aggregate::Aggregator`] **concatenates** their `data` (keeping the `family` /
    /// `kind` / `model` of the first), so no chunk is lost.
    ///
    /// Provider-hosted tool payloads do **not** use this variant — they use
    /// [`Delta::ProviderRaw`], which carries structured JSON rather than an opaque string.
    Opaque(OpaqueBlob),
    /// The structured payload of a provider-hosted tool call/result item
    /// ([`ItemKind::ProviderToolCall`] / [`ItemKind::ProviderToolResult`]). Carries the
    /// provider `family` and the full provider JSON block verbatim (key order preserved).
    ///
    /// This is the faithful passthrough carrier for provider-hosted tools: the aggregator
    /// stores `raw` and `family` directly on the resulting [`OpaqueItem`], so nothing is
    /// serialized through a string or forced into an [`OpaqueKind`]. A decoder that receives
    /// provider JSON in text fragments (e.g. Anthropic `input_json_delta`) assembles them and
    /// emits one `ProviderRaw` with the complete value; if several arrive, the last one wins
    /// (a decoder emits the authoritative full block at item close).
    ///
    /// [`OpaqueItem`]: crate::ir::OpaqueItem
    /// [`OpaqueKind`]: crate::ir::OpaqueKind
    ProviderRaw {
        /// The provider family that owns this item.
        family: ProviderFamily,
        /// The full provider JSON block (kept verbatim; never re-sorted).
        raw: Value,
    },
    /// An annotation / citation.
    Annotation(Annotation),
}

/// The aggregated response, produced by the [`crate::aggregate::Aggregator`] from an event
/// stream, for non-streaming clients, store, and logging.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct IrResponse {
    /// Client-facing response id.
    pub id: ResponseId,
    /// Model string.
    pub model: String,
    /// The produced items in order.
    pub items: Vec<Item>,
    /// Why generation stopped.
    pub stop: StopReason,
    /// Final usage.
    pub usage: Usage,
    /// Passthrough extensions (e.g. `stop_details`).
    pub ext: Extensions,
}
