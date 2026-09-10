//! The intermediate representation (IR): a protocol-neutral, item-based, stateless model of
//! a request, a streaming response, and an aggregated response (plan §5).
//!
//! The IR is *item-based* (Responses-shaped): Chat messages and Anthropic turns are bundles
//! that flatten into [`Item`]s on decode and are regrouped on encode. All IR types derive
//! `Debug + Clone + PartialEq + Serialize + Deserialize` with a fixed field order so that
//! serialization is canonical. Passthrough payloads (schemas, `ext`, provider items) use
//! `serde_json::Value` (with the `preserve_order` feature) rather than `RawValue`, because
//! `Value` is `PartialEq` (needed by the round-trip laws) and still preserves object key
//! order. [`bytes::Bytes`] fields serialize as base64 strings.

mod common;
mod config;
mod event;
mod item;

pub use common::{CallId, ItemId, JsonText, Protocol, ProviderFamily, ResponseId};
pub use config::{
    CacheControl, CacheHints, CacheTtl, ClearAt, Effort, Extensions, Instruction,
    InstructionRole, IrRequest, Limits, ModelRef, OutputConfig, OutputFormat, Position,
    ReasoningConfig, ReasoningExposure, RequestMeta, Sampling, StateConfig, SummaryLevel,
    ToolChoice, ToolDef, Verbosity,
};
pub use event::{Delta, IrEvent, IrResponse, ItemKind, StopReason, Usage};
pub use item::{
    Annotation, Item, MediaSource, OpaqueBlob, OpaqueItem, OpaqueKind, Part, ReasoningItem, Role,
};

#[cfg(test)]
mod tests;
