//! Transcript items and content parts: [`Item`], [`Part`], [`MediaSource`], and the
//! reasoning / opaque carriers.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::{bytes_b64, CallId, ItemId, JsonText, ProviderFamily};
use super::config::CacheControl;
use bytes::Bytes;

/// An atomic transcript unit. The IR is a flat list of items; Chat messages and Anthropic
/// turns are *bundles* that decode into items (lossless) and are regrouped on encode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    /// A user or assistant message. The assistant message never contains reasoning or tool
    /// calls; those are sibling items.
    Message {
        /// Message author.
        role: Role,
        /// Message content parts.
        content: Vec<Part>,
        /// Optional preserved item id.
        id: Option<ItemId>,
    },
    /// A reasoning / thinking item.
    Reasoning(ReasoningItem),
    /// A function tool call emitted by the assistant.
    ToolCall {
        /// Correlation id matching a later [`Item::ToolResult`].
        call_id: CallId,
        /// Tool name.
        name: String,
        /// Raw JSON argument text (kept as a string).
        arguments: JsonText,
        /// Optional preserved Responses item id.
        id: Option<ItemId>,
    },
    /// The result of a function tool call, supplied by the user side.
    ToolResult {
        /// Correlation id matching the [`Item::ToolCall`].
        call_id: CallId,
        /// Result content parts.
        content: Vec<Part>,
        /// Whether the tool reported an error.
        is_error: bool,
        /// Optional preserved item id.
        id: Option<ItemId>,
    },
    /// A provider-hosted tool call, passed through opaquely (assistant-side).
    ProviderToolCall(OpaqueItem),
    /// A provider-hosted tool result, passed through opaquely (assistant-side).
    ProviderToolResult(OpaqueItem),
    /// A provider compaction / context-management blob.
    Compaction(OpaqueBlob),
}

impl Item {
    /// A user message from plain text.
    pub fn user_text(text: impl Into<String>) -> Self {
        Item::Message { role: Role::User, content: vec![Part::text(text)], id: None }
    }
    /// An assistant message from plain text.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Item::Message { role: Role::Assistant, content: vec![Part::text(text)], id: None }
    }

    /// Whether this item belongs to an assistant-side run for regrouping purposes.
    ///
    /// Assistant-side: [`Item::Message`] with [`Role::Assistant`], [`Item::Reasoning`],
    /// [`Item::ToolCall`], [`Item::ProviderToolCall`], [`Item::ProviderToolResult`],
    /// [`Item::Compaction`]. User-side: user messages and [`Item::ToolResult`].
    pub fn is_assistant_side(&self) -> bool {
        match self {
            Item::Message { role, .. } => *role == Role::Assistant,
            Item::Reasoning(_)
            | Item::ToolCall { .. }
            | Item::ProviderToolCall(_)
            | Item::ProviderToolResult(_)
            | Item::Compaction(_) => true,
            Item::ToolResult { .. } => false,
        }
    }

    /// Whether this item is user-side (the complement of [`Item::is_assistant_side`]).
    pub fn is_user_side(&self) -> bool {
        !self.is_assistant_side()
    }
}

/// Message author.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The end user.
    User,
    /// The model.
    Assistant,
}

/// A content part inside a message or tool result.
///
/// Externally tagged (rather than `tag = "type"`): the `Image`/`Audio` variants wrap a
/// [`MediaSource`], which is itself an (externally tagged) enum — internal tagging cannot
/// flatten a nested tagged enum, so external tagging is the correct, round-trippable choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// Text, with optional annotations and a cache breakpoint.
    Text {
        /// The text.
        text: String,
        /// Citations / annotations attached to this text.
        annotations: Vec<Annotation>,
        /// Optional cache breakpoint after this block.
        cache_control: Option<CacheControl>,
    },
    /// An image.
    Image(MediaSource),
    /// A document (PDF or text).
    Document {
        /// Where the document bytes come from.
        source: MediaSource,
        /// Optional document title.
        title: Option<String>,
        /// MIME type (e.g. `application/pdf`, `text/plain`).
        media_type: String,
    },
    /// Audio input.
    Audio(MediaSource),
    /// A refusal string produced by the model.
    Refusal {
        /// The refusal text.
        text: String,
    },
    /// An opaque, provider-bound blob (e.g. redacted thinking carried in content).
    Opaque(OpaqueBlob),
}

impl Part {
    /// A plain text part with no annotations or cache control.
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text { text: text.into(), annotations: Vec::new(), cache_control: None }
    }
    /// If this is a text part, its text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Part::Text { text, .. } => Some(text),
            _ => None,
        }
    }
    /// Whether this is an empty text part (regrouping omits these).
    pub fn is_empty_text(&self) -> bool {
        matches!(self, Part::Text { text, .. } if text.is_empty())
    }
}

/// Where media bytes come from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaSource {
    /// Inline base64 bytes with a MIME type.
    Base64 {
        /// MIME type.
        media_type: String,
        /// The raw bytes (serialized as base64).
        #[serde(with = "bytes_b64")]
        data: Bytes,
    },
    /// A URL the provider fetches.
    Url(String),
    /// A provider-namespaced file id.
    FileRef {
        /// The family that minted the id.
        family: ProviderFamily,
        /// The file id.
        id: String,
    },
    /// Inline text (for text documents).
    Text(String),
}

/// A reasoning / thinking item.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReasoningItem {
    /// Full reasoning text, if exposed.
    pub text: Option<String>,
    /// Summary parts (one string per summary part index).
    pub summary: Vec<String>,
    /// Opaque carrier (signature / redacted / encrypted) for replay.
    pub opaque: Option<OpaqueBlob>,
    /// Optional preserved item id.
    pub id: Option<ItemId>,
}

/// An opaque, provider-bound blob: signed thinking, redacted thinking, encrypted reasoning,
/// or a compaction block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpaqueBlob {
    /// The provider family that produced (and can consume) this blob.
    pub family: ProviderFamily,
    /// What kind of blob this is.
    pub kind: OpaqueKind,
    /// The opaque payload (signature, base64 ciphertext, …).
    pub data: String,
    /// The producing model, if known — informational; used by the envelope's `m` field.
    pub model: Option<String>,
}

impl OpaqueBlob {
    /// Construct a blob with no known producing model.
    pub fn new(family: ProviderFamily, kind: OpaqueKind, data: impl Into<String>) -> Self {
        Self { family, kind, data: data.into(), model: None }
    }
}

/// The kind of an [`OpaqueBlob`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpaqueKind {
    /// An Anthropic thinking-block signature.
    Signature,
    /// Anthropic redacted-thinking data.
    Redacted,
    /// OpenAI encrypted reasoning content.
    Encrypted,
    /// A compaction / context-management blob.
    Compaction,
}

/// A provider-hosted tool call/result kept verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpaqueItem {
    /// The family that owns this item.
    pub family: ProviderFamily,
    /// The raw provider JSON (kept verbatim; `preserve_order` keeps key order).
    pub raw: Value,
}

impl OpaqueItem {
    /// Construct from a family and raw value.
    pub fn new(family: ProviderFamily, raw: Value) -> Self {
        Self { family, raw }
    }
}

/// A citation / annotation on a text part. Permissive: an opaque `kind` plus the raw
/// provider payload, so unknown annotation shapes round-trip losslessly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    /// The annotation kind string (e.g. `url_citation`, `file_citation`).
    pub kind: String,
    /// The raw provider payload (kept verbatim).
    pub raw: Value,
}

impl Annotation {
    /// Construct from a kind and raw payload.
    pub fn new(kind: impl Into<String>, raw: Value) -> Self {
        Self { kind: kind.into(), raw }
    }
}
