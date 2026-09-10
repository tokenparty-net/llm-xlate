//! Shared IR primitives: string newtypes, provider families, protocols, and the
//! base64 serde helper used for [`bytes::Bytes`] fields.

use serde::{Deserialize, Serialize};

/// Declare a transparent `String` newtype with the standard ergonomic impls
/// (`Deref<Target=str>`, `From<String>`/`From<&str>`, `Display`, `AsRef<str>`).
macro_rules! string_newtype {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Construct from anything string-like.
            pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
            /// Borrow the inner string.
            pub fn as_str(&self) -> &str { &self.0 }
            /// Consume and return the inner `String`.
            pub fn into_inner(self) -> String { self.0 }
        }
        impl core::ops::Deref for $name {
            type Target = str;
            fn deref(&self) -> &str { &self.0 }
        }
        impl From<String> for $name { fn from(s: String) -> Self { Self(s) } }
        impl From<&str> for $name { fn from(s: &str) -> Self { Self(s.to_owned()) } }
        impl From<$name> for String { fn from(v: $name) -> String { v.0 } }
        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl AsRef<str> for $name { fn as_ref(&self) -> &str { &self.0 } }
    };
}

string_newtype!(
    /// A Responses-style output item id. Preserved verbatim from the client; the only
    /// minted ids are Resp item ids when acting as a Resp server (`{kind}_{response_id}_{index}`).
    ItemId
);
string_newtype!(
    /// A tool-call correlation id, shared between a [`crate::ir::Item::ToolCall`] and its
    /// matching [`crate::ir::Item::ToolResult`]. Preserved verbatim from the client.
    CallId
);
string_newtype!(
    /// A response id (router-minted client-facing id, or an upstream one).
    ResponseId
);
string_newtype!(
    /// Raw JSON text for tool-call arguments. Kept as a **string** and only parsed when a
    /// target needs an object (Anthropic `input`); parsing uses `preserve_order`.
    JsonText
);

/// Provider family — coarser than a wire protocol; determines native compatibility of
/// opaque reasoning blobs, provider tools, and file ids.
///
/// Serializes as a bare lowercase string (`"anthropic"`, `"openai"`, or the custom label),
/// so it reads naturally in TOML (`family = "openai-compatible"`) and in IR blobs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProviderFamily {
    /// Anthropic (Claude / Fable / Mythos / Opus).
    Anthropic,
    /// OpenAI (GPT / o-series).
    OpenAI,
    /// Any other family (third-party OpenAI-compatible servers, gateways, …); the string
    /// is an opaque family label chosen by the router.
    Other(String),
}

impl ProviderFamily {
    /// A stable lowercase label for this family (`"anthropic"`, `"openai"`, or the custom string).
    pub fn label(&self) -> &str {
        match self {
            ProviderFamily::Anthropic => "anthropic",
            ProviderFamily::OpenAI => "openai",
            ProviderFamily::Other(s) => s.as_str(),
        }
    }

    /// Parse a family from its label.
    pub fn from_label(s: &str) -> ProviderFamily {
        match s {
            "anthropic" => ProviderFamily::Anthropic,
            "openai" => ProviderFamily::OpenAI,
            other => ProviderFamily::Other(other.to_string()),
        }
    }
}

impl Serialize for ProviderFamily {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.label())
    }
}

impl<'de> Deserialize<'de> for ProviderFamily {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(ProviderFamily::from_label(&s))
    }
}

impl core::fmt::Display for ProviderFamily {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
}

/// A concrete wire protocol (dialect) spoken on either side of a translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// OpenAI Chat Completions (`/v1/chat/completions`).
    OaiChat,
    /// OpenAI Responses (`/v1/responses`).
    OaiResponses,
    /// Anthropic Messages (`/v1/messages`).
    Anthropic,
}

impl Protocol {
    /// The [`ProviderFamily`] this protocol belongs to.
    pub fn family(&self) -> ProviderFamily {
        match self {
            Protocol::OaiChat | Protocol::OaiResponses => ProviderFamily::OpenAI,
            Protocol::Anthropic => ProviderFamily::Anthropic,
        }
    }

    /// Whether this is one of the two OpenAI dialects.
    pub fn is_openai(&self) -> bool {
        matches!(self, Protocol::OaiChat | Protocol::OaiResponses)
    }
}

/// Serialize/deserialize a [`bytes::Bytes`] field as a base64 (standard alphabet, padded)
/// string. IR bytes are our own storage/logging format, so base64 keeps them JSON-safe.
pub(crate) mod bytes_b64 {
    use base64::Engine;
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(b: &Bytes, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(b))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Bytes, D::Error> {
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(s.as_bytes())
            .map(Bytes::from)
            .map_err(serde::de::Error::custom)
    }
}
