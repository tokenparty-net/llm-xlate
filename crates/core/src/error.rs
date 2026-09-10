//! The unified error type [`XlateError`] and its [`ErrorKind`] taxonomy (plan §10).
//!
//! Errors decoded from any provider land in this one space; each codec's `errors.rs` maps a
//! kind back into its wire dialect. `retry_after` serializes as integer milliseconds.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::ir::ProviderFamily;

/// The unified error taxonomy. `Unknown` capabilities and unsupported translations surface as
/// [`ErrorKind::Unsupported`] / [`ErrorKind::IncompatibleHistory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// Malformed or invalid request (400).
    InvalidRequest,
    /// Missing/invalid credentials (401).
    Authentication,
    /// Authenticated but not permitted (403).
    Permission,
    /// Resource (model, response id, …) not found (404).
    NotFound,
    /// Request body too large (413).
    RequestTooLarge,
    /// Prompt exceeds the context window (400 + pattern / `context_length_exceeded`).
    ContextLengthExceeded,
    /// Rate limited (429).
    RateLimited,
    /// Upstream overloaded (Anthropic 529 / OpenAI 503).
    Overloaded,
    /// Billing / quota problem.
    Billing,
    /// Blocked by a content filter.
    ContentFilter,
    /// Upstream server error (500).
    ServerError,
    /// Upstream timeout.
    Timeout,
    /// Upstream returned a body we could not parse.
    UpstreamMalformed,
    /// The translation is not supported for the resolved capabilities.
    Unsupported,
    /// The transcript cannot be represented for the target (e.g. unresolved reasoning).
    IncompatibleHistory,
}

impl ErrorKind {
    /// The canonical HTTP status for this kind, used when a status is not supplied. Codecs
    /// may re-map per dialect (e.g. Overloaded → 529 for Anthropic, 503 for OpenAI).
    pub fn default_status(self) -> u16 {
        match self {
            ErrorKind::InvalidRequest => 400,
            ErrorKind::Authentication => 401,
            ErrorKind::Permission => 403,
            ErrorKind::NotFound => 404,
            ErrorKind::RequestTooLarge => 413,
            ErrorKind::ContextLengthExceeded => 400,
            ErrorKind::RateLimited => 429,
            ErrorKind::Overloaded => 503,
            ErrorKind::Billing => 429,
            ErrorKind::ContentFilter => 400,
            ErrorKind::ServerError => 500,
            ErrorKind::Timeout => 504,
            ErrorKind::UpstreamMalformed => 502,
            ErrorKind::Unsupported => 400,
            ErrorKind::IncompatibleHistory => 400,
        }
    }

    /// Whether this kind is retryable by default (RateLimited, Overloaded, ServerError,
    /// Timeout). The [`XlateError::retryable`] field can override per-instance.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            ErrorKind::RateLimited
                | ErrorKind::Overloaded
                | ErrorKind::ServerError
                | ErrorKind::Timeout
        )
    }

    /// A short stable slug for this kind (`snake_case`).
    pub fn slug(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request",
            ErrorKind::Authentication => "authentication",
            ErrorKind::Permission => "permission",
            ErrorKind::NotFound => "not_found",
            ErrorKind::RequestTooLarge => "request_too_large",
            ErrorKind::ContextLengthExceeded => "context_length_exceeded",
            ErrorKind::RateLimited => "rate_limited",
            ErrorKind::Overloaded => "overloaded",
            ErrorKind::Billing => "billing",
            ErrorKind::ContentFilter => "content_filter",
            ErrorKind::ServerError => "server_error",
            ErrorKind::Timeout => "timeout",
            ErrorKind::UpstreamMalformed => "upstream_malformed",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::IncompatibleHistory => "incompatible_history",
        }
    }
}

/// The unified error. Carries the canonical [`ErrorKind`], an HTTP status, and enough
/// provider context to render a faithful client-facing error in any dialect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("[{kind:?} {status}] {message}")]
pub struct XlateError {
    /// The canonical kind.
    pub kind: ErrorKind,
    /// HTTP status to render.
    pub status: u16,
    /// Human-readable message.
    pub message: String,
    /// Originating provider family, if known.
    pub provider: Option<ProviderFamily>,
    /// Provider-native error `type` string.
    pub provider_type: Option<String>,
    /// Provider-native error `code` string.
    pub provider_code: Option<String>,
    /// The offending parameter, if any (rendered into OpenAI `param`).
    pub param: Option<String>,
    /// Whether the request may be retried.
    pub retryable: bool,
    /// `retry-after` hint (serialized as integer milliseconds).
    #[serde(with = "duration_millis")]
    pub retry_after: Option<Duration>,
    /// Upstream request id (surfaced as `x-router-upstream-request-id`).
    pub upstream_request_id: Option<String>,
}

impl XlateError {
    /// Construct an error of `kind` with `message`; status and retryability default from the
    /// kind.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            status: kind.default_status(),
            message: message.into(),
            provider: None,
            provider_type: None,
            provider_code: None,
            param: None,
            retryable: kind.is_retryable(),
            retry_after: None,
            upstream_request_id: None,
        }
    }

    /// An [`ErrorKind::InvalidRequest`].
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidRequest, message)
    }

    /// An [`ErrorKind::Unsupported`] with the offending `param` set.
    pub fn unsupported(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported, message).with_param(param)
    }

    /// An [`ErrorKind::IncompatibleHistory`].
    pub fn incompatible_history(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::IncompatibleHistory, message)
    }

    /// An [`ErrorKind::UpstreamMalformed`].
    pub fn upstream_malformed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UpstreamMalformed, message)
    }

    /// Set the offending parameter.
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
    /// Set the originating provider family.
    pub fn with_provider(mut self, provider: ProviderFamily) -> Self {
        self.provider = Some(provider);
        self
    }
    /// Override the HTTP status.
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = status;
        self
    }
    /// Set the provider-native error `type`.
    pub fn with_provider_type(mut self, provider_type: impl Into<String>) -> Self {
        self.provider_type = Some(provider_type.into());
        self
    }
    /// Set the provider-native error `code`.
    pub fn with_provider_code(mut self, provider_code: impl Into<String>) -> Self {
        self.provider_code = Some(provider_code.into());
        self
    }
    /// Override retryability.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
    /// Set a `retry-after` hint.
    pub fn with_retry_after(mut self, retry_after: Duration) -> Self {
        self.retry_after = Some(retry_after);
        self
    }
    /// Set the upstream request id.
    pub fn with_upstream_request_id(mut self, id: impl Into<String>) -> Self {
        self.upstream_request_id = Some(id.into());
        self
    }
}

/// serde helper: `Option<Duration>` <-> integer milliseconds.
mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(d) => s.serialize_some(&(d.as_millis() as u64)),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        let ms = Option::<u64>::deserialize(d)?;
        Ok(ms.map(Duration::from_millis))
    }
}
