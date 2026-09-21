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

    /// The kind a bare HTTP status implies, for an upstream error whose body carries no
    /// dialect-typed error (a proxy/framework error page, FastAPI's `{"detail":…}`, …). Any
    /// unlisted 4xx is the caller's fault ([`ErrorKind::InvalidRequest`]); anything else is
    /// [`ErrorKind::ServerError`].
    pub fn from_status(status: u16) -> Self {
        match status {
            401 => ErrorKind::Authentication,
            402 => ErrorKind::Billing,
            403 => ErrorKind::Permission,
            404 => ErrorKind::NotFound,
            413 => ErrorKind::RequestTooLarge,
            429 => ErrorKind::RateLimited,
            503 | 529 => ErrorKind::Overloaded,
            504 => ErrorKind::Timeout,
            400..=499 => ErrorKind::InvalidRequest,
            _ => ErrorKind::ServerError,
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

/// Longest raw upstream body excerpt [`upstream_error_message`] returns.
const MAX_RAW_ERROR_MESSAGE_CHARS: usize = 500;

/// A human-readable message for an upstream error body that is **not** in a codec's own error
/// envelope, so a client sees why the upstream refused instead of a bare "upstream error".
///
/// Tries, in order: `error.message`, a string `error`, `message`, a string `detail`, and
/// FastAPI/pydantic validation `detail: [{loc, msg}]` (rendered `body.thinking.type: msg`,
/// `; `-joined). Otherwise the raw body, trimmed and truncated. `None` for an empty body.
pub fn upstream_error_message(body: &[u8]) -> Option<String> {
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) {
        let str_at = |v: &serde_json::Value| v.as_str().map(str::to_string);
        let found = v
            .pointer("/error/message")
            .and_then(str_at)
            .or_else(|| v.get("error").and_then(str_at))
            .or_else(|| v.get("message").and_then(str_at))
            .or_else(|| v.get("detail").and_then(str_at))
            .or_else(|| v.get("detail").and_then(|d| d.as_array()).and_then(|items| validation_detail(items)));
        if let Some(m) = found.filter(|m| !m.trim().is_empty()) {
            return Some(m);
        }
    }
    let raw = String::from_utf8_lossy(body);
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if raw.chars().count() <= MAX_RAW_ERROR_MESSAGE_CHARS {
        return Some(raw.to_string());
    }
    let mut cut: String = raw.chars().take(MAX_RAW_ERROR_MESSAGE_CHARS).collect();
    cut.push('…');
    Some(cut)
}

/// Render FastAPI/pydantic `detail: [{loc: [...], msg}]` as `loc.path: msg; …`.
fn validation_detail(items: &[serde_json::Value]) -> Option<String> {
    let parts: Vec<String> = items
        .iter()
        .filter_map(|item| {
            let msg = item.get("msg")?.as_str()?;
            let loc = item
                .get("loc")
                .and_then(|l| l.as_array())
                .map(|l| {
                    l.iter()
                        .map(|p| p.as_str().map(str::to_string).unwrap_or_else(|| p.to_string()))
                        .collect::<Vec<_>>()
                        .join(".")
                })
                .filter(|l| !l.is_empty());
            Some(match loc {
                Some(loc) => format!("{loc}: {msg}"),
                None => msg.to_string(),
            })
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join("; "))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_status_maps_unlisted_4xx_to_invalid_request() {
        assert_eq!(ErrorKind::from_status(400), ErrorKind::InvalidRequest);
        assert_eq!(ErrorKind::from_status(422), ErrorKind::InvalidRequest);
        assert_eq!(ErrorKind::from_status(401), ErrorKind::Authentication);
        assert_eq!(ErrorKind::from_status(429), ErrorKind::RateLimited);
        assert_eq!(ErrorKind::from_status(529), ErrorKind::Overloaded);
        assert_eq!(ErrorKind::from_status(504), ErrorKind::Timeout);
        assert_eq!(ErrorKind::from_status(500), ErrorKind::ServerError);
        assert_eq!(ErrorKind::from_status(502), ErrorKind::ServerError);
    }

    #[test]
    fn message_from_fastapi_validation_detail() {
        let body = br#"{"detail":[{"type":"literal_error","loc":["body","thinking","type"],"msg":"Input should be 'enabled' or 'disabled'","input":"adaptive"}]}"#;
        assert_eq!(
            upstream_error_message(body).as_deref(),
            Some("body.thinking.type: Input should be 'enabled' or 'disabled'")
        );
    }

    #[test]
    fn message_from_common_shapes() {
        assert_eq!(upstream_error_message(br#"{"detail":"nope"}"#).as_deref(), Some("nope"));
        assert_eq!(upstream_error_message(br#"{"error":"bad key"}"#).as_deref(), Some("bad key"));
        assert_eq!(upstream_error_message(br#"{"message":"m"}"#).as_deref(), Some("m"));
        assert_eq!(upstream_error_message(br#"{"error":{"message":"inner"}}"#).as_deref(), Some("inner"));
        assert_eq!(upstream_error_message(b"  <html>bad</html>
").as_deref(), Some("<html>bad</html>"));
        assert_eq!(upstream_error_message(b"   "), None);
    }

    #[test]
    fn raw_message_is_truncated() {
        let body = "x".repeat(MAX_RAW_ERROR_MESSAGE_CHARS + 50);
        let m = upstream_error_message(body.as_bytes()).unwrap();
        assert_eq!(m.chars().count(), MAX_RAW_ERROR_MESSAGE_CHARS + 1);
        assert!(m.ends_with('…'));
    }
}
