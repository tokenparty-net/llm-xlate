//! Error mapping between Anthropic's wire error dialect and the unified [`XlateError`]
//! (plan §10).
//!
//! Anthropic error bodies are `{type:"error", error:{type, message}, request_id?}` with a
//! small closed set of `error.type` strings. `decode_error` maps them (plus the HTTP status,
//! `retry-after`, and `request-id` headers) into an [`XlateError`]; `encode_error` renders an
//! [`XlateError`] back into that dialect (non-streaming body, or a streaming `event: error`
//! frame).

use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;

use llm_xlate_core::{
    canon, Capabilities, EncodedError, ErrorKind, HeaderMap, SseWriter, XlateError,
};

use crate::wire::{omap, FAMILY};

/// Map an Anthropic `error.type` string to an [`ErrorKind`].
fn kind_from_type(t: &str) -> ErrorKind {
    match t {
        "invalid_request_error" => ErrorKind::InvalidRequest,
        "authentication_error" => ErrorKind::Authentication,
        "permission_error" => ErrorKind::Permission,
        "not_found_error" => ErrorKind::NotFound,
        "request_too_large" => ErrorKind::RequestTooLarge,
        "rate_limit_error" => ErrorKind::RateLimited,
        "overloaded_error" => ErrorKind::Overloaded,
        "billing_error" => ErrorKind::Billing,
        "api_error" => ErrorKind::ServerError,
        _ => ErrorKind::ServerError,
    }
}

/// Map an [`ErrorKind`] to the Anthropic `error.type` string.
fn type_from_kind(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidRequest
        | ErrorKind::ContextLengthExceeded
        | ErrorKind::ContentFilter
        | ErrorKind::Unsupported
        | ErrorKind::IncompatibleHistory => "invalid_request_error",
        ErrorKind::Authentication => "authentication_error",
        ErrorKind::Permission => "permission_error",
        ErrorKind::NotFound => "not_found_error",
        ErrorKind::RequestTooLarge => "request_too_large",
        ErrorKind::RateLimited => "rate_limit_error",
        ErrorKind::Overloaded => "overloaded_error",
        ErrorKind::Billing => "billing_error",
        ErrorKind::ServerError | ErrorKind::Timeout | ErrorKind::UpstreamMalformed => "api_error",
    }
}

/// The Anthropic-canonical HTTP status for an [`ErrorKind`] (plan §10 "to Ant" column):
/// e.g. Overloaded → 529 and Billing → 400 (`billing_error`), regardless of the status a
/// foreign decode carried. Kinds the table does not pin (Timeout / UpstreamMalformed) keep
/// their carried status.
fn status_from_kind(kind: ErrorKind, fallback: u16) -> u16 {
    match kind {
        ErrorKind::InvalidRequest
        | ErrorKind::ContextLengthExceeded
        | ErrorKind::ContentFilter
        | ErrorKind::Unsupported
        | ErrorKind::IncompatibleHistory
        | ErrorKind::Billing => 400,
        ErrorKind::Authentication => 401,
        ErrorKind::Permission => 403,
        ErrorKind::NotFound => 404,
        ErrorKind::RequestTooLarge => 413,
        ErrorKind::RateLimited => 429,
        ErrorKind::Overloaded => 529,
        ErrorKind::ServerError => 500,
        ErrorKind::Timeout | ErrorKind::UpstreamMalformed => fallback,
    }
}

/// Build an [`XlateError`] from an Anthropic streaming `error` event's `{type, message}`.
pub fn error_from_wire(err: &Value) -> XlateError {
    let ty = err.get("type").and_then(Value::as_str).unwrap_or("api_error");
    let msg = err.get("message").and_then(Value::as_str).unwrap_or("upstream error");
    let kind = kind_from_type(ty);
    XlateError::new(kind, msg).with_provider(FAMILY).with_provider_type(ty)
}

/// Decode an Anthropic error response (status + body + headers) into an [`XlateError`].
pub fn decode_error(status: u16, body: &[u8], hdrs: &HeaderMap, caps: &Capabilities) -> XlateError {
    let value = canon::parse_upstream(body).unwrap_or(Value::Null);
    let err = value.get("error");
    let ty = err
        .and_then(|e| e.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("api_error")
        .to_string();
    let message = err
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("upstream error")
        .to_string();

    let mut kind = kind_from_type(&ty);

    // Context-length detection: an invalid_request_error whose message matches a configured
    // pattern is reclassified as ContextLengthExceeded.
    if kind == ErrorKind::InvalidRequest {
        if let Some(patterns) = &caps.errors.context_length_patterns {
            let lower = message.to_lowercase();
            if patterns.iter().any(|p| lower.contains(&p.to_lowercase())) {
                kind = ErrorKind::ContextLengthExceeded;
            }
        }
    }

    let retryable = caps
        .errors
        .retryable_status
        .as_ref()
        .map(|s| s.contains(&status))
        .unwrap_or_else(|| kind.is_retryable());

    let mut e = XlateError::new(kind, message)
        .with_status(status)
        .with_provider(FAMILY)
        .with_provider_type(ty)
        .with_retryable(retryable);

    // request id: body `request_id` or the `request-id` header.
    let request_id = value
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| header_str(hdrs, "request-id"));
    if let Some(id) = request_id {
        e = e.with_upstream_request_id(id);
    }

    // retry-after (seconds).
    if let Some(ra) = header_str(hdrs, "retry-after").and_then(|s| s.trim().parse::<u64>().ok()) {
        e = e.with_retry_after(Duration::from_secs(ra));
    }

    e
}

/// Render an [`XlateError`] into the Anthropic wire dialect.
pub fn encode_error(e: &XlateError, streaming: bool, _started: bool) -> EncodedError {
    let ty = e
        .provider_type
        .clone()
        .filter(|_| e.provider.as_ref() == Some(&FAMILY))
        .unwrap_or_else(|| type_from_kind(e.kind).to_string());

    // Router-originated rejections carry no distinct Anthropic `error.type` (they map to
    // `invalid_request_error`, same as a genuine upstream 400) and the dialect has no `code`
    // field, so prefix the message with `router: ` to let a Claude client tell a router-side
    // rejection from an upstream one (plan §10; mirrors the OpenAI `router_unsupported` code).
    let message = match e.kind {
        ErrorKind::Unsupported | ErrorKind::IncompatibleHistory
            if !e.message.starts_with("router:") =>
        {
            format!("router: {}", e.message)
        }
        _ => e.message.clone(),
    };

    let mut error_obj = omap();
    error_obj.insert("type".into(), Value::from(ty));
    error_obj.insert("message".into(), Value::from(message));

    let mut root = omap();
    root.insert("type".into(), Value::from("error"));
    root.insert("error".into(), Value::Object(error_obj));
    if let Some(id) = &e.upstream_request_id {
        root.insert("request_id".into(), Value::from(id.clone()));
    }
    let body_value = Value::Object(root);
    let body: Bytes = canon::to_bytes(&body_value);

    let status = status_from_kind(e.kind, e.status);
    let headers = error_headers(e);

    if streaming {
        let frame = SseWriter::frame(Some("error"), &canon::to_string(&body_value));
        EncodedError { status, headers, body: Bytes::new(), frames: vec![frame] }
    } else {
        EncodedError { status, headers, body, frames: Vec::new() }
    }
}

/// Build the response headers (`retry-after`, `x-router-upstream-request-id`) for a rendered error.
fn error_headers(e: &XlateError) -> HeaderMap {
    use http::header::{HeaderName, HeaderValue};
    let mut headers = HeaderMap::new();
    if let Some(ra) = e.retry_after {
        if let Ok(v) = HeaderValue::from_str(&ra.as_secs().to_string()) {
            headers.insert(HeaderName::from_static("retry-after"), v);
        }
    }
    if let Some(id) = &e.upstream_request_id {
        if let Ok(v) = HeaderValue::from_str(id) {
            headers.insert(HeaderName::from_static("x-router-upstream-request-id"), v);
        }
    }
    headers
}

/// Read a header as a `String` (first value only).
fn header_str(hdrs: &HeaderMap, name: &str) -> Option<String> {
    hdrs.get(name).and_then(|v| v.to_str().ok()).map(str::to_string)
}
