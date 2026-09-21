//! Error mapping (plan §10): Chat Completions / OpenAI error bodies ⇄ [`XlateError`].
//!
//! Decode maps an upstream `{error:{message,type,param,code}}` body plus HTTP status into the
//! unified [`XlateError`] using `status` + `code`/`type` (`invalid_api_key`, `model_not_found`,
//! `context_length_exceeded`, `insufficient_quota`, `rate_limit_exceeded`, `server_error`, …),
//! honouring `retry-after` and `x-request-id`. Encode renders a unified error back into the
//! Chat dialect: a JSON `{error:{…}}` body, or (streaming) an error data frame followed by
//! `data: [DONE]`.

use std::time::Duration;

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::{
    canon, upstream_error_message, Capabilities, EncodedError, ErrorKind, HeaderMap, ProviderFamily, SseWriter, XlateError,
};

/// Decode a provider error body + status into the unified error.
pub fn decode_error(status: u16, body: &[u8], hdrs: &HeaderMap, _caps: &Capabilities) -> XlateError {
    let root = canon::parse(body).unwrap_or(Value::Null);
    let err_obj = root.get("error").unwrap_or(&Value::Null);

    let message = str_field(err_obj, "message")
        .or_else(|| str_field(&root, "message"))
        .map(str::to_string)
        .or_else(|| upstream_error_message(body))
        .unwrap_or_else(|| "upstream error".to_string());
    let ptype = str_field(err_obj, "type").map(str::to_string);
    let pcode = str_field(err_obj, "code").map(str::to_string);
    let param = str_field(err_obj, "param").map(str::to_string);

    let kind = kind_from(status, ptype.as_deref(), pcode.as_deref());

    let mut e = XlateError::new(kind, message)
        .with_status(status_for_kind(kind, status))
        .with_provider(ProviderFamily::OpenAI)
        .with_retryable(kind.is_retryable());
    if let Some(t) = ptype {
        e = e.with_provider_type(t);
    }
    if let Some(c) = pcode {
        e = e.with_provider_code(c);
    }
    if let Some(p) = param {
        e = e.with_param(p);
    }
    if let Some(ra) = retry_after(hdrs) {
        e = e.with_retry_after(ra);
    }
    if let Some(rid) = hdrs.get("x-request-id").and_then(|v| v.to_str().ok()) {
        e = e.with_upstream_request_id(rid);
    }
    e
}

/// Build an [`XlateError`] from just an error object (used by `decode_response`).
pub fn error_from_value(err_obj: &Value) -> XlateError {
    let message = str_field(err_obj, "message").unwrap_or("upstream error").to_string();
    let ptype = str_field(err_obj, "type").map(str::to_string);
    let pcode = str_field(err_obj, "code").map(str::to_string);
    let kind = kind_from(500, ptype.as_deref(), pcode.as_deref());
    let mut e = XlateError::new(kind, message)
        .with_provider(ProviderFamily::OpenAI)
        .with_status(e_status(kind));
    if let Some(t) = ptype {
        e = e.with_provider_type(t);
    }
    if let Some(c) = pcode {
        e = e.with_provider_code(c);
    }
    e
}

fn e_status(kind: ErrorKind) -> u16 {
    match kind {
        ErrorKind::Overloaded => 503,
        _ => kind.default_status(),
    }
}

/// Map an OpenAI (status, type, code) triple to an [`ErrorKind`].
fn kind_from(status: u16, ptype: Option<&str>, pcode: Option<&str>) -> ErrorKind {
    if pcode == Some("context_length_exceeded") {
        return ErrorKind::ContextLengthExceeded;
    }
    if pcode == Some("insufficient_quota") || ptype == Some("insufficient_quota") {
        return ErrorKind::Billing;
    }
    if pcode == Some("invalid_api_key") {
        return ErrorKind::Authentication;
    }
    if pcode == Some("model_not_found") {
        return ErrorKind::NotFound;
    }
    if pcode == Some("rate_limit_exceeded") {
        return ErrorKind::RateLimited;
    }
    match status {
        400 => match ptype {
            Some("content_filter") => ErrorKind::ContentFilter,
            _ => ErrorKind::InvalidRequest,
        },
        401 => ErrorKind::Authentication,
        403 => ErrorKind::Permission,
        404 => ErrorKind::NotFound,
        413 => ErrorKind::RequestTooLarge,
        429 => ErrorKind::RateLimited,
        500 | 502 => ErrorKind::ServerError,
        503 => ErrorKind::Overloaded,
        504 => ErrorKind::Timeout,
        _ => match ptype {
            Some("invalid_request_error") => ErrorKind::InvalidRequest,
            Some("authentication_error") => ErrorKind::Authentication,
            Some("rate_limit_error") => ErrorKind::RateLimited,
            Some("server_error") => ErrorKind::ServerError,
            // Untyped and unlisted (e.g. a framework 422): classify by status class.
            _ => ErrorKind::from_status(status),
        },
    }
}

/// OpenAI renders Overloaded as 503; otherwise keep the incoming status when it is set.
fn status_for_kind(kind: ErrorKind, incoming: u16) -> u16 {
    match kind {
        ErrorKind::Overloaded => 503,
        _ if incoming >= 400 => incoming,
        _ => kind.default_status(),
    }
}

fn retry_after(hdrs: &HeaderMap) -> Option<Duration> {
    let v = hdrs.get("retry-after")?.to_str().ok()?;
    v.parse::<u64>().ok().map(Duration::from_secs)
}

/// The OpenAI `type` string for a unified kind (when the provider type is unknown).
fn type_for(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidRequest
        | ErrorKind::Unsupported
        | ErrorKind::IncompatibleHistory
        | ErrorKind::ContextLengthExceeded
        | ErrorKind::RequestTooLarge
        | ErrorKind::ContentFilter => "invalid_request_error",
        ErrorKind::Authentication => "authentication_error",
        ErrorKind::Permission => "permission_error",
        ErrorKind::NotFound => "not_found_error",
        // OpenAI uses a distinct `insufficient_quota` type for quota exhaustion (a non-retryable
        // failure); only genuine rate limiting is `rate_limit_error`. Matches the Responses
        // dialect so the two OpenAI surfaces agree (plan §10).
        ErrorKind::RateLimited => "rate_limit_error",
        ErrorKind::Billing => "insufficient_quota",
        ErrorKind::Overloaded
        | ErrorKind::ServerError
        | ErrorKind::Timeout
        | ErrorKind::UpstreamMalformed => "server_error",
    }
}

/// The HTTP status a Chat (OpenAI-dialect) client must receive for a unified kind. Used on the
/// encode side so an error decoded from a *foreign* backend (e.g. an Anthropic `Overloaded`
/// carrying 529) is rendered with the OpenAI status (503), not the foreign dialect's status.
/// Mirrors the Responses dialect's `out_status` so the two OpenAI surfaces agree (plan §10).
fn out_status(kind: ErrorKind) -> u16 {
    match kind {
        ErrorKind::Overloaded => 503,
        ErrorKind::RateLimited | ErrorKind::Billing => 429,
        ErrorKind::Authentication => 401,
        ErrorKind::Permission => 403,
        ErrorKind::NotFound => 404,
        ErrorKind::RequestTooLarge => 413,
        ErrorKind::Unsupported
        | ErrorKind::IncompatibleHistory
        | ErrorKind::ContextLengthExceeded
        | ErrorKind::ContentFilter
        | ErrorKind::InvalidRequest => 400,
        ErrorKind::ServerError | ErrorKind::UpstreamMalformed => 500,
        ErrorKind::Timeout => 504,
    }
}

/// The OpenAI `code` for a unified kind, when we have no provider-native code.
fn code_for(kind: ErrorKind) -> Option<&'static str> {
    match kind {
        ErrorKind::Authentication => Some("invalid_api_key"),
        ErrorKind::NotFound => Some("model_not_found"),
        ErrorKind::ContextLengthExceeded => Some("context_length_exceeded"),
        ErrorKind::Billing => Some("insufficient_quota"),
        // No `RateLimited` fallback: a real OpenAI rate limit carries its own provider code
        // (preserved as `provider_code`); with none, the code is null, matching the Responses
        // dialect (plan §10).
        ErrorKind::Unsupported | ErrorKind::IncompatibleHistory => Some("router_unsupported"),
        _ => None,
    }
}

/// The `{error:{message,type,param,code}}` body [`Value`] (fixed key order).
pub fn error_body_value(e: &XlateError) -> Value {
    let ty = e.provider_type.clone().unwrap_or_else(|| type_for(e.kind).to_string());
    let code = e.provider_code.clone().or_else(|| code_for(e.kind).map(str::to_string));
    let mut inner = Map::new();
    inner.insert("message".into(), Value::from(e.message.clone()));
    inner.insert("type".into(), Value::from(ty));
    inner.insert("param".into(), e.param.clone().map(Value::from).unwrap_or(Value::Null));
    inner.insert("code".into(), code.map(Value::from).unwrap_or(Value::Null));
    let mut root = Map::new();
    root.insert("error".into(), Value::Object(inner));
    Value::Object(root)
}

/// Render a unified error into the Chat client dialect. Streaming errors are an error data
/// frame followed by `data: [DONE]`; `started` does not change the Chat rendering.
pub fn encode_error(e: &XlateError, streaming: bool, _started: bool) -> EncodedError {
    let mut headers = HeaderMap::new();
    if let Ok(ct) = "application/json".parse() {
        headers.insert("content-type", ct);
    }
    if let Some(ra) = e.retry_after {
        if let Ok(v) = ra.as_secs().to_string().parse() {
            headers.insert("retry-after", v);
        }
    }
    if let Some(rid) = &e.upstream_request_id {
        if let Ok(v) = rid.parse() {
            headers.insert("x-router-upstream-request-id", v);
        }
    }

    let status = out_status(e.kind);
    let body_val = error_body_value(e);
    if streaming {
        let frame = SseWriter::frame(None, &canon::to_string(&body_val));
        let done = SseWriter::frame(None, "[DONE]");
        EncodedError { status, headers, body: Bytes::new(), frames: vec![frame, done] }
    } else {
        EncodedError { status, headers, body: canon::to_bytes(&body_val), frames: Vec::new() }
    }
}

fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}
