//! Error mapping (plan §10): Responses/OpenAI error bodies ⇄ [`XlateError`].

use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;

use llm_xlate_core::{
    canon, Capabilities, EncodedError, ErrorKind, HeaderMap, ProviderFamily, SseWriter, XlateError,
};

use crate::util::{get_str, Ob};

/// Decode a provider error body + status into the unified error.
pub(crate) fn decode_error(
    status: u16,
    body: &[u8],
    hdrs: &HeaderMap,
    _caps: &Capabilities,
) -> XlateError {
    let root = canon::parse(body).unwrap_or(Value::Null);
    let err_obj = root.get("error").unwrap_or(&Value::Null);

    let message = get_str(err_obj, "message")
        .or_else(|| get_str(&root, "message"))
        .unwrap_or("upstream error")
        .to_string();
    let ptype = get_str(err_obj, "type").map(str::to_string);
    let pcode = get_str(err_obj, "code").map(str::to_string);
    let param = get_str(err_obj, "param").map(str::to_string);

    let kind = kind_from(status, ptype.as_deref(), pcode.as_deref());

    let mut e = XlateError::new(kind, message)
        .with_status(status_for_kind(kind, status))
        .with_provider(ProviderFamily::OpenAI);
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

/// Build an [`XlateError`] from just an error object (used by `decode_response` on `failed`).
pub(crate) fn error_from_value(err_obj: &Value) -> XlateError {
    let message = get_str(err_obj, "message").unwrap_or("upstream error").to_string();
    let ptype = get_str(err_obj, "type").map(str::to_string);
    let pcode = get_str(err_obj, "code").map(str::to_string);
    let kind = kind_from(500, ptype.as_deref(), pcode.as_deref());
    let mut e = XlateError::new(kind, message).with_provider(ProviderFamily::OpenAI);
    if let Some(t) = ptype {
        e = e.with_provider_type(t);
    }
    if let Some(c) = pcode {
        e = e.with_provider_code(c);
    }
    e
}

/// Map an OpenAI (status, type, code) triple to an [`ErrorKind`].
fn kind_from(status: u16, ptype: Option<&str>, pcode: Option<&str>) -> ErrorKind {
    if pcode == Some("context_length_exceeded") {
        return ErrorKind::ContextLengthExceeded;
    }
    if pcode == Some("insufficient_quota") {
        return ErrorKind::Billing;
    }
    if ptype == Some("insufficient_quota") {
        return ErrorKind::Billing;
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
            _ => ErrorKind::ServerError,
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

/// The HTTP status a Responses (OpenAI-dialect) client must receive for a unified kind. Used on
/// the encode side so an error decoded from a *foreign* backend (e.g. an Anthropic `Overloaded`
/// carrying 529) is rendered with the OpenAI status (503), not the foreign dialect's status.
/// Same-protocol paths are unaffected: a Responses-decoded error's `e.status` already equals the
/// OpenAI-canonical status returned here.
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
        | ErrorKind::ContentFilter => "invalid_request_error",
        ErrorKind::Authentication => "authentication_error",
        ErrorKind::Permission => "permission_error",
        ErrorKind::NotFound => "not_found_error",
        ErrorKind::RequestTooLarge => "invalid_request_error",
        ErrorKind::RateLimited => "rate_limit_error",
        ErrorKind::Billing => "insufficient_quota",
        ErrorKind::Overloaded | ErrorKind::ServerError | ErrorKind::Timeout | ErrorKind::UpstreamMalformed => {
            "server_error"
        }
    }
}

/// The OpenAI `code` for a unified kind, when we have no provider-native code.
fn code_for(kind: ErrorKind) -> Option<&'static str> {
    match kind {
        ErrorKind::Authentication => Some("invalid_api_key"),
        ErrorKind::NotFound => Some("model_not_found"),
        ErrorKind::ContextLengthExceeded => Some("context_length_exceeded"),
        ErrorKind::Billing => Some("insufficient_quota"),
        ErrorKind::Unsupported | ErrorKind::IncompatibleHistory => Some("router_unsupported"),
        _ => None,
    }
}

/// The inner `{message,type,param,code}` error object [`Value`] (without the `{error: …}` wrap).
pub(crate) fn error_inner_value(e: &XlateError) -> Value {
    error_body_value(e).get("error").cloned().unwrap_or(Value::Null)
}

/// The `{error:{message,type,param,code}}` body [`Value`].
pub(crate) fn error_body_value(e: &XlateError) -> Value {
    let ty = e.provider_type.clone().unwrap_or_else(|| type_for(e.kind).to_string());
    let code = e.provider_code.clone().or_else(|| code_for(e.kind).map(str::to_string));
    let inner = Ob::new()
        .set("message", Value::from(e.message.clone()))
        .set("type", Value::from(ty))
        .set("param", e.param.clone().map(Value::from).unwrap_or(Value::Null))
        .set("code", code.map(Value::from).unwrap_or(Value::Null))
        .build();
    Ob::new().set("error", inner).build()
}

/// A streaming `error` event data payload.
fn error_event_value(e: &XlateError, seq: u64) -> Value {
    let code = e.provider_code.clone().or_else(|| code_for(e.kind).map(str::to_string));
    Ob::new()
        .set("type", "error".into())
        .set("code", code.map(Value::from).unwrap_or(Value::Null))
        .set("message", Value::from(e.message.clone()))
        .set("param", e.param.clone().map(Value::from).unwrap_or(Value::Null))
        .set("sequence_number", Value::from(seq))
        .build()
}

/// A minimal `response.failed` frame payload (used when no stream context is available).
fn failed_event_value(e: &XlateError, seq: u64) -> Value {
    let response = Ob::new()
        .set("id", Value::from(""))
        .set("object", "response".into())
        .set("status", "failed".into())
        .set("error", error_body_value(e).get("error").cloned().unwrap_or(Value::Null))
        .build();
    Ob::new()
        .set("type", "response.failed".into())
        .set("sequence_number", Value::from(seq))
        .set("response", response)
        .build()
}

/// Render a unified error into the client dialect.
pub(crate) fn encode_error(e: &XlateError, streaming: bool, started: bool) -> EncodedError {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().unwrap());
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
    if streaming {
        let data = if started {
            failed_event_value(e, 0)
        } else {
            error_event_value(e, 0)
        };
        let event = if started { "response.failed" } else { "error" };
        let frame = SseWriter::frame(Some(event), &canon::to_string(&data));
        EncodedError { status, headers, body: Bytes::new(), frames: vec![frame] }
    } else {
        let body = canon::to_bytes(&error_body_value(e));
        EncodedError { status, headers, body, frames: Vec::new() }
    }
}
