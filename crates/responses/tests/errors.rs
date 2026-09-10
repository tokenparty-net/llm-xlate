//! Error dialect mapping (plan §10): decode_error / encode_error.

mod common;
use common::*;

use llm_xlate_core::{Codec, ErrorKind, HeaderMap, XlateError};
use pretty_assertions::assert_eq;

fn decode(status: u16, body: &str, hdrs: &HeaderMap) -> XlateError {
    codec().decode_error(status, body.as_bytes(), hdrs, &caps())
}

fn err(status: u16, ty: &str, code: Option<&str>) -> XlateError {
    let code_field = code.map(|c| format!(r#","code":"{c}""#)).unwrap_or_default();
    let body = format!(r#"{{"error":{{"message":"m","type":"{ty}"{code_field}}}}}"#);
    decode(status, &body, &HeaderMap::new())
}

#[test]
fn maps_400_invalid_request() {
    assert_eq!(err(400, "invalid_request_error", None).kind, ErrorKind::InvalidRequest);
}

#[test]
fn maps_401_403_404_413_429() {
    assert_eq!(err(401, "authentication_error", None).kind, ErrorKind::Authentication);
    assert_eq!(err(403, "permission_error", None).kind, ErrorKind::Permission);
    assert_eq!(err(404, "not_found_error", None).kind, ErrorKind::NotFound);
    assert_eq!(err(413, "request_too_large", None).kind, ErrorKind::RequestTooLarge);
    assert_eq!(err(429, "rate_limit_error", None).kind, ErrorKind::RateLimited);
}

#[test]
fn maps_5xx() {
    assert_eq!(err(500, "server_error", None).kind, ErrorKind::ServerError);
    assert_eq!(err(503, "server_error", None).kind, ErrorKind::Overloaded);
    assert_eq!(err(504, "timeout", None).kind, ErrorKind::Timeout);
}

#[test]
fn maps_context_length_and_billing_and_content_filter() {
    assert_eq!(err(400, "invalid_request_error", Some("context_length_exceeded")).kind, ErrorKind::ContextLengthExceeded);
    assert_eq!(err(429, "insufficient_quota", Some("insufficient_quota")).kind, ErrorKind::Billing);
    assert_eq!(err(400, "content_filter", None).kind, ErrorKind::ContentFilter);
}

#[test]
fn preserves_provider_fields() {
    let e = err(400, "invalid_request_error", Some("bad_param"));
    assert_eq!(e.provider, Some(llm_xlate_core::ProviderFamily::OpenAI));
    assert_eq!(e.provider_type.as_deref(), Some("invalid_request_error"));
    assert_eq!(e.provider_code.as_deref(), Some("bad_param"));
    assert_eq!(e.message, "m");
}

#[test]
fn reads_retry_after_and_request_id() {
    let mut hdrs = HeaderMap::new();
    hdrs.insert("retry-after", "7".parse().unwrap());
    hdrs.insert("x-request-id", "req_123".parse().unwrap());
    let e = decode(429, r#"{"error":{"message":"slow","type":"rate_limit_error"}}"#, &hdrs);
    assert_eq!(e.retry_after, Some(std::time::Duration::from_secs(7)));
    assert_eq!(e.upstream_request_id.as_deref(), Some("req_123"));
}

#[test]
fn malformed_body_does_not_panic() {
    let e = decode(500, "<html>", &HeaderMap::new());
    assert_eq!(e.kind, ErrorKind::ServerError);
}

#[test]
fn encode_error_body_shape() {
    let e = XlateError::new(ErrorKind::InvalidRequest, "bad").with_param("model");
    let enc = codec().encode_error(&e, false, false);
    assert_eq!(enc.status, 400);
    let b = json(&enc.body);
    assert_eq!(b["error"]["message"], "bad");
    assert_eq!(b["error"]["type"], "invalid_request_error");
    assert_eq!(b["error"]["param"], "model");
    assert!(enc.frames.is_empty());
}

#[test]
fn encode_unsupported_has_router_code() {
    let e = XlateError::unsupported("tools", "no tools");
    let enc = codec().encode_error(&e, false, false);
    let b = json(&enc.body);
    assert_eq!(b["error"]["code"], "router_unsupported");
}

#[test]
fn encode_error_streaming_started_is_response_failed() {
    let e = XlateError::new(ErrorKind::ServerError, "boom");
    let enc = codec().encode_error(&e, true, true);
    assert!(enc.body.is_empty());
    assert_eq!(enc.frames.len(), 1);
    let frame = String::from_utf8(enc.frames[0].to_vec()).unwrap();
    assert!(frame.contains("event: response.failed"));
}

#[test]
fn encode_error_streaming_not_started_is_error_event() {
    let e = XlateError::new(ErrorKind::ServerError, "boom");
    let enc = codec().encode_error(&e, true, false);
    let frame = String::from_utf8(enc.frames[0].to_vec()).unwrap();
    assert!(frame.contains("event: error"));
}

#[test]
fn encode_error_carries_retry_after_header() {
    let e = XlateError::new(ErrorKind::RateLimited, "slow")
        .with_retry_after(std::time::Duration::from_secs(3));
    let enc = codec().encode_error(&e, false, false);
    assert_eq!(enc.headers.get("retry-after").unwrap(), "3");
}

#[test]
fn authentication_default_code() {
    let e = XlateError::new(ErrorKind::Authentication, "no key");
    let enc = codec().encode_error(&e, false, false);
    let b = json(&enc.body);
    assert_eq!(b["error"]["code"], "invalid_api_key");
    assert_eq!(enc.status, 401);
}

#[test]
fn billing_renders_insufficient_quota_type() {
    // Billing with no preserved provider type/code renders the OpenAI quota dialect.
    let e = XlateError::new(ErrorKind::Billing, "over quota");
    let enc = codec().encode_error(&e, false, false);
    assert_eq!(enc.status, 429);
    let b = json(&enc.body);
    assert_eq!(b["error"]["type"], "insufficient_quota");
    assert_eq!(b["error"]["code"], "insufficient_quota");
}

#[test]
fn foreign_overloaded_status_becomes_503() {
    // An Overloaded error decoded from Anthropic (status 529) must reach a Responses client as
    // HTTP 503, not the foreign dialect's 529.
    let e = XlateError::new(ErrorKind::Overloaded, "busy")
        .with_status(529)
        .with_provider(llm_xlate_core::ProviderFamily::Anthropic);
    let enc = codec().encode_error(&e, false, false);
    assert_eq!(enc.status, 503);
}

#[test]
fn foreign_billing_status_becomes_429() {
    let e = XlateError::new(ErrorKind::Billing, "over quota").with_status(402);
    let enc = codec().encode_error(&e, false, false);
    assert_eq!(enc.status, 429);
}
