#![allow(clippy::result_large_err, clippy::field_reassign_with_default)]
//! Error decode / encode (plan §10).

mod common;

use common::*;
use llm_xlate_core::{Codec, ErrorKind, HeaderMap, XlateError};
use pretty_assertions::assert_eq;
use serde_json::Value;

fn decode_err(status: u16, body: &str, hdrs: HeaderMap) -> XlateError {
    codec().decode_error(status, body.as_bytes(), &hdrs, &gpt4o())
}

fn decode_simple(status: u16, body: &str) -> XlateError {
    decode_err(status, body, HeaderMap::new())
}

const E_KEY: &str = include_str!("fixtures/err_invalid_api_key.json");
const E_MODEL: &str = include_str!("fixtures/err_model_not_found.json");
const E_CTX: &str = include_str!("fixtures/err_context_length.json");
const E_QUOTA: &str = include_str!("fixtures/err_insufficient_quota.json");
const E_RATE: &str = include_str!("fixtures/err_rate_limit.json");
const E_SERVER: &str = include_str!("fixtures/err_server_error.json");

#[test]
fn decode_invalid_api_key() {
    let e = decode_simple(401, E_KEY);
    assert_eq!(e.kind, ErrorKind::Authentication);
    assert_eq!(e.status, 401);
    assert_eq!(e.provider_code.as_deref(), Some("invalid_api_key"));
}

#[test]
fn decode_model_not_found() {
    let e = decode_simple(404, E_MODEL);
    assert_eq!(e.kind, ErrorKind::NotFound);
    assert_eq!(e.status, 404);
}

#[test]
fn decode_context_length_exceeded() {
    let e = decode_simple(400, E_CTX);
    assert_eq!(e.kind, ErrorKind::ContextLengthExceeded);
    assert_eq!(e.param.as_deref(), Some("messages"));
}

#[test]
fn decode_insufficient_quota_is_billing() {
    let e = decode_simple(429, E_QUOTA);
    assert_eq!(e.kind, ErrorKind::Billing);
}

#[test]
fn decode_rate_limit() {
    let e = decode_simple(429, E_RATE);
    assert_eq!(e.kind, ErrorKind::RateLimited);
    assert!(e.retryable);
}

#[test]
fn decode_server_error_retryable() {
    let e = decode_simple(500, E_SERVER);
    assert_eq!(e.kind, ErrorKind::ServerError);
    assert!(e.retryable);
}

#[test]
fn decode_overloaded_maps_503() {
    let e = decode_simple(503, r#"{"error":{"message":"overloaded","type":"server_error"}}"#);
    assert_eq!(e.kind, ErrorKind::Overloaded);
    assert_eq!(e.status, 503);
}

#[test]
fn decode_permission_403() {
    let e = decode_simple(403, r#"{"error":{"message":"no","type":"permission_error"}}"#);
    assert_eq!(e.kind, ErrorKind::Permission);
}

#[test]
fn decode_honors_retry_after_and_request_id() {
    let mut hdrs = HeaderMap::new();
    hdrs.insert("retry-after", "30".parse().unwrap());
    hdrs.insert("x-request-id", "req_xyz".parse().unwrap());
    let e = decode_err(429, E_RATE, hdrs);
    assert_eq!(e.retry_after, Some(std::time::Duration::from_secs(30)));
    assert_eq!(e.upstream_request_id.as_deref(), Some("req_xyz"));
}

#[test]
fn decode_malformed_error_body_does_not_panic() {
    let e = decode_simple(500, "<html>Bad Gateway</html>");
    assert_eq!(e.kind, ErrorKind::ServerError);
}

// ---------------------------------------------------------------- encode

#[test]
fn encode_error_non_streaming_body() {
    let e = decode_simple(401, E_KEY);
    let out = codec().encode_error(&e, false, false);
    assert_eq!(out.status, 401);
    assert!(out.frames.is_empty());
    let v: Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["error"]["type"], Value::from("invalid_request_error"));
    assert_eq!(v["error"]["code"], Value::from("invalid_api_key"));
    assert_eq!(v["error"]["message"], Value::from("Incorrect API key provided."));
}

#[test]
fn encode_error_streaming_frame_then_done() {
    let e = decode_simple(429, E_RATE);
    let out = codec().encode_error(&e, true, true);
    assert!(out.body.is_empty());
    assert_eq!(out.frames.len(), 2);
    let f0 = String::from_utf8(out.frames[0].to_vec()).unwrap();
    let f1 = String::from_utf8(out.frames[1].to_vec()).unwrap();
    assert!(f0.contains("\"error\""));
    assert!(f1.contains("[DONE]"));
}

#[test]
fn encode_error_retry_after_header() {
    let e = XlateError::new(ErrorKind::RateLimited, "slow down")
        .with_status(429)
        .with_retry_after(std::time::Duration::from_secs(12));
    let out = codec().encode_error(&e, false, false);
    assert_eq!(out.headers.get("retry-after").unwrap(), "12");
}

#[test]
fn encode_billing_uses_insufficient_quota_type() {
    // A Billing error with no OpenAI provider_type must render the non-retryable
    // `insufficient_quota` type (not `rate_limit_error`), matching the Responses dialect and real
    // OpenAI so an SDK keying on error.type does not treat quota exhaustion as retryable.
    let e = XlateError::new(ErrorKind::Billing, "quota exhausted");
    let out = codec().encode_error(&e, false, false);
    assert_eq!(out.status, 429);
    let v: Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["error"]["type"], Value::from("insufficient_quota"));
    assert_eq!(v["error"]["code"], Value::from("insufficient_quota"));
}

#[test]
fn encode_rate_limit_without_provider_code_is_null() {
    // Aligns with the Responses dialect: the fallback rate-limit code is null (a real OpenAI rate
    // limit carries its own preserved provider code).
    let e = XlateError::new(ErrorKind::RateLimited, "slow down");
    let v: Value = serde_json::from_slice(&codec().encode_error(&e, false, false).body).unwrap();
    assert_eq!(v["error"]["type"], Value::from("rate_limit_error"));
    assert_eq!(v["error"]["code"], Value::Null);
}

#[test]
fn encode_normalizes_foreign_status_to_openai() {
    // An error decoded from a foreign backend (e.g. Anthropic Overloaded carrying 529) is rendered
    // to a Chat client with the OpenAI-canonical status, matching the Responses dialect (plan §10).
    let overloaded = XlateError::new(ErrorKind::Overloaded, "busy").with_status(529);
    assert_eq!(codec().encode_error(&overloaded, false, false).status, 503);
    let malformed = XlateError::new(ErrorKind::UpstreamMalformed, "bad").with_status(502);
    assert_eq!(codec().encode_error(&malformed, false, false).status, 500);
}

#[test]
fn encode_unsupported_uses_router_code() {
    let e = XlateError::unsupported("audio", "no audio").with_status(400);
    let out = codec().encode_error(&e, false, false);
    let v: Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["error"]["code"], Value::from("router_unsupported"));
    assert_eq!(v["error"]["param"], Value::from("audio"));
}

#[test]
fn error_round_trip_kind_preserved() {
    for (status, body, kind) in [
        (401, E_KEY, ErrorKind::Authentication),
        (404, E_MODEL, ErrorKind::NotFound),
        (400, E_CTX, ErrorKind::ContextLengthExceeded),
        (429, E_QUOTA, ErrorKind::Billing),
        (429, E_RATE, ErrorKind::RateLimited),
        (500, E_SERVER, ErrorKind::ServerError),
    ] {
        let e1 = decode_simple(status, body);
        assert_eq!(e1.kind, kind);
        let out = codec().encode_error(&e1, false, false);
        let e2 = decode_simple(e1.status, std::str::from_utf8(&out.body).unwrap());
        assert_eq!(e2.kind, kind, "round trip changed kind for status {status}");
    }
}

#[test]
fn non_envelope_422_is_invalid_request_with_body_message() {
    let body = r#"{"detail":[{"loc":["body","model"],"msg":"field required","type":"missing"}]}"#;
    let e = decode_simple(422, body);
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
    assert!(!e.retryable);
    assert_eq!(e.message, "body.model: field required");
}
