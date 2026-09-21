//! Error dialect mapping: `decode_error` and `encode_error`.

mod common;

use common::*;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_core::{Codec, ErrorKind, HeaderMap, XlateError};
use pretty_assertions::assert_eq;

fn decode_err(status: u16, ty: &str, msg: &str) -> XlateError {
    let body = format!(r#"{{"type":"error","error":{{"type":"{ty}","message":"{msg}"}}}}"#);
    AnthropicCodec.decode_error(status, body.as_bytes(), &no_headers(), &claude_5())
}

#[test]
fn every_error_type_maps() {
    let cases = [
        (400, "invalid_request_error", ErrorKind::InvalidRequest),
        (401, "authentication_error", ErrorKind::Authentication),
        (403, "permission_error", ErrorKind::Permission),
        (404, "not_found_error", ErrorKind::NotFound),
        (413, "request_too_large", ErrorKind::RequestTooLarge),
        (429, "rate_limit_error", ErrorKind::RateLimited),
        (529, "overloaded_error", ErrorKind::Overloaded),
        (400, "billing_error", ErrorKind::Billing),
        (500, "api_error", ErrorKind::ServerError),
    ];
    for (status, ty, kind) in cases {
        let e = decode_err(status, ty, "boom");
        assert_eq!(e.kind, kind, "type {ty}");
        assert_eq!(e.status, status);
        assert_eq!(e.provider, Some(llm_xlate_core::ProviderFamily::Anthropic));
        assert_eq!(e.provider_type.as_deref(), Some(ty));
    }
}

#[test]
fn context_length_detected_from_pattern() {
    let e = decode_err(400, "invalid_request_error", "prompt is too long: 200000 tokens");
    assert_eq!(e.kind, ErrorKind::ContextLengthExceeded);
}

#[test]
fn plain_invalid_request_not_context_length() {
    let e = decode_err(400, "invalid_request_error", "missing field");
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
}

#[test]
fn retryable_from_caps_status() {
    // 529 is in claude retryable_status.
    let e = decode_err(529, "overloaded_error", "overloaded");
    assert!(e.retryable);
    // 400 is not.
    let e = decode_err(400, "invalid_request_error", "bad");
    assert!(!e.retryable);
}

#[test]
fn request_id_and_retry_after_headers() {
    let mut hdrs = HeaderMap::new();
    hdrs.insert("request-id", "req_abc".parse().unwrap());
    hdrs.insert("retry-after", "12".parse().unwrap());
    let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#;
    let e = AnthropicCodec.decode_error(429, body.as_bytes(), &hdrs, &claude_5());
    assert_eq!(e.upstream_request_id.as_deref(), Some("req_abc"));
    assert_eq!(e.retry_after, Some(std::time::Duration::from_secs(12)));
}

#[test]
fn request_id_from_body() {
    let body = r#"{"type":"error","error":{"type":"api_error","message":"x"},"request_id":"req_body"}"#;
    let e = AnthropicCodec.decode_error(500, body.as_bytes(), &no_headers(), &claude_5());
    assert_eq!(e.upstream_request_id.as_deref(), Some("req_body"));
}

#[test]
fn malformed_error_body_defaults_to_server_error() {
    let e = AnthropicCodec.decode_error(500, b"<html>bad</html>", &no_headers(), &claude_5());
    assert_eq!(e.kind, ErrorKind::ServerError);
}

#[test]
fn non_envelope_4xx_keeps_status_class_and_message() {
    // A FastAPI validation 422 from a connector: a client error with its real reason, not a
    // retryable "upstream error" 500.
    let body = br#"{"detail":[{"type":"literal_error","loc":["body","thinking","type"],"msg":"Input should be 'enabled' or 'disabled'","input":"adaptive"}]}"#;
    let e = AnthropicCodec.decode_error(422, body, &no_headers(), &claude_5());
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
    assert!(!e.retryable);
    assert_eq!(e.message, "body.thinking.type: Input should be 'enabled' or 'disabled'");
    assert_eq!(e.provider_type, None);

    let enc = AnthropicCodec.encode_error(&e, false, false);
    assert_eq!(enc.status, 400);
    let v: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(v["error"]["type"], serde_json::json!("invalid_request_error"));
    assert_eq!(
        v["error"]["message"],
        serde_json::json!("body.thinking.type: Input should be 'enabled' or 'disabled'")
    );
}

#[test]
fn non_envelope_body_surfaces_raw_text() {
    let e = AnthropicCodec.decode_error(502, b"<html>bad gateway</html>", &no_headers(), &claude_5());
    assert_eq!(e.kind, ErrorKind::ServerError);
    assert_eq!(e.message, "<html>bad gateway</html>");
    let e = AnthropicCodec.decode_error(401, b"", &no_headers(), &claude_5());
    assert_eq!(e.kind, ErrorKind::Authentication);
    assert_eq!(e.message, "upstream error");
}

#[test]
fn encode_error_nonstreaming_body_and_status() {
    let e = XlateError::new(ErrorKind::Overloaded, "overloaded");
    let enc = AnthropicCodec.encode_error(&e, false, false);
    assert_eq!(enc.status, 529, "Anthropic renders Overloaded as 529");
    let v: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(v["type"], serde_json::json!("error"));
    assert_eq!(v["error"]["type"], serde_json::json!("overloaded_error"));
    assert_eq!(v["error"]["message"], serde_json::json!("overloaded"));
    assert!(enc.frames.is_empty());
}

#[test]
fn encode_error_maps_each_kind() {
    let cases = [
        (ErrorKind::InvalidRequest, "invalid_request_error"),
        (ErrorKind::Authentication, "authentication_error"),
        (ErrorKind::Permission, "permission_error"),
        (ErrorKind::NotFound, "not_found_error"),
        (ErrorKind::RequestTooLarge, "request_too_large"),
        (ErrorKind::RateLimited, "rate_limit_error"),
        (ErrorKind::Overloaded, "overloaded_error"),
        (ErrorKind::Billing, "billing_error"),
        (ErrorKind::ServerError, "api_error"),
        (ErrorKind::ContextLengthExceeded, "invalid_request_error"),
        (ErrorKind::Unsupported, "invalid_request_error"),
    ];
    for (kind, ty) in cases {
        let enc = AnthropicCodec.encode_error(&XlateError::new(kind, "m"), false, false);
        let v: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
        assert_eq!(v["error"]["type"], serde_json::json!(ty), "{kind:?}");
    }
}

#[test]
fn encode_error_forces_anthropic_canonical_status() {
    // A Billing error decoded from OpenAI carries status 429 (insufficient_quota); rendered to
    // the Anthropic dialect it must become 400 `billing_error` (plan §10 "to Ant"). Overloaded
    // is remapped to 529 regardless of its carried status.
    let mut billing = XlateError::new(ErrorKind::Billing, "quota");
    billing.status = 429;
    let enc = AnthropicCodec.encode_error(&billing, false, false);
    assert_eq!(enc.status, 400);
    let v: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(v["error"]["type"], serde_json::json!("billing_error"));

    let mut overloaded = XlateError::new(ErrorKind::Overloaded, "busy");
    overloaded.status = 503;
    assert_eq!(AnthropicCodec.encode_error(&overloaded, false, false).status, 529);
}

#[test]
fn encode_error_router_kinds_get_message_prefix() {
    // Router-originated rejections (Unsupported / IncompatibleHistory) map to the same
    // `invalid_request_error` type as a genuine upstream 400 and the dialect has no code field,
    // so the message is prefixed `router: ` to let a Claude client tell them apart (plan §10).
    for kind in [ErrorKind::Unsupported, ErrorKind::IncompatibleHistory] {
        let enc = AnthropicCodec.encode_error(&XlateError::new(kind, "the reason"), false, false);
        let v: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
        assert_eq!(v["error"]["type"], serde_json::json!("invalid_request_error"));
        assert_eq!(v["error"]["message"], serde_json::json!("router: the reason"), "{kind:?}");
    }
    // A genuine upstream error keeps its bare message (no spurious prefix).
    let enc = AnthropicCodec.encode_error(&XlateError::new(ErrorKind::InvalidRequest, "bad"), false, false);
    let v: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(v["error"]["message"], serde_json::json!("bad"));
}

#[test]
fn encode_error_streaming_frame() {
    let e = XlateError::new(ErrorKind::RateLimited, "slow");
    let enc = AnthropicCodec.encode_error(&e, true, true);
    assert!(enc.body.is_empty());
    assert_eq!(enc.frames.len(), 1);
    let frame = String::from_utf8(enc.frames[0].to_vec()).unwrap();
    assert!(frame.starts_with("event: error\n"));
    assert!(frame.contains("rate_limit_error"));
}

#[test]
fn encode_error_headers_retry_after_and_request_id() {
    let e = XlateError::new(ErrorKind::RateLimited, "slow")
        .with_retry_after(std::time::Duration::from_secs(5))
        .with_upstream_request_id("req_z");
    let enc = AnthropicCodec.encode_error(&e, false, false);
    assert_eq!(enc.headers.get("retry-after").unwrap(), "5");
    // Plan §10: the upstream request id is surfaced under the router-canonical header, matching
    // the chat/responses codecs (not Anthropic's native `request-id`).
    assert_eq!(enc.headers.get("x-router-upstream-request-id").unwrap(), "req_z");
}

#[test]
fn preserves_native_provider_type_on_reencode() {
    // A decoded Anthropic error keeps its provider_type on re-encode.
    let e = decode_err(404, "not_found_error", "no model");
    let enc = AnthropicCodec.encode_error(&e, false, false);
    let v: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(v["error"]["type"], serde_json::json!("not_found_error"));
}
