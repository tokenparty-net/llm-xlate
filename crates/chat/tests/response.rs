#![allow(clippy::result_large_err, clippy::field_reassign_with_default)]
//! Non-streaming response decode / encode, aggregation, round-trip and sealing stability.

mod common;

use common::*;
use llm_xlate_core::ir::{
    Item, OpaqueBlob, OpaqueKind, Part, ProviderFamily, ReasoningItem, ResponseId, StopReason,
    Usage,
};
use pretty_assertions::assert_eq;
use serde_json::Value;

const RESP_TEXT: &str = include_str!("fixtures/resp_text.json");
const RESP_TOOLS: &str = include_str!("fixtures/resp_toolcalls.json");
const RESP_REFUSAL: &str = include_str!("fixtures/resp_refusal.json");
const RESP_REASONING: &str = include_str!("fixtures/resp_reasoning.json");
const RESP_LENGTH: &str = include_str!("fixtures/resp_length.json");
const RESP_CF: &str = include_str!("fixtures/resp_content_filter.json");

fn ctx() -> llm_xlate_core::EncodeCtx {
    encode_ctx("gpt-4o", false)
}

// ---------------------------------------------------------------- decode

#[test]
fn decode_text_response() {
    let r = aggregate(decode_response(RESP_TEXT, &gpt4o()));
    assert_eq!(r.id, ResponseId::new("chatcmpl-abc123"));
    assert_eq!(r.stop, StopReason::EndTurn);
    assert_eq!(r.items.len(), 1);
    match &r.items[0] {
        Item::Message { content, .. } => {
            assert_eq!(content[0].as_text(), Some("France won the 2018 World Cup."));
        }
        other => panic!("expected message, got {other:?}"),
    }
    assert_eq!(r.usage.input, 20);
    assert_eq!(r.usage.output, 9);
    assert_eq!(r.ext.get("chat.service_tier"), Some(&Value::from("default")));
}

#[test]
fn decode_toolcalls_response() {
    let r = aggregate(decode_response(RESP_TOOLS, &gpt4o()));
    assert_eq!(r.stop, StopReason::ToolUse);
    let calls: Vec<_> = r
        .items
        .iter()
        .filter_map(|i| match i {
            Item::ToolCall { call_id, name, arguments, .. } => {
                Some((call_id.as_str().to_string(), name.clone(), arguments.as_str().to_string()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0], ("call_a".into(), "get_weather".into(), "{\"city\":\"Paris\"}".into()));
    assert_eq!(calls[1], ("call_b".into(), "get_weather".into(), "{\"city\":\"Tokyo\"}".into()));
}

#[test]
fn decode_refusal_response() {
    let r = aggregate(decode_response(RESP_REFUSAL, &gpt4o()));
    assert_eq!(r.stop, StopReason::Refusal);
    match &r.items[0] {
        Item::Message { content, .. } => {
            assert_eq!(content[0], Part::Refusal { text: "I can't help with that.".to_string() });
        }
        other => panic!("expected message, got {other:?}"),
    }
    assert!(r.ext.get("stop_details").is_some());
}

#[test]
fn decode_length_response() {
    let r = aggregate(decode_response(RESP_LENGTH, &gpt4o()));
    assert_eq!(r.stop, StopReason::MaxTokens);
}

#[test]
fn decode_content_filter_response() {
    let r = aggregate(decode_response(RESP_CF, &gpt4o()));
    assert_eq!(r.stop, StopReason::ContentFilter);
}

#[test]
fn decode_reasoning_response() {
    let r = aggregate(decode_response(RESP_REASONING, &gpt4o()));
    assert_eq!(r.usage.cache_read, Some(10));
    assert_eq!(r.usage.reasoning, Some(30));
    let reasoning = r.items.iter().find_map(|i| match i {
        Item::Reasoning(ri) => Some(ri),
        _ => None,
    });
    let ri = reasoning.expect("reasoning item");
    assert_eq!(ri.text.as_deref(), Some("17 times 23 is 391."));
    let opaque = ri.opaque.as_ref().expect("opaque blob");
    assert_eq!(opaque.family, ProviderFamily::OpenAI);
    assert_eq!(opaque.data, "enc-native-blob-xyz");
}

// ---------------------------------------------------------------- encode

#[test]
fn encode_response_text_golden() {
    let r = aggregate(decode_response(RESP_TEXT, &gpt4o()));
    insta::assert_snapshot!("encode_text", encode_response(&r, &ctx()));
}

#[test]
fn encode_response_toolcalls_golden() {
    let r = aggregate(decode_response(RESP_TOOLS, &gpt4o()));
    insta::assert_snapshot!("encode_toolcalls", encode_response(&r, &ctx()));
}

#[test]
fn encode_response_refusal_golden() {
    let r = aggregate(decode_response(RESP_REFUSAL, &gpt4o()));
    insta::assert_snapshot!("encode_refusal", encode_response(&r, &ctx()));
}

#[test]
fn encode_response_usage_total_computed() {
    let mut r = llm_xlate_core::IrResponse::default();
    r.usage = Usage { input: 5, output: 7, ..Default::default() };
    let out = encode_response(&r, &ctx());
    assert!(out.contains("\"total_tokens\":12"));
}

// ---------------------------------------------------------------- round trip

#[test]
fn response_round_trip_items_stable() {
    // Opaque reasoning blobs are re-sealed by encode_response (client boundary), so RESP_REASONING
    // is exercised separately below; these fixtures carry no opaque blobs and are item-stable.
    for body in [RESP_TEXT, RESP_TOOLS, RESP_REFUSAL, RESP_LENGTH, RESP_CF] {
        let r1 = aggregate(decode_response(body, &gpt4o()));
        let bytes = encode_response(&r1, &ctx());
        let r2 = aggregate(decode_response(&bytes, &gpt4o()));
        assert_eq!(r1.items, r2.items, "items differ for a fixture");
        assert_eq!(r1.stop, r2.stop);
        assert_eq!(r1.usage.input, r2.usage.input);
        assert_eq!(r1.usage.output, r2.usage.output);
        assert_eq!(r1.usage.reasoning, r2.usage.reasoning);
    }
}

#[test]
fn response_round_trip_reasoning_text_stable() {
    // The reasoning text, stop, and usage survive; the opaque blob is re-sealed (documented).
    let r1 = aggregate(decode_response(RESP_REASONING, &gpt4o()));
    let bytes = encode_response(&r1, &ctx());
    let r2 = aggregate(decode_response(&bytes, &gpt4o()));
    let text1 = r1.items.iter().find_map(|i| match i {
        Item::Reasoning(ri) => ri.text.clone(),
        _ => None,
    });
    let text2 = r2.items.iter().find_map(|i| match i {
        Item::Reasoning(ri) => ri.text.clone(),
        _ => None,
    });
    assert_eq!(text1, text2);
    assert_eq!(r1.stop, r2.stop);
    assert_eq!(r1.usage.reasoning, r2.usage.reasoning);
}

// ---------------------------------------------------------------- sealing stability

#[test]
fn opaque_reasoning_seals_and_reopens_stable() {
    // Build a response whose reasoning item carries a native OpenAI opaque blob.
    let blob = OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Encrypted, "native-enc-123");
    let mut r = llm_xlate_core::IrResponse::default();
    r.items.push(Item::Reasoning(ReasoningItem {
        text: Some("thinking".to_string()),
        opaque: Some(blob.clone()),
        ..Default::default()
    }));
    r.items.push(Item::assistant_text("answer"));

    // encode_response seals the blob into reasoning_details (rtr1 envelope).
    let body = encode_response(&r, &ctx());
    let v: Value = serde_json::from_str(&body).unwrap();
    let details = v["choices"][0]["message"]["reasoning_details"].clone();
    let sealed = details[0]["data"].as_str().unwrap();
    assert!(sealed.starts_with("rtr1."), "expected an rtr1 envelope, got {sealed}");

    // Feed it back as client history: decode_request opens the envelope to the same blob.
    let req = serde_json::json!({
        "model": "gpt-5.4",
        "messages": [{
            "role": "assistant",
            "reasoning_content": "thinking",
            "reasoning_details": details,
        }],
    });
    let ir = decode_request(&req.to_string());
    let reopened = ir.items.iter().find_map(|i| match i {
        Item::Reasoning(ri) => ri.opaque.clone(),
        _ => None,
    });
    assert_eq!(reopened, Some(blob));
}

#[test]
fn decode_response_error_body_yields_error_event() {
    let body = r#"{"error":{"message":"boom","type":"server_error","code":null}}"#;
    let events = decode_response(body, &gpt4o());
    assert!(matches!(events.first(), Some(llm_xlate_core::IrEvent::Error(_))));
}

// ---------------------------------------------------------------- streaming-law twin

#[test]
fn stream_and_response_twin_agree_toolcalls() {
    // The tools stream fixture and the tools response fixture describe the same turn.
    let stream = include_str!("fixtures/stream_tools.txt");
    let a = aggregate(decode_stream(stream, &gpt4o()));
    let b = aggregate(decode_response(RESP_TOOLS, &gpt4o()));
    assert_eq!(a.items, b.items);
    assert_eq!(a.stop, b.stop);
    assert_eq!(a.usage.input, b.usage.input);
    assert_eq!(a.usage.output, b.usage.output);
}

// ---------------------------------------------------------------- cache accounting

/// Build a Chat response body carrying `usage`.
fn resp_with_usage(usage: &str) -> String {
    format!(
        r#"{{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o",
        "choices":[{{"index":0,"message":{{"role":"assistant","content":"ok"}},"finish_reason":"stop"}}],
        "usage":{usage}}}"#
    )
}

fn usage_of(body: &str) -> Usage {
    aggregate(decode_response(body, &gpt4o())).usage
}

#[test]
fn decode_cached_and_created_cache_tokens() {
    // The vLLM / Kimi K3 shape: a cache read and a cache write, both counted inside
    // `prompt_tokens`.
    let u = usage_of(&resp_with_usage(
        r#"{"prompt_tokens":5000,"completion_tokens":11,"total_tokens":5011,
        "prompt_tokens_details":{"cached_tokens":2560,"created_cache_tokens":256}}"#,
    ));
    assert_eq!(u.cache_read, Some(2560));
    assert_eq!(u.cache_write, Some(256));
    assert_eq!(u.input, 2184);
    assert_eq!(u.gross_prompt(), 5000);
}

#[test]
fn decode_anthropic_bridge_cache_spellings() {
    let u = usage_of(&resp_with_usage(
        r#"{"prompt_tokens":1000,"completion_tokens":10,"total_tokens":1010,
        "prompt_tokens_details":{"cached_tokens":600,"cache_creation_tokens":300,
        "cache_creation_1h_tokens":120},"completion_tokens_details":{"reasoning_tokens":4}}"#,
    ));
    assert_eq!(u.cache_read, Some(600));
    assert_eq!(u.cache_write, Some(300));
    assert_eq!(u.cache_write_1h, Some(120));
    assert_eq!(u.cache_write_5m(), Some(180));
    assert_eq!(u.reasoning, Some(4));
    assert_eq!(u.input, 100);
}

#[test]
fn decode_treats_prompt_tokens_as_gross_even_beside_anthropic_keys() {
    // Some OpenAI-compatible servers emit both conventions at once. Only the OpenAI spelling
    // says which meaning the prompt count carries, and it always means gross.
    let u = usage_of(&resp_with_usage(
        r#"{"prompt_tokens":1000,"completion_tokens":10,
        "cache_read_input_tokens":700,"cache_creation_input_tokens":200}"#,
    ));
    assert_eq!(u.cache_read, Some(700));
    assert_eq!(u.cache_write, Some(200));
    assert_eq!(u.input, 100);
}

#[test]
fn decode_one_hour_figure_without_a_total_raises_the_total() {
    let u = usage_of(&resp_with_usage(
        r#"{"prompt_tokens":1000,"completion_tokens":10,
        "prompt_tokens_details":{"cache_creation_1h_tokens":40}}"#,
    ));
    assert_eq!(u.cache_write, Some(40));
    assert_eq!(u.cache_write_1h, Some(40));
    // The recovered write is removed from the fresh input, not billed twice.
    assert_eq!(u.input, 960);
}

#[test]
fn encode_regrosses_the_prompt_and_total() {
    let u = Usage {
        input: 100,
        output: 10,
        cache_read: Some(600),
        cache_write: Some(300),
        cache_write_1h: Some(120),
        reasoning: Some(4),
        ..Default::default()
    };
    let ir = llm_xlate_core::IrResponse { usage:u, ..Default::default() };
    let body: Value = serde_json::from_str(&encode_response(&ir, &ctx())).unwrap();
    let usage = &body["usage"];
    assert_eq!(usage["prompt_tokens"], 1000);
    assert_eq!(usage["completion_tokens"], 10);
    assert_eq!(usage["total_tokens"], 1010);
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 600);
    assert_eq!(usage["prompt_tokens_details"]["cache_creation_tokens"], 300);
    assert_eq!(usage["prompt_tokens_details"]["cache_creation_1h_tokens"], 120);
    assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 4);
}

#[test]
fn usage_round_trips_through_the_ir() {
    // The full shape an Anthropic bridge emits, including a root-level `cost_usd` this codec
    // does not model.
    let original = r#"{"prompt_tokens":1000,"completion_tokens":10,"total_tokens":1010,
        "prompt_tokens_details":{"cached_tokens":600,"cache_creation_tokens":300,
        "cache_creation_1h_tokens":120,"audio_tokens":0},
        "completion_tokens_details":{"reasoning_tokens":4,"accepted_prediction_tokens":2},
        "cost_usd":0.0123}"#;
    let ir = llm_xlate_core::IrResponse { usage:usage_of(&resp_with_usage(original)), ..Default::default() };
    let body: Value = serde_json::from_str(&encode_response(&ir, &ctx())).unwrap();
    let expected: Value = serde_json::from_str(original).unwrap();
    assert_eq!(body["usage"], expected);
}

#[test]
fn unmodelled_usage_counters_are_preserved_into_ext() {
    let u = usage_of(&resp_with_usage(
        r#"{"prompt_tokens":10,"completion_tokens":2,"cost_usd":0.5,
        "prompt_tokens_details":{"cached_tokens":0,"audio_tokens":3},
        "completion_tokens_details":{"reasoning_tokens":1,"rejected_prediction_tokens":7}}"#,
    ));
    assert_eq!(u.ext.get("chat.cost_usd"), Some(&serde_json::json!(0.5)));
    assert_eq!(
        u.ext.get("chat.prompt_tokens_details.audio_tokens"),
        Some(&serde_json::json!(3)),
    );
    assert_eq!(
        u.ext.get("chat.completion_tokens_details.rejected_prediction_tokens"),
        Some(&serde_json::json!(7)),
    );
    // Mapped counters are not also duplicated into ext.
    assert_eq!(u.ext.get("chat.prompt_tokens_details.cached_tokens"), None);
}
