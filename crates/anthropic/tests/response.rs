//! Non-streaming response bridging: `decode_response` -> events -> aggregate, and
//! `encode_response`.

mod common;

use common::*;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_core::{
    Aggregator, Codec, IrResponse, Item, OpaqueKind, ProviderFamily, StopReason,
};
use pretty_assertions::assert_eq;

fn decode(body: &str) -> IrResponse {
    let events = AnthropicCodec.decode_response(body.as_bytes(), &claude_5()).unwrap();
    let mut agg = Aggregator::new();
    for e in events {
        agg.push(e);
    }
    agg.finish().unwrap()
}

#[test]
fn plain_text_response() {
    let r = decode(
        r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"Hello!"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}"#,
    );
    assert_eq!(r.id.as_str(), "msg_1");
    assert_eq!(r.items, vec![Item::assistant_text("Hello!")]);
    assert_eq!(r.stop, StopReason::EndTurn);
    assert_eq!(r.usage.input, 5);
    assert_eq!(r.usage.output, 3);
}

#[test]
fn max_tokens_cutoff() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"trunc"}],"stop_reason":"max_tokens","usage":{"input_tokens":5,"output_tokens":128000}}"#,
    );
    assert_eq!(r.stop, StopReason::MaxTokens);
}

#[test]
fn refusal_stop_reason() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"I can't help with that."}],"stop_reason":"refusal","usage":{"input_tokens":5,"output_tokens":6}}"#,
    );
    assert_eq!(r.stop, StopReason::Refusal);
}

#[test]
fn thinking_and_tool_response() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5","content":[
        {"type":"thinking","thinking":"reason","signature":"SIG=="},
        {"type":"text","text":"calling"},
        {"type":"tool_use","id":"toolu_1","name":"get","input":{"q":"x"}}],
        "stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":10}}"#,
    );
    assert_eq!(r.items.len(), 3);
    match &r.items[0] {
        Item::Reasoning(ri) => {
            assert_eq!(ri.text.as_deref(), Some("reason"));
            assert_eq!(ri.opaque.as_ref().unwrap().kind, OpaqueKind::Signature);
        }
        other => panic!("{other:?}"),
    }
    match &r.items[2] {
        Item::ToolCall { name, arguments, .. } => {
            assert_eq!(name, "get");
            assert_eq!(arguments.as_str(), r#"{"q":"x"}"#);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn provider_tool_result_roundtrips_raw() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5","content":[
        {"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"x"}},
        {"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"url":"u","title":"t"}]}],
        "stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":10}}"#,
    );
    match &r.items[0] {
        Item::ProviderToolCall(oi) => {
            assert_eq!(oi.family, ProviderFamily::Anthropic);
            assert_eq!(oi.raw["name"], serde_json::json!("web_search"));
        }
        other => panic!("{other:?}"),
    }
    match &r.items[1] {
        Item::ProviderToolResult(oi) => {
            assert_eq!(oi.raw["content"][0]["url"], serde_json::json!("u"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn redacted_thinking_response() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5","content":[
        {"type":"redacted_thinking","data":"BLOB"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#,
    );
    match &r.items[0] {
        Item::Reasoning(ri) => {
            assert!(ri.text.is_none());
            let blob = ri.opaque.as_ref().unwrap();
            assert_eq!(blob.data, "BLOB");
            // The native opaque carrier keeps its upstream-model binding (envelope boundary).
            assert_eq!(blob.model.as_deref(), Some("claude-opus-5"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn cache_usage_fields_decoded() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",
        "usage":{"input_tokens":5,"output_tokens":3,"cache_read_input_tokens":100,
        "cache_creation":{"ephemeral_5m_input_tokens":20,"ephemeral_1h_input_tokens":30}}}"#,
    );
    assert_eq!(r.usage.cache_read, Some(100));
    // The IR stores one all-TTL write total plus the 1-hour subset, so the two ephemeral
    // buckets sum; the 5m figure is derived back out for the wire.
    assert_eq!(r.usage.cache_write, Some(50));
    assert_eq!(r.usage.cache_write_1h, Some(30));
    assert_eq!(r.usage.cache_write_5m(), Some(20));
    // `input_tokens` is already the fresh count on this dialect.
    assert_eq!(r.usage.input, 5);
    assert_eq!(r.usage.gross_prompt(), 155);
}

#[test]
fn flat_cache_creation_total_has_no_ttl_split() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",
        "usage":{"input_tokens":5,"output_tokens":3,"cache_creation_input_tokens":40}}"#,
    );
    assert_eq!(r.usage.cache_write, Some(40));
    assert_eq!(r.usage.cache_write_1h, None);
    // Writes of unreported TTL bill at the short rate.
    assert_eq!(r.usage.cache_write_5m(), Some(40));
}

#[test]
fn thinking_tokens_decode_into_reasoning() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",
        "usage":{"input_tokens":5,"output_tokens":30,
        "output_tokens_details":{"thinking_tokens":12,"other_tokens":4}}}"#,
    );
    assert_eq!(r.usage.reasoning, Some(12));
    // A sibling of the mapped counter is preserved under its dotted path, not dropped, and
    // the mapped one is not also duplicated into ext.
    assert_eq!(
        r.usage.ext.get("anthropic.output_tokens_details.other_tokens"),
        Some(&serde_json::json!(4)),
    );
    assert_eq!(r.usage.ext.get("anthropic.output_tokens_details"), None);
}

#[test]
fn unknown_usage_fields_preserved_into_ext() {
    // `service_tier` (and any other unmodelled usage key) survives into `usage.ext`.
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",
        "usage":{"input_tokens":5,"output_tokens":3,"service_tier":"standard","server_tool_use":{"web_search_requests":2}}}"#,
    );
    assert_eq!(r.usage.ext.get("anthropic.service_tier"), Some(&serde_json::json!("standard")));
    assert_eq!(
        r.usage.ext.get("anthropic.server_tool_use"),
        Some(&serde_json::json!({"web_search_requests": 2})),
    );
}

#[test]
fn malformed_response_is_upstream_malformed() {
    let err = AnthropicCodec.decode_response(b"not json", &claude_5()).unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::UpstreamMalformed);
}

#[test]
fn container_ext_survives_to_response() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","container":{"id":"c_1"},
        "usage":{"input_tokens":1,"output_tokens":1}}"#,
    );
    assert_eq!(r.ext.get("anthropic.container"), Some(&serde_json::json!({"id":"c_1"})));
}

// -- encode_response --------------------------------------------------------------------

#[test]
fn encode_response_snapshot() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5","content":[
        {"type":"thinking","thinking":"reason","signature":"SIG=="},
        {"type":"text","text":"Answer."},
        {"type":"tool_use","id":"toolu_1","name":"get","input":{"q":"x"}}],
        "stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":10}}"#,
    );
    let body = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    insta::assert_snapshot!("encode_response", pretty(&body));
}

// -- client-facing reasoning exposure (plan §7.2 Resp→Ant exposure table) -----------------

fn reasoning_response(ri: llm_xlate_core::ReasoningItem) -> serde_json::Value {
    let r = IrResponse {
        model: "claude-opus-5".into(),
        items: vec![Item::Reasoning(ri), Item::assistant_text("Answer.")],
        ..Default::default()
    };
    let body = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    serde_json::from_slice(&body).unwrap()
}

#[test]
fn encode_response_summary_only_reasoning_becomes_thinking_without_signature() {
    // A Responses-origin reasoning item aggregates to text=None, summary=[..], opaque=None. The
    // client is receiving, so it becomes a `thinking` block carrying the joined summary and NO
    // signature (omitted; replayed decode treats a missing signature as no opaque).
    let v = reasoning_response(llm_xlate_core::ReasoningItem {
        text: None,
        summary: vec!["Consider the weather.".into(), "Then answer.".into()],
        opaque: None,
        id: None,
    });
    let block = &v["content"][0];
    assert_eq!(block["type"], "thinking");
    assert_eq!(block["thinking"], "Consider the weather.\n\nThen answer.");
    assert!(block.get("signature").is_none(), "signature must be omitted when no opaque exists");
}

#[test]
fn encode_response_summary_reasoning_with_opaque_seals_signature() {
    // With a foreign opaque carrier present, the same thinking block carries the sealed envelope
    // as its `signature`.
    let v = reasoning_response(llm_xlate_core::ReasoningItem {
        text: None,
        summary: vec!["Weigh options.".into()],
        opaque: Some(llm_xlate_core::OpaqueBlob::new(
            ProviderFamily::OpenAI,
            OpaqueKind::Encrypted,
            "ENC",
        )),
        id: None,
    });
    let block = &v["content"][0];
    assert_eq!(block["type"], "thinking");
    assert_eq!(block["thinking"], "Weigh options.");
    let sig = block["signature"].as_str().expect("signature present");
    assert!(sig.starts_with("rtr1."), "foreign opaque must be sealed into an rtr1. envelope");
}

#[test]
fn encode_response_no_text_no_summary_with_opaque_becomes_redacted() {
    // No text and no summary but an opaque blob → redacted_thinking carrying the sealed envelope.
    let v = reasoning_response(llm_xlate_core::ReasoningItem {
        text: None,
        summary: vec![],
        opaque: Some(llm_xlate_core::OpaqueBlob::new(
            ProviderFamily::OpenAI,
            OpaqueKind::Redacted,
            "BLOB",
        )),
        id: None,
    });
    let block = &v["content"][0];
    assert_eq!(block["type"], "redacted_thinking");
    assert!(block["data"].as_str().is_some_and(|d| d.starts_with("rtr1.")));
}

#[test]
fn encode_response_stop_sequence() {
    let mut r = IrResponse {
        model: "claude-opus-5".into(),
        items: vec![Item::assistant_text("x")],
        ..Default::default()
    };
    r.stop = StopReason::StopSequence("STOP".into());
    let body = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["stop_reason"], serde_json::json!("stop_sequence"));
    assert_eq!(v["stop_sequence"], serde_json::json!("STOP"));
}

#[test]
fn encode_response_uses_ctx_id_and_model() {
    let r = IrResponse {
        id: "ignored".into(),
        model: "ignored-model".into(),
        items: vec![Item::assistant_text("x")],
        ..Default::default()
    };
    let body = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["id"], serde_json::json!("msg_test_0001"));
    assert_eq!(v["model"], serde_json::json!("claude-opus-5"));
}

#[test]
fn encode_response_determinism() {
    let r = decode(
        r#"{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#,
    );
    let a = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    let b = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    assert_eq!(a, b);
}

#[test]
fn usage_round_trips_through_the_ir() {
    // A full Anthropic usage object, including the TTL split and the extras this codec does
    // not model, must survive decode and re-encode unchanged.
    let original = serde_json::json!({
        "input_tokens": 120,
        "cache_read_input_tokens": 5000,
        "cache_creation_input_tokens": 50,
        "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 30},
        "output_tokens": 40,
        "output_tokens_details": {"thinking_tokens": 12},
        "service_tier": "standard",
        "inference_geo": "global"
    });
    let r = decode(&format!(
        r#"{{"id":"m","type":"message","role":"assistant","model":"claude-opus-5",
        "content":[{{"type":"text","text":"x"}}],"stop_reason":"end_turn","usage":{original}}}"#
    ));
    let body = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["usage"], original);
}
