//! Non-streaming response path: `decode_response` and `encode_response`.

mod common;
use common::*;

use llm_xlate_core::{
    Codec, Delta, IrEvent, IrResponse, Item, ItemKind, OpaqueBlob, OpaqueKind, ProviderFamily,
    ReasoningItem, ResponseId, Role, StopReason, Usage,
};
use pretty_assertions::assert_eq;

fn decode_resp(body: &str) -> Vec<IrEvent> {
    codec().decode_response(body.as_bytes(), &caps()).unwrap()
}

#[test]
fn decode_basic_message() {
    let events = decode_resp(
        r#"{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.4",
            "output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed",
              "content":[{"type":"output_text","text":"Hi","annotations":[]}]}],
            "usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}"#,
    );
    assert!(matches!(events[0], IrEvent::Start { .. }));
    assert!(matches!(events[1], IrEvent::ItemStart { kind: ItemKind::Message, .. }));
    assert!(matches!(&events[2], IrEvent::Delta { delta: Delta::Text(t), .. } if t == "Hi"));
    assert!(matches!(events.last().unwrap(), IrEvent::Stop { reason: StopReason::EndTurn, .. }));
}

#[test]
fn decode_function_call_is_tool_use() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"completed","model":"m",
            "output":[{"id":"fc_1","type":"function_call","call_id":"c1","name":"get","arguments":"{}","status":"completed"}],
            "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"#,
    );
    match events.last().unwrap() {
        IrEvent::Stop { reason, .. } => assert_eq!(*reason, StopReason::ToolUse),
        other => panic!("{other:?}"),
    }
    match &events[1] {
        IrEvent::ItemStart { kind: ItemKind::ToolCall, call: Some((cid, name)), .. } => {
            assert_eq!(cid.as_str(), "c1");
            assert_eq!(name, "get");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn decode_incomplete_max_tokens() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"incomplete","model":"m","output":[],
            "incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":1,"output_tokens":9,"total_tokens":10}}"#,
    );
    match events.last().unwrap() {
        IrEvent::Stop { reason, usage, .. } => {
            assert_eq!(*reason, StopReason::MaxTokens);
            assert_eq!(usage.output, 9);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn decode_incomplete_content_filter() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"incomplete","model":"m","output":[],
            "incomplete_details":{"reason":"content_filter"}}"#,
    );
    assert!(matches!(events.last().unwrap(), IrEvent::Stop { reason: StopReason::ContentFilter, .. }));
}

#[test]
fn decode_reasoning_encrypted_and_summary() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"completed","model":"gpt-5.4",
            "output":[{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"S"}],
              "encrypted_content":"ENC","status":"completed"}],
            "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"output_tokens_details":{"reasoning_tokens":7}}}"#,
    );
    assert!(events.iter().any(|e| matches!(e, IrEvent::Delta { delta: Delta::ReasoningSummary { text, .. }, .. } if text == "S")));
    let opaque = events.iter().find_map(|e| match e {
        IrEvent::Delta { delta: Delta::Opaque(b), .. } => Some(b),
        _ => None,
    });
    let o = opaque.unwrap();
    assert_eq!(o.data, "ENC");
    assert_eq!(o.model.as_deref(), Some("gpt-5.4"));
    match events.last().unwrap() {
        IrEvent::Stop { usage, .. } => assert_eq!(usage.reasoning, Some(7)),
        other => panic!("{other:?}"),
    }
}

#[test]
fn decode_hosted_item_is_provider_raw() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"completed","model":"m",
            "output":[{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"x"}}]}"#,
    );
    assert!(events.iter().any(|e| matches!(e, IrEvent::Delta { delta: Delta::ProviderRaw { .. }, .. })));
    assert!(events.iter().any(|e| matches!(e, IrEvent::ItemStart { kind: ItemKind::ProviderToolCall, .. })));
}

#[test]
fn decode_failed_is_error() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"failed","error":{"type":"server_error","message":"boom"}}"#,
    );
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], IrEvent::Error(e) if e.message == "boom"));
}

fn usage_of(usage: &str) -> Usage {
    let body = format!(
        r#"{{"id":"r","object":"response","status":"completed","model":"m","output":[],
            "usage":{usage}}}"#
    );
    match decode_resp(&body).pop().unwrap() {
        IrEvent::Stop { usage, .. } => usage,
        other => panic!("{other:?}"),
    }
}

#[test]
fn decode_cached_tokens() {
    // `input_tokens` is the gross prompt, with the cached portion counted inside it.
    let u = usage_of(
        r#"{"input_tokens":100,"input_tokens_details":{"cached_tokens":80},"output_tokens":5,"total_tokens":105}"#,
    );
    assert_eq!(u.cache_read, Some(80));
    assert_eq!(u.input, 20);
    assert_eq!(u.gross_prompt(), 100);
}

#[test]
fn decode_openai_cache_write_tokens() {
    // OpenAI's own spelling on this dialect, captured live from gpt-4o-mini on 2026-09-10.
    let u = usage_of(
        r#"{"input_tokens":1000,"input_tokens_details":{"cache_write_tokens":300,"cached_tokens":600},
            "output_tokens":11,"output_tokens_details":{"reasoning_tokens":4},"total_tokens":1011}"#,
    );
    assert_eq!(u.cache_read, Some(600));
    assert_eq!(u.cache_write, Some(300));
    assert_eq!(u.reasoning, Some(4));
    assert_eq!(u.input, 100);
    // The mapped counter is not also duplicated into ext.
    assert_eq!(u.ext.get("responses.input_tokens_details.cache_write_tokens"), None);
}

#[test]
fn decode_one_hour_figure_without_a_total_raises_the_total() {
    let u = usage_of(
        r#"{"input_tokens":1000,"input_tokens_details":{"cache_write_1h_tokens":40},"output_tokens":1}"#,
    );
    assert_eq!(u.cache_write, Some(40));
    assert_eq!(u.cache_write_1h, Some(40));
    assert_eq!(u.input, 960);
}

#[test]
fn unmodelled_usage_counters_are_preserved_into_ext() {
    let u = usage_of(
        r#"{"input_tokens":10,"input_tokens_details":{"cached_tokens":0,"image_tokens":3},
            "output_tokens":2,"output_tokens_details":{"reasoning_tokens":1,"audio_tokens":9},
            "total_tokens":12,"cost_usd":0.5}"#,
    );
    assert_eq!(u.ext.get("responses.cost_usd"), Some(&serde_json::json!(0.5)));
    assert_eq!(
        u.ext.get("responses.input_tokens_details.image_tokens"),
        Some(&serde_json::json!(3)),
    );
    assert_eq!(
        u.ext.get("responses.output_tokens_details.audio_tokens"),
        Some(&serde_json::json!(9)),
    );
}

#[test]
fn usage_round_trips_through_the_ir() {
    let original = r#"{"input_tokens":1000,"input_tokens_details":{"cached_tokens":600,"cache_write_tokens":300},
        "output_tokens":11,"output_tokens_details":{"reasoning_tokens":4},"total_tokens":1011}"#;
    let ir = IrResponse { usage: usage_of(original), ..Default::default() };
    let rendered = json(&codec().encode_response(&ir, &ectx()));
    let expected: serde_json::Value = serde_json::from_str(original).unwrap();
    assert_eq!(rendered["usage"], expected);
}

#[test]
fn encode_basic_response() {
    let resp = IrResponse {
        id: ResponseId::new("resp_test"),
        model: "gpt-5.4".into(),
        items: vec![Item::assistant_text("hello")],
        stop: StopReason::EndTurn,
        usage: Usage::new(3, 5),
        ext: Default::default(),
    };
    let bytes = codec().encode_response(&resp, &ectx());
    let b = json(&bytes);
    assert_eq!(b["object"], "response");
    assert_eq!(b["status"], "completed");
    assert_eq!(b["id"], "resp_test");
    assert_eq!(b["created_at"], 1_726_000_000u64);
    assert_eq!(b["output"][0]["type"], "message");
    assert_eq!(b["output"][0]["content"][0]["text"], "hello");
    // Minted id: "msg_{response_id}_{index}".
    assert_eq!(b["output"][0]["id"], "msg_resp_test_0");
    assert_eq!(b["usage"]["input_tokens"], 3);
    assert_eq!(b["usage"]["output_tokens"], 5);
    assert_eq!(b["usage"]["total_tokens"], 8);
}

#[test]
fn encode_incomplete_status() {
    let resp = IrResponse {
        id: ResponseId::new("r"),
        model: "m".into(),
        items: vec![],
        stop: StopReason::MaxTokens,
        usage: Usage::new(1, 1),
        ext: Default::default(),
    };
    let b = json(&codec().encode_response(&resp, &ectx()));
    assert_eq!(b["status"], "incomplete");
    assert_eq!(b["incomplete_details"]["reason"], "max_output_tokens");
}

#[test]
fn encode_reasoning_with_encrypted_when_stateless() {
    let mut ctx = ectx();
    ctx.store = Some(false);
    let resp = IrResponse {
        id: ResponseId::new("r"),
        model: "m".into(),
        items: vec![Item::Reasoning(ReasoningItem {
            text: None,
            summary: vec!["s".into()],
            opaque: Some(OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Encrypted, "ENC")),
            id: None,
        })],
        stop: StopReason::EndTurn,
        usage: Usage::new(1, 1),
        ext: Default::default(),
    };
    let b = json(&codec().encode_response(&resp, &ctx));
    assert_eq!(b["output"][0]["type"], "reasoning");
    assert_eq!(b["output"][0]["encrypted_content"], "ENC");
}

#[test]
fn encode_foreign_reasoning_blob_is_sealed() {
    let mut ctx = ectx();
    ctx.store = Some(false);
    let resp = IrResponse {
        id: ResponseId::new("r"),
        model: "m".into(),
        items: vec![Item::Reasoning(ReasoningItem {
            text: None,
            summary: vec![],
            opaque: Some(OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "sig")),
            id: None,
        })],
        stop: StopReason::EndTurn,
        usage: Usage::new(1, 1),
        ext: Default::default(),
    };
    let b = json(&codec().encode_response(&resp, &ctx));
    let enc = b["output"][0]["encrypted_content"].as_str().unwrap();
    assert!(enc.starts_with("rtr1."));
}

#[test]
fn encode_response_echoes_request_params() {
    let ir = decode(r#"{"model":"gpt-5.4","instructions":"be nice","temperature":0.3,"reasoning":{"effort":"high"}}"#);
    let ctx = ectx_echo(&ir);
    let resp = IrResponse {
        id: ResponseId::new("resp_test"),
        model: "gpt-5.4".into(),
        items: vec![Item::assistant_text("ok")],
        stop: StopReason::EndTurn,
        usage: Usage::new(1, 1),
        ext: Default::default(),
    };
    let b = json(&codec().encode_response(&resp, &ctx));
    assert_eq!(b["instructions"], "be nice");
    assert_eq!(b["temperature"], 0.3);
    assert_eq!(b["reasoning"]["effort"], "high");
}

#[test]
fn encode_response_matches_aggregate_of_decode() {
    // aggregate(decode_response(body)) then encode_response is deterministic and stable.
    let body = r#"{"id":"resp_test","object":"response","status":"completed","model":"gpt-5.4",
        "output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed",
          "content":[{"type":"output_text","text":"hi","annotations":[]}]}],
        "usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}"#;
    let events = decode_resp(body);
    let resp = aggregate(&events);
    let a = codec().encode_response(&resp, &ectx());
    let b = codec().encode_response(&resp, &ectx());
    assert_eq!(a, b);
    let v = json(&a);
    assert_eq!(v["output"][0]["content"][0]["text"], "hi");
}

#[test]
fn decode_multiple_output_items_ordered() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"completed","model":"m",
            "output":[
              {"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"t"}],"status":"completed"},
              {"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"a","annotations":[]}]}
            ]}"#,
    );
    let resp = aggregate(&events);
    assert_eq!(resp.items.len(), 2);
    assert!(matches!(resp.items[0], Item::Reasoning(_)));
    assert!(matches!(&resp.items[1], Item::Message { role: Role::Assistant, .. }));
}

#[test]
fn decode_refusal_part() {
    let events = decode_resp(
        r#"{"id":"r","object":"response","status":"completed","model":"m",
            "output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed",
              "content":[{"type":"refusal","refusal":"cannot"}]}]}"#,
    );
    assert!(events.iter().any(|e| matches!(e, IrEvent::Delta { delta: Delta::Refusal(t), .. } if t == "cannot")));
}
