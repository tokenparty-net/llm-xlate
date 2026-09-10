#![allow(clippy::field_reassign_with_default)]
//! Capability-matrix goldens and fixture-driven tests.

mod common;
use common::*;

use llm_xlate_core::caps::preset;
use llm_xlate_core::{
    Codec, DegradationKind, Effort, ErrorKind, IrRequest, Item, OutputConfig, OutputFormat,
    ReasoningConfig, ReasoningExposure, ToolChoice, ToolDef, Verbosity,
};
use pretty_assertions::assert_eq;

const REQUEST_FULL: &str = include_str!("fixtures/request_full.json");
const RESPONSE_COMPLETED: &str = include_str!("fixtures/response_completed.json");
const STREAM_MESSAGE: &str = include_str!("fixtures/stream_message.sse");
const STREAM_RICH: &str = include_str!("fixtures/stream_rich.sse");
const RESPONSE_REFUSAL: &str = include_str!("fixtures/response_refusal.json");
const RESPONSE_CONTENT_FILTER: &str = include_str!("fixtures/response_incomplete_content_filter.json");
const RESPONSE_WEB_SEARCH: &str = include_str!("fixtures/response_web_search.json");
const ERROR_RATE_LIMIT: &str = include_str!("fixtures/error_rate_limit.json");

/// A demanding request: Max effort, Full exposure, strict json_schema, named tool choice,
/// verbosity, a function tool.
fn demanding() -> IrRequest {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("model-x");
    ir.reasoning = ReasoningConfig {
        effort: Some(Effort::Max),
        budget_tokens: None,
        enabled: None,
        expose: ReasoningExposure::Full,
    };
    ir.output = OutputConfig {
        format: OutputFormat::JsonSchema {
            name: "S".into(),
            schema: serde_json::json!({"type":"object"}),
            strict: true,
            description: None,
        },
        verbosity: Some(Verbosity::High),
    };
    ir.tools.push(ToolDef::Function {
        name: "f".into(),
        description: None,
        parameters: serde_json::json!({"type":"object"}),
        strict: Some(true),
        cache_control: None,
    });
    ir.tool_choice = ToolChoice::Named("f".into());
    ir.items.push(Item::user_text("go"));
    ir
}

#[test]
fn matrix_gpt5_responses() {
    let enc = encode(&demanding(), &preset::gpt5_responses());
    let b = json(&enc.body);
    assert_eq!(b["reasoning"]["effort"], "xhigh"); // Max → xhigh (not in effort_levels)
    // `Full` exposure is the Chat/Anthropic decoder default, not a request-side summary request:
    // it emits no `summary` and no `reasoning.summary` downgrade (a client wanting a summary sends
    // a `Summary(_)` exposure, exercised in encode_request.rs / roundtrip.rs).
    assert!(b["reasoning"].get("summary").is_none(), "Full emits no summary: {}", b["reasoning"]);
    assert_eq!(b["text"]["format"]["strict"], true);
    assert_eq!(b["text"]["verbosity"], "high");
    assert_eq!(b["tool_choice"]["name"], "f");
    assert!(enc.degradations.iter().any(|d| d.field == "reasoning.effort" && d.kind == DegradationKind::Downgraded));
    assert!(!enc.degradations.iter().any(|d| d.field == "reasoning.summary"));
}

#[test]
fn matrix_gpt6() {
    let enc = encode(&demanding(), &preset::gpt6());
    let b = json(&enc.body);
    assert_eq!(b["reasoning"]["effort"], "xhigh");
    assert_eq!(b["text"]["format"]["strict"], true);
    assert_eq!(b["text"]["verbosity"], "high");
}

#[test]
fn matrix_gpt4o_drops_verbosity() {
    let enc = encode(&demanding(), &preset::gpt4o());
    let b = json(&enc.body);
    // gpt4o has verbosity = false → dropped with a degradation.
    assert!(b.get("text").and_then(|t| t.get("verbosity")).is_none());
    assert!(enc.degradations.iter().any(|d| d.field == "verbosity"));
}

#[test]
fn matrix_openai_compatible_rejects_tools() {
    let err = codec()
        .encode_request(&demanding(), &preset::openai_compatible(), &ectx())
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Unsupported);
}

#[test]
fn fixture_request_decodes_and_reencodes_deterministically() {
    let ir = decode(REQUEST_FULL);
    assert_eq!(ir.instructions.len(), 1);
    assert_eq!(ir.items.len(), 3);
    assert_eq!(ir.tools.len(), 1);
    assert_eq!(ir.state.store, Some(false));
    assert_eq!(ir.ext.get("responses.truncation").unwrap(), "auto");
    let a = encode(&ir, &caps()).body;
    let b = encode(&ir, &caps()).body;
    assert_eq!(a, b);
}

#[test]
fn fixture_response_decodes() {
    let events = codec().decode_response(RESPONSE_COMPLETED.as_bytes(), &caps()).unwrap();
    let resp = aggregate(&events);
    assert_eq!(resp.items.len(), 2);
    assert_eq!(resp.stop, llm_xlate_core::StopReason::ToolUse);
    assert_eq!(resp.usage.input, 50);
    assert_eq!(resp.usage.cache_read, Some(10));
    assert_eq!(resp.usage.reasoning, Some(12));
    match &resp.items[0] {
        Item::Reasoning(ri) => assert_eq!(ri.opaque.as_ref().unwrap().data, "ENCBLOB"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn fixture_stream_matches_response_aggregate() {
    let mut d = codec().stream_decoder(&caps());
    let mut events = d.push(STREAM_MESSAGE.as_bytes());
    events.extend(d.finish());
    let resp = aggregate(&events);
    assert_eq!(resp.items, vec![Item::Message {
        role: llm_xlate_core::Role::Assistant,
        content: vec![llm_xlate_core::Part::text("Hi")],
        id: Some(llm_xlate_core::ItemId::new("msg_1")),
    }]);
    assert_eq!(resp.usage.input, 2);
    assert_eq!(resp.usage.output, 1);
}

#[test]
fn fixture_stream_chunk_fuzz() {
    let bytes = STREAM_MESSAGE.as_bytes();
    let mut d = codec().stream_decoder(&caps());
    let mut whole = d.push(bytes);
    whole.extend(d.finish());
    for split in 1..bytes.len() {
        let mut d = codec().stream_decoder(&caps());
        let mut got = d.push(&bytes[..split]);
        got.extend(d.push(&bytes[split..]));
        got.extend(d.finish());
        assert_eq!(got, whole, "split {split}");
    }
}

#[test]
fn fixture_error_decodes() {
    let e = codec().decode_error(429, ERROR_RATE_LIMIT.as_bytes(), &llm_xlate_core::HeaderMap::new(), &caps());
    assert_eq!(e.kind, ErrorKind::RateLimited);
    assert_eq!(e.provider_code.as_deref(), Some("rate_limit_exceeded"));
}

#[test]
fn fixture_rich_stream_decodes_reasoning_text_and_tool() {
    // A realistic stream: reasoning summary → assistant text → function-call arg deltas.
    let mut d = codec().stream_decoder(&caps());
    let mut events = d.push(STREAM_RICH.as_bytes());
    events.extend(d.finish());
    let resp = aggregate(&events);
    assert_eq!(resp.items.len(), 3);
    match &resp.items[0] {
        Item::Reasoning(ri) => {
            assert_eq!(ri.summary, vec!["Weigh the tool call.".to_string()]);
            assert_eq!(ri.opaque.as_ref().unwrap().data, "ENCBLOB");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(resp.items[1], Item::Message {
        role: llm_xlate_core::Role::Assistant,
        content: vec![llm_xlate_core::Part::text("Checking the weather.")],
        id: Some(llm_xlate_core::ItemId::new("msg_1")),
    });
    match &resp.items[2] {
        Item::ToolCall { arguments, call_id, name, .. } => {
            assert_eq!(arguments.as_str(), "{\"city\":\"NYC\"}");
            assert_eq!(call_id.as_str(), "call_abc");
            assert_eq!(name, "get_weather");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(resp.stop, llm_xlate_core::StopReason::ToolUse);
    assert_eq!(resp.usage.reasoning, Some(6));
    assert_eq!(resp.usage.cache_read, Some(8));
}

#[test]
fn fixture_rich_stream_chunk_fuzz() {
    let bytes = STREAM_RICH.as_bytes();
    let one_shot = {
        let mut d = codec().stream_decoder(&caps());
        let mut w = d.push(bytes);
        w.extend(d.finish());
        w
    };
    for split in 1..bytes.len() {
        let mut d = codec().stream_decoder(&caps());
        let mut got = d.push(&bytes[..split]);
        got.extend(d.push(&bytes[split..]));
        got.extend(d.finish());
        assert_eq!(got, one_shot, "split {split}");
    }
}

#[test]
fn fixture_refusal_response_decodes() {
    let events = codec().decode_response(RESPONSE_REFUSAL.as_bytes(), &caps()).unwrap();
    let resp = aggregate(&events);
    assert_eq!(resp.items, vec![Item::Message {
        role: llm_xlate_core::Role::Assistant,
        content: vec![llm_xlate_core::Part::Refusal { text: "I can't help with that request.".into() }],
        id: Some(llm_xlate_core::ItemId::new("msg_1")),
    }]);
    assert_eq!(resp.stop, llm_xlate_core::StopReason::EndTurn);
}

#[test]
fn fixture_content_filter_incomplete_decodes() {
    let events = codec().decode_response(RESPONSE_CONTENT_FILTER.as_bytes(), &caps()).unwrap();
    let resp = aggregate(&events);
    assert_eq!(resp.stop, llm_xlate_core::StopReason::ContentFilter);
    match &resp.items[0] {
        Item::Message { content, .. } => assert_eq!(content[0].as_text().unwrap(), "Here is the start"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn fixture_web_search_passthrough_decodes() {
    let events = codec().decode_response(RESPONSE_WEB_SEARCH.as_bytes(), &caps()).unwrap();
    let resp = aggregate(&events);
    assert_eq!(resp.items.len(), 2);
    match &resp.items[0] {
        Item::ProviderToolCall(oi) => {
            assert_eq!(oi.family, llm_xlate_core::ProviderFamily::OpenAI);
            assert_eq!(oi.raw["type"], "web_search_call");
            assert_eq!(oi.raw["action"]["query"], "weather in NYC");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(resp.items[1], Item::Message {
        role: llm_xlate_core::Role::Assistant,
        content: vec![llm_xlate_core::Part::text("It is sunny.")],
        id: Some(llm_xlate_core::ItemId::new("msg_1")),
    });
}
