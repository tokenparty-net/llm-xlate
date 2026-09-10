//! Streaming: SSE decode, chunk-boundary fuzz, stream encode, and the aggregation laws.

mod common;

use common::*;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_core::{
    Aggregator, CallId, Codec, Delta, IrEvent, IrResponse, ItemKind, OpaqueBlob, OpaqueKind,
    ProviderFamily, ResponseId, StopReason, Usage,
};
use pretty_assertions::assert_eq;

// A text -> tool_use -> text stream (a new Message item opens after the tool_use).
const TEXT_TOOL_TEXT: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me check.\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get\",\"input\":{}}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":1}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"Done.\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":2}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":25}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

const TEXT_TOOL_TEXT_TWIN: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5","content":[{"type":"text","text":"Let me check."},{"type":"tool_use","id":"toolu_1","name":"get","input":{"city":"NYC"}},{"type":"text","text":"Done."}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":25}}"#;

const THINKING: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"pondering\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"SIG==\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Answer.\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":1}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":30}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

const REDACTED: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_3\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"REDACTED-BLOB\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":7}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

const PAUSE_TURN: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_4\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"pause_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

const STOP_SEQUENCE: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_5\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stop here\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"stop_sequence\",\"stop_sequence\":\"STOP\"},\"usage\":{\"output_tokens\":9}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

const ERROR_STREAM: &str = "\
event: error
data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}

";

fn decode_stream(sse: &str) -> Vec<IrEvent> {
    let mut d = AnthropicCodec.stream_decoder(&claude_5());
    let mut evs = d.push(sse.as_bytes());
    evs.extend(d.finish());
    evs
}

fn aggregate(events: Vec<IrEvent>) -> IrResponse {
    let mut agg = Aggregator::new();
    for e in events {
        agg.push(e);
    }
    agg.finish().unwrap()
}

#[test]
fn decode_text_tool_text_snapshot() {
    let events = decode_stream(TEXT_TOOL_TEXT);
    insta::assert_debug_snapshot!("text_tool_text_events", events);
}

#[test]
fn text_tool_text_new_message_after_tool() {
    let events = decode_stream(TEXT_TOOL_TEXT);
    let starts: Vec<(u32, ItemKind)> = events
        .iter()
        .filter_map(|e| match e {
            IrEvent::ItemStart { index, kind, .. } => Some((*index, *kind)),
            _ => None,
        })
        .collect();
    assert_eq!(
        starts,
        vec![(0, ItemKind::Message), (1, ItemKind::ToolCall), (2, ItemKind::Message)]
    );
    let r = aggregate(events);
    assert_eq!(r.items.len(), 3);
    assert_eq!(r.stop, StopReason::ToolUse);
}

#[test]
fn twin_law_text_tool_text() {
    let stream = aggregate(decode_stream(TEXT_TOOL_TEXT));
    let nonstream = aggregate(
        AnthropicCodec.decode_response(TEXT_TOOL_TEXT_TWIN.as_bytes(), &claude_5()).unwrap(),
    );
    assert_eq!(stream, nonstream);
}

#[test]
fn thinking_stream_aggregates_signature() {
    let r = aggregate(decode_stream(THINKING));
    match &r.items[0] {
        llm_xlate_core::Item::Reasoning(ri) => {
            assert_eq!(ri.text.as_deref(), Some("pondering"));
            let o = ri.opaque.as_ref().unwrap();
            assert_eq!(o.kind, OpaqueKind::Signature);
            assert_eq!(o.data, "SIG==");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn redacted_thinking_stream() {
    let events = decode_stream(REDACTED);
    // The redacted block start immediately yields an Opaque(Redacted) delta.
    assert!(events.iter().any(|e| matches!(
        e,
        IrEvent::Delta { delta: Delta::Opaque(b), .. } if b.kind == OpaqueKind::Redacted
    )));
    let r = aggregate(events);
    match &r.items[0] {
        llm_xlate_core::Item::Reasoning(ri) => {
            assert!(ri.text.is_none());
            assert_eq!(ri.opaque.as_ref().unwrap().data, "REDACTED-BLOB");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn pause_turn_stop_reason() {
    let r = aggregate(decode_stream(PAUSE_TURN));
    assert_eq!(r.stop, StopReason::PauseTurn);
}

#[test]
fn stop_sequence_carries_the_sequence() {
    let r = aggregate(decode_stream(STOP_SEQUENCE));
    assert_eq!(r.stop, StopReason::StopSequence("STOP".into()));
}

#[test]
fn error_event_becomes_ir_error() {
    let events = decode_stream(ERROR_STREAM);
    match &events[0] {
        IrEvent::Error(e) => {
            assert_eq!(e.kind, llm_xlate_core::ErrorKind::Overloaded);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn malformed_sse_data_is_upstream_malformed() {
    let bad = "event: message_start\ndata: {not json\n\n";
    let events = decode_stream(bad);
    assert!(matches!(&events[0], IrEvent::Error(e) if e.kind == llm_xlate_core::ErrorKind::UpstreamMalformed));
}

#[test]
fn unknown_event_type_is_ignored() {
    let s = "event: some_future_event\ndata: {\"type\":\"some_future_event\",\"x\":1}\n\n";
    assert!(decode_stream(s).is_empty());
}

#[test]
fn prefill_usage_is_carried_on_start() {
    let events = decode_stream(TEXT_TOOL_TEXT);
    match &events[0] {
        IrEvent::Start { usage_prefill, model, response_id } => {
            assert_eq!(response_id.as_str(), "msg_1");
            assert_eq!(model, "claude-opus-5");
            assert_eq!(usage_prefill.as_ref().unwrap().input, 10);
        }
        other => panic!("{other:?}"),
    }
}

// -- chunk-boundary fuzz: split at every byte, assert identical events -------------------

fn assert_chunk_invariant(sse: &str) {
    let bytes = sse.as_bytes();
    let one_shot = decode_stream(sse);
    for i in 0..=bytes.len() {
        let mut d = AnthropicCodec.stream_decoder(&claude_5());
        let mut got = d.push(&bytes[..i]);
        got.extend(d.push(&bytes[i..]));
        got.extend(d.finish());
        assert_eq!(got, one_shot, "split at byte {i} changed the event stream");
    }
}

#[test]
fn chunk_fuzz_text_tool_text() {
    assert_chunk_invariant(TEXT_TOOL_TEXT);
}

#[test]
fn chunk_fuzz_thinking() {
    assert_chunk_invariant(THINKING);
}

#[test]
fn chunk_fuzz_redacted() {
    assert_chunk_invariant(REDACTED);
}

// -- stream encoder ---------------------------------------------------------------------

fn client_events() -> Vec<IrEvent> {
    let sig = OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "SIG==");
    vec![
        IrEvent::Start {
            response_id: ResponseId::new("msg_test_0001"),
            model: "claude-opus-5".into(),
            usage_prefill: Some(Usage::new(10, 1)),
        },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: None, call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningText("thinking...".into()) },
        IrEvent::Delta { index: 0, delta: Delta::Opaque(sig) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::ItemStart { index: 1, kind: ItemKind::Message, id: None, call: None },
        IrEvent::Delta { index: 1, delta: Delta::Text("Answer.".into()) },
        IrEvent::ItemStop { index: 1 },
        IrEvent::ItemStart {
            index: 2,
            kind: ItemKind::ToolCall,
            id: None,
            call: Some((CallId::new("toolu_1"), "get".into())),
        },
        IrEvent::Delta { index: 2, delta: Delta::ToolArgs(r#"{"q":"x"}"#.into()) },
        IrEvent::ItemStop { index: 2 },
        IrEvent::Stop { reason: StopReason::ToolUse, usage: Usage::new(10, 25), ext: Default::default() },
    ]
}

fn encode_stream(events: Vec<IrEvent>) -> Vec<u8> {
    let mut enc = AnthropicCodec.stream_encoder(ectx(llm_xlate_core::Protocol::Anthropic));
    let mut out = Vec::new();
    for e in events {
        for f in enc.push(e) {
            out.extend_from_slice(&f);
        }
    }
    for f in enc.finish() {
        out.extend_from_slice(&f);
    }
    out
}

#[test]
fn stream_encode_snapshot() {
    let bytes = encode_stream(client_events());
    insta::assert_snapshot!("stream_encode_frames", String::from_utf8(bytes).unwrap());
}

#[test]
fn stream_encode_dense_indices() {
    // Even though IR reasoning is buffered and flushed at ItemStop, Anthropic indices are a
    // dense 0..n; assert the emitted content_block_start indices are 0,1,2.
    let bytes = encode_stream(client_events());
    let text = String::from_utf8(bytes).unwrap();
    let indices: Vec<i64> = text
        .lines()
        .filter(|l| l.starts_with("data:") && l.contains("content_block_start"))
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l.trim_start_matches("data: ")).unwrap();
            v["index"].as_i64().unwrap()
        })
        .collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn stream_encode_summary_only_reasoning_omits_signature() {
    // A summary-only reasoning item with no opaque carrier: the client-facing stream renders a
    // `thinking` block carrying the summary text and NO signature (plan §7.2 exposure table),
    // matching the non-streaming encoder (law §11.8).
    let events = vec![
        IrEvent::Start {
            response_id: ResponseId::new("msg_test_0001"),
            model: "claude-opus-5".into(),
            usage_prefill: None,
        },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: None, call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 0, text: "Consider it.".into() } },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(1, 1), ext: Default::default() },
    ];
    let text = String::from_utf8(encode_stream(events)).unwrap();
    assert!(text.contains("\"type\":\"thinking\""), "expected a thinking block: {text}");
    assert!(text.contains("Consider it."));
    assert!(!text.contains("signature_delta"), "no signature when opaque absent: {text}");
    assert!(!text.contains("redacted_thinking"));
}

/// Parallel tool calls whose IR `ItemStop`s are interleaved (`start0, args0, start1, args1,
/// stop0, stop1`) — a Chat provider emits exactly this when it streams `tool_calls[0]` then
/// `tool_calls[1]`. Anthropic permits only one open content block at a time, so the encoder
/// must close block 0 before opening block 1.
fn interleaved_parallel_tools() -> Vec<IrEvent> {
    vec![
        IrEvent::Start {
            response_id: ResponseId::new("msg_test_0001"),
            model: "claude-opus-5".into(),
            usage_prefill: Some(Usage::new(9, 0)),
        },
        IrEvent::ItemStart {
            index: 0,
            kind: ItemKind::ToolCall,
            id: None,
            call: Some((CallId::new("call_a"), "get_weather".into())),
        },
        IrEvent::Delta { index: 0, delta: Delta::ToolArgs(r#"{"city":"NYC"}"#.into()) },
        IrEvent::ItemStart {
            index: 1,
            kind: ItemKind::ToolCall,
            id: None,
            call: Some((CallId::new("call_b"), "get_time".into())),
        },
        IrEvent::Delta { index: 1, delta: Delta::ToolArgs(r#"{"tz":"ET"}"#.into()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::ItemStop { index: 1 },
        IrEvent::Stop { reason: StopReason::ToolUse, usage: Usage::new(9, 12), ext: Default::default() },
    ]
}

#[test]
fn stream_encode_never_overlaps_content_blocks() {
    // Walk the emitted frames: a content_block_start must never appear while another block is
    // still open (Anthropic's strictly-sequential content-block streaming contract).
    let bytes = encode_stream(interleaved_parallel_tools());
    let text = String::from_utf8(bytes).unwrap();
    let mut open: Option<i64> = None;
    let mut starts = 0;
    for line in text.lines().filter(|l| l.starts_with("data:")) {
        let v: serde_json::Value = serde_json::from_str(line.trim_start_matches("data: ")).unwrap();
        match v["type"].as_str() {
            Some("content_block_start") => {
                assert!(
                    open.is_none(),
                    "content_block_start index={} while index={:?} is still open",
                    v["index"], open
                );
                open = v["index"].as_i64();
                starts += 1;
            }
            Some("content_block_stop") => {
                assert_eq!(open, v["index"].as_i64(), "content_block_stop for a non-open block");
                open = None;
            }
            _ => {}
        }
    }
    assert!(open.is_none(), "a content block was left open at end of stream");
    assert_eq!(starts, 2, "expected exactly two tool_use blocks");
}

#[test]
fn client_roundtrip_law() {
    // aggregate(decode_stream(encode_stream(events))) preserves items / stop / usage.
    let events = client_events();
    let mut want = aggregate(events.clone());
    // The provider-facing decoder stamps every native opaque carrier with the upstream model
    // from `message_start` (which the client-facing encoder wrote from `ctx.client_model`), so
    // the round-tripped reasoning blob is bound to `claude-opus-5`. Reflect that enrichment.
    stamp_reasoning_model(&mut want.items, "claude-opus-5");
    let bytes = encode_stream(events);
    let got = aggregate(decode_stream(&String::from_utf8(bytes).unwrap()));
    assert_eq!(got.items, want.items);
    assert_eq!(got.stop, want.stop);
    assert_eq!(got.usage, want.usage);
}

/// Set the `model` field on every reasoning item's opaque blob (mirrors the decoder's
/// upstream-model stamping across the envelope boundary).
fn stamp_reasoning_model(items: &mut [llm_xlate_core::Item], model: &str) {
    for it in items {
        if let llm_xlate_core::Item::Reasoning(r) = it {
            if let Some(b) = &mut r.opaque {
                b.model = Some(model.to_string());
            }
        }
    }
}

#[test]
fn keepalive_is_a_ping() {
    let mut enc = AnthropicCodec.stream_encoder(ectx(llm_xlate_core::Protocol::Anthropic));
    let ka = enc.keepalive().unwrap();
    let s = String::from_utf8(ka.to_vec()).unwrap();
    assert!(s.contains("event: ping"));
}

#[test]
fn encode_response_consistent_with_stream() {
    // encode_response(aggregate(stream)) has the same items / stop / usage the stream implies.
    let r = aggregate(decode_stream(TEXT_TOOL_TEXT));
    let body = AnthropicCodec.encode_response(&r, &ectx(llm_xlate_core::Protocol::Anthropic));
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["stop_reason"], serde_json::json!("tool_use"));
    assert_eq!(v["content"].as_array().unwrap().len(), 3);
    assert_eq!(v["usage"]["input_tokens"], serde_json::json!(10));
    assert_eq!(v["usage"]["output_tokens"], serde_json::json!(25));
}
