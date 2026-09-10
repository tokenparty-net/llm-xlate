//! Streaming codecs: `ResponsesStreamDecoder` and `ResponsesStreamEncoder`, plus the laws
//! (chunk-boundary fuzz, decode↔encode round trip, `encode_response == completed.response`).

mod common;
use common::*;

use llm_xlate_core::{
    CallId, Codec, Delta, ErrorKind, IrEvent, IrResponse, Item, ItemId, ItemKind, OpaqueBlob,
    OpaqueKind, ProviderFamily, Role, SseParser, StopReason, Usage,
};
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

/// Build an SSE document from `(type, extra-fields)` frames, assigning sequence numbers.
fn sse(frames: &[(&str, Value)]) -> String {
    let mut s = String::new();
    for (i, (ty, extra)) in frames.iter().enumerate() {
        let mut m = serde_json::Map::new();
        m.insert("type".to_string(), Value::from(*ty));
        m.insert("sequence_number".to_string(), Value::from(i as u64));
        if let Value::Object(o) = extra {
            for (k, v) in o {
                m.insert(k.clone(), v.clone());
            }
        }
        s.push_str(&format!("event: {ty}\ndata: {}\n\n", serde_json::to_string(&Value::Object(m)).unwrap()));
    }
    s
}

fn run_decoder(bytes: &[u8]) -> Vec<IrEvent> {
    let mut d = codec().stream_decoder(&caps());
    let mut out = d.push(bytes);
    out.extend(d.finish());
    out
}

/// A canonical message stream.
fn message_stream() -> String {
    sse(&[
        ("response.created", json!({"response":{"id":"resp_1","model":"gpt-5.4"}})),
        ("response.output_item.added", json!({"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}})),
        ("response.content_part.added", json!({"output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":""}})),
        ("response.output_text.delta", json!({"output_index":0,"content_index":0,"item_id":"msg_1","delta":"He"})),
        ("response.output_text.delta", json!({"output_index":0,"content_index":0,"item_id":"msg_1","delta":"llo"})),
        ("response.output_text.done", json!({"output_index":0,"content_index":0,"item_id":"msg_1","text":"Hello"})),
        ("response.content_part.done", json!({"output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":"Hello"}})),
        ("response.output_item.done", json!({"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}})),
        ("response.completed", json!({"response":{"id":"resp_1","model":"gpt-5.4","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}})),
    ])
}

#[test]
fn decode_message_stream() {
    let events = run_decoder(message_stream().as_bytes());
    let resp = aggregate(&events);
    assert_eq!(resp.items, vec![Item::Message {
        role: Role::Assistant,
        content: vec![llm_xlate_core::Part::text("Hello")],
        id: Some(ItemId::new("msg_1")),
    }]);
    assert_eq!(resp.stop, StopReason::EndTurn);
    assert_eq!(resp.usage.input, 3);
    assert_eq!(resp.usage.output, 2);
}

/// A stream that accumulates function-call arguments and reconciles a `.done`.
fn tool_call_stream() -> String {
    sse(&[
        ("response.created", json!({"response":{"id":"r","model":"m"}})),
        ("response.output_item.added", json!({"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"c1","name":"get","arguments":""}})),
        ("response.function_call_arguments.delta", json!({"output_index":0,"item_id":"fc_1","delta":"{\"a\":"})),
        ("response.function_call_arguments.delta", json!({"output_index":0,"item_id":"fc_1","delta":"1"})),
        ("response.function_call_arguments.done", json!({"output_index":0,"item_id":"fc_1","arguments":"{\"a\":1}"})),
        ("response.output_item.done", json!({"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"c1","name":"get","arguments":"{\"a\":1}"}})),
        ("response.completed", json!({"response":{"id":"r","model":"m","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}})),
    ])
}

/// A stream carrying two reasoning-summary parts plus encrypted content.
fn reasoning_summary_stream() -> String {
    sse(&[
        ("response.created", json!({"response":{"id":"r","model":"gpt-5.4"}})),
        ("response.output_item.added", json!({"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}})),
        ("response.reasoning_summary_part.added", json!({"output_index":0,"summary_index":0,"item_id":"rs_1","part":{"type":"summary_text","text":""}})),
        ("response.reasoning_summary_text.delta", json!({"output_index":0,"summary_index":0,"item_id":"rs_1","delta":"first"})),
        ("response.reasoning_summary_part.added", json!({"output_index":0,"summary_index":1,"item_id":"rs_1","part":{"type":"summary_text","text":""}})),
        ("response.reasoning_summary_text.delta", json!({"output_index":0,"summary_index":1,"item_id":"rs_1","delta":"second"})),
        ("response.output_item.done", json!({"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"first"},{"type":"summary_text","text":"second"}],"encrypted_content":"ENC"}})),
        ("response.completed", json!({"response":{"id":"r","model":"gpt-5.4","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}})),
    ])
}

/// A text stream whose deltas carry multi-byte UTF-8 (emoji + CJK) so byte splits land mid-char.
fn emoji_text_stream() -> String {
    sse(&[
        ("response.created", json!({"response":{"id":"resp_1","model":"gpt-5.4"}})),
        ("response.output_item.added", json!({"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}})),
        ("response.content_part.added", json!({"output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":""}})),
        ("response.output_text.delta", json!({"output_index":0,"content_index":0,"item_id":"msg_1","delta":"Hi 😀"})),
        ("response.output_text.delta", json!({"output_index":0,"content_index":0,"item_id":"msg_1","delta":"日本語"})),
        ("response.output_text.done", json!({"output_index":0,"content_index":0,"item_id":"msg_1","text":"Hi 😀日本語"})),
        ("response.content_part.done", json!({"output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":"Hi 😀日本語","annotations":[]}})),
        ("response.output_item.done", json!({"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hi 😀日本語","annotations":[]}]}})),
        ("response.completed", json!({"response":{"id":"resp_1","model":"gpt-5.4","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}})),
    ])
}

#[test]
fn tool_call_stream_reconciles_done() {
    let stream = tool_call_stream();
    let events = run_decoder(stream.as_bytes());
    let resp = aggregate(&events);
    assert_eq!(resp.stop, StopReason::ToolUse);
    match &resp.items[0] {
        Item::ToolCall { arguments, call_id, .. } => {
            assert_eq!(arguments.as_str(), "{\"a\":1}");
            assert_eq!(call_id.as_str(), "c1");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn tool_call_done_disagreement_is_malformed() {
    let stream = sse(&[
        ("response.created", json!({"response":{"id":"r","model":"m"}})),
        ("response.output_item.added", json!({"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"c1","name":"get","arguments":""}})),
        ("response.function_call_arguments.delta", json!({"output_index":0,"item_id":"fc_1","delta":"XYZ"})),
        ("response.function_call_arguments.done", json!({"output_index":0,"item_id":"fc_1","arguments":"{\"a\":1}"})),
    ]);
    let events = run_decoder(stream.as_bytes());
    assert!(events.iter().any(|e| matches!(e, IrEvent::Error(x) if x.kind == ErrorKind::UpstreamMalformed)));
}

#[test]
fn sequence_gap_poisons_decoder() {
    // Manually craft a stream whose second event skips a sequence number.
    let mut s = String::new();
    s.push_str("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"r\",\"model\":\"m\"}}\n\n");
    s.push_str("event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":5,\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n");
    s.push_str("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":6,\"output_index\":0,\"delta\":\"hi\"}\n\n");
    let events = run_decoder(s.as_bytes());
    // Start emitted, then the gap raises an error and nothing further.
    assert!(matches!(events[0], IrEvent::Start { .. }));
    assert!(matches!(events[1], IrEvent::Error(ref e) if e.kind == ErrorKind::UpstreamMalformed));
    assert_eq!(events.len(), 2);
}

#[test]
fn reasoning_summary_streaming_multiple_parts() {
    let stream = reasoning_summary_stream();
    let events = run_decoder(stream.as_bytes());
    let resp = aggregate(&events);
    match &resp.items[0] {
        Item::Reasoning(ri) => {
            assert_eq!(ri.summary, vec!["first".to_string(), "second".to_string()]);
            assert_eq!(ri.opaque.as_ref().unwrap().data, "ENC");
            assert_eq!(ri.opaque.as_ref().unwrap().model.as_deref(), Some("gpt-5.4"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn incomplete_stream_stops_max_tokens() {
    let stream = sse(&[
        ("response.created", json!({"response":{"id":"r","model":"m"}})),
        ("response.incomplete", json!({"response":{"id":"r","model":"m","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":1,"output_tokens":5,"total_tokens":6}}})),
    ]);
    let events = run_decoder(stream.as_bytes());
    assert!(matches!(events.last().unwrap(), IrEvent::Stop { reason: StopReason::MaxTokens, .. }));
}

#[test]
fn failed_stream_is_error() {
    let stream = sse(&[
        ("response.created", json!({"response":{"id":"r","model":"m"}})),
        ("response.failed", json!({"response":{"id":"r","error":{"type":"server_error","message":"boom"}}})),
    ]);
    let events = run_decoder(stream.as_bytes());
    assert!(matches!(events.last().unwrap(), IrEvent::Error(e) if e.message == "boom"));
}

#[test]
fn top_level_error_event() {
    let mut s = String::new();
    s.push_str("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"r\",\"model\":\"m\"}}\n\n");
    s.push_str("event: error\ndata: {\"type\":\"error\",\"sequence_number\":1,\"code\":null,\"message\":\"bad\",\"param\":null}\n\n");
    let events = run_decoder(s.as_bytes());
    assert!(matches!(events.last().unwrap(), IrEvent::Error(_)));
}

#[test]
fn hosted_web_search_passthrough() {
    let stream = sse(&[
        ("response.created", json!({"response":{"id":"r","model":"m"}})),
        ("response.output_item.added", json!({"output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}})),
        ("response.web_search_call.searching", json!({"output_index":0,"item_id":"ws_1"})),
        ("response.web_search_call.completed", json!({"output_index":0,"item_id":"ws_1"})),
        ("response.output_item.done", json!({"output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"x"}}})),
        ("response.completed", json!({"response":{"id":"r","model":"m","usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}})),
    ]);
    let events = run_decoder(stream.as_bytes());
    let resp = aggregate(&events);
    match &resp.items[0] {
        Item::ProviderToolCall(oi) => {
            assert_eq!(oi.family, ProviderFamily::OpenAI);
            assert_eq!(oi.raw["action"]["query"], "x");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn chunk_boundary_fuzz_every_byte() {
    // Every SSE fixture is split at every byte boundary — including the stateful tool-args and
    // multi-part reasoning decoders and a stream whose deltas carry multi-byte UTF-8, so the
    // split-safe `SseParser` path is exercised mid-codepoint.
    let docs = [
        ("message", message_stream()),
        ("tool_call", tool_call_stream()),
        ("reasoning_summary", reasoning_summary_stream()),
        ("emoji", emoji_text_stream()),
    ];
    for (name, doc) in &docs {
        let bytes = doc.as_bytes();
        let one_shot = run_decoder(bytes);
        for split in 1..bytes.len() {
            let mut d = codec().stream_decoder(&caps());
            let mut got = d.push(&bytes[..split]);
            got.extend(d.push(&bytes[split..]));
            got.extend(d.finish());
            assert_eq!(got, one_shot, "{name}: split at {split}");
        }
    }
}

#[test]
fn stream_matches_nonstream_twin() {
    // The streaming aggregate equals the non-streaming aggregate for the same content.
    let stream_events = run_decoder(message_stream().as_bytes());
    let twin = r#"{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.4",
        "output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed",
          "content":[{"type":"output_text","text":"Hello","annotations":[]}]}],
        "usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}"#;
    let nonstream_events = codec().decode_response(twin.as_bytes(), &caps()).unwrap();
    let a = aggregate(&stream_events);
    let b = aggregate(&nonstream_events);
    assert_eq!(a.items, b.items);
    assert_eq!(a.stop, b.stop);
    assert_eq!(a.usage.input, b.usage.input);
    assert_eq!(a.usage.output, b.usage.output);
}

// ---- encoder ----

fn text_tool_reasoning_events() -> Vec<IrEvent> {
    vec![
        IrEvent::Start { response_id: llm_xlate_core::ResponseId::new("resp_test"), model: "gpt-5.4".into(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: Some(ItemId::new("rs_1")), call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 0, text: "think".into() } },
        IrEvent::ItemStop { index: 0 },
        IrEvent::ItemStart { index: 1, kind: ItemKind::Message, id: Some(ItemId::new("msg_1")), call: None },
        IrEvent::Delta { index: 1, delta: Delta::Text("Answer".into()) },
        IrEvent::ItemStop { index: 1 },
        IrEvent::ItemStart { index: 2, kind: ItemKind::ToolCall, id: Some(ItemId::new("fc_1")), call: Some((CallId::new("c1"), "get".into())) },
        IrEvent::Delta { index: 2, delta: Delta::ToolArgs("{\"a\":1}".into()) },
        IrEvent::ItemStop { index: 2 },
        IrEvent::Stop { reason: StopReason::ToolUse, usage: Usage::new(4, 6), ext: Default::default() },
    ]
}

fn parse_frames(frames: &[bytes::Bytes]) -> Vec<Value> {
    let mut parser = SseParser::new();
    let mut out = Vec::new();
    for f in frames {
        for ev in parser.push(f) {
            out.push(serde_json::from_str(&ev.data).unwrap());
        }
    }
    for ev in parser.finish() {
        out.push(serde_json::from_str(&ev.data).unwrap());
    }
    out
}

#[test]
fn encoder_emits_created_and_completed() {
    let ctx = ectx();
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = Vec::new();
    for ev in text_tool_reasoning_events() {
        frames.extend(enc.push(ev));
    }
    frames.extend(enc.finish());
    let events = parse_frames(&frames);
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(events[1]["type"], "response.in_progress");
    assert_eq!(events.last().unwrap()["type"], "response.completed");
    // Sequence numbers are monotonic from 0.
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e["sequence_number"], i as u64, "frame {i}");
    }
}

#[test]
fn encoder_completed_equals_encode_response() {
    let events = text_tool_reasoning_events();
    let ctx = ectx();
    // Streamed completed payload.
    let mut enc = codec().stream_encoder(ctx.clone());
    let mut frames = Vec::new();
    for ev in &events {
        frames.extend(enc.push(ev.clone()));
    }
    frames.extend(enc.finish());
    let parsed = parse_frames(&frames);
    let completed = parsed.iter().find(|e| e["type"] == "response.completed").unwrap();
    let streamed_response = &completed["response"];

    // Non-streaming encode of the same aggregate.
    let resp = aggregate(&events);
    let bytes = codec().encode_response(&resp, &ctx);
    let non_stream: Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(streamed_response, &non_stream);
}

#[test]
fn encoder_decoder_round_trip_law() {
    // aggregate(decode(encode(events))) == aggregate(events), modulo the documented id-mint /
    // usage-detail normalizations (compared here on items, stop, and input/output tokens).
    let events = text_tool_reasoning_events();
    let ctx = ectx();
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = Vec::new();
    for ev in &events {
        frames.extend(enc.push(ev.clone()));
    }
    frames.extend(enc.finish());
    let bytes: Vec<u8> = frames.iter().flat_map(|f| f.to_vec()).collect();
    let decoded = run_decoder(&bytes);

    let a = aggregate(&decoded);
    let b = aggregate(&events);
    assert_eq!(a.items, b.items);
    assert_eq!(a.stop, b.stop);
    assert_eq!(a.usage.input, b.usage.input);
    assert_eq!(a.usage.output, b.usage.output);
}

#[test]
fn encoder_full_exposure_streams_reasoning_text() {
    let mut ctx = ectx();
    ctx.expose = llm_xlate_core::ReasoningExposure::Full;
    let events = vec![
        IrEvent::Start { response_id: llm_xlate_core::ResponseId::new("resp_test"), model: "gpt-5.4".into(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: Some(ItemId::new("rs_1")), call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningText("raw thoughts".into()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(1, 1), ext: Default::default() },
    ];
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = Vec::new();
    for ev in &events {
        frames.extend(enc.push(ev.clone()));
    }
    frames.extend(enc.finish());
    let parsed = parse_frames(&frames);
    assert!(parsed.iter().any(|e| e["type"] == "response.reasoning_text.delta"));
    // The completed reasoning item carries the raw text in content.
    let completed = parsed.iter().find(|e| e["type"] == "response.completed").unwrap();
    let item = &completed["response"]["output"][0];
    assert_eq!(item["content"][0]["text"], "raw thoughts");
}

/// Reasoning-text-only events under a given exposure, for the leak / agreement checks below.
fn reasoning_text_only_events() -> Vec<IrEvent> {
    vec![
        IrEvent::Start { response_id: llm_xlate_core::ResponseId::new("resp_test"), model: "gpt-5.4".into(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: Some(ItemId::new("rs_1")), call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningText("SECRET chain of thought".into()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(1, 1), ext: Default::default() },
    ]
}

fn streamed_completed_response(ctx: llm_xlate_core::EncodeCtx, events: &[IrEvent]) -> Value {
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = Vec::new();
    for ev in events {
        frames.extend(enc.push(ev.clone()));
    }
    frames.extend(enc.finish());
    let parsed = parse_frames(&frames);
    parsed.iter().find(|e| e["type"] == "response.completed").unwrap()["response"].clone()
}

#[test]
fn encoder_none_exposure_drops_reasoning_text_no_leak() {
    // Under `None` (the default when a Responses client sends no `reasoning` block) the raw
    // chain-of-thought must NOT surface as a summary — neither in the streamed terminal snapshot
    // nor in the non-streaming encode — and the two must agree (plan §11.8).
    let mut ctx = ectx();
    ctx.expose = llm_xlate_core::ReasoningExposure::None;
    let events = reasoning_text_only_events();

    let streamed = streamed_completed_response(ctx.clone(), &events);
    let non_stream: Value =
        serde_json::from_slice(&codec().encode_response(&aggregate(&events), &ctx)).unwrap();

    assert_eq!(streamed, non_stream, "stream/non-stream disagree under None");
    let reasoning = &streamed["output"][0];
    assert_eq!(reasoning["type"], "reasoning");
    assert_eq!(reasoning["summary"], serde_json::json!([]), "reasoning text leaked into summary");
    let dump = serde_json::to_string(&streamed).unwrap();
    assert!(!dump.contains("SECRET"), "chain-of-thought leaked: {dump}");
}

#[test]
fn encoder_summary_exposure_folds_reasoning_text_and_agrees() {
    // Under `Summary`, reasoning-text-only folds into a single summary_text part, and the streamed
    // terminal snapshot matches the non-streaming encode (plan §7.2 fold + §11.8 agreement).
    let mut ctx = ectx();
    ctx.expose = llm_xlate_core::ReasoningExposure::Summary(llm_xlate_core::SummaryLevel::Auto);
    let events = reasoning_text_only_events();

    let streamed = streamed_completed_response(ctx.clone(), &events);
    let non_stream: Value =
        serde_json::from_slice(&codec().encode_response(&aggregate(&events), &ctx)).unwrap();

    assert_eq!(streamed, non_stream, "stream/non-stream disagree under Summary");
    let summary = &streamed["output"][0]["summary"];
    assert_eq!(summary[0]["type"], "summary_text");
    assert_eq!(summary[0]["text"], "SECRET chain of thought");
    // No raw reasoning `content` under Summary (that is a Full-only field).
    assert!(streamed["output"][0].get("content").is_none());
}

#[test]
fn encoder_refusal_stream() {
    let events = vec![
        IrEvent::Start { response_id: llm_xlate_core::ResponseId::new("resp_test"), model: "gpt-5.4".into(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: Some(ItemId::new("msg_1")), call: None },
        IrEvent::Delta { index: 0, delta: Delta::Refusal("cannot".into()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::Refusal, usage: Usage::new(1, 1), ext: Default::default() },
    ];
    let ctx = ectx();
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = Vec::new();
    for ev in &events {
        frames.extend(enc.push(ev.clone()));
    }
    frames.extend(enc.finish());
    let parsed = parse_frames(&frames);
    assert!(parsed.iter().any(|e| e["type"] == "response.refusal.delta"));
    let completed = parsed.iter().find(|e| e["type"] == "response.completed").unwrap();
    assert_eq!(completed["response"]["output"][0]["content"][0]["type"], "refusal");
}

#[test]
fn encoder_error_started_is_response_failed() {
    let ctx = ectx();
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = enc.push(IrEvent::Start { response_id: llm_xlate_core::ResponseId::new("resp_test"), model: "m".into(), usage_prefill: None });
    frames.extend(enc.push(IrEvent::Error(llm_xlate_core::XlateError::new(ErrorKind::ServerError, "boom"))));
    let parsed = parse_frames(&frames);
    assert!(parsed.iter().any(|e| e["type"] == "response.failed"));
}

#[test]
fn encoder_error_not_started_is_error_event() {
    let ctx = ectx();
    let mut enc = codec().stream_encoder(ctx);
    let frames = enc.push(IrEvent::Error(llm_xlate_core::XlateError::new(ErrorKind::ServerError, "boom")));
    let parsed = parse_frames(&frames);
    assert_eq!(parsed[0]["type"], "error");
}

#[test]
fn encoder_keepalive_is_comment() {
    let ctx = ectx();
    let mut enc = codec().stream_encoder(ctx);
    let ka = enc.keepalive().unwrap();
    assert!(ka.starts_with(b":"));
}

#[test]
fn encoder_incomplete_terminal() {
    let events = vec![
        IrEvent::Start { response_id: llm_xlate_core::ResponseId::new("resp_test"), model: "m".into(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: Some(ItemId::new("msg_1")), call: None },
        IrEvent::Delta { index: 0, delta: Delta::Text("partial".into()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::MaxTokens, usage: Usage::new(1, 9), ext: Default::default() },
    ];
    let ctx = ectx();
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = Vec::new();
    for ev in events {
        frames.extend(enc.push(ev));
    }
    let parsed = parse_frames(&frames);
    let last = parsed.last().unwrap();
    assert_eq!(last["type"], "response.incomplete");
    assert_eq!(last["response"]["incomplete_details"]["reason"], "max_output_tokens");
}

#[test]
fn encoder_reasoning_encrypted_when_stateless() {
    let mut ctx = ectx();
    ctx.store = Some(false);
    let events = vec![
        IrEvent::Start { response_id: llm_xlate_core::ResponseId::new("resp_test"), model: "gpt-5.4".into(), usage_prefill: None },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: Some(ItemId::new("rs_1")), call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 0, text: "s".into() } },
        IrEvent::Delta { index: 0, delta: Delta::Opaque(OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Encrypted, "ENC")) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(1, 1), ext: Default::default() },
    ];
    let mut enc = codec().stream_encoder(ctx);
    let mut frames = Vec::new();
    for ev in events {
        frames.extend(enc.push(ev));
    }
    let parsed = parse_frames(&frames);
    let completed = parsed.iter().find(|e| e["type"] == "response.completed").unwrap();
    assert_eq!(completed["response"]["output"][0]["encrypted_content"], "ENC");
}

#[test]
fn aggregate_helper_builds_response() {
    // Guard: the aggregate helper produces a well-formed response for provider items too.
    let resp: IrResponse = aggregate(&run_decoder(message_stream().as_bytes()));
    assert_eq!(resp.model, "gpt-5.4");
}
