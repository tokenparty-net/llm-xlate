#![allow(clippy::result_large_err, clippy::field_reassign_with_default)]
//! Streaming decoder / encoder: aggregation, chunk-boundary fuzz, client round-trip laws.

mod common;

use common::*;
use llm_xlate_core::Codec;
use llm_xlate_core::ir::{
    CallId, Delta, IrEvent, Item, ItemKind, OpaqueBlob, OpaqueKind, ProviderFamily, ResponseId,
    StopReason, Usage,
};
use pretty_assertions::assert_eq;

const STREAM_TEXT: &str = include_str!("fixtures/stream_text.txt");
const STREAM_NO_USAGE: &str = include_str!("fixtures/stream_no_usage.txt");
const STREAM_TOOLS: &str = include_str!("fixtures/stream_tools.txt");
const STREAM_REASONING: &str = include_str!("fixtures/stream_reasoning.txt");
const STREAM_REFUSAL: &str = include_str!("fixtures/stream_refusal.txt");

fn all_streams() -> Vec<(&'static str, &'static str)> {
    vec![
        ("text", STREAM_TEXT),
        ("no_usage", STREAM_NO_USAGE),
        ("tools", STREAM_TOOLS),
        ("reasoning", STREAM_REASONING),
        ("refusal", STREAM_REFUSAL),
    ]
}

// ---------------------------------------------------------------- decode

#[test]
fn decode_text_stream() {
    let r = aggregate(decode_stream(STREAM_TEXT, &gpt4o()));
    assert_eq!(r.id, ResponseId::new("chatcmpl-s1"));
    assert_eq!(r.stop, StopReason::EndTurn);
    assert_eq!(r.items.len(), 1);
    assert_eq!(r.items[0].as_text_opt(), Some("Hello, world!".to_string()));
    assert_eq!(r.usage.input, 10);
    assert_eq!(r.usage.output, 3);
}

#[test]
fn decode_no_usage_stream() {
    let r = aggregate(decode_stream(STREAM_NO_USAGE, &gpt4o()));
    assert_eq!(r.stop, StopReason::EndTurn);
    assert_eq!(r.items[0].as_text_opt(), Some("No usage here".to_string()));
    // No usage chunk was sent → default zero usage.
    assert_eq!(r.usage.input, 0);
    assert_eq!(r.usage.output, 0);
}

#[test]
fn decode_tools_stream() {
    let events = decode_stream(STREAM_TOOLS, &gpt4o());
    let r = aggregate(events);
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
fn decode_reasoning_stream() {
    let r = aggregate(decode_stream(STREAM_REASONING, &gpt4o()));
    let ri = r
        .items
        .iter()
        .find_map(|i| match i {
            Item::Reasoning(ri) => Some(ri),
            _ => None,
        })
        .expect("reasoning item");
    assert_eq!(ri.text.as_deref(), Some("Let me think. 17*23=391."));
    let op = ri.opaque.as_ref().expect("opaque");
    assert_eq!(op.family, ProviderFamily::OpenAI);
    assert_eq!(op.data, "enc-stream-blob");
    // The message text opens a new item after the reasoning item closes.
    assert_eq!(r.items.last().unwrap().as_text_opt(), Some("The answer is 391.".to_string()));
    assert_eq!(r.usage.reasoning, Some(30));
}

#[test]
fn decode_refusal_stream() {
    let r = aggregate(decode_stream(STREAM_REFUSAL, &gpt4o()));
    assert_eq!(r.stop, StopReason::Refusal);
    match &r.items[0] {
        Item::Message { content, .. } => {
            assert_eq!(
                content[0],
                llm_xlate_core::ir::Part::Refusal { text: "I cannot help with that.".to_string() }
            );
        }
        other => panic!("expected message, got {other:?}"),
    }
    assert!(r.ext.get("stop_details").is_some());
}

// ---------------------------------------------------------------- chunk-boundary fuzz

#[test]
fn chunk_boundary_fuzz_all_streams() {
    for (name, sse) in all_streams() {
        let whole = decode_stream(sse, &gpt4o());
        let split = decode_stream_split(sse, &gpt4o());
        assert_eq!(whole, split, "byte-split decode differs for stream {name}");
    }
}

// ---------------------------------------------------------------- malformed

#[test]
fn malformed_chunk_yields_error() {
    let sse = "data: {not valid json}\n\n";
    let events = decode_stream(sse, &gpt4o());
    assert!(events.iter().any(|e| matches!(e, IrEvent::Error(_))));
}

// ---------------------------------------------------------------- encoder

fn text_events() -> Vec<IrEvent> {
    vec![
        IrEvent::Start {
            response_id: ResponseId::new("chatcmpl-enc"),
            model: "gpt-4o".to_string(),
            usage_prefill: None,
        },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
        IrEvent::Delta { index: 0, delta: Delta::Text("Hello".to_string()) },
        IrEvent::Delta { index: 0, delta: Delta::Text(", world!".to_string()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop {
            reason: StopReason::EndTurn,
            usage: Usage::new(10, 3),
            ext: Default::default(),
        },
    ]
}

fn tool_events() -> Vec<IrEvent> {
    vec![
        IrEvent::Start {
            response_id: ResponseId::new("chatcmpl-enc2"),
            model: "gpt-4o".to_string(),
            usage_prefill: None,
        },
        IrEvent::ItemStart {
            index: 0,
            kind: ItemKind::ToolCall,
            id: None,
            call: Some((CallId::new("call_a"), "get_weather".to_string())),
        },
        IrEvent::Delta { index: 0, delta: Delta::ToolArgs("{\"city\":\"Paris\"}".to_string()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::ItemStart {
            index: 1,
            kind: ItemKind::ToolCall,
            id: None,
            call: Some((CallId::new("call_b"), "get_weather".to_string())),
        },
        IrEvent::Delta { index: 1, delta: Delta::ToolArgs("{\"city\":\"Tokyo\"}".to_string()) },
        IrEvent::ItemStop { index: 1 },
        IrEvent::Stop {
            reason: StopReason::ToolUse,
            usage: Usage::new(40, 30),
            ext: Default::default(),
        },
    ]
}

#[test]
fn encoder_first_chunk_has_role() {
    let sse = encode_stream(text_events(), encode_ctx("gpt-4o", false));
    let first = sse_data_lines(&sse)[0].clone();
    assert!(first.contains("\"role\":\"assistant\""));
    assert!(first.contains("\"content\":\"\""));
}

#[test]
fn encoder_ends_with_done() {
    let sse = encode_stream(text_events(), encode_ctx("gpt-4o", false));
    assert_eq!(sse_data_lines(&sse).last().unwrap(), "[DONE]");
}

#[test]
fn encoder_client_round_trip_text() {
    let events = text_events();
    let sse = encode_stream(events.clone(), encode_ctx("gpt-4o", true));
    let round = aggregate(decode_stream(&sse, &gpt4o()));
    let direct = aggregate(events);
    assert_eq!(round.items, direct.items);
    assert_eq!(round.stop, direct.stop);
    assert_eq!(round.usage.input, direct.usage.input);
    assert_eq!(round.usage.output, direct.usage.output);
}

#[test]
fn encoder_client_round_trip_tools() {
    let events = tool_events();
    let sse = encode_stream(events.clone(), encode_ctx("gpt-4o", true));
    let round = aggregate(decode_stream(&sse, &gpt4o()));
    let direct = aggregate(events);
    assert_eq!(round.items, direct.items);
    assert_eq!(round.stop, direct.stop);
}

#[test]
fn encoder_include_usage_emits_usage_chunk() {
    let on = encode_stream(text_events(), encode_ctx("gpt-4o", true));
    assert!(on.contains("\"usage\""));
    let off = encode_stream(text_events(), encode_ctx("gpt-4o", false));
    assert!(!off.contains("\"usage\""));
}

#[test]
fn encoder_tool_ordinal_indices() {
    let sse = encode_stream(tool_events(), encode_ctx("gpt-4o", false));
    // First tool head carries index 0, second carries index 1.
    assert!(sse.contains("\"index\":0,\"id\":\"call_a\""));
    assert!(sse.contains("\"index\":1,\"id\":\"call_b\""));
}

#[test]
fn encoder_seals_opaque_into_reasoning_details() {
    let events = vec![
        IrEvent::Start {
            response_id: ResponseId::new("r"),
            model: "gpt-5.4".to_string(),
            usage_prefill: None,
        },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: None, call: None },
        IrEvent::Delta { index: 0, delta: Delta::ReasoningText("thinking".to_string()) },
        IrEvent::Delta {
            index: 0,
            delta: Delta::Opaque(OpaqueBlob::new(
                ProviderFamily::OpenAI,
                OpaqueKind::Encrypted,
                "enc-abc",
            )),
        },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop {
            reason: StopReason::EndTurn,
            usage: Usage::new(1, 1),
            ext: Default::default(),
        },
    ];
    let sse = encode_stream(events, encode_ctx("gpt-5.4", false));
    assert!(sse.contains("reasoning_details"));
    assert!(sse.contains("router.opaque"));
    assert!(sse.contains("rtr1."));
}

#[test]
fn encoder_error_then_done() {
    let events = vec![IrEvent::Error(llm_xlate_core::XlateError::upstream_malformed("boom"))];
    let sse = encode_stream(events, encode_ctx("gpt-4o", false));
    let lines = sse_data_lines(&sse);
    assert!(lines[0].contains("\"error\""));
    assert_eq!(lines.last().unwrap(), "[DONE]");
}

#[test]
fn keepalive_is_sse_comment() {
    let mut enc = codec().stream_encoder(encode_ctx("gpt-4o", false));
    let frame = enc.keepalive().expect("keepalive");
    let s = String::from_utf8(frame.to_vec()).unwrap();
    assert!(s.starts_with(": keepalive"));
}

// ---------------------------------------------------------------- determinism

#[test]
fn encoder_determinism() {
    let a = encode_stream(tool_events(), encode_ctx("gpt-4o", true));
    let b = encode_stream(tool_events(), encode_ctx("gpt-4o", true));
    assert_eq!(a, b);
}

// small helper on Item for tests
trait AsTextOpt {
    fn as_text_opt(&self) -> Option<String>;
}
impl AsTextOpt for Item {
    fn as_text_opt(&self) -> Option<String> {
        match self {
            Item::Message { content, .. } => {
                let s: String = content.iter().filter_map(|p| p.as_text()).collect();
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------- §11.8 streaming law

/// Plan §11.8: for one `IrEvent` sequence, `encode_response(aggregate(ev))` and the aggregated
/// re-decode of `encode_stream(ev)` must agree on items, stop reason, and usage — detail
/// counters (reasoning_tokens plus counters preserved via `usage.ext[chat.completion_*]`)
/// included. This pins the stream encoder and `encode_response` to the same usage shape.
#[test]
fn stream_and_response_encoders_agree() {
    let mut events = tool_events();
    // Terminal Stop carries a reasoning counter plus an extra completion detail counter that
    // decode stashes under `chat.completion_*`; both encoders must round-trip both.
    if let Some(IrEvent::Stop { usage, .. }) = events.last_mut() {
        usage.reasoning = Some(7);
        usage
            .ext
            .insert("chat.completion_accepted_prediction_tokens", serde_json::json!(2));
    }

    let resp = aggregate(events.clone());

    // Response path: encode → re-decode → aggregate.
    let via_response = aggregate(decode_response(
        &encode_response(&resp, &encode_ctx("gpt-4o", false)),
        &gpt4o(),
    ));
    // Stream path: encode (usage on) → re-decode → aggregate.
    let via_stream = aggregate(decode_stream(
        &encode_stream(events, encode_ctx("gpt-4o", true)),
        &gpt4o(),
    ));

    assert_eq!(via_response.items, via_stream.items);
    assert_eq!(via_response.stop, via_stream.stop);
    assert_eq!(via_response.usage.input, via_stream.usage.input);
    assert_eq!(via_response.usage.output, via_stream.usage.output);
    assert_eq!(via_response.usage.reasoning, via_stream.usage.reasoning);
    // Both encoders re-emitted the reasoning counter and the extra completion detail counter.
    assert_eq!(via_response.usage.reasoning, Some(7));
    let key = "chat.completion_accepted_prediction_tokens";
    assert_eq!(via_response.usage.ext.get(key), Some(&serde_json::json!(2)));
    assert_eq!(via_stream.usage.ext.get(key), via_response.usage.ext.get(key));
}

// ---------------------------------------------------------------- mid-stream error

#[test]
fn decode_midstream_error_frame_surfaces_error() {
    // A gateway that returns 200 + partial content and *then* emits `data: {"error":...}` must
    // surface an `IrEvent::Error` (so the router renders a client error frame and stops), not let
    // the error deserialize into an empty chunk and fabricate a clean finish.
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-e1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,",
        "\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"upstream boom\",\"type\":\"server_error\"}}\n\n",
    );
    let events = decode_stream(sse, &gpt4o());
    assert!(
        events.iter().any(|e| matches!(e, IrEvent::Error(_))),
        "expected an IrEvent::Error, got {events:?}"
    );
    // And no fabricated clean stop after the error.
    assert!(
        !events.iter().any(|e| matches!(e, IrEvent::Stop { reason: StopReason::EndTurn, .. })),
        "must not fabricate a clean EndTurn stop after an error: {events:?}"
    );
}

#[test]
fn decode_midstream_error_split_bytes_surfaces_error() {
    // The same, but fed one byte at a time (chunk-boundary robustness).
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-e2\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,",
        "\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n",
    );
    let events = decode_stream_split(sse, &gpt4o());
    assert!(
        events.iter().any(|e| matches!(e, IrEvent::Error(_))),
        "expected an IrEvent::Error, got {events:?}"
    );
}

/// vLLM (and some gateways) attach a running `usage` to EVERY chunk, `completion_tokens: 0` on
/// the first. The decoder must not treat that as the terminal usage chunk (it previously emitted
/// `Stop` after the first frame and dropped all content — found live against a vLLM backend).
const STREAM_VLLM_PER_CHUNK_USAGE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"\",\"role\":\"assistant\"},\"finish_reason\":null,\"index\":0}],\"created\":1,\"id\":\"chatcmpl-v1\",\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":0,\"prompt_tokens\":9,\"total_tokens\":9}}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"p\"},\"finish_reason\":null,\"index\":0}],\"created\":1,\"id\":\"chatcmpl-v1\",\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":1,\"prompt_tokens\":9,\"total_tokens\":10}}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"ong\"},\"finish_reason\":\"stop\",\"index\":0}],\"created\":1,\"id\":\"chatcmpl-v1\",\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":3,\"prompt_tokens\":9,\"total_tokens\":12}}\n\n\
data: {\"id\":\"chatcmpl-v1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-flash\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"total_tokens\":12,\"completion_tokens\":3}}\n\n\
data: [DONE]\n\n";

#[test]
fn decode_vllm_per_chunk_usage_keeps_streaming() {
    let r = aggregate(decode_stream(STREAM_VLLM_PER_CHUNK_USAGE, &gpt4o()));
    assert_eq!(r.stop, StopReason::EndTurn);
    assert_eq!(r.items.len(), 1, "content must not be dropped: {:?}", r.items);
    assert_eq!(r.items[0].as_text_opt(), Some("pong".to_string()));
    assert_eq!(r.usage.input, 9);
    assert_eq!(r.usage.output, 3);
}

#[test]
fn decode_per_chunk_usage_without_final_usage_chunk_uses_last_seen() {
    // Same shape but no trailing usage-only chunk: the last per-chunk usage applies at [DONE].
    let s = STREAM_VLLM_PER_CHUNK_USAGE.replace(
        "data: {\"id\":\"chatcmpl-v1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-flash\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"total_tokens\":12,\"completion_tokens\":3}}\n\n",
        "",
    );
    let r = aggregate(decode_stream(&s, &gpt4o()));
    assert_eq!(r.items[0].as_text_opt(), Some("pong".to_string()));
    assert_eq!(r.usage.output, 3);
}

/// Regression: a vLLM-style stream where the chunk carrying `finish_reason` also carries a
/// usage object, and the cache counters arrive only on the *following* choices-empty chunk.
///
/// The decoder used to finalize on the finish chunk, which set `stopped` and made the trailing
/// chunk unreachable — silently dropping `prompt_tokens_details` and with it every cache
/// counter. This is the verbatim upstream stream from a live capture: Kimi K3 behind vLLM,
/// request `01a09585-ca71-7732-8f35-9493fe379812`, 2026-09-12, where `created_cache_tokens:
/// 9216` on a 9414-token prompt reached the IR as nothing at all.
const STREAM_VLLM_CACHE_AFTER_FINISH: &str =
    include_str!("fixtures/stream_vllm_cache_after_finish.txt");

#[test]
fn cache_counters_on_the_chunk_after_finish_reason_are_not_dropped() {
    let r = aggregate(decode_stream(STREAM_VLLM_CACHE_AFTER_FINISH, &gpt4o()));
    assert_eq!(r.stop, StopReason::EndTurn);
    assert_eq!(r.usage.cache_write, Some(9216), "cache write dropped");
    assert_eq!(r.usage.cache_read, Some(0));
    assert_eq!(r.usage.output, 46);
    // 9414 gross - 0 read - 9216 written = 198 genuinely fresh prompt tokens.
    assert_eq!(r.usage.input, 198);
    assert_eq!(r.usage.gross_prompt(), 9414);
}

#[test]
fn cache_counters_after_finish_survive_arbitrary_chunk_boundaries() {
    // The same stream fed one byte at a time must produce the identical usage.
    let whole = aggregate(decode_stream(STREAM_VLLM_CACHE_AFTER_FINISH, &gpt4o()));
    let split = aggregate(decode_stream_split(STREAM_VLLM_CACHE_AFTER_FINISH, &gpt4o()));
    assert_eq!(whole.usage, split.usage);
}
