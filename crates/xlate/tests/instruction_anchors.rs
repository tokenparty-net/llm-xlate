//! Mid-context instruction anchors survive the lowering passes (plan §7.1, §7.2).
//!
//! Reduced from a captured production trace: an Anthropic-dialect agent talking to an
//! OpenAI-compatible backend over the Chat protocol. The transcript here is generic — tools
//! named `alpha`/`beta`/`gamma`, placeholder text, none of the captured content — but it keeps
//! the four features that made the original fail:
//!
//! 1. a foreign-family reasoning block that lowering drops, shifting every later item index;
//! 2. an in-array `role:"system"` message anchored after that point;
//! 3. one assistant turn carrying **two** `tool_use` blocks;
//! 4. a trailing in-array `role:"system"` message anchored at the end of the transcript.
//!
//! The backend rejected the translated request: *"tool messages need a resolvable tool name:
//! carry `tool`/`name`, or match a preceding assistant tool_call by order"*. Lowering had
//! dropped the reasoning item without re-mapping the `Position::Before` anchors, so the
//! mid-context system message landed *inside* the two-call assistant turn. That turn was
//! emitted as two separate assistant messages with both `tool` replies trailing the second,
//! leaving the backend no way to pair a result with its call. The trailing instruction, pushed
//! past the new end of the list, was dropped outright with no degradation.

mod common;

use common::*;
use llm_xlate::caps::{BackendOverrides, Capabilities, ToolsCap, Tri};
use llm_xlate::codec::TranslatorConfig;
use llm_xlate::ir::{Position, Protocol, ProviderFamily, ResponseId};
use llm_xlate::requirements::Resolutions;
use llm_xlate::{lower, HeaderMap, Translator};
use serde_json::Value;

/// The reduced transcript. Legal Anthropic: each in-array `system` follows a user message and
/// either precedes an assistant message or ends the array.
///
/// Note the results come back in the opposite order from the calls (`call_c` before `call_b`) —
/// an agent returns results as they finish, so pairing by position is not reliable even when
/// the turn structure is intact.
const REQUEST: &[u8] = br#"{
    "model": "test-model",
    "max_tokens": 64,
    "system": "LEADING",
    "messages": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "t", "signature": "sig"},
            {"type": "tool_use", "id": "call_a", "name": "alpha", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_a", "content": "ra"}]},
        {"role": "system", "content": "MID"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_b", "name": "beta", "input": {}},
            {"type": "tool_use", "id": "call_c", "name": "gamma", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_c", "content": "rc"},
            {"type": "tool_result", "tool_use_id": "call_b", "content": "rb"}]},
        {"role": "system", "content": "TAIL"}
    ]
}"#;

/// A concrete OpenAI-compatible backend: the shipped generic profile leaves tools `Unknown`,
/// so a tool-using route only exists once a backend opts in, as the router does here.
fn compat_caps(result_name: Tri) -> Capabilities {
    let overlay = BackendOverrides {
        tools: ToolsCap { function_tools: Tri::Yes, result_name, ..Default::default() },
        ..Default::default()
    };
    llm_xlate::caps::shipped().resolve(
        &ProviderFamily::Other("openai-compatible".to_string()),
        "test-model",
        Some(&overlay),
    )
}

/// Decode → lower → encode the fixture for a Chat backend, returning the wire messages.
fn encode_messages(result_name: Tri) -> Vec<Value> {
    let xl = Translator::new(TranslatorConfig::default());
    let caps = compat_caps(result_name);
    let ir = xl
        .decode_request(Protocol::Anthropic, REQUEST, &HeaderMap::new())
        .expect("fixture decodes");
    let low = lower(
        ir.clone(),
        &caps,
        Protocol::OaiChat,
        &Resolutions::new(),
        &TranslatorConfig::default(),
    )
    .expect("fixture lowers");
    let ctx = xl.encode_ctx(Protocol::OaiChat, &ir, ResponseId::new("resp_x"), 0);
    let enc =
        xl.encode_request(Protocol::OaiChat, &low.req, &caps, &ctx).expect("fixture encodes");
    let body: Value = serde_json::from_slice(&enc.body).expect("valid JSON body");
    body["messages"].as_array().cloned().expect("messages array")
}

fn role(m: &Value) -> &str {
    m["role"].as_str().unwrap_or_default()
}

fn call_ids(m: &Value) -> Vec<&str> {
    m["tool_calls"]
        .as_array()
        .map(|a| a.iter().filter_map(|t| t["id"].as_str()).collect())
        .unwrap_or_default()
}

/// The reported failure: every `tool` reply must be reachable from the assistant turn that made
/// the calls, with no other message wedged in between.
#[test]
fn a_dropped_reasoning_item_does_not_split_an_assistant_tool_turn() {
    let msgs = encode_messages(Tri::Unknown);

    // The two-call assistant turn survives as ONE message carrying both calls.
    let multi: Vec<_> = msgs.iter().filter(|m| call_ids(m).len() > 1).collect();
    assert_eq!(
        multi.len(),
        1,
        "the assistant turn with two tool_use blocks must stay one message; got {msgs:#?}"
    );
    assert_eq!(call_ids(multi[0]), vec!["call_b", "call_c"]);

    // Nothing separates an assistant `tool_calls` message from the `tool` replies to it.
    for (i, m) in msgs.iter().enumerate() {
        let want = call_ids(m).len();
        if want == 0 {
            continue;
        }
        let got = msgs[i + 1..].iter().take_while(|n| role(n) == "tool").count();
        assert_eq!(
            got, want,
            "assistant message {i} made {want} call(s) but {got} `tool` message(s) immediately \
             follow it; something was wedged into the turn: {msgs:#?}"
        );
    }
}

/// The second, silent half of the same bug: an anchor pushed past the end of the transcript was
/// dropped by the Chat encoder with no degradation, losing prompt text.
#[test]
fn every_instruction_reaches_the_wire() {
    let msgs = encode_messages(Tri::Unknown);
    let systems: Vec<&str> = msgs
        .iter()
        .filter(|m| role(m) == "system")
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert_eq!(systems, vec!["LEADING", "MID", "TAIL"]);
}

/// The mid-context instruction keeps its *place* in the transcript, not merely its existence:
/// it belongs after the first tool turn and before the two-call turn.
#[test]
fn the_mid_instruction_keeps_its_place() {
    let msgs = encode_messages(Tri::Unknown);
    let roles: Vec<&str> = msgs.iter().map(role).collect();
    assert_eq!(
        roles,
        vec![
            "system",    // LEADING
            "user",      // q1
            "assistant", // call_a
            "tool",      // ra
            "system",    // MID
            "assistant", // call_b + call_c
            "tool",      // rc
            "tool",      // rb
            "system",    // TAIL
        ]
    );
}

/// With `tools.result_name` declared, each `tool` message names its tool, so a backend that
/// would otherwise pair results with calls by position can resolve them directly — including
/// here, where the results come back in the opposite order from the calls.
#[test]
fn tool_results_are_named_when_the_backend_needs_it() {
    let msgs = encode_messages(Tri::Yes);
    let named: Vec<(&str, &str)> = msgs
        .iter()
        .filter(|m| role(m) == "tool")
        .map(|m| (m["tool_call_id"].as_str().unwrap(), m["name"].as_str().unwrap()))
        .collect();
    assert_eq!(
        named,
        vec![("call_a", "alpha"), ("call_c", "gamma"), ("call_b", "beta")]
    );

    // Conservative by default: an undeclared backend gets no `name` key at all.
    for m in encode_messages(Tri::Unknown).iter().filter(|m| role(m) == "tool") {
        assert!(m.get("name").is_none(), "name must be caps-gated: {m:#?}");
    }
}

// ===========================================================================================
// The anchor re-mapping itself, exercised directly.
// ===========================================================================================

/// Dropping items shifts an anchor down by the number of drops ahead of it.
#[test]
fn dropping_items_remaps_anchors() {
    let caps = compat_caps(Tri::Unknown);
    let items = vec![
        user("u"),                                   // 0
        reasoning_opaque(ProviderFamily::Anthropic), // 1 — foreign, dropped for a Chat target
        tool_call("c1", "alpha"),                    // 2
        tool_call("c2", "beta"),                     // 3
        tool_result("c1", "r1"),                     // 4
    ];
    let mut req = req_with_items(items);
    req.instructions = vec![mid_instruction(2, "AT-RUN-START"), mid_instruction(5, "TRAILING")];

    let low =
        lower(req, &caps, Protocol::OaiChat, &Resolutions::new(), &TranslatorConfig::default())
            .expect("lowers");

    assert_eq!(low.req.items.len(), 4);
    // `AT-RUN-START` pointed at the first tool call; one item ahead of it went away.
    assert_eq!(low.req.instructions[0].position, Position::Before(1));
    // The trailing anchor tracks the new end of the list.
    assert_eq!(low.req.instructions[1].position, Position::Before(4));
}

/// An anchor past the end of the transcript is clamped to the trailing slot and reported —
/// never silently discarded.
#[test]
fn an_out_of_range_anchor_is_clamped_and_reported() {
    let caps = compat_caps(Tri::Unknown);
    let mut req = req_with_items(vec![user("u")]);
    req.instructions = vec![mid_instruction(9, "ORPHAN")];

    let low =
        lower(req, &caps, Protocol::OaiChat, &Resolutions::new(), &TranslatorConfig::default())
            .expect("lowers");

    assert_eq!(low.req.instructions[0].position, Position::Before(1));
    assert!(
        low.degradations.iter().any(|d| d.field == "instructions.position"),
        "clamping must be reported: {:?}",
        low.degradations
    );
}
