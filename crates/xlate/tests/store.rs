//! Tests for the stored-response model and chain materialization.

mod common;
use common::*;

use llm_xlate::ir::{IrRequest, Position, ProviderFamily, ResponseId, StopReason};
use llm_xlate::store::{
    chain_binding, materialize_chain, to_stored, BackendBinding, StoredResponse, StoredStatus,
};
use pretty_assertions::assert_eq;

fn binding() -> BackendBinding {
    BackendBinding::new("cred_1", ProviderFamily::OpenAI, "gpt-5.4")
}

fn stored(id: &str, prev: Option<&str>, request: Vec<llm_xlate::ir::Item>, output: Vec<llm_xlate::ir::Item>) -> StoredResponse {
    StoredResponse {
        id: ResponseId::new(id),
        previous_id: prev.map(ResponseId::new),
        instructions: Vec::new(),
        request_items: request,
        output_items: output,
        usage: llm_xlate::ir::Usage::new(1, 1),
        stop: StopReason::EndTurn,
        status: StoredStatus::Completed,
        binding: binding(),
        created_at: 100,
        request_echo: Default::default(),
        error: None,
    }
}

#[test]
fn materialize_three_element_chain() {
    let chain = vec![
        stored("r1", None, vec![user("u1")], vec![asst("a1")]),
        stored("r2", Some("r1"), vec![user("u2")], vec![asst("a2")]),
        stored("r3", Some("r2"), vec![user("u3")], vec![asst("a3")]),
    ];
    let mut new = req_with_items(vec![user("u4")]);
    new.state.previous_response_id = Some(ResponseId::new("r3"));

    let out = materialize_chain(&chain, new).unwrap();

    let expected = vec![
        user("u1"), asst("a1"),
        user("u2"), asst("a2"),
        user("u3"), asst("a3"),
        user("u4"),
    ];
    assert_eq!(out.items, expected);
    assert_eq!(out.state.previous_response_id, None);
}

#[test]
fn materialize_broken_link_errors() {
    let chain = vec![
        stored("r1", None, vec![user("u1")], vec![asst("a1")]),
        stored("r2", Some("WRONG"), vec![user("u2")], vec![asst("a2")]),
    ];
    let mut new = req_with_items(vec![user("u3")]);
    new.state.previous_response_id = Some(ResponseId::new("r2"));
    let err = materialize_chain(&chain, new).unwrap_err();
    assert_eq!(err.param.as_deref(), Some("previous_response_id"));
    // The message names the offending element index (1), its predecessor index (0), and both
    // ids so an operator can locate the break — guard the exact args against a formatting/
    // off-by-one regression (finding 12).
    assert!(
        err.message.contains("element 1")
            && err.message.contains("element 0")
            && err.message.contains("r2")
            && err.message.contains("r1"),
        "broken-chain message missing index/id detail: {}",
        err.message
    );
}

#[test]
fn materialize_new_prev_mismatch_errors() {
    let chain = vec![stored("r1", None, vec![user("u1")], vec![asst("a1")])];
    let mut new = req_with_items(vec![user("u2")]);
    new.state.previous_response_id = Some(ResponseId::new("r_other"));
    let err = materialize_chain(&chain, new).unwrap_err();
    assert_eq!(err.param.as_deref(), Some("previous_response_id"));
}

#[test]
fn materialize_shifts_before_positions() {
    // chain prepends 2 items (u1, a1); a Before(0) instruction must shift to Before(2).
    let chain = vec![stored("r1", None, vec![user("u1")], vec![asst("a1")])];
    let mut new = req_with_items(vec![user("u2")]);
    new.instructions = vec![mid_instruction(0, "mid")];
    new.state.previous_response_id = Some(ResponseId::new("r1"));

    let out = materialize_chain(&chain, new).unwrap();
    match &out.instructions[0].position {
        Position::Before(i) => assert_eq!(*i, 2),
        other => panic!("expected Before(2), got {other:?}"),
    }
    // The new request's instructions are used as-is (not resurrected from the chain).
    assert_eq!(out.instructions.len(), 1);
}

#[test]
fn materialize_uses_new_instructions_only() {
    let mut e1 = stored("r1", None, vec![user("u1")], vec![asst("a1")]);
    e1.instructions = vec![llm_xlate::ir::Instruction::system_text("old system")];
    let chain = vec![e1];
    let mut new = req_with_items(vec![user("u2")]);
    new.state.previous_response_id = Some(ResponseId::new("r1"));
    // new has no instructions → result has none (plan §7.1: do not resurrect prior).
    let out = materialize_chain(&chain, new).unwrap();
    assert!(out.instructions.is_empty());
}

#[test]
fn materialize_empty_chain_requires_no_prev() {
    let chain: Vec<StoredResponse> = Vec::new();
    let new = req_with_items(vec![user("u1")]);
    let out = materialize_chain(&chain, new).unwrap();
    assert_eq!(out.items, vec![user("u1")]);
}

#[test]
fn to_stored_derives_status_and_previous() {
    let mut req = req_with_items(vec![user("hi")]);
    req.state.previous_response_id = Some(ResponseId::new("r_prev"));
    let out = resp(vec![asst("ok")], StopReason::EndTurn);
    let s = to_stored(&req, &out, binding(), ResponseId::new("r_new"), 42, Default::default());
    assert_eq!(s.id, ResponseId::new("r_new"));
    assert_eq!(s.previous_id, Some(ResponseId::new("r_prev")));
    assert_eq!(s.status, StoredStatus::Completed);
    assert_eq!(s.created_at, 42);
    assert_eq!(s.output_items, vec![asst("ok")]);
}

#[test]
fn stored_status_from_stop_variants() {
    assert_eq!(StoredStatus::from_stop(&StopReason::MaxTokens), StoredStatus::Incomplete);
    assert_eq!(StoredStatus::from_stop(&StopReason::ContentFilter), StoredStatus::Incomplete);
    // A paused turn (Anthropic `pause_turn`) is interrupted, not terminal → Incomplete.
    assert_eq!(StoredStatus::from_stop(&StopReason::PauseTurn), StoredStatus::Incomplete);
    assert_eq!(StoredStatus::from_stop(&StopReason::Cancelled), StoredStatus::Cancelled);
    assert_eq!(StoredStatus::from_stop(&StopReason::EndTurn), StoredStatus::Completed);
    assert_eq!(StoredStatus::from_stop(&StopReason::ToolUse), StoredStatus::Completed);
}

#[test]
fn chain_binding_returns_last() {
    let chain = vec![
        stored("r1", None, vec![], vec![]),
        stored("r2", Some("r1"), vec![], vec![]),
    ];
    let b = chain_binding(&chain).unwrap();
    assert_eq!(b.credential_id, "cred_1");
    assert!(chain_binding(&[]).is_none());
}

#[test]
fn stored_response_json_roundtrip() {
    let s = stored("r1", Some("r0"), vec![user("u1")], vec![asst("a1")]);
    let json = serde_json::to_string(&s).unwrap();
    let back: StoredResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(s, back);
}

#[test]
fn materialize_preserves_other_new_fields() {
    let chain = vec![stored("r1", None, vec![user("u1")], vec![asst("a1")])];
    let mut new = IrRequest { items: vec![user("u2")], ..Default::default() };
    new.stream = true;
    new.state.previous_response_id = Some(ResponseId::new("r1"));
    let out = materialize_chain(&chain, new).unwrap();
    assert!(out.stream);
}
