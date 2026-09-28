//! Tests for the read-only `requirements` pre-pass.

mod common;
use common::*;

use llm_xlate::caps::preset;
use llm_xlate::ir::{Protocol, ProviderFamily, ReasoningItem, ResponseId};
use llm_xlate::requirements::{requirements, FileRef};
use pretty_assertions::assert_eq;

#[test]
fn chain_from_previous_response_id() {
    let mut req = req_with_items(vec![user("hi")]);
    req.state.previous_response_id = Some(ResponseId::new("resp_prev"));
    let r = requirements(&req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert_eq!(r.chain, Some(ResponseId::new("resp_prev")));
}

#[test]
fn no_chain_when_absent() {
    let req = req_with_items(vec![user("hi")]);
    let r = requirements(&req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert_eq!(r.chain, None);
}

#[test]
fn foreign_file_reported() {
    let req = req_with_items(vec![img_fileref(ProviderFamily::OpenAI, "file_1")]);
    let r = requirements(&req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(r.foreign_files, vec![FileRef::new(ProviderFamily::OpenAI, "file_1")]);
}

#[test]
fn same_family_file_not_reported() {
    let req = req_with_items(vec![img_fileref(ProviderFamily::Anthropic, "file_1")]);
    let r = requirements(&req, &preset::claude_5(), Protocol::Anthropic);
    assert!(r.foreign_files.is_empty());
}

#[test]
fn foreign_files_dedup_first_seen_order() {
    let req = req_with_items(vec![
        img_fileref(ProviderFamily::OpenAI, "file_b"),
        img_fileref(ProviderFamily::OpenAI, "file_a"),
        img_fileref(ProviderFamily::OpenAI, "file_b"), // dup
    ]);
    let r = requirements(&req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(
        r.foreign_files,
        vec![
            FileRef::new(ProviderFamily::OpenAI, "file_b"),
            FileRef::new(ProviderFamily::OpenAI, "file_a"),
        ]
    );
}

#[test]
fn reasoning_required_on_last_tool_turn() {
    // claude_5 sets required_on_last_tool_turn = true.
    let req = req_with_items(vec![
        user("q"),
        tool_call("call_1", "search"),
        tool_result("call_1", "res"),
    ]);
    let r = requirements(&req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(r.reasoning_for_calls, vec![llm_xlate::ir::CallId::new("call_1")]);
}

#[test]
fn reasoning_not_required_when_native_present() {
    let req = req_with_items(vec![
        user("q"),
        reasoning_opaque(ProviderFamily::Anthropic),
        tool_call("call_1", "search"),
        tool_result("call_1", "res"),
    ]);
    let r = requirements(&req, &preset::claude_5(), Protocol::Anthropic);
    assert!(r.reasoning_for_calls.is_empty());
}

#[test]
fn reasoning_not_required_when_text_replays_to_a_textfield_backend() {
    // A Chat backend that requires reasoning on the last tool turn and replays it through a text
    // field: the transcript's plain reasoning text is already the carrier, so the router must not
    // be asked to fetch a sidecar blob for it.
    let mut caps = preset::gpt4o();
    caps.reasoning.replay = Some(llm_xlate::caps::ReplayMode::TextField);
    caps.reasoning.required_on_last_tool_turn = llm_xlate::caps::Tri::Yes;
    let req = req_with_items(vec![
        user("q"),
        reasoning_text(),
        tool_call("call_1", "search"),
        tool_result("call_1", "res"),
    ]);
    let r = requirements(&req, &caps, Protocol::OaiChat);
    assert!(r.reasoning_for_calls.is_empty());
}

#[test]
fn reasoning_text_still_required_without_textfield_replay() {
    // The same text against a signature-replay backend has no wire carrier, so it stays required.
    let req = req_with_items(vec![
        user("q"),
        reasoning_text(),
        tool_call("call_1", "search"),
        tool_result("call_1", "res"),
    ]);
    let r = requirements(&req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(r.reasoning_for_calls, vec![llm_xlate::ir::CallId::new("call_1")]);
}

#[test]
fn reasoning_is_asked_for_even_when_not_required() {
    // gpt4o has required_on_last_tool_turn = false, but a client that dropped reasoning still
    // gets back whatever the router kept: the lookup is asked for regardless.
    let req = req_with_items(vec![
        user("q"),
        tool_call("call_1", "search"),
        tool_result("call_1", "res"),
    ]);
    let r = requirements(&req, &preset::gpt4o(), Protocol::OaiChat);
    assert_eq!(r.reasoning_for_calls, vec![llm_xlate::ir::CallId::new("call_1")]);
}

#[test]
fn reasoning_is_asked_for_across_the_turn_in_progress_only() {
    // Two tool rounds since the user's last message are both asked for; the round before that
    // message belongs to a finished turn and is not.
    let req = req_with_items(vec![
        user("first"),
        tool_call("call_old", "search"),
        tool_result("call_old", "res"),
        user("second"),
        tool_call("call_1", "search"),
        tool_result("call_1", "res"),
        tool_call("call_2", "clock"),
        tool_result("call_2", "9pm"),
    ]);
    let r = requirements(&req, &preset::gpt4o(), Protocol::OaiChat);
    let ids: Vec<&str> = r.reasoning_for_calls.iter().map(|c| c.as_str()).collect();
    assert_eq!(ids, vec!["call_1", "call_2"]);
}

#[test]
fn reasoning_foreign_opaque_still_needs_resolution() {
    // A foreign-family opaque does not count as native replayable → still required.
    let req = req_with_items(vec![
        user("q"),
        reasoning_opaque(ProviderFamily::OpenAI),
        tool_call("call_1", "search"),
        tool_result("call_1", "res"),
    ]);
    let r = requirements(&req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(r.reasoning_for_calls, vec![llm_xlate::ir::CallId::new("call_1")]);
}

#[test]
fn the_replay_carrier_is_the_opaque_blob_else_the_text() {
    // An opaque carrier is kept alone: it is what the backend checks, and before plain text
    // was kept, it was all a router stored.
    let signed = ReasoningItem {
        text: Some("thinking".into()),
        summary: vec!["a summary".into()],
        opaque: Some(blob(ProviderFamily::Anthropic)),
        id: None,
    };
    assert_eq!(signed.replay_carrier(), Some(resolved_blob(ProviderFamily::Anthropic)));
    // No carrier: the plain text, and nothing else.
    let plain = ReasoningItem { text: Some("thinking".into()), summary: vec!["a summary".into()], ..Default::default() };
    assert_eq!(plain.replay_carrier(), Some(resolved_text("thinking")));
    // A summary alone, or empty text, is nothing to replay.
    let summary_only = ReasoningItem { summary: vec!["a summary".into()], ..Default::default() };
    assert_eq!(summary_only.replay_carrier(), None);
    let empty = ReasoningItem { text: Some(String::new()), ..Default::default() };
    assert_eq!(empty.replay_carrier(), None);
}
