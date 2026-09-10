//! Store-support surface: `encode_stored_response` and `encode_input_items`.

mod common;
use common::*;

use llm_xlate_core::{
    CallId, Codec, Instruction, InstructionRole, Item, JsonText, OpaqueBlob, OpaqueKind, Part,
    Position, ProviderFamily, ReasoningItem, ResponseId, Role, Sealer, StopReason, Usage,
};
use llm_xlate_responses::{encode_input_items, encode_stored_response, StoredStatus, StoredView};
use pretty_assertions::assert_eq;
use serde_json::{Map, Value};

fn echo() -> Map<String, Value> {
    let ir = decode(r#"{"model":"gpt-5.4","instructions":"be nice","temperature":0.2}"#);
    codec().request_echo(&ir, &ectx())
}

#[allow(clippy::too_many_arguments)]
fn view<'a>(
    id: &'a ResponseId,
    status: StoredStatus,
    output: &'a [Item],
    usage: &'a Usage,
    stop: &'a StopReason,
    echo: &'a Map<String, Value>,
    request_items: &'a [Item],
    instructions: &'a [Instruction],
    prev: Option<&'a ResponseId>,
    err: Option<&'a llm_xlate_core::XlateError>,
) -> StoredView<'a> {
    StoredView {
        id,
        previous_id: prev,
        status,
        created_at: 1_726_000_000,
        model: "gpt-5.4",
        output_items: output,
        usage,
        stop,
        request_echo: echo,
        error: err,
        request_items,
        instructions,
    }
}

#[test]
fn stored_completed_response() {
    let id = ResponseId::new("resp_stored");
    let output = vec![Item::assistant_text("done")];
    let usage = Usage::new(10, 20);
    let stop = StopReason::EndTurn;
    let e = echo();
    let v = view(&id, StoredStatus::Completed, &output, &usage, &stop, &e, &[], &[], None, None);
    let b = json(&encode_stored_response(&v, &Sealer::new(KEY)));
    assert_eq!(b["object"], "response");
    assert_eq!(b["status"], "completed");
    assert_eq!(b["id"], "resp_stored");
    assert_eq!(b["instructions"], "be nice");
    assert_eq!(b["output"][0]["content"][0]["text"], "done");
    assert_eq!(b["output"][0]["id"], "msg_resp_stored_0");
    assert_eq!(b["usage"]["input_tokens"], 10);
}

#[test]
fn stored_incomplete_response() {
    let id = ResponseId::new("r");
    let output: Vec<Item> = vec![];
    let usage = Usage::new(1, 5);
    let stop = StopReason::MaxTokens;
    let e = echo();
    let v = view(&id, StoredStatus::Incomplete, &output, &usage, &stop, &e, &[], &[], None, None);
    let b = json(&encode_stored_response(&v, &Sealer::new(KEY)));
    assert_eq!(b["status"], "incomplete");
    assert_eq!(b["incomplete_details"]["reason"], "max_output_tokens");
}

#[test]
fn stored_failed_response() {
    let id = ResponseId::new("r");
    let output: Vec<Item> = vec![];
    let usage = Usage::new(0, 0);
    let stop = StopReason::EndTurn;
    let e = echo();
    let err = llm_xlate_core::XlateError::new(llm_xlate_core::ErrorKind::ServerError, "boom");
    let v = view(&id, StoredStatus::Failed, &output, &usage, &stop, &e, &[], &[], None, Some(&err));
    let b = json(&encode_stored_response(&v, &Sealer::new(KEY)));
    assert_eq!(b["status"], "failed");
    assert_eq!(b["error"]["message"], "boom");
    assert_eq!(b["usage"], Value::Null);
}

#[test]
fn stored_queued_and_cancelled_have_null_usage() {
    let id = ResponseId::new("r");
    let output: Vec<Item> = vec![];
    let usage = Usage::new(1, 1);
    let stop = StopReason::Cancelled;
    let e = echo();
    for status in [StoredStatus::Queued, StoredStatus::InProgress, StoredStatus::Cancelled] {
        let v = view(&id, status, &output, &usage, &stop, &e, &[], &[], None, None);
        let b = json(&encode_stored_response(&v, &Sealer::new(KEY)));
        assert_eq!(b["status"], status.as_str());
        assert_eq!(b["usage"], Value::Null);
    }
}

#[test]
fn stored_previous_id_is_authoritative() {
    let id = ResponseId::new("r2");
    let prev = ResponseId::new("r1");
    let output: Vec<Item> = vec![];
    let usage = Usage::new(1, 1);
    let stop = StopReason::EndTurn;
    let e = echo();
    let v = view(&id, StoredStatus::Completed, &output, &usage, &stop, &e, &[], &[], Some(&prev), None);
    let b = json(&encode_stored_response(&v, &Sealer::new(KEY)));
    assert_eq!(b["previous_response_id"], "r1");
}

#[test]
fn stored_reasoning_blob_is_sealed_on_read() {
    let id = ResponseId::new("r");
    let output = vec![Item::Reasoning(ReasoningItem {
        text: None,
        summary: vec!["s".into()],
        opaque: Some(OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "sig")),
        id: None,
    })];
    let usage = Usage::new(1, 1);
    let stop = StopReason::EndTurn;
    let e = echo();
    let v = view(&id, StoredStatus::Completed, &output, &usage, &stop, &e, &[], &[], None, None);
    let b = json(&encode_stored_response(&v, &Sealer::new(KEY)));
    let enc = b["output"][0]["encrypted_content"].as_str().unwrap();
    assert!(enc.starts_with("rtr1."));
}

#[test]
fn input_items_lists_instructions_and_items() {
    let id = ResponseId::new("resp_x");
    let usage = Usage::new(0, 0);
    let stop = StopReason::EndTurn;
    let e = echo();
    let instructions = vec![Instruction::system_text("sys")];
    let request_items = vec![
        Item::user_text("hi"),
        Item::assistant_text("hello"),
        Item::ToolCall {
            call_id: CallId::new("c1"),
            name: "get".into(),
            arguments: JsonText::new("{}"),
            id: None,
        },
    ];
    let v = view(&id, StoredStatus::Completed, &[], &usage, &stop, &e, &request_items, &instructions, None, None);
    let b = json(&encode_input_items(&v));
    assert_eq!(b["object"], "list");
    assert_eq!(b["has_more"], false);
    let data = b["data"].as_array().unwrap();
    // instruction, user, assistant, tool call = 4 items
    assert_eq!(data.len(), 4);
    assert_eq!(data[0]["role"], "system");
    assert_eq!(data[0]["id"], "msg_resp_x_0");
    assert_eq!(data[1]["role"], "user");
    assert_eq!(data[2]["role"], "assistant");
    assert_eq!(data[3]["type"], "function_call");
    assert_eq!(b["first_id"], "msg_resp_x_0");
    assert_eq!(b["last_id"], "fc_resp_x_3");
}

#[test]
fn input_items_before_instruction_interleaved() {
    let id = ResponseId::new("r");
    let usage = Usage::new(0, 0);
    let stop = StopReason::EndTurn;
    let e = echo();
    let instructions = vec![Instruction {
        role: InstructionRole::Developer,
        position: Position::Before(1),
        content: vec![Part::text("mid")],
        cache_control: None,
        effort: None,
        clear_at: None,
    }];
    let request_items = vec![Item::user_text("a"), Item::user_text("b")];
    let v = view(&id, StoredStatus::Completed, &[], &usage, &stop, &e, &request_items, &instructions, None, None);
    let b = json(&encode_input_items(&v));
    let data = b["data"].as_array().unwrap();
    // a, developer(mid), b
    assert_eq!(data.len(), 3);
    assert_eq!(data[1]["role"], "developer");
}

#[test]
fn input_items_empty_has_null_bounds() {
    let id = ResponseId::new("r");
    let usage = Usage::new(0, 0);
    let stop = StopReason::EndTurn;
    let e = echo();
    let v = view(&id, StoredStatus::Completed, &[], &usage, &stop, &e, &[], &[], None, None);
    let b = json(&encode_input_items(&v));
    assert_eq!(b["first_id"], Value::Null);
    assert_eq!(b["last_id"], Value::Null);
    assert_eq!(b["data"].as_array().unwrap().len(), 0);
}

#[test]
fn input_items_tool_result_and_role_match() {
    let id = ResponseId::new("r");
    let usage = Usage::new(0, 0);
    let stop = StopReason::EndTurn;
    let e = echo();
    let request_items = vec![Item::ToolResult {
        call_id: CallId::new("c1"),
        content: vec![Part::text("42")],
        is_error: false,
        id: None,
    }];
    let v = view(&id, StoredStatus::Completed, &[], &usage, &stop, &e, &request_items, &[], None, None);
    let b = json(&encode_input_items(&v));
    assert_eq!(b["data"][0]["type"], "function_call_output");
    assert_eq!(b["data"][0]["output"], "42");
}

#[test]
fn stored_determinism() {
    let id = ResponseId::new("r");
    let output = vec![Item::assistant_text("x")];
    let usage = Usage::new(1, 1);
    let stop = StopReason::EndTurn;
    let e = echo();
    let v = view(&id, StoredStatus::Completed, &output, &usage, &stop, &e, &[], &[], None, None);
    let a = encode_stored_response(&v, &Sealer::new(KEY));
    let b = encode_stored_response(&v, &Sealer::new(KEY));
    assert_eq!(a, b);
    let _ = Role::User;
}
