//! Client `x-anthropic-<name>:` system header blocks (Claude Code's billing line): captured out
//! of the prompt on decode, re-emitted on encode only when `instructions.system_headers` lists
//! the name.

mod common;

use common::*;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_core::{Capabilities, Codec, EncodedRequest, IrRequest, Part, Protocol};
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

const CLAUDE_CODE: &str = include_str!("fixtures/req_claude_code_system.json");
const BILLING: &str = "x-anthropic-billing-header: cc_version=2.1.278.a6d; cc_entrypoint=cli; \
                       cch=27699; cc_prompt_id=ab501453-a14f-450c-8e60-e9cb5eadabb6; cc_turn_origin=human;";

fn decode(body: &str) -> IrRequest {
    AnthropicCodec.decode_request(body.as_bytes(), &no_headers(), &dctx()).unwrap()
}

fn encode(req: &IrRequest, caps: &Capabilities) -> (Value, EncodedRequest) {
    let out = AnthropicCodec.encode_request(req, caps, &ectx(Protocol::Anthropic)).unwrap();
    (serde_json::from_slice(&out.body).unwrap(), out)
}

fn system_texts(body: &Value) -> Vec<String> {
    body["system"]
        .as_array()
        .map(|a| a.iter().map(|b| b["text"].as_str().unwrap().to_string()).collect())
        .unwrap_or_default()
}

fn leading_texts(req: &IrRequest) -> Vec<String> {
    req.instructions
        .iter()
        .flat_map(|i| &i.content)
        .filter_map(|p| match p {
            Part::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// Caps that accept no system headers (connector-claude, any non-first-party backend).
fn no_headers_caps() -> Capabilities {
    let mut c = claude_5();
    c.instructions.system_headers = None;
    c
}

fn with_system(system: Value) -> String {
    json!({"model": "claude-opus-5", "max_tokens": 64, "system": system,
           "messages": [{"role": "user", "content": "hi"}]})
    .to_string()
}

#[test]
fn header_block_is_captured_out_of_the_prompt() {
    let req = decode(CLAUDE_CODE);
    assert!(leading_texts(&req).iter().all(|t| !t.starts_with("x-anthropic-")));
    assert_eq!(
        req.ext.get("anthropic.system_headers"),
        Some(&json!([{"name": "billing-header", "text": BILLING}]))
    );
}

#[test]
fn only_whole_single_line_header_blocks_qualify() {
    let multi = format!("{BILLING}\nThe real prompt.");
    let req = decode(&with_system(json!([
        {"type": "text", "text": multi},
        {"type": "text", "text": "x-anthropic-Bad_Name: v"},
        {"type": "text", "text": "x-other-header: v"},
    ])));
    assert_eq!(leading_texts(&req).len(), 3, "none of these is a header block");
    assert!(req.ext.get("anthropic.system_headers").is_none());

    // A plain-string system is prompt text, never scanned.
    let req = decode(&with_system(json!(BILLING)));
    assert_eq!(leading_texts(&req), vec![BILLING.to_string()]);
}

#[test]
fn header_only_system_leaves_no_empty_instruction() {
    let req = decode(&with_system(json!([{"type": "text", "text": BILLING}])));
    assert!(req.instructions.is_empty());
    let (body, _) = encode(&req, &no_headers_caps());
    assert!(body.get("system").is_none(), "no empty system array: {body}");
}

#[test]
fn listed_header_is_forwarded_first_and_verbatim() {
    let req = decode(CLAUDE_CODE);
    let (body, out) = encode(&req, &claude_5()); // first-party: ["billing-header"]
    let texts = system_texts(&body);
    assert_eq!(texts[0], BILLING);
    assert_eq!(texts[1], "You are Claude Code, Anthropic's official CLI for Claude.");
    assert_eq!(texts.len(), 4);
    assert!(body.get("system_headers").is_none(), "ext must not leak into the body");
    assert!(out.degradations.iter().all(|d| !d.field.contains("system_headers")));
}

#[test]
fn unlisted_header_is_dropped_and_reported() {
    let req = decode(CLAUDE_CODE);
    let (body, out) = encode(&req, &no_headers_caps());
    let texts = system_texts(&body);
    assert_eq!(texts.len(), 3);
    assert!(texts.iter().all(|t| !t.contains("x-anthropic-")), "{texts:?}");
    assert!(body.get("system_headers").is_none(), "ext must not leak into the body");
    assert!(out
        .degradations
        .iter()
        .any(|d| d.field == "ext.anthropic.system_headers.billing-header"));
}

#[test]
fn only_listed_names_are_forwarded() {
    let req = decode(&with_system(json!([
        {"type": "text", "text": BILLING},
        {"type": "text", "text": "x-anthropic-other: v"},
        {"type": "text", "text": "prompt"},
    ])));
    let (body, out) = encode(&req, &claude_5());
    assert_eq!(system_texts(&body), vec![BILLING.to_string(), "prompt".to_string()]);
    assert!(out.degradations.iter().any(|d| d.field == "ext.anthropic.system_headers.other"));
}

#[test]
fn forwarded_header_keeps_its_cache_control() {
    let req = decode(&with_system(json!([
        {"type": "text", "text": BILLING, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "prompt"},
    ])));
    let (body, _) = encode(&req, &claude_5());
    assert_eq!(body["system"][0]["cache_control"], json!({"type": "ephemeral"}));
}
