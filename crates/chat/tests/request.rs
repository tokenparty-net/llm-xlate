#![allow(clippy::result_large_err, clippy::field_reassign_with_default)]
//! Request decode / encode: goldens, round-trip laws, determinism, capability matrix.

mod common;

use common::*;
use llm_xlate_core::degrade::DegradationKind;
use llm_xlate_core::ir::{
    Effort, Instruction, InstructionRole, IrRequest, Item, MediaSource, OutputFormat, Part,
    Position, ProviderFamily, ReasoningExposure, Role, ToolChoice, ToolDef,
};
use pretty_assertions::assert_eq;
use serde_json::json;

const REQ_TEXT: &str = include_str!("fixtures/req_text.json");
const REQ_MULTITURN: &str = include_str!("fixtures/req_multiturn_sysdev.json");
const REQ_PARALLEL: &str = include_str!("fixtures/req_parallel_tools.json");
const REQ_LEGACY: &str = include_str!("fixtures/req_legacy_functions.json");
const REQ_JSON_SCHEMA: &str = include_str!("fixtures/req_json_schema.json");
const REQ_IMAGES: &str = include_str!("fixtures/req_images.json");
const REQ_PDF: &str = include_str!("fixtures/req_pdf.json");
const REQ_REASONING: &str = include_str!("fixtures/req_reasoning.json");

/// The fixtures for which decode → encode(full) → decode is the identity.
fn lossless_fixtures() -> Vec<(&'static str, &'static str)> {
    vec![
        ("text", REQ_TEXT),
        ("multiturn", REQ_MULTITURN),
        ("legacy_functions", REQ_LEGACY),
        ("json_schema", REQ_JSON_SCHEMA),
        ("images", REQ_IMAGES),
        ("pdf", REQ_PDF),
        ("reasoning", REQ_REASONING),
    ]
}

// ---------------------------------------------------------------- round-trip

#[test]
fn round_trip_identity_text() {
    round_trip_identity(REQ_TEXT);
}
#[test]
fn round_trip_identity_multiturn() {
    round_trip_identity(REQ_MULTITURN);
}
#[test]
fn round_trip_identity_legacy() {
    round_trip_identity(REQ_LEGACY);
}
#[test]
fn round_trip_identity_json_schema() {
    round_trip_identity(REQ_JSON_SCHEMA);
}
#[test]
fn round_trip_identity_images() {
    round_trip_identity(REQ_IMAGES);
}
#[test]
fn round_trip_identity_pdf() {
    round_trip_identity(REQ_PDF);
}
#[test]
fn round_trip_identity_reasoning() {
    round_trip_identity(REQ_REASONING);
}

fn round_trip_identity(body: &str) {
    let ir1 = decode_request(body);
    let bytes = encode_request(&ir1, &full_caps());
    let ir2 = decode_request(std::str::from_utf8(&bytes.body).unwrap());
    assert_eq!(ir1, ir2);
}

#[test]
fn all_lossless_fixtures_round_trip() {
    for (name, body) in lossless_fixtures() {
        let ir1 = decode_request(body);
        let bytes = encode_request(&ir1, &full_caps());
        let ir2 = decode_request(std::str::from_utf8(&bytes.body).unwrap());
        assert_eq!(ir1, ir2, "fixture {name} did not round-trip");
    }
}

// ---------------------------------------------------------------- determinism

#[test]
fn determinism_all_fixtures() {
    for (_, body) in lossless_fixtures() {
        let ir = decode_request(body);
        let a = encode_request(&ir, &full_caps());
        let b = encode_request(&ir, &full_caps());
        assert_eq!(a.body, b.body);
    }
}

#[test]
fn determinism_parallel_tools() {
    let ir = decode_request(REQ_PARALLEL);
    let a = encode_request(&ir, &gpt4o());
    let b = encode_request(&ir, &gpt4o());
    assert_eq!(a.body, b.body);
}

// ---------------------------------------------------------------- goldens (insta)

#[test]
fn golden_text() {
    let ir = decode_request(REQ_TEXT);
    insta::assert_snapshot!("golden_text", encode_request_str(&ir, &gpt4o()));
}

#[test]
fn golden_multiturn() {
    let ir = decode_request(REQ_MULTITURN);
    insta::assert_snapshot!("golden_multiturn", encode_request_str(&ir, &full_caps()));
}

#[test]
fn golden_parallel_tools_with_image_fold() {
    let ir = decode_request(REQ_PARALLEL);
    let enc = encode_request(&ir, &gpt4o());
    // Non-text tool-result parts fold into a following user message.
    assert!(enc.degradations.iter().any(|d| d.field == "tool_result"
        && d.kind == DegradationKind::Folded));
    insta::assert_snapshot!(
        "golden_parallel_tools",
        String::from_utf8(enc.body.to_vec()).unwrap()
    );
}

#[test]
fn golden_legacy_functions() {
    let ir = decode_request(REQ_LEGACY);
    insta::assert_snapshot!("golden_legacy", encode_request_str(&ir, &gpt4o()));
}

#[test]
fn golden_json_schema_strict() {
    let ir = decode_request(REQ_JSON_SCHEMA);
    insta::assert_snapshot!("golden_json_schema", encode_request_str(&ir, &gpt4o()));
}

#[test]
fn golden_images() {
    let ir = decode_request(REQ_IMAGES);
    insta::assert_snapshot!("golden_images", encode_request_str(&ir, &gpt4o()));
}

#[test]
fn golden_pdf() {
    let ir = decode_request(REQ_PDF);
    insta::assert_snapshot!("golden_pdf", encode_request_str(&ir, &gpt4o()));
}

// ---------------------------------------------------------------- decode units

#[test]
fn decode_expose_is_full() {
    let ir = decode_request(REQ_TEXT);
    assert_eq!(ir.reasoning.expose, ReasoningExposure::Full);
}

#[test]
fn decode_leading_system_then_before() {
    let ir = decode_request(REQ_MULTITURN);
    assert_eq!(ir.instructions[0].position, Position::Leading);
    assert_eq!(ir.instructions[0].role, InstructionRole::System);
    assert_eq!(ir.instructions[1].position, Position::Before(2));
    assert_eq!(ir.instructions[1].role, InstructionRole::Developer);
}

#[test]
fn decode_unknown_top_level_field_to_ext() {
    let body = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"future_param":42}"#;
    let ir = decode_request(body);
    assert_eq!(ir.ext.get("chat.future_param"), Some(&json!(42)));
}

#[test]
fn decode_max_tokens_field_marker() {
    let a = decode_request(r#"{"model":"m","messages":[],"max_tokens":100}"#);
    assert_eq!(a.ext.get("chat.max_tokens_field"), Some(&json!("max_tokens")));
    assert_eq!(a.limits.max_output_tokens, Some(100));
    let b = decode_request(r#"{"model":"m","messages":[],"max_completion_tokens":200}"#);
    assert_eq!(b.ext.get("chat.max_tokens_field"), Some(&json!("max_completion_tokens")));
}

#[test]
fn decode_stop_string_and_array() {
    let a = decode_request(r#"{"model":"m","messages":[],"stop":"END"}"#);
    assert_eq!(a.limits.stop_sequences, vec!["END".to_string()]);
    let b = decode_request(r#"{"model":"m","messages":[],"stop":["A","B"]}"#);
    assert_eq!(b.limits.stop_sequences, vec!["A".to_string(), "B".to_string()]);
}

#[test]
fn decode_tool_choice_variants() {
    let none = decode_request(r#"{"model":"m","messages":[],"tool_choice":"none"}"#);
    assert_eq!(none.tool_choice, ToolChoice::None);
    let req = decode_request(r#"{"model":"m","messages":[],"tool_choice":"required"}"#);
    assert_eq!(req.tool_choice, ToolChoice::Required);
    let named = decode_request(
        r#"{"model":"m","messages":[],"tool_choice":{"type":"function","function":{"name":"f"}}}"#,
    );
    assert_eq!(named.tool_choice, ToolChoice::Named("f".to_string()));
}

#[test]
fn decode_meta_and_state_fields() {
    let body = r#"{"model":"m","messages":[],"user":"u1","safety_identifier":"s1",
        "metadata":{"k":"v"},"service_tier":"auto","store":true,"prompt_cache_key":"ck"}"#;
    let ir = decode_request(body);
    assert_eq!(ir.meta.user.as_deref(), Some("u1"));
    assert_eq!(ir.meta.safety_identifier.as_deref(), Some("s1"));
    assert_eq!(ir.meta.metadata.get("k"), Some(&json!("v")));
    assert_eq!(ir.meta.service_tier.as_deref(), Some("auto"));
    assert_eq!(ir.state.store, Some(true));
    assert_eq!(ir.cache.prompt_cache_key.as_deref(), Some("ck"));
}

#[test]
fn decode_input_audio_part() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"data":"aGVsbG8=","format":"mp3"}}]}]}"#;
    let ir = decode_request(body);
    match &ir.items[0] {
        Item::Message { content, .. } => match &content[0] {
            Part::Audio(MediaSource::Base64 { media_type, .. }) => {
                assert_eq!(media_type, "audio/mp3");
            }
            other => panic!("expected audio base64, got {other:?}"),
        },
        other => panic!("expected message, got {other:?}"),
    }
}

#[test]
fn decode_assistant_refusal_field() {
    let body = r#"{"model":"m","messages":[{"role":"assistant","refusal":"nope"}]}"#;
    let ir = decode_request(body);
    match &ir.items[0] {
        Item::Message { role: Role::Assistant, content, .. } => {
            assert_eq!(content[0], Part::Refusal { text: "nope".to_string() });
        }
        other => panic!("expected assistant message, got {other:?}"),
    }
}

#[test]
fn decode_non_function_tool_is_provider() {
    let body = r#"{"model":"m","messages":[],"tools":[{"type":"web_search"}]}"#;
    let ir = decode_request(body);
    match &ir.tools[0] {
        ToolDef::Provider(item) => assert_eq!(item.family, ProviderFamily::OpenAI),
        other => panic!("expected provider tool, got {other:?}"),
    }
}

#[test]
fn decode_stream_options_include_usage() {
    let body = r#"{"model":"m","messages":[],"stream":true,"stream_options":{"include_usage":true}}"#;
    let ir = decode_request(body);
    assert!(ir.stream);
    assert_eq!(ir.ext.get("chat.include_usage"), Some(&json!(true)));
}

#[test]
fn decode_bad_body_is_invalid_request() {
    let err = try_decode_request("not json").unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::InvalidRequest);
}

#[test]
fn decode_bad_role_is_invalid_request() {
    let err = try_decode_request(r#"{"model":"m","messages":[{"role":"robot","content":"x"}]}"#)
        .unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::InvalidRequest);
}

// ---------------------------------------------------------------- encode units

#[test]
fn encode_gpt4o_drops_reasoning_text() {
    // gpt-4o has replay = none, so assistant reasoning_content is dropped + degraded.
    let ir = decode_request(REQ_REASONING);
    let enc = encode_request(&ir, &gpt4o());
    let body = String::from_utf8(enc.body.to_vec()).unwrap();
    assert!(!body.contains("reasoning_content"));
    assert!(enc.degradations.iter().any(|d| d.field == "reasoning.text"));
}

#[test]
fn encode_audio_unsupported_without_caps() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"data":"aGVsbG8=","format":"mp3"}}]}]}"#;
    let ir = decode_request(body);
    let err = codec_encode_err(&ir, &gpt4o());
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::Unsupported);
}

#[test]
fn encode_audio_supported_with_full_caps() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"data":"aGVsbG8=","format":"wav"}}]}]}"#;
    let ir = decode_request(body);
    let out = encode_request_str(&ir, &full_caps());
    assert!(out.contains("input_audio"));
}

/// Regression (testrouter session 2026-09-10, trace `tr_dc34ef14…`): Codex sends OpenAI
/// **Responses** hosted tools — `{"type":"namespace"}` groups and `{"type":"web_search"}` —
/// alongside plain functions. They decode to `ToolDef::Provider` with family `OpenAI`, which used
/// to be pushed into the Chat `tools` array **raw** because the family matched. Chat Completions
/// has no carrier for a hosted tool (every entry must be `{"type":"function","function":{…}}`), so
/// the backend rejected the whole request: vLLM 400 `Input should be 'function'` at
/// `body.tools.7.type`. The hosted entries must be dropped with a degradation instead.
///
/// The tool JSON below is the captured Codex request verbatim (sub-tool list trimmed to one).
#[test]
fn encode_drops_responses_hosted_tools_with_degradation() {
    let body = r#"{"model":"m","messages":[],"tools":[
        {"type":"function","function":{"name":"exec_command","parameters":{"type":"object"}}},
        {"type":"namespace","name":"multi_agent_v1","description":"Tools for spawning and managing sub-agents.","tools":[
            {"type":"function","name":"close_agent","description":"Close an agent.","strict":false,
             "parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"],"additionalProperties":false}}]},
        {"type":"web_search","external_web_access":false}]}"#;
    let ir = decode_request(body);
    assert_eq!(ir.tools.len(), 3, "two hosted tools decode as provider tools");

    let enc = encode_request(&ir, &gpt5_chat());
    let out: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    let tools = out["tools"].as_array().expect("tools array");

    // Only the plain function survives, and every emitted entry is a valid Chat tool.
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], json!("function"));
    assert_eq!(tools[0]["function"]["name"], json!("exec_command"));
    for t in tools {
        assert_eq!(t["type"], json!("function"), "no non-function tool may reach the wire");
        assert!(t.get("function").is_some(), "every Chat tool needs a `function` object");
    }

    // Both drops are reported, naming the tool so a trace shows what the model lost.
    let dropped: Vec<&str> = enc
        .degradations
        .iter()
        .filter(|d| d.field == "tools.provider" && d.kind == DegradationKind::Dropped)
        .map(|d| d.detail.as_str())
        .collect();
    assert_eq!(dropped.len(), 2, "got {dropped:?}");
    assert!(dropped.iter().any(|d| d.contains("namespace multi_agent_v1")), "{dropped:?}");
    assert!(dropped.iter().any(|d| d.contains("web_search")), "{dropped:?}");
}

/// When *every* declared tool is a hosted tool with no Chat carrier, `tools` is omitted entirely
/// rather than sent as `[]` (which some OpenAI-compatible backends reject).
#[test]
fn encode_all_hosted_tools_omits_tools_key() {
    let body = r#"{"model":"m","messages":[],"tools":[{"type":"web_search"}]}"#;
    let ir = decode_request(body);
    let enc = encode_request(&ir, &gpt5_chat());
    let out: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert!(out.get("tools").is_none(), "expected no `tools` key, got {out}");
    assert!(enc.degradations.iter().any(|d| d.field == "tools.provider"));
}

#[test]
fn encode_document_url_unsupported() {
    let mut ir = IrRequest::default();
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![Part::Document {
            source: MediaSource::Url("https://example.com/x.pdf".to_string()),
            title: None,
            media_type: "application/pdf".to_string(),
        }],
        id: None,
    });
    let err = codec_encode_err(&ir, &full_caps());
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::Unsupported);
}

#[test]
fn encode_document_base64_without_title_synthesizes_filename() {
    // A cross-family (e.g. Anthropic) PDF often has no title; OpenAI requires `filename` beside
    // `file_data` and 400s without it, so a default is synthesized from the media type.
    let mut ir = IrRequest::default();
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![Part::Document {
            source: MediaSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: bytes::Bytes::from_static(b"%PDF-1.7"),
            },
            title: None,
            media_type: "application/pdf".to_string(),
        }],
        id: None,
    });
    let out = encode_request_str(&ir, &full_caps());
    assert!(out.contains("\"filename\":\"document.pdf\""), "expected synthesized filename: {out}");
    assert!(out.contains("file_data"));
}

#[test]
fn encode_document_base64_keeps_supplied_title() {
    let mut ir = IrRequest::default();
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![Part::Document {
            source: MediaSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: bytes::Bytes::from_static(b"%PDF-1.7"),
            },
            title: Some("report.pdf".to_string()),
            media_type: "application/pdf".to_string(),
        }],
        id: None,
    });
    let out = encode_request_str(&ir, &full_caps());
    assert!(out.contains("\"filename\":\"report.pdf\""), "client title must be kept: {out}");
}

#[test]
fn encode_service_tier_mapping() {
    let mut ir = IrRequest::default();
    ir.meta.service_tier = Some("standard_only".to_string());
    let out = encode_request_str(&ir, &gpt4o());
    // standard_only → "default" (which gpt-4o supports).
    assert!(out.contains("\"service_tier\":\"default\""));
}

#[test]
fn encode_stream_include_usage() {
    let mut ir = IrRequest::default();
    ir.stream = true;
    let enc = encode_request(&ir, &gpt4o());
    assert!(enc.upstream_streams);
    assert!(enc.ctx.include_usage);
    let out = String::from_utf8(enc.body.to_vec()).unwrap();
    assert!(out.contains("\"stream\":true"));
    assert!(out.contains("\"include_usage\":true"));
}

#[test]
fn encode_ext_passthrough_written_back() {
    let mut ir = IrRequest::default();
    ir.ext.insert("chat.custom_flag".to_string(), json!("hi"));
    let out = encode_request_str(&ir, &gpt4o());
    assert!(out.contains("\"custom_flag\":\"hi\""));
}

#[test]
fn encode_foreign_ext_ignored() {
    let mut ir = IrRequest::default();
    ir.ext.insert("anthropic.thinking".to_string(), json!(true));
    let out = encode_request_str(&ir, &gpt4o());
    assert!(!out.contains("thinking"));
}

// ---------------------------------------------------------------- capability matrix

#[test]
fn matrix_gpt5_chat_drops_effort_with_tools() {
    // Tools present + reasoning effort → effort dropped (GPT-5.4 Chat: no tools_with_reasoning).
    let body = r#"{"model":"gpt-5.4","messages":[{"role":"user","content":"hi"}],
        "reasoning_effort":"medium",
        "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#;
    let ir = decode_request(body);
    let enc = encode_request(&ir, &gpt5_chat());
    let out = String::from_utf8(enc.body.to_vec()).unwrap();
    assert!(!out.contains("reasoning_effort"));
    assert!(enc.degradations.iter().any(|d| d.field == "reasoning_effort"));
}

#[test]
fn matrix_gpt5_chat_keeps_effort_without_tools() {
    let body = r#"{"model":"gpt-5.4","messages":[{"role":"user","content":"hi"}],
        "reasoning_effort":"medium"}"#;
    let ir = decode_request(body);
    let out = encode_request_str(&ir, &gpt5_chat());
    assert!(out.contains("\"reasoning_effort\":\"medium\""));
}

#[test]
fn matrix_openai_compatible_drops_strict_and_downgrades_developer() {
    let body = r#"{"model":"llama-x","messages":[
        {"role":"developer","content":"be terse"},
        {"role":"user","content":"hi"}],
        "tools":[{"type":"function","function":{"name":"f","parameters":{},"strict":true}}]}"#;
    let ir = decode_request(body);
    let enc = encode_request(&ir, &openai_compatible());
    let out = String::from_utf8(enc.body.to_vec()).unwrap();
    assert!(!out.contains("\"strict\""));
    assert!(!out.contains("\"developer\""));
    assert!(out.contains("\"role\":\"system\""));
    assert!(enc.degradations.iter().any(|d| d.field == "instructions.developer"));
    assert!(enc.degradations.iter().any(|d| d.field == "tools.strict"));
}

#[test]
fn matrix_openai_compatible_uses_max_tokens_field() {
    // Built directly (no chat.max_tokens_field marker) → openai-compatible uses `max_tokens`.
    let mut ir = IrRequest::default();
    ir.limits.max_output_tokens = Some(321);
    let out = encode_request_str(&ir, &openai_compatible());
    assert!(out.contains("\"max_tokens\":321"));
    assert!(!out.contains("max_completion_tokens"));
}

#[test]
fn matrix_openai_proper_uses_max_completion_tokens_field() {
    let mut ir = IrRequest::default();
    ir.limits.max_output_tokens = Some(321);
    let out = encode_request_str(&ir, &gpt4o());
    assert!(out.contains("\"max_completion_tokens\":321"));
}

#[test]
fn matrix_max_tokens_marker_wins() {
    // A Chat client that sent max_tokens keeps that spelling even against a proper OpenAI target.
    let ir = decode_request(r#"{"model":"m","messages":[],"max_tokens":50}"#);
    let out = encode_request_str(&ir, &gpt4o());
    assert!(out.contains("\"max_tokens\":50"));
    assert!(!out.contains("max_completion_tokens"));
}

#[test]
fn matrix_reasoning_effort_max_downgrades_to_xhigh() {
    let mut ir = IrRequest::default();
    ir.reasoning.effort = Some(Effort::Max);
    let enc = encode_request(&ir, &gpt5_chat());
    let out = String::from_utf8(enc.body.to_vec()).unwrap();
    assert!(out.contains("\"reasoning_effort\":\"xhigh\""));
    assert!(enc.degradations.iter().any(|d| d.field == "reasoning_effort"
        && d.kind == DegradationKind::Downgraded));
}

#[test]
fn matrix_developer_instruction_native_on_gpt4o() {
    let mut ir = IrRequest::default();
    ir.instructions.push(Instruction::developer_text("be terse"));
    let out = encode_request_str(&ir, &gpt4o());
    assert!(out.contains("\"role\":\"developer\""));
}

#[test]
fn matrix_output_format_json_object() {
    let mut ir = IrRequest::default();
    ir.output.format = OutputFormat::JsonObject;
    let out = encode_request_str(&ir, &gpt4o());
    assert!(out.contains("\"response_format\":{\"type\":\"json_object\"}"));
}

// helper: expect an encode error
fn codec_encode_err(ir: &IrRequest, caps: &llm_xlate_core::Capabilities) -> llm_xlate_core::XlateError {
    try_encode_request(ir, caps).expect_err("expected encode error")
}

#[test]
fn long_user_identifier_is_digested_to_openai_limit() {
    // Regression (Claude Code on gpt-5.6-luna, 2026-09-10): a ~150-char client user id must not
    // reach OpenAI verbatim (400 "string too long ... maximum length 64"). It becomes a
    // deterministic 64-char digest with a recorded rewrite.
    let mut ir = decode_request(REQ_REASONING);
    ir.meta.user = Some(format!("user_{}", "a".repeat(145)));
    let enc = encode_request(&ir, &gpt4o());
    let b: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    let user = b.get("user").and_then(|v| v.as_str()).unwrap();
    assert_eq!(user.len(), 64);
    assert!(user.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(enc.degradations.iter().any(|d| d.field == "user"));
    let enc2 = encode_request(&ir, &gpt4o());
    assert_eq!(enc.body, enc2.body, "digest must be deterministic");
}
