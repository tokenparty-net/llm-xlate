//! `decode_request`: Anthropic wire request -> IR, field by field.

mod common;

use common::*;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_core::{
    CacheTtl, Codec, Effort, InstructionRole, Item, MediaSource, OutputFormat, Part, Position,
    ProviderFamily, ReasoningExposure, Role, ToolChoice, ToolDef,
};
use pretty_assertions::assert_eq;

fn decode(body: &str) -> llm_xlate_core::IrRequest {
    AnthropicCodec.decode_request(body.as_bytes(), &no_headers(), &dctx()).unwrap()
}

#[test]
fn missing_max_tokens_is_invalid() {
    let err = AnthropicCodec
        .decode_request(br#"{"model":"claude-opus-5","messages":[]}"#, &no_headers(), &dctx())
        .unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::InvalidRequest);
}

#[test]
fn malformed_json_is_invalid() {
    let err = AnthropicCodec
        .decode_request(b"{not json", &no_headers(), &dctx())
        .unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::InvalidRequest);
}

#[test]
fn plain_string_message() {
    let req = decode(r#"{"model":"claude-opus-5","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(req.model.upstream(), "claude-opus-5");
    assert_eq!(req.limits.max_output_tokens, Some(1024));
    assert_eq!(req.items, vec![Item::user_text("hi")]);
    // Anthropic clients always see thinking text.
    assert_eq!(req.reasoning.expose, ReasoningExposure::Full);
}

#[test]
fn top_level_system_string() {
    let req = decode(r#"{"model":"claude-opus-5","max_tokens":8,"system":"be nice","messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(req.instructions.len(), 1);
    let instr = &req.instructions[0];
    assert_eq!(instr.role, InstructionRole::System);
    assert_eq!(instr.position, Position::Leading);
    assert_eq!(instr.text(), "be nice");
}

#[test]
fn top_level_system_blocks_with_cache_control() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,
        "system":[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{"type":"ephemeral","ttl":"1h"}}],
        "messages":[{"role":"user","content":"hi"}]}"#,
    );
    let instr = &req.instructions[0];
    assert_eq!(instr.content.len(), 2);
    match &instr.content[1] {
        Part::Text { text, cache_control, .. } => {
            assert_eq!(text, "b");
            assert_eq!(cache_control.as_ref().map(|c| c.ttl), Some(CacheTtl::OneHour));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn in_array_system_becomes_before_instruction() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"messages":[
        {"role":"user","content":"one"},
        {"role":"system","content":"mid rule"},
        {"role":"user","content":"two"}]}"#,
    );
    // One leading? No — this system is in-array, so it is a Before(1) instruction.
    assert_eq!(req.instructions.len(), 1);
    assert_eq!(req.instructions[0].position, Position::Before(1));
    assert_eq!(req.instructions[0].text(), "mid rule");
    assert_eq!(req.items.len(), 2);
}

#[test]
fn multimodal_blocks_coalesce_into_one_message() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"messages":[{"role":"user","content":[
        {"type":"text","text":"look"},
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGk="}},
        {"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}"#,
    );
    assert_eq!(req.items.len(), 1);
    match &req.items[0] {
        Item::Message { content, role, .. } => {
            assert_eq!(*role, Role::User);
            assert_eq!(content.len(), 3);
            assert!(matches!(content[1], Part::Image(MediaSource::Base64 { .. })));
            assert!(matches!(content[2], Part::Image(MediaSource::Url(_))));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn tool_use_and_result_split_items() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"messages":[
        {"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"toolu_1","name":"get","input":{"q":"x"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"done"}]}]}"#,
    );
    assert_eq!(req.items.len(), 3);
    assert!(matches!(req.items[0], Item::Message { .. }));
    match &req.items[1] {
        Item::ToolCall { call_id, name, arguments, .. } => {
            assert_eq!(call_id.as_str(), "toolu_1");
            assert_eq!(name, "get");
            assert_eq!(arguments.as_str(), r#"{"q":"x"}"#);
        }
        other => panic!("{other:?}"),
    }
    match &req.items[2] {
        Item::ToolResult { call_id, content, is_error, .. } => {
            assert_eq!(call_id.as_str(), "toolu_1");
            assert!(!is_error);
            assert_eq!(content, &vec![Part::text("done")]);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn tool_result_error_flag() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"messages":[
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"boom","is_error":true}]}]}"#,
    );
    match &req.items[0] {
        Item::ToolResult { is_error, .. } => assert!(is_error),
        other => panic!("{other:?}"),
    }
}

#[test]
fn thinking_block_becomes_reasoning_with_signature() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"messages":[
        {"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"SIG=="}]}]}"#,
    );
    match &req.items[0] {
        Item::Reasoning(r) => {
            assert_eq!(r.text.as_deref(), Some("hmm"));
            let o = r.opaque.as_ref().unwrap();
            assert_eq!(o.family, ProviderFamily::Anthropic);
            assert_eq!(o.data, "SIG==");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn redacted_thinking_has_no_text() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"messages":[
        {"role":"assistant","content":[{"type":"redacted_thinking","data":"XXXX"}]}]}"#,
    );
    match &req.items[0] {
        Item::Reasoning(r) => {
            assert!(r.text.is_none());
            assert_eq!(r.opaque.as_ref().unwrap().kind, llm_xlate_core::OpaqueKind::Redacted);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn server_tool_use_and_result_are_provider_items() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"messages":[
        {"role":"assistant","content":[
        {"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"x"}},
        {"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"url":"u"}]}]}]}"#,
    );
    assert!(matches!(req.items[0], Item::ProviderToolCall(_)));
    assert!(matches!(req.items[1], Item::ProviderToolResult(_)));
}

#[test]
fn thinking_adaptive_reads_output_config_effort() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"thinking":{"type":"adaptive"},
        "output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(req.reasoning.enabled, Some(true));
    assert_eq!(req.reasoning.effort, Some(Effort::High));
}

#[test]
fn thinking_enabled_budget() {
    let req = decode(
        r#"{"model":"claude-3-5-sonnet","max_tokens":8,"thinking":{"type":"enabled","budget_tokens":10000},
        "messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(req.reasoning.budget_tokens, Some(10000));
    assert_eq!(req.reasoning.effort, Some(Effort::from_budget_tokens(10000)));
    assert_eq!(req.reasoning.enabled, Some(true));
}

#[test]
fn thinking_disabled() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"thinking":{"type":"disabled"},
        "messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(req.reasoning.effort, Some(Effort::None));
    assert_eq!(req.reasoning.enabled, Some(false));
}

#[test]
fn output_config_json_schema() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,
        "output_config":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
        "messages":[{"role":"user","content":"hi"}]}"#,
    );
    match &req.output.format {
        OutputFormat::JsonSchema { name, strict, .. } => {
            assert_eq!(name, "response");
            assert!(strict);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn tool_choice_variants() {
    let cases = [
        (r#"{"type":"auto"}"#, ToolChoice::Auto),
        (r#"{"type":"any"}"#, ToolChoice::Required),
        (r#"{"type":"none"}"#, ToolChoice::None),
        (r#"{"type":"tool","name":"get"}"#, ToolChoice::Named("get".into())),
    ];
    for (tc, expected) in cases {
        let body = format!(
            r#"{{"model":"claude-opus-5","max_tokens":8,"tool_choice":{tc},"messages":[{{"role":"user","content":"hi"}}]}}"#
        );
        let req = decode(&body);
        assert_eq!(req.tool_choice, expected);
    }
}

#[test]
fn disable_parallel_tool_use_maps_to_parallel_false() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"tool_choice":{"type":"auto","disable_parallel_tool_use":true},
        "messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(req.parallel_tool_calls, Some(false));
}

#[test]
fn function_tool_decode() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"tools":[
        {"name":"get","description":"d","input_schema":{"type":"object"},"strict":true}],
        "messages":[{"role":"user","content":"hi"}]}"#,
    );
    match &req.tools[0] {
        ToolDef::Function { name, description, strict, .. } => {
            assert_eq!(name, "get");
            assert_eq!(description.as_deref(), Some("d"));
            assert_eq!(*strict, Some(true));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn provider_tool_decode() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"tools":[
        {"type":"web_search_20260318","name":"web_search"}],
        "messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert!(matches!(req.tools[0], ToolDef::Provider(_)));
}

#[test]
fn sampling_and_metadata_and_service_tier() {
    let req = decode(
        r#"{"model":"claude-3-5-sonnet","max_tokens":8,"temperature":0.5,"top_p":0.9,"top_k":40,
        "stop_sequences":["STOP"],"metadata":{"user_id":"u1"},"service_tier":"standard_only",
        "cache_control":{"type":"ephemeral"},"messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(req.sampling.temperature, Some(0.5));
    assert_eq!(req.sampling.top_p, Some(0.9));
    assert_eq!(req.sampling.top_k, Some(40));
    assert_eq!(req.limits.stop_sequences, vec!["STOP".to_string()]);
    assert_eq!(req.meta.user.as_deref(), Some("u1"));
    assert_eq!(req.meta.service_tier.as_deref(), Some("standard_only"));
    assert!(req.cache.request_level);
}

#[test]
fn unknown_top_level_field_goes_to_ext() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"future_field":{"x":1},"messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(req.ext.get("anthropic.future_field"), Some(&serde_json::json!({"x":1})));
}

#[test]
fn stream_flag_and_container_passthrough() {
    let req = decode(
        r#"{"model":"claude-opus-5","max_tokens":8,"stream":true,"container":"c_1","messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert!(req.stream);
    assert_eq!(req.ext.get("anthropic.container"), Some(&serde_json::json!("c_1")));
}

#[test]
fn anthropic_beta_header_captured() {
    let req = AnthropicCodec
        .decode_request(
            br#"{"model":"claude-opus-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
            &beta_headers("feature-a,feature-b"),
            &dctx(),
        )
        .unwrap();
    assert_eq!(
        req.ext.get("anthropic.betas"),
        Some(&serde_json::json!(["feature-a", "feature-b"]))
    );
}

#[test]
fn opened_envelope_reasoning_carries_foreign_family() {
    // A foreign (OpenAI) reasoning blob sealed into an rtr1. envelope, replayed via a thinking
    // block's signature slot, must open to its original family.
    let blob = llm_xlate_core::OpaqueBlob::new(
        ProviderFamily::OpenAI,
        llm_xlate_core::OpaqueKind::Encrypted,
        "cipher-xyz",
    );
    let env = sealer().seal(&blob);
    let body = format!(
        r#"{{"model":"claude-opus-5","max_tokens":8,"messages":[
        {{"role":"assistant","content":[{{"type":"thinking","thinking":"t","signature":"{env}"}}]}}]}}"#
    );
    let req = decode(&body);
    match &req.items[0] {
        Item::Reasoning(r) => {
            let o = r.opaque.as_ref().unwrap();
            assert_eq!(o.family, ProviderFamily::OpenAI);
            assert_eq!(o.data, "cipher-xyz");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn redacted_thinking_router_envelope_is_opened_to_its_true_family() {
    // Regression (Claude Code -> gpt-5.6-luna via the test router, trace tr_29a3110ec1ae4c33a4d2feaa3ed2a958):
    // the client-facing Anthropic encoder carries OpenAI encrypted reasoning to an Anthropic client as a
    // `redacted_thinking` block whose `data` is a router envelope. On replay the decoder tagged it as a
    // native Anthropic redacted blob, so lower() dropped it as foreign for the OpenAI target and every
    // multi-turn GPT conversation lost its reasoning continuity. The envelope must be opened.
    use llm_xlate_core::{OpaqueBlob, OpaqueKind, ProviderFamily};
    let blob = OpaqueBlob {
        family: ProviderFamily::OpenAI,
        kind: OpaqueKind::Encrypted,
        data: "gAAAAABqougHylct4S8294geEYhA_JKscd7U".to_string(),
        model: Some("gpt-5.6-luna".to_string()),
    };
    let env = sealer().seal(&blob);
    assert!(env.starts_with("rtr1."));
    let body = format!(
        r#"{{"model":"gpt-5.6-luna","max_tokens":64,"messages":[
            {{"role":"user","content":"hi"}},
            {{"role":"assistant","content":[{{"type":"redacted_thinking","data":"{env}"}},{{"type":"text","text":"ok"}}]}},
            {{"role":"user","content":"again"}}]}}"#
    );
    let ir = decode(&body);
    let reasoning = ir
        .items
        .iter()
        .find_map(|it| match it {
            llm_xlate_core::Item::Reasoning(r) => Some(r),
            _ => None,
        })
        .expect("reasoning item");
    let opened = reasoning.opaque.as_ref().expect("opaque");
    assert_eq!(opened.family, ProviderFamily::OpenAI, "envelope family must survive");
    assert_eq!(opened.kind, OpaqueKind::Encrypted);
    assert_eq!(opened.data, blob.data);
    assert_eq!(opened.model.as_deref(), Some("gpt-5.6-luna"));
}
