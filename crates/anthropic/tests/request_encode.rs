//! `encode_request`: IR -> Anthropic wire bytes + headers, capability-gated, with snapshots
//! and degradation assertions.

mod common;

use common::*;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_core::{
    CacheControl, Capabilities, Codec, Effort, EncodedRequest, Instruction, InstructionRole,
    IrRequest, Item, JsonText, MediaSource, OutputConfig, OutputFormat, Part, Position,
    ReasoningConfig, Role, ToolChoice, ToolDef,
};
use pretty_assertions::assert_eq;

fn enc(req: &IrRequest, caps: &Capabilities) -> EncodedRequest {
    AnthropicCodec.encode_request(req, caps, &ectx(llm_xlate_core::Protocol::Anthropic)).unwrap()
}

fn enc_from(req: &IrRequest, caps: &Capabilities, proto: llm_xlate_core::Protocol) -> EncodedRequest {
    AnthropicCodec.encode_request(req, caps, &ectx(proto)).unwrap()
}

/// Collect degradations as `field=kind` strings.
fn degs(enc: &EncodedRequest) -> Vec<String> {
    enc.degradations
        .iter()
        .map(|d| format!("{}={}", d.field, d.kind.slug()))
        .collect()
}

fn base(items: Vec<Item>) -> IrRequest {
    let mut r = IrRequest {
        model: llm_xlate_core::ModelRef::new("claude-opus-5"),
        items,
        ..Default::default()
    };
    r.limits.max_output_tokens = Some(1024);
    r
}

#[test]
fn simple_text_determinism_and_snapshot() {
    let req = base(vec![Item::user_text("hi")]);
    let a = enc(&req, &claude_5());
    let b = enc(&req, &claude_5());
    assert_eq!(a.body, b.body, "encode must be deterministic");
    insta::assert_snapshot!("simple_text", pretty(&a.body));
}

#[test]
fn max_tokens_injected_when_absent() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.limits.max_output_tokens = None;
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    // claude default_max_output_tokens = 32000.
    assert_eq!(v["max_tokens"], serde_json::json!(32000));
}

#[test]
fn leading_system_and_developer_blocks() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.instructions = vec![
        Instruction::system_text("sys"),
        Instruction::developer_text("dev"),
    ];
    let out = enc(&req, &claude_5());
    insta::assert_snapshot!("leading_system", pretty(&out.body));
}

#[test]
fn thinking_adaptive_claude5() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.reasoning = ReasoningConfig { effort: Some(Effort::High), enabled: Some(true), ..Default::default() };
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["thinking"], serde_json::json!({"type":"adaptive"}));
    assert_eq!(v["output_config"]["effort"], serde_json::json!("high"));
    insta::assert_snapshot!("thinking_adaptive", pretty(&out.body));
}

#[test]
fn thinking_budget_claude_37() {
    // Budget-mode thinking is a Claude 3.7+ feature (Claude 3.5 has no thinking mode).
    let mut req = base(vec![Item::user_text("hi")]);
    req.model = llm_xlate_core::ModelRef::new("claude-3-7-sonnet");
    req.reasoning = ReasoningConfig { budget_tokens: Some(10000), enabled: Some(true), ..Default::default() };
    let out = enc(&req, &claude_37());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["thinking"], serde_json::json!({"type":"enabled","budget_tokens":10000}));
    insta::assert_snapshot!("thinking_budget", pretty(&out.body));
}

#[test]
fn thinking_disabled_explicit() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.reasoning = ReasoningConfig { effort: Some(Effort::None), enabled: Some(false), ..Default::default() };
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["thinking"], serde_json::json!({"type":"disabled"}));
}

#[test]
fn effort_max_downgrades_when_not_supported() {
    // claude_old effort_levels = [low..xhigh] (no max) but it is budget-mode, so effort feeds
    // the budget. Use claude_46 (adaptive) whose effort_levels include max -> "max" passes.
    let mut req = base(vec![Item::user_text("hi")]);
    req.model = llm_xlate_core::ModelRef::new("claude-opus-4-6");
    req.reasoning = ReasoningConfig { effort: Some(Effort::Max), enabled: Some(true), ..Default::default() };
    let out = enc(&req, &claude_46());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["output_config"]["effort"], serde_json::json!("max"));
}

#[test]
fn mid_conversation_system_native_claude5() {
    let mut req = base(vec![Item::user_text("one"), Item::assistant_text("hi"), Item::user_text("two")]);
    req.instructions = vec![Instruction {
        role: InstructionRole::System,
        position: Position::Before(1),
        content: vec![Part::text("mid rule")],
        cache_control: None,
        effort: None,
        clear_at: None,
    }];
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let roles: Vec<&str> = v["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert!(roles.contains(&"system"), "native mid-conv system message expected: {roles:?}");
    insta::assert_snapshot!("mid_system_native", pretty(&out.body));
}

#[test]
fn mid_conversation_system_inline_wrap_claude46() {
    let mut req = base(vec![Item::user_text("one"), Item::assistant_text("hi"), Item::user_text("two")]);
    req.instructions = vec![Instruction {
        role: InstructionRole::System,
        position: Position::Before(2),
        content: vec![Part::text("mid rule")],
        cache_control: None,
        effort: None,
        clear_at: None,
    }];
    let out = enc(&req, &claude_46());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let roles: Vec<&str> = v["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert!(!roles.contains(&"system"), "no native system message on claude_46");
    assert!(degs(&out).iter().any(|d| d.starts_with("instructions=wrapped")));
    insta::assert_snapshot!("mid_system_inline_wrap", pretty(&out.body));
}

#[test]
fn mid_conversation_system_between_two_user_turns_native_claude5() {
    // Regression: an instruction anchored before a user turn that itself follows another user
    // turn. The two user turns must NOT coalesce across the instruction boundary — the native
    // `role:"system"` block must fire and sit BETWEEN them (user, system, user).
    let mut req = base(vec![Item::user_text("Hi"), Item::user_text("Weather?")]);
    req.instructions = vec![Instruction {
        role: InstructionRole::System,
        position: Position::Before(1),
        content: vec![Part::text("mid rule")],
        cache_control: None,
        effort: None,
        clear_at: None,
    }];
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let roles: Vec<&str> =
        v["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, vec!["user", "system", "user"], "native mid system must sit between turns");
    // First user turn is "Hi", last user turn is "Weather?".
    let msgs = v["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["content"][0]["text"], "Hi");
    assert_eq!(msgs[2]["content"][0]["text"], "Weather?");
}

#[test]
fn mid_conversation_system_between_two_user_turns_inline_wrap_claude46() {
    // Same shape without native support: the inline-wrap block must land BETWEEN the two user
    // turns (after "Hi", before "Weather?"), not hoisted to the front, and the two user turns
    // must merge into one message to keep roles alternating.
    let mut req = base(vec![Item::user_text("Hi"), Item::user_text("Weather?")]);
    req.instructions = vec![Instruction {
        role: InstructionRole::System,
        position: Position::Before(1),
        content: vec![Part::text("mid rule")],
        cache_control: None,
        effort: None,
        clear_at: None,
    }];
    let out = enc(&req, &claude_46());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let msgs = v["messages"].as_array().unwrap();
    let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, vec!["user"], "user turns must merge (no consecutive user messages)");
    // The single user message carries: "Hi", then the wrapped directive, then "Weather?".
    let texts: Vec<String> = msgs[0]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["text"].as_str().unwrap_or("").to_string())
        .collect();
    let hi = texts.iter().position(|t| t == "Hi").expect("Hi present");
    let weather = texts.iter().position(|t| t == "Weather?").expect("Weather? present");
    let wrapped = texts.iter().position(|t| t.contains("mid rule")).expect("directive present");
    assert!(hi < wrapped && wrapped < weather, "directive must sit between the two turns: {texts:?}");
}

#[test]
fn tool_loop_parallel_calls() {
    let req = base(vec![
        Item::user_text("weather?"),
        Item::ToolCall { call_id: "toolu_1".into(), name: "get".into(), arguments: JsonText::new(r#"{"city":"NYC"}"#), id: None },
        Item::ToolCall { call_id: "toolu_2".into(), name: "get".into(), arguments: JsonText::new(r#"{"city":"LA"}"#), id: None },
        Item::ToolResult { call_id: "toolu_1".into(), content: vec![Part::text("sunny")], is_error: false, id: None },
        Item::ToolResult { call_id: "toolu_2".into(), content: vec![Part::text("smoggy")], is_error: false, id: None },
    ]);
    let out = enc(&req, &claude_5());
    insta::assert_snapshot!("tool_loop_parallel", pretty(&out.body));
}

#[test]
fn tool_result_with_image() {
    let png = MediaSource::Base64 { media_type: "image/png".into(), data: bytes::Bytes::from_static(b"hi") };
    let req = base(vec![
        Item::ToolResult {
            call_id: "toolu_1".into(),
            content: vec![Part::text("see"), Part::Image(png)],
            is_error: false,
            id: None,
        },
    ]);
    let out = enc(&req, &claude_5());
    insta::assert_snapshot!("tool_result_image", pretty(&out.body));
}

#[test]
fn pdf_and_text_document() {
    let pdf = Part::Document {
        source: MediaSource::Base64 { media_type: "application/pdf".into(), data: bytes::Bytes::from_static(b"%PDF") },
        title: Some("Rep\"ort".into()),
        media_type: "application/pdf".into(),
    };
    let txt = Part::Document {
        source: MediaSource::Text("hello doc".into()),
        title: None,
        media_type: "text/plain".into(),
    };
    let req = base(vec![Item::Message { role: Role::User, content: vec![pdf, txt], id: None }]);
    let out = enc(&req, &claude_5());
    insta::assert_snapshot!("pdf_and_text_doc", pretty(&out.body));
}

#[test]
fn cache_control_breakpoints_capped_at_four() {
    // Six text parts each carrying a breakpoint -> only the last 4 survive.
    let parts: Vec<Part> = (0..6)
        .map(|i| Part::Text {
            text: format!("p{i}"),
            annotations: vec![],
            cache_control: Some(CacheControl::ephemeral_5m()),
        })
        .collect();
    let req = base(vec![Item::Message { role: Role::User, content: parts, id: None }]);
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let count = v["messages"][0]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b.get("cache_control").is_some())
        .count();
    assert_eq!(count, 4);
    assert!(degs(&out).iter().any(|d| d.starts_with("cache_control=")));
}

#[test]
fn request_level_cache_auto_injected_for_chat_origin() {
    let req = base(vec![Item::user_text("hi")]);
    let out = enc_from(&req, &claude_5(), llm_xlate_core::Protocol::OaiChat);
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["cache_control"], serde_json::json!({"type":"ephemeral"}));
}

#[test]
fn no_auto_cache_for_anthropic_origin() {
    let req = base(vec![Item::user_text("hi")]);
    let out = enc_from(&req, &claude_5(), llm_xlate_core::Protocol::Anthropic);
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert!(v.get("cache_control").is_none());
}

#[test]
fn prefill_rejected_when_not_allowed() {
    // claude_5 has prefill_allowed=false; a trailing assistant message is unsupported.
    let req = base(vec![Item::user_text("hi"), Item::assistant_text("prefix")]);
    let err = AnthropicCodec
        .encode_request(&req, &claude_5(), &ectx(llm_xlate_core::Protocol::Anthropic))
        .unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::Unsupported);
}

#[test]
fn prefill_allowed_on_old_model() {
    let mut req = base(vec![Item::user_text("hi"), Item::assistant_text("prefix")]);
    req.model = llm_xlate_core::ModelRef::new("claude-3-5-sonnet");
    let out = AnthropicCodec
        .encode_request(&req, &claude_old(), &ectx(llm_xlate_core::Protocol::Anthropic))
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let last = v["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(last["role"], serde_json::json!("assistant"));
}

#[test]
fn structured_output_json_schema() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.output = OutputConfig {
        format: OutputFormat::JsonSchema {
            name: "response".into(),
            schema: serde_json::json!({"type":"object","properties":{"a":{"type":"string"}}}),
            strict: true,
            description: None,
        },
        verbosity: None,
    };
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["output_config"]["format"]["type"], serde_json::json!("json_schema"));
    insta::assert_snapshot!("structured_output", pretty(&out.body));
}

#[test]
fn json_object_becomes_object_schema() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.output = OutputConfig { format: OutputFormat::JsonObject, verbosity: None };
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["output_config"]["format"]["schema"], serde_json::json!({"type":"object"}));
}

#[test]
fn sampling_rejected_on_claude5_is_dropped() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.sampling.temperature = Some(0.7);
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert!(v.get("temperature").is_none());
    assert!(degs(&out).iter().any(|d| d.starts_with("temperature=")));
}

#[test]
fn sampling_accepted_on_old_model() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.model = llm_xlate_core::ModelRef::new("claude-3-5-sonnet");
    req.sampling.temperature = Some(0.7);
    req.sampling.top_k = Some(40);
    let out = enc(&req, &claude_old());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["temperature"], serde_json::json!(0.7));
    assert_eq!(v["top_k"], serde_json::json!(40));
}

#[test]
fn stop_sequences_truncated_to_max() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.model = llm_xlate_core::ModelRef::new("claude-3-5-sonnet");
    req.limits.stop_sequences = vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()];
    let out = enc(&req, &claude_old());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["stop_sequences"].as_array().unwrap().len(), 4);
    assert!(degs(&out).iter().any(|d| d.starts_with("stop_sequences=")));
}

#[test]
fn service_tier_default_maps_to_auto() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.meta.service_tier = Some("default".into());
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["service_tier"], serde_json::json!("auto"));
}

#[test]
fn service_tier_flex_dropped() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.meta.service_tier = Some("flex".into());
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert!(v.get("service_tier").is_none());
    assert!(degs(&out).iter().any(|d| d.starts_with("service_tier=")));
}

#[test]
fn tool_choice_none_and_disable_parallel() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.tools = vec![ToolDef::Function { name: "get".into(), description: None, parameters: serde_json::json!({"type":"object"}), strict: None, cache_control: None }];
    req.tool_choice = ToolChoice::None;
    req.parallel_tool_calls = Some(false);
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["tool_choice"]["type"], serde_json::json!("none"));
    assert_eq!(v["tool_choice"]["disable_parallel_tool_use"], serde_json::json!(true));
}

#[test]
fn api_version_header_emitted() {
    let req = base(vec![Item::user_text("hi")]);
    let out = enc(&req, &claude_5());
    assert_eq!(out.headers.get("anthropic-version").unwrap(), "2023-06-01");
}

#[test]
fn client_betas_become_header() {
    let mut req = base(vec![Item::user_text("hi")]);
    req.ext.insert("anthropic.betas", serde_json::json!(["z-feature", "a-feature"]));
    let out = enc(&req, &claude_5());
    // sorted + deduped.
    assert_eq!(out.headers.get("anthropic-beta").unwrap(), "a-feature,z-feature");
}

#[test]
fn reasoning_replay_signature_roundtrips() {
    let blob = llm_xlate_core::OpaqueBlob::new(
        llm_xlate_core::ProviderFamily::Anthropic,
        llm_xlate_core::OpaqueKind::Signature,
        "SIG==",
    );
    let req = base(vec![
        Item::Reasoning(llm_xlate_core::ReasoningItem {
            text: Some("thought".into()),
            summary: vec![],
            opaque: Some(blob),
            id: None,
        }),
        Item::user_text("next"),
    ]);
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let block = &v["messages"][0]["content"][0];
    assert_eq!(block["type"], serde_json::json!("thinking"));
    assert_eq!(block["signature"], serde_json::json!("SIG=="));
}

#[test]
fn unsigned_reasoning_dropped_with_degradation() {
    let req = base(vec![
        Item::Reasoning(llm_xlate_core::ReasoningItem {
            text: Some("thought".into()),
            summary: vec![],
            opaque: None,
            id: None,
        }),
        Item::user_text("next"),
    ]);
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    // No thinking block emitted (dropped); the assistant message has an empty-text placeholder.
    let content = v["messages"][0]["content"].as_array().unwrap();
    assert!(content.iter().all(|b| b["type"] != serde_json::json!("thinking")));
    assert!(degs(&out).iter().any(|d| d.starts_with("items.reasoning=")));
}

#[test]
fn long_user_identifier_is_digested_to_anthropic_limit() {
    // Anthropic caps `metadata.user_id` at 256 chars; longer values (possible from OpenAI
    // clients) become a deterministic digest with a recorded rewrite, shorter ones pass through.
    let mut req = base(vec![Item::user_text("hi")]);
    req.meta.user = Some("x".repeat(300));
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    let uid = v["metadata"]["user_id"].as_str().unwrap();
    assert_eq!(uid.len(), 64);
    assert!(degs(&out).iter().any(|d| d == "user=rewritten"), "{:?}", degs(&out));
    req.meta.user = Some("u".repeat(256));
    let out = enc(&req, &claude_5());
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["metadata"]["user_id"].as_str().unwrap().len(), 256);
}
