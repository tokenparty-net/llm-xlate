#![allow(clippy::field_reassign_with_default)]
//! `encode_request` tests: IR → Responses request body (capability-gated + degradations).

mod common;
use common::*;

use llm_xlate_core::caps::preset;
use llm_xlate_core::{
    CallId, Codec, DegradationKind, Effort, Instruction, InstructionRole, IrRequest, Item, JsonText,
    MediaSource, OpaqueBlob, OpaqueKind, OutputConfig, OutputFormat, Part, Position, ProviderFamily,
    ReasoningConfig, ReasoningExposure, Role, Sampling, StateConfig, SummaryLevel, ToolChoice,
    ToolDef, Verbosity,
};
use pretty_assertions::assert_eq;
use serde_json::Value;

fn body(req: &IrRequest, caps: &llm_xlate_core::Capabilities) -> Value {
    json(&encode(req, caps).body)
}

#[test]
fn single_leading_instruction_becomes_string() {
    let ir = decode(r#"{"model":"gpt-5.4","instructions":"Be nice.","input":"hi"}"#);
    let b = body(&ir, &caps());
    assert_eq!(b.get("instructions").unwrap(), "Be nice.");
    // input carries only the user message (no system item).
    let input = b.get("input").unwrap().as_array().unwrap();
    assert_eq!(input.len(), 1);
    assert_eq!(input[0].get("role").unwrap(), "user");
}

#[test]
fn single_leading_developer_instruction_becomes_message_item() {
    // A single leading *Developer* instruction must NOT collapse into the top-level `instructions`
    // string (which decodes back as System); it renders as a `developer` message item so its role
    // survives the round trip (plan §7.1).
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("gpt-5.4");
    ir.instructions.push(Instruction::developer_text("Be terse."));
    ir.items.push(Item::user_text("hi"));
    let b = body(&ir, &caps());
    assert!(b.get("instructions").is_none(), "developer instruction must not be the string form");
    let input = b.get("input").unwrap().as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[0].get("type").unwrap(), "message");
    assert_eq!(input[0].get("role").unwrap(), "developer");
    assert_eq!(input[1].get("role").unwrap(), "user");
    // And it round-trips back as a leading Developer instruction.
    let encoded = encode(&ir, &caps()).body;
    let back = decode(std::str::from_utf8(&encoded).unwrap());
    assert_eq!(back.instructions.len(), 1);
    assert_eq!(back.instructions[0].role, InstructionRole::Developer);
    assert_eq!(back.instructions[0].position, Position::Leading);
}

#[test]
fn two_leading_instructions_become_items() {
    let ir = decode(
        r#"{"model":"m","input":[{"role":"system","content":"a"},{"role":"developer","content":"b"},{"role":"user","content":"u"}]}"#,
    );
    let b = body(&ir, &caps());
    assert!(b.get("instructions").is_none());
    let input = b.get("input").unwrap().as_array().unwrap();
    assert_eq!(input.len(), 3);
    assert_eq!(input[0].get("role").unwrap(), "system");
    assert_eq!(input[1].get("role").unwrap(), "developer");
    assert_eq!(input[2].get("role").unwrap(), "user");
}

#[test]
fn before_zero_forces_items_not_string() {
    // A single leading instruction but also a Before(0) → cannot use the string form.
    let mut ir = IrRequest::default();
    ir.instructions.push(Instruction::system_text("lead"));
    ir.instructions.push(Instruction {
        role: InstructionRole::System,
        position: Position::Before(0),
        content: vec![Part::text("mid")],
        cache_control: None,
        effort: None,
        clear_at: None,
    });
    ir.items.push(Item::user_text("u"));
    let b = body(&ir, &caps());
    assert!(b.get("instructions").is_none());
    let input = b.get("input").unwrap().as_array().unwrap();
    assert_eq!(input.len(), 3);
}

#[test]
fn function_call_loop_with_parallel() {
    let mut ir = IrRequest::default();
    ir.parallel_tool_calls = Some(true);
    ir.items.push(Item::user_text("weather?"));
    ir.items.push(Item::ToolCall {
        call_id: CallId::new("c1"),
        name: "get".into(),
        arguments: JsonText::new("{\"a\":1}"),
        id: None,
    });
    ir.items.push(Item::ToolCall {
        call_id: CallId::new("c2"),
        name: "get".into(),
        arguments: JsonText::new("{\"a\":2}"),
        id: None,
    });
    ir.items.push(Item::ToolResult {
        call_id: CallId::new("c1"),
        content: vec![Part::text("A")],
        is_error: false,
        id: None,
    });
    let b = body(&ir, &caps());
    assert_eq!(b.get("parallel_tool_calls").unwrap(), true);
    let input = b.get("input").unwrap().as_array().unwrap();
    assert_eq!(input[1].get("type").unwrap(), "function_call");
    assert_eq!(input[1].get("arguments").unwrap(), "{\"a\":1}");
    assert_eq!(input[3].get("type").unwrap(), "function_call_output");
    assert_eq!(input[3].get("output").unwrap(), "A");
}

#[test]
fn function_call_output_multipart_uses_input_text_not_output_text() {
    // Regression (live-proven 2026-09-10, translate1 tools_loop chat->responses): a multi-part
    // tool result must encode its text parts as `input_text` — the Responses API rejects
    // `output_text` inside a function_call_output ("Invalid value: 'output_text'. Supported
    // values are: 'input_text'"). A single-string result stays a bare string; only the array form
    // regressed.
    let mut ir = IrRequest::default();
    ir.items.push(Item::user_text("weather?"));
    ir.items.push(Item::ToolCall {
        call_id: CallId::new("c1"),
        name: "get".into(),
        arguments: JsonText::new("{}"),
        id: None,
    });
    // A non-all-text result (text + image) forces the array form (an all-text result is joined to
    // a bare string). With Responses' text-only tool-result caps the image folds to a text
    // attachment, so both parts are text parts in the array — and both must be `input_text`.
    ir.items.push(Item::ToolResult {
        call_id: CallId::new("c1"),
        content: vec![
            Part::text("Paris: sunny"),
            Part::Image(MediaSource::Base64 { media_type: "image/png".into(), data: b"x".to_vec().into() }),
        ],
        is_error: false,
        id: None,
    });
    let b = body(&ir, &caps());
    let input = b.get("input").unwrap().as_array().unwrap();
    let out = input[2].get("output").unwrap().as_array().unwrap();
    assert!(out.len() >= 2);
    // The bug was `output_text` inside function_call_output; no part may use it (text parts must be
    // input_text; an image part is input_image).
    for part in out {
        assert_ne!(
            part.get("type").and_then(|t| t.as_str()),
            Some("output_text"),
            "function_call_output must not use output_text, got {part:?}"
        );
    }
    assert!(
        out.iter().any(|p| p.get("type").and_then(|t| t.as_str()) == Some("input_text")),
        "expected at least one input_text part"
    );
}

#[test]
fn reasoning_openai_opaque_replayed_with_encrypted_content() {
    let mut ir = IrRequest::default();
    ir.state.store = Some(false);
    let blob = OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Encrypted, "ENC");
    ir.items.push(Item::Reasoning(llm_xlate_core::ReasoningItem {
        text: None,
        summary: vec!["s".into()],
        opaque: Some(blob),
        id: Some(llm_xlate_core::ItemId::new("rs_1")),
    }));
    ir.items.push(Item::assistant_text("done"));
    let b = body(&ir, &caps());
    let input = b.get("input").unwrap().as_array().unwrap();
    assert_eq!(input[0].get("type").unwrap(), "reasoning");
    assert_eq!(input[0].get("encrypted_content").unwrap(), "ENC");
    assert_eq!(b.get("store").unwrap(), false);
    let include = b.get("include").unwrap().as_array().unwrap();
    assert!(include.iter().any(|v| v == "reasoning.encrypted_content"));
}

#[test]
fn foreign_reasoning_blob_dropped_with_degradation() {
    let mut ir = IrRequest::default();
    let blob = OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "sig");
    ir.items.push(Item::Reasoning(llm_xlate_core::ReasoningItem {
        text: None,
        summary: vec![],
        opaque: Some(blob),
        id: None,
    }));
    let enc = encode(&ir, &caps());
    let b = json(&enc.body);
    let input = b.get("input").unwrap().as_array().unwrap();
    assert!(input.is_empty());
    assert!(enc.degradations.iter().any(|d| d.field == "reasoning"));
}

#[test]
fn assistant_refusal_and_text() {
    let mut ir = IrRequest::default();
    ir.items.push(Item::Message {
        role: Role::Assistant,
        content: vec![Part::text("hi"), Part::Refusal { text: "no".into() }],
        id: Some(llm_xlate_core::ItemId::new("msg_9")),
    });
    let b = body(&ir, &caps());
    let input = b.get("input").unwrap().as_array().unwrap();
    assert_eq!(input[0].get("id").unwrap(), "msg_9");
    let content = input[0].get("content").unwrap().as_array().unwrap();
    assert_eq!(content[0].get("type").unwrap(), "output_text");
    assert_eq!(content[1].get("type").unwrap(), "refusal");
}

#[test]
fn images_encode_data_https_file_id() {
    let mut ir = IrRequest::default();
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![
            Part::Image(MediaSource::Base64 { media_type: "image/png".into(), data: b"hi".to_vec().into() }),
            Part::Image(MediaSource::Url("https://x/y.png".into())),
            Part::Image(MediaSource::FileRef { family: ProviderFamily::OpenAI, id: "file_1".into() }),
        ],
        id: None,
    });
    let b = body(&ir, &caps());
    let content = b["input"][0]["content"].as_array().unwrap();
    assert!(content[0]["image_url"].as_str().unwrap().starts_with("data:image/png;base64,"));
    assert_eq!(content[1]["image_url"], "https://x/y.png");
    assert_eq!(content[2]["file_id"], "file_1");
}

#[test]
fn pdf_encode_variants_and_text_doc() {
    let mut ir = IrRequest::default();
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![
            Part::Document {
                source: MediaSource::Base64 { media_type: "application/pdf".into(), data: b"%PDF".to_vec().into() },
                title: Some("a.pdf".into()),
                media_type: "application/pdf".into(),
            },
            Part::Document {
                source: MediaSource::Url("https://x/a.pdf".into()),
                title: None,
                media_type: "application/pdf".into(),
            },
            Part::Document {
                source: MediaSource::Text("plain text".into()),
                title: Some("notes".into()),
                media_type: "text/plain".into(),
            },
        ],
        id: None,
    });
    let b = body(&ir, &caps());
    let content = b["input"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "input_file");
    assert!(content[0]["file_data"].as_str().unwrap().starts_with("data:application/pdf;base64,"));
    assert_eq!(content[1]["file_url"], "https://x/a.pdf");
    // Text doc → an input_text wrapped by wrap::document.
    assert_eq!(content[2]["type"], "input_text");
    assert!(content[2]["text"].as_str().unwrap().contains("<document title=\"notes\">"));
}

#[test]
fn input_file_base64_without_title_synthesizes_filename() {
    // OpenAI requires `filename` beside base64 `file_data`; a title-less cross-family PDF gets a
    // default name derived from the media type, while a supplied title is preserved.
    let mut ir = IrRequest::default();
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![
            Part::Document {
                source: MediaSource::Base64 { media_type: "application/pdf".into(), data: b"%PDF".to_vec().into() },
                title: None,
                media_type: "application/pdf".into(),
            },
            Part::Document {
                source: MediaSource::Base64 { media_type: "application/pdf".into(), data: b"%PDF".to_vec().into() },
                title: Some("report.pdf".into()),
                media_type: "application/pdf".into(),
            },
        ],
        id: None,
    });
    let b = body(&ir, &caps());
    let content = b["input"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "input_file");
    assert_eq!(content[0]["filename"], "document.pdf");
    assert_eq!(content[1]["filename"], "report.pdf");
}

#[test]
fn json_schema_strict_emitted_when_supported() {
    let mut ir = IrRequest::default();
    ir.output = OutputConfig {
        format: OutputFormat::JsonSchema {
            name: "S".into(),
            schema: serde_json::json!({"type":"object"}),
            strict: true,
            description: None,
        },
        verbosity: None,
    };
    let b = body(&ir, &caps());
    assert_eq!(b["text"]["format"]["type"], "json_schema");
    assert_eq!(b["text"]["format"]["strict"], true);
}

#[test]
fn json_schema_strict_downgraded_when_unsupported() {
    let mut ir = IrRequest::default();
    ir.output = OutputConfig {
        format: OutputFormat::JsonSchema {
            name: "S".into(),
            schema: serde_json::json!({"type":"object"}),
            strict: true,
            description: None,
        },
        verbosity: None,
    };
    // openai_compatible has output.strict_supported = false.
    let enc = codec_encode(&ir, &preset::openai_compatible());
    let b = json(&enc.body);
    assert_eq!(b["text"]["format"]["strict"], false);
    assert!(enc.degradations.iter().any(|d| d.field == "text.format.strict" && d.kind == DegradationKind::Downgraded));
}

fn codec_encode(req: &IrRequest, caps: &llm_xlate_core::Capabilities) -> llm_xlate_core::EncodedRequest {
    encode(req, caps)
}

#[test]
fn previous_response_id_supported_and_dropped() {
    let mut ir = IrRequest::default();
    ir.state.previous_response_id = Some(llm_xlate_core::ResponseId::new("resp_prev"));
    ir.items.push(Item::user_text("x"));
    // gpt5 supports it.
    let b = body(&ir, &caps());
    assert_eq!(b["previous_response_id"], "resp_prev");
    // gpt4o also supports (previous_response_id defaults true for openai family). Use a preset
    // that lacks it: openai_compatible (all state.* = false).
    let enc = encode(&ir, &preset::openai_compatible());
    let b2 = json(&enc.body);
    assert!(b2.get("previous_response_id").is_none());
    assert!(enc.degradations.iter().any(|d| d.field == "previous_response_id"));
}

#[test]
fn background_supported_and_dropped() {
    let mut ir = IrRequest::default();
    ir.state.background = Some(true);
    ir.items.push(Item::user_text("x"));
    let b = body(&ir, &caps());
    assert_eq!(b["background"], true);
    let enc = encode(&ir, &preset::gpt4o()); // gpt4o has background = false (default).
    let b2 = json(&enc.body);
    assert!(b2.get("background").is_none());
    assert!(enc.degradations.iter().any(|d| d.field == "background"));
}

#[test]
fn tool_choice_named_and_downgrade() {
    let mut ir = IrRequest::default();
    ir.tool_choice = ToolChoice::Named("f".into());
    ir.tools.push(ToolDef::Function {
        name: "f".into(),
        description: None,
        parameters: serde_json::json!({"type":"object"}),
        strict: None,
        cache_control: None,
    });
    let b = body(&ir, &caps());
    assert_eq!(b["tool_choice"]["type"], "function");
    assert_eq!(b["tool_choice"]["name"], "f");
}

#[test]
fn verbosity_emitted_and_dropped() {
    let mut ir = IrRequest::default();
    ir.output.verbosity = Some(Verbosity::High);
    let b = body(&ir, &caps());
    assert_eq!(b["text"]["verbosity"], "high");
    // gpt4o has verbosity=false.
    let enc = encode(&ir, &preset::gpt4o());
    let b2 = json(&enc.body);
    assert!(b2.get("text").map(|t| t.get("verbosity").is_none()).unwrap_or(true));
    assert!(enc.degradations.iter().any(|d| d.field == "verbosity"));
}

#[test]
fn effort_max_downgrades_to_xhigh() {
    let mut ir = IrRequest::default();
    ir.reasoning = ReasoningConfig {
        effort: Some(Effort::Max),
        budget_tokens: None,
        enabled: None,
        expose: ReasoningExposure::Summary(SummaryLevel::Auto),
    };
    let enc = encode(&ir, &caps());
    let b = json(&enc.body);
    assert_eq!(b["reasoning"]["effort"], "xhigh");
    assert!(enc.degradations.iter().any(|d| d.field == "reasoning.effort" && d.kind == DegradationKind::Downgraded));
}

#[test]
fn reasoning_full_exposure_emits_no_summary() {
    // `Full` exposure is the Chat/Anthropic decoder default (a response-rendering preference),
    // not a request-side summary request. A client sending only `reasoning_effort:"low"` must
    // NOT be forced into `summary:"detailed"`, and no `reasoning.summary` downgrade may fire.
    let mut ir = IrRequest::default();
    ir.reasoning = ReasoningConfig {
        effort: Some(Effort::Low),
        budget_tokens: None,
        enabled: None,
        expose: ReasoningExposure::Full,
    };
    let enc = encode(&ir, &caps());
    let b = json(&enc.body);
    assert_eq!(b["reasoning"]["effort"], "low");
    assert!(b["reasoning"].get("summary").is_none(), "no summary: {}", b["reasoning"]);
    assert!(!enc.degradations.iter().any(|d| d.field == "reasoning.summary"));
}

#[test]
fn service_tier_mapped_and_dropped() {
    let mut ir = IrRequest::default();
    ir.meta.service_tier = Some("flex".into());
    ir.items.push(Item::user_text("x"));
    let b = body(&ir, &caps());
    assert_eq!(b["service_tier"], "flex");

    ir.meta.service_tier = Some("nonsense".into());
    let enc = encode(&ir, &caps());
    let b2 = json(&enc.body);
    assert!(b2.get("service_tier").is_none());
    assert!(enc.degradations.iter().any(|d| d.field == "service_tier"));
}

#[test]
fn sampling_in_range_and_out_of_range() {
    let mut ir = IrRequest::default();
    ir.sampling = Sampling { temperature: Some(0.7), top_p: Some(0.5), ..Default::default() };
    ir.items.push(Item::user_text("x"));
    let b = body(&ir, &caps());
    assert_eq!(b["temperature"], 0.7);
    assert_eq!(b["top_p"], 0.5);

    ir.sampling = Sampling { temperature: Some(9.0), ..Default::default() };
    let enc = encode(&ir, &caps());
    let b2 = json(&enc.body);
    assert!(b2.get("temperature").is_none());
    assert!(enc.degradations.iter().any(|d| d.field == "temperature"));
}

#[test]
fn uncarriable_sampling_and_limits_are_dropped_with_degradations() {
    // Sampling/limits fields the Responses wire cannot carry (they arrive when translating from
    // Chat) must be dropped AND each records a Degradation.
    let mut ir = IrRequest::default();
    ir.sampling = Sampling {
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.25),
        seed: Some(42),
        top_k: Some(10),
        logit_bias: Some(serde_json::json!({"50256": -100})),
        ..Default::default()
    };
    ir.limits.stop_sequences = vec!["STOP".into()];
    ir.items.push(Item::user_text("x"));
    let enc = encode(&ir, &caps());
    let b = json(&enc.body);
    for f in ["frequency_penalty", "presence_penalty", "seed", "top_k", "logit_bias", "stop"] {
        assert!(b.get(f).is_none(), "`{f}` must not appear on the Responses wire");
    }
    for f in ["frequency_penalty", "presence_penalty", "seed", "top_k", "logit_bias", "stop_sequences"] {
        assert!(
            enc.degradations.iter().any(|d| d.field == f && d.kind == DegradationKind::Dropped),
            "missing Dropped degradation for `{f}`"
        );
    }
}

#[test]
fn ext_writeback_truncation() {
    let mut ir = IrRequest::default();
    ir.ext.insert("responses.truncation", Value::from("auto"));
    ir.ext.insert("responses.max_tool_calls", Value::from(3));
    ir.items.push(Item::user_text("x"));
    let b = body(&ir, &caps());
    assert_eq!(b["truncation"], "auto");
    assert_eq!(b["max_tool_calls"], 3);
}

#[test]
fn foreign_ext_namespace_ignored() {
    let mut ir = IrRequest::default();
    ir.ext.insert("chat.foo", Value::from("bar"));
    ir.items.push(Item::user_text("x"));
    let b = body(&ir, &caps());
    assert!(b.get("foo").is_none());
}

#[test]
fn tool_strict_none_with_attempted_default_emits_nothing() {
    let mut ir = IrRequest::default();
    ir.tools.push(ToolDef::Function {
        name: "f".into(),
        description: Some("d".into()),
        parameters: serde_json::json!({"type":"object"}),
        strict: None,
        cache_control: None,
    });
    let b = body(&ir, &caps());
    let tool = &b["tools"][0];
    assert_eq!(tool["type"], "function");
    assert!(tool.get("strict").is_none());
}

#[test]
fn stateless_forced_by_caps_zdr() {
    // A backend with store=false forces store:false + encrypted include even if the client did
    // not ask. openai_compatible has state.store = false.
    let mut ir = IrRequest::default();
    ir.items.push(Item::user_text("x"));
    let b = body(&ir, &preset::openai_compatible());
    assert_eq!(b["store"], false);
    assert!(b["include"].as_array().unwrap().iter().any(|v| v == "reasoning.encrypted_content"));
}

#[test]
fn function_tools_unsupported_is_error() {
    let mut ir = IrRequest::default();
    ir.tools.push(ToolDef::Function {
        name: "f".into(),
        description: None,
        parameters: serde_json::json!({}),
        strict: None,
        cache_control: None,
    });
    // openai_compatible has function_tools = Unknown (not Yes) → unsupported.
    let err = codec()
        .encode_request(&ir, &preset::openai_compatible(), &ectx())
        .unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::Unsupported);
}

#[test]
fn determinism_encode_twice_identical() {
    let ir = decode(
        r#"{"model":"gpt-5.4","instructions":"x","input":[{"role":"user","content":"hi"}],"reasoning":{"effort":"high"},"temperature":0.4}"#,
    );
    let a = encode(&ir, &caps()).body;
    let b = encode(&ir, &caps()).body;
    assert_eq!(a, b);
}

#[test]
fn state_config_default_is_stateless_noop() {
    // Sanity: a plain request against gpt5 (store default true) keeps store unset.
    let ir = decode(r#"{"model":"gpt-5.4","input":"hi"}"#);
    let b = body(&ir, &caps());
    assert!(b.get("store").is_none());
    let _ = StateConfig::default();
}

#[test]
fn long_user_identifier_is_digested_to_openai_limit() {
    // Regression (Claude Code on gpt-5.6-luna, 2026-09-10): Claude Code sends a ~150-char
    // `metadata.user_id`; forwarding it verbatim as `user` yields OpenAI 400 "Invalid 'user':
    // string too long ... maximum length 64". Over-long identifiers become a deterministic
    // 64-char sha256 digest with a `Rewritten` degradation; short ones pass through untouched.
    let mut ir = IrRequest::default();
    ir.items.push(Item::user_text("hi"));
    ir.meta.user = Some(format!("user_{}", "a".repeat(145)));
    ir.meta.safety_identifier = Some("short-id".into());
    let enc = llm_xlate_responses::ResponsesCodec
        .encode_request(&ir, &caps(), &llm_xlate_core::EncodeCtx::new(
            llm_xlate_core::Protocol::OaiResponses, "m", llm_xlate_core::ResponseId::new("r"),
            llm_xlate_core::Sealer::new(b"k")))
        .unwrap();
    let b: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    let user = b.get("user").and_then(|v| v.as_str()).unwrap();
    assert_eq!(user.len(), 64);
    assert!(user.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(b.get("safety_identifier").and_then(|v| v.as_str()), Some("short-id"));
    assert!(enc.degradations.iter().any(|d| d.field == "user"), "rewrite must be recorded");
    assert!(!enc.degradations.iter().any(|d| d.field == "safety_identifier"));
}
