#![allow(clippy::field_reassign_with_default)]
//! Round-trip law: `decode(encode(ir)) == ir` for same-family (OpenAI Responses) requests,
//! modulo the documented normalizations (see the crate report).

mod common;
use common::*;

use llm_xlate_core::{
    CallId, Effort, Instruction, IrRequest, Item, JsonText, MediaSource, OutputConfig, OutputFormat,
    Part, ProviderFamily, ReasoningConfig, ReasoningExposure, Role, Sampling, SummaryLevel,
    ToolChoice, ToolDef, Verbosity,
};
use pretty_assertions::assert_eq;

/// Encode `ir` against gpt-5 Responses caps, decode the bytes back, assert equality.
fn rt(ir: IrRequest) {
    rt_with(ir, &caps());
}

/// Like [`rt`] but against explicit capabilities (for fields the default preset gates off).
fn rt_with(ir: IrRequest, caps: &llm_xlate_core::Capabilities) {
    let bytes = encode_with(&ir, caps).body;
    let back = codec_decode(&bytes);
    assert_eq!(back, ir);
}

fn encode_with(ir: &IrRequest, caps: &llm_xlate_core::Capabilities) -> llm_xlate_core::EncodedRequest {
    use llm_xlate_core::Codec;
    codec().encode_request(ir, caps, &ectx()).unwrap()
}

/// Default gpt-5 caps with audio input enabled (no shipped OpenAI preset carries audio).
fn caps_with_audio() -> llm_xlate_core::Capabilities {
    let mut c = caps();
    c.media.audio.sources =
        Some(vec![llm_xlate_core::caps::MediaSourceKind::Base64, llm_xlate_core::caps::MediaSourceKind::Url]);
    c
}

fn codec_decode(bytes: &[u8]) -> IrRequest {
    use llm_xlate_core::Codec;
    codec().decode_request(bytes, &llm_xlate_core::HeaderMap::new(), &dctx()).unwrap()
}

#[test]
fn rt_plain_message() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("gpt-5.4");
    ir.items.push(Item::user_text("hi"));
    rt(ir);
}

#[test]
fn rt_single_system_instruction() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("gpt-5.4");
    ir.instructions.push(Instruction::system_text("be nice"));
    ir.items.push(Item::user_text("hi"));
    rt(ir);
}

#[test]
fn rt_multi_turn_with_assistant() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.items.push(Item::user_text("q"));
    ir.items.push(Item::Message {
        role: Role::Assistant,
        content: vec![Part::text("a")],
        id: None,
    });
    ir.items.push(Item::user_text("q2"));
    rt(ir);
}

#[test]
fn rt_tool_loop() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.parallel_tool_calls = Some(false);
    ir.items.push(Item::user_text("weather"));
    ir.items.push(Item::ToolCall {
        call_id: CallId::new("c1"),
        name: "get".into(),
        arguments: JsonText::new("{\"city\":\"NYC\"}"),
        id: None,
    });
    ir.items.push(Item::ToolResult {
        call_id: CallId::new("c1"),
        content: vec![Part::text("sunny")],
        is_error: false,
        id: None,
    });
    rt(ir);
}

#[test]
fn rt_reasoning_effort_and_summary() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.reasoning = ReasoningConfig {
        effort: Some(Effort::High),
        budget_tokens: None,
        enabled: None,
        expose: ReasoningExposure::Summary(SummaryLevel::Detailed),
    };
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_json_schema_output() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.output = OutputConfig {
        format: OutputFormat::JsonSchema {
            name: "S".into(),
            schema: serde_json::json!({"type":"object","properties":{"a":{"type":"string"}}}),
            strict: true,
            description: Some("d".into()),
        },
        verbosity: Some(Verbosity::Low),
    };
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_json_object_output() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.output = OutputConfig { format: OutputFormat::JsonObject, verbosity: None };
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_tools_and_choice() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.tools.push(ToolDef::Function {
        name: "f".into(),
        description: Some("d".into()),
        parameters: serde_json::json!({"type":"object"}),
        strict: Some(true),
        cache_control: None,
    });
    ir.tool_choice = ToolChoice::Required;
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_named_tool_choice() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.tools.push(ToolDef::Function {
        name: "f".into(),
        description: None,
        parameters: serde_json::json!({"type":"object"}),
        strict: Some(false),
        cache_control: None,
    });
    ir.tool_choice = ToolChoice::Named("f".into());
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_sampling() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.sampling = Sampling { temperature: Some(0.5), top_p: Some(0.9), ..Default::default() };
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_images() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![
            Part::Image(MediaSource::Base64 { media_type: "image/png".into(), data: b"abc".to_vec().into() }),
            Part::Image(MediaSource::Url("https://x/y.png".into())),
            Part::Image(MediaSource::FileRef { family: ProviderFamily::OpenAI, id: "file_1".into() }),
        ],
        id: None,
    });
    rt(ir);
}

#[test]
fn rt_meta_and_cache() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.meta.user = Some("u".into());
    ir.meta.safety_identifier = Some("s".into());
    ir.meta.service_tier = Some("flex".into());
    ir.cache.prompt_cache_key = Some("ck".into());
    // The decoder re-captures the session id from prompt_cache_key (highest-priority source).
    ir.session.id = Some("ck".into());
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_max_output_tokens_and_stream() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.limits.max_output_tokens = Some(256);
    ir.stream = true;
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_ext_passthrough() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.ext.insert("responses.truncation", serde_json::Value::from("auto"));
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_conversation_when_supported() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.state.conversation = Some("conv_1".into());
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_state_background_prev_include() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("gpt-5.4");
    ir.state.background = Some(true);
    ir.state.previous_response_id = Some(llm_xlate_core::ResponseId::new("resp_prev"));
    ir.state.include = vec!["reasoning.encrypted_content".to_string()];
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_top_logprobs() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("gpt-5.4");
    ir.sampling = Sampling { top_logprobs: Some(5), ..Default::default() };
    ir.items.push(Item::user_text("x"));
    rt(ir);
}

#[test]
fn rt_audio_input() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("gpt-5.4");
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![
            Part::Audio(MediaSource::Url("https://x/a.wav".into())),
            Part::Audio(MediaSource::Base64 { media_type: "audio/wav".into(), data: b"abc".to_vec().into() }),
        ],
        id: None,
    });
    rt_with(ir, &caps_with_audio());
}

#[test]
fn rt_image_detail_ext_survives() {
    // The index-keyed `responses.image_detail` ext entry (a documented normalization) survives
    // encode → decode when the target advertises `image_detail_param` (gpt-5 default caps do).
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("gpt-5.4");
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![Part::Image(MediaSource::Url("https://x/y.png".into()))],
        id: None,
    });
    ir.ext.insert("responses.image_detail", serde_json::json!({ "0/0": "high" }));
    rt(ir);
}

#[test]
fn rt_pdf_and_text_doc() {
    let mut ir = IrRequest::default();
    ir.model = llm_xlate_core::ModelRef::new("m");
    ir.items.push(Item::Message {
        role: Role::User,
        content: vec![Part::Document {
            source: MediaSource::Url("https://x/a.pdf".into()),
            title: None,
            media_type: "application/pdf".into(),
        }],
        id: None,
    });
    rt(ir);
}
