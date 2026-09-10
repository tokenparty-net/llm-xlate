//! `decode_request` tests: Responses request body → IR (lossless).

mod common;
use common::*;

use llm_xlate_core::{
    Codec, Effort, InstructionRole, Item, MediaSource, OutputFormat, Part, Position, ProviderFamily,
    ReasoningExposure, Role, SummaryLevel, ToolChoice, ToolDef,
};
use pretty_assertions::assert_eq;

#[test]
fn instructions_string_becomes_leading_system() {
    let ir = decode(r#"{"model":"gpt-5.4","instructions":"Be terse.","input":"hi"}"#);
    assert_eq!(ir.instructions.len(), 1);
    assert_eq!(ir.instructions[0].role, InstructionRole::System);
    assert_eq!(ir.instructions[0].position, Position::Leading);
    assert_eq!(ir.instructions[0].text(), "Be terse.");
}

#[test]
fn input_string_becomes_user_message() {
    let ir = decode(r#"{"model":"m","input":"hello world"}"#);
    assert_eq!(ir.items, vec![Item::user_text("hello world")]);
}

#[test]
fn model_is_captured() {
    let ir = decode(r#"{"model":"gpt-6-astra","input":"x"}"#);
    assert_eq!(ir.model.upstream(), "gpt-6-astra");
}

#[test]
fn message_shorthand_no_type() {
    let ir = decode(r#"{"model":"m","input":[{"role":"user","content":"hey"}]}"#);
    assert_eq!(ir.items, vec![Item::user_text("hey")]);
}

#[test]
fn message_typed_with_parts() {
    let ir = decode(
        r#"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]}]}"#,
    );
    match &ir.items[0] {
        Item::Message { role: Role::User, content, .. } => assert_eq!(content.len(), 2),
        other => panic!("{other:?}"),
    }
}

#[test]
fn leading_system_item_then_user() {
    let ir = decode(
        r#"{"model":"m","input":[{"type":"message","role":"system","content":"sys"},{"role":"user","content":"u"}]}"#,
    );
    assert_eq!(ir.instructions.len(), 1);
    assert_eq!(ir.instructions[0].position, Position::Leading);
    assert_eq!(ir.items.len(), 1);
}

#[test]
fn mid_system_becomes_before() {
    let ir = decode(
        r#"{"model":"m","input":[{"role":"user","content":"u"},{"role":"system","content":"later"}]}"#,
    );
    assert_eq!(ir.instructions.len(), 1);
    assert_eq!(ir.instructions[0].position, Position::Before(1));
}

#[test]
fn developer_role_instruction() {
    let ir = decode(
        r#"{"model":"m","input":[{"role":"developer","content":"dev"},{"role":"user","content":"u"}]}"#,
    );
    assert_eq!(ir.instructions[0].role, InstructionRole::Developer);
}

#[test]
fn function_call_and_output() {
    let ir = decode(
        r#"{"model":"m","input":[
            {"type":"function_call","call_id":"call_1","name":"get","arguments":"{\"x\":1}","id":"fc_1"},
            {"type":"function_call_output","call_id":"call_1","output":"42"}
        ]}"#,
    );
    match &ir.items[0] {
        Item::ToolCall { call_id, name, arguments, id } => {
            assert_eq!(call_id.as_str(), "call_1");
            assert_eq!(name, "get");
            assert_eq!(arguments.as_str(), "{\"x\":1}");
            assert_eq!(id.as_ref().unwrap().as_str(), "fc_1");
        }
        other => panic!("{other:?}"),
    }
    match &ir.items[1] {
        Item::ToolResult { call_id, content, .. } => {
            assert_eq!(call_id.as_str(), "call_1");
            assert_eq!(content[0].as_text(), Some("42"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn reasoning_item_with_encrypted_content() {
    let ir = decode(
        r#"{"model":"m","input":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"S"}],"content":[{"type":"reasoning_text","text":"deep"}],"encrypted_content":"BLOB"}]}"#,
    );
    match &ir.items[0] {
        Item::Reasoning(ri) => {
            assert_eq!(ri.summary, vec!["S".to_string()]);
            assert_eq!(ri.text.as_deref(), Some("deep"));
            let o = ri.opaque.as_ref().unwrap();
            assert_eq!(o.family, ProviderFamily::OpenAI);
            assert_eq!(o.data, "BLOB");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn image_data_url_https_and_file_id() {
    let ir = decode(
        r#"{"model":"m","input":[{"role":"user","content":[
            {"type":"input_image","image_url":"data:image/png;base64,aGk="},
            {"type":"input_image","image_url":"https://x/y.png"},
            {"type":"input_image","file_id":"file_9"}
        ]}]}"#,
    );
    let parts = match &ir.items[0] {
        Item::Message { content, .. } => content,
        other => panic!("{other:?}"),
    };
    assert!(matches!(parts[0], Part::Image(MediaSource::Base64 { .. })));
    assert!(matches!(&parts[1], Part::Image(MediaSource::Url(u)) if u == "https://x/y.png"));
    assert!(matches!(&parts[2], Part::Image(MediaSource::FileRef { id, .. }) if id == "file_9"));
}

#[test]
fn image_detail_recorded_in_ext() {
    let ir = decode(
        r#"{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"https://x","detail":"high"}]}]}"#,
    );
    let detail = ir.ext.get("responses.image_detail").unwrap();
    assert_eq!(detail.get("0/0").unwrap(), "high");
}

#[test]
fn input_file_variants() {
    let ir = decode(
        r#"{"model":"m","input":[{"role":"user","content":[
            {"type":"input_file","file_data":"data:application/pdf;base64,JVBER","filename":"a.pdf"},
            {"type":"input_file","file_url":"https://x/a.pdf"},
            {"type":"input_file","file_id":"file_7"}
        ]}]}"#,
    );
    let parts = match &ir.items[0] {
        Item::Message { content, .. } => content,
        other => panic!("{other:?}"),
    };
    assert!(matches!(&parts[0], Part::Document { source: MediaSource::Base64 { .. }, title, .. } if title.as_deref()==Some("a.pdf")));
    assert!(matches!(&parts[1], Part::Document { source: MediaSource::Url(_), .. }));
    assert!(matches!(&parts[2], Part::Document { source: MediaSource::FileRef { id, .. }, .. } if id=="file_7"));
}

#[test]
fn tool_choice_variants() {
    assert_eq!(decode(r#"{"model":"m","input":"x","tool_choice":"none"}"#).tool_choice, ToolChoice::None);
    assert_eq!(decode(r#"{"model":"m","input":"x","tool_choice":"required"}"#).tool_choice, ToolChoice::Required);
    assert_eq!(decode(r#"{"model":"m","input":"x","tool_choice":"auto"}"#).tool_choice, ToolChoice::Auto);
    assert_eq!(
        decode(r#"{"model":"m","input":"x","tool_choice":{"type":"function","name":"f"}}"#).tool_choice,
        ToolChoice::Named("f".into())
    );
}

#[test]
fn hosted_tool_choice_preserved_in_ext() {
    let ir = decode(
        r#"{"model":"m","input":"x","tool_choice":{"type":"allowed_tools","tools":[]}}"#,
    );
    assert_eq!(ir.tool_choice, ToolChoice::Auto);
    assert!(ir.ext.get("responses.tool_choice").is_some());
}

#[test]
fn function_tool_and_hosted_tool() {
    let ir = decode(
        r#"{"model":"m","input":"x","tools":[
            {"type":"function","name":"f","description":"d","parameters":{"type":"object"},"strict":true},
            {"type":"web_search"}
        ]}"#,
    );
    match &ir.tools[0] {
        ToolDef::Function { name, strict, .. } => {
            assert_eq!(name, "f");
            assert_eq!(*strict, Some(true));
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(&ir.tools[1], ToolDef::Provider(oi) if oi.family == ProviderFamily::OpenAI));
}

#[test]
fn text_format_json_schema() {
    let ir = decode(
        r#"{"model":"m","input":"x","text":{"format":{"type":"json_schema","name":"S","schema":{"type":"object"},"strict":true,"description":"d"},"verbosity":"low"}}"#,
    );
    match &ir.output.format {
        OutputFormat::JsonSchema { name, strict, description, .. } => {
            assert_eq!(name, "S");
            assert!(*strict);
            assert_eq!(description.as_deref(), Some("d"));
        }
        other => panic!("{other:?}"),
    }
    assert!(ir.output.verbosity.is_some());
}

#[test]
fn reasoning_effort_and_summary() {
    let ir = decode(r#"{"model":"m","input":"x","reasoning":{"effort":"high","summary":"detailed"}}"#);
    assert_eq!(ir.reasoning.effort, Some(Effort::High));
    assert_eq!(ir.reasoning.expose, ReasoningExposure::Summary(SummaryLevel::Detailed));
}

#[test]
fn reasoning_effort_without_summary_defaults_auto() {
    let ir = decode(r#"{"model":"m","input":"x","reasoning":{"effort":"low"}}"#);
    assert_eq!(ir.reasoning.expose, ReasoningExposure::Summary(SummaryLevel::Auto));
}

#[test]
fn reasoning_effort_none_does_not_default_to_summary() {
    // `effort:"none"` disables reasoning; it must NOT default to a summary (re-encoding that would
    // emit the contradictory `{effort:"none",summary:"auto"}`).
    let ir = decode(r#"{"model":"m","input":"x","reasoning":{"effort":"none"}}"#);
    assert_eq!(ir.reasoning.effort, Some(Effort::None));
    assert_eq!(ir.reasoning.expose, ReasoningExposure::None);
    // And re-encoding does not attach a summary token.
    let b = json(&encode(&ir, &caps()).body);
    assert!(b["reasoning"].get("summary").is_none(), "must not request a summary for effort:none");
}

#[test]
fn state_fields_decoded() {
    let ir = decode(
        r#"{"model":"m","input":"x","store":false,"previous_response_id":"resp_prev","background":true,"include":["reasoning.encrypted_content"],"conversation":{"id":"conv_1"}}"#,
    );
    assert_eq!(ir.state.store, Some(false));
    assert_eq!(ir.state.previous_response_id.as_ref().unwrap().as_str(), "resp_prev");
    assert_eq!(ir.state.background, Some(true));
    assert_eq!(ir.state.include, vec!["reasoning.encrypted_content".to_string()]);
    assert_eq!(ir.state.conversation.as_deref(), Some("conv_1"));
}

#[test]
fn meta_and_cache_and_sampling() {
    let ir = decode(
        r#"{"model":"m","input":"x","user":"u1","safety_identifier":"s1","service_tier":"flex","prompt_cache_key":"ck","temperature":0.5,"top_p":0.9,"metadata":{"k":"v"}}"#,
    );
    assert_eq!(ir.meta.user.as_deref(), Some("u1"));
    assert_eq!(ir.meta.safety_identifier.as_deref(), Some("s1"));
    assert_eq!(ir.meta.service_tier.as_deref(), Some("flex"));
    assert_eq!(ir.cache.prompt_cache_key.as_deref(), Some("ck"));
    assert_eq!(ir.sampling.temperature, Some(0.5));
    assert_eq!(ir.sampling.top_p, Some(0.9));
    assert_eq!(ir.meta.metadata.get("k").unwrap(), "v");
}

#[test]
fn unknown_fields_go_to_ext() {
    let ir = decode(r#"{"model":"m","input":"x","truncation":"auto","max_tool_calls":5,"prompt":{"id":"p"}}"#);
    assert_eq!(ir.ext.get("responses.truncation").unwrap(), "auto");
    assert_eq!(ir.ext.get("responses.max_tool_calls").unwrap(), 5);
    assert!(ir.ext.get("responses.prompt").is_some());
}

#[test]
fn item_reference_becomes_provider_tool_call() {
    let ir = decode(r#"{"model":"m","input":[{"type":"item_reference","id":"msg_x"}]}"#);
    assert!(matches!(&ir.items[0], Item::ProviderToolCall(oi) if oi.family == ProviderFamily::OpenAI));
}

#[test]
fn hosted_call_item_becomes_provider_tool_call() {
    let ir = decode(r#"{"model":"m","input":[{"type":"web_search_call","id":"ws_1","status":"completed"}]}"#);
    assert!(matches!(&ir.items[0], Item::ProviderToolCall(_)));
}

#[test]
fn hosted_call_output_becomes_provider_tool_result() {
    let ir = decode(r#"{"model":"m","input":[{"type":"computer_call_output","call_id":"c1","output":{}}]}"#);
    assert!(matches!(&ir.items[0], Item::ProviderToolResult(_)));
}

#[test]
fn compaction_item() {
    let ir = decode(r#"{"model":"m","input":[{"type":"compaction","id":"cmp_1","encrypted_content":"Z"}]}"#);
    match &ir.items[0] {
        Item::Compaction(b) => assert_eq!(b.data, "Z"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn stream_flag_and_refusal_part() {
    let ir = decode(
        r#"{"model":"m","stream":true,"input":[{"role":"assistant","content":[{"type":"refusal","refusal":"no"}]}]}"#,
    );
    assert!(ir.stream);
    match &ir.items[0] {
        Item::Message { content, .. } => assert!(matches!(&content[0], Part::Refusal { text } if text == "no")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn bad_body_is_invalid_request() {
    let err = codec().decode_request(b"not json", &llm_xlate_core::HeaderMap::new(), &dctx()).unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::ErrorKind::InvalidRequest);
}

#[test]
fn sealed_foreign_reasoning_opens_to_its_family() {
    // A foreign (Anthropic) reasoning blob sealed as rtr1. opens with the envelope family.
    let sealer = llm_xlate_core::Sealer::new(KEY);
    let blob = llm_xlate_core::OpaqueBlob::new(
        ProviderFamily::Anthropic,
        llm_xlate_core::OpaqueKind::Signature,
        "sig",
    );
    let sealed = sealer.seal(&blob);
    let body = format!(
        r#"{{"model":"m","input":[{{"type":"reasoning","summary":[],"encrypted_content":"{sealed}"}}]}}"#
    );
    let ir = decode(&body);
    match &ir.items[0] {
        Item::Reasoning(ri) => assert_eq!(ri.opaque.as_ref().unwrap().family, ProviderFamily::Anthropic),
        other => panic!("{other:?}"),
    }
}
