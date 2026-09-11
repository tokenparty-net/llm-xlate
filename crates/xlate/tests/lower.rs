//! Tests for the ordered lowering passes (plan §7). One (or more) per rule.

mod common;
use common::*;

use llm_xlate::caps::{preset, Capabilities, OutputFormatCap, ReplayMode, ToolChoiceKind, Tri};
use llm_xlate::codec::{ForeignProviderTool, MidInstructionFallback, TranslatorConfig, UnresolvedReasoning};
use llm_xlate::degrade::DegradationKind;
use llm_xlate::error::ErrorKind;
use llm_xlate::ir::{
    CallId, Effort, IrRequest, Item, OutputFormat, Part, Protocol, ProviderFamily, ToolChoice,
    ToolDef, Verbosity,
};
use llm_xlate::lower::lower;
use llm_xlate::requirements::{FileRef, Resolutions};
use pretty_assertions::assert_eq;
use serde_json::json;

// ── helpers ──────────────────────────────────────────────────────────────────────────────────

fn cfg() -> TranslatorConfig {
    TranslatorConfig::default()
}

fn low(req: IrRequest, caps: &Capabilities, target: Protocol) -> llm_xlate::lower::Lowered {
    lower(req, caps, target, &Resolutions::new(), &cfg()).expect("lower ok")
}

fn low_res(
    req: IrRequest,
    caps: &Capabilities,
    target: Protocol,
    res: &Resolutions,
) -> llm_xlate::lower::Lowered {
    lower(req, caps, target, res, &cfg()).expect("lower ok")
}

fn err(req: IrRequest, caps: &Capabilities, target: Protocol) -> llm_xlate::error::XlateError {
    lower(req, caps, target, &Resolutions::new(), &cfg()).expect_err("lower err")
}

/// True if any degradation touches `field`.
fn has_field(l: &llm_xlate::lower::Lowered, field: &str) -> bool {
    l.degradations.iter().any(|d| d.field == field)
}

fn kind_of(l: &llm_xlate::lower::Lowered, field: &str) -> Option<DegradationKind> {
    l.degradations.iter().find(|d| d.field == field).map(|d| d.kind)
}

// ── 1. protocol ────────────────────────────────────────────────────────────────────────────

#[test]
fn protocol_unsupported_rejected() {
    // gpt6 is Responses-only.
    let e = err(req_with_items(vec![user("hi")]), &preset::gpt6(), Protocol::OaiChat);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("protocol"));
}

#[test]
fn protocol_supported_ok() {
    let l = low(req_with_items(vec![user("hi")]), &preset::gpt6(), Protocol::OaiResponses);
    assert!(l.degradations.is_empty());
}

// ── 2. reasoning ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reasoning_keep_native_opaque() {
    let req = req_with_items(vec![reasoning_opaque(ProviderFamily::Anthropic), asst("done")]);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(matches!(l.req.items[0], Item::Reasoning(_)));
    assert!(!has_field(&l, "reasoning"));
}

#[test]
fn reasoning_drop_foreign_opaque() {
    let req = req_with_items(vec![reasoning_opaque(ProviderFamily::OpenAI), asst("done")]);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(!l.req.items.iter().any(|i| matches!(i, Item::Reasoning(_))));
    assert_eq!(kind_of(&l, "reasoning"), Some(DegradationKind::Dropped));
}

#[test]
fn reasoning_drop_text_when_no_replay_slot() {
    // gpt5_responses replay = encrypted_item (not TextField) → text-only reasoning dropped.
    let req = req_with_items(vec![reasoning_text(), asst("done")]);
    let l = low(req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert!(!l.req.items.iter().any(|i| matches!(i, Item::Reasoning(_))));
    assert!(has_field(&l, "reasoning"));
}

#[test]
fn reasoning_keep_text_when_textfield_replay() {
    let mut caps = preset::gpt4o();
    caps.reasoning.replay = Some(ReplayMode::TextField);
    let req = req_with_items(vec![reasoning_text(), asst("done")]);
    let l = low(req, &caps, Protocol::OaiChat);
    assert!(l.req.items.iter().any(|i| matches!(i, Item::Reasoning(_))));
}

/// Build an active tool-loop request (`[user, tool_call, tool_result]`) with reasoning enabled,
/// so the `required_on_last_tool_turn` check legitimately fires (see the gating in
/// `reasoning_required_skipped_when_reasoning_disabled`).
fn reasoning_tool_loop() -> IrRequest {
    let mut req = req_with_items(vec![
        user("q"),
        tool_call("call_1", "search"),
        tool_result("call_1", "r"),
    ]);
    req.reasoning.effort = Some(Effort::Medium);
    req
}

#[test]
fn reasoning_required_resolved_inserts_blob() {
    let mut res = Resolutions::new();
    res.reasoning.insert(CallId::new("call_1"), blob(ProviderFamily::Anthropic));
    let l = low_res(reasoning_tool_loop(), &preset::claude_5(), Protocol::Anthropic, &res);
    // A reasoning item now sits immediately before the tool call.
    let pos = l.req.items.iter().position(|i| matches!(i, Item::ToolCall { .. })).unwrap();
    assert!(matches!(l.req.items[pos - 1], Item::Reasoning(_)));
}

#[test]
fn reasoning_required_unresolved_fails() {
    // default cfg.unresolved_reasoning == Fail
    let e = err(reasoning_tool_loop(), &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::IncompatibleHistory);
}

#[test]
fn reasoning_required_unresolved_strips() {
    let mut c = cfg();
    c.unresolved_reasoning = UnresolvedReasoning::StripAndDegrade;
    let l = lower(reasoning_tool_loop(), &preset::claude_5(), Protocol::Anthropic, &Resolutions::new(), &c)
        .unwrap();
    assert!(l.req.items.iter().any(|i| matches!(i, Item::ToolCall { .. })));
    assert_eq!(kind_of(&l, "reasoning.required"), Some(DegradationKind::Dropped));
}

#[test]
fn reasoning_required_skipped_when_reasoning_disabled() {
    // A plain agentic loop with reasoning disabled: Anthropic only requires a thinking block on a
    // tool turn that *had* extended thinking, so the check must NOT fire and the request must not
    // be rejected (the historical blocker). No reasoning item is fabricated either.
    let req = req_with_items(vec![
        user("q"),
        tool_call("call_1", "search"),
        tool_result("call_1", "r"),
    ]);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(l.req.items.iter().any(|i| matches!(i, Item::ToolCall { .. })));
    assert!(!l.req.items.iter().any(|i| matches!(i, Item::Reasoning(_))));
    assert!(kind_of(&l, "reasoning.required").is_none());
}

#[test]
fn reasoning_required_skipped_for_historical_tool_turn() {
    // The tool turn is answered and a *later* user turn follows, so it is no longer the active
    // continuation; the required-reasoning check must not fire even with reasoning enabled.
    let mut req = req_with_items(vec![
        user("q"),
        tool_call("call_1", "search"),
        tool_result("call_1", "r"),
        user("thanks, now something else"),
    ]);
    req.reasoning.effort = Some(Effort::Medium);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(l.req.items.iter().any(|i| matches!(i, Item::ToolCall { .. })));
    assert!(kind_of(&l, "reasoning.required").is_none());
}

#[test]
fn reasoning_responses_pairing_drops_unpaired() {
    // A reasoning item followed by a user message is not paired → dropped for Responses.
    let req = req_with_items(vec![reasoning_opaque(ProviderFamily::OpenAI), user("next")]);
    // Use gpt5_responses but supply an openai (native) opaque so it survives step (a).
    let req2 = req_with_items(vec![
        Item::Reasoning(llm_xlate::ir::ReasoningItem {
            text: None,
            summary: vec![],
            opaque: Some(llm_xlate::ir::OpaqueBlob::new(
                ProviderFamily::OpenAI,
                llm_xlate::ir::OpaqueKind::Encrypted,
                "enc",
            )),
            id: None,
        }),
        user("next"),
    ]);
    let _ = req;
    let l = low(req2, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert!(!l.req.items.iter().any(|i| matches!(i, Item::Reasoning(_))));
    assert!(has_field(&l, "reasoning"));
}

#[test]
fn reasoning_responses_pairing_keeps_paired() {
    let req = req_with_items(vec![
        Item::Reasoning(llm_xlate::ir::ReasoningItem {
            text: None,
            summary: vec![],
            opaque: Some(llm_xlate::ir::OpaqueBlob::new(
                ProviderFamily::OpenAI,
                llm_xlate::ir::OpaqueKind::Encrypted,
                "enc",
            )),
            id: None,
        }),
        tool_call("call_1", "search"),
    ]);
    let l = low(req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert!(l.req.items.iter().any(|i| matches!(i, Item::Reasoning(_))));
}

#[test]
fn reasoning_effort_snaps_down() {
    // gpt5_responses effort_levels: minimal..xhigh (no Max) → Max snaps to XHigh.
    let mut req = req_with_items(vec![user("hi")]);
    req.reasoning.effort = Some(Effort::Max);
    let l = low(req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert_eq!(l.req.reasoning.effort, Some(Effort::XHigh));
    assert_eq!(kind_of(&l, "reasoning.effort"), Some(DegradationKind::Downgraded));
}

#[test]
fn reasoning_effort_snaps_minimal_up_to_low() {
    // Claude 3.7 (budget-mode) effort_levels: low..xhigh (no Minimal) → Minimal snaps up to Low.
    // (Claude 3.5 / claude_old no longer has a thinking mode, so use a real budget-mode model.)
    let caps = llm_xlate::caps::shipped()
        .resolve(&ProviderFamily::Anthropic, "claude-3-7-sonnet-latest", None);
    let mut req = req_with_items(vec![user("hi")]);
    req.reasoning.effort = Some(Effort::Minimal);
    let l = low(req, &caps, Protocol::Anthropic);
    assert_eq!(l.req.reasoning.effort, Some(Effort::Low));
    assert!(has_field(&l, "reasoning.effort"));
}

#[test]
fn reasoning_effort_supported_unchanged() {
    let mut req = req_with_items(vec![user("hi")]);
    req.reasoning.effort = Some(Effort::High);
    let l = low(req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert_eq!(l.req.reasoning.effort, Some(Effort::High));
    assert!(!has_field(&l, "reasoning.effort"));
}

#[test]
fn reasoning_mode_none_clears_effort() {
    // gpt4o has reasoning.mode = none.
    let mut req = req_with_items(vec![user("hi")]);
    req.reasoning.effort = Some(Effort::High);
    let l = low(req, &preset::gpt4o(), Protocol::OaiChat);
    assert_eq!(l.req.reasoning.effort, None);
    assert_eq!(kind_of(&l, "reasoning"), Some(DegradationKind::Dropped));
}

#[test]
fn reasoning_budget_rejected_cleared_effort_derived() {
    // claude_5 budget status = rejected.
    let mut req = req_with_items(vec![user("hi")]);
    req.reasoning.budget_tokens = Some(10000); // -> High bucket
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.reasoning.budget_tokens, None);
    assert_eq!(l.req.reasoning.effort, Some(Effort::High));
    assert!(has_field(&l, "reasoning.budget_tokens"));
}

#[test]
fn reasoning_derived_effort_is_snapped_to_supported_levels() {
    // Budget rejected + no explicit effort → effort is derived from the budget (5000 → Medium),
    // which must then be snapped to a supported level ([Low, High] → Low), not handed to the
    // codec unsnapped. Guards against the derive-after-snap ordering bug.
    let mut caps = preset::claude_5();
    caps.reasoning.effort_levels = Some(vec![Effort::Low, Effort::High]);
    let mut req = req_with_items(vec![user("hi")]);
    req.reasoning.budget_tokens = Some(5000); // -> Medium bucket, unsupported here
    let l = low(req, &caps, Protocol::Anthropic);
    assert_eq!(l.req.reasoning.budget_tokens, None);
    assert_eq!(l.req.reasoning.effort, Some(Effort::Low));
    assert!(has_field(&l, "reasoning.budget_tokens"));
}

#[test]
fn reasoning_budget_accepted_kept() {
    // Claude 3.7 budget status = accepted → budget preserved. (Claude 3.5 / claude_old no longer
    // has a thinking mode, so use a real budget-mode model.)
    let caps = llm_xlate::caps::shipped()
        .resolve(&ProviderFamily::Anthropic, "claude-3-7-sonnet-latest", None);
    let mut req = req_with_items(vec![user("hi")]);
    req.reasoning.budget_tokens = Some(4096);
    let l = low(req, &caps, Protocol::Anthropic);
    assert_eq!(l.req.reasoning.budget_tokens, Some(4096));
    assert!(!has_field(&l, "reasoning.budget_tokens"));
}

// ── 3. provider tools ────────────────────────────────────────────────────────────────────────

#[test]
fn provider_tool_foreign_unsupported_by_default() {
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![provider_tool(ProviderFamily::OpenAI, "web_search")];
    let e = err(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("tools"));
}

#[test]
fn provider_tool_foreign_dropped_with_config() {
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![provider_tool(ProviderFamily::OpenAI, "web_search")];
    let mut c = cfg();
    c.foreign_provider_tool = ForeignProviderTool::Drop;
    let l = lower(req, &preset::claude_5(), Protocol::Anthropic, &Resolutions::new(), &c).unwrap();
    assert!(l.req.tools.is_empty());
    assert_eq!(kind_of(&l, "tools"), Some(DegradationKind::Dropped));
}

#[test]
fn provider_tool_same_family_kept() {
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![provider_tool(ProviderFamily::Anthropic, "web_search_20260318")];
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.tools.len(), 1);
}

#[test]
fn provider_call_item_folded_to_assistant_text() {
    let req = req_with_items(vec![
        user("q"),
        provider_call(ProviderFamily::OpenAI, "web_search"),
    ]);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(!l.req.items.iter().any(|i| matches!(i, Item::ProviderToolCall(_))));
    // Folded to an assistant message.
    let folded = l.req.items.last().unwrap();
    assert!(matches!(folded, Item::Message { role: llm_xlate::ir::Role::Assistant, .. }));
    assert_eq!(kind_of(&l, "items"), Some(DegradationKind::Folded));
}

#[test]
fn provider_result_item_folded_to_user_text() {
    let req = req_with_items(vec![
        user("q"),
        provider_result(ProviderFamily::OpenAI, "web_search", "the answer"),
    ]);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    let folded = l.req.items.last().unwrap();
    match folded {
        Item::Message { role: llm_xlate::ir::Role::User, content, .. } => {
            let txt = content[0].as_text().unwrap();
            assert!(txt.contains("the answer"));
        }
        other => panic!("expected user message, got {other:?}"),
    }
}

#[test]
fn provider_tool_choice_named_drops_to_auto() {
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![provider_tool(ProviderFamily::OpenAI, "web_search")];
    req.tool_choice = ToolChoice::Named("web_search".to_string());
    let mut c = cfg();
    c.foreign_provider_tool = ForeignProviderTool::Drop;
    let l = lower(req, &preset::claude_5(), Protocol::Anthropic, &Resolutions::new(), &c).unwrap();
    assert_eq!(l.req.tool_choice, ToolChoice::Auto);
    assert!(has_field(&l, "tool_choice"));
}

// ── 4. media ─────────────────────────────────────────────────────────────────────────────────

#[test]
fn media_foreign_file_resolved() {
    let req = req_with_items(vec![img_fileref(ProviderFamily::OpenAI, "file_o")]);
    let mut res = Resolutions::new();
    res.files.insert(
        FileRef::new(ProviderFamily::OpenAI, "file_o"),
        FileRef::new(ProviderFamily::Anthropic, "file_a"),
    );
    let l = low_res(req, &preset::claude_5(), Protocol::Anthropic, &res);
    match &l.req.items[0] {
        Item::Message { content, .. } => match &content[0] {
            Part::Image(llm_xlate::ir::MediaSource::FileRef { family, id }) => {
                assert_eq!(*family, ProviderFamily::Anthropic);
                assert_eq!(id, "file_a");
            }
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

#[test]
fn media_foreign_file_unresolved_errors() {
    let req = req_with_items(vec![img_fileref(ProviderFamily::OpenAI, "file_o")]);
    let e = err(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("file_id"));
}

#[test]
fn media_audio_unsupported() {
    // Anthropic audio.sources is empty.
    let req = req_with_items(vec![audio_msg()]);
    let e = err(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("audio"));
}

#[test]
fn media_image_max_bytes_exceeded() {
    let mut caps = preset::claude_5();
    caps.media.image.max_bytes = Some(10);
    let req = req_with_items(vec![img_bytes(100)]);
    let e = err(req, &caps, Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("image"));
}

#[test]
fn media_image_max_count_exceeded() {
    let mut caps = preset::claude_5();
    caps.media.image.max_count = Some(1);
    let req = req_with_items(vec![img_bytes(4), img_bytes(4)]);
    let e = err(req, &caps, Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("image"));
}

#[test]
fn media_pdf_max_bytes_exceeded() {
    let mut caps = preset::claude_5();
    caps.media.pdf.max_bytes = Some(10);
    let req = req_with_items(vec![pdf_bytes(100)]);
    let e = err(req, &caps, Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("document"));
}

#[test]
fn media_pdf_max_count_exceeded() {
    let mut caps = preset::claude_5();
    caps.media.pdf.max_count = Some(1);
    let req = req_with_items(vec![pdf_bytes(4), pdf_bytes(4)]);
    let e = err(req, &caps, Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("document"));
}

// ── 5. output format ─────────────────────────────────────────────────────────────────────────

#[test]
fn output_json_object_unsupported() {
    // openai_compatible leaves output.format unset (None) → unsupported.
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonObject;
    let e = err(req, &preset::openai_compatible(), Protocol::OaiChat);
    assert_eq!(e.kind, ErrorKind::Unsupported);
}

#[test]
fn output_json_schema_on_json_object_only_backend_unsupported() {
    let mut caps = preset::gpt4o();
    caps.output.format = Some(OutputFormatCap::JsonObject);
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonSchema {
        name: "s".to_string(),
        schema: json!({"type":"object"}),
        strict: false,
        description: None,
    };
    let e = err(req, &caps, Protocol::OaiChat);
    assert_eq!(e.kind, ErrorKind::Unsupported);
}

#[test]
fn output_schema_keywords_removed() {
    // Legacy Claude (3.5) still pins schema_unsupported_keywords incl. "pattern"; current models
    // (claude_5) accept these keywords (live-verified 2026-09-10) so their blocklist is now empty.
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonSchema {
        name: "s".to_string(),
        schema: json!({"type":"string","pattern":"^x$"}),
        strict: false,
        description: None,
    };
    let l = low(req, &preset::claude_old(), Protocol::Anthropic);
    if let OutputFormat::JsonSchema { schema, .. } = &l.req.output.format {
        assert!(schema.get("pattern").is_none());
    } else {
        panic!("expected json schema");
    }
    assert_eq!(kind_of(&l, "output.schema"), Some(DegradationKind::Rewritten));
}

#[test]
fn output_schema_keywords_retained_on_current_models() {
    // Live finding (2026-09-10): current Claude accepts minLength/maxLength/pattern/format, so
    // lowering to a 5-family backend must NOT strip them (no rewrite degradation).
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonSchema {
        name: "s".to_string(),
        schema: json!({"type":"string","pattern":"^x$","minLength":1}),
        strict: false,
        description: None,
    };
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    if let OutputFormat::JsonSchema { schema, .. } = &l.req.output.format {
        assert_eq!(schema.get("pattern").and_then(|v| v.as_str()), Some("^x$"));
        assert!(schema.get("minLength").is_some());
    } else {
        panic!("expected json schema");
    }
    assert_eq!(kind_of(&l, "output.schema"), None);
}

#[test]
fn output_strict_downgraded() {
    let mut caps = preset::gpt5_responses();
    caps.output.strict_supported = Tri::No;
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonSchema {
        name: "s".to_string(),
        schema: json!({"type":"object"}),
        strict: true,
        description: None,
    };
    let l = low(req, &caps, Protocol::OaiResponses);
    if let OutputFormat::JsonSchema { strict, .. } = &l.req.output.format {
        assert!(!strict);
    } else {
        panic!();
    }
    assert_eq!(kind_of(&l, "output.strict"), Some(DegradationKind::Downgraded));
}

#[test]
fn output_strict_injects_additional_properties_false() {
    // Live regression (translate1 structured_strict anthropic->chat / ->responses, 2026-09-10):
    // a strict json_schema whose object nodes omit `additionalProperties` was emitted verbatim and
    // OpenAI rejected it ("'additionalProperties' is required to be supplied and to be false").
    // When strict stays true the lowering pass must inject it recursively.
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonSchema {
        name: "response".to_string(),
        schema: json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}),
        strict: true,
        description: None,
    };
    let l = low(req, &preset::gpt4o(), Protocol::OaiChat);
    if let OutputFormat::JsonSchema { schema, strict, .. } = &l.req.output.format {
        assert!(*strict, "strict must stay true on a strict-supporting backend");
        assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
    } else {
        panic!("expected json schema");
    }
    assert_eq!(kind_of(&l, "output.schema"), Some(DegradationKind::Rewritten));
}

#[test]
fn output_strict_downgraded_when_required_incomplete() {
    // OpenAI strict requires every property in `required`. A strict json_schema with an optional
    // property is downgraded to non-strict (rather than promoting the property to required or
    // emitting an invalid strict schema); no additionalProperties is injected once strict is off.
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonSchema {
        name: "response".to_string(),
        schema: json!({"type":"object","properties":{"answer":{"type":"string"},"note":{"type":"string"}},"required":["answer"]}),
        strict: true,
        description: None,
    };
    let l = low(req, &preset::gpt4o(), Protocol::OaiChat);
    if let OutputFormat::JsonSchema { schema, strict, .. } = &l.req.output.format {
        assert!(!strict, "strict must be downgraded when required is incomplete");
        assert!(schema.get("additionalProperties").is_none());
    } else {
        panic!("expected json schema");
    }
    assert_eq!(kind_of(&l, "output.strict"), Some(DegradationKind::Downgraded));
}

#[test]
fn output_format_with_tools_unsupported() {
    let mut caps = preset::gpt5_responses();
    caps.output.format_with_tools = Tri::No;
    let mut req = req_with_items(vec![user("hi")]);
    req.output.format = OutputFormat::JsonObject;
    req.tools = vec![function_tool("f", json!({"type":"object"}), None)];
    let e = err(req, &caps, Protocol::OaiResponses);
    assert_eq!(e.kind, ErrorKind::Unsupported);
}

#[test]
fn output_verbosity_cleared_when_unsupported() {
    // gpt4o verbosity = false.
    let mut req = req_with_items(vec![user("hi")]);
    req.output.verbosity = Some(Verbosity::High);
    let l = low(req, &preset::gpt4o(), Protocol::OaiChat);
    assert_eq!(l.req.output.verbosity, None);
    assert!(has_field(&l, "verbosity"));
}

#[test]
fn output_verbosity_kept_when_supported() {
    // gpt5_responses verbosity = true.
    let mut req = req_with_items(vec![user("hi")]);
    req.output.verbosity = Some(Verbosity::Low);
    let l = low(req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert_eq!(l.req.output.verbosity, Some(Verbosity::Low));
}

// ── 6. tools ─────────────────────────────────────────────────────────────────────────────────

#[test]
fn tools_function_unsupported() {
    // openai_compatible leaves function_tools unset (Unknown → unsupported).
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![function_tool("f", json!({"type":"object"}), None)];
    let e = err(req, &preset::openai_compatible(), Protocol::OaiChat);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("tools"));
}

#[test]
fn tools_max_tools_exceeded() {
    let mut caps = preset::claude_5();
    caps.tools.max_tools = Some(1);
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![
        function_tool("a", json!({"type":"object"}), None),
        function_tool("b", json!({"type":"object"}), None),
    ];
    let e = err(req, &caps, Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::Unsupported);
}

#[test]
fn tools_schema_keywords_removed() {
    // claude_5's tools.schema_unsupported_keywords is empty by default; assert the removal path.
    let mut caps = preset::claude_5();
    caps.tools.schema_unsupported_keywords = Some(vec!["pattern".to_string()]);
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![function_tool(
        "f",
        json!({"type":"object","properties":{"x":{"type":"string","pattern":"^a$"}}}),
        None,
    )];
    let l = low(req, &caps, Protocol::Anthropic);
    if let ToolDef::Function { parameters, .. } = &l.req.tools[0] {
        assert!(parameters["properties"]["x"].get("pattern").is_none());
    }
    assert_eq!(kind_of(&l, "tools.f.parameters"), Some(DegradationKind::Rewritten));
}

#[test]
fn tools_strict_downgraded() {
    let mut caps = preset::claude_5();
    caps.tools.strict.supported = Tri::No;
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![function_tool("f", json!({"type":"object"}), Some(true))];
    let l = low(req, &caps, Protocol::Anthropic);
    if let ToolDef::Function { strict, .. } = &l.req.tools[0] {
        assert_eq!(*strict, Some(false));
    }
    assert_eq!(kind_of(&l, "tools.f.strict"), Some(DegradationKind::Downgraded));
}

#[test]
fn tools_strict_injects_additional_properties_false() {
    // Live regression (translate1 full responses->chat, 2026-09-10): a strict function whose
    // parameters object omits `additionalProperties` was emitted verbatim and OpenAI rejected it
    // ("Invalid schema for function 'get_weather': 'additionalProperties' is required ..."). When
    // strict stays true the lowering pass must inject it recursively into the parameters schema.
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![function_tool(
        "get_weather",
        json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
        Some(true),
    )];
    let l = low(req, &preset::gpt4o(), Protocol::OaiChat);
    if let ToolDef::Function { parameters, strict, .. } = &l.req.tools[0] {
        assert_eq!(*strict, Some(true), "strict must stay true on a strict-supporting backend");
        assert_eq!(parameters.get("additionalProperties"), Some(&json!(false)));
    } else {
        panic!("expected function tool");
    }
    assert_eq!(kind_of(&l, "tools.get_weather.parameters"), Some(DegradationKind::Rewritten));
}

#[test]
fn tools_strict_downgraded_when_required_incomplete() {
    // Live regression (translate1 `full` responses->chat, 2026-09-10): a strict function schema
    // whose `properties` are not all listed in `required` is invalid to OpenAI strict even after
    // `additionalProperties:false` is injected ("'required' must contain every key in
    // 'properties'"). Rather than silently promote `city` to required (a semantic change), strict is
    // downgraded to false and no additionalProperties is injected — the non-strict schema is valid.
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![function_tool(
        "get_weather",
        json!({"type":"object","properties":{"city":{"type":"string"}}}),
        Some(true),
    )];
    let l = low(req, &preset::gpt4o(), Protocol::OaiChat);
    if let ToolDef::Function { parameters, strict, .. } = &l.req.tools[0] {
        assert_eq!(*strict, Some(false), "strict must be downgraded when required is incomplete");
        assert!(
            parameters.get("additionalProperties").is_none(),
            "no additionalProperties injected once strict is off"
        );
    } else {
        panic!("expected function tool");
    }
    assert_eq!(kind_of(&l, "tools.get_weather.strict"), Some(DegradationKind::Downgraded));
}

#[test]
fn tools_tool_choice_variant_degraded_to_auto() {
    let mut caps = preset::claude_5();
    caps.tools.tool_choice = Some(vec![ToolChoiceKind::Auto]);
    let mut req = req_with_items(vec![user("hi")]);
    req.tools = vec![function_tool("f", json!({"type":"object"}), None)];
    req.tool_choice = ToolChoice::Required;
    let l = low(req, &caps, Protocol::Anthropic);
    assert_eq!(l.req.tool_choice, ToolChoice::Auto);
    assert!(has_field(&l, "tool_choice"));
}

#[test]
fn tools_parallel_cleared_when_uncontrolled() {
    let mut caps = preset::claude_5();
    caps.tools.parallel_control = Tri::No;
    let mut req = req_with_items(vec![user("hi")]);
    req.parallel_tool_calls = Some(true);
    let l = low(req, &caps, Protocol::Anthropic);
    assert_eq!(l.req.parallel_tool_calls, None);
    assert!(has_field(&l, "parallel_tool_calls"));
}

#[test]
fn tools_id_pattern_mismatch_incompatible_history() {
    // claude_5 id_pattern rejects spaces / punctuation.
    let req = req_with_items(vec![tool_call("bad id!", "search"), tool_result("bad id!", "r")]);
    let e = err(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(e.kind, ErrorKind::IncompatibleHistory);
}

#[test]
fn tools_id_pattern_ok() {
    let req = req_with_items(vec![tool_call("call_abc-1", "search"), tool_result("call_abc-1", "r")]);
    // supply resolution so required reasoning does not error first
    let mut res = Resolutions::new();
    res.reasoning.insert(CallId::new("call_abc-1"), blob(ProviderFamily::Anthropic));
    let l = low_res(req, &preset::claude_5(), Protocol::Anthropic, &res);
    assert!(l.req.items.iter().any(|i| matches!(i, Item::ToolCall { .. })));
}

#[test]
fn tools_id_pattern_invalid_regex_is_invalid_request() {
    // A backend that ships a malformed id_pattern is a request-construction error, surfaced as
    // InvalidRequest — the only InvalidRequest-producing path in lower(). gpt4o has no
    // required-reasoning turn, so the tools pass (f) is reached.
    let mut caps = preset::gpt4o();
    caps.tools.id_pattern = Some("(".into());
    let req = req_with_items(vec![tool_call("call_1", "search")]);
    let e = err(req, &caps, Protocol::OaiChat);
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
}

#[test]
fn tools_id_pattern_anchored_rejects_substring_match() {
    // An unanchored pattern would accept an id merely *containing* a conforming substring; the
    // whole id must match.
    let mut caps = preset::gpt4o();
    caps.tools.id_pattern = Some("[a-zA-Z0-9_-]+".into());
    let req = req_with_items(vec![tool_call("bad id call_1 !!!", "search")]);
    let e = err(req, &caps, Protocol::OaiChat);
    assert_eq!(e.kind, ErrorKind::IncompatibleHistory);
}

// ── 7. sampling ──────────────────────────────────────────────────────────────────────────────

#[test]
fn sampling_n_gt_1_unsupported() {
    let mut req = req_with_items(vec![user("hi")]);
    req.sampling.n = Some(2);
    let e = err(req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("n"));
}

#[test]
fn sampling_stop_sequences_truncated() {
    // claude_5 stop_sequences.max = 4.
    let mut req = req_with_items(vec![user("hi")]);
    req.limits.stop_sequences =
        vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into(), "f".into()];
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.limits.stop_sequences.len(), 4);
    assert!(has_field(&l, "stop_sequences"));
}

#[test]
fn max_output_tokens_dropped_when_backend_rejects_it() {
    // A Codex-style backend overlay: the limit is refused upstream, so lowering drops it
    // with a Dropped degradation instead of letting the request 400.
    let mut caps = preset::gpt5_responses();
    caps.transport.max_output_tokens_rejected = Tri::Yes;
    let mut req = req_with_items(vec![user("hi")]);
    req.limits.max_output_tokens = Some(256);
    let l = low(req, &caps, Protocol::OaiResponses);
    assert_eq!(l.req.limits.max_output_tokens, None);
    assert_eq!(kind_of(&l, "max_output_tokens"), Some(DegradationKind::Dropped));
}

#[test]
fn max_output_tokens_forwarded_unless_explicitly_rejected() {
    // Unknown (the default) and No both forward the limit untouched, with no degradation.
    for tri in [Tri::Unknown, Tri::No] {
        let mut caps = preset::gpt5_responses();
        caps.transport.max_output_tokens_rejected = tri;
        let mut req = req_with_items(vec![user("hi")]);
        req.limits.max_output_tokens = Some(256);
        let l = low(req, &caps, Protocol::OaiResponses);
        assert_eq!(l.req.limits.max_output_tokens, Some(256));
        assert!(!has_field(&l, "max_output_tokens"));
    }
}

#[test]
fn sampling_service_tier_cleared_when_unknown() {
    // claude_5 service_tier = ["auto","standard_only"].
    let mut req = req_with_items(vec![user("hi")]);
    req.meta.service_tier = Some("premium".to_string());
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.meta.service_tier, None);
    assert!(has_field(&l, "service_tier"));
}

#[test]
fn sampling_service_tier_kept_when_known() {
    let mut req = req_with_items(vec![user("hi")]);
    req.meta.service_tier = Some("auto".to_string());
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.meta.service_tier, Some("auto".to_string()));
}

// ── 8. instructions ──────────────────────────────────────────────────────────────────────────

#[test]
fn instructions_mid_fail_when_not_native_anthropic() {
    // claude_old: mid_conversation_system = none.
    let mut req = req_with_items(vec![user("a"), user("b")]);
    req.instructions = vec![mid_instruction(1, "mid")];
    let mut c = cfg();
    c.mid_instruction_fallback = MidInstructionFallback::Fail;
    let e = lower(req, &preset::claude_old(), Protocol::Anthropic, &Resolutions::new(), &c)
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert_eq!(e.param.as_deref(), Some("messages"));
}

#[test]
fn instructions_mid_ok_when_native_anthropic() {
    // claude_5: mid_conversation_system = native.
    let mut req = req_with_items(vec![user("a"), user("b")]);
    req.instructions = vec![mid_instruction(1, "mid")];
    let mut c = cfg();
    c.mid_instruction_fallback = MidInstructionFallback::Fail;
    let l = lower(req, &preset::claude_5(), Protocol::Anthropic, &Resolutions::new(), &c).unwrap();
    assert_eq!(l.req.instructions.len(), 1);
}

#[test]
fn instructions_mid_ok_for_chat_always_native() {
    let mut req = req_with_items(vec![user("a"), user("b")]);
    req.instructions = vec![mid_instruction(1, "mid")];
    let mut c = cfg();
    c.mid_instruction_fallback = MidInstructionFallback::Fail;
    let l = lower(req, &preset::gpt4o(), Protocol::OaiChat, &Resolutions::new(), &c).unwrap();
    assert_eq!(l.req.instructions.len(), 1);
}

#[test]
fn instructions_effort_cleared_when_unsupported() {
    // claude_old system_effort_override = false.
    let mut req = req_with_items(vec![user("a")]);
    let mut ins = llm_xlate::ir::Instruction::system_text("sys");
    ins.effort = Some(Effort::High);
    req.instructions = vec![ins];
    let l = low(req, &preset::claude_old(), Protocol::Anthropic);
    assert_eq!(l.req.instructions[0].effort, None);
    assert!(has_field(&l, "instructions.effort"));
}

#[test]
fn instructions_clear_at_cleared_when_unsupported() {
    // claude_old system_clear_at = false.
    let mut req = req_with_items(vec![user("a")]);
    let mut ins = llm_xlate::ir::Instruction::system_text("sys");
    ins.clear_at = Some(llm_xlate::ir::ClearAt::from(json!({"turns": 2})));
    req.instructions = vec![ins];
    let l = low(req, &preset::claude_old(), Protocol::Anthropic);
    assert_eq!(l.req.instructions[0].clear_at, None);
    assert!(has_field(&l, "instructions.clear_at"));
}

// ── 9. state ─────────────────────────────────────────────────────────────────────────────────

#[test]
fn state_store_cleared_when_unsupported() {
    // claude_5 state.store = false.
    let mut req = req_with_items(vec![user("hi")]);
    req.state.store = Some(true);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.state.store, None);
    assert!(has_field(&l, "store"));
}

#[test]
fn state_zdr_forces_store_false() {
    let mut caps = preset::gpt5_responses();
    caps.state.zdr = Tri::Yes;
    let mut req = req_with_items(vec![user("hi")]);
    req.state.store = Some(true);
    let l = low(req, &caps, Protocol::OaiResponses);
    assert_eq!(l.req.state.store, Some(false));
    assert!(has_field(&l, "store"));
}

#[test]
fn state_previous_response_id_cleared() {
    // claude_5 previous_response_id = false.
    let mut req = req_with_items(vec![user("hi")]);
    req.state.previous_response_id = Some(llm_xlate::ir::ResponseId::new("r"));
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.state.previous_response_id, None);
    assert!(has_field(&l, "previous_response_id"));
}

#[test]
fn state_conversation_cleared_when_unsupported() {
    // claude_5 does not support the Responses conversation API.
    let mut req = req_with_items(vec![user("hi")]);
    req.state.conversation = Some("conv_1".to_string());
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.state.conversation, None);
    assert!(has_field(&l, "conversation"));
}

#[test]
fn state_background_cleared_when_unsupported() {
    // claude_5 does not support background mode.
    let mut req = req_with_items(vec![user("hi")]);
    req.state.background = Some(true);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(l.req.state.background, None);
    assert!(has_field(&l, "background"));
}

#[test]
fn state_include_cleared() {
    // claude_5 encrypted_reasoning_include = false.
    let mut req = req_with_items(vec![user("hi")]);
    req.state.include = vec!["reasoning.encrypted_content".to_string()];
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(l.req.state.include.is_empty());
    assert!(has_field(&l, "include"));
}

#[test]
fn state_supported_kept() {
    // gpt5_responses store = true, previous_response_id = true.
    let mut req = req_with_items(vec![user("hi")]);
    req.state.store = Some(true);
    let l = low(req, &preset::gpt5_responses(), Protocol::OaiResponses);
    assert_eq!(l.req.state.store, Some(true));
    assert!(!has_field(&l, "store"));
}

// ── 10. ext ──────────────────────────────────────────────────────────────────────────────────

#[test]
fn ext_foreign_namespace_reported_once() {
    let mut req = req_with_items(vec![user("hi")]);
    req.ext.insert("chat.logit_bias_extra", json!(1));
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(has_field(&l, "ext.chat.logit_bias_extra"));
    // The key stays in the IR (codecs ignore foreign namespaces).
    assert!(l.req.ext.get("chat.logit_bias_extra").is_some());
}

#[test]
fn ext_same_namespace_untouched() {
    let mut req = req_with_items(vec![user("hi")]);
    req.ext.insert("anthropic.thinking_extra", json!(1));
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    assert!(!has_field(&l, "ext.anthropic.thinking_extra"));
}

// ── determinism ──────────────────────────────────────────────────────────────────────────────

#[test]
fn lower_is_deterministic() {
    let build = || {
        let mut req = req_with_items(vec![
            user("q"),
            provider_call(ProviderFamily::OpenAI, "web_search"),
            reasoning_opaque(ProviderFamily::OpenAI),
        ]);
        req.reasoning.effort = Some(Effort::Max);
        req.tools = vec![function_tool(
            "f",
            json!({"type":"object","properties":{"x":{"type":"string","pattern":"^a$"}}}),
            Some(true),
        )];
        req.ext.insert("chat.x", json!(1));
        req.limits.stop_sequences = vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()];
        req
    };
    let a = low(build(), &preset::claude_5(), Protocol::Anthropic);
    let b = low(build(), &preset::claude_5(), Protocol::Anthropic);
    assert_eq!(a.req, b.req);
    assert_eq!(a.degradations.render_header_value(), b.degradations.render_header_value());
}

#[test]
fn lower_does_not_touch_unrelated_fields() {
    let mut req = req_with_items(vec![user("hello")]);
    req.sampling.temperature = Some(0.7);
    req.sampling.top_p = Some(0.9);
    let l = low(req, &preset::claude_5(), Protocol::Anthropic);
    // Per-field sampling policy is the codec's job; lower leaves temperature/top_p untouched.
    assert_eq!(l.req.sampling.temperature, Some(0.7));
    assert_eq!(l.req.sampling.top_p, Some(0.9));
}
