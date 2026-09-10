use super::*;
use crate::caps::schema::{
    BudgetStatus, MidConversationSystem, ReasoningMode, SamplingRule, Streaming, TopLevelSystem,
    TransportCap,
};
use crate::ir::{Effort, Protocol, ProviderFamily};
use pretty_assertions::assert_eq;

#[test]
fn each_shipped_file_parses() {
    assert!(Registry::from_toml(include_str!("../../../../data/caps/anthropic.toml")).is_ok());
    assert!(Registry::from_toml(include_str!("../../../../data/caps/openai.toml")).is_ok());
    assert!(
        Registry::from_toml(include_str!("../../../../data/caps/openai_compatible.toml")).is_ok()
    );
}

#[test]
fn shipped_registry_builds() {
    // Panics inside shipped() would fail here.
    let _ = shipped();
}

#[test]
fn tri_helpers() {
    assert!(Tri::Yes.is_yes());
    assert!(!Tri::No.is_yes());
    assert!(Tri::No.is_no_or_unknown());
    assert!(Tri::Unknown.is_no_or_unknown());
    assert!(Tri::Unknown.is_unknown());
    assert!(Tri::No.is_no());
}

#[test]
fn sampling_rule_accepts() {
    let r = SamplingRule::Range([0.0, 2.0]);
    assert!(r.accepts(1.0));
    assert!(!r.accepts(2.1));
    assert!(!SamplingRule::Rejected.accepts(0.5));
    assert!(!SamplingRule::Ignored.accepts(0.5));
}

#[test]
fn unknown_is_conservative() {
    let c = Capabilities::unknown();
    assert!(!c.protocol_allowed(Protocol::Anthropic));
    assert!(c.tools.function_tools.is_unknown());
    assert!(!c.effort_supported(Effort::Low));
    assert!(c.effort_supported(Effort::None)); // disabling is always fine
}

#[test]
fn preset_claude_old_has_no_thinking_mode() {
    // Claude 3.5 (claude_old) predates extended thinking (introduced in Claude 3.7): its
    // reasoning mode is `None`, so a reasoning request is dropped rather than emitting a
    // `thinking` block the real API rejects. Sampling, prefill, and system handling are intact.
    let c = preset::claude_old();
    assert!(c.protocol_allowed(Protocol::Anthropic));
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::None));
    assert_eq!(c.instructions.mid_conversation_system, Some(MidConversationSystem::None));
    assert_eq!(c.instructions.top_level_system, Some(TopLevelSystem::TextBlocks));
    assert!(matches!(c.sampling.temperature, Some(SamplingRule::Range(_))));
    assert_eq!(c.output.prefill_allowed, Tri::Yes);
    assert!(!c.effort_supported(Effort::Low)); // no thinking at all
    // 3.5 raised max output to 8192 (vs 4096 on the original Claude 3 GA models).
    assert_eq!(c.limits.max_output_tokens, Some(8192));
}

#[test]
fn preset_claude_3_ga_caps_output_at_4096_no_thinking() {
    // The original Claude 3 GA models cap output at 4096 and have no thinking mode.
    let c = shipped().resolve(&ProviderFamily::Anthropic, "claude-3-opus-20240229", None);
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::None));
    assert_eq!(c.limits.max_output_tokens, Some(4096));
    assert_eq!(c.transport.default_max_output_tokens, Some(4096));
}

#[test]
fn preset_claude_46_adaptive_budget_ignored() {
    let c = preset::claude_46();
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::Adaptive));
    let budget = c.reasoning.budget.expect("budget rule");
    assert_eq!(budget.status, Some(BudgetStatus::Ignored));
    assert!(c.effort_supported(Effort::Max));
    assert_eq!(c.instructions.mid_conversation_system, Some(MidConversationSystem::None));
    // sampling still accepted on 4.6
    assert!(matches!(c.sampling.temperature, Some(SamplingRule::Range(_))));
}

#[test]
fn preset_claude_5_adaptive_sampling_rejected_native_system() {
    let c = preset::claude_5();
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::Adaptive));
    assert_eq!(c.sampling.temperature, Some(SamplingRule::Rejected));
    assert_eq!(c.sampling.top_p, Some(SamplingRule::Rejected));
    assert_eq!(c.instructions.mid_conversation_system, Some(MidConversationSystem::Native));
    assert!(c.instructions.system_clear_at.is_yes());
    assert!(c.effort_supported(Effort::Max));
    assert!(c.effort_supported(Effort::Low));
}

#[test]
fn preset_gpt4o_no_reasoning_both_protocols() {
    let c = preset::gpt4o();
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::None));
    assert!(c.protocol_allowed(Protocol::OaiChat));
    assert!(c.protocol_allowed(Protocol::OaiResponses));
    assert_eq!(c.output.verbosity, Tri::No);
    assert!(!c.effort_supported(Effort::Medium));
}

#[test]
fn preset_gpt5_responses_reasoning_with_tools() {
    let c = preset::gpt5_responses();
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::EffortOnly));
    assert_eq!(c.reasoning.tools_with_reasoning, Tri::Yes);
    assert_eq!(c.output.verbosity, Tri::Yes);
    assert_eq!(c.state.store, Tri::Yes);
    assert!(c.protocol_allowed(Protocol::OaiChat));
    assert!(c.protocol_allowed(Protocol::OaiResponses));
    assert!(c.effort_supported(Effort::Minimal));
}

#[test]
fn preset_gpt5_chat_tools_without_reasoning() {
    let c = preset::gpt5_chat();
    // Backend overlay: Chat-only, and tools do not work with reasoning from 5.4.
    assert_eq!(c.reasoning.tools_with_reasoning, Tri::No);
    assert!(c.protocol_allowed(Protocol::OaiChat));
    assert!(!c.protocol_allowed(Protocol::OaiResponses));
}

#[test]
fn preset_gpt6_responses_only() {
    let c = preset::gpt6();
    assert!(!c.protocol_allowed(Protocol::OaiChat));
    assert!(c.protocol_allowed(Protocol::OaiResponses));
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::EffortOnly));
}

#[test]
fn preset_openai_compatible_conservative() {
    let c = preset::openai_compatible();
    assert!(c.protocol_allowed(Protocol::OaiChat));
    assert!(!c.protocol_allowed(Protocol::OaiResponses));
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::None));
    assert!(c.tools.strict.supported.is_no());
    assert!(c.state.store.is_no());
    // Not specified => Unknown / conservative.
    assert!(c.cache.prompt_cache_key.is_no());
    // The one safe positive: a leading `system` string (the Chat format guarantees it).
    assert_eq!(c.instructions.top_level_system, Some(TopLevelSystem::Plain));
    // No optimistic assertions for an arbitrary server (plan §6 "Unknown for most"):
    // mid-conversation system is unset (=> inline-wrap fallback, never native mid-array
    // role:"system"), and tools + structured output are Unknown until a backend opts in.
    assert_eq!(c.instructions.mid_conversation_system, None);
    assert!(c.tools.function_tools.is_unknown());
    assert!(c.tools.parallel_control.is_unknown());
    assert!(c.tools.tool_choice.is_none());
    assert!(c.output.format.is_none());
}

#[test]
fn backend_override_precedence() {
    let base = preset::claude_5();
    assert_eq!(base.streaming(), Streaming::Both);

    let overlay = Capabilities {
        transport: TransportCap { streaming: Some(Streaming::NonStreamOnly), ..Default::default() },
        instructions: crate::caps::schema::InstructionsCap {
            mid_conversation_system: Some(MidConversationSystem::InlineWrapOnly),
            ..Default::default()
        },
        ..Default::default()
    };
    let resolved =
        shipped().resolve(&ProviderFamily::Anthropic, "claude-opus-5", Some(&overlay));
    assert_eq!(resolved.streaming(), Streaming::NonStreamOnly);
    assert_eq!(
        resolved.instructions.mid_conversation_system,
        Some(MidConversationSystem::InlineWrapOnly)
    );
    // Untouched fields survive from the model layer.
    assert_eq!(resolved.sampling.temperature, Some(SamplingRule::Rejected));
}

#[test]
fn merge_keeps_first_registry_model_rules_ahead() {
    let mut a = Registry::from_toml(
        "family = \"anthropic\"\n[[model]]\nmatch = \"claude-x\"\n[model.reasoning]\nmode = \"adaptive\"\n",
    )
    .unwrap();
    let b = Registry::from_toml(
        "family = \"anthropic\"\n[[model]]\nmatch = \"claude-x\"\n[model.reasoning]\nmode = \"budget\"\n",
    )
    .unwrap();
    a.merge(b);
    let c = a.resolve(&ProviderFamily::Anthropic, "claude-x", None);
    // a's rule wins the tie.
    assert_eq!(c.reasoning.mode, Some(ReasoningMode::Adaptive));
}
