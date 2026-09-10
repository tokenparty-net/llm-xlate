//! Pass 10: extensions (plan §7).
//!
//! `ext` keys are namespaced by protocol (`chat.*`, `responses.*`, `anthropic.*`). A key whose
//! namespace belongs to a *different* protocol than the target is reported **once** as a
//! dropped degradation (field `ext.<key>`). The keys stay in the IR — the target codec simply
//! ignores foreign namespaces — but reporting them here (rather than in every codec) keeps the
//! degradation record single-sourced.

use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, Protocol};

const PROTOCOL_NAMESPACES: [&str; 3] = ["chat", "responses", "anthropic"];

/// Codec-internal serialization / round-trip markers that are **not** client-facing semantics: a
/// target either consumes them or maps them 1:1 onto its own native field, so reporting them as
/// lossy `Dropped` degradations on cross-protocol routing is a false positive that pollutes the
/// `x-router-degraded` signal (plan §11.7 is about genuinely lossy steps).
///
/// - `chat.max_tokens_field` — records which max-output alias the client spelled
///   (`max_tokens` vs `max_completion_tokens`); the value itself is carried to the target's
///   native max-output field.
/// - `chat.include_usage` — the streaming-usage opt-in; honored by the translator regardless of
///   target.
/// - `chat.legacy_functions` / `chat.legacy_function_call` — record that the client used the
///   legacy `functions` / `function_call` spelling; the tool semantics are carried natively in
///   `req.tools` / `req.tool_choice` and reach the target, so the spelling marker is not a
///   semantic loss (it only affects echo fidelity back to a Chat client).
///
/// Genuine client passthrough side-tables (`chat.image_detail`, `chat.msg_ext`,
/// `chat.instr_ext`, `responses.*`, `anthropic.*` semantic keys) are deliberately **not** listed:
/// a foreign target that cannot express them is a real drop worth reporting.
const INTERNAL_MARKERS: [&str; 4] = [
    "chat.max_tokens_field",
    "chat.include_usage",
    "chat.legacy_functions",
    "chat.legacy_function_call",
];

pub(crate) fn run(
    req: &mut IrRequest,
    target: Protocol,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    let target_ns = match target {
        Protocol::OaiChat => "chat",
        Protocol::OaiResponses => "responses",
        Protocol::Anthropic => "anthropic",
    };

    // BTreeMap iteration is sorted, so the reporting order is deterministic.
    for key in req.ext.keys() {
        if INTERNAL_MARKERS.contains(&key.as_str()) {
            continue;
        }
        let ns = key.split('.').next().unwrap_or("");
        if PROTOCOL_NAMESPACES.contains(&ns) && ns != target_ns {
            degr.dropped(
                format!("ext.{key}"),
                format!("extension belongs to the {ns} protocol, not the {target_ns} target"),
            );
        }
    }

    Ok(())
}
