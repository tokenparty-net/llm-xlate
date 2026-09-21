//! Pass 8: instructions (plan §7.1).
//!
//! `lower` does **not** place instructions — that is the codec's wire-shape job (top-level
//! system, developer→system fallback, inline-wrap). It only enforces the mid-conversation
//! *policy*: when the config says `Fail` and the target cannot place a mid-context instruction
//! natively, reject the request instead of letting the codec inline-wrap it. It also strips a
//! mid-conversation `effort` / `clear_at` override the backend does not support, so codecs
//! never see them, and clamps an anchor that points past the end of the transcript so no
//! instruction can be silently lost on the wire.

use llm_xlate_core::caps::{Capabilities, MidConversationSystem};
use llm_xlate_core::codec::{MidInstructionFallback, TranslatorConfig};
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, Position, Protocol};

pub(crate) fn run(
    req: &mut IrRequest,
    caps: &Capabilities,
    target: Protocol,
    cfg: &TranslatorConfig,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    let has_mid = req.instructions.iter().any(|i| matches!(i.position, Position::Before(_)));

    if has_mid && cfg.mid_instruction_fallback == MidInstructionFallback::Fail {
        // Chat/Responses always place mid-conversation system/developer messages natively; for
        // Anthropic it depends on the model.
        let native = match target {
            Protocol::OaiChat | Protocol::OaiResponses => true,
            Protocol::Anthropic => {
                caps.instructions.mid_conversation_system == Some(MidConversationSystem::Native)
            }
        };
        if !native {
            return Err(XlateError::unsupported(
                "messages",
                "the backend cannot place a mid-conversation instruction natively and the \
                 configuration forbids the inline-wrap fallback",
            ));
        }
    }

    // Clamp an anchor that points past the last item to the trailing slot.
    //
    // Only a malformed or mis-rewritten request can produce one (every decoder anchors at the
    // item count it has emitted so far, and the lowering passes re-map anchors when they add or
    // remove items). It is reported rather than tolerated silently because two of the three
    // codecs have no trailing bucket for an out-of-range index and would drop the instruction
    // altogether — losing prompt text with no degradation, which §7.1 forbids.
    let len = req.items.len();
    for ins in &mut req.instructions {
        if let Position::Before(i) = ins.position {
            if i > len {
                ins.position = Position::Before(len);
                degr.rewritten(
                    "instructions.position",
                    format!("anchored at item {i} past the {len}-item transcript; moved to the end"),
                );
            }
        }
    }

    // Strip unsupported per-instruction overrides so codecs never see them.
    for ins in &mut req.instructions {
        if ins.effort.is_some() && caps.instructions.system_effort_override.is_no_or_unknown() {
            ins.effort = None;
            degr.dropped(
                "instructions.effort",
                "mid-conversation effort override not supported; cleared",
            );
        }
        if ins.clear_at.is_some() && caps.instructions.system_clear_at.is_no_or_unknown() {
            ins.clear_at = None;
            degr.dropped("instructions.clear_at", "system clear_at not supported; cleared");
        }
    }

    Ok(())
}
