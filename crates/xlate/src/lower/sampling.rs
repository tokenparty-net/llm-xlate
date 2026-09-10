//! Pass 7: sampling (plan §7.6).
//!
//! Only the *cross-cutting* sampling policy lives here: `n>1` is unsupported, stop sequences
//! are capped to the backend's maximum, and an unsupported `service_tier` is cleared. Per-field
//! temperature/top_p/… range policy is the codec's job (it emits or drops each field per its
//! wire rule) — those values are left untouched here.

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::IrRequest;

pub(crate) fn run(
    req: &mut IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    // (a) n > 1 is unsupported (plan §14.4).
    if let Some(n) = req.sampling.n {
        if n > 1 {
            return Err(XlateError::unsupported(
                "n",
                "multiple completions (n>1) are not supported",
            ));
        }
    }

    // (b) Stop-sequence count cap.
    if let Some(rule) = &caps.sampling.stop_sequences {
        if let Some(max) = rule.max {
            if req.limits.stop_sequences.len() as u32 > max {
                req.limits.stop_sequences.truncate(max as usize);
                degr.dropped(
                    "stop_sequences",
                    format!("truncated to the backend maximum of {max} stop sequences"),
                );
            }
        }
    }

    // (c) Service tier.
    if let Some(tier) = &req.meta.service_tier {
        let ok = caps.sampling.service_tier.as_ref().is_some_and(|l| l.iter().any(|t| t == tier));
        if !ok {
            req.meta.service_tier = None;
            degr.dropped("service_tier", "requested service tier not supported; cleared");
        }
    }

    Ok(())
}
