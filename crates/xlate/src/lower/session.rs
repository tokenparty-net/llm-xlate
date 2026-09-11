//! Session-affinity lowering (plan: session mapping).
//!
//! Capture (which slot the session id came from) happens in each codec's `decode_request` via
//! [`llm_xlate_core::session::capture_session`]; *emission* (which slot it goes to) happens in
//! each codec's `encode_request` via [`llm_xlate_core::session::apply_session`]. This pass owns
//! the two protocol-agnostic policies in between:
//!
//! 1. **Conflict reporting.** If the client supplied a session id in more than one place with
//!    differing values, the decoder recorded the losers in
//!    [`SessionConfig::conflicting_sources`](llm_xlate_core::ir::SessionConfig::conflicting_sources);
//!    we surface a single `Dropped` degradation for them (the highest-priority value is used).
//! 2. **Requirement enforcement.** If the backend's [`SessionCap::required`] is `Yes` and no
//!    session id was captured, the request is rejected — the backend cannot function without
//!    one.

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::IrRequest;

/// Run the session lowering pass.
pub fn run(
    req: &mut IrRequest,
    caps: &Capabilities,
    degradations: &mut Degradations,
) -> Result<(), XlateError> {
    if !req.session.conflicting_sources.is_empty() {
        degradations.dropped(
            "session_id",
            format!(
                "client supplied differing session ids in {}; used the highest-priority source",
                req.session.conflicting_sources.join(", ")
            ),
        );
    }

    if caps.session.required.is_yes() && req.session.id.is_none() {
        return Err(XlateError::invalid_request(
            "backend requires a session id, but the request supplied none",
        ));
    }

    Ok(())
}
