//! Pass 1: the target protocol must be allowed by `caps.transport.protocols` (plan §7).

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, Protocol};

/// Reject the whole translation if the backend does not speak `target`.
pub(crate) fn run(
    _req: &mut IrRequest,
    caps: &Capabilities,
    target: Protocol,
    _degr: &mut Degradations,
) -> Result<(), XlateError> {
    if !caps.protocol_allowed(target) {
        return Err(XlateError::unsupported(
            "protocol",
            format!("backend does not support the {target:?} wire protocol"),
        ));
    }
    Ok(())
}
