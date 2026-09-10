//! Pass 9: server state (plan §9).
//!
//! Clears `store` / `previous_response_id` / `conversation` / `background` / `include` entries
//! the backend does not support, and honors a ZDR backend by forcing `store = false`.

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::IrRequest;

pub(crate) fn run(
    req: &mut IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    // store — ZDR wins (forces false); otherwise clear when unsupported.
    if caps.state.zdr.is_yes() {
        if req.state.store == Some(true) {
            degr.dropped("store", "zero-data-retention backend forces store=false");
        }
        req.state.store = Some(false);
    } else if req.state.store.is_some() && caps.state.store.is_no_or_unknown() {
        req.state.store = None;
        degr.dropped("store", "store not supported by the backend; cleared");
    }

    if req.state.previous_response_id.is_some()
        && caps.state.previous_response_id.is_no_or_unknown()
    {
        req.state.previous_response_id = None;
        degr.dropped("previous_response_id", "response chaining not supported; cleared");
    }

    if req.state.conversation.is_some() && caps.state.conversation_api.is_no_or_unknown() {
        req.state.conversation = None;
        degr.dropped("conversation", "conversation API not supported; cleared");
    }

    if req.state.background.is_some() && caps.state.background.is_no_or_unknown() {
        req.state.background = None;
        degr.dropped("background", "background mode not supported; cleared");
    }

    if !req.state.include.is_empty() && caps.state.encrypted_reasoning_include.is_no_or_unknown() {
        req.state.include.clear();
        degr.dropped("include", "include entries not supported; cleared");
    }

    Ok(())
}
