//! Stored / non-stored Responses support (plan §9).
//!
//! When a client uses the Responses `store: true` + `previous_response_id` chaining feature,
//! the router persists each completed response and, on the next turn, replays the chain into a
//! single stateless request. This module provides the crate-side pure building blocks:
//!
//! * [`StoredResponse`] — the router's persistence record (serialized as JSON by the router),
//!   built from a completed turn with [`to_stored`].
//! * [`materialize_chain`] — fold an ordered chain plus a new request into one stateless
//!   [`IrRequest`], applying the plan §7.1 instruction rule (a request without instructions
//!   does not resurrect prior ones) and shifting mid-context instruction positions.
//! * [`chain_binding`] — the backend binding of the newest stored element, so the router can
//!   pin the credential that produced in-flight encrypted reasoning.
//!
//! No I/O, no clocks: `created_at` is supplied by the caller, ids are router-minted.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{
    Instruction, IrRequest, IrResponse, Item, Position, ProviderFamily, ResponseId, StopReason,
    Usage,
};

/// Which backend produced a stored response, and how to reach its native state (plan §9).
///
/// The `provider_response_id` lets the router use native `previous_response_id` passthrough as
/// an optimization; `credential_id` + `family` let it pin the exact credential that can decrypt
/// in-flight encrypted reasoning bound to this element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendBinding {
    /// The router credential that produced this response.
    pub credential_id: String,
    /// The upstream provider's own response id, if the backend stored one.
    pub provider_response_id: Option<String>,
    /// The provider family that produced it.
    pub family: ProviderFamily,
    /// The backend model string.
    pub model: String,
}

impl BackendBinding {
    /// Construct a binding.
    pub fn new(
        credential_id: impl Into<String>,
        family: ProviderFamily,
        model: impl Into<String>,
    ) -> Self {
        Self {
            credential_id: credential_id.into(),
            provider_response_id: None,
            family,
            model: model.into(),
        }
    }
}

/// The status of a stored response (plan §9): the Responses lifecycle states the router renders
/// for `GET /responses/{id}` and background/cancel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredStatus {
    /// Queued (background, not yet started).
    Queued,
    /// In progress (background, streaming).
    InProgress,
    /// Completed successfully.
    Completed,
    /// Stopped before completion (max tokens / content filter).
    Incomplete,
    /// Failed with an error.
    Failed,
    /// Cancelled.
    Cancelled,
}

impl StoredStatus {
    /// Derive the terminal status of a *successful* turn from its stop reason. A turn that
    /// stopped for [`StopReason::MaxTokens`], [`StopReason::ContentFilter`], or
    /// [`StopReason::PauseTurn`] (an interrupted, non-terminal turn) is `Incomplete`;
    /// [`StopReason::Cancelled`] is `Cancelled`; everything else is `Completed`. (A turn that
    /// errored is [`StoredStatus::Failed`] and is built directly, not via this mapping.)
    pub fn from_stop(stop: &StopReason) -> Self {
        match stop {
            StopReason::MaxTokens | StopReason::ContentFilter | StopReason::PauseTurn => {
                StoredStatus::Incomplete
            }
            StopReason::Cancelled => StoredStatus::Cancelled,
            _ => StoredStatus::Completed,
        }
    }
}

/// A persisted response record (plan §9). The router serializes this as JSON; every field is
/// serde-round-trippable.
///
/// Opaque reasoning blobs in `output_items` are stored **unwrapped** (native, as the backend
/// produced them) — sealing into an `rtr1.` envelope happens only at the client boundary, not
/// in storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredResponse {
    /// Router-minted client-facing id of this response.
    pub id: ResponseId,
    /// The response this one chained from, if any.
    pub previous_id: Option<ResponseId>,
    /// The instructions in force for the turn that produced this response.
    pub instructions: Vec<Instruction>,
    /// The input items of the turn (the request transcript).
    pub request_items: Vec<Item>,
    /// The produced output items (opaque blobs stored unwrapped).
    pub output_items: Vec<Item>,
    /// Token usage for the turn.
    pub usage: Usage,
    /// Why the turn stopped.
    pub stop: StopReason,
    /// The lifecycle status.
    pub status: StoredStatus,
    /// Which backend produced it.
    pub binding: BackendBinding,
    /// Router-supplied creation time (unix seconds; `0` = unknown). Never read from a clock.
    pub created_at: u64,
    /// The client-dialect request params to echo back (`GET /responses/{id}`), as built by the
    /// client codec's `request_echo`.
    pub request_echo: Map<String, Value>,
    /// The error, when [`StoredStatus::Failed`].
    pub error: Option<XlateError>,
}

/// Build a [`StoredResponse`] from a completed turn (plan §9).
///
/// The status is derived from the response stop reason via [`StoredStatus::from_stop`]; `error`
/// is `None` (a failed turn is recorded by the router directly). `previous_id` comes from the
/// request's `state.previous_response_id`.
pub fn to_stored(
    req: &IrRequest,
    out: &IrResponse,
    binding: BackendBinding,
    id: ResponseId,
    created_at: u64,
    request_echo: Map<String, Value>,
) -> StoredResponse {
    StoredResponse {
        id,
        previous_id: req.state.previous_response_id.clone(),
        instructions: req.instructions.clone(),
        request_items: req.items.clone(),
        output_items: out.items.clone(),
        usage: out.usage.clone(),
        stop: out.stop.clone(),
        status: StoredStatus::from_stop(&out.stop),
        binding,
        created_at,
        request_echo,
        error: None,
    }
}

/// The backend binding of the newest element in a chain — the credential to pin for
/// in-flight encrypted reasoning replay (plan §9). `None` if the chain is empty.
pub fn chain_binding(chain: &[StoredResponse]) -> Option<&BackendBinding> {
    chain.last().map(|s| &s.binding)
}

/// Replay a stored chain plus a new request into a single stateless [`IrRequest`] (plan §9).
///
/// `chain` is ordered oldest → newest. Each element's `previous_id` must link to the previous
/// element's `id`, and the newest element's `id` must equal `new.state.previous_response_id`
/// (else [`XlateError::invalid_request`] on `previous_response_id`).
///
/// The result's items are the concatenation, over the chain, of each element's `request_items`
/// then `output_items`, followed by `new.items`. Instructions follow the plan §7.1 rule: the
/// **new** request's instructions are used as-is (prior instructions are not resurrected), with
/// every [`Position::Before`] index shifted by the number of prepended items.
/// `state.previous_response_id` is cleared; every other field comes from `new`.
pub fn materialize_chain(
    chain: &[StoredResponse],
    new: IrRequest,
) -> Result<IrRequest, XlateError> {
    // Validate the chain links: element i must chain from element i-1.
    for i in 1..chain.len() {
        if chain[i].previous_id.as_ref() != Some(&chain[i - 1].id) {
            return Err(XlateError::invalid_request(format!(
                "stored chain broken: element {i} ({}) does not link to its predecessor \
                 element {} ({})",
                chain[i].id,
                i - 1,
                chain[i - 1].id
            ))
            .with_param("previous_response_id"));
        }
    }
    // The new request must chain from the newest stored element.
    let newest = chain.last().map(|s| &s.id);
    if new.state.previous_response_id.as_ref() != newest {
        return Err(XlateError::invalid_request(
            "new request's previous_response_id does not match the newest stored response",
        )
        .with_param("previous_response_id"));
    }

    let mut new = new;

    // Concatenate prior turns: request_items then output_items for each chain element.
    let mut items = Vec::new();
    for s in chain {
        items.extend(s.request_items.iter().cloned());
        items.extend(s.output_items.iter().cloned());
    }
    let prepended = items.len();
    items.append(&mut new.items);

    // Shift mid-context instruction positions by the number of prepended items (§7.1).
    for ins in &mut new.instructions {
        if let Position::Before(i) = &mut ins.position {
            *i += prepended;
        }
    }

    new.items = items;
    new.state.previous_response_id = None;
    Ok(new)
}
