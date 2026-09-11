//! Deterministic lowering passes (plan §7).
//!
//! [`lower`] rewrites a decoded [`IrRequest`] into a form the target [`Protocol`] can
//! represent, given the resolved [`Capabilities`] and the router's [`Resolutions`]. It owns
//! the *cross-cutting, protocol-agnostic* policy that must run **before** a codec's
//! `encode_request` (which owns wire shape): reasoning-blob replay policy, foreign
//! provider-tool folding, foreign file resolution, schema-keyword normalization, `strict`
//! downgrade, effort snapping, `n>1` rejection, stop-sequence caps, state clearing, and more.
//!
//! Each pass is a small function `(&mut IrRequest, …, &mut Degradations) -> Result<(),
//! XlateError>` that mutates the request in place, records every lossy step as a
//! [`Degradation`](llm_xlate_core::degrade::Degradation), and returns a typed [`XlateError`] for anything it cannot represent. The
//! passes run in a **fixed order** so the result is deterministic; running `lower` twice on the
//! same inputs yields byte-identical output (plan §11 invariant 1).
//!
//! ## Pass order
//!
//! 1. [`protocol`] — the target protocol must be in `caps.transport.protocols`.
//! 2. [`reasoning`] — replay/drop reasoning items, resolve required reasoning, effort/budget.
//! 3. [`provider_tools`] — foreign provider tools rejected or dropped; foreign history folded.
//! 4. [`media`] — foreign file ids resolved; audio and size/count limits enforced.
//! 5. [`output_format`] — structured-output support, schema normalization, `strict`, verbosity.
//! 6. [`tools`] — function-tool schema/strict, `max_tools`, `tool_choice`, parallel, id pattern.
//! 7. [`sampling`] — `n>1`, stop-sequence cap, service tier.
//! 8. [`instructions`] — mid-conversation placement policy; effort/`clear_at` clearing.
//! 9. [`state`] — store/chain/conversation/background/include support; ZDR.
//! 10. [`session`] — session-id conflict degradation; required-session-id enforcement.
//! 11. [`ext`] — foreign-protocol `ext` keys reported once.

mod ext;
mod instructions;
mod media;
mod output_format;
mod protocol;
mod provider_tools;
mod reasoning;
mod sampling;
mod session;
mod state;
mod tools;

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::codec::TranslatorConfig;
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, Protocol};

use crate::requirements::Resolutions;

/// The result of lowering: the rewritten request plus the degradations recorded along the way.
///
/// The router surfaces `degradations` as `x-router-degraded` / `x-router-dropped` headers.
#[derive(Debug, Clone, PartialEq)]
pub struct Lowered {
    /// The lowered request, ready for the target codec's `encode_request`.
    pub req: IrRequest,
    /// Every lossy step taken while lowering, in order.
    pub degradations: Degradations,
}

/// Run every lowering pass in the fixed order (plan §7) and return the rewritten request.
///
/// Returns [`Ok`] with any [`Degradation`]s recorded, or an [`XlateError`] whose kind is
/// [`Unsupported`](llm_xlate_core::error::ErrorKind::Unsupported),
/// [`IncompatibleHistory`](llm_xlate_core::error::ErrorKind::IncompatibleHistory), or
/// [`InvalidRequest`](llm_xlate_core::error::ErrorKind::InvalidRequest) for anything the target
/// cannot represent. Never panics; never mutates anything not covered by a rule.
///
/// [`Degradation`]: llm_xlate_core::degrade::Degradation
pub fn lower(
    req: IrRequest,
    caps: &Capabilities,
    target: Protocol,
    res: &Resolutions,
    cfg: &TranslatorConfig,
) -> Result<Lowered, XlateError> {
    let mut req = req;
    let mut degradations = Degradations::new();

    protocol::run(&mut req, caps, target, &mut degradations)?;
    reasoning::run(&mut req, caps, target, res, cfg, &mut degradations)?;
    provider_tools::run(&mut req, caps, target, cfg, &mut degradations)?;
    media::run(&mut req, caps, target, res, &mut degradations)?;
    output_format::run(&mut req, caps, &mut degradations)?;
    tools::run(&mut req, caps, &mut degradations)?;
    sampling::run(&mut req, caps, &mut degradations)?;
    instructions::run(&mut req, caps, target, cfg, &mut degradations)?;
    state::run(&mut req, caps, &mut degradations)?;
    session::run(&mut req, caps, &mut degradations)?;
    ext::run(&mut req, target, &mut degradations)?;

    Ok(Lowered { req, degradations })
}
