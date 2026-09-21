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
use llm_xlate_core::ir::{IrRequest, Position, Protocol};

use crate::requirements::Resolutions;

/// Re-map every mid-context instruction anchor through `f`, which maps an old item index
/// (`0..=old_len`) to its index in the rewritten item list.
///
/// A [`Position::Before`] anchor names *the item it sits in front of*, so any pass that
/// inserts or removes items must re-map the anchors in the same step. Skipping this is not a
/// cosmetic drift: an anchor that lands inside an assistant run splits one assistant turn into
/// two on the wire, separating `tool_calls` from the `tool` messages that answer them, and an
/// anchor pushed past the end is silently unplaceable. [`crate::store::materialize_chain`]
/// does the same re-mapping when it prepends a chain.
pub(crate) fn remap_instruction_anchors(req: &mut IrRequest, f: impl Fn(usize) -> usize) {
    for ins in &mut req.instructions {
        if let Position::Before(i) = &mut ins.position {
            *i = f(*i);
        }
    }
}

/// Drop every item whose `keep` flag is false, re-mapping instruction anchors so each
/// mid-context instruction keeps its place in the transcript.
///
/// An anchor on a dropped item moves to the next surviving item. An anchor already past the
/// end of the list stays past the end by the same margin, so [`instructions`] can still report
/// it rather than have it silently become a valid-looking position.
pub(crate) fn retain_items(req: &mut IrRequest, keep: &[bool]) {
    debug_assert_eq!(keep.len(), req.items.len(), "keep mask must cover every item");
    // prefix[i] = number of kept items strictly before i; prefix[len] = the new length.
    let mut prefix = Vec::with_capacity(keep.len() + 1);
    let mut kept = 0usize;
    prefix.push(0);
    for &k in keep {
        if k {
            kept += 1;
        }
        prefix.push(kept);
    }
    let old_len = keep.len();
    remap_instruction_anchors(req, |i| {
        if i <= old_len {
            prefix[i]
        } else {
            kept + (i - old_len)
        }
    });
    let mut idx = 0;
    req.items.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Re-map instruction anchors after items were inserted, given `inserted_before[i]` = the
/// number of new items spliced in immediately ahead of old item `i`.
///
/// An instruction anchored at `i` stays ahead of whatever was inserted there: the inserted
/// item belongs to the old item `i` (a reasoning blob resolved for its tool call), so the
/// instruction must still precede the pair.
pub(crate) fn remap_after_inserts(req: &mut IrRequest, inserted_before: &[usize]) {
    let mut prefix = Vec::with_capacity(inserted_before.len() + 1);
    let mut total = 0usize;
    prefix.push(0);
    for &n in inserted_before {
        total += n;
        prefix.push(total);
    }
    let old_len = inserted_before.len();
    remap_instruction_anchors(req, |i| i + prefix[i.min(old_len)]);
}

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
