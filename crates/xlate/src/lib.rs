//! # llm-xlate
//!
//! The cross-API translation facade for the LLM router. It wires together the sans-IO
//! primitives from [`llm_xlate_core`] (the IR, the capability registry, the codecs, the
//! envelope, the aggregator) into the two-phase request path the router drives:
//!
//! ```text
//!  decode_request → requirements → [router resolves I/O] → materialize_chain → lower → encode_request
//! ```
//!
//! This crate owns the protocol-agnostic, capability-gated policy layer that sits *between*
//! decoding a client request and encoding it for a backend:
//!
//! * [`requirements`](mod@crate::requirements) — a read-only pre-pass that tells the router what external lookups a
//!   request needs before it can be lowered (foreign file ids to bridge, missing reasoning
//!   blobs on the last tool turn to fetch from the sidecar, a `previous_response_id` chain to
//!   materialize).
//! * [`lower`](mod@crate::lower) — the ordered set of deterministic lowering passes (plan §7) that rewrite an
//!   [`IrRequest`] into a form the target [`Protocol`] can represent, recording every lossy
//!   step as a [`Degradation`] and rejecting anything unrepresentable with a typed
//!   [`XlateError`].
//! * [`store`] — the router-side stored-response model (plan §9): converting a completed
//!   response into a [`store::StoredResponse`], and replaying a `previous_response_id` chain
//!   back into a single stateless [`IrRequest`] with [`store::materialize_chain`].
//!
//! Everything here is pure: no I/O, no clocks, no RNG. Identical `(request, capabilities,
//! resolutions, config)` inputs always produce identical output (plan §11 invariant 1).
//!
//! The [`Translator`] façade (plan §4) wires the three protocol codec crates onto these
//! building blocks and exposes the whole request/response path — `decode_request` →
//! `requirements` → `materialize_chain` → `lower` → `encode_request`, plus the streaming and
//! stored-response codecs — as a single object the router holds. See [`translator`].

#![forbid(unsafe_code)]
#![allow(rustdoc::private_intra_doc_links)]
// The mandated fallible signatures all return `Result<_, XlateError>`, which is a large error
// type; boxing it everywhere would deviate from the plan's API for a marginal win.
#![allow(clippy::result_large_err)]

pub mod lower;
pub mod requirements;
pub mod store;
pub mod translator;

// Re-export the core crate so the router can depend on `llm-xlate` alone (CONVENTIONS.md).
pub use llm_xlate_core as core;
pub use llm_xlate_core::*;

pub use lower::{lower, Lowered};
pub use requirements::{requirements, FileRef, Requirements, Resolutions};
pub use store::{
    chain_binding, materialize_chain, to_stored, BackendBinding, StoredResponse, StoredStatus,
};
pub use translator::Translator;

// Re-export the three protocol codec structs so the router can name them through `llm-xlate`
// alone (it never depends on the codec crates directly).
pub use llm_xlate_anthropic::{AnthropicCodec, AnthropicStreamDecoder, AnthropicStreamEncoder};
pub use llm_xlate_chat::{ChatCodec, ChatStreamDecoder, ChatStreamEncoder};
pub use llm_xlate_responses::{ResponsesCodec, ResponsesStreamDecoder, ResponsesStreamEncoder};
