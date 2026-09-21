//! # llm-xlate-core
//!
//! The shared foundation of `llm-xlate`, a sans-IO library that translates requests,
//! responses, streams, and errors between three LLM wire protocols — OpenAI Chat Completions,
//! OpenAI Responses, and Anthropic Messages — in any direction, with capability-gated,
//! explicitly-reported loss.
//!
//! This crate holds everything the three codec crates (`chat`, `responses`, `anthropic`) and
//! the `xlate` facade build on: the [intermediate representation](ir), the
//! [capability registry](caps), the [error taxonomy](error), the [`rtr1.` envelope](envelope),
//! [canonical JSON helpers](canon), the [SSE parser/writer](sse), the
//! [degradation records](degrade), the [fixed wrapper strings](wrap), the
//! [codec traits and contexts](codec), and the streaming [`Aggregator`](aggregate).
//!
//! ## Architecture (plan §3)
//!
//! ```text
//!  client bytes ─► decode_request ─► IrRequest ─► [router resolves I/O] ─► encode_request ─► provider bytes
//!  client bytes ◄─ StreamEncoder  ◄─ IrEvent*  ◄─ StreamDecoder ◄──────────────────────────── provider SSE
//!                                      └─ Aggregator ─► IrResponse (non-streaming clients, store, logging)
//! ```
//!
//! Guiding principles, all enforced here:
//!
//! * **Item-based stateless IR.** Chat messages and Anthropic turns are *bundles*: decoding
//!   flattens them into [`Item`]s (lossless); encoding regroups them by fixed rules.
//! * **Sans-IO.** No async, no I/O, no clocks, no RNG, no global counters — pure functions and
//!   push state machines only. Determinism: identical input + identical [`Capabilities`] yield
//!   byte-identical output. (Response encoders take `created_at` from the caller; they never
//!   read a clock.)
//! * **Capability-gated loss.** The [`caps`] registry decides every lossy step; `Unknown`
//!   behaves as "no" and yields [`ErrorKind::Unsupported`]. Every lossy step records a
//!   [`Degradation`] — never a silent drop.
//! * **One error space.** Providers' errors decode into [`XlateError`]; each codec renders it
//!   back into its dialect.
//! * **Opaque blobs cross safely.** Provider-bound reasoning is wrapped in the deterministic
//!   [`Sealer`] `rtr1.` envelope at the client boundary and forwarded natively at the provider
//!   boundary (see the [`codec`] module docs).
//!
//! ## Serialization conventions
//!
//! All IR types serialize with a fixed field order and preserve user JSON key order
//! (`serde_json` `preserve_order`). Tool arguments are a [`JsonText`] string, parsed only when
//! a target needs an object. Passthrough (schemas, `ext`, provider items) is
//! `serde_json::Value` — not `RawValue` — because the round-trip laws need `PartialEq`.

#![forbid(unsafe_code)]
// `XlateError` is the crate's single rich error type and appears in every fallible public
// signature the codec crates program against (the plan fixes these signatures). Boxing it
// would deviate from that API for a marginal stack-size win, so we accept the lint.
#![allow(clippy::result_large_err)]

pub mod aggregate;
pub mod canon;
pub mod caps;
pub mod codec;
pub mod degrade;
pub mod envelope;
pub mod error;
pub mod ir;
pub mod session;
pub mod sse;
pub mod wrap;

pub use aggregate::Aggregator;
pub use caps::{BackendOverrides, Capabilities, Registry, SamplingRule, Tri};
pub use codec::{
    Codec, DecodeCtx, EncodeCtx, EncodedError, EncodedRequest, ForeignProviderTool, HeaderMap,
    HeaderName, HeaderValue, MidInstructionFallback, StreamDecoder, StreamEncoder,
    TranslatorConfig, UnresolvedReasoning,
};
pub use degrade::{Degradation, DegradationKind, Degradations};
pub use envelope::{EnvelopeError, Sealer};
pub use error::{upstream_error_message, ErrorKind, XlateError};
pub use ir::*;
pub use session::{apply_session, capture_session, session_emit, SessionEmit, SESSION_HEADERS};
pub use sse::{SseEvent, SseParser, SseWriter};
pub use wrap::WRAP_VERSION;
