//! `llm-xlate-e2e` — the library surface behind the operator-driven live-API exploration and
//! capture binary.
//!
//! The binary (`src/main.rs`) is a thin CLI over these modules. They are exposed as the crate's
//! public API so the `check` / `translate` / `promote` engineers can build on the same probe
//! loader, capture format, observers and reporter without re-deriving anything:
//!
//! - [`probe`] — probe schema, loader, validation, dependency ordering, model-set expansion, and
//!   run-time placeholder substitution;
//! - [`models`] — named model sets and the free `GET /v1/models` refresh;
//! - [`keys`] — provider key loading and secret redaction / detection;
//! - [`assets`] — bundled deterministic media and the asset-payload redaction pass;
//! - [`client`] — per-protocol HTTP with verbatim raw-byte capture, redaction, retry and spacing;
//! - [`capture`] — the on-disk capture format, the [`capture::Capture`] / [`capture::Run`] loaders,
//!   and the run manifest;
//! - [`observe`] — protocol-aware automatic observations computed without `llm-xlate`;
//! - [`report`] — per-run `report.md` and run-vs-run `diff`.
//!
//! This crate never talks to the network except in [`client`] (sends, only on an explicit `run`)
//! and [`models::refresh_listings`] (the single free listing call); nothing here is part of
//! `cargo test --workspace`.

// `XlateError` (from `llm-xlate`) is a large error type; the codec crates carry the same allow.
// We surface it verbatim through the `check`/`translate` paths rather than boxing it everywhere.
#![allow(clippy::result_large_err)]

pub mod assets;
pub mod capture;
pub mod client;
pub mod keys;
pub mod models;
pub mod observe;
pub mod pricing;
pub mod probe;
pub mod promote;
pub mod report;
pub mod translate;
pub mod xlate_check;
