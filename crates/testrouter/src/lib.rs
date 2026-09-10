//! # llm-xlate-testrouter
//!
//! A standalone, tracing test router built on [`llm_xlate`] (plan §1). It serves all three API
//! surfaces, routes each request to an upstream **through the IR** (decode → requirements → lower
//! → encode → send → decode → aggregate → encode), and writes a full [`trace`] of every request.
//! It is an I/O shell only: all translation logic stays sans-IO in `llm-xlate`.
//!
//! The public entry point is [`App::build`], which wires a [`config::Config`] and a [`Deps`] bag
//! into an [`axum::Router`]. Tests inject a mock upstream and deterministic id/clock sources via
//! [`Deps`]; see the `test_support` module (compiled under the `test-support` feature / tests).

#![forbid(unsafe_code)]
// `XlateError` is large; the codec crates carry the same allow. We surface it verbatim.
#![allow(clippy::result_large_err)]

pub mod config;
pub mod ids;
pub mod pipeline;
pub mod route;
pub mod server;
pub mod sidecar;
pub mod store;
pub mod stream;
pub mod trace;
pub mod tracecli;
pub mod upstream;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

use std::sync::Arc;
use std::time::Duration;

use llm_xlate::Translator;

use crate::config::Config;
use crate::ids::{Clock, IdSource};
use crate::route::Router;
use crate::sidecar::Sidecar;
use crate::store::ResponseStore;
use crate::trace::TraceSink;
use crate::upstream::Upstream;

/// The injected dependency bag (plan §11 lib API). Tests swap in a mock upstream, sequential ids,
/// a fixed clock, and an in-memory trace sink.
#[derive(Clone)]
pub struct Deps {
    /// The cross-API translation façade.
    pub translator: Translator,
    /// Minted-id source (uuid in prod, sequential in tests).
    pub id_source: Arc<dyn IdSource>,
    /// Wall-clock source (system in prod, fixed in tests).
    pub clock: Arc<dyn Clock>,
    /// The Responses stored-response store.
    pub store: Arc<dyn ResponseStore>,
    /// The opaque-reasoning-blob sidecar.
    pub sidecar: Arc<dyn Sidecar>,
    /// Where finished trace records go.
    pub trace_sink: Arc<dyn TraceSink>,
    /// The upstream I/O implementation.
    pub upstream: Arc<dyn Upstream>,
}

/// The shared, immutable router state handed to every handler.
pub struct AppState {
    /// The injected dependencies.
    pub deps: Deps,
    /// The compiled routing table.
    pub router: Router,
    /// The shared client token; empty accepts any credential.
    pub token: String,
    /// Maximum request body size.
    pub max_body_bytes: usize,
    /// Client SSE keepalive interval.
    pub keepalive: Duration,
    /// Media-redaction threshold.
    pub redact_over: usize,
    /// Whether to keep raw SSE / frame bytes in traces.
    pub include_raw: bool,
    /// Whether tracing is enabled.
    pub trace_enabled: bool,
}

/// The router application.
pub struct App;

impl App {
    /// Build the axum router for a config + dependency bag (plan §11).
    pub fn build(config: Config, deps: Deps) -> anyhow::Result<axum::Router> {
        let keepalive = Duration::from_secs(config.server.keepalive_secs.max(1));
        let max_body_bytes = config.server.max_body_bytes;
        let token = config.server.token.clone();
        let redact_over = config.trace.redact_media_over_bytes;
        let include_raw = config.trace.include_raw_sse;
        let trace_enabled = config.trace.enabled;
        let router = Router::new(config)?;
        let state = Arc::new(AppState {
            deps,
            router,
            token,
            max_body_bytes,
            keepalive,
            redact_over,
            include_raw,
            trace_enabled,
        });
        Ok(server::router(state))
    }
}
