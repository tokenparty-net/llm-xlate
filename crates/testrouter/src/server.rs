//! The axum application (plan §4, §7): one POST handler per surface, `GET /v1/models`,
//! `GET /healthz`, working `501` stubs for the Responses store / cancel / count-tokens endpoints
//! (filled in by the store + CLI engineers), and the request-body-limit + request-log middleware.

use std::sync::Arc;
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use serde_json::{json, Value};
use tower_http::trace::TraceLayer;

use llm_xlate_core::error::{ErrorKind, XlateError};
use llm_xlate_core::ir::{Protocol, ResponseId};
use llm_xlate_core::HeaderMap;

use llm_xlate::store::StoredStatus;

use crate::config::{protocol_token_str, Overrides};
use crate::ids::mint_trace_id;
use crate::pipeline::{self, build_response};
use crate::route::Router;
use crate::trace::TraceBuilder;
use crate::upstream::UpstreamRequest;
use crate::AppState;

/// Assemble the axum router for a shared [`AppState`].
pub fn router(state: Arc<AppState>) -> axum::Router {
    let limit = state.max_body_bytes;
    axum::Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/messages", post(messages_handler))
        .route("/v1/models", get(models_handler))
        .route("/healthz", get(healthz_handler))
        // Responses stored-response surface (plan §9): GET/DELETE/input_items/cancel.
        .route(
            "/v1/responses/{id}",
            get(responses_get).delete(responses_delete),
        )
        .route("/v1/responses/{id}/input_items", get(responses_input_items))
        .route("/v1/responses/{id}/cancel", post(responses_cancel))
        // Anthropic `count_tokens` — an untranslated proxy passthrough (plan §7).
        .route("/v1/messages/count_tokens", post(count_tokens_handler))
        .layer(DefaultBodyLimit::max(limit))
        // Render an over-limit `413` (produced by `DefaultBodyLimit`) in the client dialect with a
        // trace id + trace record, so it obeys the same invariants as every other response.
        .layer(middleware::from_fn_with_state(state.clone(), body_limit_trace))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Map the request path to the client dialect it speaks (for rendering a pre-handler rejection).
fn protocol_for_path(path: &str) -> Protocol {
    if path.starts_with("/v1/chat/completions") {
        Protocol::OaiChat
    } else if path.starts_with("/v1/messages") {
        Protocol::Anthropic
    } else {
        Protocol::OaiResponses
    }
}

/// Middleware that catches the `413 Payload Too Large` produced by [`DefaultBodyLimit`] before any
/// handler runs and re-renders it as a dialect error body with an `x-xlate-trace-id` header and a
/// trace record (plan §4: every response carries a trace id and is recorded).
async fn body_limit_trace(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let headers = req.headers().clone();
    let resp = next.run(req).await;
    // Only rewrite a `413` produced by the body-limit layer itself — those carry no trace id. A
    // `413` a handler already rendered (with `x-xlate-trace-id`) is a real dialect response and is
    // left untouched.
    if resp.status() != StatusCode::PAYLOAD_TOO_LARGE || resp.headers().contains_key("x-xlate-trace-id") {
        return resp;
    }
    let protocol = protocol_for_path(&path);
    let trace_id = mint_trace_id(&*state.deps.id_source);
    let ts_start = state.deps.clock.now_rfc3339();
    let mut b = TraceBuilder::new(trace_id.clone(), ts_start, None, state.redact_over, state.include_raw);
    b.set_client(protocol_token_str(protocol), &method, &path, &headers, &[], false);

    // Keep the HTTP 413 status while rendering the body in the client dialect (the encoders map an
    // over-limit request to `invalid_request`, whose status is 400).
    let status = 413u16;
    let err = XlateError::invalid_request("request body exceeds the configured size limit");
    let enc = state.deps.translator.encode_error(protocol, &err, false, false);
    let rendered = serde_json::from_slice::<Value>(&enc.body).ok();
    b.push_error("ingress", err.kind.slug(), status, &err.message, rendered);

    let mut resp_headers = pipeline::base_response_headers(&trace_id, false, None);
    resp_headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    b.set_client_response_body(status, &resp_headers, &enc.body);
    pipeline::submit(&state, b, &state.deps.clock.now_rfc3339(), 0);
    build_response(status, resp_headers, Body::from(enc.body))
}

async fn chat_handler(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    pipeline::handle(state, Protocol::OaiChat, "POST".into(), "/v1/chat/completions".into(), headers, body).await
}

async fn responses_handler(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    pipeline::handle(state, Protocol::OaiResponses, "POST".into(), "/v1/responses".into(), headers, body).await
}

async fn messages_handler(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    pipeline::handle(state, Protocol::Anthropic, "POST".into(), "/v1/messages".into(), headers, body).await
}

/// `GET /healthz`. A pure liveness probe: it still carries an `x-xlate-trace-id` (the brief's
/// every-response invariant) but is not itself recorded as a trace.
async fn healthz_handler(State(state): State<Arc<AppState>>) -> Response {
    let trace_id = mint_trace_id(&*state.deps.id_source);
    let body = serde_json::to_vec(&json!({"status": "ok"})).unwrap_or_default();
    let mut headers = pipeline::base_response_headers(&trace_id, false, None);
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    build_response(200, headers, Body::from(body))
}

/// `GET /v1/models` (plan §7): an offline-safe merged listing built from the route rules + aliases.
/// In tests there is no network; a live cached listing may be merged in production.
async fn models_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(r) = auth_reject(&state, &headers, "GET", "/v1/models") {
        return r;
    }
    let cfg = state.router.config();
    let mut data = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    // Only concrete, usable model ids: the alias keys and their upstream targets, each filtered
    // through `Router::resolve` so it maps to a route. Route regex *patterns* are not model ids and
    // are deliberately excluded; a cached live-listing merge from both providers is a documented
    // offline gap (plan §11.5).
    let mut candidates: Vec<String> = Vec::new();
    for (client_name, target) in &cfg.aliases {
        if client_name != "default" {
            candidates.push(client_name.clone());
        }
        candidates.push(target.clone());
    }
    // Static per-provider model lists (e.g. a local vLLM server's models), still filtered
    // through the route rules below so only routable ids are advertised.
    for provider in cfg.providers.values() {
        candidates.extend(provider.models.iter().cloned());
    }
    for id in candidates {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Ok(r) = state.router.resolve(Protocol::OaiChat, Some(&id), &Overrides::default()) {
            data.push(model_row(&id, &r.provider));
        }
    }

    // `/v1/models` is a real client-facing API surface, so it carries an `x-xlate-trace-id` and is
    // recorded as a trace (visible to `trace triage`/`show`), like the store endpoints.
    let trace_id = mint_trace_id(&*state.deps.id_source);
    let ts_start = state.deps.clock.now_rfc3339();
    let mut b = TraceBuilder::new(trace_id.clone(), ts_start, None, state.redact_over, state.include_raw);
    b.set_client(protocol_token_str(Protocol::OaiResponses), "GET", "/v1/models", &headers, &[], false);
    let body = serde_json::to_vec(&json!({"object": "list", "data": data})).unwrap_or_default();
    let mut resp_headers = pipeline::base_response_headers(&trace_id, false, None);
    resp_headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    b.set_client_response_body(200, &resp_headers, &body);
    pipeline::submit(&state, b, &state.deps.clock.now_rfc3339(), 0);
    build_response(200, resp_headers, Body::from(body))
}

fn model_row(id: &str, provider: &str) -> Value {
    json!({"id": id, "object": "model", "owned_by": provider})
}

// ── Responses stored-response handlers (plan §9) ──────────────────────────────────────────────

/// `GET /v1/responses/{id}` — the stored `response` object at its persisted status. 404 for an
/// unknown id, 500 for a corrupt on-disk record; both in the OpenAI dialect.
async fn responses_get(State(state): State<Arc<AppState>>, Path(id): Path<String>, headers: HeaderMap) -> Response {
    let path = format!("/v1/responses/{id}");
    if let Some(r) = auth_reject(&state, &headers, "GET", &path) {
        return r;
    }
    let rid = ResponseId::new(id.clone());
    let op = match state.deps.store.get_checked(&rid) {
        Ok(Some(stored)) => match state.deps.translator.encode_stored_response(Protocol::OaiResponses, &stored) {
            Ok(body) => StoreOp::ok(body, id),
            Err(e) => StoreOp::err("store.get", e),
        },
        Ok(None) => StoreOp::err("store.get", missing(&id)),
        Err(e) => StoreOp::err("store.get", e),
    };
    finish_store_op(&state, "GET", &path, &headers, op)
}

/// `GET /v1/responses/{id}/input_items` — the request's input items.
async fn responses_input_items(State(state): State<Arc<AppState>>, Path(id): Path<String>, headers: HeaderMap) -> Response {
    let path = format!("/v1/responses/{id}/input_items");
    if let Some(r) = auth_reject(&state, &headers, "GET", &path) {
        return r;
    }
    let rid = ResponseId::new(id.clone());
    let op = match state.deps.store.get_checked(&rid) {
        Ok(Some(stored)) => StoreOp::ok(state.deps.translator.encode_input_items(&stored), id),
        Ok(None) => StoreOp::err("store.get", missing(&id)),
        Err(e) => StoreOp::err("store.get", e),
    };
    finish_store_op(&state, "GET", &path, &headers, op)
}

/// `DELETE /v1/responses/{id}` — `{id, object:"response.deleted", deleted:true}`; 404 if unknown.
async fn responses_delete(State(state): State<Arc<AppState>>, Path(id): Path<String>, headers: HeaderMap) -> Response {
    let path = format!("/v1/responses/{id}");
    if let Some(r) = auth_reject(&state, &headers, "DELETE", &path) {
        return r;
    }
    let rid = ResponseId::new(id.clone());
    let op = if state.deps.store.delete(&rid) {
        let body = serde_json::to_vec(&json!({"id": id, "object": "response.deleted", "deleted": true}))
            .unwrap_or_default();
        StoreOp::ok(Bytes::from(body), id)
    } else {
        StoreOp::err("store.delete", missing(&id))
    };
    finish_store_op(&state, "DELETE", &path, &headers, op)
}

/// `POST /v1/responses/{id}/cancel` — abort a background response's task and return the stored
/// object with `status:"cancelled"`; 404 if unknown.
async fn responses_cancel(State(state): State<Arc<AppState>>, Path(id): Path<String>, headers: HeaderMap) -> Response {
    let path = format!("/v1/responses/{id}/cancel");
    if let Some(r) = auth_reject(&state, &headers, "POST", &path) {
        return r;
    }
    let rid = ResponseId::new(id.clone());
    let op = match state.deps.store.get_checked(&rid) {
        Ok(Some(mut stored)) => {
            // Only an in-flight response can be cancelled. A response that already reached a
            // terminal status (completed / incomplete / failed / cancelled) is returned unchanged,
            // matching OpenAI — cancelling a finished response must not flip it to "cancelled".
            if matches!(stored.status, StoredStatus::Queued | StoredStatus::InProgress) {
                state.deps.store.abort_task(&rid);
                state.deps.store.set_status(&rid, StoredStatus::Cancelled);
                stored.status = StoredStatus::Cancelled;
            }
            match state.deps.translator.encode_stored_response(Protocol::OaiResponses, &stored) {
                Ok(body) => StoreOp::ok(body, id),
                Err(e) => StoreOp::err("store.cancel", e),
            }
        }
        Ok(None) => StoreOp::err("store.cancel", missing(&id)),
        Err(e) => StoreOp::err("store.cancel", e),
    };
    finish_store_op(&state, "POST", &path, &headers, op)
}

/// `POST /v1/messages/count_tokens` (Anthropic dialect) — an untranslated **proxy passthrough**
/// (plan §7): the client body is forwarded verbatim to the resolved provider's
/// `/v1/messages/count_tokens`, provider auth is injected by the upstream seam (client credentials
/// are never forwarded), and the upstream JSON is returned as-is. The whole leg is recorded so the
/// request is visible to `trace triage`/`show`. Errors (auth / routing / upstream) render in the
/// Anthropic dialect.
async fn count_tokens_handler(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let path = "/v1/messages/count_tokens";
    let deps = state.deps.clone();
    let start = Instant::now();
    let trace_id = mint_trace_id(&*deps.id_source);
    let ts_start = deps.clock.now_rfc3339();
    let overrides = Overrides::from_headers(&headers);
    let mut b = TraceBuilder::new(trace_id.clone(), ts_start, overrides.tag.clone(), state.redact_over, state.include_raw);
    b.set_client(protocol_token_str(Protocol::Anthropic), "POST", path, &headers, &body, false);

    // ── auth (Anthropic dialect) ──────────────────────────────────────────────────────────────
    if !state.token.is_empty() && !pipeline::token_ok(&state.token, &headers) {
        let err = XlateError::new(ErrorKind::Authentication, "invalid or missing router token");
        return finish_count_tokens_error(&state, b, &trace_id, "ingress", err, start);
    }

    // ── route ─────────────────────────────────────────────────────────────────────────────────
    let model = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.get("model").and_then(Value::as_str).map(str::to_string));
    let route = match state.router.resolve(Protocol::Anthropic, model.as_deref(), &overrides) {
        Ok(r) => r,
        Err(e) => {
            let err = XlateError::invalid_request(e.to_string()).with_param("model");
            return finish_count_tokens_error(&state, b, &trace_id, "route", err, start);
        }
    };
    b.set_route(serde_json::to_value(Router::trace(&route, &overrides)).unwrap_or(Value::Null));

    // ── forward the raw body upstream ─────────────────────────────────────────────────────────
    let base = state.router.provider_base_url(&route.provider).unwrap_or("").trim_end_matches('/');
    let url = format!("{base}/v1/messages/count_tokens");
    let up_headers = forward_anthropic_headers(&headers);
    b.set_upstream_request(&url, &up_headers, &body, false);
    let up_req = UpstreamRequest {
        provider: route.provider.clone(),
        protocol: Protocol::Anthropic,
        url,
        headers: up_headers,
        body: body.clone(),
        stream: false,
    };
    let up = match deps.upstream.send(up_req).await {
        Ok(up) => up,
        Err(e) => {
            let err = if e.timeout {
                XlateError::new(ErrorKind::Timeout, e.message)
            } else {
                XlateError::new(ErrorKind::ServerError, e.message)
            }
            .with_provider(route.family.clone());
            return finish_count_tokens_error(&state, b, &trace_id, "upstream", err, start);
        }
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let ttfb_ms = up.ttfb.as_millis() as u64;
    b.set_upstream_response(up.status, &up.headers, None, elapsed_ms, ttfb_ms);
    let status = up.status;
    let raw = pipeline::collect_body(up.body).await;
    b.set_upstream_raw(&raw);

    let mut resp_headers = pipeline::base_response_headers(&trace_id, false, None);
    resp_headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    b.set_client_response_body(status, &resp_headers, &raw);
    let total_ms = start.elapsed().as_millis() as u64;
    pipeline::submit(&state, b, &deps.clock.now_rfc3339(), total_ms);
    build_response(status, resp_headers, Body::from(raw))
}

/// Render a `count_tokens` failure (auth / routing / upstream) in the Anthropic dialect and record
/// the trace.
fn finish_count_tokens_error(
    state: &AppState,
    mut b: TraceBuilder,
    trace_id: &str,
    stage: &str,
    err: XlateError,
    start: Instant,
) -> Response {
    let enc = state.deps.translator.encode_error(Protocol::Anthropic, &err, false, false);
    let rendered = serde_json::from_slice::<Value>(&enc.body).ok();
    b.push_error(stage, err.kind.slug(), enc.status, &err.message, rendered);
    let mut headers = pipeline::base_response_headers(trace_id, false, None);
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    b.set_client_response_body(enc.status, &headers, &enc.body);
    let total_ms = start.elapsed().as_millis() as u64;
    pipeline::submit(state, b, &state.deps.clock.now_rfc3339(), total_ms);
    build_response(enc.status, headers, Body::from(enc.body))
}

/// The upstream headers for a `count_tokens` passthrough: the Anthropic protocol/beta headers the
/// client sent (`anthropic-version`, `anthropic-beta`), never the client's credentials — provider
/// auth is injected by the [`crate::upstream::Upstream`] seam.
fn forward_anthropic_headers(client: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for name in ["anthropic-version", "anthropic-beta"] {
        if let Some(v) = client.get(name) {
            out.insert(name, v.clone());
        }
    }
    out
}

/// A `NotFound` error for an unknown stored id (rendered 404 in the OpenAI dialect).
fn missing(id: &str) -> XlateError {
    XlateError::new(ErrorKind::NotFound, format!("response `{id}` not found"))
}

/// Enforce the optional shared token on the non-pipeline endpoints (the store surface and
/// `/v1/models`); the three POST surfaces enforce it inside [`pipeline::handle`]. Returns a traced
/// 401 in the OpenAI dialect when a token is configured and the request does not present it, so
/// these endpoints cannot be used to read/delete/cancel stored responses without the token. With
/// no token configured (the local-testing default) every request is accepted.
fn auth_reject(state: &AppState, headers: &HeaderMap, method: &str, path: &str) -> Option<Response> {
    if state.token.is_empty() || pipeline::token_ok(&state.token, headers) {
        return None;
    }
    let err = XlateError::new(ErrorKind::Authentication, "invalid or missing router token");
    Some(finish_store_op(state, method, path, headers, StoreOp::err("ingress", err)))
}

/// The outcome of a stored-response operation: a 200 body (+ the stored id for the trace) or an
/// error tagged with the pipeline stage that produced it.
enum StoreOp {
    Ok { body: Bytes, stored_id: String },
    Err { stage: &'static str, err: XlateError },
}

impl StoreOp {
    fn ok(body: Bytes, stored_id: String) -> StoreOp {
        StoreOp::Ok { body, stored_id }
    }
    fn err(stage: &'static str, err: XlateError) -> StoreOp {
        StoreOp::Err { stage, err }
    }
}

/// Render a stored-response operation's response and write its trace record (stage names
/// `store.get` / `store.delete` / `store.cancel`). All these endpoints are the Responses dialect.
fn finish_store_op(state: &AppState, method: &str, path: &str, headers: &HeaderMap, op: StoreOp) -> Response {
    let trace_id = mint_trace_id(&*state.deps.id_source);
    let ts_start = state.deps.clock.now_rfc3339();
    let mut b = TraceBuilder::new(trace_id.clone(), ts_start, None, state.redact_over, state.include_raw);
    b.set_client(crate::config::protocol_token_str(Protocol::OaiResponses), method, path, headers, &[], false);

    let (status, body) = match op {
        StoreOp::Ok { body, stored_id } => {
            b.set_store(Some(stored_id), None);
            (200u16, body)
        }
        StoreOp::Err { stage, err } => {
            let enc = state.deps.translator.encode_error(Protocol::OaiResponses, &err, false, false);
            let rendered = serde_json::from_slice::<Value>(&enc.body).ok();
            b.push_error(stage, err.kind.slug(), enc.status, &err.message, rendered);
            (enc.status, enc.body)
        }
    };

    let mut resp_headers = pipeline::base_response_headers(&trace_id, false, None);
    resp_headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    b.set_client_response_body(status, &resp_headers, &body);
    pipeline::submit(state, b, &state.deps.clock.now_rfc3339(), 0);
    build_response(status, resp_headers, Body::from(body))
}

/// Bind and serve the app (plan §6 `serve`).
pub async fn serve(app: axum::Router, listen: &str) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!("xlate-testrouter listening on {listen}");
    axum::serve(listener, app).await?;
    Ok(())
}
