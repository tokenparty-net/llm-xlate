//! The per-request pipeline (plan §4): a linear state machine around `llm-xlate`, sans-IO in the
//! middle and I/O at the edges. Every stage feeds a [`TraceBuilder`]; a failure at any stage is
//! rendered in the client dialect and still produces a complete trace record.
//!
//! Stages: ingress → decode → route → requirements/chain → lower → encode → send → response.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::header::CONTENT_TYPE;
use axum::response::Response;
use bytes::Bytes;
use serde_json::{json, Value};

use llm_xlate_core::codec::EncodeCtx;
use llm_xlate_core::error::{ErrorKind, XlateError};
use llm_xlate_core::ir::{IrRequest, IrResponse, Item, Protocol, ResponseId};
use llm_xlate_core::HeaderMap;

use llm_xlate::store::{BackendBinding, StoredResponse, StoredStatus};

use crate::config::{protocol_token_str, Overrides};
use crate::ids::{mint_response_id, mint_trace_id};
use crate::route::{Route, Router};
use crate::sidecar::SidecarKey;
use crate::trace::{TraceBuilder, Timing};
use crate::upstream::{upstream_path, UpstreamBody, UpstreamRequest, UpstreamResponse};
use crate::{AppState, Deps};

/// A stage failure (plan §6 brief): which stage, the error, and whether a client `Start` frame was
/// already emitted (streaming error shape).
#[derive(Debug)]
pub struct Failure {
    /// The stage that failed.
    pub stage: &'static str,
    /// The unified error.
    pub error: XlateError,
    /// Whether a `Start` was already sent to the client.
    pub started: bool,
}

/// Run one request end-to-end and return the client [`Response`]. Never panics: a panic inside
/// `llm-xlate` is caught and rendered as a 500 in the client dialect.
pub async fn handle(
    state: Arc<AppState>,
    client_protocol: Protocol,
    method: String,
    path: String,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let start = Instant::now();
    let deps = state.deps.clone();
    let trace_id = mint_trace_id(&*deps.id_source);
    let ts_start = deps.clock.now_rfc3339();
    let overrides = Overrides::from_headers(&headers);
    let client_stream = client_requested_stream(client_protocol, &body);

    let mut b = TraceBuilder::new(
        trace_id.clone(),
        ts_start,
        overrides.tag.clone(),
        state.redact_over,
        state.include_raw,
    );
    b.set_client(protocol_token_str(client_protocol), &method, &path, &headers, &body, client_stream);

    // ── ingress: token ──────────────────────────────────────────────────────────────────────
    if !state.token.is_empty() && !token_ok(&state.token, &headers) {
        let e = XlateError::new(ErrorKind::Authentication, "invalid or missing router token");
        return finish_error(&state, b, client_protocol, "ingress", e, &trace_id, start);
    }

    // ── decode ──────────────────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let ir = match guard(|| deps.translator.decode_request(client_protocol, &body, &headers)) {
        Ok(ir) => ir,
        Err(e) => return finish_error(&state, b, client_protocol, "decode", e, &trace_id, start),
    };
    let decode_us = t0.elapsed().as_micros() as u64;
    b.set_ir_request(&to_value(&ir));

    // ── route ───────────────────────────────────────────────────────────────────────────────
    let client_model = if ir.model.client_name.is_empty() { None } else { Some(ir.model.client_name.as_str()) };
    let route = match state.router.resolve(client_protocol, client_model, &overrides) {
        Ok(r) => r,
        Err(e) => {
            let e = XlateError::invalid_request(e.to_string()).with_param("model");
            return finish_error(&state, b, client_protocol, "route", e, &trace_id, start);
        }
    };
    b.set_route(to_value(&Router::trace(&route, &overrides)));
    // The router resolves the client's model name (alias, override, or verbatim) to the backend
    // model before lowering; codecs encode `model.upstream()` and echo `client_name` to the client.
    let mut ir = ir;
    ir.model.resolved = Some(route.upstream_model.clone());

    // ── R2: background (Responses only) ──────────────────────────────────────────────────────
    // `background:true` is accepted immediately with a `queued` object; the turn runs in a task
    // that updates the stored status/output. `stream:true` + background is rejected (400).
    if client_protocol == Protocol::OaiResponses && ir.state.background == Some(true) {
        return accept_background(state.clone(), b, ir, route, client_stream, &trace_id, start);
    }

    // ── requirements + chain + sidecar ──────────────────────────────────────────────────────
    let reqs = deps.translator.requirements(&ir, &route.caps, route.upstream_protocol);
    b.set_requirements(
        json!({
            "reasoning_for_calls": reqs.reasoning_for_calls.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
            "foreign_files": reqs.foreign_files.iter().map(|f| json!({"family": f.family.label(), "id": f.id})).collect::<Vec<_>>(),
            "chain": reqs.chain.as_ref().map(|c| c.as_str()),
        }),
        Value::Null,
    );

    let mut resolutions = llm_xlate::requirements::Resolutions::new();
    // R2: keep the pre-materialization request for the store — its *own* new items plus the
    // original `previous_response_id` — so a stored chain re-concatenates correctly on the next
    // turn instead of nesting an already-materialized transcript.
    let pre_materialize_ir: Option<IrRequest> =
        (client_protocol == Protocol::OaiResponses).then(|| ir.clone());
    // Chain materialization.
    if let Some(prev) = reqs.chain.clone() {
        match deps.store.chain(&prev) {
            Ok(chain) => {
                let ids: Vec<String> = chain.iter().map(|s| s.id.as_str().to_string()).collect();
                b.set_chain(Some(prev.as_str().to_string()), ids);
                match guard(|| deps.translator.materialize_chain(&chain, ir.clone())) {
                    Ok(m) => ir = m,
                    Err(e) => return finish_error(&state, b, client_protocol, "chain", e, &trace_id, start),
                }
            }
            Err(e) => return finish_error(&state, b, client_protocol, "chain", e, &trace_id, start),
        }
    }
    // Sidecar reasoning resolution (files are never bridged in this router — Unsupported).
    let mut resolved_summary = Vec::new();
    for call in &reqs.reasoning_for_calls {
        let key = crate::sidecar::SidecarKey::new(route.family.clone(), route.upstream_model.clone(), call.clone());
        if let Some(blob) = deps.sidecar.get(&key) {
            resolved_summary.push(json!({"call_id": call.as_str(), "kind": format!("{:?}", blob.kind), "len": blob.data.len()}));
            resolutions.reasoning.insert(call.clone(), blob);
        }
    }
    if !resolved_summary.is_empty() {
        b.set_requirements(
            json!({
                "reasoning_for_calls": reqs.reasoning_for_calls.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
                "foreign_files": reqs.foreign_files.iter().map(|f| json!({"family": f.family.label(), "id": f.id})).collect::<Vec<_>>(),
                "chain": reqs.chain.as_ref().map(|c| c.as_str()),
            }),
            json!({"reasoning": resolved_summary}),
        );
    }

    // ── encode ctx (from the pre-lowering, pre-materialization request) ─────────────────────
    // Building the ctx from `pre_materialize_ir` (when present, i.e. Responses) keeps the request
    // echo — most importantly `previous_response_id` — intact, since chain materialization consumes
    // it out of `ir`. The upstream request is encoded from `lowered.req`, not the echo, so this
    // only affects what the client-facing response reflects back (plan §9).
    let response_id = mint_response_id(&*deps.id_source, client_protocol);
    let created_at = deps.clock.now_unix();
    let ctx_source = pre_materialize_ir.as_ref().unwrap_or(&ir);
    let mut ctx = deps.translator.encode_ctx(client_protocol, ctx_source, response_id.clone(), created_at);
    if let Some(exp) = overrides.expose_mode() {
        ctx.expose = exp;
    }

    // Client-request re-encode round-trip check (informational; never fails the request).
    b.set_reencode_diff(reencode_diff(&deps.translator, client_protocol, &ir));

    // Keep the pre-lowering / pre-materialization request for the store (Responses `store:true`).
    let store_ir: Option<IrRequest> = if ctx.store == Some(true) { pre_materialize_ir } else { None };

    // ── lower ───────────────────────────────────────────────────────────────────────────────
    let t1 = Instant::now();
    let lowered = match guard(|| deps.translator.lower(ir, &route.caps, route.upstream_protocol, &resolutions)) {
        Ok(l) => l,
        Err(e) => {
            return finish_error(&state, b, client_protocol, "lower", e, &trace_id, start);
        }
    };
    let lower_us = t1.elapsed().as_micros() as u64;
    let degradations: Vec<Value> = to_value(&lowered.degradations).as_array().cloned().unwrap_or_default();
    b.set_lowered(to_value(&lowered.req), degradations);

    // ── encode request ─────────────────────────────────────────────────────────────────────
    let t2 = Instant::now();
    let enc = match guard(|| {
        deps.translator
            .encode_request(route.upstream_protocol, &lowered.req, &route.caps, &ctx)
    }) {
        Ok(e) => e,
        Err(e) => return finish_error(&state, b, client_protocol, "encode", e, &trace_id, start),
    };
    let encode_us = t2.elapsed().as_micros() as u64;
    // The encoder's own lossy steps (e.g. identifier rewrites) belong in the trace too.
    b.add_degradations(to_value(&enc.degradations).as_array().cloned().unwrap_or_default());
    let resp_ctx = enc.ctx.clone();

    // ── send ────────────────────────────────────────────────────────────────────────────────
    let base = state.router.provider_base_url(&route.provider).unwrap_or("").trim_end_matches('/');
    let url = format!("{base}{}", upstream_path(route.upstream_protocol));
    b.set_upstream_request(&url, &enc.headers, &enc.body, enc.upstream_streams);

    let up_req = UpstreamRequest {
        provider: route.provider.clone(),
        protocol: route.upstream_protocol,
        url: url.clone(),
        headers: enc.headers.clone(),
        body: enc.body.clone(),
        stream: enc.upstream_streams,
    };
    let up = match deps.upstream.send(up_req).await {
        Ok(up) => up,
        Err(e) => {
            let xe = if e.timeout {
                XlateError::new(ErrorKind::Timeout, e.message)
            } else {
                XlateError::new(ErrorKind::ServerError, e.message)
            }
            .with_provider(route.family.clone());
            return finish_error(&state, b, client_protocol, "upstream", xe, &trace_id, start);
        }
    };

    // Record the upstream response head.
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let ttfb_ms = up.ttfb.as_millis() as u64;
    b.set_upstream_response(up.status, &up.headers, None, elapsed_ms, ttfb_ms);

    let timing = Timing {
        decode_us,
        lower_us,
        encode_us,
        upstream_ms: ttfb_ms,
        total_ms: 0,
    };
    b.set_timing(timing);

    respond(RespondCtx {
        state: state.clone(),
        builder: b,
        client_protocol,
        upstream_protocol: route.upstream_protocol,
        caps: route.caps.clone(),
        resp_ctx,
        client_stream,
        route,
        response_id,
        created_at,
        store_ir,
        trace_id,
        start,
    }, up)
    .await
}

/// Context threaded into the response phase (shared with [`crate::stream`]).
pub(crate) struct RespondCtx {
    pub(crate) state: Arc<AppState>,
    pub(crate) builder: TraceBuilder,
    pub(crate) client_protocol: Protocol,
    pub(crate) upstream_protocol: Protocol,
    pub(crate) caps: llm_xlate_core::caps::Capabilities,
    pub(crate) resp_ctx: llm_xlate_core::codec::EncodeCtx,
    pub(crate) client_stream: bool,
    pub(crate) route: Route,
    pub(crate) response_id: llm_xlate_core::ir::ResponseId,
    pub(crate) created_at: u64,
    pub(crate) store_ir: Option<IrRequest>,
    pub(crate) trace_id: String,
    pub(crate) start: Instant,
}

async fn respond(mut c: RespondCtx, up: UpstreamResponse) -> Response {
    let translator = c.state.deps.translator.clone();
    let upstream_request_id = up
        .headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    // ── error passthrough (non-2xx) ─────────────────────────────────────────────────────────
    if up.status >= 400 {
        let raw = collect_body(up.body).await;
        c.builder.set_upstream_raw(&raw);
        let err = translator.decode_error(c.upstream_protocol, up.status, &raw, &up.headers, &c.caps);
        return finish_error_from_upstream(c, err, upstream_request_id);
    }

    // ── success ─────────────────────────────────────────────────────────────────────────────
    if c.client_stream {
        // Streaming client: hand off to the stream module, which owns the builder and submits the
        // trace when the stream completes or the client disconnects.
        return crate::stream::streaming_response(c, up, upstream_request_id);
    }

    // Non-streaming client.
    let (events, raw) = match up.body {
        UpstreamBody::Full(bytes) => {
            let evs = match guard(|| translator.decode_response(c.upstream_protocol, &bytes, &c.caps)) {
                Ok(evs) => evs,
                Err(e) => {
                    c.builder.set_upstream_raw(&bytes);
                    return finish_error_from_upstream(c, e, upstream_request_id);
                }
            };
            (evs, bytes)
        }
        UpstreamBody::Stream(_) => {
            // Streaming upstream → non-streaming client: aggregate the stream (plan §8).
            let bytes = collect_body(up.body).await;
            let mut dec = translator.stream_decoder(c.upstream_protocol, &c.caps);
            let mut evs = dec.push(&bytes);
            evs.extend(dec.finish());
            (evs, bytes)
        }
    };
    c.builder.set_upstream_raw(&raw);

    let elapsed_ms = c.start.elapsed().as_millis() as u64;
    for ev in &events {
        c.builder.push_event(elapsed_ms, ev);
    }
    let ir_response = match translator.aggregate_stream(events) {
        Ok(r) => r,
        Err(e) => return finish_error_from_upstream(c, e, upstream_request_id),
    };
    c.builder.set_ir_response(&ir_response);

    let body_bytes = translator.encode_response(c.client_protocol, &ir_response, &c.resp_ctx);

    // Plan §5/§11 client-frame law over the actual client body.
    let law = crate::trace::client_frame_law(
        &translator,
        c.client_protocol,
        &body_bytes,
        false,
        &ir_response,
        &c.resp_ctx,
    );
    c.builder.set_aggregate_check(law);

    // R2: the sidecar learns every opaque reasoning blob that precedes a tool call, regardless of
    // client protocol or `store` (plan §4.5); persist the stored response for Responses `store:true`.
    let store_ir = c.store_ir.take();
    persist_aggregate(
        &c.state.deps,
        PersistInputs {
            route: &c.route,
            response_id: &c.response_id,
            created_at: c.created_at,
            resp_ctx: &c.resp_ctx,
            store_ir: store_ir.as_ref(),
            ir_response: &ir_response,
        },
        &mut c.builder,
    );

    let degraded = c.builder_degradation_count() > 0;
    let mut headers = base_response_headers(&c.trace_id, degraded, upstream_request_id.as_deref());
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    c.builder.set_client_response_body(200, &headers, &body_bytes);

    let total_ms = c.start.elapsed().as_millis() as u64;
    submit(&c.state, c.builder, &c.state.deps.clock.now_rfc3339(), total_ms);

    build_response(200, headers, Body::from(body_bytes))
}

/// The inputs to [`persist_aggregate`]: the route/id/ctx identity of the turn plus the aggregated
/// response and (when the request asked to be stored) the pre-materialization request. Bundled so
/// the shared helper stays within clippy's argument budget.
pub(crate) struct PersistInputs<'a> {
    /// The resolved route.
    pub route: &'a Route,
    /// The minted response id.
    pub response_id: &'a ResponseId,
    /// The response's `created_at` unix timestamp.
    pub created_at: u64,
    /// The response-side encode ctx (carries the request echo).
    pub resp_ctx: &'a EncodeCtx,
    /// The pre-materialization request to store, when `store:true` (else `None`).
    pub store_ir: Option<&'a IrRequest>,
    /// The aggregated response.
    pub ir_response: &'a IrResponse,
}

/// Sidecar-learn from an aggregated response, and persist it as a stored response when the request
/// asked for it (`store_ir` is `Some`). Shared by the non-streaming ([`respond`]) and streaming
/// ([`crate::stream`]) success paths so a `store:true` turn is durable regardless of streaming
/// (plan §4.4/§4.5).
pub(crate) fn persist_aggregate(deps: &Deps, inputs: PersistInputs, builder: &mut TraceBuilder) {
    let PersistInputs { route, response_id, created_at, resp_ctx, store_ir, ir_response } = inputs;
    learn_sidecar(deps, route, ir_response);
    let Some(store_ir) = store_ir else {
        return;
    };
    let binding = make_binding(route, Some(ir_response));
    let stored = deps.translator.to_stored(
        store_ir,
        ir_response,
        binding,
        response_id.clone(),
        created_at,
        resp_ctx.request_echo.clone(),
    );
    let previous = stored.previous_id.as_ref().map(|p| p.as_str().to_string());
    deps.store.put(stored);
    builder.set_store(Some(response_id.as_str().to_string()), previous);
}

impl RespondCtx {
    fn builder_degradation_count(&self) -> u32 {
        // The builder holds the count on its checks; recompute via the recorded value.
        self.builder.degradation_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// R2: store binding, sidecar learning, and background execution
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The backend binding for a stored response: the provider name as the credential id, the family
/// and upstream model, and the provider's own response id when the aggregate carried one.
fn make_binding(route: &Route, ir_response: Option<&IrResponse>) -> BackendBinding {
    let mut binding =
        BackendBinding::new(route.provider.clone(), route.family.clone(), route.upstream_model.clone());
    if let Some(r) = ir_response {
        let rid = r.id.as_str();
        if !rid.is_empty() {
            binding.provider_response_id = Some(rid.to_string());
        }
    }
    binding
}

/// Teach the sidecar every opaque reasoning blob that precedes a tool call in an aggregated
/// response, keyed by `(provider family, upstream model, call id)` (plan §4.5). A later tool turn
/// whose transcript does not carry the blob (a Chat client replaying an Anthropic thinking turn,
/// or a Responses `store:false` encrypted-reasoning turn) resolves it back out.
fn learn_sidecar(deps: &Deps, route: &Route, ir_response: &IrResponse) {
    let mut last_blob: Option<llm_xlate_core::ir::OpaqueBlob> = None;
    for item in &ir_response.items {
        match item {
            Item::Reasoning(ri) => {
                if let Some(blob) = &ri.opaque {
                    last_blob = Some(blob.clone());
                }
            }
            Item::ToolCall { call_id, .. } => {
                if let Some(blob) = &last_blob {
                    let key = SidecarKey::new(route.family.clone(), route.upstream_model.clone(), call_id.clone());
                    deps.sidecar.put(key, blob.clone());
                }
            }
            _ => {}
        }
    }
}

/// Accept a `background:true` Responses request: reject a streaming background request (400),
/// otherwise persist a `queued` object, spawn the turn as a task (registering its abort handle for
/// `cancel`), and return the queued object immediately.
fn accept_background(
    state: Arc<AppState>,
    mut b: TraceBuilder,
    ir: IrRequest,
    route: Route,
    client_stream: bool,
    trace_id: &str,
    start: Instant,
) -> Response {
    let deps = state.deps.clone();
    if client_stream {
        let e = XlateError::invalid_request(
            "background responses cannot also stream (`stream:true` + `background:true` is unsupported)",
        )
        .with_param("stream");
        return finish_error(&state, b, Protocol::OaiResponses, "background", e, trace_id, start);
    }

    let response_id = mint_response_id(&*deps.id_source, Protocol::OaiResponses);
    let created_at = deps.clock.now_unix();
    let ctx_seed = deps.translator.encode_ctx(Protocol::OaiResponses, &ir, response_id.clone(), created_at);

    let queued = StoredResponse {
        id: response_id.clone(),
        previous_id: ir.state.previous_response_id.clone(),
        instructions: ir.instructions.clone(),
        request_items: ir.items.clone(),
        output_items: Vec::new(),
        usage: Default::default(),
        stop: llm_xlate_core::ir::StopReason::EndTurn,
        status: StoredStatus::Queued,
        binding: make_binding(&route, None),
        created_at,
        request_echo: ctx_seed.request_echo.clone(),
        error: None,
    };
    deps.store.put(queued.clone());

    let handle = tokio::spawn(run_background_turn(
        state.clone(),
        ir,
        route,
        ctx_seed,
        response_id.clone(),
        created_at,
    ));
    deps.store.register_task(&response_id, handle.abort_handle());

    let body = deps.translator.encode_stored_response(Protocol::OaiResponses, &queued).unwrap_or_default();
    let mut headers = base_response_headers(trace_id, false, None);
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    b.set_store(
        Some(response_id.as_str().to_string()),
        queued.previous_id.as_ref().map(|p| p.as_str().to_string()),
    );
    b.set_client_response_body(200, &headers, &body);
    let total_ms = start.elapsed().as_millis() as u64;
    submit(&state, b, &state.deps.clock.now_rfc3339(), total_ms);
    build_response(200, headers, Body::from(body))
}

/// Run a background turn to completion and update its stored record. On success the stored object
/// becomes `completed`/`incomplete` with the produced output; on an upstream/translation error it
/// becomes `failed` with the error. Aborting the task (via `cancel`) drops this future at an await
/// point; a cancel that lands in the post-await window is honoured by a compare-and-set re-read so
/// it cannot be clobbered by a terminal `completed`/`failed` write. The whole upstream leg is
/// captured in its own trace record (tagged `background:<id>`) so `trace triage`/`show` see it.
async fn run_background_turn(
    state: Arc<AppState>,
    ir: IrRequest,
    route: Route,
    ctx_seed: EncodeCtx,
    response_id: ResponseId,
    created_at: u64,
) {
    let deps = state.deps.clone();
    let start = Instant::now();
    deps.store.set_status(&response_id, StoredStatus::InProgress);

    // A dedicated trace record for the background upstream leg.
    let trace_id = mint_trace_id(&*deps.id_source);
    let ts_start = deps.clock.now_rfc3339();
    let mut b = TraceBuilder::new(
        trace_id,
        ts_start,
        Some(format!("background:{}", response_id.as_str())),
        state.redact_over,
        state.include_raw,
    );
    b.set_client(protocol_token_str(Protocol::OaiResponses), "POST", "/v1/responses", &HeaderMap::new(), &[], false);
    b.set_ir_request(&to_value(&ir));
    b.set_route(to_value(&Router::trace(&route, &Overrides::default())));

    let outcome = run_background_inner(&state, &route, ir, &ctx_seed, &mut b, start).await;
    match outcome {
        Ok((ir_response, resp_ctx, req)) => {
            let binding = make_binding(&route, Some(&ir_response));
            let stored = deps.translator.to_stored(
                &req,
                &ir_response,
                binding,
                response_id.clone(),
                created_at,
                resp_ctx.request_echo.clone(),
            );
            let body = deps
                .translator
                .encode_stored_response(Protocol::OaiResponses, &stored)
                .unwrap_or_default();
            let law = crate::trace::client_frame_law(
                &deps.translator,
                Protocol::OaiResponses,
                &body,
                false,
                &ir_response,
                &resp_ctx,
            );
            b.set_aggregate_check(law);
            // Atomic compare-and-set inside the store: a `cancel` that already set `Cancelled`
            // (aborting us a beat too late) wins over this terminal write, and the check + write
            // cannot be interleaved by a cancel landing between them.
            if deps.store.put_if_not_cancelled(stored) {
                learn_sidecar(&deps, &route, &ir_response);
                b.set_store(
                    Some(response_id.as_str().to_string()),
                    req.state.previous_response_id.as_ref().map(|p| p.as_str().to_string()),
                );
                let mut headers = base_response_headers(b.trace_id(), false, None);
                headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
                b.set_client_response_body(200, &headers, &body);
            } else {
                b.push_error("cancelled", "cancelled", 499, "background response cancelled before completion", None);
            }
        }
        Err(e) => {
            let enc = deps.translator.encode_error(Protocol::OaiResponses, &e, false, false);
            let rendered = serde_json::from_slice::<Value>(&enc.body).ok();
            b.push_error("upstream", e.kind.slug(), enc.status, &e.message, rendered);
            let mut headers = base_response_headers(b.trace_id(), false, None);
            headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
            b.set_client_response_body(enc.status, &headers, &enc.body);
            // Route the terminal `failed` write through the same atomic guard so a racing cancel
            // wins over it too.
            if let Some(mut s) = deps.store.get(&response_id) {
                s.status = StoredStatus::Failed;
                s.error = Some(e);
                deps.store.put_if_not_cancelled(s);
            }
        }
    }
    let total_ms = start.elapsed().as_millis() as u64;
    submit(&state, b, &deps.clock.now_rfc3339(), total_ms);
    // The task is finishing; forget its (now-defunct) abort handle.
    deps.store.abort_task(&response_id);
}

/// The I/O body of a background turn: lower → encode → send → decode → aggregate, recording each
/// stage into `b`. Returns the aggregated response, the response-side encode ctx, and the request
/// used for `to_stored`.
async fn run_background_inner(
    state: &Arc<AppState>,
    route: &Route,
    ir: IrRequest,
    ctx_seed: &EncodeCtx,
    b: &mut TraceBuilder,
    start: Instant,
) -> Result<(IrResponse, EncodeCtx, IrRequest), XlateError> {
    let deps = state.deps.clone();
    let resolutions = llm_xlate::requirements::Resolutions::new();

    // The request stored under this id keeps its own items; strip `background` before lowering so
    // it is never forwarded upstream.
    let store_req = ir.clone();
    let mut lower_req = ir;
    lower_req.state.background = None;

    let lowered = deps.translator.lower(lower_req, &route.caps, route.upstream_protocol, &resolutions)?;
    let degradations: Vec<Value> = to_value(&lowered.degradations).as_array().cloned().unwrap_or_default();
    b.set_lowered(to_value(&lowered.req), degradations);

    let enc = deps.translator.encode_request(route.upstream_protocol, &lowered.req, &route.caps, ctx_seed)?;
    b.add_degradations(to_value(&enc.degradations).as_array().cloned().unwrap_or_default());

    let base = state.router.provider_base_url(&route.provider).unwrap_or("").trim_end_matches('/');
    let url = format!("{base}{}", upstream_path(route.upstream_protocol));
    b.set_upstream_request(&url, &enc.headers, &enc.body, enc.upstream_streams);
    let up_req = UpstreamRequest {
        provider: route.provider.clone(),
        protocol: route.upstream_protocol,
        url,
        headers: enc.headers.clone(),
        body: enc.body.clone(),
        stream: enc.upstream_streams,
    };
    let up = deps
        .upstream
        .send(up_req)
        .await
        .map_err(|e| XlateError::new(ErrorKind::ServerError, e.message).with_provider(route.family.clone()))?;

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let ttfb_ms = up.ttfb.as_millis() as u64;
    b.set_upstream_response(up.status, &up.headers, None, elapsed_ms, ttfb_ms);

    if up.status >= 400 {
        let raw = collect_body(up.body).await;
        b.set_upstream_raw(&raw);
        return Err(deps.translator.decode_error(route.upstream_protocol, up.status, &raw, &up.headers, &route.caps));
    }

    let bytes = collect_body(up.body).await;
    b.set_upstream_raw(&bytes);
    let events = match deps.translator.decode_response(route.upstream_protocol, &bytes, &route.caps) {
        Ok(evs) => evs,
        Err(_) => {
            let mut dec = deps.translator.stream_decoder(route.upstream_protocol, &route.caps);
            let mut evs = dec.push(&bytes);
            evs.extend(dec.finish());
            evs
        }
    };
    let now = start.elapsed().as_millis() as u64;
    for ev in &events {
        b.push_event(now, ev);
    }
    let ir_response = deps.translator.aggregate_stream(events)?;
    b.set_ir_response(&ir_response);
    Ok((ir_response, enc.ctx, store_req))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Error rendering
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Render an error that occurred *before* the upstream response (decode/route/lower/encode/token).
fn finish_error(
    state: &Arc<AppState>,
    mut b: TraceBuilder,
    client_protocol: Protocol,
    stage: &str,
    err: XlateError,
    trace_id: &str,
    start: Instant,
) -> Response {
    let enc = state.deps.translator.encode_error(client_protocol, &err, false, false);
    let rendered = serde_json::from_slice::<Value>(&enc.body).ok();
    b.push_error(stage, err.kind.slug(), enc.status, &err.message, rendered);

    let mut headers = base_response_headers(trace_id, false, err.upstream_request_id.as_deref());
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    if let Some(ra) = err.retry_after {
        if let Ok(v) = ra.as_secs().to_string().parse() {
            headers.insert("retry-after", v);
        }
    }
    b.set_client_response_body(enc.status, &headers, &enc.body);
    let total_ms = start.elapsed().as_millis() as u64;
    submit(state, b, &state.deps.clock.now_rfc3339(), total_ms);
    build_response(enc.status, headers, Body::from(enc.body))
}

/// Render an error decoded from a non-2xx upstream response.
///
/// This is only ever reached *before* the HTTP response has been committed (a non-2xx upstream, or
/// a failure decoding a 2xx body), so the error is always a plain JSON body — even for a streaming
/// client. Real providers/gateways return JSON for a pre-stream failure; the SSE `data:{error}` /
/// `event: error` shape is only valid mid-stream (after 200 is committed), and that genuine
/// mid-stream path lives in [`crate::stream::Encoding::on_event`], not here.
fn finish_error_from_upstream(
    mut c: RespondCtx,
    err: XlateError,
    upstream_request_id: Option<String>,
) -> Response {
    let enc = c.state.deps.translator.encode_error(c.client_protocol, &err, false, false);
    let rendered = serde_json::from_slice::<Value>(&enc.body).ok();
    c.builder.push_error("upstream", err.kind.slug(), enc.status, &err.message, rendered);

    let uid = err.upstream_request_id.clone().or(upstream_request_id);
    let mut headers = base_response_headers(&c.trace_id, false, uid.as_deref());
    if let Some(ra) = err.retry_after {
        if let Ok(v) = ra.as_secs().to_string().parse() {
            headers.insert("retry-after", v);
        }
    }
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    c.builder.set_client_response_body(enc.status, &headers, &enc.body);
    let total_ms = c.start.elapsed().as_millis() as u64;
    submit(&c.state, c.builder, &c.state.deps.clock.now_rfc3339(), total_ms);
    build_response(enc.status, headers, Body::from(enc.body))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Helpers (shared with stream.rs)
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Build the always-present response headers.
pub(crate) fn base_response_headers(
    trace_id: &str,
    degraded: bool,
    upstream_request_id: Option<&str>,
) -> HeaderMap {
    let mut h = HeaderMap::new();
    if let Ok(v) = trace_id.parse() {
        h.insert("x-xlate-trace-id", v);
    }
    if degraded {
        h.insert("x-router-degraded", "true".parse().unwrap());
    }
    if let Some(id) = upstream_request_id {
        if let Ok(v) = id.parse() {
            h.insert("x-router-upstream-request-id", v);
        }
    }
    // The OpenAI/Anthropic SDKs surface `x-request-id` (`response._request_id` / `.request_id`) as
    // the support/debug id. Expose the upstream provider's request id when we have one, else fall
    // back to the router trace id, so every response carries a stable SDK-visible id.
    let req_id = upstream_request_id.unwrap_or(trace_id);
    if let Ok(v) = req_id.parse() {
        h.insert("x-request-id", v);
    }
    h
}

/// Assemble an axum [`Response`] from a status, headers, and body.
pub(crate) fn build_response(status: u16, headers: HeaderMap, body: Body) -> Response {
    let mut resp = Response::new(body);
    *resp.status_mut() = axum::http::StatusCode::from_u16(status).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    *resp.headers_mut() = headers;
    resp
}

/// Submit a finished trace record if tracing is enabled.
pub(crate) fn submit(state: &AppState, builder: TraceBuilder, ts_end: &str, total_ms: u64) {
    let mut builder = builder;
    builder.set_total_ms(total_ms);
    if state.trace_enabled {
        let record = builder.finish(ts_end.to_string());
        state.deps.trace_sink.submit(record);
    }
}

/// Collect a (possibly streaming) upstream body into a single buffer.
pub(crate) async fn collect_body(body: UpstreamBody) -> Bytes {
    match body {
        UpstreamBody::Full(b) => b,
        UpstreamBody::Stream(mut s) => {
            use futures::StreamExt;
            let mut buf = Vec::new();
            while let Some(chunk) = s.next().await {
                match chunk {
                    Ok(c) => buf.extend_from_slice(&c),
                    Err(_) => break,
                }
            }
            Bytes::from(buf)
        }
    }
}

/// Whether the client asked for a streaming response, by inspecting the body's `stream` flag.
pub fn client_requested_stream(_client_protocol: Protocol, body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// Whether the router token matches (`Authorization: Bearer <token>` or `x-xlate-token`).
pub(crate) fn token_ok(token: &str, headers: &HeaderMap) -> bool {
    if let Some(v) = headers.get("x-xlate-token").and_then(|v| v.to_str().ok()) {
        if v == token {
            return true;
        }
    }
    if let Some(v) = headers.get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if v.strip_prefix("Bearer ").map(str::trim) == Some(token) {
            return true;
        }
    }
    false
}

/// Catch a panic inside a sans-IO `llm-xlate` call and map it to a 500 [`XlateError`].
fn guard<T>(f: impl FnOnce() -> Result<T, XlateError>) -> Result<T, XlateError> {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(XlateError::new(ErrorKind::ServerError, "internal translation panic")),
    }
}

fn to_value<T: serde::Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

/// Client-request re-encode round-trip check: `decode(body) == decode(encode(decode(body)))` at
/// the IR level, using permissive caps for the client protocol. Returns the top-level field names
/// that diverge (empty = clean round trip). Never fails the request.
fn reencode_diff(translator: &llm_xlate::Translator, client_protocol: Protocol, ir: &IrRequest) -> Vec<String> {
    let mut caps = llm_xlate_core::caps::Capabilities::unknown();
    caps.transport.protocols = Some(vec![client_protocol]);
    let ctx = translator.encode_ctx(
        client_protocol,
        ir,
        llm_xlate_core::ir::ResponseId::new(""),
        0,
    );
    let reencoded = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let enc = translator.encode_request(client_protocol, ir, &caps, &ctx)?;
        translator.decode_request(client_protocol, &enc.body, &HeaderMap::new())
    }));
    let ir2 = match reencoded {
        Ok(Ok(ir2)) => ir2,
        _ => return vec!["<reencode-failed>".to_string()],
    };
    let a = to_value(ir);
    let b = to_value(&ir2);
    value_top_level_diff(&a, &b)
}

/// Report the top-level object keys whose values differ between two JSON values.
fn value_top_level_diff(a: &Value, b: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let (Some(ao), Some(bo)) = (a.as_object(), b.as_object()) {
        let mut keys: Vec<&String> = ao.keys().chain(bo.keys()).collect();
        keys.sort();
        keys.dedup();
        for k in keys {
            if ao.get(k) != bo.get(k) {
                out.push(k.clone());
            }
        }
    } else if a != b {
        out.push("<root>".to_string());
    }
    out
}
