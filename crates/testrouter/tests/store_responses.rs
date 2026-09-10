//! Hermetic R2 tests (plan §9): the persistent Responses store, sidecar, chains, background, and
//! cancel. Every request goes through an in-process mock upstream — no network, no secrets. These
//! tests build the app directly with a [`FileStore`]/[`FileSidecar`] over a temp data dir so they
//! also cover durability across an `App` rebuild.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use bytes::Bytes;
use serde_json::{json, Value};
use tower::ServiceExt;

use llm_xlate::Translator;
use llm_xlate_core::codec::TranslatorConfig;
use llm_xlate_core::ir::{
    CallId, IrResponse, Item, ItemId, OpaqueBlob, OpaqueKind, Protocol, ProviderFamily,
    ReasoningItem, ResponseId, StopReason, Usage,
};
use llm_xlate_core::HeaderMap;

use llm_xlate_testrouter::config::Config;
use llm_xlate_testrouter::ids::{FixedClock, SequentialSource};
use llm_xlate_testrouter::sidecar::{FileSidecar, Sidecar, SidecarKey};
use llm_xlate_testrouter::store::{FileStore, ResponseStore};
use llm_xlate_testrouter::test_support::{respond_full, text_response, MemoryTraceSink, MockUpstream, TestResponse};
use llm_xlate_testrouter::trace::TraceRecord;
use llm_xlate_testrouter::upstream::{
    BoxFuture, Upstream, UpstreamBody, UpstreamError, UpstreamRequest, UpstreamResponse,
};
use llm_xlate_testrouter::{App, Deps};

// ── harness ──────────────────────────────────────────────────────────────────────────────────

/// A fresh, empty temp data dir for one test.
fn tmp_dir(tag: &str) -> PathBuf {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let p = std::env::temp_dir().join(format!("xlate_tr_store_{tag}_{n}"));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// A deterministic app over a [`FileStore`]/[`FileSidecar`] rooted at `dir`.
struct H {
    app: axum::Router,
    sink: Arc<MemoryTraceSink>,
    store: Arc<FileStore>,
    sidecar: Arc<FileSidecar>,
}

fn build(dir: &Path, upstream: Arc<dyn Upstream>) -> H {
    let store = Arc::new(FileStore::new(dir).unwrap());
    let sidecar = Arc::new(FileSidecar::new(dir).unwrap());
    let sink = Arc::new(MemoryTraceSink::new());
    let deps = Deps {
        translator: Translator::new(TranslatorConfig::default()),
        id_source: Arc::new(SequentialSource::new()),
        clock: Arc::new(FixedClock::default()),
        store: store.clone(),
        sidecar: sidecar.clone(),
        trace_sink: sink.clone(),
        upstream,
    };
    let app = App::build(Config::example(), deps).expect("build app");
    H { app, sink, store, sidecar }
}

impl H {
    async fn post(&self, path: &str, headers: &[(&str, &str)], body: impl Into<Bytes>) -> TestResponse {
        let mut b = axum::http::Request::builder().method("POST").uri(path);
        for (k, v) in headers {
            b = b.header(*k, *v);
        }
        let req = b.body(Body::from(body.into())).unwrap();
        collect(self.app.clone().oneshot(req).await.unwrap()).await
    }
    async fn get(&self, path: &str) -> TestResponse {
        let req = axum::http::Request::builder().method("GET").uri(path).body(Body::empty()).unwrap();
        collect(self.app.clone().oneshot(req).await.unwrap()).await
    }
    async fn delete(&self, path: &str) -> TestResponse {
        let req = axum::http::Request::builder().method("DELETE").uri(path).body(Body::empty()).unwrap();
        collect(self.app.clone().oneshot(req).await.unwrap()).await
    }
    fn trace(&self, resp: &TestResponse) -> TraceRecord {
        let id = resp.trace_id();
        self.sink.get(&id).unwrap_or_else(|| panic!("no trace for {id}"))
    }
}

async fn collect(resp: axum::response::Response) -> TestResponse {
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    TestResponse { status, headers, body }
}

/// An [`Upstream`] wrapper that records the exact request bytes and delegates to a [`MockUpstream`].
struct RecUpstream {
    inner: MockUpstream,
    raw: Arc<Mutex<Vec<Bytes>>>,
}
impl RecUpstream {
    fn new(inner: MockUpstream) -> (Arc<Self>, Arc<Mutex<Vec<Bytes>>>) {
        let raw = Arc::new(Mutex::new(Vec::new()));
        (Arc::new(Self { inner, raw: raw.clone() }), raw)
    }
}
impl Upstream for RecUpstream {
    fn send<'a>(&'a self, req: UpstreamRequest) -> BoxFuture<'a, Result<UpstreamResponse, UpstreamError>> {
        self.raw.lock().unwrap().push(req.body.clone());
        self.inner.send(req)
    }
}

/// An [`Upstream`] that parks until released — for background/cancel timing.
struct GateUpstream {
    gate: Arc<tokio::sync::Notify>,
    body: Bytes,
}
impl Upstream for GateUpstream {
    fn send<'a>(&'a self, _req: UpstreamRequest) -> BoxFuture<'a, Result<UpstreamResponse, UpstreamError>> {
        let gate = self.gate.clone();
        let body = self.body.clone();
        Box::pin(async move {
            gate.notified().await;
            Ok(UpstreamResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: UpstreamBody::Full(body),
                ttfb: Duration::ZERO,
            })
        })
    }
}

fn responses_body(model: &str, input: Value, store: bool) -> Bytes {
    Bytes::from(serde_json::to_vec(&json!({"model": model, "input": input, "store": store})).unwrap())
}

// ── create / get / input_items / delete shapes ───────────────────────────────────────────────

#[tokio::test]
async fn create_then_get_returns_stored_object() {
    let dir = tmp_dir("get");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi back"))).into_arc());
    let resp = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("hello"), true)).await;
    assert_eq!(resp.status, 200, "create body={}", resp.text());
    let id = resp.json().get("id").and_then(Value::as_str).unwrap().to_string();
    assert!(id.starts_with("resp_"), "minted id: {id}");

    let got = h.get(&format!("/v1/responses/{id}")).await;
    assert_eq!(got.status, 200, "get body={}", got.text());
    let v = got.json();
    assert_eq!(v.get("id").and_then(Value::as_str), Some(id.as_str()));
    assert_eq!(v.get("object").and_then(Value::as_str), Some("response"));
    assert_eq!(v.get("status").and_then(Value::as_str), Some("completed"));
    assert!(v.get("output").and_then(Value::as_array).map(|a| !a.is_empty()).unwrap_or(false), "output present");
}

#[tokio::test]
async fn get_unknown_id_is_404_openai_dialect() {
    let dir = tmp_dir("get404");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "x"))).into_arc());
    let got = h.get("/v1/responses/resp_does_not_exist").await;
    assert_eq!(got.status, 404, "body={}", got.text());
    assert!(got.json().get("error").is_some(), "openai error body: {}", got.text());
    let rec = h.trace(&got);
    assert!(rec.errors.iter().any(|e| e.stage == "store.get"), "store.get stage recorded");
}

#[tokio::test]
async fn input_items_lists_request_items() {
    let dir = tmp_dir("inputitems");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "ok"))).into_arc());
    let resp = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("please summarize"), true)).await;
    let id = resp.json().get("id").and_then(Value::as_str).unwrap().to_string();

    let items = h.get(&format!("/v1/responses/{id}/input_items")).await;
    assert_eq!(items.status, 200, "body={}", items.text());
    let v = items.json();
    assert_eq!(v.get("object").and_then(Value::as_str), Some("list"));
    let data = v.get("data").and_then(Value::as_array).unwrap();
    assert!(!data.is_empty(), "input items present");
    assert_eq!(data[0].get("role").and_then(Value::as_str), Some("user"));
}

#[tokio::test]
async fn input_items_unknown_id_is_404() {
    let dir = tmp_dir("inputitems404");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "x"))).into_arc());
    let got = h.get("/v1/responses/resp_missing/input_items").await;
    assert_eq!(got.status, 404, "body={}", got.text());
    assert!(h.trace(&got).errors.iter().any(|e| e.stage == "store.get"));
}

#[tokio::test]
async fn delete_removes_and_returns_deleted_shape() {
    let dir = tmp_dir("delete");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "ok"))).into_arc());
    let resp = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("hi"), true)).await;
    let id = resp.json().get("id").and_then(Value::as_str).unwrap().to_string();

    let del = h.delete(&format!("/v1/responses/{id}")).await;
    assert_eq!(del.status, 200, "body={}", del.text());
    let v = del.json();
    assert_eq!(v.get("id").and_then(Value::as_str), Some(id.as_str()));
    assert_eq!(v.get("object").and_then(Value::as_str), Some("response.deleted"));
    assert_eq!(v.get("deleted").and_then(Value::as_bool), Some(true));

    // Gone afterwards.
    assert_eq!(h.get(&format!("/v1/responses/{id}")).await.status, 404);
}

#[tokio::test]
async fn delete_unknown_id_is_404() {
    let dir = tmp_dir("delete404");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "x"))).into_arc());
    let del = h.delete("/v1/responses/resp_nope").await;
    assert_eq!(del.status, 404, "body={}", del.text());
    assert!(h.trace(&del).errors.iter().any(|e| e.stage == "store.delete"));
}

#[tokio::test]
async fn cancel_unknown_id_is_404() {
    let dir = tmp_dir("cancel404");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "x"))).into_arc());
    let c = h.post("/v1/responses/resp_nope/cancel", &[], Bytes::new()).await;
    assert_eq!(c.status, 404, "body={}", c.text());
    assert!(h.trace(&c).errors.iter().any(|e| e.stage == "store.cancel"));
}

// ── store policy ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn store_false_does_not_persist_but_sidecar_learns() {
    let dir = tmp_dir("storefalse");
    // Upstream returns reasoning (with an opaque blob) + a tool call.
    let resp_ir = reasoning_and_tool("gpt-4o", ProviderFamily::OpenAI, "toolu_sf");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &resp_ir)).into_arc());
    let resp = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("go"), false)).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    // Nothing persisted (store:false).
    assert!(h.store.list().is_empty(), "store must be empty on store:false");
    // But the sidecar learned the blob keyed by the tool call.
    let key = SidecarKey::new(ProviderFamily::OpenAI, "gpt-4o", CallId::new("toolu_sf"));
    assert!(h.sidecar.get(&key).is_some(), "sidecar learned the reasoning blob");
}

#[tokio::test]
async fn chat_client_never_stores() {
    let dir = tmp_dir("chatnostore");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi"))).into_arc());
    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "store": true});
    let resp = h.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    assert!(h.store.list().is_empty(), "chat clients never store");
}

// ── chains ───────────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn missing_previous_response_id_is_404() {
    let dir = tmp_dir("chainmissing");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "x"))).into_arc());
    let body = json!({"model": "gpt-4o", "input": "next", "previous_response_id": "resp_ghost", "store": true});
    let resp = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 404, "body={}", resp.text());
    assert!(h.trace(&resp).errors.iter().any(|e| e.stage == "chain"), "chain-stage error");
}

#[tokio::test]
async fn get_chained_response_echoes_previous_id() {
    let dir = tmp_dir("chainget");
    let h = build(
        &dir,
        MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "answer"))).into_arc(),
    );
    let t1 = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("first"), true)).await;
    let id1 = t1.json().get("id").and_then(Value::as_str).unwrap().to_string();
    let b2 = json!({"model": "gpt-4o", "input": "second", "previous_response_id": id1, "store": true});
    let t2 = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&b2).unwrap())).await;
    assert_eq!(t2.status, 200, "t2 body={}", t2.text());
    let id2 = t2.json().get("id").and_then(Value::as_str).unwrap().to_string();

    let got = h.get(&format!("/v1/responses/{id2}")).await;
    assert_eq!(got.status, 200, "body={}", got.text());
    assert_eq!(got.json().get("previous_response_id").and_then(Value::as_str), Some(id1.as_str()));

    let chain = h.store.chain(&ResponseId::new(id2.clone())).unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].id.as_str(), id1);
    assert_eq!(chain[1].id.as_str(), id2);
}

#[tokio::test]
async fn chained_upstream_matches_stateless_twin() {
    let dir = tmp_dir("chaintwin");
    let (up, raw) = RecUpstream::new(MockUpstream::always(respond_full(
        Protocol::OaiResponses,
        &text_response("gpt-4o", "the answer"),
    )));
    let h = build(&dir, up);

    // Turn 1: fresh, stored.
    let t1 = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("first"), true)).await;
    let id1 = t1.json().get("id").and_then(Value::as_str).unwrap().to_string();
    // Turn 2: chains from turn 1.
    let b2 = json!({"model": "gpt-4o", "input": "second", "previous_response_id": id1, "store": true});
    let t2 = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&b2).unwrap())).await;
    assert_eq!(t2.status, 200, "t2 body={}", t2.text());
    let id2 = t2.json().get("id").and_then(Value::as_str).unwrap().to_string();
    // Turn 3: chains from turn 2 (2-hop walk).
    let b3 = json!({"model": "gpt-4o", "input": "third", "previous_response_id": id2, "store": true});
    let t3 = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&b3).unwrap())).await;
    assert_eq!(t3.status, 200, "t3 body={}", t3.text());

    // The chained turn-3 upstream body: the 3rd captured request.
    let chained = raw.lock().unwrap()[2].clone();

    // A hand-built stateless twin carrying the whole transcript inline.
    let out_msg = |text: &str| {
        json!({"type": "message", "role": "assistant", "id": "msg_up_1",
               "content": [{"type": "output_text", "text": text, "annotations": []}]})
    };
    let user_msg = |text: &str| json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]});
    let twin_input = json!([
        user_msg("first"), out_msg("the answer"),
        user_msg("second"), out_msg("the answer"),
        user_msg("third"),
    ]);
    // Fresh app so the twin does not chain; capture its upstream body.
    let dir2 = tmp_dir("chaintwin2");
    let (up2, raw2) = RecUpstream::new(MockUpstream::always(respond_full(
        Protocol::OaiResponses,
        &text_response("gpt-4o", "the answer"),
    )));
    let h2 = build(&dir2, up2);
    let twin = h2.post("/v1/responses", &[], responses_body("gpt-4o", twin_input, true)).await;
    assert_eq!(twin.status, 200, "twin body={}", twin.text());
    let twin_body = raw2.lock().unwrap()[0].clone();

    if chained != twin_body {
        panic!(
            "chained != twin\n--- chained ---\n{}\n--- twin ---\n{}",
            String::from_utf8_lossy(&chained),
            String::from_utf8_lossy(&twin_body)
        );
    }
}

// ── background + cancel ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn background_returns_queued_immediately() {
    let dir = tmp_dir("bgqueued");
    let gate = Arc::new(tokio::sync::Notify::new());
    let up = Arc::new(GateUpstream {
        gate: gate.clone(),
        body: full_resp_bytes("gpt-4o", "done"),
    });
    let h = build(&dir, up);
    let body = json!({"model": "gpt-4o", "input": "long job", "background": true, "store": true});
    let resp = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200, "body={}", resp.text());
    let v = resp.json();
    assert_eq!(v.get("status").and_then(Value::as_str), Some("queued"));
    let id = v.get("id").and_then(Value::as_str).unwrap().to_string();
    assert!(id.starts_with("resp_"));
    // Release so the task can finish and not leak.
    gate.notify_waiters();
}

#[tokio::test]
async fn background_completes_and_is_pollable() {
    let dir = tmp_dir("bgdone");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "final"))).into_arc());
    let body = json!({"model": "gpt-4o", "input": "job", "background": true, "store": true});
    let resp = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 200);
    let id = resp.json().get("id").and_then(Value::as_str).unwrap().to_string();

    // Poll until the background task completes.
    let mut status = String::new();
    for _ in 0..50 {
        tokio::task::yield_now().await;
        let got = h.get(&format!("/v1/responses/{id}")).await;
        status = got.json().get("status").and_then(Value::as_str).unwrap_or("").to_string();
        if status == "completed" {
            break;
        }
    }
    assert_eq!(status, "completed", "background never completed");
}

#[tokio::test]
async fn background_plus_stream_is_400() {
    let dir = tmp_dir("bgstream");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "x"))).into_arc());
    let body = json!({"model": "gpt-4o", "input": "job", "background": true, "stream": true});
    let resp = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    assert_eq!(resp.status, 400, "body={}", resp.text());
    assert!(resp.json().get("error").is_some());
    assert!(h.trace(&resp).errors.iter().any(|e| e.stage == "background"));
}

#[tokio::test]
async fn background_cancel_marks_cancelled() {
    let dir = tmp_dir("bgcancel");
    let gate = Arc::new(tokio::sync::Notify::new());
    let up = Arc::new(GateUpstream { gate: gate.clone(), body: full_resp_bytes("gpt-4o", "never") });
    let h = build(&dir, up);
    let body = json!({"model": "gpt-4o", "input": "long", "background": true, "store": true});
    let resp = h.post("/v1/responses", &[], Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    let id = resp.json().get("id").and_then(Value::as_str).unwrap().to_string();

    // Let the task reach `in_progress` (parked in the gated upstream send).
    let mut in_progress = false;
    for _ in 0..50 {
        tokio::task::yield_now().await;
        let got = h.get(&format!("/v1/responses/{id}")).await;
        if got.json().get("status").and_then(Value::as_str) == Some("in_progress") {
            in_progress = true;
            break;
        }
    }
    assert!(in_progress, "background task never reached in_progress");

    // Cancel: aborts the task, returns the object as cancelled.
    let c = h.post(&format!("/v1/responses/{id}/cancel"), &[], Bytes::new()).await;
    assert_eq!(c.status, 200, "cancel body={}", c.text());
    assert_eq!(c.json().get("status").and_then(Value::as_str), Some("cancelled"));

    // A later GET still reports cancelled (the aborted task never completed it).
    tokio::task::yield_now().await;
    let got = h.get(&format!("/v1/responses/{id}")).await;
    assert_eq!(got.json().get("status").and_then(Value::as_str), Some("cancelled"));
}

// ── durability + corruption ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn store_survives_app_rebuild() {
    let dir = tmp_dir("rebuild");
    let id = {
        let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "persist me"))).into_arc());
        let resp = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("save"), true)).await;
        resp.json().get("id").and_then(Value::as_str).unwrap().to_string()
    };
    // Rebuild a brand-new app (new id source, new trace sink) on the same data dir.
    let h2 = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "x"))).into_arc());
    let got = h2.get(&format!("/v1/responses/{id}")).await;
    assert_eq!(got.status, 200, "rebuilt-app get body={}", got.text());
    assert_eq!(got.json().get("id").and_then(Value::as_str), Some(id.as_str()));
    assert_eq!(got.json().get("status").and_then(Value::as_str), Some("completed"));
}

#[tokio::test]
async fn corrupt_store_file_is_500_others_unaffected() {
    let dir = tmp_dir("corrupt");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi"))).into_arc());
    // Two good stored responses.
    let a = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("one"), true)).await;
    let b = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("two"), true)).await;
    let id_a = a.json().get("id").and_then(Value::as_str).unwrap().to_string();
    let id_b = b.json().get("id").and_then(Value::as_str).unwrap().to_string();

    // Corrupt A's file on disk.
    let path = dir.join("store").join(format!("{id_a}.json"));
    std::fs::write(&path, b"{ this is not valid json ").unwrap();

    let got_a = h.get(&format!("/v1/responses/{id_a}")).await;
    assert_eq!(got_a.status, 500, "corrupt get body={}", got_a.text());
    let rec = h.trace(&got_a);
    assert!(rec.errors.iter().any(|e| e.stage == "store.get" && e.status == 500), "500 store.get error recorded");

    // B is unaffected.
    let got_b = h.get(&format!("/v1/responses/{id_b}")).await;
    assert_eq!(got_b.status, 200, "unaffected get body={}", got_b.text());
    assert_eq!(got_b.json().get("id").and_then(Value::as_str), Some(id_b.as_str()));
}

// ── sidecar-backed replay ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn sidecar_replays_anthropic_thinking_via_chat_client() {
    let dir = tmp_dir("replay");
    // Turn 1: a Chat client; the Anthropic upstream returns thinking (signature) + a tool call.
    let turn1 = reasoning_and_tool("claude-sonnet-5", ProviderFamily::Anthropic, "toolu_replay");
    let (up, raw) = RecUpstream::new(MockUpstream::always(respond_full(Protocol::Anthropic, &turn1)));
    let h = build(&dir, up);

    let b1 = json!({"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "use a tool"}]});
    let r1 = h.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&b1).unwrap())).await;
    assert_eq!(r1.status, 200, "turn1 body={}", r1.text());

    // The sidecar learned the blob keyed by the tool call.
    let key = SidecarKey::new(ProviderFamily::Anthropic, "claude-sonnet-5", CallId::new("toolu_replay"));
    assert!(h.sidecar.get(&key).is_some(), "sidecar learned the thinking blob on turn 1");

    // Turn 2: the Chat transcript echoes the tool call + result but carries no thinking. The
    // router must replay the sidecar blob into the upstream Anthropic request.
    // `reasoning_effort` marks the turn as thinking-enabled and the transcript ends at the tool
    // result (an active continuation), so Anthropic requires the thinking block replayed.
    let b2 = json!({
        "model": "claude-sonnet-5",
        "reasoning_effort": "high",
        "messages": [
            {"role": "user", "content": "use a tool"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "toolu_replay", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "toolu_replay", "content": "72F"}
        ]
    });
    let r2 = h.post("/v1/chat/completions", &[], Bytes::from(serde_json::to_vec(&b2).unwrap())).await;
    assert_eq!(r2.status, 200, "turn2 body={}", r2.text());

    // The turn-2 upstream request must carry the replayed signature blob.
    let up2 = String::from_utf8_lossy(&raw.lock().unwrap()[1]).into_owned();
    assert!(up2.contains("UkVBU09OQkxPQg=="), "replayed thinking signature missing from upstream:\n{up2}");
}

// ── FileStore / FileSidecar units ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn filestore_put_get_chain_delete() {
    let dir = tmp_dir("fsunit");
    let store = FileStore::new(dir.clone()).unwrap();
    let s1 = stored("resp_a", None);
    let s2 = stored("resp_b", Some("resp_a"));
    store.put(s1.clone());
    store.put(s2.clone());

    assert_eq!(store.get(&ResponseId::new("resp_a")), Some(s1.clone()));
    assert_eq!(store.list(), vec!["resp_a".to_string(), "resp_b".to_string()]);

    let chain = store.chain(&ResponseId::new("resp_b")).unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].id.as_str(), "resp_a"); // oldest first
    assert_eq!(chain[1].id.as_str(), "resp_b");

    // Missing chain link → NotFound.
    let err = store.chain(&ResponseId::new("resp_missing")).unwrap_err();
    assert_eq!(err.kind, llm_xlate_core::error::ErrorKind::NotFound);

    assert!(store.delete(&ResponseId::new("resp_a")));
    assert!(!store.delete(&ResponseId::new("resp_a")));
    assert!(store.get(&ResponseId::new("resp_a")).is_none());
}

#[tokio::test]
async fn filestore_get_checked_distinguishes_corrupt_from_missing() {
    let dir = tmp_dir("fschecked");
    let store = FileStore::new(dir.clone()).unwrap();
    store.put(stored("resp_ok", None));
    // Missing → Ok(None).
    assert!(store.get_checked(&ResponseId::new("resp_gone")).unwrap().is_none());
    // Corrupt → Err.
    std::fs::write(dir.join("store").join("resp_ok.json"), b"garbage").unwrap();
    assert!(store.get_checked(&ResponseId::new("resp_ok")).is_err());
    // Unsafe id → treated as absent, never a path escape.
    assert!(store.get_checked(&ResponseId::new("../escape")).unwrap().is_none());
}

#[tokio::test]
async fn filesidecar_put_get_len() {
    let dir = tmp_dir("scunit");
    let sc = FileSidecar::new(dir.clone()).unwrap();
    assert!(sc.is_empty());
    let key = SidecarKey::new(ProviderFamily::Anthropic, "claude-sonnet-5", CallId::new("toolu_1"));
    let blob = OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "c2ln");
    sc.put(key.clone(), blob.clone());
    assert_eq!(sc.len(), 1);
    assert_eq!(sc.get(&key), Some(blob));
    // A different call id is a distinct entry.
    let other = SidecarKey::new(ProviderFamily::Anthropic, "claude-sonnet-5", CallId::new("toolu_2"));
    assert!(sc.get(&other).is_none());
}

#[tokio::test]
async fn get_success_records_store_block() {
    let dir = tmp_dir("traceblock");
    let h = build(&dir, MockUpstream::always(respond_full(Protocol::OaiResponses, &text_response("gpt-4o", "hi"))).into_arc());
    let resp = h.post("/v1/responses", &[], responses_body("gpt-4o", json!("hi"), true)).await;
    let id = resp.json().get("id").and_then(Value::as_str).unwrap().to_string();
    let got = h.get(&format!("/v1/responses/{id}")).await;
    let rec = h.trace(&got);
    assert_eq!(rec.store.as_ref().and_then(|s| s.stored_id.clone()), Some(id));
    assert_eq!(rec.client.method, "GET");
}

// ── helpers for the fixtures above ─────────────────────────────────────────────────────────────

/// An [`IrResponse`] with an opaque-carrying reasoning item that precedes a tool call.
fn reasoning_and_tool(model: &str, family: ProviderFamily, call_id: &str) -> IrResponse {
    let kind = match family {
        ProviderFamily::Anthropic => OpaqueKind::Signature,
        _ => OpaqueKind::Encrypted,
    };
    IrResponse {
        id: ResponseId::new("resp_upstream"),
        model: model.to_string(),
        items: vec![
            Item::Reasoning(ReasoningItem {
                text: Some("let me think".to_string()),
                summary: vec!["let me think".to_string()],
                opaque: Some(OpaqueBlob::new(family, kind, "UkVBU09OQkxPQg==")),
                id: Some(ItemId::new("rs_up_1")),
            }),
            Item::ToolCall {
                call_id: CallId::new(call_id),
                name: "get_weather".to_string(),
                arguments: llm_xlate_core::ir::JsonText::new("{}"),
                id: Some(ItemId::new("fc_up_1")),
            },
        ],
        stop: StopReason::ToolUse,
        usage: Usage { input: 30, output: 15, reasoning: Some(8), ..Default::default() },
        ext: Default::default(),
    }
}

fn full_resp_bytes(model: &str, text: &str) -> Bytes {
    llm_xlate_testrouter::test_support::full_body_for(Protocol::OaiResponses, &text_response(model, text))
}

/// A minimal [`llm_xlate::store::StoredResponse`] for the FileStore unit tests.
fn stored(id: &str, previous: Option<&str>) -> llm_xlate::store::StoredResponse {
    llm_xlate::store::StoredResponse {
        id: ResponseId::new(id),
        previous_id: previous.map(ResponseId::new),
        instructions: Vec::new(),
        request_items: vec![Item::user_text("hi")],
        output_items: vec![Item::assistant_text("yo")],
        usage: Usage::new(1, 1),
        stop: StopReason::EndTurn,
        status: llm_xlate::store::StoredStatus::Completed,
        binding: llm_xlate::store::BackendBinding::new("openai", ProviderFamily::OpenAI, "gpt-4o"),
        created_at: 1_700_000_000,
        request_echo: serde_json::Map::new(),
        error: None,
    }
}

// A tiny convenience so `respond_full(...)` can flow into an `Arc<dyn Upstream>`.
trait IntoArcUpstream {
    fn into_arc(self) -> Arc<dyn Upstream>;
}
impl IntoArcUpstream for MockUpstream {
    fn into_arc(self) -> Arc<dyn Upstream> {
        Arc::new(self)
    }
}
