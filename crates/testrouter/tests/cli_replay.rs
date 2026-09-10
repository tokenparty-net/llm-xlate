//! `trace replay` — the no-spend regression tool, driven through the library entry point (plan §6,
//! R3). Replay over a freshly-produced trace must be a no-diff.

mod common;

use common::*;
use llm_xlate_core::ir::Protocol;

use llm_xlate_testrouter::config::Config;
use llm_xlate_testrouter::tracecli;

async fn assert_replay_clean(rec: &llm_xlate_testrouter::trace::TraceRecord) {
    let outcome = tracecli::replay(rec, Config::example()).await.expect("replay");
    assert!(
        outcome.matched(),
        "replay diverged for {}: upstream_diff={:?} client_diff={:?} ({} vs {})",
        rec.trace_id,
        outcome.upstream_diff,
        outcome.client_diff,
        outcome.client_status,
        outcome.recorded_status
    );
    // The render says NO DIFF.
    assert!(outcome.render().contains("NO DIFF"));
}

#[tokio::test]
async fn replay_chat_nonstream_same_surface() {
    let rec = text_trace(Protocol::OaiChat, "gpt-4o", Protocol::OaiChat, false, &[("x-xlate-upstream", "chat")]).await;
    assert_replay_clean(&rec).await;
}

#[tokio::test]
async fn replay_responses_nonstream() {
    let rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    assert_replay_clean(&rec).await;
}

#[tokio::test]
async fn replay_anthropic_stream() {
    let rec = text_trace(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
    assert_replay_clean(&rec).await;
}

#[tokio::test]
async fn replay_chat_client_anthropic_upstream_stream() {
    let rec = text_trace(Protocol::OaiChat, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
    assert_replay_clean(&rec).await;
}

#[tokio::test]
async fn replay_responses_client_responses_stream() {
    let rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, true, &[]).await;
    assert_replay_clean(&rec).await;
}

#[tokio::test]
async fn replay_reasoning_stream() {
    let rec = reasoning_trace().await;
    assert_replay_clean(&rec).await;
}

#[tokio::test]
async fn replay_is_deterministic_across_two_runs() {
    let rec = text_trace(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
    let a = tracecli::replay(&rec, Config::example()).await.unwrap();
    let b = tracecli::replay(&rec, Config::example()).await.unwrap();
    assert!(a.matched() && b.matched());
    assert_eq!(a.client_diff, b.client_diff);
    assert_eq!(a.upstream_diff, b.upstream_diff);
}

#[tokio::test]
async fn replay_detects_a_tampered_upstream_body() {
    // Corrupt the recorded upstream body: replay must now report an upstream diff (the produced
    // body no longer matches the recorded one).
    let mut rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    if let Some(up) = rec.upstream.as_mut() {
        if let Some(body) = up.body.as_mut() {
            body["model"] = serde_json::json!("tampered-model-xyz");
        }
    }
    let outcome = tracecli::replay(&rec, Config::example()).await.unwrap();
    assert!(!outcome.matched(), "tampering should surface a diff");
    assert!(!outcome.upstream_diff.is_empty(), "expected an upstream body diff");
}

#[tokio::test]
async fn replay_refuses_media_redacted_body() {
    let mut cfg = Config::example();
    cfg.trace.redact_media_over_bytes = 64;
    let big = "/9j/".to_string() + &"A".repeat(500);
    let body = serde_json::json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "x"},
            {"type": "input_image", "image_url": format!("data:image/jpeg;base64,{big}")}
        ]}]
    });
    let tr = llm_xlate_testrouter::test_support::TestRouter::build(
        cfg,
        llm_xlate_testrouter::test_support::MockUpstream::always(with_request_id(
            llm_xlate_testrouter::test_support::respond_full(
                Protocol::OaiResponses,
                &llm_xlate_testrouter::test_support::text_response("gpt-4o", "ok"),
            ),
        )),
    );
    let resp = tr.post("/v1/responses", &[], bytes::Bytes::from(serde_json::to_vec(&body).unwrap())).await;
    let rec = tr.trace(&resp);
    // The client body was media-redacted → replay refuses (the exact bytes are gone).
    assert!(tracecli::replay(&rec, Config::example()).await.is_err());
}
