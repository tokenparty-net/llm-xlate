//! `trace export` → llm-xlate-e2e capture, driven through the library entry points (plan §6, R3).
//! Both exported legs must pass `llm_xlate_e2e::xlate_check`.

mod common;

use common::*;
use llm_xlate_core::ir::Protocol;

use llm_xlate_e2e::capture::{Capture, Run};
use llm_xlate_e2e::xlate_check::{check_dir, ClientFilter};
use llm_xlate_testrouter::tracecli;

/// Export a record and assert the run is loadable and passes xlate_check with no failed items.
fn export_and_check(rec: &llm_xlate_testrouter::trace::TraceRecord, tag: &str) {
    let dir = tmp_dir(tag);
    let out = tracecli::export(rec, &dir).expect("export");
    assert_eq!(out.root, dir);
    assert_ne!(out.client_rel, out.upstream_rel, "the two legs must not collide");

    // The run loads and lists both captures.
    let run = Run::load(&dir).expect("run manifest loads");
    assert_eq!(run.manifest.entries.len(), 2);
    let caps = run.captures().expect("captures load");
    assert_eq!(caps.len(), 2);
    // Each capture directory has request.json + observe.json.
    for c in &caps {
        assert!(c.dir.join("request.json").is_file());
        assert!(c.dir.join("observe.json").is_file());
    }

    // xlate_check passes: no request_decode / response_decode failures on either leg.
    let summary = check_dir(&dir, ClientFilter::All, None).expect("check");
    assert_eq!(summary.cases.len(), 2, "one case per leg");
    for c in &summary.cases {
        for item in &c.items {
            if item.name.contains("decode") {
                assert!(item.pass, "{}/{} failed: {}", c.label, item.name, item.detail);
            }
        }
    }
    assert!(!summary.any_failed(), "xlate_check failures: {:?}", summary.failures());

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn export_nonstream_same_surface() {
    let rec = text_trace(Protocol::OaiChat, "gpt-4o", Protocol::OaiChat, false, &[("x-xlate-upstream", "chat")]).await;
    export_and_check(&rec, "exp-ns-same");
}

#[tokio::test]
async fn export_stream_cross_surface() {
    // Chat client, Anthropic upstream, streaming: both legs still export and check.
    let rec = text_trace(Protocol::OaiChat, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
    export_and_check(&rec, "exp-st-cross");
}

#[tokio::test]
async fn export_responses_nonstream() {
    let rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    export_and_check(&rec, "exp-resp-ns");
}

#[tokio::test]
async fn export_anthropic_stream() {
    let rec = text_trace(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
    export_and_check(&rec, "exp-ant-st");
}

#[tokio::test]
async fn export_preserves_redacted_headers() {
    // A secret header on the client leg is exported already redacted.
    let tr = llm_xlate_testrouter::test_support::TestRouter::example(
        llm_xlate_testrouter::test_support::MockUpstream::always(with_request_id(
            llm_xlate_testrouter::test_support::respond_full(
                Protocol::OaiResponses,
                &llm_xlate_testrouter::test_support::text_response("gpt-4o", "hi"),
            ),
        )),
    );
    let hdrs: &[(&str, &str)] = &[("authorization", "Bearer sk-shouldnotappear12345678")];
    let resp = tr
        .post(
            "/v1/responses",
            hdrs,
            llm_xlate_testrouter::test_support::simple_request(Protocol::OaiResponses, "gpt-4o", "hi", false),
        )
        .await;
    let rec = tr.trace(&resp);

    let dir = tmp_dir("exp-redact");
    let out = tracecli::export(&rec, &dir).unwrap();
    let client = Capture::load(&dir.join(&out.client_rel)).unwrap();
    let auth = client.request.headers.get("authorization");
    assert_eq!(auth.map(String::as_str), Some("<redacted>"));
    // The secret never lands anywhere on disk.
    let session = std::fs::read_to_string(dir.join(&out.client_rel).join("request.json")).unwrap();
    assert!(!session.contains("sk-shouldnotappear"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn export_without_upstream_leg_errors() {
    // A record whose upstream never ran (auth failure before send) cannot export the upstream leg.
    // A token-auth rejection fails at ingress with no upstream; simulate by clearing the upstream.
    let mut rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    rec.upstream = None;
    let dir = tmp_dir("exp-noup");
    assert!(tracecli::export(&rec, &dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
