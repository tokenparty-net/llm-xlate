//! `trace triage`, driven through the library entry points (plan §6, R3).

mod common;

use common::*;
use llm_xlate_core::ir::Protocol;
use llm_xlate_testrouter::tracecli::{self, TriageFilter};

#[tokio::test]
async fn clean_session_has_no_anomalies() {
    let recs = vec![
        text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await,
        text_trace(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await,
        reasoning_trace().await,
    ];
    let report = tracecli::triage(&recs, &TriageFilter::default());
    assert_eq!(report.total_rows, 3);
    assert_eq!(report.anomalous_rows, 0, "clean rows: {:?}", report.rows);
    assert!(!report.has_anomalies());
}

#[tokio::test]
async fn failing_upstream_is_flagged_once() {
    let recs = vec![
        text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await,
        failing_trace().await,
    ];
    let report = tracecli::triage(&recs, &TriageFilter::default());
    assert_eq!(report.anomalous_rows, 1);
    assert!(report.has_anomalies());
    // The anomaly reason mentions an error.
    let flagged = report.rows.iter().find(|r| r.is_anomalous()).unwrap();
    assert!(flagged.anomalies.iter().any(|a| a.starts_with("error:")), "{:?}", flagged.anomalies);
    // The auth error is a passthrough, not an xlate bug.
    assert!(!flagged.anomalies.iter().any(|a| a == "xlate_error"));
}

#[tokio::test]
async fn unknown_caps_is_an_anomaly() {
    let tr = llm_xlate_testrouter::test_support::TestRouter::example(
        llm_xlate_testrouter::test_support::MockUpstream::always(with_request_id(
            llm_xlate_testrouter::test_support::respond_full(
                Protocol::OaiChat,
                &llm_xlate_testrouter::test_support::text_response("zzz", "hi"),
            ),
        )),
    );
    let hdrs: &[(&str, &str)] = &[("x-xlate-provider", "mystery"), ("x-xlate-upstream", "chat")];
    let resp = tr
        .post(
            "/v1/chat/completions",
            hdrs,
            llm_xlate_testrouter::test_support::simple_request(Protocol::OaiChat, "zzz-model", "hi", false),
        )
        .await;
    let rec = tr.trace(&resp);
    let report = tracecli::triage(std::slice::from_ref(&rec), &TriageFilter::default());
    assert!(report.rows[0].anomalies.iter().any(|a| a == "caps_unknown"), "{:?}", report.rows[0].anomalies);
}

#[tokio::test]
async fn table_and_json_render() {
    let recs = vec![
        text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await,
        failing_trace().await,
    ];
    let report = tracecli::triage(&recs, &TriageFilter::default());
    let table = report.to_table();
    assert!(table.contains("trace"));
    assert!(table.contains("anomalous"));
    let json = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["total_rows"].as_u64(), Some(2));
    assert_eq!(v["anomalous_rows"].as_u64(), Some(1));
}

#[tokio::test]
async fn tag_filter_selects_only_matching() {
    // Build two traces with different tags.
    let a = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[("x-xlate-tag", "keep")]).await;
    let b = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[("x-xlate-tag", "drop")]).await;
    let recs = vec![a, b];
    let report = tracecli::triage(&recs, &TriageFilter { since: None, tag: Some("keep".into()) });
    assert_eq!(report.total_rows, 1);
    assert_eq!(report.rows[0].tag.as_deref(), Some("keep"));
}

#[tokio::test]
async fn since_filter_drops_older() {
    let rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    // The fixed clock stamps 2023-11-14…; a later `since` excludes it.
    let future = TriageFilter { since: Some("2099-01-01T00:00:00Z".into()), tag: None };
    assert_eq!(tracecli::triage(std::slice::from_ref(&rec), &future).total_rows, 0);
    let past = TriageFilter { since: Some("2000-01-01T00:00:00Z".into()), tag: None };
    assert_eq!(tracecli::triage(std::slice::from_ref(&rec), &past).total_rows, 1);
}

#[tokio::test]
async fn anomaly_counts_are_categorised() {
    let recs = vec![failing_trace().await];
    let report = tracecli::triage(&recs, &TriageFilter::default());
    // The `error:<kind>` reason is bucketed under the `error` key.
    assert!(report.anomaly_counts.contains_key("error"));
}
