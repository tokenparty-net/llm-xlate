//! `trace show` + record location, driven through the library entry points (plan §6, R3).

mod common;

use common::*;
use llm_xlate_core::ir::Protocol;
use llm_xlate_testrouter::tracecli::{self, Section};

#[test]
fn section_parse_accepts_all_names_and_rejects_junk() {
    for s in ["client", "ir", "route", "upstream", "events", "frames", "errors", "checks", "all"] {
        assert!(Section::parse(s).is_ok(), "{s} should parse");
    }
    assert!(Section::parse("bogus").is_err());
}

#[tokio::test]
async fn show_all_renders_every_section() {
    let rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    let out = tracecli::show(&rec, Section::All, false);
    assert!(out.contains("== client =="));
    assert!(out.contains("== ir_request =="));
    assert!(out.contains("== route =="));
    assert!(out.contains("== upstream =="));
    assert!(out.contains("== checks =="));
    assert!(out.contains(&rec.trace_id));
}

#[tokio::test]
async fn show_single_section_is_scoped() {
    let rec = text_trace(Protocol::OaiChat, "gpt-4o", Protocol::OaiChat, false, &[("x-xlate-upstream", "chat")]).await;
    let out = tracecli::show(&rec, Section::Route, false);
    assert!(out.contains("== route =="));
    assert!(!out.contains("== upstream =="), "route section must not print upstream");
}

#[tokio::test]
async fn show_frames_lists_streamed_frames_one_per_line() {
    let rec = text_trace(Protocol::OaiChat, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
    let out = tracecli::show(&rec, Section::Frames, false);
    assert!(out.contains("== client frames =="));
    // Each frame line is prefixed with a `[…ms]` offset.
    assert!(out.contains("ms]"), "frames should be timestamped: {out}");
    // More than one line under the header.
    let frame_lines = out.lines().filter(|l| l.contains("ms]")).count();
    assert!(frame_lines >= 1);
}

#[tokio::test]
async fn show_events_lists_ir_events() {
    let rec = text_trace(Protocol::Anthropic, "claude-sonnet-5", Protocol::Anthropic, true, &[]).await;
    let out = tracecli::show(&rec, Section::Events, false);
    assert!(out.contains("== ir_events =="));
    assert!(out.contains("ir_response:"));
}

#[tokio::test]
async fn show_raw_dumps_the_whole_record() {
    let rec = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    let out = tracecli::show(&rec, Section::All, true);
    let v: serde_json::Value = serde_json::from_str(&out).expect("raw show is valid JSON");
    assert_eq!(v["trace_id"].as_str(), Some(rec.trace_id.as_str()));
    assert_eq!(v["v"].as_u64(), Some(1));
}

#[tokio::test]
async fn locate_by_id_and_by_path_and_by_path_line() {
    let rec_a = text_trace(Protocol::OaiResponses, "gpt-4o", Protocol::OaiResponses, false, &[]).await;
    let rec_b = reasoning_trace().await;
    let data_dir = tmp_dir("locate");
    let traces = write_session(&data_dir, &[rec_a.clone(), rec_b.clone()]);

    // By id.
    let found = tracecli::find_by_id(&traces, &rec_a.trace_id).unwrap();
    assert_eq!(found.trace_id, rec_a.trace_id);

    // load_all_records returns both, in file order.
    let all = tracecli::load_all_records(&traces).unwrap();
    assert_eq!(all.len(), 2);

    // By path (first record).
    let file = traces.join("session.jsonl");
    let by_path = tracecli::load_target(&traces, file.to_str().unwrap()).unwrap();
    assert_eq!(by_path.trace_id, rec_a.trace_id);

    // By path#line (second record, 1-indexed).
    let target = format!("{}#2", file.to_str().unwrap());
    let by_line = tracecli::load_target(&traces, &target).unwrap();
    assert_eq!(by_line.trace_id, rec_b.trace_id);

    // Unknown id errors.
    assert!(tracecli::find_by_id(&traces, "tr_nope").is_err());

    let _ = std::fs::remove_dir_all(&data_dir);
}
