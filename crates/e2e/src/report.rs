//! Per-run `report.md` rendering and run-vs-run `diff` (plan §8 reporting slice).
//!
//! The report totals the run (sent / ok / error / skipped, and verdict pass/fail), then prints one
//! table per protocol: probe id, model, stream, status, stop reason, verdict, hypothesis, a
//! one-line observation summary, and the on-disk path. `diff` matches captures by
//! `(probe, model, stream)` and lists only the differences: request-hash drift, and changes in
//! status / error type / stop reason / type sequence — the signal for a real API change.

use crate::capture::{ManifestEntry, Run};
use crate::observe::{Observation, Verdict};
use anyhow::Result;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// Load the `observe.json` for a manifest entry, if present.
fn load_observation(run_root: &Path, entry: &ManifestEntry) -> Option<Observation> {
    let path = run_root.join(&entry.rel_dir).join("observe.json");
    let txt = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&txt).ok()
}

/// The probe hypothesis, read from the manifest-referenced capture is not stored; the report is
/// given hypotheses by the caller through this map (probe id → hypothesis).
pub type Hypotheses = BTreeMap<String, String>;

/// Render `report.md` for a run. `hypotheses` supplies the free-text hypothesis per probe id.
pub fn render_report(run: &Run, hypotheses: &Hypotheses) -> Result<String> {
    let m = &run.manifest;
    let mut out = String::new();

    writeln!(out, "# Run `{}`", m.run_id)?;
    writeln!(out)?;
    writeln!(out, "- label: `{}`", m.label)?;
    writeln!(out, "- created: {}", m.created_at)?;
    writeln!(out, "- tool: llm-xlate-e2e {}", m.tool_version)?;
    writeln!(out, "- llm-xlate: {}", m.xlate_version)?;
    if !m.providers.is_empty() {
        let provs: Vec<String> = m
            .providers
            .iter()
            .map(|(k, v)| format!("{k} ({})", v.base_url))
            .collect();
        writeln!(out, "- providers: {}", provs.join(", "))?;
    }
    if !m.model_sets.is_empty() {
        writeln!(out, "- model sets: {}", m.model_sets.join(", "))?;
    }
    writeln!(out)?;

    // Totals.
    let mut sent = 0usize;
    let mut ok = 0usize;
    let mut error = 0usize;
    let mut skipped = 0usize;
    let mut pass = 0usize;
    let mut fail = 0usize;
    for e in &m.entries {
        if e.skipped.is_some() {
            skipped += 1;
            continue;
        }
        sent += 1;
        match e.status {
            Some(s) if (200..300).contains(&s) => ok += 1,
            Some(_) => error += 1,
            None => {}
        }
        if let Some(o) = load_observation(&run.root, e) {
            match o.verdict {
                Verdict::Pass => pass += 1,
                Verdict::Fail => fail += 1,
                Verdict::Na => {}
            }
        }
    }
    writeln!(out, "## Totals")?;
    writeln!(out)?;
    writeln!(
        out,
        "sent **{sent}** · ok **{ok}** · error **{error}** · skipped **{skipped}** · verdict pass **{pass}** / fail **{fail}**"
    )?;
    writeln!(out)?;

    // Spend estimate (§budget). Computed from the captured usage counters and the dated price
    // table; deliberately over-estimates unknown models. Not an invoice — a spend guard.
    let tally = tally_run(run);
    writeln!(out, "## Spend (estimate, list prices dated {})", crate::pricing::PRICES_DATED)?;
    writeln!(out)?;
    if tally.by_provider.is_empty() {
        writeln!(out, "No billable usage captured.")?;
    } else {
        writeln!(out, "| provider | billed captures | estimated USD |")?;
        writeln!(out, "|---|---|---|")?;
        for (prov, (usd, n)) in &tally.by_provider {
            writeln!(out, "| {prov} | {n} | ${usd:.4} |")?;
        }
        writeln!(out, "| **total** | | **${:.4}** |", tally.total())?;
        writeln!(out)?;
        if tally.used_fallback {
            writeln!(
                out,
                "> Some captures used best-knowledge / over-estimating prices (models whose list price is unconfirmed). Actual spend is at or below these figures."
            )?;
        }
    }
    writeln!(out)?;

    // Per-protocol tables.
    for proto in ["anthropic", "chat", "responses"] {
        let rows: Vec<&ManifestEntry> = m.entries.iter().filter(|e| e.protocol == proto).collect();
        if rows.is_empty() {
            continue;
        }
        writeln!(out, "## {proto}")?;
        writeln!(out)?;
        writeln!(
            out,
            "| probe | model | stream | status | stop | verdict | hypothesis | observation | path |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|")?;
        for e in rows {
            let obs = load_observation(&run.root, e);
            let status = match (&e.skipped, e.status) {
                (Some(_), _) => "skipped".to_string(),
                (None, Some(s)) => s.to_string(),
                (None, None) => "-".to_string(),
            };
            let stop = obs
                .as_ref()
                .and_then(|o| o.stop_reason.clone())
                .unwrap_or_else(|| "-".into());
            let verdict = match (&e.skipped, obs.as_ref().map(|o| o.verdict)) {
                (Some(_), _) => "skip",
                (_, Some(Verdict::Pass)) => "pass",
                (_, Some(Verdict::Fail)) => "FAIL",
                _ => "n.a.",
            };
            let hypo = hypotheses
                .get(&e.probe_id)
                .map(|s| truncate(s, 60))
                .unwrap_or_default();
            let summary = obs
                .as_ref()
                .map(|o| o.summary_line())
                .unwrap_or_else(|| e.skipped.clone().unwrap_or_else(|| "-".into()));
            writeln!(
                out,
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | `{}` |",
                e.probe_id,
                e.model,
                if e.stream { "on" } else { "off" },
                status,
                md_escape(&stop),
                verdict,
                md_escape(&hypo),
                md_escape(&summary),
                e.rel_dir,
            )?;
        }
        writeln!(out)?;
    }
    Ok(out)
}

/// Rewrite `<runs_dir>/SPEND.md` from every run under `runs_dir`: one row per run plus per-provider
/// grand totals. Idempotent (fully regenerated each call by scanning the run manifests), so it stays
/// correct no matter how many times it runs. Returns the grand total per provider.
pub fn update_spend_ledger(runs_dir: &Path) -> Result<BTreeMap<String, f64>> {
    let mut rows: Vec<(String, String, crate::pricing::SpendTally)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(runs_dir) {
        let mut dirs: Vec<std::path::PathBuf> =
            rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for d in dirs {
            if d.join("manifest.json").exists() {
                if let Ok(run) = Run::load(&d) {
                    let created = run.manifest.created_at.clone();
                    rows.push((run.manifest.run_id.clone(), created, tally_run(&run)));
                }
            } else if d.join("translate").is_dir() {
                // A `translate` run has no top-level manifest; its live sends live under
                // `translate/<scenario>/upstream_response.json`. Bill them too so the ledger stays
                // complete (plan: spend estimated after every live run).
                let id = d.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
                let created = std::fs::metadata(&d)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|dur| {
                        chrono::DateTime::<chrono::Utc>::from_timestamp(dur.as_secs() as i64, 0)
                            .map(|dt| dt.to_rfc3339())
                            .unwrap_or_default()
                    })
                    .unwrap_or_default();
                rows.push((id, created, tally_translate_run(&d.join("translate"))));
            }
        }
    }

    let mut grand: BTreeMap<String, f64> = BTreeMap::new();
    let mut providers: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (_, _, t) in &rows {
        for (p, (usd, _)) in &t.by_provider {
            *grand.entry(p.clone()).or_default() += usd;
            providers.insert(p.clone());
        }
    }

    let mut out = String::new();
    writeln!(out, "# SPEND — running estimate for `llm-xlate-e2e` live runs")?;
    writeln!(out)?;
    writeln!(
        out,
        "Auto-generated from every `runs/*/manifest.json` and `runs/*/translate/*/upstream_response.json` (usage counters × the dated price table in `src/pricing.rs`, list prices dated {}). Over-estimates unknown models; a spend guard, not an invoice. Cap: **$8 per provider** for this task.",
        crate::pricing::PRICES_DATED
    )?;
    writeln!(out)?;
    writeln!(out, "## Grand total per provider")?;
    writeln!(out)?;
    writeln!(out, "| provider | estimated USD | cap |")?;
    writeln!(out, "|---|---|---|")?;
    for p in &providers {
        writeln!(out, "| {p} | ${:.4} | $8.00 |", grand.get(p).copied().unwrap_or(0.0))?;
    }
    if providers.is_empty() {
        writeln!(out, "| (none yet) | $0.0000 | $8.00 |")?;
    }
    writeln!(out)?;
    writeln!(out, "## Per run")?;
    writeln!(out)?;
    writeln!(out, "| run | created | provider | billed captures | estimated USD |")?;
    writeln!(out, "|---|---|---|---|---|")?;
    for (id, created, t) in &rows {
        if t.by_provider.is_empty() {
            writeln!(out, "| `{id}` | {created} | — | 0 | $0.0000 |")?;
        }
        for (p, (usd, n)) in &t.by_provider {
            writeln!(out, "| `{id}` | {created} | {p} | {n} | ${usd:.4} |")?;
        }
    }
    writeln!(out)?;

    std::fs::write(runs_dir.join("SPEND.md"), out)?;
    Ok(grand)
}

/// Accumulate the per-provider spend estimate for a run from its captures' `usage` counters.
pub fn tally_run(run: &Run) -> crate::pricing::SpendTally {
    let mut t = crate::pricing::SpendTally::default();
    for e in &run.manifest.entries {
        if e.skipped.is_some() {
            continue;
        }
        if let Some(o) = load_observation(&run.root, e) {
            t.add(&e.model, &o.usage);
        }
    }
    t
}

/// Accumulate the per-provider spend estimate for a `translate` run from each scenario's
/// `upstream_request.json` (target model + body shape) and its upstream response. A non-streaming
/// send stores `upstream_response.json` (`body.usage`); a streaming send stores
/// `upstream_response.sse` instead, whose usage is aggregated the same way `observe.rs` does (the
/// final `message_delta` for Anthropic, the `response.completed` usage for Responses, the trailing
/// usage-only chunk for Chat). Skipped scenarios (only `degradations.txt`) and error responses
/// (no `usage`) contribute nothing.
pub fn tally_translate_run(translate_dir: &Path) -> crate::pricing::SpendTally {
    let mut t = crate::pricing::SpendTally::default();
    let Ok(rd) = std::fs::read_dir(translate_dir) else {
        return t;
    };
    let mut dirs: Vec<std::path::PathBuf> =
        rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    for d in dirs {
        let req = std::fs::read_to_string(d.join("upstream_request.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let model = req
            .as_ref()
            .and_then(|v| v.get("body").and_then(|b| b.get("model")).and_then(|m| m.as_str()).map(str::to_string));
        let Some(model) = model else { continue };

        // Non-streaming: usage from the JSON response body.
        let usage = std::fs::read_to_string(d.join("upstream_response.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("body").and_then(|b| b.get("usage")).cloned())
            .and_then(|u| u.as_object().cloned())
            .map(|u| u.into_iter().collect::<BTreeMap<String, serde_json::Value>>());

        let usage = usage.or_else(|| {
            // Streaming: aggregate usage from the captured SSE the same way the observer does.
            let raw = std::fs::read(d.join("upstream_response.sse")).ok()?;
            let body = req.as_ref().and_then(|v| v.get("body"));
            let protocol = target_protocol(&model, body);
            let events = crate::capture::parse_sse(&raw);
            let obs = crate::observe::observe_stream(
                protocol,
                &serde_json::Value::Null,
                200,
                &BTreeMap::new(),
                &raw,
                &events,
                &crate::probe::Expect::Any,
            );
            if obs.usage.is_empty() {
                None
            } else {
                Some(obs.usage)
            }
        });

        if let Some(usage) = usage {
            t.add(&model, &usage);
        }
    }
    t
}

/// The wire protocol of a translate scenario's *target*, for choosing the right SSE usage
/// aggregator. Anthropic models always use the Anthropic protocol; an OpenAI target is Responses
/// when its request body carries an `input` array (the Responses shape) and Chat otherwise.
fn target_protocol(model: &str, body: Option<&serde_json::Value>) -> crate::probe::Protocol {
    use crate::probe::Protocol;
    if crate::pricing::provider_for(model) == "anthropic" {
        Protocol::Anthropic
    } else if body.and_then(|b| b.get("input")).is_some() {
        Protocol::Responses
    } else {
        Protocol::Chat
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

fn md_escape(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

// ---------------------------------------------------------------------------------------------
// Diff.
// ---------------------------------------------------------------------------------------------

/// A single (probe, model, stream) key.
type Key = (String, String, bool);

fn key(e: &ManifestEntry) -> Key {
    (e.probe_id.clone(), e.model.clone(), e.stream)
}

/// Render a markdown diff of two runs, listing only the differences.
pub fn diff_runs(a: &Run, b: &Run) -> Result<String> {
    let mut out = String::new();
    writeln!(out, "# Diff `{}` → `{}`", a.manifest.run_id, b.manifest.run_id)?;
    writeln!(out)?;

    let a_map: BTreeMap<Key, &ManifestEntry> = a.manifest.entries.iter().map(|e| (key(e), e)).collect();
    let b_map: BTreeMap<Key, &ManifestEntry> = b.manifest.entries.iter().map(|e| (key(e), e)).collect();

    let mut only_a: Vec<&Key> = a_map.keys().filter(|k| !b_map.contains_key(*k)).collect();
    let mut only_b: Vec<&Key> = b_map.keys().filter(|k| !a_map.contains_key(*k)).collect();
    only_a.sort();
    only_b.sort();

    if !only_a.is_empty() {
        writeln!(out, "## Only in A")?;
        for k in &only_a {
            writeln!(out, "- `{}` {} stream={}", k.0, k.1, k.2)?;
        }
        writeln!(out)?;
    }
    if !only_b.is_empty() {
        writeln!(out, "## Only in B")?;
        for k in &only_b {
            writeln!(out, "- `{}` {} stream={}", k.0, k.1, k.2)?;
        }
        writeln!(out)?;
    }

    let mut diffs: Vec<String> = Vec::new();
    let mut common: Vec<&Key> = a_map.keys().filter(|k| b_map.contains_key(*k)).collect();
    common.sort();
    for k in common {
        let ea = a_map[k];
        let eb = b_map[k];
        let oa = load_observation(&a.root, ea);
        let ob = load_observation(&b.root, eb);
        let mut changes: Vec<String> = Vec::new();

        if ea.request_sha256 != eb.request_sha256 {
            changes.push("request changed (hash differs)".into());
        }
        if ea.status != eb.status {
            changes.push(format!(
                "status {} → {}",
                opt_u16(ea.status),
                opt_u16(eb.status)
            ));
        }
        let (eta, etb) = (err_type(&oa), err_type(&ob));
        if eta != etb {
            changes.push(format!("error_type {} → {}", eta, etb));
        }
        let (sra, srb) = (stop(&oa), stop(&ob));
        if sra != srb {
            changes.push(format!("stop {} → {}", sra, srb));
        }
        let (tsa, tsb) = (types(&oa), types(&ob));
        if tsa != tsb {
            changes.push(format!("types [{}] → [{}]", tsa, tsb));
        }

        if !changes.is_empty() {
            diffs.push(format!(
                "- `{}` {} stream={}: {}",
                k.0,
                k.1,
                k.2,
                changes.join("; ")
            ));
        }
    }

    writeln!(out, "## Changed ({} of {} common)", diffs.len(), a_map.keys().filter(|k| b_map.contains_key(*k)).count())?;
    writeln!(out)?;
    if diffs.is_empty() {
        writeln!(out, "No differences among common captures.")?;
    } else {
        for d in diffs {
            writeln!(out, "{d}")?;
        }
    }
    Ok(out)
}

fn opt_u16(v: Option<u16>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "-".into())
}

fn err_type(o: &Option<Observation>) -> String {
    o.as_ref()
        .and_then(|x| x.error.as_ref())
        .and_then(|e| e.r#type.clone())
        .unwrap_or_else(|| "-".into())
}

fn stop(o: &Option<Observation>) -> String {
    o.as_ref()
        .and_then(|x| x.stop_reason.clone())
        .unwrap_or_else(|| "-".into())
}

fn types(o: &Option<Observation>) -> String {
    o.as_ref().map(|x| x.type_sequence.join(",")).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{Manifest, ProviderInfo};
    use std::path::PathBuf;

    fn tmpdir(name: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("llm-xlate-e2e-report-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn entry(probe: &str, proto: &str, model: &str, stream: bool, status: Option<u16>) -> ManifestEntry {
        let sfx = if stream { ".stream" } else { "" };
        ManifestEntry {
            rel_dir: format!("{probe}/{model}{sfx}"),
            probe_id: probe.into(),
            protocol: proto.into(),
            model: model.into(),
            stream,
            status,
            skipped: None,
            request_sha256: Some("aaa".into()),
            response_sha256: Some("bbb".into()),
        }
    }

    fn write_obs(root: &Path, entry: &ManifestEntry, obs: &Observation) {
        let dir = root.join(&entry.rel_dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("observe.json"), serde_json::to_string_pretty(obs).unwrap()).unwrap();
    }

    fn manifest(root: PathBuf, entries: Vec<ManifestEntry>, id: &str) -> Run {
        let m = Manifest {
            tool_version: "0.1.0".into(),
            xlate_version: "0.1.0".into(),
            run_id: id.into(),
            label: "t".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            providers: BTreeMap::from([("anthropic".into(), ProviderInfo { base_url: "https://api.anthropic.com".into() })]),
            model_sets: vec!["anthropic_all".into()],
            entries,
        };
        Run { root, manifest: m }
    }

    #[test]
    fn report_totals_and_table() {
        let root = tmpdir("report");
        let e1 = entry("ant.text", "anthropic", "claude-opus-4-8", false, Some(200));
        let e2 = entry("chat.err", "chat", "gpt-4o-mini", false, Some(401));
        let mut o1 = Observation::base_for_test("anthropic", false, 200);
        o1.verdict = Verdict::Pass;
        o1.stop_reason = Some("end_turn".into());
        o1.type_sequence = vec!["text".into()];
        write_obs(&root, &e1, &o1);
        let mut o2 = Observation::base_for_test("chat", false, 401);
        o2.verdict = Verdict::Pass;
        write_obs(&root, &e2, &o2);
        let run = manifest(root.clone(), vec![e1, e2], "20260101-000000");
        let mut hyp = Hypotheses::new();
        hyp.insert("ant.text".into(), "returns 200 text".into());
        let md = render_report(&run, &hyp).unwrap();
        assert!(md.contains("sent **2**"));
        assert!(md.contains("ok **1**"));
        assert!(md.contains("error **1**"));
        assert!(md.contains("## anthropic"));
        assert!(md.contains("`ant.text`"));
        assert!(md.contains("returns 200 text"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn diff_lists_only_changes() {
        let ra = tmpdir("diffA");
        let rb = tmpdir("diffB");
        // Common key with a stop-reason change; plus one only-in-A.
        let a1 = entry("ant.text", "anthropic", "m", false, Some(200));
        let a2 = entry("ant.only", "anthropic", "m", false, Some(200));
        let b1 = entry("ant.text", "anthropic", "m", false, Some(200));
        let mut oa = Observation::base_for_test("anthropic", false, 200);
        oa.stop_reason = Some("end_turn".into());
        write_obs(&ra, &a1, &oa);
        write_obs(&ra, &a2, &oa);
        let mut ob = Observation::base_for_test("anthropic", false, 200);
        ob.stop_reason = Some("max_tokens".into());
        write_obs(&rb, &b1, &ob);
        let run_a = manifest(ra.clone(), vec![a1, a2], "A");
        let run_b = manifest(rb.clone(), vec![b1], "B");
        let md = diff_runs(&run_a, &run_b).unwrap();
        assert!(md.contains("Only in A"));
        assert!(md.contains("ant.only"));
        assert!(md.contains("stop end_turn → max_tokens"));
        std::fs::remove_dir_all(&ra).ok();
        std::fs::remove_dir_all(&rb).ok();
    }

    #[test]
    fn translate_run_billed_from_upstream_usage() {
        // A translate run has no manifest; spend must be tallied from each scenario's target model
        // (upstream_request.body.model) and usage (upstream_response.body.usage). Error responses
        // (no usage) and skip-only dirs contribute nothing.
        let root = tmpdir("xlate-spend");
        let td = root.join("translate");
        let write = |name: &str, model: Option<&str>, usage: Option<serde_json::Value>| {
            let d = td.join(name);
            std::fs::create_dir_all(&d).unwrap();
            if let Some(m) = model {
                let req = serde_json::json!({"body": {"model": m}});
                std::fs::write(d.join("upstream_request.json"), req.to_string()).unwrap();
            }
            let mut body = serde_json::json!({});
            if let Some(u) = usage {
                body = serde_json::json!({"usage": u});
            }
            let resp = serde_json::json!({"status": 200, "body": body});
            std::fs::write(d.join("upstream_response.json"), resp.to_string()).unwrap();
        };
        write("text__chat_to_anthropic__claude-sonnet-5", Some("claude-sonnet-5"),
            Some(serde_json::json!({"input_tokens": 1_000_000, "output_tokens": 0})));
        write("text__anthropic_to_chat__gpt-4o-mini", Some("gpt-4o-mini"),
            Some(serde_json::json!({"prompt_tokens": 1_000_000, "completion_tokens": 0})));
        write("err__chat_to_anthropic__claude-sonnet-5", Some("claude-sonnet-5"), None);
        // A skip-only dir (only degradations.txt): no request/response, no bill.
        let skip = td.join("full__responses_to_anthropic__claude-sonnet-5");
        std::fs::create_dir_all(&skip).unwrap();
        std::fs::write(skip.join("degradations.txt"), "skipped").unwrap();

        let t = tally_translate_run(&td);
        // sonnet input = $3/1M → exactly $3.00 for the one priced capture; openai capture is $0.
        let (ant_usd, ant_n) = t.by_provider.get("anthropic").copied().unwrap_or((0.0, 0));
        assert_eq!(ant_n, 1, "one billed anthropic capture (the error dir has no usage)");
        assert!((ant_usd - 3.0).abs() < 1e-9, "sonnet 1M input tokens = $3.00, got {ant_usd}");
        let (oai_usd, oai_n) = t.by_provider.get("openai").copied().unwrap_or((0.0, 0));
        assert_eq!(oai_n, 1, "the gpt-4o-mini target capture is billed to openai");
        assert!(oai_usd > 0.0, "gpt-4o-mini 1M prompt tokens must cost > 0");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn translate_run_billed_from_upstream_sse_when_streamed() {
        // A streamed translate send stores `upstream_response.sse` (no `.json`); usage must still be
        // billed by aggregating the SSE the way observe.rs does (Anthropic: message_start input +
        // message_delta output).
        let root = tmpdir("xlate-spend-sse");
        let td = root.join("translate");
        let d = td.join("text__chat_to_anthropic__claude-sonnet-5");
        std::fs::create_dir_all(&d).unwrap();
        let req = serde_json::json!({"body": {"model": "claude-sonnet-5", "messages": []}});
        std::fs::write(d.join("upstream_request.json"), req.to_string()).unwrap();
        // 1M input tokens on sonnet ($3/1M) → exactly $3.00; no upstream_response.json is written.
        let sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1000000,\"output_tokens\":0}}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n"
        );
        std::fs::write(d.join("upstream_response.sse"), sse).unwrap();

        let t = tally_translate_run(&td);
        let (ant_usd, ant_n) = t.by_provider.get("anthropic").copied().unwrap_or((0.0, 0));
        assert_eq!(ant_n, 1, "the streamed sonnet capture is billed from its SSE");
        assert!((ant_usd - 3.0).abs() < 1e-9, "sonnet 1M input tokens = $3.00, got {ant_usd}");
        std::fs::remove_dir_all(&root).ok();
    }
}
