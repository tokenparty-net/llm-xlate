//! The trace CLI (plan §6, milestone R3): `show`, `triage`, `export`, and `replay`, implemented as
//! library entry points so both `xlate-testrouter trace …` (in `main.rs`) and the crate's tests
//! drive the same code without spawning a binary.
//!
//! * [`show`] locates a record by id (via `traces/index.jsonl`) or by a direct `path`/`path#line`
//!   and pretty-prints the chosen sections.
//! * [`triage`] scans records for anomalies (errors, failed invariant checks, unknown caps,
//!   keepalive-only streams, slow TTFB, cancellations, missing upstream request ids) and renders a
//!   table or JSON; the caller exits non-zero when [`TriageReport::has_anomalies`].
//! * [`export`] writes TWO `llm-xlate-e2e` captures — the client leg and the upstream leg — into a
//!   run directory so `llm-xlate-e2e check`/`promote` work on real-tool traffic unchanged.
//! * [`replay`] rebuilds the [`App`] with the recorded route overrides, a mock upstream
//!   that serves the recorded raw bytes, and ids seeded from the recorded ids, re-sends the recorded
//!   client request, and diffs the produced upstream body + client frames against the recorded ones.
//!   **This is the no-spend regression tool**: every later `llm-xlate` change is re-verified against
//!   the exact recorded upstream bytes with zero API cost.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use serde::Serialize;
use serde_json::{Map, Value};

use llm_xlate::Translator;
use llm_xlate_core::codec::TranslatorConfig;

use llm_xlate_e2e::capture::{
    self, parse_sse, EventRecord, Manifest, RequestRecord, WireResponse,
};
use llm_xlate_e2e::observe::{observe_nonstream, observe_stream};
use llm_xlate_e2e::probe::{Catalogue, Expansion, Expect, Protocol as ProbeProtocol};

use crate::config::Config;
use crate::ids::{FixedClock, IdSource};
use crate::sidecar::MemorySidecar;
use crate::store::MemoryStore;
use crate::trace::{ClientResponse, NullTraceSink, TraceIndex, TraceRecord, UpstreamTrace};
use crate::upstream::{
    BoxFuture, ByteStream, Upstream, UpstreamBody, UpstreamError, UpstreamRequest, UpstreamResponse,
};
use crate::{App, Deps};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Locating records
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The `traces/` directory inside a data dir (`data_dir/traces`).
pub fn traces_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("traces")
}

/// Load every record from every `*.jsonl` (excluding `index.jsonl`) under a `traces/` directory,
/// oldest file first. Used by `triage` and by id lookup fallback.
pub fn load_all_records(traces_dir: &Path) -> Result<Vec<TraceRecord>> {
    let mut files: Vec<PathBuf> = Vec::new();
    if traces_dir.is_dir() {
        for e in std::fs::read_dir(traces_dir)
            .with_context(|| format!("reading {}", traces_dir.display()))?
        {
            let p = e?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("jsonl")
                && p.file_name().and_then(|s| s.to_str()) != Some("index.jsonl")
            {
                files.push(p);
            }
        }
    }
    files.sort();
    let mut out = Vec::new();
    for f in files {
        out.extend(TraceRecord::read_all(&f)?);
    }
    Ok(out)
}

/// Resolve a `show`/`export`/`replay` target — a trace id, a `path`, or a `path#line` — to a record.
///
/// A bare id is looked up first via `traces/index.jsonl` (to find its file) and then by scanning
/// every trace file; a `path` loads its first record (or the record at `#line`, 1-indexed among the
/// file's non-empty lines).
pub fn load_target(traces_dir: &Path, target: &str) -> Result<TraceRecord> {
    // path#line or a filesystem path.
    if let Some((path, line)) = split_path_line(target) {
        let recs = TraceRecord::read_all(&path)
            .with_context(|| format!("reading trace file {}", path.display()))?;
        match line {
            Some(n) => recs
                .into_iter()
                .nth(n.saturating_sub(1))
                .ok_or_else(|| anyhow!("no trace at line {n} of {}", path.display())),
            None => recs
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("{} contains no trace records", path.display())),
        }
    } else {
        find_by_id(traces_dir, target)
    }
}

/// Interpret a target as a direct path (optionally `#line`) when it looks like one.
fn split_path_line(target: &str) -> Option<(PathBuf, Option<usize>)> {
    let (path_part, line_part) = match target.rsplit_once('#') {
        Some((p, l)) if l.chars().all(|c| c.is_ascii_digit()) && !l.is_empty() => {
            (p, Some(l.parse::<usize>().ok()?))
        }
        _ => (target, None),
    };
    let p = Path::new(path_part);
    // Only treat as a path if it exists on disk (ids never do).
    if p.is_file() {
        Some((p.to_path_buf(), line_part))
    } else {
        None
    }
}

/// Find a record by trace id: consult `index.jsonl` for existence, then scan the trace files.
pub fn find_by_id(traces_dir: &Path, trace_id: &str) -> Result<TraceRecord> {
    // The index tells us the id exists (fast negative), but the full record lives in the dated
    // file, so we scan those. (The index is tiny; a full scan of a day's traces is cheap.)
    let index_path = traces_dir.join("index.jsonl");
    if index_path.is_file() {
        if let Ok(idx) = TraceIndex::load(&index_path) {
            if !idx.entries.iter().any(|e| e.trace_id == trace_id) {
                bail!("trace id `{trace_id}` not found in {}", index_path.display());
            }
        }
    }
    for rec in load_all_records(traces_dir)? {
        if rec.trace_id == trace_id {
            return Ok(rec);
        }
    }
    bail!("trace id `{trace_id}` not found under {}", traces_dir.display())
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// show
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Which section(s) `trace show` renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    /// The client request block.
    Client,
    /// The decoded IR request.
    Ir,
    /// The resolved route.
    Route,
    /// The upstream request/response block.
    Upstream,
    /// The decoded IR events.
    Events,
    /// The client response frames.
    Frames,
    /// The recorded errors.
    Errors,
    /// The inline invariant checks.
    Checks,
    /// Everything.
    All,
}

impl Section {
    /// Parse the `--section` argument.
    pub fn parse(s: &str) -> Result<Section> {
        Ok(match s {
            "client" => Section::Client,
            "ir" => Section::Ir,
            "route" => Section::Route,
            "upstream" => Section::Upstream,
            "events" => Section::Events,
            "frames" => Section::Frames,
            "errors" => Section::Errors,
            "checks" => Section::Checks,
            "all" => Section::All,
            other => bail!(
                "--section must be client|ir|route|upstream|events|frames|errors|checks|all (got {other:?})"
            ),
        })
    }

    fn wants(self, other: Section) -> bool {
        self == Section::All || self == other
    }
}

/// Render a record for `trace show`. When `raw`, dumps the whole record as pretty JSON.
pub fn show(rec: &TraceRecord, section: Section, raw: bool) -> String {
    if raw {
        return serde_json::to_string_pretty(rec).unwrap_or_default();
    }
    let mut out = String::new();
    let head = format!(
        "trace {}  [{}]  {} -> {}  client={} status={}\n",
        rec.trace_id,
        rec.tag.as_deref().unwrap_or("-"),
        rec.ts_start,
        rec.ts_end,
        rec.client.protocol,
        rec.client_response.as_ref().map(|c| c.status).unwrap_or(0),
    );
    out.push_str(&head);

    if section.wants(Section::Client) {
        out.push_str("\n== client ==\n");
        out.push_str(&format!(
            "{} {} stream={}\n",
            rec.client.method, rec.client.path, rec.client.stream
        ));
        out.push_str(&format!("headers: {}\n", pretty(&Value::Object(rec.client.headers.clone()))));
        if let Some(b) = &rec.client.body {
            out.push_str(&format!("body: {}\n", pretty(b)));
        }
    }
    if section.wants(Section::Ir) {
        out.push_str("\n== ir_request ==\n");
        out.push_str(&pretty_opt(&rec.ir_request));
        out.push('\n');
    }
    if section.wants(Section::Route) {
        out.push_str("\n== route ==\n");
        out.push_str(&pretty_opt(&rec.route));
        out.push('\n');
        if let Some(l) = &rec.lowered {
            out.push_str(&format!("degradations ({}):\n", l.degradations.len()));
            for d in &l.degradations {
                out.push_str(&format!("  - {}\n", compact(d)));
            }
        }
    }
    if section.wants(Section::Upstream) {
        out.push_str("\n== upstream ==\n");
        if let Some(u) = &rec.upstream {
            out.push_str(&format!(
                "{} stream={} status={} request_id={} ttfb_ms={} elapsed_ms={}\n",
                u.url,
                u.stream,
                u.status,
                u.request_id.as_deref().unwrap_or("-"),
                u.ttfb_ms,
                u.elapsed_ms,
            ));
            if let Some(b) = &u.body {
                out.push_str(&format!("body: {}\n", pretty(b)));
            }
            if let Some(raw) = &u.raw {
                out.push_str("raw:\n");
                out.push_str(&indent_lines(raw));
            }
        } else {
            out.push_str("(no upstream leg)\n");
        }
    }
    if section.wants(Section::Events) {
        out.push_str("\n== ir_events ==\n");
        for ev in &rec.ir_events {
            out.push_str(&format!("[{:>6}ms] {}\n", ev.t_ms, compact(&ev.event)));
        }
        out.push_str(&format!("\nir_response: {}\n", pretty_opt(&rec.ir_response)));
    }
    if section.wants(Section::Frames) {
        out.push_str("\n== client frames ==\n");
        match rec.client_response.as_ref().and_then(|c| c.frames.as_ref()) {
            Some(frames) => {
                for f in frames {
                    let ka = if f.keepalive { " (keepalive)" } else { "" };
                    out.push_str(&format!("[{:>6}ms]{ka} {}\n", f.t_ms, f.data.trim_end()));
                }
            }
            None => {
                if let Some(b) = rec.client_response.as_ref().and_then(|c| c.body.as_ref()) {
                    out.push_str(&format!("(non-streaming body)\n{}\n", pretty(b)));
                } else {
                    out.push_str("(no frames)\n");
                }
            }
        }
    }
    if section.wants(Section::Errors) {
        out.push_str("\n== errors ==\n");
        if rec.errors.is_empty() {
            out.push_str("(none)\n");
        } else {
            for e in &rec.errors {
                out.push_str(&format!(
                    "  [{}] {} status={} — {}\n",
                    e.stage, e.kind, e.status, e.message
                ));
            }
        }
    }
    if section.wants(Section::Checks) {
        out.push_str("\n== checks ==\n");
        out.push_str(&format!(
            "aggregate_matches_encode_response = {}\n",
            rec.checks.aggregate_matches_encode_response
        ));
        out.push_str(&format!("degradation_count = {}\n", rec.checks.degradation_count));
        out.push_str(&format!(
            "reencode_client_request_diff = {:?}\n",
            rec.checks.reencode_client_request_diff
        ));
    }
    out
}

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_default()
}
fn pretty_opt(v: &Option<Value>) -> String {
    match v {
        Some(v) => pretty(v),
        None => "(absent)".to_string(),
    }
}
fn compact(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}
fn indent_lines(s: &str) -> String {
    let mut out = String::new();
    for line in s.lines() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// triage
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A filter over the triaged records.
#[derive(Debug, Clone, Default)]
pub struct TriageFilter {
    /// Only records whose `ts_start` is >= this RFC-3339 string.
    pub since: Option<String>,
    /// Only records with this exact tag.
    pub tag: Option<String>,
}

/// The TTFB anomaly threshold (plan §6): 10 seconds.
pub const SLOW_TTFB_MS: u64 = 10_000;

/// One triage row.
#[derive(Debug, Clone, Serialize)]
pub struct TriageRow {
    /// Trace id.
    pub trace_id: String,
    /// Start timestamp.
    pub ts: String,
    /// Session tag.
    pub tag: Option<String>,
    /// Client protocol.
    pub client_protocol: String,
    /// `provider/upstream_protocol`.
    pub route: String,
    /// Upstream HTTP status (0 if no upstream leg).
    pub upstream_status: u16,
    /// Client HTTP status.
    pub client_status: u16,
    /// First error `kind` / `stage`, if any.
    pub error: Option<String>,
    /// Degradation count.
    pub degradations: u32,
    /// Whether any inline check failed.
    pub failed_check: bool,
    /// Upstream TTFB (ms).
    pub ttfb_ms: u64,
    /// Total (ms).
    pub total_ms: u64,
    /// The anomaly reasons (empty = clean).
    pub anomalies: Vec<String>,
}

impl TriageRow {
    /// Whether this row is anomalous.
    pub fn is_anomalous(&self) -> bool {
        !self.anomalies.is_empty()
    }
}

/// A whole triage report.
#[derive(Debug, Clone, Serialize)]
pub struct TriageReport {
    /// One row per record (filtered), file order.
    pub rows: Vec<TriageRow>,
    /// Count of each anomaly reason across all rows.
    pub anomaly_counts: BTreeMap<String, usize>,
    /// Number of anomalous rows.
    pub anomalous_rows: usize,
    /// Total rows.
    pub total_rows: usize,
}

impl TriageReport {
    /// Whether any row is anomalous (drives the CLI exit code).
    pub fn has_anomalies(&self) -> bool {
        self.anomalous_rows > 0
    }

    /// A JSON rendering.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// A fixed-width table for the terminal.
    pub fn to_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<14} {:<10} {:<8} {:<22} {:>4} {:>4} {:>4} {:>6} {:<10} {}\n",
            "trace", "tag", "client", "route", "up", "cl", "deg", "ttfb", "error", "anomalies"
        ));
        for r in &self.rows {
            out.push_str(&format!(
                "{:<14} {:<10} {:<8} {:<22} {:>4} {:>4} {:>4} {:>6} {:<10} {}\n",
                truncate(&r.trace_id, 14),
                truncate(r.tag.as_deref().unwrap_or("-"), 10),
                truncate(&r.client_protocol, 8),
                truncate(&r.route, 22),
                r.upstream_status,
                r.client_status,
                r.degradations,
                r.ttfb_ms,
                truncate(r.error.as_deref().unwrap_or("-"), 10),
                r.anomalies.join(","),
            ));
        }
        out.push_str(&format!(
            "\n{} row(s), {} anomalous.\n",
            self.total_rows, self.anomalous_rows
        ));
        if !self.anomaly_counts.is_empty() {
            out.push_str("anomaly counts:\n");
            for (k, n) in &self.anomaly_counts {
                out.push_str(&format!("  {k}: {n}\n"));
            }
        }
        out
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}

/// The translation-layer stages: an error here is an `llm-xlate` bug, not a provider passthrough.
const XLATE_STAGES: &[&str] = &["decode", "lower", "encode", "decode_response"];

/// Compute the anomaly reasons for one record (plan §6).
pub fn anomalies(rec: &TraceRecord) -> Vec<String> {
    let mut out = Vec::new();
    // Cancellation.
    let cancelled = rec.errors.iter().any(|e| e.stage == "cancelled");
    if cancelled {
        out.push("cancelled".to_string());
    }
    // Any error (other than a pure cancellation, already flagged).
    if let Some(e) = rec.errors.iter().find(|e| e.stage != "cancelled") {
        out.push(format!("error:{}", e.kind));
        if XLATE_STAGES.contains(&e.stage.as_str()) {
            out.push("xlate_error".to_string());
        }
    }
    // Failed inline check.
    if !rec.checks.aggregate_matches_encode_response {
        out.push("failed_check".to_string());
    }
    // Unknown capabilities.
    if rec
        .route
        .as_ref()
        .and_then(|r| r.get("caps_source"))
        .and_then(Value::as_str)
        == Some("unknown")
    {
        out.push("caps_unknown".to_string());
    }
    // Keepalive-only stream: streamed, has frames, but none carry content.
    if let Some(cr) = &rec.client_response {
        if let Some(frames) = &cr.frames {
            if !frames.is_empty() && frames.iter().all(|f| f.keepalive) {
                out.push("keepalive_only".to_string());
            }
        }
    }
    // Slow TTFB.
    if let Some(u) = &rec.upstream {
        if u.ttfb_ms > SLOW_TTFB_MS {
            out.push("slow_ttfb".to_string());
        }
        // Missing upstream request id on a successful upstream leg.
        if u.status >= 200 && u.status < 300 && u.request_id.is_none() {
            out.push("missing_request_id".to_string());
        }
    }
    out
}

/// Build a triage report over `records`, honouring `filter`.
pub fn triage(records: &[TraceRecord], filter: &TriageFilter) -> TriageReport {
    let mut rows = Vec::new();
    let mut anomaly_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut anomalous_rows = 0;
    for rec in records {
        if let Some(since) = &filter.since {
            if rec.ts_start.as_str() < since.as_str() {
                continue;
            }
        }
        if let Some(tag) = &filter.tag {
            if rec.tag.as_deref() != Some(tag.as_str()) {
                continue;
            }
        }
        let an = anomalies(rec);
        if !an.is_empty() {
            anomalous_rows += 1;
            for a in &an {
                *anomaly_counts.entry(reason_key(a)).or_insert(0) += 1;
            }
        }
        let (provider, up_proto) = rec
            .route
            .as_ref()
            .map(|r| {
                (
                    r.get("provider").and_then(Value::as_str).unwrap_or("-").to_string(),
                    r.get("upstream_protocol").and_then(Value::as_str).unwrap_or("-").to_string(),
                )
            })
            .unwrap_or_else(|| ("-".to_string(), "-".to_string()));
        let up = rec.upstream.as_ref();
        rows.push(TriageRow {
            trace_id: rec.trace_id.clone(),
            ts: rec.ts_start.clone(),
            tag: rec.tag.clone(),
            client_protocol: rec.client.protocol.clone(),
            route: format!("{provider}/{up_proto}"),
            upstream_status: up.map(|u| u.status).unwrap_or(0),
            client_status: rec.client_response.as_ref().map(|c| c.status).unwrap_or(0),
            error: rec.errors.first().map(|e| format!("{}/{}", e.kind, e.stage)),
            degradations: rec.checks.degradation_count,
            failed_check: !rec.checks.aggregate_matches_encode_response,
            ttfb_ms: up.map(|u| u.ttfb_ms).unwrap_or(0),
            total_ms: rec.timing.total_ms,
            anomalies: an,
        });
    }
    let total_rows = rows.len();
    TriageReport { rows, anomaly_counts, anomalous_rows, total_rows }
}

/// Normalize an anomaly string to its category key (drops the `:detail` suffix).
fn reason_key(a: &str) -> String {
    a.split(':').next().unwrap_or(a).to_string()
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// export
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Where the two exported captures landed.
#[derive(Debug, Clone)]
pub struct ExportOutcome {
    /// The run root directory written.
    pub root: PathBuf,
    /// The client-leg capture directory (relative to `root`).
    pub client_rel: String,
    /// The upstream-leg capture directory (relative to `root`).
    pub upstream_rel: String,
}

/// Export a trace into an `llm-xlate-e2e` run directory holding two captures — the client leg and
/// the upstream leg — plus a `manifest.json`, so `llm-xlate-e2e check`/`promote` consume it
/// unchanged (plan §6, R3). Redaction from the trace is preserved (we copy the already-redacted
/// bodies/headers verbatim).
pub fn export(rec: &TraceRecord, to: &Path) -> Result<ExportOutcome> {
    std::fs::create_dir_all(to).with_context(|| format!("mkdir {}", to.display()))?;

    let client_proto = probe_protocol(&rec.client.protocol)
        .ok_or_else(|| anyhow!("unknown client protocol `{}`", rec.client.protocol))?;
    let up = rec
        .upstream
        .as_ref()
        .ok_or_else(|| anyhow!("trace has no upstream leg to export (it failed before send)"))?;
    let up_proto_str = rec
        .route
        .as_ref()
        .and_then(|r| r.get("upstream_protocol"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("trace route has no upstream_protocol"))?;
    let up_proto = probe_protocol(up_proto_str)
        .ok_or_else(|| anyhow!("unknown upstream protocol `{up_proto_str}`"))?;

    let client_model = rec
        .route
        .as_ref()
        .and_then(|r| r.get("client_model"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let upstream_model = rec
        .route
        .as_ref()
        .and_then(|r| r.get("upstream_model"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    // ── client leg ────────────────────────────────────────────────────────────────────────────
    let client_entry = {
        let ex = expansion(client_proto, "client_leg", &client_model, rec.client.stream)?;
        let req = RequestRecord {
            method: rec.client.method.clone(),
            url: format!("http://router{}", rec.client.path),
            stream: rec.client.stream,
            headers: header_map(&rec.client.headers),
            body: rec.client.body.clone().unwrap_or(Value::Null),
        };
        let cr = rec
            .client_response
            .as_ref()
            .ok_or_else(|| anyhow!("trace has no client response to export"))?;
        let (wire, obs) = client_wire_and_observe(client_proto, &req.body, cr, rec.timing.total_ms);
        capture::write_capture(to, &ex, &req, &wire, &obs)?
    };

    // ── upstream leg ──────────────────────────────────────────────────────────────────────────
    let upstream_entry = {
        let ex = expansion(up_proto, "upstream_leg", &upstream_model, up.stream)?;
        let req = RequestRecord {
            method: "POST".to_string(),
            url: upstream_url(up, up_proto),
            stream: up.stream,
            headers: header_map(&up.headers),
            body: up.body.clone().unwrap_or(Value::Null),
        };
        let (wire, obs) = upstream_wire_and_observe(up_proto, &req.body, up);
        capture::write_capture(to, &ex, &req, &wire, &obs)?
    };

    let manifest = Manifest {
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        xlate_version: env!("CARGO_PKG_VERSION").to_string(),
        run_id: rec.trace_id.clone(),
        label: rec.tag.clone().unwrap_or_else(|| "export".to_string()),
        created_at: rec.ts_start.clone(),
        providers: BTreeMap::new(),
        model_sets: vec![],
        entries: vec![client_entry.clone(), upstream_entry.clone()],
    };
    capture::write_manifest(to, &manifest)?;

    Ok(ExportOutcome {
        root: to.to_path_buf(),
        client_rel: client_entry.rel_dir,
        upstream_rel: upstream_entry.rel_dir,
    })
}

fn probe_protocol(s: &str) -> Option<ProbeProtocol> {
    match s {
        "chat" => Some(ProbeProtocol::Chat),
        "responses" => Some(ProbeProtocol::Responses),
        "anthropic" => Some(ProbeProtocol::Anthropic),
        _ => None,
    }
}

/// Build a one-off [`Expansion`] for a capture leg. `role` distinguishes the two legs so a
/// same-surface trace's two captures never collide on `<probe>/<model>`.
fn expansion(proto: ProbeProtocol, role: &str, model: &str, stream: bool) -> Result<Expansion> {
    let id = format!("{}{role}", proto.prefix()); // e.g. "chat.client_leg"
    let toml = format!(
        "[[probe]]\nid=\"{id}\"\nprotocol=\"{}\"\nmodels=[\"m\"]\n[probe.body]\nmodel=\"$MODEL\"\n",
        proto.as_str()
    );
    let probe = Catalogue::parse_file(&toml)
        .with_context(|| format!("building export probe {id}"))?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("empty probe catalogue"))?;
    Ok(Expansion { probe, model: model.to_string(), stream })
}

/// The client-leg wire response + observation.
fn client_wire_and_observe(
    proto: ProbeProtocol,
    request_body: &Value,
    cr: &ClientResponse,
    total_ms: u64,
) -> (WireResponse, Value) {
    let headers = header_map(&cr.headers);
    if let Some(frames) = &cr.frames {
        let raw = frames_to_raw(frames);
        let events = parse_sse(&raw);
        let obs = observe_stream(proto, request_body, cr.status, &headers, &raw, &events, &Expect::Any);
        (
            WireResponse::Stream {
                status: cr.status,
                headers,
                elapsed_ms: total_ms as u128,
                raw,
            },
            serde_json::to_value(obs).unwrap_or(Value::Null),
        )
    } else {
        let body = cr.body.clone().unwrap_or(Value::Null);
        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        let obs = observe_nonstream(proto, request_body, cr.status, &headers, &body, &Expect::Any);
        (
            WireResponse::NonStream {
                status: cr.status,
                headers,
                elapsed_ms: total_ms as u128,
                body_bytes,
            },
            serde_json::to_value(obs).unwrap_or(Value::Null),
        )
    }
}

/// The upstream-leg wire response + observation, served from the recorded raw bytes.
fn upstream_wire_and_observe(
    proto: ProbeProtocol,
    request_body: &Value,
    up: &UpstreamTrace,
) -> (WireResponse, Value) {
    let headers = header_map(&up.resp_headers);
    let raw = up
        .raw
        .clone()
        .map(|s| s.into_bytes())
        .unwrap_or_else(|| serde_json::to_vec(up.body.as_ref().unwrap_or(&Value::Null)).unwrap_or_default());
    if up.stream {
        let events = parse_sse(&raw);
        let obs = observe_stream(proto, request_body, up.status, &headers, &raw, &events, &Expect::Any);
        (
            WireResponse::Stream {
                status: up.status,
                headers,
                elapsed_ms: up.elapsed_ms as u128,
                raw,
            },
            serde_json::to_value(obs).unwrap_or(Value::Null),
        )
    } else {
        let body: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
        let obs = observe_nonstream(proto, request_body, up.status, &headers, &body, &Expect::Any);
        (
            WireResponse::NonStream {
                status: up.status,
                headers,
                elapsed_ms: up.elapsed_ms as u128,
                body_bytes: raw,
            },
            serde_json::to_value(obs).unwrap_or(Value::Null),
        )
    }
}

fn upstream_url(up: &UpstreamTrace, proto: ProbeProtocol) -> String {
    // The recorded URL already carries the wire path in production; a mock/empty base needs a path
    // the e2e protocol sniffer (`/v1/...`) can classify.
    if up.url.contains("/v1/") {
        up.url.clone()
    } else {
        let p = match proto {
            ProbeProtocol::Anthropic => "/v1/messages",
            ProbeProtocol::Chat => "/v1/chat/completions",
            ProbeProtocol::Responses => "/v1/responses",
        };
        format!("http://upstream{p}")
    }
}

/// Convert a trace header map (name → JSON string) into the e2e `BTreeMap<String, String>` form.
fn header_map(m: &Map<String, Value>) -> BTreeMap<String, String> {
    m.iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
        .collect()
}

/// Reconstruct the raw SSE bytes from recorded content frames (dropping keepalives).
fn frames_to_raw(frames: &[crate::trace::FrameTrace]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        if f.keepalive {
            continue;
        }
        out.extend_from_slice(f.data.as_bytes());
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// replay
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// An `IdSource` seeded at a fixed counter, so `replay` reproduces the recorded sequential ids
/// exactly (the fragment format matches [`crate::ids::SequentialSource`]).
struct SeededSource {
    counter: AtomicU64,
}

impl SeededSource {
    fn starting_at(n: u64) -> Self {
        Self { counter: AtomicU64::new(n) }
    }
}

impl IdSource for SeededSource {
    fn next_fragment(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        format!("{n:08}")
    }
}

/// An in-process [`Upstream`] that serves the recorded raw bytes (streaming or full) and records
/// the request body it was asked to send, so `replay` can diff the produced upstream request.
struct ReplayUpstream {
    status: u16,
    stream: bool,
    raw: Vec<u8>,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl Upstream for ReplayUpstream {
    fn send<'a>(&'a self, req: UpstreamRequest) -> BoxFuture<'a, Result<UpstreamResponse, UpstreamError>> {
        Box::pin(async move {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            self.seen.lock().unwrap().push(body);
            let headers = llm_xlate_core::HeaderMap::new();
            let raw = Bytes::from(self.raw.clone());
            let body = if self.stream {
                let stream: ByteStream = Box::pin(async_stream::stream! {
                    yield Ok::<Bytes, UpstreamError>(raw);
                });
                UpstreamBody::Stream(stream)
            } else {
                UpstreamBody::Full(raw)
            };
            Ok(UpstreamResponse { status: self.status, headers, body, ttfb: Duration::ZERO })
        })
    }
}

/// The result of a `replay`.
#[derive(Debug, Clone, Serialize)]
pub struct ReplayOutcome {
    /// Structural (JSON) diff of the produced upstream body vs the recorded one.
    pub upstream_diff: Vec<String>,
    /// Frame-level (or body) diff of the produced client output vs the recorded one.
    pub client_diff: Vec<String>,
    /// The produced client HTTP status.
    pub client_status: u16,
    /// The recorded client HTTP status.
    pub recorded_status: u16,
}

impl ReplayOutcome {
    /// Whether the replay reproduced the trace exactly (no diffs, same status).
    pub fn matched(&self) -> bool {
        self.upstream_diff.is_empty()
            && self.client_diff.is_empty()
            && self.client_status == self.recorded_status
    }

    /// A human-readable rendering.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "client status: produced={} recorded={}\n",
            self.client_status, self.recorded_status
        ));
        out.push_str(&format!("upstream body diff ({}):\n", self.upstream_diff.len()));
        for d in &self.upstream_diff {
            out.push_str(&format!("  {d}\n"));
        }
        out.push_str(&format!("client output diff ({}):\n", self.client_diff.len()));
        for d in &self.client_diff {
            out.push_str(&format!("  {d}\n"));
        }
        out.push_str(if self.matched() { "\nNO DIFF ✓\n" } else { "\nDIFFERENCES FOUND ✗\n" });
        out
    }
}

/// Re-run the recorded client request through the current build against a mock upstream that serves
/// the recorded raw bytes, and diff (plan §6, R3). Deterministic when the trace used sequential ids
/// (the test harness and `serve` in a fresh session); production uuid traces reproduce the upstream
/// body structurally but their minted ids differ (documented limit).
pub async fn replay(rec: &TraceRecord, base_config: Config) -> Result<ReplayOutcome> {
    use tower::ServiceExt;

    let up = rec
        .upstream
        .as_ref()
        .ok_or_else(|| anyhow!("trace has no upstream leg to replay"))?;

    // Refuse to replay a trace whose bodies were media-redacted — the exact bytes are gone.
    if body_was_redacted(rec.client.body.as_ref()) {
        bail!("cannot replay: the client body was media-redacted in this trace");
    }

    let mut config = base_config;
    config.server.token = String::new(); // replay reproduces translation, not auth.

    let seed = seed_from_trace_id(&rec.trace_id);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let raw = up
        .raw
        .clone()
        .map(|s| s.into_bytes())
        .unwrap_or_else(|| serde_json::to_vec(up.body.as_ref().unwrap_or(&Value::Null)).unwrap_or_default());
    let replay_upstream = Arc::new(ReplayUpstream {
        status: up.status,
        stream: up.stream,
        raw,
        seen: seen.clone(),
    });

    let deps = Deps {
        translator: Translator::new(TranslatorConfig::default()),
        id_source: Arc::new(SeededSource::starting_at(seed)),
        clock: Arc::new(FixedClock::default()),
        store: Arc::new(MemoryStore::new()),
        sidecar: Arc::new(MemorySidecar::new()),
        trace_sink: Arc::new(NullTraceSink),
        upstream: replay_upstream,
    };
    let app = App::build(config, deps)?;

    // Rebuild the client request: same path + body + the recorded x-xlate-* overrides so routing
    // reproduces.
    let body_bytes = serde_json::to_vec(rec.client.body.as_ref().unwrap_or(&Value::Null))?;
    let mut builder = axum::http::Request::builder()
        .method(rec.client.method.as_str())
        .uri(rec.client.path.as_str())
        .header("content-type", "application/json");
    for (k, v) in override_headers(rec) {
        builder = builder.header(k, v);
    }
    let request = builder.body(axum::body::Body::from(body_bytes))?;
    let resp = app.oneshot(request).await.map_err(|e| anyhow!("replay dispatch failed: {e}"))?;
    let client_status = resp.status().as_u16();
    let out_body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .map_err(|e| anyhow!("collecting replay body: {e}"))?;

    // ── upstream body diff ─────────────────────────────────────────────────────────────────────
    let produced_up = seen.lock().unwrap().first().cloned().unwrap_or(Value::Null);
    let recorded_up = up.body.clone().unwrap_or(Value::Null);
    let upstream_diff = json_diff("", &recorded_up, &produced_up);

    // ── client output diff ─────────────────────────────────────────────────────────────────────
    let cr = rec.client_response.as_ref();
    let recorded_status = cr.map(|c| c.status).unwrap_or(0);
    let client_diff = if rec.client.stream {
        let recorded_raw = cr.and_then(|c| c.frames.as_ref()).map(|f| frames_to_raw(f)).unwrap_or_default();
        sse_diff(&recorded_raw, &out_body)
    } else {
        let recorded_body = cr.and_then(|c| c.body.clone()).unwrap_or(Value::Null);
        let produced_body: Value = serde_json::from_slice(&out_body).unwrap_or(Value::Null);
        json_diff("", &recorded_body, &produced_body)
    };

    Ok(ReplayOutcome { upstream_diff, client_diff, client_status, recorded_status })
}

fn body_was_redacted(body: Option<&Value>) -> bool {
    fn walk(v: &Value) -> bool {
        match v {
            Value::Object(o) => o.contains_key("$redacted") || o.values().any(walk),
            Value::Array(a) => a.iter().any(walk),
            _ => false,
        }
    }
    body.map(walk).unwrap_or(false)
}

/// Parse `tr_<n>` into the seed counter; a non-numeric (uuid) fragment seeds at 1.
fn seed_from_trace_id(trace_id: &str) -> u64 {
    trace_id
        .strip_prefix("tr_")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1)
}

/// Reconstruct the `x-xlate-*` override headers from the recorded route so routing reproduces.
fn override_headers(rec: &TraceRecord) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(ov) = rec.route.as_ref().and_then(|r| r.get("overrides")) else {
        return out;
    };
    let add = |field: &str, header: &str, out: &mut Vec<(String, String)>| {
        if let Some(s) = ov.get(field).and_then(Value::as_str) {
            out.push((header.to_string(), s.to_string()));
        }
    };
    add("model", "x-xlate-model", &mut out);
    add("upstream", "x-xlate-upstream", &mut out);
    add("provider", "x-xlate-provider", &mut out);
    add("caps", "x-xlate-caps", &mut out);
    add("expose", "x-xlate-expose", &mut out);
    add("tag", "x-xlate-tag", &mut out);
    out
}

/// A frame-level diff of two SSE byte streams: compares the `(event, data)` sequence.
fn sse_diff(recorded: &[u8], produced: &[u8]) -> Vec<String> {
    let a = parse_sse(recorded);
    let b = parse_sse(produced);
    let mut out = Vec::new();
    if a.len() != b.len() {
        out.push(format!("frame count {} != {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if event_key(x) != event_key(y) {
            out.push(format!("frame[{i}]: {} != {}", event_key(x), event_key(y)));
        }
    }
    out
}

fn event_key(e: &EventRecord) -> String {
    format!("{}|{}", e.event.as_deref().unwrap_or(""), e.data)
}

/// A minimal structural JSON diff: pointer paths whose presence or scalar value differs.
pub fn json_diff(path: &str, a: &Value, b: &Value) -> Vec<String> {
    let mut out = Vec::new();
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            let mut keys: std::collections::BTreeSet<&String> = ma.keys().collect();
            keys.extend(mb.keys());
            for k in keys {
                let child = format!("{path}/{k}");
                match (ma.get(k), mb.get(k)) {
                    (Some(x), Some(y)) => out.extend(json_diff(&child, x, y)),
                    (Some(_), None) => out.push(format!("-{child}")),
                    (None, Some(_)) => out.push(format!("+{child}")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(xa), Value::Array(xb)) => {
            if xa.len() != xb.len() {
                out.push(format!("~{path}[len {}!={}]", xa.len(), xb.len()));
            }
            for (i, (x, y)) in xa.iter().zip(xb.iter()).enumerate() {
                out.extend(json_diff(&format!("{path}/{i}"), x, y));
            }
        }
        (x, y) => {
            if x != y {
                out.push(format!("~{path}"));
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Config → traces-dir helpers for the CLI
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Resolve the `data_dir/traces` for a loaded config (used by the CLI to locate records).
pub fn traces_dir_for(config: &Config) -> PathBuf {
    traces_dir(Path::new(&config.server.data_dir))
}

/// A tiny convenience for the CLI: load a config and return its traces dir.
pub fn traces_dir_from_config(config_path: &str) -> Result<PathBuf> {
    let cfg = Config::load(config_path)?;
    Ok(traces_dir_for(&cfg))
}
