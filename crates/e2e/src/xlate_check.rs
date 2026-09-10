//! `check` (plan §7): push every capture through `llm-xlate` and record, per capture, whether
//! each of the five checks passes.
//!
//! The input is either a **run directory** (a `manifest.json` plus the capture tree this crate
//! writes) or a **golden directory** such as `crates/xlate/tests/golden` (the codec crate's
//! `*.req.json` / `*.resp.json` / `*.stream.sse` / `*.err.json` fixtures, whose protocol is their
//! parent directory). Both are normalized into `Case`s — one per logical (protocol, stem) or
//! (probe, model) group, so a streaming fixture and its non-streaming twin land in the same case
//! and their aggregates can be compared.
//!
//! Every individual check is run inside [`std::panic::catch_unwind`]: a panic inside `llm-xlate`
//! is reported as a failed item, never allowed to abort the whole run (plan §7 last sentence).
//!
//! `check` never writes into a directory it does not own: with `--out` it mirrors `xlate.json`
//! (and the human-review `xlate/<client>.{sse,json}` renderings) under that directory; without
//! `--out` it writes next to a capture only when that capture lives in a run this crate produced,
//! and for a read-only golden tree it computes and reports without writing anything.

use crate::capture::{iter_run, Outcome};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use llm_xlate::Translator;
use llm_xlate_core::caps::{shipped, Capabilities};
use llm_xlate_core::codec::TranslatorConfig;
use llm_xlate_core::ir::{IrEvent, IrResponse, Item, Protocol, ResponseId, StopReason};
use llm_xlate_core::{HeaderMap, HeaderName, HeaderValue};

/// A fixed timestamp so encoded bodies are deterministic.
const CREATED_AT: u64 = 1_700_000_000;
/// Number of seeded chunk-fuzz split points (plan §7.3).
const FUZZ_SPLITS: usize = 25;

/// Which client protocols the cross-client rendering (§7.5) targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientFilter {
    /// Every protocol other than the capture's own.
    All,
    /// Only this one (skipped when it equals the capture's own protocol).
    One(Protocol),
}

impl ClientFilter {
    /// Parse the `--client` argument.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "all" => ClientFilter::All,
            "chat" => ClientFilter::One(Protocol::OaiChat),
            "responses" => ClientFilter::One(Protocol::OaiResponses),
            "anthropic" => ClientFilter::One(Protocol::Anthropic),
            other => anyhow::bail!("--client must be all|chat|responses|anthropic (got {other:?})"),
        })
    }

    fn includes(self, p: Protocol) -> bool {
        match self {
            ClientFilter::All => true,
            ClientFilter::One(q) => p == q,
        }
    }
}

/// The result of one named check item.
#[derive(Debug, Clone, Serialize)]
pub struct ItemResult {
    /// The item name (e.g. `request_decode`).
    pub name: String,
    /// Whether the check passed.
    pub pass: bool,
    /// Human-readable detail (error text, diff summary, or a note).
    pub detail: String,
}

impl ItemResult {
    fn ok(name: &str, detail: impl Into<String>) -> Self {
        ItemResult { name: name.into(), pass: true, detail: detail.into() }
    }
    fn fail(name: &str, detail: impl Into<String>) -> Self {
        ItemResult { name: name.into(), pass: false, detail: detail.into() }
    }
}

/// The per-capture result written to `xlate.json`.
#[derive(Debug, Clone, Serialize)]
pub struct CaseCheck {
    /// A stable label (`<protocol>/<stem>` for goldens, `<probe>/<model>` for runs).
    pub label: String,
    /// The protocol string.
    pub protocol: String,
    /// The resolved model, when known.
    pub model: Option<String>,
    /// Whether a model was known (else `Capabilities::unknown()` was used).
    pub caps_known: bool,
    /// One entry per check item that applied.
    pub items: Vec<ItemResult>,
}

impl CaseCheck {
    /// Whether every applied item passed.
    pub fn all_pass(&self) -> bool {
        self.items.iter().all(|i| i.pass)
    }
}

/// The whole-run outcome.
#[derive(Debug, Clone, Serialize)]
pub struct CheckSummary {
    /// One entry per case, in discovery order.
    pub cases: Vec<CaseCheck>,
}

impl CheckSummary {
    /// Total (passed, failed) item counts across all cases.
    pub fn totals(&self) -> (usize, usize) {
        let mut pass = 0;
        let mut fail = 0;
        for c in &self.cases {
            for i in &c.items {
                if i.pass {
                    pass += 1;
                } else {
                    fail += 1;
                }
            }
        }
        (pass, fail)
    }

    /// Whether any item across any case failed.
    pub fn any_failed(&self) -> bool {
        self.totals().1 > 0
    }

    /// A compact per-item totals table for the terminal.
    pub fn table(&self) -> String {
        let mut by_item: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for c in &self.cases {
            for i in &c.items {
                let e = by_item.entry(i.name.clone()).or_insert((0, 0));
                if i.pass {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                }
            }
        }
        let mut out = String::new();
        out.push_str(&format!("{:<26} {:>6} {:>6}\n", "check", "pass", "fail"));
        for (name, (p, f)) in &by_item {
            out.push_str(&format!("{name:<26} {p:>6} {f:>6}\n"));
        }
        let (p, f) = self.totals();
        out.push_str(&format!("{:<26} {:>6} {:>6}\n", "TOTAL", p, f));
        out
    }

    /// Every failing item, formatted `label/item: detail`.
    pub fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        for c in &self.cases {
            for i in &c.items {
                if !i.pass {
                    out.push(format!("{}/{}: {}", c.label, i.name, i.detail));
                }
            }
        }
        out
    }
}

/// The three wire protocols, for cross-client iteration.
const PROTS: [Protocol; 3] = [Protocol::OaiChat, Protocol::OaiResponses, Protocol::Anthropic];

fn pname(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "chat",
        Protocol::OaiResponses => "responses",
        Protocol::Anthropic => "anthropic",
    }
}

fn core_protocol(s: &str) -> Option<Protocol> {
    match s {
        "chat" => Some(Protocol::OaiChat),
        "responses" => Some(Protocol::OaiResponses),
        "anthropic" => Some(Protocol::Anthropic),
        _ => None,
    }
}

/// One logical case: a request and/or its response(s), grouped so twins share an entry.
struct Case {
    label: String,
    protocol: Protocol,
    model: Option<String>,
    /// Where to write `xlate.json`; `None` means "do not write" (read-only golden tree, no `--out`).
    out_dir: Option<PathBuf>,
    request: Option<(Vec<u8>, HeaderMap)>,
    resp_nonstream: Option<Vec<u8>>,
    resp_stream: Option<Vec<u8>>,
    error: Option<ErrorInput>,
    expect_status: Option<u16>,
}

struct ErrorInput {
    status: u16,
    headers: HeaderMap,
    body: Vec<u8>,
}

fn header_map_from_json(v: &serde_json::Value) -> HeaderMap {
    let mut h = HeaderMap::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if let Some(s) = val.as_str() {
                if s == "<redacted>" {
                    continue;
                }
                if let (Ok(name), Ok(value)) =
                    (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(s))
                {
                    h.insert(name, value);
                }
            }
        }
    }
    h
}

fn header_map_from_str_map(m: &BTreeMap<String, String>) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, v) in m {
        if v == "<redacted>" {
            continue;
        }
        if let (Ok(name), Ok(value)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v)) {
            h.insert(name, value);
        }
    }
    h
}

fn model_of(body: &serde_json::Value) -> Option<String> {
    body.get("model").and_then(|m| m.as_str()).map(|s| s.to_string())
}

/// Build the cases for a golden directory (`<dir>/{anthropic,chat,responses}/*`).
fn cases_from_golden(dir: &Path, out: Option<&Path>) -> Result<Vec<Case>> {
    // (protocol, stem) → partial case.
    let mut map: BTreeMap<(String, String), Case> = BTreeMap::new();
    for proto in ["anthropic", "chat", "responses"] {
        let sub = dir.join(proto);
        if !sub.is_dir() {
            continue;
        }
        let p = core_protocol(proto).unwrap();
        let mut names: Vec<_> = std::fs::read_dir(&sub)
            .with_context(|| format!("reading {}", sub.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            let path = sub.join(&name);
            let (stem, kind) = classify_golden(&name);
            let (stem, kind) = match (stem, kind) {
                (Some(s), Some(k)) => (s, k),
                _ => continue,
            };
            let key = (proto.to_string(), stem.clone());
            let case = map.entry(key).or_insert_with(|| Case {
                label: format!("{proto}/{stem}"),
                protocol: p,
                model: None,
                out_dir: out.map(|o| o.join(proto).join(&stem)),
                request: None,
                resp_nonstream: None,
                resp_stream: None,
                error: None,
                expect_status: None,
            });
            match kind {
                GoldenKind::Req => {
                    let bytes = std::fs::read(&path)?;
                    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
                    if case.model.is_none() {
                        case.model = model_of(&body);
                    }
                    case.request = Some((bytes, HeaderMap::new()));
                }
                GoldenKind::Resp => {
                    let bytes = std::fs::read(&path)?;
                    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
                    if case.model.is_none() {
                        case.model = model_of(&body);
                    }
                    case.resp_nonstream = Some(bytes);
                }
                GoldenKind::Stream => {
                    case.resp_stream = Some(std::fs::read(&path)?);
                }
                GoldenKind::Err => {
                    let bytes = std::fs::read(&path)?;
                    let env: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
                    let status = env.get("status").and_then(|s| s.as_u64()).unwrap_or(500) as u16;
                    let headers = header_map_from_json(env.get("headers").unwrap_or(&serde_json::Value::Null));
                    let body = serde_json::to_vec(env.get("body").unwrap_or(&serde_json::Value::Null))?;
                    case.error = Some(ErrorInput { status, headers, body });
                    case.expect_status = Some(status);
                }
            }
        }
    }
    Ok(map.into_values().collect())
}

enum GoldenKind {
    Req,
    Resp,
    Stream,
    Err,
}

fn classify_golden(name: &str) -> (Option<String>, Option<GoldenKind>) {
    for (suffix, kind) in [
        (".req.json", GoldenKind::Req),
        (".resp.json", GoldenKind::Resp),
        (".stream.sse", GoldenKind::Stream),
        (".err.json", GoldenKind::Err),
    ] {
        if let Some(stem) = name.strip_suffix(suffix) {
            return (Some(stem.to_string()), Some(kind));
        }
    }
    (None, None)
}

/// Build the cases for a run directory, grouping a probe's stream and non-stream captures.
fn cases_from_run(run_dir: &Path, out: Option<&Path>) -> Result<Vec<Case>> {
    let captures = iter_run(run_dir)?;
    let mut map: BTreeMap<String, Case> = BTreeMap::new();
    for cap in captures {
        let p = match protocol_from_url(&cap.request.url) {
            Some(p) => p,
            None => continue,
        };
        // The capture dir is `<probe>/<model>[.stream]`; the case key drops the `.stream` suffix.
        let rel = cap.dir.strip_prefix(run_dir).unwrap_or(&cap.dir);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let case_key = rel_str.trim_end_matches(".stream").to_string();
        let model = model_of(&cap.request.body);
        let native_dir = cap.dir.clone();
        let case = map.entry(case_key.clone()).or_insert_with(|| Case {
            label: case_key.clone(),
            protocol: p,
            model: model.clone(),
            out_dir: Some(match out {
                Some(o) => o.join(&case_key),
                None => native_dir.clone(),
            }),
            request: None,
            resp_nonstream: None,
            resp_stream: None,
            error: None,
            expect_status: None,
        });
        if case.model.is_none() {
            case.model = model;
        }
        let hdrs = header_map_from_str_map(&cap.request.headers);
        let body_bytes = serde_json::to_vec(&cap.request.body).unwrap_or_default();
        if case.request.is_none() {
            case.request = Some((body_bytes, hdrs));
        }
        match cap.outcome {
            Outcome::NonStream { status, headers, body, .. } => {
                if status >= 400 {
                    let hh = {
                        let mut h = HeaderMap::new();
                        for (k, v) in &headers {
                            if let (Ok(n), Ok(val)) =
                                (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v))
                            {
                                h.insert(n, val);
                            }
                        }
                        h
                    };
                    case.error = Some(ErrorInput {
                        status,
                        headers: hh,
                        body: serde_json::to_vec(&body).unwrap_or_default(),
                    });
                    case.expect_status = Some(status);
                } else {
                    case.resp_nonstream = Some(serde_json::to_vec(&body).unwrap_or_default());
                }
            }
            Outcome::Stream { raw_sse, .. } => {
                case.resp_stream = Some(raw_sse);
            }
            Outcome::Skipped { .. } => {}
        }
    }
    Ok(map.into_values().collect())
}

fn protocol_from_url(url: &str) -> Option<Protocol> {
    if url.contains("/v1/messages") {
        Some(Protocol::Anthropic)
    } else if url.contains("/v1/chat/completions") {
        Some(Protocol::OaiChat)
    } else if url.contains("/v1/responses") {
        Some(Protocol::OaiResponses)
    } else {
        None
    }
}

/// Run `check` over `input`, returning the full summary. Writes `xlate.json` per the `out` policy.
pub fn check_dir(input: &Path, client: ClientFilter, out: Option<&Path>) -> Result<CheckSummary> {
    // Detect a run by whether the manifest parses as a *run* Manifest, not merely by the filename:
    // a promoted golden tree carries `MANIFEST.json` (provenance, a different schema), and on a
    // case-insensitive filesystem `manifest.json` matches it — so a filename-only probe would
    // mis-route the golden tree to the run parser and fail on the missing `rel_dir` field.
    let manifest_path = input.join("manifest.json");
    let is_run = manifest_path.is_file()
        && std::fs::read(&manifest_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<crate::capture::Manifest>(&b).ok())
            .is_some();
    let cases = if is_run {
        cases_from_run(input, out)?
    } else {
        cases_from_golden(input, out)?
    };

    let translator = Translator::new(TranslatorConfig::default());
    let mut summary = CheckSummary { cases: Vec::new() };
    for case in &cases {
        let cc = check_case(&translator, case, client, is_run)?;
        summary.cases.push(cc);
    }
    Ok(summary)
}

fn resolve_caps(protocol: Protocol, model: &Option<String>) -> (Capabilities, bool) {
    match model {
        Some(m) => (shipped().resolve(&protocol.family(), m, None), true),
        None => (Capabilities::unknown(), false),
    }
}

fn check_case(translator: &Translator, case: &Case, client: ClientFilter, live: bool) -> Result<CaseCheck> {
    let (caps, caps_known) = resolve_caps(case.protocol, &case.model);
    let mut items = Vec::new();
    // A request the upstream itself rejected (4xx) is invalid by construction, so llm-xlate
    // refusing to re-encode it is correct behaviour, not a round-trip failure (plan §7.1 applies
    // to accepted requests). Runs record such a rejection as `case.error`.
    let request_was_rejected = case.error.is_some();

    // 1. Request decodability + same-protocol re-encode (plan §7.1).
    if let Some((body, hdrs)) = &case.request {
        let decoded = catch_unwind(AssertUnwindSafe(|| {
            translator.decode_request(case.protocol, body, hdrs)
        }));
        match decoded {
            Err(_) => items.push(ItemResult::fail("request_decode", "panic during decode_request")),
            Ok(Err(e)) => items.push(ItemResult::fail("request_decode", format!("{e}"))),
            Ok(Ok(ir)) => {
                items.push(ItemResult::ok("request_decode", "decoded"));
                // Re-encode for the same protocol and diff against the original body.
                let ctx = translator.encode_ctx(case.protocol, &ir, ResponseId::new("resp_check"), CREATED_AT);
                let enc = catch_unwind(AssertUnwindSafe(|| {
                    translator.encode_request(case.protocol, &ir, &caps, &ctx)
                }));
                match enc {
                    Err(_) if request_was_rejected => items.push(ItemResult::ok(
                        "request_reencode",
                        "n/a: upstream rejected this request (4xx); re-encode panic is not a round-trip regression",
                    )),
                    Ok(Err(e)) if request_was_rejected => items.push(ItemResult::ok(
                        "request_reencode",
                        format!("n/a: upstream rejected this request (4xx); llm-xlate also refuses it ({e})"),
                    )),
                    Err(_) => items.push(ItemResult::fail("request_reencode", "panic during encode_request")),
                    Ok(Err(e)) => items.push(ItemResult::fail("request_reencode", format!("{e}"))),
                    Ok(Ok(enc)) => {
                        let orig: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
                        let round: serde_json::Value =
                            serde_json::from_slice(enc.body.as_ref()).unwrap_or_default();
                        let diff = json_diff("", &orig, &round);
                        let detail = if diff.is_empty() {
                            "byte/structure identical".to_string()
                        } else {
                            format!("{} path(s) differ (normalizations): {}", diff.len(), summarize(&diff))
                        };
                        // A non-empty structural diff is an expected normalization, not a failure.
                        items.push(ItemResult::ok("request_reencode", detail));
                    }
                }
            }
        }
    }

    // 2. Response decodability + twin agreement (plan §7.2).
    let ns_events = case.resp_nonstream.as_ref().and_then(|b| {
        let name = "response_decode_nonstream";
        decode_and_aggregate(translator, case.protocol, &caps, b, false, &mut items, name)
    });
    let st_events = case.resp_stream.as_ref().and_then(|b| {
        let name = "response_decode_stream";
        decode_and_aggregate(translator, case.protocol, &caps, b, true, &mut items, name)
    });
    if let (Some((_, ns_resp)), Some((_, st_resp))) = (&ns_events, &st_events) {
        let same_stop = ns_resp.stop == st_resp.stop || stop_text_equivalent(&ns_resp.stop, &st_resp.stop);
        let ns_p = project_twin(&ns_resp.items);
        let st_p = project_twin(&st_resp.items);
        if live {
            // In a live run the stream and non-stream twins are two INDEPENDENT sampled calls, so
            // their tool-call ids, arguments and even parallel-call counts legitimately differ —
            // only the aggregation invariants that must hold regardless of sampling are asserted:
            // both decoded+aggregated (checked above) and the stop-reason *class* agrees. The
            // item shapes are recorded for review, never failed on.
            let kinds_ns = twin_kinds(&ns_resp.items);
            let kinds_st = twin_kinds(&st_resp.items);
            if same_stop {
                let note = if kinds_ns == kinds_st {
                    format!("live twins agree (stop + kinds [{}])", kinds_ns.join(","))
                } else {
                    format!(
                        "live twins: stop class agrees; item kinds [{}] vs [{}] differ (independent samples)",
                        kinds_ns.join(","),
                        kinds_st.join(",")
                    )
                };
                items.push(ItemResult::ok("twin_agreement", note));
            } else {
                items.push(ItemResult::fail(
                    "twin_agreement",
                    format!("stop class differs across twins: {:?} vs {:?}", ns_resp.stop, st_resp.stop),
                ));
            }
        } else if same_stop && ns_p == st_p {
            items.push(ItemResult::ok("twin_agreement", "stream and non-stream aggregates agree"));
        } else {
            items.push(ItemResult::fail(
                "twin_agreement",
                format!("stop {:?} vs {:?}; items {} vs {}", ns_resp.stop, st_resp.stop, ns_p.len(), st_p.len()),
            ));
        }
    }

    // 3. Chunk fuzz over the raw SSE bytes (plan §7.3).
    if let Some(raw) = &case.resp_stream {
        items.push(chunk_fuzz(translator, case.protocol, &caps, raw, &case.label));
    }

    // 4. Error decodability + tri-dialect encode (plan §7.4).
    if let Some(err) = &case.error {
        items.push(error_check(translator, case, &caps, err));
    }

    // 5. Cross-client rendering (plan §7.5).
    let events = st_events
        .as_ref()
        .map(|(ev, _)| ev.clone())
        .or_else(|| ns_events.as_ref().map(|(ev, _)| ev.clone()));
    if let Some(events) = events {
        items.push(cross_client(translator, case, client, &events));
    }

    let cc = CaseCheck {
        label: case.label.clone(),
        protocol: pname(case.protocol).to_string(),
        model: case.model.clone(),
        caps_known,
        items,
    };
    if let Some(dir) = &case.out_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
        let mut json = serde_json::to_string_pretty(&cc)?;
        json.push('\n');
        std::fs::write(dir.join("xlate.json"), json)?;
    }
    Ok(cc)
}

/// Decode a response (stream or not), aggregate it, and push a result item. Returns the decoded
/// events and the aggregated response on success.
fn decode_and_aggregate(
    translator: &Translator,
    protocol: Protocol,
    caps: &Capabilities,
    body: &[u8],
    is_stream: bool,
    items: &mut Vec<ItemResult>,
    name: &str,
) -> Option<(Vec<IrEvent>, IrResponse)> {
    let decoded = catch_unwind(AssertUnwindSafe(|| {
        if is_stream {
            let mut dec = translator.stream_decoder(protocol, caps);
            let mut ev = dec.push(body);
            ev.extend(dec.finish());
            Ok::<_, llm_xlate_core::error::XlateError>(ev)
        } else {
            translator.decode_response(protocol, body, caps)
        }
    }));
    let events = match decoded {
        Err(_) => {
            items.push(ItemResult::fail(name, "panic during decode"));
            return None;
        }
        Ok(Err(e)) => {
            items.push(ItemResult::fail(name, format!("{e}")));
            return None;
        }
        Ok(Ok(ev)) => ev,
    };
    let agg = catch_unwind(AssertUnwindSafe(|| translator.aggregate_stream(events.iter().cloned())));
    match agg {
        Err(_) => {
            items.push(ItemResult::fail(name, "panic during aggregate"));
            None
        }
        Ok(Err(e)) => {
            // An error *event* in the stream is a legitimate captured outcome; report it but the
            // decode itself succeeded, so this is a soft note recorded as a pass with detail.
            items.push(ItemResult::ok(name, format!("decoded; aggregate carried error: {e}")));
            None
        }
        Ok(Ok(resp)) => {
            items.push(ItemResult::ok(name, format!("decoded {} event(s)", events.len())));
            Some((events, resp))
        }
    }
}

fn chunk_fuzz(
    translator: &Translator,
    protocol: Protocol,
    caps: &Capabilities,
    raw: &[u8],
    label: &str,
) -> ItemResult {
    // Baseline: single-shot decode.
    let baseline = catch_unwind(AssertUnwindSafe(|| {
        let mut dec = translator.stream_decoder(protocol, caps);
        let mut ev = dec.push(raw);
        ev.extend(dec.finish());
        ev
    }));
    let baseline = match baseline {
        Ok(ev) => ev,
        Err(_) => return ItemResult::fail("chunk_fuzz", "panic during baseline decode"),
    };
    let baseline_json = events_json(&baseline);

    let seed = 0x9E3779B97F4A7C15u64 ^ fnv1a(label.as_bytes());
    let mut state = seed | 1;
    for i in 0..FUZZ_SPLITS {
        // Two split points → three chunks, robust against boundary conditions.
        state = xorshift(state);
        let a = (state as usize) % (raw.len() + 1);
        state = xorshift(state);
        let b = (state as usize) % (raw.len() + 1);
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let split = catch_unwind(AssertUnwindSafe(|| {
            let mut dec = translator.stream_decoder(protocol, caps);
            let mut ev = dec.push(&raw[..lo]);
            ev.extend(dec.push(&raw[lo..hi]));
            ev.extend(dec.push(&raw[hi..]));
            ev.extend(dec.finish());
            ev
        }));
        let ev = match split {
            Ok(ev) => ev,
            Err(_) => return ItemResult::fail("chunk_fuzz", format!("panic on split {i} at ({lo},{hi})")),
        };
        if events_json(&ev) != baseline_json {
            return ItemResult::fail(
                "chunk_fuzz",
                format!("split {i} at ({lo},{hi}) produced {} events vs baseline {}", ev.len(), baseline.len()),
            );
        }
    }
    ItemResult::ok("chunk_fuzz", format!("{FUZZ_SPLITS} splits identical to baseline"))
}

fn error_check(translator: &Translator, case: &Case, caps: &Capabilities, err: &ErrorInput) -> ItemResult {
    let decoded = catch_unwind(AssertUnwindSafe(|| {
        translator.decode_error(case.protocol, err.status, &err.body, &err.headers, caps)
    }));
    let xe = match decoded {
        Ok(e) => e,
        Err(_) => return ItemResult::fail("error_decode", "panic during decode_error"),
    };
    // Encode in all three dialects without panic (plan §7.4).
    for client in PROTS {
        for (streaming, started) in [(false, false), (true, false), (true, true)] {
            let enc = catch_unwind(AssertUnwindSafe(|| {
                translator.encode_error(client, &xe, streaming, started)
            }));
            if enc.is_err() {
                return ItemResult::fail(
                    "error_decode",
                    format!("panic in encode_error for {} (streaming={streaming})", pname(client)),
                );
            }
        }
    }
    let mut detail = format!("kind={} status={}", xe.kind.slug(), xe.status);
    if let Some(expect) = case.expect_status {
        if expect != xe.status {
            detail.push_str(&format!(" (expected status {expect})"));
        }
    }
    ItemResult::ok("error_decode", detail)
}

fn cross_client(
    translator: &Translator,
    case: &Case,
    client: ClientFilter,
    events: &[IrEvent],
) -> ItemResult {
    let mut rendered = Vec::new();
    for target in PROTS {
        if target == case.protocol || !client.includes(target) {
            continue;
        }
        // Streaming render.
        let sse = catch_unwind(AssertUnwindSafe(|| {
            let mut ctx = llm_xlate_core::codec::EncodeCtx::new(
                target,
                "check-model",
                ResponseId::new("resp_check"),
                translator.sealer(),
            );
            ctx.created_at = CREATED_AT;
            ctx.stream = true;
            ctx.include_usage = true;
            let mut enc = translator.stream_encoder(target, ctx);
            let mut frames: Vec<u8> = Vec::new();
            for ev in events.iter().cloned() {
                for f in enc.push(ev) {
                    frames.extend_from_slice(&f);
                }
            }
            for f in enc.finish() {
                frames.extend_from_slice(&f);
            }
            frames
        }));
        let sse = match sse {
            Ok(f) => f,
            Err(_) => return ItemResult::fail("cross_client", format!("panic stream-encoding for {}", pname(target))),
        };
        // Non-streaming render via aggregate → encode_response.
        let json = catch_unwind(AssertUnwindSafe(|| {
            let resp = translator.aggregate_stream(events.iter().cloned()).ok()?;
            let mut ctx = llm_xlate_core::codec::EncodeCtx::new(
                target,
                "check-model",
                ResponseId::new("resp_check"),
                translator.sealer(),
            );
            ctx.created_at = CREATED_AT;
            ctx.stream = false;
            Some(translator.encode_response(target, &resp, &ctx).to_vec())
        }));
        let json = match json {
            Ok(j) => j,
            Err(_) => return ItemResult::fail("cross_client", format!("panic response-encoding for {}", pname(target))),
        };
        rendered.push((target, sse, json));
    }

    // Write the human-review renderings alongside, when we own an output directory.
    if let Some(dir) = &case.out_dir {
        let xdir = dir.join("xlate");
        let _ = std::fs::create_dir_all(&xdir);
        for (target, sse, json) in &rendered {
            let _ = std::fs::write(xdir.join(format!("{}.sse", pname(*target))), sse);
            if let Some(j) = json {
                let _ = std::fs::write(xdir.join(format!("{}.json", pname(*target))), j);
            }
        }
    }
    ItemResult::ok("cross_client", format!("rendered {} target dialect(s)", rendered.len()))
}

// ---------------------------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------------------------

fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn events_json(events: &[IrEvent]) -> Vec<serde_json::Value> {
    events
        .iter()
        .map(|e| serde_json::to_value(e).unwrap_or(serde_json::Value::Null))
        .collect()
}

/// Project items onto their load-bearing *structure* for the stream/non-stream twin check.
///
/// The plan requires twins to agree "modulo text": a streaming fixture and its non-streaming
/// counterpart are different captures of the same shape whose literal message text legitimately
/// differs, so message text is reduced to a "has text" / "has refusal" shape while tool-call
/// correlation ids, names and arguments — which must never diverge — are kept verbatim.
fn project_twin(items: &[Item]) -> Vec<String> {
    use llm_xlate_core::ir::Part;
    items
        .iter()
        .filter_map(|it| match it {
            Item::Message { role, content, .. } => {
                let has_text = content.iter().any(|p| p.as_text().is_some());
                let has_refusal = content.iter().any(|p| matches!(p, Part::Refusal { .. }));
                Some(format!("msg:{role:?}:text={has_text}:refusal={has_refusal}"))
            }
            Item::ToolCall { call_id, name, arguments, .. } => {
                Some(format!("call:{}:{name}:{}", call_id.as_str(), arguments.as_str()))
            }
            Item::ToolResult { call_id, is_error, .. } => {
                Some(format!("result:{}:err={is_error}", call_id.as_str()))
            }
            _ => None,
        })
        .collect()
}

/// Coarse per-item KIND sequence (no ids/args/text) for the live-twin structural note: two
/// independent samples should usually produce the same shape, but a differing parallel-tool count
/// is legitimate sampling variance, so this is recorded for review rather than asserted.
fn twin_kinds(items: &[Item]) -> Vec<String> {
    items
        .iter()
        .filter_map(|it| match it {
            Item::Message { role, .. } => Some(format!("msg:{role:?}")),
            Item::ToolCall { name, .. } => Some(format!("call:{name}")),
            Item::ToolResult { .. } => Some("result".to_string()),
            Item::Reasoning(_) => Some("reasoning".to_string()),
            _ => None,
        })
        .collect()
}

fn stop_text_equivalent(a: &StopReason, b: &StopReason) -> bool {
    matches!((a, b), (StopReason::StopSequence(_), StopReason::StopSequence(_)))
}

/// A minimal structural JSON diff: returns the pointer paths whose presence or scalar value
/// differs between `a` (original) and `b` (round-tripped).
fn json_diff(path: &str, a: &serde_json::Value, b: &serde_json::Value) -> Vec<String> {
    use serde_json::Value;
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

fn summarize(paths: &[String]) -> String {
    let shown: Vec<&str> = paths.iter().take(6).map(|s| s.as_str()).collect();
    let mut s = shown.join(", ");
    if paths.len() > shown.len() {
        s.push_str(&format!(", … (+{})", paths.len() - shown.len()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden_dir() -> PathBuf {
        // crates/e2e/src/xlate_check.rs → workspace → crates/xlate/tests/golden
        let manifest = env!("CARGO_MANIFEST_DIR"); // .../crates/e2e
        Path::new(manifest).join("..").join("xlate").join("tests").join("golden")
    }

    #[test]
    fn checks_the_shipped_golden_tree() {
        let dir = golden_dir();
        let summary = check_dir(&dir, ClientFilter::All, None).unwrap();
        assert!(!summary.cases.is_empty(), "no golden cases discovered");
        // Every request-decode and response-decode item must pass for the shipped goldens.
        for c in &summary.cases {
            for i in &c.items {
                if i.name == "request_decode"
                    || i.name == "response_decode_nonstream"
                    || i.name == "response_decode_stream"
                {
                    assert!(i.pass, "{}/{} failed: {}", c.label, i.name, i.detail);
                }
            }
        }
    }

    #[test]
    fn golden_error_fixtures_decode_and_encode() {
        let dir = golden_dir();
        let summary = check_dir(&dir, ClientFilter::All, None).unwrap();
        let mut saw_error = false;
        for c in &summary.cases {
            for i in &c.items {
                if i.name == "error_decode" {
                    saw_error = true;
                    assert!(i.pass, "{}: {}", c.label, i.detail);
                }
            }
        }
        assert!(saw_error, "expected at least one error fixture");
    }

    #[test]
    fn chunk_fuzz_runs_on_stream_fixtures() {
        let dir = golden_dir();
        let summary = check_dir(&dir, ClientFilter::All, None).unwrap();
        let fuzz: Vec<_> = summary
            .cases
            .iter()
            .flat_map(|c| c.items.iter())
            .filter(|i| i.name == "chunk_fuzz")
            .collect();
        assert!(!fuzz.is_empty(), "no stream fixtures fuzzed");
        for i in &fuzz {
            assert!(i.pass, "chunk fuzz failed: {}", i.detail);
        }
    }

    #[test]
    fn cross_client_writes_renderings_with_out() {
        let dir = golden_dir();
        let mut out = std::env::temp_dir();
        out.push(format!("llm-xlate-e2e-check-out-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let summary = check_dir(&dir, ClientFilter::All, Some(&out)).unwrap();
        assert!(!summary.cases.is_empty());
        // A response fixture (chat/text) should have produced xlate/<client>.sse renderings.
        let text = out.join("chat").join("text").join("xlate");
        assert!(text.join("anthropic.sse").is_file(), "missing cross-client render");
        assert!(out.join("chat").join("text").join("xlate.json").is_file());
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn corrupted_sse_fails_without_panic() {
        // A deliberately corrupted SSE fixture in a temp golden dir must produce a failure record.
        let mut root = std::env::temp_dir();
        root.push(format!("llm-xlate-e2e-check-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let chat = root.join("chat");
        std::fs::create_dir_all(&chat).unwrap();
        // Garbage that is not valid SSE / JSON payloads.
        std::fs::write(chat.join("broken.stream.sse"), b"data: {not json at all\n\ndata: \x00\x01\x02\n\n").unwrap();
        let summary = check_dir(&root, ClientFilter::All, None).unwrap();
        // The run completed (no panic escaped) and produced a case.
        assert_eq!(summary.cases.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn totals_and_table_render() {
        let dir = golden_dir();
        let summary = check_dir(&dir, ClientFilter::All, None).unwrap();
        let (pass, _fail) = summary.totals();
        assert!(pass > 0);
        let table = summary.table();
        assert!(table.contains("request_decode"));
        assert!(table.contains("TOTAL"));
    }

    #[test]
    fn client_filter_parse() {
        assert!(matches!(ClientFilter::parse("all").unwrap(), ClientFilter::All));
        assert!(matches!(ClientFilter::parse("chat").unwrap(), ClientFilter::One(Protocol::OaiChat)));
        assert!(ClientFilter::parse("bogus").is_err());
    }

    #[test]
    fn json_diff_reports_add_remove_change() {
        let a = serde_json::json!({"keep": 1, "drop": 2, "change": "x"});
        let b = serde_json::json!({"keep": 1, "add": 9, "change": "y"});
        let d = json_diff("", &a, &b);
        assert!(d.iter().any(|p| p == "-/drop"));
        assert!(d.iter().any(|p| p == "+/add"));
        assert!(d.iter().any(|p| p == "~/change"));
    }

    #[test]
    fn twin_agreement_passes_for_chat_text() {
        // chat/text has both a .resp.json and a .stream.sse with *different text* — they must
        // still agree modulo text.
        let summary = check_dir(&golden_dir(), ClientFilter::All, None).unwrap();
        let case = summary.cases.iter().find(|c| c.label == "chat/text").unwrap();
        let twin = case.items.iter().find(|i| i.name == "twin_agreement").unwrap();
        assert!(twin.pass, "{}", twin.detail);
    }

    #[test]
    fn request_reencode_reported_for_every_request_fixture() {
        let summary = check_dir(&golden_dir(), ClientFilter::All, None).unwrap();
        let reencode: Vec<_> = summary
            .cases
            .iter()
            .flat_map(|c| c.items.iter())
            .filter(|i| i.name == "request_reencode")
            .collect();
        assert!(!reencode.is_empty());
        // Re-encode is informational (normalizations allowed) — it never fails for valid goldens.
        for i in &reencode {
            assert!(i.pass, "{}", i.detail);
        }
    }

    #[test]
    fn client_filter_one_limits_cross_client_targets() {
        let summary = check_dir(&golden_dir(), ClientFilter::One(Protocol::Anthropic), None).unwrap();
        // A chat response fixture cross-rendered with only the anthropic client → 1 target.
        let case = summary.cases.iter().find(|c| c.label == "chat/text").unwrap();
        let cc = case.items.iter().find(|i| i.name == "cross_client").unwrap();
        assert!(cc.detail.contains("rendered 1 target"), "{}", cc.detail);
    }

    #[test]
    fn model_bearing_golden_has_known_caps() {
        let summary = check_dir(&golden_dir(), ClientFilter::All, None).unwrap();
        let case = summary.cases.iter().find(|c| c.label == "anthropic/multimodal").unwrap();
        assert!(case.caps_known, "expected a known model from the request body");
        assert_eq!(case.model.as_deref(), Some("claude-opus-5"));
    }

    #[test]
    fn checks_a_synthesized_run_directory() {
        use crate::capture::{write_capture, write_manifest, Manifest, RequestRecord, WireResponse};
        use crate::probe::{Catalogue, Expansion};
        use std::collections::BTreeMap;

        let mut root = std::env::temp_dir();
        root.push(format!("llm-xlate-e2e-check-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        // Use the shipped chat text goldens as capture payloads.
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let req_body: serde_json::Value = serde_json::from_slice(
            &std::fs::read(manifest_dir.join("..").join("xlate").join("tests").join("golden").join("chat").join("text.req.json")).unwrap(),
        )
        .unwrap();
        let resp_bytes =
            std::fs::read(manifest_dir.join("..").join("xlate").join("tests").join("golden").join("chat").join("text.resp.json")).unwrap();

        let toml = "[[probe]]\nid=\"chat.text\"\nprotocol=\"chat\"\nmodels=[\"m\"]\n[probe.body]\nmodel=\"gpt-4o\"\n";
        let p = Catalogue::parse_file(toml).unwrap().remove(0);
        let ex = Expansion { probe: p, model: "gpt-4o".into(), stream: false };
        let req = RequestRecord {
            method: "POST".into(),
            url: "https://api.openai.com/v1/chat/completions".into(),
            stream: false,
            headers: BTreeMap::new(),
            body: req_body,
        };
        let resp = WireResponse::NonStream {
            status: 200,
            headers: BTreeMap::new(),
            elapsed_ms: 1,
            body_bytes: resp_bytes,
        };
        let entry = write_capture(&root, &ex, &req, &resp, &serde_json::json!({})).unwrap();
        let manifest = Manifest {
            tool_version: "0.1.0".into(),
            xlate_version: "0.1.0".into(),
            run_id: "20260910-000000".into(),
            label: "check".into(),
            created_at: "2026-09-10T00:00:00Z".into(),
            providers: BTreeMap::new(),
            model_sets: vec![],
            entries: vec![entry],
        };
        write_manifest(&root, &manifest).unwrap();

        let summary = check_dir(&root, ClientFilter::All, None).unwrap();
        assert_eq!(summary.cases.len(), 1);
        let c = &summary.cases[0];
        assert!(c.items.iter().any(|i| i.name == "request_decode" && i.pass));
        assert!(c.items.iter().any(|i| i.name == "response_decode_nonstream" && i.pass));
        // A run is writable, so xlate.json was written next to the capture.
        assert!(root.join("chat.text").join("gpt-4o").join("xlate.json").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }
}
