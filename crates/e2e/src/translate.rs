//! `translate` (plan §8): translate a captured or golden request from one client protocol to
//! another, optionally sending it for real, and report the round trip.
//!
//! The pure part — `decode_request → requirements → lower → encode_request` — is exactly the
//! path the golden-matrix runner drives, so a `--dry-run` upstream body is byte-for-byte what
//! [`llm_xlate::Translator`] produces. Live sends (default on; `--dry-run` disables) reuse the
//! skeleton's [`crate::client`] and obey the same guard rails as `run`.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

use llm_xlate::Translator;
use llm_xlate_core::caps::shipped;
use llm_xlate_core::codec::{EncodedRequest, TranslatorConfig, UnresolvedReasoning};
use llm_xlate_core::ir::{Protocol, ResponseId};
use llm_xlate_core::HeaderMap;

use crate::assets::Assets;
use crate::models::{ModelSets, ModelsSpec};
use crate::probe::{substitute, Catalogue, RefSource, SubstCtx};

/// Describe the router-side requirements a request still has as short lines.
fn describe_requirements(reqs: &llm_xlate::requirements::Requirements) -> Vec<String> {
    let mut out = Vec::new();
    for c in &reqs.reasoning_for_calls {
        out.push(format!("reasoning blob for tool call {}", c.as_str()));
    }
    for f in &reqs.foreign_files {
        out.push(format!("foreign file {} (family {})", f.id, f.family.label()));
    }
    if let Some(chain) = &reqs.chain {
        out.push(format!("previous_response_id chain {}", chain.as_str()));
    }
    out
}

/// A fixed timestamp so dry-run bodies are deterministic (matches the golden-matrix runner).
pub const CREATED_AT: u64 = 1_700_000_000;

fn core_protocol(s: &str) -> Result<Protocol> {
    Ok(match s {
        "chat" => Protocol::OaiChat,
        "responses" => Protocol::OaiResponses,
        "anthropic" => Protocol::Anthropic,
        other => bail!("unknown protocol `{other}` (want chat|responses|anthropic)"),
    })
}

fn pname(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "chat",
        Protocol::OaiResponses => "responses",
        Protocol::Anthropic => "anthropic",
    }
}

/// A representative upstream model for a target protocol when none is specified.
fn default_model(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "gpt-4o",
        Protocol::OaiResponses => "gpt-5.4",
        Protocol::Anthropic => "claude-opus-5",
    }
}

/// Where a scenario's source request body comes from (exactly one is set in the TOML).
#[derive(Debug, Clone)]
pub enum Source {
    /// A `*.req.json` path relative to `crates/e2e` (or absolute).
    Request(String),
    /// A probe id whose `[probe.body]` (assets/`$MODEL` substituted, refs disallowed) is the body.
    Probe(String),
}

/// A source request (one row of `[[scenario]]`).
#[derive(Debug, Clone)]
pub struct Scenario {
    /// Scenario key (the report groups by this).
    pub name: String,
    /// Free-text kind tag (text, tools_loop, …).
    pub kind: String,
    /// The source wire protocol.
    pub source: Protocol,
    /// Where the body comes from.
    pub src: Source,
}

/// A directed source→target hop with a target model set (one row of `[[pair]]`).
#[derive(Debug, Clone)]
pub struct Pair {
    /// Source protocol (must match a scenario's `source` to run).
    pub source: Protocol,
    /// Target protocol.
    pub target: Protocol,
    /// A model-set name from `models.toml`; the cheap (first) member is used.
    pub target_models: String,
}

/// The whole scenario file: source requests plus the directed pairs to run them across.
#[derive(Debug, Clone)]
pub struct ScenarioSet {
    /// The source requests.
    pub scenarios: Vec<Scenario>,
    /// The directed pairs.
    pub pairs: Vec<Pair>,
}

#[derive(Debug, serde::Deserialize)]
struct RawFile {
    #[serde(default)]
    scenario: Vec<RawScenario>,
    #[serde(default)]
    pair: Vec<RawPair>,
}

#[derive(Debug, serde::Deserialize)]
struct RawScenario {
    name: String,
    #[serde(default)]
    kind: String,
    source: String,
    #[serde(default)]
    request: Option<String>,
    #[serde(default)]
    probe: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct RawPair {
    source: String,
    target: String,
    target_models: String,
}

/// Load the two-table scenario file (`[[scenario]]` + `[[pair]]`).
pub fn load_scenarios(path: &Path) -> Result<ScenarioSet> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let raw: RawFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let mut scenarios = Vec::new();
    for s in raw.scenario {
        let source = core_protocol(&s.source)?;
        let src = match (s.request, s.probe) {
            (Some(r), None) => Source::Request(r),
            (None, Some(p)) => Source::Probe(p),
            (Some(_), Some(_)) => bail!("scenario `{}` sets both `request` and `probe`", s.name),
            (None, None) => bail!("scenario `{}` sets neither `request` nor `probe`", s.name),
        };
        scenarios.push(Scenario { name: s.name, kind: s.kind, source, src });
    }
    let mut pairs = Vec::new();
    for p in raw.pair {
        pairs.push(Pair {
            source: core_protocol(&p.source)?,
            target: core_protocol(&p.target)?,
            target_models: p.target_models,
        });
    }
    Ok(ScenarioSet { scenarios, pairs })
}

/// Resolve the default scenarios path pair under a crate directory.
pub fn scenarios_path(e2e_dir: &Path, explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    let real = e2e_dir.join("translate").join("scenarios.toml");
    if real.is_file() {
        real
    } else {
        e2e_dir.join("translate").join("scenarios.example.toml")
    }
}

/// A directed pair filter parsed from `--pair chat->anthropic`.
#[derive(Debug, Clone, Copy)]
pub struct PairFilter {
    /// Source protocol.
    pub client: Protocol,
    /// Target protocol.
    pub target: Protocol,
}

impl PairFilter {
    /// Parse `chat->anthropic`.
    pub fn parse(s: &str) -> Result<Self> {
        let (a, b) = s.split_once("->").context("--pair must look like chat->anthropic")?;
        Ok(PairFilter {
            client: core_protocol(a.trim())?,
            target: core_protocol(b.trim())?,
        })
    }
}

/// The outcome of translating one scenario for one target protocol (dry-run view).
#[derive(Debug, Clone, Serialize)]
pub struct TranslateOutcome {
    /// Scenario name.
    pub scenario: String,
    /// Scenario kind tag.
    pub kind: String,
    /// Source client protocol.
    pub client: String,
    /// Target protocol.
    pub target: String,
    /// Upstream model used.
    pub model: String,
    /// Whether the upstream body requested streaming.
    pub upstream_streams: bool,
    /// The upstream request body bytes (empty when skipped/errored).
    #[serde(skip)]
    pub upstream_body: Vec<u8>,
    /// Upstream header names + values recorded for the report.
    pub upstream_headers: Vec<(String, String)>,
    /// Rendered lowering + wiring degradations.
    pub degradations: Vec<String>,
    /// Unresolved requirements (empty means fully resolvable).
    pub unresolved: Vec<String>,
    /// An `XlateError` message, when a step rejected the request.
    pub error: Option<String>,
    /// A skip reason (unresolved requirements without `--strip`).
    pub skipped: Option<String>,
}

/// Options for a translate invocation.
#[derive(Debug, Clone)]
pub struct TranslateOpts {
    /// Restrict to this directed pair.
    pub pair: Option<PairFilter>,
    /// Override the upstream model for every pair.
    pub model: Option<String>,
    /// Continue past unresolved reasoning with `StripAndDegrade`.
    pub strip: bool,
}

/// The engine used to build upstream bodies. Kept separate so tests can reproduce it exactly.
pub struct Engine {
    /// The plain translator (default config).
    pub translator: Translator,
    /// The strip-and-degrade translator (only differs in `unresolved_reasoning`).
    pub strip_translator: Translator,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Build both translators.
    pub fn new() -> Self {
        let strip_cfg = TranslatorConfig {
            unresolved_reasoning: UnresolvedReasoning::StripAndDegrade,
            ..TranslatorConfig::default()
        };
        Engine {
            translator: Translator::new(TranslatorConfig::default()),
            strip_translator: Translator::new(strip_cfg),
        }
    }

    /// Build the upstream request for one (client, target, model) triple from a source body.
    ///
    /// This is the exact `decode → requirements → lower → encode` path the golden-matrix runner
    /// uses, so the resulting body matches [`Translator`] byte for byte.
    pub fn build_upstream(
        &self,
        client: Protocol,
        target: Protocol,
        model: &str,
        body: &[u8],
        strip: bool,
    ) -> Result<(EncodedRequest, Vec<String>), String> {
        let tr = if strip { &self.strip_translator } else { &self.translator };
        let ir = tr.decode_request(client, body, &HeaderMap::new()).map_err(|e| format!("{e}"))?;
        let caps = shipped().resolve(&target.family(), model, None);
        let ctx = tr.encode_ctx(client, &ir, ResponseId::new("resp_xlate"), CREATED_AT);
        let mut ir2 = ir.clone();
        ir2.model.resolved = Some(model.to_string());
        let low = tr.lower(ir2, &caps, target, &llm_xlate::requirements::Resolutions::new())
            .map_err(|e| format!("{e}"))?;
        let mut enc = tr.encode_request(target, &low.req, &caps, &ctx).map_err(|e| format!("{e}"))?;
        // Merge lowering degradations ahead of wiring degradations (as `translate_request` does).
        let mut degr = low.degradations;
        degr.extend(std::mem::take(&mut enc.degradations));
        let rendered: Vec<String> = degr.iter().map(|d| format!("{}: {}", d.field, d.detail)).collect();
        enc.degradations = degr;
        Ok((enc, rendered))
    }
}

/// Everything the source-resolution and model-set lookup need at run time.
pub struct TranslateCtx {
    /// The `crates/e2e` directory (scenario `request` paths are relative to it).
    pub e2e_dir: PathBuf,
    /// The named model sets from `models.toml`.
    pub models: ModelSets,
    /// The probe catalogue, for `probe`-sourced scenarios.
    pub catalogue: Catalogue,
    /// Bundled deterministic assets, for `$ASSET` substitution in probe bodies.
    pub assets: Assets,
}

impl TranslateCtx {
    /// Load the context for the crate's `crates/e2e` directory.
    pub fn load(e2e_dir: &Path) -> Result<Self> {
        let models = ModelSets::load(&e2e_dir.join("models.toml"))?;
        let catalogue = Catalogue::load(&e2e_dir.join("probes"), &models)?;
        Ok(TranslateCtx {
            e2e_dir: e2e_dir.to_path_buf(),
            models,
            catalogue,
            assets: Assets,
        })
    }
}

/// A `RefSource` that refuses every `${ref:…}` — translate sources must be ref-free (plan §8).
struct NoRefs;
impl RefSource for NoRefs {
    fn resolve(&self, probe_id: &str, pointer: &str) -> Result<serde_json::Value> {
        bail!("source references ${{ref:{probe_id}:{pointer}}}; pick a ref-free probe or a fixture path")
    }
}

/// The `$MODEL` a source body is materialized with (it becomes the client model name; the target
/// model is applied separately at lowering time).
fn source_default_model(p: Protocol) -> &'static str {
    default_model(p)
}

/// Read (or materialize) a scenario's source request body.
pub fn read_source(s: &Scenario, ctx: &TranslateCtx) -> Result<Vec<u8>> {
    match &s.src {
        Source::Request(p) => {
            let path = if Path::new(p).is_absolute() {
                PathBuf::from(p)
            } else {
                ctx.e2e_dir.join(p)
            };
            std::fs::read(&path).with_context(|| format!("reading source {}", path.display()))
        }
        Source::Probe(id) => {
            let probe = ctx
                .catalogue
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("scenario probe source `{id}` not found in the catalogue"))?;
            let refs = NoRefs;
            let subst = SubstCtx {
                model: source_default_model(s.source),
                run_id: "dryrun",
                assets: &ctx.assets,
                refs: &refs,
            };
            let body = substitute(&probe.body, &subst)
                .with_context(|| format!("materializing probe source `{id}`"))?;
            Ok(serde_json::to_vec(&body)?)
        }
    }
}

/// A single unit of work: one (scenario, pair) with a resolved model and a loaded body.
struct Job {
    scenario: Scenario,
    target: Protocol,
    model: String,
    body: std::result::Result<Vec<u8>, String>,
}

fn pair_selected(p: &Pair, opts: &TranslateOpts) -> bool {
    match opts.pair {
        Some(pf) => pf.client == p.source && pf.target == p.target,
        None => true,
    }
}

/// The cross product of pairs × matching scenarios, with the target model resolved.
fn build_jobs(set: &ScenarioSet, opts: &TranslateOpts, ctx: &TranslateCtx) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for p in &set.pairs {
        if !pair_selected(p, opts) {
            continue;
        }
        let model = match &opts.model {
            Some(m) => m.clone(),
            None => {
                let ids = ctx
                    .models
                    .resolve(&ModelsSpec::Set(p.target_models.clone()), true)
                    .with_context(|| format!("resolving model set `{}`", p.target_models))?;
                ids.into_iter()
                    .next()
                    .unwrap_or_else(|| default_model(p.target).to_string())
            }
        };
        for s in &set.scenarios {
            if s.source != p.source {
                continue;
            }
            let body = read_source(s, ctx).map_err(|e| format!("{e}"));
            jobs.push(Job {
                scenario: s.clone(),
                target: p.target,
                model: model.clone(),
                body,
            });
        }
    }
    Ok(jobs)
}

/// Compute the dry-run outcomes for the whole scenario set.
pub fn dry_run(
    set: &ScenarioSet,
    opts: &TranslateOpts,
    ctx: &TranslateCtx,
) -> Result<Vec<TranslateOutcome>> {
    let engine = Engine::new();
    let jobs = build_jobs(set, opts, ctx)?;
    let mut out = Vec::new();
    for job in jobs {
        out.push(build_outcome(&engine, &job, opts.strip));
    }
    Ok(out)
}

fn build_outcome(engine: &Engine, job: &Job, strip: bool) -> TranslateOutcome {
    let s = &job.scenario;
    let target = job.target;
    let model = &job.model;
    let mut outcome = TranslateOutcome {
        scenario: s.name.clone(),
        kind: s.kind.clone(),
        client: pname(s.source).to_string(),
        target: pname(target).to_string(),
        model: model.clone(),
        upstream_streams: false,
        upstream_body: Vec::new(),
        upstream_headers: Vec::new(),
        degradations: Vec::new(),
        unresolved: Vec::new(),
        error: None,
        skipped: None,
    };

    let body = match &job.body {
        Ok(b) => b,
        Err(e) => {
            outcome.error = Some(format!("source: {e}"));
            return outcome;
        }
    };

    // Requirements pre-pass on the plain translator.
    let tr = &engine.translator;
    let ir = match tr.decode_request(s.source, body, &HeaderMap::new()) {
        Ok(ir) => ir,
        Err(e) => {
            outcome.error = Some(format!("decode: {e}"));
            return outcome;
        }
    };
    let caps = shipped().resolve(&target.family(), model, None);
    let reqs = tr.requirements(&ir, &caps, target);
    outcome.unresolved = describe_requirements(&reqs);
    if !outcome.unresolved.is_empty() && !strip {
        outcome.skipped = Some("unresolved requirements (pass --strip to degrade)".to_string());
        return outcome;
    }

    match engine.build_upstream(s.source, target, model, body, strip) {
        Ok((enc, degr)) => {
            outcome.upstream_streams = enc.upstream_streams;
            outcome.upstream_body = enc.body.to_vec();
            let mut headers: Vec<(String, String)> = enc
                .headers
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned()))
                .collect();
            headers.sort();
            outcome.upstream_headers = headers;
            outcome.degradations = degr;
        }
        Err(e) => outcome.error = Some(e),
    }
    outcome
}

/// Render the dry-run outcomes as a human-readable report.
pub fn render_dry_run(outcomes: &[TranslateOutcome]) -> String {
    let mut out = String::new();
    for o in outcomes {
        out.push_str(&format!("=== {} : {} -> {} ({}) ===\n", o.scenario, o.client, o.target, o.model));
        if let Some(skip) = &o.skipped {
            out.push_str(&format!("SKIPPED: {skip}\n"));
            for u in &o.unresolved {
                out.push_str(&format!("  unresolved: {u}\n"));
            }
            continue;
        }
        if let Some(err) = &o.error {
            out.push_str(&format!("XlateError: {err}\n"));
            continue;
        }
        out.push_str(&format!("upstream_streams: {}\n", o.upstream_streams));
        out.push_str("headers:\n");
        for (k, v) in &o.upstream_headers {
            out.push_str(&format!("  {k}: {v}\n"));
        }
        out.push_str("degradations:\n");
        if o.degradations.is_empty() {
            out.push_str("  (none)\n");
        } else {
            for d in &o.degradations {
                out.push_str(&format!("  {d}\n"));
            }
        }
        out.push_str("body:\n");
        match serde_json::from_slice::<serde_json::Value>(&o.upstream_body) {
            Ok(v) => out.push_str(&serde_json::to_string_pretty(&v).unwrap_or_default()),
            Err(_) => out.push_str(&String::from_utf8_lossy(&o.upstream_body)),
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Live send (plan §8). Never exercised in the offline test suite; runs only without --dry-run.
// ---------------------------------------------------------------------------------------------

use crate::capture::WireResponse;
use crate::client::{Built, Client, ClientConfig};
use crate::keys::{is_secret_header, Provider};
use crate::probe::{sanitize, Protocol as EProtocol};

fn e_protocol(p: Protocol) -> EProtocol {
    match p {
        Protocol::OaiChat => EProtocol::Chat,
        Protocol::OaiResponses => EProtocol::Responses,
        Protocol::Anthropic => EProtocol::Anthropic,
    }
}

/// Construct a client [`Built`] for a target upstream request from an encoded body.
fn upstream_built(cfg: &ClientConfig, target: Protocol, enc: &EncodedRequest) -> Result<Built> {
    let (provider, url) = match target {
        Protocol::OaiChat => (
            Provider::OpenAI,
            format!("{}/v1/chat/completions", cfg.openai_base.trim_end_matches('/')),
        ),
        Protocol::OaiResponses => {
            (Provider::OpenAI, format!("{}/v1/responses", cfg.openai_base.trim_end_matches('/')))
        }
        Protocol::Anthropic => {
            (Provider::Anthropic, format!("{}/v1/messages", cfg.anthropic_base.trim_end_matches('/')))
        }
    };
    let wire_body: serde_json::Value = serde_json::from_slice(enc.body.as_ref())
        .context("parsing encoded upstream body")?;
    let mut headers = std::collections::BTreeMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    match provider {
        Provider::Anthropic => {
            headers.insert("x-api-key".into(), "<redacted>".into());
            headers.insert("anthropic-version".into(), cfg.anthropic_version.clone());
        }
        Provider::OpenAI => {
            headers.insert("authorization".into(), "<redacted>".into());
        }
    }
    let mut extra = std::collections::BTreeMap::new();
    for (k, v) in enc.headers.iter() {
        let name = k.as_str().to_ascii_lowercase();
        if is_secret_header(&name) || name == "content-type" || name == "anthropic-version" {
            continue;
        }
        let val = String::from_utf8_lossy(v.as_bytes()).to_string();
        extra.insert(name.clone(), val.clone());
        headers.insert(name, val);
    }
    Ok(Built {
        provider,
        protocol: e_protocol(target),
        url,
        stream: enc.upstream_streams,
        wire_body,
        headers,
        extra_headers: extra,
    })
}

fn write_pretty_guarded(path: &Path, value: &serde_json::Value) -> Result<()> {
    let mut s = serde_json::to_string_pretty(value)?;
    s.push('\n');
    write_bytes_guarded(path, s.as_bytes())
}

/// Write raw bytes to `path`, refusing anything that looks like a key (defence in depth on
/// provider/derived response bytes, mirroring the capture writer's guard).
fn write_bytes_guarded(path: &Path, bytes: &[u8]) -> Result<()> {
    if crate::keys::looks_like_key(&String::from_utf8_lossy(bytes)) {
        bail!("refusing to write {}: contains a key-shaped string", path.display());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Send each scenario × target upstream request for real, capturing the round trip under
/// `<run_dir>/translate/<scenario>__<client>_to_<target>__<model>/`. Obeys `--max-requests`.
pub async fn live_run(
    set: &ScenarioSet,
    opts: &TranslateOpts,
    ctx: &TranslateCtx,
    run_dir: &Path,
    max_requests: usize,
) -> Result<String> {
    let engine = Engine::new();
    let cfg = ClientConfig::default();
    let client = Client::new(cfg.clone())?;

    // Build the job list first so --max-requests can abort before any send.
    let jobs = build_jobs(set, opts, ctx)?;
    if jobs.len() > max_requests {
        bail!("translate would issue {} requests, exceeding --max-requests {max_requests}", jobs.len());
    }

    let tr = &engine.translator;
    let mut report = String::new();
    for job in jobs {
        let s = &job.scenario;
        let target = job.target;
        let model = job.model.clone();
        let case = format!(
            "{}__{}_to_{}__{}",
            s.name,
            pname(s.source),
            pname(target),
            sanitize(&model)
        );
        report.push_str(&format!("=== {case} ===\n"));
        let dir = run_dir.join("translate").join(&case);
        std::fs::create_dir_all(&dir)?;

        let body = match &job.body {
            Ok(b) => b.clone(),
            Err(e) => {
                report.push_str(&format!("source error: {e}\n"));
                continue;
            }
        };

        let (enc, degr) = match engine.build_upstream(s.source, target, &model, &body, opts.strip) {
            Ok(v) => v,
            Err(e) => {
                report.push_str(&format!("XlateError: {e}\n"));
                std::fs::write(dir.join("degradations.txt"), format!("XlateError: {e}\n"))?;
                continue;
            }
        };
        std::fs::write(dir.join("degradations.txt"), degr.join("\n") + "\n")?;

        let built = upstream_built(&cfg, target, &enc)?;
        // Record the redacted upstream request (asset payloads rewritten).
        let rec = built.request_record();
        let disk = serde_json::json!({
            "method": rec.method, "url": rec.url, "stream": rec.stream,
            "headers": rec.headers, "body": crate::assets::redact_assets(&rec.body),
        });
        write_pretty_guarded(&dir.join("upstream_request.json"), &disk)?;

        let resp = client.send(&built).await?;
        let caps = shipped().resolve(&target.family(), &model, None);
        let (status, upstream_events) = match &resp {
            WireResponse::NonStream { status, headers, body_bytes, elapsed_ms } => {
                let body: serde_json::Value = serde_json::from_slice(body_bytes)
                    .unwrap_or_else(|_| serde_json::json!({"_unparsed": String::from_utf8_lossy(body_bytes)}));
                let env = serde_json::json!({"status": status, "headers": headers, "elapsed_ms": elapsed_ms, "body": body});
                write_pretty_guarded(&dir.join("upstream_response.json"), &env)?;
                let ev = if *status < 400 {
                    tr.decode_response(target, body_bytes, &caps).ok()
                } else {
                    None
                };
                (*status, ev)
            }
            WireResponse::Stream { status, raw, .. } => {
                write_bytes_guarded(&dir.join("upstream_response.sse"), raw)?;
                let mut dec = tr.stream_decoder(target, &caps);
                let mut ev = dec.push(raw);
                ev.extend(dec.finish());
                (*status, Some(ev))
            }
        };
        report.push_str(&format!("upstream status: {status}\n"));

        if let Some(events) = upstream_events {
            // Aggregate the upstream reply and render it back for the original client protocol.
            let upstream_calls = tool_call_ids(&events);
            if let Ok(up_resp) = tr.aggregate_stream(events.iter().cloned()) {
                report.push_str(&format!("upstream stop: {:?}\n", up_resp.stop));
            }
            // Build a client-side EncodeCtx from the original request.
            if let Ok(client_ir) = tr.decode_request(s.source, &body, &HeaderMap::new()) {
                let mut ctx =
                    tr.encode_ctx(s.source, &client_ir, ResponseId::new("resp_xlate"), CREATED_AT);
                ctx.stream = false;
                if let Ok(cresp) = tr.aggregate_stream(events.iter().cloned()) {
                    let cbody = tr.encode_response(s.source, &cresp, &ctx);
                    write_bytes_guarded(&dir.join("client_response.json"), cbody.as_ref())?;
                    report.push_str(&format!("client stop: {:?}\n", cresp.stop));
                }
                // Streaming rendering.
                let mut sctx =
                    tr.encode_ctx(s.source, &client_ir, ResponseId::new("resp_xlate"), CREATED_AT);
                sctx.stream = true;
                let mut senc = tr.stream_encoder(s.source, sctx);
                let mut frames: Vec<u8> = Vec::new();
                for ev in events.iter().cloned() {
                    for f in senc.push(ev) {
                        frames.extend_from_slice(&f);
                    }
                }
                for f in senc.finish() {
                    frames.extend_from_slice(&f);
                }
                write_bytes_guarded(&dir.join("client_stream.sse"), &frames)?;
                // Tool-call id preservation (set equality upstream vs client rendering).
                let client_ir_events = tr.decode_response(s.source, {
                    &std::fs::read(dir.join("client_response.json")).unwrap_or_default()
                }, &shipped().resolve(&s.source.family(), &client_ir.model.client_name, None));
                if let Ok(cev) = client_ir_events {
                    let client_calls = tool_call_ids(&cev);
                    let preserved = upstream_calls == client_calls;
                    report.push_str(&format!("tool-call ids preserved: {preserved}\n"));
                }
            }
        }
    }
    Ok(report)
}

fn tool_call_ids(events: &[llm_xlate_core::ir::IrEvent]) -> std::collections::BTreeSet<String> {
    use llm_xlate_core::ir::Item;
    let mut out = std::collections::BTreeSet::new();
    let tr = Translator::new(TranslatorConfig::default());
    if let Ok(resp) = tr.aggregate_stream(events.iter().cloned()) {
        for it in &resp.items {
            if let Item::ToolCall { call_id, .. } = it {
                out.insert(call_id.as_str().to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e2e_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn ctx() -> TranslateCtx {
        TranslateCtx::load(&e2e_dir()).unwrap()
    }

    /// The authoritative scenario file the catalogue engineer wrote.
    fn real_set() -> ScenarioSet {
        load_scenarios(&e2e_dir().join("translate").join("scenarios.toml")).unwrap()
    }

    fn example_set() -> ScenarioSet {
        load_scenarios(&e2e_dir().join("translate").join("scenarios.example.toml")).unwrap()
    }

    #[test]
    fn real_scenarios_load_with_both_tables() {
        let set = real_set();
        assert!(!set.scenarios.is_empty());
        assert_eq!(set.pairs.len(), 6, "expected the six directed pairs");
    }

    #[test]
    fn example_scenarios_load() {
        let set = example_set();
        assert_eq!(set.pairs.len(), 6);
        assert_eq!(set.scenarios.len(), 3);
    }

    #[test]
    fn dry_run_covers_all_six_pairs_over_the_real_file() {
        let opts = TranslateOpts { pair: None, model: None, strip: false };
        let outcomes = dry_run(&real_set(), &opts, &ctx()).unwrap();
        let pairs: std::collections::BTreeSet<(String, String)> = outcomes
            .iter()
            .map(|o| (o.client.clone(), o.target.clone()))
            .collect();
        assert_eq!(pairs.len(), 6, "expected six distinct directed pairs, got {pairs:?}");
    }

    #[test]
    fn example_dry_run_has_one_outcome_per_pair() {
        // The example has exactly one scenario per source protocol, so each pair yields one.
        let opts = TranslateOpts { pair: None, model: None, strip: false };
        let outcomes = dry_run(&example_set(), &opts, &ctx()).unwrap();
        assert_eq!(outcomes.len(), 6);
    }

    #[test]
    fn dry_run_bodies_match_translator_byte_for_byte() {
        let opts = TranslateOpts { pair: None, model: None, strip: false };
        let ctx = ctx();
        let set = real_set();
        let outcomes = dry_run(&set, &opts, &ctx).unwrap();

        // Reproduce each produced upstream body independently through a fresh Translator.
        let tr = Translator::new(TranslatorConfig::default());
        let mut checked = 0;
        for o in &outcomes {
            if o.error.is_some() || o.skipped.is_some() {
                continue;
            }
            // Find the matching scenario (name + source protocol) and re-read its body.
            let s = set
                .scenarios
                .iter()
                .find(|s| s.name == o.scenario && pname(s.source) == o.client)
                .unwrap();
            let body = read_source(s, &ctx).unwrap();
            let target = super::core_protocol(&o.target).unwrap();
            let ir = tr.decode_request(s.source, &body, &HeaderMap::new()).unwrap();
            let caps = shipped().resolve(&target.family(), &o.model, None);
            let ectx = tr.encode_ctx(s.source, &ir, ResponseId::new("resp_xlate"), CREATED_AT);
            let mut ir2 = ir.clone();
            ir2.model.resolved = Some(o.model.clone());
            let low = tr
                .lower(ir2, &caps, target, &llm_xlate::requirements::Resolutions::new())
                .unwrap();
            let enc = tr.encode_request(target, &low.req, &caps, &ectx).unwrap();
            assert_eq!(
                enc.body.to_vec(),
                o.upstream_body,
                "{} {}->{}: upstream body diverged from Translator output",
                o.scenario,
                o.client,
                o.target
            );
            checked += 1;
        }
        assert!(checked > 0, "no upstream bodies were produced to compare");
    }

    #[test]
    fn every_outcome_is_a_body_or_a_reason() {
        let opts = TranslateOpts { pair: None, model: None, strip: false };
        let outcomes = dry_run(&real_set(), &opts, &ctx()).unwrap();
        for o in &outcomes {
            let produced = !o.upstream_body.is_empty();
            let explained = o.error.is_some() || o.skipped.is_some();
            assert!(produced || explained, "{} {}->{} produced nothing", o.scenario, o.client, o.target);
        }
    }

    #[test]
    fn pair_filter_restricts_to_one_directed_pair() {
        let opts = TranslateOpts {
            pair: Some(PairFilter::parse("chat->anthropic").unwrap()),
            model: None,
            strip: false,
        };
        let outcomes = dry_run(&real_set(), &opts, &ctx()).unwrap();
        assert!(!outcomes.is_empty());
        assert!(outcomes.iter().all(|o| o.client == "chat" && o.target == "anthropic"));
    }

    #[test]
    fn model_override_is_applied() {
        let opts = TranslateOpts {
            pair: Some(PairFilter::parse("chat->anthropic").unwrap()),
            model: Some("claude-opus-4-6".to_string()),
            strip: false,
        };
        let outcomes = dry_run(&example_set(), &opts, &ctx()).unwrap();
        assert!(outcomes.iter().all(|o| o.model == "claude-opus-4-6"));
    }

    #[test]
    fn model_set_resolves_cheap_first_member() {
        // chat->responses uses openai_responses, whose cheap default is gpt-4o-mini.
        let opts = TranslateOpts {
            pair: Some(PairFilter::parse("chat->responses").unwrap()),
            model: None,
            strip: false,
        };
        let outcomes = dry_run(&example_set(), &opts, &ctx()).unwrap();
        assert!(outcomes.iter().all(|o| o.model == "gpt-4o-mini"), "got {:?}", outcomes.iter().map(|o| &o.model).collect::<Vec<_>>());
    }

    #[test]
    fn probe_sourced_scenarios_materialize() {
        // The real file has probe-sourced scenarios (e.g. ant.example.text); they must produce a
        // body (or an explained skip/error), never crash the run.
        let opts = TranslateOpts { pair: None, model: None, strip: false };
        let outcomes = dry_run(&real_set(), &opts, &ctx()).unwrap();
        // At least some anthropic-sourced outcomes come from probe bodies.
        assert!(outcomes.iter().any(|o| o.client == "anthropic"));
    }

    #[test]
    fn render_is_nonempty_and_labels_pairs() {
        let opts = TranslateOpts { pair: None, model: None, strip: false };
        let outcomes = dry_run(&example_set(), &opts, &ctx()).unwrap();
        let text = render_dry_run(&outcomes);
        assert!(text.contains("chat -> anthropic"));
    }

    #[test]
    fn pair_parse_rejects_garbage() {
        assert!(PairFilter::parse("chatanthropic").is_err());
        assert!(PairFilter::parse("chat->bogus").is_err());
    }

    #[test]
    fn scenarios_path_prefers_real_then_example() {
        // The crate ships a real scenarios.toml, so the default resolves to it.
        let p = scenarios_path(&e2e_dir(), None);
        assert!(p.ends_with("scenarios.toml"));
        // An explicit path wins.
        let ex = e2e_dir().join("translate").join("scenarios.example.toml");
        assert_eq!(scenarios_path(&e2e_dir(), Some(&ex)), ex);
    }

    #[test]
    fn scenario_rejects_both_and_neither_source() {
        assert!(load_scenarios_str("[[scenario]]\nname='x'\nsource='chat'\nrequest='a'\nprobe='b'\n").is_err());
        assert!(load_scenarios_str("[[scenario]]\nname='x'\nsource='chat'\n").is_err());
    }

    fn load_scenarios_str(s: &str) -> Result<ScenarioSet> {
        let mut p = std::env::temp_dir();
        p.push(format!("llm-xlate-e2e-scn-{}.toml", std::process::id()));
        std::fs::write(&p, s).unwrap();
        let r = load_scenarios(&p);
        let _ = std::fs::remove_file(&p);
        r
    }
}
