//! `llm-xlate-e2e` — operator-driven live-API exploration and capture utility for `llm-xlate`.
//!
//! This binary is deliberately **not** part of `cargo test --workspace`: it talks to the real
//! Anthropic / OpenAI endpoints, needs keys, and costs money. Everything it knows is data (probes
//! and model sets are TOML; captures are JSON + raw SSE); the code is only the runner, redactor,
//! observer and reporter. See `plan.md` (§10) for the CLI contract.
//!
//! Offline guard rails: `--dry-run` prints without sending, `--max-requests` bounds a run, tokens
//! are clamped, secret headers are redacted, and the tool refuses to persist anything that looks
//! like a key. The only live call made during offline development is `models --refresh` (a free
//! `GET /v1/models`).

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::PathBuf;

use llm_xlate_e2e::{assets, capture, keys, models, promote, report, translate, xlate_check};

use assets::Assets;
use capture::{Manifest, ManifestEntry, ProviderInfo, WireResponse};
use llm_xlate_e2e::client::{Client, ClientConfig};
use llm_xlate_e2e::observe::{observe_nonstream, observe_stream};
use llm_xlate_e2e::probe::{
    clamp_tokens, substitute, AssetSource, Catalogue, ExpandOpts, Expansion, Protocol, RefSource,
    StreamMode, SubstCtx,
};
use models::ModelSets;

/// The tool version (also the manifest's `tool_version`).
const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "llm-xlate-e2e",
    version,
    about = "Live-API exploration and capture utility for llm-xlate (operator-driven)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show probes, expansions, dependency order and an estimated request count.
    List(ListArgs),
    /// Run probes against the live APIs (or print with --dry-run).
    Run(RunArgs),
    /// Regenerate report.md for a completed run.
    Report(ReportArgs),
    /// Diff two runs: same request, different outcome? (API drift).
    Diff(DiffArgs),
    /// Print bundled asset names, sizes and sha256.
    Assets,
    /// Fetch the live /v1/models listings into models.listing.json (the only free live call).
    Models(ModelsArgs),
    /// Push captures (or golden fixtures) through llm-xlate and report per-item pass/fail (§7).
    Check(CheckArgs),
    /// Translate a source request to another protocol and (unless --dry-run) send it (§8).
    Translate(TranslateArgs),
    /// Promote selected captures into the golden fixtures + dataset (§9).
    Promote(PromoteArgs),
}

#[derive(Args)]
struct ListArgs {
    /// Only probes carrying this tag (repeatable).
    #[arg(long)]
    tag: Vec<String>,
    /// Only this protocol (anthropic|chat|responses).
    #[arg(long)]
    protocol: Option<String>,
    /// Restrict to the cheapest model of each set (the default; explicit for symmetry).
    #[arg(long)]
    cheap: bool,
    /// Expand every model in each set (overrides --cheap).
    #[arg(long)]
    all_models: bool,
}

#[derive(Args)]
struct RunArgs {
    /// Only probes carrying this tag (repeatable).
    #[arg(long)]
    tag: Vec<String>,
    /// Only these probe ids (repeatable).
    #[arg(long)]
    probe: Vec<String>,
    /// Override every probe's model set with this set name.
    #[arg(long)]
    models: Option<String>,
    /// Restrict to the cheapest model of each set (the default; explicit for symmetry).
    #[arg(long)]
    cheap: bool,
    /// Expand every model in each set (overrides --cheap).
    #[arg(long)]
    all_models: bool,
    /// Force a stream mode for every probe: both|on|off.
    #[arg(long)]
    stream: Option<String>,
    /// Global maximum concurrent requests.
    #[arg(long, default_value_t = 2)]
    concurrency: usize,
    /// Abort before exceeding this many live requests.
    #[arg(long, default_value_t = 150)]
    max_requests: usize,
    /// Print every request without sending; write nothing.
    #[arg(long)]
    dry_run: bool,
    /// Run label (part of the run directory name).
    #[arg(long, default_value = "run")]
    label: String,
    /// Do not retry 5xx/429; capture the error instead.
    #[arg(long)]
    no_retry: bool,
}

#[derive(Args)]
struct ReportArgs {
    /// Path to the run directory.
    run: PathBuf,
}

#[derive(Args)]
struct DiffArgs {
    /// Run A directory.
    run_a: PathBuf,
    /// Run B directory.
    run_b: PathBuf,
}

#[derive(Args)]
struct ModelsArgs {
    /// Fetch the live listings (otherwise just prints the curated sets).
    #[arg(long)]
    refresh: bool,
}

#[derive(Args)]
struct CheckArgs {
    /// A run directory or a golden directory (e.g. crates/xlate/tests/golden).
    input: PathBuf,
    /// Which client protocol(s) to cross-render: all|chat|responses|anthropic.
    #[arg(long, default_value = "all")]
    client: String,
    /// Write xlate.json (and cross-client renderings) under this directory instead of in place.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct TranslateArgs {
    /// Scenario file (defaults to translate/scenarios.toml, then scenarios.example.toml).
    #[arg(long)]
    scenarios: Option<PathBuf>,
    /// Restrict to one directed pair, e.g. chat->anthropic.
    #[arg(long)]
    pair: Option<String>,
    /// Override the upstream model for every pair.
    #[arg(long)]
    model: Option<String>,
    /// Continue past unresolved reasoning with StripAndDegrade.
    #[arg(long)]
    strip: bool,
    /// Print upstream bodies without sending (default off; sending obeys run guard rails).
    #[arg(long)]
    dry_run: bool,
    /// Abort before exceeding this many live requests.
    #[arg(long, default_value_t = 150)]
    max_requests: usize,
    /// Run label (part of the run directory name for live sends).
    #[arg(long, default_value = "translate")]
    label: String,
}

#[derive(Args)]
struct PromoteArgs {
    /// The run directory to promote from.
    #[arg(long)]
    run: PathBuf,
    /// A glob over probe ids selecting which captures to promote.
    #[arg(long)]
    select: String,
    /// An explicit scenario name (else derived from the probe id).
    #[arg(long)]
    name: Option<String>,
    /// The golden destination directory (defaults to crates/xlate/tests/golden).
    #[arg(long)]
    dest: Option<PathBuf>,
    /// Print the plan without writing.
    #[arg(long)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::List(a) => cmd_list(a),
        Command::Run(a) => cmd_run(a).await,
        Command::Report(a) => cmd_report(a),
        Command::Diff(a) => cmd_diff(a),
        Command::Assets => cmd_assets(),
        Command::Models(a) => cmd_models(a).await,
        Command::Check(a) => cmd_check(a),
        Command::Translate(a) => cmd_translate(a).await,
        Command::Promote(a) => cmd_promote(a),
    }
}

// ---------------------------------------------------------------------------------------------
// Paths.
// ---------------------------------------------------------------------------------------------

fn e2e_dir() -> Result<PathBuf> {
    Ok(keys::workspace_root()?.join("crates").join("e2e"))
}

fn load_catalogue_and_models() -> Result<(Catalogue, ModelSets)> {
    let e2e = e2e_dir()?;
    let models = ModelSets::load(&e2e.join("models.toml"))
        .with_context(|| format!("loading {}", e2e.join("models.toml").display()))?;
    let cat = Catalogue::load(&e2e.join("probes"), &models)?;
    Ok((cat, models))
}

fn protocol_arg(s: &Option<String>) -> Result<Option<Protocol>> {
    match s {
        None => Ok(None),
        Some(p) => Ok(Some(Protocol::parse(p)?)),
    }
}

fn stream_arg(s: &Option<String>) -> Result<Option<StreamMode>> {
    match s.as_deref() {
        None => Ok(None),
        Some("both") => Ok(Some(StreamMode::Both)),
        Some("on") => Ok(Some(StreamMode::On)),
        Some("off") => Ok(Some(StreamMode::Off)),
        Some(other) => bail!("--stream must be both|on|off (got {other:?})"),
    }
}

// ---------------------------------------------------------------------------------------------
// list.
// ---------------------------------------------------------------------------------------------

fn cmd_list(a: ListArgs) -> Result<()> {
    let (cat, models) = load_catalogue_and_models()?;
    let opts = ExpandOpts {
        tags: a.tag,
        probe_ids: vec![],
        protocol: protocol_arg(&a.protocol)?,
        models_override: None,
        cheap: {
            let _ = a.cheap; // cheap is the default; --all-models is what disables it
            !a.all_models
        },
        stream_override: None,
    };
    let expansions = cat.expand(&models, &opts)?;

    println!("Dependency order ({} probes):", cat.probes.len());
    for p in &cat.probes {
        let deps = p.dependencies();
        if deps.is_empty() {
            println!("  {}", p.id);
        } else {
            println!("  {} <- {}", p.id, deps.iter().cloned().collect::<Vec<_>>().join(", "));
        }
    }
    println!();
    println!("Expansions ({}):", expansions.len());
    for e in &expansions {
        let stream = if e.stream { " [stream]" } else { "" };
        let tags = if e.probe.tags.is_empty() {
            String::new()
        } else {
            format!("  tags={}", e.probe.tags.join(","))
        };
        println!("  {}  {}{}{}", e.probe.id, e.model, stream, tags);
    }
    println!();
    let manual = expansions.iter().filter(|e| e.probe.requires.contains(&"manual".to_string())).count();
    println!(
        "Estimated live requests: {} ({} tagged `manual` would be skipped)",
        expansions.len(),
        manual
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Ref sources.
// ---------------------------------------------------------------------------------------------

/// A dry-run ref source: leaves `${ref:id:ptr}` visible so the printed body shows the dependency.
struct DryRefs;
impl RefSource for DryRefs {
    fn resolve(&self, probe_id: &str, pointer: &str) -> Result<serde_json::Value> {
        Ok(serde_json::Value::String(format!("${{ref:{probe_id}:{pointer}}}")))
    }
}

/// A live ref source over the already-captured non-stream responses of the current model.
struct LiveRefs<'a> {
    model: &'a str,
    results: &'a BTreeMap<(String, String), serde_json::Value>,
}
impl RefSource for LiveRefs<'_> {
    fn resolve(&self, probe_id: &str, pointer: &str) -> Result<serde_json::Value> {
        let body = self
            .results
            .get(&(self.model.to_string(), probe_id.to_string()))
            .ok_or_else(|| anyhow!("no captured result for `{probe_id}` on model `{}`", self.model))?;
        let ptr = if pointer.is_empty() { "" } else { pointer };
        body.pointer(ptr)
            .cloned()
            .ok_or_else(|| anyhow!("pointer `{pointer}` not found in result for `{probe_id}`"))
    }
}

// ---------------------------------------------------------------------------------------------
// run.
// ---------------------------------------------------------------------------------------------

async fn cmd_run(a: RunArgs) -> Result<()> {
    let (cat, models) = load_catalogue_and_models()?;
    let opts = ExpandOpts {
        tags: a.tag.clone(),
        probe_ids: a.probe.clone(),
        protocol: None,
        models_override: a.models.clone(),
        cheap: {
            let _ = a.cheap; // cheap is the default; --all-models is what disables it
            !a.all_models
        },
        stream_override: stream_arg(&a.stream)?,
    };
    let expansions = cat.expand(&models, &opts)?;

    // A `manual`-tagged probe cannot be triggered reliably; it is skipped rather than sent.
    let sendable = expansions
        .iter()
        .filter(|e| !e.probe.requires.contains(&"manual".to_string()))
        .count();

    let run_id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let cfg = ClientConfig {
        no_retry: a.no_retry,
        ..Default::default()
    };
    let client = Client::new(cfg.clone())?;
    let assets = Assets;

    // A dry run sends nothing, so the --max-requests spend guard must not gate it: it prints the
    // whole selection (plan §11 E1 acceptance: `run --dry-run --tag explore` prints all bodies).
    if a.dry_run {
        println!("DRY RUN — {} expansions, nothing will be sent.", expansions.len());
        println!("concurrency={} (execution is sequential in this build)", a.concurrency);
        for e in &expansions {
            print_dry(&client, e, &run_id, &assets)?;
        }
        return Ok(());
    }

    if sendable > a.max_requests {
        bail!(
            "run would issue {sendable} requests, exceeding --max-requests {}; narrow the selection or raise the cap",
            a.max_requests
        );
    }

    // Live run.
    let e2e = e2e_dir()?;
    let run_dir = e2e.join("runs").join(format!("{run_id}-{}", a.label));
    std::fs::create_dir_all(&run_dir)?;
    println!(
        "LIVE RUN {run_id} — {sendable} requests, concurrency {} (sequential), writing to {}",
        a.concurrency,
        run_dir.display()
    );

    let mut results: BTreeMap<(String, String), serde_json::Value> = BTreeMap::new();
    let mut entries: Vec<ManifestEntry> = Vec::new();
    let mut model_sets: Vec<String> = Vec::new();

    for e in &expansions {
        if let models::ModelsSpec::Set(name) = &e.probe.models {
            if !model_sets.contains(name) {
                model_sets.push(name.clone());
            }
        }
        if e.probe.requires.contains(&"manual".to_string()) {
            let entry = capture::write_skipped(&run_dir, e, None, "requires manual trigger")?;
            entries.push(entry);
            continue;
        }
        // Substitute; a missing ref means an unmet dependency → skip.
        let subst = {
            let refs = LiveRefs {
                model: &e.model,
                results: &results,
            };
            let ctx = SubstCtx {
                model: &e.model,
                run_id: &run_id,
                assets: &assets,
                refs: &refs,
            };
            substitute(&e.probe.body, &ctx)
        };
        let body = match subst {
            Ok(b) => clamp_tokens(b, e.probe.allow_long, e.probe.protocol),
            Err(err) => {
                let entry = capture::write_skipped(&run_dir, e, None, &format!("dependency failed: {err}"))?;
                entries.push(entry);
                continue;
            }
        };
        guard_body(e, &body)?;
        let built = client.build(e, body)?;
        let request = built.request_record();
        let response = client.send(&built).await?;
        let obs = observe_response(e, &request.body, &response);
        // Record the non-stream body for later refs (stream=false preferred).
        if !e.stream {
            if let WireResponse::NonStream { body_bytes, .. } = &response {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(body_bytes) {
                    results.insert((e.model.clone(), e.probe.id.clone()), v);
                }
            }
        }
        let entry = capture::write_capture(&run_dir, e, &request, &response, &obs)?;
        entries.push(entry);
    }

    let mut providers = BTreeMap::new();
    if expansions.iter().any(|e| e.probe.protocol == Protocol::Anthropic) {
        providers.insert("anthropic".to_string(), ProviderInfo { base_url: cfg.anthropic_base.clone() });
    }
    if expansions.iter().any(|e| matches!(e.probe.protocol, Protocol::Chat | Protocol::Responses)) {
        providers.insert("openai".to_string(), ProviderInfo { base_url: cfg.openai_base.clone() });
    }
    let manifest = Manifest {
        tool_version: TOOL_VERSION.to_string(),
        xlate_version: TOOL_VERSION.to_string(),
        run_id: run_id.clone(),
        label: a.label.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        providers,
        model_sets,
        entries,
    };
    capture::write_manifest(&run_dir, &manifest)?;
    let run = capture::Run::load(&run_dir)?;
    let md = report::render_report(&run, &hypotheses(&cat))?;
    std::fs::write(run_dir.join("report.md"), md)?;
    println!("Wrote {}", run_dir.join("report.md").display());

    // Spend accounting: this run's estimate, then the running per-provider total across all runs.
    let this = report::tally_run(&run);
    for (p, (usd, n)) in &this.by_provider {
        println!("  spend this run: {p} ${usd:.4} ({n} billed captures)");
    }
    let grand = report::update_spend_ledger(&e2e.join("runs"))?;
    for (p, usd) in &grand {
        let flag = if *usd >= 8.0 { "  *** OVER $8 CAP ***" } else { "" };
        println!("  RUNNING TOTAL: {p} ${usd:.4} / $8.00 cap{flag}");
    }
    Ok(())
}

/// Build and print one request without sending.
fn print_dry(client: &Client, e: &Expansion, run_id: &str, assets: &dyn AssetSource) -> Result<()> {
    let refs = DryRefs;
    let ctx = SubstCtx {
        model: &e.model,
        run_id,
        assets,
        refs: &refs,
    };
    let body = clamp_tokens(substitute(&e.probe.body, &ctx)?, e.probe.allow_long, e.probe.protocol);
    guard_body(e, &body)?;
    let built = client.build(e, body)?;
    println!("---");
    println!("{}  model={}  stream={}", e.probe.id, e.model, e.stream);
    println!("POST {}", built.url);
    for (k, v) in &built.headers {
        println!("  {k}: {v}");
    }
    // The on-disk request rewrites asset payloads to placeholders; show that view.
    let disk_body = assets::redact_assets(&built.wire_body);
    println!("{}", serde_json::to_string_pretty(&disk_body)?);
    Ok(())
}

/// Refuse to send/print a body containing anything that looks like a key.
fn guard_body(e: &Expansion, body: &serde_json::Value) -> Result<()> {
    let s = serde_json::to_string(body).unwrap_or_default();
    if keys::looks_like_key(&s) {
        bail!("probe `{}` body contains a string that looks like an API key", e.probe.id);
    }
    Ok(())
}

fn observe_response(e: &Expansion, request_body: &serde_json::Value, resp: &WireResponse) -> serde_json::Value {
    let obs = match resp {
        WireResponse::NonStream {
            status,
            headers,
            body_bytes,
            ..
        } => {
            let body: serde_json::Value = serde_json::from_slice(body_bytes)
                .unwrap_or_else(|_| serde_json::json!({"_unparsed": String::from_utf8_lossy(body_bytes)}));
            observe_nonstream(e.probe.protocol, request_body, *status, headers, &body, &e.probe.expect)
        }
        WireResponse::Stream {
            status,
            headers,
            raw,
            ..
        } => {
            let events = capture::parse_sse(raw);
            observe_stream(e.probe.protocol, request_body, *status, headers, raw, &events, &e.probe.expect)
        }
    };
    serde_json::to_value(obs).unwrap_or(serde_json::Value::Null)
}

fn hypotheses(cat: &Catalogue) -> report::Hypotheses {
    let mut h = report::Hypotheses::new();
    for p in &cat.probes {
        if let Some(hyp) = &p.hypothesis {
            h.insert(p.id.clone(), hyp.clone());
        }
    }
    h
}

// ---------------------------------------------------------------------------------------------
// report / diff.
// ---------------------------------------------------------------------------------------------

fn cmd_report(a: ReportArgs) -> Result<()> {
    let run = capture::Run::load(&a.run)?;
    let (cat, _models) = load_catalogue_and_models().unwrap_or_else(|_| {
        // Reports must render even without the probe catalogue; hypotheses just go blank.
        (Catalogue { probes: vec![] }, ModelSets::default())
    });
    let md = report::render_report(&run, &hypotheses(&cat))?;
    let out = a.run.join("report.md");
    std::fs::write(&out, &md)?;
    println!("Wrote {}", out.display());
    // Keep the running spend ledger current when a report is regenerated in place.
    if let Some(runs_dir) = a.run.parent() {
        if runs_dir.file_name().map(|n| n == "runs").unwrap_or(false) {
            let _ = report::update_spend_ledger(runs_dir);
        }
    }
    Ok(())
}

fn cmd_diff(a: DiffArgs) -> Result<()> {
    let ra = capture::Run::load(&a.run_a)?;
    let rb = capture::Run::load(&a.run_b)?;
    let md = report::diff_runs(&ra, &rb)?;
    println!("{md}");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// assets.
// ---------------------------------------------------------------------------------------------

fn cmd_assets() -> Result<()> {
    for (name, bytes, mime) in assets::ALL {
        let sha = capture::sha256_hex(bytes);
        println!("{name:<12} {:>6} bytes  {mime:<16} sha256={sha}", bytes.len());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// models --refresh.
// ---------------------------------------------------------------------------------------------

async fn cmd_models(a: ModelsArgs) -> Result<()> {
    let e2e = e2e_dir()?;
    let sets = ModelSets::load(&e2e.join("models.toml"))?;
    if !a.refresh {
        println!("models.toml (refreshed_at {}):", sets.refreshed_at);
        for (name, set) in &sets.sets {
            let flag = if set.unverified { " [unverified]" } else { "" };
            println!("  {name}{flag}: {}", set.models.join(", "));
        }
        return Ok(());
    }

    let cfg = ClientConfig::default();
    let http = reqwest::Client::builder()
        .timeout(cfg.timeout)
        .build()
        .context("building http client")?;
    let (listing, errors) = models::refresh_listings(&http, &cfg.anthropic_base, &cfg.openai_base).await;
    let path = e2e.join("models.listing.json");
    let mut json = capture::to_pretty(&listing)?;
    if !json.ends_with('\n') {
        json.push('\n');
    }
    std::fs::write(&path, json)?;
    println!(
        "Wrote {} — anthropic {} models, openai {} models (refreshed_at {})",
        path.display(),
        listing.anthropic.len(),
        listing.openai.len(),
        listing.refreshed_at
    );
    for e in &errors {
        eprintln!("warning: {e}");
    }
    println!("Now hand-curate crates/e2e/models.toml from the listing (ids only; no keys).");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// check (§7).
// ---------------------------------------------------------------------------------------------

fn cmd_check(a: CheckArgs) -> Result<()> {
    let client = xlate_check::ClientFilter::parse(&a.client)?;
    let summary = xlate_check::check_dir(&a.input, client, a.out.as_deref())?;
    print!("{}", summary.table());
    let failures = summary.failures();
    if !failures.is_empty() {
        eprintln!("\n{} failing item(s):", failures.len());
        for f in &failures {
            eprintln!("  {f}");
        }
        // A failure here is a bug in llm-xlate or an API change — surface it via the exit code.
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// translate (§8).
// ---------------------------------------------------------------------------------------------

async fn cmd_translate(a: TranslateArgs) -> Result<()> {
    let e2e = e2e_dir()?;
    let path = translate::scenarios_path(&e2e, a.scenarios.as_deref());
    let set = translate::load_scenarios(&path)?;
    let ctx = translate::TranslateCtx::load(&e2e)?;
    let opts = translate::TranslateOpts {
        pair: a.pair.as_deref().map(translate::PairFilter::parse).transpose()?,
        model: a.model.clone(),
        strip: a.strip,
    };
    if a.dry_run {
        let outcomes = translate::dry_run(&set, &opts, &ctx)?;
        print!("{}", translate::render_dry_run(&outcomes));
        return Ok(());
    }
    let run_id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let run_dir = e2e.join("runs").join(format!("{run_id}-{}", a.label));
    std::fs::create_dir_all(&run_dir)?;
    println!("LIVE TRANSLATE {run_id} — writing to {}", run_dir.display());
    let report = translate::live_run(&set, &opts, &ctx, &run_dir, a.max_requests).await?;
    print!("{report}");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// promote (§9).
// ---------------------------------------------------------------------------------------------

fn cmd_promote(a: PromoteArgs) -> Result<()> {
    let dest = match a.dest {
        Some(d) => d,
        None => keys::workspace_root()?
            .join("crates")
            .join("xlate")
            .join("tests")
            .join("golden"),
    };
    let dataset = e2e_dir()?.join("dataset");
    let opts = promote::PromoteOpts {
        run: a.run,
        select: a.select,
        name: a.name,
        dest,
        dataset,
        dry_run: a.dry_run,
    };
    let report = promote::promote(&opts)?;
    println!("Promoted {} fixture(s):", report.entries.len());
    for e in &report.entries {
        println!("  {}/{}  <- {} ({})", e.protocol, e.name, e.probe_id, e.model);
    }
    println!("\nThe golden-matrix runner discovers fixtures from hard-coded name lists (not a glob),");
    println!("so add these fixture names to crates/xlate/tests/golden_matrix.rs:");
    for l in &report.matrix_lines {
        println!("  {l}");
    }
    Ok(())
}
