//! `xlate-testrouter` CLI (plan §6): `serve`, `routes`, and the `trace {show|triage|export|replay}`
//! subcommands (left as stubs for the trace-CLI engineer).

#![forbid(unsafe_code)]
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use llm_xlate::Translator;
use llm_xlate_core::codec::TranslatorConfig;
use llm_xlate_core::ir::Protocol;

use llm_xlate_testrouter::config::{Config, Overrides};
use llm_xlate_testrouter::ids::{SystemClock, UuidSource};
use llm_xlate_testrouter::route::Router;
use llm_xlate_testrouter::sidecar::FileSidecar;
use llm_xlate_testrouter::store::FileStore;
use llm_xlate_testrouter::trace::{FileTraceWriter, NullTraceSink, TraceSink};
use llm_xlate_testrouter::upstream::{ProviderKey, ReqwestUpstream};
use llm_xlate_testrouter::{server, App, Deps};

#[derive(Parser)]
#[command(name = "xlate-testrouter", about = "A tracing test router built on llm-xlate")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the router.
    Serve {
        /// Config file path.
        #[arg(long, default_value = "crates/testrouter/router.example.toml")]
        config: String,
        /// Override the listen address.
        #[arg(long)]
        listen: Option<String>,
        /// A session tag copied into the trace (informational; header overrides win per request).
        #[arg(long)]
        tag: Option<String>,
    },
    /// Explain how a model would be routed (no network).
    Routes {
        /// Config file path.
        #[arg(long, default_value = "crates/testrouter/router.example.toml")]
        config: String,
        /// The client model name to explain.
        #[arg(long)]
        model: String,
    },
    /// Trace CLI: show, triage, export and replay recorded traces.
    #[command(subcommand)]
    Trace(TraceCmd),
}

#[derive(Subcommand)]
enum TraceCmd {
    /// Pretty-print a trace (by id, or a `path`/`path#line`).
    Show {
        /// Trace id, a `traces/*.jsonl` path, or `path#line`.
        target: String,
        /// Config used to locate `data_dir/traces` when `target` is a bare id.
        #[arg(long, default_value = "crates/testrouter/router.example.toml")]
        config: String,
        /// Which section to print.
        #[arg(long, default_value = "all")]
        section: String,
        /// Dump the whole record as JSON.
        #[arg(long)]
        raw: bool,
    },
    /// Triage anomalies across traces (exit 1 when anomalies exist).
    Triage {
        /// Config used to locate `data_dir/traces`.
        #[arg(long, default_value = "crates/testrouter/router.example.toml")]
        config: String,
        /// Read a single trace file instead of the whole `traces/` directory.
        #[arg(long)]
        file: Option<String>,
        /// Only records at or after this RFC-3339 timestamp.
        #[arg(long)]
        since: Option<String>,
        /// Only records with this tag.
        #[arg(long)]
        tag: Option<String>,
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Export a trace into an llm-xlate-e2e run directory (both legs).
    Export {
        /// The trace id to export.
        trace_id: String,
        /// Config used to locate `data_dir/traces`.
        #[arg(long, default_value = "crates/testrouter/router.example.toml")]
        config: String,
        /// The run directory to write.
        #[arg(long)]
        to: String,
    },
    /// Re-run a recorded request through the current build and diff (no network, no spend).
    Replay {
        /// The trace id to replay.
        trace_id: String,
        /// Config used to locate `data_dir/traces` and to rebuild the router.
        #[arg(long, default_value = "crates/testrouter/router.example.toml")]
        config: String,
        /// Print the structural + frame-level diff even when it is empty.
        #[arg(long)]
        diff: bool,
        /// Replay against the recorded upstream bytes (plan §6). This is the offline default and
        /// the only supported mode in this build; the flag documents that intent explicitly.
        #[arg(long)]
        upstream_from_trace: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve { config, listen, tag } => serve(config, listen, tag).await,
        Command::Routes { config, model } => routes(config, model),
        Command::Trace(cmd) => trace_cmd(cmd).await,
    }
}

async fn serve(config: String, listen: Option<String>, tag: Option<String>) -> Result<()> {
    let mut cfg = Config::load(&config)?;
    if let Some(l) = listen {
        cfg.server.listen = l;
    }
    let _ = tag; // per-request `x-xlate-tag` is the authoritative tag; this is informational.

    // Warn (do not fail) if provider keys are unresolvable — an operator may lint without secrets.
    if let Ok(root) = llm_xlate_e2e::keys::workspace_root() {
        if let Some(parent) = root.parent() {
            let bad = cfg.unresolvable_providers(parent);
            if !bad.is_empty() {
                tracing::warn!("providers without a resolvable key (will fail at send): {bad:?}");
            }
        }
    }

    let listen = cfg.server.listen.clone();
    let data_dir = cfg.server.data_dir.clone();
    let trace_enabled = cfg.trace.enabled;

    let keys = load_keys(&cfg);
    let upstream = Arc::new(
        ReqwestUpstream::new(&cfg.providers, &keys, Duration::from_secs(600))
            .map_err(|e| anyhow::anyhow!(e.message))?,
    );
    let trace_sink: Arc<dyn TraceSink> = if trace_enabled {
        FileTraceWriter::spawn(&data_dir, &cfg.trace.file)
    } else {
        Arc::new(NullTraceSink)
    };

    let deps = Deps {
        translator: Translator::new(TranslatorConfig::default()),
        id_source: Arc::new(UuidSource),
        clock: Arc::new(SystemClock),
        // Durable under `data_dir/{store,sidecar}` so stored Responses and learned reasoning blobs
        // survive a restart (the in-memory variants remain the test-harness default).
        store: Arc::new(FileStore::new(&data_dir).context("opening the response store")?),
        sidecar: Arc::new(FileSidecar::new(&data_dir).context("opening the reasoning sidecar")?),
        trace_sink,
        upstream,
    };

    let app = App::build(cfg, deps)?;
    server::serve(app, &listen).await
}

/// Load provider keys from the config, resolved relative to the workspace parent (like e2e). Keys
/// are never printed or logged.
fn load_keys(cfg: &Config) -> BTreeMap<String, ProviderKey> {
    let mut out = BTreeMap::new();
    let parent = llm_xlate_e2e::keys::workspace_root().ok().and_then(|r| r.parent().map(|p| p.to_path_buf()));
    for (name, p) in &cfg.providers {
        if let Some(file) = &p.key_file {
            if let Some(parent) = &parent {
                let path = parent.join(file);
                if let Ok(raw) = std::fs::read_to_string(&path) {
                    let t = raw.trim();
                    if !t.is_empty() {
                        out.insert(name.clone(), ProviderKey::new(t));
                        continue;
                    }
                }
            }
        }
        if let Some(env) = &p.key_env {
            if let Ok(v) = std::env::var(env) {
                let t = v.trim();
                if !t.is_empty() {
                    out.insert(name.clone(), ProviderKey::new(t));
                }
            }
        }
    }
    out
}

fn routes(config: String, model: String) -> Result<()> {
    let cfg = Config::load(&config).context("loading config")?;
    let router = Router::new(cfg)?;
    let route = router
        .resolve(Protocol::OaiChat, Some(&model), &Overrides::default())
        .map_err(|e| anyhow::anyhow!(e))?;
    println!("model            : {}", route.client_model);
    println!("provider         : {}", route.provider);
    println!("upstream protocol: {:?}", route.upstream_protocol);
    println!("upstream model   : {}", route.upstream_model);
    println!("caps source      : {}", route.caps_source);
    println!("family           : {}", route.family.label());
    Ok(())
}

async fn trace_cmd(cmd: TraceCmd) -> Result<()> {
    use llm_xlate_testrouter::tracecli;
    match cmd {
        TraceCmd::Show { target, config, section, raw } => {
            let dir = tracecli::traces_dir_from_config(&config)?;
            let rec = tracecli::load_target(&dir, &target)?;
            let section = tracecli::Section::parse(&section)?;
            print!("{}", tracecli::show(&rec, section, raw));
        }
        TraceCmd::Triage { config, file, since, tag, json } => {
            let records = match &file {
                Some(f) => llm_xlate_testrouter::trace::TraceRecord::read_all(f)?,
                None => {
                    let dir = tracecli::traces_dir_from_config(&config)?;
                    tracecli::load_all_records(&dir)?
                }
            };
            let report = tracecli::triage(&records, &tracecli::TriageFilter { since, tag });
            if json {
                println!("{}", report.to_json());
            } else {
                print!("{}", report.to_table());
            }
            if report.has_anomalies() {
                std::process::exit(1);
            }
        }
        TraceCmd::Export { trace_id, config, to } => {
            let dir = tracecli::traces_dir_from_config(&config)?;
            let rec = tracecli::find_by_id(&dir, &trace_id)?;
            let outcome = tracecli::export(&rec, std::path::Path::new(&to))?;
            println!(
                "exported to {}\n  client leg:   {}\n  upstream leg: {}",
                outcome.root.display(),
                outcome.client_rel,
                outcome.upstream_rel
            );
        }
        TraceCmd::Replay { trace_id, config, diff, upstream_from_trace } => {
            // `--upstream-from-trace` is the offline default (replay always uses the recorded
            // upstream bytes); accept the flag for CLI-surface parity with plan §6.
            let _ = upstream_from_trace;
            let dir = tracecli::traces_dir_from_config(&config)?;
            let rec = tracecli::find_by_id(&dir, &trace_id)?;
            let cfg = Config::load(&config)?;
            let outcome = tracecli::replay(&rec, cfg).await?;
            if diff || !outcome.matched() {
                print!("{}", outcome.render());
            } else {
                println!("replay of {trace_id}: NO DIFF ✓");
            }
            if !outcome.matched() {
                std::process::exit(1);
            }
        }
    }
    Ok(())
}
