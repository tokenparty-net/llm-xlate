//! `promote` (plan §9): copy selected captures out of a run and into the golden fixture tree
//! using the codec crate's naming conventions, maintaining a `MANIFEST.json`, and mirroring the
//! full capture into the committed `crates/e2e/dataset/` browsable dataset.
//!
//! Naming: `<scenario>__<model>__<yyyymmdd>.{req.json,resp.json,stream.sse,err.json}` under
//! `<dest>/<protocol>/`. Promotion is **idempotent** — re-promoting the same capture rewrites the
//! files and updates (rather than duplicates) the manifest entry keyed by `(protocol, name)`.
//!
//! `promote` refuses to write any file whose bytes contain something that looks like an API key
//! (defence in depth on top of the capture writer's own guard).

use crate::capture::{sha256_hex, Capture, Outcome, Run};
use crate::keys::looks_like_key;
use crate::probe::sanitize;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The header redactions the capture writer applies (recorded in the manifest for provenance).
const REDACTIONS: &[&str] = &[
    "authorization",
    "x-api-key",
    "openai-organization",
    "openai-project",
    "cookie",
    "anthropic-organization-id",
    "anthropic-workspace-id",
];

/// Options for a promote invocation.
#[derive(Debug, Clone)]
pub struct PromoteOpts {
    /// The run directory to promote from.
    pub run: PathBuf,
    /// A glob over probe ids selecting which captures to promote.
    pub select: String,
    /// An explicit scenario name (else derived from the probe id).
    pub name: Option<String>,
    /// The golden destination directory (`crates/xlate/tests/golden`).
    pub dest: PathBuf,
    /// The dataset mirror directory (`crates/e2e/dataset`).
    pub dataset: PathBuf,
    /// Print the plan without writing.
    pub dry_run: bool,
}

/// One promoted fixture file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileRecord {
    /// The file name relative to `<dest>/<protocol>/`.
    pub file: String,
    /// SHA-256 (hex) of the written bytes.
    pub sha256: String,
}

/// One manifest entry (also the unit of idempotency).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoldenEntry {
    /// The fixture base name `<scenario>__<model>__<date>`.
    pub name: String,
    /// The protocol directory.
    pub protocol: String,
    /// The source probe id.
    pub probe_id: String,
    /// The model id.
    pub model: String,
    /// The promotion date (yyyymmdd).
    pub date: String,
    /// The source run id.
    pub source_run: String,
    /// The files written for this fixture, keyed by kind (`req`/`resp`/`err`/`stream`).
    pub files: BTreeMap<String, FileRecord>,
    /// Header redactions applied when the capture was recorded.
    pub redactions: Vec<String>,
    /// HTTP status, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// The observed verdict against the probe's expectation, when the capture recorded one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<String>,
}

/// The `MANIFEST.json` shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoldenManifest {
    /// One entry per promoted fixture.
    pub entries: Vec<GoldenEntry>,
}

/// The result of a promote run.
#[derive(Debug, Clone)]
pub struct PromoteReport {
    /// The entries promoted (or that would be, in dry-run).
    pub entries: Vec<GoldenEntry>,
    /// The exact `golden_matrix.rs` fixture-list additions needed (discovery is a hard-coded
    /// list, not a glob — see the report's desired-changes note).
    pub matrix_lines: Vec<String>,
}

fn pname_from_url(url: &str) -> Option<&'static str> {
    if url.contains("/v1/messages") {
        Some("anthropic")
    } else if url.contains("/v1/chat/completions") {
        Some("chat")
    } else if url.contains("/v1/responses") {
        Some("responses")
    } else {
        None
    }
}

/// Derive the scenario token from an explicit name or a probe id (`ant.tools.call` → `tools_call`,
/// dropping the protocol prefix segment).
fn scenario_token(name: &Option<String>, probe_id: &str) -> String {
    if let Some(n) = name {
        return sanitize(n).replace('.', "_");
    }
    let without_prefix = probe_id.split_once('.').map(|(_, r)| r).unwrap_or(probe_id);
    sanitize(without_prefix).replace('.', "_")
}

fn date_from_run(run: &Run) -> String {
    let id = &run.manifest.run_id;
    let head = id.split('-').next().unwrap_or("");
    if head.len() == 8 && head.chars().all(|c| c.is_ascii_digit()) {
        head.to_string()
    } else {
        chrono::Local::now().format("%Y%m%d").to_string()
    }
}

fn guard_bytes(context: &str, bytes: &[u8]) -> Result<()> {
    let s = String::from_utf8_lossy(bytes);
    if looks_like_key(&s) {
        bail!("refusing to promote {context}: it contains a string that looks like an API key");
    }
    Ok(())
}

fn write_pretty(path: &Path, value: &serde_json::Value) -> Result<Vec<u8>> {
    let mut s = serde_json::to_string_pretty(value)?;
    s.push('\n');
    guard_bytes(&path.display().to_string(), s.as_bytes())?;
    std::fs::write(path, &s)?;
    Ok(s.into_bytes())
}

/// Run `promote` and return the report.
pub fn promote(opts: &PromoteOpts) -> Result<PromoteReport> {
    let pattern = glob::Pattern::new(&opts.select)
        .with_context(|| format!("invalid --select glob `{}`", opts.select))?;
    let run = Run::load(&opts.run)?;
    let date = date_from_run(&run);

    let mut entries: Vec<GoldenEntry> = Vec::new();
    let mut matrix: Vec<String> = Vec::new();

    for entry in &run.manifest.entries {
        if entry.skipped.is_some() {
            continue;
        }
        if !pattern.matches(&entry.probe_id) {
            continue;
        }
        let cap = Capture::load(&opts.run.join(&entry.rel_dir))
            .with_context(|| format!("loading capture {}", entry.rel_dir))?;
        let protocol = match pname_from_url(&cap.request.url) {
            Some(p) => p.to_string(),
            None => entry.protocol.clone(),
        };
        let scenario = scenario_token(&opts.name, &entry.probe_id);
        let model_tok = sanitize(&entry.model);
        let base = format!("{scenario}__{model_tok}__{date}");

        let proto_dir = opts.dest.join(&protocol);
        if !opts.dry_run {
            std::fs::create_dir_all(&proto_dir)?;
        }

        let mut files: BTreeMap<String, FileRecord> = BTreeMap::new();
        let status: Option<u16>;
        let mut verdict = None;
        if let Some(obs) = &cap.observe {
            verdict = obs.get("verdict").and_then(|v| v.as_str()).map(|s| s.to_string());
        }

        // request → .req.json
        {
            let file = format!("{base}.req.json");
            let bytes = if opts.dry_run {
                let mut s = serde_json::to_string_pretty(&cap.request.body)?;
                s.push('\n');
                guard_bytes(&file, s.as_bytes())?;
                s.into_bytes()
            } else {
                write_pretty(&proto_dir.join(&file), &cap.request.body)?
            };
            files.insert("req".into(), FileRecord { file: file.clone(), sha256: sha256_hex(&bytes) });
            matrix.push(matrix_line(&protocol, "req", &base));
        }

        match &cap.outcome {
            Outcome::NonStream { status: st, headers, body, .. } => {
                status = Some(*st);
                if *st >= 400 {
                    let file = format!("{base}.err.json");
                    let env = serde_json::json!({ "status": st, "headers": headers, "body": body });
                    let bytes = if opts.dry_run {
                        let mut s = serde_json::to_string_pretty(&env)?;
                        s.push('\n');
                        guard_bytes(&file, s.as_bytes())?;
                        s.into_bytes()
                    } else {
                        write_pretty(&proto_dir.join(&file), &env)?
                    };
                    files.insert("err".into(), FileRecord { file, sha256: sha256_hex(&bytes) });
                    matrix.push(matrix_line(&protocol, "err", &base));
                } else {
                    let file = format!("{base}.resp.json");
                    let bytes = if opts.dry_run {
                        let mut s = serde_json::to_string_pretty(body)?;
                        s.push('\n');
                        guard_bytes(&file, s.as_bytes())?;
                        s.into_bytes()
                    } else {
                        write_pretty(&proto_dir.join(&file), body)?
                    };
                    files.insert("resp".into(), FileRecord { file, sha256: sha256_hex(&bytes) });
                    matrix.push(matrix_line(&protocol, "resp", &base));
                }
            }
            Outcome::Stream { status: st, raw_sse, .. } => {
                status = Some(*st);
                let file = format!("{base}.stream.sse");
                guard_bytes(&file, raw_sse)?;
                if !opts.dry_run {
                    std::fs::write(proto_dir.join(&file), raw_sse)?;
                }
                files.insert("stream".into(), FileRecord { file, sha256: sha256_hex(raw_sse) });
                matrix.push(matrix_line(&protocol, "stream", &base));
            }
            Outcome::Skipped { .. } => continue,
        }

        // Mirror the full capture into the dataset.
        if !opts.dry_run {
            let mirror = opts.dataset.join(&protocol).join(&base);
            mirror_capture(&cap.dir, &mirror)?;
        }

        entries.push(GoldenEntry {
            name: base,
            protocol,
            probe_id: entry.probe_id.clone(),
            model: entry.model.clone(),
            date: date.clone(),
            source_run: run.manifest.run_id.clone(),
            files,
            redactions: REDACTIONS.iter().map(|s| s.to_string()).collect(),
            status,
            verdict,
        });
    }

    if entries.is_empty() {
        bail!("no non-skipped captures in {} matched `{}`", opts.run.display(), opts.select);
    }

    // Merge into the destination manifest (idempotent by (protocol, name)).
    if !opts.dry_run {
        let manifest_path = opts.dest.join("MANIFEST.json");
        let mut manifest: GoldenManifest = if manifest_path.is_file() {
            let txt = std::fs::read_to_string(&manifest_path)?;
            // Never silently discard existing provenance: a malformed/foreign manifest halts the
            // promote rather than being clobbered with only the current batch.
            serde_json::from_str(&txt)
                .with_context(|| format!("parsing existing {}", manifest_path.display()))?
        } else {
            GoldenManifest::default()
        };
        for e in &entries {
            if let Some(existing) = manifest
                .entries
                .iter_mut()
                .find(|x| x.protocol == e.protocol && x.name == e.name)
            {
                *existing = e.clone();
            } else {
                manifest.entries.push(e.clone());
            }
        }
        manifest.entries.sort_by_key(|e| (e.protocol.clone(), e.name.clone()));
        let mut json = serde_json::to_string_pretty(&manifest)?;
        json.push('\n');
        std::fs::write(&manifest_path, json)?;
    }

    Ok(PromoteReport { entries, matrix_lines: matrix })
}

fn matrix_line(protocol: &str, kind: &str, base: &str) -> String {
    let (func, list) = match (protocol, kind) {
        ("chat", "req") => ("chat_request_matrix", "run_request list"),
        ("responses", "req") => ("responses_request_matrix", "run_request list"),
        ("anthropic", "req") => ("anthropic_request_matrix", "run_request list"),
        ("chat", "resp") => ("chat_response_matrix", "run_response(.., false) list"),
        ("responses", "resp") => ("responses_response_matrix", "run_response(.., false) list"),
        ("anthropic", "resp") => ("anthropic_response_matrix", "run_response(.., false) list"),
        ("chat", "stream") => ("chat_response_matrix", "run_response(.., true) list"),
        ("responses", "stream") => ("responses_response_matrix", "run_response(.., true) list"),
        ("anthropic", "stream") => ("anthropic_response_matrix", "run_response(.., true) list"),
        ("chat", "err") => ("chat_error_matrix", "run_error list"),
        ("responses", "err") => ("responses_error_matrix", "run_error list"),
        ("anthropic", "err") => ("anthropic_error_matrix", "run_error list"),
        _ => ("?", "?"),
    };
    format!("golden_matrix.rs :: fn {func}() ({list}): add \"{base}\"")
}

/// Copy every file in a capture directory into `dest` (non-recursive; captures are flat).
fn mirror_capture(src: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let name = entry.file_name();
            let bytes = std::fs::read(&path)?;
            // Defensive: never mirror a file that looks like it leaked a key.
            guard_bytes(&format!("dataset/{}", name.to_string_lossy()), &bytes)?;
            std::fs::write(dest.join(name), bytes)?;
        }
    }
    // Also handle the xlate/ subdir when present.
    let xdir = src.join("xlate");
    if xdir.is_dir() {
        let xdest = dest.join("xlate");
        std::fs::create_dir_all(&xdest)?;
        for entry in std::fs::read_dir(&xdir)? {
            let entry = entry?;
            if entry.path().is_file() {
                std::fs::copy(entry.path(), xdest.join(entry.file_name()))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{write_capture, write_manifest, Manifest, RequestRecord, WireResponse};
    use crate::probe::{Catalogue, Expansion};
    use std::collections::BTreeMap as Map;

    fn tmpdir(name: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("llm-xlate-e2e-promote-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn expansion(id: &str, proto: &str, model: &str, stream: bool) -> Expansion {
        let toml = format!(
            "[[probe]]\nid=\"{id}\"\nprotocol=\"{proto}\"\nmodels=[\"m\"]\n[probe.body]\nmodel=\"$MODEL\"\n"
        );
        let p = Catalogue::parse_file(&toml).unwrap().remove(0);
        Expansion { probe: p, model: model.to_string(), stream }
    }

    /// Build a tiny run directory with one Anthropic non-stream capture and a manifest.
    fn build_run(root: &Path) {
        let ex = expansion("ant.tools.call", "anthropic", "claude-opus-5", false);
        let req = RequestRecord {
            method: "POST".into(),
            url: "https://api.anthropic.com/v1/messages".into(),
            stream: false,
            headers: Map::from([("x-api-key".into(), "<redacted>".into())]),
            body: serde_json::json!({"model":"claude-opus-5","max_tokens":256,"messages":[]}),
        };
        let resp = WireResponse::NonStream {
            status: 200,
            headers: Map::from([("content-type".into(), "application/json".into())]),
            elapsed_ms: 5,
            body_bytes: br#"{"id":"msg_1","type":"message","role":"assistant"}"#.to_vec(),
        };
        let entry = write_capture(root, &ex, &req, &resp, &serde_json::json!({"verdict":"pass"})).unwrap();
        let manifest = Manifest {
            tool_version: "0.1.0".into(),
            xlate_version: "0.1.0".into(),
            run_id: "20260910-120000".into(),
            label: "test".into(),
            created_at: "2026-09-10T12:00:00Z".into(),
            providers: Map::new(),
            model_sets: vec![],
            entries: vec![entry],
        };
        write_manifest(root, &manifest).unwrap();
    }

    #[test]
    fn round_trip_writes_expected_names() {
        let run = tmpdir("rt-run");
        build_run(&run);
        let dest = tmpdir("rt-dest");
        let dataset = tmpdir("rt-dataset");
        let opts = PromoteOpts {
            run: run.clone(),
            select: "ant.*".into(),
            name: None,
            dest: dest.clone(),
            dataset: dataset.clone(),
            dry_run: false,
        };
        let report = promote(&opts).unwrap();
        assert_eq!(report.entries.len(), 1);
        let e = &report.entries[0];
        assert_eq!(e.name, "tools_call__claude-opus-5__20260910");
        assert_eq!(e.protocol, "anthropic");
        // The req file exists under dest/anthropic/.
        let req = dest.join("anthropic").join("tools_call__claude-opus-5__20260910.req.json");
        assert!(req.is_file(), "missing {}", req.display());
        // A resp file exists (status 200).
        assert!(e.files.contains_key("resp"));
        // Dataset mirror exists.
        assert!(dataset.join("anthropic").join(&e.name).join("request.json").is_file());
        // MANIFEST written.
        assert!(dest.join("MANIFEST.json").is_file());
        for d in [run, dest, dataset] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn promotion_is_idempotent() {
        let run = tmpdir("idem-run");
        build_run(&run);
        let dest = tmpdir("idem-dest");
        let dataset = tmpdir("idem-dataset");
        let opts = PromoteOpts {
            run: run.clone(),
            select: "ant.*".into(),
            name: None,
            dest: dest.clone(),
            dataset: dataset.clone(),
            dry_run: false,
        };
        promote(&opts).unwrap();
        promote(&opts).unwrap();
        let txt = std::fs::read_to_string(dest.join("MANIFEST.json")).unwrap();
        let manifest: GoldenManifest = serde_json::from_str(&txt).unwrap();
        assert_eq!(manifest.entries.len(), 1, "re-promote duplicated the manifest entry");
        for d in [run, dest, dataset] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn prints_matrix_lines() {
        let run = tmpdir("matrix-run");
        build_run(&run);
        let dest = tmpdir("matrix-dest");
        let dataset = tmpdir("matrix-dataset");
        let opts = PromoteOpts {
            run: run.clone(),
            select: "ant.*".into(),
            name: Some("tools_parallel".into()),
            dest: dest.clone(),
            dataset: dataset.clone(),
            dry_run: true,
        };
        let report = promote(&opts).unwrap();
        assert!(report.matrix_lines.iter().any(|l| l.contains("anthropic_request_matrix")));
        assert!(report.matrix_lines.iter().any(|l| l.contains("anthropic_response_matrix")));
        // Dry-run wrote nothing.
        assert!(!dest.join("anthropic").exists());
        for d in [run, dest, dataset] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn refuses_keylike_content() {
        // Hand-build a run whose request.json body carries a key-shaped string, bypassing the
        // capture writer's own guard, then assert promote refuses.
        let run = tmpdir("leak-run");
        let capdir = run.join("ant.leak").join("claude-opus-5");
        std::fs::create_dir_all(&capdir).unwrap();
        let req = serde_json::json!({
            "method":"POST","url":"https://api.anthropic.com/v1/messages","stream":false,
            "headers":{},"body":{"note":"sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"}
        });
        std::fs::write(capdir.join("request.json"), serde_json::to_string_pretty(&req).unwrap()).unwrap();
        std::fs::write(
            capdir.join("response.json"),
            serde_json::to_string_pretty(&serde_json::json!({"status":200,"headers":{},"elapsed_ms":1,"body":{}})).unwrap(),
        )
        .unwrap();
        let manifest = serde_json::json!({
            "tool_version":"0.1.0","xlate_version":"0.1.0","run_id":"20260910-120000","label":"leak",
            "created_at":"2026-09-10T12:00:00Z","providers":{},"model_sets":[],
            "entries":[{"rel_dir":"ant.leak/claude-opus-5","probe_id":"ant.leak","protocol":"anthropic","model":"claude-opus-5","stream":false,"status":200}]
        });
        std::fs::write(run.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

        let dest = tmpdir("leak-dest");
        let dataset = tmpdir("leak-dataset");
        let opts = PromoteOpts {
            run: run.clone(),
            select: "ant.*".into(),
            name: None,
            dest: dest.clone(),
            dataset: dataset.clone(),
            dry_run: false,
        };
        let err = promote(&opts).unwrap_err();
        assert!(err.to_string().contains("looks like an API key"), "got: {err}");
        for d in [run, dest, dataset] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
