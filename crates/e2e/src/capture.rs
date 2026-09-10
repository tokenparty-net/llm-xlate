//! On-disk capture format (plan §4): a per-run directory holding one sub-directory per
//! (probe × model × stream) with the exact request, the raw response, parsed SSE events, and the
//! automatic observations, plus a `manifest.json` index.
//!
//! Layout:
//! ```text
//! runs/<yyyymmdd-hhmmss>-<label>/
//!   manifest.json
//!   report.md
//!   <probe id>/<model>[.stream]/
//!     request.json      { method, url, stream, headers (redacted), body }
//!     response.json     { status, headers, elapsed_ms, body }   (non-streaming)
//!     response.sse      raw bytes exactly as received            (streaming)
//!     events.json       parsed SSE events (derived; streaming)
//!     response.meta.json{ status, headers, elapsed_ms }          (streaming)
//!     observe.json      automatic observations (§6)
//!     skipped.json      { reason }                               (skipped)
//! ```
//!
//! [`Capture`] and [`Run`] load this format back so the `check`/`translate`/`promote` engineers
//! can consume a run without re-deriving anything.

use crate::keys::{is_secret_header, looks_like_key};
use crate::probe::Expansion;
use anyhow::{bail, Context, Result};
use llm_xlate_core::sse::SseParser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The exact request that was (or would be) sent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestRecord {
    /// HTTP method (always `POST` for the completion endpoints).
    pub method: String,
    /// Full request URL (never contains key material).
    pub url: String,
    /// Whether the wire body requested streaming.
    pub stream: bool,
    /// Non-secret request headers (secret ones already redacted).
    pub headers: BTreeMap<String, String>,
    /// The JSON body as sent, with large asset payloads rewritten to `$ASSET:<name>`.
    pub body: serde_json::Value,
}

/// A parsed SSE event, mirroring [`llm_xlate_core::sse::SseEvent`] for serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventRecord {
    /// The `event:` field, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    /// The joined `data:` payload.
    pub data: String,
    /// The last-seen `id:` field, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// The raw wire response as produced by the client, before it is written to disk.
#[derive(Debug, Clone)]
pub enum WireResponse {
    /// A non-streaming JSON response.
    NonStream {
        /// HTTP status.
        status: u16,
        /// Response headers (secret ones already redacted).
        headers: BTreeMap<String, String>,
        /// Wall-clock time to receive the whole body.
        elapsed_ms: u128,
        /// Raw response body bytes.
        body_bytes: Vec<u8>,
    },
    /// A streaming (SSE) response.
    Stream {
        /// HTTP status.
        status: u16,
        /// Response headers (secret ones already redacted).
        headers: BTreeMap<String, String>,
        /// Wall-clock time to receive the whole stream.
        elapsed_ms: u128,
        /// Raw SSE bytes exactly as received.
        raw: Vec<u8>,
    },
}

/// The outcome of one expansion, as loaded from disk.
#[derive(Debug, Clone)]
pub enum Outcome {
    /// A non-streaming response.
    NonStream {
        /// HTTP status.
        status: u16,
        /// Redacted response headers.
        headers: BTreeMap<String, String>,
        /// Elapsed milliseconds.
        elapsed_ms: u128,
        /// Parsed JSON body.
        body: serde_json::Value,
    },
    /// A streaming response.
    Stream {
        /// HTTP status.
        status: u16,
        /// Redacted response headers.
        headers: BTreeMap<String, String>,
        /// Elapsed milliseconds.
        elapsed_ms: u128,
        /// Raw SSE bytes.
        raw_sse: Vec<u8>,
        /// Parsed SSE events.
        events: Vec<EventRecord>,
    },
    /// The expansion was skipped (e.g. an unmet dependency).
    Skipped {
        /// Human-readable reason.
        reason: String,
    },
}

/// A fully-loaded capture for one expansion.
#[derive(Debug, Clone)]
pub struct Capture {
    /// The capture directory.
    pub dir: PathBuf,
    /// The request that was sent (or planned).
    pub request: RequestRecord,
    /// The outcome.
    pub outcome: Outcome,
    /// The raw `observe.json` value, if present.
    pub observe: Option<serde_json::Value>,
}

impl Capture {
    /// Load a single capture from its directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let request: RequestRecord = read_json(&dir.join("request.json"))
            .with_context(|| format!("loading request.json in {}", dir.display()))?;
        let observe = maybe_read_json(&dir.join("observe.json"))?;
        let skipped_path = dir.join("skipped.json");
        let outcome = if skipped_path.is_file() {
            let s: SkippedRecord = read_json(&skipped_path)?;
            Outcome::Skipped { reason: s.reason }
        } else if dir.join("response.sse").is_file() {
            let raw_sse = std::fs::read(dir.join("response.sse"))?;
            let meta: StreamMeta = read_json(&dir.join("response.meta.json"))?;
            let events: Vec<EventRecord> = read_json(&dir.join("events.json"))?;
            Outcome::Stream {
                status: meta.status,
                headers: meta.headers,
                elapsed_ms: meta.elapsed_ms,
                raw_sse,
                events,
            }
        } else {
            let r: NonStreamRecord = read_json(&dir.join("response.json"))?;
            Outcome::NonStream {
                status: r.status,
                headers: r.headers,
                elapsed_ms: r.elapsed_ms,
                body: r.body,
            }
        };
        Ok(Capture {
            dir: dir.to_path_buf(),
            request,
            outcome,
            observe,
        })
    }
}

/// A loaded run: its manifest plus a handle to its root directory.
#[derive(Debug, Clone)]
pub struct Run {
    /// The run root directory.
    pub root: PathBuf,
    /// The parsed manifest.
    pub manifest: Manifest,
}

impl Run {
    /// Load a run's `manifest.json`.
    pub fn load(run_dir: &Path) -> Result<Self> {
        let manifest: Manifest = read_json(&run_dir.join("manifest.json"))
            .with_context(|| format!("loading manifest.json in {}", run_dir.display()))?;
        Ok(Run {
            root: run_dir.to_path_buf(),
            manifest,
        })
    }

    /// Load every capture referenced by the manifest, in manifest order.
    pub fn captures(&self) -> Result<Vec<Capture>> {
        iter_run(&self.root)
    }
}

/// Load every non-skipped capture in a run directory, in manifest order.
///
/// Skipped entries (`manual` probes, dependency-failure skips) are written with only a
/// `skipped.json` and no `request.json`, so [`Capture::load`] cannot load them; they carry no
/// response to consume, so they are omitted here rather than erroring the whole iteration.
pub fn iter_run(run_dir: &Path) -> Result<Vec<Capture>> {
    let run = Run::load(run_dir)?;
    let mut out = Vec::new();
    for e in &run.manifest.entries {
        if e.skipped.is_some() {
            continue;
        }
        let dir = run_dir.join(&e.rel_dir);
        out.push(Capture::load(&dir)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Manifest.
// ---------------------------------------------------------------------------------------------

/// Provider connection info recorded in the manifest (never any key material).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderInfo {
    /// The base URL requests were sent to.
    pub base_url: String,
}

/// One manifest row per expansion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    /// Directory relative to the run root.
    pub rel_dir: String,
    /// Probe id.
    pub probe_id: String,
    /// Protocol string.
    pub protocol: String,
    /// Model id.
    pub model: String,
    /// Whether streaming was requested.
    pub stream: bool,
    /// HTTP status (absent when skipped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Skip reason, when skipped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
    /// SHA-256 (hex) of the on-disk request body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_sha256: Option<String>,
    /// SHA-256 (hex) of the raw response body / SSE bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_sha256: Option<String>,
}

/// The run manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    /// `llm-xlate-e2e` version.
    pub tool_version: String,
    /// `llm-xlate` version.
    pub xlate_version: String,
    /// The run id (also the directory stem).
    pub run_id: String,
    /// The run label.
    pub label: String,
    /// ISO-8601 creation time.
    pub created_at: String,
    /// Providers used, keyed by name.
    pub providers: BTreeMap<String, ProviderInfo>,
    /// Model-set names referenced.
    pub model_sets: Vec<String>,
    /// One row per expansion.
    pub entries: Vec<ManifestEntry>,
}

// ---------------------------------------------------------------------------------------------
// Raw on-disk shapes for the response variants.
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct NonStreamRecord {
    status: u16,
    headers: BTreeMap<String, String>,
    elapsed_ms: u128,
    body: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct StreamMeta {
    status: u16,
    headers: BTreeMap<String, String>,
    elapsed_ms: u128,
}

#[derive(Serialize, Deserialize)]
struct SkippedRecord {
    reason: String,
}

// ---------------------------------------------------------------------------------------------
// Writing.
// ---------------------------------------------------------------------------------------------

/// Parse raw SSE bytes into event records (used by the writer and by observers).
pub fn parse_sse(raw: &[u8]) -> Vec<EventRecord> {
    let mut parser = SseParser::new();
    let mut events = parser.push(raw);
    events.extend(parser.finish());
    events
        .into_iter()
        .map(|e| EventRecord {
            event: e.event,
            data: e.data,
            id: e.id,
        })
        .collect()
}

/// SHA-256 hex digest.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Guard: refuse to persist a string containing anything that looks like a key.
fn guard_no_secrets(context: &str, value: &serde_json::Value) -> Result<()> {
    let s = serde_json::to_string(value).unwrap_or_default();
    if looks_like_key(&s) {
        bail!("refusing to write {context}: it contains a string that looks like an API key");
    }
    Ok(())
}

/// Guard: refuse to persist raw bytes containing anything that looks like a key.
fn guard_bytes(context: &str, bytes: &[u8]) -> Result<()> {
    if looks_like_key(&String::from_utf8_lossy(bytes)) {
        bail!("refusing to write {context}: it contains a string that looks like an API key");
    }
    Ok(())
}

/// Ensure header maps carry no secret values (defensive; the client should have redacted them).
fn assert_headers_redacted(headers: &BTreeMap<String, String>) -> Result<()> {
    for (k, v) in headers {
        if is_secret_header(k) && v != "<redacted>" {
            bail!("header `{k}` was not redacted before capture");
        }
        if looks_like_key(v) {
            bail!("header `{k}` value looks like a key");
        }
    }
    Ok(())
}

/// Write one capture (request + response) and return its manifest entry.
///
/// The request body has its asset payloads rewritten to `$ASSET:<name>` before it is written.
/// Both request and response are scanned to refuse persisting anything that looks like a secret.
pub fn write_capture(
    run_root: &Path,
    expansion: &Expansion,
    request: &RequestRecord,
    response: &WireResponse,
    observe: &serde_json::Value,
) -> Result<ManifestEntry> {
    let rel_dir = expansion.rel_dir();
    let dir = run_root.join(&rel_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;

    // Redact assets in the request body before writing.
    let redacted_body = crate::assets::redact_assets(&request.body);
    let disk_request = RequestRecord {
        method: request.method.clone(),
        url: request.url.clone(),
        stream: request.stream,
        headers: request.headers.clone(),
        body: redacted_body,
    };
    assert_headers_redacted(&disk_request.headers)?;
    guard_no_secrets("request.json", &disk_request.body)?;
    let req_json = to_pretty(&disk_request)?;
    std::fs::write(dir.join("request.json"), &req_json)?;
    let request_sha256 = sha256_hex(canonical_bytes(&disk_request.body).as_bytes());

    let (status, response_sha256) = match response {
        WireResponse::NonStream {
            status,
            headers,
            elapsed_ms,
            body_bytes,
        } => {
            assert_headers_redacted(headers)?;
            let body: serde_json::Value = serde_json::from_slice(body_bytes)
                .unwrap_or_else(|_| serde_json::json!({ "_unparsed": String::from_utf8_lossy(body_bytes) }));
            guard_no_secrets("response.json", &body)?;
            let record = NonStreamRecord {
                status: *status,
                headers: headers.clone(),
                elapsed_ms: *elapsed_ms,
                body,
            };
            std::fs::write(dir.join("response.json"), to_pretty(&record)?)?;
            (Some(*status), Some(sha256_hex(body_bytes)))
        }
        WireResponse::Stream {
            status,
            headers,
            elapsed_ms,
            raw,
        } => {
            assert_headers_redacted(headers)?;
            guard_bytes("response.sse", raw)?;
            std::fs::write(dir.join("response.sse"), raw)?;
            let events = parse_sse(raw);
            let events_json = to_pretty(&events)?;
            guard_bytes("events.json", events_json.as_bytes())?;
            std::fs::write(dir.join("events.json"), events_json)?;
            let meta = StreamMeta {
                status: *status,
                headers: headers.clone(),
                elapsed_ms: *elapsed_ms,
            };
            std::fs::write(dir.join("response.meta.json"), to_pretty(&meta)?)?;
            (Some(*status), Some(sha256_hex(raw)))
        }
    };

    std::fs::write(dir.join("observe.json"), to_pretty(observe)?)?;

    Ok(ManifestEntry {
        rel_dir,
        probe_id: expansion.probe.id.clone(),
        protocol: expansion.probe.protocol.as_str().to_string(),
        model: expansion.model.clone(),
        stream: expansion.stream,
        status,
        skipped: None,
        request_sha256: Some(request_sha256),
        response_sha256,
    })
}

/// Write a skipped-expansion marker and return its manifest entry.
pub fn write_skipped(
    run_root: &Path,
    expansion: &Expansion,
    request: Option<&RequestRecord>,
    reason: &str,
) -> Result<ManifestEntry> {
    let rel_dir = expansion.rel_dir();
    let dir = run_root.join(&rel_dir);
    std::fs::create_dir_all(&dir)?;
    let mut request_sha256 = None;
    if let Some(req) = request {
        let redacted_body = crate::assets::redact_assets(&req.body);
        let disk_request = RequestRecord {
            method: req.method.clone(),
            url: req.url.clone(),
            stream: req.stream,
            headers: req.headers.clone(),
            body: redacted_body,
        };
        guard_no_secrets("request.json", &disk_request.body)?;
        std::fs::write(dir.join("request.json"), to_pretty(&disk_request)?)?;
        request_sha256 = Some(sha256_hex(canonical_bytes(&disk_request.body).as_bytes()));
    }
    let rec = SkippedRecord {
        reason: reason.to_string(),
    };
    std::fs::write(dir.join("skipped.json"), to_pretty(&rec)?)?;
    Ok(ManifestEntry {
        rel_dir,
        probe_id: expansion.probe.id.clone(),
        protocol: expansion.probe.protocol.as_str().to_string(),
        model: expansion.model.clone(),
        stream: expansion.stream,
        status: None,
        skipped: Some(reason.to_string()),
        request_sha256,
        response_sha256: None,
    })
}

/// Write the run manifest.
pub fn write_manifest(run_root: &Path, manifest: &Manifest) -> Result<()> {
    std::fs::create_dir_all(run_root)?;
    std::fs::write(run_root.join("manifest.json"), to_pretty(manifest)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// JSON helpers.
// ---------------------------------------------------------------------------------------------

/// Serialize to pretty JSON with a trailing newline (deterministic; `preserve_order`).
pub fn to_pretty<T: Serialize>(v: &T) -> Result<String> {
    let mut s = serde_json::to_string_pretty(v).context("serializing JSON")?;
    s.push('\n');
    Ok(s)
}

/// Compact, key-order-preserving bytes used for hashing a JSON body.
fn canonical_bytes(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&txt).with_context(|| format!("parsing {}", path.display()))
}

fn maybe_read_json(path: &Path) -> Result<Option<serde_json::Value>> {
    if path.is_file() {
        Ok(Some(read_json(path)?))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{Catalogue, Expansion};

    fn expansion(id: &str, proto: &str, model: &str, stream: bool) -> Expansion {
        let toml = format!(
            "[[probe]]\nid=\"{id}\"\nprotocol=\"{proto}\"\nmodels=[\"m\"]\n[probe.body]\nmodel=\"$MODEL\"\n"
        );
        let p = Catalogue::parse_file(&toml).unwrap().remove(0);
        Expansion {
            probe: p,
            model: model.to_string(),
            stream,
        }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("llm-xlate-e2e-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn nonstream_round_trip() {
        let root = tmpdir("nonstream");
        let ex = expansion("ant.text", "anthropic", "claude-opus-4-8", false);
        let req = RequestRecord {
            method: "POST".into(),
            url: "https://api.anthropic.com/v1/messages".into(),
            stream: false,
            headers: BTreeMap::from([("x-api-key".into(), "<redacted>".into())]),
            body: serde_json::json!({"model":"claude-opus-4-8","max_tokens":256}),
        };
        let resp = WireResponse::NonStream {
            status: 200,
            headers: BTreeMap::from([("content-type".into(), "application/json".into())]),
            elapsed_ms: 12,
            body_bytes: br#"{"id":"msg_1","type":"message"}"#.to_vec(),
        };
        let obs = serde_json::json!({"status":200});
        let entry = write_capture(&root, &ex, &req, &resp, &obs).unwrap();
        assert_eq!(entry.status, Some(200));
        assert!(entry.request_sha256.is_some());

        let cap = Capture::load(&root.join(&entry.rel_dir)).unwrap();
        assert_eq!(cap.request.url, req.url);
        match cap.outcome {
            Outcome::NonStream { status, body, .. } => {
                assert_eq!(status, 200);
                assert_eq!(body["id"], serde_json::json!("msg_1"));
            }
            _ => panic!("expected nonstream"),
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn stream_round_trip_and_events() {
        let root = tmpdir("stream");
        let ex = expansion("chat.text", "chat", "gpt-4o-mini", true);
        let req = RequestRecord {
            method: "POST".into(),
            url: "https://api.openai.com/v1/chat/completions".into(),
            stream: true,
            headers: BTreeMap::new(),
            body: serde_json::json!({"model":"gpt-4o-mini","stream":true}),
        };
        let raw = b"data: {\"a\":1}\n\ndata: [DONE]\n\n".to_vec();
        let resp = WireResponse::Stream {
            status: 200,
            headers: BTreeMap::new(),
            elapsed_ms: 5,
            raw: raw.clone(),
        };
        let entry = write_capture(&root, &ex, &req, &resp, &serde_json::json!({})).unwrap();
        assert_eq!(entry.response_sha256, Some(sha256_hex(&raw)));
        let cap = Capture::load(&root.join(&entry.rel_dir)).unwrap();
        match cap.outcome {
            Outcome::Stream { events, raw_sse, .. } => {
                assert_eq!(raw_sse, raw);
                assert_eq!(events.len(), 2);
                assert_eq!(events[0].data, "{\"a\":1}");
                assert_eq!(events[1].data, "[DONE]");
            }
            _ => panic!("expected stream"),
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn skipped_round_trip() {
        let root = tmpdir("skipped");
        let ex = expansion("resp.dep", "responses", "gpt-5.4", false);
        let entry = write_skipped(&root, &ex, None, "dependency failed").unwrap();
        assert_eq!(entry.skipped.as_deref(), Some("dependency failed"));
        // Skipped has no request.json here, so Capture::load requires one; assert the marker file.
        assert!(root.join(&entry.rel_dir).join("skipped.json").is_file());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn refuses_to_persist_keylike_body() {
        let root = tmpdir("secret");
        let ex = expansion("chat.leak", "chat", "gpt-4o-mini", false);
        let req = RequestRecord {
            method: "POST".into(),
            url: "u".into(),
            stream: false,
            headers: BTreeMap::new(),
            body: serde_json::json!({"note":"sk-ant-api03-abcdefghijklmnopqrstuvwx"}),
        };
        let resp = WireResponse::NonStream {
            status: 200,
            headers: BTreeMap::new(),
            elapsed_ms: 1,
            body_bytes: b"{}".to_vec(),
        };
        let err = write_capture(&root, &ex, &req, &resp, &serde_json::json!({})).unwrap_err();
        assert!(err.to_string().contains("looks like an API key"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn run_manifest_round_trip() {
        let root = tmpdir("manifest");
        let ex = expansion("ant.text", "anthropic", "m", false);
        let req = RequestRecord {
            method: "POST".into(),
            url: "u".into(),
            stream: false,
            headers: BTreeMap::new(),
            body: serde_json::json!({"model":"m"}),
        };
        let resp = WireResponse::NonStream {
            status: 200,
            headers: BTreeMap::new(),
            elapsed_ms: 1,
            body_bytes: b"{\"ok\":true}".to_vec(),
        };
        let entry = write_capture(&root, &ex, &req, &resp, &serde_json::json!({"status":200})).unwrap();
        let manifest = Manifest {
            tool_version: "0.1.0".into(),
            xlate_version: "0.1.0".into(),
            run_id: "20260101-000000".into(),
            label: "test".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            providers: BTreeMap::new(),
            model_sets: vec![],
            entries: vec![entry],
        };
        write_manifest(&root, &manifest).unwrap();
        let run = Run::load(&root).unwrap();
        assert_eq!(run.manifest.entries.len(), 1);
        let caps = run.captures().unwrap();
        assert_eq!(caps.len(), 1);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn iter_run_skips_skipped_entries() {
        // A run with one good non-stream capture plus a skipped entry (skipped.json only, no
        // request.json) must iterate cleanly — the skipped one is omitted, not an error.
        let root = tmpdir("iter-skip");
        let good = expansion("ant.text", "anthropic", "m", false);
        let req = RequestRecord {
            method: "POST".into(),
            url: "u".into(),
            stream: false,
            headers: BTreeMap::new(),
            body: serde_json::json!({"model":"m"}),
        };
        let resp = WireResponse::NonStream {
            status: 200,
            headers: BTreeMap::new(),
            elapsed_ms: 1,
            body_bytes: b"{\"ok\":true}".to_vec(),
        };
        let good_entry = write_capture(&root, &good, &req, &resp, &serde_json::json!({})).unwrap();
        // A `manual`-style skip: skipped.json only, request.json absent.
        let skip = expansion("ant.manual", "anthropic", "m", false);
        let skip_entry = write_skipped(&root, &skip, None, "requires manual trigger").unwrap();
        assert!(!root.join(&skip_entry.rel_dir).join("request.json").exists());
        let manifest = Manifest {
            tool_version: "0.1.0".into(),
            xlate_version: "0.1.0".into(),
            run_id: "20260101-000000".into(),
            label: "test".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            providers: BTreeMap::new(),
            model_sets: vec![],
            entries: vec![good_entry, skip_entry],
        };
        write_manifest(&root, &manifest).unwrap();
        let caps = iter_run(&root).unwrap();
        assert_eq!(caps.len(), 1, "skipped entry must be omitted, not loaded");
        assert_eq!(caps[0].request.url, "u");
        std::fs::remove_dir_all(&root).ok();
    }
}
