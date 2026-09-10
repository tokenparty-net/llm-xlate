//! Per-protocol HTTP client: URL and header construction, the wire stream flag, verbatim raw-byte
//! capture of the response, header redaction, retry/backoff, and per-provider request spacing.
//!
//! The client never follows a redirect to a different host, reads the whole response body to
//! completion and keeps the bytes verbatim (so the SSE decoder fixture is exact), and separates the
//! *redacted* header view (recorded on disk via [`Built`]) from the real secret headers (injected
//! only at [`Client::send`] time).

use crate::capture::{RequestRecord, WireResponse};
use crate::keys::{is_secret_header, load_key, ApiKey, Provider};
use crate::probe::{Expansion, Protocol};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::Instant;

/// The default Anthropic API version header.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Client configuration (base URLs, spacing, retry policy).
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Anthropic base URL.
    pub anthropic_base: String,
    /// OpenAI base URL.
    pub openai_base: String,
    /// The `anthropic-version` header value.
    pub anthropic_version: String,
    /// Disable retries (capture the error instead).
    pub no_retry: bool,
    /// Minimum spacing between requests to the same provider.
    pub min_spacing: Duration,
    /// Overall request timeout.
    pub timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            anthropic_base: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".into()),
            openai_base: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com".into()),
            anthropic_version: ANTHROPIC_VERSION.to_string(),
            no_retry: false,
            min_spacing: Duration::from_millis(250),
            timeout: Duration::from_secs(120),
        }
    }
}

/// A fully-built request (no secrets): URL, wire body, and the redacted header view recorded on
/// disk. The probe's extra headers are kept separately so [`Client::send`] can replay them.
#[derive(Debug, Clone)]
pub struct Built {
    /// Which provider this goes to.
    pub provider: Provider,
    /// The wire protocol.
    pub protocol: Protocol,
    /// Full request URL.
    pub url: String,
    /// Whether the request streams.
    pub stream: bool,
    /// The wire body as it will be sent (with the stream flag set).
    pub wire_body: serde_json::Value,
    /// The redacted header view for the capture (secret headers show `<redacted>`).
    pub headers: BTreeMap<String, String>,
    /// The probe's extra (non-secret) headers, replayed verbatim at send time.
    pub extra_headers: BTreeMap<String, String>,
}

impl Built {
    /// The on-disk request record (redacted headers, wire body).
    pub fn request_record(&self) -> RequestRecord {
        RequestRecord {
            method: "POST".into(),
            url: self.url.clone(),
            stream: self.stream,
            headers: self.headers.clone(),
            body: self.wire_body.clone(),
        }
    }
}

/// The async HTTP client.
pub struct Client {
    http: reqwest::Client,
    cfg: ClientConfig,
    keys: Mutex<HashMap<&'static str, ApiKey>>,
    last_send: Arc<Mutex<HashMap<&'static str, Instant>>>,
}

impl Client {
    /// Build a client. The HTTP layer refuses cross-host redirects.
    pub fn new(cfg: ClientConfig) -> Result<Self> {
        let redirect = reqwest::redirect::Policy::custom(|attempt| {
            let same_host = attempt.previous().last().and_then(|u| u.host_str())
                == attempt.url().host_str();
            if same_host && attempt.previous().len() < 5 {
                attempt.follow()
            } else {
                attempt.stop()
            }
        });
        let http = reqwest::Client::builder()
            .redirect(redirect)
            .timeout(cfg.timeout)
            .build()
            .context("building reqwest client")?;
        Ok(Client {
            http,
            cfg,
            keys: Mutex::new(HashMap::new()),
            last_send: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn provider_of(protocol: Protocol) -> Provider {
        match protocol {
            Protocol::Anthropic => Provider::Anthropic,
            Protocol::Chat | Protocol::Responses => Provider::OpenAI,
        }
    }

    fn provider_tag(p: Provider) -> &'static str {
        match p {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
        }
    }

    /// Build a request from an expansion and its already-substituted body. No key is needed here;
    /// the redacted header view is safe to record and print.
    pub fn build(&self, expansion: &Expansion, mut body: serde_json::Value) -> Result<Built> {
        let protocol = expansion.probe.protocol;
        let provider = Self::provider_of(protocol);
        // Set the wire stream flag.
        if let Some(obj) = body.as_object_mut() {
            if expansion.stream {
                obj.insert("stream".into(), serde_json::json!(true));
            } else {
                obj.remove("stream");
            }
        }
        let url = match protocol {
            Protocol::Anthropic => format!("{}/v1/messages", self.cfg.anthropic_base.trim_end_matches('/')),
            Protocol::Chat => format!("{}/v1/chat/completions", self.cfg.openai_base.trim_end_matches('/')),
            Protocol::Responses => format!("{}/v1/responses", self.cfg.openai_base.trim_end_matches('/')),
        };

        let mut headers = BTreeMap::new();
        headers.insert("content-type".into(), "application/json".into());
        match provider {
            Provider::Anthropic => {
                headers.insert("x-api-key".into(), "<redacted>".into());
                headers.insert("anthropic-version".into(), self.cfg.anthropic_version.clone());
            }
            Provider::OpenAI => {
                headers.insert("authorization".into(), "<redacted>".into());
            }
        }
        let mut extra_headers = BTreeMap::new();
        for (k, v) in &expansion.probe.headers {
            // A probe must never supply a secret header; reject if it tries.
            if is_secret_header(k) {
                bail!("probe `{}` sets secret header `{k}`", expansion.probe.id);
            }
            extra_headers.insert(k.clone(), v.clone());
            headers.insert(k.clone(), v.clone());
        }

        Ok(Built {
            provider,
            protocol,
            url,
            stream: expansion.stream,
            wire_body: body,
            headers,
            extra_headers,
        })
    }

    async fn key_for(&self, provider: Provider) -> Result<ApiKey> {
        let tag = Self::provider_tag(provider);
        let mut guard = self.keys.lock().await;
        if let Some(k) = guard.get(tag) {
            return Ok(k.clone());
        }
        let k = load_key(provider)?;
        guard.insert(tag, k.clone());
        Ok(k)
    }

    /// Enforce the per-provider minimum spacing.
    async fn space(&self, provider: Provider) {
        let tag = Self::provider_tag(provider);
        let mut guard = self.last_send.lock().await;
        let now = Instant::now();
        if let Some(prev) = guard.get(tag) {
            let elapsed = now.duration_since(*prev);
            if elapsed < self.cfg.min_spacing {
                let wait = self.cfg.min_spacing - elapsed;
                // Hold the lock across the sleep so concurrent sends to the same provider serialize.
                tokio::time::sleep(wait).await;
            }
        }
        guard.insert(tag, Instant::now());
    }

    /// Send a built request, applying retry/backoff and spacing, and return the raw capture.
    pub async fn send(&self, built: &Built) -> Result<WireResponse> {
        let key = self.key_for(built.provider).await?;
        let body_bytes = serde_json::to_vec(&built.wire_body)?;
        let max_tries = if self.cfg.no_retry { 1 } else { 3 };
        let mut attempt = 0;
        loop {
            attempt += 1;
            self.space(built.provider).await;
            let start = Instant::now();
            let result = self.one_shot(built, &key, &body_bytes).await;
            match result {
                Ok((status, headers, bytes)) => {
                    let retryable = status == 429 || (500..600).contains(&status);
                    if retryable && attempt < max_tries {
                        backoff(attempt).await;
                        continue;
                    }
                    let elapsed_ms = start.elapsed().as_millis();
                    return Ok(if built.stream {
                        WireResponse::Stream {
                            status,
                            headers,
                            elapsed_ms,
                            raw: bytes,
                        }
                    } else {
                        WireResponse::NonStream {
                            status,
                            headers,
                            elapsed_ms,
                            body_bytes: bytes,
                        }
                    });
                }
                Err(e) => {
                    if attempt < max_tries {
                        backoff(attempt).await;
                        continue;
                    }
                    return Err(e).context("HTTP request failed after retries");
                }
            }
        }
    }

    async fn one_shot(
        &self,
        built: &Built,
        key: &ApiKey,
        body_bytes: &[u8],
    ) -> Result<(u16, BTreeMap<String, String>, Vec<u8>)> {
        let mut rb = self
            .http
            .post(&built.url)
            .header("content-type", "application/json")
            .body(body_bytes.to_vec());
        match built.provider {
            Provider::Anthropic => {
                rb = rb
                    .header("x-api-key", key.expose())
                    .header("anthropic-version", &self.cfg.anthropic_version);
            }
            Provider::OpenAI => {
                rb = rb.header("authorization", format!("Bearer {}", key.expose()));
            }
        }
        for (k, v) in &built.extra_headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        let resp = rb.send().await.map_err(|e| anyhow!("send: {e}"))?;
        let status = resp.status().as_u16();
        let headers = redact_response_headers(resp.headers());
        let bytes = resp.bytes().await.map_err(|e| anyhow!("read body: {e}"))?;
        Ok((status, headers, bytes.to_vec()))
    }
}

/// Exponential backoff for attempt `n` (1-based): 0.5s, 1s, 2s …
async fn backoff(attempt: u32) {
    let ms = 500u64.saturating_mul(1 << (attempt.saturating_sub(1)).min(4));
    tokio::time::sleep(Duration::from_millis(ms)).await;
}

/// Copy response headers into a string map, redacting secret ones.
fn redact_response_headers(headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (name, value) in headers {
        let n = name.as_str().to_string();
        let v = if is_secret_header(&n) {
            "<redacted>".to_string()
        } else {
            value.to_str().unwrap_or("<non-utf8>").to_string()
        };
        out.insert(n, v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::Catalogue;

    fn expansion(id: &str, proto: &str, stream: bool, headers: &str) -> Expansion {
        let toml = format!(
            "[[probe]]\nid=\"{id}\"\nprotocol=\"{proto}\"\nmodels=[\"m\"]\n[probe.body]\nmodel=\"$MODEL\"\nmax_tokens=256\n{headers}"
        );
        let p = Catalogue::parse_file(&toml).unwrap().remove(0);
        Expansion {
            probe: p,
            model: "test-model".into(),
            stream,
        }
    }

    fn client() -> Client {
        Client::new(ClientConfig {
            anthropic_base: "https://api.anthropic.com".into(),
            openai_base: "https://api.openai.com".into(),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn builds_anthropic_request() {
        let c = client();
        let ex = expansion("ant.text", "anthropic", false, "[probe.headers]\n\"anthropic-beta\" = \"beta-x\"\n");
        let built = c.build(&ex, ex.probe.body.clone()).unwrap();
        assert_eq!(built.url, "https://api.anthropic.com/v1/messages");
        assert_eq!(built.headers.get("x-api-key").map(String::as_str), Some("<redacted>"));
        assert_eq!(built.headers.get("anthropic-version").map(String::as_str), Some("2023-06-01"));
        assert_eq!(built.headers.get("anthropic-beta").map(String::as_str), Some("beta-x"));
        assert!(built.wire_body.get("stream").is_none());
    }

    #[test]
    fn builds_chat_stream_request() {
        let c = client();
        let ex = expansion("chat.text", "chat", true, "");
        let built = c.build(&ex, ex.probe.body.clone()).unwrap();
        assert_eq!(built.url, "https://api.openai.com/v1/chat/completions");
        assert_eq!(built.headers.get("authorization").map(String::as_str), Some("<redacted>"));
        assert_eq!(built.wire_body["stream"], serde_json::json!(true));
    }

    #[test]
    fn builds_responses_url() {
        let c = client();
        let ex = expansion("resp.text", "responses", false, "");
        let built = c.build(&ex, ex.probe.body.clone()).unwrap();
        assert_eq!(built.url, "https://api.openai.com/v1/responses");
        assert_eq!(built.provider, Provider::OpenAI);
    }

    #[test]
    fn rejects_probe_secret_header() {
        let c = client();
        let ex = expansion("ant.leak", "anthropic", false, "[probe.headers]\n\"x-api-key\" = \"nope\"\n");
        assert!(c.build(&ex, ex.probe.body.clone()).is_err());
    }

    #[test]
    fn request_record_is_redacted() {
        let c = client();
        let ex = expansion("ant.text", "anthropic", false, "");
        let built = c.build(&ex, ex.probe.body.clone()).unwrap();
        let rec = built.request_record();
        assert_eq!(rec.method, "POST");
        assert_eq!(rec.headers.get("x-api-key").map(String::as_str), Some("<redacted>"));
    }
}
