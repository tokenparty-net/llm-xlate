//! The upstream I/O seam (plan §4.7, §4.8): one HTTP client per provider, provider auth from
//! keys (never from client headers), and a streaming-or-full response body.
//!
//! The trait is object-safe (futures are boxed) so [`Deps`](crate::Deps) can hold an
//! `Arc<dyn Upstream>` and tests can inject a mock.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use futures::stream::Stream;

use llm_xlate_core::ir::Protocol;
use llm_xlate_core::HeaderMap;

/// A boxed, `Send` future — the object-safe return of [`Upstream::send`].
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A boxed, `Send` byte stream — a streaming upstream body.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, UpstreamError>> + Send>>;

/// A transport-level upstream failure (connection, timeout, mid-stream read error). Protocol
/// error bodies are **not** this — those come back as a non-2xx [`UpstreamResponse`].
#[derive(Debug, Clone)]
pub struct UpstreamError {
    /// Human-readable message (never contains a secret).
    pub message: String,
    /// Whether the failure was a timeout.
    pub timeout: bool,
}

impl UpstreamError {
    /// A generic transport error.
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), timeout: false }
    }
    /// A timeout error.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self { message: message.into(), timeout: true }
    }
}

impl std::fmt::Display for UpstreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for UpstreamError {}

/// A request to send upstream. The URL and non-auth headers are shaped by the caller; the
/// [`Upstream`] impl adds provider auth from its keys.
pub struct UpstreamRequest {
    /// Provider name (`[providers.<name>]`).
    pub provider: String,
    /// The wire protocol (selects the path when the caller does not override it).
    pub protocol: Protocol,
    /// Full request URL.
    pub url: String,
    /// Protocol/beta headers to send (e.g. `anthropic-version`, `anthropic-beta`). Auth is added
    /// by the impl and never taken from client headers.
    pub headers: HeaderMap,
    /// The serialized request body.
    pub body: Bytes,
    /// Whether the upstream request asks for a streaming response.
    pub stream: bool,
}

/// A response body: a fully-buffered blob or a byte stream.
pub enum UpstreamBody {
    /// A complete, buffered body.
    Full(Bytes),
    /// A streaming body (SSE).
    Stream(ByteStream),
}

impl std::fmt::Debug for UpstreamBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpstreamBody::Full(b) => f.debug_tuple("Full").field(&b.len()).finish(),
            UpstreamBody::Stream(_) => f.write_str("Stream(..)"),
        }
    }
}

/// An upstream response.
pub struct UpstreamResponse {
    /// HTTP status.
    pub status: u16,
    /// Response headers (redacted before they reach a trace).
    pub headers: HeaderMap,
    /// The response body.
    pub body: UpstreamBody,
    /// Time to first byte / first frame.
    pub ttfb: Duration,
}

/// The upstream I/O trait (plan §4). Object-safe: [`Upstream::send`] returns a boxed future.
pub trait Upstream: Send + Sync {
    /// Send a request upstream and return the (possibly streaming) response.
    fn send<'a>(&'a self, req: UpstreamRequest) -> BoxFuture<'a, Result<UpstreamResponse, UpstreamError>>;
}

/// The wire path for an upstream protocol.
pub fn upstream_path(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::OaiChat => "/v1/chat/completions",
        Protocol::OaiResponses => "/v1/responses",
        Protocol::Anthropic => "/v1/messages",
    }
}

/// A provider secret held so it is never accidentally logged.
#[derive(Clone)]
pub struct ProviderKey(String);

impl ProviderKey {
    /// Wrap a secret string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    /// Expose the raw secret (only the HTTP client calls this, at send time).
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for ProviderKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// The production [`Upstream`]: one [`reqwest::Client`] per provider, provider auth injected from
/// keys, no redirects. Compiled always but exercised only in the (operator-run) `--features live`
/// path; `cargo test` uses the mock.
pub struct ReqwestUpstream {
    clients: std::collections::HashMap<String, ProviderClient>,
}

struct ProviderClient {
    client: reqwest::Client,
    family: ProviderFamilyKind,
    key: Option<ProviderKey>,
}

#[derive(Clone, Copy)]
enum ProviderFamilyKind {
    Anthropic,
    OpenAI,
    Other,
}

impl ReqwestUpstream {
    /// Build clients for each provider. `keys` supplies the (already-loaded) secret per provider
    /// name; providers without a key can still be reached for keyless compat servers.
    pub fn new(
        providers: &std::collections::BTreeMap<String, crate::config::ProviderConfig>,
        keys: &std::collections::BTreeMap<String, ProviderKey>,
        timeout: Duration,
    ) -> Result<Self, UpstreamError> {
        let mut clients = std::collections::HashMap::new();
        for name in providers.keys() {
            let client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(timeout)
                .build()
                .map_err(|e| UpstreamError::new(format!("building reqwest client: {e}")))?;
            let family = match crate::route::Router::provider_family(name) {
                llm_xlate_core::ir::ProviderFamily::Anthropic => ProviderFamilyKind::Anthropic,
                llm_xlate_core::ir::ProviderFamily::OpenAI => ProviderFamilyKind::OpenAI,
                _ => ProviderFamilyKind::Other,
            };
            clients.insert(
                name.clone(),
                ProviderClient { client, family, key: keys.get(name).cloned() },
            );
        }
        Ok(Self { clients })
    }
}

impl Upstream for ReqwestUpstream {
    fn send<'a>(&'a self, req: UpstreamRequest) -> BoxFuture<'a, Result<UpstreamResponse, UpstreamError>> {
        Box::pin(async move {
            let pc = self
                .clients
                .get(&req.provider)
                .ok_or_else(|| UpstreamError::new(format!("no client for provider `{}`", req.provider)))?;

            let mut headers = req.headers.clone();
            // Provider auth from keys only — never from client headers.
            if let Some(key) = &pc.key {
                match pc.family {
                    ProviderFamilyKind::Anthropic => {
                        if let Ok(v) = key.expose().parse() {
                            headers.insert("x-api-key", v);
                        }
                    }
                    ProviderFamilyKind::OpenAI | ProviderFamilyKind::Other => {
                        if let Ok(v) = format!("Bearer {}", key.expose()).parse() {
                            headers.insert(http::header::AUTHORIZATION, v);
                        }
                    }
                }
            }
            if !headers.contains_key(http::header::CONTENT_TYPE) {
                headers.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
            }

            let started = std::time::Instant::now();
            let resp = pc
                .client
                .post(&req.url)
                .headers(headers)
                .body(req.body.to_vec())
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        UpstreamError::timeout(format!("upstream request timed out: {e}"))
                    } else {
                        UpstreamError::new(format!("upstream request failed: {e}"))
                    }
                })?;

            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let ttfb = started.elapsed();

            if req.stream {
                use futures::StreamExt;
                let stream = resp
                    .bytes_stream()
                    .map(|r| r.map_err(|e| UpstreamError::new(format!("upstream stream error: {e}"))));
                Ok(UpstreamResponse {
                    status,
                    headers,
                    body: UpstreamBody::Stream(Box::pin(stream)),
                    ttfb,
                })
            } else {
                let body = resp
                    .bytes()
                    .await
                    .map_err(|e| UpstreamError::new(format!("reading upstream body: {e}")))?;
                Ok(UpstreamResponse { status, headers, body: UpstreamBody::Full(body), ttfb })
            }
        })
    }
}
