//! Router configuration (plan §3): the `router.toml` schema, defaults, validation, and the
//! per-request header overrides.
//!
//! Validation is split in two: [`Config::validate`] runs at parse time (route regexes compile;
//! every provider a route names exists), while key-file resolvability is deferred to
//! [`Config::unresolvable_providers`] at `serve` time — an operator can inspect or lint a config
//! (`routes` subcommand) without the provider secrets present.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::ir::{Protocol, ReasoningExposure, SummaryLevel};
use llm_xlate_core::HeaderMap;

/// The whole `router.toml`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    /// Server + I/O settings.
    pub server: ServerConfig,
    /// Trace-writer settings.
    pub trace: TraceConfig,
    /// Upstream providers, keyed by the name routes reference (`anthropic`, `openai`, …).
    pub providers: BTreeMap<String, ProviderConfig>,
    /// Ordered route rules; first regex match on the client model name wins.
    #[serde(rename = "route")]
    pub routes: Vec<RouteRule>,
    /// Client model name → upstream model name (client name is echoed back); `default` is used
    /// when the client omits `model`.
    pub aliases: BTreeMap<String, String>,
    /// Optional per-provider [`BackendOverrides`](llm_xlate_core::caps::BackendOverrides) merged
    /// onto registry caps.
    pub backend_overrides: BTreeMap<String, Capabilities>,
}

/// `[server]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Listen address.
    pub listen: String,
    /// Shared client token; empty accepts any client credential (default for local testing).
    pub token: String,
    /// Maximum request body size in bytes.
    pub max_body_bytes: usize,
    /// Client SSE keepalive interval (seconds) when the upstream is slow or non-streaming.
    pub keepalive_secs: u64,
    /// Data directory holding `traces/`, `store/`, `sidecar/`.
    pub data_dir: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8787".to_string(),
            token: String::new(),
            max_body_bytes: 33_554_432,
            keepalive_secs: 15,
            data_dir: "crates/testrouter/data".to_string(),
        }
    }
}

/// `[trace]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct TraceConfig {
    /// Whether tracing is enabled.
    pub enabled: bool,
    /// Trace file template (`{date}` is substituted); relative to `data_dir`.
    pub file: String,
    /// Base64 payloads larger than this are replaced by `{sha256, len}` in the trace.
    pub redact_media_over_bytes: usize,
    /// Keep the exact upstream SSE bytes and exact client frames.
    pub include_raw_sse: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            file: "traces/{date}.jsonl".to_string(),
            redact_media_over_bytes: 65_536,
            include_raw_sse: true,
        }
    }
}

/// `[providers.<name>]`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct ProviderConfig {
    /// Upstream base URL (e.g. `https://api.anthropic.com`).
    pub base_url: String,
    /// Key file name resolved relative to the workspace parent (like `llm-xlate-e2e`).
    pub key_file: Option<String>,
    /// Environment variable holding the key (fallback / OpenAI-compatible servers).
    pub key_env: Option<String>,
    /// Model ids this provider serves, advertised on `GET /v1/models` (offline; no live listing).
    /// Only ids that also match a route are listed.
    #[serde(default)]
    pub models: Vec<String>,
}

/// One `[[route]]` rule.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouteRule {
    /// Regex tried against the client model name (first match wins).
    #[serde(rename = "match")]
    pub match_: String,
    /// Provider name (must exist in `[providers]`).
    pub provider: String,
    /// Upstream wire protocol (`chat` | `responses` | `anthropic`).
    #[serde(with = "protocol_token")]
    pub upstream: Protocol,
}

impl Config {
    /// Parse and validate a config from a TOML string.
    pub fn from_toml(src: &str) -> Result<Config> {
        let cfg: Config = toml::from_str(src).context("parsing router.toml")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load and validate a config from a file path.
    pub fn load(path: impl AsRef<Path>) -> Result<Config> {
        let path = path.as_ref();
        let src = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        Config::from_toml(&src)
    }

    /// The canonical example configuration (matches `router.example.toml`).
    pub fn example() -> Config {
        Config::from_toml(EXAMPLE_TOML).expect("example config is valid")
    }

    /// Parse-time validation: route regexes compile and every provider a route names exists.
    /// Key-file resolvability is checked later, at serve time.
    pub fn validate(&self) -> Result<()> {
        for (i, r) in self.routes.iter().enumerate() {
            Regex::new(&r.match_)
                .with_context(|| format!("route[{i}] has an invalid regex `{}`", r.match_))?;
            if !self.providers.contains_key(&r.provider) {
                bail!("route[{i}] references unknown provider `{}`", r.provider);
            }
        }
        Ok(())
    }

    /// Serve-time check that every provider referenced by a route can supply a key (a resolvable
    /// `key_file` or a set `key_env`). Returns the names that cannot.
    pub fn unresolvable_providers(&self, workspace_parent: &Path) -> Vec<String> {
        let mut bad = Vec::new();
        for r in &self.routes {
            if let Some(p) = self.providers.get(&r.provider) {
                let has_file = p
                    .key_file
                    .as_ref()
                    .is_some_and(|f| workspace_parent.join(f).is_file());
                let has_env = p.key_env.as_ref().is_some_and(|e| std::env::var(e).is_ok());
                if !has_file && !has_env {
                    bad.push(r.provider.clone());
                }
            }
        }
        bad.sort();
        bad.dedup();
        bad
    }

    /// Compile the route regexes once (used by [`crate::route::Router`]).
    pub fn compiled_routes(&self) -> Result<Vec<(Regex, RouteRule)>> {
        self.routes
            .iter()
            .map(|r| Ok((Regex::new(&r.match_)?, r.clone())))
            .collect()
    }
}

/// Per-request header overrides (plan §3), all optional and all recorded in the trace.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Overrides {
    /// `x-xlate-model` — the upstream model string.
    pub model: Option<String>,
    /// `x-xlate-upstream` — the upstream protocol.
    #[serde(with = "opt_protocol_token")]
    pub upstream: Option<Protocol>,
    /// `x-xlate-provider` — the provider name.
    pub provider: Option<String>,
    /// `x-xlate-caps` — a preset name from `llm_xlate::caps::preset`.
    pub caps: Option<String>,
    /// `x-xlate-expose` — reasoning exposure (`none` | `summary` | `full`).
    pub expose: Option<String>,
    /// `x-xlate-tag` — free text copied into the trace for grouping a session.
    pub tag: Option<String>,
}

impl Overrides {
    /// Parse the `x-xlate-*` overrides from request headers. Unknown values for `upstream` are
    /// dropped (recorded as absent); everything else is copied verbatim.
    pub fn from_headers(hdrs: &HeaderMap) -> Overrides {
        let get = |name: &str| {
            hdrs.get(name)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        Overrides {
            model: get("x-xlate-model"),
            upstream: get("x-xlate-upstream").and_then(|s| parse_protocol_token(&s)),
            provider: get("x-xlate-provider"),
            caps: get("x-xlate-caps"),
            expose: get("x-xlate-expose"),
            tag: get("x-xlate-tag"),
        }
    }

    /// The requested reasoning exposure, if `x-xlate-expose` was set and understood.
    pub fn expose_mode(&self) -> Option<ReasoningExposure> {
        match self.expose.as_deref() {
            Some("none") => Some(ReasoningExposure::None),
            Some("summary") => Some(ReasoningExposure::Summary(SummaryLevel::Auto)),
            Some("full") => Some(ReasoningExposure::Full),
            _ => None,
        }
    }
}

/// Parse a protocol token (`chat` | `responses` | `anthropic`).
pub fn parse_protocol_token(s: &str) -> Option<Protocol> {
    match s {
        "chat" => Some(Protocol::OaiChat),
        "responses" => Some(Protocol::OaiResponses),
        "anthropic" => Some(Protocol::Anthropic),
        _ => None,
    }
}

/// The wire token for a protocol.
pub fn protocol_token_str(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "chat",
        Protocol::OaiResponses => "responses",
        Protocol::Anthropic => "anthropic",
    }
}

mod protocol_token {
    use super::*;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(p: &Protocol, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(super::protocol_token_str(*p))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Protocol, D::Error> {
        let s = String::deserialize(d)?;
        super::parse_protocol_token(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown upstream protocol `{s}`")))
    }
}

mod opt_protocol_token {
    use super::*;
    use serde::Serializer;

    pub fn serialize<S: Serializer>(p: &Option<Protocol>, s: S) -> Result<S::Ok, S::Error> {
        match p {
            Some(p) => s.serialize_str(super::protocol_token_str(*p)),
            None => s.serialize_none(),
        }
    }
}

/// The example config text, kept in sync with `router.example.toml`.
pub const EXAMPLE_TOML: &str = r#"[server]
listen = "127.0.0.1:8787"
token = ""
max_body_bytes = 33554432
keepalive_secs = 15
data_dir = "crates/testrouter/data"

[trace]
enabled = true
file = "traces/{date}.jsonl"
redact_media_over_bytes = 65536
include_raw_sse = true

[providers.anthropic]
base_url = "https://api.anthropic.com"
key_file = ".xlate_e2e_claude"

[providers.openai]
base_url = "https://api.openai.com"
key_file = ".xlate_e2e_oai"

[providers.compat]
base_url = ""
key_env = "E2E_OPENAI_COMPAT_API_KEY"

[[route]]
match = "^claude-"
provider = "anthropic"
upstream = "anthropic"

[[route]]
match = "^(gpt-6|o[0-9])"
provider = "openai"
upstream = "responses"

[[route]]
match = "^gpt-"
provider = "openai"
upstream = "responses"

[[route]]
match = "^chatgpt-|^compat/"
provider = "openai"
upstream = "chat"

[aliases]
"claude" = "claude-sonnet-5"
"gpt" = "gpt-5.4"
"default" = "claude-sonnet-5"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_parses_and_validates() {
        let cfg = Config::example();
        assert_eq!(cfg.server.listen, "127.0.0.1:8787");
        assert_eq!(cfg.routes.len(), 4);
        assert_eq!(cfg.routes[0].upstream, Protocol::Anthropic);
        assert_eq!(cfg.aliases.get("default").map(String::as_str), Some("claude-sonnet-5"));
        assert!(cfg.providers.contains_key("openai"));
    }

    #[test]
    fn rejects_unknown_provider() {
        let toml = r#"
            [providers.openai]
            base_url = "x"
            [[route]]
            match = "^gpt-"
            provider = "nope"
            upstream = "chat"
        "#;
        assert!(Config::from_toml(toml).is_err());
    }

    #[test]
    fn rejects_bad_regex() {
        let toml = r#"
            [providers.openai]
            base_url = "x"
            [[route]]
            match = "("
            provider = "openai"
            upstream = "chat"
        "#;
        assert!(Config::from_toml(toml).is_err());
    }

    #[test]
    fn parses_header_overrides() {
        let mut h = HeaderMap::new();
        h.insert("x-xlate-model", "gpt-5.4".parse().unwrap());
        h.insert("x-xlate-upstream", "chat".parse().unwrap());
        h.insert("x-xlate-caps", "gpt5_chat".parse().unwrap());
        h.insert("x-xlate-expose", "summary".parse().unwrap());
        h.insert("x-xlate-tag", "sess-1".parse().unwrap());
        let o = Overrides::from_headers(&h);
        assert_eq!(o.model.as_deref(), Some("gpt-5.4"));
        assert_eq!(o.upstream, Some(Protocol::OaiChat));
        assert_eq!(o.caps.as_deref(), Some("gpt5_chat"));
        assert_eq!(o.expose_mode(), Some(ReasoningExposure::Summary(SummaryLevel::Auto)));
        assert_eq!(o.tag.as_deref(), Some("sess-1"));
    }

    #[test]
    fn ignores_unknown_upstream_override() {
        let mut h = HeaderMap::new();
        h.insert("x-xlate-upstream", "grpc".parse().unwrap());
        assert_eq!(Overrides::from_headers(&h).upstream, None);
    }
}
