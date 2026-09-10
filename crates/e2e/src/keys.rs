//! Provider API-key loading and secret redaction.
//!
//! Keys are read at runtime only, from files at the *parent* of the workspace root
//! (`../.xlate_e2e_claude`, `../.xlate_e2e_oai`) with `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
//! as a fallback. A key value is never printed, logged, or written to disk: [`ApiKey`]'s
//! [`Debug`] renders `<redacted>`, and [`looks_like_key`] lets the capture writer refuse to
//! persist anything that resembles a secret.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};

/// A provider API key. Its [`Debug`] / [`std::fmt::Display`] never reveal the value.
#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    /// The raw secret. Only the HTTP client calls this, at send time.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

impl std::fmt::Display for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// Which provider a key belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI (Chat Completions and Responses).
    OpenAI,
}

impl Provider {
    /// The key file name at the workspace parent directory.
    fn key_file(self) -> &'static str {
        match self {
            Provider::Anthropic => ".xlate_e2e_claude",
            Provider::OpenAI => ".xlate_e2e_oai",
        }
    }

    /// The environment-variable fallback.
    fn env_var(self) -> &'static str {
        match self {
            Provider::Anthropic => "ANTHROPIC_API_KEY",
            Provider::OpenAI => "OPENAI_API_KEY",
        }
    }
}

/// Walk up from `start` until a `Cargo.toml` containing `[workspace]` is found; return that
/// directory (the workspace root).
pub fn find_workspace_root(start: &Path) -> Result<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let cargo = d.join("Cargo.toml");
        if cargo.is_file() {
            if let Ok(txt) = std::fs::read_to_string(&cargo) {
                if txt.contains("[workspace]") {
                    return Ok(d.to_path_buf());
                }
            }
        }
        dir = d.parent();
    }
    Err(anyhow!(
        "could not find a workspace-root Cargo.toml (with [workspace]) above {}",
        start.display()
    ))
}

/// The workspace root, discovered from the current working directory.
pub fn workspace_root() -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("current_dir")?;
    find_workspace_root(&cwd)
}

/// Load a provider key: the file `../<key_file>` relative to the workspace root, else the
/// environment fallback. The value is trimmed of surrounding whitespace.
pub fn load_key(provider: Provider) -> Result<ApiKey> {
    let root = workspace_root()?;
    let parent = root
        .parent()
        .ok_or_else(|| anyhow!("workspace root {} has no parent", root.display()))?;
    let path = parent.join(provider.key_file());
    if path.is_file() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading key file {}", path.display()))?;
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Ok(ApiKey(trimmed.to_string()));
        }
    }
    if let Ok(v) = std::env::var(provider.env_var()) {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            return Ok(ApiKey(trimmed.to_string()));
        }
    }
    Err(anyhow!(
        "no key for {:?}: neither {} nor ${}",
        provider,
        path.display(),
        provider.env_var()
    ))
}

/// Heuristic detector for provider secrets, used to refuse persisting a body or header that
/// contains one. Matches OpenAI (`sk-`, `sk-proj-`, `sk-svcacct-`) and Anthropic
/// (`sk-ant-`) key shapes: the prefix followed by a run of key-ish characters.
pub fn looks_like_key(s: &str) -> bool {
    // Scan for any of the known prefixes and require enough trailing key characters that it is
    // implausibly a normal English word.
    // Work entirely at the byte level: the prefixes and key characters are all ASCII, and slicing
    // `s[start..]` at an arbitrary byte offset panics when `start` falls inside a multi-byte UTF-8
    // char (real response bodies contain e.g. `’`). Byte comparison is boundary-safe.
    const PREFIXES: [&[u8]; 2] = [b"sk-ant-", b"sk-"];
    let bytes = s.as_bytes();
    for start in 0..bytes.len() {
        for pfx in PREFIXES {
            if bytes[start..].starts_with(pfx) {
                let rest = &bytes[start + pfx.len()..];
                let run = rest
                    .iter()
                    .take_while(|b| b.is_ascii_alphanumeric() || **b == b'-' || **b == b'_')
                    .count();
                if run >= 16 {
                    return true;
                }
            }
        }
    }
    false
}

/// Header names whose values must be redacted in every capture. Besides the request-side
/// credentials, this covers the account identifiers Anthropic echoes on *responses*
/// (`anthropic-organization-id` / `anthropic-workspace-id` are real account UUIDs) so they never
/// reach a committed capture or promoted fixture.
pub const REDACT_HEADERS: [&str; 8] = [
    "authorization",
    "x-api-key",
    "openai-organization",
    "openai-project",
    "cookie",
    "set-cookie",
    "anthropic-organization-id",
    "anthropic-workspace-id",
];

/// Whether a header (case-insensitive) must have its value redacted.
pub fn is_secret_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    REDACT_HEADERS.contains(&lower.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts() {
        let k = ApiKey("sk-ant-secretsecretsecret".into());
        assert_eq!(format!("{k:?}"), "<redacted>");
        assert_eq!(format!("{k}"), "<redacted>");
        assert_eq!(k.expose(), "sk-ant-secretsecretsecret");
    }

    #[test]
    fn detects_openai_keys() {
        assert!(looks_like_key("sk-abcdefghijklmnop0123"));
        assert!(looks_like_key("sk-proj-abcdefghijklmnop0123"));
        assert!(looks_like_key(
            "prefix sk-svcacct-abcdefghijklmnop0123 suffix"
        ));
    }

    #[test]
    fn detects_anthropic_keys() {
        assert!(looks_like_key("sk-ant-api03-abcdefghijklmnop0123"));
        assert!(looks_like_key("here is a sk-ant-abcdefghijklmnop key"));
    }

    #[test]
    fn ignores_normal_text() {
        assert!(!looks_like_key("this is a normal sentence"));
        assert!(!looks_like_key("sk-"));
        assert!(!looks_like_key("sk-short"));
        assert!(!looks_like_key("ask-me-anything"));
    }

    #[test]
    fn no_panic_on_multibyte_utf8_body() {
        // Regression: `looks_like_key` used to slice `s[start..]` at raw byte offsets and panicked
        // when `start` landed inside a multi-byte char. Real response bodies carry `’`, `—`, emoji.
        let body = "The user’s reply — 世界 🌍 — contains no key. sk-not-a-real-key";
        assert!(!looks_like_key(body));
        // A key embedded in a multibyte context is still detected.
        let with_key = "reply ’’’ sk-ant-abcdefghijklmnop0123 世界";
        assert!(looks_like_key(with_key));
    }

    #[test]
    fn secret_header_matching_is_case_insensitive() {
        assert!(is_secret_header("Authorization"));
        assert!(is_secret_header("X-Api-Key"));
        assert!(is_secret_header("OpenAI-Organization"));
        assert!(is_secret_header("cookie"));
        assert!(is_secret_header("Anthropic-Organization-Id"));
        assert!(is_secret_header("anthropic-workspace-id"));
        assert!(!is_secret_header("anthropic-version"));
        assert!(!is_secret_header("x-request-id"));
    }

    #[test]
    fn finds_workspace_root() {
        // The e2e crate dir is <root>/crates/e2e; walking up must reach <root>.
        let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = find_workspace_root(here).unwrap();
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.ends_with("llm-xlate"));
    }
}
