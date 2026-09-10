//! Named model sets (`models.toml`) and the `models --refresh` listing.
//!
//! Every probe names either an explicit list of model ids or a *set* defined here. Each set's
//! first member is the cheapest suitable model, which `--cheap` uses on its own. The exact ids
//! are curated from the live `GET /v1/models` listings (see [`refresh_listings`]); `refreshed_at`
//! records when, and `unverified` flags a set filled from best-knowledge because a listing call
//! failed.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One named set of model ids. The first member is the cheapest suitable model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelSet {
    /// Model ids, cheapest first.
    #[serde(default)]
    pub models: Vec<String>,
    /// True when the ids were not confirmed against a live listing.
    #[serde(default)]
    pub unverified: bool,
}

/// The whole `models.toml`: a date stamp plus the named sets.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelSets {
    /// ISO date of the last `--refresh` that informed these sets.
    #[serde(default)]
    pub refreshed_at: String,
    /// The named sets, keyed by set name.
    #[serde(default)]
    pub sets: BTreeMap<String, ModelSet>,
}

impl ModelSets {
    /// Parse from TOML text.
    pub fn from_toml(s: &str) -> Result<Self> {
        toml::from_str(s).context("parsing models.toml")
    }

    /// Load from a file path.
    pub fn load(path: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::from_toml(&txt)
    }

    /// Look up a set by name.
    pub fn get(&self, name: &str) -> Option<&ModelSet> {
        self.sets.get(name)
    }

    /// Whether a set name is known.
    pub fn has_set(&self, name: &str) -> bool {
        self.sets.contains_key(name)
    }

    /// Resolve a probe's `models` field (a set name or an explicit list) into concrete ids.
    /// When `cheap`, only the first member of a named set is returned (explicit lists are
    /// returned whole, but reduced to their first element under `cheap`).
    pub fn resolve(&self, spec: &ModelsSpec, cheap: bool) -> Result<Vec<String>> {
        let full: Vec<String> = match spec {
            ModelsSpec::Set(name) => {
                let set = self
                    .get(name)
                    .ok_or_else(|| anyhow!("unknown model set `{name}`"))?;
                set.models.clone()
            }
            ModelsSpec::List(list) => list.clone(),
        };
        if full.is_empty() {
            return Ok(full);
        }
        if cheap {
            Ok(vec![full[0].clone()])
        } else {
            Ok(full)
        }
    }
}

/// A probe's `models` value: a named set, or an explicit list of ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelsSpec {
    /// A set name defined in `models.toml`.
    Set(String),
    /// An explicit list of model ids.
    List(Vec<String>),
}

/// The default path of `models.toml` (crate-relative when running from source).
pub fn default_models_path(crate_dir: &Path) -> PathBuf {
    crate_dir.join("models.toml")
}

// ---------------------------------------------------------------------------------------------
// Live refresh (the ONE allowed live call — free `GET /v1/models`).
// ---------------------------------------------------------------------------------------------

/// One entry from a provider's `/v1/models` listing.
#[derive(Debug, Clone, Serialize)]
pub struct ListingEntry {
    /// Model id.
    pub id: String,
    /// Unix `created` timestamp when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<i64>,
}

/// A dated, per-provider `/v1/models` listing, written to `models.listing.json`.
#[derive(Debug, Clone, Serialize)]
pub struct Listing {
    /// ISO date of the refresh.
    pub refreshed_at: String,
    /// Anthropic model ids.
    pub anthropic: Vec<ListingEntry>,
    /// OpenAI model ids.
    pub openai: Vec<ListingEntry>,
}

/// Parse an Anthropic `/v1/models` body (`{ "data": [ { "id", "created_at" } ] }`).
pub fn parse_anthropic_listing(body: &serde_json::Value) -> Vec<ListingEntry> {
    parse_listing_generic(body, "created_at")
}

/// Parse an OpenAI `/v1/models` body (`{ "data": [ { "id", "created" } ] }`).
pub fn parse_openai_listing(body: &serde_json::Value) -> Vec<ListingEntry> {
    parse_listing_generic(body, "created")
}

/// Fetch both providers' `/v1/models` listings (the single free live call permitted in the offline
/// phase). Keys are read at call time and never logged. Either provider may fail independently; a
/// failed provider yields an empty list and the error is returned to the caller for reporting.
pub async fn refresh_listings(
    http: &reqwest::Client,
    anthropic_base: &str,
    openai_base: &str,
) -> (Listing, Vec<String>) {
    use crate::keys::{load_key, Provider};
    let mut errors = Vec::new();
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

    let anthropic = match fetch_one(
        http,
        &format!("{}/v1/models", anthropic_base.trim_end_matches('/')),
        Provider::Anthropic,
        load_key(Provider::Anthropic),
    )
    .await
    {
        Ok(v) => parse_anthropic_listing(&v),
        Err(e) => {
            errors.push(format!("anthropic: {e}"));
            Vec::new()
        }
    };
    let openai = match fetch_one(
        http,
        &format!("{}/v1/models", openai_base.trim_end_matches('/')),
        Provider::OpenAI,
        load_key(Provider::OpenAI),
    )
    .await
    {
        Ok(v) => parse_openai_listing(&v),
        Err(e) => {
            errors.push(format!("openai: {e}"));
            Vec::new()
        }
    };
    (
        Listing {
            refreshed_at: today,
            anthropic,
            openai,
        },
        errors,
    )
}

async fn fetch_one(
    http: &reqwest::Client,
    url: &str,
    provider: crate::keys::Provider,
    key: Result<crate::keys::ApiKey>,
) -> Result<serde_json::Value> {
    use crate::keys::Provider;
    let key = key?;
    let mut rb = http.get(url);
    match provider {
        Provider::Anthropic => {
            rb = rb
                .header("x-api-key", key.expose())
                .header("anthropic-version", "2023-06-01");
        }
        Provider::OpenAI => {
            rb = rb.header("authorization", format!("Bearer {}", key.expose()));
        }
    }
    let resp = rb.send().await.context("GET /v1/models")?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.context("parsing /v1/models body")?;
    if !status.is_success() {
        anyhow::bail!("GET {url} returned {status}");
    }
    Ok(body)
}

fn parse_listing_generic(body: &serde_json::Value, created_key: &str) -> Vec<ListingEntry> {
    let mut out = Vec::new();
    if let Some(arr) = body.get("data").and_then(|d| d.as_array()) {
        for e in arr {
            let Some(id) = e.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            // Anthropic's `created_at` is an RFC3339 string; OpenAI's `created` is a unix int.
            let created = match e.get(created_key) {
                Some(serde_json::Value::Number(n)) => n.as_i64(),
                Some(serde_json::Value::String(s)) => {
                    chrono::DateTime::parse_from_rfc3339(s).ok().map(|d| d.timestamp())
                }
                _ => None,
            };
            out.push(ListingEntry { id: id.to_string(), created });
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
refreshed_at = "2026-09-10"

[sets.anthropic_new]
models = ["claude-opus-4-8", "claude-sonnet-4-8"]

[sets.openai_chat]
models = ["gpt-4o-mini", "gpt-4o"]

[sets.openai_compat]
models = []
unverified = true
"#;

    #[test]
    fn parse_and_lookup() {
        let m = ModelSets::from_toml(SAMPLE).unwrap();
        assert_eq!(m.refreshed_at, "2026-09-10");
        assert!(m.has_set("anthropic_new"));
        assert!(!m.has_set("nope"));
        assert_eq!(m.get("anthropic_new").unwrap().models.len(), 2);
        assert!(m.get("openai_compat").unwrap().unverified);
    }

    #[test]
    fn resolve_full_and_cheap() {
        let m = ModelSets::from_toml(SAMPLE).unwrap();
        let full = m.resolve(&ModelsSpec::Set("anthropic_new".into()), false).unwrap();
        assert_eq!(full, vec!["claude-opus-4-8", "claude-sonnet-4-8"]);
        let cheap = m.resolve(&ModelsSpec::Set("anthropic_new".into()), true).unwrap();
        assert_eq!(cheap, vec!["claude-opus-4-8"]);
    }

    #[test]
    fn resolve_explicit_list() {
        let m = ModelSets::from_toml(SAMPLE).unwrap();
        let list = ModelsSpec::List(vec!["a".into(), "b".into()]);
        assert_eq!(m.resolve(&list, false).unwrap(), vec!["a", "b"]);
        assert_eq!(m.resolve(&list, true).unwrap(), vec!["a"]);
    }

    #[test]
    fn resolve_unknown_set_errors() {
        let m = ModelSets::from_toml(SAMPLE).unwrap();
        assert!(m.resolve(&ModelsSpec::Set("ghost".into()), false).is_err());
    }

    #[test]
    fn parse_openai_listing_sorts_and_reads_created() {
        let body = serde_json::json!({
            "data": [
                {"id": "gpt-4o", "created": 1700000000},
                {"id": "gpt-3.5", "created": 1600000000}
            ]
        });
        let l = parse_openai_listing(&body);
        assert_eq!(l[0].id, "gpt-3.5");
        assert_eq!(l[1].id, "gpt-4o");
        assert_eq!(l[1].created, Some(1700000000));
    }

    #[test]
    fn parse_anthropic_listing_reads_rfc3339() {
        let body = serde_json::json!({
            "data": [{"id": "claude-opus-4-8", "created_at": "2026-01-01T00:00:00Z"}]
        });
        let l = parse_anthropic_listing(&body);
        assert_eq!(l[0].id, "claude-opus-4-8");
        assert!(l[0].created.is_some());
    }
}
