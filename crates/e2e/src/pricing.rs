//! Spend estimation for live runs.
//!
//! `llm-xlate-e2e` never has access to a real invoice, so every live run computes a *deliberately
//! conservative* dollar estimate from the captured `usage` counters and a hand-maintained price
//! table. The table holds published list prices per 1,000,000 tokens (USD), dated below; where a
//! model's price is not known (e.g. an unreleased flagship in this account's listing) we fall back
//! to the **most expensive tier of that provider**, so the estimate over-counts rather than
//! under-counts — the point of the estimate is a spend guard, not an accounting record.
//!
//! Prices are a floor for planning only. Update `PRICE_TABLE` and `PRICES_DATED` when the
//! providers change list prices.
//!
//! Usage-key shapes handled (the observer flattens nested usage to dotted keys):
//! - Anthropic Messages: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
//!   `cache_read_input_tokens` (input excludes cached; the three cache/output buckets are additive).
//! - OpenAI Chat: `prompt_tokens` (includes cached), `completion_tokens`,
//!   `prompt_tokens_details.cached_tokens`.
//! - OpenAI Responses: `input_tokens` (includes cached), `output_tokens`,
//!   `input_tokens_details.cached_tokens`.

use serde_json::Value;
use std::collections::BTreeMap;

/// Date the price table was last checked against the providers' published list prices.
pub const PRICES_DATED: &str = "2026-09-10";

/// List prices for one model, USD per 1,000,000 tokens.
#[derive(Debug, Clone, Copy)]
pub struct Prices {
    /// Non-cached input / prompt tokens.
    pub input: f64,
    /// Output / completion tokens (reasoning tokens are billed here).
    pub output: f64,
    /// Cache *write* (Anthropic `cache_creation_input_tokens`, 5m TTL list price).
    pub cache_write: f64,
    /// Cache *read* (Anthropic `cache_read_input_tokens`; OpenAI cached prompt tokens).
    pub cache_read: f64,
}

impl Prices {
    const fn new(input: f64, output: f64, cache_write: f64, cache_read: f64) -> Self {
        Prices {
            input,
            output,
            cache_write,
            cache_read,
        }
    }
}

/// Which provider a model id belongs to (`"anthropic"` or `"openai"`).
pub fn provider_for(model: &str) -> &'static str {
    if model.starts_with("claude") || model.starts_with("anthropic") {
        "anthropic"
    } else {
        "openai"
    }
}

/// The most-expensive tier per provider — the over-estimating fallback for unknown ids.
fn fallback_prices(provider: &str) -> Prices {
    match provider {
        // Opus tier (the priciest Anthropic mainline).
        "anthropic" => Prices::new(15.0, 75.0, 18.75, 1.50),
        // A flagship OpenAI tier; deliberately high so unknown ids over-count.
        _ => Prices::new(15.0, 60.0, 15.0, 1.875),
    }
}

/// Look up list prices for a model id, and whether the value is a real published price (`true`) or
/// an over-estimating fallback / best-knowledge figure for a model whose list price is unconfirmed
/// (`false`).
///
/// Matching is by id prefix/substring so dated variants (`claude-sonnet-4-5-20250929`) resolve to
/// their family. Ordering matters: check the most specific families first.
pub fn prices_for(model: &str) -> (Prices, bool) {
    for (needle, prices, verified) in PRICE_TABLE {
        if model.contains(needle) {
            return (*prices, *verified);
        }
    }
    (fallback_prices(provider_for(model)), false)
}

/// The price table, checked most-specific-first. `(id substring, prices, verified)`.
///
/// `verified = true` = the provider's published list price on [`PRICES_DATED`]. `verified = false`
/// = a best-knowledge / over-estimating figure for a model whose list price this account could not
/// confirm (the newer flagships in the listing) — always set at or above the real figure.
#[allow(clippy::type_complexity)]
static PRICE_TABLE: &[(&str, Prices, bool)] = &[
    // ---- Anthropic (input / output / cache-write-5m / cache-read) --------------------------
    // Haiku tier.
    ("claude-haiku-4-5", Prices::new(1.0, 5.0, 1.25, 0.10), true),
    ("claude-haiku", Prices::new(1.0, 5.0, 1.25, 0.10), false),
    // Opus tier (all 4.x opus + opus-5). Priciest; list at the standard Anthropic opus rate.
    ("claude-opus-4-5", Prices::new(15.0, 75.0, 18.75, 1.50), true),
    ("claude-opus-4-6", Prices::new(15.0, 75.0, 18.75, 1.50), true),
    ("claude-opus-4-7", Prices::new(15.0, 75.0, 18.75, 1.50), false),
    ("claude-opus-4-8", Prices::new(15.0, 75.0, 18.75, 1.50), false),
    ("claude-opus-5", Prices::new(15.0, 75.0, 18.75, 1.50), false),
    ("claude-opus", Prices::new(15.0, 75.0, 18.75, 1.50), false),
    // Sonnet tier.
    ("claude-sonnet-4-5", Prices::new(3.0, 15.0, 3.75, 0.30), true),
    ("claude-sonnet-4-6", Prices::new(3.0, 15.0, 3.75, 0.30), false),
    ("claude-sonnet-5", Prices::new(3.0, 15.0, 3.75, 0.30), false),
    ("claude-sonnet", Prices::new(3.0, 15.0, 3.75, 0.30), false),
    // Fictional-in-listing family names.
    ("claude-fable", Prices::new(15.0, 75.0, 18.75, 1.50), false),
    // ---- OpenAI (input / output / — / cached-input) ----------------------------------------
    ("gpt-4o-mini", Prices::new(0.15, 0.60, 0.0, 0.075), true),
    ("gpt-4.1-mini", Prices::new(0.40, 1.60, 0.0, 0.10), true),
    ("gpt-4.1-nano", Prices::new(0.10, 0.40, 0.0, 0.025), true),
    ("gpt-4o", Prices::new(2.50, 10.0, 0.0, 1.25), true),
    ("gpt-4.1", Prices::new(2.00, 8.0, 0.0, 0.50), true),
    ("o4-mini", Prices::new(1.10, 4.40, 0.0, 0.275), true),
    ("o3-mini", Prices::new(1.10, 4.40, 0.0, 0.55), true),
    ("o3", Prices::new(2.00, 8.0, 0.0, 0.50), true),
    // Newer flagships in this account's listing — list prices unconfirmed; over-estimate.
    ("gpt-5.4-mini", Prices::new(0.50, 2.0, 0.0, 0.125), false),
    ("gpt-5.4", Prices::new(3.0, 12.0, 0.0, 0.75), false),
    ("gpt-6-astra", Prices::new(15.0, 60.0, 0.0, 1.875), false),
    ("gpt-6", Prices::new(15.0, 60.0, 0.0, 1.875), false),
];

/// Read a flattened usage counter as u64 (0 if absent or non-numeric).
fn u(usage: &BTreeMap<String, Value>, key: &str) -> u64 {
    usage.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// Estimate the USD cost of one capture from its `model` and flattened `usage` map.
///
/// Provider-aware: Anthropic's `input_tokens` excludes cached tokens (the cache buckets are added);
/// OpenAI's `prompt_tokens` / `input_tokens` *include* cached tokens (the cached share is re-priced
/// at the cache-read rate and subtracted from the full-price input).
pub fn estimate_usd(model: &str, usage: &BTreeMap<String, Value>) -> f64 {
    if usage.is_empty() {
        return 0.0;
    }
    let (p, _) = prices_for(model);
    let per = |tokens: u64, rate: f64| (tokens as f64) * rate / 1_000_000.0;

    if provider_for(model) == "anthropic" {
        let input = u(usage, "input_tokens");
        let output = u(usage, "output_tokens");
        let cw = u(usage, "cache_creation_input_tokens");
        let cr = u(usage, "cache_read_input_tokens");
        per(input, p.input) + per(output, p.output) + per(cw, p.cache_write) + per(cr, p.cache_read)
    } else {
        // OpenAI Chat uses prompt/completion; Responses uses input/output. Cached tokens live in a
        // `*_details.cached_tokens` bucket and are already inside the prompt/input total.
        let prompt = u(usage, "prompt_tokens").max(u(usage, "input_tokens"));
        let completion = u(usage, "completion_tokens").max(u(usage, "output_tokens"));
        let cached = u(usage, "prompt_tokens_details.cached_tokens")
            .max(u(usage, "input_tokens_details.cached_tokens"));
        let uncached = prompt.saturating_sub(cached);
        per(uncached, p.input) + per(cached, p.cache_read) + per(completion, p.output)
    }
}

/// A per-provider spend accumulator over a run.
#[derive(Debug, Clone, Default)]
pub struct SpendTally {
    /// provider → (usd, number of billed captures).
    pub by_provider: BTreeMap<String, (f64, usize)>,
    /// Whether any billed capture used a non-verified (over-estimated) price.
    pub used_fallback: bool,
}

impl SpendTally {
    /// Fold one capture into the tally.
    pub fn add(&mut self, model: &str, usage: &BTreeMap<String, Value>) {
        let usd = estimate_usd(model, usage);
        if usd == 0.0 && usage.is_empty() {
            return;
        }
        let (_, verified) = prices_for(model);
        if !verified {
            self.used_fallback = true;
        }
        let e = self.by_provider.entry(provider_for(model).to_string()).or_default();
        e.0 += usd;
        e.1 += 1;
    }

    /// Total across all providers.
    pub fn total(&self) -> f64 {
        self.by_provider.values().map(|(u, _)| u).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn usage(pairs: &[(&str, u64)]) -> BTreeMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), json!(v))).collect()
    }

    #[test]
    fn anthropic_input_output_priced_at_family_rate() {
        // sonnet: 3 / 15 per 1M.
        let u = usage(&[("input_tokens", 1_000_000), ("output_tokens", 1_000_000)]);
        let cost = estimate_usd("claude-sonnet-5", &u);
        assert!((cost - 18.0).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn anthropic_cache_buckets_are_additive() {
        // opus: input 15, cache_read 1.5. 1M cached-read tokens = $1.50.
        let u = usage(&[("cache_read_input_tokens", 1_000_000)]);
        let cost = estimate_usd("claude-opus-4-8", &u);
        assert!((cost - 1.50).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn openai_cached_share_repriced_and_subtracted() {
        // gpt-4o-mini: input 0.15, cached 0.075, output 0.60. prompt 1M incl 1M cached => all cached.
        let u = usage(&[
            ("prompt_tokens", 1_000_000),
            ("prompt_tokens_details.cached_tokens", 1_000_000),
            ("completion_tokens", 0),
        ]);
        let cost = estimate_usd("gpt-4o-mini", &u);
        assert!((cost - 0.075).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn responses_input_output_keys() {
        // gpt-4o: input 2.5, output 10. 0.5M in + 0.1M out.
        let u = usage(&[("input_tokens", 500_000), ("output_tokens", 100_000)]);
        let cost = estimate_usd("gpt-4o", &u);
        assert!((cost - (1.25 + 1.0)).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn unknown_model_uses_over_estimating_fallback() {
        let (_, verified) = prices_for("claude-quasar-9");
        assert!(!verified);
        let (_, v2) = prices_for("gpt-9-nebula");
        assert!(!v2);
        // fallback is the priciest tier, non-zero.
        let u = usage(&[("input_tokens", 1_000_000)]);
        assert!(estimate_usd("claude-quasar-9", &u) >= 15.0);
    }

    #[test]
    fn provider_split() {
        assert_eq!(provider_for("claude-sonnet-5"), "anthropic");
        assert_eq!(provider_for("gpt-4o-mini"), "openai");
        assert_eq!(provider_for("o3"), "openai");
    }

    #[test]
    fn tally_folds_by_provider() {
        let mut t = SpendTally::default();
        t.add("claude-sonnet-5", &usage(&[("input_tokens", 1_000_000)]));
        t.add("gpt-4o-mini", &usage(&[("prompt_tokens", 1_000_000), ("completion_tokens", 0)]));
        assert_eq!(t.by_provider.len(), 2);
        assert_eq!(t.by_provider["anthropic"].1, 1);
        assert!(t.used_fallback); // sonnet-5 is best-knowledge
    }
}
