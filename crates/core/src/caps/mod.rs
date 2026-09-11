//! The capability registry (plan §6): three layers — provider-family `[defaults]`, per-model
//! overrides (first match wins), and a runtime backend overlay — merged by [`Registry::resolve`].
//!
//! `match` patterns are treated as **anchored globs** (`*` = any run, `?` = one char, `|` =
//! alternation), which is how the plan's example patterns (`claude-opus-4-8*|claude-*-5*`)
//! are written. This is a deliberate reading of the plan's "regex; first match wins": the
//! shipped patterns are glob-shaped, and glob semantics avoid a bare `*` being read as a
//! regex quantifier.

mod schema;

pub use schema::{
    BackendOverrides, BreakpointRule, BudgetRule, BudgetStatus, Capabilities, CacheCap,
    CompactionMode, ErrorsCap, ExposureMode, InstructionsCap, LimitsCap, MaxRule, MediaCap,
    MediaKind, MediaSourceKind, OutputCap, OutputFormatCap, ReasoningCap, ReasoningMode,
    ReplayMode, ResultContentKind, SamplingCap, SamplingRule, SessionCap, SessionSink, StateCap,
    Streaming, StrictDefault, StrictRule, ToolChoiceKind, ToolsCap, TopLevelSystem, TransportCap,
    Tri, UsageTiming, MidConversationSystem,
};

use std::collections::BTreeMap;
use std::sync::OnceLock;

use regex::Regex;
use serde::Deserialize;

use crate::error::XlateError;
use crate::ir::ProviderFamily;

/// A compiled per-model rule.
struct ModelRule {
    regex: Regex,
    caps: Capabilities,
}

/// All rules for one provider family.
struct FamilyBlock {
    family: ProviderFamily,
    defaults: Capabilities,
    models: Vec<ModelRule>,
}

/// A capability registry: family blocks plus named backend overlays.
#[derive(Default)]
pub struct Registry {
    families: Vec<FamilyBlock>,
    backends: BTreeMap<String, Capabilities>,
}

#[derive(Deserialize, Default)]
struct RawFile {
    family: Option<ProviderFamily>,
    #[serde(default)]
    defaults: Capabilities,
    #[serde(default)]
    model: Vec<RawModel>,
    #[serde(default)]
    backend: BTreeMap<String, Capabilities>,
}

#[derive(Deserialize)]
struct RawModel {
    #[serde(rename = "match")]
    match_: String,
    #[serde(flatten)]
    caps: Capabilities,
}

/// Convert a `match` glob into an anchored regex source.
fn glob_to_regex(pattern: &str) -> String {
    let mut alts = Vec::new();
    for alt in pattern.split('|') {
        let mut r = String::from("^");
        for ch in alt.chars() {
            match ch {
                '*' => r.push_str(".*"),
                '?' => r.push('.'),
                // Escape regex metacharacters so a literal model name is matched literally.
                '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '\\' => {
                    r.push('\\');
                    r.push(ch);
                }
                c => r.push(c),
            }
        }
        r.push('$');
        alts.push(r);
    }
    alts.join("|")
}

impl Registry {
    /// Parse one TOML file into a registry.
    pub fn from_toml(src: &str) -> Result<Registry, XlateError> {
        let raw: RawFile = toml::from_str(src)
            .map_err(|e| XlateError::invalid_request(format!("caps TOML parse error: {e}")))?;
        let family = raw.family.unwrap_or(ProviderFamily::Other("unknown".to_string()));

        let mut models = Vec::with_capacity(raw.model.len());
        for m in raw.model {
            let regex = Regex::new(&glob_to_regex(&m.match_)).map_err(|e| {
                XlateError::invalid_request(format!("invalid model match `{}`: {e}", m.match_))
            })?;
            models.push(ModelRule { regex, caps: m.caps });
        }

        Ok(Registry {
            families: vec![FamilyBlock { family, defaults: raw.defaults, models }],
            backends: raw.backend,
        })
    }

    /// Merge another registry into this one. Same-family blocks combine (defaults overlaid,
    /// this registry's model rules kept ahead of `other`'s so they win ties); named backends
    /// from `other` are added (this registry wins on a name collision).
    pub fn merge(&mut self, mut other: Registry) {
        for ob in other.families.drain(..) {
            if let Some(existing) = self.families.iter_mut().find(|b| b.family == ob.family) {
                existing.defaults.overlay(&ob.defaults);
                existing.models.extend(ob.models);
            } else {
                self.families.push(ob);
            }
        }
        for (k, v) in other.backends {
            self.backends.entry(k).or_insert(v);
        }
    }

    fn family_block(&self, family: &ProviderFamily) -> Option<&FamilyBlock> {
        self.families.iter().find(|b| &b.family == family)
    }

    /// Look up a named backend overlay parsed from a `[backend."name"]` table.
    pub fn backend(&self, name: &str) -> Option<&BackendOverrides> {
        self.backends.get(name)
    }

    /// Resolve capabilities for a `(family, model)` pair, applying an optional backend
    /// overlay last. Layers: conservative base → family `[defaults]` → first matching
    /// `[[model]]` → `backend`.
    pub fn resolve(
        &self,
        family: &ProviderFamily,
        model: &str,
        backend: Option<&BackendOverrides>,
    ) -> Capabilities {
        let mut caps = Capabilities::unknown();
        if let Some(block) = self.family_block(family) {
            caps.overlay(&block.defaults);
            for rule in &block.models {
                if rule.regex.is_match(model) {
                    caps.overlay(&rule.caps);
                    break;
                }
            }
        }
        if let Some(b) = backend {
            caps.overlay(b);
        }
        caps
    }
}

/// The shipped registry (Anthropic + OpenAI + generic OpenAI-compatible), parsed once.
pub fn shipped() -> &'static Registry {
    static SHIPPED: OnceLock<Registry> = OnceLock::new();
    SHIPPED.get_or_init(|| {
        let mut reg = Registry::from_toml(include_str!("../../../../data/caps/anthropic.toml"))
            .expect("shipped anthropic.toml parses");
        reg.merge(
            Registry::from_toml(include_str!("../../../../data/caps/openai.toml"))
                .expect("shipped openai.toml parses"),
        );
        reg.merge(
            Registry::from_toml(include_str!("../../../../data/caps/openai_compatible.toml"))
                .expect("shipped openai_compatible.toml parses"),
        );
        reg
    })
}

/// Named capability presets over the [`shipped`] registry, for tests and examples.
pub mod preset {
    use super::*;
    use crate::caps::schema::{ReasoningCap, TransportCap};
    use crate::ir::Protocol;

    /// Claude 3.5 Sonnet (budget-mode reasoning, no mid-conversation system).
    pub fn claude_old() -> Capabilities {
        shipped().resolve(&ProviderFamily::Anthropic, "claude-3-5-sonnet-latest", None)
    }
    /// Claude Opus 4.6 (adaptive, budget ignored).
    pub fn claude_46() -> Capabilities {
        shipped().resolve(&ProviderFamily::Anthropic, "claude-opus-4-6", None)
    }
    /// Claude Opus 5 (adaptive, sampling rejected, native mid-conversation system).
    pub fn claude_5() -> Capabilities {
        shipped().resolve(&ProviderFamily::Anthropic, "claude-opus-5", None)
    }
    /// GPT-4o (chat + responses, no reasoning).
    pub fn gpt4o() -> Capabilities {
        shipped().resolve(&ProviderFamily::OpenAI, "gpt-4o", None)
    }
    /// GPT-5.4 accessed as a **Chat** backend: `tools_with_reasoning=false` (plan §2), chat
    /// protocol only. Modeled as the resolved model caps plus a Chat backend overlay.
    pub fn gpt5_chat() -> Capabilities {
        let overlay = Capabilities {
            transport: TransportCap {
                protocols: Some(vec![Protocol::OaiChat]),
                ..Default::default()
            },
            reasoning: ReasoningCap { tools_with_reasoning: Tri::No, ..Default::default() },
            ..Default::default()
        };
        shipped().resolve(&ProviderFamily::OpenAI, "gpt-5.4", Some(&overlay))
    }
    /// GPT-5.4 accessed as a **Responses** backend (tools work with reasoning).
    pub fn gpt5_responses() -> Capabilities {
        shipped().resolve(&ProviderFamily::OpenAI, "gpt-5.4", None)
    }
    /// GPT-6 Astra (`protocols=["responses"]`).
    pub fn gpt6() -> Capabilities {
        shipped().resolve(&ProviderFamily::OpenAI, "gpt-6-astra", None)
    }
    /// Generic OpenAI-compatible model (chat only, most fields Unknown).
    pub fn openai_compatible() -> Capabilities {
        shipped().resolve(&ProviderFamily::Other("openai-compatible".to_string()), "llama-x", None)
    }
}

#[cfg(test)]
mod tests;
