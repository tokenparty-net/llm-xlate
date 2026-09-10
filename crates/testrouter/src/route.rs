//! Model → `(provider, upstream protocol, upstream model, capabilities)` resolution (plan §4.3).
//!
//! Order of decisions: client model name → `[aliases]` (upstream model + `default` fallback) →
//! the first matching `[[route]]` rule (provider + upstream protocol) → per-request header
//! overrides → capability resolution from the shipped registry (or a named preset). Everything
//! the router decided is captured on the returned [`Route`] so a bad registry entry or a
//! surprising override is visible in the trace.

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::Serialize;

use llm_xlate_core::caps::{preset, shipped, BackendOverrides, Capabilities};
use llm_xlate_core::ir::{Protocol, ProviderFamily};

use crate::config::{Config, Overrides, RouteRule};

/// A fully resolved route for one request (plan §4.3).
#[derive(Debug, Clone)]
pub struct Route {
    /// The provider name (`[providers.<name>]`).
    pub provider: String,
    /// The wire protocol used upstream.
    pub upstream_protocol: Protocol,
    /// The client-facing model name (echoed back to the client).
    pub client_model: String,
    /// The model string sent upstream.
    pub upstream_model: String,
    /// The resolved capability profile.
    pub caps: Capabilities,
    /// Where the capabilities came from: `registry` | `override` | `preset` | `unknown`.
    pub caps_source: String,
    /// The provider family the capabilities were resolved against.
    pub family: ProviderFamily,
}

/// A serializable summary of a route for the trace `route` block (plan §5).
#[derive(Debug, Clone, Serialize)]
pub struct RouteTrace {
    /// Provider name.
    pub provider: String,
    /// Upstream protocol token.
    pub upstream_protocol: String,
    /// Client model name.
    pub client_model: String,
    /// Upstream model name.
    pub upstream_model: String,
    /// Capability source.
    pub caps_source: String,
    /// The full resolved capability profile.
    pub caps: Capabilities,
    /// The header overrides in force.
    pub overrides: Overrides,
}

/// The routing table: compiled route rules plus the aliases, backend overrides, and provider
/// families needed to resolve a request.
pub struct Router {
    routes: Vec<(Regex, RouteRule)>,
    config: Config,
}

impl Router {
    /// Build a router from a validated [`Config`].
    pub fn new(config: Config) -> Result<Router> {
        let routes = config.compiled_routes()?;
        Ok(Router { routes, config })
    }

    /// The provider family a provider name resolves to (`anthropic` / `openai` → the built-in
    /// families; anything else is a custom [`ProviderFamily::Other`] label).
    pub fn provider_family(provider: &str) -> ProviderFamily {
        match provider {
            "anthropic" => ProviderFamily::Anthropic,
            "openai" => ProviderFamily::OpenAI,
            "compat" => ProviderFamily::Other("openai-compatible".to_string()),
            other => ProviderFamily::Other(other.to_string()),
        }
    }

    /// Resolve a request's route (plan §4.3).
    pub fn resolve(
        &self,
        _client_protocol: Protocol,
        client_model: Option<&str>,
        overrides: &Overrides,
    ) -> Result<Route> {
        // Client-facing model name: what the client sent, else the `default` alias.
        let client_model = match client_model {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => self
                .config
                .aliases
                .get("default")
                .cloned()
                .ok_or_else(|| anyhow!("request omitted `model` and no [aliases].default is set"))?,
        };

        // Upstream model: x-xlate-model wins, else the alias mapping (plan §4.3: aliases resolve
        // before route rules), else the client name.
        let upstream_model = overrides
            .model
            .clone()
            .or_else(|| self.config.aliases.get(&client_model).cloned())
            .unwrap_or_else(|| client_model.clone());

        // First matching route rule wins, matched against the resolved upstream model so an alias
        // (`gpt` → `gpt-5.4`) routes as its target does.
        let rule = self.routes.iter().find(|(re, _)| re.is_match(&upstream_model)).map(|(_, r)| r);

        // Provider + upstream protocol: overrides beat the matched rule.
        let provider = overrides
            .provider
            .clone()
            .or_else(|| rule.map(|r| r.provider.clone()))
            .ok_or_else(|| anyhow!("no route matched model `{upstream_model}` and no x-xlate-provider override"))?;
        let upstream_protocol = overrides
            .upstream
            .or_else(|| rule.map(|r| r.upstream))
            .ok_or_else(|| anyhow!("no route matched model `{upstream_model}` and no x-xlate-upstream override"))?;

        let family = Self::provider_family(&provider);

        // Capabilities: a named preset override wins; otherwise the shipped registry (with an
        // optional per-provider backend overlay); an unrecognized model falls back to `unknown`.
        let (caps, caps_source) = if let Some(name) = &overrides.caps {
            match preset_by_name(name) {
                Some(c) => (c, "preset".to_string()),
                None => (Capabilities::unknown(), "unknown".to_string()),
            }
        } else {
            let backend: Option<&BackendOverrides> = self.config.backend_overrides.get(&provider);
            let caps = shipped().resolve(&family, &upstream_model, backend);
            let source = if caps == Capabilities::unknown() {
                "unknown"
            } else if backend.is_some() {
                "override"
            } else {
                "registry"
            };
            (caps, source.to_string())
        };

        // Apply the `x-xlate-expose` override onto the profile-independent request path later;
        // here we only record it on the RouteTrace via `overrides`.

        Ok(Route {
            provider,
            upstream_protocol,
            client_model,
            upstream_model,
            caps,
            caps_source,
            family,
        })
    }

    /// The provider's configured base URL, if any.
    pub fn provider_base_url(&self, provider: &str) -> Option<&str> {
        self.config.providers.get(provider).map(|p| p.base_url.as_str())
    }

    /// Borrow the underlying config.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Build the serializable trace summary for a resolved route.
    pub fn trace(route: &Route, overrides: &Overrides) -> RouteTrace {
        RouteTrace {
            provider: route.provider.clone(),
            upstream_protocol: crate::config::protocol_token_str(route.upstream_protocol).to_string(),
            client_model: route.client_model.clone(),
            upstream_model: route.upstream_model.clone(),
            caps_source: route.caps_source.clone(),
            caps: route.caps.clone(),
            overrides: overrides.clone(),
        }
    }
}

/// Look up a named capability preset (`x-xlate-caps`).
pub fn preset_by_name(name: &str) -> Option<Capabilities> {
    Some(match name {
        "claude_old" => preset::claude_old(),
        "claude_46" => preset::claude_46(),
        "claude_5" => preset::claude_5(),
        "gpt4o" => preset::gpt4o(),
        "gpt5_chat" => preset::gpt5_chat(),
        "gpt5_responses" => preset::gpt5_responses(),
        "gpt6" => preset::gpt6(),
        "openai_compatible" => preset::openai_compatible(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> Router {
        Router::new(Config::example()).unwrap()
    }

    #[test]
    fn claude_routes_to_anthropic() {
        let r = router().resolve(Protocol::Anthropic, Some("claude-sonnet-5"), &Overrides::default()).unwrap();
        assert_eq!(r.provider, "anthropic");
        assert_eq!(r.upstream_protocol, Protocol::Anthropic);
        assert_eq!(r.caps_source, "registry");
    }

    #[test]
    fn gpt_defaults_to_responses_upstream() {
        let r = router().resolve(Protocol::OaiChat, Some("gpt-5.4"), &Overrides::default()).unwrap();
        assert_eq!(r.provider, "openai");
        assert_eq!(r.upstream_protocol, Protocol::OaiResponses);
    }

    #[test]
    fn alias_maps_upstream_model_but_echoes_client_name() {
        let r = router().resolve(Protocol::OaiChat, Some("gpt"), &Overrides::default()).unwrap();
        assert_eq!(r.client_model, "gpt");
        assert_eq!(r.upstream_model, "gpt-5.4");
    }

    #[test]
    fn default_alias_used_when_model_omitted() {
        let r = router().resolve(Protocol::Anthropic, None, &Overrides::default()).unwrap();
        assert_eq!(r.client_model, "claude-sonnet-5");
    }

    #[test]
    fn header_overrides_win() {
        let o = Overrides {
            upstream: Some(Protocol::OaiChat),
            model: Some("gpt-5.4".into()),
            caps: Some("gpt5_chat".into()),
            ..Default::default()
        };
        let r = router().resolve(Protocol::OaiChat, Some("gpt-4o"), &o).unwrap();
        assert_eq!(r.upstream_protocol, Protocol::OaiChat);
        assert_eq!(r.upstream_model, "gpt-5.4");
        assert_eq!(r.caps_source, "preset");
    }

    #[test]
    fn unknown_model_yields_unknown_caps() {
        let o = Overrides { provider: Some("openai".into()), upstream: Some(Protocol::OaiChat), ..Default::default() };
        let r = router().resolve(Protocol::OaiChat, Some("totally-made-up-9000"), &o).unwrap();
        // Registry has an OpenAI family default, so this is `registry`, not `unknown`; assert the
        // genuinely-unknown family path instead.
        let _ = r;
        let o2 = Overrides { provider: Some("mystery".into()), upstream: Some(Protocol::OaiChat), ..Default::default() };
        let r2 = router().resolve(Protocol::OaiChat, Some("zzz"), &o2).unwrap();
        assert_eq!(r2.caps_source, "unknown");
    }
}
