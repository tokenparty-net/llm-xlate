//! Pass 3: provider-hosted tools (plan §7.4).
//!
//! A foreign-family [`ToolDef::Provider`] in the *current* request is rejected (default) or
//! dropped (config). Foreign provider tool call/result **items** in history are folded into
//! plain text messages so a failover backend can still read the transcript. Same-family
//! provider tools and items pass through untouched.

use llm_xlate_core::canon;
use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::codec::{ForeignProviderTool, TranslatorConfig};
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, Item, Part, Protocol, Role, ToolChoice, ToolDef};
use llm_xlate_core::wrap;
use serde_json::Value;

pub(crate) fn run(
    req: &mut IrRequest,
    _caps: &Capabilities,
    target: Protocol,
    cfg: &TranslatorConfig,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    let target_family = target.family();

    // ── current-request provider tools ───────────────────────────────────────────────────
    let mut dropped_names: Vec<String> = Vec::new();
    let mut kept_tools = Vec::with_capacity(req.tools.len());
    for tool in std::mem::take(&mut req.tools) {
        match &tool {
            ToolDef::Provider(item) if item.family != target_family => {
                if cfg.foreign_provider_tool == ForeignProviderTool::Unsupported {
                    return Err(XlateError::unsupported(
                        "tools",
                        format!(
                            "provider-hosted tool of family {} is not supported by the {} backend",
                            item.family, target_family
                        ),
                    ));
                }
                if let Some(name) = tool_name(&item.raw) {
                    dropped_names.push(name);
                }
                degr.dropped(
                    "tools",
                    format!("foreign provider tool ({}) dropped", item.family),
                );
            }
            _ => kept_tools.push(tool),
        }
    }
    req.tools = kept_tools;

    // A ToolChoice::Named referencing a dropped provider tool degrades to Auto.
    if let ToolChoice::Named(n) = &req.tool_choice {
        if dropped_names.iter().any(|d| d == n) {
            req.tool_choice = ToolChoice::Auto;
            degr.rewritten(
                "tool_choice",
                "named tool was a dropped provider tool; using auto",
            );
        }
    }

    // ── history provider items → fold foreign ones to text ───────────────────────────────
    let old = std::mem::take(&mut req.items);
    let mut new_items = Vec::with_capacity(old.len());
    for item in old {
        match item {
            Item::ProviderToolCall(op) if op.family != target_family => {
                let name = tool_name(&op.raw).unwrap_or_else(|| "provider_tool".to_string());
                let text = wrap::provider_tool_fold(&name, &canon::to_string(&op.raw));
                new_items.push(Item::Message {
                    role: Role::Assistant,
                    content: vec![Part::text(text)],
                    id: None,
                });
                degr.folded(
                    "items",
                    format!("foreign provider tool call ({}) folded to text", op.family),
                );
            }
            Item::ProviderToolResult(op) if op.family != target_family => {
                let name = tool_name(&op.raw).unwrap_or_else(|| "provider_tool".to_string());
                let text = wrap::provider_tool_fold(&name, &extract_result_text(&op.raw));
                new_items.push(Item::Message {
                    role: Role::User,
                    content: vec![Part::text(text)],
                    id: None,
                });
                degr.folded(
                    "items",
                    format!("foreign provider tool result ({}) folded to text", op.family),
                );
            }
            other => new_items.push(other),
        }
    }
    req.items = new_items;

    Ok(())
}

/// Best-effort tool name from a provider item's raw JSON: the `name` field, else `type`.
fn tool_name(raw: &Value) -> Option<String> {
    raw.get("name")
        .or_else(|| raw.get("type"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract human-readable text from a provider tool result's raw JSON: the concatenated string
/// values reachable under `text` / `output` / `content` fields; else the compact JSON.
fn extract_result_text(raw: &Value) -> String {
    let mut out = String::new();
    if let Value::Object(map) = raw {
        for key in ["text", "output", "content"] {
            if let Some(v) = map.get(key) {
                collect_strings(v, &mut out);
            }
        }
    }
    if out.is_empty() {
        canon::to_string(raw)
    } else {
        out
    }
}

/// Recursively append string values reachable under `text` / `output` / `content` keys.
fn collect_strings(v: &Value, out: &mut String) {
    match v {
        Value::String(s) => out.push_str(s),
        Value::Array(a) => {
            for e in a {
                collect_strings(e, out);
            }
        }
        Value::Object(m) => {
            for key in ["text", "output", "content"] {
                if let Some(x) = m.get(key) {
                    collect_strings(x, out);
                }
            }
        }
        _ => {}
    }
}
