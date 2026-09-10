//! Pass 6: function tools (plan §7.4).
//!
//! Rejects function tools the backend cannot accept, normalizes tool parameter schemas,
//! downgrades unsupported `strict`, caps the tool count, degrades an unsupported `tool_choice`
//! variant to `auto`, clears an unsupported `parallel_tool_calls`, and validates tool-call ids
//! against the backend's id pattern (ids are preserved verbatim, never rewritten).

use llm_xlate_core::canon;
use llm_xlate_core::caps::{Capabilities, ToolChoiceKind};
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, Item, ToolChoice, ToolDef};
use regex::Regex;

pub(crate) fn run(
    req: &mut IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    let has_function_tools = req.tools.iter().any(|t| matches!(t, ToolDef::Function { .. }));

    // (a) Function tools supported at all.
    if has_function_tools && caps.tools.function_tools.is_no_or_unknown() {
        return Err(XlateError::unsupported(
            "tools",
            "the backend does not support function tools",
        ));
    }

    // (b) Tool count cap.
    if let Some(max) = caps.tools.max_tools {
        let n = req.tools.len() as u32;
        if n > max {
            return Err(XlateError::unsupported(
                "tools",
                format!("too many tools ({n} > max_tools {max})"),
            ));
        }
    }

    // (c) Per-tool schema normalization + strict downgrade.
    let kws = caps.tools.schema_unsupported_keywords.as_ref();
    let strict_supported = caps.tools.strict.supported;
    for tool in &mut req.tools {
        if let ToolDef::Function { name, parameters, strict, .. } = tool {
            if let Some(kws) = kws {
                if !kws.is_empty() {
                    let before = parameters.clone();
                    canon::remove_keywords_recursive(parameters, kws);
                    if *parameters != before {
                        degr.rewritten(
                            format!("tools.{name}.parameters"),
                            "removed schema keywords the backend does not support",
                        );
                    }
                }
            }
            if *strict == Some(true) && strict_supported.is_no_or_unknown() {
                *strict = Some(false);
                degr.downgraded(
                    format!("tools.{name}.strict"),
                    "strict tool schema not supported; downgraded",
                );
            }
            // OpenAI strict also requires every property to appear in `required`. A strict schema
            // with optional properties is invalid; rather than silently promote those properties to
            // required (a semantic change) or emit an invalid strict schema, downgrade `strict`.
            if *strict == Some(true)
                && !canon::strict_schema_required_covers_properties(parameters)
            {
                *strict = Some(false);
                degr.downgraded(
                    format!("tools.{name}.strict"),
                    "strict function schema requires every property in `required`; downgraded",
                );
            }
            // A strict function schema must carry `additionalProperties: false` on every object
            // node — OpenAI strict rejects it otherwise ("Invalid schema for function ...";
            // live-verified 2026-09-10, translate1 full responses->chat).
            if *strict == Some(true) {
                let before = parameters.clone();
                canon::ensure_additional_properties_false_recursive(parameters);
                if *parameters != before {
                    degr.rewritten(
                        format!("tools.{name}.parameters"),
                        "added additionalProperties:false for strict function schema",
                    );
                }
            }
        }
    }

    // (d) tool_choice variant support.
    if !matches!(req.tool_choice, ToolChoice::Auto) {
        let kind = match &req.tool_choice {
            ToolChoice::Auto => ToolChoiceKind::Auto,
            ToolChoice::None => ToolChoiceKind::None,
            ToolChoice::Required => ToolChoiceKind::Required,
            ToolChoice::Named(_) => ToolChoiceKind::Named,
        };
        let allowed = caps.tools.tool_choice.as_ref().is_some_and(|l| l.contains(&kind));
        if !allowed {
            req.tool_choice = ToolChoice::Auto;
            degr.rewritten("tool_choice", "tool_choice variant not supported; using auto");
        }
    }

    // (e) parallel_tool_calls control.
    if req.parallel_tool_calls.is_some() && caps.tools.parallel_control.is_no_or_unknown() {
        req.parallel_tool_calls = None;
        degr.dropped("parallel_tool_calls", "parallel tool-call control not supported; cleared");
    }

    // (f) Tool-call id pattern (ids preserved verbatim; a mismatch is incompatible history).
    if let Some(pattern) = &caps.tools.id_pattern {
        // Anchor the pattern so the whole id must match — an unanchored `is_match` would accept
        // any id merely *containing* a conforming substring (e.g. `"bad id call_1 !!!"`).
        let re = Regex::new(&format!("^(?:{pattern})$")).map_err(|e| {
            XlateError::invalid_request(format!("backend id_pattern is not a valid regex: {e}"))
        })?;
        for item in &req.items {
            let call_id = match item {
                Item::ToolCall { call_id, .. } | Item::ToolResult { call_id, .. } => call_id,
                _ => continue,
            };
            if !re.is_match(call_id.as_str()) {
                return Err(XlateError::incompatible_history(format!(
                    "tool-call id {:?} does not match the backend id pattern and cannot be \
                     rewritten",
                    call_id.as_str()
                )));
            }
        }
    }

    Ok(())
}
