//! Pass 5: structured output (plan §7.3).
//!
//! Rejects a requested output format the backend cannot represent, normalizes a JSON schema by
//! removing unsupported keywords, downgrades `strict` when unsupported, rejects structured
//! output combined with tools when the backend forbids it, and clears an unsupported
//! `verbosity` hint.

use llm_xlate_core::canon;
use llm_xlate_core::caps::{Capabilities, OutputFormatCap};
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, OutputFormat};

pub(crate) fn run(
    req: &mut IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    let fmt = caps.output.format;
    // An object is representable whenever any structured format is supported; a schema needs
    // explicit schema support.
    let supports_object = matches!(
        fmt,
        Some(OutputFormatCap::JsonObject) | Some(OutputFormatCap::JsonSchema) | Some(OutputFormatCap::Both)
    );
    let supports_schema =
        matches!(fmt, Some(OutputFormatCap::JsonSchema) | Some(OutputFormatCap::Both));

    // (a) Format support.
    match &req.output.format {
        OutputFormat::Text => {}
        OutputFormat::JsonObject => {
            if !supports_object {
                return Err(XlateError::unsupported(
                    "response_format",
                    "the backend does not support structured (json_object) output",
                ));
            }
        }
        OutputFormat::JsonSchema { .. } => {
            if !supports_schema {
                return Err(XlateError::unsupported(
                    "response_format",
                    "the backend does not support json_schema structured output",
                ));
            }
        }
    }

    // (b) Structured output together with tools.
    let format_active = !matches!(req.output.format, OutputFormat::Text);
    if format_active
        && !req.tools.is_empty()
        && caps.output.format_with_tools.is_no_or_unknown()
    {
        return Err(XlateError::unsupported(
            "response_format",
            "the backend does not support structured output together with tools",
        ));
    }

    // (c) Schema normalization + strict downgrade (only for json_schema).
    if let OutputFormat::JsonSchema { schema, strict, .. } = &mut req.output.format {
        if let Some(kws) = &caps.output.schema_unsupported_keywords {
            if !kws.is_empty() {
                let before = schema.clone();
                canon::remove_keywords_recursive(schema, kws);
                if *schema != before {
                    degr.rewritten(
                        "output.schema",
                        "removed schema keywords the backend does not support",
                    );
                }
            }
        }
        if *strict && caps.output.strict_supported.is_no_or_unknown() {
            *strict = false;
            degr.downgraded("output.strict", "strict structured output not supported; downgraded");
        }
        // OpenAI strict also requires every property to appear in `required`. A strict schema with
        // optional properties is invalid; rather than silently promote those properties to required
        // (a semantic change) or emit an invalid strict schema, downgrade `strict` to false.
        if *strict && !canon::strict_schema_required_covers_properties(schema) {
            *strict = false;
            degr.downgraded(
                "output.strict",
                "strict structured output requires every property in `required`; downgraded",
            );
        }
        // A schema that stays strict must carry `additionalProperties: false` on every object
        // node — OpenAI strict rejects it otherwise ("'additionalProperties' is required to be
        // supplied and to be false"; live-verified 2026-09-10, translate1 structured_strict).
        if *strict {
            let before = schema.clone();
            canon::ensure_additional_properties_false_recursive(schema);
            if *schema != before {
                degr.rewritten(
                    "output.schema",
                    "added additionalProperties:false for strict structured output",
                );
            }
        }
    }

    // (d) Verbosity.
    if req.output.verbosity.is_some() && caps.output.verbosity.is_no_or_unknown() {
        req.output.verbosity = None;
        degr.dropped("verbosity", "verbosity hint not supported; cleared");
    }

    Ok(())
}
