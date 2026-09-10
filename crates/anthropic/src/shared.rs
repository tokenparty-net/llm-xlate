//! Cross-cutting helpers shared by the encode, response, and streaming paths: usage mapping
//! in both directions, stop-reason mapping, and opaque-blob sealing at the client boundary.

use serde_json::Value;

use llm_xlate_core::{
    EncodeCtx, OpaqueBlob, StopReason, Usage,
};

use crate::wire::{ext_key, omap, FAMILY};

/// Anthropic `usage` keys that map to typed [`Usage`] counters (everything else is preserved
/// into `usage.ext`).
const KNOWN_USAGE_KEYS: &[&str] = &[
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation",
];

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Decode an Anthropic `usage` object into an IR [`Usage`]. Missing / absent fields default to
/// `0` (for `input`/`output`) or `None` (cache/reasoning counters). Unknown usage keys are
/// preserved into `usage.ext` under the `anthropic.` namespace.
pub fn usage_from_wire(v: &Value) -> Usage {
    let obj = v.as_object();
    let get_u32 = |k: &str| -> Option<u32> {
        obj.and_then(|o| o.get(k)).and_then(Value::as_u64).map(|n| n as u32)
    };
    let mut usage = Usage::new(get_u32("input_tokens").unwrap_or(0), get_u32("output_tokens").unwrap_or(0));
    usage.cache_read = get_u32("cache_read_input_tokens");

    // cache_creation split (ephemeral_5m / ephemeral_1h) takes precedence over the flat
    // cache_creation_input_tokens total.
    let creation = obj.and_then(|o| o.get("cache_creation")).and_then(Value::as_object);
    if let Some(c) = creation {
        usage.cache_write_5m =
            c.get("ephemeral_5m_input_tokens").and_then(Value::as_u64).map(|n| n as u32);
        usage.cache_write_1h =
            c.get("ephemeral_1h_input_tokens").and_then(Value::as_u64).map(|n| n as u32);
    }
    if usage.cache_write_5m.is_none() && usage.cache_write_1h.is_none() {
        usage.cache_write_5m = get_u32("cache_creation_input_tokens");
    }

    // Preserve any remaining usage fields (e.g. `service_tier`, `server_tool_use`) into
    // `usage.ext` under the `anthropic.` namespace so they survive the round trip.
    if let Some(o) = obj {
        for (k, v) in o {
            if !KNOWN_USAGE_KEYS.contains(&k.as_str()) && !v.is_null() {
                usage.ext.insert(ext_key(k), v.clone());
            }
        }
    }
    usage
}

/// Encode an IR [`Usage`] as an Anthropic `usage` object with a fixed field order.
pub fn usage_to_wire(u: &Usage) -> Value {
    let mut m = omap();
    m.insert("input_tokens".into(), Value::from(u.input));
    if let Some(cr) = u.cache_read {
        m.insert("cache_read_input_tokens".into(), Value::from(cr));
    }
    let creation_total = u.cache_write_5m.unwrap_or(0) + u.cache_write_1h.unwrap_or(0);
    if u.cache_write_5m.is_some() || u.cache_write_1h.is_some() {
        m.insert("cache_creation_input_tokens".into(), Value::from(creation_total));
        let mut cc = omap();
        cc.insert("ephemeral_5m_input_tokens".into(), Value::from(u.cache_write_5m.unwrap_or(0)));
        cc.insert("ephemeral_1h_input_tokens".into(), Value::from(u.cache_write_1h.unwrap_or(0)));
        m.insert("cache_creation".into(), Value::Object(cc));
    }
    m.insert("output_tokens".into(), Value::from(u.output));
    Value::Object(m)
}

// ---------------------------------------------------------------------------
// Stop reason
// ---------------------------------------------------------------------------

/// Map an Anthropic `stop_reason` (+ `stop_sequence`) to an IR [`StopReason`].
pub fn stop_reason_from_wire(reason: Option<&str>, stop_sequence: Option<&str>) -> StopReason {
    match reason {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence(stop_sequence.unwrap_or("").to_string()),
        Some("tool_use") => StopReason::ToolUse,
        Some("pause_turn") => StopReason::PauseTurn,
        Some("refusal") => StopReason::Refusal,
        _ => StopReason::EndTurn,
    }
}

/// Map an IR [`StopReason`] to an Anthropic `stop_reason` string, plus the optional
/// `stop_sequence` payload.
pub fn stop_reason_to_wire(reason: &StopReason) -> (&'static str, Option<String>) {
    match reason {
        StopReason::EndTurn => ("end_turn", None),
        StopReason::MaxTokens => ("max_tokens", None),
        StopReason::StopSequence(s) => ("stop_sequence", Some(s.clone())),
        StopReason::ToolUse => ("tool_use", None),
        StopReason::ContentFilter => ("refusal", None),
        StopReason::Refusal => ("refusal", None),
        StopReason::PauseTurn => ("pause_turn", None),
        StopReason::Cancelled => ("end_turn", None),
    }
}

// ---------------------------------------------------------------------------
// Opaque sealing (client-facing boundary)
// ---------------------------------------------------------------------------

/// Produce the string to write into an Anthropic replay carrier for a reasoning/redacted
/// blob: a same-family (Anthropic) blob is written **verbatim** (native signature / redacted
/// data), a foreign-family blob is **sealed** into an `rtr1.` envelope so the client can
/// replay it later. This matches the crate's envelope boundary (`core::codec` docs).
pub fn seal_or_native(ctx: &EncodeCtx, blob: &OpaqueBlob) -> String {
    if blob.family == FAMILY {
        blob.data.clone()
    } else {
        ctx.sealer.seal(blob)
    }
}
