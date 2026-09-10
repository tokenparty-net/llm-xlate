//! Shared helpers used across the Chat codec: the `chat.` extension namespace and its
//! reserved control keys, stop-reason / finish-reason mapping, usage mapping, and small
//! JSON utilities.

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::ir::{Extensions, StopReason, Usage};
use serde_json::{Map, Value};

/// The extension namespace this codec owns.
pub const NS: &str = "chat.";

/// Build a namespaced extension key (`chat.<field>`).
pub fn ns_key(field: &str) -> String {
    format!("{NS}{field}")
}

/// `chat.*` extension keys that are *control* state for this codec (round-trip fidelity
/// markers and side tables), **not** verbatim top-level request fields. They are consumed on
/// encode and must never be written back to the wire as top-level keys.
pub const RESERVED_EXT_KEYS: &[&str] = &[
    "chat.legacy_functions",
    "chat.legacy_function_call",
    "chat.max_tokens_field",
    "chat.include_usage",
    "chat.image_detail",
    "chat.msg_ext",
    "chat.instr_ext",
];

/// Whether a `chat.*` ext key is a reserved control key (not a passthrough top-level field).
pub fn is_reserved_ext(key: &str) -> bool {
    RESERVED_EXT_KEYS.contains(&key)
}

/// Map a Chat `finish_reason` string plus the presence of a `refusal` field to an IR
/// [`StopReason`] (plan §5 stop-reason table).
pub fn finish_reason_to_stop(finish: Option<&str>, has_refusal: bool) -> StopReason {
    match finish {
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some("stop") if has_refusal => StopReason::Refusal,
        _ => StopReason::EndTurn,
    }
}

/// Map an IR [`StopReason`] to a Chat `finish_reason` string (plan §5; `Refusal` → `stop`).
pub fn stop_to_finish_reason(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::MaxTokens => "length",
        StopReason::ToolUse => "tool_calls",
        StopReason::ContentFilter => "content_filter",
        StopReason::EndTurn
        | StopReason::StopSequence(_)
        | StopReason::Refusal
        | StopReason::PauseTurn
        | StopReason::Cancelled => "stop",
    }
}

/// Decode a Chat `usage` object into an IR [`Usage`]:
/// `prompt_tokens`→input, `completion_tokens`→output,
/// `prompt_tokens_details.cached_tokens`→cache_read,
/// `completion_tokens_details.reasoning_tokens`→reasoning. Other detail counters land in
/// `usage.ext` under `chat.*` so nothing is lost.
pub fn decode_usage(v: &Value) -> Usage {
    let mut usage = Usage::default();
    let obj = match v.as_object() {
        Some(o) => o,
        None => return usage,
    };
    usage.input = u32_of(obj.get("prompt_tokens"));
    usage.output = u32_of(obj.get("completion_tokens"));
    if let Some(ptd) = obj.get("prompt_tokens_details").and_then(Value::as_object) {
        if let Some(c) = ptd.get("cached_tokens").and_then(Value::as_u64) {
            usage.cache_read = Some(c as u32);
        }
        if let Some(a) = ptd.get("audio_tokens") {
            usage.ext.insert("chat.prompt_audio_tokens", a.clone());
        }
    }
    if let Some(ctd) = obj.get("completion_tokens_details").and_then(Value::as_object) {
        if let Some(r) = ctd.get("reasoning_tokens").and_then(Value::as_u64) {
            usage.reasoning = Some(r as u32);
        }
        for (k, val) in ctd {
            if k != "reasoning_tokens" {
                usage.ext.insert(format!("chat.completion_{k}"), val.clone());
            }
        }
    }
    usage
}

/// Encode an IR [`Usage`] into a Chat `usage` object (fixed key order). `total_tokens` is
/// computed as `input + output`. `prompt_tokens_details.cached_tokens` and
/// `completion_tokens_details.reasoning_tokens` are emitted when present.
pub fn encode_usage(usage: &Usage) -> Value {
    let mut obj = Map::new();
    obj.insert("prompt_tokens".into(), Value::from(usage.input));
    obj.insert("completion_tokens".into(), Value::from(usage.output));
    obj.insert("total_tokens".into(), Value::from(usage.input + usage.output));
    if usage.cache_read.is_some() || usage.ext.get("chat.prompt_audio_tokens").is_some() {
        let mut ptd = Map::new();
        ptd.insert("cached_tokens".into(), Value::from(usage.cache_read.unwrap_or(0)));
        if let Some(a) = usage.ext.get("chat.prompt_audio_tokens") {
            ptd.insert("audio_tokens".into(), a.clone());
        }
        obj.insert("prompt_tokens_details".into(), Value::Object(ptd));
    }
    // Re-emit completion_tokens_details: reasoning_tokens plus any extra counters that
    // decode_usage stashed into usage.ext under `chat.completion_*` (audio_tokens,
    // accepted_prediction_tokens, rejected_prediction_tokens, …), so a Chat response round
    // trip does not silently drop them.
    let has_completion_ext = usage
        .ext
        .iter()
        .any(|(k, _)| k.starts_with("chat.completion_"));
    if usage.reasoning.is_some() || has_completion_ext {
        let mut ctd = Map::new();
        if let Some(r) = usage.reasoning {
            ctd.insert("reasoning_tokens".into(), Value::from(r));
        }
        for (k, val) in usage.ext.iter() {
            if let Some(name) = k.strip_prefix("chat.completion_") {
                ctd.entry(name.to_string()).or_insert_with(|| val.clone());
            }
        }
        obj.insert("completion_tokens_details".into(), Value::Object(ctd));
    }
    Value::Object(obj)
}

/// Read a `u32` from an optional JSON value (defaulting to 0).
pub fn u32_of(v: Option<&Value>) -> u32 {
    v.and_then(Value::as_u64).unwrap_or(0) as u32
}

/// Whether the resolved capabilities describe a *proper* OpenAI backend (as opposed to a
/// generic `openai-compatible` server). Used only to pick the `max_completion_tokens` vs
/// `max_tokens` field name when the request carries no explicit `chat.max_tokens_field`
/// marker. Keyed off the OpenAI file-id namespace, which the shipped OpenAI data asserts and
/// the conservative `openai-compatible` data leaves unset.
pub fn is_openai_proper(caps: &Capabilities) -> bool {
    caps.media.file_id_namespace.as_deref() == Some("openai")
}

/// Merge response-level extension entries (`stop_details`, `service_tier`, …) into `ext`.
pub fn merge_ext(ext: &mut Extensions, key: &str, value: Value) {
    ext.insert(key.to_string(), value);
}
