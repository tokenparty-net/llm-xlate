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

/// The `prompt_tokens_details` key naming the **cache-write** counter, in precedence order.
///
/// Cache writes are a billed prompt category with no standard place in the Chat dialect, so
/// each server that reports them invented a spelling. Only spellings observed in real traffic
/// are listed; an unrecognized counter is preserved into `usage.ext`, never guessed into a
/// typed slot (plan L4).
///
/// - `cache_creation_tokens` — the spelling used by OpenAI-compatible bridges in front of an
///   Anthropic backend, which have a genuine cache-write figure to report and no standard field
///   to put it in. It mirrors Anthropic's own `cache_creation_input_tokens`, and it is the one
///   [`encode_usage`] emits: OpenAI Chat reports no cache writes at all (verified live against
///   `gpt-4o-mini`, 2026-09-12), so there is no native spelling to prefer.
/// - `created_cache_tokens` — vLLM (observed on Kimi K3 behind vLLM 0.27.x).
/// - `cache_write_tokens` — OpenAI's spelling on the Responses dialect, accepted here in case
///   a compatible server carries it over to Chat.
const CACHE_WRITE_KEYS: &[&str] =
    &["cache_creation_tokens", "created_cache_tokens", "cache_write_tokens"];

/// The `prompt_tokens_details` key naming the 1-hour-TTL subset of the cache writes.
const CACHE_WRITE_1H_KEYS: &[&str] = &["cache_creation_1h_tokens"];

/// `usage` root keys this codec maps to typed counters or recomputes; everything else at the
/// root is preserved verbatim into `usage.ext` under `chat.<key>`.
const KNOWN_ROOT_KEYS: &[&str] = &[
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
    // Anthropic-shaped cache keys, which some OpenAI-compatible servers emit alongside the
    // OpenAI ones. Read as a fallback, so never also duplicated into `ext`.
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation",
];

/// Ext-key prefix for a counter preserved out of `prompt_tokens_details`.
const PROMPT_DETAILS_NS: &str = "chat.prompt_tokens_details.";
/// Ext-key prefix for a counter preserved out of `completion_tokens_details`.
const COMPLETION_DETAILS_NS: &str = "chat.completion_tokens_details.";

fn opt_u32(v: Option<&Value>) -> Option<u32> {
    v.and_then(Value::as_u64).map(|n| n as u32)
}

/// First present key out of `keys`, read as a `u32`.
fn first_u32(obj: Option<&Map<String, Value>>, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|k| opt_u32(obj.and_then(|o| o.get(*k))))
}

/// Decode a Chat `usage` object into an IR [`Usage`].
///
/// `prompt_tokens` is the **gross** prompt (it counts the cached and written portions inside
/// it), so it is reduced to the IR's fresh `input` via [`Usage::from_gross`]. `prompt_tokens`
/// is treated as gross even when Anthropic-shaped cache keys coexist in the same object — some
/// OpenAI-compatible servers emit both conventions at once, and only the OpenAI spelling
/// tells us which meaning the number carries (plan L3).
///
/// `completion_tokens`→output, `prompt_tokens_details.cached_tokens`→cache_read, the
/// cache-write spellings in [`CACHE_WRITE_KEYS`]→cache_write,
/// `completion_tokens_details.reasoning_tokens`→reasoning. Every other key, at the root or in
/// either details object, lands in `usage.ext` under its dotted path so nothing is lost.
pub fn decode_usage(v: &Value) -> Usage {
    let obj = match v.as_object() {
        Some(o) => o,
        None => return Usage::default(),
    };
    let ptd = obj.get("prompt_tokens_details").and_then(Value::as_object);
    let ctd = obj.get("completion_tokens_details").and_then(Value::as_object);

    let cache_read = opt_u32(ptd.and_then(|d| d.get("cached_tokens")))
        .or_else(|| opt_u32(obj.get("cache_read_input_tokens")));
    let cache_write = first_u32(ptd, CACHE_WRITE_KEYS)
        .or_else(|| opt_u32(obj.get("cache_creation_input_tokens")));
    let cache_write_1h = first_u32(ptd, CACHE_WRITE_1H_KEYS).or_else(|| {
        opt_u32(
            obj.get("cache_creation")
                .and_then(Value::as_object)
                .and_then(|c| c.get("ephemeral_1h_input_tokens")),
        )
    });

    let mut usage = Usage::from_gross(
        u32_of(obj.get("prompt_tokens")),
        u32_of(obj.get("completion_tokens")),
        cache_read,
        cache_write,
        cache_write_1h,
    );
    usage.reasoning = opt_u32(ctd.and_then(|d| d.get("reasoning_tokens")));

    // Preserve everything this codec did not map, keyed by its path within the usage object.
    for (k, val) in obj {
        if !KNOWN_ROOT_KEYS.contains(&k.as_str()) && !val.is_null() {
            usage.ext.insert(format!("chat.{k}"), val.clone());
        }
    }
    if let Some(d) = ptd {
        let consumed = |k: &str| {
            k == "cached_tokens" || CACHE_WRITE_KEYS.contains(&k) || CACHE_WRITE_1H_KEYS.contains(&k)
        };
        for (k, val) in d {
            if !consumed(k) && !val.is_null() {
                usage.ext.insert(format!("{PROMPT_DETAILS_NS}{k}"), val.clone());
            }
        }
    }
    if let Some(d) = ctd {
        for (k, val) in d {
            if k != "reasoning_tokens" && !val.is_null() {
                usage.ext.insert(format!("{COMPLETION_DETAILS_NS}{k}"), val.clone());
            }
        }
    }
    usage
}

/// Encode an IR [`Usage`] into a Chat `usage` object (fixed key order).
///
/// `prompt_tokens` is re-grossed from the IR's disjoint counters
/// ([`Usage::gross_prompt`]) and `total_tokens` is that plus `output`, so a client's prompt
/// count always contains its own cached and written portions — the Chat dialect's convention.
///
/// `prompt_tokens_details` carries `cached_tokens`, and — when the upstream reported them —
/// the non-standard `cache_creation_tokens` / `cache_creation_1h_tokens`. Cache writes are
/// billed prompt tokens, so surfacing them under a non-standard key is strictly better than
/// dropping them; OpenAI SDKs ignore unknown fields. Counters that [`decode_usage`] preserved
/// into `usage.ext` are re-emitted into the object they came from, so a Chat response round
/// trip is faithful.
pub fn encode_usage(usage: &Usage) -> Value {
    let gross = usage.gross_prompt();
    let mut obj = Map::new();
    obj.insert("prompt_tokens".into(), Value::from(gross));
    obj.insert("completion_tokens".into(), Value::from(usage.output));
    obj.insert("total_tokens".into(), Value::from(gross + usage.output));

    let has_prompt_ext = usage.ext.iter().any(|(k, _)| k.starts_with(PROMPT_DETAILS_NS));
    if usage.cache_read.is_some() || usage.cache_write.is_some() || has_prompt_ext {
        let mut ptd = Map::new();
        ptd.insert("cached_tokens".into(), Value::from(usage.cache_read.unwrap_or(0)));
        if let Some(w) = usage.cache_write {
            ptd.insert("cache_creation_tokens".into(), Value::from(w));
        }
        if let Some(w) = usage.cache_write_1h {
            ptd.insert("cache_creation_1h_tokens".into(), Value::from(w));
        }
        merge_preserved(&mut ptd, usage, PROMPT_DETAILS_NS);
        obj.insert("prompt_tokens_details".into(), Value::Object(ptd));
    }

    let has_completion_ext = usage.ext.iter().any(|(k, _)| k.starts_with(COMPLETION_DETAILS_NS));
    if usage.reasoning.is_some() || has_completion_ext {
        let mut ctd = Map::new();
        if let Some(r) = usage.reasoning {
            ctd.insert("reasoning_tokens".into(), Value::from(r));
        }
        merge_preserved(&mut ctd, usage, COMPLETION_DETAILS_NS);
        obj.insert("completion_tokens_details".into(), Value::Object(ctd));
    }

    // Root-level counters this codec does not model (a `cost_usd` some gateways attach, say),
    // re-emitted after the standard fields. Nested namespaces are excluded by the `.` in their
    // prefixes, which a bare root key never contains.
    for (k, val) in usage.ext.iter() {
        if let Some(name) = k.strip_prefix("chat.") {
            if !name.contains('.') {
                obj.entry(name.to_string()).or_insert_with(|| val.clone());
            }
        }
    }
    Value::Object(obj)
}

/// Copy the `usage.ext` entries under `ns` back into the details object they were decoded
/// from, without overwriting a counter this codec already wrote.
fn merge_preserved(dst: &mut Map<String, Value>, usage: &Usage, ns: &str) {
    for (k, val) in usage.ext.iter() {
        if let Some(name) = k.strip_prefix(ns) {
            dst.entry(name.to_string()).or_insert_with(|| val.clone());
        }
    }
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
