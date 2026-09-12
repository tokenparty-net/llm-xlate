//! Non-streaming response path: `decode_response` (Responses `response` object → events) and
//! `encode_response` (aggregated [`IrResponse`] → a `response` object identical to the stream
//! encoder's terminal `response.completed` snapshot).

use bytes::Bytes;
use serde_json::Value;

use llm_xlate_core::{
    canon, Annotation, CallId, Capabilities, Delta, EncodeCtx, Extensions, IrEvent, IrResponse,
    ItemId, ItemKind, OpaqueBlob, OpaqueKind, ProviderFamily, ResponseId, StopReason, Usage,
    XlateError,
};

use crate::render::{build_response_object, render_output_item, status_for, RenderCtx};
use crate::util::{get_str, get_u32};

/// Decode a non-streaming Responses `response` body into IR events.
pub(crate) fn decode_response(body: &[u8], _caps: &Capabilities) -> Result<Vec<IrEvent>, XlateError> {
    let root = canon::parse_upstream(body)?;
    let status = get_str(&root, "status").unwrap_or("completed");

    if status == "failed" {
        let err = root
            .get("error")
            .map(crate::errors::error_from_value)
            .unwrap_or_else(|| XlateError::upstream_malformed("response failed without an error body"));
        return Ok(vec![IrEvent::Error(err)]);
    }

    let response_id = get_str(&root, "id").unwrap_or_default().to_string();
    let model = get_str(&root, "model").unwrap_or_default().to_string();

    let mut events = vec![IrEvent::Start {
        response_id: ResponseId::new(response_id),
        model: model.clone(),
        usage_prefill: None,
    }];

    let empty = Vec::new();
    let output = root.get("output").and_then(Value::as_array).unwrap_or(&empty);
    let mut has_function_call = false;
    for (index, item) in output.iter().enumerate() {
        if item.get("type").and_then(Value::as_str) == Some("function_call") {
            has_function_call = true;
        }
        decode_output_item(item, index as u32, &model, &mut events);
    }

    let usage = decode_usage(root.get("usage"));
    let reason = stop_reason(&root, status, has_function_call);
    events.push(IrEvent::Stop { reason, usage, ext: Extensions::new() });
    Ok(events)
}

/// Emit `ItemStart` + deltas + `ItemStop` for one output item.
fn decode_output_item(item: &Value, index: u32, model: &str, events: &mut Vec<IrEvent>) {
    let ty = item.get("type").and_then(Value::as_str).unwrap_or("");
    let id = item.get("id").and_then(Value::as_str).map(ItemId::new);
    match ty {
        "message" => {
            events.push(IrEvent::ItemStart { index, kind: ItemKind::Message, id, call: None });
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for part in content {
                    match part.get("type").and_then(Value::as_str) {
                        Some("output_text") | Some("text") => {
                            let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                            events.push(IrEvent::Delta { index, delta: Delta::Text(text.to_string()) });
                            if let Some(anns) = part.get("annotations").and_then(Value::as_array) {
                                for a in anns {
                                    let kind = a.get("type").and_then(Value::as_str).unwrap_or("annotation");
                                    events.push(IrEvent::Delta {
                                        index,
                                        delta: Delta::Annotation(Annotation::new(kind, a.clone())),
                                    });
                                }
                            }
                        }
                        Some("refusal") => {
                            let text = part.get("refusal").and_then(Value::as_str).unwrap_or_default();
                            events.push(IrEvent::Delta { index, delta: Delta::Refusal(text.to_string()) });
                        }
                        _ => {}
                    }
                }
            }
            events.push(IrEvent::ItemStop { index });
        }
        "function_call" => {
            let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or_default();
            let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
            events.push(IrEvent::ItemStart {
                index,
                kind: ItemKind::ToolCall,
                id,
                call: Some((CallId::new(call_id), name.to_string())),
            });
            let args = item.get("arguments").and_then(Value::as_str).unwrap_or("");
            events.push(IrEvent::Delta { index, delta: Delta::ToolArgs(args.to_string()) });
            events.push(IrEvent::ItemStop { index });
        }
        "reasoning" => {
            events.push(IrEvent::ItemStart { index, kind: ItemKind::Reasoning, id, call: None });
            if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                for (i, s) in summary.iter().enumerate() {
                    let text = s.get("text").and_then(Value::as_str).unwrap_or_default();
                    events.push(IrEvent::Delta {
                        index,
                        delta: Delta::ReasoningSummary { part: i as u32, text: text.to_string() },
                    });
                }
            }
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for c in content {
                    if let Some(text) = c.get("text").and_then(Value::as_str) {
                        events.push(IrEvent::Delta { index, delta: Delta::ReasoningText(text.to_string()) });
                    }
                }
            }
            if let Some(enc) = item.get("encrypted_content").and_then(Value::as_str) {
                if !enc.is_empty() {
                    let mut blob = OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Encrypted, enc);
                    blob.model = Some(model.to_string());
                    events.push(IrEvent::Delta { index, delta: Delta::Opaque(blob) });
                }
            }
            events.push(IrEvent::ItemStop { index });
        }
        "compaction" => {
            events.push(IrEvent::ItemStart { index, kind: ItemKind::Compaction, id, call: None });
            let enc = item.get("encrypted_content").and_then(Value::as_str).unwrap_or_default();
            let mut blob = OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Compaction, enc);
            blob.model = Some(model.to_string());
            events.push(IrEvent::Delta { index, delta: Delta::Opaque(blob) });
            events.push(IrEvent::ItemStop { index });
        }
        _ => {
            // Hosted tool call/result item — pass through verbatim.
            events.push(IrEvent::ItemStart { index, kind: ItemKind::ProviderToolCall, id, call: None });
            events.push(IrEvent::Delta {
                index,
                delta: Delta::ProviderRaw { family: ProviderFamily::OpenAI, raw: item.clone() },
            });
            events.push(IrEvent::ItemStop { index });
        }
    }
}

/// The `input_tokens_details` key naming the **cache-write** counter, in precedence order.
/// Only spellings observed in real traffic are listed (plan L4).
///
/// - `cache_write_tokens` — OpenAI's own spelling on this dialect, and the one
///   [`render_usage`] emits. Captured live from `gpt-4o-mini` and `gpt-6-astra` on 2026-09-10
///   (`crates/e2e/dataset/responses/*/response.json`).
/// - `cache_creation_tokens` — the Anthropic-bridge spelling, carried over from the Chat surface.
/// - `created_cache_tokens` — vLLM (observed on Kimi K3 behind vLLM 0.27.x).
///
/// [`render_usage`]: crate::render::render_usage
pub(crate) const CACHE_WRITE_KEYS: &[&str] =
    &["cache_write_tokens", "cache_creation_tokens", "created_cache_tokens"];

/// The `input_tokens_details` key naming the 1-hour-TTL subset of the cache writes. OpenAI has
/// no spelling for this — it reports no TTL split at all — so [`render_usage`] emits
/// `cache_write_1h_tokens`, which reads as the subset of the `cache_write_tokens` beside it.
/// The Anthropic-bridge spelling `cache_creation_1h_tokens` is accepted on decode as well.
///
/// [`render_usage`]: crate::render::render_usage
pub(crate) const CACHE_WRITE_1H_KEYS: &[&str] =
    &["cache_write_1h_tokens", "cache_creation_1h_tokens"];

/// `usage` root keys this codec maps to typed counters or recomputes; everything else at the
/// root is preserved verbatim into `usage.ext` under `responses.<key>`.
const KNOWN_ROOT_KEYS: &[&str] = &[
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_tokens_details",
    "output_tokens_details",
    // Anthropic-shaped cache keys, which some compatible servers emit alongside the OpenAI
    // ones. Read as a fallback, so never also duplicated into `ext`.
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation",
];

/// Ext-key prefix for a counter preserved out of `input_tokens_details`.
pub(crate) const INPUT_DETAILS_NS: &str = "responses.input_tokens_details.";
/// Ext-key prefix for a counter preserved out of `output_tokens_details`.
pub(crate) const OUTPUT_DETAILS_NS: &str = "responses.output_tokens_details.";

/// First present key out of `keys`, read as a `u32`.
fn first_u32(v: Option<&Value>, keys: &[&str]) -> Option<u32> {
    let v = v?;
    keys.iter().find_map(|k| get_u32(v, k))
}

/// Decode the `usage` block.
///
/// `input_tokens` is the **gross** prompt (it counts the cached and written portions inside
/// it), so it is reduced to the IR's fresh `input` via [`Usage::from_gross`]. Every key this
/// codec does not map — at the root or in either details object — is preserved into
/// `usage.ext` under its dotted path.
pub(crate) fn decode_usage(v: Option<&Value>) -> Usage {
    let Some(v) = v else { return Usage::default() };
    let itd = v.get("input_tokens_details");
    let otd = v.get("output_tokens_details");

    let cache_read = itd
        .and_then(|d| get_u32(d, "cached_tokens"))
        .or_else(|| get_u32(v, "cache_read_input_tokens"));
    let cache_write = first_u32(itd, CACHE_WRITE_KEYS)
        .or_else(|| get_u32(v, "cache_creation_input_tokens"));
    let cache_write_1h = first_u32(itd, CACHE_WRITE_1H_KEYS).or_else(|| {
        v.get("cache_creation").and_then(|c| get_u32(c, "ephemeral_1h_input_tokens"))
    });

    let mut usage = Usage::from_gross(
        get_u32(v, "input_tokens").unwrap_or(0),
        get_u32(v, "output_tokens").unwrap_or(0),
        cache_read,
        cache_write,
        cache_write_1h,
    );
    usage.reasoning = otd.and_then(|d| get_u32(d, "reasoning_tokens"));

    // Preserve everything this codec did not map, keyed by its path within the usage object.
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !KNOWN_ROOT_KEYS.contains(&k.as_str()) && !val.is_null() {
                usage.ext.insert(format!("responses.{k}"), val.clone());
            }
        }
    }
    let consumed_input = |k: &str| {
        k == "cached_tokens" || CACHE_WRITE_KEYS.contains(&k) || CACHE_WRITE_1H_KEYS.contains(&k)
    };
    preserve_details(&mut usage, itd, INPUT_DETAILS_NS, &consumed_input);
    preserve_details(&mut usage, otd, OUTPUT_DETAILS_NS, &|k| k == "reasoning_tokens");
    usage
}

/// Copy the unmapped entries of a `*_tokens_details` object into `usage.ext` under `ns`.
fn preserve_details(
    usage: &mut Usage,
    details: Option<&Value>,
    ns: &str,
    consumed: &dyn Fn(&str) -> bool,
) {
    let Some(obj) = details.and_then(Value::as_object) else { return };
    for (k, val) in obj {
        if !consumed(k) && !val.is_null() {
            usage.ext.insert(format!("{ns}{k}"), val.clone());
        }
    }
}

/// Derive the stop reason from a completed/incomplete response.
fn stop_reason(root: &Value, status: &str, has_function_call: bool) -> StopReason {
    match status {
        "incomplete" => {
            let reason = root
                .get("incomplete_details")
                .and_then(|d| d.get("reason"))
                .and_then(Value::as_str)
                .unwrap_or("");
            match reason {
                "content_filter" => StopReason::ContentFilter,
                _ => StopReason::MaxTokens,
            }
        }
        "cancelled" => StopReason::Cancelled,
        _ if has_function_call => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

/// Encode an aggregated response into a `response` object.
pub(crate) fn encode_response(r: &IrResponse, ctx: &EncodeCtx) -> Bytes {
    let rctx = RenderCtx {
        response_id: ctx.response_id.as_str(),
        created_at: ctx.created_at,
        model: &ctx.client_model,
        request_echo: &ctx.request_echo,
        sealer: &ctx.sealer,
        expose: ctx.expose.clone(),
        include: &ctx.include,
        store: ctx.store,
    };
    let obj = build_response_value(r, &rctx);
    canon::to_bytes(&obj)
}

/// Build the `response` object [`Value`] for an [`IrResponse`] under a render context.
pub(crate) fn build_response_value(r: &IrResponse, rctx: &RenderCtx) -> Value {
    let (status, incomplete) = status_for(&r.stop);
    let output: Vec<Value> =
        r.items.iter().enumerate().map(|(i, item)| render_output_item(item, i, rctx)).collect();
    build_response_object(status, output, Some(&r.usage), None, incomplete, rctx)
}
