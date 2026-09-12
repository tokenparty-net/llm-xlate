//! Shared rendering: IR items → Responses wire items (input + output shapes), the `usage`
//! block, status derivation, id minting, opaque-carrier sealing, and the full `response`
//! object. Both [`crate::response::encode_response`], the stream encoder's terminal snapshot,
//! and [`crate::stored`] render through here so the three agree byte-for-byte.

use serde_json::{Map, Value};

use llm_xlate_core::{
    Extensions, Item, MediaSource, OpaqueBlob, OpaqueKind, Part, ProviderFamily, ReasoningExposure,
    ReasoningItem, Role, Sealer, StopReason, Usage,
};

use crate::util::Ob;

/// Rendering context threaded through response/stored rendering.
pub(crate) struct RenderCtx<'a> {
    /// Client-facing response id.
    pub response_id: &'a str,
    /// Router-supplied unix seconds.
    pub created_at: u64,
    /// Model string to echo.
    pub model: &'a str,
    /// Echoed request params (built by [`crate::encode::request_echo`]).
    pub request_echo: &'a Map<String, Value>,
    /// Client-facing sealer for foreign opaque blobs.
    pub sealer: &'a Sealer,
    /// How reasoning is exposed to the client.
    pub expose: ReasoningExposure,
    /// The `include` list (gates `reasoning.encrypted_content`).
    pub include: &'a [String],
    /// The `store` flag (gates `reasoning.encrypted_content`).
    pub store: Option<bool>,
}

/// The `{kind}_{response_id}_{index}` prefix for an item kind.
pub(crate) fn kind_prefix(item: &Item) -> &'static str {
    match item {
        Item::Message { .. } => "msg",
        Item::ToolCall { .. } => "fc",
        Item::Reasoning(_) => "rs",
        Item::ProviderToolCall(_) => "item",
        Item::ProviderToolResult(_) => "item",
        Item::ToolResult { .. } => "fc",
        Item::Compaction(_) => "cmp",
    }
}

/// The item-id prefix for a streaming [`llm_xlate_core::ItemKind`] (used by the stream encoder
/// before the full [`Item`] exists). Kept in lockstep with [`kind_prefix`].
pub(crate) fn kind_prefix_for_kind(kind: llm_xlate_core::ItemKind) -> &'static str {
    use llm_xlate_core::ItemKind;
    match kind {
        ItemKind::Message => "msg",
        ItemKind::ToolCall => "fc",
        ItemKind::Reasoning => "rs",
        ItemKind::ProviderToolCall | ItemKind::ProviderToolResult => "item",
        ItemKind::Compaction => "cmp",
    }
}

/// Mint a Responses item id per plan §5: `"{kind}_{response_id}_{index}"`.
pub(crate) fn mint_id(prefix: &str, response_id: &str, index: usize) -> String {
    format!("{prefix}_{response_id}_{index}")
}

/// The output-item id for an item: its own id when it carries one, else a minted
/// `"{kind}_{response_id}_{index}"`. Shared by [`render_output_item`] and the stream encoder so
/// an item's streamed `output_item.added` / `output_item.done` and the terminal-snapshot id all
/// agree.
pub(crate) fn item_id_for(item: &Item, index: usize, response_id: &str) -> String {
    let existing = match item {
        Item::Message { id, .. } => id.as_ref(),
        Item::ToolCall { id, .. } => id.as_ref(),
        Item::Reasoning(ri) => ri.id.as_ref(),
        _ => None,
    };
    match existing {
        Some(i) => i.as_str().to_string(),
        None => mint_id(kind_prefix(item), response_id, index),
    }
}

/// The native/sealed carrier string for an opaque blob at the client boundary: native when
/// same-family (OpenAI), a sealed `rtr1.` envelope when foreign.
pub(crate) fn seal_carrier(blob: &OpaqueBlob, sealer: &Sealer) -> String {
    if blob.family == ProviderFamily::OpenAI {
        blob.data.clone()
    } else {
        sealer.seal(blob)
    }
}

/// The status string plus optional `incomplete_details` for a stop reason.
pub(crate) fn status_for(stop: &StopReason) -> (&'static str, Option<Value>) {
    match stop {
        StopReason::MaxTokens => (
            "incomplete",
            Some(Ob::new().set("reason", Value::from("max_output_tokens")).build()),
        ),
        StopReason::ContentFilter => (
            "incomplete",
            Some(Ob::new().set("reason", Value::from("content_filter")).build()),
        ),
        StopReason::Cancelled => ("cancelled", None),
        _ => ("completed", None),
    }
}

/// Render the `usage` block in the Responses shape.
///
/// `input_tokens` is re-grossed from the IR's disjoint counters ([`Usage::gross_prompt`]) and
/// `total_tokens` is that plus `output`, so a client's prompt count contains its own cached and
/// written portions — the OpenAI dialects' convention.
///
/// `input_tokens_details` also carries the cache-write counters when the upstream reported
/// them: `cache_write_tokens`, which is OpenAI's own spelling on this dialect, and the
/// non-standard `cache_write_1h_tokens` for the 1-hour subset, which has no native
/// spelling. Cache writes are billed prompt tokens, so surfacing the TTL split under a
/// non-standard key beats dropping it, and OpenAI SDKs ignore unknown fields. Counters that
/// `decode_usage` preserved into `usage.ext` are re-emitted into the object they came from.
pub(crate) fn render_usage(u: &Usage) -> Value {
    use crate::response::{INPUT_DETAILS_NS, OUTPUT_DETAILS_NS};

    let gross = u.gross_prompt();

    let mut itd = Map::new();
    itd.insert("cached_tokens".into(), Value::from(u.cache_read.unwrap_or(0)));
    if let Some(w) = u.cache_write {
        itd.insert("cache_write_tokens".into(), Value::from(w));
    }
    if let Some(w) = u.cache_write_1h {
        itd.insert("cache_write_1h_tokens".into(), Value::from(w));
    }
    merge_preserved(&mut itd, u, INPUT_DETAILS_NS);

    let mut otd = Map::new();
    otd.insert("reasoning_tokens".into(), Value::from(u.reasoning.unwrap_or(0)));
    merge_preserved(&mut otd, u, OUTPUT_DETAILS_NS);

    let mut obj = Ob::new()
        .set("input_tokens", Value::from(gross))
        .set("input_tokens_details", Value::Object(itd))
        .set("output_tokens", Value::from(u.output))
        .set("output_tokens_details", Value::Object(otd))
        .set("total_tokens", Value::from(gross + u.output))
        .build();

    // Root-level counters this codec does not model, re-emitted after the standard fields.
    // Nested namespaces are excluded by the `.` in their prefixes, which a bare root key never
    // contains.
    if let Some(map) = obj.as_object_mut() {
        for (k, val) in u.ext.iter() {
            if let Some(name) = k.strip_prefix("responses.") {
                if !name.contains('.') {
                    map.entry(name.to_string()).or_insert_with(|| val.clone());
                }
            }
        }
    }
    obj
}

/// Copy the `usage.ext` entries under `ns` back into the details object they were decoded
/// from, without overwriting a counter this codec already wrote.
fn merge_preserved(dst: &mut Map<String, Value>, u: &Usage, ns: &str) {
    for (k, val) in u.ext.iter() {
        if let Some(name) = k.strip_prefix(ns) {
            dst.entry(name.to_string()).or_insert_with(|| val.clone());
        }
    }
}

/// Render one image/document/audio media source into an input content part body (without the
/// `type` key, which the caller sets).
fn media_input_part(kind: &str, src: &MediaSource, detail: Option<&str>) -> Value {
    match kind {
        "image" => {
            let ob = match src {
                MediaSource::Base64 { media_type, data } => {
                    let url = llm_xlate_core::canon::data_url::build(media_type, data);
                    Ob::new().set("type", "input_image".into()).set("image_url", Value::from(url))
                }
                MediaSource::Url(u) => {
                    Ob::new().set("type", "input_image".into()).set("image_url", Value::from(u.clone()))
                }
                MediaSource::FileRef { id, .. } => {
                    Ob::new().set("type", "input_image".into()).set("file_id", Value::from(id.clone()))
                }
                MediaSource::Text(t) => {
                    Ob::new().set("type", "input_image".into()).set("image_url", Value::from(t.clone()))
                }
            };
            ob.opt_str("detail", detail).build()
        }
        _ => Value::Null,
    }
}

/// Render an input-side content part (`input_text`/`input_image`/`input_file`/`input_audio`).
/// `detail` is the resolved `image_url.detail` for image parts.
pub(crate) fn render_input_part(part: &Part, title_carry: Option<&str>, detail: Option<&str>) -> Option<Value> {
    match part {
        Part::Text { text, .. } => {
            Some(Ob::new().set("type", "input_text".into()).set("text", Value::from(text.clone())).build())
        }
        Part::Image(src) => Some(media_input_part("image", src, detail)),
        Part::Document { source, title, media_type } => {
            let _ = media_type;
            match source {
                MediaSource::Base64 { media_type, data } => {
                    let url = llm_xlate_core::canon::data_url::build(media_type, data);
                    // OpenAI requires `filename` beside base64 `file_data` (the extension is how
                    // it infers the type) and 400s without it; a cross-family PDF often has no
                    // title, so synthesize a default name from the media type when absent.
                    let name = title
                        .clone()
                        .unwrap_or_else(|| llm_xlate_core::canon::filename::for_media_type(media_type));
                    Some(
                        Ob::new()
                            .set("type", "input_file".into())
                            .set("filename", Value::from(name))
                            .set("file_data", Value::from(url))
                            .build(),
                    )
                }
                MediaSource::Url(u) => Some(
                    Ob::new().set("type", "input_file".into()).set("file_url", Value::from(u.clone())).build(),
                ),
                MediaSource::FileRef { id, .. } => Some(
                    Ob::new().set("type", "input_file".into()).set("file_id", Value::from(id.clone())).build(),
                ),
                MediaSource::Text(t) => {
                    // Inline text doc → an `input_text` part wrapped by `wrap::document`.
                    let wrapped = llm_xlate_core::wrap::document(title.as_deref().or(title_carry), t);
                    Some(Ob::new().set("type", "input_text".into()).set("text", Value::from(wrapped)).build())
                }
            }
        }
        Part::Audio(src) => {
            let ob = match src {
                MediaSource::Base64 { media_type, data } => {
                    let b64 = base64_encode(data);
                    let fmt = media_type.rsplit('/').next().unwrap_or("wav").to_string();
                    Ob::new()
                        .set("type", "input_audio".into())
                        .set(
                            "input_audio",
                            Ob::new().set("data", Value::from(b64)).set("format", Value::from(fmt)).build(),
                        )
                }
                MediaSource::Url(u) => Ob::new()
                    .set("type", "input_audio".into())
                    .set("input_audio", Ob::new().set("url", Value::from(u.clone())).build()),
                MediaSource::FileRef { id, .. } => Ob::new()
                    .set("type", "input_audio".into())
                    .set("input_audio", Ob::new().set("file_id", Value::from(id.clone())).build()),
                MediaSource::Text(t) => Ob::new()
                    .set("type", "input_audio".into())
                    .set("input_audio", Ob::new().set("data", Value::from(t.clone())).build()),
            };
            Some(ob.build())
        }
        Part::Refusal { text } => {
            Some(Ob::new().set("type", "refusal".into()).set("refusal", Value::from(text.clone())).build())
        }
        Part::Opaque(_) => None,
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Render an assistant message's output content parts (`output_text` / `refusal`).
fn render_output_message_content(parts: &[Part]) -> Vec<Value> {
    let mut out = Vec::new();
    for p in parts {
        match p {
            Part::Text { text, annotations, .. } => {
                let anns: Vec<Value> = annotations.iter().map(|a| a.raw.clone()).collect();
                out.push(
                    Ob::new()
                        .set("type", "output_text".into())
                        .set("text", Value::from(text.clone()))
                        .set("annotations", Value::Array(anns))
                        .build(),
                );
            }
            Part::Refusal { text } => {
                out.push(
                    Ob::new().set("type", "refusal".into()).set("refusal", Value::from(text.clone())).build(),
                );
            }
            _ => {}
        }
    }
    out
}

/// Render a reasoning item as an OUTPUT item body (summary/content/encrypted_content).
fn render_reasoning_output(ri: &ReasoningItem, rctx: &RenderCtx, id: Option<String>) -> Value {
    // Under `Summary`, an item that carries only reasoning text (no summary parts) folds that
    // text into a single `summary_text` entry, so a thinking-capable backend bridged to a
    // summary-exposing Responses client still yields a summary. This mirrors the streaming
    // encoder's fold, making the two paths byte-identical (plan §11.8). Under `None` the fold
    // does not apply and the chain-of-thought stays hidden; under `Full` it is exposed as
    // `content` below instead.
    let summary_texts: Vec<String> =
        if matches!(rctx.expose, ReasoningExposure::Summary(_)) && ri.summary.is_empty() {
            ri.text.clone().filter(|t| !t.is_empty()).into_iter().collect()
        } else {
            ri.summary.clone()
        };
    let summary: Vec<Value> = summary_texts
        .iter()
        .map(|s| Ob::new().set("type", "summary_text".into()).set("text", Value::from(s.clone())).build())
        .collect();
    let mut ob = Ob::new().set("type", "reasoning".into()).opt_str("id", id);
    ob = ob.set("summary", Value::Array(summary));
    if rctx.expose == ReasoningExposure::Full {
        if let Some(t) = &ri.text {
            let content = vec![Ob::new()
                .set("type", "reasoning_text".into())
                .set("text", Value::from(t.clone()))
                .build()];
            ob = ob.set("content", Value::Array(content));
        }
    }
    if let Some(blob) = &ri.opaque {
        if reasoning_encrypted_allowed(rctx) {
            ob = ob.set("encrypted_content", Value::from(seal_carrier(blob, rctx.sealer)));
        }
    }
    ob.set("status", "completed".into()).build()
}

/// Whether `reasoning.encrypted_content` should be emitted on output reasoning items.
pub(crate) fn reasoning_encrypted_allowed(rctx: &RenderCtx) -> bool {
    rctx.store == Some(false)
        || rctx.include.iter().any(|s| s == "reasoning.encrypted_content")
}

/// Render one IR item as a Responses OUTPUT item.
pub(crate) fn render_output_item(item: &Item, index: usize, rctx: &RenderCtx) -> Value {
    let id = item_id_for(item, index, rctx.response_id);
    match item {
        Item::Message { role, content, .. } => {
            let role_s = match role {
                Role::Assistant => "assistant",
                Role::User => "user",
            };
            Ob::new()
                .set("id", Value::from(id))
                .set("type", "message".into())
                .set("status", "completed".into())
                .set("role", Value::from(role_s))
                .set("content", Value::Array(render_output_message_content(content)))
                .build()
        }
        Item::ToolCall { call_id, name, arguments, .. } => Ob::new()
            .set("id", Value::from(id))
            .set("type", "function_call".into())
            .set("status", "completed".into())
            .set("call_id", Value::from(call_id.as_str().to_string()))
            .set("name", Value::from(name.clone()))
            .set("arguments", Value::from(arguments.as_str().to_string()))
            .build(),
        Item::Reasoning(ri) => render_reasoning_output(ri, rctx, Some(id)),
        Item::ProviderToolCall(oi) | Item::ProviderToolResult(oi) => oi.raw.clone(),
        // A tool result is a user-side input item, not an output item; render it faithfully for
        // the rare transcript that carries one through the output list.
        Item::ToolResult { call_id, content, .. } => {
            let output = if content.iter().all(|p| matches!(p, Part::Text { .. })) {
                Value::String(content.iter().filter_map(Part::as_text).collect())
            } else {
                Value::Array(
                    content
                        .iter()
                        .filter_map(|p| render_input_part(p, None, None))
                        .collect(),
                )
            };
            Ob::new()
                .set("id", Value::from(id))
                .set("type", "function_call_output".into())
                .set("call_id", Value::from(call_id.as_str().to_string()))
                .set("output", output)
                .build()
        }
        Item::Compaction(blob) => Ob::new()
            .set("type", "compaction".into())
            .set("encrypted_content", Value::from(seal_carrier(blob, rctx.sealer)))
            .build(),
    }
}

/// Assemble the full Responses `response` object shared by the non-streaming encoder, the
/// stream encoder's terminal snapshot, and the stored-response encoder.
pub(crate) fn build_response_object(
    status: &str,
    output_items: Vec<Value>,
    usage: Option<&Usage>,
    error: Option<Value>,
    incomplete: Option<Value>,
    rctx: &RenderCtx,
) -> Value {
    let echo = rctx.request_echo;
    let get = |k: &str| echo.get(k).cloned();
    Ob::new()
        .set("id", Value::from(rctx.response_id.to_string()))
        .set("object", "response".into())
        .set("created_at", Value::from(rctx.created_at))
        .set("status", Value::from(status.to_string()))
        .set("background", get("background").unwrap_or(Value::Null))
        .set("error", error.unwrap_or(Value::Null))
        .set("incomplete_details", incomplete.unwrap_or(Value::Null))
        .set("instructions", get("instructions").unwrap_or(Value::Null))
        .set("max_output_tokens", get("max_output_tokens").unwrap_or(Value::Null))
        .set("metadata", get("metadata").unwrap_or(Value::Object(Map::new())))
        .set("model", Value::from(rctx.model.to_string()))
        .set("output", Value::Array(output_items))
        .set("parallel_tool_calls", get("parallel_tool_calls").unwrap_or(Value::Bool(true)))
        .set("previous_response_id", get("previous_response_id").unwrap_or(Value::Null))
        .set("reasoning", get("reasoning").unwrap_or(Value::Null))
        .set("service_tier", get("service_tier").unwrap_or(Value::Null))
        .set("store", get("store").unwrap_or(Value::Null))
        .set("temperature", get("temperature").unwrap_or(Value::Null))
        .set("text", get("text").unwrap_or(Value::Null))
        .set("tool_choice", get("tool_choice").unwrap_or(Value::from("auto")))
        .set("tools", get("tools").unwrap_or(Value::Array(Vec::new())))
        .set("top_p", get("top_p").unwrap_or(Value::Null))
        .set("truncation", get("truncation").unwrap_or(Value::from("disabled")))
        .set("usage", usage.map(render_usage).unwrap_or(Value::Null))
        .set("user", get("user").unwrap_or(Value::Null))
        .opt("safety_identifier", get("safety_identifier"))
        .build()
}

/// Look up an `image_url.detail` value keyed `"<item>/<part>"` in the ext map.
pub(crate) fn detail_lookup(ext: &Extensions, item: usize, part: usize) -> Option<String> {
    ext.get("responses.image_detail")
        .and_then(|m| m.get(format!("{item}/{part}")))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Opaque kind for a reasoning carrier (always Encrypted for Responses).
pub(crate) const REASONING_KIND: OpaqueKind = OpaqueKind::Encrypted;
