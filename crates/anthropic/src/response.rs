//! Non-streaming response bridging: `decode_response` (provider body -> [`IrEvent`]s) and
//! `encode_response` ([`IrResponse`] -> client body).
//!
//! `decode_response` emits the same event shape the [`crate::stream_dec`] decoder would for
//! the streaming twin, so `Aggregator(decode_response(body))` equals the aggregated stream
//! (plan §11.8). `encode_response` regroups an aggregated response into an Anthropic
//! `message` object using the shared block builders (opaque carriers sealed for the client).

use bytes::Bytes;
use serde_json::Value;

use llm_xlate_core::{
    canon, Annotation, Capabilities, Delta, EncodeCtx, Extensions, IrEvent, IrResponse, ItemKind,
    OpaqueBlob, OpaqueKind, ResponseId, XlateError,
};

use crate::encode::assistant_items_to_blocks;
use crate::shared::{stop_reason_from_wire, stop_reason_to_wire, usage_from_wire, usage_to_wire};
use crate::wire::{ext_key, is_provider_tool_result_type, is_server_tool_use_type, omap, EXT_PREFIX, FAMILY};

/// Decode a non-streaming Anthropic response body into an [`IrEvent`] list.
pub fn decode_response(body: &[u8], _caps: &Capabilities) -> Result<Vec<IrEvent>, XlateError> {
    let value = canon::parse_upstream(body)?;
    let obj = value
        .as_object()
        .ok_or_else(|| XlateError::upstream_malformed("Anthropic response is not an object"))?;

    let id = obj.get("id").and_then(Value::as_str).unwrap_or("").to_string();
    let model = obj.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let usage = obj.get("usage").map(usage_from_wire);

    let mut out = Vec::new();
    out.push(IrEvent::Start {
        response_id: ResponseId::new(id),
        model,
        usage_prefill: usage.clone(),
    });

    let model_bind = obj.get("model").and_then(Value::as_str).filter(|s| !s.is_empty());
    if let Some(Value::Array(content)) = obj.get("content") {
        for (i, block) in content.iter().enumerate() {
            decode_block(i as u32, block, model_bind, &mut out);
        }
    }

    let reason = obj.get("stop_reason").and_then(Value::as_str);
    let stop_sequence = obj.get("stop_sequence").and_then(Value::as_str);
    let mut ext = Extensions::new();
    for key in ["container", "context_management"] {
        if let Some(v) = obj.get(key) {
            if !v.is_null() {
                ext.insert(ext_key(key), v.clone());
            }
        }
    }
    out.push(IrEvent::Stop {
        reason: stop_reason_from_wire(reason, stop_sequence),
        usage: usage.unwrap_or_default(),
        ext,
    });

    Ok(out)
}

/// Emit the `ItemStart` / `Delta` / `ItemStop` triple for one response content block.
///
/// `model` is the upstream response model, stamped onto native opaque carriers so the
/// reasoning/compaction blob keeps its model binding across the envelope boundary.
fn decode_block(index: u32, block: &Value, model: Option<&str>, out: &mut Vec<IrEvent>) {
    let native_blob = |kind: OpaqueKind, data: String| -> OpaqueBlob {
        let mut b = OpaqueBlob::new(FAMILY, kind, data);
        b.model = model.map(str::to_string);
        b
    };
    let btype = block.get("type").and_then(Value::as_str).unwrap_or("");
    match btype {
        "text" => {
            out.push(item_start(index, ItemKind::Message, None));
            let text = block.get("text").and_then(Value::as_str).unwrap_or("");
            if !text.is_empty() {
                out.push(delta(index, Delta::Text(text.to_string())));
            }
            if let Some(Value::Array(cites)) = block.get("citations") {
                for c in cites {
                    let kind = c.get("type").and_then(Value::as_str).unwrap_or("citation").to_string();
                    out.push(delta(index, Delta::Annotation(Annotation::new(kind, c.clone()))));
                }
            }
        }
        "thinking" => {
            out.push(item_start(index, ItemKind::Reasoning, None));
            let text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
            if !text.is_empty() {
                out.push(delta(index, Delta::ReasoningText(text.to_string())));
            }
            let sig = block.get("signature").and_then(Value::as_str).unwrap_or("").to_string();
            out.push(delta(index, Delta::Opaque(native_blob(OpaqueKind::Signature, sig))));
        }
        "redacted_thinking" => {
            out.push(item_start(index, ItemKind::Reasoning, None));
            let data = block.get("data").and_then(Value::as_str).unwrap_or("").to_string();
            out.push(delta(index, Delta::Opaque(native_blob(OpaqueKind::Redacted, data))));
        }
        "tool_use" => {
            let id = block.get("id").and_then(Value::as_str).unwrap_or("").to_string();
            let name = block.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            out.push(item_start(index, ItemKind::ToolCall, Some((id.into(), name))));
            let input = block.get("input").cloned().unwrap_or(Value::Null);
            let is_empty_obj = input.as_object().map(|o| o.is_empty()).unwrap_or(true);
            if !is_empty_obj {
                out.push(delta(index, Delta::ToolArgs(canon::to_string(&input))));
            }
        }
        "compaction" => {
            out.push(item_start(index, ItemKind::Compaction, None));
            out.push(delta(
                index,
                Delta::Opaque(native_blob(OpaqueKind::Compaction, block.to_string())),
            ));
        }
        t if is_server_tool_use_type(t) => {
            out.push(item_start(index, ItemKind::ProviderToolCall, None));
            out.push(delta(index, Delta::ProviderRaw { family: FAMILY, raw: block.clone() }));
        }
        t if is_provider_tool_result_type(t) => {
            out.push(item_start(index, ItemKind::ProviderToolResult, None));
            out.push(delta(index, Delta::ProviderRaw { family: FAMILY, raw: block.clone() }));
        }
        _ => {
            out.push(item_start(index, ItemKind::ProviderToolCall, None));
            out.push(delta(index, Delta::ProviderRaw { family: FAMILY, raw: block.clone() }));
        }
    }
    out.push(IrEvent::ItemStop { index });
}

fn item_start(index: u32, kind: ItemKind, call: Option<(llm_xlate_core::CallId, String)>) -> IrEvent {
    IrEvent::ItemStart { index, kind, id: None, call }
}

fn delta(index: u32, delta: Delta) -> IrEvent {
    IrEvent::Delta { index, delta }
}

/// Encode an aggregated [`IrResponse`] into an Anthropic `message` object.
pub fn encode_response(r: &IrResponse, ctx: &EncodeCtx) -> Bytes {
    // `encode_response` gets no capabilities; assistant-side blocks never need result-content
    // gating, so a default (permissive-enough) capability set is used for block building.
    let caps = Capabilities::default();
    let content = assistant_items_to_blocks(&r.items, ctx, &caps);
    let (reason, stop_sequence) = stop_reason_to_wire(&r.stop);

    let mut root = omap();
    root.insert("id".into(), Value::from(ctx.response_id.as_str()));
    root.insert("type".into(), Value::from("message"));
    root.insert("role".into(), Value::from("assistant"));
    root.insert("model".into(), Value::from(ctx.client_model.clone()));
    root.insert("content".into(), Value::Array(content));
    root.insert("stop_reason".into(), Value::from(reason));
    root.insert("stop_sequence".into(), stop_sequence.map(Value::from).unwrap_or(Value::Null));
    root.insert("usage".into(), usage_to_wire(&r.usage));

    for (k, v) in r.ext.iter() {
        if let Some(field) = k.strip_prefix(EXT_PREFIX) {
            if field == "container" || field == "context_management" {
                root.insert(field.to_string(), v.clone());
            }
        }
    }

    canon::to_bytes(&Value::Object(root))
}
