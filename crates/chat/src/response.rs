//! Non-streaming response path: `decode_response` (a `chat.completion` body → [`IrEvent`]s,
//! `choices[0]` only) and `encode_response` (an aggregated [`IrResponse`] → a `chat.completion`
//! object, regrouped like the stream encoder, with sealed opaque reasoning in
//! `reasoning_details` and mapped usage).

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::canon;
use llm_xlate_core::codec::EncodeCtx;
use llm_xlate_core::ir::{
    Annotation, CallId, Delta, Extensions, IrEvent, IrResponse, Item, ItemKind, OpaqueBlob,
    OpaqueKind, Part, ProviderFamily, ResponseId, StopReason,
};
use llm_xlate_core::{Capabilities, XlateError};

use crate::common::{decode_usage, encode_usage, finish_reason_to_stop, ns_key, stop_to_finish_reason};

/// Decode a non-streaming `chat.completion` body into IR events (`choices[0]` only).
pub fn decode_response(body: &[u8], _caps: &Capabilities) -> Result<Vec<IrEvent>, XlateError> {
    let root = canon::parse_upstream(body)?;

    if let Some(err_obj) = root.get("error") {
        if !err_obj.is_null() {
            return Ok(vec![IrEvent::Error(crate::errors::error_from_value(err_obj))]);
        }
    }

    let response_id = root.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
    let model = root.get("model").and_then(Value::as_str).unwrap_or_default().to_string();

    let mut events = vec![IrEvent::Start {
        response_id: ResponseId::new(response_id),
        model: model.clone(),
        usage_prefill: None,
    }];

    let choice = root
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().find(|c| c.get("index").and_then(Value::as_u64).unwrap_or(0) == 0).or(a.first()));

    let mut index: u32 = 0;
    let mut has_refusal = false;
    let mut finish = None;

    if let Some(choice) = choice {
        let msg = choice.get("message").cloned().unwrap_or(Value::Null);
        finish = choice.get("finish_reason").and_then(Value::as_str).map(str::to_string);

        // 1. reasoning
        let reasoning_text = msg
            .get("reasoning_content")
            .and_then(Value::as_str)
            .or_else(|| msg.get("reasoning").and_then(Value::as_str));
        let details = msg.get("reasoning_details").and_then(Value::as_array);
        if reasoning_text.is_some() || details.is_some() {
            events.push(IrEvent::ItemStart {
                index,
                kind: ItemKind::Reasoning,
                id: None,
                call: None,
            });
            if let Some(rt) = reasoning_text {
                if !rt.is_empty() {
                    events.push(IrEvent::Delta { index, delta: Delta::ReasoningText(rt.to_string()) });
                }
            }
            if let Some(details) = details {
                for entry in details {
                    let blob = opaque_from_detail(entry, &model);
                    events.push(IrEvent::Delta { index, delta: Delta::Opaque(blob) });
                }
            }
            events.push(IrEvent::ItemStop { index });
            index += 1;
        }

        // 2. message content / refusal / annotations
        let content = msg.get("content");
        let refusal = msg.get("refusal").and_then(Value::as_str);
        let annotations = msg.get("annotations").and_then(Value::as_array);
        let content_deltas = content_to_deltas(content);
        if !content_deltas.is_empty() || refusal.is_some() || annotations.is_some() {
            events.push(IrEvent::ItemStart { index, kind: ItemKind::Message, id: None, call: None });
            for d in content_deltas {
                if matches!(d, Delta::Refusal(_)) {
                    has_refusal = true;
                }
                events.push(IrEvent::Delta { index, delta: d });
            }
            if let Some(r) = refusal {
                has_refusal = true;
                events.push(IrEvent::Delta { index, delta: Delta::Refusal(r.to_string()) });
            }
            if let Some(anns) = annotations {
                for a in anns {
                    let kind = a.get("type").and_then(Value::as_str).unwrap_or("annotation").to_string();
                    events.push(IrEvent::Delta {
                        index,
                        delta: Delta::Annotation(Annotation::new(kind, a.clone())),
                    });
                }
            }
            events.push(IrEvent::ItemStop { index });
            index += 1;
        }

        // 3. tool calls
        if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let id = call.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                let func = call.get("function");
                let name = func.and_then(|f| f.get("name")).and_then(Value::as_str).unwrap_or("").to_string();
                let args = func
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                events.push(IrEvent::ItemStart {
                    index,
                    kind: ItemKind::ToolCall,
                    id: None,
                    call: Some((CallId::new(id), name)),
                });
                if !args.is_empty() {
                    events.push(IrEvent::Delta { index, delta: Delta::ToolArgs(args) });
                }
                events.push(IrEvent::ItemStop { index });
                index += 1;
            }
        }

        // 4. legacy function_call
        if let Some(fc) = msg.get("function_call") {
            if !fc.is_null() {
                let name = fc.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let args = fc.get("arguments").and_then(Value::as_str).unwrap_or("").to_string();
                events.push(IrEvent::ItemStart {
                    index,
                    kind: ItemKind::ToolCall,
                    id: None,
                    call: Some((CallId::new(name.clone()), name)),
                });
                if !args.is_empty() {
                    events.push(IrEvent::Delta { index, delta: Delta::ToolArgs(args) });
                }
                events.push(IrEvent::ItemStop { index });
            }
        }
    }

    let usage = decode_usage(root.get("usage").unwrap_or(&Value::Null));
    let reason = finish_reason_to_stop(finish.as_deref(), has_refusal);
    let mut ext = Extensions::new();
    if let Some(st) = root.get("service_tier").and_then(Value::as_str) {
        ext.insert(ns_key("service_tier"), Value::from(st));
    }
    if reason == StopReason::Refusal {
        let mut sd = Map::new();
        sd.insert("type".into(), Value::from("refusal"));
        ext.insert("stop_details".to_string(), Value::Object(sd));
    }
    events.push(IrEvent::Stop { reason, usage, ext });
    Ok(events)
}

/// Map a `content` value (string | array of parts | null) into a sequence of deltas.
fn content_to_deltas(content: Option<&Value>) -> Vec<Delta> {
    match content {
        Some(Value::String(s)) if !s.is_empty() => vec![Delta::Text(s.clone())],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|p| {
                let ty = p.get("type").and_then(Value::as_str)?;
                match ty {
                    "text" => Some(Delta::Text(p.get("text").and_then(Value::as_str).unwrap_or("").to_string())),
                    "refusal" => Some(Delta::Refusal(
                        p.get("refusal").and_then(Value::as_str).unwrap_or("").to_string(),
                    )),
                    _ => None,
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Build a native opaque blob from one `reasoning_details` entry (provider-facing).
fn opaque_from_detail(entry: &Value, model: &str) -> OpaqueBlob {
    let ty = entry.get("type").and_then(Value::as_str).unwrap_or("");
    let data = if ty == "router.opaque" {
        entry.get("data").and_then(Value::as_str).unwrap_or("").to_string()
    } else {
        canon::to_string(entry)
    };
    OpaqueBlob {
        family: ProviderFamily::OpenAI,
        kind: OpaqueKind::Encrypted,
        data,
        model: if model.is_empty() { None } else { Some(model.to_string()) },
    }
}

/// Encode an aggregated response into a `chat.completion` body (client-facing).
pub fn encode_response(r: &IrResponse, ctx: &EncodeCtx) -> Bytes {
    let mut text = String::new();
    let mut refusal: Option<String> = None;
    let mut reasoning_text: Option<String> = None;
    let mut opaque: Vec<&OpaqueBlob> = Vec::new();
    let mut annotations: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for item in &r.items {
        match item {
            Item::Message { content, .. } => {
                for p in content {
                    match p {
                        Part::Text { text: t, annotations: anns, .. } => {
                            text.push_str(t);
                            for a in anns {
                                annotations.push(a.raw.clone());
                            }
                        }
                        Part::Refusal { text: t } => refusal = Some(t.clone()),
                        _ => {}
                    }
                }
            }
            Item::Reasoning(ri) => {
                if reasoning_text.is_none() {
                    reasoning_text = ri.text.clone().or_else(|| {
                        if ri.summary.is_empty() {
                            None
                        } else {
                            Some(ri.summary.join("\n\n"))
                        }
                    });
                }
                if let Some(o) = &ri.opaque {
                    opaque.push(o);
                }
            }
            Item::ToolCall { call_id, name, arguments, .. } => {
                let mut func = Map::new();
                func.insert("name".into(), Value::from(name.clone()));
                func.insert("arguments".into(), Value::from(arguments.as_str()));
                let mut tc = Map::new();
                tc.insert("id".into(), Value::from(call_id.as_str()));
                tc.insert("type".into(), Value::from("function"));
                tc.insert("function".into(), Value::Object(func));
                tool_calls.push(Value::Object(tc));
            }
            _ => {}
        }
    }

    // Assemble the assistant message (fixed key order).
    let mut message = Map::new();
    message.insert("role".into(), Value::from("assistant"));
    if text.is_empty() {
        message.insert("content".into(), Value::Null);
    } else {
        message.insert("content".into(), Value::from(text));
    }
    if let Some(rf) = refusal {
        message.insert("refusal".into(), Value::from(rf));
    } else {
        message.insert("refusal".into(), Value::Null);
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !annotations.is_empty() {
        message.insert("annotations".into(), Value::Array(annotations));
    }
    if let Some(rt) = reasoning_text {
        message.insert("reasoning_content".into(), Value::from(rt));
    }
    if !opaque.is_empty() {
        let details: Vec<Value> = opaque
            .iter()
            .map(|blob| {
                let sealed = ctx.sealer.seal(blob);
                let mut entry = Map::new();
                entry.insert("type".into(), Value::from("router.opaque"));
                entry.insert("data".into(), Value::from(sealed));
                Value::Object(entry)
            })
            .collect();
        message.insert("reasoning_details".into(), Value::Array(details));
    }

    let mut choice = Map::new();
    choice.insert("index".into(), Value::from(0));
    choice.insert("message".into(), Value::Object(message));
    choice.insert("finish_reason".into(), Value::from(stop_to_finish_reason(&r.stop)));
    choice.insert("logprobs".into(), Value::Null);

    let mut obj = Map::new();
    obj.insert("id".into(), Value::from(ctx.response_id.as_str()));
    obj.insert("object".into(), Value::from("chat.completion"));
    obj.insert("created".into(), Value::from(ctx.created_at));
    obj.insert("model".into(), Value::from(ctx.client_model.clone()));
    obj.insert("choices".into(), Value::Array(vec![Value::Object(choice)]));
    obj.insert("usage".into(), encode_usage(&r.usage));
    if let Some(st) = r.ext.get(&ns_key("service_tier")).and_then(Value::as_str) {
        obj.insert("service_tier".into(), Value::from(st));
    }

    canon::to_bytes(&Value::Object(obj))
}
