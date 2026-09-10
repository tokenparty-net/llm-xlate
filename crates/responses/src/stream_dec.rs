//! [`ResponsesStreamDecoder`]: an OpenAI Responses SSE stream → [`IrEvent`]s (provider-facing).
//!
//! Built on [`SseParser`]. Every Responses stream event carries a monotonic `sequence_number`
//! and a `type` (in both the `event:` line and the JSON `data`); a gap or regression in the
//! sequence poisons the decoder — it emits a single [`XlateError`] (`UpstreamMalformed`) and
//! nothing further (plan §8).
//!
//! Provider-hosted tool items (`web_search_call`, `mcp_call`, …) stream as
//! `ItemStart{kind: ProviderToolCall}` → [`Delta::ProviderRaw`] (the finished raw block at
//! `output_item.done`) → `ItemStop`, per the core provider-item mechanism. Reasoning items
//! carry their `encrypted_content` out as a [`Delta::Opaque`] at `output_item.done`.

use std::collections::HashMap;

use serde_json::Value;

use llm_xlate_core::{
    canon, Annotation, CallId, Delta, Extensions, IrEvent, ItemId, ItemKind, OpaqueBlob, OpaqueKind,
    ProviderFamily, ResponseId, SseParser, StopReason, StreamDecoder, XlateError,
};

use crate::errors::error_from_value;
use crate::response::decode_usage;
use crate::util::{get_str, get_u32};

/// Streaming decoder for the Responses SSE dialect.
#[derive(Default)]
pub struct ResponsesStreamDecoder {
    parser: SseParser,
    /// Upstream model from `response.created`, stamped onto opaque reasoning carriers.
    model: String,
    /// Last seen `sequence_number` (for gap/regression detection).
    last_seq: Option<u64>,
    /// Once malformed, emit nothing further.
    poisoned: bool,
    /// Whether any `function_call` output item was seen (→ `ToolUse` stop reason).
    has_function_call: bool,
    /// Accumulated `function_call_arguments` deltas per output index, for the `.done` check.
    tool_args: HashMap<u32, String>,
}

impl ResponsesStreamDecoder {
    /// A fresh decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process one parsed SSE event's JSON `data`, appending any IR events.
    fn on_event(&mut self, data: &str, out: &mut Vec<IrEvent>) {
        if self.poisoned {
            return;
        }
        let Ok(root) = canon::parse_upstream(data.as_bytes()) else {
            // Non-JSON payload (e.g. a stray keepalive line rendered as data) — ignore.
            return;
        };

        // Sequence-number discipline: every event carries one; a gap or regression is fatal.
        if let Some(seq) = get_u64(&root, "sequence_number") {
            match self.last_seq {
                Some(prev) if seq != prev + 1 => {
                    self.poisoned = true;
                    out.push(IrEvent::Error(XlateError::upstream_malformed(format!(
                        "responses stream sequence_number gap: expected {}, got {seq}",
                        prev + 1
                    ))));
                    return;
                }
                _ => self.last_seq = Some(seq),
            }
        }

        let ty = get_str(&root, "type").unwrap_or_default().to_string();
        self.dispatch(&ty, &root, out);
    }

    fn dispatch(&mut self, ty: &str, root: &Value, out: &mut Vec<IrEvent>) {
        match ty {
            "response.created" => {
                let resp = root.get("response").unwrap_or(&Value::Null);
                let id = get_str(resp, "id").unwrap_or_default().to_string();
                self.model = get_str(resp, "model").unwrap_or_default().to_string();
                out.push(IrEvent::Start {
                    response_id: ResponseId::new(id),
                    model: self.model.clone(),
                    usage_prefill: None,
                });
            }
            "response.in_progress" | "response.queued" => {}
            "response.output_item.added" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let item = root.get("item").unwrap_or(&Value::Null);
                let item_ty = get_str(item, "type").unwrap_or_default();
                let id = get_str(item, "id").map(ItemId::new);
                let (kind, call) = match item_ty {
                    "message" => (ItemKind::Message, None),
                    "function_call" => {
                        self.has_function_call = true;
                        self.tool_args.insert(index, String::new());
                        let call_id = get_str(item, "call_id").unwrap_or_default();
                        let name = get_str(item, "name").unwrap_or_default();
                        (ItemKind::ToolCall, Some((CallId::new(call_id), name.to_string())))
                    }
                    "reasoning" => (ItemKind::Reasoning, None),
                    "compaction" => (ItemKind::Compaction, None),
                    _ => (ItemKind::ProviderToolCall, None),
                };
                out.push(IrEvent::ItemStart { index, kind, id, call });
            }
            "response.output_text.delta" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let delta = get_str(root, "delta").unwrap_or_default();
                out.push(IrEvent::Delta { index, delta: Delta::Text(delta.to_string()) });
            }
            "response.output_text.annotation.added" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                if let Some(a) = root.get("annotation") {
                    let kind = get_str(a, "type").unwrap_or("annotation");
                    out.push(IrEvent::Delta {
                        index,
                        delta: Delta::Annotation(Annotation::new(kind, a.clone())),
                    });
                }
            }
            "response.refusal.delta" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let delta = get_str(root, "delta").unwrap_or_default();
                out.push(IrEvent::Delta { index, delta: Delta::Refusal(delta.to_string()) });
            }
            "response.function_call_arguments.delta" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let delta = get_str(root, "delta").unwrap_or_default();
                self.tool_args.entry(index).or_default().push_str(delta);
                out.push(IrEvent::Delta { index, delta: Delta::ToolArgs(delta.to_string()) });
            }
            "response.function_call_arguments.done" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let full = get_str(root, "arguments").unwrap_or_default();
                let acc = self.tool_args.get(&index).cloned().unwrap_or_default();
                if acc == full {
                    // Already emitted in full.
                } else if let Some(rest) = full.strip_prefix(&acc) {
                    out.push(IrEvent::Delta { index, delta: Delta::ToolArgs(rest.to_string()) });
                    self.tool_args.insert(index, full.to_string());
                } else {
                    self.poisoned = true;
                    out.push(IrEvent::Error(XlateError::upstream_malformed(
                        "function_call_arguments.done disagrees with accumulated deltas",
                    )));
                }
            }
            "response.reasoning_summary_text.delta" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let part = get_u32(root, "summary_index").unwrap_or(0);
                let delta = get_str(root, "delta").unwrap_or_default();
                out.push(IrEvent::Delta {
                    index,
                    delta: Delta::ReasoningSummary { part, text: delta.to_string() },
                });
            }
            "response.reasoning_text.delta" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let delta = get_str(root, "delta").unwrap_or_default();
                out.push(IrEvent::Delta { index, delta: Delta::ReasoningText(delta.to_string()) });
            }
            "response.output_item.done" => {
                let index = get_u32(root, "output_index").unwrap_or(0);
                let item = root.get("item").cloned().unwrap_or(Value::Null);
                let item_ty = get_str(&item, "type").unwrap_or_default();
                match item_ty {
                    "message" | "function_call" => {}
                    "reasoning" => {
                        if let Some(enc) = get_str(&item, "encrypted_content") {
                            if !enc.is_empty() {
                                let mut blob =
                                    OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Encrypted, enc);
                                blob.model = Some(self.model.clone());
                                out.push(IrEvent::Delta { index, delta: Delta::Opaque(blob) });
                            }
                        }
                    }
                    "compaction" => {
                        let enc = get_str(&item, "encrypted_content").unwrap_or_default();
                        let mut blob =
                            OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Compaction, enc);
                        blob.model = Some(self.model.clone());
                        out.push(IrEvent::Delta { index, delta: Delta::Opaque(blob) });
                    }
                    _ => {
                        // Hosted tool item — pass the finished raw block through verbatim.
                        out.push(IrEvent::Delta {
                            index,
                            delta: Delta::ProviderRaw { family: ProviderFamily::OpenAI, raw: item },
                        });
                    }
                }
                out.push(IrEvent::ItemStop { index });
            }
            "response.completed" => {
                let resp = root.get("response").unwrap_or(&Value::Null);
                let usage = decode_usage(resp.get("usage"));
                let reason =
                    if self.has_function_call { StopReason::ToolUse } else { StopReason::EndTurn };
                out.push(IrEvent::Stop { reason, usage, ext: Extensions::new() });
            }
            "response.incomplete" => {
                let resp = root.get("response").unwrap_or(&Value::Null);
                let usage = decode_usage(resp.get("usage"));
                let reason = match resp
                    .get("incomplete_details")
                    .and_then(|d| get_str(d, "reason"))
                {
                    Some("content_filter") => StopReason::ContentFilter,
                    _ => StopReason::MaxTokens,
                };
                out.push(IrEvent::Stop { reason, usage, ext: Extensions::new() });
            }
            "response.failed" => {
                let resp = root.get("response").unwrap_or(&Value::Null);
                let err = resp
                    .get("error")
                    .map(error_from_value)
                    .unwrap_or_else(|| XlateError::upstream_malformed("response.failed without error"));
                self.poisoned = true;
                out.push(IrEvent::Error(err));
            }
            "error" => {
                self.poisoned = true;
                out.push(IrEvent::Error(error_from_value(root)));
            }
            // `response.content_part.added/done`, `*.done` text/summary/refusal events,
            // hosted-tool progress events, and unknown events: no IR effect.
            _ => {}
        }
    }
}

impl StreamDecoder for ResponsesStreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<IrEvent> {
        let events = self.parser.push(bytes);
        let mut out = Vec::new();
        for ev in events {
            self.on_event(&ev.data, &mut out);
        }
        out
    }

    fn finish(&mut self) -> Vec<IrEvent> {
        let events = self.parser.finish();
        let mut out = Vec::new();
        for ev in events {
            self.on_event(&ev.data, &mut out);
        }
        out
    }
}

/// Read a `u64` field permissively.
fn get_u64(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}
