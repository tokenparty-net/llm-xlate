//! [`AnthropicStreamDecoder`]: Anthropic Messages SSE -> [`IrEvent`] stream (provider-facing).
//!
//! Built on [`SseParser`]. Block indices on the wire become IR item indices 1:1. Provider-
//! hosted tool blocks (`server_tool_use`, `*_tool_result`) are assembled and forwarded as a
//! single [`Delta::ProviderRaw`] carrying the full block JSON (core's structured passthrough
//! carrier). Reasoning signatures / redacted data ride [`Delta::Opaque`]. Unknown event types
//! are ignored (forward-compatible); malformed JSON yields an [`IrEvent::Error`].

use std::collections::HashMap;

use serde_json::Value;

use llm_xlate_core::{
    Annotation, Delta, Extensions, IrEvent, ItemKind, OpaqueBlob, OpaqueKind, ResponseId,
    SseParser, StreamDecoder, Usage, XlateError,
};

use crate::shared::{stop_reason_from_wire, usage_from_wire};
use crate::wire::{ext_key, is_provider_tool_result_type, is_server_tool_use_type, FAMILY};

/// The wire kind of an open content block, tracked so deltas and the closing block are routed
/// correctly.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    Redacted,
    ToolCall,
    ProviderCall,
    ProviderResult,
    Compaction,
}

/// Per-block accumulation state.
struct BlockState {
    kind: BlockKind,
    /// The initial content_block JSON (provider / compaction blocks are re-emitted from it).
    raw: Value,
    /// Accumulated `input_json_delta` fragments for a provider block's `input`.
    json_buf: String,
}

/// A push-based Anthropic SSE decoder.
pub struct AnthropicStreamDecoder {
    parser: SseParser,
    blocks: HashMap<u32, BlockState>,
    prefill: Option<Usage>,
    /// The upstream model seen in `message_start`; stamped onto every native opaque blob so
    /// the reasoning/compaction carrier keeps its model binding across the envelope boundary.
    model: Option<String>,
}

impl AnthropicStreamDecoder {
    /// A fresh decoder.
    pub fn new() -> Self {
        Self { parser: SseParser::new(), blocks: HashMap::new(), prefill: None, model: None }
    }

    /// Build a native (Anthropic-family) opaque blob stamped with the upstream model.
    fn native_blob(&self, kind: OpaqueKind, data: impl Into<String>) -> OpaqueBlob {
        let mut b = OpaqueBlob::new(FAMILY, kind, data);
        b.model = self.model.clone();
        b
    }

    fn handle_event(&mut self, event: Option<&str>, data: &str, out: &mut Vec<IrEvent>) {
        let value = match serde_json::from_str::<Value>(data) {
            Ok(v) => v,
            Err(e) => {
                out.push(IrEvent::Error(XlateError::upstream_malformed(format!(
                    "malformed Anthropic SSE data: {e}"
                ))));
                return;
            }
        };
        let ty = event
            .map(str::to_string)
            .or_else(|| value.get("type").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();

        match ty.as_str() {
            "message_start" => self.on_message_start(&value, out),
            "content_block_start" => self.on_block_start(&value, out),
            "content_block_delta" => self.on_block_delta(&value, out),
            "content_block_stop" => self.on_block_stop(&value, out),
            "message_delta" => self.on_message_delta(&value, out),
            "message_stop" | "ping" => {}
            "error" => {
                let err = value.get("error").cloned().unwrap_or(Value::Null);
                out.push(IrEvent::Error(crate::errors::error_from_wire(&err)));
            }
            _ => {} // unknown event type: ignored (forward-compatible)
        }
    }

    fn on_message_start(&mut self, value: &Value, out: &mut Vec<IrEvent>) {
        let msg = value.get("message");
        let id = msg
            .and_then(|m| m.get("id"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let model = msg
            .and_then(|m| m.get("model"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let prefill = msg.and_then(|m| m.get("usage")).map(usage_from_wire);
        self.prefill = prefill.clone();
        self.model = if model.is_empty() { None } else { Some(model.clone()) };
        out.push(IrEvent::Start { response_id: ResponseId::new(id), model, usage_prefill: prefill });
    }

    fn on_block_start(&mut self, value: &Value, out: &mut Vec<IrEvent>) {
        let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
        let block = value.get("content_block").cloned().unwrap_or(Value::Null);
        let btype = block.get("type").and_then(Value::as_str).unwrap_or("");

        let (kind, kind_ir, call) = match btype {
            "text" => (BlockKind::Text, ItemKind::Message, None),
            "thinking" => (BlockKind::Thinking, ItemKind::Reasoning, None),
            "redacted_thinking" => (BlockKind::Redacted, ItemKind::Reasoning, None),
            "tool_use" => {
                let id = block.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                let name = block.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                (BlockKind::ToolCall, ItemKind::ToolCall, Some((id.into(), name)))
            }
            "compaction" => (BlockKind::Compaction, ItemKind::Compaction, None),
            t if is_server_tool_use_type(t) => {
                (BlockKind::ProviderCall, ItemKind::ProviderToolCall, None)
            }
            t if is_provider_tool_result_type(t) => {
                (BlockKind::ProviderResult, ItemKind::ProviderToolResult, None)
            }
            _ => (BlockKind::ProviderCall, ItemKind::ProviderToolCall, None),
        };

        out.push(IrEvent::ItemStart { index, kind: kind_ir, id: None, call });

        // A redacted_thinking block start immediately carries its opaque data.
        if kind == BlockKind::Redacted {
            let data = block.get("data").and_then(Value::as_str).unwrap_or("").to_string();
            out.push(IrEvent::Delta {
                index,
                delta: Delta::Opaque(self.native_blob(OpaqueKind::Redacted, data)),
            });
        }

        self.blocks.insert(index, BlockState { kind, raw: block, json_buf: String::new() });
    }

    fn on_block_delta(&mut self, value: &Value, out: &mut Vec<IrEvent>) {
        let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
        let delta = match value.get("delta") {
            Some(d) => d,
            None => return,
        };
        let dtype = delta.get("type").and_then(Value::as_str).unwrap_or("");
        let kind = self.blocks.get(&index).map(|b| b.kind);

        match dtype {
            "text_delta" => {
                let text = delta.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                out.push(IrEvent::Delta { index, delta: Delta::Text(text) });
            }
            "input_json_delta" => {
                let pj = delta.get("partial_json").and_then(Value::as_str).unwrap_or("");
                match kind {
                    Some(BlockKind::ToolCall) => {
                        out.push(IrEvent::Delta { index, delta: Delta::ToolArgs(pj.to_string()) });
                    }
                    _ => {
                        if let Some(b) = self.blocks.get_mut(&index) {
                            b.json_buf.push_str(pj);
                        }
                    }
                }
            }
            "thinking_delta" => {
                let text = delta.get("thinking").and_then(Value::as_str).unwrap_or("").to_string();
                out.push(IrEvent::Delta { index, delta: Delta::ReasoningText(text) });
            }
            "signature_delta" => {
                let sig = delta.get("signature").and_then(Value::as_str).unwrap_or("").to_string();
                out.push(IrEvent::Delta {
                    index,
                    delta: Delta::Opaque(self.native_blob(OpaqueKind::Signature, sig)),
                });
            }
            "citations_delta" => {
                let citation = delta.get("citation").cloned().unwrap_or(Value::Null);
                let ckind = citation
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("citation")
                    .to_string();
                out.push(IrEvent::Delta {
                    index,
                    delta: Delta::Annotation(Annotation::new(ckind, citation)),
                });
            }
            _ => {}
        }
    }

    fn on_block_stop(&mut self, value: &Value, out: &mut Vec<IrEvent>) {
        let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
        if let Some(state) = self.blocks.remove(&index) {
            match state.kind {
                BlockKind::ProviderCall | BlockKind::ProviderResult => {
                    let mut raw = state.raw;
                    if !state.json_buf.is_empty() {
                        if let Ok(input) = serde_json::from_str::<Value>(&state.json_buf) {
                            if let Some(obj) = raw.as_object_mut() {
                                obj.insert("input".to_string(), input);
                            }
                        }
                    }
                    out.push(IrEvent::Delta {
                        index,
                        delta: Delta::ProviderRaw { family: FAMILY, raw },
                    });
                }
                BlockKind::Compaction => {
                    let blob = self.native_blob(OpaqueKind::Compaction, state.raw.to_string());
                    out.push(IrEvent::Delta { index, delta: Delta::Opaque(blob) });
                }
                BlockKind::Text
                | BlockKind::Thinking
                | BlockKind::Redacted
                | BlockKind::ToolCall => {}
            }
        }
        out.push(IrEvent::ItemStop { index });
    }

    fn on_message_delta(&mut self, value: &Value, out: &mut Vec<IrEvent>) {
        let delta = value.get("delta");
        let reason = delta.and_then(|d| d.get("stop_reason")).and_then(Value::as_str);
        let stop_sequence = delta.and_then(|d| d.get("stop_sequence")).and_then(Value::as_str);
        let stop = stop_reason_from_wire(reason, stop_sequence);

        let mut usage = self.prefill.clone().unwrap_or_default();
        if let Some(u) = value.get("usage") {
            let md = usage_from_wire(u);
            usage.output = md.output;
            if md.input > 0 {
                usage.input = md.input;
            }
            if md.cache_read.is_some() {
                usage.cache_read = md.cache_read;
            }
            if md.cache_write.is_some() {
                usage.cache_write = md.cache_write;
            }
            if md.cache_write_1h.is_some() {
                usage.cache_write_1h = md.cache_write_1h;
            }
            if md.reasoning.is_some() {
                usage.reasoning = md.reasoning;
            }
            // Carry any preserved extra usage fields (e.g. `service_tier`, `server_tool_use`).
            for (k, v) in md.ext.iter() {
                usage.ext.insert(k.clone(), v.clone());
            }
            usage.enforce_invariants();
        }

        let mut ext = Extensions::new();
        if let Some(d) = delta {
            for key in ["container", "context_management"] {
                if let Some(v) = d.get(key) {
                    if !v.is_null() {
                        ext.insert(ext_key(key), v.clone());
                    }
                }
            }
        }

        out.push(IrEvent::Stop { reason: stop, usage, ext });
    }
}

impl Default for AnthropicStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder for AnthropicStreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<IrEvent> {
        let events = self.parser.push(bytes);
        let mut out = Vec::new();
        for ev in events {
            self.handle_event(ev.event.as_deref(), &ev.data, &mut out);
        }
        out
    }

    fn finish(&mut self) -> Vec<IrEvent> {
        let events = self.parser.finish();
        let mut out = Vec::new();
        for ev in events {
            self.handle_event(ev.event.as_deref(), &ev.data, &mut out);
        }
        out
    }
}
