//! [`ChatStreamDecoder`]: a Chat Completions SSE stream (`chat.completion.chunk`) → [`IrEvent`]
//! push state machine (plan §8), built on [`SseParser`].
//!
//! State machine over `choices[0].delta`: the first chunk emits [`IrEvent::Start`];
//! `delta.content` opens/continues a Message item; `delta.refusal` adds a refusal delta;
//! `delta.reasoning_content`/`delta.reasoning` open/continue a Reasoning item (closed when the
//! first content/tool-call arrives); `delta.tool_calls[j]` with an `id` opens a ToolCall item
//! (chat `index` → IR index) and `function.arguments` become [`Delta::ToolArgs`];
//! `delta.reasoning_details` entries become [`Delta::Opaque`] on the reasoning item;
//! `finish_reason` closes all open items and records the stop reason; a usage chunk (or usage
//! on the final chunk) yields [`IrEvent::Stop`]; `data: [DONE]` yields `Stop` if not already
//! emitted. Choices with index != 0 are ignored (`n>1` unsupported). A malformed chunk yields
//! [`IrEvent::Error`]. Provider-facing: opaque blobs are native (`family = OpenAI`).

use serde_json::Value;

use llm_xlate_core::canon;
use llm_xlate_core::ir::{
    CallId, Delta, Extensions, IrEvent, ItemKind, OpaqueBlob, OpaqueKind, ProviderFamily,
    ResponseId, StopReason, Usage,
};
use llm_xlate_core::{SseParser, XlateError};

use crate::common::{decode_usage, finish_reason_to_stop, ns_key};
use crate::wire::{WireChunk, WireDelta};

/// A Chat Completions streaming decoder.
pub struct ChatStreamDecoder {
    parser: SseParser,
    started: bool,
    stopped: bool,
    model: String,
    next_index: u32,
    open_reasoning: Option<u32>,
    open_message: Option<u32>,
    /// Chat tool-call `index` → IR item index.
    tool_map: Vec<(u32, u32)>,
    /// All currently-open item indices, in open order (closed at `finish_reason`/end).
    open_stack: Vec<u32>,
    has_refusal: bool,
    finish_reason: Option<StopReason>,
    service_tier: Option<String>,
    /// The most recent `usage` object seen on any chunk. Providers differ in where they put it —
    /// OpenAI sends one final choices-empty chunk, vLLM attaches a running total to EVERY chunk
    /// (`completion_tokens: 0` on the first) *and* sends a final choices-empty chunk that is the
    /// only one carrying `prompt_tokens_details`. The last object seen always wins, and it is
    /// applied at `[DONE]` / end of stream rather than when `finish_reason` arrives.
    last_usage: Option<Usage>,
}

impl ChatStreamDecoder {
    /// A fresh decoder.
    pub fn new() -> Self {
        Self {
            parser: SseParser::new(),
            started: false,
            stopped: false,
            model: String::new(),
            next_index: 0,
            open_reasoning: None,
            open_message: None,
            tool_map: Vec::new(),
            open_stack: Vec::new(),
            has_refusal: false,
            finish_reason: None,
            service_tier: None,
            last_usage: None,
        }
    }

    fn handle_data(&mut self, data: &str, out: &mut Vec<IrEvent>) {
        // Once an error frame has poisoned the stream (or a normal Stop has been emitted), ignore
        // any trailing data so a mid-stream error cannot be followed by fabricated content.
        if self.stopped {
            return;
        }
        let data = data.trim();
        if data.is_empty() {
            return;
        }
        if data == "[DONE]" {
            let usage = self.last_usage.take().unwrap_or_default();
            self.finalize(out, usage);
            return;
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => {
                out.push(IrEvent::Error(XlateError::upstream_malformed(
                    "malformed chat.completion.chunk",
                )));
                return;
            }
        };
        // A mid-stream upstream error frame: `data: {"error": {...}}`. Some gateways emit the error
        // as an SSE data frame after a 200 + partial content. Surface it as an [`IrEvent::Error`]
        // (so the router renders a client error frame and stops) rather than letting it deserialize
        // into an empty chunk and silently truncating the turn with a fabricated `finish_reason`.
        if let Some(err_obj) = value.get("error") {
            if !err_obj.is_null() {
                out.push(IrEvent::Error(crate::errors::error_from_value(err_obj)));
                self.stopped = true;
                return;
            }
        }
        let chunk: WireChunk = match serde_json::from_value(value) {
            Ok(c) => c,
            Err(_) => {
                out.push(IrEvent::Error(XlateError::upstream_malformed(
                    "malformed chat.completion.chunk",
                )));
                return;
            }
        };

        if !self.started {
            self.started = true;
            self.model = chunk.model.clone();
            out.push(IrEvent::Start {
                response_id: ResponseId::new(chunk.id.clone()),
                model: chunk.model.clone(),
                usage_prefill: None,
            });
        }
        if let Some(st) = &chunk.service_tier {
            self.service_tier = Some(st.clone());
        }

        if let Some(choice) = chunk.choices.iter().find(|c| c.index == 0) {
            self.handle_delta(&choice.delta, out);
            if let Some(fr) = &choice.finish_reason {
                self.close_all(out);
                self.finish_reason = Some(finish_reason_to_stop(Some(fr), self.has_refusal));
            }
        }

        if let Some(u) = &chunk.usage {
            // Never finalize from inside a usage chunk: the *last* usage object a provider
            // sends is the authoritative one, and it can arrive after the chunk that carried
            // `finish_reason`. vLLM attaches a running usage to every chunk and then sends one
            // final choices-empty chunk that is the only one carrying
            // `prompt_tokens_details` — finalizing on the finish chunk discarded exactly the
            // cache counters this codec exists to preserve. Stop is emitted at `[DONE]` or at
            // end of stream instead; items already closed at `finish_reason`, so nothing else
            // moves.
            self.last_usage = Some(decode_usage(u));
        }
    }

    fn handle_delta(&mut self, delta: &WireDelta, out: &mut Vec<IrEvent>) {
        // reasoning text
        if let Some(rt) = delta.reasoning_content.as_deref().or(delta.reasoning.as_deref()) {
            if !rt.is_empty() {
                let idx = self.ensure_reasoning(out);
                out.push(IrEvent::Delta { index: idx, delta: Delta::ReasoningText(rt.to_string()) });
            }
        }
        // reasoning_details → opaque carriers on the reasoning item
        if let Some(details) = &delta.reasoning_details {
            for entry in details {
                let idx = self.ensure_reasoning(out);
                let blob = self.opaque_from_detail(entry);
                out.push(IrEvent::Delta { index: idx, delta: Delta::Opaque(blob) });
            }
        }
        // content text
        if let Some(c) = &delta.content {
            if !c.is_empty() {
                self.close_reasoning(out);
                let idx = self.ensure_message(out);
                out.push(IrEvent::Delta { index: idx, delta: Delta::Text(c.clone()) });
            }
        }
        // refusal
        if let Some(r) = &delta.refusal {
            if !r.is_empty() {
                self.has_refusal = true;
                self.close_reasoning(out);
                let idx = self.ensure_message(out);
                out.push(IrEvent::Delta { index: idx, delta: Delta::Refusal(r.clone()) });
            }
        }
        // tool calls
        if let Some(tcs) = &delta.tool_calls {
            for tc in tcs {
                self.close_reasoning(out);
                self.close_message(out);
                let chat_idx = tc.index.unwrap_or(0);
                let has_id = tc.id.as_deref().is_some_and(|s| !s.is_empty());
                if has_id {
                    let ir_idx = self.next_index;
                    self.next_index += 1;
                    self.open_stack.push(ir_idx);
                    self.tool_map.push((chat_idx, ir_idx));
                    let name = tc.function.name.clone().unwrap_or_default();
                    let id = tc.id.clone().unwrap_or_default();
                    out.push(IrEvent::ItemStart {
                        index: ir_idx,
                        kind: ItemKind::ToolCall,
                        id: None,
                        call: Some((CallId::new(id), name)),
                    });
                    if let Some(args) = &tc.function.arguments {
                        if !args.is_empty() {
                            out.push(IrEvent::Delta {
                                index: ir_idx,
                                delta: Delta::ToolArgs(args.clone()),
                            });
                        }
                    }
                } else if let Some(ir_idx) = self.tool_map_get(chat_idx) {
                    if let Some(args) = &tc.function.arguments {
                        if !args.is_empty() {
                            out.push(IrEvent::Delta {
                                index: ir_idx,
                                delta: Delta::ToolArgs(args.clone()),
                            });
                        }
                    }
                }
            }
        }
    }

    fn opaque_from_detail(&self, entry: &Value) -> OpaqueBlob {
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
            model: if self.model.is_empty() { None } else { Some(self.model.clone()) },
        }
    }

    fn tool_map_get(&self, chat_idx: u32) -> Option<u32> {
        self.tool_map.iter().find(|(c, _)| *c == chat_idx).map(|(_, ir)| *ir)
    }

    fn ensure_reasoning(&mut self, out: &mut Vec<IrEvent>) -> u32 {
        if let Some(idx) = self.open_reasoning {
            return idx;
        }
        let idx = self.next_index;
        self.next_index += 1;
        self.open_reasoning = Some(idx);
        self.open_stack.push(idx);
        out.push(IrEvent::ItemStart { index: idx, kind: ItemKind::Reasoning, id: None, call: None });
        idx
    }

    fn ensure_message(&mut self, out: &mut Vec<IrEvent>) -> u32 {
        if let Some(idx) = self.open_message {
            return idx;
        }
        let idx = self.next_index;
        self.next_index += 1;
        self.open_message = Some(idx);
        self.open_stack.push(idx);
        out.push(IrEvent::ItemStart { index: idx, kind: ItemKind::Message, id: None, call: None });
        idx
    }

    fn close_reasoning(&mut self, out: &mut Vec<IrEvent>) {
        if let Some(idx) = self.open_reasoning.take() {
            self.remove_open(idx);
            out.push(IrEvent::ItemStop { index: idx });
        }
    }

    fn close_message(&mut self, out: &mut Vec<IrEvent>) {
        if let Some(idx) = self.open_message.take() {
            self.remove_open(idx);
            out.push(IrEvent::ItemStop { index: idx });
        }
    }

    fn remove_open(&mut self, idx: u32) {
        if let Some(pos) = self.open_stack.iter().position(|&i| i == idx) {
            self.open_stack.remove(pos);
        }
    }

    fn close_all(&mut self, out: &mut Vec<IrEvent>) {
        for idx in std::mem::take(&mut self.open_stack) {
            out.push(IrEvent::ItemStop { index: idx });
        }
        self.open_reasoning = None;
        self.open_message = None;
    }

    fn finalize(&mut self, out: &mut Vec<IrEvent>, usage: Usage) {
        if self.stopped {
            return;
        }
        self.close_all(out);
        let reason = self.finish_reason.clone().unwrap_or(StopReason::EndTurn);
        let mut ext = Extensions::new();
        if let Some(st) = &self.service_tier {
            ext.insert(ns_key("service_tier"), Value::from(st.clone()));
        }
        if reason == StopReason::Refusal {
            let mut sd = serde_json::Map::new();
            sd.insert("type".into(), Value::from("refusal"));
            ext.insert("stop_details".to_string(), Value::Object(sd));
        }
        out.push(IrEvent::Stop { reason, usage, ext });
        self.stopped = true;
    }
}

impl Default for ChatStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl llm_xlate_core::StreamDecoder for ChatStreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<IrEvent> {
        let mut out = Vec::new();
        for ev in self.parser.push(bytes) {
            self.handle_data(&ev.data, &mut out);
        }
        out
    }

    fn finish(&mut self) -> Vec<IrEvent> {
        let mut out = Vec::new();
        for ev in self.parser.finish() {
            self.handle_data(&ev.data, &mut out);
        }
        if !self.stopped {
            let usage = self.last_usage.take().unwrap_or_default();
            self.finalize(&mut out, usage);
        }
        out
    }
}
