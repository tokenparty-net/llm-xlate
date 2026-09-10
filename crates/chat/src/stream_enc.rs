//! [`ChatStreamEncoder`]: [`IrEvent`] → client Chat Completions SSE frames (plan §8).
//!
//! Every frame is a `chat.completion.chunk` object with `id = ctx.response_id`,
//! `created = ctx.created_at`, `model = ctx.client_model`, and a single
//! `choices[{index:0, delta, finish_reason}]`. The first emitted chunk carries
//! `delta.role:"assistant"` (and `content:""`, like OpenAI). Text → `delta.content`; Refusal →
//! `delta.refusal`; a ToolCall start → a `delta.tool_calls` head (with the tool ordinal as
//! `index`); ToolArgs → `delta.tool_calls[].function.arguments`; ReasoningText / ReasoningSummary
//! → `delta.reasoning_content`; Opaque is buffered and sealed into `delta.reasoning_details` at
//! `ItemStop`; Annotation → `delta.annotations`. `Stop` emits a `finish_reason` chunk, an
//! optional usage chunk (when `ctx.include_usage`), and `data: [DONE]`. `Error` emits an error
//! chunk then `[DONE]`. `keepalive()` emits an SSE comment.

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::canon;
use llm_xlate_core::codec::EncodeCtx;
use llm_xlate_core::ir::{Delta, IrEvent, ItemKind, OpaqueBlob, StopReason};
use llm_xlate_core::SseWriter;

use crate::common::{encode_usage, stop_to_finish_reason};
use crate::errors::error_body_value;

/// A Chat Completions streaming encoder.
pub struct ChatStreamEncoder {
    ctx: EncodeCtx,
    started: bool,
    done: bool,
    kinds: Vec<(u32, ItemKind)>,
    opaque_buf: Vec<(u32, OpaqueBlob)>,
    tool_ord: Vec<(u32, u32)>,
    next_tool_ord: u32,
    last_summary_part: Option<u32>,
}

impl ChatStreamEncoder {
    /// A fresh encoder bound to a response context.
    pub fn new(ctx: EncodeCtx) -> Self {
        Self {
            ctx,
            started: false,
            done: false,
            kinds: Vec::new(),
            opaque_buf: Vec::new(),
            tool_ord: Vec::new(),
            next_tool_ord: 0,
            last_summary_part: None,
        }
    }

    fn kind_of(&self, index: u32) -> Option<ItemKind> {
        self.kinds.iter().find(|(i, _)| *i == index).map(|(_, k)| *k)
    }

    fn tool_ordinal(&self, index: u32) -> Option<u32> {
        self.tool_ord.iter().find(|(i, _)| *i == index).map(|(_, n)| *n)
    }

    /// Build a `chat.completion.chunk` frame carrying `delta` and `finish_reason`.
    fn chunk(&self, delta: Value, finish_reason: Value) -> Bytes {
        let mut choice = Map::new();
        choice.insert("index".into(), Value::from(0));
        choice.insert("delta".into(), delta);
        choice.insert("finish_reason".into(), finish_reason);
        self.envelope(vec![Value::Object(choice)], None)
    }

    /// Build the chunk envelope with the given choices and optional usage.
    fn envelope(&self, choices: Vec<Value>, usage: Option<Value>) -> Bytes {
        let mut obj = Map::new();
        obj.insert("id".into(), Value::from(self.ctx.response_id.as_str()));
        obj.insert("object".into(), Value::from("chat.completion.chunk"));
        obj.insert("created".into(), Value::from(self.ctx.created_at));
        obj.insert("model".into(), Value::from(self.ctx.client_model.clone()));
        obj.insert("choices".into(), Value::Array(choices));
        if let Some(u) = usage {
            obj.insert("usage".into(), u);
        }
        SseWriter::frame(None, &canon::to_string(&Value::Object(obj)))
    }

    /// Emit the initial `role:"assistant"` chunk if it has not been emitted yet.
    fn ensure_started(&mut self, out: &mut Vec<Bytes>) {
        if self.started {
            return;
        }
        self.started = true;
        let mut delta = Map::new();
        delta.insert("role".into(), Value::from("assistant"));
        delta.insert("content".into(), Value::from(""));
        out.push(self.chunk(Value::Object(delta), Value::Null));
    }

    fn on_item_start(
        &mut self,
        index: u32,
        kind: ItemKind,
        call: Option<(llm_xlate_core::ir::CallId, String)>,
        out: &mut Vec<Bytes>,
    ) {
        self.kinds.push((index, kind));
        if kind == ItemKind::ToolCall {
            self.ensure_started(out);
            let n = self.next_tool_ord;
            self.next_tool_ord += 1;
            self.tool_ord.push((index, n));
            let (call_id, name) = call.unwrap_or_else(|| (llm_xlate_core::ir::CallId::new(""), String::new()));
            let mut func = Map::new();
            func.insert("name".into(), Value::from(name));
            func.insert("arguments".into(), Value::from(""));
            let mut tc = Map::new();
            tc.insert("index".into(), Value::from(n));
            tc.insert("id".into(), Value::from(call_id.as_str()));
            tc.insert("type".into(), Value::from("function"));
            tc.insert("function".into(), Value::Object(func));
            let mut delta = Map::new();
            delta.insert("tool_calls".into(), Value::Array(vec![Value::Object(tc)]));
            out.push(self.chunk(Value::Object(delta), Value::Null));
        }
    }

    fn on_delta(&mut self, index: u32, delta: Delta, out: &mut Vec<Bytes>) {
        match delta {
            Delta::Text(s) => {
                self.ensure_started(out);
                let mut d = Map::new();
                d.insert("content".into(), Value::from(s));
                out.push(self.chunk(Value::Object(d), Value::Null));
            }
            Delta::Refusal(s) => {
                self.ensure_started(out);
                let mut d = Map::new();
                d.insert("refusal".into(), Value::from(s));
                out.push(self.chunk(Value::Object(d), Value::Null));
            }
            Delta::ReasoningText(s) => {
                self.ensure_started(out);
                let mut d = Map::new();
                d.insert("reasoning_content".into(), Value::from(s));
                out.push(self.chunk(Value::Object(d), Value::Null));
            }
            Delta::ReasoningSummary { part, text } => {
                self.ensure_started(out);
                let mut s = String::new();
                if self.last_summary_part.is_some_and(|p| p != part) {
                    s.push_str("\n\n");
                }
                self.last_summary_part = Some(part);
                s.push_str(&text);
                let mut d = Map::new();
                d.insert("reasoning_content".into(), Value::from(s));
                out.push(self.chunk(Value::Object(d), Value::Null));
            }
            Delta::ToolArgs(s) => {
                if let Some(n) = self.tool_ordinal(index) {
                    self.ensure_started(out);
                    let mut func = Map::new();
                    func.insert("arguments".into(), Value::from(s));
                    let mut tc = Map::new();
                    tc.insert("index".into(), Value::from(n));
                    tc.insert("function".into(), Value::Object(func));
                    let mut d = Map::new();
                    d.insert("tool_calls".into(), Value::Array(vec![Value::Object(tc)]));
                    out.push(self.chunk(Value::Object(d), Value::Null));
                }
            }
            Delta::Opaque(blob) => {
                // Buffer (concatenate) until ItemStop, then seal into reasoning_details.
                if let Some(entry) = self.opaque_buf.iter_mut().find(|(i, _)| *i == index) {
                    entry.1.data.push_str(&blob.data);
                } else {
                    self.opaque_buf.push((index, blob));
                }
            }
            Delta::Annotation(a) => {
                self.ensure_started(out);
                let mut d = Map::new();
                d.insert("annotations".into(), Value::Array(vec![a.raw]));
                out.push(self.chunk(Value::Object(d), Value::Null));
            }
            Delta::ProviderRaw { .. } => { /* no Chat carrier for provider-hosted tools */ }
        }
    }

    fn on_item_stop(&mut self, index: u32, out: &mut Vec<Bytes>) {
        if let Some(pos) = self.opaque_buf.iter().position(|(i, _)| *i == index) {
            let (_, blob) = self.opaque_buf.remove(pos);
            self.ensure_started(out);
            let sealed = self.ctx.sealer.seal(&blob);
            let mut entry = Map::new();
            entry.insert("type".into(), Value::from("router.opaque"));
            entry.insert("data".into(), Value::from(sealed));
            let mut d = Map::new();
            d.insert("reasoning_details".into(), Value::Array(vec![Value::Object(entry)]));
            out.push(self.chunk(Value::Object(d), Value::Null));
        }
        let _ = self.kind_of(index);
    }

    fn on_stop(&mut self, reason: &StopReason, usage: &llm_xlate_core::ir::Usage, out: &mut Vec<Bytes>) {
        self.ensure_started(out);
        let finish = stop_to_finish_reason(reason);
        out.push(self.chunk(Value::Object(Map::new()), Value::from(finish)));
        if self.ctx.include_usage {
            out.push(self.envelope(Vec::new(), Some(encode_usage(usage))));
        }
    }

    fn on_error(&mut self, e: &llm_xlate_core::XlateError, out: &mut Vec<Bytes>) {
        self.started = true;
        let body = error_body_value(e);
        out.push(SseWriter::frame(None, &canon::to_string(&body)));
    }
}

impl llm_xlate_core::StreamEncoder for ChatStreamEncoder {
    fn push(&mut self, ev: IrEvent) -> Vec<Bytes> {
        let mut out = Vec::new();
        match ev {
            IrEvent::Start { .. } => {
                self.ensure_started(&mut out);
            }
            IrEvent::ItemStart { index, kind, call, .. } => {
                self.on_item_start(index, kind, call, &mut out);
            }
            IrEvent::Delta { index, delta } => {
                self.on_delta(index, delta, &mut out);
            }
            IrEvent::ItemStop { index } => {
                self.on_item_stop(index, &mut out);
            }
            IrEvent::Stop { reason, usage, .. } => {
                self.on_stop(&reason, &usage, &mut out);
            }
            IrEvent::Error(e) => {
                self.on_error(&e, &mut out);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<Bytes> {
        let mut out = Vec::new();
        if !self.done {
            self.done = true;
            if self.started {
                out.push(SseWriter::frame(None, "[DONE]"));
            }
        }
        out
    }

    fn keepalive(&mut self) -> Option<Bytes> {
        Some(SseWriter::comment("keepalive"))
    }
}
