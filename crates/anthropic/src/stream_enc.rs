//! [`AnthropicStreamEncoder`]: [`IrEvent`] stream -> Anthropic Messages SSE (client-facing).
//!
//! Anthropic block indices must be dense (`0..n`), while IR indices may skip, so a running
//! counter assigns a fresh Anthropic index to each block as it is emitted. Streamed items
//! (text / tool calls) are rendered incrementally; reasoning, provider-hosted tool, and
//! compaction items are buffered until their [`IrEvent::ItemStop`] because the block *type*
//! (thinking vs redacted_thinking) and full payload are only known once their deltas arrive.
//! Opaque carriers bound to a foreign family are sealed into `rtr1.` envelopes on the way out.

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::{
    Delta, EncodeCtx, IrEvent, ItemKind, OpaqueBlob, ProviderFamily, SseWriter,
    StopReason, StreamEncoder, Usage,
};

use crate::shared::{seal_or_native, stop_reason_to_wire, usage_to_wire};
use crate::wire::{omap, EXT_PREFIX, FAMILY};

/// A streamed item that renders incrementally.
enum StreamKind {
    Text,
    Tool,
}

/// A buffered reasoning / provider / compaction item, flushed at `ItemStop`.
#[derive(Default)]
struct Buffered {
    is_reasoning: bool,
    is_provider: bool,
    text: String,
    summaries: Vec<(u32, String)>,
    opaque: Option<OpaqueBlob>,
    raw: Option<Value>,
    /// The provider family that owns a buffered provider-hosted tool block (from
    /// [`Delta::ProviderRaw`]). A foreign-family block has no valid Anthropic wire shape and is
    /// dropped in [`AnthropicStreamEncoder::flush_buffered`], mirroring `encode.rs`'s
    /// `oi.family == FAMILY` gate on the non-streaming path.
    family: Option<ProviderFamily>,
}

/// Per-IR-index state.
enum ItemState {
    Streamed { ant_index: u32, kind: StreamKind },
    Buffered(Box<Buffered>),
}

/// A push-based Anthropic SSE encoder.
pub struct AnthropicStreamEncoder {
    ctx: EncodeCtx,
    next_index: u32,
    items: Vec<(u32, ItemState)>,
    /// The IR index of the currently-open *streamed* content block, if any. Anthropic permits
    /// only one open content block at a time, so this must be closed (its `content_block_stop`
    /// emitted) before any new block is started.
    open_streamed: Option<u32>,
    started: bool,
    stopped: bool,
    /// The `input_tokens` reported at `message_start` (0 when the provider gives usage only at
    /// the end, e.g. Chat / Responses bridged to an Anthropic client).
    start_input: u32,
}

impl AnthropicStreamEncoder {
    /// A fresh encoder bound to a response context.
    pub fn new(ctx: EncodeCtx) -> Self {
        Self {
            ctx,
            next_index: 0,
            items: Vec::new(),
            open_streamed: None,
            started: false,
            stopped: false,
            start_input: 0,
        }
    }

    /// Close the currently-open streamed content block, if any, emitting its
    /// `content_block_stop` and removing its state (so the later [`IrEvent::ItemStop`] for that
    /// IR index becomes an idempotent no-op). Anthropic's wire contract requires each content
    /// block to be fully delimited (`start … stop`) before the next one opens.
    fn close_open_streamed(&mut self, out: &mut Vec<Bytes>) {
        let Some(open) = self.open_streamed.take() else { return };
        if let Some(ItemState::Streamed { ant_index, .. }) = self.take(open) {
            out.push(self.block_stop(ant_index));
        }
    }

    fn frame(ty: &str, value: &Value) -> Bytes {
        SseWriter::frame(Some(ty), &llm_xlate_core::canon::to_string(value))
    }

    fn take(&mut self, index: u32) -> Option<ItemState> {
        let pos = self.items.iter().position(|(i, _)| *i == index)?;
        Some(self.items.remove(pos).1)
    }

    fn get_mut(&mut self, index: u32) -> Option<&mut ItemState> {
        self.items.iter_mut().find(|(i, _)| *i == index).map(|(_, s)| s)
    }

    fn on_start(&mut self, usage_prefill: Option<Usage>, out: &mut Vec<Bytes>) {
        self.started = true;
        let usage = usage_prefill.unwrap_or_default();
        self.start_input = usage.input;
        let mut msg = omap();
        msg.insert("id".into(), Value::from(self.ctx.response_id.as_str()));
        msg.insert("type".into(), Value::from("message"));
        msg.insert("role".into(), Value::from("assistant"));
        msg.insert("model".into(), Value::from(self.ctx.client_model.clone()));
        msg.insert("content".into(), Value::Array(Vec::new()));
        msg.insert("stop_reason".into(), Value::Null);
        msg.insert("stop_sequence".into(), Value::Null);
        msg.insert("usage".into(), usage_to_wire(&usage));

        let mut root = omap();
        root.insert("type".into(), Value::from("message_start"));
        root.insert("message".into(), Value::Object(msg));
        out.push(Self::frame("message_start", &Value::Object(root)));
    }

    fn on_item_start(
        &mut self,
        index: u32,
        kind: ItemKind,
        call: Option<(llm_xlate_core::CallId, String)>,
        out: &mut Vec<Bytes>,
    ) {
        match kind {
            ItemKind::Message => {
                self.close_open_streamed(out);
                let ant = self.alloc_index();
                let mut block = omap();
                block.insert("type".into(), Value::from("text"));
                block.insert("text".into(), Value::from(""));
                out.push(self.block_start(ant, Value::Object(block)));
                self.items.push((index, ItemState::Streamed { ant_index: ant, kind: StreamKind::Text }));
                self.open_streamed = Some(index);
            }
            ItemKind::ToolCall => {
                self.close_open_streamed(out);
                let ant = self.alloc_index();
                let (call_id, name) = call.unwrap_or_else(|| ("".into(), String::new()));
                let mut block = omap();
                block.insert("type".into(), Value::from("tool_use"));
                block.insert("id".into(), Value::from(call_id.as_str()));
                block.insert("name".into(), Value::from(name));
                block.insert("input".into(), Value::Object(Map::new()));
                out.push(self.block_start(ant, Value::Object(block)));
                self.items.push((index, ItemState::Streamed { ant_index: ant, kind: StreamKind::Tool }));
                self.open_streamed = Some(index);
            }
            ItemKind::Reasoning => {
                self.items.push((
                    index,
                    ItemState::Buffered(Box::new(Buffered { is_reasoning: true, ..Default::default() })),
                ));
            }
            ItemKind::ProviderToolCall | ItemKind::ProviderToolResult => {
                self.items.push((
                    index,
                    ItemState::Buffered(Box::new(Buffered { is_provider: true, ..Default::default() })),
                ));
            }
            ItemKind::Compaction => {
                self.items.push((index, ItemState::Buffered(Box::<Buffered>::default())));
            }
        }
    }

    fn alloc_index(&mut self) -> u32 {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    fn block_start(&self, ant: u32, block: Value) -> Bytes {
        let mut root = omap();
        root.insert("type".into(), Value::from("content_block_start"));
        root.insert("index".into(), Value::from(ant));
        root.insert("content_block".into(), block);
        Self::frame("content_block_start", &Value::Object(root))
    }

    fn block_delta(&self, ant: u32, delta: Value) -> Bytes {
        let mut root = omap();
        root.insert("type".into(), Value::from("content_block_delta"));
        root.insert("index".into(), Value::from(ant));
        root.insert("delta".into(), delta);
        Self::frame("content_block_delta", &Value::Object(root))
    }

    fn block_stop(&self, ant: u32) -> Bytes {
        let mut root = omap();
        root.insert("type".into(), Value::from("content_block_stop"));
        root.insert("index".into(), Value::from(ant));
        Self::frame("content_block_stop", &Value::Object(root))
    }

    fn on_delta(&mut self, index: u32, delta: Delta, out: &mut Vec<Bytes>) {
        // Streamed items render immediately; buffered items accumulate.
        let ant_and_kind = match self.get_mut(index) {
            Some(ItemState::Streamed { ant_index, kind }) => Some((*ant_index, matches!(kind, StreamKind::Text))),
            Some(ItemState::Buffered(buf)) => {
                Self::accumulate(buf, delta);
                return;
            }
            None => None,
        };
        let (ant, is_text) = match ant_and_kind {
            Some(v) => v,
            None => return,
        };
        if is_text {
            match delta {
                Delta::Text(t) | Delta::Refusal(t) => {
                    let mut d = omap();
                    d.insert("type".into(), Value::from("text_delta"));
                    d.insert("text".into(), Value::from(t));
                    out.push(self.block_delta(ant, Value::Object(d)));
                }
                Delta::Annotation(a) => {
                    let mut d = omap();
                    d.insert("type".into(), Value::from("citations_delta"));
                    d.insert("citation".into(), a.raw);
                    out.push(self.block_delta(ant, Value::Object(d)));
                }
                _ => {}
            }
        } else if let Delta::ToolArgs(t) = delta {
            let mut d = omap();
            d.insert("type".into(), Value::from("input_json_delta"));
            d.insert("partial_json".into(), Value::from(t));
            out.push(self.block_delta(ant, Value::Object(d)));
        }
    }

    fn accumulate(buf: &mut Buffered, delta: Delta) {
        match delta {
            Delta::ReasoningText(t) => buf.text.push_str(&t),
            Delta::ReasoningSummary { part, text } => {
                match buf.summaries.iter_mut().find(|(p, _)| *p == part) {
                    Some((_, s)) => s.push_str(&text),
                    None => buf.summaries.push((part, text)),
                }
            }
            Delta::Opaque(blob) => match &mut buf.opaque {
                Some(existing) => existing.data.push_str(&blob.data),
                None => buf.opaque = Some(blob),
            },
            Delta::ProviderRaw { family, raw } => {
                buf.family = Some(family);
                buf.raw = Some(raw);
            }
            _ => {}
        }
    }

    fn on_item_stop(&mut self, index: u32, out: &mut Vec<Bytes>) {
        match self.take(index) {
            Some(ItemState::Streamed { ant_index, .. }) => {
                if self.open_streamed == Some(index) {
                    self.open_streamed = None;
                }
                out.push(self.block_stop(ant_index));
            }
            Some(ItemState::Buffered(buf)) => {
                // A buffered block is emitted as a self-contained `start … stop`; make sure no
                // streamed block is still open so the two never overlap.
                self.close_open_streamed(out);
                self.flush_buffered(*buf, out);
            }
            // The IR index was already closed by `close_open_streamed` (parallel/interleaved
            // streamed blocks are closed when the next block opens); nothing to do.
            None => {}
        }
    }

    fn flush_buffered(&mut self, buf: Buffered, out: &mut Vec<Bytes>) {
        if buf.is_provider {
            // A foreign-family provider-hosted tool block (e.g. an OpenAI `web_search_call`) has
            // no valid Anthropic content-block shape; emitting it verbatim would leak an
            // OpenAI-shaped `content_block_start` an Anthropic client rejects. Drop it, mirroring
            // the `oi.family == FAMILY` gate on the non-streaming `encode_response` path.
            if buf.family != Some(FAMILY) {
                return;
            }
            let raw = buf.raw.unwrap_or(Value::Null);
            let ant = self.alloc_index();
            out.push(self.block_start(ant, raw));
            out.push(self.block_stop(ant));
            return;
        }
        if buf.is_reasoning {
            self.flush_reasoning(buf, out);
            return;
        }
        // Compaction: the opaque carrier's data is the block JSON.
        if let Some(blob) = &buf.opaque {
            let raw = serde_json::from_str::<Value>(&blob.data).unwrap_or(Value::Null);
            let ant = self.alloc_index();
            out.push(self.block_start(ant, raw));
            out.push(self.block_stop(ant));
        }
    }

    fn flush_reasoning(&mut self, buf: Buffered, out: &mut Vec<Bytes>) {
        let summary_text = if buf.summaries.is_empty() {
            String::new()
        } else {
            buf.summaries.iter().map(|(_, s)| s.as_str()).collect::<Vec<_>>().join("\n\n")
        };
        // Text carrier (§7.2 Resp→Ant exposure): the reasoning text takes precedence, else the
        // joined summary (parts joined by two newlines).
        let text = if !buf.text.is_empty() { buf.text.clone() } else { summary_text };

        if !text.is_empty() {
            // A `thinking` block carrying the reasoning/summary text. This path is client-facing
            // (the client is receiving, not sending), so the `signature` is the sealed envelope
            // when an opaque carrier exists and is simply OMITTED when none does: on replay the
            // router decoder treats a missing signature as no opaque and `lower()` applies the
            // replay policy. Non-`Redacted` opaque and no opaque both land here when text exists.
            let ant = self.alloc_index();
            let mut block = omap();
            block.insert("type".into(), Value::from("thinking"));
            block.insert("thinking".into(), Value::from(""));
            out.push(self.block_start(ant, Value::Object(block)));
            let mut d = omap();
            d.insert("type".into(), Value::from("thinking_delta"));
            d.insert("thinking".into(), Value::from(text));
            out.push(self.block_delta(ant, Value::Object(d)));
            if let Some(blob) = &buf.opaque {
                let mut d = omap();
                d.insert("type".into(), Value::from("signature_delta"));
                d.insert("signature".into(), Value::from(seal_or_native(&self.ctx, blob)));
                out.push(self.block_delta(ant, Value::Object(d)));
            }
            out.push(self.block_stop(ant));
            return;
        }

        // No text and no summary. An opaque carrier becomes a `redacted_thinking` block whose
        // `data` is the sealed envelope; without an opaque carrier there is nothing to render.
        if let Some(blob) = &buf.opaque {
            let ant = self.alloc_index();
            let mut block = omap();
            block.insert("type".into(), Value::from("redacted_thinking"));
            block.insert("data".into(), Value::from(seal_or_native(&self.ctx, blob)));
            out.push(self.block_start(ant, Value::Object(block)));
            out.push(self.block_stop(ant));
        }
    }

    fn on_stop(
        &mut self,
        reason: StopReason,
        usage: Usage,
        ext: llm_xlate_core::Extensions,
        out: &mut Vec<Bytes>,
    ) {
        self.stopped = true;
        let (reason_str, stop_sequence) = stop_reason_to_wire(&reason);

        let mut delta = omap();
        delta.insert("stop_reason".into(), Value::from(reason_str));
        delta.insert(
            "stop_sequence".into(),
            stop_sequence.map(Value::from).unwrap_or(Value::Null),
        );
        for (k, v) in ext.iter() {
            if let Some(field) = k.strip_prefix(EXT_PREFIX) {
                if field == "container" || field == "context_management" {
                    delta.insert(field.to_string(), v.clone());
                }
            }
        }

        let mut u = omap();
        // When the provider only reports usage at the end (Chat / Responses bridged to an
        // Anthropic client), `message_start` carried `input_tokens: 0`; surface the final
        // input / cache counts here so token accounting survives the bridge. Same-family
        // Anthropic streams already reported input at `message_start`, so leave those untouched.
        if self.start_input == 0 && usage.input > 0 {
            u.insert("input_tokens".into(), Value::from(usage.input));
        }
        if self.start_input == 0 {
            if let Some(cr) = usage.cache_read {
                u.insert("cache_read_input_tokens".into(), Value::from(cr));
            }
            if let Some(total) = usage.cache_write {
                // The `cache_creation` split has to appear here too, not just the flat total:
                // the non-streaming encoder emits both, and the crate's law is that a stream
                // and its non-streaming twin aggregate to the same response.
                u.insert("cache_creation_input_tokens".into(), Value::from(total));
                let mut cc = omap();
                cc.insert(
                    "ephemeral_5m_input_tokens".into(),
                    Value::from(usage.cache_write_5m().unwrap_or(0)),
                );
                cc.insert(
                    "ephemeral_1h_input_tokens".into(),
                    Value::from(usage.cache_write_1h.unwrap_or(0)),
                );
                u.insert("cache_creation".into(), Value::Object(cc));
            }
        }
        u.insert("output_tokens".into(), Value::from(usage.output));
        if let Some(r) = usage.reasoning {
            let mut otd = omap();
            otd.insert("thinking_tokens".into(), Value::from(r));
            u.insert("output_tokens_details".into(), Value::Object(otd));
        }

        let mut root = omap();
        root.insert("type".into(), Value::from("message_delta"));
        root.insert("delta".into(), Value::Object(delta));
        root.insert("usage".into(), Value::Object(u));
        out.push(Self::frame("message_delta", &Value::Object(root)));

        let mut stop = omap();
        stop.insert("type".into(), Value::from("message_stop"));
        out.push(Self::frame("message_stop", &Value::Object(stop)));
    }
}

impl StreamEncoder for AnthropicStreamEncoder {
    fn push(&mut self, ev: IrEvent) -> Vec<Bytes> {
        let mut out = Vec::new();
        match ev {
            IrEvent::Start { usage_prefill, .. } => self.on_start(usage_prefill, &mut out),
            IrEvent::ItemStart { index, kind, call, .. } => {
                if !self.started {
                    self.on_start(None, &mut out);
                }
                self.on_item_start(index, kind, call, &mut out);
            }
            IrEvent::Delta { index, delta } => self.on_delta(index, delta, &mut out),
            IrEvent::ItemStop { index } => self.on_item_stop(index, &mut out),
            IrEvent::Stop { reason, usage, ext } => {
                if !self.started {
                    self.on_start(None, &mut out);
                }
                self.on_stop(reason, usage, ext, &mut out);
            }
            IrEvent::Error(e) => {
                let enc = crate::errors::encode_error(&e, true, self.started);
                out.extend(enc.frames);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<Bytes> {
        Vec::new()
    }

    fn keepalive(&mut self) -> Option<Bytes> {
        let mut ping = omap();
        ping.insert("type".into(), Value::from("ping"));
        Some(Self::frame("ping", &Value::Object(ping)))
    }
}
