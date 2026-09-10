//! [`ResponsesStreamEncoder`]: [`IrEvent`]s → OpenAI Responses SSE frames (client-facing).
//!
//! Emits the Responses lifecycle: `response.created` + `response.in_progress` snapshots on
//! [`IrEvent::Start`], per-item `output_item.added` / `content_part.*` / typed deltas /
//! `output_item.done` frames, and a terminal `response.completed` (or `response.incomplete`)
//! carrying the full response object. `sequence_number` is monotonic from `0`. The terminal
//! snapshot is built through the same [`crate::render`] path as
//! [`crate::response::encode_response`], so `encode_response(aggregate(events))` and the
//! streamed `response.completed` payload are byte-identical (plan §11.8).
//!
//! Item ids follow [`crate::render::item_id_for`] (an item's own id, else a minted
//! `"{kind}_{response_id}_{index}"`); the streamed `output_item.added`, `output_item.done`, and
//! terminal snapshot therefore all agree.

use std::collections::HashMap;

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::{
    canon, Annotation, CallId, Delta, EncodeCtx, Extensions, IrEvent, IrResponse, Item, ItemId,
    ItemKind, JsonText, OpaqueBlob, OpaqueItem, ProviderFamily, ReasoningExposure, ReasoningItem,
    Role, SseWriter, StopReason, StreamEncoder, Usage, XlateError,
};

use crate::render::{
    build_response_object, kind_prefix_for_kind, render_output_item, RenderCtx,
};
use crate::response::build_response_value;
use crate::util::Ob;

/// Which content part (if any) has been opened on a message item.
#[derive(PartialEq)]
enum PartStarted {
    None,
    Text,
    Refusal,
}

/// Per-item streaming accumulator.
struct ItemBuf {
    kind: ItemKind,
    id: Option<ItemId>,
    /// The wire id used across this item's frames (preserved id or minted).
    wire_id: String,
    call: Option<(CallId, String)>,
    text: String,
    refusal: String,
    annotations: Vec<Annotation>,
    args: String,
    /// Summary parts keyed by their part index, in first-seen order.
    summaries: Vec<(u32, String)>,
    reasoning_text: String,
    opaque: Option<OpaqueBlob>,
    provider_raw: Option<Value>,
    started: PartStarted,
    summary_parts_open: Vec<u32>,
}

impl ItemBuf {
    fn new(kind: ItemKind, id: Option<ItemId>, wire_id: String, call: Option<(CallId, String)>) -> Self {
        Self {
            kind,
            id,
            wire_id,
            call,
            text: String::new(),
            refusal: String::new(),
            annotations: Vec::new(),
            args: String::new(),
            summaries: Vec::new(),
            reasoning_text: String::new(),
            opaque: None,
            provider_raw: None,
            started: PartStarted::None,
            summary_parts_open: Vec::new(),
        }
    }
}

/// Streaming encoder for the Responses SSE dialect.
pub struct ResponsesStreamEncoder {
    ctx: EncodeCtx,
    seq: u64,
    started: bool,
    bufs: HashMap<u32, ItemBuf>,
    /// Reconstructed items in index order, for the terminal snapshot.
    items: Vec<(u32, Item)>,
}

impl ResponsesStreamEncoder {
    /// A fresh encoder for the given response context.
    pub fn new(ctx: EncodeCtx) -> Self {
        Self { ctx, seq: 0, started: false, bufs: HashMap::new(), items: Vec::new() }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    fn render_ctx(&self) -> RenderCtx<'_> {
        RenderCtx {
            response_id: self.ctx.response_id.as_str(),
            created_at: self.ctx.created_at,
            model: &self.ctx.client_model,
            request_echo: &self.ctx.request_echo,
            sealer: &self.ctx.sealer,
            expose: self.ctx.expose.clone(),
            include: &self.ctx.include,
            store: self.ctx.store,
        }
    }

    /// Build one SSE frame: `{type, sequence_number, ...extra}`.
    fn frame(&mut self, event: &str, extra: Map<String, Value>) -> Bytes {
        let seq = self.next_seq();
        let mut m = Map::new();
        m.insert("type".to_string(), Value::from(event));
        m.insert("sequence_number".to_string(), Value::from(seq));
        for (k, v) in extra {
            m.insert(k, v);
        }
        SseWriter::frame(Some(event), &canon::to_string(&Value::Object(m)))
    }

    /// A lifecycle frame carrying a `response` snapshot.
    fn lifecycle_frame(&mut self, event: &str, response_obj: Value) -> Bytes {
        let extra = Ob::new().set("response", response_obj).into_map();
        self.frame(event, extra)
    }

    /// The in-progress `response` snapshot (empty output, null usage).
    fn in_progress_snapshot(&self) -> Value {
        let rctx = self.render_ctx();
        build_response_object("in_progress", Vec::new(), None, None, None, &rctx)
    }

    fn on_start(&mut self, out: &mut Vec<Bytes>) {
        self.started = true;
        let snap = self.in_progress_snapshot();
        let f1 = self.lifecycle_frame("response.created", snap.clone());
        let f2 = self.lifecycle_frame("response.in_progress", snap);
        out.push(f1);
        out.push(f2);
    }

    fn on_item_start(
        &mut self,
        index: u32,
        kind: ItemKind,
        id: Option<ItemId>,
        call: Option<(CallId, String)>,
        out: &mut Vec<Bytes>,
    ) {
        let prefix = kind_prefix_for_kind(kind);
        let wire_id = match &id {
            Some(i) => i.as_str().to_string(),
            None => crate::render::mint_id(prefix, self.ctx.response_id.as_str(), index as usize),
        };
        let buf = ItemBuf::new(kind, id, wire_id.clone(), call.clone());
        self.bufs.insert(index, buf);

        // Every output item — provider/hosted calls and compaction included — must be opened with
        // an `output_item.added` so the added/done pair is balanced (a `.done` with no matching
        // `.added` is a malformed stream spec-compliant clients reject). The raw item is not yet
        // known at ItemStart, so provider/compaction items get a minimal `{id,type,status}` stub
        // here; the authoritative block arrives later in `output_item.done` via `Delta::ProviderRaw`.
        let stub = match kind {
            ItemKind::Message => Some(
                Ob::new()
                    .set("id", Value::from(wire_id))
                    .set("type", "message".into())
                    .set("status", "in_progress".into())
                    .set("role", "assistant".into())
                    .set("content", Value::Array(Vec::new()))
                    .build(),
            ),
            ItemKind::ToolCall => {
                let (call_id, name) = call.unwrap_or_else(|| (CallId::new(""), String::new()));
                Some(
                    Ob::new()
                        .set("id", Value::from(wire_id))
                        .set("type", "function_call".into())
                        .set("status", "in_progress".into())
                        .set("call_id", Value::from(call_id.as_str().to_string()))
                        .set("name", Value::from(name))
                        .set("arguments", "".into())
                        .build(),
                )
            }
            ItemKind::Reasoning => Some(
                Ob::new()
                    .set("id", Value::from(wire_id))
                    .set("type", "reasoning".into())
                    .set("status", "in_progress".into())
                    .set("summary", Value::Array(Vec::new()))
                    .build(),
            ),
            ItemKind::Compaction => Some(
                Ob::new()
                    .set("id", Value::from(wire_id))
                    .set("type", "compaction".into())
                    .set("status", "in_progress".into())
                    .build(),
            ),
            // Provider/hosted call items: the concrete type is unknown until the raw block
            // arrives, so open with a generic hosted-call stub; `output_item.done` carries the
            // authoritative item.
            ItemKind::ProviderToolCall | ItemKind::ProviderToolResult => Some(
                Ob::new()
                    .set("id", Value::from(wire_id))
                    .set("type", "web_search_call".into())
                    .set("status", "in_progress".into())
                    .build(),
            ),
        };
        if let Some(item) = stub {
            let extra = Ob::new()
                .set("output_index", Value::from(index))
                .set("item", item)
                .into_map();
            let f = self.frame("response.output_item.added", extra);
            out.push(f);
        }
    }

    fn on_delta(&mut self, index: u32, delta: Delta, out: &mut Vec<Bytes>) {
        // Frames that must be emitted are computed here (buf borrow ends before `self.frame`).
        enum Emit {
            None,
            One(String, Map<String, Value>),
            Two((String, Map<String, Value>), (String, Map<String, Value>)),
        }
        let full = ReasoningExposure::Full;
        let expose_full = self.ctx.expose == full;
        let wire_id = self.bufs.get(&index).map(|b| b.wire_id.clone()).unwrap_or_default();

        let emit = {
            let Some(buf) = self.bufs.get_mut(&index) else { return };
            match delta {
                Delta::Text(t) => {
                    let open = if buf.started == PartStarted::None {
                        buf.started = PartStarted::Text;
                        Some(content_part_added_text(&wire_id, index))
                    } else {
                        None
                    };
                    buf.text.push_str(&t);
                    let d = Ob::new()
                        .set("item_id", Value::from(wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("content_index", Value::from(0u32))
                        .set("delta", Value::from(t))
                        .into_map();
                    match open {
                        Some(o) => Emit::Two(o, ("response.output_text.delta".into(), d)),
                        None => Emit::One("response.output_text.delta".into(), d),
                    }
                }
                Delta::Refusal(t) => {
                    let open = if buf.started == PartStarted::None {
                        buf.started = PartStarted::Refusal;
                        Some(content_part_added_refusal(&wire_id, index))
                    } else {
                        None
                    };
                    buf.refusal.push_str(&t);
                    let d = Ob::new()
                        .set("item_id", Value::from(wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("content_index", Value::from(0u32))
                        .set("delta", Value::from(t))
                        .into_map();
                    match open {
                        Some(o) => Emit::Two(o, ("response.refusal.delta".into(), d)),
                        None => Emit::One("response.refusal.delta".into(), d),
                    }
                }
                Delta::Annotation(a) => {
                    let ann_index = buf.annotations.len() as u32;
                    let d = Ob::new()
                        .set("item_id", Value::from(wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("content_index", Value::from(0u32))
                        .set("annotation_index", Value::from(ann_index))
                        .set("annotation", a.raw.clone())
                        .into_map();
                    buf.annotations.push(a);
                    Emit::One("response.output_text.annotation.added".into(), d)
                }
                Delta::ToolArgs(t) => {
                    buf.args.push_str(&t);
                    let d = Ob::new()
                        .set("item_id", Value::from(wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("delta", Value::from(t))
                        .into_map();
                    Emit::One("response.function_call_arguments.delta".into(), d)
                }
                Delta::ReasoningSummary { part, text } => {
                    let open = if !buf.summary_parts_open.contains(&part) {
                        buf.summary_parts_open.push(part);
                        Some((
                            "response.reasoning_summary_part.added".to_string(),
                            reasoning_summary_part_added(&wire_id, index, part),
                        ))
                    } else {
                        None
                    };
                    match buf.summaries.iter_mut().find(|(p, _)| *p == part) {
                        Some((_, s)) => s.push_str(&text),
                        None => buf.summaries.push((part, text.clone())),
                    }
                    let d = Ob::new()
                        .set("item_id", Value::from(wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("summary_index", Value::from(part))
                        .set("delta", Value::from(text))
                        .into_map();
                    match open {
                        Some(o) => Emit::Two(o, ("response.reasoning_summary_text.delta".into(), d)),
                        None => Emit::One("response.reasoning_summary_text.delta".into(), d),
                    }
                }
                Delta::ReasoningText(t) => {
                    buf.reasoning_text.push_str(&t);
                    if expose_full {
                        let d = Ob::new()
                            .set("item_id", Value::from(wire_id.clone()))
                            .set("output_index", Value::from(index))
                            .set("content_index", Value::from(0u32))
                            .set("delta", Value::from(t))
                            .into_map();
                        Emit::One("response.reasoning_text.delta".into(), d)
                    } else {
                        Emit::None
                    }
                }
                Delta::Opaque(blob) => {
                    match &mut buf.opaque {
                        Some(existing) => existing.data.push_str(&blob.data),
                        None => buf.opaque = Some(blob),
                    }
                    Emit::None
                }
                Delta::ProviderRaw { raw, .. } => {
                    buf.provider_raw = Some(raw);
                    Emit::None
                }
            }
        };

        match emit {
            Emit::None => {}
            Emit::One(ev, m) => {
                let f = self.frame(&ev, m);
                out.push(f);
            }
            Emit::Two((e1, m1), (e2, m2)) => {
                let f1 = self.frame(&e1, m1);
                let f2 = self.frame(&e2, m2);
                out.push(f1);
                out.push(f2);
            }
        }
    }

    fn on_item_stop(&mut self, index: u32, out: &mut Vec<Bytes>) {
        let Some(mut buf) = self.bufs.remove(&index) else { return };
        let expose_full = self.ctx.expose == ReasoningExposure::Full;

        match buf.kind {
            ItemKind::Message => {
                match buf.started {
                    PartStarted::Text => {
                        let done = Ob::new()
                            .set("item_id", Value::from(buf.wire_id.clone()))
                            .set("output_index", Value::from(index))
                            .set("content_index", Value::from(0u32))
                            .set("text", Value::from(buf.text.clone()))
                            .into_map();
                        let f = self.frame("response.output_text.done", done);
                        out.push(f);
                        let part = Ob::new()
                            .set("type", "output_text".into())
                            .set("text", Value::from(buf.text.clone()))
                            .set(
                                "annotations",
                                Value::Array(buf.annotations.iter().map(|a| a.raw.clone()).collect()),
                            )
                            .build();
                        let cpd = content_part_done(&buf.wire_id, index, part);
                        let f = self.frame("response.content_part.done", cpd);
                        out.push(f);
                    }
                    PartStarted::Refusal => {
                        let done = Ob::new()
                            .set("item_id", Value::from(buf.wire_id.clone()))
                            .set("output_index", Value::from(index))
                            .set("content_index", Value::from(0u32))
                            .set("refusal", Value::from(buf.refusal.clone()))
                            .into_map();
                        let f = self.frame("response.refusal.done", done);
                        out.push(f);
                        let part = Ob::new()
                            .set("type", "refusal".into())
                            .set("refusal", Value::from(buf.refusal.clone()))
                            .build();
                        let cpd = content_part_done(&buf.wire_id, index, part);
                        let f = self.frame("response.content_part.done", cpd);
                        out.push(f);
                    }
                    PartStarted::None => {}
                }
            }
            ItemKind::ToolCall => {
                let done = Ob::new()
                    .set("item_id", Value::from(buf.wire_id.clone()))
                    .set("output_index", Value::from(index))
                    .set("arguments", Value::from(buf.args.clone()))
                    .into_map();
                let f = self.frame("response.function_call_arguments.done", done);
                out.push(f);
            }
            ItemKind::Reasoning => {
                // `Summary` targets that never received a summary fold the raw reasoning text into
                // one summary part so the summary is not lost (documented normalization). This is
                // gated to `Summary(_)` — NOT merely "not Full" — so under `None` (the default for
                // a Responses client that sends no `reasoning` block) the chain-of-thought is
                // dropped, exactly as `render_reasoning_output` withholds it, closing the leak and
                // keeping the stream terminal in agreement with `encode_response` (plan §11.8).
                if matches!(self.ctx.expose, ReasoningExposure::Summary(_))
                    && buf.summaries.is_empty()
                    && !buf.reasoning_text.is_empty()
                {
                    let text = buf.reasoning_text.clone();
                    buf.summaries.push((0, text.clone()));
                    let added = reasoning_summary_part_added(&buf.wire_id, index, 0);
                    let f = self.frame("response.reasoning_summary_part.added", added);
                    out.push(f);
                    let d = Ob::new()
                        .set("item_id", Value::from(buf.wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("summary_index", Value::from(0u32))
                        .set("delta", Value::from(text))
                        .into_map();
                    let f = self.frame("response.reasoning_summary_text.delta", d);
                    out.push(f);
                }
                let summaries = buf.summaries.clone();
                for (part, text) in &summaries {
                    let td = Ob::new()
                        .set("item_id", Value::from(buf.wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("summary_index", Value::from(*part))
                        .set("text", Value::from(text.clone()))
                        .into_map();
                    let f = self.frame("response.reasoning_summary_text.done", td);
                    out.push(f);
                    let pd = Ob::new()
                        .set("item_id", Value::from(buf.wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("summary_index", Value::from(*part))
                        .set(
                            "part",
                            Ob::new()
                                .set("type", "summary_text".into())
                                .set("text", Value::from(text.clone()))
                                .build(),
                        )
                        .into_map();
                    let f = self.frame("response.reasoning_summary_part.done", pd);
                    out.push(f);
                }
                if expose_full && !buf.reasoning_text.is_empty() {
                    let td = Ob::new()
                        .set("item_id", Value::from(buf.wire_id.clone()))
                        .set("output_index", Value::from(index))
                        .set("content_index", Value::from(0u32))
                        .set("text", Value::from(buf.reasoning_text.clone()))
                        .into_map();
                    let f = self.frame("response.reasoning_text.done", td);
                    out.push(f);
                }
            }
            _ => {}
        }

        // Reconstruct the finished IR item, render it, and emit `output_item.done`.
        let item = reconstruct_item(&buf, &self.ctx.expose);
        let rendered = {
            let rctx = self.render_ctx();
            render_output_item(&item, index as usize, &rctx)
        };
        self.items.push((index, item));
        let extra = Ob::new()
            .set("output_index", Value::from(index))
            .set("item", rendered)
            .into_map();
        let f = self.frame("response.output_item.done", extra);
        out.push(f);
    }

    fn on_stop(&mut self, reason: StopReason, usage: Usage, ext: Extensions, out: &mut Vec<Bytes>) {
        self.items.sort_by_key(|(i, _)| *i);
        let items: Vec<Item> = self.items.iter().map(|(_, it)| it.clone()).collect();
        let resp = IrResponse {
            id: self.ctx.response_id.clone(),
            model: self.ctx.client_model.clone(),
            items,
            stop: reason.clone(),
            usage,
            ext,
        };
        let response_obj = {
            let rctx = self.render_ctx();
            build_response_value(&resp, &rctx)
        };
        let event = match reason {
            StopReason::MaxTokens | StopReason::ContentFilter => "response.incomplete",
            _ => "response.completed",
        };
        let f = self.lifecycle_frame(event, response_obj);
        out.push(f);
    }

    fn on_error(&mut self, e: XlateError, out: &mut Vec<Bytes>) {
        if self.started {
            let inner = crate::errors::error_inner_value(&e);
            self.items.sort_by_key(|(i, _)| *i);
            let rendered: Vec<Value> = {
                let rctx = self.render_ctx();
                self.items.iter().map(|(i, it)| render_output_item(it, *i as usize, &rctx)).collect()
            };
            let response_obj = {
                let rctx = self.render_ctx();
                build_response_object("failed", rendered, None, Some(inner), None, &rctx)
            };
            let f = self.lifecycle_frame("response.failed", response_obj);
            out.push(f);
        } else {
            let extra = Ob::new()
                .set("code", e.provider_code.clone().map(Value::from).unwrap_or(Value::Null))
                .set("message", Value::from(e.message.clone()))
                .set("param", e.param.clone().map(Value::from).unwrap_or(Value::Null))
                .into_map();
            let f = self.frame("error", extra);
            out.push(f);
        }
    }
}

impl StreamEncoder for ResponsesStreamEncoder {
    fn push(&mut self, ev: IrEvent) -> Vec<Bytes> {
        let mut out = Vec::new();
        match ev {
            IrEvent::Start { .. } => self.on_start(&mut out),
            IrEvent::ItemStart { index, kind, id, call } => {
                self.on_item_start(index, kind, id, call, &mut out)
            }
            IrEvent::Delta { index, delta } => self.on_delta(index, delta, &mut out),
            IrEvent::ItemStop { index } => self.on_item_stop(index, &mut out),
            IrEvent::Stop { reason, usage, ext } => self.on_stop(reason, usage, ext, &mut out),
            IrEvent::Error(e) => self.on_error(e, &mut out),
        }
        out
    }

    fn finish(&mut self) -> Vec<Bytes> {
        // The Responses stream has no `[DONE]` sentinel; the terminal event closes it.
        Vec::new()
    }

    fn keepalive(&mut self) -> Option<Bytes> {
        Some(SseWriter::comment("keepalive"))
    }
}

/// Reconstruct the finished IR item from its accumulator (mirrors the core aggregator, with the
/// non-Full reasoning-text fold applied so the streamed item and the terminal snapshot agree).
fn reconstruct_item(buf: &ItemBuf, expose: &ReasoningExposure) -> Item {
    match buf.kind {
        ItemKind::Message => {
            let mut content = Vec::new();
            if !buf.text.is_empty() || !buf.annotations.is_empty() {
                content.push(llm_xlate_core::Part::Text {
                    text: buf.text.clone(),
                    annotations: buf.annotations.clone(),
                    cache_control: None,
                });
            }
            if !buf.refusal.is_empty() {
                content.push(llm_xlate_core::Part::Refusal { text: buf.refusal.clone() });
            }
            Item::Message { role: Role::Assistant, content, id: buf.id.clone() }
        }
        ItemKind::ToolCall => {
            let (call_id, name) =
                buf.call.clone().unwrap_or_else(|| (CallId::new(""), String::new()));
            Item::ToolCall {
                call_id,
                name,
                arguments: JsonText::new(buf.args.clone()),
                id: buf.id.clone(),
            }
        }
        ItemKind::Reasoning => {
            // Fold reasoning text into a summary part only under `Summary(_)` (not under `None`,
            // where the chain-of-thought must be dropped, nor `Full`, where it is exposed as
            // content). Mirrors the streaming fold gate above (plan §11.8).
            let (text, summary): (Option<String>, Vec<String>) =
                if matches!(expose, ReasoningExposure::Summary(_))
                    && buf.summaries.is_empty()
                    && !buf.reasoning_text.is_empty()
                {
                    (None, vec![buf.reasoning_text.clone()])
                } else {
                    let text = if buf.reasoning_text.is_empty() {
                        None
                    } else {
                        Some(buf.reasoning_text.clone())
                    };
                    (text, buf.summaries.iter().map(|(_, s)| s.clone()).collect())
                };
            Item::Reasoning(ReasoningItem { text, summary, opaque: buf.opaque.clone(), id: buf.id.clone() })
        }
        ItemKind::ProviderToolResult => Item::ProviderToolResult(OpaqueItem::new(
            ProviderFamily::OpenAI,
            buf.provider_raw.clone().unwrap_or(Value::Null),
        )),
        ItemKind::ProviderToolCall => Item::ProviderToolCall(OpaqueItem::new(
            ProviderFamily::OpenAI,
            buf.provider_raw.clone().unwrap_or(Value::Null),
        )),
        ItemKind::Compaction => {
            let blob = buf.opaque.clone().unwrap_or_else(|| {
                OpaqueBlob::new(ProviderFamily::OpenAI, llm_xlate_core::OpaqueKind::Compaction, "")
            });
            Item::Compaction(blob)
        }
    }
}

fn content_part_added_text(wire_id: &str, index: u32) -> (String, Map<String, Value>) {
    let part = Ob::new()
        .set("type", "output_text".into())
        .set("text", "".into())
        .set("annotations", Value::Array(Vec::new()))
        .build();
    let m = Ob::new()
        .set("item_id", Value::from(wire_id))
        .set("output_index", Value::from(index))
        .set("content_index", Value::from(0u32))
        .set("part", part)
        .into_map();
    ("response.content_part.added".into(), m)
}

fn content_part_added_refusal(wire_id: &str, index: u32) -> (String, Map<String, Value>) {
    let part = Ob::new().set("type", "refusal".into()).set("refusal", "".into()).build();
    let m = Ob::new()
        .set("item_id", Value::from(wire_id))
        .set("output_index", Value::from(index))
        .set("content_index", Value::from(0u32))
        .set("part", part)
        .into_map();
    ("response.content_part.added".into(), m)
}

fn content_part_done(wire_id: &str, index: u32, part: Value) -> Map<String, Value> {
    Ob::new()
        .set("item_id", Value::from(wire_id))
        .set("output_index", Value::from(index))
        .set("content_index", Value::from(0u32))
        .set("part", part)
        .into_map()
}

fn reasoning_summary_part_added(wire_id: &str, index: u32, part: u32) -> Map<String, Value> {
    Ob::new()
        .set("item_id", Value::from(wire_id))
        .set("output_index", Value::from(index))
        .set("summary_index", Value::from(part))
        .set(
            "part",
            Ob::new().set("type", "summary_text".into()).set("text", "".into()).build(),
        )
        .into_map()
}
