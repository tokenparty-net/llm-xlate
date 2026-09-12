//! [`Aggregator`]: folds an [`IrEvent`] stream into an [`IrResponse`] for non-streaming
//! clients, store, and logging. Upholds the invariant
//! `aggregate(encode_stream(ev)) ≡ encode_response(aggregate(ev))` (plan §3, §11.8).

use serde_json::Value;

use crate::error::XlateError;
use crate::ir::{
    Annotation, CallId, Delta, Extensions, IrEvent, IrResponse, Item, ItemId, ItemKind, JsonText,
    OpaqueBlob, OpaqueItem, OpaqueKind, Part, ProviderFamily, ReasoningItem, ResponseId, Role,
    StopReason, Usage,
};

/// A per-item accumulator.
enum Building {
    Message {
        id: Option<ItemId>,
        text: String,
        refusal: String,
        annotations: Vec<Annotation>,
    },
    Reasoning {
        id: Option<ItemId>,
        text: String,
        /// Summary parts keyed by their summary-part index. Deltas for a part index append to
        /// its string wherever it appears (contiguous or not), matching the documented
        /// "same part index concatenates" contract.
        summaries: Vec<(u32, String)>,
        opaque: Option<OpaqueBlob>,
    },
    ToolCall {
        call_id: CallId,
        name: String,
        args: String,
        id: Option<ItemId>,
    },
    /// A provider-hosted tool call/result. Its payload is the structured provider JSON carried
    /// by [`Delta::ProviderRaw`] — never a re-parsed string.
    Provider {
        is_result: bool,
        family: Option<ProviderFamily>,
        raw: Option<Value>,
    },
    /// A provider compaction / context-management block, built from accumulated
    /// [`Delta::Opaque`] carriers.
    Compaction {
        opaque: Option<OpaqueBlob>,
    },
}

/// Construct a fresh `Message` accumulator.
fn new_message(id: Option<ItemId>) -> Building {
    Building::Message { id, text: String::new(), refusal: String::new(), annotations: Vec::new() }
}

/// Construct a fresh `Reasoning` accumulator.
fn new_reasoning(id: Option<ItemId>) -> Building {
    Building::Reasoning { id, text: String::new(), summaries: Vec::new(), opaque: None }
}

/// Append an opaque carrier chunk into `slot`, concatenating `data` when a carrier is already
/// present (keeping the first chunk's `family` / `kind` / `model`). Used for reasoning and
/// compaction carriers so a chunked signature/encrypted/compaction blob is never truncated.
fn accumulate_opaque(slot: &mut Option<OpaqueBlob>, blob: OpaqueBlob) {
    match slot {
        Some(existing) => existing.data.push_str(&blob.data),
        None => *slot = Some(blob),
    }
}

/// Accumulates [`IrEvent`]s into an [`IrResponse`]. Missing `ItemStop`/`Stop` are tolerated;
/// an [`IrEvent::Error`] makes [`Aggregator::finish`] return `Err`.
#[derive(Default)]
pub struct Aggregator {
    response_id: Option<ResponseId>,
    model: Option<String>,
    prefill: Option<Usage>,
    items: Vec<(u32, Building)>,
    stop: Option<StopReason>,
    usage: Option<Usage>,
    response_ext: Extensions,
    error: Option<XlateError>,
}

impl Aggregator {
    /// A fresh aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    fn slot(&mut self, index: u32) -> Option<&mut Building> {
        self.items.iter_mut().find(|(i, _)| *i == index).map(|(_, b)| b)
    }

    fn ensure(&mut self, index: u32, make: impl FnOnce() -> Building) {
        if !self.items.iter().any(|(i, _)| *i == index) {
            self.items.push((index, make()));
        }
    }

    /// Feed the next event.
    pub fn push(&mut self, ev: IrEvent) {
        match ev {
            IrEvent::Start { response_id, model, usage_prefill } => {
                self.response_id = Some(response_id);
                self.model = Some(model);
                if usage_prefill.is_some() {
                    self.prefill = usage_prefill;
                }
            }
            IrEvent::ItemStart { index, kind, id, call } => {
                let builder = match kind {
                    ItemKind::Message => new_message(id),
                    ItemKind::Reasoning => new_reasoning(id),
                    ItemKind::ToolCall => {
                        let (call_id, name) = call.unwrap_or_else(|| (CallId::new(""), String::new()));
                        Building::ToolCall { call_id, name, args: String::new(), id }
                    }
                    ItemKind::ProviderToolCall => {
                        let _ = id;
                        Building::Provider { is_result: false, family: None, raw: None }
                    }
                    ItemKind::ProviderToolResult => {
                        let _ = id;
                        Building::Provider { is_result: true, family: None, raw: None }
                    }
                    ItemKind::Compaction => {
                        let _ = id;
                        Building::Compaction { opaque: None }
                    }
                };
                // Replace any lazily-created placeholder at this index, else push.
                if let Some((_, b)) = self.items.iter_mut().find(|(i, _)| *i == index) {
                    *b = builder;
                } else {
                    self.items.push((index, builder));
                }
            }
            IrEvent::Delta { index, delta } => self.apply_delta(index, delta),
            IrEvent::ItemStop { .. } => { /* items accumulate incrementally; nothing to finalize */ }
            IrEvent::Stop { reason, usage, ext } => {
                self.stop = Some(reason);
                self.usage = Some(usage);
                // Response-level metadata (stop_details, service_tier, …) reaches IrResponse.ext
                // unchanged; merge so it survives the streaming path (plan §11.8). Later keys win.
                for (k, v) in ext.iter() {
                    self.response_ext.insert(k.clone(), v.clone());
                }
            }
            IrEvent::Error(e) => {
                if self.error.is_none() {
                    self.error = Some(e);
                }
            }
        }
    }

    fn apply_delta(&mut self, index: u32, delta: Delta) {
        match delta {
            Delta::Text(t) => {
                self.ensure(index, || new_message(None));
                if let Some(Building::Message { text, .. }) = self.slot(index) {
                    text.push_str(&t);
                }
            }
            Delta::Refusal(t) => {
                self.ensure(index, || new_message(None));
                if let Some(Building::Message { refusal, .. }) = self.slot(index) {
                    refusal.push_str(&t);
                }
            }
            Delta::Annotation(a) => {
                self.ensure(index, || new_message(None));
                if let Some(Building::Message { annotations, .. }) = self.slot(index) {
                    annotations.push(a);
                }
            }
            Delta::ReasoningText(t) => {
                self.ensure(index, || new_reasoning(None));
                if let Some(Building::Reasoning { text, .. }) = self.slot(index) {
                    text.push_str(&t);
                }
            }
            Delta::ReasoningSummary { part, text: chunk } => {
                self.ensure(index, || new_reasoning(None));
                if let Some(Building::Reasoning { summaries, .. }) = self.slot(index) {
                    // Append to the existing string for this part index wherever it sits
                    // (parts may be interleaved), else start a new part.
                    match summaries.iter_mut().find(|(p, _)| *p == part) {
                        Some((_, s)) => s.push_str(&chunk),
                        None => summaries.push((part, chunk)),
                    }
                }
            }
            Delta::ToolArgs(t) => {
                self.ensure(index, || Building::ToolCall {
                    call_id: CallId::new(""),
                    name: String::new(),
                    args: String::new(),
                    id: None,
                });
                if let Some(Building::ToolCall { args, .. }) = self.slot(index) {
                    args.push_str(&t);
                }
            }
            Delta::Opaque(blob) => {
                // Opaque deltas carry a reasoning replay carrier or a compaction blob; they are
                // accumulated (never overwritten) so a chunked carrier is not truncated.
                match self.slot(index) {
                    Some(Building::Reasoning { opaque, .. })
                    | Some(Building::Compaction { opaque, .. }) => accumulate_opaque(opaque, blob),
                    _ => {
                        // Unknown index / no ItemStart: route by kind — a compaction blob opens
                        // a Compaction item, anything else a reasoning carrier.
                        let building = if blob.kind == OpaqueKind::Compaction {
                            Building::Compaction { opaque: Some(blob) }
                        } else {
                            Building::Reasoning {
                                id: None,
                                text: String::new(),
                                summaries: Vec::new(),
                                opaque: Some(blob),
                            }
                        };
                        self.items.push((index, building));
                    }
                }
            }
            Delta::ProviderRaw { family: fam, raw: value } => {
                // The structured provider-hosted-tool payload. Store it directly; the decoder
                // emits the authoritative full block (last wins).
                self.ensure(index, || Building::Provider { is_result: false, family: None, raw: None });
                if let Some(Building::Provider { family, raw, .. }) = self.slot(index) {
                    *family = Some(fam);
                    *raw = Some(value);
                }
            }
        }
    }

    /// Finalize. Returns `Err` if an [`IrEvent::Error`] was seen. Missing `Stop` yields
    /// [`StopReason::EndTurn`] and default usage.
    pub fn finish(mut self) -> Result<IrResponse, XlateError> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }

        let mut usage = self.usage.take().unwrap_or_default();
        if let Some(prefill) = &self.prefill {
            if usage.input == 0 {
                usage.input = prefill.input;
            }
            // Every prompt-side counter can arrive at message start and be omitted at stop
            // (Anthropic reports the whole breakdown in `message_start`), so each one falls
            // back to the prefill independently.
            if usage.cache_read.is_none() {
                usage.cache_read = prefill.cache_read;
            }
            if usage.cache_write.is_none() {
                usage.cache_write = prefill.cache_write;
            }
            if usage.cache_write_1h.is_none() {
                usage.cache_write_1h = prefill.cache_write_1h;
            }
            if usage.reasoning.is_none() {
                usage.reasoning = prefill.reasoning;
            }
        }
        usage.enforce_invariants();
        debug_assert!(
            usage.cache_write_1h.unwrap_or(0) <= usage.cache_write.unwrap_or(0),
            "Usage invariant broken: cache_write_1h > cache_write ({usage:?})"
        );

        let items = self.items.into_iter().map(|(_, b)| build_item(b)).collect();

        Ok(IrResponse {
            id: self.response_id.unwrap_or_default(),
            model: self.model.unwrap_or_default(),
            items,
            stop: self.stop.unwrap_or(StopReason::EndTurn),
            usage,
            ext: self.response_ext,
        })
    }
}

fn build_item(b: Building) -> Item {
    match b {
        Building::Message { id, text, refusal, annotations } => {
            let mut content = Vec::new();
            if !text.is_empty() || !annotations.is_empty() {
                content.push(Part::Text { text, annotations, cache_control: None });
            }
            if !refusal.is_empty() {
                content.push(Part::Refusal { text: refusal });
            }
            Item::Message { role: Role::Assistant, content, id }
        }
        Building::Reasoning { id, text, summaries, opaque } => Item::Reasoning(ReasoningItem {
            text: if text.is_empty() { None } else { Some(text) },
            summary: summaries.into_iter().map(|(_, s)| s).collect(),
            opaque,
            id,
        }),
        Building::ToolCall { call_id, name, args, id } => {
            Item::ToolCall { call_id, name, arguments: JsonText::new(args), id }
        }
        Building::Provider { is_result, family, raw } => {
            let item = OpaqueItem {
                family: family.unwrap_or_else(|| ProviderFamily::Other("unknown".into())),
                raw: raw.unwrap_or(Value::Null),
            };
            if is_result {
                Item::ProviderToolResult(item)
            } else {
                Item::ProviderToolCall(item)
            }
        }
        Building::Compaction { opaque } => Item::Compaction(opaque.unwrap_or_else(|| {
            OpaqueBlob::new(ProviderFamily::Other("unknown".into()), OpaqueKind::Compaction, "")
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use crate::ir::OpaqueKind;
    use pretty_assertions::assert_eq;

    fn start() -> IrEvent {
        IrEvent::Start {
            response_id: ResponseId::new("resp_1"),
            model: "m".into(),
            usage_prefill: None,
        }
    }

    fn run(events: Vec<IrEvent>) -> Result<IrResponse, XlateError> {
        let mut agg = Aggregator::new();
        for e in events {
            agg.push(e);
        }
        agg.finish()
    }

    #[test]
    fn simple_message() {
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::Text("Hel".into()) },
            IrEvent::Delta { index: 0, delta: Delta::Text("lo".into()) },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(3, 5), ext: Default::default() },
        ])
        .unwrap();
        assert_eq!(r.id, ResponseId::new("resp_1"));
        assert_eq!(r.model, "m");
        assert_eq!(r.stop, StopReason::EndTurn);
        assert_eq!(r.usage, Usage::new(3, 5));
        assert_eq!(r.items, vec![Item::assistant_text("Hello")]);
    }

    #[test]
    fn reasoning_summary_parts_concatenate() {
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 0, text: "a".into() } },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 0, text: "b".into() } },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 1, text: "c".into() } },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        match &r.items[0] {
            Item::Reasoning(ri) => assert_eq!(ri.summary, vec!["ab".to_string(), "c".to_string()]),
            other => panic!("expected reasoning, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_text_and_opaque() {
        let blob = OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "sig");
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: Some(ItemId::new("rs_1")), call: None },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningText("think".into()) },
            IrEvent::Delta { index: 0, delta: Delta::Opaque(blob.clone()) },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        match &r.items[0] {
            Item::Reasoning(ri) => {
                assert_eq!(ri.text.as_deref(), Some("think"));
                assert_eq!(ri.opaque, Some(blob));
                assert_eq!(ri.id, Some(ItemId::new("rs_1")));
            }
            other => panic!("expected reasoning, got {other:?}"),
        }
    }

    #[test]
    fn tool_call_args_accumulate() {
        let r = run(vec![
            start(),
            IrEvent::ItemStart {
                index: 0,
                kind: ItemKind::ToolCall,
                id: Some(ItemId::new("fc_1")),
                call: Some((CallId::new("call_9"), "get_weather".into())),
            },
            IrEvent::Delta { index: 0, delta: Delta::ToolArgs("{\"city\":".into()) },
            IrEvent::Delta { index: 0, delta: Delta::ToolArgs("\"NYC\"}".into()) },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::ToolUse, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        assert_eq!(
            r.items[0],
            Item::ToolCall {
                call_id: CallId::new("call_9"),
                name: "get_weather".into(),
                arguments: JsonText::new("{\"city\":\"NYC\"}"),
                id: Some(ItemId::new("fc_1")),
            }
        );
        assert_eq!(r.stop, StopReason::ToolUse);
    }

    #[test]
    fn refusal_becomes_part() {
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::Refusal("no".into()) },
            IrEvent::Stop { reason: StopReason::Refusal, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        assert_eq!(
            r.items[0],
            Item::Message { role: Role::Assistant, content: vec![Part::Refusal { text: "no".into() }], id: None }
        );
    }

    #[test]
    fn provider_tool_call_from_provider_raw() {
        // The structured carrier stores the full provider JSON verbatim — no re-parse, no
        // OpaqueKind abuse, and no silent Value::String corruption on non-JSON.
        let raw = serde_json::json!({"type": "server_tool_use", "name": "web_search", "input": {"q": "x"}});
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::ProviderToolCall, id: None, call: None },
            IrEvent::Delta {
                index: 0,
                delta: Delta::ProviderRaw { family: ProviderFamily::Anthropic, raw: raw.clone() },
            },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        match &r.items[0] {
            Item::ProviderToolCall(oi) => {
                assert_eq!(oi.family, ProviderFamily::Anthropic);
                assert_eq!(oi.raw, raw);
            }
            other => panic!("expected provider tool call, got {other:?}"),
        }
    }

    #[test]
    fn provider_tool_result_last_raw_wins() {
        // A decoder may emit a partial then the authoritative full block; the last wins.
        let full = serde_json::json!({"type": "web_search_tool_result", "content": [{"url": "u"}]});
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::ProviderToolResult, id: None, call: None },
            IrEvent::Delta {
                index: 0,
                delta: Delta::ProviderRaw { family: ProviderFamily::Anthropic, raw: serde_json::json!({}) },
            },
            IrEvent::Delta {
                index: 0,
                delta: Delta::ProviderRaw { family: ProviderFamily::Anthropic, raw: full.clone() },
            },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        match &r.items[0] {
            Item::ProviderToolResult(oi) => assert_eq!(oi.raw, full),
            other => panic!("expected provider tool result, got {other:?}"),
        }
    }

    #[test]
    fn compaction_item_from_opaque() {
        let blob = OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Compaction, "cmp");
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Compaction, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::Opaque(blob.clone()) },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        assert_eq!(r.items[0], Item::Compaction(blob));
    }

    #[test]
    fn reasoning_opaque_chunks_accumulate() {
        // A signature/encrypted carrier split across several Opaque deltas must not truncate.
        let a = OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "sig-");
        let b = OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "part2");
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::Opaque(a) },
            IrEvent::Delta { index: 0, delta: Delta::Opaque(b) },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        match &r.items[0] {
            Item::Reasoning(ri) => {
                let o = ri.opaque.as_ref().expect("opaque");
                assert_eq!(o.data, "sig-part2");
                assert_eq!(o.kind, OpaqueKind::Signature);
            }
            other => panic!("expected reasoning, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_summary_noncontiguous_parts_concatenate() {
        // Interleaved part indices (0,1,0) still concatenate per part: ["ab","c"].
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 0, text: "a".into() } },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 1, text: "c".into() } },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningSummary { part: 0, text: "b".into() } },
            IrEvent::ItemStop { index: 0 },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        match &r.items[0] {
            Item::Reasoning(ri) => assert_eq!(ri.summary, vec!["ab".to_string(), "c".to_string()]),
            other => panic!("expected reasoning, got {other:?}"),
        }
    }

    #[test]
    fn stop_ext_reaches_response() {
        // Response-level metadata on Stop must survive the aggregator (plan §11.8).
        let mut ext = Extensions::new();
        ext.insert("stop_details", serde_json::json!({"type": "refusal"}));
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::Refusal("no".into()) },
            IrEvent::Stop { reason: StopReason::Refusal, usage: Usage::default(), ext },
        ])
        .unwrap();
        assert_eq!(r.stop, StopReason::Refusal);
        assert_eq!(r.ext.get("stop_details"), Some(&serde_json::json!({"type": "refusal"})));
    }

    #[test]
    fn missing_stop_defaults_end_turn() {
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::Text("hi".into()) },
        ])
        .unwrap();
        assert_eq!(r.stop, StopReason::EndTurn);
        assert_eq!(r.usage, Usage::default());
    }

    #[test]
    fn prefill_fills_input_when_stop_has_zero() {
        let r = run(vec![
            IrEvent::Start {
                response_id: ResponseId::new("r"),
                model: "m".into(),
                usage_prefill: Some(Usage::new(11, 0)),
            },
            IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::Text("x".into()) },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(0, 7), ext: Default::default() },
        ])
        .unwrap();
        assert_eq!(r.usage.input, 11);
        assert_eq!(r.usage.output, 7);
    }

    #[test]
    fn error_event_yields_err() {
        let err = run(vec![
            start(),
            IrEvent::Error(XlateError::new(ErrorKind::ServerError, "boom")),
        ])
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ServerError);
    }

    #[test]
    fn multiple_items_preserve_order() {
        let r = run(vec![
            start(),
            IrEvent::ItemStart { index: 0, kind: ItemKind::Reasoning, id: None, call: None },
            IrEvent::Delta { index: 0, delta: Delta::ReasoningText("t".into()) },
            IrEvent::ItemStart { index: 1, kind: ItemKind::Message, id: None, call: None },
            IrEvent::Delta { index: 1, delta: Delta::Text("answer".into()) },
            IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::default(), ext: Default::default() },
        ])
        .unwrap();
        assert_eq!(r.items.len(), 2);
        assert!(matches!(r.items[0], Item::Reasoning(_)));
        assert_eq!(r.items[1], Item::assistant_text("answer"));
    }
}
