//! Round-trip and helper tests for the IR. These pin the serde tagging choices (internal
//! tag for `Item`/`IrEvent`/`OutputFormat`/`ToolDef`, external elsewhere) as actually
//! round-trippable, which the codec crates rely on.

use super::*;
use crate::error::XlateError;
use bytes::Bytes;
use pretty_assertions::assert_eq;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::json;

fn rt<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(v: T) {
    let s = serde_json::to_string(&v).expect("serialize");
    let back: T = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(v, back, "round-trip mismatch via {s}");
}

#[test]
fn ir_request_default_round_trips() {
    rt(IrRequest::default());
}

#[test]
fn item_message_with_rich_parts_round_trips() {
    let item = Item::Message {
        role: Role::User,
        content: vec![
            Part::Text {
                text: "hi".into(),
                annotations: vec![Annotation::new("url_citation", json!({"url": "x"}))],
                cache_control: Some(CacheControl::ephemeral_1h()),
            },
            Part::Image(MediaSource::Base64 {
                media_type: "image/png".into(),
                data: Bytes::from_static(&[1, 2, 3]),
            }),
            Part::Document {
                source: MediaSource::Text("doc body".into()),
                title: Some("T".into()),
                media_type: "text/plain".into(),
            },
            Part::Refusal { text: "no".into() },
            Part::Opaque(OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Redacted, "d")),
        ],
        id: Some(ItemId::new("msg_1")),
    };
    rt(item);
}

#[test]
fn item_variants_round_trip() {
    rt(Item::Reasoning(ReasoningItem {
        text: Some("t".into()),
        summary: vec!["a".into(), "b".into()],
        opaque: Some(OpaqueBlob::new(ProviderFamily::OpenAI, OpaqueKind::Encrypted, "e")),
        id: Some(ItemId::new("rs")),
    }));
    rt(Item::ToolCall {
        call_id: CallId::new("call_1"),
        name: "f".into(),
        arguments: JsonText::new("{\"a\":1}"),
        id: Some(ItemId::new("fc")),
    });
    rt(Item::ToolResult {
        call_id: CallId::new("call_1"),
        content: vec![Part::text("ok")],
        is_error: false,
        id: None,
    });
    rt(Item::ProviderToolCall(OpaqueItem::new(
        ProviderFamily::Anthropic,
        json!({"type": "server_tool_use", "name": "web_search"}),
    )));
    rt(Item::Compaction(OpaqueBlob::new(
        ProviderFamily::Anthropic,
        OpaqueKind::Compaction,
        "c",
    )));
}

#[test]
fn media_source_variants_round_trip() {
    rt(MediaSource::Base64 { media_type: "application/pdf".into(), data: Bytes::from_static(b"pdf") });
    rt(MediaSource::Url("https://x/y.png".into()));
    rt(MediaSource::FileRef { family: ProviderFamily::OpenAI, id: "file_1".into() });
    rt(MediaSource::Text("inline".into()));
}

#[test]
fn bytes_serialize_as_base64() {
    let s = MediaSource::Base64 { media_type: "image/png".into(), data: Bytes::from_static(&[1, 2, 3, 4]) };
    let v = serde_json::to_value(&s).unwrap();
    assert_eq!(v["base64"]["data"], json!("AQIDBA=="));
}

#[test]
fn ir_event_variants_round_trip() {
    rt(IrEvent::Start {
        response_id: ResponseId::new("r"),
        model: "m".into(),
        usage_prefill: Some(Usage::new(1, 0)),
    });
    rt(IrEvent::ItemStart {
        index: 0,
        kind: ItemKind::ToolCall,
        id: Some(ItemId::new("fc")),
        call: Some((CallId::new("c"), "name".into())),
    });
    for d in [
        Delta::Text("t".into()),
        Delta::ToolArgs("{".into()),
        Delta::ReasoningText("r".into()),
        Delta::ReasoningSummary { part: 2, text: "s".into() },
        Delta::Refusal("no".into()),
        Delta::Opaque(OpaqueBlob::new(ProviderFamily::Anthropic, OpaqueKind::Signature, "x")),
        Delta::ProviderRaw { family: ProviderFamily::Anthropic, raw: json!({"type": "server_tool_use"}) },
        Delta::Annotation(Annotation::new("k", json!({"a": 1}))),
    ] {
        rt(IrEvent::Delta { index: 1, delta: d });
    }
    rt(IrEvent::ItemStop { index: 0 });
    rt(IrEvent::Stop {
        reason: StopReason::StopSequence("STOP".into()),
        usage: Usage::new(2, 3),
        ext: Extensions::new(),
    });
    // Stop with response-level ext (skip_serializing_if keeps the empty case clean).
    let mut ext = Extensions::new();
    ext.insert("service_tier", json!("default"));
    rt(IrEvent::Stop { reason: StopReason::Refusal, usage: Usage::new(1, 1), ext });
    rt(IrEvent::Error(XlateError::invalid_request("bad")));
    // ItemKind carries the new compaction variant.
    for k in [ItemKind::Message, ItemKind::ProviderToolResult, ItemKind::Compaction] {
        rt(IrEvent::ItemStart { index: 0, kind: k, id: None, call: None });
    }
}

#[test]
fn output_format_and_tooldef_round_trip() {
    rt(OutputFormat::Text);
    rt(OutputFormat::JsonObject);
    rt(OutputFormat::JsonSchema {
        name: "S".into(),
        schema: json!({"type": "object"}),
        strict: true,
        description: Some("d".into()),
    });
    rt(ToolDef::Function {
        name: "f".into(),
        description: None,
        parameters: json!({"type": "object"}),
        strict: Some(false),
        cache_control: None,
    });
    rt(ToolDef::Provider(OpaqueItem::new(ProviderFamily::Anthropic, json!({"type": "web_search_20260318"}))));
}

#[test]
fn effort_ordering_and_budgets() {
    assert!(Effort::None < Effort::Minimal);
    assert!(Effort::Minimal < Effort::Low);
    assert!(Effort::Low < Effort::Medium);
    assert!(Effort::Medium < Effort::High);
    assert!(Effort::High < Effort::XHigh);
    assert!(Effort::XHigh < Effort::Max);

    assert_eq!(Effort::from_budget_tokens(0), Effort::Low);
    assert_eq!(Effort::from_budget_tokens(2047), Effort::Low);
    assert_eq!(Effort::from_budget_tokens(2048), Effort::Medium);
    assert_eq!(Effort::from_budget_tokens(8191), Effort::Medium);
    assert_eq!(Effort::from_budget_tokens(8192), Effort::High);
    assert_eq!(Effort::from_budget_tokens(24575), Effort::High);
    assert_eq!(Effort::from_budget_tokens(24576), Effort::XHigh);

    // ant_budget with a generous cap.
    assert_eq!(Effort::Minimal.ant_budget(64000), 1024);
    assert_eq!(Effort::Low.ant_budget(64000), 2048);
    assert_eq!(Effort::Medium.ant_budget(64000), 8192);
    assert_eq!(Effort::High.ant_budget(64000), 16384);
    assert_eq!(Effort::XHigh.ant_budget(64000), 32768);
    assert_eq!(Effort::Max.ant_budget(64000), 63999);
    // All capped at max_tokens - 1.
    assert_eq!(Effort::XHigh.ant_budget(1000), 999);
    assert_eq!(Effort::Max.ant_budget(1000), 999);
    assert_eq!(Effort::None.ant_budget(1000), 0);
}

#[test]
fn effort_openai_token() {
    assert_eq!(Effort::None.openai_token(), "none");
    assert_eq!(Effort::Minimal.openai_token(), "minimal");
    assert_eq!(Effort::Max.openai_token(), "xhigh");
    assert_eq!(Effort::XHigh.openai_token(), "xhigh");
}

#[test]
fn effort_ant_effort_token() {
    // Anthropic has no "minimal" (Minimal -> "low") and does accept "max".
    assert_eq!(Effort::None.ant_effort_token(), "low");
    assert_eq!(Effort::Minimal.ant_effort_token(), "low");
    assert_eq!(Effort::Low.ant_effort_token(), "low");
    assert_eq!(Effort::Medium.ant_effort_token(), "medium");
    assert_eq!(Effort::High.ant_effort_token(), "high");
    assert_eq!(Effort::XHigh.ant_effort_token(), "xhigh");
    assert_eq!(Effort::Max.ant_effort_token(), "max");
}

#[test]
fn usage_add_combines() {
    let mut a = Usage { input: 1, output: 2, cache_read: Some(3), ..Default::default() };
    let b = Usage { input: 10, output: 20, cache_write_5m: Some(5), reasoning: Some(7), ..Default::default() };
    a.add(&b);
    assert_eq!(a.input, 11);
    assert_eq!(a.output, 22);
    assert_eq!(a.cache_read, Some(3));
    assert_eq!(a.cache_write_5m, Some(5));
    assert_eq!(a.reasoning, Some(7));
}

#[test]
fn is_assistant_side_classification() {
    assert!(Item::assistant_text("x").is_assistant_side());
    assert!(!Item::user_text("x").is_assistant_side());
    assert!(Item::Reasoning(ReasoningItem::default()).is_assistant_side());
    assert!(Item::ProviderToolCall(OpaqueItem::new(ProviderFamily::Anthropic, json!({}))).is_assistant_side());
    assert!(Item::ProviderToolResult(OpaqueItem::new(ProviderFamily::Anthropic, json!({}))).is_assistant_side());
    assert!(Item::ToolResult { call_id: CallId::new("c"), content: vec![], is_error: false, id: None }.is_user_side());
}

#[test]
fn newtype_ergonomics() {
    let id = ItemId::new("abc");
    assert_eq!(&*id, "abc");
    assert_eq!(id.as_ref() as &str, "abc");
    assert_eq!(format!("{id}"), "abc");
    assert_eq!(ItemId::from("x"), ItemId::new("x"));
    assert_eq!(String::from(ItemId::new("y")), "y".to_string());
}

#[test]
fn extensions_key_order_is_deterministic() {
    let mut e = Extensions::new();
    e.insert("z", json!(1));
    e.insert("a", json!(2));
    // BTreeMap => sorted keys regardless of insertion order.
    assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"a":2,"z":1}"#);
}

#[test]
fn protocol_family_mapping() {
    assert_eq!(Protocol::OaiChat.family(), ProviderFamily::OpenAI);
    assert_eq!(Protocol::OaiResponses.family(), ProviderFamily::OpenAI);
    assert_eq!(Protocol::Anthropic.family(), ProviderFamily::Anthropic);
    assert_eq!(serde_json::to_string(&ProviderFamily::Other("x".into())).unwrap(), r#""x""#);
    assert_eq!(
        serde_json::from_str::<ProviderFamily>(r#""anthropic""#).unwrap(),
        ProviderFamily::Anthropic
    );
}
