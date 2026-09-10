//! Round-trip and capability-matrix tests: `encode(decode(bytes))` byte-stable snapshots for
//! realistic fixtures, `decode(encode(ir))` structural stability, and the same IR encoded
//! against several capability presets.

mod common;

use common::*;
use llm_xlate_anthropic::AnthropicCodec;
use llm_xlate_core::{Codec, Effort, Item, IrRequest, Part, Protocol, ReasoningConfig};
use pretty_assertions::assert_eq;

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path).expect("fixture file")
}

fn decode(body: &str) -> IrRequest {
    AnthropicCodec.decode_request(body.as_bytes(), &no_headers(), &dctx()).unwrap()
}

fn encode(req: &IrRequest) -> Vec<u8> {
    AnthropicCodec
        .encode_request(req, &claude_5(), &ectx(Protocol::Anthropic))
        .unwrap()
        .body
        .to_vec()
}

/// `encode(decode(bytes))` is byte-stable and deterministic for each fixture.
fn roundtrip_snapshot(name: &str, snap: &str) {
    let body = fixture(name);
    let req = decode(&body);
    let a = encode(&req);
    let b = encode(&req);
    assert_eq!(a, b, "re-encode must be deterministic for {name}");
    insta::assert_snapshot!(snap, pretty(&a));
}

#[test]
fn roundtrip_tools_thinking() {
    roundtrip_snapshot("req_tools_thinking.json", "rt_tools_thinking");
}

#[test]
fn roundtrip_multimodal() {
    roundtrip_snapshot("req_multimodal.json", "rt_multimodal");
}

#[test]
fn roundtrip_structured() {
    roundtrip_snapshot("req_structured.json", "rt_structured");
}

/// `decode(encode(decode(bytes)))` yields the same IR as the first decode for native inputs
/// (the transformation is idempotent through a re-encode) — checked on the core fields that
/// survive a native Anthropic round trip.
#[test]
fn decode_encode_decode_is_stable() {
    for name in ["req_tools_thinking.json", "req_multimodal.json", "req_structured.json"] {
        let first = decode(&fixture(name));
        let bytes = encode(&first);
        let second = decode(std::str::from_utf8(&bytes).unwrap());
        assert_eq!(first.items, second.items, "items diverged for {name}");
        assert_eq!(first.tools, second.tools, "tools diverged for {name}");
        assert_eq!(first.model, second.model, "model diverged for {name}");
        assert_eq!(first.output.format, second.output.format, "format diverged for {name}");
    }
}

/// A decode → encode → decode round trip over a rich hand-written request must preserve the
/// cross-cutting IR fields. Crucially it exercises a MULTI-BLOCK top-level `system` whose
/// `cache_control` sits on the FIRST (non-last) block — the cached-preamble + dynamic-tail
/// pattern that previously gained a spurious `cache_control` breakpoint on the trailing block
/// (instruction-level cache_control re-application). It also covers reasoning config, sampling,
/// tool_choice, cache hints, and meta.
#[test]
fn decode_encode_decode_preserves_rich_request() {
    let body = r#"{
        "model":"claude-3-5-sonnet",
        "max_tokens":1024,
        "system":[
            {"type":"text","text":"Cached preamble.","cache_control":{"type":"ephemeral"}},
            {"type":"text","text":"Dynamic tail."}
        ],
        "messages":[{"role":"user","content":"hello"}],
        "tools":[{"name":"get","description":"Get","input_schema":{"type":"object"}}],
        "tool_choice":{"type":"auto"},
        "thinking":{"type":"disabled"},
        "temperature":0.5,
        "stop_sequences":["STOP"],
        "metadata":{"user_id":"u1"},
        "cache_control":{"type":"ephemeral"}
    }"#;
    let first = decode(body);

    // Precondition: the cached preamble sits on the FIRST of two system blocks, and no
    // instruction-level breakpoint was fabricated.
    let sys = &first.instructions[0];
    assert_eq!(sys.content.len(), 2, "expected a two-block system");
    assert!(matches!(&sys.content[0], Part::Text { cache_control: Some(_), .. }));
    assert!(matches!(&sys.content[1], Part::Text { cache_control: None, .. }));
    assert!(sys.cache_control.is_none(), "instruction-level cache_control must stay None");

    // claude_37 permits the sampling values (temperature is rejected on claude_5) *and* has a
    // (budget-mode) thinking mode, so reasoning survives the round trip (claude_old / 3.5 has no
    // thinking mode and would drop it).
    let bytes = AnthropicCodec
        .encode_request(&first, &claude_37(), &ectx(Protocol::Anthropic))
        .unwrap()
        .body;
    let second = decode(std::str::from_utf8(&bytes).unwrap());

    assert_eq!(first.instructions, second.instructions, "instructions diverged");
    assert_eq!(first.reasoning, second.reasoning, "reasoning diverged");
    assert_eq!(first.sampling, second.sampling, "sampling diverged");
    assert_eq!(first.tool_choice, second.tool_choice, "tool_choice diverged");
    assert_eq!(first.cache, second.cache, "cache diverged");
    assert_eq!(first.meta, second.meta, "meta diverged");
}

// -- capability matrix ------------------------------------------------------------------

fn reasoning_req(model: &str) -> IrRequest {
    let mut r = IrRequest {
        model: llm_xlate_core::ModelRef::new(model),
        items: vec![Item::user_text("hi")],
        ..Default::default()
    };
    r.limits.max_output_tokens = Some(4096);
    r.reasoning = ReasoningConfig { effort: Some(Effort::High), enabled: Some(true), ..Default::default() };
    r
}

fn thinking_json(req: &IrRequest, caps: &llm_xlate_core::Capabilities) -> serde_json::Value {
    let body = AnthropicCodec.encode_request(req, caps, &ectx(Protocol::Anthropic)).unwrap().body;
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    v.get("thinking").cloned().unwrap_or(serde_json::Value::Null)
}

#[test]
fn caps_matrix_reasoning_shape_differs() {
    // Adaptive model -> {"type":"adaptive"}; budget model -> {"type":"enabled", budget_tokens}.
    let adaptive = thinking_json(&reasoning_req("claude-opus-5"), &claude_5());
    assert_eq!(adaptive, serde_json::json!({"type": "adaptive"}));

    // Budget-mode thinking requires Claude 3.7+ (Claude 3.5 / claude_old has no thinking mode).
    let budget = thinking_json(&reasoning_req("claude-3-7-sonnet"), &claude_37());
    assert_eq!(budget["type"], serde_json::json!("enabled"));
    assert!(budget["budget_tokens"].is_number());
}

#[test]
fn caps_matrix_effort_snapshot() {
    let req = reasoning_req("claude-opus-5");
    let out = AnthropicCodec.encode_request(&req, &claude_5(), &ectx(Protocol::Anthropic)).unwrap();
    insta::assert_snapshot!("caps_adaptive_high", pretty(&out.body));
}

#[test]
fn caps_matrix_sampling_gate() {
    // temperature accepted on old model, rejected on claude_5.
    let mut req = reasoning_req("claude-3-5-sonnet");
    req.reasoning = Default::default();
    req.sampling.temperature = Some(0.5);
    let old = AnthropicCodec.encode_request(&req, &claude_old(), &ectx(Protocol::Anthropic)).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&old.body).unwrap();
    assert_eq!(v["temperature"], serde_json::json!(0.5));

    let mut req5 = req.clone();
    req5.model = llm_xlate_core::ModelRef::new("claude-opus-5");
    let five = AnthropicCodec.encode_request(&req5, &claude_5(), &ectx(Protocol::Anthropic)).unwrap();
    let v5: serde_json::Value = serde_json::from_slice(&five.body).unwrap();
    assert!(v5.get("temperature").is_none());
    assert!(!five.degradations.is_empty());
}

#[test]
fn empty_request_encodes_minimally() {
    let mut req = IrRequest { model: llm_xlate_core::ModelRef::new("claude-opus-5"), ..Default::default() };
    req.limits.max_output_tokens = Some(16);
    let out = AnthropicCodec.encode_request(&req, &claude_5(), &ectx(Protocol::Anthropic)).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.body).unwrap();
    assert_eq!(v["model"], serde_json::json!("claude-opus-5"));
    assert_eq!(v["max_tokens"], serde_json::json!(16));
    assert_eq!(v["messages"], serde_json::json!([]));
}
