//! Plain-text reasoning survives an Anthropic-client round trip to a Chat backend (plan §7.2).
//!
//! Reduced from a captured production trace: an Anthropic-dialect agent driving an
//! OpenAI-compatible backend that returns reasoning as plain `reasoning_content` and declares
//! `reasoning.replay = "text_field"`. The content here is generic placeholder text; only the
//! shape of the round trip is kept.
//!
//! The backend's reasoning made it out to the client correctly — as a `thinking` block with no
//! `signature`, since there is no opaque carrier to attach — and was destroyed on the way back
//! in. `decode_thinking_block` wrapped *every* thinking block in an opaque carrier, so an
//! absent or empty signature became a native Anthropic blob with empty data. `lower()` then saw
//! an Anthropic-family blob bound for an OpenAI-family target and dropped it: "foreign-family
//! reasoning blob dropped (not replayable)". Every turn of the captured conversation lost all
//! of its reasoning that way, even though the backend could have replayed it as text.
//!
//! The client-facing encoders already documented the intended contract — "on replay the router
//! decoder treats a missing signature as no opaque" — so only the decoder had to change.
//!
//! `redacted_thinking` carried the same flaw and is covered here too: an empty `data` is no
//! carrier either, and claiming a native Anthropic blob made every downstream message describe
//! something that was never there.

mod common;

use llm_xlate::caps::{
    BackendOverrides, Capabilities, ReasoningCap, ReplayMode, ToolsCap, Tri,
};
use llm_xlate::codec::{EncodeCtx, TranslatorConfig};
use llm_xlate::envelope::Sealer;
use llm_xlate::ir::{
    Item, Protocol, ProviderFamily, ReasoningExposure, ResponseId,
};
use llm_xlate::requirements::Resolutions;
use llm_xlate::{lower, HeaderMap, Translator};
use serde_json::Value;

const REASONING: &str = "REASONING-TEXT";

/// The backend: OpenAI-compatible, returns reasoning as text and can replay it the same way.
fn backend_caps() -> Capabilities {
    let overlay = BackendOverrides {
        reasoning: ReasoningCap {
            replay: Some(ReplayMode::TextField),
            exposure: Some(llm_xlate::caps::ExposureMode::FullText),
            ..Default::default()
        },
        tools: ToolsCap { function_tools: Tri::Yes, ..Default::default() },
        ..Default::default()
    };
    llm_xlate::caps::shipped().resolve(
        &ProviderFamily::Other("openai-compatible".to_string()),
        "test-model",
        Some(&overlay),
    )
}

fn xl() -> Translator {
    Translator::new(TranslatorConfig::default())
}

/// Leg 1: what the backend returned, rendered for the Anthropic client.
///
/// A Chat completion carrying `reasoning_content` → IR → the client's `thinking` block.
fn thinking_block_sent_to_the_client() -> Value {
    let x = xl();
    let caps = backend_caps();
    let upstream = format!(
        r#"{{"id":"c1","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{{"index":0,"finish_reason":"stop","message":{{
                "role":"assistant","reasoning_content":"{REASONING}","content":"answer"}}}}]}}"#
    );
    let events = x.decode_response(Protocol::OaiChat, upstream.as_bytes(), &caps).expect("decodes");
    let resp = x.aggregate_stream(events).expect("aggregates");

    let sealer = Sealer::new(&TranslatorConfig::default().envelope_key);
    let mut ctx = EncodeCtx::new(Protocol::Anthropic, "test-model", ResponseId::new("r1"), sealer);
    ctx.expose = ReasoningExposure::Full;
    let body: Value =
        serde_json::from_slice(&x.encode_response(Protocol::Anthropic, &resp, &ctx)).unwrap();

    body["content"]
        .as_array()
        .expect("content array")
        .iter()
        .find(|b| b["type"] == "thinking")
        .cloned()
        .expect("a thinking block reached the client")
}

/// Leg 2: the client replays that block on the next turn. Returns the encoded upstream body and
/// the lowering degradations.
fn replay(thinking: Value) -> (String, Vec<String>) {
    let x = xl();
    let caps = backend_caps();
    let request = serde_json::json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [thinking, {"type": "text", "text": "answer"}]},
            {"role": "user", "content": "q2"}
        ]
    });
    let ir = x
        .decode_request(Protocol::Anthropic, request.to_string().as_bytes(), &HeaderMap::new())
        .expect("decodes");
    let low = lower(
        ir.clone(),
        &caps,
        Protocol::OaiChat,
        &Resolutions::new(),
        &TranslatorConfig::default(),
    )
    .expect("lowers");
    let ctx = x.encode_ctx(Protocol::OaiChat, &ir, ResponseId::new("r2"), 0);
    let enc = x.encode_request(Protocol::OaiChat, &low.req, &caps, &ctx).expect("encodes");
    (
        String::from_utf8(enc.body.to_vec()).expect("utf8 body"),
        low.degradations.iter().map(|d| d.detail.clone()).collect(),
    )
}

/// The outbound leg was already correct and must stay that way: no opaque carrier means no
/// `signature` key, rather than a fabricated empty one.
#[test]
fn plaintext_reasoning_reaches_the_client_without_a_signature() {
    let block = thinking_block_sent_to_the_client();
    assert_eq!(block["thinking"], Value::from(REASONING));
    assert!(
        block.get("signature").is_none(),
        "a signature must not be fabricated for carrier-less reasoning: {block:#?}"
    );
}

/// The reported failure. The client echoes the block back with an empty `signature` (its own
/// types require the field), and the reasoning has to survive to the backend as text.
#[test]
fn replayed_plaintext_reasoning_survives_to_the_backend() {
    let mut block = thinking_block_sent_to_the_client();
    block["signature"] = Value::from("");

    let (body, degradations) = replay(block);

    assert!(
        body.contains(REASONING),
        "reasoning was dropped on replay; upstream body: {body}"
    );
    assert!(
        !degradations.iter().any(|d| d.contains("foreign-family")),
        "reasoning must not be treated as foreign: {degradations:?}"
    );
}

/// The same holds when the client omits the key entirely — an absent signature and an empty one
/// must not diverge, or a faithful client would still lose its reasoning.
#[test]
fn an_absent_signature_behaves_like_an_empty_one() {
    let block = thinking_block_sent_to_the_client();
    assert!(block.get("signature").is_none());

    let (body, degradations) = replay(block);

    assert!(body.contains(REASONING), "reasoning was dropped on replay; upstream body: {body}");
    assert!(
        !degradations.iter().any(|d| d.contains("foreign-family")),
        "reasoning must not be treated as foreign: {degradations:?}"
    );
}

/// The guard that keeps the fix honest: a genuine Anthropic thinking block carries a real
/// signature, and that still decodes to a native Anthropic carrier — which is correctly foreign
/// to an OpenAI-family target and still dropped.
#[test]
fn a_real_signature_is_still_a_native_carrier() {
    let x = xl();
    let request = serde_json::json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": REASONING, "signature": "real-signature"}]},
            {"role": "user", "content": "q2"}
        ]
    });
    let ir = x
        .decode_request(Protocol::Anthropic, request.to_string().as_bytes(), &HeaderMap::new())
        .expect("decodes");

    let blob = ir
        .items
        .iter()
        .find_map(|i| match i {
            Item::Reasoning(r) => r.opaque.clone(),
            _ => None,
        })
        .expect("a signed thinking block keeps its opaque carrier");
    assert_eq!(blob.family, ProviderFamily::Anthropic);
    assert_eq!(blob.data, "real-signature");

    let low = lower(
        ir,
        &backend_caps(),
        Protocol::OaiChat,
        &Resolutions::new(),
        &TranslatorConfig::default(),
    )
    .expect("lowers");
    assert!(
        low.degradations.iter().any(|d| d.detail.contains("foreign-family")),
        "a native Anthropic blob is still foreign to a Chat target: {:?}",
        low.degradations
    );
}

/// The sibling shape: `redacted_thinking` whose `data` is empty carries nothing, so it must not
/// yield a carrier either. The item is inert — no text, summary or blob — and the encoders and
/// `lower()` each describe it accurately instead of reporting a foreign Anthropic blob.
#[test]
fn an_empty_redacted_block_yields_no_carrier() {
    let x = xl();
    let request = serde_json::json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": ""},
                {"type": "text", "text": "answer"}]},
            {"role": "user", "content": "q2"}
        ]
    });
    let ir = x
        .decode_request(Protocol::Anthropic, request.to_string().as_bytes(), &HeaderMap::new())
        .expect("decodes");

    let reasoning = ir
        .items
        .iter()
        .find_map(|i| match i {
            Item::Reasoning(r) => Some(r.clone()),
            _ => None,
        })
        .expect("the block still decodes to a reasoning item");
    assert!(reasoning.opaque.is_none(), "an empty `data` is not a carrier: {reasoning:?}");

    let low = lower(
        ir,
        &backend_caps(),
        Protocol::OaiChat,
        &Resolutions::new(),
        &TranslatorConfig::default(),
    )
    .expect("lowers");
    assert!(
        !low.degradations.iter().any(|d| d.detail.contains("foreign-family")),
        "nothing foreign was ever carried here: {:?}",
        low.degradations
    );
}

/// A `redacted_thinking` block that does carry data is untouched: still a native Anthropic
/// carrier, still correctly foreign to a Chat target.
#[test]
fn a_populated_redacted_block_keeps_its_carrier() {
    let x = xl();
    let request = serde_json::json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "REDACTED-BLOB"}]},
            {"role": "user", "content": "q2"}
        ]
    });
    let ir = x
        .decode_request(Protocol::Anthropic, request.to_string().as_bytes(), &HeaderMap::new())
        .expect("decodes");

    let blob = ir
        .items
        .iter()
        .find_map(|i| match i {
            Item::Reasoning(r) => r.opaque.clone(),
            _ => None,
        })
        .expect("a populated redacted block keeps its carrier");
    assert_eq!(blob.family, ProviderFamily::Anthropic);
    assert_eq!(blob.data, "REDACTED-BLOB");
}
