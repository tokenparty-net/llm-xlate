//! Façade-method coverage (plan §4, §8, §9): the [`Translator`] convenience helpers that the
//! pair-matrix runner does not otherwise exercise — `translate_request`, `synthesize_stream`,
//! and the stored-response encoders.

mod common;

use llm_xlate::caps::preset;
use llm_xlate::codec::{EncodeCtx, TranslatorConfig};
use llm_xlate::error::ErrorKind;
use llm_xlate::envelope::Sealer;
use llm_xlate::ir::{Item, Protocol, ReasoningExposure, ResponseId, StopReason};
use llm_xlate::requirements::Resolutions;
use llm_xlate::store::{to_stored, BackendBinding, StoredStatus};
use llm_xlate::{HeaderMap, Translator};
use llm_xlate_core::ProviderFamily;

fn xl() -> Translator {
    Translator::new(TranslatorConfig::default())
}

const CREATED_AT: u64 = 1_700_000_000;

// ===========================================================================================
// translate_request (plan §4): decode → requirements → lower → encode in one call, with the
// lowering degradations merged ahead of the codec's wiring degradations.
// ===========================================================================================

/// Regression: a Chat→Chat agentic loop whose assistant turn carries a plain-text
/// `reasoning_content` must survive translation to a backend that both requires reasoning on the
/// last tool turn and replays it through that same text field. It used to be rejected as
/// `incompatible_history` because only opaque carriers counted as replayable reasoning.
#[test]
fn text_reasoning_replays_to_a_textfield_backend() {
    let body = br#"{
        "model": "kimi-k3",
        "messages": [
            {"role": "user", "content": "explore this project"},
            {"role": "assistant",
             "content": "I'll look at the structure first.",
             "reasoning_content": "The user wants me to explore. Let me look around.",
             "tool_calls": [{"id":"call_1","type":"function",
                             "function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a.txt b.txt"}
        ],
        "tools": [{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
        "reasoning_effort": "high"
    }"#;
    let mut caps = preset::gpt4o();
    caps.reasoning.replay = Some(llm_xlate::caps::ReplayMode::TextField);
    caps.reasoning.required_on_last_tool_turn = llm_xlate::caps::Tri::Yes;

    let x = xl();
    let (enc, reqs) = x
        .translate_request(
            Protocol::OaiChat,
            body,
            &HeaderMap::new(),
            Protocol::OaiChat,
            &caps,
            &Resolutions::new(),
        )
        .expect("a replayable text reasoning turn must not be rejected");

    // The text is already the carrier, so the router is asked to resolve nothing …
    assert!(reqs.reasoning_for_calls.is_empty());
    // … and it goes back out on the wire for the backend to see.
    let out = String::from_utf8(enc.body.to_vec()).unwrap();
    assert!(out.contains("reasoning_content"), "reasoning text was not replayed: {out}");
}

#[test]
fn translate_request_matches_stepwise() {
    // A Chat request carrying a tool + reasoning so the lowering to Anthropic produces real
    // degradations to check the merge order on.
    let body = br#"{
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],
        "reasoning_effort": "high",
        "temperature": 0.7
    }"#;
    let x = xl();
    let caps = preset::claude_5();
    let res = Resolutions::new();

    let (enc, reqs) = x
        .translate_request(Protocol::OaiChat, body, &HeaderMap::new(), Protocol::Anthropic, &caps, &res)
        .expect("translate_request ok");

    // Stepwise reconstruction: exactly what the façade is documented to do internally.
    let ir = x.decode_request(Protocol::OaiChat, body, &HeaderMap::new()).unwrap();
    let reqs2 = x.requirements(&ir, &caps, Protocol::Anthropic);
    let ctx = x.encode_ctx(Protocol::OaiChat, &ir, ResponseId::new(""), 0);
    let low = x.lower(ir, &caps, Protocol::Anthropic, &res).unwrap();
    let enc2 = x.encode_request(Protocol::Anthropic, &low.req, &caps, &ctx).unwrap();

    assert_eq!(enc.body, enc2.body, "translate_request body differs from stepwise");
    assert_eq!(reqs, reqs2, "translate_request requirements differ from stepwise");
    assert!(!enc.body.is_empty(), "translate_request produced an empty body");

    // Degradations: lowering degradations first, then the codec's wiring degradations.
    let expected: Vec<(String, _)> = low
        .degradations
        .iter()
        .chain(enc2.degradations.iter())
        .map(|d| (d.field.clone(), d.kind))
        .collect();
    let got: Vec<(String, _)> =
        enc.degradations.iter().map(|d| (d.field.clone(), d.kind)).collect();
    assert_eq!(got, expected, "translate_request merged degradations differ / out of order");

    // ---- Independent assertions on concrete expected behavior (not just self-consistency) ----
    // Lowering Chat→claude-opus-5 (sampling rejected) drops `temperature` (0.7) with a genuine
    // Dropped degradation, and — crucially — the codec-internal `ext.chat.max_tokens_field`
    // marker is NOT reported as a false-positive drop (finding 5 allowlist).
    use llm_xlate::degrade::DegradationKind;
    assert_eq!(
        got,
        vec![("temperature".to_string(), DegradationKind::Dropped)],
        "expected exactly a temperature drop; got {got:?}"
    );

    // The Anthropic body materializes the tool, adaptive thinking, and the effort override; and
    // carries no `temperature` (dropped above).
    let body: serde_json::Value = serde_json::from_slice(&enc.body).unwrap();
    assert_eq!(body["thinking"]["type"], "adaptive", "adaptive thinking block: {body}");
    assert_eq!(body["output_config"]["effort"], "high", "effort override: {body}");
    assert_eq!(body["tools"][0]["name"], "get_weather", "function tool carried: {body}");
    assert!(body.get("temperature").is_none(), "temperature must be dropped: {body}");
}

// ===========================================================================================
// synthesize_stream (plan §8): a non-streaming provider body → a streaming client of a
// *different* protocol. The synthesized frames must decode+aggregate back to the same
// IrResponse as decoding the provider body directly.
// ===========================================================================================

fn golden(sub: &str, file: &str) -> Vec<u8> {
    let path = format!("{}/tests/golden/{sub}/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn client_stream_ctx(client: Protocol) -> EncodeCtx {
    let sealer = Sealer::new(&TranslatorConfig::default().envelope_key);
    let mut ctx = EncodeCtx::new(client, "golden-model", ResponseId::new("resp_syn"), sealer);
    ctx.created_at = CREATED_AT;
    ctx.expose = ReasoningExposure::Full;
    ctx.include_usage = true;
    ctx.stream = true;
    ctx
}

fn strip_ids(items: &[Item]) -> Vec<Item> {
    items
        .iter()
        .cloned()
        .map(|it| match it {
            Item::Message { role, content, .. } => Item::Message { role, content, id: None },
            Item::ToolCall { call_id, name, arguments, .. } => {
                Item::ToolCall { call_id, name, arguments, id: None }
            }
            Item::ToolResult { call_id, content, is_error, .. } => {
                Item::ToolResult { call_id, content, is_error, id: None }
            }
            Item::Reasoning(mut r) => {
                r.id = None;
                Item::Reasoning(r)
            }
            other => other,
        })
        .collect()
}

/// Feed a non-streaming provider `*.resp.json` through `synthesize_stream` into a streaming
/// client of a different protocol, then decode+aggregate the frames and assert they match the
/// provider body decoded directly.
fn assert_synthesize_roundtrips(provider: Protocol, sub: &str, file: &str, client: Protocol) {
    assert_ne!(provider, client, "the mismatch bridge is only interesting cross-protocol");
    let x = xl();
    let provider_caps = match provider {
        Protocol::OaiChat => preset::gpt5_chat(),
        Protocol::OaiResponses => preset::gpt5_responses(),
        Protocol::Anthropic => preset::claude_5(),
    };
    let client_caps = match client {
        Protocol::OaiChat => preset::gpt5_chat(),
        Protocol::OaiResponses => preset::gpt5_responses(),
        Protocol::Anthropic => preset::claude_5(),
    };
    let body = golden(sub, file);

    // Baseline: decode the provider body directly and aggregate (provider dialect).
    let base_events = x.decode_response(provider, &body, &provider_caps).expect("decode_response");
    let base = x.aggregate_stream(base_events).expect("aggregate base");

    // Bridge: synthesize a client stream from the same non-streaming body, then decode+aggregate.
    let frames = x
        .synthesize_stream(provider, &body, &provider_caps, client_stream_ctx(client))
        .expect("synthesize_stream");
    assert!(!frames.is_empty(), "synthesize_stream produced no frames");
    let joined: Vec<u8> = frames.iter().flat_map(|b| b.iter().copied()).collect();
    let mut dec = x.stream_decoder(client, &client_caps);
    let mut redec = dec.push(&joined);
    redec.extend(dec.finish());
    let via_synth = x.aggregate_stream(redec).expect("aggregate via synthesized stream");

    // The synthesized *stream* must render the response identically to the client's
    // *non-streaming* `encode_response` of the same `base` (plan §11.8, applied to the §8
    // mismatch bridge). Both go through the client dialect's documented normalizations, so we
    // compare the two client-side renderings rather than the raw provider-dialect `base`
    // (cross-protocol reasoning/opaque normalization is expected and is not identity).
    let mut nsctx = client_stream_ctx(client);
    nsctx.stream = false;
    let client_body = x.encode_response(client, &base, &nsctx);
    let client_events = x.decode_response(client, &client_body, &client_caps).expect("decode client");
    let via_encode = x.aggregate_stream(client_events).expect("aggregate via client encode");

    assert_eq!(
        strip_ids(&via_encode.items),
        strip_ids(&via_synth.items),
        "{provider:?}->{client:?}: synthesized-stream items disagree with non-streaming encode"
    );
    assert_eq!(
        via_encode.stop, via_synth.stop,
        "{provider:?}->{client:?}: stop differs between stream and non-stream client rendering"
    );
    assert_eq!(
        via_encode.usage.output, via_synth.usage.output,
        "{provider:?}->{client:?}: output usage differs between stream and non-stream rendering"
    );
}

#[test]
fn synthesize_stream_responses_to_chat() {
    assert_synthesize_roundtrips(Protocol::OaiResponses, "responses", "completed.resp.json", Protocol::OaiChat);
}

#[test]
fn synthesize_stream_chat_to_responses() {
    assert_synthesize_roundtrips(Protocol::OaiChat, "chat", "toolcalls.resp.json", Protocol::OaiResponses);
}

#[test]
fn synthesize_stream_anthropic_to_chat() {
    assert_synthesize_roundtrips(Protocol::Anthropic, "anthropic", "text.resp.json", Protocol::OaiChat);
}

// ===========================================================================================
// Stored-response encoders (plan §9): the Responses-only `GET /responses/{id}` +
// `/input_items` bodies, and the Unsupported branch for the other client protocols.
// ===========================================================================================

fn sample_stored(status: StoredStatus) -> llm_xlate::store::StoredResponse {
    let mut req = common::req_with_items(vec![common::user("Q1")]);
    req.model = llm_xlate::ir::ModelRef::new("gpt-5.4");
    req.state.previous_response_id = Some(ResponseId::new("resp_prev"));
    let out = common::resp(vec![common::asst("A1")], StopReason::EndTurn);
    let binding = BackendBinding::new("cred_1", ProviderFamily::OpenAI, "gpt-5.4");
    let mut s = to_stored(
        &req,
        &out,
        binding,
        ResponseId::new("resp_stored_1"),
        CREATED_AT,
        Default::default(),
    );
    s.status = status;
    s
}

#[test]
fn encode_stored_response_all_statuses_snapshot() {
    let x = xl();
    let statuses = [
        StoredStatus::Queued,
        StoredStatus::InProgress,
        StoredStatus::Completed,
        StoredStatus::Incomplete,
        StoredStatus::Failed,
        StoredStatus::Cancelled,
    ];
    let mut out = String::new();
    for st in statuses {
        let s = sample_stored(st);
        let body = x.encode_stored_response(Protocol::OaiResponses, &s).expect("encode stored");
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        out.push_str(&format!("=== {st:?} ===\n"));
        out.push_str(&serde_json::to_string_pretty(&v).unwrap());
        out.push('\n');
    }
    insta::assert_snapshot!("encode_stored_response__statuses", out);
}

#[test]
fn encode_input_items_snapshot() {
    let x = xl();
    let s = sample_stored(StoredStatus::Completed);
    let body = x.encode_input_items(&s);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    insta::assert_snapshot!("encode_input_items", serde_json::to_string_pretty(&v).unwrap());
}

#[test]
fn encode_stored_response_unsupported_for_non_responses() {
    let x = xl();
    let s = sample_stored(StoredStatus::Completed);
    for p in [Protocol::OaiChat, Protocol::Anthropic] {
        let err = x
            .encode_stored_response(p, &s)
            .expect_err("stored-response retrieval is Responses-only");
        assert_eq!(err.kind, ErrorKind::Unsupported, "{p:?} should be Unsupported");
    }
}

/// Claude Code's per-request `x-anthropic-billing-header:` system block is metadata, not prompt
/// text. Translated to a non-Anthropic target it must not become the first line of the
/// instructions (it changes every request, so it would defeat prefix caching); the drop is
/// reported once as a foreign ext key.
#[test]
fn claude_code_billing_header_never_reaches_a_chat_backend() {
    let body = br#"{
        "model": "claude-opus-5",
        "max_tokens": 1024,
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.278.a6d; cch=27699;"},
            {"type": "text", "text": "You are Claude Code."}
        ],
        "messages": [{"role": "user", "content": "hi"}]
    }"#;
    let (enc, _) = xl()
        .translate_request(
            Protocol::Anthropic,
            body,
            &HeaderMap::new(),
            Protocol::OaiChat,
            &preset::gpt4o(),
            &Resolutions::new(),
        )
        .unwrap();
    let out = String::from_utf8(enc.body.to_vec()).unwrap();
    assert!(!out.contains("x-anthropic-billing-header"), "header leaked: {out}");
    assert!(out.contains("You are Claude Code."), "prompt lost: {out}");
    assert!(enc.degradations.iter().any(|d| d.field == "ext.anthropic.system_headers"));
}
