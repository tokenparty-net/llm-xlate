//! The backend's streaming mode decides how the upstream request is sent (plan §6, §8).
//!
//! From a reported failure: a connector converted to an Anthropic-shaped, **stream-only** API
//! rejected every client that did not itself ask for a stream. `Streaming::StreamOnly` was
//! declared in the capability schema but read by nothing — each encoder derived the request's
//! `stream` flag from the client's intent alone (`req.stream && caps != NonStreamOnly`), so a
//! non-streaming client produced an upstream body with no `stream` key.
//!
//! It went unnoticed because the OpenAI-compatible backends in use answer SSE regardless, and
//! the router believes the response's content type rather than what it asked for — so the
//! aggregation path worked and the request shape was never the thing that failed.
//!
//! Two independent facts are asserted here, because conflating them was the second half of the
//! bug: what goes **upstream** follows the backend's mode, and `EncodeCtx::stream` stays the
//! **client's** intent, since the router reads it back to decide whether to stream to the
//! caller.

mod common;

use llm_xlate::caps::{BackendOverrides, Capabilities, Streaming, ToolsCap, Tri, TransportCap};
use llm_xlate::codec::TranslatorConfig;
use llm_xlate::ir::{IrRequest, ModelRef, Protocol, ProviderFamily, ResponseId};
use llm_xlate::Translator;
use serde_json::Value;

/// Caps for `protocol` with an explicit streaming mode.
fn caps_for(protocol: Protocol, streaming: Streaming) -> Capabilities {
    let overlay = BackendOverrides {
        transport: TransportCap {
            protocols: Some(vec![protocol]),
            streaming: Some(streaming),
            ..Default::default()
        },
        tools: ToolsCap { function_tools: Tri::Yes, ..Default::default() },
        ..Default::default()
    };
    llm_xlate::caps::shipped().resolve(
        &ProviderFamily::Other("test-backend".to_string()),
        "test-model",
        Some(&overlay),
    )
}

/// Encode a minimal request and report `(upstream body "stream" flag, EncodedRequest::
/// upstream_streams, EncodeCtx::stream)`.
fn encode(protocol: Protocol, streaming: Streaming, client_wants_stream: bool) -> (bool, bool, bool) {
    let xl = Translator::new(TranslatorConfig::default());
    let caps = caps_for(protocol, streaming);
    let req = IrRequest {
        model: ModelRef::new("test-model"),
        items: vec![common::user("hi")],
        limits: llm_xlate::ir::Limits { max_output_tokens: Some(64), ..Default::default() },
        stream: client_wants_stream,
        ..Default::default()
    };
    let ctx = xl.encode_ctx(protocol, &req, ResponseId::new("r1"), 0);
    let enc = xl.encode_request(protocol, &req, &caps, &ctx).expect("encodes");
    let body: Value = serde_json::from_slice(&enc.body).expect("valid JSON body");
    let on_wire = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    (on_wire, enc.upstream_streams, enc.ctx.stream)
}

const PROTOCOLS: [Protocol; 3] = [Protocol::OaiChat, Protocol::OaiResponses, Protocol::Anthropic];

/// The reported failure: a stream-only backend must be sent `stream: true` even when the client
/// asked for a single JSON body. The router aggregates the SSE back into one response.
#[test]
fn a_stream_only_backend_is_always_sent_a_streaming_request() {
    for p in PROTOCOLS {
        for client_wants_stream in [false, true] {
            let (on_wire, upstream_streams, _) =
                encode(p, Streaming::StreamOnly, client_wants_stream);
            assert!(
                on_wire,
                "{p:?}: stream-only backend must receive `stream: true` \
                 (client asked stream={client_wants_stream})"
            );
            assert!(upstream_streams, "{p:?}: the router must be told to read SSE");
        }
    }
}

/// The mirror case: a non-streaming-only backend is never sent `stream`, even for a streaming
/// client — the client's stream is synthesized from the response instead.
#[test]
fn a_non_stream_only_backend_is_never_sent_a_streaming_request() {
    for p in PROTOCOLS {
        for client_wants_stream in [false, true] {
            let (on_wire, upstream_streams, _) =
                encode(p, Streaming::NonStreamOnly, client_wants_stream);
            assert!(
                !on_wire,
                "{p:?}: non-streaming-only backend must not receive `stream` \
                 (client asked stream={client_wants_stream})"
            );
            assert!(!upstream_streams, "{p:?}: the router must be told to read a JSON body");
        }
    }
}

/// When the backend accepts both, the client's own choice carries through unchanged.
#[test]
fn a_dual_mode_backend_follows_the_client() {
    for p in PROTOCOLS {
        for client_wants_stream in [false, true] {
            let (on_wire, upstream_streams, _) = encode(p, Streaming::Both, client_wants_stream);
            assert_eq!(on_wire, client_wants_stream, "{p:?}: body flag must follow the client");
            assert_eq!(upstream_streams, client_wants_stream, "{p:?}: read mode follows too");
        }
    }
}

/// The second half of the bug. `EncodeCtx::stream` is what the router reads back to decide
/// whether to stream to the caller, so it must stay the client's intent no matter what the
/// backend's mode forced upstream — otherwise a stream-only backend pushes SSE at a client that
/// asked for one JSON body.
#[test]
fn the_client_intent_survives_whatever_the_backend_forces() {
    for p in PROTOCOLS {
        for streaming in [Streaming::StreamOnly, Streaming::NonStreamOnly, Streaming::Both] {
            for client_wants_stream in [false, true] {
                let (_, _, ctx_stream) = encode(p, streaming, client_wants_stream);
                assert_eq!(
                    ctx_stream, client_wants_stream,
                    "{p:?}/{streaming:?}: EncodeCtx::stream must stay the client's intent"
                );
            }
        }
    }
}
