//! Shared helpers for the `llm-xlate-anthropic` integration tests.
#![allow(dead_code)]

use llm_xlate_core::{
    caps::preset, Capabilities, DecodeCtx, EncodeCtx, HeaderMap, Protocol, ResponseId, Sealer,
    TranslatorConfig,
};

/// The dev envelope key shared by the decode and encode contexts (so `rtr1.` blobs round trip).
pub const KEY: &[u8] = b"llm-xlate-dev-key";

/// A fresh sealer on the dev key.
pub fn sealer() -> Sealer {
    Sealer::new(KEY)
}

/// A decode context on the default translator config.
pub fn dctx() -> DecodeCtx {
    DecodeCtx::from_config(TranslatorConfig::default())
}

/// An encode context whose client speaks `proto`, model `claude-opus-5`.
pub fn ectx(proto: Protocol) -> EncodeCtx {
    let mut ctx = EncodeCtx::new(proto, "claude-opus-5", ResponseId::new("msg_test_0001"), sealer());
    ctx.created_at = 1_700_000_000;
    ctx
}

/// An encode context for a specific client model.
pub fn ectx_model(proto: Protocol, model: &str) -> EncodeCtx {
    let mut ctx = EncodeCtx::new(proto, model, ResponseId::new("msg_test_0001"), sealer());
    ctx.created_at = 1_700_000_000;
    ctx
}

/// Empty request headers.
pub fn no_headers() -> HeaderMap {
    HeaderMap::new()
}

/// Headers carrying one `anthropic-beta` value.
pub fn beta_headers(v: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert("anthropic-beta", v.parse().unwrap());
    h
}

/// Pretty-print a JSON byte body for snapshotting (the on-wire bytes are compact; this is
/// only for a readable snapshot — byte-stability is asserted separately).
pub fn pretty(bytes: &[u8]) -> String {
    let v: serde_json::Value = serde_json::from_slice(bytes).expect("valid JSON body");
    serde_json::to_string_pretty(&v).unwrap()
}

/// Capability presets used across the tests.
pub fn claude_5() -> Capabilities {
    preset::claude_5()
}
pub fn claude_46() -> Capabilities {
    preset::claude_46()
}
pub fn claude_old() -> Capabilities {
    preset::claude_old()
}
/// Claude 3.7 Sonnet: budget-mode thinking (the earliest era with extended thinking). Claude 3.5
/// (`claude_old`) has no thinking mode, so budget-mode encoding is exercised against 3.7.
pub fn claude_37() -> Capabilities {
    llm_xlate_core::caps::shipped().resolve(
        &llm_xlate_core::ProviderFamily::Anthropic,
        "claude-3-7-sonnet-latest",
        None,
    )
}
pub fn gpt5_responses() -> Capabilities {
    preset::gpt5_responses()
}
