# llm-xlate — engineering conventions (read with ../llm_xlate_plan.md)

This file records the decisions that adapt it to a Cargo workspace so several people can build in parallel.

## Workspace layout (deviation from the plan's single crate, for parallel development)

```
llm-xlate/                       workspace root; data/caps/*.toml lives here
  crates/core       -> llm-xlate-core       ir/, caps/, error.rs, canon.rs, envelope.rs, degrade.rs, wrap.rs,
                                            codec traits (StreamDecoder/StreamEncoder/Codec), EncodeCtx, Aggregator, sse.rs
  crates/chat       -> llm-xlate-chat       OpenAI Chat Completions codec (plan proto/chat)
  crates/responses  -> llm-xlate-responses  OpenAI Responses codec      (plan proto/responses)
  crates/anthropic  -> llm-xlate-anthropic  Anthropic Messages codec    (plan proto/anthropic)
  crates/xlate      -> llm-xlate            Translator facade, lower/, store.rs, pair-matrix goldens, tests/laws.rs
```
Dependency direction: core <- {chat, responses, anthropic} <- xlate. Codec crates never depend on each other.
The router imports only `llm-xlate` (which re-exports core).

## Ownership while building in parallel
- Only edit files inside the crate you were assigned. If you need something changed in `core`, do NOT edit it;
  report the exact change you need in your final output and work around it locally (private helper) meanwhile.
- Use a private target dir to avoid build-lock contention: `CARGO_TARGET_DIR=target/<crate-name>` (Bash: `export CARGO_TARGET_DIR=...`).
- Build/test only your package: `cargo test -p llm-xlate-<name>`.

## Code rules (from the plan §3, §5, §11)
- Sans-IO: no async, no tokio, no I/O, no clocks, no RNG, no global counters. Everything is a pure function or a push state machine.
- Determinism: identical input + identical capabilities => byte-identical output.
- JSON: `serde_json` with `preserve_order` + `raw_value`. Never sort keys of user data. Tool arguments are a `String` (JsonText).
  Parse only when the target needs an object (Anthropic `input`), using `serde_json::Value` (preserve_order keeps ordering).
- Every lossy step records a `Degradation`; never a silent drop. Unknown capability => conservative + `Unsupported`.
- Fixed strings that enter prompts live in core `wrap.rs` (`wrap_v1`) and are referenced, never re-typed.
- Errors: `XlateError` from core; per-protocol wire mapping lives in each codec (`errors.rs`).
- Tests: unit tests in-crate; golden fixtures as JSON under `<crate>/tests/fixtures/`; `insta` snapshots allowed
  (commit `.snap` files; run `INSTA_UPDATE=always cargo test` / `cargo insta accept` is NOT available — use
  `INSTA_UPDATE=always` env var on first run, then re-run without it to verify they are stable).
- Use `pretty_assertions::assert_eq` for readable diffs.
- Public items documented with `///`. No `unwrap()` on untrusted input paths; return `XlateError::upstream_malformed(..)` etc.
