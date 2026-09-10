# llm-xlate

**Cross-API translation core for an LLM router.** `llm-xlate` decodes a client request in one
provider dialect — OpenAI **Chat Completions**, OpenAI **Responses**, or **Anthropic
Messages** — and re-encodes it for a *different* backend dialect, faithfully bridging the
request, the streaming and non-streaming response, the error body, and the stored-response
(`previous_response_id`) chain. It is **sans-IO**: no async, no clocks, no RNG, no global
counters — every entry point is a pure function or a push state machine, so identical
`(bytes, capabilities, config)` always produce byte-identical output.

The router holds one [`Translator`] and drives the whole path through it.

## What it does

* **Decode** a client request body → a shared intermediate representation (`IrRequest`).
* **Lower** that IR for a target protocol under a **capability profile**: reasoning-replay
  policy, foreign provider-tool folding, foreign file-id bridging, schema-keyword
  normalization, `strict` downgrade, effort snapping, `n>1` rejection, stop-sequence caps,
  state clearing, ZDR — every lossy step recorded as a `Degradation`, every unrepresentable
  request rejected with a typed `XlateError`.
* **Encode** the lowered IR for the backend dialect (request body + headers).
* **Stream** the response both ways (provider SSE → `IrEvent` → client SSE) and **aggregate**
  a stream into one `IrResponse` (and vice-versa via `synthesize_stream`).
* **Bridge errors** between dialects (plan §10) and **persist / replay** stored responses
  (plan §9).

## Workspace split

The plan is one logical crate; for parallel development it is a Cargo workspace. Dependency
direction is `core <- {chat, responses, anthropic} <- xlate`; codec crates never depend on
each other, and the router imports **only** `llm-xlate` (which re-exports `core` and the codec
structs).

| Crate | Package | Owns |
|---|---|---|
| `crates/core` | `llm-xlate-core` | IR, capability registry, codec traits, envelope/sealer, aggregator, SSE, `wrap`, `canon`, degradations, errors |
| `crates/chat` | `llm-xlate-chat` | OpenAI Chat Completions codec |
| `crates/responses` | `llm-xlate-responses` | OpenAI Responses codec (+ stored-response views) |
| `crates/anthropic` | `llm-xlate-anthropic` | Anthropic Messages codec |
| `crates/xlate` | `llm-xlate` | the `Translator` façade, `lower/`, `requirements`, `store`, pair-matrix goldens, invariant laws |

## The router loop (plan §4)

```rust
use llm_xlate::{Translator, TranslatorConfig, Protocol, Resolutions, ResponseId};

let xl = Translator::new(TranslatorConfig::default());

// 1. decode the client request
let ir = xl.decode_request(client_p, &body, &hdrs)?;
// 2. tell the router what to resolve first (foreign files, missing reasoning, a chain)
let reqs = xl.requirements(&ir, &caps, backend_p);
// 3. router resolves I/O asynchronously → Resolutions (+ materialize_chain for a chain)
let ir = match reqs.chain { Some(_) => xl.materialize_chain(&chain, ir)?, None => ir };
let res: Resolutions = resolve(reqs).await;
// 4. build the response context from the *pre-lowering* client request …
let ctx = xl.encode_ctx(client_p, &ir, ResponseId::new(minted_id), created_at);
// 5. … lower and encode for the backend
let low = xl.lower(ir, &caps, backend_p, &res)?;
let enc = xl.encode_request(backend_p, &low.req, &caps, &ctx)?;
// 6. stream: provider SSE → IrEvent → client SSE, aggregating in parallel
let mut dec = xl.stream_decoder(backend_p, &caps);
let mut out = xl.stream_encoder(client_p, enc.ctx);
```

The steps are separate because requirements resolve **asynchronously** between `requirements`
and `lower`. For the common no-resolution case there is a `translate_request` convenience that
runs `decode → requirements → lower → encode` in one call (lowering + wiring degradations are
merged into `EncodedRequest::degradations`).

## The envelope

Reasoning that a client replays must survive being routed to *any* backend. On the way **in**,
a client-facing decoder opens `rtr1.` envelopes (`Sealer::open_or_native`); on the way **out**,
a client-facing encoder **seals** blobs whose family is foreign to the client into an `rtr1.`
envelope (HMAC-SHA256, deterministic, fixed key order). Provider-facing encoders/decoders never
seal: same-family reasoning replays natively, foreign reasoning is dropped (never replayed as
text). The key lives in `TranslatorConfig::envelope_key`.

## The capability registry

A backend's abilities are a `Capabilities` profile resolved from `data/caps/*.toml` by
`(family, model)`. Presets in `caps::preset` cover the shipped tiers used across the tests:
`claude_old`, `claude_46`, `claude_5`, `gpt4o`, `gpt5_chat`, `gpt5_responses`, `gpt6`,
`openai_compatible`. `openai_compatible` is deliberately conservative — most fields `Unknown`
⇒ treated as unsupported ⇒ inline-wrap fallback — until a concrete backend opts in via
`BackendOverrides`.

## Responsibility split

* **`lower/`** owns the *protocol-agnostic, capability-gated policy* that must run **before** a
  codec's `encode_request`: what to drop, downgrade, fold, or reject, and why. It never shapes
  wire bytes.
* **the codec crates** own *wire shape*: the exact JSON/SSE of each dialect, and the
  client/provider envelope boundaries.
* **the façade** owns *wiring*: the codec registry (`codec_for`), the `EncodeCtx` construction
  policy (`encode_ctx`), the stored-response mapping, and the two streaming mismatch helpers
  (`synthesize_stream` for a non-streaming backend → streaming client, `aggregate_stream` for a
  streaming backend → non-streaming client).

## Degradation reporting

Every lossy step is a `Degradation { kind, field, detail }` — never a silent drop. Lowering
degradations and codec wiring degradations are surfaced together on
`EncodedRequest::degradations`; `Degradations::render_header_value()` renders the deterministic
`x-router-degraded` header value (`field=kind;field=kind`, insertion order). An unrepresentable
request is a typed `XlateError` (`Unsupported` / `IncompatibleHistory` / `InvalidRequest`),
never a lie on the wire.

## Tests

```sh
export CARGO_TARGET_DIR=target/xlate     # avoid build-lock contention

cargo test  --workspace                  # all five crates
cargo test  -p llm-xlate --test golden_matrix   # pair-matrix goldens (insta snapshots)
cargo test  -p llm-xlate --test laws            # §11 invariant laws
cargo clippy --workspace --all-targets          # lint clean
cargo doc   --workspace --no-deps               # warning-free rustdoc
```

* **`tests/golden_matrix.rs`** — every request fixture × every applicable directed protocol
  pair × every capability preset whose transport allows the target: `decode → lower → encode`,
  snapshotting the encoded body, sorted headers, and `x-router-degraded` (or the typed error);
  plus every response/stream fixture × every client protocol, snapshotting the client
  stream-encoded frames **and** the aggregated `encode_response` side by side. The committed
  `.snap` files under `tests/snapshots/` are the reviewed evidence. Regenerate with
  `INSTA_UPDATE=always` on the first run, then re-run without it to confirm stability.
* **`tests/laws.rs`** — the plan §11 invariants: cross-process determinism (SHA-256 of a fixed
  corpus, hashed in two child processes), canonical bytes (key order + float text preserved),
  prefix stability (Anthropic + Responses), no hoisting of mid-context instructions,
  `wrap_v1` wrapper versioning, no minted ids in prompts, degradation coverage against
  `openai_compatible`, stream/response round-trip consistency, total lowering (proptest),
  streaming split-boundary fuzz (proptest), and a materialized chain equalling its stateless
  twin.
