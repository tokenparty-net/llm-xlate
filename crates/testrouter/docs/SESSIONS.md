# Harness session log

One row per operator/agent session run against the test router (plan §9, §10 R4). Fill this in
after each session: run `xlate-testrouter trace triage --tag <tag>`, record the anomalies, and link
the trace ids you exported into the dataset.

## How to record a session

1. Start the router: `xlate-testrouter serve --config crates/testrouter/router.example.toml`.
2. Point the tool at it (see `HARNESSES.md`), sending `x-xlate-tag: <tool>-<date>`.
3. Run the 5-step exercise script.
4. `xlate-testrouter trace triage --tag <tool>-<date>` — note the anomaly count and kinds.
5. For each anomaly: `trace show <trace_id>`, then `trace export <trace_id> --to runs/<name>` and
   drop the trace id into the table.
6. Fix the `llm-xlate` bug; `trace replay <trace_id>` must go from a diff to NO DIFF.

## Template

| Date       | Tool / version          | Tag                 | Surface   | Anomalies (kind × n)        | Trace ids (interesting)      | Fix / status |
|------------|-------------------------|---------------------|-----------|-----------------------------|------------------------------|--------------|
| YYYY-MM-DD | OpenAI Python x.y.z      | oai-py-YYYYMMDD     | chat      | none                        | —                            | clean        |
| YYYY-MM-DD | OpenAI Node x.y.z        | oai-node-YYYYMMDD   | responses | none                        | —                            | clean        |
| YYYY-MM-DD | Anthropic Python x.y.z   | ant-py-YYYYMMDD     | anthropic | none                        | —                            | clean        |
| YYYY-MM-DD | Anthropic Node x.y.z     | ant-node-YYYYMMDD   | anthropic | none                        | —                            | clean        |
| YYYY-MM-DD | Claude Code x.y.z        | claude-code-YYYYMMDD| anthropic | e.g. caps_unknown × 1       | `tr_…`                       | in progress  |
| YYYY-MM-DD | Codex CLI x.y.z          | codex-YYYYMMDD      | responses | e.g. xlate_error × 1        | `tr_…`                       | fixed (PR #…)|
| YYYY-MM-DD | Cursor / Continue x.y.z  | cursor-YYYYMMDD     | chat      | none                        | —                            | clean        |
| YYYY-MM-DD | LangChain x.y.z          | langchain-YYYYMMDD  | chat      | none                        | —                            | clean        |
| YYYY-MM-DD | LiteLLM x.y.z            | litellm-YYYYMMDD    | mixed     | none                        | —                            | clean        |

Anomaly kinds (from `trace triage`): `error:<kind>`, `xlate_error`, `failed_check`, `caps_unknown`,
`keepalive_only`, `slow_ttfb`, `cancelled`, `missing_request_id`. An `xlate_error` (a decode/lower/
encode/decode_response stage failure) is a translation bug; a bare `error:*` is usually a provider
passthrough and expected.

## 2026-09-10 — Claude Code → gpt-5.6-luna (operator session)
- Symptom: `API Error: 400 Invalid 'user': string too long. Expected a string with maximum length 64, but got a string with length 150`.
- Cause: Claude Code sends a ~150-char `metadata.user_id`; the translation forwarded it verbatim as OpenAI `user` (64-char limit).
- Fix: `llm_xlate_core::canon::identifier::fit` — over-long `user`/`safety_identifier` (OpenAI, 64) and `metadata.user_id` (Anthropic, 256) become a deterministic sha256 digest with a `user=rewritten` degradation. Regression tests in chat/responses/anthropic encode suites; router test `i11b_long_client_user_id_is_rewritten_and_traced`.
- Also fixed: encoder-side degradations were missing from the trace and the `x-router-degraded` header (`TraceBuilder::add_degradations`).
- Verified live: the same request shape now returns 200 through gpt-4o-mini.

## 2026-09-10 — vLLM backend added (`compat` provider, http://127.0.0.1:3301) + triage of the Claude Code session
- Added `providers.compat` (base_url, `.token_tf_api`, `models = [deepseek-v4-flash, kimi-k3]`), a pass-through route, and
  `backend_overrides."compat"` (tools, JSON output, `reasoning_content` replay) in `router.example.toml`. `/v1/models`
  now advertises static per-provider `models` lists (new `ProviderConfig.models`).
- Bug (llm-xlate, `crates/chat/src/stream_dec.rs`): vLLM attaches `usage` to EVERY chunk; the decoder treated the first
  one as the terminal usage chunk and dropped all content. Now per-chunk usage is only terminal on a choices-less chunk
  or after `finish_reason`; otherwise the last value applies at `[DONE]`. Tests `decode_vllm_per_chunk_usage_keeps_streaming`,
  `decode_per_chunk_usage_without_final_usage_chunk_uses_last_seen`.
- False positive (testrouter): 25 `failed_check` rows on the Claude Code session (thinking + tool_use streams) came from
  comparing tool arguments as raw strings (streamed `partial_json` keeps provider whitespace, the non-streaming side
  re-serializes compactly). `trace.rs::project` and `trace_laws.rs` now compare arguments as JSON values. Verified live:
  a thinking+tool stream through claude-haiku-4-5 triages clean; exported trace passes `llm-xlate-e2e check` on both legs.
- Note: vLLM sends no `x-request-id`, so `missing_request_id` is expected for `compat` routes.

## 2026-09-10 — Codex → `deepseek-v4-flash` (compat/vLLM): Responses hosted tools on a Chat wire
- Symptom: every Codex turn failed with `400 … 10 validation errors: {'type': 'literal_error',
  'loc': ('body','tools',7,'type'), 'msg': "Input should be 'function'", 'input': 'namespace'}`. Trace
  `tr_dc34ef142352404c9a2d6e2486d601cb` (client `responses` → `compat`/`chat`).
- Cause (llm-xlate, `crates/chat/src/encode.rs`): Codex declares 15 tools — 10 plain functions, 4
  `{"type":"namespace"}` groups and 1 `{"type":"web_search"}`. The hosted ones decode to
  `ToolDef::Provider` with family `OpenAI`; the Chat encoder pushed a provider tool through **raw**
  whenever `item.family == ProviderFamily::OpenAI`. Family equality is the wrong test: OpenAI Chat and
  OpenAI Responses share a family but only Responses can carry a hosted tool — Chat Completions requires
  every `tools[]` entry to be `{"type":"function","function":{…}}`. `lower/provider_tools.rs` also let
  them through for the same reason (`target.family()` is `OpenAI` for both dialects), so nothing upstream
  of the codec caught it.
- Fix: a `ToolDef::Provider` has no Chat carrier — drop it with a `tools.provider` degradation naming the
  tool (`"provider-hosted tool (namespace multi_agent_v1) has no Chat carrier; dropped"`), mirroring the
  item-side `provider_tool` rule. When every tool was hosted, `tools` is omitted rather than sent as `[]`.
  Regression tests: `chat::request::encode_drops_responses_hosted_tools_with_degradation`,
  `encode_all_hosted_tools_omits_tools_key` (both built from the captured Codex tool JSON) and router
  test `i11c_responses_hosted_tools_are_dropped_for_a_chat_upstream`.
- Verified live (free vLLM backend, 1 request, trace `tr_261deb1dc98440c7b91eb415280d52d5`): the upstream
  body now carries 10 `function` tools and zero hosted entries, with 5 degradations naming each dropped
  tool. The `body.tools.7.type` validation error is gone.
- Open, for the operator:
  1. `.token_tf_api` is **rejected by the backend** (`401 Incorrect API key provided`) — reproduced by
     curling `127.0.0.1:3301/v1/chat/completions` directly, so it is a rotated credential, not a router
     bug. The live check above could not get past auth.
  2. Dropping the four `namespace` groups costs Codex its sub-agent / MCP tools. The backend's
     `/v1/models` advertises `"endpoints":["/v1/chat/completions","/v1/responses"]`, so pointing the
     `compat` route at `upstream = "responses"` would let the hosted tools through natively instead of
     being dropped. Untested (blocked on the token); `backend_overrides."compat"` was tuned for chat and
     would need review.
  3. Flattening a `namespace` into its member function tools is the other option, but the call-name
     convention Codex expects back (`close_agent` vs `multi_agent_v1.close_agent`) is unverified, so it
     was not guessed at here.

## 2026-09-10 — Claude Code → gpt-5.6-luna: reasoning continuity lost on every turn (trace `tr_29a3110ec1ae4c33a4d2feaa3ed2a958`)
- Symptom: five `reasoning=dropped ("foreign-family reasoning blob dropped (not replayable)")` degradations per request on an
  all-OpenAI conversation, so GPT's encrypted reasoning was never replayed on tool loops.
- Cause (llm-xlate, `crates/anthropic/src/decode.rs`): the client-facing Anthropic encoder correctly carries OpenAI
  `encrypted_content` to an Anthropic client as `redacted_thinking` with a router envelope in `data` (decoded envelope:
  provider openai, kind encrypted, model gpt-5.6-luna). On replay, `decode_redacted_thinking` never opened the envelope
  and tagged every blob as a native Anthropic redacted blob; `lower()` then dropped it as foreign for the OpenAI target.
  The `thinking` path already used `Sealer::open_or_native`; `redacted_thinking` (and `compaction`) did not.
- Fix: both paths now open router envelopes (`open_or_native`), keeping the true family/kind/model. Regression test
  `redacted_thinking_router_envelope_is_opened_to_its_true_family` (crates/anthropic/tests/request_decode.rs).
- Verified offline: the recorded client request (exported with `trace export`, run through `llm-xlate-e2e translate
  --dry-run` for anthropic->responses) now yields 5 `reasoning` items with `encrypted_content` in the upstream body and
  no "foreign-family reasoning blob dropped" degradation (`trace replay` refuses this trace because its media was redacted).
