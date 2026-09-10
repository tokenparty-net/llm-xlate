# Pointing real tools at the test router

The router listens on `http://127.0.0.1:8787` (override with `serve --listen`). It serves all three
surfaces:

| Surface     | Path                     | Point tools here when they speak… |
|-------------|--------------------------|-----------------------------------|
| Chat        | `POST /v1/chat/completions` | OpenAI Chat Completions        |
| Responses   | `POST /v1/responses`     | OpenAI Responses                  |
| Anthropic   | `POST /v1/messages`      | Anthropic Messages                |

Every request is routed to a real upstream **through the IR** (decode → lower → encode → send →
decode → aggregate → encode), even when the client and upstream surfaces are identical, and a full
trace is written to `data_dir/traces/{date}.jsonl`. The point is to find deep integration bugs in
`llm-xlate` with real-world traffic.

## Rules that apply to every tool

- **Base URL forms differ.** Some SDKs want the `/v1` included (`http://127.0.0.1:8787/v1`), others
  append it themselves and want the bare origin (`http://127.0.0.1:8787`). Both are noted per tool
  below. If you get 404s, you have the `/v1` doubled or missing.
- **Credentials are never forwarded.** The router authenticates to the upstream from its own key
  files (`.xlate_e2e_claude`, `.xlate_e2e_oai`); a client `Authorization` / `x-api-key` header is
  never sent upstream and never recorded. Set the SDK's key to any non-empty dummy value
  (`sk-local`), unless you configured `[server].token`, in which case the client must send exactly
  that token as its bearer/api-key and it is checked but still not forwarded.
- **Tag your session.** Send `x-xlate-tag: <name>` on every request (most SDKs support a default
  header) so `trace triage --tag <name>` scopes to your run. `serve --tag` is informational only;
  the per-request header is authoritative.
- **Per-request overrides** (all optional, all recorded): `x-xlate-model` (upstream model),
  `x-xlate-upstream` (`chat|responses|anthropic`), `x-xlate-provider`, `x-xlate-caps` (a preset
  name), `x-xlate-expose` (`none|summary|full`).
- **After the session:** `xlate-testrouter trace triage --tag <name>` (exit code 1 ⇒ anomalies).
  Then `trace show <trace_id>` on anything flagged, or read the whole session in the
  [trace viewer](#the-trace-viewer).

## The 5-step exercise script

Run this against each tool, tagging the session, then triage:

1. **Plain chat** — one user turn, non-streaming.
2. **Multi-turn** — a follow-up that depends on the first answer.
3. **Tool use** — define one function/tool, let the model call it, return a tool result, get the
   final answer (a full tool loop).
4. **Image / file** — send an image (or a PDF, for Anthropic/Responses) in the prompt.
5. **Streaming + cancel** — a streaming request you interrupt part-way (Ctrl-C the client) to
   exercise the disconnect/cancellation path.

---

## OpenAI Python SDK (`openai`)

- Surface: **Chat** (`client.chat.completions`) or **Responses** (`client.responses`).
- Point it: `OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="sk-local")` — the Python SDK
  appends the surface path to the base, so include `/v1`. Or set `OPENAI_BASE_URL=http://127.0.0.1:8787/v1`.
- Tag: `OpenAI(default_headers={"x-xlate-tag": "oai-py-YYYYMMDD"})`.
- Quirks: the SDK **retries** 429/5xx with backoff (set `max_retries=0` while probing so one logical
  request is one trace); `stream_options={"include_usage": true}` adds a usage-only final chunk;
  Responses defaults to `store=true` (the router persists it — chainable via `previous_response_id`).

## OpenAI Node SDK (`openai`)

- `new OpenAI({ baseURL: "http://127.0.0.1:8787/v1", apiKey: "sk-local", defaultHeaders: { "x-xlate-tag": "oai-node-YYYYMMDD" }, maxRetries: 0 })`.
- Same surfaces/quirks as the Python SDK.

## Anthropic Python SDK (`anthropic`)

- Surface: **Anthropic** (`client.messages`).
- Point it: `Anthropic(base_url="http://127.0.0.1:8787", api_key="sk-local")` — the Anthropic SDK
  appends `/v1/messages`, so use the **bare origin** (no `/v1`). Or `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`.
- Tag: `default_headers={"x-xlate-tag": "ant-py-YYYYMMDD"}`.
- Quirks: sends `anthropic-version` and often several `anthropic-beta` values — the router passes
  through what the registry allows and records the rest; `message_start` carries prefill usage;
  `ping` frames and `signature_delta` appear in thinking streams; `count_tokens` is a passthrough
  (recorded, not translated).

## Anthropic Node SDK (`@anthropic-ai/sdk`)

- `new Anthropic({ baseURL: "http://127.0.0.1:8787", apiKey: "sk-local", defaultHeaders: { "x-xlate-tag": "ant-node-YYYYMMDD" } })`.

## Claude Code

- Point it at the router as the Anthropic endpoint: `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`
  (bare origin), any dummy `ANTHROPIC_API_KEY`.
- Surface: **Anthropic**. Tag via a proxy/wrapper that injects `x-xlate-tag`, or filter by time
  window in triage (`--since`).
- Quirks: Claude Code sends **multiple `anthropic-beta`** headers and tool definitions with large
  system prompts; it runs long multi-turn tool loops with thinking replay (the sidecar re-supplies
  reasoning blobs on later turns). Watch traces for `caps_unknown` (an unrecognised model id) and
  for degradations on betas the registry does not model.

## Codex CLI

- Point it at the router as the OpenAI endpoint: `OPENAI_BASE_URL=http://127.0.0.1:8787/v1`.
- Surface: **Responses**. Quirks: Codex uses Responses with **`store: false`** and **encrypted
  reasoning** envelopes — the router replays those envelopes through the IR without persistence;
  verify the encrypted-reasoning round-trips on multi-step tool runs. It also streams; exercise a
  cancel.

## Cursor / Continue

- Both accept an OpenAI-compatible base URL and model list in settings. Point the base URL at
  `http://127.0.0.1:8787/v1`; set a dummy key. Surface: **Chat** (some flows use **Responses**).
- Quirks: they call `GET /v1/models` on connect — the router answers an offline listing built from
  the route rules + aliases; pick a model id it lists (e.g. `claude`, `gpt`, or a `claude-…`/`gpt-…`
  name matched by a route).

## LangChain

- `ChatOpenAI(base_url="http://127.0.0.1:8787/v1", api_key="sk-local", default_headers={"x-xlate-tag": "..."})`
  (Chat), or `ChatAnthropic(base_url="http://127.0.0.1:8787", ...)` (Anthropic).
- Quirks: LangChain agents fan out many short tool-call requests; triage after a run and look for
  `xlate_error` (a translation bug) vs `error:*` (a provider passthrough).

## LiteLLM

- As a proxy or SDK, set the provider base URL to the router (`/v1` for OpenAI-style, bare origin
  for Anthropic). Choose the model so a route matches (`claude-…` → Anthropic, `gpt-…` → Responses).
- Quirks: LiteLLM normalises across providers itself; disable its own retries while probing.

## curl

Chat (needs `x-xlate-upstream: chat` unless the model routes to Chat):

```sh
curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-xlate-tag: curl-smoke' \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
```

Anthropic streaming:

```sh
curl -N http://127.0.0.1:8787/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-xlate-tag: curl-smoke' \
  -d '{"model":"claude-sonnet-5","max_tokens":256,"stream":true,
       "messages":[{"role":"user","content":"count to 5"}]}'
```

Every response carries `x-xlate-trace-id`; pass it to `trace show`.

---

## The trace viewer

For reading a whole harness session at once, open
[`crates/testrouter/tools/trace_viewer.html`](../tools/trace_viewer.html) directly from disk
(`start …\trace_viewer.html` on Windows, `open …/trace_viewer.html` elsewhere) and drop that day's
`data/traces/<date>.jsonl` on the page. It is one self-contained file: no server, no network, no
build step, and the trace never leaves the machine. It handles tens of MB (files are parsed in
chunks; rows render lazily, and a record's panels are only built when you expand it).

Use it the way you'd use `trace triage` plus `trace show`, but with both sides of the translation
side by side: **Request** (client body → IR → lowered IR + degradations → upstream body) and
**Response** (upstream status/headers/raw SSE → IR events timeline → IR response → the exact client
frames). Tick **anomalies only** to get the triage set, or filter by tag to scope to your session —
filters live in the URL hash, so a filtered view can be pasted into a bug report alongside the trace
id (click any short id to copy the full one).

Row colours, at a glance:

- **red** — an error was recorded for that request.
- **amber** — an inline check failed (the client frames did not re-aggregate to `ir_response`) —
  the highest-value rows in a harness session.
- **purple** — the request was cancelled (client disconnect).
- **blue left border** — lowering degraded something (a field dropped or rewritten to fit the
  provider); the row is otherwise clean.
- **grey `reencode N` badge** — the client request did not survive a decode/re-encode round trip in
  its own dialect. Common and *not* an anomaly, but worth a look when it names a field you care
  about (hover the badge for the list).

Once a row looks wrong, take its trace id back to the CLI: `trace show <id>`, `trace export <id>`,
`trace replay <id>`.

## The findings loop

`triage --tag <name>` → `trace show <id>` on each anomaly → fix `llm-xlate` with a regression test
built from `trace export <id>` → `trace replay <id>` proves the fix with no spend → `llm-xlate-e2e
promote` adds the case to the dataset. See the crate `README.md` for the full workflow.
