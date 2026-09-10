# llm-xlate-testrouter

A standalone, **tracing test router** built on [`llm-xlate`](../xlate). Point real-world tools and
agent harnesses at it (OpenAI SDKs, Anthropic SDKs, Claude Code, Codex CLI, Cursor/Continue,
LangChain, LiteLLM, curl) and it drives their traffic through the translation IR to surface deep
integration bugs — then hands you a full trace of every request and a no-spend way to turn any bug
into a regression test.

It is an **I/O shell only**: all translation logic stays sans-IO in `llm-xlate`; this crate wires it
to axum, reqwest, a trace writer, and a CLI.

## What it does

1. Serves all three API surfaces: `POST /v1/chat/completions`, `POST /v1/responses`,
   `POST /v1/messages` (plus the Responses store/cancel/count-tokens endpoints).
2. Routes each request to a real upstream by model, and **always** goes through the IR — even for
   same-surface traffic, because the client's expectations are then exact and the sharpest bug
   detector.
3. Writes one JSONL **trace record** per request: what the client sent, the IR it became, what was
   sent upstream, what came back, the IR events, and what the client actually received — plus every
   routing/capability/degradation/error/timing decision, and inline invariant checks.
4. Ships a **trace CLI**: pretty-print, triage across thousands of records, export into the
   `llm-xlate-e2e` capture format, and replay against recorded upstream bytes.

## Quick start

```sh
# Serve (uses the workspace key files for the real upstreams; keys are never logged).
xlate-testrouter serve --config crates/testrouter/router.example.toml

# Explain how a model routes (no network).
xlate-testrouter routes --config crates/testrouter/router.example.toml --model gpt-5.4
```

Point a tool at `http://127.0.0.1:8787` (see [`docs/HARNESSES.md`](docs/HARNESSES.md) for the exact
base-URL form, surface, and quirks per tool), tagging the session with `x-xlate-tag: <name>`.

## The pipeline

```
client request
   │  decode_request            (client dialect → IR)
   ▼
IrRequest ── route ──► (provider, upstream protocol, upstream model, capabilities)
   │  requirements + chain      (sidecar reasoning blobs; previous_response_id)
   │  lower                     (IR → lowered IR + degradations, gated by caps)
   │  encode_request            (lowered IR → upstream dialect)
   ▼
upstream ──► SEND ──► raw response (JSON or SSE)
   │  stream_decoder            (upstream bytes → IrEvents)
   │  aggregate                 (IrEvents → IrResponse)
   │  encode_response / stream_encoder   (IR → client dialect; keepalive; mismatch paths)
   ▼
client response  (+ x-xlate-trace-id, x-router-degraded, x-router-upstream-request-id)
   │
   └─► one trace record written to data_dir/traces/{date}.jsonl (+ index.jsonl)
```

Cancellation (client disconnect) aborts the upstream and marks the trace `cancelled`. A panic inside
`llm-xlate` is caught and rendered as a 500 in the client dialect — the server never dies.

## The trace record

One JSON object per line, with a **fixed field order** so the on-disk schema is stable:

```
v, trace_id, ts_start, ts_end, tag,
client{protocol,method,path,headers,body,body_raw_sha256,stream},
ir_request, route{provider,upstream_protocol,client_model,upstream_model,caps_source,caps,overrides},
chain, requirements, resolutions_summary,
lowered{ir,degradations},
upstream{url,headers,body,stream,status,resp_headers,request_id,raw,elapsed_ms,ttfb_ms},
ir_events[]{t_ms,event}, ir_response,
client_response{status,headers,frames[]{t_ms,keepalive,data} | body},
errors[]{stage,kind,status,message,rendered}, store, timing,
checks{aggregate_matches_encode_response,reencode_client_request_diff,degradation_count}
```

Secret headers (`authorization`, `x-api-key`, `openai-organization`, …) are redacted; media payloads
over the configured threshold become `{"$redacted":{sha256,len}}`. A compact `index.jsonl` carries
one small line per trace for fast triage. Client credentials are never forwarded or recorded.

## The CLI

```
xlate-testrouter trace show   <trace_id | path | path#line> [--section client|ir|route|upstream|events|frames|errors|checks|all] [--raw]
xlate-testrouter trace triage [--config …] [--file f] [--since <rfc3339>] [--tag t] [--json]
xlate-testrouter trace export <trace_id> --to <run dir>
xlate-testrouter trace replay <trace_id> [--config …] [--diff]
```

- **show** — locate a record by id (via `traces/index.jsonl`) or a direct path, and pretty-print the
  chosen section(s); SSE frames render one per line with their `t_ms`; `--raw` dumps the record.
- **triage** — one row per trace (ts, tag, client→route, upstream/client status, error, degradations,
  failed checks, ttfb/total ms) with **anomaly flags**: any error, any translation-layer
  (`xlate_error`) failure, a failed inline check, `caps_source = unknown`, keepalive-only streams,
  ttfb > 10 s, cancellations, and missing upstream request ids. Summary counts at the end; `--json`
  for machines; **exit code 1 when anomalies exist**, so a session can be scripted.
- **export** — writes **two** `llm-xlate-e2e` captures into a run directory: the *client leg* (client
  request → client response/frames) and the *upstream leg* (upstream request → raw upstream
  body/SSE), each with an `observe.json`, so `llm-xlate-e2e check` and `promote` work unchanged.
  Redaction is preserved.
- **replay** — the **no-spend regression tool**. Rebuilds the router with the recorded route
  overrides, a mock upstream that serves the recorded raw bytes, and ids seeded from the recorded
  ids; re-sends the recorded client request; and diffs the produced upstream body and client
  frames/body against the recorded ones (exit 1 on any difference). Once a session is captured, every
  later `llm-xlate` change is re-verified against the exact recorded upstream bytes with **zero API
  cost**.

## The trace viewer (browser)

[`tools/trace_viewer.html`](tools/trace_viewer.html) is a single self-contained page — no build step,
no network, no CDN — for reading a whole session at once. Open it straight from disk and drop the
trace files on it:

```
# macOS / Linux
open   crates/testrouter/tools/trace_viewer.html
# Windows
start  crates\testrouter\tools\trace_viewer.html
```

Then pick (or drag and drop) one or more `data/traces/<date>.jsonl` files — `index.jsonl` also loads,
as index-only rows. Files are read locally in the page; nothing is uploaded. Blank and corrupt lines
are counted and reported instead of aborting the load, and a record missing any optional field
renders it as `n/a` rather than failing.

**Log view** — one collapsed row per record: start time (local; hover for the RFC-3339 `ts_start`),
short trace id (**click to copy the full id**), tag, `client protocol → provider/upstream protocol`,
`client model → upstream model`, path, upstream and client status, stream + frame count, ttfb and
total, degradation count, re-encode diff count, and the first error `stage/kind`. The summary bar
totals records, errors, anomalies and per-protocol counts; filters (free text over id/tag/model/
path/provider/error, protocol, provider, status class, tag, anomalies-only, streaming-only) and the
sort order are kept in the URL hash, so a filtered view is a shareable link. `j`/`k` move, `enter`
expands, `x` copies the selected trace id, `/` focuses the search box.

**Row colours** — the left border and row tint:

| colour | meaning |
|---|---|
| red | an error was recorded (`errors[]` with a stage other than `cancelled`) |
| amber | a failed inline check (`checks.aggregate_matches_encode_response == false`) |
| purple | the request was cancelled (an error at stage `cancelled`) |
| blue border | lowering recorded degradations (the row is otherwise clean) |
| none | clean |

The anomaly set behind "anomalies only" is the same one `trace triage` uses: errors, `xlate_error`
stages, failed checks, `caps_source = unknown`, keepalive-only streams, ttfb > 10 s, cancellations
and missing upstream request ids. `reencode_client_request_diff` is shown as a neutral `reencode N`
badge (hover for the field list) and deliberately **not** counted as an anomaly — most real traffic
has one.

**Expanded row** — a metadata strip (ids, route + caps source, models, timings, checks, chain, store,
anomalies, degradations, errors) over two panels: **Request** (client body → `ir_request` → lowered
IR + degradations → upstream body, plus route/caps, requirements and resolutions) and **Response**
(upstream status/headers/raw → `ir_events` as a `t_ms`/type/index/summary timeline → `ir_response` →
the client response). Every JSON blob is a collapsible tree, collapsed one level down by default,
with per-panel expand-all/collapse-all, click-to-expand for truncated long strings, a `copy` button
that yields the exact JSON, and `{"$redacted":{…}}` media markers rendered as a compact chip. Click
any event row for its full IR event; streamed records offer a **raw** toggle that shows the exact SSE
bytes (upstream and client frames).

## From a bug to a regression test

```
trace triage --tag <session>          # find the anomalous trace
trace show   <trace_id>               # read what happened
trace export <trace_id> --to runs/bug # both legs, in e2e capture format
llm-xlate-e2e check runs/bug          # confirm the bug reproduces under llm-xlate
# … fix llm-xlate …
trace replay <trace_id>               # NO DIFF ✓  — the fix holds against the recorded bytes
llm-xlate-e2e promote runs/bug        # add the case to the shipped dataset
```

## Configuration

One `router.toml` (see [`router.example.toml`](router.example.toml)): `[server]` (listen, optional
shared `token`, body limit, keepalive, data dir), `[trace]` (enable, media-redaction threshold, raw
capture), `[providers.*]` (base URL + key file/env), ordered `[[route]]` rules (first regex match on
the resolved model wins), `[aliases]` (client name → upstream model; `default` when the client omits
`model`), and optional `[backend_overrides.*]`. Per-request `x-xlate-*` headers override model,
upstream protocol, provider, caps preset, reasoning exposure, and session tag.

## Tests

Everything is **offline and hermetic** — every request goes through an in-process mock upstream, no
sockets, no secrets. `cargo test -p llm-xlate-testrouter` covers the pipeline (every surface ×
stream/non-stream × text/tools/reasoning/media/structured, error passthrough, keepalive, cancel,
oversize, token auth, overrides, redaction), the **trace laws** (`tests/trace_laws.rs`: schema round
trip, the aggregate law over the client frames, no minted ids leaking upstream, degradation
bookkeeping, replay determinism, and triage anomaly counts), and the **CLI** (`tests/cli_*.rs`,
driven through the library entry points; `export` output is validated with `llm-xlate-e2e`).

A `--features live` smoke path exists for the operator; it never runs under `cargo test`.
