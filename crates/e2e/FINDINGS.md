# FINDINGS — llm-xlate live-API exploration

Curated answers to `llm_xlate_e2e_plan.md §5`, filled from the E4 first live run (2026-09-10). Runs:
explore1 (all-explore cheap, 154 sent), explore2 (anthropic_all sweep of the top questions, 126 sent),
explore3 (legal-slot + budget confirm, 12 sent), translate1 (translate-and-send), placement_audit (1 sent —
the decisive in-array-system conjunction case, §2). Spend for the whole exploration: Anthropic ~$0.27,
OpenAI ~$0.03 (see runs/SPEND.md). This account's Anthropic listing has
no 3.x models, so 3.x/3.7 rows keep their prior verified_at.

Legend: CONFIRMED / REFUTED / PARTIAL / UNTESTED.

## Anthropic Messages

### 1. cache_control — CONFIRMED
Request-root cache_control:{type:ephemeral} accepted (200) on ALL 9 models (haiku-4-5 .. opus-5).
Block-level, system+message breakpoints, ttl:1h all accepted. >4 breakpoints -> 400 (max=4 confirmed).
Old model accepts. Settles the reviewers' doubt: llm-xlate emitting a request-root ephemeral breakpoint
is valid. (cache_creation/read counters ~0: padded prompt under the ~1024-token min; acceptance was the
question.) Evidence: explore1+explore2 ant.cache.* ; re-settled in translate1 (chat->anthropic bodies
carry request-root cache_control, accepted). Registry: [defaults.cache] auto_request_level=true, max=4,
ttl=[5m,1h] CONFIRMED, verified_at 2026-09-10. No change.

The zero counters noted above were a probe artefact, not an API fact: the `cacheacct` family
(2026-09-12, ~3.8k-token prefix) moves them off zero. See §C1.

### 2. in-array role:system — REFUTED (placement) / CONFIRMED (feature)
Content-bearing in-array system is accepted (200) on current models iff it satisfies BOTH constraints of a
CONJUNCTIVE rule; the API enforces them with TWO distinct verbatim messages, and which one you see depends
on which half fails:
 - (a) what PRECEDES it (verbatim): "role 'system' must follow a 'user' message or an 'assistant' message
   ending in a server tool result; the directive-only form (content: [] with output_config) is accepted at
   any position" — the 400 when the system follows a plain assistant.
 - (b) what FOLLOWS it (verbatim): "role 'system' must precede an 'assistant' message or end the array; the
   directive-only form (content: [] with output_config) is accepted at any position" — the 400 when the
   system is followed by a user and is not the last element.
So: accepted iff (follows a user OR a server-tool-result assistant) AND (precedes an assistant OR ends the
array). Evidence: explore3 ends_array [user,system] and precedes_assistant [user,system,assistant,user] =
200 (both halves satisfied); explore2 after_user [user,system,user] = 400 message (b) (follows a user ✓ but
is followed by a user ✗); explore1 after_user [user,assistant,system,user] = 400 message (a) (follows a
plain assistant ✗); placement_audit [user,assistant,system,assistant,user] = 400 message (a) — the decisive
case: the system PRECEDES an assistant (would satisfy (b)) yet is still rejected because it FOLLOWS a plain
assistant. First-message system 400s. No beta header changes it. Directive-only form is not a plain message
field: {role:system,content:[],output_config:{}} -> 400 "output_config: Extra inputs are not permitted".
Both the old after_user_not_first hint and the interim precedes_assistant_or_ends_array flag were WRONG —
each modeled only one half of the conjunction (precedes_assistant_or_ends_array would have ACCEPTED the
placement_audit case above). Registry: mid_conversation_system="native" CONFIRMED; mid_system_placement now
follows_user_or_server_tool_and_precedes_assistant_or_ends_array (verified_at 2026-09-10).

### 3. sampling — CONFIRMED (matches registry exactly)
Non-default temperature/top_p/top_k accepted (200) on 4.5 and 4.6; rejected (400) on 4.7/4.8/5-family
("temperature is deprecated for this model"). Default temperature=1 accepted. Evidence: explore2 sweep
across anthropic_all (clean boundary at 4.7), explore1 combined_all_three (sonnet-5, 400). Registry:
[model.sampling] boundary CONFIRMED, verified_at 2026-09-10. No change.

### 4. thinking/adaptive — CONFIRMED
adaptive + output_config.effort {low,medium,high,xhigh,max} accepted (200) on 4.6+, rejected (400) on 4.5
("adaptive thinking is not supported on this model"). disabled accepted; invalid effort -> 400. Legacy
thinking:{type:enabled,budget_tokens>=1024} rejected (400) on 4.7+/5 ("thinking.type.enabled is not
supported for this model. Use thinking.type.adaptive and output_config.effort"). thinking + non-default
temperature -> 400 ("temperature may only be set to 1 when thinking is enabled or in adaptive mode").
Evidence: explore1/2/3. Registry: mode=adaptive, effort incl max on 4.6+, budget.status=rejected on
4.7+/5 CONFIRMED, verified_at 2026-09-10.

### 5. thinking replay — PARTIAL/UNTESTED (probe bug)
Base ant.thinking.tools_base returned 200 but with type_sequence=["tool_use"], has_signature=false: with
adaptive thinking at effort=medium the model emitted NO thinking block — the response was a single tool_use
at /content/0 (n=1), no thinking, no signature. So the replay probes are all unresolvable, in the opposite
way the earlier note claimed: replay_verbatim/replay_disabled reference ${ref:.../content/1} (a tool_use
they assumed sat AFTER a thinking block) but there is no /content/1; replay_tampered_signature references
${ref:.../content/0/thinking} and /content/0/signature which do not exist (content[0] is the tool_use). The
fix is NOT "thinking-aware pointers": it is to index the tool_use DYNAMICALLY and tolerate an absent thinking
block, and/or to force a thinking block via a prompt/effort that reliably elicits one (effort=medium did not)
before capturing the base. Probe-authoring bug, not an API/llm-xlate finding. Redacted-thinking manual.
Registry: none.

### 6. structured output — REFUTED (blocklist) / CONFIRMED (prefill, additionalProperties)
minLength/maxLength/pattern/format ACCEPTED (200) on every current model (anthropic_all sweep). enum,
nested, $ref/$defs accepted. Missing additionalProperties -> 400 (required-field rule). Structured output
with tools accepted. Trailing assistant prefill -> 400 ("does not support assistant message prefill").
Evidence: explore2 string_constraints+format_keyword (200 x9); explore1 missing_additional_properties,
prefill_rejected (400). Registry: [defaults.output] schema_unsupported_keywords ["minLength","maxLength",
"pattern","format"] -> []; re-pinned on legacy 3.7/3.5/3.0 blocks. Fixed crates/xlate/tests/lower.rs
(output_schema_keywords_removed now uses claude_old; added output_schema_keywords_retained_on_current_models).

### 7. tools — CONFIRMED (mostly)/PARTIAL
tool_choice auto/any/tool/none 200; disable_parallel 200; parallel capture 2+ tool_use (200); strict field
accepted (200). tool_result for an absent call -> 400 (result_missing). tool_choice:none -> thinking then
end_turn. Result-ordering probes needing a captured tool_use id skipped (same pointer-index issue as §5).
Registry: tool_choice/parallel/strict CONFIRMED; tool_result must map to a real prior tool_use.

### 8. media — PARTIAL
text document accepted (200); document+citations streams citations_delta (200). Image URL unconfirmed:
API fetches the placeholder URL, 400 "Unable to download the file" (network, not shape). Unsupported mime
-> 400 (correct). base64 image/PDF + Files-API file_id are dataset/files_api (need real upload). Registry:
text_doc + unsupported-mime CONFIRMED; url image UNTESTED (needs a reachable host).

### 9. regrouping — REFUTED (partly): Anthropic more permissive
Consecutive user -> 200; consecutive assistant -> 200; leading assistant -> 200 (no strict alternation, no
assistant-first rejection). Rejected (400): empty text block, whitespace-only trailing assistant (empty
prefill), empty messages. Registry: §7.7 merging consecutive same-role is OPTIONAL for Anthropic (not
needed to avoid a 400); empty/whitespace blocks must still be pruned.

### 10. streaming — CONFIRMED (decoders handle all live streams)
text, tool_use, text->tool->text, citations_delta, cache-usage streams decode+aggregate (check:
response_decode_stream 17/17, chunk_fuzz 17/17). web_search stream manual. §8 codecs CONFIRMED on real bytes.

### 11. errors — CONFIRMED
max_tokens too large, unknown top field, unknown anthropic-beta all -> 400; decode to typed ErrorKind
(check error_decode 48/48). 401/429/529 + prompt-too-long manual/expensive. context_length_patterns
["prompt is too long"] UNTESTED (expensive trigger).

### 12. misc — PARTIAL (two rejections)
stop_sequences echo works (stop_reason:stop_sequence); limit + service_tier accepted. context_management
(top-level) -> 400 "Extra inputs are not permitted"; container -> 400 "Container identifier can only be
provided when using the code execution tool". Registry flag: llm-xlate must not emit context_management/
container to Anthropic unless the enabling beta/tool is present.

## OpenAI Chat Completions

### 13. instruction roles — CONFIRMED. system/developer/both/mid-context/name all 200 (gpt-4o-mini, o4-mini).
### 14. reasoning — CONFIRMED (plan §2 holds). gpt-5.4 Chat: function tools + reasoning_effort!=none -> 400
("...use /v1/responses or set reasoning_effort to 'none'"); effort=none tools work (200). o4-mini: 'none'/
'minimal' rejected (supported low/medium/high/xhigh). max_tokens on reasoning -> 400 (use max_completion_
tokens). Registry: plan §2 row 6 + gpt5_chat() overlay (tools_with_reasoning=No) CONFIRMED. Open: o4-mini
rejects 'minimal' (caps keep the lenient superset; flagged).
### 15. structured output — PARTIAL. strict/non-strict json_schema 200; json_object without "JSON" -> 400,
with "JSON" -> 200; missing additionalProperties (strict) -> 400. strict_unsupported_keywords=200 on
gpt-4o-mini (hints OpenAI blocklist may be over-stated; single model, NOT changed). Open: strict-mode sweep.
### 16. tools — CONFIRMED. required/named, parallel_tool_calls:false, strict:true, unsatisfiable strict
(200), legacy functions/function_call (stop=function_call, 200). Tool msg w/o prior call -> 400.
### 17. media — PARTIAL. file PDF WITHOUT filename -> 400 (filename required). https image URL+detail -> 400
(unreachable). bad-mime data URL accepted (200). input_audio manual. Registry: always emit filename for
Chat file parts.
### 18. streaming — CONFIRMED. include_usage on -> trailing usage-only chunk; off -> none; [DONE]; tool-arg
deltas + finish reasons decode; check stream+fuzz pass.
### 19. sampling/meta — CONFIRMED. n:2, logprobs, seed+fingerprint, service_tier, cache/safety keys, store+
metadata, user, logit_bias all 200. (n:2 accepted at API; llm-xlate n.max=1 is router policy.)
### 20. errors — CONFIRMED. unknown field, temperature out of range (decimal_), empty messages all -> 400,
decode to typed errors.
### 21. openai-compatible — UNTESTED. openai_compat set empty; no server configured; all manual + env-gated.

## OpenAI Responses

### 22. instructions & chaining — CONFIRMED. instructions, mid-input system/developer item, message
shorthand, typed input_text, stored chain (store:true+previous_response_id), not-inherited all 200.
### 23. store & encrypted-reasoning replay — PARTIAL. store:false + include:[reasoning.encrypted_content]
-> reasoning item w/ encrypted content (200); replay verbatim 200. Reasoning item replayed WITHOUT its
paired function_call was ACCEPTED (200) — strict pairing rejection NOT reproduced (may be model/store-mode
specific). second-key/org-bound, GET/DELETE, background+cancel manual. Registry: encrypted_reasoning_include=
true, replay=encrypted_item CONFIRMED (shape); pairing-required NOT confirmed (llm-xlate pairing-drop is
conservative regardless). Open item.
### 24. reasoning — CONFIRMED. effort low/high 200; summary auto/detailed -> reasoning items; raw-reasoning-
text -> reasoning+message; invalid effort -> 400; reasoning on non-reasoning model -> 400.
### 25. tools — CONFIRMED. function tools (default non-strict), strict:true, allowed_tools, reasoning+
function capture all 200; no 400 on a non-strict-able schema (silent acceptance, consistent w/ §2 fallback).
### 26. structured output — CONFIRMED. text.format strict/non-strict/json_object 200; verbosity on
gpt-4o-mini -> 400 (unsupported; verbosity is a gpt-5/gpt-6 capability).
### 27. media — PARTIAL. input_file via file_url = 200; input_image via URL+detail -> 400 (unreachable).
data-URL image + file_id are dataset/files_api.
### 28. streaming — CONFIRMED. text/tool-call/reasoning-summary streams decode+aggregate; check stream+fuzz
pass. §8 Responses event model + sequence_number CONFIRMED.
### 29. GPT-6 — CONFIRMED. gpt-6-astra Responses: text/tools/effort all 200. Chat: function tools+reasoning
-> 400 ("...not supported for gpt-6-astra in /v1/chat/completions"). Tool calling Responses-only.
[[model]] match=gpt-6* protocols=[responses] CONFIRMED.
### 30. errors — CONFIRMED. unknown field -> 400 (unknown_parameter); empty input -> 400 (missing_); decode
to typed errors. 404 unknown prev id, encrypted-from-another-org, stream error event, bad-key manual.

## Cache accounting (2026-09-12, runs cacheacct + cacheacct2, 22 requests, ~$0.06)

Added because every cache counter in the committed dataset was zero: the older `ant.cache.*` probes
proved *acceptance* of `cache_control` but their padded prompts sat under the minimum cacheable
size, so no capture had ever exercised a non-zero cache figure through the crate. The `cacheacct`
family sends a ~3.8k-token prefix cold, then warm, in one run.

### C1. Anthropic reports the FRESH prompt beside the cache counters — CONFIRMED
Cold: `input_tokens 15`, `cache_creation_input_tokens 3935`, `ephemeral_5m 3935`, read 0.
Warm: `input_tokens 15`, `cache_read_input_tokens 3935`, creation 0.
`input_tokens` stays at 15 across both, so it is the uncached remainder and the gross prompt is
`15 + 3935 = 3950`. Evidence: cacheacct2 ant.cacheacct.write / ant.cacheacct.read.

### C2. The OpenAI dialects report the GROSS prompt with the cached portion inside it — CONFIRMED
Chat cold: `prompt_tokens 2860`, `cached_tokens 0`. Chat warm: `prompt_tokens 2860`,
`cached_tokens 2816` — the prompt count does not move, so the 2816 are counted *within* it.
Responses behaves identically (`input_tokens 2860`, `cached_tokens 2688`).
This is the opposite convention to C1 under the same field name, which is exactly the defect
`llm_xlate_usage_plan.md` D1 describes. Evidence: cacheacct chat.cacheacct.write vs
cacheacct2 chat.cacheacct.read / resp.cacheacct.read.

### C3. `ttl = "1h"` produces a real 1-hour write — CONFIRMED
`cache_creation.ephemeral_1h_input_tokens 3975`, `ephemeral_5m 0`, total 3975; the warm twin then
reads 3975. No beta header was needed on this account.
**Prefix caveat:** a body identical to the 5-minute probe apart from `ttl` does NOT write a 1-hour
entry — it reads the existing 5-minute one (observed in run cacheacct, where all four 1h figures
came back 0). Anthropic matches by prefix, so a 1-hour probe needs its own leading text. The
`cacheacct` 1h probes carry a distinct policy preamble for this reason.
Evidence: cacheacct2 ant.cacheacct.ttl_1h_write / ant.cacheacct.ttl_1h_read.

### C4. OpenAI Chat reports no cache-WRITE counter at all — CONFIRMED
`prompt_tokens_details` carries only `cached_tokens` and `audio_tokens`, cold or warm. A cache-write
figure on the Chat dialect can therefore only come from a compatible server (an Anthropic bridge's
`cache_creation_tokens`, vLLM's `created_cache_tokens`). Evidence: cacheacct chat.cacheacct.write.

### C5. OpenAI Responses carries `input_tokens_details.cache_write_tokens`, always 0 — CONFIRMED
Present on every Responses capture in the dataset, cold and warm, and always zero. OpenAI reports
the field but never bills a write through it. It is nonetheless OpenAI's own spelling for the
counter, so llm-xlate decodes it and the Responses encoder emits it.
Evidence: cacheacct resp.cacheacct.write (cold, 0) and cacheacct2 resp.cacheacct.read (warm, 0).

### C6. Anthropic reports `output_tokens_details.thinking_tokens` — CONFIRMED
Present on every Anthropic capture. llm-xlate now maps it to `Usage.reasoning`; before
2026-09-12 it fell into `usage.ext` and was lost on every Anthropic-to-OpenAI route.

### C7. A Chat stream reports NO usage without `stream_options.include_usage` — CONFIRMED
The run-1 `chat.cacheacct.read` stream twin captured `usage: null`, which silently produced an
all-zero cross-protocol rendering in `check`. `stream_options` is rejected when `stream` is false,
so a streaming Chat usage probe has to be a separate probe id — hence `chat.cacheacct.read_stream`.
Registry: `[defaults.transport] stream_usage_opt_in = true` for Chat CONFIRMED.

## Runner notes
- Sequential execution (concurrency reported but one request at a time): ${ref} chaining needs a captured
  predecessor before dependents build. Deferred to a ${ref}-aware scheduler.
- --max-requests 175 for explore1 (168 cheap expansions > the brief's 120; true guard is the $8/provider
  cap; 168 cheap 256-token calls cost < $0.10).

## Open items (more budget / manual triggering)
1. Thinking-replay + tool-result ordering probes are unresolvable: the base capture returned only a single
   tool_use at /content/0 and NO thinking block (adaptive at effort=medium emitted none), so both the
   /content/1 tool_use pointer and the /content/0/thinking pointers dangle. Fix = index the tool_use
   dynamically + tolerate an absent thinking block (or force a thinking block with a prompt/effort that
   reliably elicits one), then rerun families 5 & 7 ordering.
2. Image/file URL sources unconfirmed (API fetches URL; placeholders 400 on download). Needs a reachable host.
3. budget_tokens on 4.6 (ignored vs rejected) not retested with a valid value (explore3 overrode to
   anthropic_new); caps `ignored` for 4.6 unverified this round.
4. Responses reasoning pairing — unpaired replay accepted; strict pairing rejection not reproduced.
5. o4-mini effort enum (minimal rejected) — per-model caps refinement (currently a lenient superset).
6. OpenAI strict schema_unsupported_keywords — a single 200 hints over-statement; needs a strict sweep.
7. Manual-only: 401/429/529, content_filter, second-key/org-bound replay, GET/DELETE /responses/{id},
   background+cancel, Files-API uploads, context-length (expensive), OpenAI-compatible server.
