//! `encode_request`: [`IrRequest`] → Anthropic Messages wire bytes + headers
//! (provider-facing; capability-gated; every lossy step records a [`Degradation`]).
//!
//! Also hosts the shared block-building helpers (`assistant_items_to_blocks`,
//! `media_source_json`, reasoning-block builders) reused by the response and stream encoders.

use base64::Engine;
use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::{
    canon,
    caps::{MidConversationSystem, OutputFormatCap, ReasoningMode, SamplingRule},
    wrap, CacheControl, CacheTtl, Capabilities, Degradations, EncodeCtx, EncodedRequest, Effort,
    HeaderMap, Instruction, IrRequest, Item, MediaSource, OpaqueKind,
    OutputFormat, Part, Position, Role, ToolChoice, ToolDef, XlateError,
};

use crate::shared::seal_or_native;
use crate::wire::{omap, DEFAULT_API_VERSION, EXT_PREFIX, FAMILY, SYSTEM_HEADERS_EXT};

/// The side of the conversation an item belongs to when regrouping into Anthropic turns.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    User,
    Assistant,
}

impl Side {
    fn role(self) -> &'static str {
        match self {
            Side::User => "user",
            Side::Assistant => "assistant",
        }
    }
}

/// A regrouped run of contiguous same-side items.
struct Group {
    side: Side,
    items: Vec<usize>,
}

/// Flags gathered during encoding that drive conditional beta headers.
#[derive(Default)]
struct BetaFlags {
    structured_output: bool,
    mid_conversation_effort: bool,
    system_clear_at: bool,
}

/// Encode an IR request into an Anthropic Messages request.
pub fn encode_request(
    req: &IrRequest,
    caps: &Capabilities,
    ctx: &EncodeCtx,
) -> Result<EncodedRequest, XlateError> {
    let mut degradations = Degradations::new();
    let mut betas = BetaFlags::default();

    // ---- max_tokens (required by Anthropic) ----
    let max_tokens = req
        .limits
        .max_output_tokens
        .or(caps.transport.default_max_output_tokens)
        .unwrap_or(4096);

    // ---- reasoning → thinking + output_config.effort ----
    // Top-level `output_config.effort` is generally available (plan §2); the
    // `mid_conversation_effort` beta is only needed for per-instruction effort overrides,
    // which set the flag in `build_instruction_message`.
    let (thinking, effort_token, drop_temperature) =
        build_thinking(req, caps, max_tokens, &mut degradations);

    // ---- output format ----
    let output_format = build_output_format(req, caps, &mut degradations);
    if output_format.is_some() {
        betas.structured_output = true;
    }

    // ---- system (leading instructions) ----
    let system = build_system(req, caps, &mut degradations);

    // ---- messages (regroup + mid instructions) ----
    let messages = build_messages(req, caps, ctx, &mut degradations, &mut betas)?;

    // ---- tools ----
    let tools = build_tools(req, caps, &mut degradations);

    // ---- tool_choice (+ parallel) ----
    let tool_choice = build_tool_choice(req, caps, &mut degradations);

    // ---- output_config assembly (effort + format) ----
    let output_config = build_output_config(effort_token, output_format, req);

    // ---- assemble body in fixed field order ----
    let mut body = omap();
    body.insert("model".into(), Value::from(req.model.upstream()));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("max_tokens".into(), Value::from(max_tokens));
    if let Some(system) = system {
        body.insert("system".into(), system);
    }
    if let Some(tools) = tools {
        body.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tc) = tool_choice {
        body.insert("tool_choice".into(), tc);
    }
    if let Some(th) = thinking {
        body.insert("thinking".into(), th);
    }
    if let Some(oc) = output_config {
        body.insert("output_config".into(), oc);
    }

    build_sampling(req, caps, drop_temperature, &mut body, &mut degradations);

    // The backend's streaming mode decides, not the client's: a stream-only backend must be
    // sent `stream: true` even for a non-streaming client (the router aggregates the SSE
    // back into one response), and a non-streaming-only backend must not be sent it at all.
    let upstream_streams = caps.upstream_streams(req.stream);
    if upstream_streams {
        body.insert("stream".into(), Value::Bool(true));
    }

    build_meta(req, caps, &mut body, &mut degradations);

    // request-level cache: explicit hint, or auto-injection for a non-Anthropic client.
    let want_request_cache = req.cache.request_level
        || (ctx.client_protocol.family() != FAMILY && caps.cache.auto_request_level.is_yes());
    if want_request_cache {
        let mut cc = omap();
        cc.insert("type".into(), Value::from("ephemeral"));
        body.insert("cache_control".into(), Value::Object(cc));
    }

    // ---- ext writeback (anthropic.* → top level / output_config) ----
    write_back_ext(req, &mut body, &mut betas);

    // ---- headers ----
    let mut headers = build_headers(req, caps, &betas);

    // ---- session affinity: emit into the first accepted place (field or header) ----
    llm_xlate_core::session::apply_session(
        caps,
        req.session.id.as_deref(),
        req.cache.prompt_cache_key.as_deref(),
        &mut body,
        &mut headers,
        &mut degradations,
    );

    // ---- enforce ≤ 4 cache_control breakpoints ----
    let mut body_value = Value::Object(body);
    enforce_cache_breakpoints(&mut body_value, &mut degradations);

    let bytes: Bytes = canon::to_bytes(&body_value);

    let mut out_ctx = ctx.clone();
    // `EncodeCtx::stream` is the *client's* intent — the router reads it back to decide whether
    // to stream the response to the caller — so it must not be overwritten with the upstream
    // decision. How the upstream is read is carried separately by `EncodedRequest::
    // upstream_streams`. Conflating the two made a stream-only backend force SSE onto a client
    // that asked for a single JSON body (and a non-streaming-only backend swallow a client's
    // stream).
    out_ctx.stream = req.stream;

    Ok(EncodedRequest {
        body: bytes,
        headers,
        upstream_streams,
        ctx: out_ctx,
        degradations,
    })
}

// ---------------------------------------------------------------------------
// System / instructions
// ---------------------------------------------------------------------------

/// Build the top-level `system` from leading instructions (System + Developer, original
/// order; Developer text emitted as-is), preceded by the captured client header blocks the
/// backend accepts.
fn build_system(
    req: &IrRequest,
    caps: &Capabilities,
    degradations: &mut Degradations,
) -> Option<Value> {
    let mut blocks = build_system_headers(req, caps, degradations);
    for instr in &req.instructions {
        if !matches!(instr.position, Position::Leading) {
            continue;
        }
        for part in &instr.content {
            if let Part::Text { text, cache_control, .. } = part {
                let mut b = omap();
                b.insert("type".into(), Value::from("text"));
                b.insert("text".into(), Value::from(text.clone()));
                if let Some(cc) = cache_control {
                    b.insert("cache_control".into(), cache_control_json(cc));
                }
                blocks.push(Value::Object(b));
            }
        }
        // Leading instruction-level cache_control attaches to the last block.
        if let Some(cc) = &instr.cache_control {
            if let Some(Value::Object(last)) = blocks.last_mut() {
                last.entry("cache_control").or_insert_with(|| cache_control_json(cc));
            }
        }
        if instr.effort.is_some() {
            degradations.dropped("instructions.effort", "leading system effort override is not representable");
        }
        if instr.clear_at.is_some() {
            degradations.dropped("instructions.clear_at", "leading system clear_at is not representable");
        }
    }
    if blocks.is_empty() {
        None
    } else {
        Some(Value::Array(blocks))
    }
}

/// Re-emit the captured `anthropic.system_headers` blocks whose name the backend lists in
/// `instructions.system_headers`, verbatim and in captured order; drop (and report) the rest.
/// A header such as Claude Code's billing line changes every request, so forwarding it to a
/// backend that treats it as prompt text would defeat prefix caching.
fn build_system_headers(
    req: &IrRequest,
    caps: &Capabilities,
    degradations: &mut Degradations,
) -> Vec<Value> {
    let mut blocks = Vec::new();
    let Some(Value::Array(headers)) = req.ext.get(&format!("{EXT_PREFIX}{SYSTEM_HEADERS_EXT}"))
    else {
        return blocks;
    };
    for h in headers {
        let name = h.get("name").and_then(Value::as_str).unwrap_or("");
        let Some(text) = h.get("text").and_then(Value::as_str) else { continue };
        if !caps.forwards_system_header(name) {
            degradations.dropped(
                format!("ext.{EXT_PREFIX}{SYSTEM_HEADERS_EXT}.{name}"),
                "backend does not accept this system header (instructions.system_headers)",
            );
            continue;
        }
        let mut b = omap();
        b.insert("type".into(), Value::from("text"));
        b.insert("text".into(), Value::from(text));
        if let Some(cc) = h.get("cache_control") {
            b.insert("cache_control".into(), cc.clone());
        }
        blocks.push(Value::Object(b));
    }
    blocks
}

// ---------------------------------------------------------------------------
// Messages (regroup + mid instructions)
// ---------------------------------------------------------------------------

/// Wrap ordered content blocks into a `{role, content:[blocks]}` message object.
fn message_object(role: &str, blocks: Vec<Value>) -> Value {
    let mut o = omap();
    o.insert("role".into(), Value::from(role));
    o.insert("content".into(), Value::Array(blocks));
    Value::Object(o)
}

fn build_messages(
    req: &IrRequest,
    caps: &Capabilities,
    ctx: &EncodeCtx,
    degradations: &mut Degradations,
    betas: &mut BetaFlags,
) -> Result<Vec<Value>, XlateError> {
    // Item indices a mid-context instruction is anchored before (`Position::Before(i)`). Such an
    // index forces a group boundary so a run of same-side items is split at the instruction —
    // otherwise two user turns straddling a mid-system are coalesced, the instruction's target
    // group becomes the first turn (never user-preceded), the native `role:"system"` block never
    // fires, and the inline-wrap fallback is hoisted to the FRONT of the merged turn instead of
    // landing between the two (§11.4). Groups split only for placement; a split that does not get
    // a native system between its halves is re-merged at emit time to keep roles alternating.
    let mut instr_boundary = vec![false; req.items.len()];
    for instr in &req.instructions {
        if let Position::Before(i) = instr.position {
            if i < req.items.len() {
                instr_boundary[i] = true;
            }
        }
    }

    // Build side groups.
    let mut groups: Vec<Group> = Vec::new();
    for (i, item) in req.items.iter().enumerate() {
        let side = if item.is_user_side() { Side::User } else { Side::Assistant };
        match groups.last_mut() {
            Some(g) if g.side == side && !instr_boundary[i] => g.items.push(i),
            _ => groups.push(Group { side, items: vec![i] }),
        }
    }

    // Trailing assistant message + prefill disallowed → unsupported.
    if let Some(last) = groups.last() {
        if last.side == Side::Assistant && caps.output.prefill_allowed.is_no_or_unknown() {
            return Err(XlateError::unsupported(
                "messages",
                "trailing assistant message (prefill) is not allowed for this model",
            ));
        }
    }

    // Map each item index → its group index.
    let mut item_group = vec![0usize; req.items.len()];
    for (gi, g) in groups.iter().enumerate() {
        for &it in &g.items {
            item_group[it] = gi;
        }
    }

    // Classify mid instructions: native system messages (before a group) vs inline-wrap blocks.
    // An inline-wrap block goes into a user message at the position closest to its anchor that
    // keeps the wire valid: appended to the END of the preceding user group when there is one
    // (exactly where it was in the conversation), otherwise into the next user group AFTER its
    // `tool_result` blocks — Anthropic rejects a user message whose tool results do not come
    // first, so a wrap must never be placed in front of them. Anchors past the last group go to
    // a trailing user message.
    let mut native_before: Vec<Vec<&Instruction>> = vec![Vec::new(); groups.len() + 1];
    let mut wrap_append: Vec<Vec<Value>> = vec![Vec::new(); groups.len()];
    let mut wrap_after_results: Vec<Vec<Value>> = vec![Vec::new(); groups.len()];
    let mut wrap_trailing: Vec<Value> = Vec::new();

    let native_supported =
        caps.instructions.mid_conversation_system == Some(MidConversationSystem::Native);

    for instr in &req.instructions {
        let i = match instr.position {
            Position::Before(i) => i,
            Position::Leading => continue,
        };
        let target_group = if i >= req.items.len() { groups.len() } else { item_group[i] };
        // Native condition: caps Native AND previous group is user-side AND not first.
        let prev_user = target_group >= 1 && groups.get(target_group - 1).map(|g| g.side) == Some(Side::User);
        if native_supported && prev_user && target_group >= 1 && target_group < groups.len() {
            native_before[target_group].push(instr);
        } else {
            let text = instr.text();
            let block = wrap_system_block(&text);
            if target_group >= groups.len() {
                wrap_trailing.push(block);
            } else if prev_user {
                wrap_append[target_group - 1].push(block);
            } else {
                match (target_group..groups.len()).find(|&gi| groups[gi].side == Side::User) {
                    Some(gi) => wrap_after_results[gi].push(block),
                    None => wrap_trailing.push(block),
                }
            }
            degradations.wrapped(
                "instructions",
                "mid-context system inline-wrapped into a user message",
            );
            if instr.effort.is_some() {
                degradations.dropped("instructions.effort", "effort override lost by inline-wrap");
            }
            if instr.clear_at.is_some() {
                degradations.dropped("instructions.clear_at", "clear_at lost by inline-wrap");
            }
        }
    }

    // Emit.
    let mut out: Vec<Value> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        // Native system messages before this group.
        for instr in &native_before[gi] {
            out.push(build_native_system(instr, caps, degradations, betas));
        }
        // Build the group's blocks, splicing in its inline-wrap blocks (user groups only): after
        // the leading tool_result blocks, and at the end.
        let mut blocks = build_group_blocks(req, g, ctx, caps, degradations);
        if g.side == Side::User {
            let n_results = blocks
                .iter()
                .take_while(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                .count();
            blocks.splice(n_results..n_results, wrap_after_results[gi].iter().cloned());
            blocks.extend(wrap_append[gi].iter().cloned());
        }

        // Re-merge into the previous message when it is the same role and no native system was
        // emitted between them (an instruction boundary split a run of same-side items but the
        // instruction was inline-wrapped, not made native). This keeps roles alternating; the
        // wrap block (appended to the first half) then sits between the two turns.
        let role = g.side.role();
        if native_before[gi].is_empty() {
            if let Some(Value::Object(prev)) = out.last_mut() {
                if prev.get("role").and_then(Value::as_str) == Some(role) {
                    if let Some(Value::Array(content)) = prev.get_mut("content") {
                        content.extend(blocks);
                        continue;
                    }
                }
            }
        }

        if blocks.is_empty() {
            // A group that produced no blocks (e.g. all-empty text) would violate Anthropic's
            // non-empty-content rule; emit a single empty text block placeholder.
            let mut t = omap();
            t.insert("type".into(), Value::from("text"));
            t.insert("text".into(), Value::from(""));
            blocks.push(Value::Object(t));
        }
        out.push(message_object(role, blocks));
    }

    // Trailing inline-wrap blocks → a final user message.
    if !wrap_trailing.is_empty() {
        out.push(message_object("user", wrap_trailing));
    }

    Ok(out)
}

/// Build a native `{role:"system"}` message from a mid instruction, with optional block-level
/// `effort` / `clear_at` when the model supports overriding them.
fn build_native_system(
    instr: &Instruction,
    caps: &Capabilities,
    degradations: &mut Degradations,
    betas: &mut BetaFlags,
) -> Value {
    let mut blocks = Vec::new();
    for part in &instr.content {
        if let Part::Text { text, cache_control, .. } = part {
            let mut b = omap();
            b.insert("type".into(), Value::from("text"));
            b.insert("text".into(), Value::from(text.clone()));
            if let Some(cc) = cache_control {
                b.insert("cache_control".into(), cache_control_json(cc));
            }
            blocks.push(Value::Object(b));
        }
    }
    if blocks.is_empty() {
        let mut t = omap();
        t.insert("type".into(), Value::from("text"));
        t.insert("text".into(), Value::from(instr.text()));
        blocks.push(Value::Object(t));
    }

    let mut msg = omap();
    msg.insert("role".into(), Value::from("system"));
    msg.insert("content".into(), Value::Array(blocks));

    if let Some(effort) = &instr.effort {
        if caps.instructions.system_effort_override.is_yes() {
            msg.insert("effort".into(), Value::from(effort.ant_effort_token()));
            betas.mid_conversation_effort = true;
        } else {
            degradations.dropped("instructions.effort", "system effort override unsupported");
        }
    }
    if let Some(clear_at) = &instr.clear_at {
        if caps.instructions.system_clear_at.is_yes() {
            msg.insert("clear_at".into(), clear_at.raw.clone());
            betas.system_clear_at = true;
        } else {
            degradations.dropped("instructions.clear_at", "system clear_at unsupported");
        }
    }

    Value::Object(msg)
}

/// Build a text block wrapping mid-context instruction text (`wrap_v1`).
fn wrap_system_block(text: &str) -> Value {
    let mut b = omap();
    b.insert("type".into(), Value::from("text"));
    b.insert("text".into(), Value::from(wrap::system_message(text)));
    Value::Object(b)
}

/// Build the ordered content blocks for one regrouped message.
fn build_group_blocks(
    req: &IrRequest,
    g: &Group,
    ctx: &EncodeCtx,
    caps: &Capabilities,
    degradations: &mut Degradations,
) -> Vec<Value> {
    let mut blocks = Vec::new();
    match g.side {
        Side::User => {
            // All tool_result blocks first (item order), then other parts.
            for &i in &g.items {
                if let Item::ToolResult { call_id, content, is_error, .. } = &req.items[i] {
                    blocks.push(build_tool_result(call_id.as_str(), content, *is_error, caps, degradations));
                }
            }
            for &i in &g.items {
                if let Item::Message { content, .. } = &req.items[i] {
                    blocks.extend(build_parts(content, caps, degradations));
                }
            }
        }
        Side::Assistant => {
            // thinking / redacted first, then text, then tool_use, then provider items, compaction.
            for &i in &g.items {
                if let Item::Reasoning(r) = &req.items[i] {
                    if let Some(b) = build_reasoning_block(r, ctx, false, degradations) {
                        blocks.push(b);
                    }
                }
            }
            for &i in &g.items {
                if let Item::Message { content, .. } = &req.items[i] {
                    blocks.extend(build_parts(content, caps, degradations));
                }
            }
            for &i in &g.items {
                if let Item::ToolCall { call_id, name, arguments, .. } = &req.items[i] {
                    match build_tool_use(call_id.as_str(), name, arguments) {
                        Ok(b) => blocks.push(b),
                        Err(_) => degradations.dropped(
                            "items.tool_call",
                            "tool-call arguments were not valid JSON",
                        ),
                    }
                }
            }
            for &i in &g.items {
                match &req.items[i] {
                    Item::ProviderToolCall(oi) | Item::ProviderToolResult(oi) => {
                        if oi.family == FAMILY {
                            blocks.push(oi.raw.clone());
                        } else {
                            degradations.dropped("items.provider_tool", "foreign provider tool dropped");
                        }
                    }
                    Item::Compaction(blob) => {
                        if let Ok(v) = serde_json::from_str::<Value>(&blob.data) {
                            blocks.push(v);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    blocks
}

/// Build the content blocks for a list of message parts (text/image/document).
fn build_parts(content: &[Part], caps: &Capabilities, degradations: &mut Degradations) -> Vec<Value> {
    let mut out = Vec::new();
    for part in content {
        match part {
            Part::Text { text, annotations, cache_control } => {
                if text.is_empty() && annotations.is_empty() {
                    continue; // empty text blocks omitted
                }
                let mut b = omap();
                b.insert("type".into(), Value::from("text"));
                b.insert("text".into(), Value::from(text.clone()));
                if !annotations.is_empty() {
                    b.insert(
                        "citations".into(),
                        Value::Array(annotations.iter().map(|a| a.raw.clone()).collect()),
                    );
                }
                if let Some(cc) = cache_control {
                    b.insert("cache_control".into(), cache_control_json(cc));
                }
                out.push(Value::Object(b));
            }
            Part::Image(src) => {
                let mut b = omap();
                b.insert("type".into(), Value::from("image"));
                b.insert("source".into(), media_source_json(src));
                out.push(Value::Object(b));
            }
            Part::Document { source, title, media_type } => {
                let mut b = omap();
                b.insert("type".into(), Value::from("document"));
                b.insert("source".into(), document_source_json(source, media_type));
                if let Some(t) = title {
                    b.insert("title".into(), Value::from(t.clone()));
                }
                out.push(Value::Object(b));
            }
            Part::Audio(_) => {
                if caps.media.audio.sources.as_ref().is_some_and(|s| !s.is_empty()) {
                    // (Sources present but no concrete Anthropic audio shape is defined yet.)
                    degradations.dropped("items.audio", "audio input not representable for Anthropic");
                } else {
                    degradations.dropped("items.audio", "audio input unsupported by target");
                }
            }
            Part::Refusal { text } => {
                let mut b = omap();
                b.insert("type".into(), Value::from("text"));
                b.insert("text".into(), Value::from(text.clone()));
                out.push(Value::Object(b));
            }
            Part::Opaque(blob) => {
                // A redacted-thinking blob carried in content.
                if blob.kind == OpaqueKind::Redacted && blob.family == FAMILY {
                    let mut b = omap();
                    b.insert("type".into(), Value::from("redacted_thinking"));
                    b.insert("data".into(), Value::from(blob.data.clone()));
                    out.push(Value::Object(b));
                } else {
                    degradations.dropped("items.opaque", "opaque content block dropped");
                }
            }
        }
    }
    out
}

/// Build a `tool_result` block.
fn build_tool_result(
    call_id: &str,
    content: &[Part],
    is_error: bool,
    caps: &Capabilities,
    degradations: &mut Degradations,
) -> Value {
    let mut b = omap();
    b.insert("type".into(), Value::from("tool_result"));
    b.insert("tool_use_id".into(), Value::from(call_id));

    // Single plain-text part → string content; else an array of allowed blocks.
    let single_text = content.len() == 1
        && matches!(&content[0], Part::Text { annotations, cache_control, .. } if annotations.is_empty() && cache_control.is_none());
    if single_text {
        if let Part::Text { text, .. } = &content[0] {
            b.insert("content".into(), Value::from(text.clone()));
        }
    } else {
        let allowed = caps.tools.result_content.clone().unwrap_or_default();
        let allows = |k: llm_xlate_core::caps::ResultContentKind| allowed.contains(&k);
        use llm_xlate_core::caps::ResultContentKind as RC;
        let mut blocks = Vec::new();
        for part in content {
            match part {
                Part::Text { .. } => blocks.extend(build_parts(std::slice::from_ref(part), caps, degradations)),
                Part::Image(_) if allows(RC::Image) => {
                    blocks.extend(build_parts(std::slice::from_ref(part), caps, degradations))
                }
                Part::Document { .. } if allows(RC::Document) => {
                    blocks.extend(build_parts(std::slice::from_ref(part), caps, degradations))
                }
                other => {
                    // Fallback: fold into a text block.
                    let txt = match other {
                        Part::Image(_) => "[image attachment]".to_string(),
                        Part::Document { title, .. } => {
                            wrap::document(title.as_deref(), "[document attachment]")
                        }
                        _ => "[attachment]".to_string(),
                    };
                    let mut tb = omap();
                    tb.insert("type".into(), Value::from("text"));
                    tb.insert("text".into(), Value::from(txt));
                    blocks.push(Value::Object(tb));
                    degradations.folded("tool_result.content", "unsupported tool-result part folded to text");
                }
            }
        }
        b.insert("content".into(), Value::Array(blocks));
    }
    if is_error {
        b.insert("is_error".into(), Value::Bool(true));
    }
    Value::Object(b)
}

/// Build a `tool_use` block; `input` is the parsed tool arguments (key order preserved).
fn build_tool_use(call_id: &str, name: &str, arguments: &llm_xlate_core::JsonText) -> Result<Value, XlateError> {
    let input = if arguments.as_str().trim().is_empty() {
        Value::Object(Map::new())
    } else {
        canon::json_text::to_value(arguments)?
    };
    let mut b = omap();
    b.insert("type".into(), Value::from("tool_use"));
    b.insert("id".into(), Value::from(call_id));
    b.insert("name".into(), Value::from(name));
    b.insert("input".into(), input);
    Ok(Value::Object(b))
}

/// Build a `thinking` / `redacted_thinking` block for reasoning replay.
///
/// `client_facing` distinguishes the two envelope boundaries: on the client-facing path
/// (`encode_response`) a foreign-family blob is sealed into an `rtr1.` envelope so the client
/// can replay it; on the provider-facing path (`encode_request`) a foreign-family blob cannot
/// be replayed to the real Anthropic API and is dropped with a Degradation. Anthropic rejects
/// thinking blocks lacking a valid signature, so a reasoning item without an opaque envelope
/// is always dropped (never fabricated as `signature:""`).
fn build_reasoning_block(
    r: &llm_xlate_core::ReasoningItem,
    ctx: &EncodeCtx,
    client_facing: bool,
    degradations: &mut Degradations,
) -> Option<Value> {
    // Client-facing (`encode_response`) boundary (§7.2 Resp→Ant exposure table). The client is
    // receiving, not sending, so a thinking block need not carry a signature: render the
    // reasoning/summary text and attach the sealed envelope only when an opaque carrier exists.
    // On replay the router decoder treats a missing signature as no opaque and `lower()` applies
    // the replay policy. Kept in agreement with the streaming encoder (law §11.8).
    if client_facing {
        // Text carrier: reasoning text takes precedence, else the joined summary (two newlines).
        let text = match &r.text {
            Some(t) if !t.is_empty() => Some(t.clone()),
            _ if !r.summary.is_empty() => Some(r.summary.join("\n\n")),
            _ => None,
        };
        if let Some(t) = text {
            let mut b = omap();
            b.insert("type".into(), Value::from("thinking"));
            b.insert("thinking".into(), Value::from(t));
            if let Some(blob) = &r.opaque {
                b.insert("signature".into(), Value::from(seal_or_native(ctx, blob)));
            }
            return Some(Value::Object(b));
        }
        // No text and no summary: an opaque carrier becomes redacted_thinking, else nothing.
        return match &r.opaque {
            Some(blob) => {
                let mut b = omap();
                b.insert("type".into(), Value::from("redacted_thinking"));
                b.insert("data".into(), Value::from(seal_or_native(ctx, blob)));
                Some(Value::Object(b))
            }
            None => {
                degradations
                    .dropped("items.reasoning", "reasoning with no text, summary, or opaque dropped");
                None
            }
        };
    }
    let blob = match &r.opaque {
        Some(b) => b,
        None => {
            degradations.dropped("items.reasoning", "reasoning without a signature dropped");
            return None;
        }
    };
    // Provider-facing boundary: a foreign-family blob has no valid Anthropic carrier.
    if !client_facing && blob.family != FAMILY {
        degradations.dropped("items.reasoning", "foreign reasoning cannot be replayed to Anthropic");
        return None;
    }
    match blob.kind {
        OpaqueKind::Signature | OpaqueKind::Encrypted => {
            let mut b = omap();
            b.insert("type".into(), Value::from("thinking"));
            b.insert("thinking".into(), Value::from(r.text.clone().unwrap_or_default()));
            b.insert("signature".into(), Value::from(seal_or_native(ctx, blob)));
            Some(Value::Object(b))
        }
        OpaqueKind::Redacted => {
            let mut b = omap();
            b.insert("type".into(), Value::from("redacted_thinking"));
            b.insert("data".into(), Value::from(seal_or_native(ctx, blob)));
            Some(Value::Object(b))
        }
        OpaqueKind::Compaction => {
            degradations.dropped("items.reasoning", "compaction blob in reasoning slot dropped");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Media
// ---------------------------------------------------------------------------

/// Build an image `source` object from a [`MediaSource`].
pub(crate) fn media_source_json(src: &MediaSource) -> Value {
    let mut m = omap();
    match src {
        MediaSource::Base64 { media_type, data } => {
            m.insert("type".into(), Value::from("base64"));
            m.insert("media_type".into(), Value::from(media_type.clone()));
            m.insert(
                "data".into(),
                Value::from(base64::engine::general_purpose::STANDARD.encode(data)),
            );
        }
        MediaSource::Url(u) => {
            m.insert("type".into(), Value::from("url"));
            m.insert("url".into(), Value::from(u.clone()));
        }
        MediaSource::FileRef { id, .. } => {
            m.insert("type".into(), Value::from("file"));
            m.insert("file_id".into(), Value::from(id.clone()));
        }
        MediaSource::Text(t) => {
            m.insert("type".into(), Value::from("text"));
            m.insert("media_type".into(), Value::from("text/plain"));
            m.insert("data".into(), Value::from(t.clone()));
        }
    }
    Value::Object(m)
}

/// Build a document `source` object.
fn document_source_json(src: &MediaSource, media_type: &str) -> Value {
    match src {
        MediaSource::Text(t) => {
            let mut m = omap();
            m.insert("type".into(), Value::from("text"));
            m.insert("media_type".into(), Value::from("text/plain"));
            m.insert("data".into(), Value::from(t.clone()));
            Value::Object(m)
        }
        MediaSource::Base64 { data, .. } => {
            let mut m = omap();
            m.insert("type".into(), Value::from("base64"));
            m.insert("media_type".into(), Value::from(media_type.to_string()));
            m.insert(
                "data".into(),
                Value::from(base64::engine::general_purpose::STANDARD.encode(data)),
            );
            Value::Object(m)
        }
        _ => media_source_json(src),
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

fn build_tools(req: &IrRequest, caps: &Capabilities, degradations: &mut Degradations) -> Option<Vec<Value>> {
    if req.tools.is_empty() {
        return None;
    }
    let strict_ok = caps.tools.strict.supported.is_yes();
    let mut out = Vec::new();
    for tool in &req.tools {
        match tool {
            ToolDef::Function { name, description, parameters, strict, cache_control } => {
                let mut t = omap();
                t.insert("name".into(), Value::from(name.clone()));
                if let Some(d) = description {
                    t.insert("description".into(), Value::from(d.clone()));
                }
                t.insert("input_schema".into(), parameters.clone());
                if let Some(s) = strict {
                    if strict_ok {
                        t.insert("strict".into(), Value::Bool(*s));
                    } else {
                        degradations.downgraded("tools.strict", "strict not supported; dropped");
                    }
                }
                if let Some(cc) = cache_control {
                    t.insert("cache_control".into(), cache_control_json(cc));
                }
                out.push(Value::Object(t));
            }
            ToolDef::Provider(oi) => {
                if oi.family == FAMILY {
                    out.push(oi.raw.clone());
                } else {
                    degradations.dropped("tools.provider", "foreign provider tool dropped");
                }
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn build_tool_choice(req: &IrRequest, _caps: &Capabilities, _degradations: &mut Degradations) -> Option<Value> {
    let has_tools = !req.tools.is_empty();
    let disable_parallel = req.parallel_tool_calls == Some(false);

    let mut tc = omap();
    match &req.tool_choice {
        ToolChoice::Auto => {
            if !has_tools && !disable_parallel {
                return None;
            }
            tc.insert("type".into(), Value::from("auto"));
        }
        ToolChoice::None => {
            tc.insert("type".into(), Value::from("none"));
        }
        ToolChoice::Required => {
            tc.insert("type".into(), Value::from("any"));
        }
        ToolChoice::Named(name) => {
            tc.insert("type".into(), Value::from("tool"));
            tc.insert("name".into(), Value::from(name.clone()));
        }
    }
    if disable_parallel {
        tc.insert("disable_parallel_tool_use".into(), Value::Bool(true));
    }
    Some(Value::Object(tc))
}

// ---------------------------------------------------------------------------
// Reasoning / thinking
// ---------------------------------------------------------------------------

/// Returns `(thinking, effort_token, drop_temperature)`.
fn build_thinking(
    req: &IrRequest,
    caps: &Capabilities,
    max_tokens: u32,
    degradations: &mut Degradations,
) -> (Option<Value>, Option<String>, bool) {
    let r = &req.reasoning;
    let has = r.effort.is_some() || r.enabled.is_some() || r.budget_tokens.is_some();
    if !has {
        return (None, None, false);
    }
    let disabled = r.enabled == Some(false) || r.effort == Some(Effort::None);
    let forced_temp = caps.reasoning.forced_temperature_with_thinking.is_yes();

    let disabled_thinking = || {
        let mut t = omap();
        t.insert("type".into(), Value::from("disabled"));
        Value::Object(t)
    };

    match caps.reasoning.mode {
        Some(ReasoningMode::Adaptive) | Some(ReasoningMode::EffortOnly) => {
            if disabled {
                return (Some(disabled_thinking()), None, false);
            }
            let mut t = omap();
            t.insert("type".into(), Value::from("adaptive"));
            let token = r.effort.and_then(|e| effort_token_caps(e, caps, degradations));
            (Some(Value::Object(t)), token, forced_temp)
        }
        Some(ReasoningMode::Budget) => {
            if disabled {
                return (Some(disabled_thinking()), None, false);
            }
            let budget = r
                .budget_tokens
                .unwrap_or_else(|| r.effort.unwrap_or(Effort::Medium).ant_budget(max_tokens));
            let mut t = omap();
            t.insert("type".into(), Value::from("enabled"));
            t.insert("budget_tokens".into(), Value::from(budget));
            (Some(Value::Object(t)), None, forced_temp)
        }
        Some(ReasoningMode::None) | None => {
            degradations.dropped("reasoning", "target model has no reasoning mode");
            (None, None, false)
        }
    }
}

/// The adaptive-effort token for the given effort, applying the caps `max`/`xhigh` rule.
fn effort_token_caps(e: Effort, caps: &Capabilities, degradations: &mut Degradations) -> Option<String> {
    if e == Effort::None {
        return None;
    }
    if e == Effort::Max {
        let has_max = caps
            .reasoning
            .effort_levels
            .as_ref()
            .is_some_and(|levels| levels.contains(&Effort::Max));
        if has_max {
            return Some("max".to_string());
        }
        degradations.downgraded("reasoning.effort", "max not supported; downgraded to xhigh");
        return Some("xhigh".to_string());
    }
    Some(e.ant_effort_token().to_string())
}

// ---------------------------------------------------------------------------
// Output format / output_config
// ---------------------------------------------------------------------------

fn build_output_format(req: &IrRequest, caps: &Capabilities, degradations: &mut Degradations) -> Option<Value> {
    let cap = caps.output.format;
    let allows_object = matches!(cap, Some(OutputFormatCap::JsonObject) | Some(OutputFormatCap::Both));
    let allows_schema = matches!(cap, Some(OutputFormatCap::JsonSchema) | Some(OutputFormatCap::Both));

    match &req.output.format {
        OutputFormat::Text => None,
        OutputFormat::JsonObject => {
            if !allows_object && !allows_schema {
                degradations.dropped("output.format", "structured output unsupported");
                return None;
            }
            let mut fmt = omap();
            fmt.insert("type".into(), Value::from("json_schema"));
            let mut schema = omap();
            schema.insert("type".into(), Value::from("object"));
            fmt.insert("schema".into(), Value::Object(schema));
            Some(Value::Object(fmt))
        }
        OutputFormat::JsonSchema { schema, .. } => {
            if !allows_schema {
                degradations.dropped("output.format", "json_schema unsupported");
                return None;
            }
            let mut fmt = omap();
            fmt.insert("type".into(), Value::from("json_schema"));
            fmt.insert("schema".into(), schema.clone());
            Some(Value::Object(fmt))
        }
    }
}

/// Merge the adaptive effort token, structured-output format, and any preserved
/// `anthropic.output_config.*` ext keys into a single `output_config` object.
fn build_output_config(effort_token: Option<String>, format: Option<Value>, req: &IrRequest) -> Option<Value> {
    let mut oc = omap();
    if let Some(e) = effort_token {
        oc.insert("effort".into(), Value::from(e));
    }
    if let Some(f) = format {
        oc.insert("format".into(), f);
    }
    for (k, v) in req.ext.iter() {
        if let Some(rest) = k.strip_prefix("anthropic.output_config.") {
            oc.insert(rest.to_string(), v.clone());
        }
    }
    if oc.is_empty() {
        None
    } else {
        Some(Value::Object(oc))
    }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

fn build_sampling(
    req: &IrRequest,
    caps: &Capabilities,
    drop_temperature: bool,
    body: &mut Map<String, Value>,
    degradations: &mut Degradations,
) {
    let s = &req.sampling;

    // temperature
    if let Some(v) = s.temperature {
        if drop_temperature {
            degradations.dropped("temperature", "thinking forces temperature to be dropped");
        } else {
            emit_sampling(body, degradations, "temperature", v, caps.sampling.temperature);
        }
    }
    if let Some(v) = s.top_p {
        emit_sampling(body, degradations, "top_p", v, caps.sampling.top_p);
    }
    if let Some(v) = s.top_k {
        emit_sampling(body, degradations, "top_k", v as f64, caps.sampling.top_k);
    }

    // stop_sequences (with count cap)
    if !req.limits.stop_sequences.is_empty() {
        let max = caps.sampling.stop_sequences.and_then(|m| m.max);
        let mut seqs = req.limits.stop_sequences.clone();
        if let Some(m) = max {
            if seqs.len() > m as usize {
                seqs.truncate(m as usize);
                degradations.dropped("stop_sequences", "truncated to model maximum");
            }
        }
        body.insert("stop_sequences".into(), Value::Array(seqs.into_iter().map(Value::from).collect()));
    }

    // Unsupported sampling knobs present in IR but with no Anthropic slot.
    for (name, present) in [
        ("seed", s.seed.is_some()),
        ("frequency_penalty", s.frequency_penalty.is_some()),
        ("presence_penalty", s.presence_penalty.is_some()),
        ("logit_bias", s.logit_bias.is_some()),
        ("logprobs", s.logprobs.is_some()),
    ] {
        if present {
            degradations.dropped(name, "sampling parameter unsupported by Anthropic");
        }
    }
}

fn emit_sampling(
    body: &mut Map<String, Value>,
    degradations: &mut Degradations,
    field: &str,
    value: f64,
    rule: Option<SamplingRule>,
) {
    match rule {
        Some(SamplingRule::Range(_)) if rule.unwrap().accepts(value) => {
            if field == "top_k" {
                body.insert(field.into(), Value::from(value as u64));
            } else {
                body.insert(field.into(), Value::from(value));
            }
        }
        _ => {
            degradations.dropped(field, "value out of accepted range / not accepted");
        }
    }
}

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------

fn build_meta(req: &IrRequest, caps: &Capabilities, body: &mut Map<String, Value>, degradations: &mut Degradations) {
    // The Anthropic Messages `metadata` object only accepts `user_id`; any other cross-provider
    // metadata keys are not part of the wire shape and are dropped with a Degradation.
    if let Some(user) = &req.meta.user {
        // Anthropic rejects `metadata.user_id` longer than 256 chars: replace over-long values
        // with a deterministic digest and record the rewrite.
        use llm_xlate_core::canon::identifier;
        let value = match identifier::fit(user, identifier::ANTHROPIC_MAX) {
            Some(replacement) => {
                degradations.push(llm_xlate_core::degrade::Degradation {
                    kind: llm_xlate_core::degrade::DegradationKind::Rewritten,
                    field: "user".to_string(),
                    detail: format!("exceeded {} chars; replaced by sha256 digest", identifier::ANTHROPIC_MAX),
                });
                replacement
            }
            None => user.clone(),
        };
        let mut md = omap();
        md.insert("user_id".into(), Value::from(value));
        body.insert("metadata".into(), Value::Object(md));
    }
    if !req.meta.metadata.is_empty() {
        degradations.dropped("metadata", "Anthropic metadata only supports user_id");
    }

    if let Some(tier) = &req.meta.service_tier {
        let allowed = caps.sampling.service_tier.clone().unwrap_or_default();
        let mapped = match tier.as_str() {
            "auto" | "standard_only" => Some(tier.clone()),
            "default" => Some("auto".to_string()),
            _ => None,
        };
        match mapped {
            Some(m) if allowed.is_empty() || allowed.contains(&m) => {
                body.insert("service_tier".into(), Value::from(m));
            }
            _ => degradations.dropped("service_tier", "service tier not supported; dropped"),
        }
    }
}

// ---------------------------------------------------------------------------
// ext writeback / headers / cache breakpoints
// ---------------------------------------------------------------------------

fn write_back_ext(req: &IrRequest, body: &mut Map<String, Value>, _betas: &mut BetaFlags) {
    for (k, v) in req.ext.iter() {
        let field = match k.strip_prefix(EXT_PREFIX) {
            Some(f) => f,
            None => continue, // foreign namespace ignored (lower() reports once)
        };
        if field == "betas" {
            continue; // becomes a header
        }
        if field == SYSTEM_HEADERS_EXT {
            continue; // re-emitted into `system` by build_system
        }
        if field.starts_with("output_config.") {
            continue; // merged into output_config already
        }
        if field.contains('.') {
            continue; // nested namespaced keys handled elsewhere / skipped
        }
        body.entry(field.to_string()).or_insert_with(|| v.clone());
    }
}

fn build_headers(req: &IrRequest, caps: &Capabilities, betas: &BetaFlags) -> HeaderMap {
    use http::header::{HeaderName, HeaderValue};
    let mut headers = HeaderMap::new();

    let version = caps.transport.api_version.as_deref().unwrap_or(DEFAULT_API_VERSION);
    if let Ok(v) = HeaderValue::from_str(version) {
        headers.insert(HeaderName::from_static("anthropic-version"), v);
    }

    // Collect beta tokens: from used features (looked up in caps.transport.beta_headers) plus
    // any client-supplied betas.
    let mut tokens: Vec<String> = Vec::new();
    let bh = caps.transport.beta_headers.clone().unwrap_or_default();
    if betas.structured_output {
        if let Some(v) = bh.get("structured_output") {
            tokens.push(v.clone());
        }
    }
    if betas.mid_conversation_effort {
        if let Some(v) = bh.get("mid_conversation_effort") {
            tokens.push(v.clone());
        }
    }
    if betas.system_clear_at {
        if let Some(v) = bh.get("system_clear_at") {
            tokens.push(v.clone());
        }
    }
    if let Some(Value::Array(client)) = req.ext.get(&format!("{EXT_PREFIX}betas")) {
        for b in client {
            if let Some(s) = b.as_str() {
                tokens.push(s.to_string());
            }
        }
    }
    tokens.sort();
    tokens.dedup();
    if !tokens.is_empty() {
        if let Ok(v) = HeaderValue::from_str(&tokens.join(",")) {
            headers.insert(HeaderName::from_static("anthropic-beta"), v);
        }
    }

    headers
}

/// Cache-control JSON for a breakpoint.
pub(crate) fn cache_control_json(cc: &CacheControl) -> Value {
    let mut m = omap();
    m.insert("type".into(), Value::from("ephemeral"));
    if cc.ttl == CacheTtl::OneHour {
        m.insert("ttl".into(), Value::from("1h"));
    }
    Value::Object(m)
}

/// Keep only the last 4 `cache_control` breakpoints in request order (system → tools →
/// message blocks, including nested tool_result blocks). Earlier ones are dropped + degraded.
fn enforce_cache_breakpoints(body: &mut Value, degradations: &mut Degradations) {
    // Count.
    let mut total = 0usize;
    for_each_block_mut(body, &mut |m| {
        if m.contains_key("cache_control") {
            total += 1;
        }
    });
    if total <= 4 {
        return;
    }
    let to_drop = total - 4;
    let mut dropped = 0usize;
    for_each_block_mut(body, &mut |m| {
        if m.contains_key("cache_control") && dropped < to_drop {
            m.remove("cache_control");
            dropped += 1;
        }
    });
    degradations.dropped("cache_control", "more than 4 breakpoints; earliest dropped");
}

/// Visit every candidate block object (system, tools, message content, nested tool_result
/// content) in request order.
fn for_each_block_mut(body: &mut Value, f: &mut impl FnMut(&mut Map<String, Value>)) {
    let obj = match body.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    if let Some(Value::Array(a)) = obj.get_mut("system") {
        for b in a.iter_mut() {
            if let Some(m) = b.as_object_mut() {
                f(m);
            }
        }
    }
    if let Some(Value::Array(a)) = obj.get_mut("tools") {
        for b in a.iter_mut() {
            if let Some(m) = b.as_object_mut() {
                f(m);
            }
        }
    }
    if let Some(Value::Array(msgs)) = obj.get_mut("messages") {
        for msg in msgs.iter_mut() {
            if let Some(Value::Array(content)) = msg.get_mut("content") {
                for b in content.iter_mut() {
                    if let Some(m) = b.as_object_mut() {
                        // Nested tool_result content first-level blocks, then the block itself.
                        if let Some(Value::Array(inner)) = m.get_mut("content") {
                            for ib in inner.iter_mut() {
                                if let Some(im) = ib.as_object_mut() {
                                    f(im);
                                }
                            }
                        }
                        f(m);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared with response.rs / stream_enc.rs
// ---------------------------------------------------------------------------

/// Convert a response's (all assistant-side) items into ordered Anthropic content blocks:
/// thinking/redacted first, then text/refusal, then tool_use, then provider items.
pub(crate) fn assistant_items_to_blocks(
    items: &[Item],
    ctx: &EncodeCtx,
    caps: &Capabilities,
) -> Vec<Value> {
    let mut degradations = Degradations::new();
    let mut blocks = Vec::new();
    for item in items {
        if let Item::Reasoning(r) = item {
            if let Some(b) = build_reasoning_block(r, ctx, true, &mut degradations) {
                blocks.push(b);
            }
        }
    }
    for item in items {
        if let Item::Message { role: Role::Assistant, content, .. } = item {
            blocks.extend(build_parts(content, caps, &mut degradations));
        }
    }
    for item in items {
        if let Item::ToolCall { call_id, name, arguments, .. } = item {
            if let Ok(b) = build_tool_use(call_id.as_str(), name, arguments) {
                blocks.push(b);
            }
        }
    }
    for item in items {
        match item {
            Item::ProviderToolCall(oi) | Item::ProviderToolResult(oi) if oi.family == FAMILY => {
                blocks.push(oi.raw.clone());
            }
            Item::Compaction(blob) => {
                if let Ok(v) = serde_json::from_str::<Value>(&blob.data) {
                    blocks.push(v);
                }
            }
            _ => {}
        }
    }
    blocks
}

