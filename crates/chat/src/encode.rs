//! `encode_request`: [`IrRequest`] → Chat Completions wire bytes (provider-facing).
//!
//! This owns the *wire shape* of a Chat request, driven by `caps`: instruction placement
//! (developer→system fallback), assistant-run regrouping (plan §7.7 "→ Chat"), reasoning /
//! output / tools / media / sampling wire shapes, and the `max_completion_tokens` vs
//! `max_tokens` field name. Every lossy step records a [`Degradation`](llm_xlate_core::Degradation); nothing is dropped
//! silently. Cross-cutting policy (foreign reasoning/tool folding, schema-keyword
//! normalization, `n>1` rejection, effort snapping) is the facade's `lower()` — not repeated
//! here.

use serde_json::{Map, Value};

use llm_xlate_core::canon;
use llm_xlate_core::caps::{Capabilities, ReplayMode, SamplingRule, ToolChoiceKind};
use llm_xlate_core::codec::{EncodeCtx, EncodedRequest, HeaderMap};
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::ir::{
    CallId, Effort, Instruction, InstructionRole, IrRequest, Item, MediaSource, OutputFormat,
    Part, Position, ToolChoice, ToolDef, Verbosity,
};
use llm_xlate_core::{wrap, XlateError};

use crate::common::{is_openai_proper, is_reserved_ext, NS};

/// Encode an IR request into a Chat Completions request body.
pub fn encode_request(
    req: &IrRequest,
    caps: &Capabilities,
    ctx: &EncodeCtx,
) -> Result<EncodedRequest, XlateError> {
    let mut degr = Degradations::new();
    let mut out = Map::new();

    out.insert("model".into(), Value::from(req.model.upstream()));

    let messages = build_messages(req, caps, &mut degr)?;
    out.insert("messages".into(), Value::Array(messages));

    // ---- tools / tool_choice (legacy or modern) ----
    let legacy = req.ext.get(&format!("{NS}legacy_functions")).and_then(Value::as_bool)
        == Some(true);
    encode_tools(req, caps, &mut degr, &mut out, legacy)?;

    if let Some(p) = req.parallel_tool_calls {
        if caps.tools.parallel_control.is_yes() {
            out.insert("parallel_tool_calls".into(), Value::from(p));
        } else if !req.tools.is_empty() {
            degr.dropped("parallel_tool_calls", "target does not support parallel_tool_calls control");
        }
    }

    // ---- reasoning_effort ----
    encode_reasoning(req, caps, &mut degr, &mut out);

    // ---- response_format / verbosity ----
    encode_output(req, caps, &mut degr, &mut out);

    // ---- sampling ----
    encode_sampling(req, caps, &mut degr, &mut out);

    // ---- limits ----
    if let Some(max) = req.limits.max_output_tokens {
        let field = max_tokens_field(req, caps);
        out.insert(field.into(), Value::from(max));
    }
    if !req.limits.stop_sequences.is_empty() {
        let stops = &req.limits.stop_sequences;
        let val = if stops.len() == 1 {
            Value::from(stops[0].clone())
        } else {
            Value::Array(stops.iter().map(|s| Value::from(s.clone())).collect())
        };
        out.insert("stop".into(), val);
    }

    // ---- meta / state / cache ----
    encode_meta(req, caps, &mut degr, &mut out);

    // ---- streaming ----
    let upstream_streams = req.stream && caps.streaming() != llm_xlate_core::caps::Streaming::NonStreamOnly;
    let mut include_usage = false;
    if upstream_streams {
        out.insert("stream".into(), Value::Bool(true));
        if caps.transport.stream_usage_opt_in.is_yes() {
            let mut so = Map::new();
            so.insert("include_usage".into(), Value::Bool(true));
            out.insert("stream_options".into(), Value::Object(so));
            include_usage = true;
        }
    }

    // ---- ext passthrough (chat.* non-reserved → top-level) ----
    for (k, v) in req.ext.iter() {
        if let Some(field) = k.strip_prefix(NS) {
            if is_reserved_ext(k) {
                continue;
            }
            out.entry(field.to_string()).or_insert_with(|| v.clone());
        }
        // Foreign namespaces are ignored silently (lower() reports them once).
    }

    // ---- session affinity ----
    let mut headers = HeaderMap::new();
    llm_xlate_core::session::apply_session(
        caps,
        req.session.id.as_deref(),
        req.cache.prompt_cache_key.as_deref(),
        &mut out,
        &mut headers,
        &mut degr,
    );

    let body = canon::to_bytes(&Value::Object(out));
    let mut ectx = ctx.clone();
    ectx.include_usage = include_usage;

    Ok(EncodedRequest {
        body,
        headers,
        upstream_streams,
        ctx: ectx,
        degradations: degr,
    })
}

/// Pick the max-output-tokens field name. An explicit `chat.max_tokens_field` marker (from a
/// Chat client) wins for a byte-stable Chat→Chat round trip; otherwise proper OpenAI backends
/// use `max_completion_tokens` and generic openai-compatible servers use `max_tokens`.
fn max_tokens_field(req: &IrRequest, caps: &Capabilities) -> &'static str {
    match req.ext.get(&format!("{NS}max_tokens_field")).and_then(Value::as_str) {
        Some("max_tokens") => "max_tokens",
        Some("max_completion_tokens") => "max_completion_tokens",
        _ => {
            if is_openai_proper(caps) {
                "max_completion_tokens"
            } else {
                "max_tokens"
            }
        }
    }
}

// ------------------------------------------------------------------ messages

fn build_messages(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Result<Vec<Value>, XlateError> {
    let mut messages = Vec::new();

    // Leading instructions first.
    for instr in &req.instructions {
        if instr.position == Position::Leading {
            messages.push(encode_instruction(instr, caps, degr));
        }
    }

    let mut run: Vec<&Item> = Vec::new();
    // Folded tool-result media, buffered across a contiguous run of `tool` messages and
    // flushed as one `user` message AFTER the run ends — a `user` message may not interrupt
    // the block of `tool` messages that answers an assistant `tool_calls` turn (strict
    // openai-compatible servers reject that).
    let mut tool_attachments: Vec<Value> = Vec::new();
    for (i, item) in req.items.iter().enumerate() {
        // Emit any Before(i) instructions, breaking the current assistant / tool run.
        let has_before = req
            .instructions
            .iter()
            .any(|ins| ins.position == Position::Before(i));
        if has_before {
            flush_run(&mut run, req, caps, degr, &mut messages)?;
            flush_tool_attachments(&mut tool_attachments, &mut messages);
            for instr in &req.instructions {
                if instr.position == Position::Before(i) {
                    messages.push(encode_instruction(instr, caps, degr));
                }
            }
        }

        if item.is_assistant_side() {
            // An assistant item ends any pending tool-result block.
            flush_tool_attachments(&mut tool_attachments, &mut messages);
            run.push(item);
        } else {
            flush_run(&mut run, req, caps, degr, &mut messages)?;
            match item {
                Item::ToolResult { .. } => {
                    emit_tool_result(item, req, caps, degr, &mut messages, &mut tool_attachments)?;
                }
                _ => {
                    // A user message ends any pending tool-result block.
                    flush_tool_attachments(&mut tool_attachments, &mut messages);
                    emit_user_side(item, req, caps, degr, &mut messages)?;
                }
            }
        }
    }
    flush_run(&mut run, req, caps, degr, &mut messages)?;
    flush_tool_attachments(&mut tool_attachments, &mut messages);

    // Trailing instructions: anchored at the end, or past it. An out-of-range anchor can only
    // reach a codec on a hand-built request (`lower` clamps and reports one), but it is placed
    // here rather than dropped so no instruction can vanish from the wire.
    let len = req.items.len();
    for instr in &req.instructions {
        if matches!(instr.position, Position::Before(i) if i >= len) {
            messages.push(encode_instruction(instr, caps, degr));
        }
    }

    Ok(messages)
}

/// Encode a system/developer instruction as a Chat message.
fn encode_instruction(instr: &Instruction, caps: &Capabilities, degr: &mut Degradations) -> Value {
    let mut obj = Map::new();
    let role = match instr.role {
        InstructionRole::System => "system",
        InstructionRole::Developer => {
            if caps.instructions.developer_role.is_yes() {
                "developer"
            } else {
                degr.downgraded("instructions.developer", "developer role unsupported; sent as system");
                "system"
            }
        }
    };
    obj.insert("role".into(), Value::from(role));
    obj.insert("content".into(), instruction_content(instr));
    if instr.effort.is_some() {
        degr.dropped("instructions.effort", "mid-conversation effort override unsupported on Chat");
    }
    if instr.clear_at.is_some() {
        degr.dropped("instructions.clear_at", "system_clear_at unsupported on Chat");
    }
    Value::Object(obj)
}

/// Build instruction content: a string for a single text part, else an array of text blocks.
fn instruction_content(instr: &Instruction) -> Value {
    let texts: Vec<&str> = instr
        .content
        .iter()
        .filter_map(Part::as_text)
        .collect();
    if texts.len() == 1 {
        Value::from(texts[0])
    } else if texts.is_empty() {
        Value::from("")
    } else {
        Value::Array(
            texts
                .iter()
                .map(|t| {
                    let mut m = Map::new();
                    m.insert("type".into(), Value::from("text"));
                    m.insert("text".into(), Value::from(*t));
                    Value::Object(m)
                })
                .collect(),
        )
    }
}

/// Flush a buffered assistant-side run into a single assistant message.
fn flush_run(
    run: &mut Vec<&Item>,
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    messages: &mut Vec<Value>,
) -> Result<(), XlateError> {
    if run.is_empty() {
        return Ok(());
    }
    let items = std::mem::take(run);
    let first_index = item_index(req, items[0]);

    let mut text_parts: Vec<String> = Vec::new();
    let mut refusal: Option<String> = None;
    // All reasoning-item texts in the run, kept in order and joined on encode — a run may hold
    // more than one Reasoning item when lowering Anthropic/Responses-origin IR (multiple
    // thinking blocks) to a single Chat assistant message; none is dropped silently (§11.7).
    let mut reasoning_texts: Vec<String> = Vec::new();
    let mut has_opaque = false;
    let mut tool_calls: Vec<Value> = Vec::new();

    for item in &items {
        match item {
            Item::Message { content, .. } => {
                for p in content {
                    match p {
                        Part::Text { text, .. } => text_parts.push(text.clone()),
                        Part::Refusal { text } => refusal = Some(text.clone()),
                        _ => {}
                    }
                }
            }
            Item::Reasoning(ri) => {
                if let Some(t) = ri
                    .text
                    .clone()
                    .or_else(|| if ri.summary.is_empty() { None } else { Some(ri.summary.join("\n\n")) })
                {
                    reasoning_texts.push(t);
                }
                if ri.opaque.is_some() {
                    has_opaque = true;
                }
            }
            Item::ToolCall { call_id, name, arguments, .. } => {
                let mut func = Map::new();
                func.insert("name".into(), Value::from(name.clone()));
                func.insert("arguments".into(), Value::from(arguments.as_str()));
                let mut tc = Map::new();
                tc.insert("id".into(), Value::from(call_id.as_str()));
                tc.insert("type".into(), Value::from("function"));
                tc.insert("function".into(), Value::Object(func));
                tool_calls.push(Value::Object(tc));
            }
            Item::ProviderToolCall(_) | Item::ProviderToolResult(_) => {
                degr.dropped("provider_tool", "provider-hosted tool item has no Chat carrier");
            }
            Item::Compaction(_) => {
                degr.dropped("compaction", "compaction block has no Chat carrier");
            }
            Item::ToolResult { .. } => { /* not assistant-side */ }
        }
    }

    let legacy_fc = req.ext.get(&format!("{NS}legacy_function_call")).and_then(Value::as_bool)
        == Some(true);

    let mut obj = Map::new();
    obj.insert("role".into(), Value::from("assistant"));

    // content
    if text_parts.len() == 1 {
        obj.insert("content".into(), Value::from(text_parts[0].clone()));
    } else if text_parts.is_empty() {
        obj.insert("content".into(), Value::Null);
    } else {
        obj.insert(
            "content".into(),
            Value::Array(
                text_parts
                    .iter()
                    .map(|t| {
                        let mut m = Map::new();
                        m.insert("type".into(), Value::from("text"));
                        m.insert("text".into(), Value::from(t.clone()));
                        Value::Object(m)
                    })
                    .collect(),
            ),
        );
    }

    if let Some(r) = refusal {
        obj.insert("refusal".into(), Value::from(r));
    }

    // reasoning_content only when the target replays reasoning via a text field. Multiple
    // reasoning items in the run are joined so none is silently dropped.
    if !reasoning_texts.is_empty() {
        if caps.reasoning.replay == Some(ReplayMode::TextField) {
            obj.insert("reasoning_content".into(), Value::from(reasoning_texts.join("\n\n")));
        } else {
            degr.dropped("reasoning.text", "target has no reasoning text replay slot");
        }
    }
    if has_opaque {
        degr.dropped("reasoning.opaque", "opaque reasoning cannot be replayed to a Chat provider");
    }

    // tool calls (or legacy function_call)
    if !tool_calls.is_empty() {
        if legacy_fc {
            if let Some(Value::Object(first)) = tool_calls.first() {
                if let Some(func) = first.get("function") {
                    obj.insert("function_call".into(), func.clone());
                }
            }
        } else {
            obj.insert("tool_calls".into(), Value::Array(tool_calls));
        }
    }

    splice_extras(&mut obj, req, &format!("{NS}msg_ext"), first_index);
    messages.push(Value::Object(obj));
    Ok(())
}

/// Emit a user-side `Message{User}` item as one message.
fn emit_user_side(
    item: &Item,
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    messages: &mut Vec<Value>,
) -> Result<(), XlateError> {
    let index = item_index(req, item);
    if let Item::Message { content, .. } = item {
        let mut obj = Map::new();
        obj.insert("role".into(), Value::from("user"));
        obj.insert("content".into(), encode_user_content(content, req, caps, index, degr)?);
        splice_extras(&mut obj, req, &format!("{NS}msg_ext"), index);
        messages.push(Value::Object(obj));
    }
    Ok(())
}

/// The name of the `ToolCall` that `call_id` answers, if it is present in the transcript.
fn tool_call_name<'a>(req: &'a IrRequest, call_id: &CallId) -> Option<&'a str> {
    req.items.iter().find_map(|it| match it {
        Item::ToolCall { call_id: c, name, .. } if c == call_id => Some(name.as_str()),
        _ => None,
    })
}

/// Emit a `ToolResult` as a `tool` message; any non-text (media) parts are folded and their
/// content parts buffered into `tool_attachments` for a single trailing `user` message so the
/// block of `tool` messages stays contiguous after the assistant `tool_calls` turn.
fn emit_tool_result(
    item: &Item,
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    messages: &mut Vec<Value>,
    tool_attachments: &mut Vec<Value>,
) -> Result<(), XlateError> {
    let Item::ToolResult { call_id, content, .. } = item else {
        return Ok(());
    };
    let index = item_index(req, item);
    let mut text = String::new();
    let mut attachments: Vec<Part> = Vec::new();
    for p in content {
        match p {
            Part::Text { text: t, .. } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            other => attachments.push(other.clone()),
        }
    }
    let mut obj = Map::new();
    obj.insert("role".into(), Value::from("tool"));
    obj.insert("tool_call_id".into(), Value::from(call_id.as_str()));
    // Name the tool explicitly where the backend wants it, so the result does not have to be
    // matched to its call by position (`tools.result_name`). The name is recovered from the
    // `ToolCall` this result answers; if that call is not in the window there is nothing
    // truthful to emit, so the key is omitted rather than guessed.
    if caps.tools.result_name.is_yes() {
        if let Some(name) = tool_call_name(req, call_id) {
            obj.insert("name".into(), Value::from(name));
        }
    }
    obj.insert("content".into(), Value::from(text));
    splice_extras(&mut obj, req, &format!("{NS}msg_ext"), index);
    messages.push(Value::Object(obj));

    if !attachments.is_empty() {
        degr.folded("tool_result", "non-text tool result parts folded into a following user message");
        tool_attachments.push(text_part(&wrap::tool_result_attachment(call_id.as_str())));
        for (pi, a) in attachments.iter().enumerate() {
            if let Some(v) = encode_media_part(a, req, caps, index, pi, degr)? {
                tool_attachments.push(v);
            }
        }
    }
    Ok(())
}

/// Flush any buffered folded tool-result media into a single trailing `user` message.
fn flush_tool_attachments(tool_attachments: &mut Vec<Value>, messages: &mut Vec<Value>) {
    if tool_attachments.is_empty() {
        return;
    }
    let parts = std::mem::take(tool_attachments);
    let mut um = Map::new();
    um.insert("role".into(), Value::from("user"));
    um.insert("content".into(), Value::Array(parts));
    messages.push(Value::Object(um));
}

/// Encode user message content into a string (single text) or an array of content parts.
fn encode_user_content(
    content: &[Part],
    req: &IrRequest,
    caps: &Capabilities,
    item_index: usize,
    degr: &mut Degradations,
) -> Result<Value, XlateError> {
    // A single text part becomes a bare string.
    if content.len() == 1 {
        if let Part::Text { text, .. } = &content[0] {
            return Ok(Value::from(text.clone()));
        }
    }
    let mut parts = Vec::new();
    for (part_index, p) in content.iter().enumerate() {
        if let Some(v) = encode_media_part(p, req, caps, item_index, part_index, degr)? {
            parts.push(v);
        }
    }
    Ok(Value::Array(parts))
}

/// Encode one content part into a Chat content-part object (plan §7.5 media, Chat column).
fn encode_media_part(
    p: &Part,
    req: &IrRequest,
    caps: &Capabilities,
    item_index: usize,
    part_index: usize,
    degr: &mut Degradations,
) -> Result<Option<Value>, XlateError> {
    match p {
        Part::Text { text, .. } => Ok(Some(text_part(text))),
        Part::Refusal { text } => {
            let mut m = Map::new();
            m.insert("type".into(), Value::from("refusal"));
            m.insert("refusal".into(), Value::from(text.clone()));
            Ok(Some(Value::Object(m)))
        }
        Part::Image(src) => encode_image(src, req, item_index, part_index, degr),
        Part::Document { source, title, media_type } => {
            encode_document(source, title.as_deref(), media_type)
        }
        Part::Audio(src) => encode_audio(src, caps),
        Part::Opaque(_) => {
            degr.dropped("content.opaque", "opaque content part has no Chat carrier");
            Ok(None)
        }
    }
}

fn encode_image(
    src: &MediaSource,
    req: &IrRequest,
    item_index: usize,
    part_index: usize,
    degr: &mut Degradations,
) -> Result<Option<Value>, XlateError> {
    let url = match src {
        MediaSource::Base64 { media_type, data } => canon::data_url::build(media_type, data),
        MediaSource::Url(u) => u.clone(),
        MediaSource::FileRef { .. } => {
            degr.dropped("image.file_ref", "Chat image_url has no file-id form; dropped");
            return Ok(None);
        }
        MediaSource::Text(_) => {
            degr.dropped("image.text", "text is not a valid image source");
            return Ok(None);
        }
    };
    let mut iu = Map::new();
    iu.insert("url".into(), Value::from(url));
    // Re-apply a preserved non-"auto" image detail, keyed exactly by "<item>/<part>".
    if let Some(detail) = lookup_image_detail(req, item_index, part_index) {
        iu.insert("detail".into(), Value::from(detail));
    }
    let mut m = Map::new();
    m.insert("type".into(), Value::from("image_url"));
    m.insert("image_url".into(), Value::Object(iu));
    Ok(Some(Value::Object(m)))
}

fn encode_document(
    source: &MediaSource,
    title: Option<&str>,
    _media_type: &str,
) -> Result<Option<Value>, XlateError> {
    match source {
        MediaSource::Base64 { media_type, data } => {
            let mut file = Map::new();
            file.insert("file_data".into(), Value::from(canon::data_url::build(media_type, data)));
            // OpenAI requires `filename` alongside `file_data` (it infers the type from the
            // extension) and 400s without it; a cross-family (e.g. Anthropic) PDF often has no
            // title, so synthesize a default name from the media type when none was supplied.
            let name = title
                .map(str::to_string)
                .unwrap_or_else(|| canon::filename::for_media_type(media_type));
            file.insert("filename".into(), Value::from(name));
            let mut m = Map::new();
            m.insert("type".into(), Value::from("file"));
            m.insert("file".into(), Value::Object(file));
            Ok(Some(Value::Object(m)))
        }
        MediaSource::FileRef { id, .. } => {
            let mut file = Map::new();
            file.insert("file_id".into(), Value::from(id.clone()));
            if let Some(t) = title {
                file.insert("filename".into(), Value::from(t));
            }
            let mut m = Map::new();
            m.insert("type".into(), Value::from("file"));
            m.insert("file".into(), Value::Object(file));
            Ok(Some(Value::Object(m)))
        }
        MediaSource::Text(text) => Ok(Some(text_part(&wrap::document(title, text)))),
        MediaSource::Url(_) => Err(XlateError::unsupported(
            "document.url",
            "Chat cannot fetch a document by URL",
        )),
    }
}

fn encode_audio(src: &MediaSource, caps: &Capabilities) -> Result<Option<Value>, XlateError> {
    if caps.media.audio.sources.as_ref().is_none_or(|s| s.is_empty()) {
        return Err(XlateError::unsupported("audio", "target does not accept audio input"));
    }
    match src {
        MediaSource::Base64 { media_type, data } => {
            let format = media_type.strip_prefix("audio/").unwrap_or("wav").to_string();
            let mut ia = Map::new();
            ia.insert(
                "data".into(),
                Value::from(base64::engine::general_purpose::STANDARD.encode(data)),
            );
            ia.insert("format".into(), Value::from(format));
            let mut m = Map::new();
            m.insert("type".into(), Value::from("input_audio"));
            m.insert("input_audio".into(), Value::Object(ia));
            Ok(Some(Value::Object(m)))
        }
        _ => Err(XlateError::unsupported("audio", "only inline base64 audio is supported")),
    }
}

use base64::Engine as _;

/// A `{type:"text", text}` content-part object.
fn text_part(text: &str) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), Value::from("text"));
    m.insert("text".into(), Value::from(text));
    Value::Object(m)
}

/// The index of an item within `req.items` (by pointer identity via position).
fn item_index(req: &IrRequest, item: &Item) -> usize {
    req.items
        .iter()
        .position(|it| std::ptr::eq(it, item))
        .unwrap_or(0)
}

/// Splice preserved message/instruction extras (`name`, unknown fields) into `obj`.
fn splice_extras(obj: &mut Map<String, Value>, req: &IrRequest, table_key: &str, index: usize) {
    if let Some(Value::Object(table)) = req.ext.get(table_key) {
        if let Some(Value::Object(extras)) = table.get(&index.to_string()) {
            for (k, v) in extras {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
}

/// Look up a preserved non-"auto" image detail for the exact `"<item>/<part>"` key.
fn lookup_image_detail(req: &IrRequest, item_index: usize, part_index: usize) -> Option<String> {
    let table = req.ext.get(&format!("{NS}image_detail"))?.as_object()?;
    table
        .get(&format!("{item_index}/{part_index}"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

// ------------------------------------------------------------------ tools

fn encode_tools(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    out: &mut Map<String, Value>,
    legacy: bool,
) -> Result<(), XlateError> {
    if req.tools.is_empty() {
        // Still emit tool_choice? OpenAI ignores tool_choice without tools; omit it.
        return Ok(());
    }
    if caps.tools.function_tools.is_no() {
        return Err(XlateError::unsupported("tools", "target does not support function tools"));
    }

    if legacy {
        let mut functions = Vec::new();
        for tool in &req.tools {
            if let ToolDef::Function { name, description, parameters, .. } = tool {
                let mut f = Map::new();
                f.insert("name".into(), Value::from(name.clone()));
                if let Some(d) = description {
                    f.insert("description".into(), Value::from(d.clone()));
                }
                f.insert("parameters".into(), parameters.clone());
                functions.push(Value::Object(f));
            }
        }
        out.insert("functions".into(), Value::Array(functions));
        // legacy function_call (tool_choice equivalent)
        match &req.tool_choice {
            ToolChoice::None => {
                out.insert("function_call".into(), Value::from("none"));
            }
            ToolChoice::Named(n) => {
                let mut m = Map::new();
                m.insert("name".into(), Value::from(n.clone()));
                out.insert("function_call".into(), Value::Object(m));
            }
            ToolChoice::Required => {
                degr.downgraded("tool_choice", "legacy functions have no `required`; sent as auto");
            }
            ToolChoice::Auto => {}
        }
        return Ok(());
    }

    let mut tools = Vec::new();
    for tool in &req.tools {
        match tool {
            ToolDef::Function { name, description, parameters, strict, .. } => {
                let mut func = Map::new();
                func.insert("name".into(), Value::from(name.clone()));
                if let Some(d) = description {
                    func.insert("description".into(), Value::from(d.clone()));
                }
                func.insert("parameters".into(), parameters.clone());
                if let Some(s) = strict {
                    if caps.tools.strict.supported.is_yes() {
                        func.insert("strict".into(), Value::from(*s));
                    } else if *s {
                        degr.downgraded("tools.strict", "strict schema unsupported; dropped");
                    }
                }
                let mut m = Map::new();
                m.insert("type".into(), Value::from("function"));
                m.insert("function".into(), Value::Object(func));
                tools.push(Value::Object(m));
            }
            ToolDef::Provider(item) => {
                // The Chat Completions request schema has **no carrier** for a provider-hosted
                // tool: every `tools[]` entry must be `{"type":"function","function":{…}}`.
                // Family equality is not enough — OpenAI Chat and OpenAI Responses share a
                // family but only Responses can carry `web_search` / `namespace` /
                // `code_interpreter` / `file_search` / … . Emitting the hosted entry raw is
                // always rejected by the backend (live-verified 2026-09-10: a Codex
                // `{"type":"namespace"}` tool through a vLLM chat backend → 400
                // "Input should be 'function'" at `body.tools.7.type`). Drop it with a
                // degradation, mirroring the item-side `provider_tool` rule above.
                degr.dropped(
                    "tools.provider",
                    format!(
                        "provider-hosted tool ({}) has no Chat carrier; dropped",
                        provider_tool_label(&item.raw)
                    ),
                );
            }
        }
    }
    // Every tool was a provider-hosted tool we had to drop: omit `tools` entirely rather than
    // sending `"tools": []`, which some backends reject.
    if tools.is_empty() {
        return Ok(());
    }
    out.insert("tools".into(), Value::Array(tools));

    encode_tool_choice(req, caps, degr, out);
    Ok(())
}

/// A short human label for a provider-hosted tool in a degradation message: its `type`, refined
/// with `name` when the raw entry carries one (e.g. `namespace multi_agent_v1`).
fn provider_tool_label(raw: &Value) -> String {
    let ty = raw.get("type").and_then(Value::as_str).unwrap_or("provider_tool");
    match raw.get("name").and_then(Value::as_str) {
        Some(name) => format!("{ty} {name}"),
        None => ty.to_string(),
    }
}

fn encode_tool_choice(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    out: &mut Map<String, Value>,
) {
    let allowed = |k: ToolChoiceKind| -> bool {
        caps.tools
            .tool_choice
            .as_ref()
            .map(|list| list.contains(&k))
            .unwrap_or(true) // unknown list ⇒ best effort
    };
    match &req.tool_choice {
        ToolChoice::Auto => {}
        ToolChoice::None => {
            if allowed(ToolChoiceKind::None) {
                out.insert("tool_choice".into(), Value::from("none"));
            } else {
                degr.downgraded("tool_choice", "`none` unsupported; left as auto");
            }
        }
        ToolChoice::Required => {
            if allowed(ToolChoiceKind::Required) {
                out.insert("tool_choice".into(), Value::from("required"));
            } else {
                degr.downgraded("tool_choice", "`required` unsupported; left as auto");
            }
        }
        ToolChoice::Named(n) => {
            if allowed(ToolChoiceKind::Named) {
                let mut func = Map::new();
                func.insert("name".into(), Value::from(n.clone()));
                let mut m = Map::new();
                m.insert("type".into(), Value::from("function"));
                m.insert("function".into(), Value::Object(func));
                out.insert("tool_choice".into(), Value::Object(m));
            } else {
                degr.downgraded("tool_choice", "named tool choice unsupported; left as auto");
            }
        }
    }
}

// ------------------------------------------------------------------ reasoning / output

fn encode_reasoning(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    out: &mut Map<String, Value>,
) {
    let Some(effort) = req.reasoning.effort else {
        return; // no reasoning config ⇒ emit nothing
    };
    use llm_xlate_core::caps::ReasoningMode;
    if matches!(caps.reasoning.mode, Some(ReasoningMode::None) | None) {
        degr.dropped("reasoning_effort", "target has no reasoning mode");
        return;
    }
    // GPT-5.4 Chat: tools + reasoning are mutually exclusive.
    if effort != Effort::None
        && !req.tools.is_empty()
        && caps.reasoning.tools_with_reasoning.is_no()
    {
        degr.dropped("reasoning_effort", "target cannot use tools with reasoning; effort dropped");
        return;
    }
    let token = if effort == Effort::Max {
        let supports_xhigh = caps
            .reasoning
            .effort_levels
            .as_ref()
            .map(|l| l.contains(&Effort::XHigh))
            .unwrap_or(true);
        if supports_xhigh {
            degr.downgraded("reasoning_effort", "Max downgraded to xhigh for Chat");
            "xhigh"
        } else {
            degr.downgraded("reasoning_effort", "Max downgraded to highest supported effort");
            highest_effort_token(caps)
        }
    } else {
        effort.openai_token()
    };
    out.insert("reasoning_effort".into(), Value::from(token));
}

/// The highest effort token this backend lists (fallback `high`).
fn highest_effort_token(caps: &Capabilities) -> &'static str {
    caps.reasoning
        .effort_levels
        .as_ref()
        .and_then(|l| l.iter().max().copied())
        .map(|e| e.openai_token())
        .unwrap_or("high")
}

fn encode_output(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    out: &mut Map<String, Value>,
) {
    match &req.output.format {
        OutputFormat::Text => {}
        OutputFormat::JsonObject => {
            let mut m = Map::new();
            m.insert("type".into(), Value::from("json_object"));
            out.insert("response_format".into(), Value::Object(m));
        }
        OutputFormat::JsonSchema { name, schema, strict, description } => {
            let mut js = Map::new();
            js.insert("name".into(), Value::from(name.clone()));
            if let Some(d) = description {
                js.insert("description".into(), Value::from(d.clone()));
            }
            js.insert("schema".into(), schema.clone());
            if *strict {
                if caps.output.strict_supported.is_yes() {
                    js.insert("strict".into(), Value::Bool(true));
                } else {
                    degr.downgraded("response_format.strict", "strict structured output unsupported; dropped");
                }
            } else {
                js.insert("strict".into(), Value::Bool(false));
            }
            let mut m = Map::new();
            m.insert("type".into(), Value::from("json_schema"));
            m.insert("json_schema".into(), Value::Object(js));
            out.insert("response_format".into(), Value::Object(m));
        }
    }

    if let Some(v) = req.output.verbosity {
        if caps.output.verbosity.is_yes() {
            out.insert("verbosity".into(), Value::from(verbosity_token(v)));
        } else {
            degr.dropped("verbosity", "target does not support verbosity");
        }
    }
}

fn verbosity_token(v: Verbosity) -> &'static str {
    match v {
        Verbosity::Low => "low",
        Verbosity::Medium => "medium",
        Verbosity::High => "high",
    }
}

// ------------------------------------------------------------------ sampling / meta

fn encode_sampling(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    out: &mut Map<String, Value>,
) {
    let s = &req.sampling;
    emit_ranged(out, degr, "temperature", s.temperature, caps.sampling.temperature);
    emit_ranged(out, degr, "top_p", s.top_p, caps.sampling.top_p);

    // Chat Completions has no `top_k` wire field, so any IR carrying it (e.g. from an
    // Anthropic-origin request) loses it. Never silent (plan §11.7).
    if s.top_k.is_some() {
        degr.dropped("top_k", "Chat has no top_k parameter");
    }

    emit_ranged(out, degr, "presence_penalty", s.presence_penalty, caps.sampling.presence_penalty);
    emit_ranged(out, degr, "frequency_penalty", s.frequency_penalty, caps.sampling.frequency_penalty);

    if let Some(seed) = s.seed {
        if rule_accepts(caps.sampling.seed, seed as f64) {
            out.insert("seed".into(), Value::from(seed));
        } else {
            degr.dropped("seed", "seed not accepted by target");
        }
    }
    if let Some(lb) = &s.logit_bias {
        if matches!(caps.sampling.logit_bias, Some(SamplingRule::Range(_))) {
            out.insert("logit_bias".into(), lb.clone());
        } else {
            degr.dropped("logit_bias", "logit_bias not accepted by target");
        }
    }
    if let Some(lp) = s.logprobs {
        if matches!(caps.sampling.logprobs, Some(SamplingRule::Range(_))) {
            out.insert("logprobs".into(), Value::from(lp));
            if let Some(tlp) = s.top_logprobs {
                if lp && rule_accepts(caps.sampling.logprobs, tlp as f64) {
                    out.insert("top_logprobs".into(), Value::from(tlp));
                }
            }
        } else {
            degr.dropped("logprobs", "logprobs not accepted by target");
        }
    }
    if let Some(n) = s.n {
        let max = caps.sampling.n.and_then(|r| r.max).unwrap_or(1);
        if n <= max {
            if n != 1 {
                out.insert("n".into(), Value::from(n));
            }
        } else {
            degr.dropped("n", "n exceeds target maximum");
        }
    }
}

fn emit_ranged(
    out: &mut Map<String, Value>,
    degr: &mut Degradations,
    field: &str,
    value: Option<f64>,
    rule: Option<SamplingRule>,
) {
    if let Some(v) = value {
        if rule_accepts(rule, v) {
            out.insert(field.into(), json_num(v));
        } else {
            degr.dropped(field, "value outside accepted range or parameter unsupported");
        }
    }
}

fn rule_accepts(rule: Option<SamplingRule>, value: f64) -> bool {
    matches!(rule, Some(SamplingRule::Range(_))) && rule.unwrap().accepts(value)
}

/// Preserve integral floats as integers (e.g. `1.0` → `1`) so canonical output is stable and
/// natural; non-integral values keep their JSON number form.
fn json_num(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 9.007e15 {
        Value::from(v as i64)
    } else {
        serde_json::Number::from_f64(v).map(Value::Number).unwrap_or(Value::Null)
    }
}

/// OpenAI rejects `user` / `safety_identifier` longer than 64 chars (Claude Code sends ~150-char
/// composite ids on the Anthropic surface). Replace over-long values with a deterministic digest
/// and record the rewrite.
fn fit_identifier(value: &str, field: &str, degr: &mut Degradations) -> String {
    use llm_xlate_core::canon::identifier;
    match identifier::fit(value, identifier::OPENAI_MAX) {
        Some(replacement) => {
            degr.push(llm_xlate_core::degrade::Degradation {
                kind: llm_xlate_core::degrade::DegradationKind::Rewritten,
                field: field.to_string(),
                detail: format!("exceeded {} chars; replaced by sha256 digest", identifier::OPENAI_MAX),
            });
            replacement
        }
        None => value.to_string(),
    }
}

fn encode_meta(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
    out: &mut Map<String, Value>,
) {
    if let Some(store) = req.state.store {
        if caps.state.store.is_yes() {
            out.insert("store".into(), Value::from(store));
        } else if store {
            degr.dropped("store", "target does not support store");
        }
    }
    if !req.meta.metadata.is_empty() {
        out.insert("metadata".into(), Value::Object(req.meta.metadata.clone()));
    }
    if let Some(u) = &req.meta.user {
        out.insert("user".into(), Value::from(fit_identifier(u, "user", degr)));
    }
    if let Some(sid) = &req.meta.safety_identifier {
        out.insert("safety_identifier".into(), Value::from(fit_identifier(sid, "safety_identifier", degr)));
    }
    if let Some(k) = &req.cache.prompt_cache_key {
        if caps.cache.prompt_cache_key.is_yes() {
            out.insert("prompt_cache_key".into(), Value::from(k.clone()));
        } else {
            degr.dropped("prompt_cache_key", "target does not support prompt_cache_key");
        }
    }
    if let Some(tier) = &req.meta.service_tier {
        let mapped = match tier.as_str() {
            "auto" => "auto",
            "standard_only" => "default",
            other => other,
        };
        let ok = caps
            .sampling
            .service_tier
            .as_ref()
            .map(|list| list.iter().any(|t| t == mapped))
            .unwrap_or(false);
        if ok {
            out.insert("service_tier".into(), Value::from(mapped));
        } else {
            degr.dropped("service_tier", "service tier not supported by target");
        }
    }
}
