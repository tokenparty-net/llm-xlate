//! `decode_request`: Anthropic Messages wire request → [`IrRequest`] (client-facing, lossless).
//!
//! Opaque reasoning carriers (thinking signatures, redacted-thinking data) that start with
//! `rtr1.` are opened with the client sealer; anything else is native Anthropic (see the
//! envelope boundary in `core::codec`).

use serde_json::{Map, Value};

use llm_xlate_core::{
    canon, CacheControl, CacheTtl, CallId, DecodeCtx, Effort, HeaderMap, Instruction,
    InstructionRole, IrRequest, Item, MediaSource, ModelRef, OpaqueBlob, OpaqueItem,
    OpaqueKind, OutputFormat, Part, Position, ReasoningExposure, Role, ToolChoice, ToolDef,
    XlateError,
};

use llm_xlate_core::session::capture_session;

use crate::wire::{
    ext_key, is_provider_tool_result_type, is_server_tool_use_type, system_header_name,
    tool_is_provider, AntMessageWire, AntRequestWire, FAMILY, SYSTEM_HEADERS_EXT,
};

/// Decode an Anthropic request body (+ headers) into the IR.
pub fn decode_request(
    body: &[u8],
    hdrs: &HeaderMap,
    ctx: &DecodeCtx,
) -> Result<IrRequest, XlateError> {
    let value = canon::parse(body)?;
    let wire: AntRequestWire = serde_json::from_value(value)
        .map_err(|e| XlateError::invalid_request(format!("invalid Anthropic request: {e}")))?;

    let max_tokens = wire
        .max_tokens
        .ok_or_else(|| XlateError::invalid_request("Anthropic request missing required `max_tokens`"))?;

    let mut req = IrRequest {
        model: ModelRef::new(wire.model.clone()),
        ..Default::default()
    };
    req.limits.max_output_tokens = Some(max_tokens);
    req.stream = wire.stream.unwrap_or(false);

    // Anthropic clients always see thinking text.
    req.reasoning.expose = ReasoningExposure::Full;

    // ---- top-level system ----
    // Client `x-anthropic-<name>:` header blocks are metadata, not prompt text: they are
    // captured into ext and re-emitted only for a backend that accepts them.
    if let Some(system) = &wire.system {
        let (instr, headers) = decode_top_level_system(system);
        if !instr.content.is_empty() {
            req.instructions.push(instr);
        }
        if !headers.is_empty() {
            req.ext.insert(ext_key(SYSTEM_HEADERS_EXT), Value::Array(headers));
        }
    }

    // ---- messages (also produce mid-context system instructions) ----
    decode_messages(&wire.messages, ctx, &mut req)?;

    // ---- tools ----
    if let Some(tools) = &wire.tools {
        for tool in tools {
            req.tools.push(decode_tool(tool));
        }
    }

    // ---- tool_choice ----
    if let Some(tc) = &wire.tool_choice {
        decode_tool_choice(tc, &mut req);
    }

    // ---- thinking + output_config ----
    decode_thinking_and_output(&wire, &mut req);

    // ---- sampling ----
    req.sampling.temperature = wire.temperature;
    req.sampling.top_p = wire.top_p;
    req.sampling.top_k = wire.top_k;
    if let Some(stop) = wire.stop_sequences {
        req.limits.stop_sequences = stop;
    }

    // ---- metadata / service tier ----
    // Anthropic `metadata` only carries `user_id` (→ meta.user); no other keys are modelled.
    if let Some(md) = &wire.metadata {
        if let Some(uid) = md.get("user_id").and_then(Value::as_str) {
            req.meta.user = Some(uid.to_string());
        }
    }
    req.meta.service_tier = wire.service_tier;

    // ---- request-level cache ----
    if wire.cache_control.is_some() {
        req.cache.request_level = true;
    }

    // ---- session affinity ----
    // Anthropic Messages has no `prompt_cache_key`; a non-standard `session_id` field (if any)
    // lands in `extra` and also round-trips via ext below. Headers supply the rest.
    let session_id_field = wire.extra.get("session_id").and_then(Value::as_str);
    req.session = capture_session(None, session_id_field, hdrs);

    // ---- opaque passthrough → ext ----
    if let Some(v) = wire.container {
        req.ext.insert(ext_key("container"), v);
    }
    if let Some(v) = wire.context_management {
        req.ext.insert(ext_key("context_management"), v);
    }
    if let Some(v) = wire.mcp_servers {
        req.ext.insert(ext_key("mcp_servers"), v);
    }
    for (k, v) in wire.extra {
        req.ext.insert(ext_key(&k), v);
    }

    // ---- client anthropic-beta header → ext["anthropic.betas"] ----
    if let Some(betas) = beta_header(hdrs) {
        req.ext.insert(
            ext_key("betas"),
            Value::Array(betas.into_iter().map(Value::from).collect()),
        );
    }

    Ok(req)
}

/// Parse the (possibly comma-joined, multi-valued) `anthropic-beta` header into a list.
fn beta_header(hdrs: &HeaderMap) -> Option<Vec<String>> {
    let mut out = Vec::new();
    for v in hdrs.get_all("anthropic-beta").iter() {
        if let Ok(s) = v.to_str() {
            for part in s.split(',') {
                let p = part.trim();
                if !p.is_empty() {
                    out.push(p.to_string());
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

/// Decode the top-level `system` (string or array of text blocks) into one leading
/// [`Instruction`], preserving `cache_control` on each block 1:1, plus the client header blocks
/// ([`system_header_name`]) lifted out of it, as `anthropic.system_headers` entries.
fn decode_top_level_system(system: &Value) -> (Instruction, Vec<Value>) {
    let mut content = Vec::new();
    let mut headers = Vec::new();
    match system {
        Value::String(s) => content.push(Part::text(s.clone())),
        Value::Array(blocks) => {
            for b in blocks {
                let text = b.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                if let Some(name) = system_header_name(&text) {
                    let mut h = Map::new();
                    h.insert("name".into(), Value::from(name));
                    h.insert("text".into(), Value::from(text.clone()));
                    if let Some(cc) = b.get("cache_control") {
                        h.insert("cache_control".into(), cc.clone());
                    }
                    headers.push(Value::Object(h));
                    continue;
                }
                let cc = decode_cache_control(b.get("cache_control"));
                content.push(Part::Text { text, annotations: Vec::new(), cache_control: cc });
            }
        }
        _ => {}
    }
    // Each block's `cache_control` round-trips on its own `Part`; the instruction-level slot is
    // left `None` so re-encoding does not fabricate an extra breakpoint on the trailing block
    // (a multi-block system with a cached preamble + uncached tail must not gain a marker).
    let instr = Instruction {
        role: InstructionRole::System,
        position: Position::Leading,
        content,
        cache_control: None,
        effort: None,
        clear_at: None,
    };
    (instr, headers)
}

/// Decode a `cache_control` value into an IR [`CacheControl`].
fn decode_cache_control(v: Option<&Value>) -> Option<CacheControl> {
    let obj = v?.as_object()?;
    let ttl = match obj.get("ttl").and_then(Value::as_str) {
        Some("1h") => CacheTtl::OneHour,
        _ => CacheTtl::FiveMinutes,
    };
    Some(CacheControl { ttl })
}

/// Walk the messages, flattening each into IR items (and mid-context system instructions).
fn decode_messages(
    messages: &[AntMessageWire],
    ctx: &DecodeCtx,
    req: &mut IrRequest,
) -> Result<(), XlateError> {
    for msg in messages {
        if msg.role == "system" {
            // In-array system message → Instruction before the next item.
            let idx = req.items.len();
            req.instructions.push(decode_in_array_system(&msg.content, idx));
            continue;
        }
        let role = if msg.role == "assistant" { Role::Assistant } else { Role::User };
        decode_message_blocks(role, &msg.content, ctx, req)?;
    }
    Ok(())
}

/// An in-array `role:"system"` message → `Instruction { position: Before(idx) }`.
fn decode_in_array_system(content: &Value, idx: usize) -> Instruction {
    let mut parts = Vec::new();
    match content {
        Value::String(s) => parts.push(Part::text(s.clone())),
        Value::Array(blocks) => {
            for b in blocks {
                let text = b.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                let cc = decode_cache_control(b.get("cache_control"));
                parts.push(Part::Text { text, annotations: Vec::new(), cache_control: cc });
            }
        }
        _ => {}
    }
    Instruction {
        role: InstructionRole::System,
        position: Position::Before(idx),
        content: parts,
        cache_control: None,
        effort: None,
        clear_at: None,
    }
}

/// Flatten one user/assistant message's content into IR items. Contiguous non-tool blocks
/// (text/image/document) coalesce into a single [`Item::Message`].
fn decode_message_blocks(
    role: Role,
    content: &Value,
    ctx: &DecodeCtx,
    req: &mut IrRequest,
) -> Result<(), XlateError> {
    let mut pending: Vec<Part> = Vec::new();
    let flush = |pending: &mut Vec<Part>, req: &mut IrRequest| {
        if !pending.is_empty() {
            req.items.push(Item::Message {
                role,
                content: std::mem::take(pending),
                id: None,
            });
        }
    };

    match content {
        Value::String(s) => {
            req.items.push(Item::Message { role, content: vec![Part::text(s.clone())], id: None });
        }
        Value::Array(blocks) => {
            for b in blocks {
                let btype = b.get("type").and_then(Value::as_str).unwrap_or("");
                match btype {
                    "text" => pending.push(decode_text_block(b)),
                    "image" => pending.push(Part::Image(decode_media_source(b.get("source")))),
                    "document" => pending.push(decode_document_block(b)),
                    "tool_use" => {
                        flush(&mut pending, req);
                        req.items.push(decode_tool_use(b));
                    }
                    "tool_result" => {
                        flush(&mut pending, req);
                        req.items.push(decode_tool_result(b));
                    }
                    "thinking" => {
                        flush(&mut pending, req);
                        req.items.push(decode_thinking_block(b, ctx));
                    }
                    "redacted_thinking" => {
                        flush(&mut pending, req);
                        req.items.push(decode_redacted_thinking(b, ctx));
                    }
                    "compaction" => {
                        flush(&mut pending, req);
                        // A compaction block may also be a router envelope carrying a foreign
                        // provider's compaction item; open it so its family survives replay.
                        let raw = b.to_string();
                        let blob = match b.get("data").and_then(Value::as_str) {
                            Some(data) if llm_xlate_core::envelope::Sealer::is_envelope(data) => {
                                ctx.sealer.open_or_native(data, FAMILY, OpaqueKind::Compaction)
                            }
                            _ => OpaqueBlob::new(FAMILY, OpaqueKind::Compaction, raw),
                        };
                        req.items.push(Item::Compaction(blob));
                    }
                    t if is_server_tool_use_type(t) => {
                        flush(&mut pending, req);
                        req.items.push(Item::ProviderToolCall(OpaqueItem::new(FAMILY, b.clone())));
                    }
                    t if is_provider_tool_result_type(t) => {
                        flush(&mut pending, req);
                        req.items
                            .push(Item::ProviderToolResult(OpaqueItem::new(FAMILY, b.clone())));
                    }
                    _ => {
                        // Unknown block: keep it verbatim as an opaque provider item so nothing
                        // is lost.
                        flush(&mut pending, req);
                        req.items.push(Item::ProviderToolCall(OpaqueItem::new(FAMILY, b.clone())));
                    }
                }
            }
            flush(&mut pending, req);
        }
        _ => {}
    }
    Ok(())
}

/// Decode a `text` content block into a [`Part::Text`] with annotations + cache_control.
fn decode_text_block(b: &Value) -> Part {
    let text = b.get("text").and_then(Value::as_str).unwrap_or("").to_string();
    let cache_control = decode_cache_control(b.get("cache_control"));
    let annotations = decode_citations(b.get("citations"));
    Part::Text { text, annotations, cache_control }
}

/// Decode a `citations` array into IR annotations.
fn decode_citations(v: Option<&Value>) -> Vec<llm_xlate_core::Annotation> {
    let mut out = Vec::new();
    if let Some(Value::Array(arr)) = v {
        for c in arr {
            let kind = c.get("type").and_then(Value::as_str).unwrap_or("citation").to_string();
            out.push(llm_xlate_core::Annotation::new(kind, c.clone()));
        }
    }
    out
}

/// Decode an image/document `source` object into an IR [`MediaSource`].
fn decode_media_source(source: Option<&Value>) -> MediaSource {
    let obj = match source.and_then(Value::as_object) {
        Some(o) => o,
        None => return MediaSource::Text(String::new()),
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type =
                obj.get("media_type").and_then(Value::as_str).unwrap_or("application/octet-stream");
            let data = obj.get("data").and_then(Value::as_str).unwrap_or("");
            let bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                data.as_bytes(),
            )
            .unwrap_or_default();
            MediaSource::Base64 { media_type: media_type.to_string(), data: bytes.into() }
        }
        Some("url") => MediaSource::Url(obj.get("url").and_then(Value::as_str).unwrap_or("").to_string()),
        Some("file") => MediaSource::FileRef {
            family: FAMILY,
            id: obj.get("file_id").and_then(Value::as_str).unwrap_or("").to_string(),
        },
        Some("text") => {
            MediaSource::Text(obj.get("data").and_then(Value::as_str).unwrap_or("").to_string())
        }
        _ => MediaSource::Text(String::new()),
    }
}

/// Decode a `document` content block into a [`Part::Document`].
fn decode_document_block(b: &Value) -> Part {
    let source = decode_media_source(b.get("source"));
    let title = b.get("title").and_then(Value::as_str).map(str::to_string);
    let media_type = match &source {
        MediaSource::Base64 { media_type, .. } => media_type.clone(),
        MediaSource::Text(_) => "text/plain".to_string(),
        _ => b
            .get("source")
            .and_then(|s| s.get("media_type"))
            .and_then(Value::as_str)
            .unwrap_or("application/pdf")
            .to_string(),
    };
    Part::Document { source, title, media_type }
}

/// Decode a `tool_use` block into [`Item::ToolCall`] (arguments = canonical text of `input`).
fn decode_tool_use(b: &Value) -> Item {
    let call_id = b.get("id").and_then(Value::as_str).unwrap_or("").to_string();
    let name = b.get("name").and_then(Value::as_str).unwrap_or("").to_string();
    let input = b.get("input").cloned().unwrap_or(Value::Object(Map::new()));
    Item::ToolCall {
        call_id: CallId::new(call_id),
        name,
        arguments: canon::json_text::from_value(&input),
        id: None,
    }
}

/// Decode a `tool_result` block into [`Item::ToolResult`].
fn decode_tool_result(b: &Value) -> Item {
    let call_id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("").to_string();
    let is_error = b.get("is_error").and_then(Value::as_bool).unwrap_or(false);
    let content = match b.get("content") {
        Some(Value::String(s)) => vec![Part::text(s.clone())],
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for blk in blocks {
                match blk.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(decode_text_block(blk)),
                    Some("image") => parts.push(Part::Image(decode_media_source(blk.get("source")))),
                    Some("document") => parts.push(decode_document_block(blk)),
                    _ => parts.push(Part::text(blk.to_string())),
                }
            }
            parts
        }
        _ => Vec::new(),
    };
    Item::ToolResult { call_id: CallId::new(call_id), content, is_error, id: None }
}

/// Decode a `thinking` block into [`Item::Reasoning`] with a native/opened signature carrier.
///
/// An absent or empty `signature` means there is **no** opaque carrier, not a native Anthropic
/// one: that is the shape the client-facing encoders emit for reasoning that arrived as plain
/// text from a Chat backend (`build_reasoning_block`, and the streaming encoder, both attach a
/// signature only when a blob exists). Fabricating a carrier here made `lower()` see an
/// Anthropic-family blob on the way back and drop it as foreign, so plain-text reasoning was
/// destroyed on every turn of an Anthropic-client conversation with a `replay = "text_field"`
/// backend — even though that backend can replay it. A real Anthropic thinking block always
/// carries a signature, so it still decodes to `Some`.
fn decode_thinking_block(b: &Value, ctx: &DecodeCtx) -> Item {
    let text = b.get("thinking").and_then(Value::as_str).unwrap_or("").to_string();
    let opaque = b
        .get("signature")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|s| ctx.sealer.open_or_native(s, FAMILY, OpaqueKind::Signature));
    Item::Reasoning(llm_xlate_core::ReasoningItem {
        text: Some(text),
        summary: Vec::new(),
        opaque,
        id: None,
    })
}

/// Decode a `redacted_thinking` block into [`Item::Reasoning`] (no text; redacted carrier).
///
/// The `data` may be a router envelope (`rtr1.…`): that is how the client-facing encoders carry
/// a foreign provider's opaque reasoning (e.g. OpenAI `encrypted_content`) to an Anthropic
/// client. It must be opened here so the blob keeps its true family/kind and can be replayed to
/// that provider; treating it as a native Anthropic blob made `lower()` drop it as foreign on
/// every multi-turn GPT conversation through the Anthropic surface.
///
/// An absent or empty `data` carries nothing, so it yields no opaque carrier — the same rule
/// [`decode_thinking_block`] applies to `signature`. The resulting item holds no text, summary
/// or blob, which the encoders and `lower()` each report accurately; claiming a native
/// Anthropic blob instead made every downstream message describe a carrier that was never there.
fn decode_redacted_thinking(b: &Value, ctx: &DecodeCtx) -> Item {
    let opaque = b
        .get("data")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|s| ctx.sealer.open_or_native(s, FAMILY, OpaqueKind::Redacted));
    Item::Reasoning(llm_xlate_core::ReasoningItem {
        text: None,
        summary: Vec::new(),
        opaque,
        id: None,
    })
}

/// Decode one `tools[]` entry into a [`ToolDef`].
fn decode_tool(tool: &Value) -> ToolDef {
    if tool_is_provider(tool) {
        return ToolDef::Provider(OpaqueItem::new(FAMILY, tool.clone()));
    }
    let name = tool.get("name").and_then(Value::as_str).unwrap_or("").to_string();
    let description = tool.get("description").and_then(Value::as_str).map(str::to_string);
    let parameters = tool.get("input_schema").cloned().unwrap_or(Value::Object(Map::new()));
    let strict = tool.get("strict").and_then(Value::as_bool);
    let cache_control = decode_cache_control(tool.get("cache_control"));
    ToolDef::Function { name, description, parameters, strict, cache_control }
}

/// Decode `tool_choice` into IR [`ToolChoice`] + `parallel_tool_calls`.
fn decode_tool_choice(tc: &Value, req: &mut IrRequest) {
    let obj = match tc.as_object() {
        Some(o) => o,
        None => return,
    };
    req.tool_choice = match obj.get("type").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => ToolChoice::Named(
            obj.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
        ),
        _ => ToolChoice::Auto,
    };
    if let Some(disable) = obj.get("disable_parallel_tool_use").and_then(Value::as_bool) {
        req.parallel_tool_calls = Some(!disable);
    }
}

/// Decode `thinking` config + `output_config` into reasoning + output-format IR.
fn decode_thinking_and_output(wire: &AntRequestWire, req: &mut IrRequest) {
    // output_config.format → OutputFormat; output_config.effort feeds adaptive reasoning.
    let oc_effort = wire
        .output_config
        .as_ref()
        .and_then(|oc| oc.get("effort"))
        .and_then(Value::as_str)
        .and_then(effort_from_token);

    if let Some(oc) = &wire.output_config {
        if let Some(format) = oc.get("format").and_then(Value::as_object) {
            if format.get("type").and_then(Value::as_str) == Some("json_schema") {
                let schema = format.get("schema").cloned().unwrap_or(Value::Object(Map::new()));
                req.output.format = OutputFormat::JsonSchema {
                    name: "response".to_string(),
                    schema,
                    strict: true,
                    description: None,
                };
            }
        }
        // Preserve any other output_config keys verbatim.
        if let Some(obj) = oc.as_object() {
            for (k, v) in obj {
                if k != "format" && k != "effort" {
                    req.ext.insert(ext_key(&format!("output_config.{k}")), v.clone());
                }
            }
        }
    }

    match wire.thinking.as_ref().and_then(|t| t.get("type")).and_then(Value::as_str) {
        Some("enabled") => {
            let budget =
                wire.thinking.as_ref().and_then(|t| t.get("budget_tokens")).and_then(Value::as_u64);
            if let Some(b) = budget {
                let b = b as u32;
                req.reasoning.budget_tokens = Some(b);
                req.reasoning.effort = Some(Effort::from_budget_tokens(b));
            }
            req.reasoning.enabled = Some(true);
        }
        Some("disabled") => {
            req.reasoning.effort = Some(Effort::None);
            req.reasoning.enabled = Some(false);
        }
        Some("adaptive") => {
            req.reasoning.enabled = Some(true);
            req.reasoning.effort = oc_effort;
        }
        _ => {}
    }
}

/// Map an Anthropic effort token to an IR [`Effort`].
fn effort_from_token(t: &str) -> Option<Effort> {
    Some(match t {
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        "xhigh" => Effort::XHigh,
        "max" => Effort::Max,
        _ => return None,
    })
}
