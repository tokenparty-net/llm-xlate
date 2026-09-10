//! `decode_request`: a Chat Completions request body → [`IrRequest`] (client-facing, lossless).
//!
//! The top-level object is walked key by key: every field we understand is mapped into the
//! IR; every field we do not is captured verbatim into `ext["chat.<field>"]`. Message-level
//! `name` and any unknown message/instruction fields are captured into the side tables
//! `ext["chat.msg_ext"]` / `ext["chat.instr_ext"]`, keyed by the produced item (or instruction)
//! index, so a Chat→Chat round trip re-emits them in place.

use base64::Engine;
use serde_json::{Map, Value};

use llm_xlate_core::canon;
use llm_xlate_core::codec::DecodeCtx;
use llm_xlate_core::ir::{
    CallId, Effort, Instruction, InstructionRole, IrRequest, Item, JsonText, MediaSource, ModelRef,
    OpaqueKind, Part, Position, ProviderFamily, ReasoningExposure, ReasoningItem, Role, ToolChoice,
    ToolDef, Verbosity,
};
use llm_xlate_core::{OpaqueItem, XlateError};

use crate::common::ns_key;

/// Decode a Chat Completions request body into the IR.
pub fn decode_request(body: &[u8], ctx: &DecodeCtx) -> Result<IrRequest, XlateError> {
    let value = canon::parse(body)?;
    let mut obj = match value {
        Value::Object(m) => m,
        _ => return Err(XlateError::invalid_request("request body must be a JSON object")),
    };

    let mut req = IrRequest::default();
    // Chat clients always see the de-facto `reasoning_content` field, so expose is Full.
    req.reasoning.expose = ReasoningExposure::Full;

    // ---- model ----
    if let Some(Value::String(m)) = obj.remove("model") {
        req.model = ModelRef::new(m);
    }

    // ---- messages ----
    let mut image_detail = Map::new();
    let mut msg_ext = Map::new();
    let mut instr_ext = Map::new();
    let mut legacy_function_call = false;
    if let Some(msgs) = obj.remove("messages") {
        let arr = msgs
            .as_array()
            .ok_or_else(|| XlateError::invalid_request("`messages` must be an array"))?;
        for msg in arr {
            decode_message(
                msg,
                ctx,
                &mut req,
                &mut image_detail,
                &mut msg_ext,
                &mut instr_ext,
                &mut legacy_function_call,
            )?;
        }
    }

    // ---- tools / functions (legacy) / tool_choice / function_call (legacy) ----
    let mut legacy_functions = false;
    if let Some(tools) = obj.remove("tools") {
        decode_tools(&tools, &mut req)?;
    }
    if let Some(funcs) = obj.remove("functions") {
        legacy_functions = true;
        if let Some(arr) = funcs.as_array() {
            for f in arr {
                if let Some(o) = f.as_object() {
                    req.tools.push(ToolDef::Function {
                        name: o.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                        description: o
                            .get("description")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        parameters: o.get("parameters").cloned().unwrap_or(Value::Null),
                        strict: None,
                        cache_control: None,
                    });
                }
            }
        }
    }
    if let Some(tc) = obj.remove("tool_choice") {
        req.tool_choice = decode_tool_choice(&tc);
    }
    if let Some(fc) = obj.remove("function_call") {
        // Legacy top-level function_call is the tool-choice equivalent.
        req.tool_choice = decode_tool_choice(&fc);
    }
    if let Some(p) = obj.remove("parallel_tool_calls") {
        req.parallel_tool_calls = p.as_bool();
    }

    // ---- response_format ----
    if let Some(rf) = obj.remove("response_format") {
        req.output.format = decode_response_format(&rf);
    }

    // ---- reasoning_effort / verbosity ----
    if let Some(Value::String(e)) = obj.remove("reasoning_effort") {
        req.reasoning.effort = Some(decode_effort(&e));
    }
    if let Some(Value::String(v)) = obj.remove("verbosity") {
        req.output.verbosity = decode_verbosity(&v);
    }

    // ---- limits ----
    if let Some(v) = obj.remove("max_completion_tokens") {
        req.limits.max_output_tokens = v.as_u64().map(|n| n as u32);
        req.ext.insert(ns_key("max_tokens_field"), Value::from("max_completion_tokens"));
    } else if let Some(v) = obj.remove("max_tokens") {
        req.limits.max_output_tokens = v.as_u64().map(|n| n as u32);
        req.ext.insert(ns_key("max_tokens_field"), Value::from("max_tokens"));
    }
    if let Some(stop) = obj.remove("stop") {
        req.limits.stop_sequences = decode_stop(&stop);
    }

    // ---- sampling ----
    if let Some(v) = obj.remove("temperature") {
        req.sampling.temperature = v.as_f64();
    }
    if let Some(v) = obj.remove("top_p") {
        req.sampling.top_p = v.as_f64();
    }
    if let Some(v) = obj.remove("n") {
        req.sampling.n = v.as_u64().map(|n| n as u32);
    }
    if let Some(v) = obj.remove("presence_penalty") {
        req.sampling.presence_penalty = v.as_f64();
    }
    if let Some(v) = obj.remove("frequency_penalty") {
        req.sampling.frequency_penalty = v.as_f64();
    }
    if let Some(v) = obj.remove("logit_bias") {
        if !v.is_null() {
            req.sampling.logit_bias = Some(v);
        }
    }
    if let Some(v) = obj.remove("logprobs") {
        req.sampling.logprobs = v.as_bool();
    }
    if let Some(v) = obj.remove("top_logprobs") {
        req.sampling.top_logprobs = v.as_u64().map(|n| n as u32);
    }
    if let Some(v) = obj.remove("seed") {
        req.sampling.seed = v.as_i64();
    }

    // ---- meta / state / cache ----
    if let Some(Value::String(u)) = obj.remove("user") {
        req.meta.user = Some(u);
    }
    if let Some(Value::String(s)) = obj.remove("safety_identifier") {
        req.meta.safety_identifier = Some(s);
    }
    if let Some(Value::Object(m)) = obj.remove("metadata") {
        req.meta.metadata = m;
    }
    if let Some(Value::String(t)) = obj.remove("service_tier") {
        req.meta.service_tier = Some(t);
    }
    if let Some(v) = obj.remove("store") {
        req.state.store = v.as_bool();
    }
    if let Some(Value::String(k)) = obj.remove("prompt_cache_key") {
        req.cache.prompt_cache_key = Some(k);
    }

    // ---- streaming ----
    if let Some(v) = obj.remove("stream") {
        req.stream = v.as_bool().unwrap_or(false);
    }
    if let Some(so) = obj.remove("stream_options") {
        if so.get("include_usage").and_then(Value::as_bool) == Some(true) {
            req.ext.insert(ns_key("include_usage"), Value::Bool(true));
        }
    }

    // ---- control-key side tables ----
    if legacy_functions {
        req.ext.insert(ns_key("legacy_functions"), Value::Bool(true));
    }
    if legacy_function_call {
        req.ext.insert(ns_key("legacy_function_call"), Value::Bool(true));
    }
    if !image_detail.is_empty() {
        req.ext.insert(ns_key("image_detail"), Value::Object(image_detail));
    }
    if !msg_ext.is_empty() {
        req.ext.insert(ns_key("msg_ext"), Value::Object(msg_ext));
    }
    if !instr_ext.is_empty() {
        req.ext.insert(ns_key("instr_ext"), Value::Object(instr_ext));
    }

    // ---- remaining unknown top-level fields → ext["chat.<field>"] ----
    for (k, v) in obj {
        req.ext.insert(ns_key(&k), v);
    }

    Ok(req)
}

/// Consume the fields we map from a message object, leaving `name` plus any unknown fields as
/// the "extras" carried into a side table.
fn message_extras(obj: &Map<String, Value>, consumed: &[&str]) -> Option<Value> {
    let mut extras = Map::new();
    for (k, v) in obj {
        if k == "role" || consumed.contains(&k.as_str()) {
            continue;
        }
        extras.insert(k.clone(), v.clone());
    }
    if extras.is_empty() {
        None
    } else {
        Some(Value::Object(extras))
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_message(
    msg: &Value,
    ctx: &DecodeCtx,
    req: &mut IrRequest,
    image_detail: &mut Map<String, Value>,
    msg_ext: &mut Map<String, Value>,
    instr_ext: &mut Map<String, Value>,
    legacy_function_call: &mut bool,
) -> Result<(), XlateError> {
    let obj = msg
        .as_object()
        .ok_or_else(|| XlateError::invalid_request("each message must be an object"))?;
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("");

    match role {
        "system" | "developer" => {
            let instr_index = req.instructions.len();
            let position = if req.items.is_empty() {
                Position::Leading
            } else {
                Position::Before(req.items.len())
            };
            let content = decode_instruction_content(obj.get("content"));
            req.instructions.push(Instruction {
                role: if role == "developer" {
                    InstructionRole::Developer
                } else {
                    InstructionRole::System
                },
                position,
                content,
                cache_control: None,
                effort: None,
                clear_at: None,
            });
            if let Some(extras) = message_extras(obj, &["content"]) {
                instr_ext.insert(instr_index.to_string(), extras);
            }
        }
        "user" => {
            let item_index = req.items.len();
            let content = decode_user_content(obj.get("content"), item_index, image_detail);
            req.items.push(Item::Message { role: Role::User, content, id: None });
            if let Some(extras) = message_extras(obj, &["content"]) {
                msg_ext.insert(item_index.to_string(), extras);
            }
        }
        "assistant" => {
            let first_index = req.items.len();
            // 1. reasoning
            let reasoning_text = obj
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| obj.get("reasoning").and_then(Value::as_str))
                .map(str::to_string);
            let details = obj.get("reasoning_details").and_then(Value::as_array);
            if reasoning_text.is_some() || details.is_some() {
                let mut item = ReasoningItem { text: reasoning_text, ..Default::default() };
                if let Some(details) = details {
                    for entry in details {
                        apply_reasoning_detail(entry, &mut item, ctx);
                    }
                }
                req.items.push(Item::Reasoning(item));
            }
            // 2. content + refusal
            let mut parts =
                decode_assistant_content(obj.get("content"));
            if let Some(r) = obj.get("refusal").and_then(Value::as_str) {
                parts.push(Part::Refusal { text: r.to_string() });
            }
            if !parts.is_empty() {
                req.items.push(Item::Message { role: Role::Assistant, content: parts, id: None });
            }
            // 3. tool_calls
            if let Some(calls) = obj.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    if let Some(o) = call.as_object() {
                        let func = o.get("function").and_then(Value::as_object);
                        req.items.push(Item::ToolCall {
                            call_id: CallId::new(
                                o.get("id").and_then(Value::as_str).unwrap_or(""),
                            ),
                            name: func
                                .and_then(|f| f.get("name"))
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            arguments: JsonText::new(
                                func.and_then(|f| f.get("arguments"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string(),
                            ),
                            id: None,
                        });
                    }
                }
            }
            // 4. legacy function_call
            if let Some(fc) = obj.get("function_call").and_then(Value::as_object) {
                *legacy_function_call = true;
                let name = fc.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                req.items.push(Item::ToolCall {
                    call_id: CallId::new(name.clone()),
                    name,
                    arguments: JsonText::new(
                        fc.get("arguments").and_then(Value::as_str).unwrap_or("").to_string(),
                    ),
                    id: None,
                });
            }
            if req.items.len() > first_index {
                if let Some(extras) = message_extras(
                    obj,
                    &[
                        "content",
                        "refusal",
                        "tool_calls",
                        "function_call",
                        "reasoning_content",
                        "reasoning",
                        "reasoning_details",
                    ],
                ) {
                    msg_ext.insert(first_index.to_string(), extras);
                }
            }
        }
        "tool" | "function" => {
            let item_index = req.items.len();
            let call_id = obj
                .get("tool_call_id")
                .and_then(Value::as_str)
                .or_else(|| obj.get("name").and_then(Value::as_str))
                .unwrap_or("")
                .to_string();
            let content = decode_tool_content(obj.get("content"), item_index, image_detail);
            req.items.push(Item::ToolResult {
                call_id: CallId::new(call_id),
                content,
                is_error: false,
                id: None,
            });
            if let Some(extras) = message_extras(obj, &["content", "tool_call_id"]) {
                msg_ext.insert(item_index.to_string(), extras);
            }
        }
        other => {
            return Err(XlateError::invalid_request(format!("unknown message role: {other}")));
        }
    }
    Ok(())
}

/// Apply one `reasoning_details[]` entry to a reasoning item. The single documented lossy
/// decode step is that entries of an unrecognized shape are dropped.
fn apply_reasoning_detail(entry: &Value, item: &mut ReasoningItem, ctx: &DecodeCtx) {
    let ty = entry.get("type").and_then(Value::as_str).unwrap_or("");
    match ty {
        "router.opaque" => {
            if let Some(data) = entry.get("data").and_then(Value::as_str) {
                item.opaque = Some(ctx.sealer.open_or_native(
                    data,
                    ProviderFamily::OpenAI,
                    OpaqueKind::Encrypted,
                ));
            }
        }
        // Real gateways (e.g. OpenRouter) carry replayable/redacted reasoning as
        // `reasoning.encrypted` / `reasoning.redacted` entries with a `data` payload. Keep
        // them as opaque so Chat→Chat losslessness holds for that field.
        "reasoning.encrypted" => {
            if let Some(data) = entry.get("data").and_then(Value::as_str) {
                item.opaque = Some(ctx.sealer.open_or_native(
                    data,
                    ProviderFamily::OpenAI,
                    OpaqueKind::Encrypted,
                ));
            }
        }
        "reasoning.redacted" => {
            if let Some(data) = entry.get("data").and_then(Value::as_str) {
                item.opaque = Some(ctx.sealer.open_or_native(
                    data,
                    ProviderFamily::OpenAI,
                    OpaqueKind::Redacted,
                ));
            }
        }
        "reasoning.summary" => {
            if let Some(s) = entry.get("summary").and_then(Value::as_str) {
                item.summary.push(s.to_string());
            }
        }
        "reasoning.text" => {
            if let Some(t) = entry.get("text").and_then(Value::as_str) {
                item.text = Some(t.to_string());
            }
        }
        _ => { /* unrecognized entry dropped (the single lossy decode step) */ }
    }
}

/// Decode instruction content (`system`/`developer` message) into text parts.
fn decode_instruction_content(content: Option<&Value>) -> Vec<Part> {
    match content {
        Some(Value::String(s)) => vec![Part::text(s.clone())],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).map(Part::text))
            .collect(),
        _ => Vec::new(),
    }
}

/// Decode a user message's content into IR parts.
fn decode_user_content(
    content: Option<&Value>,
    item_index: usize,
    image_detail: &mut Map<String, Value>,
) -> Vec<Part> {
    match content {
        Some(Value::String(s)) => vec![Part::text(s.clone())],
        Some(Value::Array(arr)) => {
            let mut parts = Vec::new();
            for (part_index, p) in arr.iter().enumerate() {
                if let Some(part) = decode_user_part(p, item_index, part_index, image_detail) {
                    parts.push(part);
                }
            }
            parts
        }
        _ => Vec::new(),
    }
}

fn decode_user_part(
    p: &Value,
    item_index: usize,
    part_index: usize,
    image_detail: &mut Map<String, Value>,
) -> Option<Part> {
    let ty = p.get("type").and_then(Value::as_str)?;
    match ty {
        "text" => Some(Part::text(p.get("text").and_then(Value::as_str).unwrap_or(""))),
        "image_url" => {
            let iu = p.get("image_url")?;
            let url = iu.get("url").and_then(Value::as_str).unwrap_or("");
            if let Some(detail) = iu.get("detail").and_then(Value::as_str) {
                if detail != "auto" {
                    image_detail
                        .insert(format!("{item_index}/{part_index}"), Value::from(detail));
                }
            }
            let source = match canon::data_url::parse(url) {
                Some((media_type, data)) => MediaSource::Base64 { media_type, data },
                None => MediaSource::Url(url.to_string()),
            };
            Some(Part::Image(source))
        }
        "input_audio" => {
            let ia = p.get("input_audio")?;
            let format = ia.get("format").and_then(Value::as_str).unwrap_or("wav");
            let data_str = ia.get("data").and_then(Value::as_str).unwrap_or("");
            let data = base64::engine::general_purpose::STANDARD
                .decode(data_str.as_bytes())
                .unwrap_or_default();
            Some(Part::Audio(MediaSource::Base64 {
                media_type: format!("audio/{format}"),
                data: data.into(),
            }))
        }
        "file" => {
            let file = p.get("file")?;
            let filename =
                file.get("filename").and_then(Value::as_str).map(str::to_string);
            if let Some(fid) = file.get("file_id").and_then(Value::as_str) {
                Some(Part::Document {
                    source: MediaSource::FileRef {
                        family: ProviderFamily::OpenAI,
                        id: fid.to_string(),
                    },
                    title: filename,
                    media_type: "application/pdf".to_string(),
                })
            } else if let Some(fd) = file.get("file_data").and_then(Value::as_str) {
                let (media_type, data) = canon::data_url::parse(fd).unwrap_or_else(|| {
                    (
                        "application/pdf".to_string(),
                        base64::engine::general_purpose::STANDARD
                            .decode(fd.as_bytes())
                            .unwrap_or_default()
                            .into(),
                    )
                });
                Some(Part::Document {
                    source: MediaSource::Base64 { media_type: media_type.clone(), data },
                    title: filename,
                    media_type,
                })
            } else {
                None
            }
        }
        "refusal" => {
            Some(Part::Refusal { text: p.get("refusal").and_then(Value::as_str).unwrap_or("").to_string() })
        }
        _ => None,
    }
}

/// Decode an assistant message's content (string | parts | null) into IR parts.
fn decode_assistant_content(content: Option<&Value>) -> Vec<Part> {
    match content {
        Some(Value::String(s)) if !s.is_empty() => vec![Part::text(s.clone())],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|p| {
                let ty = p.get("type").and_then(Value::as_str)?;
                match ty {
                    "text" => Some(Part::text(p.get("text").and_then(Value::as_str).unwrap_or(""))),
                    "refusal" => Some(Part::Refusal {
                        text: p.get("refusal").and_then(Value::as_str).unwrap_or("").to_string(),
                    }),
                    _ => None,
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Decode a tool message's content into IR parts. Text parts are kept; media parts
/// (`image_url` / `file` / `input_audio`) are decoded too (so a target that cannot carry media
/// on a tool result can fold them per plan §7.4).
fn decode_tool_content(
    content: Option<&Value>,
    item_index: usize,
    image_detail: &mut Map<String, Value>,
) -> Vec<Part> {
    match content {
        Some(Value::String(s)) => vec![Part::text(s.clone())],
        Some(Value::Array(arr)) => {
            let mut parts = Vec::new();
            for (part_index, p) in arr.iter().enumerate() {
                match p {
                    Value::String(s) => parts.push(Part::text(s.clone())),
                    Value::Object(_) => {
                        if let Some(part) =
                            decode_user_part(p, item_index, part_index, image_detail)
                        {
                            parts.push(part);
                        }
                    }
                    _ => {}
                }
            }
            parts
        }
        _ => Vec::new(),
    }
}

/// Decode the `tools` array.
fn decode_tools(tools: &Value, req: &mut IrRequest) -> Result<(), XlateError> {
    let arr = tools
        .as_array()
        .ok_or_else(|| XlateError::invalid_request("`tools` must be an array"))?;
    for tool in arr {
        let ty = tool.get("type").and_then(Value::as_str).unwrap_or("function");
        if ty == "function" {
            let func = tool.get("function").and_then(Value::as_object);
            req.tools.push(ToolDef::Function {
                name: func
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                description: func
                    .and_then(|f| f.get("description"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                parameters: func
                    .and_then(|f| f.get("parameters"))
                    .cloned()
                    .unwrap_or(Value::Null),
                strict: func.and_then(|f| f.get("strict")).and_then(Value::as_bool),
                cache_control: None,
            });
        } else {
            // Non-function tool types pass through opaquely (OpenAI family).
            req.tools.push(ToolDef::Provider(OpaqueItem::new(ProviderFamily::OpenAI, tool.clone())));
        }
    }
    Ok(())
}

/// Decode a `tool_choice` / legacy `function_call` value.
fn decode_tool_choice(tc: &Value) -> ToolChoice {
    match tc {
        Value::String(s) => match s.as_str() {
            "none" => ToolChoice::None,
            "required" | "any" => ToolChoice::Required,
            "auto" => ToolChoice::Auto,
            _ => ToolChoice::Auto,
        },
        Value::Object(o) => {
            // Modern {type:"function", function:{name}} or legacy {name}.
            if let Some(name) = o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .or_else(|| o.get("name").and_then(Value::as_str))
            {
                ToolChoice::Named(name.to_string())
            } else {
                ToolChoice::Auto
            }
        }
        _ => ToolChoice::Auto,
    }
}

/// Decode a `response_format` value.
fn decode_response_format(rf: &Value) -> llm_xlate_core::ir::OutputFormat {
    use llm_xlate_core::ir::OutputFormat;
    let ty = rf.get("type").and_then(Value::as_str).unwrap_or("text");
    match ty {
        "json_object" => OutputFormat::JsonObject,
        "json_schema" => {
            let js = rf.get("json_schema");
            OutputFormat::JsonSchema {
                name: js
                    .and_then(|j| j.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("schema")
                    .to_string(),
                schema: js.and_then(|j| j.get("schema")).cloned().unwrap_or(Value::Null),
                strict: js
                    .and_then(|j| j.get("strict"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                description: js
                    .and_then(|j| j.get("description"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }
        }
        _ => OutputFormat::Text,
    }
}

/// Decode a `reasoning_effort` token.
fn decode_effort(s: &str) -> Effort {
    match s {
        "none" => Effort::None,
        "minimal" => Effort::Minimal,
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        "xhigh" => Effort::XHigh,
        _ => Effort::Medium,
    }
}

/// Decode a `verbosity` token.
fn decode_verbosity(s: &str) -> Option<Verbosity> {
    match s {
        "low" => Some(Verbosity::Low),
        "medium" => Some(Verbosity::Medium),
        "high" => Some(Verbosity::High),
        _ => None,
    }
}

/// Decode a `stop` value (string or array of strings).
fn decode_stop(stop: &Value) -> Vec<String> {
    match stop {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => {
            arr.iter().filter_map(Value::as_str).map(str::to_string).collect()
        }
        _ => Vec::new(),
    }
}
