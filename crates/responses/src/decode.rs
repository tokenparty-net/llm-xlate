//! `decode_request`: an OpenAI Responses request body → [`IrRequest`] (lossless; unknown
//! top-level fields land in `ext` under the `responses.` namespace).

use serde_json::{Map, Value};

use llm_xlate_core::codec::HeaderMap;
use llm_xlate_core::session::capture_session;
use llm_xlate_core::{
    canon, CacheHints, DecodeCtx, Effort, Extensions, Instruction, InstructionRole, IrRequest, Item,
    ItemId, JsonText, Limits, MediaSource, ModelRef, OpaqueBlob, OpaqueItem, OpaqueKind, OutputConfig,
    OutputFormat, Part, Position, ProviderFamily, ReasoningConfig, ReasoningExposure, RequestMeta,
    Role, Sampling, StateConfig, SummaryLevel, ToolChoice, ToolDef, Verbosity, XlateError,
};

use crate::render::REASONING_KIND;
use crate::util::{get_bool, get_str, get_u32};

/// Top-level request fields the decoder understands natively (everything else → `ext`).
const KNOWN: &[&str] = &[
    "model",
    "instructions",
    "input",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "reasoning",
    "max_output_tokens",
    "temperature",
    "top_p",
    "top_logprobs",
    "store",
    "previous_response_id",
    "conversation",
    "background",
    "include",
    "metadata",
    "user",
    "safety_identifier",
    "prompt_cache_key",
    "service_tier",
    "stream",
];

/// Decode a Responses request body into the IR.
pub(crate) fn decode_request(
    body: &[u8],
    hdrs: &HeaderMap,
    ctx: &DecodeCtx,
) -> Result<IrRequest, XlateError> {
    let root = canon::parse(body)?;
    let obj = root.as_object().ok_or_else(|| XlateError::invalid_request("request body must be a JSON object"))?;

    let mut req = IrRequest {
        model: ModelRef::new(get_str(&root, "model").unwrap_or_default()),
        ..Default::default()
    };
    let mut ext = Extensions::new();

    // instructions (string) → leading system instruction, first.
    if let Some(s) = get_str(&root, "instructions") {
        req.instructions.push(Instruction::system_text(s));
    }

    // input: string | [items]
    decode_input(&root, &mut req, ctx, &mut ext)?;

    // tools
    if let Some(Value::Array(arr)) = obj.get("tools") {
        for t in arr {
            req.tools.push(decode_tool(t));
        }
    }

    // tool_choice
    if let Some(tc) = obj.get("tool_choice") {
        req.tool_choice = decode_tool_choice(tc, &mut ext);
    }

    if let Some(b) = get_bool(&root, "parallel_tool_calls") {
        req.parallel_tool_calls = Some(b);
    }

    // text {format, verbosity}
    if let Some(text) = obj.get("text") {
        decode_text(text, &mut req.output);
    }

    // reasoning {effort, summary}
    if let Some(r) = obj.get("reasoning") {
        decode_reasoning(r, &mut req.reasoning);
    }

    if let Some(n) = get_u32(&root, "max_output_tokens") {
        req.limits = Limits { max_output_tokens: Some(n), ..req.limits };
    }

    // sampling
    req.sampling = Sampling {
        temperature: root.get("temperature").and_then(Value::as_f64),
        top_p: root.get("top_p").and_then(Value::as_f64),
        top_logprobs: get_u32(&root, "top_logprobs"),
        ..Default::default()
    };

    // state
    req.state = StateConfig {
        store: get_bool(&root, "store"),
        previous_response_id: get_str(&root, "previous_response_id").map(Into::into),
        conversation: decode_conversation(obj.get("conversation")),
        background: get_bool(&root, "background"),
        include: decode_str_list(obj.get("include")),
    };

    // meta
    req.meta = RequestMeta {
        user: get_str(&root, "user").map(str::to_string),
        safety_identifier: get_str(&root, "safety_identifier").map(str::to_string),
        metadata: obj.get("metadata").and_then(Value::as_object).cloned().unwrap_or_default(),
        service_tier: get_str(&root, "service_tier").map(str::to_string),
    };

    req.cache = CacheHints {
        request_level: false,
        prompt_cache_key: get_str(&root, "prompt_cache_key").map(str::to_string),
    };

    // session affinity (priority: prompt_cache_key, session_id field, then headers). A
    // `session_id` field is non-standard for Responses and also round-trips via ext below.
    req.session = capture_session(
        req.cache.prompt_cache_key.as_deref(),
        get_str(&root, "session_id"),
        hdrs,
    );

    req.stream = get_bool(&root, "stream").unwrap_or(false);

    // Unknown top-level fields → ext["responses.<field>"].
    for (k, v) in obj {
        if !KNOWN.contains(&k.as_str()) {
            ext.insert(format!("responses.{k}"), v.clone());
        }
    }
    req.ext = ext;

    Ok(req)
}

/// Decode `input`: a string (single user message) or an array of input items. Leading
/// `system`/`developer` message items become leading instructions; later ones become
/// `Before(i)`.
fn decode_input(
    root: &Value,
    req: &mut IrRequest,
    ctx: &DecodeCtx,
    ext: &mut Extensions,
) -> Result<(), XlateError> {
    match root.get("input") {
        Some(Value::String(s)) => {
            req.items.push(Item::user_text(s.clone()));
            Ok(())
        }
        Some(Value::Array(arr)) => {
            let mut seen_non_instruction = false;
            for el in arr {
                if let Some((role, content)) = as_instruction_message(el) {
                    let ir_role = if role == "developer" {
                        InstructionRole::Developer
                    } else {
                        InstructionRole::System
                    };
                    let position = if seen_non_instruction {
                        Position::Before(req.items.len())
                    } else {
                        Position::Leading
                    };
                    req.instructions.push(Instruction {
                        role: ir_role,
                        position,
                        content: decode_instruction_content(content),
                        cache_control: None,
                        effort: None,
                        clear_at: None,
                    });
                } else {
                    seen_non_instruction = true;
                    let item_idx = req.items.len();
                    let item = decode_item(el, item_idx, ctx, ext)?;
                    req.items.push(item);
                }
            }
            Ok(())
        }
        Some(_) => Err(XlateError::invalid_request("`input` must be a string or array")),
        None => Ok(()),
    }
}

/// If `el` is a `system`/`developer` message item (typed or shorthand), return its role and
/// content value.
fn as_instruction_message(el: &Value) -> Option<(&str, &Value)> {
    let ty = el.get("type").and_then(Value::as_str);
    let role = el.get("role").and_then(Value::as_str)?;
    if (ty == Some("message") || ty.is_none()) && (role == "system" || role == "developer") {
        Some((role, el.get("content").unwrap_or(&Value::Null)))
    } else {
        None
    }
}

/// Decode instruction content (string or parts) into text parts.
fn decode_instruction_content(content: &Value) -> Vec<Part> {
    match content {
        Value::String(s) => vec![Part::text(s.clone())],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).map(Part::text))
            .collect(),
        _ => Vec::new(),
    }
}

/// Decode one input item (non-instruction) into an [`Item`].
fn decode_item(
    el: &Value,
    item_idx: usize,
    ctx: &DecodeCtx,
    ext: &mut Extensions,
) -> Result<Item, XlateError> {
    let ty = el.get("type").and_then(Value::as_str);
    let id = el.get("id").and_then(Value::as_str).map(ItemId::new);

    // Shorthand `{role, content}` with no type is a message.
    let effective_ty = ty.unwrap_or_else(|| if el.get("role").is_some() { "message" } else { "" });

    match effective_ty {
        "message" => {
            let role = el.get("role").and_then(Value::as_str).unwrap_or("user");
            let ir_role = if role == "assistant" { Role::Assistant } else { Role::User };
            let content = decode_message_content(
                el.get("content").unwrap_or(&Value::Null),
                item_idx,
                ext,
            );
            Ok(Item::Message { role: ir_role, content, id })
        }
        "function_call" => {
            let call_id = el.get("call_id").and_then(Value::as_str).unwrap_or_default();
            let name = el.get("name").and_then(Value::as_str).unwrap_or_default();
            let arguments = el.get("arguments").and_then(Value::as_str).unwrap_or("{}");
            Ok(Item::ToolCall {
                call_id: call_id.into(),
                name: name.to_string(),
                arguments: JsonText::new(arguments),
                id,
            })
        }
        "function_call_output" => {
            let call_id = el.get("call_id").and_then(Value::as_str).unwrap_or_default();
            let content = decode_tool_output(el.get("output").unwrap_or(&Value::Null));
            Ok(Item::ToolResult { call_id: call_id.into(), content, is_error: false, id })
        }
        "reasoning" => Ok(Item::Reasoning(decode_reasoning_item(el, id, ctx))),
        "compaction" => {
            let data = el.get("encrypted_content").and_then(Value::as_str).unwrap_or_default();
            let blob = ctx.sealer.open_or_native(data, ProviderFamily::OpenAI, OpaqueKind::Compaction);
            Ok(Item::Compaction(blob))
        }
        "item_reference" => {
            Ok(Item::ProviderToolCall(OpaqueItem::new(ProviderFamily::OpenAI, el.clone())))
        }
        // Hosted call outputs → ProviderToolResult.
        "computer_call_output" | "mcp_approval_response" | "custom_tool_call_output"
        | "local_shell_call_output" | "shell_call_output" | "apply_patch_call_output" => {
            Ok(Item::ProviderToolResult(OpaqueItem::new(ProviderFamily::OpenAI, el.clone())))
        }
        // Any other hosted call item → ProviderToolCall.
        "" => Err(XlateError::invalid_request("input item missing `type`")),
        _ => Ok(Item::ProviderToolCall(OpaqueItem::new(ProviderFamily::OpenAI, el.clone()))),
    }
}

/// Decode a message's `content` (string or content parts) into [`Part`]s.
fn decode_message_content(content: &Value, item_idx: usize, ext: &mut Extensions) -> Vec<Part> {
    match content {
        Value::String(s) => vec![Part::text(s.clone())],
        Value::Array(arr) => {
            let mut parts = Vec::new();
            for (part_idx, p) in arr.iter().enumerate() {
                if let Some(part) = decode_content_part(p, item_idx, part_idx, ext) {
                    parts.push(part);
                }
            }
            parts
        }
        _ => Vec::new(),
    }
}

/// Decode a single content part.
fn decode_content_part(p: &Value, item_idx: usize, part_idx: usize, ext: &mut Extensions) -> Option<Part> {
    match p.get("type").and_then(Value::as_str) {
        Some("input_text") | Some("output_text") | Some("text") => {
            let text = p.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
            let annotations = decode_annotations(p.get("annotations"));
            Some(Part::Text { text, annotations, cache_control: None })
        }
        Some("input_image") => {
            let detail = p.get("detail").and_then(Value::as_str);
            if let Some(d) = detail {
                if d != "auto" {
                    record_detail(ext, item_idx, part_idx, d);
                }
            }
            let src = if let Some(fid) = p.get("file_id").and_then(Value::as_str) {
                MediaSource::FileRef { family: ProviderFamily::OpenAI, id: fid.to_string() }
            } else if let Some(url) = p.get("image_url").and_then(Value::as_str) {
                image_source_from_url(url)
            } else {
                MediaSource::Url(String::new())
            };
            Some(Part::Image(src))
        }
        Some("input_file") => Some(decode_input_file(p)),
        Some("input_audio") => {
            let audio = p.get("input_audio").unwrap_or(&Value::Null);
            let src = if let Some(url) = audio.get("url").and_then(Value::as_str) {
                MediaSource::Url(url.to_string())
            } else if let Some(fid) = audio.get("file_id").and_then(Value::as_str) {
                MediaSource::FileRef { family: ProviderFamily::OpenAI, id: fid.to_string() }
            } else if let Some(data) = audio.get("data").and_then(Value::as_str) {
                let fmt = audio.get("format").and_then(Value::as_str).unwrap_or("wav");
                let media_type = format!("audio/{fmt}");
                match base64_decode(data) {
                    Some(bytes) => MediaSource::Base64 { media_type, data: bytes },
                    None => MediaSource::Text(data.to_string()),
                }
            } else {
                MediaSource::Text(String::new())
            };
            Some(Part::Audio(src))
        }
        Some("refusal") => {
            let text = p.get("refusal").and_then(Value::as_str).unwrap_or_default().to_string();
            Some(Part::Refusal { text })
        }
        _ => None,
    }
}

fn image_source_from_url(url: &str) -> MediaSource {
    if let Some((media_type, bytes)) = canon::data_url::parse(url) {
        MediaSource::Base64 { media_type, data: bytes }
    } else {
        MediaSource::Url(url.to_string())
    }
}

fn decode_input_file(p: &Value) -> Part {
    let filename = p.get("filename").and_then(Value::as_str).map(str::to_string);
    if let Some(fid) = p.get("file_id").and_then(Value::as_str) {
        return Part::Document {
            source: MediaSource::FileRef { family: ProviderFamily::OpenAI, id: fid.to_string() },
            title: filename,
            media_type: "application/pdf".to_string(),
        };
    }
    if let Some(url) = p.get("file_url").and_then(Value::as_str) {
        return Part::Document {
            source: MediaSource::Url(url.to_string()),
            title: filename,
            media_type: "application/pdf".to_string(),
        };
    }
    if let Some(data) = p.get("file_data").and_then(Value::as_str) {
        let (media_type, bytes) = canon::data_url::parse(data)
            .unwrap_or_else(|| ("application/pdf".to_string(), bytes::Bytes::new()));
        return Part::Document {
            source: MediaSource::Base64 { media_type: media_type.clone(), data: bytes },
            title: filename,
            media_type,
        };
    }
    Part::Document {
        source: MediaSource::Text(String::new()),
        title: filename,
        media_type: "text/plain".to_string(),
    }
}

/// Decode a `function_call_output.output` (string or content parts) into result parts.
fn decode_tool_output(output: &Value) -> Vec<Part> {
    match output {
        Value::String(s) => vec![Part::text(s.clone())],
        Value::Array(arr) => {
            let mut parts = Vec::new();
            for p in arr {
                match p.get("type").and_then(Value::as_str) {
                    Some("output_text") | Some("input_text") | Some("text") => {
                        let text = p.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
                        parts.push(Part::text(text));
                    }
                    Some("input_image") => {
                        if let Some(url) = p.get("image_url").and_then(Value::as_str) {
                            parts.push(Part::Image(image_source_from_url(url)));
                        } else if let Some(fid) = p.get("file_id").and_then(Value::as_str) {
                            parts.push(Part::Image(MediaSource::FileRef {
                                family: ProviderFamily::OpenAI,
                                id: fid.to_string(),
                            }));
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

/// Decode a `reasoning` input item.
fn decode_reasoning_item(el: &Value, id: Option<ItemId>, ctx: &DecodeCtx) -> llm_xlate_core::ReasoningItem {
    let summary: Vec<String> = el
        .get("summary")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter().filter_map(|s| s.get("text").and_then(Value::as_str).map(str::to_string)).collect()
        })
        .unwrap_or_default();
    let text = el.get("content").and_then(Value::as_array).map(|arr| {
        let joined: String = arr
            .iter()
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("");
        joined
    });
    let text = text.filter(|t| !t.is_empty());
    let opaque = el
        .get("encrypted_content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|d| ctx.sealer.open_or_native(d, ProviderFamily::OpenAI, REASONING_KIND));
    llm_xlate_core::ReasoningItem { text, summary, opaque, id }
}

/// Decode a tool definition.
fn decode_tool(t: &Value) -> ToolDef {
    match t.get("type").and_then(Value::as_str) {
        Some("function") => ToolDef::Function {
            name: t.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
            description: t.get("description").and_then(Value::as_str).map(str::to_string),
            parameters: t.get("parameters").cloned().unwrap_or(Value::Null),
            strict: t.get("strict").and_then(Value::as_bool),
            cache_control: None,
        },
        _ => ToolDef::Provider(OpaqueItem::new(ProviderFamily::OpenAI, t.clone())),
    }
}

/// Decode `tool_choice`.
fn decode_tool_choice(tc: &Value, ext: &mut Extensions) -> ToolChoice {
    match tc {
        Value::String(s) => match s.as_str() {
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        Value::Object(o) => match o.get("type").and_then(Value::as_str) {
            Some("function") => {
                let name = o.get("name").and_then(Value::as_str).unwrap_or_default();
                ToolChoice::Named(name.to_string())
            }
            _ => {
                ext.insert("responses.tool_choice", tc.clone());
                ToolChoice::Auto
            }
        },
        _ => ToolChoice::Auto,
    }
}

/// Decode `text` → output format + verbosity.
fn decode_text(text: &Value, output: &mut OutputConfig) {
    if let Some(fmt) = text.get("format") {
        output.format = match fmt.get("type").and_then(Value::as_str) {
            Some("json_object") => OutputFormat::JsonObject,
            Some("json_schema") => OutputFormat::JsonSchema {
                name: fmt.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                schema: fmt.get("schema").cloned().unwrap_or(Value::Null),
                strict: fmt.get("strict").and_then(Value::as_bool).unwrap_or(false),
                description: fmt.get("description").and_then(Value::as_str).map(str::to_string),
            },
            _ => OutputFormat::Text,
        };
    }
    output.verbosity = match text.get("verbosity").and_then(Value::as_str) {
        Some("low") => Some(Verbosity::Low),
        Some("medium") => Some(Verbosity::Medium),
        Some("high") => Some(Verbosity::High),
        _ => None,
    };
}

/// Decode `reasoning` → effort + exposure.
fn decode_reasoning(r: &Value, cfg: &mut ReasoningConfig) {
    let effort = r.get("effort").and_then(Value::as_str).map(effort_from_str);
    cfg.effort = effort;
    let summary = r.get("summary").and_then(Value::as_str);
    let generate = r.get("generate_summary").and_then(Value::as_str);
    let requested_summary = summary.or(generate);
    cfg.expose = match requested_summary {
        Some(s) => ReasoningExposure::Summary(level_from_str(s)),
        None => {
            // `effort:"none"` disables reasoning; defaulting it to a summary would emit the
            // contradictory `{effort:"none",summary:"auto"}` on re-encode, so treat it like absent
            // effort for the exposure default.
            if effort.is_some() && effort != Some(Effort::None) {
                ReasoningExposure::Summary(SummaryLevel::Auto)
            } else {
                ReasoningExposure::None
            }
        }
    };
    // A Responses client that asks for a summary *without* an `effort` is still expressing genuine
    // reasoning intent. Record it as an explicit enable so request-path encoders (which key off
    // intent, not `expose`) preserve the summary request rather than dropping it.
    if requested_summary.is_some() && effort.is_none() {
        cfg.enabled = Some(true);
    }
}

fn effort_from_str(s: &str) -> Effort {
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

fn level_from_str(s: &str) -> SummaryLevel {
    match s {
        "concise" => SummaryLevel::Concise,
        "detailed" => SummaryLevel::Detailed,
        _ => SummaryLevel::Auto,
    }
}

/// Decode `conversation`: string or `{id}`.
fn decode_conversation(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("id").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

/// Decode a string list (e.g. `include`).
fn decode_str_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(arr)) => {
            arr.iter().filter_map(Value::as_str).map(str::to_string).collect()
        }
        _ => Vec::new(),
    }
}

/// Decode annotations on a text part.
fn decode_annotations(v: Option<&Value>) -> Vec<llm_xlate_core::Annotation> {
    match v {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|a| {
                let kind = a.get("type").and_then(Value::as_str).unwrap_or("annotation").to_string();
                llm_xlate_core::Annotation::new(kind, a.clone())
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn record_detail(ext: &mut Extensions, item: usize, part: usize, detail: &str) {
    let entry = ext.0.entry("responses.image_detail".to_string()).or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(m) = entry {
        m.insert(format!("{item}/{part}"), Value::from(detail));
    }
}

fn base64_decode(s: &str) -> Option<bytes::Bytes> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s.as_bytes()).ok().map(Into::into)
}

/// Suppress an unused-import lint for `OpaqueBlob` (used only via type inference elsewhere).
#[allow(dead_code)]
type _Blob = OpaqueBlob;
