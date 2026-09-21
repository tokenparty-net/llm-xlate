//! `encode_request`: [`IrRequest`] → an OpenAI Responses request body (capability-gated;
//! every lossy step records a [`Degradation`]), plus `request_echo` for the response envelope.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use llm_xlate_core::caps::SamplingRule;
use llm_xlate_core::{
    canon, Capabilities, Degradations, EncodeCtx, EncodedRequest, Extensions, HeaderMap, Instruction,
    InstructionRole, IrRequest, Item, MediaSource, OutputConfig, OutputFormat, Part, Position,
    ProviderFamily, ReasoningExposure, Role, StopReason, SummaryLevel, ToolChoice,
    ToolDef, Verbosity, XlateError,
};
use llm_xlate_core::caps::ToolChoiceKind;

use crate::render::detail_lookup;
use crate::util::Ob;

/// Encode an IR request into a Responses request body + context.
pub(crate) fn encode_request(
    req: &IrRequest,
    caps: &Capabilities,
    ctx: &EncodeCtx,
) -> Result<EncodedRequest, XlateError> {
    let mut degr = Degradations::new();

    // Function-tool support gate.
    let has_function_tool = req.tools.iter().any(|t| matches!(t, ToolDef::Function { .. }));
    if has_function_tool && !caps.tools.function_tools.is_yes() {
        return Err(XlateError::unsupported("tools", "function tools not supported by target"));
    }

    let mut top = Ob::new().set("model", Value::from(req.model.upstream().to_string()));

    // Instruction placement.
    let plan = instruction_plan(req);
    if plan.use_string {
        if let Some(ins) = plan.leading.first() {
            top = top.set("instructions", Value::from(ins.text()));
        }
    }

    // Input array (items with instructions interleaved when not using the string form).
    let input = build_input(req, &plan, caps, &mut degr)?;
    top = top.set("input", Value::Array(input));

    // Tools.
    if !req.tools.is_empty() {
        if let Some(tools) = build_tools_value(req, Some(caps), Some(&mut degr)) {
            top = top.set("tools", tools);
        }
    }

    // tool_choice.
    if let Some(tc) = build_tool_choice_value(req, caps, &mut degr) {
        top = top.set("tool_choice", tc);
    }

    if let Some(b) = req.parallel_tool_calls {
        top = top.set("parallel_tool_calls", Value::Bool(b));
    }

    // text {format, verbosity}.
    if let Some(text) = build_text_value(req, Some(caps), Some(&mut degr)) {
        top = top.set("text", text);
    }

    // reasoning.
    if let Some(r) = build_reasoning_value(req, Some(caps), Some(&mut degr)) {
        top = top.set("reasoning", r);
    }

    if let Some(n) = req.limits.max_output_tokens {
        top = top.set("max_output_tokens", Value::from(n));
    }

    // Sampling (drop + degrade when out of range / rejected / ignored).
    top = emit_sampling_f64(top, "temperature", req.sampling.temperature, caps.sampling.temperature, &mut degr);
    top = emit_sampling_f64(top, "top_p", req.sampling.top_p, caps.sampling.top_p, &mut degr);
    top = emit_sampling_f64(
        top,
        "top_logprobs",
        req.sampling.top_logprobs.map(|n| n as f64),
        caps.sampling.logprobs,
        &mut degr,
    );

    // Sampling / limits fields the Responses wire cannot carry (they routinely arrive when
    // translating from Chat). Each set field is dropped with a Degradation. (`n>1` rejection and
    // `n==1` no-op live in the facade's `lower()`, so `n` is not touched here.)
    if req.sampling.frequency_penalty.is_some() {
        degr.dropped("frequency_penalty", "Responses API has no frequency_penalty");
    }
    if req.sampling.presence_penalty.is_some() {
        degr.dropped("presence_penalty", "Responses API has no presence_penalty");
    }
    if req.sampling.seed.is_some() {
        degr.dropped("seed", "Responses API has no seed");
    }
    if req.sampling.top_k.is_some() {
        degr.dropped("top_k", "Responses API has no top_k");
    }
    if req.sampling.logit_bias.is_some() {
        degr.dropped("logit_bias", "Responses API has no logit_bias");
    }
    if !req.limits.stop_sequences.is_empty() {
        degr.dropped("stop_sequences", "Responses API has no stop parameter");
    }

    // Stateless / encrypted-reasoning policy.
    let want_stateless = req.state.store == Some(false)
        || caps.state.store.is_no()
        || caps.state.zdr.is_yes();
    let store_value = if want_stateless { Some(false) } else { req.state.store };
    let mut include = req.state.include.clone();
    if want_stateless && !include.iter().any(|s| s == "reasoning.encrypted_content") {
        include.push("reasoning.encrypted_content".to_string());
    }

    if let Some(s) = store_value {
        top = top.set("store", Value::Bool(s));
    }
    if let Some(id) = &req.state.previous_response_id {
        if caps.state.previous_response_id.is_yes() {
            top = top.set("previous_response_id", Value::from(id.as_str().to_string()));
        } else {
            degr.dropped("previous_response_id", "target does not support response chaining");
        }
    }
    if let Some(c) = &req.state.conversation {
        if caps.state.conversation_api.is_yes() {
            top = top.set("conversation", Value::from(c.clone()));
        } else {
            degr.dropped("conversation", "target does not support the conversation API");
        }
    }
    if let Some(b) = req.state.background {
        if caps.state.background.is_yes() {
            top = top.set("background", Value::Bool(b));
        } else if b {
            degr.dropped("background", "target does not support background mode");
        }
    }
    if !include.is_empty() {
        top = top.set("include", Value::Array(include.iter().map(|s| Value::from(s.clone())).collect()));
    }

    // Metadata / identifiers.
    if !req.meta.metadata.is_empty() {
        top = top.set("metadata", Value::Object(req.meta.metadata.clone()));
    }
    if let Some(u) = &req.meta.user {
        top = top.set("user", Value::from(fit_identifier(u, "user", &mut degr)));
    }
    if let Some(s) = &req.meta.safety_identifier {
        top = top.set("safety_identifier", Value::from(fit_identifier(s, "safety_identifier", &mut degr)));
    }
    if let Some(t) = &req.meta.service_tier {
        if caps.sampling.service_tier.as_ref().is_some_and(|list| list.iter().any(|x| x == t)) {
            top = top.set("service_tier", Value::from(t.clone()));
        } else {
            degr.dropped("service_tier", format!("service tier `{t}` not supported by target"));
        }
    }

    // Cache key.
    if let Some(k) = &req.cache.prompt_cache_key {
        if caps.cache.prompt_cache_key.is_yes() {
            top = top.set("prompt_cache_key", Value::from(k.clone()));
        } else {
            degr.dropped("prompt_cache_key", "target does not support prompt_cache_key");
        }
    }

    if req.stream {
        top = top.set("stream", Value::Bool(true));
    }

    // ext writeback: responses.* keys not already present (image_detail is synthetic).
    ext_writeback(&mut top, &req.ext);

    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().unwrap());

    // session affinity: emit the captured id into the first accepted place (field or header).
    let mut body_map = top.into_map();
    llm_xlate_core::session::apply_session(
        caps,
        req.session.id.as_deref(),
        req.cache.prompt_cache_key.as_deref(),
        &mut body_map,
        &mut headers,
        &mut degr,
    );
    let body = canon::to_bytes(&Value::Object(body_map));

    let mut out_ctx = ctx.clone();
    out_ctx.store = store_value;
    out_ctx.include = include;
    out_ctx.expose = req.reasoning.expose.clone();
    out_ctx.stream = req.stream;

    Ok(EncodedRequest {
        body,
        headers,
        upstream_streams: req.stream,
        ctx: out_ctx,
        degradations: degr,
    })
}

/// Where each instruction goes on encode.
pub(crate) struct InstrPlan<'a> {
    /// Emit a top-level `instructions` string from the single leading instruction.
    pub use_string: bool,
    /// Leading instructions in order.
    pub leading: Vec<&'a Instruction>,
    /// `Before(i)` instructions grouped by item index.
    pub before: BTreeMap<usize, Vec<&'a Instruction>>,
}

/// Compute the instruction placement plan (shared by encode and `request_echo`).
pub(crate) fn instruction_plan(req: &IrRequest) -> InstrPlan<'_> {
    let mut leading = Vec::new();
    let mut before: BTreeMap<usize, Vec<&Instruction>> = BTreeMap::new();
    for ins in &req.instructions {
        match ins.position {
            Position::Leading => leading.push(ins),
            Position::Before(i) => before.entry(i).or_default().push(ins),
        }
    }
    let text_only = |ins: &Instruction| ins.content.iter().all(|p| matches!(p, Part::Text { .. }));
    // Only a single leading *System* instruction becomes the top-level `instructions` string; a
    // leading Developer instruction must render as a `{type:"message",role:"developer"}` item
    // (plan §7.1) so its role survives the round trip.
    let use_string = leading.len() == 1
        && leading[0].role == InstructionRole::System
        && text_only(leading[0])
        && !before.contains_key(&0)
        && leading[0].effort.is_none()
        && leading[0].clear_at.is_none();
    InstrPlan { use_string, leading, before }
}

/// Build the `input` array with instructions interleaved.
fn build_input(
    req: &IrRequest,
    plan: &InstrPlan,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Result<Vec<Value>, XlateError> {
    let mut input = Vec::new();
    let n = req.items.len();
    for idx in 0..=n {
        if idx == 0 && !plan.use_string {
            for ins in &plan.leading {
                input.push(encode_instruction_item(ins, degr));
            }
        }
        // At the final slot, also flush any anchor pointing past the end. `lower` clamps and
        // reports such an anchor; placing it here keeps a hand-built request from silently
        // losing the instruction.
        if idx == n {
            for (_, list) in plan.before.range(idx..) {
                for ins in list {
                    input.push(encode_instruction_item(ins, degr));
                }
            }
        } else if let Some(list) = plan.before.get(&idx) {
            for ins in list {
                input.push(encode_instruction_item(ins, degr));
            }
        }
        if idx < n {
            if let Some(v) = encode_item(&req.items[idx], idx, req, caps, degr)? {
                input.push(v);
            }
        }
    }
    Ok(input)
}

/// Encode a leading/mid instruction as a `message` item.
fn encode_instruction_item(ins: &Instruction, degr: &mut Degradations) -> Value {
    if ins.effort.is_some() {
        degr.dropped("instructions.effort", "Responses has no mid-conversation effort override");
    }
    if ins.clear_at.is_some() {
        degr.dropped("instructions.clear_at", "Responses has no system_clear_at");
    }
    let role = match ins.role {
        InstructionRole::Developer => "developer",
        InstructionRole::System => "system",
    };
    let content: Vec<Value> = ins
        .content
        .iter()
        .filter_map(|p| match p {
            Part::Text { text, .. } => {
                Some(Ob::new().set("type", "input_text".into()).set("text", Value::from(text.clone())).build())
            }
            _ => None,
        })
        .collect();
    Ob::new().set("type", "message".into()).set("role", Value::from(role)).set("content", Value::Array(content)).build()
}

/// Encode one IR item as a Responses input item (returns `None` when the item is dropped).
fn encode_item(
    item: &Item,
    idx: usize,
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Result<Option<Value>, XlateError> {
    match item {
        Item::Message { role: Role::User, content, .. } => {
            let parts = encode_user_content(content, idx, &req.ext, caps, degr);
            Ok(Some(Ob::new().set("type", "message".into()).set("role", "user".into()).set("content", Value::Array(parts)).build()))
        }
        Item::Message { role: Role::Assistant, content, id } => {
            let parts = encode_assistant_content(content);
            let mut ob = Ob::new().set("type", "message".into()).set("role", "assistant".into());
            if let Some(i) = id {
                ob = ob.set("id", Value::from(i.as_str().to_string()));
            }
            Ok(Some(ob.set("content", Value::Array(parts)).build()))
        }
        Item::ToolCall { call_id, name, arguments, id } => {
            let mut ob = Ob::new()
                .set("type", "function_call".into())
                .set("call_id", Value::from(call_id.as_str().to_string()))
                .set("name", Value::from(name.clone()))
                .set("arguments", Value::from(arguments.as_str().to_string()));
            if let Some(i) = id {
                ob = ob.set("id", Value::from(i.as_str().to_string()));
            }
            Ok(Some(ob.build()))
        }
        Item::ToolResult { call_id, content, .. } => {
            let output = encode_tool_result_output(content, call_id.as_str(), caps, degr);
            Ok(Some(
                Ob::new()
                    .set("type", "function_call_output".into())
                    .set("call_id", Value::from(call_id.as_str().to_string()))
                    .set("output", output)
                    .build(),
            ))
        }
        Item::Reasoning(ri) => {
            match &ri.opaque {
                Some(blob) if blob.family == ProviderFamily::OpenAI => {
                    let summary: Vec<Value> = ri
                        .summary
                        .iter()
                        .map(|s| Ob::new().set("type", "summary_text".into()).set("text", Value::from(s.clone())).build())
                        .collect();
                    let mut ob = Ob::new().set("type", "reasoning".into());
                    if let Some(i) = &ri.id {
                        ob = ob.set("id", Value::from(i.as_str().to_string()));
                    }
                    ob = ob.set("summary", Value::Array(summary));
                    ob = ob.set("encrypted_content", Value::from(blob.data.clone()));
                    Ok(Some(ob.build()))
                }
                Some(_) => {
                    degr.dropped("reasoning", "foreign reasoning blob cannot be replayed on Responses");
                    Ok(None)
                }
                None => {
                    // Server-side replay: only when the id is known and store is available.
                    match (&ri.id, caps.state.store.is_yes()) {
                        (Some(id), true) => {
                            let summary: Vec<Value> = ri
                                .summary
                                .iter()
                                .map(|s| Ob::new().set("type", "summary_text".into()).set("text", Value::from(s.clone())).build())
                                .collect();
                            let ob = Ob::new()
                                .set("type", "reasoning".into())
                                .set("id", Value::from(id.as_str().to_string()))
                                .set("summary", Value::Array(summary));
                            Ok(Some(ob.build()))
                        }
                        _ => {
                            degr.dropped("reasoning", "reasoning item without replay carrier dropped");
                            Ok(None)
                        }
                    }
                }
            }
        }
        Item::ProviderToolCall(oi) | Item::ProviderToolResult(oi) => {
            if oi.family == ProviderFamily::OpenAI {
                Ok(Some(oi.raw.clone()))
            } else {
                degr.dropped("provider_tool", "foreign provider tool item dropped");
                Ok(None)
            }
        }
        Item::Compaction(blob) => {
            if blob.family == ProviderFamily::OpenAI {
                Ok(Some(
                    Ob::new()
                        .set("type", "compaction".into())
                        .set("encrypted_content", Value::from(blob.data.clone()))
                        .build(),
                ))
            } else {
                degr.dropped("compaction", "foreign compaction block dropped");
                Ok(None)
            }
        }
    }
}

/// Encode user-message content parts.
fn encode_user_content(
    content: &[Part],
    item_idx: usize,
    ext: &Extensions,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (part_idx, p) in content.iter().enumerate() {
        // image_url.detail dropped when the target lacks the param.
        let detail = detail_lookup(ext, item_idx, part_idx);
        let detail_ref = if matches!(p, Part::Image(_)) && caps.media.image_detail_param.is_yes() {
            detail.as_deref()
        } else {
            if detail.is_some() {
                degr.dropped("image_url.detail", "target does not support image detail");
            }
            None
        };
        if let Part::Audio(_) = p {
            if !caps.media.audio.allows(llm_xlate_core::caps::MediaSourceKind::Base64)
                && !caps.media.audio.allows(llm_xlate_core::caps::MediaSourceKind::Url)
            {
                degr.dropped("audio", "target does not support audio input");
                continue;
            }
        }
        if let Some(v) = crate::render::render_input_part(p, None, detail_ref) {
            out.push(v);
        }
    }
    out
}

/// Encode assistant-message content (output_text / refusal, annotations empty on input).
fn encode_assistant_content(content: &[Part]) -> Vec<Value> {
    let mut out = Vec::new();
    for p in content {
        match p {
            Part::Text { text, .. } => out.push(
                Ob::new()
                    .set("type", "output_text".into())
                    .set("text", Value::from(text.clone()))
                    .set("annotations", Value::Array(Vec::new()))
                    .build(),
            ),
            Part::Refusal { text } => out.push(
                Ob::new().set("type", "refusal".into()).set("refusal", Value::from(text.clone())).build(),
            ),
            _ => {}
        }
    }
    out
}

/// Encode a tool result `output`: a bare string when text-only, else an array of parts.
fn encode_tool_result_output(
    content: &[Part],
    call_id: &str,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Value {
    let all_text = content.iter().all(|p| matches!(p, Part::Text { .. }));
    if all_text {
        let joined: String = content.iter().filter_map(Part::as_text).collect();
        return Value::String(joined);
    }
    let allow_image = caps
        .tools
        .result_content
        .as_ref()
        .is_some_and(|k| k.contains(&llm_xlate_core::caps::ResultContentKind::Image));
    let mut parts = Vec::new();
    for p in content {
        match p {
            Part::Text { text, .. } => {
                // A `function_call_output` is developer-supplied INPUT: the Responses API requires
                // `input_text` here and rejects `output_text` ("Invalid value: 'output_text'.
                // Supported values are: 'input_text'"). Verified live 2026-09-10 (translate1
                // tools_loop chat->responses). Only `output`/assistant messages use `output_text`.
                parts.push(Ob::new().set("type", "input_text".into()).set("text", Value::from(text.clone())).build());
            }
            Part::Image(src) if allow_image => {
                let v = match src {
                    MediaSource::Base64 { media_type, data } => {
                        let url = canon::data_url::build(media_type, data);
                        Ob::new().set("type", "input_image".into()).set("image_url", Value::from(url)).build()
                    }
                    MediaSource::Url(u) => {
                        Ob::new().set("type", "input_image".into()).set("image_url", Value::from(u.clone())).build()
                    }
                    MediaSource::FileRef { id, .. } => {
                        Ob::new().set("type", "input_image".into()).set("file_id", Value::from(id.clone())).build()
                    }
                    MediaSource::Text(t) => {
                        Ob::new().set("type", "input_image".into()).set("image_url", Value::from(t.clone())).build()
                    }
                };
                parts.push(v);
            }
            Part::Image(_) => {
                degr.folded(
                    "tool_result.image",
                    format!("image tool result folded to text for call {call_id}"),
                );
                parts.push(
                    Ob::new()
                        .set("type", "input_text".into())
                        .set("text", Value::from(llm_xlate_core::wrap::tool_result_attachment(call_id)))
                        .build(),
                );
            }
            _ => {}
        }
    }
    Value::Array(parts)
}

/// Build the `tools` array. `caps == None` disables gating (used by `request_echo`).
pub(crate) fn build_tools_value(
    req: &IrRequest,
    caps: Option<&Capabilities>,
    mut degr: Option<&mut Degradations>,
) -> Option<Value> {
    if req.tools.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for (i, t) in req.tools.iter().enumerate() {
        match t {
            ToolDef::Function { name, description, parameters, strict, .. } => {
                let mut ob = Ob::new()
                    .set("type", "function".into())
                    .set("name", Value::from(name.clone()))
                    .opt_str("description", description.clone())
                    .set("parameters", parameters.clone());
                let strict_supported = caps.map(|c| c.tools.strict.supported.is_yes()).unwrap_or(true);
                if strict_supported {
                    if let Some(s) = strict {
                        ob = ob.set("strict", Value::Bool(*s));
                    }
                    // strict None with Attempted default → emit nothing.
                } else if strict.is_some() {
                    if let Some(d) = degr.as_deref_mut() {
                        d.downgraded(format!("tools[{i}].strict"), "target does not support strict schema");
                    }
                }
                out.push(ob.build());
            }
            ToolDef::Provider(oi) => {
                if oi.family == ProviderFamily::OpenAI {
                    out.push(oi.raw.clone());
                } else if let Some(d) = degr.as_deref_mut() {
                    d.dropped(format!("tools[{i}]"), "foreign provider tool dropped");
                }
            }
        }
    }
    Some(Value::Array(out))
}

/// Build `tool_choice`; degrades unsupported variants to `auto`.
fn build_tool_choice_value(
    req: &IrRequest,
    caps: &Capabilities,
    degr: &mut Degradations,
) -> Option<Value> {
    // A hosted / allowed_tools tool_choice preserved on decode wins verbatim.
    if let Some(raw) = req.ext.get("responses.tool_choice") {
        return Some(raw.clone());
    }
    let allowed = |k: ToolChoiceKind| caps.tools.tool_choice.as_ref().is_some_and(|v| v.contains(&k));
    match &req.tool_choice {
        ToolChoice::Auto => None,
        ToolChoice::None => {
            if allowed(ToolChoiceKind::None) {
                Some(Value::from("none"))
            } else {
                degr.downgraded("tool_choice", "`none` not supported; using auto");
                None
            }
        }
        ToolChoice::Required => {
            if allowed(ToolChoiceKind::Required) {
                Some(Value::from("required"))
            } else {
                degr.downgraded("tool_choice", "`required` not supported; using auto");
                None
            }
        }
        ToolChoice::Named(name) => {
            if allowed(ToolChoiceKind::Named) {
                Some(Ob::new().set("type", "function".into()).set("name", Value::from(name.clone())).build())
            } else {
                degr.downgraded("tool_choice", "named tool choice not supported; using auto");
                None
            }
        }
    }
}

/// Build `text` {format, verbosity}. `caps == None` disables gating.
#[allow(clippy::needless_option_as_deref)]
pub(crate) fn build_text_value(
    req: &IrRequest,
    caps: Option<&Capabilities>,
    mut degr: Option<&mut Degradations>,
) -> Option<Value> {
    let has_format = !matches!(req.output.format, OutputFormat::Text);
    let verbosity_supported = caps.map(|c| c.output.verbosity.is_yes()).unwrap_or(true);
    let want_verbosity = req.output.verbosity.is_some();
    if !has_format && !want_verbosity {
        return None;
    }
    let mut ob = Ob::new();
    let format = build_output_format(&req.output, caps, degr.as_deref_mut());
    ob = ob.set("format", format);
    if let Some(v) = &req.output.verbosity {
        if verbosity_supported {
            let s = match v {
                Verbosity::Low => "low",
                Verbosity::Medium => "medium",
                Verbosity::High => "high",
            };
            ob = ob.set("verbosity", Value::from(s));
        } else if let Some(d) = degr.as_deref_mut() {
            d.dropped("verbosity", "target does not support verbosity");
        }
    }
    Some(ob.build())
}

/// Build the `text.format` object.
fn build_output_format(
    output: &OutputConfig,
    caps: Option<&Capabilities>,
    degr: Option<&mut Degradations>,
) -> Value {
    match &output.format {
        OutputFormat::Text => Ob::new().set("type", "text".into()).build(),
        OutputFormat::JsonObject => Ob::new().set("type", "json_object".into()).build(),
        OutputFormat::JsonSchema { name, schema, strict, description } => {
            let strict_supported = caps.map(|c| c.output.strict_supported.is_yes()).unwrap_or(true);
            let effective_strict = *strict && strict_supported;
            if *strict && !strict_supported {
                if let Some(d) = degr {
                    d.downgraded("text.format.strict", "target does not support strict structured output");
                }
            }
            Ob::new()
                .set("type", "json_schema".into())
                .set("name", Value::from(name.clone()))
                .opt_str("description", description.clone())
                .set("schema", schema.clone())
                .set("strict", Value::Bool(effective_strict))
                .build()
        }
    }
}

/// Build the `reasoning` {effort, summary} object. `caps == None` disables gating.
#[allow(clippy::needless_option_as_deref)]
pub(crate) fn build_reasoning_value(
    req: &IrRequest,
    caps: Option<&Capabilities>,
    mut degr: Option<&mut Degradations>,
) -> Option<Value> {
    let cfg = &req.reasoning;
    // Request-side reasoning is emitted only on genuine client intent (effort / explicit enable /
    // budget). `expose` is a response-path rendering preference — the Chat and Anthropic decoders
    // hard-set it to `Full` so their clients always see reasoning text — and must NOT by itself
    // request reasoning summaries on the Responses request (plan §7.2: "if a client sends no
    // reasoning config, send nothing").
    let want = cfg.effort.is_some() || cfg.enabled == Some(true) || cfg.budget_tokens.is_some();
    if !want {
        return None;
    }
    let mut ob = Ob::new();
    if let Some(e) = cfg.effort {
        let token = e.openai_token();
        if e == llm_xlate_core::Effort::Max {
            let max_listed =
                caps.and_then(|c| c.reasoning.effort_levels.as_ref()).is_some_and(|l| l.contains(&llm_xlate_core::Effort::Max));
            if !max_listed {
                if let Some(d) = degr.as_deref_mut() {
                    d.downgraded("reasoning.effort", "`max` unavailable on Responses; using xhigh");
                }
            }
        }
        ob = ob.set("effort", Value::from(token));
    }
    if let Some(s) = summary_token(&cfg.expose) {
        ob = ob.set("summary", Value::from(s));
    }
    if ob.is_empty() {
        None
    } else {
        Some(ob.build())
    }
}

/// The `reasoning.summary` token for a reasoning exposure — emitted only on genuine client summary
/// intent.
///
/// A `Summary(_)` exposure comes only from a Responses client that explicitly asked for a summary
/// (or that sent an `effort`, which the decoder maps to `auto`). `Full` is the *decoder default*
/// the Chat and Anthropic decoders hard-set so their clients always see reasoning text — a
/// response-path rendering preference, not a request-side summary request. The Responses request
/// API has no "full" concept, so `Full` must never synthesize a `summary` field (nor the
/// misleading `reasoning.summary` downgrade it used to): a Chat client sending only
/// `reasoning_effort:"low"` must not be forced into `summary:"detailed"` (plan §7.2).
fn summary_token(expose: &ReasoningExposure) -> Option<&'static str> {
    match expose {
        ReasoningExposure::None | ReasoningExposure::Full => None,
        ReasoningExposure::Summary(level) => Some(match level {
            SummaryLevel::Auto => "auto",
            SummaryLevel::Concise => "concise",
            SummaryLevel::Detailed => "detailed",
        }),
    }
}

/// Emit a sampling `f64` field when the rule accepts it, else drop + degrade.
fn emit_sampling_f64(
    top: Ob,
    field: &str,
    value: Option<f64>,
    rule: Option<SamplingRule>,
    degr: &mut Degradations,
) -> Ob {
    let Some(v) = value else { return top };
    match rule {
        Some(r) if r.accepts(v) => {
            if field == "top_logprobs" {
                top.set(field, Value::from(v as u64))
            } else {
                top.set(field, Value::from(v))
            }
        }
        _ => {
            degr.dropped(field.to_string(), "value not accepted by target sampling rule");
            top
        }
    }
}

/// Write `responses.*` ext entries back at the top level (verbatim), skipping the synthetic
/// `image_detail` key and any field already emitted.
fn ext_writeback(top: &mut Ob, ext: &Extensions) {
    let mut current = std::mem::replace(top, Ob::new()).into_map();
    for (k, v) in ext.iter() {
        if let Some(field) = k.strip_prefix("responses.") {
            if field == "image_detail" {
                continue;
            }
            if !current.contains_key(field) {
                current.insert(field.to_string(), v.clone());
            }
        }
    }
    *top = rebuild(current);
}

fn rebuild(map: Map<String, Value>) -> Ob {
    let mut ob = Ob::new();
    for (k, v) in map {
        ob = ob.set_owned(k, v);
    }
    ob
}

/// Build the `request_echo` map the response envelope echoes (client-side, no gating).
pub(crate) fn request_echo(req: &IrRequest) -> Map<String, Value> {
    let mut ob = Ob::new();
    let plan = instruction_plan(req);
    if plan.use_string {
        ob = ob.set("instructions", Value::from(plan.leading[0].text()));
    } else {
        ob = ob.set("instructions", Value::Null);
    }
    ob = ob.set("max_output_tokens", req.limits.max_output_tokens.map(Value::from).unwrap_or(Value::Null));
    ob = ob.set(
        "metadata",
        if req.meta.metadata.is_empty() { Value::Object(Map::new()) } else { Value::Object(req.meta.metadata.clone()) },
    );
    ob = ob.set("parallel_tool_calls", Value::Bool(req.parallel_tool_calls.unwrap_or(true)));
    ob = ob.set(
        "previous_response_id",
        req.state.previous_response_id.as_ref().map(|i| Value::from(i.as_str().to_string())).unwrap_or(Value::Null),
    );
    ob = ob.set("reasoning", build_reasoning_value(req, None, None).unwrap_or(Value::Null));
    ob = ob.set("service_tier", req.meta.service_tier.clone().map(Value::from).unwrap_or(Value::Null));
    ob = ob.set("store", req.state.store.map(Value::Bool).unwrap_or(Value::Null));
    ob = ob.set("temperature", req.sampling.temperature.map(Value::from).unwrap_or(Value::Null));
    ob = ob.set("text", build_text_value(req, None, None).unwrap_or_else(|| Ob::new().set("format", Ob::new().set("type", "text".into()).build()).build()));
    ob = ob.set(
        "tool_choice",
        build_tool_choice_echo(req),
    );
    ob = ob.set("tools", build_tools_value(req, None, None).unwrap_or(Value::Array(Vec::new())));
    ob = ob.set("top_p", req.sampling.top_p.map(Value::from).unwrap_or(Value::Null));
    ob = ob.set(
        "truncation",
        req.ext.get("responses.truncation").cloned().unwrap_or(Value::from("disabled")),
    );
    ob = ob.set("background", req.state.background.map(Value::Bool).unwrap_or(Value::Null));
    ob = ob.set("user", req.meta.user.clone().map(Value::from).unwrap_or(Value::Null));
    ob = ob.set("safety_identifier", req.meta.safety_identifier.clone().map(Value::from).unwrap_or(Value::Null));
    ob.into_map()
}

fn build_tool_choice_echo(req: &IrRequest) -> Value {
    if let Some(raw) = req.ext.get("responses.tool_choice") {
        return raw.clone();
    }
    match &req.tool_choice {
        ToolChoice::Auto => Value::from("auto"),
        ToolChoice::None => Value::from("none"),
        ToolChoice::Required => Value::from("required"),
        ToolChoice::Named(name) => {
            Ob::new().set("type", "function".into()).set("name", Value::from(name.clone())).build()
        }
    }
}

/// Suppress an unused warning for `StopReason` (kept for symmetry with response.rs imports).
#[allow(dead_code)]
type _Stop = StopReason;

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
