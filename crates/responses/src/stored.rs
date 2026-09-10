//! Store-support helpers (plan §9). The facade owns the persisted `StoredResponse` type and
//! maps it to a [`StoredView`]; these functions render the two stored-response GET surfaces:
//!
//! * [`encode_stored_response`] — `GET /v1/responses/{id}`: the full `response` object at its
//!   persisted status, output items rendered exactly like [`crate::response::encode_response`]
//!   (opaque blobs re-sealed with the supplied [`Sealer`]), echoing the stored request params.
//! * [`encode_input_items`] — `GET /v1/responses/{id}/input_items`: the list of the request's
//!   input items (instructions rendered as leading `system`/`developer` message items), each
//!   with a minted `"{kind}_{response_id}_{index}"` id.

use bytes::Bytes;
use serde_json::{Map, Value};

use llm_xlate_core::{
    canon, Instruction, InstructionRole, Item, Part, ProviderFamily, ReasoningExposure, ResponseId,
    Role, Sealer, StopReason, Usage, XlateError,
};

use crate::render::{
    build_response_object, mint_id, render_input_part, render_output_item, RenderCtx,
};
use crate::util::Ob;

/// The persisted status of a stored response (the facade maps its `StoredResponse` status here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredStatus {
    /// Accepted, not yet started (background).
    Queued,
    /// Generating (background).
    InProgress,
    /// Finished normally.
    Completed,
    /// Stopped early (max tokens / content filter).
    Incomplete,
    /// Failed with an error.
    Failed,
    /// Cancelled.
    Cancelled,
}

impl StoredStatus {
    /// The wire status string.
    pub fn as_str(self) -> &'static str {
        match self {
            StoredStatus::Queued => "queued",
            StoredStatus::InProgress => "in_progress",
            StoredStatus::Completed => "completed",
            StoredStatus::Incomplete => "incomplete",
            StoredStatus::Failed => "failed",
            StoredStatus::Cancelled => "cancelled",
        }
    }
}

/// A borrowed, codec-facing view of a stored response. The facade builds this from its own
/// persisted `StoredResponse` (which stores opaque blobs **unwrapped**); the codec re-seals them.
pub struct StoredView<'a> {
    /// Client-facing response id.
    pub id: &'a ResponseId,
    /// The `previous_response_id` this response chained from, if any.
    pub previous_id: Option<&'a ResponseId>,
    /// Persisted status.
    pub status: StoredStatus,
    /// Router-supplied unix seconds.
    pub created_at: u64,
    /// Model string to echo.
    pub model: &'a str,
    /// The produced output items.
    pub output_items: &'a [Item],
    /// Final usage.
    pub usage: &'a Usage,
    /// The stop reason (drives `incomplete_details`).
    pub stop: &'a StopReason,
    /// The stored request params to echo (built by [`crate::encode::request_echo`]).
    pub request_echo: &'a Map<String, Value>,
    /// The error, when `status == Failed`.
    pub error: Option<&'a XlateError>,
    /// The request's input items (for the `input_items` listing).
    pub request_items: &'a [Item],
    /// The request's instructions (rendered as leading messages in the `input_items` listing).
    pub instructions: &'a [Instruction],
}

/// Render `GET /v1/responses/{id}` for a stored response.
pub fn encode_stored_response(v: &StoredView, sealer: &Sealer) -> Bytes {
    // Stored responses persist reasoning blobs unwrapped; re-seal on read and always surface
    // `encrypted_content` when present (superset of the original `include`).
    let include = vec!["reasoning.encrypted_content".to_string()];
    let rctx = RenderCtx {
        response_id: v.id.as_str(),
        created_at: v.created_at,
        model: v.model,
        request_echo: v.request_echo,
        sealer,
        expose: ReasoningExposure::None,
        include: &include,
        store: Some(false),
    };

    let output: Vec<Value> =
        v.output_items.iter().enumerate().map(|(i, it)| render_output_item(it, i, &rctx)).collect();

    let incomplete = match v.status {
        StoredStatus::Incomplete => incomplete_details(v.stop),
        _ => None,
    };
    let error = match v.status {
        StoredStatus::Failed => v.error.map(crate::errors::error_inner_value),
        _ => None,
    };
    let usage = match v.status {
        StoredStatus::Completed | StoredStatus::Incomplete => Some(v.usage),
        _ => None,
    };

    let mut obj = build_response_object(v.status.as_str(), output, usage, error, incomplete, &rctx);
    // The stored `previous_response_id` is authoritative over whatever the echo carried.
    if let Value::Object(m) = &mut obj {
        let pid = v.previous_id.map(|p| Value::from(p.as_str().to_string())).unwrap_or(Value::Null);
        m.insert("previous_response_id".to_string(), pid);
    }
    canon::to_bytes(&obj)
}

/// The `incomplete_details` object for a stop reason (max tokens / content filter).
fn incomplete_details(stop: &StopReason) -> Option<Value> {
    match stop {
        StopReason::MaxTokens => Some(Ob::new().set("reason", "max_output_tokens".into()).build()),
        StopReason::ContentFilter => Some(Ob::new().set("reason", "content_filter".into()).build()),
        _ => None,
    }
}

/// Render `GET /v1/responses/{id}/input_items` for a stored response.
pub fn encode_input_items(v: &StoredView) -> Bytes {
    let response_id = v.id.as_str();
    let mut data: Vec<Value> = Vec::new();

    // Leading instructions first, then each item with its `Before(i)` instructions.
    let leading: Vec<&Instruction> = v
        .instructions
        .iter()
        .filter(|i| matches!(i.position, llm_xlate_core::Position::Leading))
        .collect();
    let before = |idx: usize| -> Vec<&Instruction> {
        v.instructions
            .iter()
            .filter(move |i| matches!(i.position, llm_xlate_core::Position::Before(b) if b == idx))
            .collect()
    };

    for ins in &leading {
        let idx = data.len();
        data.push(render_instruction_item(ins, idx, response_id));
    }
    for (i, item) in v.request_items.iter().enumerate() {
        for ins in before(i) {
            let idx = data.len();
            data.push(render_instruction_item(ins, idx, response_id));
        }
        let idx = data.len();
        data.push(render_input_item(item, idx, response_id));
    }

    let first_id = data.first().and_then(item_id).map(Value::from).unwrap_or(Value::Null);
    let last_id = data.last().and_then(item_id).map(Value::from).unwrap_or(Value::Null);

    let obj = Ob::new()
        .set("object", "list".into())
        .set("data", Value::Array(data))
        .set("first_id", first_id)
        .set("last_id", last_id)
        .set("has_more", Value::Bool(false))
        .build();
    canon::to_bytes(&obj)
}

/// The `id` field of a rendered input item, if present.
fn item_id(v: &Value) -> Option<String> {
    v.get("id").and_then(Value::as_str).map(str::to_string)
}

/// Render an instruction as a leading `system`/`developer` message input item with a minted id.
fn render_instruction_item(ins: &Instruction, index: usize, response_id: &str) -> Value {
    let role = match ins.role {
        InstructionRole::Developer => "developer",
        InstructionRole::System => "system",
    };
    let content: Vec<Value> = ins
        .content
        .iter()
        .filter_map(|p| match p {
            Part::Text { text, .. } => Some(
                Ob::new().set("type", "input_text".into()).set("text", Value::from(text.clone())).build(),
            ),
            _ => None,
        })
        .collect();
    Ob::new()
        .set("id", Value::from(mint_id("msg", response_id, index)))
        .set("type", "message".into())
        .set("role", Value::from(role))
        .set("content", Value::Array(content))
        .build()
}

/// Render one request item as a Responses **input** item with a minted id (no capability gating;
/// this is a faithful listing, not a lowering).
fn render_input_item(item: &Item, index: usize, response_id: &str) -> Value {
    match item {
        Item::Message { role: Role::User, content, .. } => {
            let parts: Vec<Value> =
                content.iter().filter_map(|p| render_input_part(p, None, None)).collect();
            Ob::new()
                .set("id", Value::from(mint_id("msg", response_id, index)))
                .set("type", "message".into())
                .set("role", "user".into())
                .set("content", Value::Array(parts))
                .build()
        }
        Item::Message { role: Role::Assistant, content, .. } => {
            let parts: Vec<Value> = content
                .iter()
                .filter_map(|p| match p {
                    Part::Text { text, .. } => Some(
                        Ob::new()
                            .set("type", "output_text".into())
                            .set("text", Value::from(text.clone()))
                            .set("annotations", Value::Array(Vec::new()))
                            .build(),
                    ),
                    Part::Refusal { text } => Some(
                        Ob::new()
                            .set("type", "refusal".into())
                            .set("refusal", Value::from(text.clone()))
                            .build(),
                    ),
                    _ => None,
                })
                .collect();
            Ob::new()
                .set("id", Value::from(mint_id("msg", response_id, index)))
                .set("type", "message".into())
                .set("role", "assistant".into())
                .set("content", Value::Array(parts))
                .build()
        }
        Item::ToolCall { call_id, name, arguments, .. } => Ob::new()
            .set("id", Value::from(mint_id("fc", response_id, index)))
            .set("type", "function_call".into())
            .set("call_id", Value::from(call_id.as_str().to_string()))
            .set("name", Value::from(name.clone()))
            .set("arguments", Value::from(arguments.as_str().to_string()))
            .build(),
        Item::ToolResult { call_id, content, .. } => {
            let output = tool_result_output(content);
            Ob::new()
                .set("id", Value::from(mint_id("fc", response_id, index)))
                .set("type", "function_call_output".into())
                .set("call_id", Value::from(call_id.as_str().to_string()))
                .set("output", output)
                .build()
        }
        Item::Reasoning(ri) => {
            let summary: Vec<Value> = ri
                .summary
                .iter()
                .map(|s| Ob::new().set("type", "summary_text".into()).set("text", Value::from(s.clone())).build())
                .collect();
            let mut ob = Ob::new()
                .set("id", Value::from(mint_id("rs", response_id, index)))
                .set("type", "reasoning".into())
                .set("summary", Value::Array(summary));
            if let Some(blob) = &ri.opaque {
                if blob.family == ProviderFamily::OpenAI {
                    ob = ob.set("encrypted_content", Value::from(blob.data.clone()));
                }
            }
            ob.build()
        }
        Item::ProviderToolCall(oi) | Item::ProviderToolResult(oi) => oi.raw.clone(),
        Item::Compaction(blob) => Ob::new()
            .set("id", Value::from(mint_id("cmp", response_id, index)))
            .set("type", "compaction".into())
            .set("encrypted_content", Value::from(blob.data.clone()))
            .build(),
    }
}

/// A tool result `output`: a bare string when text-only, else an array of input parts.
fn tool_result_output(content: &[Part]) -> Value {
    if content.iter().all(|p| matches!(p, Part::Text { .. })) {
        let joined: String = content.iter().filter_map(Part::as_text).collect();
        return Value::String(joined);
    }
    let parts: Vec<Value> = content
        .iter()
        .filter_map(|p| match p {
            Part::Text { text, .. } => Some(
                Ob::new().set("type", "output_text".into()).set("text", Value::from(text.clone())).build(),
            ),
            other => render_input_part(other, None, None),
        })
        .collect();
    Value::Array(parts)
}
