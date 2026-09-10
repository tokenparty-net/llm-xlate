//! Shared constructors for the lower / requirements / store integration tests.

#![allow(dead_code)]

use llm_xlate::ir::{
    CallId, Instruction, IrRequest, IrResponse, Item, JsonText, MediaSource, OpaqueBlob,
    OpaqueItem, OpaqueKind, Part, Position, ProviderFamily, ReasoningItem, ResponseId, Role,
    StopReason, ToolDef, Usage,
};
use serde_json::{json, Value};

/// A user text message.
pub fn user(text: &str) -> Item {
    Item::user_text(text)
}

/// An assistant text message.
pub fn asst(text: &str) -> Item {
    Item::assistant_text(text)
}

/// An assistant tool call.
pub fn tool_call(call_id: &str, name: &str) -> Item {
    Item::ToolCall {
        call_id: CallId::new(call_id),
        name: name.to_string(),
        arguments: JsonText::new("{}"),
        id: None,
    }
}

/// A user-side tool result.
pub fn tool_result(call_id: &str, text: &str) -> Item {
    Item::ToolResult {
        call_id: CallId::new(call_id),
        content: vec![Part::text(text)],
        is_error: false,
        id: None,
    }
}

/// A reasoning item carrying an opaque blob of `family`.
pub fn reasoning_opaque(family: ProviderFamily) -> Item {
    Item::Reasoning(ReasoningItem {
        text: None,
        summary: Vec::new(),
        opaque: Some(OpaqueBlob::new(family, OpaqueKind::Signature, "sig-data")),
        id: None,
    })
}

/// A reasoning item with no opaque carrier (text-only).
pub fn reasoning_text() -> Item {
    Item::Reasoning(ReasoningItem {
        text: Some("thinking".to_string()),
        summary: Vec::new(),
        opaque: None,
        id: None,
    })
}

/// An opaque reasoning blob of `family` (for `Resolutions`).
pub fn blob(family: ProviderFamily) -> OpaqueBlob {
    OpaqueBlob::new(family, OpaqueKind::Signature, "resolved-sig")
}

/// A user message carrying an image `FileRef`.
pub fn img_fileref(family: ProviderFamily, id: &str) -> Item {
    Item::Message {
        role: Role::User,
        content: vec![Part::Image(MediaSource::FileRef { family, id: id.to_string() })],
        id: None,
    }
}

/// A user message carrying a base64 image of `nbytes` bytes.
pub fn img_bytes(nbytes: usize) -> Item {
    Item::Message {
        role: Role::User,
        content: vec![Part::Image(MediaSource::Base64 {
            media_type: "image/png".to_string(),
            data: bytes::Bytes::from(vec![0u8; nbytes]),
        })],
        id: None,
    }
}

/// A user message carrying a base64 PDF document of `nbytes` bytes.
pub fn pdf_bytes(nbytes: usize) -> Item {
    Item::Message {
        role: Role::User,
        content: vec![Part::Document {
            source: MediaSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: bytes::Bytes::from(vec![0u8; nbytes]),
            },
            title: None,
            media_type: "application/pdf".to_string(),
        }],
        id: None,
    }
}

/// A user message carrying an audio part.
pub fn audio_msg() -> Item {
    Item::Message {
        role: Role::User,
        content: vec![Part::Audio(MediaSource::Base64 {
            media_type: "audio/wav".to_string(),
            data: bytes::Bytes::from_static(b"wavdata"),
        })],
        id: None,
    }
}

/// A provider-hosted tool definition of `family`, named `name`.
pub fn provider_tool(family: ProviderFamily, name: &str) -> ToolDef {
    ToolDef::Provider(OpaqueItem::new(family, json!({ "type": name, "name": name })))
}

/// A function tool with the given JSON-schema parameters and `strict` flag.
pub fn function_tool(name: &str, parameters: Value, strict: Option<bool>) -> ToolDef {
    ToolDef::Function {
        name: name.to_string(),
        description: None,
        parameters,
        strict,
        cache_control: None,
    }
}

/// A provider tool-call history item of `family`.
pub fn provider_call(family: ProviderFamily, name: &str) -> Item {
    Item::ProviderToolCall(OpaqueItem::new(
        family,
        json!({ "type": name, "name": name, "id": "srvtool_1" }),
    ))
}

/// A provider tool-result history item of `family`, carrying `text`.
pub fn provider_result(family: ProviderFamily, name: &str, text: &str) -> Item {
    Item::ProviderToolResult(OpaqueItem::new(
        family,
        json!({ "type": name, "name": name, "content": [{ "text": text }] }),
    ))
}

/// A mid-conversation (`Before`) system instruction.
pub fn mid_instruction(idx: usize, text: &str) -> Instruction {
    let mut ins = Instruction::system_text(text);
    ins.position = Position::Before(idx);
    ins
}

/// A minimal `IrRequest` with the given items.
pub fn req_with_items(items: Vec<Item>) -> IrRequest {
    IrRequest { items, ..Default::default() }
}

/// A minimal successful `IrResponse`.
pub fn resp(items: Vec<Item>, stop: StopReason) -> IrResponse {
    IrResponse {
        id: ResponseId::new("resp_x"),
        model: "m".to_string(),
        items,
        stop,
        usage: Usage::new(10, 20),
        ext: Default::default(),
    }
}
