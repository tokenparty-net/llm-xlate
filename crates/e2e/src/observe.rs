//! Automatic, protocol-aware observations (plan §6), computed directly from the raw wire
//! JSON/SSE — deliberately *without* `llm-xlate`, so they are independent evidence against which
//! the crate's own decoding is later checked.
//!
//! For every capture we extract: HTTP status and any error `{type, code, param, message}`, the
//! request id, `retry-after` and rate-limit headers; for non-streaming responses the stop/finish
//! reason, flattened usage, the content-block / output-item **type sequence**, echoed request
//! fields (Responses), tool-call count, and the presence of a signature / encrypted content /
//! refusal; for streaming the ordered event-type sequence with indices, `sequence_number` gaps,
//! whether a usage-only final chunk exists, whether `[DONE]` was seen, and the ping count. Each
//! observation carries a verdict against the probe's `expect`.

use crate::capture::EventRecord;
use crate::probe::{Expect, Protocol};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// A verdict of the observed outcome against the probe's `expect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The outcome matched the expectation.
    Pass,
    /// The outcome contradicted the expectation.
    Fail,
    /// No expectation was set (`expect = "any"`).
    #[default]
    Na,
}

/// The error fields extracted from an error body.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorObs {
    /// Provider error `type`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Provider error `code`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Provider error `param`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Provider error `message`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl ErrorObs {
    fn is_empty(&self) -> bool {
        self.r#type.is_none() && self.code.is_none() && self.param.is_none() && self.message.is_none()
    }
}

/// The full set of automatic observations for one capture.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Observation {
    /// Protocol string.
    pub protocol: String,
    /// Whether the response was streamed.
    pub stream: bool,
    /// HTTP status.
    pub status: u16,
    /// Verdict against `expect`.
    pub verdict: Verdict,
    /// Detail when the verdict is `Fail`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict_detail: Option<String>,
    /// Extracted error, if the body carried one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorObs>,
    /// Upstream request id, from headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// `retry-after` header, if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<String>,
    /// Rate-limit headers (name → value).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub rate_limit: BTreeMap<String, String>,
    /// Stop / finish reason (or Responses status + incomplete reason).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Flattened usage counters (dotted keys).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub usage: BTreeMap<String, Value>,
    /// Content-block / output-item type sequence (non-stream).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub type_sequence: Vec<String>,
    /// Request top-level fields echoed back in the response (Responses).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub echoed_request_fields: Vec<String>,
    /// Number of tool calls in the response.
    pub tool_call_count: usize,
    /// Whether a reasoning signature was present.
    pub has_signature: bool,
    /// Whether encrypted reasoning content was present.
    pub has_encrypted_content: bool,
    /// Whether a refusal was present.
    pub has_refusal: bool,
    /// Ordered stream event-type sequence with indices (stream).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub event_sequence: Vec<String>,
    /// Gaps in `sequence_number` as `[prev, next]` pairs (stream, Responses).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sequence_number_gaps: Vec<[i64; 2]>,
    /// Whether a usage-only final chunk exists (stream).
    pub usage_only_final_chunk: bool,
    /// Whether a `[DONE]` sentinel was seen (stream).
    pub done_present: bool,
    /// Number of `ping` events (stream).
    pub ping_count: usize,
}

impl Observation {
    fn base(protocol: Protocol, stream: bool, status: u16) -> Self {
        Observation {
            protocol: protocol.as_str().to_string(),
            stream,
            status,
            verdict: Verdict::Na,
            verdict_detail: None,
            error: None,
            request_id: None,
            retry_after: None,
            rate_limit: BTreeMap::new(),
            stop_reason: None,
            usage: BTreeMap::new(),
            type_sequence: Vec::new(),
            echoed_request_fields: Vec::new(),
            tool_call_count: 0,
            has_signature: false,
            has_encrypted_content: false,
            has_refusal: false,
            event_sequence: Vec::new(),
            sequence_number_gaps: Vec::new(),
            usage_only_final_chunk: false,
            done_present: false,
            ping_count: 0,
        }
    }

    /// A bare observation with the given protocol string, for use in other modules' tests.
    #[cfg(test)]
    pub(crate) fn base_for_test(protocol: &str, stream: bool, status: u16) -> Self {
        let mut o = Observation::base(Protocol::Chat, stream, status);
        o.protocol = protocol.to_string();
        o
    }

    /// A one-line summary for the report table.
    pub fn summary_line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(e) = &self.error {
            if let Some(t) = &e.r#type {
                parts.push(format!("error={t}"));
            }
            if let Some(c) = &e.code {
                parts.push(format!("code={c}"));
            }
        }
        if let Some(sr) = &self.stop_reason {
            parts.push(format!("stop={sr}"));
        }
        if !self.type_sequence.is_empty() {
            parts.push(format!("blocks=[{}]", self.type_sequence.join(",")));
        }
        if self.tool_call_count > 0 {
            parts.push(format!("tools={}", self.tool_call_count));
        }
        if self.stream {
            parts.push(format!("events={}", self.event_sequence.len()));
            if self.done_present {
                parts.push("[DONE]".into());
            }
        }
        if self.has_signature {
            parts.push("sig".into());
        }
        if self.has_encrypted_content {
            parts.push("enc".into());
        }
        if self.has_refusal {
            parts.push("refusal".into());
        }
        if parts.is_empty() {
            "-".into()
        } else {
            parts.join(" ")
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Header extraction (shared).
// ---------------------------------------------------------------------------------------------

fn header_ci<'a>(headers: &'a BTreeMap<String, String>, name: &str) -> Option<&'a String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v)
}

fn extract_headers(obs: &mut Observation, headers: &BTreeMap<String, String>) {
    obs.request_id = header_ci(headers, "request-id")
        .or_else(|| header_ci(headers, "x-request-id"))
        .cloned();
    obs.retry_after = header_ci(headers, "retry-after").cloned();
    for (k, v) in headers {
        let lk = k.to_ascii_lowercase();
        if lk.starts_with("anthropic-ratelimit-") || lk.starts_with("x-ratelimit-") {
            obs.rate_limit.insert(lk, v.clone());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Error extraction (shared).
// ---------------------------------------------------------------------------------------------

fn extract_error(body: &Value) -> Option<ErrorObs> {
    let err = body.get("error")?;
    let obs = ErrorObs {
        r#type: str_field(err, "type"),
        code: str_field(err, "code"),
        param: str_field(err, "param"),
        message: str_field(err, "message"),
    };
    if obs.is_empty() {
        None
    } else {
        Some(obs)
    }
}

fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_string)
}

// ---------------------------------------------------------------------------------------------
// Usage flattening.
// ---------------------------------------------------------------------------------------------

fn flatten_usage(usage: &Value, prefix: &str, out: &mut BTreeMap<String, Value>) {
    if let Some(obj) = usage.as_object() {
        for (k, v) in obj {
            let key = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}.{k}")
            };
            match v {
                Value::Object(_) => flatten_usage(v, &key, out),
                Value::Number(_) | Value::Bool(_) => {
                    out.insert(key, v.clone());
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Verdict.
// ---------------------------------------------------------------------------------------------

fn apply_verdict(obs: &mut Observation, expect: &Expect) {
    match expect {
        Expect::Any => obs.verdict = Verdict::Na,
        Expect::Status(want) => {
            if obs.status == *want {
                obs.verdict = Verdict::Pass;
            } else {
                obs.verdict = Verdict::Fail;
                obs.verdict_detail = Some(format!("expected status {want}, got {}", obs.status));
            }
        }
        Expect::StatusAndType { status, error_type } => {
            let got_type = obs.error.as_ref().and_then(|e| e.r#type.clone());
            if obs.status == *status && got_type.as_deref() == Some(error_type.as_str()) {
                obs.verdict = Verdict::Pass;
            } else {
                obs.verdict = Verdict::Fail;
                obs.verdict_detail = Some(format!(
                    "expected {status}/{error_type}, got {}/{}",
                    obs.status,
                    got_type.as_deref().unwrap_or("<none>")
                ));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Non-streaming.
// ---------------------------------------------------------------------------------------------

/// Observe a non-streaming response.
pub fn observe_nonstream(
    protocol: Protocol,
    request_body: &Value,
    status: u16,
    headers: &BTreeMap<String, String>,
    body: &Value,
    expect: &Expect,
) -> Observation {
    let mut obs = Observation::base(protocol, false, status);
    extract_headers(&mut obs, headers);
    obs.error = extract_error(body);

    if obs.error.is_none() {
        match protocol {
            Protocol::Anthropic => nonstream_anthropic(&mut obs, body),
            Protocol::Chat => nonstream_chat(&mut obs, body),
            Protocol::Responses => nonstream_responses(&mut obs, request_body, body),
        }
    }
    apply_verdict(&mut obs, expect);
    obs
}

fn nonstream_anthropic(obs: &mut Observation, body: &Value) {
    obs.stop_reason = str_field(body, "stop_reason");
    if let Some(u) = body.get("usage") {
        flatten_usage(u, "", &mut obs.usage);
    }
    if let Some(blocks) = body.get("content").and_then(|c| c.as_array()) {
        for b in blocks {
            if let Some(t) = str_field(b, "type") {
                obs.type_sequence.push(t.clone());
                match t.as_str() {
                    "tool_use" | "server_tool_use" => obs.tool_call_count += 1,
                    "refusal" => obs.has_refusal = true,
                    "redacted_thinking" => obs.has_encrypted_content = true,
                    _ => {}
                }
            }
            if b.get("signature").is_some() {
                obs.has_signature = true;
            }
        }
    }
    if obs.stop_reason.as_deref() == Some("refusal") {
        obs.has_refusal = true;
    }
}

fn nonstream_chat(obs: &mut Observation, body: &Value) {
    if let Some(u) = body.get("usage") {
        flatten_usage(u, "", &mut obs.usage);
    }
    let Some(choice) = body.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()) else {
        return;
    };
    obs.stop_reason = str_field(choice, "finish_reason");
    let Some(msg) = choice.get("message") else {
        return;
    };
    if msg
        .get("content")
        .and_then(|c| c.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        obs.type_sequence.push("text".into());
    }
    if msg.get("refusal").and_then(|r| r.as_str()).is_some() {
        obs.type_sequence.push("refusal".into());
        obs.has_refusal = true;
    }
    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for _ in tcs {
            obs.type_sequence.push("tool_call".into());
        }
        obs.tool_call_count = tcs.len();
    }
}

fn nonstream_responses(obs: &mut Observation, request_body: &Value, body: &Value) {
    // Responses reports `status` (completed | incomplete | ...) rather than a stop reason.
    let mut stop = str_field(body, "status");
    if let Some(reason) = body
        .get("incomplete_details")
        .and_then(|d| d.get("reason"))
        .and_then(|r| r.as_str())
    {
        stop = Some(match stop {
            Some(s) => format!("{s}:{reason}"),
            None => reason.to_string(),
        });
    }
    obs.stop_reason = stop;
    if let Some(u) = body.get("usage") {
        flatten_usage(u, "", &mut obs.usage);
    }
    if let Some(items) = body.get("output").and_then(|o| o.as_array()) {
        for item in items {
            if let Some(t) = str_field(item, "type") {
                obs.type_sequence.push(t.clone());
                match t.as_str() {
                    "function_call" | "custom_tool_call" | "web_search_call" | "file_search_call" => {
                        obs.tool_call_count += 1
                    }
                    _ => {}
                }
            }
            if item.get("encrypted_content").is_some() {
                obs.has_encrypted_content = true;
            }
            if item.get("signature").is_some() {
                obs.has_signature = true;
            }
            // A message item may carry a refusal content part.
            if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                for p in parts {
                    if str_field(p, "type").as_deref() == Some("refusal") {
                        obs.has_refusal = true;
                    }
                }
            }
        }
    }
    // Echoed request fields: top-level request keys present with an equal value in the response.
    if let (Some(req), Some(resp)) = (request_body.as_object(), body.as_object()) {
        for (k, v) in req {
            if let Some(rv) = resp.get(k) {
                if rv == v {
                    obs.echoed_request_fields.push(k.clone());
                }
            }
        }
        obs.echoed_request_fields.sort();
    }
}

// ---------------------------------------------------------------------------------------------
// Streaming.
// ---------------------------------------------------------------------------------------------

/// Observe a streaming response. `raw` is the fallback for error bodies delivered as JSON when the
/// stream never opened (4xx/5xx).
pub fn observe_stream(
    protocol: Protocol,
    _request_body: &Value,
    status: u16,
    headers: &BTreeMap<String, String>,
    raw: &[u8],
    events: &[EventRecord],
    expect: &Expect,
) -> Observation {
    let mut obs = Observation::base(protocol, true, status);
    extract_headers(&mut obs, headers);

    // A non-2xx stream is usually a JSON error body, not SSE.
    if events.is_empty() {
        if let Ok(body) = serde_json::from_slice::<Value>(raw) {
            obs.error = extract_error(&body);
        }
        apply_verdict(&mut obs, expect);
        return obs;
    }

    match protocol {
        Protocol::Anthropic => stream_anthropic(&mut obs, events),
        Protocol::Chat => stream_chat(&mut obs, events),
        Protocol::Responses => stream_responses(&mut obs, events),
    }
    apply_verdict(&mut obs, expect);
    obs
}

fn event_json(e: &EventRecord) -> Option<Value> {
    if e.data == "[DONE]" {
        return None;
    }
    serde_json::from_str(&e.data).ok()
}

fn stream_anthropic(obs: &mut Observation, events: &[EventRecord]) {
    for e in events {
        let name = e.event.clone().unwrap_or_default();
        let data = event_json(e);
        match name.as_str() {
            "ping" => {
                obs.ping_count += 1;
                obs.event_sequence.push("ping".into());
            }
            "content_block_start" => {
                let idx = data
                    .as_ref()
                    .and_then(|d| d.get("index"))
                    .and_then(|i| i.as_i64())
                    .unwrap_or(-1);
                let bt = data
                    .as_ref()
                    .and_then(|d| d.get("content_block"))
                    .and_then(|b| str_field(b, "type"))
                    .unwrap_or_default();
                obs.event_sequence.push(format!("content_block_start[{idx}]:{bt}"));
                match bt.as_str() {
                    "tool_use" | "server_tool_use" => obs.tool_call_count += 1,
                    "redacted_thinking" => obs.has_encrypted_content = true,
                    _ => {}
                }
            }
            "content_block_delta" => {
                let idx = data
                    .as_ref()
                    .and_then(|d| d.get("index"))
                    .and_then(|i| i.as_i64())
                    .unwrap_or(-1);
                let dt = data
                    .as_ref()
                    .and_then(|d| d.get("delta"))
                    .and_then(|b| str_field(b, "type"))
                    .unwrap_or_default();
                if dt == "signature_delta" {
                    obs.has_signature = true;
                }
                obs.event_sequence.push(format!("content_block_delta[{idx}]:{dt}"));
            }
            "message_delta" => {
                if let Some(d) = &data {
                    if let Some(sr) = d.get("delta").and_then(|x| str_field(x, "stop_reason")) {
                        obs.stop_reason = Some(sr);
                    }
                    if let Some(u) = d.get("usage") {
                        flatten_usage(u, "", &mut obs.usage);
                    }
                }
                obs.event_sequence.push("message_delta".into());
            }
            "message_start" => {
                if let Some(u) = data
                    .as_ref()
                    .and_then(|d| d.get("message"))
                    .and_then(|m| m.get("usage"))
                {
                    flatten_usage(u, "", &mut obs.usage);
                }
                obs.event_sequence.push("message_start".into());
            }
            other => obs.event_sequence.push(other.to_string()),
        }
    }
    if obs.stop_reason.as_deref() == Some("refusal") {
        obs.has_refusal = true;
    }
}

fn stream_chat(obs: &mut Observation, events: &[EventRecord]) {
    let mut last_usage_only = false;
    for e in events {
        if e.data == "[DONE]" {
            obs.done_present = true;
            obs.event_sequence.push("[DONE]".into());
            continue;
        }
        let Some(data) = event_json(e) else { continue };
        let choices = data.get("choices").and_then(|c| c.as_array());
        let has_usage = data.get("usage").map(|u| !u.is_null()).unwrap_or(false);
        match choices {
            Some(arr) if arr.is_empty() => {
                if has_usage {
                    flatten_usage(data.get("usage").unwrap(), "", &mut obs.usage);
                    obs.event_sequence.push("usage".into());
                    last_usage_only = true;
                } else {
                    obs.event_sequence.push("empty".into());
                    last_usage_only = false;
                }
            }
            Some(arr) => {
                last_usage_only = false;
                let choice = &arr[0];
                let delta = choice.get("delta");
                if let Some(fr) = str_field(choice, "finish_reason") {
                    obs.stop_reason = Some(fr.clone());
                    obs.event_sequence.push(format!("finish:{fr}"));
                } else if let Some(d) = delta {
                    if let Some(tcs) = d.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let idx = tc.get("index").and_then(|i| i.as_i64()).unwrap_or(-1);
                            obs.event_sequence.push(format!("tool_call_delta[{idx}]"));
                        }
                        let maxidx = tcs
                            .iter()
                            .filter_map(|t| t.get("index").and_then(|i| i.as_i64()))
                            .max()
                            .unwrap_or(-1);
                        obs.tool_call_count = obs.tool_call_count.max((maxidx + 1) as usize);
                    } else if d.get("refusal").and_then(|r| r.as_str()).is_some() {
                        obs.has_refusal = true;
                        obs.event_sequence.push("refusal_delta".into());
                    } else if d.get("content").and_then(|c| c.as_str()).is_some() {
                        obs.event_sequence.push("text_delta".into());
                    } else if d.get("role").is_some() {
                        obs.event_sequence.push("role".into());
                    } else {
                        obs.event_sequence.push("delta".into());
                    }
                }
            }
            None => {
                if has_usage {
                    flatten_usage(data.get("usage").unwrap(), "", &mut obs.usage);
                    obs.event_sequence.push("usage".into());
                    last_usage_only = true;
                }
            }
        }
    }
    obs.usage_only_final_chunk = last_usage_only;
}

fn stream_responses(obs: &mut Observation, events: &[EventRecord]) {
    let mut seqs: Vec<i64> = Vec::new();
    let mut last_usage = false;
    for e in events {
        let name = e.event.clone().unwrap_or_default();
        let data = event_json(e);
        if let Some(sn) = data
            .as_ref()
            .and_then(|d| d.get("sequence_number"))
            .and_then(|s| s.as_i64())
        {
            seqs.push(sn);
        }
        match name.as_str() {
            "response.output_item.added" => {
                let it = data
                    .as_ref()
                    .and_then(|d| d.get("item"))
                    .and_then(|i| str_field(i, "type"))
                    .unwrap_or_default();
                obs.event_sequence.push(format!("output_item.added:{it}"));
                match it.as_str() {
                    "function_call" | "custom_tool_call" | "web_search_call" | "file_search_call" => {
                        obs.tool_call_count += 1
                    }
                    _ => {}
                }
                last_usage = false;
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                if let Some(u) = data
                    .as_ref()
                    .and_then(|d| d.get("response"))
                    .and_then(|r| r.get("usage"))
                {
                    flatten_usage(u, "", &mut obs.usage);
                    last_usage = true;
                }
                let st = data
                    .as_ref()
                    .and_then(|d| d.get("response"))
                    .and_then(|r| str_field(r, "status"));
                obs.stop_reason = st.or_else(|| Some(name.trim_start_matches("response.").to_string()));
                obs.event_sequence.push(name.clone());
            }
            "response.refusal.delta" | "response.refusal.done" => {
                obs.has_refusal = true;
                obs.event_sequence.push(name.clone());
                last_usage = false;
            }
            "error" => {
                obs.error = data.as_ref().and_then(extract_error).or(Some(ErrorObs {
                    r#type: Some("stream_error".into()),
                    ..Default::default()
                }));
                obs.event_sequence.push("error".into());
                last_usage = false;
            }
            other => {
                obs.event_sequence.push(other.to_string());
                last_usage = false;
            }
        }
    }
    // Sequence-number gaps.
    for w in seqs.windows(2) {
        if w[1] != w[0] + 1 {
            obs.sequence_number_gaps.push([w[0], w[1]]);
        }
    }
    obs.usage_only_final_chunk = last_usage;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::parse_sse;

    fn hdrs(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn anthropic_text_nonstream() {
        let body = serde_json::json!({
            "id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",
            "content":[{"type":"text","text":"Hello!"}],
            "stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}
        });
        let obs = observe_nonstream(Protocol::Anthropic, &serde_json::json!({}), 200, &BTreeMap::new(), &body, &Expect::Status(200));
        assert_eq!(obs.verdict, Verdict::Pass);
        assert_eq!(obs.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(obs.type_sequence, vec!["text"]);
        assert_eq!(obs.usage.get("input_tokens"), Some(&serde_json::json!(5)));
        assert_eq!(obs.tool_call_count, 0);
    }

    #[test]
    fn anthropic_thinking_tools_nonstream() {
        let body = serde_json::json!({
            "id":"m","type":"message","role":"assistant","model":"claude-opus-5",
            "content":[
                {"type":"thinking","thinking":"reason","signature":"SIG=="},
                {"type":"text","text":"calling"},
                {"type":"tool_use","id":"toolu_1","name":"get","input":{"q":"x"}}
            ],
            "stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":10}
        });
        let obs = observe_nonstream(Protocol::Anthropic, &serde_json::json!({}), 200, &BTreeMap::new(), &body, &Expect::Any);
        assert_eq!(obs.type_sequence, vec!["thinking", "text", "tool_use"]);
        assert_eq!(obs.tool_call_count, 1);
        assert!(obs.has_signature);
        assert_eq!(obs.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(obs.verdict, Verdict::Na);
    }

    #[test]
    fn anthropic_error_and_verdict() {
        let body = serde_json::json!({"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: must be greater than 0"}});
        let obs = observe_nonstream(
            Protocol::Anthropic,
            &serde_json::json!({}),
            400,
            &hdrs(&[("request-id", "req_1")]),
            &body,
            &Expect::StatusAndType { status: 400, error_type: "invalid_request_error".into() },
        );
        assert_eq!(obs.verdict, Verdict::Pass);
        assert_eq!(obs.error.as_ref().unwrap().r#type.as_deref(), Some("invalid_request_error"));
        assert_eq!(obs.request_id.as_deref(), Some("req_1"));
    }

    #[test]
    fn chat_toolcalls_nonstream() {
        let body = serde_json::json!({
            "choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
                {"id":"call_a","type":"function","function":{"name":"get","arguments":"{}"}},
                {"id":"call_b","type":"function","function":{"name":"get","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":40,"completion_tokens":30,"total_tokens":70}
        });
        let obs = observe_nonstream(Protocol::Chat, &serde_json::json!({}), 200, &BTreeMap::new(), &body, &Expect::Status(200));
        assert_eq!(obs.stop_reason.as_deref(), Some("tool_calls"));
        assert_eq!(obs.tool_call_count, 2);
        assert_eq!(obs.type_sequence, vec!["tool_call", "tool_call"]);
        assert_eq!(obs.usage.get("total_tokens"), Some(&serde_json::json!(70)));
    }

    #[test]
    fn chat_error_headers() {
        let body = serde_json::json!({"error":{"message":"Incorrect API key provided.","type":"invalid_request_error","code":"invalid_api_key"}});
        let obs = observe_nonstream(Protocol::Chat, &serde_json::json!({}), 401, &hdrs(&[("x-request-id","req_auth_1")]), &body, &Expect::Status(401));
        assert_eq!(obs.error.as_ref().unwrap().code.as_deref(), Some("invalid_api_key"));
        assert_eq!(obs.request_id.as_deref(), Some("req_auth_1"));
        assert_eq!(obs.verdict, Verdict::Pass);
    }

    #[test]
    fn responses_completed_nonstream() {
        let body = serde_json::json!({
            "id":"resp_abc123","object":"response","status":"completed","model":"gpt-5.4",
            "output":[
                {"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"x"}],"encrypted_content":"ENCBLOB","status":"completed"},
                {"id":"fc_1","type":"function_call","call_id":"call_abc","name":"get","arguments":"{}","status":"completed"}
            ],
            "usage":{"input_tokens":50,"output_tokens":20,"total_tokens":70}
        });
        let req = serde_json::json!({"model":"gpt-5.4","status":"completed"});
        let obs = observe_nonstream(Protocol::Responses, &req, 200, &BTreeMap::new(), &body, &Expect::Status(200));
        assert_eq!(obs.type_sequence, vec!["reasoning", "function_call"]);
        assert_eq!(obs.tool_call_count, 1);
        assert!(obs.has_encrypted_content);
        assert_eq!(obs.stop_reason.as_deref(), Some("completed"));
        // model + status echoed.
        assert!(obs.echoed_request_fields.contains(&"model".to_string()));
        assert!(obs.echoed_request_fields.contains(&"status".to_string()));
    }

    #[test]
    fn responses_incomplete_reason() {
        let body = serde_json::json!({"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]});
        let obs = observe_nonstream(Protocol::Responses, &serde_json::json!({}), 200, &BTreeMap::new(), &body, &Expect::Any);
        assert_eq!(obs.stop_reason.as_deref(), Some("incomplete:max_output_tokens"));
    }

    #[test]
    fn anthropic_stream_sequence() {
        let raw = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get\",\"input\":{}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":15}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let events = parse_sse(raw);
        let obs = observe_stream(Protocol::Anthropic, &serde_json::json!({}), 200, &BTreeMap::new(), raw, &events, &Expect::Status(200));
        assert_eq!(obs.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(obs.tool_call_count, 1);
        assert!(obs.event_sequence.contains(&"content_block_start[0]:text".to_string()));
        assert!(obs.event_sequence.contains(&"content_block_start[1]:tool_use".to_string()));
        assert_eq!(obs.usage.get("output_tokens"), Some(&serde_json::json!(15)));
    }

    #[test]
    fn chat_stream_done_and_usage() {
        let raw = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"total_tokens\":13}}\n\ndata: [DONE]\n\n";
        let events = parse_sse(raw);
        let obs = observe_stream(Protocol::Chat, &serde_json::json!({}), 200, &BTreeMap::new(), raw, &events, &Expect::Status(200));
        assert!(obs.done_present);
        assert_eq!(obs.stop_reason.as_deref(), Some("stop"));
        assert_eq!(obs.usage.get("total_tokens"), Some(&serde_json::json!(13)));
        // The usage chunk precedes [DONE], but [DONE] carries no usage, so the last usage-bearing
        // chunk was usage-only.
        assert!(obs.usage_only_final_chunk);
    }

    #[test]
    fn responses_stream_sequence_and_gaps() {
        let raw = b"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"r\"}}\n\nevent: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\"}}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"delta\":\"Hi\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"r\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n";
        let events = parse_sse(raw);
        let obs = observe_stream(Protocol::Responses, &serde_json::json!({}), 200, &BTreeMap::new(), raw, &events, &Expect::Status(200));
        assert_eq!(obs.stop_reason.as_deref(), Some("completed"));
        assert_eq!(obs.sequence_number_gaps, vec![[1, 3]]);
        assert_eq!(obs.usage.get("total_tokens"), Some(&serde_json::json!(3)));
        assert!(obs.usage_only_final_chunk);
        assert!(obs.event_sequence.contains(&"output_item.added:message".to_string()));
    }

    #[test]
    fn stream_error_body_when_no_events() {
        let raw = br#"{"error":{"type":"invalid_request_error","message":"bad","code":"model_not_found"}}"#;
        let events = parse_sse(raw); // no SSE frames -> empty
        let obs = observe_stream(
            Protocol::Chat,
            &serde_json::json!({}),
            404,
            &BTreeMap::new(),
            raw,
            &events,
            &Expect::StatusAndType { status: 404, error_type: "invalid_request_error".into() },
        );
        assert_eq!(obs.verdict, Verdict::Pass);
        assert_eq!(obs.error.as_ref().unwrap().code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn verdict_fail_on_wrong_status() {
        let body = serde_json::json!({"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]});
        let obs = observe_nonstream(Protocol::Chat, &serde_json::json!({}), 200, &BTreeMap::new(), &body, &Expect::Status(400));
        assert_eq!(obs.verdict, Verdict::Fail);
        assert!(obs.verdict_detail.as_ref().unwrap().contains("expected status 400"));
    }

    #[test]
    fn rate_limit_headers_collected() {
        let body = serde_json::json!({"choices":[]});
        let h = hdrs(&[
            ("anthropic-ratelimit-requests-remaining", "10"),
            ("x-ratelimit-limit-tokens", "1000"),
            ("retry-after", "30"),
            ("content-type", "application/json"),
        ]);
        let obs = observe_nonstream(Protocol::Chat, &serde_json::json!({}), 200, &h, &body, &Expect::Any);
        assert_eq!(obs.retry_after.as_deref(), Some("30"));
        assert_eq!(obs.rate_limit.len(), 2);
        assert!(obs.rate_limit.contains_key("x-ratelimit-limit-tokens"));
    }
}
