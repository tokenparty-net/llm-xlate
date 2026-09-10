//! Pair-matrix golden runner (plan §12).
//!
//! For every request fixture under `tests/golden/{chat,responses,anthropic}/*.req.json` the
//! runner decodes it with its own protocol and, for each *other* protocol as the target and
//! each representative preset whose transport allows that target, runs the full
//! `decode → lower → encode` path and snapshots the encoded body, the sorted headers, and the
//! `x-router-degraded` value — or, when a step rejects the request, the typed [`XlateError`].
//!
//! For every response fixture (`*.resp.json`, non-streaming) and stream fixture
//! (`*.stream.sse`) the runner decodes it with its own protocol codec and, for each client
//! protocol, snapshots both the client stream-encoded frames and the aggregated
//! `encode_response` body — the two must be mutually consistent (plan §11.8), and side-by-side
//! snapshots make any divergence visible.
//!
//! Snapshot names are deterministic (`<fixture>__<client>_to_<target>__<preset>` for requests,
//! `<fixture>__<provider>_to_<client>__{stream,response}` for responses), so the committed
//! `.snap` files under `tests/snapshots/` are the spec's reviewed evidence.

use llm_xlate::caps::{preset, Capabilities};
use llm_xlate::codec::{EncodeCtx, EncodedRequest, TranslatorConfig};
use llm_xlate::degrade::Degradations;
use llm_xlate::envelope::Sealer;
use llm_xlate::error::XlateError;
use llm_xlate::ir::{IrEvent, Protocol, ReasoningExposure, ResponseId};
use llm_xlate::requirements::Resolutions;
use llm_xlate::Translator;
use llm_xlate_core::HeaderMap;

const CREATED_AT: u64 = 1_700_000_000;

fn dir() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden")
}

fn read(sub: &str, file: &str) -> Vec<u8> {
    let path = format!("{}/{sub}/{file}", dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// List the fixture base names under `tests/golden/<sub>/` whose file name ends in `suffix`,
/// sorted for determinism. The matrix globs its directories (plan §9) so a promoted capture enters
/// the matrix with no code edit — dropping `<name>.req.json` / `.resp.json` / `.stream.sse` /
/// `.err.json` into the right dir is all it takes.
fn fixtures(sub: &str, suffix: &str) -> Vec<String> {
    let d = format!("{}/{sub}", dir());
    let mut out: Vec<String> = std::fs::read_dir(&d)
        .unwrap_or_else(|e| panic!("read_dir {d}: {e}"))
        .flatten()
        .filter_map(|e| {
            e.file_name()
                .to_string_lossy()
                .strip_suffix(suffix)
                .map(|s| s.to_string())
        })
        .collect();
    out.sort();
    out
}

fn pname(p: Protocol) -> &'static str {
    match p {
        Protocol::OaiChat => "chat",
        Protocol::OaiResponses => "responses",
        Protocol::Anthropic => "anthropic",
    }
}

const PROTS: [Protocol; 3] = [Protocol::OaiChat, Protocol::OaiResponses, Protocol::Anthropic];

/// The eight representative presets (plan §12), each paired with a stable name and the
/// upstream model the router would resolve the client's model name to.
fn presets() -> Vec<(&'static str, &'static str, Capabilities)> {
    vec![
        ("claude_old", "claude-3-5-sonnet-latest", preset::claude_old()),
        ("claude_46", "claude-opus-4-6", preset::claude_46()),
        ("claude_5", "claude-opus-5", preset::claude_5()),
        ("gpt4o", "gpt-4o", preset::gpt4o()),
        ("gpt5_chat", "gpt-5.4", preset::gpt5_chat()),
        ("gpt5_responses", "gpt-5.4", preset::gpt5_responses()),
        ("gpt6", "gpt-6-astra", preset::gpt6()),
        ("openai_compatible", "llama-x", preset::openai_compatible()),
    ]
}

/// The preset used to *decode* a provider response of the given protocol.
fn provider_caps(p: Protocol) -> Capabilities {
    match p {
        Protocol::OaiChat => preset::gpt5_chat(),
        Protocol::OaiResponses => preset::gpt5_responses(),
        Protocol::Anthropic => preset::claude_5(),
    }
}

fn render_ok(enc: &EncodedRequest, lower_degr: &Degradations) -> String {
    let mut out = String::new();
    out.push_str("--- body ---\n");
    match serde_json::from_slice::<serde_json::Value>(&enc.body) {
        Ok(v) => out.push_str(&serde_json::to_string_pretty(&v).unwrap()),
        Err(_) => out.push_str(&String::from_utf8_lossy(&enc.body)),
    }
    out.push_str("\n--- headers ---\n");
    let mut headers: Vec<(String, String)> = enc
        .headers
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned()))
        .collect();
    headers.sort();
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\n"));
    }
    out.push_str(&format!("upstream_streams: {}\n", enc.upstream_streams));
    // Combined lowering + wiring degradations, in order.
    let mut degr = lower_degr.clone();
    degr.extend(clone_degr(&enc.degradations));
    out.push_str("--- degraded ---\n");
    out.push_str(&degr.render_header_value());
    out
}

fn clone_degr(d: &Degradations) -> Degradations {
    let mut out = Degradations::new();
    for x in d.iter() {
        out.push(x.clone());
    }
    out
}

fn render_err(e: &XlateError) -> String {
    format!(
        "ERROR kind={} status={} param={}\nmessage: {}",
        e.kind.slug(),
        e.status,
        e.param.as_deref().unwrap_or("-"),
        e.message
    )
}

fn run_request(client: Protocol, fixture: &str) {
    let file = format!("{fixture}.req.json");
    let body = read(pname(client), &file);
    let xl = Translator::new(TranslatorConfig::default());
    let ir = xl
        .decode_request(client, &body, &HeaderMap::new())
        .unwrap_or_else(|e| panic!("decode {} {fixture}: {e}", pname(client)));
    let ctx = xl.encode_ctx(client, &ir, ResponseId::new("resp_golden"), CREATED_AT);
    let res = Resolutions::new();
    for target in PROTS {
        if target == client {
            continue;
        }
        for (preset_name, upstream_model, caps) in presets() {
            if !caps.protocol_allowed(target) {
                continue;
            }
            // The router resolves the client's model name to the backend model before lowering.
            let mut ir = ir.clone();
            ir.model.resolved = Some(upstream_model.to_string());
            let rendered = match xl.lower(ir, &caps, target, &res) {
                Ok(low) => match xl.encode_request(target, &low.req, &caps, &ctx) {
                    Ok(enc) => render_ok(&enc, &low.degradations),
                    Err(e) => render_err(&e),
                },
                Err(e) => render_err(&e),
            };
            let name = format!(
                "{fixture}__{}_to_{}__{preset_name}",
                pname(client),
                pname(target)
            );
            insta::assert_snapshot!(name, rendered);
        }
    }
}

fn decode_events(xl: &Translator, provider: Protocol, file: &str, is_stream: bool) -> Vec<IrEvent> {
    let caps = provider_caps(provider);
    if is_stream {
        let body = read(pname(provider), file);
        let mut dec = xl.stream_decoder(provider, &caps);
        let mut events = dec.push(&body);
        events.extend(dec.finish());
        events
    } else {
        let body = read(pname(provider), file);
        xl.decode_response(provider, &body, &caps)
            .unwrap_or_else(|e| panic!("decode_response {} {file}: {e}", pname(provider)))
    }
}

fn client_ctx(client: Protocol) -> EncodeCtx {
    let sealer = Sealer::new(&TranslatorConfig::default().envelope_key);
    let mut ctx = EncodeCtx::new(client, "golden-model", ResponseId::new("resp_golden"), sealer);
    ctx.created_at = CREATED_AT;
    ctx.expose = ReasoningExposure::Full;
    ctx.include_usage = true;
    ctx.stream = true;
    ctx
}

fn render_frames(frames: &[bytes::Bytes]) -> String {
    let mut s = String::new();
    for f in frames {
        s.push_str(&String::from_utf8_lossy(f));
    }
    s
}

fn run_response(provider: Protocol, fixture: &str, is_stream: bool) {
    let ext = if is_stream { "stream.sse" } else { "resp.json" };
    let file = format!("{fixture}.{ext}");
    let xl = Translator::new(TranslatorConfig::default());
    let events = decode_events(&xl, provider, &file, is_stream);
    let kind = if is_stream { "stream" } else { "resp" };
    for client in PROTS {
        // Client stream-encode (concatenated frames).
        let ctx = client_ctx(client);
        let mut enc = xl.stream_encoder(client, ctx);
        let mut frames = Vec::new();
        for ev in events.iter().cloned() {
            frames.extend(enc.push(ev));
        }
        frames.extend(enc.finish());
        let stream_name = format!(
            "{fixture}_{kind}__{}_to_{}__stream",
            pname(provider),
            pname(client)
        );
        insta::assert_snapshot!(stream_name, render_frames(&frames));

        // Aggregate → encode_response.
        let resp_rendered = match xl.aggregate_stream(events.iter().cloned()) {
            Ok(resp) => {
                let ctx = client_ctx(client);
                let body = xl.encode_response(client, &resp, &ctx);
                match serde_json::from_slice::<serde_json::Value>(&body) {
                    Ok(v) => serde_json::to_string_pretty(&v).unwrap(),
                    Err(_) => String::from_utf8_lossy(&body).into_owned(),
                }
            }
            Err(e) => render_err(&e),
        };
        let resp_name = format!(
            "{fixture}_{kind}__{}_to_{}__response",
            pname(provider),
            pname(client)
        );
        insta::assert_snapshot!(resp_name, resp_rendered);

        // Plan §11.8: the client's *streaming* and *non-streaming* renderings of the same
        // response must aggregate to the same items and stop. The snapshots above show the two
        // side by side; this assertion makes the equality load-bearing (so a regression in the
        // rich tool / reasoning / refusal paths fails a test, not just an eyeballed snapshot).
        if let Ok(resp) = xl.aggregate_stream(events.iter().cloned()) {
            let cc = provider_caps(client);
            // Streaming rendering → re-decode → aggregate.
            let joined: Vec<u8> = frames.iter().flat_map(|b| b.iter().copied()).collect();
            let mut sdec = xl.stream_decoder(client, &cc);
            let mut sev = sdec.push(&joined);
            sev.extend(sdec.finish());
            let via_stream = xl.aggregate_stream(sev).expect("aggregate client stream");
            // Non-streaming rendering → re-decode → aggregate.
            let mut nctx = client_ctx(client);
            nctx.stream = false;
            let body = xl.encode_response(client, &resp, &nctx);
            let nev = xl.decode_response(client, &body, &cc).expect("decode client response");
            let via_resp = xl.aggregate_stream(nev).expect("aggregate client response");
            assert_eq!(
                project(&via_stream.items),
                project(&via_resp.items),
                "{fixture} {kind} {}_to_{}: streamed vs non-streamed client items diverge",
                pname(provider),
                pname(client)
            );
            assert_eq!(
                via_stream.stop,
                via_resp.stop,
                "{fixture} {kind} {}_to_{}: streamed vs non-streamed stop diverges",
                pname(provider),
                pname(client)
            );
        }
    }
}

/// Project items onto the load-bearing, dialect-stable surface for the §11.8 stream/non-stream
/// consistency check: message role + text + refusal, function tool-call id/name/args, and tool
/// results — the parts the router must never silently reorder or drop, and crucially where
/// parallel / interleaved tool-call block ordering lives. Reasoning and provider-hosted tool
/// items are excluded: their text/summary and hosted-block rendering legitimately differ
/// between a codec's streaming and non-streaming encoders (documented per-dialect
/// normalizations), and those divergences are captured by the side-by-side snapshots, not this
/// equality law.
fn project(items: &[llm_xlate::ir::Item]) -> Vec<String> {
    use llm_xlate::ir::{Item, Part};
    items
        .iter()
        .filter_map(|it| match it {
            Item::Message { role, content, .. } => {
                let text: String = content.iter().filter_map(|p| p.as_text()).collect();
                let refusal: String = content
                    .iter()
                    .filter_map(|p| match p {
                        Part::Refusal { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                // A genuinely EMPTY assistant message (no text, no refusal) is a degenerate
                // response — e.g. a reasoning model that spends its whole budget on hidden
                // reasoning and hits `length` with no visible output (real captures do this). The
                // streaming encoder emits an empty message block while the non-streaming encoder
                // emits none; that presence/absence is a documented per-dialect normalization
                // captured by the side-by-side snapshots, not this "modulo text" equality law, so
                // project it to nothing on both sides.
                if text.is_empty() && refusal.is_empty() {
                    return None;
                }
                Some(format!("msg:{role:?}:text={text}:refusal={refusal}"))
            }
            Item::ToolCall { call_id, name, arguments, .. } => {
                // Compare arguments modulo JSON formatting: the streaming encoder replays the
                // model's original argument bytes (which may carry insignificant whitespace) while
                // the non-streaming encoder canonicalizes them. That whitespace is a documented
                // per-dialect normalization (snapshots capture it), not a divergence this law
                // should fail on — so canonicalize to compact JSON for the comparison key.
                let args = serde_json::from_str::<serde_json::Value>(arguments.as_str())
                    .map(|v| v.to_string())
                    .unwrap_or_else(|_| arguments.as_str().to_string());
                Some(format!("call:{}:{name}:{}", call_id.as_str(), args))
            }
            Item::ToolResult { call_id, is_error, .. } => {
                Some(format!("result:{}:err={is_error}", call_id.as_str()))
            }
            _ => None,
        })
        .collect()
}

// -------------------------------------------------------------------------------------------
// Error-body fixtures (plan §10): decode a provider error body → render it in every client
// dialect. Each `*.err.json` carries `{status, headers, body}`; the runner decodes it with the
// provider codec and snapshots the typed [`XlateError`] plus, for every client protocol, the
// non-streaming and streaming `encode_error` output (status + headers + body/frames).
// -------------------------------------------------------------------------------------------

fn hmap_from(v: &serde_json::Value) -> HeaderMap {
    let mut h = HeaderMap::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if let Some(s) = val.as_str() {
                if let (Ok(name), Ok(value)) = (
                    llm_xlate_core::HeaderName::from_bytes(k.as_bytes()),
                    llm_xlate_core::HeaderValue::from_str(s),
                ) {
                    h.insert(name, value);
                }
            }
        }
    }
    h
}

fn render_error_headers(h: &HeaderMap) -> String {
    let mut headers: Vec<(String, String)> = h
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned()))
        .collect();
    headers.sort();
    let mut out = String::new();
    for (k, v) in headers {
        out.push_str(&format!("  {k}: {v}\n"));
    }
    out
}

fn run_error(provider: Protocol, fixture: &str) {
    let file = format!("{fixture}.err.json");
    let raw = read(pname(provider), &file);
    let env: serde_json::Value = serde_json::from_slice(&raw)
        .unwrap_or_else(|e| panic!("parse {} {file}: {e}", pname(provider)));
    let status = env.get("status").and_then(serde_json::Value::as_u64).unwrap_or(500) as u16;
    let hdrs = hmap_from(env.get("headers").unwrap_or(&serde_json::Value::Null));
    let body = serde_json::to_vec(env.get("body").unwrap_or(&serde_json::Value::Null)).unwrap();

    let xl = Translator::new(TranslatorConfig::default());
    let caps = provider_caps(provider);
    let e = xl.decode_error(provider, status, &body, &hdrs, &caps);

    let mut out = String::new();
    out.push_str("--- decoded XlateError ---\n");
    out.push_str(&format!(
        "kind={} status={} retryable={} param={} provider_type={} retry_after={} upstream_request_id={}\nmessage: {}\n",
        e.kind.slug(),
        e.status,
        e.retryable,
        e.param.as_deref().unwrap_or("-"),
        e.provider_type.as_deref().unwrap_or("-"),
        e.retry_after.map(|d| format!("{}ms", d.as_millis())).unwrap_or_else(|| "-".into()),
        e.upstream_request_id.as_deref().unwrap_or("-"),
        e.message,
    ));

    for client in PROTS {
        out.push_str(&format!("\n--- {} (non-streaming) ---\n", pname(client)));
        let enc = xl.encode_error(client, &e, false, false);
        out.push_str(&format!("status: {}\n", enc.status));
        out.push_str("headers:\n");
        out.push_str(&render_error_headers(&enc.headers));
        out.push_str("body:\n");
        match serde_json::from_slice::<serde_json::Value>(&enc.body) {
            Ok(v) => out.push_str(&serde_json::to_string_pretty(&v).unwrap()),
            Err(_) => out.push_str(&String::from_utf8_lossy(&enc.body)),
        }
        out.push('\n');

        out.push_str(&format!("--- {} (streaming, pre-start) ---\n", pname(client)));
        let senc = xl.encode_error(client, &e, true, false);
        out.push_str(&format!("status: {}\n", senc.status));
        out.push_str("frames:\n");
        out.push_str(&render_frames(&senc.frames));
    }

    let name = format!("{fixture}__{}", pname(provider));
    insta::assert_snapshot!(name, out);
}

// The matrices below glob their golden directories (plan §9) so a promoted capture enters the
// matrix with no code edit. `run_error`/`run_request`/`run_response` still key off the fixture
// base name, so snapshot names are unchanged for the existing fixtures.

#[test]
fn chat_error_matrix() {
    for f in fixtures("chat", ".err.json") {
        run_error(Protocol::OaiChat, &f);
    }
}

#[test]
fn responses_error_matrix() {
    for f in fixtures("responses", ".err.json") {
        run_error(Protocol::OaiResponses, &f);
    }
}

#[test]
fn anthropic_error_matrix() {
    for f in fixtures("anthropic", ".err.json") {
        run_error(Protocol::Anthropic, &f);
    }
}

// -------------------------------------------------------------------------------------------
// Request fixtures
// -------------------------------------------------------------------------------------------

#[test]
fn chat_request_matrix() {
    for f in fixtures("chat", ".req.json") {
        run_request(Protocol::OaiChat, &f);
    }
}

#[test]
fn responses_request_matrix() {
    for f in fixtures("responses", ".req.json") {
        run_request(Protocol::OaiResponses, &f);
    }
}

#[test]
fn anthropic_request_matrix() {
    for f in fixtures("anthropic", ".req.json") {
        run_request(Protocol::Anthropic, &f);
    }
}

// -------------------------------------------------------------------------------------------
// Response / stream fixtures
// -------------------------------------------------------------------------------------------

#[test]
fn chat_response_matrix() {
    for f in fixtures("chat", ".resp.json") {
        run_response(Protocol::OaiChat, &f, false);
    }
    for f in fixtures("chat", ".stream.sse") {
        run_response(Protocol::OaiChat, &f, true);
    }
}

#[test]
fn responses_response_matrix() {
    for f in fixtures("responses", ".resp.json") {
        run_response(Protocol::OaiResponses, &f, false);
    }
    for f in fixtures("responses", ".stream.sse") {
        run_response(Protocol::OaiResponses, &f, true);
    }
}

#[test]
fn anthropic_response_matrix() {
    for f in fixtures("anthropic", ".resp.json") {
        run_response(Protocol::Anthropic, &f, false);
    }
    for f in fixtures("anthropic", ".stream.sse") {
        run_response(Protocol::Anthropic, &f, true);
    }
}
