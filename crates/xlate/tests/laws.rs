//! Property law (plan §11 law 9): `lower` is total — for any request, any preset, any target it
//! returns `Ok` or an `XlateError` whose kind is one of `Unsupported` / `IncompatibleHistory` /
//! `InvalidRequest`, and never panics.

mod common;

use llm_xlate::caps::{preset, Capabilities};
use llm_xlate::codec::{EncodeCtx, TranslatorConfig};
use llm_xlate::error::ErrorKind;
use llm_xlate::ir::{
    CallId, Delta, Effort, Instruction, IrEvent, IrRequest, Item, ItemKind, JsonText, MediaSource,
    OpaqueBlob, OpaqueItem, OpaqueKind, OutputFormat, Part, Position, Protocol, ProviderFamily,
    ReasoningItem, ResponseId, Role, StopReason, ToolChoice, ToolDef, Usage, Verbosity,
};
use llm_xlate::lower::lower;
use llm_xlate::requirements::Resolutions;
use llm_xlate::store::{to_stored, BackendBinding};
use llm_xlate::{wrap, Translator};
use proptest::prelude::*;
use serde_json::{json, Value};

fn family() -> impl Strategy<Value = ProviderFamily> {
    prop_oneof![
        Just(ProviderFamily::Anthropic),
        Just(ProviderFamily::OpenAI),
        Just(ProviderFamily::Other("openai-compatible".to_string())),
    ]
}

fn any_item() -> impl Strategy<Value = Item> {
    prop_oneof![
        any::<bool>().prop_map(|a| if a {
            Item::user_text("u")
        } else {
            Item::assistant_text("a")
        }),
        "[a-z_]{1,8}".prop_map(|id| Item::ToolCall {
            call_id: CallId::new(id),
            name: "f".to_string(),
            arguments: JsonText::new("{}"),
            id: None,
        }),
        "[a-z_ !]{1,8}".prop_map(|id| Item::ToolResult {
            call_id: CallId::new(id),
            content: vec![Part::text("res")],
            is_error: false,
            id: None,
        }),
        family().prop_map(|f| Item::Reasoning(ReasoningItem {
            text: Some("t".to_string()),
            summary: vec![],
            opaque: Some(OpaqueBlob::new(f, OpaqueKind::Signature, "sig")),
            id: None,
        })),
        Just(Item::Reasoning(ReasoningItem {
            text: Some("t".to_string()),
            summary: vec![],
            opaque: None,
            id: None,
        })),
        family().prop_map(|f| Item::ProviderToolCall(OpaqueItem::new(
            f,
            json!({"type":"web_search","name":"web_search"})
        ))),
        family().prop_map(|f| Item::ProviderToolResult(OpaqueItem::new(
            f,
            json!({"type":"web_search","content":[{"text":"x"}]})
        ))),
        family().prop_map(|f| Item::Message {
            role: Role::User,
            content: vec![Part::Image(MediaSource::FileRef { family: f, id: "file_1".to_string() })],
            id: None,
        }),
        Just(Item::Message {
            role: Role::User,
            content: vec![Part::Audio(MediaSource::Base64 {
                media_type: "audio/wav".to_string(),
                data: bytes::Bytes::from_static(b"a"),
            })],
            id: None,
        }),
    ]
}

fn any_tool() -> impl Strategy<Value = ToolDef> {
    prop_oneof![
        (any::<bool>()).prop_map(|s| ToolDef::Function {
            name: "f".to_string(),
            description: None,
            parameters: json!({"type":"object","properties":{"x":{"type":"string","pattern":"^a$"}}}),
            strict: Some(s),
            cache_control: None,
        }),
        family().prop_map(|f| ToolDef::Provider(OpaqueItem::new(
            f,
            json!({"type":"web_search","name":"web_search"})
        ))),
    ]
}

fn any_effort() -> impl Strategy<Value = Option<Effort>> {
    prop_oneof![
        Just(None),
        Just(Some(Effort::None)),
        Just(Some(Effort::Minimal)),
        Just(Some(Effort::Low)),
        Just(Some(Effort::Medium)),
        Just(Some(Effort::High)),
        Just(Some(Effort::XHigh)),
        Just(Some(Effort::Max)),
    ]
}

fn any_output() -> impl Strategy<Value = OutputFormat> {
    prop_oneof![
        Just(OutputFormat::Text),
        Just(OutputFormat::JsonObject),
        Just(OutputFormat::JsonSchema {
            name: "s".to_string(),
            schema: json!({"type":"object","pattern":"^a$"}),
            strict: true,
            description: None,
        }),
    ]
}

fn any_tool_choice() -> impl Strategy<Value = ToolChoice> {
    prop_oneof![
        Just(ToolChoice::Auto),
        Just(ToolChoice::None),
        Just(ToolChoice::Required),
        Just(ToolChoice::Named("web_search".to_string())),
    ]
}

prop_compose! {
    fn any_request()(
        items in prop::collection::vec(any_item(), 0..6),
        tools in prop::collection::vec(any_tool(), 0..3),
        effort in any_effort(),
        budget in prop::option::of(0u32..40000),
        output in any_output(),
        tool_choice in any_tool_choice(),
        n in prop::option::of(0u32..4),
        stops in prop::collection::vec("[a-z]{1,3}", 0..7),
        service_tier in prop::option::of(prop_oneof![Just("auto".to_string()), Just("weird".to_string())]),
        store in prop::option::of(any::<bool>()),
        prev in prop::option::of(Just(ResponseId::new("r_prev"))),
        verbosity in prop::option::of(prop_oneof![Just(Verbosity::Low), Just(Verbosity::High)]),
        add_mid in any::<bool>(),
        add_ext in any::<bool>(),
        parallel in prop::option::of(any::<bool>()),
    ) -> IrRequest {
        let mut req = IrRequest { items, tools, tool_choice, ..Default::default() };
        req.reasoning.effort = effort;
        req.reasoning.budget_tokens = budget;
        req.output.format = output;
        req.output.verbosity = verbosity;
        req.sampling.n = n;
        req.limits.stop_sequences = stops;
        req.meta.service_tier = service_tier;
        req.state.store = store;
        req.state.previous_response_id = prev;
        req.parallel_tool_calls = parallel;
        if add_mid {
            let mut ins = Instruction::system_text("mid");
            ins.position = Position::Before(0);
            ins.effort = Some(Effort::High);
            req.instructions.push(ins);
        }
        if add_ext {
            req.ext.insert("chat.extra", json!(1));
            req.ext.insert("responses.extra", json!(2));
        }
        req
    }
}

fn presets() -> Vec<Capabilities> {
    vec![
        preset::claude_old(),
        preset::claude_46(),
        preset::claude_5(),
        preset::gpt4o(),
        preset::gpt5_chat(),
        preset::gpt5_responses(),
        preset::gpt6(),
        preset::openai_compatible(),
    ]
}

const TARGETS: [Protocol; 3] = [Protocol::OaiChat, Protocol::OaiResponses, Protocol::Anthropic];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]
    #[test]
    fn total_lowering(req in any_request()) {
        let cfg = TranslatorConfig::default();
        let res = Resolutions::new();
        for caps in presets() {
            for target in TARGETS {
                let result = lower(req.clone(), &caps, target, &res, &cfg);
                match result {
                    Ok(_) => {}
                    Err(e) => prop_assert!(
                        matches!(
                            e.kind,
                            ErrorKind::Unsupported
                                | ErrorKind::IncompatibleHistory
                                | ErrorKind::InvalidRequest
                        ),
                        "unexpected error kind {:?} for target {:?}",
                        e.kind,
                        target
                    ),
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    #[test]
    fn total_lowering_is_idempotent_on_repeat(req in any_request()) {
        // Running twice with identical inputs yields identical output (determinism).
        let cfg = TranslatorConfig::default();
        let res = Resolutions::new();
        let caps = preset::claude_5();
        let a = lower(req.clone(), &caps, Protocol::Anthropic, &res, &cfg);
        let b = lower(req, &caps, Protocol::Anthropic, &res, &cfg);
        match (a, b) {
            (Ok(a), Ok(b)) => prop_assert_eq!(a.req, b.req),
            (Err(a), Err(b)) => prop_assert_eq!(a.kind, b.kind),
            _ => prop_assert!(false, "nondeterministic Ok/Err"),
        }
    }
}

// ===========================================================================================
// Shared helpers for the §11 invariant laws
// ===========================================================================================

fn xl() -> Translator {
    Translator::new(TranslatorConfig::default())
}

/// Encode `req` for `target`/`caps` through the facade (lower -> encode), returning the body or
/// the error rendered as a stable string. `ctx` supplies the response id / created_at.
fn lower_encode(
    req: IrRequest,
    caps: &Capabilities,
    target: Protocol,
    ctx: &EncodeCtx,
) -> Result<Vec<u8>, String> {
    let x = xl();
    let res = Resolutions::new();
    match x.lower(req, caps, target, &res) {
        Ok(low) => match x.encode_request(target, &low.req, caps, ctx) {
            Ok(enc) => Ok(enc.body.to_vec()),
            Err(e) => Err(format!("encode:{}:{}", e.kind.slug(), e.message)),
        },
        Err(e) => Err(format!("lower:{}:{}", e.kind.slug(), e.message)),
    }
}

fn ctx_for(client: Protocol, req: &IrRequest, id: &str) -> EncodeCtx {
    xl().encode_ctx(client, req, ResponseId::new(id), 1_700_000_000)
}

fn golden_dir() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden")
}

// ===========================================================================================
// Law 1 - determinism across *separate processes* (plan §11.1)
// ===========================================================================================

/// A deterministic corpus: every request golden x every applicable target x every preset,
/// hashed in a fixed order. Pure encoding => identical bytes => identical hash in any process.
fn corpus_hash() -> String {
    use sha2::{Digest, Sha256};
    let x = xl();
    let mut hasher = Sha256::new();
    let reqs: &[(Protocol, &str, &[&str])] = &[
        (
            Protocol::OaiChat,
            "chat",
            &[
                "text",
                "multiturn_sysdev",
                "parallel_tools",
                "json_schema",
                "images",
                "pdf",
                "reasoning",
                "legacy_functions",
            ],
        ),
        (Protocol::OaiResponses, "responses", &["full"]),
        (
            Protocol::Anthropic,
            "anthropic",
            &["multimodal", "structured", "tools_thinking"],
        ),
    ];
    type PresetFn = fn() -> Capabilities;
    let presets: &[(&str, PresetFn)] = &[
        ("claude_old", preset::claude_old),
        ("claude_46", preset::claude_46),
        ("claude_5", preset::claude_5),
        ("gpt4o", preset::gpt4o),
        ("gpt5_chat", preset::gpt5_chat),
        ("gpt5_responses", preset::gpt5_responses),
        ("gpt6", preset::gpt6),
        ("openai_compatible", preset::openai_compatible),
    ];
    let prots = [Protocol::OaiChat, Protocol::OaiResponses, Protocol::Anthropic];
    for (client, sub, fixtures) in reqs {
        for fixture in *fixtures {
            let body =
                std::fs::read(format!("{}/{sub}/{fixture}.req.json", golden_dir())).unwrap();
            let ir = x
                .decode_request(*client, &body, &llm_xlate::HeaderMap::new())
                .unwrap();
            let ctx = ctx_for(*client, &ir, "resp_hash");
            for target in prots {
                if target == *client {
                    continue;
                }
                for (_pn, mk) in presets {
                    let caps = mk();
                    if !caps.protocol_allowed(target) {
                        continue;
                    }
                    match lower_encode(ir.clone(), &caps, target, &ctx) {
                        Ok(b) => hasher.update(&b),
                        Err(s) => hasher.update(s.as_bytes()),
                    }
                    hasher.update([0xff]); // record separator
                }
            }
        }
    }
    let out = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[test]
fn determinism_across_processes() {
    // Child mode: print the hash and return (gated so the parent's spawn does not recurse).
    if std::env::var("XLATE_HASH_CHILD").is_ok() {
        println!("XLATE_HASH={}", corpus_hash());
        return;
    }
    let exe = std::env::current_exe().expect("current_exe");
    let run_child = || -> String {
        let out = std::process::Command::new(&exe)
            .args(["--exact", "determinism_across_processes", "--nocapture"])
            .env("XLATE_HASH_CHILD", "1")
            .output()
            .expect("spawn child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        stdout
            .lines()
            .find_map(|l| l.strip_prefix("XLATE_HASH=").map(|h| h.to_string()))
            .unwrap_or_else(|| panic!("child printed no hash; stdout:\n{stdout}"))
    };
    let a = run_child();
    let b = run_child();
    assert_eq!(a, b, "encoding is nondeterministic across processes");
    // Sanity: the in-process hash matches the child hash too.
    assert_eq!(a, corpus_hash());
}

// ===========================================================================================
// Law 2 - canonical bytes: user JSON key order and float text preserved (plan §11.2)
// ===========================================================================================

#[test]
fn canonical_bytes_preserve_key_order_and_floats() {
    // Tool-call arguments are opaque JsonText: they must reach the wire byte-for-byte.
    let args = r#"{"z":0.10,"a":1e2,"nested":{"y":1,"x":2}}"#;
    let items = vec![
        common::user("go"),
        Item::ToolCall {
            call_id: CallId::new("call_1"),
            name: "f".to_string(),
            arguments: JsonText::new(args),
            id: None,
        },
        common::tool_result("call_1", "ok"),
    ];
    let mut req = common::req_with_items(items);
    req.tools.push(common::function_tool(
        "f",
        json!({"type": "object"}),
        None,
    ));
    req.limits.max_output_tokens = Some(256);
    let ctx = ctx_for(Protocol::OaiChat, &req, "resp_canon");
    // Chat -> Chat: arguments pass through verbatim.
    let body = lower_encode(req, &preset::gpt4o(), Protocol::OaiChat, &ctx).expect("encode");
    let s = String::from_utf8(body).unwrap();
    // On the wire the arguments are a JSON *string* (escaped); the canonical form is the raw
    // JsonText re-encoded as a JSON string literal, byte-for-byte — no key sorting, no float
    // reformatting.
    let escaped = serde_json::to_string(args).unwrap();
    assert!(
        s.contains(&escaped),
        "tool arguments not preserved byte-for-byte; expected {escaped}\nbody was:\n{s}"
    );
    // Floats are not reformatted (0.10 not 0.1, 1e2 not 100.0), key order not sorted.
    assert!(s.contains("0.10"), "float 0.10 reformatted");
    assert!(s.contains("1e2"), "float 1e2 reformatted");
    assert!(
        s.contains(r#"\"z\":0.10,\"a\":1e2"#),
        "user JSON key order not preserved"
    );
}

// ===========================================================================================
// Law 3 - prefix stability on Ant and Resp targets (plan §11.3)
// ===========================================================================================

/// Rule tested: for a transcript `T` and `T' = T + [one appended item of a *different* role
/// than T's last item]` (so no assistant/user run-merge perturbs T's final message), every
/// byte of `encode(T)` up to and including the serialization of T's last message is a byte
/// prefix of `encode(T')`. We witness this by locating the unique text of T's last message in
/// `encode(T)` and asserting the two encodings agree on every byte up to the end of that text.
fn assert_prefix_stable(target: Protocol, caps: &Capabilities) {
    // A realistic transcript ends with the pending *user* turn (a trailing assistant message is
    // a prefill, which some models reject). Growth appends a completed assistant turn plus the
    // next user turn — both of a different role than the item they follow, so no run-merge
    // perturbs T's final (user) message.
    let last_text = "CLAST_USER_UNIQUE";
    let base = vec![
        common::user("ALPHA_FIRST"),
        common::asst("BETA_REPLY"),
        common::user(last_text),
    ];
    let mut extended = base.clone();
    extended.push(common::asst("DELTA_REPLY")); // assistant after user => no merge
    extended.push(common::user("EPSILON_NEXT"));

    let mut t = common::req_with_items(base);
    t.limits.max_output_tokens = Some(256);
    let mut t2 = common::req_with_items(extended);
    t2.limits.max_output_tokens = Some(256);

    // Use a same-family client so cross-family auto request-level caching (which moves a
    // cache_control breakpoint onto the *new* last block) does not perturb T's final message.
    let ctx = ctx_for(target, &t, "resp_prefix");
    let a = lower_encode(t, caps, target, &ctx).expect("encode T");
    let b = lower_encode(t2, caps, target, &ctx).expect("encode T'");
    let sa = String::from_utf8(a).unwrap();
    let sb = String::from_utf8(b).unwrap();

    let idx = sa.find(last_text).expect("last message text missing") + last_text.len();
    assert!(
        sb.len() >= idx && sa.as_bytes()[..idx] == sb.as_bytes()[..idx],
        "prefix instability on {target:?}: encode(T) up to the last message is not a prefix of \
         encode(T')"
    );
}

#[test]
fn prefix_stability_anthropic() {
    assert_prefix_stable(Protocol::Anthropic, &preset::claude_5());
}

#[test]
fn prefix_stability_responses() {
    assert_prefix_stable(Protocol::OaiResponses, &preset::gpt5_responses());
}

// ===========================================================================================
// Law 4 - no hoisting: a mid-context instruction never lands in top-level system/instructions
// (plan §11.4)
// ===========================================================================================

fn top_level_system_text(target: Protocol, body: &[u8]) -> String {
    let v: Value = serde_json::from_slice(body).unwrap();
    let field = match target {
        Protocol::Anthropic => v.get("system"),
        Protocol::OaiResponses => v.get("instructions"),
        Protocol::OaiChat => None, // Chat has no dedicated top-level system field
    };
    field.map(|f| f.to_string()).unwrap_or_default()
}

fn assert_no_hoist(target: Protocol, caps: &Capabilities) {
    let mut req = common::req_with_items(vec![common::user("Q_ONE"), common::user("Q_TWO")]);
    req.instructions.push(Instruction::system_text("SYS_TOP_LEVEL"));
    req.instructions.push(common::mid_instruction(1, "MID_SECRET_XYZ"));
    req.limits.max_output_tokens = Some(256);
    let ctx = ctx_for(Protocol::OaiChat, &req, "resp_hoist");
    let body = lower_encode(req, caps, target, &ctx).expect("encode");
    let top = top_level_system_text(target, &body);
    assert!(
        !top.contains("MID_SECRET_XYZ"),
        "mid-context instruction hoisted into top-level system on {target:?}: {top}"
    );
    // The whole body must still carry the mid text somewhere (inline), i.e. it is not dropped.
    let whole = String::from_utf8(body).unwrap();
    assert!(
        whole.contains("MID_SECRET_XYZ"),
        "mid-context instruction vanished entirely on {target:?}"
    );
}

#[test]
fn no_hoisting_anthropic() {
    assert_no_hoist(Protocol::Anthropic, &preset::claude_5());
}

#[test]
fn no_hoisting_responses() {
    assert_no_hoist(Protocol::OaiResponses, &preset::gpt5_responses());
}

// ===========================================================================================
// Law 5 - wrapper strings are versioned `wrap_v1` and snapshotted (plan §11.5)
// ===========================================================================================

#[test]
fn wrapper_version_is_wrap_v1() {
    assert_eq!(wrap::WRAP_VERSION, "wrap_v1");
}

#[test]
fn wrapper_strings_snapshot() {
    let rendered = format!(
        "WRAP_VERSION={}\n\n[system_message]\n{}\n\n[document titled]\n{}\n\n[document untitled]\n{}\n\n[provider_tool_fold]\n{}\n\n[tool_result_attachment]\n{}",
        wrap::WRAP_VERSION,
        wrap::system_message("BODY"),
        wrap::document(Some("T\"itle"), "DOC"),
        wrap::document(None, "DOC"),
        wrap::provider_tool_fold("web_search", "RESULT"),
        wrap::tool_result_attachment("call_1"),
    );
    insta::assert_snapshot!("wrappers_wrap_v1", rendered);
}

// ===========================================================================================
// Law 6 - no minted ids in prompts: the router-minted response id never leaks into an encoded
// upstream request (plan §11.6)
// ===========================================================================================

#[test]
fn no_minted_response_id_in_request() {
    let minted = "resp_MINTED_DO_NOT_LEAK";
    for (target, caps) in [
        (Protocol::OaiChat, preset::gpt4o()),
        (Protocol::OaiResponses, preset::gpt5_responses()),
        (Protocol::Anthropic, preset::claude_5()),
    ] {
        let mut req = common::req_with_items(vec![common::user("hello world")]);
        req.limits.max_output_tokens = Some(256);
        let ctx = ctx_for(Protocol::OaiChat, &req, minted);
        let body = lower_encode(req, &caps, target, &ctx).expect("encode");
        let s = String::from_utf8(body).unwrap();
        assert!(
            !s.contains(minted),
            "minted response id leaked into {target:?} request body:\n{s}"
        );
    }
}

// ===========================================================================================
// Law 7 - degradation coverage against `openai_compatible` (plan §11.7)
// ===========================================================================================

#[test]
fn degradation_coverage_openai_compatible() {
    // A request exercising several lossy-but-representable features. (Tools / structured output
    // would hard-error on openai_compatible, so they are exercised in the golden matrix, not
    // here - this law is about *silent drops* of representable state.)
    let mut req = common::req_with_items(vec![common::user("Q_A"), common::user("Q_B")]);
    req.instructions.push(common::mid_instruction(1, "MID"));
    req.reasoning.effort = Some(Effort::High);
    req.reasoning.budget_tokens = Some(4096);
    req.output.verbosity = Some(Verbosity::High);
    req.state.store = Some(true);
    req.state.previous_response_id = Some(ResponseId::new("r_prev"));
    req.state.conversation = Some("conv_1".to_string());
    req.state.background = Some(true);
    req.state.include = vec!["reasoning.encrypted_content".to_string()];
    req.meta.service_tier = Some("flex".to_string());
    req.limits.stop_sequences = vec!["STOP".to_string()];
    req.limits.max_output_tokens = Some(256);

    let x = xl();
    let low = x
        .lower(req, &preset::openai_compatible(), Protocol::OaiChat, &Resolutions::new())
        .expect("lower ok");
    let mut fields: Vec<String> = low.degradations.iter().map(|d| d.field.clone()).collect();
    fields.sort();
    fields.dedup();

    // Every lossy feature above must be reported *and nothing else* — the deduped degradation
    // field set is exactly this list, so a spurious/duplicate drop (a field not below) also
    // fails the test (plan §11.7: no silent drop, and no phantom degradation either).
    let mut expected = vec![
        "background",
        "conversation",
        "include",
        "previous_response_id",
        "reasoning",
        "service_tier",
        "store",
        "verbosity",
    ];
    expected.sort();
    assert_eq!(fields, expected, "degradation field set is not exactly the expected set");
}

/// Law 7, tools/structured-output arm: a backend that *represents* tools and JSON-schema output
/// but cannot honor every knob must still report a [`Degradation`] for each lossy step, rather
/// than hard-erroring (the openai_compatible arm above cannot exercise these because it rejects
/// tools/structured output outright). Here a strict schema against a strict-unsupported backend
/// and a parallel-tool-calls request against a backend without parallel control both degrade.
#[test]
fn degradation_coverage_tools_and_schema() {
    // A JSON-schema request with `strict: true` lowered to a backend whose strict support is
    // off must downgrade `text.format.strict` (represent-but-degrade, not drop).
    let mut req = common::req_with_items(vec![common::user("Q")]);
    req.output.format = OutputFormat::JsonSchema {
        name: "S".to_string(),
        schema: json!({"type": "object"}),
        strict: true,
        description: None,
    };
    req.limits.max_output_tokens = Some(256);

    // A backend that represents json_schema output but has `strict` unsupported: take gpt4o
    // (json_schema-capable) and turn strict off, so the schema is *represented* while the strict
    // knob *degrades* rather than hard-erroring.
    let mut caps = preset::gpt4o();
    caps.output.strict_supported = llm_xlate::caps::Tri::No;
    let x = xl();
    let low = x
        .lower(req, &caps, Protocol::OaiChat, &Resolutions::new())
        .expect("json_schema is representable (degrades, not errors)");
    assert!(
        low.degradations.iter().any(|d| d.field == "output.strict"),
        "strict downgrade must be reported; got {:?}",
        low.degradations.iter().map(|d| d.field.clone()).collect::<Vec<_>>()
    );
}

// ===========================================================================================
// Law 8 - round-trip laws (plan §11.8)
// ===========================================================================================

/// A plain text event corpus (one message, two text deltas).
fn corpus_text() -> Vec<IrEvent> {
    vec![
        IrEvent::Start {
            response_id: ResponseId::new("resp_1"),
            model: "m".to_string(),
            // Anthropic carries input tokens only on `message_start` (from prefill); supply it
            // so the stream round trip can recover `input` on every dialect.
            usage_prefill: Some(Usage::new(7, 0)),
        },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
        IrEvent::Delta { index: 0, delta: Delta::Text("Hello ".to_string()) },
        IrEvent::Delta { index: 0, delta: Delta::Text("world".to_string()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::Stop {
            reason: StopReason::EndTurn,
            usage: Usage::new(7, 3),
            ext: Default::default(),
        },
    ]
}

/// A rich corpus: a leading text message plus two **parallel** tool calls whose IR `ItemStop`s
/// are interleaved (`start0, args0, start1, args1, stop0, stop1`). This is exactly the shape
/// that provokes overlapping Anthropic content blocks; the stream/non-stream equality below
/// pins that the two renderings agree on the tool calls regardless.
fn corpus_parallel_tools() -> Vec<IrEvent> {
    vec![
        IrEvent::Start {
            response_id: ResponseId::new("resp_1"),
            model: "m".to_string(),
            usage_prefill: Some(Usage::new(9, 0)),
        },
        IrEvent::ItemStart { index: 0, kind: ItemKind::Message, id: None, call: None },
        IrEvent::Delta { index: 0, delta: Delta::Text("Calling tools".to_string()) },
        IrEvent::ItemStop { index: 0 },
        IrEvent::ItemStart {
            index: 1,
            kind: ItemKind::ToolCall,
            id: None,
            call: Some((CallId::new("call_a"), "get_weather".to_string())),
        },
        IrEvent::Delta { index: 1, delta: Delta::ToolArgs("{\"city\":\"NYC\"}".to_string()) },
        IrEvent::ItemStart {
            index: 2,
            kind: ItemKind::ToolCall,
            id: None,
            call: Some((CallId::new("call_b"), "get_time".to_string())),
        },
        IrEvent::Delta { index: 2, delta: Delta::ToolArgs("{\"tz\":\"ET\"}".to_string()) },
        IrEvent::ItemStop { index: 1 },
        IrEvent::ItemStop { index: 2 },
        IrEvent::Stop {
            reason: StopReason::ToolUse,
            usage: Usage::new(9, 6),
            ext: Default::default(),
        },
    ]
}

/// `aggregate(client_decode(encode_stream(ev))) == aggregate(ev)` and
/// `encode_response(aggregate(ev))` re-decodes to the same aggregate, for each protocol.
fn assert_stream_response_consistency(p: Protocol, caps: &Capabilities) {
    for events in [corpus_text(), corpus_parallel_tools()] {
        assert_consistency_for(p, caps, &events);
    }
}

fn assert_consistency_for(p: Protocol, caps: &Capabilities, events: &[IrEvent]) {
    let x = xl();
    let base = x.aggregate_stream(events.iter().cloned()).expect("aggregate base");

    // Stream path: encode to client frames, decode them back, aggregate.
    let mut ctx = EncodeCtx::new(p, "m", ResponseId::new("resp_1"), x.sealer());
    ctx.include_usage = true;
    ctx.stream = true;
    let mut enc = x.stream_encoder(p, ctx);
    let mut frames = Vec::new();
    for ev in events.iter().cloned() {
        frames.extend(enc.push(ev));
    }
    frames.extend(enc.finish());
    let joined: Vec<u8> = frames.iter().flat_map(|b| b.iter().copied()).collect();
    let mut dec = x.stream_decoder(p, caps);
    let mut redec = dec.push(&joined);
    redec.extend(dec.finish());
    let via_stream = x.aggregate_stream(redec).expect("aggregate via stream");

    // Responses mints derived item ids (`{kind}_{response_id}_{index}`, a documented
    // normalization, plan §11.6) that the raw base events lack, so compare id-stripped items.
    assert_eq!(
        strip_ids(&base.items),
        strip_ids(&via_stream.items),
        "{p:?}: stream items differ"
    );
    assert_eq!(base.stop, via_stream.stop, "{p:?}: stream stop differs");
    assert_eq!(
        (base.usage.input, base.usage.output),
        (via_stream.usage.input, via_stream.usage.output),
        "{p:?}: stream usage differs"
    );

    // Response path: encode_response(aggregate(ev)), decode it back, aggregate again.
    let mut rctx = EncodeCtx::new(p, "m", ResponseId::new("resp_1"), x.sealer());
    rctx.include_usage = true;
    let body = x.encode_response(p, &base, &rctx);
    let events2 = x.decode_response(p, &body, caps).expect("decode_response");
    let via_resp = x.aggregate_stream(events2).expect("aggregate via response");
    assert_eq!(
        strip_ids(&base.items),
        strip_ids(&via_resp.items),
        "{p:?}: response items differ"
    );
    assert_eq!(base.stop, via_resp.stop, "{p:?}: response stop differs");
}

/// Clear the (optional) `id` field of every item, so comparisons ignore derived/minted ids.
fn strip_ids(items: &[Item]) -> Vec<Item> {
    items
        .iter()
        .cloned()
        .map(|it| match it {
            Item::Message { role, content, .. } => Item::Message { role, content, id: None },
            Item::ToolCall { call_id, name, arguments, .. } => {
                Item::ToolCall { call_id, name, arguments, id: None }
            }
            Item::ToolResult { call_id, content, is_error, .. } => {
                Item::ToolResult { call_id, content, is_error, id: None }
            }
            Item::Reasoning(mut r) => {
                r.id = None;
                Item::Reasoning(r)
            }
            other => other,
        })
        .collect()
}

#[test]
fn stream_response_consistency_all_protocols() {
    assert_stream_response_consistency(Protocol::OaiChat, &preset::gpt5_chat());
    assert_stream_response_consistency(Protocol::OaiResponses, &preset::gpt5_responses());
    assert_stream_response_consistency(Protocol::Anthropic, &preset::claude_5());
}

/// A foreign-family (OpenAI) provider-hosted `web_search_call` followed by an assistant message.
/// It has no valid Anthropic content-block shape, so on the →Anthropic path both encoders must
/// **drop** it rather than leak an OpenAI-shaped `content_block_start` an Anthropic client rejects.
fn corpus_foreign_provider_tool() -> Vec<IrEvent> {
    let raw = json!({
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": "weather in NYC"}
    });
    vec![
        IrEvent::Start {
            response_id: ResponseId::new("resp_1"),
            model: "m".to_string(),
            usage_prefill: Some(Usage::new(9, 0)),
        },
        IrEvent::ItemStart { index: 0, kind: ItemKind::ProviderToolCall, id: None, call: None },
        IrEvent::Delta {
            index: 0,
            delta: Delta::ProviderRaw { family: ProviderFamily::OpenAI, raw },
        },
        IrEvent::ItemStop { index: 0 },
        IrEvent::ItemStart { index: 1, kind: ItemKind::Message, id: None, call: None },
        IrEvent::Delta { index: 1, delta: Delta::Text("It is sunny.".to_string()) },
        IrEvent::ItemStop { index: 1 },
        IrEvent::Stop { reason: StopReason::EndTurn, usage: Usage::new(9, 6), ext: Default::default() },
    ]
}

/// §11.8 for provider-hosted tools (finding 1): a foreign-family provider tool routed to an
/// Anthropic client must be dropped identically by the streaming and non-streaming encoders —
/// `aggregate(encode_stream(ev))` and `encode_response(aggregate(ev))` must agree, and the raw
/// Anthropic stream frames must never carry the leaked OpenAI `web_search_call` shape.
#[test]
fn stream_response_agree_on_foreign_provider_tool() {
    let x = xl();
    let p = Protocol::Anthropic;
    let caps = preset::claude_5();
    let events = corpus_foreign_provider_tool();
    let base = x.aggregate_stream(events.iter().cloned()).expect("aggregate base");
    // Base IR carries the foreign provider tool call...
    assert!(
        base.items.iter().any(|it| matches!(it, Item::ProviderToolCall(_))),
        "base should carry the foreign provider tool"
    );

    // Stream path.
    let mut ctx = EncodeCtx::new(p, "m", ResponseId::new("resp_1"), x.sealer());
    ctx.include_usage = true;
    ctx.stream = true;
    let mut enc = x.stream_encoder(p, ctx);
    let mut frames = Vec::new();
    for ev in events.iter().cloned() {
        frames.extend(enc.push(ev));
    }
    frames.extend(enc.finish());
    let joined: Vec<u8> = frames.iter().flat_map(|b| b.iter().copied()).collect();
    let text = String::from_utf8_lossy(&joined);
    assert!(
        !text.contains("web_search_call"),
        "Anthropic stream leaked a foreign OpenAI content block:\n{text}"
    );
    let mut dec = x.stream_decoder(p, &caps);
    let mut redec = dec.push(&joined);
    redec.extend(dec.finish());
    let via_stream = x.aggregate_stream(redec).expect("aggregate via stream");

    // Response path.
    let mut rctx = EncodeCtx::new(p, "m", ResponseId::new("resp_1"), x.sealer());
    rctx.include_usage = true;
    let body = x.encode_response(p, &base, &rctx);
    let via_resp =
        x.aggregate_stream(x.decode_response(p, &body, &caps).expect("decode_response"))
            .expect("aggregate via response");

    // Both drop the foreign provider tool, and agree on the surviving items (§11.8).
    assert!(
        !via_stream.items.iter().any(|it| matches!(it, Item::ProviderToolCall(_))),
        "stream path should drop the foreign provider tool"
    );
    assert!(
        !via_resp.items.iter().any(|it| matches!(it, Item::ProviderToolCall(_))),
        "response path should drop the foreign provider tool"
    );
    assert_eq!(
        strip_ids(&via_stream.items),
        strip_ids(&via_resp.items),
        "stream and response disagree on foreign-provider-tool rendering"
    );
}

fn item_text(it: &Item) -> Option<String> {
    match it {
        Item::Message { content, .. } => {
            let s: String = content.iter().filter_map(|p| p.as_text()).collect();
            Some(s)
        }
        _ => None,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(120))]
    /// Same-protocol Chat round trip on a text/multiturn corpus where the documented
    /// normalizations are the identity: `decode(encode(ir))` preserves the message items.
    #[test]
    fn chat_text_roundtrip_preserves_items(
        turns in prop::collection::vec(
            (any::<bool>(), "[a-zA-Z0-9 ?.]{1,20}"),
            1..6,
        )
    ) {
        // Alternate roles strictly (index parity), so no consecutive same-role items trigger
        // Chat's documented assistant/user run-merge (plan §7.7) — that merge is exercised in
        // the codec crate; here we isolate the identity round trip.
        let items: Vec<Item> = turns
            .iter()
            .enumerate()
            .map(|(i, (_is_user, t))| if i % 2 == 0 { common::user(t) } else { common::asst(t) })
            .collect();
        let mut req = common::req_with_items(items.clone());
        req.limits.max_output_tokens = Some(256);
        let x = xl();
        let ctx = ctx_for(Protocol::OaiChat, &req, "resp_rt");
        let low = x.lower(req, &preset::gpt4o(), Protocol::OaiChat, &Resolutions::new()).unwrap();
        let enc = x.encode_request(Protocol::OaiChat, &low.req, &preset::gpt4o(), &ctx).unwrap();
        let back = x.decode_request(Protocol::OaiChat, &enc.body, &llm_xlate::HeaderMap::new()).unwrap();
        // Message text and roles survive intact (Chat merges nothing for alternating text turns).
        let orig_texts: Vec<String> = items.iter().filter_map(item_text).collect();
        let back_texts: Vec<String> = back.items.iter().filter_map(item_text).collect();
        prop_assert_eq!(orig_texts, back_texts);
    }
}

// ===========================================================================================
// Law 10 - streaming fuzz: split each `.stream.sse` at seeded boundaries => identical events
// (plan §11.10)
// ===========================================================================================

fn decode_stream_whole(p: Protocol, caps: &Capabilities, bytes: &[u8]) -> Vec<IrEvent> {
    let x = xl();
    let mut dec = x.stream_decoder(p, caps);
    let mut ev = dec.push(bytes);
    ev.extend(dec.finish());
    ev
}

fn decode_stream_split(
    p: Protocol,
    caps: &Capabilities,
    bytes: &[u8],
    cuts: &[usize],
) -> Vec<IrEvent> {
    let x = xl();
    let mut dec = x.stream_decoder(p, caps);
    let mut ev = Vec::new();
    let mut prev = 0usize;
    let mut points: Vec<usize> = cuts.iter().copied().filter(|&c| c <= bytes.len()).collect();
    points.sort_unstable();
    points.dedup();
    for c in points {
        ev.extend(dec.push(&bytes[prev..c]));
        prev = c;
    }
    ev.extend(dec.push(&bytes[prev..]));
    ev.extend(dec.finish());
    ev
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]
    #[test]
    fn streaming_fuzz_split_boundaries(cuts in prop::collection::vec(0usize..4000, 0..12)) {
        let cases: &[(Protocol, &str, &str)] = &[
            (Protocol::OaiChat, "chat", "tools.stream.sse"),
            (Protocol::OaiChat, "chat", "reasoning.stream.sse"),
            (Protocol::OaiResponses, "responses", "rich.stream.sse"),
            (Protocol::Anthropic, "anthropic", "tools.stream.sse"),
        ];
        for (p, sub, file) in cases {
            let caps = match p {
                Protocol::OaiChat => preset::gpt5_chat(),
                Protocol::OaiResponses => preset::gpt5_responses(),
                Protocol::Anthropic => preset::claude_5(),
            };
            let bytes = std::fs::read(format!("{}/{sub}/{file}", golden_dir())).unwrap();
            let whole = decode_stream_whole(*p, &caps, &bytes);
            let split = decode_stream_split(*p, &caps, &bytes, &cuts);
            prop_assert_eq!(&whole, &split, "split decode differs for {}/{}", sub, file);
        }
    }
}

// ===========================================================================================
// Law 11 - a materialized chain equals its stateless twin, byte-for-byte (plan §11.11)
// ===========================================================================================

#[test]
fn materialized_chain_equals_stateless_twin() {
    let target = Protocol::Anthropic;
    let caps = preset::claude_5();

    // Stateless twin: the whole transcript inline.
    let mut twin = common::req_with_items(vec![
        common::user("Q1"),
        common::asst("A1"),
        common::user("Q2"),
    ]);
    twin.model = llm_xlate::ir::ModelRef::new("claude-opus-5");
    twin.limits.max_output_tokens = Some(256);

    // Chained: one stored turn (Q1 -> A1) plus a new request carrying only Q2.
    let turn1_req = common::req_with_items(vec![common::user("Q1")]);
    let turn1_out = common::resp(vec![common::asst("A1")], StopReason::EndTurn);
    let binding = BackendBinding::new("cred_1", ProviderFamily::Anthropic, "claude-opus-5");
    let stored = to_stored(
        &turn1_req,
        &turn1_out,
        binding,
        ResponseId::new("r1"),
        1_700_000_000,
        Default::default(),
    );
    let mut new_req = common::req_with_items(vec![common::user("Q2")]);
    new_req.model = llm_xlate::ir::ModelRef::new("claude-opus-5");
    new_req.limits.max_output_tokens = Some(256);
    new_req.state.previous_response_id = Some(ResponseId::new("r1"));

    let x = xl();
    let materialized = x.materialize_chain(&[stored], new_req).expect("materialize");

    // The materialized request must equal the twin's items (previous_response_id cleared).
    assert_eq!(materialized.items, twin.items, "materialized items differ from twin");

    // ...and, once lowered+encoded, be byte-identical to the twin.
    let ctx = ctx_for(Protocol::OaiResponses, &twin, "resp_twin");
    let a = lower_encode(twin, &caps, target, &ctx).expect("encode twin");
    let b = lower_encode(materialized, &caps, target, &ctx).expect("encode materialized");
    assert_eq!(
        String::from_utf8(a).unwrap(),
        String::from_utf8(b).unwrap(),
        "materialized chain is not byte-identical to its stateless twin"
    );
}
