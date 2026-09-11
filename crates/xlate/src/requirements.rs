//! The read-only requirements pre-pass (plan §4, §7.2, §7.5, §9).
//!
//! Before a request can be lowered for a target backend the router may need to resolve some
//! external state: file ids minted by a *different* provider family that must be bridged, an
//! opaque reasoning blob missing from the last tool-bearing assistant turn (fetched from the
//! sidecar), and a `previous_response_id` chain to materialize. [`requirements`] inspects a
//! decoded [`IrRequest`] and reports exactly those needs, deterministically and without
//! mutating anything. The router resolves them (producing [`Resolutions`] and, for the chain,
//! calling [`crate::store::materialize_chain`]) and then calls [`crate::lower::lower`].

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

use llm_xlate_core::caps::{Capabilities, ReplayMode};
use llm_xlate_core::ir::{
    CallId, IrRequest, Item, MediaSource, OpaqueBlob, Part, Protocol, ProviderFamily, ResponseId,
};

/// What the router must resolve before [`crate::lower::lower`] can run (plan §4).
///
/// All three lists/fields are produced in deterministic item order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Requirements {
    /// Tool-call ids on the last tool-bearing assistant turn that need an opaque reasoning
    /// blob replayed (the backend requires it, and the transcript does not carry one). The
    /// router looks each up in its sidecar and supplies it via [`Resolutions::reasoning`].
    pub reasoning_for_calls: Vec<CallId>,
    /// File ids referenced in the transcript that were minted by a provider family other than
    /// the target's. The router bridges each to a target-family file id via
    /// [`Resolutions::files`].
    pub foreign_files: Vec<FileRef>,
    /// The `previous_response_id` this request chains from, if any. The router loads the chain
    /// and calls [`crate::store::materialize_chain`] before lowering.
    pub chain: Option<ResponseId>,
}

/// A provider-namespaced file id: the family that minted it plus the id string. Mirrors the
/// [`MediaSource::FileRef`] shape as a standalone, orderable value so it can key the
/// [`Resolutions::files`] bridge map.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileRef {
    /// The family that minted the id.
    pub family: ProviderFamily,
    /// The file id.
    pub id: String,
}

impl FileRef {
    /// Construct a file reference.
    pub fn new(family: ProviderFamily, id: impl Into<String>) -> Self {
        Self { family, id: id.into() }
    }
}

// `ProviderFamily` is deliberately not `Ord` (it is a bare string in most contexts); order
// `FileRef` by the stable family label then id so it can be a deterministic `BTreeMap` key.
impl Ord for FileRef {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.family.label(), &self.id).cmp(&(other.family.label(), &other.id))
    }
}
impl PartialOrd for FileRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The resolutions the router supplies back to [`crate::lower::lower`] (plan §4).
///
/// `BTreeMap`s (not `HashMap`s) so iteration and any derived output are deterministic.
#[derive(Debug, Clone, Default)]
pub struct Resolutions {
    /// Opaque reasoning blobs to inject before their tool calls, keyed by [`CallId`].
    pub reasoning: BTreeMap<CallId, OpaqueBlob>,
    /// Foreign → target-family file id substitutions.
    pub files: BTreeMap<FileRef, FileRef>,
}

impl Resolutions {
    /// An empty resolution set.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Inspect a decoded request and report what the router must resolve for `target` (plan §4).
///
/// Pure and read-only. Determinism: every output list is in transcript (item) order.
pub fn requirements(req: &IrRequest, caps: &Capabilities, target: Protocol) -> Requirements {
    let target_family = target.family();

    // ── chain ────────────────────────────────────────────────────────────────────────────
    let chain = req.state.previous_response_id.clone();

    // ── foreign files ────────────────────────────────────────────────────────────────────
    let mut foreign_files = Vec::new();
    let mut seen = HashSet::new();
    for_each_media_source(req, |ms| {
        if let MediaSource::FileRef { family, id } = ms {
            if *family != target_family {
                let fref = FileRef::new(family.clone(), id.clone());
                if seen.insert(fref.clone()) {
                    foreign_files.push(fref);
                }
            }
        }
    });

    // ── reasoning required on the last tool turn ─────────────────────────────────────────
    let mut reasoning_for_calls = Vec::new();
    if caps.reasoning.required_on_last_tool_turn.is_yes() {
        if let Some(range) = last_assistant_run(&req.items) {
            if !run_has_replayable_reasoning(&req.items[range.clone()], &target_family, caps) {
                for item in &req.items[range] {
                    if let Item::ToolCall { call_id, .. } = item {
                        reasoning_for_calls.push(call_id.clone());
                    }
                }
            }
        }
    }

    Requirements { reasoning_for_calls, foreign_files, chain }
}

/// The index range of the last maximal assistant-side run — the run that precedes the trailing
/// user-side items (plan §7.2). Skips any trailing user-side items (tool results), then walks
/// back over the contiguous assistant-side items. `None` if there is no assistant-side run.
pub(crate) fn last_assistant_run(items: &[Item]) -> Option<std::ops::Range<usize>> {
    let mut end = items.len();
    while end > 0 && items[end - 1].is_user_side() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut start = end;
    while start > 0 && items[start - 1].is_assistant_side() {
        start -= 1;
    }
    Some(start..end)
}

/// Whether a run already carries reasoning the target backend can replay, so no sidecar lookup
/// is needed.
///
/// Two carriers qualify, matching what the request encoders actually put back on the wire:
///
/// * an opaque blob of the target family (verbatim signature / encrypted-item replay), and
/// * plain reasoning text or a summary, when the backend's replay slot **is** a text field
///   (`ReplayMode::TextField`) — a Chat provider that echoes `reasoning_content` back.
///
/// Missing the second carrier is what made a transcript carrying `reasoning_content` look like
/// it had no reasoning at all, so lowering rejected it as `incompatible_history`.
pub(crate) fn run_has_replayable_reasoning(
    run: &[Item],
    target_family: &ProviderFamily,
    caps: &Capabilities,
) -> bool {
    let text_replay = caps.reasoning.replay == Some(ReplayMode::TextField);
    run.iter().any(|it| match it {
        Item::Reasoning(r) => {
            r.opaque.as_ref().is_some_and(|b| &b.family == target_family)
                || (text_replay && (r.text.is_some() || !r.summary.is_empty()))
        }
        _ => false,
    })
}

/// Visit every [`MediaSource`] in a request's instructions and items, in order.
pub(crate) fn for_each_media_source(req: &IrRequest, mut f: impl FnMut(&MediaSource)) {
    fn visit_parts(parts: &[Part], f: &mut impl FnMut(&MediaSource)) {
        for p in parts {
            match p {
                Part::Image(ms) | Part::Audio(ms) => f(ms),
                Part::Document { source, .. } => f(source),
                _ => {}
            }
        }
    }
    for ins in &req.instructions {
        visit_parts(&ins.content, &mut f);
    }
    for item in &req.items {
        match item {
            Item::Message { content, .. } | Item::ToolResult { content, .. } => {
                visit_parts(content, &mut f)
            }
            _ => {}
        }
    }
}
