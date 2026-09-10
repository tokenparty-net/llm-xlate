//! Pass 2: reasoning / effort lowering (plan §7.2).
//!
//! Handles, in order: replay vs drop of existing reasoning items; resolution of reasoning that
//! the backend *requires* on the last tool-bearing turn; the Responses reasoning-pairing rule;
//! effort snapping to the model's supported levels; clearing reasoning when the backend has no
//! reasoning mode; and clearing a rejected/ignored explicit budget.

use llm_xlate_core::caps::{Capabilities, BudgetStatus, ReasoningMode, ReplayMode};
use llm_xlate_core::codec::{TranslatorConfig, UnresolvedReasoning};
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{Effort, Item, Protocol, ReasoningExposure, ReasoningItem, Role};

use crate::requirements::{last_assistant_run, run_has_native_reasoning};

pub(crate) fn run(
    req: &mut llm_xlate_core::ir::IrRequest,
    caps: &Capabilities,
    target: Protocol,
    res: &crate::requirements::Resolutions,
    cfg: &TranslatorConfig,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    let target_family = target.family();
    let text_replay = caps.reasoning.replay == Some(ReplayMode::TextField);

    // (a) Replay policy for every existing Reasoning item (§7.2 lowering).
    req.items.retain(|item| {
        let Item::Reasoning(r) = item else { return true };
        match &r.opaque {
            // Opaque of the target family → keep for native replay.
            Some(blob) if blob.family == target_family => true,
            // Opaque of a foreign family → never replay foreign reasoning as text; drop it.
            Some(_) => {
                degr.dropped("reasoning", "foreign-family reasoning blob dropped (not replayable)");
                false
            }
            // No opaque carrier → keep only where a text replay slot exists (Chat providers).
            None => {
                if text_replay {
                    true
                } else {
                    degr.dropped(
                        "reasoning",
                        "reasoning without an opaque carrier dropped (no text replay slot)",
                    );
                    false
                }
            }
        }
    });

    // (b) Reasoning required on the last tool-bearing turn (§7.2).
    //
    // Anthropic only requires a thinking block to be replayed on a tool_use turn that *had*
    // extended thinking; a tool call produced with thinking disabled (or adaptively skipped) is
    // accepted with no thinking block. So enforce this only when (1) reasoning is actually
    // enabled on the request, and (2) the last assistant run is the *active* continuation —
    // every item after it is a tool_result (not a later user turn, which would make it a
    // historical, already-answered turn Anthropic strips thinking from). Without these gates the
    // check rejects a plain agentic loop (`[user, assistant(tool_call), tool_result]`) that
    // Anthropic accepts.
    let reasoning_enabled = req.reasoning.effort.is_some_and(|e| e != Effort::None)
        || req.reasoning.enabled == Some(true)
        || req.reasoning.budget_tokens.is_some();
    if caps.reasoning.required_on_last_tool_turn.is_yes() && reasoning_enabled {
        if let Some(range) = last_assistant_run(&req.items) {
            let is_active_continuation =
                req.items[range.end..].iter().all(|it| matches!(it, Item::ToolResult { .. }));
            let run = &req.items[range.clone()];
            if is_active_continuation && !run_has_native_reasoning(run, &target_family) {
                let calls: Vec<_> = run
                    .iter()
                    .filter_map(|it| match it {
                        Item::ToolCall { call_id, .. } => Some(call_id.clone()),
                        _ => None,
                    })
                    .collect();
                if !calls.is_empty() {
                    let all_resolved = calls.iter().all(|c| res.reasoning.contains_key(c));
                    if all_resolved {
                        insert_resolved_reasoning(req, range, res);
                    } else if cfg.unresolved_reasoning == UnresolvedReasoning::Fail {
                        return Err(XlateError::incompatible_history(
                            "the backend requires reasoning on the last tool turn but none is \
                             available to replay",
                        ));
                    } else {
                        degr.dropped(
                            "reasoning.required",
                            "unresolved required reasoning stripped; tool calls left in place",
                        );
                    }
                }
            }
        }
    }

    // (c) Responses pairing rule: every reasoning item must be immediately followed by a
    // ToolCall or an assistant Message; otherwise drop it (§7.2).
    if target == Protocol::OaiResponses {
        let n = req.items.len();
        let keep: Vec<bool> = (0..n)
            .map(|i| {
                if matches!(req.items[i], Item::Reasoning(_)) {
                    matches!(
                        req.items.get(i + 1),
                        Some(Item::ToolCall { .. })
                            | Some(Item::Message { role: Role::Assistant, .. })
                    )
                } else {
                    true
                }
            })
            .collect();
        let mut idx = 0;
        req.items.retain(|_| {
            let k = keep[idx];
            if !k {
                degr.dropped(
                    "reasoning",
                    "Responses reasoning item is not followed by its paired call/message; dropped",
                );
            }
            idx += 1;
            k
        });
    }

    // (d) Effort snapping to supported levels (§7.2).
    if let (Some(e), Some(levels)) = (req.reasoning.effort, &caps.reasoning.effort_levels) {
        if let Some(s) = snap_effort(e, levels) {
            req.reasoning.effort = Some(s);
            degr.downgraded(
                "reasoning.effort",
                format!("effort {e:?} unsupported; snapped to {s:?}"),
            );
        }
    }

    // (e) No reasoning mode at all → clear effort/budget (§7.2).
    let no_reasoning = matches!(caps.reasoning.mode, None | Some(ReasoningMode::None));
    if no_reasoning {
        // A backend with no reasoning mode cannot expose reasoning either; clear the response-path
        // exposure preference so no downstream request encoder mistakes it for reasoning intent.
        req.reasoning.expose = ReasoningExposure::None;
        let present = req.reasoning.effort.is_some_and(|e| e != Effort::None)
            || req.reasoning.budget_tokens.is_some()
            || req.reasoning.enabled == Some(true);
        if present {
            req.reasoning.effort = None;
            req.reasoning.budget_tokens = None;
            req.reasoning.enabled = None;
            degr.dropped("reasoning", "backend has no reasoning mode; cleared effort/budget");
        }
    } else if let Some(budget) = &caps.reasoning.budget {
        // (f) Rejected/ignored explicit budget → clear it, deriving effort if absent (§7.2).
        if matches!(budget.status, Some(BudgetStatus::Rejected) | Some(BudgetStatus::Ignored)) {
            if let Some(bt) = req.reasoning.budget_tokens.take() {
                if req.reasoning.effort.is_none() {
                    // Derive the effort the dropped budget implied, then snap it to a level the
                    // backend supports (mirrors step (d); the budget derivation runs after (d),
                    // so an unsnapped derived level would otherwise reach the codec).
                    let derived = Effort::from_budget_tokens(bt);
                    let effort = caps
                        .reasoning
                        .effort_levels
                        .as_ref()
                        .and_then(|levels| snap_effort(derived, levels))
                        .unwrap_or(derived);
                    req.reasoning.effort = Some(effort);
                }
                degr.dropped(
                    "reasoning.budget_tokens",
                    "explicit budget rejected/ignored by backend; cleared (effort derived)",
                );
            }
        }
        // mode == Budget with effort present but no budget: leave for the codec to compute the
        // table value (no action here).
    }

    Ok(())
}

/// Snap an effort level to the backend's supported set: the nearest lower non-`None` level, or
/// if none is lower, the nearest higher one (so `Minimal` snaps up to `Low` rather than disabling
/// reasoning). Returns `None` when no change is needed (`e == None`, or already supported).
fn snap_effort(e: Effort, levels: &[Effort]) -> Option<Effort> {
    if e == Effort::None || levels.contains(&e) {
        return None;
    }
    let lower = levels.iter().copied().filter(|l| *l != Effort::None && *l < e).max();
    lower.or_else(|| levels.iter().copied().filter(|l| *l != Effort::None && *l > e).min())
}

/// Insert a resolved opaque reasoning item immediately before each tool call in `range` whose
/// `call_id` has an entry in `res.reasoning`.
fn insert_resolved_reasoning(
    req: &mut llm_xlate_core::ir::IrRequest,
    range: std::ops::Range<usize>,
    res: &crate::requirements::Resolutions,
) {
    let old = std::mem::take(&mut req.items);
    let mut out = Vec::with_capacity(old.len() + range.len());
    for (i, item) in old.into_iter().enumerate() {
        if range.contains(&i) {
            if let Item::ToolCall { call_id, .. } = &item {
                if let Some(blob) = res.reasoning.get(call_id) {
                    out.push(Item::Reasoning(ReasoningItem {
                        text: None,
                        summary: Vec::new(),
                        opaque: Some(blob.clone()),
                        id: None,
                    }));
                }
            }
        }
        out.push(item);
    }
    req.items = out;
}
