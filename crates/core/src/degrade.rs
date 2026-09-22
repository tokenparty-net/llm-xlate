//! Structured records of lossy translation steps. Every lossy step must record a
//! [`Degradation`]; the router surfaces them as `x-router-degraded` headers.

use serde::{Deserialize, Serialize};

/// The nature of a lossy step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationKind {
    /// A field or capability was dropped entirely.
    Dropped,
    /// A value was rewritten into a different but faithful form.
    Rewritten,
    /// Multiple items were folded together (e.g. foreign provider-tool history into text).
    Folded,
    /// Content was wrapped in a fixed prompt template (`wrap_v1`).
    Wrapped,
    /// A capability was downgraded (e.g. `strict` schema → non-strict).
    Downgraded,
}

impl DegradationKind {
    /// A stable lowercase slug for header rendering.
    pub fn slug(self) -> &'static str {
        match self {
            DegradationKind::Dropped => "dropped",
            DegradationKind::Rewritten => "rewritten",
            DegradationKind::Folded => "folded",
            DegradationKind::Wrapped => "wrapped",
            DegradationKind::Downgraded => "downgraded",
        }
    }
}

/// A single lossy step: what happened, to which field, and a human detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Degradation {
    /// What kind of loss this was.
    pub kind: DegradationKind,
    /// The logical field affected (e.g. `temperature`, `tools[2]`, `reasoning`).
    pub field: String,
    /// A short human-readable explanation.
    pub detail: String,
}

impl Degradation {
    /// Construct a degradation.
    pub fn new(kind: DegradationKind, field: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { kind, field: field.into(), detail: detail.into() }
    }
}

/// An ordered collection of [`Degradation`]s (insertion order preserved for deterministic
/// header rendering).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Degradations(pub Vec<Degradation>);

impl Degradations {
    /// An empty collection.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Push a degradation.
    pub fn push(&mut self, d: Degradation) {
        self.0.push(d);
    }

    /// Extend from another collection, consuming it.
    pub fn extend(&mut self, other: Degradations) {
        self.0.extend(other.0);
    }

    /// Record a [`DegradationKind::Dropped`].
    pub fn dropped(&mut self, field: impl Into<String>, detail: impl Into<String>) {
        self.push(Degradation::new(DegradationKind::Dropped, field, detail));
    }
    /// Record a [`DegradationKind::Rewritten`].
    pub fn rewritten(&mut self, field: impl Into<String>, detail: impl Into<String>) {
        self.push(Degradation::new(DegradationKind::Rewritten, field, detail));
    }
    /// Record a [`DegradationKind::Folded`].
    pub fn folded(&mut self, field: impl Into<String>, detail: impl Into<String>) {
        self.push(Degradation::new(DegradationKind::Folded, field, detail));
    }
    /// Record a [`DegradationKind::Wrapped`].
    pub fn wrapped(&mut self, field: impl Into<String>, detail: impl Into<String>) {
        self.push(Degradation::new(DegradationKind::Wrapped, field, detail));
    }
    /// Record a [`DegradationKind::Downgraded`].
    pub fn downgraded(&mut self, field: impl Into<String>, detail: impl Into<String>) {
        self.push(Degradation::new(DegradationKind::Downgraded, field, detail));
    }

    /// Whether there are no degradations.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of degradations.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate in insertion order.
    pub fn iter(&self) -> std::slice::Iter<'_, Degradation> {
        self.0.iter()
    }

    /// Render the deterministic `x-router-degraded` header value.
    ///
    /// Entries are `field=kind`, separated by `;`, in first-occurrence order. Identical
    /// `field=kind` pairs are deduplicated and rendered once with a `*N` count suffix when
    /// they occur more than once (e.g. `instructions=wrapped*177`) — `*` is plaintext ASCII
    /// and never appears in a field name or kind slug, so the value stays trivially parseable.
    /// Deduplication is essential: a long conversation can wrap hundreds of mid-context
    /// system turns, and rendering each one verbatim produces a multi-kilobyte header that
    /// overflows a downstream proxy's header buffer (nginx `proxy_buffer_size`), turning an
    /// otherwise-successful response into a 502.
    ///
    /// The rendered value is additionally hard-capped at [`Self::MAX_HEADER_LEN`] bytes as a
    /// belt-and-suspenders guard against any future high-cardinality degradation source; when
    /// the cap is hit, whole trailing entries are dropped and a final `truncated=N` entry
    /// records how many distinct entries were omitted.
    ///
    /// Returns an empty string when there are no degradations. Full, un-deduplicated detail
    /// is preserved in the trace record; this is only the header projection.
    pub fn render_header_value(&self) -> String {
        // Collapse identical field=kind pairs, preserving first-occurrence order.
        let mut order: Vec<(usize, DegradationKind, u32)> = Vec::new();
        for d in &self.0 {
            match order.iter_mut().find(|(fi, k, _)| {
                *k == d.kind && self.0[*fi].field == d.field
            }) {
                Some((_, _, n)) => *n += 1,
                None => order.push((
                    self.0.iter().position(|x| x.field == d.field && x.kind == d.kind).unwrap(),
                    d.kind,
                    1,
                )),
            }
        }

        let mut out = String::new();
        for (idx, (fi, kind, count)) in order.iter().enumerate() {
            let entry_len = self.0[*fi].field.len()
                + 1 // '='
                + kind.slug().len()
                + if *count > 1 { 1 + count_digits(*count) } else { 0 }
                + if idx > 0 { 1 } else { 0 }; // ';'
            // If appending this entry would overflow the cap, stop and record how many
            // distinct entries (including this one) were dropped.
            if !out.is_empty() && out.len() + entry_len > Self::MAX_HEADER_LEN {
                let omitted = order.len() - idx;
                out.push_str(&format!(";truncated={omitted}"));
                break;
            }
            if idx > 0 {
                out.push(';');
            }
            out.push_str(&self.0[*fi].field);
            out.push('=');
            out.push_str(kind.slug());
            if *count > 1 {
                out.push('*');
                out.push_str(&count.to_string());
            }
        }
        out
    }

    /// Maximum rendered length of the degraded header value, in bytes. Kept well under a
    /// typical proxy header buffer so the value can never by itself cause a 502.
    pub const MAX_HEADER_LEN: usize = 1024;
}

fn count_digits(mut n: u32) -> usize {
    let mut d = 1;
    while n >= 10 {
        n /= 10;
        d += 1;
    }
    d
}

impl IntoIterator for Degradations {
    type Item = Degradation;
    type IntoIter = std::vec::IntoIter<Degradation>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Degradations {
    type Item = &'a Degradation;
    type IntoIter = std::slice::Iter<'a, Degradation>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_renders_empty() {
        assert_eq!(Degradations::new().render_header_value(), "");
    }

    #[test]
    fn single_entries_render_field_equals_kind() {
        let mut d = Degradations::new();
        d.dropped("temperature", "");
        d.wrapped("instructions", "");
        assert_eq!(d.render_header_value(), "temperature=dropped;instructions=wrapped");
    }

    #[test]
    fn identical_pairs_collapse_with_count_suffix() {
        let mut d = Degradations::new();
        d.dropped("ext.anthropic.system_headers.billing-header", "");
        for _ in 0..177 {
            d.wrapped("instructions", "");
        }
        // 178 raw degradations collapse to two entries; length stays tiny regardless of turns.
        assert_eq!(
            d.render_header_value(),
            "ext.anthropic.system_headers.billing-header=dropped;instructions=wrapped*177"
        );
        assert!(d.render_header_value().len() < 100);
    }

    #[test]
    fn same_field_different_kind_stays_distinct() {
        let mut d = Degradations::new();
        d.wrapped("instructions", "");
        d.dropped("instructions.effort", "");
        d.wrapped("instructions", "");
        d.dropped("instructions.effort", "");
        assert_eq!(
            d.render_header_value(),
            "instructions=wrapped*2;instructions.effort=dropped*2"
        );
    }

    #[test]
    fn first_occurrence_order_is_preserved() {
        let mut d = Degradations::new();
        d.wrapped("b", "");
        d.dropped("a", "");
        d.wrapped("b", "");
        assert_eq!(d.render_header_value(), "b=wrapped*2;a=dropped");
    }

    #[test]
    fn high_cardinality_is_capped_with_truncated_marker() {
        let mut d = Degradations::new();
        for i in 0..5000 {
            // Every field distinct so nothing dedups: this is the pathological case the cap guards.
            d.dropped(format!("field_{i}"), "");
        }
        let v = d.render_header_value();
        assert!(v.len() <= Degradations::MAX_HEADER_LEN, "len={}", v.len());
        assert!(v.contains(";truncated="), "expected truncation marker: {v}");
        // The marker reports a positive count of omitted distinct entries.
        let n: usize = v.rsplit(";truncated=").next().unwrap().parse().unwrap();
        assert!(n > 0);
    }
}
