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

    /// Render the deterministic `x-router-degraded` header value: `field=kind;field=kind`
    /// in insertion order. Returns an empty string when there are no degradations.
    pub fn render_header_value(&self) -> String {
        let mut out = String::new();
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                out.push(';');
            }
            out.push_str(&d.field);
            out.push('=');
            out.push_str(d.kind.slug());
        }
        out
    }
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
