//! Small JSON-object building helpers used throughout the codec.
//!
//! All wire objects are assembled with [`Ob`] so field order is fixed by insertion order
//! (`serde_json`'s `preserve_order` keeps it), which keeps byte output canonical.

use serde_json::{Map, Value};

/// A fluent, insertion-ordered JSON object builder.
///
/// Insertion order is the serialization order (`preserve_order`), so building objects with a
/// fixed sequence of `.set(..)` calls yields deterministic, byte-stable JSON.
pub(crate) struct Ob(Map<String, Value>);

impl Ob {
    /// A fresh empty object.
    pub(crate) fn new() -> Self {
        Self(Map::new())
    }

    /// Insert a key unconditionally.
    pub(crate) fn set(mut self, k: &str, v: Value) -> Self {
        self.0.insert(k.to_string(), v);
        self
    }

    /// Insert a key only when the value is `Some`.
    pub(crate) fn opt(self, k: &str, v: Option<Value>) -> Self {
        match v {
            Some(v) => self.set(k, v),
            None => self,
        }
    }

    /// Insert a string key only when the value is `Some`.
    pub(crate) fn opt_str(self, k: &str, v: Option<impl Into<String>>) -> Self {
        match v {
            Some(v) => self.set(k, Value::String(v.into())),
            None => self,
        }
    }

    /// Whether the object is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Insert into the underlying map by owned key (used when the key is dynamic).
    pub(crate) fn set_owned(mut self, k: String, v: Value) -> Self {
        self.0.insert(k, v);
        self
    }

    /// Finish and return the object [`Value`].
    pub(crate) fn build(self) -> Value {
        Value::Object(self.0)
    }

    /// Finish and return the raw [`Map`].
    pub(crate) fn into_map(self) -> Map<String, Value> {
        self.0
    }
}

/// Borrow a string field from a JSON object.
pub(crate) fn get_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// Read a `u32` field from a JSON object (permissive about float encodings).
pub(crate) fn get_u32(v: &Value, key: &str) -> Option<u32> {
    v.get(key).and_then(Value::as_u64).map(|n| n as u32)
}

/// Read a `bool` field from a JSON object.
pub(crate) fn get_bool(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(Value::as_bool)
}
