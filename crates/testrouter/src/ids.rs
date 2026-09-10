//! Deterministic id and time sources (plan §3, CONVENTIONS "Determinism").
//!
//! Minted ids and timestamps come from injected traits so tests and `trace replay` are
//! byte-stable. Production uses [`UuidSource`] + [`SystemClock`]; tests use
//! [`SequentialSource`] + [`FixedClock`].
//!
//! Minted id formats (plan §4.1):
//! * trace id — `tr_<id>`
//! * Responses response id — `resp_<id>`
//! * Chat response id — `chatcmpl-<id>`
//! * Anthropic response id — `msg_<id>`

use std::sync::atomic::{AtomicU64, Ordering};

use llm_xlate_core::ir::{Protocol, ResponseId};

/// A source of fresh, opaque id fragments (the part after the `resp_` / `tr_` prefix).
pub trait IdSource: Send + Sync {
    /// A fresh id fragment. Successive calls must return distinct values.
    fn next_fragment(&self) -> String;
}

/// A source of wall-clock time. Codecs never read a clock; the router supplies time here so a
/// [`FixedClock`] makes minted `created_at` / trace timestamps reproducible.
pub trait Clock: Send + Sync {
    /// Unix seconds (for `created` / `created_at`).
    fn now_unix(&self) -> u64;
    /// An RFC-3339 timestamp string (for the trace `ts_start` / `ts_end`).
    fn now_rfc3339(&self) -> String;
}

/// Production id source: random UUID v4 fragments.
#[derive(Debug, Default, Clone)]
pub struct UuidSource;

impl IdSource for UuidSource {
    fn next_fragment(&self) -> String {
        uuid::Uuid::new_v4().simple().to_string()
    }
}

/// Deterministic id source for tests: a monotonic counter rendered in decimal.
#[derive(Debug, Default)]
pub struct SequentialSource {
    counter: AtomicU64,
}

impl SequentialSource {
    /// A fresh sequential source starting at 1.
    pub fn new() -> Self {
        Self { counter: AtomicU64::new(1) }
    }
}

impl IdSource for SequentialSource {
    fn next_fragment(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        format!("{n:08}")
    }
}

/// Production clock: the system time.
#[derive(Debug, Default, Clone)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_unix(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
    fn now_rfc3339(&self) -> String {
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }
}

/// Deterministic clock for tests: a fixed unix-seconds value.
#[derive(Debug, Clone)]
pub struct FixedClock {
    /// The unix-seconds value every call returns.
    pub unix: u64,
}

impl FixedClock {
    /// A fixed clock at `unix` seconds.
    pub fn new(unix: u64) -> Self {
        Self { unix }
    }
}

impl Default for FixedClock {
    fn default() -> Self {
        Self { unix: 1_700_000_000 }
    }
}

impl Clock for FixedClock {
    fn now_unix(&self) -> u64 {
        self.unix
    }
    fn now_rfc3339(&self) -> String {
        chrono::DateTime::from_timestamp(self.unix as i64, 0)
            .unwrap_or_else(|| chrono::DateTime::from_timestamp(0, 0).unwrap())
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }
}

/// Mint a router trace id: `tr_<fragment>`.
pub fn mint_trace_id(src: &dyn IdSource) -> String {
    format!("tr_{}", src.next_fragment())
}

/// Mint a client-facing response id for `client_protocol` (plan §4.1).
pub fn mint_response_id(src: &dyn IdSource, client_protocol: Protocol) -> ResponseId {
    let frag = src.next_fragment();
    let s = match client_protocol {
        Protocol::OaiResponses => format!("resp_{frag}"),
        Protocol::OaiChat => format!("chatcmpl-{frag}"),
        Protocol::Anthropic => format!("msg_{frag}"),
    };
    ResponseId::new(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_is_monotonic_and_padded() {
        let s = SequentialSource::new();
        assert_eq!(s.next_fragment(), "00000001");
        assert_eq!(s.next_fragment(), "00000002");
    }

    #[test]
    fn minted_id_formats() {
        let s = SequentialSource::new();
        assert_eq!(mint_trace_id(&s), "tr_00000001");
        assert_eq!(mint_response_id(&s, Protocol::OaiResponses).as_str(), "resp_00000002");
        assert_eq!(mint_response_id(&s, Protocol::OaiChat).as_str(), "chatcmpl-00000003");
        assert_eq!(mint_response_id(&s, Protocol::Anthropic).as_str(), "msg_00000004");
    }

    #[test]
    fn fixed_clock_is_stable() {
        let c = FixedClock::new(1_726_000_000);
        assert_eq!(c.now_unix(), 1_726_000_000);
        assert_eq!(c.now_unix(), 1_726_000_000);
        assert!(c.now_rfc3339().starts_with("2024-"));
    }
}
