//! The opaque-reasoning-blob sidecar behind a trait seam (plan §4.5, §9).
//!
//! Keyed by `(provider family, upstream model, call id)`, the sidecar caches the opaque reasoning
//! carriers a backend emitted so a later tool turn can replay them even when the client transcript
//! does not carry one. Two implementations sit behind [`Sidecar`]: an in-memory [`MemorySidecar`]
//! (test harness / R0/R1) and a durable [`FileSidecar`] that persists each blob under `sidecar/`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use llm_xlate_core::ir::{CallId, OpaqueBlob, ProviderFamily};

/// The sidecar key: which backend produced the blob, and the tool call it belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SidecarKey {
    /// Provider family that produced the blob.
    pub family: ProviderFamily,
    /// Upstream model string.
    pub model: String,
    /// The tool-call id the blob is bound to.
    pub call_id: CallId,
}

impl SidecarKey {
    /// Construct a sidecar key.
    pub fn new(family: ProviderFamily, model: impl Into<String>, call_id: CallId) -> Self {
        Self { family, model: model.into(), call_id }
    }
}

/// A cache of opaque reasoning blobs keyed by [`SidecarKey`] (plan §4.5).
pub trait Sidecar: Send + Sync {
    /// Store a blob for a `(family, model, call_id)`.
    fn put(&self, key: SidecarKey, blob: OpaqueBlob);
    /// Fetch a blob for a `(family, model, call_id)`.
    fn get(&self, key: &SidecarKey) -> Option<OpaqueBlob>;
    /// Number of cached blobs (introspection).
    fn len(&self) -> usize;
    /// Whether the sidecar is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory [`Sidecar`] — enough for R0/R1. Not durable across restarts.
#[derive(Default)]
pub struct MemorySidecar {
    map: Mutex<HashMap<SidecarKey, OpaqueBlob>>,
}

impl MemorySidecar {
    /// A fresh empty sidecar.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Sidecar for MemorySidecar {
    fn put(&self, key: SidecarKey, blob: OpaqueBlob) {
        self.map.lock().unwrap().insert(key, blob);
    }
    fn get(&self, key: &SidecarKey) -> Option<OpaqueBlob> {
        self.map.lock().unwrap().get(key).cloned()
    }
    fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }
}

/// A durable [`Sidecar`] that persists each blob as `sidecar/<hash>.json` under a data directory
/// (plan §4.5). The file name is a SHA-256 of the `(family, model, call_id)` key, so arbitrary
/// model / call-id strings map to a safe, stable name. Writes are atomic (temp + rename).
pub struct FileSidecar {
    dir: PathBuf,
    write_lock: Mutex<()>,
}

impl FileSidecar {
    /// Open (creating if needed) the `sidecar/` subdirectory under `data_dir`.
    pub fn new(data_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = data_dir.into().join("sidecar");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, write_lock: Mutex::new(()) })
    }

    fn path_for(&self, key: &SidecarKey) -> PathBuf {
        self.dir.join(key_filename(key))
    }
}

impl Sidecar for FileSidecar {
    fn put(&self, key: SidecarKey, blob: OpaqueBlob) {
        let _guard = self.write_lock.lock().unwrap();
        let path = self.path_for(&key);
        if let Ok(bytes) = serde_json::to_vec(&blob) {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, &bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }

    fn get(&self, key: &SidecarKey) -> Option<OpaqueBlob> {
        let bytes = std::fs::read(self.path_for(key)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    fn len(&self) -> usize {
        std::fs::read_dir(&self.dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| {
                        let n = e.file_name();
                        let n = n.to_string_lossy();
                        n.ends_with(".json") && !n.ends_with(".json.tmp")
                    })
                    .count()
            })
            .unwrap_or(0)
    }
}

/// The stable file name for a sidecar key: `<sha256(family\0model\0call_id)>.json`.
fn key_filename(key: &SidecarKey) -> String {
    let canonical = format!("{}\u{0}{}\u{0}{}", key.family.label(), key.model, key.call_id.as_str());
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    let digest = h.finalize();
    let mut s = String::with_capacity(digest.len() * 2 + 5);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s.push_str(".json");
    s
}
