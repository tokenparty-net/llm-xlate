//! The Responses stored-response model behind a trait seam (plan §9, plan §4.4).
//!
//! Two implementations sit behind [`ResponseStore`]: an in-memory [`MemoryStore`] (used by the
//! test harness and the R0/R1 path) and a durable [`FileStore`] that persists each
//! [`StoredResponse`] as `store/<id>.json` (atomic temp + rename). The pipeline only ever sees
//! the trait, so the two are interchangeable.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use tokio::task::AbortHandle;

use llm_xlate::store::{StoredResponse, StoredStatus};
use llm_xlate_core::error::{ErrorKind, XlateError};
use llm_xlate_core::ir::ResponseId;

/// A persisted-response store (plan §9).
///
/// [`ResponseStore::get`], [`delete`](ResponseStore::delete), and
/// [`set_status`](ResponseStore::set_status) are infallible (a missing / unreadable record simply
/// reads as absent). [`chain`](ResponseStore::chain) and
/// [`get_checked`](ResponseStore::get_checked) return a [`XlateError`] so the handlers can
/// distinguish a *missing* id (404) from a *corrupt* record (500).
pub trait ResponseStore: Send + Sync {
    /// Persist (or replace) a stored response.
    fn put(&self, resp: StoredResponse);
    /// Fetch a stored response by id (a corrupt record reads as absent; use
    /// [`get_checked`](ResponseStore::get_checked) to surface corruption).
    fn get(&self, id: &ResponseId) -> Option<StoredResponse>;
    /// Delete a stored response; returns whether it existed.
    fn delete(&self, id: &ResponseId) -> bool;
    /// Set the lifecycle status of a stored response (background / cancel).
    fn set_status(&self, id: &ResponseId, status: StoredStatus) -> bool;
    /// Load the chain ending at `id`, oldest → newest, ready for
    /// [`llm_xlate::store::materialize_chain`]. A missing id is a
    /// [`ErrorKind::NotFound`]; a corrupt on-disk record is a
    /// [`ErrorKind::ServerError`].
    fn chain(&self, id: &ResponseId) -> Result<Vec<StoredResponse>, XlateError>;

    /// Atomically persist `resp` **unless** the stored record for its id is already in the terminal
    /// [`StoredStatus::Cancelled`] state; returns `true` if it was written, `false` if a cancel had
    /// already won. A background turn routes its terminal `completed`/`failed`/… write through this
    /// so a `cancel` that lands in the post-await window cannot be clobbered by a `completed`.
    ///
    /// The default is a non-atomic get-then-put (adequate for single-threaded stores); the concrete
    /// stores override it to hold their write lock across the check and the write.
    fn put_if_not_cancelled(&self, resp: StoredResponse) -> bool {
        if self.get(&resp.id).map(|s| s.status == StoredStatus::Cancelled).unwrap_or(false) {
            return false;
        }
        self.put(resp);
        true
    }

    /// Fetch a stored response, distinguishing a *corrupt* record (`Err`, rendered 500 by the
    /// handler) from a *missing* one (`Ok(None)`, rendered 404).
    ///
    /// Additive to the R0/R1 trait: the default delegates to [`get`](ResponseStore::get) (so a
    /// record simply present-or-absent), and [`FileStore`] overrides it to report a parse error.
    fn get_checked(&self, id: &ResponseId) -> Result<Option<StoredResponse>, XlateError> {
        Ok(self.get(id))
    }

    /// Register a running background task's abort handle so a later `cancel` can stop it. Default:
    /// no-op (a store that does not run background work).
    fn register_task(&self, _id: &ResponseId, _handle: AbortHandle) {}

    /// Abort (and forget) a registered background task, if any. Returns whether one was aborted.
    fn abort_task(&self, _id: &ResponseId) -> bool {
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// In-memory store
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// In-memory [`ResponseStore`] — the test-harness store and the R0/R1 default. Not durable across
/// restarts. Background abort handles are held in memory alongside the records.
#[derive(Default)]
pub struct MemoryStore {
    map: Mutex<HashMap<String, StoredResponse>>,
    tasks: Mutex<HashMap<String, AbortHandle>>,
}

impl MemoryStore {
    /// A fresh empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored responses (test/introspection helper).
    pub fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ResponseStore for MemoryStore {
    fn put(&self, resp: StoredResponse) {
        self.map.lock().unwrap().insert(resp.id.as_str().to_string(), resp);
    }

    fn put_if_not_cancelled(&self, resp: StoredResponse) -> bool {
        // Hold the map lock across the status check and the insert so a concurrent `cancel` cannot
        // interleave between them.
        let mut map = self.map.lock().unwrap();
        if let Some(existing) = map.get(resp.id.as_str()) {
            if existing.status == StoredStatus::Cancelled {
                return false;
            }
        }
        map.insert(resp.id.as_str().to_string(), resp);
        true
    }

    fn get(&self, id: &ResponseId) -> Option<StoredResponse> {
        self.map.lock().unwrap().get(id.as_str()).cloned()
    }

    fn delete(&self, id: &ResponseId) -> bool {
        self.tasks.lock().unwrap().remove(id.as_str());
        self.map.lock().unwrap().remove(id.as_str()).is_some()
    }

    fn set_status(&self, id: &ResponseId, status: StoredStatus) -> bool {
        let mut map = self.map.lock().unwrap();
        if let Some(r) = map.get_mut(id.as_str()) {
            r.status = status;
            true
        } else {
            false
        }
    }

    fn chain(&self, id: &ResponseId) -> Result<Vec<StoredResponse>, XlateError> {
        let map = self.map.lock().unwrap();
        let mut out = Vec::new();
        let mut cur = Some(id.clone());
        while let Some(c) = cur {
            let Some(r) = map.get(c.as_str()) else {
                return Err(not_found(&c));
            };
            cur = r.previous_id.clone();
            out.push(r.clone());
        }
        out.reverse(); // oldest → newest
        Ok(out)
    }

    fn register_task(&self, id: &ResponseId, handle: AbortHandle) {
        self.tasks.lock().unwrap().insert(id.as_str().to_string(), handle);
    }

    fn abort_task(&self, id: &ResponseId) -> bool {
        if let Some(h) = self.tasks.lock().unwrap().remove(id.as_str()) {
            h.abort();
            true
        } else {
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// File store
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A durable [`ResponseStore`] that persists each [`StoredResponse`] as `store/<id>.json` under a
/// data directory (plan §4.4). Writes are atomic (temp file + rename); reads are tolerant of a
/// missing file (absent) and surface a corrupt file as a [`ErrorKind::ServerError`] through
/// [`get_checked`](ResponseStore::get_checked) / [`chain`](ResponseStore::chain). Background abort
/// handles are held in memory (they are process-local by nature).
pub struct FileStore {
    dir: PathBuf,
    /// Serializes writes / renames to the same directory.
    write_lock: Mutex<()>,
    /// In-flight background task handles.
    tasks: Mutex<HashMap<String, AbortHandle>>,
}

impl FileStore {
    /// Open (creating if needed) the `store/` subdirectory under `data_dir`.
    pub fn new(data_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = data_dir.into().join("store");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, write_lock: Mutex::new(()), tasks: Mutex::new(HashMap::new()) })
    }

    /// The path for an id, or `None` if the id is not a safe file name (rejected as absent).
    fn path_for(&self, id: &str) -> Option<PathBuf> {
        id_filename(id).map(|n| self.dir.join(n))
    }

    /// Read + parse a record: `Ok(None)` if the file is absent, `Err` if it is unreadable or
    /// corrupt.
    fn read(&self, id: &str) -> Result<Option<StoredResponse>, XlateError> {
        let Some(path) = self.path_for(id) else {
            return Ok(None);
        };
        match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<StoredResponse>(&bytes) {
                Ok(s) => Ok(Some(s)),
                Err(e) => Err(XlateError::new(
                    ErrorKind::ServerError,
                    format!("corrupt stored response `{id}`: {e}"),
                )),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => {
                Err(XlateError::new(ErrorKind::ServerError, format!("store read error for `{id}`: {e}")))
            }
        }
    }

    /// Atomically write a record to `store/<id>.json`.
    fn write(&self, resp: &StoredResponse) -> std::io::Result<()> {
        let _guard = self.write_lock.lock().unwrap();
        self.write_locked(resp)
    }

    /// The body of [`write`](FileStore::write) with the write lock **already held** by the caller,
    /// so a check-then-write (e.g. [`set_status`], [`put_if_not_cancelled`]) can stay atomic under a
    /// single lock acquisition.
    fn write_locked(&self, resp: &StoredResponse) -> std::io::Result<()> {
        let Some(path) = self.path_for(resp.id.as_str()) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsafe stored-response id `{}`", resp.id),
            ));
        };
        let bytes = serde_json::to_vec(resp).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &bytes)?;
        // Windows rename replaces an existing destination (MoveFileEx REPLACE_EXISTING).
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// List the ids of every stored response currently on disk.
    pub fn list(&self) -> Vec<String> {
        let _guard = self.write_lock.lock().unwrap();
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if let Some(id) = name.strip_suffix(".json") {
                    if !name.ends_with(".json.tmp") {
                        out.push(id.to_string());
                    }
                }
            }
        }
        out.sort();
        out
    }
}

impl ResponseStore for FileStore {
    fn put(&self, resp: StoredResponse) {
        // Best-effort: a write failure is surfaced later as a missing / corrupt read.
        let _ = self.write(&resp);
    }

    fn put_if_not_cancelled(&self, resp: StoredResponse) -> bool {
        // Hold the write lock across the status read and the write so a concurrent `cancel`
        // (`set_status(Cancelled)`, itself under the lock) cannot interleave between them.
        let _guard = self.write_lock.lock().unwrap();
        if let Ok(Some(existing)) = self.read(resp.id.as_str()) {
            if existing.status == StoredStatus::Cancelled {
                return false;
            }
        }
        self.write_locked(&resp).is_ok()
    }

    fn get(&self, id: &ResponseId) -> Option<StoredResponse> {
        self.read(id.as_str()).ok().flatten()
    }

    fn get_checked(&self, id: &ResponseId) -> Result<Option<StoredResponse>, XlateError> {
        self.read(id.as_str())
    }

    fn delete(&self, id: &ResponseId) -> bool {
        self.tasks.lock().unwrap().remove(id.as_str());
        let _guard = self.write_lock.lock().unwrap();
        let Some(path) = self.path_for(id.as_str()) else {
            return false;
        };
        std::fs::remove_file(&path).is_ok()
    }

    fn set_status(&self, id: &ResponseId, status: StoredStatus) -> bool {
        // Atomic read-modify-write under the write lock (a missing / corrupt record cannot have its
        // status set), so a concurrent terminal `put_if_not_cancelled` sees a consistent status.
        let _guard = self.write_lock.lock().unwrap();
        match self.read(id.as_str()) {
            Ok(Some(mut r)) => {
                r.status = status;
                self.write_locked(&r).is_ok()
            }
            _ => false,
        }
    }

    fn chain(&self, id: &ResponseId) -> Result<Vec<StoredResponse>, XlateError> {
        let mut out = Vec::new();
        let mut cur = Some(id.clone());
        while let Some(c) = cur {
            match self.read(c.as_str())? {
                Some(r) => {
                    cur = r.previous_id.clone();
                    out.push(r);
                }
                None => return Err(not_found(&c)),
            }
        }
        out.reverse(); // oldest → newest
        Ok(out)
    }

    fn register_task(&self, id: &ResponseId, handle: AbortHandle) {
        self.tasks.lock().unwrap().insert(id.as_str().to_string(), handle);
    }

    fn abort_task(&self, id: &ResponseId) -> bool {
        if let Some(h) = self.tasks.lock().unwrap().remove(id.as_str()) {
            h.abort();
            true
        } else {
            false
        }
    }
}

/// A `NotFound` error for a missing stored id (rendered 404 in the client dialect).
fn not_found(id: &ResponseId) -> XlateError {
    XlateError::new(ErrorKind::NotFound, format!("stored response `{id}` not found"))
}

/// Map a response id to a safe `<id>.json` file name, or `None` if the id contains anything but
/// `[A-Za-z0-9_.-]` (or a `..` traversal). Router-minted ids (`resp_…`, `chatcmpl-…`, `msg_…`) are
/// always safe; a hostile path param is rejected (read as absent).
fn id_filename(id: &str) -> Option<String> {
    if id.is_empty() || id.len() > 200 || id.contains("..") {
        return None;
    }
    if id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')) {
        Some(format!("{id}.json"))
    } else {
        None
    }
}
