//! Default-off per-tenant request capture.
//!
//! When `MEMRA_CAPTURE_DIR` is set, tenants explicitly marked through the admin surface
//! get their SETTLED request/response pairs appended as JSONL rows under that directory.
//! The mark carries a reason chosen by the provisioning system (`trial` or `consent`);
//! the server never infers one, and an unmarked tenant is never captured. The mark is
//! re-read at settle time, so clearing it stops capture immediately, including for
//! requests already in flight. A `trial`-marked tenant is additionally captured only
//! while its settled prepaid balance is non-negative: a request that spends past the
//! marked allowance is dropped rather than retained (under-capture by construction).
//!
//! Capture is fail-open for serving: rows go through a bounded queue to one writer
//! thread, a full queue or a dead writer DROPS the row (counted, reported on the admin
//! status read) and never delays or fails a request. Row content never reaches the
//! server log; the capture files are the only sink.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};

pub(crate) const FORMAT: &str = "memra.capture.v1";
const QUEUE_DEPTH: usize = 4096;
const ROTATE_BYTES: u64 = 256 * 1024 * 1024;
const STATE_FILE: &str = "state.toml";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CaptureMode {
    Trial,
    Consent,
}

impl CaptureMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Trial => "trial",
            Self::Consent => "consent",
        }
    }

    /// The row tag. `consent` rows are tagged `opt_in` so the corpus reader never has
    /// to know the admin surface's vocabulary.
    pub(crate) fn posture(self) -> &'static str {
        match self {
            Self::Trial => "trial",
            Self::Consent => "opt_in",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "trial" => Some(Self::Trial),
            "consent" => Some(Self::Consent),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
struct StateFile {
    #[serde(default)]
    tenants: Vec<StateRow>,
}

#[derive(Serialize, Deserialize)]
struct StateRow {
    tenant: String,
    mode: String,
}

struct StoreInner {
    state_path: PathBuf,
    modes: RwLock<HashMap<String, CaptureMode>>,
    /// Serializes state-file rewrites; the in-memory map is the read path.
    state_lock: Mutex<()>,
    tx: SyncSender<Vec<u8>>,
    dropped: AtomicU64,
    written: Arc<AtomicU64>,
}

#[derive(Clone)]
pub(crate) struct CaptureStore {
    inner: Arc<StoreInner>,
}

impl CaptureStore {
    pub(crate) fn from_env(ledger_enabled: bool) -> Result<Option<Self>, String> {
        let Some(dir) = std::env::var_os("MEMRA_CAPTURE_DIR") else {
            return Ok(None);
        };
        if dir.is_empty() {
            return Err("MEMRA_CAPTURE_DIR must not be empty".into());
        }
        if !ledger_enabled {
            return Err(
                "MEMRA_CAPTURE_DIR requires MEMRA_REQUEST_LEDGER; capture rows are settled \
                 request receipts and cannot exist without the ledger"
                    .into(),
            );
        }
        Self::open(Path::new(&dir)).map(Some)
    }

    pub(crate) fn open(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("create MEMRA_CAPTURE_DIR {}: {e}", dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            use std::os::unix::fs::PermissionsExt as _;
            // hermes (fixed 2026-08-19): `set_permissions` FOLLOWS symlinks and the dir
            // is operator-supplied — a symlink parked at MEMRA_CAPTURE_DIR
            // (`create_dir_all` succeeds when the link's target exists) redirected the
            // 0750 chmod onto whatever it pointed at. Refuse symlinks and foreign-owned
            // directories BEFORE touching modes: the capture store never operates
            // through a link or inside another user's directory.
            let meta = std::fs::symlink_metadata(dir)
                .map_err(|e| format!("stat MEMRA_CAPTURE_DIR {}: {e}", dir.display()))?;
            if meta.file_type().is_symlink() {
                return Err(format!(
                    "MEMRA_CAPTURE_DIR {} is a symlink; refusing to chmod through it — \
                     point the flag at the real directory",
                    dir.display()
                ));
            }
            if !meta.is_dir() {
                return Err(format!(
                    "MEMRA_CAPTURE_DIR {} is not a directory",
                    dir.display()
                ));
            }
            let euid = unsafe { libc::geteuid() };
            if meta.uid() != euid {
                return Err(format!(
                    "MEMRA_CAPTURE_DIR {} is owned by uid {} but the server runs as uid \
                     {euid}; refusing to take over another user's directory",
                    dir.display(),
                    meta.uid()
                ));
            }
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o750))
                .map_err(|e| format!("chmod MEMRA_CAPTURE_DIR {}: {e}", dir.display()))?;
        }
        let state_path = dir.join(STATE_FILE);
        let modes = load_state(&state_path)?;
        let written = Arc::new(AtomicU64::new(0));
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(QUEUE_DEPTH);
        spawn_writer(dir.to_path_buf(), rx, Arc::clone(&written))?;
        eprintln!(
            "[capture] per-tenant request capture enabled: {} ({} tenant(s) marked)",
            dir.display(),
            modes.len(),
        );
        Ok(Self {
            inner: Arc::new(StoreInner {
                state_path,
                modes: RwLock::new(modes),
                state_lock: Mutex::new(()),
                tx,
                dropped: AtomicU64::new(0),
                written,
            }),
        })
    }

    /// Cheap request-start check: is this tenant marked at all right now?
    /// The authoritative decision is re-taken at settle via `mode()`.
    pub(crate) fn is_armed(&self, tenant: &str) -> bool {
        self.inner
            .modes
            .read()
            .map(|modes| modes.contains_key(tenant))
            .unwrap_or(false)
    }

    pub(crate) fn mode(&self, tenant: &str) -> Option<CaptureMode> {
        self.inner
            .modes
            .read()
            .ok()
            .and_then(|modes| modes.get(tenant).copied())
    }

    /// Set (`Some`) or clear (`None`) a tenant's capture mark. The durable state file is
    /// rewritten before the in-memory map changes, so a crash never resurrects a cleared
    /// mark — the file is the boot source and clearing is the privacy-critical direction.
    pub(crate) fn set_mode(&self, tenant: &str, mode: Option<CaptureMode>) -> Result<(), String> {
        let _guard = self
            .inner
            .state_lock
            .lock()
            .map_err(|_| "capture state lock is poisoned".to_string())?;
        let mut next = self
            .inner
            .modes
            .read()
            .map_err(|_| "capture mode map is poisoned".to_string())?
            .clone();
        match mode {
            Some(mode) => next.insert(tenant.to_string(), mode),
            None => next.remove(tenant),
        };
        persist_state(&self.inner.state_path, &next)?;
        *self
            .inner
            .modes
            .write()
            .map_err(|_| "capture mode map is poisoned".to_string())? = next;
        Ok(())
    }

    /// Fail-open enqueue: serialization or a saturated/dead writer drops the row and
    /// bumps the counter. Never blocks, never errors into the serving path.
    pub(crate) fn submit(&self, row: &serde_json::Value) {
        let mut bytes = match serde_json::to_vec(row) {
            Ok(bytes) => bytes,
            Err(_) => {
                self.inner.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        bytes.push(b'\n');
        match self.inner.tx.try_send(bytes) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.inner.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn status(&self, tenant: &str) -> serde_json::Value {
        serde_json::json!({
            "tenant": tenant,
            "mode": self.mode(tenant).map(CaptureMode::as_str).unwrap_or("off"),
            "rows_written": self.inner.written.load(Ordering::Relaxed),
            "rows_dropped": self.inner.dropped.load(Ordering::Relaxed),
        })
    }

    #[cfg(test)]
    pub(crate) fn dropped(&self) -> u64 {
        self.inner.dropped.load(Ordering::Relaxed)
    }
}

/// Admin-surface mode strings: `trial` / `consent` mark, `off` clears.
pub(crate) fn parse_mode_request(value: &str) -> Result<Option<CaptureMode>, ()> {
    if value == "off" {
        return Ok(None);
    }
    CaptureMode::parse(value).map(Some).ok_or(())
}

fn load_state(path: &Path) -> Result<HashMap<String, CaptureMode>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(err) => return Err(format!("read capture state {}: {err}", path.display())),
    };
    let file: StateFile = toml::from_str(&text)
        .map_err(|e| format!("parse capture state {}: {e}", path.display()))?;
    let mut modes = HashMap::new();
    for row in file.tenants {
        let mode = CaptureMode::parse(&row.mode).ok_or_else(|| {
            format!(
                "capture state {}: tenant {:?} has unknown mode {:?}",
                path.display(),
                row.tenant,
                row.mode,
            )
        })?;
        modes.insert(row.tenant, mode);
    }
    Ok(modes)
}

fn persist_state(path: &Path, modes: &HashMap<String, CaptureMode>) -> Result<(), String> {
    let mut tenants: Vec<StateRow> = modes
        .iter()
        .map(|(tenant, mode)| StateRow {
            tenant: tenant.clone(),
            mode: mode.as_str().to_string(),
        })
        .collect();
    tenants.sort_by(|a, b| a.tenant.cmp(&b.tenant));
    let text = toml::to_string(&StateFile { tenants })
        .map_err(|e| format!("serialize capture state: {e}"))?;
    let tmp = path.with_extension("toml.tmp");
    write_private(&tmp, text.as_bytes())?;
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("commit capture state {}: {e}", path.display()))?;
    sync_parent(path)
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;

    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    options.mode(0o640);
    let mut file = options
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_data())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn sync_parent(path: &Path) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| format!("sync capture directory {}: {e}", parent.display()))
}

/// One writer owns the row files. Files are `capture-NNNNNN.jsonl`, appended in order,
/// rolled by size. Boot resumes on the highest existing index rather than renaming
/// anything — rows are append-only evidence.
fn spawn_writer(
    dir: PathBuf,
    rx: Receiver<Vec<u8>>,
    written: Arc<AtomicU64>,
) -> Result<(), String> {
    let mut index = highest_index(&dir)?;
    let mut file = open_row_file(&dir, index)?;
    let mut size = file
        .metadata()
        .map_err(|e| format!("stat capture row file: {e}"))?
        .len();
    std::thread::Builder::new()
        .name("capture-writer".into())
        .spawn(move || {
            while let Ok(row) = rx.recv() {
                if size >= ROTATE_BYTES {
                    index += 1;
                    match open_row_file(&dir, index) {
                        Ok(next) => {
                            file = next;
                            size = 0;
                        }
                        Err(err) => {
                            eprintln!("[capture] ERROR: rotate row file: {err}");
                            return; // sender's try_send now counts every row as dropped
                        }
                    }
                }
                match file.write_all(&row).and_then(|()| file.flush()) {
                    Ok(()) => {
                        size += row.len() as u64;
                        written.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(err) => {
                        eprintln!("[capture] ERROR: append row: {err}");
                        return;
                    }
                }
            }
        })
        .map_err(|e| format!("spawn capture writer: {e}"))?;
    Ok(())
}

fn row_file_path(dir: &Path, index: u64) -> PathBuf {
    dir.join(format!("capture-{index:06}.jsonl"))
}

fn open_row_file(dir: &Path, index: u64) -> Result<File, String> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;

    let path = row_file_path(dir, index);
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    options.mode(0o640).custom_flags(libc::O_NOFOLLOW);
    options
        .open(&path)
        .map_err(|e| format!("open capture row file {}: {e}", path.display()))
}

fn highest_index(dir: &Path) -> Result<u64, String> {
    let mut highest = 0;
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("read MEMRA_CAPTURE_DIR {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read MEMRA_CAPTURE_DIR entry: {e}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(index) = name
            .strip_prefix("capture-")
            .and_then(|rest| rest.strip_suffix(".jsonl"))
            .and_then(|digits| digits.parse::<u64>().ok())
        {
            highest = highest.max(index);
        }
    }
    Ok(highest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "memra-capture-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ))
    }

    fn wait_for_rows(dir: &Path, count: usize) -> Vec<serde_json::Value> {
        let path = row_file_path(dir, 0);
        for _ in 0..200 {
            if let Ok(text) = std::fs::read_to_string(&path) {
                let rows: Vec<serde_json::Value> = text
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                if rows.len() >= count {
                    return rows;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("capture rows never arrived in {}", path.display());
    }

    #[test]
    #[cfg(unix)]
    fn open_refuses_a_symlinked_capture_dir_and_never_chmods_through_it() {
        use std::os::unix::fs::PermissionsExt as _;
        // the pre-fix failure: MEMRA_CAPTURE_DIR parked as a symlink redirected the
        // 0750 chmod onto the link's TARGET. The store must refuse the link and leave
        // the target's mode untouched.
        let target = test_dir("symlink-target");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).unwrap();
        let link = test_dir("symlink-link");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let err = match CaptureStore::open(&link) {
            Err(e) => e,
            Ok(_) => panic!("symlinked capture dir was accepted"),
        };
        assert!(err.contains("symlink"), "{err}");
        let mode = std::fs::metadata(&target).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "target mode must be untouched, got {mode:o}");

        std::fs::remove_file(&link).unwrap();
        std::fs::remove_dir_all(&target).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn open_tightens_a_real_dir_to_0750() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = test_dir("chmod");
        let store = CaptureStore::open(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o750);
        drop(store);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn marks_persist_and_clear_across_reopen() {
        let dir = test_dir("state");
        let store = CaptureStore::open(&dir).unwrap();
        assert_eq!(store.mode("acme"), None);
        store.set_mode("acme", Some(CaptureMode::Trial)).unwrap();
        store.set_mode("beta", Some(CaptureMode::Consent)).unwrap();
        drop(store);

        let store = CaptureStore::open(&dir).unwrap();
        assert_eq!(store.mode("acme"), Some(CaptureMode::Trial));
        assert_eq!(store.mode("beta"), Some(CaptureMode::Consent));
        store.set_mode("acme", None).unwrap();
        drop(store);

        let store = CaptureStore::open(&dir).unwrap();
        assert_eq!(store.mode("acme"), None, "cleared mark must not resurrect");
        assert_eq!(store.mode("beta"), Some(CaptureMode::Consent));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn submitted_rows_land_as_jsonl() {
        let dir = test_dir("rows");
        let store = CaptureStore::open(&dir).unwrap();
        store.submit(&serde_json::json!({ "format": FORMAT, "tenant": "acme", "n": 1 }));
        store.submit(&serde_json::json!({ "format": FORMAT, "tenant": "acme", "n": 2 }));
        let rows = wait_for_rows(&dir, 2);
        assert_eq!(rows[0]["n"], 1);
        assert_eq!(rows[1]["n"], 2);
        assert_eq!(store.dropped(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unknown_mode_string_is_rejected_and_off_clears() {
        assert!(parse_mode_request("nonsense").is_err());
        assert_eq!(parse_mode_request("off"), Ok(None));
        assert_eq!(parse_mode_request("trial"), Ok(Some(CaptureMode::Trial)));
        assert_eq!(
            parse_mode_request("consent"),
            Ok(Some(CaptureMode::Consent))
        );
    }

    #[test]
    fn corrupt_state_file_fails_loud_at_open() {
        let dir = test_dir("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(STATE_FILE),
            "tenants = [{tenant = \"a\", mode = \"paid\"}]\n",
        )
        .unwrap();
        let err = match CaptureStore::open(&dir) {
            Ok(_) => panic!("a corrupt state file must refuse to load"),
            Err(err) => err,
        };
        assert!(err.contains("unknown mode"), "got: {err}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
