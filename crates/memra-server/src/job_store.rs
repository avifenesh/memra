//! In-memory reference implementation of [`crate::metering::JobStore`] (memra#550,
//! `docs/decisions/COMPLETE-RESULT-PATH-V1.md`).
//!
//! Bounded map, TTL sweep, resident-byte cap. Both the TTL and the cap are deployment
//! decisions the design doc explicitly leaves open ("Owed" section, item 4): the defaults
//! below are conservative starting points, not a measured number, and both are
//! env-configurable so the owner can move them without a code change. A deployment that
//! wants a job to survive a process restart, or a store shared across replicas, supplies its
//! own [`crate::metering::JobStore`] instead of this one, the same way it supplies its own
//! `Metering`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::metering::{JobRecord, JobStatus, JobStore, JobStoreError};

/// Conservative default: fifteen minutes after a job goes terminal, its buffered output is
/// evicted. A caller that never polls within this window loses the text the same way a caller
/// that disconnects mid-stream today loses nothing it was not already billed for (the design
/// doc's "Uncollected results" clause). Env-configurable; see [`ttl_from_env`].
pub const DEFAULT_TTL_SECS: u64 = 900;

/// Conservative default resident cap for the whole store: 64 MiB of buffered job output
/// across every job, terminal or not. Sized to be small enough that a background feature no
/// deployment has opted into cannot become an unbounded memory sink, not to fit any measured
/// workload. Env-configurable; see [`max_bytes_from_env`].
pub const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Env var naming the TTL, in whole seconds, a finished job's output survives before the
/// sweep evicts it. Unset, empty, zero, or unparsable falls back to [`DEFAULT_TTL_SECS`].
pub const TTL_ENV: &str = "MEMRA_BACKGROUND_JOB_TTL_SECS";

/// Env var naming the store's resident-byte cap. Unset, empty, zero, or unparsable falls
/// back to [`DEFAULT_MAX_BYTES`].
pub const MAX_BYTES_ENV: &str = "MEMRA_BACKGROUND_JOB_MAX_BYTES";

pub fn ttl_from_env() -> Duration {
    std::env::var(TTL_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_TTL_SECS))
}

pub fn max_bytes_from_env() -> usize {
    std::env::var(MAX_BYTES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|bytes| *bytes > 0)
        .unwrap_or(DEFAULT_MAX_BYTES)
}

/// Rough resident cost of one record: the serialized size of its buffered output plus its
/// error text plus a fixed bookkeeping allowance. Approximate on purpose (an exact accounting
/// of `serde_json::Value` heap layout is not worth carrying here); it is used only to keep the
/// store's aggregate bounded, not to bill anyone.
fn record_size_bytes(record: &JobRecord) -> usize {
    const BOOKKEEPING_OVERHEAD: usize = 96;
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len())
                .ok_or_else(|| std::io::Error::other("job size overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    if let Some(output) = &record.output
        && serde_json::to_writer(&mut count, output).is_err()
    {
        return usize::MAX;
    }
    let output_len = count.0;
    let error_len = record.error.as_ref().map(|e| e.len()).unwrap_or(0);
    // Reserve enough space at admission for the fixed terminal storage-failure record.
    output_len
        .saturating_add(error_len)
        .saturating_add(BOOKKEEPING_OVERHEAD)
        .max(256)
}

struct Entry {
    record: JobRecord,
    size_bytes: usize,
    /// Set the moment the record's status becomes terminal; the TTL sweep reads this, not
    /// the record's creation time, so a long-running in-progress job is never evicted for
    /// simply taking a while.
    finished_at: Option<Instant>,
    /// Complete output held during settlement, invisible to readers. Mutation,
    /// cancellation and collection refuse while publication owns this reservation.
    pending: Option<JobRecord>,
}

struct Inner {
    map: HashMap<String, Entry>,
    total_bytes: usize,
}

/// The stock in-memory [`JobStore`]. One process, one lifetime: nothing here survives a
/// restart, and nothing here is shared across replicas. See the module doc for why that is
/// the deliberate scope of the stock implementation.
pub struct InMemoryJobStore {
    ttl: Duration,
    max_bytes: usize,
    inner: Mutex<Inner>,
}

impl InMemoryJobStore {
    pub fn new(ttl: Duration, max_bytes: usize) -> Self {
        InMemoryJobStore {
            ttl,
            max_bytes,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                total_bytes: 0,
            }),
        }
    }

    /// Build the store from the two env vars, falling back to the conservative defaults.
    pub fn from_env() -> Self {
        Self::new(ttl_from_env(), max_bytes_from_env())
    }

    fn sweep_locked(&self, inner: &mut Inner) -> usize {
        let now = Instant::now();
        let ttl = self.ttl;
        let expired: Vec<String> = inner
            .map
            .iter()
            .filter(|(_, e)| e.finished_at.is_some_and(|f| now.duration_since(f) > ttl))
            .map(|(k, _)| k.clone())
            .collect();
        for key in &expired {
            if let Some(e) = inner.map.remove(key) {
                inner.total_bytes = inner.total_bytes.saturating_sub(e.size_bytes);
            }
        }
        expired.len()
    }
}

impl JobStore for InMemoryJobStore {
    fn reserve_output(&self, id: &str, bytes: usize) -> Result<(), JobStoreError> {
        let mut inner = self.inner.lock().unwrap();
        let entry = inner.map.get(id).ok_or(JobStoreError::NotFound)?;
        if entry.record.status.is_terminal() || entry.pending.is_some() {
            return Err(JobStoreError::AlreadyTerminal);
        }
        let old = entry.size_bytes;
        let size = old.max(bytes);
        let projected = inner
            .total_bytes
            .saturating_sub(old)
            .checked_add(size)
            .ok_or(JobStoreError::CapacityExceeded)?;
        if projected > self.max_bytes {
            return Err(JobStoreError::CapacityExceeded);
        }
        inner.total_bytes = projected;
        inner.map.get_mut(id).unwrap().size_bytes = size;
        Ok(())
    }

    fn publish_terminal(
        &self,
        id: &str,
        record: JobRecord,
        settle: &mut dyn FnMut() -> Result<(), String>,
    ) -> Result<(), JobStoreError> {
        if !record.status.is_terminal() {
            return Err(JobStoreError::PublicationUnsupported);
        }
        let size = record_size_bytes(&record);
        {
            let mut inner = self.inner.lock().unwrap();
            let entry = inner.map.get(id).ok_or(JobStoreError::NotFound)?;
            if entry.record.status.is_terminal() || entry.pending.is_some() {
                return Err(JobStoreError::AlreadyTerminal);
            }
            let held = entry.size_bytes.max(size);
            let projected = inner
                .total_bytes
                .saturating_sub(entry.size_bytes)
                .checked_add(held)
                .ok_or(JobStoreError::CapacityExceeded)?;
            if projected > self.max_bytes {
                return Err(JobStoreError::CapacityExceeded);
            }
            inner.total_bytes = projected;
            let entry = inner.map.get_mut(id).unwrap();
            entry.size_bytes = held;
            entry.pending = Some(record);
        }
        // The callback may read the store. No store lock is held, and the pending
        // reservation prevents mutation/take/cancel from invalidating the commit.
        let settled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(settle));
        let mut inner = self.inner.lock().unwrap();
        let entry = inner
            .map
            .get_mut(id)
            .expect("pending publication owns its entry");
        let pending = entry.pending.take().expect("pending record is retained");
        if !matches!(settled, Ok(Ok(()))) {
            return Err(JobStoreError::SettlementFailed);
        }
        let old = entry.size_bytes;
        entry.record = pending;
        entry.finished_at = Some(Instant::now());
        entry.size_bytes = size;
        inner.total_bytes = inner.total_bytes.saturating_sub(old) + size;
        Ok(())
    }

    fn put(&self, id: &str, record: JobRecord) -> Result<(), JobStoreError> {
        let mut inner = self.inner.lock().unwrap();
        self.sweep_locked(&mut inner);

        match inner.map.get(id) {
            Some(existing)
                if existing.record.status.is_terminal() || existing.pending.is_some() =>
            {
                return Err(JobStoreError::AlreadyTerminal);
            }
            None if record.status != JobStatus::Queued => {
                // An id the store has no record of may only be admitted as Queued (the
                // write that mints it). Anything else under an unknown id is a stale write:
                // most likely a worker finishing after its job was already cancelled and
                // collected (take), or TTL-evicted, and it must not resurrect a job the
                // store has already forgotten (revuto finding on the initial version of
                // this file: without this check, take()/TTL eviction of a Cancelled record
                // opened exactly that window).
                return Err(JobStoreError::NotFound);
            }
            _ => {}
        }

        let size = record_size_bytes(&record).max(if record.status.is_terminal() {
            0
        } else {
            inner.map.get(id).map(|e| e.size_bytes).unwrap_or(0)
        });
        let finished_at = record.status.is_terminal().then(Instant::now);
        let existing_size = inner.map.get(id).map(|e| e.size_bytes).unwrap_or(0);
        let projected = inner
            .total_bytes
            .saturating_sub(existing_size)
            .checked_add(size)
            .ok_or(JobStoreError::CapacityExceeded)?;
        if projected > self.max_bytes {
            return Err(JobStoreError::CapacityExceeded);
        }

        inner.total_bytes = projected;
        inner.map.insert(
            id.to_string(),
            Entry {
                record,
                size_bytes: size,
                finished_at,
                pending: None,
            },
        );
        Ok(())
    }

    fn get(&self, id: &str) -> Option<JobRecord> {
        let mut inner = self.inner.lock().unwrap();
        self.sweep_locked(&mut inner);
        inner.map.get(id).map(|e| e.record.clone())
    }

    fn take(&self, id: &str) -> Option<JobRecord> {
        let mut inner = self.inner.lock().unwrap();
        self.sweep_locked(&mut inner);
        if inner
            .map
            .get(id)
            .is_some_and(|entry| entry.pending.is_some())
        {
            return None;
        }
        inner.map.remove(id).map(|e| {
            inner.total_bytes = inner.total_bytes.saturating_sub(e.size_bytes);
            e.record
        })
    }

    fn cancel(&self, id: &str) -> Result<(), JobStoreError> {
        let mut inner = self.inner.lock().unwrap();
        self.sweep_locked(&mut inner);
        match inner.map.get_mut(id) {
            None => Err(JobStoreError::NotFound),
            Some(e) if e.record.status.is_terminal() || e.pending.is_some() => {
                Err(JobStoreError::AlreadyTerminal)
            }
            Some(e) => {
                e.record.status = JobStatus::Cancelled;
                e.finished_at = Some(Instant::now());
                Ok(())
            }
        }
    }

    fn sweep(&self) -> usize {
        let mut inner = self.inner.lock().unwrap();
        self.sweep_locked(&mut inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::thread::sleep;

    fn store(ttl: Duration, max_bytes: usize) -> InMemoryJobStore {
        InMemoryJobStore::new(ttl, max_bytes)
    }

    #[test]
    fn poll_unknown_id_is_none() {
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        assert_eq!(s.get("nope"), None);
        assert_eq!(s.take("nope"), None);
        assert_eq!(s.cancel("nope"), Err(JobStoreError::NotFound));
    }

    #[test]
    fn queued_to_completed_round_trip() {
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        s.put("job-1", JobRecord::queued()).unwrap();
        assert_eq!(s.get("job-1").unwrap().status, JobStatus::Queued);

        s.put(
            "job-1",
            JobRecord {
                status: JobStatus::InProgress,
                output: None,
                error: None,
            },
        )
        .unwrap();
        assert_eq!(s.get("job-1").unwrap().status, JobStatus::InProgress);

        s.put(
            "job-1",
            JobRecord {
                status: JobStatus::Completed,
                output: Some(json!({"text": "hello"})),
                error: None,
            },
        )
        .unwrap();
        let rec = s.get("job-1").unwrap();
        assert_eq!(rec.status, JobStatus::Completed);
        assert_eq!(rec.output, Some(json!({"text": "hello"})));

        // get() peeks; the record is still there after.
        assert!(s.get("job-1").is_some());
        // take() consumes.
        let taken = s.take("job-1").unwrap();
        assert_eq!(taken.status, JobStatus::Completed);
        assert_eq!(s.get("job-1"), None);
    }

    #[test]
    fn cancel_after_completion_is_refused_and_does_not_clobber() {
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        s.put("job-2", JobRecord::queued()).unwrap();
        s.put(
            "job-2",
            JobRecord {
                status: JobStatus::Completed,
                output: Some(json!({"text": "final answer"})),
                error: None,
            },
        )
        .unwrap();

        assert_eq!(s.cancel("job-2"), Err(JobStoreError::AlreadyTerminal));

        // The completed record is untouched: a cancel racing a completion never overwrites
        // the result a poll would otherwise see.
        let rec = s.get("job-2").unwrap();
        assert_eq!(rec.status, JobStatus::Completed);
        assert_eq!(rec.output, Some(json!({"text": "final answer"})));
    }

    #[test]
    fn cancel_in_progress_keeps_partial_output() {
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        s.put("job-3", JobRecord::queued()).unwrap();
        s.put(
            "job-3",
            JobRecord {
                status: JobStatus::InProgress,
                output: Some(json!({"text": "partial so far"})),
                error: None,
            },
        )
        .unwrap();

        s.cancel("job-3").unwrap();
        let rec = s.get("job-3").unwrap();
        assert_eq!(rec.status, JobStatus::Cancelled);
        assert_eq!(rec.output, Some(json!({"text": "partial so far"})));

        // A second cancel on an already-cancelled (terminal) job is refused, same as a
        // cancel after a normal completion.
        assert_eq!(s.cancel("job-3"), Err(JobStoreError::AlreadyTerminal));
    }

    #[test]
    fn put_past_terminal_is_refused() {
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        s.put("job-4", JobRecord::queued()).unwrap();
        s.put(
            "job-4",
            JobRecord {
                status: JobStatus::Failed,
                output: None,
                error: Some("worker died".to_string()),
            },
        )
        .unwrap();

        let attempt = s.put(
            "job-4",
            JobRecord {
                status: JobStatus::Completed,
                output: Some(json!({"text": "should not land"})),
                error: None,
            },
        );
        assert_eq!(attempt, Err(JobStoreError::AlreadyTerminal));

        // The failed record is exactly as it was.
        let rec = s.get("job-4").unwrap();
        assert_eq!(rec.status, JobStatus::Failed);
        assert_eq!(rec.error, Some("worker died".to_string()));
    }

    #[test]
    fn ttl_eviction_removes_finished_jobs_only() {
        let s = store(Duration::from_millis(20), DEFAULT_MAX_BYTES);
        s.put("finished", JobRecord::queued()).unwrap();
        s.put(
            "finished",
            JobRecord {
                status: JobStatus::Completed,
                output: Some(json!({"text": "done"})),
                error: None,
            },
        )
        .unwrap();
        s.put("still-running", JobRecord::queued()).unwrap();

        sleep(Duration::from_millis(60));

        // The finished job is past its TTL and is gone on the next touch.
        assert_eq!(s.get("finished"), None);
        // A job that never went terminal is never TTL-evicted, no matter its age.
        assert_eq!(s.get("still-running").unwrap().status, JobStatus::Queued);
    }

    #[test]
    fn sweep_reports_the_eviction_count() {
        let s = store(Duration::from_millis(10), DEFAULT_MAX_BYTES);
        for i in 0..3 {
            let id = format!("job-{i}");
            s.put(&id, JobRecord::queued()).unwrap();
            s.put(
                &id,
                JobRecord {
                    status: JobStatus::Completed,
                    output: Some(json!({"i": i})),
                    error: None,
                },
            )
            .unwrap();
        }
        sleep(Duration::from_millis(40));
        assert_eq!(s.sweep(), 3);
        assert_eq!(s.sweep(), 0);
    }

    #[test]
    fn byte_cap_refuses_a_record_that_would_exceed_it() {
        // A cap that admits the initial Queued placeholder (the only way to mint an id) but
        // is far too small for the real (large) completed output that follows it.
        let s = store(Duration::from_secs(60), 256);
        s.put("too-big", JobRecord::queued()).unwrap();
        let big_output = json!({"text": "x".repeat(1024)});
        let attempt = s.put(
            "too-big",
            JobRecord {
                status: JobStatus::Completed,
                output: Some(big_output),
                error: None,
            },
        );
        assert_eq!(attempt, Err(JobStoreError::CapacityExceeded));
        // The refused update did not land; the job is still Queued.
        assert_eq!(s.get("too-big").unwrap().status, JobStatus::Queued);
    }

    #[test]
    fn byte_cap_admits_small_records_up_to_the_cap() {
        let small = JobRecord {
            status: JobStatus::Completed,
            output: Some(json!({"t": "ok"})),
            error: None,
        };
        let size = record_size_bytes(&small);
        let s = store(Duration::from_secs(60), size);
        s.put("fits", JobRecord::queued()).unwrap();
        assert!(s.put("fits", small).is_ok());
    }

    #[test]
    fn put_on_an_unknown_id_is_refused_unless_queued() {
        // A brand-new id may only be admitted as Queued: that put is the one that mints it.
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        let attempt = s.put(
            "never-seen",
            JobRecord {
                status: JobStatus::InProgress,
                output: None,
                error: None,
            },
        );
        assert_eq!(attempt, Err(JobStoreError::NotFound));
        assert_eq!(s.get("never-seen"), None);
    }

    #[test]
    fn a_stale_write_after_cancel_and_collection_is_refused_not_resurrected() {
        // The race revuto's review of the first version of this file named: cancel, then
        // take() (collect) removes the record; a worker that has not yet noticed the cancel
        // then calls put() on the same id. Before the unknown-id check this put() succeeded
        // and brought the job back after it was already terminal and collected.
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        s.put("job-6", JobRecord::queued()).unwrap();
        s.cancel("job-6").unwrap();
        let taken = s.take("job-6").unwrap();
        assert_eq!(taken.status, JobStatus::Cancelled);
        assert_eq!(s.get("job-6"), None);

        // The worker's late write, non-terminal or terminal, is refused either way: the id
        // is unknown to the store and the write is not Queued.
        let late_in_progress = s.put(
            "job-6",
            JobRecord {
                status: JobStatus::InProgress,
                output: Some(json!({"text": "still going, did not get the memo"})),
                error: None,
            },
        );
        assert_eq!(late_in_progress, Err(JobStoreError::NotFound));

        let late_completed = s.put(
            "job-6",
            JobRecord {
                status: JobStatus::Completed,
                output: Some(json!({"text": "finished after all"})),
                error: None,
            },
        );
        assert_eq!(late_completed, Err(JobStoreError::NotFound));

        // Nothing was resurrected.
        assert_eq!(s.get("job-6"), None);
    }

    #[test]
    fn updating_a_non_terminal_record_replaces_its_byte_accounting() {
        let s = store(Duration::from_secs(60), DEFAULT_MAX_BYTES);
        s.put("job-5", JobRecord::queued()).unwrap();
        s.put(
            "job-5",
            JobRecord {
                status: JobStatus::InProgress,
                output: Some(json!({"text": "streaming in"})),
                error: None,
            },
        )
        .unwrap();
        let rec = s.get("job-5").unwrap();
        assert_eq!(rec.status, JobStatus::InProgress);
    }

    #[test]
    fn publication_capacity_failure_never_calls_settlement() {
        let s = store(Duration::from_secs(60), 256);
        s.put("job", JobRecord::queued()).unwrap();
        let mut calls = 0;
        let result = s.publish_terminal(
            "job",
            JobRecord {
                status: JobStatus::Completed,
                output: Some(json!({"text": "x".repeat(4096)})),
                error: None,
            },
            &mut || {
                calls += 1;
                Ok(())
            },
        );
        assert_eq!(result, Err(JobStoreError::CapacityExceeded));
        assert_eq!(calls, 0);
        assert_eq!(s.get("job").unwrap().status, JobStatus::Queued);
    }

    #[test]
    fn publication_is_hidden_and_protected_until_settlement_commits() {
        let s = store(Duration::from_secs(60), 1024);
        s.put("job", JobRecord::queued()).unwrap();
        s.reserve_output("job", 800).unwrap();
        // Non-terminal updates cannot release the working reservation.
        s.put(
            "job",
            JobRecord {
                status: JobStatus::InProgress,
                output: None,
                error: None,
            },
        )
        .unwrap();
        assert_eq!(
            s.put("peer", JobRecord::queued()),
            Err(JobStoreError::CapacityExceeded)
        );
        let record = JobRecord {
            status: JobStatus::Completed,
            output: Some(json!({"text":"done"})),
            error: None,
        };
        let mut calls = 0;
        s.publish_terminal("job", record.clone(), &mut || {
            calls += 1;
            assert_eq!(s.get("job").unwrap().status, JobStatus::InProgress);
            assert!(s.take("job").is_none());
            assert_eq!(s.cancel("job"), Err(JobStoreError::AlreadyTerminal));
            assert_eq!(
                s.put("job", JobRecord::queued()),
                Err(JobStoreError::AlreadyTerminal)
            );
            assert_eq!(s.sweep(), 0);
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(s.get("job"), Some(record));
        s.put("peer", JobRecord::queued()).unwrap();
    }

    #[test]
    fn failed_or_panicked_settlement_does_not_expose_output() {
        for panic in [false, true] {
            let s = store(Duration::from_secs(60), 1024);
            s.put("job", JobRecord::queued()).unwrap();
            let result = s.publish_terminal(
                "job",
                JobRecord {
                    status: JobStatus::Completed,
                    output: Some(json!({"text":"hidden"})),
                    error: None,
                },
                &mut || {
                    if panic {
                        panic!("injected settlement panic");
                    }
                    Err("injected settlement failure".into())
                },
            );
            assert_eq!(result, Err(JobStoreError::SettlementFailed));
            assert!(s.get("job").unwrap().output.is_none());
            s.put(
                "job",
                JobRecord {
                    status: JobStatus::Failed,
                    output: None,
                    error: Some("ledger failure".into()),
                },
            )
            .unwrap();
            assert_eq!(s.get("job").unwrap().status, JobStatus::Failed);
        }
    }

    #[test]
    fn env_defaults_are_used_when_unset() {
        // Best-effort: only asserts the fallback path when the vars are genuinely absent.
        // Does not mutate the process environment (parallel tests share it).
        if std::env::var(TTL_ENV).is_err() {
            assert_eq!(ttl_from_env(), Duration::from_secs(DEFAULT_TTL_SECS));
        }
        if std::env::var(MAX_BYTES_ENV).is_err() {
            assert_eq!(max_bytes_from_env(), DEFAULT_MAX_BYTES);
        }
    }
}
