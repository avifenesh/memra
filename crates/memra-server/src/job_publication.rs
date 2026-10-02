//! Output reservations and settlement guarded by the configured JobStore.
use crate::metering::{JobRecord, JobStore, JobStoreError, Receipt, UsageCounts};
use std::sync::{Arc, Mutex};

enum Pending {
    Complete(UsageCounts, f64),
    Partial(UsageCounts, f64),
}

struct Settlement {
    receipt: Option<Box<dyn Receipt>>,
    pending: Option<Pending>,
    settled: bool,
}

#[derive(Clone)]
pub(crate) struct Publication {
    pub(crate) store: Arc<dyn JobStore>,
    pub(crate) key: String,
    settlement: Arc<Mutex<Settlement>>,
    buffered: Arc<Mutex<usize>>,
}

impl Publication {
    pub(crate) fn new(
        store: Arc<dyn JobStore>,
        key: String,
        receipt: Option<Box<dyn Receipt>>,
    ) -> (Self, Option<Box<dyn Receipt>>) {
        let settlement = Arc::new(Mutex::new(Settlement {
            receipt,
            pending: None,
            settled: false,
        }));
        let publication = Self {
            store,
            key,
            settlement,
            buffered: Arc::new(Mutex::new(0)),
        };
        let proxy: Box<dyn Receipt> = Box::new(DeferredReceipt(publication.clone()));
        (publication, Some(proxy))
    }

    /// Charge a conservative working budget before retaining a delta. The factor
    /// covers parser strings/calls, duplicated reasoning fields, escaped JSON,
    /// and simultaneous encoded/decoded envelopes. This is proportional accounting,
    /// not an exact RSS claim. The configured aggregate store cap remains the limit.
    pub(crate) fn reserve_delta(&self, text: &str, tokens: usize) -> Result<(), JobStoreError> {
        let mut bytes = self.buffered.lock().unwrap();
        let next = bytes
            .checked_add(
                text.len()
                    .checked_mul(64)
                    .ok_or(JobStoreError::CapacityExceeded)?,
            )
            .and_then(|v| v.checked_add(tokens.checked_mul(128)?))
            .ok_or(JobStoreError::CapacityExceeded)?;
        self.store.reserve_output(
            &self.key,
            next.checked_add(256)
                .ok_or(JobStoreError::CapacityExceeded)?,
        )?;
        *bytes = next;
        Ok(())
    }

    pub(crate) fn reserve_snapshot(&self, tokens: usize) -> Result<(), JobStoreError> {
        self.reserve_delta("", tokens)
    }

    pub(crate) fn initial_reservation(&self) -> Result<(), JobStoreError> {
        self.store.reserve_output(&self.key, 256)
    }

    pub(crate) fn body_limit(&self) -> usize {
        self.buffered.lock().unwrap().saturating_add(256)
    }

    /// Count without allocating, reserve before encoding, then use a capped writer.
    /// A body can never allocate past the accepted encoding reservation.
    pub(crate) fn encode<T: serde::Serialize>(&self, value: &T) -> Result<Vec<u8>, JobStoreError> {
        struct Counter(usize);
        impl std::io::Write for Counter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_add(bytes.len())
                    .ok_or_else(|| std::io::Error::other("size overflow"))?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Counter(0);
        serde_json::to_writer(&mut count, value).map_err(|_| JobStoreError::CapacityExceeded)?;
        let working = count
            .0
            .checked_mul(3)
            .and_then(|v| v.checked_add(256))
            .ok_or(JobStoreError::CapacityExceeded)?;
        self.store
            .reserve_output(&self.key, self.body_limit().max(working))?;
        struct Capped {
            bytes: Vec<u8>,
            limit: usize,
        }
        impl std::io::Write for Capped {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
                    return Err(std::io::Error::other("encoding overflow"));
                }
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut writer = Capped {
            bytes: Vec::with_capacity(count.0),
            limit: count.0,
        };
        serde_json::to_writer(&mut writer, value).map_err(|_| JobStoreError::CapacityExceeded)?;
        let mut buffered = self.buffered.lock().unwrap();
        *buffered = (*buffered).max(count.0);
        Ok(writer.bytes)
    }

    pub(crate) fn fail_storage(&self, error: JobStoreError) {
        let code = if error == JobStoreError::CapacityExceeded {
            "background_job_store_capacity_exceeded"
        } else {
            "background_job_store_unavailable"
        };
        let mut state = self.settlement.lock().unwrap_or_else(|e| e.into_inner());
        state.pending = None;
        if !state.settled {
            if let Some(receipt) = state.receipt.as_mut() {
                if receipt
                    .settle_unbilled("background_storage_failed", 503, code)
                    .is_err()
                {
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                }
            }
            state.settled = true;
        }
    }

    pub(crate) fn publish(&self, record: JobRecord) -> Result<(), JobStoreError> {
        if record.output.is_none() {
            self.fail_storage(JobStoreError::CapacityExceeded);
            return self.store.put(&self.key, record);
        }
        if let Some(output) = &record.output {
            if let Err(error) = self.encode(output) {
                self.fail_storage(error);
                return Err(error);
            }
        }
        let result = self.store.publish_terminal(&self.key, record, &mut || {
            let mut state = self.settlement.lock().unwrap_or_else(|e| e.into_inner());
            if state.settled {
                return Ok(());
            }
            let pending = state.pending.take();
            if let Some(receipt) = state.receipt.as_mut() {
                match pending {
                    Some(Pending::Complete(usage, elapsed)) => receipt.complete(usage, elapsed)?,
                    Some(Pending::Partial(usage, elapsed)) => {
                        receipt.complete_deadline_partial(usage, elapsed)?
                    }
                    None => {}
                }
            }
            state.settled = true;
            Ok(())
        });
        if let Err(error) = &result {
            if *error == JobStoreError::SettlementFailed {
                let mut state = self.settlement.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(receipt) = state.receipt.as_mut() {
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                }
                state.settled = true;
            } else {
                self.fail_storage(match error {
                    JobStoreError::CapacityExceeded => JobStoreError::CapacityExceeded,
                    _ => JobStoreError::PublicationUnsupported,
                });
            }
        }
        result
    }
}

/// Used by the existing Responses finalizer, so both dialect implementations
/// reserve and publish through the same configured backend and receipt state.
pub(crate) struct PublishingStore(pub(crate) Publication);
impl JobStore for PublishingStore {
    fn reserve_output(&self, id: &str, bytes: usize) -> Result<(), JobStoreError> {
        self.0.store.reserve_output(id, bytes)
    }
    fn publish_terminal(
        &self,
        id: &str,
        record: JobRecord,
        settle: &mut dyn FnMut() -> Result<(), String>,
    ) -> Result<(), JobStoreError> {
        self.0.store.publish_terminal(id, record, settle)
    }
    fn put(&self, id: &str, record: JobRecord) -> Result<(), JobStoreError> {
        if record.status.is_terminal() {
            self.0.publish(record)
        } else {
            self.0.store.put(id, record)
        }
    }
    fn get(&self, id: &str) -> Option<JobRecord> {
        self.0.store.get(id)
    }
    fn take(&self, id: &str) -> Option<JobRecord> {
        self.0.store.take(id)
    }
    fn cancel(&self, id: &str) -> Result<(), JobStoreError> {
        self.0.store.cancel(id)
    }
    fn sweep(&self) -> usize {
        self.0.store.sweep()
    }
}

impl Publication {
    /// The configured backend, including wrappers, remains the authority for
    /// both reservation and publication. This check is performed before buffering.
    pub(crate) fn storage_response(&self, error: JobStoreError) -> axum::response::Response {
        let code = if error == JobStoreError::CapacityExceeded {
            "background_job_store_capacity_exceeded"
        } else {
            "background_job_store_unavailable"
        };
        self.fail_storage(error);
        crate::error_response_coded(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "background output could not be buffered by the configured store",
            "server_error",
            None,
            Some(code),
        )
    }
}

struct DeferredReceipt(Publication);
impl Receipt for DeferredReceipt {
    fn wants_capture(&self) -> bool {
        self.0
            .settlement
            .lock()
            .unwrap()
            .receipt
            .as_ref()
            .is_some_and(|r| r.wants_capture())
    }
    fn arm_capture(&mut self, prompt: serde_json::Value) {
        if let Some(r) = self
            .0
            .settlement
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .receipt
            .as_mut()
        {
            r.arm_capture(prompt);
        }
    }
    fn capture_completion_delta(&mut self, text: &str) {
        if let Some(r) = self
            .0
            .settlement
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .receipt
            .as_mut()
        {
            r.capture_completion_delta(text);
        }
    }
    fn record_prompt_usage(&mut self, prompt: u64, cached: u64) -> Result<(), String> {
        match self
            .0
            .settlement
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .receipt
            .as_mut()
        {
            Some(r) => r.record_prompt_usage(prompt, cached),
            None => Ok(()),
        }
    }
    fn record_completion_token(&mut self) -> Result<(), String> {
        match self
            .0
            .settlement
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .receipt
            .as_mut()
        {
            Some(r) => r.record_completion_token(),
            None => Ok(()),
        }
    }
    fn complete(&mut self, usage: UsageCounts, elapsed: f64) -> Result<(), String> {
        self.0
            .settlement
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pending = Some(Pending::Complete(usage, elapsed));
        Ok(())
    }
    fn complete_deadline_partial(
        &mut self,
        usage: UsageCounts,
        elapsed: f64,
    ) -> Result<(), String> {
        self.0
            .settlement
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pending = Some(Pending::Partial(usage, elapsed));
        Ok(())
    }
    fn reject(&mut self, status: u16, code: &str) -> Result<(), String> {
        let mut state = self.0.settlement.lock().unwrap_or_else(|e| e.into_inner());
        if state.settled {
            return Ok(());
        }
        state.pending = None;
        let result = match state.receipt.as_mut() {
            Some(r) => r.reject(status, code),
            None => Ok(()),
        };
        if result.is_ok() {
            state.settled = true;
        }
        result
    }
    fn settle_unbilled(
        &mut self,
        outcome: &'static str,
        status: u16,
        code: &str,
    ) -> Result<(), String> {
        let mut state = self.0.settlement.lock().unwrap_or_else(|e| e.into_inner());
        if state.settled {
            return Ok(());
        }
        state.pending = None;
        let result = match state.receipt.as_mut() {
            Some(r) => r.settle_unbilled(outcome, status, code),
            None => Ok(()),
        };
        if result.is_ok() {
            state.settled = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_store::InMemoryJobStore;
    use std::time::Duration;

    #[test]
    fn incremental_output_cannot_retain_an_over_cap_delta_or_snapshot() {
        let store = Arc::new(InMemoryJobStore::new(Duration::from_secs(60), 1024));
        store.put("job", JobRecord::queued()).unwrap();
        let (publication, _) = Publication::new(store.clone(), "job".into(), None);
        publication.reserve_delta("hello", 1).unwrap();
        assert_eq!(publication.body_limit(), 704);
        assert_eq!(
            publication.reserve_delta(&"x".repeat(1000), 1),
            Err(JobStoreError::CapacityExceeded)
        );
        assert_eq!(
            publication.body_limit(),
            704,
            "refused bytes were never retained"
        );
        assert_eq!(
            publication.reserve_snapshot(1000),
            Err(JobStoreError::CapacityExceeded)
        );
        assert_eq!(publication.body_limit(), 704);
        assert!(store.get("job").unwrap().output.is_none());
    }

    #[test]
    fn encoding_reserves_escaped_size_before_allocating_the_body() {
        let store = Arc::new(InMemoryJobStore::new(Duration::from_secs(60), 1024));
        store.put("job", JobRecord::queued()).unwrap();
        let (publication, _) = Publication::new(store, "job".into(), None);
        let encoded = publication
            .encode(&serde_json::json!({"text":"hello"}))
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&encoded).unwrap()["text"],
            "hello"
        );
        assert_eq!(
            publication.encode(&"\u{0000}".repeat(100)),
            Err(JobStoreError::CapacityExceeded)
        );
    }
}
