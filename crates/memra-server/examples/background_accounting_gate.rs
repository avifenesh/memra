//! Native endpoint gate: the real server with observable, money-free accounting callbacks.
//! Run with the normal MEMRA_MODELS/auth/bind settings. Receipt rows go to stdout.
use memra_server::metering::*;
use serde_json::json;
use std::sync::Arc;

struct Counts;

/// Test-only store observer. Publish after the successful state write so a collector
/// can wait on an event instead of polling a background job for completion.
struct ObservedStore(memra_server::job_store::InMemoryJobStore);
impl JobStore for ObservedStore {
    fn put(&self, key: &str, record: JobRecord) -> Result<(), JobStoreError> {
        let terminal = record.status.is_terminal();
        let status = format!("{:?}", record.status);
        let id = record
            .output
            .as_ref()
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        self.0.put(key, record)?;
        if terminal {
            println!("GATE_STORED {}", json!({"id":id,"status":status}));
        }
        Ok(())
    }
    fn get(&self, key: &str) -> Option<JobRecord> {
        self.0.get(key)
    }
    fn take(&self, key: &str) -> Option<JobRecord> {
        self.0.take(key)
    }
    fn cancel(&self, key: &str) -> Result<(), JobStoreError> {
        self.0.cancel(key)
    }
    fn sweep(&self) -> usize {
        self.0.sweep()
    }
}
struct Row {
    id: String,
    tenant: String,
    usage: UsageCounts,
    terminal: bool,
}
impl Row {
    fn finish(
        &mut self,
        outcome: &str,
        usage: UsageCounts,
        status: u16,
        code: &str,
    ) -> Result<(), String> {
        if self.terminal {
            return Err("duplicate terminal callback".into());
        }
        println!(
            "GATE_RECEIPT {}",
            json!({"id":self.id,"tenant":self.tenant,"outcome":outcome,"status":status,"code":code,"prompt_tokens":usage.prompt_tokens,"cached_tokens":usage.cached_prompt_tokens,"completion_tokens":usage.completion_tokens})
        );
        self.terminal = true;
        Ok(())
    }
}
impl Drop for Row {
    fn drop(&mut self) {
        if !self.terminal {
            println!("GATE_UNSETTLED {}", self.id);
        }
    }
}
impl Metering for Counts {
    fn enforces_limits(&self) -> bool {
        false
    }
    fn is_limited(&self, _: &str) -> Result<bool, AdmitError> {
        Ok(false)
    }
    fn reserve(
        &self,
        _: &str,
        _: Option<&str>,
        _: &str,
        _: u64,
        _: u64,
    ) -> Result<Option<Permit>, AdmitError> {
        Ok(None)
    }
    fn open(&self, meta: &RequestMeta<'_>, _: Option<Permit>) -> Box<dyn Receipt> {
        Box::new(Row {
            id: meta.request_id.into(),
            tenant: meta.tenant.into(),
            usage: UsageCounts::default(),
            terminal: false,
        })
    }
    fn limits_health(&self) -> Option<LimitsHealth> {
        None
    }
}
impl Receipt for Row {
    fn arm_capture(&mut self, _: serde_json::Value) {}
    fn capture_completion_delta(&mut self, _: &str) {}
    fn record_prompt_usage(&mut self, prompt: u64, cached: u64) -> Result<(), String> {
        self.usage.prompt_tokens = prompt;
        self.usage.cached_prompt_tokens = cached;
        Ok(())
    }
    fn record_completion_token(&mut self) -> Result<(), String> {
        self.usage.completion_tokens += 1;
        if self.usage.completion_tokens == 32 {
            println!(
                "GATE_PROGRESS {}",
                json!({"id":self.id,"completion_tokens":32})
            );
        }
        Ok(())
    }
    fn complete(&mut self, usage: UsageCounts, _: f64) -> Result<(), String> {
        self.finish("complete", usage, 200, "")
    }
    fn complete_deadline_partial(&mut self, usage: UsageCounts, _: f64) -> Result<(), String> {
        self.finish("cancel_partial", usage, 200, "cancelled")
    }
    fn reject(&mut self, status: u16, code: &str) -> Result<(), String> {
        self.finish("reject", self.usage, status, code)
    }
    fn settle_unbilled(
        &mut self,
        outcome: &'static str,
        status: u16,
        code: &str,
    ) -> Result<(), String> {
        self.finish(outcome, self.usage, status, code)
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(ObservedStore(
        memra_server::job_store::InMemoryJobStore::from_env(),
    ));
    let wiring =
        memra_server::ServerWiring::with_metering(Box::new(|_| Ok(Some(Arc::new(Counts)))))
            .with_job_store(store);
    memra_server::serve_with(wiring).await
}
