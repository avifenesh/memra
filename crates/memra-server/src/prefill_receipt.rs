//! Shared terminal ownership while an extended prefill outlives its response-header hold.
use crate::metering::{Receipt, UsageCounts};
use std::sync::{Arc, Mutex, MutexGuard};

struct State {
    receipt: Box<dyn Receipt>,
    terminal: Option<Result<(), String>>,
}

#[derive(Clone)]
pub(crate) struct SharedReceipt(Arc<Mutex<State>>);

impl SharedReceipt {
    pub(crate) fn wrap(receipt: &mut Option<Box<dyn Receipt>>) -> Option<Self> {
        let inner = receipt.take()?;
        let shared = Self(Arc::new(Mutex::new(State {
            receipt: inner,
            terminal: None,
        })));
        *receipt = Some(Box::new(shared.clone()));
        Some(shared)
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn finish(&self, f: impl FnOnce(&mut dyn Receipt) -> Result<(), String>) -> Result<(), String> {
        let mut state = self.state();
        if let Some(result) = &state.terminal {
            return result.clone();
        }
        let result = f(state.receipt.as_mut());
        if result.is_err() {
            // A failed completion/settlement may leave the inner receipt unfinalized.
            // Preserve the billing layer's fail-closed fallback before its Drop can run.
            let _ = state.receipt.reject(500, "request_ledger_unavailable");
        }
        state.terminal = Some(result.clone());
        result
    }
}

impl Receipt for SharedReceipt {
    fn wants_capture(&self) -> bool {
        self.state().receipt.wants_capture()
    }
    fn arm_capture(&mut self, prompt: serde_json::Value) {
        let mut state = self.state();
        if state.terminal.is_none() {
            state.receipt.arm_capture(prompt);
        }
    }
    fn capture_completion_delta(&mut self, text: &str) {
        let mut state = self.state();
        if state.terminal.is_none() {
            state.receipt.capture_completion_delta(text);
        }
    }
    fn record_prompt_usage(&mut self, prompt: u64, cached: u64) -> Result<(), String> {
        let mut state = self.state();
        if let Some(result) = &state.terminal {
            return result.clone();
        }
        state.receipt.record_prompt_usage(prompt, cached)
    }
    fn record_completion_token(&mut self) -> Result<(), String> {
        let mut state = self.state();
        if let Some(result) = &state.terminal {
            return result.clone();
        }
        state.receipt.record_completion_token()
    }
    fn complete(&mut self, usage: UsageCounts, elapsed: f64) -> Result<(), String> {
        self.finish(|r| r.complete(usage, elapsed))
    }
    fn complete_deadline_partial(
        &mut self,
        usage: UsageCounts,
        elapsed: f64,
    ) -> Result<(), String> {
        self.finish(|r| r.complete_deadline_partial(usage, elapsed))
    }
    fn reject(&mut self, status: u16, code: &str) -> Result<(), String> {
        self.finish(|r| r.reject(status, code))
    }
    fn settle_unbilled(
        &mut self,
        outcome: &'static str,
        status: u16,
        code: &str,
    ) -> Result<(), String> {
        self.finish(|r| r.settle_unbilled(outcome, status, code))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct FailingCompletion {
        events: Arc<Mutex<Vec<&'static str>>>,
        finalized: bool,
    }
    impl Drop for FailingCompletion {
        fn drop(&mut self) {
            if !self.finalized {
                self.events.lock().unwrap().push("abandoned");
            }
        }
    }
    impl Receipt for FailingCompletion {
        fn arm_capture(&mut self, _: serde_json::Value) {}
        fn capture_completion_delta(&mut self, _: &str) {}
        fn record_prompt_usage(&mut self, _: u64, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn record_completion_token(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn complete(&mut self, _: UsageCounts, _: f64) -> Result<(), String> {
            Err("accounting failed".into())
        }
        fn complete_deadline_partial(
            &mut self,
            usage: UsageCounts,
            elapsed: f64,
        ) -> Result<(), String> {
            self.complete(usage, elapsed)
        }
        fn reject(&mut self, _: u16, _: &str) -> Result<(), String> {
            self.finalized = true;
            self.events.lock().unwrap().push("rejected");
            Ok(())
        }
        fn settle_unbilled(
            &mut self,
            _: &'static str,
            status: u16,
            code: &str,
        ) -> Result<(), String> {
            self.reject(status, code)
        }
    }
    #[test]
    fn completion_error_rejects_inner_receipt_before_drop() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut receipt: Option<Box<dyn Receipt>> = Some(Box::new(FailingCompletion {
            events: events.clone(),
            finalized: false,
        }));
        let shared = SharedReceipt::wrap(&mut receipt);
        assert!(
            receipt
                .as_mut()
                .unwrap()
                .complete(UsageCounts::default(), 0.0)
                .is_err()
        );
        drop(receipt);
        drop(shared);
        assert_eq!(*events.lock().unwrap(), vec!["rejected"]);
    }
}
