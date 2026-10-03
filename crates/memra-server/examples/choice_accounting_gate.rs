//! Diagnostic server: observe real HTTP accounting calls without a billing backend.
use memra_server::metering::*;
use serde_json::json;
use std::sync::Arc;
struct Counts {
    bound: u64,
}
struct Row {
    id: String,
    seen: UsageCounts,
    terminal: bool,
}
impl Metering for Counts {
    fn enforces_limits(&self) -> bool {
        true
    }
    fn is_limited(&self, _: &str) -> Result<bool, AdmitError> {
        Ok(true)
    }
    fn reserve(
        &self,
        tenant: &str,
        _: Option<&str>,
        model: &str,
        prompt: u64,
        output: u64,
    ) -> Result<Option<Permit>, AdmitError> {
        eprintln!(
            "CHOICE_RESERVE {}",
            json!({"tenant":tenant,"model":model,"prompt":prompt,"output":output,"limit":self.bound,"accepted":output<=self.bound})
        );
        if output > self.bound {
            Err(AdmitError::Insufficient)
        } else {
            Ok(None)
        }
    }
    fn open(&self, m: &RequestMeta<'_>, _: Option<Permit>) -> Box<dyn Receipt> {
        eprintln!(
            "CHOICE_OPEN {}",
            json!({"id":m.request_id,"model":m.model,"tenant":m.tenant,"max_tokens":m.max_tokens,"reserved_ctx":m.reserved_ctx,"stream":m.stream})
        );
        Box::new(Row {
            id: m.request_id.into(),
            seen: UsageCounts::default(),
            terminal: false,
        })
    }
    fn limits_health(&self) -> Option<LimitsHealth> {
        None
    }
}
impl Row {
    fn finish(
        &mut self,
        kind: &str,
        usage: UsageCounts,
        status: u16,
        code: &str,
    ) -> Result<(), String> {
        if self.terminal {
            return Err("duplicate terminal call".into());
        }
        eprintln!(
            "CHOICE_TERMINAL {}",
            json!({"id":self.id,"kind":kind,"status":status,"code":code,"prompt":usage.prompt_tokens,"cached":usage.cached_prompt_tokens,"output":usage.completion_tokens,"observed_prompt":self.seen.prompt_tokens,"observed_cached":self.seen.cached_prompt_tokens,"observed_output":self.seen.completion_tokens})
        );
        if kind == "complete"
            && (usage.prompt_tokens != self.seen.prompt_tokens
                || usage.cached_prompt_tokens != self.seen.cached_prompt_tokens
                || usage.completion_tokens != self.seen.completion_tokens)
        {
            return Err("completion differs from independent callback inputs".into());
        }
        self.terminal = true;
        Ok(())
    }
}
impl Receipt for Row {
    fn arm_capture(&mut self, _: serde_json::Value) {}
    fn capture_completion_delta(&mut self, _: &str) {}
    fn record_prompt_usage(&mut self, p: u64, c: u64) -> Result<(), String> {
        self.seen.prompt_tokens = p;
        self.seen.cached_prompt_tokens = c;
        eprintln!(
            "CHOICE_PROMPT {}",
            json!({"id":self.id,"prompt":p,"cached":c})
        );
        Ok(())
    }
    fn record_completion_token(&mut self) -> Result<(), String> {
        self.seen.completion_tokens += 1;
        eprintln!(
            "CHOICE_TOKEN {}",
            json!({"id":self.id,"output":self.seen.completion_tokens})
        );
        Ok(())
    }
    fn complete(&mut self, u: UsageCounts, _: f64) -> Result<(), String> {
        self.finish("complete", u, 200, "")
    }
    fn complete_deadline_partial(&mut self, u: UsageCounts, _: f64) -> Result<(), String> {
        self.finish("deadline_partial", u, 200, "deadline_exceeded")
    }
    fn reject(&mut self, s: u16, c: &str) -> Result<(), String> {
        self.finish("reject", self.seen, s, c)
    }
    fn settle_unbilled(&mut self, k: &'static str, s: u16, c: &str) -> Result<(), String> {
        self.finish(k, self.seen, s, c)
    }
}
impl Drop for Row {
    fn drop(&mut self) {
        eprintln!(
            "CHOICE_DROP {}",
            json!({"id":self.id,"terminal":self.terminal,"prompt":self.seen.prompt_tokens,"cached":self.seen.cached_prompt_tokens,"output":self.seen.completion_tokens})
        );
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bound = std::env::var("CHOICE_GATE_OUTPUT_BUDGET")?.parse()?;
    memra_server::serve_with(memra_server::ServerWiring::with_metering(Box::new(
        move |_| Ok(Some(Arc::new(Counts { bound }))),
    )))
    .await
}
