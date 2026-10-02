//! HTTP outcomes, distinct from worker attempts. Labels are canonical loaded models,
//! fixed backend/lane names and bounded HTTP codes. Unparsed/refused model names collapse
//! to one `unknown` row. A completed max_tokens response is a valid terminal response.
//! Stream truncation means the body ended without its protocol's terminal completion;
//! client drops are counted separately. No response bytes or request bodies are copied.
use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use hyper::body::{Body as HttpBody, Frame, SizeHint};
use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
struct Key {
    model: String,
    route: &'static str,
    lane: &'static str,
}
struct Counts {
    codes: [AtomicU64; 1000],
    refusals: AtomicU64,
    incomplete: AtomicU64,
    truncated: AtomicU64,
    cancelled: AtomicU64,
    preheader: AtomicU64,
    errors: AtomicU64,
}
impl Default for Counts {
    fn default() -> Self {
        Self {
            codes: std::array::from_fn(|_| AtomicU64::new(0)),
            refusals: AtomicU64::new(0),
            incomplete: AtomicU64::new(0),
            truncated: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            preheader: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }
}
fn registry() -> &'static Mutex<BTreeMap<Key, Arc<Counts>>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<Key, Arc<Counts>>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}
struct State {
    key: Key,
    status: Option<u16>,
    stream: bool,
    terminal: bool,
    failed: bool,
    recorded: bool,
}
#[derive(Clone)]
pub(crate) struct Observation(Arc<Mutex<State>>);
tokio::task_local! { static CURRENT: Observation; }
impl Observation {
    fn new(lane: &'static str) -> Self {
        Self(Arc::new(Mutex::new(State {
            key: Key {
                model: "unknown".into(),
                route: "unresolved",
                lane,
            },
            status: None,
            stream: false,
            terminal: false,
            failed: false,
            recorded: false,
        })))
    }
    pub(crate) fn terminal(&self) {
        let ready = {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            state.terminal = true;
            state.status.is_some() && state.stream
        };
        // Publish before yielding the terminal frame: a client may close immediately
        // after receiving it, without asking the body for another frame.
        if ready {
            self.finish(false);
        }
    }
    pub(crate) fn failed(&self) {
        let ready = {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if state.terminal {
                return;
            }
            state.failed = true;
            state.status.is_some() && state.stream
        };
        if ready {
            self.finish(false);
        }
    }
    fn finish(&self, dropped: bool) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.recorded {
            return;
        }
        state.recorded = true;
        let cancelled = dropped && !std::thread::panicking() && !state.terminal && !state.failed;
        let code = if cancelled {
            499
        } else {
            state.status.unwrap_or(500)
        };
        let row = registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(state.key.clone())
            .or_default()
            .clone();
        row.codes[code as usize].fetch_add(1, Ordering::Relaxed);
        if !cancelled && (code >= 400 || state.failed || (state.stream && !state.terminal)) {
            row.errors.fetch_add(1, Ordering::Relaxed);
        }
        if cancelled {
            row.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        if !state.stream && code >= 400 && !cancelled {
            row.refusals.fetch_add(1, Ordering::Relaxed);
        }
        if state.status == Some(408) && !state.stream {
            row.preheader.fetch_add(1, Ordering::Relaxed);
        }
        if state.stream && !state.terminal {
            row.incomplete.fetch_add(1, Ordering::Relaxed);
            if !cancelled {
                row.truncated.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
pub(crate) fn current() -> Option<Observation> {
    CURRENT.try_with(Clone::clone).ok()
}
/// Call only with the result of canonical_model_id, never an unchecked request string.
pub(crate) fn bind_model(model: Option<String>) {
    if let Some(model) = model
        && let Some(observation) = current()
    {
        let mut state = observation.0.lock().unwrap_or_else(|e| e.into_inner());
        state.key.route = if crate::route_telemetry::lookup(&model).is_some() {
            crate::request_metrics::Backend::Dsv4
        } else {
            crate::request_metrics::Backend::Hybrid
        }
        .as_str();
        state.key.model = model;
    }
}

/// Replace the bounded pre-authentication lane hint with the scheduler's resolved lane.
pub(crate) fn bind_lane(lane: crate::lanes::Lane) {
    if let Some(observation) = current() {
        observation
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .key
            .lane = lane.as_str();
    }
}

struct Guard(Observation);
impl Drop for Guard {
    fn drop(&mut self) {
        self.0.finish(true);
    }
}
struct ObservedBody {
    inner: Pin<Box<Body>>,
    guard: Guard,
}
impl HttpBody for ObservedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let next = self.inner.as_mut().poll_frame(cx);
        if matches!(&next, Poll::Ready(None) | Poll::Ready(Some(Err(_)))) {
            self.guard.0.finish(false);
        }
        next
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

pub(crate) async fn observe(req: Request, next: Next) -> Response {
    if req.method() != axum::http::Method::POST
        || !matches!(
            req.uri().path(),
            "/v1/completions" | "/v1/chat/completions" | "/v1/responses" | "/v1/messages"
        )
    {
        return next.run(req).await;
    }
    let lane = req
        .headers()
        .get("x-lane")
        .and_then(|v| v.to_str().ok())
        .and_then(crate::lanes::Lane::parse)
        .unwrap_or(crate::lanes::Lane::Interactive);
    let observation = Observation::new(lane.as_str());
    let guard = Guard(observation.clone());
    let response = CURRENT.scope(observation.clone(), next.run(req)).await;
    let stream = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    {
        let mut state = observation.0.lock().unwrap_or_else(|e| e.into_inner());
        state.status = Some(response.status().as_u16());
        state.stream = stream;
    }
    if !stream {
        observation.finish(false);
        return response;
    }
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(ObservedBody {
            inner: Box::pin(body),
            guard,
        }),
    )
}

pub(crate) fn render() -> String {
    let registry: Vec<_> = registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(key, row)| (key.clone(), row.clone()))
        .collect();
    let labels = |key: &Key| {
        format!(
            "model=\"{}\",route=\"{}\",lane=\"{}\"",
            crate::prometheus_label(&key.model),
            key.route,
            key.lane
        )
    };
    let mut out = String::new();
    crate::prometheus_header(
        &mut out,
        "memra_requests_total",
        "counter",
        "Logical generation HTTP outcomes by response code; 499 denotes client cancellation before a terminal response.",
    );
    for (key, counts) in registry.iter() {
        for (code, value) in counts.codes.iter().enumerate() {
            let value = value.load(Ordering::Relaxed);
            if value > 0 {
                out.push_str(&format!(
                    "memra_requests_total{{{},code=\"{code}\"}} {value}\n",
                    labels(key)
                ));
            }
        }
    }
    for (index, (name, help)) in [
        ("memra_requests_refused_total", "HTTP errors returned before a generation stream was opened, excluding client cancellations."),
        ("memra_streams_incomplete_total", "Generation streams without a terminal completion, including client cancellations."),
        ("memra_streams_truncated_total", "Generation streams that ended without a terminal completion, excluding client cancellations."),
        ("memra_requests_cancelled_total", "Client cancellations before a terminal generation response."),
        ("memra_pre_header_deadline_total", "Generation requests answered with HTTP 408 before streaming headers."),
        ("memra_response_errors_total", "HTTP errors and failed generation responses, including errors carried inside HTTP 200 bodies."),
    ].into_iter().enumerate() {
        crate::prometheus_header(&mut out, name, "counter", help);
        for (key, counts) in registry.iter() {
            let value = [&counts.refusals, &counts.incomplete, &counts.truncated, &counts.cancelled, &counts.preheader, &counts.errors][index].load(Ordering::Relaxed);
            out.push_str(&format!("{name}{{{}}} {value}\n", labels(key)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(model: &str, code: u16, stream: bool) -> Observation {
        let o = Observation::new("interactive");
        {
            let mut s = o.0.lock().unwrap();
            s.key.model = model.into();
            s.status = Some(code);
            s.stream = stream;
        }
        o
    }
    fn row(o: &Observation) -> Arc<Counts> {
        registry()
            .lock()
            .unwrap()
            .get(&o.0.lock().unwrap().key)
            .unwrap()
            .clone()
    }
    #[test]
    fn statuses_and_preheader_deadlines_are_separate_from_stream_failures() {
        for code in [408, 413, 429, 503] {
            let o = fixture(&format!("http-code-{code}"), code, false);
            o.finish(false);
            o.finish(true);
            let row = row(&o);
            assert_eq!(row.codes[code as usize].load(Ordering::Relaxed), 1);
            assert_eq!(row.refusals.load(Ordering::Relaxed), 1);
            assert_eq!(
                row.preheader.load(Ordering::Relaxed),
                u64::from(code == 408)
            );
            assert_eq!(row.incomplete.load(Ordering::Relaxed), 0);
        }
    }
    #[tokio::test]
    async fn registered_backend_binding_is_a_cpu_http_contract() {
        let model = "http-registered-backend";
        crate::route_telemetry::register(model, 1);
        let observation = fixture(model, 503, false);
        CURRENT
            .scope(observation.clone(), async {
                bind_model(Some(model.into()));
            })
            .await;
        observation.finish(false);
        assert_eq!(observation.0.lock().unwrap().key.route, "dsv4");
        assert_eq!(row(&observation).codes[503].load(Ordering::Relaxed), 1);
    }

    #[test]
    fn every_status_accepted_by_the_http_type_fits_the_counter_table() {
        let code = axum::http::StatusCode::from_u16(999).unwrap();
        let observation = fixture("http-extension-code", code.as_u16(), false);
        observation.finish(false);
        assert_eq!(row(&observation).codes[999].load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unwinding_a_handler_counts_a_server_error_instead_of_a_client_cancel() {
        let observation = fixture("http-handler-panic", 500, false);
        observation.0.lock().unwrap().status = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = Guard(observation.clone());
            panic!("controlled handler failure");
        }));
        assert!(result.is_err());
        let row = row(&observation);
        assert_eq!(row.codes[500].load(Ordering::Relaxed), 1);
        assert_eq!(row.errors.load(Ordering::Relaxed), 1);
        assert_eq!(row.cancelled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn completed_truncated_and_cancelled_streams_have_distinct_outcomes() {
        let clean = fixture("http-stream-clean", 200, true);
        clean.terminal();
        clean.finish(true);
        let truncated = fixture("http-stream-truncated", 200, true);
        truncated.finish(false);
        let cancelled = fixture("http-stream-cancelled", 200, true);
        cancelled.finish(true);
        assert_eq!(row(&clean).codes[200].load(Ordering::Relaxed), 1);
        assert_eq!(row(&clean).incomplete.load(Ordering::Relaxed), 0);
        assert_eq!(row(&truncated).codes[200].load(Ordering::Relaxed), 1);
        assert_eq!(row(&truncated).truncated.load(Ordering::Relaxed), 1);
        assert_eq!(row(&cancelled).codes[499].load(Ordering::Relaxed), 1);
        assert_eq!(row(&cancelled).incomplete.load(Ordering::Relaxed), 1);
        assert_eq!(row(&cancelled).truncated.load(Ordering::Relaxed), 0);
    }
}
