//! Cumulative generation timing for the hybrid worker, with fixed backend/lane labels.
//! Labels come from loaded models, two backend names and the three fixed scheduler lanes.
//! TTFT and successful E2E start at worker submission, including queue wait. Token gaps
//! measure successful worker token-event sends, not compute rounds or network flushes.

use crate::histogram::{Histogram, HistogramSnapshot};
use crate::lanes::Lane;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Backend {
    Hybrid,
    Dsv4,
}
impl Backend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Dsv4 => "dsv4",
        }
    }
}

const TOKEN_BUCKETS: [u64; 15] = [
    0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
];
#[derive(Default)]
struct Tokens {
    buckets: [AtomicU64; 16],
    sum: AtomicU64,
}
impl Tokens {
    fn record(&self, n: usize) {
        let n = n as u64;
        let bucket = TOKEN_BUCKETS
            .iter()
            .position(|&edge| n <= edge)
            .unwrap_or(15);
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(n, Ordering::Relaxed);
    }
    fn snapshot(&self) -> HistogramSnapshot {
        let mut count = 0;
        let cumulative = self
            .buckets
            .iter()
            .enumerate()
            .map(|(i, value)| {
                count += value.load(Ordering::Relaxed);
                (
                    TOKEN_BUCKETS.get(i).map_or(f64::INFINITY, |&n| n as f64),
                    count,
                )
            })
            .collect();
        HistogramSnapshot {
            cumulative,
            sum_seconds: self.sum.load(Ordering::Relaxed) as f64,
            count,
        }
    }
}

#[derive(Default)]
struct Series {
    queue: Histogram,
    ttft: Histogram,
    e2e: Histogram,
    gap: Histogram,
    cached: AtomicU64,
    emitted: AtomicU64,
    prefill: Tokens,
    completion: Tokens,
}
type Key = (String, Backend, usize);
fn registry() -> &'static RwLock<BTreeMap<Key, Arc<Series>>> {
    static REGISTRY: OnceLock<RwLock<BTreeMap<Key, Arc<Series>>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}
fn series(model: &str, backend: Backend, lane: Lane) -> Arc<Series> {
    let key = (model.to_owned(), backend, lane.idx());
    if let Some(row) = registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
    {
        return row.clone();
    }
    registry()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .entry(key)
        .or_default()
        .clone()
}

/// One admitted generation attempt. Internal worker retries are distinct attempts; the
/// HTTP outcome counters separately count the logical request. Capture-only work is excluded.
#[derive(Default)]
pub(crate) struct Clock {
    row: Option<Arc<Series>>,
    queued: Option<Instant>,
    last: Option<Instant>,
}
impl Clock {
    pub(crate) fn admitted(
        model: &str,
        backend: Backend,
        lane: Lane,
        queued: Instant,
        admitted: Instant,
    ) -> Self {
        let row = series(model, backend, lane);
        row.queue
            .record(admitted.saturating_duration_since(queued).as_secs_f64());
        Self {
            row: Some(row),
            queued: Some(queued),
            last: None,
        }
    }
    pub(crate) fn prompt(&self, n: usize, cached: usize) {
        if let Some(row) = &self.row {
            row.prefill.record(n);
            row.cached.fetch_add(cached as u64, Ordering::Relaxed);
        }
    }
    pub(crate) fn cache_credit(&self, tokens: usize) {
        if let Some(row) = &self.row {
            row.cached.fetch_add(tokens as u64, Ordering::Relaxed);
        }
    }
    pub(crate) fn sent(&mut self) {
        self.sent_at(Instant::now());
    }
    fn sent_at(&mut self, now: Instant) {
        if let Some(row) = &self.row {
            row.emitted.fetch_add(1, Ordering::Relaxed);
            if let Some(last) = self.last.replace(now) {
                row.gap
                    .record(now.saturating_duration_since(last).as_secs_f64());
            } else if let Some(queued) = self.queued {
                row.ttft
                    .record(now.saturating_duration_since(queued).as_secs_f64());
            }
        }
    }
    pub(crate) fn finished(&mut self, tokens: usize) {
        self.finished_at(tokens, Instant::now());
    }
    fn finished_at(&mut self, tokens: usize, now: Instant) {
        if let Some(row) = self.row.take() {
            row.completion.record(tokens);
            if let Some(queued) = self.queued {
                row.e2e
                    .record(now.saturating_duration_since(queued).as_secs_f64());
            }
        }
    }
}

pub(crate) fn render() -> String {
    // Keep formatting and histogram loads outside the registry lock so scrapes do
    // not hold up admission. Existing sessions record through their own Arc.
    let registry: Vec<_> = registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(key, row)| (key.clone(), row.clone()))
        .collect();
    let rows: Vec<_> = registry
        .iter()
        .map(|((model, backend, lane), s)| {
            let labels = format!(
                "model=\"{}\",route=\"{}\",lane=\"{}\"",
                crate::prometheus_label(model),
                backend.as_str(),
                ["interactive", "judge", "harvest"][*lane]
            );
            (
                labels,
                [
                    s.queue.snapshot(),
                    s.ttft.snapshot(),
                    s.e2e.snapshot(),
                    s.gap.snapshot(),
                    s.prefill.snapshot(),
                    s.completion.snapshot(),
                ],
            )
        })
        .collect();
    let mut out = String::new();
    for (i, (name, help)) in [
        ("memra_queue_wait_seconds", "Worker submission to admission, per generation attempt, cumulative and unsampled."),
        ("memra_ttft_seconds", "Worker submission to first accepted token event, including queue wait, cumulative and unsampled."),
        ("memra_e2e_seconds", "Worker submission to successful generation finish, including queue wait, cumulative and unsampled."),
        ("memra_tpot_seconds", "Intervals between accepted worker token events within one generation attempt, cumulative and unsampled."),
        ("memra_prefill_tokens", "Prompt token count including cache reads, once per admitted generation attempt."),
        ("memra_completion_tokens", "Generated token count of successful generation attempts, including length-limited completions."),
    ].into_iter().enumerate() {
        crate::prometheus_header(&mut out, name, "histogram", help);
        for (labels, snapshots) in &rows { out.push_str(&snapshots[i].render_prometheus(name, labels)); }
    }
    crate::prometheus_header(
        &mut out,
        "memra_cached_tokens_total",
        "counter",
        "Prompt tokens restored from cache for admitted hybrid generation attempts.",
    );
    for ((model, backend, lane), row) in registry.iter() {
        out.push_str(&format!(
            "memra_cached_tokens_total{{model=\"{}\",route=\"{}\",lane=\"{}\"}} {}\n",
            crate::prometheus_label(model),
            backend.as_str(),
            ["interactive", "judge", "harvest"][*lane],
            row.cached.load(Ordering::Relaxed)
        ));
    }
    crate::prometheus_header(
        &mut out,
        "memra_emitted_token_events_total",
        "counter",
        "Accepted worker token-event sends, including partial and cancelled requests.",
    );
    for ((model, backend, lane), row) in registry.iter() {
        out.push_str(&format!(
            "memra_emitted_token_events_total{{model=\"{}\",route=\"{}\",lane=\"{}\"}} {}\n",
            crate::prometheus_label(model),
            backend.as_str(),
            ["interactive", "judge", "harvest"][*lane],
            row.emitted.load(Ordering::Relaxed)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn route_and_lane_timings_measure_emission_and_include_queue() {
        let now = Instant::now();
        let mut clock = Clock::admitted(
            "request-clock-a",
            Backend::Dsv4,
            Lane::Judge,
            now,
            now + Duration::from_secs(2),
        );
        clock.prompt(1000, 512);
        clock.cache_credit(256);
        clock.sent_at(now + Duration::from_secs(3));
        clock.sent_at(now + Duration::from_secs(4));
        clock.sent_at(now + Duration::from_secs(6));
        clock.finished_at(3, now + Duration::from_secs(7));
        clock.finished_at(3, now + Duration::from_secs(8));
        let row = series("request-clock-a", Backend::Dsv4, Lane::Judge);
        assert_eq!(row.queue.snapshot().sum_seconds, 2.0);
        assert_eq!(row.ttft.snapshot().sum_seconds, 3.0);
        assert_eq!(row.e2e.snapshot().sum_seconds, 7.0);
        assert_eq!(row.e2e.snapshot().count, 1);
        assert_eq!(row.gap.snapshot().count, 2);
        assert_eq!(row.emitted.load(Ordering::Relaxed), 3);
        assert_eq!(row.gap.snapshot().sum_seconds, 3.0);
        assert_eq!(row.prefill.snapshot().sum_seconds, 1000.0);
        assert_eq!(row.cached.load(Ordering::Relaxed), 768);
        assert_eq!(row.completion.snapshot().sum_seconds, 3.0);
        assert_eq!(
            series("request-clock-a", Backend::Hybrid, Lane::Judge)
                .queue
                .snapshot()
                .count,
            0
        );
        assert_eq!(
            series("request-clock-a", Backend::Dsv4, Lane::Interactive)
                .queue
                .snapshot()
                .count,
            0
        );
    }
    #[test]
    fn abandoned_attempt_excludes_success_histograms() {
        let now = Instant::now();
        let mut clock = Clock::admitted(
            "request-clock-abort",
            Backend::Hybrid,
            Lane::Interactive,
            now,
            now,
        );
        clock.prompt(12, 0);
        clock.sent_at(now);
        drop(clock);
        let row = series("request-clock-abort", Backend::Hybrid, Lane::Interactive);
        assert_eq!(row.ttft.snapshot().count, 1);
        assert_eq!(row.gap.snapshot().count, 0);
        assert_eq!(row.e2e.snapshot().count, 0);
        assert_eq!(row.completion.snapshot().count, 0);
    }
    #[test]
    fn token_buckets_keep_exact_totals_and_infinite_tail() {
        let tokens = Tokens::default();
        for n in [0, 16, 17, 262144] {
            tokens.record(n);
        }
        let snapshot = tokens.snapshot();
        assert_eq!(snapshot.count, 4);
        assert_eq!(snapshot.sum_seconds, 262177.0);
        assert_eq!(snapshot.cumulative[0], (0.0, 1));
        assert_eq!(snapshot.cumulative[1], (16.0, 2));
        assert_eq!(snapshot.cumulative[2], (32.0, 3));
        assert_eq!(snapshot.cumulative[15], (f64::INFINITY, 4));
    }
}
