//! Per-model latency histograms for the batched central worker's hybrid lane (memra#522
//! follow-up to PR #896, which wired the same [`crate::histogram::Histogram`] primitive
//! and Prometheus exposition into the one dedicated serve route in the codebase, DSv4).
//!
//! The hybrid lane is every model without a dedicated route: the central worker's shared
//! scheduler batches admitted sessions across models. There is no per-route book here, so
//! this module is a plain per-model registry: cardinality is bounded by the number of
//! distinct model names the operator has loaded, the same bound `route_telemetry` relies
//! on for its own per-route histograms.
//!
//! Queue, TTFT and E2E histograms per model:
//!   * `queue_wait`: time between a request entering the worker's admission queue
//!     (`Request::queued_at`, stamped by the HTTP handler at submission, or reset at
//!     construction for every internally-built request (a step-OOM park replay, a test)
//!     and the worker admitting it (`Session::t0`). Recorded once per admitted request,
//!     at the single admission call site in the scheduler loop.
//!   * `ttft`: admission (`Session::t0`) to this session's first committed token
//!     (`Session::generated` going from empty to non-empty). Recorded once per session,
//!     at the shared `push_generated` choke point every decode path (plain, spec, gspec,
//!     dspark, glm5, step) already funnels through. Admission-based, not arrival-based, so
//!     it composes with `queue_wait` the same way the route's `queue_wait` + `round`
//!     decomposition already does: total client wait to first token ~= queue_wait + ttft.
//!   * `e2e`: admission to finish, SUCCESS PATH ONLY (memra#896's contract: a failed,
//!     aborted, or OOM-torn-down session never widens the latency a client-facing SLO
//!     reads). Recorded once per session, at retire, gated on the same
//!     `!oom_teardown && !aborted && !errored` predicate the completion-history record
//!     already uses.
//!
//! Emitted-token gaps use three fixed lane rows per model. They record every interval
//! between successfully accepted worker token events, including empty UTF-8 fragments
//! and EOS IDs. A session emitting N token IDs contributes N-1 samples. This is server
//! event emission timing, not network flush timing or a compute-round proxy.
//!
//! KNOWN IMPRECISION (documented, not silently accepted): a step-OOM park replay rebuilds
//! a fresh `Session` (`generated: Vec::new()`), so if the original attempt had already
//! emitted at least one token before the OOM, the replay's first token records a SECOND
//! `ttft` sample for the same logical client request. Bounded by the OOM retry ceiling,
//! rare in practice, and the same class of imprecision the route's own `service_estimate_s`
//! doc already accepts for prime time.

use crate::histogram::{Histogram, HistogramSnapshot};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::Duration;

#[derive(Default)]
struct ModelHist {
    queue_wait: Histogram,
    ttft: Histogram,
    e2e: Histogram,
    token_gap: [Histogram; 3],
}

/// The published view of one model's hybrid-lane histograms (the /metrics Prometheus
/// exposition's `hybrid` rows). Not carried into the JSON body, same as the route
/// histograms: that body stays byte-for-byte unchanged for every existing consumer.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct HybridSnapshot {
    pub model: String,
    pub queue_wait_hist: HistogramSnapshot,
    pub ttft_hist: HistogramSnapshot,
    pub e2e_hist: HistogramSnapshot,
    pub token_gap_hists: [HistogramSnapshot; 3],
}

fn registry() -> &'static RwLock<HashMap<String, ModelHist>> {
    static R: OnceLock<RwLock<HashMap<String, ModelHist>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

fn with_model<F: FnOnce(&ModelHist)>(model: &str, f: F) {
    // Fast path: the model is already registered (every model after its first request).
    if let Ok(map) = registry().read()
        && let Some(h) = map.get(model)
    {
        f(h);
        return;
    }
    // Slow path: get-or-create under the write lock. A respawn or a concurrent first
    // request from two models races here exactly like `route_telemetry::register`.
    let mut map = registry().write().unwrap_or_else(|p| p.into_inner());
    let h = map.entry(model.to_string()).or_default();
    f(h);
}

pub(crate) fn record_queue_wait(model: &str, waited: Duration) {
    with_model(model, |h| h.queue_wait.record(waited.as_secs_f64()));
}

pub(crate) fn record_ttft(model: &str, elapsed: Duration) {
    with_model(model, |h| h.ttft.record(elapsed.as_secs_f64()));
}

pub(crate) fn record_e2e(model: &str, elapsed: Duration) {
    with_model(model, |h| h.e2e.record(elapsed.as_secs_f64()));
}

/// A per-session clock for accepted worker token events. UTF-8 fragments and EOS
/// still represent token IDs; the first event starts the clock without a gap sample.
/// Observations are cumulative and unsampled, with exactly three possible lane labels.
#[derive(Default)]
pub(crate) struct EmissionClock {
    pub(crate) request: crate::request_metrics::Clock,
    last: Option<std::time::Instant>,
}

impl EmissionClock {
    pub(crate) fn sent(&mut self, model: &str, lane: crate::lanes::Lane) {
        self.request.sent();
        self.sent_at(model, lane, std::time::Instant::now());
    }

    fn sent_at(&mut self, model: &str, lane: crate::lanes::Lane, now: std::time::Instant) {
        if let Some(previous) = self.last.replace(now) {
            with_model(model, |h| {
                h.token_gap[lane.idx()]
                    .record(now.saturating_duration_since(previous).as_secs_f64());
            });
        }
    }
}

/// Every registered model's snapshot, sorted by name (a stable /metrics order, same
/// convention as `route_telemetry::all`).
pub(crate) fn all() -> Vec<HybridSnapshot> {
    let map = registry().read().unwrap_or_else(|p| p.into_inner());
    let mut v: Vec<HybridSnapshot> = map
        .iter()
        .map(|(model, h)| HybridSnapshot {
            model: model.clone(),
            queue_wait_hist: h.queue_wait.snapshot(),
            ttft_hist: h.ttft.snapshot(),
            e2e_hist: h.e2e.snapshot(),
            token_gap_hists: std::array::from_fn(|i| h.token_gap[i].snapshot()),
        })
        .collect();
    v.sort_by(|a, b| a.model.cmp(&b.model));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_gaps_count_intervals_per_session_and_lane() {
        let now = std::time::Instant::now();
        let model = "t-emitted-gaps";
        let mut clock = EmissionClock::default();
        clock.sent_at(model, crate::lanes::Lane::Interactive, now);
        clock.sent_at(
            model,
            crate::lanes::Lane::Interactive,
            now + Duration::from_millis(20),
        );
        clock.sent_at(
            model,
            crate::lanes::Lane::Interactive,
            now + Duration::from_millis(50),
        );
        let mut next = EmissionClock::default();
        next.sent_at(
            model,
            crate::lanes::Lane::Harvest,
            now + Duration::from_secs(1),
        );
        next.sent_at(
            model,
            crate::lanes::Lane::Harvest,
            now + Duration::from_millis(1010),
        );
        let snap = all().into_iter().find(|s| s.model == model).unwrap();
        assert_eq!(snap.token_gap_hists[0].count, 2);
        assert!((snap.token_gap_hists[0].sum_seconds - 0.05).abs() < 1e-9);
        assert_eq!(snap.token_gap_hists[1].count, 0);
        assert_eq!(snap.token_gap_hists[2].count, 1);
        assert!((snap.token_gap_hists[2].sum_seconds - 0.01).abs() < 1e-9);
    }

    /// Registration is get-or-create and lookups are by model name, mirroring
    /// `route_telemetry`'s own registry test.
    #[test]
    fn recording_is_get_or_create_and_scoped_per_model() {
        record_queue_wait("t-hybrid-a", Duration::from_millis(5));
        record_queue_wait("t-hybrid-b", Duration::from_millis(9));
        let snaps = all();
        let a = snaps.iter().find(|s| s.model == "t-hybrid-a").unwrap();
        let b = snaps.iter().find(|s| s.model == "t-hybrid-b").unwrap();
        assert_eq!(a.queue_wait_hist.count, 1);
        assert_eq!(b.queue_wait_hist.count, 1);
        assert_eq!(a.ttft_hist.count, 0, "no ttft sample recorded for a yet");
    }

    /// memra#522: a sample lands in exactly one bucket (the histogram primitive's own
    /// cumulative-count contract, exercised again here through this module's public
    /// recording seam rather than the primitive directly).
    #[test]
    fn one_sample_lands_in_exactly_one_bucket() {
        record_ttft("t-hybrid-bucket", Duration::from_millis(1));
        let snap = all()
            .into_iter()
            .find(|s| s.model == "t-hybrid-bucket")
            .unwrap();
        let hist = snap.ttft_hist;
        assert_eq!(hist.count, 1);
        // Exactly one bucket boundary's cumulative count increases by 1 versus the bucket
        // immediately below it; every other boundary is unchanged.
        let mut prev = 0u64;
        let mut risers = 0usize;
        for (_, count) in &hist.cumulative {
            if *count > prev {
                risers += 1;
            }
            prev = *count;
        }
        assert_eq!(
            risers, 1,
            "a single sample must move exactly one cumulative bucket boundary"
        );
    }

    /// RED ARM (memra#522 acceptance: "reusing the same histogram primitive ... with
    /// success-only E2E like #896"): a failed/cancelled request must never land in the
    /// success-only E2E histogram. This module has no session/retire concept of its own
    /// (that lives in `worker.rs`'s retire loop, which gates the `record_e2e` call on
    /// `!oom_teardown && !aborted && !errored`); this test proves the module-level
    /// contract that `record_e2e` is the ONLY writer to the `e2e` histogram, so a caller
    /// that never calls it (the failure/cancel paths) contributes nothing, while
    /// `record_queue_wait` and `record_ttft` on the same model do not leak into `e2e`.
    #[test]
    fn a_request_that_never_reaches_record_e2e_leaves_the_e2e_histogram_untouched() {
        record_queue_wait("t-hybrid-red", Duration::from_millis(3));
        record_ttft("t-hybrid-red", Duration::from_millis(4));
        // Deliberately no record_e2e call: the stand-in for a failed/cancelled/OOM-torn-down
        // session, which the worker retire loop never routes into record_e2e.
        let snap = all()
            .into_iter()
            .find(|s| s.model == "t-hybrid-red")
            .unwrap();
        assert_eq!(
            snap.e2e_hist.count, 0,
            "a request that never finished successfully must not widen the E2E histogram"
        );
        assert_eq!(snap.queue_wait_hist.count, 1);
        assert_eq!(snap.ttft_hist.count, 1);
        // Now the success path lands, and only it moves e2e.
        record_e2e("t-hybrid-red", Duration::from_millis(50));
        let snap2 = all()
            .into_iter()
            .find(|s| s.model == "t-hybrid-red")
            .unwrap();
        assert_eq!(snap2.e2e_hist.count, 1);
    }

    #[test]
    fn snapshots_are_sorted_by_model_name() {
        record_queue_wait("t-hybrid-z", Duration::from_millis(1));
        record_queue_wait("t-hybrid-m", Duration::from_millis(1));
        let names: Vec<String> = all()
            .into_iter()
            .filter(|s| s.model.starts_with("t-hybrid-"))
            .map(|s| s.model)
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }
}
