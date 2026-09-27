//! Fixed-bucket latency histograms for the Prometheus exposition (memra#522).
//!
//! Cardinality is bounded by construction: bucket edges are a fixed compile-time array
//! (never derived from request content), and every histogram this module exports is
//! attached to an existing bounded label (a registered route/model name, capped by
//! `route_telemetry`'s own registry size, which is operator-configured and small).
//!
//! Cumulative Prometheus histogram semantics: bucket `i` holds the count of samples
//! `<= edges[i]`, and the last edge is `+Inf` so every sample lands somewhere.

use std::sync::atomic::{AtomicU64, Ordering};

/// Latency bucket upper bounds in seconds. Chosen to span sub-millisecond step costs
/// through multi-minute prefill/queue waits (the range the current /metrics p50/p99
/// boards report), with the standard Prometheus-style `+Inf` closer.
pub const LATENCY_BUCKETS_S: [f64; 16] = [
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    f64::INFINITY,
];

const N: usize = LATENCY_BUCKETS_S.len();

/// A bounded fixed-bucket histogram: `N` atomic bucket counters plus sum and count for the
/// Prometheus `_sum`/`_count` series. Every field is `Relaxed`: this is a monitoring counter,
/// not a synchronization point, and the existing `Metrics` mutex publish cadence already
/// gives scrapers a consistent-enough snapshot.
#[derive(Debug, Default)]
pub struct Histogram {
    buckets: [AtomicU64; N],
    sum_micros: AtomicU64,
    count: AtomicU64,
}

/// A point-in-time read of a [`Histogram`], the shape both the JSON and Prometheus
/// exposition render from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HistogramSnapshot {
    /// Per-bucket upper bound (seconds) paired with the CUMULATIVE count at or under it,
    /// in ascending order, ending at `+Inf`.
    pub cumulative: Vec<(f64, u64)>,
    pub sum_seconds: f64,
    pub count: u64,
}

impl Histogram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one sample. Negative or NaN durations (a clock oddity, never a real
    /// measurement) are dropped rather than corrupting `sum`/`count`.
    pub fn record(&self, seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        let bucket = LATENCY_BUCKETS_S
            .iter()
            .position(|&edge| seconds <= edge)
            .unwrap_or(N - 1);
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        // Microsecond fixed-point sum avoids float-add races changing the reported total
        // depending on interleaving; this only loses sub-microsecond precision.
        let micros = (seconds * 1_000_000.0).round().clamp(0.0, u64::MAX as f64) as u64;
        self.sum_micros.fetch_add(micros, Ordering::Relaxed);
    }

    /// Convenience for millisecond samples (most call sites already have a `u64` ms value).
    pub fn record_ms(&self, ms: u64) {
        self.record(ms as f64 / 1000.0);
    }

    pub fn snapshot(&self) -> HistogramSnapshot {
        let mut cumulative = 0u64;
        let cumulative_counts = self
            .buckets
            .iter()
            .zip(LATENCY_BUCKETS_S.iter())
            .map(|(b, &edge)| {
                cumulative += b.load(Ordering::Relaxed);
                (edge, cumulative)
            })
            .collect();
        HistogramSnapshot {
            cumulative: cumulative_counts,
            sum_seconds: self.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            count: self.count.load(Ordering::Relaxed),
        }
    }
}

impl HistogramSnapshot {
    /// Render as Prometheus text exposition lines for one metric `name` with `labels`
    /// (already formatted, e.g. `model="foo",route="bar"`, or empty for none).
    pub fn render_prometheus(&self, name: &str, labels: &str) -> String {
        let mut out = String::new();
        let with = |extra: &str| -> String {
            if labels.is_empty() {
                format!("{{{extra}}}")
            } else if extra.is_empty() {
                format!("{{{labels}}}")
            } else {
                format!("{{{labels},{extra}}}")
            }
        };
        for (edge, count) in &self.cumulative {
            let le = if edge.is_infinite() {
                "+Inf".to_string()
            } else {
                format!("{edge}")
            };
            out.push_str(&format!(
                "{name}_bucket{} {count}\n",
                with(&format!("le=\"{le}\""))
            ));
        }
        out.push_str(&format!("{name}_sum{} {}\n", with(""), self.sum_seconds));
        out.push_str(&format!("{name}_count{} {}\n", with(""), self.count));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_cumulative_and_every_sample_lands_somewhere() {
        let h = Histogram::new();
        h.record(0.001);
        h.record(0.05);
        h.record(10_000.0); // far past the last finite edge; must land in +Inf, not panic.
        let snap = h.snapshot();
        assert_eq!(snap.count, 3);
        let (_, last_count) = *snap.cumulative.last().unwrap();
        assert_eq!(last_count, 3);
        // Monotonic non-decreasing cumulative counts.
        let mut prev = 0u64;
        for (_, c) in &snap.cumulative {
            assert!(*c >= prev);
            prev = *c;
        }
    }

    #[test]
    fn negative_and_nan_samples_are_dropped() {
        let h = Histogram::new();
        h.record(-1.0);
        h.record(f64::NAN);
        assert_eq!(h.snapshot().count, 0);
    }

    #[test]
    fn record_ms_matches_record_seconds() {
        let h = Histogram::new();
        h.record_ms(500);
        let snap = h.snapshot();
        assert_eq!(snap.count, 1);
        assert!((snap.sum_seconds - 0.5).abs() < 1e-6);
    }

    #[test]
    fn prometheus_rendering_is_well_formed_and_ends_in_inf() {
        let h = Histogram::new();
        h.record(0.2);
        let snap = h.snapshot();
        let text = snap.render_prometheus("memra_test_seconds", "model=\"m\"");
        assert!(text.contains("le=\"+Inf\""));
        assert!(text.contains("memra_test_seconds_sum{model=\"m\"}"));
        assert!(text.contains("memra_test_seconds_count{model=\"m\"} 1"));
    }
}
