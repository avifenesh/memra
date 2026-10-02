//! Read-only allocator observations. KV gauges cover canonical live and parked state
//! planes, including recurrent/indexer/draft planes; prefix snapshots and transient
//! rollback/graph workspace are separate allocations. Capacity is reserved plane
//! bytes, not free GPU memory. Device samples are cached at the existing worker/admission
//! cadence and carry an age so a quiescent worker never presents an old sample as fresh.
use crate::request_metrics::Backend;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy, Default)]
pub(crate) struct Snapshot {
    pub used: u64,
    pub capacity: u64,
    pub active: u64,
    pub queued: u64,
}
impl Snapshot {
    pub(crate) fn add(&mut self, (used, capacity): (usize, usize)) {
        self.used += used as u64;
        self.capacity += capacity as u64;
    }
}
#[derive(Default)]
struct Registry {
    models: BTreeMap<(String, Backend), Snapshot>,
    devices: BTreeMap<usize, (u64, Instant)>,
}
fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(Default::default)
}
pub(crate) struct HybridLifetime;
impl Drop for HybridLifetime {
    fn drop(&mut self) {
        let mut r = registry().lock().unwrap_or_else(|e| e.into_inner());
        for ((_, backend), row) in &mut r.models {
            if *backend == Backend::Hybrid {
                *row = Snapshot::default();
            }
        }
    }
}
pub(crate) fn publish_hybrid(rows: BTreeMap<String, Snapshot>) {
    let mut r = registry().lock().unwrap_or_else(|e| e.into_inner());
    r.models
        .retain(|(_, backend), _| *backend != Backend::Hybrid);
    r.models.extend(
        rows.into_iter()
            .map(|(model, snapshot)| ((model, Backend::Hybrid), snapshot)),
    );
}
pub(crate) fn device_free(device: usize, bytes: u64) {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .devices
        .insert(device, (bytes, Instant::now()));
}
pub(crate) fn render(m: &crate::worker::Metrics) -> String {
    let (rows, devices) = {
        let r = registry().lock().unwrap_or_else(|e| e.into_inner());
        (r.models.clone(), r.devices.clone())
    };
    let mut out = String::new();
    for (index, (name, help)) in [
        ("memra_kv_used_bytes", "Backed canonical KV/recurrent/indexer/draft state-plane bytes in live and parked sessions; excludes prefix and rollback snapshots."),
        ("memra_kv_capacity_bytes", "Reserved capacity of the canonical state planes counted by memra_kv_used_bytes, including VMM allocation granularity."),
        ("memra_active_sessions", "Currently active hybrid sessions at the latest worker publication."),
        ("memra_queued_sessions", "Hybrid sessions waiting for admission at the latest worker publication."),
    ].into_iter().enumerate() {
        crate::prometheus_header(&mut out, name, "gauge", help);
        for ((model, backend), row) in &rows {
            let value = [row.used, row.capacity, row.active, row.queued][index];
            out.push_str(&format!("{name}{{model=\"{}\",route=\"{}\"}} {value}\n", crate::prometheus_label(model), backend.as_str()));
        }
    }
    crate::prometheus_header(
        &mut out,
        "memra_device_free_bytes",
        "gauge",
        "Driver free bytes on each hybrid-model-owned CUDA device at its latest worker sample.",
    );
    for (device, (bytes, _)) in &devices {
        out.push_str(&format!(
            "memra_device_free_bytes{{device=\"{device}\"}} {bytes}\n"
        ));
    }
    crate::prometheus_header(
        &mut out,
        "memra_device_memory_sample_age_seconds",
        "gauge",
        "Age of the cached device-memory observation; no device query is issued by a metrics scrape.",
    );
    for (device, (_, at)) in &devices {
        out.push_str(&format!(
            "memra_device_memory_sample_age_seconds{{device=\"{device}\"}} {}\n",
            at.elapsed().as_secs_f64()
        ));
    }
    let prefix = [(
        "hybrid",
        "device",
        [
            m.prefix_hits,
            m.prefix_misses,
            m.prefix_inserts,
            m.prefix_evictions,
            m.prefix_bytes,
        ],
    )];
    for (index, (name, kind, help)) in [
        (
            "memra_backend_prefix_cache_hits_total",
            "counter",
            "Successful prefix lookup probes by backend and storage tier.",
        ),
        (
            "memra_backend_prefix_cache_misses_total",
            "counter",
            "Unsuccessful prefix lookup probes by backend and storage tier.",
        ),
        (
            "memra_backend_prefix_cache_inserts_total",
            "counter",
            "Prefix insertions by backend and storage tier.",
        ),
        (
            "memra_backend_prefix_cache_evictions_total",
            "counter",
            "Prefix evictions and superseded entries by backend and storage tier.",
        ),
        (
            "memra_backend_prefix_cache_bytes",
            "gauge",
            "Currently retained prefix bytes by backend and storage tier.",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        crate::prometheus_header(&mut out, name, kind, help);
        for (backend, tier, values) in &prefix {
            out.push_str(&format!(
                "{name}{{route=\"{backend}\",tier=\"{tier}\"}} {}\n",
                values[index]
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_lifetime_clears_live_gauges_on_exit() {
        let model = "capacity-worker-exit".to_string();
        publish_hybrid(BTreeMap::from([(
            model.clone(),
            Snapshot {
                used: 100,
                capacity: 200,
                active: 1,
                queued: 2,
            },
        )]));
        let lifetime = HybridLifetime;
        assert_eq!(
            registry().lock().unwrap().models[&(model.clone(), Backend::Hybrid)].active,
            1
        );
        drop(lifetime);
        let r = registry().lock().unwrap();
        let row = r.models[&(model, Backend::Hybrid)];
        assert_eq!(
            (row.used, row.capacity, row.active, row.queued),
            (0, 0, 0, 0)
        );
    }
}
