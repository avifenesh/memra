//! Bounded operation-delta -> D telemetry-schema join. Disk submitted bytes are
//! cumulative; physical bytes remain null after any uninstrumented operation.
//! Empty interval distributions mean no observed waits, not measured zero latency.
use super::StorageSample;
use crate::contracts::{Error, Result, Wire};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy)]
pub enum StorageDirection {
    Read,
    Write,
}
/// External collector supplies instantaneous device facts. This adapter never
/// fabricates hardware data from StorageSample or upgrades a CPU fixture to GPU.
pub enum DeviceSample {
    CpuFixture {
        device: u32,
    },
    Gpu {
        device: u32,
        clock_mhz: u64,
        power_w: f64,
        temperature_c: f64,
        vram_bytes: u64,
        routes: [[u64; 2]; 5],
    },
}
impl DeviceSample {
    fn values(&self) -> Result<(&'static str, Value)> {
        let (kind, id, clock, power, temperature, vram, routes) = match *self {
            Self::CpuFixture { device } => (
                "cpu-fixture",
                device,
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
                [[0; 2]; 5],
            ),
            Self::Gpu {
                device,
                clock_mhz,
                power_w,
                temperature_c,
                vram_bytes,
                routes,
            } => {
                if !power_w.is_finite()
                    || power_w < 0.0
                    || !temperature_c.is_finite()
                    || temperature_c < 0.0
                {
                    return Err(Error::InvalidLayout);
                }
                (
                    "gpu",
                    device,
                    json!(clock_mhz),
                    json!(power_w),
                    json!(temperature_c),
                    json!(vram_bytes),
                    routes,
                )
            }
        };
        // Route order: host, host-bounce, local, nvme, pcie-p2p. Counters are
        // cumulative actual device movement supplied by the device collector.
        let route = |i: usize| json!({"bytes_in": routes[i][0], "bytes_out":routes[i][1]});
        Ok((
            kind,
            json!({"device":id,"clock_mhz":clock,"power_w":power,
            "temperature_c":temperature,"vram_bytes":vram,
            "routes":{"host":route(0),"host-bounce":route(1),"local":route(2),"nvme":route(3),"pcie-p2p":route(4)}}),
        ))
    }
}
pub struct StorageTelemetry {
    limit: usize,
    samples: Vec<StorageSample>,
    read_bytes: u64,
    write_bytes: u64,
    physical_bytes: Option<u64>,
    last_ns: Option<u64>,
}
impl StorageTelemetry {
    pub fn new(max_samples_per_interval: usize) -> Result<Self> {
        if max_samples_per_interval == 0 {
            return Err(Error::InvalidLayout);
        }
        Ok(Self {
            limit: max_samples_per_interval,
            samples: Vec::new(),
            read_bytes: 0,
            write_bytes: 0,
            physical_bytes: Some(0),
            last_ns: None,
        })
    }
    pub fn observe(&mut self, sample: &StorageSample, direction: StorageDirection) -> Result<()> {
        sample.validate()?;
        if self.samples.len() >= self.limit {
            return Err(Error::Capacity);
        }
        let counter = match direction {
            StorageDirection::Read => &mut self.read_bytes,
            StorageDirection::Write => &mut self.write_bytes,
        };
        let next = counter
            .checked_add(sample.io_bytes)
            .ok_or(Error::Overflow)?;
        let physical = match (self.physical_bytes, sample.physical_bytes) {
            (Some(old), Some(delta)) => Some(old.checked_add(delta).ok_or(Error::Overflow)?),
            _ => None,
        };
        *counter = next;
        self.physical_bytes = physical;
        self.samples.push(sample.clone());
        Ok(())
    }
    /// Call once per nominal 250ms collector tick (<=500ms actual gap); this function neither sleeps nor
    /// labels a per-operation timestamp as interval telemetry. Missing/late ticks
    /// fail, preserving the pending sample set for diagnostics.
    pub fn json_line(
        &mut self,
        monotonic_ns: u64,
        device: &DeviceSample,
        queue_depth: u64,
        pinned_bytes: u64,
        pageable_bytes: Option<u64>,
    ) -> Result<String> {
        if self.last_ns.is_some_and(|last| {
            monotonic_ns
                .checked_sub(last)
                .is_none_or(|delta| delta == 0 || delta > 500_000_000)
        }) {
            return Err(Error::InvalidLayout);
        }
        let (kind, device_row) = device.values()?;
        let waits = |select: fn(&StorageSample) -> Option<u64>| {
            percentiles(self.samples.iter().filter_map(select).collect())
        };
        let row = json!({
            "schema_version":1,"kind":kind,"monotonic_ns":monotonic_ns,"interval_ms":250,
            "devices":[device_row],
            "host":{"pinned_bytes":pinned_bytes,"pageable_bytes":pageable_bytes},
            "nvme":{"queue_depth":queue_depth,"read_bytes":self.read_bytes,
                "write_bytes":self.write_bytes,"physical_bytes":self.physical_bytes},
            "wait_ns":{"io":waits(|s| s.io_ns),"queue":waits(|s| s.queue_ns),
                "h2d":waits(|s| s.h2d_ns),"d2h":waits(|s| s.d2h_ns),"p2p":waits(|s| s.p2p_ns)}
        });
        let line = serde_json::to_string(&row).map_err(|_| Error::Corrupt)?;
        self.last_ns = Some(monotonic_ns);
        self.samples.clear();
        Ok(line)
    }
}
fn percentiles(mut values: Vec<u64>) -> Value {
    values.sort_unstable();
    let percentile = |p: usize| {
        if values.is_empty() {
            0
        } else {
            values[(values.len() * p).div_ceil(100) - 1]
        }
    };
    json!({"p50":percentile(50),"p95":percentile(95),"p99":percentile(99)})
}
