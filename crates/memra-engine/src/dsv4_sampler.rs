//! Plain DSV4 device sampling. CPU radix remains the oracle and default.
use crate::dsv4_gpu::{Dsv4PenaltyCfg, Dsv4SampleCfg, dsv4_pos_uniform};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::Arc;

type Res<T> = Result<T, String>;
pub const NUMERIC_CLASS: &str = "device-f64-exp-tree-cdf-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dsv4Sampler {
    Host,
    Device,
}
pub fn dsv4_sampler() -> Res<Dsv4Sampler> {
    static MODE: std::sync::OnceLock<Res<Dsv4Sampler>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| match std::env::var("MEMRA_DSV4_SAMPLER") {
        Err(std::env::VarError::NotPresent) => Ok(Dsv4Sampler::Host),
        Ok(v) if v == "host" => Ok(Dsv4Sampler::Host),
        Ok(v) if v == "device" => Ok(Dsv4Sampler::Device),
        v => Err(format!(
            "MEMRA_DSV4_SAMPLER expected host|device, got {v:?}"
        )),
    })
    .clone()
}

/// Lazily allocated host counts and reused lists of touched IDs. A draw only
/// clears the previous window and counts the current one, never the whole vocab.
#[derive(Default)]
struct PenaltyScratch {
    counts: Vec<i32>,
    ids: Vec<usize>,
    updates: Vec<usize>,
}
impl PenaltyScratch {
    fn update(&mut self, n: usize, window: &[u32], last_n: usize) {
        self.counts.resize(n, 0);
        // Retain pending IDs if an earlier sparse upload returned an error.
        self.updates.extend_from_slice(&self.ids);
        for &id in &self.ids {
            self.counts[id] = 0;
        }
        self.ids.clear();
        for &id in &window[window.len().saturating_sub(last_n)..] {
            let id = id as usize;
            if id < n {
                if self.counts[id] == 0 {
                    self.ids.push(id);
                }
                self.counts[id] += 1;
            }
        }
        self.updates.extend_from_slice(&self.ids);
        self.updates.sort_unstable();
        self.updates.dedup();
    }
    fn uploaded(&mut self) {
        self.updates.clear();
    }
}

/// Request-owned scratch. No global buffers or request-time policy mutation.
/// Inputs are never modified. The result allocation has a trailing gate canary.
pub struct Dsv4DeviceSampler {
    stream: Arc<CudaStream>,
    n: usize,
    input: CudaSlice<f32>,
    values: CudaSlice<f32>,
    keys0: CudaSlice<u64>,
    keys1: CudaSlice<u64>,
    prefix: CudaSlice<f64>,
    blocks: CudaSlice<f64>,
    counts: CudaSlice<i32>,
    penalty: PenaltyScratch,
    result: CudaSlice<u32>,
    calls: u64,
}
impl Dsv4DeviceSampler {
    pub fn new(stream: Arc<CudaStream>, n: usize) -> Res<Self> {
        if n == 0 || n > (1 << 24) {
            return Err("device sampler vocab outside 1..=2^24".into());
        }
        let input = stream.alloc_zeros(n + 1).map_err(|e| e.to_string())?;
        let values = stream.alloc_zeros(n + 1).map_err(|e| e.to_string())?;
        let keys0 = stream.alloc_zeros(n + 1).map_err(|e| e.to_string())?;
        let keys1 = stream.alloc_zeros(n + 1).map_err(|e| e.to_string())?;
        let prefix = stream.alloc_zeros(n + 1).map_err(|e| e.to_string())?;
        let blocks = stream
            .alloc_zeros(n.div_ceil(256) + 2)
            .map_err(|e| e.to_string())?;
        let counts = stream.alloc_zeros(n + 1).map_err(|e| e.to_string())?;
        let result = stream
            .clone_htod(&[0u32, 0, 0x5a17cafe])
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            n,
            input,
            values,
            keys0,
            keys1,
            prefix,
            blocks,
            counts,
            penalty: PenaltyScratch::default(),
            result,
            calls: 0,
        })
    }
    pub(crate) fn validate_source(&self, stream: &Arc<CudaStream>, n: usize) -> Res<()> {
        if !Arc::ptr_eq(&self.stream, stream) || self.n != n {
            return Err("device sampler stream or vocabulary mismatch".into());
        }
        Ok(())
    }
    pub fn engagements(&self) -> u64 {
        self.calls
    }
    pub fn check_canary_for_gate(&self) -> Res<()> {
        let words = self
            .stream
            .clone_dtoh(&self.result)
            .map_err(|e| e.to_string())?;
        if words[2] != 0x5a17cafe {
            return Err("device sampler output canary".into());
        }
        macro_rules! zero_guard {
            ($buffer:expr) => {{
                let buffer = &$buffer;
                let tail = self
                    .stream
                    .clone_dtoh(&buffer.slice(buffer.len() - 1..))
                    .map_err(|e| e.to_string())?;
                if tail[0] != 0 as _ {
                    return Err("device sampler scratch canary".into());
                }
            }};
        }
        zero_guard!(self.input);
        zero_guard!(self.values);
        zero_guard!(self.keys0);
        zero_guard!(self.keys1);
        zero_guard!(self.prefix);
        zero_guard!(self.blocks);
        zero_guard!(self.counts);
        Ok(())
    }
    /// Prefill/restore adapter: the existing cache API owns a host logits row.
    pub fn sample_host_row(
        &mut self,
        row: &[f32],
        pos: usize,
        cfg: &Dsv4SampleCfg,
        window: &[u32],
        penalty: Option<&Dsv4PenaltyCfg>,
    ) -> Res<u32> {
        if row.len() != self.n {
            return Err("device sampler row length".into());
        }
        self.stream
            .memcpy_htod(row, &mut self.input.slice_mut(0..self.n))
            .map_err(|e| e.to_string())?;
        let input = self.input.device_ptr(&self.stream).0;
        // The allocation is retained on self and ordered on this exact stream.
        unsafe { self.sample_ptr(input as *const f32, pos, cfg, window, penalty) }
    }
    /// # Safety
    /// `input` must name a full vocabulary f32 row on this sampler's context,
    /// whose producer and lifetime are ordered on this sampler's stream.
    pub unsafe fn sample_ptr(
        &mut self,
        input: *const f32,
        pos: usize,
        cfg: &Dsv4SampleCfg,
        window: &[u32],
        penalty: Option<&Dsv4PenaltyCfg>,
    ) -> Res<u32> {
        if cfg.temperature.is_nan()
            || cfg.temperature <= 0.0
            || !(cfg.top_p > 0.0 && cfg.top_p <= 1.0)
        {
            return Err("dsv4 sampled path: need temperature > 0 and top_p in (0,1]".into());
        }
        let n = self.n;
        let pc = penalty.filter(|pc| pc.armed());
        // Keep the unpenalized branch and its CUDA program unchanged. Armed
        // draws upload only the union of the previous/current window's IDs,
        // including zero counts for IDs leaving the window. Adjacent IDs share
        // a copy; all host storage remains owned through the final stream drain.
        if let Some(pc) = pc {
            // An empty window may issue no H2D copy, so bind explicitly here.
            self.stream
                .context()
                .bind_to_thread()
                .map_err(|e| e.to_string())?;
            self.penalty.update(n, window, pc.last_n);
            let mut next = 0;
            while next < self.penalty.updates.len() {
                let start = self.penalty.updates[next];
                let mut end = start + 1;
                next += 1;
                while next < self.penalty.updates.len() && self.penalty.updates[next] == end {
                    end += 1;
                    next += 1;
                }
                self.stream
                    .memcpy_htod(
                        &self.penalty.counts[start..end],
                        &mut self.counts.slice_mut(start..end),
                    )
                    .map_err(|e| e.to_string())?;
            }
            self.penalty.uploaded();
        } else {
            self.stream
                .memset_zeros(&mut self.counts)
                .map_err(|e| e.to_string())?;
        }
        let k = if cfg.top_k == 0 { n } else { cfg.top_k.min(n) };
        unsafe {
            crate::dsv4_ffi::ck(
                "device sampler",
                crate::dsv4_ffi::memra_dsv4_sample_device(
                    input,
                    self.values.device_ptr_mut(&self.stream).0 as *mut f32,
                    self.keys0.device_ptr_mut(&self.stream).0 as *mut u64,
                    self.keys1.device_ptr_mut(&self.stream).0 as *mut u64,
                    self.prefix.device_ptr_mut(&self.stream).0 as *mut f64,
                    self.blocks.device_ptr_mut(&self.stream).0 as *mut f64,
                    self.counts.device_ptr(&self.stream).0 as *const i32,
                    self.result.device_ptr_mut(&self.stream).0 as *mut u32,
                    n as i32,
                    k as i32,
                    cfg.temperature as f64,
                    cfg.top_p as f64,
                    dsv4_pos_uniform(cfg.seed, pos),
                    pc.map_or(1.0, |p| p.repeat),
                    pc.map_or(0.0, |p| p.freq),
                    pc.map_or(0.0, |p| p.present),
                    self.stream.cu_stream() as *mut std::ffi::c_void,
                ),
            )?;
        }
        let mut token = [0u32];
        self.stream
            .memcpy_dtoh(&self.result.slice(0..1), &mut token)
            .map_err(|e| e.to_string())?;
        self.stream.synchronize().map_err(|e| e.to_string())?;
        if token[0] as usize >= n {
            return Err("device sampler refuses nonfinite penalized logits/probabilities".into());
        }
        self.calls += 1;
        Ok(token[0])
    }
}

#[cfg(test)]
mod penalty_tests {
    use super::PenaltyScratch;

    #[test]
    fn sparse_counts_clear_departed_ids_and_respect_the_window() {
        let mut scratch = PenaltyScratch::default();
        scratch.update(16, &[1, 2, 2, 15, 99], 4);
        assert_eq!(scratch.updates, [2, 15]);
        assert_eq!(scratch.counts[2], 2);
        assert_eq!(scratch.counts[15], 1);
        let allocation = scratch.counts.as_ptr();
        scratch.uploaded();
        scratch.update(16, &[2, 3, 3, 4], 3);
        assert_eq!(scratch.counts.as_ptr(), allocation, "reuse host allocation");
        assert_eq!(scratch.updates, [2, 3, 4, 15]);
        assert_eq!(scratch.counts[2], 0);
        assert_eq!(scratch.counts[15], 0);
        assert_eq!(scratch.counts[3], 2);
        assert_eq!(scratch.counts[4], 1);
        scratch.uploaded();
        scratch.update(16, &[], 3);
        assert_eq!(scratch.updates, [3, 4]);
        assert!(scratch.counts.iter().all(|&count| count == 0));
    }

    #[test]
    fn sparse_counts_are_complete_after_an_unpenalized_device_clear() {
        let mut scratch = PenaltyScratch::default();
        scratch.update(8, &[1, 1, 7], 8);
        // The unpenalized arm clears the device without modifying host scratch.
        let mut device = [0; 8];
        scratch.update(8, &[1, 2, 2], 8);
        for &id in &scratch.updates {
            device[id] = scratch.counts[id];
        }
        assert_eq!(device, [0, 1, 2, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn incomplete_upload_keeps_departed_ids_pending() {
        let mut scratch = PenaltyScratch::default();
        scratch.update(8, &[7], 1);
        scratch.uploaded();
        scratch.update(8, &[1], 1);
        // Upload fails before the old ID 7 has been cleared on device.
        scratch.update(8, &[2], 1);
        assert_eq!(scratch.updates, [1, 2, 7]);
        assert_eq!(scratch.counts[1], 0);
        assert_eq!(scratch.counts[2], 1);
        assert_eq!(scratch.counts[7], 0);
    }
}
