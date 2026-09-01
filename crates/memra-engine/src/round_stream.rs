//! ROUND-STREAM: the model-generic device machinery for pre-issued M-round speculative
//! bursts with zero per-round host readbacks (extracted from the qwen spec loop 2026-07-12
//! so the gemma/next-model loops reuse it instead of re-growing their own).
//!
//! The pieces that live here are pure device-buffer plumbing — every model-specific thing
//! (the draft-chain graph, the verify trunk, commit semantics) stays in the caller:
//!   - `StreamBufs`: the per-burst device buffers (verify tokens, break, pending, ring,
//!     accept counters, device position) sized from (k, m_rounds).
//!   - `kv_len_ptr_table`: the per-layer `kvl.len_d` pointer table the device rollback
//!     kernel walks (pointers are stable for the cache's lifetime — cache.rs note).
//!   - `drain_ring`: the one host sync per M rounds — reads the ring and appends tokens.
//!
//! The kernels these feed (`spec_accept_greedy_dc`, `spec_ring_commit`,
//! `spec_rollback_stream`, `spec_seed_gather`, `spec_assemble_verify`) are already
//! model-generic in lib.rs; this module is the buffer/lifecycle half.

use crate::Engine;
use crate::cache::Cache;
use cudarc::driver::CudaSlice;
use std::ops::Range;

pub struct StreamBufs {
    /// assembled verify tokens [k+1]
    pub vtok_d: CudaSlice<u32>,
    /// p-min break markers [2]
    pub brk_d: CudaSlice<u32>,
    /// pending (bonus-fold) token [1]
    pub pend_d: CudaSlice<u32>,
    /// last verify prediction [1]
    pub last_pred_d: CudaSlice<u32>,
    /// device position counter (rope/append base)
    pub pos_ctr: CudaSlice<i32>,
    /// round-start position (rollback anchor)
    pub pos_start_d: CudaSlice<i32>,
    /// committed-token ring [m*(k+1)+1] (slot 0 = count)
    pub ring_d: CudaSlice<u32>,
    /// device accept counters [2]
    pub acc_d: CudaSlice<u32>,
    pub m_rounds: usize,
    pub k: usize,
}

impl StreamBufs {
    pub fn new(e: &Engine, k: usize, m_rounds: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(StreamBufs {
            vtok_d: e.alloc_u32_zeroed(k + 1)?,
            brk_d: e.alloc_u32_zeroed(2)?,
            pend_d: e.alloc_u32_zeroed(1)?,
            last_pred_d: e.alloc_u32_zeroed(1)?,
            pos_ctr: e.htod_i32(&[0])?,
            pos_start_d: e.htod_i32(&[0])?,
            ring_d: e.alloc_u32_zeroed(m_rounds * (k + 1) + 1)?,
            acc_d: e.alloc_u32_zeroed(2)?,
            m_rounds,
            k,
        })
    }

    /// Drain the ring after a burst (THE one host sync per M rounds): returns the committed
    /// tokens in order and resets nothing — the caller zeroes the ring count for the next
    /// burst via `e.set_u32_one(&mut ring_d, 0)`.
    pub fn drain_ring(&self, e: &Engine) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let h = e.dtoh_u32(&self.ring_d)?;
        let cnt = (h[0] as usize).min(self.ring_d.len() - 1);
        Ok(h[1..1 + cnt].to_vec())
    }
}

/// Per-layer `kvl.len_d` device-pointer table (+ the position counter appended when
/// `pos_ctr` is given) for `spec_rollback_stream`. Pointers are stable for the cache's
/// lifetime; 0 marks layers without KV (linear-attention / KV-shared).
pub fn kv_len_ptr_table(
    e: &Engine,
    cache: &Cache,
    pos_ctr: Option<&CudaSlice<i32>>,
) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
    kv_len_ptr_table_range(e, cache, 0..cache.kv.len(), pos_ctr)
}

/// Stage-local twin of [`kv_len_ptr_table`]. Only pointers owned by `layers` enter the table,
/// so a reconcile kernel launched through a PP stage's engine never dereferences another
/// device's `len_d`. The returned table is dense over the requested range; callers pass
/// `layers.len()` to the matching kernel.
pub fn kv_len_ptr_table_range(
    e: &Engine,
    cache: &Cache,
    layers: Range<usize>,
    pos_ctr: Option<&CudaSlice<i32>>,
) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
    use cudarc::driver::DevicePtr;
    assert!(layers.start <= layers.end && layers.end <= cache.kv.len());
    let mut ptrs: Vec<u64> = cache
        .kv
        .get(layers)
        .expect("validated KV layer range")
        .iter()
        .map(|kv| match kv.as_ref() {
            Some(kvl) => {
                let __s_g = e.stream();
                let (p, _g) = kvl.len_d.device_ptr(&__s_g);
                p
            }
            None => 0u64,
        })
        .collect();
    if let Some(pc) = pos_ctr {
        let __s_g = e.stream();
        let (p, _g) = pc.device_ptr(&__s_g);
        ptrs.push(p);
    }
    e.htod_u64(&ptrs)
}
