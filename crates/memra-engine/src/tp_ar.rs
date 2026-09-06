//! Small-message cross-rank all-reduce for TP decode (lane/tp-allreduce-20260906).
//!
//! The kernels and the reasoning live in `cu/tp_ar.cu`. Short version: the TP-2 join costs about
//! 500 us today because `tp_transport`'s default bounces every hop through host and
//! `Engine::dtoh` drains the stream, so each of the ~90 joins per token waits for that layer's
//! compute, twice. The link on the served pair is NV18, eighteen NVLink links at 53.125 GB/s;
//! an 8 KB two-rank all-reduce belongs in single-digit microseconds.
//!
//! The primitive: each rank pushes its partial straight into the peer's staging buffer, and each
//! rank folds the buffer its peer wrote. Ordering is one cross-stream event per direction, the
//! same contract `tp_transport::PeerPullLink::publish` uses. No host boundary, no `synchronize`,
//! and nothing occupying the device while it waits.
//!
//! [`ArLink::broadcast`] and [`ArLink::all_gather`] are the same push under different offsets, and
//! they matter more than the reduce for the walk as it stands: the glm5 TP MLA layer moves its
//! bytes in three PURE-MOVEMENT hops (fan out `h` and the positions, all-gather the head parts,
//! concat the column-parallel `wo` parts back onto root), which is why the current arm is
//! byte-identical to the unsharded walk by construction. Replacing only the transport keeps that
//! property exactly: the same bytes arrive in the same places, they just stop crossing PCIe and
//! stop draining a stream on the way.
use crate::Engine;
use cudarc::driver::{CudaEvent, CudaSlice, DevicePtr, DevicePtrMut};
use std::os::raw::c_void;

unsafe extern "C" {
    /// Push `n` f32 from `src` into the PEER's `peer_stage`. Enqueued on the caller's stream.
    pub fn memra_tp_ar_push(
        src: *const f32,
        peer_stage: *mut f32,
        n: i64,
        stream: *mut c_void,
    ) -> i32;
    /// `dst += stage`, `n` f32. Enqueued on the caller's stream, which must already be ordered
    /// after the peer's push.
    pub fn memra_tp_ar_fold(dst: *mut f32, stage: *const f32, n: i64, stream: *mut c_void) -> i32;
}

/// Per-rank all-reduce state. `stage[r]` lives on rank `r` and is written by its peer;
/// `pushed[r]` is recorded on rank `r`'s stream after its push and awaited by the peer.
///
/// LIFETIME, and it bites: `all_reduce` returns with work still enqueued on BOTH ranks. Dropping
/// the link while either rank's fold is in flight hands its staging buffer back to the
/// stream-ordered allocator, which will hand the same memory to the next allocation while the
/// pending fold is still writing into it. Every rank must be drained before the link goes away.
/// Found the hard way 2026-09-06: the gate read only rank 0 after a repeat call, and the next
/// size in the sweep came back with rank 1's staging garbage in it.
pub struct ArLink {
    stage: Vec<CudaSlice<f32>>,
    pushed: Vec<CudaEvent>,
    n: usize,
}

impl ArLink {
    /// Allocate for `engines.len()` ranks at a fixed element count. `engines[r]` is rank `r`, and
    /// peer access must already be granted both ways (`tp::grant_peer_access`).
    ///
    /// TWO DEVICES, NOT NEGOTIABLE. The other TP arms can be exercised by the two-context
    /// same-device emulation because their bytes go through host or through `cudaMemcpyPeer`,
    /// which handles cross-context copies on one card. This primitive dereferences the peer's
    /// pointer INSIDE a kernel, and two contexts on the same device neither share an address
    /// space nor can grant peer access to each other (`cudaDeviceCanAccessPeer(d, d)` is false),
    /// so that store is undefined there. It read correctly at some sizes and returned the local
    /// partial at others (2026-09-06), which is exactly what an undefined address does. The
    /// constructor refuses two engines on one ordinal rather than let a gate pass on an accident.
    pub fn new(engines: &[&Engine], n: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if engines.len() != 2 {
            return Err("tp all-reduce: this arm is two ranks".into());
        }
        if engines[0].ctx().ordinal() == engines[1].ctx().ordinal() {
            return Err(format!(
                "tp all-reduce needs two DEVICES; both engines are on ordinal {} (see the \
                 constructor's note: a peer store across two contexts on one card is undefined)",
                engines[0].ctx().ordinal()
            )
            .into());
        }
        if n == 0 {
            return Err("tp all-reduce needs a non-zero element count".into());
        }
        let mut stage = Vec::with_capacity(2);
        let mut pushed = Vec::with_capacity(2);
        for e in engines {
            let _main = e.gpu.enter_main()?;
            stage.push(e.zeros(n)?);
            pushed.push(e.ctx().new_event(None)?);
        }
        Ok(Self { stage, pushed, n })
    }

    pub fn ranks(&self) -> usize {
        self.stage.len()
    }

    /// Broadcast rank `from`'s `src` to every other rank's `dst`. Pure movement, so the result is
    /// byte-identical to the host-bounce fan-out it replaces: the same bytes arrive, they just do
    /// not cross the PCIe boundary or drain a stream on the way.
    pub fn broadcast(
        &mut self,
        engines: &[&Engine],
        from: usize,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if engines.len() != 2 || from > 1 {
            return Err("tp broadcast: this arm is two ranks".into());
        }
        if n > self.n {
            return Err(format!("tp broadcast: {n} exceeds the link's {} elements", self.n).into());
        }
        let to = 1 - from;
        let dst_ptr = {
            let e = engines[to];
            let _main = e.gpu.enter_main()?;
            let s = e.stream();
            dst.device_ptr_mut(&s).0 as *mut f32
        };
        {
            let e = engines[from];
            let _main = e.gpu.enter_main()?;
            let s = e.stream();
            let src_ptr = src.device_ptr(&s).0 as *const f32;
            // SAFETY: peer access is granted both ways, `dst` holds at least `n` floats on the
            // peer, and `src` at least `n` here.
            let rc = unsafe {
                memra_tp_ar_push(src_ptr, dst_ptr, n as i64, s.cu_stream() as *mut c_void)
            };
            if rc != 0 {
                return Err(format!("memra_tp_ar_push rc {rc}").into());
            }
            self.pushed[from].record(&s)?;
        }
        {
            let e = engines[to];
            let _main = e.gpu.enter_main()?;
            e.stream().wait(&self.pushed[from])?;
        }
        Ok(())
    }

    /// All-gather: rank `r` holds `part[r]` of `span` floats, and every rank ends with the ranks'
    /// parts concatenated in rank order into its own `full`. Pure movement and therefore
    /// byte-identical to the host-bounce gather it replaces.
    ///
    /// Each rank writes its OWN part locally and pushes it into the peer's `full` at the same
    /// offset, so the two directions never touch the same bytes.
    pub fn all_gather(
        &mut self,
        engines: &[&Engine],
        parts: &[&CudaSlice<f32>],
        full: &mut [&mut CudaSlice<f32>],
        span: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if engines.len() != 2 || parts.len() != 2 || full.len() != 2 {
            return Err("tp all-gather: this arm is two ranks".into());
        }
        let full_ptr: Vec<*mut f32> = {
            let mut v = Vec::with_capacity(2);
            for (r, e) in engines.iter().enumerate() {
                let _main = e.gpu.enter_main()?;
                let s = e.stream();
                v.push(full[r].device_ptr_mut(&s).0 as *mut f32);
            }
            v
        };
        for (from, to) in [(0usize, 1usize), (1, 0)] {
            let e = engines[from];
            let _main = e.gpu.enter_main()?;
            let s = e.stream();
            let src = parts[from].device_ptr(&s).0 as *const f32;
            // SAFETY: both destinations hold `2 * span` floats and the offset is `from * span`,
            // so each push writes inside its own rank-slot; `src` holds `span`.
            let rc_local = unsafe {
                memra_tp_ar_push(
                    src,
                    full_ptr[from].add(from * span),
                    span as i64,
                    s.cu_stream() as *mut c_void,
                )
            };
            if rc_local != 0 {
                return Err(format!("memra_tp_ar_push (local slot) rc {rc_local}").into());
            }
            let rc = unsafe {
                memra_tp_ar_push(
                    src,
                    full_ptr[to].add(from * span),
                    span as i64,
                    s.cu_stream() as *mut c_void,
                )
            };
            if rc != 0 {
                return Err(format!("memra_tp_ar_push (peer slot) rc {rc}").into());
            }
            self.pushed[from].record(&s)?;
        }
        for (r, peer) in [(0usize, 1usize), (1, 0)] {
            let e = engines[r];
            let _main = e.gpu.enter_main()?;
            e.stream().wait(&self.pushed[peer])?;
        }
        Ok(())
    }

    /// One all-reduce over `x[r]`, `x[r]` living on `engines[r]`. Every rank ends holding the
    /// elementwise sum.
    ///
    /// Order is the safety argument: both pushes are enqueued and both events recorded before
    /// either fold waits, so no stream can be waiting on an event whose recording kernel has not
    /// been submitted. Nothing here drains a stream, so the two ranks stay concurrent.
    pub fn all_reduce(
        &mut self,
        engines: &[&Engine],
        x: &mut [&mut CudaSlice<f32>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if engines.len() != 2 || x.len() != 2 {
            return Err("tp all-reduce: this arm is two ranks".into());
        }
        // Resolve every staging address under ITS OWN rank's context BEFORE any push enters a
        // context of its own. Reading a peer's `device_ptr` while another context is current
        // resolved to the wrong address for the second link allocated in a process: the gate saw
        // rank 1 keep its own partial at n=1024 whenever a smaller link had been built first,
        // and only then (2026-09-06).
        let stage_addr: Vec<*mut f32> = {
            let mut v = Vec::with_capacity(2);
            for (r, e) in engines.iter().enumerate() {
                let _main = e.gpu.enter_main()?;
                let s = e.stream();
                v.push(self.stage[r].device_ptr(&s).0 as *mut f32);
            }
            v
        };
        for (from, to) in [(0usize, 1usize), (1, 0)] {
            let e = engines[from];
            let _main = e.gpu.enter_main()?;
            let s = e.stream();
            let src = x[from].device_ptr(&s).0 as *const f32;
            let stage_ptr = stage_addr[to];
            // SAFETY: peer access is granted between the two devices, so the peer's staging
            // pointer is dereferenceable from this context, and it holds `n` floats. `src` is
            // rank `from`'s own buffer of at least `n`.
            let rc = unsafe {
                memra_tp_ar_push(src, stage_ptr, self.n as i64, s.cu_stream() as *mut c_void)
            };
            if rc != 0 {
                return Err(format!("memra_tp_ar_push rc {rc}").into());
            }
            self.pushed[from].record(&s)?;
        }
        for (r, peer) in [(0usize, 1usize), (1, 0)] {
            let e = engines[r];
            let _main = e.gpu.enter_main()?;
            let s = e.stream();
            s.wait(&self.pushed[peer])?;
            let dst = x[r].device_ptr_mut(&s).0 as *mut f32;
            let stage = self.stage[r].device_ptr(&s).0 as *const f32;
            // SAFETY: both pointers are rank `r`'s own allocations of at least `n` floats, and
            // the stream is ordered after the peer's push by the wait above.
            let rc = unsafe {
                memra_tp_ar_fold(dst, stage, self.n as i64, s.cu_stream() as *mut c_void)
            };
            if rc != 0 {
                return Err(format!("memra_tp_ar_fold rc {rc}").into());
            }
        }
        Ok(())
    }
}
