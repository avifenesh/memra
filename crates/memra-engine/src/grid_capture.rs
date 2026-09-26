//! GRID CAPTURE INSIDE ONE PRIME CALL (WP-B day 44, `research/spill-b-20260919/DAY44.md` 1.2 (a)).
//!
//! A resumable session needs a checkpoint at a GDN grid point near its prompt end: the state a
//! cold prime of the same tokens has there (the grid law). Taking it as a prime STOP splits the
//! call in two, and the second call costs a whole trunk pass at small T (the rewind arm's TTFT
//! x2, DAY41 2.2). This module takes it INSIDE the call instead:
//!
//! - GDN layers: after the layer's chunked scan, one extra state-pass launch over the call's first
//!   `rel / 32` chunks writes the f32 state at the capture point (the same kernel on a prefix of
//!   the same K1 to K3 buffers: the state pass is sequential over chunks, so its state after chunk
//!   `k` does not depend on later chunks); the conv ring at the capture point is the ring update
//!   kernel over the call's first `rel` input rows (a copy of rows `rel - 3 .. rel`).
//! - Full-attention layers: nothing; the KV rows below the capture point are the call's rows.
//!
//! The capture is armed by the caller for ONE prime call on this thread (`arm`), read by the
//! prime walk (`begin_call`, `layer_rel`, `put_conv`, `put_ssm`), and collected after the call
//! (`take`). A call that does not reach every GDN layer through the per-sequence chunked scan
//! (the batched, varlen, pipelined, Hopper wgmma or sequential-scan paths) leaves the capture
//! incomplete, and `take` returns `None`: no checkpoint, never a partial one.

use std::cell::RefCell;

use cudarc::driver::CudaSlice;

/// GDN chunk grid (rows) the capture point must sit on; `Engine::gdn_chunk_size`.
fn grain() -> usize {
    crate::Engine::gdn_chunk_size()
}

struct Pending {
    /// Absolute capture position.
    pos: usize,
    /// Rows into the current call, once `begin_call` accepted it.
    rel: Option<usize>,
    conv: Vec<Option<CudaSlice<f32>>>,
    ssm: Vec<Option<CudaSlice<f32>>>,
    /// Layers that must be filled (the linear-attention layers of the cache).
    need: Vec<bool>,
    /// Filled from the live cache at a call boundary (`note_boundary`).
    boundary: bool,
    /// A call accepted the capture (its layers fill `conv` and `ssm`).
    accepted: bool,
}

thread_local! {
    static PENDING: RefCell<Option<Pending>> = const { RefCell::new(None) };
}

/// A completed capture: the recurrent planes of every linear-attention layer at `pos`.
pub struct GridCapture {
    pub pos: usize,
    pub conv: Vec<Option<CudaSlice<f32>>>,
    pub ssm: Vec<Option<CudaSlice<f32>>>,
}

/// The capture point rule (DAY44 1.2 (a)): the largest grid point `g` in `(base, base + t)` with
/// at least `floor` rows of the call before it and at least `floor` rows after it. `None` when
/// the call is too short to hold one.
pub fn capture_point(base: usize, t: usize, floor: usize, grain: usize) -> Option<usize> {
    let end = base + t;
    let hi = end.checked_sub(floor)?;
    let g = hi / grain * grain;
    (g >= base + floor && g > base && base.is_multiple_of(grain)).then_some(g)
}

/// Disarms on drop, so an error path that skips `take` never leaves a capture armed for the next
/// session's prime on this thread.
#[must_use = "the capture stays armed only while the guard lives"]
pub struct ArmGuard(());

impl Drop for ArmGuard {
    fn drop(&mut self) {
        disarm();
    }
}

/// Arm a capture at absolute `pos` for the next prime call on this thread. `need[il]` is true for
/// every linear-attention layer. Replaces any earlier arming. Collect with `take` while the guard
/// lives.
pub fn arm(pos: usize, need: Vec<bool>) -> ArmGuard {
    let n = need.len();
    PENDING.with(|p| {
        *p.borrow_mut() = Some(Pending {
            pos,
            rel: None,
            conv: (0..n).map(|_| None).collect(),
            ssm: (0..n).map(|_| None).collect(),
            need,
            boundary: false,
            accepted: false,
        })
    });
    ArmGuard(())
}

/// Disarm without collecting.
pub fn disarm() {
    PENDING.with(|p| *p.borrow_mut() = None);
}

/// True while a capture is armed on this thread (the prime graph and batched paths refuse it).
pub fn armed() -> bool {
    PENDING.with(|p| p.borrow().is_some())
}

/// Called by the prime walk once per call with the call's base and row count. Accepts the armed
/// capture for this call when its position lies inside it, on the grid, with `PRIME_MIN_T` rows on
/// each side; returns the rows into the call.
pub(crate) fn begin_call(base: usize, t: usize) -> Option<usize> {
    let floor = crate::hybrid_forward::PRIME_MIN_T;
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        let pending = p.as_mut()?;
        pending.rel = None;
        if pending.boundary || pending.accepted {
            return None;
        }
        let rel = pending.pos.checked_sub(base)?;
        let ok = pending.pos.is_multiple_of(grain())
            && rel.is_multiple_of(grain())
            && rel >= floor
            && t >= rel + floor;
        if ok {
            pending.rel = Some(rel);
            pending.accepted = true;
        }
        pending.rel
    })
}

/// At a prime call boundary: when the armed position is the cache's position and nothing was
/// captured inside a call, the live state IS the capture (a plain snapshot, no split).
pub(crate) fn note_boundary(e: &crate::Engine, cache: &crate::cache::Cache) {
    let want = PENDING.with(|p| {
        p.borrow()
            .as_ref()
            .is_some_and(|x| x.pos == cache.pos && !x.boundary && !x.accepted)
    });
    if !want {
        return;
    }
    let Ok(snap) = cache.snapshot(e) else { return };
    PENDING.with(|p| {
        if let Some(x) = p.borrow_mut().as_mut() {
            x.conv = snap.conv;
            x.ssm = snap.ssm;
            x.boundary = true;
        }
    });
}

/// The accepted capture's rows into the current call, if any.
pub(crate) fn layer_rel() -> Option<usize> {
    PENDING.with(|p| p.borrow().as_ref().and_then(|x| x.rel))
}

pub(crate) fn put_conv(il: usize, buf: CudaSlice<f32>) {
    PENDING.with(|p| {
        if let Some(x) = p.borrow_mut().as_mut()
            && il < x.conv.len()
        {
            x.conv[il] = Some(buf);
        }
    });
}

pub(crate) fn put_ssm(il: usize, buf: CudaSlice<f32>) {
    PENDING.with(|p| {
        if let Some(x) = p.borrow_mut().as_mut()
            && il < x.ssm.len()
        {
            x.ssm[il] = Some(buf);
        }
    });
}

/// Collect and disarm. `Some` only when the call accepted the capture and every needed layer
/// was filled.
pub fn take() -> Option<GridCapture> {
    let x = PENDING.with(|p| p.borrow_mut().take())?;
    if !x.accepted && !x.boundary {
        return None;
    }
    let complete = x
        .need
        .iter()
        .enumerate()
        .all(|(il, &need)| !need || (x.conv[il].is_some() && x.ssm[il].is_some()));
    complete.then_some(GridCapture {
        pos: x.pos,
        conv: x.conv,
        ssm: x.ssm,
    })
}

impl GridCapture {
    /// The capture as a `CacheSnapshot` of `cache` (the cache it was taken from, now past it):
    /// every full-attention layer's length is the capture position, the recurrent planes are the
    /// captured ones. Refuses planes a snapshot cannot rewind (TP KV, latent), as
    /// `maybe_plain_checkpoint` does.
    pub fn into_snapshot(
        self,
        cache: &crate::cache::Cache,
    ) -> Result<crate::cache::CacheSnapshot, String> {
        if cache.tp_kv.iter().any(Option::is_some) || cache.latent.iter().any(Option::is_some) {
            return Err(
                "grid capture: TP KV or latent planes are not rewound by a snapshot".into(),
            );
        }
        let n = cache.kv.len();
        if self.conv.len() != n || self.ssm.len() != n || cache.recur.len() != n {
            return Err("grid capture: layer count differs from the cache".into());
        }
        if cache.pos < self.pos {
            return Err(format!(
                "grid capture at {} is past the cache position {}",
                self.pos, cache.pos
            ));
        }
        // A layer the prime walk does not advance (the MTP head's trunk-cache slot) keeps its
        // own length, below the capture point.
        let kv_len = cache
            .kv
            .iter()
            .map(|kv| kv.as_ref().map(|kv| kv.len.min(self.pos)))
            .collect();
        let tp_kv_len = (0..n).map(|_| None).collect();
        Ok(crate::cache::CacheSnapshot {
            kv_len,
            tp_kv_len,
            conv: self.conv,
            ssm: self.ssm,
            pos: self.pos,
        })
    }
}

/// The linear-attention layers of `cache` (the layers a capture must fill).
pub fn needed_layers(cache: &crate::cache::Cache) -> Vec<bool> {
    cache.recur.iter().map(Option::is_some).collect()
}

#[cfg(test)]
mod tests {
    use super::capture_point;

    #[test]
    fn the_capture_point_is_the_last_grid_point_with_a_floor_on_each_side() {
        // A resume from 6112 priming 128 rows (to 6240): the last grid point at or below 6224.
        assert_eq!(capture_point(6112, 128, 16, 32), Some(6208));
        // A cold prime of 6144 rows from 0: 6112 (6128 floored to the grid).
        assert_eq!(capture_point(0, 6144, 16, 32), Some(6112));
        // Too short: 40 rows from 6112 hold no grid point with 16 rows on each side.
        assert_eq!(capture_point(6112, 40, 16, 32), None);
        // 48 rows from 6112: 6144 has 32 rows before and 16 after.
        assert_eq!(capture_point(6112, 48, 16, 32), Some(6144));
        // A call that does not start on the grid holds no grid-law capture.
        assert_eq!(capture_point(6113, 128, 16, 32), None);
    }
}
