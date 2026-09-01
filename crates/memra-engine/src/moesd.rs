//! Diagnostic-only expert-union capture for the MoESD target-efficiency harness.
//!
//! The timed forward never enables this collector. The harness rolls its caches back and
//! replays the same target step with capture enabled, so route D2H cannot contaminate T_T.

use crate::Engine;
use cudarc::driver::CudaSlice;
use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct MoesdLayerUnion {
    pub id: u16,
    pub union_size: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub assignments: usize,
}

#[derive(Default)]
struct LayerCapture {
    experts: HashSet<u32>,
    n_expert: usize,
    n_used: usize,
    assignments: usize,
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
static CAPTURE: Mutex<Option<BTreeMap<u16, LayerCapture>>> = Mutex::new(None);

pub fn begin_capture() -> Result<(), Box<dyn std::error::Error>> {
    let mut capture = CAPTURE
        .lock()
        .map_err(|_| "MoESD capture lock is poisoned")?;
    if ACTIVE.load(Ordering::Relaxed) || capture.is_some() {
        return Err("MoESD capture is already active".into());
    }
    *capture = Some(BTreeMap::new());
    ACTIVE.store(true, Ordering::Release);
    Ok(())
}

pub fn finish_capture() -> Result<Vec<MoesdLayerUnion>, Box<dyn std::error::Error>> {
    if !ACTIVE.swap(false, Ordering::AcqRel) {
        return Err("MoESD capture is not active".into());
    }
    let mut capture = CAPTURE
        .lock()
        .map_err(|_| "MoESD capture lock is poisoned")?;
    let layers = capture.take().ok_or("MoESD capture state is missing")?;
    Ok(layers
        .into_iter()
        .map(|(id, layer)| MoesdLayerUnion {
            id,
            union_size: layer.experts.len(),
            n_expert: layer.n_expert,
            n_used: layer.n_used,
            assignments: layer.assignments,
        })
        .collect())
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
pub(crate) fn record_host_routes(
    il: u16,
    n_expert: usize,
    n_used: usize,
    selected: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    if !ACTIVE.load(Ordering::Acquire) {
        return Ok(());
    }
    if n_used == 0 {
        return Err(format!("MoESD layer {il} has router n_used=0").into());
    }
    if selected.len() % n_used != 0 {
        return Err(format!(
            "MoESD route shape mismatch at layer {il}: {} selections is not divisible by {n_used}",
            selected.len(),
        )
        .into());
    }
    if let Some(expert) = selected.iter().find(|&&expert| expert as usize >= n_expert) {
        return Err(format!(
            "MoESD layer {il} selected out-of-range expert {expert} from bank {n_expert}",
        )
        .into());
    }
    let mut capture = CAPTURE
        .lock()
        .map_err(|_| "MoESD capture lock is poisoned")?;
    let layers = capture.as_mut().ok_or("MoESD capture state is missing")?;
    let layer = layers.entry(il).or_default();
    if layer.assignments != 0 && (layer.n_expert != n_expert || layer.n_used != n_used) {
        return Err(format!("MoESD route metadata changed within layer {il}").into());
    }
    layer.n_expert = n_expert;
    layer.n_used = n_used;
    layer.assignments += selected.len();
    layer.experts.extend(selected.iter().copied());
    Ok(())
}

pub(crate) fn record_device_routes(
    e: &Engine,
    il: u16,
    n_expert: usize,
    n_used: usize,
    selected: &CudaSlice<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !ACTIVE.load(Ordering::Acquire) {
        return Ok(());
    }
    let selected: Vec<u32> = e
        .dtoh_i32(selected)?
        .into_iter()
        .map(|expert| expert as u32)
        .collect();
    record_host_routes(il, n_expert, n_used, &selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_capture_counts_distinct_experts_and_assignments() {
        begin_capture().unwrap();
        record_host_routes(7, 16, 2, &[1, 2, 2, 3, 3, 4]).unwrap();
        let layers = finish_capture().unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].id, 7);
        assert_eq!(layers[0].union_size, 4);
        assert_eq!(layers[0].n_expert, 16);
        assert_eq!(layers[0].n_used, 2);
        assert_eq!(layers[0].assignments, 6);
    }
}
