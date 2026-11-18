//! Per-layer last-row tap for prefill-against-decode bisects (memra#1203).
//!
//! A research binary arms the tap around one call (`forward_last` over a prefix, or the one
//! `decode_step` at the row it studies) and takes the recorded rows back afterwards. Each
//! record is the LAST token row of one stage of one layer, read back to the host as f32.
//!
//! No environment variable: only code that calls [`arm`] turns it on. Disarmed, a tap site
//! costs one relaxed atomic load. Armed, it adds a device-to-host copy per stage, which
//! synchronizes the stream but does not change any kernel or its arithmetic.

use cudarc::driver::CudaSlice;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// One recorded stage: `name` (`mix`, `x1`, `ffn`), the layer index and the last row.
#[derive(Debug, Clone)]
pub struct Tap {
    pub name: &'static str,
    pub layer: usize,
    pub row: Vec<f32>,
}

static ARMED: AtomicBool = AtomicBool::new(false);
static TAPS: Mutex<Vec<Tap>> = Mutex::new(Vec::new());
static LAYER: AtomicUsize = AtomicUsize::new(0);

/// The trunk walk names the layer it is in, so a tap inside a mixer that does not know its own
/// index (`emit` with `layer == usize::MAX`) records the right one.
#[inline]
pub(crate) fn set_layer(il: usize) {
    LAYER.store(il, Ordering::Relaxed);
}

/// Start recording. Clears anything a previous arming left behind.
pub fn arm() {
    TAPS.lock().expect("row tap poisoned").clear();
    ARMED.store(true, Ordering::Release);
}

/// Stop recording and return what was recorded, in emission order.
pub fn take() -> Vec<Tap> {
    ARMED.store(false, Ordering::Release);
    std::mem::take(&mut *TAPS.lock().expect("row tap poisoned"))
}

#[inline]
pub fn armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Record the last of `rows` rows of width `width` held in `buf`. `layer == usize::MAX` takes the
/// layer the walk last named through [`set_layer`].
pub(crate) fn emit(
    e: &crate::Engine,
    name: &'static str,
    layer: usize,
    buf: &CudaSlice<f32>,
    rows: usize,
    width: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if !armed() {
        return Ok(());
    }
    let layer = if layer == usize::MAX {
        LAYER.load(Ordering::Relaxed)
    } else {
        layer
    };
    let view = e.view(buf, rows * width);
    let last = view.slice((rows - 1) * width..rows * width);
    let mut host = e.uninit(width)?;
    e.copy_view_into(&mut host, 0, &last, width)?;
    let row = e.dtoh(&host)?;
    TAPS.lock()
        .expect("row tap poisoned")
        .push(Tap { name, layer, row });
    Ok(())
}

static KV_ROUNDTRIP: AtomicBool = AtomicBool::new(false);

/// Research switch (memra#1203): while on, the prefill's full-attention layers round-trip their
/// post-RoPE K through the cache's q8_0 format and V through q5_1 (fp16 block scales) on the
/// host before attention, so an otherwise f32 prefill sees exactly the precision the decode's
/// quantized cache holds. Only a research binary turns it on.
pub fn set_kv_roundtrip(on: bool) {
    KV_ROUNDTRIP.store(on, Ordering::Release);
}

pub(crate) fn kv_roundtrip() -> bool {
    KV_ROUNDTRIP.load(Ordering::Relaxed)
}

fn f16_round(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let e = x.abs().log2().floor();
    let step = (e - 10.0).exp2();
    (x / step).round() * step
}

/// q8_0: blocks of 32, d = amax / 127 (fp16), q = round(x / d).
pub(crate) fn q8_0_roundtrip(v: &mut [f32]) {
    for b in v.chunks_mut(32) {
        let amax = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = f16_round(amax / 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for x in b.iter_mut() {
            *x = (*x * id).round().clamp(-127.0, 127.0) * d;
        }
    }
}

/// q5_1: blocks of 32, d = (max - min) / 31 and m = min (both fp16), q = trunc((x - m) / d + 0.5).
pub(crate) fn q5_1_roundtrip(v: &mut [f32]) {
    for b in v.chunks_mut(32) {
        let mn = b.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let d = f16_round((mx - mn) / 31.0);
        let m = f16_round(mn);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for x in b.iter_mut() {
            let q = ((*x - mn) * id + 0.5).trunc().clamp(0.0, 31.0);
            *x = q * d + m;
        }
    }
}
