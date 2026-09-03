//! GATE-HARNESS SELECTION LEDGER for the glm5_next decode-graph door — never a serving flag.
//!
//! `MEMRA_GLM5_GRAPH_SEL_LEDGER=1` arms it. It exists to answer one question the token stream
//! alone cannot: does the DEVICE-table MoE arm (`vrows_t1_dev`, door `MEMRA_GLM5_DECODE_GRAPH`)
//! select the same experts, with the same routing weights, as the host oracle the eager walk
//! reads back — per layer, per token, bit-for-bit?
//!
//! Every other route tap in this engine (`moesd`, `MEMRA_MOE_WEIGHT_TRACE`, `hidden_trace`)
//! reads the selection back on the HOST, which is precisely what a capture region forbids —
//! which is why the decode-graph door refuses when any of them is armed. This one is
//! CAPTURE-LEGAL by construction:
//!
//!   * the device arm records with a `memcpy_dtod` into a PERSISTENT per-(device, layer) slot,
//!     which captures as a memcpy node and replays at a fixed address. No sync, no DtoH, no
//!     allocation inside the capture (the slots are pre-armed before the capture opens, by
//!     `HybridModel::glm5_capture_stage`);
//!   * the host arm records into a plain host vector, off the device entirely;
//!   * the gate drains the device slots with `drain_device` AFTER the token, outside capture.
//!
//! It changes no kernel, no operand and no launch order on either arm — the D2D copy is a pure
//! observation of a buffer the round already wrote. It stays OFF in every serving configuration
//! and carries no `docs/FLAGS.md` serving row for the same reason `MEMRA_GLM5_TP_GATE_RED` does
//! not: it is a gate instrument.

use crate::Engine;
use cudarc::driver::CudaSlice;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One layer's recorded selection, either arm.
#[derive(Clone, Debug, PartialEq)]
pub struct SelRow {
    pub dev: usize,
    pub layer: u16,
    pub sel: Vec<i32>,
    pub w: Vec<f32>,
}

#[derive(Default)]
struct Ledger {
    /// Persistent device slots, one per (device ordinal, layer). Addresses are baked into the
    /// captured graphs, so they are allocated ONCE and never reissued.
    slots: BTreeMap<(usize, u16), (CudaSlice<i32>, CudaSlice<f32>)>,
    /// What the host-oracle arm read back this run.
    host: Vec<SelRow>,
}

fn ledger() -> &'static Mutex<Ledger> {
    static L: OnceLock<Mutex<Ledger>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(Ledger::default()))
}

/// `MEMRA_GLM5_GRAPH_SEL_LEDGER=1`. Read per call: the gate arms it around one arm at a time.
pub fn armed() -> bool {
    std::env::var("MEMRA_GLM5_GRAPH_SEL_LEDGER").as_deref() == Ok("1")
}

/// Allocate this layer's device slot if it does not exist yet. MUST be called OUTSIDE a capture
/// region — an allocation inside one becomes a graph mem node whose address the next launch
/// recycles, and the ledger would read freed memory.
pub fn prearm(e: &Engine, layer: u16, n_used: usize) -> Res<()> {
    if !armed() {
        return Ok(());
    }
    let key = (e.ctx().ordinal(), layer);
    let mut l = ledger().lock().unwrap();
    // `entry` rather than contains_key + insert: the allocation is fallible, so the vacant arm is
    // written out instead of using `or_insert_with`.
    if let std::collections::btree_map::Entry::Vacant(slot) = l.slots.entry(key) {
        // Sentinel `-1` expert ids: a slot the device arm never wrote reads as an impossible
        // selection rather than as expert 0, so a wiring miss shows up as a mismatch instead of
        // an accidental match.
        slot.insert((e.htod_i32(&vec![-1i32; n_used])?, e.zeros(n_used)?));
    }
    Ok(())
}

/// DEVICE arm: copy the router's own `sel`/`w` into this layer's persistent slot. Capture-legal
/// (a memcpy node), replay-stable (fixed addresses). A layer with no pre-armed slot is skipped
/// rather than allocating one, so this can never introduce a mem node into a capture.
pub fn record_device(
    e: &Engine,
    layer: u16,
    sel_d: &CudaSlice<i32>,
    w_d: &CudaSlice<f32>,
) -> Res<()> {
    if !armed() {
        return Ok(());
    }
    let key = (e.ctx().ordinal(), layer);
    let mut l = ledger().lock().unwrap();
    let Some((sel_slot, w_slot)) = l.slots.get_mut(&key) else {
        return Ok(());
    };
    let n = sel_slot.len().min(sel_d.len());
    let mut sv = sel_slot.slice_mut(0..n);
    e.stream().memcpy_dtod(&sel_d.slice(0..n), &mut sv)?;
    let m = w_slot.len().min(w_d.len());
    let mut wv = w_slot.slice_mut(0..m);
    e.stream().memcpy_dtod(&w_d.slice(0..m), &mut wv)?;
    Ok(())
}

/// HOST arm: the selection the eager walk read back through the pinned stage.
pub fn record_host(dev: usize, layer: u16, sel: &[u32], w: &[f32]) {
    if !armed() {
        return;
    }
    ledger().lock().unwrap().host.push(SelRow {
        dev,
        layer,
        sel: sel.iter().map(|&s| s as i32).collect(),
        w: w.to_vec(),
    });
}

/// Drain the DEVICE slots for `e`'s device into host rows, ordered by layer. Call after the
/// token completes, never inside a capture.
pub fn drain_device(e: &Engine) -> Res<Vec<SelRow>> {
    let dev = e.ctx().ordinal();
    let l = ledger().lock().unwrap();
    let mut out = Vec::new();
    for ((d, layer), (sel_slot, w_slot)) in l.slots.iter() {
        if *d != dev {
            continue;
        }
        out.push(SelRow {
            dev: *d,
            layer: *layer,
            sel: e.dtoh_i32(sel_slot)?,
            w: e.dtoh(w_slot)?,
        });
    }
    Ok(out)
}

/// Take and clear the HOST rows recorded since the last call.
pub fn take_host() -> Vec<SelRow> {
    std::mem::take(&mut ledger().lock().unwrap().host)
}

/// Clear the host rows without reading them (between arms).
pub fn reset_host() {
    ledger().lock().unwrap().host.clear();
}
