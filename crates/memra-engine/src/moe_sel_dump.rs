//! `MEMRA_MOE_SEL_DUMP=<path>` (default OFF): the per-token routed-expert selection dump
//! behind the expert co-activation question (lane/moe-coactivation-20260902, owner question
//! 2026-09-02: can GLM-5.3-Flash's 288 routed experts per MoE layer be split across two
//! cards by co-activation, always-active experts replicated on both, so a token's 8 selected
//! experts rarely cross cards?). The host tool `tools/moe_coact.py` reads this file.
//!
//! WHY A THIRD ROUTE TAP. `MEMRA_MOE_TRACE` / `MEMRA_MOE_WEIGHT_TRACE` are text, one line
//! per (layer, forward), and they DIVERT dispatch: setting either one flips `observe_routes`
//! in `moe_ffn_inner`, which forces the host-routed path and refuses the device-routed
//! walks by name. That is right for a placement mint that must never miss rows, and wrong
//! for this question, which wants the selection the SERVED arms actually make, decode
//! device-table arm included. `moesd` is a harness-only window capture and `glm5_sel_ledger`
//! a gate instrument; neither writes a file. So this tap:
//!
//!   * changes NO dispatch decision (it is in no `observe_routes` conjunct);
//!   * rides the host readbacks the walks already perform (`trace_moe_routes`), and on the
//!     device-routed single-device arms (`vrows_dev` door D, and the Step-3.7 resident
//!     `moe_ffn_sigmoid_dev`) reads the freshly selected `[sel, w]` back with one extra
//!     DtoH per layer-call, DIAGNOSTIC ONLY, never in a timed arm;
//!   * refuses by name on the device-routed TP walks whose selection never returns to the
//!     host (the same law the text taps hold: a dump missing whole layers would poison the
//!     partition built on it).
//!
//! ZERO COST WHEN UNSET: every entry point is one `OnceLock` read (`armed()`), no env scan
//! per call. The record path is buffered (1 MiB `BufWriter`, flushed at most once per
//! second from the record path itself, so a SIGKILL loses at most the last second).
//!
//! FILE FORMAT `memra-moe-sel-v1`, little-endian, no header, records back to back, one
//! record per (routed token, MoE layer), appended in the order the forward makes them:
//!
//! ```text
//! u8  layer          the MoE layer index `il`
//! u8  n_sel          number of selected experts for this token (glm5_next: 8)
//! n_sel x { u16 expert_id, f32 routing_weight }   slot order, as the router emitted them
//! ```
//!
//! A glm5_next record is 2 + 8 * 6 = 50 bytes. Prime (grouped prefill, one call per chunk
//! per layer) and decode (t = 1) records are indistinguishable by design: the question is
//! about the routing distribution, and the tool reports pooled and per-layer statistics.
//! Split prime from decode with two boots (two files) when that distinction matters.

use crate::Engine;
use cudarc::driver::CudaSlice;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// The door. Read once per process (`OnceLock`); the value is the dump path.
pub const ENV: &str = "MEMRA_MOE_SEL_DUMP";

/// Records appended since process start: the engagement receipt. A run whose dump file
/// exists but whose counter is 0 recorded nothing (every arm declined or the model has no
/// routed MoE), and the box note must say so rather than analyze an empty file.
pub static RECORDS: AtomicU64 = AtomicU64::new(0);

struct Sink {
    w: std::io::BufWriter<std::fs::File>,
    last_flush: Instant,
}

static SINK: OnceLock<Option<Mutex<Sink>>> = OnceLock::new();

fn sink() -> Option<&'static Mutex<Sink>> {
    SINK.get_or_init(|| {
        let path = std::env::var_os(ENV)?;
        if path.is_empty() {
            eprintln!("[moe-sel-dump] {ENV} is set but empty: dump disarmed");
            return None;
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(f) => {
                eprintln!(
                    "[moe-sel-dump] armed path={} format=memra-moe-sel-v1 (u8 layer, u8 n_sel, \
                     n_sel x (u16 expert, f32 weight), LE; one record per routed token per MoE \
                     layer, prime and decode)",
                    path.to_string_lossy()
                );
                Some(Mutex::new(Sink {
                    w: std::io::BufWriter::with_capacity(1 << 20, f),
                    last_flush: Instant::now(),
                }))
            }
            Err(err) => {
                eprintln!(
                    "[moe-sel-dump] REFUSED: cannot open {} for append: {err} (dump disarmed; \
                     nothing is recorded)",
                    path.to_string_lossy()
                );
                None
            }
        }
    })
    .as_ref()
}

/// Whether the dump is armed. One `OnceLock` read; the only cost an unset process pays.
pub fn armed() -> bool {
    sink().is_some()
}

/// Append `t = sel.len() / n_used` records from a host-visible selection. `sel`/`w` are the
/// token-major `[t * n_used]` rows every host-routed walk already reads back. An EMPTY
/// selection is a no-op by contract: the device-table arm of `moe_ffn_inner` hands
/// `trace_moe_routes` an empty host twin and records itself through [`record_device`].
pub(crate) fn record_host(
    il: u16,
    t: usize,
    sel: &[u32],
    w: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(s) = sink() else {
        return Ok(());
    };
    if sel.is_empty() {
        return Ok(());
    }
    if t == 0 || !sel.len().is_multiple_of(t) {
        return Err(format!(
            "{ENV}: layer {il} selection of {} entries is not token-major over t={t}",
            sel.len()
        )
        .into());
    }
    let n_used = sel.len() / t;
    if n_used == 0 || n_used > u8::MAX as usize || il > u8::MAX as u16 {
        return Err(format!(
            "{ENV}: record geometry outside the v1 format (layer {il}, n_sel {n_used}; both \
             must fit a u8)"
        )
        .into());
    }
    if w.len() < sel.len() {
        return Err(format!(
            "{ENV}: layer {il} has {} weights for {} selections",
            w.len(),
            sel.len()
        )
        .into());
    }
    let mut buf = Vec::with_capacity(t * (2 + 6 * n_used));
    for tok in 0..t {
        buf.push(il as u8);
        buf.push(n_used as u8);
        for j in 0..n_used {
            let p = tok * n_used + j;
            let ex = sel[p];
            if ex > u16::MAX as u32 {
                return Err(format!("{ENV}: layer {il} expert id {ex} does not fit a u16").into());
            }
            buf.extend_from_slice(&(ex as u16).to_le_bytes());
            buf.extend_from_slice(&w[p].to_le_bytes());
        }
    }
    let mut g = s.lock().map_err(|_| format!("{ENV}: sink lock poisoned"))?;
    g.w.write_all(&buf)?;
    if g.last_flush.elapsed().as_secs() >= 1 {
        g.w.flush()?;
        g.last_flush = Instant::now();
    }
    RECORDS.fetch_add(t as u64, Ordering::Relaxed);
    Ok(())
}

/// The device-routed twin: read the router's own `[sel, w]` rows back (one DtoH pair plus
/// one stream sync per layer-call, diagnostic only) and append them. Both slices must be
/// exactly `t * n_used` long: the two callers hand over freshly allocated selections, and a
/// capacity buffer with an inactive tail would dump stale rows as tokens.
pub(crate) fn record_device(
    e: &Engine,
    il: u16,
    t: usize,
    n_used: usize,
    sel_d: &CudaSlice<i32>,
    w_d: &CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !armed() {
        return Ok(());
    }
    let n = t * n_used;
    if n == 0 || sel_d.len() != n || w_d.len() != n {
        return Err(format!(
            "{ENV}: layer {il} device selection is {} ids / {} weights, expected exactly \
             t*n_used = {n}",
            sel_d.len(),
            w_d.len()
        )
        .into());
    }
    let sel: Vec<u32> = e.dtoh_i32(sel_d)?.into_iter().map(|x| x as u32).collect();
    let w = e.dtoh(w_d)?;
    record_host(il, t, &sel, &w)
}

/// The refusal for walks whose selection never returns to the host and is not read back
/// here either (the device-routed TP walks). Same law as the text taps: an armed dump that
/// silently misses whole layers poisons every partition built on it.
pub(crate) fn refuse_device_only(walk: &str) -> Result<(), Box<dyn std::error::Error>> {
    if armed() {
        return Err(format!(
            "{ENV} cannot record {walk} (its selection never returns to the host and this dump \
             does not add a sync there); run the dump on the single-device walk, refused rather \
             than silently dropping rows"
        )
        .into());
    }
    Ok(())
}

/// Flush the buffered records to the file. The record path flushes on its own once per
/// second; this is for a caller that wants the file complete NOW (a gate reading the file
/// back in the same process).
pub fn flush() -> Result<(), Box<dyn std::error::Error>> {
    if let Some(s) = sink() {
        let mut g = s.lock().map_err(|_| format!("{ENV}: sink lock poisoned"))?;
        g.w.flush()?;
        g.last_flush = Instant::now();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The encoder is exercised through the same byte layout the tool decodes, without
    /// touching the process-global sink (the env is not set in the unit-test process, so
    /// `record_host` is a no-op there and the layout is asserted on a local encoding).
    fn encode(il: u8, sel: &[u16], w: &[f32]) -> Vec<u8> {
        let mut buf = vec![il, sel.len() as u8];
        for (ex, wt) in sel.iter().zip(w) {
            buf.extend_from_slice(&ex.to_le_bytes());
            buf.extend_from_slice(&wt.to_le_bytes());
        }
        buf
    }

    #[test]
    fn v1_record_is_2_plus_6_per_slot_little_endian() {
        let rec = encode(3, &[287, 0, 42], &[0.5, 0.25, 1.0]);
        assert_eq!(rec.len(), 2 + 3 * 6);
        assert_eq!(&rec[..2], &[3, 3]);
        assert_eq!(&rec[2..4], &287u16.to_le_bytes());
        assert_eq!(&rec[4..8], &0.5f32.to_le_bytes());
        assert_eq!(&rec[8..10], &0u16.to_le_bytes());
        assert_eq!(&rec[14..16], &42u16.to_le_bytes());
        assert_eq!(&rec[16..20], &1.0f32.to_le_bytes());
    }

    #[test]
    fn unset_env_records_nothing_and_costs_one_oncelock_read() {
        // The test process does not set MEMRA_MOE_SEL_DUMP; every entry point is a no-op.
        // A harness that armed it for the whole process is measuring something else.
        if std::env::var_os(super::ENV).is_some() {
            return;
        }
        assert!(!super::armed());
        super::record_host(0, 2, &[1, 2, 3, 4], &[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(super::RECORDS.load(std::sync::atomic::Ordering::Relaxed), 0);
        super::refuse_device_only("a device-routed walk").unwrap();
        super::flush().unwrap();
    }
}
