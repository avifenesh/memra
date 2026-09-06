//! Batched decode step — B sequences share one fused pass (ARCHITECTURE-H100.md §3 B2').
//!
//! The bandwidth thesis: decode is weight-stream-bound, so every projection at m=B rows
//! amortizes one weight read across B sequences. Row-parallel ops (norm/rope/quantize/
//! activation) batch trivially — they are the SAME kernels prefill already runs at T rows.
//! Only truly per-sequence state stays in a loop: KV append + fa_decode over each cache,
//! and the GDN/conv recurrent step (v1: per-seq loop via the existing single-seq path;
//! a blockIdx.z-batched GDN state kernel is the v2 fusion).
//!
//! EXACTNESS CONTRACT (the law this module lives under):
//! - B == 1 must be BIT-IDENTICAL to `decode_step_h` (gate: decode-batch-gate).
//! - 2 <= B <= 8: each row rides the m=2..9 verify-tier mmvq kernels, which are per-row
//!   bit-identical to m=1 (the spec-exactness machinery decode_step_t relies on). Each
//!   sequence's token stream must equal its isolated single-seq run (worker.rs contract:
//!   "byte-identical to isolated").
//! - 9 <= B <= 16 (the EXACT-16 tier, inc3 2026-08-01): admitted iff
//!   `decode_batch_exact16_ok` — every matmul rides the b16 batched-mmvq class
//!   (bit-identical per (token,row) to m=1; Q8_0 needs the q8rp mirror) under a
//!   verify_exact scope that disables the m>=16 GEMM/MMQ arms. gate2 bit-strength
//!   PASS at B=12/16 (research/batched-tick-inc3-20260801). Refused otherwise.
//! - B > 16 crosses into GEMM/dp4a-tail numeric configs with NO exact kernel class —
//!   refused (MEMRA_DECODE_BATCH_CAP stays a measurement door).
//!
//! v1 scope: the hybrid (Qwen3.5-class) non-gemma4 trunk. Fused m=1 micro-launches
//! (fused3 QKV, cross-layer add+norm+q8 chain) are NOT used — the unfused sequence is
//! bit-identical (kernel_check: add_rms_norm == add;rms_norm; _q8_1 == +quantize_q8_1)
//! and keeps the batched path simple. Batched fusions are tuning work, not correctness.

use crate::Engine;
use crate::cache::Cache;
use crate::hybrid::{HybridModel, Mixer};
use cudarc::driver::{CudaEvent, CudaSlice};
use std::collections::VecDeque;
use std::sync::Arc;

type DualPpCudaSpan = Option<(CudaEvent, CudaEvent)>;

/// One disjoint request wave moving through the PP3/PP4 anti-diagonal schedule. The activation
/// itself lives in `PpNRt`'s persistent boundary slot; this host state carries only ownership of
/// the request caches and the slot selected by the preceding stage.
struct PpDecodeWave<'slice, 'cache> {
    row_lo: usize,
    tokens: &'slice [u32],
    caches: &'slice mut [&'cache mut Cache],
    phase_last: std::time::Instant,
    #[allow(clippy::type_complexity)]
    // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    result: Option<(Vec<Vec<f32>>, Vec<Option<u32>>)>,
    committed: bool,
}

impl Drop for PpDecodeWave<'_, '_> {
    fn drop(&mut self) {
        if !self.committed {
            for cache in self.caches.iter_mut() {
                cache.mark_tainted();
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PpWaveTransfer {
    boundary: usize,
    wave: usize,
    slot: usize,
}

#[derive(Debug)]
enum PpWaveMessage {
    Transfer(PpWaveTransfer),
    WorkerError {
        boundary: usize,
        wave: usize,
        error: String,
    },
}

struct PpWaveIncoming {
    boundary: usize,
    transfers: std::sync::mpsc::Receiver<PpWaveMessage>,
    acknowledgements: std::sync::mpsc::Sender<PpWaveTransfer>,
}

impl PpWaveIncoming {
    fn receive(&self, expected_wave: usize) -> Result<PpWaveTransfer, String> {
        match self.transfers.recv() {
            Ok(PpWaveMessage::Transfer(transfer)) => {
                if transfer.boundary != self.boundary || transfer.wave != expected_wave {
                    return Err(format!(
                        "PP wave boundary {} expected wave {expected_wave}, got boundary {} wave {} slot {}",
                        self.boundary, transfer.boundary, transfer.wave, transfer.slot,
                    ));
                }
                if transfer.slot >= 2 {
                    return Err(format!(
                        "PP wave boundary {} wave {expected_wave} carried invalid slot {}",
                        self.boundary, transfer.slot,
                    ));
                }
                Ok(transfer)
            }
            Ok(PpWaveMessage::WorkerError {
                boundary,
                wave,
                error,
            }) => {
                if boundary != self.boundary {
                    return Err(format!(
                        "PP wave boundary {} received worker failure for boundary {boundary} wave {wave}: {error}",
                        self.boundary,
                    ));
                }
                Err(format!(
                    "PP wave boundary {boundary} upstream worker failed at wave {wave} while receiver expected wave {expected_wave}: {error}"
                ))
            }
            Err(_) => Err(format!(
                "PP wave boundary {} transfer channel closed before wave {expected_wave}",
                self.boundary,
            )),
        }
    }

    fn acknowledge(&self, transfer: PpWaveTransfer) -> Result<(), String> {
        if transfer.boundary != self.boundary {
            return Err(format!(
                "PP wave acknowledgement boundary mismatch: receiver {} transfer {}",
                self.boundary, transfer.boundary,
            ));
        }
        self.acknowledgements.send(transfer).map_err(|_| {
            format!(
                "PP wave boundary {} acknowledgement channel closed at wave {} slot {}",
                self.boundary, transfer.wave, transfer.slot,
            )
        })
    }
}

struct PpWaveOutgoing {
    boundary: usize,
    transfers: std::sync::mpsc::Sender<PpWaveMessage>,
    acknowledgements: std::sync::mpsc::Receiver<PpWaveTransfer>,
    slot_owner: [Option<PpWaveTransfer>; 2],
    pending: VecDeque<PpWaveTransfer>,
    next_slot: Option<usize>,
    next_wave: usize,
}

impl PpWaveOutgoing {
    fn new(
        boundary: usize,
        transfers: std::sync::mpsc::Sender<PpWaveMessage>,
        acknowledgements: std::sync::mpsc::Receiver<PpWaveTransfer>,
    ) -> Self {
        Self {
            boundary,
            transfers,
            acknowledgements,
            slot_owner: [None, None],
            pending: VecDeque::new(),
            next_slot: None,
            next_wave: 0,
        }
    }

    fn receive_ack(&mut self, expected: PpWaveTransfer) -> Result<(), String> {
        let actual = self.acknowledgements.recv().map_err(|_| {
            format!(
                "PP wave boundary {} acknowledgement channel closed waiting for wave {} slot {}",
                self.boundary, expected.wave, expected.slot,
            )
        })?;
        if actual != expected {
            return Err(format!(
                "PP wave boundary {} expected acknowledgement wave {} slot {}, got boundary {} wave {} slot {}",
                self.boundary,
                expected.wave,
                expected.slot,
                actual.boundary,
                actual.wave,
                actual.slot,
            ));
        }
        let pending = self.pending.pop_front().ok_or_else(|| {
            "PP wave acknowledgement arrived with no pending transfer".to_string()
        })?;
        if pending != expected {
            return Err(format!(
                "PP wave boundary {} acknowledgement order mismatch: pending wave {} slot {}, expected wave {} slot {}",
                self.boundary, pending.wave, pending.slot, expected.wave, expected.slot,
            ));
        }
        self.slot_owner[expected.slot] = None;
        Ok(())
    }

    /// Return the slot `tx_pipelined` must select next. If that slot still belongs to an older
    /// wave, wait for the exact downstream acknowledgement proving `rx` recorded `ev_rx` for it.
    fn prepare(&mut self, wave: usize) -> Result<Option<usize>, String> {
        if wave != self.next_wave {
            return Err(format!(
                "PP wave boundary {} producer order mismatch: expected wave {}, got {wave}",
                self.boundary, self.next_wave,
            ));
        }
        if let Some(slot) = self.next_slot
            && let Some(owner) = self.slot_owner[slot]
        {
            self.receive_ack(owner)?;
        }
        Ok(self.next_slot)
    }

    fn publish(
        &mut self,
        wave: usize,
        slot: usize,
        expected_slot: Option<usize>,
    ) -> Result<(), String> {
        if wave != self.next_wave {
            return Err(format!(
                "PP wave boundary {} publish order mismatch: expected wave {}, got {wave}",
                self.boundary, self.next_wave,
            ));
        }
        if slot >= 2 {
            return Err(format!(
                "PP wave boundary {} wave {wave} selected invalid slot {slot}",
                self.boundary,
            ));
        }
        if let Some(expected) = expected_slot
            && slot != expected
        {
            return Err(format!(
                "PP wave boundary {} wave {wave} broke slot alternation: expected {expected}, got {slot}",
                self.boundary,
            ));
        }
        if let Some(owner) = self.slot_owner[slot] {
            return Err(format!(
                "PP wave boundary {} attempted to reuse slot {slot} for wave {wave} before acknowledgement of wave {}",
                self.boundary, owner.wave,
            ));
        }
        let transfer = PpWaveTransfer {
            boundary: self.boundary,
            wave,
            slot,
        };
        self.slot_owner[slot] = Some(transfer);
        self.pending.push_back(transfer);
        self.next_slot = Some(slot ^ 1);
        self.next_wave += 1;
        self.transfers
            .send(PpWaveMessage::Transfer(transfer))
            .map_err(|_| {
                format!(
                    "PP wave boundary {} transfer channel closed publishing wave {wave} slot {slot}",
                    self.boundary,
                )
            })
    }

    fn finish(&mut self) -> Result<(), String> {
        while let Some(expected) = self.pending.front().copied() {
            self.receive_ack(expected)?;
        }
        Ok(())
    }

    fn publish_worker_error(&self, error: &str) {
        let _ = self.transfers.send(PpWaveMessage::WorkerError {
            boundary: self.boundary,
            wave: self.next_wave,
            error: error.to_string(),
        });
    }
}

fn pp_wave_channels(
    boundaries: usize,
) -> (Vec<Option<PpWaveOutgoing>>, Vec<Option<PpWaveIncoming>>) {
    let mut outgoing = Vec::with_capacity(boundaries);
    let mut incoming = Vec::with_capacity(boundaries);
    for boundary in 0..boundaries {
        let (transfer_tx, transfer_rx) = std::sync::mpsc::channel();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        outgoing.push(Some(PpWaveOutgoing::new(boundary, transfer_tx, ack_rx)));
        incoming.push(Some(PpWaveIncoming {
            boundary,
            transfers: transfer_rx,
            acknowledgements: ack_tx,
        }));
    }
    (outgoing, incoming)
}

fn dual_pp_timing_event(e: &Engine, context: &str) -> Option<CudaEvent> {
    if !crate::pp::dual_pp_timing_on() {
        return None;
    }
    match e
        .stream()
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
    {
        Ok(event) => Some(event),
        Err(err) => {
            crate::pp::record_dual_pp_timing_drop(context, &err);
            None
        }
    }
}

/// Per-step, per-LAYER-RANGE invariants the batched trunk needs: the device state-pointer
/// table for the range's layers, the arm picks, and the per-row `t_kv` snapshot. Built once
/// per step per range by `HybridModel::batch_layer_ctx`, consumed by `decode_batch_layers`.
///
/// WHY IT IS RANGE-SCOPED AND NOT STEP-SCOPED (this is the whole point of the struct):
/// `ptr_table` is a `CudaSlice<u64>` of DEVICE ADDRESSES, uploaded through `e` — so it lives
/// on `e`'s device, and its entries are pointers into caches that live on the device that
/// OWNS those layers. Under a pp stage split, stage s runs layers [fence[s], fence[s+1])
/// whose cache state was allocated by stage s's engine (`pp::new_cache` -> `Cache::new_ppn`),
/// so stage s must build its OWN table through its OWN engine. One step-wide table built on
/// the primary would put every stage's kernel arguments in stage-0's HBM — a peer read per
/// pointer fetch, which is the exact cliff `pp::refuse_unsplit_if_remote` exists to stop.
/// `lo`/`hi` are recorded so the consumer can assert the ctx it was handed matches the range
/// it was asked to run (the offsets in `lin_base`/`attn_base` are only valid for that range).
pub(crate) struct BatchLayerCtx {
    /// Offset into `ptr_table` of layer il's [conv x B][ssm_in x B][ssm_out x B] block
    /// (linear-attn layers only). Indexed by ABSOLUTE layer id; `None` off-range.
    lin_base: Vec<Option<usize>>,
    /// Offset into `ptr_table` of layer il's [k0,v0,k1,v1,..] block (full-attn layers only).
    /// Indexed by ABSOLUTE layer id; `None` off-range.
    attn_base: Vec<Option<usize>>,
    ptr_table: Option<CudaSlice<u64>>,
    /// Per-row `pos + 1` — the t_kv each sequence attends at this step. Layer-invariant
    /// within a step, so the arm picks below are decided once.
    t_kvs: Vec<usize>,
    t_kv_max: usize,
    /// The single `fa_split_keys` rung every row shares (the rows-twins straddle law).
    sp0: usize,
    seqs_fa: bool,
    lo: usize,
    hi: usize,
}

// ---- MEMRA_BATCH_PHASE=1 (diagnostics): sync-bounded per-phase accumulators for the batched
// tick. Each boundary syncs the stream, so the TOTAL inflates (launch pipelining is destroyed);
// the value is the RANKING/shares, not absolute ms. Read via `batch_phase_report()`.
pub(crate) static BATCH_PHASE: std::sync::Mutex<[f64; 12]> = std::sync::Mutex::new([0.0; 12]);
/// Device-sample request for one batched row.
/// `top_k=0` / `top_p>=1.0` / `min_p<=0.0` = that filter off. Greedy = temp<=0 (device
/// argmax); pure temperature = seeded gumbel; any filter on = filter_stats floor + the
/// filtered gumbel draw. `penalty` carries host-maintained sparse counts for the exact active
/// history window; the epilogue applies them on device before filters and sampling.
#[derive(Clone, Debug)]
pub struct DevSamp {
    pub temp: f32,
    pub seed: u64,
    pub ctr: u32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub penalty: Option<DevPenalty>,
}

#[derive(Clone, Debug)]
pub struct DevPenalty {
    repeat: f32,
    freq: f32,
    present: f32,
    counts: Vec<(u32, u32)>,
}

/// A one-row decode whose device work has been enqueued but whose result has not crossed back
/// to the host yet. The worker owns the CUDA context, so this is deliberately a poll-at-the-next
/// scheduler-boundary handoff rather than a background CUDA thread. Keeping the completion event
/// and output buffers alive prevents the async-pool from recycling them while the next step runs.
pub struct PendingBatchStep {
    logits: CudaSlice<f32>,
    pristine: Vec<Option<CudaSlice<f32>>>,
    tokens: Option<CudaSlice<u32>>,
    sampled: Vec<bool>,
    n_vocab: usize,
    lean: bool,
    done: CudaEvent,
    readback: Arc<cudarc::driver::CudaStream>,
}

impl PendingBatchStep {
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn new(
        logits: CudaSlice<f32>,
        pristine: Vec<Option<CudaSlice<f32>>>,
        tokens: Option<CudaSlice<u32>>,
        sampled: Vec<bool>,
        n_vocab: usize,
        lean: bool,
        done: CudaEvent,
        readback: Arc<cudarc::driver::CudaStream>,
    ) -> Self {
        Self {
            logits,
            pristine,
            tokens,
            sampled,
            n_vocab,
            lean,
            done,
            readback,
        }
    }

    /// Wait for this step only, then perform one ordered readback of its host-visible results.
    /// The compute stream may already be carrying the following step when this runs.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn wait(self) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        self.readback.wait(&self.done)?;
        // Lean device-sampled rows already parked their pristine logits in the session cache;
        // their only host-visible result is the sampled token id. Avoid recreating the large
        // vocab-row D2H that this path was introduced to remove.
        let need_logits = !self.lean || self.sampled.iter().any(|sampled| !sampled);
        let host_logits = need_logits
            .then(|| self.readback.clone_dtoh(&self.logits))
            .transpose()?;
        let host_pristine: Vec<Option<Vec<f32>>> = self
            .pristine
            .iter()
            .map(|row| {
                row.as_ref()
                    .map(|row| self.readback.clone_dtoh(row))
                    .transpose()
            })
            .collect::<Result<_, _>>()?;
        let host_tokens = self
            .tokens
            .as_ref()
            .map(|tokens| self.readback.clone_dtoh(tokens))
            .transpose()?;
        self.readback.synchronize()?;

        let mut rows = Vec::with_capacity(self.sampled.len());
        for (bi, sampled) in self.sampled.iter().copied().enumerate() {
            if self.lean && sampled {
                rows.push(Vec::new());
            } else if let Some(row) = host_pristine[bi].as_ref() {
                rows.push(row.clone());
            } else {
                let start = bi * self.n_vocab;
                let logits = host_logits
                    .as_ref()
                    .ok_or("pending step did not retain host logits for an unsampled row")?;
                rows.push(logits[start..start + self.n_vocab].to_vec());
            }
        }
        let next = host_tokens.map_or_else(
            || vec![None; self.sampled.len()],
            |tokens| {
                self.sampled
                    .iter()
                    .enumerate()
                    .map(|(bi, sampled)| sampled.then_some(tokens[bi]))
                    .collect()
            },
        );
        Ok((rows, next))
    }
}

impl DevPenalty {
    /// Checked constructor for callers that do not already own a unique count map.
    pub fn try_new(
        repeat: f32,
        freq: f32,
        present: f32,
        counts: Vec<(u32, u32)>,
    ) -> Result<Self, &'static str> {
        let mut seen = std::collections::HashSet::with_capacity(counts.len());
        for &(id, count) in &counts {
            if count == 0 {
                return Err("device penalty counts must be positive");
            }
            if !seen.insert(id) {
                return Err("device penalty token ids must be unique");
            }
        }
        Ok(Self {
            repeat,
            freq,
            present,
            counts,
        })
    }

    /// Zero-copy validation seam for a producer that already owns a unique count map.
    ///
    /// # Safety
    ///
    /// `counts` must contain each token id at most once and every count must be positive. The
    /// batched kernel assigns one CUDA thread to each entry and performs a non-atomic
    /// read/modify/write of that token's logit.
    pub unsafe fn from_unique_counts_unchecked(
        repeat: f32,
        freq: f32,
        present: f32,
        counts: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            repeat,
            freq,
            present,
            counts,
        }
    }
}

impl DevSamp {
    pub fn new(temp: f32, seed: u64, ctr: u32, top_k: i32, top_p: f32, min_p: f32) -> Self {
        Self {
            temp,
            seed,
            ctr,
            top_k,
            top_p,
            min_p,
            penalty: None,
        }
    }

    pub fn with_penalty(mut self, penalty: DevPenalty) -> Self {
        self.penalty = Some(penalty);
        self
    }
}

pub const BATCH_PHASE_NAMES: [&str; 12] = [
    "setup(ptrs+embed H2D)",
    "attn batched pre (norm/qkv/rope)",
    "attn per-seq: kv append",
    "attn per-seq: q/a dtod copies",
    "attn per-seq: fa_decode",
    "attn post (gate+o-proj)",
    "gdn batched projections",
    "gdn state ops (conv/prep/scan)",
    "gdn out (gated norm+proj)",
    "ffn (add/norm/gate/up/act/down)",
    "lm_head (norm+matmul)",
    "logits D2H + host split",
];
pub fn batch_phase_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_BATCH_PHASE").as_deref() == Ok("1"))
}
/// Accumulate the elapsed time since `last` into phase slot `slot` and re-stamp `last`.
/// No-op unless `MEMRA_BATCH_PHASE=1`. Syncs the ambient stream first, so under a pp stage
/// scope this bounds the STAGE's stream, which is what the caller is timing.
///
/// A free fn rather than the closure it replaced: `decode_batch_layers` (the pp stage seam)
/// runs the instrumented layer loop, so the marker has to be callable from both the seam
/// and its caller's epilogue. `batch_phase_on()` is a `OnceLock` memo, so per-call cost is
/// the same atomic load the hoisted `ph_on` local was.
fn ph_mark(
    e: &Engine,
    slot: usize,
    last: &mut std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch_phase_on() {
        e.stream().synchronize()?;
        let now = std::time::Instant::now();
        BATCH_PHASE.lock().unwrap()[slot] += (now - *last).as_secs_f64();
        *last = now;
    }
    Ok(())
}

pub fn batch_phase_report() -> String {
    let ph = BATCH_PHASE.lock().unwrap();
    let tot: f64 = ph.iter().sum();
    let mut rows: Vec<(usize, f64)> = ph.iter().copied().enumerate().collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut s = format!(
        "[batch-phase] total {:.1} ms (sync-bounded; shares rank, not walltime)\n",
        tot * 1e3
    );
    for (i, v) in rows {
        s += &format!(
            "  {:>6.1} ms {:>5.1}%  {}\n",
            v * 1e3,
            v / tot * 100.0,
            BATCH_PHASE_NAMES[i]
        );
    }
    s
}

impl HybridModel {
    /// Batched-decode width cap. 8 = the exactness-tier default (see the assert below);
    /// MEMRA_DECODE_BATCH_CAP overrides for tier-probe measurement, clamped to 32.
    pub fn decode_batch_cap() -> usize {
        use std::sync::OnceLock;
        static CAP: OnceLock<usize> = OnceLock::new();
        *CAP.get_or_init(|| {
            std::env::var("MEMRA_DECODE_BATCH_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(|c: usize| c.clamp(1, 32))
                .unwrap_or(8)
        })
    }

    /// EXACT-16 TIER admission (increment 3a, 2026-08-01, 5090 receipts
    /// research/batched-tick-inc3-20260801): true iff EVERY matmul the batched decode step
    /// runs has a per-(token,row) bit-exact kernel class at m=9..16 under the verify_exact
    /// scope — i.e. the batched-mmvq b16 family (32-thread warp reduce, the exact m=1 mmvq
    /// program per column) or the e4m3 grid.y=m mmvq catch-all. Q8_0 qualifies only with
    /// the split-plane mirror (rp4, MEMRA_Q8RP): its b16 kernel exists only as the _rp twin.
    /// Float matmuls (cuBLASLt, n-dependent reductions) and MoE FFNs disqualify the model.
    /// Measured attribution for WHY the naked m=16 tier is not exact: the m>=16 arms
    /// (MMQ int8-MMA `mul_mat_q` — MEMRA_PP_Q8MMQ default-on — and `qmatvec_gemm`, both
    /// block-scale f32) and the m=9..15 dp4a tail (128-thread two-level reduce) all break
    /// per-row bit-identity vs isolated decode (gate2 step-0 bit-diffs, maxdiff ~1.3-2.3e-1).
    pub fn decode_batch_exact16_ok(&self) -> bool {
        fn ok(w: &crate::model::GpuTensor) -> bool {
            match w {
                crate::model::GpuTensor::Quant { qtype, .. } => {
                    *qtype == crate::QT_Q4_0 || *qtype == crate::QT_Q6_K
                    || *qtype == crate::QT_F8_E4M3
                    // BLOCK-128 FP8-ST (lane/rp-on-st, 2026-08-06): admitted now that the class
                    // has a b16 batched kernel (`qmatvec_e4m3_blk_mmvq_b16`), bit-identical per
                    // (token,row) to its m=1 launch. Before that kernel existed this class fell to
                    // the grid.y=m form at every width — still EXACT, so the tier's correctness
                    // bar was met, but it re-read the weight m times, which is why admitting it
                    // without the kernel would have been a throughput trap rather than a win.
                    || *qtype == crate::QT_F8_E4M3_BLK
                    // NVFP4 (lane/rp-on-st, 2026-08-06) — THE blocker this lane measured. The
                    // mixed FP8-ST 27B is 193 NVFP4 dense-MLP tensors, and this predicate is an
                    // ALL over every matmul, so NVFP4's missing b16 refused the whole checkpoint
                    // (`B=16 > cap 8 with no exact tier ... refused`) even with both e4m3 classes
                    // admitted. It now has base + _rp b16 twins off its existing batched template
                    // (bit-identical per (token,row) to the m=1 mmvq: same nibble decode, dp4a
                    // order, ue4m3 scale, warp reduce). This also opens the tier for pure-NVFP4
                    // GGUF models, which is a behavior change on the primary format — hence the
                    // full decode-batch config+strict battery on both.
                    || *qtype == crate::QT_NVFP4
                    // Q4_K (lane/rp-on-st): named by MEMRA_EXACT16_WHY as the 9B NVFP4 GGUF's
                    // refusing class (`L0.wqkv qtype=1`) — mixed NVFP4 checkpoints keep Q4_K
                    // attention. Now has base + _rp b16.
                    || *qtype == crate::QT_Q4_K
                    // Q5_K (lane/rp-on-st): the FOURTH class the diagnostic named on the same 9B
                    // GGUF (`L0.wqkv_gate qtype=3`). A shipped mixed checkpoint spreads ~500
                    // matmuls over four/five classes, and this predicate is an ALL — so chunk 16
                    // was unreachable for every real artifact until every class had a b16.
                    || *qtype == crate::QT_Q5_K
                    // Q8_0 NO LONGER requires the mirror (rp4): it has a base b16 too, so the
                    // tier is reachable at zero VRAM. Named by the diagnostic as the FP8-ST
                    // refusal — `L0.ssm_beta qtype=0 rp4=false`, a 23.9 MiB residual class that
                    // was gating chunk 16 for a 16.4 GiB checkpoint.
                    || *qtype == crate::QT_Q8_0
                }
                _ => false,
            }
        }
        // WHY-NOT DIAGNOSTIC (lane/rp-on-st, 2026-08-06): this predicate is a bare bool over
        // ~500 tensors, so a refusal produced only `B=16 > cap 8 with no exact tier ... refused`
        // with no way to tell WHICH class refused. That cost this lane two wrong hypotheses (the
        // rp mirror, then e4m3-only) before the NVFP4 gap was found. MEMRA_EXACT16_WHY=1 names
        // the first refusing tensor + its qtype. Diagnostic-only per flags doctrine; default off,
        // zero cost when unread.
        let why = std::env::var("MEMRA_EXACT16_WHY").is_ok();
        macro_rules! chk {
            ($t:expr, $label:expr) => {{
                let r = ok($t);
                if !r && why {
                    // qtype = -1 means the tensor is NOT Quant at all (a float/BF16/F16
                    // container), which the tier can never admit — a distinct diagnosis from
                    // "quantized, but in a class with no b16 kernel".
                    let (qt, rp4) = match $t {
                        crate::model::GpuTensor::Quant { qtype, rp4, .. } => {
                            (*qtype, rp4.is_some())
                        }
                        _ => (-1, false),
                    };
                    eprintln!("[exact16] REFUSED by {} qtype={qt} rp4={rp4}", $label);
                }
                r
            }};
        }
        let operations = self.plan.trunk_operations();
        if operations.contains(&memra_gguf::model_plan::OperationKind::SwiGluOaiActivation)
            || self.is_gemma4_e4b()
            || crate::plan_backend::decode_batch_program(&self.plan)
                == crate::plan_backend::DecodeBatchProgram::Gemma
        {
            if why {
                eprintln!("[exact16] REFUSED by architecture (m3/gemma4)");
            }
            return false;
        }
        self.layers.iter().enumerate().all(|(li, l)| {
            let mix_ok = match &l.mixer {
                Mixer::Full(fa) => {
                    chk!(&fa.wq, format!("L{li}.wq"))
                        && chk!(&fa.wk, format!("L{li}.wk"))
                        && chk!(&fa.wv, format!("L{li}.wv"))
                        && chk!(&fa.wo, format!("L{li}.wo"))
                }
                Mixer::Linear(la) => {
                    chk!(&la.wqkv, format!("L{li}.wqkv"))
                        && chk!(&la.wqkv_gate, format!("L{li}.wqkv_gate"))
                        && chk!(&la.ssm_beta, format!("L{li}.ssm_beta"))
                        && chk!(&la.ssm_alpha, format!("L{li}.ssm_alpha"))
                        && chk!(&la.ssm_out, format!("L{li}.ssm_out"))
                }
                // MLA rides its own increment-4 arm; never admitted to the exact-16 tier here.
                Mixer::Mla(_) => {
                    if why {
                        eprintln!("[exact16] REFUSED by L{li} MLA mixer");
                    }
                    false
                }
                // KDA rides its own eager arm; never admitted to the exact-16 tier here.
                Mixer::Kda(_) => {
                    if why {
                        eprintln!("[exact16] REFUSED by L{li} KDA mixer");
                    }
                    false
                }
            };
            let ffn_ok = match &l.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    // memra#253: this site inspects or moves weights and runs no GEMM on an
                    // activation, so the AWQ activation-side scale plays no part in it.
                    ffn_down_pqs: _,
                } => {
                    chk!(ffn_gate, format!("L{li}.ffn_gate"))
                        && chk!(ffn_up, format!("L{li}.ffn_up"))
                        && chk!(ffn_down, format!("L{li}.ffn_down"))
                }
                crate::hybrid::Ffn::Moe(m) => {
                    // lane/orndecode-20260822: the categorical refusal here was the c16 wall on
                    // MoE checkpoints — serve chunked c16 into two B<=8 waves (agg flat ~700 on
                    // ornith15 while the frozen vLLM column reads ~1190). The MoE stage itself is
                    // width-exact by construction at decode widths: the dev/pairs expert kernels
                    // replay one per-(token,expert) program whose arithmetic never sees batch
                    // width, the router (gemv f32 + sigmoid + topk) is row-wise, and the shexp
                    // trio rides the per-column decode-exact arm at every verify width
                    // (t in 2..PRIME_MIN_T), so no b16 qmatvec class is ever demanded of it.
                    // "By construction" is NOT the qualification — the CSR-NVFP4
                    // batch-composition defect (v0.99.0, research/samplat-20260821) shipped on
                    // exactly that reasoning. STATUS (orndecode, 2026-08-22): byte gates are
                    // GREEN on ornith15 (decode-batch-gate config gate2+gate3 PASS at B=12 and
                    // B=16, bit-checked vs isolated) but the tier LOSES throughput today —
                    // B=16 exact measured 220 agg vs 551 at B=8 same-window, because the
                    // exact-verify scope drives the shexp trio (and friends) to per-column m=1
                    // decode-exact launches. MEMRA_EXACT16_MOE=1 is therefore an OPT-IN
                    // measurement door until the b16-class stage kernels land; serve must not
                    // pick a tier that halves the aggregate it exists to raise.
                    if std::env::var("MEMRA_EXACT16_MOE").as_deref() != Ok("1") {
                        if why {
                            eprintln!(
                                "[exact16] REFUSED by L{li} MoE ffn (opt-in: MEMRA_EXACT16_MOE=1 \
                                 — byte-safe but slower than two B<=8 waves today)"
                            );
                        }
                        false
                    } else {
                        match (&m.gate_shexp, &m.up_shexp, &m.down_shexp) {
                            (Some(g), Some(u), Some(d)) => {
                                chk!(g, format!("L{li}.gate_shexp"))
                                    && chk!(u, format!("L{li}.up_shexp"))
                                    && chk!(d, format!("L{li}.down_shexp"))
                            }
                            _ => true,
                        }
                    }
                }
            };
            mix_ok && ffn_ok
        }) && chk!(&self.output, "output".to_string())
    }

    /// Opt-in/A-B seam for the eager B=1 fusion program. `MEMRA_SERVE_B1FAST=1` sends an
    /// eligible solo tick through that program; unset/other values keep B=1 on the generic
    /// batched body, the same numeric class used at B>=2.
    ///
    /// EXACTNESS, stated precisely (measured on-box 2026-08-05, sm_120 q9 NVFP4-MTP):
    /// the fast path is BIT-IDENTICAL TO `decode_step_h` — decode-batch-gate's STRICT
    /// gate1 (`--mode strict`) PASSes with it ON and FAILs with it OFF at maxdiff
    /// 1.591e-1. It is deliberately NOT bit-identical to the batched body: the two
    /// carry a decode-config FP-composition gap (same class gate1's config mode measures).
    /// That gap became correctness-visible under live load: Step35, Q35-MoE, and finally
    /// dense Q27 all produced load-history-dependent token streams, including early EOS,
    /// when a request crossed between the two programs. The generic body is therefore the
    /// correctness default; the eager program remains available only for fixed-solo A/Bs.
    /// Historical token-stream/performance receipts:
    /// research/servepath-p2-20260805 (greedy 150 ids + seeded-sampled identical to the
    /// run-gen oracle AND cross-arm, so the gap is sub-token here as designed).
    ///
    /// Read fresh (an `AtomicU8` memo, not a `OnceLock`): decode-batch-gate flips this
    /// seam BETWEEN gates in-process — gate1 needs the fast path ON to prove bit-identity,
    /// gate2 needs it pinned OFF to keep testing the batched body. A latch-once read would
    /// bake whichever gate ran first, so the gate could never test both sides. The memo
    /// caches the parse but `set_b1_fast` invalidates it.
    pub fn b1_fast_on() -> bool {
        // 0 = unknown/invalidated, 1 = off, 2 = on
        match Self::b1_fast_memo().load(std::sync::atomic::Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let value = std::env::var("MEMRA_SERVE_B1FAST").ok();
                let on = b1_fast_env_on(value.as_deref());
                Self::b1_fast_memo()
                    .store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
                on
            }
        }
    }

    fn b1_fast_memo() -> &'static std::sync::atomic::AtomicU8 {
        static MEMO: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        &MEMO
    }

    /// Test/gate seam: force the B=1 fast path on or off for the rest of the process,
    /// overriding the env. Used by decode-batch-gate to exercise the opt-in eager arm and
    /// pin gate2's default reference arm.
    pub fn set_b1_fast(on: bool) {
        Self::b1_fast_memo().store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether this architecture may switch a live serving row onto the eager B=1 fusion
    /// class. Qwen35-MoE must stay on the batched trunk at every width: its eager and batched
    /// hybrid/MoE walks are each deterministic, but crossing B=1 -> B>=2 changes greedy token
    /// ids and can introduce an early EOS (Q35 sellgate, 2026-08-12).
    pub fn b1_fast_plan_eligible(&self) -> bool {
        b1_fast_plan_eligible(&self.plan)
    }

    /// H3 body: the m=1 FUSED trunk (`decode_layers_eager` — shared verbatim with
    /// `decode_step_h`/the ppN stages) plus the batched path's own serving epilogue
    /// (grammar mask, device sample, lean-logits park). See the call-site comment in
    /// `decode_step_batch_sampled_lean_masked` for why this is bit-identical.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_step_b1_fast(
        &self,
        e: &Engine,
        token: u32,
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let pos = caches[0].pos;
        let pos_d = e.htod_i32(&[pos as i32])?;
        let x = e.htod(&self.embd.try_gather(n_embd, &[token])?)?;
        // the SHARED m=1 trunk: same function decode_step_h runs, so every m=1 fusion
        // (cross-layer add+norm+q8_1, fused SwiGLU, lever 1's gate+up dual) fires here.
        let x = self.decode_layers_eager(e, x, 0, self.layers.len(), &pos_d, pos, caches[0])?;
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;

        // ---- epilogue: byte-for-byte the batched path's, at b_n=1 ----
        let n_vocab = self.output.out_features();
        let mut logits = logits;
        let mut pristine: Option<CudaSlice<f32>> = None;
        if let Some((mask, words)) = masks.first().copied().flatten() {
            assert!(
                samp.first().and_then(Option::as_ref).is_some(),
                "grammar-masked row 0 must request a device sample"
            );
            if lean {
                let cache = &mut caches[0];
                if cache
                    .last_logits_dev
                    .as_ref()
                    .map(|d| d.len() < n_vocab)
                    .unwrap_or(true)
                {
                    cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                }
                let dst = cache.last_logits_dev.as_mut().unwrap();
                e.dtod_copy_view(&logits.slice(0..n_vocab), dst)?;
            } else {
                let mut p = e.uninit(n_vocab)?;
                e.dtod_copy_view(&logits.slice(0..n_vocab), &mut p)?;
                pristine = Some(p);
            }
            e.mask_logits_col(&mut logits, mask, 0, n_vocab, words)?;
        }

        let mut next: Vec<Option<u32>> = vec![None; 1];
        if let Some(s) = samp.first().and_then(Option::as_ref) {
            let mut toks = e.alloc_u32_zeroed(1)?;
            // Filtered-greedy degenerates to plain argmax (the max always survives every
            // truncation filter), so temp<=0 short-circuits regardless of filters.
            let filtered = s.temp > 0.0 && (s.top_k > 0 || s.top_p < 1.0 || s.min_p > 0.0);
            if s.temp <= 0.0 {
                e.argmax_token_device_col(&logits, 0, n_vocab, &mut toks, 0)?;
            } else if filtered {
                let mut pb = e.zeros(n_vocab)?;
                self.devsample_filtered_col(
                    e, &logits, 0, n_vocab, s.temp, s.seed, s.ctr, s.top_k, s.top_p, s.min_p,
                    &mut pb, &mut toks, 0,
                )?;
            } else {
                let mut pb = e.zeros(n_vocab)?;
                e.gumbel_perturb_col(&logits, 0, &mut pb, n_vocab, s.seed, s.ctr, s.temp)?;
                e.argmax_token_device_col(&pb, 0, n_vocab, &mut toks, 0)?;
            }
            next[0] = Some(e.dtoh_u32(&toks)?[0]);
        }

        let sampled = samp.first().and_then(Option::as_ref).is_some();
        let rows: Vec<Vec<f32>> = if lean && sampled {
            if masks.first().copied().flatten().is_none() {
                let cache = &mut caches[0];
                if cache
                    .last_logits_dev
                    .as_ref()
                    .map(|d| d.len() < n_vocab)
                    .unwrap_or(true)
                {
                    cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                }
                let dst = cache.last_logits_dev.as_mut().unwrap();
                e.dtod_copy_view(&logits.slice(0..n_vocab), dst)?;
            }
            vec![Vec::new()]
        } else if let Some(p) = pristine.as_ref() {
            vec![e.dtoh(p)?]
        } else {
            vec![e.dtoh(&logits)?]
        };
        // decode_layers_eager does NOT advance cache.pos (decode_step_h advances it after
        // the head); the batched path advances every cache at the tail — same here.
        caches[0].pos += 1;
        Ok((rows, next))
    }

    /// One filtered device draw for stacked-logits row `col`: `filter_stats` solves the
    /// single unnormalized-prob floor that encodes top-k AND top-p AND min-p (block-internal
    /// binary search, bit-stable), then the filtered gumbel perturb + argmax draws one token
    /// from the truncated softmax into `toks[slot]`. All device-side — no stat D2H, no row
    /// copy; the only host traffic stays the caller's one [B]-u32 token readback.
    #[allow(clippy::too_many_arguments)]
    fn devsample_filtered_col(
        &self,
        e: &Engine,
        logits: &CudaSlice<f32>,
        col: usize,
        n_vocab: usize,
        temp: f32,
        seed: u64,
        ctr: u32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        pb: &mut CudaSlice<f32>,
        toks: &mut CudaSlice<u32>,
        slot: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rows = e.htod_i32(&[col as i32])?;
        let mut th = e.zeros(1)?;
        let mut z = e.zeros(1)?;
        let mut mx = e.zeros(1)?;
        e.filter_stats(
            logits, n_vocab, &rows, &mut th, &mut z, &mut mx, n_vocab, 1, temp, top_k, top_p, min_p,
        )?;
        e.gumbel_perturb_filtered_col(logits, col, pb, n_vocab, seed, ctr, temp, &mx, &th, 0)?;
        e.argmax_token_device_col(pb, 0, n_vocab, toks, slot)?;
        Ok(())
    }

    /// One batched greedy-decode step over B independent sequences.
    /// `tokens[b]` is sequence b's input token; `caches[b]` its private cache (position,
    /// quantized KV, GDN/conv state). Returns the B logits rows (host, [n_vocab] each).
    /// Each cache's pos/len advance exactly as `decode_step_h` would.
    pub fn decode_step_batch(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let (rows, _) = self.decode_step_batch_sampled(e, tokens, caches, &[])?;
        Ok(rows)
    }

    /// `decode_step_batch` + DEVICE-SIDE SAMPLING for eligible rows (the batched-tick lever,
    /// 2026-08-01): the host sampler's temp-path is O(n_vocab) with a full-vocab exp per row
    /// (measured 1.36 ms/row at the 9B's 248320 vocab = 10.9 ms/tick at B=8 — the single
    /// largest component of the serving tick). Here each requested row samples ON DEVICE
    /// between the lm_head matmul and the logits D2H:
    ///   temp <= 0 (greedy): the 2-pass device argmax — bit-identical to host argmax
    ///     (argmax-gate contract, same kernels as the dc serving path).
    ///   temp > 0: gumbel_perturb(seed, ctr, temp) + the same argmax = ONE categorical draw
    ///     from softmax(logits/temp) — the sampled-spec Philox machinery. Deterministic per
    ///     (seed, ctr) and INDEPENDENT of batch composition (the isolation contract;
    ///     decode-batch-gate gate3). NOTE: the draw stream differs from the host sampler's
    ///     SplitMix64 (distribution-equal, seed-deterministic, NOT byte-equal to the old
    ///     host draws) — greedy rows are unchanged bit-exact.
    /// `samp[bi] = Some(DevSamp { .. })` requests a device sample for row bi; the full
    /// logits rows are still returned (worker keeps last_logits semantics + fallback rows).
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn decode_step_batch_sampled(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        self.decode_step_batch_sampled_lean(e, tokens, caches, samp, false)
    }

    /// `decode_step_batch_sampled` + LEAN LOGITS (increment 2 component 3, 2026-08-01):
    /// with `lean`, device-sampled rows SKIP the [n_vocab] logits D2H (9.4%/32.5% of the
    /// pre-/post-inc2 tick profile) — their returned row is EMPTY. The audit-mapped
    /// consumers: (a) the next tick's host sample — never fires, `device_next` carries the
    /// token; (b) the graph-promotion argmax — reads only prefill logits (generated empty);
    /// (c) the KV-reuse pool park at retire — the REAL consumer, served by a per-cache
    /// device park: the row is dtod-copied into `cache.last_logits_dev` (device bandwidth)
    /// and D2H'd ONCE at retire by the worker. Rows without a device sample keep a per-row
    /// D2H. `lean=false` is bit-for-bit the previous method (gates + non-serving callers).
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn decode_step_batch_sampled_lean(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        self.decode_step_batch_sampled_lean_masked(e, tokens, caches, samp, &[], lean)
    }

    /// `decode_step_batch_sampled_lean` + GRAMMAR MASKS (constrained decoding, 2026-08-03):
    /// `masks[bi] = Some((packed_bitset, words))` bans every unset-bit vocab id on row bi
    /// (mask_logits_f32, -FLT_MAX) BETWEEN the lm_head matmul and the device sampler, so a
    /// constrained row rides the SAME device-sample/lean-logits tick as everyone else — no
    /// full-row D2H, no host O(n_vocab) sample. Contract: a masked row must also request a
    /// device sample. The row's PRISTINE logits are preserved for their consumers before the
    /// in-place ban: lean rows park the unmasked row into `cache.last_logits_dev` (the
    /// retire-time reuse-pool park stays unmasked — continuations resume grammar-free, the
    /// v1 host-path contract), non-lean rows D2H the unmasked row. `masks = &[]` is
    /// bit-for-bit the unmasked method.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn decode_step_batch_sampled_lean_masked(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        self.decode_step_batch_sampled_lean_masked_schedule(
            e, tokens, caches, samp, masks, lean, None, None,
        )
    }

    /// Whether the generic, unsplit batched trunk can leave its result on the device for one
    /// scheduler boundary. The pending path is intentionally c=1-only today: PP stages, model
    /// specific batched programs, and the fixed-solo fusion arm each have different output
    /// ownership and keep their established synchronous readback contract.
    pub fn decode_step_overlap_eligible(&self) -> bool {
        !batch_phase_on()
            && crate::pp::pp_cuts(self.layers.len()).is_none()
            // mHC trunks are excluded: the pending (deferred-readback) epilogue is only
            // wired for the generic trunk body, and the hyper walk keeps the synchronous
            // readback contract (see the named refusal in `_pending`).
            && self.hyper.is_none()
            && !Self::b1_fast_on()
            && self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::DecodeBatch)
            && crate::plan_backend::decode_batch_program(&self.plan)
                == crate::plan_backend::DecodeBatchProgram::Generic
    }

    /// Enqueue one generic B=1 decode and defer its D2H until [`PendingBatchStep::wait`]. This
    /// is the engine half of the overlap scheduler: the server can publish the token selected
    /// from step n before it waits for step n+1's logits.
    pub fn decode_step_batch_sampled_lean_masked_pending(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<PendingBatchStep, Box<dyn std::error::Error>> {
        if self.hyper.is_some() {
            return Err(
                "decode_step_batch_sampled_lean_masked_pending: the overlap scheduler's \
                 deferred-readback step is not wired for the HyperConnections residual — \
                 the hyper batched walk keeps the synchronous epilogue (its `pending_out` \
                 plumbing through `decode_step_batch_hyper` does not exist yet). Serve mHC \
                 sessions through the synchronous batched chain or the eager per-session \
                 loop; `decode_step_overlap_eligible` already reports false for this trunk."
                    .into(),
            );
        }
        if tokens.len() != 1 || caches.len() != 1 {
            return Err("overlap scheduler requires a single decode row".into());
        }
        if !self.decode_step_overlap_eligible() {
            return Err(
                "overlap scheduler is unavailable for this model, topology, or diagnostic arm"
                    .into(),
            );
        }
        let mut pending = None;
        let _ = self.decode_step_batch_sampled_lean_masked_schedule(
            e,
            tokens,
            caches,
            samp,
            masks,
            lean,
            None,
            Some(&mut pending),
        )?;
        pending.ok_or_else(|| "overlap scheduler did not produce a pending step".into())
    }

    /// Worker-scheduled twin of [`Self::decode_step_batch_sampled_lean_masked`]. The worker
    /// supplies the balanced dual-wave boundary it used when forming this tick. Direct engine
    /// callers keep the automatic midpoint above; the explicit seam makes scheduler chunking and
    /// engine execution one checked contract instead of two coincident width calculations.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn decode_step_batch_sampled_lean_masked_scheduled(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        dual_wave_mid: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        if self.hyper.is_some() {
            return Err(
                "decode_step_batch_sampled_lean_masked_scheduled: the dual-active PP-2 \
                 wave schedule has no HyperConnections trunk — `decode_step_batch_dual`'s \
                 two host walkers drive the generic/step35 layer bodies only, and no \
                 dual-wave twin of `hyper_batch_range_decode` exists. mHC chunks are \
                 serial ticks: the worker's chunk policy must not form dual waves for \
                 this topology (decode_step_batch_hyper owns the serial PP-N split)."
                    .into(),
            );
        }
        self.decode_step_batch_sampled_lean_masked_schedule(
            e,
            tokens,
            caches,
            samp,
            masks,
            lean,
            Some(dual_wave_mid),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_step_batch_sampled_lean_masked_schedule(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        scheduled_dual_mid: Option<usize>,
        pending_out: Option<&mut Option<PendingBatchStep>>,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        for cache in caches.iter() {
            cache.ensure_usable("decode_step_batch")?;
        }
        if crate::pp::pp_cuts(self.layers.len()).is_some()
            && !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::Pipeline)
        {
            return Err("pipeline rewrite is not qualified for batched decode".into());
        }
        if !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::DecodeBatch) {
            if !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::DecodeEager) {
                return Err("neither batch nor eager decode rewrite is qualified".into());
            }
            if masks.iter().any(Option::is_some) {
                return Err(
                    "unqualified batch rewrite cannot fall back with device grammar masks".into(),
                );
            }
            if tokens.len() != caches.len() {
                return Err("batch fallback token/cache shape mismatch".into());
            }
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[rewrite] decode-batch.v1 unqualified; using receipt-backed native eager rows"
                );
            });
            let mut rows = Vec::with_capacity(tokens.len());
            for (token, cache) in tokens.iter().copied().zip(caches.iter_mut()) {
                rows.push(self.decode_step_h(e, token, cache)?.0);
            }
            return Ok((rows, vec![None; tokens.len()]));
        }
        // NOTE (inc3 3c, 2026-08-01, KILLED ARM): a deferred-token-readback variant (all
        // chunks of a tick writing device-sampled tokens into one shared buffer, ONE
        // dtoh_u32 after the last chunk instead of one per chunk) measured FLAT at serve
        // level on the 5090 (N=4 medians within +-0.7% at c=8/16/32 — 3 saved syncs
        // against a ~100 ms weight-bound tick is ~0.1%, below resolution). Killed per the
        // flags doctrine; receipts research/batched-tick-inc3-20260801 (serve-points.jsonl
        // base vs defer arms) are the record. The per-chunk [B]-u32 readback below IS the
        // tick's only steady-state D2H — one per chunk, none per seq.
        let b_n = tokens.len();
        assert!(
            b_n >= 1 && b_n == caches.len(),
            "tokens/caches length mismatch"
        );
        // ---- mHC DOOR (lane/glm53-batched-decode, 2026-08-28): the HyperConnections trunk
        // takes its OWN batched walk. Every body below this point runs the serial residual —
        // on an hc model that is a DIFFERENT function computed fluently (the failure class
        // `refuse_hyper` exists for) — so the hyper route must come before the pp door, the
        // b1 fast path, and the width tiers, and it owns its own PP-N stage split inside.
        // The dual-wave and pending entries refused above with named reasons; this guard is
        // the defense-in-depth backstop for a future caller that reaches here with either.
        if self.hyper.is_some() {
            if scheduled_dual_mid.is_some() {
                return Err(
                    "decode_step_batch: a dual-wave schedule reached the hyper trunk — \
                     no dual-wave twin of hyper_batch_range_decode exists; mHC chunks are \
                     serial ticks"
                        .into(),
                );
            }
            if pending_out.is_some() {
                return Err(
                    "decode_step_batch: the pending (deferred-readback) epilogue reached \
                     the hyper trunk — the hyper walk keeps the synchronous readback \
                     contract"
                        .into(),
                );
            }
            return self.decode_step_batch_hyper(e, tokens, caches, samp, masks, lean);
        }
        let _pp_walk =
            if crate::pp::pp_cuts(self.layers.len()).is_some() && !crate::pp::pp2_streams_off() {
                let rt = crate::pp::PpNRt::get(e)?;
                Some(rt.acquire_walk("decode_step_batch")?)
            } else {
                None
            };
        // ---- PP DOOR: THE BATCHED STAGE SPLIT (pp2-batch 2026-08-06) ----------------------
        // Until this increment this body had NO pp arm: it walked lo=0..n_layers on the
        // primary engine's stream, with no stage split, no boundary, and no `rt.enter()`. With
        // the door open and a sharded cross-device placement, every projection for the remote
        // stages' layers was read over PCIe, per step, silently — measured 7.4 vs 208.9 tok/s
        // at B=1 (28x), 47.4 vs 657.0 at B=8 (13.9x) on a PRO 6000 pair over Gen5 x16 P2P.
        // Nothing failed or warned, because peer reads return identical bytes and all three
        // `decode-batch-gate` gates PASS on that config — the failure mode was performance,
        // and a green exactness battery hid it. `pp2-hardening` made that regime FAIL CLOSED
        // (research/pp2-hardening-20260806); this lane makes it legitimately split, so the
        // refusal lifts for the batched path.
        //
        // `decode_step_batch_ppn` runs each stage's layer range through that stage's engine
        // and stream with a [B, n_embd] boundary transfer between them, i.e. every stage
        // touches only LOCAL weights and LOCAL cache state. The refusal below still guards
        // the residue: the door open with `MEMRA_PP_STREAMS=0` (the same-stream rollback,
        // which also disables the sharded loader, so nothing is remote — `pp_shard_off` and
        // `pp2_streams_off` both make `pp_sharded_cross_device()` false) or a placement whose
        // PpNRt fails to build. Keeping the call means a future path that reaches here in a
        // remote regime still refuses instead of regressing 28x.
        if let Some(fence) = crate::pp::pp_cuts(self.layers.len())
            && !crate::pp::pp2_streams_off()
            && crate::pp::batch_pp_on()
        {
            let n_stages = fence.len() - 1;
            let wave_on = crate::pp::pp_wave_on()
                .map_err(|reason| -> Box<dyn std::error::Error> { reason.into() })?;
            if crate::pp::pp_wave_route_enabled(wave_on, crate::pp::pp2_overlap(), n_stages, b_n) {
                if scheduled_dual_mid.is_some() {
                    return Err(
                        "decode_step_batch: worker supplied a PP2 midpoint to a PP3/PP4 wavefront"
                            .into(),
                    );
                }
                return self
                    .decode_step_batch_wavefront(e, tokens, caches, samp, masks, lean, &fence);
            }
            // Auto (flipped default) routes dual only in the re-gated regime and
            // degrades serially elsewhere; Forced keeps every ineligible placement on
            // the refusing dual body so the binding negative cells stay reachable.
            let route_dual = crate::pp::dual_pp_route(
                crate::pp::dual_pp_mode(),
                b_n,
                fence.len() - 1,
                crate::pp::pp2_overlap(),
                crate::pp::pp_host_bounce_active(),
            );
            if route_dual {
                let mid = scheduled_dual_mid
                    .or_else(|| crate::pp::dual_pp_wave_mid(b_n))
                    .expect("dual PP B>=2 must have a wave midpoint");
                return self
                    .decode_step_batch_dual(e, tokens, caches, samp, masks, lean, &fence, mid);
            }
            return self.decode_step_batch_ppn(e, tokens, caches, samp, masks, lean, &fence);
        }
        if scheduled_dual_mid.is_some() {
            return Err(
                "decode_step_batch: worker supplied a dual-wave schedule but the PP-2 dual path is unavailable"
                    .into(),
            );
        }
        crate::pp::refuse_unsplit_if_remote(
            "decode_step_batch",
            "drop MEMRA_PP_STREAMS=0 / MEMRA_BATCH_PP=0 so the batched path takes its OWN \
             stage split (decode_step_batch_ppn), or serve single-stream over the eager pp \
             arm (decode_step_h), which is also split",
        )?;
        // ---- H3: B=1 FAST-PATH (serve-path phase 2, 2026-08-05) ----------------------------
        // At b_n==1 every projection below calls `matmul_pre(.., b_n)` with m=1, which is
        // ALREADY the m=1 mmvq dispatch — so the m=1 *kernel family* was never the gap. What
        // this body does NOT have is the m=1 *fusion chain* that `decode_step_h` carries:
        //   - the cross-layer add+norm+quantize fusion (`add_rms_norm_q8_1`: 3 launches -> 1),
        //   - the fused SwiGLU epilogue (`silu_mul_scaled_q8_1`: folds ffn_down's quantize
        //     into its producer) and, with it, `matmul_pre_dual_noscale`'s gate+up pair
        //     fusion — i.e. phase-1 LEVER 1.
        // Routing b_n==1 through `decode_layers_eager` (the SHARED trunk `decode_step_h` and
        // the ppN stages already use, lifted verbatim — not a copy) makes every present and
        // future m=1 lever fire on the opt-in path automatically. The epilogue (grammar mask ->
        // device sample -> lean logits park) stays exactly as the batched path runs it; the trunk's
        // different FP composition is why this path cannot be a load-changing default.
        // BIT-IDENTITY: the trunk is the same function `decode_step_h` calls, and every
        // fusion it enables is kernel-check-pinned bit-identical to its unfused sequence
        // (add_rms_norm == add;rms_norm | _q8_1 == +quantize_q8_1 | dual_noscale == two
        // matmul_pre_noscale). Gate: decode-batch-gate B=1 vs decode_step_h + serve stream
        // identity. MEMRA_SERVE_B1FAST=1 is the fixed-solo opt-in/A-B seam; the default
        // stays on this function's generic body so batch-width changes cannot change the
        // FP program mid-request.
        if b_n == 1
            && Self::b1_fast_on()
            && !samp.iter().flatten().any(|s| s.penalty.is_some())
            && self.b1_fast_plan_eligible()
            && !self.is_gemma4_e4b()
            && crate::plan_backend::decode_batch_program(&self.plan)
                == crate::plan_backend::DecodeBatchProgram::Generic
            && !self
                .plan
                .trunk_operations()
                .contains(&memra_gguf::model_plan::OperationKind::SwiGluOaiActivation)
            && crate::pp::pp_cuts(self.layers.len()).is_none()
            && !e.verify_exact_on()
        {
            return self.decode_step_b1_fast(e, tokens[0], caches, samp, masks, lean);
        }
        // MEMRA_DECODE_BATCH_CAP (experimental door, serving-lane tier probe 2026-08-01):
        // default 8 keeps the v1 exactness policy — B=2..8 rides the verify-tier batched
        // mmvq arms, per-row bit-identical to isolated m=1 decode. Values >8 are a
        // MEASUREMENT DOOR ONLY: m=9..15 falls to the grid.y=m dp4a tail (m weight
        // re-reads + a different reduce shape) and m>=16 crosses into the GEMM tier
        // (block-scale f32 rounding) — BOTH break the "byte-identical to isolated"
        // serving contract. Never default this above 8 without the batched-tier
        // exactness policy landing.
        let cap = Self::decode_batch_cap();
        // EXACT-16 TIER (increment 3a): chunks of 9..=16 are admitted WITHOUT the env door
        // when every matmul has a bit-exact b16-class kernel (see decode_batch_exact16_ok).
        // The verify_exact scope below pins that dispatch for the whole step: it turns off
        // the m>=16 GEMM arms (qmatvec_gemm + MMQ + fp8/f16/fp4 — all block-scale/foreign
        // numeric configs) so every projection rides the batched-mmvq b16 tier, which is
        // per-(token,row) bit-identical to isolated m=1 decode (gate2 bit-strength PASS at
        // B=12/16, s32+s160, 5090 receipts research/batched-tick-inc3-20260801). Without
        // the exact tier, B>cap stays refused; the env door (MEMRA_DECODE_BATCH_CAP) keeps
        // its old meaning as the non-exact measurement probe.
        let exact16 = b_n > 8 && b_n <= 16 && self.decode_batch_exact16_ok();
        assert!(
            b_n <= cap || exact16,
            "decode_step_batch: B={b_n} > cap {cap} with no exact tier — refused. Either \
             B>16 (there is NO exact kernel class above 16: m>16 crosses GEMM/dp4a numeric \
             configs; the serve scheduler chunks wider concurrency into <=16 groups instead), \
             or some matmul in this checkpoint has no bit-exact b16 kernel — run with \
             MEMRA_EXACT16_WHY=1 to see which tensor and qtype refuses"
        );
        struct ExactScope<'a>(&'a Engine, bool);
        impl Drop for ExactScope<'_> {
            fn drop(&mut self) {
                if self.1 {
                    self.0.set_verify_exact(false);
                }
            }
        }
        let _exact_scope = ExactScope(e, exact16);
        if exact16 {
            e.set_verify_exact(true);
        }
        // gemma4: NO batched arm at any B (per-layer SWA/global geometry, hd-512 MQA globals,
        // weightless V-norm, softcapped head — none of it in the generic body below). This was
        // an assert until 2026-08-07: one serve request panicked the worker, the respawn
        // re-panicked on the queued request, and the process FATALed
        // (research/gemma4-serve-20260807/raw/repro-panic-server-*.log). The worker now routes
        // gemma4 sessions to the per-session eager loop and never calls here; this Err is the
        // defense-in-depth backstop — a future path that reaches it refuses PER-REQUEST
        // instead of killing the process. The eager arm (gemma4_decode_step_h) is the
        // supported decode.
        let batch_program = crate::plan_backend::decode_batch_program(&self.plan);
        if self.is_gemma4_e4b() || batch_program == crate::plan_backend::DecodeBatchProgram::Gemma {
            // BATCHED ARM (lane/gemma-batched, 2026-08-16): the dense 31B gets its own
            // per-session batched walk (gemma4_decode_batch) — DEFAULT ON since the owner
            // flip (MEMRA_GEMMA4_BATCH=0 = the eager kill switch). Same shape law as
            // step35: projections/norms/rope/FFN/head run at m=B (one weight stream, B
            // rows — decode is weight-BW-bound), KV append + fa_decode stay a per-session
            // loop (each session's own len drives its SWA/global view). E4B keeps its
            // dedicated decode; it never enters here.
            if batch_program == crate::plan_backend::DecodeBatchProgram::Gemma
                && !self.is_gemma4_e4b()
                && Self::gemma4_batch_on()
            {
                return self.gemma4_decode_batch(e, tokens, caches, samp, masks, lean);
            }
            return Err(
                "decode_step_batch has no gemma4 arm for this model class (per-layer \
                        swa/global geometry, softcapped head; the dense-31B batched arm is \
                        default-on, MEMRA_GEMMA4_BATCH=0 forces eager) — serve gemma4 on the \
                        eager per-session path"
                    .into(),
            );
        }
        // step35 (lane/step35-batched-decode, 2026-08-08): its OWN batched walk. The generic
        // body below is the uniform Full arm — global n_head, 128-dim rope on every layer, no
        // SWA window, no head-wise gate — which on step35 produced HTTP-200 GARBAGE at c>1
        // (research/step-sku-20260807/raw/b2ab-pre-*.log), so step35 NEVER enters it at any B.
        // `step35_decode_batch_layers` carries the real geometry: per-layer n_head (64/96),
        // partial rope (64 full / 128 SWA, dual base, rope_freqs on FULL only), per-SESSION
        // SWA view offsets from each session's own kvl.len, the separate head-wise gate at
        // m=B, and the sigmoid-router MoE via the same moe_ffn_il_zq8 the eager path uses.
        // MEMRA_STEP35_BATCH=0 = the fail-closed rollback seam. The server caps chunks at
        // B=1; on PP-N the B=1 correctness default also refuses the eager numeric class, while
        // an unsplit deployment can still use its existing eager B=1 route.
        if batch_program == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe {
            if !Self::step35_batch_on() {
                return Err(
                    "step35 batched decode is disabled (MEMRA_STEP35_BATCH=0) — \
                            only a non-PP eager B=1 route remains available"
                        .into(),
                );
            }
            let n_embd = self.cfg.n_embd as usize;
            let eps = self.cfg.rms_eps;
            let mut ph_last = std::time::Instant::now();
            let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
            let pos_d = e.htod_i32(&pos_v)?;
            let x = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
            ph_mark(e, 0, &mut ph_last)?;
            let x = self.step35_decode_batch_layers(
                e,
                x,
                caches,
                &pos_v,
                &pos_d,
                0,
                self.layers.len(),
                &mut ph_last,
            )?;
            let mut hn = e.uninit(b_n * n_embd)?;
            e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
            let logits = e.matmul(&self.output, &hn, b_n)?;
            ph_mark(e, 10, &mut ph_last)?;
            return self.decode_batch_epilogue(
                e,
                caches,
                samp,
                masks,
                lean,
                logits,
                b_n,
                &mut ph_last,
                None,
            );
        }
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;

        // MEMRA_BATCH_PHASE=1: sync-bounded phase accumulation (diagnostics — see header note).
        // Initialized BEFORE the tick-input assembly below so slot 0 covers the HOST side of
        // setup (pos_v/ptr-table builds, embed gather) as well as the H2D sync — the audit-fix
        // lane's Q6 instrumentation gap (research/audit-fixes2-20260805): the old placement
        // started the clock after the assembly, so slot 0 under-reported setup.
        let mut ph_last = std::time::Instant::now();

        // Per-row rope positions (each sequence at its own depth).
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = e.htod_i32(&pos_v)?;

        // Per-step, whole-trunk layer context: state pointer table + arm picks. Under a pp
        // split this call is made once PER STAGE with that stage's engine and range instead
        // (see `batch_layer_ctx`'s doc for why the table cannot be shared across devices).
        let n_layers = self.layers.len();
        let ctx = self.batch_layer_ctx(e, caches, 0, n_layers)?;

        // Embed all B tokens -> x [B, n_embd] (host gather, one H2D).
        let x = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
        ph_mark(e, 0, &mut ph_last)?;

        let x = self.decode_batch_layers(e, x, caches, &ctx, &pos_d, &mut ph_last)?;

        // ---- output norm + lm_head at m=B, one D2H ----
        let mut hn = e.uninit(b_n * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let logits = e.matmul(&self.output, &hn, b_n)?;
        ph_mark(e, 10, &mut ph_last)?;

        self.decode_batch_epilogue(
            e,
            caches,
            samp,
            masks,
            lean,
            logits,
            b_n,
            &mut ph_last,
            pending_out,
        )
    }

    /// PP3/PP4 WAVEFRONT DECODE: split one scheduler tick into up to one wave per stage and drive
    /// the `(wave, stage)` grid through one persistent host worker per non-head stage. The caller
    /// owns the head stage. Explicit boundary messages carry `(wave, slot)` forward and exact
    /// post-rx acknowledgements carry slot credit back; every simultaneous cell owns distinct
    /// request caches and a distinct stage Engine. The arithmetic inside every cell remains the
    /// existing stage-scoped batched program verbatim.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_step_batch_wavefront<'cache>(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&'cache mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        fence: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let batch = tokens.len();
        if batch < 2 || batch != caches.len() {
            return Err("PP wavefront requires matching token/cache batches with B>=2".into());
        }
        if !samp.is_empty() && samp.len() != batch {
            return Err("PP wavefront sampling metadata must be empty or match B".into());
        }
        if !masks.is_empty() && masks.len() != batch {
            return Err("PP wavefront grammar masks must be empty or match B".into());
        }
        let stages = fence.len().saturating_sub(1);
        let rt = crate::pp::PpNRt::get(e)?;
        crate::pp::pp_wave_eligibility(
            stages,
            crate::pp::pp2_overlap(),
            rt.host_bounce_active(),
            rt.repeated_stage_device(),
        )
        .map_err(|reason| -> Box<dyn std::error::Error> { reason.into() })?;
        crate::pp::pp_wave_numeric_eligibility(
            self.cfg
                .hy3
                .as_ref()
                .is_some_and(|hy3| hy3.weight_only_nvfp4),
            Engine::bf16_mmv_on(),
        )
        .map_err(|reason| -> Box<dyn std::error::Error> { reason.into() })?;
        if self.is_gemma4_e4b()
            || crate::plan_backend::decode_batch_program(&self.plan)
                == crate::plan_backend::DecodeBatchProgram::Gemma
        {
            return Err(
                "PP wavefront has no Gemma batched arm; use the model's qualified eager path"
                    .into(),
            );
        }

        let ranges = crate::pp::pp_wave_ranges(batch, stages);
        let max_wave = ranges.iter().map(|(lo, hi)| hi - lo).max().unwrap_or(0);
        let cap = Self::decode_batch_cap();
        let exact16 = max_wave > 8 && max_wave <= 16 && self.decode_batch_exact16_ok();
        if max_wave > cap && !exact16 {
            return Err(format!(
                "PP wavefront B={batch} produces a {max_wave}-row wave above cap {cap} with no exact tier"
            )
            .into());
        }
        let step35_batched = crate::plan_backend::decode_batch_program(&self.plan)
            == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe;
        if step35_batched && !Self::step35_batch_on() {
            return Err(
                "step35 batched decode is disabled; PP wavefront has no correct fallback trunk"
                    .into(),
            );
        }

        if rt.n_stages() != stages {
            return Err(format!(
                "PpNRt stage count {} != PP wavefront stages {stages}",
                rt.n_stages()
            )
            .into());
        }
        let caller_stream = e.stream();
        let primary_context = crate::pp::PrimaryContextRestore::new(e);
        rt.fence_stages_behind(&caller_stream)?;
        let n_embd = self.cfg.n_embd as usize;
        let slot_capacity = max_wave.saturating_mul(n_embd);
        for boundary in 0..stages - 1 {
            rt.prepare_overlap_slots(boundary, slot_capacity)?;
        }

        let _exact_scopes = if exact16 {
            let engines: Vec<&Engine> = (0..stages).map(|stage| rt.engine(stage, e)).collect();
            engines
                .into_iter()
                .map(|engine| engine.exact_scope(true))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut cache_tail: &mut [&'cache mut Cache] = caches;
        let mut waves = Vec::with_capacity(ranges.len());
        for &(lo, hi) in &ranges {
            let width = hi - lo;
            let (wave_caches, tail) = cache_tail.split_at_mut(width);
            cache_tail = tail;
            waves.push(std::sync::Mutex::new(PpDecodeWave {
                row_lo: lo,
                tokens: &tokens[lo..hi],
                caches: wave_caches,
                phase_last: std::time::Instant::now(),
                result: None,
                committed: false,
            }));
        }
        debug_assert!(cache_tail.is_empty());

        let (mut outgoing, mut incoming) = pp_wave_channels(stages - 1);
        let walk_result = std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut workers = Vec::with_capacity(stages - 1);
            for stage in 0..stages - 1 {
                let stage_incoming = if stage == 0 {
                    None
                } else {
                    Some(
                        incoming[stage - 1]
                            .take()
                            .expect("PP wave incoming endpoint already moved"),
                    )
                };
                let stage_outgoing = outgoing[stage]
                    .take()
                    .expect("PP wave outgoing endpoint already moved");
                let wave_states = &waves;
                workers.push(scope.spawn(move || {
                    self.decode_step_batch_wave_worker(
                        e,
                        rt,
                        wave_states,
                        stage,
                        stage_incoming,
                        stage_outgoing,
                        fence,
                        step35_batched,
                    )
                }));
            }

            let head_incoming = incoming[stages - 2]
                .take()
                .expect("PP wave head incoming endpoint already moved");
            let head_result = self.decode_step_batch_wave_head(
                e,
                rt,
                &waves,
                head_incoming,
                fence,
                step35_batched,
                samp,
                masks,
                lean,
            );

            let mut worker_errors = Vec::new();
            let mut worker_panic = None;
            for worker in workers {
                match worker.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        worker_errors.push(error);
                    }
                    Err(payload) => {
                        if worker_panic.is_none() {
                            worker_panic = Some(payload);
                        }
                    }
                }
            }
            if let Some(payload) = worker_panic {
                std::panic::resume_unwind(payload);
            }
            if let Some(error) = worker_errors.iter().find(|error| {
                !error.contains("channel closed") && !error.contains("upstream worker failed")
            }) {
                return Err(error.clone().into());
            }
            match head_result {
                Err(error) => Err(error),
                Ok(()) => match worker_errors.into_iter().next() {
                    Some(error) => Err(error.into()),
                    None => Ok(()),
                },
            }
        });
        let publish_result = if walk_result.is_ok() {
            Some(rt.publish_to(stages - 1, &caller_stream))
        } else {
            None
        };
        let restore_result = primary_context.restore();
        walk_result?;
        if let Some(result) = publish_result {
            result?;
        }
        restore_result?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            eprintln!(
                "[pp-wave] PP{stages} decode wavefront engaged: waves={} max_wave={} (experimental, MEMRA_PP_WAVE=1)",
                ranges.len(),
                max_wave,
            );
        });

        let mut completed = Vec::with_capacity(waves.len());
        for wave in waves {
            let state = wave
                .into_inner()
                .map_err(|_| "PP wave state lock poisoned")?;
            if state.result.is_none() {
                return Err("PP wavefront completed without a head-stage result".into());
            }
            completed.push(state);
        }
        let mut rows = Vec::with_capacity(batch);
        let mut next = Vec::with_capacity(batch);
        for state in &mut completed {
            let (wave_rows, wave_next) = state.result.take().expect("validated PP wave result");
            rows.extend(wave_rows);
            next.extend(wave_next);
            state.committed = true;
        }
        crate::pp::record_pp_wave_tick();
        Ok((rows, next))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_step_batch_wave_worker<'slice, 'cache>(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        waves: &[std::sync::Mutex<PpDecodeWave<'slice, 'cache>>],
        stage: usize,
        incoming: Option<PpWaveIncoming>,
        mut outgoing: PpWaveOutgoing,
        fence: &[usize],
        step35_batched: bool,
    ) -> Result<(), String> {
        let result = (|| -> Result<(), String> {
            if (stage == 0) != incoming.is_none() {
                return Err(format!(
                    "PP wave stage {stage} incoming endpoint shape is invalid"
                ));
            }
            for (wave_index, state) in waves.iter().enumerate() {
                let transfer = match incoming.as_ref() {
                    Some(incoming) => Some(incoming.receive(wave_index)?),
                    None => None,
                };
                let mut state = state
                    .lock()
                    .map_err(|_| "PP wave state lock poisoned".to_string())?;
                self.decode_step_batch_wave_stage(
                    e,
                    rt,
                    &mut state,
                    wave_index,
                    stage,
                    transfer,
                    incoming.as_ref(),
                    &mut outgoing,
                    fence,
                    step35_batched,
                )
                .map_err(|error| error.to_string())?;
            }
            outgoing.finish()
        })();
        if let Err(error) = &result {
            outgoing.publish_worker_error(error);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_step_batch_wave_head<'slice, 'cache>(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        waves: &[std::sync::Mutex<PpDecodeWave<'slice, 'cache>>],
        incoming: PpWaveIncoming,
        fence: &[usize],
        step35_batched: bool,
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (wave_index, state) in waves.iter().enumerate() {
            let transfer = incoming
                .receive(wave_index)
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            let mut state = state.lock().map_err(|_| "PP wave state lock poisoned")?;
            self.decode_step_batch_wave_final(
                e,
                rt,
                &mut state,
                transfer,
                &incoming,
                fence,
                step35_batched,
                samp,
                masks,
                lean,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_step_batch_wave_stage(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        wave: &mut PpDecodeWave<'_, '_>,
        wave_index: usize,
        stage: usize,
        transfer: Option<PpWaveTransfer>,
        incoming: Option<&PpWaveIncoming>,
        outgoing: &mut PpWaveOutgoing,
        fence: &[usize],
        step35_batched: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug_assert!(stage + 1 < fence.len() - 1);
        rt.bind_stage(stage)?;
        let _stage = rt.enter(stage);
        let engine = rt.engine(stage, e);
        wave.phase_last = std::time::Instant::now();
        let width = wave.tokens.len();
        let n_embd = self.cfg.n_embd as usize;
        let payload = width * n_embd;
        let positions: Vec<i32> = wave.caches.iter().map(|cache| cache.pos as i32).collect();
        let positions_d = engine.htod_i32(&positions)?;
        let x = if stage == 0 {
            if transfer.is_some() || incoming.is_some() {
                return Err("PP wave stage 0 received an incoming transfer".into());
            }
            let x = engine.htod(&self.embd.try_gather(n_embd, wave.tokens)?)?;
            ph_mark(engine, 0, &mut wave.phase_last)?;
            x
        } else {
            let transfer = transfer.ok_or("PP wavefront stage has no incoming transfer")?;
            let incoming = incoming.ok_or("PP wavefront stage has no incoming endpoint")?;
            let x = rt.rx(stage - 1, transfer.slot, payload)?;
            incoming
                .acknowledge(transfer)
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            x
        };
        let expected_slot = outgoing
            .prepare(wave_index)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
        let _active = crate::pp::enter_pp_wave_cell();
        let x = if step35_batched {
            self.step35_decode_batch_layers(
                engine,
                x,
                wave.caches,
                &positions,
                &positions_d,
                fence[stage],
                fence[stage + 1],
                &mut wave.phase_last,
            )?
        } else {
            let ctx = self.batch_layer_ctx(engine, wave.caches, fence[stage], fence[stage + 1])?;
            self.decode_batch_layers(
                engine,
                x,
                wave.caches,
                &ctx,
                &positions_d,
                &mut wave.phase_last,
            )?
        };
        let slot = rt.tx_pipelined(stage, &x, payload)?;
        outgoing
            .publish(wave_index, slot, expected_slot)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_step_batch_wave_final(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        wave: &mut PpDecodeWave<'_, '_>,
        transfer: PpWaveTransfer,
        incoming: &PpWaveIncoming,
        fence: &[usize],
        step35_batched: bool,
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let stage = fence.len() - 2;
        rt.bind_stage(stage)?;
        let _stage = rt.enter(stage);
        let engine = rt.engine(stage, e);
        wave.phase_last = std::time::Instant::now();
        let width = wave.tokens.len();
        let n_embd = self.cfg.n_embd as usize;
        let payload = width * n_embd;
        let positions: Vec<i32> = wave.caches.iter().map(|cache| cache.pos as i32).collect();
        let positions_d = engine.htod_i32(&positions)?;
        let x = rt.rx(stage - 1, transfer.slot, payload)?;
        incoming
            .acknowledge(transfer)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
        let _active = crate::pp::enter_pp_wave_cell();
        let x = if step35_batched {
            self.step35_decode_batch_layers(
                engine,
                x,
                wave.caches,
                &positions,
                &positions_d,
                fence[stage],
                fence[stage + 1],
                &mut wave.phase_last,
            )?
        } else {
            let ctx = self.batch_layer_ctx(engine, wave.caches, fence[stage], fence[stage + 1])?;
            self.decode_batch_layers(
                engine,
                x,
                wave.caches,
                &ctx,
                &positions_d,
                &mut wave.phase_last,
            )?
        };
        let mut normalized = engine.uninit(payload)?;
        engine.rms_norm(
            &x,
            self.output_norm.float_data(),
            &mut normalized,
            n_embd,
            width,
            self.cfg.rms_eps,
        )?;
        let logits = engine.matmul(&self.output, &normalized, width)?;
        ph_mark(engine, 10, &mut wave.phase_last)?;
        let hi = wave.row_lo + width;
        let wave_samp = if samp.is_empty() {
            &[][..]
        } else {
            &samp[wave.row_lo..hi]
        };
        let wave_masks = if masks.is_empty() {
            &[][..]
        } else {
            &masks[wave.row_lo..hi]
        };
        wave.result = Some(self.decode_batch_epilogue(
            engine,
            wave.caches,
            wave_samp,
            wave_masks,
            lean,
            logits,
            width,
            &mut wave.phase_last,
            None,
        )?);
        Ok(())
    }

    /// DUAL-ACTIVE PP-2 DECODE (increment 0): split one batch into wave A/B and drive
    /// stage 0(B) from a scoped host walker while this thread drives stage 1(A). Step's
    /// per-layer router readback synchronizes the host, so two CUDA streams issued by one
    /// host thread would remain serial; this mirrors the proven prime PP-2 host schedule.
    ///
    /// This arm is the naked PP-2 default since the 2026-08-11 owner flip (`MEMRA_DUAL_PP`
    /// unset = Auto; `0` is the serial rollback seam). It is fail-closed unless the
    /// double-slot door is open, prewarms both slots, and uses `tx_pipelined` exclusively.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_step_batch_dual(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        fence: &[usize],
        mid: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let b_n = tokens.len();
        assert!(
            b_n >= 1 && b_n == caches.len(),
            "tokens/caches length mismatch"
        );
        let Some(expected_mid) = crate::pp::dual_pp_wave_mid(b_n) else {
            return self.decode_step_batch_ppn(e, tokens, caches, samp, masks, lean, fence);
        };
        if mid != expected_mid {
            return Err(format!(
                "decode_step_batch_dual: worker midpoint {mid} is not the balanced midpoint {expected_mid} for B={b_n}"
            ).into());
        }
        if self.is_gemma4_e4b()
            || crate::plan_backend::decode_batch_program(&self.plan)
                == crate::plan_backend::DecodeBatchProgram::Gemma
        {
            return Err(
                "decode_step_batch_dual has no gemma4 arm — serve gemma4 on the eager \
                        per-session path"
                    .into(),
            );
        }
        assert!(
            samp.is_empty() || samp.len() == b_n,
            "decode_step_batch_dual: samp must be empty or have one entry per row"
        );
        assert!(
            masks.is_empty() || masks.len() == b_n,
            "decode_step_batch_dual: masks must be empty or have one entry per row"
        );

        let cap = Self::decode_batch_cap();
        let max_wave = mid.max(b_n - mid);
        let exact16 = max_wave > 8 && max_wave <= 16 && self.decode_batch_exact16_ok();
        if max_wave > cap && !exact16 {
            return Err(format!(
                "decode_step_batch_dual: B={b_n} waves {mid}+{} exceed per-wave cap {cap} with no exact tier — refused",
                b_n - mid,
            ).into());
        }
        let n_st = fence.len() - 1;
        crate::pp::dual_pp_eligibility(
            n_st,
            crate::pp::pp2_overlap(),
            crate::pp::pp_host_bounce_active(),
        )
        .map_err(|msg| -> Box<dyn std::error::Error> { msg.into() })?;
        let rt = crate::pp::PpNRt::get(e)?;
        assert_eq!(
            rt.n_stages(),
            n_st,
            "PpNRt stage count {} != fence stages {n_st}",
            rt.n_stages()
        );
        let caller_stream = e.stream();
        rt.fence_stages_behind(&caller_stream)?;

        let n_embd = self.cfg.n_embd as usize;
        let wave_cap = mid.max(b_n - mid) * n_embd;
        rt.prepare_overlap_slots(0, wave_cap)?;

        // EXACT-16 is a property of either scheduled wave, not the combined live width. Keep
        // the scope live across both host walkers and set it on both stage-owned Engines.
        let _exact_scopes = if exact16 {
            let engines: Vec<&Engine> = (0..n_st).map(|s| rt.engine(s, e)).collect();
            engines
                .into_iter()
                .map(|engine| engine.exact_scope(true))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let step35_batched = crate::plan_backend::decode_batch_program(&self.plan)
            == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe;
        if step35_batched && !Self::step35_batch_on() {
            return Err(
                "step35 batched decode is disabled (MEMRA_STEP35_BATCH=0) — \
                        dual-active PP-2 decode has no correct fallback trunk"
                    .into(),
            );
        }

        let (tokens_a, tokens_b) = tokens.split_at(mid);
        let (caches_a, caches_b) = caches.split_at_mut(mid);
        let (samp_a, samp_b) = if samp.is_empty() {
            (&[][..], &[][..])
        } else {
            samp.split_at(mid)
        };
        let (masks_a, masks_b) = if masks.is_empty() {
            (&[][..], &[][..])
        } else {
            masks.split_at(mid)
        };

        let (slot_a, ph_a, span_a0) = self.decode_step_batch_dual_stage0(
            e,
            rt,
            tokens_a,
            caches_a,
            fence,
            step35_batched,
            false,
        )?;

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            eprintln!("[dual-pp] dual-active PP-2 decode engaged (naked default since 2026-08-11; two waves)");
        });

        let (out_a, out_b, span_b0, span_b1) = std::thread::scope(
            |scope| -> Result<_, Box<dyn std::error::Error>> {
                let stage0_b = scope.spawn(move || {
                    let staged = self
                        .decode_step_batch_dual_stage0(
                            e,
                            rt,
                            tokens_b,
                            caches_b,
                            fence,
                            step35_batched,
                            true,
                        )
                        .map_err(|err| err.to_string())?;
                    Ok::<_, String>((staged, caches_b))
                });

                let out_a = self.decode_step_batch_dual_stage1(
                    e,
                    rt,
                    slot_a,
                    caches_a,
                    samp_a,
                    masks_a,
                    lean,
                    fence,
                    step35_batched,
                    ph_a,
                    true,
                )?;
                let ((slot_b, ph_b, span_b0), caches_b) = stage0_b
                    .join()
                    .map_err(|_| "dual PP stage-0 wave-B host walker panicked")?
                    .map_err(|err| -> Box<dyn std::error::Error> { err.into() })?;
                if !crate::pp::record_dual_pp_slot_pair(slot_a, slot_b) {
                    return Err(format!(
                        "decode_step_batch_dual: refused: wave A and B both selected boundary slot {slot_a}"
                    ).into());
                }
                let (out_b, span_b1) = self.decode_step_batch_dual_stage1(
                    e,
                    rt,
                    slot_b,
                    caches_b,
                    samp_b,
                    masks_b,
                    lean,
                    fence,
                    step35_batched,
                    ph_b,
                    false,
                )?;
                Ok((out_a, out_b, span_b0, span_b1))
            },
        )?;

        // Wave B is the final producer. One event publishes all last-stage work back to the
        // caller after both epilogues, preserving the ordinary PP-N exit law.
        rt.publish_to(1, &caller_stream)?;
        let (out_a, span_a1) = out_a;
        for (stage, span) in [span_a0, span_a1, span_b0, span_b1].into_iter().enumerate() {
            if let Some((start, end)) = span {
                crate::pp::record_dual_pp_stage_result(stage, start.elapsed_ms(&end));
            }
        }
        let (mut rows, mut next) = out_a;
        rows.extend(out_b.0);
        next.extend(out_b.1);
        Ok((rows, next))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_step_batch_dual_stage0(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        fence: &[usize],
        step35_batched: bool,
        track_overlap: bool,
    ) -> Result<(usize, std::time::Instant, DualPpCudaSpan), Box<dyn std::error::Error>> {
        let b_n = tokens.len();
        let n_embd = self.cfg.n_embd as usize;
        let mut ph_last = std::time::Instant::now();
        rt.bind_stage(0)?;
        let _st0 = rt.enter(0);
        let e0 = rt.engine(0, e);
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = e0.htod_i32(&pos_v)?;
        let x = e0.htod(&self.embd.try_gather(n_embd, tokens)?)?;
        ph_mark(e0, 0, &mut ph_last)?;
        let timing_start = dual_pp_timing_event(e0, "stage0 start event");
        let x = {
            let _overlap = track_overlap.then(crate::pp::enter_dual_pp_stage);
            if step35_batched {
                self.step35_decode_batch_layers(
                    e0,
                    x,
                    caches,
                    &pos_v,
                    &pos_d,
                    fence[0],
                    fence[1],
                    &mut ph_last,
                )?
            } else {
                let ctx = self.batch_layer_ctx(e0, caches, fence[0], fence[1])?;
                self.decode_batch_layers(e0, x, caches, &ctx, &pos_d, &mut ph_last)?
            }
        };
        let timing = timing_start.zip(dual_pp_timing_event(e0, "stage0 end event"));
        let slot = rt.tx_pipelined(0, &x, b_n * n_embd)?;
        Ok((slot, ph_last, timing))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_step_batch_dual_stage1(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        slot: usize,
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        fence: &[usize],
        step35_batched: bool,
        mut ph_last: std::time::Instant,
        track_overlap: bool,
    ) -> Result<((Vec<Vec<f32>>, Vec<Option<u32>>), DualPpCudaSpan), Box<dyn std::error::Error>>
    {
        let b_n = caches.len();
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        rt.bind_stage(1)?;
        let _st1 = rt.enter(1);
        let e1 = rt.engine(1, e);
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = e1.htod_i32(&pos_v)?;
        let x = rt.rx(0, slot, b_n * n_embd)?;
        let timing_start = dual_pp_timing_event(e1, "stage1 start event");
        let x = {
            let _overlap = track_overlap.then(crate::pp::enter_dual_pp_stage);
            if step35_batched {
                self.step35_decode_batch_layers(
                    e1,
                    x,
                    caches,
                    &pos_v,
                    &pos_d,
                    fence[1],
                    fence[2],
                    &mut ph_last,
                )?
            } else {
                let ctx = self.batch_layer_ctx(e1, caches, fence[1], fence[2])?;
                self.decode_batch_layers(e1, x, caches, &ctx, &pos_d, &mut ph_last)?
            }
        };
        let timing = timing_start.zip(dual_pp_timing_event(e1, "stage1 end event"));
        let mut hn = e1.uninit(b_n * n_embd)?;
        e1.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let logits = e1.matmul(&self.output, &hn, b_n)?;
        ph_mark(e1, 10, &mut ph_last)?;
        Ok((
            self.decode_batch_epilogue(
                e1,
                caches,
                samp,
                masks,
                lean,
                logits,
                b_n,
                &mut ph_last,
                None,
            )?,
            timing,
        ))
    }

    /// THE BATCHED PP-N STEP (pp2-batch increment 2, 2026-08-06): the batched tick split
    /// across `fence.len()-1` stages, each stage running ONLY its own layer range through
    /// ITS OWN engine and stream, with a `[B, n_embd]` boundary activation between them.
    /// The batched twin of `decode_step_h_ppn`, and the #1 item on the PP-2 serving bill —
    /// without it a >VRAM SKU (Step-3.7-Flash: 105 GB, fits only across two cards) serves
    /// SINGLE-STREAM only, because the batched path was the one loop with no stage split.
    ///
    /// STRUCTURE (mirrors the eager arm exactly, so the two stay comparable):
    ///   stage 0        `rt.enter(0)` -> per-stage pos_d + embed -> range -> `rt.tx`
    ///   middle stages  `rt.rx` -> per-stage pos_d -> range -> `rt.tx`
    ///   last stage     `rt.rx` -> per-stage pos_d -> range -> output_norm + lm_head ->
    ///                  the batched serving epilogue (masks, device sample, lean park)
    ///
    /// FOUR THINGS ARE PER-STAGE, and each is per-stage for a measured reason:
    ///
    /// 1. THE ENGINE (`rt.engine(s, e)`). Not just for the remote device: `Engine` owns
    ///    lazily-grown stable-pointer scratch pools (`fa_part_pool`, `fa_vf16_scratch`,
    ///    `argmax_partials`) that are single-stream-safe BY DESIGN. Two stage streams
    ///    through one Engine is the shared-scratch race the pp2 lane hit (2026-08-02
    ///    nondeterministic all-logits divergence, 35% flake). `PpNRt::build` already gives
    ///    every stage s>0 its own Engine even on the primary device, so honouring
    ///    `rt.engine(s, e)` here is what scopes the pools per stage — the batched path
    ///    allocates MORE of that scratch than the eager one (fa at m=B), so this is the
    ///    load-bearing half of the trap's mitigation, not an inherited nicety.
    ///
    /// 2. THE POINTER TABLE (`batch_layer_ctx(es, caches, lo, hi)`). See [`BatchLayerCtx`]:
    ///    it holds DEVICE ADDRESSES of that range's cache state, uploaded through that
    ///    stage's engine. One step-wide table on the primary would put every stage's kernel
    ///    arguments in stage-0's HBM — a peer read per pointer fetch, the exact cliff this
    ///    whole lane exists to remove.
    ///
    /// 3. `pos_d` (the M2 pipelining law, learned on the eager arm): each stage uploads its
    ///    own copy of the step's per-row positions on ITS stream, so the buffer is
    ///    allocated, consumed and freed on one stream. A shared stage-0 `pos_d` freed at fn
    ///    return breaks under deferred readback — the free enqueues on stream 0 while later
    ///    stages still dereference it.
    ///
    /// 4. THE HEAD + EPILOGUE run on the LAST stage: `output_norm`/`output` were uploaded
    ///    through the last stage's engine by the sharded loader (`hybrid.rs`: `e_head =
    ///    layer_engine(e, n_trunk, n_trunk-1)`), and `cache.last_logits_dev` must be
    ///    allocated where the logits are.
    ///
    /// EXACTNESS: PP-N adds ZERO deviation. Each stage runs the SAME kernels on the SAME
    /// bytes in the same order — the split only moves where the residual is materialized,
    /// and the boundary is a straight f32 copy (dtod same-device / `cudaMemcpyPeerAsync`
    /// cross-device, no conversion). So batched PP-N must be BIT-IDENTICAL to single-device
    /// batched at the same B, in both placement orders. Gate: `decode-batch-gate --mode
    /// pp` (logit-dump, both orders) — the batched analogue of the eager arm's 48 steps x
    /// 248,320 f32 logits with zero differing bits.
    ///
    /// The B=1 fast path is NOT taken here (its condition already excludes an open door):
    /// it routes through `decode_layers_eager` whole-trunk on one engine, which is exactly
    /// the unsplit walk. B=1 under the door rides this function's B=1 case instead — the
    /// same trade the eager arm's own ppn step makes, and the reason the pp2 lane measured
    /// B=1 door-open at 0.854x (the lost fusion chain), not a cliff.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_step_batch_ppn(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        fence: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let b_n = tokens.len();
        assert!(
            b_n >= 1 && b_n == caches.len(),
            "tokens/caches length mismatch"
        );
        // gemma4: same no-arm refusal as the unsplit body (see decode_step_batch), Err not
        // assert — a request must never kill the worker process.
        if self.is_gemma4_e4b()
            || crate::plan_backend::decode_batch_program(&self.plan)
                == crate::plan_backend::DecodeBatchProgram::Gemma
        {
            return Err(
                "decode_step_batch_ppn has no gemma4 arm — serve gemma4 on the eager \
                        per-session path"
                    .into(),
            );
        }
        // Same width policy as the unsplit body — the stage split changes WHERE kernels run,
        // never WHICH tier admits the width. Duplicated deliberately rather than hoisted:
        // the exact-16 scope must wrap the whole multi-stage walk (`set_verify_exact` is
        // per-Engine state read at dispatch on every stage), so it has to be established
        // here, and a shared helper returning a guard would have to own `e` plus the flag.
        let cap = Self::decode_batch_cap();
        let exact16 = b_n > 8 && b_n <= 16 && self.decode_batch_exact16_ok();
        assert!(
            b_n <= cap || exact16,
            "decode_step_batch_ppn: B={b_n} > cap {cap} with no exact tier — refused"
        );
        let rt = crate::pp::PpNRt::get(e)?;
        let n_st = fence.len() - 1;
        assert_eq!(
            rt.n_stages(),
            n_st,
            "PpNRt stage count {} != fence stages {n_st}",
            rt.n_stages()
        );
        // #87 REVERSE PUBLICATION (lane/pp2spec-crash): order every stage stream behind
        // the caller before this body's first stage allocation can reuse a pool block
        // whose queued primary-stream consumer has not read it yet. Anatomy:
        // `PpNRt::fence_stages_behind`. (This body dtoh+syncs its own logits, but its
        // PP-mode callers interleave with the spec verify's device-resident outputs in
        // the same worker, so the entry fence is the uniform law, not an optimization.)
        rt.fence_stages_behind(&e.stream())?;
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let payload = b_n * n_embd;

        // EXACT-16 SCOPE, PER STAGE ENGINE: `verify_exact` is per-Engine state (an AtomicBool
        // on the Engine the dispatch reads), and each stage runs through a DIFFERENT Engine —
        // so setting it on the primary alone would leave stages 1..N-1 dispatching the m>=16
        // GEMM/MMQ arms while stage 0 used the exact b16 tier. That is a silent per-stage
        // numeric split (the failure this tier exists to prevent), so the flag is set on
        // every stage engine and cleared on all of them at scope exit.
        let _exact_scopes = if exact16 {
            let engines: Vec<&Engine> = (0..n_st).map(|s| rt.engine(s, e)).collect();
            engines
                .into_iter()
                .map(|engine| engine.exact_scope(true))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut ph_last = std::time::Instant::now();

        // B=1 PER-STAGE FAST PATH (measured 2026-08-06, PRO 6000 pair). The unsplit body's
        // b1_fast guard includes `pp_cuts().is_none()`, so opening the pp door dropped every
        // solo session off the m=1 FUSION chain (cross-layer add+norm+q8_1, fused SwiGLU,
        // lever 1's gate+up dual) and onto the batched m=1 walk. Cost, arm A vs arm C at B=1:
        // 208.5 vs 177.3 tok/s = -15.0% — and NOT a split cost, since arm B (stages=2 on ONE
        // card) pays the same 177, and the prior lane's `MEMRA_PP_SHARD=0` batched-body B=1
        // was 178.5. It was the fusion chain going missing, on the config the Step SKU serves
        // solo requests from.
        //
        // `decode_layers_eager(lo, hi)` is ALREADY range-scoped and is exactly what the eager
        // ppn arm (`decode_step_h_ppn`) calls per stage, so B=1 rides the same per-stage
        // structure: same engines, same streams, same [1, n_embd] boundary slots, same
        // stage-owned caches. Only the trunk kernels differ, and they differ identically to
        // how they differ off-door. Exactness is therefore the SAME accepted decode-config FP
        // class the unsplit b1_fast lever already carries (strict gate1 PASSes with it on,
        // FAILs with it off at maxdiff 1.591e-1) — which is why the pp gate pins
        // `set_b1_fast(false)`: with it on, the B=1 reference and the split arm would
        // legitimately sit on opposite sides of that gap and the bit-identity arm would
        // report a fake stage-split failure.
        //
        // Step3.5/Step3.7 are an exception (lane/cx-b1fix, 2026-08-10): their B>1 route is
        // `step35_decode_batch_layers`, and the live scheduler may move a session from B=1
        // to B>1. The eager/fused class and that batched class produce different greedy bytes,
        // so selecting the eager arm at B=1 made output depend on load history. Keep one
        // numeric class for this model family: Step35 always takes its stage-scoped batched
        // trunk at every width. The live transition gate in step35-b2-geometry-gate pins it.
        // Qwen35-MoE is the second exception (lane/cx-q35bug, 2026-08-12): on the Q35
        // sellgate workload the eager-B1 -> batched-B2 transition changed emitted token ids and
        // selected EOS at tokens 15/17/25. Keep that family on this generic batched trunk at B=1
        // too; dense Qwen35 retains the measured eager fast path.
        let b1_stage_fast = b_n == 1
            && Self::b1_fast_on()
            && self.b1_fast_plan_eligible()
            && !self.is_gemma4_e4b()
            && crate::plan_backend::decode_batch_program(&self.plan)
                == crate::plan_backend::DecodeBatchProgram::Generic
            && !self
                .plan
                .trunk_operations()
                .contains(&memra_gguf::model_plan::OperationKind::SwiGluOaiActivation)
            && !e.verify_exact_on();
        // step35 (lane/step35-batched-decode, 2026-08-08): B>1 rides its OWN stage-scoped
        // batched walk (`step35_decode_batch_layers`) — the generic `decode_batch_layers`
        // remains OFF-LIMITS for this arch at every B (its uniform geometry produced the
        // b2ab HTTP-200 garbage: research/step-sku-20260807/raw/b2ab-pre-*.log). Since
        // lane/cx-b1fix, B=1 also takes this walk: a Step35 PP-N session must not change
        // numeric class when live decode width changes. The refusal below guards the
        // rollback residue; under PP-N, disabling the only correct trunk makes Step35
        // requests fail closed instead of falling back to the eager class.
        let step35_batched = crate::plan_backend::decode_batch_program(&self.plan)
            == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe;
        if step35_batched && !Self::step35_batch_on() {
            return Err(
                "step35 batched decode is disabled (MEMRA_STEP35_BATCH=0) — \
                        PP-N Step35 decode is unavailable because eager B=1 is a different \
                        numeric class"
                    .into(),
            );
        }
        // Hoisted: `caches[0].pos` as a value argument alongside `caches[0]` as `&mut` in one
        // call is a borrow conflict; `pos` is Copy and the epilogue is what advances it.
        let pos0 = if b1_stage_fast { caches[0].pos } else { 0 };

        // ---- STAGE 0: embed (the table lives with stage 0) + layers [0, fence[1]) + TX ----
        let mut slot = {
            let _st0 = rt.enter(0);
            let e0 = rt.engine(0, e);
            let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
            let pos_d = e0.htod_i32(&pos_v)?;
            let x = e0.htod(&self.embd.try_gather(n_embd, tokens)?)?;
            ph_mark(e0, 0, &mut ph_last)?;
            let x = if b1_stage_fast {
                self.decode_layers_eager(e0, x, fence[0], fence[1], &pos_d, pos0, caches[0])?
            } else if step35_batched {
                self.step35_decode_batch_layers(
                    e0,
                    x,
                    caches,
                    &pos_v,
                    &pos_d,
                    fence[0],
                    fence[1],
                    &mut ph_last,
                )?
            } else {
                let ctx = self.batch_layer_ctx(e0, caches, fence[0], fence[1])?;
                self.decode_batch_layers(e0, x, caches, &ctx, &pos_d, &mut ph_last)?
            };
            rt.tx(0, &x, payload)?
            // x + pos_d + ctx.ptr_table drop here: freed stream-ordered on stage-0's stream.
        };

        // ---- MIDDLE STAGES: RX boundary s-1 -> range -> TX boundary s ----
        for s in 1..n_st - 1 {
            let _st = rt.enter(s);
            let es = rt.engine(s, e);
            let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
            let pos_d = es.htod_i32(&pos_v)?;
            let x = rt.rx(s - 1, slot, payload)?;
            let x = if b1_stage_fast {
                self.decode_layers_eager(es, x, fence[s], fence[s + 1], &pos_d, pos0, caches[0])?
            } else if step35_batched {
                self.step35_decode_batch_layers(
                    es,
                    x,
                    caches,
                    &pos_v,
                    &pos_d,
                    fence[s],
                    fence[s + 1],
                    &mut ph_last,
                )?
            } else {
                let ctx = self.batch_layer_ctx(es, caches, fence[s], fence[s + 1])?;
                self.decode_batch_layers(es, x, caches, &ctx, &pos_d, &mut ph_last)?
            };
            slot = rt.tx(s, &x, payload)?;
        }

        // ---- LAST STAGE: RX + final range + head + the batched serving epilogue ----
        let _stl = rt.enter(n_st - 1);
        let el = rt.engine(n_st - 1, e);
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = el.htod_i32(&pos_v)?;
        let x = rt.rx(n_st - 2, slot, payload)?;
        let x = if b1_stage_fast {
            self.decode_layers_eager(el, x, fence[n_st - 1], fence[n_st], &pos_d, pos0, caches[0])?
        } else if step35_batched {
            self.step35_decode_batch_layers(
                el,
                x,
                caches,
                &pos_v,
                &pos_d,
                fence[n_st - 1],
                fence[n_st],
                &mut ph_last,
            )?
        } else {
            let ctx = self.batch_layer_ctx(el, caches, fence[n_st - 1], fence[n_st])?;
            self.decode_batch_layers(el, x, caches, &ctx, &pos_d, &mut ph_last)?
        };

        let mut hn = el.uninit(payload)?;
        el.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let logits = el.matmul(&self.output, &hn, b_n)?;
        ph_mark(el, 10, &mut ph_last)?;

        self.decode_batch_epilogue(
            el,
            caches,
            samp,
            masks,
            lean,
            logits,
            b_n,
            &mut ph_last,
            None,
        )
    }

    /// The mHC (HyperConnections) batched decode arm (lane/glm53-batched-decode,
    /// 2026-08-28). DEFAULT ON since 2026-08-31, on the hbatch-battery box receipts
    /// (research/glm53-flash-bringup-20260827/hbatch-battery-20260831/): interleaved x3
    /// ladder on the 3-card serving shape — ON wins every rung c>=2 (aggregate 1.095x at
    /// c=2 up to 1.214x at c=12, plateau ~1.20x from c=8), B=1 cost -0.30%, TTFT under
    /// load ON <= OFF at every rung, 36/36 concurrent tapes byte-identical to solo (incl.
    /// ON-solo == OFF-solo), admission clean to c=20, loop-law 0/448. `MEMRA_HYPER_BATCH=0`
    /// is the rollback seam (eager per-session decode). Any OTHER value REFUSES LOUD at
    /// first use — a mis-typed serving switch must not silently pick a path.
    pub fn hyper_batch_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| match std::env::var("MEMRA_HYPER_BATCH").as_deref() {
            Ok("0") => false,
            Err(_) | Ok("1") => true,
            Ok(v) => panic!(
                "MEMRA_HYPER_BATCH={v:?} is not a recognized value (want unset/0 = eager \
                 per-session decode, 1 = batched mHC decode chunks) — refusing to guess a \
                 serving path"
            ),
        })
    }

    /// The mHC batched-decode width cap, DERIVED rather than inherited (owner challenge,
    /// 2026-08-28: "why 8?"). The audit of every term that grows with B found exactly ONE
    /// numeric-class knee, and it is not memory, not the boundary payload, and not the
    /// per-session mixer loop:
    ///
    ///   * per-session mixers (KDA + MLA/kpool): step latency grows ~linearly in B on that
    ///     segment — a throughput term, no correctness wall at any width; per-session state
    ///     is ~104 MB MLA latent at 8k ctx + trivial KDA, so memory does not bind either.
    ///   * hc glue: block-per-token kernels (grid.y chunked at 65535 — B=64 is 256 rows on
    ///     `hc_post`), and the hc-mix GEMM runs per-row m=1 by construction (`pre_exact`).
    ///   * lm_head via `matmul_decode_exact`: per-row exact at every m (float per-token
    ///     m=1; quant b-tier to 16, grid.y=m mmvq above — re-reads, not rounding).
    ///   * MoE router (`router_gemv`): m-invariant at every t under defaults; sigmoid
    ///     top-k and routed-expert execution are per-token programs at any t.
    ///   * MoE SHARED EXPERT — THE BINDER: `hybrid_forward.rs` shexp trio,
    ///     `verify_t = t > 1 && t < PRIME_MIN_T`. At t >= 16 gate/up/down cross from
    ///     `matmul_decode_exact` onto the plain prefill matmul (cuBLASLt n-dependent for
    ///     float; the m>16 MMQ/GEMM block-scale class for quant) and per-row bit-identity
    ///     vs the isolated t=1 chain breaks — measured, not argued: the gate's knee probe
    ///     at B=16 mismatches from the first tick (`31-KNEE-b16-forced.log`), B=15 is green.
    ///
    /// So the exact tier is `1..=PRIME_MIN_T-1` = 15. Widening to 32/64 needs a
    /// decode-exact shexp arm for t >= 16 — which must NOT be flipped inside the shared
    /// `!prefill` branch, because step35's MoESD target forward (t up to 256) rides the
    /// same branch and its banked spec receipts pin the current bytes. Named follow-up.
    /// `MEMRA_DECODE_BATCH_CAP` narrows only, never widens past the knee.
    pub fn hyper_batch_cap() -> usize {
        let knee = crate::hybrid_forward::PRIME_MIN_T - 1;
        std::env::var("MEMRA_DECODE_BATCH_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|c| c.clamp(1, knee))
            .unwrap_or(knee)
    }

    /// THE mHC BATCHED DECODE STEP (lane/glm53-batched-decode, 2026-08-28): B sessions,
    /// one walk over the `[B, streams, n_embd]` stream state. This is the production
    /// blocker this lane lifts — with every batched entry refusing the hc residual,
    /// GLM-5.3-Flash served SINGLE-STREAM ONLY at any `MEMRA_MAX_SESSIONS`.
    ///
    /// The trunk is `hyper_batch_range_decode` (hybrid_forward.rs — see its doc for the
    /// batched/per-session/decode-exact shape law); the exit is `hyper::collapse` +
    /// output_norm + a DECODE-EXACT lm_head at m=B (each row's head program is the m=1
    /// program its solo step runs — `matmul_decode_exact`); the tail is the SAME
    /// `decode_batch_epilogue` every other batched arm serves (masks, device sampling,
    /// lean park, pos bump), so the serving contract is shared rather than duplicated.
    ///
    /// CONCURRENCY SHAPES: sessions may sit at DIFFERENT positions with different KDA
    /// recurrent states and different kpool index planes — each row carries its own
    /// single-position buffer and its own cache, which is what the gate's staggered-depth
    /// arm pins. One token per session per tick (pure decode); there is no mixed
    /// prefill/decode shape at this entry by construction. Width: B <= 8, the per-row
    /// exactness tier — there is NO exact16 tier here (`decode_batch_exact16_ok` refuses
    /// the Mla/Kda mixers), and the MoE router's fixed per-row program is only the decode
    /// arm below PRIME_MIN_T; wider concurrency is the scheduler's job to chunk.
    ///
    /// EXACTNESS BAR AND GATE: row b of a B-row tick is BIT-IDENTICAL (full logits, every
    /// step) to session b decoding alone through `decode_step_hyper`, including B=1 — one
    /// numeric class at every live width, the step35/Q35 class-crossing law. Gate:
    /// `glm5-hyper-batch-gate`, red-armed with a swapped-row and a wrong-cache-slot
    /// mutation (cross-session contamination is the silent-corruption failure mode).
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub(crate) fn decode_step_batch_hyper(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let topology = *self
            .hyper
            .as_ref()
            .ok_or("decode_step_batch_hyper on a model with no HyperConnections topology")?;
        if !Self::hyper_batch_on() {
            return Err(
                "mHC batched decode is disabled (MEMRA_HYPER_BATCH unset/0, the fail-closed \
                 default until serving-box receipts land) — serve hyper-connection sessions \
                 on the eager per-session path, or set MEMRA_HYPER_BATCH=1"
                    .into(),
            );
        }
        // THE DOOR IS NOT ON THIS WALK, AND A SILENT NO-OP IS THE SAME FAILURE CLASS AS A
        // VACUOUS GATE. Take 13's serving A/B (2026-09-03) passed `MEMRA_GLM5_DECODE_GRAPH=1`
        // through the serve script and got SIX boots with zero `[glm5-decode-graph]` lines of any
        // kind, refusals included, while the same binary's gate harness armed the door fine
        // (`replays=564`). The missing conjunct is the walk: `MEMRA_GLM5_DECODE_GRAPH` is wired
        // into `hybrid_forward::hyper_range_decode`, the per-session SERIAL hc walk that
        // `decode_step_hyper` / `decode_step_hyper_ppn` run, and serving with
        // `MEMRA_HYPER_BATCH=1` routes every session through THIS batched walk instead, including
        // B=1. So the A/B measured the door's absence and read flat, which is the correct number
        // for the wrong question.
        //
        // Extending capture to the batched walk itself is still a separate lane (its trunk is
        // `decode_batch_layers`, with its own per-row geometry). Until then the honest behaviour
        // is to say so, once, rather than let a serving log's silence be read as a refusal.
        //
        // `MEMRA_HYPER_BATCH_SOLO=1` CHANGES WHAT IS HONEST HERE, so this line is now keyed on
        // it. At B=1 that door delegates to `hyper_range_decode`, which is exactly the walk the
        // graph door is wired into, so the door DOES engage and printing "NOT ON THIS PATH"
        // would contradict the `[glm5-decode-graph] engaged` lines a few entries below it in the
        // same log. Measured on the 2x B200 pair 2026-09-03: with the solo door armed, serving
        // printed `engaged dev=0 stage=[0, 24) runs=6 captured_layers=18` and `engaged dev=1
        // stage=[24, 45) runs=6 captured_layers=16` — the first time this door has engaged in
        // serving. A stale warning next to a working door is the same failure class the warning
        // was written to prevent.
        if crate::glm5_decode_graph_on() {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                if crate::hyper_batch_solo_on() {
                    eprintln!(
                        "[glm5-decode-graph] reachable via MEMRA_HYPER_BATCH_SOLO=1: this session \
                         enters the BATCHED hc walk, which delegates to the serial walk \
                         (hyper_range_decode) at B=1, and that is the walk the door is wired \
                         into. Expect [glm5-decode-graph] engaged/eager lines. At B>1 the batched \
                         trunk runs and the door is still not on that path."
                    );
                } else {
                    eprintln!(
                        "[glm5-decode-graph] NOT ON THIS PATH: MEMRA_GLM5_DECODE_GRAPH is on (the \
                         default since 2026-09-04; =0 disarms it) but this session decodes \
                         through the BATCHED hc walk (MEMRA_HYPER_BATCH=1), and \
                         the door is wired into the per-session serial walk (hyper_range_decode) \
                         only. The door will not engage, and will not refuse either, for as long \
                         as the batched walk is in use. Set MEMRA_HYPER_BATCH_SOLO=1 to reach it \
                         at B=1, unset MEMRA_HYPER_BATCH to price the serial walk, or read this \
                         line as the reason a serving log carries no [glm5-decode-graph] lines."
                    );
                }
            });
        }
        let b_n = tokens.len();
        if b_n == 0 || b_n != caches.len() {
            return Err("decode_step_batch_hyper: tokens/caches length mismatch".into());
        }
        // Width: Err, never assert — a request must not kill the worker (the gemma4
        // process-FATAL lesson). The cap is DERIVED, not inherited — see `hyper_batch_cap`.
        let cap = Self::hyper_batch_cap();
        if b_n > cap {
            return Err(format!(
                "decode_step_batch_hyper: B={b_n} > cap {cap} — at t >= PRIME_MIN_T (16) \
                 the MoE shared-expert trio crosses from matmul_decode_exact onto the \
                 prefill matmul class (cuBLASLt n-dependent / m>16 MMQ-GEMM), so per-row \
                 bit-identity vs isolated decode breaks at exactly B=16 (gate knee probe \
                 31-KNEE-b16-forced). Every other term is width-safe; widening needs a \
                 decode-exact shexp arm for t>=16 (named follow-up — the shared !prefill \
                 branch also carries step35 MoESD bytes and must not be flipped). Chunk \
                 wider concurrency into <={cap} groups"
            )
            .into());
        }
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;

        // M2 ppN door — the batched hc walk owns its own stage split, exactly as the
        // serial hc walks do (forward_hyper's note). Loud refusal on an unqualified
        // pipeline rewrite, never a single-engine walk over stage-sharded weights.
        if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
            if !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::Pipeline) {
                return Err("pipeline rewrite is not qualified for this ModelPlan".into());
            }
            return self.decode_step_batch_hyper_ppn(
                e, tokens, caches, samp, masks, lean, &topology, &fence,
            );
        }

        let mut ph_last = std::time::Instant::now();
        let pos_rows = Self::hyper_batch_pos_rows(e, caches)?;
        let embedded = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
        let mut x = crate::hyper::expand(e, &topology, &embedded, b_n, n_embd)?;
        ph_mark(e, 0, &mut ph_last)?;
        x = self.hyper_batch_range_decode(
            e,
            &topology,
            x,
            0,
            self.layers.len(),
            &pos_rows,
            caches,
        )?;
        let logits = self.hyper_batch_head_logits(e, &topology, &x, b_n, n_embd, eps)?;
        ph_mark(e, 10, &mut ph_last)?;
        self.decode_batch_epilogue(
            e,
            caches,
            samp,
            masks,
            lean,
            logits,
            b_n,
            &mut ph_last,
            None,
        )
    }

    /// Per-row single-position device buffers, uploaded through THIS engine (under a pp
    /// split, the stage's engine — the per-stage pos_d law). The mixers take a t=1 `pos_d`
    /// exactly as their solo step does, so each session's row is a one-element buffer, not
    /// a shared [B] table.
    fn hyper_batch_pos_rows(
        e: &Engine,
        caches: &[&mut Cache],
    ) -> Result<Vec<CudaSlice<i32>>, Box<dyn std::error::Error>> {
        caches.iter().map(|c| e.htod_i32(&[c.pos as i32])).collect()
    }

    /// The batched hc trunk exit: mean/gated collapse + output_norm + DECODE-EXACT lm_head.
    /// `matmul_decode_exact` at m=B runs each row through the m=1 head program the serial
    /// `hyper_decode_tail` runs (float: per-token m=1 cuBLASLt; quant: the per-(token,row)
    /// bit-exact batched mmvq tier), so the head cannot be the arm that breaks per-row
    /// identity. Returns device logits `[B, n_vocab]` for the shared epilogue.
    fn hyper_batch_head_logits(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        x: &CudaSlice<f32>,
        b_n: usize,
        n_embd: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let collapsed =
            crate::hyper::collapse(e, topology, self.hyper_head.as_ref(), x, b_n, n_embd)?;
        let mut hn = e.uninit(b_n * n_embd)?;
        e.rms_norm(
            &collapsed,
            self.output_norm.float_data(),
            &mut hn,
            n_embd,
            b_n,
            eps,
        )?;
        e.matmul_decode_exact(&self.output, &hn, b_n)
    }

    /// ppN twin of `decode_step_batch_hyper`: the batched hc tick as N stage subgraphs,
    /// mirroring `decode_step_hyper_ppn` (per-stage engine, per-stage pos uploads, a
    /// `[B, streams, n_embd]` boundary payload) and `decode_step_batch_ppn` (the #87 entry
    /// fence, head + epilogue on the LAST stage's engine, where the loader put the head and
    /// where `cache.last_logits_dev` must live). No exact16 scope (no exact16 tier here)
    /// and no B=1 fast path (one numeric class at every width).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_step_batch_hyper_ppn(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        topology: &crate::hyper::HyperTopology,
        fence: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let b_n = tokens.len();
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let payload = b_n * topology.streams * n_embd;
        let mut ph_last = std::time::Instant::now();

        // The same-stream seam (MEMRA_PP_STREAMS=0 also disables the sharded loader, so
        // nothing is remote): one engine, boundary copies between ranges — the shape the
        // serial hc ppn walk uses for this knob.
        if crate::pp::pp2_streams_off() {
            let pos_rows = Self::hyper_batch_pos_rows(e, caches)?;
            let embedded = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
            let mut x = crate::hyper::expand(e, topology, &embedded, b_n, n_embd)?;
            ph_mark(e, 0, &mut ph_last)?;
            x = self
                .hyper_batch_range_decode(e, topology, x, fence[0], fence[1], &pos_rows, caches)?;
            for s in 1..fence.len() - 1 {
                let boundary_tx = e.clone_dtod(&x)?;
                let boundary_rx = e.clone_dtod(&boundary_tx)?;
                x = self.hyper_batch_range_decode(
                    e,
                    topology,
                    boundary_rx,
                    fence[s],
                    fence[s + 1],
                    &pos_rows,
                    caches,
                )?;
            }
            let logits = self.hyper_batch_head_logits(e, topology, &x, b_n, n_embd, eps)?;
            ph_mark(e, 10, &mut ph_last)?;
            return self.decode_batch_epilogue(
                e,
                caches,
                samp,
                masks,
                lean,
                logits,
                b_n,
                &mut ph_last,
                None,
            );
        }

        let rt = crate::pp::PpNRt::get(e)?;
        let n_st = fence.len() - 1;
        assert_eq!(
            rt.n_stages(),
            n_st,
            "PpNRt stage count {} != fence stages {n_st}",
            rt.n_stages()
        );
        // #87 reverse publication (see decode_step_batch_ppn): order every stage stream
        // behind the caller before this body's first stage allocation.
        rt.fence_stages_behind(&e.stream())?;

        // ---- STAGE 0: embed + expand (no weights) + layers [0, fence[1]) + TX ----
        let mut slot = {
            let _st0 = rt.enter(0);
            let e0 = rt.engine(0, e);
            let pos_rows = Self::hyper_batch_pos_rows(e0, caches)?;
            let embedded = e0.htod(&self.embd.try_gather(n_embd, tokens)?)?;
            let x = crate::hyper::expand(e0, topology, &embedded, b_n, n_embd)?;
            ph_mark(e0, 0, &mut ph_last)?;
            let x = self
                .hyper_batch_range_decode(e0, topology, x, fence[0], fence[1], &pos_rows, caches)?;
            rt.tx(0, &x, payload)?
        };

        // ---- MIDDLE STAGES: RX -> range -> TX ----
        for s in 1..n_st - 1 {
            let _st = rt.enter(s);
            let es = rt.engine(s, e);
            let pos_rows = Self::hyper_batch_pos_rows(es, caches)?;
            let x = rt.rx(s - 1, slot, payload)?;
            let x = self.hyper_batch_range_decode(
                es,
                topology,
                x,
                fence[s],
                fence[s + 1],
                &pos_rows,
                caches,
            )?;
            slot = rt.tx(s, &x, payload)?;
        }

        // ---- LAST STAGE: RX + final range + collapse/head + the shared epilogue ----
        let _stl = rt.enter(n_st - 1);
        let el = rt.engine(n_st - 1, e);
        let pos_rows = Self::hyper_batch_pos_rows(el, caches)?;
        let x = rt.rx(n_st - 2, slot, payload)?;
        let x = self.hyper_batch_range_decode(
            el,
            topology,
            x,
            fence[n_st - 1],
            fence[n_st],
            &pos_rows,
            caches,
        )?;
        let logits = self.hyper_batch_head_logits(el, topology, &x, b_n, n_embd, eps)?;
        ph_mark(el, 10, &mut ph_last)?;
        self.decode_batch_epilogue(
            el,
            caches,
            samp,
            masks,
            lean,
            logits,
            b_n,
            &mut ph_last,
            None,
        )
    }

    /// Build the per-step layer context for layers `[lo, hi)`: the device state-pointer
    /// table plus the step's arm picks. See [`BatchLayerCtx`] for why this is RANGE-scoped
    /// (the table holds device addresses and must be uploaded through the engine whose
    /// device runs those layers).
    ///
    /// Table layout is unchanged from the whole-trunk version — `lin_base`/`attn_base` are
    /// still indexed by ABSOLUTE layer id, so `decode_batch_layers`' body indexes them
    /// exactly as the old inline loop did. Only layers in `[lo, hi)` contribute entries; the
    /// rest stay `None`, which is a loud `expect` if a range ever reads outside its own.
    pub(crate) fn batch_layer_ctx(
        &self,
        e: &Engine,
        caches: &[&mut Cache],
        lo: usize,
        hi: usize,
    ) -> Result<BatchLayerCtx, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let head_dim = cfg.head_dim_k as usize;
        // Per-step STATE POINTER TABLE (one H2D): for every linear layer, [conv x B]
        // [ssm_in x B][ssm_out x B] device addresses. The batched state kernels read their
        // sequence's pointer from these arrays — states stay per-cache (no pooling refactor),
        // yet conv/prep/scan collapse from 3xB launches per layer to 3. Rebuilt every step
        // because the ssm ping-pong swaps pointers host-side after each scan.
        // INCREMENT 2 (2026-08-01): the SAME table now also carries, for every FULL-attn
        // layer, [k0,v0,k1,v1,...] cache base addresses — the z-batched seqs append and
        // seqs fa_decode kernels read their sequence's cache through it (the MoE
        // expert-table pattern), collapsing 2xB launches per attn layer to 2.
        let mut lin_base: Vec<Option<usize>> = vec![None; self.layers.len()];
        let mut attn_base: Vec<Option<usize>> = vec![None; self.layers.len()];
        let mut ptrs: Vec<u64> = Vec::new();
        {
            use cudarc::driver::DevicePtr;
            let s = &e.gpu.stream();
            for il in lo..hi {
                match &self.layers[il].mixer {
                    Mixer::Linear(_) => {
                        lin_base[il] = Some(ptrs.len());
                        for c in caches.iter() {
                            let rl = c.recur[il].as_ref().unwrap();
                            let (p, _g) = rl.conv_state.device_ptr(s);
                            ptrs.push(p);
                        }
                        for c in caches.iter() {
                            let rl = c.recur[il].as_ref().unwrap();
                            let (p, _g) = rl.ssm_state.device_ptr(s);
                            ptrs.push(p);
                        }
                        for c in caches.iter() {
                            let rl = c.recur[il].as_ref().unwrap();
                            let (p, _g) = rl.ssm_state_alt.device_ptr(s);
                            ptrs.push(p);
                        }
                    }
                    Mixer::Full(_) => {
                        attn_base[il] = Some(ptrs.len());
                        for c in caches.iter() {
                            let kvl = c.kv[il].as_ref().unwrap();
                            let (pk, _g) = kvl.k.device_ptr(s);
                            let (pv, _g2) = kvl.v.device_ptr(s);
                            ptrs.push(pk);
                            ptrs.push(pv);
                        }
                    }
                    Mixer::Mla(_) => crate::hybrid::mla_path_unimplemented("batched PP decode"),
                    Mixer::Kda(_) => crate::hybrid::kda_path_unimplemented("batched decode layer"),
                }
            }
        }
        let ptr_table = if ptrs.is_empty() {
            None
        } else {
            Some(e.htod_u64(&ptrs)?)
        };

        // INCREMENT 2 arm picks (per STEP — t_kv is layer-invariant within a tick):
        // - seqs APPEND: format-only condition (per-row program is t_kv-independent);
        //   default flash module only (fp8-KV rides the per-seq g-module path).
        // - seqs FA: every row must take the v4 eager arm at ITS OWN t_kv AND all rows
        //   must share ONE fa_split_keys rung (the rows-twins' straddle law) — a rung
        //   crossing inside the batch keeps the per-seq loop for that step, so each
        //   sequence always executes the exact program its isolated run would.
        //
        // The picks are t_kv-driven, and t_kv is layer-INVARIANT within a step, so every
        // stage of a pp split independently computes the SAME arms from the same `caches`
        // — a stage cannot silently take a different program than its unsplit self.
        let t_kvs: Vec<usize> = caches.iter().map(|c| c.pos + 1).collect();
        let t_kv_max = *t_kvs.iter().max().unwrap();
        let sp0 = crate::fa_split_keys(t_kvs[0], cfg.n_head_kv as usize);
        let seqs_fa = t_kvs.iter().all(|&t| crate::fa_seqs_eligible(t, head_dim))
            && t_kvs
                .iter()
                .all(|&t| crate::fa_split_keys(t, cfg.n_head_kv as usize) == sp0);

        Ok(BatchLayerCtx {
            lin_base,
            attn_base,
            ptr_table,
            t_kvs,
            t_kv_max,
            sp0,
            seqs_fa,
            lo,
            hi,
        })
    }

    /// THE PP SEAM (pp2-batch increment 1, 2026-08-06): run the batched trunk over layers
    /// `[ctx.lo, ctx.hi)`, entering with a materialized `[B, n_embd]` residual and exiting
    /// with the range's final residual materialized. The batched twin of
    /// `decode_layers_eager` — the eager arm has had this seam since M1-PP2 and every ppN
    /// stage calls it; the batched body had no equivalent, which is why every later PP-2
    /// increment (and spec-over-PP2, whose verify is a batched T=K+1 forward) waited on this
    /// extraction (`research/pp2-hardening-20260806/PROGRESS.md` bill item 1).
    ///
    /// SINGLE-DEVICE SEMANTICS ARE UNCHANGED BY CONSTRUCTION: the body is the old
    /// `for (il, layer) in self.layers.iter().enumerate()` loop moved verbatim, with `for il
    /// in ctx.lo..ctx.hi` as the header and the per-step invariants (`ptr_table`, arm picks,
    /// `t_kv`) read from `ctx` instead of enclosing locals. At `lo=0, hi=n_layers` — every
    /// call today — the launch sequence is identical, so the exactness contract in this
    /// module's header carries over untouched rather than needing a re-proof.
    ///
    /// UNLIKE the eager seam, this one is NOT yet stage-callable: `caches` is `&mut [&mut
    /// Cache]` mutated in place (KV `len` bumps, ssm ping-pong swaps), and `pos_d`/`x` come
    /// from the caller's device. Wiring a stage split means per-stage `pos_d` + a boundary
    /// `[B, n_embd]` transfer around this call, which is the NEXT increment. The seam exists
    /// so that increment is a call-site change, not a 250-line surgery.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_batch_layers(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        caches: &mut [&mut Cache],
        ctx: &BatchLayerCtx,
        pos_d: &CudaSlice<i32>,
        ph_last: &mut std::time::Instant,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let b_n = caches.len();
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let (lin_base, attn_base) = (&ctx.lin_base, &ctx.attn_base);
        let ptr_table = &ctx.ptr_table;
        let (seqs_fa, sp0, t_kv_max) = (ctx.seqs_fa, ctx.sp0, ctx.t_kv_max);
        debug_assert_eq!(
            ctx.t_kvs.len(),
            b_n,
            "ctx built for a different batch width"
        );

        for il in ctx.lo..ctx.hi {
            let layer = &self.layers[il];
            // ---- attn_norm + q8_1 quantize, batched (B rows) ----
            let anorm = layer.attn_norm.float_data();
            let mut xn = e.uninit(b_n * n_embd)?;
            e.rms_norm(&x, anorm, &mut xn, n_embd, b_n, eps)?;
            let (hq, hd) = e.quantize_q8_1(&xn, b_n, n_embd)?;

            // ---- mixer ----
            let mixed: CudaSlice<f32> = match &layer.mixer {
                Mixer::Mla(_) => crate::hybrid::mla_path_unimplemented("batched PP decode"),
                Mixer::Kda(_) => crate::hybrid::kda_path_unimplemented("batched decode"),
                Mixer::Full(fa) => {
                    let geometry = cfg.full_attention_geometry_at(il as u32);
                    let n_head = geometry.n_head as usize;
                    let n_head_kv = geometry.n_head_kv as usize;
                    let head_dim = geometry.head_dim_k as usize;
                    let rope_dims = geometry.n_rot as usize;
                    let rope_base = geometry.rope_base;
                    let scale = geometry.attention_scale();
                    // Batched projections: one weight read serves all B rows. At B=1 the
                    // QKV triple fuses into ONE launch (rig-native decode increment 1 —
                    // bit-identical per (tensor,row), RIG-NATIVE-DECODE.md); B>1 and
                    // non-NVFP4 trunks keep the three singles.
                    let (qf, mut k, v) =
                        match e.matmul_nvfp4_fused3(&fa.wq, &fa.wk, &fa.wv, &hq, &hd, b_n)? {
                            Some(t) => t,
                            None => (
                                e.matmul_pre(&fa.wq, &hq, &hd, &xn, b_n)?,
                                e.matmul_pre(&fa.wk, &hq, &hd, &xn, b_n)?,
                                e.matmul_pre(&fa.wv, &hq, &hd, &xn, b_n)?,
                            ),
                        };

                    let gated =
                        geometry.attention_gate == memra_gguf::config::AttentionGateKind::FusedQ;
                    let (mut q, gate) = if gated {
                        let mut qs = e.uninit(b_n * n_head * head_dim)?;
                        let mut gs = e.uninit(b_n * n_head * head_dim)?;
                        e.q_gate_split(&qf, &mut qs, &mut gs, head_dim, n_head, b_n)?;
                        (qs, Some(gs))
                    } else {
                        (qf, None)
                    };

                    // QK-norm over B*n_head rows, rope with per-row positions.
                    let mut qn = e.uninit(b_n * n_head * head_dim)?;
                    e.rms_norm_opt(&q, fa.q_norm_w(), &mut qn, head_dim, b_n * n_head, eps)?;
                    q = qn;
                    let mut kn = e.uninit(b_n * n_head_kv * head_dim)?;
                    e.rms_norm_opt(&k, fa.k_norm_w(), &mut kn, head_dim, b_n * n_head_kv, eps)?;
                    k = kn;
                    e.rope_neox(
                        &mut q, pos_d, head_dim, rope_dims, n_head, b_n, rope_base, 1.0,
                    )?;
                    e.rope_neox(
                        &mut k, pos_d, head_dim, rope_dims, n_head_kv, b_n, rope_base, 1.0,
                    )?;
                    ph_mark(e, 1, ph_last)?;

                    // INCREMENT 2 (2026-08-01): the per-seq (append, attend) launch train
                    // becomes two phases. Phase A appends all B rows (one z-batched launch,
                    // or the per-seq loop on the seam/fp8 path); phase B attends all B
                    // sequences (one blockIdx.z launch + one combine on the batched arm —
                    // which also reads q / writes attn at row offsets, killing the per-seq
                    // q/a dtod copies — or the per-seq loop when any row is outside the v4
                    // arm / a split rung crosses inside the batch). Caches are disjoint per
                    // sequence, so the phase split leaves every row's math untouched.
                    let q_dim = n_head * head_dim;
                    let mut attn = e.uninit(b_n * q_dim)?;
                    // ---- phase A: KV append (all B rows) ----
                    let (kdk, kdv, ktb, vtb) = {
                        let kvl = caches[0].kv[il].as_ref().unwrap();
                        (kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes)
                    };
                    let base = attn_base[il].expect("full layer missing from pointer table");
                    let table = ptr_table.as_ref().expect("pointer table missing");
                    let kv_view = table.slice(base..base + 2 * b_n);
                    e.append_kv_quantized_seqs(&k, &v, &kv_view, pos_d, b_n, kdk, kdv, ktb, vtb)?;
                    for cache in caches.iter_mut() {
                        let kvl = cache.kv[il].as_mut().unwrap();
                        debug_assert_eq!(kvl.len, cache.pos, "kv len / pos out of lockstep");
                        kvl.len += 1;
                    }
                    ph_mark(e, 2, ph_last)?;
                    // ---- phase B: attention (all B sequences) ----
                    if seqs_fa {
                        let (ktb, vtb) = {
                            let kvl = caches[0].kv[il].as_ref().unwrap();
                            (kvl.k_tok_bytes, kvl.v_tok_bytes)
                        };
                        let base = attn_base[il].expect("full layer missing from pointer table");
                        let table = ptr_table.as_ref().expect("pointer table missing");
                        let kv_view = table.slice(base..base + 2 * b_n);
                        e.fa_decode_batch_seqs_v4(
                            &q, &kv_view, pos_d, &mut attn, head_dim, n_head, n_head_kv, b_n,
                            t_kv_max, scale, sp0, ktb, vtb,
                        )?;
                        ph_mark(e, 4, ph_last)?;
                    } else {
                        for (bi, cache) in caches.iter_mut().enumerate() {
                            let kvl = cache.kv[il].as_mut().unwrap();
                            let t_kv = kvl.len;
                            let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
                            let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
                            // The fallback keeps one FA launch per distinct KV view, but Q and
                            // attention already live in packed row-major buffers. Pass those row
                            // views directly; only the arithmetic-free materialization copies go.
                            let q_row = q.slice(bi * q_dim..(bi + 1) * q_dim);
                            let mut a_row = attn.slice_mut(bi * q_dim..(bi + 1) * q_dim);
                            e.fa_decode_kvmod_view(
                                &q_row,
                                &k_view,
                                &v_view,
                                &mut a_row,
                                head_dim,
                                n_head,
                                n_head_kv,
                                t_kv,
                                scale,
                                kvl.k_tok_bytes,
                                kvl.v_tok_bytes,
                                false,
                            )?;
                            ph_mark(e, 4, ph_last)?;
                        }
                    }

                    // Output gate (element-wise — batches whole) + o-proj at m=B.
                    let attn_g = match &gate {
                        Some(g) => {
                            let n = b_n * q_dim;
                            let mut gsig = e.uninit(n)?;
                            e.sigmoid(g, &mut gsig, n)?;
                            let mut ag = e.uninit(n)?;
                            e.mul(&attn, &gsig, &mut ag, n)?;
                            ag
                        }
                        None => attn,
                    };
                    let o = {
                        // AWQ (memra#253): o_proj carries its own per-input-channel scale.
                        let __wpqs = e.pre_quant_scaled(
                            &attn_g,
                            fa.wo_pqs.as_ref(),
                            fa.wo.in_features(),
                            b_n,
                        )?;
                        e.matmul(&fa.wo, __wpqs.as_ref().unwrap_or(&attn_g), b_n)
                    }?;
                    ph_mark(e, 5, ph_last)?;
                    o
                }
                Mixer::Linear(la) => {
                    // v2 (the B-scaling fix): the GDN mixer's PROJECTIONS carry the layer's
                    // weight mass — batch them at m=B so wqkv/gate/beta/alpha/ssm_out stream
                    // ONCE per step instead of once per sequence. Only the recurrent state ops
                    // (fused conv ring, gdn prep, gdn scan) stay per-seq — they are state-bound
                    // micro-kernels, not weight readers. Composition unchanged vs v1 (matmul_pre
                    // == fused2 per (tensor,row); _bN mmvq per-row == m=1): same numeric config.
                    let geometry = la.geometry;
                    let d_state = geometry.key_head_dim as usize;
                    let num_k = geometry.key_heads as usize;
                    let num_v = geometry.value_heads as usize;
                    let d_conv = geometry.conv_kernel as usize;
                    let key_dim = d_state * num_k;
                    let value_dim = geometry.value_head_dim as usize * num_v;
                    let conv_dim = key_dim * 2 + value_dim;
                    let gdn_scale = 1.0 / (d_state as f32).sqrt();

                    // ---- batched projections (the weight win) ----
                    // At B=1 the mixer quartet fuses into ONE launch (rig-native decode
                    // increment 2 — bit-identical per (tensor,row), RIG-NATIVE-DECODE.md);
                    // B>1 and non-NVFP4 trunks keep the four singles.
                    let (qkv_mixed, z, beta_raw, alpha) = match e.matmul_nvfp4_fused4(
                        &la.wqkv,
                        &la.wqkv_gate,
                        &la.ssm_beta,
                        &la.ssm_alpha,
                        &hq,
                        &hd,
                        b_n,
                    )? {
                        Some(t) => t,
                        None => (
                            e.matmul_pre(&la.wqkv, &hq, &hd, &xn, b_n)?,
                            e.matmul_pre(&la.wqkv_gate, &hq, &hd, &xn, b_n)?,
                            e.matmul_pre(&la.ssm_beta, &hq, &hd, &xn, b_n)?,
                            e.matmul_pre(&la.ssm_alpha, &hq, &hd, &xn, b_n)?,
                        ),
                    };
                    ph_mark(e, 6, ph_last)?;

                    // ---- batched recurrent state ops (3 launches for all B sequences) ----
                    let base = lin_base[il].expect("linear layer missing from pointer table");
                    let table = ptr_table.as_ref().expect("pointer table missing");
                    let conv_view = table.slice(base..base + b_n);
                    let in_view = table.slice(base + b_n..base + 2 * b_n);
                    let out_view = table.slice(base + 2 * b_n..base + 3 * b_n);
                    let mut conv_outs = e.uninit(b_n * conv_dim)?;
                    e.ssm_conv1d_fused_decode_b(
                        &qkv_mixed,
                        &conv_view,
                        la.ssm_conv1d.float_data(),
                        &mut conv_outs,
                        conv_dim,
                        d_conv,
                        b_n,
                    )?;
                    let mut q_l2 = e.uninit(b_n * value_dim)?;
                    let mut k_l2 = e.uninit(b_n * value_dim)?;
                    let mut v_gd = e.uninit(b_n * value_dim)?;
                    let mut beta_b = e.uninit(b_n * num_v)?;
                    let mut g_log = e.uninit(b_n * num_v)?;
                    e.gdn_prep_decode_b(
                        &conv_outs,
                        &beta_raw,
                        &alpha,
                        la.ssm_dt.float_data(),
                        la.ssm_a.float_data(),
                        &mut q_l2,
                        &mut k_l2,
                        &mut v_gd,
                        &mut beta_b,
                        &mut g_log,
                        d_state,
                        num_v,
                        num_k,
                        key_dim,
                        eps,
                        conv_dim,
                        b_n,
                    )?;
                    let mut o_all = e.uninit(b_n * value_dim)?;
                    e.gdn_scan_s128_batched(
                        &q_l2, &k_l2, &v_gd, &g_log, &beta_b, &in_view, &out_view, &mut o_all,
                        num_v, b_n, gdn_scale,
                    )?;
                    // ping-pong: scan wrote each seq's alt buffer; swap host handles (the
                    // NEXT step's table rebuild picks up the new canonical pointers).
                    for cache in caches.iter_mut() {
                        let rl = cache.recur[il].as_mut().unwrap();
                        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
                    }
                    ph_mark(e, 7, ph_last)?;

                    // ---- batched gated norm + out-projection ----
                    let o = if e.uses_q8_1_fast(&la.ssm_out) {
                        let (gq, gd) = e.gated_rmsnorm_q8_1(
                            &o_all,
                            la.ssm_norm.float_data(),
                            &z,
                            d_state,
                            b_n * num_v,
                            eps,
                        )?;
                        let g0 = e.zeros(0)?;
                        e.matmul_pre(&la.ssm_out, &gq, &gd, &g0, b_n)?
                    } else {
                        let mut gn = e.uninit(b_n * value_dim)?;
                        e.gated_rmsnorm(
                            &o_all,
                            la.ssm_norm.float_data(),
                            &z,
                            &mut gn,
                            d_state,
                            b_n * num_v,
                            eps,
                        )?;
                        e.matmul(&la.ssm_out, &gn, b_n)?
                    };
                    ph_mark(e, 8, ph_last)?;
                    o
                }
            };

            // ---- residual add + post_attn_norm + FFN, batched ----
            let pnorm = layer.post_attn_norm.float_data();
            let mut x1 = e.uninit(b_n * n_embd)?;
            let mut z = e.uninit(b_n * n_embd)?;
            e.add_rms_norm(&x, &mixed, pnorm, &mut x1, &mut z, n_embd, b_n, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    ffn_down_pqs,
                } => {
                    // v1 covers the SiLU family; M3's swigluoai clamp rides a scaled epilogue
                    // (m=1 fused tier) — batched M3 lands with the batched-fusion pass.
                    assert!(
                        !self
                            .plan
                            .trunk_operations()
                            .contains(&memra_gguf::model_plan::OperationKind::SwiGluOaiActivation,),
                        "decode_step_batch v1: M3 swigluoai FFN not yet batched"
                    );
                    let n_ff = ffn_gate.out_features();
                    let (zq, zd) = e.quantize_q8_1(&z, b_n, n_embd)?;
                    // REFUTED ARM (lane/q27-deepdive, 2026-08-05): fusing this gate+up pair
                    // into `matmul_q8_fused2_t` (the fused2_b8 tier) measured FLAT-TO-NEGATIVE
                    // at the serving tick — bench c=8 213.1/213.8, 213.9/214.4, 214.4/213.5
                    // (sign flips) and serve c=8 paired mean −0.20% over 3 passes. Mechanism:
                    // unlike m=1 (where the pair is 128 of 1015 launches in a 7.67%-gap tick),
                    // the c=8 tick is 73.2% one weight-bound kernel class with launch cost
                    // already hidden — halving 128 launches of ~28k buys nothing. The m=1 arm
                    // in `matmul_pre_dual_noscale` (+0.94%) stays; this call site keeps the two
                    // launches. Kernel + fused2_b8 wrapper retained: kernel-check gates it at
                    // m=5/8 and matmul_q8_fused2_t serves the verify tier. Receipts:
                    // research/q27-deepdive-20260805/ (lever3-bench-*, serve-points.jsonl).
                    let g = e.matmul_pre(ffn_gate, &zq, &zd, &z, b_n)?;
                    let u = e.matmul_pre(ffn_up, &zq, &zd, &z, b_n)?;
                    let mut act = e.uninit(b_n * n_ff)?;
                    e.silu_mul(&g, &u, &mut act, b_n * n_ff)?;
                    // AWQ (memra#253): the f32 activation exists here, so the
                    // per-input-channel scale is applied BEFORE the q8 quantize — both the
                    // quantized operand and the f32 fallback then carry it.
                    if let Some(pqs) = ffn_down_pqs.as_ref() {
                        e.apply_pre_quant_scale(
                            &mut act,
                            pqs.float_data(),
                            ffn_down.in_features(),
                            b_n,
                        )?;
                    }
                    let (aq, ad) = e.quantize_q8_1(&act, b_n, n_ff)?;
                    e.matmul_pre(ffn_down, &aq, &ad, &act, b_n)?
                }
                crate::hybrid::Ffn::Moe(m) => {
                    // b_n==1: feed the zq8 seam (orndecode B2, see decode.rs twin). Wider
                    // ticks keep None — the dev arm quantizes per-token views there and the
                    // shexp pair rides the batched matmul, so there is nothing to share.
                    if b_n == 1 {
                        let zq8 = e.quantize_q8_1(&z, 1, n_embd)?;
                        self.moe_ffn_il_zq8(e, m, &z, Some(&zq8), b_n, il as u16)?
                    } else {
                        self.moe_ffn_il_zq8(e, m, &z, None, b_n, il as u16)?
                    }
                }
            };
            // next-layer input x = x1 + ffn_out (batched element-wise add)
            let mut x2 = e.uninit(b_n * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, b_n * n_embd)?;
            x = x2;
            ph_mark(e, 9, ph_last)?;
        }
        Ok(x)
    }

    /// Rollback seam for the step35 batched decode arm (lane/step35-batched-decode,
    /// 2026-08-08). Default ON; `MEMRA_STEP35_BATCH=0` caps serving at B=1 and makes the
    /// batched bodies return Err. Since lane/cx-b1fix, PP-N also refuses the eager B=1
    /// numeric class, so the seam disables PP-N Step35 decode rather than serving unstable
    /// bytes. Also the b2geo35 gate's CANARY seam — the live assertions must fail under it.
    pub fn step35_batch_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("MEMRA_STEP35_BATCH").as_deref() != Ok("0"))
    }

    /// THE step35 BATCHED LAYER WALK (lane/step35-batched-decode, 2026-08-08): B sequences
    /// share one pass over layers `[lo, hi)` with the REAL step35 geometry — the arm that
    /// kills the B=1 pin (34 tok/s aggregate FLAT across c=1..8, round-robin serialized;
    /// research/step-sku-20260807 §4) without re-opening the b2ab garbage hole (the generic
    /// `decode_batch_layers` ran uniform n_head/full-width rope/no window/no gate over
    /// step35 weights and returned HTTP-200 garbage at c>1).
    ///
    /// SHAPE — batched where the weights are, per-session where the state is:
    ///   * attn_norm + quantize + wq/wk/wv/attn_gate projections + q/k norms + rope + head
    ///     gate + wo + residual/post-norm + FFN all run at m=B: ONE weight stream serves B
    ///     rows (decode is weight-BW-bound; this is the entire win).
    ///   * KV append + fa_decode stay a per-session loop — the SWA window makes each
    ///     session's KV view a function of ITS OWN `kvl.len` (`off = len-win` when past the
    ///     window), and the z-batched seqs kernels take one shared t_kv/rung, not per-row
    ///     offsets. This is the same shape as `decode_batch_layers`' per-seq fallback arm,
    ///     and it costs launches, not weight bandwidth (KV is per-session state either way).
    ///
    /// PER-LAYER GEOMETRY (the five mechanisms that make the generic body wrong here, all
    /// from `step35_geom`/cfg): n_head 64 full / 96 SWA (wq/wo/attn_gate widths per layer),
    /// partial rope (n_rot 64 full / 128 SWA), dual base (5e6/1e4) + `rope_freqs` factors
    /// on FULL layers only, SWA window 512 with per-SESSION view offsets, and the separate
    /// head-wise `attn_gate` (one pre-sigmoid scalar per (token, head), input = the
    /// post-attn_norm hidden, applied before wo).
    ///
    /// EXACTNESS (the isolation contract, decode-batch-gate gate2's bar): every kernel here
    /// is row-independent at m=B or per-session:
    ///   * `rms_norm`/`add_rms_norm`/`quantize_q8_1`/`attn_head_gate`/activations: per-row
    ///     programs, grid over rows — row bi's bytes are the 1-row call's bytes.
    ///   * projections via `matmul_pre` at m=2..8: Q8_0/Q6_K-class rides the b2/b4/b8
    ///     batched-mmvq tier (bit-identical per (token,row) to m=1 mmvq); IQ4_XS — this
    ///     SKU's trunk class — has no mmvq/batched kernel, so BOTH m=1 decode and the m=B
    ///     walk ride `qmatvec_iq4_XS_dp4a` (grid (out_f, m): each column IS the m=1 dp4a
    ///     program). Same class at every width = the decode-parity law by construction.
    ///   * `rope_neox2` takes per-row positions (tok = row / n_heads) — row bi rotates at
    ///     ITS pos with the layer's (n_rot, base, ff), same bits as its solo call.
    ///   * per-session append/fa_decode_kvmod: literally the eager arm's calls on that
    ///     session's own cache and views.
    ///   * MoE (`moe_ffn_il_zq8` at t=B): the router is per-column decode-exact at
    ///     t < PRIME_MIN_T (m=1 program per column), sigmoid routing + expert dispatch are
    ///     per-token — a session's experts are a function of its own row only.
    ///     The known eager-vs-batched FP gap is why PP-N Step35 deliberately serves THIS walk at
    ///     B=1 too: the scheduler can change width during a session, so one numeric class must
    ///     cover every live width. `b2geo35` pins static widths and an explicit B=1 -> B>1
    ///     transition under live defaults.
    ///
    /// STAGE-SCOPED FROM BIRTH: `[lo, hi)` + caller-supplied engine/pos_d, so
    /// `decode_step_batch_ppn` calls it per stage (per-stage engine, per-stage pos_d, the
    /// #87 entry fence and boundary slots unchanged) — the pp2-batch seam lesson.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step35_decode_batch_layers(
        &self,
        e: &Engine,
        x: CudaSlice<f32>,
        caches: &mut [&mut Cache],
        positions: &[i32],
        pos_d: &CudaSlice<i32>,
        lo: usize,
        hi: usize,
        ph_last: &mut std::time::Instant,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.step35_decode_rows_layers(e, x, caches, positions, pos_d, None, lo, hi, ph_last)
    }

    /// Diagnostic generalization of the serving walk: `row_to_cache[r]` names the session
    /// whose KV row is consumed by hidden row `r`. Serving passes `None`, preserving the
    /// identity mapping and its launch sequence. The MoESD harness passes B groups of gamma
    /// consecutive rows so each session's verify columns append causally while projections and
    /// MoE dispatch see the full B*gamma target width.
    #[allow(clippy::too_many_arguments)]
    fn step35_decode_rows_layers(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        caches: &mut [&mut Cache],
        positions: &[i32],
        pos_d: &CudaSlice<i32>,
        row_to_cache: Option<&[usize]>,
        lo: usize,
        hi: usize,
        ph_last: &mut std::time::Instant,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let b_n = row_to_cache.map_or(caches.len(), |rows| rows.len());
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        if !self.uses_sliding_gated_moe_program() {
            return Err(
                "sliding-gated-MoE batch rewrite requires its canonical operation class".into(),
            );
        }
        if b_n == 0 || x.len() != b_n * n_embd || positions.len() != b_n || pos_d.len() != b_n {
            return Err(format!(
                "step35 row mapping shape mismatch: rows={b_n} x={} host_pos={} device_pos={} \
                 n_embd={n_embd}",
                x.len(),
                positions.len(),
                pos_d.len(),
            )
            .into());
        }
        if row_to_cache.is_some_and(|rows| rows.iter().any(|&ci| ci >= caches.len())) {
            return Err("step35 row mapping names a missing cache".into());
        }
        let cache_index = |row: usize| row_to_cache.map_or(row, |rows| rows[row]);
        let has_rank_local_tp = self.layers[lo..hi].iter().any(|layer| {
            matches!(
                &layer.mixer,
                Mixer::Full(fa)
                    if fa
                        .step_tp_qkv
                        .as_ref()
                        .is_some_and(|tp| tp.attention.is_some())
            )
        });
        // MEMRA_STEP_TP_BATCH=1: the t-row batched step-TP walk — per layer, ONE t-grid
        // attn norm + ONE weight-amortized QKV over all rows, per-row attention on its
        // OWN session cache (the unmodified t=1 program via the col-select door), the
        // o_proj deferred and joined once per layer, one t-grid residual norm, one
        // t-row routed-expert sweep with a single combine per rank, and the exact t=1
        // shexp per row. Every kernel is the per-row-exact twin from the verify walk's
        // pedigree, so each session's greedy output is bit-equal to the layer-major-b1
        // replay below. Rows chunk at the tcol width (8).
        static TPB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let tp_batch =
            *TPB.get_or_init(|| std::env::var("MEMRA_STEP_TP_BATCH").as_deref() == Ok("1"));
        if b_n > 1
            && b_n <= 8
            && has_rank_local_tp
            && tp_batch
            && crate::tp::step_tp_qkv_fused_enabled().unwrap_or(false)
            && self.layers[lo..hi].iter().all(|layer| {
                matches!(
                    &layer.mixer,
                    Mixer::Full(fa)
                        if fa.step_tp_qkv.as_ref().is_some_and(|tp| {
                            tp.attention.is_some() && tp.runtime.native_p2p()
                        })
                )
            })
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[step-tp-batch-trow] rows={b_n} execution=t-row-batched \
                     attention=per-session-rank-local kv_cache=per-session-distributed \
                     exactness=per-row-b1-twins performance_claim=false"
                );
            });
            let mut row_positions = Vec::with_capacity(b_n);
            for &position in positions {
                row_positions.push(e.htod_i32(&[position])?);
            }
            let mut x_t = x;
            let mut h_row = e.uninit(n_embd)?;
            let mut mixed_row = e.uninit(n_embd)?;
            let t = b_n;
            let mut pos_staged = false;
            for il in lo..hi {
                let layer = &self.layers[il];
                let mut h_t = e.uninit(t * n_embd)?;
                e.rms_norm(&x_t, layer.attn_norm.float_data(), &mut h_t, n_embd, t, eps)?;
                if !self.step35_verify_qkv_precompute(e, il, &h_t, t)? {
                    return Err(format!(
                        "step-tp-batch layer {il} lost tcol eligibility mid-walk \
                         (weights/doors changed under a live batch)"
                    )
                    .into());
                }
                // Per-session t-row fa: when every row's session clears the dcw doors,
                // the per-row pass stashes q+gate (append still lands per session) and
                // ONE table-kernel launch per rank attends all rows.
                let fa_rows =
                    self.step35_batch_fa_rows_precheck(caches, cache_index, positions, il)?;
                let mut next = e.uninit(t * n_embd)?;
                let mut deferred: Vec<usize> = Vec::new();
                let mut fa_deferred: Vec<usize> = Vec::new();
                // FULL t-row attention pass (rope/append + fa + combine + o_proj join in
                // 3 launches/rank): skips the per-row loop entirely. The device counters
                // advance in-kernel; mirror the HOST cache bookkeeping exactly as the
                // per-row tail would (staged/committed txn + local len + lazy mirror).
                static RR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let rope_rows_on =
                    *RR.get_or_init(|| std::env::var("MEMRA_ROPE_ROWS").as_deref() != Ok("0"));
                static RRL: std::sync::OnceLock<Option<(usize, usize)>> =
                    std::sync::OnceLock::new();
                let rr_layer = *RRL.get_or_init(|| {
                    let v = std::env::var("MEMRA_ROPE_ROWS_LAYER").ok()?;
                    if let Some((a, b)) = v.split_once('-') {
                        Some((a.parse().ok()?, b.parse().ok()?))
                    } else {
                        let x: usize = v.parse().ok()?;
                        Some((x, x))
                    }
                });
                let rr_this = rr_layer.is_none_or(|(a, b)| il >= a && il <= b);
                let full_mixed = if fa_rows && rope_rows_on && rr_this {
                    self.step35_batch_rope_fa_pass(
                        e,
                        il,
                        caches,
                        cache_index,
                        positions,
                        t,
                        !pos_staged,
                    )?
                } else {
                    None
                };
                if let Some(mixed_t) = &full_mixed {
                    pos_staged = true;
                    #[allow(clippy::needless_range_loop)]
                    // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                    for r in 0..t {
                        let ci = cache_index(r);
                        let cache = &mut *caches[ci];
                        let tp_kv = cache.tp_kv[il]
                            .as_mut()
                            .expect("precheck verified the distributed cache");
                        let transaction = tp_kv.begin_transaction()?;
                        let Mixer::Full(fa) = &self.layers[il].mixer else {
                            return Err("step-tp-batch expects full attention".into());
                        };
                        let tp = fa
                            .step_tp_qkv
                            .as_ref()
                            .ok_or("step-tp-batch lost its TP state")?;
                        let empty: [CudaSlice<f32>; 0] = [];
                        tp.runtime.append_tp_kv_transaction_inner(
                            tp_kv,
                            transaction,
                            &empty,
                            &empty,
                            1,
                            true,
                        )?;
                        tp.runtime
                            .commit_tp_kv_transaction_external(tp_kv, transaction, 1)?;
                        if let Some(local) = cache.kv[il].as_mut() {
                            local.len = positions[r] as usize + 1;
                            if !crate::tp::len_mirror_lazy_on() {
                                let _main = e.gpu.enter_main()?;
                                e.set_i32_one(&mut local.len_d, local.len as i32)?;
                            }
                        }
                    }
                    let o_out = mixed_t.len() / t;
                    {
                        for r in 0..t {
                            e.dtod_copy_view(
                                &mixed_t.slice(r * o_out..(r + 1) * o_out),
                                &mut mixed_row,
                            )?;
                            let mut x_row = e.uninit(n_embd)?;
                            e.dtod_copy_view(&x_t.slice(r * n_embd..(r + 1) * n_embd), &mut x_row)?;
                            let (x1, ffn_out) = self
                                .residual_norm_ffn(e, layer, &x_row, &mixed_row, n_embd, il, eps)?;
                            let mut x2 = e.uninit(n_embd)?;
                            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
                            e.dtod_copy_into(&x2, &mut next, r * n_embd)?;
                        }
                    }
                    x_t = next;
                    continue;
                }
                for r in 0..t {
                    e.dtod_copy_view(&h_t.slice(r * n_embd..(r + 1) * n_embd), &mut h_row)?;
                    crate::tp::set_verify_tcol(Some(r));
                    if fa_rows {
                        crate::tp::set_spec_fa2_defer(Some(r));
                    } else {
                        crate::tp::set_tcol_oproj_defer(Some(r));
                    }
                    let mixed = match &layer.mixer {
                        Mixer::Full(fa) => {
                            let ci = cache_index(r);
                            self.full_attn_decode(
                                e,
                                fa,
                                &h_row,
                                &row_positions[r],
                                positions[r] as usize,
                                &mut *caches[ci],
                                il,
                            )
                        }
                        _ => Err("step-tp-batch expects full attention".into()),
                    };
                    crate::tp::set_verify_tcol(None);
                    crate::tp::set_spec_fa2_defer(None);
                    crate::tp::set_tcol_oproj_defer(None);
                    let mixed = mixed?;
                    if fa_rows && crate::tp::take_spec_fa2_stashed() {
                        fa_deferred.push(r);
                    } else if crate::tp::take_tcol_oproj_stashed() {
                        deferred.push(r);
                    } else {
                        // Ineligible column (sub-floor ctx / rebase): finish this row
                        // with the ordinary per-row body.
                        let mut x_row = e.uninit(n_embd)?;
                        e.dtod_copy_view(&x_t.slice(r * n_embd..(r + 1) * n_embd), &mut x_row)?;
                        let (x1, ffn_out) =
                            self.residual_norm_ffn(e, layer, &x_row, &mixed, n_embd, il, eps)?;
                        let mut x2 = e.uninit(n_embd)?;
                        e.add(&x1, &ffn_out, &mut x2, n_embd)?;
                        e.dtod_copy_into(&x2, &mut next, r * n_embd)?;
                    }
                }
                if !fa_deferred.is_empty() && fa_deferred.len() != t {
                    return Err("step-tp-batch fa rows stashed a strict subset of rows".into());
                }
                if fa_deferred.len() == t {
                    deferred = fa_deferred;
                }
                if !deferred.is_empty() {
                    let mixed_t = if fa_rows && deferred.len() == t {
                        self.step35_batch_fa_rows_join(e, il, caches, cache_index, positions, t)?
                    } else {
                        self.step35_verify_oproj_tcol(e, il, t)?
                    };
                    let o_out = mixed_t.len() / t;
                    {
                        for &r in &deferred {
                            e.dtod_copy_view(
                                &mixed_t.slice(r * o_out..(r + 1) * o_out),
                                &mut mixed_row,
                            )?;
                            let mut x_row = e.uninit(n_embd)?;
                            e.dtod_copy_view(&x_t.slice(r * n_embd..(r + 1) * n_embd), &mut x_row)?;
                            let (x1, ffn_out) = self
                                .residual_norm_ffn(e, layer, &x_row, &mixed_row, n_embd, il, eps)?;
                            let mut x2 = e.uninit(n_embd)?;
                            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
                            e.dtod_copy_into(&x2, &mut next, r * n_embd)?;
                        }
                    }
                }
                x_t = next;
            }
            return Ok(x_t);
        }
        if b_n > 1 && has_rank_local_tp {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[step-tp-batch-exact] rows={b_n} execution=layer-major-b1 \
                     attention=rank-local kv_cache=per-session-distributed \
                     transport=native-p2p exactness=b1-full-layer-program \
                     performance_claim=false"
                );
            });
            // Preserve the isolated B=1 numerical program for every live session. The scheduler
            // may change width after any token; allowing norms, residuals, experts, or the head
            // to select a B-dependent kernel changes greedy output even when attention itself is
            // rowwise. Replay one layer across all rows before advancing so the same TP/EP
            // weights remain hot, while every row still executes the qualified B=1 program.
            let mut row_states = Vec::with_capacity(b_n);
            let mut row_positions = Vec::with_capacity(b_n);
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for row in 0..b_n {
                let mut h_row = e.uninit(n_embd)?;
                e.copy_view_into(
                    &mut h_row,
                    0,
                    &x.slice(row * n_embd..(row + 1) * n_embd),
                    n_embd,
                )?;
                row_states.push(h_row);
                row_positions.push(e.htod_i32(&[positions[row]])?);
            }
            for il in lo..hi {
                let mut next_states = Vec::with_capacity(b_n);
                for (row, h_row) in row_states.into_iter().enumerate() {
                    let position = [positions[row]];
                    let cache = cache_index(row);
                    let mut one = [&mut *caches[cache]];
                    next_states.push(self.step35_decode_rows_layers(
                        e,
                        h_row,
                        &mut one,
                        &position,
                        &row_positions[row],
                        None,
                        il,
                        il + 1,
                        ph_last,
                    )?);
                }
                row_states = next_states;
            }
            let mut outputs = e.uninit(b_n * n_embd)?;
            for (row, output) in row_states.iter().enumerate() {
                e.copy_into(&mut outputs, row * n_embd, output, n_embd)?;
            }
            return Ok(outputs);
        }
        let rank_local_positions = if has_rank_local_tp {
            let mut device_positions = Vec::with_capacity(b_n);
            for &position in positions {
                device_positions.push(e.htod_i32(&[position])?);
            }
            Some(device_positions)
        } else {
            None
        };
        // b2geo35 gate evidence: one line, first B>1 walk only (grep-stable prefix).
        if b_n > 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[step35-batch] first B>1 batched step35 walk: B={b_n} layers=[{lo},{hi})"
                );
            });
        }

        for il in lo..hi {
            let layer = &self.layers[il];
            let Mixer::Full(fa) = &layer.mixer else {
                return Err(format!("step35 layer {il} is not full-attn — corrupt config").into());
            };
            let geometry = self.step35_geom(il);
            let hd = geometry.head_dim_k as usize;
            let nkv = geometry.n_head_kv as usize;
            let nh = geometry.n_head as usize;
            let rbase = geometry.rope_base;
            let scale = geometry.attention_scale();
            let swa = geometry.window.is_some();
            let win = geometry.window.unwrap_or(0) as usize;
            let n_rot = geometry.n_rot as usize;
            let q_dim = nh * hd;
            let kv_dim = nkv * hd;

            // ---- attn_norm + q8_1 quantize, batched (B rows) ----
            let anorm = layer.attn_norm.float_data();
            let mut xn = e.uninit(b_n * n_embd)?;
            e.rms_norm(&x, anorm, &mut xn, n_embd, b_n, eps)?;
            let rank_local_tp = fa
                .step_tp_qkv
                .as_ref()
                .is_some_and(|tp| tp.attention.is_some());
            let mixed = if rank_local_tp {
                // The B>1 path returns through the full-row oracle above. This branch is therefore
                // the qualified B=1 rank-local TP attention program.
                let row_positions = rank_local_positions
                    .as_ref()
                    .expect("rank-local TP positions were prepared");
                let mut outputs = e.uninit(b_n * n_embd)?;
                #[allow(clippy::needless_range_loop)]
                // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                for row in 0..b_n {
                    let mut h_row = e.uninit(n_embd)?;
                    e.copy_view_into(
                        &mut h_row,
                        0,
                        &xn.slice(row * n_embd..(row + 1) * n_embd),
                        n_embd,
                    )?;
                    let cache = cache_index(row);
                    let output = self.step35_decode_attn(
                        e,
                        fa,
                        il,
                        &h_row,
                        None,
                        &row_positions[row],
                        caches[cache],
                    )?;
                    e.copy_into(&mut outputs, row * n_embd, &output, n_embd)?;
                }
                outputs
            } else {
                let (hq, hdq) = e.quantize_q8_1(&xn, b_n, n_embd)?;

                // ---- batched projections: q/k/v + the separate head-wise gate (one weight
                // stream for B rows; xn is the live f32 fallback for non-q8_1-fast classes) ----
                let q0 = e.matmul_pre(&fa.wq, &hq, &hdq, &xn, b_n)?;
                let k0 = e.matmul_pre(&fa.wk, &hq, &hdq, &xn, b_n)?;
                let v0 = e.matmul_pre(&fa.wv, &hq, &hdq, &xn, b_n)?;
                let gw = fa
                    .attn_gate
                    .as_ref()
                    .ok_or("step35 layer is missing attn_gate.weight (head-wise attention gate)")?;
                // gate input = the post-attn_norm hidden (upstream `cur`) — same xn/q8 pair.
                let gt = e.matmul_pre(gw, &hq, &hdq, &xn, b_n)?;

                // ---- q/k RMSNorm over head_dim rows + the per-layer PARTIAL rope ----
                let mut q = e.uninit(b_n * q_dim)?;
                e.rms_norm_opt(&q0, fa.q_norm_w(), &mut q, hd, b_n * nh, eps)?;
                let mut k = e.uninit(b_n * kv_dim)?;
                e.rms_norm_opt(&k0, fa.k_norm_w(), &mut k, hd, b_n * nkv, eps)?;
                let ff = if geometry.rope_factors {
                    self.step35_aux.as_ref().and_then(|a| a.rope_freqs(e))
                } else {
                    None
                };
                e.rope_neox2(
                    &mut q, &mut k, pos_d, hd, n_rot, nh, nkv, b_n, rbase, 1.0, ff,
                )?;
                ph_mark(e, 1, ph_last)?;

                // ---- per-session: KV append + windowed/global fa_decode (each session's OWN
                // len drives its view offset — the iso-gap law, no cross-session term) ----
                let mut attn = e.uninit(b_n * q_dim)?;
                if b_n == 1 {
                    // B=1 SPECIALIZED ENTRY (lane/cx-eagerpar): the general row loop below
                    // materializes q_row and a_row because a B>1 FA call consumes/produces one
                    // contiguous row at a time. At B=1, q and attn already ARE those whole rows.
                    // Pass them directly to the same fa_decode_kvmod call: this removes two
                    // arithmetic-free D2D copies (90 launches/token on Step3.7's 45 layers)
                    // without changing any arithmetic kernel, shape, argument value, or order.
                    // Keep the B>1 body verbatim below; b1fix's one-class/transition gates are
                    // the promotion bar, not an FP-similarity tolerance.
                    let kvl = caches[cache_index(0)].kv[il].as_mut().unwrap();
                    let k_row = k.slice(0..kv_dim);
                    let v_row = v0.slice(0..kv_dim);
                    let next_len = kvl.len + 1;
                    let (off, t_kv) = if swa && next_len > win {
                        (next_len - win, win)
                    } else {
                        (0, next_len)
                    };
                    let write_row = e.prepare_kv_append(kvl, off & !31usize, 1)?;
                    e.append_kv_quantized_view(
                        &k_row,
                        &v_row,
                        &mut kvl.k,
                        &mut kvl.v,
                        write_row,
                        kvl.kv_dim_k,
                        kvl.kv_dim_v,
                        kvl.k_tok_bytes,
                        kvl.v_tok_bytes,
                        false,
                    )?;
                    kvl.len = next_len;
                    ph_mark(e, 2, ph_last)?;
                    let physical = kvl.physical_rows(off, off + t_kv)?;
                    let k_view = e.view_u8_range(
                        &kvl.k,
                        physical.start * kvl.k_tok_bytes,
                        physical.end * kvl.k_tok_bytes,
                    );
                    let v_view = e.view_u8_range(
                        &kvl.v,
                        physical.start * kvl.v_tok_bytes,
                        physical.end * kvl.v_tok_bytes,
                    );
                    e.fa_decode_kvmod(
                        &q,
                        &k_view,
                        &v_view,
                        &mut attn,
                        hd,
                        nh,
                        nkv,
                        t_kv,
                        scale,
                        kvl.k_tok_bytes,
                        kvl.v_tok_bytes,
                        false,
                    )?;
                    ph_mark(e, 4, ph_last)?;
                } else {
                    for bi in 0..b_n {
                        let cache = &mut caches[cache_index(bi)];
                        let kvl = cache.kv[il].as_mut().unwrap();
                        let k_row = k.slice(bi * kv_dim..(bi + 1) * kv_dim);
                        let v_row = v0.slice(bi * kv_dim..(bi + 1) * kv_dim);
                        let next_len = kvl.len + 1;
                        let (off, t_kv) = if swa && next_len > win {
                            (next_len - win, win)
                        } else {
                            (0, next_len)
                        };
                        let write_row = e.prepare_kv_append(kvl, off & !31usize, 1)?;
                        e.append_kv_quantized_view(
                            &k_row,
                            &v_row,
                            &mut kvl.k,
                            &mut kvl.v,
                            write_row,
                            kvl.kv_dim_k,
                            kvl.kv_dim_v,
                            kvl.k_tok_bytes,
                            kvl.v_tok_bytes,
                            false,
                        )?;
                        kvl.len = next_len;
                        ph_mark(e, 2, ph_last)?;
                        // the eager arm's SWA view arithmetic, verbatim (step35_decode_attn):
                        // token-aligned offset, keys carry absolute rope, mask is positional.
                        let physical = kvl.physical_rows(off, off + t_kv)?;
                        let k_view = e.view_u8_range(
                            &kvl.k,
                            physical.start * kvl.k_tok_bytes,
                            physical.end * kvl.k_tok_bytes,
                        );
                        let v_view = e.view_u8_range(
                            &kvl.v,
                            physical.start * kvl.v_tok_bytes,
                            physical.end * kvl.v_tok_bytes,
                        );
                        // The per-session cache view remains authoritative (including SWA's
                        // physical-row rebase), while Q/O use their existing packed row views.
                        // This preserves the exact FA program and removes only the two D2D copies.
                        let q_row = q.slice(bi * q_dim..(bi + 1) * q_dim);
                        let mut a_row = attn.slice_mut(bi * q_dim..(bi + 1) * q_dim);
                        e.fa_decode_kvmod_view(
                            &q_row,
                            &k_view,
                            &v_view,
                            &mut a_row,
                            hd,
                            nh,
                            nkv,
                            t_kv,
                            scale,
                            kvl.k_tok_bytes,
                            kvl.v_tok_bytes,
                            false,
                        )?;
                        ph_mark(e, 4, ph_last)?;
                    }
                }

                // ---- head-wise gate (one sigmoid per (token, head), pre-wo) + o-proj at m=B ----
                let mut ag = e.uninit(b_n * q_dim)?;
                e.attn_head_gate(&attn, &gt, &mut ag, None, hd, nh, b_n)?;
                {
                    // AWQ (memra#253): o_proj carries its own per-input-channel scale.
                    let __wpqs =
                        e.pre_quant_scaled(&ag, fa.wo_pqs.as_ref(), fa.wo.in_features(), b_n)?;
                    e.matmul(&fa.wo, __wpqs.as_ref().unwrap_or(&ag), b_n)
                }?
            };
            ph_mark(e, 5, ph_last)?;

            // ---- residual add + post_attn_norm + FFN, batched ----
            let pnorm = layer.post_attn_norm.float_data();
            let mut x1 = e.uninit(b_n * n_embd)?;
            let mut z = e.uninit(b_n * n_embd)?;
            e.add_rms_norm(&x, &mixed, pnorm, &mut x1, &mut z, n_embd, b_n, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    ffn_down_pqs,
                } => {
                    // A dense step35 FFN's clamp is the SHEXP array (upstream's one
                    // build_ffn serves dense + shared expert, llama-graph.cpp:1751);
                    // ffn_act_lim dispatches clamped/plain per layer. Layers 0-2 (the
                    // leading dense) have no live limit on this artifact, but the route
                    // is correct by construction, not by artifact.
                    let n_ff = ffn_gate.out_features();
                    let (zq, zd) = e.quantize_q8_1(&z, b_n, n_embd)?;
                    let g = e.matmul_pre(ffn_gate, &zq, &zd, &z, b_n)?;
                    let u = e.matmul_pre(ffn_up, &zq, &zd, &z, b_n)?;
                    let mut act = e.uninit(b_n * n_ff)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &g,
                        &u,
                        1.0,
                        1.0,
                        cfg.clamp_shexp_at(il as u32),
                        &mut act,
                        b_n * n_ff,
                    )?;
                    // AWQ (memra#253): the f32 activation exists here, so the
                    // per-input-channel scale is applied BEFORE the q8 quantize — both the
                    // quantized operand and the f32 fallback then carry it.
                    if let Some(pqs) = ffn_down_pqs.as_ref() {
                        e.apply_pre_quant_scale(
                            &mut act,
                            pqs.float_data(),
                            ffn_down.in_features(),
                            b_n,
                        )?;
                    }
                    let (aq, ad) = e.quantize_q8_1(&act, b_n, n_ff)?;
                    e.matmul_pre(ffn_down, &aq, &ad, &act, b_n)?
                }
                // t=B < PRIME_MIN_T: per-column decode-exact router + host sigmoid routing
                // + per-token expert dispatch — the same per-token program as eager t=1,
                // including the per-layer SwiGLU clamp (43/44) via the sequential path's
                // ffn_act_lim. The sigmoid-router deny on dev/pairs holds by predicate.
                crate::hybrid::Ffn::Moe(m) => {
                    // b_n==1: feed the zq8 seam (orndecode B2, see decode.rs twin). Wider
                    // ticks keep None — the dev arm quantizes per-token views there and the
                    // shexp pair rides the batched matmul, so there is nothing to share.
                    if b_n == 1 {
                        let zq8 = e.quantize_q8_1(&z, 1, n_embd)?;
                        self.moe_ffn_il_zq8(e, m, &z, Some(&zq8), b_n, il as u16)?
                    } else {
                        self.moe_ffn_il_zq8(e, m, &z, None, b_n, il as u16)?
                    }
                }
            };
            let mut x2 = e.uninit(b_n * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, b_n * n_embd)?;
            x = x2;
            ph_mark(e, 9, ph_last)?;
        }
        Ok(x)
    }

    /// Kill-switch seam for the gemma4 dense-31B batched decode arm. DEFAULT ON since the
    /// 2026-08-16 owner flip ("if the performance are so strong in favor... we serve the
    /// correctness and best performance"): the arm's exactness battery is green at B=4/8,
    /// the served identity gate is byte-exact vs eager at c1/c4, and the served aggregate
    /// read 55→257 tok/s c16 on the NVFP4mix artifact at 450W (SERVED-AGGREGATE.md).
    /// `MEMRA_GEMMA4_BATCH=0` forces the eager per-session path (the rollback);
    /// `1` is the old opt-in spelling, still accepted. Any OTHER value REFUSES LOUD at
    /// first use — a mis-typed kill switch must not silently pick a serving path.
    pub fn gemma4_batch_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| match std::env::var("MEMRA_GEMMA4_BATCH").as_deref() {
            Err(_) | Ok("1") => true,
            Ok("0") => false,
            Ok(v) => panic!(
                "MEMRA_GEMMA4_BATCH={v:?} is not a recognized value (want unset/1 = batched \
                 decode, 0 = eager kill switch) — refusing to guess a serving path"
            ),
        })
    }

    /// THE gemma4 dense-31B BATCHED DECODE ARM (lane/gemma-batched, 2026-08-16).
    ///
    /// gemma4 served eager-only — the c1→c8 aggregate was FLAT (~55 tok/s, per-stream
    /// collapse) because there was no batched arm, not because of quantization. This is it.
    ///
    /// SHAPE — batched where the weights are, per-session where the state is (the step35
    /// law, applied to gemma4's own geometry):
    ///   * embed+scale, attn_norm+q8_1 quantize, wq/wk/wv projections, q/k RMSNorm +
    ///     weightless-V norm + dual rope (fused `rms_norm_qkv_rope`), post_attn_norm, the
    ///     layer-scale tail with its dense GEGLU FFN (`gemma4_layer_tail_add_nq`), output
    ///     norm, softcapped head — ALL at m=B: one weight stream serves B rows (decode is
    ///     weight-BW-bound; that is the entire aggregate win). Every one of these is the
    ///     SAME batch-capable function the proven verify trunk (`gemma4_verify_trunk`) runs
    ///     at width t, so this arm inherits the verify path's numerics wholesale.
    ///   * KV append + fa_decode stay a PER-SESSION loop: each session appends its one new
    ///     token to its own cache and attends its own [win_off .. len] view — the SWA
    ///     window + global-vs-windowed geometry makes each session's t_kv independent, so
    ///     there is no cross-session batched attention (identical to eager per session).
    ///
    /// EXACTNESS: v1 routes every session's attention through `fa_decode_kvmod` (the eager
    /// arm's unconditional fallback — same call `gemma4_decode_attn` makes with the rows_w
    /// fast arms off), so a B=1 run is the eager decode's own attention program and the
    /// batch is per-row independent by construction. The rows / rows_w per-session fast
    /// arms are a later perf increment gated behind their own seam.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn gemma4_decode_batch(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let b_n = tokens.len();
        if b_n == 0 || b_n != caches.len() {
            return Err(format!(
                "gemma4_decode_batch: tokens/caches mismatch (tokens={b_n}, caches={})",
                caches.len()
            )
            .into());
        }
        // Exactness tier boundary: the battery is green at B<=8 (per-row mmvq); m>8
        // crosses the dp4a-tail/GEMM numeric configs it never proved. The worker's chunk
        // policy caps gemma4 at 8; this is the per-request backstop (Err, never a panic —
        // the 2026-08-07 worker-FATAL law).
        if b_n > 8 {
            return Err(format!(
                "gemma4_decode_batch: B={b_n} > 8, past the proven exactness tier — \
                 the scheduler must chunk gemma4 at <=8"
            )
            .into());
        }
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        if b_n > 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!("[gemma4-batch] first B>1 batched gemma4 walk: B={b_n}");
            });
        }
        // per-session rope positions (each sequence at its own depth).
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = e.htod_i32(&pos_v)?;
        let mut x = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), b_n * n_embd)?;
        // cross-layer carry: each tail emits the next layer's attn-normed q8_1 input.
        let mut h_carry: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        let n_layers = self.layers.len();
        for (il, layer) in self.layers.iter().enumerate() {
            let (hq, hdq) = match h_carry.take() {
                Some(p) => p,
                None => {
                    e.rms_norm_q8_1(&x, self.layers[0].attn_norm.float_data(), n_embd, b_n, eps)?
                }
            };
            let Mixer::Full(fa) = &layer.mixer else {
                return Err(format!("gemma4 layer {il} not full-attn — corrupt config").into());
            };
            // STAGE-A ORACLE ARM (MEMRA_FAST=0) ONLY. `matmul_pre`'s raw-f32 escape needs the f32
            // attn-normed activation, and this trunk never materializes one — `rms_norm_q8_1`
            // above returns just the (i8, f32-scales) pair, which is exactly why the projections
            // used to be handed `e.zeros(0)` and read out of bounds.
            //
            // `rms_norm_decode` is the right producer and not merely a convenient one: it is
            // documented BIT-IDENTICAL to `rms_norm_q8_1`'s sum-of-squares reduction (same
            // blockDim=1024, same shfl tree), which is the property the spec verify path already
            // depends on. So the f32 recomputed here is precisely the tensor `rms_norm_q8_1`
            // quantized — the oracle compares against the same activation the fast path saw,
            // differing only in the weight-side arithmetic it is meant to be checking.
            //
            // Cost on the daily path: ONE branch on a OnceLock bool. Nothing is allocated and no
            // kernel is launched unless MEMRA_FAST=0.
            let h_raw = if Engine::stage_a_raw_needed() {
                let mut hf = e.uninit(b_n * n_embd)?;
                e.rms_norm_decode(&x, layer.attn_norm.float_data(), &mut hf, n_embd, b_n, eps)?;
                Some(hf)
            } else {
                None
            };
            let o =
                self.gemma4_batch_attn(e, fa, il, &hq, &hdq, h_raw.as_ref(), &pos_d, b_n, caches)?;
            let next_norm = if il + 1 < n_layers {
                Some(self.layers[il + 1].attn_norm.float_data())
            } else {
                None
            };
            // pn-fold front (lane/gemma-pnfold merge): the batched arm rides the SAME
            // tail front as the eager/verify trio, so batched == eager holds by
            // construction at either MEMRA_G4_PNFOLD value (seam-off falls through to
            // the unfused rms_norm + tail chain this arm shipped with).
            let (xn, hn) = self.gemma4_layer_tail_add_nq_pn(e, layer, &o, &x, b_n, next_norm)?;
            x = xn;
            h_carry = hn;
        }
        let mut hn = e.uninit(b_n * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let mut ld = e.matmul(&self.output, &hn, b_n)?;
        let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
        e.softcap(&mut ld, cap, b_n * self.output.out_features())?;
        self.gemma4_suppress(e, &mut ld, b_n)?; // non-monotonic — before any argmax/sample
        let mut ph_last = std::time::Instant::now();
        self.decode_batch_epilogue(e, caches, samp, masks, lean, ld, b_n, &mut ph_last, None)
    }

    /// Per-session gemma4 attention for the batched arm: batched projections + fused
    /// q/k-norm + weightless-V-norm + dual rope over all B rows (per-row independent, the
    /// verify path's exact kernels), then a per-session KV append + `fa_decode_kvmod` over
    /// each session's own window/global view, then one batched wo matmul. Mirrors the eager
    /// `gemma4_decode_attn` fallback per row.
    #[allow(clippy::too_many_arguments)]
    fn gemma4_batch_attn(
        &self,
        e: &Engine,
        fa: &crate::hybrid::FullAttnLayer,
        il: usize,
        hq: &CudaSlice<i8>,
        hdq: &CudaSlice<f32>,
        h_raw: Option<&CudaSlice<f32>>,
        pos_d: &CudaSlice<i32>,
        b_n: usize,
        caches: &mut [&mut Cache],
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();
        let ones = aux.ones(e);
        // `h_raw` is Some ONLY under MEMRA_FAST=0, where matmul_pre takes its raw-f32 escape and
        // therefore needs a real activation; on the daily path it is None and the empty slice keeps
        // the old behaviour exactly (matmul_pre reads the q8_1 pair and never touches this buffer).
        let h0 = e.zeros(0)?;
        let h = h_raw.unwrap_or(&h0);
        // projections at m=B (on the fast path the f32 fallback `h` is empty and matmul_pre uses
        // the q8_1 pair; under the Stage-A oracle `h` carries the real f32 attn-normed rows).
        let q0 = e.matmul_pre(&fa.wq, hq, hdq, h, b_n)?;
        let k0 = e.matmul_pre(&fa.wk, hq, hdq, h, b_n)?;
        let v0 = if swa {
            e.matmul_pre(&fa.wv, hq, hdq, h, b_n)?
        } else {
            e.clone_dtod(&k0)? // globals: V := K clone (weightless V-norm, never roped)
        };
        let mut q = e.uninit(b_n * nh * hd)?;
        let mut k = e.uninit(b_n * nkv * hd)?;
        let mut v = e.uninit(b_n * nkv * hd)?;
        let ff = if swa {
            None
        } else {
            Some(
                aux.rope_freqs(e)
                    .expect("gemma4 global rope needs rope_freqs.weight"),
            )
        };
        e.rms_norm_qkv_rope(
            &q0,
            &k0,
            &v0,
            fa.q_norm_w(),
            fa.k_norm_w(),
            ones,
            &mut q,
            &mut k,
            &mut v,
            hd,
            self.gemma4_rope_dims(il),
            nh * b_n,
            nkv * b_n,
            pos_d,
            nh,
            nkv,
            base,
            1.0,
            ff,
            eps,
        )?;
        let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let mut attn = e.uninit(b_n * q_dim)?;
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for bi in 0..b_n {
            let kvl = caches[bi].kv[il].as_mut().unwrap();
            let k_row = k.slice(bi * kv_dim..(bi + 1) * kv_dim);
            let v_row = v.slice(bi * kv_dim..(bi + 1) * kv_dim);
            // gemma4's KV is a linear buffer (no ring rebase — the SWA view below is a plain
            // token-offset), so append at kvl.len exactly as eager gemma4_decode_attn does.
            e.append_kv_quantized_view(
                &k_row,
                &v_row,
                &mut kvl.k,
                &mut kvl.v,
                kvl.len,
                kvl.kv_dim_k,
                kvl.kv_dim_v,
                kvl.k_tok_bytes,
                kvl.v_tok_bytes,
                (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on()),
            )?;
            kvl.len += 1;
            // eager SWA view arithmetic (gemma4_decode_attn): token-aligned window offset;
            // keys carry absolute rope, the mask is purely positional.
            let (off_tok, t_kv) = if swa && kvl.len > win {
                (kvl.len - win, win)
            } else {
                (0, kvl.len)
            };
            let k_view = e.view_u8_range(
                &kvl.k,
                off_tok * kvl.k_tok_bytes,
                (off_tok + t_kv) * kvl.k_tok_bytes,
            );
            let v_view = e.view_u8_range(
                &kvl.v,
                off_tok * kvl.v_tok_bytes,
                (off_tok + t_kv) * kvl.v_tok_bytes,
            );
            let q_row = q.slice(bi * q_dim..(bi + 1) * q_dim);
            let mut a_row = attn.slice_mut(bi * q_dim..(bi + 1) * q_dim);
            e.fa_decode_kvmod_view(
                &q_row,
                &k_view,
                &v_view,
                &mut a_row,
                hd,
                nh,
                nkv,
                t_kv,
                scale,
                kvl.k_tok_bytes,
                kvl.v_tok_bytes,
                swa && crate::Engine::wkv_on(),
            )?;
        }
        {
            // AWQ (memra#253): o_proj carries its own per-input-channel scale.
            let __wpqs = e.pre_quant_scaled(&attn, fa.wo_pqs.as_ref(), fa.wo.in_features(), b_n)?;
            e.matmul(&fa.wo, __wpqs.as_ref().unwrap_or(&attn), b_n)
        }
    }

    /// Standalone MoESD target forward. This entrypoint is not used by serving: it widens the
    /// existing Step-3.7 batched layer walk to B*gamma rows while preserving one causal KV chain
    /// per session. It returns device logits and performs no sampling or logits D2H, matching the
    /// target-model term T_T measured by the paper.
    pub fn moesd_target_forward(
        &self,
        e: &Engine,
        tokens: &[u32],
        batch: usize,
        gamma: usize,
        caches: &mut [&mut Cache],
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.hyper.is_some() {
            return Err(
                "moesd_target_forward: the MoESD speculative target walk has no \
                 HyperConnections trunk — it drives `step35_decode_rows_layers`, a serial \
                 residual rows-walk, and no [B*gamma, streams, n_embd] hyper rows-walk with \
                 causal per-session verify appends exists. mHC speculative verify is a \
                 separate lane, not this entry point."
                    .into(),
            );
        }
        for cache in caches.iter() {
            cache.ensure_usable("moesd_target_forward")?;
        }
        if crate::plan_backend::decode_batch_program(&self.plan)
            != crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe
        {
            return Err("MoESD target forward currently requires Step-3.7/Step35 geometry".into());
        }
        if batch == 0 || gamma == 0 || caches.len() != batch || tokens.len() != batch * gamma {
            return Err(format!(
                "MoESD shape mismatch: B={batch} gamma={gamma} caches={} tokens={}",
                caches.len(),
                tokens.len(),
            )
            .into());
        }
        let rows = batch * gamma;
        if rows > 256 {
            return Err(format!("MoESD target width {rows} exceeds the frozen 32*8 matrix").into());
        }
        let _pp_walk =
            if crate::pp::pp_cuts(self.layers.len()).is_some() && !crate::pp::pp2_streams_off() {
                let rt = crate::pp::PpNRt::get(e)?;
                Some(rt.acquire_walk("moesd_target_forward")?)
            } else {
                None
            };
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let payload = rows * n_embd;
        let row_to_cache: Vec<usize> = (0..batch)
            .flat_map(|session| (0..gamma).map(move |_| session))
            .collect();
        let positions: Vec<i32> = row_to_cache
            .iter()
            .enumerate()
            .map(|(row, &session)| (caches[session].pos + row % gamma) as i32)
            .collect();
        let mut ph_last = std::time::Instant::now();

        let logits = if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
            if fence.len() != 3 || crate::pp::pp2_streams_off() {
                return Err(
                    "MoESD PP target forward requires the live two-stage stream split".into(),
                );
            }
            let rt = crate::pp::PpNRt::get(e)?;
            if rt.n_stages() != 2 {
                return Err(format!("MoESD expected two PP stages, got {}", rt.n_stages()).into());
            }
            let caller_stream = e.stream();
            rt.fence_stages_behind(&caller_stream)?;
            let slot = {
                let _st0 = rt.enter(0);
                let e0 = rt.engine(0, e);
                let pos_d = e0.htod_i32(&positions)?;
                let x = e0.htod(&self.embd.try_gather(n_embd, tokens)?)?;
                ph_mark(e0, 0, &mut ph_last)?;
                let x = self.step35_decode_rows_layers(
                    e0,
                    x,
                    caches,
                    &positions,
                    &pos_d,
                    Some(&row_to_cache),
                    fence[0],
                    fence[1],
                    &mut ph_last,
                )?;
                rt.tx(0, &x, payload)?
            };

            {
                let _st1 = rt.enter(1);
                let e1 = rt.engine(1, e);
                let pos_d = e1.htod_i32(&positions)?;
                let x = rt.rx(0, slot, payload)?;
                let x = self.step35_decode_rows_layers(
                    e1,
                    x,
                    caches,
                    &positions,
                    &pos_d,
                    Some(&row_to_cache),
                    fence[1],
                    fence[2],
                    &mut ph_last,
                )?;
                let mut hn = e1.uninit(payload)?;
                e1.rms_norm(
                    &x,
                    self.output_norm.float_data(),
                    &mut hn,
                    n_embd,
                    rows,
                    eps,
                )?;
                let logits = e1.matmul(&self.output, &hn, rows)?;
                rt.publish_to(1, &caller_stream)?;
                logits
            }
        } else {
            let pos_d = e.htod_i32(&positions)?;
            let x = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
            ph_mark(e, 0, &mut ph_last)?;
            let x = self.step35_decode_rows_layers(
                e,
                x,
                caches,
                &positions,
                &pos_d,
                Some(&row_to_cache),
                0,
                self.layers.len(),
                &mut ph_last,
            )?;
            let mut hn = e.uninit(payload)?;
            e.rms_norm(
                &x,
                self.output_norm.float_data(),
                &mut hn,
                n_embd,
                rows,
                eps,
            )?;
            e.matmul(&self.output, &hn, rows)?
        };
        for cache in caches.iter_mut() {
            cache.pos += gamma;
        }
        Ok(logits)
    }

    /// The batched tick's TAIL, after the trunk: grammar masks -> device sampling -> lean
    /// logits park -> `pos` bump. Split out with the pp seam (`decode_batch_layers`) because
    /// under a stage split this runs on the LAST stage's engine and device — the lm_head, the
    /// masks, the sampler, and `cache.last_logits_dev` all live where the final residual
    /// lands, and the caller must be able to place them there without duplicating 90 lines of
    /// serving contract. `logits` is `[b_n, n_vocab]` already computed by the caller (the
    /// output_norm + lm_head pair stays at the call site so a stage split can fence around
    /// it); everything after it is here, verbatim.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn decode_batch_epilogue(
        &self,
        e: &Engine,
        caches: &mut [&mut Cache],
        samp: &[Option<DevSamp>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        logits: CudaSlice<f32>,
        b_n: usize,
        ph_last: &mut std::time::Instant,
        pending_out: Option<&mut Option<PendingBatchStep>>,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        // Grammar masks and penalties both mutate the sampling copy. Preserve each affected
        // row's PRISTINE logits first: continuation/reuse consumers must never inherit a mask
        // or get penalized twice after restore.
        let n_vocab = self.output.out_features();
        let mut logits = logits;
        let mut pristine: Vec<Option<CudaSlice<f32>>> = Vec::new();
        let row_mutates = |bi: usize| {
            masks.get(bi).is_some_and(Option::is_some)
                || samp
                    .get(bi)
                    .and_then(Option::as_ref)
                    .is_some_and(|s| s.penalty.is_some())
        };
        if (0..b_n).any(row_mutates) {
            pristine.resize_with(b_n, || None);
            for bi in 0..b_n {
                if !row_mutates(bi) {
                    continue;
                }
                if lean {
                    let cache = &mut caches[bi];
                    if cache
                        .last_logits_dev
                        .as_ref()
                        .map(|d| d.len() < n_vocab)
                        .unwrap_or(true)
                    {
                        cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                    }
                    let dst = cache.last_logits_dev.as_mut().unwrap();
                    e.dtod_copy_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab), dst)?;
                } else {
                    let mut p = e.uninit(n_vocab)?;
                    e.dtod_copy_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab), &mut p)?;
                    pristine[bi] = Some(p);
                }
            }
        }

        // Penalties precede grammar and probability filters, matching the host sampler chain.
        // Flatten only unique sparse counts for affected rows; heterogeneous requests keep
        // independent windows and coefficients in one launch.
        let penalized: Vec<(usize, &DevPenalty)> = samp
            .iter()
            .take(b_n)
            .enumerate()
            .filter_map(|(bi, s)| s.as_ref()?.penalty.as_ref().map(|p| (bi, p)))
            .filter(|(_, p)| !p.counts.is_empty())
            .collect();
        if !penalized.is_empty() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let unique: usize = penalized.iter().map(|(_, p)| p.counts.len()).sum();
                eprintln!(
                    "[device-penalty] sparse sampled rows={} unique-counts={} \
                     execution=one-ragged-launch raw-logits=preserved",
                    penalized.len(),
                    unique,
                );
            });
            let mut ids = Vec::new();
            let mut counts = Vec::new();
            let mut offsets = Vec::with_capacity(penalized.len() + 1);
            let mut rows = Vec::with_capacity(penalized.len());
            let mut reps = Vec::with_capacity(penalized.len());
            let mut freqs = Vec::with_capacity(penalized.len());
            let mut presents = Vec::with_capacity(penalized.len());
            offsets.push(0i32);
            for (bi, p) in penalized {
                rows.push(bi as i32);
                reps.push(p.repeat);
                freqs.push(p.freq);
                presents.push(p.present);
                for &(id, count) in &p.counts {
                    ids.push(id);
                    counts.push(count);
                }
                offsets.push(ids.len() as i32);
            }
            // SAFETY: rows come from `enumerate()` over this batch; DevPenalty's opaque count
            // set guarantees unique ids; and offsets are appended from the flattened vectors.
            unsafe {
                e.penalize_logits_sparse_rows_unchecked(
                    &mut logits,
                    &ids,
                    &counts,
                    &offsets,
                    &rows,
                    &reps,
                    &freqs,
                    &presents,
                    n_vocab,
                )?;
            }
        }

        // GRAMMAR MASKS (constrained decoding): ban in place AFTER penalties and before the
        // device sampler. Penalized constrained rows remain on the host until their combined
        // composition gate exists, but keep the ordering correct as defense in depth.
        for (bi, m) in masks.iter().take(b_n).enumerate() {
            if let Some((mask, words)) = m {
                assert!(
                    samp.get(bi).and_then(Option::as_ref).is_some(),
                    "grammar-masked row {bi} must request a device sample"
                );
                e.mask_logits_col(&mut logits, mask, bi, n_vocab, *words)?;
            }
        }

        // Device-side sampling for requested rows (see the method doc). Enqueued before the
        // big logits D2H so the tiny [B] token readback rides the same sync.
        let pending = pending_out.is_some();
        let mut next: Vec<Option<u32>> = vec![None; b_n];
        let mut device_tokens: Option<CudaSlice<u32>> = None;
        if samp.iter().take(b_n).any(|s| s.is_some()) {
            let mut toks = e.alloc_u32_zeroed(b_n)?;
            let mut perturb: Option<CudaSlice<f32>> = None;
            // FILTERED rows batch their filter_stats (lane/moebatch-q35moe): the per-row
            // devsample_filtered_col shape paid 1 HtoD + 3 tiny allocs + a 1-block launch PER
            // ROW PER TICK, serializing B single-SM kernels on the stream — measured as the
            // whole filtered-vs-temp-only serve gap at c8 (487 vs 700+ agg tok/s). Group rows
            // by (temp, top_k, top_p, min_p) — filter_stats takes scalar knobs — and solve
            // each group's thresholds in ONE grid=F launch over shared stat buffers, then
            // per-row perturb+argmax read their stat slot. Same kernels, same expressions,
            // same per-row (seed, ctr) draw — only the launch/alloc shape changes.
            let filt: Vec<(usize, &DevSamp)> = samp
                .iter()
                .take(b_n)
                .enumerate()
                .filter_map(|(bi, s)| s.as_ref().map(|s| (bi, s)))
                .filter(|(_, s)| s.temp > 0.0 && (s.top_k > 0 || s.top_p < 1.0 || s.min_p > 0.0))
                .collect();
            // Per-group stat buffers (one filter_stats launch per distinct knob tuple —
            // usually exactly one group per tick). Z is computed for output-shape parity
            // with the per-row form; the draw itself reads th/max only.
            let mut group_stats: Vec<(CudaSlice<f32>, CudaSlice<f32>)> = Vec::new();
            let mut row_stat: Vec<Option<(usize, usize)>> = vec![None; b_n];
            if !filt.is_empty() {
                #[allow(clippy::type_complexity)]
                // allow: one-shot composite type; naming it would hide the shape that matters at the call site
                let mut groups: Vec<((f32, i32, f32, f32), Vec<usize>)> = Vec::new();
                for &(bi, s) in &filt {
                    let key = (s.temp, s.top_k, s.top_p, s.min_p);
                    match groups.iter_mut().find(|(k, _)| *k == key) {
                        Some((_, rows)) => rows.push(bi),
                        None => groups.push((key, vec![bi])),
                    }
                }
                for ((temp, top_k, top_p, min_p), rows) in &groups {
                    let rows_i32: Vec<i32> = rows.iter().map(|&bi| bi as i32).collect();
                    let rows_d = e.htod_i32(&rows_i32)?;
                    let mut th = e.zeros(rows.len())?;
                    let mut z = e.zeros(rows.len())?;
                    let mut mx = e.zeros(rows.len())?;
                    e.filter_stats(
                        &logits,
                        n_vocab,
                        &rows_d,
                        &mut th,
                        &mut z,
                        &mut mx,
                        n_vocab,
                        rows.len(),
                        *temp,
                        *top_k,
                        *top_p,
                        *min_p,
                    )?;
                    let g = group_stats.len();
                    for (i, &bi) in rows.iter().enumerate() {
                        row_stat[bi] = Some((g, i));
                    }
                    group_stats.push((th, mx));
                }
            }
            for (bi, s) in samp.iter().take(b_n).enumerate() {
                let Some(s) = s else {
                    continue;
                };
                let filtered = s.temp > 0.0 && (s.top_k > 0 || s.top_p < 1.0 || s.min_p > 0.0);
                if s.temp <= 0.0 {
                    e.argmax_token_device_col(&logits, bi, n_vocab, &mut toks, bi)?;
                } else if filtered {
                    if perturb.is_none() {
                        perturb = Some(e.zeros(n_vocab)?);
                    }
                    let pb = perturb.as_mut().unwrap();
                    let (g, i) = row_stat[bi].expect("filtered row missing batched stats");
                    let (th, mx) = &group_stats[g];
                    e.gumbel_perturb_filtered_col(
                        &logits, bi, pb, n_vocab, s.seed, s.ctr, s.temp, mx, th, i,
                    )?;
                    e.argmax_token_device_col(pb, 0, n_vocab, &mut toks, bi)?;
                } else {
                    if perturb.is_none() {
                        perturb = Some(e.zeros(n_vocab)?);
                    }
                    let pb = perturb.as_mut().unwrap();
                    e.gumbel_perturb_col(&logits, bi, pb, n_vocab, s.seed, s.ctr, s.temp)?;
                    e.argmax_token_device_col(pb, 0, n_vocab, &mut toks, bi)?;
                }
            }
            if !pending {
                let host_toks = e.dtoh_u32(&toks)?;
                for (bi, s) in samp.iter().take(b_n).enumerate() {
                    if s.is_some() {
                        next[bi] = Some(host_toks[bi]);
                    }
                }
            }
            device_tokens = Some(toks);
        }

        if let Some(slot) = pending_out {
            for c in caches.iter_mut() {
                c.pos += 1;
            }
            ph_mark(e, 11, ph_last)?;
            let done = e.stream().record_event(None)?;
            *slot = Some(PendingBatchStep::new(
                logits,
                pristine,
                device_tokens,
                samp.iter().take(b_n).map(Option::is_some).collect(),
                n_vocab,
                lean,
                done,
                e.copy_stream.clone(),
            ));
            return Ok((Vec::new(), vec![None; b_n]));
        }

        let lean_any = lean && samp.iter().take(b_n).any(|s| s.is_some());
        let rows: Vec<Vec<f32>> = if lean_any {
            // LEAN: park device-sampled rows on-device (per-cache buffer, dtod); D2H only
            // the rows that still need host logits. No sampled rows + no fallback rows =
            // the big D2H disappears (the [B] token readback above already synced).
            for (bi, s) in samp.iter().take(b_n).enumerate() {
                if s.is_none() {
                    continue;
                }
                // Mutated rows already parked their PRISTINE copy above — neither a grammar
                // ban nor a penalty may poison the reuse-pool consumer.
                if masks.get(bi).copied().flatten().is_some()
                    || s.as_ref().is_some_and(|s| s.penalty.is_some())
                {
                    continue;
                }
                let cache = &mut caches[bi];
                if cache
                    .last_logits_dev
                    .as_ref()
                    .map(|d| d.len() < n_vocab)
                    .unwrap_or(true)
                {
                    cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                }
                let dst = cache.last_logits_dev.as_mut().unwrap();
                e.dtod_copy_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab), dst)?;
            }
            (0..b_n)
                .map(|bi| {
                    if samp.get(bi).and_then(Option::as_ref).is_some() {
                        Ok(Vec::new())
                    } else {
                        e.dtoh_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab))
                    }
                })
                .collect::<Result<_, _>>()?
        } else {
            let host = e.dtoh(&logits)?;
            (0..b_n)
                .map(|bi| {
                    // grammar-masked non-lean rows return the PRISTINE copy (the in-place ban
                    // must never leak into last_logits — reuse-pool/park semantics unchanged).
                    if let Some(p) = pristine.get(bi).and_then(|p| p.as_ref()) {
                        return e.dtoh(p);
                    }
                    Ok(host[bi * n_vocab..(bi + 1) * n_vocab].to_vec())
                })
                .collect::<Result<_, _>>()?
        };
        for c in caches.iter_mut() {
            c.pos += 1;
        }
        ph_mark(e, 11, ph_last)?;
        Ok((rows, next))
    }
}

fn b1_fast_plan_eligible(plan: &memra_gguf::model_plan::ModelPlan) -> bool {
    // Every GDN plan is excluded: spec verify for this recurrent operation runs
    // the generic batched numeric class (spec.rs batched_serving_numeric_class), so live B=1 serving
    // must stay in that same class. B1FAST's eager program would reopen the near-tie-flip
    // divergence the 2026-08-14 exactness fix closed (1 ULP at layer 2 -> 2.3e-1 head
    // maxdiff, amplified by the GDN recurrence).
    !plan
        .trunk_operations()
        .contains(&memra_gguf::model_plan::OperationKind::GatedDeltaNet)
}

fn b1_fast_env_on(value: Option<&str>) -> bool {
    value == Some("1")
}

#[cfg(test)]
mod tests {
    use super::{
        PpWaveIncoming, PpWaveOutgoing, b1_fast_env_on, b1_fast_plan_eligible, pp_wave_channels,
    };
    use memra_gguf::config::{HfConfig, ModelConfig};

    fn protocol_pair(boundary: usize) -> (PpWaveOutgoing, PpWaveIncoming) {
        let (mut outgoing, mut incoming) = pp_wave_channels(boundary + 1);
        (
            outgoing[boundary].take().unwrap(),
            incoming[boundary].take().unwrap(),
        )
    }

    #[test]
    fn pp_wave_credit_requires_exact_ack_before_slot_reuse() {
        let (mut outgoing, incoming) = protocol_pair(0);

        let expected0 = outgoing.prepare(0).unwrap();
        assert_eq!(expected0, None);
        outgoing.publish(0, 1, expected0).unwrap();
        let transfer0 = incoming.receive(0).unwrap();

        let expected1 = outgoing.prepare(1).unwrap();
        assert_eq!(expected1, Some(0));
        outgoing.publish(1, 0, expected1).unwrap();
        let transfer1 = incoming.receive(1).unwrap();

        // Wave 2 wants slot 1 again. Credit arrives only through the exact wave-0/slot-1
        // acknowledgement that a real consumer sends after rt.rx records ev_rx.
        incoming.acknowledge(transfer0).unwrap();
        let expected2 = outgoing.prepare(2).unwrap();
        assert_eq!(expected2, Some(1));
        outgoing.publish(2, 1, expected2).unwrap();
        let transfer2 = incoming.receive(2).unwrap();

        incoming.acknowledge(transfer1).unwrap();
        incoming.acknowledge(transfer2).unwrap();
        outgoing.finish().unwrap();
        assert!(outgoing.pending.is_empty());
        assert_eq!(outgoing.slot_owner, [None, None]);
    }

    #[test]
    fn pp_wave_protocol_rejects_order_and_propagates_worker_error() {
        let (mut outgoing, incoming) = protocol_pair(0);
        let expected = outgoing.prepare(0).unwrap();
        outgoing.publish(0, 0, expected).unwrap();
        let order_error = incoming.receive(1).unwrap_err();
        assert!(order_error.contains("expected wave 1"), "{order_error}");

        let (outgoing, incoming) = protocol_pair(1);
        outgoing.publish_worker_error("injected stage failure");
        let worker_error = incoming.receive(0).unwrap_err();
        assert!(
            worker_error.contains("injected stage failure"),
            "{worker_error}"
        );
        assert!(worker_error.contains("boundary 1"), "{worker_error}");
        assert!(worker_error.contains("wave 0"), "{worker_error}");
    }

    #[test]
    fn pp_wave_credit_rejects_wrong_ack_and_slot_generation() {
        let (mut outgoing, incoming) = protocol_pair(0);
        let expected0 = outgoing.prepare(0).unwrap();
        outgoing.publish(0, 0, expected0).unwrap();
        let _transfer0 = incoming.receive(0).unwrap();
        let expected1 = outgoing.prepare(1).unwrap();
        assert_eq!(expected1, Some(1));
        let wrong_slot = outgoing.publish(1, 0, expected1).unwrap_err();
        assert!(
            wrong_slot.contains("broke slot alternation"),
            "{wrong_slot}"
        );

        // Rebuild after the rejected TX and inject an acknowledgement for wave 1 before wave 0.
        let (mut outgoing, incoming) = protocol_pair(0);
        let expected0 = outgoing.prepare(0).unwrap();
        outgoing.publish(0, 0, expected0).unwrap();
        let transfer0 = incoming.receive(0).unwrap();
        let expected1 = outgoing.prepare(1).unwrap();
        outgoing.publish(1, 1, expected1).unwrap();
        let transfer1 = incoming.receive(1).unwrap();
        incoming.acknowledgements.send(transfer1).unwrap();
        let wrong_ack = outgoing.prepare(2).unwrap_err();
        assert!(wrong_ack.contains("expected acknowledgement wave 0 slot 0"));
        assert!(wrong_ack.contains("got boundary 0 wave 1 slot 1"));

        // Keep the compiler honest that the expected transfer really was the earlier one.
        assert_eq!(transfer0.wave, 0);
    }

    #[test]
    fn pp_wave_protocol_reports_forward_and_ack_channel_closure() {
        let (outgoing, incoming) = protocol_pair(0);
        drop(outgoing);
        let forward_closed = incoming.receive(0).unwrap_err();
        assert!(forward_closed.contains("transfer channel closed"));

        let (mut outgoing, incoming) = protocol_pair(0);
        let expected0 = outgoing.prepare(0).unwrap();
        outgoing.publish(0, 0, expected0).unwrap();
        let _ = incoming.receive(0).unwrap();
        let expected1 = outgoing.prepare(1).unwrap();
        outgoing.publish(1, 1, expected1).unwrap();
        let _ = incoming.receive(1).unwrap();
        drop(incoming);
        let ack_closed = outgoing.prepare(2).unwrap_err();
        assert!(ack_closed.contains("acknowledgement channel closed"));

        let (mut outgoing, incoming) = protocol_pair(0);
        drop(incoming);
        let expected = outgoing.prepare(0).unwrap();
        let publish_closed = outgoing.publish(0, 0, expected).unwrap_err();
        assert!(publish_closed.contains("transfer channel closed"));
    }

    #[test]
    fn gdn_plans_stay_in_one_decode_numeric_class_across_widths() {
        let compile = |json| {
            memra_gguf::model_plan::ModelPlan::compile(&ModelConfig::from_hf(&HfConfig::parse(
                json,
            )))
            .unwrap()
        };
        let gdn = compile(
            r#"{"model_type":"qwen3_5","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,
            "full_attention_interval":2,"linear_conv_kernel_dim":3,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":1,"linear_num_value_heads":2}"#,
        );
        let full = compile(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        );
        assert!(!b1_fast_plan_eligible(&gdn));
        assert!(b1_fast_plan_eligible(&full));
    }

    #[test]
    fn b1_eager_program_requires_explicit_opt_in() {
        assert!(!b1_fast_env_on(None));
        assert!(!b1_fast_env_on(Some("0")));
        assert!(!b1_fast_env_on(Some("true")));
        assert!(b1_fast_env_on(Some("1")));
    }
}
