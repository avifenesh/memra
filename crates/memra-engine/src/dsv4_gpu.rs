//! DeepSeek-V4-Flash GPU trunk forward (lane 4): 2-card layer-split placement,
//! correctness bring-up gated against the lane-2/3 CPU oracle fixtures.
//!
//! Plan of record: wt-dsv4-loader research/dsv4-flash-loader-20260818/RECEIPTS.md
//! "Lane 4" (placement math, quant rungs, threshold derivation — banked BEFORE this
//! module was written). Semantic law: darklanes SEMANTICS.md; arithmetic contract: the
//! lane-3 CPU oracle (memra_gguf::dsv4_forward), whose host-side pieces
//! (hc_split_sinkhorn, rope tables, index builders, routing math) are REUSED here
//! verbatim so the CPU/GPU forks share one implementation of every host-side rule.
//!
//! Rungs (explicit): trunk routed experts stay AS-STORED NVFP4 on GPU and are
//! dequantized per activated expert into a reused bf16 scratch (exact in bf16), all
//! other quantized linears are host-dequantized (lane-1 proven decoders) to bf16 at
//! load with a bit-level exactness refusal; f32 islands (SEMANTICS §7.2) run in
//! dedicated f32/f64 kernels or on the host. bf16 enters ONLY at the activation inputs
//! of the non-island GEMMs (cuBLASLt bf16, f32 accumulate).
//!
//! Multi-GPU: PP layer split (the engine's only executing multi-GPU idiom, pp.rs /
//! Step-3.7-Flash precedent), split point derived from per-layer byte math, ONE hc-state
//! boundary copy per forward via host bounce (peer copy is a perf-lane step).
//!
//! NOT a serving path: prefill-only, greedy continuation by re-prefill per step (the
//! accepted O(n²) bring-up rung). Decode KV caching, CUDA-graph, batched serving and any
//! perf claims belong to later lanes.

use std::collections::BTreeMap;
use std::ops::Range;
use std::os::raw::c_void;
use std::path::Path;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use memra_gguf::dsv4_forward::{
    ActQuantVariant, Dsv4Model, FreqsCis, compress_topk_idxs, hc_split_sinkhorn,
    precompute_freqs_cis, window_topk_idxs,
};

use crate::dsv4_ffi as k;
use crate::dsv4_ffi::ck;
use crate::mmq_ffi::{memra_bind_device, memra_moe_kq_gemm_sk};

type Res<T> = Result<T, String>;

fn e<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> String + '_ {
    move |err| format!("{what}: {err}")
}

// ---------------------------------------------------------------- host math (oracle twins)

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch softplus (beta=1, threshold=20) — same as the oracle's private softplus_f32.
#[inline]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

// ---------------------------------------------------------------- device buffers

/// One stage = one GPU: its runtime handle plus the resident weights of its layer range.
pub struct Stage {
    pub dev: usize,
    pub gpu: memra_runtime::Gpu,
    pub layers: Vec<LayerDev>,
    pub embed: Option<CudaSlice<u8>>, // bf16 raw [vocab, hidden] (stage 0)
    pub head: Option<CudaSlice<u8>>,  // bf16 raw [vocab, hidden] (last stage)
    pub trunk_norm: Option<CudaSlice<f32>>,
    pub hc_head_fn: Option<CudaSlice<f32>>, // [hc, hc*hidden]
    pub fc_yarn: CudaSlice<f32>,            // rope table, compressor layers [max_seq, rd]
    pub fc_plain: CudaSlice<f32>,           // rope table, ratio-0 layers    [max_seq, rd]
    pub ws: CudaSlice<u8>,                  // cuBLASLt workspace
    pub deq: [CudaSlice<u8>; 3],            // expert dequant scratch, bf16 [inter*hidden] each
    pub loaded_bytes: u64,                  // resident weight bytes uploaded to this device
    // lane 8: device twins of the trunk hc_head gate constants (last stage)
    pub hc_head_base_dev: Option<CudaSlice<f32>>,
    pub hc_head_scale_dev: Option<CudaSlice<f32>>,
}

pub struct CmpDev {
    pub ratio: usize,
    pub d: usize,
    pub latent: usize,
    pub overlap: bool,
    pub rotate: bool,
    pub wkv: CudaSlice<f32>,   // f32 island
    pub wgate: CudaSlice<f32>, // f32 island
    pub norm: CudaSlice<f32>,
    pub ape: CudaSlice<f32>, // [ratio, latent]
}

pub struct IdxDev {
    pub wq_b: DenseBf16,         // bf16 [heads*hd, q_lora]
    pub weights_proj: DenseBf16, // bf16 [heads, hidden]
    pub wq_b_fp8: Option<Fp8Dense>,
    pub weights_proj_fp8: Option<Fp8Dense>,
    pub cmp: CmpDev,
    pub heads: usize,
    pub hd: usize,
    pub topk: usize,
}

/// Iteration-5 FP8 dense arm (`MEMRA_DSV4_DENSE_ARM=fp8`): an FP8-blk linear held
/// AS-STORED — e4m3 codes `[rows, cols]` plus the 128x128 block-scale grid decoded to
/// f32 on the host (exact: every e8m0 code is a pow2; 0xFF refused at load). The device
/// GEMV twins decode `e4m3(code) * scale` in-register — the SAME f32 value the bf16
/// dequant slab holds (the loader's `f32_to_bf16_exact` refusal proves exactness), with
/// the accumulation order VERBATIM — so the arm is bit-identical to the bf16 arm by
/// construction and its gate is a no-regression proof. It5 ledger item 3: when this
/// pair exists, the bf16 twin is NOT device-resident — it drops to [`DenseBf16::Host`]
/// staged residency (the dual-residency +~2.7 GiB/card is gone).
pub struct Fp8Dense {
    pub codes: CudaSlice<u8>,   // e4m3, [rows, cols] row-major as stored
    pub scales: CudaSlice<f32>, // [ceil(rows/128), sc_cols] host-decoded e8m0
    pub sc_cols: usize,         // ceil(cols/128)
    pub rows: usize,
    pub cols: usize,
}

/// It5 ledger item 3 — residency of a dense bf16 slab. `Dev` = device-resident, today's
/// exact bytes: the only realization when the dense arm is bf16, and always the
/// realization for the drafter/MTP blocks (no fp8 twins this rung). `Host` = the fp8
/// dense arm's STAGED residency: the same host-dequantized bf16 bytes the loader would
/// have uploaded, kept host-side; the fp8 pair owns every device decode/verify read
/// (via [`dwsel`]) and the prefill pass stages these bytes H2D per consuming call,
/// the transient copy freed stream-ordered when the [`DenseView`] drops. This is the
/// engine's existing staged-residency idiom (hybrid EDGE-1 `HostExps` / the moe-cache
/// host-resident expert staging) translated to dsv4; dsv4 has no CUDA-graph capture,
/// so the "release after capture" boundary degenerates to "never resident outside a
/// prefill pass". Prefill's bf16 path is byte-identical by construction: the staged
/// upload is the SAME `f32_to_bf16_exact` byte vector the resident slab held.
pub enum DenseBf16 {
    Dev(CudaSlice<u8>),
    Host(Vec<u8>),
}

impl DenseBf16 {
    /// The device-resident slab. Host residency here is an ENGINE bug, never an env
    /// error: `Host` exists only when the fp8 arm is on, and every path that reaches
    /// this accessor under fp8 (legacy decode combos, bf16-slab probes) is already a
    /// boot refusal (hermes a4e3d9a8eab4cf17) or dwsel-routed to the fp8 twin.
    pub fn dev(&self) -> &CudaSlice<u8> {
        match self {
            DenseBf16::Dev(d) => d,
            DenseBf16::Host(_) => unreachable!(
                "bf16 dense slab is host-staged (fp8 dense arm): this consumer must \
                 ride the fp8 twins (dwsel) or the staged prefill view"
            ),
        }
    }

    /// Prefill-class access (block_forward / shared-expert finish): borrow the
    /// resident slab, or stage the host bytes into a transient device copy freed
    /// (stream-ordered, after the enqueued consumers) when the view drops.
    fn staged(&self, stream: &std::sync::Arc<CudaStream>) -> Res<DenseView<'_>> {
        Ok(match self {
            DenseBf16::Dev(d) => DenseView::Res(d),
            DenseBf16::Host(b) => DenseView::Tmp(upload_u8(stream, b)?),
        })
    }
}

/// A borrowed resident slab or a staged transient copy — see [`DenseBf16::staged`].
pub enum DenseView<'a> {
    Res(&'a CudaSlice<u8>),
    Tmp(CudaSlice<u8>),
}

impl DenseView<'_> {
    fn slab(&self) -> &CudaSlice<u8> {
        match self {
            DenseView::Res(d) => d,
            DenseView::Tmp(d) => d,
        }
    }
}

/// Dense-weight pointer for the device-path GEMV wrappers: the bf16 dequant slab, or
/// the as-stored FP8 pair when the dense arm is on. Copy of raw pointers only — built
/// per call from the owning slabs via [`dwsel`].
#[derive(Clone, Copy)]
pub enum DW {
    Bf16(*const c_void),
    Fp8 {
        codes: *const c_void,
        scales: *const f32,
        sc_cols: i32,
    },
}

impl DW {
    /// Row-offset view (the grouped wo_a slices): `rows_off` rows into the weight, row
    /// width `cols`. The fp8 arm requires the offset to land on a scale-grid row
    /// boundary (o_lora = 1024 = 8x128 — asserted, never assumed).
    fn offset_rows(self, rows_off: usize, cols: usize) -> DW {
        match self {
            DW::Bf16(p) => DW::Bf16((p as usize + rows_off * cols * 2) as *const c_void),
            DW::Fp8 {
                codes,
                scales,
                sc_cols,
            } => {
                assert_eq!(
                    rows_off % 128,
                    0,
                    "fp8 dense arm: grouped row offset {rows_off} not on the 128-row \
                     scale-grid boundary"
                );
                DW::Fp8 {
                    codes: (codes as usize + rows_off * cols) as *const c_void,
                    scales: unsafe { scales.add((rows_off / 128) * sc_cols as usize) },
                    sc_cols,
                }
            }
        }
    }
}

/// Select the weight realization for a device-path GEMV: the fp8 pair when the dense
/// arm is on AND this tensor is FP8-blk stored, else the bf16 slab. `active` is
/// `self.dense_fp8` — passed explicitly because the wrappers are associated fns.
fn dwsel(
    active: bool,
    stream: &cudarc::driver::CudaStream,
    w_bf16: &DenseBf16,
    fp8: &Option<Fp8Dense>,
) -> DW {
    match fp8 {
        Some(f) if active => DW::Fp8 {
            codes: f.codes.device_ptr(stream).0 as *const c_void,
            scales: f.scales.device_ptr(stream).0 as *const f32,
            sc_cols: f.sc_cols as i32,
        },
        // item 3: reached only when the fp8 twin is absent or the arm is off, i.e.
        // exactly when the bf16 slab IS device-resident — .dev() is the invariant.
        _ => DW::Bf16(w_bf16.dev().device_ptr(stream).0 as *const c_void),
    }
}

/// Routed-expert quantization recipe of a layer (lane-1 census: trunk = modelopt NVFP4,
/// MTP = OCP MXFP4). Never inferred from ancestry — detected from the stored dtypes and
/// sibling names, refused on any surprise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertKind {
    Nvfp4,
    Mxfp4,
}

/// Lane 7: which expert-GEMM realization runs. `Bf16Dequant` = the lane-4 gated rung
/// (on-the-fly exact dequant + cuBLASLt bf16, the fallback and A/B reference).
/// `Native` = the reference-law quantized GEMMs (act_quant per-128 FP8 codes ×
/// as-stored NVFP4/MXFP4 slabs, kernel.py fp4_gemm arithmetic — RECEIPTS.md "Lane 7").
/// Selected by `MEMRA_DSV4_EXPERT_ARM=native` via [`memra_gguf::dsv4_forward::
/// expert_arm_native`] — the SAME seam the CPU oracle reads, so one invocation can
/// never mix numeric classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertArm {
    Bf16Dequant,
    Native,
}

/// Lane 8: which decode-step realization runs (RECEIPTS.md "Lane 8"). `Legacy` = the
/// lane-6/7 gated host-driven loop, byte-stable. `Device` = the device-resident step:
/// preallocated workspace arena, device index build / fine top-k / router / Sinkhorn /
/// head gate, one-launch-per-projection indirect expert dispatch, peer-copy PP
/// boundary. `host_math: true` (seam `device-hostmath`) keeps Sinkhorn + router +
/// fine-top-k + head-gate math on the HOST — the byte-identity instrument for the
/// mechanical rungs; `false` (seam `device`) runs them as kernels (expf/log1pf
/// realization fork, gated at class bounds per the lane-6/7 doctrine). Selected by
/// MEMRA_DSV4_DECODE_PATH — read once at load and printed; one binary carries both
/// arms for the interleaved A/B law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    Legacy,
    Device { host_math: bool },
}

pub struct LayerDev {
    pub il: u32,
    pub ratio: usize,
    pub expert_kind: ExpertKind,
    // attention (bf16 unless island; staged host residency under the fp8 dense arm)
    pub wq_a: DenseBf16,
    pub wq_b: DenseBf16,
    pub wkv: DenseBf16,
    pub wo_a: DenseBf16, // [o_groups*o_lora, hidden-group-width] grouped rows
    pub wo_b: DenseBf16,
    pub q_norm: CudaSlice<f32>,
    pub kv_norm: CudaSlice<f32>,
    pub attn_norm: CudaSlice<f32>,
    pub ffn_norm: CudaSlice<f32>,
    pub sink: CudaSlice<f32>,
    pub cmp: Option<CmpDev>,
    pub idx: Option<IdxDev>,
    // hyper-connections (f32 island; base/scale live host-side)
    pub hc_attn_fn: CudaSlice<f32>,
    pub hc_ffn_fn: CudaSlice<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    // lane 8: device twins of the host-side routing/hc constants (tiny; loaded always)
    pub hc_attn_base_dev: CudaSlice<f32>,
    pub hc_attn_scale_dev: CudaSlice<f32>,
    pub hc_ffn_base_dev: CudaSlice<f32>,
    pub hc_ffn_scale_dev: CudaSlice<f32>,
    pub gate_bias_dev: Option<CudaSlice<f32>>,
    /// i32 cast of tid2eid, range- and distinctness-validated at LOAD (the legacy path
    /// asserts per token at route time; the device route kernel cannot).
    pub tid2eid_dev: Option<CudaSlice<i32>>,
    pub experts_s2_dev: CudaSlice<f32>,
    /// Six pointer planes, code/scale for w1/w2/w3. No repacking or duplicate bank.
    experts_modelopt_table: Option<CudaSlice<u64>>,
    // MoE
    pub gate_w: CudaSlice<f32>, // f32 island [ne, hidden]
    pub gate_bias: Option<Vec<f32>>,
    pub tid2eid: Option<Vec<i64>>, // host routing table (hash layers)
    pub experts_w: CudaSlice<u8>,  // expert slab: per (e, proj) nibble-pair bytes
    pub experts_sc: CudaSlice<u8>, // expert slab: per (e, proj) scales (e4m3/16 or e8m0/32)
    pub experts_s2: Vec<f32>,      // host [ne*3] scale_2 (NVFP4 only, asserted pow2)
    pub shared_w: [DenseBf16; 3],  // bf16 shared expert w1/w2/w3
    // iteration-5 FP8 dense arm: as-stored twins of the FP8-blk linears, Some only when
    // MEMRA_DSV4_DENSE_ARM=fp8 AND the tensor is F8_E4M3-stored (trunk layers only —
    // the drafter/MTP blocks ride the prefill helpers and keep bf16 this rung).
    pub wq_a_fp8: Option<Fp8Dense>,
    pub wq_b_fp8: Option<Fp8Dense>,
    pub wkv_fp8: Option<Fp8Dense>,
    pub wo_a_fp8: Option<Fp8Dense>,
    pub wo_b_fp8: Option<Fp8Dense>,
    pub shared_fp8: [Option<Fp8Dense>; 3],
}

/// Fixture-array capture (GPU twin of the oracle's BlockCapture, gathered host-side).
#[derive(Default)]
pub struct GpuCapture {
    pub embed_out: Option<Vec<f32>>,
    pub layer_out: BTreeMap<u32, Vec<f32>>,
    pub attn_out: BTreeMap<u32, Vec<f32>>,
    /// diagnostic (lane-6 bisect probe): post-attn-norm x, post-rope q, post-QAT kv,
    /// post-derotation o — full [s, ...] arrays
    pub x_dbg: BTreeMap<u32, Vec<f32>>,
    pub q_dbg: BTreeMap<u32, Vec<f32>>,
    pub kv_dbg: BTreeMap<u32, Vec<f32>>,
    pub o_dbg: BTreeMap<u32, Vec<f32>>,
    pub compressor_kv: BTreeMap<u32, (Vec<f32>, usize)>,
    pub indexer_kv: BTreeMap<u32, (Vec<f32>, usize)>,
    pub index_score: BTreeMap<u32, (Vec<f32>, usize)>,
    /// lane 7: post-ffn-norm MoE input rows [s, hidden] (the real activation vectors
    /// the native-GEMM kernel gate feeds to sampled experts)
    pub moe_x: BTreeMap<u32, Vec<f32>>,
    pub want: std::collections::BTreeSet<u32>,
}

/// MTP (NextN) block on the LAST stage (pp idiom: MTP -> last stage). Shares the trunk
/// embed (host-gathered) and head; own norms/projections/block/hc_head (SEMANTICS §5).
pub struct MtpDev {
    pub layer: LayerDev, // layer id = n_trunk: ratio 0, score-routed, MXFP4 experts
    pub enorm: CudaSlice<f32>,
    pub hnorm: CudaSlice<f32>,
    pub norm: CudaSlice<f32>,
    pub e_proj: CudaSlice<u8>, // bf16
    pub h_proj: CudaSlice<u8>, // bf16
    pub hc_head_fn: CudaSlice<f32>,
    pub hc_head_base: Vec<f32>,
    pub hc_head_scale: Vec<f32>,
}

/// DSpark drafter on the LAST stage (iteration 3; semantics DSPARK-SEMANTICS.md,
/// numeric truth = the lane-10 CPU oracle `memra_gguf::dsv4_dspark`). Loaded only
/// under MEMRA_DSV4_DRAFTER=dspark (≈10.7 GiB resident on dev1 — VRAM plan in the
/// iteration-3 receipts); config census pins ride the oracle's own
/// `DsparkConfig::load` (refuse-on-drift, NextN refusal included).
pub struct DsparkDev {
    /// mtp.0..2 — layer ids n_trunk+k, ratio 0 (window-only), score-routed MXFP4.
    pub blocks: Vec<LayerDev>,
    pub main_proj: CudaSlice<u8>, // bf16 [hidden, n_targets*hidden]
    pub main_norm: CudaSlice<f32>,
    pub norm: CudaSlice<f32>, // mtp.2.norm (exit head)
    /// markov factors held f32 at runtime (M:795-804 reference convention); the
    /// bias GEMV runs the f32-island dots kernel (f64 accumulation, oracle class).
    pub markov_w1: CudaSlice<f32>, // [vocab, rank]
    pub markov_w2: CudaSlice<f32>, // [vocab, rank]
    pub markov_w1_host: Vec<f32>, // host copy (row gather per chained id)
    pub conf_w: CudaSlice<f32>, // f32 [hidden + rank] (fp32 head, M:810)
    pub hc_head_fn: CudaSlice<f32>, // mtp.2 hc_head trio
    pub hc_head_base: Vec<f32>,
    pub hc_head_scale: Vec<f32>,
    pub block_size: usize,
    pub noise_token: u32,
    pub targets: Vec<usize>, // [40, 41, 42]
    pub rank: usize,
    pub vocab: usize,
}

/// DSpark decode-side state: the 3 per-block main_kv rings, each allocated
/// [win + block_size, hd] — rows [0, win) are the persistent ring (slot = pos % win,
/// M:783), rows [win, win+block) hold the CURRENT round's transient draft kv (the
/// M:784 cat([kv_cache, draft_kv]) gather realized in one buffer; rewritten every
/// propose, never read as ring). Rings advance ONLY for committed positions
/// (`dspark_write_rings`) — the §3.1 drafter rule.
pub struct DsparkState {
    pub rings: Vec<CudaSlice<f32>>,
    /// tap rows [t_max, n_targets*hidden] on the last stage: the hc-mean concat of
    /// layers 40/41/42, written by the decode step (and consumed by write_rings /
    /// forward_spec).
    pub taps: CudaSlice<f32>,
    /// Row in `taps` that carries the newest committed position. The speculative driver used
    /// to keep this cursor only on its stack; making it session state is what lets a parked
    /// conversation resume without re-prefilling the full prompt to reconstruct the drafter.
    pub tap_head: usize,
}

/// One drafter proposal (host view). `out_ids[0]` is the input token; margins/top1
/// are adjudication instruments (populated only under `capture`).
pub struct DsparkProposal {
    pub out_ids: Vec<u32>,
    pub confidence: Vec<f32>,
    pub margins: Vec<f32>,
    pub top1_logits: Vec<f32>,
    /// captured component arrays for the gate (dtoh): main_x, per-block outs,
    /// x_collapsed (pre-norm), post-markov logits rows, markov_embed.
    pub capture: Option<DsparkCaptureOut>,
}

pub struct DsparkCaptureOut {
    /// the trunk tap row itself (hc-mean concat of layers 40/41/42) — the CPU gate's
    /// `pos{p}_main_hidden` array; captured here so the GPU gate compares the SAME
    /// seven arrays the lane-10 CPU components gate does.
    pub main_hidden: Vec<f32>,
    pub main_x: Vec<f32>,
    pub block_outs: Vec<Vec<f32>>,
    pub x_collapsed: Vec<f32>,
    /// shared-trunk-head logits BEFORE any markov bias add (`pos{p}_logits_pre`).
    pub logits_pre: Vec<f32>,
    pub logits_post: Vec<f32>,
    pub markov_embed: Vec<f32>,
}

/// An exact reduction rewrite, encoded explicitly in the CUDA launch parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum Dsv4Fp4Reduce {
    #[default]
    Block = 0,
    Warp = 1,
}

impl Dsv4Fp4Reduce {
    pub fn resolve(value: Option<&str>) -> Res<Self> {
        match value {
            None | Some("") | Some("block") => Ok(Self::Block),
            Some("warp") => Ok(Self::Warp),
            Some(other) => Err(format!(
                "MEMRA_DSV4_FP4_REDUCE '{other}' unknown (block | warp)"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dsv4IndexerScore {
    #[default]
    Scalar,
    Tiled,
}

impl Dsv4IndexerScore {
    pub fn resolve(value: Option<&str>) -> Res<Self> {
        match value {
            None | Some("") | Some("scalar") => Ok(Self::Scalar),
            Some("tiled") => Ok(Self::Tiled),
            Some(other) => Err(format!(
                "MEMRA_DSV4_INDEXER_SCORE '{other}' unknown (scalar | tiled)"
            )),
        }
    }
}

fn resolve_prefill_moe(value: Option<&str>) -> Res<bool> {
    match value {
        None | Some("") | Some("reference") => Ok(false),
        Some("grouped") => Ok(true),
        Some(other) => Err(format!(
            "MEMRA_DSV4_PREFILL_MOE '{other}' unknown (reference | grouped)"
        )),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dsv4PrefillHead {
    #[default]
    All,
    Last,
}

impl Dsv4PrefillHead {
    pub fn resolve(value: Option<&str>) -> Res<Self> {
        match value {
            None | Some("") | Some("all") => Ok(Self::All),
            Some("last") => Ok(Self::Last),
            Some(other) => Err(format!(
                "MEMRA_DSV4_PREFILL_HEAD '{other}' unknown (all | last)"
            )),
        }
    }

    fn output(self, final_chunk: bool) -> VerifyOutput {
        match (self, final_chunk) {
            (Self::All, false) => VerifyOutput::Argmax,
            (Self::All, true) => VerifyOutput::Full,
            (Self::Last, false) => VerifyOutput::None,
            (Self::Last, true) => VerifyOutput::Last,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifyOutput {
    Full,
    Argmax,
    None,
    Last,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dsv4PrefillDraft {
    #[default]
    All,
    Tail,
}

impl Dsv4PrefillDraft {
    pub fn resolve(value: Option<&str>) -> Res<Self> {
        match value {
            None | Some("") | Some("all") => Ok(Self::All),
            Some("tail") => Ok(Self::Tail),
            Some(other) => Err(format!(
                "MEMRA_DSV4_PREFILL_DRAFT '{other}' unknown (all | tail)"
            )),
        }
    }

    fn keep_from(self, suffix_len: usize, window: usize) -> usize {
        match self {
            Self::All => 0,
            Self::Tail => suffix_len.saturating_sub(window),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dsv4DraftProposal {
    #[default]
    Greedy,
    Coupled,
}

impl Dsv4DraftProposal {
    pub fn resolve(value: Option<&str>) -> Res<Self> {
        match value {
            None | Some("") | Some("greedy") => Ok(Self::Greedy),
            Some("coupled") => Ok(Self::Coupled),
            Some(other) => Err(format!(
                "MEMRA_DSV4_DSPARK_PROPOSAL '{other}' unknown (greedy | coupled)"
            )),
        }
    }
}

fn dspark_draft_position(tap_position: usize, slot: usize) -> usize {
    // Input/head token is at tap_position+1. Draft slot 0 predicts the next one.
    tap_position + slot + 2
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dsv4PrefillHeadStats {
    pub full_rows: u64,
    pub last_rows: u64,
    pub skipped_chunks: u64,
    pub draft_prime_rows: u64,
}

#[derive(Default)]
struct PrefillHeadCounters {
    full_rows: std::sync::atomic::AtomicU64,
    last_rows: std::sync::atomic::AtomicU64,
    skipped_chunks: std::sync::atomic::AtomicU64,
    draft_prime_rows: std::sync::atomic::AtomicU64,
}

pub struct Dsv4Gpu {
    pub model: Dsv4Model,
    pub stages: Vec<Stage>,
    pub layer_stage: Vec<usize>, // trunk layer -> stage idx
    pub split_at: u32,           // first layer of stage 1
    pub max_seq: usize,
    pub variant: ActQuantVariant,
    pub fc_yarn_host: FreqsCis,
    pub fc_plain_host: FreqsCis,
    pub mtp: Option<MtpDev>,
    /// iteration 3: the DSpark drafter (0731 lineage), loaded under
    /// MEMRA_DSV4_DRAFTER=dspark; None = today's exact behavior everywhere.
    pub dspark: Option<DsparkDev>,
    pub expert_arm: ExpertArm,
    pub decode_path: DecodePath,
    /// lane 9 (owner ruling 2026-08-19): island dots on the DEVICE decode path run the
    /// f32-accumulation serving arm when true (fork-gated); false = the f64
    /// oracle-truth arm (MEMRA_DSV4_DOTS_ARM=f64). Legacy path and prefill NEVER
    /// consult this (they stay the pinned reference realizations).
    pub dots_f32: bool,
    /// 0731 re-gate extension rung — RATIFIED by the owner 2026-08-19 and now the
    /// DEFAULT (unset env == f32x): the remaining f64 dependency chains on the DEVICE
    /// decode path (sink scores/soft/out, rmsnorm, headrms, rowsq_scale,
    /// indexer_score) run f32-accumulation twins when true. false = those chains keep
    /// the f64 kernels (MEMRA_DSV4_DOTS_ARM=f64|f32 — oracle/debug arms, bytes
    /// untouched). hc_sinkhorn is NOT in f32x (never authorized). Legacy path and
    /// prefill NEVER consult this.
    pub chains_f32: bool,
    /// iteration-3 rung 4c MEASURED FORK (`MEMRA_DSV4_DSPARK_HEAD_ARM=f32x`, default
    /// f64 = the lane-10-gated bytes): the DSpark drafter's shared-trunk-head projection
    /// over block_size rows uses the f32-accumulation hoisted kernel instead of the f64
    /// one. Affects WHICH tokens are drafted, never the emitted stream (verification
    /// always emits the trunk's own argmax — the greedy identity law).
    pub dspark_head_f32: bool,
    /// DSpark-only fused selected-expert dispatch. Routing stays on the host
    /// oracle; only the per-expert launch loop is collapsed.
    pub dspark_fused_moe: bool,
    /// Chosen per model, never a process-global mutable test switch.
    fp4_reduce: Dsv4Fp4Reduce,
    indexer_score: Dsv4IndexerScore,
    prefill_grouped: bool,
    prefill_head: Dsv4PrefillHead,
    prefill_draft: Dsv4PrefillDraft,
    prefill_head_counts: PrefillHeadCounters,
    draft_proposal: Dsv4DraftProposal,
    coupled_draft_draws: std::sync::atomic::AtomicU64,
    /// iteration-5 FP8 dense arm (`MEMRA_DSV4_DENSE_ARM`; DEFAULT fp8 on the device
    /// decode path since the 2026-08-20 ratification, bf16 selectable and the legacy
    /// default): the DEVICE decode/verify paths read the FP8-blk linears as-stored
    /// (e4m3 + f32 block scales) through the bit-identical GEMV twins, halving the
    /// dense weight traffic (79.9% of a step's bytes). It5 ledger item 3: the trunk
    /// bf16 slabs are NOT device-resident under this arm — they hold [`DenseBf16::Host`]
    /// staged residency (same bytes, staged H2D per prefill pass); the legacy path is a
    /// boot refusal and the drafter's cuBLASLt linears keep resident bf16 (no twins).
    pub dense_fp8: bool,
    /// lane 8: cross-stage boundary events (peer transport), one per boundary,
    /// created in the TX stage's context (cuEventRecord requires event ctx == stream ctx).
    boundary_ev: Vec<cudarc::driver::CudaEvent>,
    hc_head_base: Vec<f32>,
    hc_head_scale: Vec<f32>,
}

/// A full-trunk forward's outputs: last-position logits + the final hc state (resident
/// on the LAST stage — the MTP drafter's input).
pub struct ForwardOut {
    pub logits: Vec<f32>,
    pub h_last: CudaSlice<f32>,
}

/// Lane-6 decode cache for ONE trunk layer, on the layer's owning stage. Layout mirrors
/// the reference (model.py:473-474, :491): `kvc` = [win + cap_blocks, hd] f32 with the
/// 128-slot window ring at rows [0, win) (slot = pos % win, M:530) and compressed block
/// j at row win + j (decode index offset = win, M:509). Pending state = RAW wkv/wgate
/// rows (ape added at pool time — see the lane-6 receipts): fine [2·ratio, latent] with
/// rows [0, ratio) = previous block / [ratio, 2·ratio) = current (M:344-370 state
/// machine); coarse [ratio, latent]. `pend_score` is initialized to −inf so a block
/// with no predecessor reproduces the reference j==0 masking bit-exactly.
pub struct LayerCache {
    pub kvc: CudaSlice<f32>,
    pub n_blocks: usize,
    pub pend_kv: Option<CudaSlice<f32>>,
    pub pend_score: Option<CudaSlice<f32>>,
    /// indexer compressed-kv store [cap_blocks, index_head_dim] (FP4-QAT'd values) +
    /// its own pending pair — fine layers only.
    pub ikvc: Option<CudaSlice<f32>>,
    pub i_blocks: usize,
    pub ipend_kv: Option<CudaSlice<f32>>,
    pub ipend_score: Option<CudaSlice<f32>>,
}

/// Lane-8 per-stage decode workspace: every per-step buffer preallocated ONCE (the
/// legacy path issues ~3,086 allocAsync+memset+free triplets per step — rung-0
/// profile). Every buffer is fully rewritten before it is read within a step; the
/// consumers (sink_attn via idx pads, combine via order, top-k via exact nb) read
/// exactly the regions written this step, so no per-step zeroing exists at all.
pub struct StepWs {
    pub h_a: CudaSlice<f32>, // [hc*hidden] layer io (in h_a -> h2 in h_b -> h3 in h_a)
    pub h_b: CudaSlice<f32>, // [hc*hidden]
    pub h_rx: CudaSlice<f32>, // [hc*hidden] boundary RX slot (peer TX writes here)
    pub emb: CudaSlice<f32>, // [hidden]
    pub mixes: CudaSlice<f32>, // [(2+hc)*hc]
    pub pre: CudaSlice<f32>, // [hc]
    pub post: CudaSlice<f32>, // [hc]
    pub comb: CudaSlice<f32>, // [hc*hc]
    pub y_hc: CudaSlice<f32>, // [hidden] hc_pre collapse out
    pub x: CudaSlice<f32>,   // [hidden] post-attn-norm
    pub xf: CudaSlice<f32>,  // [hidden] post-ffn-norm
    pub qr: CudaSlice<f32>,  // [q_lora]
    pub qr_b: CudaSlice<u8>, // [q_lora*2]
    pub q: CudaSlice<f32>,   // [heads*hd]
    pub kv: CudaSlice<f32>,  // [hd]
    pub qi: CudaSlice<f32>,  // [iheads*ihd]
    pub wproj: CudaSlice<f32>, // [iheads]
    pub score: CudaSlice<f32>, // [max_seq/ratio_min]
    pub topk_a: CudaSlice<u64>, // bounded hierarchical-selector scratch
    pub topk_b: CudaSlice<u64>,
    pub topk_stride: usize,
    pub idx: CudaSlice<i32>,      // [win + max(topk, max_seq/128)]
    pub o: CudaSlice<f32>,        // [heads*hd]
    pub o_b: CudaSlice<u8>,       // [heads*hd*2] (bf16 cvt of o, once — grouped wo reads slices)
    pub og: CudaSlice<f32>,       // [o_groups*o_lora]
    pub attn_out: CudaSlice<f32>, // [hidden]
    pub gemm_xb: CudaSlice<u8>,   // [max_gemm_k*2] per-call cvt scratch
    // MoE
    pub raw: CudaSlice<f32>,     // [ne]
    pub sel: CudaSlice<i32>,     // [topk]
    pub selw: CudaSlice<f32>,    // [topk]
    pub order: CudaSlice<i32>,   // [topk]
    pub xq: CudaSlice<u8>,       // [hidden]
    pub xs: CudaSlice<f32>,      // [hidden/128]
    pub g1: CudaSlice<f32>,      // [topk*inter]
    pub g3: CudaSlice<f32>,      // [topk*inter]
    pub hbuf: CudaSlice<f32>,    // [topk*inter]
    pub hq: CudaSlice<u8>,       // [topk*inter]
    pub hs: CudaSlice<f32>,      // [topk*inter/128]
    pub contrib: CudaSlice<f32>, // [topk*hidden]
    pub y: CudaSlice<f32>,       // [hidden]
    pub xb: CudaSlice<u8>,       // [hidden*2] shared-expert input (bf16 cvt of xf)
    pub sg1: CudaSlice<f32>,     // [sh_inter]
    pub sg3: CudaSlice<f32>,
    pub shbuf: CudaSlice<f32>,
    pub shb16: CudaSlice<u8>,   // [sh_inter*2]
    pub sh_out: CudaSlice<f32>, // [hidden]
    // compressor scratch (max class dims across attn fine/coarse + indexer)
    pub cmp_kv_row: CudaSlice<f32>, // [max latent]
    pub cmp_sc_row: CudaSlice<f32>, // [max latent]
    pub cmp_emit: CudaSlice<f32>,   // [2*max d]
    pub cmp_shift: CudaSlice<f32>,  // [max overlap ratio*latent]
    // sink attention (three-kernel split): scores/evals [heads, win+idx_tail], f64 den
    pub sink_scores: CudaSlice<f32>,
    pub sink_evals: CudaSlice<f32>,
    pub sink_den: CudaSlice<f64>,
    // head (allocated on every stage; consumed on the last)
    pub head_mixes: CudaSlice<f32>, // [hc]
    pub head_pre: CudaSlice<f32>,   // [hc]
    pub collapsed: CudaSlice<f32>,  // [hidden]
    pub logits: CudaSlice<f32>,     // [vocab]
    pub argmax: CudaSlice<i32>,     // [1]
    pub tok: CudaSlice<i32>,        // [1]
}

/// Whole-trunk decode state: one [`LayerCache`] per trunk layer + the stream position.
/// `pos` = tokens consumed so far (the next decode_step processes position `pos`).
pub struct DecodeState {
    pub caches: Vec<LayerCache>,
    pub pos: usize,
    /// Token capacity of this session's cache allocation. This is independently planned
    /// below the model-wide `Dsv4Gpu::max_seq`, so a 1M-capable server does not charge every
    /// short concurrent request for one million rows.
    pub capacity: usize,
    /// allocated cache bytes per device index (gate (e): measured vs design math)
    pub cache_bytes: Vec<u64>,
    /// Rows reserved after each layer's persistent compressed store for a batched
    /// speculative or chunked-prefill transaction.
    transient_rows: usize,
    /// lane 8: per-stage step workspace (Some iff the load-time decode path is Device)
    pub ws: Option<Vec<StepWs>>,
}

/// Number of persistent compressed rows admitted for one session. Decode allocation
/// and batched verification must share this exact planner because the latter places its
/// transient rows immediately after this store.
/// Engine bring-up ceiling. The serving surface separately retains its qualified limit.
pub const DSV4_BATCH_WIDTH_MAX: usize = 512;
pub const DSV4_SERVING_BATCH_WIDTH_MAX: usize = 64;

fn ring_commit_plan(pos0: usize, n_commit: usize, win: usize) -> (usize, Vec<i32>) {
    assert!(win > 0);
    let start = n_commit.saturating_sub(win);
    let slots = (start..n_commit)
        .map(|row| ((pos0 + row) % win) as i32)
        .collect();
    (start, slots)
}

fn dsv4_cache_cap_blocks(capacity: usize, ratio: usize) -> usize {
    capacity.checked_div(ratio).unwrap_or(0)
}

fn dsv4_split_for_tail_reserve(layer_bytes: &[u64], tail_reserve: u64) -> usize {
    assert!(
        layer_bytes.len() >= 2,
        "dsv4 PP2 needs at least two trunk layers"
    );
    let total: u64 = layer_bytes.iter().sum();
    let mut left = 0u64;
    let mut best = (u64::MAX, 1usize);
    for cut in 1..layer_bytes.len() {
        left += layer_bytes[cut - 1];
        let right = total - left + tail_reserve;
        let peak = left.max(right);
        // Equal estimated peaks choose the later cut: the tail stage owns DSpark and
        // the head, so preserving extra dynamic-cache headroom there is preferable.
        if peak < best.0 || (peak == best.0 && cut > best.1) {
            best = (peak, cut);
        }
    }
    best.1
}

/// One contiguous f32 range in a stage-owned pinned-host slab. DSV4 keeps its compact
/// long-context state on the layer's owning GPU while a session is active; this is the
/// lossless parked-session representation used to free that VRAM without expanding the
/// compressed state back into token-wise K/V. The range is in f32 elements, not bytes.
#[derive(Clone)]
struct Dsv4HostSpan {
    stage: usize,
    range: Range<usize>,
}

/// Field-for-field live-state map for one trunk layer. Append-only stores copy only their
/// high-water rows; SWA rings and compressor pending state copy in full. Verify-transient
/// rows and [`StepWs`] are scratch, so they are deliberately absent and freshly allocated
/// on restore.
struct Dsv4HostLayer {
    kvc: Dsv4HostSpan,
    n_blocks: usize,
    pend_kv: Option<Dsv4HostSpan>,
    pend_score: Option<Dsv4HostSpan>,
    ikvc: Option<Dsv4HostSpan>,
    i_blocks: usize,
    ipend_kv: Option<Dsv4HostSpan>,
    ipend_score: Option<Dsv4HostSpan>,
}

/// Pinned-host image of one DSV4 decode state. One allocation per pipeline stage keeps a
/// 1M-token park to two large DMA slabs rather than thousands of page-locked allocations.
/// The type is intentionally opaque: only [`Dsv4Gpu::snapshot_decode_state`] and
/// [`Dsv4Gpu::restore_decode_state`] may interpret its layout.
pub struct Dsv4HostDecodeState {
    stages: Vec<crate::PinnedHostBuf>,
    layers: Vec<Dsv4HostLayer>,
    pos: usize,
    capacity: usize,
    bytes: usize,
}

impl Dsv4HostDecodeState {
    /// Bytes resident in pinned host RAM (live compact state only).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Number of already-consumed tokens represented by this image.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// Pinned-host image of the small DSpark session state: three persistent 128-token rings and
/// the newest trunk tap. Draft-transient ring rows and the other tap rows are scratch and are
/// not copied. This is kept separate from [`Dsv4HostDecodeState`] so plain deployments pay
/// neither the allocation nor the metadata.
pub struct Dsv4HostDsparkState {
    slab: crate::PinnedHostBuf,
    rings: Vec<Range<usize>>,
    tap: Range<usize>,
    bytes: usize,
    block_size: usize,
}

impl Dsv4HostDsparkState {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

// ---------------------------------------------------------------- small launch helpers

fn dsv4_host_span(stage: usize, elems: usize, totals: &mut [usize]) -> Dsv4HostSpan {
    let start = totals[stage];
    totals[stage] = start
        .checked_add(elems)
        .expect("dsv4 host-state element count overflow");
    Dsv4HostSpan {
        stage,
        range: start..start + elems,
    }
}

fn dsv4_pinned_f32(buf: &crate::PinnedHostBuf, range: Range<usize>) -> &[f32] {
    let byte_start = range.start * std::mem::size_of::<f32>();
    let byte_end = range.end * std::mem::size_of::<f32>();
    assert!(byte_end <= buf.len(), "dsv4 pinned read outside slab");
    // SAFETY: cudaHostAlloc is page-aligned, byte_start is f32-aligned, and the bound above
    // proves the returned region lies wholly inside the owned allocation.
    unsafe {
        std::slice::from_raw_parts(
            buf.as_slice()[byte_start..byte_end].as_ptr() as *const f32,
            range.len(),
        )
    }
}

fn dsv4_pinned_f32_mut(buf: &mut crate::PinnedHostBuf, range: Range<usize>) -> &mut [f32] {
    let byte_start = range.start * std::mem::size_of::<f32>();
    let byte_end = range.end * std::mem::size_of::<f32>();
    assert!(byte_end <= buf.len(), "dsv4 pinned write outside slab");
    // SAFETY: same alignment and ownership proof as dsv4_pinned_f32; the mutable borrow of
    // `buf` guarantees this range cannot alias another host slice in this call.
    unsafe {
        std::slice::from_raw_parts_mut(
            buf.as_mut_slice()[byte_start..byte_end].as_mut_ptr() as *mut f32,
            range.len(),
        )
    }
}

fn dsv4_dtoh_span(
    stream: &std::sync::Arc<CudaStream>,
    src: &CudaSlice<f32>,
    slab: &mut crate::PinnedHostBuf,
    span: &Dsv4HostSpan,
) -> Res<()> {
    let n = span.range.len();
    if n > src.len() {
        return Err(format!(
            "dsv4 host snapshot range {n} exceeds device plane {}",
            src.len()
        ));
    }
    if n == 0 {
        return Ok(());
    }
    let host = dsv4_pinned_f32_mut(slab, span.range.clone());
    stream
        .memcpy_dtoh(&src.slice(0..n), host)
        .map_err(e("dsv4 state dtoh"))
}

fn dsv4_htod_span(
    stream: &std::sync::Arc<CudaStream>,
    slab: &crate::PinnedHostBuf,
    span: &Dsv4HostSpan,
    dst: &mut CudaSlice<f32>,
) -> Res<()> {
    let host = dsv4_pinned_f32(slab, span.range.clone());
    if host.len() > dst.len() {
        return Err(format!(
            "dsv4 host restore range {} exceeds device plane {}",
            host.len(),
            dst.len()
        ));
    }
    if host.is_empty() {
        return Ok(());
    }
    let mut view = dst.slice_mut(0..host.len());
    stream
        .memcpy_htod(host, &mut view)
        .map_err(e("dsv4 state htod"))
}

fn sp(stream: &CudaStream) -> *mut c_void {
    stream.cu_stream() as *mut c_void
}

fn upload_f32(stream: &std::sync::Arc<CudaStream>, v: &[f32]) -> Res<CudaSlice<f32>> {
    let mut d = stream.alloc_zeros::<f32>(v.len()).map_err(e("alloc f32"))?;
    stream.memcpy_htod(v, &mut d).map_err(e("htod f32"))?;
    Ok(d)
}

fn upload_i32(stream: &std::sync::Arc<CudaStream>, v: &[i32]) -> Res<CudaSlice<i32>> {
    let mut d = stream.alloc_zeros::<i32>(v.len()).map_err(e("alloc i32"))?;
    stream.memcpy_htod(v, &mut d).map_err(e("htod i32"))?;
    Ok(d)
}

fn upload_u64(stream: &std::sync::Arc<CudaStream>, v: &[u64]) -> Res<CudaSlice<u64>> {
    let mut d = stream.alloc_zeros::<u64>(v.len()).map_err(e("alloc u64"))?;
    stream.memcpy_htod(v, &mut d).map_err(e("htod u64"))?;
    Ok(d)
}

fn upload_u8(stream: &std::sync::Arc<CudaStream>, v: &[u8]) -> Res<CudaSlice<u8>> {
    let mut d = stream.alloc_zeros::<u8>(v.len()).map_err(e("alloc u8"))?;
    stream.memcpy_htod(v, &mut d).map_err(e("htod u8"))?;
    Ok(d)
}

fn dtoh_f32(stream: &std::sync::Arc<CudaStream>, d: &CudaSlice<f32>) -> Res<Vec<f32>> {
    let mut v = vec![0f32; d.len()];
    stream.memcpy_dtoh(d, &mut v[..]).map_err(e("dtoh"))?;
    stream.synchronize().map_err(e("sync dtoh"))?;
    Ok(v)
}

// -------------------------------------------------- lane 8: peer byte-integrity probe
//
// (lane/hermes-perf-fixes, 2026-08-23 — the "DSv4 device-path PP copies hidden state with no
// peer byte probe" finding.) The lane-8 setup used to cuCtxEnablePeerAccess +
// cuMemPoolSetAccess and eprintln success; the pp.rs boot probe exists precisely because a
// fabric can grant peer access and still corrupt bytes in flight (Pod B: official simpleP2P
// reproduced it while bandwidth-test returned rc=0). The probe here runs the PRODUCTION
// program — stream-ordered pool allocations moved by the exact `memcpy_peer_async`-on-the-
// TX-stream call shape the boundary copy uses (the cx-peerprobe lesson: probing legacy
// cuMemAlloc buffers validates a different allocation class) — over every cross-device
// boundary, both directions, on a byte ladder up to the prefill hidden-state payload class.
// FAIL-CLOSED: dsv4's device PP path has no host-bounce twin, so a mismatch refuses at load.

/// Deterministic per-(bytes, boundary, src, dst) xorshift pattern (pp.rs idiom): a stuck or
/// crossed lane cannot alias another probe's expected bytes.
fn dsv4_peer_probe_pattern(
    bytes: usize,
    boundary: usize,
    src_dev: usize,
    dst_dev: usize,
) -> Vec<u8> {
    let mut state = 0xD1B5_4A32_D192_ED03u64
        ^ (bytes as u64).rotate_left(7)
        ^ (boundary as u64).rotate_left(19)
        ^ (src_dev as u64).rotate_left(31)
        ^ (dst_dev as u64).rotate_left(43);
    (0..bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn dsv4_peer_probe_mismatches(expected: &[u8], readback: &[u8]) -> usize {
    expected
        .iter()
        .zip(readback)
        .filter(|(a, b)| a != b)
        .count()
        + expected.len().abs_diff(readback.len())
}

fn dsv4_peer_probe_ladder(hidden: usize, hc: usize) -> Vec<usize> {
    let hc_state = hidden * hc * std::mem::size_of::<f32>();
    let mut ladder = vec![
        16 << 10,
        hc_state,
        8 * hc_state,
        1 << 20,
        (4096usize * hidden * std::mem::size_of::<f32>()).min(64 << 20),
    ];
    ladder.sort_unstable();
    ladder.dedup();
    ladder
}

/// One probed copy src->dst at `bytes`. Destination is poisoned with the inverted pattern
/// first, so a silently dropped copy reads back as full-length corruption, never as PASS.
fn dsv4_peer_probe_copy(src: &Stage, dst: &Stage, boundary: usize, bytes: usize) -> Res<()> {
    let expected = dsv4_peer_probe_pattern(bytes, boundary, src.dev, dst.dev);
    src.gpu.ctx.bind_to_thread().map_err(e("probe bind src"))?;
    let src_stream = src.gpu.stream();
    let src_buf = upload_u8(&src_stream, &expected)?;
    src_stream.synchronize().map_err(e("probe sync src htod"))?;

    dst.gpu.ctx.bind_to_thread().map_err(e("probe bind dst"))?;
    let dst_stream = dst.gpu.stream();
    let poison: Vec<u8> = expected.iter().map(|b| !b).collect();
    let mut dst_buf = upload_u8(&dst_stream, &poison)?;
    dst_stream.synchronize().map_err(e("probe sync poison"))?;

    // the production call shape: peer copy issued on the TX (source) stream.
    src.gpu.ctx.bind_to_thread().map_err(e("probe bind tx"))?;
    {
        let (sp, _g0) = src_buf.device_ptr(&src_stream);
        let (dp, _g1) = dst_buf.device_ptr_mut(&src_stream);
        unsafe {
            cudarc::driver::result::memcpy_peer_async(
                dst.gpu.ctx.cu_ctx(),
                dp,
                src.gpu.ctx.cu_ctx(),
                sp,
                bytes,
                src_stream.cu_stream(),
            )
            .map_err(e("probe peer copy"))?;
        }
    }
    src_stream.synchronize().map_err(e("probe sync copy"))?;

    dst.gpu.ctx.bind_to_thread().map_err(e("probe bind rx"))?;
    let mut readback = vec![0u8; bytes];
    dst_stream
        .memcpy_dtoh(&dst_buf, &mut readback[..])
        .map_err(e("probe readback"))?;
    dst_stream.synchronize().map_err(e("probe sync readback"))?;
    // TEETH DOOR (diagnostics only, never a tuning knob): MEMRA_DSV4_PEER_PROBE_POISON=1
    // flips one readback byte so the refusal arm can be proven live on a healthy fabric —
    // a probe that can only be observed passing proves nothing (serve-stress-gate law).
    if std::env::var("MEMRA_DSV4_PEER_PROBE_POISON").as_deref() == Ok("1") && !readback.is_empty() {
        readback[0] ^= 1;
    }
    let mismatches = dsv4_peer_probe_mismatches(&expected, &readback);
    if mismatches == 0 {
        Ok(())
    } else {
        Err(format!("{mismatches} mismatched byte(s) of {bytes}"))
    }
}

// ================================================== iteration-5: drafted-round phase instruments
//
// WHY: iteration 4 measured `cost(T) = F + 0.272*T` plain steps with F = 1.057 plain steps on
// the f32x exit head, proved the marginal term is ~65-70% irreducible expert-union traffic, and
// showed the ENTIRE drafted gap to the bar is F. F cannot be attacked until it is itemised into
// named components with sizes, which is what these two instruments produce. Both are OFF by
// default and their env knobs are read ONCE through a `OnceLock` (never per round), so the
// shipping path is untouched: with both unset `Dsv4Phase::new` returns `None` before any work.
//
//   MEMRA_DSV4_ROUND_PROFILE=1 -- sync-bracketed host timers. Every phase boundary
//       synchronizes the head stage's stream, so per-phase wall times SUM to the round's wall
//       time and can be quoted in F's own unit (plain steps). It PERTURBS: the added syncs
//       expose latency a queued round would have overlapped, so the report always prints the
//       bracketed round total for comparison against the unbracketed A/B baseline. A
//       sync-bracketed run is a rung-0 instrument, NEVER an A/B observation.
//
//   MEMRA_DSV4_NVTX=1 -- NVTX push/pop only, no added syncs, so the round is undisturbed.
//       `nsys profile -t cuda,nvtx` then gives `nvtx_gpu_proj_sum` (GPU-busy attributed to the
//       range that launched each op) and `nvtx_sum` (host wall per range). GPU-busy is the real
//       kernel work; wall minus GPU-busy inside a sync-terminated phase is the exposed stall.
//
// The accumulator is thread-local and the phase stack makes nesting exact: each row keeps
// INCLUSIVE time plus the time its direct children consumed, so `self = inclusive - children`
// is a true exclusive cost and the leaves partition the round.
#[derive(Default, Clone)]
struct Dsv4PhaseAcc {
    /// (label, inclusive_us, direct_child_us, calls)
    rows: Vec<(&'static str, u64, u64, u64)>,
    /// (row index, direct-child us accumulated for the open range)
    stack: Vec<(usize, u64)>,
}

thread_local! {
    static DSV4_PHASES: std::cell::RefCell<Dsv4PhaseAcc> =
        std::cell::RefCell::new(Dsv4PhaseAcc::default());
}

fn dsv4_prof_sync() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MEMRA_DSV4_ROUND_PROFILE").as_deref() == Ok("1"))
}

fn dsv4_prof_nvtx() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MEMRA_DSV4_NVTX").as_deref() == Ok("1"))
}

/// True when either phase instrument is armed. Checked first in `Dsv4Phase::new` so an
/// unprofiled build pays one relaxed load per bracket and nothing else.
pub fn dsv4_prof_on() -> bool {
    dsv4_prof_sync() || dsv4_prof_nvtx()
}

/// A named, nestable phase bracket. Constructed through the `phase!` macro, which supplies a
/// NUL-terminated literal so the NVTX push needs no allocation.
pub struct Dsv4Phase<'a> {
    stream: Option<&'a std::sync::Arc<CudaStream>>,
    t0: std::time::Instant,
    nvtx: bool,
}

impl<'a> Dsv4Phase<'a> {
    /// `name` MUST end in `\0` (use the `phase!` macro). `stream` is the stream whose queue
    /// this phase's work rides; it is synchronized on drop under `MEMRA_DSV4_ROUND_PROFILE=1`
    /// and ignored otherwise.
    pub fn new(name: &'static str, stream: Option<&'a std::sync::Arc<CudaStream>>) -> Option<Self> {
        if !dsv4_prof_on() {
            return None;
        }
        let nvtx = dsv4_prof_nvtx();
        if nvtx {
            unsafe {
                k::memra_dsv4_nvtx_push(name.as_ptr() as *const std::os::raw::c_char);
            }
        }
        let label = &name[..name.len() - 1];
        DSV4_PHASES.with(|p| {
            let mut p = p.borrow_mut();
            let idx = match p.rows.iter().position(|r| r.0 == label) {
                Some(i) => i,
                None => {
                    p.rows.push((label, 0, 0, 0));
                    p.rows.len() - 1
                }
            };
            p.stack.push((idx, 0));
        });
        Some(Dsv4Phase {
            stream: if dsv4_prof_sync() { stream } else { None },
            t0: std::time::Instant::now(),
            nvtx,
        })
    }
}

impl Drop for Dsv4Phase<'_> {
    fn drop(&mut self) {
        // sync BEFORE stopping the clock: under the sync-bracketed instrument the phase's cost
        // includes the GPU work it queued, which is the only way the rows can sum to the round.
        if let Some(s) = self.stream {
            let _ = s.synchronize();
        }
        let us = self.t0.elapsed().as_micros() as u64;
        if self.nvtx {
            unsafe {
                k::memra_dsv4_nvtx_pop();
            }
        }
        DSV4_PHASES.with(|p| {
            let mut p = p.borrow_mut();
            if let Some((idx, child)) = p.stack.pop() {
                let r = &mut p.rows[idx];
                r.1 += us;
                r.2 += child;
                r.3 += 1;
                if let Some(top) = p.stack.last_mut() {
                    top.1 += us;
                }
            }
        });
    }
}

/// Bracket a phase. `phase!("name", stream_opt)` -> `Option<Dsv4Phase>`; bind it to a `_p`
/// local so it drops at the end of the scope.
macro_rules! phase {
    ($name:literal, $stream:expr) => {
        crate::dsv4_gpu::Dsv4Phase::new(concat!($name, "\0"), $stream)
    };
}

/// Print the accumulated itemisation. `plain_us` is the measured PLAIN step wall time so each
/// row can be quoted in plain steps, which is the unit `F` is expressed in; pass 0.0 to omit.
pub fn dsv4_phase_report(tag: &str, rounds: u64, plain_us: f64) {
    DSV4_PHASES.with(|p| {
        let p = p.borrow();
        if p.rows.is_empty() {
            return;
        }
        let mode = if dsv4_prof_sync() {
            "sync-bracketed (PERTURBS: compare the round total against the unbracketed A/B)"
        } else {
            "nvtx-only (host wall; GPU-busy comes from nsys nvtx_gpu_proj_sum)"
        };
        println!("\n[phase] === F ITEMISATION: {tag} ===");
        println!("[phase] rounds={rounds}  plain step={plain_us:.1} us  mode={mode}");
        println!(
            "[phase] {:<26} {:>11} {:>11} {:>9} {:>12} {:>12}",
            "phase", "incl_us/rd", "self_us/rd", "calls/rd", "self_plainstp", "incl_plainstp"
        );
        let mut rows = p.rows.clone();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.saturating_sub(r.2)));
        let r = rounds.max(1) as f64;
        let mut leaf_sum = 0f64;
        for (name, incl, child, calls) in rows {
            let selfus = incl.saturating_sub(child) as f64 / r;
            let inclus = incl as f64 / r;
            leaf_sum += selfus;
            let (sp, ip) = if plain_us > 0.0 {
                (selfus / plain_us, inclus / plain_us)
            } else {
                (0.0, 0.0)
            };
            println!(
                "[phase] {name:<26} {inclus:>11.1} {selfus:>11.1} {:>9.2} {sp:>12.4} {ip:>12.4}",
                calls as f64 / r
            );
        }
        println!(
            "[phase] {:<26} {:>11} {:>11.1} {:>9} {:>12.4}",
            "SUM of self",
            "",
            leaf_sum,
            "",
            if plain_us > 0.0 {
                leaf_sum / plain_us
            } else {
                0.0
            }
        );
    });
}

/// `MEMRA_DSV4_DSPARK_CHAIN=device` keeps the DSpark markov chain resident on the device (see
/// `dspark_forward_spec`). Default (`host`, or unset) reproduces the pre-iteration-5 transport
/// exactly, including its ten per-round stream drains, so the shipped arm is unchanged until an
/// A/B and the gate battery say otherwise.
fn dsv4_dspark_chain_device() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let on = std::env::var("MEMRA_DSV4_DSPARK_CHAIN").as_deref() == Ok("device");
        if on {
            println!(
                "[spec] DSpark markov chain RESIDENT ON DEVICE (MEMRA_DSV4_DSPARK_CHAIN=device): \
                 one D2H per round instead of 2 x block_size"
            );
        }
        on
    })
}

/// `MEMRA_DSV4_DSPARK_MARKOV=rowblk` runs the DSpark markov bias GEMV through the row-blocked
/// twin of the f64 island dots. Bit-identical output (same accumulation order and reduction tree,
/// only R rows share a block), so this is a pure geometry change; the default `base` keeps the
/// shipped kernel. Measured defect it addresses: 5 x 318 us/round at 416 GB/s = 26% of roofline,
/// latency-bound on one 7-level reduction tree per 1 KB of weights read.
fn dsv4_dspark_markov_rowblk() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let on = std::env::var("MEMRA_DSV4_DSPARK_MARKOV").as_deref() == Ok("rowblk");
        if on {
            println!(
                "[spec] DSpark markov bias GEMV on the ROW-BLOCKED dots twin \
                 (MEMRA_DSV4_DSPARK_MARKOV=rowblk; bit-identical, geometry only)"
            );
        }
        on
    })
}

/// Drop everything accumulated so far (used to keep the plain arm's brackets out of the
/// drafted arm's table).
pub fn dsv4_phase_reset() {
    DSV4_PHASES.with(|p| *p.borrow_mut() = Dsv4PhaseAcc::default());
}

macro_rules! dp {
    ($slice:expr, $stream:expr) => {{ $slice.device_ptr($stream).0 as *const c_void }};
}
macro_rules! dpf {
    ($slice:expr, $stream:expr) => {{ $slice.device_ptr($stream).0 as *const f32 }};
}
macro_rules! dpm {
    ($slice:expr, $stream:expr) => {{ $slice.device_ptr_mut($stream).0 as *mut f32 }};
}

// ---------------------------------------------------------------- loading

/// f32 (already NaN-checked by tensor_f32) -> bf16 with a bit-level exactness REFUSAL:
/// every value in the lane-4 rungs is exactly representable (see receipts); a non-zero
/// low half means the exactness proof broke and the load must stop, not round.
fn f32_to_bf16_exact(name: &str, v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for (i, x) in v.iter().enumerate() {
        let bits = x.to_bits();
        assert!(
            bits & 0xFFFF == 0,
            "{name}: element {i} = {x} not exactly representable in bf16 — lane-4 rung \
             exactness violated"
        );
        out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
    }
    out
}

impl Dsv4Gpu {
    pub fn coupled_draft_draws(&self) -> u64 {
        self.coupled_draft_draws
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Exclusive measurement seam; greedy verification never consults this policy.
    pub fn set_draft_proposal_for_gate(
        &mut self,
        arm: Dsv4DraftProposal,
    ) -> Res<Dsv4DraftProposal> {
        for stage in &self.stages {
            stage
                .gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind proposal gate"))?;
            stage
                .gpu
                .stream()
                .synchronize()
                .map_err(e("drain proposal gate"))?;
        }
        let previous = self.draft_proposal;
        self.draft_proposal = arm;
        Ok(previous)
    }

    pub fn prefill_head_stats(&self) -> Dsv4PrefillHeadStats {
        use std::sync::atomic::Ordering::Relaxed;
        Dsv4PrefillHeadStats {
            full_rows: self.prefill_head_counts.full_rows.load(Relaxed),
            last_rows: self.prefill_head_counts.last_rows.load(Relaxed),
            skipped_chunks: self.prefill_head_counts.skipped_chunks.load(Relaxed),
            draft_prime_rows: self.prefill_head_counts.draft_prime_rows.load(Relaxed),
        }
    }

    /// Exclusive gate seam. No serving request can switch the model's head policy.
    pub fn set_prefill_head_for_gate(&mut self, arm: Dsv4PrefillHead) -> Res<Dsv4PrefillHead> {
        if !matches!(self.decode_path, DecodePath::Device { .. }) {
            return Err("prefill head gate requires device decode".into());
        }
        for stage in &self.stages {
            stage
                .gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind prefill head gate"))?;
            stage
                .gpu
                .stream()
                .synchronize()
                .map_err(e("drain prefill head gate"))?;
        }
        let previous = self.prefill_head;
        self.prefill_head = arm;
        Ok(previous)
    }

    /// Exclusive gate seam for the chunked DSpark final-window prime.
    pub fn set_prefill_draft_for_gate(&mut self, arm: Dsv4PrefillDraft) -> Res<Dsv4PrefillDraft> {
        if !matches!(self.decode_path, DecodePath::Device { .. }) {
            return Err("prefill draft gate requires device decode".into());
        }
        for stage in &self.stages {
            stage
                .gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind prefill draft gate"))?;
            stage
                .gpu
                .stream()
                .synchronize()
                .map_err(e("drain prefill draft gate"))?;
        }
        let previous = self.prefill_draft;
        self.prefill_draft = arm;
        Ok(previous)
    }

    /// One-load experimental gate control. No request may mutate the model arm.
    pub fn set_prefill_grouped_for_gate(&mut self, enabled: bool) -> Res<bool> {
        if !matches!(self.decode_path, DecodePath::Device { host_math: false }) {
            return Err("grouped prefill gate requires the device path".into());
        }
        for stage in &self.stages {
            stage
                .gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind grouped gate stage"))?;
            stage
                .gpu
                .stream()
                .synchronize()
                .map_err(e("drain grouped gate stage"))?;
        }
        let previous = self.prefill_grouped;
        self.prefill_grouped = enabled;
        Ok(previous)
    }

    /// Exclusive one-load gate seam. Persistent graph support must include this
    /// arm in its executable key before it can reuse graphs across this setter.
    pub fn set_indexer_score_for_gate(&mut self, arm: Dsv4IndexerScore) -> Res<Dsv4IndexerScore> {
        if !matches!(self.decode_path, DecodePath::Device { host_math: false }) || !self.chains_f32
        {
            return Err("indexer gate requires device f32x path".into());
        }
        if self.model.cfg().index_n_heads != 64 || self.model.cfg().index_head_dim != 128 {
            return Err("tiled indexer requires 64 heads of width 128".into());
        }
        for stage in &self.stages {
            stage
                .gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind indexer gate stage"))?;
            stage
                .gpu
                .stream()
                .synchronize()
                .map_err(e("drain indexer gate stage"))?;
        }
        let previous = self.indexer_score;
        self.indexer_score = arm;
        Ok(previous)
    }

    /// Isolated gate control, not a serving request option. Requires exclusive
    /// model access and drains both stage streams before changing the next launch.
    /// DSV4 currently owns no captured graph; future graph support must invalidate
    /// or re-capture its executables here before this seam may be used with it.
    pub fn set_fp4_reduce_for_gate(&mut self, arm: Dsv4Fp4Reduce) -> Res<Dsv4Fp4Reduce> {
        if !matches!(self.decode_path, DecodePath::Device { .. }) {
            return Err("FP4 reduction gate requires the native device path".to_string());
        }
        for stage in &self.stages {
            stage
                .gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind FP4 gate stage"))?;
            stage
                .gpu
                .stream()
                .synchronize()
                .map_err(e("drain FP4 gate stage"))?;
        }
        let previous = self.fp4_reduce;
        self.fp4_reduce = arm;
        Ok(previous)
    }

    /// Upload a tensor as bf16: BF16-stored tensors ride raw bytes; FP8-blk tensors are
    /// host-dequantized (lane-1 decoder) and cast with the exactness refusal.
    fn tensor_bf16(&mut self, stage: usize, name: &str) -> Res<CudaSlice<u8>> {
        let raw_name = format!("{name}.weight");
        let is_bf16_raw = self
            .model
            .st
            .raw(&raw_name)
            .map(|(i, _)| i.dtype == "BF16")
            .unwrap_or(false)
            || self
                .model
                .st
                .raw(name)
                .map(|(i, _)| i.dtype == "BF16")
                .unwrap_or(false);
        let stream = self.stages[stage].gpu.stream();
        let bytes: u64;
        let out = if is_bf16_raw {
            let (_, raw) = self
                .model
                .st
                .raw(&raw_name)
                .or_else(|| self.model.st.raw(name))
                .unwrap();
            bytes = raw.len() as u64;
            upload_u8(&stream, raw)?
        } else {
            let (_, v) = self.model.tensor_f32(name);
            let b = f32_to_bf16_exact(name, &v);
            bytes = b.len() as u64;
            upload_u8(&stream, &b)?
        };
        self.stages[stage].loaded_bytes += bytes;
        Ok(out)
    }

    /// Iteration-5 FP8 dense arm loader. bf16 arm (or no fp8 twin): the device-resident
    /// bf16 dequant slab, today's exact bytes. fp8 arm on an F8_E4M3-stored `fp8_ok`
    /// tensor (trunk layers only this rung): the as-stored codes + host-decoded f32
    /// scale grid go to the device, and the bf16 slab drops to STAGED residency
    /// ([`DenseBf16::Host`], it5 ledger item 3) — the fp8 twins own every device
    /// decode/verify read and prefill stages the same bytes per pass, so the
    /// +~2.7 GiB/card dual residency is gone. Load-time refusals: missing/mis-shaped
    /// scale grid, e8m0 NaN code, cols not a multiple of 8 (the uint2 chunk contract),
    /// and a 1,024-element stride-sampled BIT check
    /// `e4m3(code[r,c]) * sc[r/128, c/128] == host_dequant[r,c]` — the layout/indexing
    /// proof, in the load-refusal tradition of the bf16 slab's own exactness check.
    fn tensor_dense(
        &mut self,
        stage: usize,
        name: &str,
        fp8_ok: bool,
    ) -> Res<(DenseBf16, Option<Fp8Dense>)> {
        let raw_name = format!("{name}.weight");
        let is_bf16_raw = self
            .model
            .st
            .raw(&raw_name)
            .map(|(i, _)| i.dtype == "BF16")
            .unwrap_or(false)
            || self
                .model
                .st
                .raw(name)
                .map(|(i, _)| i.dtype == "BF16")
                .unwrap_or(false);
        if is_bf16_raw || !fp8_ok || !self.dense_fp8 {
            return Ok((DenseBf16::Dev(self.tensor_bf16(stage, name)?), None));
        }
        // FP8-blk path: resolve the weight raw + its scale sibling.
        let (wi, wraw, stem) = if let Some((i, r)) = self.model.st.raw(&raw_name) {
            (i.clone(), r.to_vec(), name.to_string())
        } else {
            let (i, r) = self
                .model
                .st
                .raw(name)
                .unwrap_or_else(|| panic!("missing dense tensor {name}"));
            let stem = name.strip_suffix(".weight").unwrap_or(name).to_string();
            (i.clone(), r.to_vec(), stem)
        };
        if wi.dtype != "F8_E4M3" {
            // not the FP8-blk class (e.g. a BF16-raw special) — bf16 slab only.
            return Ok((DenseBf16::Dev(self.tensor_bf16(stage, name)?), None));
        }
        assert_eq!(wi.shape.len(), 2, "{name}: fp8 dense tensor must be 2-D");
        let rows = wi.shape[0] as usize;
        let cols = wi.shape[1] as usize;
        assert_eq!(cols % 8, 0, "{name}: fp8 dense cols {cols} % 8 != 0");
        assert_eq!(wraw.len(), rows * cols, "{name}: fp8 byte count");
        let scale_name = format!("{stem}.scale");
        let (si, sraw) = self
            .model
            .st
            .raw(&scale_name)
            .unwrap_or_else(|| panic!("{name}: F8_E4M3 weight without {scale_name}"));
        assert_eq!(si.dtype, "F8_E8M0", "{scale_name}: dtype");
        let sc_rows = rows.div_ceil(128);
        let sc_cols = cols.div_ceil(128);
        assert_eq!(
            (si.shape[0] as usize, si.shape[1] as usize),
            (sc_rows, sc_cols),
            "{scale_name}: scale grid shape vs [ceil({rows}/128), ceil({cols}/128)]"
        );
        let sc_f32: Vec<f32> = sraw
            .iter()
            .map(|&b| {
                assert_ne!(b, 0xFF, "{scale_name}: e8m0 NaN code");
                memra_gguf::dsv4::e8m0_to_f32(b)
            })
            .collect();
        // host dequant (the bf16 slab's own source) + the sampled layout bit-check.
        let (_, v) = self.model.tensor_f32(name);
        assert_eq!(v.len(), rows * cols, "{name}: dequant len");
        let step = (v.len() / 1024).max(1);
        for idx in (0..v.len()).step_by(step) {
            let (r, c) = (idx / cols, idx % cols);
            let got = memra_gguf::nvfp4_repack::fp8_e4m3_to_f32(wraw[idx])
                * sc_f32[(r / 128) * sc_cols + c / 128];
            assert_eq!(
                got.to_bits(),
                v[idx].to_bits(),
                "{name}: fp8 arm layout check failed at [{r},{c}] ({got} vs {})",
                v[idx]
            );
        }
        let b = f32_to_bf16_exact(name, &v);
        let stream = self.stages[stage].gpu.stream();
        // item 3: the bf16 slab is NOT uploaded — the fp8 pair owns every device
        // decode/verify read (dwsel) and prefill stages `b` per pass. loaded_bytes
        // counts DEVICE bytes only, so vram_report stays honest.
        let codes = upload_u8(&stream, &wraw)?;
        let scales = upload_f32(&stream, &sc_f32)?;
        self.stages[stage].loaded_bytes += (wraw.len() + sc_f32.len() * 4) as u64;
        Ok((
            DenseBf16::Host(b),
            Some(Fp8Dense {
                codes,
                scales,
                sc_cols,
                rows,
                cols,
            }),
        ))
    }

    /// Upload a tensor as f32 (islands): any storage dtype goes through the proven
    /// tensor_f32 decode.
    fn tensor_f32_dev(&mut self, stage: usize, name: &str) -> Res<CudaSlice<f32>> {
        let (_, v) = self.model.tensor_f32(name);
        let stream = self.stages[stage].gpu.stream();
        self.stages[stage].loaded_bytes += (v.len() * 4) as u64;
        upload_f32(&stream, &v)
    }

    fn load_cmp(
        &mut self,
        stage: usize,
        prefix: &str,
        ratio: usize,
        d: usize,
        rotate: bool,
    ) -> Res<CmpDev> {
        let (wkv_shape, _) = self.model.tensor_f32(&format!("{prefix}.wkv.weight"));
        let latent = wkv_shape[0];
        let overlap = ratio == 4;
        assert_eq!(latent, if overlap { 2 * d } else { d }, "{prefix} latent");
        Ok(CmpDev {
            ratio,
            d,
            latent,
            overlap,
            rotate,
            wkv: self.tensor_f32_dev(stage, &format!("{prefix}.wkv.weight"))?,
            wgate: self.tensor_f32_dev(stage, &format!("{prefix}.wgate.weight"))?,
            norm: self.tensor_f32_dev(stage, &format!("{prefix}.norm.weight"))?,
            ape: self.tensor_f32_dev(stage, &format!("{prefix}.ape"))?,
        })
    }

    /// Load one block's device weights. `prefix` is "layers.N" for trunk, "mtp.0" for
    /// the MTP block (whose layer id is n_trunk — ratio 0, score-routed, MXFP4 experts).
    fn load_layer(&mut self, stage: usize, il: u32, prefix: &str) -> Res<LayerDev> {
        let d = self.model.cfg().clone();
        let moe = self.model.mc.moe.clone().expect("moe block");
        let ratio = d.compress_ratio(il) as usize;
        let hd = d.head_dim as usize;
        let p = prefix.to_string();
        let hash = d.is_hash_layer(il);
        let ne = moe.expert_count as usize;
        let inter = moe.expert_ff_length as usize;
        let hidden = self.model.mc.n_embd as usize;

        // hc host params
        let hc_load = |m: &Dsv4Model, fam: &str| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let fn_w = m.tensor_f32(&format!("{p}.hc_{fam}_fn")).1;
            let base = m.tensor_f32(&format!("{p}.hc_{fam}_base")).1;
            let scale = m.tensor_f32(&format!("{p}.hc_{fam}_scale")).1;
            (fn_w, base, scale)
        };
        let (attn_fn, attn_base, attn_scale) = hc_load(&self.model, "attn");
        let (ffn_fn, ffn_base, ffn_scale) = hc_load(&self.model, "ffn");
        let stream = self.stages[stage].gpu.stream();
        let hc_attn_fn = upload_f32(&stream, &attn_fn)?;
        let hc_ffn_fn = upload_f32(&stream, &ffn_fn)?;
        self.stages[stage].loaded_bytes += ((attn_fn.len() + ffn_fn.len()) * 4) as u64;

        // expert slab (as-stored quant bytes) — geometry derived from config; the recipe
        // is DETECTED from the stored dtype (U8+weight_scale+weight_scale_2 = modelopt
        // NVFP4; I8+scale = OCP MXFP4, the MTP experts) and refused on any surprise.
        let (wi0, _) = self
            .model
            .st
            .raw(&format!("{p}.ffn.experts.0.w1.weight"))
            .unwrap_or_else(|| panic!("missing {p}.ffn.experts.0.w1.weight"));
        let expert_kind = match wi0.dtype.as_str() {
            "U8" => ExpertKind::Nvfp4,
            "I8" => ExpertKind::Mxfp4,
            other => panic!("{p}: unexpected expert weight dtype {other}"),
        };
        let wbytes = inter * hidden / 2; // same for w1/w2/w3 (transposed dims)
        let sbytes = match expert_kind {
            ExpertKind::Nvfp4 => inter * hidden / 16,
            ExpertKind::Mxfp4 => inter * hidden / 32,
        };
        let mut experts_w = stream
            .alloc_zeros::<u8>(ne * 3 * wbytes)
            .map_err(e("alloc expert slab"))?;
        let mut experts_sc = stream
            .alloc_zeros::<u8>(ne * 3 * sbytes)
            .map_err(e("alloc expert scale slab"))?;
        let mut experts_s2 = Vec::with_capacity(ne * 3);
        for ex in 0..ne {
            for (pi, pname) in ["w1", "w2", "w3"].iter().enumerate() {
                let base = format!("{p}.ffn.experts.{ex}.{pname}");
                let (wi, wb) = self
                    .model
                    .st
                    .raw(&format!("{base}.weight"))
                    .unwrap_or_else(|| panic!("missing {base}.weight"));
                assert_eq!(wb.len(), wbytes, "{base}: weight bytes");
                let sb = match expert_kind {
                    ExpertKind::Nvfp4 => {
                        assert_eq!(wi.dtype, "U8", "{base}: expected NVFP4 U8 weight");
                        let (_, sb) = self
                            .model
                            .st
                            .raw(&format!("{base}.weight_scale"))
                            .unwrap_or_else(|| panic!("missing {base}.weight_scale"));
                        let (_, s2b) = self
                            .model
                            .st
                            .raw(&format!("{base}.weight_scale_2"))
                            .unwrap_or_else(|| panic!("missing {base}.weight_scale_2"));
                        let s2 = f32::from_le_bytes(s2b.try_into().expect("scale_2 4B"));
                        // pow2 refusal: the bf16-exactness proof of the on-the-fly dequant
                        // rung requires a pow2 scale_2 (receipts, "Quant rungs" §1).
                        assert!(
                            s2 > 0.0 && s2.to_bits() & 0x007F_FFFF == 0,
                            "{base}: scale_2 {s2} not a power of two — rung exactness violated"
                        );
                        experts_s2.push(s2);
                        sb
                    }
                    ExpertKind::Mxfp4 => {
                        assert_eq!(wi.dtype, "I8", "{base}: expected MXFP4 I8 weight");
                        let (si, sb) = self
                            .model
                            .st
                            .raw(&format!("{base}.scale"))
                            .unwrap_or_else(|| panic!("missing {base}.scale"));
                        assert_eq!(si.dtype, "F8_E8M0", "{base}: expected E8M0 scale");
                        // e8m0 0xFF is the NaN code — refuse at load, never zero a scale
                        assert!(
                            !sb.contains(&0xFFu8),
                            "{base}: E8M0 NaN scale code — refusing"
                        );
                        experts_s2.push(1.0);
                        sb
                    }
                };
                assert_eq!(sb.len(), sbytes, "{base}: scale bytes");
                let off = (ex * 3 + pi) * wbytes;
                let mut view = experts_w.slice_mut(off..off + wbytes);
                stream
                    .memcpy_htod(wb, &mut view)
                    .map_err(e("htod expert w"))?;
                let soff = (ex * 3 + pi) * sbytes;
                let mut sview = experts_sc.slice_mut(soff..soff + sbytes);
                stream
                    .memcpy_htod(sb, &mut sview)
                    .map_err(e("htod expert sc"))?;
            }
        }
        self.stages[stage].loaded_bytes +=
            (ne * 3 * (wbytes + sbytes)) as u64 + (ne * 3 * 4) as u64;

        let cmp = if ratio != 0 {
            Some(self.load_cmp(stage, &format!("{p}.attn.compressor"), ratio, hd, false)?)
        } else {
            None
        };
        // iteration-5 FP8 dense arm: trunk layers only this rung (the drafter/MTP
        // blocks ride the prefill helpers, which consume the bf16 slabs).
        let fp8_ok = p.starts_with("layers.");
        let idx = if d.has_indexer(il) {
            let heads = d.index_n_heads as usize;
            let ihd = d.index_head_dim as usize;
            let (iwq_b, iwq_b_fp8) =
                self.tensor_dense(stage, &format!("{p}.attn.indexer.wq_b"), fp8_ok)?;
            let (iwp, iwp_fp8) = self.tensor_dense(
                stage,
                &format!("{p}.attn.indexer.weights_proj.weight"),
                fp8_ok,
            )?;
            Some(IdxDev {
                wq_b: iwq_b,
                weights_proj: iwp,
                wq_b_fp8: iwq_b_fp8,
                weights_proj_fp8: iwp_fp8,
                cmp: self.load_cmp(
                    stage,
                    &format!("{p}.attn.indexer.compressor"),
                    ratio,
                    ihd,
                    true,
                )?,
                heads,
                hd: ihd,
                topk: d.index_topk as usize,
            })
        } else {
            None
        };

        // lane 8: device twins of the host routing/hc constants. tid2eid is validated
        // here ONCE (range + per-row distinctness — the checks the legacy route_host
        // asserts per token) because the device route kernel cannot refuse.
        let stream = self.stages[stage].gpu.stream();
        let hc_attn_base_dev = upload_f32(&stream, &attn_base)?;
        let hc_attn_scale_dev = upload_f32(&stream, &attn_scale)?;
        let hc_ffn_base_dev = upload_f32(&stream, &ffn_base)?;
        let hc_ffn_scale_dev = upload_f32(&stream, &ffn_scale)?;
        let gate_bias_host: Option<Vec<f32>> = if hash {
            None
        } else {
            Some(self.model.tensor_f32(&format!("{p}.ffn.gate.bias")).1)
        };
        let gate_bias_dev = match &gate_bias_host {
            Some(b) => Some(upload_f32(&stream, b)?),
            None => None,
        };
        let tid2eid_host: Option<Vec<i64>> = if hash {
            Some(self.model.tensor_i64(&format!("{p}.ffn.gate.tid2eid")).1)
        } else {
            None
        };
        let tid2eid_dev = match &tid2eid_host {
            Some(t) => {
                let topk = moe.expert_used_count as usize;
                assert_eq!(t.len() % topk, 0, "{p}: tid2eid rows");
                let mut t32 = Vec::with_capacity(t.len());
                for row in t.chunks(topk) {
                    let mut seen = std::collections::BTreeSet::new();
                    for &ex in row {
                        assert!(
                            (0..ne as i64).contains(&ex),
                            "{p}: tid2eid out of range at load"
                        );
                        assert!(seen.insert(ex), "{p}: duplicate expert id in tid2eid row");
                        t32.push(ex as i32);
                    }
                }
                Some(upload_i32(&stream, &t32)?)
            }
            None => None,
        };
        let experts_s2_dev = upload_f32(&stream, &experts_s2)?;
        let experts_modelopt_table = if expert_kind == ExpertKind::Nvfp4 {
            let (wp, _wg) = experts_w.device_ptr(&stream);
            let (sptr, _sg) = experts_sc.device_ptr(&stream);
            let mut pointers = vec![0u64; 6 * ne];
            for ex in 0..ne {
                for pi in 0..3 {
                    pointers[2 * pi * ne + ex] = wp + ((ex * 3 + pi) * wbytes) as u64;
                    pointers[(2 * pi + 1) * ne + ex] = sptr + ((ex * 3 + pi) * sbytes) as u64;
                }
            }
            Some(upload_u64(&stream, &pointers)?)
        } else {
            None
        };

        let (wq_a, wq_a_fp8) = self.tensor_dense(stage, &format!("{p}.attn.wq_a"), fp8_ok)?;
        let (wq_b, wq_b_fp8) = self.tensor_dense(stage, &format!("{p}.attn.wq_b"), fp8_ok)?;
        let (wkv, wkv_fp8) = self.tensor_dense(stage, &format!("{p}.attn.wkv"), fp8_ok)?;
        let (wo_a, wo_a_fp8) = self.tensor_dense(stage, &format!("{p}.attn.wo_a"), fp8_ok)?;
        let (wo_b, wo_b_fp8) = self.tensor_dense(stage, &format!("{p}.attn.wo_b"), fp8_ok)?;
        let (sw1, sw1_fp8) =
            self.tensor_dense(stage, &format!("{p}.ffn.shared_experts.w1"), fp8_ok)?;
        let (sw2, sw2_fp8) =
            self.tensor_dense(stage, &format!("{p}.ffn.shared_experts.w2"), fp8_ok)?;
        let (sw3, sw3_fp8) =
            self.tensor_dense(stage, &format!("{p}.ffn.shared_experts.w3"), fp8_ok)?;

        Ok(LayerDev {
            il,
            ratio,
            expert_kind,
            hc_attn_base_dev,
            hc_attn_scale_dev,
            hc_ffn_base_dev,
            hc_ffn_scale_dev,
            gate_bias_dev,
            tid2eid_dev,
            experts_s2_dev,
            experts_modelopt_table,
            wq_a,
            wq_b,
            wkv,
            wo_a,
            wo_b,
            wq_a_fp8,
            wq_b_fp8,
            wkv_fp8,
            wo_a_fp8,
            wo_b_fp8,
            q_norm: self.tensor_f32_dev(stage, &format!("{p}.attn.q_norm.weight"))?,
            kv_norm: self.tensor_f32_dev(stage, &format!("{p}.attn.kv_norm.weight"))?,
            attn_norm: self.tensor_f32_dev(stage, &format!("{p}.attn_norm.weight"))?,
            ffn_norm: self.tensor_f32_dev(stage, &format!("{p}.ffn_norm.weight"))?,
            sink: self.tensor_f32_dev(stage, &format!("{p}.attn.attn_sink"))?,
            cmp,
            idx,
            hc_attn_fn,
            hc_ffn_fn,
            hc_attn_base: attn_base,
            hc_attn_scale: attn_scale,
            hc_ffn_base: ffn_base,
            hc_ffn_scale: ffn_scale,
            gate_w: self.tensor_f32_dev(stage, &format!("{p}.ffn.gate.weight"))?,
            gate_bias: gate_bias_host,
            tid2eid: tid2eid_host,
            experts_w,
            experts_sc,
            experts_s2,
            shared_w: [sw1, sw2, sw3],
            shared_fp8: [sw1_fp8, sw2_fp8, sw3_fp8],
        })
    }

    /// Open the artifact and place the trunk across `devices`. `split_at` = first layer
    /// of stage 1, derived from per-layer byte math unless overridden.
    pub fn load(
        dir: &Path,
        devices: &[usize],
        variant: ActQuantVariant,
        max_seq: usize,
    ) -> Res<Self> {
        assert_eq!(devices.len(), 2, "lane 4 placement is a 2-card layer split");
        let fp4_reduce_env = match std::env::var("MEMRA_DSV4_FP4_REDUCE") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => return Err(format!("MEMRA_DSV4_FP4_REDUCE: {err}")),
        };
        let fp4_reduce = Dsv4Fp4Reduce::resolve(fp4_reduce_env.as_deref())?;
        let grouped_env = match std::env::var("MEMRA_DSV4_PREFILL_MOE") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => return Err(format!("MEMRA_DSV4_PREFILL_MOE: {err}")),
        };
        let prefill_grouped = resolve_prefill_moe(grouped_env.as_deref())?;
        let head_env = match std::env::var("MEMRA_DSV4_PREFILL_HEAD") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => return Err(format!("MEMRA_DSV4_PREFILL_HEAD: {err}")),
        };
        let prefill_head = Dsv4PrefillHead::resolve(head_env.as_deref())?;
        let draft_env = match std::env::var("MEMRA_DSV4_PREFILL_DRAFT") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => return Err(format!("MEMRA_DSV4_PREFILL_DRAFT: {err}")),
        };
        let prefill_draft = Dsv4PrefillDraft::resolve(draft_env.as_deref())?;
        let proposal_env = match std::env::var("MEMRA_DSV4_DSPARK_PROPOSAL") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => return Err(format!("MEMRA_DSV4_DSPARK_PROPOSAL: {err}")),
        };
        let draft_proposal = Dsv4DraftProposal::resolve(proposal_env.as_deref())?;
        let indexer_env = match std::env::var("MEMRA_DSV4_INDEXER_SCORE") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => return Err(format!("MEMRA_DSV4_INDEXER_SCORE: {err}")),
        };
        let indexer_score = Dsv4IndexerScore::resolve(indexer_env.as_deref())?;
        let model = Dsv4Model::open(dir)?;
        let d = model.cfg().clone();
        let mc = model.mc.clone();
        let n_trunk = mc.n_layer - mc.nextn_predict_layers;
        let rd = d.qk_rope_head_dim as usize;

        // split point: balance per-layer resident bytes (experts uniform; fine layers
        // carry the indexer). Computed from config, not hardcoded.
        let layer_bytes = |il: u32| -> u64 {
            let ratio = d.compress_ratio(il);
            let base = 3_875_000_000u64; // experts slab + attn bf16 (measured class math)
            match ratio {
                4 => base + 66_000_000,
                _ => base,
            }
        };
        let trunk_bytes: Vec<u64> = (0..n_trunk).map(layer_bytes).collect();
        // The bundled 0731 DSpark is resident on the tail stage. The old trunk-only
        // cut balanced 22/20 layers and then added all three blocks to card 1, creating
        // the measured ~7.3 GiB load skew that blocked an otherwise-fitting 1M cache.
        // Reserve its config-derived block count before choosing the cut. One base-layer
        // charge per block is conservative for the MXFP4 blocks and stable across mints;
        // main_proj/markov auxiliaries fit inside that overestimate.
        let dspark_tail_reserve = if std::env::var("MEMRA_DSV4_DRAFTER").as_deref() == Ok("dspark")
            && !model.has("mtp.0.e_proj.weight")
        {
            let cfg = memra_gguf::dsv4_dspark::DsparkConfig::load(dir, &model);
            cfg.n_blocks as u64 * 3_875_000_000u64
        } else {
            0
        };
        let split_at = dsv4_split_for_tail_reserve(&trunk_bytes, dspark_tail_reserve) as u32;
        eprintln!(
            "[load] placement: split_at={split_at} trunk_layers={n_trunk} dspark_tail_reserve={:.2} GiB",
            dspark_tail_reserve as f64 / (1u64 << 30) as f64,
        );

        let fc_yarn_host = precompute_freqs_cis(
            rd,
            max_seq,
            d.rope_yarn_orig_ctx,
            d.compress_rope_theta,
            d.rope_yarn_factor,
            d.rope_yarn_beta_fast,
            d.rope_yarn_beta_slow,
        );
        let fc_plain_host = precompute_freqs_cis(
            rd,
            max_seq,
            0,
            mc.rope_freq_base,
            d.rope_yarn_factor,
            d.rope_yarn_beta_fast,
            d.rope_yarn_beta_slow,
        );
        let flat =
            |fc: &FreqsCis| -> Vec<f32> { fc.cs.iter().flat_map(|&(c, s)| [c, s]).collect() };

        let inter = mc.moe.as_ref().expect("moe").expert_ff_length as usize;
        let hidden = mc.n_embd as usize;
        let mut stages = Vec::new();
        for &dev in devices {
            let gpu = memra_runtime::Gpu::new(dev).map_err(e("Gpu::new"))?;
            // Engine::new idiom (lib.rs:1172): single stream per stage, explicit syncs at
            // the boundary — cudarc per-arg event tracking off.
            unsafe { gpu.ctx.disable_event_tracking() };
            let stream = gpu.stream();
            let fc_yarn = upload_f32(&stream, &flat(&fc_yarn_host))?;
            let fc_plain = upload_f32(&stream, &flat(&fc_plain_host))?;
            let ws = stream.alloc_zeros::<u8>(64 << 20).map_err(e("ws alloc"))?;
            let deq = [
                stream
                    .alloc_zeros::<u8>(inter * hidden * 2)
                    .map_err(e("deq"))?,
                stream
                    .alloc_zeros::<u8>(inter * hidden * 2)
                    .map_err(e("deq"))?,
                stream
                    .alloc_zeros::<u8>(inter * hidden * 2)
                    .map_err(e("deq"))?,
            ];
            stages.push(Stage {
                dev,
                gpu,
                layers: Vec::new(),
                embed: None,
                head: None,
                trunk_norm: None,
                hc_head_fn: None,
                fc_yarn,
                fc_plain,
                ws,
                deq,
                loaded_bytes: 0,
                hc_head_base_dev: None,
                hc_head_scale_dev: None,
            });
        }

        // lane 8: decode-path seam (read once, printed; one binary carries both arms)
        let decode_path = match std::env::var("MEMRA_DSV4_DECODE_PATH").as_deref() {
            Err(_) | Ok("") | Ok("legacy") => DecodePath::Legacy,
            Ok("device-hostmath") => DecodePath::Device { host_math: true },
            Ok("device") => DecodePath::Device { host_math: false },
            Ok(other) => {
                return Err(format!(
                    "MEMRA_DSV4_DECODE_PATH '{other}' unknown (legacy | device | device-hostmath)"
                ));
            }
        };
        // lane 9: island-dots arm seam (owner-gated fork; f64 = the oracle-truth arm).
        // 0731 re-gate extension rung: `f32x` = the f32 dots arm PLUS f32-accumulation
        // twins for the remaining device-path f64 chains (owner-authorized fork).
        // OWNER RATIFICATION 2026-08-19: f32x is the DEFAULT device-decode dots arm
        // (quality-stays condition met at the owner bar — 0731 re-gate Task B gates:
        // decode 52/52, CPU teacher-forcing 257/260 all-in-band, tf-gate 158/160,
        // determinism ×2). f64 stays the selectable oracle-truth arm; hc_sinkhorn is
        // NOT part of f32x (never authorized). Legacy path and prefill are untouched.
        // The unset default is DEVICE-decode-scoped by the ratification's own words:
        // the legacy path never consults the flag, so on Legacy an UNSET env resolves
        // to the f64 oracle bytes rather than tripping the f32-requires-device refusal
        // (box4 find, 2026-08-20: dsv4-gpu-gate under the flipped default panicked at
        // load on the legacy path — the refusal is for EXPLICIT f32/f32x only).
        // Illegal combos are BOOT REFUSALS (Err), never post-build aborts — hermes
        // review fingerprint a4e3d9a8eab4cf17: an assert! after Dsv4Gpu is built dies
        // as a process ABORT, which a serving watchdog restarts in a crash loop; the
        // unknown-enum arms already refuse at parse, so the combo checks live here too.
        let on_device = matches!(decode_path, DecodePath::Device { .. });
        if prefill_head == Dsv4PrefillHead::Last && !on_device {
            return Err("MEMRA_DSV4_PREFILL_HEAD=last requires device decode".into());
        }
        if prefill_draft == Dsv4PrefillDraft::Tail && !on_device {
            return Err("MEMRA_DSV4_PREFILL_DRAFT=tail requires device decode".into());
        }
        if draft_proposal == Dsv4DraftProposal::Coupled && !on_device {
            return Err("MEMRA_DSV4_DSPARK_PROPOSAL=coupled requires device decode".into());
        }
        let (dots_f32, chains_f32) = match std::env::var("MEMRA_DSV4_DOTS_ARM").as_deref() {
            Err(_) | Ok("") => {
                // ratified default, DEVICE-decode-scoped: legacy resolves f64.
                if on_device {
                    (true, true)
                } else {
                    (false, false)
                }
            }
            Ok(explicit @ ("f32x" | "f32")) if !on_device => {
                return Err(format!(
                    "MEMRA_DSV4_DOTS_ARM={explicit} requires MEMRA_DSV4_DECODE_PATH=device \
                     (the f32 dots arms exist on the device decode path only)"
                ));
            }
            Ok("f32x") => (true, true),
            Ok("f64") => (false, false),
            Ok("f32") => (true, false),
            Ok(other) => {
                return Err(format!(
                    "MEMRA_DSV4_DOTS_ARM '{other}' unknown (f64 | f32 | f32x)"
                ));
            }
        };

        if indexer_score == Dsv4IndexerScore::Tiled
            && (!matches!(decode_path, DecodePath::Device { host_math: false })
                || !chains_f32
                || d.index_n_heads != 64
                || d.index_head_dim != 128)
        {
            return Err(
                "MEMRA_DSV4_INDEXER_SCORE=tiled requires device f32x and indexer 64x128".into(),
            );
        }
        if prefill_grouped
            && (!matches!(decode_path, DecodePath::Device { host_math: false })
                || crate::moe_f16g_mode() < 2
                || crate::moe_f16g_sk_params().0 < 0
                || !crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT))
        {
            return Err("MEMRA_DSV4_PREFILL_MOE=grouped requires device decode, MEMRA_MOE_F16G=2, visitor form and direct quant loader".into());
        }

        // Iteration-3 rung 4c, MEASURED FORK (nsys, drafted rounds [4,12)): the DRAFTER's
        // shared-trunk-head projection runs `dsv4_dots_f32` — the f64 kernel — over
        // block_size rows, and it measured **16.3 ms of a 78 ms drafted round (21%)**, one
        // instance at 13-14.7 ms. The trunk's OWN head already runs the ratified f32x arm
        // on the SAME weights; the drafter's copy only picks DRAFTS (verification always
        // emits the trunk's argmax, so output identity cannot depend on it). This arm makes
        // the drafter's exit head follow the ratified class. Default is f64 — today's gated
        // bytes, untouched — because the lane-10 components gate ran the drafter at f64 and
        // a gated component does not change default without its gate; `f32x` is the
        // measured arm offered for owner ratification with the acceptance delta reported.
        // OWNER RATIFICATION 2026-08-19 (relayed to the box4 lane 2026-08-20): f32x is
        // the DEFAULT drafter exit-head arm — the fork was measured quality-INERT
        // (acceptance digest byte-identical across arms on the gate fixture AND 3,321
        // corpora rounds, iteration-3 rung 4c) and it only picks DRAFTS (the greedy
        // identity law keeps the emitted stream the trunk's own argmax either way).
        // f64 stays selectable as the lane-10 oracle-truth arm. hc_sinkhorn remains f64
        // in every arm — never authorized.
        let dspark_head_f32 = match std::env::var("MEMRA_DSV4_DSPARK_HEAD_ARM").as_deref() {
            Err(_) | Ok("") | Ok("f32x") => true,
            Ok("f64") => false,
            Ok(other) => {
                return Err(format!(
                    "MEMRA_DSV4_DSPARK_HEAD_ARM '{other}' unknown (f64 | f32x)"
                ));
            }
        };

        // iteration-5 FP8 dense arm seam — OWNER RATIFICATION 2026-08-20 (the ratified
        // bundle, executed in the v0.98 train once the it5 item-3 cells went green on
        // box7): **fp8 is the DEFAULT DEVICE-DECODE dense arm.** Receipts: bit-identical
        // to bf16 on four boxes / five binaries (dsgate accept shas
        // 150342bae32b38b5/85603e87fadf7876 one bit pattern, tf-gate 158/160 with the
        // banked near-ties at steps 22+134), completed interleaved x5 A/B plain
        // 41.06 -> 47.19 median (+14.9%, box5), and the item-3 staged residency turns
        // the arm's +2.7 GiB/card dual-residency cost into a saving (box7: -5.56/-5.34
        // GiB/card vs the dual-resident builds, every item-3 bit-gate green).
        // DEVICE-scoped exactly like the ratified dots default (82a754fbec): unset on
        // the LEGACY path resolves bf16 (legacy has no fp8 twins and must keep
        // booting); explicit fp8 on legacy still refuses; bf16 stays selectable
        // everywhere. Resolution is the pure `resolve_dense_arm` so the flip is
        // toothed-testable; the `[load] dense arm:` line below is the boot receipt.
        let dense_fp8 = resolve_dense_arm(
            std::env::var("MEMRA_DSV4_DENSE_ARM").ok().as_deref(),
            on_device,
        )?;
        let dspark_fused_moe = resolve_dspark_fused_moe(
            std::env::var("MEMRA_DSV4_DSPARK_FUSED_MOE").ok().as_deref(),
            on_device,
        )?;

        let mut me = Dsv4Gpu {
            model,
            stages,
            layer_stage: (0..n_trunk).map(|il| usize::from(il >= split_at)).collect(),
            split_at,
            max_seq,
            variant,
            fc_yarn_host,
            fc_plain_host,
            mtp: None,
            expert_arm: if memra_gguf::dsv4_forward::expert_arm_native() {
                ExpertArm::Native
            } else {
                ExpertArm::Bf16Dequant
            },
            decode_path,
            dots_f32,
            chains_f32,
            dspark_head_f32,
            dspark_fused_moe,
            fp4_reduce,
            indexer_score,
            prefill_grouped,
            prefill_head,
            prefill_draft,
            prefill_head_counts: PrefillHeadCounters::default(),
            draft_proposal,
            coupled_draft_draws: std::sync::atomic::AtomicU64::new(0),
            dense_fp8,
            dspark: None,
            boundary_ev: Vec::new(),
            hc_head_base: Vec::new(),
            hc_head_scale: Vec::new(),
        };
        eprintln!(
            "[load] expert arm: {:?} | decode path: {:?} | dots arm: {}",
            me.expert_arm,
            me.decode_path,
            if me.chains_f32 {
                "f32x (dots + sink/norm/indexer chains)"
            } else if me.dots_f32 {
                "f32"
            } else {
                "f64"
            }
        );
        eprintln!("[load] selected FP4 reduction: {:?}", me.fp4_reduce);
        eprintln!("[load] indexer score: {:?}", me.indexer_score);
        eprintln!("[load] prefill head: {:?}", me.prefill_head);
        eprintln!("[load] prefill DSpark prime: {:?}", me.prefill_draft);
        eprintln!("[load] sampled DSpark proposal: {:?}", me.draft_proposal);
        eprintln!(
            "[load] prefill MoE: {}",
            if me.prefill_grouped {
                "grouped FP8-QAT half mirror (experimental)"
            } else {
                "reference"
            }
        );
        eprintln!(
            "[load] dspark exit-head dots arm: {} (rung-4c fork; drafts only, never the \
             emitted stream)",
            if me.dspark_head_f32 { "f32x" } else { "f64" }
        );
        eprintln!(
            "[load] dspark selected-expert dispatch: {}",
            if me.dspark_fused_moe {
                "FUSED (host-oracle route, device indirect projections)"
            } else {
                "per-expert reference"
            }
        );
        eprintln!(
            "[load] dense arm: {} (iteration-5; fp8 = FP8-blk linears as-stored on the \
             device decode/verify paths, bit-identical twins)",
            if me.dense_fp8 { "fp8" } else { "bf16" }
        );
        if matches!(me.decode_path, DecodePath::Device { .. }) && me.expert_arm != ExpertArm::Native
        {
            // the indirect fused dispatch is an fp4-slab program — the bf16-dequant arm
            // has no device-indirect twin. Boot refusal, not a post-build abort
            // (hermes fingerprint a4e3d9a8eab4cf17); the dots/dense combos refuse at
            // env-parse above for the same reason.
            return Err(
                "MEMRA_DSV4_DECODE_PATH=device requires MEMRA_DSV4_EXPERT_ARM=native".to_string(),
            );
        }

        // lane 8: peer transport for the PP boundary (pp.rs idiom: cuCtxEnablePeerAccess
        // both directions + default-mempool access grants — cudarc buffers are
        // stream-ordered-pool allocations, unmapped by EnablePeerAccess alone).
        if matches!(me.decode_path, DecodePath::Device { .. }) && me.stages.len() > 1 {
            use cudarc::driver::sys as cus;
            for a in 0..me.stages.len() {
                for b in 0..me.stages.len() {
                    if a == b || me.stages[a].dev == me.stages[b].dev {
                        continue;
                    }
                    me.stages[a]
                        .gpu
                        .ctx
                        .bind_to_thread()
                        .map_err(e("peer bind"))?;
                    let rc =
                        unsafe { cus::cuCtxEnablePeerAccess(me.stages[b].gpu.ctx.cu_ctx(), 0) };
                    if rc != cus::cudaError_enum::CUDA_SUCCESS
                        && rc != cus::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                    {
                        return Err(format!(
                            "cuCtxEnablePeerAccess(dev{} -> dev{}) failed: {rc:?}",
                            me.stages[a].dev, me.stages[b].dev
                        ));
                    }
                    let dev = cudarc::driver::result::device::get(me.stages[a].dev as i32)
                        .map_err(e("device get"))?;
                    let mut pool: cus::CUmemoryPool = std::ptr::null_mut();
                    unsafe {
                        cus::cuDeviceGetDefaultMemPool(&mut pool, dev)
                            .result()
                            .map_err(e("default pool"))?;
                    }
                    let desc = cus::CUmemAccessDesc {
                        location: cus::CUmemLocation {
                            type_: cus::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                            id: me.stages[b].dev as i32,
                        },
                        flags: cus::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                    };
                    let rc = unsafe { cus::cuMemPoolSetAccess(pool, &desc, 1) };
                    if rc != cus::cudaError_enum::CUDA_SUCCESS {
                        return Err(format!(
                            "cuMemPoolSetAccess(dev{} pool -> dev{}) failed: {rc:?}",
                            me.stages[a].dev, me.stages[b].dev
                        ));
                    }
                }
            }
            for bnd in 0..me.stages.len() - 1 {
                let ev = me.stages[bnd]
                    .gpu
                    .ctx
                    .new_event(None)
                    .map_err(e("boundary event"))?;
                me.boundary_ev.push(ev);
            }
            // PEER BYTE-INTEGRITY PROBE (lane/hermes-perf-fixes, 2026-08-23): enable +
            // pool grants alone prove ADDRESSABILITY, not integrity — see the probe
            // helpers' header for the Pod B receipt. Ladder up to the prefill
            // hidden-state payload class; both directions per cross-device boundary;
            // FAIL-CLOSED at load (the device PP path has no host-bounce twin).
            {
                let hidden = me.model.mc.n_embd as usize;
                let hc = me.model.cfg().hc_mult as usize;
                // Include the exact persistent decode payload (hc*hidden f32) and the
                // maximum served verify-width payload, not only nearby powers of two. Peer
                // corruption on the affected drivers is size-class-sensitive (Hermes
                // `58843bb6b924125b`).
                let ladder = dsv4_peer_probe_ladder(hidden, hc);
                let probe_t0 = std::time::Instant::now();
                let mut copies = 0usize;
                for bnd in 0..me.stages.len() - 1 {
                    if me.stages[bnd].dev == me.stages[bnd + 1].dev {
                        continue;
                    }
                    for (s, d) in [(bnd, bnd + 1), (bnd + 1, bnd)] {
                        for &bytes in &ladder {
                            dsv4_peer_probe_copy(&me.stages[s], &me.stages[d], bnd, bytes)
                                .map_err(|err| {
                                    format!(
                                        "dsv4 PP peer byte-integrity probe FAILED: \
                                         boundary={bnd} dev{}->dev{} bytes={bytes}: {err}; \
                                         refusing the device PP path (silent hidden-state \
                                         corruption class — fix the P2P fabric or serve a \
                                         non-device MEMRA_DSV4_DECODE_PATH)",
                                        me.stages[s].dev, me.stages[d].dev,
                                    )
                                })?;
                            copies += 1;
                        }
                    }
                }
                eprintln!(
                    "[load] lane-8 peer byte-integrity probe PASS: {} boundaries, \
                     {copies} copies, ladder {ladder:?} bytes, {:.1}ms",
                    me.boundary_ev.len(),
                    probe_t0.elapsed().as_secs_f64() * 1e3,
                );
            }
            eprintln!(
                "[load] lane-8 peer transport enabled ({} boundaries)",
                me.boundary_ev.len()
            );
        }

        // stage 0: embed; last stage: head + trunk hc_head/norm
        me.stages[0].embed = Some({
            let (_, raw) = me.model.st.raw("embed.weight").expect("embed.weight");
            let stream = me.stages[0].gpu.stream();
            me.stages[0].loaded_bytes += raw.len() as u64;
            upload_u8(&stream, raw)?
        });
        let last = me.stages.len() - 1;
        me.stages[last].head = Some({
            let (_, raw) = me.model.st.raw("head.weight").expect("head.weight");
            let stream = me.stages[last].gpu.stream();
            me.stages[last].loaded_bytes += raw.len() as u64;
            upload_u8(&stream, raw)?
        });
        me.stages[last].trunk_norm = Some(me.tensor_f32_dev(last, "norm.weight")?);
        me.stages[last].hc_head_fn = Some(me.tensor_f32_dev(last, "hc_head_fn")?);
        me.hc_head_base = me.model.tensor_f32("hc_head_base").1;
        me.hc_head_scale = me.model.tensor_f32("hc_head_scale").1;
        {
            let stream = me.stages[last].gpu.stream();
            let base_dev = upload_f32(&stream, &me.hc_head_base)?;
            let scale_dev = upload_f32(&stream, &me.hc_head_scale)?;
            me.stages[last].hc_head_base_dev = Some(base_dev);
            me.stages[last].hc_head_scale_dev = Some(scale_dev);
        }

        let t0 = std::time::Instant::now();
        for il in 0..n_trunk {
            let stage = me.layer_stage[il as usize];
            let l = me.load_layer(stage, il, &format!("layers.{il}"))?;
            me.stages[stage].layers.push(l);
            if il % 4 == 3 || il + 1 == n_trunk {
                eprintln!(
                    "[load] layer {il} -> dev{} done t={:.0}s",
                    me.stages[stage].dev,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // MTP (NextN) block on the last stage — optional path taken because the trunk
        // landed with box time to spare (lane brief); layer id = n_trunk from config.
        // 0731 lineage: the `mtp.*` namespace is the DSPARK drafter (3 window-only
        // blocks; census per the mint receipts: mtp.0 main_proj/main_norm, mtp.2
        // markov_w1/w2 + confidence_head — no e_proj/enorm), NOT a NextN head. Its GPU
        // path is a separate lane; the trunk forward never consumes it. Discriminate on
        // the artifact's own stored structure (lane-1 law: stored tensor names are the
        // recipe truth): a NextN block carries `mtp.0.e_proj.weight` (RAW safetensors
        // name — measured on both artifacts: preview has e_proj.weight+.scale, 0731 has
        // no e_proj keys; the stem alone misses because `has` is raw-exact).
        let nextn = me.model.mc.nextn_predict_layers;
        if nextn > 0 && me.model.has("mtp.0.e_proj.weight") {
            assert_eq!(
                nextn, 1,
                "multi-NextN chains not wired (single MTP layer expected)"
            );
            let p = "mtp.0";
            let layer = me.load_layer(last, n_trunk, p)?;
            assert_eq!(
                layer.expert_kind,
                ExpertKind::Mxfp4,
                "MTP experts must be MXFP4"
            );
            let mtp = MtpDev {
                layer,
                enorm: me.tensor_f32_dev(last, &format!("{p}.enorm.weight"))?,
                hnorm: me.tensor_f32_dev(last, &format!("{p}.hnorm.weight"))?,
                norm: me.tensor_f32_dev(last, &format!("{p}.norm.weight"))?,
                e_proj: me.tensor_bf16(last, &format!("{p}.e_proj"))?,
                h_proj: me.tensor_bf16(last, &format!("{p}.h_proj"))?,
                hc_head_fn: me.tensor_f32_dev(last, &format!("{p}.hc_head_fn"))?,
                hc_head_base: me.model.tensor_f32(&format!("{p}.hc_head_base")).1,
                hc_head_scale: me.model.tensor_f32(&format!("{p}.hc_head_scale")).1,
            };
            me.mtp = Some(mtp);
        } else if nextn > 0 {
            if std::env::var("MEMRA_DSV4_DRAFTER").as_deref() == Ok("dspark") {
                // iteration 3: the DSpark drafter, whole module on the LAST stage
                // (tap layers 40/41/42 + shared head locality — VRAM plan in the
                // iteration-3 receipts). Census pins + NextN refusal ride the CPU
                // oracle's own config loader (one refusal program, two realizations).
                let cfg = memra_gguf::dsv4_dspark::DsparkConfig::load(dir, &me.model);
                let hidden = me.model.mc.n_embd as usize;
                let mut blocks = Vec::with_capacity(cfg.n_blocks);
                for k in 0..cfg.n_blocks {
                    let layer = me.load_layer(last, n_trunk + k as u32, &format!("mtp.{k}"))?;
                    assert_eq!(layer.ratio, 0, "dspark block mtp.{k} must be ratio 0");
                    assert_eq!(
                        layer.expert_kind,
                        ExpertKind::Mxfp4,
                        "dspark experts must be MXFP4"
                    );
                    blocks.push(layer);
                }
                let last_p = format!("mtp.{}", cfg.n_blocks - 1);
                let (mp_shape, _) = me.model.tensor_f32("mtp.0.main_proj");
                assert_eq!(
                    mp_shape,
                    vec![hidden, cfg.target_layer_ids.len() * hidden],
                    "main_proj shape"
                );
                let (w1_shape, w1) = me
                    .model
                    .tensor_f32(&format!("{last_p}.markov_head.markov_w1.weight"));
                let (w2_shape, w2) = me
                    .model
                    .tensor_f32(&format!("{last_p}.markov_head.markov_w2.weight"));
                let vocab = w1_shape[0];
                assert_eq!(w1_shape[1], cfg.markov_rank, "markov_w1 rank");
                assert_eq!(w2_shape, vec![vocab, cfg.markov_rank], "markov_w2 shape");
                let (cf_shape, _) = me
                    .model
                    .tensor_f32(&format!("{last_p}.confidence_head.proj.weight"));
                assert_eq!(
                    cf_shape,
                    vec![1, hidden + cfg.markov_rank],
                    "confidence proj shape"
                );
                let st_stream = me.stages[last].gpu.stream();
                let markov_w1 = upload_f32(&st_stream, &w1)?;
                let markov_w2 = upload_f32(&st_stream, &w2)?;
                let dspark = DsparkDev {
                    blocks,
                    main_proj: me.tensor_bf16(last, "mtp.0.main_proj")?,
                    main_norm: me.tensor_f32_dev(last, "mtp.0.main_norm.weight")?,
                    norm: me.tensor_f32_dev(last, &format!("{last_p}.norm.weight"))?,
                    markov_w1,
                    markov_w2,
                    markov_w1_host: w1,
                    conf_w: me
                        .tensor_f32_dev(last, &format!("{last_p}.confidence_head.proj.weight"))?,
                    hc_head_fn: me.tensor_f32_dev(last, &format!("{last_p}.hc_head_fn"))?,
                    hc_head_base: me.model.tensor_f32(&format!("{last_p}.hc_head_base")).1,
                    hc_head_scale: me.model.tensor_f32(&format!("{last_p}.hc_head_scale")).1,
                    block_size: cfg.block_size,
                    noise_token: cfg.noise_token_id,
                    targets: cfg.target_layer_ids.clone(),
                    rank: cfg.markov_rank,
                    vocab,
                };
                eprintln!(
                    "[load] drafter: DSpark ({} blocks, block_size {}, targets {:?}) \
                     resident on stage {last}",
                    cfg.n_blocks, cfg.block_size, cfg.target_layer_ids
                );
                me.dspark = Some(dspark);
            } else {
                eprintln!(
                    "[load] drafter: {nextn} DSpark block(s) (mtp.0.e_proj absent) — GPU \
                     drafter path off (set MEMRA_DSV4_DRAFTER=dspark); trunk-only"
                );
            }
        }
        for st in &me.stages {
            st.gpu.stream().synchronize().map_err(e("load sync"))?;
        }
        Ok(me)
    }

    /// (free, total, resident-by-loader) bytes per device — the placement table source.
    pub fn vram_report(&self) -> Res<Vec<(usize, u64, u64, u64)>> {
        let mut out = Vec::new();
        for st in &self.stages {
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx"))?;
            let (free, total) = st.gpu.ctx.mem_get_info().map_err(e("mem_get_info"))?;
            out.push((st.dev, free as u64, total as u64, st.loaded_bytes));
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- forward pieces

    /// bf16 GEMM y[mxn] f32 = x[mxk] (f32, cast here) @ w[nxk]ᵀ (bf16 resident).
    /// `w_off_elems` slices the weight (grouped wo_a).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        st: &Stage,
        x_f32: &CudaSlice<f32>,
        w_bf16: &CudaSlice<u8>,
        w_off_elems: usize,
        m: usize,
        n: usize,
        kdim: usize,
        y: &mut CudaSlice<f32>,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        let mut xb = stream
            .alloc_zeros::<u8>(m * kdim * 2)
            .map_err(e("alloc xb"))?;
        unsafe {
            ck(
                "cvt_bf16",
                k::memra_dsv4_cvt_bf16(
                    dpf!(x_f32, &stream),
                    xb.device_ptr_mut(&stream).0 as *mut c_void,
                    (m * kdim) as i64,
                    sp(&stream),
                ),
            )?;
            ck(
                "gemm_bf16",
                k::memra_dsv4_gemm_bf16(
                    (w_bf16.device_ptr(&stream).0 as usize + w_off_elems * 2) as *const c_void,
                    dp!(xb, &stream),
                    dpm!(y, &stream),
                    m as i32,
                    n as i32,
                    kdim as i32,
                    st.dev as i32,
                    st.ws.device_ptr(&stream).0 as *mut c_void,
                    st.ws.len(),
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// bf16 GEMM from an ALREADY-bf16 activation buffer.
    #[allow(clippy::too_many_arguments)]
    fn gemm_pre(
        st: &Stage,
        xb: &CudaSlice<u8>,
        w_bf16_ptr: *const c_void,
        m: usize,
        n: usize,
        kdim: usize,
        y: &mut CudaSlice<f32>,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        unsafe {
            ck(
                "gemm_bf16",
                k::memra_dsv4_gemm_bf16(
                    w_bf16_ptr,
                    dp!(xb, &stream),
                    dpm!(y, &stream),
                    m as i32,
                    n as i32,
                    kdim as i32,
                    st.dev as i32,
                    st.ws.device_ptr(&stream).0 as *mut c_void,
                    st.ws.len(),
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// f32-island GEMM (f64-accumulated dots kernel).
    fn dots(
        st: &Stage,
        x: &CudaSlice<f32>,
        w_f32: &CudaSlice<f32>,
        s: usize,
        kdim: usize,
        n: usize,
        y: &mut CudaSlice<f32>,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        unsafe {
            ck(
                "dots_f32",
                k::memra_dsv4_dots_f32(
                    dpf!(x, &stream),
                    dp!(w_f32, &stream),
                    0,
                    dpm!(y, &stream),
                    s as i32,
                    kdim as i32,
                    n as i32,
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// Island dots on the DEVICE decode path (lane 9): routes to the f64 oracle-truth
    /// arm (default — byte-identical to `Self::dots`) or the owner-gated
    /// f32-accumulation serving arm (MEMRA_DSV4_DOTS_ARM=f32; fork gated by
    /// decode-gate + oracle teacher-forcing, RECEIPTS.md "Lane 9").
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn dots_dev(
        &self,
        st: &Stage,
        x: &CudaSlice<f32>,
        w_f32: &CudaSlice<f32>,
        s: usize,
        kdim: usize,
        n: usize,
        y: &mut CudaSlice<f32>,
    ) -> Res<()> {
        if !self.dots_f32 {
            return Self::dots(st, x, w_f32, s, kdim, n, y);
        }
        let stream = st.gpu.stream();
        unsafe {
            ck(
                "dots_f32acc",
                k::memra_dsv4_dots_f32acc(
                    dpf!(x, &stream),
                    dp!(w_f32, &stream),
                    0,
                    dpm!(y, &stream),
                    s as i32,
                    kdim as i32,
                    n as i32,
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// Compressor forward (f32 island end-to-end). Returns (Some((ckv [nb, d], nb)) or
    /// None when no complete block, kv_raw [s, latent], score_raw [s, latent]).
    /// The raw GEMM outputs are ALWAYS computed (the reference does too, M:330-331) —
    /// lane 6 seeds the decode pending state from their trailing rows.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn compressor(
        &self,
        st: &Stage,
        cmp: &CmpDev,
        x: &CudaSlice<f32>, // [s, hidden] post-attn-norm
        s: usize,
        hidden: usize,
        fc_dev: &CudaSlice<f32>,
        rd: usize,
        eps: f32,
    ) -> Res<(
        Option<(CudaSlice<f32>, usize)>,
        CudaSlice<f32>,
        CudaSlice<f32>,
    )> {
        let stream = st.gpu.stream();
        let mut kv = stream
            .alloc_zeros::<f32>(s * cmp.latent)
            .map_err(e("cmp kv"))?;
        let mut score = stream
            .alloc_zeros::<f32>(s * cmp.latent)
            .map_err(e("cmp score"))?;
        Self::dots(st, x, &cmp.wkv, s, hidden, cmp.latent, &mut kv)?;
        Self::dots(st, x, &cmp.wgate, s, hidden, cmp.latent, &mut score)?;
        if s < cmp.ratio {
            return Ok((None, kv, score));
        }
        let cutoff = s - s % cmp.ratio;
        let nb = cutoff / cmp.ratio;
        let mut pooled = stream
            .alloc_zeros::<f32>(nb * cmp.d)
            .map_err(e("cmp out"))?;
        unsafe {
            ck(
                "compressor_pool",
                k::memra_dsv4_compressor_pool(
                    dpf!(kv, &stream),
                    dpf!(score, &stream),
                    dpf!(cmp.ape, &stream),
                    dpm!(pooled, &stream),
                    nb as i32,
                    cmp.ratio as i32,
                    cmp.d as i32,
                    cmp.latent as i32,
                    cmp.overlap as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "rmsnorm cmp",
                k::memra_dsv4_rmsnorm(
                    dpf!(pooled, &stream),
                    dpf!(cmp.norm, &stream),
                    dpm!(pooled, &stream),
                    nb as i32,
                    cmp.d as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            let positions: Vec<i32> = (0..nb).map(|j| (j * cmp.ratio) as i32).collect();
            let pos_dev = upload_i32(&stream, &positions)?;
            ck(
                "rope cmp",
                k::memra_dsv4_rope(
                    dpm!(pooled, &stream),
                    nb as i32,
                    1,
                    cmp.d as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
            if cmp.rotate {
                // oracle hadamard scale: (d as f32).powf(-0.5)
                let scale = (cmp.d as f32).powf(-0.5);
                ck(
                    "hadamard cmp",
                    k::memra_dsv4_hadamard(
                        dpm!(pooled, &stream),
                        nb as i32,
                        cmp.d as i32,
                        scale,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "fp4 cmp",
                    k::memra_dsv4_fp4_act_quant(
                        dpm!(pooled, &stream),
                        nb as i32,
                        cmp.d as i64,
                        cmp.d as i32,
                        sp(&stream),
                    ),
                )?;
            } else {
                ck(
                    "act_quant cmp",
                    k::memra_dsv4_act_quant(
                        dpm!(pooled, &stream),
                        nb as i32,
                        cmp.d as i64,
                        (cmp.d - rd) as i32,
                        64,
                        (self.variant == ActQuantVariant::ClampOnly) as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
        Ok((Some((pooled, nb)), kv, score))
    }

    /// Prefill→decode handoff for one compressor: copy the pooled blocks into the
    /// store rows [row0, row0+nb) and seed the pending state from the raw kv/score
    /// trailing rows (fine: last COMPLETE block → prev slots + remainder → cur slots,
    /// M:346-352; coarse: remainder → slots [0, rem)).
    #[allow(clippy::too_many_arguments)]
    fn populate_cmp_cache(
        stream: &std::sync::Arc<CudaStream>,
        s: usize,
        cmp_ratio: usize,
        latent: usize,
        d: usize,
        pooled: &Option<(CudaSlice<f32>, usize)>,
        kv_raw: &CudaSlice<f32>,
        score_raw: &CudaSlice<f32>,
        store: &mut CudaSlice<f32>,
        row0: usize,
        blocks: &mut usize,
        pend_kv: &mut CudaSlice<f32>,
        pend_score: &mut CudaSlice<f32>,
        overlap: bool,
    ) -> Res<()> {
        *blocks = 0;
        if let Some((buf, nb)) = pooled {
            let src = buf.slice(0..nb * d);
            let mut dst = store.slice_mut(row0 * d..(row0 + nb) * d);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("cmp store"))?;
            *blocks = *nb;
        }
        let cutoff = s - s % cmp_ratio;
        let rem = s - cutoff;
        if overlap {
            if cutoff >= cmp_ratio {
                let a = (cutoff - cmp_ratio) * latent;
                let b = cutoff * latent;
                let src = kv_raw.slice(a..b);
                let mut dst = pend_kv.slice_mut(0..cmp_ratio * latent);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("pend kv prev"))?;
                let src = score_raw.slice(a..b);
                let mut dst = pend_score.slice_mut(0..cmp_ratio * latent);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("pend sc prev"))?;
            }
            if rem > 0 {
                let a = cutoff * latent;
                let src = kv_raw.slice(a..s * latent);
                let mut dst = pend_kv.slice_mut(cmp_ratio * latent..(cmp_ratio + rem) * latent);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("pend kv cur"))?;
                let src = score_raw.slice(a..s * latent);
                let mut dst = pend_score.slice_mut(cmp_ratio * latent..(cmp_ratio + rem) * latent);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("pend sc cur"))?;
            }
        } else if rem > 0 {
            let a = cutoff * latent;
            let src = kv_raw.slice(a..s * latent);
            let mut dst = pend_kv.slice_mut(0..rem * latent);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("pend kv"))?;
            let src = score_raw.slice(a..s * latent);
            let mut dst = pend_score.slice_mut(0..rem * latent);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("pend sc"))?;
        }
        Ok(())
    }

    /// hc_pre: mixes GEMM (f32 island) + rowsq scale on GPU, Sinkhorn on HOST via the
    /// oracle's own hc_split_sinkhorn. Returns (y [s,hidden] dev, post dev, comb dev).
    #[allow(clippy::too_many_arguments)]
    fn hc_pre(
        st: &Stage,
        h: &CudaSlice<f32>, // [s, hc, hidden]
        fn_w: &CudaSlice<f32>,
        base: &[f32],
        scale: &[f32],
        s: usize,
        hc: usize,
        hidden: usize,
        iters: u32,
        hc_eps: f32,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
        let stream = st.gpu.stream();
        let w = hc * hidden;
        let rows = (2 + hc) * hc;
        let mut mixes = stream.alloc_zeros::<f32>(s * rows).map_err(e("mixes"))?;
        Self::dots(st, h, fn_w, s, w, rows, &mut mixes)?;
        unsafe {
            ck(
                "rowsq_scale",
                k::memra_dsv4_rowsq_scale(
                    dpf!(h, &stream),
                    dpm!(mixes, &stream),
                    s as i32,
                    w as i32,
                    rows as i32,
                    hc_eps,
                    sp(&stream),
                ),
            )?;
        }
        let mixes_h = dtoh_f32(&stream, &mixes)?;
        let (pre, post, comb) = hc_split_sinkhorn(&mixes_h, s, hc, scale, base, iters, hc_eps);
        let pre_d = upload_f32(&stream, &pre)?;
        let post_d = upload_f32(&stream, &post)?;
        let comb_d = upload_f32(&stream, &comb)?;
        let mut y = stream.alloc_zeros::<f32>(s * hidden).map_err(e("hc y"))?;
        unsafe {
            ck(
                "hc_collapse",
                k::memra_dsv4_hc_collapse(
                    dpf!(h, &stream),
                    dpf!(pre_d, &stream),
                    dpm!(y, &stream),
                    s as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        Ok((y, post_d, comb_d))
    }

    /// Host routing — the oracle MoeW::forward selection/weight math verbatim.
    #[allow(clippy::too_many_arguments)]
    fn route_host(
        layer: &LayerDev,
        raw_scores: &[f32], // [s, ne] gate GEMM output (pre-softplus)
        ids: &[u32],
        s: usize,
        ne: usize,
        topk: usize,
        route_scale: f32,
    ) -> (Vec<usize>, Vec<f32>) {
        let mut scores = raw_scores.to_vec();
        for v in &mut scores {
            *v = softplus_f32(*v).sqrt();
        }
        let mut indices = vec![0usize; s * topk];
        if let Some(tid2eid) = &layer.tid2eid {
            for t in 0..s {
                let row = &tid2eid[ids[t] as usize * topk..(ids[t] as usize + 1) * topk];
                let mut seen = std::collections::BTreeSet::new();
                for (kk, &ex) in row.iter().enumerate() {
                    assert!(
                        (0..ne as i64).contains(&ex),
                        "layer {}: tid2eid out of range",
                        layer.il
                    );
                    assert!(
                        seen.insert(ex),
                        "layer {}: duplicate expert id in tid2eid row {}",
                        layer.il,
                        ids[t]
                    );
                    indices[t * topk + kk] = ex as usize;
                }
            }
        } else {
            let bias = layer.gate_bias.as_ref().expect("score layer needs bias");
            for t in 0..s {
                let biased: Vec<f32> = (0..ne).map(|ex| scores[t * ne + ex] + bias[ex]).collect();
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|&a, &b| {
                    biased[b]
                        .partial_cmp(&biased[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                for kk in 0..topk {
                    indices[t * topk + kk] = order[kk];
                }
            }
        }
        let mut weights = vec![0f32; s * topk];
        for t in 0..s {
            let mut sum = 0f32;
            for kk in 0..topk {
                let w = scores[t * ne + indices[t * topk + kk]];
                weights[t * topk + kk] = w;
                sum += w;
            }
            for kk in 0..topk {
                weights[t * topk + kk] = weights[t * topk + kk] / sum * route_scale;
            }
        }
        (indices, weights)
    }

    /// One trunk block on its stage. h is [s, hc, hidden] f32 on the stage device.
    /// `cache` (lane 6): populate this layer's decode cache while prefilling.
    #[allow(clippy::too_many_arguments)]
    fn block_forward(
        &self,
        st: &Stage,
        layer: &LayerDev,
        h: &CudaSlice<f32>,
        s: usize,
        ids: &[u32],
        mut capture: Option<&mut GpuCapture>,
        mut cache: Option<&mut LayerCache>,
    ) -> Res<CudaSlice<f32>> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let heads = mc.n_head as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let eps = mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;
        // Runtime-API kernel launches in the FFI TU need this stage's context current on
        // the calling thread (cudarc binds it inside its own ops, but the previous op may
        // have been another stage's).
        st.gpu.ctx.bind_to_thread().map_err(e("bind ctx"))?;
        let stream = st.gpu.stream();
        let fc_dev = if layer.ratio != 0 {
            &st.fc_yarn
        } else {
            &st.fc_plain
        };
        let clamp_only = (self.variant == ActQuantVariant::ClampOnly) as i32;

        // ---- attention sub-block
        let (y, post, comb) = Self::hc_pre(
            st,
            h,
            &layer.hc_attn_fn,
            &layer.hc_attn_base,
            &layer.hc_attn_scale,
            s,
            hc,
            hidden,
            iters,
            hc_eps,
        )?;
        let mut x = stream.alloc_zeros::<f32>(s * hidden).map_err(e("x"))?;
        unsafe {
            ck(
                "rmsnorm attn",
                k::memra_dsv4_rmsnorm(
                    dpf!(y, &stream),
                    dpf!(layer.attn_norm, &stream),
                    dpm!(x, &stream),
                    s as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }

        // q path (item 3: under the fp8 dense arm the bf16 slabs are host-staged —
        // each `staged` view uploads a transient device copy freed, stream-ordered,
        // when the view drops at the end of this pass; on the bf16 arm it borrows
        // the resident slab and stages nothing)
        let wq_a_v = layer.wq_a.staged(&stream)?;
        let mut qr = stream.alloc_zeros::<f32>(s * q_lora).map_err(e("qr"))?;
        Self::gemm(st, &x, wq_a_v.slab(), 0, s, q_lora, hidden, &mut qr)?;
        unsafe {
            ck(
                "rmsnorm q",
                k::memra_dsv4_rmsnorm(
                    dpf!(qr, &stream),
                    dpf!(layer.q_norm, &stream),
                    dpm!(qr, &stream),
                    s as i32,
                    q_lora as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        // qr as bf16 once (feeds wq_b and the indexer wq_b, oracle reuses qr the same way)
        let mut qr_b = stream
            .alloc_zeros::<u8>(s * q_lora * 2)
            .map_err(e("qr_b"))?;
        unsafe {
            ck(
                "cvt qr",
                k::memra_dsv4_cvt_bf16(
                    dpf!(qr, &stream),
                    qr_b.device_ptr_mut(&stream).0 as *mut c_void,
                    (s * q_lora) as i64,
                    sp(&stream),
                ),
            )?;
        }
        let wq_b_v = layer.wq_b.staged(&stream)?;
        let mut q = stream.alloc_zeros::<f32>(s * heads * hd).map_err(e("q"))?;
        Self::gemm_pre(
            st,
            &qr_b,
            wq_b_v.slab().device_ptr(&stream).0 as *const c_void,
            s,
            heads * hd,
            q_lora,
            &mut q,
        )?;
        let positions: Vec<i32> = (0..s as i32).collect();
        let pos_dev = upload_i32(&stream, &positions)?;
        unsafe {
            ck(
                "headrms",
                k::memra_dsv4_headrms(
                    dpm!(q, &stream),
                    (s * heads) as i32,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope q",
                k::memra_dsv4_rope(
                    dpm!(q, &stream),
                    s as i32,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
        }

        // shared K==V latent + window QAT
        let wkv_v = layer.wkv.staged(&stream)?;
        let mut kv = stream.alloc_zeros::<f32>(s * hd).map_err(e("kv"))?;
        Self::gemm(st, &x, wkv_v.slab(), 0, s, hd, hidden, &mut kv)?;
        unsafe {
            ck(
                "rmsnorm kv",
                k::memra_dsv4_rmsnorm(
                    dpf!(kv, &stream),
                    dpf!(layer.kv_norm, &stream),
                    dpm!(kv, &stream),
                    s as i32,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope kv",
                k::memra_dsv4_rope(
                    dpm!(kv, &stream),
                    s as i32,
                    1,
                    hd as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
            ck(
                "act_quant kv",
                k::memra_dsv4_act_quant(
                    dpm!(kv, &stream),
                    s as i32,
                    hd as i64,
                    (hd - rd) as i32,
                    64,
                    clamp_only,
                    sp(&stream),
                ),
            )?;
        }
        // lane 6: window ring handoff — last min(s, win) post-QAT rows at slot p % win
        // (M:524-527: prefill leaves the cache exactly as if the ring had been written
        // position by position).
        if let Some(c) = cache.as_deref_mut() {
            for p in s.saturating_sub(win)..s {
                let slot = p % win;
                let src = kv.slice(p * hd..(p + 1) * hd);
                let mut dst = c.kvc.slice_mut(slot * hd..(slot + 1) * hd);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("ring copy"))?;
            }
        }

        // index assembly (host, oracle builders) + compressed kv
        let (widx, wslots) = window_topk_idxs(win, s);
        let mut idxs: Vec<i64> = widx;
        let mut slots = wslots;
        let mut n_kv = s;
        let mut kv_full = kv;
        let mut cap_cmp: Option<(Vec<f32>, usize)> = None;
        let mut cap_ikv: Option<(Vec<f32>, usize)> = None;
        let mut cap_isc: Option<(Vec<f32>, usize)> = None;
        let want_cap = capture
            .as_ref()
            .map(|c| c.want.contains(&layer.il))
            .unwrap_or(false);
        if layer.ratio != 0 {
            let offset = s;
            let (cidx, cslots) = if let Some(ix) = &layer.idx {
                // indexer q
                let mut qi = stream
                    .alloc_zeros::<f32>(s * ix.heads * ix.hd)
                    .map_err(e("qi"))?;
                let iwq_b_v = ix.wq_b.staged(&stream)?;
                Self::gemm_pre(
                    st,
                    &qr_b,
                    iwq_b_v.slab().device_ptr(&stream).0 as *const c_void,
                    s,
                    ix.heads * ix.hd,
                    q_lora,
                    &mut qi,
                )?;
                unsafe {
                    ck(
                        "rope qi",
                        k::memra_dsv4_rope(
                            dpm!(qi, &stream),
                            s as i32,
                            ix.heads as i32,
                            ix.hd as i32,
                            rd as i32,
                            dpf!(fc_dev, &stream),
                            pos_dev.device_ptr(&stream).0 as *const i32,
                            0,
                            sp(&stream),
                        ),
                    )?;
                    let scale = (ix.hd as f32).powf(-0.5);
                    ck(
                        "hadamard qi",
                        k::memra_dsv4_hadamard(
                            dpm!(qi, &stream),
                            (s * ix.heads) as i32,
                            ix.hd as i32,
                            scale,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "fp4 qi",
                        k::memra_dsv4_fp4_act_quant(
                            dpm!(qi, &stream),
                            (s * ix.heads) as i32,
                            ix.hd as i64,
                            ix.hd as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                // indexer compressed kv
                let (ckv_i, ikv_raw, isc_raw) =
                    self.compressor(st, &ix.cmp, &x, s, hidden, fc_dev, rd, eps)?;
                if want_cap && let Some((buf, nb)) = &ckv_i {
                    cap_ikv = Some((dtoh_f32(&stream, buf)?, *nb));
                }
                if let Some(c) = cache.as_deref_mut() {
                    let mut i_blocks = c.i_blocks;
                    Self::populate_cmp_cache(
                        &stream,
                        s,
                        ix.cmp.ratio,
                        ix.cmp.latent,
                        ix.cmp.d,
                        &ckv_i,
                        &ikv_raw,
                        &isc_raw,
                        c.ikvc.as_mut().expect("fine layer has indexer store"),
                        0,
                        &mut i_blocks,
                        c.ipend_kv.as_mut().expect("ipend"),
                        c.ipend_score.as_mut().expect("ipend"),
                        ix.cmp.overlap,
                    )?;
                    c.i_blocks = i_blocks;
                }
                // head weights (weights_proj is BF16 — lawful bf16 GEMM)
                let iwp_v = ix.weights_proj.staged(&stream)?;
                let mut wproj = stream.alloc_zeros::<f32>(s * ix.heads).map_err(e("wp"))?;
                Self::gemm(st, &x, iwp_v.slab(), 0, s, ix.heads, hidden, &mut wproj)?;
                if let Some((ckv, nb)) = &ckv_i {
                    let wscale = ((ix.hd as f64).powf(-0.5) * (ix.heads as f64).powf(-0.5)) as f32;
                    let mut score = stream.alloc_zeros::<f32>(s * nb).map_err(e("iscore"))?;
                    unsafe {
                        ck(
                            "indexer_score",
                            k::memra_dsv4_indexer_score(
                                dpf!(qi, &stream),
                                dpf!(ckv, &stream),
                                dpf!(wproj, &stream),
                                wscale,
                                dpm!(score, &stream),
                                s as i32,
                                ix.heads as i32,
                                ix.hd as i32,
                                *nb as i32,
                                layer.ratio as i32,
                                -1, // prefill law: lim = (t+1)/ratio with local t
                                sp(&stream),
                            ),
                        )?;
                    }
                    let score_h = dtoh_f32(&stream, &score)?;
                    if want_cap {
                        cap_isc = Some((score_h.clone(), *nb));
                    }
                    // host topk with the oracle's exact ordering + re-mask (model.py:508-510)
                    let kk = ix.topk.min(*nb);
                    let mut cidx = vec![-1i64; s * kk];
                    for t in 0..s {
                        let lim = (t + 1) / layer.ratio;
                        let mut order: Vec<usize> = (0..*nb).collect();
                        order.sort_by(|&a, &b| {
                            score_h[t * nb + b]
                                .partial_cmp(&score_h[t * nb + a])
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.cmp(&b))
                        });
                        for (slot, &j) in order.iter().take(kk).enumerate() {
                            cidx[t * kk + slot] = if j >= lim { -1 } else { (j + offset) as i64 };
                        }
                    }
                    (cidx, kk)
                } else {
                    (Vec::new(), 0)
                }
            } else {
                compress_topk_idxs(layer.ratio, s, offset)
            };
            if cslots > 0 {
                let mut merged = vec![-1i64; s * (slots + cslots)];
                for t in 0..s {
                    merged[t * (slots + cslots)..t * (slots + cslots) + slots]
                        .copy_from_slice(&idxs[t * slots..(t + 1) * slots]);
                    merged[t * (slots + cslots) + slots..(t + 1) * (slots + cslots)]
                        .copy_from_slice(&cidx[t * cslots..(t + 1) * cslots]);
                }
                idxs = merged;
                slots += cslots;
            }
            // attention-side compressed kv appended to the kv stream
            let acmp = layer.cmp.as_ref().expect("ratio!=0 has compressor");
            let (ckv, akv_raw, asc_raw) =
                self.compressor(st, acmp, &x, s, hidden, fc_dev, rd, eps)?;
            if want_cap && let Some((buf, nb)) = &ckv {
                cap_cmp = Some((dtoh_f32(&stream, buf)?, *nb));
            }
            if let Some(c) = cache {
                let mut n_blocks = c.n_blocks;
                Self::populate_cmp_cache(
                    &stream,
                    s,
                    acmp.ratio,
                    acmp.latent,
                    acmp.d,
                    &ckv,
                    &akv_raw,
                    &asc_raw,
                    &mut c.kvc,
                    win,
                    &mut n_blocks,
                    c.pend_kv.as_mut().expect("pend"),
                    c.pend_score.as_mut().expect("pend"),
                    acmp.overlap,
                )?;
                c.n_blocks = n_blocks;
            }
            if let Some((ckv_buf, nb)) = ckv {
                let mut merged_kv = stream
                    .alloc_zeros::<f32>((s + nb) * hd)
                    .map_err(e("kv_full"))?;
                {
                    let mut head_view = merged_kv.slice_mut(0..s * hd);
                    stream
                        .memcpy_dtod(&kv_full.slice(0..s * hd), &mut head_view)
                        .map_err(e("kv copy"))?;
                }
                {
                    let mut tail = merged_kv.slice_mut(s * hd..(s + nb) * hd);
                    stream
                        .memcpy_dtod(&ckv_buf.slice(0..nb * hd), &mut tail)
                        .map_err(e("ckv copy"))?;
                }
                kv_full = merged_kv;
                n_kv += nb;
            }
        }
        let _ = n_kv;
        let idxs_i32: Vec<i32> = idxs.iter().map(|&v| v as i32).collect();
        let idx_dev = upload_i32(&stream, &idxs_i32)?;

        // sparse sink attention + query-position de-rotation
        let mut o = stream.alloc_zeros::<f32>(s * heads * hd).map_err(e("o"))?;
        let scale = (hd as f64).powf(-0.5) as f32;
        unsafe {
            ck(
                "sink_attn",
                k::memra_dsv4_sink_attn(
                    dpf!(q, &stream),
                    dpf!(kv_full, &stream),
                    idx_dev.device_ptr(&stream).0 as *const i32,
                    dpf!(layer.sink, &stream),
                    dpm!(o, &stream),
                    s as i32,
                    heads as i32,
                    hd as i32,
                    slots as i32,
                    scale,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope o inv",
                k::memra_dsv4_rope(
                    dpm!(o, &stream),
                    s as i32,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    1,
                    sp(&stream),
                ),
            )?;
        }

        // grouped wo: per group g, og[:, g*o_lora..] = o_g @ wo_a[g]ᵀ; then wo_b.
        let gw = heads / o_groups * hd;
        let mut og = stream
            .alloc_zeros::<f32>(s * o_groups * o_lora)
            .map_err(e("og"))?;
        let mut o_grp = stream.alloc_zeros::<f32>(s * gw).map_err(e("o_grp"))?;
        let mut y_grp = stream.alloc_zeros::<f32>(s * o_lora).map_err(e("y_grp"))?;
        let wo_a_v = layer.wo_a.staged(&stream)?; // once, outside the group loop
        for g in 0..o_groups {
            unsafe {
                ck(
                    "take_cols",
                    k::memra_dsv4_take_cols(
                        dpf!(o, &stream),
                        dpm!(o_grp, &stream),
                        s as i32,
                        gw as i32,
                        (heads * hd) as i64,
                        (g * gw) as i64,
                        sp(&stream),
                    ),
                )?;
            }
            Self::gemm(
                st,
                &o_grp,
                wo_a_v.slab(),
                g * o_lora * gw,
                s,
                o_lora,
                gw,
                &mut y_grp,
            )?;
            unsafe {
                ck(
                    "place_cols",
                    k::memra_dsv4_place_cols(
                        dpf!(y_grp, &stream),
                        dpm!(og, &stream),
                        s as i32,
                        o_lora as i32,
                        (o_groups * o_lora) as i64,
                        (g * o_lora) as i64,
                        sp(&stream),
                    ),
                )?;
            }
        }
        let wo_b_v = layer.wo_b.staged(&stream)?;
        let mut attn_out = stream.alloc_zeros::<f32>(s * hidden).map_err(e("ao"))?;
        Self::gemm(
            st,
            &og,
            wo_b_v.slab(),
            0,
            s,
            hidden,
            o_groups * o_lora,
            &mut attn_out,
        )?;

        let mut cap_attn: Option<Vec<f32>> = None;
        if want_cap {
            cap_attn = Some(dtoh_f32(&stream, &attn_out)?);
        }

        // hc_post (attention)
        let mut h2 = stream
            .alloc_zeros::<f32>(s * hc * hidden)
            .map_err(e("h2"))?;
        unsafe {
            ck(
                "hc_post attn",
                k::memra_dsv4_hc_post(
                    dpf!(attn_out, &stream),
                    dpf!(h, &stream),
                    dpf!(post, &stream),
                    dpf!(comb, &stream),
                    dpm!(h2, &stream),
                    s as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }

        // ---- ffn sub-block
        let (y2, post2, comb2) = Self::hc_pre(
            st,
            &h2,
            &layer.hc_ffn_fn,
            &layer.hc_ffn_base,
            &layer.hc_ffn_scale,
            s,
            hc,
            hidden,
            iters,
            hc_eps,
        )?;
        let mut xf = stream.alloc_zeros::<f32>(s * hidden).map_err(e("xf"))?;
        unsafe {
            ck(
                "rmsnorm ffn",
                k::memra_dsv4_rmsnorm(
                    dpf!(y2, &stream),
                    dpf!(layer.ffn_norm, &stream),
                    dpm!(xf, &stream),
                    s as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        if let Some(c) = capture.as_deref_mut()
            && c.want.contains(&layer.il)
        {
            c.moe_x.insert(layer.il, dtoh_f32(&stream, &xf)?);
        }
        let moe_out = self.moe_forward(st, layer, &xf, s, ids)?;
        let mut h3 = stream
            .alloc_zeros::<f32>(s * hc * hidden)
            .map_err(e("h3"))?;
        unsafe {
            ck(
                "hc_post ffn",
                k::memra_dsv4_hc_post(
                    dpf!(moe_out, &stream),
                    dpf!(h2, &stream),
                    dpf!(post2, &stream),
                    dpf!(comb2, &stream),
                    dpm!(h3, &stream),
                    s as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }

        if let Some(c) = capture
            && c.want.contains(&layer.il)
        {
            c.layer_out.insert(layer.il, dtoh_f32(&stream, &h3)?);
            c.x_dbg.insert(layer.il, dtoh_f32(&stream, &x)?);
            c.q_dbg.insert(layer.il, dtoh_f32(&stream, &q)?);
            {
                let mut kvh = vec![0f32; s * hd];
                stream
                    .memcpy_dtoh(&kv_full.slice(0..s * hd), &mut kvh[..])
                    .map_err(e("dtoh kv_dbg"))?;
                stream.synchronize().map_err(e("sync kv_dbg"))?;
                c.kv_dbg.insert(layer.il, kvh);
            }
            c.o_dbg.insert(layer.il, dtoh_f32(&stream, &o)?);
            if let Some(a) = cap_attn {
                c.attn_out.insert(layer.il, a);
            }
            if let Some(v) = cap_cmp {
                c.compressor_kv.insert(layer.il, v);
            }
            if let Some(v) = cap_ikv {
                c.indexer_kv.insert(layer.il, v);
            }
            if let Some(v) = cap_isc {
                c.index_score.insert(layer.il, v);
            }
        }
        Ok(h3)
    }

    /// MoE on GPU: gate GEMM f32 island -> host routing (oracle math) -> per-expert
    /// on-the-fly NVFP4 dequant + bf16 GEMMs (ascending expert order, oracle
    /// accumulation order) -> shared expert last.
    fn moe_forward(
        &self,
        st: &Stage,
        layer: &LayerDev,
        x: &CudaSlice<f32>, // [s, hidden] post-ffn-norm
        s: usize,
        ids: &[u32],
    ) -> Res<CudaSlice<f32>> {
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let moe = mc.moe.as_ref().expect("moe");
        let hidden = mc.n_embd as usize;
        let ne = moe.expert_count as usize;
        let topk = moe.expert_used_count as usize;
        let inter = moe.expert_ff_length as usize;
        let limit = d.swiglu_limit;
        let stream = st.gpu.stream();

        let mut raw = stream.alloc_zeros::<f32>(s * ne).map_err(e("gate raw"))?;
        Self::dots(st, x, &layer.gate_w, s, hidden, ne, &mut raw)?;
        let raw_h = dtoh_f32(&stream, &raw)?;
        let (indices, weights) =
            Self::route_host(layer, &raw_h, ids, s, ne, topk, d.routed_scaling_factor);

        // x as bf16 once for all expert GEMMs
        let mut xb = stream
            .alloc_zeros::<u8>(s * hidden * 2)
            .map_err(e("xb moe"))?;
        unsafe {
            ck(
                "cvt moe x",
                k::memra_dsv4_cvt_bf16(
                    dpf!(x, &stream),
                    xb.device_ptr_mut(&stream).0 as *mut c_void,
                    (s * hidden) as i64,
                    sp(&stream),
                ),
            )?;
        }
        let mut y = stream.alloc_zeros::<f32>(s * hidden).map_err(e("moe y"))?;

        let wbytes = inter * hidden / 2;
        let sbytes = match layer.expert_kind {
            ExpertKind::Nvfp4 => inter * hidden / 16,
            ExpertKind::Mxfp4 => inter * hidden / 32,
        };
        if self.dspark_fused_moe
            && self.expert_arm == ExpertArm::Native
            && layer.expert_kind == ExpertKind::Mxfp4
            && self.dspark.as_ref().is_some_and(|ds| s == ds.block_size)
        {
            // DSpark's three MTP blocks use the host-oracle score router above.
            // Keep those exact selected ids/weights and collapse only the per-expert
            // projection loop: one indirect launch per projection over all s*topk
            // slots. Per-output accumulation and the ascending-expert combine order
            // are the same as the reference loop.
            let slots = s * topk;
            let sel: Vec<i32> = indices.iter().map(|&x| x as i32).collect();
            let mut order = vec![0i32; slots];
            for p in 0..s {
                let mut row: Vec<i32> = (0..topk as i32).collect();
                row.sort_by_key(|&slot| indices[p * topk + slot as usize]);
                order[p * topk..(p + 1) * topk].copy_from_slice(&row);
            }
            let sel_d = upload_i32(&stream, &sel)?;
            let selw_d = upload_f32(&stream, &weights)?;
            let order_d = upload_i32(&stream, &order)?;
            let kq_x = hidden / 128;
            let kq_h = inter / 128;
            let mut xq = stream.alloc_zeros::<u8>(s * hidden).map_err(e("ds xq"))?;
            let mut xs = stream.alloc_zeros::<f32>(s * kq_x).map_err(e("ds xs"))?;
            let mut g1 = stream
                .alloc_zeros::<f32>(slots * inter)
                .map_err(e("ds g1"))?;
            let mut g3 = stream
                .alloc_zeros::<f32>(slots * inter)
                .map_err(e("ds g3"))?;
            let mut hbuf = stream
                .alloc_zeros::<f32>(slots * inter)
                .map_err(e("ds hbuf"))?;
            let mut hq = stream
                .alloc_zeros::<u8>(slots * inter)
                .map_err(e("ds hq"))?;
            let mut hs = stream
                .alloc_zeros::<f32>(slots * kq_h)
                .map_err(e("ds hs"))?;
            let mut contrib = stream
                .alloc_zeros::<f32>(slots * hidden)
                .map_err(e("ds contrib"))?;
            unsafe {
                ck(
                    "dspark fused act_quant x",
                    k::memra_dsv4_act_quant_fp8(
                        dpf!(x, &stream),
                        xq.device_ptr_mut(&stream).0 as *mut c_void,
                        dpm!(xs, &stream),
                        s as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
                for (proj, dst) in [(0i32, &mut g1), (2i32, &mut g3)] {
                    ck(
                        "dspark fused w1/w3",
                        k::memra_dsv4_fp4_gemm_sel_g_arm(
                            dp!(xq, &stream),
                            dpf!(xs, &stream),
                            dp!(layer.experts_w, &stream),
                            dp!(layer.experts_sc, &stream),
                            dpf!(layer.experts_s2_dev, &stream),
                            sel_d.device_ptr(&stream).0 as *const i32,
                            proj,
                            0,
                            1,
                            dpm!(*dst, &stream),
                            slots as i32,
                            inter as i32,
                            hidden as i32,
                            wbytes as i64,
                            sbytes as i64,
                            topk as i32,
                            self.fp4_reduce as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                ck(
                    "dspark fused swiglu",
                    k::memra_dsv4_swiglu(
                        dpf!(g1, &stream),
                        dpf!(g3, &stream),
                        dpm!(hbuf, &stream),
                        slots as i32,
                        inter as i32,
                        limit,
                        dpf!(selw_d, &stream),
                        sp(&stream),
                    ),
                )?;
                ck(
                    "dspark fused act_quant h",
                    k::memra_dsv4_act_quant_fp8(
                        dpf!(hbuf, &stream),
                        hq.device_ptr_mut(&stream).0 as *mut c_void,
                        dpm!(hs, &stream),
                        slots as i32,
                        inter as i32,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "dspark fused w2",
                    k::memra_dsv4_fp4_gemm_sel_g_arm(
                        dp!(hq, &stream),
                        dpf!(hs, &stream),
                        dp!(layer.experts_w, &stream),
                        dp!(layer.experts_sc, &stream),
                        dpf!(layer.experts_s2_dev, &stream),
                        sel_d.device_ptr(&stream).0 as *const i32,
                        1,
                        1,
                        1,
                        dpm!(contrib, &stream),
                        slots as i32,
                        hidden as i32,
                        inter as i32,
                        wbytes as i64,
                        sbytes as i64,
                        0,
                        self.fp4_reduce as i32,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "dspark fused combine",
                    k::memra_dsv4_combine_rows_m(
                        dpf!(contrib, &stream),
                        order_d.device_ptr(&stream).0 as *const i32,
                        topk as i32,
                        dpm!(y, &stream),
                        hidden as i64,
                        s as i32,
                        sp(&stream),
                    ),
                )?;
            }
            return self.moe_shared_and_finish(st, layer, &xb, s, y);
        }
        let mut uniq: Vec<usize> = indices.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if self.expert_arm == ExpertArm::Native {
            // lane 7: reference-law quantized expert GEMMs (RECEIPTS.md "Lane 7").
            // x quantized ONCE per-row-per-128 (model.py:113-115); code/scale rows
            // gathered per expert (row-local quant commutes with gathering exactly);
            // h re-quantized AFTER the routing-weight multiply (M:604-606) before w2.
            let kind = match layer.expert_kind {
                ExpertKind::Nvfp4 => 0i32,
                ExpertKind::Mxfp4 => 1i32,
            };
            let kq_x = hidden / 128;
            let kq_h = inter / 128;
            let mut xq = stream.alloc_zeros::<u8>(s * hidden).map_err(e("xq"))?;
            let mut xs = stream.alloc_zeros::<f32>(s * kq_x).map_err(e("xs"))?;
            unsafe {
                ck(
                    "act_quant_fp8 x",
                    k::memra_dsv4_act_quant_fp8(
                        dpf!(x, &stream),
                        xq.device_ptr_mut(&stream).0 as *mut c_void,
                        dpm!(xs, &stream),
                        s as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
            }
            let mut xgq = stream.alloc_zeros::<u8>(s * hidden).map_err(e("xgq"))?;
            let mut xgs = stream.alloc_zeros::<f32>(s * kq_x).map_err(e("xgs"))?;
            let mut g1 = stream.alloc_zeros::<f32>(s * inter).map_err(e("g1"))?;
            let mut g3 = stream.alloc_zeros::<f32>(s * inter).map_err(e("g3"))?;
            let mut hbuf = stream.alloc_zeros::<f32>(s * inter).map_err(e("hbuf"))?;
            let mut hq = stream.alloc_zeros::<u8>(s * inter).map_err(e("hq"))?;
            let mut hs = stream.alloc_zeros::<f32>(s * kq_h).map_err(e("hs"))?;
            let mut contrib = stream
                .alloc_zeros::<f32>(s * hidden)
                .map_err(e("contrib"))?;
            for &ex in &uniq {
                let toks: Vec<(usize, usize)> = (0..s * topk)
                    .filter(|i| indices[*i] == ex)
                    .map(|i| (i / topk, i % topk))
                    .collect();
                let g = toks.len();
                let tok_rows: Vec<i32> = toks.iter().map(|&(t, _)| t as i32).collect();
                let wrow: Vec<f32> = toks.iter().map(|&(t, kk)| weights[t * topk + kk]).collect();
                let rows_dev = upload_i32(&stream, &tok_rows)?;
                let wrow_dev = upload_f32(&stream, &wrow)?;
                unsafe {
                    ck(
                        "gather xq",
                        k::memra_dsv4_gather_rows_u8(
                            dp!(xq, &stream),
                            rows_dev.device_ptr(&stream).0 as *const i32,
                            xgq.device_ptr_mut(&stream).0 as *mut c_void,
                            g as i32,
                            hidden as i64,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "gather xs",
                        k::memra_dsv4_gather_rows_u8(
                            xs.device_ptr(&stream).0 as *const c_void,
                            rows_dev.device_ptr(&stream).0 as *const i32,
                            xgs.device_ptr_mut(&stream).0 as *mut c_void,
                            g as i32,
                            (kq_x * 4) as i64,
                            sp(&stream),
                        ),
                    )?;
                    // w1 (out inter), w3 (out inter) from x codes; w2 (out hidden) from h codes
                    for (pi, dst) in [(0usize, &mut g1), (2usize, &mut g3)] {
                        let woff = (ex * 3 + pi) * wbytes;
                        let soff = (ex * 3 + pi) * sbytes;
                        ck(
                            "fp4_gemm w1/w3",
                            k::memra_dsv4_fp4_gemm(
                                dp!(xgq, &stream),
                                dpf!(xgs, &stream),
                                (layer.experts_w.device_ptr(&stream).0 as usize + woff)
                                    as *const c_void,
                                (layer.experts_sc.device_ptr(&stream).0 as usize + soff)
                                    as *const c_void,
                                layer.experts_s2[ex * 3 + pi],
                                kind,
                                dpm!(*dst, &stream),
                                g as i32,
                                inter as i32,
                                hidden as i32,
                                sp(&stream),
                            ),
                        )?;
                    }
                    ck(
                        "swiglu",
                        k::memra_dsv4_swiglu(
                            dpf!(g1, &stream),
                            dpf!(g3, &stream),
                            dpm!(hbuf, &stream),
                            g as i32,
                            inter as i32,
                            limit,
                            wrow_dev.device_ptr(&stream).0 as *const f32,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "act_quant_fp8 h",
                        k::memra_dsv4_act_quant_fp8(
                            dpf!(hbuf, &stream),
                            hq.device_ptr_mut(&stream).0 as *mut c_void,
                            dpm!(hs, &stream),
                            g as i32,
                            inter as i32,
                            sp(&stream),
                        ),
                    )?;
                    let woff2 = (ex * 3 + 1) * wbytes;
                    let soff2 = (ex * 3 + 1) * sbytes;
                    ck(
                        "fp4_gemm w2",
                        k::memra_dsv4_fp4_gemm(
                            dp!(hq, &stream),
                            dpf!(hs, &stream),
                            (layer.experts_w.device_ptr(&stream).0 as usize + woff2)
                                as *const c_void,
                            (layer.experts_sc.device_ptr(&stream).0 as usize + soff2)
                                as *const c_void,
                            layer.experts_s2[ex * 3 + 1],
                            kind,
                            dpm!(contrib, &stream),
                            g as i32,
                            hidden as i32,
                            inter as i32,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "scatter",
                        k::memra_dsv4_scatter_add(
                            dpm!(y, &stream),
                            dpf!(contrib, &stream),
                            rows_dev.device_ptr(&stream).0 as *const i32,
                            g as i32,
                            hidden as i32,
                            sp(&stream),
                        ),
                    )?;
                }
            }
            return self.moe_shared_and_finish(st, layer, &xb, s, y);
        }
        // reusable per-expert buffers sized for the worst case (all tokens on one expert)
        let mut xg = stream.alloc_zeros::<u8>(s * hidden * 2).map_err(e("xg"))?;
        let mut g1 = stream.alloc_zeros::<f32>(s * inter).map_err(e("g1"))?;
        let mut g3 = stream.alloc_zeros::<f32>(s * inter).map_err(e("g3"))?;
        let mut hbuf = stream.alloc_zeros::<f32>(s * inter).map_err(e("hbuf"))?;
        let mut hb = stream.alloc_zeros::<u8>(s * inter * 2).map_err(e("hb"))?;
        let mut contrib = stream
            .alloc_zeros::<f32>(s * hidden)
            .map_err(e("contrib"))?;
        for &ex in &uniq {
            let toks: Vec<(usize, usize)> = (0..s * topk)
                .filter(|i| indices[*i] == ex)
                .map(|i| (i / topk, i % topk))
                .collect();
            let g = toks.len();
            let tok_rows: Vec<i32> = toks.iter().map(|&(t, _)| t as i32).collect();
            let wrow: Vec<f32> = toks.iter().map(|&(t, kk)| weights[t * topk + kk]).collect();
            let rows_dev = upload_i32(&stream, &tok_rows)?;
            let wrow_dev = upload_f32(&stream, &wrow)?;
            unsafe {
                ck(
                    "gather",
                    k::memra_dsv4_gather_bf16(
                        dp!(xb, &stream),
                        rows_dev.device_ptr(&stream).0 as *const i32,
                        xg.device_ptr_mut(&stream).0 as *mut c_void,
                        g as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
                // dequant w1 (rows=inter, cols=hidden), w2 (rows=hidden, cols=inter), w3
                for (pi, (rows, cols)) in [(inter, hidden), (hidden, inter), (inter, hidden)]
                    .iter()
                    .enumerate()
                {
                    let woff = (ex * 3 + pi) * wbytes;
                    let soff = (ex * 3 + pi) * sbytes;
                    let wp =
                        (layer.experts_w.device_ptr(&stream).0 as usize + woff) as *const c_void;
                    let scp =
                        (layer.experts_sc.device_ptr(&stream).0 as usize + soff) as *const c_void;
                    let dst = st.deq[pi].device_ptr(&stream).0 as *mut c_void;
                    match layer.expert_kind {
                        ExpertKind::Nvfp4 => ck(
                            "nvfp4 deq",
                            k::memra_dsv4_nvfp4_deq_bf16(
                                wp,
                                scp,
                                layer.experts_s2[ex * 3 + pi],
                                *rows as i32,
                                *cols as i32,
                                dst,
                                sp(&stream),
                            ),
                        )?,
                        ExpertKind::Mxfp4 => ck(
                            "mxfp4 deq",
                            k::memra_dsv4_mxfp4_deq_bf16(
                                wp,
                                scp,
                                *rows as i32,
                                *cols as i32,
                                dst,
                                sp(&stream),
                            ),
                        )?,
                    }
                }
                ck(
                    "gemm w1",
                    k::memra_dsv4_gemm_bf16(
                        st.deq[0].device_ptr(&stream).0 as *const c_void,
                        dp!(xg, &stream),
                        dpm!(g1, &stream),
                        g as i32,
                        inter as i32,
                        hidden as i32,
                        st.dev as i32,
                        st.ws.device_ptr(&stream).0 as *mut c_void,
                        st.ws.len(),
                        sp(&stream),
                    ),
                )?;
                ck(
                    "gemm w3",
                    k::memra_dsv4_gemm_bf16(
                        st.deq[2].device_ptr(&stream).0 as *const c_void,
                        dp!(xg, &stream),
                        dpm!(g3, &stream),
                        g as i32,
                        inter as i32,
                        hidden as i32,
                        st.dev as i32,
                        st.ws.device_ptr(&stream).0 as *mut c_void,
                        st.ws.len(),
                        sp(&stream),
                    ),
                )?;
                ck(
                    "swiglu",
                    k::memra_dsv4_swiglu(
                        dpf!(g1, &stream),
                        dpf!(g3, &stream),
                        dpm!(hbuf, &stream),
                        g as i32,
                        inter as i32,
                        limit,
                        wrow_dev.device_ptr(&stream).0 as *const f32,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "cvt h",
                    k::memra_dsv4_cvt_bf16(
                        dpf!(hbuf, &stream),
                        hb.device_ptr_mut(&stream).0 as *mut c_void,
                        (g * inter) as i64,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "gemm w2",
                    k::memra_dsv4_gemm_bf16(
                        st.deq[1].device_ptr(&stream).0 as *const c_void,
                        dp!(hb, &stream),
                        dpm!(contrib, &stream),
                        g as i32,
                        hidden as i32,
                        inter as i32,
                        st.dev as i32,
                        st.ws.device_ptr(&stream).0 as *mut c_void,
                        st.ws.len(),
                        sp(&stream),
                    ),
                )?;
                ck(
                    "scatter",
                    k::memra_dsv4_scatter_add(
                        dpm!(y, &stream),
                        dpf!(contrib, &stream),
                        rows_dev.device_ptr(&stream).0 as *const i32,
                        g as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
        self.moe_shared_and_finish(st, layer, &xb, s, y)
    }

    /// Shared expert (unweighted, added last — oracle order) + return. Stays on the
    /// lane-4 bf16 rung under BOTH expert arms (lane-7 banked deviation: shared experts
    /// are FP8-blk weights — the FP8-linear stay-bf16 decision).
    fn moe_shared_and_finish(
        &self,
        st: &Stage,
        layer: &LayerDev,
        xb: &CudaSlice<u8>,
        s: usize,
        mut y: CudaSlice<f32>,
    ) -> Res<CudaSlice<f32>> {
        let d = self.model.cfg();
        let hidden = self.model.mc.n_embd as usize;
        let limit = d.swiglu_limit;
        let stream = st.gpu.stream();
        let sh_inter = {
            // width derived from the tensor itself (n_shared_experts * inter)
            let (shape, _) = self
                .model
                .st
                .raw("layers.0.ffn.shared_experts.w1.weight")
                .map(|(i, _)| (i.shape.clone(), ()))
                .expect("shared w1");
            shape[0] as usize
        };
        let mut sg1 = stream.alloc_zeros::<f32>(s * sh_inter).map_err(e("sg1"))?;
        let mut sg3 = stream.alloc_zeros::<f32>(s * sh_inter).map_err(e("sg3"))?;
        let mut shbuf = stream.alloc_zeros::<f32>(s * sh_inter).map_err(e("shb"))?;
        let mut shb16 = stream
            .alloc_zeros::<u8>(s * sh_inter * 2)
            .map_err(e("shb16"))?;
        let mut sh_out = stream.alloc_zeros::<f32>(s * hidden).map_err(e("sh_out"))?;
        // item 3: staged views (transient upload under the fp8 arm, borrow otherwise)
        let sw = [
            layer.shared_w[0].staged(&stream)?,
            layer.shared_w[1].staged(&stream)?,
            layer.shared_w[2].staged(&stream)?,
        ];
        Self::gemm_pre(
            st,
            xb,
            sw[0].slab().device_ptr(&stream).0 as *const c_void,
            s,
            sh_inter,
            hidden,
            &mut sg1,
        )?;
        Self::gemm_pre(
            st,
            xb,
            sw[2].slab().device_ptr(&stream).0 as *const c_void,
            s,
            sh_inter,
            hidden,
            &mut sg3,
        )?;
        unsafe {
            ck(
                "swiglu sh",
                k::memra_dsv4_swiglu(
                    dpf!(sg1, &stream),
                    dpf!(sg3, &stream),
                    dpm!(shbuf, &stream),
                    s as i32,
                    sh_inter as i32,
                    limit,
                    std::ptr::null(),
                    sp(&stream),
                ),
            )?;
            ck(
                "cvt sh",
                k::memra_dsv4_cvt_bf16(
                    dpf!(shbuf, &stream),
                    shb16.device_ptr_mut(&stream).0 as *mut c_void,
                    (s * sh_inter) as i64,
                    sp(&stream),
                ),
            )?;
        }
        Self::gemm_pre(
            st,
            &shb16,
            sw[1].slab().device_ptr(&stream).0 as *const c_void,
            s,
            hidden,
            sh_inter,
            &mut sh_out,
        )?;
        unsafe {
            ck(
                "add shared",
                k::memra_dsv4_add_inplace(
                    dpm!(y, &stream),
                    dpf!(sh_out, &stream),
                    (s * hidden) as i64,
                    sp(&stream),
                ),
            )?;
        }
        Ok(y)
    }

    /// Full trunk prefill. Returns last-position logits, or None on early exit.
    /// `early_exit_after` stops after that layer (fixture Input B replays layers 0..=3).
    pub fn forward(
        &self,
        ids: &[u32],
        capture: Option<&mut GpuCapture>,
        early_exit_after: Option<u32>,
    ) -> Res<Option<ForwardOut>> {
        self.forward_impl(ids, capture, early_exit_after, None)
    }

    /// Lane 6: prefill the prompt with the lane-4 path while POPULATING the decode
    /// caches, so decode_step can continue incrementally from ids.len().
    pub fn prefill_with_cache(&self, ids: &[u32], state: &mut DecodeState) -> Res<ForwardOut> {
        assert_eq!(state.pos, 0, "prefill_with_cache needs a fresh DecodeState");
        assert!(!ids.is_empty(), "empty prompt");
        if ids.len() > state.capacity {
            return Err(format!(
                "dsv4 prefill {} tokens exceeds session cache capacity {}",
                ids.len(),
                state.capacity
            ));
        }
        let out = self
            .forward_impl(ids, None, None, Some(state))?
            .expect("prefill logits");
        state.pos = ids.len();
        Ok(out)
    }

    /// Bounded-memory prefill through the device batched-transaction path. The first
    /// token establishes every cache class; subsequent chunks are teacher-forced and
    /// committed in full. This keeps activation and indexer-score memory proportional to
    /// `chunk`, while persistent compact state remains proportional to admitted context.
    pub fn prefill_with_cache_chunked(
        &self,
        ids: &[u32],
        state: &mut DecodeState,
        chunk: usize,
    ) -> Res<Vec<f32>> {
        assert_eq!(state.pos, 0, "chunked prefill needs a fresh DecodeState");
        if ids.is_empty() {
            return Err("empty dsv4 chunked prefill".into());
        }
        if chunk == 0 || chunk > DSV4_BATCH_WIDTH_MAX || chunk > state.transient_rows {
            return Err(format!(
                "dsv4 prefill chunk {chunk} outside 1..={} allocated transient rows",
                state.transient_rows
            ));
        }
        if ids.len() > state.capacity {
            return Err(format!(
                "dsv4 prefill {} tokens exceeds session cache capacity {}",
                ids.len(),
                state.capacity
            ));
        }
        let first = self.prefill_with_cache(&ids[..1], state)?;
        if ids.len() == 1 {
            return Ok(first.logits);
        }
        self.continue_prefix_chunked(&ids[1..], state, chunk)
    }

    /// Teacher-force a non-empty suffix through bounded batched transactions and return
    /// the next-token row after its final token.
    pub fn continue_prefix_chunked(
        &self,
        suffix: &[u32],
        state: &mut DecodeState,
        chunk: usize,
    ) -> Res<Vec<f32>> {
        if suffix.is_empty() {
            return Err("dsv4 chunked continuation needs a non-empty suffix".into());
        }
        if chunk == 0 || chunk > DSV4_BATCH_WIDTH_MAX || chunk > state.transient_rows {
            return Err(format!(
                "dsv4 continuation chunk {chunk} outside 1..={} allocated transient rows",
                state.transient_rows
            ));
        }
        if state.pos == 0 || state.pos + suffix.len() > state.capacity {
            return Err(format!(
                "dsv4 chunked continuation {} + {} outside primed capacity {}",
                state.pos,
                suffix.len(),
                state.capacity
            ));
        }
        let width = chunk.min(suffix.len());
        let mut vstate = self.alloc_prefill_state_for(state.capacity, width)?;
        let mut last_logits = None;
        for (i, toks) in suffix.chunks(width).enumerate() {
            let final_chunk = (i + 1) * width >= suffix.len();
            let output = self.prefill_head.output(final_chunk);
            let (logits, _) =
                self.verify_batch_dev_output(toks, state, &mut vstate, None, output)?;
            self.commit_verify_dev(state, &mut vstate, toks.len())?;
            if let Some(rows) = logits {
                last_logits = Some(if output == VerifyOutput::Last {
                    rows
                } else {
                    let vocab = rows.len() / toks.len();
                    rows[(toks.len() - 1) * vocab..].to_vec()
                });
            }
        }
        Ok(last_logits.expect("final chunk requested logits"))
    }

    fn forward_impl(
        &self,
        ids: &[u32],
        mut capture: Option<&mut GpuCapture>,
        early_exit_after: Option<u32>,
        mut state: Option<&mut DecodeState>,
    ) -> Res<Option<ForwardOut>> {
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let s = ids.len();
        assert!(s <= self.max_seq, "seq {s} > max_seq {}", self.max_seq);
        if let Some(cache) = state.as_deref()
            && s > cache.capacity
        {
            return Err(format!(
                "dsv4 forward {s} tokens exceeds session cache capacity {}",
                cache.capacity
            ));
        }
        let hidden = mc.n_embd as usize;
        let hc = d.hc_mult as usize;
        let n_trunk = mc.n_layer - mc.nextn_predict_layers;

        // stage 0: embed -> hc state
        let st0 = &self.stages[0];
        st0.gpu.ctx.bind_to_thread().map_err(e("bind ctx0"))?;
        let stream0 = st0.gpu.stream();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_dev = upload_i32(&stream0, &ids_i32)?;
        let mut emb = stream0.alloc_zeros::<f32>(s * hidden).map_err(e("emb"))?;
        unsafe {
            ck(
                "embed_rows",
                k::memra_dsv4_embed_rows(
                    st0.embed
                        .as_ref()
                        .expect("embed on stage 0")
                        .device_ptr(&stream0)
                        .0 as *const c_void,
                    ids_dev.device_ptr(&stream0).0 as *const i32,
                    dpm!(emb, &stream0),
                    s as i32,
                    hidden as i32,
                    sp(&stream0),
                ),
            )?;
        }
        if let Some(c) = capture.as_deref_mut()
            && c.embed_out.is_none()
        {
            c.embed_out = Some(dtoh_f32(&stream0, &emb)?);
        }
        let mut h = stream0
            .alloc_zeros::<f32>(s * hc * hidden)
            .map_err(e("h0"))?;
        unsafe {
            ck(
                "repeat_hc",
                k::memra_dsv4_repeat_hc(
                    dpf!(emb, &stream0),
                    dpm!(h, &stream0),
                    s as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream0),
                ),
            )?;
        }

        // layers, stage by stage; ONE host-bounce boundary copy at the split
        let mut cur_stage = 0usize;
        for il in 0..n_trunk {
            let stage = self.layer_stage[il as usize];
            if stage != cur_stage {
                let src_stream = self.stages[cur_stage].gpu.stream();
                let host = dtoh_f32(&src_stream, &h)?;
                let dst_stream = self.stages[stage].gpu.stream();
                self.stages[stage]
                    .gpu
                    .ctx
                    .bind_to_thread()
                    .map_err(e("bind"))?;
                h = upload_f32(&dst_stream, &host)?;
                cur_stage = stage;
            }
            let st = &self.stages[stage];
            let lidx = st
                .layers
                .iter()
                .position(|l| l.il == il)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage}"));
            let layer_cache = state.as_deref_mut().map(|ds| &mut ds.caches[il as usize]);
            h = self.block_forward(
                st,
                &st.layers[lidx],
                &h,
                s,
                ids,
                capture.as_deref_mut(),
                layer_cache,
            )?;
            if early_exit_after == Some(il) {
                self.stages[cur_stage]
                    .gpu
                    .stream()
                    .synchronize()
                    .map_err(e("sync"))?;
                return Ok(None);
            }
        }

        // head (last stage): hc_head collapse (host sigmoid gates) -> norm -> logits
        let last = self.stages.len() - 1;
        if cur_stage != last {
            let src_stream = self.stages[cur_stage].gpu.stream();
            let host = dtoh_f32(&src_stream, &h)?;
            let dst_stream = self.stages[last].gpu.stream();
            h = upload_f32(&dst_stream, &host)?;
        }
        let hc_head_fn = self.stages[last].hc_head_fn.as_ref().expect("hc_head_fn");
        let trunk_norm = self.stages[last].trunk_norm.as_ref().expect("trunk norm");
        let logits = self.head_logits_from(
            &h,
            s,
            hc_head_fn,
            &self.hc_head_base,
            &self.hc_head_scale,
            trunk_norm,
        )?;
        Ok(Some(ForwardOut { logits, h_last: h }))
    }

    /// ParallelHead (model.py:713-735): hc_head collapse (mix GEMM f32 island + host
    /// sigmoid gates, the oracle's own arithmetic) -> final RMSNorm -> last-position
    /// logits over the SHARED bf16 head. Used by the trunk head and the MTP head.
    fn head_logits_from(
        &self,
        h: &CudaSlice<f32>,
        s: usize,
        fn_w: &CudaSlice<f32>,
        base: &[f32],
        scale: &[f32],
        norm: &CudaSlice<f32>,
    ) -> Res<Vec<f32>> {
        self.head_logits_row(h, s, s - 1, fn_w, base, scale, norm)
    }

    /// Same head, logits at an arbitrary position row (lane-6 m-sensitivity probe:
    /// the reference's own realization noise is measured by comparing the SAME row
    /// under two prefill lengths).
    #[allow(clippy::too_many_arguments)]
    fn head_logits_row(
        &self,
        h: &CudaSlice<f32>,
        s: usize,
        row: usize,
        fn_w: &CudaSlice<f32>,
        base: &[f32],
        scale: &[f32],
        norm: &CudaSlice<f32>,
    ) -> Res<Vec<f32>> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let eps = mc.rms_eps;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        st.gpu.ctx.bind_to_thread().map_err(e("bind ctx head"))?;
        let stream = st.gpu.stream();
        let w = hc * hidden;
        let mut mixes = stream.alloc_zeros::<f32>(s * hc).map_err(e("hm"))?;
        Self::dots(st, h, fn_w, s, w, hc, &mut mixes)?;
        unsafe {
            ck(
                "rowsq head",
                k::memra_dsv4_rowsq_scale(
                    dpf!(h, &stream),
                    dpm!(mixes, &stream),
                    s as i32,
                    w as i32,
                    hc as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        // oracle hc_head: pre = sigmoid(mix*scale + base) + hc_eps (note: RMS eps is the
        // model rms_eps inside the mean, hc_eps only in the gate — mirrored exactly)
        let mut mixes_h = dtoh_f32(&stream, &mixes)?;
        for t in 0..s {
            for c in 0..hc {
                let m = mixes_h[t * hc + c];
                mixes_h[t * hc + c] = sigmoid_f32(m * scale[0] + base[c]) + d.hc_eps;
            }
        }
        let pre_d = upload_f32(&stream, &mixes_h)?;
        let mut collapsed = stream.alloc_zeros::<f32>(s * hidden).map_err(e("col"))?;
        unsafe {
            ck(
                "hc_collapse head",
                k::memra_dsv4_hc_collapse(
                    dpf!(h, &stream),
                    dpf!(pre_d, &stream),
                    dpm!(collapsed, &stream),
                    s as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "rmsnorm head",
                k::memra_dsv4_rmsnorm(
                    dpf!(collapsed, &stream),
                    dpf!(norm, &stream),
                    dpm!(collapsed, &stream),
                    s as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        // logits for the selected position (f32 island GEMM over bf16 head rows)
        assert!(row < s, "logits row {row} out of range (s = {s})");
        let vocab = {
            let (info, _) = self.model.st.raw("head.weight").expect("head");
            info.shape[0] as usize
        };
        let last_row = collapsed.slice(row * hidden..(row + 1) * hidden);
        let mut logits = stream.alloc_zeros::<f32>(vocab).map_err(e("logits"))?;
        unsafe {
            ck(
                "head dots",
                k::memra_dsv4_dots_f32(
                    last_row.device_ptr(&stream).0 as *const f32,
                    st.head.as_ref().expect("head").device_ptr(&stream).0 as *const c_void,
                    1,
                    dpm!(logits, &stream),
                    1,
                    hidden as i32,
                    vocab as i32,
                    sp(&stream),
                ),
            )?;
        }
        dtoh_f32(&stream, &logits)
    }

    /// MTP logits at the fixture call shape (model.py:826 — same ids to trunk and MTP;
    /// the V3 NextN drafter shift is the spec-decode lane's wiring, not claimed here).
    /// `h_trunk` = the trunk's final hc state on the LAST stage (ForwardOut::h_last).
    pub fn mtp_logits_last(&self, h_trunk: &CudaSlice<f32>, ids: &[u32]) -> Res<Vec<f32>> {
        self.mtp_logits_last_cap(h_trunk, ids, None)
    }

    /// [`Self::mtp_logits_last`] with a capture pass-through (lane 7: the native-GEMM
    /// kernel gate captures the MTP block's moe_x under want = {n_trunk}).
    pub fn mtp_logits_last_cap(
        &self,
        h_trunk: &CudaSlice<f32>,
        ids: &[u32],
        capture: Option<&mut GpuCapture>,
    ) -> Res<Vec<f32>> {
        let mtp = self.mtp.as_ref().expect("MTP not loaded");
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let eps = mc.rms_eps;
        let s = ids.len();
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        st.gpu.ctx.bind_to_thread().map_err(e("bind ctx mtp"))?;
        let stream = st.gpu.stream();

        // e = rmsnorm(embed(ids), enorm): embed rows gathered HOST-side (bit-exact bf16
        // decode, same as the oracle's embed_rows) — the embed table lives on stage 0.
        let e_host = self.model.embed_rows(ids);
        let mut e_dev = upload_f32(&stream, &e_host)?;
        unsafe {
            ck(
                "rmsnorm enorm",
                k::memra_dsv4_rmsnorm(
                    dpf!(e_dev, &stream),
                    dpf!(mtp.enorm, &stream),
                    dpm!(e_dev, &stream),
                    s as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        // x = hnorm(h_trunk) per hc copy
        let mut xh = stream
            .alloc_zeros::<f32>(s * hc * hidden)
            .map_err(e("mtp xh"))?;
        unsafe {
            ck(
                "rmsnorm hnorm",
                k::memra_dsv4_rmsnorm(
                    dpf!(h_trunk, &stream),
                    dpf!(mtp.hnorm, &stream),
                    dpm!(xh, &stream),
                    (s * hc) as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        // ep = e_proj(e) [s, hidden]; hp = h_proj(xh) per copy [s*hc, hidden]
        let mut ep = stream.alloc_zeros::<f32>(s * hidden).map_err(e("mtp ep"))?;
        Self::gemm(st, &e_dev, &mtp.e_proj, 0, s, hidden, hidden, &mut ep)?;
        let mut hp = stream
            .alloc_zeros::<f32>(s * hc * hidden)
            .map_err(e("mtp hp"))?;
        Self::gemm(st, &xh, &mtp.h_proj, 0, s * hc, hidden, hidden, &mut hp)?;
        // xm[t, c, :] = ep[t, :] + hp[t, c, :]  (e broadcast over the hc copies)
        let mut xm = stream
            .alloc_zeros::<f32>(s * hc * hidden)
            .map_err(e("mtp xm"))?;
        unsafe {
            ck(
                "repeat ep",
                k::memra_dsv4_repeat_hc(
                    dpf!(ep, &stream),
                    dpm!(xm, &stream),
                    s as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "add hp",
                k::memra_dsv4_add_inplace(
                    dpm!(xm, &stream),
                    dpf!(hp, &stream),
                    (s * hc * hidden) as i64,
                    sp(&stream),
                ),
            )?;
        }
        let xm = self.block_forward(st, &mtp.layer, &xm, s, ids, capture, None)?;
        self.head_logits_from(
            &xm,
            s,
            &mtp.hc_head_fn,
            &mtp.hc_head_base,
            &mtp.hc_head_scale,
            &mtp.norm,
        )
    }

    /// Trunk-head logits at position `row` of a ForwardOut hc state (m-sensitivity probe).
    pub fn trunk_logits_row(&self, h: &CudaSlice<f32>, s: usize, row: usize) -> Res<Vec<f32>> {
        let last = self.stages.len() - 1;
        let hc_head_fn = self.stages[last].hc_head_fn.as_ref().expect("hc_head_fn");
        let trunk_norm = self.stages[last].trunk_norm.as_ref().expect("trunk norm");
        self.head_logits_row(
            h,
            s,
            row,
            hc_head_fn,
            &self.hc_head_base,
            &self.hc_head_scale,
            trunk_norm,
        )
    }

    // ---------------------------------------------------------------- lane 6: decode

    /// Allocate the per-layer decode caches (capacity = max_seq, the reference
    /// register_buffer shape) on each layer's owning stage. Returns a fresh state
    /// (pos = 0) ready for [`Self::prefill_with_cache`].
    pub fn alloc_decode_state(&self) -> Res<DecodeState> {
        self.alloc_decode_state_for(self.max_seq)
    }

    /// Capacity-planned twin of [`Self::alloc_decode_state`]. `capacity` is the maximum
    /// token position this one session may consume; model-wide RoPE and kernel workspaces
    /// remain built for `self.max_seq`. This is the concurrency seam for a 1M-capable
    /// server: one 1M request may reserve the full compact cache, while ordinary sessions
    /// reserve only prompt + output and coexist in the remaining VRAM.
    pub fn alloc_decode_state_for(&self, capacity: usize) -> Res<DecodeState> {
        self.alloc_decode_state_for_transient(capacity, self.verify_tmax())
    }

    /// Capacity-planned allocation with an explicit batched-transaction width. The
    /// width is scratch, not semantic state, and is therefore absent from host snapshots.
    pub fn alloc_decode_state_for_transient(
        &self,
        capacity: usize,
        transient_rows: usize,
    ) -> Res<DecodeState> {
        if capacity == 0 || capacity > self.max_seq {
            return Err(format!(
                "dsv4 decode capacity {capacity} outside 1..={} model limit",
                self.max_seq
            ));
        }
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let win = d.sliding_window as usize;
        let hd = d.head_dim as usize;
        let n_trunk = mc.n_layer - mc.nextn_predict_layers;
        let mut caches = Vec::with_capacity(n_trunk as usize);
        let mut cache_bytes = vec![0u64; self.stages.len()];
        // iteration 3, rung 4: reserve T_max TRANSIENT window-kv rows per layer at
        // kvc rows [win + cap_blocks, win + cap_blocks + T_max) — where a batched verify
        // round's kv lands so the persistent ring stays read-only until commit (§3.1).
        // Zero rows when the drafter is not loaded: today's exact allocation, byte for byte.
        let trans_rows = transient_rows;
        for il in 0..n_trunk {
            let stage_i = self.layer_stage[il as usize];
            let st = &self.stages[stage_i];
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx cache"))?;
            let stream = st.gpu.stream();
            let lidx = st
                .layers
                .iter()
                .position(|l| l.il == il)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage_i}"));
            let layer = &st.layers[lidx];
            let ratio = layer.ratio;
            let cap_blocks = dsv4_cache_cap_blocks(capacity, ratio);
            let kvc_rows = win + cap_blocks + trans_rows;
            let mut bytes = (kvc_rows * hd * 4) as u64;
            let kvc = stream
                .alloc_zeros::<f32>(kvc_rows * hd)
                .map_err(e("kvc alloc"))?;
            // pending pair: kv zeros, score -inf (block-0-at-decode masking, receipts)
            let mk_pend = |latent: usize, slots: usize| -> Res<(CudaSlice<f32>, CudaSlice<f32>)> {
                let kv = stream
                    .alloc_zeros::<f32>(slots * latent)
                    .map_err(e("pend kv alloc"))?;
                let sc = upload_f32(&stream, &vec![f32::NEG_INFINITY; slots * latent])?;
                Ok((kv, sc))
            };
            let (pend_kv, pend_score) = if let Some(cmp) = &layer.cmp {
                let slots = if cmp.overlap {
                    2 * cmp.ratio
                } else {
                    cmp.ratio
                };
                bytes += (2 * slots * cmp.latent * 4) as u64;
                let (a, b) = mk_pend(cmp.latent, slots)?;
                (Some(a), Some(b))
            } else {
                (None, None)
            };
            let (ikvc, ipend_kv, ipend_score) = if let Some(ix) = &layer.idx {
                bytes += (cap_blocks * ix.cmp.d * 4) as u64;
                let store = stream
                    .alloc_zeros::<f32>(cap_blocks * ix.cmp.d)
                    .map_err(e("ikvc alloc"))?;
                let slots = if ix.cmp.overlap {
                    2 * ix.cmp.ratio
                } else {
                    ix.cmp.ratio
                };
                bytes += (2 * slots * ix.cmp.latent * 4) as u64;
                let (a, b) = mk_pend(ix.cmp.latent, slots)?;
                (Some(store), Some(a), Some(b))
            } else {
                (None, None, None)
            };
            cache_bytes[stage_i] += bytes;
            caches.push(LayerCache {
                kvc,
                n_blocks: 0,
                pend_kv,
                pend_score,
                ikvc,
                i_blocks: 0,
                ipend_kv,
                ipend_score,
            });
        }
        let ws = if matches!(self.decode_path, DecodePath::Device { .. }) {
            Some(self.alloc_step_ws()?)
        } else {
            None
        };
        for st in &self.stages {
            st.gpu.stream().synchronize().map_err(e("cache sync"))?;
        }
        Ok(DecodeState {
            caches,
            pos: 0,
            capacity,
            cache_bytes,
            transient_rows,
            ws,
        })
    }

    /// Move the LIVE compact state of a parked DSV4 session into pinned host RAM.
    ///
    /// The full-capacity device allocations are intentionally not copied. At one million
    /// tokens that would preserve dead tail bytes and the verify scratch, defeating the
    /// model's compressed-state advantage. Instead this copies the 128-token SWA rings,
    /// append-only compressed rows through each high-water mark, and the compressor/indexer
    /// pending state. Copy commands are queued per stage and synchronized once per stage.
    pub fn snapshot_decode_state(&self, state: &DecodeState) -> Res<Dsv4HostDecodeState> {
        if state.pos == 0 {
            return Err("cannot snapshot an unprimed dsv4 decode state".into());
        }
        let d = self.model.cfg();
        let win = d.sliding_window as usize;
        let hd = d.head_dim as usize;
        if state.caches.len() != self.layer_stage.len() {
            return Err(format!(
                "dsv4 state has {} layer caches, runtime expects {}",
                state.caches.len(),
                self.layer_stage.len()
            ));
        }

        // Layout pass: one monotonically packed slab per owning stage.
        let mut totals = vec![0usize; self.stages.len()];
        let mut layers = Vec::with_capacity(state.caches.len());
        for (il, cache) in state.caches.iter().enumerate() {
            let stage = self.layer_stage[il];
            let st = &self.stages[stage];
            let layer = st
                .layers
                .iter()
                .find(|l| l.il == il as u32)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage}"));
            let kvc_elems = (win + cache.n_blocks)
                .checked_mul(hd)
                .ok_or_else(|| format!("layer {il} kvc live-size overflow"))?;
            if kvc_elems > cache.kvc.len() {
                return Err(format!(
                    "layer {il} live kvc rows exceed allocation: {kvc_elems} > {}",
                    cache.kvc.len()
                ));
            }
            let span_opt = |p: Option<&CudaSlice<f32>>, totals: &mut [usize]| {
                p.map(|p| dsv4_host_span(stage, p.len(), totals))
            };
            let ikvc = match (&cache.ikvc, &layer.idx) {
                (Some(p), Some(ix)) => {
                    let elems = cache
                        .i_blocks
                        .checked_mul(ix.cmp.d)
                        .ok_or_else(|| format!("layer {il} indexer live-size overflow"))?;
                    if elems > p.len() {
                        return Err(format!(
                            "layer {il} live indexer rows exceed allocation: {elems} > {}",
                            p.len()
                        ));
                    }
                    Some(dsv4_host_span(stage, elems, &mut totals))
                }
                (None, None) => None,
                _ => {
                    return Err(format!("layer {il} indexer cache/runtime shape mismatch"));
                }
            };
            layers.push(Dsv4HostLayer {
                kvc: dsv4_host_span(stage, kvc_elems, &mut totals),
                n_blocks: cache.n_blocks,
                pend_kv: span_opt(cache.pend_kv.as_ref(), &mut totals),
                pend_score: span_opt(cache.pend_score.as_ref(), &mut totals),
                ikvc,
                i_blocks: cache.i_blocks,
                ipend_kv: span_opt(cache.ipend_kv.as_ref(), &mut totals),
                ipend_score: span_opt(cache.ipend_score.as_ref(), &mut totals),
            });
        }
        let bytes = totals.iter().try_fold(0usize, |sum, &n| {
            n.checked_mul(std::mem::size_of::<f32>())
                .and_then(|b| sum.checked_add(b))
                .ok_or_else(|| "dsv4 host-state byte count overflow".to_string())
        })?;
        let mut stages = totals
            .iter()
            .map(|&n| {
                crate::PinnedHostBuf::new(n * std::mem::size_of::<f32>())
                    .map_err(|err| format!("dsv4 pinned host slab alloc failed: {err}"))
            })
            .collect::<Res<Vec<_>>>()?;

        // Copy pass. All planes for one layer have the same owner stage by construction.
        for (il, meta) in layers.iter().enumerate() {
            let stage = meta.kvc.stage;
            let st = &self.stages[stage];
            st.gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind dsv4 snapshot ctx"))?;
            let stream = st.gpu.stream();
            let cache = &state.caches[il];
            let slab = &mut stages[stage];
            dsv4_dtoh_span(&stream, &cache.kvc, slab, &meta.kvc)?;
            for (src, span) in [
                (cache.pend_kv.as_ref(), meta.pend_kv.as_ref()),
                (cache.pend_score.as_ref(), meta.pend_score.as_ref()),
                (cache.ikvc.as_ref(), meta.ikvc.as_ref()),
                (cache.ipend_kv.as_ref(), meta.ipend_kv.as_ref()),
                (cache.ipend_score.as_ref(), meta.ipend_score.as_ref()),
            ] {
                match (src, span) {
                    (Some(src), Some(span)) => dsv4_dtoh_span(&stream, src, slab, span)?,
                    (None, None) => {}
                    _ => return Err(format!("layer {il} host snapshot optional-plane mismatch")),
                }
            }
        }
        for st in &self.stages {
            st.gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind dsv4 snapshot sync ctx"))?;
            st.gpu
                .stream()
                .synchronize()
                .map_err(e("dsv4 snapshot sync"))?;
        }
        Ok(Dsv4HostDecodeState {
            stages,
            layers,
            pos: state.pos,
            capacity: state.capacity,
            bytes,
        })
    }

    /// Re-materialize a normal device decode state from a pinned-host parked image.
    /// The caller gets fresh verify/step scratch at the parked allocation capacity; only
    /// live semantic state is uploaded. A runtime/layout mismatch refuses before copying.
    pub fn restore_decode_state(&self, host: &Dsv4HostDecodeState) -> Res<DecodeState> {
        self.restore_decode_state_for(host, host.capacity)
    }

    /// Restore while resizing the device allocation for the incoming request. This is the
    /// multi-turn growth seam: a short first turn can park a small cache, then a longer turn
    /// re-materializes it at `capacity` without cold-prefilling the shared prefix.
    pub fn restore_decode_state_for(
        &self,
        host: &Dsv4HostDecodeState,
        capacity: usize,
    ) -> Res<DecodeState> {
        self.restore_decode_state_for_transient(host, capacity, self.verify_tmax())
    }

    /// Restore at a new capacity with scratch sized for the caller's largest batched
    /// speculative or chunked-prefill transaction.
    pub fn restore_decode_state_for_transient(
        &self,
        host: &Dsv4HostDecodeState,
        capacity: usize,
        transient_rows: usize,
    ) -> Res<DecodeState> {
        if capacity < host.pos || capacity > self.max_seq {
            return Err(format!(
                "dsv4 restore capacity {capacity} outside parked pos {}..={} model limit",
                host.pos, self.max_seq
            ));
        }
        if host.stages.len() != self.stages.len() || host.layers.len() != self.layer_stage.len() {
            return Err(format!(
                "dsv4 host state layout mismatch: {} stages/{} layers, runtime {}/{}",
                host.stages.len(),
                host.layers.len(),
                self.stages.len(),
                self.layer_stage.len()
            ));
        }
        let mut state = self.alloc_decode_state_for_transient(capacity, transient_rows)?;
        for (il, meta) in host.layers.iter().enumerate() {
            let stage = self.layer_stage[il];
            if meta.kvc.stage != stage {
                return Err(format!(
                    "layer {il} host owner {} != runtime stage {stage}",
                    meta.kvc.stage
                ));
            }
            let st = &self.stages[stage];
            st.gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind dsv4 restore ctx"))?;
            let stream = st.gpu.stream();
            let slab = &host.stages[stage];
            let cache = &mut state.caches[il];
            dsv4_htod_span(&stream, slab, &meta.kvc, &mut cache.kvc)?;
            for (dst, span) in [
                (cache.pend_kv.as_mut(), meta.pend_kv.as_ref()),
                (cache.pend_score.as_mut(), meta.pend_score.as_ref()),
                (cache.ikvc.as_mut(), meta.ikvc.as_ref()),
                (cache.ipend_kv.as_mut(), meta.ipend_kv.as_ref()),
                (cache.ipend_score.as_mut(), meta.ipend_score.as_ref()),
            ] {
                match (dst, span) {
                    (Some(dst), Some(span)) if span.stage == stage => {
                        dsv4_htod_span(&stream, slab, span, dst)?
                    }
                    (None, None) => {}
                    _ => return Err(format!("layer {il} host restore optional-plane mismatch")),
                }
            }
            cache.n_blocks = meta.n_blocks;
            cache.i_blocks = meta.i_blocks;
        }
        for st in &self.stages {
            st.gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind dsv4 restore sync ctx"))?;
            st.gpu
                .stream()
                .synchronize()
                .map_err(e("dsv4 restore sync"))?;
        }
        state.pos = host.pos;
        Ok(state)
    }

    /// Lane 8: allocate the per-stage step workspace (device decode path only).
    fn alloc_step_ws(&self) -> Res<Vec<StepWs>> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let moe = mc.moe.as_ref().expect("moe");
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let heads = mc.n_head as usize;
        let hd = d.head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let iheads = d.index_n_heads as usize;
        let ihd = d.index_head_dim as usize;
        let topk = moe.expert_used_count as usize;
        let ne = moe.expert_count as usize;
        let inter = moe.expert_ff_length as usize;
        let itopk = d.index_topk as usize;
        let vocab = {
            let (info, _) = self.model.st.raw("head.weight").expect("head");
            info.shape[0] as usize
        };
        let sh_inter = {
            let (info, _) = self
                .model
                .st
                .raw("layers.0.ffn.shared_experts.w1.weight")
                .expect("shared w1");
            info.shape[0] as usize
        };
        // fine ratio (indexer-carrying) and per-class compressor maxima, config-derived
        let mut max_latent = 0usize;
        let mut max_d = 0usize;
        let mut max_shift = 0usize;
        let mut min_index_ratio = usize::MAX;
        for st in &self.stages {
            for l in &st.layers {
                for cmp in l.cmp.iter().chain(l.idx.as_ref().map(|ix| &ix.cmp)) {
                    max_latent = max_latent.max(cmp.latent);
                    max_d = max_d.max(cmp.d);
                    if cmp.overlap {
                        max_shift = max_shift.max(cmp.ratio * cmp.latent);
                    }
                }
                if let Some(ix) = &l.idx {
                    min_index_ratio = min_index_ratio.min(ix.cmp.ratio);
                }
            }
        }
        assert!(min_index_ratio != usize::MAX, "no indexer layers?");
        let score_cap = self.max_seq / min_index_ratio + 1;
        let topk_stride = score_cap.div_ceil(4096) * 512;
        let idx_tail = itopk.max(self.max_seq / 128 + 1);
        // the largest bf16 cvt any device-path gemm() performs (activation side, m=1):
        // wo_b consumes o_groups*o_lora, the o cvt covers heads*hd separately.
        let max_gemm_k = (o_groups * o_lora).max(hidden).max(q_lora).max(sh_inter);
        let mut out = Vec::with_capacity(self.stages.len());
        for st in &self.stages {
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx ws"))?;
            let s = st.gpu.stream();
            let f = |n: usize| s.alloc_zeros::<f32>(n).map_err(e("ws f32"));
            let b = |n: usize| s.alloc_zeros::<u8>(n).map_err(e("ws u8"));
            let i = |n: usize| s.alloc_zeros::<i32>(n).map_err(e("ws i32"));
            let u = |n: usize| s.alloc_zeros::<u64>(n).map_err(e("ws u64"));
            out.push(StepWs {
                h_a: f(hc * hidden)?,
                h_b: f(hc * hidden)?,
                h_rx: f(hc * hidden)?,
                emb: f(hidden)?,
                mixes: f((2 + hc) * hc)?,
                pre: f(hc)?,
                post: f(hc)?,
                comb: f(hc * hc)?,
                y_hc: f(hidden)?,
                x: f(hidden)?,
                xf: f(hidden)?,
                qr: f(q_lora)?,
                qr_b: b(q_lora * 2)?,
                q: f(heads * hd)?,
                kv: f(hd)?,
                qi: f(iheads * ihd)?,
                wproj: f(iheads)?,
                score: f(score_cap)?,
                topk_a: u(topk_stride)?,
                topk_b: u(topk_stride)?,
                topk_stride,
                idx: i(win + idx_tail)?,
                o: f(heads * hd)?,
                o_b: b(heads * hd * 2)?,
                og: f(o_groups * o_lora)?,
                attn_out: f(hidden)?,
                gemm_xb: b(max_gemm_k * 2)?,
                raw: f(ne)?,
                sel: i(topk)?,
                selw: f(topk)?,
                order: i(topk)?,
                xq: b(hidden)?,
                xs: f(hidden / 128)?,
                g1: f(topk * inter)?,
                g3: f(topk * inter)?,
                hbuf: f(topk * inter)?,
                hq: b(topk * inter)?,
                hs: f(topk * inter / 128)?,
                contrib: f(topk * hidden)?,
                y: f(hidden)?,
                xb: b(hidden * 2)?,
                sg1: f(sh_inter)?,
                sg3: f(sh_inter)?,
                shbuf: f(sh_inter)?,
                shb16: b(sh_inter * 2)?,
                sh_out: f(hidden)?,
                cmp_kv_row: f(max_latent)?,
                cmp_sc_row: f(max_latent)?,
                cmp_emit: f(2 * max_d)?,
                cmp_shift: f(max_shift.max(1))?,
                sink_scores: f(heads * (win + idx_tail))?,
                sink_evals: f(heads * (win + idx_tail))?,
                sink_den: s.alloc_zeros::<f64>(heads).map_err(e("ws f64"))?,
                head_mixes: f(hc)?,
                head_pre: f(hc)?,
                collapsed: f(hidden)?,
                logits: f(vocab)?,
                argmax: i(1)?,
                tok: i(1)?,
            });
        }
        Ok(out)
    }

    /// Incremental compressor step (reference decode state machine, M:344-377): append
    /// this position's RAW wkv/wgate rows to the pending state; when the block
    /// completes ((pos+1) % ratio == 0), emit block pos/ratio into `store` row
    /// row0 + j via the SAME pooling kernel prefill uses (overlap rides a 2-block
    /// launch whose block 1 reads prev rows [0,ratio) through dims [0,d) and cur rows
    /// [ratio,2ratio) through dims [d,2d) — the emission pooling verbatim), then
    /// norm→rope(j·ratio)→QAT, and shift cur→prev.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    fn cmp_decode(
        &self,
        st: &Stage,
        cmp: &CmpDev,
        x: &CudaSlice<f32>, // [1, hidden] post-attn-norm
        pos: usize,
        hidden: usize,
        fc_dev: &CudaSlice<f32>,
        rd: usize,
        eps: f32,
        pend_kv: &mut CudaSlice<f32>,
        pend_score: &mut CudaSlice<f32>,
        store: &mut CudaSlice<f32>,
        row0: usize,
        blocks: &mut usize,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        let (ratio, d, latent) = (cmp.ratio, cmp.d, cmp.latent);
        let mut kv_row = stream.alloc_zeros::<f32>(latent).map_err(e("dkv"))?;
        let mut sc_row = stream.alloc_zeros::<f32>(latent).map_err(e("dsc"))?;
        Self::dots(st, x, &cmp.wkv, 1, hidden, latent, &mut kv_row)?;
        Self::dots(st, x, &cmp.wgate, 1, hidden, latent, &mut sc_row)?;
        let slot = if cmp.overlap {
            ratio + pos % ratio
        } else {
            pos % ratio
        };
        {
            let src = kv_row.slice(0..latent);
            let mut dst = pend_kv.slice_mut(slot * latent..(slot + 1) * latent);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("pend kv"))?;
            let src = sc_row.slice(0..latent);
            let mut dst = pend_score.slice_mut(slot * latent..(slot + 1) * latent);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("pend sc"))?;
        }
        if (pos + 1) % ratio != 0 {
            return Ok(());
        }
        let j = pos / ratio;
        let nb_launch = if cmp.overlap { 2usize } else { 1 };
        let row_off = if cmp.overlap { d } else { 0 };
        let mut out = stream
            .alloc_zeros::<f32>(nb_launch * d)
            .map_err(e("emit"))?;
        unsafe {
            ck(
                "compressor_pool dec",
                k::memra_dsv4_compressor_pool(
                    dpf!(pend_kv, &stream),
                    dpf!(pend_score, &stream),
                    dpf!(cmp.ape, &stream),
                    dpm!(out, &stream),
                    nb_launch as i32,
                    ratio as i32,
                    d as i32,
                    latent as i32,
                    cmp.overlap as i32,
                    sp(&stream),
                ),
            )?;
            // in-place row ops at the emitted row (base + row_off), lane-4 ptr idiom
            let row_c = (out.device_ptr(&stream).0 as usize + row_off * 4) as *const f32;
            let row_m = (out.device_ptr_mut(&stream).0 as usize + row_off * 4) as *mut f32;
            ck(
                "rmsnorm dec cmp",
                k::memra_dsv4_rmsnorm(
                    row_c,
                    dpf!(cmp.norm, &stream),
                    row_m,
                    1,
                    d as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            let pos_dev = upload_i32(&stream, &[(j * ratio) as i32])?;
            ck(
                "rope dec cmp",
                k::memra_dsv4_rope(
                    row_m,
                    1,
                    1,
                    d as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
            if cmp.rotate {
                let scale = (d as f32).powf(-0.5);
                ck(
                    "hadamard dec cmp",
                    k::memra_dsv4_hadamard(row_m, 1, d as i32, scale, sp(&stream)),
                )?;
                ck(
                    "fp4 dec cmp",
                    k::memra_dsv4_fp4_act_quant(row_m, 1, d as i64, d as i32, sp(&stream)),
                )?;
            } else {
                ck(
                    "act_quant dec cmp",
                    k::memra_dsv4_act_quant(
                        row_m,
                        1,
                        d as i64,
                        (d - rd) as i32,
                        64,
                        (self.variant == ActQuantVariant::ClampOnly) as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
        {
            let src = out.slice(row_off..row_off + d);
            let mut dst = store.slice_mut((row0 + j) * d..(row0 + j + 1) * d);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(e("emit store"))?;
        }
        if cmp.overlap {
            // shift cur -> prev through a bounce (same-buffer D2D ranges must not alias)
            let mut tmp = stream
                .alloc_zeros::<f32>(ratio * latent)
                .map_err(e("shift tmp"))?;
            {
                let src = pend_kv.slice(ratio * latent..2 * ratio * latent);
                stream.memcpy_dtod(&src, &mut tmp).map_err(e("shift1"))?;
            }
            {
                let mut dst = pend_kv.slice_mut(0..ratio * latent);
                stream
                    .memcpy_dtod(&tmp.slice(0..ratio * latent), &mut dst)
                    .map_err(e("shift2"))?;
            }
            {
                let src = pend_score.slice(ratio * latent..2 * ratio * latent);
                stream.memcpy_dtod(&src, &mut tmp).map_err(e("shift3"))?;
            }
            {
                let mut dst = pend_score.slice_mut(0..ratio * latent);
                stream
                    .memcpy_dtod(&tmp.slice(0..ratio * latent), &mut dst)
                    .map_err(e("shift4"))?;
            }
        }
        *blocks = j + 1;
        Ok(())
    }

    /// One trunk block, single-token decode. h is [1, hc, hidden] f32 on the stage.
    /// Mirrors the reference decode branches: ring write (M:530), indexer with its
    /// compressor BEFORE scoring (M:415), attention compressor before sparse_attn
    /// (M:531), window/compressed index law (M:255-276). `dump` (diagnostic only)
    /// collects named intermediates for the bisect probe.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_checked_ops)] // allow: the explicit zero guard names the degenerate-ratio case; checked ops would hide the sentinel
    fn block_decode(
        &self,
        st: &Stage,
        layer: &LayerDev,
        cache: &mut LayerCache,
        h: &CudaSlice<f32>,
        pos: usize,
        tok: u32,
        mut dump: Option<&mut Vec<(String, Vec<f32>)>>,
    ) -> Res<CudaSlice<f32>> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let heads = mc.n_head as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let eps = mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;
        st.gpu.ctx.bind_to_thread().map_err(e("bind ctx"))?;
        let stream = st.gpu.stream();
        let fc_dev = if layer.ratio != 0 {
            &st.fc_yarn
        } else {
            &st.fc_plain
        };
        let clamp_only = (self.variant == ActQuantVariant::ClampOnly) as i32;
        let LayerCache {
            kvc,
            n_blocks,
            pend_kv,
            pend_score,
            ikvc,
            i_blocks,
            ipend_kv,
            ipend_score,
        } = cache;

        // ---- attention sub-block
        let (y, post, comb) = Self::hc_pre(
            st,
            h,
            &layer.hc_attn_fn,
            &layer.hc_attn_base,
            &layer.hc_attn_scale,
            1,
            hc,
            hidden,
            iters,
            hc_eps,
        )?;
        let mut x = stream.alloc_zeros::<f32>(hidden).map_err(e("x"))?;
        unsafe {
            ck(
                "rmsnorm attn",
                k::memra_dsv4_rmsnorm(
                    dpf!(y, &stream),
                    dpf!(layer.attn_norm, &stream),
                    dpm!(x, &stream),
                    1,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        if let Some(dm) = dump.as_deref_mut() {
            dm.push((format!("layer{}.x", layer.il), dtoh_f32(&stream, &x)?));
        }

        // q path (item 3: `.dev()` is lawful here — the legacy path with the fp8
        // dense arm is a BOOT refusal, so these slabs are always device-resident)
        let mut qr = stream.alloc_zeros::<f32>(q_lora).map_err(e("qr"))?;
        Self::gemm(st, &x, layer.wq_a.dev(), 0, 1, q_lora, hidden, &mut qr)?;
        unsafe {
            ck(
                "rmsnorm q",
                k::memra_dsv4_rmsnorm(
                    dpf!(qr, &stream),
                    dpf!(layer.q_norm, &stream),
                    dpm!(qr, &stream),
                    1,
                    q_lora as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        let mut qr_b = stream.alloc_zeros::<u8>(q_lora * 2).map_err(e("qr_b"))?;
        unsafe {
            ck(
                "cvt qr",
                k::memra_dsv4_cvt_bf16(
                    dpf!(qr, &stream),
                    qr_b.device_ptr_mut(&stream).0 as *mut c_void,
                    q_lora as i64,
                    sp(&stream),
                ),
            )?;
        }
        let mut q = stream.alloc_zeros::<f32>(heads * hd).map_err(e("q"))?;
        Self::gemm_pre(
            st,
            &qr_b,
            layer.wq_b.dev().device_ptr(&stream).0 as *const c_void,
            1,
            heads * hd,
            q_lora,
            &mut q,
        )?;
        let pos_dev = upload_i32(&stream, &[pos as i32])?;
        unsafe {
            ck(
                "headrms",
                k::memra_dsv4_headrms(dpm!(q, &stream), heads as i32, hd as i32, eps, sp(&stream)),
            )?;
            ck(
                "rope q",
                k::memra_dsv4_rope(
                    dpm!(q, &stream),
                    1,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
        }

        if let Some(dm) = dump.as_deref_mut() {
            dm.push((format!("layer{}.q", layer.il), dtoh_f32(&stream, &q)?));
        }
        // shared K==V latent row + window QAT, written into the ring at pos % win
        let mut kv = stream.alloc_zeros::<f32>(hd).map_err(e("kv"))?;
        Self::gemm(st, &x, layer.wkv.dev(), 0, 1, hd, hidden, &mut kv)?;
        unsafe {
            ck(
                "rmsnorm kv",
                k::memra_dsv4_rmsnorm(
                    dpf!(kv, &stream),
                    dpf!(layer.kv_norm, &stream),
                    dpm!(kv, &stream),
                    1,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope kv",
                k::memra_dsv4_rope(
                    dpm!(kv, &stream),
                    1,
                    1,
                    hd as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
            ck(
                "act_quant kv",
                k::memra_dsv4_act_quant(
                    dpm!(kv, &stream),
                    1,
                    hd as i64,
                    (hd - rd) as i32,
                    64,
                    clamp_only,
                    sp(&stream),
                ),
            )?;
        }
        {
            let slot = pos % win;
            let src = kv.slice(0..hd);
            let mut dst = kvc.slice_mut(slot * hd..(slot + 1) * hd);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(e("ring write"))?;
        }
        if let Some(dm) = dump.as_deref_mut() {
            dm.push((format!("layer{}.kv", layer.il), dtoh_f32(&stream, &kv)?));
        }

        // index assembly: window part (M:255-262 decode branches), fixed width win
        let mut idxs: Vec<i64> = vec![-1; win];
        if pos >= win - 1 {
            let sp_ = pos % win;
            let mut k_ = 0usize;
            for s_ in (sp_ + 1)..win {
                idxs[k_] = s_ as i64;
                k_ += 1;
            }
            for s_ in 0..=sp_ {
                idxs[k_] = s_ as i64;
                k_ += 1;
            }
        } else {
            for (p, v) in idxs.iter_mut().enumerate().take(pos + 1) {
                *v = p as i64;
            }
        }

        if layer.ratio != 0 {
            let cidx: Vec<i64> = if let Some(ix) = &layer.idx {
                // indexer q
                let mut qi = stream
                    .alloc_zeros::<f32>(ix.heads * ix.hd)
                    .map_err(e("qi"))?;
                Self::gemm_pre(
                    st,
                    &qr_b,
                    ix.wq_b.dev().device_ptr(&stream).0 as *const c_void,
                    1,
                    ix.heads * ix.hd,
                    q_lora,
                    &mut qi,
                )?;
                unsafe {
                    ck(
                        "rope qi",
                        k::memra_dsv4_rope(
                            dpm!(qi, &stream),
                            1,
                            ix.heads as i32,
                            ix.hd as i32,
                            rd as i32,
                            dpf!(fc_dev, &stream),
                            pos_dev.device_ptr(&stream).0 as *const i32,
                            0,
                            sp(&stream),
                        ),
                    )?;
                    let scale = (ix.hd as f32).powf(-0.5);
                    ck(
                        "hadamard qi",
                        k::memra_dsv4_hadamard(
                            dpm!(qi, &stream),
                            ix.heads as i32,
                            ix.hd as i32,
                            scale,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "fp4 qi",
                        k::memra_dsv4_fp4_act_quant(
                            dpm!(qi, &stream),
                            ix.heads as i32,
                            ix.hd as i64,
                            ix.hd as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                // indexer compressor BEFORE scoring (M:415): this step's block is scored
                self.cmp_decode(
                    st,
                    &ix.cmp,
                    &x,
                    pos,
                    hidden,
                    fc_dev,
                    rd,
                    eps,
                    ipend_kv.as_mut().expect("ipend"),
                    ipend_score.as_mut().expect("ipend"),
                    ikvc.as_mut().expect("ikvc"),
                    0,
                    i_blocks,
                )?;
                let nb = *i_blocks;
                debug_assert_eq!(nb, (pos + 1) / layer.ratio, "indexer block count");
                if nb > 0 {
                    let mut wproj = stream.alloc_zeros::<f32>(ix.heads).map_err(e("wp"))?;
                    Self::gemm(
                        st,
                        &x,
                        ix.weights_proj.dev(),
                        0,
                        1,
                        ix.heads,
                        hidden,
                        &mut wproj,
                    )?;
                    let wscale = ((ix.hd as f64).powf(-0.5) * (ix.heads as f64).powf(-0.5)) as f32;
                    let mut score = stream.alloc_zeros::<f32>(nb).map_err(e("iscore"))?;
                    unsafe {
                        ck(
                            "indexer_score dec",
                            k::memra_dsv4_indexer_score(
                                dpf!(qi, &stream),
                                dpf!(ikvc.as_ref().expect("ikvc"), &stream),
                                dpf!(wproj, &stream),
                                wscale,
                                dpm!(score, &stream),
                                1,
                                ix.heads as i32,
                                ix.hd as i32,
                                nb as i32,
                                layer.ratio as i32,
                                nb as i32, // decode law: store is causal, lim = nb
                                sp(&stream),
                            ),
                        )?;
                    }
                    let score_h = dtoh_f32(&stream, &score)?;
                    // host topk, oracle ordering (value desc, index asc), offset = win
                    let kk = ix.topk.min(nb);
                    let mut order: Vec<usize> = (0..nb).collect();
                    order.sort_by(|&a, &b| {
                        score_h[b]
                            .partial_cmp(&score_h[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.cmp(&b))
                    });
                    order
                        .into_iter()
                        .take(kk)
                        .map(|j| (j + win) as i64)
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                // coarse: all blocks incl. the one emitted this step (M:268-271 decode)
                let nb = (pos + 1) / layer.ratio;
                (0..nb).map(|j| (j + win) as i64).collect()
            };
            // attention compressor before sparse_attn (M:531)
            self.cmp_decode(
                st,
                layer.cmp.as_ref().expect("ratio!=0 has compressor"),
                &x,
                pos,
                hidden,
                fc_dev,
                rd,
                eps,
                pend_kv.as_mut().expect("pend"),
                pend_score.as_mut().expect("pend"),
                kvc,
                win,
                n_blocks,
            )?;
            debug_assert_eq!(*n_blocks, (pos + 1) / layer.ratio, "attn block count");
            idxs.extend_from_slice(&cidx);
        }
        let slots = idxs.len();
        let idxs_i32: Vec<i32> = idxs.iter().map(|&v| v as i32).collect();
        let idx_dev = upload_i32(&stream, &idxs_i32)?;

        // sparse sink attention over the layer cache + query-position de-rotation
        let mut o = stream.alloc_zeros::<f32>(heads * hd).map_err(e("o"))?;
        let scale = (hd as f64).powf(-0.5) as f32;
        unsafe {
            ck(
                "sink_attn dec",
                k::memra_dsv4_sink_attn(
                    dpf!(q, &stream),
                    dpf!(kvc, &stream),
                    idx_dev.device_ptr(&stream).0 as *const i32,
                    dpf!(layer.sink, &stream),
                    dpm!(o, &stream),
                    1,
                    heads as i32,
                    hd as i32,
                    slots as i32,
                    scale,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope o inv",
                k::memra_dsv4_rope(
                    dpm!(o, &stream),
                    1,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    1,
                    sp(&stream),
                ),
            )?;
        }

        if let Some(dm) = dump.as_deref_mut() {
            dm.push((format!("layer{}.o", layer.il), dtoh_f32(&stream, &o)?));
        }
        // grouped wo (identical to prefill at s=1)
        let gw = heads / o_groups * hd;
        let mut og = stream
            .alloc_zeros::<f32>(o_groups * o_lora)
            .map_err(e("og"))?;
        let mut o_grp = stream.alloc_zeros::<f32>(gw).map_err(e("o_grp"))?;
        let mut y_grp = stream.alloc_zeros::<f32>(o_lora).map_err(e("y_grp"))?;
        for g in 0..o_groups {
            unsafe {
                ck(
                    "take_cols",
                    k::memra_dsv4_take_cols(
                        dpf!(o, &stream),
                        dpm!(o_grp, &stream),
                        1,
                        gw as i32,
                        (heads * hd) as i64,
                        (g * gw) as i64,
                        sp(&stream),
                    ),
                )?;
            }
            Self::gemm(
                st,
                &o_grp,
                layer.wo_a.dev(),
                g * o_lora * gw,
                1,
                o_lora,
                gw,
                &mut y_grp,
            )?;
            unsafe {
                ck(
                    "place_cols",
                    k::memra_dsv4_place_cols(
                        dpf!(y_grp, &stream),
                        dpm!(og, &stream),
                        1,
                        o_lora as i32,
                        (o_groups * o_lora) as i64,
                        (g * o_lora) as i64,
                        sp(&stream),
                    ),
                )?;
            }
        }
        let mut attn_out = stream.alloc_zeros::<f32>(hidden).map_err(e("ao"))?;
        Self::gemm(
            st,
            &og,
            layer.wo_b.dev(),
            0,
            1,
            hidden,
            o_groups * o_lora,
            &mut attn_out,
        )?;

        if let Some(dm) = dump.as_deref_mut() {
            dm.push((
                format!("layer{}.attn_out", layer.il),
                dtoh_f32(&stream, &attn_out)?,
            ));
        }
        // hc_post (attention)
        let mut h2 = stream.alloc_zeros::<f32>(hc * hidden).map_err(e("h2"))?;
        unsafe {
            ck(
                "hc_post attn",
                k::memra_dsv4_hc_post(
                    dpf!(attn_out, &stream),
                    dpf!(h, &stream),
                    dpf!(post, &stream),
                    dpf!(comb, &stream),
                    dpm!(h2, &stream),
                    1,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }

        // ---- ffn sub-block
        let (y2, post2, comb2) = Self::hc_pre(
            st,
            &h2,
            &layer.hc_ffn_fn,
            &layer.hc_ffn_base,
            &layer.hc_ffn_scale,
            1,
            hc,
            hidden,
            iters,
            hc_eps,
        )?;
        let mut xf = stream.alloc_zeros::<f32>(hidden).map_err(e("xf"))?;
        unsafe {
            ck(
                "rmsnorm ffn",
                k::memra_dsv4_rmsnorm(
                    dpf!(y2, &stream),
                    dpf!(layer.ffn_norm, &stream),
                    dpm!(xf, &stream),
                    1,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        let moe_out = self.moe_forward(st, layer, &xf, 1, &[tok])?;
        if let Some(dm) = dump.as_deref_mut() {
            dm.push((
                format!("layer{}.moe_out", layer.il),
                dtoh_f32(&stream, &moe_out)?,
            ));
        }
        let mut h3 = stream.alloc_zeros::<f32>(hc * hidden).map_err(e("h3"))?;
        unsafe {
            ck(
                "hc_post ffn",
                k::memra_dsv4_hc_post(
                    dpf!(moe_out, &stream),
                    dpf!(h2, &stream),
                    dpf!(post2, &stream),
                    dpf!(comb2, &stream),
                    dpm!(h3, &stream),
                    1,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        if let Some(dm) = dump {
            dm.push((format!("layer{}.h3", layer.il), dtoh_f32(&stream, &h3)?));
        }
        Ok(h3)
    }

    /// One incremental decode step: consume `tok` at position state.pos through all
    /// trunk layers + head using the caches (hc state carried across the PP boundary
    /// by host bounce, one copy per step). Returns the full logits row predicting
    /// position state.pos + 1.
    pub fn decode_step(&self, tok: u32, state: &mut DecodeState) -> Res<Vec<f32>> {
        self.decode_step_impl(tok, state, None)
    }

    /// Diagnostic twin: returns (logits, named per-layer intermediates).
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn decode_step_probe(
        &self,
        tok: u32,
        state: &mut DecodeState,
    ) -> Res<(Vec<f32>, Vec<(String, Vec<f32>)>)> {
        let mut dump = Vec::new();
        let logits = self.decode_step_impl(tok, state, Some(&mut dump))?;
        Ok((logits, dump))
    }

    // ------------------------------------------------------------ lane 8: device path

    /// bf16 GEMV with the arena cvt scratch and raw pointers (device decode path,
    /// m = 1): cvt_bf16 then the deterministic fixed-tree memra_dsv4_gemv_bf16 —
    /// the lane-8 class-II realization of the cuBLASLt m=1 GEMMs (gated).
    #[allow(clippy::too_many_arguments)]
    fn gemm_dev(
        st: &Stage,
        x_f32: *const f32,
        xb: &mut CudaSlice<u8>,
        w: DW,
        m: usize,
        n: usize,
        kdim: usize,
        y_ptr: *mut f32,
    ) -> Res<()> {
        assert_eq!(m, 1, "gemm_dev is the m=1 decode path");
        let stream = st.gpu.stream();
        unsafe {
            ck(
                "cvt_bf16 dev",
                k::memra_dsv4_cvt_bf16(
                    x_f32,
                    xb.device_ptr_mut(&stream).0 as *mut c_void,
                    kdim as i64,
                    sp(&stream),
                ),
            )?;
        }
        let xb_ptr = xb.device_ptr(&stream).0 as *const c_void;
        Self::gemv_pre_dev(st, xb_ptr, w, n, kdim, y_ptr)
    }

    /// GEMV from an already-bf16 activation buffer (device decode path, m = 1).
    /// Dispatches on the dense-weight realization: bf16 slab, or the iteration-5 FP8
    /// pair through the bit-identical twin.
    fn gemv_pre_dev(
        st: &Stage,
        xb_ptr: *const c_void,
        w: DW,
        n: usize,
        kdim: usize,
        y_ptr: *mut f32,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        unsafe {
            match w {
                DW::Bf16(w_ptr) => ck(
                    "gemv_bf16 pre dev",
                    k::memra_dsv4_gemv_bf16(
                        w_ptr,
                        xb_ptr,
                        y_ptr,
                        n as i32,
                        kdim as i32,
                        sp(&stream),
                    ),
                )?,
                DW::Fp8 {
                    codes,
                    scales,
                    sc_cols,
                } => ck(
                    "gemv_fp8 pre dev",
                    k::memra_dsv4_gemv_fp8(
                        codes,
                        scales,
                        sc_cols,
                        xb_ptr,
                        y_ptr,
                        n as i32,
                        kdim as i32,
                        sp(&stream),
                    ),
                )?,
            }
        }
        Ok(())
    }

    /// hc_pre on the device path: dots + rowsq (unchanged kernels) then Sinkhorn either
    /// on the HOST (byte-identity arm — hc_split_sinkhorn verbatim, results uploaded
    /// into the arena) or as the single-thread device kernel (realization fork, class
    /// gated). Writes ws {mixes, pre, post, comb, y_hc}.
    #[allow(clippy::too_many_arguments)]
    // ── 0731 re-gate extension rung dispatch (MEMRA_DSV4_DOTS_ARM=f32x): each helper
    // picks the f64 kernel (default — the pinned oracle-truth bytes, also the lane-9
    // `f32` arm's bytes) or its f32acc twin. DEVICE decode path only; prefill and the
    // legacy path never route through these.
    #[allow(clippy::too_many_arguments)]
    unsafe fn rmsnorm_arm(
        &self,
        x: *const f32,
        w: *const f32,
        dst: *mut f32,
        rows: i32,
        ncols: i32,
        eps: f32,
        sv: *mut c_void,
    ) -> i32 {
        unsafe {
            if self.chains_f32 {
                k::memra_dsv4_rmsnorm_f32acc(x, w, dst, rows, ncols, eps, sv)
            } else {
                k::memra_dsv4_rmsnorm(x, w, dst, rows, ncols, eps, sv)
            }
        }
    }

    unsafe fn headrms_arm(&self, x: *mut f32, rows: i32, d: i32, eps: f32, sv: *mut c_void) -> i32 {
        unsafe {
            if self.chains_f32 {
                k::memra_dsv4_headrms_f32acc(x, rows, d, eps, sv)
            } else {
                k::memra_dsv4_headrms(x, rows, d, eps, sv)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn rowsq_scale_arm(
        &self,
        x: *const f32,
        mixes: *mut f32,
        s: i32,
        w: i32,
        rows: i32,
        eps: f32,
        sv: *mut c_void,
    ) -> i32 {
        unsafe {
            if self.chains_f32 {
                k::memra_dsv4_rowsq_scale_f32acc(x, mixes, s, w, rows, eps, sv)
            } else {
                k::memra_dsv4_rowsq_scale(x, mixes, s, w, rows, eps, sv)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn indexer_score_arm(
        &self,
        q: *const f32,
        ckv: *const f32,
        w: *const f32,
        wscale: f32,
        score: *mut f32,
        s: i32,
        heads: i32,
        hd: i32,
        nb: i32,
        ratio: i32,
        lim0: i32,
        sv: *mut c_void,
    ) -> i32 {
        unsafe {
            if self.indexer_score == Dsv4IndexerScore::Tiled {
                k::memra_dsv4_indexer_score_tiled(
                    q, ckv, w, wscale, score, s, heads, hd, nb, ratio, lim0, -1, sv,
                )
            } else if self.chains_f32 {
                k::memra_dsv4_indexer_score_f32acc(
                    q, ckv, w, wscale, score, s, heads, hd, nb, ratio, lim0, sv,
                )
            } else {
                k::memra_dsv4_indexer_score(
                    q, ckv, w, wscale, score, s, heads, hd, nb, ratio, lim0, sv,
                )
            }
        }
    }

    /// `den` is the f64 workspace either way; the f32acc twin rides a FLOAT view of the
    /// same allocation (K2 writes it, K3 reads it, within the one FFI entry).
    #[allow(clippy::too_many_arguments)]
    unsafe fn sink_attn_dec_arm(
        &self,
        q: *const f32,
        kv: *const f32,
        idxs: *const i32,
        sink: *const f32,
        scores: *mut f32,
        evals: *mut f32,
        den: *mut f64,
        o: *mut f32,
        heads: i32,
        hd: i32,
        slots: i32,
        scale: f32,
        sv: *mut c_void,
    ) -> i32 {
        unsafe {
            if self.chains_f32 {
                k::memra_dsv4_sink_attn_dec_f32acc(
                    q,
                    kv,
                    idxs,
                    sink,
                    scores,
                    evals,
                    den as *mut f32,
                    o,
                    heads,
                    hd,
                    slots,
                    scale,
                    sv,
                )
            } else {
                k::memra_dsv4_sink_attn_dec(
                    q, kv, idxs, sink, scores, evals, den, o, heads, hd, slots, scale, sv,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn hc_pre_dev(
        &self,
        st: &Stage,
        h: &CudaSlice<f32>,
        fn_w: &CudaSlice<f32>,
        base_host: &[f32],
        scale_host: &[f32],
        base_dev: &CudaSlice<f32>,
        scale_dev: &CudaSlice<f32>,
        mixes: &mut CudaSlice<f32>,
        pre: &mut CudaSlice<f32>,
        post: &mut CudaSlice<f32>,
        comb: &mut CudaSlice<f32>,
        y_hc: &mut CudaSlice<f32>,
        hc: usize,
        hidden: usize,
        iters: u32,
        hc_eps: f32,
        host_math: bool,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        let w = hc * hidden;
        let rows = (2 + hc) * hc;
        self.dots_dev(st, h, fn_w, 1, w, rows, mixes)?;
        unsafe {
            ck(
                "rowsq_scale dev",
                self.rowsq_scale_arm(
                    dpf!(h, &stream),
                    dpm!(*mixes, &stream),
                    1,
                    w as i32,
                    rows as i32,
                    hc_eps,
                    sp(&stream),
                ),
            )?;
        }
        if host_math {
            let mixes_h = dtoh_f32(&stream, mixes)?;
            let (pre_h, post_h, comb_h) =
                hc_split_sinkhorn(&mixes_h, 1, hc, scale_host, base_host, iters, hc_eps);
            stream.memcpy_htod(&pre_h, pre).map_err(e("htod pre"))?;
            stream.memcpy_htod(&post_h, post).map_err(e("htod post"))?;
            stream.memcpy_htod(&comb_h, comb).map_err(e("htod comb"))?;
        } else {
            unsafe {
                ck(
                    "hc_sinkhorn",
                    k::memra_dsv4_hc_sinkhorn(
                        dpf!(*mixes, &stream),
                        dpf!(scale_dev, &stream),
                        dpf!(base_dev, &stream),
                        dpm!(*pre, &stream),
                        dpm!(*post, &stream),
                        dpm!(*comb, &stream),
                        hc as i32,
                        iters as i32,
                        hc_eps,
                        sp(&stream),
                    ),
                )?;
            }
        }
        unsafe {
            ck(
                "hc_collapse dev",
                k::memra_dsv4_hc_collapse(
                    dpf!(h, &stream),
                    dpf!(*pre, &stream),
                    dpm!(*y_hc, &stream),
                    1,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// Incremental compressor step on the arena (cmp_decode's arithmetic verbatim:
    /// same kernels, same D2D moves; rope via the scalar-position launcher — identical
    /// kernel body). No allocations.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    fn cmp_decode_dev(
        &self,
        st: &Stage,
        cmp: &CmpDev,
        x: &CudaSlice<f32>,
        pos: usize,
        hidden: usize,
        fc_dev: &CudaSlice<f32>,
        rd: usize,
        eps: f32,
        kv_row: &mut CudaSlice<f32>,
        sc_row: &mut CudaSlice<f32>,
        emit: &mut CudaSlice<f32>,
        shift: &mut CudaSlice<f32>,
        pend_kv: &mut CudaSlice<f32>,
        pend_score: &mut CudaSlice<f32>,
        store: &mut CudaSlice<f32>,
        row0: usize,
        blocks: &mut usize,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        let (ratio, d, latent) = (cmp.ratio, cmp.d, cmp.latent);
        self.dots_dev(st, x, &cmp.wkv, 1, hidden, latent, kv_row)?;
        self.dots_dev(st, x, &cmp.wgate, 1, hidden, latent, sc_row)?;
        let slot = if cmp.overlap {
            ratio + pos % ratio
        } else {
            pos % ratio
        };
        {
            let src = kv_row.slice(0..latent);
            let mut dst = pend_kv.slice_mut(slot * latent..(slot + 1) * latent);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("pend kv"))?;
            let src = sc_row.slice(0..latent);
            let mut dst = pend_score.slice_mut(slot * latent..(slot + 1) * latent);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("pend sc"))?;
        }
        if (pos + 1) % ratio != 0 {
            return Ok(());
        }
        let j = pos / ratio;
        let nb_launch = if cmp.overlap { 2usize } else { 1 };
        let row_off = if cmp.overlap { d } else { 0 };
        unsafe {
            ck(
                "compressor_pool dec",
                k::memra_dsv4_compressor_pool(
                    dpf!(*pend_kv, &stream),
                    dpf!(*pend_score, &stream),
                    dpf!(cmp.ape, &stream),
                    dpm!(*emit, &stream),
                    nb_launch as i32,
                    ratio as i32,
                    d as i32,
                    latent as i32,
                    cmp.overlap as i32,
                    sp(&stream),
                ),
            )?;
            let row_c = (emit.device_ptr(&stream).0 as usize + row_off * 4) as *const f32;
            let row_m = (emit.device_ptr_mut(&stream).0 as usize + row_off * 4) as *mut f32;
            ck(
                "rmsnorm dec cmp",
                self.rmsnorm_arm(
                    row_c,
                    dpf!(cmp.norm, &stream),
                    row_m,
                    1,
                    d as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope_at dec cmp",
                k::memra_dsv4_rope_at(
                    row_m,
                    1,
                    d as i32,
                    rd as i32,
                    dpf!(fc_dev, &stream),
                    (j * ratio) as i32,
                    0,
                    sp(&stream),
                ),
            )?;
            if cmp.rotate {
                let scale = (d as f32).powf(-0.5);
                ck(
                    "hadamard dec cmp",
                    k::memra_dsv4_hadamard(row_m, 1, d as i32, scale, sp(&stream)),
                )?;
                ck(
                    "fp4 dec cmp",
                    k::memra_dsv4_fp4_act_quant(row_m, 1, d as i64, d as i32, sp(&stream)),
                )?;
            } else {
                ck(
                    "act_quant dec cmp",
                    k::memra_dsv4_act_quant(
                        row_m,
                        1,
                        d as i64,
                        (d - rd) as i32,
                        64,
                        (self.variant == ActQuantVariant::ClampOnly) as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
        {
            let src = emit.slice(row_off..row_off + d);
            let mut dst = store.slice_mut((row0 + j) * d..(row0 + j + 1) * d);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(e("emit store"))?;
        }
        if cmp.overlap {
            {
                let src = pend_kv.slice(ratio * latent..2 * ratio * latent);
                let mut dst = shift.slice_mut(0..ratio * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("shift1"))?;
            }
            {
                let src = shift.slice(0..ratio * latent);
                let mut dst = pend_kv.slice_mut(0..ratio * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("shift2"))?;
            }
            {
                let src = pend_score.slice(ratio * latent..2 * ratio * latent);
                let mut dst = shift.slice_mut(0..ratio * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("shift3"))?;
            }
            {
                let src = shift.slice(0..ratio * latent);
                let mut dst = pend_score.slice_mut(0..ratio * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("shift4"))?;
            }
        }
        *blocks = j + 1;
        Ok(())
    }

    /// One trunk block, single-token decode, device path (block_decode's flow on the
    /// arena; per-value arithmetic identical under host_math — deviations under device
    /// math are the banked Sinkhorn/router realization forks). Input h is ws.h_a
    /// (or ws.h_rx right after the boundary); output lands in ws.h_a.
    #[allow(clippy::too_many_arguments)]
    fn block_decode_dev(
        &self,
        st: &Stage,
        layer: &LayerDev,
        cache: &mut LayerCache,
        ws: &mut StepWs,
        input_rx: bool,
        pos: usize,
        tok: u32,
        host_math: bool,
    ) -> Res<()> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let heads = mc.n_head as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let eps = mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;
        let stream = st.gpu.stream();
        let fc_dev: *const f32 = if layer.ratio != 0 {
            st.fc_yarn.device_ptr(&stream).0 as *const f32
        } else {
            st.fc_plain.device_ptr(&stream).0 as *const f32
        };
        let clamp_only = (self.variant == ActQuantVariant::ClampOnly) as i32;
        let LayerCache {
            kvc,
            n_blocks,
            pend_kv,
            pend_score,
            ikvc,
            i_blocks,
            ipend_kv,
            ipend_score,
        } = cache;

        // ---- attention sub-block
        {
            // split-borrow the arena fields we need for hc_pre
            let StepWs {
                h_a,
                h_rx,
                mixes,
                pre,
                post,
                comb,
                y_hc,
                ..
            } = ws;
            let h_in: &CudaSlice<f32> = if input_rx { h_rx } else { h_a };
            self.hc_pre_dev(
                st,
                h_in,
                &layer.hc_attn_fn,
                &layer.hc_attn_base,
                &layer.hc_attn_scale,
                &layer.hc_attn_base_dev,
                &layer.hc_attn_scale_dev,
                mixes,
                pre,
                post,
                comb,
                y_hc,
                hc,
                hidden,
                iters,
                hc_eps,
                host_math,
            )?;
        }
        unsafe {
            ck(
                "rmsnorm attn dev",
                self.rmsnorm_arm(
                    dpf!(ws.y_hc, &stream),
                    dpf!(layer.attn_norm, &stream),
                    dpm!(ws.x, &stream),
                    1,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }

        // q path
        Self::gemm_dev(
            st,
            ws.x.device_ptr(&stream).0 as *const f32,
            &mut ws.gemm_xb,
            dwsel(self.dense_fp8, &stream, &layer.wq_a, &layer.wq_a_fp8),
            1,
            q_lora,
            hidden,
            ws.qr.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "rmsnorm q dev",
                self.rmsnorm_arm(
                    dpf!(ws.qr, &stream),
                    dpf!(layer.q_norm, &stream),
                    dpm!(ws.qr, &stream),
                    1,
                    q_lora as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "cvt qr dev",
                k::memra_dsv4_cvt_bf16(
                    dpf!(ws.qr, &stream),
                    ws.qr_b.device_ptr_mut(&stream).0 as *mut c_void,
                    q_lora as i64,
                    sp(&stream),
                ),
            )?;
        }
        Self::gemv_pre_dev(
            st,
            ws.qr_b.device_ptr(&stream).0 as *const c_void,
            dwsel(self.dense_fp8, &stream, &layer.wq_b, &layer.wq_b_fp8),
            heads * hd,
            q_lora,
            ws.q.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "headrms dev",
                self.headrms_arm(
                    dpm!(ws.q, &stream),
                    heads as i32,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope_at q dev",
                k::memra_dsv4_rope_at(
                    dpm!(ws.q, &stream),
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    fc_dev,
                    pos as i32,
                    0,
                    sp(&stream),
                ),
            )?;
        }

        // shared K==V latent row + window QAT + ring write
        Self::gemm_dev(
            st,
            ws.x.device_ptr(&stream).0 as *const f32,
            &mut ws.gemm_xb,
            dwsel(self.dense_fp8, &stream, &layer.wkv, &layer.wkv_fp8),
            1,
            hd,
            hidden,
            ws.kv.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "rmsnorm kv dev",
                self.rmsnorm_arm(
                    dpf!(ws.kv, &stream),
                    dpf!(layer.kv_norm, &stream),
                    dpm!(ws.kv, &stream),
                    1,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope_at kv dev",
                k::memra_dsv4_rope_at(
                    dpm!(ws.kv, &stream),
                    1,
                    hd as i32,
                    rd as i32,
                    fc_dev,
                    pos as i32,
                    0,
                    sp(&stream),
                ),
            )?;
            ck(
                "act_quant kv dev",
                k::memra_dsv4_act_quant(
                    dpm!(ws.kv, &stream),
                    1,
                    hd as i64,
                    (hd - rd) as i32,
                    64,
                    clamp_only,
                    sp(&stream),
                ),
            )?;
        }
        {
            let slot = pos % win;
            let src = ws.kv.slice(0..hd);
            let mut dst = kvc.slice_mut(slot * hd..(slot + 1) * hd);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(e("ring write"))?;
        }

        // index list: window part on device (block_decode's builder verbatim)
        let mut slots = win;
        if layer.ratio != 0 {
            if let Some(ix) = &layer.idx {
                // indexer q
                Self::gemv_pre_dev(
                    st,
                    ws.qr_b.device_ptr(&stream).0 as *const c_void,
                    dwsel(self.dense_fp8, &stream, &ix.wq_b, &ix.wq_b_fp8),
                    ix.heads * ix.hd,
                    q_lora,
                    ws.qi.device_ptr_mut(&stream).0 as *mut f32,
                )?;
                unsafe {
                    ck(
                        "rope_at qi dev",
                        k::memra_dsv4_rope_at(
                            dpm!(ws.qi, &stream),
                            ix.heads as i32,
                            ix.hd as i32,
                            rd as i32,
                            fc_dev,
                            pos as i32,
                            0,
                            sp(&stream),
                        ),
                    )?;
                    let scale = (ix.hd as f32).powf(-0.5);
                    ck(
                        "hadamard qi dev",
                        k::memra_dsv4_hadamard(
                            dpm!(ws.qi, &stream),
                            ix.heads as i32,
                            ix.hd as i32,
                            scale,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "fp4 qi dev",
                        k::memra_dsv4_fp4_act_quant(
                            dpm!(ws.qi, &stream),
                            ix.heads as i32,
                            ix.hd as i64,
                            ix.hd as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                // indexer compressor BEFORE scoring (M:415)
                {
                    let StepWs {
                        x,
                        cmp_kv_row,
                        cmp_sc_row,
                        cmp_emit,
                        cmp_shift,
                        ..
                    } = ws;
                    self.cmp_decode_dev(
                        st,
                        &ix.cmp,
                        x,
                        pos,
                        hidden,
                        if layer.ratio != 0 {
                            &st.fc_yarn
                        } else {
                            &st.fc_plain
                        },
                        rd,
                        eps,
                        cmp_kv_row,
                        cmp_sc_row,
                        cmp_emit,
                        cmp_shift,
                        ipend_kv.as_mut().expect("ipend"),
                        ipend_score.as_mut().expect("ipend"),
                        ikvc.as_mut().expect("ikvc"),
                        0,
                        i_blocks,
                    )?;
                }
                let nb = *i_blocks;
                debug_assert_eq!(nb, (pos + 1) / layer.ratio, "indexer block count");
                // window part (fills [0, win)); fine tail written by the top-k below
                unsafe {
                    ck(
                        "build_idx win",
                        k::memra_dsv4_build_idx(
                            ws.idx.device_ptr_mut(&stream).0 as *mut i32,
                            pos as i32,
                            win as i32,
                            -1,
                            win as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                if nb > 0 {
                    Self::gemm_dev(
                        st,
                        ws.x.device_ptr(&stream).0 as *const f32,
                        &mut ws.gemm_xb,
                        dwsel(
                            self.dense_fp8,
                            &stream,
                            &ix.weights_proj,
                            &ix.weights_proj_fp8,
                        ),
                        1,
                        ix.heads,
                        hidden,
                        ws.wproj.device_ptr_mut(&stream).0 as *mut f32,
                    )?;
                    let wscale = ((ix.hd as f64).powf(-0.5) * (ix.heads as f64).powf(-0.5)) as f32;
                    unsafe {
                        ck(
                            "indexer_score dev",
                            self.indexer_score_arm(
                                dpf!(ws.qi, &stream),
                                dpf!(ikvc.as_ref().expect("ikvc"), &stream),
                                dpf!(ws.wproj, &stream),
                                wscale,
                                dpm!(ws.score, &stream),
                                1,
                                ix.heads as i32,
                                ix.hd as i32,
                                nb as i32,
                                layer.ratio as i32,
                                nb as i32,
                                sp(&stream),
                            ),
                        )?;
                    }
                    let kk = ix.topk.min(nb);
                    if host_math {
                        // byte-identity arm: the legacy host sort verbatim, uploaded
                        // into the arena index tail
                        let score_h = {
                            let view = ws.score.slice(0..nb);
                            let mut v = vec![0f32; nb];
                            stream
                                .memcpy_dtoh(&view, &mut v[..])
                                .map_err(e("dtoh sc"))?;
                            stream.synchronize().map_err(e("sync sc"))?;
                            v
                        };
                        let mut order: Vec<usize> = (0..nb).collect();
                        order.sort_by(|&a, &b| {
                            score_h[b]
                                .partial_cmp(&score_h[a])
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.cmp(&b))
                        });
                        let cidx: Vec<i32> = order
                            .into_iter()
                            .take(kk)
                            .map(|j| (j + win) as i32)
                            .collect();
                        let mut dst = ws.idx.slice_mut(win..win + kk);
                        stream.memcpy_htod(&cidx, &mut dst).map_err(e("htod idx"))?;
                    } else {
                        let idx_stride = ws.idx.len();
                        unsafe {
                            let rc = if nb > 4096 {
                                // The legacy bitonic selector refuses nb>4096.
                                // Decode needs the same exact long selector as prefill.
                                k::memra_dsv4_topk_idx_stream_m(
                                    dpf!(ws.score, &stream),
                                    1,
                                    nb as i32,
                                    kk as i32,
                                    win as i32,
                                    ws.idx.device_ptr_mut(&stream).0 as *mut i32,
                                    idx_stride as i32,
                                    ws.topk_a.device_ptr_mut(&stream).0 as *mut u64,
                                    ws.topk_b.device_ptr_mut(&stream).0 as *mut u64,
                                    ws.topk_stride as i32,
                                    sp(&stream),
                                )
                            } else {
                                let idx_tail = (ws.idx.device_ptr_mut(&stream).0 as usize + win * 4)
                                    as *mut i32;
                                k::memra_dsv4_topk_idx(
                                    dpf!(ws.score, &stream),
                                    nb as i32,
                                    kk as i32,
                                    win as i32,
                                    idx_tail,
                                    sp(&stream),
                                )
                            };
                            ck("topk_idx dev", rc)?;
                        }
                    }
                    slots = win + kk;
                }
            } else {
                // coarse: all blocks incl. the one emitted this step — but the ATTENTION
                // compressor below is what emits it, so the count is (pos+1)/ratio
                let nb = (pos + 1) / layer.ratio;
                unsafe {
                    ck(
                        "build_idx coarse",
                        k::memra_dsv4_build_idx(
                            ws.idx.device_ptr_mut(&stream).0 as *mut i32,
                            pos as i32,
                            win as i32,
                            nb as i32,
                            (win + nb) as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                slots = win + nb;
            }
            // attention compressor before sparse_attn (M:531)
            {
                let StepWs {
                    x,
                    cmp_kv_row,
                    cmp_sc_row,
                    cmp_emit,
                    cmp_shift,
                    ..
                } = ws;
                self.cmp_decode_dev(
                    st,
                    layer.cmp.as_ref().expect("ratio!=0 has compressor"),
                    x,
                    pos,
                    hidden,
                    &st.fc_yarn,
                    rd,
                    eps,
                    cmp_kv_row,
                    cmp_sc_row,
                    cmp_emit,
                    cmp_shift,
                    pend_kv.as_mut().expect("pend"),
                    pend_score.as_mut().expect("pend"),
                    kvc,
                    win,
                    n_blocks,
                )?;
            }
            debug_assert_eq!(*n_blocks, (pos + 1) / layer.ratio, "attn block count");
        } else {
            // window-only layer: fixed-width window part with -1 pads (legacy widths)
            unsafe {
                ck(
                    "build_idx window-only",
                    k::memra_dsv4_build_idx(
                        ws.idx.device_ptr_mut(&stream).0 as *mut i32,
                        pos as i32,
                        win as i32,
                        -1,
                        win as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }

        // sparse sink attention (lane-8 three-kernel split, bit-exact — see the .cu
        // notes) + query-position de-rotation
        let scale = (hd as f64).powf(-0.5) as f32;
        unsafe {
            ck(
                "sink_attn_dec dev",
                self.sink_attn_dec_arm(
                    dpf!(ws.q, &stream),
                    dpf!(kvc, &stream),
                    ws.idx.device_ptr(&stream).0 as *const i32,
                    dpf!(layer.sink, &stream),
                    dpm!(ws.sink_scores, &stream),
                    dpm!(ws.sink_evals, &stream),
                    ws.sink_den.device_ptr_mut(&stream).0 as *mut f64,
                    dpm!(ws.o, &stream),
                    heads as i32,
                    hd as i32,
                    slots as i32,
                    scale,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope_at o inv dev",
                k::memra_dsv4_rope_at(
                    dpm!(ws.o, &stream),
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    fc_dev,
                    pos as i32,
                    1,
                    sp(&stream),
                ),
            )?;
        }

        // grouped wo: cvt o ONCE (elementwise — bit-equal to the legacy per-group cvt),
        // then per-group offset GEMMs straight into og slices (take/place_cols are pure
        // offsets at s=1), then wo_b.
        let gw = heads / o_groups * hd;
        unsafe {
            ck(
                "cvt o dev",
                k::memra_dsv4_cvt_bf16(
                    dpf!(ws.o, &stream),
                    ws.o_b.device_ptr_mut(&stream).0 as *mut c_void,
                    (heads * hd) as i64,
                    sp(&stream),
                ),
            )?;
        }
        let wo_a_dw = dwsel(self.dense_fp8, &stream, &layer.wo_a, &layer.wo_a_fp8);
        for g in 0..o_groups {
            Self::gemv_pre_dev(
                st,
                (ws.o_b.device_ptr(&stream).0 as usize + g * gw * 2) as *const c_void,
                wo_a_dw.offset_rows(g * o_lora, gw),
                o_lora,
                gw,
                (ws.og.device_ptr_mut(&stream).0 as usize + g * o_lora * 4) as *mut f32,
            )?;
        }
        Self::gemm_dev(
            st,
            ws.og.device_ptr(&stream).0 as *const f32,
            &mut ws.gemm_xb,
            dwsel(self.dense_fp8, &stream, &layer.wo_b, &layer.wo_b_fp8),
            1,
            hidden,
            o_groups * o_lora,
            ws.attn_out.device_ptr_mut(&stream).0 as *mut f32,
        )?;

        // hc_post (attention): h2 = ws.h_b from residual h_in
        {
            let StepWs {
                h_a,
                h_b,
                h_rx,
                attn_out,
                post,
                comb,
                ..
            } = ws;
            let h_in: &CudaSlice<f32> = if input_rx { h_rx } else { h_a };
            unsafe {
                ck(
                    "hc_post attn dev",
                    k::memra_dsv4_hc_post(
                        dpf!(attn_out, &stream),
                        dpf!(h_in, &stream),
                        dpf!(post, &stream),
                        dpf!(comb, &stream),
                        dpm!(*h_b, &stream),
                        1,
                        hc as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }

        // ---- ffn sub-block (input h2 = ws.h_b, output h3 = ws.h_a)
        {
            let StepWs {
                h_b,
                mixes,
                pre,
                post,
                comb,
                y_hc,
                ..
            } = ws;
            self.hc_pre_dev(
                st,
                h_b,
                &layer.hc_ffn_fn,
                &layer.hc_ffn_base,
                &layer.hc_ffn_scale,
                &layer.hc_ffn_base_dev,
                &layer.hc_ffn_scale_dev,
                mixes,
                pre,
                post,
                comb,
                y_hc,
                hc,
                hidden,
                iters,
                hc_eps,
                host_math,
            )?;
        }
        unsafe {
            ck(
                "rmsnorm ffn dev",
                self.rmsnorm_arm(
                    dpf!(ws.y_hc, &stream),
                    dpf!(layer.ffn_norm, &stream),
                    dpm!(ws.xf, &stream),
                    1,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        self.moe_forward_dev(st, layer, ws, tok, host_math)?;
        {
            let StepWs {
                h_a,
                h_b,
                y,
                post,
                comb,
                ..
            } = ws;
            unsafe {
                ck(
                    "hc_post ffn dev",
                    k::memra_dsv4_hc_post(
                        dpf!(y, &stream),
                        dpf!(h_b, &stream),
                        dpf!(post, &stream),
                        dpf!(comb, &stream),
                        dpm!(*h_a, &stream),
                        1,
                        hc as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// MoE on the device path (native fp4 arm only, asserted at load): routing via the
    /// device kernel (or route_host under host_math), then ONE launch per projection
    /// over all active-expert slots (indirect fused dispatch — attack #3 at s=1),
    /// combine in ascending-expert-id order (the legacy scatter sequence), shared
    /// expert on the lane-4 bf16 rung. Writes ws.y.
    fn moe_forward_dev(
        &self,
        st: &Stage,
        layer: &LayerDev,
        ws: &mut StepWs,
        tok: u32,
        host_math: bool,
    ) -> Res<()> {
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let moe = mc.moe.as_ref().expect("moe");
        let hidden = mc.n_embd as usize;
        let ne = moe.expert_count as usize;
        let topk = moe.expert_used_count as usize;
        let inter = moe.expert_ff_length as usize;
        let limit = d.swiglu_limit;
        let stream = st.gpu.stream();
        let kind = match layer.expert_kind {
            ExpertKind::Nvfp4 => 0i32,
            ExpertKind::Mxfp4 => 1i32,
        };
        let wstride = (inter * hidden / 2) as i64;
        let sstride = match layer.expert_kind {
            ExpertKind::Nvfp4 => (inter * hidden / 16) as i64,
            ExpertKind::Mxfp4 => (inter * hidden / 32) as i64,
        };

        self.dots_dev(st, &ws.xf, &layer.gate_w, 1, hidden, ne, &mut ws.raw)?;
        if host_math {
            let raw_h = dtoh_f32(&stream, &ws.raw)?;
            let (indices, weights) =
                Self::route_host(layer, &raw_h, &[tok], 1, ne, topk, d.routed_scaling_factor);
            let sel: Vec<i32> = indices.iter().map(|&x| x as i32).collect();
            let mut order: Vec<i32> = (0..topk as i32).collect();
            order.sort_by_key(|&s| indices[s as usize]);
            stream
                .memcpy_htod(&sel, &mut ws.sel)
                .map_err(e("htod sel"))?;
            stream
                .memcpy_htod(&weights, &mut ws.selw)
                .map_err(e("htod selw"))?;
            stream
                .memcpy_htod(&order, &mut ws.order)
                .map_err(e("htod order"))?;
        } else {
            unsafe {
                ck(
                    "route dev",
                    k::memra_dsv4_route(
                        dpf!(ws.raw, &stream),
                        layer
                            .gate_bias_dev
                            .as_ref()
                            .map(|b| b.device_ptr(&stream).0 as *const f32)
                            .unwrap_or(std::ptr::null()),
                        layer
                            .tid2eid_dev
                            .as_ref()
                            .map(|t| t.device_ptr(&stream).0 as *const i32)
                            .unwrap_or(std::ptr::null()),
                        ws.tok.device_ptr(&stream).0 as *const i32,
                        ne as i32,
                        topk as i32,
                        d.routed_scaling_factor,
                        ws.sel.device_ptr_mut(&stream).0 as *mut i32,
                        ws.selw.device_ptr_mut(&stream).0 as *mut f32,
                        ws.order.device_ptr_mut(&stream).0 as *mut i32,
                        sp(&stream),
                    ),
                )?;
            }
        }

        unsafe {
            ck(
                "act_quant_fp8 x dev",
                k::memra_dsv4_act_quant_fp8(
                    dpf!(ws.xf, &stream),
                    ws.xq.device_ptr_mut(&stream).0 as *mut c_void,
                    dpm!(ws.xs, &stream),
                    1,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
            for (proj, dst) in [(0i32, &mut ws.g1), (2i32, &mut ws.g3)] {
                ck(
                    "fp4_gemm_sel w1/w3",
                    k::memra_dsv4_fp4_gemm_sel_g_arm(
                        dp!(ws.xq, &stream),
                        dpf!(ws.xs, &stream),
                        dp!(layer.experts_w, &stream),
                        dp!(layer.experts_sc, &stream),
                        dpf!(layer.experts_s2_dev, &stream),
                        ws.sel.device_ptr(&stream).0 as *const i32,
                        proj,
                        0,
                        kind,
                        dpm!(*dst, &stream),
                        topk as i32,
                        inter as i32,
                        hidden as i32,
                        wstride,
                        sstride,
                        0,
                        self.fp4_reduce as i32,
                        sp(&stream),
                    ),
                )?;
            }
            ck(
                "swiglu dev",
                k::memra_dsv4_swiglu(
                    dpf!(ws.g1, &stream),
                    dpf!(ws.g3, &stream),
                    dpm!(ws.hbuf, &stream),
                    topk as i32,
                    inter as i32,
                    limit,
                    ws.selw.device_ptr(&stream).0 as *const f32,
                    sp(&stream),
                ),
            )?;
            ck(
                "act_quant_fp8 h dev",
                k::memra_dsv4_act_quant_fp8(
                    dpf!(ws.hbuf, &stream),
                    ws.hq.device_ptr_mut(&stream).0 as *mut c_void,
                    dpm!(ws.hs, &stream),
                    topk as i32,
                    inter as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "fp4_gemm_sel w2",
                k::memra_dsv4_fp4_gemm_sel_g_arm(
                    dp!(ws.hq, &stream),
                    dpf!(ws.hs, &stream),
                    dp!(layer.experts_w, &stream),
                    dp!(layer.experts_sc, &stream),
                    dpf!(layer.experts_s2_dev, &stream),
                    ws.sel.device_ptr(&stream).0 as *const i32,
                    1,
                    1,
                    kind,
                    dpm!(ws.contrib, &stream),
                    topk as i32,
                    hidden as i32,
                    inter as i32,
                    wstride,
                    sstride,
                    0,
                    self.fp4_reduce as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "combine dev",
                k::memra_dsv4_combine_rows(
                    dpf!(ws.contrib, &stream),
                    ws.order.device_ptr(&stream).0 as *const i32,
                    topk as i32,
                    dpm!(ws.y, &stream),
                    hidden as i64,
                    sp(&stream),
                ),
            )?;
            // shared expert (lane-4 bf16 rung — the lane-7 FP8-linear decision)
            ck(
                "cvt xb dev",
                k::memra_dsv4_cvt_bf16(
                    dpf!(ws.xf, &stream),
                    ws.xb.device_ptr_mut(&stream).0 as *mut c_void,
                    hidden as i64,
                    sp(&stream),
                ),
            )?;
        }
        let sh_inter = ws.sg1.len();
        Self::gemv_pre_dev(
            st,
            ws.xb.device_ptr(&stream).0 as *const c_void,
            dwsel(
                self.dense_fp8,
                &stream,
                &layer.shared_w[0],
                &layer.shared_fp8[0],
            ),
            sh_inter,
            hidden,
            ws.sg1.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        Self::gemv_pre_dev(
            st,
            ws.xb.device_ptr(&stream).0 as *const c_void,
            dwsel(
                self.dense_fp8,
                &stream,
                &layer.shared_w[2],
                &layer.shared_fp8[2],
            ),
            sh_inter,
            hidden,
            ws.sg3.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "swiglu sh dev",
                k::memra_dsv4_swiglu(
                    dpf!(ws.sg1, &stream),
                    dpf!(ws.sg3, &stream),
                    dpm!(ws.shbuf, &stream),
                    1,
                    sh_inter as i32,
                    limit,
                    std::ptr::null(),
                    sp(&stream),
                ),
            )?;
            ck(
                "cvt sh dev",
                k::memra_dsv4_cvt_bf16(
                    dpf!(ws.shbuf, &stream),
                    ws.shb16.device_ptr_mut(&stream).0 as *mut c_void,
                    sh_inter as i64,
                    sp(&stream),
                ),
            )?;
        }
        Self::gemv_pre_dev(
            st,
            ws.shb16.device_ptr(&stream).0 as *const c_void,
            dwsel(
                self.dense_fp8,
                &stream,
                &layer.shared_w[1],
                &layer.shared_fp8[1],
            ),
            hidden,
            sh_inter,
            ws.sh_out.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "add shared dev",
                k::memra_dsv4_add_inplace(
                    dpm!(ws.y, &stream),
                    dpf!(ws.sh_out, &stream),
                    hidden as i64,
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// Head on the device path: hc_head gate + collapse + trunk norm + vocab dots into
    /// ws.logits (dtoh'd by the caller when wanted). head_logits_row's arithmetic with
    /// the host sigmoid either kept (host_math) or run as the tiny gate kernel.
    fn head_logits_dev(&self, ws: &mut StepWs, host_math: bool) -> Res<()> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let eps = mc.rms_eps;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        let stream = st.gpu.stream();
        let w = hc * hidden;
        let fn_w = st.hc_head_fn.as_ref().expect("hc_head_fn");
        let norm = st.trunk_norm.as_ref().expect("trunk norm");
        self.dots_dev(st, &ws.h_a, fn_w, 1, w, hc, &mut ws.head_mixes)?;
        unsafe {
            ck(
                "rowsq head dev",
                self.rowsq_scale_arm(
                    dpf!(ws.h_a, &stream),
                    dpm!(ws.head_mixes, &stream),
                    1,
                    w as i32,
                    hc as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        if host_math {
            let mut mixes_h = dtoh_f32(&stream, &ws.head_mixes)?;
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for c in 0..hc {
                let m = mixes_h[c];
                mixes_h[c] =
                    sigmoid_f32(m * self.hc_head_scale[0] + self.hc_head_base[c]) + d.hc_eps;
            }
            stream
                .memcpy_htod(&mixes_h, &mut ws.head_pre)
                .map_err(e("htod head pre"))?;
        } else {
            unsafe {
                ck(
                    "hc_head_pre dev",
                    k::memra_dsv4_hc_head_pre(
                        dpf!(ws.head_mixes, &stream),
                        st.hc_head_scale_dev
                            .as_ref()
                            .expect("head scale dev")
                            .device_ptr(&stream)
                            .0 as *const f32,
                        st.hc_head_base_dev
                            .as_ref()
                            .expect("head base dev")
                            .device_ptr(&stream)
                            .0 as *const f32,
                        dpm!(ws.head_pre, &stream),
                        hc as i32,
                        d.hc_eps,
                        sp(&stream),
                    ),
                )?;
            }
        }
        unsafe {
            ck(
                "hc_collapse head dev",
                k::memra_dsv4_hc_collapse(
                    dpf!(ws.h_a, &stream),
                    dpf!(ws.head_pre, &stream),
                    dpm!(ws.collapsed, &stream),
                    1,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "rmsnorm head dev",
                self.rmsnorm_arm(
                    dpf!(ws.collapsed, &stream),
                    dpf!(norm, &stream),
                    dpm!(ws.collapsed, &stream),
                    1,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            let head_ptr = st.head.as_ref().expect("head").device_ptr(&stream).0 as *const c_void;
            if self.dots_f32 {
                ck(
                    "head dots f32acc dev",
                    k::memra_dsv4_dots_f32acc(
                        dpf!(ws.collapsed, &stream),
                        head_ptr,
                        1,
                        dpm!(ws.logits, &stream),
                        1,
                        hidden as i32,
                        ws.logits.len() as i32,
                        sp(&stream),
                    ),
                )?;
            } else {
                ck(
                    "head dots dev",
                    k::memra_dsv4_dots_f32(
                        dpf!(ws.collapsed, &stream),
                        head_ptr,
                        1,
                        dpm!(ws.logits, &stream),
                        1,
                        hidden as i32,
                        ws.logits.len() as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// One device-path decode step. `want_logits` = dtoh the full row (the gates'
    /// contract); otherwise the greedy token comes back through the device argmax
    /// (4-byte D2H). Exactly one boundary peer copy per crossed stage boundary.
    fn decode_step_fast(
        &self,
        tok: u32,
        state: &mut DecodeState,
        want_logits: bool,
        host_math: bool,
    ) -> Res<(Option<Vec<f32>>, u32)> {
        self.decode_step_fast_tap(tok, state, want_logits, host_math, None)
    }

    /// [`Self::decode_step_fast`] with the iteration-3 DSpark trunk tap: when `taps`
    /// is Some((buffer, base)), the hc-mean of the post-block hc state at each drafter
    /// target layer (40/41/42) is written at buffer[base + k*hidden ..] (concat in
    /// target order, M:917-925) — a pure capture; no kernel computes anything
    /// differently.
    fn decode_step_fast_tap(
        &self,
        tok: u32,
        state: &mut DecodeState,
        want_logits: bool,
        host_math: bool,
        mut taps: Option<(&mut CudaSlice<f32>, usize)>,
    ) -> Res<(Option<Vec<f32>>, u32)> {
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let pos = state.pos;
        assert!(pos > 0, "decode_step needs prefill_with_cache first");
        assert!(
            pos < state.capacity,
            "pos {pos} >= session capacity {}",
            state.capacity
        );
        let hidden = mc.n_embd as usize;
        let hc = d.hc_mult as usize;
        let n_trunk = mc.n_layer - mc.nextn_predict_layers;
        let ws_all = state.ws.as_mut().expect("device path needs StepWs");

        // stage 0: token -> embed -> hc state
        let st0 = &self.stages[0];
        st0.gpu.ctx.bind_to_thread().map_err(e("bind ctx0"))?;
        let stream0 = st0.gpu.stream();
        {
            let ws0 = &mut ws_all[0];
            stream0
                .memcpy_htod(&[tok as i32], &mut ws0.tok)
                .map_err(e("htod tok"))?;
            unsafe {
                ck(
                    "embed_rows dev",
                    k::memra_dsv4_embed_rows(
                        st0.embed
                            .as_ref()
                            .expect("embed on stage 0")
                            .device_ptr(&stream0)
                            .0 as *const c_void,
                        ws0.tok.device_ptr(&stream0).0 as *const i32,
                        dpm!(ws0.emb, &stream0),
                        1,
                        hidden as i32,
                        sp(&stream0),
                    ),
                )?;
                ck(
                    "repeat_hc dev",
                    k::memra_dsv4_repeat_hc(
                        dpf!(ws0.emb, &stream0),
                        dpm!(ws0.h_a, &stream0),
                        1,
                        hc as i32,
                        hidden as i32,
                        sp(&stream0),
                    ),
                )?;
            }
        }

        let mut cur_stage = 0usize;
        let mut input_rx = false;
        for il in 0..n_trunk {
            let stage = self.layer_stage[il as usize];
            if stage != cur_stage {
                // boundary: peer-copy h (TX stream) + event; tok for the hash layers
                // never crosses (they live on stage 0)
                let bytes = hc * hidden * std::mem::size_of::<f32>();
                let src_stream = self.stages[cur_stage].gpu.stream();
                let dst_stream = self.stages[stage].gpu.stream();
                let (ws_src, ws_dst) = ws_all.split_at_mut(stage);
                let src_ws = &ws_src[cur_stage];
                let dst_ws = &mut ws_dst[0];
                self.stages[cur_stage]
                    .gpu
                    .ctx
                    .bind_to_thread()
                    .map_err(e("bind tx"))?;
                let (sp_, _g0) = src_ws.h_a.device_ptr(&src_stream);
                let (dp_, _g1) = dst_ws.h_rx.device_ptr_mut(&src_stream);
                unsafe {
                    cudarc::driver::result::memcpy_peer_async(
                        self.stages[stage].gpu.ctx.cu_ctx(),
                        dp_,
                        self.stages[cur_stage].gpu.ctx.cu_ctx(),
                        sp_,
                        bytes,
                        src_stream.cu_stream(),
                    )
                    .map_err(e("peer copy h"))?;
                }
                let bnd = stage - 1;
                self.boundary_ev[bnd]
                    .record(&src_stream)
                    .map_err(e("ev record"))?;
                dst_stream
                    .wait(&self.boundary_ev[bnd])
                    .map_err(e("ev wait"))?;
                self.stages[stage]
                    .gpu
                    .ctx
                    .bind_to_thread()
                    .map_err(e("bind rx"))?;
                cur_stage = stage;
                input_rx = true;
            }
            let st = &self.stages[stage];
            let lidx = st
                .layers
                .iter()
                .position(|l| l.il == il)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage}"));
            self.block_decode_dev(
                st,
                &st.layers[lidx],
                &mut state.caches[il as usize],
                &mut ws_all[stage],
                input_rx,
                pos,
                tok,
                host_math,
            )?;
            input_rx = false;
            // iteration-3 DSpark tap (capture-only): hc-mean of this layer's output
            // hc state into the tap row at the target's concat offset.
            if let Some((t, base)) = taps.as_mut()
                && let Some(ds) = &self.dspark
                && let Some(k) = ds.targets.iter().position(|&tl| tl == il as usize)
            {
                let stream = self.stages[stage].gpu.stream();
                let hidden_i = hidden as i32;
                unsafe {
                    ck(
                        "hc_mean tap dev",
                        k::memra_dsv4_hc_mean(
                            dpf!(ws_all[stage].h_a, &stream),
                            (t.device_ptr_mut(&stream).0 as usize + (*base + k * hidden) * 4)
                                as *mut f32,
                            1,
                            hc as i32,
                            hidden_i,
                            sp(&stream),
                        ),
                    )?;
                }
            }
        }

        let last = self.stages.len() - 1;
        assert_eq!(cur_stage, last, "device path expects the head stage last");
        self.head_logits_dev(&mut ws_all[last], host_math)?;
        let stream_last = self.stages[last].gpu.stream();
        state.pos += 1;
        if want_logits {
            let logits = dtoh_f32(&stream_last, &ws_all[last].logits)?;
            let mut best = 0usize;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            Ok((Some(logits), best as u32))
        } else {
            unsafe {
                ck(
                    "argmax dev",
                    k::memra_dsv4_argmax(
                        dpf!(ws_all[last].logits, &stream_last),
                        ws_all[last].logits.len() as i64,
                        ws_all[last].argmax.device_ptr_mut(&stream_last).0 as *mut i32,
                        sp(&stream_last),
                    ),
                )?;
            }
            let mut out = [0i32; 1];
            stream_last
                .memcpy_dtoh(&ws_all[last].argmax, &mut out[..])
                .map_err(e("dtoh argmax"))?;
            stream_last.synchronize().map_err(e("sync argmax"))?;
            Ok((None, out[0] as u32))
        }
    }

    /// Greedy decode step (bench serving shape): returns ONLY the next token; on the
    /// device path the argmax runs on-device and 4 bytes cross back. Legacy path
    /// falls back to the full-logits step + host argmax (same value by the argmax
    /// tie-rule equivalence).
    pub fn decode_step_greedy(&self, tok: u32, state: &mut DecodeState) -> Res<u32> {
        match self.decode_path {
            DecodePath::Legacy => {
                let logits = self.decode_step_impl(tok, state, None)?;
                let mut best = 0usize;
                for i in 1..logits.len() {
                    if logits[i] > logits[best] {
                        best = i;
                    }
                }
                Ok(best as u32)
            }
            DecodePath::Device { host_math } => {
                Ok(self.decode_step_fast(tok, state, false, host_math)?.1)
            }
        }
    }

    fn decode_step_impl(
        &self,
        tok: u32,
        state: &mut DecodeState,
        mut dump: Option<&mut Vec<(String, Vec<f32>)>>,
    ) -> Res<Vec<f32>> {
        if let DecodePath::Device { host_math } = self.decode_path {
            assert!(
                dump.is_none(),
                "decode_step_probe is a legacy-path diagnostic (set MEMRA_DSV4_DECODE_PATH=legacy)"
            );
            let (logits, _) = self.decode_step_fast(tok, state, true, host_math)?;
            return Ok(logits.expect("want_logits"));
        }
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let pos = state.pos;
        assert!(pos > 0, "decode_step needs prefill_with_cache first");
        assert!(
            pos < state.capacity,
            "pos {pos} >= session capacity {}",
            state.capacity
        );
        let hidden = mc.n_embd as usize;
        let hc = d.hc_mult as usize;
        let n_trunk = mc.n_layer - mc.nextn_predict_layers;

        // stage 0: embed row -> hc state
        let st0 = &self.stages[0];
        st0.gpu.ctx.bind_to_thread().map_err(e("bind ctx0"))?;
        let stream0 = st0.gpu.stream();
        let ids_dev = upload_i32(&stream0, &[tok as i32])?;
        let mut emb = stream0.alloc_zeros::<f32>(hidden).map_err(e("emb"))?;
        unsafe {
            ck(
                "embed_rows",
                k::memra_dsv4_embed_rows(
                    st0.embed
                        .as_ref()
                        .expect("embed on stage 0")
                        .device_ptr(&stream0)
                        .0 as *const c_void,
                    ids_dev.device_ptr(&stream0).0 as *const i32,
                    dpm!(emb, &stream0),
                    1,
                    hidden as i32,
                    sp(&stream0),
                ),
            )?;
        }
        let mut h = stream0.alloc_zeros::<f32>(hc * hidden).map_err(e("h0"))?;
        unsafe {
            ck(
                "repeat_hc",
                k::memra_dsv4_repeat_hc(
                    dpf!(emb, &stream0),
                    dpm!(h, &stream0),
                    1,
                    hc as i32,
                    hidden as i32,
                    sp(&stream0),
                ),
            )?;
        }

        let mut cur_stage = 0usize;
        for il in 0..n_trunk {
            let stage = self.layer_stage[il as usize];
            if stage != cur_stage {
                let src_stream = self.stages[cur_stage].gpu.stream();
                let host = dtoh_f32(&src_stream, &h)?;
                let dst_stream = self.stages[stage].gpu.stream();
                self.stages[stage]
                    .gpu
                    .ctx
                    .bind_to_thread()
                    .map_err(e("bind"))?;
                h = upload_f32(&dst_stream, &host)?;
                cur_stage = stage;
            }
            let st = &self.stages[stage];
            let lidx = st
                .layers
                .iter()
                .position(|l| l.il == il)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage}"));
            h = self.block_decode(
                st,
                &st.layers[lidx],
                &mut state.caches[il as usize],
                &h,
                pos,
                tok,
                dump.as_deref_mut(),
            )?;
        }

        let last = self.stages.len() - 1;
        if cur_stage != last {
            let src_stream = self.stages[cur_stage].gpu.stream();
            let host = dtoh_f32(&src_stream, &h)?;
            let dst_stream = self.stages[last].gpu.stream();
            h = upload_f32(&dst_stream, &host)?;
        }
        let hc_head_fn = self.stages[last].hc_head_fn.as_ref().expect("hc_head_fn");
        let trunk_norm = self.stages[last].trunk_norm.as_ref().expect("trunk norm");
        let logits = self.head_logits_from(
            &h,
            1,
            hc_head_fn,
            &self.hc_head_base,
            &self.hc_head_scale,
            trunk_norm,
        )?;
        state.pos += 1;
        Ok(logits)
    }
}

// ================================================================ iteration 3: DSpark drafter (device)
//
// Semantic law: DSPARK-SEMANTICS.md (M-cites); numeric truth: the lane-10 CPU oracle
// (memra_gguf::dsv4_dspark) — every gate compares against its fixtures/trajectory.
// Realization: the PREFILL-class helpers (Self::hc_pre host-Sinkhorn, cuBLASLt bf16
// gemm, moe_forward bf16-dequant experts, prefill sink_attn) at s = block_size —
// the lane-4-gated numeric class; the drafter's arena/native-expert perf rungs are
// banked follow-ups, never correctness requirements.
impl Dsv4Gpu {
    fn dspark(&self) -> &DsparkDev {
        self.dspark
            .as_ref()
            .expect("MEMRA_DSV4_DRAFTER=dspark not loaded")
    }

    /// The drafter's exit-head island dots, HOISTED across the block's rows (weight row
    /// read once instead of once per row) with the rung-4c arm selection. The f64 branch
    /// is BIT-EXACT vs the pinned `Self::dots` — identical per-(t, j) element order and
    /// reduction tree — so the default arm's bytes are unchanged by the hoist; the f32x
    /// branch is the measured fork (`MEMRA_DSV4_DSPARK_HEAD_ARM=f32x`) offered for owner
    /// ratification. Either way this touches only WHICH tokens are drafted: verification
    /// always emits the trunk's own argmax, so the emitted stream cannot depend on it.
    #[allow(clippy::too_many_arguments)]
    fn dspark_head_dots(
        &self,
        st: &Stage,
        x: *const f32,
        w: *const c_void,
        w_is_bf16: i32,
        s: usize,
        kdim: usize,
        n: usize,
        y: *mut f32,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        unsafe {
            if self.dspark_head_f32 {
                ck(
                    "dspark head dots f32acc_mrow",
                    k::memra_dsv4_dots_f32acc_mrow(
                        x,
                        w,
                        w_is_bf16,
                        y,
                        s as i32,
                        kdim as i32,
                        n as i32,
                        sp(&stream),
                    ),
                )
            } else {
                ck(
                    "dspark head dots f32_mrow",
                    k::memra_dsv4_dots_f32_mrow(
                        x,
                        w,
                        w_is_bf16,
                        y,
                        s as i32,
                        kdim as i32,
                        n as i32,
                        sp(&stream),
                    ),
                )
            }
        }
    }

    /// Allocate the drafter decode state on the last stage: 3 rings [win + block, hd]
    /// (ring + transient draft rows, struct doc) + the tap rows [block+1, n_t*hidden].
    pub fn dspark_alloc_state(&self) -> Res<DsparkState> {
        let ds = self.dspark();
        let d = self.model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let hidden = self.model.mc.n_embd as usize;
        let last = self.stages.len() - 1;
        let stream = self.stages[last].gpu.stream();
        let mut rings = Vec::with_capacity(ds.blocks.len());
        for _ in 0..ds.blocks.len() {
            rings.push(
                stream
                    .alloc_zeros::<f32>((win + ds.block_size) * hd)
                    .map_err(e("dspark ring"))?,
            );
        }
        let taps = stream
            .alloc_zeros::<f32>((ds.block_size + 1) * ds.targets.len() * hidden)
            .map_err(e("dspark taps"))?;
        Ok(DsparkState {
            rings,
            taps,
            tap_head: 0,
        })
    }

    /// Park the persistent DSpark state beside a trunk host snapshot. The drafter is tiny
    /// relative to the trunk weights, but its per-session rings are semantically required:
    /// dropping them would make a restored turn draft from zeros while the trunk remained
    /// correct, a silent performance fork.
    pub fn snapshot_dspark_state(&self, state: &DsparkState) -> Res<Dsv4HostDsparkState> {
        let ds = self.dspark();
        if state.rings.len() != ds.blocks.len() || state.tap_head > ds.block_size {
            return Err("dsv4 dspark state layout/cursor mismatch".into());
        }
        let d = self.model.cfg();
        let win = d.sliding_window as usize;
        let hd = d.head_dim as usize;
        let hidden = self.model.mc.n_embd as usize;
        let tap_elems = ds.targets.len() * hidden;
        let mut off = 0usize;
        let mut rings = Vec::with_capacity(state.rings.len());
        for ring in &state.rings {
            let n = win * hd;
            if n > ring.len() {
                return Err("dsv4 dspark persistent ring exceeds allocation".into());
            }
            rings.push(off..off + n);
            off += n;
        }
        let tap = off..off + tap_elems;
        off += tap_elems;
        let bytes = off
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "dsv4 dspark host-state byte count overflow".to_string())?;
        let mut slab = crate::PinnedHostBuf::new(bytes)
            .map_err(|err| format!("dsv4 dspark pinned host alloc failed: {err}"))?;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        st.gpu
            .ctx
            .bind_to_thread()
            .map_err(e("bind dsv4 dspark snapshot ctx"))?;
        let stream = st.gpu.stream();
        for (src, range) in state.rings.iter().zip(&rings) {
            let host = dsv4_pinned_f32_mut(&mut slab, range.clone());
            stream
                .memcpy_dtoh(&src.slice(0..range.len()), host)
                .map_err(e("dsv4 dspark ring dtoh"))?;
        }
        let tap_src = state
            .taps
            .slice(state.tap_head * tap_elems..(state.tap_head + 1) * tap_elems);
        let tap_host = dsv4_pinned_f32_mut(&mut slab, tap.clone());
        stream
            .memcpy_dtoh(&tap_src, tap_host)
            .map_err(e("dsv4 dspark tap dtoh"))?;
        stream
            .synchronize()
            .map_err(e("dsv4 dspark snapshot sync"))?;
        Ok(Dsv4HostDsparkState {
            slab,
            rings,
            tap,
            bytes,
            block_size: ds.block_size,
        })
    }

    /// Restore a parked DSpark image. The newest tap is normalized to row zero; all other
    /// tap rows and the draft tail are scratch and stay freshly zeroed.
    pub fn restore_dspark_state(&self, host: &Dsv4HostDsparkState) -> Res<DsparkState> {
        let ds = self.dspark();
        if host.block_size != ds.block_size || host.rings.len() != ds.blocks.len() {
            return Err("dsv4 dspark host state/runtime layout mismatch".into());
        }
        let mut state = self.dspark_alloc_state()?;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        st.gpu
            .ctx
            .bind_to_thread()
            .map_err(e("bind dsv4 dspark restore ctx"))?;
        let stream = st.gpu.stream();
        for (dst, range) in state.rings.iter_mut().zip(&host.rings) {
            let src = dsv4_pinned_f32(&host.slab, range.clone());
            if src.len() > dst.len() {
                return Err("dsv4 dspark host ring exceeds runtime allocation".into());
            }
            let mut view = dst.slice_mut(0..src.len());
            stream
                .memcpy_htod(src, &mut view)
                .map_err(e("dsv4 dspark ring htod"))?;
        }
        let tap = dsv4_pinned_f32(&host.slab, host.tap.clone());
        if tap.len() > state.taps.len() {
            return Err("dsv4 dspark host tap exceeds runtime allocation".into());
        }
        let mut tap_dst = state.taps.slice_mut(0..tap.len());
        stream
            .memcpy_htod(tap, &mut tap_dst)
            .map_err(e("dsv4 dspark tap htod"))?;
        stream
            .synchronize()
            .map_err(e("dsv4 dspark restore sync"))?;
        state.tap_head = 0;
        Ok(state)
    }

    /// Consume a non-empty prompt suffix after restoring trunk + DSpark state, returning
    /// logits for the next position. Every suffix token updates both the trunk's compact
    /// caches and the drafter's accepted-position rings exactly like cold sequential prime.
    pub fn dspark_continue_prefix(
        &self,
        suffix: &[u32],
        state: &mut DecodeState,
        dstate: &mut DsparkState,
    ) -> Res<Vec<f32>> {
        if suffix.is_empty() {
            return Err("dsv4 restored continuation needs a non-empty suffix".into());
        }
        if state.pos + suffix.len() > state.capacity {
            return Err(format!(
                "dsv4 restored continuation {} + {} exceeds session capacity {}",
                state.pos,
                suffix.len(),
                state.capacity
            ));
        }
        let mut last_logits = None;
        for (i, &tok) in suffix.iter().enumerate() {
            let pos = state.pos;
            if i + 1 == suffix.len() {
                last_logits = Some(self.decode_step_tap(tok, state, dstate, 0)?);
            } else {
                let _ = self.decode_step_greedy_tap(tok, state, dstate, 0)?;
            }
            self.dspark_write_rings(dstate, 0, pos)?;
        }
        let last = self.stages.len() - 1;
        self.stages[last]
            .gpu
            .stream()
            .synchronize()
            .map_err(e("dsv4 restored continuation sync"))?;
        Ok(last_logits.expect("non-empty suffix has last logits"))
    }

    /// main_x = main_norm(main_proj(main_hidden)) (M:853), s rows on the last stage.
    fn dspark_main_x(&self, main_hidden: &CudaSlice<f32>, s: usize) -> Res<CudaSlice<f32>> {
        let ds = self.dspark();
        let hidden = self.model.mc.n_embd as usize;
        let k = ds.targets.len() * hidden;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        let stream = st.gpu.stream();
        let mut mx = stream.alloc_zeros::<f32>(s * hidden).map_err(e("main_x"))?;
        Self::gemm(st, main_hidden, &ds.main_proj, 0, s, hidden, k, &mut mx)?;
        unsafe {
            ck(
                "rmsnorm main_x",
                k::memra_dsv4_rmsnorm(
                    dpf!(mx, &stream),
                    dpf!(ds.main_norm, &stream),
                    dpm!(mx, &stream),
                    s as i32,
                    hidden as i32,
                    self.model.mc.rms_eps,
                    sp(&stream),
                ),
            )?;
        }
        Ok(mx)
    }

    /// Per-block main_kv rows (M:758-761): kv_norm(wkv(main_x)) + rope(REAL positions)
    /// + group-64 FP8 QAT on the nope dims. Returns [s, hd] on the last stage.
    fn dspark_main_kv(
        &self,
        blk: &LayerDev,
        main_x: &CudaSlice<f32>,
        s: usize,
        positions: &[i32],
    ) -> Res<CudaSlice<f32>> {
        let d = self.model.cfg();
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let hidden = self.model.mc.n_embd as usize;
        let eps = self.model.mc.rms_eps;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        let stream = st.gpu.stream();
        let clamp_only = (self.variant == ActQuantVariant::ClampOnly) as i32;
        // item 3: `.dev()` — the drafter blocks carry no fp8 twins this rung, so
        // their bf16 slabs are always device-resident; if a future rung stages them,
        // this must fail loudly rather than pay a per-round upload silently.
        let mut kv = stream.alloc_zeros::<f32>(s * hd).map_err(e("dspark kv"))?;
        Self::gemm(st, main_x, blk.wkv.dev(), 0, s, hd, hidden, &mut kv)?;
        let pos_dev = upload_i32(&stream, positions)?;
        unsafe {
            ck(
                "rmsnorm dspark kv",
                k::memra_dsv4_rmsnorm(
                    dpf!(kv, &stream),
                    dpf!(blk.kv_norm, &stream),
                    dpm!(kv, &stream),
                    s as i32,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope dspark kv",
                k::memra_dsv4_rope(
                    dpm!(kv, &stream),
                    s as i32,
                    1,
                    hd as i32,
                    rd as i32,
                    dpf!(st.fc_plain, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
            ck(
                "act_quant dspark kv",
                k::memra_dsv4_act_quant(
                    dpm!(kv, &stream),
                    s as i32,
                    hd as i64,
                    (hd - rd) as i32,
                    64,
                    clamp_only,
                    sp(&stream),
                ),
            )?;
        }
        Ok(kv)
    }

    /// Prefill ring priming (M:763-769): last min(s, win) positions land at slot
    /// p % win. `main_hidden` = [s, n_t*hidden] tap rows from the prefill.
    pub fn dspark_prime_prefill(
        &self,
        state: &mut DsparkState,
        main_hidden: &CudaSlice<f32>,
        s: usize,
    ) -> Res<()> {
        let d = self.model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let last = self.stages.len() - 1;
        let stream = self.stages[last].gpu.stream();
        let mx = self.dspark_main_x(main_hidden, s)?;
        let positions: Vec<i32> = (0..s as i32).collect();
        let n_blocks = self.dspark().blocks.len();
        for bi in 0..n_blocks {
            let blk = &self.dspark().blocks[bi];
            let kv = self.dspark_main_kv(blk, &mx, s, &positions)?;
            for p in s.saturating_sub(win)..s {
                let slot = p % win;
                let src = kv.slice(p * hd..(p + 1) * hd);
                let mut dst = state.rings[bi].slice_mut(slot * hd..(slot + 1) * hd);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("prime ring"))?;
            }
        }
        Ok(())
    }

    /// Trunk prefill + DSpark ring prime in ONE pass — the device twin of the CPU
    /// oracle's `trunk.forward(&seq[..p0], 0)` + `dspark.prime_prefill(&pre.main_hidden,
    /// p0)` pair (dsv4_dspark_gate components mode).
    ///
    /// The prefill taps come from the existing `GpuCapture::layer_out` hook (the target
    /// layers' full hc state `[s, hc, hidden]`), then run through the SAME
    /// `memra_dsv4_hc_mean` kernel the decode tap uses — prefill and decode taps must
    /// not be two numeric realizations of one tap — and are placed at the target's
    /// concat stride with `place_cols`, reproducing the oracle's
    /// `main_hidden[(p*n_t + k)*hidden ..]` layout exactly.
    pub fn dspark_prefill_prime(
        &self,
        ids: &[u32],
        state: &mut DecodeState,
        dstate: &mut DsparkState,
    ) -> Res<ForwardOut> {
        assert_eq!(
            state.pos, 0,
            "dspark_prefill_prime needs a fresh DecodeState"
        );
        assert!(!ids.is_empty(), "empty prompt");
        let hidden = self.model.mc.n_embd as usize;
        let hc = self.model.cfg().hc_mult as usize;
        let s = ids.len();
        let targets = self.dspark().targets.clone();
        let n_t = targets.len();
        let mut cap = GpuCapture {
            want: targets.iter().map(|&t| t as u32).collect(),
            ..Default::default()
        };
        let out = self
            .forward_impl(ids, Some(&mut cap), None, Some(state))?
            .expect("prefill logits");
        state.pos = s;

        let last = self.stages.len() - 1;
        let stream = self.stages[last].gpu.stream();
        self.stages[last]
            .gpu
            .ctx
            .bind_to_thread()
            .map_err(e("bind ctx prime"))?;
        let mut main_hidden = stream
            .alloc_zeros::<f32>(s * n_t * hidden)
            .map_err(e("prefill main_hidden"))?;
        let mut tmp = stream
            .alloc_zeros::<f32>(s * hidden)
            .map_err(e("tap tmp"))?;
        for (k, &il) in targets.iter().enumerate() {
            let h = cap
                .layer_out
                .get(&(il as u32))
                .unwrap_or_else(|| panic!("prefill capture missing target layer {il}"));
            assert_eq!(
                h.len(),
                s * hc * hidden,
                "target layer {il} capture is not [s, hc, hidden]"
            );
            let h_dev = upload_f32(&stream, h)?;
            unsafe {
                ck(
                    "hc_mean prefill tap",
                    k::memra_dsv4_hc_mean(
                        dpf!(h_dev, &stream),
                        dpm!(tmp, &stream),
                        s as i32,
                        hc as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "place_cols prefill tap",
                    k::memra_dsv4_place_cols(
                        dpf!(tmp, &stream),
                        dpm!(main_hidden, &stream),
                        s as i32,
                        hidden as i32,
                        (n_t * hidden) as i64,
                        (k * hidden) as i64,
                        sp(&stream),
                    ),
                )?;
            }
        }
        self.dspark_prime_prefill(dstate, &main_hidden, s)?;
        // Seed taps row 0 with the LAST prefill position's tap: the generic spec loop's
        // first proposal is `propose(t, mh_last, p0-1)` with mh_last = pre_taps row
        // p0-1 (spec_oracle::run_spec_greedy) — without this the first round would draft
        // off a zeroed tap.
        {
            let src = main_hidden.slice((s - 1) * n_t * hidden..s * n_t * hidden);
            let mut dst = dstate.taps.slice_mut(0..n_t * hidden);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(e("seed tap row"))?;
        }
        dstate.tap_head = 0;
        stream.synchronize().map_err(e("prime sync"))?;
        Ok(out)
    }

    /// Bounded-memory trunk prefill plus DSpark prime. The first token uses the canonical
    /// prefill path; every later chunk uses the same batched trunk transaction as
    /// speculative verification, commits all rows, and advances the drafter rings from
    /// the captured target-layer taps at their absolute positions.
    pub fn dspark_prefill_prime_chunked(
        &self,
        ids: &[u32],
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        chunk: usize,
    ) -> Res<Vec<f32>> {
        if ids.is_empty() {
            return Err("empty dsv4 DSpark chunked prefill".into());
        }
        if chunk == 0 || chunk > DSV4_BATCH_WIDTH_MAX || chunk > state.transient_rows {
            return Err(format!(
                "dsv4 DSpark prefill chunk {chunk} outside 1..={} allocated transient rows (kernel max {DSV4_BATCH_WIDTH_MAX})",
                state.transient_rows
            ));
        }
        if ids.len() > state.capacity {
            return Err(format!(
                "dsv4 DSpark prefill {} tokens exceeds session cache capacity {}",
                ids.len(),
                state.capacity
            ));
        }
        let first = self.dspark_prefill_prime(&ids[..1], state, dstate)?;
        if ids.len() == 1 {
            return Ok(first.logits);
        }
        self.dspark_continue_prefix_chunked(&ids[1..], state, dstate, chunk)
    }

    /// Chunked teacher-forced suffix twin for a restored trunk + DSpark state.
    pub fn dspark_continue_prefix_chunked(
        &self,
        suffix: &[u32],
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        chunk: usize,
    ) -> Res<Vec<f32>> {
        if suffix.is_empty() {
            return Err("dsv4 DSpark chunked continuation needs a non-empty suffix".into());
        }
        if chunk == 0 || chunk > DSV4_BATCH_WIDTH_MAX || chunk > state.transient_rows {
            return Err(format!(
                "dsv4 DSpark continuation chunk {chunk} outside 1..={} allocated transient rows",
                state.transient_rows
            ));
        }
        if state.pos == 0 || state.pos + suffix.len() > state.capacity {
            return Err(format!(
                "dsv4 DSpark chunked continuation {} + {} outside primed capacity {}",
                state.pos,
                suffix.len(),
                state.capacity
            ));
        }
        let width = chunk.min(suffix.len());
        let mut vstate = self.alloc_prefill_state_for(state.capacity, width)?;
        let last = self.stages.len() - 1;
        let stream = self.stages[last].gpu.stream();
        let hidden = self.model.mc.n_embd as usize;
        let n_t = self.dspark().targets.len();
        let mut taps = stream
            .alloc_zeros::<f32>(width * n_t * hidden)
            .map_err(e("dsv4 chunk taps"))?;
        let mut tap_row = stream
            .alloc_zeros::<f32>(n_t * hidden)
            .map_err(e("dsv4 chunk tap row"))?;
        let keep_from = self
            .prefill_draft
            .keep_from(suffix.len(), self.model.cfg().sliding_window as usize);
        let mut last_logits = None;
        for (i, toks) in suffix.chunks(width).enumerate() {
            let pos0 = state.pos;
            let final_chunk = (i + 1) * width >= suffix.len();
            let output = self.prefill_head.output(final_chunk);
            let first_kept = keep_from.saturating_sub(i * width).min(toks.len());
            let capture_taps = if first_kept < toks.len() {
                Some(&mut taps)
            } else {
                None
            };
            let (logits, _) =
                self.verify_batch_dev_output(toks, state, &mut vstate, capture_taps, output)?;
            self.commit_verify_dev(state, &mut vstate, toks.len())?;
            // The existing drafter prime projections use a general GEMM whose numeric
            // realization changes with m. Keep their canonical m=1 realization until a
            // separately exact-gated batched DSpark-prime twin lands; trunk chunking still
            // amortizes the 43-layer target path.
            // DSpark has SWA-only rings: older suffix rows are overwritten before
            // this method returns. Short suffixes retain every row and preserve
            // the still-live ring slots restored from the previous prefix.
            for j in first_kept..toks.len() {
                let tap_elems = n_t * hidden;
                let src = taps.slice(j * tap_elems..(j + 1) * tap_elems);
                let mut dst = tap_row.slice_mut(0..tap_elems);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("dsv4 chunk tap row copy"))?;
                self.dspark_commit_prefill_taps(dstate, &tap_row, 1, pos0 + j)?;
                self.prefill_head_counts
                    .draft_prime_rows
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(rows) = logits {
                last_logits = Some(if output == VerifyOutput::Last {
                    rows
                } else {
                    let vocab = rows.len() / toks.len();
                    rows[(toks.len() - 1) * vocab..].to_vec()
                });
            }
        }
        Ok(last_logits.expect("final DSpark chunk requested logits"))
    }

    /// Advance persistent drafter rings from target-layer mean taps for one fully
    /// committed prefill chunk, then normalize the newest tap to row zero for proposal.
    fn dspark_commit_prefill_taps(
        &self,
        state: &mut DsparkState,
        main_hidden: &CudaSlice<f32>,
        s: usize,
        pos0: usize,
    ) -> Res<()> {
        let d = self.model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let hidden = self.model.mc.n_embd as usize;
        let n_t = self.dspark().targets.len();
        let last = self.stages.len() - 1;
        let stream = self.stages[last].gpu.stream();
        let mx = self.dspark_main_x(main_hidden, s)?;
        let positions: Vec<i32> = (0..s).map(|i| (pos0 + i) as i32).collect();
        for (bi, blk) in self.dspark().blocks.iter().enumerate() {
            let kv = self.dspark_main_kv(blk, &mx, s, &positions)?;
            for i in 0..s {
                let slot = (pos0 + i) % win;
                let src = kv.slice(i * hd..(i + 1) * hd);
                let mut dst = state.rings[bi].slice_mut(slot * hd..(slot + 1) * hd);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("dsv4 chunk prime ring"))?;
            }
        }
        let tap_elems = n_t * hidden;
        let src = main_hidden.slice((s - 1) * tap_elems..s * tap_elems);
        let mut dst = state.taps.slice_mut(0..tap_elems);
        stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(e("dsv4 chunk seed tap"))?;
        state.tap_head = 0;
        stream.synchronize().map_err(e("dsv4 chunk prime sync"))?;
        Ok(())
    }

    /// Ring advance for ONE committed position (§3.1 drafter rule: accepted positions
    /// only). `tap_row` indexes into `state.taps` (the row that holds position `pos`'s
    /// hc-mean concat).
    pub fn dspark_write_rings(
        &self,
        state: &mut DsparkState,
        tap_row: usize,
        pos: usize,
    ) -> Res<()> {
        let d = self.model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let hidden = self.model.mc.n_embd as usize;
        let n_t = self.dspark().targets.len();
        let last = self.stages.len() - 1;
        let stream = self.stages[last].gpu.stream();
        let tap = {
            // one-row view as an owned slice copy (gemm wants a base slice)
            let mut row = stream
                .alloc_zeros::<f32>(n_t * hidden)
                .map_err(e("tap row"))?;
            let src = state
                .taps
                .slice(tap_row * n_t * hidden..(tap_row + 1) * n_t * hidden);
            stream.memcpy_dtod(&src, &mut row).map_err(e("tap copy"))?;
            row
        };
        let mx = self.dspark_main_x(&tap, 1)?;
        let n_blocks = self.dspark().blocks.len();
        for bi in 0..n_blocks {
            let blk = &self.dspark().blocks[bi];
            let kv = self.dspark_main_kv(blk, &mx, 1, &[pos as i32])?;
            let slot = pos % win;
            let src = kv.slice(0..hd);
            let mut dst = state.rings[bi].slice_mut(slot * hd..(slot + 1) * hd);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(e("ring write"))?;
        }
        state.tap_head = tap_row;
        Ok(())
    }

    /// One DSpark draft-block forward (M:695-707 body with DSparkAttention M:771-792):
    /// h [block, hc, hidden] -> same shape. Reads the ring; writes ONLY the transient
    /// draft-kv rows [win, win+block) of `ring` (never persistent ring slots).
    #[allow(clippy::too_many_arguments)]
    fn dspark_block_forward(
        &self,
        blk: &LayerDev,
        ring: &mut CudaSlice<f32>,
        h: &CudaSlice<f32>,
        block: usize,
        pos: usize,
    ) -> Res<CudaSlice<f32>> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let heads = mc.n_head as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let eps = mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        st.gpu.ctx.bind_to_thread().map_err(e("bind ctx dspark"))?;
        let stream = st.gpu.stream();
        let clamp_only = (self.variant == ActQuantVariant::ClampOnly) as i32;
        // draft positions pos+1 .. pos+block (M:772)
        let positions: Vec<i32> = (1..=block as i32).map(|j| pos as i32 + j).collect();
        let pos_dev = upload_i32(&stream, &positions)?;

        // ---- attention sub-block
        let (y, post, comb) = Self::hc_pre(
            st,
            h,
            &blk.hc_attn_fn,
            &blk.hc_attn_base,
            &blk.hc_attn_scale,
            block,
            hc,
            hidden,
            iters,
            hc_eps,
        )?;
        let mut x = stream.alloc_zeros::<f32>(block * hidden).map_err(e("x"))?;
        unsafe {
            ck(
                "rmsnorm dspark attn",
                k::memra_dsv4_rmsnorm(
                    dpf!(y, &stream),
                    dpf!(blk.attn_norm, &stream),
                    dpm!(x, &stream),
                    block as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        // q path (trunk-identical, M:774-777)
        let mut qr = stream.alloc_zeros::<f32>(block * q_lora).map_err(e("qr"))?;
        Self::gemm(st, &x, blk.wq_a.dev(), 0, block, q_lora, hidden, &mut qr)?;
        unsafe {
            ck(
                "rmsnorm dspark q",
                k::memra_dsv4_rmsnorm(
                    dpf!(qr, &stream),
                    dpf!(blk.q_norm, &stream),
                    dpm!(qr, &stream),
                    block as i32,
                    q_lora as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        let mut qr_b = stream
            .alloc_zeros::<u8>(block * q_lora * 2)
            .map_err(e("qr_b"))?;
        unsafe {
            ck(
                "cvt dspark qr",
                k::memra_dsv4_cvt_bf16(
                    dpf!(qr, &stream),
                    qr_b.device_ptr_mut(&stream).0 as *mut c_void,
                    (block * q_lora) as i64,
                    sp(&stream),
                ),
            )?;
        }
        let mut q = stream
            .alloc_zeros::<f32>(block * heads * hd)
            .map_err(e("q"))?;
        Self::gemm_pre(
            st,
            &qr_b,
            blk.wq_b.dev().device_ptr(&stream).0 as *const c_void,
            block,
            heads * hd,
            q_lora,
            &mut q,
        )?;
        unsafe {
            ck(
                "headrms dspark",
                k::memra_dsv4_headrms(
                    dpm!(q, &stream),
                    (block * heads) as i32,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope dspark q",
                k::memra_dsv4_rope(
                    dpm!(q, &stream),
                    block as i32,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    dpf!(st.fc_plain, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
        }
        // draft kv (M:778-780) -> transient rows [win, win+block) of the ring buffer
        {
            let mut kv = stream.alloc_zeros::<f32>(block * hd).map_err(e("dkv"))?;
            Self::gemm(st, &x, blk.wkv.dev(), 0, block, hd, hidden, &mut kv)?;
            unsafe {
                ck(
                    "rmsnorm dspark dkv",
                    k::memra_dsv4_rmsnorm(
                        dpf!(kv, &stream),
                        dpf!(blk.kv_norm, &stream),
                        dpm!(kv, &stream),
                        block as i32,
                        hd as i32,
                        eps,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "rope dspark dkv",
                    k::memra_dsv4_rope(
                        dpm!(kv, &stream),
                        block as i32,
                        1,
                        hd as i32,
                        rd as i32,
                        dpf!(st.fc_plain, &stream),
                        pos_dev.device_ptr(&stream).0 as *const i32,
                        0,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "act_quant dspark dkv",
                    k::memra_dsv4_act_quant(
                        dpm!(kv, &stream),
                        block as i32,
                        hd as i64,
                        (hd - rd) as i32,
                        64,
                        clamp_only,
                        sp(&stream),
                    ),
                )?;
            }
            let src = kv.slice(0..block * hd);
            let mut dst = ring.slice_mut(win * hd..(win + block) * hd);
            stream.memcpy_dtod(&src, &mut dst).map_err(e("draft kv"))?;
        }
        // attention set (M:743-747): ring slots 0..min(win, pos+1) then the block's
        // transient rows — ONE shared row for every draft query (bidirectional
        // intra-block attention), replicated per query for the prefill kernel.
        let n_ring = win.min(pos + 1);
        let mut idx_row: Vec<i32> = (0..n_ring as i32).collect();
        idx_row.extend((0..block as i32).map(|j| win as i32 + j));
        let slots = idx_row.len();
        let mut idxs = Vec::with_capacity(block * slots);
        for _ in 0..block {
            idxs.extend_from_slice(&idx_row);
        }
        let idx_dev = upload_i32(&stream, &idxs)?;
        let mut o = stream
            .alloc_zeros::<f32>(block * heads * hd)
            .map_err(e("o"))?;
        let scale = (hd as f64).powf(-0.5) as f32;
        unsafe {
            ck(
                "sink_attn dspark",
                k::memra_dsv4_sink_attn(
                    dpf!(q, &stream),
                    dpf!(ring, &stream),
                    idx_dev.device_ptr(&stream).0 as *const i32,
                    dpf!(blk.sink, &stream),
                    dpm!(o, &stream),
                    block as i32,
                    heads as i32,
                    hd as i32,
                    slots as i32,
                    scale,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope dspark o inv",
                k::memra_dsv4_rope(
                    dpm!(o, &stream),
                    block as i32,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    dpf!(st.fc_plain, &stream),
                    pos_dev.device_ptr(&stream).0 as *const i32,
                    1,
                    sp(&stream),
                ),
            )?;
        }
        // grouped wo (trunk-identical)
        let gw = heads / o_groups * hd;
        let mut og = stream
            .alloc_zeros::<f32>(block * o_groups * o_lora)
            .map_err(e("og"))?;
        let mut o_grp = stream.alloc_zeros::<f32>(block * gw).map_err(e("o_grp"))?;
        let mut y_grp = stream
            .alloc_zeros::<f32>(block * o_lora)
            .map_err(e("y_grp"))?;
        for g in 0..o_groups {
            unsafe {
                ck(
                    "take_cols dspark",
                    k::memra_dsv4_take_cols(
                        dpf!(o, &stream),
                        dpm!(o_grp, &stream),
                        block as i32,
                        gw as i32,
                        (heads * hd) as i64,
                        (g * gw) as i64,
                        sp(&stream),
                    ),
                )?;
            }
            Self::gemm(
                st,
                &o_grp,
                blk.wo_a.dev(),
                g * o_lora * gw,
                block,
                o_lora,
                gw,
                &mut y_grp,
            )?;
            unsafe {
                ck(
                    "place_cols dspark",
                    k::memra_dsv4_place_cols(
                        dpf!(y_grp, &stream),
                        dpm!(og, &stream),
                        block as i32,
                        o_lora as i32,
                        (o_groups * o_lora) as i64,
                        (g * o_lora) as i64,
                        sp(&stream),
                    ),
                )?;
            }
        }
        let mut attn_out = stream.alloc_zeros::<f32>(block * hidden).map_err(e("ao"))?;
        Self::gemm(
            st,
            &og,
            blk.wo_b.dev(),
            0,
            block,
            hidden,
            o_groups * o_lora,
            &mut attn_out,
        )?;
        let mut h2 = stream
            .alloc_zeros::<f32>(block * hc * hidden)
            .map_err(e("h2"))?;
        unsafe {
            ck(
                "hc_post dspark attn",
                k::memra_dsv4_hc_post(
                    dpf!(attn_out, &stream),
                    dpf!(h, &stream),
                    dpf!(post, &stream),
                    dpf!(comb, &stream),
                    dpm!(h2, &stream),
                    block as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        // ---- ffn sub-block (score-routed MoE; ids unused by a non-hash gate)
        let (y2, post2, comb2) = Self::hc_pre(
            st,
            &h2,
            &blk.hc_ffn_fn,
            &blk.hc_ffn_base,
            &blk.hc_ffn_scale,
            block,
            hc,
            hidden,
            iters,
            hc_eps,
        )?;
        let mut xf = stream.alloc_zeros::<f32>(block * hidden).map_err(e("xf"))?;
        unsafe {
            ck(
                "rmsnorm dspark ffn",
                k::memra_dsv4_rmsnorm(
                    dpf!(y2, &stream),
                    dpf!(blk.ffn_norm, &stream),
                    dpm!(xf, &stream),
                    block as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        let ids = vec![0u32; block];
        let moe_out = self.moe_forward(st, blk, &xf, block, &ids)?;
        let mut h3 = stream
            .alloc_zeros::<f32>(block * hc * hidden)
            .map_err(e("h3"))?;
        unsafe {
            ck(
                "hc_post dspark ffn",
                k::memra_dsv4_hc_post(
                    dpf!(moe_out, &stream),
                    dpf!(h2, &stream),
                    dpf!(post2, &stream),
                    dpf!(comb2, &stream),
                    dpm!(h3, &stream),
                    block as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        Ok(h3)
    }

    /// forward_spec (M:928-936) + forward_head (M:860-874) on the device: ONE parallel
    /// noise-block draft through the 3 blocks, shared trunk head over all block rows,
    /// sequential rank-256 markov chaining (greedy), fp32 confidence. Mutates ONLY the
    /// rings' transient rows (drafting is side-effect-free on trunk + persistent ring
    /// state — §3.1). `tap_row` = the taps row holding position `pos`'s hc-mean concat.
    pub fn dspark_forward_spec(
        &self,
        state: &mut DsparkState,
        input_token: u32,
        tap_row: usize,
        pos: usize,
        capture: bool,
    ) -> Res<DsparkProposal> {
        self.dspark_forward_spec_inner(state, input_token, tap_row, pos, capture, None)
    }

    fn dspark_forward_spec_inner(
        &self,
        state: &mut DsparkState,
        input_token: u32,
        tap_row: usize,
        pos: usize,
        capture: bool,
        sample: Option<&Dsv4SampleCfg>,
    ) -> Res<DsparkProposal> {
        let ds = self.dspark();
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let eps = mc.rms_eps;
        let block = ds.block_size;
        let rank = ds.rank;
        let vocab = ds.vocab;
        let n_t = ds.targets.len();
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        st.gpu.ctx.bind_to_thread().map_err(e("bind ctx spec"))?;
        let stream = st.gpu.stream();

        let prof = if dsv4_prof_on() {
            Some(stream.clone())
        } else {
            None
        };
        // main_x from the tap row (computed once per call, M:930-932)
        let tap = {
            let _p = phase!("1a.tap_copy", prof.as_ref());
            let mut row = stream.alloc_zeros::<f32>(n_t * hidden).map_err(e("tapr"))?;
            let src = state
                .taps
                .slice(tap_row * n_t * hidden..(tap_row + 1) * n_t * hidden);
            stream.memcpy_dtod(&src, &mut row).map_err(e("tap cp"))?;
            row
        };
        let mx = {
            let _p = phase!("1b.main_x", prof.as_ref());
            self.dspark_main_x(&tap, 1)?
        };
        let (cap_main_hidden, cap_main_x) = if capture {
            (
                Some(dtoh_f32(&stream, &tap)?),
                Some(dtoh_f32(&stream, &mx)?),
            )
        } else {
            (None, None)
        };
        // draft block ids: [input token, noise ×(block-1)] via the SHARED trunk embed
        // (host-gathered — the MtpDev precedent)
        let _p_embed = phase!("1c.embed_h2d_repeat", prof.as_ref());
        let mut draft_ids = vec![ds.noise_token; block];
        draft_ids[0] = input_token;
        let e_rows = self.model.embed_rows(&draft_ids);
        let e_dev = upload_f32(&stream, &e_rows)?;
        let mut h = stream
            .alloc_zeros::<f32>(block * hc * hidden)
            .map_err(e("h0"))?;
        unsafe {
            ck(
                "repeat_hc dspark",
                k::memra_dsv4_repeat_hc(
                    dpf!(e_dev, &stream),
                    dpm!(h, &stream),
                    block as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        drop(_p_embed);
        let mut block_outs: Vec<Vec<f32>> = Vec::new();
        let _p_blocks = phase!("1d.drafter_blocks", prof.as_ref());
        let n_blocks = ds.blocks.len();
        for bi in 0..n_blocks {
            // rings[bi] transient rows are rewritten; persistent slots untouched
            let mut ring = std::mem::replace(
                &mut state.rings[bi],
                stream.alloc_zeros::<f32>(0).map_err(e("swap"))?,
            );
            let out =
                self.dspark_block_forward(&self.dspark().blocks[bi], &mut ring, &h, block, pos);
            state.rings[bi] = ring;
            h = out?;
            if capture {
                block_outs.push(dtoh_f32(&stream, &h)?);
            }
        }
        drop(_p_blocks);
        // exit head (mtp.2): pre-only hc collapse -> xc (pre-norm, feeds confidence),
        // norm, shared trunk head over ALL block rows
        let w = hc * hidden;
        let _p_mix = phase!("1e.exit_mix_dots", prof.as_ref());
        let mut mixes = stream.alloc_zeros::<f32>(block * hc).map_err(e("mx"))?;
        // hoisted (weight row read once across the block's rows). The f64 twin is
        // BIT-EXACT vs `Self::dots` — same per-(t, j) element order and reduction tree —
        // so the default arm's values are untouched by the hoist.
        self.dspark_head_dots(
            st,
            h.device_ptr(&stream).0 as *const f32,
            ds.hc_head_fn.device_ptr(&stream).0 as *const c_void,
            0,
            block,
            w,
            hc,
            mixes.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "rowsq dspark head",
                k::memra_dsv4_rowsq_scale(
                    dpf!(h, &stream),
                    dpm!(mixes, &stream),
                    block as i32,
                    w as i32,
                    hc as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        drop(_p_mix);
        let _p_mixrt = phase!("1f.mix_D2H_host_H2D", prof.as_ref());
        let mut mixes_h = dtoh_f32(&stream, &mixes)?;
        for t in 0..block {
            for c in 0..hc {
                let m = mixes_h[t * hc + c];
                mixes_h[t * hc + c] =
                    sigmoid_f32(m * ds.hc_head_scale[0] + ds.hc_head_base[c]) + d.hc_eps;
            }
        }
        let pre_d = upload_f32(&stream, &mixes_h)?;
        drop(_p_mixrt);
        let _p_cn = phase!("1g.collapse_norm", prof.as_ref());
        let mut xc = stream.alloc_zeros::<f32>(block * hidden).map_err(e("xc"))?;
        unsafe {
            ck(
                "hc_collapse dspark",
                k::memra_dsv4_hc_collapse(
                    dpf!(h, &stream),
                    dpf!(pre_d, &stream),
                    dpm!(xc, &stream),
                    block as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        let mut normed = stream.alloc_zeros::<f32>(block * hidden).map_err(e("nr"))?;
        unsafe {
            ck(
                "rmsnorm dspark head",
                k::memra_dsv4_rmsnorm(
                    dpf!(xc, &stream),
                    dpf!(ds.norm, &stream),
                    dpm!(normed, &stream),
                    block as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        drop(_p_cn);
        let _p_head = phase!("1h.exit_head_dots", prof.as_ref());
        let mut logits = stream.alloc_zeros::<f32>(block * vocab).map_err(e("lg"))?;
        // THE 21%-of-a-round instance (nsys, rung 4c): vocab x block over the 1.06 GiB
        // shared head. f64 default (gated bytes, hoisted bit-exactly);
        // MEMRA_DSV4_DSPARK_HEAD_ARM=f32x switches it to the ratified accumulation class.
        self.dspark_head_dots(
            st,
            normed.device_ptr(&stream).0 as *const f32,
            st.head.as_ref().expect("head").device_ptr(&stream).0 as *const c_void,
            1,
            block,
            hidden,
            vocab,
            logits.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        drop(_p_head);
        // pre-markov head logits (the gate's logits_pre array) — captured BEFORE the
        // chaining loop adds any bias row in place.
        let cap_logits_pre = if capture {
            Some(dtoh_f32(&stream, &logits)?)
        } else {
            None
        };
        // Sequential Markov chaining. The public path remains greedy; sampled
        // serving may explicitly use the request's coupled position-keyed draw.
        //
        // ITERATION-5 (F itemisation, rung 2): the chain is inherently sequential -- draft i+1's
        // markov row is indexed by draft i -- but the DEPENDENCY never needed a HOST round trip.
        // The shipped loop reads each argmax back (4 B D2H + `stream.synchronize()`) and each
        // confidence back the same way, so a block_size-5 chain DRAINS the only stream TEN times
        // per round. Those drains are pure F: T-independent, all latency, no work.
        // `MEMRA_DSV4_DSPARK_CHAIN=device` keeps the chain resident -- the argmax lands in
        // `am_dev[i + 1]`, the next markov row is gathered BY DEVICE INDEX, confidences
        // accumulate into `conf_out[i]`, and ONE D2H at the end of the loop returns every id and
        // confidence together. Same kernels, same operands, same reduction order: the arm is
        // bit-identical BY CONSTRUCTION rather than by tolerance, because only transport moved.
        // The coupled experiment uses the existing host categorical sampler.
        // Keep the next Markov gather on host until a separately gated GPU sampler
        // preserves this exact position-keyed sampling program.
        let chain_device = sample.is_none() && dsv4_dspark_chain_device();
        let markov_rowblk = dsv4_dspark_markov_rowblk();
        let mut w1_row = stream.alloc_zeros::<f32>(rank).map_err(e("w1r"))?;
        let mut bias = stream.alloc_zeros::<f32>(vocab).map_err(e("bias"))?;
        // Slot 0 carries the round's input token so even the FIRST gather is device-indexed and
        // the two arms share one code path.
        let mut am_dev = stream.alloc_zeros::<i32>(block + 1).map_err(e("am"))?;
        {
            let mut dst = am_dev.slice_mut(0..1);
            stream
                .memcpy_htod(&[input_token as i32][..], &mut dst)
                .map_err(e("htod am0"))?;
        }
        let mut out_ids = vec![input_token];
        let mut margins = Vec::with_capacity(block);
        let mut top1_logits = Vec::with_capacity(block);
        let mut conf_in = stream.alloc_zeros::<f32>(hidden + rank).map_err(e("cin"))?;
        let mut conf_out = stream.alloc_zeros::<f32>(block).map_err(e("cout"))?;
        let mut confidence = Vec::with_capacity(block);
        let _p_mk = phase!("1i.markov_chain", prof.as_ref());
        for i in 0..block {
            {
                let _p = phase!("1i1.markov_w1_gather", prof.as_ref());
                if chain_device {
                    unsafe {
                        ck(
                            "markov w1 gather dev",
                            k::memra_dsv4_gather_row_by_idx(
                                dpf!(ds.markov_w1, &stream),
                                am_dev.device_ptr(&stream).0 as *const i32,
                                i as i32,
                                dpm!(w1_row, &stream),
                                rank as i32,
                                sp(&stream),
                            ),
                        )?;
                    }
                } else {
                    let prev = out_ids[i] as usize;
                    let src = ds.markov_w1.slice(prev * rank..(prev + 1) * rank);
                    stream.memcpy_dtod(&src, &mut w1_row).map_err(e("w1 cp"))?;
                }
            }
            {
                let _p = phase!("1i2.markov_bias_gemv", prof.as_ref());
                if markov_rowblk {
                    unsafe {
                        ck(
                            "dots_f32 markov rowblk",
                            k::memra_dsv4_dots_f32_rowblk(
                                dpf!(w1_row, &stream),
                                dp!(ds.markov_w2, &stream),
                                0,
                                dpm!(bias, &stream),
                                1,
                                rank as i32,
                                vocab as i32,
                                sp(&stream),
                            ),
                        )?;
                    }
                } else {
                    Self::dots(st, &w1_row, &ds.markov_w2, 1, rank, vocab, &mut bias)?;
                }
            }
            let _p_aa = phase!("1i3.markov_add_argmax", prof.as_ref());
            unsafe {
                ck(
                    "markov add dspark",
                    k::memra_dsv4_add_inplace(
                        (logits.device_ptr_mut(&stream).0 as usize + i * vocab * 4) as *mut f32,
                        dpf!(bias, &stream),
                        vocab as i64,
                        sp(&stream),
                    ),
                )?;
                if sample.is_none() {
                    ck(
                        "argmax dspark",
                        k::memra_dsv4_argmax(
                            (logits.device_ptr(&stream).0 as usize + i * vocab * 4) as *const f32,
                            vocab as i64,
                            (am_dev.device_ptr_mut(&stream).0 as usize + (i + 1) * 4) as *mut i32,
                            sp(&stream),
                        ),
                    )?;
                }
            }
            drop(_p_aa);
            if let Some(cfg) = sample {
                let _p_draw = phase!("1i4.coupled_sample_D2H_SYNC", None);
                let view = logits.slice(i * vocab..(i + 1) * vocab);
                let mut row = vec![0f32; vocab];
                stream
                    .memcpy_dtoh(&view, &mut row)
                    .map_err(e("coupled draft logits"))?;
                stream
                    .synchronize()
                    .map_err(e("sync coupled draft logits"))?;
                let token = dsv4_sample_row(&row, dspark_draft_position(pos, i), cfg)?;
                out_ids.push(token);
                self.coupled_draft_draws
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else if !chain_device {
                let _p_d2h = phase!("1i4.markov_argmax_D2H_SYNC", None);
                let mut am = [0i32; 1];
                let view = am_dev.slice(i + 1..i + 2);
                stream
                    .memcpy_dtoh(&view, &mut am[..])
                    .map_err(e("dtoh am"))?;
                stream.synchronize().map_err(e("sync am"))?;
                out_ids.push(am[0] as u32);
            }
            // confidence (M:807-815): fp32 proj of concat(PRE-norm xc row, markov_embed)
            {
                let _p = phase!("1i5.conf_in_copies", prof.as_ref());
                let src = xc.slice(i * hidden..(i + 1) * hidden);
                let mut dst = conf_in.slice_mut(0..hidden);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("cin x"))?;
                let src = w1_row.slice(0..rank);
                let mut dst = conf_in.slice_mut(hidden..hidden + rank);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("cin m"))?;
            }
            {
                // `Self::dots` writes y[0]; the confidence now lands in slot i of a
                // block-wide buffer, so the launcher is called with the offset directly
                // (the same pointer-arithmetic pattern the add/argmax above use). Kernel,
                // f64 accumulation and operand order are untouched.
                let _p = phase!("1i6.conf_dots", prof.as_ref());
                unsafe {
                    ck(
                        "dots_f32 conf dspark",
                        k::memra_dsv4_dots_f32(
                            dpf!(conf_in, &stream),
                            dp!(ds.conf_w, &stream),
                            0,
                            (conf_out.device_ptr_mut(&stream).0 as usize + i * 4) as *mut f32,
                            1,
                            (hidden + rank) as i32,
                            1,
                            sp(&stream),
                        ),
                    )?;
                }
            }
            if !chain_device {
                let _p = phase!("1i7.conf_D2H_SYNC", None);
                let mut c = [0f32; 1];
                let view = conf_out.slice(i..i + 1);
                stream
                    .memcpy_dtoh(&view, &mut c[..])
                    .map_err(e("dtoh cf"))?;
                stream.synchronize().map_err(e("sync cf"))?;
                confidence.push(c[0]);
            }
        }
        if chain_device {
            // ONE drain for the whole chain: block ids + block confidences.
            let _p = phase!("1i8.chain_D2H_SYNC_once", None);
            let mut ids = vec![0i32; block];
            let view = am_dev.slice(1..block + 1);
            stream
                .memcpy_dtoh(&view, &mut ids[..])
                .map_err(e("dtoh chain ids"))?;
            let mut cf = vec![0f32; block];
            stream
                .memcpy_dtoh(&conf_out, &mut cf[..])
                .map_err(e("dtoh chain conf"))?;
            stream.synchronize().map_err(e("sync chain"))?;
            out_ids.extend(ids.iter().map(|&x| x as u32));
            confidence.extend_from_slice(&cf);
        }
        drop(_p_mk);
        // `markov_embed`, `margins` and `top1_logits` are CAPTURE-ONLY observables, and wanting
        // them mid-chain was the other reason the shipped loop had to know each id on the host.
        // `add_inplace` touches logits row i only at step i, so every row is final once the loop
        // ends and one post-loop read is bit-identical to the per-step reads it replaces.
        let membeds: Vec<f32> = if capture {
            let mut m = Vec::with_capacity(block * rank);
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for i in 0..block {
                let prev = out_ids[i] as usize;
                m.extend_from_slice(&ds.markov_w1_host[prev * rank..(prev + 1) * rank]);
            }
            m
        } else {
            Vec::new()
        };
        let cap = if capture {
            let logits_post = dtoh_f32(&stream, &logits)?;
            for i in 0..block {
                let row = &logits_post[i * vocab..(i + 1) * vocab];
                let top = out_ids[i + 1];
                let mut second = f32::NEG_INFINITY;
                for (vv, &val) in row.iter().enumerate() {
                    if vv as u32 != top && val > second {
                        second = val;
                    }
                }
                margins.push(row[top as usize] - second);
                top1_logits.push(row[top as usize]);
            }
            Some(DsparkCaptureOut {
                main_hidden: cap_main_hidden.unwrap(),
                main_x: cap_main_x.unwrap(),
                block_outs,
                x_collapsed: dtoh_f32(&stream, &xc)?,
                logits_pre: cap_logits_pre.unwrap(),
                logits_post,
                markov_embed: membeds,
            })
        } else {
            None
        };
        Ok(DsparkProposal {
            out_ids,
            confidence,
            margins,
            top1_logits,
            capture: cap,
        })
    }

    /// Device decode step + the DSpark tap into `dspark_state.taps` row `tap_row`
    /// (full logits — the gates' contract).
    pub fn decode_step_tap(
        &self,
        tok: u32,
        state: &mut DecodeState,
        dspark_state: &mut DsparkState,
        tap_row: usize,
    ) -> Res<Vec<f32>> {
        let DecodePath::Device { host_math } = self.decode_path else {
            return Err("decode_step_tap requires MEMRA_DSV4_DECODE_PATH=device".into());
        };
        let n_t = self.dspark().targets.len();
        let hidden = self.model.mc.n_embd as usize;
        let (logits, _) = self.decode_step_fast_tap(
            tok,
            state,
            true,
            host_math,
            Some((&mut dspark_state.taps, tap_row * n_t * hidden)),
        )?;
        Ok(logits.expect("want_logits"))
    }

    /// Greedy twin of [`Self::decode_step_tap`] (device argmax, 4-byte D2H).
    pub fn decode_step_greedy_tap(
        &self,
        tok: u32,
        state: &mut DecodeState,
        dspark_state: &mut DsparkState,
        tap_row: usize,
    ) -> Res<u32> {
        let DecodePath::Device { host_math } = self.decode_path else {
            return Err("decode_step_greedy_tap requires MEMRA_DSV4_DECODE_PATH=device".into());
        };
        let ds = self.dspark();
        let hidden = self.model.mc.n_embd as usize;
        let n_t = ds.targets.len();
        let (_, tok_next) = self.decode_step_fast_tap(
            tok,
            state,
            false,
            host_math,
            Some((&mut dspark_state.taps, tap_row * n_t * hidden)),
        )?;
        Ok(tok_next)
    }
}

// ================================================================ iteration 3, rung 4: batched T=k+1 device verify
//
// The rung that makes drafted decode pay. Design law (banked in the iteration-3 receipts
// before this code was written, and restated in cu/dsv4_gpu.cu's batched section):
//
//   1. BIT-EXACT against T sequential single-position steps. The greedy spec==plain
//      identity law is this lane's verdict instrument; if the verify pass computed
//      different logits than the plain pass, identity would break silently at every
//      near-tie and no gate could tell a port bug from a rounding fork. Achievable
//      because the device decode path's dense projections are OUR kernels: the batched
//      twins hoist the WEIGHT load across T activation rows without touching any
//      accumulation order. cuBLASLt is deliberately absent from this path (its m-order
//      changes split-K plans and shifts logits 0.18-3.08 — banked).
//   2. §3.1 ring hazard, exactly as GATED on the CPU oracle: window-ring writes go to
//      TRANSIENT rows (kvc rows [win+cap_blocks, win+cap_blocks+T)) and reads of
//      in-round positions are redirected there (`dsv4_build_idx_redirect`); the
//      compressor/indexer pending state advances in place with a snapshot + replay
//      payload; the append-only stores roll back by high-water mark; the drafter rings
//      advance for ACCEPTED positions only.
//   3. Where a kernel cannot batch (per-position compressor state machine, per-position
//      indexer top-k), the loop runs t = 0..T-1 in POSITION ORDER — the sequential
//      program's order, so in-round block emissions are visible to later queries exactly
//      as they would be sequentially.
//
// The one place uniformity is imposed: the batched sink attention takes ONE `slots`
// width for all T queries (the max over the round) and shorter queries' index tails are
// -1 pads. That is bit-inert by the pinned kernels' own pad contract (score -inf ->
// eval +0.0 -> skipped in both the denominator and the output chain), which is why it is
// legal rather than merely convenient.

/// Per-stage batched-verify workspace: the lane-8 arena widened to `tmax` rows. Held
/// separately from [`StepWs`] so the gated single-position path's allocations, launches
/// and bytes are literally untouched by this rung.
pub struct VerifyWs {
    pub tmax: usize,
    /// Phase identity, not inferred from row count. Spec verification never sets it.
    is_prefill: bool,
    h_a: CudaSlice<f32>,
    h_b: CudaSlice<f32>,
    h_rx: CudaSlice<f32>,
    emb: CudaSlice<f32>,
    mixes: CudaSlice<f32>,
    pre: CudaSlice<f32>,
    post: CudaSlice<f32>,
    comb: CudaSlice<f32>,
    y_hc: CudaSlice<f32>,
    x: CudaSlice<f32>,
    xf: CudaSlice<f32>,
    qr: CudaSlice<f32>,
    qr_b: CudaSlice<u8>,
    q: CudaSlice<f32>,
    kv: CudaSlice<f32>,
    qi: CudaSlice<f32>,
    wproj: CudaSlice<f32>,
    score: CudaSlice<f32>,
    topk_a: CudaSlice<u64>,
    topk_b: CudaSlice<u64>,
    topk_stride: usize,
    idx: CudaSlice<i32>,
    idx_stride: usize,
    o: CudaSlice<f32>,
    o_b: CudaSlice<u8>,
    og: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    gemm_xb: CudaSlice<u8>,
    raw: CudaSlice<f32>,
    sel: CudaSlice<i32>,
    selw: CudaSlice<f32>,
    order: CudaSlice<i32>,
    xq: CudaSlice<u8>,
    xs: CudaSlice<f32>,
    g1: CudaSlice<f32>,
    g3: CudaSlice<f32>,
    hbuf: CudaSlice<f32>,
    hq: CudaSlice<u8>,
    hs: CudaSlice<f32>,
    contrib: CudaSlice<f32>,
    y: CudaSlice<f32>,
    xb: CudaSlice<u8>,
    sg1: CudaSlice<f32>,
    sg3: CudaSlice<f32>,
    shbuf: CudaSlice<f32>,
    shb16: CudaSlice<u8>,
    sh_out: CudaSlice<f32>,
    cmp_emit: CudaSlice<f32>,
    cmp_shift: CudaSlice<f32>,
    sink_scores: CudaSlice<f32>,
    sink_evals: CudaSlice<f32>,
    sink_den: CudaSlice<f64>,
    head_mixes: CudaSlice<f32>,
    head_pre: CudaSlice<f32>,
    collapsed: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    tok: CudaSlice<i32>,
    pos_dev: CudaSlice<i32>,
    argmax: CudaSlice<i32>,
    /// ring-commit staging: transient rows copied out, then scattered to ring slots
    /// (source and destination live in the same `kvc` allocation, so the bounce is a
    /// borrow requirement, not a numeric one).
    bounce: CudaSlice<f32>,
    slot_rows: CudaSlice<i32>,
    /// hc-mean staging for the DSpark tap (one target at a time, then `place_cols`)
    tap_tmp: CudaSlice<f32>,
}

/// One compressor's verify-round checkpoint on device — the CPU oracle's `CompCkpt`,
/// device-realized: full pending snapshot + the per-position RAW (kv, score) rows that
/// were written, plus the store high-water mark. `dst` and `emitted` are pure functions
/// of the position, so nothing has to come back to the host to replay.
struct CmpCkptDev {
    kv_snap: CudaSlice<f32>,
    sc_snap: CudaSlice<f32>,
    rows_kv: CudaSlice<f32>,
    rows_sc: CudaSlice<f32>,
    latent: usize,
    ratio: usize,
    overlap: bool,
    n_blocks0: usize,
}

/// One trunk layer's verify-round checkpoint: the two compressor payloads. The window
/// ring needs no payload at all — the round never wrote it (transient rows instead).
struct LayerCkptDev {
    cmp: Option<CmpCkptDev>,
    idx: Option<CmpCkptDev>,
    /// first transient row id in this layer's `kvc` (== win + cap_blocks)
    trans_base: usize,
}

/// Whole-round verify state: the per-stage arenas + the per-layer §3.1 checkpoints.
pub struct VerifyState {
    ws: Vec<VerifyWs>,
    layers: Vec<LayerCkptDev>,
    pub tmax: usize,
    /// Decode-cache capacity this verify layout was planned against. The transient
    /// rows live immediately after each layer's capacity-sized compressed store, so
    /// using a model-wide verify state with a smaller session allocation is unsafe.
    capacity: usize,
    /// (pos0, t) of the open round; `None` between rounds. `commit_verify_dev` closes it.
    open: Option<(usize, usize)>,
    /// allocated bytes per device index (reported next to the drafter VRAM plan)
    pub bytes: Vec<u64>,
}

impl Dsv4Gpu {
    /// Verify-round depth ceiling: block_size + 1 with the drafter loaded, else 0 (and
    /// then no transient rows are reserved anywhere — today's exact allocation).
    pub fn verify_tmax(&self) -> usize {
        self.dspark.as_ref().map(|d| d.block_size + 1).unwrap_or(0)
    }

    /// Allocate the model-limit batched-verify state (arenas + §3.1 checkpoints).
    /// Capacity-planned serving should use [`Self::alloc_verify_state_for`] instead.
    pub fn alloc_verify_state(&self) -> Res<VerifyState> {
        self.alloc_verify_state_for(self.max_seq)
    }

    /// Allocate batched-verify scratch for the same admitted capacity as its
    /// [`DecodeState`]. In particular, every transient ring base is placed after the
    /// session-sized compressed store, not after the model-wide 1M-token store.
    pub fn alloc_verify_state_for(&self, capacity: usize) -> Res<VerifyState> {
        let tmax = self.verify_tmax();
        if tmax == 0 {
            return Err("alloc_verify_state needs MEMRA_DSV4_DRAFTER=dspark".into());
        }
        self.alloc_batched_state_for(capacity, tmax)
    }

    /// Allocate the same transaction machinery at a wider, explicit chunk width for
    /// bounded-memory prefill. Unlike speculative verification, this does not require a
    /// drafter; every row is teacher-forced and committed.
    pub fn alloc_prefill_state_for(&self, capacity: usize, width: usize) -> Res<VerifyState> {
        if width == 0 || width > DSV4_BATCH_WIDTH_MAX {
            return Err(format!(
                "dsv4 prefill chunk width {width} outside 1..={DSV4_BATCH_WIDTH_MAX}"
            ));
        }
        let mut state = self.alloc_batched_state_for(capacity, width)?;
        for workspace in &mut state.ws {
            workspace.is_prefill = true;
        }
        Ok(state)
    }

    fn alloc_batched_state_for(&self, capacity: usize, tmax: usize) -> Res<VerifyState> {
        if capacity == 0 || capacity > self.max_seq {
            return Err(format!(
                "dsv4 verify capacity {capacity} outside 1..={} model limit",
                self.max_seq
            ));
        }
        if !matches!(self.decode_path, DecodePath::Device { .. }) {
            return Err(
                "batched verify is a device-path rung (MEMRA_DSV4_DECODE_PATH=device)".into(),
            );
        }
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let moe = mc.moe.as_ref().expect("moe");
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let heads = mc.n_head as usize;
        let hd = d.head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let iheads = d.index_n_heads as usize;
        let ihd = d.index_head_dim as usize;
        let topk = moe.expert_used_count as usize;
        let ne = moe.expert_count as usize;
        let inter = moe.expert_ff_length as usize;
        let itopk = d.index_topk as usize;
        let n_trunk = (mc.n_layer - mc.nextn_predict_layers) as usize;
        let vocab = {
            let (info, _) = self.model.st.raw("head.weight").expect("head");
            info.shape[0] as usize
        };
        let sh_inter = {
            let (info, _) = self
                .model
                .st
                .raw("layers.0.ffn.shared_experts.w1.weight")
                .expect("shared w1");
            info.shape[0] as usize
        };
        let mut max_d = 0usize;
        let mut max_shift = 0usize;
        let mut min_index_ratio = usize::MAX;
        for st in &self.stages {
            for l in &st.layers {
                for cmp in l.cmp.iter().chain(l.idx.as_ref().map(|ix| &ix.cmp)) {
                    max_d = max_d.max(cmp.d);
                    if cmp.overlap {
                        max_shift = max_shift.max(cmp.ratio * cmp.latent);
                    }
                }
                if let Some(ix) = &l.idx {
                    min_index_ratio = min_index_ratio.min(ix.cmp.ratio);
                }
            }
        }
        assert!(min_index_ratio != usize::MAX, "no indexer layers?");
        let score_cap = self.max_seq / min_index_ratio + 1;
        let topk_stride = score_cap.div_ceil(4096) * 512;
        let idx_tail = itopk.max(self.max_seq / 128 + 1);
        let idx_stride = win + idx_tail;
        let max_gemm_k = (o_groups * o_lora).max(hidden).max(q_lora).max(sh_inter);
        let mut bytes = vec![0u64; self.stages.len()];
        let mut ws = Vec::with_capacity(self.stages.len());
        for st in &self.stages {
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx vws"))?;
            let s = st.gpu.stream();
            let acc = std::cell::Cell::new(0u64);
            let f = |n: usize| {
                acc.set(acc.get() + (n * 4) as u64);
                s.alloc_zeros::<f32>(n).map_err(e("vws f32"))
            };
            let b = |n: usize| {
                acc.set(acc.get() + n as u64);
                s.alloc_zeros::<u8>(n).map_err(e("vws u8"))
            };
            let i = |n: usize| {
                acc.set(acc.get() + (n * 4) as u64);
                s.alloc_zeros::<i32>(n).map_err(e("vws i32"))
            };
            let u = |n: usize| {
                acc.set(acc.get() + (n * 8) as u64);
                s.alloc_zeros::<u64>(n).map_err(e("vws u64"))
            };
            let w = VerifyWs {
                tmax,
                is_prefill: false,
                h_a: f(tmax * hc * hidden)?,
                h_b: f(tmax * hc * hidden)?,
                h_rx: f(tmax * hc * hidden)?,
                emb: f(tmax * hidden)?,
                mixes: f(tmax * (2 + hc) * hc)?,
                pre: f(tmax * hc)?,
                post: f(tmax * hc)?,
                comb: f(tmax * hc * hc)?,
                y_hc: f(tmax * hidden)?,
                x: f(tmax * hidden)?,
                xf: f(tmax * hidden)?,
                qr: f(tmax * q_lora)?,
                qr_b: b(tmax * q_lora * 2)?,
                q: f(tmax * heads * hd)?,
                kv: f(tmax * hd)?,
                qi: f(tmax * iheads * ihd)?,
                wproj: f(tmax * iheads)?,
                score: f(tmax * score_cap)?,
                topk_a: u(tmax * topk_stride)?,
                topk_b: u(tmax * topk_stride)?,
                topk_stride,
                idx: i(tmax * idx_stride)?,
                idx_stride,
                o: f(tmax * heads * hd)?,
                o_b: b(tmax * heads * hd * 2)?,
                og: f(tmax * o_groups * o_lora)?,
                attn_out: f(tmax * hidden)?,
                gemm_xb: b(tmax * max_gemm_k * 2)?,
                raw: f(tmax * ne)?,
                sel: i(tmax * topk)?,
                selw: f(tmax * topk)?,
                order: i(tmax * topk)?,
                xq: b(tmax * hidden)?,
                xs: f(tmax * hidden / 128)?,
                g1: f(tmax * topk * inter)?,
                g3: f(tmax * topk * inter)?,
                hbuf: f(tmax * topk * inter)?,
                hq: b(tmax * topk * inter)?,
                hs: f(tmax * topk * inter / 128)?,
                contrib: f(tmax * topk * hidden)?,
                y: f(tmax * hidden)?,
                xb: b(tmax * hidden * 2)?,
                sg1: f(tmax * sh_inter)?,
                sg3: f(tmax * sh_inter)?,
                shbuf: f(tmax * sh_inter)?,
                shb16: b(tmax * sh_inter * 2)?,
                sh_out: f(tmax * hidden)?,
                cmp_emit: f(2 * max_d)?,
                cmp_shift: f(max_shift.max(1))?,
                sink_scores: f(tmax * heads * idx_stride)?,
                sink_evals: f(tmax * heads * idx_stride)?,
                sink_den: {
                    acc.set(acc.get() + (tmax * heads * 8) as u64);
                    s.alloc_zeros::<f64>(tmax * heads).map_err(e("vws f64"))?
                },
                head_mixes: f(tmax * hc)?,
                head_pre: f(tmax * hc)?,
                collapsed: f(tmax * hidden)?,
                logits: f(tmax * vocab)?,
                tok: i(tmax)?,
                pos_dev: i(tmax)?,
                argmax: i(tmax)?,
                bounce: f(tmax * hd)?,
                slot_rows: i(tmax)?,
                tap_tmp: f(tmax * hidden)?,
            };
            bytes[st.dev] += acc.get();
            ws.push(w);
        }
        // per-layer §3.1 checkpoints, each on the layer's own device
        let mut layers = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk {
            let stage_i = self.layer_stage[il];
            let st = &self.stages[stage_i];
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx ckpt"))?;
            let stream = st.gpu.stream();
            let lidx = st
                .layers
                .iter()
                .position(|l| l.il == il as u32)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage_i}"));
            let layer = &st.layers[lidx];
            let cap_blocks = dsv4_cache_cap_blocks(capacity, layer.ratio);
            let mk = |cmp: &CmpDev| -> Res<CmpCkptDev> {
                let slots = if cmp.overlap {
                    2 * cmp.ratio
                } else {
                    cmp.ratio
                };
                Ok(CmpCkptDev {
                    kv_snap: stream
                        .alloc_zeros::<f32>(slots * cmp.latent)
                        .map_err(e("ckpt kv snap"))?,
                    sc_snap: stream
                        .alloc_zeros::<f32>(slots * cmp.latent)
                        .map_err(e("ckpt sc snap"))?,
                    rows_kv: stream
                        .alloc_zeros::<f32>(tmax * cmp.latent)
                        .map_err(e("ckpt rows kv"))?,
                    rows_sc: stream
                        .alloc_zeros::<f32>(tmax * cmp.latent)
                        .map_err(e("ckpt rows sc"))?,
                    latent: cmp.latent,
                    ratio: cmp.ratio,
                    overlap: cmp.overlap,
                    n_blocks0: 0,
                })
            };
            let cmp = match &layer.cmp {
                Some(c) => Some(mk(c)?),
                None => None,
            };
            let idxc = match &layer.idx {
                Some(ix) => Some(mk(&ix.cmp)?),
                None => None,
            };
            for c in cmp.iter().chain(idxc.iter()) {
                let slots = if c.overlap { 2 * c.ratio } else { c.ratio };
                bytes[st.dev] += ((2 * slots * c.latent + 2 * tmax * c.latent) * 4) as u64;
            }
            layers.push(LayerCkptDev {
                cmp,
                idx: idxc,
                trans_base: d.sliding_window as usize + cap_blocks,
            });
        }
        for st in &self.stages {
            st.gpu.stream().synchronize().map_err(e("vws sync"))?;
        }
        Ok(VerifyState {
            ws,
            layers,
            tmax,
            capacity,
            open: None,
            bytes,
        })
    }

    /// Batched bf16 GEMV: y[m, n] = x[m, k] @ W[n, k]^T with the weight row read once.
    /// `xstride`/`ystride` in elements (0 == packed) — the grouped output projection is
    /// the only caller that needs them.
    #[allow(clippy::too_many_arguments)]
    fn gemv_m_dev(
        st: &Stage,
        w: DW,
        x_ptr: *const c_void,
        y_ptr: *mut f32,
        m: usize,
        n: usize,
        kdim: usize,
        xstride: usize,
        ystride: usize,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        unsafe {
            match w {
                DW::Bf16(w_ptr) => ck(
                    "gemv_bf16_m dev",
                    k::memra_dsv4_gemv_bf16_m(
                        w_ptr,
                        x_ptr,
                        y_ptr,
                        m as i32,
                        n as i32,
                        kdim as i32,
                        xstride as i32,
                        ystride as i32,
                        sp(&stream),
                    ),
                ),
                DW::Fp8 {
                    codes,
                    scales,
                    sc_cols,
                } => ck(
                    "gemv_fp8_m dev",
                    k::memra_dsv4_gemv_fp8_m(
                        codes,
                        scales,
                        sc_cols,
                        x_ptr,
                        y_ptr,
                        m as i32,
                        n as i32,
                        kdim as i32,
                        xstride as i32,
                        ystride as i32,
                        sp(&stream),
                    ),
                ),
            }
        }
    }

    /// f32 cvt + batched GEMV (the m=T twin of `gemm_dev`).
    #[allow(clippy::too_many_arguments)]
    fn gemm_m_dev(
        st: &Stage,
        x_f32: *const f32,
        xb: &mut CudaSlice<u8>,
        w: DW,
        m: usize,
        n: usize,
        kdim: usize,
        y_ptr: *mut f32,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        unsafe {
            ck(
                "cvt_bf16 m dev",
                k::memra_dsv4_cvt_bf16(
                    x_f32,
                    xb.device_ptr_mut(&stream).0 as *mut c_void,
                    (m * kdim) as i64,
                    sp(&stream),
                ),
            )?;
        }
        Self::gemv_m_dev(
            st,
            w,
            xb.device_ptr(&stream).0 as *const c_void,
            y_ptr,
            m,
            n,
            kdim,
            0,
            0,
        )
    }

    /// Island dots, batched rows, weight row hoisted. Same arm selection as `dots_dev`.
    #[allow(clippy::too_many_arguments)]
    fn dots_m_dev(
        &self,
        st: &Stage,
        x: *const f32,
        w_f32: *const c_void,
        w_is_bf16: i32,
        s: usize,
        kdim: usize,
        n: usize,
        y: *mut f32,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        unsafe {
            if self.dots_f32 {
                ck(
                    "dots_f32acc_mrow",
                    k::memra_dsv4_dots_f32acc_mrow(
                        x,
                        w_f32,
                        w_is_bf16,
                        y,
                        s as i32,
                        kdim as i32,
                        n as i32,
                        sp(&stream),
                    ),
                )
            } else {
                ck(
                    "dots_f32_mrow",
                    k::memra_dsv4_dots_f32_mrow(
                        x,
                        w_f32,
                        w_is_bf16,
                        y,
                        s as i32,
                        kdim as i32,
                        n as i32,
                        sp(&stream),
                    ),
                )
            }
        }
    }
}

impl Dsv4Gpu {
    /// hc_pre for T rows: the `hc_pre_dev` program with every kernel taking the row
    /// count (Sinkhorn either the host closure per row — byte-identity arm — or the
    /// one-block-per-position device twin).
    #[allow(clippy::too_many_arguments)]
    fn hc_pre_batch_dev(
        &self,
        st: &Stage,
        h_ptr: *const f32,
        fn_w: &CudaSlice<f32>,
        base_host: &[f32],
        scale_host: &[f32],
        base_dev: &CudaSlice<f32>,
        scale_dev: &CudaSlice<f32>,
        vws: &mut VerifyWs,
        t: usize,
        hc: usize,
        hidden: usize,
        iters: u32,
        hc_eps: f32,
        host_math: bool,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        let w = hc * hidden;
        let rows = (2 + hc) * hc;
        self.dots_m_dev(
            st,
            h_ptr,
            fn_w.device_ptr(&stream).0 as *const c_void,
            0,
            t,
            w,
            rows,
            vws.mixes.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "rowsq_scale batch",
                self.rowsq_scale_arm(
                    h_ptr,
                    dpm!(vws.mixes, &stream),
                    t as i32,
                    w as i32,
                    rows as i32,
                    hc_eps,
                    sp(&stream),
                ),
            )?;
        }
        if host_math {
            let mut mixes_h = vec![0f32; t * rows];
            let view = vws.mixes.slice(0..t * rows);
            stream
                .memcpy_dtoh(&view, &mut mixes_h[..])
                .map_err(e("dtoh mixes batch"))?;
            stream.synchronize().map_err(e("sync mixes batch"))?;
            let (pre_h, post_h, comb_h) =
                hc_split_sinkhorn(&mixes_h, t, hc, scale_host, base_host, iters, hc_eps);
            let mut dp = vws.pre.slice_mut(0..t * hc);
            stream
                .memcpy_htod(&pre_h, &mut dp)
                .map_err(e("htod pre b"))?;
            let mut dp = vws.post.slice_mut(0..t * hc);
            stream
                .memcpy_htod(&post_h, &mut dp)
                .map_err(e("htod post b"))?;
            let mut dp = vws.comb.slice_mut(0..t * hc * hc);
            stream
                .memcpy_htod(&comb_h, &mut dp)
                .map_err(e("htod comb b"))?;
        } else {
            unsafe {
                ck(
                    "hc_sinkhorn_m",
                    k::memra_dsv4_hc_sinkhorn_m(
                        dpf!(vws.mixes, &stream),
                        dpf!(scale_dev, &stream),
                        dpf!(base_dev, &stream),
                        dpm!(vws.pre, &stream),
                        dpm!(vws.post, &stream),
                        dpm!(vws.comb, &stream),
                        t as i32,
                        hc as i32,
                        iters as i32,
                        hc_eps,
                        sp(&stream),
                    ),
                )?;
            }
        }
        unsafe {
            ck(
                "hc_collapse batch",
                k::memra_dsv4_hc_collapse(
                    h_ptr,
                    dpf!(vws.pre, &stream),
                    dpm!(vws.y_hc, &stream),
                    t as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// Compressor advance for a whole verify round (§3.1): the two projection GEMMs run
    /// batched STRAIGHT INTO the checkpoint's row payload (which is both the record and
    /// the source of the pending writes — one copy, not two), then the pending state
    /// machine + emissions run t = 0..T-1 in POSITION ORDER, exactly the sequential
    /// program. The snapshot is taken before the first write.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    fn cmp_decode_batch_dev(
        &self,
        st: &Stage,
        cmp: &CmpDev,
        x_ptr: *const f32,
        t: usize,
        pos0: usize,
        hidden: usize,
        fc_dev: &CudaSlice<f32>,
        rd: usize,
        eps: f32,
        ck_dev: &mut CmpCkptDev,
        emit: &mut CudaSlice<f32>,
        shift: &mut CudaSlice<f32>,
        pend_kv: &mut CudaSlice<f32>,
        pend_score: &mut CudaSlice<f32>,
        store: &mut CudaSlice<f32>,
        row0: usize,
        blocks: &mut usize,
    ) -> Res<()> {
        let stream = st.gpu.stream();
        let (ratio, d, latent) = (cmp.ratio, cmp.d, cmp.latent);
        // snapshot + high-water mark BEFORE anything is written
        stream
            .memcpy_dtod(pend_kv, &mut ck_dev.kv_snap)
            .map_err(e("ckpt snap kv"))?;
        stream
            .memcpy_dtod(pend_score, &mut ck_dev.sc_snap)
            .map_err(e("ckpt snap sc"))?;
        ck_dev.n_blocks0 = *blocks;
        self.dots_m_dev(
            st,
            x_ptr,
            cmp.wkv.device_ptr(&stream).0 as *const c_void,
            0,
            t,
            hidden,
            latent,
            ck_dev.rows_kv.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        self.dots_m_dev(
            st,
            x_ptr,
            cmp.wgate.device_ptr(&stream).0 as *const c_void,
            0,
            t,
            hidden,
            latent,
            ck_dev.rows_sc.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        for i in 0..t {
            let pos = pos0 + i;
            let slot = if cmp.overlap {
                ratio + pos % ratio
            } else {
                pos % ratio
            };
            {
                let src = ck_dev.rows_kv.slice(i * latent..(i + 1) * latent);
                let mut dst = pend_kv.slice_mut(slot * latent..(slot + 1) * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("pend kv b"))?;
                let src = ck_dev.rows_sc.slice(i * latent..(i + 1) * latent);
                let mut dst = pend_score.slice_mut(slot * latent..(slot + 1) * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("pend sc b"))?;
            }
            if (pos + 1) % ratio != 0 {
                continue;
            }
            let j = pos / ratio;
            let nb_launch = if cmp.overlap { 2usize } else { 1 };
            let row_off = if cmp.overlap { d } else { 0 };
            unsafe {
                ck(
                    "compressor_pool batch",
                    k::memra_dsv4_compressor_pool(
                        dpf!(*pend_kv, &stream),
                        dpf!(*pend_score, &stream),
                        dpf!(cmp.ape, &stream),
                        dpm!(*emit, &stream),
                        nb_launch as i32,
                        ratio as i32,
                        d as i32,
                        latent as i32,
                        cmp.overlap as i32,
                        sp(&stream),
                    ),
                )?;
                let row_c = (emit.device_ptr(&stream).0 as usize + row_off * 4) as *const f32;
                let row_m = (emit.device_ptr_mut(&stream).0 as usize + row_off * 4) as *mut f32;
                ck(
                    "rmsnorm batch cmp",
                    self.rmsnorm_arm(
                        row_c,
                        dpf!(cmp.norm, &stream),
                        row_m,
                        1,
                        d as i32,
                        eps,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "rope_at batch cmp",
                    k::memra_dsv4_rope_at(
                        row_m,
                        1,
                        d as i32,
                        rd as i32,
                        dpf!(fc_dev, &stream),
                        (j * ratio) as i32,
                        0,
                        sp(&stream),
                    ),
                )?;
                if cmp.rotate {
                    let scale = (d as f32).powf(-0.5);
                    ck(
                        "hadamard batch cmp",
                        k::memra_dsv4_hadamard(row_m, 1, d as i32, scale, sp(&stream)),
                    )?;
                    ck(
                        "fp4 batch cmp",
                        k::memra_dsv4_fp4_act_quant(row_m, 1, d as i64, d as i32, sp(&stream)),
                    )?;
                } else {
                    ck(
                        "act_quant batch cmp",
                        k::memra_dsv4_act_quant(
                            row_m,
                            1,
                            d as i64,
                            (d - rd) as i32,
                            64,
                            (self.variant == ActQuantVariant::ClampOnly) as i32,
                            sp(&stream),
                        ),
                    )?;
                }
            }
            {
                let src = emit.slice(row_off..row_off + d);
                let mut dst = store.slice_mut((row0 + j) * d..(row0 + j + 1) * d);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("emit store b"))?;
            }
            if cmp.overlap {
                {
                    let src = pend_kv.slice(ratio * latent..2 * ratio * latent);
                    let mut dst = shift.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("bshift1"))?;
                }
                {
                    let src = shift.slice(0..ratio * latent);
                    let mut dst = pend_kv.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("bshift2"))?;
                }
                {
                    let src = pend_score.slice(ratio * latent..2 * ratio * latent);
                    let mut dst = shift.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("bshift3"))?;
                }
                {
                    let src = shift.slice(0..ratio * latent);
                    let mut dst = pend_score.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("bshift4"))?;
                }
            }
            *blocks = j + 1;
        }
        Ok(())
    }

    /// §3.1 compressor rollback: restore the snapshot, then REPLAY the committed
    /// positions' row writes + cur->prev shifts + block accounting. Emitted store rows
    /// of the committed prefix are kept as the round wrote them (bit-identical to the
    /// sequential twin — the batch advanced the pending in position order, so every
    /// emission pooled the same inputs). The CPU oracle's `rollback_replay`, verbatim.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    fn cmp_rollback_replay_dev(
        &self,
        st: &Stage,
        ck_dev: &CmpCkptDev,
        n_commit: usize,
        t: usize,
        pos0: usize,
        shift: &mut CudaSlice<f32>,
        pend_kv: &mut CudaSlice<f32>,
        pend_score: &mut CudaSlice<f32>,
        blocks: &mut usize,
    ) -> Res<()> {
        if n_commit == t {
            return Ok(()); // fully committed: the in-place batch state is already exact
        }
        let stream = st.gpu.stream();
        let (ratio, latent, overlap) = (ck_dev.ratio, ck_dev.latent, ck_dev.overlap);
        stream
            .memcpy_dtod(&ck_dev.kv_snap, pend_kv)
            .map_err(e("rb kv snap"))?;
        stream
            .memcpy_dtod(&ck_dev.sc_snap, pend_score)
            .map_err(e("rb sc snap"))?;
        *blocks = ck_dev.n_blocks0;
        for i in 0..n_commit {
            let pos = pos0 + i;
            let slot = if overlap {
                ratio + pos % ratio
            } else {
                pos % ratio
            };
            {
                let src = ck_dev.rows_kv.slice(i * latent..(i + 1) * latent);
                let mut dst = pend_kv.slice_mut(slot * latent..(slot + 1) * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("rb row kv"))?;
                let src = ck_dev.rows_sc.slice(i * latent..(i + 1) * latent);
                let mut dst = pend_score.slice_mut(slot * latent..(slot + 1) * latent);
                stream.memcpy_dtod(&src, &mut dst).map_err(e("rb row sc"))?;
            }
            if (pos + 1) % ratio != 0 {
                continue;
            }
            if overlap {
                {
                    let src = pend_kv.slice(ratio * latent..2 * ratio * latent);
                    let mut dst = shift.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("rbshift1"))?;
                }
                {
                    let src = shift.slice(0..ratio * latent);
                    let mut dst = pend_kv.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("rbshift2"))?;
                }
                {
                    let src = pend_score.slice(ratio * latent..2 * ratio * latent);
                    let mut dst = shift.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("rbshift3"))?;
                }
                {
                    let src = shift.slice(0..ratio * latent);
                    let mut dst = pend_score.slice_mut(0..ratio * latent);
                    stream.memcpy_dtod(&src, &mut dst).map_err(e("rbshift4"))?;
                }
            }
            *blocks += 1;
        }
        Ok(())
    }
}

impl Dsv4Gpu {
    /// One trunk block, BATCHED T-position verify (§3.1). Positions pos0..pos0+t-1,
    /// tokens `toks`. Input h is vws.h_a (or vws.h_rx right after a stage boundary);
    /// output lands in vws.h_a. Window-ring writes go to the layer's TRANSIENT kvc rows
    /// and every query's index list is built with the redirect, so the persistent ring
    /// is read-only for the whole round.
    #[allow(clippy::too_many_arguments)]
    fn block_verify_dev(
        &self,
        st: &Stage,
        layer: &LayerDev,
        cache: &mut LayerCache,
        lck: &mut LayerCkptDev,
        vws: &mut VerifyWs,
        input_rx: bool,
        pos0: usize,
        t: usize,
        toks: &[u32],
        host_math: bool,
    ) -> Res<()> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let heads = mc.n_head as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let eps = mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;
        let stream = st.gpu.stream();
        let fc_dev: *const f32 = if layer.ratio != 0 {
            st.fc_yarn.device_ptr(&stream).0 as *const f32
        } else {
            st.fc_plain.device_ptr(&stream).0 as *const f32
        };
        let clamp_only = (self.variant == ActQuantVariant::ClampOnly) as i32;
        let trans_base = lck.trans_base;
        let LayerCache {
            kvc,
            n_blocks,
            pend_kv,
            pend_score,
            ikvc,
            i_blocks,
            ipend_kv,
            ipend_score,
        } = cache;

        // ---- attention sub-block
        let h_in_ptr: *const f32 = if input_rx {
            vws.h_rx.device_ptr(&stream).0 as *const f32
        } else {
            vws.h_a.device_ptr(&stream).0 as *const f32
        };
        self.hc_pre_batch_dev(
            st,
            h_in_ptr,
            &layer.hc_attn_fn,
            &layer.hc_attn_base,
            &layer.hc_attn_scale,
            &layer.hc_attn_base_dev,
            &layer.hc_attn_scale_dev,
            vws,
            t,
            hc,
            hidden,
            iters,
            hc_eps,
            host_math,
        )?;
        unsafe {
            ck(
                "rmsnorm attn batch",
                self.rmsnorm_arm(
                    dpf!(vws.y_hc, &stream),
                    dpf!(layer.attn_norm, &stream),
                    dpm!(vws.x, &stream),
                    t as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }

        // q path (weights read once for all t rows)
        Self::gemm_m_dev(
            st,
            vws.x.device_ptr(&stream).0 as *const f32,
            &mut vws.gemm_xb,
            dwsel(self.dense_fp8, &stream, &layer.wq_a, &layer.wq_a_fp8),
            t,
            q_lora,
            hidden,
            vws.qr.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "rmsnorm q batch",
                self.rmsnorm_arm(
                    dpf!(vws.qr, &stream),
                    dpf!(layer.q_norm, &stream),
                    dpm!(vws.qr, &stream),
                    t as i32,
                    q_lora as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "cvt qr batch",
                k::memra_dsv4_cvt_bf16(
                    dpf!(vws.qr, &stream),
                    vws.qr_b.device_ptr_mut(&stream).0 as *mut c_void,
                    (t * q_lora) as i64,
                    sp(&stream),
                ),
            )?;
        }
        Self::gemv_m_dev(
            st,
            dwsel(self.dense_fp8, &stream, &layer.wq_b, &layer.wq_b_fp8),
            vws.qr_b.device_ptr(&stream).0 as *const c_void,
            vws.q.device_ptr_mut(&stream).0 as *mut f32,
            t,
            heads * hd,
            q_lora,
            0,
            0,
        )?;
        unsafe {
            ck(
                "headrms batch",
                self.headrms_arm(
                    dpm!(vws.q, &stream),
                    (t * heads) as i32,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope q batch",
                k::memra_dsv4_rope(
                    dpm!(vws.q, &stream),
                    t as i32,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    fc_dev,
                    vws.pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
        }

        // shared K==V latent rows + window QAT, then the TRANSIENT ring write
        Self::gemm_m_dev(
            st,
            vws.x.device_ptr(&stream).0 as *const f32,
            &mut vws.gemm_xb,
            dwsel(self.dense_fp8, &stream, &layer.wkv, &layer.wkv_fp8),
            t,
            hd,
            hidden,
            vws.kv.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "rmsnorm kv batch",
                self.rmsnorm_arm(
                    dpf!(vws.kv, &stream),
                    dpf!(layer.kv_norm, &stream),
                    dpm!(vws.kv, &stream),
                    t as i32,
                    hd as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
            ck(
                "rope kv batch",
                k::memra_dsv4_rope(
                    dpm!(vws.kv, &stream),
                    t as i32,
                    1,
                    hd as i32,
                    rd as i32,
                    fc_dev,
                    vws.pos_dev.device_ptr(&stream).0 as *const i32,
                    0,
                    sp(&stream),
                ),
            )?;
            ck(
                "act_quant kv batch",
                k::memra_dsv4_act_quant(
                    dpm!(vws.kv, &stream),
                    t as i32,
                    hd as i64,
                    (hd - rd) as i32,
                    64,
                    clamp_only,
                    sp(&stream),
                ),
            )?;
        }
        {
            let src = vws.kv.slice(0..t * hd);
            let mut dst = kvc.slice_mut(trans_base * hd..(trans_base + t) * hd);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(e("transient ring write"))?;
        }

        // ---- per-position index lists (redirected) + compressor advances
        let mut slots = win;
        if layer.ratio != 0 {
            let ratio = layer.ratio;
            // the round's per-position block counts (host arithmetic, exactly the
            // sequential program's `(pos+1)/ratio`)
            let nbs: Vec<usize> = (0..t).map(|i| (pos0 + i + 1) / ratio).collect();
            if let Some(ix) = &layer.idx {
                // indexer q, batched
                Self::gemv_m_dev(
                    st,
                    dwsel(self.dense_fp8, &stream, &ix.wq_b, &ix.wq_b_fp8),
                    vws.qr_b.device_ptr(&stream).0 as *const c_void,
                    vws.qi.device_ptr_mut(&stream).0 as *mut f32,
                    t,
                    ix.heads * ix.hd,
                    q_lora,
                    0,
                    0,
                )?;
                unsafe {
                    ck(
                        "rope qi batch",
                        k::memra_dsv4_rope(
                            dpm!(vws.qi, &stream),
                            t as i32,
                            ix.heads as i32,
                            ix.hd as i32,
                            rd as i32,
                            fc_dev,
                            vws.pos_dev.device_ptr(&stream).0 as *const i32,
                            0,
                            sp(&stream),
                        ),
                    )?;
                    let scale = (ix.hd as f32).powf(-0.5);
                    ck(
                        "hadamard qi batch",
                        k::memra_dsv4_hadamard(
                            dpm!(vws.qi, &stream),
                            (t * ix.heads) as i32,
                            ix.hd as i32,
                            scale,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "fp4 qi batch",
                        k::memra_dsv4_fp4_act_quant(
                            dpm!(vws.qi, &stream),
                            (t * ix.heads) as i32,
                            ix.hd as i64,
                            ix.hd as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                // indexer weights projection, batched
                Self::gemm_m_dev(
                    st,
                    vws.x.device_ptr(&stream).0 as *const f32,
                    &mut vws.gemm_xb,
                    dwsel(
                        self.dense_fp8,
                        &stream,
                        &ix.weights_proj,
                        &ix.weights_proj_fp8,
                    ),
                    t,
                    ix.heads,
                    hidden,
                    vws.wproj.device_ptr_mut(&stream).0 as *mut f32,
                )?;
                // indexer compressor: batched projections + position-ordered state machine
                {
                    let VerifyWs {
                        x,
                        cmp_emit,
                        cmp_shift,
                        ..
                    } = vws;
                    self.cmp_decode_batch_dev(
                        st,
                        &ix.cmp,
                        x.device_ptr(&stream).0 as *const f32,
                        t,
                        pos0,
                        hidden,
                        &st.fc_yarn,
                        rd,
                        eps,
                        lck.idx.as_mut().expect("idx ckpt"),
                        cmp_emit,
                        cmp_shift,
                        ipend_kv.as_mut().expect("ipend"),
                        ipend_score.as_mut().expect("ipend"),
                        ikvc.as_mut().expect("ikvc"),
                        0,
                        i_blocks,
                    )?;
                }
                debug_assert_eq!(*i_blocks, nbs[t - 1], "indexer block count (batch)");
                let kks: Vec<usize> = nbs.iter().map(|&nb| ix.topk.min(nb)).collect();
                let tail_max = kks.iter().cloned().max().unwrap_or(0);
                slots = win + tail_max;
                // Keep the shipped speculative range (today T<=6, conservatively <=8)
                // on the pre-batch scalar sequence: the batched indexer is a long-prefill
                // optimization and measured no short-decode win. T=1 also remains the
                // always-live exactness witness for the width-64 hardware gate.
                if host_math || t <= 8 {
                    for i in 0..t {
                        let pos = pos0 + i;
                        let idx_off = i * vws.idx_stride;
                        unsafe {
                            ck(
                                "build_idx_redirect fine",
                                k::memra_dsv4_build_idx_redirect(
                                    (vws.idx.device_ptr_mut(&stream).0 as usize + idx_off * 4)
                                        as *mut i32,
                                    pos as i32,
                                    win as i32,
                                    0,
                                    slots as i32,
                                    pos0 as i32,
                                    trans_base as i32,
                                    sp(&stream),
                                ),
                            )?;
                        }
                        let nb = nbs[i];
                        if nb == 0 {
                            continue;
                        }
                        let wscale =
                            ((ix.hd as f64).powf(-0.5) * (ix.heads as f64).powf(-0.5)) as f32;
                        unsafe {
                            ck(
                                "indexer_score batch",
                                self.indexer_score_arm(
                                    (vws.qi.device_ptr(&stream).0 as usize
                                        + i * ix.heads * ix.hd * 4)
                                        as *const f32,
                                    dpf!(ikvc.as_ref().expect("ikvc"), &stream),
                                    (vws.wproj.device_ptr(&stream).0 as usize + i * ix.heads * 4)
                                        as *const f32,
                                    wscale,
                                    dpm!(vws.score, &stream),
                                    1,
                                    ix.heads as i32,
                                    ix.hd as i32,
                                    nb as i32,
                                    ratio as i32,
                                    nb as i32,
                                    sp(&stream),
                                ),
                            )?;
                        }
                        let kk = kks[i];
                        if !host_math && nb > 4096 {
                            // Keep the small-context witness unchanged. Large-history
                            // verification must not copy and sort all scores on the CPU.
                            // Each row has its own nb; scratch is safely reused on this
                            // single stream before the following row overwrites score.
                            unsafe {
                                ck(
                                    "topk_idx_stream narrow verify",
                                    k::memra_dsv4_topk_idx_stream_m(
                                        dpf!(vws.score, &stream),
                                        1,
                                        nb as i32,
                                        kk as i32,
                                        win as i32,
                                        (vws.idx.device_ptr_mut(&stream).0 as usize + idx_off * 4)
                                            as *mut i32,
                                        vws.idx_stride as i32,
                                        vws.topk_a.device_ptr_mut(&stream).0 as *mut u64,
                                        vws.topk_b.device_ptr_mut(&stream).0 as *mut u64,
                                        vws.topk_stride as i32,
                                        sp(&stream),
                                    ),
                                )?;
                            }
                            continue;
                        }
                        let score_h = {
                            let view = vws.score.slice(0..nb);
                            let mut v = vec![0f32; nb];
                            stream
                                .memcpy_dtoh(&view, &mut v[..])
                                .map_err(e("dtoh sc b"))?;
                            stream.synchronize().map_err(e("sync sc b"))?;
                            v
                        };
                        let mut order: Vec<usize> = (0..nb).collect();
                        order.sort_by(|&a, &b| {
                            score_h[b]
                                .partial_cmp(&score_h[a])
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.cmp(&b))
                        });
                        let cidx: Vec<i32> = order
                            .into_iter()
                            .take(kk)
                            .map(|j| (j + win) as i32)
                            .collect();
                        let mut dst = vws.idx.slice_mut(idx_off + win..idx_off + win + kk);
                        stream
                            .memcpy_htod(&cidx, &mut dst)
                            .map_err(e("htod idx b"))?;
                    }
                } else {
                    let nb = nbs[t - 1];
                    unsafe {
                        ck(
                            "build_idx_redirect_m fine",
                            k::memra_dsv4_build_idx_redirect_m(
                                vws.idx.device_ptr_mut(&stream).0 as *mut i32,
                                pos0 as i32,
                                t as i32,
                                win as i32,
                                ratio as i32,
                                slots as i32,
                                vws.idx_stride as i32,
                                trans_base as i32,
                                1,
                                sp(&stream),
                            ),
                        )?;
                    }
                    if nb > 0 {
                        let wscale =
                            ((ix.hd as f64).powf(-0.5) * (ix.heads as f64).powf(-0.5)) as f32;
                        unsafe {
                            let score_rc = if self.indexer_score == Dsv4IndexerScore::Tiled {
                                k::memra_dsv4_indexer_score_tiled(
                                    dpf!(vws.qi, &stream),
                                    dpf!(ikvc.as_ref().expect("ikvc"), &stream),
                                    dpf!(vws.wproj, &stream),
                                    wscale,
                                    dpm!(vws.score, &stream),
                                    t as i32,
                                    ix.heads as i32,
                                    ix.hd as i32,
                                    nb as i32,
                                    ratio as i32,
                                    -1,
                                    pos0 as i32,
                                    sp(&stream),
                                )
                            } else {
                                k::memra_dsv4_indexer_score_f32acc_pos_m(
                                    dpf!(vws.qi, &stream),
                                    dpf!(ikvc.as_ref().expect("ikvc"), &stream),
                                    dpf!(vws.wproj, &stream),
                                    wscale,
                                    dpm!(vws.score, &stream),
                                    t as i32,
                                    ix.heads as i32,
                                    ix.hd as i32,
                                    nb as i32,
                                    ratio as i32,
                                    pos0 as i32,
                                    sp(&stream),
                                )
                            };
                            ck("indexer_score_pos_m", score_rc)?;
                            let topk_rc = if nb <= 4096 {
                                k::memra_dsv4_topk_idx_m(
                                    dpf!(vws.score, &stream),
                                    t as i32,
                                    nb as i32,
                                    ix.topk as i32,
                                    win as i32,
                                    vws.idx.device_ptr_mut(&stream).0 as *mut i32,
                                    vws.idx_stride as i32,
                                    pos0 as i32,
                                    ratio as i32,
                                    sp(&stream),
                                )
                            } else {
                                k::memra_dsv4_topk_idx_stream_m(
                                    dpf!(vws.score, &stream),
                                    t as i32,
                                    nb as i32,
                                    ix.topk as i32,
                                    win as i32,
                                    vws.idx.device_ptr_mut(&stream).0 as *mut i32,
                                    vws.idx_stride as i32,
                                    vws.topk_a.device_ptr_mut(&stream).0 as *mut u64,
                                    vws.topk_b.device_ptr_mut(&stream).0 as *mut u64,
                                    vws.topk_stride as i32,
                                    sp(&stream),
                                )
                            };
                            ck("topk_idx_m", topk_rc)?;
                        }
                    }
                }
            } else {
                let tail_max = nbs.iter().cloned().max().unwrap_or(0);
                slots = win + tail_max;
                unsafe {
                    ck(
                        "build_idx_redirect_m coarse",
                        k::memra_dsv4_build_idx_redirect_m(
                            vws.idx.device_ptr_mut(&stream).0 as *mut i32,
                            pos0 as i32,
                            t as i32,
                            win as i32,
                            ratio as i32,
                            slots as i32,
                            vws.idx_stride as i32,
                            trans_base as i32,
                            0,
                            sp(&stream),
                        ),
                    )?;
                }
            }
            // attention compressor: batched projections + position-ordered state machine
            {
                let VerifyWs {
                    x,
                    cmp_emit,
                    cmp_shift,
                    ..
                } = vws;
                self.cmp_decode_batch_dev(
                    st,
                    layer.cmp.as_ref().expect("ratio!=0 has compressor"),
                    x.device_ptr(&stream).0 as *const f32,
                    t,
                    pos0,
                    hidden,
                    &st.fc_yarn,
                    rd,
                    eps,
                    lck.cmp.as_mut().expect("cmp ckpt"),
                    cmp_emit,
                    cmp_shift,
                    pend_kv.as_mut().expect("pend"),
                    pend_score.as_mut().expect("pend"),
                    kvc,
                    win,
                    n_blocks,
                )?;
            }
            debug_assert_eq!(*n_blocks, nbs[t - 1], "attn block count (batch)");
        } else {
            unsafe {
                ck(
                    "build_idx_redirect_m window-only",
                    k::memra_dsv4_build_idx_redirect_m(
                        vws.idx.device_ptr_mut(&stream).0 as *mut i32,
                        pos0 as i32,
                        t as i32,
                        win as i32,
                        0,
                        win as i32,
                        vws.idx_stride as i32,
                        trans_base as i32,
                        0,
                        sp(&stream),
                    ),
                )?;
            }
        }

        // sparse sink attention, T queries in one launch (uniform `slots`, -1 pads —
        // bit-inert by the pinned pad contract) + per-position de-rotation
        let scale = (hd as f64).powf(-0.5) as f32;
        unsafe {
            if self.chains_f32 {
                ck(
                    "sink_attn_dec_mq_f32acc",
                    k::memra_dsv4_sink_attn_dec_mq_f32acc(
                        dpf!(vws.q, &stream),
                        dpf!(kvc, &stream),
                        vws.idx.device_ptr(&stream).0 as *const i32,
                        dpf!(layer.sink, &stream),
                        dpm!(vws.sink_scores, &stream),
                        dpm!(vws.sink_evals, &stream),
                        vws.sink_den.device_ptr_mut(&stream).0 as *mut f32,
                        dpm!(vws.o, &stream),
                        t as i32,
                        heads as i32,
                        hd as i32,
                        slots as i32,
                        vws.idx_stride as i32,
                        scale,
                        sp(&stream),
                    ),
                )?;
            } else {
                ck(
                    "sink_attn_dec_mq",
                    k::memra_dsv4_sink_attn_dec_mq(
                        dpf!(vws.q, &stream),
                        dpf!(kvc, &stream),
                        vws.idx.device_ptr(&stream).0 as *const i32,
                        dpf!(layer.sink, &stream),
                        dpm!(vws.sink_scores, &stream),
                        dpm!(vws.sink_evals, &stream),
                        vws.sink_den.device_ptr_mut(&stream).0 as *mut f64,
                        dpm!(vws.o, &stream),
                        t as i32,
                        heads as i32,
                        hd as i32,
                        slots as i32,
                        vws.idx_stride as i32,
                        scale,
                        sp(&stream),
                    ),
                )?;
            }
            ck(
                "rope o inv batch",
                k::memra_dsv4_rope(
                    dpm!(vws.o, &stream),
                    t as i32,
                    heads as i32,
                    hd as i32,
                    rd as i32,
                    fc_dev,
                    vws.pos_dev.device_ptr(&stream).0 as *const i32,
                    1,
                    sp(&stream),
                ),
            )?;
        }

        // grouped output projection: cvt o once, then per-group strided batched GEMVs
        let gw = heads / o_groups * hd;
        unsafe {
            ck(
                "cvt o batch",
                k::memra_dsv4_cvt_bf16(
                    dpf!(vws.o, &stream),
                    vws.o_b.device_ptr_mut(&stream).0 as *mut c_void,
                    (t * heads * hd) as i64,
                    sp(&stream),
                ),
            )?;
        }
        let wo_a_dw = dwsel(self.dense_fp8, &stream, &layer.wo_a, &layer.wo_a_fp8);
        for g in 0..o_groups {
            Self::gemv_m_dev(
                st,
                wo_a_dw.offset_rows(g * o_lora, gw),
                (vws.o_b.device_ptr(&stream).0 as usize + g * gw * 2) as *const c_void,
                (vws.og.device_ptr_mut(&stream).0 as usize + g * o_lora * 4) as *mut f32,
                t,
                o_lora,
                gw,
                heads * hd,
                o_groups * o_lora,
            )?;
        }
        Self::gemm_m_dev(
            st,
            vws.og.device_ptr(&stream).0 as *const f32,
            &mut vws.gemm_xb,
            dwsel(self.dense_fp8, &stream, &layer.wo_b, &layer.wo_b_fp8),
            t,
            hidden,
            o_groups * o_lora,
            vws.attn_out.device_ptr_mut(&stream).0 as *mut f32,
        )?;

        // hc_post (attention) -> vws.h_b
        unsafe {
            ck(
                "hc_post attn batch",
                k::memra_dsv4_hc_post(
                    dpf!(vws.attn_out, &stream),
                    h_in_ptr,
                    dpf!(vws.post, &stream),
                    dpf!(vws.comb, &stream),
                    dpm!(vws.h_b, &stream),
                    t as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }

        // ---- ffn sub-block (input vws.h_b, output vws.h_a)
        let h_b_ptr = vws.h_b.device_ptr(&stream).0 as *const f32;
        self.hc_pre_batch_dev(
            st,
            h_b_ptr,
            &layer.hc_ffn_fn,
            &layer.hc_ffn_base,
            &layer.hc_ffn_scale,
            &layer.hc_ffn_base_dev,
            &layer.hc_ffn_scale_dev,
            vws,
            t,
            hc,
            hidden,
            iters,
            hc_eps,
            host_math,
        )?;
        unsafe {
            ck(
                "rmsnorm ffn batch",
                self.rmsnorm_arm(
                    dpf!(vws.y_hc, &stream),
                    dpf!(layer.ffn_norm, &stream),
                    dpm!(vws.xf, &stream),
                    t as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        self.moe_verify_dev(st, layer, vws, t, toks, host_math)?;
        unsafe {
            ck(
                "hc_post ffn batch",
                k::memra_dsv4_hc_post(
                    dpf!(vws.y, &stream),
                    dpf!(vws.h_b, &stream),
                    dpf!(vws.post, &stream),
                    dpf!(vws.comb, &stream),
                    dpm!(vws.h_a, &stream),
                    t as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// Numeric composition gate on actual checkpoint weights; no serving caller.
    pub fn moe_components_for_gate(
        &self,
        il: usize,
        tokens: &[u32],
        input: &[f32],
    ) -> Res<Vec<(String, Vec<f32>)>> {
        let t = tokens.len();
        let hidden = self.model.mc.n_embd as usize;
        let topk = self.model.mc.moe.as_ref().expect("moe").expert_used_count as usize;
        if t == 0 || t > DSV4_BATCH_WIDTH_MAX || input.len() != t * hidden {
            return Err("invalid MoE gate shape".into());
        }
        let stage = *self.layer_stage.get(il).ok_or("invalid MoE gate layer")?;
        let st = &self.stages[stage];
        let layer = st
            .layers
            .iter()
            .find(|layer| layer.il as usize == il)
            .ok_or("missing gate layer")?;
        let mut state = self.alloc_prefill_state_for(t + 1, t)?;
        st.gpu.ctx.bind_to_thread().map_err(e("bind MoE gate"))?;
        let stream = st.gpu.stream();
        let ws = &mut state.ws[stage];
        stream
            .memcpy_htod(input, &mut ws.xf)
            .map_err(e("MoE gate input"))?;
        let ids: Vec<i32> = tokens.iter().map(|&id| id as i32).collect();
        stream
            .memcpy_htod(&ids, &mut ws.tok)
            .map_err(e("MoE gate tokens"))?;
        self.moe_verify_dev(st, layer, ws, t, tokens, false)?;
        let contributions = dtoh_f32(&stream, &ws.contrib)?;
        let shared = dtoh_f32(&stream, &ws.sh_out)?;
        let total = dtoh_f32(&stream, &ws.y)?;
        let mut order = vec![0i32; t * topk];
        stream
            .memcpy_dtoh(&ws.order, &mut order)
            .map_err(e("MoE gate order"))?;
        stream.synchronize().map_err(e("MoE gate synchronize"))?;
        let mut routed = vec![0f32; t * hidden];
        for row in 0..t {
            for col in 0..hidden {
                for slot in 0..topk {
                    let index = usize::try_from(order[row * topk + slot])
                        .map_err(|_| "negative gate slot")?;
                    if index >= topk {
                        return Err("invalid gate slot".into());
                    }
                    routed[row * hidden + col] +=
                        contributions[(row * topk + index) * hidden + col];
                }
            }
        }
        Ok(vec![
            ("routed".into(), routed),
            ("shared".into(), shared),
            ("total".into(), total),
        ])
    }

    /// Exact transport only. Nonrepresentable half mirrors fail before GEMM.
    fn gather_fp8_half(
        &self,
        st: &Stage,
        codes: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        row_ids: Option<&CudaSlice<i32>>,
        rows: usize,
        cols: usize,
    ) -> Res<(CudaSlice<u8>, CudaSlice<f32>)> {
        let stream = st.gpu.stream();
        let mut half = stream
            .alloc_zeros::<u8>(rows * cols * 2)
            .map_err(e("FP8 half mirror"))?;
        let mut row_scale = stream
            .alloc_zeros::<f32>(rows)
            .map_err(e("FP8 mirror scales"))?;
        let mut status = stream
            .alloc_zeros::<i32>(rows)
            .map_err(e("FP8 mirror status"))?;
        unsafe {
            ck(
                "FP8 half gather",
                k::memra_dsv4_fp8_gather_half(
                    codes.device_ptr(&stream).0 as *const c_void,
                    scales.device_ptr(&stream).0 as *const f32,
                    row_ids
                        .map(|ids| ids.device_ptr(&stream).0 as *const i32)
                        .unwrap_or(std::ptr::null()),
                    half.device_ptr_mut(&stream).0 as *mut c_void,
                    row_scale.device_ptr_mut(&stream).0 as *mut f32,
                    status.device_ptr_mut(&stream).0 as *mut i32,
                    rows as i32,
                    cols as i32,
                    sp(&stream),
                ),
            )?;
        }
        let mut check = vec![0i32; rows];
        stream
            .memcpy_dtoh(&status, &mut check)
            .map_err(e("FP8 mirror check"))?;
        stream.synchronize().map_err(e("sync FP8 mirror check"))?;
        if let Some(row) = check.iter().position(|&value| value != 0) {
            return Err(format!(
                "FP8-QAT half mirror is not lossless at gathered row {row}, cols={cols}"
            ));
        }
        Ok((half, row_scale))
    }

    /// Routed contribution only. The caller ALWAYS combines original slot order
    /// and adds the complete shared expert after either routed realization.
    #[allow(clippy::too_many_arguments)]
    fn moe_verify_grouped(
        &self,
        st: &Stage,
        layer: &LayerDev,
        vws: &mut VerifyWs,
        t: usize,
        hidden: usize,
        ne: usize,
        topk: usize,
        inter: usize,
        limit: f32,
    ) -> Res<()> {
        if crate::moe_f16g_mode() < 2 || crate::moe_f16g_sk_params().0 < 0 {
            return Err(
                "DSV4 wide-prefill grouped MoE requires MEMRA_MOE_F16G=2 with the visitor form"
                    .to_string(),
            );
        }
        if !crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT) {
            return Err(
                "DSV4 wide-prefill grouped MoE requires the direct quant loader \
                 (MEMRA_F16G_DIRECT must not be 0)"
                    .to_string(),
            );
        }
        let table = layer.experts_modelopt_table.as_ref().ok_or_else(|| {
            "DSV4 wide-prefill grouped MoE requires ModelOpt NVFP4 experts".to_string()
        })?;
        let stream = st.gpu.stream();
        let slots = t * topk;

        let mut sel_h = vec![0i32; slots];
        let mut selw_h = vec![0f32; slots];
        stream
            .memcpy_dtoh(&vws.sel.slice(0..slots), &mut sel_h)
            .map_err(e("dtoh grouped selections"))?;
        stream
            .memcpy_dtoh(&vws.selw.slice(0..slots), &mut selw_h)
            .map_err(e("dtoh grouped route weights"))?;
        stream.synchronize().map_err(e("sync grouped routing"))?;

        let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); ne];
        for (pair, &expert) in sel_h.iter().enumerate() {
            let expert =
                usize::try_from(expert).map_err(|_| format!("negative DSV4 expert id {expert}"))?;
            if expert >= ne {
                return Err(format!("DSV4 expert id {expert} outside 0..{ne}"));
            }
            buckets[expert].push(pair as i32);
        }
        let mut ex_ids = Vec::new();
        let mut ex_off = vec![0i32];
        let mut ex_pairs = Vec::with_capacity(slots);
        for (expert, pairs) in buckets.iter().enumerate() {
            if pairs.is_empty() {
                continue;
            }
            ex_ids.push(expert as i32);
            ex_pairs.extend_from_slice(pairs);
            ex_off.push(ex_pairs.len() as i32);
        }
        let n_active = ex_ids.len();
        let max_m = ex_off.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        let csr_tok: Vec<i32> = ex_pairs.iter().map(|&p| p / topk as i32).collect();
        let pair_w: Vec<f32> = ex_pairs.iter().map(|&p| selw_h[p as usize]).collect();
        let macro_for = |proj: usize| -> Vec<f32> {
            ex_pairs
                .iter()
                .map(|&p| layer.experts_s2[sel_h[p as usize] as usize * 3 + proj])
                .collect()
        };
        let (macro_w1, macro_w2, macro_w3) = (macro_for(0), macro_for(1), macro_for(2));
        let ex_ids_d = upload_i32(&stream, &ex_ids)?;
        let ex_off_d = upload_i32(&stream, &ex_off)?;
        let ex_pairs_d = upload_i32(&stream, &ex_pairs)?;
        let csr_tok_d = upload_i32(&stream, &csr_tok)?;
        let pair_w_d = upload_f32(&stream, &pair_w)?;
        let macro_w1_d = upload_f32(&stream, &macro_w1)?;
        let macro_w2_d = upload_f32(&stream, &macro_w2)?;
        let macro_w3_d = upload_f32(&stream, &macro_w3)?;

        let rc = unsafe { memra_bind_device(st.dev as i32) };
        if rc != 0 {
            return Err(format!(
                "DSV4 grouped prefill cudaSetDevice({}) rc={rc}",
                st.dev
            ));
        }
        unsafe {
            ck(
                "grouped FP8-QAT x",
                k::memra_dsv4_act_quant_fp8(
                    dpf!(vws.xf, &stream),
                    vws.xq.device_ptr_mut(&stream).0 as *mut c_void,
                    dpm!(vws.xs, &stream),
                    t as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        let (act16, act_scale) =
            self.gather_fp8_half(st, &vws.xq, &vws.xs, Some(&csr_tok_d), slots, hidden)?;
        let (shape_sel, cross) = crate::moe_f16g_sk_params();
        let launch = |proj: i32,
                      input: &CudaSlice<u8>,
                      input_scale: &CudaSlice<f32>,
                      in_f: usize,
                      out_f: usize,
                      out: &mut CudaSlice<f32>|
         -> Res<()> {
            let rc = unsafe {
                memra_moe_kq_gemm_sk(
                    table.device_ptr(&stream).0 as *const u64,
                    proj,
                    ne as i32,
                    ex_ids_d.device_ptr(&stream).0 as *const i32,
                    input.device_ptr(&stream).0 as *const c_void,
                    out.device_ptr_mut(&stream).0 as *mut f32,
                    input_scale.device_ptr(&stream).0 as *const f32,
                    ex_off_d.device_ptr(&stream).0 as *const i32,
                    ex_off.as_ptr(),
                    n_active as i32,
                    max_m,
                    in_f as i32,
                    out_f as i32,
                    crate::QT_NVFP4_MODELOPT,
                    cross,
                    crate::moe_f16g_tail_on() as i32,
                    (in_f / 2) as i64,
                    sp(&stream),
                )
            };
            if rc == 0 {
                Ok(())
            } else {
                Err(format!("DSV4 grouped projection {proj} failed rc={rc}"))
            }
        };
        launch(0, &act16, &act_scale, hidden, inter, &mut vws.g1)?;
        launch(2, &act16, &act_scale, hidden, inter, &mut vws.g3)?;
        unsafe {
            ck(
                "grouped scale w1",
                k::memra_dsv4_scale_rows(
                    dpm!(vws.g1, &stream),
                    dpf!(macro_w1_d, &stream),
                    slots as i32,
                    inter as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "grouped scale w3",
                k::memra_dsv4_scale_rows(
                    dpm!(vws.g3, &stream),
                    dpf!(macro_w3_d, &stream),
                    slots as i32,
                    inter as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "grouped swiglu",
                k::memra_dsv4_swiglu(
                    dpf!(vws.g1, &stream),
                    dpf!(vws.g3, &stream),
                    dpm!(vws.hbuf, &stream),
                    slots as i32,
                    inter as i32,
                    limit,
                    dpf!(pair_w_d, &stream),
                    sp(&stream),
                ),
            )?;
        }
        unsafe {
            ck(
                "grouped FP8-QAT h",
                k::memra_dsv4_act_quant_fp8(
                    dpf!(vws.hbuf, &stream),
                    vws.hq.device_ptr_mut(&stream).0 as *mut c_void,
                    dpm!(vws.hs, &stream),
                    slots as i32,
                    inter as i32,
                    sp(&stream),
                ),
            )?;
        }
        let (h16, h_scale) = self.gather_fp8_half(st, &vws.hq, &vws.hs, None, slots, inter)?;
        let mut grouped_contrib = stream
            .alloc_zeros::<f32>(slots * hidden)
            .map_err(e("grouped CSR contribution"))?;
        launch(1, &h16, &h_scale, inter, hidden, &mut grouped_contrib)?;
        unsafe {
            ck(
                "grouped scale w2",
                k::memra_dsv4_scale_rows(
                    dpm!(grouped_contrib, &stream),
                    dpf!(macro_w2_d, &stream),
                    slots as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        unsafe {
            ck(
                "grouped csr permute",
                k::memra_dsv4_scatter_rows(
                    dpf!(grouped_contrib, &stream),
                    dpm!(vws.contrib, &stream),
                    ex_pairs_d.device_ptr(&stream).0 as *const i32,
                    slots as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        }
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[dsv4-prefill-f16g] ENGAGED: t={t} pairs={slots} active_experts={n_active} \
                 max_rows_per_expert={max_m} FP8-QAT-mirrored half, split-plane ModelOpt NVFP4"
            );
        }
        let _ = shape_sel;
        Ok(())
    }

    /// MoE for T rows: per-position routing (the hash layers need the per-position TOKEN,
    /// which is why a round carries a token array), then ONE launch per projection over
    /// the whole T x topk slot set — routed-expert weight traffic scales with T (each
    /// position's experts are its own) while the shared expert and the gate amortize.
    fn moe_verify_dev(
        &self,
        st: &Stage,
        layer: &LayerDev,
        vws: &mut VerifyWs,
        t: usize,
        toks: &[u32],
        host_math: bool,
    ) -> Res<()> {
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let moe = mc.moe.as_ref().expect("moe");
        let hidden = mc.n_embd as usize;
        let ne = moe.expert_count as usize;
        let topk = moe.expert_used_count as usize;
        let inter = moe.expert_ff_length as usize;
        let limit = d.swiglu_limit;
        let stream = st.gpu.stream();
        let kind = match layer.expert_kind {
            ExpertKind::Nvfp4 => 0i32,
            ExpertKind::Mxfp4 => 1i32,
        };
        let wstride = (inter * hidden / 2) as i64;
        let sstride = match layer.expert_kind {
            ExpertKind::Nvfp4 => (inter * hidden / 16) as i64,
            ExpertKind::Mxfp4 => (inter * hidden / 32) as i64,
        };
        let slots = t * topk;

        self.dots_m_dev(
            st,
            vws.xf.device_ptr(&stream).0 as *const f32,
            layer.gate_w.device_ptr(&stream).0 as *const c_void,
            0,
            t,
            hidden,
            ne,
            vws.raw.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        if host_math {
            let raw_h = {
                let view = vws.raw.slice(0..t * ne);
                let mut v = vec![0f32; t * ne];
                stream
                    .memcpy_dtoh(&view, &mut v[..])
                    .map_err(e("dtoh raw b"))?;
                stream.synchronize().map_err(e("sync raw b"))?;
                v
            };
            let (indices, weights) =
                Self::route_host(layer, &raw_h, toks, t, ne, topk, d.routed_scaling_factor);
            let sel: Vec<i32> = indices.iter().map(|&x| x as i32).collect();
            let mut order = vec![0i32; t * topk];
            for p in 0..t {
                let mut o: Vec<i32> = (0..topk as i32).collect();
                o.sort_by_key(|&s| indices[p * topk + s as usize]);
                order[p * topk..(p + 1) * topk].copy_from_slice(&o);
            }
            let mut dst = vws.sel.slice_mut(0..t * topk);
            stream
                .memcpy_htod(&sel, &mut dst)
                .map_err(e("htod sel b"))?;
            let mut dst = vws.selw.slice_mut(0..t * topk);
            stream
                .memcpy_htod(&weights, &mut dst)
                .map_err(e("htod selw b"))?;
            let mut dst = vws.order.slice_mut(0..t * topk);
            stream
                .memcpy_htod(&order, &mut dst)
                .map_err(e("htod order b"))?;
        } else {
            unsafe {
                ck(
                    "route_m",
                    k::memra_dsv4_route_m(
                        dpf!(vws.raw, &stream),
                        layer
                            .gate_bias_dev
                            .as_ref()
                            .map(|b| b.device_ptr(&stream).0 as *const f32)
                            .unwrap_or(std::ptr::null()),
                        layer
                            .tid2eid_dev
                            .as_ref()
                            .map(|x| x.device_ptr(&stream).0 as *const i32)
                            .unwrap_or(std::ptr::null()),
                        vws.tok.device_ptr(&stream).0 as *const i32,
                        t as i32,
                        ne as i32,
                        topk as i32,
                        d.routed_scaling_factor,
                        vws.sel.device_ptr_mut(&stream).0 as *mut i32,
                        vws.selw.device_ptr_mut(&stream).0 as *mut f32,
                        vws.order.device_ptr_mut(&stream).0 as *mut i32,
                        sp(&stream),
                    ),
                )?;
            }
        }

        if self.prefill_grouped && vws.is_prefill && layer.expert_kind == ExpertKind::Nvfp4 {
            self.moe_verify_grouped(st, layer, vws, t, hidden, ne, topk, inter, limit)?;
        } else {
            unsafe {
                ck(
                    "act_quant_fp8 x batch",
                    k::memra_dsv4_act_quant_fp8(
                        dpf!(vws.xf, &stream),
                        vws.xq.device_ptr_mut(&stream).0 as *mut c_void,
                        dpm!(vws.xs, &stream),
                        t as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
                for (proj, dst) in [(0i32, &mut vws.g1), (2i32, &mut vws.g3)] {
                    ck(
                        "fp4_gemm_sel_g w1/w3",
                        k::memra_dsv4_fp4_gemm_sel_g_arm(
                            dp!(vws.xq, &stream),
                            dpf!(vws.xs, &stream),
                            dp!(layer.experts_w, &stream),
                            dp!(layer.experts_sc, &stream),
                            dpf!(layer.experts_s2_dev, &stream),
                            vws.sel.device_ptr(&stream).0 as *const i32,
                            proj,
                            0,
                            kind,
                            dpm!(*dst, &stream),
                            slots as i32,
                            inter as i32,
                            hidden as i32,
                            wstride,
                            sstride,
                            topk as i32,
                            self.fp4_reduce as i32,
                            sp(&stream),
                        ),
                    )?;
                }
                ck(
                    "swiglu batch",
                    k::memra_dsv4_swiglu(
                        dpf!(vws.g1, &stream),
                        dpf!(vws.g3, &stream),
                        dpm!(vws.hbuf, &stream),
                        slots as i32,
                        inter as i32,
                        limit,
                        vws.selw.device_ptr(&stream).0 as *const f32,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "act_quant_fp8 h batch",
                    k::memra_dsv4_act_quant_fp8(
                        dpf!(vws.hbuf, &stream),
                        vws.hq.device_ptr_mut(&stream).0 as *mut c_void,
                        dpm!(vws.hs, &stream),
                        slots as i32,
                        inter as i32,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "fp4_gemm_sel_g w2",
                    k::memra_dsv4_fp4_gemm_sel_g_arm(
                        dp!(vws.hq, &stream),
                        dpf!(vws.hs, &stream),
                        dp!(layer.experts_w, &stream),
                        dp!(layer.experts_sc, &stream),
                        dpf!(layer.experts_s2_dev, &stream),
                        vws.sel.device_ptr(&stream).0 as *const i32,
                        1,
                        1,
                        kind,
                        dpm!(vws.contrib, &stream),
                        slots as i32,
                        hidden as i32,
                        inter as i32,
                        wstride,
                        sstride,
                        0,
                        self.fp4_reduce as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
        // Both routed arms produce slot-aligned contributions. The combine and
        // entire shared expert are mandatory common work, never an early return.
        unsafe {
            ck(
                "combine_rows_m",
                k::memra_dsv4_combine_rows_m(
                    dpf!(vws.contrib, &stream),
                    vws.order.device_ptr(&stream).0 as *const i32,
                    topk as i32,
                    dpm!(vws.y, &stream),
                    hidden as i64,
                    t as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "cvt xb batch",
                k::memra_dsv4_cvt_bf16(
                    dpf!(vws.xf, &stream),
                    vws.xb.device_ptr_mut(&stream).0 as *mut c_void,
                    (t * hidden) as i64,
                    sp(&stream),
                ),
            )?;
        }
        let sh_inter = vws.sg1.len() / vws.tmax;
        Self::gemv_m_dev(
            st,
            dwsel(
                self.dense_fp8,
                &stream,
                &layer.shared_w[0],
                &layer.shared_fp8[0],
            ),
            vws.xb.device_ptr(&stream).0 as *const c_void,
            vws.sg1.device_ptr_mut(&stream).0 as *mut f32,
            t,
            sh_inter,
            hidden,
            0,
            0,
        )?;
        Self::gemv_m_dev(
            st,
            dwsel(
                self.dense_fp8,
                &stream,
                &layer.shared_w[2],
                &layer.shared_fp8[2],
            ),
            vws.xb.device_ptr(&stream).0 as *const c_void,
            vws.sg3.device_ptr_mut(&stream).0 as *mut f32,
            t,
            sh_inter,
            hidden,
            0,
            0,
        )?;
        unsafe {
            ck(
                "swiglu sh batch",
                k::memra_dsv4_swiglu(
                    dpf!(vws.sg1, &stream),
                    dpf!(vws.sg3, &stream),
                    dpm!(vws.shbuf, &stream),
                    t as i32,
                    sh_inter as i32,
                    limit,
                    std::ptr::null(),
                    sp(&stream),
                ),
            )?;
            ck(
                "cvt sh batch",
                k::memra_dsv4_cvt_bf16(
                    dpf!(vws.shbuf, &stream),
                    vws.shb16.device_ptr_mut(&stream).0 as *mut c_void,
                    (t * sh_inter) as i64,
                    sp(&stream),
                ),
            )?;
        }
        Self::gemv_m_dev(
            st,
            dwsel(
                self.dense_fp8,
                &stream,
                &layer.shared_w[1],
                &layer.shared_fp8[1],
            ),
            vws.shb16.device_ptr(&stream).0 as *const c_void,
            vws.sh_out.device_ptr_mut(&stream).0 as *mut f32,
            t,
            hidden,
            sh_inter,
            0,
            0,
        )?;
        unsafe {
            ck(
                "add shared batch",
                k::memra_dsv4_add_inplace(
                    dpm!(vws.y, &stream),
                    dpf!(vws.sh_out, &stream),
                    (t * hidden) as i64,
                    sp(&stream),
                ),
            )?;
        }
        Ok(())
    }

    /// Head for T rows: the `head_logits_dev` program with the row count, and the vocab
    /// dots on the batched island kernel so the 1.06 GiB head slab is read ONCE per round
    /// instead of once per verified position.
    fn head_logits_batch_dev(&self, vws: &mut VerifyWs, t: usize, host_math: bool) -> Res<()> {
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hc = d.hc_mult as usize;
        let hidden = mc.n_embd as usize;
        let eps = mc.rms_eps;
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        let stream = st.gpu.stream();
        let w = hc * hidden;
        let fn_w = st.hc_head_fn.as_ref().expect("hc_head_fn");
        let norm = st.trunk_norm.as_ref().expect("trunk norm");
        let vocab = vws.logits.len() / vws.tmax;
        self.dots_m_dev(
            st,
            vws.h_a.device_ptr(&stream).0 as *const f32,
            fn_w.device_ptr(&stream).0 as *const c_void,
            0,
            t,
            w,
            hc,
            vws.head_mixes.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        unsafe {
            ck(
                "rowsq head batch",
                self.rowsq_scale_arm(
                    dpf!(vws.h_a, &stream),
                    dpm!(vws.head_mixes, &stream),
                    t as i32,
                    w as i32,
                    hc as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        if host_math {
            let mut mixes_h = vec![0f32; t * hc];
            let view = vws.head_mixes.slice(0..t * hc);
            stream
                .memcpy_dtoh(&view, &mut mixes_h[..])
                .map_err(e("dtoh head mixes b"))?;
            stream.synchronize().map_err(e("sync head mixes b"))?;
            for p in 0..t {
                for c in 0..hc {
                    let m = mixes_h[p * hc + c];
                    mixes_h[p * hc + c] =
                        sigmoid_f32(m * self.hc_head_scale[0] + self.hc_head_base[c]) + d.hc_eps;
                }
            }
            let mut dst = vws.head_pre.slice_mut(0..t * hc);
            stream
                .memcpy_htod(&mixes_h, &mut dst)
                .map_err(e("htod head pre b"))?;
        } else {
            unsafe {
                ck(
                    "hc_head_pre_m",
                    k::memra_dsv4_hc_head_pre_m(
                        dpf!(vws.head_mixes, &stream),
                        st.hc_head_scale_dev
                            .as_ref()
                            .expect("head scale dev")
                            .device_ptr(&stream)
                            .0 as *const f32,
                        st.hc_head_base_dev
                            .as_ref()
                            .expect("head base dev")
                            .device_ptr(&stream)
                            .0 as *const f32,
                        dpm!(vws.head_pre, &stream),
                        t as i32,
                        hc as i32,
                        d.hc_eps,
                        sp(&stream),
                    ),
                )?;
            }
        }
        unsafe {
            ck(
                "hc_collapse head batch",
                k::memra_dsv4_hc_collapse(
                    dpf!(vws.h_a, &stream),
                    dpf!(vws.head_pre, &stream),
                    dpm!(vws.collapsed, &stream),
                    t as i32,
                    hc as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
            ck(
                "rmsnorm head batch",
                self.rmsnorm_arm(
                    dpf!(vws.collapsed, &stream),
                    dpf!(norm, &stream),
                    dpm!(vws.collapsed, &stream),
                    t as i32,
                    hidden as i32,
                    eps,
                    sp(&stream),
                ),
            )?;
        }
        let head_ptr = st.head.as_ref().expect("head").device_ptr(&stream).0 as *const c_void;
        self.dots_m_dev(
            st,
            vws.collapsed.device_ptr(&stream).0 as *const f32,
            head_ptr,
            1,
            t,
            hidden,
            vocab,
            vws.logits.device_ptr_mut(&stream).0 as *mut f32,
        )?;
        Ok(())
    }
}

/// One verify round's bookkeeping (the device twin of `spec_oracle::SpecRound`).
pub struct SpecRoundGpu {
    pub start_pos: usize,
    pub drafts: Vec<u32>,
    pub accepts: usize,
    pub verified: usize,
    /// batch depth actually forwarded this round (T = 1 + verifiable drafts)
    pub t_batch: usize,
    /// STRUCTURAL depth ceiling for this round: min(k_drafts + 1, MEMRA_DSV4_SPEC_DEPTH,
    /// vstate.tmax) -- i.e. `t_batch` before the n_new budget is applied. `t_batch < t_cap`
    /// is exactly "the budget truncated this round", which is what `carry_pending` keys on.
    pub t_cap: usize,
    /// The drafter's fp32 per-slot confidence for this round's proposal (pre-sigmoid
    /// logits; the head is supervised on c* = 1 - TV, i.e. conditional acceptance
    /// probability). Banked per round so the DSpark Algorithm-1 scheduler can be scored
    /// offline against measured round costs -- never consumed by the round itself.
    pub confidence: Vec<f32>,
    /// tokens this round contributed to the output stream (head + accepted drafts)
    pub emitted: usize,
    /// wall time of the whole round — proposal, batched verify, commit/rollback, drafter
    /// ring advance — with the drafter stream synchronized at the round boundary so no
    /// work leaks into the next round's measurement. The A/B instrument.
    pub round_us: u64,
}

pub struct SpecRunGpu {
    pub tokens: Vec<u32>,
    pub rounds: Vec<SpecRoundGpu>,
}

impl Dsv4Gpu {
    /// Batched T=k+1 verify forward (§3.1): ONE trunk pass over `toks` at positions
    /// state.pos .. state.pos+T-1, logits for EVERY position (the accept walk needs them
    /// all), state advanced PROVISIONALLY for all T. Exactly one
    /// [`Self::commit_verify_dev`] must follow, which makes the accepted prefix permanent
    /// and rolls the rest back. The DSpark trunk tap is written for all T rows when
    /// `taps` is Some (rows 0..T-1 of the drafter's taps buffer).
    ///
    /// Returns (logits `[T, vocab]` when `want_logits`, per-position argmax `[T]`).
    pub fn verify_batch_dev(
        &self,
        toks: &[u32],
        state: &mut DecodeState,
        vstate: &mut VerifyState,
        taps: Option<&mut CudaSlice<f32>>,
        want_logits: bool,
    ) -> Res<(Option<Vec<f32>>, Vec<u32>)> {
        self.verify_batch_dev_output(
            toks,
            state,
            vstate,
            taps,
            if want_logits {
                VerifyOutput::Full
            } else {
                VerifyOutput::Argmax
            },
        )
    }

    /// The public verifier retains its full-logits/argmax contract. Only the
    /// prefill caller may discard intermediate output or request its final row.
    fn verify_batch_dev_output(
        &self,
        toks: &[u32],
        state: &mut DecodeState,
        vstate: &mut VerifyState,
        taps: Option<&mut CudaSlice<f32>>,
        output: VerifyOutput,
    ) -> Res<(Option<Vec<f32>>, Vec<u32>)> {
        let DecodePath::Device { host_math } = self.decode_path else {
            return Err("verify_batch_dev requires MEMRA_DSV4_DECODE_PATH=device".into());
        };
        let mc = &self.model.mc;
        let d = self.model.cfg();
        let t = toks.len();
        assert!(
            t >= 1 && t <= vstate.tmax,
            "round depth {t} > tmax {}",
            vstate.tmax
        );
        assert!(vstate.open.is_none(), "verify_batch_dev with an open round");
        if vstate.capacity != state.capacity {
            return Err(format!(
                "dsv4 verify capacity {} != decode-state capacity {}; allocate both for the same session admission",
                vstate.capacity, state.capacity
            ));
        }
        if vstate.tmax > state.transient_rows {
            return Err(format!(
                "dsv4 transaction width {} exceeds decode-state transient rows {}; allocate the state for the same or wider transaction",
                vstate.tmax, state.transient_rows
            ));
        }
        let pos0 = state.pos;
        assert!(pos0 > 0, "batched verify needs prefill_with_cache first");
        assert!(
            pos0 + t <= state.capacity,
            "round [{pos0}, {}) exceeds session capacity {}",
            pos0 + t,
            state.capacity
        );
        let hidden = mc.n_embd as usize;
        let hc = d.hc_mult as usize;
        let n_trunk = (mc.n_layer - mc.nextn_predict_layers) as usize;
        let tok_i32: Vec<i32> = toks.iter().map(|&x| x as i32).collect();
        let pos_i32: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();

        // per-stage round constants (the hash layers read the token array; every layer's
        // ropes read the position array — both live on whichever stage the layer does)
        for (si, st) in self.stages.iter().enumerate() {
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx round"))?;
            let stream = st.gpu.stream();
            let vws = &mut vstate.ws[si];
            let mut dst = vws.tok.slice_mut(0..t);
            stream
                .memcpy_htod(&tok_i32, &mut dst)
                .map_err(e("htod tok round"))?;
            let mut dst = vws.pos_dev.slice_mut(0..t);
            stream
                .memcpy_htod(&pos_i32, &mut dst)
                .map_err(e("htod pos round"))?;
        }

        // stage 0: tokens -> embed rows -> hc state
        {
            let st0 = &self.stages[0];
            st0.gpu.ctx.bind_to_thread().map_err(e("bind ctx0 round"))?;
            let stream0 = st0.gpu.stream();
            let vws0 = &mut vstate.ws[0];
            unsafe {
                ck(
                    "embed_rows batch",
                    k::memra_dsv4_embed_rows(
                        st0.embed
                            .as_ref()
                            .expect("embed on stage 0")
                            .device_ptr(&stream0)
                            .0 as *const c_void,
                        vws0.tok.device_ptr(&stream0).0 as *const i32,
                        dpm!(vws0.emb, &stream0),
                        t as i32,
                        hidden as i32,
                        sp(&stream0),
                    ),
                )?;
                ck(
                    "repeat_hc batch",
                    k::memra_dsv4_repeat_hc(
                        dpf!(vws0.emb, &stream0),
                        dpm!(vws0.h_a, &stream0),
                        t as i32,
                        hc as i32,
                        hidden as i32,
                        sp(&stream0),
                    ),
                )?;
            }
        }

        let targets = self.dspark.as_ref().map(|ds| ds.targets.clone());
        let n_t = targets.as_ref().map(|x| x.len()).unwrap_or(0);
        let mut taps = taps;
        let mut cur_stage = 0usize;
        let mut input_rx = false;
        for il in 0..n_trunk {
            let stage = self.layer_stage[il];
            if stage != cur_stage {
                let bytes = t * hc * hidden * std::mem::size_of::<f32>();
                let src_stream = self.stages[cur_stage].gpu.stream();
                let dst_stream = self.stages[stage].gpu.stream();
                let (ws_src, ws_dst) = vstate.ws.split_at_mut(stage);
                let src_ws = &ws_src[cur_stage];
                let dst_ws = &mut ws_dst[0];
                self.stages[cur_stage]
                    .gpu
                    .ctx
                    .bind_to_thread()
                    .map_err(e("bind tx round"))?;
                let (sp_, _g0) = src_ws.h_a.device_ptr(&src_stream);
                let (dp_, _g1) = dst_ws.h_rx.device_ptr_mut(&src_stream);
                unsafe {
                    cudarc::driver::result::memcpy_peer_async(
                        self.stages[stage].gpu.ctx.cu_ctx(),
                        dp_,
                        self.stages[cur_stage].gpu.ctx.cu_ctx(),
                        sp_,
                        bytes,
                        src_stream.cu_stream(),
                    )
                    .map_err(e("peer copy h round"))?;
                }
                let bnd = stage - 1;
                self.boundary_ev[bnd]
                    .record(&src_stream)
                    .map_err(e("ev record round"))?;
                dst_stream
                    .wait(&self.boundary_ev[bnd])
                    .map_err(e("ev wait round"))?;
                self.stages[stage]
                    .gpu
                    .ctx
                    .bind_to_thread()
                    .map_err(e("bind rx round"))?;
                cur_stage = stage;
                input_rx = true;
            }
            let st = &self.stages[stage];
            let lidx = st
                .layers
                .iter()
                .position(|l| l.il == il as u32)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage}"));
            self.block_verify_dev(
                st,
                &st.layers[lidx],
                &mut state.caches[il],
                &mut vstate.layers[il],
                &mut vstate.ws[stage],
                input_rx,
                pos0,
                t,
                toks,
                host_math,
            )?;
            input_rx = false;
            // DSpark trunk tap for all T rows (capture only)
            if let (Some(tp), Some(tg)) = (taps.as_mut(), targets.as_ref())
                && let Some(kk) = tg.iter().position(|&tl| tl == il)
            {
                let stream = self.stages[stage].gpu.stream();
                let vws = &mut vstate.ws[stage];
                unsafe {
                    ck(
                        "hc_mean tap batch",
                        k::memra_dsv4_hc_mean(
                            dpf!(vws.h_a, &stream),
                            dpm!(vws.tap_tmp, &stream),
                            t as i32,
                            hc as i32,
                            hidden as i32,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "place_cols tap batch",
                        k::memra_dsv4_place_cols(
                            dpf!(vws.tap_tmp, &stream),
                            dpm!(**tp, &stream),
                            t as i32,
                            hidden as i32,
                            (n_t * hidden) as i64,
                            (kk * hidden) as i64,
                            sp(&stream),
                        ),
                    )?;
                }
            }
        }

        let last = self.stages.len() - 1;
        assert_eq!(cur_stage, last, "device path expects the head stage last");
        if matches!(output, VerifyOutput::None | VerifyOutput::Last) {
            assert!(
                vstate.ws[last].is_prefill,
                "prefill output policy used for verification"
            );
            if output == VerifyOutput::None {
                self.prefill_head_counts
                    .skipped_chunks
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                vstate.open = Some((pos0, t));
                return Ok((None, Vec::new()));
            }
            let stream = self.stages[last].gpu.stream();
            let row_size = hc * hidden;
            let final_h = vstate.ws[last].h_a.slice((t - 1) * row_size..t * row_size);
            let step = &mut state.ws.as_mut().expect("device scratch")[last];
            stream
                .memcpy_dtod(&final_h, &mut step.h_a)
                .map_err(e("prefill final head row"))?;
            self.head_logits_dev(step, host_math)?;
            let logits = dtoh_f32(&stream, &step.logits)?;
            self.prefill_head_counts
                .last_rows
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            vstate.open = Some((pos0, t));
            return Ok((Some(logits), Vec::new()));
        }
        self.head_logits_batch_dev(&mut vstate.ws[last], t, host_math)?;
        if vstate.ws[last].is_prefill {
            self.prefill_head_counts
                .full_rows
                .fetch_add(t as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let stream_last = self.stages[last].gpu.stream();
        let vws = &mut vstate.ws[last];
        let vocab = vws.logits.len() / vws.tmax;
        let logits = if output == VerifyOutput::Full {
            let mut v = vec![0f32; t * vocab];
            let view = vws.logits.slice(0..t * vocab);
            stream_last
                .memcpy_dtoh(&view, &mut v[..])
                .map_err(e("dtoh logits batch"))?;
            stream_last.synchronize().map_err(e("sync logits batch"))?;
            Some(v)
        } else {
            None
        };
        let mut am = vec![0i32; t];
        if let Some(lg) = &logits {
            for (i, slot) in am.iter_mut().enumerate() {
                let row = &lg[i * vocab..(i + 1) * vocab];
                let mut best = 0usize;
                for j in 1..vocab {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                *slot = best as i32;
            }
        } else {
            unsafe {
                for i in 0..t {
                    ck(
                        "argmax batch",
                        k::memra_dsv4_argmax(
                            (vws.logits.device_ptr(&stream_last).0 as usize + i * vocab * 4)
                                as *const f32,
                            vocab as i64,
                            (vws.argmax.device_ptr_mut(&stream_last).0 as usize + i * 4)
                                as *mut i32,
                            sp(&stream_last),
                        ),
                    )?;
                }
            }
            let view = vws.argmax.slice(0..t);
            stream_last
                .memcpy_dtoh(&view, &mut am[..])
                .map_err(e("dtoh argmax batch"))?;
            stream_last.synchronize().map_err(e("sync argmax batch"))?;
        }
        vstate.open = Some((pos0, t));
        Ok((logits, am.into_iter().map(|x| x as u32).collect()))
    }

    /// Commit the first `n_commit` positions of the open round and roll the rest back
    /// (§3.1 invariant: every trunk cache class ends bit-identical to plain sequential
    /// decode of exactly the committed positions). Ring slots take their transient rows;
    /// the compressors replay; the append-only stores fall back to their high-water mark.
    pub fn commit_verify_dev(
        &self,
        state: &mut DecodeState,
        vstate: &mut VerifyState,
        n_commit: usize,
    ) -> Res<()> {
        let (pos0, t) = vstate
            .open
            .take()
            .ok_or_else(|| "commit_verify_dev without an open round".to_string())?;
        assert!(
            n_commit >= 1 && n_commit <= t,
            "commit {n_commit} outside round width {t}"
        );
        let d = self.model.cfg();
        let mc = &self.model.mc;
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let n_trunk = (mc.n_layer - mc.nextn_predict_layers) as usize;
        let (ring_start, slot_rows) = ring_commit_plan(pos0, n_commit, win);
        let ring_keep = slot_rows.len();
        for il in 0..n_trunk {
            let stage = self.layer_stage[il];
            let st = &self.stages[stage];
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx commit"))?;
            let stream = st.gpu.stream();
            let lck = &mut vstate.layers[il];
            let vws = &mut vstate.ws[stage];
            let cache = &mut state.caches[il];
            let trans_base = lck.trans_base;
            // Only the newest window survives. Scattering every committed row
            // when n_commit>win races multiple writers to the same ring slot.
            {
                let src = cache
                    .kvc
                    .slice((trans_base + ring_start) * hd..(trans_base + n_commit) * hd);
                let mut dst = vws.bounce.slice_mut(0..ring_keep * hd);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(e("commit bounce"))?;
            }
            {
                let mut dst = vws.slot_rows.slice_mut(0..ring_keep);
                stream
                    .memcpy_htod(&slot_rows, &mut dst)
                    .map_err(e("htod slot rows"))?;
            }
            unsafe {
                ck(
                    "scatter_rows commit",
                    k::memra_dsv4_scatter_rows(
                        dpf!(vws.bounce, &stream),
                        dpm!(cache.kvc, &stream),
                        vws.slot_rows.device_ptr(&stream).0 as *const i32,
                        ring_keep as i32,
                        hd as i32,
                        sp(&stream),
                    ),
                )?;
            }
            if let Some(ckd) = &lck.cmp {
                self.cmp_rollback_replay_dev(
                    st,
                    ckd,
                    n_commit,
                    t,
                    pos0,
                    &mut vws.cmp_shift,
                    cache.pend_kv.as_mut().expect("pend kv"),
                    cache.pend_score.as_mut().expect("pend sc"),
                    &mut cache.n_blocks,
                )?;
            }
            if let Some(ckd) = &lck.idx {
                self.cmp_rollback_replay_dev(
                    st,
                    ckd,
                    n_commit,
                    t,
                    pos0,
                    &mut vws.cmp_shift,
                    cache.ipend_kv.as_mut().expect("ipend kv"),
                    cache.ipend_score.as_mut().expect("ipend sc"),
                    &mut cache.i_blocks,
                )?;
            }
        }
        for st in &self.stages {
            st.gpu
                .ctx
                .bind_to_thread()
                .map_err(e("bind ctx commit sync"))?;
            st.gpu.stream().synchronize().map_err(e("commit sync"))?;
        }
        state.pos = pos0 + n_commit;
        Ok(())
    }

    /// The device propose-then-verify greedy loop with BATCHED verification — the
    /// engine-side twin of `spec_oracle::run_spec_greedy_batched`, including its
    /// round/budget accounting (the budget-truncated final round and its pending-carry
    /// no-propose tail), so proposal streams and token streams are comparable
    /// item-for-item with the CPU oracle's.
    ///
    /// Greedy law: the trunk's own argmax is ALWAYS the emitted token, so the output
    /// stream is plain greedy by construction — and because every batched kernel on this
    /// path is bit-exact against its single-position twin, that identity is byte-exact on
    /// device too, not merely mathematical.
    /// Reads the `MEMRA_DSV4_SPEC_DEPTH` knob and delegates to
    /// [`Self::spec_greedy_batched_depth`]. Every existing gate and bench calls this form,
    /// so their behaviour is decided by the environment exactly as before.
    pub fn spec_greedy_batched_with(
        &self,
        prompt: &[u32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
    ) -> Res<SpecRunGpu> {
        // MEMRA_DSV4_SPEC_DEPTH=T: structural cap on the batched verify depth (T rows =
        // 1 head + T-1 verified drafts). Unset or 0 => no cap, which reproduces the
        // pre-knob driver exactly. Clamped to >= 1 so a typo cannot ask for a zero-row
        // verify.
        let depth_cap = std::env::var("MEMRA_DSV4_SPEC_DEPTH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|t| *t > 0)
            .unwrap_or(usize::MAX)
            .max(1);
        if depth_cap != usize::MAX {
            println!("[spec] verify depth capped at T={depth_cap} (MEMRA_DSV4_SPEC_DEPTH)");
        }
        self.spec_greedy_batched_depth(prompt, n_new, state, dstate, vstate, depth_cap)
    }

    /// [`Self::spec_greedy_batched_with`] with the verify-depth ceiling passed explicitly.
    /// `usize::MAX` means "no cap" (the drafter's own `block_size + 1`).
    ///
    /// Greedy identity is preserved at every cap by construction: truncating the proposal
    /// only shortens the accepted prefix, and the head token of every round is the trunk's
    /// own argmax. That is what makes a depth sweep measurable without re-earning the
    /// identity law at each rung -- though the sweep still asserts it per arm.
    pub fn spec_greedy_batched_depth(
        &self,
        prompt: &[u32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
    ) -> Res<SpecRunGpu> {
        // ds4f rung 1: confidence-window policy, read once per run (see resolve_vt).
        // Off reproduces the pre-policy t_cap expression exactly (vt_drafts == k_drafts).
        let vt = resolve_vt(
            std::env::var("MEMRA_DSV4_VT").ok().as_deref(),
            std::env::var("MEMRA_DSV4_VT_TAU").ok().as_deref(),
            std::env::var("MEMRA_DSV4_VT_FLOOR").ok().as_deref(),
        )?;
        self.spec_greedy_batched_policy(prompt, n_new, state, dstate, vstate, depth_cap, vt)
    }

    /// [`Self::spec_greedy_batched_depth`] with the vt policy passed EXPLICITLY — the
    /// in-process multi-arm sweep entry (one load, one thermal window; the env seam
    /// stays the serving/gate path). `Dsv4Vt::Off` + the same depth_cap is
    /// byte-identical to the env path with `MEMRA_DSV4_VT` unset.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn spec_greedy_batched_policy(
        &self,
        prompt: &[u32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
    ) -> Res<SpecRunGpu> {
        self.spec_greedy_batched_stream(prompt, n_new, state, dstate, vstate, depth_cap, vt, None)
    }

    /// ds4f rung 3 — [`Self::spec_greedy_batched_policy`] with a per-round COMMIT
    /// callback: `round_cb` receives every newly committed token slice after the
    /// round's ring writes + close sync (i.e. the tokens are final), and returning
    /// `false` stops generation at that round boundary — the serve door's streaming,
    /// EOS/stop-string, and client-disconnect cancel all ride this one seam. `None`
    /// is byte-identical to the gated driver (the closure is never constructed).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn spec_greedy_batched_stream(
        &self,
        prompt: &[u32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        round_cb: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Res<SpecRunGpu> {
        let p0 = prompt.len();
        assert!(n_new >= 1, "n_new must be positive");
        let pre = self.dspark_prefill_prime(prompt, state, dstate)?;
        self.spec_greedy_batched_stream_seeded(
            p0,
            &pre.logits,
            n_new,
            state,
            dstate,
            vstate,
            depth_cap,
            vt,
            round_cb,
        )
    }

    /// Speculative greedy generation over a trunk + DSpark state restored at the exact
    /// prompt boundary. `initial_logits` are produced while feeding the non-empty suffix
    /// after the cached prefix, so this path performs no cold re-prefill.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn spec_greedy_batched_stream_restored(
        &self,
        prompt_len: usize,
        initial_logits: &[f32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        round_cb: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Res<SpecRunGpu> {
        if state.pos != prompt_len {
            return Err(format!(
                "dsv4 restored spec state pos {} != prompt len {prompt_len}",
                state.pos
            ));
        }
        if dstate.tap_head != 0 {
            return Err(format!(
                "dsv4 restored spec tap cursor {} != normalized row 0",
                dstate.tap_head
            ));
        }
        self.spec_greedy_batched_stream_seeded(
            prompt_len,
            initial_logits,
            n_new,
            state,
            dstate,
            vstate,
            depth_cap,
            vt,
            round_cb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn spec_greedy_batched_stream_seeded(
        &self,
        p0: usize,
        initial_logits: &[f32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        mut round_cb: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Res<SpecRunGpu> {
        assert!(n_new >= 1, "n_new must be positive");
        let mut t_tok = {
            let lg = initial_logits;
            let mut best = 0usize;
            for i in 1..lg.len() {
                if lg[i] > lg[best] {
                    best = i;
                }
            }
            best as u32
        };
        let mut tokens: Vec<u32> = Vec::with_capacity(n_new);
        let mut rounds: Vec<SpecRoundGpu> = Vec::new();
        let mut mh_row = 0usize; // taps row holding the tap of the position behind `t_tok`
        let mut carry_pending = false;
        // MEMRA_DSV4_BENCH_PROFILE=1: bracket steady-state ROUNDS [4, 12) with
        // cudaProfilerStart/Stop so `nsys profile -c cudaProfilerApi` captures only
        // rounds — no load, no prefill/prime, no warmup. Read ONCE (never per round).
        // Profiling runs are rung-0 instruments, never A/B observations.
        let profile_bracket = std::env::var("MEMRA_DSV4_BENCH_PROFILE").as_deref() == Ok("1");
        let depth_cap = depth_cap.max(1);
        if let Dsv4Vt::Slot { tau_logit, floor } = vt {
            println!(
                "[spec] vt policy: slot (tau_logit {tau_logit:.6}, floor {floor}) — \
                 per-round verify window from the confidence head"
            );
        }
        while tokens.len() < n_new {
            if profile_bracket && rounds.len() == 4 {
                cudarc::driver::safe::profiler_start().map_err(e("profiler_start"))?;
            }
            if profile_bracket && rounds.len() == 12 {
                cudarc::driver::safe::profiler_stop().map_err(e("profiler_stop"))?;
            }
            let cb_from = tokens.len();
            if carry_pending {
                tokens.push(t_tok);
                if let Some(cb) = round_cb.as_deref_mut() {
                    cb(&tokens[cb_from..]);
                }
                break;
            }
            let round_t0 = std::time::Instant::now();
            let prof_stream = if dsv4_prof_on() {
                Some(self.stages[self.stages.len() - 1].gpu.stream())
            } else {
                None
            };
            let _p_round = phase!("round", prof_stream.as_ref());
            let m0 = p0 + tokens.len();
            let prop = {
                let _p = phase!("1.drafter_forward", prof_stream.as_ref());
                self.dspark_forward_spec(dstate, t_tok, mh_row, m0 - 1, false)?
            };
            let k_drafts = prop.out_ids.len() - 1;
            tokens.push(t_tok);
            if tokens.len() == n_new {
                rounds.push(SpecRoundGpu {
                    start_pos: m0 - 1,
                    drafts: prop.out_ids[1..].to_vec(),
                    accepts: 0,
                    verified: 0,
                    t_batch: 0,
                    t_cap: 0,
                    confidence: prop.confidence.clone(),
                    emitted: 1,
                    round_us: round_t0.elapsed().as_micros() as u64,
                });
                if let Some(cb) = round_cb.as_deref_mut() {
                    cb(&tokens[cb_from..]);
                }
                break;
            }
            let forwards_left = n_new - tokens.len();
            // STRUCTURAL ceiling (drafts available / depth knob / vt window /
            // verify-state capacity), then the n_new BUDGET on top. Keeping them
            // separate is what lets the depth knob (and the vt window, which is a
            // per-round depth) shorten a round without it looking like "we ran out of
            // tokens" — carry_pending below fires on the BUDGET only.
            let vt_drafts = match vt {
                Dsv4Vt::Off => k_drafts,
                Dsv4Vt::Slot { tau_logit, floor } => {
                    vt_slot_drafts(&prop.confidence, tau_logit, floor)
                }
            };
            let t_cap = (vt_drafts + 1)
                .min(k_drafts + 1)
                .min(depth_cap)
                .min(vstate.tmax);
            let t_batch = t_cap.min(forwards_left);
            let kv = t_batch - 1;
            let mut batch_ids = Vec::with_capacity(t_batch);
            batch_ids.push(t_tok);
            batch_ids.extend_from_slice(&prop.out_ids[1..1 + kv]);
            let (_, am) = {
                let _p = phase!("2.verify_batch", prof_stream.as_ref());
                self.verify_batch_dev(&batch_ids, state, vstate, Some(&mut dstate.taps), false)?
            };
            // accept walk: row i (position m0+i) arbitrates draft i+1
            let mut c_d = 0usize;
            let mut t_next = 0u32;
            for i in 0..t_batch {
                let a = am[i];
                if i < kv && a == batch_ids[i + 1] {
                    c_d += 1;
                    continue;
                }
                t_next = a;
                break;
            }
            let n_commit = c_d + 1;
            {
                let _p = phase!("3.commit_rollback", prof_stream.as_ref());
                self.commit_verify_dev(state, vstate, n_commit)?;
            }
            // drafter rings advance for EVERY accepted position and no rejected one
            {
                let _p = phase!("4.ring_writes", prof_stream.as_ref());
                for i in 0..n_commit {
                    self.dspark_write_rings(dstate, i, m0 + i)?;
                }
            }
            {
                let _p = phase!("5.round_close_sync", None);
                // close the round on device too, so the ring advance is inside THIS
                // round's measurement and not the next one's
                let last = self.stages.len() - 1;
                self.stages[last]
                    .gpu
                    .stream()
                    .synchronize()
                    .map_err(e("round close sync"))?;
            }
            mh_row = c_d;
            for i in 0..c_d {
                tokens.push(batch_ids[i + 1]);
            }
            // Carry (= stop after emitting the bonus token) only when the n_new BUDGET
            // truncated this round -- never when the depth knob did. Identical to the old
            // `kv < k_drafts` whenever the knob is unset and vstate.tmax >= k_drafts + 1.
            carry_pending = c_d == kv && t_batch < t_cap;
            rounds.push(SpecRoundGpu {
                start_pos: m0 - 1,
                drafts: prop.out_ids[1..].to_vec(),
                accepts: c_d,
                verified: (c_d + 1).min(kv),
                t_batch,
                t_cap,
                confidence: prop.confidence.clone(),
                emitted: 1 + c_d,
                round_us: round_t0.elapsed().as_micros() as u64,
            });
            t_tok = t_next;
            if let Some(cb) = round_cb.as_deref_mut()
                && !cb(&tokens[cb_from..])
            {
                break;
            }
        }
        Ok(SpecRunGpu { tokens, rounds })
    }

    /// [`Self::spec_greedy_batched_with`] with freshly allocated state (gate shape).
    pub fn spec_greedy_batched(&self, prompt: &[u32], n_new: usize) -> Res<SpecRunGpu> {
        let mut state = self.alloc_decode_state()?;
        let mut dstate = self.dspark_alloc_state()?;
        let mut vstate = self.alloc_verify_state()?;
        self.spec_greedy_batched_with(prompt, n_new, &mut state, &mut dstate, &mut vstate)
    }

    /// ds4f rung 2 (slice 1) — the SAMPLED propose-then-verify loop (it5 item 8).
    ///
    /// A deliberate near-copy of [`Self::spec_greedy_batched_policy`] with the accept
    /// walk arbitrated by POSITION-KEYED seeded target draws instead of argmax — the
    /// gated greedy driver's bytes are not touched (its accept-sha receipts stay the
    /// witness; a shared parameterized loop would put those bytes at refactor risk for
    /// zero measurement gain). Identity law: the emitted stream equals the plain
    /// sampled stream at the same seed BY CONSTRUCTION — row i of the batched verify
    /// is bit-exact against the sequential step's row at the same position (the it3
    /// gate (c) proof) and [`dsv4_sample_row`] is a pure function of (row, pos, seed).
    /// The drafter proposes greedily (deterministic one-hot proposal); a draft is
    /// accepted iff it EQUALS the target draw at its position — the correct
    /// arbitration for a one-hot proposal (full min(1, p/q) rejection sampling
    /// degenerates to exactly this when q is one-hot). Penalties are slice 2 and NOT
    /// claimed here.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_sampled_batched_policy(
        &self,
        prompt: &[u32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        sample: &Dsv4SampleCfg,
    ) -> Res<SpecRunGpu> {
        self.spec_sampled_batched_stream(
            prompt, n_new, state, dstate, vstate, depth_cap, vt, sample, None,
        )
    }

    /// [`Self::spec_sampled_batched_policy`] with the rung-3 per-round commit callback
    /// (see [`Self::spec_greedy_batched_stream`] — same seam, same None-is-byte-identical
    /// contract).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn spec_sampled_batched_stream(
        &self,
        prompt: &[u32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        sample: &Dsv4SampleCfg,
        mut round_cb: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Res<SpecRunGpu> {
        self.spec_sampled_batched_pen(
            prompt,
            n_new,
            state,
            dstate,
            vstate,
            depth_cap,
            vt,
            sample,
            None,
            round_cb.take(),
        )
    }

    /// ds4f rung-2 slice 2 — the sampled driver with PENALTIES over the true
    /// per-state window (row-incremental: row r penalizes over prompt ++ committed
    /// ++ this round's accepts before r — the q38 penalized-sampled law). `None` is
    /// byte-identical to the unpenalized driver. Identity vs the plain penalized
    /// loop is structural for the same reason as the unpenalized path: the window at
    /// a given position is a pure function of the shared committed prefix.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn spec_sampled_batched_pen(
        &self,
        prompt: &[u32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        sample: &Dsv4SampleCfg,
        pen: Option<&Dsv4PenaltyCfg>,
        round_cb: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Res<SpecRunGpu> {
        assert!(n_new >= 1, "n_new must be positive");
        let pre = self.dspark_prefill_prime(prompt, state, dstate)?;
        self.spec_sampled_batched_pen_seeded(
            prompt,
            &pre.logits,
            n_new,
            state,
            dstate,
            vstate,
            depth_cap,
            vt,
            sample,
            pen,
            round_cb,
        )
    }

    /// Sampled DSpark generation over an exact restored prompt boundary. Sampling remains
    /// position-keyed against the full logical prompt, so the restored and cold routes share
    /// the same token stream for a fixed seed.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn spec_sampled_batched_pen_restored(
        &self,
        prompt: &[u32],
        initial_logits: &[f32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        sample: &Dsv4SampleCfg,
        pen: Option<&Dsv4PenaltyCfg>,
        round_cb: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Res<SpecRunGpu> {
        if state.pos != prompt.len() {
            return Err(format!(
                "dsv4 restored sampled state pos {} != prompt len {}",
                state.pos,
                prompt.len()
            ));
        }
        if dstate.tap_head != 0 {
            return Err(format!(
                "dsv4 restored sampled tap cursor {} != normalized row 0",
                dstate.tap_head
            ));
        }
        self.spec_sampled_batched_pen_seeded(
            prompt,
            initial_logits,
            n_new,
            state,
            dstate,
            vstate,
            depth_cap,
            vt,
            sample,
            pen,
            round_cb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn spec_sampled_batched_pen_seeded(
        &self,
        prompt: &[u32],
        initial_logits: &[f32],
        n_new: usize,
        state: &mut DecodeState,
        dstate: &mut DsparkState,
        vstate: &mut VerifyState,
        depth_cap: usize,
        vt: Dsv4Vt,
        sample: &Dsv4SampleCfg,
        pen: Option<&Dsv4PenaltyCfg>,
        mut round_cb: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Res<SpecRunGpu> {
        let p0 = prompt.len();
        assert!(n_new >= 1, "n_new must be positive");
        // token at absolute position p0 (output index 0): the seeded draw, keyed p0
        let mut t_tok = if let Some(pc) = pen {
            let mut row = initial_logits.to_vec();
            dsv4_penalize_row(&mut row, prompt, pc);
            dsv4_sample_row(&row, p0, sample)?
        } else {
            dsv4_sample_row(initial_logits, p0, sample)?
        };
        let mut tokens: Vec<u32> = Vec::with_capacity(n_new);
        let mut rounds: Vec<SpecRoundGpu> = Vec::new();
        let mut mh_row = 0usize;
        let mut carry_pending = false;
        let depth_cap = depth_cap.max(1);
        while tokens.len() < n_new {
            let cb_from = tokens.len();
            if carry_pending {
                tokens.push(t_tok);
                if let Some(cb) = round_cb.as_deref_mut() {
                    cb(&tokens[cb_from..]);
                }
                break;
            }
            let round_t0 = std::time::Instant::now();
            let m0 = p0 + tokens.len();
            let proposal_sample = match self.draft_proposal {
                Dsv4DraftProposal::Greedy => None,
                Dsv4DraftProposal::Coupled => Some(sample),
            };
            let prop = self.dspark_forward_spec_inner(
                dstate,
                t_tok,
                mh_row,
                m0 - 1,
                false,
                proposal_sample,
            )?;
            let k_drafts = prop.out_ids.len() - 1;
            tokens.push(t_tok);
            if tokens.len() == n_new {
                rounds.push(SpecRoundGpu {
                    start_pos: m0 - 1,
                    drafts: prop.out_ids[1..].to_vec(),
                    accepts: 0,
                    verified: 0,
                    t_batch: 0,
                    t_cap: 0,
                    confidence: prop.confidence.clone(),
                    emitted: 1,
                    round_us: round_t0.elapsed().as_micros() as u64,
                });
                if let Some(cb) = round_cb.as_deref_mut() {
                    cb(&tokens[cb_from..]);
                }
                break;
            }
            let forwards_left = n_new - tokens.len();
            let vt_drafts = match vt {
                Dsv4Vt::Off => k_drafts,
                Dsv4Vt::Slot { tau_logit, floor } => {
                    vt_slot_drafts(&prop.confidence, tau_logit, floor)
                }
            };
            let t_cap = (vt_drafts + 1)
                .min(k_drafts + 1)
                .min(depth_cap)
                .min(vstate.tmax);
            let t_batch = t_cap.min(forwards_left);
            let kv = t_batch - 1;
            let mut batch_ids = Vec::with_capacity(t_batch);
            batch_ids.push(t_tok);
            batch_ids.extend_from_slice(&prop.out_ids[1..1 + kv]);
            let (rows, _am) =
                self.verify_batch_dev(&batch_ids, state, vstate, Some(&mut dstate.taps), true)?;
            let rows = rows.expect("verify_batch_dev(want_logits=true) returned rows");
            let vocab = rows.len() / t_batch;
            // sampled accept walk: row i's input token sits at position m0 + i, so the
            // row PREDICTS the token at position m0 + i + 1 — that predicted position
            // is the draw key (the plain loop keys every token by its own absolute
            // position; misaligning this by one would silently break the identity law
            // at every accepted draft). Draft i+1 is accepted iff it equals the draw.
            let mut c_d = 0usize;
            let mut t_next = 0u32;
            // row-incremental penalty window: prompt ++ tokens (head included) ++ the
            // accepts of rows < i in THIS round (batch_ids[1..=c_d] at walk time).
            let mut wround: Vec<u32> = Vec::new();
            for i in 0..t_batch {
                let s = if let Some(pc) = pen {
                    let mut row = rows[i * vocab..(i + 1) * vocab].to_vec();
                    let mut window = Vec::with_capacity(prompt.len() + tokens.len() + wround.len());
                    window.extend_from_slice(prompt);
                    window.extend_from_slice(&tokens);
                    window.extend_from_slice(&wround);
                    dsv4_penalize_row(&mut row, &window, pc);
                    dsv4_sample_row(&row, m0 + i + 1, sample)?
                } else {
                    dsv4_sample_row(&rows[i * vocab..(i + 1) * vocab], m0 + i + 1, sample)?
                };
                if i < kv && s == batch_ids[i + 1] {
                    c_d += 1;
                    wround.push(batch_ids[i + 1]);
                    continue;
                }
                t_next = s;
                break;
            }
            let n_commit = c_d + 1;
            self.commit_verify_dev(state, vstate, n_commit)?;
            for i in 0..n_commit {
                self.dspark_write_rings(dstate, i, m0 + i)?;
            }
            {
                let last = self.stages.len() - 1;
                self.stages[last]
                    .gpu
                    .stream()
                    .synchronize()
                    .map_err(e("round close sync"))?;
            }
            mh_row = c_d;
            for i in 0..c_d {
                tokens.push(batch_ids[i + 1]);
            }
            carry_pending = c_d == kv && t_batch < t_cap;
            rounds.push(SpecRoundGpu {
                start_pos: m0 - 1,
                drafts: prop.out_ids[1..].to_vec(),
                accepts: c_d,
                verified: (c_d + 1).min(kv),
                t_batch,
                t_cap,
                confidence: prop.confidence.clone(),
                emitted: 1 + c_d,
                round_us: round_t0.elapsed().as_micros() as u64,
            });
            t_tok = t_next;
            if let Some(cb) = round_cb.as_deref_mut()
                && !cb(&tokens[cb_from..])
            {
                break;
            }
        }
        Ok(SpecRunGpu { tokens, rounds })
    }
}

impl Dsv4Gpu {
    /// Every LIVE trunk cache class, per layer, as host f32 arrays — the instrument for
    /// the §3.1 device state gate (batched round + commit vs plain sequential decode of
    /// the committed tokens, bit for bit). "Live" is load-bearing: bytes past `n_blocks`
    /// in an append-only store, and the TRANSIENT verify rows, are dead scratch and are
    /// deliberately excluded (the CPU-oracle gate draws the same line).
    pub fn cache_classes(&self, state: &DecodeState) -> Res<Vec<(String, Vec<f32>)>> {
        let d = self.model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let mut out = Vec::new();
        for (il, cache) in state.caches.iter().enumerate() {
            let stage_i = self.layer_stage[il];
            let st = &self.stages[stage_i];
            st.gpu.ctx.bind_to_thread().map_err(e("bind ctx classes"))?;
            let stream = st.gpu.stream();
            let lidx = st
                .layers
                .iter()
                .position(|l| l.il == il as u32)
                .unwrap_or_else(|| panic!("layer {il} not on stage {stage_i}"));
            let layer = &st.layers[lidx];
            let read = |sl: cudarc::driver::CudaView<'_, f32>| -> Res<Vec<f32>> {
                let mut v = vec![0f32; sl.len()];
                stream
                    .memcpy_dtoh(&sl, &mut v[..])
                    .map_err(e("dtoh class"))?;
                stream.synchronize().map_err(e("sync class"))?;
                Ok(v)
            };
            out.push((format!("l{il}.ring"), read(cache.kvc.slice(0..win * hd))?));
            if let Some(cmp) = &layer.cmp {
                out.push((
                    format!("l{il}.cmp_store"),
                    read(cache.kvc.slice(win * hd..(win + cache.n_blocks) * cmp.d))?,
                ));
                out.push((
                    format!("l{il}.cmp_pend_kv"),
                    read(cache.pend_kv.as_ref().expect("pend kv").slice(..))?,
                ));
                out.push((
                    format!("l{il}.cmp_pend_score"),
                    read(cache.pend_score.as_ref().expect("pend sc").slice(..))?,
                ));
            }
            if let Some(ix) = &layer.idx {
                let ikvc = cache.ikvc.as_ref().expect("ikvc");
                out.push((
                    format!("l{il}.idx_store"),
                    read(ikvc.slice(0..cache.i_blocks * ix.cmp.d))?,
                ));
                out.push((
                    format!("l{il}.idx_pend_kv"),
                    read(cache.ipend_kv.as_ref().expect("ipend kv").slice(..))?,
                ));
                out.push((
                    format!("l{il}.idx_pend_score"),
                    read(cache.ipend_score.as_ref().expect("ipend sc").slice(..))?,
                ));
            }
        }
        Ok(out)
    }

    /// The DSpark drafter's main_kv rings as host f32 arrays (accepted-position-only
    /// ring-write rule gate: the batched drafted arm's rings must end bit-identical to a
    /// plain greedy run that wrote a ring row at EVERY decoded position).
    pub fn dspark_ring_classes(&self, dstate: &DsparkState) -> Res<Vec<(String, Vec<f32>)>> {
        let last = self.stages.len() - 1;
        let st = &self.stages[last];
        st.gpu.ctx.bind_to_thread().map_err(e("bind ctx rings"))?;
        let stream = st.gpu.stream();
        let d = self.model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let mut out = Vec::new();
        for (bi, ring) in dstate.rings.iter().enumerate() {
            // persistent ring only — rows [win, win+block) are the drafter's transient
            // draft-kv scratch, rewritten by every propose and never state.
            let view = ring.slice(0..win * hd);
            let mut v = vec![0f32; view.len()];
            stream
                .memcpy_dtoh(&view, &mut v[..])
                .map_err(e("dtoh ring class"))?;
            stream.synchronize().map_err(e("sync ring class"))?;
            out.push((format!("dspark.ring{bi}"), v));
        }
        Ok(out)
    }
}

/// The dense-arm resolution, pure for the flip's toothed tests (owner ratification
/// 2026-08-20, executed v0.98): unset = `fp8` on the DEVICE decode path, `bf16` on
/// legacy (device-scoped default, the 82a754fbec dots-default shape); explicit values
/// keep their exact prior semantics including the legacy+fp8 refusal and the
/// unknown-value refusal.
pub fn resolve_dense_arm(v: Option<&str>, on_device: bool) -> Result<bool, String> {
    match v {
        None | Some("") => Ok(on_device),
        Some("bf16") => Ok(false),
        Some("fp8") if !on_device => Err(
            "MEMRA_DSV4_DENSE_ARM=fp8 requires MEMRA_DSV4_DECODE_PATH=device (the \
             fp8 GEMV twins exist on the device decode/verify paths only; prefill \
             and the legacy path consume the bf16 slabs)"
                .to_string(),
        ),
        Some("fp8") => Ok(true),
        Some(other) => Err(format!(
            "MEMRA_DSV4_DENSE_ARM '{other}' unknown (bf16 | fp8)"
        )),
    }
}

pub fn resolve_dspark_fused_moe(v: Option<&str>, on_device: bool) -> Result<bool, String> {
    match v {
        None | Some("") | Some("0") => Ok(false),
        Some("1") if on_device => Ok(true),
        Some("1") => {
            Err("MEMRA_DSV4_DSPARK_FUSED_MOE=1 requires MEMRA_DSV4_DECODE_PATH=device".to_string())
        }
        Some(other) => Err(format!(
            "MEMRA_DSV4_DSPARK_FUSED_MOE '{other}' unknown (0 | 1)"
        )),
    }
}

/// ds4f rung 1 — per-round verify-window policy from the drafter's OWN confidence head
/// (`MEMRA_DSV4_VT={off|slot}`, unset = off = the byte-identical round driver).
///
/// `slot` is the owner-directive per-slot reading. The q38 H4 verdict transfers as a
/// MECHANISM, never as receipts (no-generic-support): their head emits MARGINAL accept
/// probabilities, so cumprod-survival double-counts depth decay — and dsv4's own head
/// was independently measured discriminative per-slot (AUC 0.871–0.918, it5 rung 4,
/// where STS recalibration was the measured NEGATIVE — the policy consumes RAW
/// sigmoids by design). Verification still arbitrates every forwarded draft, so the
/// policy moves acceptance ECONOMICS only; greedy identity holds at any window (the
/// `MEMRA_DSV4_SPEC_DEPTH` argument, verbatim — this is a per-round depth).
///
/// Knobs: `MEMRA_DSV4_VT_TAU` in (0,1) exclusive, default 0.5; `MEMRA_DSV4_VT_FLOOR`
/// = minimum drafts forwarded, default 0, max `DSPARK_BLOCK-1` (0 is legal: a
/// fully-unconfident proposal degenerates to a 1-row verify — the it5 Algorithm-1
/// scans price exactly that round shape). Unknown values, out-of-range tau/floor, and
/// orphan knobs (tau/floor set without `slot`) REFUSE BY NAME.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dsv4Vt {
    Off,
    /// tau stored in LOGIT space (sigmoid(c) >= tau  <=>  c >= tau_logit, exact for
    /// tau = 0.5 -> 0.0); floor = minimum number of drafts forwarded per round.
    Slot {
        tau_logit: f32,
        floor: usize,
    },
}

pub fn resolve_vt(
    policy: Option<&str>,
    tau: Option<&str>,
    floor: Option<&str>,
) -> Result<Dsv4Vt, String> {
    match policy {
        None | Some("") | Some("off") => {
            if let Some(t) = tau {
                return Err(format!(
                    "MEMRA_DSV4_VT_TAU='{t}' set without MEMRA_DSV4_VT=slot (orphan knob \
                     would be silently inert — refuse instead)"
                ));
            }
            if let Some(f) = floor {
                return Err(format!(
                    "MEMRA_DSV4_VT_FLOOR='{f}' set without MEMRA_DSV4_VT=slot (orphan \
                     knob would be silently inert — refuse instead)"
                ));
            }
            Ok(Dsv4Vt::Off)
        }
        Some("slot") => {
            let tau_v: f32 = match tau {
                None => 0.5,
                Some(s) => s
                    .trim()
                    .parse::<f32>()
                    .map_err(|_| format!("MEMRA_DSV4_VT_TAU '{s}' is not a float in (0,1)"))?,
            };
            if !(tau_v > 0.0 && tau_v < 1.0) {
                return Err(format!(
                    "MEMRA_DSV4_VT_TAU {tau_v} out of range: need 0 < tau < 1 \
                     (a probability threshold on the per-slot sigmoid)"
                ));
            }
            let floor_v: usize = match floor {
                None => 0,
                Some(s) => s.trim().parse::<usize>().map_err(|_| {
                    format!("MEMRA_DSV4_VT_FLOOR '{s}' is not a non-negative integer")
                })?,
            };
            // block_size is baked into the weights at 5 (DSPARK-SEMANTICS §1.5); a
            // floor >= block would pin the window fully open, i.e. silently disable
            // the policy while claiming to run it.
            if floor_v >= 5 {
                return Err(format!(
                    "MEMRA_DSV4_VT_FLOOR {floor_v} >= dspark block size 5 would pin the \
                     window fully open (use MEMRA_DSV4_VT=off to disable)"
                ));
            }
            Ok(Dsv4Vt::Slot {
                tau_logit: (tau_v / (1.0 - tau_v)).ln(),
                floor: floor_v,
            })
        }
        Some(other) => Err(format!("MEMRA_DSV4_VT '{other}' unknown (off | slot)")),
    }
}

/// ds4f rung 2 (slice 1) — the dsv4 SAMPLED path's sampler: deterministic,
/// POSITION-KEYED seeded draws over a temperature/top-k/top-p-filtered target row.
///
/// Position keying is the identity law's load-bearing choice: the uniform draw for
/// absolute position `pos` is a pure function of (seed, pos), never of how many draws
/// happened before — so the plain sampled loop and the sampled-leader verify walk
/// consume IDENTICAL randomness at every position, and (because the batched verify's
/// logits rows are bit-exact against the sequential step's — the it3 gate (c) proof)
/// **sampled spec == sampled plain identity is structural, per seed**, exactly like
/// greedy. Arbitration always matches against the target's own position-keyed
/// draw. The default draft is greedy; the opt-in coupled proposal uses the same
/// draw key on its own row. Either proposal is only a hint: emitted tokens always
/// follow the target draw on the accepted prefix, with no extra RNG advancement.
///
/// Filter semantics (vendor-posture defaults live at the call sites: temperature 1.0,
/// top_p 1.0 for 0731, top_k off): logits/T -> softmax -> top-k by (value desc, index asc)
/// -> smallest prefix of that order with cumulative mass >= top_p (always >= 1 token)
/// -> renormalize -> inverse-CDF draw at u(seed, pos). temperature <= 0 REFUSES BY
/// NAME (greedy is the greedy driver's job; a silent argmax fallback here would be
/// the q38 penalized-greedy footgun).
#[derive(Clone, Copy, Debug)]
pub struct Dsv4SampleCfg {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// The uniform draw for absolute position `pos` under `seed` — in [0, 1).
pub fn dsv4_pos_uniform(seed: u64, pos: usize) -> f64 {
    let h = splitmix64(seed ^ (pos as u64).wrapping_mul(0xa24baed4963ee407));
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// One sampled token from a full-vocab logits row at absolute position `pos`.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // allow: NaN must take this branch; !(a > b) is not a <= b under IEEE comparisons
pub fn dsv4_sample_row(logits: &[f32], pos: usize, cfg: &Dsv4SampleCfg) -> Result<u32, String> {
    if !(cfg.temperature > 0.0) {
        return Err(format!(
            "dsv4 sampled path: temperature {} refused (need > 0; greedy is served by \
             the greedy driver, never a silent argmax fallback)",
            cfg.temperature
        ));
    }
    if !(cfg.top_p > 0.0 && cfg.top_p <= 1.0) {
        return Err(format!(
            "dsv4 sampled path: top_p {} out of (0, 1]",
            cfg.top_p
        ));
    }
    // candidate order: value desc, index asc (the house tie ordering)
    let k = if cfg.top_k == 0 || cfg.top_k > logits.len() {
        logits.len()
    } else {
        cfg.top_k
    };
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let (va, vb) = (logits[a as usize], logits[b as usize]);
        vb.partial_cmp(&va)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    // softmax over the kept set in kept order (f64 accumulation, max-shifted)
    let m = logits[idx[0] as usize] as f64;
    let t = cfg.temperature as f64;
    let mut probs: Vec<f64> = idx
        .iter()
        .map(|&i| (((logits[i as usize] as f64) - m) / t).exp())
        .collect();
    let z: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= z;
    }
    // nucleus: smallest prefix with cumulative >= top_p (>= 1 token), renormalize
    let mut cum = 0.0f64;
    let mut keep = probs.len();
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if cum >= cfg.top_p as f64 {
            keep = i + 1;
            break;
        }
    }
    probs.truncate(keep);
    idx.truncate(keep);
    let z2: f64 = probs.iter().sum();
    let u = dsv4_pos_uniform(cfg.seed, pos) * z2;
    let mut acc = 0.0f64;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return Ok(idx[i]);
        }
    }
    Ok(idx[keep - 1]) // u landed on the tail boundary (float roundoff)
}

/// ds4f rung 2 slice 2 — penalties for the dsv4 sampled path, over an EXPLICIT
/// window. The rule is `memra-sampling`'s own `Sampler::apply_penalties` (Keskar
/// repeat divide/multiply toward 0 + frequency*count + presence), replicated here
/// because the dsv4 path needs per-ROW windows (the spec verify's row-incremental
/// state: row r penalizes over prompt ++ committed ++ this round's accepts < r),
/// and CROSS-PINNED by unit test against a real `Sampler` so the two
/// implementations cannot drift apart silently.
#[derive(Clone, Copy, Debug)]
pub struct Dsv4PenaltyCfg {
    pub last_n: usize,
    pub repeat: f32,
    pub freq: f32,
    pub present: f32,
}

impl Dsv4PenaltyCfg {
    pub fn armed(&self) -> bool {
        self.last_n > 0 && (self.repeat != 1.0 || self.freq != 0.0 || self.present != 0.0)
    }
}

/// Apply the Keskar penalties in place over `window`'s last `cfg.last_n` entries.
pub fn dsv4_penalize_row(logits: &mut [f32], window: &[u32], cfg: &Dsv4PenaltyCfg) {
    if !cfg.armed() {
        return;
    }
    let start = window.len().saturating_sub(cfg.last_n);
    let win = &window[start..];
    if win.is_empty() {
        return;
    }
    let mut counts: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
    for &t in win {
        *counts.entry(t).or_insert(0) += 1;
    }
    for (&id, &cnt) in &counts {
        let Some(l) = logits.get_mut(id as usize) else {
            continue;
        };
        if cfg.repeat != 1.0 {
            if *l > 0.0 {
                *l /= cfg.repeat;
            } else {
                *l *= cfg.repeat;
            }
        }
        *l -= cfg.freq * cnt as f32;
        if cnt > 0 {
            *l -= cfg.present;
        }
    }
}

/// Drafts to forward under the slot policy: the longest LEADING prefix of `conf`
/// (the drafter's pre-sigmoid per-slot logits) with `c >= tau_logit`, raised to
/// `floor`, clamped to `conf.len()`. A NaN slot compares false = unconfident
/// (conservative: it truncates, and verification still owns correctness).
pub fn vt_slot_drafts(conf: &[f32], tau_logit: f32, floor: usize) -> usize {
    let mut k = 0usize;
    for &c in conf {
        if c >= tau_logit {
            k += 1;
        } else {
            break;
        }
    }
    k.max(floor).min(conf.len())
}

#[cfg(test)]
mod capacity_planner_tests {
    use super::{dsv4_cache_cap_blocks, dsv4_split_for_tail_reserve, ring_commit_plan};

    #[test]
    fn wide_ring_commits_are_unique_and_equal_sequential_writes() {
        let win = 128;
        for pos0 in 0..2 * win {
            for count in [1, 6, 64, 127, 128, 129, 256, 511, 512] {
                let (start, slots) = ring_commit_plan(pos0, count, win);
                assert_eq!(slots.len(), count.min(win));
                assert_eq!(
                    slots
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len(),
                    slots.len()
                );
                let mut sequential = vec![usize::MAX; win];
                for row in 0..count {
                    sequential[(pos0 + row) % win] = row;
                }
                let mut gathered = vec![usize::MAX; win];
                for (offset, &slot) in slots.iter().enumerate() {
                    gathered[slot as usize] = start + offset;
                }
                assert_eq!(gathered, sequential, "pos0={pos0} count={count}");
            }
        }
        let old: Vec<_> = (0..512).map(|row| row % win).collect();
        assert_ne!(
            old.iter().collect::<std::collections::BTreeSet<_>>().len(),
            old.len(),
            "old plan must expose duplicate writes"
        );
    }

    #[test]
    fn verify_transient_base_tracks_session_not_model_capacity() {
        let win = 128usize;
        let ratio = 128usize;
        let short_capacity = 235usize;
        let model_capacity = 1_048_576usize;

        let short_base = win + dsv4_cache_cap_blocks(short_capacity, ratio);
        let model_base = win + dsv4_cache_cap_blocks(model_capacity, ratio);
        assert_eq!(short_base, 129);
        assert_eq!(model_base, 8_320);
        assert_ne!(
            short_base, model_base,
            "session verify must not use 1M base"
        );
        assert_eq!(dsv4_cache_cap_blocks(short_capacity, 0), 0);
    }

    #[test]
    fn pp_cut_accounts_for_a_tail_resident_drafter() {
        let layers = vec![100u64; 42];
        assert_eq!(dsv4_split_for_tail_reserve(&layers, 0), 21);
        assert_eq!(dsv4_split_for_tail_reserve(&layers, 300), 23);
    }
}

#[cfg(test)]
mod prefill_work_policy_tests {
    use super::{Dsv4PrefillDraft, Dsv4PrefillHead, VerifyOutput};

    #[test]
    fn output_policy_is_explicit_and_defaults_to_legacy_work() {
        for value in [None, Some(""), Some("all")] {
            assert_eq!(Dsv4PrefillHead::resolve(value), Ok(Dsv4PrefillHead::All));
            assert_eq!(Dsv4PrefillDraft::resolve(value), Ok(Dsv4PrefillDraft::All));
        }
        assert_eq!(Dsv4PrefillHead::All.output(false), VerifyOutput::Argmax);
        assert_eq!(Dsv4PrefillHead::All.output(true), VerifyOutput::Full);
        assert_eq!(Dsv4PrefillHead::Last.output(false), VerifyOutput::None);
        assert_eq!(Dsv4PrefillHead::Last.output(true), VerifyOutput::Last);
        assert_eq!(
            Dsv4PrefillHead::resolve(Some("last")),
            Ok(Dsv4PrefillHead::Last)
        );
        assert_eq!(
            Dsv4PrefillDraft::resolve(Some("tail")),
            Ok(Dsv4PrefillDraft::Tail)
        );
        for value in ["1", "LAST", " tail"] {
            assert!(Dsv4PrefillHead::resolve(Some(value)).is_err());
            assert!(Dsv4PrefillDraft::resolve(Some(value)).is_err());
        }
    }

    #[test]
    fn tail_prime_has_the_same_final_ring_as_every_suffix_row() {
        let win = 128;
        for pos0 in [1, 63, 127, 128, 129, 511, 1000] {
            for len in [1, 31, 64, 127, 128, 129, 160, 512, 1024] {
                let keep = Dsv4PrefillDraft::Tail.keep_from(len, win);
                assert_eq!(len - keep, len.min(win));
                assert_eq!(Dsv4PrefillDraft::All.keep_from(len, win), 0);
                let mut reference: Vec<usize> = (0..win).map(|i| 10_000 + i).collect();
                let mut tail = reference.clone();
                for i in 0..len {
                    reference[(pos0 + i) % win] = i;
                }
                for i in keep..len {
                    tail[(pos0 + i) % win] = i;
                }
                assert_eq!(reference, tail, "pos0={pos0} suffix={len}");
            }
        }
    }
}

#[cfg(test)]
mod coupled_proposal_tests {
    use super::{Dsv4DraftProposal, Dsv4SampleCfg, dspark_draft_position, dsv4_sample_row};

    #[test]
    fn proposal_policy_is_explicit_and_default_greedy() {
        for raw in [None, Some(""), Some("greedy")] {
            assert_eq!(
                Dsv4DraftProposal::resolve(raw),
                Ok(Dsv4DraftProposal::Greedy)
            );
        }
        assert_eq!(
            Dsv4DraftProposal::resolve(Some("coupled")),
            Ok(Dsv4DraftProposal::Coupled)
        );
        for raw in ["sample", "1", "COUPLED"] {
            assert!(Dsv4DraftProposal::resolve(Some(raw)).is_err());
        }
    }

    #[test]
    fn identical_proposals_share_the_target_draw_key_not_the_input_position() {
        let cfg = Dsv4SampleCfg {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 20260905,
        };
        let logits = [0.2, 0.1, -0.3, 0.0];
        let mut shifted_mismatches = 0;
        for tap_position in 31..100 {
            let head_position = tap_position + 1;
            for slot in 0..5 {
                let target_position = head_position + slot + 1;
                let draft_position = dspark_draft_position(tap_position, slot);
                assert_eq!(draft_position, target_position);
                let target = dsv4_sample_row(&logits, target_position, &cfg).unwrap();
                assert_eq!(
                    target,
                    dsv4_sample_row(&logits, draft_position, &cfg).unwrap()
                );
                shifted_mismatches += usize::from(
                    target != dsv4_sample_row(&logits, draft_position - 1, &cfg).unwrap(),
                );
            }
        }
        assert!(
            shifted_mismatches > 0,
            "off-by-one control must fail agreement"
        );
    }
}

#[cfg(test)]
mod peer_probe_tests {
    use super::{dsv4_peer_probe_ladder, dsv4_peer_probe_mismatches, dsv4_peer_probe_pattern};

    /// TOOTH for the lane-8 byte probe (host-side halves; the on-box halves are the boot
    /// PASS line and the MEMRA_DSV4_PEER_PROBE_POISON refusal arm): the pattern must be
    /// deterministic, non-trivial, and keyed per (bytes, boundary, src, dst) so a stuck or
    /// crossed lane cannot alias another probe's expectation; the mismatch count must see
    /// single-byte flips, inversion (the poison), and truncation.
    #[test]
    fn peer_probe_pattern_is_keyed_and_mismatches_are_counted() {
        let a = dsv4_peer_probe_pattern(4096, 0, 0, 1);
        assert_eq!(a.len(), 4096);
        assert_eq!(a, dsv4_peer_probe_pattern(4096, 0, 0, 1), "deterministic");
        assert_ne!(a, dsv4_peer_probe_pattern(4096, 0, 1, 0), "direction-keyed");
        assert_ne!(a, dsv4_peer_probe_pattern(4096, 1, 0, 1), "boundary-keyed");
        assert!(a.iter().any(|&b| b != a[0]), "non-constant pattern");

        assert_eq!(dsv4_peer_probe_mismatches(&a, &a), 0);
        let mut flipped = a.clone();
        flipped[17] ^= 1;
        assert_eq!(dsv4_peer_probe_mismatches(&a, &flipped), 1);
        let poison: Vec<u8> = a.iter().map(|b| !b).collect();
        assert_eq!(dsv4_peer_probe_mismatches(&a, &poison), a.len());
        assert_eq!(dsv4_peer_probe_mismatches(&a, &a[..4000]), 96);
    }

    #[test]
    fn peer_probe_ladder_contains_live_hc_payloads() {
        let ladder = dsv4_peer_probe_ladder(4096, 4);
        assert!(ladder.contains(&(64 << 10)), "one-token hc state");
        assert!(ladder.contains(&(512 << 10)), "eight-row verify hc state");
        assert!(ladder.contains(&(64 << 20)), "maximum prefill handoff");
    }
}

#[cfg(test)]
mod dense_arm_default_tests {
    use super::{
        Dsv4Fp4Reduce, Dsv4IndexerScore, resolve_dense_arm, resolve_dspark_fused_moe,
        resolve_prefill_moe,
    };

    #[test]
    fn grouped_prefill_requires_literal_opt_in() {
        for raw in [None, Some(""), Some("reference")] {
            assert_eq!(resolve_prefill_moe(raw), Ok(false));
        }
        assert_eq!(resolve_prefill_moe(Some("grouped")), Ok(true));
        for raw in ["1", "GROUPED", " grouped", "f16"] {
            assert!(resolve_prefill_moe(Some(raw)).is_err());
        }
    }

    #[test]
    fn tiled_indexer_is_literal_and_default_off() {
        for raw in [None, Some(""), Some("scalar")] {
            assert_eq!(Dsv4IndexerScore::resolve(raw), Ok(Dsv4IndexerScore::Scalar));
        }
        assert_eq!(
            Dsv4IndexerScore::resolve(Some("tiled")),
            Ok(Dsv4IndexerScore::Tiled)
        );
        for raw in ["1", "TILED", " tiled", "fused"] {
            assert!(Dsv4IndexerScore::resolve(Some(raw)).is_err());
        }
    }

    #[test]
    fn fp4_reduction_resolution_is_literal_and_default_off() {
        for value in [None, Some(""), Some("block")] {
            assert_eq!(Dsv4Fp4Reduce::resolve(value), Ok(Dsv4Fp4Reduce::Block));
        }
        assert_eq!(Dsv4Fp4Reduce::default(), Dsv4Fp4Reduce::Block);
        assert_eq!(
            Dsv4Fp4Reduce::resolve(Some("warp")),
            Ok(Dsv4Fp4Reduce::Warp)
        );
        for value in ["1", "yes", "warp ", "WARP", " block"] {
            assert!(Dsv4Fp4Reduce::resolve(Some(value)).is_err());
        }
        // The explicit CUDA ABI has only these two legal arm values.
        assert_eq!(Dsv4Fp4Reduce::Block as i32, 0);
        assert_eq!(Dsv4Fp4Reduce::Warp as i32, 1);
    }

    /// The owner-ratified flip (2026-08-20): unset env on the device decode path = fp8.
    /// Mutating the default back to bf16 fails this with the evidence named.
    #[test]
    fn ratified_default_dense_arm_is_fp8_on_device() {
        assert_eq!(
            resolve_dense_arm(None, true),
            Ok(true),
            "owner-ratified 2026-08-20: unset MEMRA_DSV4_DENSE_ARM defaults the DEVICE \
             decode path to fp8 (bit-identical on four boxes, x5 A/B 41.06->47.19, \
             item-3 residency green on box7)"
        );
        assert_eq!(resolve_dense_arm(Some(""), true), Ok(true));
        // Legacy path: unset resolves bf16 (no fp8 twins there — must keep booting).
        assert_eq!(resolve_dense_arm(None, false), Ok(false));
        // Explicit values keep their exact prior semantics.
        assert_eq!(resolve_dense_arm(Some("bf16"), true), Ok(false));
        assert_eq!(resolve_dense_arm(Some("fp8"), true), Ok(true));
        assert!(
            resolve_dense_arm(Some("fp8"), false).is_err(),
            "legacy+fp8 stays a refusal"
        );
        assert!(
            resolve_dense_arm(Some("q8"), true).is_err(),
            "unknown values refuse"
        );
    }

    #[test]
    fn dspark_fused_moe_is_strict_default_off_and_device_only() {
        assert_eq!(resolve_dspark_fused_moe(None, true), Ok(false));
        assert_eq!(resolve_dspark_fused_moe(Some(""), true), Ok(false));
        assert_eq!(resolve_dspark_fused_moe(Some("0"), true), Ok(false));
        assert_eq!(resolve_dspark_fused_moe(Some("1"), true), Ok(true));
        assert!(resolve_dspark_fused_moe(Some("1"), false).is_err());
        assert!(resolve_dspark_fused_moe(Some("yes"), true).is_err());
    }
}

#[cfg(test)]
mod vt_policy_tests {
    use super::{Dsv4Vt, resolve_vt, vt_slot_drafts};

    /// Unset env = Off = the byte-identical round driver. Mutating the default fails
    /// this by name.
    #[test]
    fn default_vt_is_off_and_byte_inert() {
        assert_eq!(resolve_vt(None, None, None), Ok(Dsv4Vt::Off));
        assert_eq!(resolve_vt(Some(""), None, None), Ok(Dsv4Vt::Off));
        assert_eq!(resolve_vt(Some("off"), None, None), Ok(Dsv4Vt::Off));
    }

    #[test]
    fn slot_defaults_tau_half_floor_zero() {
        // tau 0.5 must land on tau_logit 0.0 EXACTLY (ln(0.5/0.5) = ln(1) = 0), so the
        // default threshold admits c = 0.0 with no float fuzz.
        match resolve_vt(Some("slot"), None, None) {
            Ok(Dsv4Vt::Slot { tau_logit, floor }) => {
                assert_eq!(tau_logit, 0.0);
                assert_eq!(floor, 0);
            }
            other => panic!("slot default parse broke: {other:?}"),
        }
        // explicit tau round-trips through logit space
        match resolve_vt(Some("slot"), Some("0.6"), Some("2")) {
            Ok(Dsv4Vt::Slot { tau_logit, floor }) => {
                assert!((tau_logit - (0.6f32 / 0.4).ln()).abs() < 1e-6);
                assert_eq!(floor, 2);
            }
            other => panic!("slot tau/floor parse broke: {other:?}"),
        }
    }

    #[test]
    fn unknown_and_out_of_range_refuse_by_name() {
        for (p, t, f) in [
            (Some("banana"), None, None),    // unknown policy
            (Some("slot"), Some("0"), None), // tau not in (0,1)
            (Some("slot"), Some("1"), None),
            (Some("slot"), Some("nan"), None),
            (Some("slot"), Some("x"), None),
            (Some("slot"), None, Some("5")), // floor pins window open
            (Some("slot"), None, Some("-1")),
            (None, Some("0.5"), None),      // orphan tau
            (Some("off"), None, Some("2")), // orphan floor
        ] {
            let r = resolve_vt(p, t, f);
            assert!(r.is_err(), "({p:?},{t:?},{f:?}) must refuse, got {r:?}");
            let msg = r.unwrap_err();
            assert!(
                msg.contains("MEMRA_DSV4_VT"),
                "refusal must name the knob: {msg}"
            );
        }
    }

    /// The slot rule is a LEADING-prefix rule: a confident slot after an unconfident
    /// one is never forwarded (chained markov ids past a rejected slot are garbage).
    #[test]
    fn slot_truncation_is_leading_prefix_with_floor() {
        let up = 3.0f32; // sigmoid ~0.95
        let dn = -3.0f32; // sigmoid ~0.05
        assert_eq!(vt_slot_drafts(&[up, up, up, up, up], 0.0, 0), 5);
        assert_eq!(vt_slot_drafts(&[dn, dn, dn, dn, dn], 0.0, 0), 0);
        assert_eq!(vt_slot_drafts(&[up, up, dn, up, up], 0.0, 0), 2);
        // boundary equality counts as confident (>=): tau 0.5 admits c = 0.0
        assert_eq!(vt_slot_drafts(&[0.0, dn, dn, dn, dn], 0.0, 0), 1);
        // floor raises a fully-unconfident round; clamped to the block
        assert_eq!(vt_slot_drafts(&[dn, dn, dn, dn, dn], 0.0, 2), 2);
        assert_eq!(vt_slot_drafts(&[dn, dn], 0.0, 4), 2);
        // NaN slot is unconfident (conservative), never a panic
        assert_eq!(vt_slot_drafts(&[f32::NAN, up, up, up, up], 0.0, 0), 0);
        assert_eq!(vt_slot_drafts(&[], 0.0, 0), 0);
    }

    /// Off must reproduce the pre-policy t_cap expression exactly: with
    /// vt_drafts == k_drafts, (vt_drafts+1).min(k_drafts+1) == k_drafts+1.
    #[test]
    fn off_arm_t_cap_expression_is_identity() {
        for k_drafts in 0usize..=5 {
            let vt_drafts = k_drafts; // the Off branch in the driver
            assert_eq!((vt_drafts + 1).min(k_drafts + 1), k_drafts + 1);
        }
    }
}

#[cfg(test)]
mod penalty_cross_pin_tests {
    use super::{Dsv4PenaltyCfg, dsv4_penalize_row};

    /// The dsv4 explicit-window penalty rule must equal memra-sampling's own
    /// `Sampler::apply_penalties` (the house Keskar law) — pinned by running BOTH on
    /// the same rows/windows and comparing the penalized-greedy argmax, plus a direct
    /// per-element check through the Sampler's greedy path. Drift in either
    /// implementation fails here by name.
    #[test]
    fn penalize_matches_the_sampling_crate_reference() {
        let mk_row = |seed: u32| -> Vec<f32> {
            (0..64u32)
                .map(|i| {
                    let h = i.wrapping_mul(2654435761).wrapping_add(seed);
                    ((h % 2000) as f32 / 100.0) - 10.0
                })
                .collect()
        };
        for (seed, window, last_n, rep, freq, present) in [
            (
                1u32,
                vec![3u32, 3, 3, 7, 12, 3],
                8usize,
                1.8f32,
                0.4f32,
                0.6f32,
            ),
            (2, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4, 1.3, 0.0, 0.0),
            (3, vec![63, 63, 63, 63], 64, 1.0, 1.1, 0.0),
            (4, vec![5], 1, 2.5, 0.7, 1.3),
        ] {
            let row = mk_row(seed);
            // ours
            let mut ours = row.clone();
            dsv4_penalize_row(
                &mut ours,
                &window,
                &Dsv4PenaltyCfg {
                    last_n,
                    repeat: rep,
                    freq,
                    present,
                },
            );
            let our_pick = ours
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            // the house reference: greedy Sampler with penalties + the window as history
            let mut sampler = memra_sampling::Sampler::new(memra_sampling::SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                penalty_last_n: last_n,
                penalty_repeat: rep,
                penalty_freq: freq,
                penalty_present: present,
                seed: 0,
            });
            for &t in &window {
                sampler.accept(t);
            }
            let ref_pick = sampler.sample(&row);
            assert_eq!(
                our_pick, ref_pick,
                "penalized argmax diverged from memra-sampling (seed {seed}): \
                 ours {our_pick} vs reference {ref_pick}"
            );
        }
    }
}

#[cfg(test)]
mod sampled_path_tests {
    use super::{Dsv4SampleCfg, dsv4_pos_uniform, dsv4_sample_row};

    fn cfg(seed: u64) -> Dsv4SampleCfg {
        Dsv4SampleCfg {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            seed,
        }
    }

    /// The identity law's anchor: the draw is a pure function of (row, pos, seed) —
    /// same inputs, same token, always; different positions decouple.
    #[test]
    fn draws_are_position_keyed_and_deterministic() {
        let row = [0.1f32, 2.0, -1.0, 1.9, 0.0];
        let a = dsv4_sample_row(&row, 40, &cfg(20260822)).unwrap();
        for _ in 0..8 {
            assert_eq!(dsv4_sample_row(&row, 40, &cfg(20260822)).unwrap(), a);
        }
        // uniforms at neighboring positions must not be equal (keying is real)
        let u0 = dsv4_pos_uniform(20260822, 40);
        let u1 = dsv4_pos_uniform(20260822, 41);
        let v0 = dsv4_pos_uniform(7, 40);
        assert_ne!(u0, u1);
        assert_ne!(u0, v0);
        assert!((0.0..1.0).contains(&u0));
    }

    /// temperature <= 0 refuses BY NAME (the penalized-greedy footgun class);
    /// bad top_p refuses too.
    #[test]
    fn t0_and_bad_topp_refuse_by_name() {
        let row = [0.0f32, 1.0];
        let mut c = cfg(1);
        c.temperature = 0.0;
        let e = dsv4_sample_row(&row, 0, &c).unwrap_err();
        assert!(e.contains("temperature"), "{e}");
        let mut c2 = cfg(1);
        c2.top_p = 0.0;
        assert!(dsv4_sample_row(&row, 0, &c2).is_err());
    }

    /// top-k 1 and a tight nucleus both collapse to argmax regardless of the draw;
    /// ties break by lowest index (the house ordering).
    #[test]
    fn filters_collapse_to_argmax_and_ties_break_low_index() {
        let row = [0.0f32, 5.0, 5.0, -2.0];
        let mut c = cfg(99);
        c.top_k = 1;
        for pos in 0..64 {
            assert_eq!(dsv4_sample_row(&row, pos, &c).unwrap(), 1);
        }
        let mut c2 = cfg(99);
        c2.top_p = 1e-9; // nucleus keeps exactly the top-1
        for pos in 0..64 {
            assert_eq!(dsv4_sample_row(&row, pos, &c2).unwrap(), 1);
        }
    }

    /// The sampled distribution honors the filtered target: over many positions a
    /// dominant token wins the majority, and a token outside top-k never appears.
    #[test]
    fn draw_frequencies_track_the_filtered_target() {
        let row = [3.0f32, 1.0, 0.0, -50.0];
        let mut c = cfg(20260822);
        c.top_k = 3;
        c.top_p = 1.0;
        let mut counts = [0usize; 4];
        for pos in 0..4096 {
            counts[dsv4_sample_row(&row, pos, &c).unwrap() as usize] += 1;
        }
        assert_eq!(counts[3], 0, "outside top-k must never be drawn");
        assert!(counts[0] > 2600, "p(tok0) ~ 0.84, got {}/4096", counts[0]);
        assert!(counts[1] > 100, "tail token starved: {}", counts[1]);
    }
}
