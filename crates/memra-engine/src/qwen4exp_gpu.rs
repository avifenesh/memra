//! qwen4_exp (Qwen3.8-Flash-Next) GPU EAGER forward — onboarding-ladder phase 7, eager arm.
//!
//! Lane: research/qwen4exp-bringup-20260829 (SEMANTICS.md is the math, ARCH.md the census
//! geometry). Scope = text-only single-request prefill + incremental decode, correctness-
//! gated against the memra-reference oracle (`qwen4exp-gpu-gate`). DELIBERATELY DEFERRED
//! (each resumes in a named perf/serving lane): CUDA graphs, batching > 1, speculative /
//! MTP execution, vision, a gather/compact QSA kernel (the eager arm runs dense attention
//! under the causal∧selection mask per SEMANTICS.md §QSA), and ngram-table async prefetch
//! (the eager gather is synchronous host math).
//!
//! Execution doctrine (the dsv4_gpu precedent): every tensor-scale op runs on the device
//! (cuBLASLt f32 GEMMs + the engine's f32 elementwise/norm/rope kernels + the three
//! qwen4_exp eager kernels in cu/kernels.cu); CONTROL decisions and per-token scalars run
//! as host twins of the exact reference code (MoE routing top-k, the QSA micro-block
//! selection, PLE n-gram hashing, the PLE signed-sqrt gate scalars). Host twins are pinned
//! to their reference functions by name in comments; the gate catches drift loudly because
//! a selection/routing mismatch blows the logit tolerance.
//!
//! Weight residency: everything device-resident f32 (bf16 checkpoints dequantize exactly),
//! EXCEPT (a) the n-gram embedding table — HOST-resident (it is a pure gather source; the
//! 51B-row table never fits device, SEMANTICS.md §Loading notes / HF `_no_placement_params`)
//! — and (b) modelopt-NVFP4 stacked expert banks, which stay AS-STORED on device and
//! dequantize per routed expert through the existing `memra_dsv4_nvfp4_deq_bf16` kernel
//! (macro applied post-upcast in f32 — exact for any finite macro; the real mint's
//! `weight_scale_2` values are amax-derived non-pow2, see `dequant_nvfp4_expert_f32`).
//!
//! Norm-weight convention: this module binds EFFECTIVE norm weights (the reference crate's
//! convention). `from_reference_weights` takes them as-is; `load_from_dir` folds the
//! checkpoint's zero-centered (1+w) values at load for every RMSNorm EXCEPT
//! `linear_attn.norm` (the qwen35 receipt: hf_mapping.rs qwen.py:302-303 exempts exactly
//! that row; SEMANTICS.md §GDN says the GDN program is qwen3_5's except the sigmoid gate).

// Shape lints allowed module-wide (lane/clippy-zero-restore-20260901): this is the qwen4exp
// bring-up lane's kernel-adjacent host code — host twins pinned line-for-line to their
// reference functions — and its just-gated shape is load-bearing, so index loops, control
// flow, and `% == 0` idioms are not reshaped here (is_multiple_of also changes zero-divisor
// semantics from panic to defined). The last four rows (unwrap/question-mark/as_deref/drain)
// are allowed for the same reason, not because they are harmless: their fixes rewrite
// control flow and expression order in the pinned twins. Truly mechanical lints (unused
// imports/mut, no-op casts, needless borrows, doc shape) stay live. NOTE: a module-wide
// allow exempts FUTURE code in this file too, not just the banked sites — when the bring-up
// lanes close, narrowing these to per-site allows is fair game.
#![allow(
    clippy::manual_is_multiple_of,
    clippy::collapsible_if,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_unwrap,
    clippy::needless_question_mark,
    clippy::needless_option_as_deref,
    clippy::extend_with_drain,
    clippy::type_complexity,
    clippy::large_enum_variant
)]

use std::os::raw::c_void;

use cudarc::driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use memra_gguf::model_plan::{
    AttentionPlan, FullAttentionPlan, GatedDeltaNetPlan, GdnGateActivation, MicroBlockIndexPlan,
    MlpPlan, ModelPlan, MoeMlpPlan, PleEmbeddingPlan, ResidualTopology, RopeFactors, RopePlan,
    RouterPlan, TensorPresence, yarn_attention_factor, yarn_frequency_divisors,
};
use memra_gguf::tensor_contract::{LayerTensor, TensorId};
use memra_reference::{ReferenceTensor, ReferenceWeights};

use crate::Engine;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------- weights

/// One gated-residual read/write gate set (attn_/mlp_hyper_connection.*) or the exit
/// mixer (`inject == None`, use_combine=false). Stream-major slicing happens at load so
/// the forward composes from existing per-plane ops (see `gate_read`).
struct GateW {
    /// Per-stream [hidden] slices of hc_norm [wide].
    norm: Vec<CudaSlice<f32>>,
    /// The same norm weights stacked [streams, hidden] — the batched-norm kernel
    /// (`hc_norm_planes_f32`, hcmicro seam) indexes them by stream in one launch.
    norm_stack: CudaSlice<f32>,
    /// Per-stream [rank, hidden] column-slices of input_mix_weight_down [rank, wide].
    down: Vec<CudaSlice<f32>>,
    /// Per-stream [hidden, rank] row-slices of input_mix_weight_up [wide, rank].
    up: Vec<CudaSlice<f32>>,
    /// block_inject_weight [streams, streams*hidden] whole, for the fused inject-gate
    /// kernel (`hc_inject_gates_f32`); `None` for the exit mixer (census carries no
    /// block_inject there).
    inject: Option<CudaSlice<f32>>,
    /// bf16 trunk-residency twins (see `TRUNK_BF16`): the down/up twins are STACKED
    /// across streams ([S, rank, hidden] / [S, hidden, rank]) so the fused read gate
    /// runs each projection as ONE batched `qmatvec_bf16w_f32` launch over the
    /// stream-major slab instead of `streams` cuBLASLt GEMVs.
    down_b16: Option<CudaSlice<u8>>,
    up_b16: Option<CudaSlice<u8>>,
    inject_b16: Option<CudaSlice<u8>>,
}

/// Low-rank width of a gate — from the f32 slices, or from the bf16 stacked twin when
/// `trunk_f32_diet` dropped them (the twin is [S, rank, hidden] bf16).
fn gate_rank(gate: &GateW, hidden: usize, streams: usize) -> Res<usize> {
    if gate.down[0].len() >= hidden {
        return Ok(gate.down[0].len() / hidden);
    }
    match gate.down_b16.as_ref() {
        Some(w) => Ok(w.len() / (2 * streams * hidden)),
        None => Err("qwen4exp_gpu: gate rank underivable (f32 dropped and no bf16 twin)".into()),
    }
}

struct QsaW {
    attn: FullAttentionPlan,
    overlay: MicroBlockIndexPlan,
    wq: CudaSlice<f32>, // [2*nh*hd, H] fused [q|gate] per head
    wk: CudaSlice<f32>, // [nkv*hd, H]
    wv: CudaSlice<f32>, // [nkv*hd, H]
    wo: CudaSlice<f32>, // [H, nh*hd]
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    idx_proj: CudaSlice<f32>, // [(ih+ikv)*id, H]
    /// Indexer norms live host-side: the selection is a host twin of
    /// `memra_reference::micro_block_selection_mask`.
    idx_q_norm: Vec<f32>,
    idx_k_norm: Vec<f32>,
    /// bf16 trunk-residency twins (`TRUNK_BF16` guards + receipts). wq/wk/wv live in
    /// ONE row-stacked twin (proj-stack residency — see `GdnW::proj_b16`).
    proj_b16: Option<CudaSlice<u8>>,
    wo_b16: Option<CudaSlice<u8>>,
    /// YaRN rope tables (long-context lane) — `None` on the shipped config.
    yarn: Option<YarnRopeW>,
}

/// YaRN rope consumption (qwen4_exp long-context lane): the per-pair frequency divisors
/// (device copy for `rope_neox_ffm`, host copy for the indexer twin) plus the derived
/// attention factor on cos/sin. Built once at load from `RopeFactors::Yarn` through the
/// memra-gguf transformers-twin helpers (pinned against the banked receipt). The QSA q/k
/// rope, the indexer q/pooled-k rope, and the MTP draft all consume ONE table — the
/// indexer shares the main rotary (SEMANTICS.md §Rope), enforced at build by the
/// overlay-vs-attention rope-width check.
struct YarnRopeW {
    ff: CudaSlice<f32>,
    ff_host: Vec<f32>,
    mscale: f32,
}

/// Resolve a QSA rope plan into the yarn tables (or `None` for the plain-rope shipped
/// config). PartialRotary/Checkpoint stay refused — this family's plan never emits them.
/// `overlay` = `Some` at single-card load (the shared-table width check); the TP2 half
/// builder passes `None` because the same plan already passed the check on card 0.
fn build_yarn(
    e: &Engine,
    rope: &RopePlan,
    overlay: Option<&MicroBlockIndexPlan>,
    layer: u32,
) -> Res<Option<YarnRopeW>> {
    match rope.factors {
        RopeFactors::None => Ok(None),
        RopeFactors::Yarn {
            factor,
            original_context,
            beta_fast,
            beta_slow,
        } => {
            if let Some(overlay) = overlay
                && overlay.rope_dimensions != rope.dimensions
            {
                return Err(format!(
                    "qwen4exp_gpu: layer {layer} indexer rope width {} != attention rope \
                     width {} — the shared yarn table would be wrong",
                    overlay.rope_dimensions, rope.dimensions
                )
                .into());
            }
            let ff_host = yarn_frequency_divisors(
                rope.dimensions,
                rope.base,
                factor,
                original_context,
                beta_fast,
                beta_slow,
            );
            Ok(Some(YarnRopeW {
                ff: e.htod(&ff_host)?,
                ff_host,
                mscale: yarn_attention_factor(factor),
            }))
        }
        _ => Err(format!(
            "qwen4exp_gpu: layer {layer}: only plain or yarn rope factors are supported"
        )
        .into()),
    }
}

struct GdnW {
    plan: GatedDeltaNetPlan,
    qkv: CudaSlice<f32>,    // [conv_dim, H]
    z: CudaSlice<f32>,      // [nv*hv, H]
    beta: CudaSlice<f32>,   // [nv, H]
    alpha: CudaSlice<f32>,  // [nv, H]
    conv_w: CudaSlice<f32>, // [conv_dim, K]
    a: CudaSlice<f32>,      // [nv] — the reference's `a` multiplier, used as-is by gdn_glog
    dt: CudaSlice<f32>,     // [nv]
    norm: CudaSlice<f32>,   // [hv]
    out: CudaSlice<f32>,    // [H, nv*hv]
    /// bf16 trunk-residency twins (`TRUNK_BF16` guards + receipts). The same-activation
    /// projections live in ONE row-stacked twin [qkv; z; beta; alpha] (proj-stack
    /// residency, VRAM-neutral): the per-mat arm launches against row-offset views, the
    /// proj-stack seam launches the whole stack in one `qmatvec_bf16w_multi4_f32`.
    proj_b16: Option<CudaSlice<u8>>,
    out_b16: Option<CudaSlice<u8>>,
}

enum MixerW {
    Qsa(QsaW),
    Gdn(GdnW),
}

/// One resident half of a routed expert bank (fused gate_up [E, 2ff, H] with gate rows
/// first per expert — SplitExpertGateUp orientation — or down [E, H, ff]).
/// F32 = fixture / bf16 checkpoints (dequantized exactly at load). Nvfp4 = modelopt
/// stacked as-stored (codes [E, out, in/2] u8 + e4m3 scales [E, out, in/16] + finite
/// macros); per routed expert the eager path dequants through the existing dsv4 kernel
/// then upcasts, macro post-upcast in f32 (`dequant_nvfp4_expert_f32`). Halves mix
/// freely — NVFP4 needs in_f % 16 == 0, which geometry (not policy) decides per
/// projection.
enum BankHalf {
    F32(CudaSlice<f32>),
    Nvfp4 {
        codes: CudaSlice<u8>,
        scales: CudaSlice<u8>,
        macros: Vec<f32>,
        /// Device twin of `macros` for the grouped decode path
        /// (`qmatvec_nvfp4_modelopt_sel_f32` folds the macro in its epilogue).
        macros_dev: CudaSlice<f32>,
    },
    /// HOST-resident raw bf16 bank (logical [E, out, in]) — the real-checkpoint gate
    /// residency for BF16 artifacts whose f32 banks exceed device memory (the 360 GB
    /// export: f32 banks ≈ 483 GB). Each routed expert's rows are uploaded and upcast
    /// per forward call (`LoadOptions::host_bf16_banks`); bf16→f32 is exact, so the
    /// value chain equals the device-resident F32 arm. Gate-mode residency only —
    /// never a serving configuration.
    HostBf16(Vec<u8>),
    /// DEVICE-resident raw bf16 bank (logical [E, out, in], row-major bf16 bytes) — the
    /// MTP draft bank residency (mtp-spec lane): the graft ships the 512-expert bank
    /// BF16 (~5 GB device) and the decode path runs per-selected-expert
    /// `qmatvec_bf16w_f32` row-offset launches straight off the resident bytes
    /// (exact-widening products, the trunk-bf16 accumulation class). Half the bytes of
    /// an f32 residency; no dequant materialization.
    DeviceBf16(CudaSlice<u8>),
}

struct ExpertBank {
    gate: BankHalf, // logical [E, ff, H]
    up: BankHalf,   // logical [E, ff, H]
    down: BankHalf, // logical [E, H, ff]
}

struct MoeW {
    plan: MoeMlpPlan,
    router: CudaSlice<f32>, // [E, H]
    /// bf16 residency twin of the router (set_router_bf16 seam; same guards as trunk).
    router_b16: Option<CudaSlice<u8>>,
    bank: ExpertBank,
    shared_gate: CudaSlice<f32>,
    shared_up: CudaSlice<f32>,
    shared_down: CudaSlice<f32>,
    shared_input_gate: Option<CudaSlice<f32>>, // [H]
    /// bf16 residency twins for the shared-expert mats (hcmicro seam; same
    /// representability/geometry guards as the trunk twins). gate/up live in ONE
    /// row-stacked twin (proj-stack residency — see `GdnW::proj_b16`).
    shared_gu_b16: Option<CudaSlice<u8>>,
    shared_down_b16: Option<CudaSlice<u8>>,
}

/// The n-gram embedding table stays HOST-resident (pure gather source).
enum NgramTable {
    F32(Vec<f32>),
    Bf16(Vec<u8>),
}

impl NgramTable {
    fn rows(&self, head_dim: usize) -> usize {
        match self {
            Self::F32(data) => data.len() / head_dim,
            Self::Bf16(bytes) => bytes.len() / 2 / head_dim,
        }
    }

    fn gather_into(&self, row: usize, head_dim: usize, dst: &mut [f32]) {
        match self {
            Self::F32(data) => {
                dst.copy_from_slice(&data[row * head_dim..(row + 1) * head_dim]);
            }
            Self::Bf16(bytes) => {
                let start = row * head_dim * 2;
                for (i, out) in dst.iter_mut().enumerate() {
                    let b = u16::from_le_bytes([bytes[start + 2 * i], bytes[start + 2 * i + 1]]);
                    *out = f32::from_bits(u32::from(b) << 16);
                }
            }
        }
    }
}

struct PleW {
    plan: PleEmbeddingPlan,
    key_proj: Vec<CudaSlice<f32>>, // per-stream [H, embed] row slices of [wide, embed]
    value_proj: CudaSlice<f32>,    // [H, embed]
    norm_key: Vec<CudaSlice<f32>>, // per-stream [H]
    norm_query: Vec<CudaSlice<f32>>,
    norm_conv: Vec<CudaSlice<f32>>,
    conv_w: Vec<CudaSlice<f32>>, // per-stream [H, K] row slices of [wide, K]
    multipliers: Vec<i64>,
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    table: NgramTable,
}

struct LayerW {
    index: u32,
    eps_attn: f32,
    eps_mlp: f32,
    attn_gate: GateW,
    mlp_gate: GateW,
    mixer: MixerW,
    moe: MoeW,
    ple: Option<PleW>,
}

/// The MTP/NextN draft block (SEMANTICS.md §MTP, mtp-spec lane): input fusion =
/// `fc_embedding(zero-centered-RMSNorm(embed(tok)))` broadcast over streams +
/// per-stream `fc_hidden(FLAT GemmaRMSNorm_wide(trunk wide hidden))`; ONE decoder layer
/// (QSA + MoE, own indexer, no PLE) at global index n_trunk; exit through the draft's
/// OWN hyper_connection_mixer into the SHARED trunk lm_head. The post-layer wide state
/// is the K>1 multi-step carrier. Norm rows arrive (1+w)-FOLDED from the loader (the
/// family GemmaRMSNorm convention).
struct MtpW {
    /// Fusion-norm epsilons from the plan (`MtpInputPlan`).
    eps_embed: f32,
    eps_hidden: f32,
    /// mtp.pre_fc_norm_embedding [hidden], folded.
    pre_norm_embed: CudaSlice<f32>,
    /// mtp.pre_fc_norm_hidden [wide] — FLAT over the whole wide vector, folded.
    pre_norm_hidden: CudaSlice<f32>,
    fc_embed: CudaSlice<f32>, // mtp.fc_embedding [hidden, hidden]
    fc_embed_b16: Option<CudaSlice<u8>>,
    fc_hidden: CudaSlice<f32>, // mtp.fc_hidden [hidden, hidden]
    fc_hidden_b16: Option<CudaSlice<u8>>,
    /// The draft decoder layer (index n_trunk; QSA mixer + MoE, bank DeviceBf16 on the
    /// real graft).
    layer: LayerW,
    /// mtp.hyper_connection_mixer (read-only exit gate, no inject).
    mixer: GateW,
}

/// Card-1 draft placement (mtp10): the MTP block's device tensors (weights + the ~5 GB
/// DeviceBf16 expert bank), the draft state, and the draft workspace all live on a SECOND
/// card, with a private full copy of the shared lm head beside them so the draft's head
/// matvec never crosses the bus. What crosses per round is SMALL: the trunk's captured
/// wide rows for the draft replay ((a+1) x wide f32, P2P) and the drafted token ids
/// (4-byte dtoh each). Exactness is untouched by construction — the draft only proposes;
/// the card-0 verify chunk arbitrates. Why this exists (measured, mtp9): the co-resident
/// placement leaves ~2.6 GiB on card 0, which OOMs any spec run on a prompt past ~400
/// tokens — the two-card placement is a PREREQUISITE for agentic-length prompts, not an
/// optimization.
struct MtpDev1 {
    /// Engine ordinal the draft was built on (every draft call must present it).
    dev: usize,
    /// [vocab, H] f32 copy of the shared lm head (the cuBLASLt fallback arm).
    output: CudaSlice<f32>,
    /// bf16 twin of the head copy (the arm `linear_trunk_into` takes at the default
    /// trunk_bf16 seam) — same bytes as card 0's twin, so a draft logit is bit-identical
    /// to its single-card twin at the same row.
    output_b16: Option<CudaSlice<u8>>,
}

/// FR-Spec draft-head trim (DRAFT-REGIME.md law 1, mtp9 lane): the DRAFT scores only
/// the top-N own-gen rank subset, the TARGET verify stays full-vocab — so the spec
/// byte-identity contract is untouched BY CONSTRUCTION and only ACCEPTANCE can move
/// (a token outside the trim set is unproposable, i.e. a guaranteed one-round miss).
/// Rows are gathered D2D from the shared lm head, so the trimmed head is the SAME
/// bytes the full head would have read — a trimmed draft logit is bit-identical to its
/// full-vocab twin at the same row.
struct DraftTrim {
    /// Rows in the trimmed head.
    n: usize,
    /// `d2t[i]` = the TARGET vocab id of trimmed row i (rank order, most frequent first).
    d2t: Vec<u32>,
    /// [n, hidden] gathered bf16 rows (the `qmatvec_bf16w_f32` arm — what the trunk seam
    /// runs by default, and the ONLY residency built when the full head has a bf16 twin:
    /// an f32 twin would cost 2x the bytes for a path the default never takes, and this
    /// artifact's post-load headroom is ~2.5 GiB).
    head_b16: Option<CudaSlice<u8>>,
    /// [n, hidden] gathered f32 rows — built ONLY when there is no bf16 twin to gather
    /// from (the cuBLASLt fallback arm). At least one of the two is always present.
    head: Option<CudaSlice<f32>>,
}

/// The draft head's linear when the trim is armed — `linear_trunk_into`'s arm chain over
/// the trimmed residency (bf16 twin first, f32 fallback), so a trimmed row's value chain
/// is the full head's VERBATIM at the same row.
fn linear_trim_into(
    e: &Engine,
    trim: &DraftTrim,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    t: usize,
    in_f: usize,
) -> Res<()> {
    if trunk_bf16_on() {
        if let Some(w) = trim.head_b16.as_ref() {
            if (2..=12).contains(&t) && verify_mt_on() {
                return launch_qmatvec_bf16w_mt(e, w, 0, x, y, in_f, trim.n, t);
            }
            return launch_qmatvec_bf16w(e, w, x, y, in_f, trim.n, t, 1, 0, 0, in_f, 0);
        }
    }
    let w = trim.head.as_ref().ok_or(
        "qwen4exp_gpu: the draft trim was gathered bf16-only — the f32 head arm needs \
         trunk_bf16 on (or a checkpoint without a bf16 lm-head twin)",
    )?;
    e.linear_device_into(x, w, y, t, in_f, trim.n)
}

pub struct Qwen4ExpGpu {
    pub plan: ModelPlan,
    hidden: usize,
    streams: usize,
    vocab: usize,
    embed_host: Vec<f32>, // [vocab, H] — host row-gather source (reference embed twin)
    output: CudaSlice<f32>, // [vocab, H] lm head (tied to embed when absent)
    /// bf16 trunk-residency twin of the lm head (`TRUNK_BF16` guards + receipts).
    output_b16: Option<CudaSlice<u8>>,
    layers: Vec<LayerW>,
    exit_mixer: GateW,
    exit_eps: f32,
    /// The MTP draft block — present when the checkpoint's mtp.* rows were materialized
    /// (`LoadOptions::load_mtp`, or a fixture whose weights carry them).
    mtp: Option<MtpW>,
    /// Card-1 draft placement (mtp10): when present, `mtp`'s device tensors live on the
    /// SECOND card and this holds that card's private lm-head copy. Every draft call
    /// (mtp_state / mtp_draft_forward / spec_generate's draft engine) must then present
    /// an engine on `mtp_dev1.dev` — enforced, not assumed.
    mtp_dev1: Option<MtpDev1>,
    /// FR-Spec draft-head trim — `None` (full-vocab draft head) unless a caller armed it
    /// with `build_draft_trim`. Default OFF: a trim is a per-model, per-requant rank
    /// artifact (law 1), never an inferred default.
    draft_trim: Option<DraftTrim>,
    /// A built trim PARKED by `set_draft_trim(false)` — the A/B's OFF arm keeps the
    /// gathered head allocated (no per-rep realloc churn) while the draft runs full-vocab.
    draft_trim_parked: Option<DraftTrim>,
    /// Deferred-chain device embed table (mtp11, `SpecOpts::defer`) — `None` until a
    /// caller armed it with `arm_spec_devchain`. Default OFF (flags law): the host
    /// chain is the shipped mtp10 program until the deferred round carries its own
    /// interleaved receipts.
    chain_embed: Option<ChainEmbed>,
}

/// The deferred chain's embed rows, resident on the DRAFT engine (mtp11): the chain's
/// device argmax feeds the next step's embed gather without a host round trip. Rows are
/// raw bf16 when every source value is bf16-clean (this artifact's embed is a bf16
/// export, so `f32 -> bits>>16 -> bits<<16` is the identity and the device gather's
/// QT_BF16 deq reproduces the host `embed_host` row BITWISE — checked value-by-value at
/// arm time, never assumed), else raw f32 (always exact, 2x bytes). With the FR-Spec
/// trim armed the rows are gathered in TRIM-RANK order (row i = embed[d2t[i]]), so the
/// RAW trim-space argmax index gathers its own next-step row and no d2t table crosses
/// to the device; the round's drain maps raw -> target ids through `draft_token`.
struct ChainEmbed {
    table: CudaSlice<u8>,
    qt: i32,
    row_bytes: usize,
    /// Rows in the table == `draft_logits_width()` at arm time (trim.n or vocab).
    rows: usize,
    /// Armed against a live trim (the table is trim-rank-gathered)?
    for_trim: bool,
    /// Device ordinal the table lives on (must be the draft engine's).
    dev: usize,
}

// ---------------------------------------------------------------- state

struct PleState {
    /// Per-stream [pad_ple, H] device history of the NORMED gated value rows
    /// (pad_ple = (K-1)*dilation = 9 on the artifact). Zeros = fresh context.
    conv_hist: Vec<CudaSlice<f32>>,
    /// INCREMENTAL n-gram id cache (`plecache` seam, 262k perf lane). `host_ngram_ids` is a
    /// `ngram_ids` twin over the FULL token history and the caller then slices the last `t`
    /// rows — so a decode step at a 150,000-token fill rebuilds 150,000 rows of hashes to
    /// use ONE. Measured: `ple.host_ngram_gather` is **7.3 ms, 19.5% of a deep decode
    /// token** (PROFILE-11 §5), second only to `qsa.sdpa`, and it is O(context) per token.
    ///
    /// Cacheable EXACTLY, and the proof is local: `shift_right_ignore_eos` at position p
    /// reads `history[p - shift]` and an eos scan that only ever moves left-to-right, so
    /// `ids[token]` is a pure function of `token_ids[..=token]` and NEVER changes when a
    /// token is appended. The cache therefore appends; it never recomputes a row.
    ///
    /// `history` carries the `max_ngram - 1` eos prefix exactly as the twin builds it, and
    /// `last_eos` is the twin's running `last_eos_inclusive` at the end of `history`. On a
    /// rewind (spec reject) both truncate, which is the same discipline the `idxcache` seam
    /// needed for its device mirror.
    ngram_ids: Vec<i64>,
    ngram_history: Vec<i64>,
    ngram_last_eos: i64,
}

// ---- Quantized-cache storage (kvq/idxq lanes) --------------------------------------

/// q8_0 row bytes for a `dim`-wide f32 row (34 B per 32-elem block, zero-padded tail).
fn q8_row_bytes(dim: usize) -> usize {
    dim.div_ceil(32) * 34
}
/// q5_1 row bytes (24 B per 32-elem block).
fn q5_row_bytes(dim: usize) -> usize {
    dim.div_ceil(32) * 24
}

/// QSA KV cache storage. `F32` is the historical exactness arm (every banked receipt);
/// `Q8Q5` stores K rows as q8_0 and V rows as q5_1 byte caches (the owner's asymmetric
/// K=q8/V=q5 default), token-slot-addressed exactly like the f32 rows — rewind stays a
/// position rewrite, replay overwrites slots in place.
enum QsaKvStore {
    F32 {
        k: CudaSlice<f32>, // [cap, nkv*hd] post-norm+rope keys
        v: CudaSlice<f32>, // [cap, nkv*hd]
    },
    Q8Q5 {
        k: CudaSlice<u8>, // [cap * q8_row_bytes(nkv*hd)]
        v: CudaSlice<u8>, // [cap * q5_row_bytes(nkv*hd)]
    },
}

impl QsaKvStore {
    fn is_quant(&self) -> bool {
        matches!(self, QsaKvStore::Q8Q5 { .. })
    }
    fn capacity_rows(&self, kv_dim: usize) -> usize {
        match self {
            QsaKvStore::F32 { k, .. } => k.len() / kv_dim,
            QsaKvStore::Q8Q5 { k, .. } => k.len() / q8_row_bytes(kv_dim),
        }
    }
}

/// Host twin of the device q8_0 quantize warp program (`q4e_quant_q8_block`) — must stay
/// BIT-IDENTICAL to it (the idxcache seam's contract: host- and device-quantized rows
/// interleave in one cache), pinned by the tiny gate's quant-twin arm. `lrintf` under the
/// default rounding mode == `round_ties_even`; the amax fold is order-free (fmaxf over
/// |x| is associative + commutative); the f16 scale conversion is RNE on both sides.
fn host_quant_q8_row(row: &[f32], dim: usize, out: &mut Vec<u8>) {
    for b in 0..dim.div_ceil(32) {
        let mut amax = 0.0f32;
        for l in 0..32 {
            let e = b * 32 + l;
            let x = if e < dim { row[e] } else { 0.0 };
            amax = amax.max(x.abs());
        }
        let d = amax / 127.0f32;
        let mut id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
        // Subnormal-amax guard — mirrors the device kernel (contract totality).
        if !id.is_finite() {
            id = 0.0;
        }
        out.extend_from_slice(&memra_gguf::nvfp4_repack::f32_to_f16_bits(d).to_le_bytes());
        for l in 0..32 {
            let e = b * 32 + l;
            let x = if e < dim { row[e] } else { 0.0 };
            let q = ((x * id).round_ties_even() as i32).clamp(-127, 127);
            out.push(q as i8 as u8);
        }
    }
}

/// Host twin of `q4e_deq_q8` (d single-mul q — one f32 multiply, same bits as the
/// device `__fmul_rn`).
fn host_deq_q8_rows(bytes: &[u8], row0: usize, rows: usize, dim: usize, out: &mut Vec<f32>) {
    let rb = q8_row_bytes(dim);
    for r in row0..row0 + rows {
        let row = &bytes[r * rb..(r + 1) * rb];
        for e in 0..dim {
            let blk = &row[(e >> 5) * 34..];
            let d = memra_gguf::dequant::fp16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let q = blk[2 + (e & 31)] as i8 as f32;
            out.push(d * q);
        }
    }
}

/// Host twin of the device q5_1 quantize warp program (`q4e_quant_q5_block`); min/max
/// folds are order-free (fminf/fmaxf associative + commutative), the rest is per-lane.
fn host_quant_q5_row(row: &[f32], dim: usize, out: &mut Vec<u8>) {
    for b in 0..dim.div_ceil(32) {
        let lane = |l: usize| -> f32 {
            let e = b * 32 + l;
            if e < dim { row[e] } else { 0.0 }
        };
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for l in 0..32 {
            mn = mn.min(lane(l));
            mx = mx.max(lane(l));
        }
        let d = (mx - mn) / 31.0f32;
        let mut id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
        // Subnormal-amax guard — mirrors the device kernel (contract totality).
        if !id.is_finite() {
            id = 0.0;
        }
        let q5 = |l: usize| -> u32 {
            (((lane(l) - mn) * id).round_ties_even() as i32).clamp(0, 31) as u32
        };
        let mut qh = 0u32;
        for l in 0..32 {
            qh |= ((q5(l) >> 4) & 1) << l;
        }
        out.extend_from_slice(&memra_gguf::nvfp4_repack::f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&memra_gguf::nvfp4_repack::f32_to_f16_bits(mn).to_le_bytes());
        out.extend_from_slice(&qh.to_le_bytes());
        for l in 0..16 {
            out.push(((q5(l) & 0x0F) | ((q5(l + 16) & 0x0F) << 4)) as u8);
        }
    }
}

/// Host twin of `q4e_deq_q5` (`__fmaf_rn(d, q5, m)` == `f32::mul_add`).
fn host_deq_q5_rows(bytes: &[u8], row0: usize, rows: usize, dim: usize, out: &mut Vec<f32>) {
    let rb = q5_row_bytes(dim);
    for r in row0..row0 + rows {
        let row = &bytes[r * rb..(r + 1) * rb];
        for e in 0..dim {
            let blk = &row[(e >> 5) * 24..];
            let d = memra_gguf::dequant::fp16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let m = memra_gguf::dequant::fp16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
            let lane = e & 31;
            let lo = if lane < 16 {
                blk[8 + lane] & 0x0F
            } else {
                blk[8 + lane - 16] >> 4
            };
            let q5 = (lo as u32) | (((qh >> lane) & 1) << 4);
            out.push(d.mul_add(q5 as f32, m));
        }
    }
}

/// Host twin of the device `__float2bfloat16` RNE conversion (finite domain; the raw
/// keys are finite projection outputs — NaN handling is not part of the pin).
fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Indexer raw-key HOST cache (idxq lane): rows of `idx_dim` keys, stored f32
/// (historical), q8_0 blocks, or bf16. Consumed ONLY through `rows_f32` into the fp32
/// mean-pooling — quantize the cache, dequant at read, pooling math identical.
enum IdxRawCache {
    F32(Vec<f32>),
    Q8(Vec<u8>),
    Bf16(Vec<u16>),
}

impl IdxRawCache {
    fn new(mode: IdxQMode) -> Self {
        match mode {
            IdxQMode::F32 => IdxRawCache::F32(Vec::new()),
            IdxQMode::Q8 => IdxRawCache::Q8(Vec::new()),
            IdxQMode::Bf16 => IdxRawCache::Bf16(Vec::new()),
        }
    }
    fn rows(&self, idx_dim: usize) -> usize {
        match self {
            IdxRawCache::F32(v) => v.len() / idx_dim,
            IdxRawCache::Q8(v) => v.len() / q8_row_bytes(idx_dim),
            IdxRawCache::Bf16(v) => v.len() / idx_dim,
        }
    }
    fn truncate_rows(&mut self, rows: usize, idx_dim: usize) {
        match self {
            IdxRawCache::F32(v) => v.truncate(rows * idx_dim),
            IdxRawCache::Q8(v) => v.truncate(rows * q8_row_bytes(idx_dim)),
            IdxRawCache::Bf16(v) => v.truncate(rows * idx_dim),
        }
    }
    /// Append `n` rows given as f32 (host-side quantize twin — bit-identical to the
    /// device append kernels, so host/device-quantized rows interleave freely).
    fn append_rows_f32(&mut self, rows: &[f32], n: usize, idx_dim: usize) {
        match self {
            IdxRawCache::F32(v) => v.extend_from_slice(&rows[..n * idx_dim]),
            IdxRawCache::Q8(v) => {
                for r in 0..n {
                    host_quant_q8_row(&rows[r * idx_dim..(r + 1) * idx_dim], idx_dim, v);
                }
            }
            IdxRawCache::Bf16(v) => {
                v.extend(rows[..n * idx_dim].iter().map(|&x| f32_to_bf16_rne(x)));
            }
        }
    }
    /// Dequant rows [row0, row0+n) to f32 (the pooling read).
    fn rows_f32(&self, row0: usize, n: usize, idx_dim: usize, out: &mut Vec<f32>) {
        out.clear();
        match self {
            IdxRawCache::F32(v) => out.extend_from_slice(&v[row0 * idx_dim..(row0 + n) * idx_dim]),
            IdxRawCache::Q8(v) => host_deq_q8_rows(v, row0, n, idx_dim, out),
            IdxRawCache::Bf16(v) => out.extend(
                v[row0 * idx_dim..(row0 + n) * idx_dim]
                    .iter()
                    .map(|&b| memra_gguf::dequant::bf16_to_f32(b)),
            ),
        }
    }
}

/// Indexer raw-key DEVICE cache (idxcache seam), format-matched to the host cache.
/// Bf16 rows live as u16; Q8 rows as q8_0 bytes. The host cache materializes from these
/// by dtoh VERBATIM (no re-quant), so lazy materialization stays bit-identical.
enum IdxRawDev {
    F32(CudaSlice<f32>),
    Q8(CudaSlice<u8>),
    Bf16(CudaSlice<u16>),
}

/// Pay the idxcache lazy-materialization debt: dtoh device rows [host_rows, dev_rows)
/// into the host cache VERBATIM — format-matched bytes, no re-quant, so the seam's
/// bit-identity contract holds per format.
fn idx_materialize_host(
    e: &Engine,
    raw_keys: &mut IdxRawCache,
    raw_dev: &Option<IdxRawDev>,
    raw_dev_rows: usize,
    idx_dim: usize,
) -> Res<()> {
    let host_rows = raw_keys.rows(idx_dim);
    if raw_dev_rows <= host_rows {
        return Ok(());
    }
    let m = raw_dev
        .as_ref()
        .ok_or("idxcache: rows counted without a cache")?;
    match (m, raw_keys) {
        (IdxRawDev::F32(d), IdxRawCache::F32(h)) => {
            let delta = e.dtoh_view(&d.slice(host_rows * idx_dim..raw_dev_rows * idx_dim))?;
            h.extend_from_slice(&delta);
        }
        (IdxRawDev::Q8(d), IdxRawCache::Q8(h)) => {
            let rb = q8_row_bytes(idx_dim);
            let delta = e.dtoh_u8_view(&d.slice(host_rows * rb..raw_dev_rows * rb))?;
            h.extend_from_slice(&delta);
        }
        (IdxRawDev::Bf16(d), IdxRawCache::Bf16(h)) => {
            let delta = e
                .gpu
                .stream()
                .clone_dtoh(&d.slice(host_rows * idx_dim..raw_dev_rows * idx_dim))?;
            e.gpu.stream().synchronize()?;
            h.extend_from_slice(&delta);
        }
        _ => return Err("idxcache: device/host raw-key formats disagree".into()),
    }
    Ok(())
}

/// The idxq selection-identity audit twin (instrument): parallel f32 raw/pooled caches
/// fed from the per-chunk idx_proj dtoh, selection recomputed on host per scored row.
struct IdxAudit {
    raw_f32: IdxRawCache, // always the F32 variant
    pooled_f32: Vec<f32>,
}

enum MixerState {
    Qsa {
        kv: QsaKvStore,
        /// Indexer RAW key cache — pre-norm, pre-rope, host-resident
        /// (`update_indexer`, SEMANTICS.md §QSA: 128 dims/token/QSA-layer). Precision
        /// per the idxq lane (f32 / q8_0 / bf16), latched at alloc.
        raw_keys: IdxRawCache,
        /// POOLED indexer key cache — the per-block mean/k_layernorm/rope form the
        /// scorer consumes, host-resident, one row per COMPLETE block. A block's pooled
        /// key depends only on its 4 raw keys + its start position, never on the query
        /// row, so it is computed ONCE (bit-identical to the historical per-row
        /// recompute: same op order per block) and extended as blocks complete.
        /// Truncated with `raw_keys` on rewind.
        pooled_keys: Vec<f32>,
        /// DEVICE mirror of `pooled_keys` for the device scorer (long-context lane):
        /// same rows, grown by H2D of the delta as blocks complete. `None` until the
        /// scorer engages (below the drop point nothing is scored at all).
        pooled_dev: Option<CudaSlice<f32>>,
        /// Rows currently mirrored (<= pooled_keys.len()/head_dim).
        pooled_dev_rows: usize,
        /// DEVICE raw-key cache (devtwin stage 3, `idxcache` seam): the k-part rows
        /// appended d2d as chunks land; below the selection horizon `raw_keys` LAGS
        /// this (the lazy host materialization dtohs the delta at the first scored
        /// chunk). Row r here is absolute cache row r — rewind clamps `raw_dev_rows`
        /// alongside the host truncation.
        raw_dev: Option<IdxRawDev>,
        /// Rows valid in `raw_dev` (>= raw_keys rows while the seam is on).
        raw_dev_rows: usize,
        /// idxq selection-identity audit twin (instrument, `MEMRA_Q4E_IDXQ_AUDIT=1`).
        idx_audit: Option<Box<IdxAudit>>,
    },
    Gdn {
        conv: CudaSlice<f32>,  // [pad, conv_dim] raw pre-conv qkv history rows
        state: CudaSlice<f32>, // [nv, hv, hk] recurrent matrix (reference layout)
    },
}

struct LayerState {
    mixer: MixerState,
    ple: Option<PleState>,
}

/// Per-GDN-layer verify-chunk stash (mtp-spec lane): per-column recurrent snapshots +
/// the chunk's conv-rewind inputs. Sized once at `spec_arm` (k_cap columns).
struct GdnStash {
    /// [k_cap, nv*hv*hk] — recurrent state AFTER column i (D2D snapshot per column).
    states: CudaSlice<f32>,
    /// [pad, conv_dim] — pre-chunk conv history (rewind rebuild input).
    conv_pre: CudaSlice<f32>,
    /// [k_cap, conv_dim] — the chunk's raw pre-conv qkv rows (rewind rebuild input).
    qkv_rows: CudaSlice<f32>,
    /// Verify SCAN-CHAIN segment graph (mtp9, `set_verify_graphs`): dwconv + the t
    /// per-column {scan step, state snapshot} launches + the conv-history roll, captured
    /// at ONE chunk width. The chain is serially DEPENDENT (every column reads and writes
    /// the recurrent state), so each launch's issue latency is fully exposed — this is the
    /// densest all-device launch run in the verify chunk (t=6: 14 launches x 36 GDN
    /// layers). `Some((t, graph))`; a chunk at a different t invalidates it.
    scan_graph: Option<(usize, GraphEntry)>,
    /// Chunk widths already WARMED at: the first chunk of a width runs eager so every
    /// workspace slot is allocated and parked outside the capture region (allocations
    /// inside a capture become graph mem nodes — the trunk's draft-graph lesson).
    scan_warm: Option<usize>,
}

/// Per-PLE-layer verify-chunk stash: pre-chunk conv history + the chunk's normed
/// gated-value rows, per stream.
struct PleStash {
    hist_pre: Vec<CudaSlice<f32>>,    // per stream [pad_ple, hidden]
    normed_rows: Vec<CudaSlice<f32>>, // per stream [k_cap, hidden]
}

/// The verify-chunk instrument (mtp-spec lane). Armed by `spec_arm`; while armed,
/// every forward captures the trunk's FINAL WIDE rows at their absolute positions
/// (the draft's hidden seeds) and every 1 < t <= k_cap chunk (a) runs the EXACT row
/// programs (each row bit-identical to the t == 1 decode program — the spec
/// byte-identity contract) and (b) stashes per-column GDN/PLE state so
/// `verify_rewind` can drop rejected columns without replay.
pub struct VerifyStash {
    k_cap: usize,
    /// The live chunk (base_pos, t) — set by the last exact chunk, consumed by rewind.
    chunk: Option<(usize, usize)>,
    /// The last FUSED verify chunk (`vfuse` cost instrument), for the rewind refusal
    /// message only. A fused chunk leaves no per-column stash, so the state it produced
    /// cannot be rewound; naming the shape keeps that refusal readable as the seam's
    /// documented limit instead of an internal inconsistency.
    fused_chunk: Option<(usize, usize)>,
    gdn: Vec<Option<GdnStash>>,
    ple: Vec<Option<PleStash>>,
    /// Trunk final wide rows, RING-slotted: absolute row r lives at slot
    /// `r % ring_rows`. `ring_rows == capacity` (the `spec_arm` default) is the
    /// historical whole-history layout; the long-context arm (`spec_arm_ring`) bounds it
    /// (see that doc for the freshness contract).
    wide: CudaSlice<f32>,
    ring_rows: usize,
    /// Card-1 mirror of `wide` (mtp10 dev1 draft placement): the draft's hidden seeds,
    /// P2P-copied row-range by row-range (prefill once, then (a+1) rows per round).
    /// Allocated lazily by `spec_generate` when the draft engine is a different card.
    wide_dev1: Option<CudaSlice<f32>>,
    /// Per-row argmax of the last exact chunk (device argmax, 4t-byte dtoh).
    argmax: Vec<u32>,
    /// Device argmax staging [k_cap].
    toks: CudaSlice<u32>,
    /// Skip the [t, vocab] logits dtoh on exact chunks and fill `argmax` instead
    /// (forward returns an EMPTY vec in that mode — the spec loop's fast path).
    want_argmax: bool,
    /// Extend the argmax fast path to t == 1 forwards (mtp11 deferred round): the
    /// zero-draft verify and the dynk plain tail commit a device argmax + 4-byte dtoh
    /// instead of the full [1, vocab] row + host scan (bit-identical token by the
    /// argmax-gate contract). Only honored with `want_argmax` (greedy non-trace).
    want_argmax_t1: bool,
    /// Big (t > k_cap, i.e. prefill) forwards dtoh only the LAST logits row (mtp11):
    /// the spec loop consumes exactly one row for x0, and the full block is ~1 MB/row.
    /// Exact chunks and t == 1 steps are untouched (sampled verify samples EVERY row).
    last_row_only: bool,
}

pub struct Qwen4ExpState {
    pos: usize,
    capacity: usize,
    /// Workspace-slot reserve unit (tokens). Equals `capacity` from `alloc_state` (the
    /// historical behavior: one allocation serves the largest possible chunk); a
    /// long-context state (`alloc_state_reserve`) caps it at the CHUNK bound so a
    /// 1M-capacity state does not reserve 1M-token transients. Forwards longer than
    /// this still work (slots grow), they just reallocate.
    reserve: usize,
    /// Full token history (host). PLE n-gram hashing needs the EOS-segment structure of
    /// the whole context (reference `shift_right_ignore_eos`), and eager memory cost is
    /// 4 B/token.
    tokens: Vec<u32>,
    layers: Vec<LayerState>,
    /// Named-slot step workspace (perf lane item 2a — see `StepPool`).
    ws: StepPool,
    /// Captured decode-step graphs (perf lane item 2b — see `StepGraphs`).
    graphs: StepGraphs,
    /// TP2 half-state (perf round 3). `Some` after the first `decode_step_tp2`: the
    /// single-card mixer state is migrated into per-card halves and goes STALE — a
    /// TP2-touched state refuses single-card forwards (fresh state per mode; the A/B
    /// harness allocates per arm).
    tp2: Option<Tp2State>,
    /// Verify-chunk stash (mtp-spec lane), armed by `spec_arm`.
    verify: Option<VerifyStash>,
}

/// Named-slot device workspace for the forward step (perf lane item 2a: PROFILE-0
/// counted 11,366 pooled allocs + 1,685 memsets per token; 2,234 allocs remained after
/// round 1). Every step-transient buffer is TAKEN from a named slot and PUT back at its
/// last use; with the seam ON (`set_step_ws`, default per receipts) the same CudaSlice —
/// and therefore the same device ADDRESS — serves every step, which both removes the
/// cuMemAllocAsync/FreeAsync churn and is the address-stability prerequisite for CUDA
/// graph capture (item 2b). With the seam OFF every take allocates fresh and every put
/// drops — byte-identical to the prior pooled-alloc behavior, the A/B twin. A slot is
/// allocated at `reserve` elements on first take (capacity-derived at the call sites)
/// so a growing shape (the decode mask) never reallocates mid-run.
#[derive(Default)]
struct StepPool {
    f32s: std::collections::BTreeMap<&'static str, CudaSlice<f32>>,
    i32s: std::collections::BTreeMap<&'static str, CudaSlice<i32>>,
    u8s: std::collections::BTreeMap<&'static str, CudaSlice<u8>>,
    u64s: std::collections::BTreeMap<&'static str, CudaSlice<u64>>,
}

/// Per-stream slot names (hc_count is 4 on the artifact, 2 on the tiny plan; the loader
/// refuses streams > 8).
/// TP2-prefill exit slots: the last row of each plane, copied into t == 1 buffers so
/// the decode exit segment runs unchanged on a chunk's final row.
const EXIT_PLANE_SLOTS: [&str; 8] = [
    "exit.p0", "exit.p1", "exit.p2", "exit.p3", "exit.p4", "exit.p5", "exit.p6", "exit.p7",
];
const PLANE_SLOTS: [&str; 8] = [
    "plane.0", "plane.1", "plane.2", "plane.3", "plane.4", "plane.5", "plane.6", "plane.7",
];
const INJECT_SLOTS: [&str; 8] = [
    "hc.inj.0", "hc.inj.1", "hc.inj.2", "hc.inj.3", "hc.inj.4", "hc.inj.5", "hc.inj.6", "hc.inj.7",
];

impl StepPool {
    fn take_f32(
        &mut self,
        e: &Engine,
        name: &'static str,
        len: usize,
        reserve: usize,
    ) -> Res<CudaSlice<f32>> {
        if step_ws_on() {
            if let Some(buf) = self.f32s.remove(name) {
                if buf.len() >= len {
                    return Ok(buf);
                }
            }
            e.uninit(len.max(reserve))
        } else {
            e.uninit(len)
        }
    }

    fn put_f32(&mut self, name: &'static str, buf: CudaSlice<f32>) {
        if step_ws_on() {
            self.f32s.insert(name, buf);
        }
    }

    fn take_i32(
        &mut self,
        e: &Engine,
        name: &'static str,
        host: &[i32],
        reserve: usize,
    ) -> Res<CudaSlice<i32>> {
        if step_ws_on() {
            let mut buf = match self.i32s.remove(name) {
                Some(buf) if buf.len() >= host.len() => buf,
                _ => e.alloc_uninit::<i32>(host.len().max(reserve))?,
            };
            let mut view = buf.slice_mut(0..host.len());
            e.gpu.stream().memcpy_htod(host, &mut view)?;
            Ok(buf)
        } else {
            e.htod_i32(host)
        }
    }

    fn put_i32(&mut self, name: &'static str, buf: CudaSlice<i32>) {
        if step_ws_on() {
            self.i32s.insert(name, buf);
        }
    }

    /// Take an i32 slot WITHOUT uploading (device router: contents arrive from the
    /// `qwen4exp_route_topk_f32` launch — the take_u8 discipline for i32).
    fn take_i32_slot(
        &mut self,
        e: &Engine,
        name: &'static str,
        len: usize,
        reserve: usize,
    ) -> Res<CudaSlice<i32>> {
        if step_ws_on() {
            if let Some(buf) = self.i32s.remove(name) {
                if buf.len() >= len {
                    return Ok(buf);
                }
            }
        }
        e.alloc_uninit::<i32>(len.max(reserve))
    }

    fn take_f32_h2d(
        &mut self,
        e: &Engine,
        name: &'static str,
        host: &[f32],
        reserve: usize,
    ) -> Res<CudaSlice<f32>> {
        if step_ws_on() {
            let mut buf = match self.f32s.remove(name) {
                Some(buf) if buf.len() >= host.len() => buf,
                _ => e.uninit(host.len().max(reserve))?,
            };
            let mut view = buf.slice_mut(0..host.len());
            e.gpu.stream().memcpy_htod(host, &mut view)?;
            Ok(buf)
        } else {
            e.htod(host)
        }
    }

    fn take_u8_h2d(
        &mut self,
        e: &Engine,
        name: &'static str,
        host: &[u8],
        reserve: usize,
    ) -> Res<CudaSlice<u8>> {
        if step_ws_on() {
            let mut buf = match self.u8s.remove(name) {
                Some(buf) if buf.len() >= host.len() => buf,
                _ => e.alloc_u8_uninit(host.len().max(reserve))?,
            };
            let mut view = buf.slice_mut(0..host.len());
            e.gpu.stream().memcpy_htod(host, &mut view)?;
            Ok(buf)
        } else {
            e.htod_bytes(host)
        }
    }

    fn put_u8(&mut self, name: &'static str, buf: CudaSlice<u8>) {
        if step_ws_on() {
            self.u8s.insert(name, buf);
        }
    }

    /// Take a u8 slot WITHOUT uploading (TP2 pack blobs: contents arrive via
    /// `upsert_u8` before the consuming segment runs/replays).
    fn take_u8(
        &mut self,
        e: &Engine,
        name: &'static str,
        len: usize,
        reserve: usize,
    ) -> Res<CudaSlice<u8>> {
        if step_ws_on() {
            if let Some(buf) = self.u8s.remove(name) {
                if buf.len() >= len {
                    return Ok(buf);
                }
            }
        }
        e.alloc_u8_uninit(len.max(reserve))
    }

    /// H2D into a PARKED u8 slot, seeding it on first use (the write_i32 discipline
    /// with a bootstrap arm — a captured graph bakes the slot address, so after the
    /// first take the buffer must never rebind).
    fn upsert_u8(
        &mut self,
        e: &Engine,
        name: &'static str,
        host: &[u8],
        reserve: usize,
    ) -> Res<()> {
        if !self.u8s.contains_key(name) {
            let buf = self.take_u8(e, name, host.len(), reserve)?;
            self.put_u8(name, buf);
        }
        let buf = self
            .u8s
            .get_mut(name)
            .ok_or_else(|| format!("step workspace: slot {name} is not parked"))?;
        if buf.len() < host.len() {
            return Err(format!("step workspace: slot {name} is too small").into());
        }
        let mut view = buf.slice_mut(0..host.len());
        e.gpu.stream().memcpy_htod(host, &mut view)?;
        Ok(())
    }

    /// Borrow a parked u8 slot without removing it (graph segments read the pack blob
    /// a driver upsert wrote).
    fn peek_u8(&self, name: &'static str) -> Res<&CudaSlice<u8>> {
        self.u8s
            .get(name)
            .ok_or_else(|| format!("step workspace: slot {name} is not parked").into())
    }

    fn take_u64_h2d(
        &mut self,
        e: &Engine,
        name: &'static str,
        host: &[u64],
        reserve: usize,
    ) -> Res<CudaSlice<u64>> {
        if step_ws_on() {
            let mut buf = match self.u64s.remove(name) {
                Some(buf) if buf.len() >= host.len() => buf,
                _ => e.alloc_uninit::<u64>(host.len().max(reserve))?,
            };
            let mut view = buf.slice_mut(0..host.len());
            e.gpu.stream().memcpy_htod(host, &mut view)?;
            Ok(buf)
        } else {
            e.htod_u64(host)
        }
    }

    fn put_u64(&mut self, name: &'static str, buf: CudaSlice<u64>) {
        if step_ws_on() {
            self.u64s.insert(name, buf);
        }
    }

    /// Borrow a parked slot without removing it (graph driver: the router logits dtoh
    /// reads the slot a captured graph wrote).
    fn peek_f32(&self, name: &'static str) -> Res<&CudaSlice<f32>> {
        self.f32s
            .get(name)
            .ok_or_else(|| format!("step workspace: slot {name} is not parked").into())
    }

    /// H2D into an EXISTING slot in place (graph driver: per-step routing inputs into
    /// the addresses the captured graph baked). Errors if the slot is missing or short —
    /// a captured graph must never silently rebind.
    fn write_i32(&mut self, e: &Engine, name: &'static str, host: &[i32]) -> Res<()> {
        let buf = self
            .i32s
            .get_mut(name)
            .ok_or_else(|| format!("step workspace: slot {name} is not parked"))?;
        if buf.len() < host.len() {
            return Err(format!("step workspace: slot {name} is too small").into());
        }
        let mut view = buf.slice_mut(0..host.len());
        e.gpu.stream().memcpy_htod(host, &mut view)?;
        Ok(())
    }

    fn write_f32(&mut self, e: &Engine, name: &'static str, host: &[f32]) -> Res<()> {
        let buf = self
            .f32s
            .get_mut(name)
            .ok_or_else(|| format!("step workspace: slot {name} is not parked"))?;
        if buf.len() < host.len() {
            return Err(format!("step workspace: slot {name} is too small").into());
        }
        let mut view = buf.slice_mut(0..host.len());
        e.gpu.stream().memcpy_htod(host, &mut view)?;
        Ok(())
    }
}

/// Captured decode-step graphs (perf lane item 2b). Layer graphs bake the workspace
/// slot ADDRESSES (StepPool, item 2a), the state buffers, and the resident weights, so
/// they live beside the state they were captured against. `a[l]` = the device-only
/// layer interior (attn read gate → GDN mixer → write → mlp read gate) for GDN layers
/// without a PLE block; `b[l]` = the grouped-MoE tail (sel matvecs → shared expert →
/// mlp write) for all-NVFP4 layers; `exit` = exit mixer + lm head. QSA layers keep
/// their eager interior (the indexer host twin + mask h2d live there). Capture is
/// no-warmup (`capture_graph_retained_nowarm`): stream capture enqueues WITHOUT
/// executing, and the step's side effects (GDN state/conv advance, plane writes) must
/// not run twice.
#[derive(Default)]
struct StepGraphs {
    /// The first graph-eligible decode step runs EAGER to warm every slot (allocations
    /// inside a capture region become graph mem nodes — the draft-graph lesson).
    warm: bool,
    a: Vec<Option<GraphEntry>>,
    b: Vec<Option<GraphEntry>>,
    exit: Option<GraphEntry>,
}

type GraphEntry = (
    cudarc::driver::CudaGraph,
    Vec<Box<dyn std::any::Any + Send>>,
);

impl Qwen4ExpState {
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// Per-layer parity capture from a prefill — mirrors the transformers hidden-goldens
/// hook points (`make-goldens.py`): decoder-layer outputs on the WIDE stream and the
/// exit `hyper_connection_mixer` output.
pub struct PrefillCapture {
    /// One entry per trunk layer: post-layer wide rows, token-major [t, streams*hidden].
    pub layer_wide: Vec<Vec<f32>>,
    /// Exit mixer output [t, hidden].
    pub exit_mixed: Vec<f32>,
}

// ---------------------------------------------------------------- profiling (perf lane)

/// Wall-clock section profiler for the eager forward (perf lane:
/// research/qwen4exp-bringup-20260829/perf/). Disabled (default) the wrappers are
/// zero-cost passthroughs; enabled, every section boundary synchronizes the stream so a
/// section's time covers everything it queued. Synchronization itself distorts the step
/// total — the receipt therefore always banks the UNPROFILED warm ms/token beside the
/// profiled table and reads shares, not absolutes, from the latter.
pub mod prof {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    thread_local! {
        static STATE: RefCell<Option<BTreeMap<&'static str, (f64, u64)>>> =
            const { RefCell::new(None) };
    }

    /// Start accumulating (resets any previous accumulation).
    pub fn enable() {
        STATE.with(|s| *s.borrow_mut() = Some(BTreeMap::new()));
    }

    pub fn on() -> bool {
        STATE.with(|s| s.borrow().is_some())
    }

    /// Drain the accumulated rows (section, total_seconds, calls) and disable.
    pub fn take() -> Vec<(&'static str, f64, u64)> {
        STATE.with(|s| {
            s.borrow_mut()
                .take()
                .map(|map| map.into_iter().map(|(k, (t, c))| (k, t, c)).collect())
                .unwrap_or_default()
        })
    }

    pub(super) fn add(name: &'static str, seconds: f64) {
        STATE.with(|s| {
            if let Some(map) = s.borrow_mut().as_mut() {
                let entry = map.entry(name).or_insert((0.0, 0));
                entry.0 += seconds;
                entry.1 += 1;
            }
        });
    }
}

/// Grouped selected-experts decode path (attack (a) of the perf lane). Default ON —
/// better-wins-by-default with the interleaved A/B receipts in
/// research/qwen4exp-bringup-20260829/perf/; the per-expert path stays as the prefill
/// executor, the non-NVFP4 arm, and the A/B twin. Flipped per-arm by the gate binary.
static MOE_SEL_PATH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_moe_sel_path(on: bool) {
    MOE_SEL_PATH.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn moe_sel_path_on() -> bool {
    MOE_SEL_PATH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fused hyper-connection read gate (attack (c)). Default ON with the interleaved A/B
/// receipts in the perf lane; the unfused chain stays as the A/B twin (`gate_read_legacy`)
/// and as the readable statement of the reference program.
static HC_FUSED_GATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_hc_fused_gate(on: bool) {
    HC_FUSED_GATE.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn hc_fused_gate_on() -> bool {
    HC_FUSED_GATE.load(std::sync::atomic::Ordering::Relaxed)
}

/// bf16 trunk residency (perf lane item: PROFILE-1 residual §2 — gdn.proj/qsa.proj/
/// lm_head/gate GEMVs are memory-bound on f32 trunk weights at ~1.3 TB/s). Dense trunk
/// mats keep their f32 residency AND gain a bf16 twin when (a) every value is exactly
/// bf16-representable (true for BF16 checkpoints — dequant was exact, so the twin equals
/// the artifact bytes) and (b) in_f % 8 == 0 (the matvec kernel's uint4 vector width) —
/// geometry/value guards, never policy. Default ON with the interleaved A/B receipts in
/// research/qwen4exp-bringup-20260829/perf/PROFILE-2.md; the f32 cuBLASLt path stays
/// resident as the A/B twin (`--ab-seam trunk`) and the fallback for guarded tensors.
static TRUNK_BF16: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_trunk_bf16(on: bool) {
    TRUNK_BF16.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn trunk_bf16_on() -> bool {
    TRUNK_BF16.load(std::sync::atomic::Ordering::Relaxed)
}

/// INSTRUMENT-ONLY: run a `HeadMode::All` single-card forward on the GROUPED MoE executor
/// instead of the per-expert one.
///
/// **Default OFF, and that is a decision with a reason, not an implementation accident.**
/// OFF is byte-for-byte today's behavior: `HeadMode::All` selects the per-expert executor,
/// which is the reference-shaped program the goldens capture and every hidden/greedy
/// exactness receipt in this lane rest on. Flipping the default would silently re-base
/// every one of those receipts, so OFF is the only safe default and there is no perf
/// argument on the other side (the per-expert path is the SLOW one — see the executor
/// comment at the `grouped` selection).
///
/// ON exists for exactly one caller: the TP2-prefill CLASS gate's PRIME regime. That regime
/// compares an all-rows single-card forward against an all-rows TP2 forward, and TP2's
/// `tp2_moe_rows` is grouped on both cards. With this flag OFF the comparison therefore
/// straddles TWO independent variables — the TP2 expert-half split AND the
/// grouped-vs-per-expert executor difference — and the executor term DOMINATES: measured on
/// this artifact, grouped-vs-grouped lands at 1.4e-5 while per-expert-vs-grouped lands at
/// 2e-3..4e-3, and the tiny gate's own `prefill-extend` arm prices the executor difference
/// alone at 1.865e-4 on a fixture. A band calibrated against the straddled number would be
/// ~100x too loose for the question the gate is asking, which is the same "calibrated
/// against nothing" failure the two-regime gate was written to end.
///
/// It is an instrument, not a serving seam: nothing in a serving path reads it, and
/// long-context prefill already rides the grouped program through `HeadMode::LastRow`.
static PREFILL_GROUPED_ALL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_prefill_grouped_all(on: bool) {
    PREFILL_GROUPED_ALL.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn prefill_grouped_all_on() -> bool {
    PREFILL_GROUPED_ALL.load(std::sync::atomic::Ordering::Relaxed)
}

/// Allocation-stable decode step (perf lane item 2a — see `StepPool`). Default ON with
/// the interleaved A/B receipts in PROFILE-2.md; OFF reproduces the pooled-alloc
/// behavior exactly (`--ab-seam ws`).
static STEP_WS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_step_ws(on: bool) {
    STEP_WS.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn step_ws_on() -> bool {
    STEP_WS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Decode-step CUDA graphs (perf lane item 2b — see `StepGraphs`). Replay is
/// bit-identical to the ws-eager path by construction (same kernels, same launch
/// parameters, same baked addresses, same order — only the CPU issue path changes), so
/// the graph A/B's rep-0 chains must be IDENTICAL, a stronger bar than the
/// accumulation-class seams. Requires the step workspace (item 2a); disabled while the
/// section profiler is on (sync boundaries cannot cross a replay) and during prefill
/// capture. Default ON with the PROFILE-2.md receipts; `--ab-seam graph`.
static DECODE_GRAPHS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_decode_graphs(on: bool) {
    DECODE_GRAPHS.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn decode_graphs_on() -> bool {
    DECODE_GRAPHS.load(std::sync::atomic::Ordering::Relaxed)
}

/// VERIFY scan-chain segment graphs (mtp9): the spec verify chunk's per-GDN-layer
/// {dwconv, t x (scan step + state snapshot), conv-history roll} run, captured once per
/// chunk width and replayed. Replay is bit-identical to the eager chain BY CONSTRUCTION
/// (same kernels, same launch parameters, same baked addresses, same order — only the CPU
/// issue path changes), so `--verify-bit-gate` must stay 24/24 and `--spec-gate` byte
/// identity must hold; those are the gates, not a tolerance.
///
/// **Default OFF, deliberately** (new-flags law): the trunk's own decode-graph receipt on
/// this box is +1.3% for an 84-graph, 2,400-launch reduction (PROFILE-2.md), so launch
/// issue is mostly overlapped here and the expected value is small. This seam exists to
/// MEASURE the one case the trunk receipt does not cover — a serially dependent chain,
/// where issue latency cannot overlap — and it flips only on its own interleaved A/B.
/// Requires the step workspace (address stability) and no section profiler (sync
/// boundaries cannot cross a replay).
static VERIFY_GRAPHS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_verify_graphs(on: bool) {
    VERIFY_GRAPHS.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn verify_graphs_on() -> bool {
    VERIFY_GRAPHS.load(std::sync::atomic::Ordering::Relaxed)
}

/// v2 grouped sel matvec (perf lane item 3: PROFILE-1 residual §3 — the v1 kernel sits
/// at ~225-275 GB/s, scalar byte loads). Default ON with the PROFILE-2.md receipts; v1
/// stays the fallback for guarded geometry and the A/B twin (`--ab-seam selv2`).
/// NOTE: flipping this invalidates nothing structurally, but captured decode graphs
/// bake the kernel choice — the A/B harness allocates a fresh state per arm.
static SEL_V2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_sel_v2(on: bool) {
    SEL_V2.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn sel_v2_on() -> bool {
    SEL_V2.load(std::sync::atomic::Ordering::Relaxed)
}

/// v3 grouped sel matvec (perf round 3: PROFILE-2 residual — v2 sits at ~340-420 GB/s;
/// at the artifact's down geometry a v2 thread runs at most ONE strided iteration, so
/// the warp has almost no memory-level parallelism). v3 = 4 rows/warp sharing the
/// activation registers. Default ON with the round-3 receipts
/// (perf/ab-selv3-nvfp4.tsv: interleaved ×5, 17.06 → 16.57 ms mean-of-means, rep-0
/// chains identical); v2 stays the fallback for guarded geometry (out_f % 4 != 0) and
/// the A/B twin (`--ab-seam selv3`).
pub const SEL_V3_DEFAULT: bool = true;
static SEL_V3: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(SEL_V3_DEFAULT);

pub fn set_sel_v3(on: bool) {
    SEL_V3.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn sel_v3_on() -> bool {
    SEL_V3.load(std::sync::atomic::Ordering::Relaxed)
}

// ---- sel matvec SUB-WARP pair groups (`selgroup`, downsel lane mtp14) ------------------
//
// THE DEFECT, and why it is the priced-next lever in this section. v3/gufuse partition the
// pair loop over all 32 lanes (`for p = lane; p < pairs; p += 32`, `pairs = in_f/32`). At
// this artifact's geometry that does not fill a warp:
//
// | launch  | in_f            | pairs | lane occupancy                            |
// |---------|-----------------|-------|-------------------------------------------|
// | down    | expert ff 640   |    20 | 20/32 = **62.5%** (lanes 20-31 idle, ONE iteration each) |
// | gate+up | hidden 2560     |    80 | 80/96 = **83.3%** (3 warp iterations for 2.5 iterations of work) |
//
// KNEE:q4e-sel-slots-not-bytes measured that this section is per-SLOT-WORK bound, not
// weight-traffic bound (10 -> 60 slots costs 4.13x at fixed bytes; a 6x distinct-byte cut
// buys 1.101x, inside the instrument's own 8.6-11.7% spread). Idle lanes are exactly
// wasted per-slot work, so occupancy is where the section's time is.
//
// THE SHAPE. `qmatvec_nvfp4_modelopt_sel_g_f32` / `..._gu_silu_g_f32` make the pair loop a
// SUB-WARP of `g` lanes: the warp carries `32/g` groups, group `gi` owns `rows` consecutive
// output rows, and the reduce is log2(g) shfl steps inside the group. Rows per warp is
// `(32/g) * rows`, which is what the grid is tiled by. `(g=32, rows=4)` is the shipped v3 /
// gufuse program EXACTLY — byte-compared in `gate_nvfp4_sel_matvec`.
//
// **DEFAULT OFF at introduction, by design (new-flags law).** The ceiling is priced
// (research/qwen4exp-bringup-20260829/spec/downsel/DOWNSEL.md: recovering both kernels'
// idle lanes is worth ~5-7% of the K=5 round, 136.2 -> ~144-146 tok/s) and the exactness
// arms are green on the rig, but this lane had NO timing hardware — the rig is
// exactness-only (LAW:rig-gpu-exactness-only) and no cloud box was approved. A default
// flip needs the interleaved A/B rows the three OWED (scripted, none ran) cells in
// `spec/downsel/` produce; until those exist, ON would be an unmeasured default.
//
// Arm: `MEMRA_Q4E_SEAMS=selgroup` (both families AUTO). Per-family shapes for the A/B
// ladder: `selgroup=dn:4:1+gu:16:2`, `selgroup=dn:8:1+gu:off`, ... Roll back: `selgroup=0` (omitting the name arms AUTO since the 2026-09-02 default flip).
//
// AUTO derives the shape from the geometry rather than pinning a number, because the two
// families have different `pairs` and a single global shape would starve one of them:
// `g` = the largest power of two dividing `pairs`, and `rows` = 4 ALWAYS — the ladder
// inverted the first design here (rows_per_warp≈4 was BACKWARDS): rows-per-LANE is what
// pays, because one pair's activation float4 loads amortize across 4 independent rows'
// code loads, and arms that bought 100% occupancy by spending rows measured WORSE than
// the 62.5% kernel (gu (16,2): −12-14%; down (8,1): −27%; DOWNSEL.md §3). At the serving
// geometry AUTO resolves to **gu (g=16, rows=4)** and **down (g=4, rows=4)** — more rows
// per warp than the shipped kernels, fewer warps in the grid. That grid shrink is the one
// thing the box A/B has to check at t=1, where the down launch already runs only
// out_f/4 * selected warps.
// SCOPE (revuto, PR #27): the `dn` seam reaches ONLY `launch_nvfp4_sel_matvec`. The TP2
// seg-C tail has a third launcher, `launch_nvfp4_sel_matvec_pack` (below, ~L19840), which
// hardcodes `qmatvec_nvfp4_modelopt_sel_f32_v3c` and never reads the seam — while that
// same seg C's gate+up goes through the seam-aware `launch_nvfp4_sel_gu_silu`. Deliberate
// for now: TP2 is not the shipped route on this family (depth regression, receipts in
// ROUND-BUDGET-COMPOSITION.md) and the pack kernel's shape differs; if the EP2 lane
// revives a two-card route through seg C, extend the seam there THEN, with its own gate
// arm, rather than silently inheriting a shape never measured on the pack kernel.
// DEFAULT FLIPPED TO AUTO 2026-09-02 on box receipts (research/qwen4exp-bringup-20260829/spec/
// downsel/box/): cell B (K=5 spec A/B, serving caches q8_0/q5_1 + idxq q8, 5x64 interleaved,
// arm order flipped per hold, spec-vs-plain byte identity on every arm) auto vs off =
// 90.07/87.38, 90.60/87.14, 90.08/87.47 tok/s (+3.1/+4.0/+3.0%); cell C t=1 decode 32k
// 0.9999x/1.0001x, cell D 262k rung 1.0003x (5 reps each) — no depth regression. The
// pre-registered bar "gain > both arms' spread" was MISSED BY A HAIR on each hold (gain
// 2.9-3.8% vs spreads 2.3-4.0%) while the sign never flipped across six holds; the owner
// took the flip on that record (2026-09-02, PR #56). Rollback: `MEMRA_Q4E_SEAMS=selgroup=0`.
const SEL_GROUP_OFF: u32 = 0;
const SEL_GROUP_AUTO: u32 = 1;
/// Down-projection family (`launch_nvfp4_sel_matvec`).
static SEL_GROUP_DN: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(SEL_GROUP_AUTO);
/// Fused gate+up+silu family (`launch_nvfp4_sel_gu_silu`).
static SEL_GROUP_GU: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(SEL_GROUP_AUTO);

fn sel_group_dn() -> u32 {
    SEL_GROUP_DN.load(std::sync::atomic::Ordering::Relaxed)
}

fn sel_group_gu() -> u32 {
    SEL_GROUP_GU.load(std::sync::atomic::Ordering::Relaxed)
}

/// The seam's current spec, in the grammar `set_sel_group` accepts — for exact
/// save/restore around an A/B that flips it (`seam_state` cannot carry it: this seam is not
/// boolean, like `idxq`).
pub fn sel_group_spec() -> String {
    let one = |c: u32| -> String {
        match c {
            SEL_GROUP_OFF => "off".to_string(),
            SEL_GROUP_AUTO => "auto".to_string(),
            v => format!("{}:{}", (v >> 8) & 0xff, v & 0xff),
        }
    };
    format!("dn:{}+gu:{}", one(sel_group_dn()), one(sel_group_gu()))
}

/// Parse and apply the `selgroup` seam value. Grammar (no commas — `MEMRA_Q4E_SEAMS`
/// splits on them):
///
/// - `0` / `off` — both families OFF (the shipped v3 / gufuse kernels).
/// - `` (bare) / `auto` / `1` — both families AUTO.
/// - `dn:<spec>` / `gu:<spec>` joined by `+`, where `<spec>` is `off`, `auto`, or
///   `<g>:<rows>` with `g` a power of two in [1,32] and `rows` in {1,2,4}.
///
/// Returns false (applying nothing) on a malformed spec, so a typo in a cell script fails
/// the seam-name check instead of silently measuring the default arm.
pub fn set_sel_group(spec: &str) -> bool {
    let parse_one = |s: &str| -> Option<u32> {
        match s {
            "off" | "0" => Some(SEL_GROUP_OFF),
            "auto" | "1" | "" => Some(SEL_GROUP_AUTO),
            other => {
                let (g, rows) = other.split_once(':')?;
                let g: u32 = g.parse().ok()?;
                let rows: u32 = rows.parse().ok()?;
                if !matches!(g, 1 | 2 | 4 | 8 | 16 | 32) || !matches!(rows, 1 | 2 | 4) {
                    return None;
                }
                Some((g << 8) | rows)
            }
        }
    };
    if let Some(both) = parse_one(spec) {
        SEL_GROUP_DN.store(both, std::sync::atomic::Ordering::Relaxed);
        SEL_GROUP_GU.store(both, std::sync::atomic::Ordering::Relaxed);
        return true;
    }
    let mut dn = None;
    let mut gu = None;
    for part in spec.split('+').filter(|p| !p.is_empty()) {
        let Some((family, rest)) = part.split_once(':') else {
            return false;
        };
        let Some(code) = parse_one(rest) else {
            return false;
        };
        match family {
            "dn" | "down" => dn = Some(code),
            "gu" | "gateup" => gu = Some(code),
            _ => return false,
        }
    }
    if dn.is_none() && gu.is_none() {
        return false;
    }
    if let Some(c) = dn {
        SEL_GROUP_DN.store(c, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(c) = gu {
        SEL_GROUP_GU.store(c, std::sync::atomic::Ordering::Relaxed);
    }
    true
}

/// Resolve a family's seam code to a concrete `(g, rows)` for THIS launch's geometry, or
/// `None` to take the shipped kernel. Every geometry that the sub-warp form cannot tile
/// exactly falls back rather than clamping: groups inside one warp have different `o0`, so
/// a ragged tile would put lanes with live and dead rows in the same `__shfl_down_sync`.
fn sel_group_resolve(code: u32, in_f: usize, out_f: usize) -> Option<(usize, usize)> {
    if code == SEL_GROUP_OFF || in_f % 32 != 0 {
        return None;
    }
    let pairs = in_f / 32;
    if code != SEL_GROUP_AUTO {
        let (g, rows) = (((code >> 8) & 0xff) as usize, (code & 0xff) as usize);
        if !matches!(g, 1 | 2 | 4 | 8 | 16 | 32) || !matches!(rows, 1 | 2 | 4) {
            return None;
        }
        if out_f % ((32 / g) * rows) != 0 {
            return None;
        }
        return Some((g, rows));
    }
    // AUTO. Largest power-of-two lane group that divides `pairs` exactly (100% lane
    // occupancy); the chain is monotone (2^k | pairs implies 2^(k-1) | pairs), so the first
    // miss ends it.
    let mut g = 1usize;
    for cand in [2usize, 4, 8, 16, 32] {
        if pairs % cand != 0 {
            break;
        }
        g = cand;
    }
    // `rows` (rows per LANE) is held at 4, and that is the measured shape rule rather than
    // an arbitrary pick — an earlier AUTO derived `rows` from `g` so that `rows_per_warp`
    // stayed at the shipped 4, and the ladder says that rule is BACKWARDS. Rows per LANE is
    // what pays, not lane occupancy alone: v3's body exists to share one pair's 8 activation
    // float4 loads across 4 rows and keep 4 independent uint4 code loads in flight, and an
    // arm that reaches 100% lane occupancy by SPENDING rows-per-lane loses that and measures
    // WORSE than the shipped kernel (gate+up g=16 rows=2 -> 100% lanes but ~12% slower;
    // down g=8 rows=1 -> flat). Filling the lanes is only worth doing at rows=4.
    // Rows per warp therefore GROWS to (32/g)*4, which costs warp count — the thing the
    // box cell has to confirm, since these rows were taken on the rig for DIRECTION only
    // (research/qwen4exp-bringup-20260829/spec/downsel/DOWNSEL.md §4).
    let mut rows = 4usize;
    while rows > 1 && out_f % ((32 / g) * rows) != 0 {
        rows /= 2;
    }
    let rows_per_warp = (32 / g) * rows;
    if out_f % rows_per_warp != 0 {
        return None;
    }
    Some((g, rows))
}

/// Read/write-gate micro bundle (perf lane, after items 1-3 the residue is EXECUTION):
/// batched per-stream gate norms (384 one-block launches → 96 stream-batched), the
/// two-stage inject (the single-stage kernel ran 4 blocks on a 188-SM card), slab gate
/// writes (kills 384 add_scaled_rows + 384 inject-row d2d copies per token), and bf16
/// residency for the shared-expert mats (~2.5 GB/token of f32 reads). Default ON with
/// the PROFILE-2.md receipts; OFF is the exact item-3-era composition (`--ab-seam
/// hcmicro`).
static HC_MICRO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_hc_micro(on: bool) {
    HC_MICRO.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn hc_micro_on() -> bool {
    HC_MICRO.load(std::sync::atomic::Ordering::Relaxed)
}

/// GDN decode-step scan twin (perf round 3: PROFILE-2 residual — `gdn_scan_naive_f32`
/// at t=1 runs `nv` blocks (48) with the whole state row per thread in registers,
/// latency-bound). The twin launches grid (nv, hv) with one state ELEMENT per thread;
/// same per-element math, block reduction trees instead of sequential row sums — the
/// accumulation class, gated by `gate_gdn_step_kernels` + the real gates. Default ON
/// with the round-3 receipts (perf/ab-gdnstep-nvfp4.tsv: interleaved ×5, 16.59 → 15.60
/// ms mean-of-means, rep-0 chains identical); the naive kernel stays the prefill
/// executor, the tiny-geometry fallback (hk % 32 != 0), and the A/B twin
/// (`--ab-seam gdnstep`).
pub const GDN_STEP_DEFAULT: bool = true;
static GDN_STEP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(GDN_STEP_DEFAULT);

pub fn set_gdn_step(on: bool) {
    GDN_STEP.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn gdn_step_on() -> bool {
    GDN_STEP.load(std::sync::atomic::Ordering::Relaxed)
}

/// GDN norm+gate fusion (perf round 3): `rms_sigmul_f32` folds the mixer's rms_norm +
/// sigmoid + mul chain into one launch — rms_norm_f32-verbatim reduction, sigmoid_f32
/// gate, no contraction seam, so BIT-IDENTICAL to the chain (asserted exactly by
/// `gate_gdn_step_kernels`). Sigmoid gate arm only; Silu keeps the chain. Default ON
/// with the round-3 receipts (perf/ab-gdnfuse-nvfp4.tsv: interleaved ×5, 16.65 → 16.52
/// ms mean-of-means, rep-0 chains identical; small but real, and the kernel is
/// bit-identical to the chain it replaces); `--ab-seam gdnfuse`.
pub const GDN_FUSE_DEFAULT: bool = true;
static GDN_FUSE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(GDN_FUSE_DEFAULT);

pub fn set_gdn_fuse(on: bool) {
    GDN_FUSE.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn gdn_fuse_on() -> bool {
    GDN_FUSE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Projection stack (perf round 4): same-activation trunk projections that ran as
/// separate `qmatvec_bf16w_f32` launches — GDN qkv/z/beta/alpha (4), QSA wq/wk/wv (3),
/// shared-expert gate/up (2) — collapse into ONE `qmatvec_bf16w_multi4_f32` launch over
/// a load-time row-stacked bf16 twin, each output row routed to its original slot buffer
/// by row range. Per-row math is the bf16w kernel VERBATIM, so outputs are BIT-IDENTICAL
/// to the per-mat launches; decode only (t == 1), requires the bf16 trunk seam. Default
/// ON with the round-4 receipts (perf20/ab-projstack-nvfp4.tsv: interleaved x5,
/// 15.72 -> 15.25 ms mean-of-means, rep-0 chains IDENTICAL; tiny gate ON/OFF receipts
/// byte-identical; real gate r4-on: argmax 10/10, greedy forks unchanged, tp2-gate
/// 24/24); the per-mat row-offset-view launches stay the OFF arm; `--ab-seam projstack`.
pub const PROJ_STACK_DEFAULT: bool = true;
static PROJ_STACK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(PROJ_STACK_DEFAULT);

pub fn set_proj_stack(on: bool) {
    PROJ_STACK.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn proj_stack_on() -> bool {
    PROJ_STACK.load(std::sync::atomic::Ordering::Relaxed)
}

/// Hyper-gate diet (perf round 4): the read gate's 7-launch serial chain (norm, batched
/// down GEMV, lowrank reduce, batched up GEMV, mix epilogue, inject partials + reduce)
/// re-fuses into THREE launches at t == 1 — stage 1 (per-stream RMS recompute + normed
/// smem row + down/inject rows), stage 2 (silu mean + inject sigmoid), stage 3 (up dots +
/// mix epilogue from the stage-1 inv scalars). ACCUMULATION CLASS (new reduce widths);
/// gated by `gate_hc_diet_kernels` (real geometry vs the classic fused chain) + the real
/// gates. Requires the bf16 trunk twins + hcmicro inject posture (the Slab inject form);
/// geometry guards hidden % 8 == 0 && rank % 8 == 0 (tiny plans fall back). Default ON
/// with the round-4 receipts (perf20/ab-hcdiet-nvfp4.tsv: interleaved x5, 15.69 ->
/// 15.32 ms mean-of-means, rep-0 chains IDENTICAL; oracle arm 0e worst rel 2.369e-6;
/// real gate r4-on: argmax 10/10, greedy forks unchanged, tp2-gate 24/24); the fused
/// chain stays the OFF arm; `--ab-seam hcdiet`.
pub const HC_DIET_DEFAULT: bool = true;
static HC_DIET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(HC_DIET_DEFAULT);

pub fn set_hc_diet(on: bool) {
    HC_DIET.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn hc_diet_on() -> bool {
    HC_DIET.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fused gate+up+silu sel matvec (perf round 4, post the W4A4 owner retirement — the
/// activation-precision-NEUTRAL half of the sel lever): the MoE tail's gate launch +
/// up launch + silu launch collapse into ONE `qmatvec_nvfp4_modelopt_sel_gu_silu_f32`
/// (each warp runs 4 gate + 4 up rows off shared f32 activation registers; per-row
/// arithmetic v3-VERBATIM, epilogue silu_mul_f32-VERBATIM => BIT-IDENTICAL to the
/// chain, asserted by the sel oracle's gufuse mode). Cuts the sel serial chain 5 -> 3
/// launches and doubles outstanding code loads per warp (the slice is latency-bound at
/// ~27% of card bandwidth — PROFILE-4 re-profile). Geometry in_f % 32 == 0 &&
/// ff % 4 == 0, else the v3 chain. Default ON with the round-4 receipts
/// (perf24/ab-gufuse-nvfp4{,-tp2}.tsv: interleaved x5, single 14.75 -> 14.58, TP2
/// route 13.43 -> 13.10, rep-0 chains IDENTICAL both configs; oracle gufuse mode
/// asserts byte identity incl. the count-gated pack twin); `--ab-seam gufuse`.
pub const SEL_GUFUSE_DEFAULT: bool = true;
static SEL_GUFUSE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(SEL_GUFUSE_DEFAULT);

pub fn set_sel_gufuse(on: bool) {
    SEL_GUFUSE.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn sel_gufuse_on() -> bool {
    SEL_GUFUSE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Verify multi-token WEIGHT-SHARED kernels (mtp-spec): trunk dense mats run
/// `qmatvec_bf16w_mt_f32` (one block per output row, W read ONCE for every verify
/// column — the qwen38 t-parallel pattern) and the MoE verify columns merge into ONE
/// grouped launch per projection via the gufuse tok_map. Every per-(row,token) fma
/// chain is the t == 1 program VERBATIM => rows stay BIT-IDENTICAL to per-token
/// launches (asserted by the bf16-matvec oracle's mt mode and the verify-bit gate);
/// only weight-read counts and launch counts drop. Engages ONLY at 2 <= t <= 12 exact
/// chunks (plain decode and prefill untouched). Default ON with the mtp-spec lane's
/// receipts (spec/MTP-SPEC.md: verify-bit-gate bit-identity + interleaved spec A/B);
/// OFF twin = the per-token grid path, `--ab-seam vmt`.
pub const VERIFY_MT_DEFAULT: bool = true;
static VERIFY_MT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(VERIFY_MT_DEFAULT);

pub fn set_verify_mt(on: bool) {
    VERIFY_MT.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn verify_mt_on() -> bool {
    VERIFY_MT.load(std::sync::atomic::Ordering::Relaxed)
}

/// FUSED verify program (`vfuse`, mtp12 cost lane): route a `1 < t <= k_cap` verify chunk
/// through the FUSED (prefill-style) program instead of the EXACT per-row programs.
///
/// **This is a COST INSTRUMENT, not a serving arm, and it is default OFF forever unless a
/// receipt moves it** (new-flags law). What it changes and what it cannot:
///
/// - Changes (the only sections where exact and fused differ): trunk dense mats
///   (`qmatvec_bf16w_mt` W-once → cuBLASLt m=t), the hyper read gate (hc-diet MT 3-launch
///   → the t-generic fused chain), the GDN scan (per-column `gdn_scan_step_at` + snapshots
///   → chunk scan), the QSA indexer projection and the PLE projections (t × m=1 → 1 × m=t).
/// - Cannot change, BY CONSTRUCTION: the MoE routed union (already ONE grouped gufuse
///   launch over every column on the exact arm — this seam FORCES `grouped` so the fused
///   chunk does not fall into the per-expert prefill executor, which costs minutes/chunk),
///   `sdpa_naive_mask` (same kernel both arms), and the ~12 ms/round of per-layer HOST
///   TWIN bubbles (48 MoE router dtoh + 12 QSA indexer masks are PER CHUNK, not per
///   column, so a fused chunk pays them identically).
///
/// **No rewind exists on this arm.** The exact program stashes per-column GDN recurrent
/// state and PLE segment state so `verify_rewind` can drop rejected columns replay-free;
/// the fused chunk scan materializes only the final state, so `verify_rewind` refuses
/// loudly (`vfuse_chunk`) rather than silently rewinding to a wrong state. That is why the
/// seam is a timing probe on a throwaway state and NOT wired into `spec_generate`.
pub const VERIFY_FUSED_DEFAULT: bool = false;
static VERIFY_FUSED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(VERIFY_FUSED_DEFAULT);

pub fn set_verify_fused(on: bool) {
    VERIFY_FUSED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn verify_fused_on() -> bool {
    VERIFY_FUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Router bf16 residency (perf round 4): the MoE router GEMV was the last dense trunk
/// mat still on f32 cuBLASLt (the TP2 nsys counts it among the ~70 f32 gemvx
/// calls/token). Same guards and arithmetic class as the trunk seam (exact bf16
/// widening, accumulation-class reduction change — routing near-ties are gated by the
/// real gate's argmax/greedy battery). Default ON with the round-4 receipts
/// (perf24/ab-routerb16-nvfp4{,-tp2}.tsv: interleaved x5, single 14.75 -> 14.68, TP2
/// route 13.47 -> 13.36, rep-0 chains IDENTICAL; decode-row seam-gate 24/24 argmax,
/// worst KL 0.00116 — the trunk accumulation class); `--ab-seam routerb16`.
pub const ROUTER_B16_DEFAULT: bool = true;
static ROUTER_B16: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(ROUTER_B16_DEFAULT);

pub fn set_router_bf16(on: bool) {
    ROUTER_B16.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn router_bf16_on() -> bool {
    ROUTER_B16.load(std::sync::atomic::Ordering::Relaxed)
}

/// Gate/battery instrumentation: apply `MEMRA_Q4E_SEAMS` ("name" or "name=0", comma
/// separated) to the seam setters, so the tiny + real gates can prove a NEW seam green
/// while its shipped default is still OFF (flags law: correctness receipts precede the
/// default flip). Names match the `--ab-seam` vocabulary.
/// The masked SDPA kernel's smem score bound in KV tokens (48 KB of f32 scores). Past
/// this the dense-mask path is impossible; the block-list kernel takes over.
const SDPA_MASK_TKV_BOUND: usize = 12288;

/// Device QSA indexer block scorer (long-context lane). Default ON: scores are
/// BIT-IDENTICAL to the host twin (same dim order, relu-sum and division), the host twin
/// stays the reference/TP2 path, and the host cost it replaces is O(context) per token
/// per layer — 52% of the decode token at a 32k fill and quadratic across a long
/// prefill (receipts in research/qwen4exp-bringup-20260829/yarn/). Rollback:
/// `MEMRA_Q4E_SEAMS=idxdev=0`.
static IDX_DEV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
fn idx_dev_on() -> bool {
    IDX_DEV.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_idx_dev(on: bool) {
    IDX_DEV.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Device QSA indexer top-k SELECTION (262k perf lane, `qsa_index_topk_u32`). The
/// `idxdev` seam above moved the block SCORING to the GPU and then dtoh'd the whole score
/// slab so the HOST could run `top_blocks_ascending` per row. At the product window that
/// host half is the wall: at a 131,072 fill `qsa.idx_host` measured **51,235 ms — 83% of a
/// prefill chunk** while every GPU section stayed flat within 4%, and it is what prices the
/// whole 262k window down from ~32 to ~15-18 tok/s
/// (research/qwen4exp-bringup-20260829/round2-box-receipts/LADDER.md §4c). This seam runs
/// the selection on device and reads back `rows x budget` u32 instead of `rows x blocks`
/// f32 (4 MB instead of up to 128 MB per sub-batch).
///
/// Selection is EXACT by construction, not by tolerance: the kernel's u64 key orders
/// ascending exactly as the host `sel_cmp` (score desc under `total_cmp`, block index asc)
/// over the whole f32 domain, keys are distinct, and the emitted order is ascending block
/// index. Gated by `gate_qsa_index_topk` (real geometry + tie batteries incl. the
/// structural all-zero-score class) and by the live cross-surface audit
/// `MEMRA_Q4E_IDXSEL_AUDIT=1`, which recomputes the host twin from the SAME slab and
/// hard-compares ids AND order.
///
/// **Default ON (2026-09-01), FLIPPED on receipts** (new-flags law: a default is a decision
/// with its reasons and receipts stated, and it flips only once both arms are measured).
/// Introduced default OFF the day before; the flip carries:
///
/// - **Interleaved same-fill A/B at 131,072** (`--ladder-ab-seam idxsel`, both arms on ONE
///   prefill, exclusive measurement lock, sole tenant): off 56.64 ms / 17.66 tok/s vs on
///   32.25 ms / 31.01 tok/s = **1.7562x**, 7 reps per arm (escalated from 5), 224 warm
///   samples per arm, within-arm spreads 2.40% / 2.12% — the verdict is ~18x the pooled
///   spread. Reproduces the independent two-process pair (1.76x) on both arms.
/// - **The target window**: 262,144 tokens goes **15.21 -> 23.44 tok/s (1.54x)** with the
///   prefill wall **4,779.1 -> 1,439.2 s (3.32x)**, spread 2.56% (escalated x5) -> 0.30%.
/// - **The cliff is gone**: 100,000 -> 131,072 was 1.9x slower for 1.31x depth; it is now
///   -7.6%, and prefill per chunk is flat across a continuous 262k fill (82.8 -> 96.4 s per
///   16k, where the OFF arm stepped 105 -> 475).
/// - **Exactness**: tie-battery oracle EXACT on ids AND order at real budget 512 up to
///   65,536 blocks, on BOTH card classes; live at-depth audit **1,549,452 rows / 0
///   mismatches / deepest_blocks 32,793**; decode-row-volume audit **120,000 decode-row
///   selections / 0 mismatches**; greedy chain byte-identical across the seam at a
///   100,000-token fill; all four rule gates green and identical to the prior battery.
/// - **Variance improves too**: both deep rungs auto-escalated to x5 on the OFF arm (2.74% /
///   1.62%) and sit at 0.36% / 0.02% with the seam on — the 48-thread host top-k pool was
///   also the jitter source.
///
/// Rollback: `MEMRA_Q4E_SEAMS=idxsel=0` (the pure host top-k over the dtoh'd slab). Unlike
/// the devtwin pair, this seam has NO pairing requirement — it wins alone on every measured
/// surface and it is measured on top of the shipped `routerdev` + `idxcache` + `kvq` stack.
/// Receipts: research/qwen4exp-bringup-20260829/perf/PROFILE-11.md.
pub const IDX_SEL_DEFAULT: bool = true;
static IDX_SEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(IDX_SEL_DEFAULT);
fn idx_sel_on() -> bool {
    IDX_SEL.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_idx_sel(on: bool) {
    IDX_SEL.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// INCREMENTAL PLE n-gram id cache (262k perf lane, `plecache`). `ple_block` calls
/// `host_ngram_ids`, a `ngram_ids` twin over the FULL token history, and then slices the last
/// `t` rows — so a decode step at a 150,000-token fill rebuilds 150,000 rows of hashes to
/// consume ONE. Measured on the deep decode profile with `idxsel` armed:
/// `ple.host_ngram_gather` is **7.3 ms, 19.5% of the token**, second only to `qsa.sdpa` and
/// the largest remaining HOST section (PROFILE-11 §5).
///
/// This is the correction the deep profile forced on the owner's stated prefetch lever, and
/// it is worth stating rather than quietly fixing: the assumed mechanism was "the gather from
/// the 102 GB host table is synchronous, so overlap it with compute". The gather itself is
/// `t * 16` random rows — 16 reads of 160 f32 at decode, microseconds. The 7.3 ms is the
/// O(context) ID RECOMPUTE in front of it. Async-prefetching the table would have bought
/// ~nothing; caching the ids removes essentially all of it. Same class as the yarn lane's
/// O(context) host selection, in a different section.
///
/// Exact by construction (see `host_ngram_ids_cached`): `ids[token]` is a pure function of
/// `token_ids[..=token]`, so the cache appends and never recomputes. Divergence and rewind
/// are handled by a real longest-common-PREFIX compare, not a length compare.
///
/// **Default ON as of 2026-09-01, by design** (new-flags law: the decision and its reasons are
/// written, and the receipts landed before the flip). Introduced default OFF on 2026-08-31 with
/// no perf receipts; flipped after the A/B and the exactness battery below. Rollback is one
/// token: `MEMRA_Q4E_SEAMS=plecache=0`.
///
/// PERFORMANCE — x3 interleaved, both arms sharing one prefill and one exclusive lock hold, lead
/// flipped on odd reps, no escalation owed on any arm (PROFILE-12 §2, §10):
///
/// | depth | OFF | ON | speedup | this section, OFF arm |
/// |---|---|---|---|---|
/// | 131,072 | 33.52 ms / 29.83 tok/s | 25.91 ms / 38.60 tok/s | 1.2938x | 7.8 ms (20.3%) |
/// | 262,144 | 41.38 ms / 24.17 tok/s | 28.30 ms / 35.34 tok/s | **1.4620x** | **13.2 ms (28.9%)** |
///
/// The gain GROWS with depth because the deleted work is O(fill) per token, and at the target
/// window this was **the largest section of the whole token**, ahead of `qsa.sdpa`. With the seam
/// armed it leaves the top twelve entirely while every other section holds to a tenth of a
/// millisecond. It also removes decode JITTER: cv 2.48% -> 0.11% at 131,072, p99 41.70 -> 28.38 ms
/// at 262,144 — a p99-latency result, which is what a deep-context agentic workload feels.
///
/// EXACTNESS — the flip rests on the two arms that can actually falsify it, not on the many that
/// cannot:
/// - **Real-geometry truth pin** (`MEMRA_Q4E_PLECACHE_AUDIT=1`): `rows=32828 mismatched=0
///   deepest_fill=32828`. Cached ids hard-compared against the full `host_ngram_ids` twin at the
///   CHECKPOINT's own multipliers/sizes/offsets, over both growth shapes (2,048-token prefill
///   chunks and one-at-a-time decode appends).
/// - **Behavioural control**: the greedy chain is IDENTICAL across the seam on the same artifact
///   (`-1/0/-1/26` both arms, hidden-goldens argmax 10/10 both arms).
/// - Host oracle vs the full twin: EXACT over 69,635 cumulative-sequence comparisons across 6 case
///   families (decode growth, ragged prefill chunks, eos resets incl. adjacent/leading/trailing,
///   all-eos, repeated rewinds to DIVERGING prefixes, shorter-unrelated-sequence state reuse).
/// - `verify-bit` 24 `mismatched=0 policy=bit-identity`; spec byte-identity 256
///   `policy=byte-identity pass=true` with `first_divergence=-1` on all four prompts.
///
/// **Why those last two carry less weight than they look like they do, stated so the flip is not
/// over-credited:** `verify-bit` and spec byte-identity are INTRA-ARM, and an intra-arm identity
/// gate cannot detect a CONSISTENT error — a uniformly-wrong id set is perfectly self-consistent
/// and passes both with full marks. The truth pin and the greedy control are what close it.
///
/// STILL OWED (PROFILE-12 §9): `--verify-bit-deep 131072` with the seam armed has not passed — it
/// failed three times on this box with ~96 GB free, i.e. on the instrument rather than on the seam.
/// It is intra-arm, so it cannot add exactness assurance the truth pin does not already give; the
/// flip does not wait on it, and it stays owed rather than being quietly dropped.
///
/// COST: one `i64` id vector per state, `fill * 16 * 8` bytes = 33.5 MB of HOST memory at 262,144
/// (the box carries 499 GB), plus the token-history mirror. No device memory, no new kernel.
pub const PLE_CACHE_DEFAULT: bool = true;
static PLE_CACHE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(PLE_CACHE_DEFAULT);
fn ple_cache_on() -> bool {
    PLE_CACHE.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_ple_cache(on: bool) {
    PLE_CACHE.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// Live cross-surface audit for the PLE id cache (`MEMRA_Q4E_PLECACHE_AUDIT=1`): recompute
/// the FULL `host_ngram_ids` twin and hard-compare the chunk's rows against the cached ones.
/// Instrument only — it restores exactly the O(context) work the seam deletes.
fn ple_cache_audit_on() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MEMRA_Q4E_PLECACHE_AUDIT").as_deref() == Ok("1"))
}
static PLE_CACHE_AUDIT_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PLE_CACHE_AUDIT_MISMATCH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PLE_CACHE_AUDIT_MAX_FILL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// (rows audited, id mismatches, deepest history length seen) since process start.
pub fn ple_cache_audit_stats() -> (u64, u64, u64) {
    (
        PLE_CACHE_AUDIT_ROWS.load(std::sync::atomic::Ordering::Relaxed),
        PLE_CACHE_AUDIT_MISMATCH.load(std::sync::atomic::Ordering::Relaxed),
        PLE_CACHE_AUDIT_MAX_FILL.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Live device-vs-host indexer-selection audit (`MEMRA_Q4E_IDXSEL_AUDIT=1`): every device
/// selection ALSO dtohs the score slab and runs `top_blocks_ascending` on the same bytes,
/// hard-comparing the block ids AND their emitted order. Instrument only — it restores the
/// very dtoh this seam deletes, so it is never a perf arm. Counters feed the receipt
/// (`idx_sel_audit_stats`); `rows=0` is the silent-no-op failure the counter exists to
/// catch.
fn idx_sel_audit_on() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MEMRA_Q4E_IDXSEL_AUDIT").as_deref() == Ok("1"))
}
static IDX_SEL_AUDIT_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDX_SEL_AUDIT_MISMATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDX_SEL_AUDIT_MAX_BLOCKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// (rows audited, selection mismatches, deepest block count seen) since process start.
pub fn idx_sel_audit_stats() -> (u64, u64, u64) {
    (
        IDX_SEL_AUDIT_ROWS.load(std::sync::atomic::Ordering::Relaxed),
        IDX_SEL_AUDIT_MISMATCH.load(std::sync::atomic::Ordering::Relaxed),
        IDX_SEL_AUDIT_MAX_BLOCKS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Device MoE router (devtwin lane): `qwen4exp_route_topk_f32` replaces the per-layer
/// router dtoh + `host_route_softmax_topk` + selection h2d — the census's 48 blocking
/// drains per forward and the round-3 doctrine's whole-step-graph blocker. Engages on
/// the GROUPED dispatch paths only (NVFP4 t==1 decode / verify columns / graph-driver
/// slots); the per-expert prefill executor and the TP2 route keep the host twin (they
/// consume host expert ids by construction). Selection set + order are gated EXACTLY
/// against the host twin (gate_route_kernel + MEMRA_Q4E_ROUTER_AUDIT); weights within
/// documented ULP (exp is the one non-bit-pinned op — kernel doc).
///
/// **Default ON (2026-08-31), decided on receipts** (better-wins-by-default): the
/// combined devtwin stack wins every measured surface — spec at ship admission thinkon
/// 1.168x / thinkoff 1.174x / efflow 1.160x / raw 1.194x / long-724 1.116x with
/// BYTE-IDENTICAL 256-token chains, K ladder 1.14-1.18x over K=1..8, plain decode
/// 1.099x with decode graphs ON and 1.112x with them OFF — under all three rule gates
/// green (verify-bit 24/24, spec-gate byte identity, tp2-gate) plus a 250k-row live
/// host-twin audit with ZERO selection mismatches. **Pair with `idxcache`: this seam
/// ALONE with decode graphs ON measured 0.906x** (PROFILE-9 §3/§3a) — the stack is the
/// unit, which is why both defaults flip together. Rollback:
/// `MEMRA_Q4E_SEAMS=routerdev=0`.
pub const ROUTER_DEV_DEFAULT: bool = true;
static ROUTER_DEV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(ROUTER_DEV_DEFAULT);
fn router_dev_on() -> bool {
    ROUTER_DEV.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_router_dev(on: bool) {
    ROUTER_DEV.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// The device router's geometry envelope: register top-k (<= 32 slots) + smem softmax
/// slab (experts f32 <= 48 KB). Real geometry 512/10 sits comfortably inside; a plan
/// outside the envelope keeps the host twin.
fn route_dev_geometry(experts: usize, selected: usize) -> bool {
    // Even expert count: the u64 selection-key slab follows the f32 weight slab in
    // dynamic smem (12 B/expert total) and needs 8-byte alignment.
    selected > 0
        && selected <= 32
        && selected <= experts
        && experts % 2 == 0
        && experts * 12 <= 48 * 1024
}

/// Live device-vs-host router twin audit (`MEMRA_Q4E_ROUTER_AUDIT=1`): every device
/// route ALSO computes the host twin from the same logits and hard-compares — selection
/// ids order-exact (Err on any mismatch), weights within `ROUTE_AUDIT_ULP_BOUND` ULP
/// (worst observed kept for the receipt). The sigrouter-precedent cross-surface
/// contract, run over REAL decode rows by any existing gate invocation. Instrument
/// only: it dtohs per route, so it is never a perf arm.
fn router_audit_on() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MEMRA_Q4E_ROUTER_AUDIT").as_deref() == Ok("1"))
}
/// DIAGNOSTIC seam (`MEMRA_Q4E_ROUTE_SYNC=1`): keep the device route but restore the
/// host arm's per-layer stream sync — the instrument that separates kernel cost from
/// sync-structure cost in the graphs-ON regression. Never a serving arm; no FLAGS row
/// because it is an instrument, and it is read once per process.
fn route_sync_diag() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MEMRA_Q4E_ROUTE_SYNC").as_deref() == Ok("1"))
}

const ROUTE_AUDIT_ULP_BOUND: u32 = 8;
static ROUTE_AUDIT_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ROUTE_AUDIT_MAX_ULP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// (rows audited, worst weight ULP distance) since process start.
pub fn route_audit_stats() -> (u64, u32) {
    (
        ROUTE_AUDIT_ROWS.load(std::sync::atomic::Ordering::Relaxed),
        ROUTE_AUDIT_MAX_ULP.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Device-resident indexer raw-key cache (devtwin stage 3): below the QSA selection
/// horizon ((base_pos + t)/block <= budget — every row structurally full), the
/// idx_proj dtoh exists ONLY to feed the host raw-key cache for a possible future
/// scored row. This seam appends the k-part rows d2d (`copy_rows_col_f32`, exact byte
/// moves) and materializes the host cache LAZILY at the first scored chunk — the same
/// bytes dtoh'd later, so the scored path is bit-identical by construction. Kills the
/// census's 12 idx_proj blocking dtoh per forward (+1 per draft chain step) on every
/// sub-horizon shape.
///
/// **Default ON (2026-08-31), decided on receipts** with `routerdev` as ONE stack (see
/// that seam's note and PROFILE-9): isolated plain-decode row 1.024x, and it is the half
/// that makes the router's sign positive with decode graphs ON. Rollback:
/// `MEMRA_Q4E_SEAMS=idxcache=0`.
pub const IDX_CACHE_DEFAULT: bool = true;
static IDX_CACHE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(IDX_CACHE_DEFAULT);
fn idx_cache_on() -> bool {
    IDX_CACHE.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_idx_cache(on: bool) {
    IDX_CACHE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Quantized QSA KV cache (kvq lane): K = q8_0, V = q5_1 — the owner's asymmetric
/// default (K feeds the score dots + rope, so it keeps symmetric 8-bit; V errors
/// average under the attention weighting, so affine 5-bit suffices). The format is
/// LATCHED PER STATE at `alloc_state`/`mtp_state` time (a byte cache cannot flip
/// mid-run); the f32 arm stays the exactness instrument and the rollback seam
/// (`MEMRA_Q4E_SEAMS=kvq=0`). Storage-only: attention math runs f32 on dequanted
/// values (the block-list kernel's program with in-place dequant, gated bit-identical
/// to the dequant-rows + f32-kernel composition). Default ON per the owner decision,
/// with this lane's receipts attached (flags law): within-config exactness green
/// (spec byte-identity 6/6, verify-bit 24/24 x3, envelope 24/24 @ 3.0e-5), cross-config
/// drift is the near-tie quant class stated in KVQ-CELL.md (worst rows flip between the
/// two eos ids; greedy forks on the valid raw instrument match the f32 class).
///
/// PERF JUSTIFICATION, DEPTH-SCOPED (corrected 2026-08-31; docs/FLAGS.md carried the scoping
/// and this doc comment did not, so the stale claim was still riding here). The flip cited
/// "the quantized cache is FASTER, 13.36-13.39 vs 13.53-13.57 ms/token interleaved". That was
/// measured at a SHALLOW fill and the sign REVERSES with depth: at a 100,000-token fill kvq is
/// **-7.4% decode and -7.3% prefill wall** vs the f32 twin (LADDER.md, KVQ-CELL.md round 2).
/// Never quote "kvq is faster" at depth. The DECISION stands on memory: 11.08 vs 49.0
/// KiB/token, and at the 262,144 target window kvq is memory-REQUIRED (the f32 arm does not
/// allocate that state at all), so there is no alternative to compare against.
/// The -7.4% is a READ-PATTERN artifact, not the cost of quantization -- see `KV_HOIST_DEFAULT`.
/// Receipts: research/qwen4exp-bringup-20260829/kvq/ + box ~/realgate/kvq.
pub const KV_QUANT_DEFAULT: bool = true;
static KV_QUANT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(KV_QUANT_DEFAULT);
fn kv_quant_on() -> bool {
    KV_QUANT.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_kv_quant(on: bool) {
    KV_QUANT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// HOISTED K block scale in the quantized block-list attention (`kvhoist`, memory lane
/// 2026-08-31). Selects `q4e_sdpa_blocklist_q8q5_hoist` over `q4e_sdpa_blocklist_q8q5`;
/// BIT-IDENTICAL by construction (same product, same `acc +=` order, phase 2 and phase 3
/// verbatim), so this is a pure read-pattern seam and the bar is bit-identity, not a band.
///
/// It exists because it is the mechanism behind the kvq perf SIGN FLIP, and the flip turns out
/// to be a layout artifact rather than a tax. `q4e_deq_q8` recomputes the block pointer from the
/// element index, so the score loop reloads the fp16 block scale ONCE PER ELEMENT. Measured
/// statically in the sm_120 SASS (`PROFILE-C0.md` §2), score-phase inner loop per 8 K elements:
///
/// | kernel | instrs | KV-cache loads | fp16 scale loads |
/// |---|---|---|---|
/// | `sdpa_blocklist_f32` | 37 | 8 | -- |
/// | `q4e_sdpa_blocklist_q8q5` | **120** | 8 | **8** |
/// | `q4e_sdpa_blocklist_q8q5_hoist` | **52** | 8 | **0** (1 per 32-elem block) |
///
/// Phase 1 is thread-per-position (lanes sit on 32 different tokens, `k_tok_bytes` apart), so
/// every load instruction replays 32 ways into 32 distinct sectors. The quantized cache
/// therefore issued 2x the f32 twin's KV transactions while reading 3.76x fewer bytes: the byte
/// saving cannot land, and the extra instruction stream is a straight loss. That is the -7.4%
/// at a 100,000-token fill, and it is why the +1.3% shallow flip receipt had the opposite sign
/// (a shallow fill reads almost no rows, so phase 1 barely runs).
///
/// Default OFF at introduction, by design (new-flags law): the correctness receipts land with
/// the seam and the default flip is a separate change carrying the interleaved A/B. Arm with
/// `MEMRA_Q4E_SEAMS=kvhoist`; rollback `kvhoist=0`. Mid-run flippable (no layout latch).
pub const KV_HOIST_DEFAULT: bool = false;
static KV_HOIST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(KV_HOIST_DEFAULT);
fn kv_hoist_on() -> bool {
    KV_HOIST.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_kv_hoist(on: bool) {
    KV_HOIST.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// For receipt headers, same reason as `kv_quant_is_on`.
pub fn kv_hoist_is_on() -> bool {
    kv_hoist_on()
}

/// DIM-MAJOR pooled-key device plane (`poolT`, memory lane 2026-08-31). Selects
/// `qsa_index_score_f32_t` over `qsa_index_score_f32` and mirrors the pooled cache transposed;
/// BIT-IDENTICAL by construction (identical loop order and identical explicit
/// `__fmul_rn`/`__fadd_rn`/`__fdiv_rn` -- only the address of `pooled` changes).
///
/// It targets the SECOND depth-scaling term in the deep decode profile. With `idxsel` armed
/// (`ladder-r2prof-step-idxsel.tsv`), `qsa.idx_host` is 2.5 ms at 100,000 / 3.0 at 131,072 /
/// 3.2 at 150,000 -- linear in context, extrapolating to ~5.7 ms at 262,144, behind only
/// `ple.host_ngram_gather` among the terms that grow. The score kernel is thread-per-block over
/// the pooled plane, so lane L reads `pooled[(block0+L)*head_dim + d]`: lanes are head_dim*4 =
/// 512 B apart, one warp's `k[d]` touches 32 DISTINCT sectors and moves 1024 B to use 128 B.
/// Dim-major makes the same 32 lanes read 32 consecutive floats: 4 sectors, zero waste, 8x less
/// sector traffic on the one array whose size IS the context.
///
/// Default OFF at introduction, by design (new-flags law). Arm `MEMRA_Q4E_SEAMS=poolT`, roll
/// back `poolT=0`. **Mid-run flippable with NO rebuild**: both layouts are maintained on every
/// append (see the append site for why), so `**mirrored` is the single truth for both and a flip
/// can neither read a stale plane nor leave one behind. The seam selects only the kernel.
pub const POOL_T_DEFAULT: bool = false;
static POOL_T: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(POOL_T_DEFAULT);
fn pool_t_on() -> bool {
    POOL_T.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_pool_t(on: bool) {
    POOL_T.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// For receipt headers.
pub fn pool_t_is_on() -> bool {
    pool_t_on()
}

/// The live KV cache format, for RECEIPT HEADERS. A receipt that does not record which
/// cache arm it ran cannot be read: the round-2 ladder measured the f32 arm for a full
/// rung while its commit message said "kvq ship defaults", and nothing in the receipt
/// could have contradicted that. Reported, not inferred.
pub fn kv_quant_is_on() -> bool {
    kv_quant_on()
}

/// Indexer raw-key cache precision (idxq lane). The 128-dim raw keys are cached
/// pre-norm/pre-rope and consumed ONLY through fp32 mean-pooling into pooled keys —
/// this seam quantizes the CACHE and dequants at read; the pooling math is identical.
/// Precision is picked by measurement (selection-identity flip rate on real prompts at
/// depth): q8 is the target, bf16 the fallback if q8 flips selections, f32 the
/// rollback/reference. Latched per state at alloc. Default Q8 per the measured
/// receipt: the q8-vs-f32 seam gate came back BIT-ZERO on the real checkpoint
/// (selection provably unmoved, 24/24 argmax, worst_abs 0.000e0 —
/// kvq/seam-gate-idxq-idxq1.tsv), so the cheaper cache wins by measurement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IdxQMode {
    F32,
    Q8,
    Bf16,
}
static IDXQ_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);
fn idxq_mode() -> IdxQMode {
    match IDXQ_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => IdxQMode::Q8,
        2 => IdxQMode::Bf16,
        _ => IdxQMode::F32,
    }
}
pub fn set_idxq(mode: &str) {
    let v = match mode {
        "q8" | "1" => 1,
        "bf16" => 2,
        _ => 0,
    };
    IDXQ_MODE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The live indexer raw-key cache precision, for RECEIPT HEADERS (see `kv_quant_is_on`).
pub fn idxq_mode_name() -> &'static str {
    match idxq_mode() {
        IdxQMode::F32 => "f32",
        IdxQMode::Q8 => "q8",
        IdxQMode::Bf16 => "bf16",
    }
}

/// Selection-identity audit (`MEMRA_Q4E_IDXQ_AUDIT=1`): with a quantized raw-key cache,
/// ALSO maintain an f32 twin cache (forcing the idx_proj dtoh the idxcache seam
/// removed — instrument, never a perf arm) and compute every scored row's selection
/// twice; count rows whose selected block set differs. The 1-ULP FMA lesson says
/// near-tie blocks CAN flip — this measures the rate on real prompts at depth.
fn idxq_audit_on() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MEMRA_Q4E_IDXQ_AUDIT").as_deref() == Ok("1"))
}
static IDXQ_AUDIT_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDXQ_AUDIT_FLIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDXQ_AUDIT_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// (scored rows audited, rows with a flipped selection set, total symmetric-difference
/// blocks) since process start.
pub fn idxq_audit_stats() -> (u64, u64, u64) {
    (
        IDXQ_AUDIT_ROWS.load(std::sync::atomic::Ordering::Relaxed),
        IDXQ_AUDIT_FLIPPED.load(std::sync::atomic::Ordering::Relaxed),
        IDXQ_AUDIT_BLOCKS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Long-context QSA attention form (yarn lane). Auto = block-list kernel ONLY past the
/// masked kernel's smem bound (every historical receipt is byte-stable below it);
/// Force = block-list everywhere (the gate arms' A/B); Off = refuse long contexts (the
/// historical error). FLAGS.md row `q4e-longatt`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LongAttMode {
    Auto,
    Force,
    Off,
}
static LONGATT_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
fn longatt_mode() -> LongAttMode {
    match LONGATT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => LongAttMode::Force,
        2 => LongAttMode::Off,
        _ => LongAttMode::Auto,
    }
}
pub fn set_longatt(mode: &str) {
    let v = match mode {
        "force" | "1" => 1,
        "off" | "0" => 2,
        _ => 0,
    };
    LONGATT_MODE.store(v, std::sync::atomic::Ordering::Relaxed);
}

// ------------------------------------------------------------ TP2 MoE expert placement
//
// Owner directive 2026-08-31 (LAW:coactivation-expert-placement): expert placement is
// MEASURED, never even-split — bundles by co-activation, the always-active set pinned to
// a KNOWN card the token enters and leaves. This lane does NOT do that measurement; it
// makes the seam exist so the placement lane is a measurement + config exercise instead
// of an engine rewrite.
//
// The artifact is the FROZEN shared format `memra-ep-map-v1`, minted by
// `tools/build_expert_placement_map.py` (merged on main, 4e46be545) from
// `MEMRA_MOE_TRACE` route traces. Reading the shared format rather than a lane-local one
// is the whole point: the glm5 arm consumes the same maps through `MEMRA_GLM5_EP_MAP`,
// so a map minted from qwen4_exp traces is comparable with theirs.
//
// Door: `MEMRA_Q4E_EP_MAP=<path>`. UNSET is the EVEN split — this lane's CONTROL ARM,
// and bit-identical to the pre-placement engine BY CONSTRUCTION, not by tolerance: an
// even assignment makes the card-1 bank gather a contiguous copy of exactly the suffix
// the old code sliced, and leaves card 0 addressing its full resident bank by global id.
// Default OFF is a deliberate decision under the new-flags law: an unmeasured placement
// must not become the serving default, and no placement has been measured yet.
//
// Fail-closed, loudly, on every mismatch (a map that silently half-applies would move
// expert weights under the router and read as a model bug):
//   * format != memra-ep-map-v1, or ranks != 2 (TP2 is a two-card route)
//   * expert_count != the plan's expert count
//   * a MoE layer in the plan missing from the map, or an assignment of the wrong length
//   * a rank id outside {0, 1}
//   * an UNBALANCED layer: card 1 must own exactly experts/2. The card-1 bank halves are
//     equal-size device allocations, so an unbalanced map is not a slower placement, it
//     is an out-of-bounds one. The placement lane must balance inside the tool (it has
//     `--balance-tolerance`) and ranks==2 with expert_count even means exact halves.
#[derive(Debug, Clone)]
pub struct Tp2Placement {
    /// layer index -> rank (0 or 1) per GLOBAL expert id. Empty map = even split.
    by_layer: std::collections::BTreeMap<u32, Vec<u8>>,
    expert_count: usize,
    entry_rank: u8,
    strategy: String,
    source: String,
}

/// One layer's resolved placement. Card 0 keeps the FULL resident bank, so a card-0
/// expert's local slot IS its global id (no remap, exactly as the even split behaved);
/// card 1 holds a gathered half, so its local slot is the position in `card1`.
#[derive(Debug, Clone)]
pub struct LayerPlacement {
    /// GLOBAL expert ids owned by card 1, ASCENDING — the bank gather order and the
    /// local-slot order. Ascending is load-bearing: it makes the even case a contiguous
    /// copy, and it makes the gather order a function of the map alone (no host set
    /// iteration order can leak into device bytes).
    pub card1: Vec<u32>,
    /// global expert id -> local slot on its owner card.
    local_of: Vec<u32>,
    /// global expert id -> owner rank.
    rank_of: Vec<u8>,
}

impl LayerPlacement {
    #[inline]
    pub fn rank(&self, expert: usize) -> u8 {
        self.rank_of[expert]
    }
    #[inline]
    pub fn local(&self, expert: usize) -> usize {
        self.local_of[expert] as usize
    }
    /// True when this layer is the plain contiguous even split (the control arm).
    pub fn is_even(&self) -> bool {
        let half = self.rank_of.len() / 2;
        self.card1.len() == half
            && self
                .card1
                .iter()
                .enumerate()
                .all(|(i, &e)| e as usize == half + i)
    }
}

impl Tp2Placement {
    /// The even split: `rank = expert / (experts / 2)`, the engine's historical law.
    pub fn even(expert_count: usize) -> Self {
        Self {
            by_layer: std::collections::BTreeMap::new(),
            expert_count,
            entry_rank: 0,
            strategy: "even".to_string(),
            source: "built-in (MEMRA_Q4E_EP_MAP unset)".to_string(),
        }
    }

    pub fn strategy(&self) -> &str {
        &self.strategy
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn entry_rank(&self) -> u8 {
        self.entry_rank
    }

    /// Read `MEMRA_Q4E_EP_MAP`; `Ok(None)` when the door is closed (even split).
    pub fn from_env(expert_count: usize) -> Res<Option<Self>> {
        let Ok(path) = std::env::var("MEMRA_Q4E_EP_MAP") else {
            return Ok(None);
        };
        if path.is_empty() || path == "0" {
            return Ok(None);
        }
        Some(Self::load(std::path::Path::new(&path), expert_count)).transpose()
    }

    pub fn load(path: &std::path::Path, expert_count: usize) -> Res<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("MEMRA_Q4E_EP_MAP {}: {e}", path.display()))?;
        let v = memra_tokenizer::json::parse(&text)
            .map_err(|e| format!("MEMRA_Q4E_EP_MAP {}: {e}", path.display()))?;
        // Every refusal in this function names the file and the exact contract clause
        // broken: a rejected map has to tell the placement lane what to fix.
        let want = |k: &str| -> Res<Self> {
            Err(format!("MEMRA_Q4E_EP_MAP {}: {k}", path.display()).into())
        };
        match v.get("format").and_then(|f| f.as_str()) {
            Some("memra-ep-map-v1") => {}
            other => {
                return want(&format!(
                    "format is {other:?}, expected \"memra-ep-map-v1\" (mint it with \
                     tools/build_expert_placement_map.py)"
                ));
            }
        }
        let ranks = v.get("ranks").and_then(|r| r.as_u64()).unwrap_or(0);
        if ranks != 2 {
            return want(&format!(
                "ranks={ranks}, but the TP2 route is exactly two cards"
            ));
        }
        let map_experts = v.get("expert_count").and_then(|r| r.as_u64()).unwrap_or(0) as usize;
        if map_experts != expert_count {
            return want(&format!(
                "expert_count={map_experts} but this plan has {expert_count} experts"
            ));
        }
        // An ODD routed bank has no equal halves. Note precisely what the balance clause below
        // does and does not do here, because "it was already covered" is the easy wrong reading:
        // `half = expert_count / 2` FLOORS, so on 5 experts a map placing exactly 2 on card 1
        // SATISFIES `on1 == half` and loaded clean before this check existed. The balance clause
        // caught only the unbalanced odd maps, and for those it named the wrong problem (it read
        // as a rebalance request against a bank that cannot be balanced). Refuse the geometry by
        // name instead. Checked here AND in `layer()` because the built-in even split never
        // passes through this parser.
        if expert_count % 2 != 0 {
            return want(&format!(
                "this plan has {expert_count} routed experts, which is ODD: the TP2 route \
                 splits the bank into two EQUAL-size device allocations, so no two-card \
                 placement exists for it"
            ));
        }
        let entry_rank = v.get("entry_rank").and_then(|r| r.as_u64()).unwrap_or(0) as u8;
        if entry_rank > 1 {
            return want(&format!("entry_rank={entry_rank} outside {{0,1}}"));
        }
        let strategy = v
            .get("strategy")
            .and_then(|s| s.as_str())
            .unwrap_or("unnamed")
            .to_string();
        let Some(layers) = v.get("layers").and_then(|l| l.as_arr()) else {
            return want("no `layers` array");
        };
        let half = expert_count / 2;
        let mut by_layer = std::collections::BTreeMap::new();
        for row in layers {
            let Some(index) = row.get("layer").and_then(|l| l.as_u64()) else {
                return want("a layer row without an integer `layer`");
            };
            let Some(assign) = row.get("assignment").and_then(|a| a.as_arr()) else {
                return want(&format!("layer {index}: no `assignment` array"));
            };
            if assign.len() != expert_count {
                return want(&format!(
                    "layer {index}: assignment has {} entries, expected {expert_count}",
                    assign.len()
                ));
            }
            let mut ranks_vec = Vec::with_capacity(expert_count);
            for (eid, a) in assign.iter().enumerate() {
                match a.as_u64() {
                    Some(r) if r <= 1 => ranks_vec.push(r as u8),
                    other => {
                        return want(&format!(
                            "layer {index} expert {eid}: rank {other:?} outside {{0,1}}"
                        ));
                    }
                }
            }
            let on1 = ranks_vec.iter().filter(|&&r| r == 1).count();
            if on1 != half {
                return want(&format!(
                    "layer {index}: card 1 owns {on1} experts but the bank halves are \
                     equal-size allocations, so it must own exactly {half} — rebalance \
                     the map (build_expert_placement_map.py --balance-tolerance)"
                ));
            }
            by_layer.insert(index as u32, ranks_vec);
        }
        if by_layer.is_empty() {
            return want("`layers` is empty");
        }
        Ok(Self {
            by_layer,
            expert_count,
            entry_rank,
            strategy,
            source: path.display().to_string(),
        })
    }

    /// Resolve one MoE layer. A loaded map MUST cover every MoE layer it is asked about
    /// (fail-closed: silently falling one layer back to even would make the receipt a
    /// lie about which placement ran).
    pub fn layer(&self, index: u32, expert_count: usize) -> Res<LayerPlacement> {
        if expert_count != self.expert_count {
            return Err(format!(
                "qwen4exp_gpu tp2 placement: layer {index} has {expert_count} experts, \
                 map is for {}",
                self.expert_count
            )
            .into());
        }
        // DEFENSE IN DEPTH ON A `pub` API, and scoped honestly: production cannot reach this
        // with an odd bank, because the only caller (`build_tp2_shard`) already refuses
        // `experts % 2 != 0` eleven lines before it asks for a `LayerPlacement`. So this is not
        // a latent out-of-bounds and nothing was silently wrong: the card-1 bank upload sizes
        // its allocation on `place.card1.len()`, so an odd split would have produced an
        // UNBALANCED (3-of-5) card-1 half, not an overflowing one.
        //
        // What it does buy: `layer()` and `load()` are `pub`, and on an odd bank the even split
        // `rank = expert / (experts/2)` has no two-card answer at all. Naming that geometry here
        // means a future caller gets the refusal from the function whose contract it breaks
        // instead of relying on an upstream check it may not have. It also closes a real hole in
        // `load()`: `half` FLOORS, so a map placing exactly 2 of 5 experts on card 1 satisfied
        // the balance clause and loaded clean before this check existed.
        if expert_count % 2 != 0 {
            return Err(format!(
                "qwen4exp_gpu tp2 placement: layer {index} has {expert_count} routed \
                 experts, which is ODD: the TP2 route splits the bank into two EQUAL-size \
                 device allocations, so no two-card placement exists for it"
            )
            .into());
        }
        let half = expert_count / 2;
        let rank_of: Vec<u8> = if self.by_layer.is_empty() {
            (0..expert_count).map(|e| u8::from(e >= half)).collect()
        } else {
            self.by_layer
                .get(&index)
                .ok_or_else(|| {
                    format!(
                        "qwen4exp_gpu tp2 placement: map {} does not cover MoE layer \
                         {index} (fail-closed; a partly-applied map is not a placement)",
                        self.source
                    )
                })?
                .clone()
        };
        let card1: Vec<u32> = (0..expert_count)
            .filter(|&e| rank_of[e] == 1)
            .map(|e| e as u32)
            .collect();
        let mut local_of = vec![0u32; expert_count];
        for (slot, &eid) in card1.iter().enumerate() {
            local_of[eid as usize] = slot as u32;
        }
        // Card 0 addresses its FULL resident bank by global id.
        for e in 0..expert_count {
            if rank_of[e] == 0 {
                local_of[e] = e as u32;
            }
        }
        Ok(LayerPlacement {
            card1,
            local_of,
            rank_of,
        })
    }
}

/// Engagement counter: PEER-owned (card 1) expert slots dispatched by the TP2 MoE split,
/// since process start. Copied from the glm5 TP lane's
/// `GLM5_EP_PEER_SLOT_DISPATCHES` for the reason that lane learned the hard way — its
/// first seed search found a token stream that NEVER routed a peer expert, so the arm's
/// identity claim would have been VACUOUS. Any TP2 exactness claim must assert this
/// counter moved, or it is a claim about a program that did not run.
static TP2_PEER_EXPERT_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Same for card 0, so a receipt can print the per-rank token-touch/byte split instead of
/// only proving non-vacuity (the glm5 lane reported its ~99.3% peer-touch and ~64%
/// slowest-rank byte fraction as CLOSED-FORM derivations with no measurement behind them;
/// these two counters are what make ours measured).
static TP2_HOME_EXPERT_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Layer-tokens whose top-k touched BOTH cards (the "peer is on the critical path"
/// fraction — the number the glm5 lane derived as ~99.3% for its 288/top-8 geometry).
static TP2_BOTH_TOUCH_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TP2_TOUCH_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (peer slots, home slots, rows touching both cards, rows counted) since process start.
pub fn tp2_expert_split_stats() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        TP2_PEER_EXPERT_SLOTS.load(Relaxed),
        TP2_HOME_EXPERT_SLOTS.load(Relaxed),
        TP2_BOTH_TOUCH_ROWS.load(Relaxed),
        TP2_TOUCH_ROWS.load(Relaxed),
    )
}

fn tp2_count_split(routes0: &[Vec<(usize, f32)>], routes1: &[Vec<(usize, f32)>]) {
    use std::sync::atomic::Ordering::Relaxed;
    let (mut peer, mut home, mut both) = (0u64, 0u64, 0u64);
    for (r0, r1) in routes0.iter().zip(routes1.iter()) {
        home += r0.len() as u64;
        peer += r1.len() as u64;
        if !r0.is_empty() && !r1.is_empty() {
            both += 1;
        }
    }
    TP2_HOME_EXPERT_SLOTS.fetch_add(home, Relaxed);
    TP2_PEER_EXPERT_SLOTS.fetch_add(peer, Relaxed);
    TP2_BOTH_TOUCH_ROWS.fetch_add(both, Relaxed);
    TP2_TOUCH_ROWS.fetch_add(routes0.len() as u64, Relaxed);
}

/// Gate-only deliberate defects for the TP2 class gate: `MEMRA_Q4E_TP2_GATE_RED=<name>`.
/// A band is only a bar if a WRONG program lands orders outside it, so the gate runs
/// these and REQUIRES them to be loud (the glm5 `MEMRA_GLM5_TP_GATE_RED` pattern).
/// Never a serving door — an unknown value refuses at the first MoE layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tp2GateRed {
    None,
    /// Drop the peer card's routed-expert contribution from the join.
    SkipPeerMoe,
    /// Route peer-owned experts to card 0's bank at their LOCAL slot — a plausible
    /// off-by-remap bug (right magnitudes, wrong experts).
    PeerLocalIds,
    /// Feed the peer half its slot weights in reversed order within each token.
    ReverseePeerWeights,
}

fn tp2_gate_red() -> Res<Tp2GateRed> {
    static C: std::sync::OnceLock<Result<Tp2GateRed, String>> = std::sync::OnceLock::new();
    C.get_or_init(
        || match std::env::var("MEMRA_Q4E_TP2_GATE_RED").as_deref() {
            Err(_) | Ok("") | Ok("0") | Ok("none") => Ok(Tp2GateRed::None),
            Ok("skip-peer-moe") => Ok(Tp2GateRed::SkipPeerMoe),
            Ok("peer-local-ids") => Ok(Tp2GateRed::PeerLocalIds),
            Ok("reverse-peer-weights") => Ok(Tp2GateRed::ReverseePeerWeights),
            Ok(other) => Err(format!(
                "MEMRA_Q4E_TP2_GATE_RED={other:?}: want skip-peer-moe|peer-local-ids|\
             reverse-peer-weights|none"
            )),
        },
    )
    .clone()
    .map_err(Into::into)
}

/// Per-layer MoE route trace in the FROZEN shared format
/// `tools/build_expert_placement_map.py` consumes (`<layer> <t> <id,id,...>`, one line
/// per (layer, forward); decode steps are t == 1) — byte-compatible with
/// `hybrid_forward.rs::trace_moe_routes` so one tool reads both arms' traces.
///
/// Doors: `MEMRA_MOE_TRACE` (ids) and `MEMRA_MOE_WEIGHT_TRACE` (`<expert>:<weight>`).
/// Both OFF by default: this writes an unbounded append-only file and costs host I/O per
/// layer per forward, which is fine for a battery and wrong for serving.
///
/// Where it taps, and the honest limit: the qwen4_exp MoE route exists on the HOST on
/// the TP2 route (which keeps the host router twin by construction) and on the
/// per-expert prefill executor. Under the shipped single-card default the route is
/// DEVICE-side (`routerdev`, PROFILE-9) with no readback at all, so there is nothing to
/// tap without re-adding the very sync that lane deleted — arming
/// `MEMRA_Q4E_ROUTER_AUDIT=1` restores a host recompute of every device route and the
/// trace rides THAT readback at zero new syncs. So: TP2 batteries trace for free;
/// single-card batteries trace with the audit armed.
fn trace_moe_routes(layer: u32, t: usize, routes: &[Vec<(usize, f32)>]) {
    use std::io::Write as _;
    static IDS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    static WEIGHTS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let ids = IDS.get_or_init(|| {
        std::env::var("MEMRA_MOE_TRACE")
            .ok()
            .filter(|p| !p.is_empty())
    });
    let weights = WEIGHTS.get_or_init(|| {
        std::env::var("MEMRA_MOE_WEIGHT_TRACE")
            .ok()
            .filter(|p| !p.is_empty())
    });
    if ids.is_none() && weights.is_none() {
        return;
    }
    // One line per (layer, forward) with EVERY row's selections concatenated is what the
    // shared format specifies for t > 1 forwards, and the tool's co-occurrence is
    // "within-line", so a prefill chunk's line legitimately carries t tokens' picks.
    let flat: Vec<&(usize, f32)> = routes.iter().flatten().collect();
    let append = |path: &str, body: String| {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{layer} {t} {body}");
        }
    };
    if let Some(path) = ids {
        let body: Vec<String> = flat.iter().map(|(e, _)| e.to_string()).collect();
        append(path, body.join(","));
    }
    if let Some(path) = weights {
        let body: Vec<String> = flat.iter().map(|(e, w)| format!("{e}:{w:.9}")).collect();
        append(path, body.join(","));
    }
}

/// Arm or disarm ONE seam by its `MEMRA_Q4E_SEAMS` name, returning false when the name is
/// unknown. Extracted from `apply_env_seams` (which is now its only-at-startup caller) so a
/// measurement harness can flip a seam BETWEEN timed rounds inside one process — the
/// interleaved-A/B instrument the 262k host lane needs, because at these depths a per-arm
/// process pays a fresh 25-80 minute prefill for a decode-only lever and box clock drift
/// then sits between the arms. `value` carries the raw `name=value` right-hand side for the
/// three-valued seams; `on` is already decoded for the boolean ones.
///
/// This is a MEASUREMENT seam-setter, not a serving one: flipping a seam mid-run is sound
/// only for seams whose state is rebuildable from the token history (`plecache` appends to a
/// cache it can also rebuild by longest-common-prefix), and the caller owns that judgement.
pub fn set_seam(name: &str, on: bool, value: Option<&str>) -> bool {
    seam_dispatch(name, on, value, true)
}

/// The CURRENT boolean state of a seam, for exact save/restore around a measurement that
/// flips it. `None` for a name with no boolean state (`idxq` is three-valued, `longatt`
/// three-valued) and for an unknown name.
///
/// This exists because the alternative — "restore by re-running `apply_env_seams`" — is
/// wrong in a way that would not show up as a failure: a seam absent from
/// `MEMRA_Q4E_SEAMS` is not reset by that call, so the run would silently continue on
/// whichever arm happened to execute last. Save/restore has to read the real state.
pub fn seam_state(name: &str) -> Option<bool> {
    Some(match name {
        "moe" => moe_sel_path_on(),
        "hc" => hc_fused_gate_on(),
        "trunk" => trunk_bf16_on(),
        "ws" => step_ws_on(),
        "graph" => decode_graphs_on(),
        "selv2" => sel_v2_on(),
        "hcmicro" => hc_micro_on(),
        "selv3" => sel_v3_on(),
        "gdnstep" => gdn_step_on(),
        "gdnfuse" => gdn_fuse_on(),
        "projstack" => proj_stack_on(),
        "hcdiet" => hc_diet_on(),
        "gufuse" => sel_gufuse_on(),
        "routerb16" => router_bf16_on(),
        "vgraph" => verify_graphs_on(),
        "vfuse" => verify_fused_on(),
        "idxdev" => idx_dev_on(),
        "idxsel" => idx_sel_on(),
        "plecache" => ple_cache_on(),
        "routerdev" => router_dev_on(),
        "idxcache" => idx_cache_on(),
        "kvq" => kv_quant_on(),
        "kvhoist" => kv_hoist_on(),
        "poolT" => pool_t_on(),
        // Shape-valued, but it DOES carry a boolean state and must report it: the shared
        // `--ab-seam` / `--ladder-ab-seam` harness restores the entry arm only when
        // `seam_state` answers, and returning None there would leave the ON arm armed for
        // every number after the A/B block — the silent arm flip that harness's own comment
        // warns about. OFF <-> AUTO (the two arms a seam A/B runs) round-trips exactly.
        // Restoring a PINNED shape does not: it comes back as AUTO, so a cell that pins
        // `dn:8:1` must carry it in `MEMRA_Q4E_SEAMS` per invocation (which is how the
        // banked downsel cells run their ladder) or save/restore `sel_group_spec()`.
        "selgroup" => sel_group_dn() != SEL_GROUP_OFF || sel_group_gu() != SEL_GROUP_OFF,
        _ => return None,
    })
}

/// Does this seam name exist? Same table as `set_seam`, applying NOTHING. A harness that
/// validates a seam name up front (before a 25-80 minute prefill it would otherwise waste on
/// a typo) must not have to arm or disarm the seam to find out — a validator with a silent
/// side effect on global state is the kind of thing that later reads as a mystery flip.
pub fn seam_exists(name: &str) -> bool {
    seam_dispatch(name, false, None, false)
}

/// Every seam name `seam_dispatch` accepts, as DATA, sitting directly above the match so the two
/// are read together. `gate_seam_table` walks this list, so a name here that the match does not
/// accept fails that gate loudly.
///
/// The reverse drift — a match arm added without a list entry — is NOT machine-detectable from
/// here, and that seam is then uncovered rather than wrong. Said out loud instead of dressed up
/// as completeness, because a non-vacuity check that cannot fail is worse than no check.
/// **Adding a seam: add its arm below AND its name here.**
pub fn seam_names() -> &'static [&'static str] {
    &[
        "moe",
        "hc",
        "trunk",
        "ws",
        "graph",
        "selv2",
        "hcmicro",
        "selv3",
        "gdnstep",
        "gdnfuse",
        "projstack",
        "hcdiet",
        "gufuse",
        "routerb16",
        "vgraph",
        "vfuse",
        "longatt",
        "idxdev",
        "idxsel",
        "plecache",
        "routerdev",
        "idxcache",
        "kvq",
        "idxq",
        "kvhoist",
        "poolT",
        "selgroup",
    ]
}

/// The one seam name table. `apply` false walks the same arms and calls no setter, so the
/// name check and the action can never drift apart.
fn seam_dispatch(name: &str, on: bool, value: Option<&str>, apply: bool) -> bool {
    macro_rules! seam {
        ($call:expr) => {{
            if apply {
                $call;
            }
            true
        }};
    }
    match name {
        "moe" => seam!(set_moe_sel_path(on)),
        "hc" => seam!(set_hc_fused_gate(on)),
        "trunk" => seam!(set_trunk_bf16(on)),
        "ws" => seam!(set_step_ws(on)),
        "graph" => seam!(set_decode_graphs(on)),
        "selv2" => seam!(set_sel_v2(on)),
        "hcmicro" => seam!(set_hc_micro(on)),
        "selv3" => seam!(set_sel_v3(on)),
        "gdnstep" => seam!(set_gdn_step(on)),
        "gdnfuse" => seam!(set_gdn_fuse(on)),
        "projstack" => seam!(set_proj_stack(on)),
        "hcdiet" => seam!(set_hc_diet(on)),
        "gufuse" => seam!(set_sel_gufuse(on)),
        "routerb16" => seam!(set_router_bf16(on)),
        "vgraph" => seam!(set_verify_graphs(on)),
        // COST INSTRUMENT, no rewind (see VERIFY_FUSED_DEFAULT): a spec loop with this
        // armed refuses at the first `verify_rewind`. Timing probes only.
        "vfuse" => seam!(set_verify_fused(on)),
        "longatt" => seam!(set_longatt(if on { "force" } else { "off" })),
        "idxdev" => seam!(set_idx_dev(on)),
        "idxsel" => seam!(set_idx_sel(on)),
        "plecache" => seam!(set_ple_cache(on)),
        "routerdev" => seam!(set_router_dev(on)),
        "idxcache" => seam!(set_idx_cache(on)),
        "kvq" => seam!(set_kv_quant(on)),
        // Bit-identical READ-PATTERN seams (memory lane): no layout latch on `kvhoist`,
        // and `poolT` re-mirrors on flip, so both are sound to flip between timed rounds.
        "kvhoist" => seam!(set_kv_hoist(on)),
        "poolT" => seam!(set_pool_t(on)),
        // Three-valued: `idxq=q8`, `idxq=bf16`, `idxq=0`/`idxq=f32` (rollback);
        // bare `idxq` arms the q8 target.
        "idxq" => seam!(set_idxq(value.unwrap_or("q8"))),
        // SHAPE-valued (see set_sel_group): bare `selgroup` = both families AUTO,
        // `selgroup=dn:4:1+gu:16:2` pins the A/B ladder's arms, `selgroup=0` rolls back.
        // A malformed spec must not read as "seam applied": it returns false here so the
        // caller reports an unknown/bad seam instead of measuring the default arm.
        "selgroup" => {
            if apply {
                set_sel_group(if on { value.unwrap_or("auto") } else { "off" })
            } else {
                true
            }
        }
        _ => {
            debug_assert!(
                !seam_names().contains(&name),
                "seam_names() lists {name:?} but seam_dispatch has no arm for it"
            );
            false
        }
    }
}

pub fn apply_env_seams() {
    let Ok(spec) = std::env::var("MEMRA_Q4E_SEAMS") else {
        return;
    };
    for part in spec.split(',').filter(|p| !p.is_empty()) {
        let (name, on) = match part.split_once('=') {
            Some((n, v)) => (n, v != "0"),
            None => (part, true),
        };
        if !set_seam(name, on, part.split_once('=').map(|(_, v)| v)) {
            eprintln!("MEMRA_Q4E_SEAMS: unknown seam {name:?} ignored");
        }
    }
}

/// Per-piece kill switches for the hcmicro bundle (bisect instrumentation: set
/// MEMRA_Q4E_MICRO_{NORM,INJ,SHEXP}=0 to fall a single piece back while the seam stays
/// on). Read once per process.
fn micro_env_on(name: &'static str, cell: &'static std::sync::OnceLock<bool>) -> bool {
    *cell.get_or_init(|| std::env::var(name).as_deref() != Ok("0"))
}

fn micro_norm_on() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    hc_micro_on() && micro_env_on("MEMRA_Q4E_MICRO_NORM", &C)
}

fn micro_inj_on() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    hc_micro_on() && micro_env_on("MEMRA_Q4E_MICRO_INJ", &C)
}

fn micro_shexp_on() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    hc_micro_on() && micro_env_on("MEMRA_Q4E_MICRO_SHEXP", &C)
}

/// Run `f` as a named profile section (sync–time–sync when profiling is on).
fn prof_section<T>(e: &Engine, name: &'static str, f: impl FnOnce() -> Res<T>) -> Res<T> {
    if !prof::on() {
        return f();
    }
    e.gpu.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    let out = f()?;
    e.gpu.stream().synchronize()?;
    prof::add(name, t0.elapsed().as_secs_f64());
    Ok(out)
}

// ---------------------------------------------------------------- host twins (oracle math)

fn host_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// memra_reference `softmax_in_place` twin.
fn host_softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

/// The router renorm denominator floor (memra_reference `route_experts`, mirrored by
/// the device twin). Note it is UNBINDABLE on real softmax geometry: the top-k weights
/// are the k largest of a distribution summing to 1, so their sum is >= k/experts
/// (10/512 ~ 0.0195 >> 6.1e-5) — kept because the reference ships it.
const ROUTE_DENOM_FLOOR: f32 = 6.103_515_6e-5;

/// memra_reference `route_experts` twin, Softmax arm only (qwen4_exp router — softmax,
/// top-k renormalized with the 6.1035156e-5 floor, tie rule score-desc/index-asc).
fn host_route_softmax_topk(logits: &[f32], selected: usize) -> Vec<(usize, f32)> {
    let mut weights = logits.to_vec();
    host_softmax(&mut weights);
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&left, &right| {
        weights[right]
            .total_cmp(&weights[left])
            .then(left.cmp(&right))
    });
    indices.truncate(selected);
    let denominator = indices
        .iter()
        .map(|&index| weights[index])
        .sum::<f32>()
        .max(ROUTE_DENOM_FLOOR);
    indices
        .into_iter()
        .map(|index| (index, weights[index] / denominator))
        .collect()
}

/// memra_reference `rms_norm` twin (host, effective weights).
fn host_rms_norm(x: &mut [f32], width: usize, weight: &[f32], epsilon: f32) {
    for row in x.chunks_exact_mut(width) {
        let mean_square = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inverse = 1.0 / (mean_square + epsilon).sqrt();
        for (value, w) in row.iter_mut().zip(weight) {
            *value = *value * inverse * w;
        }
    }
}

/// memra_reference `apply_rope_at_position` twin (NeoX split-half). `yarn` = the shared
/// (divisor table, mscale) pair when the plan carries YaRN factors — identical divisor
/// semantics to the reference (`frequency / divisor`, cos/sin scaled by mscale); `None`
/// keeps the historical byte-exact plain path.
fn host_rope_at(
    values: &mut [f32],
    head_dim: usize,
    dimensions: usize,
    base: f32,
    yarn: Option<(&[f32], f32)>,
    position: usize,
) {
    let dimensions = dimensions.min(head_dim) / 2 * 2;
    let half = dimensions / 2;
    for head in values.chunks_exact_mut(head_dim) {
        for index in 0..half {
            let frequency = base.powf(-2.0 * index as f32 / dimensions as f32);
            let frequency = match yarn {
                Some((ff, _)) => frequency / ff[index],
                None => frequency,
            };
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let (sin, cos) = match yarn {
                Some((_, mscale)) => (sin * mscale, cos * mscale),
                None => (sin, cos),
            };
            let first = head[index];
            let second = head[index + half];
            head[index] = first * cos - second * sin;
            head[index + half] = first * sin + second * cos;
        }
    }
}

/// What the forward's exit computes (chunked long-context prefill skips the head: the
/// [t, vocab] logits block of a big chunk is gigabytes and reads/writes no state).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HeadMode {
    /// Exit mixer + lm_head on every row ([t, vocab] logits) — the historical shape.
    All,
    /// Exit mixer on the chunk, lm_head on the LAST row only ([vocab] logits).
    LastRow,
    /// No exit mixer, no lm_head, empty return (mid-prefill chunks).
    Skip,
}

/// One query row's QSA visibility in BLOCK form — the selection's native shape (the
/// dense [t, t_kv] mask is a rendering of this for the smem-bounded masked kernel; the
/// long-context block-list kernel consumes it directly).
struct RowSel {
    /// Structural fast path (complete <= budget): the FULL causal prefix is visible.
    full: bool,
    /// Selected complete blocks, ascending. Empty when `full`.
    blocks: Vec<u32>,
    /// Visible prefix length (absolute row + 1). Positions
    /// [complete*block_size .. visible) are the always-visible incomplete tail.
    visible: usize,
}

/// Extend the POOLED indexer-key cache to cover every complete block of `raw_keys`:
/// fp32 mean over the block's raw rows (offset-outer/dim-inner, the historical loop
/// order), k_layernorm, rope at the block-start position + pos_off. A block's pooled key
/// never depends on the query row, so each block is computed ONCE — bit-identical to the
/// historical per-(row, block) recompute.
#[allow(clippy::too_many_arguments)]
fn extend_pooled_keys(
    pooled_keys: &mut Vec<f32>,
    raw_keys: &IdxRawCache,
    head_dim: usize,
    block_size: usize,
    idx_k_norm: &[f32],
    epsilon: f32,
    rope_dims: usize,
    rope_base: f32,
    yarn: Option<(&[f32], f32)>,
    pos_off: usize,
) {
    let complete_total = raw_keys.rows(head_dim) / block_size;
    let cached = pooled_keys.len() / head_dim;
    let mut block_rows: Vec<f32> = Vec::new();
    for block in cached..complete_total {
        let start = block * block_size;
        // idxq lane: dequant the block's raw rows at read; the fp32 mean-pool below is
        // the historical op order verbatim (f32 arm: an exact copy of the same rows).
        raw_keys.rows_f32(start, block_size, head_dim, &mut block_rows);
        let mut pooled = vec![0.0f32; head_dim];
        for offset in 0..block_size {
            for dim in 0..head_dim {
                pooled[dim] += block_rows[offset * head_dim + dim];
            }
        }
        for value in &mut pooled {
            *value /= block_size as f32;
        }
        host_rms_norm(&mut pooled, head_dim, idx_k_norm, epsilon);
        host_rope_at(
            &mut pooled,
            head_dim,
            rope_dims,
            rope_base,
            yarn,
            start + pos_off,
        );
        pooled_keys.extend_from_slice(&pooled);
    }
}

/// Comparator of the pinned tie rule: score desc, block index asc (a STRICT total order
/// — `total_cmp` plus the index tiebreak leaves no equal pair).
#[inline]
fn sel_cmp(scores: &[f32], a: u32, b: u32) -> std::cmp::Ordering {
    scores[b as usize]
        .total_cmp(&scores[a as usize])
        .then(a.cmp(&b))
}

/// Top-`budget` blocks under the pinned tie rule, returned ASCENDING. Replaces the
/// historical full `sort_by` + `take(budget)` with `select_nth_unstable_by` under the
/// SAME strict total order — the kept SET is identical by definition of a total order
/// (both keep exactly the `budget` smallest elements under the comparator), and the
/// emitted ascending order erases any within-set permutation. When the block count is
/// large, disjoint ranges are reduced to per-range top-`budget` candidates first: any
/// global top-`budget` element is beaten by fewer than `budget` blocks overall, hence by
/// fewer than `budget` in its own range, hence survives its range cut — the union of
/// range winners contains the global set, and the final cut recovers it EXACTLY.
fn top_blocks_ascending(scores: &[f32], budget: usize, threads: usize) -> Vec<u32> {
    fn cut(scores: &[f32], idx: &mut Vec<u32>, budget: usize) {
        let k = budget.min(idx.len());
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| sel_cmp(scores, a, b));
            idx.truncate(k);
        }
    }
    let complete = scores.len();
    debug_assert!(budget < complete);
    const PAR_MIN: usize = 1 << 15;
    let mut candidates: Vec<u32> = if threads > 1 && complete >= PAR_MIN {
        let ranges: Vec<(u32, u32)> = {
            let per = complete.div_ceil(threads);
            (0..threads)
                .map(|i| ((i * per) as u32, ((i + 1) * per).min(complete) as u32))
                .filter(|(a, b)| a < b)
                .collect()
        };
        std::thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .iter()
                .map(|&(a, b)| {
                    scope.spawn(move || {
                        let mut idx: Vec<u32> = (a..b).collect();
                        cut(scores, &mut idx, budget);
                        idx
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    } else {
        (0..complete as u32).collect()
    };
    cut(scores, &mut candidates, budget);
    candidates.sort_unstable();
    candidates
}

/// Score every complete block for one prepared query row (relu-sum over heads / sqrt(d),
/// fp32 — the reference arithmetic verbatim, reading the pooled cache). Parallel over
/// DISJOINT block ranges when large: per-block values are independent, so the split
/// changes nothing but wall time.
fn score_blocks(
    query: &[f32],
    pooled_keys: &[f32],
    heads: usize,
    head_dim: usize,
    complete: usize,
    scale: f32,
    threads: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; complete];
    let run = |scores: &mut [f32], block0: usize| {
        for (i, slot) in scores.iter_mut().enumerate() {
            let block = block0 + i;
            let pooled = &pooled_keys[block * head_dim..(block + 1) * head_dim];
            let mut score = 0.0f32;
            for head in 0..heads {
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += query[head * head_dim + dim] * pooled[dim];
                }
                score += dot.max(0.0);
            }
            *slot = score / scale;
        }
    };
    const PAR_MIN: usize = 1 << 14;
    if threads > 1 && complete >= PAR_MIN {
        let per = complete.div_ceil(threads);
        let run = &run;
        std::thread::scope(|scope| {
            for (i, chunk) in scores.chunks_mut(per).enumerate() {
                scope.spawn(move || run(chunk, i * per));
            }
        });
    } else {
        run(&mut scores, 0);
    }
    scores
}

/// memra_reference `micro_block_selection_mask` twin over the raw-key CACHE — the decode
/// form of the same program in BLOCK form: per query token at absolute position
/// `base_pos + qt`, score the pooled complete blocks (cache: `extend_pooled_keys`), then
/// the pinned tie rule (score desc, block index asc) and the always-visible incomplete
/// tail. Values and selected sets are bit-identical to the historical per-row recompute
/// (see the helper docs above); rows are computed in PARALLEL when the work is large
/// (rows are independent; single-row chunks parallelize across block ranges instead).
#[allow(clippy::too_many_arguments)]
fn indexer_select_rows(
    overlay: &MicroBlockIndexPlan,
    rope_base: f32,
    // YaRN (divisors, mscale) — the indexer consumes the MAIN rotary (SEMANTICS.md §Rope),
    // so the caller passes the layer's shared table; `None` on the shipped config.
    yarn: Option<(&[f32], f32)>,
    epsilon: f32,
    idx_q_norm: &[f32],
    idx_k_norm: &[f32],
    proj_rows: &[f32],      // [t, (ih+ikv)*id] this chunk's index_qk_proj output
    raw_keys: &IdxRawCache, // [t_kv, id] cache INCLUDING the current chunk
    pooled_keys: &mut Vec<f32>,
    // Device scorer (long-context lane): `Some((engine, device pooled mirror, mirrored
    // rows))` runs block scoring on the GPU with the host twin's exact arithmetic
    // (thread-per-block sequential dim loop, same relu-sum, same division — bit-identical
    // scores, identical selected sets); the mirror grows by H2D of the new rows. `None`
    // keeps the pure-host path (the tiny/reference shape).
    mut dev: Option<(&Engine, &mut Option<CudaSlice<f32>>, &mut usize)>,
    base_pos: usize,
    t: usize,
    t_kv: usize,
    // Rope-position offset: cache row i carries absolute position i + pos_off. 0 for
    // the trunk; 1 for the MTP draft, whose row i holds TARGET position i + 1
    // (position 0 never enters the draft — SGLang alignment, SEMANTICS.md §MTP).
    pos_off: usize,
) -> Res<Vec<RowSel>> {
    let heads = overlay.query_heads as usize;
    let head_dim = overlay.head_dim as usize;
    let block_size = overlay.block_size as usize;
    let budget_blocks = overlay.budget_blocks as usize;
    let rope_dims = overlay.rope_dimensions as usize;
    let qk_width = (heads + overlay.kv_heads as usize) * head_dim;
    let scale = (head_dim as f32).sqrt();
    debug_assert_eq!(raw_keys.rows(head_dim), t_kv);
    extend_pooled_keys(
        pooled_keys,
        raw_keys,
        head_dim,
        block_size,
        idx_k_norm,
        epsilon,
        rope_dims,
        rope_base,
        yarn,
        pos_off,
    );
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // ---- device scoring path: mirror the new pooled rows, then score in row
    // sub-batches (the score slab is rows x n_blocks floats — at 250k blocks a whole
    // prefill chunk of rows would be terabytes, so rows batch).
    if let Some((e, mirror, mirrored)) = dev.as_mut() {
        let rows_needed: Vec<usize> = (0..t)
            .map(|qt| (base_pos + qt + 1) / block_size)
            .filter(|&c| c > budget_blocks)
            .collect();
        if let Some(&max_blocks) = rows_needed.iter().max() {
            let pooled_rows = pooled_keys.len() / head_dim;
            // Grow + fill the device mirror with any rows it does not have yet.
            let want = pooled_rows.max(max_blocks);
            // POOL_PLANES regions of `cap_rows * head_dim`: the row-major mirror, then the
            // dim-major `poolT` plane. The pitch of the plane is `cap_rows`, so it is baked at
            // allocation and a capacity change invalidates the plane's addressing — hence the
            // full re-mirror below rather than a strided forward copy of the old plane.
            if mirror
                .as_ref()
                .is_none_or(|m| m.len() < want * head_dim * POOL_PLANES)
            {
                let cap_rows = want.next_power_of_two().max(1024);
                let fresh = e.zeros(cap_rows * head_dim * POOL_PLANES)?;
                // The old growth path copied the mirrored prefix forward and kept `**mirrored`.
                // That is not sound for the plane (new pitch => every dim lands elsewhere), and a
                // half-addressed plane scores stale keys silently. Re-mirror from the host cache
                // instead, which holds every row and is the same source the append already uses.
                // Costs one H2D of the pooled cache per capacity DOUBLING (log2 times over a
                // fill), against a class of wrong-value bug this lane has already paid for twice.
                **mirror = Some(fresh);
                **mirrored = 0;
            }
            let m = mirror.as_mut().expect("allocated above");
            if pooled_rows > **mirrored {
                let delta = &pooled_keys[**mirrored * head_dim..pooled_rows * head_dim];
                let mut view = m.slice_mut(**mirrored * head_dim..pooled_rows * head_dim);
                e.gpu.stream().memcpy_htod(delta, &mut view)?;
                // `poolT`: keep the DIM-MAJOR twin of the same rows in the second half of the
                // buffer. Both layouts are maintained UNCONDITIONALLY and only the kernel choice
                // reads the seam. Two reasons, and the second is the important one:
                //
                //  - Experimental design. The append is then identical in both A/B arms, so the
                //    measurement isolates exactly the variable under test (the READ pattern) and
                //    the transpose cost cannot flatter or penalise either arm.
                //  - There is no silent-wrong-value mode. A seam that is flippable between timed
                //    rounds plus a layout that is only maintained while armed means an arm that
                //    was OFF for a while leaves the plane missing every row appended meanwhile —
                //    and a stale pooled plane scores stale keys, which reads as plausible output
                //    rather than as a failure. Maintaining both makes `**mirrored` the single
                //    truth for BOTH layouts, so a flip needs no rebuild and can leave nothing
                //    behind. (Same class as the `pooled_dev_rows` truncation trap already
                //    recorded at the rewind sites.)
                //
                // Instrument cost, stated: one pooled plane of extra VRAM (33.5 MB at the 262,144
                // target geometry, 1.6% of the ~2 GB free there) plus one transpose over the
                // delta — 512 rows per 2,048-token prefill chunk, 0-1 rows per decode step. When
                // the A/B verdict lands, the losing layout goes away in the same commit; carrying
                // both is an A/B instrument, not a shipping design.
                let cap_rows = m.len() / (head_dim * POOL_PLANES);
                launch_qsa_pooled_transpose(
                    e,
                    m,
                    **mirrored,
                    pooled_rows - **mirrored,
                    head_dim,
                    cap_rows,
                )?;
                **mirrored = pooled_rows;
            }
            // Per-row prepared queries (norm + rope) — the host twin's own preparation.
            let mut sels: Vec<RowSel> = Vec::with_capacity(t);
            let mut queries: Vec<f32> = Vec::new();
            let mut scored_rows: Vec<usize> = Vec::new();
            for qt in 0..t {
                let row = base_pos + qt;
                let visible = row + 1;
                let complete = visible / block_size;
                if complete <= budget_blocks {
                    sels.push(RowSel {
                        full: true,
                        blocks: Vec::new(),
                        visible,
                    });
                    continue;
                }
                let mut query = proj_rows[qt * qk_width..qt * qk_width + heads * head_dim].to_vec();
                host_rms_norm(&mut query, head_dim, idx_q_norm, epsilon);
                host_rope_at(
                    &mut query,
                    head_dim,
                    rope_dims,
                    rope_base,
                    yarn,
                    row + pos_off,
                );
                queries.extend_from_slice(&query);
                scored_rows.push(qt);
                sels.push(RowSel {
                    full: false,
                    blocks: Vec::new(),
                    visible,
                });
            }
            // Row sub-batches bounded by the score slab (default 32 M floats = 128 MB).
            //
            // TUNABLE because this constant appears to SET THE 262k PERFORMANCE CLIFF.
            // `qsa.idx_host` grows linearly with fill up to 120,000 (2,710 -> 3,199 ms) and then
            // jumps 16x to 51,235 ms — 83% of a prefill chunk — somewhere before 131,072. The
            // arithmetic lands exactly there: rows per sub-batch is `SCORE_CAP / complete`, and
            // at fill 131,072 `complete = 32,768`, so `per = 1,024` and 2,048 scored rows fit in
            // EXACTLY 2 sub-batches; one block deeper it becomes 3. Each sub-batch does an
            // `e.htod` plus an `e.uninit` of up to 128 MB and ends in a BLOCKING `dtoh`, at
            // depths where card 0 has ~2-4 GB free.
            //
            // The test this knob exists for: if the cliff MOVES with the cap, the mechanism is
            // the sub-batch transition (and the fix is a persistent pooled slab, or a cap that
            // keeps the transition out of the product window). If the cliff does NOT move, the
            // hypothesis is dead and the next suspect is the blocking dtoh count.
            // Default 32 reproduces today's behaviour exactly.
            let score_cap_mf: usize = std::env::var("MEMRA_Q4E_IDX_SCORE_CAP_MF")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(32);
            let score_cap: usize = score_cap_mf << 20;
            #[allow(non_snake_case)]
            let SCORE_CAP = score_cap;
            let mut done = 0usize;
            while done < scored_rows.len() {
                // Every row in a batch scores its OWN block count; the kernel writes a
                // rows x max_blocks slab and each row reads its own prefix.
                let batch_max = scored_rows[done..]
                    .iter()
                    .map(|&qt| (base_pos + qt + 1) / block_size)
                    .max()
                    .unwrap_or(0);
                let per = (SCORE_CAP / batch_max.max(1)).max(1);
                let n = per.min(scored_rows.len() - done);
                let qslab = &queries[done * heads * head_dim..(done + n) * heads * head_dim];
                let q_dev = e.htod(qslab)?;
                let mut scores_dev = e.uninit(n * batch_max)?;
                launch_qsa_index_score(
                    e,
                    &q_dev,
                    m,
                    &mut scores_dev,
                    heads,
                    head_dim,
                    batch_max,
                    n,
                    scale,
                )?;
                if idx_sel_on() {
                    // Device selection (`idxsel`): read back rows x budget u32 instead of
                    // the rows x batch_max f32 slab, and never touch the scores on the
                    // host at all. The audit arm below is the ONLY thing that restores
                    // the slab dtoh, which is why it is an instrument and not an arm.
                    let counts: Vec<usize> = (0..n)
                        .map(|i| (base_pos + scored_rows[done + i] + 1) / block_size)
                        .collect();
                    let picked =
                        launch_qsa_index_topk(e, &scores_dev, &counts, batch_max, budget_blocks)?;
                    if idx_sel_audit_on() {
                        let host = e.dtoh(&scores_dev)?;
                        let mut mismatched = 0u64;
                        let mut deepest = 0u64;
                        for i in 0..n {
                            let complete = counts[i];
                            let row_scores = &host[i * batch_max..i * batch_max + complete];
                            let twin = top_blocks_ascending(row_scores, budget_blocks, threads);
                            if twin != picked[i] {
                                mismatched += 1;
                            }
                            deepest = deepest.max(complete as u64);
                        }
                        IDX_SEL_AUDIT_ROWS
                            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                        IDX_SEL_AUDIT_MISMATCH
                            .fetch_add(mismatched, std::sync::atomic::Ordering::Relaxed);
                        IDX_SEL_AUDIT_MAX_BLOCKS
                            .fetch_max(deepest, std::sync::atomic::Ordering::Relaxed);
                        if mismatched > 0 {
                            return Err(format!(
                                "idxsel audit: {mismatched} of {n} device selections differ \
                                 from the host twin (ids or order) at fill {t_kv}"
                            )
                            .into());
                        }
                    }
                    for (i, blocks) in picked.into_iter().enumerate() {
                        sels[scored_rows[done + i]].blocks = blocks;
                    }
                } else {
                    let host = e.dtoh(&scores_dev)?;
                    for i in 0..n {
                        let qt = scored_rows[done + i];
                        let complete = (base_pos + qt + 1) / block_size;
                        let row_scores = &host[i * batch_max..i * batch_max + complete];
                        sels[qt].blocks = top_blocks_ascending(row_scores, budget_blocks, threads);
                    }
                }
                done += n;
            }
            for sel in &sels {
                if sel.visible == 0
                    || (!sel.full && sel.blocks.is_empty() && sel.visible % block_size == 0)
                {
                    return Err("indexer selection left a query with no visible source".into());
                }
            }
            return Ok(sels);
        }
    }
    let pooled_ref: &[f32] = pooled_keys;
    let select_row = |qt: usize, threads_in_row: usize| -> RowSel {
        let row = base_pos + qt;
        let position = row + pos_off;
        let visible = row + 1;
        let complete = visible / block_size;
        // Structural fast path (perf lane, semantic no-op): with complete <= budget the
        // top-k keeps EVERY complete block whatever the scores say, and the incomplete
        // tail is always visible — the row is the full causal prefix. Real geometry:
        // budget 512 x block 4 => every position < 2051 takes this path (SEMANTICS.md
        // §QSA); the scoring arm below stays the reference for long contexts and is
        // exercised by the tiny gate's budget-2 fixture at every position past 11.
        if complete <= budget_blocks {
            return RowSel {
                full: true,
                blocks: Vec::new(),
                visible,
            };
        }
        let mut query = proj_rows[qt * qk_width..qt * qk_width + heads * head_dim].to_vec();
        host_rms_norm(&mut query, head_dim, idx_q_norm, epsilon);
        host_rope_at(&mut query, head_dim, rope_dims, rope_base, yarn, position);
        let scores = score_blocks(
            &query,
            pooled_ref,
            heads,
            head_dim,
            complete,
            scale,
            threads_in_row,
        );
        let blocks = top_blocks_ascending(&scores, budget_blocks, threads_in_row);
        RowSel {
            full: false,
            blocks,
            visible,
        }
    };
    const ROW_PAR_MIN_WORK: usize = 1 << 16;
    let total_scored_blocks: usize = (0..t)
        .map(|qt| {
            let complete = (base_pos + qt + 1) / block_size;
            if complete <= budget_blocks {
                0
            } else {
                complete
            }
        })
        .sum();
    let sels: Vec<RowSel> = if t > 1 && threads > 1 && total_scored_blocks >= ROW_PAR_MIN_WORK {
        // Rows are independent: a work-stealing cursor over rows, each row sequential
        // inside (identical arithmetic to the sequential path).
        let cursor = std::sync::atomic::AtomicUsize::new(0);
        let mut out: Vec<Option<RowSel>> = (0..t).map(|_| None).collect();
        let slots = std::sync::Mutex::new(&mut out);
        std::thread::scope(|scope| {
            for _ in 0..threads.min(t) {
                scope.spawn(|| {
                    loop {
                        let qt = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if qt >= t {
                            break;
                        }
                        let sel = select_row(qt, 1);
                        slots.lock().unwrap()[qt] = Some(sel);
                    }
                });
            }
        });
        out.into_iter().map(|s| s.unwrap()).collect()
    } else {
        (0..t).map(|qt| select_row(qt, threads)).collect()
    };
    for sel in &sels {
        if sel.visible == 0 || (!sel.full && sel.blocks.is_empty() && sel.visible % block_size == 0)
        {
            return Err("indexer selection left a query with no visible source".into());
        }
    }
    Ok(sels)
}

/// Render row selections as the dense [t, t_kv] u8 mask the smem-bounded masked kernel
/// consumes — byte-identical to the historical `indexer_mask_rows` output.
fn rowsel_to_mask(sels: &[RowSel], block_size: usize, t_kv: usize) -> Vec<u8> {
    let t = sels.len();
    let mut mask = vec![0u8; t * t_kv];
    for (qt, sel) in sels.iter().enumerate() {
        let row = &mut mask[qt * t_kv..(qt + 1) * t_kv];
        if sel.full {
            for slot in row.iter_mut().take(sel.visible) {
                *slot = 1;
            }
            continue;
        }
        for &block in &sel.blocks {
            for offset in 0..block_size {
                row[block as usize * block_size + offset] = 1;
            }
        }
        let complete = sel.visible / block_size;
        for slot in row.iter_mut().take(sel.visible).skip(complete * block_size) {
            *slot = 1;
        }
    }
    mask
}

/// Render row selections as ASCENDING position lists for the block-list attention
/// kernel: flat i32 positions + per-row (offset, count) meta. Every row is bounded by
/// budget*block + (block-1) + ... <= 2052 positions on real geometry, so the kernel's
/// smem stays fixed whatever t_kv is.
fn rowsel_positions(sels: &[RowSel], block_size: usize) -> (Vec<i32>, Vec<i32>, usize) {
    let mut flat: Vec<i32> = Vec::new();
    let mut meta: Vec<i32> = Vec::with_capacity(sels.len() * 2);
    let mut max_count = 0usize;
    for sel in sels {
        let start = flat.len();
        if sel.full {
            flat.extend(0..sel.visible as i32);
        } else {
            for &block in &sel.blocks {
                let first = block as usize * block_size;
                flat.extend(first as i32..(first + block_size) as i32);
            }
            let complete = sel.visible / block_size;
            flat.extend((complete * block_size) as i32..sel.visible as i32);
        }
        let count = flat.len() - start;
        max_count = max_count.max(count);
        meta.push(start as i32);
        meta.push(count as i32);
    }
    (flat, meta, max_count)
}

/// One launch of the QSA indexer block scorer (`qsa_index_score_f32`): thread-per-block
/// over a [rows, n_blocks] slab. Per-score arithmetic is the host twin's verbatim (same
/// dim order, same relu-sum, same division by sqrt(head_dim)) — bit-identical scores.
#[allow(clippy::too_many_arguments)]
fn launch_qsa_index_score(
    e: &Engine,
    q: &CudaSlice<f32>,
    pooled: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    heads: usize,
    head_dim: usize,
    n_blocks: usize,
    rows: usize,
    scale: f32,
) -> Res<()> {
    if rows == 0 || n_blocks == 0 {
        return Ok(());
    }
    if out.len() < rows * n_blocks {
        return Err("qsa_index_score_f32: score slab too short".into());
    }
    if rows > 65535 {
        return Err("qsa_index_score_f32: rows exceed grid.y (caller sub-batches)".into());
    }
    // `poolT`: read the dim-major plane in the second half of the mirror (bit-identical twin —
    // see POOL_T_DEFAULT). The plane's pitch is the mirror's block CAPACITY, not `n_blocks`:
    // passing `n_blocks` would read dim d of block b as dim d of some other block for every
    // d > 0, which is silent wrong values, so the pitch is derived from the allocation.
    let cap_rows = pooled.len() / (head_dim * POOL_PLANES);
    let pool_t = pool_t_on();
    if pool_t && cap_rows < n_blocks {
        return Err("qsa_index_score_f32_t: pooled plane capacity below n_blocks".into());
    }
    let f = e.func(if pool_t {
        "qsa_index_score_f32_t"
    } else {
        "qsa_index_score_f32"
    });
    const TPB: usize = 128;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks.div_ceil(TPB) as u32, rows as u32, 1),
        block_dim: (TPB as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (h, hd, nb, r) = (heads as i32, head_dim as i32, n_blocks as i32, rows as i32);
    let pitch = cap_rows as i64;
    let stream = e.gpu.stream();
    if pool_t {
        // The plane starts at `cap_rows * head_dim`; the kernel indexes `pooled_t[d*pitch + b]`
        // from that base, so the slice is the plane region, not the whole buffer.
        let plane = pooled.slice(cap_rows * head_dim..cap_rows * head_dim * POOL_PLANES);
        let mut b = stream.launch_builder(&f);
        b.arg(q)
            .arg(&plane)
            .arg(&mut *out)
            .arg(&h)
            .arg(&hd)
            .arg(&nb)
            .arg(&r)
            .arg(&scale)
            .arg(&pitch);
        unsafe {
            b.launch(cfg)?;
        }
        return Ok(());
    }
    let mut b = stream.launch_builder(&f);
    b.arg(q)
        .arg(pooled)
        .arg(&mut *out)
        .arg(&h)
        .arg(&hd)
        .arg(&nb)
        .arg(&r)
        .arg(&scale);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// How many `cap_rows * head_dim` regions the pooled device mirror carries: the row-major
/// mirror, then the dim-major `poolT` plane. See the append site for why both are maintained
/// unconditionally (A/B isolation, and no stale-plane failure mode on a mid-run seam flip).
const POOL_PLANES: usize = 2;

/// Mirror the freshly-appended pooled rows `[r0, r0+rows)` into the dim-major plane. Pure data
/// movement inside one buffer; `cap_rows` is the plane pitch (the mirror's block capacity).
fn launch_qsa_pooled_transpose(
    e: &Engine,
    buf: &mut CudaSlice<f32>,
    r0: usize,
    rows: usize,
    head_dim: usize,
    cap_rows: usize,
) -> Res<()> {
    if rows == 0 {
        return Ok(());
    }
    if r0 + rows > cap_rows {
        return Err("qsa_pooled_transpose_f32: delta exceeds the plane capacity".into());
    }
    let f = e.func("qsa_pooled_transpose_f32");
    const TPB: usize = 128;
    let cfg = LaunchConfig {
        grid_dim: (rows.div_ceil(TPB) as u32, head_dim as u32, 1),
        block_dim: (TPB as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r, hd, r0i) = (rows as i32, head_dim as i32, r0 as i32);
    let cap = cap_rows as i64;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(buf).arg(&r).arg(&hd).arg(&r0i).arg(&cap);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One launch of the device indexer top-k (`qsa_index_topk_u32`) over a score slab, plus
/// the structural checks that make a silent mis-write loud: every row's block count must
/// EXCEED the budget (so the row genuinely needs a selection and every out slot is
/// written), and the returned lists come back strictly ascending and in range. Returns the
/// `rows x budget` block ids.
fn launch_qsa_index_topk(
    e: &Engine,
    scores: &CudaSlice<f32>,
    counts: &[usize],
    stride: usize,
    budget: usize,
) -> Res<Vec<Vec<u32>>> {
    let rows = counts.len();
    if rows == 0 || budget == 0 {
        return Ok(Vec::new());
    }
    if rows > 65535 {
        return Err("qsa_index_topk_u32: rows exceed grid.x (caller sub-batches)".into());
    }
    if scores.len() < rows * stride {
        return Err("qsa_index_topk_u32: score slab too short".into());
    }
    for (r, &c) in counts.iter().enumerate() {
        if c <= budget || c > stride {
            return Err(format!(
                "qsa_index_topk_u32: row {r} block count {c} outside (budget {budget}, \
                 stride {stride}] — the caller only routes scored rows here"
            )
            .into());
        }
    }
    let counts_i32: Vec<i32> = counts.iter().map(|&c| c as i32).collect();
    let counts_dev = e.htod_i32(&counts_i32)?;
    // -1 fill: an unwritten slot is then VISIBLE (the check below), not a plausible block.
    let mut out = e.htod_i32(&vec![-1i32; rows * budget])?;
    let f = e.func("qsa_index_topk_u32");
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (st, bu, ro) = (stride as i32, budget as i32, rows as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(scores)
        .arg(&counts_dev)
        .arg(&mut out)
        .arg(&st)
        .arg(&bu)
        .arg(&ro);
    unsafe {
        b.launch(cfg)?;
    }
    let host = e.gpu.stream().clone_dtoh(&out)?;
    e.gpu.stream().synchronize()?;
    let mut out_rows: Vec<Vec<u32>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &host[r * budget..(r + 1) * budget];
        let mut blocks: Vec<u32> = Vec::with_capacity(budget);
        let mut prev: i64 = -1;
        for (j, &v) in row.iter().enumerate() {
            if v < 0 || (v as usize) >= counts[r] || (v as i64) <= prev {
                return Err(format!(
                    "qsa_index_topk_u32: row {r} slot {j} = {v} is not a strictly ascending \
                     in-range block id (blocks {}, budget {budget})",
                    counts[r]
                )
                .into());
            }
            prev = v as i64;
            blocks.push(v as u32);
        }
        out_rows.push(blocks);
    }
    Ok(out_rows)
}

/// One launch of the row-window column-slice copy (`copy_rows_col_f32`): append the
/// k-part of `rows` idx_proj rows (column offset `src_col`, row stride `src_stride`)
/// to the device raw-key cache at row `dst_row`. Exact byte moves, no arithmetic.
#[allow(clippy::too_many_arguments)]
fn launch_copy_rows_col(
    e: &Engine,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    rows: usize,
    width: usize,
    src_stride: usize,
    src_col: usize,
    dst_row: usize,
) -> Res<()> {
    if rows == 0 {
        return Ok(());
    }
    if src.len() < (rows - 1) * src_stride + src_col + width || dst.len() < (dst_row + rows) * width
    {
        return Err("copy_rows_col_f32: window out of range".into());
    }
    let f = e.func("copy_rows_col_f32");
    let total = rows * width;
    let cfg = LaunchConfig::for_num_elems(total as u32);
    let (r, w) = (rows as i32, width as i32);
    let (ss, sc, dr) = (src_stride as i64, src_col as i64, dst_row as i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(src)
        .arg(&mut *dst)
        .arg(&r)
        .arg(&w)
        .arg(&ss)
        .arg(&sc)
        .arg(&dr);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One launch of the device MoE router (`qwen4exp_route_topk_f32`): per token row, the
/// full host_route_softmax_topk program on device (kernel doc — order-sensitive
/// reductions sequential on thread 0, host op order verbatim; exp through double).
/// `tok` = optional (slot->token map, tok_base) for the gufuse merged verify path.
/// Geometry guards live in the CALLER's engage condition; violations here are errors,
/// never silent fallbacks.
#[allow(clippy::too_many_arguments)]
fn launch_route_topk(
    e: &Engine,
    logits: &CudaSlice<f32>,
    sel: &mut CudaSlice<i32>,
    w: &mut CudaSlice<f32>,
    tok: Option<(&mut CudaSlice<i32>, usize)>,
    experts: usize,
    selected: usize,
    rows: usize,
) -> Res<()> {
    if rows == 0 {
        return Ok(());
    }
    if selected == 0 || selected > 32 || selected > experts {
        return Err("qwen4exp_route_topk_f32: selected out of range (caller guards)".into());
    }
    if experts % 2 != 0 {
        // The u64 key slab sits after the f32 weight slab in dynamic smem; an even
        // expert count keeps it 8-byte aligned (caller guards via route_dev_geometry).
        return Err("qwen4exp_route_topk_f32: odd expert count".into());
    }
    if logits.len() < rows * experts || sel.len() < rows * selected || w.len() < rows * selected {
        return Err("qwen4exp_route_topk_f32: buffer too short".into());
    }
    let smem = experts * 12; // f32 weights + u64 selection keys
    if smem > 48 * 1024 {
        return Err("qwen4exp_route_topk_f32: experts exceed the smem bound".into());
    }
    let stream = e.gpu.stream();
    let (tok_raw, tok_base) = match tok {
        Some((buf, base)) => {
            if buf.len() < rows * selected {
                return Err("qwen4exp_route_topk_f32: tok map too short".into());
            }
            (buf.device_ptr(&stream).0, base)
        }
        None => (0u64, 0usize),
    };
    let f = e.func("qwen4exp_route_topk_f32");
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (ex, se, ro, tb) = (
        experts as i32,
        selected as i32,
        rows as i32,
        tok_base as i32,
    );
    let floor = ROUTE_DENOM_FLOOR;
    let mut b = stream.launch_builder(&f);
    b.arg(logits)
        .arg(&mut *sel)
        .arg(&mut *w)
        .arg(&tok_raw)
        .arg(&ex)
        .arg(&se)
        .arg(&ro)
        .arg(&tb)
        .arg(&floor);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Device route + (MEMRA_Q4E_ROUTER_AUDIT=1) the host-twin cross-check over the SAME
/// logits: selection ids order-exact or Err; weights within ROUTE_AUDIT_ULP_BOUND ULP,
/// worst observed kept for the gate receipt (`route_audit_stats`).
#[allow(clippy::too_many_arguments)]
fn route_topk_device(
    e: &Engine,
    logits: &CudaSlice<f32>,
    sel: &mut CudaSlice<i32>,
    w: &mut CudaSlice<f32>,
    tok: Option<(&mut CudaSlice<i32>, usize)>,
    experts: usize,
    selected: usize,
    rows: usize,
    layer: u32,
) -> Res<()> {
    launch_route_topk(e, logits, sel, w, tok, experts, selected, rows)?;
    if !router_audit_on() {
        return Ok(());
    }
    let k = selected.min(experts);
    let lg = e.dtoh_view(&logits.slice(0..rows * experts))?;
    let sel_h = e.gpu.stream().clone_dtoh(&sel.slice(0..rows * selected))?;
    let w_h = e.gpu.stream().clone_dtoh(&w.slice(0..rows * selected))?;
    // Emit the shared-format route trace off THIS readback (`trace_moe_routes`, the frozen
    // `memra-ep-map-v1` producer). Its own doc comment already promised exactly this — "arming
    // MEMRA_Q4E_ROUTER_AUDIT=1 restores a host recompute of every device route and the trace
    // rides THAT readback at zero new syncs ... single-card batteries trace with the audit
    // armed" — but the call was never made, so the tracer fired ONLY from the TP2 paths and the
    // shipped single-card device-routed default emitted nothing at all. The box's traces
    // directory was empty for that reason and not for lack of running, and the expert-placement
    // lane's only input silently did not exist. Prose describing a wiring that is not there is
    // the failure class this lane has hit twice; the wiring is here now.
    //
    // Traced from the DEVICE arrays, not from the host twin below: the trace must record the
    // route that actually ran. The audit's job is to prove the two agree, and it does that on
    // the next lines — so if they ever disagree this call has already errored out.
    {
        let routes: Vec<Vec<(usize, f32)>> = (0..rows)
            .map(|row| {
                (0..selected)
                    .map(|j| {
                        (
                            sel_h[row * selected + j].max(0) as usize,
                            w_h[row * selected + j],
                        )
                    })
                    .collect()
            })
            .collect();
        trace_moe_routes(layer, rows, &routes);
    }
    let mut worst: u32 = 0;
    for row in 0..rows {
        let twin = host_route_softmax_topk(&lg[row * experts..(row + 1) * experts], selected);
        if twin.len() != k {
            return Err("router audit: host twin emitted an unexpected selection width".into());
        }
        for (j, &(idx, wt)) in twin.iter().enumerate() {
            let ds = sel_h[row * selected + j];
            let dw = w_h[row * selected + j];
            if ds != idx as i32 {
                return Err(format!(
                    "router audit: selection mismatch at row {row} slot {j}: \
                     device {ds} vs host {idx} (host w {wt:e})"
                )
                .into());
            }
            let ulp = (dw.to_bits() as i64 - wt.to_bits() as i64).unsigned_abs();
            let ulp = u32::try_from(ulp).unwrap_or(u32::MAX);
            worst = worst.max(ulp);
            if ulp > ROUTE_AUDIT_ULP_BOUND {
                return Err(format!(
                    "router audit: weight ULP {ulp} > bound {ROUTE_AUDIT_ULP_BOUND} at \
                     row {row} slot {j}: device {dw:e} vs host {wt:e}"
                )
                .into());
            }
        }
    }
    ROUTE_AUDIT_ROWS.fetch_add(rows as u64, std::sync::atomic::Ordering::Relaxed);
    ROUTE_AUDIT_MAX_ULP.fetch_max(worst, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// The historical mask-producing entry point, now select + render (byte-identical mask;
/// the TP2 decode path and the masked-kernel arm consume it).
// dead_code: bring-up scaffolding the in-flight qwen4exp lanes still call; not deleted in
// the clippy-zero lane (bit-neutral by construction).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn indexer_mask_rows(
    overlay: &MicroBlockIndexPlan,
    rope_base: f32,
    yarn: Option<(&[f32], f32)>,
    epsilon: f32,
    idx_q_norm: &[f32],
    idx_k_norm: &[f32],
    proj_rows: &[f32],
    raw_keys: &IdxRawCache,
    pooled_keys: &mut Vec<f32>,
    base_pos: usize,
    t: usize,
    t_kv: usize,
    pos_off: usize,
) -> Res<Vec<u8>> {
    let sels = indexer_select_rows(
        overlay,
        rope_base,
        yarn,
        epsilon,
        idx_q_norm,
        idx_k_norm,
        proj_rows,
        raw_keys,
        pooled_keys,
        // TP2 decode + the reference/mask arm keep the host scorer (TP2's selection runs
        // on card 0's projection and feeds both halves; its depths are decode-class).
        None,
        base_pos,
        t,
        t_kv,
        pos_off,
    )?;
    Ok(rowsel_to_mask(&sels, overlay.block_size as usize, t_kv))
}

/// memra_reference `shift_right_ignore_eos` twin.
fn shift_right_ignore_eos(history: &[i64], shift: usize, eos: i64) -> Vec<i64> {
    if shift == 0 {
        return history.to_vec();
    }
    let mut last_eos_inclusive: i64 = -1;
    let mut output = Vec::with_capacity(history.len());
    for (position, &token) in history.iter().enumerate() {
        let previous_eos = last_eos_inclusive;
        if token == eos {
            last_eos_inclusive = position as i64;
        }
        let segment_start = previous_eos + 1;
        let position_in_segment = position as i64 - segment_start;
        let source = position as i64 - shift as i64;
        let valid = position_in_segment >= shift as i64 && source >= 0;
        output.push(if valid { history[source as usize] } else { eos });
    }
    output
}

/// INCREMENTAL twin of `host_ngram_ids` (`plecache` seam, 262k perf lane): extend a cached
/// id vector to cover `token_ids` instead of rebuilding it. Returns the last `t` rows'
/// worth of ids, i.e. exactly what the caller slices.
///
/// Bit-identical to `host_ngram_ids` by construction, not by tolerance. Two local facts do
/// it. (1) `shift_right_ignore_eos` at position p emits `history[p - shift]` guarded by an
/// eos scan that only moves left-to-right, so its value at p depends on `history[..=p]`
/// alone. (2) the id loop at `token` reads only `shifted[*][context + token]`. Therefore
/// `ids[token]` is a pure function of `token_ids[..=token]` and never changes when a token
/// is appended — so appending rows is not an approximation of rebuilding them, it is the
/// same arithmetic in the same order on the same inputs.
///
/// A shrinking or diverging history (spec reject / rewind / a fresh sequence in a reused
/// state) is handled by TRUNCATING the cache to the longest common prefix and re-extending.
/// The check is a real prefix compare rather than a length compare, because a length-only
/// check would silently keep another sequence's hashes — the failure mode would be fluent
/// output from the wrong n-gram rows, which is invisible.
#[allow(clippy::too_many_arguments)]
fn host_ngram_ids_cached(
    cache_ids: &mut Vec<i64>,
    cache_history: &mut Vec<i64>,
    cache_last_eos: &mut i64,
    token_ids: &[u32],
    multipliers: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    max_ngram: usize,
    heads_per_ngram: usize,
    eos_token_id: u32,
) {
    let context = max_ngram - 1;
    let eos = eos_token_id as i64;
    let total_heads = (max_ngram - 1) * heads_per_ngram;
    if cache_history.is_empty() {
        cache_history.extend(std::iter::repeat_n(eos, context));
        *cache_last_eos = context as i64 - 1; // every prefix row IS an eos
        cache_ids.clear();
    }
    let cached_tokens = (cache_history.len() - context).min(cache_ids.len() / total_heads);
    // Longest common prefix of the cached tokens and the requested ones.
    let mut keep = cached_tokens.min(token_ids.len());
    for i in 0..keep {
        if cache_history[context + i] != token_ids[i] as i64 {
            keep = i;
            break;
        }
    }
    if keep < cached_tokens {
        // Rewind: drop the diverged tail and rebuild the eos scan over what survives.
        cache_history.truncate(context + keep);
        cache_ids.truncate(keep * total_heads);
        *cache_last_eos = cache_history
            .iter()
            .rposition(|&v| v == eos)
            .map(|p| p as i64)
            .unwrap_or(-1);
    }
    for &token in &token_ids[keep..] {
        let position = cache_history.len();
        let value = token as i64;
        cache_history.push(value);
        // `shift_right_ignore_eos`: `previous_eos` is read BEFORE this position updates it.
        let previous_eos = *cache_last_eos;
        if value == eos {
            *cache_last_eos = position as i64;
        }
        let segment_start = previous_eos + 1;
        let position_in_segment = position as i64 - segment_start;
        let shifted_at = |shift: usize| -> i64 {
            if shift == 0 {
                return cache_history[position];
            }
            let source = position as i64 - shift as i64;
            if position_in_segment >= shift as i64 && source >= 0 {
                cache_history[source as usize]
            } else {
                eos
            }
        };
        // Same op order as the twin: shift 0 multiply, then xor the higher shifts in order.
        let mut row = vec![0i64; total_heads];
        for ngram in 2..=max_ngram {
            let head_start = (ngram - 2) * heads_per_ngram;
            let mut mixed = shifted_at(0).wrapping_mul(multipliers[0]);
            for shift in 1..ngram {
                mixed ^= shifted_at(shift).wrapping_mul(multipliers[shift]);
            }
            for head in 0..heads_per_ngram {
                let index = head_start + head;
                row[index] = mixed.rem_euclid(sizes[index]) + offsets[index];
            }
        }
        cache_ids.extend_from_slice(&row);
    }
    debug_assert_eq!(cache_ids.len(), token_ids.len() * total_heads);
    // Returns nothing on purpose: the caller reads the tail of `cache_ids` in place. Handing
    // back a `Vec` would clone the whole history's ids on every decode step (19 MB at a
    // 150,000-token fill), which is the O(context) cost this seam exists to delete.
}

/// memra_reference `ngram_ids` twin over the FULL token history (context EOS rows
/// prepended); the caller slices the last `t` rows for the current chunk.
fn host_ngram_ids(
    token_ids: &[u32],
    multipliers: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    max_ngram: usize,
    heads_per_ngram: usize,
    eos_token_id: u32,
) -> Vec<i64> {
    let context = max_ngram - 1;
    let eos = eos_token_id as i64;
    let total_heads = (max_ngram - 1) * heads_per_ngram;
    let mut history = Vec::with_capacity(context + token_ids.len());
    history.extend(std::iter::repeat_n(eos, context));
    history.extend(token_ids.iter().map(|&token| token as i64));
    let shifted: Vec<Vec<i64>> = (0..max_ngram)
        .map(|shift| shift_right_ignore_eos(&history, shift, eos))
        .collect();
    let tokens = token_ids.len();
    let mut ids = vec![0i64; tokens * total_heads];
    for ngram in 2..=max_ngram {
        let head_start = (ngram - 2) * heads_per_ngram;
        for token in 0..tokens {
            let position = context + token;
            let mut mixed = shifted[0][position].wrapping_mul(multipliers[0]);
            for (shift, row) in shifted.iter().enumerate().take(ngram).skip(1) {
                mixed ^= row[position].wrapping_mul(multipliers[shift]);
            }
            for head in 0..heads_per_ngram {
                let index = head_start + head;
                ids[token * total_heads + index] = mixed.rem_euclid(sizes[index]) + offsets[index];
            }
        }
    }
    ids
}

// ---------------------------------------------------------------- kernel launchers

#[allow(clippy::too_many_arguments)]
fn launch_sdpa_mask(
    e: &Engine,
    q: &CudaSlice<f32>,
    k: &CudaView<'_, f32>,
    v: &CudaView<'_, f32>,
    o: &mut CudaSlice<f32>,
    mask: &CudaSlice<u8>,
    head_dim: usize,
    n_head: usize,
    n_head_kv: usize,
    t: usize,
    t_kv: usize,
    scale: f32,
) -> Res<()> {
    if t_kv * 4 > 48 * 1024 {
        return Err(
            "sdpa_naive_mask_f32: T_kv exceeds the smem bound; the gmem twin is perf-lane work"
                .into(),
        );
    }
    let f = e.func("sdpa_naive_mask_f32");
    let cfg = LaunchConfig {
        grid_dim: (n_head as u32, t as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: (t_kv * 4) as u32,
    };
    let (hd, nh, nkv, ti, tkvi) = (
        head_dim as i32,
        n_head as i32,
        n_head_kv as i32,
        t as i32,
        t_kv as i32,
    );
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(q)
        .arg(k)
        .arg(v)
        .arg(o)
        .arg(mask)
        .arg(&hd)
        .arg(&nh)
        .arg(&nkv)
        .arg(&ti)
        .arg(&tkvi)
        .arg(&scale);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Block-list QSA attention (long-context form): per query row, attend the row's own
/// ASCENDING position list (`rowsel_positions`) — smem scales with the bounded per-row
/// selection (<= 2052 on real geometry), never with t_kv. BIT-IDENTICAL to
/// `sdpa_naive_mask_f32` on the same selection: masked entries there contribute exact
/// 0.0 softmax/V terms in the same ascending order (gate arm + kernel oracle).
#[allow(clippy::too_many_arguments)]
fn launch_sdpa_blocklist(
    e: &Engine,
    q: &CudaSlice<f32>,
    k: &CudaView<'_, f32>,
    v: &CudaView<'_, f32>,
    o: &mut CudaSlice<f32>,
    pos: &CudaSlice<i32>,
    meta: &CudaSlice<i32>,
    head_dim: usize,
    n_head: usize,
    n_head_kv: usize,
    t: usize,
    max_count: usize,
    scale: f32,
) -> Res<()> {
    // positions (i32) + scores (f32) per selected entry. Production rows are bounded by
    // budget*block + block = 2052 entries (16.4 KB); 48 KB is the no-attribute smem cap.
    let smem = (max_count * 8) as u32;
    if smem > 48 * 1024 {
        return Err("sdpa_blocklist_f32: selection exceeds the smem budget".into());
    }
    let f = e.func("sdpa_blocklist_f32");
    let cfg = LaunchConfig {
        grid_dim: (n_head as u32, t as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: smem,
    };
    let (hd, nh, nkv, ti, mc) = (
        head_dim as i32,
        n_head as i32,
        n_head_kv as i32,
        t as i32,
        max_count as i32,
    );
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(q)
        .arg(k)
        .arg(v)
        .arg(o)
        .arg(pos)
        .arg(meta)
        .arg(&hd)
        .arg(&nh)
        .arg(&nkv)
        .arg(&ti)
        .arg(&mc)
        .arg(&scale);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Append-quantize `t` post-RoPE K/V rows into the byte caches at slots
/// [base_pos, base_pos + t) (kvq lane; K=q8_0, V=q5_1).
#[allow(clippy::too_many_arguments)]
fn launch_q4e_kv_append(
    e: &Engine,
    k_rows: &CudaSlice<f32>,
    v_rows: &CudaSlice<f32>,
    k: &mut CudaSlice<u8>,
    v: &mut CudaSlice<u8>,
    base_pos: usize,
    t: usize,
    kv_dim: usize,
) -> Res<()> {
    let f = e.func("q4e_kv_append_q8q5_rows");
    let blocks = kv_dim.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, t as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (t0, dk, dv) = (base_pos as i32, kv_dim as i32, kv_dim as i32);
    let (ktb, vtb) = (q8_row_bytes(kv_dim) as i64, q5_row_bytes(kv_dim) as i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(k_rows)
        .arg(v_rows)
        .arg(k)
        .arg(v)
        .arg(&t0)
        .arg(&dk)
        .arg(&dv)
        .arg(&ktb)
        .arg(&vtb);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Dequant cache rows [r0, r0+rows) into f32 buffers (gates + TP2 migration seam).
#[allow(clippy::too_many_arguments)]
fn launch_q4e_kv_dequant_rows(
    e: &Engine,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    k_out: &mut CudaSlice<f32>,
    v_out: &mut CudaSlice<f32>,
    r0: usize,
    rows: usize,
    kv_dim: usize,
) -> Res<()> {
    let f = e.func("q4e_kv_dequant_rows");
    let blocks = kv_dim.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, rows as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r0i, dk, dv) = (r0 as i32, kv_dim as i32, kv_dim as i32);
    let (ktb, vtb) = (q8_row_bytes(kv_dim) as i64, q5_row_bytes(kv_dim) as i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(k)
        .arg(v)
        .arg(k_out)
        .arg(v_out)
        .arg(&r0i)
        .arg(&dk)
        .arg(&dv)
        .arg(&ktb)
        .arg(&vtb);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Block-list QSA attention over the QUANTIZED cache (kvq lane) — the f32 launcher's
/// twin with byte-cache K/V and their row strides.
#[allow(clippy::too_many_arguments)]
fn launch_q4e_sdpa_blocklist_q8q5(
    e: &Engine,
    q: &CudaSlice<f32>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<f32>,
    pos: &CudaSlice<i32>,
    meta: &CudaSlice<i32>,
    head_dim: usize,
    n_head: usize,
    n_head_kv: usize,
    t: usize,
    max_count: usize,
    scale: f32,
) -> Res<()> {
    let smem = (max_count * 8) as u32;
    if smem > 48 * 1024 {
        return Err("q4e_sdpa_blocklist_q8q5: selection exceeds the smem budget".into());
    }
    // `kvhoist`: the scale-hoisted twin, bit-identical, selected by seam (see KV_HOIST_DEFAULT).
    let f = e.func(if kv_hoist_on() {
        "q4e_sdpa_blocklist_q8q5_hoist"
    } else {
        "q4e_sdpa_blocklist_q8q5"
    });
    let cfg = LaunchConfig {
        grid_dim: (n_head as u32, t as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: smem,
    };
    let kv_dim = n_head_kv * head_dim;
    let (hd, nh, nkv, ti, mc) = (
        head_dim as i32,
        n_head as i32,
        n_head_kv as i32,
        t as i32,
        max_count as i32,
    );
    let (ktb, vtb) = (q8_row_bytes(kv_dim) as i64, q5_row_bytes(kv_dim) as i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(q)
        .arg(k)
        .arg(v)
        .arg(o)
        .arg(pos)
        .arg(meta)
        .arg(&hd)
        .arg(&nh)
        .arg(&nkv)
        .arg(&ti)
        .arg(&mc)
        .arg(&scale)
        .arg(&ktb)
        .arg(&vtb);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Quantize-append the k-part columns of `rows` idx_proj rows into the q8_0 device
/// raw-key cache (idxq=q8 x idxcache).
#[allow(clippy::too_many_arguments)]
fn launch_q4e_idx_append_q8(
    e: &Engine,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<u8>,
    rows: usize,
    width: usize,
    src_stride: usize,
    src_col: usize,
    dst_row: usize,
) -> Res<()> {
    let f = e.func("q4e_idx_append_q8");
    let cfg = LaunchConfig {
        grid_dim: (width.div_ceil(32) as u32, rows as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r, w) = (rows as i32, width as i32);
    let (ss, sc, dr) = (src_stride as i64, src_col as i64, dst_row as i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(src)
        .arg(dst)
        .arg(&r)
        .arg(&w)
        .arg(&ss)
        .arg(&sc)
        .arg(&dr);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Convert-append (bf16 RNE) the k-part columns into the bf16 device raw-key cache.
#[allow(clippy::too_many_arguments)]
fn launch_q4e_idx_append_bf16(
    e: &Engine,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<u16>,
    rows: usize,
    width: usize,
    src_stride: usize,
    src_col: usize,
    dst_row: usize,
) -> Res<()> {
    let f = e.func("q4e_idx_append_bf16");
    let total = rows * width;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r, w) = (rows as i32, width as i32);
    let (ss, sc, dr) = (src_stride as i64, src_col as i64, dst_row as i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(src)
        .arg(dst)
        .arg(&r)
        .arg(&w)
        .arg(&ss)
        .arg(&sc)
        .arg(&dr);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_gdn_scan(
    e: &Engine,
    qkv: &CudaSlice<f32>,
    g_log: &CudaSlice<f32>,
    beta_raw: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    t: usize,
    scale: f32,
    eps: f32,
) -> Res<()> {
    if hk > 128 {
        return Err("gdn_scan_naive_f32: hk > 128".into());
    }
    let f = e.func("gdn_scan_naive_f32");
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, 1, 1),
        block_dim: (hv as u32, 1, 1),
        shared_mem_bytes: ((2 * hk + 2) * 4) as u32,
    };
    let (nki, nvi, hki, hvi, ti) = (nk as i32, nv as i32, hk as i32, hv as i32, t as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(qkv)
        .arg(g_log)
        .arg(beta_raw)
        .arg(state)
        .arg(o)
        .arg(&nki)
        .arg(&nvi)
        .arg(&hki)
        .arg(&hvi)
        .arg(&ti)
        .arg(&scale)
        .arg(&eps);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One launch of the decode-step scan twin (`gdn_scan_step_f32`, t == 1): grid
/// (nv, hv), block hk — one state element per thread (see the kernel doc; the
/// accumulation class vs the naive kernel's sequential row sums).
#[allow(clippy::too_many_arguments)]
fn launch_gdn_scan_step(
    e: &Engine,
    qkv: &CudaSlice<f32>,
    g_log: &CudaSlice<f32>,
    beta_raw: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    scale: f32,
    eps: f32,
) -> Res<()> {
    if hk % 32 != 0 || hk > 1024 {
        return Err("gdn_scan_step_f32: hk must be a multiple of 32 and <= 1024".into());
    }
    let f = e.func("gdn_scan_step_f32");
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, hv as u32, 1),
        block_dim: (hk as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nki, nvi, hki, hvi) = (nk as i32, nv as i32, hk as i32, hv as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(qkv)
        .arg(g_log)
        .arg(beta_raw)
        .arg(state)
        .arg(o)
        .arg(&nki)
        .arg(&nvi)
        .arg(&hki)
        .arg(&hvi)
        .arg(&scale)
        .arg(&eps);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Per-token step-scan launch at COLUMN `tok` of a chunk (verify-exact rows): views of
/// the token's post-conv row / g_log / beta / output row, the SAME kernel and grid as
/// the decode step — each column is bit-identical to the t == 1 decode launch.
#[allow(clippy::too_many_arguments)]
fn launch_gdn_scan_step_at(
    e: &Engine,
    conv_out: &CudaSlice<f32>,
    g_log: &CudaSlice<f32>,
    beta_raw: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    tok: usize,
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    scale: f32,
    eps: f32,
) -> Res<()> {
    if hk % 32 != 0 || hk > 1024 {
        return Err("gdn_scan_step_f32: hk must be a multiple of 32 and <= 1024".into());
    }
    let conv_dim = 2 * nk * hk + nv * hv;
    let qv = conv_out.slice(tok * conv_dim..(tok + 1) * conv_dim);
    let gv = g_log.slice(tok * nv..(tok + 1) * nv);
    let bv = beta_raw.slice(tok * nv..(tok + 1) * nv);
    let mut ov = o.slice_mut(tok * nv * hv..(tok + 1) * nv * hv);
    let f = e.func("gdn_scan_step_f32");
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, hv as u32, 1),
        block_dim: (hk as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nki, nvi, hki, hvi) = (nk as i32, nv as i32, hk as i32, hv as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(&qv)
        .arg(&gv)
        .arg(&bv)
        .arg(&mut *state)
        .arg(&mut ov)
        .arg(&nki)
        .arg(&nvi)
        .arg(&hki)
        .arg(&hvi)
        .arg(&scale)
        .arg(&eps);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Per-token NAIVE-scan launch at column `tok` (t == 1 views) — the exact-verify twin
/// for geometries the step kernel refuses (tiny hk): identical to the t == 1 decode
/// dispatch on those plans.
#[allow(clippy::too_many_arguments)]
fn launch_gdn_scan_at(
    e: &Engine,
    conv_out: &CudaSlice<f32>,
    g_log: &CudaSlice<f32>,
    beta_raw: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    tok: usize,
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    scale: f32,
    eps: f32,
) -> Res<()> {
    if hk > 128 {
        return Err("gdn_scan_naive_f32: hk > 128".into());
    }
    let conv_dim = 2 * nk * hk + nv * hv;
    let qv = conv_out.slice(tok * conv_dim..(tok + 1) * conv_dim);
    let gv = g_log.slice(tok * nv..(tok + 1) * nv);
    let bv = beta_raw.slice(tok * nv..(tok + 1) * nv);
    let mut ov = o.slice_mut(tok * nv * hv..(tok + 1) * nv * hv);
    let f = e.func("gdn_scan_naive_f32");
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, 1, 1),
        block_dim: (hv as u32, 1, 1),
        shared_mem_bytes: ((2 * hk + 2) * 4) as u32,
    };
    let (nki, nvi, hki, hvi, ti) = (nk as i32, nv as i32, hk as i32, hv as i32, 1i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(&qv)
        .arg(&gv)
        .arg(&bv)
        .arg(&mut *state)
        .arg(&mut ov)
        .arg(&nki)
        .arg(&nvi)
        .arg(&hki)
        .arg(&hvi)
        .arg(&ti)
        .arg(&scale)
        .arg(&eps);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One launch of the fused GDN norm+gate (`rms_sigmul_f32`): dst = rms_norm(x, w) *
/// sigmoid(z) over `nrows` rows of `ncols` — bit-identical to the rms_norm + sigmoid +
/// mul chain (kernel doc).
#[allow(clippy::too_many_arguments)]
fn launch_rms_sigmul(
    e: &Engine,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    z: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    ncols: usize,
    nrows: usize,
    eps: f32,
) -> Res<()> {
    let f = e.func("rms_sigmul_f32");
    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (crate::rms_block(), 1, 1),
        shared_mem_bytes: 0,
    };
    let (nc, ep) = (ncols as i32, eps);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(x).arg(w).arg(z).arg(dst).arg(&nc).arg(&ep);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_dwconv(
    e: &Engine,
    x: &CudaSlice<f32>,
    hist: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    t: usize,
    th: usize,
    c: usize,
    k: usize,
    dilation: usize,
    mode: i32,
) -> Res<()> {
    let f = e.func("dwconv_causal_f32");
    let cfg = LaunchConfig::for_num_elems((t * c) as u32);
    let (ti, thi, ci, ki, di) = (t as i32, th as i32, c as i32, k as i32, dilation as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(x)
        .arg(hist)
        .arg(w)
        .arg(y)
        .arg(&ti)
        .arg(&thi)
        .arg(&ci)
        .arg(&ki)
        .arg(&di)
        .arg(&mode);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One routed expert's SwiGLU: gate/up GEMMs on the gathered token rows, silu_mul, down.
#[allow(clippy::too_many_arguments)]
fn run_routed_expert(
    e: &Engine,
    xg: &CudaSlice<f32>,
    gate: &CudaView<'_, f32>,
    up: &CudaView<'_, f32>,
    down: &CudaView<'_, f32>,
    m_e: usize,
    hidden: usize,
    ff: usize,
) -> Res<CudaSlice<f32>> {
    let xg_view = xg.slice(0..m_e * hidden);
    let mut gate_out = e.uninit(m_e * ff)?;
    e.linear_device_into(&xg_view, gate, &mut gate_out, m_e, hidden, ff)?;
    let mut up_out = e.uninit(m_e * ff)?;
    e.linear_device_into(&xg_view, up, &mut up_out, m_e, hidden, ff)?;
    let mut act = e.uninit(m_e * ff)?;
    e.silu_mul(&gate_out, &up_out, &mut act, m_e * ff)?;
    let mut down_out = e.uninit(m_e * hidden)?;
    e.linear_device_into(
        &act.slice(0..m_e * ff),
        down,
        &mut down_out,
        m_e,
        ff,
        hidden,
    )?;
    Ok(down_out)
}

/// View-destination twin of `Engine::rms_norm` — same kernel, same block size, same args,
/// so BIT-IDENTICAL; it exists only so the gate can normalize into one contiguous
/// stream-major buffer instead of `streams` separate allocations (the fused gate kernels
/// need every stream in one launch). PDL is skipped: dependent launch changes scheduling,
/// not arithmetic.
fn launch_rms_norm_into_view(
    e: &Engine,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    dst: &mut cudarc::driver::CudaViewMut<'_, f32>,
    ncols: usize,
    nrows: usize,
    eps: f32,
) -> Res<()> {
    let kname = if Engine::norm_ilp_on() {
        "rms_norm_f32_v2"
    } else {
        "rms_norm_f32"
    };
    let f = e.func(kname);
    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (crate::rms_block(), 1, 1),
        shared_mem_bytes: 0,
    };
    let (nc, ep) = (ncols as i32, eps);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(x).arg(w).arg(dst).arg(&nc).arg(&ep);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// `hc_lowrank_reduce_f32`: low_act[t, rank] = silu(inv_streams · Σ_s parts[s, t, rank]).
fn launch_hc_lowrank_reduce(
    e: &Engine,
    parts: &CudaSlice<f32>,
    low_act: &mut CudaSlice<f32>,
    streams: usize,
    t: usize,
    rank: usize,
) -> Res<()> {
    let f = e.func("hc_lowrank_reduce_f32");
    let cfg = LaunchConfig::for_num_elems((t * rank) as u32);
    let (si, ti, ri) = (streams as i32, t as i32, rank as i32);
    let inv = 1.0f32 / streams as f32;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(parts)
        .arg(low_act)
        .arg(&si)
        .arg(&ti)
        .arg(&ri)
        .arg(&inv);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// `hc_mix_epilogue_f32`: mixed = inv_streams · Σ_s sigmoid(gates_s) ⊙ normed_s.
fn launch_hc_mix_epilogue(
    e: &Engine,
    gates: &CudaSlice<f32>,
    normed: &CudaSlice<f32>,
    mixed: &mut CudaSlice<f32>,
    streams: usize,
    t: usize,
    hidden: usize,
) -> Res<()> {
    let f = e.func("hc_mix_epilogue_f32");
    let cfg = LaunchConfig::for_num_elems((t * hidden) as u32);
    let (si, ti, hi) = (streams as i32, t as i32, hidden as i32);
    let inv = 1.0f32 / streams as f32;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(gates)
        .arg(normed)
        .arg(mixed)
        .arg(&si)
        .arg(&ti)
        .arg(&hi)
        .arg(&inv);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// `hc_inject_gates_f32`: out[s, t] = 2·sigmoid(inv_streams · ⟨w_s, wide_normed_t⟩).
fn launch_hc_inject_gates(
    e: &Engine,
    normed: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    streams: usize,
    t: usize,
    hidden: usize,
) -> Res<()> {
    let f = e.func("hc_inject_gates_f32");
    let cfg = LaunchConfig {
        grid_dim: (streams as u32, t as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (si, ti, hi) = (streams as i32, t as i32, hidden as i32);
    let inv = 1.0f32 / streams as f32;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(normed)
        .arg(w)
        .arg(out)
        .arg(&si)
        .arg(&ti)
        .arg(&hi)
        .arg(&inv);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Inject scalars as either per-stream rows (the item-1-era plumbing, hcmicro OFF and
/// the legacy gate) or the [streams, t] slab straight out of the two-stage inject
/// (hcmicro ON — no per-stream d2d copies; `gate_write` consumes it in one launch).
enum InjectOut {
    Rows(Vec<CudaSlice<f32>>),
    Slab(CudaSlice<f32>),
}

/// Park an inject result back into its slots (the form is flag-determined, so takes and
/// puts pair up step over step).
fn put_inject(ws: &mut StepPool, inject: InjectOut) {
    match inject {
        InjectOut::Rows(rows) => {
            for (s, row) in rows.into_iter().enumerate() {
                ws.put_f32(INJECT_SLOTS[s], row);
            }
        }
        InjectOut::Slab(slab) => ws.put_f32("hc.inj_all", slab),
    }
}

/// Take the parked inject scalars in the form the current seams produce (graph driver's
/// MoE tail — the mlp read gate parked them in the interior segment).
fn take_inject(e: &Engine, ws: &mut StepPool, streams: usize, t: usize) -> Res<InjectOut> {
    // The diet emits the Slab form and requires micro_inj at dispatch, so this predicate
    // stays in lockstep with what gate_read parked.
    if micro_inj_on() && hc_fused_gate_on() {
        Ok(InjectOut::Slab(ws.take_f32(
            e,
            "hc.inj_all",
            streams * t,
            0,
        )?))
    } else {
        let mut rows = Vec::with_capacity(streams);
        for s in 0..streams {
            rows.push(ws.take_f32(e, INJECT_SLOTS[s], t, 0)?);
        }
        Ok(InjectOut::Rows(rows))
    }
}

/// `hc_norm_planes_f32`: per-(stream, token) RMSNorm over the plane pointer table into
/// the stream-major normed slab — one launch for all streams (hcmicro seam).
#[allow(clippy::too_many_arguments)]
fn launch_hc_norm_planes(
    e: &Engine,
    ptrs: &CudaSlice<u64>,
    w_stack: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    hidden: usize,
    t: usize,
    streams: usize,
    eps: f32,
) -> Res<()> {
    let f = e.func("hc_norm_planes_f32");
    let cfg = LaunchConfig {
        grid_dim: (t as u32, streams as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (hi, ti) = (hidden as i32, t as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(ptrs)
        .arg(w_stack)
        .arg(dst)
        .arg(&hi)
        .arg(&ti)
        .arg(&eps);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Two-stage inject (hcmicro seam): chunked partial dots (fills the card; the
/// single-stage kernel ran `streams` blocks) then a sequential-order reduce + sigmoid.
/// Deterministic — no atomics (greedy replays must stay byte-stable).
#[allow(clippy::too_many_arguments)]
fn launch_hc_inject_two_stage(
    e: &Engine,
    normed: &CudaSlice<f32>,
    w_f32: &CudaSlice<f32>,
    w_b16: Option<&CudaSlice<u8>>,
    partials: &mut CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    streams: usize,
    t: usize,
    hidden: usize,
    chunks: usize,
) -> Res<()> {
    let cfg = LaunchConfig {
        grid_dim: (streams as u32, t as u32, chunks as u32),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (si, ti, hi, ci) = (streams as i32, t as i32, hidden as i32, chunks as i32);
    let stream = e.gpu.stream();
    if let Some(w) = w_b16 {
        let f = e.func("hc_inject_partials_bf16w_f32");
        let mut b = stream.launch_builder(&f);
        b.arg(normed)
            .arg(w)
            .arg(&mut *partials)
            .arg(&si)
            .arg(&ti)
            .arg(&hi)
            .arg(&ci);
        unsafe {
            b.launch(cfg)?;
        }
    } else {
        let f = e.func("hc_inject_partials_f32");
        let mut b = stream.launch_builder(&f);
        b.arg(normed)
            .arg(w_f32)
            .arg(&mut *partials)
            .arg(&si)
            .arg(&ti)
            .arg(&hi)
            .arg(&ci);
        unsafe {
            b.launch(cfg)?;
        }
    }
    let rows = (streams * t) as i32;
    let inv = 1.0f32 / streams as f32;
    let f = e.func("hc_inject_reduce_f32");
    let cfg = LaunchConfig::for_num_elems((streams * t) as u32);
    let mut b = stream.launch_builder(&f);
    b.arg(&*partials).arg(out).arg(&rows).arg(&ci).arg(&inv);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// hc-diet stage 1 (`hc_diet_stage1_f32`): per (row-chunk, stream) block — RMS recompute
/// from the raw plane, normed row in smem, this chunk's down rows + inject partial rows.
/// Emits parts [S, rank], inj_parts [n_inj, S], inv [S].
#[allow(clippy::too_many_arguments)]
fn launch_hc_diet_stage1(
    e: &Engine,
    ptrs: &CudaSlice<u64>,
    nw_stack: &CudaSlice<f32>,
    wdown_b16: &CudaSlice<u8>,
    winj_b16: Option<&CudaSlice<u8>>,
    parts: &mut CudaSlice<f32>,
    inj_parts: &mut CudaSlice<f32>,
    inv_out: &mut CudaSlice<f32>,
    hidden: usize,
    rank: usize,
    streams: usize,
    t: usize,
    eps: f32,
) -> Res<()> {
    if hidden % 8 != 0 {
        return Err("hc_diet_stage1_f32: hidden % 8 != 0".into());
    }
    let n_inj = if winj_b16.is_some() { streams } else { 0 };
    const ROWS_PB: usize = 4;
    let total_rows = rank + n_inj;
    if parts.len() < t * streams * rank
        || (n_inj > 0 && inj_parts.len() < t * n_inj * streams)
        || inv_out.len() < t * streams
    {
        return Err("hc_diet_stage1_f32: output buffers too short".into());
    }
    let f = e.func("hc_diet_stage1_f32");
    let cfg = LaunchConfig {
        grid_dim: (
            total_rows.div_ceil(ROWS_PB) as u32,
            t as u32,
            streams as u32,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (hidden * 4) as u32,
    };
    let (hi, ri, si, nji, rpb) = (
        hidden as i32,
        rank as i32,
        streams as i32,
        n_inj as i32,
        ROWS_PB as i32,
    );
    let winj = winj_b16.unwrap_or(wdown_b16); // unread when n_inj == 0
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(ptrs)
        .arg(nw_stack)
        .arg(wdown_b16)
        .arg(winj)
        .arg(&mut *parts)
        .arg(&mut *inj_parts)
        .arg(&mut *inv_out)
        .arg(&hi)
        .arg(&ri)
        .arg(&si)
        .arg(&nji)
        .arg(&rpb)
        .arg(&eps);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// hc-diet stage 2 (`hc_diet_stage2_f32`): low_act = silu(mean_s parts) (the
/// hc_lowrank_reduce association verbatim) + inj = 2*sigmoid(mean_s2 inj_parts).
#[allow(clippy::too_many_arguments)]
fn launch_hc_diet_stage2(
    e: &Engine,
    parts: &CudaSlice<f32>,
    inj_parts: &CudaSlice<f32>,
    low_act: &mut CudaSlice<f32>,
    inj_all: &mut CudaSlice<f32>,
    rank: usize,
    streams: usize,
    t: usize,
    with_inject: bool,
) -> Res<()> {
    let n_inj = if with_inject { streams } else { 0 };
    if low_act.len() < t * rank || (n_inj > 0 && inj_all.len() < n_inj * t) {
        return Err("hc_diet_stage2_f32: output buffers too short".into());
    }
    let f = e.func("hc_diet_stage2_f32");
    let cfg = LaunchConfig {
        grid_dim: (((rank + n_inj) as u32).div_ceil(256), t as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (ri, si, nji, ti) = (rank as i32, streams as i32, n_inj as i32, t as i32);
    let inv = 1.0f32 / streams as f32;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(parts)
        .arg(inj_parts)
        .arg(&mut *low_act)
        .arg(&mut *inj_all)
        .arg(&ri)
        .arg(&si)
        .arg(&nji)
        .arg(&ti)
        .arg(&inv);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// hc-diet stage 3 (`hc_diet_stage3_f32`): per dim-chunk block — the up dots for all
/// streams from a smem low_act copy, then the mix epilogue from the stage-1 inv scalars.
#[allow(clippy::too_many_arguments)]
fn launch_hc_diet_stage3(
    e: &Engine,
    ptrs: &CudaSlice<u64>,
    nw_stack: &CudaSlice<f32>,
    inv_in: &CudaSlice<f32>,
    wup_b16: &CudaSlice<u8>,
    low_act: &CudaSlice<f32>,
    mixed: &mut CudaSlice<f32>,
    hidden: usize,
    rank: usize,
    streams: usize,
    t: usize,
) -> Res<()> {
    const DIMS_PB: usize = 8;
    if mixed.len() < t * hidden {
        return Err("hc_diet_stage3_f32: output buffer too short".into());
    }
    let f = e.func("hc_diet_stage3_f32");
    let cfg = LaunchConfig {
        grid_dim: (hidden.div_ceil(DIMS_PB) as u32, t as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: ((rank + DIMS_PB * streams) * 4) as u32,
    };
    let (hi, ri, si, dpb) = (hidden as i32, rank as i32, streams as i32, DIMS_PB as i32);
    let inv_streams = 1.0f32 / streams as f32;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(ptrs)
        .arg(nw_stack)
        .arg(inv_in)
        .arg(wup_b16)
        .arg(low_act)
        .arg(&mut *mixed)
        .arg(&hi)
        .arg(&ri)
        .arg(&si)
        .arg(&dpb)
        .arg(&inv_streams);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// hc-diet MT stage 0 (`hc_diet_stage0_mt_f32`): the stage-1 RMS reduce EXACTLY, per
/// (token, stream) — bit-equal inv scalars for the weight-shared stages.
fn launch_hc_diet_stage0_mt(
    e: &Engine,
    ptrs: &CudaSlice<u64>,
    inv_out: &mut CudaSlice<f32>,
    hidden: usize,
    streams: usize,
    t: usize,
    eps: f32,
) -> Res<()> {
    if inv_out.len() < t * streams {
        return Err("hc_diet_stage0_mt_f32: inv buffer too short".into());
    }
    let f = e.func("hc_diet_stage0_mt_f32");
    let cfg = LaunchConfig {
        grid_dim: (t as u32, streams as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (hi, si, ti) = (hidden as i32, streams as i32, t as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(ptrs)
        .arg(&mut *inv_out)
        .arg(&hi)
        .arg(&si)
        .arg(&ti)
        .arg(&eps);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// hc-diet MT stage 1: weight rows read ONCE, tokens iterated inside with inline
/// normalization — per-(row, token) chains VERBATIM vs the token-grid stage 1.
#[allow(clippy::too_many_arguments)]
fn launch_hc_diet_stage1_mt(
    e: &Engine,
    ptrs: &CudaSlice<u64>,
    nw_stack: &CudaSlice<f32>,
    inv_in: &CudaSlice<f32>,
    wdown_b16: &CudaSlice<u8>,
    winj_b16: Option<&CudaSlice<u8>>,
    parts: &mut CudaSlice<f32>,
    inj_parts: &mut CudaSlice<f32>,
    hidden: usize,
    rank: usize,
    streams: usize,
    t: usize,
) -> Res<()> {
    if hidden % 8 != 0 || !(2..=12).contains(&t) {
        return Err("hc_diet_stage1_mt_f32: geometry".into());
    }
    let n_inj = if winj_b16.is_some() { streams } else { 0 };
    const ROWS_PB: usize = 4;
    let total_rows = rank + n_inj;
    if parts.len() < t * streams * rank || (n_inj > 0 && inj_parts.len() < t * n_inj * streams) {
        return Err("hc_diet_stage1_mt_f32: output buffers too short".into());
    }
    let f = e.func("hc_diet_stage1_mt_f32");
    let cfg = LaunchConfig {
        grid_dim: (total_rows.div_ceil(ROWS_PB) as u32, 1, streams as u32),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (hi, ri, si, nji, rpb, ti) = (
        hidden as i32,
        rank as i32,
        streams as i32,
        n_inj as i32,
        ROWS_PB as i32,
        t as i32,
    );
    let winj = winj_b16.unwrap_or(wdown_b16);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(ptrs)
        .arg(nw_stack)
        .arg(inv_in)
        .arg(wdown_b16)
        .arg(winj)
        .arg(&mut *parts)
        .arg(&mut *inj_parts)
        .arg(&hi)
        .arg(&ri)
        .arg(&si)
        .arg(&nji)
        .arg(&rpb)
        .arg(&ti);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// hc-diet MT stage 3: up rows read once, all T low_act rows resident in smem.
#[allow(clippy::too_many_arguments)]
fn launch_hc_diet_stage3_mt(
    e: &Engine,
    ptrs: &CudaSlice<u64>,
    nw_stack: &CudaSlice<f32>,
    inv_in: &CudaSlice<f32>,
    wup_b16: &CudaSlice<u8>,
    low_act: &CudaSlice<f32>,
    mixed: &mut CudaSlice<f32>,
    hidden: usize,
    rank: usize,
    streams: usize,
    t: usize,
) -> Res<()> {
    const DIMS_PB: usize = 8;
    if !(2..=12).contains(&t) || mixed.len() < t * hidden {
        return Err("hc_diet_stage3_mt_f32: geometry".into());
    }
    let smem = ((t * rank + DIMS_PB * streams * t) * 4) as u32;
    if smem > 96 * 1024 {
        return Err("hc_diet_stage3_mt_f32: smem over budget".into());
    }
    let f = e.func("hc_diet_stage3_mt_f32");
    let cfg = LaunchConfig {
        grid_dim: (hidden.div_ceil(DIMS_PB) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem,
    };
    let (hi, ri, si, dpb, ti) = (
        hidden as i32,
        rank as i32,
        streams as i32,
        DIMS_PB as i32,
        t as i32,
    );
    let inv_streams = 1.0f32 / streams as f32;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(ptrs)
        .arg(nw_stack)
        .arg(inv_in)
        .arg(wup_b16)
        .arg(low_act)
        .arg(&mut *mixed)
        .arg(&hi)
        .arg(&ri)
        .arg(&si)
        .arg(&dpb)
        .arg(&ti)
        .arg(&inv_streams);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// `hc_write_planes_f32`: plane_s += block_out ⊗ inj[s] for every stream in one launch
/// over the plane pointer table (hcmicro seam).
fn launch_hc_write_planes(
    e: &Engine,
    ptrs: &CudaSlice<u64>,
    block_out: &CudaSlice<f32>,
    inj: &CudaSlice<f32>,
    hidden: usize,
    t: usize,
    streams: usize,
) -> Res<()> {
    let f = e.func("hc_write_planes_f32");
    let n = (t * hidden) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256), streams as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (hi, ti) = (hidden as i32, t as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(ptrs).arg(block_out).arg(inj).arg(&hi).arg(&ti);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// `hc_inject_gates_bf16w_f32`: the bf16-weight twin of `launch_hc_inject_gates` — same
/// grid, same loop order, same reduction tree, exact bf16→f32 widening, so BIT-IDENTICAL
/// to the f32 arm when the resident bytes match (the `bf16_twin` representability guard).
fn launch_hc_inject_gates_b16(
    e: &Engine,
    normed: &CudaSlice<f32>,
    w: &CudaSlice<u8>,
    out: &mut CudaSlice<f32>,
    streams: usize,
    t: usize,
    hidden: usize,
) -> Res<()> {
    let f = e.func("hc_inject_gates_bf16w_f32");
    let cfg = LaunchConfig {
        grid_dim: (streams as u32, t as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (si, ti, hi) = (streams as i32, t as i32, hidden as i32);
    let inv = 1.0f32 / streams as f32;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(normed)
        .arg(w)
        .arg(out)
        .arg(&si)
        .arg(&ti)
        .arg(&hi)
        .arg(&inv);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// bf16 trunk-residency twin builder (load time). Returns the packed bf16 device bytes
/// iff BOTH guards pass: in_f % 8 == 0 (the kernel's uint4 vector width — geometry, not
/// policy) and every value is exactly bf16-representable (low 16 mantissa bits zero —
/// true whenever the checkpoint row was BF16, since dequant is an exact widening; the
/// f32 tiny fixture fails this and keeps its f32-only residency).
fn bf16_twin(e: &Engine, data: &[f32], in_f: usize) -> Res<Option<CudaSlice<u8>>> {
    if in_f % 8 != 0 {
        return Ok(None);
    }
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let bits = v.to_bits();
        if bits & 0xFFFF != 0 {
            return Ok(None);
        }
        bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
    }
    Ok(Some(e.htod_bytes(&bytes)?))
}

/// One launch of `qmatvec_bf16w_f32`: y[b, tok, :out_f] = W_b(bf16) @ x_{b,tok}, f32
/// accumulate. Strides in ELEMENTS; `x_bstride == 0` shares one activation across the
/// batch (the read gate's up projection). Products are exact (bf16→f32 widening); only
/// the reduction tree differs from cuBLASLt — the accumulation class.
#[allow(clippy::too_many_arguments)]
fn launch_qmatvec_bf16w(
    e: &Engine,
    w: &CudaSlice<u8>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    in_f: usize,
    out_f: usize,
    t: usize,
    batch: usize,
    w_bstride: usize,
    x_bstride: usize,
    x_tstride: usize,
    y_bstride: usize,
) -> Res<()> {
    if in_f % 8 != 0 || x_bstride % 8 != 0 || x_tstride % 8 != 0 {
        return Err("qmatvec_bf16w_f32: stride breaks the uint4/float4 vector width".into());
    }
    if y.len() < (batch - 1) * y_bstride + t * out_f {
        return Err("qmatvec_bf16w_f32: output buffer too short".into());
    }
    let f = e.func("qmatvec_bf16w_f32");
    let cfg = LaunchConfig {
        grid_dim: (out_f as u32, t as u32, batch as u32),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, outf, ti) = (in_f as i32, out_f as i32, t as i32);
    let (wb, xb, xt, yb) = (
        w_bstride as i64,
        x_bstride as i64,
        x_tstride as i64,
        y_bstride as i64,
    );
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(w)
        .arg(x)
        .arg(y)
        .arg(&inf)
        .arg(&outf)
        .arg(&ti)
        .arg(&wb)
        .arg(&xb)
        .arg(&xt)
        .arg(&yb);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Stacked bf16 twin over several same-in_f projections (the proj-stack seam): concat
/// the host f32 rows and build one packed twin. `None` under the same guards as
/// `bf16_twin` (in_f % 8, exact representability of EVERY part). The stack REPLACES the
/// per-mat twins (VRAM-neutral): the per-mat arm launches against row-offset VIEWS of
/// the stack — same bytes, same kernel, bit-identical to separate residency.
fn bf16_stack_twin(e: &Engine, parts: &[&[f32]], in_f: usize) -> Res<Option<CudaSlice<u8>>> {
    let mut cat: Vec<f32> = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        cat.extend_from_slice(p);
    }
    bf16_twin(e, &cat, in_f)
}

/// Required-stack twin (the TP2 `need_twin` posture).
fn need_stack_twin(e: &Engine, parts: &[&[f32]], in_f: usize, what: &str) -> Res<CudaSlice<u8>> {
    bf16_stack_twin(e, parts, in_f)?.ok_or_else(|| {
        format!("qwen4exp_gpu tp2: {what} has no exact bf16 stack twin (in_f {in_f})").into()
    })
}

/// One `qmatvec_bf16w_f32` launch against a ROW-OFFSET VIEW of a stacked twin (the
/// per-mat arm of the proj-stack seam): W = stack rows [row_off, row_off+out_f), batch 1.
/// Identical kernel, grid, and bytes as a separately-resident twin => bit-identical.
#[allow(clippy::too_many_arguments)]
fn launch_qmatvec_bf16w_off(
    e: &Engine,
    w_stack: &CudaSlice<u8>,
    row_off: usize,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    in_f: usize,
    out_f: usize,
    t: usize,
) -> Res<()> {
    if in_f % 8 != 0 {
        return Err("qmatvec_bf16w_f32: stride breaks the uint4/float4 vector width".into());
    }
    if y.len() < t * out_f {
        return Err("qmatvec_bf16w_f32: output buffer too short".into());
    }
    let byte_off = row_off * in_f * 2;
    if w_stack.len() < byte_off + out_f * in_f * 2 {
        return Err("qmatvec_bf16w_f32: stacked twin shorter than the row window".into());
    }
    let wv = w_stack.slice(byte_off..w_stack.len());
    let f = e.func("qmatvec_bf16w_f32");
    let cfg = LaunchConfig {
        grid_dim: (out_f as u32, t as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, outf, ti) = (in_f as i32, out_f as i32, t as i32);
    let (wb, xb, xt, yb) = (0i64, 0i64, in_f as i64, 0i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(&wv)
        .arg(x)
        .arg(y)
        .arg(&inf)
        .arg(&outf)
        .arg(&ti)
        .arg(&wb)
        .arg(&xb)
        .arg(&xt)
        .arg(&yb);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// `qmatvec_bf16w_f32` against row-offset W, x, and y VIEWS (t == 1): the per-selected-
/// expert arm of the DeviceBf16 draft bank (mtp-spec lane) — expert `e`'s projection is
/// rows [w_row_off, w_row_off+out_f) of the resident [E*out_f, in_f] bf16 stack. Same
/// kernel and per-row program as every other qmatvec_bf16w launch (exact-widening
/// products, block-128 reduce) => rows are bit-identical to a separately-resident twin.
#[allow(clippy::too_many_arguments)]
fn launch_qmatvec_bf16w_off_into(
    e: &Engine,
    w_stack: &CudaSlice<u8>,
    w_row_off: usize,
    x: &CudaSlice<f32>,
    x_off: usize,
    y: &mut CudaSlice<f32>,
    y_off: usize,
    in_f: usize,
    out_f: usize,
) -> Res<()> {
    if in_f % 8 != 0 {
        return Err("qmatvec_bf16w_f32: stride breaks the uint4/float4 vector width".into());
    }
    let byte_off = w_row_off * in_f * 2;
    if w_stack.len() < byte_off + out_f * in_f * 2 {
        return Err("qmatvec_bf16w_f32: bank shorter than the expert row window".into());
    }
    if x.len() < x_off + in_f || y.len() < y_off + out_f {
        return Err("qmatvec_bf16w_f32: operand views out of range".into());
    }
    let wv = w_stack.slice(byte_off..w_stack.len());
    let xv = x.slice(x_off..x_off + in_f);
    let mut yv = y.slice_mut(y_off..y_off + out_f);
    let f = e.func("qmatvec_bf16w_f32");
    let cfg = LaunchConfig {
        grid_dim: (out_f as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, outf, ti) = (in_f as i32, out_f as i32, 1i32);
    let (wb, xb, xt, yb) = (0i64, 0i64, in_f as i64, 0i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(&wv)
        .arg(&xv)
        .arg(&mut yv)
        .arg(&inf)
        .arg(&outf)
        .arg(&ti)
        .arg(&wb)
        .arg(&xb)
        .arg(&xt)
        .arg(&yb);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Device-selected expert launch over a DeviceBf16 bank (`qmatvec_bf16w_sel_f32`,
/// devtwin lane): one launch per projection covers every routed expert — slot s reads
/// its expert id from the DEVICE `sel` array at `sel_off + s` and writes y at s*out_f.
/// Per-row program qmatvec_bf16w_f32 VERBATIM => bit-identical to the per-slot
/// `launch_qmatvec_bf16w_off_into` chain (asserted by the bf16 oracle's sel mode).
#[allow(clippy::too_many_arguments)]
fn launch_qmatvec_bf16w_sel(
    e: &Engine,
    bank: &CudaSlice<u8>,
    sel: &CudaSlice<i32>,
    sel_off: usize,
    x: &CudaSlice<f32>,
    x_off: usize,
    // Per-slot activation stride in elements: 0 = shared row (gate/up), in_f = each
    // slot its own row (down over the act slab).
    x_sstride: usize,
    y: &mut CudaSlice<f32>,
    n_sel: usize,
    in_f: usize,
    out_f: usize,
) -> Res<()> {
    if in_f % 8 != 0 {
        return Err("qmatvec_bf16w_sel_f32: stride breaks the uint4/float4 vector width".into());
    }
    if sel.len() < sel_off + n_sel
        || x.len() < x_off + (n_sel - 1) * x_sstride + in_f
        || y.len() < n_sel * out_f
        || n_sel == 0
    {
        return Err("qmatvec_bf16w_sel_f32: operand views out of range".into());
    }
    let sv = sel.slice(sel_off..sel_off + n_sel);
    let xv = x.slice(x_off..x.len());
    let f = e.func("qmatvec_bf16w_sel_f32");
    let cfg = LaunchConfig {
        grid_dim: (out_f as u32, 1, n_sel as u32),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, outf, ns) = (in_f as i32, out_f as i32, n_sel as i32);
    let xs = x_sstride as i64;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(bank)
        .arg(&sv)
        .arg(&xv)
        .arg(&mut *y)
        .arg(&inf)
        .arg(&outf)
        .arg(&ns)
        .arg(&xs);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Multi-token weight-shared launch (`qmatvec_bf16w_mt_f32`, mtp-spec verify): one
/// block per output row reads W once and fills EVERY token's output — per (row, token)
/// bit-identical to the per-token grid (kernel doc). 2 <= t <= 12; `w_row_off` selects
/// a row window of a stacked twin.
#[allow(clippy::too_many_arguments)]
fn launch_qmatvec_bf16w_mt(
    e: &Engine,
    w_stack: &CudaSlice<u8>,
    w_row_off: usize,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    in_f: usize,
    out_f: usize,
    t: usize,
) -> Res<()> {
    if in_f % 8 != 0 {
        return Err("qmatvec_bf16w_mt_f32: in_f % 8 != 0".into());
    }
    if !(2..=12).contains(&t) {
        return Err("qmatvec_bf16w_mt_f32: t out of range (2..=12)".into());
    }
    let byte_off = w_row_off * in_f * 2;
    if w_stack.len() < byte_off + out_f * in_f * 2 || y.len() < t * out_f || x.len() < t * in_f {
        return Err("qmatvec_bf16w_mt_f32: operands out of range".into());
    }
    let wv = w_stack.slice(byte_off..w_stack.len());
    let f = e.func("qmatvec_bf16w_mt_f32");
    let cfg = LaunchConfig {
        grid_dim: (out_f as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, outf, ti) = (in_f as i32, out_f as i32, t as i32);
    let (wb, xb, xt, yb) = (0i64, 0i64, in_f as i64, 0i64);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(&wv)
        .arg(x)
        .arg(y)
        .arg(&inf)
        .arg(&outf)
        .arg(&ti)
        .arg(&wb)
        .arg(&xb)
        .arg(&xt)
        .arg(&yb);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Trunk dense linear off a STACKED bf16 twin (proj-stack residency): the bf16 arm is a
/// row-offset view launch when the twin exists and the trunk seam is on, else the f32
/// cuBLASLt path.
#[allow(clippy::too_many_arguments)]
fn linear_trunk_stacked_into(
    e: &Engine,
    w_f32: &CudaSlice<f32>,
    stack_b16: &Option<CudaSlice<u8>>,
    row_off: usize,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    t: usize,
    in_f: usize,
    out_f: usize,
) -> Res<()> {
    if trunk_bf16_on() {
        if let Some(w) = stack_b16 {
            if (2..=12).contains(&t) && verify_mt_on() {
                return launch_qmatvec_bf16w_mt(e, w, row_off, x, y, in_f, out_f, t);
            }
            return launch_qmatvec_bf16w_off(e, w, row_off, x, y, in_f, out_f, t);
        }
    }
    if w_f32.len() < in_f * out_f {
        return Err(
            "qwen4exp_gpu: trunk f32 original dropped (trunk_f32_diet) — the bf16 \
                    twin path is required (keep trunk seams ON)"
                .into(),
        );
    }
    e.linear_device_into(x, w_f32, y, t, in_f, out_f)
}

/// One launch of `qmatvec_bf16w_multi4_f32`: the row-stacked twin against ONE t==1
/// activation, each output row routed into its original destination buffer by row range
/// (raw device pointers — no copies). Per-row math is qmatvec_bf16w_f32 VERBATIM, so
/// outputs are BIT-IDENTICAL to the per-mat launches this replaces.
fn launch_qmatvec_bf16w_multi4(
    e: &Engine,
    w_stack: &CudaSlice<u8>,
    x: &CudaSlice<f32>,
    parts: &[(&CudaSlice<f32>, usize)],
    in_f: usize,
) -> Res<()> {
    if in_f % 8 != 0 {
        return Err("qmatvec_bf16w_multi4_f32: in_f % 8 != 0".into());
    }
    if parts.is_empty() || parts.len() > 4 {
        return Err("qmatvec_bf16w_multi4_f32: 1..=4 parts".into());
    }
    let total: usize = parts.iter().map(|&(_, r)| r).sum();
    if w_stack.len() < total * in_f * 2 {
        return Err("qmatvec_bf16w_multi4_f32: stacked twin shorter than the row plan".into());
    }
    let stream = e.gpu.stream();
    let mut ptrs = [0u64; 4];
    let mut rows = [0i32; 4];
    for (i, &(buf, r)) in parts.iter().enumerate() {
        if buf.len() < r {
            return Err("qmatvec_bf16w_multi4_f32: destination shorter than its rows".into());
        }
        ptrs[i] = buf.device_ptr(&stream).0;
        rows[i] = r as i32;
    }
    let f = e.func("qmatvec_bf16w_multi4_f32");
    let cfg = LaunchConfig {
        grid_dim: (total as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let inf = in_f as i32;
    let mut b = stream.launch_builder(&f);
    b.arg(w_stack)
        .arg(x)
        .arg(&ptrs[0])
        .arg(&rows[0])
        .arg(&ptrs[1])
        .arg(&rows[1])
        .arg(&ptrs[2])
        .arg(&rows[2])
        .arg(&ptrs[3])
        .arg(&rows[3])
        .arg(&inf);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Trunk dense linear into a caller-provided buffer: the bf16 twin (one
/// `qmatvec_bf16w_f32` launch) when resident and the seam is on, else the f32
/// cuBLASLt path — the A/B twin (the step-workspace form, item 2a).
#[allow(clippy::too_many_arguments)]
fn linear_trunk_into(
    e: &Engine,
    w_f32: &CudaSlice<f32>,
    w_b16: &Option<CudaSlice<u8>>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    t: usize,
    in_f: usize,
    out_f: usize,
) -> Res<()> {
    if trunk_bf16_on() {
        if let Some(w) = w_b16 {
            if (2..=12).contains(&t) && verify_mt_on() {
                return launch_qmatvec_bf16w_mt(e, w, 0, x, y, in_f, out_f, t);
            }
            return launch_qmatvec_bf16w(e, w, x, y, in_f, out_f, t, 1, 0, 0, in_f, 0);
        }
    }
    if w_f32.len() < in_f * out_f {
        return Err(
            "qwen4exp_gpu: trunk f32 original dropped (trunk_f32_diet) — the bf16 \
                    twin path is required (keep trunk seams ON)"
                .into(),
        );
    }
    e.linear_device_into(x, w_f32, y, t, in_f, out_f)
}

/// One launch of the grouped selected-experts matvec: y[slot, :out_f] =
/// macros[sel[slot]] × (W_{sel[slot]} @ x_slot) over the AS-STORED modelopt bank (no
/// repack). `x_stride` = 0 shares one activation row across slots (gate/up); = in_f
/// reads per-slot rows (down). Dispatches the v2 kernel (uint4 code loads + 2 rows per
/// warp — perf lane item 3) when the seam is on and the geometry admits it
/// (in_f % 32 == 0, out_f % 2 == 0); v1 is the fallback and the A/B twin. Round 3 adds
/// the v3 kernel (4 rows/warp, `set_sel_v3`, out_f % 4 == 0) ahead of v2 in the chain.
#[allow(clippy::too_many_arguments)]
fn launch_nvfp4_sel_matvec(
    e: &Engine,
    codes: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    macros_dev: &CudaSlice<f32>,
    sel: &CudaSlice<i32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    n_sel: usize,
    in_f: usize,
    out_f: usize,
    x_stride: usize,
) -> Res<()> {
    if in_f % 16 != 0 {
        return Err("qmatvec_nvfp4_modelopt_sel_f32: in_f % 16 != 0".into());
    }
    if y.len() < n_sel * out_f {
        return Err("qmatvec_nvfp4_modelopt_sel_f32: output shorter than n_sel*out_f".into());
    }
    // Sub-warp pair groups (`selgroup`, default OFF) take precedence over the v3/v2/v1
    // chain when the geometry tiles exactly; `(g=32, rows=4)` reproduces v3's bits.
    let grp = sel_group_resolve(sel_group_dn(), in_f, out_f);
    let v3 = grp.is_none() && sel_v3_on() && in_f % 32 == 0 && out_f % 4 == 0;
    let v2 = grp.is_none() && !v3 && sel_v2_on() && in_f % 32 == 0 && out_f % 2 == 0;
    let f = e.func(if grp.is_some() {
        "qmatvec_nvfp4_modelopt_sel_g_f32"
    } else if v3 {
        "qmatvec_nvfp4_modelopt_sel_f32_v3"
    } else if v2 {
        "qmatvec_nvfp4_modelopt_sel_f32_v2"
    } else {
        "qmatvec_nvfp4_modelopt_sel_f32"
    });
    // Warp packing (4 warps/block) was tried here and REVERTED: measured NEGATIVE on
    // decode (plain arm 14.38 -> 15.13 ms) and flat on verify sel (mtp6 battery,
    // spec/mtp6) — the sel slice is not SM-block-slot-limited. The kernels keep the
    // lane-based indexing (identical at block 32); launch stays one warp per block. The
    // `selgroup` kernels honour `blockDim.x >> 5` too, but this lane deliberately leaves
    // block 32 alone so the A/B attributes ONE change (the lane partition) — a warps-per-
    // block knob would re-open the reverted measurement as a second free variable.
    let grid_x = match grp {
        Some((g, rows)) => out_f / ((32 / g) * rows),
        None if v3 => out_f / 4,
        None if v2 => out_f / 2,
        None => out_f,
    };
    let cfg = LaunchConfig {
        grid_dim: (grid_x as u32, n_sel as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, outf) = (in_f as i32, out_f as i32);
    let xs = x_stride as i64;
    let (gi, ri) = grp.map_or((0i32, 0i32), |(g, rows)| (g as i32, rows as i32));
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(codes)
        .arg(scales)
        .arg(macros_dev)
        .arg(sel)
        .arg(x)
        .arg(y)
        .arg(&inf)
        .arg(&outf)
        .arg(&xs);
    if grp.is_some() {
        b.arg(&gi).arg(&ri);
    }
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One launch of the fused gate+up+silu sel matvec
/// (`qmatvec_nvfp4_modelopt_sel_gu_silu_f32`): act[slot, :ff] = silu(gate) * up over
/// the shared activation row. `sel`/`pack_raw` pick the addressing mode (host sel
/// array vs the TP2 count-gated pack blob). Bit-identical to the v3 gate + v3 up +
/// silu_mul chain (kernel doc).
#[allow(clippy::too_many_arguments)]
fn launch_nvfp4_sel_gu_silu(
    e: &Engine,
    gate: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
    up: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
    sel: Option<&CudaSlice<i32>>,
    pack_raw: u64,
    n_sel: usize,
    x: &CudaSlice<f32>,
    act: &mut CudaSlice<f32>,
    in_f: usize,
    ff: usize,
    // (slot -> token map, x token stride): ONE launch over every verify column's
    // routed experts (per-slot program unchanged — bit-identical). None = shared x.
    tok: Option<(&CudaSlice<i32>, usize)>,
) -> Res<()> {
    if in_f % 32 != 0 || ff % 4 != 0 {
        return Err("qmatvec_nvfp4_modelopt_sel_gu_silu_f32: geometry".into());
    }
    if act.len() < n_sel * ff {
        return Err("qmatvec_nvfp4_modelopt_sel_gu_silu_f32: act buffer too short".into());
    }
    if sel.is_none() == (pack_raw == 0) {
        return Err("qmatvec_nvfp4_modelopt_sel_gu_silu_f32: exactly one of sel/pack".into());
    }
    // Sub-warp pair groups (`selgroup`, default OFF); `(g=32, rows=4)` reproduces the
    // shipped kernel's bits, pack and tok_map modes included.
    let grp = sel_group_resolve(sel_group_gu(), in_f, ff);
    let f = e.func(if grp.is_some() {
        "qmatvec_nvfp4_modelopt_sel_gu_silu_g_f32"
    } else {
        "qmatvec_nvfp4_modelopt_sel_gu_silu_f32"
    });
    // Warp packing reverted (see launch_nvfp4_sel_matvec): one warp per block.
    let grid_x = match grp {
        Some((g, rows)) => ff / ((32 / g) * rows),
        None => ff / 4,
    };
    let cfg = LaunchConfig {
        grid_dim: (grid_x as u32, n_sel as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, ffi, ms) = (in_f as i32, ff as i32, n_sel as i32);
    let (gi, ri) = grp.map_or((0i32, 0i32), |(g, rows)| (g as i32, rows as i32));
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(gate.0)
        .arg(gate.1)
        .arg(gate.2)
        .arg(up.0)
        .arg(up.1)
        .arg(up.2);
    match sel {
        Some(s) => {
            b.arg(s);
        }
        None => {
            // unread in pack mode; any live device pointer keeps the arg slot filled
            b.arg(gate.2);
        }
    }
    let stream2 = e.gpu.stream();
    let (tok_raw, x_tstride) = match tok {
        Some((tm, stride)) => (tm.device_ptr(&stream2).0, stride as i64),
        None => (0u64, 0i64),
    };
    b.arg(&pack_raw)
        .arg(&ms)
        .arg(x)
        .arg(&mut *act)
        .arg(&inf)
        .arg(&ffi)
        .arg(&tok_raw)
        .arg(&x_tstride);
    if grp.is_some() {
        b.arg(&gi).arg(&ri);
    }
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One row of the MoE routed-union cost probe (`moeu` lane, mtp13).
#[derive(Debug, Clone, Copy)]
pub struct MoeUnionRow {
    /// Verify columns fed (t). 1 = the plain-decode reference shape.
    pub t: usize,
    /// (token, expert) pairs dispatched = the grid.y extent of both launches.
    pub slots: usize,
    /// DISTINCT experts among those slots — the only quantity a union gather changes.
    pub union_size: usize,
    /// Median us/launch of the fused gate+up+silu sel matvec.
    pub gu_us: f64,
    /// Median us/launch of the down sel matvec.
    pub down_us: f64,
    /// (max-min)/median over the arm's reps. Reported so a reader can see whether an arm's
    /// delta against another arm is inside its own noise; LAW:interleaved-ab wants every arm
    /// to report its spread, and at realistic union sizes this lever's delta is smaller than
    /// this column.
    pub gu_spread_rel: f64,
    pub down_spread_rel: f64,
}

/// COST INSTRUMENT for the MoE routed-union lever (`moeu`), and the reason it exists
/// instead of a kernel: the union gather changes exactly ONE thing about the MoE verify
/// section — how many DISTINCT experts' NVFP4 bytes the chunk reads — while leaving the
/// per-slot arithmetic, the slot count and the launch geometry alone. So the lever can be
/// priced WITHOUT writing it, by running the shipped kernels at a fixed slot count and
/// varying only the number of distinct experts those slots name.
///
/// The three-point decomposition each sweep yields, at t verify columns and k selected:
///
/// - `slots = t*k, union = t*k` — TODAY. Every slot reads its expert's bytes; duplicates
///   across tokens re-read (the kernel doc says so in as many words: "the weight banks are
///   read once per selected slot either way — the launch count is what drops").
/// - `slots = t*k, union = U` — the IDEALISED union gather: same arithmetic, same slots,
///   only `U` experts' bytes touched. A real union-major kernel cannot beat this by much
///   and cannot be slower on traffic, so this row is the lever's payoff, measured.
/// - `slots = k, union = k` — the t=1 plain reference, for the round arithmetic.
///
/// If the middle row does not beat the first, the section's cost is not the duplicated
/// bytes and the lever has no surface REGARDLESS of what the routed union sizes turn out
/// to be — the card's 128 MiB L2 is large enough to hold a whole chunk's routed working
/// set at this geometry (60 slots x 1.76 MiB gate+up = 105.5 MiB), so the hardware may
/// already be deduplicating what the kernel re-reads.
///
/// SYNTHETIC BANKS, stated because a probe that looks like a gate is how a wrong number
/// gets quoted later. This loads NO checkpoint: it allocates a bank of the serving
/// geometry (`experts` x `ff` x `hidden` gate + up, `experts` x `hidden` x `ff` down) and
/// fills it with deterministic pseudo-random bytes. That is sound for a TRAFFIC and
/// LATENCY probe and for nothing else: the NVFP4 lane program is branch-free and
/// data-independent (LUT extract, fixed shfl tree), so bytes decide addresses and never
/// control flow. Scale bytes are held in a modest ue4m3 range so the f32 chain stays in
/// normal range; no output of this probe is a correctness claim and none is compared to an
/// oracle. Expert ids are SPREAD across the bank by a fixed stride, because a clustered
/// id set would make the sweep measure address locality instead of distinct-byte count.
///
/// Numbers from this probe are per-LAUNCH; the section cost is per LAYER (one gu + one
/// down launch each) times the model's MoE layer count.
pub fn moe_union_cost_probe(
    e: &Engine,
    experts: usize,
    hidden: usize,
    ff: usize,
    selected: usize,
    t: usize,
    reps: usize,
) -> Res<Vec<MoeUnionRow>> {
    if hidden % 32 != 0 || ff % 4 != 0 {
        return Err("moe_union_cost_probe: needs the gufuse geometry (hidden%32, ff%4)".into());
    }
    if selected == 0 || t == 0 || reps == 0 {
        return Err("moe_union_cost_probe: selected/t/reps must be non-zero".into());
    }
    // Deterministic byte fill. Codes index a 16-entry LUT so every byte is legal; scale
    // bytes are confined to a mid ue4m3 range so no product leaves normal f32 range.
    let code_byte = |i: usize| -> u8 { (i.wrapping_mul(2_654_435_761) >> 13) as u8 };
    let scale_byte = |i: usize| -> u8 { 0x38 | ((i.wrapping_mul(40_503) >> 7) & 0x07) as u8 };
    let mk = |n: usize, f: &dyn Fn(usize) -> u8| -> Res<CudaSlice<u8>> {
        let host: Vec<u8> = (0..n).map(f).collect();
        let d = e.htod_bytes(&host)?;
        drop(host);
        Ok(d)
    };
    // gate/up: [experts, ff, hidden]; down: [experts, hidden, ff]. Gate and up get
    // SEPARATE allocations on purpose — aliasing them would halve the distinct bytes and
    // silently turn the sweep into a cache-hit measurement.
    let gu_codes_n = experts * ff * (hidden / 2);
    let gu_scales_n = experts * ff * (hidden / 16);
    let dn_codes_n = experts * hidden * (ff / 2);
    let dn_scales_n = experts * hidden * (ff / 16);
    let gc = mk(gu_codes_n, &code_byte)?;
    let gs = mk(gu_scales_n, &scale_byte)?;
    let uc = mk(gu_codes_n, &|i| code_byte(i ^ 0x5A5A_5A5A))?;
    let us = mk(gu_scales_n, &|i| scale_byte(i ^ 0x3C3C_3C3C))?;
    let dc = mk(dn_codes_n, &|i| code_byte(i ^ 0x0F0F_0F0F))?;
    let ds = mk(dn_scales_n, &|i| scale_byte(i ^ 0x1111_1111))?;
    let gm = e.htod(&vec![1.0f32; experts])?;
    let um = e.htod(&vec![1.0f32; experts])?;
    let dm = e.htod(&vec![1.0f32; experts])?;
    // Activations: small normal values, one row per verify column.
    let mixed_h: Vec<f32> = (0..t * hidden)
        .map(|i| ((i.wrapping_mul(40_503) % 1000) as f32) / 4000.0 - 0.125)
        .collect();
    let mixed = e.htod(&mixed_h)?;

    // Spread candidate expert ids over the whole bank by a fixed stride.
    let pool: Vec<i32> = {
        let stride = (experts / (t * selected).max(1)).max(1);
        (0..t * selected)
            .map(|i| ((i * stride) % experts) as i32)
            .collect()
    };

    let mut rows: Vec<MoeUnionRow> = Vec::new();
    // (t, union target). `new` fresh experts per extra column: union = k + (t-1)*new.
    let mut cells: Vec<(usize, usize)> = vec![(1, selected)];
    for new in 0..=selected {
        cells.push((t, selected + (t - 1) * new));
    }
    // Build EVERY cell's device state first, then interleave the arms rep by rep.
    //
    // WHY THE ARMS ARE INTERLEAVED AND NOT SWEPT (LAW:interleaved-ab): the union sizes are
    // arms of a perf A/B, and a contiguous block per arm ordered monotonically in union size
    // lets any clock/thermal drift over the run masquerade as a union effect -- in this
    // sweep's natural order (small union first) drift would INFLATE the apparent payoff,
    // which is the direction that would have made a dead lever look alive. Interleaving puts
    // every arm at every point of the drift curve.
    struct Cell {
        t: usize,
        slots: usize,
        union_size: usize,
        sel: CudaSlice<i32>,
        tokm: CudaSlice<i32>,
        act: CudaSlice<f32>,
        partial: CudaSlice<f32>,
        gu: Vec<f64>,
        dn: Vec<f64>,
    }
    let mut built: Vec<Cell> = Vec::with_capacity(cells.len());
    for (cells_t, want_union) in cells {
        let slots = cells_t * selected;
        // Build the slot->expert map: column 0 takes the first k of the pool; each later
        // column re-uses `shared` of column 0's experts and takes `new` fresh ones. Within
        // a column the ids stay DISTINCT, which is what top-k routing guarantees.
        let new = if cells_t > 1 {
            (want_union - selected) / (cells_t - 1)
        } else {
            0
        };
        let shared = selected - new;
        let mut sel_h: Vec<i32> = Vec::with_capacity(slots);
        let mut tok_h: Vec<i32> = Vec::with_capacity(slots);
        let mut fresh = selected;
        for col in 0..cells_t {
            if col == 0 {
                sel_h.extend_from_slice(&pool[0..selected]);
            } else {
                sel_h.extend_from_slice(&pool[0..shared]);
                for _ in 0..new {
                    sel_h.push(pool[fresh % pool.len()]);
                    fresh += 1;
                }
            }
            for _ in 0..selected {
                tok_h.push(col as i32);
            }
        }
        let union_size = {
            let mut u: Vec<i32> = sel_h.clone();
            u.sort_unstable();
            u.dedup();
            u.len()
        };
        built.push(Cell {
            t: cells_t,
            slots,
            union_size,
            sel: e.htod_i32(&sel_h)?,
            tokm: e.htod_i32(&tok_h)?,
            act: e.zeros(slots * ff)?,
            partial: e.zeros(slots * hidden)?,
            gu: Vec::with_capacity(reps),
            dn: Vec::with_capacity(reps),
        });
    }
    // Rep 0 is a warmed throwaway for EVERY arm: the first launch of a width pays workspace
    // allocation and a cold instruction cache (the scan_warm lesson).
    for rep in 0..(reps + 1) {
        for c in built.iter_mut() {
            let tok_arg = if c.t > 1 {
                Some((&c.tokm, hidden))
            } else {
                None
            };
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            launch_nvfp4_sel_gu_silu(
                e,
                (&gc, &gs, &gm),
                (&uc, &us, &um),
                Some(&c.sel),
                0,
                c.slots,
                &mixed,
                &mut c.act,
                hidden,
                ff,
                tok_arg,
            )?;
            e.stream().synchronize()?;
            let t1 = std::time::Instant::now();
            launch_nvfp4_sel_matvec(
                e,
                &dc,
                &ds,
                &dm,
                &c.sel,
                &c.act,
                &mut c.partial,
                c.slots,
                ff,
                hidden,
                ff,
            )?;
            e.stream().synchronize()?;
            let t2 = std::time::Instant::now();
            if rep > 0 {
                c.gu.push(t1.duration_since(t0).as_secs_f64() * 1e6);
                c.dn.push(t2.duration_since(t1).as_secs_f64() * 1e6);
            }
        }
    }
    let stat = |v: &[f64]| -> (f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = s[s.len() / 2];
        // Spread of the decision statistic, so a reader can see whether an arm's delta is
        // inside its own noise (the escalation rule's input).
        let spread = if med > 0.0 {
            (s[s.len() - 1] - s[0]) / med
        } else {
            0.0
        };
        (med, spread)
    };
    for c in &built {
        let (gu_us, gu_spread) = stat(&c.gu);
        let (down_us, down_spread) = stat(&c.dn);
        rows.push(MoeUnionRow {
            t: c.t,
            slots: c.slots,
            union_size: c.union_size,
            gu_us,
            down_us,
            gu_spread_rel: gu_spread,
            down_spread_rel: down_spread,
        });
    }
    Ok(rows)
}

/// One row of the sel-kernel SHAPE cost probe (`downsel` lane, mtp14).
#[derive(Debug, Clone)]
pub struct SelShapeRow {
    /// Verify columns fed (t). 1 = the plain-decode shape.
    pub t: usize,
    /// (token, expert) slots = grid.y of both launches.
    pub slots: usize,
    /// The `selgroup` spec this arm ran (`off` = the shipped v3 / gufuse kernels).
    pub arm: String,
    /// Resolved (g, rows) per family, and the grid.x each launch used — the whole point of
    /// the table is that a shape trades lane occupancy against warp count, so both have to
    /// be readable next to the time.
    pub gu_shape: String,
    pub dn_shape: String,
    pub gu_grid_x: usize,
    pub dn_grid_x: usize,
    pub gu_us: f64,
    pub down_us: f64,
    pub gu_spread_rel: f64,
    pub down_spread_rel: f64,
}

/// Cost probe for the sel matvecs' SUB-WARP pair-group shapes (`downsel` lane, mtp14),
/// on synthetic banks of the serving geometry with NO checkpoint (~1.3 GiB, ~30 s) — so it
/// interleaves between any other lane's cells the way the `moeu` probe does.
///
/// WHAT IT MEASURES. `moe_union_probe` established that this section is per-slot-work bound
/// (KNEE:q4e-sel-slots-not-bytes). Per-slot work is what an idle lane wastes, and at this
/// artifact's geometry the pair loop leaves 37.5% of the down launch's lanes and 16.7% of
/// the gate+up launch's lane-slots empty. This probe runs the SAME slots, the SAME distinct
/// experts and the SAME banks through each candidate lane partition, so the only thing
/// varying between arms is the shape.
///
/// TWO CONTROLS BUILT IN, because a shape table without them is unreadable:
///
/// 1. **`off` vs `dn:32:4+gu:32:4`.** The second arm is the sub-warp kernel at the shape
///    where it degenerates to the shipped program — bit-identical output (gated by
///    `gate_nvfp4_sel_group`). It went in as a noise floor (LAW:ab-arm-identity applied to a
///    perf table: an arm running the same program must measure the same) and it EARNED its
///    place by not being one — it reproducibly measures a few percent faster than `off`,
///    because the source restructure changes nvcc's scheduling for identical bits. So
///    `arm / off` mixes two effects and only `arm / control` is the shape's. Anyone reading
///    this table for a shape claim reads the control-relative column.
/// 2. **Per-arm spread**, reported per arm, never averaged away.
///
/// Arms are interleaved REP BY REP (LAW:interleaved-ab / TRAP:monotone-sweep-inflates-the-
/// lever): a shape ladder run as contiguous blocks would let clock/thermal drift over the
/// run read as a shape effect, and the natural order (baseline first) inflates the payoff.
/// Rep 0 of every arm is a warmed throwaway (the `scan_warm` lesson).
///
/// TIMING ARM: hold `flock -x` around the WHOLE invocation, and never quote a row measured
/// on the rig (LAW:rig-gpu-exactness-only — the rig is for the exactness arms above).
#[allow(clippy::too_many_arguments)]
pub fn sel_shape_cost_probe(
    e: &Engine,
    experts: usize,
    hidden: usize,
    ff: usize,
    selected: usize,
    t: usize,
    reps: usize,
    arms: &[String],
) -> Res<Vec<SelShapeRow>> {
    if hidden % 32 != 0 || ff % 4 != 0 {
        return Err("sel_shape_cost_probe: needs the gufuse geometry (hidden%32, ff%4)".into());
    }
    if selected == 0 || t == 0 || reps == 0 || arms.is_empty() {
        return Err("sel_shape_cost_probe: selected/t/reps/arms must be non-empty".into());
    }
    let saved = sel_group_spec();
    let out = sel_shape_cost_probe_inner(e, experts, hidden, ff, selected, t, reps, arms);
    set_sel_group(&saved);
    out
}

#[allow(clippy::too_many_arguments)]
fn sel_shape_cost_probe_inner(
    e: &Engine,
    experts: usize,
    hidden: usize,
    ff: usize,
    selected: usize,
    t: usize,
    reps: usize,
    arms: &[String],
) -> Res<Vec<SelShapeRow>> {
    // Bank fill and the honesty notes are `moe_union_cost_probe`'s, deliberately: codes
    // index a 16-entry LUT so every byte is legal, scale bytes sit in a mid ue4m3 range so
    // no product leaves normal f32 range, and gate/up get SEPARATE allocations (aliasing
    // them would halve the distinct bytes). No output is compared to an oracle here — that
    // is `gate_nvfp4_sel_group`'s job; this is a latency arm only.
    let code_byte = |i: usize| -> u8 { (i.wrapping_mul(2_654_435_761) >> 13) as u8 };
    let scale_byte = |i: usize| -> u8 { 0x38 | ((i.wrapping_mul(40_503) >> 7) & 0x07) as u8 };
    let mk = |n: usize, f: &dyn Fn(usize) -> u8| -> Res<CudaSlice<u8>> {
        let host: Vec<u8> = (0..n).map(f).collect();
        let d = e.htod_bytes(&host)?;
        drop(host);
        Ok(d)
    };
    let gc = mk(experts * ff * (hidden / 2), &code_byte)?;
    let gs = mk(experts * ff * (hidden / 16), &scale_byte)?;
    let uc = mk(experts * ff * (hidden / 2), &|i| code_byte(i ^ 0x5A5A_5A5A))?;
    let us = mk(experts * ff * (hidden / 16), &|i| {
        scale_byte(i ^ 0x3C3C_3C3C)
    })?;
    let dc = mk(experts * hidden * (ff / 2), &|i| code_byte(i ^ 0x0F0F_0F0F))?;
    let ds = mk(experts * hidden * (ff / 16), &|i| {
        scale_byte(i ^ 0x1111_1111)
    })?;
    let gm = e.htod(&vec![1.0f32; experts])?;
    let um = e.htod(&vec![1.0f32; experts])?;
    let dm = e.htod(&vec![1.0f32; experts])?;
    let mixed_h: Vec<f32> = (0..t * hidden)
        .map(|i| ((i.wrapping_mul(40_503) % 1000) as f32) / 4000.0 - 0.125)
        .collect();
    let mixed = e.htod(&mixed_h)?;

    // ONE routing shape for every arm: `slots` distinct experts spread across the bank by a
    // fixed stride. Distinct, because a shape change must not be read through a cache-hit
    // difference — the union axis is `moe_union_probe`'s and it is already priced dead.
    let slots = t * selected;
    let stride = (experts / slots.max(1)).max(1);
    let sel_h: Vec<i32> = (0..slots)
        .map(|i| ((i * stride) % experts) as i32)
        .collect();
    let tok_h: Vec<i32> = (0..slots).map(|i| (i / selected) as i32).collect();
    let sel = e.htod_i32(&sel_h)?;
    let tokm = e.htod_i32(&tok_h)?;
    let mut act = e.zeros(slots * ff)?;
    let mut partial = e.zeros(slots * hidden)?;

    struct Arm {
        spec: String,
        gu_shape: String,
        dn_shape: String,
        gu_grid_x: usize,
        dn_grid_x: usize,
        gu: Vec<f64>,
        dn: Vec<f64>,
    }
    let describe = |code: u32, in_f: usize, out_f: usize| -> (String, usize) {
        match sel_group_resolve(code, in_f, out_f) {
            Some((g, rows)) => {
                let rpw = (32 / g) * rows;
                (format!("g{g}r{rows}/rpw{rpw}"), out_f / rpw)
            }
            None => ("shipped".to_string(), out_f / 4),
        }
    };
    let mut built: Vec<Arm> = Vec::with_capacity(arms.len());
    for spec in arms {
        if !set_sel_group(spec) {
            return Err(format!("sel_shape_cost_probe: bad arm spec {spec:?}").into());
        }
        let (gu_shape, gu_grid_x) = describe(sel_group_gu(), hidden, ff);
        let (dn_shape, dn_grid_x) = describe(sel_group_dn(), ff, hidden);
        built.push(Arm {
            spec: spec.clone(),
            gu_shape,
            dn_shape,
            gu_grid_x,
            dn_grid_x,
            gu: Vec::with_capacity(reps),
            dn: Vec::with_capacity(reps),
        });
    }
    let tok_arg = if t > 1 { Some((&tokm, hidden)) } else { None };
    for rep in 0..(reps + 1) {
        for a in built.iter_mut() {
            // Arm identity is re-asserted every rep, not set once outside the loop: the
            // interleave is the whole point, and a seam left over from the previous arm
            // would silently measure it twice.
            set_sel_group(&a.spec);
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            launch_nvfp4_sel_gu_silu(
                e,
                (&gc, &gs, &gm),
                (&uc, &us, &um),
                Some(&sel),
                0,
                slots,
                &mixed,
                &mut act,
                hidden,
                ff,
                tok_arg,
            )?;
            e.stream().synchronize()?;
            let t1 = std::time::Instant::now();
            launch_nvfp4_sel_matvec(
                e,
                &dc,
                &ds,
                &dm,
                &sel,
                &act,
                &mut partial,
                slots,
                ff,
                hidden,
                ff,
            )?;
            e.stream().synchronize()?;
            let t2 = std::time::Instant::now();
            if rep > 0 {
                a.gu.push(t1.duration_since(t0).as_secs_f64() * 1e6);
                a.dn.push(t2.duration_since(t1).as_secs_f64() * 1e6);
            }
        }
    }
    let stat = |v: &[f64]| -> (f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = s[s.len() / 2];
        let spread = if med > 0.0 {
            (s[s.len() - 1] - s[0]) / med
        } else {
            0.0
        };
        (med, spread)
    };
    Ok(built
        .iter()
        .map(|a| {
            let (gu_us, gu_spread_rel) = stat(&a.gu);
            let (down_us, down_spread_rel) = stat(&a.dn);
            SelShapeRow {
                t,
                slots,
                arm: a.spec.clone(),
                gu_shape: a.gu_shape.clone(),
                dn_shape: a.dn_shape.clone(),
                gu_grid_x: a.gu_grid_x,
                dn_grid_x: a.dn_grid_x,
                gu_us,
                down_us,
                gu_spread_rel,
                down_spread_rel,
            }
        })
        .collect())
}

/// Sequential slot-combine over a WINDOW of a taller partial slab (mtp-spec verify):
/// rows [x_row0, x_row0+n_rows) x weights [w_off..] into y row `y_row` — the
/// axpy_rows_seq_f32 chain VERBATIM over that window (per-token combine order equals
/// the decode combine).
#[allow(clippy::too_many_arguments)]
fn launch_axpy_rows_seq_at(
    e: &Engine,
    x: &CudaSlice<f32>,
    x_row0: usize,
    w: &CudaSlice<f32>,
    w_off: usize,
    y: &mut CudaSlice<f32>,
    y_row: usize,
    width: usize,
    n_rows: usize,
) -> Res<()> {
    if x.len() < (x_row0 + n_rows) * width
        || w.len() < w_off + n_rows
        || y.len() < (y_row + 1) * width
    {
        return Err("axpy_rows_seq_f32: window out of range".into());
    }
    let xv = x.slice(x_row0 * width..(x_row0 + n_rows) * width);
    let wv = w.slice(w_off..w_off + n_rows);
    let mut yv = y.slice_mut(y_row * width..(y_row + 1) * width);
    let f = e.func("axpy_rows_seq_f32");
    let cfg = LaunchConfig::for_num_elems(width as u32);
    let (wi, nr) = (width as i32, n_rows as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(&xv).arg(&wv).arg(&mut yv).arg(&wi).arg(&nr);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Kernel-vs-host oracle for the grouped decode kernel (`qmatvec_nvfp4_modelopt_sel_f32`).
/// The tiny four-arm gate cannot reach that kernel (the tiny down projection is BF16 by
/// geometry, so the grouped path never engages there); this synthetic arm gates the
/// kernel directly against the host decoder chain (`dsv4::dequant_nvfp4_expert` + host
/// f32 matvec): deterministic codes/scales including planted NaN scale bytes (modelopt
/// NaN -> 0.0) , mixed pow2/non-pow2 macros (the real mint's class), duplicate slots in
/// `sel`, and BOTH x_stride modes (shared gate/up row, per-slot down rows). Products are
/// exact; only summation order differs from the host chain — tolerance 1e-5 rel.
pub fn gate_nvfp4_sel_matvec(e: &Engine) -> Res<String> {
    let mut lcg = 0x2545_f491_u64;
    let mut next_u32 = move || -> u32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (lcg >> 33) as u32
    };
    let macros = [
        1.0f32,
        0.5,
        5.9945243e-5, // the measured non-pow2 mint class
        2.0,
        0.25,
        3.7e-3,
        1.0,
        8.0,
    ];
    let sel_host: Vec<i32> = vec![3, 5, 3, 0]; // duplicate slot on purpose
    let n_sel = sel_host.len();
    let mut worst = (0.0f32, 0.0f32); // (max_abs, max_rel)
    // Shapes + per-mode seam forcing pick the dispatched kernel: v3 modes force the
    // 4-row kernel (its guard is out_f % 4 == 0, which the v2 shapes also satisfy, so
    // the seam is toggled per mode and restored to the shipped default after); v2
    // shapes take the 2-row kernel with v3 off; in_f 48 and the odd out_f take the v1
    // fallback — all three kernels and every geometry guard are gated in one pass.
    for (mode, out_f, in_f) in [
        ("gate_up_v1", 16usize, 48usize),
        ("down_v1", 32, 16),
        ("gate_up_v1_oddrows", 7, 64),
        ("gate_up_v2", 16, 64),
        ("down_v2", 32, 32),
        ("gate_up_v3", 16, 64),
        ("down_v3", 32, 32),
        ("gate_up_v3_v2rows", 6, 64), // out_f % 4 != 0 falls v3 -> v2 under the v3 seam
    ] {
        set_sel_v3(mode.contains("v3"));
        let n_expert = macros.len();
        let mut codes = vec![0u8; n_expert * out_f * in_f / 2];
        for byte in &mut codes {
            *byte = next_u32() as u8;
        }
        let mut scales = vec![0u8; n_expert * out_f * in_f / 16];
        for byte in &mut scales {
            *byte = (next_u32() as u8) & 0xBF; // mag < 0x40 keeps magnitudes tame
        }
        scales[0] = 0x7F; // NaN code -> 0.0 (modelopt convention), pinned here
        scales[3] = 0xFF; // signed NaN code -> 0.0 too
        let x_stride = if mode.starts_with("down") { in_f } else { 0 };
        let x_rows = if x_stride == 0 { 1 } else { n_sel };
        let x_host: Vec<f32> = (0..x_rows * in_f)
            .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let codes_dev = e.htod_bytes(&codes)?;
        let scales_dev = e.htod_bytes(&scales)?;
        let macros_dev = e.htod(&macros)?;
        let sel_dev = e.htod_i32(&sel_host)?;
        let x_dev = e.htod(&x_host)?;
        let mut y_dev = e.uninit(n_sel * out_f)?;
        launch_nvfp4_sel_matvec(
            e,
            &codes_dev,
            &scales_dev,
            &macros_dev,
            &sel_dev,
            &x_dev,
            &mut y_dev,
            n_sel,
            in_f,
            out_f,
            x_stride,
        )?;
        let y = e.dtoh(&y_dev)?;
        let wbytes = out_f * in_f / 2;
        let sbytes = out_f * in_f / 16;
        for (slot, &expert) in sel_host.iter().enumerate() {
            let expert = expert as usize;
            let w = memra_gguf::dsv4::dequant_nvfp4_expert(
                &codes[expert * wbytes..(expert + 1) * wbytes],
                &scales[expert * sbytes..(expert + 1) * sbytes],
                macros[expert],
                out_f,
                in_f,
            );
            let xrow = &x_host[slot * x_stride..slot * x_stride + in_f];
            for o in 0..out_f {
                let mut want = 0.0f32;
                for i in 0..in_f {
                    want += w[o * in_f + i] * xrow[i];
                }
                let got = y[slot * out_f + o];
                let abs = (want - got).abs();
                let rel = abs / want.abs().max(1.0);
                if abs > worst.0 {
                    worst.0 = abs;
                }
                if rel > worst.1 {
                    worst.1 = rel;
                }
                if rel > 1e-5 {
                    return Err(format!(
                        "nvfp4-sel-matvec oracle: {mode} slot {slot} row {o}: want {want} \
                         got {got} (rel {rel:.3e})"
                    )
                    .into());
                }
            }
        }
    }
    set_sel_v3(SEL_V3_DEFAULT);

    // gufuse mode: the fused gate+up+silu kernel must be BIT-IDENTICAL to the
    // v3 gate launch + v3 up launch + silu_mul chain (same per-row arithmetic, same
    // epilogue element form — kernel doc). Byte-compare, plus the count-gated pack
    // twin's dead-slot sentinel.
    {
        set_sel_v3(true);
        let (ff, in_f) = (16usize, 64usize);
        let n_expert = macros.len();
        let mut mk = |seed: u8| -> (Vec<u8>, Vec<u8>) {
            let mut codes = vec![0u8; n_expert * ff * in_f / 2];
            for byte in &mut codes {
                *byte = (next_u32() as u8) ^ seed;
            }
            let mut scales = vec![0u8; n_expert * ff * in_f / 16];
            for byte in &mut scales {
                *byte = (next_u32() as u8) & 0xBF;
            }
            scales[1] = 0x7F; // NaN scale byte -> 0.0
            (codes, scales)
        };
        let (g_codes, g_scales) = mk(0x00);
        let (u_codes, u_scales) = mk(0x5A);
        let gmac: Vec<f32> = macros.to_vec();
        let umac: Vec<f32> = macros.iter().map(|m| m * 0.5).collect();
        let x_host: Vec<f32> = (0..in_f)
            .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let gc = e.htod_bytes(&g_codes)?;
        let gs = e.htod_bytes(&g_scales)?;
        let gm = e.htod(&gmac)?;
        let uc = e.htod_bytes(&u_codes)?;
        let us = e.htod_bytes(&u_scales)?;
        let um = e.htod(&umac)?;
        let sel_dev = e.htod_i32(&sel_host)?;
        let x_dev = e.htod(&x_host)?;
        // Chain arm: v3 gate + v3 up + silu_mul.
        let mut yg = e.uninit(n_sel * ff)?;
        let mut yu = e.uninit(n_sel * ff)?;
        launch_nvfp4_sel_matvec(
            e, &gc, &gs, &gm, &sel_dev, &x_dev, &mut yg, n_sel, in_f, ff, 0,
        )?;
        launch_nvfp4_sel_matvec(
            e, &uc, &us, &um, &sel_dev, &x_dev, &mut yu, n_sel, in_f, ff, 0,
        )?;
        let mut act_chain = e.zeros(n_sel * ff)?;
        e.silu_mul(&yg, &yu, &mut act_chain, n_sel * ff)?;
        // Fused arm.
        let mut act_fused = e.zeros(n_sel * ff)?;
        launch_nvfp4_sel_gu_silu(
            e,
            (&gc, &gs, &gm),
            (&uc, &us, &um),
            Some(&sel_dev),
            0,
            n_sel,
            &x_dev,
            &mut act_fused,
            in_f,
            ff,
            None,
        )?;
        let a = e.dtoh(&act_chain)?;
        let b = e.dtoh(&act_fused)?;
        for (i, (&x1, &x2)) in a.iter().zip(&b).enumerate() {
            if x1.to_bits() != x2.to_bits() {
                return Err(format!(
                    "nvfp4-sel-matvec oracle: gufuse idx {i} not bit-identical \
                     (chain {x1} fused {x2})"
                )
                .into());
            }
        }
        // Pack twin: live count 2 of 4 — live slots bit-match, dead slots keep the
        // sentinel.
        let pack_bytes = tp2_pack_bytes(&sel_host[..2], &[0.5, 0.25], n_sel);
        let pack = e.htod_bytes(&pack_bytes)?;
        let pack_raw = {
            let stream = e.gpu.stream();
            pack.device_ptr(&stream).0
        };
        let sentinel = vec![-777.0f32; n_sel * ff];
        let mut act_pack = e.htod(&sentinel)?;
        launch_nvfp4_sel_gu_silu(
            e,
            (&gc, &gs, &gm),
            (&uc, &us, &um),
            None,
            pack_raw,
            n_sel,
            &x_dev,
            &mut act_pack,
            in_f,
            ff,
            None,
        )?;
        let c = e.dtoh(&act_pack)?;
        for slot in 0..n_sel {
            for o in 0..ff {
                let got = c[slot * ff + o];
                if slot < 2 {
                    if got.to_bits() != a[slot * ff + o].to_bits() {
                        return Err(format!(
                            "nvfp4-sel-matvec oracle: gufuse pack slot {slot} o {o} \
                             not bit-identical"
                        )
                        .into());
                    }
                } else if got != -777.0 {
                    return Err(format!(
                        "nvfp4-sel-matvec oracle: gufuse pack dead slot {slot} written"
                    )
                    .into());
                }
            }
        }
        // tok_map twin (mtp-spec verify merge): TWO tokens' slots in ONE launch via the
        // slot->token map must bit-match per-token launches over each token's x row.
        {
            let t2 = 2usize;
            let x2_host: Vec<f32> = (0..t2 * in_f)
                .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let x2 = e.htod(&x2_host)?;
            let tok_host: Vec<i32> = (0..n_sel).map(|s| (s % t2) as i32).collect();
            let tokm = e.htod_i32(&tok_host)?;
            let mut act_map = e.zeros(n_sel * ff)?;
            launch_nvfp4_sel_gu_silu(
                e,
                (&gc, &gs, &gm),
                (&uc, &us, &um),
                Some(&sel_dev),
                0,
                n_sel,
                &x2,
                &mut act_map,
                in_f,
                ff,
                Some((&tokm, in_f)),
            )?;
            let got = e.dtoh(&act_map)?;
            for tok in 0..t2 {
                let slots: Vec<usize> = (0..n_sel).filter(|s| s % t2 == tok).collect();
                let sel_tok: Vec<i32> = slots.iter().map(|&s| sel_host[s]).collect();
                let sel_tok_dev = e.htod_i32(&sel_tok)?;
                let xrow = e.htod(&x2_host[tok * in_f..(tok + 1) * in_f])?;
                let mut act_tok = e.zeros(sel_tok.len() * ff)?;
                launch_nvfp4_sel_gu_silu(
                    e,
                    (&gc, &gs, &gm),
                    (&uc, &us, &um),
                    Some(&sel_tok_dev),
                    0,
                    sel_tok.len(),
                    &xrow,
                    &mut act_tok,
                    in_f,
                    ff,
                    None,
                )?;
                let want = e.dtoh(&act_tok)?;
                for (local, &slot) in slots.iter().enumerate() {
                    for o in 0..ff {
                        let a = got[slot * ff + o];
                        let b = want[local * ff + o];
                        if a.to_bits() != b.to_bits() {
                            return Err(format!(
                                "nvfp4-sel-matvec oracle: gufuse tok_map slot {slot} o {o} \
                                 not bit-identical (map {a} per-token {b})"
                            )
                            .into());
                        }
                    }
                }
            }
        }
        set_sel_v3(SEL_V3_DEFAULT);
    }
    Ok(format!(
        "nvfp4-sel-matvec kernel oracle: worst abs {:.3e} rel {:.3e} over gate_up+down \
         v1/v2/v3 modes, NaN scales + non-pow2 macros + duplicate slots; gufuse \
         BIT-IDENTICAL to the v3+silu chain incl. the count-gated pack twin + the \
         tok_map verify merge",
        worst.0, worst.1
    ))
}

/// Kernel oracle for the SUB-WARP pair-group sel matvecs (`selgroup`, downsel lane mtp14),
/// at the artifact's REAL MoE geometry — which is the whole point of the arm: the defect
/// being fixed is a property of `pairs = in_f/32` against a 32-lane loop, so it only exists
/// at `in_f = 640` (pairs 20, lanes 20-31 idle) and `in_f = 2560` (pairs 80, 3-vs-2 tail).
/// A tiny fixture has `pairs` 1 or 2 and cannot reach either shape; both are gated here.
///
/// Three claims, in ascending strength:
///
/// 1. **`(g=32, rows=4)` is BIT-IDENTICAL to the shipped v3 / gufuse kernels.** The
///    sub-warp form degenerates to their exact program at that shape (same per-lane pair
///    set, same 5-step tree, same write lane), so this is a byte compare, not a tolerance.
///    It is what makes the seam a rollback rather than a rewrite, and it is the arm that
///    would catch a per-row expression drift introduced while restructuring.
/// 2. **Every other shape is within the sel oracle's accumulation-class tolerance
///    (1e-5 rel) of the HOST DECODER CHAIN** (`dsv4::dequant_nvfp4_expert` + host f32
///    matvec), the same reference and the same bound `gate_nvfp4_sel_matvec` holds v1/v2/v3
///    to. Those shapes DO change the order the pairs are summed in — a lane chains several
///    pairs and the tree is shallower — so bit-identity is not the right claim and asserting
///    it would be a lie that happened to pass at some shapes.
/// 3. **The fusion property survives the reshape:** `gu_g` is bit-identical to
///    `sel_g` gate + `sel_g` up + `silu_mul` at the SAME `(g, rows)`, with the count-gated
///    pack twin and the slot->token verify merge included.
///
/// Same hostile inputs as the shipped arm: planted modelopt NaN scale bytes (0x7F/0xFF ->
/// 0.0), mixed pow2 / non-pow2 (the real mint's amax class) macros, a DUPLICATE expert in
/// `sel`, and both `x_stride` modes (shared gate/up row, per-slot down rows).
pub fn gate_nvfp4_sel_group(e: &Engine) -> Res<String> {
    let saved = sel_group_spec();
    let out = gate_nvfp4_sel_group_inner(e);
    // Restore on BOTH paths: a gate arm that leaks a seam leaves every later arm measuring
    // a shape nobody asked for (the seam_state save/restore lesson).
    set_sel_group(&saved);
    set_sel_v3(SEL_V3_DEFAULT);
    out
}

fn gate_nvfp4_sel_group_inner(e: &Engine) -> Res<String> {
    let mut lcg = 0x2545_f491_u64; // the shipped sel arm's seed, deliberately
    let mut next_u32 = move || -> u32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (lcg >> 33) as u32
    };
    let macros = [
        1.0f32,
        0.5,
        5.9945243e-5, // the measured non-pow2 mint class
        2.0,
        0.25,
        3.7e-3,
        1.0,
        8.0,
    ];
    let n_expert = macros.len();
    let sel_host: Vec<i32> = vec![3, 5, 3, 0]; // duplicate slot on purpose
    let n_sel = sel_host.len();
    let mut worst = (0.0f32, 0.0f32);
    let mut shapes_checked = 0usize;
    let mut bits_checked = 0usize;
    let mut calib: Vec<String> = Vec::new();

    // ---- single-bank family (down projection AND the unfused gate/up shape) -------------
    // (label, out_f, in_f, per-slot x rows). The two REAL rows are the launches the verify
    // chunk actually dispatches: down out_f=hidden 2560 / in_f=ff 640, and the gate/up
    // shape out_f=ff 640 / in_f=hidden 2560 (SEMANTICS.md "MoE (L510-527)": experts fused
    // gate_up [512,1280,2560], down [512,2560,640]).
    for (geom, out_f, in_f, per_slot_x) in [
        ("down_real", 2560usize, 640usize, true),
        ("gateup_real", 640, 2560, false),
        ("down_tiny", 32, 32, true),
        ("gateup_tiny", 16, 64, false),
    ] {
        let mut codes = vec![0u8; n_expert * out_f * in_f / 2];
        for byte in &mut codes {
            *byte = next_u32() as u8;
        }
        let mut scales = vec![0u8; n_expert * out_f * in_f / 16];
        for byte in &mut scales {
            *byte = (next_u32() as u8) & 0xBF; // mag < 0x40 keeps magnitudes tame
        }
        scales[0] = 0x7F; // modelopt NaN code -> 0.0
        scales[3] = 0xFF; // signed NaN code -> 0.0 too
        let x_stride = if per_slot_x { in_f } else { 0 };
        let x_rows = if per_slot_x { n_sel } else { 1 };
        let x_host: Vec<f32> = (0..x_rows * in_f)
            .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let codes_dev = e.htod_bytes(&codes)?;
        let scales_dev = e.htod_bytes(&scales)?;
        let macros_dev = e.htod(&macros)?;
        let sel_dev = e.htod_i32(&sel_host)?;
        let x_dev = e.htod(&x_host)?;
        let run = |spec: &str| -> Res<Vec<f32>> {
            set_sel_group(spec);
            let mut y = e.uninit(n_sel * out_f)?;
            launch_nvfp4_sel_matvec(
                e,
                &codes_dev,
                &scales_dev,
                &macros_dev,
                &sel_dev,
                &x_dev,
                &mut y,
                n_sel,
                in_f,
                out_f,
                x_stride,
            )?;
            e.dtoh(&y)
        };
        // The shipped arm (seam OFF) and the host reference, built once per geometry. The
        // shipped arm is not just a bit-identity control: its OWN deviation from the host
        // chain is this geometry's calibration (see `class_tol` below). PIN sel_v3 rather
        // than inheriting ambient seam state (revuto, PR #27): under `selv3=0` in
        // MEMRA_Q4E_SEAMS the "shipped" control would silently become the v2 kernel and
        // the calibration would be measured against the wrong program — mirror the fused
        // family, which pins its control the same way.
        set_sel_v3(true);
        let shipped = run("off")?;
        let wbytes = out_f * in_f / 2;
        let sbytes = out_f * in_f / 16;
        let mut want = vec![0.0f32; n_sel * out_f];
        for (slot, &expert) in sel_host.iter().enumerate() {
            let expert = expert as usize;
            let w = memra_gguf::dsv4::dequant_nvfp4_expert(
                &codes[expert * wbytes..(expert + 1) * wbytes],
                &scales[expert * sbytes..(expert + 1) * sbytes],
                macros[expert],
                out_f,
                in_f,
            );
            let xrow = &x_host[slot * x_stride..slot * x_stride + in_f];
            for o in 0..out_f {
                let mut acc = 0.0f32;
                for i in 0..in_f {
                    acc += w[o * in_f + i] * xrow[i];
                }
                want[slot * out_f + o] = acc;
            }
        }
        // The SHIPPED kernel's own worst deviation from the host chain, at THIS width. This
        // is the arm's calibration, and measuring it is load-bearing rather than tidy:
        // `gate_nvfp4_sel_matvec`'s 1e-5 rel bound was set on TINY shapes (in_f 16-64) and
        // does NOT transfer to the real MoE widths — a length-`in_f` f32 reduction has an
        // order-dependent error that grows with the sum, and at in_f=640 the SHIPPED v3
        // kernel already measures ~2.7e-5 against the exact host chain. Holding a reshaped
        // twin to 1e-5 there would fail it for being a different (equally valid) summation
        // order of a sum the shipped kernel cannot hold to 1e-5 either.
        let ship_vs_host = want
            .iter()
            .zip(&shipped)
            .map(|(&w, &s)| (w - s).abs() / w.abs().max(1.0))
            .fold(0.0f32, f32::max);
        // Same-accumulation-class bound: no worse than 4x what the kernel we ship already
        // deviates by, with a floor so the tiny geometries (where the shipped kernel can be
        // near-exact) do not set an unreachable bar.
        let class_tol = (4.0 * ship_vs_host).max(1e-5);
        calib.push(format!(
            "{geom} ship_vs_host={ship_vs_host:.3e} tol={class_tol:.3e}"
        ));
        // Every shape the ladder can pin at this geometry, plus AUTO and the control.
        // `dn:1:1` is the extreme: one output row per LANE, no shfl reduce at all — kept in
        // the oracle because it is the arm most likely to expose an indexing error, even
        // though its coalescing makes it a poor perf candidate.
        for spec in [
            "dn:32:4", "dn:auto", "dn:16:4", "dn:16:2", "dn:8:4", "dn:8:2", "dn:8:1", "dn:4:4",
            "dn:4:2", "dn:4:1", "dn:2:4", "dn:2:2", "dn:2:1", "dn:1:1",
        ] {
            let Some((g, rows)) = sel_group_resolve(
                match spec {
                    "dn:auto" => SEL_GROUP_AUTO,
                    _ => {
                        let (gs, rs) = spec.trim_start_matches("dn:").split_once(':').unwrap();
                        (gs.parse::<u32>().unwrap() << 8) | rs.parse::<u32>().unwrap()
                    }
                },
                in_f,
                out_f,
            ) else {
                continue; // geometry cannot tile this shape — the launcher takes v3
            };
            let got = run(spec)?;
            shapes_checked += 1;
            if (g, rows) == (32, 4) {
                // Claim 1: the degenerate shape IS v3.
                for (i, (&a, &b)) in shipped.iter().zip(&got).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        return Err(format!(
                            "sel-group oracle: {geom} g=32 rows=4 idx {i} NOT bit-identical to \
                             the shipped v3 kernel (v3 {a} group {b}) — the sub-warp form must \
                             degenerate to v3 exactly"
                        )
                        .into());
                    }
                }
                bits_checked += shipped.len();
            }
            // Claim 2: same accumulation class as the kernel we ship. Checked BOTH ways —
            // against the exact host chain, and against the shipped kernel's own output.
            // The second is the one that would catch a reshape that drifted while staying
            // coincidentally close to the reference.
            for (i, (&w, &got)) in want.iter().zip(&got).enumerate() {
                let abs = (w - got).abs();
                let rel = abs / w.abs().max(1.0);
                worst.0 = worst.0.max(abs);
                worst.1 = worst.1.max(rel);
                if rel > class_tol {
                    return Err(format!(
                        "sel-group oracle: {geom} {spec} (g={g} rows={rows}) idx {i} vs HOST \
                         chain: want {w} got {got} (rel {rel:.3e} > tol {class_tol:.3e}, \
                         shipped v3 itself is {ship_vs_host:.3e})"
                    )
                    .into());
                }
            }
            for (i, (&s, &got)) in shipped.iter().zip(&got).enumerate() {
                let rel = (s - got).abs() / s.abs().max(1.0);
                if rel > class_tol {
                    return Err(format!(
                        "sel-group oracle: {geom} {spec} (g={g} rows={rows}) idx {i} vs SHIPPED \
                         v3: v3 {s} group {got} (rel {rel:.3e} > tol {class_tol:.3e})"
                    )
                    .into());
                }
            }
        }
        set_sel_group("off");
    }

    // ---- fused gate+up+silu family -----------------------------------------------------
    // Claim 3: the fusion survives the reshape. The chain arm runs the SAME (g, rows) on
    // the single-bank kernel, so a mismatch is the fusion breaking, not the shape.
    for (geom, ff, in_f) in [("gu_real", 640usize, 2560usize), ("gu_tiny", 16, 64)] {
        let mut mk = |seed: u8| -> (Vec<u8>, Vec<u8>) {
            let mut codes = vec![0u8; n_expert * ff * in_f / 2];
            for byte in &mut codes {
                *byte = (next_u32() as u8) ^ seed;
            }
            let mut scales = vec![0u8; n_expert * ff * in_f / 16];
            for byte in &mut scales {
                *byte = (next_u32() as u8) & 0xBF;
            }
            scales[1] = 0x7F; // NaN scale byte -> 0.0
            (codes, scales)
        };
        let (g_codes, g_scales) = mk(0x00);
        let (u_codes, u_scales) = mk(0x5A);
        let gmac: Vec<f32> = macros.to_vec();
        let umac: Vec<f32> = macros.iter().map(|m| m * 0.5).collect();
        let x_host: Vec<f32> = (0..in_f)
            .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let gc = e.htod_bytes(&g_codes)?;
        let gs = e.htod_bytes(&g_scales)?;
        let gm = e.htod(&gmac)?;
        let uc = e.htod_bytes(&u_codes)?;
        let us = e.htod_bytes(&u_scales)?;
        let um = e.htod(&umac)?;
        let sel_dev = e.htod_i32(&sel_host)?;
        let x_dev = e.htod(&x_host)?;
        set_sel_group("off");
        set_sel_v3(true);
        let shipped_fused = {
            let mut act = e.zeros(n_sel * ff)?;
            launch_nvfp4_sel_gu_silu(
                e,
                (&gc, &gs, &gm),
                (&uc, &us, &um),
                Some(&sel_dev),
                0,
                n_sel,
                &x_dev,
                &mut act,
                in_f,
                ff,
                None,
            )?;
            e.dtoh(&act)?
        };
        for spec in ["32:4", "auto", "16:4", "16:2", "8:4", "8:1", "4:4"] {
            let Some((g, rows)) = sel_group_resolve(
                match spec {
                    "auto" => SEL_GROUP_AUTO,
                    _ => {
                        let (gs, rs) = spec.split_once(':').unwrap();
                        (gs.parse::<u32>().unwrap() << 8) | rs.parse::<u32>().unwrap()
                    }
                },
                in_f,
                ff,
            ) else {
                continue;
            };
            // Chain arm at the same shape: sel_g(gate) + sel_g(up) + silu_mul.
            set_sel_group(&format!("dn:{spec}+gu:off"));
            let mut yg = e.uninit(n_sel * ff)?;
            let mut yu = e.uninit(n_sel * ff)?;
            launch_nvfp4_sel_matvec(
                e, &gc, &gs, &gm, &sel_dev, &x_dev, &mut yg, n_sel, in_f, ff, 0,
            )?;
            launch_nvfp4_sel_matvec(
                e, &uc, &us, &um, &sel_dev, &x_dev, &mut yu, n_sel, in_f, ff, 0,
            )?;
            let mut act_chain = e.zeros(n_sel * ff)?;
            e.silu_mul(&yg, &yu, &mut act_chain, n_sel * ff)?;
            let chain = e.dtoh(&act_chain)?;
            // Fused arm at the same shape.
            set_sel_group(&format!("dn:off+gu:{spec}"));
            let mut act_fused = e.zeros(n_sel * ff)?;
            launch_nvfp4_sel_gu_silu(
                e,
                (&gc, &gs, &gm),
                (&uc, &us, &um),
                Some(&sel_dev),
                0,
                n_sel,
                &x_dev,
                &mut act_fused,
                in_f,
                ff,
                None,
            )?;
            let fused = e.dtoh(&act_fused)?;
            for (i, (&a, &b)) in chain.iter().zip(&fused).enumerate() {
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "sel-group oracle: {geom} gu {spec} (g={g} rows={rows}) idx {i} fused \
                         NOT bit-identical to the same-shape chain (chain {a} fused {b})"
                    )
                    .into());
                }
            }
            bits_checked += chain.len();
            shapes_checked += 1;
            if (g, rows) == (32, 4) {
                for (i, (&a, &b)) in shipped_fused.iter().zip(&fused).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        return Err(format!(
                            "sel-group oracle: {geom} gu g=32 rows=4 idx {i} NOT bit-identical \
                             to the shipped gufuse kernel (gufuse {a} group {b})"
                        )
                        .into());
                    }
                }
                bits_checked += shipped_fused.len();
            }
        }
        // Count-gated pack twin and the slot->token verify merge, under AUTO — the two
        // addressing modes the serving path uses that the plain arm above does not reach.
        set_sel_group("dn:off+gu:auto");
        if sel_group_resolve(SEL_GROUP_AUTO, in_f, ff).is_some() {
            let auto_plain = {
                let mut act = e.zeros(n_sel * ff)?;
                launch_nvfp4_sel_gu_silu(
                    e,
                    (&gc, &gs, &gm),
                    (&uc, &us, &um),
                    Some(&sel_dev),
                    0,
                    n_sel,
                    &x_dev,
                    &mut act,
                    in_f,
                    ff,
                    None,
                )?;
                e.dtoh(&act)?
            };
            let pack_bytes = tp2_pack_bytes(&sel_host[..2], &[0.5, 0.25], n_sel);
            let pack = e.htod_bytes(&pack_bytes)?;
            let pack_raw = {
                let stream = e.gpu.stream();
                pack.device_ptr(&stream).0
            };
            let sentinel = vec![-777.0f32; n_sel * ff];
            let mut act_pack = e.htod(&sentinel)?;
            launch_nvfp4_sel_gu_silu(
                e,
                (&gc, &gs, &gm),
                (&uc, &us, &um),
                None,
                pack_raw,
                n_sel,
                &x_dev,
                &mut act_pack,
                in_f,
                ff,
                None,
            )?;
            let packed = e.dtoh(&act_pack)?;
            for slot in 0..n_sel {
                for o in 0..ff {
                    let got = packed[slot * ff + o];
                    if slot < 2 {
                        if got.to_bits() != auto_plain[slot * ff + o].to_bits() {
                            return Err(format!(
                                "sel-group oracle: {geom} gu auto pack slot {slot} o {o} not \
                                 bit-identical to the sel-array arm"
                            )
                            .into());
                        }
                    } else if got != -777.0 {
                        return Err(format!(
                            "sel-group oracle: {geom} gu auto pack dead slot {slot} written"
                        )
                        .into());
                    }
                }
            }
            // tok_map: two tokens' slots in ONE launch must bit-match per-token launches.
            let t2 = 2usize;
            let x2_host: Vec<f32> = (0..t2 * in_f)
                .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let x2 = e.htod(&x2_host)?;
            let tok_host: Vec<i32> = (0..n_sel).map(|s| (s % t2) as i32).collect();
            let tokm = e.htod_i32(&tok_host)?;
            let mut act_map = e.zeros(n_sel * ff)?;
            launch_nvfp4_sel_gu_silu(
                e,
                (&gc, &gs, &gm),
                (&uc, &us, &um),
                Some(&sel_dev),
                0,
                n_sel,
                &x2,
                &mut act_map,
                in_f,
                ff,
                Some((&tokm, in_f)),
            )?;
            let mapped = e.dtoh(&act_map)?;
            for tok in 0..t2 {
                let slots: Vec<usize> = (0..n_sel).filter(|s| s % t2 == tok).collect();
                let sel_tok: Vec<i32> = slots.iter().map(|&s| sel_host[s]).collect();
                let sel_tok_dev = e.htod_i32(&sel_tok)?;
                let xrow = e.htod(&x2_host[tok * in_f..(tok + 1) * in_f])?;
                let mut act_tok = e.zeros(sel_tok.len() * ff)?;
                launch_nvfp4_sel_gu_silu(
                    e,
                    (&gc, &gs, &gm),
                    (&uc, &us, &um),
                    Some(&sel_tok_dev),
                    0,
                    sel_tok.len(),
                    &xrow,
                    &mut act_tok,
                    in_f,
                    ff,
                    None,
                )?;
                let want = e.dtoh(&act_tok)?;
                for (local, &slot) in slots.iter().enumerate() {
                    for o in 0..ff {
                        let a = mapped[slot * ff + o];
                        let b = want[local * ff + o];
                        if a.to_bits() != b.to_bits() {
                            return Err(format!(
                                "sel-group oracle: {geom} gu auto tok_map slot {slot} o {o} not \
                                 bit-identical (map {a} per-token {b})"
                            )
                            .into());
                        }
                    }
                }
            }
            bits_checked += auto_plain.len() + mapped.len();
        }
        set_sel_group("off");
    }

    Ok(format!(
        "nvfp4-sel-GROUP kernel oracle: {shapes_checked} (geometry, shape) cells over REAL \
         MoE geometry (down 2560x640 pairs=20, gate_up 640x2560 pairs=80) + tiny, worst abs \
         {:.3e} rel {:.3e} vs the host decoder chain; (g=32,rows=4) BIT-IDENTICAL to the \
         shipped v3 and gufuse kernels and every shape's fused arm BIT-IDENTICAL to its \
         same-shape chain ({bits_checked} f32 byte-compared), incl. the count-gated pack \
         twin + the tok_map verify merge; NaN scales + non-pow2 macros + duplicate slots; \
         per-geometry class calibration [{}]",
        worst.0,
        worst.1,
        calib.join("; ")
    ))
}

/// REAL-GEOMETRY oracle for the round-4 hyper-gate diet (the tiny plan's rank 4 fails
/// the %8 geometry guard, so the tiny arms never reach these kernels): the THREE-launch
/// diet chain (stage 1/2/3) vs the classic fused chain (hc_norm_planes + batched bf16w
/// down + lowrank reduce + batched bf16w up + mix epilogue + two-stage inject) on
/// IDENTICAL bf16 weights at streams 4, hidden 2560, rank 320, t 1. Tolerance class
/// (new reduce widths; 1e-4 rel, worst reported) over low_act, the inject slab, and
/// mixed.
pub fn gate_hc_diet_kernels(e: &Engine) -> Res<String> {
    let (streams, hidden, rank, t) = (4usize, 2560usize, 320usize, 1usize);
    let wide = streams * hidden;
    let mut lcg = 0x8badf00d_u64;
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as u32) % 2000) as f32 / 1000.0 - 1.0
    };
    let mut rand_vec = |n: usize| -> Vec<f32> { (0..n).map(|_| next_f32()).collect() };
    // bf16-representable weights (truncate the low mantissa bits) so bf16_twin builds.
    let to_b16_vals = |v: Vec<f32>| -> Vec<f32> {
        v.into_iter()
            .map(|x| f32::from_bits(x.to_bits() & 0xFFFF_0000))
            .collect()
    };
    let planes_host: Vec<Vec<f32>> = (0..streams).map(|_| rand_vec(t * hidden)).collect();
    let planes: Vec<CudaSlice<f32>> = planes_host
        .iter()
        .map(|v| e.htod(v))
        .collect::<Result<_, _>>()?;
    let ptr_vals: Vec<u64> = {
        let stream = e.gpu.stream();
        planes.iter().map(|p| p.device_ptr(&stream).0).collect()
    };
    let ptrs = e.htod_u64(&ptr_vals)?;
    let norm_stack_host = rand_vec(wide);
    let norm_stack = e.htod(&norm_stack_host)?;
    let down_host = to_b16_vals(rand_vec(streams * rank * hidden));
    let up_host = to_b16_vals(rand_vec(streams * hidden * rank));
    let inj_host = to_b16_vals(rand_vec(streams * wide));
    let down_b16 = bf16_twin(e, &down_host, hidden)?.ok_or("hc-diet oracle: down twin")?;
    let up_b16 = bf16_twin(e, &up_host, rank)?.ok_or("hc-diet oracle: up twin")?;
    let inj_b16 = bf16_twin(e, &inj_host, hidden)?.ok_or("hc-diet oracle: inject twin")?;
    let inj_f32 = e.htod(&inj_host)?;
    let eps = 1e-6f32;

    // Classic fused chain (the current default path) on the same operands.
    let mut normed = e.zeros(streams * t * hidden)?;
    launch_hc_norm_planes(e, &ptrs, &norm_stack, &mut normed, hidden, t, streams, eps)?;
    let mut parts_c = e.zeros(streams * t * rank)?;
    launch_qmatvec_bf16w(
        e,
        &down_b16,
        &normed,
        &mut parts_c,
        hidden,
        rank,
        t,
        streams,
        rank * hidden,
        t * hidden,
        hidden,
        t * rank,
    )?;
    let mut low_c = e.zeros(t * rank)?;
    launch_hc_lowrank_reduce(e, &parts_c, &mut low_c, streams, t, rank)?;
    let mut gates_c = e.zeros(streams * t * hidden)?;
    launch_qmatvec_bf16w(
        e,
        &up_b16,
        &low_c,
        &mut gates_c,
        rank,
        hidden,
        t,
        streams,
        hidden * rank,
        0,
        rank,
        t * hidden,
    )?;
    let mut mixed_c = e.zeros(t * hidden)?;
    launch_hc_mix_epilogue(e, &gates_c, &normed, &mut mixed_c, streams, t, hidden)?;
    let mut partials_c = e.zeros(streams * t * 16)?;
    let mut all_c = e.zeros(streams * t)?;
    launch_hc_inject_two_stage(
        e,
        &normed,
        &inj_f32,
        Some(&inj_b16),
        &mut partials_c,
        &mut all_c,
        streams,
        t,
        hidden,
        16,
    )?;

    // Diet chain.
    let mut parts_d = e.zeros(streams * rank)?;
    let mut injp_d = e.zeros(streams * streams)?;
    let mut inv_d = e.zeros(streams)?;
    launch_hc_diet_stage1(
        e,
        &ptrs,
        &norm_stack,
        &down_b16,
        Some(&inj_b16),
        &mut parts_d,
        &mut injp_d,
        &mut inv_d,
        hidden,
        rank,
        streams,
        1,
        eps,
    )?;
    let mut low_d = e.zeros(rank)?;
    let mut all_d = e.zeros(streams)?;
    launch_hc_diet_stage2(
        e, &parts_d, &injp_d, &mut low_d, &mut all_d, rank, streams, 1, true,
    )?;
    let mut mixed_d = e.zeros(hidden)?;
    launch_hc_diet_stage3(
        e,
        &ptrs,
        &norm_stack,
        &inv_d,
        &up_b16,
        &low_d,
        &mut mixed_d,
        hidden,
        rank,
        streams,
        1,
    )?;

    let mut worst = 0.0f32;
    let check = |name: &str, a: &[f32], b: &[f32], worst: &mut f32| -> Res<()> {
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            let rel = (x - y).abs() / y.abs().max(1.0);
            if rel > *worst {
                *worst = rel;
            }
            if rel > 1e-4 {
                return Err(format!(
                    "hc-diet oracle: {name} idx {i}: diet {x} classic {y} (rel {rel:.3e})"
                )
                .into());
            }
        }
        Ok(())
    };
    check("low_act", &e.dtoh(&low_d)?, &e.dtoh(&low_c)?, &mut worst)?;
    check("inject", &e.dtoh(&all_d)?, &e.dtoh(&all_c)?, &mut worst)?;
    check("mixed", &e.dtoh(&mixed_d)?, &e.dtoh(&mixed_c)?, &mut worst)?;

    // Token-dim extension (mtp-spec verify chunks): the SAME kernels at t = 3 must
    // produce per-token rows BIT-IDENTICAL to three t = 1 launches at plane offsets —
    // the spec byte-identity contract for the read gates.
    {
        let t3 = 3usize;
        let planes3_host: Vec<Vec<f32>> = (0..streams).map(|_| rand_vec(t3 * hidden)).collect();
        let planes3: Vec<CudaSlice<f32>> = planes3_host
            .iter()
            .map(|v| e.htod(v))
            .collect::<Result<_, _>>()?;
        let ptr_vals3: Vec<u64> = {
            let stream = e.gpu.stream();
            planes3.iter().map(|p| p.device_ptr(&stream).0).collect()
        };
        let ptrs3 = e.htod_u64(&ptr_vals3)?;
        let mut parts3 = e.zeros(t3 * streams * rank)?;
        let mut injp3 = e.zeros(t3 * streams * streams)?;
        let mut inv3 = e.zeros(t3 * streams)?;
        launch_hc_diet_stage1(
            e,
            &ptrs3,
            &norm_stack,
            &down_b16,
            Some(&inj_b16),
            &mut parts3,
            &mut injp3,
            &mut inv3,
            hidden,
            rank,
            streams,
            t3,
            eps,
        )?;
        let mut low3 = e.zeros(t3 * rank)?;
        let mut all3 = e.zeros(streams * t3)?;
        launch_hc_diet_stage2(
            e, &parts3, &injp3, &mut low3, &mut all3, rank, streams, t3, true,
        )?;
        let mut mixed3 = e.zeros(t3 * hidden)?;
        launch_hc_diet_stage3(
            e,
            &ptrs3,
            &norm_stack,
            &inv3,
            &up_b16,
            &low3,
            &mut mixed3,
            hidden,
            rank,
            streams,
            t3,
        )?;
        let low3_h = e.dtoh(&low3)?;
        let all3_h = e.dtoh(&all3)?;
        let mixed3_h = e.dtoh(&mixed3)?;
        // MT weight-shared stages (set_verify_mt): stage0 inv + stage1_mt parts +
        // stage3_mt mixed must be BIT-IDENTICAL to the token-grid stages above.
        {
            let mut inv_mt = e.zeros(t3 * streams)?;
            launch_hc_diet_stage0_mt(e, &ptrs3, &mut inv_mt, hidden, streams, t3, eps)?;
            let mut parts_mt = e.zeros(t3 * streams * rank)?;
            let mut injp_mt = e.zeros(t3 * streams * streams)?;
            launch_hc_diet_stage1_mt(
                e,
                &ptrs3,
                &norm_stack,
                &inv_mt,
                &down_b16,
                Some(&inj_b16),
                &mut parts_mt,
                &mut injp_mt,
                hidden,
                rank,
                streams,
                t3,
            )?;
            let mut low_mt = e.zeros(t3 * rank)?;
            let mut all_mt = e.zeros(streams * t3)?;
            launch_hc_diet_stage2(
                e,
                &parts_mt,
                &injp_mt,
                &mut low_mt,
                &mut all_mt,
                rank,
                streams,
                t3,
                true,
            )?;
            let mut mixed_mt = e.zeros(t3 * hidden)?;
            launch_hc_diet_stage3_mt(
                e,
                &ptrs3,
                &norm_stack,
                &inv_mt,
                &up_b16,
                &low_mt,
                &mut mixed_mt,
                hidden,
                rank,
                streams,
                t3,
            )?;
            let bit_check_mt = |name: &str, a: &[f32], b: &[f32]| -> Res<()> {
                for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                    if x.to_bits() != y.to_bits() {
                        return Err(format!(
                            "hc-diet mt oracle: {name} idx {i}: mt {x} vs grid {y} NOT \
                             bit-identical"
                        )
                        .into());
                    }
                }
                Ok(())
            };
            bit_check_mt("inv", &e.dtoh(&inv_mt)?, &e.dtoh(&inv3)?)?;
            bit_check_mt("low_act", &e.dtoh(&low_mt)?, &low3_h)?;
            bit_check_mt("inject", &e.dtoh(&all_mt)?, &all3_h)?;
            bit_check_mt("mixed", &e.dtoh(&mixed_mt)?, &mixed3_h)?;
        }
        let bit_check = |name: &str, a: &[f32], b: &[f32]| -> Res<()> {
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                if x.to_bits() != y.to_bits() {
                    return Err(format!(
                        "hc-diet t-ext oracle: {name} idx {i}: t3 {x} vs t1 {y} NOT bit-identical"
                    )
                    .into());
                }
            }
            Ok(())
        };
        for tok in 0..t3 {
            let ptr_tok: Vec<u64> = ptr_vals3
                .iter()
                .map(|&base| base + (tok * hidden * 4) as u64)
                .collect();
            let ptrs_tok = e.htod_u64(&ptr_tok)?;
            let mut parts1 = e.zeros(streams * rank)?;
            let mut injp1 = e.zeros(streams * streams)?;
            let mut inv1 = e.zeros(streams)?;
            launch_hc_diet_stage1(
                e,
                &ptrs_tok,
                &norm_stack,
                &down_b16,
                Some(&inj_b16),
                &mut parts1,
                &mut injp1,
                &mut inv1,
                hidden,
                rank,
                streams,
                1,
                eps,
            )?;
            let mut low1 = e.zeros(rank)?;
            let mut all1 = e.zeros(streams)?;
            launch_hc_diet_stage2(
                e, &parts1, &injp1, &mut low1, &mut all1, rank, streams, 1, true,
            )?;
            let mut mixed1 = e.zeros(hidden)?;
            launch_hc_diet_stage3(
                e,
                &ptrs_tok,
                &norm_stack,
                &inv1,
                &up_b16,
                &low1,
                &mut mixed1,
                hidden,
                rank,
                streams,
                1,
            )?;
            bit_check(
                "low_act",
                &low3_h[tok * rank..(tok + 1) * rank],
                &e.dtoh(&low1)?,
            )?;
            let all1_h = e.dtoh(&all1)?;
            let col: Vec<f32> = (0..streams).map(|s| all3_h[s * t3 + tok]).collect();
            bit_check("inject", &col, &all1_h)?;
            bit_check(
                "mixed",
                &mixed3_h[tok * hidden..(tok + 1) * hidden],
                &e.dtoh(&mixed1)?,
            )?;
        }
    }
    Ok(format!(
        "hc-diet real-geometry oracle: streams 4 hidden 2560 rank 320, worst rel \
         {worst:.3e} vs the classic fused chain at t 1; t 3 token-dim AND the mt \
         weight-shared stages BIT-IDENTICAL to per-token t 1 launches"
    ))
}

/// Kernel-vs-host oracle for the bf16 trunk matvec (`qmatvec_bf16w_f32`). The tiny
/// four-arm gate's FIXTURE weights are random f32 (never bf16-representable), so its
/// bf16 twins are skipped by the value guard there and only the dir arms exercise the
/// path end to end; this synthetic arm gates the kernel directly against a host f32
/// matvec over identical bf16-widened weights: batch > 1, BOTH x_bstride modes (shared
/// plane like the up projection, per-batch planes like down), t > 1, negative/denormal
/// bf16 values, and a non-multiple-of-blockDim group count. Products are exact; only
/// summation order differs from the sequential host chain — tolerance 1e-5 rel.
/// REAL-GEOMETRY oracle for the hcmicro kernels (streams 4, hidden 2560, t 10 — the
/// artifact's read-gate shape, which the tiny plan (streams 2, hidden 16) cannot
/// reach). Each micro kernel runs against the classic composition it replaces on the
/// same random inputs: batched plane norms vs per-stream rms_norm, the two-stage inject
/// vs the single-stage kernel, the slab write vs the add_scaled_rows chain. Born from
/// the perf7 incident: the bundle shipped tiny-green and broke real prefill at layer 0.
pub fn gate_hc_micro_kernels(e: &Engine) -> Res<String> {
    let (streams, hidden, t) = (4usize, 2560usize, 10usize);
    let wide = streams * hidden;
    let mut lcg = 0x1357_9bdf_u64;
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as u32) % 2000) as f32 / 1000.0 - 1.0
    };
    let mut rand_vec = |n: usize| -> Vec<f32> { (0..n).map(|_| next_f32()).collect() };
    let planes_host: Vec<Vec<f32>> = (0..streams).map(|_| rand_vec(t * hidden)).collect();
    let planes: Vec<CudaSlice<f32>> = planes_host
        .iter()
        .map(|v| e.htod(v))
        .collect::<Result<_, _>>()?;
    let ptr_vals: Vec<u64> = {
        let stream = e.gpu.stream();
        planes.iter().map(|p| p.device_ptr(&stream).0).collect()
    };
    let ptrs = e.htod_u64(&ptr_vals)?;
    let mut worst = 0.0f32;
    let check = |name: &str, a: &[f32], b: &[f32], worst: &mut f32| -> Res<()> {
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            let rel = (x - y).abs() / y.abs().max(1.0);
            if rel > *worst {
                *worst = rel;
            }
            if rel > 1e-4 {
                return Err(format!(
                    "hc-micro oracle: {name} idx {i}: micro {x} classic {y} (rel {rel:.3e})"
                )
                .into());
            }
        }
        Ok(())
    };

    // (a) batched plane norms vs per-stream rms_norm_into_view.
    let norm_stack_host = rand_vec(wide);
    let norm_stack = e.htod(&norm_stack_host)?;
    let eps = 1e-6f32;
    let mut normed_a = e.zeros(streams * t * hidden)?;
    launch_hc_norm_planes(
        e,
        &ptrs,
        &norm_stack,
        &mut normed_a,
        hidden,
        t,
        streams,
        eps,
    )?;
    let mut normed_b = e.zeros(streams * t * hidden)?;
    for s in 0..streams {
        let w = e.htod(&norm_stack_host[s * hidden..(s + 1) * hidden])?;
        let mut dst = normed_b.slice_mut(s * t * hidden..(s + 1) * t * hidden);
        launch_rms_norm_into_view(e, &planes[s], &w, &mut dst, hidden, t, eps)?;
    }
    check("norm", &e.dtoh(&normed_a)?, &e.dtoh(&normed_b)?, &mut worst)?;

    // (b) two-stage inject vs the single-stage kernel, over the SAME normed slab.
    let inj_w_host = rand_vec(streams * wide);
    let inj_w = e.htod(&inj_w_host)?;
    let mut all_a = e.zeros(streams * t)?;
    let mut partials = e.zeros(streams * t * 16)?;
    launch_hc_inject_two_stage(
        e,
        &normed_b,
        &inj_w,
        None,
        &mut partials,
        &mut all_a,
        streams,
        t,
        hidden,
        16,
    )?;
    let mut all_b = e.zeros(streams * t)?;
    launch_hc_inject_gates(e, &normed_b, &inj_w, &mut all_b, streams, t, hidden)?;
    check("inject", &e.dtoh(&all_a)?, &e.dtoh(&all_b)?, &mut worst)?;

    // (c) slab write vs the add_scaled_rows chain, from identical plane states.
    let block_out = e.htod(&rand_vec(t * hidden))?;
    launch_hc_write_planes(e, &ptrs, &block_out, &all_b, hidden, t, streams)?;
    let mut expect: Vec<Vec<f32>> = Vec::with_capacity(streams);
    let all_host = e.dtoh(&all_b)?;
    let bo_host = e.dtoh(&block_out)?;
    for (s, base) in planes_host.iter().enumerate() {
        let mut rows = base.clone();
        for tok in 0..t {
            let g = all_host[s * t + tok];
            for d in 0..hidden {
                rows[tok * hidden + d] += bo_host[tok * hidden + d] * g;
            }
        }
        expect.push(rows);
    }
    for (s, plane) in planes.iter().enumerate() {
        check(
            &format!("write plane {s}"),
            &e.dtoh(plane)?,
            &expect[s],
            &mut worst,
        )?;
    }
    Ok(format!(
        "hc-micro real-geometry oracle: streams 4 hidden 2560 t 10, worst rel {worst:.3e} \
         over norm/inject/write vs the classic composition"
    ))
}

/// REAL-GEOMETRY oracle for the perf-round-3 GDN kernels (the tiny plan cannot reach
/// either: hk 4 fails the step twin's warp guard, and the fused norm's win is only
/// meaningful at real widths). (a) `gdn_scan_step_f32` vs `gdn_scan_naive_f32` at t=1
/// on identical inputs and state copies — same per-element math, block-tree vs
/// sequential row sums, so tolerance-gated (1e-4 rel, worst reported); covers the
/// artifact geometry (nk 16, nv 48, hk/hv 128 — head sharing h%nk) and the minimum
/// hk=32 shape. (b) `rms_sigmul_f32` vs the rms_norm + sigmoid + mul chain it replaces
/// — asserted BIT-IDENTICAL (the kernel is rms_norm_f32-verbatim + sigmoid_f32 with no
/// contraction seam).
/// Block-list attention kernel oracle (long-context lane), real QSA geometry (hd 256,
/// 24/2 heads). Arm A: masked kernel vs block-list kernel over the SAME selections at
/// t_kv 4096 — BIT identity (the masked kernel's -1e30 entries contribute exact-0 terms
/// in the same ascending order; see the kernel comment). Arm B: t_kv 16384 — past the
/// masked kernel's smem bound, where only the block-list form runs — vs a HOST f32 twin
/// of the same phase order (expf vs libm exp differ in ULPs; tolerance class).
/// Selections come through the PRODUCTION renderers (`rowsel_to_mask`/`rowsel_positions`)
/// so the emission code is gated with the kernel.
pub fn gate_sdpa_blocklist(e: &Engine) -> Res<String> {
    let mut lcg = 0x51ee_7bad_u64;
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as u32) % 2000) as f32 / 1000.0 - 1.0
    };
    let (hd, nh, nkv, t) = (256usize, 24usize, 2usize, 3usize);
    let block_size = 4usize;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut bit_rows = 0usize;
    let mut worst_rel = 0.0f32;
    for (t_kv, vs_masked) in [(4096usize, true), (16384usize, false)] {
        let q_host: Vec<f32> = (0..t * nh * hd).map(|_| next_f32()).collect();
        let k_host: Vec<f32> = (0..t_kv * nkv * hd).map(|_| next_f32()).collect();
        let v_host: Vec<f32> = (0..t_kv * nkv * hd).map(|_| next_f32()).collect();
        // Per-row selections: row 0 full causal prefix; rows 1/2 scored-form block lists
        // (stride-3 / tail-heavy) with the always-visible incomplete tail.
        let sels: Vec<RowSel> = (0..t)
            .map(|qt| {
                let visible = t_kv - t + qt + 1;
                let complete = visible / block_size;
                // A full-prefix row (production: complete <= budget) only in the
                // 4096 case — its position list scales with `visible`, and the
                // 16384 full form would blow the 48 KB smem cap production never
                // approaches (full rows are <= 2052 positions there).
                if qt == 0 && vs_masked {
                    return RowSel {
                        full: true,
                        blocks: Vec::new(),
                        visible,
                    };
                }
                let stride = if qt == 1 { 3 } else { 7 };
                let blocks: Vec<u32> = (0..complete as u32)
                    .rev()
                    .step_by(stride)
                    .take(512)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                RowSel {
                    full: false,
                    blocks,
                    visible,
                }
            })
            .collect();
        let (pos_flat, meta, max_count) = rowsel_positions(&sels, block_size);
        let q = e.htod(&q_host)?;
        let k = e.htod(&k_host)?;
        let v = e.htod(&v_host)?;
        let pos = e.htod_i32(&pos_flat)?;
        let meta_dev = e.htod_i32(&meta)?;
        let mut o_list = e.zeros(t * nh * hd)?;
        launch_sdpa_blocklist(
            e,
            &q,
            &k.slice(0..t_kv * nkv * hd),
            &v.slice(0..t_kv * nkv * hd),
            &mut o_list,
            &pos,
            &meta_dev,
            hd,
            nh,
            nkv,
            t,
            max_count,
            scale,
        )?;
        let ours = e.dtoh(&o_list)?;
        if vs_masked {
            let mask = rowsel_to_mask(&sels, block_size, t_kv);
            let mask_dev = e.htod_bytes(&mask)?;
            let mut o_mask = e.zeros(t * nh * hd)?;
            launch_sdpa_mask(
                e,
                &q,
                &k.slice(0..t_kv * nkv * hd),
                &v.slice(0..t_kv * nkv * hd),
                &mut o_mask,
                &mask_dev,
                hd,
                nh,
                nkv,
                t,
                t_kv,
                scale,
            )?;
            let masked = e.dtoh(&o_mask)?;
            for (i, (a, b)) in masked.iter().zip(ours.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "sdpa_blocklist vs masked: bit mismatch at {i}: {a} vs {b} (t_kv {t_kv})"
                    )
                    .into());
                }
            }
            bit_rows = t * nh * hd;
        } else {
            // HOST twin, same phase order: per (row, head) dots ascending over the
            // selection, single-pass max/exp/normalize, weighted V ascending.
            for qt in 0..t {
                let off = meta[2 * qt] as usize;
                let count = meta[2 * qt + 1] as usize;
                for head in 0..nh {
                    let kvh = head / (nh / nkv);
                    let qrow = &q_host[(qt * nh + head) * hd..(qt * nh + head + 1) * hd];
                    let mut scores: Vec<f32> = (0..count)
                        .map(|i| {
                            let p = pos_flat[off + i] as usize;
                            let krow = &k_host[(p * nkv + kvh) * hd..(p * nkv + kvh + 1) * hd];
                            let mut acc = 0.0f32;
                            for d in 0..hd {
                                acc += qrow[d] * krow[d];
                            }
                            acc * scale
                        })
                        .collect();
                    let mx = scores.iter().copied().fold(-1e30f32, f32::max);
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    for s in scores.iter_mut() {
                        *s *= inv;
                    }
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for (i, s) in scores.iter().enumerate() {
                            let p = pos_flat[off + i] as usize;
                            acc += s * v_host[(p * nkv + kvh) * hd + d];
                        }
                        let got = ours[(qt * nh + head) * hd + d];
                        let rel = (got - acc).abs() / acc.abs().max(1e-3);
                        worst_rel = worst_rel.max(rel);
                        if rel > 1e-4 {
                            return Err(format!(
                                "sdpa_blocklist vs host twin: rel {rel} at row {qt} head {head} \
                                 dim {d} (t_kv {t_kv})"
                            )
                            .into());
                        }
                    }
                }
            }
        }
    }
    Ok(format!(
        "sdpa-blocklist oracle: BIT-IDENTICAL to the masked kernel over {bit_rows} values \
         (t_kv 4096, full+stride selections); past the mask bound (t_kv 16384) worst rel \
         {worst_rel:.3e} vs the host twin"
    ))
}

/// kvq/idxq kernel oracles (KV-quant lane). Four pins, all BIT-exact:
/// (1) the append-quantize kernels vs the host quantize twins (q8_0 K rows, q5_1 V
///     rows) over random + adversarial blocks (zeros, half-ulp rounding ties, subnormal
///     scales, constant blocks) at real (512) and padded-tail (40) widths;
/// (2) the row-dequant kernel vs the host dequant twins on those bytes;
/// (3) the FUSED quantized block-list attention vs the composition
///     "q4e_kv_dequant_rows then sdpa_blocklist_f32" — the load-bearing oracle: it
///     proves in-kernel dequant reads the same f32 values the storage contract defines
///     (the qsa_index_score 1-ULP FMA lesson made both sides explicit-intrinsic);
/// (4) the indexer q8/bf16 device appenders vs the host cache twins (the idxcache
///     host/device interleave contract).
/// Caveat, stated: blocks mixing +0.0 and -0.0 are outside the pin (fminf/fmaxf zero
/// sign order is unspecified); projection outputs do not produce signed-zero ties.
pub fn gate_kvq_kernels(e: &Engine) -> Res<String> {
    let mut lcg = 0x6b_7671_5eed_u64; // "kvq"-seeded LCG
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as u32) % 2000) as f32 / 1000.0 - 1.0
    };
    let mut report = Vec::new();

    // ---- (1) + (2): quantize + dequant twins ----
    for &dim in &[512usize, 40usize] {
        let rows = 9usize;
        let mut host_rows_f: Vec<f32> = (0..rows * dim).map(|_| next_f32()).collect();
        // Adversarial rows: 0 = all zeros; 1 = constant block (d == 0 path for q5's
        // mx == mn); 2 = rounding ties (values at exact half steps of the block scale).
        for v in host_rows_f[0..dim].iter_mut() {
            *v = 0.0;
        }
        for v in host_rows_f[dim..2 * dim].iter_mut() {
            *v = 0.75;
        }
        for (i, v) in host_rows_f[2 * dim..3 * dim].iter_mut().enumerate() {
            // amax = 1.0 at lane 0; others sit at k*(1/127)*0.5 half-steps.
            *v = if i == 0 {
                1.0
            } else {
                (i as f32) * 0.5 / 127.0
            };
        }
        // Subnormal-scale row.
        for v in host_rows_f[3 * dim..4 * dim].iter_mut() {
            *v *= 1e-40;
        }
        let dev_rows = e.htod(&host_rows_f)?;
        let mut kq = e.alloc_u8(rows * q8_row_bytes(dim))?;
        let mut vq = e.alloc_u8(rows * q5_row_bytes(dim))?;
        launch_q4e_kv_append(e, &dev_rows, &dev_rows, &mut kq, &mut vq, 0, rows, dim)?;
        let kq_host = e.dtoh_u8(&kq)?;
        let vq_host = e.dtoh_u8(&vq)?;
        let mut k_twin = Vec::new();
        let mut v_twin = Vec::new();
        for r in 0..rows {
            host_quant_q8_row(&host_rows_f[r * dim..(r + 1) * dim], dim, &mut k_twin);
            host_quant_q5_row(&host_rows_f[r * dim..(r + 1) * dim], dim, &mut v_twin);
        }
        if kq_host != k_twin {
            let i = kq_host.iter().zip(&k_twin).position(|(a, b)| a != b);
            return Err(format!("kvq q8 quantize twin: byte mismatch at {i:?} (dim {dim})").into());
        }
        if vq_host != v_twin {
            let i = vq_host.iter().zip(&v_twin).position(|(a, b)| a != b);
            return Err(format!("kvq q5 quantize twin: byte mismatch at {i:?} (dim {dim})").into());
        }
        // Dequant twin.
        let mut kf = e.zeros(rows * dim)?;
        let mut vf = e.zeros(rows * dim)?;
        launch_q4e_kv_dequant_rows(e, &kq, &vq, &mut kf, &mut vf, 0, rows, dim)?;
        let kf_host = e.dtoh(&kf)?;
        let vf_host = e.dtoh(&vf)?;
        let mut kf_twin = Vec::new();
        let mut vf_twin = Vec::new();
        host_deq_q8_rows(&kq_host, 0, rows, dim, &mut kf_twin);
        host_deq_q5_rows(&vq_host, 0, rows, dim, &mut vf_twin);
        for (i, (a, b)) in kf_host.iter().zip(&kf_twin).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!("kvq q8 dequant twin: bit mismatch at {i} (dim {dim})").into());
            }
        }
        for (i, (a, b)) in vf_host.iter().zip(&vf_twin).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!("kvq q5 dequant twin: bit mismatch at {i} (dim {dim})").into());
            }
        }
        report.push(format!("quant+dequant twins dim {dim}: BYTE/BIT-IDENTICAL"));
    }

    // ---- (3) fused quant attention vs the dequant-rows composition ----
    {
        let (hd, nh, nkv, t) = (256usize, 24usize, 2usize, 3usize);
        let kv_dim = nkv * hd;
        let block_size = 4usize;
        let scale = 1.0 / (hd as f32).sqrt();
        let t_kv = 4096usize;
        let q_host: Vec<f32> = (0..t * nh * hd).map(|_| next_f32()).collect();
        let k_host: Vec<f32> = (0..t_kv * kv_dim).map(|_| next_f32()).collect();
        let v_host: Vec<f32> = (0..t_kv * kv_dim).map(|_| next_f32()).collect();
        let k_rows = e.htod(&k_host)?;
        let v_rows = e.htod(&v_host)?;
        let mut kq = e.alloc_u8(t_kv * q8_row_bytes(kv_dim))?;
        let mut vq = e.alloc_u8(t_kv * q5_row_bytes(kv_dim))?;
        launch_q4e_kv_append(e, &k_rows, &v_rows, &mut kq, &mut vq, 0, t_kv, kv_dim)?;
        // Selections: one full-prefix row + two scored stride rows (the
        // gate_sdpa_blocklist shapes, bounded to the production smem class).
        let sels: Vec<RowSel> = (0..t)
            .map(|qt| {
                let visible = (t_kv - t + qt + 1).min(2052);
                if qt == 0 {
                    return RowSel {
                        full: true,
                        blocks: Vec::new(),
                        visible,
                    };
                }
                let complete = (t_kv - t + qt + 1) / block_size;
                let stride = if qt == 1 { 3 } else { 7 };
                let blocks: Vec<u32> = (0..complete as u32)
                    .rev()
                    .step_by(stride)
                    .take(512)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                RowSel {
                    full: false,
                    blocks,
                    visible: t_kv - t + qt + 1,
                }
            })
            .collect();
        let (pos_flat, meta, max_count) = rowsel_positions(&sels, block_size);
        let q = e.htod(&q_host)?;
        let pos = e.htod_i32(&pos_flat)?;
        let meta_dev = e.htod_i32(&meta)?;
        let mut o_fused = e.zeros(t * nh * hd)?;
        launch_q4e_sdpa_blocklist_q8q5(
            e,
            &q,
            &kq,
            &vq,
            &mut o_fused,
            &pos,
            &meta_dev,
            hd,
            nh,
            nkv,
            t,
            max_count,
            scale,
        )?;
        let mut k_deq = e.zeros(t_kv * kv_dim)?;
        let mut v_deq = e.zeros(t_kv * kv_dim)?;
        launch_q4e_kv_dequant_rows(e, &kq, &vq, &mut k_deq, &mut v_deq, 0, t_kv, kv_dim)?;
        let mut o_comp = e.zeros(t * nh * hd)?;
        launch_sdpa_blocklist(
            e,
            &q,
            &k_deq.slice(0..t_kv * kv_dim),
            &v_deq.slice(0..t_kv * kv_dim),
            &mut o_comp,
            &pos,
            &meta_dev,
            hd,
            nh,
            nkv,
            t,
            max_count,
            scale,
        )?;
        let fused = e.dtoh(&o_fused)?;
        let comp = e.dtoh(&o_comp)?;
        for (i, (a, b)) in fused.iter().zip(&comp).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "kvq fused attention vs dequant composition: bit mismatch at {i}: {a} vs {b}"
                )
                .into());
            }
        }
        // ---- (3b) `kvhoist` vs the un-hoisted kernel, SAME real geometry ----
        // The hoist is a pure read-pattern change (fp16 K block scale loaded once per 32-element
        // block instead of once per element), so the bar is bit-identity and nothing weaker.
        //
        // This arm rides arm (3)'s geometry deliberately: hd=256 is EIGHT 32-element blocks per
        // head slice and nkv=2 means the second KV head starts at element 256, so the hoisted
        // loop's block walk and its `e0 = kv_head*head_dim` offset are both genuinely exercised.
        // At a tiny head_dim the loop would run ONE iteration and the per-block scale advance —
        // the only thing the seam changes — would never be taken. That is precisely the
        // tiny-green/real-broken shape this lane has been bitten by twice, so the arm is written
        // where it cannot happen rather than trusted to a comment.
        {
            let was = kv_hoist_on();
            set_kv_hoist(true);
            let mut o_hoist = e.zeros(t * nh * hd)?;
            let launched = launch_q4e_sdpa_blocklist_q8q5(
                e,
                &q,
                &kq,
                &vq,
                &mut o_hoist,
                &pos,
                &meta_dev,
                hd,
                nh,
                nkv,
                t,
                max_count,
                scale,
            );
            set_kv_hoist(was);
            launched?;
            let hoist = e.dtoh(&o_hoist)?;
            let mut worst: Option<(usize, f32, f32)> = None;
            for (i, (a, b)) in hoist.iter().zip(&fused).enumerate() {
                if a.to_bits() != b.to_bits() && worst.is_none() {
                    worst = Some((i, *a, *b));
                }
            }
            if let Some((i, a, b)) = worst {
                return Err(format!(
                    "kvhoist vs un-hoisted q8q5 blocklist: bit mismatch at {i}: {a} vs {b} \
                     (hd={hd} nh={nh} nkv={nkv} t={t} t_kv={t_kv} max_count={max_count})"
                )
                .into());
            }
            // A no-op arm would also compare equal. Prove the seam actually selected the other
            // kernel: `kv_hoist_on()` gates the `e.func` name, and an unknown name would have
            // failed the launch above rather than silently falling through — so a green compare
            // plus a completed launch under the armed seam is the engagement evidence. State the
            // count so a zero-value compare cannot pass as a pass.
            report.push(format!(
                "kvhoist vs un-hoisted q8q5 blocklist: BIT-IDENTICAL over {} values \
                 (real geometry hd={hd} nh={nh} nkv={nkv}, {} blocks/head slice, max_count={max_count})",
                t * nh * hd,
                hd / 32
            ));
        }
        report.push(format!(
            "fused q8q5 blocklist vs dequant+f32 composition: BIT-IDENTICAL over {} values",
            t * nh * hd
        ));
    }

    // ---- (4) indexer appenders vs the host cache twins ----
    {
        let idx_dim = 128usize;
        let qk_width = 5 * idx_dim; // 4 query heads + 1 key head
        let rows = 7usize;
        let src_host: Vec<f32> = (0..rows * qk_width).map(|_| next_f32()).collect();
        let src = e.htod(&src_host)?;
        let q_off = 4 * idx_dim;
        // q8 arm.
        let mut dst_q8 = e.alloc_u8((rows + 2) * q8_row_bytes(idx_dim))?;
        launch_q4e_idx_append_q8(e, &src, &mut dst_q8, rows, idx_dim, qk_width, q_off, 2)?;
        let got = e.dtoh_u8(&dst_q8)?;
        let mut twin = vec![0u8; 2 * q8_row_bytes(idx_dim)];
        for r in 0..rows {
            host_quant_q8_row(
                &src_host[r * qk_width + q_off..(r + 1) * qk_width],
                idx_dim,
                &mut twin,
            );
        }
        if got[2 * q8_row_bytes(idx_dim)..] != twin[2 * q8_row_bytes(idx_dim)..] {
            return Err("idxq q8 append twin: byte mismatch".into());
        }
        // bf16 arm.
        let mut dst_bf = unsafe { e.gpu.stream().alloc::<u16>((rows + 2) * idx_dim)? };
        e.gpu.stream().memset_zeros(&mut dst_bf)?;
        launch_q4e_idx_append_bf16(e, &src, &mut dst_bf, rows, idx_dim, qk_width, q_off, 2)?;
        let got_bf: Vec<u16> = {
            let v = e
                .gpu
                .stream()
                .clone_dtoh(&dst_bf.slice(0..(rows + 2) * idx_dim))?;
            e.gpu.stream().synchronize()?;
            v
        };
        for r in 0..rows {
            for c in 0..idx_dim {
                let want = f32_to_bf16_rne(src_host[r * qk_width + q_off + c]);
                if got_bf[(2 + r) * idx_dim + c] != want {
                    return Err(format!("idxq bf16 append twin: mismatch row {r} col {c}").into());
                }
            }
        }
        report.push("idx q8/bf16 appenders vs host twins: BYTE-IDENTICAL".to_string());
    }

    Ok(format!("kvq kernel oracles: {}", report.join("; ")))
}

/// Device QSA index-scorer oracle at REAL indexer geometry (4 heads x 128, block 4):
/// `qsa_index_score_f32` vs the host twin's arithmetic, BIT for BIT, over a block count
/// past the real budget (so the scoring arm — not the structural fast path — is what
/// runs), plus the top-k SET equality that the selection actually depends on.
pub fn gate_qsa_index_score(e: &Engine) -> Res<String> {
    let mut lcg = 0xfeed_1234_u64;
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as u32) % 4000) as f32 / 2000.0 - 1.0
    };
    let (heads, head_dim) = (4usize, 128usize);
    let scale = (head_dim as f32).sqrt();
    let budget = 512usize;
    let mut worst_rows = 0usize;
    for (rows, n_blocks) in [(1usize, 4096usize), (7, 1031)] {
        let q_host: Vec<f32> = (0..rows * heads * head_dim).map(|_| next_f32()).collect();
        let pooled_host: Vec<f32> = (0..n_blocks * head_dim).map(|_| next_f32()).collect();
        let q = e.htod(&q_host)?;
        let pooled = e.htod(&pooled_host)?;
        let mut scores_dev = e.uninit(rows * n_blocks)?;
        launch_qsa_index_score(
            e,
            &q,
            &pooled,
            &mut scores_dev,
            heads,
            head_dim,
            n_blocks,
            rows,
            scale,
        )?;
        let got = e.dtoh(&scores_dev)?;
        for row in 0..rows {
            let qr = &q_host[row * heads * head_dim..(row + 1) * heads * head_dim];
            let host = score_blocks(qr, &pooled_host, heads, head_dim, n_blocks, scale, 1);
            for (b, want) in host.iter().enumerate() {
                let g = got[row * n_blocks + b];
                if g.to_bits() != want.to_bits() {
                    return Err(format!(
                        "qsa_index_score: bit mismatch row {row} block {b}: {g} vs host {want}"
                    )
                    .into());
                }
            }
            let a = top_blocks_ascending(&host, budget, 1);
            let b = top_blocks_ascending(&got[row * n_blocks..(row + 1) * n_blocks], budget, 1);
            if a != b {
                return Err(format!("qsa_index_score: top-k set differs at row {row}").into());
            }
            worst_rows += 1;
        }
    }
    // ---- `poolT`: the dim-major plane, through the SAME host-twin bar ----
    // Validates the whole chain, not just the kernel: the transpose kernel writes the plane from
    // the row-major region on device, and the transposed score kernel reads it. Bit-identity to
    // the host twin (not merely to the row-major device kernel) is the bar, because the row-major
    // kernel is itself gated against the host above — comparing only device-to-device would let a
    // shared mistake pass twice.
    //
    // The case is chosen to catch the ONE mistake this layout invites: `cap_rows != n_blocks`.
    // The plane's pitch is the mirror's block CAPACITY, and the mirror grows to a power of two
    // while `n_blocks` is whatever the fill happens to be — so `cap_rows == n_blocks` is the
    // ABNORMAL state, and a kernel handed `n_blocks` as its pitch would read dim d of block b as
    // dim d of a different block for every d > 0. That is silent wrong values, and it would be
    // green in any gate where the two numbers happen to coincide. Here they deliberately do not
    // (1031 blocks in a 4096-block plane), and a second case pins the aligned edge.
    let mut pool_t_rows = 0usize;
    for (rows, n_blocks, cap_rows) in [(1usize, 1031usize, 4096usize), (5, 2048, 2048)] {
        let q_host: Vec<f32> = (0..rows * heads * head_dim).map(|_| next_f32()).collect();
        let pooled_host: Vec<f32> = (0..n_blocks * head_dim).map(|_| next_f32()).collect();
        let q = e.htod(&q_host)?;
        // The mirror as `indexer_select_rows` builds it: POOL_PLANES regions of cap_rows*head_dim,
        // the row-major rows H2D'd into the first, the plane filled by the transpose kernel.
        let mut mirror = e.zeros(cap_rows * head_dim * POOL_PLANES)?;
        {
            let mut view = mirror.slice_mut(0..n_blocks * head_dim);
            e.gpu.stream().memcpy_htod(&pooled_host, &mut view)?;
        }
        launch_qsa_pooled_transpose(e, &mut mirror, 0, n_blocks, head_dim, cap_rows)?;
        let was = pool_t_on();
        set_pool_t(true);
        let mut scores_dev = e.uninit(rows * n_blocks)?;
        let launched = launch_qsa_index_score(
            e,
            &q,
            &mirror,
            &mut scores_dev,
            heads,
            head_dim,
            n_blocks,
            rows,
            scale,
        );
        set_pool_t(was);
        launched?;
        let got = e.dtoh(&scores_dev)?;
        for row in 0..rows {
            let qr = &q_host[row * heads * head_dim..(row + 1) * heads * head_dim];
            let host = score_blocks(qr, &pooled_host, heads, head_dim, n_blocks, scale, 1);
            for (b, want) in host.iter().enumerate() {
                let g = got[row * n_blocks + b];
                if g.to_bits() != want.to_bits() {
                    return Err(format!(
                        "poolT qsa_index_score_f32_t: bit mismatch row {row} block {b}: \
                         {g} vs host {want} (n_blocks={n_blocks} cap_rows={cap_rows})"
                    )
                    .into());
                }
            }
            if top_blocks_ascending(&host, budget, 1)
                != top_blocks_ascending(&got[row * n_blocks..(row + 1) * n_blocks], budget, 1)
            {
                return Err(format!(
                    "poolT qsa_index_score_f32_t: top-{budget} set differs at row {row} \
                     (n_blocks={n_blocks} cap_rows={cap_rows})"
                )
                .into());
            }
            pool_t_rows += 1;
        }
    }
    Ok(format!(
        "qsa-index-score oracle: device scores BIT-IDENTICAL to the host twin over \
         {worst_rows} rows (4096 + 1031 blocks, real 4x128 geometry) and top-512 sets equal; \
         poolT dim-major plane (transpose + transposed kernel) BIT-IDENTICAL to the SAME host \
         twin over {pool_t_rows} rows, incl. the pitch-trap case cap_rows=4096 != n_blocks=1031"
    ))
}

/// PLE n-gram id CACHE oracle (262k perf lane, `plecache`): `host_ngram_ids_cached` vs the
/// full `host_ngram_ids` twin, ids compared EXACTLY (they are table row indices — one wrong
/// id gathers a different embedding row and the output is fluent and wrong, so there is no
/// tolerance to have). Host-only, so it costs nothing and runs on every gate invocation.
///
/// The cases are the ones a cache gets wrong, not the ones it gets right:
/// - **one-token-at-a-time growth** (the decode shape) and **chunked growth** (the prefill
///   shape) over the same sequence, interleaved lengths, against a fresh full recompute at
///   every length.
/// - **EOS inside the sequence**: `shift_right_ignore_eos` resets its segment at an eos, and
///   the running `last_eos_inclusive` is the one piece of cross-token state the incremental
///   form has to carry. A cache that ignored it would be green on eos-free text.
/// - **rewind to a DIVERGING prefix** (the spec-reject shape): extend, then ask for a
///   sequence that shares only a prefix. The cache must truncate at the divergence, not at
///   the length — a length-only check keeps another sequence's hashes and produces fluent
///   output from the wrong rows, which is invisible.
/// - **a SHORTER unrelated sequence in the same cache** (state reuse).
/// - **eos as the very first token** and **an all-eos sequence** (segment_start edges).
pub fn gate_ple_ngram_cache() -> Res<String> {
    // Real artifact geometry: max_ngram 3, 16 heads (8 per ngram size), per-head vocab.
    let max_ngram = 3usize;
    let heads_per_ngram = 8usize;
    let total_heads = (max_ngram - 1) * heads_per_ngram;
    let multipliers: Vec<i64> = vec![
        0x2545_F491_4F6C_DD1D,
        0x9E37_79B9_7F4A_7C15u64 as i64,
        0x1234_5678_9ABC_DEF1,
    ];
    let sizes: Vec<i64> = (0..total_heads)
        .map(|i| 2_500_012_160 - (i as i64) * 7)
        .collect();
    let offsets: Vec<i64> = (0..total_heads)
        .map(|i| (i as i64) * 2_500_012_160)
        .collect();
    let eos = 248_046u32;
    let full = |ids: &[u32]| -> Vec<i64> {
        host_ngram_ids(
            ids,
            &multipliers,
            &sizes,
            &offsets,
            max_ngram,
            heads_per_ngram,
            eos,
        )
    };
    let mut lcg = 0x0be1_10ca_u64;
    let mut next_tok = move || -> u32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 33) as u32) % 250_000
    };
    let mut checks = 0usize;
    let run = |label: &str, steps: Vec<Vec<u32>>| -> Res<usize> {
        // `steps` are cumulative sequences fed to ONE cache, in order.
        let (mut ci, mut ch, mut ce) = (Vec::new(), Vec::new(), -1i64);
        let mut n = 0usize;
        for seq in &steps {
            host_ngram_ids_cached(
                &mut ci,
                &mut ch,
                &mut ce,
                seq,
                &multipliers,
                &sizes,
                &offsets,
                max_ngram,
                heads_per_ngram,
                eos,
            );
            let want = full(seq);
            if ci.len() != want.len() {
                return Err(format!(
                    "plecache oracle {label}: cache has {} ids, twin {} at len {}",
                    ci.len(),
                    want.len(),
                    seq.len()
                )
                .into());
            }
            if let Some(i) = ci.iter().zip(&want).position(|(a, b)| a != b) {
                return Err(format!(
                    "plecache oracle {label}: id {i} differs at len {} (token {}, head {}): \
                     cache {} vs twin {}",
                    seq.len(),
                    i / total_heads,
                    i % total_heads,
                    ci[i],
                    want[i]
                )
                .into());
            }
            n += seq.len();
        }
        Ok(n)
    };
    // 1. Decode shape: grow one token at a time, eos-free.
    {
        let base: Vec<u32> = (0..200).map(|_| next_tok()).collect();
        let steps: Vec<Vec<u32>> = (1..=base.len()).map(|n| base[..n].to_vec()).collect();
        checks += run("decode-growth", steps)?;
    }
    // 2. Prefill shape: chunked growth with ragged chunk sizes.
    {
        let base: Vec<u32> = (0..600).map(|_| next_tok()).collect();
        let mut steps = Vec::new();
        let mut n = 0usize;
        for step in [7usize, 1, 64, 3, 128, 2, 200, 195] {
            n = (n + step).min(base.len());
            steps.push(base[..n].to_vec());
        }
        checks += run("prefill-chunks", steps)?;
    }
    // 3. EOS inside the sequence (segment resets), incl. adjacent eos and a trailing eos.
    {
        let mut base: Vec<u32> = (0..300).map(|_| next_tok()).collect();
        for p in [0usize, 1, 2, 37, 38, 100, 101, 102, 299] {
            base[p] = eos;
        }
        let steps: Vec<Vec<u32>> = (1..=base.len()).map(|n| base[..n].to_vec()).collect();
        checks += run("eos-segments", steps)?;
    }
    // 4. All-eos: every position resets its own segment.
    {
        let base: Vec<u32> = vec![eos; 40];
        let steps: Vec<Vec<u32>> = (1..=base.len()).map(|n| base[..n].to_vec()).collect();
        checks += run("all-eos", steps)?;
    }
    // 5. Rewind to a DIVERGING prefix, repeatedly, then past the old length.
    {
        let a: Vec<u32> = (0..300).map(|_| next_tok()).collect();
        let mut b = a.clone();
        b[150] = a[150].wrapping_add(1) % 250_000;
        let mut c = b.clone();
        c[7] = b[7].wrapping_add(3) % 250_000;
        let mut d = c.clone();
        d.truncate(9);
        d.extend((0..100).map(|_| next_tok()));
        checks += run(
            "rewind-divergent",
            vec![
                a.clone(),
                a[..151].to_vec(),
                b.clone(),
                b[..8].to_vec(),
                c.clone(),
                d.clone(),
                a.clone(),
            ],
        )?;
    }
    // 6. A shorter unrelated sequence in the same cache (state reuse), and back up again.
    {
        let a: Vec<u32> = (0..250).map(|_| next_tok()).collect();
        let mut s: Vec<u32> = (0..11).map(|_| next_tok()).collect();
        s[0] = eos;
        checks += run("state-reuse", vec![a.clone(), s.clone(), a.clone(), s])?;
    }
    Ok(format!(
        "plecache oracle: incremental n-gram ids EXACT vs the full host_ngram_ids twin over \
         {checks} cumulative-sequence comparisons across 6 case families (decode one-at-a-time \
         growth, ragged prefill chunks, eos segment resets incl. adjacent + leading + trailing \
         eos, all-eos, repeated rewinds to DIVERGING prefixes, and shorter-unrelated-sequence \
         state reuse)"
    ))
}

/// SEAM TABLE oracle (host-only, 262k host-lever lane): every `MEMRA_Q4E_SEAMS` name maps to its
/// OWN switch, `set_seam` and `seam_state` agree, and arming one seam changes NOTHING else.
///
/// This exists because the name table was refactored out of `apply_env_seams` into `set_seam` so
/// a measurement harness could flip a seam between timed rounds, and three agents add arms to it
/// concurrently. The failure mode of a mechanical refactor like that is not a crash: it is one
/// arm wired to a neighbour's switch, which arms the wrong seam and produces a fully fluent,
/// fully green run measuring something other than what the receipt claims. A copy-paste arm that
/// duplicates the line above it is exactly what a per-name distinctness check catches and what
/// reading the diff does not.
///
/// The strong assertion is the CROSS one: for each name, snapshot every other seam's state, flip
/// this one, and require that every other state is unchanged. That is what makes it a wiring
/// test rather than a smoke test — a table where two names share a switch passes "set then read
/// it back" and fails this.
pub fn gate_seam_table() -> Res<String> {
    // Derived from `seam_names()` — the engine's own list — so adding a seam extends this gate
    // automatically instead of silently escaping it. The three-valued names (`idxq`, `longatt`)
    // have no boolean `seam_state` and are filtered out here, but they are still required below
    // to be ACCEPTED by both entry points.
    let all: &[&str] = seam_names();
    let boolean: Vec<&str> = all
        .iter()
        .copied()
        .filter(|n| seam_state(n).is_some())
        .collect();
    // Non-vacuity, and it has to be able to FAIL: a collapsed list would make every assertion
    // below pass over nothing. Both bounds are real — the table carries 20+ boolean seams today,
    // and at least the two three-valued ones (`idxq`, `longatt`) must be present and filtered
    // out — so a list that lost either class trips here instead of reporting a green over a stub.
    if boolean.len() < 20 || all.len() < boolean.len() + 2 {
        return Err(format!(
            "seam-table oracle: refusing to report on {} boolean names out of {} total — the \
             seam list collapsed, so every assertion below would be vacuous",
            boolean.len(),
            all.len()
        )
        .into());
    }
    let names: &[&str] = &boolean;
    let snapshot = || -> Res<Vec<bool>> {
        names
            .iter()
            .map(|n| {
                seam_state(n).ok_or_else(|| {
                    Box::<dyn std::error::Error>::from(format!(
                        "seam-table oracle: seam_state({n:?}) is None — the name is in set_seam \
                         but not in seam_state, so save/restore around a measurement would \
                         silently not restore it"
                    ))
                })
            })
            .collect()
    };
    let restore = |v: &[bool]| {
        for (n, &b) in names.iter().zip(v) {
            set_seam(n, b, None);
        }
    };
    let entry = snapshot()?;
    let mut checks = 0usize;
    for (i, name) in names.iter().enumerate() {
        for &want in &[true, false, true] {
            let before = snapshot()?;
            if !set_seam(name, want, None) {
                restore(&entry);
                return Err(format!("seam-table oracle: set_seam({name:?}) refused").into());
            }
            let after = snapshot()?;
            if after[i] != want {
                restore(&entry);
                return Err(format!(
                    "seam-table oracle: set_seam({name:?}, {want}) then seam_state read {} — the \
                     two tables disagree on this name",
                    after[i]
                )
                .into());
            }
            // THE CROSS-CHECK, and the reason this is a wiring test rather than a smoke test: a
            // copy-paste arm wired to a neighbour's switch passes "set it then read it back" and
            // fails only here.
            for (j, other) in names.iter().enumerate() {
                if j != i && after[j] != before[j] {
                    restore(&entry);
                    return Err(format!(
                        "seam-table oracle: arming {name:?} also changed {other:?} ({} -> {}) — \
                         two names share one switch",
                        before[j], after[j]
                    )
                    .into());
                }
            }
            checks += 1;
        }
    }
    // Every name in the engine's own list — including the three-valued ones — must be accepted by
    // both entry points, or `apply_env_seams` would silently ignore a documented seam and the
    // run would measure the default while its receipt named the seam.
    for name in all {
        if !seam_exists(name) {
            restore(&entry);
            return Err(format!(
                "seam-table oracle: seam_names() lists {name:?} but seam_exists refuses it"
            )
            .into());
        }
        if !set_seam(name, seam_state(name).unwrap_or(false), None) {
            restore(&entry);
            return Err(format!(
                "seam-table oracle: seam_names() lists {name:?} but set_seam refuses it"
            )
            .into());
        }
    }
    // An unknown name must be refused by BOTH entry points, not silently accepted.
    if seam_exists("definitely-not-a-seam") || set_seam("definitely-not-a-seam", true, None) {
        restore(&entry);
        return Err("seam-table oracle: an unknown seam name was accepted".into());
    }
    // And `seam_exists` must apply NOTHING — the property the interleaved-A/B harness relies on
    // when it validates a seam name before a 25-80 minute prefill begins.
    let before = snapshot()?;
    for name in all {
        let _ = seam_exists(name);
    }
    if snapshot()? != before {
        restore(&entry);
        return Err("seam-table oracle: seam_exists mutated a seam (it must be name-only)".into());
    }
    restore(&entry);
    if snapshot()? != entry {
        return Err("seam-table oracle: the gate did not restore the entry state".into());
    }
    Ok(format!(
        "seam-table oracle: {} boolean seam names of {} total, {checks} set/read cycles, each \
         verified to change its OWN state and NO other (the cross-check that catches an arm \
         wired to a neighbour's switch), every listed name accepted by both entry points, \
         unknown names refused by both, seam_exists proven side-effect-free, entry state restored",
        names.len(),
        all.len()
    ))
}

/// Device QSA indexer top-k SELECTION oracle (262k perf lane): `qsa_index_topk_u32` vs
/// `top_blocks_ascending` over the SAME score slab. Contract: the selected block ids AND
/// their emitted (ascending) order are EXACT — hard fail on any difference, no tolerance,
/// because a differing selection changes which KV rows the attention reads.
///
/// Geometry is REAL, not tiny: budget 512 (the shipped `budget_blocks`) at block counts up
/// to **65,536 — the 262,144-token target window's `fill/4`** — plus non-multiple counts
/// and RAGGED batches where each row reads its own prefix of a wider slab, which is the
/// exact shape the sub-batched caller produces. The tiny-green/real-broken trap has bitten
/// this lane twice; a budget-2 fixture would pass a kernel that cannot address 2^16 blocks.
///
/// Tie batteries a random draw cannot produce, and they are the point rather than an edge
/// case — the pinned rule is score desc then block index ASC:
/// - **all-zero**: every score +0.0. The whole selection is decided by the index tiebreak,
///   and this class is STRUCTURAL here (the scores are a relu-sum, so a deep row really
///   does carry long runs of exact +0.0). A tie-blind kernel is green everywhere else and
///   silently wrong here.
/// - **duplicate group straddling the budget boundary**: more equal scores than remaining
///   slots, so the boundary itself is resolved by index.
/// - **signed zeros / subnormals / negative / NaN**: outside the reachable score domain
///   (the caller's scores are >= +0.0), but the kernel's key is `f32::total_cmp` verbatim
///   over the whole domain, so the oracle proves that rather than assuming the domain.
pub fn gate_qsa_index_topk(e: &Engine) -> Res<String> {
    let budget = 512usize;
    let mut lcg = 0x1d5e_10ca_u64;
    let mut rows_checked = 0usize;
    let mut deepest = 0usize;
    // (label, per-row block counts, slab stride, score generator)
    let mut cases: Vec<(String, Vec<usize>, usize, Vec<f32>)> = Vec::new();
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Relu-sum scores are >= 0 with a heavy mass at exactly +0.0 — draw that shape.
        let r = ((lcg >> 33) as u32) % 1000;
        if r < 250 { 0.0 } else { (r as f32) / 250.0 }
    };
    for (label, counts) in [
        ("real-262k-depth", vec![65_536usize]),
        ("real-131k-depth", vec![32_768usize, 32_768]),
        ("shallow", vec![513usize, 1_031, 4_096]),
        ("ragged-batch", vec![2_049usize, 8_191, 65_536, 4_097]),
    ] {
        let stride = *counts.iter().max().unwrap();
        let slab: Vec<f32> = (0..counts.len() * stride).map(|_| next_f32()).collect();
        cases.push((label.to_string(), counts, stride, slab));
    }
    // all-zero: index tiebreak alone decides the whole selection.
    cases.push((
        "all-zero".into(),
        vec![65_536usize],
        65_536,
        vec![0.0f32; 65_536],
    ));
    // duplicate group straddling the boundary: 600 equal scores for the last 500 slots.
    {
        let n = 4_096usize;
        let mut v = vec![0.0f32; n];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = if i < 12 {
                100.0 - i as f32
            } else if i % 7 == 0 {
                2.5 // ~585 exact duplicates, straddling slot 512
            } else {
                (i % 3) as f32 * 0.25
            };
        }
        cases.push(("dup-straddle".into(), vec![n], n, v));
    }
    // Signed zeros, subnormals, negatives and NaN: the total_cmp domain, not the score
    // domain. total_cmp orders -0.0 below +0.0 and every NaN by its sign bit.
    {
        let n = 2_048usize;
        let mut v = vec![0.0f32; n];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = match i % 8 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1),  // smallest positive subnormal
                3 => -f32::from_bits(1), // smallest negative subnormal
                4 => -(i as f32) * 0.5,
                5 => f32::NAN,
                6 => -f32::NAN,
                _ => (i % 5) as f32,
            };
        }
        cases.push(("total-cmp-domain".into(), vec![n], n, v));
    }
    for (label, counts, stride, slab) in &cases {
        let scores = e.htod(slab)?;
        let picked = launch_qsa_index_topk(e, &scores, counts, *stride, budget)?;
        if picked.len() != counts.len() {
            return Err(format!("idxsel oracle {label}: {} rows back", picked.len()).into());
        }
        for (r, &complete) in counts.iter().enumerate() {
            let row = &slab[r * *stride..r * *stride + complete];
            let twin = top_blocks_ascending(row, budget, 1);
            if twin != picked[r] {
                let first = twin
                    .iter()
                    .zip(picked[r].iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(twin.len().min(picked[r].len()));
                return Err(format!(
                    "idxsel oracle {label}: selection differs at row {r} (blocks {complete}), \
                     first differing slot {first}: host {:?} vs device {:?}",
                    twin.get(first),
                    picked[r].get(first)
                )
                .into());
            }
            rows_checked += 1;
            deepest = deepest.max(complete);
        }
    }
    Ok(format!(
        "qsa-index-topk oracle: device selection ids + ASCENDING order EXACT vs \
         top_blocks_ascending over {rows_checked} rows / {} cases at budget {budget}, \
         deepest {deepest} blocks (= the 262,144-token window), incl. the all-zero, \
         boundary-straddling-duplicate and total_cmp-domain (signed zero / subnormal / \
         negative / NaN) tie classes",
        cases.len()
    ))
}

/// Device-router oracle at REAL geometry (devtwin lane): `qwen4exp_route_topk_f32` vs
/// `host_route_softmax_topk` on the SAME logits. Contract: the selection (ids AND their
/// emitted order — the combine reads slots sequentially) is EXACT, hard fail on any
/// mismatch; weights within a documented ULP bound (exp is the one op not bit-pinned to
/// host libm — kernel doc), worst observed printed in the receipt. Rows include the tie
/// batteries a random draw cannot produce: duplicate-logit groups STRADDLING the top-k
/// boundary (weight ties resolve by index — the rule a logits-ordered top-k would get
/// wrong), an all-equal row, and underflow rows (subnormal/zero weight ties). The renorm
/// denominator floor is unbindable on softmax geometry (top-k sum >= k/experts — see
/// ROUTE_DENOM_FLOOR) so it carries no arm; the twin computes the same fmaxf.
pub fn gate_route_kernel(e: &Engine) -> Res<String> {
    let mut lcg = 0x00de_7710_u64;
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as u32) % 8000) as f32 / 200.0 - 20.0 // router-logit-scale [-20, 20)
    };
    const ULP_BOUND: u32 = 2;
    let mut worst_ulp: u32 = 0;
    let mut rows_checked = 0usize;
    let run = |e: &Engine,
               label: &str,
               logits_host: &[f32],
               experts: usize,
               selected: usize,
               rows: usize,
               worst_ulp: &mut u32|
     -> Res<()> {
        let logits = e.htod(logits_host)?;
        let mut sel = e.alloc_uninit::<i32>(rows * selected)?;
        let mut w = e.uninit(rows * selected)?;
        let mut tok = e.alloc_uninit::<i32>(rows * selected)?;
        launch_route_topk(
            e,
            &logits,
            &mut sel,
            &mut w,
            Some((&mut tok, 3)),
            experts,
            selected,
            rows,
        )?;
        let sel_h = e.gpu.stream().clone_dtoh(&sel.slice(0..rows * selected))?;
        let w_h = e.dtoh(&w)?;
        let tok_h = e.gpu.stream().clone_dtoh(&tok.slice(0..rows * selected))?;
        let k = selected.min(experts);
        for row in 0..rows {
            let twin =
                host_route_softmax_topk(&logits_host[row * experts..(row + 1) * experts], selected);
            if twin.len() != k {
                return Err(format!("route oracle {label}: host twin width {}", twin.len()).into());
            }
            for (j, &(idx, wt)) in twin.iter().enumerate() {
                let ds = sel_h[row * selected + j];
                let dw = w_h[row * selected + j];
                if ds != idx as i32 {
                    return Err(format!(
                        "route oracle {label}: selection mismatch row {row} slot {j}: \
                         device {ds} vs host {idx}"
                    )
                    .into());
                }
                let ulp = (dw.to_bits() as i64 - wt.to_bits() as i64).unsigned_abs();
                let ulp = u32::try_from(ulp).unwrap_or(u32::MAX);
                if ulp > ULP_BOUND {
                    return Err(format!(
                        "route oracle {label}: weight ULP {ulp} > {ULP_BOUND} at row {row} \
                         slot {j}: device {dw:e} vs host {wt:e}"
                    )
                    .into());
                }
                *worst_ulp = (*worst_ulp).max(ulp);
                if tok_h[row * selected + j] != (3 + row) as i32 {
                    return Err(format!(
                        "route oracle {label}: tok map wrong at row {row} slot {j}"
                    )
                    .into());
                }
            }
        }
        Ok(())
    };
    // Real geometry, random router-scale logits, batched rows (the verify shape).
    let (experts, selected) = (512usize, 10usize);
    for rows in [1usize, 6, 16] {
        let logits: Vec<f32> = (0..rows * experts).map(|_| next_f32()).collect();
        run(e, "real", &logits, experts, selected, rows, &mut worst_ulp)?;
        rows_checked += rows;
    }
    // Tie batteries (single rows).
    let mut tie_rows: Vec<(String, Vec<f32>)> = Vec::new();
    {
        // A 12-wide duplicate group straddling the top-10 boundary at positions 4..16:
        // host keeps the six lowest indices of the group after the four strict leaders.
        let mut v: Vec<f32> = (0..experts).map(|i| -30.0 - (i as f32) * 0.01).collect();
        for (rank, slot) in [40usize, 7, 300, 11].iter().enumerate() {
            v[*slot] = 10.0 - rank as f32;
        }
        for slot in [500usize, 3, 77, 210, 8, 401, 129, 64, 255, 380, 17, 450] {
            v[slot] = 2.5;
        }
        tie_rows.push(("dup-straddle".into(), v));
        // All-equal: the selection is indices 0..k by the tie rule alone.
        tie_rows.push(("all-equal".into(), vec![0.125f32; experts]));
        // Underflow: one dominant logit, the rest deep negative — weights tie at
        // 0.0/subnormal and the boundary resolves by index among bit-equal weights.
        let mut v = vec![-200.0f32; experts];
        v[100] = 5.0;
        for (i, slot) in [479usize, 2, 33].iter().enumerate() {
            v[*slot] = -80.0 - i as f32; // subnormal-weight class
        }
        tie_rows.push(("underflow".into(), v));
    }
    for (label, v) in &tie_rows {
        run(e, label, v, experts, selected, 1, &mut worst_ulp)?;
        rows_checked += 1;
    }
    // Off-real geometry (the envelope's edges): small expert counts, selected == experts.
    for (ex, se) in [(64usize, 4usize), (16, 16), (128, 32)] {
        let logits: Vec<f32> = (0..3 * ex).map(|_| next_f32()).collect();
        run(e, "geom", &logits, ex, se, 3, &mut worst_ulp)?;
        rows_checked += 3;
    }
    Ok(format!(
        "route oracle: device selection ids+order EXACT vs host twin over {rows_checked} rows \
         (real 512/10 + tie straddle/all-equal/underflow + geometry edges), worst weight \
         ULP {worst_ulp} (bound {ULP_BOUND}), tok map exact"
    ))
}

pub fn gate_gdn_step_kernels(e: &Engine) -> Res<String> {
    let mut lcg = 0x0bad_cafe_u64;
    let mut next_f32 = move || -> f32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as u32) % 2000) as f32 / 1000.0 - 1.0
    };
    let mut worst = 0.0f32;
    for (nk, nv, hk, hv) in [(16usize, 48usize, 128usize, 128usize), (2, 4, 32, 8)] {
        let conv_dim = 2 * nk * hk + nv * hv;
        let qkv_host: Vec<f32> = (0..conv_dim).map(|_| next_f32()).collect();
        let g_log_host: Vec<f32> = (0..nv).map(|_| next_f32().abs() * -2.0).collect();
        let beta_host: Vec<f32> = (0..nv).map(|_| next_f32()).collect();
        let state_host: Vec<f32> = (0..nv * hv * hk).map(|_| next_f32()).collect();
        let qkv = e.htod(&qkv_host)?;
        let g_log = e.htod(&g_log_host)?;
        let beta = e.htod(&beta_host)?;
        let scale = 1.0 / (hk as f32).sqrt();
        let eps = 1e-6f32;
        let mut state_a = e.htod(&state_host)?;
        let mut o_a = e.zeros(nv * hv)?;
        launch_gdn_scan(
            e,
            &qkv,
            &g_log,
            &beta,
            &mut state_a,
            &mut o_a,
            nk,
            nv,
            hk,
            hv,
            1,
            scale,
            eps,
        )?;
        let mut state_b = e.htod(&state_host)?;
        let mut o_b = e.zeros(nv * hv)?;
        launch_gdn_scan_step(
            e,
            &qkv,
            &g_log,
            &beta,
            &mut state_b,
            &mut o_b,
            nk,
            nv,
            hk,
            hv,
            scale,
            eps,
        )?;
        for (name, reference, candidate) in [
            ("o", e.dtoh(&o_a)?, e.dtoh(&o_b)?),
            ("state", e.dtoh(&state_a)?, e.dtoh(&state_b)?),
        ] {
            for (i, (&r, &c)) in reference.iter().zip(&candidate).enumerate() {
                let rel = (r - c).abs() / r.abs().max(1.0);
                if rel > worst {
                    worst = rel;
                }
                if rel > 1e-4 {
                    return Err(format!(
                        "gdn-step oracle: nk{nk}/nv{nv}/hk{hk}/hv{hv} {name} idx {i}: \
                         naive {r} step {c} (rel {rel:.3e})"
                    )
                    .into());
                }
            }
        }
    }
    // (b) fused norm+gate bit-identity at the artifact norm shape (48 rows of 128).
    let (rows, cols) = (48usize, 128usize);
    let x = e.htod(&(0..rows * cols).map(|_| next_f32()).collect::<Vec<_>>())?;
    let w = e.htod(&(0..cols).map(|_| next_f32()).collect::<Vec<_>>())?;
    let z = e.htod(&(0..rows * cols).map(|_| next_f32()).collect::<Vec<_>>())?;
    let eps = 1e-6f32;
    let mut normed = e.zeros(rows * cols)?;
    e.rms_norm(&x, &w, &mut normed, cols, rows, eps)?;
    let mut sg = e.zeros(rows * cols)?;
    e.sigmoid(&z, &mut sg, rows * cols)?;
    let mut chain = e.zeros(rows * cols)?;
    e.mul(&normed, &sg, &mut chain, rows * cols)?;
    let mut fused = e.zeros(rows * cols)?;
    launch_rms_sigmul(e, &x, &w, &z, &mut fused, cols, rows, eps)?;
    let (chain_h, fused_h) = (e.dtoh(&chain)?, e.dtoh(&fused)?);
    for (i, (&a, &b)) in chain_h.iter().zip(&fused_h).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "rms_sigmul oracle: idx {i} not bit-identical: chain {a:?} fused {b:?}"
            )
            .into());
        }
    }
    Ok(format!(
        "gdn-step kernel oracle: scan step twin worst rel {worst:.3e} over artifact + \
         hk32 geometries; rms_sigmul bit-identical to the norm/sigmoid/mul chain ({rows}x{cols})"
    ))
}

pub fn gate_qmatvec_bf16(e: &Engine) -> Res<String> {
    let mut lcg = 0x9e37_79b9_u64;
    let mut next_u32 = move || -> u32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (lcg >> 33) as u32
    };
    let mut worst = (0.0f32, 0.0f32);
    for (mode, batch, t, out_f, in_f, x_bstride) in [
        ("per_batch_x", 3usize, 2usize, 5usize, 48usize, 2 * 48usize),
        ("shared_x", 4, 3, 7, 16, 0usize),
    ] {
        // bf16 weights minted as bf16 BYTES first (so the host twin widens the same
        // values the kernel reads), incl. sign and small-exponent coverage.
        let w_elems = batch * out_f * in_f;
        let mut w_bytes = Vec::with_capacity(w_elems * 2);
        let mut w_host = Vec::with_capacity(w_elems);
        for _ in 0..w_elems {
            // Magnitude bits below 0x4000 (= 2.0): denormals through ~2.0, signed —
            // keeps a 48-term dot far from overflow while covering the exponent range.
            let h = ((next_u32() % 0x4000) as u16) | (((next_u32() & 1) as u16) << 15);
            w_bytes.extend_from_slice(&h.to_le_bytes());
            w_host.push(f32::from_bits(u32::from(h) << 16));
        }
        let x_rows = if x_bstride == 0 { t } else { batch * t };
        let x_host: Vec<f32> = (0..x_rows * in_f)
            .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let w_dev = e.htod_bytes(&w_bytes)?;
        let x_dev = e.htod(&x_host)?;
        let mut y_dev = e.uninit(batch * t * out_f)?;
        launch_qmatvec_bf16w(
            e,
            &w_dev,
            &x_dev,
            &mut y_dev,
            in_f,
            out_f,
            t,
            batch,
            out_f * in_f,
            x_bstride,
            in_f,
            t * out_f,
        )?;
        let y = e.dtoh(&y_dev)?;
        for b in 0..batch {
            for tok in 0..t {
                let xrow = &x_host[b * x_bstride + tok * in_f..][..in_f];
                for o in 0..out_f {
                    let wrow = &w_host[(b * out_f + o) * in_f..][..in_f];
                    let mut want = 0.0f32;
                    for i in 0..in_f {
                        want += wrow[i] * xrow[i];
                    }
                    let got = y[(b * t + tok) * out_f + o];
                    let abs = (want - got).abs();
                    let rel = abs / want.abs().max(1.0);
                    worst.0 = worst.0.max(abs);
                    worst.1 = worst.1.max(rel);
                    if rel > 1e-5 {
                        return Err(format!(
                            "bf16-matvec oracle: {mode} b {b} tok {tok} row {o}: want {want} \
                             got {got} (rel {rel:.3e})"
                        )
                        .into());
                    }
                }
            }
        }
    }
    // MT weight-shared mode (mtp-spec verify): the multi-token kernel must be
    // BIT-IDENTICAL per (row, token) to the per-token grid on the same operands —
    // artifact-class geometry (in_f % 8, wide rows) + odd t.
    {
        let (out_f, in_f, t) = (33usize, 64usize, 5usize);
        let w_elems = out_f * in_f;
        let mut w_bytes = Vec::with_capacity(w_elems * 2);
        for _ in 0..w_elems {
            let h = ((next_u32() % 0x4000) as u16) | (((next_u32() & 1) as u16) << 15);
            w_bytes.extend_from_slice(&h.to_le_bytes());
        }
        let x_host: Vec<f32> = (0..t * in_f)
            .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let w_dev = e.htod_bytes(&w_bytes)?;
        let x_dev = e.htod(&x_host)?;
        let mut y_grid = e.uninit(t * out_f)?;
        launch_qmatvec_bf16w(
            e,
            &w_dev,
            &x_dev,
            &mut y_grid,
            in_f,
            out_f,
            t,
            1,
            0,
            0,
            in_f,
            0,
        )?;
        let mut y_mt = e.uninit(t * out_f)?;
        launch_qmatvec_bf16w_mt(e, &w_dev, 0, &x_dev, &mut y_mt, in_f, out_f, t)?;
        let (a, b) = (e.dtoh(&y_grid)?, e.dtoh(&y_mt)?);
        for (i, (&x1, &x2)) in a.iter().zip(&b).enumerate() {
            if x1.to_bits() != x2.to_bits() {
                return Err(format!(
                    "bf16-matvec mt oracle: idx {i}: grid {x1} vs mt {x2} NOT bit-identical"
                )
                .into());
            }
        }
    }
    // SEL mode (devtwin stage 2, the DeviceBf16 draft bank): the device-selected
    // grouped kernel must be BIT-IDENTICAL per slot to the per-slot off_into chain on
    // the same bank + sel (duplicate slots included), in BOTH stride shapes — shared x
    // (gate/up) and per-slot x rows (down).
    {
        let (experts, out_f, in_f, n_sel) = (16usize, 24usize, 32usize, 6usize);
        let w_elems = experts * out_f * in_f;
        let mut w_bytes = Vec::with_capacity(w_elems * 2);
        for _ in 0..w_elems {
            let h = ((next_u32() % 0x4000) as u16) | (((next_u32() & 1) as u16) << 15);
            w_bytes.extend_from_slice(&h.to_le_bytes());
        }
        let sel_host: Vec<i32> = vec![7, 0, 15, 7, 3, 9]; // duplicate expert on purpose
        let bank = e.htod_bytes(&w_bytes)?;
        let sel = e.htod_i32(&sel_host)?;
        for (label, x_rows, x_sstride) in [("shared-x", 1usize, 0usize), ("slot-x", n_sel, in_f)] {
            let x_host: Vec<f32> = (0..x_rows * in_f)
                .map(|_| (next_u32() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let x_dev = e.htod(&x_host)?;
            let mut y_sel = e.uninit(n_sel * out_f)?;
            launch_qmatvec_bf16w_sel(
                e, &bank, &sel, 0, &x_dev, 0, x_sstride, &mut y_sel, n_sel, in_f, out_f,
            )?;
            let mut y_ref = e.uninit(n_sel * out_f)?;
            for (slot, &eid) in sel_host.iter().enumerate() {
                launch_qmatvec_bf16w_off_into(
                    e,
                    &bank,
                    eid as usize * out_f,
                    &x_dev,
                    slot * x_sstride,
                    &mut y_ref,
                    slot * out_f,
                    in_f,
                    out_f,
                )?;
            }
            let (a, b) = (e.dtoh(&y_sel)?, e.dtoh(&y_ref)?);
            for (i, (&x1, &x2)) in a.iter().zip(&b).enumerate() {
                if x1.to_bits() != x2.to_bits() {
                    return Err(format!(
                        "bf16-matvec sel oracle ({label}): idx {i}: sel {x1} vs off_into {x2} \
                         NOT bit-identical"
                    )
                    .into());
                }
            }
        }
    }
    Ok(format!(
        "bf16-matvec kernel oracle: worst abs {:.3e} rel {:.3e} over per-batch + shared-x \
         modes, batch>1, t>1, signed/denormal bf16; mt weight-shared twin BIT-IDENTICAL \
         at t 5; sel grouped twin BIT-IDENTICAL to the off_into chain (shared-x + slot-x, \
         duplicate slots)",
        worst.0, worst.1
    ))
}

/// Dequantize ONE expert of a device-resident modelopt-NVFP4 stacked bank to f32.
///
/// The existing dsv4 kernel (`memra_dsv4_nvfp4_deq_bf16`) emits bf16, so the macro is
/// NOT passed into it: e2m1 × e4m3 products carry ≤ 6 significand bits and are EXACT in
/// bf16, and the macro multiplies AFTER the exact f32 upcast. That reproduces the host
/// decoder (`dsv4::dequant_nvfp4_expert`: `(code * scale) * scale_2`, one f32 rounding)
/// bit-for-bit for ANY finite macro — the real qwen4_exp mint ships modelopt's
/// amax-derived NON-pow2 `weight_scale_2` (measured 5.9945243e-5), which the dsv4-era
/// in-kernel-macro chain would round in bf16 (hence its pow2 law; not needed here).
fn dequant_nvfp4_expert_f32(
    e: &Engine,
    codes: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    macro_scale: f32,
    expert: usize,
    rows: usize,
    cols: usize,
) -> Res<CudaSlice<f32>> {
    let wbytes = rows * cols / 2;
    let sbytes = rows * cols / 16;
    let bf = e.alloc_u8(rows * cols * 2)?;
    let stream = e.gpu.stream();
    let wp = (codes.device_ptr(&stream).0 as usize + expert * wbytes) as *const c_void;
    let scp = (scales.device_ptr(&stream).0 as usize + expert * sbytes) as *const c_void;
    let dst = bf.device_ptr(&stream).0 as usize as *mut c_void;
    let rc = unsafe {
        crate::dsv4_ffi::memra_dsv4_nvfp4_deq_bf16(
            wp,
            scp,
            1.0, // macro applied post-upcast in f32 (see the doc comment)
            rows as i32,
            cols as i32,
            dst,
            stream.cu_stream() as *mut c_void,
        )
    };
    if rc != 0 {
        return Err(format!("memra_dsv4_nvfp4_deq_bf16 rc={rc}").into());
    }
    let mut out = e.bf16_to_f32(&bf.slice(0..rows * cols * 2), rows * cols)?;
    if macro_scale != 1.0 {
        e.scale_inplace(&mut out, macro_scale, rows * cols)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------- loading

fn expect(weights: &ReferenceWeights, id: &TensorId) -> Res<ReferenceTensor> {
    weights
        .get(id)
        .cloned()
        .ok_or_else(|| format!("qwen4exp_gpu: missing weight {id:?}").into())
}

fn family_id(key: String) -> TensorId {
    TensorId::Family {
        family: "qwen4_exp",
        key,
    }
}

fn layer_id(index: u32, tensor: LayerTensor) -> TensorId {
    TensorId::Layer { index, tensor }
}

fn upload(e: &Engine, tensor: &ReferenceTensor) -> Res<CudaSlice<f32>> {
    e.htod(&tensor.data)
}

/// Slice a [rows, wide] row-major tensor into per-stream [rows, hidden] column blocks.
fn split_columns(data: &[f32], rows: usize, streams: usize, hidden: usize) -> Vec<Vec<f32>> {
    let wide = streams * hidden;
    (0..streams)
        .map(|s| {
            let mut out = Vec::with_capacity(rows * hidden);
            for row in 0..rows {
                out.extend_from_slice(
                    &data[row * wide + s * hidden..row * wide + (s + 1) * hidden],
                );
            }
            out
        })
        .collect()
}

/// Slice a [wide, cols] row-major tensor into per-stream [hidden, cols] row blocks.
fn split_rows(data: &[f32], streams: usize, hidden: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..streams)
        .map(|s| data[s * hidden * cols..(s + 1) * hidden * cols].to_vec())
        .collect()
}

fn load_gate(
    e: &Engine,
    weights: &ReferenceWeights,
    prefix: &str,
    sublayer: &str,
    streams: usize,
    hidden: usize,
    rank: usize,
    with_inject: bool,
) -> Res<GateW> {
    let wide = streams * hidden;
    let norm = expect(
        weights,
        &family_id(format!("{prefix}{sublayer}hc_norm.weight")),
    )?;
    let down = expect(
        weights,
        &family_id(format!("{prefix}{sublayer}input_mix_weight_down.weight")),
    )?;
    let up = expect(
        weights,
        &family_id(format!("{prefix}{sublayer}input_mix_weight_up.weight")),
    )?;
    if norm.data.len() != wide || down.data.len() != rank * wide || up.data.len() != wide * rank {
        return Err(format!("qwen4exp_gpu: gate {prefix}{sublayer} shape mismatch").into());
    }
    let norm_slices = split_rows(&norm.data, streams, hidden, 1);
    let down_slices = split_columns(&down.data, rank, streams, hidden);
    let up_slices = split_rows(&up.data, streams, hidden, rank);
    // bf16 trunk twins, STACKED across streams so the fused gate runs one batched
    // launch per projection (guards in `bf16_twin`).
    let stack = |slices: &[Vec<f32>]| -> Vec<f32> {
        let mut out = Vec::with_capacity(slices.len() * slices[0].len());
        for s in slices {
            out.extend_from_slice(s);
        }
        out
    };
    let down_b16 = bf16_twin(e, &stack(&down_slices), hidden)?;
    let up_b16 = bf16_twin(e, &stack(&up_slices), rank)?;
    let (inject, inject_b16) = if with_inject {
        let inject = expect(
            weights,
            &family_id(format!("{prefix}{sublayer}block_inject_weight.weight")),
        )?;
        if inject.data.len() != streams * wide {
            return Err(format!("qwen4exp_gpu: inject {prefix}{sublayer} shape mismatch").into());
        }
        // Kept whole: the fused inject kernel walks [s][s2*hidden + d] directly, which is
        // exactly this tensor's row-major layout against the stream-major normed planes.
        (
            Some(e.htod(&inject.data)?),
            bf16_twin(e, &inject.data, hidden)?,
        )
    } else {
        (None, None)
    };
    Ok(GateW {
        norm_stack: e.htod(&stack(&norm_slices))?,
        norm: norm_slices
            .into_iter()
            .map(|v| e.htod(&v))
            .collect::<Result<_, _>>()?,
        down: down_slices
            .into_iter()
            .map(|v| e.htod(&v))
            .collect::<Result<_, _>>()?,
        up: up_slices
            .into_iter()
            .map(|v| e.htod(&v))
            .collect::<Result<_, _>>()?,
        inject,
        down_b16,
        up_b16,
        inject_b16,
    })
}

/// Loader-side carriers that bypass `ReferenceWeights` (the real artifact cannot
/// materialize them host-f32): device-bound expert banks and host n-gram tables,
/// keyed by trunk layer index.
#[derive(Default)]
pub struct ExternalParts {
    expert_banks: std::collections::BTreeMap<u32, ExpertBank>,
    ngram_tables: std::collections::BTreeMap<u32, NgramTable>,
}

/// Build one decoder layer's engine-resident weights from TensorId-keyed reference
/// weights — shared by the trunk loop and the MTP draft block (mtp-spec lane), which is
/// the same layer schema at global index n_trunk under the `mtp.layers.{depth}.` prefix.
#[allow(clippy::too_many_arguments)]
fn build_layer_w(
    e: &Engine,
    weights: &ReferenceWeights,
    layer: &memra_gguf::model_plan::LayerPlan,
    prefix: &str,
    streams: usize,
    hidden: usize,
    rank: usize,
    bank_override: Option<ExpertBank>,
    table_override: Option<NgramTable>,
) -> Res<LayerW> {
    let ResidualTopology::GatedResidual { .. } = layer.residual else {
        return Err(format!("qwen4exp_gpu: layer {} is not gated-residual", layer.index).into());
    };
    let attn_gate = load_gate(
        e,
        weights,
        prefix,
        "attn_hyper_connection.",
        streams,
        hidden,
        rank,
        true,
    )?;
    let mlp_gate = load_gate(
        e,
        weights,
        prefix,
        "mlp_hyper_connection.",
        streams,
        hidden,
        rank,
        true,
    )?;
    let mixer = match &layer.attention {
        AttentionPlan::Full(attn) => {
            let overlay = layer.sparse_overlay.ok_or_else(|| {
                format!(
                    "qwen4exp_gpu: QSA layer {} has no indexer overlay",
                    layer.index
                )
            })?;
            // Plain partial rope or YaRN (long-context lane); anything else refuses in
            // `build_yarn`.
            let yarn = build_yarn(e, &attn.rope, Some(&overlay), layer.index)?;
            // The eager attention path lays q/k/v/attended out with ONE head_dim
            // and gates full-width; unequal key/value dims would be silently
            // wrong, so refuse (family: 256/256).
            if attn.key_head_dim != attn.value_head_dim {
                return Err(format!(
                    "qwen4exp_gpu: layer {} key_head_dim {} != value_head_dim {}",
                    layer.index, attn.key_head_dim, attn.value_head_dim
                )
                .into());
            }
            let load_opt_norm = |tensor: LayerTensor| -> Res<Option<CudaSlice<f32>>> {
                match weights.get(&layer_id(layer.index, tensor)) {
                    Some(t) => Ok(Some(e.htod(&t.data)?)),
                    None if attn.qk_norm == TensorPresence::Required => {
                        Err(format!("qwen4exp_gpu: layer {} missing qk norm", layer.index).into())
                    }
                    None => Ok(None),
                }
            };
            let wq_t = expect(weights, &layer_id(layer.index, LayerTensor::Query))?;
            let wk_t = expect(weights, &layer_id(layer.index, LayerTensor::Key))?;
            let wv_t = expect(weights, &layer_id(layer.index, LayerTensor::Value))?;
            let wo_t = expect(
                weights,
                &layer_id(layer.index, LayerTensor::AttentionOutput),
            )?;
            let o_in = (attn.query_heads * attn.key_head_dim) as usize;
            MixerW::Qsa(QsaW {
                attn: attn.clone(),
                overlay,
                yarn,
                proj_b16: bf16_stack_twin(e, &[&wq_t.data, &wk_t.data, &wv_t.data], hidden)?,
                wo_b16: bf16_twin(e, &wo_t.data, o_in)?,
                wq: upload(e, &wq_t)?,
                wk: upload(e, &wk_t)?,
                wv: upload(e, &wv_t)?,
                wo: upload(e, &wo_t)?,
                q_norm: load_opt_norm(LayerTensor::QueryNorm)?,
                k_norm: load_opt_norm(LayerTensor::KeyNorm)?,
                idx_proj: upload(
                    e,
                    &expect(
                        weights,
                        &family_id(format!("{prefix}self_attn.indexer.index_qk_proj.weight")),
                    )?,
                )?,
                idx_q_norm: expect(
                    weights,
                    &family_id(format!("{prefix}self_attn.indexer.q_layernorm.weight")),
                )?
                .data,
                idx_k_norm: expect(
                    weights,
                    &family_id(format!("{prefix}self_attn.indexer.k_layernorm.weight")),
                )?
                .data,
            })
        }
        AttentionPlan::GatedDeltaNet(gdn) => {
            let qkv_t = expect(weights, &layer_id(layer.index, LayerTensor::GdnQkv))?;
            let z_t = expect(weights, &layer_id(layer.index, LayerTensor::GdnGate))?;
            let beta_t = expect(weights, &layer_id(layer.index, LayerTensor::GdnBeta))?;
            let alpha_t = expect(weights, &layer_id(layer.index, LayerTensor::GdnAlpha))?;
            let out_t = expect(weights, &layer_id(layer.index, LayerTensor::GdnOutput))?;
            let o_in = (gdn.value_heads * gdn.value_head_dim) as usize;
            MixerW::Gdn(GdnW {
                plan: *gdn,
                proj_b16: bf16_stack_twin(
                    e,
                    &[&qkv_t.data, &z_t.data, &beta_t.data, &alpha_t.data],
                    hidden,
                )?,
                out_b16: bf16_twin(e, &out_t.data, o_in)?,
                qkv: upload(e, &qkv_t)?,
                z: upload(e, &z_t)?,
                beta: upload(e, &beta_t)?,
                alpha: upload(e, &alpha_t)?,
                conv_w: upload(
                    e,
                    &expect(weights, &layer_id(layer.index, LayerTensor::GdnConv1d))?,
                )?,
                a: upload(
                    e,
                    &expect(weights, &layer_id(layer.index, LayerTensor::GdnA))?,
                )?,
                dt: upload(
                    e,
                    &expect(weights, &layer_id(layer.index, LayerTensor::GdnDtBias))?,
                )?,
                norm: upload(
                    e,
                    &expect(weights, &layer_id(layer.index, LayerTensor::GdnNorm))?,
                )?,
                out: upload(e, &out_t)?,
            })
        }
        other => {
            return Err(format!(
                "qwen4exp_gpu: unsupported mixer {other:?} at layer {}",
                layer.index
            )
            .into());
        }
    };
    let MlpPlan::Moe(moe_plan) = &layer.mlp else {
        return Err(format!("qwen4exp_gpu: layer {} is not MoE", layer.index).into());
    };
    if !matches!(moe_plan.router, RouterPlan::Softmax) {
        return Err("qwen4exp_gpu: only the softmax router arm is implemented".into());
    }
    let shared = moe_plan
        .shared
        .as_ref()
        .ok_or("qwen4exp_gpu: missing shared expert plan")?;
    let bank = match bank_override {
        Some(bank) => bank,
        None => {
            let gate = expect(
                weights,
                &layer_id(layer.index, LayerTensor::MoeExpertGateBank),
            )?;
            let up = expect(
                weights,
                &layer_id(layer.index, LayerTensor::MoeExpertUpBank),
            )?;
            let down = expect(
                weights,
                &layer_id(layer.index, LayerTensor::MoeExpertDownBank),
            )?;
            let experts = moe_plan.expert_count as usize;
            let ff = moe_plan.expert_intermediate_size as usize;
            if gate.data.len() != experts * ff * hidden
                || up.data.len() != experts * ff * hidden
                || down.data.len() != experts * hidden * ff
            {
                return Err(format!(
                    "qwen4exp_gpu: layer {} expert bank shape mismatch",
                    layer.index
                )
                .into());
            }
            ExpertBank {
                gate: BankHalf::F32(e.htod(&gate.data)?),
                up: BankHalf::F32(e.htod(&up.data)?),
                down: BankHalf::F32(e.htod(&down.data)?),
            }
        }
    };
    let sh_gate_t = expect(weights, &layer_id(layer.index, LayerTensor::SharedMlpGate))?;
    let sh_up_t = expect(weights, &layer_id(layer.index, LayerTensor::SharedMlpUp))?;
    let sh_down_t = expect(weights, &layer_id(layer.index, LayerTensor::SharedMlpDown))?;
    let sff = shared.intermediate_size as usize;
    let router_t = expect(weights, &layer_id(layer.index, LayerTensor::MoeRouter))?;
    let moe = MoeW {
        plan: moe_plan.clone(),
        router_b16: bf16_twin(e, &router_t.data, hidden)?,
        router: upload(e, &router_t)?,
        bank,
        shared_gu_b16: bf16_stack_twin(e, &[&sh_gate_t.data, &sh_up_t.data], hidden)?,
        shared_down_b16: bf16_twin(e, &sh_down_t.data, sff)?,
        shared_gate: upload(e, &sh_gate_t)?,
        shared_up: upload(e, &sh_up_t)?,
        shared_down: upload(e, &sh_down_t)?,
        shared_input_gate: if shared.gated {
            Some(upload(
                e,
                &expect(
                    weights,
                    &layer_id(layer.index, LayerTensor::SharedMlpInputGate),
                )?,
            )?)
        } else {
            None
        },
    };
    let ple = match layer.ple.as_ref() {
        None => None,
        Some(ple_plan) => {
            let embed_dim = ple_plan.embed_dim as usize;
            let head_dim = ple_plan.head_embed_dim as usize;
            let wide = streams * hidden;
            let key_proj = expect(weights, &family_id(format!("{prefix}ple.key_proj.weight")))?;
            let conv_w = expect(weights, &family_id(format!("{prefix}ple.conv1d.weight")))?;
            if key_proj.data.len() != wide * embed_dim {
                return Err("qwen4exp_gpu: ple key_proj shape mismatch".into());
            }
            let norm_slices = |name: &str| -> Res<Vec<CudaSlice<f32>>> {
                let t = expect(weights, &family_id(format!("{prefix}ple.{name}.weight")))?;
                split_rows(&t.data, streams, hidden, 1)
                    .into_iter()
                    .map(|v| e.htod(&v))
                    .collect::<Result<_, _>>()
            };
            let ints = |name: &str| -> Res<Vec<i64>> {
                let t = expect(
                    weights,
                    &family_id(format!("{prefix}ple.ple_embedding.{name}")),
                )?;
                t.ints
                    .clone()
                    .ok_or_else(|| "qwen4exp_gpu: n-gram buffer must be I64".into())
            };
            let table = match table_override {
                Some(table) => table,
                None => {
                    let t = expect(
                        weights,
                        &family_id(format!("{prefix}ple.ple_embedding.ngram_embedding")),
                    )?;
                    if t.shape.len() != 2 || t.shape[1] != head_dim {
                        return Err("qwen4exp_gpu: n-gram table shape mismatch".into());
                    }
                    NgramTable::F32(t.data)
                }
            };
            Some(PleW {
                plan: *ple_plan,
                key_proj: split_rows(&key_proj.data, streams, hidden, embed_dim)
                    .into_iter()
                    .map(|v| e.htod(&v))
                    .collect::<Result<_, _>>()?,
                value_proj: upload(
                    e,
                    &expect(
                        weights,
                        &family_id(format!("{prefix}ple.value_proj.weight")),
                    )?,
                )?,
                norm_key: norm_slices("norm_key")?,
                norm_query: norm_slices("norm_query")?,
                norm_conv: norm_slices("norm_conv")?,
                conv_w: split_rows(&conv_w.data, streams, hidden, ple_plan.conv_kernel as usize)
                    .into_iter()
                    .map(|v| e.htod(&v))
                    .collect::<Result<_, _>>()?,
                multipliers: ints("layer_multipliers")?,
                sizes: ints("ngram_heads_vocab_sizes")?,
                offsets: ints("ngram_heads_offsets")?,
                table,
            })
        }
    };
    Ok(LayerW {
        index: layer.index,
        eps_attn: layer.pre_attention_norm.epsilon,
        eps_mlp: layer.pre_mlp_norm.epsilon,
        attn_gate,
        mlp_gate,
        mixer,
        moe,
        ple,
    })
}

/// Build the MTP draft block (SEMANTICS.md §MTP): fusion glue + the one decoder layer +
/// the draft's own exit mixer. The lm_head is SHARED with the trunk (`self.output`).
#[allow(clippy::too_many_arguments)]
fn build_mtp_w(
    e: &Engine,
    weights: &ReferenceWeights,
    block: &memra_gguf::model_plan::MtpBlockPlan,
    streams: usize,
    hidden: usize,
    rank: usize,
    bank_override: Option<ExpertBank>,
) -> Res<MtpW> {
    use memra_gguf::tensor_contract::MtpTensor;
    if block.input.fusion != memra_gguf::model_plan::MtpFusionPlan::SeparateProjections {
        return Err("qwen4exp_gpu: MTP block is not the separate-projections family".into());
    }
    let wide = streams * hidden;
    let depth = block.depth;
    let mtp_id = |tensor: MtpTensor| TensorId::Mtp { depth, tensor };
    let pre_e = expect(weights, &mtp_id(MtpTensor::EmbeddingNorm))?;
    let pre_h = expect(weights, &mtp_id(MtpTensor::HiddenNorm))?;
    let fc_e = expect(weights, &mtp_id(MtpTensor::EmbeddingProjection))?;
    let fc_h = expect(weights, &mtp_id(MtpTensor::HiddenProjection))?;
    if pre_e.data.len() != hidden
        || pre_h.data.len() != wide
        || fc_e.data.len() != hidden * hidden
        || fc_h.data.len() != hidden * hidden
    {
        return Err("qwen4exp_gpu: MTP fusion tensor shape mismatch".into());
    }
    let prefix = format!("mtp.layers.{depth}.");
    let layer = build_layer_w(
        e,
        weights,
        &block.layer,
        &prefix,
        streams,
        hidden,
        rank,
        bank_override,
        None,
    )?;
    let mixer = load_gate(
        e,
        weights,
        "mtp.hyper_connection_mixer.",
        "",
        streams,
        hidden,
        rank,
        false,
    )?;
    Ok(MtpW {
        eps_embed: block.input.embedding_norm.epsilon,
        eps_hidden: block.input.hidden_norm.epsilon,
        fc_embed_b16: bf16_twin(e, &fc_e.data, hidden)?,
        fc_hidden_b16: bf16_twin(e, &fc_h.data, hidden)?,
        pre_norm_embed: upload(e, &pre_e)?,
        pre_norm_hidden: upload(e, &pre_h)?,
        fc_embed: upload(e, &fc_e)?,
        fc_hidden: upload(e, &fc_h)?,
        layer,
        mixer,
    })
}

impl Qwen4ExpGpu {
    /// Build the eager model from TensorId-keyed reference weights (the deterministic tiny
    /// fixture, or a checkpoint materialized through `read_checkpoint`'s binding walk).
    /// Effective (already-folded) norm weights; reference layout throughout.
    pub fn from_reference_weights(
        e: &Engine,
        plan: &ModelPlan,
        weights: &ReferenceWeights,
    ) -> Res<Self> {
        Self::from_reference_weights_with(e, None, plan, weights, ExternalParts::default())
    }

    fn from_reference_weights_with(
        e: &Engine,
        // Card-1 draft placement (mtp10): when given, the MTP block's device tensors and
        // a private lm-head copy build on THIS engine instead of `e`.
        draft_e: Option<&Engine>,
        plan: &ModelPlan,
        weights: &ReferenceWeights,
        mut parts: ExternalParts,
    ) -> Res<Self> {
        let hidden = plan.hidden_size as usize;
        let vocab = plan.vocab_size as usize;
        let Some(mixer_plan) = plan.exit_mixer else {
            return Err("qwen4exp_gpu requires the gated-residual exit mixer".into());
        };
        let streams = mixer_plan.streams as usize;
        if streams > PLANE_SLOTS.len() {
            return Err("qwen4exp_gpu: hc_count exceeds the step-workspace slot table".into());
        }
        let rank = mixer_plan.bottleneck_rank as usize;
        if !plan.logits.is_empty() {
            return Err("qwen4exp_gpu: logits transforms are not part of this family".into());
        }

        let embed = expect(weights, &TensorId::TokenEmbedding)?;
        if embed.data.len() != vocab * hidden {
            return Err("qwen4exp_gpu: embedding shape mismatch".into());
        }
        let (output, output_b16) = match weights.get(&TensorId::OutputProjection) {
            Some(tensor) => (e.htod(&tensor.data)?, bf16_twin(e, &tensor.data, hidden)?),
            None => (e.htod(&embed.data)?, bf16_twin(e, &embed.data, hidden)?),
        };

        let mut layers = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            let prefix = format!("trunk.layers.{}.", layer.index);
            layers.push(build_layer_w(
                e,
                weights,
                layer,
                &prefix,
                streams,
                hidden,
                rank,
                parts.expert_banks.remove(&layer.index),
                parts.ngram_tables.remove(&layer.index),
            )?);
        }
        // The MTP draft block (mtp-spec lane): built when its rows are present in the
        // materialized weights — presence-driven, so the deterministic fixture carries
        // it and a checkpoint loaded without `LoadOptions::load_mtp` skips it.
        let mtp = match plan.mtp_blocks.first() {
            Some(block)
                if weights
                    .get(&TensorId::Mtp {
                        depth: block.depth,
                        tensor: memra_gguf::tensor_contract::MtpTensor::EmbeddingProjection,
                    })
                    .is_some() =>
            {
                Some(build_mtp_w(
                    draft_e.unwrap_or(e),
                    weights,
                    block,
                    streams,
                    hidden,
                    rank,
                    parts.expert_banks.remove(&block.layer.index),
                )?)
            }
            _ => None,
        };
        // Card-1 lm-head copy for the dev1 draft: the SAME f32 rows and the SAME bf16
        // twin bytes as card 0's head, so the draft head program is verbatim.
        let mtp_dev1 = match (draft_e, mtp.as_ref()) {
            (Some(de), Some(_)) => {
                let head_data: &[f32] = match weights.get(&TensorId::OutputProjection) {
                    Some(tensor) => &tensor.data,
                    None => &embed.data,
                };
                Some(MtpDev1 {
                    dev: de.ctx().ordinal(),
                    output: de.htod(head_data)?,
                    output_b16: bf16_twin(de, head_data, hidden)?,
                })
            }
            (Some(_), None) => {
                return Err(
                    "qwen4exp_gpu: a draft engine was given but no mtp.* rows were \
                     materialized (LoadOptions::load_mtp)"
                        .into(),
                );
            }
            _ => None,
        };
        let exit_mixer = load_gate(
            e,
            weights,
            "trunk.hyper_connection_mixer.",
            "",
            streams,
            hidden,
            rank,
            false,
        )?;
        Ok(Self {
            plan: plan.clone(),
            hidden,
            streams,
            vocab,
            embed_host: embed.data,
            output,
            output_b16,
            layers,
            exit_mixer,
            exit_eps: plan.output_norm.epsilon,
            mtp,
            mtp_dev1,
            draft_trim: None,
            draft_trim_parked: None,
            chain_embed: None,
        })
    }

    /// Arm the FR-Spec draft-head trim (mtp9): gather the `ids` rows of the SHARED lm head
    /// into a [n, hidden] trimmed head, D2D — same bytes, so every trimmed logit is
    /// bit-identical to its full-vocab twin. `ids` is the own-gen rank list in rank order
    /// (most frequent first); duplicates and out-of-range ids are rejected.
    ///
    /// Arming changes what the DRAFT can propose (acceptance), never what the model
    /// commits: the verify chunk is full-vocab and the accept walk compares against it.
    pub fn build_draft_trim(&mut self, e: &Engine, ids: &[u32]) -> Res<()> {
        // Card-1 placement: the trim gathers from the DEV1 head copy (same bytes as
        // card 0's) and its rows live beside the draft — `e` must be the draft engine.
        self.check_draft_engine(e)?;
        let n = ids.len();
        if n == 0 || n > self.vocab {
            return Err(format!("qwen4exp_gpu: draft trim wants 1..={} ids", self.vocab).into());
        }
        let mut seen = vec![false; self.vocab];
        for &id in ids {
            let id = id as usize;
            if id >= self.vocab {
                return Err(format!("qwen4exp_gpu: draft trim id {id} out of vocab").into());
            }
            if std::mem::replace(&mut seen[id], true) {
                return Err(format!("qwen4exp_gpu: draft trim id {id} repeats").into());
            }
        }
        let hidden = self.hidden;
        let (src_f32, src_b16) = match self.mtp_dev1.as_ref() {
            Some(d) => (&d.output, d.output_b16.as_ref()),
            None => (&self.output, self.output_b16.as_ref()),
        };
        // Gather the bf16 twin when it exists (the arm the trunk seam runs) and SKIP the
        // f32 gather entirely — at N=32768 that is 168 MB instead of 503 MB, and the f32
        // arm would be dead residency. No twin => gather f32, the only arm available.
        let (head_b16, head) = match src_b16 {
            Some(full) => {
                let mut trim = e.alloc_u8_uninit(n * hidden * 2)?;
                for (row, &id) in ids.iter().enumerate() {
                    e.copy_u8_range_into(
                        &mut trim,
                        row * hidden * 2,
                        full,
                        id as usize * hidden * 2,
                        hidden * 2,
                    )?;
                }
                (Some(trim), None)
            }
            None => {
                let mut head = e.uninit(n * hidden)?;
                for (row, &id) in ids.iter().enumerate() {
                    e.copy_range_into(
                        &mut head,
                        row * hidden,
                        src_f32,
                        id as usize * hidden,
                        hidden,
                    )?;
                }
                (None, Some(head))
            }
        };
        self.draft_trim = Some(DraftTrim {
            n,
            d2t: ids.to_vec(),
            head,
            head_b16,
        });
        self.draft_trim_parked = None;
        Ok(())
    }

    /// Flip a BUILT trim between live and parked (the interleaved A/B's two arms) without
    /// reallocating the gathered head. No-op when no trim was ever built.
    pub fn set_draft_trim(&mut self, on: bool) {
        if on {
            if let Some(t) = self.draft_trim_parked.take() {
                self.draft_trim = Some(t);
            }
        } else if let Some(t) = self.draft_trim.take() {
            self.draft_trim_parked = Some(t);
        }
    }

    /// Drop the draft trim entirely (both live and parked).
    pub fn clear_draft_trim(&mut self) {
        self.draft_trim = None;
        self.draft_trim_parked = None;
    }

    /// Arm the deferred-chain embed table (mtp11, `SpecOpts::defer`): the chain's
    /// next-step embed rows, resident on the DRAFT engine, so the device argmax feeds
    /// the next chain step without a host round trip (see [`ChainEmbed`] for the
    /// bf16-clean bit-identity contract and the trim-rank row order). Re-arm after any
    /// trim change — `spec_generate_ext` refuses a table whose trim state or width
    /// disagrees with the live draft head.
    pub fn arm_spec_devchain(&mut self, de: &Engine) -> Res<()> {
        self.check_draft_engine(de)?;
        let hidden = self.hidden;
        let (rows, for_trim) = match self.draft_trim.as_ref() {
            Some(tr) => (tr.n, true),
            None => (self.vocab, false),
        };
        let src_row = |r: usize| -> &[f32] {
            let id = match self.draft_trim.as_ref() {
                Some(tr) => tr.d2t[r] as usize,
                None => r,
            };
            &self.embed_host[id * hidden..(id + 1) * hidden]
        };
        // bf16-clean scan over the SELECTED rows: every value must round-trip
        // f32 -> bits>>16 -> bits<<16 exactly, or the table falls back to raw f32.
        let clean = (0..rows).all(|r| src_row(r).iter().all(|x| x.to_bits() & 0xFFFF == 0));
        let (bytes, qt, row_bytes) = if clean {
            let mut b = vec![0u8; rows * hidden * 2];
            for r in 0..rows {
                for (j, &x) in src_row(r).iter().enumerate() {
                    let h = (x.to_bits() >> 16) as u16;
                    b[(r * hidden + j) * 2..(r * hidden + j) * 2 + 2]
                        .copy_from_slice(&h.to_le_bytes());
                }
            }
            (b, crate::QT_BF16, hidden * 2)
        } else {
            let mut b = vec![0u8; rows * hidden * 4];
            for r in 0..rows {
                for (j, &x) in src_row(r).iter().enumerate() {
                    b[(r * hidden + j) * 4..(r * hidden + j) * 4 + 4]
                        .copy_from_slice(&x.to_le_bytes());
                }
            }
            (b, crate::QT_F32, hidden * 4)
        };
        let table = de.upload_u8(&bytes)?;
        println!(
            "[qwen4exp-spec] deferred-chain embed table armed: {} rows x {hidden} ({}, {:.1} MiB, dev {}{})",
            rows,
            if clean {
                "bf16 bit-clean"
            } else {
                "f32 fallback"
            },
            (rows * row_bytes) as f64 / (1024.0 * 1024.0),
            de.ctx().ordinal(),
            if for_trim { ", trim-rank order" } else { "" },
        );
        self.chain_embed = Some(ChainEmbed {
            table,
            qt,
            row_bytes,
            rows,
            for_trim,
            dev: de.ctx().ordinal(),
        });
        Ok(())
    }

    /// Drop the deferred-chain embed table (frees the card-1 residency).
    pub fn clear_spec_devchain(&mut self) {
        self.chain_embed = None;
    }

    /// Rows the draft's lm_head produces: the trim width when armed, else full vocab.
    /// Draft logits live in TRIMMED space when armed; `draft_token` maps a row back.
    pub fn draft_logits_width(&self) -> usize {
        match self.draft_trim.as_ref() {
            Some(t) => t.n,
            None => self.vocab,
        }
    }

    /// Map a draft-logits row index back to its TARGET vocab id (identity when the trim
    /// is off).
    fn draft_token(&self, row: u32) -> Res<u32> {
        match self.draft_trim.as_ref() {
            Some(t) => t
                .d2t
                .get(row as usize)
                .copied()
                .ok_or_else(|| format!("qwen4exp_gpu: draft row {row} outside the trim").into()),
            None => Ok(row),
        }
    }

    /// Trunk f32 diet (yarn-cell follow-up 3): FREE the f32 originals whose bf16 twins
    /// are resident — under the ship seams (trunk-bf16 + fused-gate, both default ON)
    /// every consumer of these tensors runs the bf16 kernels at every t, so the f32
    /// copies are pure dead residency (~6 GiB on card 0 at the real geometry). Each
    /// dropped tensor becomes a 1-element stub; every f32 fallback path guards on the
    /// stub and errs loudly instead of reading it (flipping the trunk seams OFF after
    /// the diet refuses rather than corrupting). Returns bytes freed. NOT applied to
    /// the MTP draft weights (card-1 slack; the reference-parity gates read them).
    pub fn trunk_f32_diet(&mut self, e: &Engine) -> Res<usize> {
        if !trunk_bf16_on() || !hc_fused_gate_on() {
            return Err(
                "qwen4exp_gpu: trunk_f32_diet requires the trunk-bf16 + fused-gate seams ON \
                 (the bf16 paths must be the ones serving)"
                    .into(),
            );
        }
        let mut freed = 0usize;
        fn stub(e: &Engine, s: &mut CudaSlice<f32>, freed: &mut usize) -> Res<()> {
            if s.len() > 1 {
                *freed += s.len() * 4;
                *s = e.zeros(1)?;
            }
            Ok(())
        }
        fn diet_gate(e: &Engine, g: &mut GateW, freed: &mut usize) -> Res<()> {
            if g.down_b16.is_none()
                || g.up_b16.is_none()
                || (g.inject.is_some() && g.inject_b16.is_none())
            {
                return Ok(()); // partial twins: keep the f32 arm whole
            }
            for s in g.down.iter_mut() {
                stub(e, s, freed)?;
            }
            for s in g.up.iter_mut() {
                stub(e, s, freed)?;
            }
            if let Some(inj) = g.inject.as_mut() {
                stub(e, inj, freed)?;
            }
            Ok(())
        }
        for layer in self.layers.iter_mut() {
            diet_gate(e, &mut layer.attn_gate, &mut freed)?;
            diet_gate(e, &mut layer.mlp_gate, &mut freed)?;
            match &mut layer.mixer {
                MixerW::Qsa(q) => {
                    if q.proj_b16.is_some() {
                        stub(e, &mut q.wq, &mut freed)?;
                        stub(e, &mut q.wk, &mut freed)?;
                        stub(e, &mut q.wv, &mut freed)?;
                    }
                    if q.wo_b16.is_some() {
                        stub(e, &mut q.wo, &mut freed)?;
                    }
                }
                MixerW::Gdn(g) => {
                    if g.proj_b16.is_some() {
                        stub(e, &mut g.qkv, &mut freed)?;
                        stub(e, &mut g.z, &mut freed)?;
                        stub(e, &mut g.beta, &mut freed)?;
                        stub(e, &mut g.alpha, &mut freed)?;
                    }
                    if g.out_b16.is_some() {
                        stub(e, &mut g.out, &mut freed)?;
                    }
                }
            }
            let moe = &mut layer.moe;
            if moe.router_b16.is_some() {
                stub(e, &mut moe.router, &mut freed)?;
            }
            if moe.shared_gu_b16.is_some() {
                stub(e, &mut moe.shared_gate, &mut freed)?;
                stub(e, &mut moe.shared_up, &mut freed)?;
            }
            if moe.shared_down_b16.is_some() {
                stub(e, &mut moe.shared_down, &mut freed)?;
            }
        }
        diet_gate(e, &mut self.exit_mixer, &mut freed)?;
        if self.output_b16.is_some() {
            stub(e, &mut self.output, &mut freed)?;
        }
        Ok(freed)
    }

    pub fn alloc_state(&self, e: &Engine, capacity: usize) -> Res<Qwen4ExpState> {
        self.alloc_state_reserve(e, capacity, capacity, None)
    }

    /// Long-context state: `reserve` caps the workspace-slot unit at the chunk bound
    /// (see `Qwen4ExpState::reserve`), and `kv_engine` optionally places the QSA KV
    /// caches on ANOTHER card (the kv-dev1 ladder arm: card 0 holds the trunk at
    /// ~90 GiB; the attention kernels read K/V over UVA P2P). `None` = same card.
    pub fn alloc_state_reserve(
        &self,
        e: &Engine,
        capacity: usize,
        reserve: usize,
        kv_engine: Option<&Engine>,
    ) -> Res<Qwen4ExpState> {
        let kv_e = kv_engine.unwrap_or(e);
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mixer = match &layer.mixer {
                MixerW::Qsa(qsa) => {
                    let kv_width = qsa.attn.kv_heads as usize * qsa.attn.key_head_dim as usize;
                    let v_width = qsa.attn.kv_heads as usize * qsa.attn.value_head_dim as usize;
                    // kvq/idxq lanes: the storage format latches PER STATE here (a byte
                    // cache cannot flip mid-run; the A/B harness allocates per arm).
                    let kv = if kv_quant_on() {
                        QsaKvStore::Q8Q5 {
                            k: kv_e.alloc_u8(capacity * q8_row_bytes(kv_width))?,
                            v: kv_e.alloc_u8(capacity * q5_row_bytes(v_width))?,
                        }
                    } else {
                        QsaKvStore::F32 {
                            k: kv_e.zeros(capacity * kv_width)?,
                            v: kv_e.zeros(capacity * v_width)?,
                        }
                    };
                    MixerState::Qsa {
                        kv,
                        raw_keys: IdxRawCache::new(idxq_mode()),
                        pooled_keys: Vec::new(),
                        pooled_dev: None,
                        pooled_dev_rows: 0,
                        raw_dev: None,
                        raw_dev_rows: 0,
                        idx_audit: (idxq_mode() != IdxQMode::F32 && idxq_audit_on()).then(|| {
                            Box::new(IdxAudit {
                                raw_f32: IdxRawCache::F32(Vec::new()),
                                pooled_f32: Vec::new(),
                            })
                        }),
                    }
                }
                MixerW::Gdn(gdn) => {
                    let p = &gdn.plan;
                    let conv_dim = 2 * (p.key_heads * p.key_head_dim) as usize
                        + (p.value_heads * p.value_head_dim) as usize;
                    let pad = p.conv_kernel as usize - 1;
                    MixerState::Gdn {
                        conv: e.zeros(pad * conv_dim)?,
                        state: e
                            .zeros((p.value_heads * p.value_head_dim * p.key_head_dim) as usize)?,
                    }
                }
            };
            let ple = match layer.ple.as_ref() {
                None => None,
                Some(ple) => {
                    let pad = (ple.plan.conv_kernel as usize - 1) * ple.plan.max_ngram as usize;
                    let mut conv_hist = Vec::with_capacity(self.streams);
                    for _ in 0..self.streams {
                        conv_hist.push(e.zeros(pad * self.hidden)?);
                    }
                    Some(PleState {
                        conv_hist,
                        ngram_ids: Vec::new(),
                        ngram_history: Vec::new(),
                        ngram_last_eos: -1,
                    })
                }
            };
            layers.push(LayerState { mixer, ple });
        }
        Ok(Qwen4ExpState {
            pos: 0,
            capacity,
            reserve,
            tokens: Vec::new(),
            layers,
            ws: StepPool::default(),
            graphs: StepGraphs::default(),
            tp2: None,
            verify: None,
        })
    }

    /// Prefill `ids` from the state's current position. Returns [t, vocab] logits (host).
    pub fn prefill(&self, e: &Engine, ids: &[u32], state: &mut Qwen4ExpState) -> Res<Vec<f32>> {
        self.forward(e, ids, state, None)
    }

    /// LONG-context chunked prefill: forward `ids` in `chunk`-sized pieces from the
    /// state's current position, skipping the exit mixer + lm_head on every chunk but
    /// materializing ONLY the final row's logits at the end. State-identical to one big
    /// `prefill` (the head reads no state and writes none); the [t, vocab] logits block
    /// a big chunk would otherwise materialize is the thing being skipped (16 GB at
    /// chunk 16384 on this vocab). Returns the LAST row's logits [vocab].
    pub fn prefill_extend(
        &self,
        e: &Engine,
        ids: &[u32],
        state: &mut Qwen4ExpState,
        chunk: usize,
    ) -> Res<Vec<f32>> {
        if ids.is_empty() || chunk == 0 {
            return Err("qwen4exp_gpu: prefill_extend needs ids and a chunk size".into());
        }
        let mut last = Vec::new();
        for piece in ids.chunks(chunk) {
            let is_last =
                piece.as_ptr() as usize + piece.len() * 4 == ids.as_ptr() as usize + ids.len() * 4;
            let head = if is_last {
                HeadMode::LastRow
            } else {
                HeadMode::Skip
            };
            last = self.forward_with(e, piece, state, None, head)?;
        }
        Ok(last)
    }

    /// One incremental decode step (no prompt recompute). Returns [vocab] logits (host).
    pub fn decode_step(&self, e: &Engine, token: u32, state: &mut Qwen4ExpState) -> Res<Vec<f32>> {
        self.forward(e, &[token], state, None)
    }

    /// Prefill with per-layer parity capture (the transformers hidden-goldens hook
    /// points): post-layer WIDE rows per trunk layer + the exit mixer output.
    pub fn prefill_captured(
        &self,
        e: &Engine,
        ids: &[u32],
        state: &mut Qwen4ExpState,
    ) -> Res<(Vec<f32>, PrefillCapture)> {
        let mut capture = PrefillCapture {
            layer_wide: Vec::with_capacity(self.layers.len()),
            exit_mixed: Vec::new(),
        };
        let logits = self.forward(e, ids, state, Some(&mut capture))?;
        Ok((logits, capture))
    }

    /// Interleave stream-major planes into token-major wide rows [t, streams*hidden]
    /// (the HF wide-stream layout: token row = concat over streams).
    fn planes_to_wide(&self, e: &Engine, planes: &[CudaSlice<f32>], t: usize) -> Res<Vec<f32>> {
        let hidden = self.hidden;
        let wide = self.streams * hidden;
        let mut out = vec![0.0f32; t * wide];
        for (s, plane) in planes.iter().enumerate() {
            // Slice: workspace planes are reserve-sized (>= t*hidden).
            let host = e.dtoh_view(&plane.slice(0..t * hidden))?;
            for row in 0..t {
                out[row * wide + s * hidden..row * wide + (s + 1) * hidden]
                    .copy_from_slice(&host[row * hidden..(row + 1) * hidden]);
            }
        }
        Ok(out)
    }

    fn forward(
        &self,
        e: &Engine,
        ids: &[u32],
        state: &mut Qwen4ExpState,
        capture: Option<&mut PrefillCapture>,
    ) -> Res<Vec<f32>> {
        self.forward_with(e, ids, state, capture, HeadMode::All)
    }

    fn forward_with(
        &self,
        e: &Engine,
        ids: &[u32],
        state: &mut Qwen4ExpState,
        mut capture: Option<&mut PrefillCapture>,
        head: HeadMode,
    ) -> Res<Vec<f32>> {
        let t = ids.len();
        let hidden = self.hidden;
        if t == 0 {
            return Err("qwen4exp_gpu: empty input".into());
        }
        if head != HeadMode::All {
            // Head-skipping forwards are a chunked-prefill shape: goldens capture wants
            // every row, and a verify-EXACT chunk (t <= k_cap) or a t == 1 step feeds
            // the argmax sink from the full logits block. Big verify-armed chunks are
            // fine — the wide capture happens before the head, and the spec co-prefill
            // is exactly this shape.
            if capture.is_some() {
                return Err("qwen4exp_gpu: prefill capture wants every logits row".into());
            }
            if let Some(v) = state.verify.as_ref()
                && (t == 1 || t <= v.k_cap)
            {
                return Err(
                    "qwen4exp_gpu: head-skipping forward on a verify-exact chunk shape".into(),
                );
            }
        }
        if state.pos + t > state.capacity {
            return Err("qwen4exp_gpu: state capacity exceeded".into());
        }
        if state.tp2.is_some() {
            return Err(
                "qwen4exp_gpu: state already decoded in TP2 mode; single-card forward \
                 requires a fresh state (the half-state migration is one-way)"
                    .into(),
            );
        }
        let base_pos = state.pos;
        state.tokens.extend_from_slice(ids);
        // A multi-token chunk can GROW workspace slots (reallocation) — any captured
        // graph would keep the stale baked addresses, so invalidate them first.
        if t > 1 {
            state.graphs = StepGraphs::default();
        }
        // Decode graphs never engage on an ARMED-verify state (mtp11): the graphs tail
        // (`forward_graphs_tail`) carries neither the wide capture nor the argmax sink,
        // so two consecutive t == 1 forwards with verify armed (= consecutive zero-draft
        // rounds under the p-min guard) would route the second through the tail and skip
        // the wide row the next replay seeds from — an acceptance-only degradation the
        // byte-identity gates cannot see (the mtp11 audit's found-while-auditing item).
        let graphs_mode = t == 1
            && decode_graphs_on()
            && step_ws_on()
            && hc_fused_gate_on()
            && !prof::on()
            && capture.is_none()
            && state.verify.is_none();
        let tokens = &state.tokens;
        let ws = &mut state.ws;
        // Verify instrument (mtp-spec lane): while armed, capture the final wide rows
        // every forward; 1 < t <= k_cap chunks additionally run the EXACT row programs
        // (each row bit-identical to t == 1 decode) and stash per-column GDN/PLE state.
        let verify = state.verify.as_mut();
        let (exact, vfused, stash_gdn, stash_ple, stash_wide, argmax_sink, last_row_only) =
            match verify {
                Some(v) => {
                    // Verify chunks NEVER include the prefill (base_pos == 0): a
                    // prompt shorter than k_cap would otherwise prefill through the
                    // per-row DECODE programs while the plain baseline prefills FUSED —
                    // bit-different state from token 0 that drifts until the first
                    // thin-margin argmax flips. Found by the mtp11 256-token battery
                    // (raw prompt 2, len 6, K=5: k_cap 6 >= 6 -> exact prefill ->
                    // divergence at gen 157; K<=4 fused the same prefill and passed);
                    // latent since mtp-spec (every green spec-gate ran 64 tokens, and
                    // the tiny fixture's 18-token prompt never fit inside k_cap).
                    // The `vfuse` cost instrument moves the SAME chunk shape onto the
                    // fused program; it does not widen the shape, so this gen-157 rule
                    // holds unchanged on both arms.
                    let vchunk = base_pos > 0 && t > 1 && t <= v.k_cap;
                    let vfused = vchunk && verify_fused_on();
                    let exact = vchunk && !vfused;
                    // mtp11 deferred round: the t == 1 steps (zero-draft verify, dynk
                    // plain tail) take the argmax fast path too — same sink, same
                    // bit-identical device argmax, a 4-byte dtoh instead of ~1 MB.
                    let amx_t1 = t == 1 && v.want_argmax_t1;
                    if exact {
                        v.chunk = Some((base_pos, t));
                        v.argmax.clear();
                    } else if (vfused || amx_t1) && v.want_argmax {
                        v.argmax.clear();
                    }
                    if vfused {
                        // Rewind has no per-column stash to restore from on this arm —
                        // record the shape so `verify_rewind` can refuse by NAME instead
                        // of reporting "no live verify chunk" and reading like a bug.
                        v.fused_chunk = Some((base_pos, t));
                    }
                    (
                        exact,
                        vfused,
                        Some(&mut v.gdn),
                        Some(&mut v.ple),
                        Some((&mut v.wide, v.ring_rows)),
                        // The fused arm feeds the SAME argmax sink, so the A/B compares
                        // programs and not readback sizes (a full [t, vocab] dtoh on one
                        // arm only would be ~6 MB of measured noise at t=6).
                        if (exact || vfused || amx_t1) && v.want_argmax {
                            Some((&mut v.argmax, &mut v.toks))
                        } else {
                            None
                        },
                        v.last_row_only && t > 1 && !exact && !vfused,
                    )
                }
                None => (false, false, None, None, None, None, false),
            };
        let mut stash_gdn = stash_gdn;
        let mut stash_ple = stash_ple;
        // Slot RESERVE unit: reserve-derived so a growing decode never reallocates a
        // slot mid-run (address stability, item 2b's prerequisite). Transients scale
        // with the CHUNK length t; `reserve` = capacity by default, but a LONG-context
        // state (alloc_state_reserve) caps it at the chunk bound — a 1M-capacity state
        // must not reserve 1M-token transients (plane slots alone would be ~41 GB).
        let cap = state.reserve.max(t);

        // Entry: wide stream = `streams` copies of the embedding (modular L1012), held as
        // stream-major planes so every per-stream op is a contiguous existing kernel.
        let mut planes = prof_section(e, "entry.embed", || {
            let mut embedded = vec![0.0f32; t * hidden];
            for (row, &token) in ids.iter().enumerate() {
                let token = token as usize;
                if token >= self.vocab {
                    return Err(format!("qwen4exp_gpu: token {token} out of range").into());
                }
                embedded[row * hidden..(row + 1) * hidden]
                    .copy_from_slice(&self.embed_host[token * hidden..(token + 1) * hidden]);
            }
            let embedded_dev = ws.take_f32_h2d(e, "entry.embed", &embedded, cap * hidden)?;
            let mut planes: Vec<CudaSlice<f32>> = Vec::with_capacity(self.streams);
            for s in 0..self.streams {
                let mut plane = ws.take_f32(e, PLANE_SLOTS[s], t * hidden, cap * hidden)?;
                e.copy_into(&mut plane, 0, &embedded_dev, t * hidden)?;
                planes.push(plane);
            }
            ws.put_f32("entry.embed", embedded_dev);
            Ok(planes)
        })?;

        // Plane pointer table for the stream-batched kernels (hcmicro): refreshed every
        // step (eagerly, outside any graph) into a stable slot the captured launches
        // read at run time.
        let ptr_vals: Vec<u64> = {
            let stream = e.gpu.stream();
            planes.iter().map(|p| p.device_ptr(&stream).0).collect()
        };
        let ptrs = ws.take_u64_h2d(e, "hc.ptrs", &ptr_vals, 0)?;

        if graphs_mode {
            if state.graphs.warm {
                return self.forward_graphs_tail(e, state, planes, ptrs, base_pos);
            }
            // First graph-eligible step: run EAGER to warm/park every slot (allocations
            // inside a capture region become graph mem nodes); capture starts next step.
            state.graphs.warm = true;
        }

        for (li, (layer, lstate)) in self.layers.iter().zip(state.layers.iter_mut()).enumerate() {
            if let (Some(ple), Some(ple_state)) = (layer.ple.as_ref(), lstate.ple.as_mut()) {
                let ps = if exact {
                    stash_ple
                        .as_mut()
                        .and_then(|v| v.get_mut(li))
                        .and_then(|s| s.as_mut())
                } else {
                    None
                };
                self.ple_block(
                    e,
                    layer,
                    ple,
                    &ple.table,
                    ple_state,
                    &mut planes,
                    tokens,
                    t,
                    exact,
                    ps,
                )?;
            }
            let (mixed, inject) = prof_section(e, "hyper.read", || {
                self.gate_read(
                    e,
                    ws,
                    &ptrs,
                    &layer.attn_gate,
                    &planes,
                    t,
                    layer.eps_attn,
                    exact,
                )
            })?;
            let block_out = match &layer.mixer {
                MixerW::Qsa(qsa) => self.qsa_forward(
                    e,
                    ws,
                    layer,
                    qsa,
                    &mixed,
                    &mut lstate.mixer,
                    base_pos,
                    t,
                    0,
                    exact,
                )?,
                MixerW::Gdn(gdn) => {
                    let gs = if exact {
                        stash_gdn
                            .as_mut()
                            .and_then(|v| v.get_mut(li))
                            .and_then(|s| s.as_mut())
                    } else {
                        None
                    };
                    self.gdn_forward(e, ws, layer, gdn, &mixed, &mut lstate.mixer, t, gs)?
                }
            };
            ws.put_f32("hc.mixed", mixed);
            prof_section(e, "hyper.write", || {
                self.gate_write(e, &mut planes, &ptrs, &block_out, &inject, t)
            })?;
            ws.put_f32("mixer.out", block_out);
            put_inject(ws, inject);
            let (mixed, inject) = prof_section(e, "hyper.read", || {
                self.gate_read(
                    e,
                    ws,
                    &ptrs,
                    &layer.mlp_gate,
                    &planes,
                    t,
                    layer.eps_mlp,
                    exact,
                )
            })?;
            // Chunked long-context prefill (head-skipping forwards) rides the GROUPED
            // MoE program like verify chunks do: the per-expert prefill executor pays
            // 3 dequants + several small syncing H2Ds + GEMMs PER ROUTED EXPERT per
            // chunk (~512 x 48 per chunk = minutes/chunk measured on the smoke ladder);
            // the grouped path is 2 launches + t combines per layer on NVFP4 banks.
            // Decode-class rows (per-slot programs bit-identical to t == 1) — the
            // chunked-prefill gates are tolerance-class by design.
            // `prefill_grouped_all_on()` is the TP2 class gate's PRIME instrument (default
            // OFF = today's behavior exactly): it lets an all-rows single-card forward run
            // the GROUPED executor so a TP2 comparison isolates the expert-half split
            // instead of straddling it and the executor difference. See the flag's doc.
            // `vfused` forces grouped too: the MoE routed union is ALREADY one grouped
            // gufuse launch over every verify column on the exact arm, so letting a fused
            // verify chunk fall into the per-expert prefill executor would measure that
            // executor (minutes/chunk, above) instead of the fusion. Identical MoE program
            // on both arms is also the honest cost model — this section cannot be a vfuse
            // win, and the A/B must not pretend otherwise in either direction.
            let grouped = exact || vfused || head != HeadMode::All || prefill_grouped_all_on();
            let mlp = self.moe_forward(e, ws, &layer.moe, &mixed, t, grouped, layer.index)?;
            ws.put_f32("hc.mixed", mixed);
            prof_section(e, "hyper.write", || {
                self.gate_write(e, &mut planes, &ptrs, &mlp, &inject, t)
            })?;
            ws.put_f32("moe.out", mlp);
            put_inject(ws, inject);
            if let Some(capture) = capture.as_deref_mut() {
                capture.layer_wide.push(self.planes_to_wide(e, &planes, t)?);
            }
        }

        // Verify wide capture: the trunk's FINAL wide rows at their absolute positions,
        // ring-slotted (row % ring_rows; ring == capacity is the historical identity
        // layout) — the draft's hidden seeds (SEMANTICS.md §MTP).
        if let Some((wide_buf, ring_rows)) = stash_wide {
            let wide = self.streams * hidden;
            for (s, plane) in planes.iter().enumerate() {
                for tok in 0..t {
                    e.copy_range_into(
                        wide_buf,
                        ((base_pos + tok) % ring_rows) * wide + s * hidden,
                        plane,
                        tok * hidden,
                        hidden,
                    )?;
                }
            }
        }

        // Head skip (chunked long-context prefill): the exit mixer + lm_head read no
        // state and write none — a mid-prefill chunk stops here, state-identical.
        if head == HeadMode::Skip {
            ws.put_u64("hc.ptrs", ptrs);
            state.pos += t;
            for (s, plane) in planes.into_iter().enumerate() {
                ws.put_f32(PLANE_SLOTS[s], plane);
            }
            return Ok(Vec::new());
        }

        // Exit downmix replaces the final norm (SEMANTICS.md §Layer stack).
        let x = prof_section(e, "exit.mixer", || {
            Ok(self
                .gate_read_inner(
                    e,
                    ws,
                    &ptrs,
                    &self.exit_mixer,
                    &planes,
                    t,
                    self.exit_eps,
                    false,
                    exact,
                )?
                .0)
        })?;
        ws.put_u64("hc.ptrs", ptrs);
        if let Some(capture) = capture.as_deref_mut() {
            capture.exit_mixed = e.dtoh(&x)?;
        }
        // LastRow (chunked prefill's final chunk): lm_head on ONE row — a [t, vocab]
        // logits block at long-context chunk sizes is gigabytes.
        let head_rows = if head == HeadMode::LastRow { 1 } else { t };
        let logits = prof_section(e, "lm_head", || {
            let mut logits =
                ws.take_f32(e, "logits", head_rows * self.vocab, head_rows * self.vocab)?;
            let x_head = if head == HeadMode::LastRow {
                let mut last = ws.take_f32(e, "exit.last", hidden, hidden)?;
                e.copy_range_into(&mut last, 0, &x, (t - 1) * hidden, hidden)?;
                last
            } else {
                x
            };
            linear_trunk_into(
                e,
                &self.output,
                &self.output_b16,
                &x_head,
                &mut logits,
                head_rows,
                hidden,
                self.vocab,
            )?;
            ws.put_f32(
                if head == HeadMode::LastRow {
                    "exit.last"
                } else {
                    "hc.mixed"
                },
                x_head,
            );
            Ok(logits)
        })?;
        state.pos += t;
        if head == HeadMode::LastRow {
            let out = prof_section(e, "logits.dtoh", || {
                Ok(e.dtoh_view(&logits.slice(0..self.vocab))?)
            })?;
            ws.put_f32("logits", logits);
            for (s, plane) in planes.into_iter().enumerate() {
                ws.put_f32(PLANE_SLOTS[s], plane);
            }
            return Ok(out);
        }
        // Verify fast path: per-row device argmax + a 4t-byte dtoh instead of the
        // [t, vocab] block (the spec loop reads target rows only).
        let out = if let Some((argmax_rows, toks)) = argmax_sink {
            prof_section(e, "logits.argmax", || {
                for row in 0..t {
                    e.argmax_token_device_col(&logits, row, self.vocab, toks, row)?;
                }
                let host = e.gpu.stream().clone_dtoh(&toks.slice(0..t))?;
                argmax_rows.extend_from_slice(&host);
                Ok(Vec::new())
            })?
        } else if last_row_only {
            // mtp11: big-t (prefill) forwards under the deferred seam dtoh ONE row —
            // the spec loop reads exactly one (x0). Same bytes for that row.
            prof_section(e, "logits.dtoh", || {
                Ok(e.dtoh_view(&logits.slice((t - 1) * self.vocab..t * self.vocab))?)
            })?
        } else {
            prof_section(e, "logits.dtoh", || {
                Ok(e.dtoh_view(&logits.slice(0..t * self.vocab))?)
            })?
        };
        ws.put_f32("logits", logits);
        for (s, plane) in planes.into_iter().enumerate() {
            ws.put_f32(PLANE_SLOTS[s], plane);
        }
        Ok(out)
    }

    /// Gated-residual read gate (`gated_residual_read` twin): grouped (effective-weight)
    /// RMSNorm per stream, `w = sigmoid(up(silu(down(normed)/S)))`, `mixed = mean_s(w ⊙
    /// normed_s)`, inject scalars `2*sigmoid(block_inject(normed)/S)` per stream.
    #[allow(clippy::too_many_arguments)]
    fn gate_read(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        gate: &GateW,
        planes: &[CudaSlice<f32>],
        t: usize,
        eps: f32,
        exact: bool,
    ) -> Res<(CudaSlice<f32>, InjectOut)> {
        self.gate_read_inner(e, ws, ptrs, gate, planes, t, eps, true, exact)
    }

    #[allow(clippy::too_many_arguments)]
    fn gate_read_inner(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        gate: &GateW,
        planes: &[CudaSlice<f32>],
        t: usize,
        eps: f32,
        with_inject: bool,
        // Verify-chunk exactness (mtp-spec lane): engage the DIET kernels at t > 1 so
        // every verify row runs the DECODE gate program verbatim per token (the diet
        // kernels' token dim is the t == 1 program at a plane offset — bit-identical
        // rows). Plain prefill keeps the fused chain (banked-goldens numerics stay).
        exact: bool,
    ) -> Res<(CudaSlice<f32>, InjectOut)> {
        if !hc_fused_gate_on() {
            return self.gate_read_legacy(e, ws, gate, planes, t, eps, with_inject);
        }
        let hidden = self.hidden;
        let streams = self.streams;
        let rank = gate_rank(gate, hidden, streams)?;
        let micro_norm = micro_norm_on();
        let micro_inj = micro_inj_on();
        // Hyper-gate diet (round 4): the whole read gate in THREE launches. Requires the
        // bf16 twins + the Slab inject posture (micro_inj — take_inject's form contract)
        // + real geometry; anything else falls back to the fused chain below.
        if hc_diet_on()
            && (t == 1 || exact)
            && trunk_bf16_on()
            && micro_inj
            && hidden % 8 == 0
            && rank % 8 == 0
            && gate.down_b16.is_some()
            && gate.up_b16.is_some()
            && (!with_inject || gate.inject_b16.is_some())
        {
            let mut parts = ws.take_f32(e, "hc.parts", t * streams * rank, 0)?;
            let mut injp = ws.take_f32(e, "hc.injp", t * streams * streams, 0)?;
            let mut inv = ws.take_f32(e, "hc.inv", t * streams, 0)?;
            let winj = if with_inject {
                gate.inject_b16.as_ref()
            } else {
                None
            };
            // Weight-shared MT stages (set_verify_mt) at verify chunks: bit-identical
            // per token to the token-grid stages (kernel docs + gate oracle), weight
            // reads 1x instead of t x.
            let mt = t > 1 && verify_mt_on() && (2..=12).contains(&t);
            if mt {
                launch_hc_diet_stage0_mt(e, ptrs, &mut inv, hidden, streams, t, eps)?;
                launch_hc_diet_stage1_mt(
                    e,
                    ptrs,
                    &gate.norm_stack,
                    &inv,
                    gate.down_b16.as_ref().expect("guarded above"),
                    winj,
                    &mut parts,
                    &mut injp,
                    hidden,
                    rank,
                    streams,
                    t,
                )?;
            } else {
                launch_hc_diet_stage1(
                    e,
                    ptrs,
                    &gate.norm_stack,
                    gate.down_b16.as_ref().expect("guarded above"),
                    winj,
                    &mut parts,
                    &mut injp,
                    &mut inv,
                    hidden,
                    rank,
                    streams,
                    t,
                    eps,
                )?;
            }
            let mut low_act = ws.take_f32(e, "hc.low_act", t * rank, 0)?;
            let mut all = ws.take_f32(e, "hc.inj_all", streams * t, 0)?;
            launch_hc_diet_stage2(
                e,
                &parts,
                &injp,
                &mut low_act,
                &mut all,
                rank,
                streams,
                t,
                with_inject,
            )?;
            let mut mixed = ws.take_f32(e, "hc.mixed", t * hidden, 0)?;
            if mt && (t * rank + 8 * streams * t) * 4 <= 96 * 1024 {
                launch_hc_diet_stage3_mt(
                    e,
                    ptrs,
                    &gate.norm_stack,
                    &inv,
                    gate.up_b16.as_ref().expect("guarded above"),
                    &low_act,
                    &mut mixed,
                    hidden,
                    rank,
                    streams,
                    t,
                )?;
            } else {
                launch_hc_diet_stage3(
                    e,
                    ptrs,
                    &gate.norm_stack,
                    &inv,
                    gate.up_b16.as_ref().expect("guarded above"),
                    &low_act,
                    &mut mixed,
                    hidden,
                    rank,
                    streams,
                    t,
                )?;
            }
            ws.put_f32("hc.parts", parts);
            ws.put_f32("hc.injp", injp);
            ws.put_f32("hc.inv", inv);
            ws.put_f32("hc.low_act", low_act);
            let inject_out = if with_inject {
                InjectOut::Slab(all)
            } else {
                ws.put_f32("hc.inj_all", all);
                InjectOut::Rows(Vec::new())
            };
            return Ok((mixed, inject_out));
        }

        // Buffers are STREAM-MAJOR and CONTIGUOUS ([streams, t, width]) so the three fused
        // gate kernels (perf lane attack (c)) each read every stream in one launch; the
        // 12 GEMVs stay cuBLASLt. Launches per read gate: 4 norms + 4 down + 1 reduce +
        // 4 up + 1 epilogue + 1 inject = 15, vs ~71 before (PROFILE-0: 27.7% of the token
        // across 96 calls, nearly all issue latency).
        let mut normed = ws.take_f32(e, "hc.normed", streams * t * hidden, 0)?;
        if micro_norm {
            // One launch for all streams over the plane pointer table (hcmicro).
            launch_hc_norm_planes(
                e,
                ptrs,
                &gate.norm_stack,
                &mut normed,
                hidden,
                t,
                streams,
                eps,
            )?;
        } else {
            for s in 0..streams {
                let mut dst = normed.slice_mut(s * t * hidden..(s + 1) * t * hidden);
                launch_rms_norm_into_view(e, &planes[s], &gate.norm[s], &mut dst, hidden, t, eps)?;
            }
        }
        // low_act = silu(mean_s down_s @ normed_s). bf16 trunk residency runs the
        // projection as ONE batched launch over the stream-major slab (stacked twin,
        // same output layout as the per-stream cuBLASLt chain — the A/B/fallback arm).
        let trunk_b16 = trunk_bf16_on();
        let mut parts = ws.take_f32(e, "hc.parts", streams * t * rank, 0)?;
        if let (true, Some(w)) = (trunk_b16, gate.down_b16.as_ref()) {
            launch_qmatvec_bf16w(
                e,
                w,
                &normed,
                &mut parts,
                hidden,
                rank,
                t,
                streams,
                rank * hidden,
                t * hidden,
                hidden,
                t * rank,
            )?;
        } else {
            if gate.down[0].len() < rank * hidden {
                return Err(
                    "qwen4exp_gpu: gate down f32 dropped (trunk_f32_diet) — keep the \
                            trunk-bf16 seam ON"
                        .into(),
                );
            }
            for s in 0..streams {
                let x = normed.slice(s * t * hidden..(s + 1) * t * hidden);
                let w = gate.down[s].slice(0..rank * hidden);
                let mut out = parts.slice_mut(s * t * rank..(s + 1) * t * rank);
                e.linear_device_into(&x, &w, &mut out, t, hidden, rank)?;
            }
        }
        let mut low_act = ws.take_f32(e, "hc.low_act", t * rank, 0)?;
        launch_hc_lowrank_reduce(e, &parts, &mut low_act, streams, t, rank)?;
        ws.put_f32("hc.parts", parts);

        // mixed = mean_s sigmoid(up_s @ low_act) ⊙ normed_s (batched twin: x_bstride 0
        // shares the one low_act plane across streams).
        let mut gates = ws.take_f32(e, "hc.gates", streams * t * hidden, 0)?;
        if let (true, Some(w)) = (trunk_b16, gate.up_b16.as_ref()) {
            launch_qmatvec_bf16w(
                e,
                w,
                &low_act,
                &mut gates,
                rank,
                hidden,
                t,
                streams,
                hidden * rank,
                0,
                rank,
                t * hidden,
            )?;
        } else {
            if gate.up[0].len() < hidden * rank {
                return Err(
                    "qwen4exp_gpu: gate up f32 dropped (trunk_f32_diet) — keep the \
                            trunk-bf16 seam ON"
                        .into(),
                );
            }
            for s in 0..streams {
                let x = low_act.slice(0..t * rank);
                let w = gate.up[s].slice(0..hidden * rank);
                let mut out = gates.slice_mut(s * t * hidden..(s + 1) * t * hidden);
                e.linear_device_into(&x, &w, &mut out, t, rank, hidden)?;
            }
        }
        let mut mixed = ws.take_f32(e, "hc.mixed", t * hidden, 0)?;
        launch_hc_mix_epilogue(e, &gates, &normed, &mut mixed, streams, t, hidden)?;
        ws.put_f32("hc.gates", gates);
        ws.put_f32("hc.low_act", low_act);

        let mut inject_out = InjectOut::Rows(Vec::new());
        if with_inject {
            let inject = gate
                .inject
                .as_ref()
                .ok_or("qwen4exp_gpu: read gate missing inject weights")?;
            // trunk_f32_diet: the f32 inject may be a dropped stub — every non-b16
            // consumer below must refuse it rather than read garbage.
            let inject_dropped = inject.len() < streams * streams * hidden;
            let inject_guard = || -> Res<()> {
                if inject_dropped {
                    return Err("qwen4exp_gpu: inject f32 dropped (trunk_f32_diet) — keep \
                                the trunk-bf16 seam ON"
                        .into());
                }
                Ok(())
            };
            let mut all = ws.take_f32(e, "hc.inj_all", streams * t, 0)?;
            if micro_inj {
                // Two-stage inject (hcmicro): chunked partials fill the card, the reduce
                // applies the sigmoid; the slab goes straight to `gate_write`.
                const CHUNKS: usize = 16;
                let mut partials = ws.take_f32(e, "hc.inj_part", streams * t * CHUNKS, 0)?;
                let w_b16 = if trunk_b16 {
                    gate.inject_b16.as_ref()
                } else {
                    None
                };
                if w_b16.is_none() {
                    inject_guard()?;
                }
                launch_hc_inject_two_stage(
                    e,
                    &normed,
                    inject,
                    w_b16,
                    &mut partials,
                    &mut all,
                    streams,
                    t,
                    hidden,
                    CHUNKS,
                )?;
                ws.put_f32("hc.inj_part", partials);
                inject_out = InjectOut::Slab(all);
            } else {
                // [streams, t] scalars in one launch; `gate_write` consumes one row per
                // stream.
                if let (true, Some(w)) = (trunk_b16, gate.inject_b16.as_ref()) {
                    launch_hc_inject_gates_b16(e, &normed, w, &mut all, streams, t, hidden)?;
                } else {
                    inject_guard()?;
                    launch_hc_inject_gates(e, &normed, inject, &mut all, streams, t, hidden)?;
                }
                let mut rows = Vec::with_capacity(streams);
                for s in 0..streams {
                    let mut row = ws.take_f32(e, INJECT_SLOTS[s], t, 0)?;
                    e.copy_range_into(&mut row, 0, &all, s * t, t)?;
                    rows.push(row);
                }
                ws.put_f32("hc.inj_all", all);
                inject_out = InjectOut::Rows(rows);
            }
        }
        ws.put_f32("hc.normed", normed);
        Ok((mixed, inject_out))
    }

    /// Unfused read gate — the literal `gated_residual_read` composition from existing
    /// engine ops, kept as the A/B twin of the fused arm (`set_hc_fused_gate(false)`) and
    /// as the readable statement of the program. ~71 launches per call at hc_count 4.
    /// Deliberately NOT workspace-pooled: it is the hc-off measurement twin.
    #[allow(clippy::too_many_arguments)]
    fn gate_read_legacy(
        &self,
        e: &Engine,
        _ws: &mut StepPool,
        gate: &GateW,
        planes: &[CudaSlice<f32>],
        t: usize,
        eps: f32,
        with_inject: bool,
    ) -> Res<(CudaSlice<f32>, InjectOut)> {
        let hidden = self.hidden;
        let streams = self.streams;
        let rank = gate_rank(gate, hidden, streams)?;
        if gate.down[0].len() < rank * hidden {
            return Err(
                "qwen4exp_gpu: gate f32 originals dropped (trunk_f32_diet) — the \
                        legacy gate path needs them (keep hc seams ON)"
                    .into(),
            );
        }
        let inv_streams = 1.0 / streams as f32; // pow2 (hc_count 4 / tiny 2) — exact

        let mut normed = Vec::with_capacity(streams);
        for s in 0..streams {
            let mut dst = e.uninit(t * hidden)?;
            e.rms_norm(&planes[s], &gate.norm[s], &mut dst, hidden, t, eps)?;
            normed.push(dst);
        }
        // low = silu(sum_s down_s @ normed_s / S)
        let mut low = e.linear(&normed[0], &gate.down[0], t, hidden, rank)?;
        for s in 1..streams {
            let part = e.linear(&normed[s], &gate.down[s], t, hidden, rank)?;
            let mut view = low.slice_mut(0..t * rank);
            e.axpy_into(&part, 1.0, &mut view, t * rank)?;
        }
        e.scale_inplace(&mut low, inv_streams, t * rank)?;
        let ones = e.htod(&vec![1.0f32; t * rank.max(1)])?;
        let mut low_act = e.uninit(t * rank)?;
        e.silu_mul(&low, &ones, &mut low_act, t * rank)?;

        // mixed = mean_s sigmoid(up_s @ low) ⊙ normed_s
        let mut mixed = e.zeros(t * hidden)?;
        let mut gate_buf = e.uninit(t * hidden)?;
        let mut prod = e.uninit(t * hidden)?;
        for s in 0..streams {
            let g = e.linear(&low_act, &gate.up[s], t, rank, hidden)?;
            e.sigmoid(&g, &mut gate_buf, t * hidden)?;
            e.mul(&gate_buf, &normed[s], &mut prod, t * hidden)?;
            let mut view = mixed.slice_mut(0..t * hidden);
            e.axpy_into(&prod, 1.0, &mut view, t * hidden)?;
        }
        e.scale_inplace(&mut mixed, inv_streams, t * hidden)?;

        let mut inject_out = Vec::new();
        if with_inject {
            let inject = gate
                .inject
                .as_ref()
                .ok_or("qwen4exp_gpu: read gate missing inject weights")?;
            let wide = streams * hidden;
            for s in 0..streams {
                // Per-(s, s2) [hidden] weight windows of block_inject_weight row s.
                let mut acc = {
                    let w = inject.slice(s * wide..s * wide + hidden);
                    let x = normed[0].slice(0..t * hidden);
                    let mut out = e.uninit(t)?;
                    e.linear_device_into(&x, &w, &mut out, t, hidden, 1)?;
                    out
                };
                for s2 in 1..streams {
                    let w = inject.slice(s * wide + s2 * hidden..s * wide + (s2 + 1) * hidden);
                    let x = normed[s2].slice(0..t * hidden);
                    let mut part = e.uninit(t)?;
                    e.linear_device_into(&x, &w, &mut part, t, hidden, 1)?;
                    let mut view = acc.slice_mut(0..t);
                    e.axpy_into(&part, 1.0, &mut view, t)?;
                }
                e.scale_inplace(&mut acc, inv_streams, t)?;
                let mut sg = e.uninit(t)?;
                e.sigmoid(&acc, &mut sg, t)?;
                e.scale_inplace(&mut sg, 2.0, t)?;
                inject_out.push(sg);
            }
        }
        Ok((mixed, InjectOut::Rows(inject_out)))
    }

    /// Write half (`gated_residual_write` twin): plane_s += block_out ⊗ inject_s.
    /// Rows = per-stream add_scaled_rows (item-1-era plumbing); Slab = one launch over
    /// the plane pointer table (hcmicro).
    fn gate_write(
        &self,
        e: &Engine,
        planes: &mut [CudaSlice<f32>],
        ptrs: &CudaSlice<u64>,
        block_out: &CudaSlice<f32>,
        inject: &InjectOut,
        t: usize,
    ) -> Res<()> {
        match inject {
            InjectOut::Rows(rows) => {
                for (plane, inj) in planes.iter_mut().zip(rows) {
                    e.add_scaled_rows(block_out, inj, plane, self.hidden, t)?;
                }
                Ok(())
            }
            InjectOut::Slab(slab) => {
                launch_hc_write_planes(e, ptrs, block_out, slab, self.hidden, t, self.streams)
            }
        }
    }

    /// One decode layer's INTERIOR at t == 1 (graph driver, item 2b): PLE (when
    /// present) → attn read gate → mixer → write → mlp read gate, ending with the mlp
    /// `mixed`/inject scalars PARKED in their slots for the MoE tail. The exact
    /// semantics of the eager `forward` loop body up to `moe_forward`; device-only for
    /// GDN layers without PLE, which is what makes those capturable.
    #[allow(clippy::too_many_arguments)]
    fn layer_interior(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        layer: &LayerW,
        lstate: &mut LayerState,
        planes: &mut [CudaSlice<f32>],
        tokens: &[u32],
        base_pos: usize,
    ) -> Res<()> {
        if let (Some(ple), Some(ple_state)) = (layer.ple.as_ref(), lstate.ple.as_mut()) {
            self.ple_block(
                e, layer, ple, &ple.table, ple_state, planes, tokens, 1, false, None,
            )?;
        }
        let (mixed, inject) = self.gate_read(
            e,
            ws,
            ptrs,
            &layer.attn_gate,
            planes,
            1,
            layer.eps_attn,
            false,
        )?;
        let block_out = match &layer.mixer {
            MixerW::Qsa(qsa) => self.qsa_forward(
                e,
                ws,
                layer,
                qsa,
                &mixed,
                &mut lstate.mixer,
                base_pos,
                1,
                0,
                false,
            )?,
            MixerW::Gdn(gdn) => {
                self.gdn_forward(e, ws, layer, gdn, &mixed, &mut lstate.mixer, 1, None)?
            }
        };
        ws.put_f32("hc.mixed", mixed);
        self.gate_write(e, planes, ptrs, &block_out, &inject, 1)?;
        ws.put_f32("mixer.out", block_out);
        put_inject(ws, inject);
        let (mixed, inject) = self.gate_read(
            e,
            ws,
            ptrs,
            &layer.mlp_gate,
            planes,
            1,
            layer.eps_mlp,
            false,
        )?;
        ws.put_f32("hc.mixed", mixed);
        put_inject(ws, inject);
        Ok(())
    }

    /// Per-step MoE routing (graph driver): router GEMV over the parked mlp `mixed`,
    /// dtoh (the per-layer host boundary — routing is a HOST twin by lane doctrine, so
    /// a whole-step graph is structurally impossible; this is the sync the segment
    /// graphs meet at), reference top-k, then H2D of the selection into the slot
    /// addresses the captured MoE-tail graph baked.
    fn moe_route_slots(&self, e: &Engine, ws: &mut StepPool, moe: &MoeW, layer: u32) -> Res<()> {
        let hidden = self.hidden;
        let experts = moe.plan.expert_count as usize;
        let selected = moe.plan.experts_per_token as usize;
        let mixed = ws.take_f32(e, "hc.mixed", hidden, 0)?;
        let mut router_out = ws.take_f32(e, "moe.router", experts, 0)?;
        let none: Option<CudaSlice<u8>> = None;
        let rb = if router_bf16_on() {
            &moe.router_b16
        } else {
            &none
        };
        linear_trunk_into(
            e,
            &moe.router,
            rb,
            &mixed,
            &mut router_out,
            1,
            hidden,
            experts,
        )?;
        // Device router (devtwin lane): the route stays on device — no dtoh, no host
        // top-k, no selection h2d. Writes land in the SAME parked slots the captured
        // MoE-tail graph baked (take-without-upload + put preserves the address).
        if router_dev_on() && route_dev_geometry(experts, selected) {
            let mut sel = ws.take_i32_slot(e, "moe.sel", selected, 0)?;
            let mut w = ws.take_f32(e, "moe.w", selected, 0)?;
            route_topk_device(
                e,
                &router_out,
                &mut sel,
                &mut w,
                None,
                experts,
                selected,
                1,
                layer,
            )?;
            // DIAGNOSTIC ONLY (`MEMRA_Q4E_ROUTE_SYNC=1`, never a serving arm): restore the
            // host arm's per-layer SYNC structure while keeping the device route, to
            // separate "the kernel costs" from "the missing sync costs" in the
            // graphs-ON regression (devtwin: graphs OFF the seam wins 1.083x, graphs ON
            // it loses — PROFILE-9 §3).
            if route_sync_diag() {
                e.gpu.stream().synchronize()?;
            }
            ws.put_i32("moe.sel", sel);
            ws.put_f32("moe.w", w);
            ws.put_f32("moe.router", router_out);
            ws.put_f32("hc.mixed", mixed);
            return Ok(());
        }
        let logits = e.dtoh_view(&router_out.slice(0..experts))?;
        ws.put_f32("moe.router", router_out);
        ws.put_f32("hc.mixed", mixed);
        let route = host_route_softmax_topk(&logits, selected);
        let sel_host: Vec<i32> = route.iter().map(|&(x, _)| x as i32).collect();
        let w_host: Vec<f32> = route.iter().map(|&(_, w)| w).collect();
        ws.write_i32(e, "moe.sel", &sel_host)?;
        ws.write_f32(e, "moe.w", &w_host)?;
        Ok(())
    }

    /// The grouped-MoE tail at t == 1 over PARKED slots (graph driver): sel matvecs →
    /// shared expert → mlp gate_write. Same kernels/order as the `moe_forward` grouped
    /// block; the selection indices/weights arrive via `moe_route_slots` into the baked
    /// slot addresses.
    fn moe_grouped_tail_slots(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        moe: &MoeW,
        planes: &mut [CudaSlice<f32>],
    ) -> Res<()> {
        let hidden = self.hidden;
        let ff = moe.plan.expert_intermediate_size as usize;
        let n_sel = moe.plan.experts_per_token as usize;
        let (
            BankHalf::Nvfp4 {
                codes: gc,
                scales: gs,
                macros_dev: gm,
                ..
            },
            BankHalf::Nvfp4 {
                codes: uc,
                scales: us,
                macros_dev: um,
                ..
            },
            BankHalf::Nvfp4 {
                codes: dc,
                scales: ds,
                macros_dev: dm,
                ..
            },
        ) = (&moe.bank.gate, &moe.bank.up, &moe.bank.down)
        else {
            return Err("qwen4exp_gpu: grouped tail on a non-NVFP4 bank".into());
        };
        let mixed = ws.take_f32(e, "hc.mixed", hidden, 0)?;
        let sel = ws
            .i32s
            .remove("moe.sel")
            .ok_or("step workspace: moe.sel is not parked")?;
        let w_dev = ws.take_f32(e, "moe.w", n_sel, 0)?;
        let mut act = ws.take_f32(e, "moe.act", n_sel * ff, 0)?;
        // Fused gate+up+silu (round 4): the graph bakes whichever arm is live at
        // capture (fresh state per A/B arm); bit-identical to the chain.
        if sel_gufuse_on() && hidden % 32 == 0 && ff % 4 == 0 {
            launch_nvfp4_sel_gu_silu(
                e,
                (gc, gs, gm),
                (uc, us, um),
                Some(&sel),
                0,
                n_sel,
                &mixed,
                &mut act,
                hidden,
                ff,
                None,
            )?;
        } else {
            let mut yg = ws.take_f32(e, "moe.yg", n_sel * ff, 0)?;
            let mut yu = ws.take_f32(e, "moe.yu", n_sel * ff, 0)?;
            launch_nvfp4_sel_matvec(e, gc, gs, gm, &sel, &mixed, &mut yg, n_sel, hidden, ff, 0)?;
            launch_nvfp4_sel_matvec(e, uc, us, um, &sel, &mixed, &mut yu, n_sel, hidden, ff, 0)?;
            e.silu_mul(&yg, &yu, &mut act, n_sel * ff)?;
            ws.put_f32("moe.yg", yg);
            ws.put_f32("moe.yu", yu);
        }
        let mut partial = ws.take_f32(e, "moe.partial", n_sel * hidden, 0)?;
        launch_nvfp4_sel_matvec(
            e,
            dc,
            ds,
            dm,
            &sel,
            &act,
            &mut partial,
            n_sel,
            ff,
            hidden,
            ff,
        )?;
        let mut out = ws.take_f32(e, "moe.out", hidden, 0)?;
        e.axpy_rows_seq_into(&partial, &w_dev, &mut out, hidden, n_sel)?;
        ws.put_i32("moe.sel", sel);
        ws.put_f32("moe.w", w_dev);
        ws.put_f32("moe.act", act);
        ws.put_f32("moe.partial", partial);
        let out = self.moe_shared_tail(e, ws, moe, &mixed, out, 1)?;
        ws.put_f32("hc.mixed", mixed);
        let inject = take_inject(e, ws, self.streams, 1)?;
        self.gate_write(e, planes, ptrs, &out, &inject, 1)?;
        ws.put_f32("moe.out", out);
        put_inject(ws, inject);
        Ok(())
    }

    /// Graph-mode decode tail (item 2b): per layer, replay (or lazily capture) the
    /// interior graph, run the host routing boundary, replay the MoE-tail graph; then
    /// the exit graph and one logits dtoh. Falls back to the eager helpers per layer
    /// where a graph is structurally unavailable (QSA/PLE interiors — the indexer host
    /// twin and PLE host hashing live there; non-NVFP4 banks for the tail).
    fn forward_graphs_tail(
        &self,
        e: &Engine,
        state: &mut Qwen4ExpState,
        mut planes: Vec<CudaSlice<f32>>,
        ptrs: CudaSlice<u64>,
        base_pos: usize,
    ) -> Res<Vec<f32>> {
        let mut graphs = std::mem::take(&mut state.graphs);
        if graphs.a.len() != self.layers.len() {
            graphs.a = (0..self.layers.len()).map(|_| None).collect();
            graphs.b = (0..self.layers.len()).map(|_| None).collect();
        }
        let ws = &mut state.ws;
        let tokens = &state.tokens;
        for (li, (layer, lstate)) in self.layers.iter().zip(state.layers.iter_mut()).enumerate() {
            let a_ok = matches!(layer.mixer, MixerW::Gdn(_)) && layer.ple.is_none();
            if a_ok {
                if graphs.a[li].is_none() {
                    graphs.a[li] = Some(e.capture_graph_retained_nowarm(|eng| {
                        self.layer_interior(
                            eng,
                            ws,
                            &ptrs,
                            layer,
                            lstate,
                            &mut planes,
                            tokens,
                            base_pos,
                        )
                    })?);
                }
                graphs.a[li].as_ref().unwrap().0.launch()?;
            } else {
                self.layer_interior(e, ws, &ptrs, layer, lstate, &mut planes, tokens, base_pos)?;
            }
            let b_ok = moe_sel_path_on()
                && matches!(
                    (
                        &layer.moe.bank.gate,
                        &layer.moe.bank.up,
                        &layer.moe.bank.down
                    ),
                    (
                        BankHalf::Nvfp4 { .. },
                        BankHalf::Nvfp4 { .. },
                        BankHalf::Nvfp4 { .. }
                    )
                );
            if b_ok {
                self.moe_route_slots(e, ws, &layer.moe, layer.index)?;
                if graphs.b[li].is_none() {
                    graphs.b[li] = Some(e.capture_graph_retained_nowarm(|eng| {
                        self.moe_grouped_tail_slots(eng, ws, &ptrs, &layer.moe, &mut planes)
                    })?);
                }
                graphs.b[li].as_ref().unwrap().0.launch()?;
            } else {
                // Eager MoE (per-expert path routes internally) + mlp write.
                let mixed = ws.take_f32(e, "hc.mixed", self.hidden, 0)?;
                let mlp = self.moe_forward(e, ws, &layer.moe, &mixed, 1, false, layer.index)?;
                ws.put_f32("hc.mixed", mixed);
                let inject = take_inject(e, ws, self.streams, 1)?;
                self.gate_write(e, &mut planes, &ptrs, &mlp, &inject, 1)?;
                ws.put_f32("moe.out", mlp);
                put_inject(ws, inject);
            }
        }
        if graphs.exit.is_none() {
            graphs.exit = Some(e.capture_graph_retained_nowarm(|eng| {
                let x = self
                    .gate_read_inner(
                        eng,
                        ws,
                        &ptrs,
                        &self.exit_mixer,
                        &planes,
                        1,
                        self.exit_eps,
                        false,
                        false,
                    )?
                    .0;
                let mut logits = ws.take_f32(eng, "logits", self.vocab, 0)?;
                linear_trunk_into(
                    eng,
                    &self.output,
                    &self.output_b16,
                    &x,
                    &mut logits,
                    1,
                    self.hidden,
                    self.vocab,
                )?;
                ws.put_f32("hc.mixed", x);
                ws.put_f32("logits", logits);
                Ok(())
            })?);
        }
        graphs.exit.as_ref().unwrap().0.launch()?;
        let out = {
            let logits = ws.peek_f32("logits")?;
            e.dtoh_view(&logits.slice(0..self.vocab))?
        };
        for (s, plane) in planes.into_iter().enumerate() {
            ws.put_f32(PLANE_SLOTS[s], plane);
        }
        ws.put_u64("hc.ptrs", ptrs);
        state.pos += 1;
        state.graphs = graphs;
        Ok(out)
    }

    /// QSA layer: fused [q|gate] projection, q/k RMSNorm, partial rope, KV append, the
    /// host indexer-selection twin, dense masked attention, sigmoid fused output gate.
    ///
    /// Indexer update + selection for one chunk (factored from `qsa_forward` so the
    /// TP2 route shares it verbatim): idx projection, the idxcache device raw-key
    /// cache maintenance, host/pooled cache updates, the device-scorer selection, and
    /// the idxq audit twin. Returns per-row selections (`RowSel`).
    #[allow(clippy::too_many_arguments)]
    fn qsa_update_select(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        qsa: &QsaW,
        eps: f32,
        mixed: &CudaSlice<f32>,
        raw_keys: &mut IdxRawCache,
        pooled_keys: &mut Vec<f32>,
        pooled_dev: &mut Option<CudaSlice<f32>>,
        pooled_dev_rows: &mut usize,
        raw_dev: &mut Option<IdxRawDev>,
        raw_dev_rows: &mut usize,
        mut idx_audit: Option<&mut Box<IdxAudit>>,
        base_pos: usize,
        t: usize,
        pos_off: usize,
        exact: bool,
    ) -> Res<Vec<RowSel>> {
        let hidden = self.hidden;
        let base = qsa.attn.rope.base;
        let t_kv = base_pos + t;
        // Indexer selection: host twin of micro_block_selection_mask over the raw-key cache.
        let overlay = &qsa.overlay;
        let idx_dim = overlay.head_dim as usize;
        let qk_width = (overlay.query_heads as usize + overlay.kv_heads as usize) * idx_dim;
        if overlay.kv_heads != 1 {
            return Err("qwen4exp_gpu: indexer with more than one key head".into());
        }
        let idx_proj = prof_section(e, "qsa.idx_proj", || {
            let mut idx_proj = ws.take_f32(e, "qsa.idxp", t * qk_width, 0)?;
            if exact && t > 1 {
                let wv = qsa.idx_proj.slice(0..qsa.idx_proj.len());
                for tok in 0..t {
                    let xv = mixed.slice(tok * hidden..(tok + 1) * hidden);
                    let mut yv = idx_proj.slice_mut(tok * qk_width..(tok + 1) * qk_width);
                    e.linear_device_into(&xv, &wv, &mut yv, 1, hidden, qk_width)?;
                }
            } else {
                e.linear_device_into(mixed, &qsa.idx_proj, &mut idx_proj, t, hidden, qk_width)?;
            }
            Ok(idx_proj)
        })?;
        // Device raw-key cache (devtwin stage 3, `idxcache`): row r of `raw_dev` is
        // absolute cache row r. Below the selection horizon ((base_pos + t)/block <=
        // budget — the indexer_select_rows fast path, decided from positions alone)
        // the selection needs NO device data, so the k-part rows append d2d and the
        // idx_proj dtoh dies; the host cache lags and materializes LAZILY at the first
        // scored chunk — the same bytes dtoh'd later, bit-identical by construction.
        // Mid-run seam flips on a live state pay their debt loudly here: OFF->ON
        // backfills the device from the host (h2d, exact bytes); any host lag is paid
        // BEFORE this chunk lands whenever the fast path does not take it.
        let dev_cache = idx_cache_on();
        let block_size = overlay.block_size as usize;
        let all_full = (base_pos + t) / block_size <= overlay.budget_blocks as usize;
        let host_rows = raw_keys.rows(idx_dim);
        if *raw_dev_rows > host_rows && !(dev_cache && all_full) {
            // Lazy host materialization (or an ON->OFF flip's debt): dtoh the delta
            // VERBATIM — quantized formats materialize their own bytes, no re-quant,
            // so the seam's bit-identity contract is preserved per format.
            idx_materialize_host(e, raw_keys, raw_dev, *raw_dev_rows, idx_dim)?;
        }
        if dev_cache {
            let host_rows = raw_keys.rows(idx_dim);
            let base_rows = (*raw_dev_rows).max(host_rows);
            let cap_rows = (base_rows + t).next_power_of_two().max(64);
            let q_off = overlay.query_heads as usize * idx_dim;
            match &mut *raw_keys {
                IdxRawCache::F32(h) => {
                    let want = (base_rows + t) * idx_dim;
                    let grow = match raw_dev.as_ref() {
                        Some(IdxRawDev::F32(m)) => m.len() < want,
                        Some(_) => return Err("idxcache: device format lag on f32".into()),
                        None => true,
                    };
                    if grow {
                        let mut fresh = e.uninit(cap_rows * idx_dim)?;
                        if let (Some(IdxRawDev::F32(old)), rows) = (raw_dev.as_ref(), *raw_dev_rows)
                        {
                            if rows > 0 {
                                e.copy_range_into(&mut fresh, 0, old, 0, rows * idx_dim)?;
                            }
                        }
                        *raw_dev = Some(IdxRawDev::F32(fresh));
                    }
                    let Some(IdxRawDev::F32(m)) = raw_dev.as_mut() else {
                        unreachable!("allocated above");
                    };
                    if host_rows > *raw_dev_rows {
                        // OFF->ON flip on a live state: backfill the device from host.
                        let mut view = m.slice_mut(*raw_dev_rows * idx_dim..host_rows * idx_dim);
                        e.gpu
                            .stream()
                            .memcpy_htod(&h[*raw_dev_rows * idx_dim..], &mut view)?;
                        *raw_dev_rows = host_rows;
                    }
                    launch_copy_rows_col(
                        e,
                        &idx_proj,
                        m,
                        t,
                        idx_dim,
                        qk_width,
                        q_off,
                        *raw_dev_rows,
                    )?;
                }
                IdxRawCache::Q8(h) => {
                    let rb = q8_row_bytes(idx_dim);
                    let want = (base_rows + t) * rb;
                    let grow = match raw_dev.as_ref() {
                        Some(IdxRawDev::Q8(m)) => m.len() < want,
                        Some(_) => return Err("idxcache: device format lag on q8".into()),
                        None => true,
                    };
                    if grow {
                        let mut fresh = e.alloc_u8_uninit(cap_rows * rb)?;
                        if let (Some(IdxRawDev::Q8(old)), rows) = (raw_dev.as_ref(), *raw_dev_rows)
                        {
                            if rows > 0 {
                                let mut dst = fresh.slice_mut(0..rows * rb);
                                e.gpu
                                    .stream()
                                    .memcpy_dtod(&old.slice(0..rows * rb), &mut dst)?;
                            }
                        }
                        *raw_dev = Some(IdxRawDev::Q8(fresh));
                    }
                    let Some(IdxRawDev::Q8(m)) = raw_dev.as_mut() else {
                        unreachable!("allocated above");
                    };
                    if host_rows > *raw_dev_rows {
                        let mut view = m.slice_mut(*raw_dev_rows * rb..host_rows * rb);
                        e.gpu
                            .stream()
                            .memcpy_htod(&h[*raw_dev_rows * rb..host_rows * rb], &mut view)?;
                        *raw_dev_rows = host_rows;
                    }
                    launch_q4e_idx_append_q8(
                        e,
                        &idx_proj,
                        m,
                        t,
                        idx_dim,
                        qk_width,
                        q_off,
                        *raw_dev_rows,
                    )?;
                }
                IdxRawCache::Bf16(h) => {
                    let want = (base_rows + t) * idx_dim;
                    let grow = match raw_dev.as_ref() {
                        Some(IdxRawDev::Bf16(m)) => m.len() < want,
                        Some(_) => return Err("idxcache: device format lag on bf16".into()),
                        None => true,
                    };
                    if grow {
                        let mut fresh = unsafe { e.gpu.stream().alloc::<u16>(cap_rows * idx_dim)? };
                        if let (Some(IdxRawDev::Bf16(old)), rows) =
                            (raw_dev.as_ref(), *raw_dev_rows)
                        {
                            if rows > 0 {
                                let mut dst = fresh.slice_mut(0..rows * idx_dim);
                                e.gpu
                                    .stream()
                                    .memcpy_dtod(&old.slice(0..rows * idx_dim), &mut dst)?;
                            }
                        }
                        *raw_dev = Some(IdxRawDev::Bf16(fresh));
                    }
                    let Some(IdxRawDev::Bf16(m)) = raw_dev.as_mut() else {
                        unreachable!("allocated above");
                    };
                    if host_rows > *raw_dev_rows {
                        let mut view = m.slice_mut(*raw_dev_rows * idx_dim..host_rows * idx_dim);
                        e.gpu.stream().memcpy_htod(
                            &h[*raw_dev_rows * idx_dim..host_rows * idx_dim],
                            &mut view,
                        )?;
                        *raw_dev_rows = host_rows;
                    }
                    launch_q4e_idx_append_bf16(
                        e,
                        &idx_proj,
                        m,
                        t,
                        idx_dim,
                        qk_width,
                        q_off,
                        *raw_dev_rows,
                    )?;
                }
            }
            *raw_dev_rows += t;
        }
        // idxq selection-identity audit (instrument): the f32 twin cache is fed on
        // EVERY chunk — this re-adds the idx_proj dtoh the idxcache seam removed, and
        // is never a perf arm. Fed BEFORE selection so the twin includes this chunk.
        if let Some(audit) = idx_audit.as_deref_mut() {
            let q_off = overlay.query_heads as usize * idx_dim;
            let rows_f = e.dtoh_view(&idx_proj.slice(0..t * qk_width))?;
            let IdxRawCache::F32(twin) = &mut audit.raw_f32 else {
                return Err("idxq audit: twin cache is not f32".into());
            };
            for row in 0..t {
                twin.extend_from_slice(&rows_f[row * qk_width + q_off..(row + 1) * qk_width]);
            }
        }
        let sels: Vec<RowSel> = if dev_cache && all_full {
            ws.put_f32("qsa.idxp", idx_proj);
            (0..t)
                .map(|qt| RowSel {
                    full: true,
                    blocks: Vec::new(),
                    visible: base_pos + qt + 1,
                })
                .collect()
        } else {
            let idx_rows = e.dtoh_view(&idx_proj.slice(0..t * qk_width))?;
            ws.put_f32("qsa.idxp", idx_proj);
            let q_off = overlay.query_heads as usize * idx_dim;
            for row in 0..t {
                raw_keys.append_rows_f32(
                    &idx_rows[row * qk_width + q_off..(row + 1) * qk_width],
                    1,
                    idx_dim,
                );
            }
            // Device block scorer (long-context lane): the host twin is O(context) per
            // token per layer — 52% of the decode token at a 32k fill (smoke ladder),
            // and quadratic across a long prefill. Scores are bit-identical (same
            // arithmetic order), so the selection is the same set. `idx_dev` (default
            // ON) is the rollback seam; the host twin remains the reference and the
            // TP2 path.
            let dev_scorer = idx_dev_on();
            let sels = prof_section(e, "qsa.idx_host", || {
                indexer_select_rows(
                    overlay,
                    base,
                    qsa.yarn.as_ref().map(|y| (y.ff_host.as_slice(), y.mscale)),
                    eps,
                    &qsa.idx_q_norm,
                    &qsa.idx_k_norm,
                    &idx_rows,
                    raw_keys,
                    pooled_keys,
                    if dev_scorer {
                        Some((e, pooled_dev, pooled_dev_rows))
                    } else {
                        None
                    },
                    base_pos,
                    t,
                    t_kv,
                    pos_off,
                )
            })?;
            // Audit compare: recompute every scored row's selection from the f32 twin
            // caches (host scorer) and count flipped sets. Full rows cannot flip (the
            // structural fast path reads no scores) and are skipped. BOUNDED to
            // decode/draft/verify shapes (t <= 8): a prefill chunk would pay the
            // O(context) host selection PER ROW x 2048 rows x every chunk — quadratic
            // across a long prefill, the exact cost the device scorer retired. Prefill
            // chunks still FEED the twin (above); the twin's pooled cache catches up
            // lazily inside its next compare. Stated in the receipt: the flip rate is
            // measured on decode/verify rows at depth.
            if let Some(audit) = idx_audit.as_deref_mut() {
                if t <= 8 && sels.iter().any(|s| !s.full) {
                    let twin_sels = indexer_select_rows(
                        overlay,
                        base,
                        qsa.yarn.as_ref().map(|y| (y.ff_host.as_slice(), y.mscale)),
                        eps,
                        &qsa.idx_q_norm,
                        &qsa.idx_k_norm,
                        &idx_rows,
                        &audit.raw_f32,
                        &mut audit.pooled_f32,
                        None,
                        base_pos,
                        t,
                        t_kv,
                        pos_off,
                    )?;
                    use std::sync::atomic::Ordering::Relaxed;
                    for (a, b) in sels.iter().zip(&twin_sels) {
                        if a.full && b.full {
                            continue;
                        }
                        IDXQ_AUDIT_ROWS.fetch_add(1, Relaxed);
                        if a.full != b.full || a.blocks != b.blocks {
                            IDXQ_AUDIT_FLIPPED.fetch_add(1, Relaxed);
                            let mut diff = 0u64;
                            let (sa, sb) = (&a.blocks, &b.blocks);
                            let seta: std::collections::BTreeSet<_> = sa.iter().collect();
                            let setb: std::collections::BTreeSet<_> = sb.iter().collect();
                            diff += seta.symmetric_difference(&setb).count() as u64;
                            IDXQ_AUDIT_BLOCKS.fetch_add(diff, Relaxed);
                        }
                    }
                }
            }
            sels
        };
        Ok(sels)
    }

    fn qsa_forward(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        layer: &LayerW,
        qsa: &QsaW,
        mixed: &CudaSlice<f32>,
        mstate: &mut MixerState,
        base_pos: usize,
        t: usize,
        // Rope/indexer position offset (0 = trunk; 1 = the MTP draft, see
        // `indexer_mask_rows`). Causality stays cache-row based either way.
        pos_off: usize,
        // Verify-exact rows (mtp-spec): per-token indexer-projection launches — the
        // one cuBLASLt op in this path whose m > 1 algorithm may differ from the
        // decode-shape GEMV; m == 1 per token keeps rows bit-identical to decode.
        exact: bool,
    ) -> Res<CudaSlice<f32>> {
        let MixerState::Qsa {
            kv,
            raw_keys,
            pooled_keys,
            pooled_dev,
            pooled_dev_rows,
            raw_dev,
            raw_dev_rows,
            idx_audit,
        } = mstate
        else {
            return Err(format!(
                "qwen4exp_gpu: QSA layer {} bound to non-QSA state",
                layer.index
            )
            .into());
        };
        let hidden = self.hidden;
        let nh = qsa.attn.query_heads as usize;
        let nkv = qsa.attn.kv_heads as usize;
        let hd = qsa.attn.key_head_dim as usize;
        let eps = layer.eps_attn;
        // Mask-slot reserve: [t, capacity] never grows mid-run (t_kv does, every step).
        let cap = kv.capacity_rows(nkv * hd);

        let n_rot = qsa.attn.rope.dimensions as usize;
        let base = qsa.attn.rope.base;
        let (q, gate) = prof_section(e, "qsa.proj", || {
            let mut q_fused = ws.take_f32(e, "qsa.qf", t * 2 * nh * hd, 0)?;
            let mut k_new = ws.take_f32(e, "qsa.k", t * nkv * hd, 0)?;
            let mut v_new = ws.take_f32(e, "qsa.v", t * nkv * hd, 0)?;
            // Proj stack (round 4): wq/wk/wv in ONE launch over the row-stacked twin;
            // per-row bit-identical to the per-mat launches (OFF arm = row-offset views
            // of the same stack).
            if let (true, Some(stack)) = (
                t == 1 && proj_stack_on() && trunk_bf16_on(),
                qsa.proj_b16.as_ref(),
            ) {
                launch_qmatvec_bf16w_multi4(
                    e,
                    stack,
                    mixed,
                    &[
                        (&q_fused, 2 * nh * hd),
                        (&k_new, nkv * hd),
                        (&v_new, nkv * hd),
                    ],
                    hidden,
                )?;
            } else {
                linear_trunk_stacked_into(
                    e,
                    &qsa.wq,
                    &qsa.proj_b16,
                    0,
                    mixed,
                    &mut q_fused,
                    t,
                    hidden,
                    2 * nh * hd,
                )?;
                linear_trunk_stacked_into(
                    e,
                    &qsa.wk,
                    &qsa.proj_b16,
                    2 * nh * hd,
                    mixed,
                    &mut k_new,
                    t,
                    hidden,
                    nkv * hd,
                )?;
                linear_trunk_stacked_into(
                    e,
                    &qsa.wv,
                    &qsa.proj_b16,
                    2 * nh * hd + nkv * hd,
                    mixed,
                    &mut v_new,
                    t,
                    hidden,
                    nkv * hd,
                )?;
            }
            let mut q = ws.take_f32(e, "qsa.q", t * nh * hd, 0)?;
            let mut gate = ws.take_f32(e, "qsa.gate", t * nh * hd, 0)?;
            e.q_gate_split(&q_fused, &mut q, &mut gate, hd, nh, t)?;
            ws.put_f32("qsa.qf", q_fused);
            let mut q = if let Some(norm) = qsa.q_norm.as_ref() {
                let mut dst = ws.take_f32(e, "qsa.qn", t * nh * hd, 0)?;
                e.rms_norm(&q, norm, &mut dst, hd, t * nh, eps)?;
                ws.put_f32("qsa.q", q);
                dst
            } else {
                q
            };
            let mut k_new = if let Some(norm) = qsa.k_norm.as_ref() {
                let mut dst = ws.take_f32(e, "qsa.kn", t * nkv * hd, 0)?;
                e.rms_norm(&k_new, norm, &mut dst, hd, t * nkv, eps)?;
                ws.put_f32("qsa.k", k_new);
                dst
            } else {
                k_new
            };
            let positions: Vec<i32> = (0..t).map(|i| (base_pos + i + pos_off) as i32).collect();
            let pos_dev = ws.take_i32(e, "qsa.pos", &positions, 0)?;
            if let Some(yarn) = qsa.yarn.as_ref() {
                e.rope_neox_ffm(
                    &mut q,
                    &pos_dev,
                    hd,
                    n_rot,
                    nh,
                    t,
                    base,
                    1.0,
                    &yarn.ff,
                    yarn.mscale,
                )?;
                e.rope_neox_ffm(
                    &mut k_new,
                    &pos_dev,
                    hd,
                    n_rot,
                    nkv,
                    t,
                    base,
                    1.0,
                    &yarn.ff,
                    yarn.mscale,
                )?;
            } else {
                e.rope_neox(&mut q, &pos_dev, hd, n_rot, nh, t, base, 1.0)?;
                e.rope_neox(&mut k_new, &pos_dev, hd, n_rot, nkv, t, base, 1.0)?;
            }
            ws.put_i32("qsa.pos", pos_dev);
            // Explicit lengths: workspace slots may be larger than this chunk.
            match kv {
                QsaKvStore::F32 { k, v } => {
                    e.copy_range_into(k, base_pos * nkv * hd, &k_new, 0, t * nkv * hd)?;
                    e.copy_range_into(v, base_pos * nkv * hd, &v_new, 0, t * nkv * hd)?;
                }
                // kvq lane: append-quantize the post-RoPE rows in place (K=q8_0,
                // V=q5_1) — same slot addressing, no host round trip.
                QsaKvStore::Q8Q5 { k, v } => {
                    launch_q4e_kv_append(e, &k_new, &v_new, k, v, base_pos, t, nkv * hd)?;
                }
            }
            ws.put_f32(
                if qsa.k_norm.is_some() {
                    "qsa.kn"
                } else {
                    "qsa.k"
                },
                k_new,
            );
            ws.put_f32("qsa.v", v_new);
            Ok((q, gate))
        })?;
        let t_kv = base_pos + t;
        let sels = self.qsa_update_select(
            e,
            ws,
            qsa,
            eps,
            mixed,
            raw_keys,
            pooled_keys,
            pooled_dev,
            pooled_dev_rows,
            raw_dev,
            raw_dev_rows,
            idx_audit.as_mut(),
            base_pos,
            t,
            pos_off,
            exact,
        )?;
        let overlay = &qsa.overlay;

        let scale = match qsa.attn.scale {
            memra_gguf::model_plan::AttentionScale::InverseSqrtKeyDim => 1.0 / (hd as f32).sqrt(),
            memra_gguf::model_plan::AttentionScale::Fixed(scale) => scale,
        };
        // Long-context attention form: past the masked kernel's smem bound the dense
        // [t, t_kv] mask is impossible (bytes scale with context), so the block-list
        // kernel consumes the selection directly — BIT-IDENTICAL math (the masked
        // kernel's -1e30 rows contribute exact 0.0 terms in the same ascending order;
        // gate arm `fixture-longatt` + the blocklist kernel oracle).
        //
        // AUTO engages when the block-list form reads STRICTLY FEWER KV rows than the
        // dense form — i.e. as soon as the indexer actually drops blocks (any non-full
        // row, which on real geometry means position >= 2051) — and always past the
        // masked kernel's smem bound. This is where QSA's bounded-attention claim
        // becomes real: the dense mask still READS every t_kv row (the mask only zeroes
        // scores), so masked decode is O(context) bytes, while the block-list form reads
        // the <= 2052 selected rows at ANY depth. Measured motivation (smoke ladder,
        // yarn-1M, KV on card 1): masked decode at a 4k fill spent 97% of the token in
        // `qsa.sdpa` at 673 ms/token. Below the drop point every row IS the full prefix,
        // so the two forms read the same rows and AUTO keeps the historical masked path
        // (byte-stable receipts). `MEMRA_Q4E_SEAMS=longatt` forces it for the gate A/B;
        // `longatt=0` restores the masked-only behavior (and its long-context refusal).
        // kvq lane: the quantized cache has no masked-kernel form — the block-list
        // program (with in-place dequant) is the ONLY read path, at every depth. Below
        // the drop point every row is the full prefix, so the block-list form reads the
        // same rows the masked kernel would; there is no byte-stability question because
        // a quantized state has no historical masked receipts.
        let long_att = if kv.is_quant() {
            if longatt_mode() == LongAttMode::Off {
                return Err(
                    "qwen4exp_gpu: kvq requires the block-list attention form (longatt=off)".into(),
                );
            }
            true
        } else {
            match longatt_mode() {
                LongAttMode::Force => true,
                LongAttMode::Auto => t_kv > SDPA_MASK_TKV_BOUND || sels.iter().any(|s| !s.full),
                LongAttMode::Off => false,
            }
        };
        let block_size = overlay.block_size as usize;
        let attended = if long_att {
            let (pos_flat, meta, max_count) = rowsel_positions(&sels, block_size);
            let pos_dev = prof_section(e, "qsa.mask_h2d", || {
                ws.take_i32(e, "qsa.selpos", &pos_flat, 0)
            })?;
            let meta_dev = ws.take_i32(e, "qsa.selmeta", &meta, 0)?;
            let attended = prof_section(e, "qsa.sdpa", || {
                let mut attended = ws.take_f32(e, "qsa.att", t * nh * hd, 0)?;
                match kv {
                    QsaKvStore::F32 { k, v } => {
                        let k_view = k.slice(0..t_kv * nkv * hd);
                        let v_view = v.slice(0..t_kv * nkv * hd);
                        launch_sdpa_blocklist(
                            e,
                            &q,
                            &k_view,
                            &v_view,
                            &mut attended,
                            &pos_dev,
                            &meta_dev,
                            hd,
                            nh,
                            nkv,
                            t,
                            max_count,
                            scale,
                        )?;
                    }
                    QsaKvStore::Q8Q5 { k, v } => {
                        launch_q4e_sdpa_blocklist_q8q5(
                            e,
                            &q,
                            k,
                            v,
                            &mut attended,
                            &pos_dev,
                            &meta_dev,
                            hd,
                            nh,
                            nkv,
                            t,
                            max_count,
                            scale,
                        )?;
                    }
                }
                Ok(attended)
            })?;
            ws.put_i32("qsa.selpos", pos_dev);
            ws.put_i32("qsa.selmeta", meta_dev);
            attended
        } else {
            let QsaKvStore::F32 { k, v } = &*kv else {
                return Err("qwen4exp_gpu: masked SDPA reached with a quantized cache".into());
            };
            let mask = rowsel_to_mask(&sels, block_size, t_kv);
            let mask_dev = prof_section(e, "qsa.mask_h2d", || {
                // Masked-kernel rows never exceed the smem bound, so the slot reserve is
                // bounded even on a long-context-capacity state.
                ws.take_u8_h2d(e, "qsa.mask", &mask, t * cap.min(SDPA_MASK_TKV_BOUND))
            })?;
            let attended = prof_section(e, "qsa.sdpa", || {
                let mut attended = ws.take_f32(e, "qsa.att", t * nh * hd, 0)?;
                let k_view = k.slice(0..t_kv * nkv * hd);
                let v_view = v.slice(0..t_kv * nkv * hd);
                launch_sdpa_mask(
                    e,
                    &q,
                    &k_view,
                    &v_view,
                    &mut attended,
                    &mask_dev,
                    hd,
                    nh,
                    nkv,
                    t,
                    t_kv,
                    scale,
                )?;
                Ok(attended)
            })?;
            ws.put_u8("qsa.mask", mask_dev);
            attended
        };
        ws.put_f32(
            if qsa.q_norm.is_some() {
                "qsa.qn"
            } else {
                "qsa.q"
            },
            q,
        );
        let out = prof_section(e, "qsa.gate_wo", || {
            // fused per-(head, dim) sigmoid output gate (family convention).
            let mut sg = ws.take_f32(e, "qsa.sg", t * nh * hd, 0)?;
            e.sigmoid(&gate, &mut sg, t * nh * hd)?;
            let mut gated = ws.take_f32(e, "qsa.gated", t * nh * hd, 0)?;
            e.mul(&attended, &sg, &mut gated, t * nh * hd)?;
            let mut out = ws.take_f32(e, "mixer.out", t * hidden, 0)?;
            linear_trunk_into(
                e,
                &qsa.wo,
                &qsa.wo_b16,
                &gated,
                &mut out,
                t,
                nh * hd,
                hidden,
            )?;
            ws.put_f32("qsa.sg", sg);
            ws.put_f32("qsa.gated", gated);
            Ok(out)
        })?;
        ws.put_f32("qsa.att", attended);
        ws.put_f32("qsa.gate", gate);
        Ok(out)
    }

    /// GDN layer (`gated_delta_net` twin): fused qkv/z/beta/alpha projections, causal
    /// conv (dilation 1, silu) over cached raw rows, the geometry-generic sequential scan,
    /// gated RMSNorm with the family's SIGMOID z-gate (SEMANTICS.md §GDN).
    #[allow(clippy::too_many_arguments)]
    fn gdn_forward(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        layer: &LayerW,
        gdn: &GdnW,
        mixed: &CudaSlice<f32>,
        mstate: &mut MixerState,
        t: usize,
        // Verify-exact stash (mtp-spec): Some => per-token scan (each column the t == 1
        // decode kernel dispatch, bit-identical) + per-column state snapshots + the
        // chunk's conv-rewind inputs.
        mut stash: Option<&mut GdnStash>,
    ) -> Res<CudaSlice<f32>> {
        let MixerState::Gdn { conv, state } = mstate else {
            return Err(format!(
                "qwen4exp_gpu: GDN layer {} bound to non-GDN state",
                layer.index
            )
            .into());
        };
        let hidden = self.hidden;
        let p = &gdn.plan;
        let (nk, nv) = (p.key_heads as usize, p.value_heads as usize);
        let (hk, hv) = (p.key_head_dim as usize, p.value_head_dim as usize);
        let kernel = p.conv_kernel as usize;
        let pad = kernel - 1;
        let conv_dim = 2 * nk * hk + nv * hv;
        let eps = layer.eps_attn;

        let (qkv, z, beta_raw, g_log) = prof_section(e, "gdn.proj", || {
            let mut qkv = ws.take_f32(e, "gdn.qkv", t * conv_dim, 0)?;
            let mut z = ws.take_f32(e, "gdn.z", t * nv * hv, 0)?;
            let mut beta_raw = ws.take_f32(e, "gdn.beta", t * nv, 0)?;
            let mut alpha = ws.take_f32(e, "gdn.alpha", t * nv, 0)?;
            // Proj stack (round 4): the 4 same-activation projections in ONE launch over
            // the row-stacked twin; per-row bit-identical to the per-mat launches (the
            // OFF arm reads row-offset views of the SAME stack — same bytes, same
            // kernel, VRAM-neutral residency).
            if let (true, Some(stack)) = (
                t == 1 && proj_stack_on() && trunk_bf16_on(),
                gdn.proj_b16.as_ref(),
            ) {
                launch_qmatvec_bf16w_multi4(
                    e,
                    stack,
                    mixed,
                    &[
                        (&qkv, conv_dim),
                        (&z, nv * hv),
                        (&beta_raw, nv),
                        (&alpha, nv),
                    ],
                    hidden,
                )?;
            } else {
                linear_trunk_stacked_into(
                    e,
                    &gdn.qkv,
                    &gdn.proj_b16,
                    0,
                    mixed,
                    &mut qkv,
                    t,
                    hidden,
                    conv_dim,
                )?;
                linear_trunk_stacked_into(
                    e,
                    &gdn.z,
                    &gdn.proj_b16,
                    conv_dim,
                    mixed,
                    &mut z,
                    t,
                    hidden,
                    nv * hv,
                )?;
                linear_trunk_stacked_into(
                    e,
                    &gdn.beta,
                    &gdn.proj_b16,
                    conv_dim + nv * hv,
                    mixed,
                    &mut beta_raw,
                    t,
                    hidden,
                    nv,
                )?;
                linear_trunk_stacked_into(
                    e,
                    &gdn.alpha,
                    &gdn.proj_b16,
                    conv_dim + nv * hv + nv,
                    mixed,
                    &mut alpha,
                    t,
                    hidden,
                    nv,
                )?;
            }
            let mut g_log = ws.take_f32(e, "gdn.glog", t * nv, 0)?;
            e.gdn_glog_v(&alpha.slice(0..t * nv), &gdn.dt, &gdn.a, &mut g_log, nv, t)?;
            ws.put_f32("gdn.alpha", alpha);
            Ok((qkv, z, beta_raw, g_log))
        })?;

        let o = prof_section(e, "gdn.conv_scan", || {
            // Verify stash: the pre-chunk conv history + the chunk's raw rows are the
            // rewind rebuild inputs (pure retains — no kernel sees them). Kept OUTSIDE the
            // segment graph: they are the only part whose destination is the stash itself.
            if let Some(st) = stash.as_deref_mut() {
                e.copy_range_into(&mut st.conv_pre, 0, conv, 0, pad * conv_dim)?;
                e.copy_range_into(&mut st.qkv_rows, 0, &qkv, 0, t * conv_dim)?;
            }
            // Slots are taken (and so ALLOCATED, if this is their first use) before any
            // capture region opens; addresses are stable from here on.
            let mut conv_out = ws.take_f32(e, "gdn.conv_out", t * conv_dim, 0)?;
            let mut o = ws.take_f32(e, "gdn.o", t * nv * hv, 0)?;
            let mut tmp = if t >= pad {
                None
            } else {
                Some(ws.take_f32(e, "gdn.tmp", (pad - t) * conv_dim, 0)?)
            };
            let scale = 1.0 / (hk as f32).sqrt();
            let step_ok = gdn_step_on() && hk % 32 == 0 && hk <= 1024;
            // The dwconv -> per-column scan -> conv-history roll chain, as ONE callable
            // unit so the eager arm and the captured arm run the IDENTICAL launch
            // sequence (the graph A/B's bit-identity is by construction, not by review).
            //
            // Decode-step twin (perf round 3): one state element per thread instead of
            // one state row — geometry guard keeps the tiny plan (hk 4) on the naive
            // kernel; prefill (t > 1) always takes the naive sequential scan. VERIFY
            // chunks (stash Some) run per-token launches of the SAME dispatch decode
            // takes (step when the guard admits, else naive-at-1) with a per-column
            // state snapshot after each token — the rewind checkpoints.
            let chain = |eng: &Engine,
                         conv: &mut CudaSlice<f32>,
                         state: &mut CudaSlice<f32>,
                         states_snap: Option<&mut CudaSlice<f32>>,
                         conv_out: &mut CudaSlice<f32>,
                         o: &mut CudaSlice<f32>,
                         tmp: Option<&mut CudaSlice<f32>>|
             -> Res<()> {
                launch_dwconv(
                    eng,
                    &qkv,
                    conv,
                    &gdn.conv_w,
                    conv_out,
                    t,
                    pad,
                    conv_dim,
                    kernel,
                    1,
                    1,
                )?;
                match states_snap {
                    Some(states) => {
                        let state_len = nv * hv * hk;
                        for tok in 0..t {
                            if step_ok {
                                launch_gdn_scan_step_at(
                                    eng, conv_out, &g_log, &beta_raw, state, o, tok, nk, nv, hk,
                                    hv, scale, eps,
                                )?;
                            } else {
                                launch_gdn_scan_at(
                                    eng, conv_out, &g_log, &beta_raw, state, o, tok, nk, nv, hk,
                                    hv, scale, eps,
                                )?;
                            }
                            eng.copy_range_into(states, tok * state_len, state, 0, state_len)?;
                        }
                    }
                    None if t == 1 && step_ok => {
                        launch_gdn_scan_step(
                            eng, conv_out, &g_log, &beta_raw, state, o, nk, nv, hk, hv, scale, eps,
                        )?;
                    }
                    None => {
                        launch_gdn_scan(
                            eng, conv_out, &g_log, &beta_raw, state, o, nk, nv, hk, hv, t, scale,
                            eps,
                        )?;
                    }
                }
                // conv history <- last `pad` raw qkv rows (zeros keep their place when
                // t < pad).
                if t >= pad {
                    eng.copy_range_into(conv, 0, &qkv, (t - pad) * conv_dim, pad * conv_dim)?;
                } else {
                    let keep = pad - t;
                    let tmp = tmp.ok_or("qwen4exp_gpu: gdn conv roll needs the tmp slot")?;
                    eng.copy_range_into(tmp, 0, conv, t * conv_dim, keep * conv_dim)?;
                    eng.copy_range_into(conv, 0, tmp, 0, keep * conv_dim)?;
                    eng.copy_range_into(conv, keep * conv_dim, &qkv, 0, t * conv_dim)?;
                }
                Ok(())
            };
            // Segment graph (mtp9, default OFF): only the verify shape is graphed — plain
            // decode already has its own whole-interior graph, and prefill shapes vary.
            let graphable = stash.is_some() && verify_graphs_on() && step_ws_on() && !prof::on();
            match stash.as_deref_mut() {
                Some(st) if graphable => {
                    // Take the graph out so the snapshot buffer can be borrowed mutably.
                    // A different chunk width invalidates the capture (baked shapes).
                    let entry = match st.scan_graph.take() {
                        Some((gt, g)) if gt == t => Some(g),
                        _ => None,
                    };
                    let warm = st.scan_warm == Some(t);
                    st.scan_warm = Some(t);
                    // EXACTLY ONE of the three arms executes the chain once.
                    let entry = match (warm, entry) {
                        // First chunk at this width: eager, so every slot is allocated
                        // and parked before any capture region opens.
                        (false, _) => {
                            chain(
                                e,
                                conv,
                                state,
                                Some(&mut st.states),
                                &mut conv_out,
                                &mut o,
                                tmp.as_mut(),
                            )?;
                            None
                        }
                        // Captured at this width already: replay, no eager pass.
                        (true, Some(g)) => {
                            g.0.launch()?;
                            Some(g)
                        }
                        // Warm but not yet captured: capture WITHOUT executing
                        // (`nowarm`), then launch once — capture + launch is exactly one
                        // execution, so the column snapshots and the state advance happen
                        // exactly once.
                        (true, None) => {
                            let states = &mut st.states;
                            let mut tmp_ref = tmp.as_mut();
                            let g = e.capture_graph_retained_nowarm(|eng| {
                                chain(
                                    eng,
                                    conv,
                                    state,
                                    Some(states),
                                    &mut conv_out,
                                    &mut o,
                                    tmp_ref.as_deref_mut(),
                                )
                            })?;
                            g.0.launch()?;
                            Some(g)
                        }
                    };
                    if let Some(g) = entry {
                        st.scan_graph = Some((t, g));
                    }
                }
                Some(st) => chain(
                    e,
                    conv,
                    state,
                    Some(&mut st.states),
                    &mut conv_out,
                    &mut o,
                    tmp.as_mut(),
                )?,
                None => chain(e, conv, state, None, &mut conv_out, &mut o, tmp.as_mut())?,
            }
            ws.put_f32("gdn.conv_out", conv_out);
            if let Some(tmp) = tmp {
                ws.put_f32("gdn.tmp", tmp);
            }
            Ok(o)
        })?;
        ws.put_f32("gdn.qkv", qkv);
        ws.put_f32("gdn.beta", beta_raw);
        ws.put_f32("gdn.glog", g_log);

        let out = prof_section(e, "gdn.norm_gate_out", || {
            let mut gated = ws.take_f32(e, "gdn.gated", t * nv * hv, 0)?;
            match p.gate_activation {
                // Fused norm+gate (perf round 3): one launch, bit-identical to the
                // rms_norm + sigmoid + mul chain below (rms_sigmul_f32 kernel doc).
                GdnGateActivation::Sigmoid if gdn_fuse_on() => {
                    launch_rms_sigmul(e, &o, &gdn.norm, &z, &mut gated, hv, t * nv, eps)?;
                }
                GdnGateActivation::Sigmoid => {
                    let mut normed = ws.take_f32(e, "gdn.normed", t * nv * hv, 0)?;
                    e.rms_norm(&o, &gdn.norm, &mut normed, hv, t * nv, eps)?;
                    let mut sg = ws.take_f32(e, "gdn.sg", t * nv * hv, 0)?;
                    e.sigmoid(&z, &mut sg, t * nv * hv)?;
                    e.mul(&normed, &sg, &mut gated, t * nv * hv)?;
                    ws.put_f32("gdn.sg", sg);
                    ws.put_f32("gdn.normed", normed);
                }
                GdnGateActivation::Silu => {
                    let mut normed = ws.take_f32(e, "gdn.normed", t * nv * hv, 0)?;
                    e.rms_norm(&o, &gdn.norm, &mut normed, hv, t * nv, eps)?;
                    e.silu_mul(&z, &normed, &mut gated, t * nv * hv)?;
                    ws.put_f32("gdn.normed", normed);
                }
            }
            let mut out = ws.take_f32(e, "mixer.out", t * hidden, 0)?;
            linear_trunk_into(
                e,
                &gdn.out,
                &gdn.out_b16,
                &gated,
                &mut out,
                t,
                nv * hv,
                hidden,
            )?;
            ws.put_f32("gdn.gated", gated);
            Ok(out)
        })?;
        ws.put_f32("gdn.z", z);
        ws.put_f32("gdn.o", o);
        Ok(out)
    }

    /// MoE (`moe_mlp` twin): device router GEMM, HOST softmax-top-k routing (reference
    /// tie rule + renorm floor), per-expert gathered GEMMs, slot scatter/FMA-reduce, and
    /// the sigmoid-gated shared expert.
    fn moe_forward(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        moe: &MoeW,
        mixed: &CudaSlice<f32>,
        t: usize,
        // Rows mode (MTP draft + spec verify chunks): at t > 1, run the GROUPED decode
        // program per TOKEN — each token's launch sequence is the t == 1 program
        // verbatim (bit-identical rows), instead of the prefill per-expert executor.
        rows_grouped: bool,
        // Layer index, for the shared-format MoE route trace only (`MEMRA_MOE_TRACE`); it does
        // not select any behaviour. Threaded rather than kept in a thread-local because a hidden
        // ambient layer id is the kind of state that mislabels a whole trace file silently.
        layer: u32,
    ) -> Res<CudaSlice<f32>> {
        let hidden = self.hidden;
        let experts = moe.plan.expert_count as usize;
        let selected = moe.plan.experts_per_token as usize;
        let ff = moe.plan.expert_intermediate_size as usize;

        // Device router engage (devtwin lane): grouped dispatch only — those consumers
        // read device sel/w(/tok) arrays, so the route never crosses. NVFP4 (trunk):
        // t == 1 decode or the merged verify path (the per-token grouped twin addresses
        // its sel slot per token, which needs the host arrays). DeviceBf16 (the card-1
        // draft bank, devtwin stage 2): all rows-mode shapes via `qmatvec_bf16w_sel_f32`
        // (per-token launches read sel at a device offset — no host expert ids). The
        // per-expert prefill executor keeps the host twin (host-gathered rows by
        // construction).
        let nvfp4_bank = matches!(
            (&moe.bank.gate, &moe.bank.up, &moe.bank.down),
            (
                BankHalf::Nvfp4 { .. },
                BankHalf::Nvfp4 { .. },
                BankHalf::Nvfp4 { .. }
            )
        );
        let devbf16_bank = matches!(
            (&moe.bank.gate, &moe.bank.up, &moe.bank.down),
            (
                BankHalf::DeviceBf16(_),
                BankHalf::DeviceBf16(_),
                BankHalf::DeviceBf16(_)
            )
        );
        let use_dev_router = router_dev_on()
            && moe_sel_path_on()
            && route_dev_geometry(experts, selected)
            && ((nvfp4_bank
                && hidden % 32 == 0
                && ff % 4 == 0
                && (t == 1
                    || (rows_grouped
                        && verify_mt_on()
                        && sel_gufuse_on()
                        && t * selected <= 8192)))
                || (devbf16_bank && hidden % 8 == 0 && ff % 8 == 0 && (t == 1 || rows_grouped)));
        // (routes, device route). Exactly one is populated: host routes for the host
        // twin arms, or the device sel/w(/tok) triplet for the grouped device arms.
        type DevRoute = (CudaSlice<i32>, CudaSlice<f32>, Option<CudaSlice<i32>>);
        let (routes, mut dev_route): (Vec<Vec<(usize, f32)>>, Option<DevRoute>) =
            prof_section(e, "moe.router", || {
                let mut router_out = ws.take_f32(e, "moe.router", t * experts, 0)?;
                let none: Option<CudaSlice<u8>> = None;
                let rb = if router_bf16_on() {
                    &moe.router_b16
                } else {
                    &none
                };
                linear_trunk_into(
                    e,
                    &moe.router,
                    rb,
                    mixed,
                    &mut router_out,
                    t,
                    hidden,
                    experts,
                )?;
                if use_dev_router {
                    let mut sel = ws.take_i32_slot(e, "moe.sel", t * selected, 0)?;
                    let mut w = ws.take_f32(e, "moe.w", t * selected, 0)?;
                    let mut tokm = if t > 1 {
                        Some(ws.take_i32_slot(e, "moe.tok", t * selected, 0)?)
                    } else {
                        None
                    };
                    route_topk_device(
                        e,
                        &router_out,
                        &mut sel,
                        &mut w,
                        tokm.as_mut().map(|m| (m, 0)),
                        experts,
                        selected,
                        t,
                        layer,
                    )?;
                    ws.put_f32("moe.router", router_out);
                    return Ok((Vec::new(), Some((sel, w, tokm))));
                }
                let logits = e.dtoh_view(&router_out.slice(0..t * experts))?;
                ws.put_f32("moe.router", router_out);
                let mut routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t);
                for token in 0..t {
                    routes.push(host_route_softmax_topk(
                        &logits[token * experts..(token + 1) * experts],
                        selected,
                    ));
                }
                Ok((routes, None))
            })?;
        // Grouped decode path (perf-lane attack (a)): one kernel launch per PROJECTION
        // covers every selected expert — the per-expert dispatch below (dequant chain +
        // three tiny GEMVs + scatter per routed expert, ~52% of the decode token in
        // PROFILE-0) collapses to 6 launches per layer. NVFP4 banks + single-token decode
        // only (prefill keeps the gathered per-expert path); W4A16 — the kernel computes
        // the eager dequant chain's per-element products with a different summation order
        // (accumulation class, kernel doc), gated by the tiny four-arm + real gates.
        if (t == 1 || rows_grouped) && moe_sel_path_on() {
            if let (
                BankHalf::Nvfp4 {
                    codes: gc,
                    scales: gs,
                    macros_dev: gm,
                    ..
                },
                BankHalf::Nvfp4 {
                    codes: uc,
                    scales: us,
                    macros_dev: um,
                    ..
                },
                BankHalf::Nvfp4 {
                    codes: dc,
                    scales: ds,
                    macros_dev: dm,
                    ..
                },
            ) = (&moe.bank.gate, &moe.bank.up, &moe.bank.down)
            {
                // Merged verify columns (set_verify_mt): ONE gufuse launch over every
                // column's routed experts via the slot->token map + ONE down launch over
                // all slots + per-token windowed combines. Per-slot programs and the
                // per-token combine order are the decode program VERBATIM (bit-identical);
                // launch count per layer drops from 3t to 2 + t combines.
                if t > 1 && verify_mt_on() && sel_gufuse_on() && hidden % 32 == 0 && ff % 4 == 0 {
                    // Device-routed merged verify (devtwin): ONE batch (the engage
                    // guard bounds t*selected <= 8192 <= SLOT_CAP), per-slot programs
                    // and the per-token combine order the decode program VERBATIM —
                    // bit-identical rows; only the route's residency changed. The
                    // slot->token map comes from the route kernel, not a host build.
                    if let Some((sel, w_dev, tokm)) = dev_route.take() {
                        let tokm =
                            tokm.ok_or("moe_forward: device route at t > 1 without a tok map")?;
                        let out = prof_section(e, "moe.sel_grouped", || {
                            let mut out = ws.take_f32(e, "moe.out", t * hidden, 0)?;
                            let s_total = t * selected;
                            let mut act = ws.take_f32(e, "moe.act", s_total * ff, 0)?;
                            launch_nvfp4_sel_gu_silu(
                                e,
                                (gc, gs, gm),
                                (uc, us, um),
                                Some(&sel),
                                0,
                                s_total,
                                mixed,
                                &mut act,
                                hidden,
                                ff,
                                Some((&tokm, hidden)),
                            )?;
                            let mut partial = ws.take_f32(e, "moe.partial", s_total * hidden, 0)?;
                            launch_nvfp4_sel_matvec(
                                e,
                                dc,
                                ds,
                                dm,
                                &sel,
                                &act,
                                &mut partial,
                                s_total,
                                ff,
                                hidden,
                                ff,
                            )?;
                            for tok in 0..t {
                                launch_axpy_rows_seq_at(
                                    e,
                                    &partial,
                                    tok * selected,
                                    &w_dev,
                                    tok * selected,
                                    &mut out,
                                    tok,
                                    hidden,
                                    selected,
                                )?;
                            }
                            ws.put_i32("moe.sel", sel);
                            ws.put_i32("moe.tok", tokm);
                            ws.put_f32("moe.w", w_dev);
                            ws.put_f32("moe.act", act);
                            ws.put_f32("moe.partial", partial);
                            Ok(out)
                        })?;
                        return self.moe_shared_tail(e, ws, moe, mixed, out, t);
                    }
                    let out = prof_section(e, "moe.sel_grouped", || {
                        let mut out = ws.take_f32(e, "moe.out", t * hidden, 0)?;
                        // Slot sub-batching: the grouped kernels index slots on grid.y,
                        // which CUDA caps at 65,535 — a long-context prefill chunk
                        // (t 8192 x 10 selected = 81,920 slots) overflowed it with
                        // CUDA_ERROR_INVALID_VALUE (smoke ladder, rung 32768). Sub-batches
                        // also bound the transients (act s*ff, partial s*hidden) on a card
                        // already holding the trunk. Sub-batching changes NOTHING per slot
                        // or per token: each slot's program and each token's combine order
                        // are identical to one big batch (and to the t == 1 decode
                        // program) — the boundary only splits launches.
                        const SLOT_CAP: usize = 8192;
                        let tok_step = (SLOT_CAP / selected.max(1)).max(1);
                        let mut tok0 = 0usize;
                        while tok0 < t {
                            let tok_n = tok_step.min(t - tok0);
                            let batch = &routes[tok0..tok0 + tok_n];
                            let mut sel_all: Vec<i32> = Vec::with_capacity(tok_n * selected);
                            let mut w_all: Vec<f32> = Vec::with_capacity(tok_n * selected);
                            let mut tok_all: Vec<i32> = Vec::with_capacity(tok_n * selected);
                            let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(tok_n);
                            for (i, route) in batch.iter().enumerate() {
                                ranges.push((sel_all.len(), route.len()));
                                for &(eid, wgt) in route {
                                    sel_all.push(eid as i32);
                                    w_all.push(wgt);
                                    // ABSOLUTE token index: the kernel reads the
                                    // activation row at tok * hidden from the same
                                    // `mixed` buffer, so a sub-batch reads exactly the
                                    // rows one big batch would (no view, no offset math).
                                    tok_all.push((tok0 + i) as i32);
                                }
                            }
                            let s_total = sel_all.len();
                            let sel = ws.take_i32(e, "moe.sel", &sel_all, 0)?;
                            let w_dev = ws.take_f32_h2d(e, "moe.w", &w_all, 0)?;
                            let tokm = ws.take_i32(e, "moe.tok", &tok_all, 0)?;
                            let mut act = ws.take_f32(e, "moe.act", s_total * ff, 0)?;
                            launch_nvfp4_sel_gu_silu(
                                e,
                                (gc, gs, gm),
                                (uc, us, um),
                                Some(&sel),
                                0,
                                s_total,
                                mixed,
                                &mut act,
                                hidden,
                                ff,
                                Some((&tokm, hidden)),
                            )?;
                            let mut partial = ws.take_f32(e, "moe.partial", s_total * hidden, 0)?;
                            launch_nvfp4_sel_matvec(
                                e,
                                dc,
                                ds,
                                dm,
                                &sel,
                                &act,
                                &mut partial,
                                s_total,
                                ff,
                                hidden,
                                ff,
                            )?;
                            for (i, &(start, len)) in ranges.iter().enumerate() {
                                launch_axpy_rows_seq_at(
                                    e,
                                    &partial,
                                    start,
                                    &w_dev,
                                    start,
                                    &mut out,
                                    tok0 + i,
                                    hidden,
                                    len,
                                )?;
                            }
                            ws.put_i32("moe.sel", sel);
                            ws.put_i32("moe.tok", tokm);
                            ws.put_f32("moe.w", w_dev);
                            ws.put_f32("moe.act", act);
                            ws.put_f32("moe.partial", partial);
                            tok0 += tok_n;
                        }
                        Ok(out)
                    })?;
                    return self.moe_shared_tail(e, ws, moe, mixed, out, t);
                }
                // Device-routed decode step (devtwin, t == 1 by the engage guard): the
                // grouped decode program launch-for-launch, sel/w read from the device
                // route — bit-identical to the host-routed chain on the same selection.
                if let Some((sel, w_dev, _)) = dev_route.take() {
                    let out = prof_section(e, "moe.sel_grouped", || {
                        let mut out = ws.take_f32(e, "moe.out", hidden, 0)?;
                        let mut act = ws.take_f32(e, "moe.act", selected * ff, 0)?;
                        if sel_gufuse_on() && hidden % 32 == 0 && ff % 4 == 0 {
                            launch_nvfp4_sel_gu_silu(
                                e,
                                (gc, gs, gm),
                                (uc, us, um),
                                Some(&sel),
                                0,
                                selected,
                                mixed,
                                &mut act,
                                hidden,
                                ff,
                                None,
                            )?;
                        } else {
                            let mut yg = ws.take_f32(e, "moe.yg", selected * ff, 0)?;
                            let mut yu = ws.take_f32(e, "moe.yu", selected * ff, 0)?;
                            launch_nvfp4_sel_matvec(
                                e, gc, gs, gm, &sel, mixed, &mut yg, selected, hidden, ff, 0,
                            )?;
                            launch_nvfp4_sel_matvec(
                                e, uc, us, um, &sel, mixed, &mut yu, selected, hidden, ff, 0,
                            )?;
                            e.silu_mul(&yg, &yu, &mut act, selected * ff)?;
                            ws.put_f32("moe.yg", yg);
                            ws.put_f32("moe.yu", yu);
                        }
                        let mut partial = ws.take_f32(e, "moe.partial", selected * hidden, 0)?;
                        launch_nvfp4_sel_matvec(
                            e,
                            dc,
                            ds,
                            dm,
                            &sel,
                            &act,
                            &mut partial,
                            selected,
                            ff,
                            hidden,
                            ff,
                        )?;
                        e.axpy_rows_seq_into(&partial, &w_dev, &mut out, hidden, selected)?;
                        ws.put_i32("moe.sel", sel);
                        ws.put_f32("moe.w", w_dev);
                        ws.put_f32("moe.act", act);
                        ws.put_f32("moe.partial", partial);
                        Ok(out)
                    })?;
                    return self.moe_shared_tail(e, ws, moe, mixed, out, t);
                }
                let out = prof_section(e, "moe.sel_grouped", || {
                    let mut out = ws.take_f32(e, "moe.out", t * hidden, 0)?;
                    for (tok, route) in routes.iter().enumerate() {
                        let n_sel = route.len();
                        let sel_host: Vec<i32> = route.iter().map(|&(x, _)| x as i32).collect();
                        let w_host: Vec<f32> = route.iter().map(|&(_, w)| w).collect();
                        let sel = ws.take_i32(e, "moe.sel", &sel_host, 0)?;
                        let w_dev = ws.take_f32_h2d(e, "moe.w", &w_host, 0)?;
                        // Activation operand: t == 1 reads `mixed` in place (the decode
                        // program, launch-for-launch unchanged); rows mode stages the
                        // token's row in a stable slot (exact copy — the kernel reads
                        // identical values, so rows stay bit-identical to decode).
                        let x_tok = if t == 1 {
                            None
                        } else {
                            let mut x = ws.take_f32(e, "moe.x", hidden, 0)?;
                            e.copy_range_into(&mut x, 0, mixed, tok * hidden, hidden)?;
                            Some(x)
                        };
                        let x_ref = x_tok.as_ref().unwrap_or(mixed);
                        let mut act = ws.take_f32(e, "moe.act", n_sel * ff, 0)?;
                        // Fused gate+up+silu (round 4): ONE launch, bit-identical to the
                        // three-op chain below (kernel doc + oracle gufuse mode).
                        if sel_gufuse_on() && hidden % 32 == 0 && ff % 4 == 0 {
                            launch_nvfp4_sel_gu_silu(
                                e,
                                (gc, gs, gm),
                                (uc, us, um),
                                Some(&sel),
                                0,
                                n_sel,
                                x_ref,
                                &mut act,
                                hidden,
                                ff,
                                None,
                            )?;
                        } else {
                            let mut yg = ws.take_f32(e, "moe.yg", n_sel * ff, 0)?;
                            let mut yu = ws.take_f32(e, "moe.yu", n_sel * ff, 0)?;
                            launch_nvfp4_sel_matvec(
                                e, gc, gs, gm, &sel, x_ref, &mut yg, n_sel, hidden, ff, 0,
                            )?;
                            launch_nvfp4_sel_matvec(
                                e, uc, us, um, &sel, x_ref, &mut yu, n_sel, hidden, ff, 0,
                            )?;
                            e.silu_mul(&yg, &yu, &mut act, n_sel * ff)?;
                            ws.put_f32("moe.yg", yg);
                            ws.put_f32("moe.yu", yu);
                        }
                        let mut partial = ws.take_f32(e, "moe.partial", n_sel * hidden, 0)?;
                        launch_nvfp4_sel_matvec(
                            e,
                            dc,
                            ds,
                            dm,
                            &sel,
                            &act,
                            &mut partial,
                            n_sel,
                            ff,
                            hidden,
                            ff,
                        )?;
                        // Slot-ordered sequential combine (axpy_rows_seq_f32
                        // self-initializes); rows mode lands the row by exact copy.
                        if t == 1 {
                            e.axpy_rows_seq_into(&partial, &w_dev, &mut out, hidden, n_sel)?;
                        } else {
                            let mut row = ws.take_f32(e, "moe.row", hidden, 0)?;
                            e.axpy_rows_seq_into(&partial, &w_dev, &mut row, hidden, n_sel)?;
                            e.copy_range_into(&mut out, tok * hidden, &row, 0, hidden)?;
                            ws.put_f32("moe.row", row);
                        }
                        ws.put_i32("moe.sel", sel);
                        ws.put_f32("moe.w", w_dev);
                        ws.put_f32("moe.act", act);
                        ws.put_f32("moe.partial", partial);
                        if let Some(x) = x_tok {
                            ws.put_f32("moe.x", x);
                        }
                    }
                    Ok(out)
                })?;
                return self.moe_shared_tail(e, ws, moe, mixed, out, t);
            }
            // DeviceBf16 bank (the MTP draft): per-selected-expert row-offset bf16
            // matvecs straight off the resident bytes — n_sel launches per projection
            // (arbitrary expert ids cannot batch through the strided kernel), silu and
            // combine exactly like the NVFP4 grouped chain.
            if let (BankHalf::DeviceBf16(gb), BankHalf::DeviceBf16(ub), BankHalf::DeviceBf16(db)) =
                (&moe.bank.gate, &moe.bank.up, &moe.bank.down)
            {
                // Device-routed draft MoE (devtwin stage 2): per token, ONE
                // `qmatvec_bf16w_sel_f32` launch per projection reads its expert ids
                // from the device route at a sel offset — no host expert ids, no
                // per-slot launch chain. Per-row programs are the off_into chain
                // VERBATIM (kernel doc + the bf16 oracle's sel mode) and the combine
                // writes the same window `axpy_rows_seq` initialized — bit-identical.
                if let Some((sel, w_dev, _)) = dev_route.take() {
                    let out = prof_section(e, "moe.sel_bf16", || {
                        let mut out = ws.take_f32(e, "moe.out", t * hidden, 0)?;
                        for tok in 0..t {
                            let mut yg = ws.take_f32(e, "moe.yg", selected * ff, 0)?;
                            let mut yu = ws.take_f32(e, "moe.yu", selected * ff, 0)?;
                            launch_qmatvec_bf16w_sel(
                                e,
                                gb,
                                &sel,
                                tok * selected,
                                mixed,
                                tok * hidden,
                                0,
                                &mut yg,
                                selected,
                                hidden,
                                ff,
                            )?;
                            launch_qmatvec_bf16w_sel(
                                e,
                                ub,
                                &sel,
                                tok * selected,
                                mixed,
                                tok * hidden,
                                0,
                                &mut yu,
                                selected,
                                hidden,
                                ff,
                            )?;
                            let mut act = ws.take_f32(e, "moe.act", selected * ff, 0)?;
                            e.silu_mul(&yg, &yu, &mut act, selected * ff)?;
                            let mut partial =
                                ws.take_f32(e, "moe.partial", selected * hidden, 0)?;
                            launch_qmatvec_bf16w_sel(
                                e,
                                db,
                                &sel,
                                tok * selected,
                                &act,
                                0,
                                ff,
                                &mut partial,
                                selected,
                                ff,
                                hidden,
                            )?;
                            launch_axpy_rows_seq_at(
                                e,
                                &partial,
                                0,
                                &w_dev,
                                tok * selected,
                                &mut out,
                                tok,
                                hidden,
                                selected,
                            )?;
                            ws.put_f32("moe.yg", yg);
                            ws.put_f32("moe.yu", yu);
                            ws.put_f32("moe.act", act);
                            ws.put_f32("moe.partial", partial);
                        }
                        ws.put_i32("moe.sel", sel);
                        ws.put_f32("moe.w", w_dev);
                        Ok(out)
                    })?;
                    return self.moe_shared_tail(e, ws, moe, mixed, out, t);
                }
                let out = prof_section(e, "moe.sel_bf16", || {
                    let mut out = ws.take_f32(e, "moe.out", t * hidden, 0)?;
                    for (tok, route) in routes.iter().enumerate() {
                        let n_sel = route.len();
                        let w_host: Vec<f32> = route.iter().map(|&(_, w)| w).collect();
                        let w_dev = ws.take_f32_h2d(e, "moe.w", &w_host, 0)?;
                        let mut yg = ws.take_f32(e, "moe.yg", n_sel * ff, 0)?;
                        let mut yu = ws.take_f32(e, "moe.yu", n_sel * ff, 0)?;
                        for (slot, &(eid, _)) in route.iter().enumerate() {
                            launch_qmatvec_bf16w_off_into(
                                e,
                                gb,
                                eid * ff,
                                mixed,
                                tok * hidden,
                                &mut yg,
                                slot * ff,
                                hidden,
                                ff,
                            )?;
                            launch_qmatvec_bf16w_off_into(
                                e,
                                ub,
                                eid * ff,
                                mixed,
                                tok * hidden,
                                &mut yu,
                                slot * ff,
                                hidden,
                                ff,
                            )?;
                        }
                        let mut act = ws.take_f32(e, "moe.act", n_sel * ff, 0)?;
                        e.silu_mul(&yg, &yu, &mut act, n_sel * ff)?;
                        let mut partial = ws.take_f32(e, "moe.partial", n_sel * hidden, 0)?;
                        for (slot, &(eid, _)) in route.iter().enumerate() {
                            launch_qmatvec_bf16w_off_into(
                                e,
                                db,
                                eid * hidden,
                                &act,
                                slot * ff,
                                &mut partial,
                                slot * hidden,
                                ff,
                                hidden,
                            )?;
                        }
                        let mut row = ws.take_f32(e, "moe.row", hidden, 0)?;
                        e.axpy_rows_seq_into(&partial, &w_dev, &mut row, hidden, n_sel)?;
                        e.copy_range_into(&mut out, tok * hidden, &row, 0, hidden)?;
                        ws.put_f32("moe.row", row);
                        ws.put_f32("moe.w", w_dev);
                        ws.put_f32("moe.yg", yg);
                        ws.put_f32("moe.yu", yu);
                        ws.put_f32("moe.act", act);
                        ws.put_f32("moe.partial", partial);
                    }
                    Ok(out)
                })?;
                return self.moe_shared_tail(e, ws, moe, mixed, out, t);
            }
        }

        // A device route that reaches here would feed the per-expert executor EMPTY
        // host routes and silently compute nothing — fail loud instead (the engage
        // guard and the dispatch arms must stay in lockstep).
        if dev_route.is_some() {
            return Err(
                "moe_forward: device route left unconsumed (engage guard drifted from the \
                 dispatch arms)"
                    .into(),
            );
        }
        // expert -> [(token, slot, weight)]
        let mut by_expert: Vec<Vec<(i32, i32, f32)>> = vec![Vec::new(); experts];
        for (token, token_routes) in routes.iter().enumerate() {
            for (slot, &(expert, weight)) in token_routes.iter().enumerate() {
                by_expert[expert].push((token as i32, slot as i32, weight));
            }
        }
        let mut slots = e.zeros(t * selected * hidden)?;
        let mut wbuf = e.zeros(t * selected)?;
        for (expert, entries) in by_expert.iter().enumerate() {
            if entries.is_empty() {
                continue;
            }
            let m_e = entries.len();
            let (tok_dev, slot_dev, w_dev, xg) = prof_section(e, "moe.idx_gather", || {
                let tok_idx: Vec<i32> = entries.iter().map(|&(tok, _, _)| tok).collect();
                let slot_idx: Vec<i32> = entries.iter().map(|&(_, slot, _)| slot).collect();
                let weights: Vec<f32> = entries.iter().map(|&(_, _, w)| w).collect();
                let tok_dev = e.htod_i32(&tok_idx)?;
                let slot_dev = e.htod_i32(&slot_idx)?;
                let w_dev = e.htod(&weights)?;
                let mut xg = e.uninit(m_e * hidden)?;
                e.gather_rows(mixed, &tok_dev, &mut xg, hidden, m_e)?;
                Ok((tok_dev, slot_dev, w_dev, xg))
            })?;
            // Resolve this expert's operand views per bank half (F32 = view into the
            // resident bank; NVFP4 = per-expert kernel dequant into a transient f32).
            let resolve = |half: &BankHalf,
                           out_f: usize,
                           in_f: usize|
             -> Res<(Option<CudaSlice<f32>>, usize)> {
                match half {
                    BankHalf::F32(_) => Ok((None, expert * out_f * in_f)),
                    BankHalf::Nvfp4 {
                        codes,
                        scales,
                        macros,
                        ..
                    } => Ok((
                        Some(dequant_nvfp4_expert_f32(
                            e,
                            codes,
                            scales,
                            macros[expert],
                            expert,
                            out_f,
                            in_f,
                        )?),
                        0,
                    )),
                    // Host-resident bf16 bank: upload THIS expert's rows and upcast
                    // (exact) — the per-routed-expert twin of the load-time dequant.
                    BankHalf::HostBf16(bytes) => {
                        let row_bytes = out_f * in_f * 2;
                        let dev =
                            e.htod_bytes(&bytes[expert * row_bytes..(expert + 1) * row_bytes])?;
                        Ok((
                            Some(e.bf16_to_f32(&dev.slice(0..row_bytes), out_f * in_f)?),
                            0,
                        ))
                    }
                    // Device-resident bf16 bank (MTP draft): widen THIS expert's rows
                    // in place (exact) — the multi-token replay/prefill arm; the t == 1
                    // draft decode takes the grouped row-offset matvec path instead.
                    BankHalf::DeviceBf16(bytes) => {
                        let row_bytes = out_f * in_f * 2;
                        let view = bytes.slice(expert * row_bytes..(expert + 1) * row_bytes);
                        Ok((Some(e.bf16_to_f32(&view, out_f * in_f)?), 0))
                    }
                }
            };
            let ((gate_owned, gate_base), (up_owned, up_base), (down_owned, down_base)) =
                prof_section(e, "moe.dequant", || {
                    Ok((
                        resolve(&moe.bank.gate, ff, hidden)?,
                        resolve(&moe.bank.up, ff, hidden)?,
                        resolve(&moe.bank.down, hidden, ff)?,
                    ))
                })?;
            let gate_view = match (&moe.bank.gate, &gate_owned) {
                (_, Some(owned)) => owned.slice(0..ff * hidden),
                (BankHalf::F32(bank), None) => bank.slice(gate_base..gate_base + ff * hidden),
                (
                    BankHalf::Nvfp4 { .. } | BankHalf::HostBf16(_) | BankHalf::DeviceBf16(_),
                    None,
                ) => {
                    unreachable!("quantized/host/device-bf16 halves always resolve owned")
                }
            };
            let up_view = match (&moe.bank.up, &up_owned) {
                (_, Some(owned)) => owned.slice(0..ff * hidden),
                (BankHalf::F32(bank), None) => bank.slice(up_base..up_base + ff * hidden),
                (
                    BankHalf::Nvfp4 { .. } | BankHalf::HostBf16(_) | BankHalf::DeviceBf16(_),
                    None,
                ) => {
                    unreachable!("quantized/host/device-bf16 halves always resolve owned")
                }
            };
            let down_view = match (&moe.bank.down, &down_owned) {
                (_, Some(owned)) => owned.slice(0..hidden * ff),
                (BankHalf::F32(bank), None) => bank.slice(down_base..down_base + hidden * ff),
                (
                    BankHalf::Nvfp4 { .. } | BankHalf::HostBf16(_) | BankHalf::DeviceBf16(_),
                    None,
                ) => {
                    unreachable!("quantized/host/device-bf16 halves always resolve owned")
                }
            };
            prof_section(e, "moe.expert_gemms", || {
                let down_out =
                    run_routed_expert(e, &xg, &gate_view, &up_view, &down_view, m_e, hidden, ff)?;
                e.scatter_slot(
                    &down_out, &tok_dev, &slot_dev, &w_dev, &mut slots, &mut wbuf, hidden,
                    selected, m_e,
                )
            })?;
        }
        let out = prof_section(e, "moe.reduce", || {
            let mut out = e.zeros(t * hidden)?;
            e.reduce_slots(&slots, &wbuf, &mut out, hidden, selected, t)?;
            Ok(out)
        })?;
        self.moe_shared_tail(e, ws, moe, mixed, out, t)
    }

    /// Shared expert, sigmoid input gate (Qwen3NextSparseMoeBlock convention) — the
    /// common tail of both routed-expert executors.
    fn moe_shared_tail(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        moe: &MoeW,
        mixed: &CudaSlice<f32>,
        mut out: CudaSlice<f32>,
        t: usize,
    ) -> Res<CudaSlice<f32>> {
        let hidden = self.hidden;
        let sff = moe
            .plan
            .shared
            .as_ref()
            .map(|s| s.intermediate_size as usize)
            .unwrap_or(0);
        if sff > 0 {
            prof_section(e, "moe.shared", || {
                // hcmicro: the shared-expert mats ride the bf16 trunk residency (their
                // f32 reads were ~2.5 GB/token); OFF keeps the f32 cuBLASLt chain.
                let none: Option<CudaSlice<u8>> = None;
                let (gu, db) = if micro_shexp_on() {
                    (&moe.shared_gu_b16, &moe.shared_down_b16)
                } else {
                    (&none, &none)
                };
                let mut gate = ws.take_f32(e, "moe.sh_gate", t * sff, 0)?;
                let mut up = ws.take_f32(e, "moe.sh_up", t * sff, 0)?;
                // Proj stack (round 4): shared gate/up in ONE launch (bit-identical
                // rows; OFF arm = row-offset views of the same stack).
                if let (true, Some(stack)) =
                    (t == 1 && proj_stack_on() && trunk_bf16_on(), gu.as_ref())
                {
                    launch_qmatvec_bf16w_multi4(
                        e,
                        stack,
                        mixed,
                        &[(&gate, sff), (&up, sff)],
                        hidden,
                    )?;
                } else {
                    linear_trunk_stacked_into(
                        e,
                        &moe.shared_gate,
                        gu,
                        0,
                        mixed,
                        &mut gate,
                        t,
                        hidden,
                        sff,
                    )?;
                    linear_trunk_stacked_into(
                        e,
                        &moe.shared_up,
                        gu,
                        sff,
                        mixed,
                        &mut up,
                        t,
                        hidden,
                        sff,
                    )?;
                }
                let mut act = ws.take_f32(e, "moe.sh_act", t * sff, 0)?;
                e.silu_mul(&gate, &up, &mut act, t * sff)?;
                let mut shared = ws.take_f32(e, "moe.sh_down", t * hidden, 0)?;
                linear_trunk_into(e, &moe.shared_down, db, &act, &mut shared, t, sff, hidden)?;
                if let Some(input_gate) = moe.shared_input_gate.as_ref() {
                    // Into-variant (same kernel, same launch shape as `sigmoid_dot_rows`;
                    // the owned form allocates per call — graph capture forbids that).
                    let mut g = ws.take_f32(e, "moe.g", t, 0)?;
                    e.sigmoid_dot_rows_into(mixed, input_gate, &mut g, hidden, t)?;
                    e.add_scaled_rows(&shared, &g, &mut out, hidden, t)?;
                    ws.put_f32("moe.g", g);
                } else {
                    let mut view = out.slice_mut(0..t * hidden);
                    e.axpy_into(&shared, 1.0, &mut view, t * hidden)?;
                }
                ws.put_f32("moe.sh_gate", gate);
                ws.put_f32("moe.sh_up", up);
                ws.put_f32("moe.sh_act", act);
                ws.put_f32("moe.sh_down", shared);
                Ok(())
            })?;
        }
        Ok(out)
    }

    /// PLE block (`ple_block` twin): host n-gram hashing + host gather from the
    /// host-resident table, H2D of the gathered rows, device projections / grouped norms /
    /// dilated depthwise conv, host signed-sqrt sigmoid gate scalars.
    #[allow(clippy::too_many_arguments)]
    fn ple_block(
        &self,
        e: &Engine,
        layer: &LayerW,
        ple: &PleW,
        table: &NgramTable,
        ple_state: &mut PleState,
        planes: &mut [CudaSlice<f32>],
        tokens: &[u32],
        t: usize,
        // Verify-exact rows: per-token cuBLASLt launches (m == 1, the decode shape) so
        // chunk rows stay bit-identical to decode; `stash` retains the pre-chunk conv
        // history + the chunk's normed rows (the rewind rebuild inputs).
        exact: bool,
        mut stash: Option<&mut PleStash>,
    ) -> Res<()> {
        let hidden = self.hidden;
        let streams = self.streams;
        let plan = &ple.plan;
        let heads = plan.ngram_heads as usize;
        let head_dim = plan.head_embed_dim as usize;
        let embed_dim = plan.embed_dim as usize;
        let kernel = plan.conv_kernel as usize;
        let max_ngram = plan.max_ngram as usize;
        let dilation = max_ngram;
        let pad = (kernel - 1) * dilation;
        let eps = layer.eps_attn;

        // Host n-gram ids over the FULL history (exact segment semantics), last t rows.
        let gathered = prof_section(e, "ple.host_ngram_gather", || {
            let total_heads = heads;
            // `plecache`: extend the state's id cache instead of rebuilding the whole
            // history's hashes. `ids` owns the vector only on the OFF arm; on the ON arm the
            // chunk rows are read in place out of the state (no O(context) clone).
            let mut ids: Vec<i64> = Vec::new();
            if ple_cache_on() {
                host_ngram_ids_cached(
                    &mut ple_state.ngram_ids,
                    &mut ple_state.ngram_history,
                    &mut ple_state.ngram_last_eos,
                    tokens,
                    &ple.multipliers,
                    &ple.sizes,
                    &ple.offsets,
                    max_ngram,
                    heads / (max_ngram - 1),
                    plan.eos_token_id,
                );
                if ple_cache_audit_on() {
                    let twin = host_ngram_ids(
                        tokens,
                        &ple.multipliers,
                        &ple.sizes,
                        &ple.offsets,
                        max_ngram,
                        heads / (max_ngram - 1),
                        plan.eos_token_id,
                    );
                    let from = (tokens.len() - t) * total_heads;
                    let mism = twin[from..]
                        .iter()
                        .zip(&ple_state.ngram_ids[from..])
                        .filter(|(a, b)| a != b)
                        .count() as u64;
                    PLE_CACHE_AUDIT_ROWS.fetch_add(t as u64, std::sync::atomic::Ordering::Relaxed);
                    PLE_CACHE_AUDIT_MISMATCH.fetch_add(mism, std::sync::atomic::Ordering::Relaxed);
                    PLE_CACHE_AUDIT_MAX_FILL
                        .fetch_max(tokens.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    if mism > 0 {
                        return Err(format!(
                            "plecache audit: {mism} cached n-gram ids differ from the full twin \
                             at history {} (t={t})",
                            tokens.len()
                        )
                        .into());
                    }
                }
            } else {
                ids = host_ngram_ids(
                    tokens,
                    &ple.multipliers,
                    &ple.sizes,
                    &ple.offsets,
                    max_ngram,
                    heads / (max_ngram - 1),
                    plan.eos_token_id,
                );
            }
            let all_ids: &[i64] = if ple_cache_on() {
                &ple_state.ngram_ids
            } else {
                &ids
            };
            let chunk_ids = &all_ids[(tokens.len() - t) * total_heads..];
            let table_rows = table.rows(head_dim);
            let mut gathered = vec![0.0f32; t * embed_dim];
            for token in 0..t {
                for head in 0..heads {
                    let id = chunk_ids[token * total_heads + head];
                    if id < 0 || id as usize >= table_rows {
                        return Err("qwen4exp_gpu: n-gram id outside the embedding table".into());
                    }
                    table.gather_into(
                        id as usize,
                        head_dim,
                        &mut gathered[token * embed_dim + head * head_dim
                            ..token * embed_dim + (head + 1) * head_dim],
                    );
                }
            }
            Ok(gathered)
        })?;
        let emb = prof_section(e, "ple.h2d", || e.htod(&gathered))?;

        // Per-token cuBLASLt twin (verify-exact): every m == 1 launch matches the
        // decode dispatch for that projection, so chunk rows equal decode rows bitwise.
        let lin_rows = |x: &CudaSlice<f32>,
                        w: &CudaSlice<f32>,
                        in_f: usize,
                        out_f: usize|
         -> Res<CudaSlice<f32>> {
            let mut out = e.uninit(t * out_f)?;
            if exact && t > 1 {
                let wv = w.slice(0..w.len());
                for tok in 0..t {
                    let xv = x.slice(tok * in_f..(tok + 1) * in_f);
                    let mut yv = out.slice_mut(tok * out_f..(tok + 1) * out_f);
                    e.linear_device_into(&xv, &wv, &mut yv, 1, in_f, out_f)?;
                }
            } else {
                e.linear_device_into(x, w, &mut out, t, in_f, out_f)?;
            }
            Ok(out)
        };
        let (value, mut dots_host) = prof_section(e, "ple.key_gate", || {
            let value = lin_rows(&emb, &ple.value_proj, embed_dim, hidden)?;
            let ones = e.htod(&vec![1.0f32; hidden])?;
            let mut dots_host = vec![0.0f32; streams * t];
            for s in 0..streams {
                let key = lin_rows(&emb, &ple.key_proj[s], embed_dim, hidden)?;
                let mut key_normed = e.uninit(t * hidden)?;
                e.rms_norm(&key, &ple.norm_key[s], &mut key_normed, hidden, t, eps)?;
                let mut query = e.uninit(t * hidden)?;
                e.rms_norm(&planes[s], &ple.norm_query[s], &mut query, hidden, t, eps)?;
                let mut prod = e.uninit(t * hidden)?;
                e.mul(&key_normed, &query, &mut prod, t * hidden)?;
                let dots = lin_rows(&prod, &ones, hidden, 1)?;
                dots_host[s * t..(s + 1) * t].copy_from_slice(&e.dtoh(&dots)?);
            }
            Ok((value, dots_host))
        })?;
        // signed sqrt + sigmoid (modular L770; torch sign(0) = 0) — host scalars.
        for dot in dots_host.iter_mut() {
            let gate = *dot / (hidden as f32).sqrt();
            let magnitude = gate.abs().max(1e-6).sqrt();
            let signed = if gate > 0.0 {
                magnitude
            } else if gate < 0.0 {
                -magnitude
            } else {
                0.0
            };
            *dot = host_sigmoid(signed);
        }

        prof_section(e, "ple.conv_write", || {
            for s in 0..streams {
                let g = e.htod(&dots_host[s * t..(s + 1) * t])?;
                let mut gated = e.zeros(t * hidden)?;
                e.add_scaled_rows(&value, &g, &mut gated, hidden, t)?;
                let mut normed = e.uninit(t * hidden)?;
                e.rms_norm(&gated, &ple.norm_conv[s], &mut normed, hidden, t, eps)?;
                // Verify stash: pre-chunk history + this chunk's normed rows (rewind
                // rebuild inputs; pure retains).
                if let Some(st) = stash.as_deref_mut() {
                    e.copy_range_into(
                        &mut st.hist_pre[s],
                        0,
                        &ple_state.conv_hist[s],
                        0,
                        pad * hidden,
                    )?;
                    e.copy_range_into(&mut st.normed_rows[s], 0, &normed, 0, t * hidden)?;
                }
                // out = gated + silu(dilated causal conv(normed)) — dwconv mode 2 adds in place.
                launch_dwconv(
                    e,
                    &normed,
                    &ple_state.conv_hist[s],
                    &ple.conv_w[s],
                    &mut gated,
                    t,
                    pad,
                    hidden,
                    kernel,
                    dilation,
                    2,
                )?;
                // conv history <- last `pad` NORMED rows.
                let hist = &mut ple_state.conv_hist[s];
                if t >= pad {
                    e.copy_range_into(hist, 0, &normed, (t - pad) * hidden, pad * hidden)?;
                } else {
                    let keep = pad - t;
                    let mut tmp = e.uninit(keep * hidden)?;
                    e.copy_range_into(&mut tmp, 0, hist, t * hidden, keep * hidden)?;
                    e.copy_range_into(hist, 0, &tmp, 0, keep * hidden)?;
                    e.copy_range_into(hist, keep * hidden, &normed, 0, t * hidden)?;
                }
                // wide stream gains the PLE output BEFORE the attention read gate.
                let mut view = planes[s].slice_mut(0..t * hidden);
                e.axpy_into(&gated, 1.0, &mut view, t * hidden)?;
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------- MTP draft (mtp-spec lane)

/// The MTP draft's persistent state: its own QSA KV rows + indexer raw-key cache + a
/// dedicated step workspace. DRAFT CACHE ROW i HOLDS TARGET POSITION i + 1 (position 0
/// never enters the draft — its first input pairs token x_1 with trunk hidden h_0), so
/// every spec-loop forward runs at `pos_off = 1`; the reference-parity gate runs at
/// `pos_off = 0` to match the reference executor's row-indexed positions.
pub struct MtpDraftState {
    mixer: MixerState,
    /// Rows currently in the cache (committed + speculative chain rows).
    rows: usize,
    /// Rows whose inputs were TRUE trunk hidden states (survive a round). The spec loop
    /// truncates to here and replays accepted tokens with verify-produced hiddens.
    pub committed: usize,
    capacity: usize,
    ws: StepPool,
}

impl MtpDraftState {
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// The draft forward's token source (mtp11): host ids (the mtp10 program), host ids
/// GATHERED ON DEVICE from the full-vocab chain table (the defer arm's prefill/replay
/// shape — a 4t-byte htod replaces t 10 KB pageable embed rows, the spec.rs
/// embed_gather_device_t precedent), or ONE device slot holding the previous chain
/// step's RAW argmax (the deferred chain).
#[derive(Clone, Copy)]
enum DraftTokSrc<'a> {
    Host(&'a [u32]),
    HostDev(&'a [u32]),
    DevSlot(&'a CudaSlice<u32>, usize),
}

impl Qwen4ExpGpu {
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Card-1 draft placement armed? (`load_from_dir_dev1` — the draft's device tensors
    /// live on `mtp_dev1.dev`, and every draft call must present an engine there.)
    pub fn mtp_on_dev1(&self) -> bool {
        self.mtp_dev1.is_some()
    }

    /// The draft's device tensors were built on ONE engine; a call presenting another
    /// engine would launch kernels on the wrong context (UVA would make it "work"
    /// slowly instead of failing). Enforced, never assumed.
    fn check_draft_engine(&self, e: &Engine) -> Res<()> {
        if let Some(d) = self.mtp_dev1.as_ref() {
            if e.ctx().ordinal() != d.dev {
                return Err(format!(
                    "qwen4exp_gpu: the draft lives on device {} (card-1 placement); \
                     this call presented device {}",
                    d.dev,
                    e.ctx().ordinal()
                )
                .into());
            }
        }
        Ok(())
    }

    /// Allocate the draft's persistent state (its own KV plane; `capacity` rows).
    /// With the card-1 placement, `e` must be the DRAFT engine.
    pub fn mtp_state(&self, e: &Engine, capacity: usize) -> Res<MtpDraftState> {
        self.check_draft_engine(e)?;
        let mtp = self
            .mtp
            .as_ref()
            .ok_or("qwen4exp_gpu: no MTP block loaded (LoadOptions::load_mtp)")?;
        let MixerW::Qsa(qsa) = &mtp.layer.mixer else {
            return Err("qwen4exp_gpu: MTP mixer is not QSA".into());
        };
        let kv_width = qsa.attn.kv_heads as usize * qsa.attn.key_head_dim as usize;
        let v_width = qsa.attn.kv_heads as usize * qsa.attn.value_head_dim as usize;
        // kvq/idxq: the draft's QSA cache follows the same latched formats as the trunk
        // (uniform storage; the spec byte-identity gates run same-config on both arms).
        let kv = if kv_quant_on() {
            QsaKvStore::Q8Q5 {
                k: e.alloc_u8(capacity * q8_row_bytes(kv_width))?,
                v: e.alloc_u8(capacity * q5_row_bytes(v_width))?,
            }
        } else {
            QsaKvStore::F32 {
                k: e.zeros(capacity * kv_width)?,
                v: e.zeros(capacity * v_width)?,
            }
        };
        Ok(MtpDraftState {
            mixer: MixerState::Qsa {
                kv,
                raw_keys: IdxRawCache::new(idxq_mode()),
                pooled_keys: Vec::new(),
                pooled_dev: None,
                pooled_dev_rows: 0,
                raw_dev: None,
                raw_dev_rows: 0,
                idx_audit: None,
            },
            rows: 0,
            committed: 0,
            capacity,
            ws: StepPool::default(),
        })
    }

    /// Truncate the draft cache to `rows` (speculative chain rows die; KV rows are
    /// overwritten in place by the next append, the host raw-key cache truncates).
    pub fn mtp_rewind(&self, dstate: &mut MtpDraftState, rows: usize) -> Res<()> {
        if rows > dstate.rows {
            return Err("qwen4exp_gpu: mtp_rewind past the cache".into());
        }
        let mtp = self.mtp.as_ref().ok_or("qwen4exp_gpu: no MTP block")?;
        let MixerW::Qsa(qsa) = &mtp.layer.mixer else {
            return Err("qwen4exp_gpu: MTP mixer is not QSA".into());
        };
        let MixerState::Qsa {
            raw_keys,
            pooled_keys,
            pooled_dev_rows,
            raw_dev_rows,
            ..
        } = &mut dstate.mixer
        else {
            return Err("qwen4exp_gpu: MTP state is not QSA".into());
        };
        let idx_dim = qsa.overlay.head_dim as usize;
        raw_keys.truncate_rows(rows, idx_dim);
        let block = qsa.overlay.block_size as usize;
        pooled_keys.truncate((rows / block) * idx_dim);
        // The device mirror's row count MUST follow the host truncation, or the next
        // scorer call skips the H2D of rebuilt rows and scores STALE keys (caught by the
        // spec byte-identity arms).
        *pooled_dev_rows = (*pooled_dev_rows).min(pooled_keys.len() / idx_dim);
        // Device raw-key cache (idxcache): clamp to the ABSOLUTE kept row count — the
        // host cache may legitimately lag below it (the lazy materialization).
        *raw_dev_rows = (*raw_dev_rows).min(rows);
        dstate.rows = rows;
        dstate.committed = dstate.committed.min(rows);
        Ok(())
    }

    /// One MTP draft forward over `t` rows (SEMANTICS.md §MTP): fused input =
    /// `fc_embedding(norm(embed(tok)))` broadcast over streams + per-stream
    /// `fc_hidden(FLAT norm(wide hidden))`; ONE QSA+MoE decoder layer on the draft's own
    /// cache; exit through the draft mixer into the SHARED lm_head. Returns
    /// `(logits [t, vocab], carrier [t, wide])` — the carrier is the POST-LAYER wide
    /// state, the K > 1 multi-step seed. Recycle both via `mtp_recycle`.
    ///
    /// `hidden_wide` rows start at row `wide_off` of the given buffer; row r seeds
    /// token r. `pos_off` = 1 in the spec loop (draft row i ↔ target position i+1),
    /// 0 in the reference-parity gate.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_draft_forward(
        &self,
        e: &Engine,
        tokens: &[u32],
        hidden_wide: &CudaSlice<f32>,
        wide_off: usize,
        dstate: &mut MtpDraftState,
        pos_off: usize,
        // true => logits for EVERY row (the parity gates); false => the LAST row only
        // (the spec loop's shape — earlier rows exist for the KV cache + carrier, and
        // the full-vocab head must not scale with the replay length).
        logits_all: bool,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>)> {
        self.mtp_draft_forward_impl(
            e,
            DraftTokSrc::Host(tokens),
            hidden_wide,
            wide_off,
            dstate,
            pos_off,
            logits_all,
        )
    }

    /// One DEFERRED chain step (mtp11): the input token is the previous step's device
    /// argmax, read from `toks[slot]` (RAW draft-index space; embeds through the armed
    /// chain table). t == 1 by construction; `pos_off` is the spec loop's 1.
    fn mtp_draft_forward_devslot(
        &self,
        e: &Engine,
        toks: &CudaSlice<u32>,
        slot: usize,
        hidden_wide: &CudaSlice<f32>,
        wide_off: usize,
        dstate: &mut MtpDraftState,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>)> {
        self.mtp_draft_forward_impl(
            e,
            DraftTokSrc::DevSlot(toks, slot),
            hidden_wide,
            wide_off,
            dstate,
            1,
            false,
        )
    }

    /// Spec-loop host-token draft forward (prefill / bootstrap / replay shapes):
    /// `dev_embed` keys the defer arm's device-gather embed (full-vocab chain table)
    /// vs the mtp10 host embed — the control arm stays byte- AND structure-frozen.
    fn mtp_draft_forward_spec(
        &self,
        e: &Engine,
        tokens: &[u32],
        dev_embed: bool,
        hidden_wide: &CudaSlice<f32>,
        wide_off: usize,
        dstate: &mut MtpDraftState,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>)> {
        let src = if dev_embed {
            DraftTokSrc::HostDev(tokens)
        } else {
            DraftTokSrc::Host(tokens)
        };
        self.mtp_draft_forward_impl(e, src, hidden_wide, wide_off, dstate, 1, false)
    }

    /// `mtp_draft_forward_spec` over RING-slotted seed rows: absolute seed row
    /// `first_row + i` lives at slot `(first_row + i) % ring`, and a range crossing the
    /// ring seam splits into two draft calls. The split changes the draft GEMM shape on
    /// seam rounds (drafted tokens may differ there — acceptance-only; commits are
    /// always the target rows, so spec byte-identity is untouched by construction).
    /// Returns the LAST piece's (logits row, carrier, piece length).
    fn draft_consume_ring(
        &self,
        de: &Engine,
        tokens: &[u32],
        dev_embed: bool,
        seed: &CudaSlice<f32>,
        ring: usize,
        first_row: usize,
        dstate: &mut MtpDraftState,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>, usize)> {
        let mut out: Option<(CudaSlice<f32>, CudaSlice<f32>, usize)> = None;
        let mut done = 0usize;
        while done < tokens.len() {
            let slot = (first_row + done) % ring;
            let len = (tokens.len() - done).min(ring - slot);
            let (l, c) = self.mtp_draft_forward_spec(
                de,
                &tokens[done..done + len],
                dev_embed,
                seed,
                slot,
                dstate,
            )?;
            if let Some((pl, pc, _)) = out.take() {
                self.mtp_recycle(dstate, pl, pc);
            }
            out = Some((l, c, len));
            done += len;
        }
        out.ok_or("qwen4exp_gpu: empty draft consume".into())
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_draft_forward_impl(
        &self,
        e: &Engine,
        tok_src: DraftTokSrc<'_>,
        hidden_wide: &CudaSlice<f32>,
        wide_off: usize,
        dstate: &mut MtpDraftState,
        pos_off: usize,
        logits_all: bool,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>)> {
        self.check_draft_engine(e)?;
        let mtp = self
            .mtp
            .as_ref()
            .ok_or("qwen4exp_gpu: no MTP block loaded (LoadOptions::load_mtp)")?;
        let t = match tok_src {
            DraftTokSrc::Host(tokens) | DraftTokSrc::HostDev(tokens) => tokens.len(),
            DraftTokSrc::DevSlot(..) => 1,
        };
        let hidden = self.hidden;
        let streams = self.streams;
        let wide = streams * hidden;
        if t == 0 {
            return Err("qwen4exp_gpu: empty draft input".into());
        }
        if dstate.rows + t > dstate.capacity {
            return Err("qwen4exp_gpu: draft state capacity exceeded".into());
        }
        if hidden_wide.len() < (wide_off + t) * wide {
            return Err("qwen4exp_gpu: draft hidden seed rows out of range".into());
        }
        let base = dstate.rows;
        let ws = &mut dstate.ws;
        let cap = dstate.capacity;

        // ---- input fusion
        let mut planes = prof_section(e, "mtp.fuse", || {
            let emb = match tok_src {
                DraftTokSrc::Host(tokens) => {
                    let mut embedded = vec![0.0f32; t * hidden];
                    for (row, &token) in tokens.iter().enumerate() {
                        let token = token as usize;
                        if token >= self.vocab {
                            return Err(
                                format!("qwen4exp_gpu: draft token {token} out of range").into()
                            );
                        }
                        embedded[row * hidden..(row + 1) * hidden].copy_from_slice(
                            &self.embed_host[token * hidden..(token + 1) * hidden],
                        );
                    }
                    ws.take_f32_h2d(e, "mtp.emb", &embedded, cap * hidden)?
                }
                DraftTokSrc::HostDev(tokens) => {
                    // Defer arm's prefill/replay embed (mtp11): host ids validated
                    // here, then a 4t-byte htod + device gather from the FULL-VOCAB
                    // chain table — bit-identical rows (ChainEmbed contract), no
                    // t x 10 KB pageable h2d. The caller keys this on a full-vocab
                    // table (a trim table cannot embed arbitrary target ids).
                    let ce = self
                        .chain_embed
                        .as_ref()
                        .filter(|ce| !ce.for_trim && ce.rows == self.vocab)
                        .ok_or("qwen4exp_gpu: HostDev embed needs the full-vocab chain table")?;
                    for &token in tokens {
                        if token as usize >= self.vocab {
                            return Err(
                                format!("qwen4exp_gpu: draft token {token} out of range").into()
                            );
                        }
                    }
                    let tok_d = e.gpu.stream().clone_htod(tokens)?;
                    let mut emb = ws.take_f32(e, "mtp.emb", t * hidden, cap * hidden)?;
                    let tv = tok_d.slice(0..t);
                    embed_gather_rows_into(
                        e,
                        &ce.table,
                        &tv,
                        &mut emb,
                        t,
                        hidden,
                        ce.qt,
                        ce.row_bytes,
                    )?;
                    emb
                }
                DraftTokSrc::DevSlot(toks, slot) => {
                    // Deferred chain (mtp11): gather THE row for the RAW draft index
                    // in `toks[slot]` from the armed chain table — bit-identical to
                    // the host row (ChainEmbed contract), no host round trip of the
                    // token id, no pageable h2d. Index bound is by construction:
                    // the argmax that wrote the slot scanned exactly `rows` columns.
                    let ce = self
                        .chain_embed
                        .as_ref()
                        .ok_or("qwen4exp_gpu: deferred draft step without arm_spec_devchain")?;
                    let mut emb = ws.take_f32(e, "mtp.emb", hidden, cap * hidden)?;
                    let tv = toks.slice(slot..slot + 1);
                    embed_gather_rows_into(
                        e,
                        &ce.table,
                        &tv,
                        &mut emb,
                        1,
                        hidden,
                        ce.qt,
                        ce.row_bytes,
                    )?;
                    emb
                }
            };
            let mut enorm = ws.take_f32(e, "mtp.enorm", t * hidden, 0)?;
            e.rms_norm(
                &emb,
                &mtp.pre_norm_embed,
                &mut enorm,
                hidden,
                t,
                mtp.eps_embed,
            )?;
            let mut evec = ws.take_f32(e, "mtp.evec", t * hidden, 0)?;
            linear_trunk_into(
                e,
                &mtp.fc_embed,
                &mtp.fc_embed_b16,
                &enorm,
                &mut evec,
                t,
                hidden,
                hidden,
            )?;
            // Stage the seed rows at offset 0 (exact copy), then FLAT-norm the whole
            // wide vector per token (GemmaRMSNorm_wide — SEMANTICS.md §MTP).
            let mut hin = ws.take_f32(e, "mtp.hin", t * wide, 0)?;
            e.copy_range_into(&mut hin, 0, hidden_wide, wide_off * wide, t * wide)?;
            let mut hnorm = ws.take_f32(e, "mtp.hnorm", t * wide, 0)?;
            e.rms_norm(
                &hin,
                &mtp.pre_norm_hidden,
                &mut hnorm,
                wide,
                t,
                mtp.eps_hidden,
            )?;
            // fc_hidden per stream = the same [H, H] mat over every (token, stream) row
            // of the normed wide buffer viewed [t*streams, H].
            let mut fused = ws.take_f32(e, "mtp.fused", t * wide, 0)?;
            linear_trunk_into(
                e,
                &mtp.fc_hidden,
                &mtp.fc_hidden_b16,
                &hnorm,
                &mut fused,
                t * streams,
                hidden,
                hidden,
            )?;
            let mut planes: Vec<CudaSlice<f32>> = Vec::with_capacity(streams);
            for s in 0..streams {
                let mut plane = ws.take_f32(e, PLANE_SLOTS[s], t * hidden, cap * hidden)?;
                for tok in 0..t {
                    e.copy_range_into(
                        &mut plane,
                        tok * hidden,
                        &fused,
                        (tok * streams + s) * hidden,
                        hidden,
                    )?;
                }
                let mut view = plane.slice_mut(0..t * hidden);
                e.axpy_into(&evec, 1.0, &mut view, t * hidden)?;
                planes.push(plane);
            }
            ws.put_f32("mtp.emb", emb);
            ws.put_f32("mtp.enorm", enorm);
            ws.put_f32("mtp.evec", evec);
            ws.put_f32("mtp.hin", hin);
            ws.put_f32("mtp.hnorm", hnorm);
            ws.put_f32("mtp.fused", fused);
            Ok(planes)
        })?;

        let ptr_vals: Vec<u64> = {
            let stream = e.gpu.stream();
            planes.iter().map(|p| p.device_ptr(&stream).0).collect()
        };
        let ptrs = ws.take_u64_h2d(e, "hc.ptrs", &ptr_vals, 0)?;

        // ---- the one decoder layer (trunk program, draft weights/cache)
        let layer = &mtp.layer;
        let (mixed, inject) = prof_section(e, "mtp.hyper.read", || {
            self.gate_read(
                e,
                ws,
                &ptrs,
                &layer.attn_gate,
                &planes,
                t,
                layer.eps_attn,
                false,
            )
        })?;
        let MixerW::Qsa(qsa) = &layer.mixer else {
            return Err("qwen4exp_gpu: MTP mixer is not QSA".into());
        };
        let block_out = prof_section(e, "mtp.qsa", || {
            self.qsa_forward(
                e,
                ws,
                layer,
                qsa,
                &mixed,
                &mut dstate.mixer,
                base,
                t,
                pos_off,
                false,
            )
        })?;
        ws.put_f32("hc.mixed", mixed);
        prof_section(e, "mtp.hyper.write", || {
            self.gate_write(e, &mut planes, &ptrs, &block_out, &inject, t)
        })?;
        ws.put_f32("mixer.out", block_out);
        put_inject(ws, inject);
        let (mixed, inject) = prof_section(e, "mtp.hyper.read", || {
            self.gate_read(
                e,
                ws,
                &ptrs,
                &layer.mlp_gate,
                &planes,
                t,
                layer.eps_mlp,
                false,
            )
        })?;
        let mlp = prof_section(e, "mtp.moe", || {
            // Rows mode for chain/replay shapes; the big draft PREFILL takes the
            // per-expert executor (each expert's rows widen once for all its tokens).
            self.moe_forward(e, ws, &layer.moe, &mixed, t, t <= 32, layer.index)
        })?;
        ws.put_f32("hc.mixed", mixed);
        prof_section(e, "mtp.hyper.write", || {
            self.gate_write(e, &mut planes, &ptrs, &mlp, &inject, t)
        })?;
        ws.put_f32("moe.out", mlp);
        put_inject(ws, inject);

        // ---- carrier (post-layer wide state, PRE exit mixer — the K>1 seed)
        let mut carrier = ws.take_f32(e, "mtp.carrier", t * wide, 0)?;
        for (s, plane) in planes.iter().enumerate() {
            for tok in 0..t {
                e.copy_range_into(
                    &mut carrier,
                    tok * wide + s * hidden,
                    plane,
                    tok * hidden,
                    hidden,
                )?;
            }
        }

        // ---- exit: the draft's own mixer read (no inject) -> shared lm_head.
        // Only the LAST row's logits are ever consumed (chain steps run t == 1; the
        // replay/prefill rows exist for the KV cache and the carrier), so the head
        // reads one hidden row — the full-vocab matvec is the draft's single largest
        // cost (mtp4 profile) and must not scale with the replay length.
        let x = prof_section(e, "mtp.exit", || {
            Ok(self
                .gate_read_inner(
                    e,
                    ws,
                    &ptrs,
                    &mtp.mixer,
                    &planes,
                    t,
                    self.exit_eps,
                    false,
                    false,
                )?
                .0)
        })?;
        ws.put_u64("hc.ptrs", ptrs);
        // The head is the SHARED trunk head, or its FR-Spec trimmed gather when the draft
        // trim is armed (mtp9): out_f drops from the 248,320 vocab to N, which is the
        // draft's single largest cost. Same bytes either way — a trimmed row's logit is
        // bit-identical to its full-vocab twin.
        let trim = self.draft_trim.as_ref();
        let out_f = trim.map_or(self.vocab, |t| t.n);
        // Card-1 placement reads its private head copy (same bytes, same program);
        // otherwise the shared trunk head. Trim + dev1 is refused at build time.
        let (head_w, head_b16) = match self.mtp_dev1.as_ref() {
            Some(d) => (&d.output, &d.output_b16),
            None => (&self.output, &self.output_b16),
        };
        let head_into =
            |e: &Engine, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, rows: usize| -> Res<()> {
                match trim {
                    Some(trim) => linear_trim_into(e, trim, x, y, rows, hidden),
                    None => linear_trunk_into(e, head_w, head_b16, x, y, rows, hidden, self.vocab),
                }
            };
        let logits = prof_section(e, "mtp.lm_head", || {
            if logits_all {
                let mut logits = ws.take_f32(e, "mtp.logits", t * out_f, 0)?;
                head_into(e, &x, &mut logits, t)?;
                return Ok(logits);
            }
            let mut logits = ws.take_f32(e, "mtp.logits", out_f, 0)?;
            let mut x_last = ws.take_f32(e, "mtp.xlast", hidden, 0)?;
            e.copy_range_into(&mut x_last, 0, &x, (t - 1) * hidden, hidden)?;
            head_into(e, &x_last, &mut logits, 1)?;
            ws.put_f32("mtp.xlast", x_last);
            Ok(logits)
        })?;
        ws.put_f32("hc.mixed", x);
        for (s, plane) in planes.into_iter().enumerate() {
            ws.put_f32(PLANE_SLOTS[s], plane);
        }
        dstate.rows += t;
        Ok((logits, carrier))
    }

    /// Return a draft step's logits/carrier buffers to the draft workspace (address
    /// reuse across the hot loop).
    pub fn mtp_recycle(
        &self,
        dstate: &mut MtpDraftState,
        logits: CudaSlice<f32>,
        carrier: CudaSlice<f32>,
    ) {
        dstate.ws.put_f32("mtp.logits", logits);
        dstate.ws.put_f32("mtp.carrier", carrier);
    }
}

// ---------------------------------------------------------------- spec decode (mtp-spec lane)

/// Vendor-default sampling config for the SAMPLED spec run (the serving law's probe
/// shape): temp 1.0 / top_p 0.95 / top_k 20 on qwen4_exp. Greedy (None) stays the
/// byte-identity instrument.
#[derive(Clone, Copy)]
pub struct SpecSamplerCfg {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
}

/// xorshift64* — deterministic, seedable, dependency-free (receipt reproducibility).
struct SpecRng(u64);

impl SpecRng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
        bits as f32 / (1u64 << 24) as f32
    }
}

/// Host top-k/top-p/temperature sample over one logits row.
fn sample_row(cfg: &SpecSamplerCfg, rng: &mut SpecRng, row: &[f32]) -> u32 {
    let k = cfg.top_k.max(1).min(row.len());
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    let mut top: Vec<(u32, f32)> = idx[..k].iter().map(|&i| (i, row[i as usize])).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    let temp = cfg.temperature.max(1e-6);
    let mx = top[0].1;
    let mut probs: Vec<f32> = top.iter().map(|&(_, v)| ((v - mx) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    // top_p nucleus over the sorted tail.
    let mut cut = probs.len();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= cfg.top_p {
            cut = i + 1;
            break;
        }
    }
    let renorm: f32 = probs[..cut].iter().sum();
    let draw = rng.next_f32() * renorm;
    let mut acc = 0.0f32;
    for (i, &p) in probs[..cut].iter().enumerate() {
        acc += p;
        if draw < acc {
            return top[i].0;
        }
    }
    top[cut - 1].0
}

/// Host argmax with the plain chain's tie rule (strictly-greater keeps the smallest
/// index) — bit-identical to the device 2-pass argmax (argmax-gate contract), which is
/// what lets the trace and plain-tail paths commit host argmaxes without moving a chain.
fn host_argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// P2P-copy `t` wide rows at row offset `off` from the card-0 verify wide stash into
/// the card-1 mirror (mtp10 dev1 draft placement), issued on the DRAFT engine's stream
/// and host-synced — the sync is where the crossing is TIMED, and the draft's next
/// kernels queue behind the copy on the same stream either way. Host ordering
/// guarantees the source rows are complete: every call site sits after a `forward`
/// whose host dtoh (logits or argmax) synced card 0's stream.
/// Ring-contiguous pieces of an absolute wide-row range [off, off+t): (slot_off, len)
/// per piece — one piece unless the range crosses the ring seam (then two). Identity
/// slots when ring >= off + t never wraps (the historical whole-history stash).
fn ring_pieces(ring: usize, off: usize, t: usize) -> Vec<(usize, usize)> {
    debug_assert!(t <= ring, "wide-ring consumer wider than the ring");
    let slot = off % ring;
    if slot + t <= ring {
        vec![(slot, t)]
    } else {
        vec![(slot, ring - slot), (0, t - (ring - slot))]
    }
}

fn cross_wide_rows(
    e: &Engine,
    de: &Engine,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    off: usize,
    t: usize,
    wide: usize,
) -> Res<f64> {
    let t0 = std::time::Instant::now();
    let stream = de.gpu.stream();
    let bytes = t * wide * 4;
    let byte_off = (off * wide * 4) as u64;
    let (sp, _g0) = src.device_ptr(&stream);
    let (dp, _g1) = dst.device_ptr_mut(&stream);
    unsafe {
        cudarc::driver::result::memcpy_peer_async(
            de.ctx().cu_ctx(),
            dp + byte_off,
            e.ctx().cu_ctx(),
            sp + byte_off,
            bytes,
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e3)
}

/// Launch `embed_gather_u32_t` for ONE device-slot token into the pooled `mtp.emb`
/// buffer (mtp11 deferred chain). Same kernel as the lib.rs `embed_gather_device_*`
/// family — bit-identical rows by the same per-dtype deq contract. Lives here (not as
/// an Engine method) because the deferred chain is this module's machinery.
fn embed_gather_rows_into(
    e: &Engine,
    table: &CudaSlice<u8>,
    tok_v: &CudaView<u32>,
    x_out: &mut CudaSlice<f32>,
    t: usize,
    n_embd: usize,
    qtype: i32,
    row_bytes: usize,
) -> Res<()> {
    let f = e.func("embed_gather_u32_t");
    let cfg = LaunchConfig {
        grid_dim: (((n_embd as u32).div_ceil(256)).max(1), t as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (ne, qt, rb, ti) = (n_embd as i32, qtype, row_bytes as i64, t as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(table)
        .arg(tok_v)
        .arg(x_out)
        .arg(&ne)
        .arg(&qt)
        .arg(&rb)
        .arg(&ti);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// One spec run's counters (the accept-length table's source).
#[derive(Debug, Default, Clone)]
pub struct SpecReport {
    pub tokens: Vec<u32>,
    pub rounds: usize,
    pub drafted: u64,
    pub accepted: u64,
    /// hist[a] = rounds that accepted exactly `a` drafts (a in 0..=k).
    pub accept_hist: Vec<u64>,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub prefill_ms: f64,
    pub total_ms: f64,
    /// draft_ms split (the round-cost identity table): the K-step chain, the accepted-
    /// token catch-up replay, and the one-time draft prefill. draft_ms is their sum.
    pub chain_ms: f64,
    pub replay_ms: f64,
    pub draft_prefill_ms: f64,
    /// Card-1 crossing cost (mtp10 dev1 placement): wall time and bytes of the P2P
    /// wide-row copies (prefill seed + per-round replay seeds). 0 on one card.
    pub cross_ms: f64,
    pub cross_bytes: u64,
    /// Dynamic-K admission (mtp10): every decay as (round, new_k). Empty = K never moved.
    pub k_decays: Vec<(usize, usize)>,
    /// Token count at which the policy turned spec fully OFF (k reached 0); the rest of
    /// the generation ran plain decode steps (counted in `plain_steps`, not `rounds`).
    pub spec_off_at: Option<usize>,
    pub plain_steps: usize,
    /// Per-round wall samples: (tokens committed so far, ms since generation start),
    /// appended after every round/plain step — lets a caller derive N timing sub-rounds
    /// from ONE generation (the x3-rounds protocol where a fresh prefill per timing
    /// round is prohibitive, e.g. the 1M ladder).
    pub round_wall: Vec<(usize, f64)>,
    /// p-min guard accounting: rounds that drafted NOTHING (verify = plain t==1 step)
    /// and chain steps cut short (the sub-threshold token discarded uncounted).
    pub zero_draft_rounds: usize,
    pub guard_stops: usize,
}

/// Bounded shape-aware spec admission (mtp10): rolling-accept-driven K decay. Every
/// round pushes its accept count into a window of the last `window` rounds; when the
/// window is full and its mean accept < `thr` (draft tokens per round, 0..=k), K steps
/// DOWN by one (never up — decay only, bounded and monotone) and the window resets so
/// decays are at least `window` rounds apart. At K = `k_floor` the decay stops; with
/// `k_floor` = 0 reaching it turns spec OFF for the REST of the generation (plain
/// greedy decode steps — the draft cost is what the collapsed shape was paying for).
/// Byte identity is untouched BY CONSTRUCTION at every K: committed tokens are always
/// the target rows' argmax, and the plain tail IS the plain program.
#[derive(Clone, Copy, Debug)]
pub struct DynKCfg {
    pub window: usize,
    pub thr: f64,
    pub k_floor: usize,
}

/// Spec-round admission options (mtp10). Every knob defaults OFF; each is a bounded
/// policy that can only shrink the drafted window — the committed output is the target
/// rows' argmax at every setting, so byte identity is untouched by construction.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecOpts {
    /// Rolling-window K decay (the last-resort shape bound). See `DynKCfg`.
    pub dynk: Option<DynKCfg>,
    /// Adaptive per-round window (the dflash MEMRA_DFLASH_ADAPT "accepted+1" recipe):
    /// next round drafts clamp(last_accept + 1, k_lo, k). `Some(k_lo)` arms it.
    pub adapt_k_lo: Option<usize>,
    /// p-min draft-confidence guard (the MEMRA_SPEC_PMIN mechanism, sub-threshold token
    /// DISCARDED UNCOUNTED — the reference engines' normalization). Applies at j == 0
    /// too (the MEMRA_SPEC_PMIN0 zero-draft-round semantics): a low-confidence round
    /// drafts NOTHING and its verify is a plain t == 1 step that still commits one
    /// token — unpredictable stretches never pay draft + verify-column overhead.
    /// 0.0 = off.
    pub pmin: f32,
    /// Deferred round readback (mtp11, the spec.rs slice-2 structure ported): the
    /// chain's argmax feeds the next step ON DEVICE through the armed chain-embed
    /// table (`arm_spec_devchain` required), the guard's confidences land in device
    /// slots, and the chain drains ONCE per round before the verify (the PLE host
    /// n-gram gather needs the chunk's token ids, so this family's floor is a 2-drain
    /// round, not spec.rs's 1). t == 1 steps take the device-argmax fast path and the
    /// prefill dtoh shrinks to one row. Committed bytes identical BY CONSTRUCTION
    /// (same kernels, same picks; spec-gate arbitrates). Default OFF (flags law);
    /// mutually exclusive with `trace` (trace reads per-step host rows).
    pub defer: bool,
    /// With `defer` + `pmin`: keep the guard SEQUENTIAL — one 4-byte prob dtoh per
    /// chain step, the chain stops exactly at the sub-threshold step (today's cost
    /// shape). Default OFF = the deferred guard: probabilities drain with the chain
    /// and truncate at the FIRST sub-threshold step — same picks and counters
    /// bit-for-bit, but the dispatched suffix past the stop is work the sequential
    /// arm never paid. The guard-forces-a-readback A/B the owner asked to measure.
    pub defer_guard_sync: bool,
    /// Long-context lane: chunked co-prefill (trunk chunk forward with the head
    /// skipped, then the draft consumes that chunk's wide rows) instead of the one-shot
    /// prompt forward — the one-shot shape at 500k+ would materialize chunk-sized
    /// transients per plane AND a [n, vocab] logits block. `None` = the historical
    /// one-shot (byte-stable receipts).
    pub prefill_chunk: Option<usize>,
    /// Long-context lane: RING-bounded wide stash rows (`spec_arm_ring`) — at 1M
    /// capacity the whole-history stash is ~41 GB/card. Requires `prefill_chunk` (the
    /// co-prefill consumes each chunk before the ring overwrites it) and must be
    /// >= 2 * prefill_chunk. `None` = whole-history (the historical layout).
    pub wide_ring: Option<usize>,
}

/// The deferred guard's drain-time truncation (mtp11): the FIRST sub-threshold
/// confidence (predicate `p < pmin` — the host chain's exact stop rule, boundary
/// p == pmin PASSES) ends the drafted window; picks before it survive, the
/// sub-threshold pick is discarded uncounted, everything after is dispatch the
/// sequential arm never paid. Pure so the tiny gate can pin the walk on arbitrary
/// windows: mid-chain dips are unreachable on the deterministic tiny fixture
/// (intra-round confidence never crosses a passed threshold there), so this pin plus
/// the real-model `--defer-ab` counter identity are the mid-chain coverage.
pub fn spec_guard_trunc(probs: &[f32], pmin: f32) -> usize {
    probs.iter().position(|&p| p < pmin).unwrap_or(probs.len())
}

/// One traced spec round (the mtp10 thinkon-decay diagnosis instrument). Trace mode
/// changes NOTHING the accept walk sees — it only reads: draft logit rows, carrier
/// seeds, and the verify's captured wide rows come to host for margin/drift stats.
/// (Greedy trace runs with host-argmax targets — the same argmax the plain chain uses,
/// proven equal to the device walk by the spec-gate.)
#[derive(Debug, Default, Clone)]
pub struct SpecTraceRound {
    pub round: usize,
    /// Committed generation length BEFORE this round (position within the generation).
    pub gen_pos: usize,
    /// Trunk committed rows before the round (the tip's absolute position).
    pub base: usize,
    pub k: usize,
    pub a: usize,
    pub drafts: Vec<u32>,
    /// k+1 target rows (the committed prefix is targets[0..=a]).
    pub targets: Vec<u32>,
    /// Fork-row stats (row `a`, present when a < k): the draft's top-2 logits, the
    /// draft's logit and rank of the token the TARGET wanted, the target's top-2 logits,
    /// the target's logit of the token the DRAFT proposed, and the target row's softmax
    /// entropy (nats). NaN/0 when the round accepted everything (no fork).
    pub draft_top1: f32,
    pub draft_top2: f32,
    pub draft_tgt_logit: f32,
    pub draft_tgt_rank: usize,
    pub target_top1: f32,
    pub target_top2: f32,
    pub target_draft_logit: f32,
    pub target_entropy: f64,
    /// Carrier drift per carrier-seeded chain step j = 1..k-1: the seed the draft used
    /// (its own predicted wide for position base+j-1) vs the trunk's TRUE wide row at
    /// that position (captured by the verify chunk). rel_l2 = ||seed-true||/||true||.
    pub carrier_rel_l2: Vec<f32>,
    pub carrier_cos: Vec<f32>,
}

impl SpecReport {
    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
    /// Mean committed tokens per round (accepted + bonus).
    pub fn mean_accept_len(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.tokens.len() as f64 / self.rounds as f64
        }
    }
}

impl Qwen4ExpGpu {
    /// Arm the verify instrument on `state`: absolute-position wide capture (the
    /// draft's hidden seeds) + per-column GDN/PLE stashes for chunks up to `k_cap`
    /// columns. Idempotent for the same k_cap.
    pub fn spec_arm(&self, e: &Engine, state: &mut Qwen4ExpState, k_cap: usize) -> Res<()> {
        self.spec_arm_ring(e, state, k_cap, state.capacity)
    }

    /// `spec_arm` with a RING-bounded wide stash (long-context lane): the stash holds the
    /// last `ring_rows` wide rows (slot = row % ring_rows) instead of `capacity` rows —
    /// at 1M capacity the full stash is ~41 GB/card, the ring ~0.7 GB. Every consumer
    /// reads rows within `ring_rows` of the write head (chunked co-prefill consumes each
    /// chunk before the next lands; rounds read the last k+2 rows), asserted at the read
    /// helpers. `spec_arm` (ring = capacity) keeps the historical byte-stable layout.
    pub fn spec_arm_ring(
        &self,
        e: &Engine,
        state: &mut Qwen4ExpState,
        k_cap: usize,
        ring_rows: usize,
    ) -> Res<()> {
        let ring_rows = ring_rows.min(state.capacity).max(k_cap + 2);
        if let Some(v) = state.verify.as_ref() {
            if v.k_cap == k_cap && v.ring_rows == ring_rows {
                return Ok(());
            }
        }
        let wide = self.streams * self.hidden;
        let mut gdn = Vec::with_capacity(self.layers.len());
        let mut ple = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            gdn.push(match &layer.mixer {
                MixerW::Gdn(g) => {
                    let p = &g.plan;
                    let (nk, nv) = (p.key_heads as usize, p.value_heads as usize);
                    let (hk, hv) = (p.key_head_dim as usize, p.value_head_dim as usize);
                    let conv_dim = 2 * nk * hk + nv * hv;
                    let pad = p.conv_kernel as usize - 1;
                    Some(GdnStash {
                        states: e.zeros(k_cap * nv * hv * hk)?,
                        conv_pre: e.zeros(pad * conv_dim)?,
                        qkv_rows: e.zeros(k_cap * conv_dim)?,
                        scan_graph: None,
                        scan_warm: None,
                    })
                }
                MixerW::Qsa(_) => None,
            });
            ple.push(match layer.ple.as_ref() {
                Some(pw) => {
                    let pad = (pw.plan.conv_kernel as usize - 1) * pw.plan.max_ngram as usize;
                    let mut hist_pre = Vec::with_capacity(self.streams);
                    let mut normed_rows = Vec::with_capacity(self.streams);
                    for _ in 0..self.streams {
                        hist_pre.push(e.zeros(pad * self.hidden)?);
                        normed_rows.push(e.zeros(k_cap * self.hidden)?);
                    }
                    Some(PleStash {
                        hist_pre,
                        normed_rows,
                    })
                }
                None => None,
            });
        }
        state.verify = Some(VerifyStash {
            k_cap,
            chunk: None,
            fused_chunk: None,
            gdn,
            ple,
            wide: e.zeros(ring_rows * wide)?,
            ring_rows,
            wide_dev1: None,
            argmax: Vec::new(),
            toks: unsafe { e.gpu.stream().alloc::<u32>(k_cap)? },
            want_argmax: false,
            want_argmax_t1: false,
            last_row_only: false,
        });
        Ok(())
    }

    pub fn spec_disarm(&self, state: &mut Qwen4ExpState) {
        state.verify = None;
    }

    pub fn set_verify_want_argmax(&self, state: &mut Qwen4ExpState, on: bool) -> Res<()> {
        state
            .verify
            .as_mut()
            .ok_or("qwen4exp_gpu: verify not armed")?
            .want_argmax = on;
        Ok(())
    }

    /// The last exact chunk's per-row device-argmax tokens (want_argmax mode).
    pub fn verify_argmax_rows<'s>(&self, state: &'s Qwen4ExpState) -> Res<&'s [u32]> {
        Ok(&state
            .verify
            .as_ref()
            .ok_or("qwen4exp_gpu: verify not armed")?
            .argmax)
    }

    /// Rewind the trunk state to the first `keep` rows of the live verify chunk:
    /// bookkeeping truncation + GDN state restore from the per-column snapshots + GDN/
    /// PLE conv-history rebuild from the stashed pre-chunk history and chunk rows.
    /// `keep == t` is the all-accepted fast path (state already correct).
    pub fn verify_rewind(&self, e: &Engine, state: &mut Qwen4ExpState, keep: usize) -> Res<()> {
        let Some(v) = state.verify.as_mut() else {
            return Err("qwen4exp_gpu: verify not armed".into());
        };
        let Some((base, t)) = v.chunk.take() else {
            if let Some((fb, ft)) = v.fused_chunk.take() {
                return Err(format!(
                    "qwen4exp_gpu: verify chunk (base {fb}, t {ft}) ran the FUSED program \
                     (`vfuse` cost instrument) — no per-column GDN/PLE stash exists, so it \
                     cannot be rewound. vfuse is a timing probe on a throwaway state; drop \
                     the seam to run a spec loop."
                )
                .into());
            }
            return Err("qwen4exp_gpu: no live verify chunk to rewind".into());
        };
        if keep == 0 || keep > t {
            return Err("qwen4exp_gpu: rewind keep out of range".into());
        }
        if keep == t {
            return Ok(());
        }
        state.pos = base + keep;
        state.tokens.truncate(base + keep);
        for (li, (layer, lstate)) in self.layers.iter().zip(state.layers.iter_mut()).enumerate() {
            match (&layer.mixer, &mut lstate.mixer) {
                (
                    MixerW::Qsa(qsa),
                    MixerState::Qsa {
                        raw_keys,
                        pooled_keys,
                        pooled_dev_rows,
                        raw_dev_rows,
                        idx_audit,
                        ..
                    },
                ) => {
                    let idx_dim = qsa.overlay.head_dim as usize;
                    raw_keys.truncate_rows(base + keep, idx_dim);
                    let block = qsa.overlay.block_size as usize;
                    pooled_keys.truncate(((base + keep) / block) * idx_dim);
                    // Device mirror follows the host truncation (see mtp_rewind).
                    *pooled_dev_rows = (*pooled_dev_rows).min(pooled_keys.len() / idx_dim);
                    // Device raw-key cache (idxcache): clamp to the ABSOLUTE kept row
                    // count (the host cache may lag below it — lazy materialization).
                    *raw_dev_rows = (*raw_dev_rows).min(base + keep);
                    // The audit twin tracks the cache rows exactly (instrument).
                    if let Some(audit) = idx_audit.as_deref_mut() {
                        audit.raw_f32.truncate_rows(base + keep, idx_dim);
                        audit.pooled_f32.truncate(((base + keep) / block) * idx_dim);
                    }
                }
                (MixerW::Gdn(g), MixerState::Gdn { conv, state: rec }) => {
                    let st = v.gdn[li]
                        .as_mut()
                        .ok_or("qwen4exp_gpu: GDN layer without a verify stash")?;
                    let p = &g.plan;
                    let (nk, nv) = (p.key_heads as usize, p.value_heads as usize);
                    let (hk, hv) = (p.key_head_dim as usize, p.value_head_dim as usize);
                    let conv_dim = 2 * nk * hk + nv * hv;
                    let pad = p.conv_kernel as usize - 1;
                    let state_len = nv * hv * hk;
                    e.copy_range_into(rec, 0, &st.states, (keep - 1) * state_len, state_len)?;
                    if keep >= pad {
                        e.copy_range_into(
                            conv,
                            0,
                            &st.qkv_rows,
                            (keep - pad) * conv_dim,
                            pad * conv_dim,
                        )?;
                    } else {
                        let keep_hist = pad - keep;
                        e.copy_range_into(
                            conv,
                            0,
                            &st.conv_pre,
                            keep * conv_dim,
                            keep_hist * conv_dim,
                        )?;
                        e.copy_range_into(
                            conv,
                            keep_hist * conv_dim,
                            &st.qkv_rows,
                            0,
                            keep * conv_dim,
                        )?;
                    }
                }
                _ => return Err("qwen4exp_gpu: mixer/state mismatch in rewind".into()),
            }
            if let (Some(pw), Some(ps)) = (layer.ple.as_ref(), lstate.ple.as_mut()) {
                let st = v.ple[li]
                    .as_mut()
                    .ok_or("qwen4exp_gpu: PLE layer without a verify stash")?;
                let pad = (pw.plan.conv_kernel as usize - 1) * pw.plan.max_ngram as usize;
                let hidden = self.hidden;
                for s in 0..self.streams {
                    let hist = &mut ps.conv_hist[s];
                    if keep >= pad {
                        e.copy_range_into(
                            hist,
                            0,
                            &st.normed_rows[s],
                            (keep - pad) * hidden,
                            pad * hidden,
                        )?;
                    } else {
                        let keep_hist = pad - keep;
                        e.copy_range_into(
                            hist,
                            0,
                            &st.hist_pre[s],
                            keep * hidden,
                            keep_hist * hidden,
                        )?;
                        e.copy_range_into(
                            hist,
                            keep_hist * hidden,
                            &st.normed_rows[s],
                            0,
                            keep * hidden,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Device argmax of ONE draft-logits row (4-byte dtoh), returned as a TARGET vocab
    /// id: the row width is the trim width when armed and the winning row maps back
    /// through d2t (identity when the trim is off). `conf` (the p-min guard, prior art
    /// MEMRA_SPEC_PMIN / gemma confidence-adaptive draft depth — the SAME
    /// prob_of_token kernels) additionally returns the head's softmax confidence in its
    /// own pick: one extra 2-pass sum-exp launch + a 4-byte dtoh. Under a trim the
    /// confidence reads the TRIMMED row (inflated vs full softmax — thresholds are
    /// per-configuration, stated in the receipt).
    fn draft_row_argmax(
        &self,
        e: &Engine,
        logits: &CudaSlice<f32>,
        row: usize,
        conf: bool,
    ) -> Res<(u32, f32)> {
        let width = self.draft_logits_width();
        let mut tok = unsafe { e.gpu.stream().alloc::<u32>(1)? };
        e.argmax_token_device_col(logits, row, width, &mut tok, 0)?;
        let p = if conf {
            if row != 0 {
                // The chain shape is single-row; prob_of_token reads logits[0..width].
                return Err("qwen4exp_gpu: draft confidence reads row 0 (the chain shape)".into());
            }
            let pd = e.prob_of_token_device(logits, &tok, width)?;
            e.gpu.stream().clone_dtoh(&pd)?[0]
        } else {
            1.0
        };
        Ok((self.draft_token(e.gpu.stream().clone_dtoh(&tok)?[0])?, p))
    }

    /// MTP speculative decode (mtp-spec lane): prefill, draft-prefill the MTP block
    /// over the prompt, then rounds of K-token drafting (single-layer draft, carrier-
    /// chained) + ONE trunk verify chunk (t = K+1, every row bit-identical to the
    /// t == 1 decode program) + greedy accept walk + replay-free partial rewind.
    ///
    /// Greedy (sampler None) is the byte-identity instrument: output must equal the
    /// spec-off greedy chain token for token. `Some(cfg)` runs the vendor-default
    /// sampled shape: targets are SAMPLED per verify row (draft accepted on exact
    /// match — distribution-preserving), the serving law's probe.
    ///
    /// This wrapper is the single-card, no-admission, no-trace shape; the full seam is
    /// `spec_generate_ext`.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_generate(
        &self,
        e: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
        state: &mut Qwen4ExpState,
        dstate: &mut MtpDraftState,
        sampler: Option<SpecSamplerCfg>,
    ) -> Res<SpecReport> {
        self.spec_generate_ext(
            e,
            e,
            prompt,
            max_new,
            k,
            state,
            dstate,
            sampler,
            SpecOpts::default(),
            None,
        )
    }

    /// `spec_generate` with the mtp10 seams:
    /// - `de` — the DRAFT engine. Same card as `e` by default; the card-1 placement
    ///   (`load_from_dir_dev1`) requires the dev1 engine here and P2P-crosses the wide
    ///   seed rows per round (timed into `report.cross_ms`).
    /// - `opts` — bounded spec admission (p-min guard / adaptive K / dyn-K decay), all
    ///   default OFF. Every knob only shrinks the drafted window; commits are always
    ///   the target rows, so byte identity holds at every setting by construction.
    /// - `trace` — per-round diagnosis records (accept positions, fork margins, carrier
    ///   drift). Trace mode only ADDS reads (dtoh) and swaps the device accept-argmax
    ///   for the bit-identical host argmax; the committed chain is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_generate_ext(
        &self,
        e: &Engine,
        de: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
        state: &mut Qwen4ExpState,
        dstate: &mut MtpDraftState,
        sampler: Option<SpecSamplerCfg>,
        opts: SpecOpts,
        mut trace: Option<&mut Vec<SpecTraceRound>>,
    ) -> Res<SpecReport> {
        use std::time::Instant;
        if k == 0 {
            return Err("qwen4exp_gpu: spec needs k >= 1".into());
        }
        let n = prompt.len();
        if n < 2 {
            return Err("qwen4exp_gpu: spec needs a >= 2 token prompt".into());
        }
        if state.pos != 0 || dstate.rows != 0 {
            return Err("qwen4exp_gpu: spec_generate wants FRESH trunk + draft states".into());
        }
        if state.capacity < n + max_new + k + 2 || dstate.capacity < n + max_new + k + 2 {
            return Err("qwen4exp_gpu: state capacity too small for prompt + max_new + k".into());
        }
        self.check_draft_engine(de)?;
        let dev1 = self.mtp_dev1.is_some();
        if !dev1 && de.ctx().ordinal() != e.ctx().ordinal() {
            return Err(
                "qwen4exp_gpu: draft engine on another card, but the draft was not \
                 built there (load_from_dir_dev1)"
                    .into(),
            );
        }
        let vocab = self.vocab;
        let wide_w = self.streams * self.hidden;
        let greedy = sampler.is_none();
        let tracing = trace.is_some();
        let guard = opts.pmin > 0.0;
        let deferred = opts.defer;
        if deferred && tracing {
            return Err(
                "qwen4exp_gpu: spec defer + trace are mutually exclusive (the trace \
                 instrument reads per-step host rows); run the trace on the host-chain arm"
                    .into(),
            );
        }
        if deferred {
            let ce = self.chain_embed.as_ref().ok_or(
                "qwen4exp_gpu: SpecOpts::defer needs arm_spec_devchain on the draft engine",
            )?;
            if ce.dev != de.ctx().ordinal() {
                return Err(format!(
                    "qwen4exp_gpu: the chain-embed table lives on device {} but the \
                     draft engine is device {} — re-arm arm_spec_devchain",
                    ce.dev,
                    de.ctx().ordinal()
                )
                .into());
            }
            if ce.for_trim != self.draft_trim.is_some() || ce.rows != self.draft_logits_width() {
                return Err(
                    "qwen4exp_gpu: the chain-embed table was armed for a different trim \
                     state — re-arm arm_spec_devchain after trim changes"
                        .into(),
                );
            }
        }
        // Deferred-round device slots (ONE alloc per generation, on the draft engine):
        // chain picks in RAW draft-index space + the guard's per-step confidence.
        let (mut chain_toks_d, mut chain_probs_d) = if deferred {
            (
                Some(unsafe { de.gpu.stream().alloc::<u32>(k)? }),
                Some(de.zeros(k)?),
            )
        } else {
            (None, None)
        };
        let mut rng = sampler
            .as_ref()
            .map(|cfg| SpecRng(cfg.seed | 1))
            .unwrap_or(SpecRng(1));
        let t_total = Instant::now();
        let mut report = SpecReport {
            accept_hist: vec![0; k + 1],
            ..Default::default()
        };

        match opts.wide_ring {
            Some(ring) => {
                let chunk = opts
                    .prefill_chunk
                    .ok_or("qwen4exp_gpu: SpecOpts::wide_ring needs prefill_chunk")?;
                if ring < 2 * chunk || ring < 2 * (k + 2) {
                    return Err("qwen4exp_gpu: wide_ring must cover 2 prefill chunks".into());
                }
                self.spec_arm_ring(e, state, k + 1, ring)?;
            }
            None => self.spec_arm(e, state, k + 1)?,
        }
        self.set_verify_want_argmax(state, false)?;
        if let Some(v) = state.verify.as_mut() {
            // mtp11 deferred seam: t == 1 steps commit through the device argmax
            // (greedy only) and big-t prefills dtoh one row instead of the block.
            v.want_argmax_t1 = deferred && greedy && !tracing;
            v.last_row_only = deferred;
        }
        // Card-1 mirror of the wide stash (the draft's seed source on the dev1 route) —
        // ring-sized like the stash itself (same slot addressing on both cards).
        let ring = state.verify.as_ref().expect("armed above").ring_rows;
        if dev1 {
            let v = state.verify.as_mut().expect("armed above");
            if v.wide_dev1.as_ref().is_none_or(|m| m.len() < ring * wide_w) {
                v.wide_dev1 = Some(de.zeros(ring * wide_w)?);
            }
        }
        // Defer arm's draft-side embed route: device gather from the full-vocab chain
        // table (a trim table cannot embed arbitrary target/prompt ids — host embed
        // stays the trim fallback, stated). Control arm (defer off): host embed,
        // structure-frozen.
        let dev_embed = deferred
            && self
                .chain_embed
                .as_ref()
                .is_some_and(|ce| !ce.for_trim && ce.rows == self.vocab);
        let t_prefill = Instant::now();
        let mut draft_prefill_ms = 0f64;
        let x0: u32 = match opts.prefill_chunk {
            // ---- Long-context CO-PREFILL (chunked): trunk chunk forward with the head
            // skipped (LastRow on the final chunk — a [n, vocab] logits block at 500k
            // would be hundreds of GB), then the dev1 crossing + the draft consuming
            // THAT chunk's wide rows before the ring overwrites them. Piece boundaries
            // keep the final piece past k_cap so no prefill chunk takes the verify-exact
            // path.
            Some(chunk) if n > chunk => {
                let mut b = 0usize;
                let mut last = Vec::new();
                while b < n {
                    let mut t = chunk.min(n - b);
                    // Never leave a <= k_cap remainder as its own final piece.
                    if n - (b + t) > 0 && n - (b + t) <= k + 1 {
                        t = n - b;
                    }
                    let is_last = b + t == n;
                    let head = if is_last {
                        HeadMode::LastRow
                    } else {
                        HeadMode::Skip
                    };
                    let piece = self.forward_with(e, &prompt[b..b + t], state, None, head)?;
                    let t_draft = Instant::now();
                    if dev1 {
                        let v = state.verify.as_mut().expect("armed above");
                        let VerifyStash {
                            wide, wide_dev1, ..
                        } = v;
                        let mirror = wide_dev1.as_mut().expect("allocated above");
                        for (slot, len) in ring_pieces(ring, b, t) {
                            report.cross_ms +=
                                cross_wide_rows(e, de, wide, mirror, slot, len, wide_w)?;
                        }
                        report.cross_bytes += (t * wide_w * 4) as u64;
                    }
                    // Draft rows for positions [max(b,1), b+t): token p seeds wide[p-1]
                    // (the previous chunk's last row stays live: ring >= 2 chunks).
                    let p0 = b.max(1);
                    if b + t > p0 {
                        let v = state.verify.as_ref().expect("armed above");
                        let seed: &CudaSlice<f32> = v.wide_dev1.as_ref().unwrap_or(&v.wide);
                        let (ld, cd, _) = self.draft_consume_ring(
                            de,
                            &prompt[p0..b + t],
                            dev_embed,
                            seed,
                            ring,
                            p0 - 1,
                            dstate,
                        )?;
                        self.mtp_recycle(dstate, ld, cd);
                    }
                    draft_prefill_ms += t_draft.elapsed().as_secs_f64() * 1e3;
                    b += t;
                    if is_last {
                        last = piece;
                    }
                }
                dstate.committed = n - 1;
                debug_assert_eq!(last.len(), vocab);
                match sampler.as_ref() {
                    None => host_argmax(&last) as u32,
                    Some(cfg) => sample_row(cfg, &mut rng, &last),
                }
            }
            // ---- Historical one-shot prefill (byte-stable receipts).
            _ => {
                let prefill = self.forward(e, prompt, state, None)?;
                // Shape-agnostic last-row read: the deferred seam's prefill dtoh is ONE
                // row (last_row_only), the control arm's is the full block; both end at
                // the row x0 reads. (A prompt shorter than k+2 runs the prefill as an
                // exact chunk and returns full rows on both arms.)
                let last = &prefill[prefill.len() - vocab..];
                let x0 = match sampler.as_ref() {
                    None => host_argmax(last) as u32,
                    Some(cfg) => sample_row(cfg, &mut rng, last),
                };
                let t_draft0 = Instant::now();
                if dev1 {
                    let v = state.verify.as_mut().expect("armed above");
                    let VerifyStash {
                        wide, wide_dev1, ..
                    } = v;
                    let mirror = wide_dev1.as_mut().expect("allocated above");
                    for (slot, len) in ring_pieces(ring, 0, n) {
                        report.cross_ms += cross_wide_rows(e, de, wide, mirror, slot, len, wide_w)?;
                    }
                    report.cross_bytes += (n * wide_w * 4) as u64;
                }
                {
                    let v = state.verify.as_ref().expect("armed above");
                    let seed: &CudaSlice<f32> = v.wide_dev1.as_ref().unwrap_or(&v.wide);
                    if n >= 2 {
                        let (ld, cd, _) = self.draft_consume_ring(
                            de,
                            &prompt[1..],
                            dev_embed,
                            seed,
                            ring,
                            0,
                            dstate,
                        )?;
                        self.mtp_recycle(dstate, ld, cd);
                    }
                    dstate.committed = n - 1;
                }
                draft_prefill_ms += t_draft0.elapsed().as_secs_f64() * 1e3;
                x0
            }
        };
        report.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3 - draft_prefill_ms;
        // Trace mode keeps the full verify-logits dtoh (want_argmax off) so fork
        // margins can be read; targets then come from the bit-identical host argmax.
        self.set_verify_want_argmax(state, greedy && !tracing)?;
        // x0 is the first generated token (parity with the plain chain's first argmax).
        report.tokens.push(x0);

        // Bootstrap tip row: (x0 at position n, hidden wide[n-1]).
        let t_boot = Instant::now();
        let (mut tip_logits, mut tip_carrier) = {
            let v = state.verify.as_ref().expect("armed above");
            let seed: &CudaSlice<f32> = v.wide_dev1.as_ref().unwrap_or(&v.wide);
            self.mtp_draft_forward_spec(de, &[x0], dev_embed, seed, (n - 1) % ring, dstate)?
        };
        let mut tip_rows = 1usize;
        dstate.committed = dstate.rows;
        report.draft_prefill_ms = draft_prefill_ms + t_boot.elapsed().as_secs_f64() * 1e3;
        report.draft_ms += report.draft_prefill_ms;

        let mut m = n; // trunk committed rows; tip sits at position m
        let mut tip = x0;
        // Admission state: k_cur = the dyn-K ceiling (decay-only), k_next = the
        // adaptive per-round window (accepted+1 recipe), window = the dyn-K ring.
        let mut k_cur = k;
        let mut k_next = k;
        let mut window: Vec<usize> = Vec::new();
        let mut round_idx = 0usize;
        while report.tokens.len() < max_new {
            if k_cur == 0 {
                // Dyn-K floored at 0: spec OFF for the remainder. Plain decode steps
                // (host argmax = the plain program — byte identity by construction);
                // the draft never runs again, which is exactly the saved cost.
                let row = self.forward(e, &[tip], state, None)?;
                let next: u32 = match sampler.as_ref() {
                    // Deferred seam: the plain step's token is the device argmax
                    // (bit-identical, argmax-gate contract); `row` is empty here.
                    None if deferred => self.verify_argmax_rows(state)?[0],
                    None => host_argmax(&row) as u32,
                    Some(cfg) => sample_row(cfg, &mut rng, &row),
                };
                report.tokens.push(next);
                report.plain_steps += 1;
                report
                    .round_wall
                    .push((report.tokens.len(), t_total.elapsed().as_secs_f64() * 1e3));
                m += 1;
                tip = next;
                continue;
            }
            let k_round = k_next.min(k_cur).max(1);
            // ---- draft chain: d1 from the tip row; steps 2..k_round carrier-chained.
            // The p-min guard stops the chain at the first sub-threshold pick (token
            // discarded uncounted); at j == 0 that makes a ZERO-draft round whose
            // verify is a plain t == 1 step.
            let t_draft = Instant::now();
            let mut drafts: Vec<u32> = Vec::with_capacity(k_round);
            let mut chain_rows_h: Vec<Vec<f32>> = Vec::new(); // trace: draft logit rows
            let mut seeds_h: Vec<Vec<f32>> = Vec::new(); // trace: carrier seeds used
            if let (Some(toks), Some(probs)) = (chain_toks_d.as_mut(), chain_probs_d.as_mut()) {
                // ---- DEFERRED chain (mtp11): picks and confidences stay in device
                // slots; the next step's embed gathers from the chain table, so host
                // dispatch of step j+1 overlaps device execution of step j and the
                // round drains ONCE (below) instead of blocking 2 dtoh per step.
                let width = self.draft_logits_width();
                de.argmax_token_device_col(&tip_logits, 0, width, toks, 0)?;
                if guard {
                    de.prob_of_token_device_col(&tip_logits, toks, 0, probs, 0, width)?;
                }
                let mut prev_logits = tip_logits;
                let mut prev_carrier = tip_carrier;
                let mut prev_rows = tip_rows;
                // Device slots holding a pick so far (guard_sync: a CHECKED pick).
                let mut drafted = 1usize;
                let mut stopped = false;
                if guard && opts.defer_guard_sync {
                    // Sequential-guard sub-arm: one 4-byte prob dtoh per step, the
                    // chain stops exactly where the host arm would (the discarded
                    // sub-threshold pick stays in its slot, uncounted).
                    let p = de.gpu.stream().clone_dtoh(&probs.slice(0..1))?[0];
                    if p < opts.pmin {
                        drafted = 0;
                        stopped = true;
                        report.guard_stops += 1;
                    }
                }
                while !stopped && drafted < k_round {
                    let (l2, c2) = self.mtp_draft_forward_devslot(
                        de,
                        toks,
                        drafted - 1,
                        &prev_carrier,
                        prev_rows - 1,
                        dstate,
                    )?;
                    self.mtp_recycle(dstate, prev_logits, prev_carrier);
                    prev_logits = l2;
                    prev_carrier = c2;
                    prev_rows = 1;
                    de.argmax_token_device_col(&prev_logits, 0, width, toks, drafted)?;
                    if guard {
                        de.prob_of_token_device_col(
                            &prev_logits,
                            toks,
                            drafted,
                            probs,
                            drafted,
                            width,
                        )?;
                        if opts.defer_guard_sync {
                            let p = de
                                .gpu
                                .stream()
                                .clone_dtoh(&probs.slice(drafted..drafted + 1))?[0];
                            if p < opts.pmin {
                                report.guard_stops += 1;
                                break;
                            }
                        }
                    }
                    drafted += 1;
                }
                self.mtp_recycle(dstate, prev_logits, prev_carrier);
                // ---- the round's ONE chain drain: the picks (and the deferred
                // guard's confidences) cross together; raw indices map to target ids
                // through draft_token, and the deferred guard truncates at the FIRST
                // sub-threshold step — the same discard the sequential arm makes.
                if drafted > 0 {
                    let raw = de.gpu.stream().clone_dtoh(&toks.slice(0..drafted))?;
                    let trunc = if guard && !opts.defer_guard_sync {
                        let pw = de.gpu.stream().clone_dtoh(&probs.slice(0..drafted))?;
                        let trunc = spec_guard_trunc(&pw, opts.pmin);
                        if trunc < drafted {
                            report.guard_stops += 1;
                        }
                        trunc
                    } else {
                        drafted
                    };
                    for &r in raw.iter().take(trunc) {
                        drafts.push(self.draft_token(r)?);
                    }
                }
            } else {
                let (d1, c1) = self.draft_row_argmax(de, &tip_logits, 0, guard)?;
                if !(guard && c1 < opts.pmin) {
                    drafts.push(d1);
                    if tracing {
                        chain_rows_h
                            .push(de.dtoh_view(&tip_logits.slice(0..self.draft_logits_width()))?);
                    }
                } else {
                    report.guard_stops += 1;
                }
                let mut prev_logits = tip_logits;
                let mut prev_carrier = tip_carrier;
                let mut prev_rows = tip_rows;
                while !drafts.is_empty() && drafts.len() < k_round {
                    if tracing {
                        seeds_h.push(de.dtoh_view(
                            &prev_carrier.slice((prev_rows - 1) * wide_w..prev_rows * wide_w),
                        )?);
                    }
                    let lastd = *drafts.last().expect("non-empty");
                    let (l2, c2) = self.mtp_draft_forward(
                        de,
                        &[lastd],
                        &prev_carrier,
                        prev_rows - 1,
                        dstate,
                        1,
                        false,
                    )?;
                    self.mtp_recycle(dstate, prev_logits, prev_carrier);
                    prev_logits = l2;
                    prev_carrier = c2;
                    prev_rows = 1;
                    let (dn, cn) = self.draft_row_argmax(de, &prev_logits, 0, guard)?;
                    if guard && cn < opts.pmin {
                        report.guard_stops += 1;
                        break;
                    }
                    drafts.push(dn);
                    if tracing {
                        chain_rows_h
                            .push(de.dtoh_view(&prev_logits.slice(0..self.draft_logits_width()))?);
                    }
                }
                self.mtp_recycle(dstate, prev_logits, prev_carrier);
            }
            let chain_ms = t_draft.elapsed().as_secs_f64() * 1e3;
            report.chain_ms += chain_ms;
            report.draft_ms += chain_ms;

            // ---- verify chunk [tip, d1..] at base m (t == 1 on a zero-draft round —
            // a plain decode step that still commits one token).
            let t_ver = Instant::now();
            let mut chunk = Vec::with_capacity(drafts.len() + 1);
            chunk.push(tip);
            chunk.extend_from_slice(&drafts);
            let tlen = chunk.len();
            let host_logits = self.forward(e, &chunk, state, None)?;
            // Deferred seam: the t == 1 zero-draft verify also commits through the
            // device argmax (want_argmax_t1) — no [1, vocab] row + host scan.
            let targets: Vec<u32> = if greedy && !tracing && (tlen > 1 || deferred) {
                self.verify_argmax_rows(state)?.to_vec()
            } else if greedy {
                (0..tlen)
                    .map(|row| host_argmax(&host_logits[row * vocab..(row + 1) * vocab]) as u32)
                    .collect()
            } else {
                let cfg = sampler.as_ref().expect("sampled mode");
                (0..tlen)
                    .map(|row| {
                        sample_row(cfg, &mut rng, &host_logits[row * vocab..(row + 1) * vocab])
                    })
                    .collect()
            };
            report.verify_ms += t_ver.elapsed().as_secs_f64() * 1e3;
            if targets.len() != tlen {
                return Err("qwen4exp_gpu: verify produced the wrong row count".into());
            }

            // ---- greedy accept walk (exact match to the target row).
            let mut a = 0usize;
            while a < drafts.len() && drafts[a] == targets[a] {
                a += 1;
            }
            report.rounds += 1;
            report.drafted += drafts.len() as u64;
            report.accepted += a as u64;
            report.accept_hist[a] += 1;
            if drafts.is_empty() {
                report.zero_draft_rounds += 1;
            }
            report.tokens.extend_from_slice(&targets[0..=a]);

            // ---- trace record (fork margins from the stashed rows; carrier drift vs
            // the verify chunk's TRUE wide rows).
            if let Some(tr) = trace.as_deref_mut() {
                let mut rec = SpecTraceRound {
                    round: round_idx,
                    gen_pos: report.tokens.len() - (a + 1),
                    base: m,
                    k: drafts.len(),
                    a,
                    drafts: drafts.clone(),
                    targets: targets.clone(),
                    draft_top1: f32::NAN,
                    draft_top2: f32::NAN,
                    draft_tgt_logit: f32::NAN,
                    draft_tgt_rank: 0,
                    target_top1: f32::NAN,
                    target_top2: f32::NAN,
                    target_draft_logit: f32::NAN,
                    target_entropy: 0.0,
                    carrier_rel_l2: Vec::new(),
                    carrier_cos: Vec::new(),
                };
                if a < drafts.len() {
                    let drow = &chain_rows_h[a];
                    let trow = &host_logits[a * vocab..(a + 1) * vocab];
                    let tgt = targets[a] as usize;
                    let dtok = drafts[a] as usize;
                    let (mut d1v, mut d2v) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    let mut rank = 0usize;
                    let dt = drow[tgt];
                    for &v in drow.iter() {
                        if v > d1v {
                            d2v = d1v;
                            d1v = v;
                        } else if v > d2v {
                            d2v = v;
                        }
                        if v > dt {
                            rank += 1;
                        }
                    }
                    let (mut t1v, mut t2v) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    for &v in trow.iter() {
                        if v > t1v {
                            t2v = t1v;
                            t1v = v;
                        } else if v > t2v {
                            t2v = v;
                        }
                    }
                    // Softmax entropy of the target row (nats), f64 accumulation.
                    let mx = t1v as f64;
                    let mut z = 0.0f64;
                    let mut sxl = 0.0f64;
                    for &v in trow.iter() {
                        let ev = ((v as f64) - mx).exp();
                        z += ev;
                        sxl += ev * ((v as f64) - mx);
                    }
                    rec.draft_top1 = d1v;
                    rec.draft_top2 = d2v;
                    rec.draft_tgt_logit = dt;
                    rec.draft_tgt_rank = rank;
                    rec.target_top1 = t1v;
                    rec.target_top2 = t2v;
                    rec.target_draft_logit = trow[dtok];
                    rec.target_entropy = z.ln() - sxl / z;
                }
                let v = state.verify.as_ref().expect("armed above");
                for (j, seed) in seeds_h.iter().enumerate() {
                    let slot = (m + j) % ring;
                    let truth = e.dtoh_view(&v.wide.slice(slot * wide_w..(slot + 1) * wide_w))?;
                    let mut dd = 0.0f64;
                    let mut tt = 0.0f64;
                    let mut st = 0.0f64;
                    let mut ss = 0.0f64;
                    for (&s, &t) in seed.iter().zip(truth.iter()) {
                        let (s, t) = (s as f64, t as f64);
                        dd += (s - t) * (s - t);
                        tt += t * t;
                        st += s * t;
                        ss += s * s;
                    }
                    rec.carrier_rel_l2
                        .push((dd.sqrt() / tt.sqrt().max(1e-30)) as f32);
                    rec.carrier_cos
                        .push((st / (ss.sqrt() * tt.sqrt()).max(1e-30)) as f32);
                }
                tr.push(rec);
            }

            // ---- rewind trunk to the accepted rows; draft catch-up replay.
            if tlen > 1 {
                self.verify_rewind(e, state, a + 1)?;
            }
            self.mtp_rewind(dstate, m)?;
            let t_draft2 = Instant::now();
            let x_next = targets[a];
            let mut replay: Vec<u32> = drafts[0..a].to_vec();
            replay.push(x_next);
            if dev1 {
                let v = state.verify.as_mut().expect("armed above");
                let VerifyStash {
                    wide, wide_dev1, ..
                } = v;
                let mirror = wide_dev1.as_mut().expect("allocated above");
                for (slot, len) in ring_pieces(ring, m, replay.len()) {
                    report.cross_ms += cross_wide_rows(e, de, wide, mirror, slot, len, wide_w)?;
                }
                report.cross_bytes += (replay.len() * wide_w * 4) as u64;
            }
            let (l, c, last_len) = {
                let v = state.verify.as_ref().expect("armed above");
                let seed: &CudaSlice<f32> = v.wide_dev1.as_ref().unwrap_or(&v.wide);
                self.draft_consume_ring(de, &replay, dev_embed, seed, ring, m, dstate)?
            };
            tip_logits = l;
            tip_carrier = c;
            tip_rows = last_len;
            dstate.committed = dstate.rows;
            let replay_ms = t_draft2.elapsed().as_secs_f64() * 1e3;
            report.replay_ms += replay_ms;
            report.draft_ms += replay_ms;
            m += a + 1;
            tip = x_next;
            report
                .round_wall
                .push((report.tokens.len(), t_total.elapsed().as_secs_f64() * 1e3));

            // ---- bounded admission updates (both decay-only within the round budget).
            if let Some(lo) = opts.adapt_k_lo {
                k_next = (a + 1).clamp(lo.max(1), k);
            }
            if let Some(cfg) = opts.dynk {
                window.push(a);
                if window.len() >= cfg.window.max(1) {
                    let mean = window.iter().sum::<usize>() as f64 / window.len() as f64;
                    if mean < cfg.thr {
                        let new_k = k_cur.saturating_sub(1).max(cfg.k_floor);
                        if new_k < k_cur {
                            k_cur = new_k;
                            report.k_decays.push((round_idx, k_cur));
                            if k_cur == 0 {
                                report.spec_off_at = Some(report.tokens.len());
                            }
                        }
                    }
                    window.clear();
                }
            }
            round_idx += 1;
        }
        report.tokens.truncate(max_new);
        report.total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok(report)
    }
}

// ---------------------------------------------------------------- checkpoint loading
//
// The pack/plan/contract walk over an HF safetensors dir. The loader PROBES the artifact
// for its routed-expert dialect (ExpertDialect: the BF16 export's fused 3D banks, or the
// NVFP4 mint's per-expert modelopt projections — census receipt
// research/qwen4exp-bringup-20260829/raw/nvfp4-census-names.tsv) and binds through the
// pack's dialect contract. Trunk + globals materialize into reference-layout weights; the
// n-gram table stays host-resident (sharded or the mint's single tensor); expert banks
// admit BF16 (dequantized) or modelopt NVFP4 (as-stored device residency). `input_scale`
// (modelopt static activation scale) is contract-declared as an auxiliary, VALIDATED here
// (F32 scalar) and deliberately UNUSED: the eager arm is W4A16-class (weights dequantize
// to f32, activations stay f32), so the scale has no consumer until the W4A4 kernel lane
// quantizes activations — the dsv4 precedent ("W4A8 activation scale, unused for decode").
// MTP and vision tensors are validated owners but not materialized — the eager arm
// executes neither (module header).

/// One expert bank tensor (one PROJECTION: gate, up, or down), assembled across experts.
enum BankTensorSrc {
    /// Dequantized f32, logical [n_expert, out_f, in_f].
    F32(Vec<f32>),
    /// modelopt NVFP4: e2m1 codes [E, out, in/2], e4m3 scales [E, out, in/16],
    /// per-expert finite macro scales (the real mint's are amax-derived non-pow2), and
    /// the projection's STATIC ACTIVATION scale — the max of the per-expert
    /// `input_scale` siblings. RECORDED-ONLY by owner order (2026-08-30): activation
    /// quantization is retired as a serving lever (it measurably moved decode argmax —
    /// perf22 seam-gate receipt, PROFILE-4 §W4A4); no compute path consumes this value,
    /// and no future lane re-proposes consuming it without a fresh owner ruling.
    Nvfp4 {
        codes: Vec<u8>,
        scales: Vec<u8>,
        macros: Vec<f32>,
        act_scale: Option<f32>,
    },
    /// Raw bf16 bytes at the logical shape [n_expert, out_f, in_f] — kept when
    /// `LoadOptions::host_bf16_banks` asks for the host-resident gate residency.
    Bf16(Vec<u8>),
}

struct BankSrc {
    gate: BankTensorSrc, // logical [E, ff, H]
    up: BankTensorSrc,   // logical [E, ff, H]
    down: BankTensorSrc, // logical [E, H, ff]
    n_expert: usize,
    ff: usize,
    hidden: usize,
}

/// One fused bank tensor's read address: the artifact name plus the contract shape the
/// walk already validated it against ([E, out_f, in_f], logical).
struct FusedTensorPlan {
    name: String,
    shape: [usize; 3],
}

/// One PER-EXPERT projection's read addresses, in EXPERT ORDER 0..E. The walk builds this
/// from a numerically-keyed map and checks contiguity there, so this vector's index IS the
/// expert id — the lexicographic-arrival trap (`experts.10` before `experts.2`) is already
/// absorbed before anything is read.
struct PerExpertPlan {
    names: Vec<String>, // expert order 0..E
    out_f: usize,
    in_f: usize,
    quant: memra_gguf::tensor_contract::QuantConstraint,
}

enum BankPlanSrc {
    /// FusedBanks dialect: the fused [E, 2ff, H] gate_up tensor + the [E, H, ff] down.
    Fused {
        gate_up: FusedTensorPlan,
        down: FusedTensorPlan,
        /// Residency decided at WALK time, exactly as before
        /// (`LoadOptions::host_bf16_banks`, or an MTP bank at index >= n_trunk): a bf16
        /// bank keeps raw bytes instead of dequantizing to f32. Fused-only — the
        /// per-expert modelopt rows are F32 or NVFP4 by geometry, never a bf16 arm.
        keep_bf16: bool,
    },
    /// PerExpertModelopt dialect: one name list per projection.
    PerExpert {
        gate: PerExpertPlan,
        up: PerExpertPlan,
        down: PerExpertPlan,
    },
}

/// WHERE one layer's expert bank lives in the artifact and HOW to bind it — everything
/// `BankSrc` needs except the bytes.
///
/// This is the streaming seam. The walk validates names/shapes/dtypes/geometry/expert
/// contiguity and records this plan; the bytes are read one LAYER at a time inside the
/// consuming loop (`from_loaded_checkpoint_dual`, `build_tp2_shard`,
/// `into_reference_weights`) straight off the safetensors mmap, uploaded, and dropped.
/// Pre-materializing all 48 layers cost the whole artifact in host anon memory at once
/// (~72 GB of banks on top of the 102 GB n-gram table and ~20 GB of trunk f32), which
/// OOM-killed the real gate at 179.7 GB anon-RSS on a 180 GB-RAM box — the cheapest
/// 2-card class. Nothing about the BYTES changes: the same
/// `read_bank_tensor`/`read_per_expert`/`assemble_per_expert_bank`/`split_fused_gate_up`
/// chain runs on the same file offsets in the same expert order, just later.
struct BankPlan {
    n_expert: usize,
    ff: usize,
    hidden: usize,
    src: BankPlanSrc,
}

impl BankPlan {
    /// Read + assemble THIS layer's bank off the mmap. Peak host cost is one layer's bank
    /// (~1.5 GB on the real mint), not the artifact.
    fn read(&self, model: &memra_gguf::safetensors::StModel) -> Res<BankSrc> {
        let (gate, up, down) = match &self.src {
            BankPlanSrc::Fused {
                gate_up,
                down,
                keep_bf16,
            } => {
                let fused = read_bank_tensor(
                    model,
                    &gate_up.name,
                    gate_up.shape[0],
                    gate_up.shape[1],
                    gate_up.shape[2],
                    *keep_bf16,
                )?;
                let (gate, up) = split_fused_gate_up(fused, self.n_expert, self.ff, self.hidden)?;
                let down = read_bank_tensor(
                    model,
                    &down.name,
                    down.shape[0],
                    down.shape[1],
                    down.shape[2],
                    *keep_bf16,
                )?;
                (gate, up, down)
            }
            BankPlanSrc::PerExpert { gate, up, down } => (
                read_per_expert_bank(model, gate)?,
                read_per_expert_bank(model, up)?,
                read_per_expert_bank(model, down)?,
            ),
        };
        Ok(BankSrc {
            gate,
            up,
            down,
            n_expert: self.n_expert,
            ff: self.ff,
            hidden: self.hidden,
        })
    }
}

/// WALK-time refusal for a bank tensor whose PAYLOAD is read later: the name must exist in
/// the census and carry a dtype the bank readers admit.
///
/// `StModel::info` is header-only, so this faults no weight page and costs no host memory.
/// It keeps the contract walk's "every declared name exists" property — a mint missing a
/// projection is refused before the 102 GB table is allocated and before one byte reaches
/// the device. Shape, scale siblings, macro finiteness and `input_scale` validity are
/// checked by `read_bank_tensor`/`read_per_expert` when the layer is read, which is still
/// load time (before any forward), just per layer instead of all at once.
fn check_bank_header(model: &memra_gguf::safetensors::StModel, name: &str) -> Res<()> {
    let info = model
        .info(name)
        .ok_or_else(|| format!("qwen4exp_gpu: checkpoint is missing {name}"))?;
    match info.dtype.as_str() {
        "BF16" | "F32" | "U8" => Ok(()),
        other => Err(format!("qwen4exp_gpu: {name} bank dtype {other} unsupported").into()),
    }
}

/// Read one per-expert projection in expert order and stack it — the deferred half of the
/// old walk's `read_per_expert` + `assemble_per_expert_bank` pair, byte-for-byte.
fn read_per_expert_bank(
    model: &memra_gguf::safetensors::StModel,
    plan: &PerExpertPlan,
) -> Res<BankTensorSrc> {
    let mut experts = Vec::with_capacity(plan.names.len());
    for name in &plan.names {
        experts.push(read_per_expert(
            model, name, plan.out_f, plan.in_f, plan.quant,
        )?);
    }
    assemble_per_expert_bank(experts)
}

/// A checkpoint materialized through the pack contract: reference-layout weights for the
/// trunk + globals (effective norms — the (1+w) fold applied per the module-header rule),
/// plus the table carrier that stays out of `ReferenceWeights` and the LAZY bank plans.
///
/// The open safetensors mmap is part of the value: expert banks are read from it per layer
/// at consume time (see `BankPlan`). It stays mapped until the checkpoint is dropped, so a
/// consumer must not outlive it — every consumer here is a constructor that finishes
/// uploading before returning.
pub struct LoadedCheckpoint {
    pub plan: ModelPlan,
    pub weights: ReferenceWeights,
    model: memra_gguf::safetensors::StModel,
    bank_plans: std::collections::BTreeMap<u32, BankPlan>,
    tables: std::collections::BTreeMap<u32, Vec<u8>>, // bf16 bytes, [rows, head_dim]
}

/// (1+w) fold rule for checkpoint norm rows — the qwen35 receipt (hf_mapping.rs,
/// qwen.py:302-303): every `*norm*.weight` EXCEPT `linear_attn.norm` (RMSNormGated binds
/// raw weights; SEMANTICS.md §GDN keeps the qwen3_5 GDN program). VERIFY vs the goldens
/// lane for the indexer layernorms (assumed the family (1+w) class — the zero-init
/// receipt, modular L860).
fn norm_fold_add_one(name: &str) -> bool {
    name.contains("norm") && name.ends_with(".weight") && !name.ends_with("linear_attn.norm.weight")
}

/// The QSA indexer's q/k layernorm rows — the SEMANTICS.md VERIFY subject. The default
/// fold treats them as family (1+w); `LoadOptions::indexer_norm_raw` binds them raw so
/// the real-checkpoint per-layer gate can measure both arms and settle the question.
fn indexer_layernorm(name: &str) -> bool {
    name.contains(".indexer.")
        && (name.ends_with("q_layernorm.weight") || name.ends_with("k_layernorm.weight"))
}

/// Real-checkpoint loader knobs (defaults = the tiny-gate behavior).
#[derive(Default, Clone, Copy)]
pub struct LoadOptions {
    /// Keep BF16 expert banks HOST-resident (raw bf16) and upload+upcast per ROUTED
    /// expert at forward time. Gate-mode residency for artifacts whose f32 banks
    /// exceed device memory; value chain identical to the f32 device arm (bf16→f32
    /// is exact). Never a serving configuration.
    pub host_bf16_banks: bool,
    /// Bind the indexer q/k layernorms RAW (skip the (1+w) fold) — the two-arm probe
    /// for the SEMANTICS.md VERIFY marker. Default keeps the family fold.
    pub indexer_norm_raw: bool,
    /// Materialize the mtp.* namespace (the NextN draft block) — the mtp-spec lane.
    /// The MTP expert bank keeps its raw BF16 bytes at read time and goes DEVICE
    /// bf16-resident at build (`BankHalf::DeviceBf16`, ~5 GB beside the NVFP4 trunk).
    /// Default OFF: the plain eager arm executes no draft.
    pub load_mtp: bool,
}

fn bridge_transform(
    transform: memra_gguf::tensor_contract::TensorTransform,
) -> Res<memra_gguf::hf_mapping::TransformKind> {
    use memra_gguf::hf_mapping::TransformKind as K;
    use memra_gguf::tensor_contract::TensorTransform as T;
    Ok(match transform {
        T::Identity => K::Identity,
        T::NormAddOne => K::NormPlusOne,
        T::QkvVReorderRows => K::QkvVReorderRows,
        T::ZReorderRows => K::ZReorderRows,
        T::AbReorderRows => K::AbReorderRows,
        T::NegExpReorderHeads => K::NegExpReorderHeads,
        T::ReorderHeads => K::ReorderHeads,
        T::Conv1dSqueezeReorder => K::Conv1dSqueezeReorder,
        T::OutReorderColumns => K::OutReorderCols,
        other => return Err(format!("qwen4exp_gpu: unsupported transform {other:?}").into()),
    })
}

fn dequant_float(
    name: &str,
    info: &memra_gguf::safetensors::StInfo,
    bytes: &[u8],
) -> Res<Vec<f32>> {
    let elements: usize = info.shape.iter().map(|&d| d as usize).product();
    match info.dtype.as_str() {
        "BF16" | "F32" => Ok(memra_gguf::dequant::dequantize(
            info.ggml_type()
                .map_err(|error| format!("qwen4exp_gpu: {name}: {error}"))?,
            bytes,
            elements,
        )),
        other => Err(format!("qwen4exp_gpu: {name} has unsupported float dtype {other}").into()),
    }
}

fn read_i64(name: &str, info: &memra_gguf::safetensors::StInfo, bytes: &[u8]) -> Res<Vec<i64>> {
    if info.dtype != "I64" {
        return Err(format!("qwen4exp_gpu: {name} must be I64, got {}", info.dtype).into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

/// Macro-scale validation. The dsv4 pow2 law does NOT apply here: this module's dequant
/// chain applies the macro post-upcast in f32 (`dequant_nvfp4_expert_f32`), which is
/// exact-then-single-rounding for ANY finite positive macro — and the real qwen4_exp
/// mint ships modelopt's amax-derived NON-pow2 `weight_scale_2` (first value refused by
/// the inherited pow2 assert on the fleet box, 2026-08-29: 5.9945243e-5 on
/// layers.0.mlp.experts.0.down_proj). Refusal is reserved for values that poison the
/// arithmetic outright.
fn validate_macro(stem: &str, value: f32) -> Res<()> {
    if !(value.is_finite() && value > 0.0) {
        return Err(format!(
            "qwen4exp_gpu: {stem}.weight_scale_2 carries a non-finite/non-positive \
             macro {value}"
        )
        .into());
    }
    Ok(())
}

/// Read one STACKED expert bank (FusedBanks dialect): BF16 at the declared logical shape,
/// or the modelopt-NVFP4 stacked triplet whose validation mirrors
/// `find_nvfp4_stacked_native` (source.rs): U8 codes [E, out, in/2] + F8_E4M3
/// `weight_scale` [E, out, in/16] + optional F32 `weight_scale_2` [E] (absent -> 1.0).
fn read_bank_tensor(
    model: &memra_gguf::safetensors::StModel,
    name: &str,
    n_expert: usize,
    out_f: usize,
    in_f: usize,
    host_bf16: bool,
) -> Res<BankTensorSrc> {
    let (info, bytes) = model
        .raw(name)
        .ok_or_else(|| format!("qwen4exp_gpu: checkpoint is missing {name}"))?;
    match info.dtype.as_str() {
        "BF16" | "F32" => {
            if info.shape != [n_expert as u64, out_f as u64, in_f as u64] {
                return Err(format!("qwen4exp_gpu: {name} bank shape mismatch").into());
            }
            if host_bf16 && info.dtype == "BF16" {
                if bytes.len() != n_expert * out_f * in_f * 2 {
                    return Err(format!("qwen4exp_gpu: {name} bank byte-length mismatch").into());
                }
                return Ok(BankTensorSrc::Bf16(bytes.to_vec()));
            }
            Ok(BankTensorSrc::F32(dequant_float(name, info, bytes)?))
        }
        "U8" => {
            if in_f % 16 != 0
                || info.shape != [n_expert as u64, out_f as u64, (in_f / 2) as u64]
                || bytes.len() != n_expert * out_f * in_f / 2
            {
                return Err(format!("qwen4exp_gpu: {name} NVFP4 code shape mismatch").into());
            }
            let stem = name.strip_suffix(".weight").unwrap_or(name);
            let scale_name = format!("{stem}.weight_scale");
            let (scale_info, scale_bytes) = model
                .raw(&scale_name)
                .ok_or_else(|| format!("qwen4exp_gpu: missing {scale_name}"))?;
            if scale_info.dtype != "F8_E4M3"
                || scale_info.shape != [n_expert as u64, out_f as u64, (in_f / 16) as u64]
                || scale_bytes.len() != n_expert * out_f * in_f / 16
            {
                return Err(format!("qwen4exp_gpu: {scale_name} shape mismatch").into());
            }
            let macros = match model.raw(&format!("{stem}.weight_scale_2")) {
                Some((macro_info, macro_bytes))
                    if macro_info.dtype == "F32" && macro_bytes.len() == n_expert * 4 =>
                {
                    macro_bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect()
                }
                None => vec![1.0; n_expert],
                _ => return Err(format!("qwen4exp_gpu: {stem}.weight_scale_2 malformed").into()),
            };
            for &m in &macros {
                validate_macro(stem, m)?;
            }
            // Optional stacked input_scale [E] (the per-expert mint carries scalars via
            // the PerExpertModelopt path; a stacked artifact may carry the vector) —
            // reduced to the per-layer max for the W4A4 activation quantization.
            let act_scale = match model.raw(&format!("{stem}.input_scale")) {
                Some((is_info, is_bytes))
                    if is_info.dtype == "F32" && is_bytes.len() == n_expert * 4 =>
                {
                    let mut mx = 0.0f32;
                    for chunk in is_bytes.chunks_exact(4) {
                        let v = f32::from_le_bytes(chunk.try_into().unwrap());
                        if !(v.is_finite() && v > 0.0) {
                            return Err(format!(
                                "qwen4exp_gpu: {stem}.input_scale carries a non-finite/\
                                 non-positive value {v}"
                            )
                            .into());
                        }
                        mx = mx.max(v);
                    }
                    Some(mx)
                }
                Some(_) => {
                    return Err(format!("qwen4exp_gpu: {stem}.input_scale malformed").into());
                }
                None => None,
            };
            Ok(BankTensorSrc::Nvfp4 {
                codes: bytes.to_vec(),
                scales: scale_bytes.to_vec(),
                macros,
                act_scale,
            })
        }
        other => Err(format!("qwen4exp_gpu: {name} bank dtype {other} unsupported").into()),
    }
}

/// Split a FUSED gate_up source ([E, 2ff, H], gate rows first per expert) into per-
/// projection gate/up sources. F32 splits data rows; NVFP4 splits code/scale byte rows
/// (row-granular, byte-clean) and duplicates the per-expert macro to both halves.
fn split_fused_gate_up(
    fused: BankTensorSrc,
    n_expert: usize,
    ff: usize,
    hidden: usize,
) -> Res<(BankTensorSrc, BankTensorSrc)> {
    match fused {
        BankTensorSrc::F32(data) => {
            if data.len() != n_expert * 2 * ff * hidden {
                return Err("qwen4exp_gpu: fused gate_up bank size mismatch".into());
            }
            let mut gate = Vec::with_capacity(n_expert * ff * hidden);
            let mut up = Vec::with_capacity(n_expert * ff * hidden);
            for expert in 0..n_expert {
                let base = expert * 2 * ff * hidden;
                gate.extend_from_slice(&data[base..base + ff * hidden]);
                up.extend_from_slice(&data[base + ff * hidden..base + 2 * ff * hidden]);
            }
            Ok((BankTensorSrc::F32(gate), BankTensorSrc::F32(up)))
        }
        BankTensorSrc::Bf16(bytes) => {
            let row = hidden * 2; // bf16 bytes per fused row
            if bytes.len() != n_expert * 2 * ff * row {
                return Err("qwen4exp_gpu: fused bf16 gate_up bank size mismatch".into());
            }
            let mut gate = Vec::with_capacity(n_expert * ff * row);
            let mut up = Vec::with_capacity(n_expert * ff * row);
            for expert in 0..n_expert {
                let base = expert * 2 * ff * row;
                gate.extend_from_slice(&bytes[base..base + ff * row]);
                up.extend_from_slice(&bytes[base + ff * row..base + 2 * ff * row]);
            }
            Ok((BankTensorSrc::Bf16(gate), BankTensorSrc::Bf16(up)))
        }
        BankTensorSrc::Nvfp4 {
            codes,
            scales,
            macros,
            act_scale,
        } => {
            let code_row = hidden / 2;
            let scale_row = hidden / 16;
            let mut gate_codes = Vec::with_capacity(n_expert * ff * code_row);
            let mut up_codes = Vec::with_capacity(n_expert * ff * code_row);
            let mut gate_scales = Vec::with_capacity(n_expert * ff * scale_row);
            let mut up_scales = Vec::with_capacity(n_expert * ff * scale_row);
            for expert in 0..n_expert {
                let cbase = expert * 2 * ff * code_row;
                gate_codes.extend_from_slice(&codes[cbase..cbase + ff * code_row]);
                up_codes
                    .extend_from_slice(&codes[cbase + ff * code_row..cbase + 2 * ff * code_row]);
                let sbase = expert * 2 * ff * scale_row;
                gate_scales.extend_from_slice(&scales[sbase..sbase + ff * scale_row]);
                up_scales
                    .extend_from_slice(&scales[sbase + ff * scale_row..sbase + 2 * ff * scale_row]);
            }
            Ok((
                BankTensorSrc::Nvfp4 {
                    codes: gate_codes,
                    scales: gate_scales,
                    macros: macros.clone(),
                    act_scale,
                },
                BankTensorSrc::Nvfp4 {
                    codes: up_codes,
                    scales: up_scales,
                    macros,
                    act_scale,
                },
            ))
        }
    }
}

/// One PER-EXPERT projection (PerExpertModelopt dialect): the modelopt sibling schema
/// (`nvfp4_quant`'s modelopt arm, source.rs — weight U8 [out, in/2] + weight_scale +
/// scalar weight_scale_2), or a plain BF16 row where geometry forbids per-16 groups.
/// `input_scale` is validated (F32 scalar) and dropped — see the section header.
enum PerExpertSrc {
    F32(Vec<f32>),
    Nvfp4 {
        codes: Vec<u8>,
        scales: Vec<u8>,
        macro_scale: f32,
        input_scale: Option<f32>,
    },
}

fn read_per_expert(
    model: &memra_gguf::safetensors::StModel,
    name: &str,
    out_f: usize,
    in_f: usize,
    quant: memra_gguf::tensor_contract::QuantConstraint,
) -> Res<PerExpertSrc> {
    use memra_gguf::tensor_contract::QuantConstraint;
    let (info, bytes) = model
        .raw(name)
        .ok_or_else(|| format!("qwen4exp_gpu: checkpoint is missing {name}"))?;
    match quant {
        QuantConstraint::ExactFloat(_) => {
            if info.shape != [out_f as u64, in_f as u64] {
                return Err(format!("qwen4exp_gpu: {name} shape mismatch").into());
            }
            Ok(PerExpertSrc::F32(dequant_float(name, info, bytes)?))
        }
        QuantConstraint::Nvfp4 => {
            if info.dtype != "U8"
                || in_f % 16 != 0
                || info.shape != [out_f as u64, (in_f / 2) as u64]
                || bytes.len() != out_f * in_f / 2
            {
                return Err(format!("qwen4exp_gpu: {name} NVFP4 code shape mismatch").into());
            }
            let stem = name.strip_suffix(".weight").unwrap_or(name);
            let (scale_info, scale_bytes) = model
                .raw(&format!("{stem}.weight_scale"))
                .ok_or_else(|| format!("qwen4exp_gpu: missing {stem}.weight_scale"))?;
            if scale_info.dtype != "F8_E4M3"
                || scale_info.shape != [out_f as u64, (in_f / 16) as u64]
                || scale_bytes.len() != out_f * in_f / 16
            {
                return Err(format!("qwen4exp_gpu: {stem}.weight_scale shape mismatch").into());
            }
            let macro_scale = match model.raw(&format!("{stem}.weight_scale_2")) {
                Some((macro_info, macro_bytes))
                    if macro_info.dtype == "F32" && macro_bytes.len() == 4 =>
                {
                    f32::from_le_bytes(macro_bytes.try_into().unwrap())
                }
                None => 1.0,
                _ => return Err(format!("qwen4exp_gpu: {stem}.weight_scale_2 malformed").into()),
            };
            validate_macro(stem, macro_scale)?;
            // input_scale: modelopt's STATIC ACTIVATION scale (= calibrated amax /
            // (448*6)) — validated AND consumed since round 4: the W4A4 expert path
            // quantizes activations against the per-layer max of these (see
            // BankTensorSrc::Nvfp4::act_scale).
            let input_scale = match model.raw(&format!("{stem}.input_scale")) {
                Some((input_info, input_bytes)) => {
                    if input_info.dtype != "F32" || input_bytes.len() != 4 {
                        return Err(format!("qwen4exp_gpu: {stem}.input_scale malformed").into());
                    }
                    let v = f32::from_le_bytes(input_bytes.try_into().unwrap());
                    if !(v.is_finite() && v > 0.0) {
                        return Err(format!(
                            "qwen4exp_gpu: {stem}.input_scale carries a non-finite/non-positive \
                             value {v}"
                        )
                        .into());
                    }
                    Some(v)
                }
                None => None,
            };
            Ok(PerExpertSrc::Nvfp4 {
                codes: bytes.to_vec(),
                scales: scale_bytes.to_vec(),
                macro_scale,
                input_scale,
            })
        }
        other => Err(format!("qwen4exp_gpu: per-expert quant {other:?} unsupported").into()),
    }
}

/// Concatenate per-expert sources (expert order 0..E) into one stacked BankTensorSrc.
/// Kinds must be uniform across a projection (the census derives them per geometry).
fn assemble_per_expert_bank(experts: Vec<PerExpertSrc>) -> Res<BankTensorSrc> {
    let mut f32_data: Vec<f32> = Vec::new();
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<u8> = Vec::new();
    let mut macros: Vec<f32> = Vec::new();
    let mut act_scale: Option<f32> = None;
    let mut act_scale_complete = true;
    let mut kinds = (false, false);
    for expert in experts {
        match expert {
            PerExpertSrc::F32(data) => {
                kinds.0 = true;
                f32_data.extend_from_slice(&data);
            }
            PerExpertSrc::Nvfp4 {
                codes: c,
                scales: s,
                macro_scale,
                input_scale,
            } => {
                kinds.1 = true;
                codes.extend_from_slice(&c);
                scales.extend_from_slice(&s);
                macros.push(macro_scale);
                match input_scale {
                    Some(v) => act_scale = Some(act_scale.map_or(v, |a: f32| a.max(v))),
                    None => act_scale_complete = false,
                }
            }
        }
    }
    match kinds {
        (true, false) => Ok(BankTensorSrc::F32(f32_data)),
        (false, true) => Ok(BankTensorSrc::Nvfp4 {
            codes,
            scales,
            macros,
            act_scale: if act_scale_complete { act_scale } else { None },
        }),
        _ => Err("qwen4exp_gpu: mixed per-expert kinds within one projection".into()),
    }
}

/// Trunk layer index of a family-keyed requirement (`trunk.layers.{il}. ...`).
fn family_layer_index(key: &str) -> Option<u32> {
    key.strip_prefix("trunk.layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Walk the pack contract over an HF safetensors dir and materialize the eager arm's
/// weight set. The expert dialect is PROBED from the artifact (per-expert names present
/// => the NVFP4 mint layout). Fails loudly on any missing name, shape/dtype mismatch, or
/// unsupported transform — nothing is skipped silently except the declared MTP/vision
/// owners.
/// Resolve a layer plan by GLOBAL index: trunk layers [0, n_trunk), then MTP blocks at
/// n_trunk + depth (the pack's mtp.layers.* mapping).
fn plan_layer_at(plan: &ModelPlan, index: u32) -> Option<&memra_gguf::model_plan::LayerPlan> {
    let n_trunk = plan.layers.len() as u32;
    if index < n_trunk {
        plan.layers.get(index as usize)
    } else {
        plan.mtp_blocks
            .iter()
            .find(|block| block.layer.index == index)
            .map(|block| &block.layer)
    }
}

pub fn read_checkpoint(dir: &std::path::Path) -> Res<LoadedCheckpoint> {
    read_checkpoint_with(dir, LoadOptions::default())
}

/// `read_checkpoint` with real-checkpoint loader knobs (`LoadOptions`).
pub fn read_checkpoint_with(dir: &std::path::Path, opts: LoadOptions) -> Res<LoadedCheckpoint> {
    use memra_gguf::model_packs::qwen4_exp::{ExpertDialect, tensor_contract_for};
    use memra_gguf::tensor_contract::{TensorMatch, TensorOwner};
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let cfg =
        memra_gguf::config::ModelConfig::from_hf(&memra_gguf::config::HfConfig::parse(&config));
    let pack = memra_gguf::model_packs::for_config(&cfg)
        .ok_or("qwen4exp_gpu: no model pack matches this config")?;
    if pack.family != "qwen4_exp" {
        return Err(format!("qwen4exp_gpu: config resolves to pack {}", pack.family).into());
    }
    let plan = pack.compile_plan(&cfg)?;
    let model = memra_gguf::safetensors::StModel::open(dir)?;
    // Dialect probe: layer 0 is always MoE; the mint un-fuses its experts.
    let dialect = if model
        .raw("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
        .is_some()
    {
        ExpertDialect::PerExpertModelopt
    } else {
        ExpertDialect::FusedBanks
    };
    let contract = tensor_contract_for(&cfg, &plan, dialect)?;

    let mut weights = ReferenceWeights::new();
    let mut gate_up_banks: std::collections::BTreeMap<u32, FusedTensorPlan> = Default::default();
    // Keyed by numeric expert index: the contract iterates the census BTreeMap in
    // LEXICOGRAPHIC name order (experts.10 before experts.2), so per-expert rows arrive
    // out of numeric order on any E > 9 — assembly must not assume arrival order.
    let mut per_expert: std::collections::BTreeMap<
        (u32, u8),
        std::collections::BTreeMap<
            u32,
            (
                String,
                usize,
                usize,
                memra_gguf::tensor_contract::QuantConstraint,
            ),
        >,
    > = Default::default();
    let mut down_banks: std::collections::BTreeMap<u32, FusedTensorPlan> = Default::default();
    let mut tables: std::collections::BTreeMap<u32, Vec<u8>> = Default::default();
    let n_trunk = plan.layers.len() as u32;

    for requirement in &contract.requirements {
        match requirement.owner {
            // The eager trunk executes neither; vision rows stay contract-declared for
            // the census/checkpoint-parity gates but are never materialized here. MTP
            // rows materialize when the mtp-spec lane asks (`LoadOptions::load_mtp`).
            TensorOwner::Mtp(_) if !opts.load_mtp => continue,
            TensorOwner::Vision(_) => continue,
            TensorOwner::Global | TensorOwner::Layer(_) | TensorOwner::Mtp(_) => {}
        }
        // The n-gram shard bank: one semantic tensor, `names` in shard order (pack sorts).
        if requirement.match_mode == TensorMatch::All {
            let TensorId::Family { key, .. } = &requirement.id else {
                return Err("qwen4exp_gpu: unexpected All-mode requirement".into());
            };
            let layer =
                family_layer_index(key).ok_or("qwen4exp_gpu: n-gram bank outside a trunk layer")?;
            let mut bytes = Vec::new();
            for name in &requirement.names {
                let (info, shard) = model
                    .raw(name)
                    .ok_or_else(|| format!("qwen4exp_gpu: checkpoint is missing {name}"))?;
                if info.dtype != "BF16" || info.shape != requirement.shape {
                    return Err(format!("qwen4exp_gpu: {name} shard shape/dtype mismatch").into());
                }
                bytes.extend_from_slice(shard);
            }
            tables.insert(layer, bytes);
            continue;
        }
        let name = &requirement.names[0];
        // The mint's UNSHARDED table: same Family bank id, one BF16 tensor — read raw
        // bytes (a host f32 materialization of 51B rows is not a thing).
        if let TensorId::Family { key, .. } = &requirement.id {
            if key.ends_with(".ple_embedding.ngram_embedding") {
                let layer = family_layer_index(key)
                    .ok_or("qwen4exp_gpu: n-gram table outside a trunk layer")?;
                let (info, bytes) = model
                    .raw(name)
                    .ok_or_else(|| format!("qwen4exp_gpu: checkpoint is missing {name}"))?;
                if info.dtype != "BF16" || info.shape != requirement.shape {
                    return Err(format!("qwen4exp_gpu: {name} table shape/dtype mismatch").into());
                }
                tables.insert(layer, bytes.to_vec());
                continue;
            }
        }
        // Per-expert projections (PerExpertModelopt).
        if let TensorId::Expert {
            layer,
            expert,
            tensor,
        } = requirement.id
        {
            let (out_f, in_f) = (requirement.shape[0] as usize, requirement.shape[1] as usize);
            check_bank_header(&model, name)?;
            let proj = match tensor {
                memra_gguf::tensor_contract::ExpertTensor::Gate => 0u8,
                memra_gguf::tensor_contract::ExpertTensor::Up => 1,
                memra_gguf::tensor_contract::ExpertTensor::Down => 2,
            };
            if per_expert
                .entry((layer, proj))
                .or_default()
                .insert(expert, (name.clone(), out_f, in_f, requirement.quant))
                .is_some()
            {
                return Err(format!(
                    "qwen4exp_gpu: duplicate per-expert row layer {layer} expert {expert}"
                )
                .into());
            }
            continue;
        }
        // Fused expert banks (FusedBanks) bypass ReferenceWeights (device residency).
        if let TensorId::Layer { index, tensor } = requirement.id {
            if matches!(
                tensor,
                LayerTensor::MoeExpertGateUpBank | LayerTensor::MoeExpertDownBank
            ) {
                let shape = [
                    requirement.shape[0] as usize,
                    requirement.shape[1] as usize,
                    requirement.shape[2] as usize,
                ];
                check_bank_header(&model, name)?;
                let address = FusedTensorPlan {
                    name: name.clone(),
                    shape,
                };
                if tensor == LayerTensor::MoeExpertGateUpBank {
                    gate_up_banks.insert(index, address);
                } else {
                    down_banks.insert(index, address);
                }
                continue;
            }
        }
        let (info, bytes) = model
            .raw(name)
            .ok_or_else(|| format!("qwen4exp_gpu: checkpoint is missing {name}"))?;
        if info.shape != requirement.shape {
            return Err(format!(
                "qwen4exp_gpu: {name} shape {:?} != contract {:?}",
                info.shape, requirement.shape
            )
            .into());
        }
        if info.dtype == "I64" {
            let ints = read_i64(name, info, bytes)?;
            let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
            weights.insert(
                requirement.id.clone(),
                ReferenceTensor::new_i64(shape, ints)?,
            );
            continue;
        }
        let mut data = dequant_float(name, info, bytes)?;
        if norm_fold_add_one(name) && !(opts.indexer_norm_raw && indexer_layernorm(name)) {
            for value in &mut data {
                *value += 1.0;
            }
        }
        let kind = bridge_transform(requirement.transform)?;
        let (ne_out, out_bytes) = kind.apply(&mut data, info.ne(), &cfg);
        let data: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut shape: Vec<usize> = ne_out.iter().rev().map(|&d| d as usize).collect();
        // The PLE conv ships [wide, 1, K]; the reference executor (and the depthwise
        // kernel) consume the squeezed [wide, K] form — same bytes, GDN-conv precedent.
        if name.ends_with("ple.conv1d.weight") && shape.len() == 3 && shape[1] == 1 {
            shape = vec![shape[0], shape[2]];
        }
        // shared_expert_gate ships [1, H]; the reference binds the squeezed [H] row.
        if name.ends_with("mlp.shared_expert_gate.weight") && shape.len() == 2 && shape[0] == 1 {
            shape = vec![shape[1]];
        }
        weights.insert(requirement.id.clone(), ReferenceTensor::new(shape, data)?);
    }

    let mut bank_plans = std::collections::BTreeMap::new();
    // FusedBanks: pair the fused gate_up with its down twin (trunk layers AND the MTP
    // block, whose layer plan lives at index n_trunk in plan.mtp_blocks). The fused split
    // itself happens per layer at read time (`BankPlan::read`).
    for (index, gate_up) in gate_up_banks {
        let down = down_banks
            .remove(&index)
            .ok_or_else(|| format!("qwen4exp_gpu: layer {index} has gate_up but no down bank"))?;
        let layer_plan = plan_layer_at(&plan, index)
            .ok_or_else(|| format!("qwen4exp_gpu: bank at unknown layer index {index}"))?;
        let MlpPlan::Moe(moe) = &layer_plan.mlp else {
            return Err(format!("qwen4exp_gpu: bank on non-MoE layer {index}").into());
        };
        bank_plans.insert(
            index,
            BankPlan {
                n_expert: moe.expert_count as usize,
                ff: moe.expert_intermediate_size as usize,
                hidden: plan.hidden_size as usize,
                src: BankPlanSrc::Fused {
                    gate_up,
                    down,
                    // The MTP bank (index >= n_trunk) keeps raw bf16 bytes: it goes DEVICE
                    // bf16-resident at build (never f32-expanded — 10 GB vs 5 GB).
                    keep_bf16: opts.host_bf16_banks || index >= n_trunk,
                },
            },
        );
    }
    if !down_banks.is_empty() {
        return Err("qwen4exp_gpu: down bank without a gate_up twin".into());
    }
    // PerExpertModelopt: order the per-projection name lists by expert index. The STACK
    // itself is read+concatenated per layer at consume time (`read_per_expert_bank`).
    let mut per_layer: std::collections::BTreeMap<u32, [Option<PerExpertPlan>; 3]> =
        Default::default();
    for ((layer, proj), experts) in per_expert {
        let layer_plan = plan_layer_at(&plan, layer)
            .ok_or_else(|| format!("qwen4exp_gpu: per-expert rows at unknown layer {layer}"))?;
        let MlpPlan::Moe(moe) = &layer_plan.mlp else {
            return Err(format!("qwen4exp_gpu: per-expert rows on non-MoE layer {layer}").into());
        };
        let count = moe.expert_count as usize;
        // Contiguity check: BTreeMap<u32, _> iteration is numeric order; every expert
        // index 0..E must be present exactly once.
        if experts.len() != count || experts.keys().last().copied() != Some(count as u32 - 1) {
            return Err(format!(
                "qwen4exp_gpu: layer {layer} proj {proj} has {} experts, plan says {count}",
                experts.len()
            )
            .into());
        }
        // Geometry is uniform across a projection (the census derives it per requirement);
        // the contiguity check above pins the map to exactly experts 0..E, so
        // `into_values` yields expert order and its index IS the expert id.
        let mut names = Vec::with_capacity(count);
        let mut geometry: Option<(usize, usize, memra_gguf::tensor_contract::QuantConstraint)> =
            None;
        for (name, out_f, in_f, quant) in experts.into_values() {
            match geometry {
                None => geometry = Some((out_f, in_f, quant)),
                Some((o, i, q)) if (o, i) == (out_f, in_f) && q == quant => {}
                Some((o, i, _)) => {
                    return Err(format!(
                        "qwen4exp_gpu: layer {layer} proj {proj} mixes expert geometry \
                         ({out_f}, {in_f}) vs ({o}, {i}) or quant classes"
                    )
                    .into());
                }
            }
            names.push(name);
        }
        let (out_f, in_f, quant) = geometry
            .ok_or_else(|| format!("qwen4exp_gpu: layer {layer} proj {proj} has no expert rows"))?;
        per_layer.entry(layer).or_default()[proj as usize] = Some(PerExpertPlan {
            names,
            out_f,
            in_f,
            quant,
        });
    }
    for (layer, mut projections) in per_layer {
        let MlpPlan::Moe(moe) = &plan_layer_at(&plan, layer).expect("checked above").mlp else {
            unreachable!("checked above");
        };
        let take = |slot: &mut Option<PerExpertPlan>, what: &str| -> Res<PerExpertPlan> {
            slot.take()
                .ok_or_else(|| format!("qwen4exp_gpu: layer {layer} missing {what} experts").into())
        };
        bank_plans.insert(
            layer,
            BankPlan {
                n_expert: moe.expert_count as usize,
                ff: moe.expert_intermediate_size as usize,
                hidden: plan.hidden_size as usize,
                src: BankPlanSrc::PerExpert {
                    gate: take(&mut projections[0], "gate")?,
                    up: take(&mut projections[1], "up")?,
                    down: take(&mut projections[2], "down")?,
                },
            },
        );
    }
    Ok(LoadedCheckpoint {
        plan,
        weights,
        model,
        bank_plans,
        tables,
    })
}

/// One expert-bank projection's byte fingerprint — the loader's memory-ordering gate.
///
/// A change that only moves WHEN bank bytes are materialized has to leave WHICH bytes
/// untouched, and "untouched" is a digest, not an argument. `digest` is sha256 over the
/// projection's payload in device-upload order (NVFP4: codes then scales then the
/// little-endian macro row; f32/bf16: the raw uploaded bytes), so it pins the expert order,
/// the fused gate/up split, and the per-expert stack concatenation at once.
pub struct BankFingerprint {
    pub layer: u32,
    /// "gate" | "up" | "down".
    pub projection: &'static str,
    /// "f32" | "bf16" | "nvfp4".
    pub kind: &'static str,
    pub bytes: usize,
    /// Lowercase hex sha256.
    pub digest: String,
}

impl LoadedCheckpoint {
    /// Read ONE layer's expert bank off the still-open mmap. The streaming seam every bank
    /// consumer goes through; the returned source is the caller's to drop.
    fn read_bank(&self, index: u32) -> Res<BankSrc> {
        self.bank_plans
            .get(&index)
            .ok_or_else(|| format!("qwen4exp_gpu: no bank source for layer {index}"))?
            .read(&self.model)
    }

    /// Per-projection byte fingerprints for every bank, read one layer at a time (so this
    /// costs one layer of host memory, not the artifact). Gate instrument only — see
    /// `BankFingerprint`; the tiny-fixture gate compares these against banked goldens.
    pub fn bank_fingerprints(&self) -> Res<Vec<BankFingerprint>> {
        use sha2::{Digest, Sha256};
        let mut out = Vec::new();
        for (&layer, plan) in &self.bank_plans {
            let bank = plan.read(&self.model)?;
            for (projection, src) in [("gate", &bank.gate), ("up", &bank.up), ("down", &bank.down)]
            {
                let mut hasher = Sha256::new();
                let (kind, bytes) = match src {
                    // f32 bit patterns little-endian: the exact bytes `htod` uploads.
                    BankTensorSrc::F32(data) => {
                        for value in data {
                            hasher.update(value.to_le_bytes());
                        }
                        ("f32", data.len() * 4)
                    }
                    BankTensorSrc::Bf16(raw) => {
                        hasher.update(raw);
                        ("bf16", raw.len())
                    }
                    BankTensorSrc::Nvfp4 {
                        codes,
                        scales,
                        macros,
                        ..
                    } => {
                        hasher.update(codes);
                        hasher.update(scales);
                        for m in macros {
                            hasher.update(m.to_le_bytes());
                        }
                        ("nvfp4", codes.len() + scales.len() + macros.len() * 4)
                    }
                };
                out.push(BankFingerprint {
                    layer,
                    projection,
                    kind,
                    bytes,
                    digest: hasher
                        .finalize()
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect(),
                });
            }
        }
        Ok(out)
    }

    /// Expand banks and n-gram tables into plain `ReferenceWeights` entries so
    /// memra-reference can execute the checkpoint. TINY/SIBLING SCALE ONLY — the real
    /// artifact's banks/table do not fit host f32; the GPU path never takes this.
    pub fn into_reference_weights(mut self) -> Res<ReferenceWeights> {
        let bank_plans = std::mem::take(&mut self.bank_plans);
        for (index, plan) in bank_plans {
            let bank = plan.read(&self.model)?;
            let gate = bank_to_f32(&bank.gate, bank.n_expert, bank.ff, bank.hidden)?;
            let up = bank_to_f32(&bank.up, bank.n_expert, bank.ff, bank.hidden)?;
            let down = bank_to_f32(&bank.down, bank.n_expert, bank.hidden, bank.ff)?;
            self.weights.insert(
                layer_id(index, LayerTensor::MoeExpertGateBank),
                ReferenceTensor::new(vec![bank.n_expert, bank.ff, bank.hidden], gate)?,
            );
            self.weights.insert(
                layer_id(index, LayerTensor::MoeExpertUpBank),
                ReferenceTensor::new(vec![bank.n_expert, bank.ff, bank.hidden], up)?,
            );
            self.weights.insert(
                layer_id(index, LayerTensor::MoeExpertDownBank),
                ReferenceTensor::new(vec![bank.n_expert, bank.hidden, bank.ff], down)?,
            );
        }
        for (index, bytes) in self.tables {
            let ple = self.plan.layers[index as usize]
                .ple
                .as_ref()
                .ok_or("qwen4exp_gpu: table on a non-PLE layer")?;
            let head_dim = ple.head_embed_dim as usize;
            let table = NgramTable::Bf16(bytes);
            let rows = table.rows(head_dim);
            let mut data = vec![0.0f32; rows * head_dim];
            for row in 0..rows {
                table.gather_into(
                    row,
                    head_dim,
                    &mut data[row * head_dim..(row + 1) * head_dim],
                );
            }
            self.weights.insert(
                family_id(format!(
                    "trunk.layers.{index}.ple.ple_embedding.ngram_embedding"
                )),
                ReferenceTensor::new(vec![rows, head_dim], data)?,
            );
        }
        Ok(self.weights)
    }
}

impl LoadedCheckpoint {
    /// CLONE the float weights and expand ONLY the MTP bank(s) into `ReferenceWeights`
    /// entries — the real-checkpoint draft-parity instrument (mtp-spec lane): the host
    /// reference twin needs the mtp.* rows + embed/head, and must NOT expand the trunk
    /// banks (48 layers of f32 experts do not fit anywhere). Borrowing form so ONE
    /// checkpoint read serves both the engine model and the host twin.
    pub fn mtp_reference_weights(&self) -> Res<ReferenceWeights> {
        let mut weights = self.weights.clone();
        let n_trunk = self.plan.layers.len() as u32;
        for (index, plan) in &self.bank_plans {
            if *index < n_trunk {
                continue;
            }
            let bank = plan.read(&self.model)?;
            let gate = bank_to_f32(&bank.gate, bank.n_expert, bank.ff, bank.hidden)?;
            let up = bank_to_f32(&bank.up, bank.n_expert, bank.ff, bank.hidden)?;
            let down = bank_to_f32(&bank.down, bank.n_expert, bank.hidden, bank.ff)?;
            weights.insert(
                layer_id(*index, LayerTensor::MoeExpertGateBank),
                ReferenceTensor::new(vec![bank.n_expert, bank.ff, bank.hidden], gate)?,
            );
            weights.insert(
                layer_id(*index, LayerTensor::MoeExpertUpBank),
                ReferenceTensor::new(vec![bank.n_expert, bank.ff, bank.hidden], up)?,
            );
            weights.insert(
                layer_id(*index, LayerTensor::MoeExpertDownBank),
                ReferenceTensor::new(vec![bank.n_expert, bank.hidden, bank.ff], down)?,
            );
        }
        Ok(weights)
    }
}

/// Host-dequant a bank tensor to f32 [E, out, in] (NVFP4 via the pub dsv4 decoder — the
/// same value chain the device kernel reproduces).
fn bank_to_f32(bank: &BankTensorSrc, n_expert: usize, out_f: usize, in_f: usize) -> Res<Vec<f32>> {
    match bank {
        BankTensorSrc::F32(data) => Ok(data.clone()),
        BankTensorSrc::Bf16(bytes) => Ok(bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect()),
        BankTensorSrc::Nvfp4 {
            codes,
            scales,
            macros,
            ..
        } => {
            let mut out = Vec::with_capacity(n_expert * out_f * in_f);
            let wbytes = out_f * in_f / 2;
            let sbytes = out_f * in_f / 16;
            for expert in 0..n_expert {
                out.extend(memra_gguf::dsv4::dequant_nvfp4_expert(
                    &codes[expert * wbytes..(expert + 1) * wbytes],
                    &scales[expert * sbytes..(expert + 1) * sbytes],
                    macros[expert],
                    out_f,
                    in_f,
                ));
            }
            Ok(out)
        }
    }
}

impl Qwen4ExpGpu {
    /// Load a qwen4_exp checkpoint dir (config.json + safetensors; the BF16 export or the
    /// per-expert modelopt NVFP4 mint) through the pack/plan/contract into engine-resident
    /// weights: trunk f32 on device, n-gram table host-resident bf16, NVFP4 expert banks
    /// as-stored on device.
    pub fn load_from_dir(e: &Engine, dir: &std::path::Path) -> Res<Self> {
        Self::from_loaded_checkpoint(e, read_checkpoint(dir)?)
    }

    /// `load_from_dir` with real-checkpoint loader knobs (`LoadOptions`).
    pub fn load_from_dir_with(e: &Engine, dir: &std::path::Path, opts: LoadOptions) -> Res<Self> {
        Self::from_loaded_checkpoint(e, read_checkpoint_with(dir, opts)?)
    }

    /// Card-1 draft placement (mtp10): the trunk builds on `e` (card 0) and the MTP
    /// draft block — weights, ~5 GB DeviceBf16 expert bank, private lm-head copy — on
    /// `draft_e` (card 1). Requires `opts.load_mtp` and P2P between the pair
    /// (`tp2_enable_p2p`); the spec loop's wide rows cross per round.
    pub fn load_from_dir_dev1(
        e: &Engine,
        draft_e: &Engine,
        dir: &std::path::Path,
        opts: LoadOptions,
    ) -> Res<Self> {
        Self::from_loaded_checkpoint_dual(e, Some(draft_e), read_checkpoint_with(dir, opts)?)
    }

    /// Consume a `LoadedCheckpoint` into the engine-resident model. Banks and n-gram
    /// tables MOVE (the real artifact's 102 GB table must not be cloned).
    pub fn from_loaded_checkpoint(e: &Engine, checkpoint: LoadedCheckpoint) -> Res<Self> {
        Self::from_loaded_checkpoint_dual(e, None, checkpoint)
    }

    /// `from_loaded_checkpoint` with the optional card-1 draft engine: the MTP bank
    /// (layer index >= n_trunk) uploads to `draft_e` when given; the trunk banks stay
    /// on `e` either way.
    pub fn from_loaded_checkpoint_dual(
        e: &Engine,
        draft_e: Option<&Engine>,
        checkpoint: LoadedCheckpoint,
    ) -> Res<Self> {
        let LoadedCheckpoint {
            plan,
            weights,
            model,
            bank_plans,
            tables,
        } = checkpoint;
        let mut parts = ExternalParts::default();
        let n_trunk = plan.layers.len() as u32;
        let upload_half = |e: &Engine, src: BankTensorSrc, device_bf16: bool| -> Res<BankHalf> {
            Ok(match src {
                BankTensorSrc::F32(data) => BankHalf::F32(e.htod(&data)?),
                BankTensorSrc::Nvfp4 {
                    codes,
                    scales,
                    macros,
                    ..
                } => BankHalf::Nvfp4 {
                    codes: e.htod_bytes(&codes)?,
                    scales: e.htod_bytes(&scales)?,
                    macros_dev: e.htod(&macros)?,
                    macros,
                },
                // Residency was decided at read time (LoadOptions::host_bf16_banks /
                // load_mtp): trunk bf16 stays host (gate-mode); the MTP draft bank goes
                // device-resident bf16 (the draft decode path reads it in place).
                BankTensorSrc::Bf16(bytes) if device_bf16 => {
                    BankHalf::DeviceBf16(e.htod_bytes(&bytes)?)
                }
                BankTensorSrc::Bf16(bytes) => BankHalf::HostBf16(bytes),
            })
        };
        // STREAMED: one layer's bank is read off the mmap, uploaded, and dropped before the
        // next is read. Peak host cost is ONE layer (~1.5 GB on the real mint) instead of
        // the whole ~72 GB stack, which is what let the real gate load on a 180 GB-RAM box
        // (receipt: research/qwen4exp-bringup-20260829/loader/LOADER-STREAM.md).
        // `upload_half` moves each projection into the device slice, so the host copy is
        // freed at the end of every iteration.
        for (index, bank_plan) in bank_plans {
            let bank = bank_plan.read(&model)?;
            let device_bf16 = index >= n_trunk;
            // The MTP bank follows the draft's placement (card 1 when dev1 is armed).
            let bank_e = if device_bf16 { draft_e.unwrap_or(e) } else { e };
            parts.expert_banks.insert(
                index,
                ExpertBank {
                    gate: upload_half(bank_e, bank.gate, device_bf16)?,
                    up: upload_half(bank_e, bank.up, device_bf16)?,
                    down: upload_half(bank_e, bank.down, device_bf16)?,
                },
            );
        }
        // The mmap has no more readers; the model's weights are device-resident and the
        // table below is host-owned bytes.
        drop(model);
        for (index, bytes) in tables {
            parts.ngram_tables.insert(index, NgramTable::Bf16(bytes));
        }
        Self::from_reference_weights_with(e, draft_e, &plan, &weights, parts)
    }
}

// ==================================== TP2 (perf round 3) ====================================
//
// Two-card tensor-parallel DECODE over PCIe P2P (no NVLink) — the PROFILE-2 §TP2
// projection made real. Structure (the tp2-join-diet playbook, step37 lane):
//
// - The RESIDUAL IS REPLICATED: both cards hold the wide planes and run the entry embed,
//   PLE block, hyper-connection read/write gates, and exit mixer with bit-identical
//   weights on bit-identical inputs (replicated deterministic compute — kills every
//   broadcast except the two joins below). All replicated device math runs deterministic
//   kernels (bf16w matvecs, fused gates); TP2 therefore REQUIRES the bf16 trunk twins.
// - SPLIT: GDN by key-head blocks (card d owns orig key heads [d·nk/2, (d+1)·nk/2) and
//   the value heads mapping to them — compact per-card head order keeps kh = h % nk_h)),
//   QSA by head halves (12/12 query heads, 1/1 KV heads), MoE routed experts by expert-id
//   halves (card d owns experts [d·E/2, (d+1)·E/2); top-10 splits ~5/5 on average),
//   shared expert by ff halves, lm_head by vocab halves (card 0 reads the resident twin's
//   row prefix; card 1 holds the suffix copy).
// - JOINS: exactly 2 per layer (mixer out-proj partials, MoE+shared partials), each a
//   [hidden] f32 row pushed as a P2P kernel store into the peer's resident staging buffer
//   (`q4e_push_f32`, the direct-join mechanism) + one cross-device event wait each way;
//   BOTH cards then compute out = partial0 + partial1 in the SAME rank order, so the
//   replicated residual stays bit-identical across cards.
// - HOST twins unchanged: MoE routing (router GEMV + dtoh on card 0, top-k once, filtered
//   selection H2D to both), QSA indexer (card 0 projects + host mask, mask H2D to both),
//   PLE n-gram hashing (host, gathered rows H2D to both; the 102 GB table stays host-
//   resident and SHARED — the card-1 PLE replica carries no table).
// - Decode graphs stay OFF in TP2 (eager issue; the joins are the schedule). Prefill
//   stays single-card; the first `decode_step_tp2` migrates the mixer state into
//   per-card halves (host bounce, one-time) and the state is TP2-latched from then on.
//
// EXACTNESS CLASS (the gate statement): TP2 output matches single-card to TOLERANCE, not
// bit — the split out-projections sum row halves in a different association than the
// full GEMV, the expert combine becomes (Σ card-0 slots) + (Σ card-1 slots) instead of
// the slot-sequential chain, and the join add reorders those partial sums. Same
// accumulation class as every banked seam; gated by `--tp2-gate` per-row envelope +
// argmax vs the single-card twin, plus the greedy-divergence battery.

/// Per-card compact GDN half (see the head-map comment on `tp2_gdn_head_map`).
struct GdnHalfW {
    nk_h: usize,
    nv_h: usize,
    hk: usize,
    hv: usize,
    kernel: usize,
    gate_activation: GdnGateActivation,
    /// Row-stacked [qkv; z; beta; alpha] half twin (proj-stack residency: per-mat
    /// launches read row-offset views; the seam launches the whole stack).
    proj_b16: CudaSlice<u8>,
    out_b16: CudaSlice<u8>, // [hidden, nv_h*hv] (compact column block)
    conv_w: CudaSlice<f32>, // [conv_dim_h, K]
    a: CudaSlice<f32>,      // [nv_h]
    dt: CudaSlice<f32>,     // [nv_h]
    norm: CudaSlice<f32>,   // [hv] (replicated)
}

/// Per-card QSA half: query heads [d*nh_h, (d+1)*nh_h), KV heads [d*nkv_h, ...).
struct QsaHalfW {
    nh_h: usize,
    nkv_h: usize,
    hd: usize,
    n_rot: usize,
    rope_base: f32,
    scale: f32,
    /// Row-stacked [wq; wk; wv] half twin (proj-stack residency; wq rows are the fused
    /// [q|gate] block).
    proj_b16: CudaSlice<u8>,
    wo_b16: CudaSlice<u8>, // [hidden, nh_h*hd] (compact column block)
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    /// YaRN tables on THIS half's card (long-context lane); `None` on the shipped config.
    yarn: Option<YarnRopeW>,
}

enum MixerHalfW {
    Gdn(GdnHalfW),
    Qsa(QsaHalfW),
}

/// Card-1 NVFP4 expert-bank half (experts [E/2, E), local ids 0..E/2).
struct Nvfp4Half {
    codes: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    macros_dev: CudaSlice<f32>,
}

struct MoeHalfW {
    /// Card-1 bank halves (card 0 addresses the resident full bank with original ids).
    gate1: Nvfp4Half,
    up1: Nvfp4Half,
    down1: Nvfp4Half,
    /// Shared expert: card 0 reads the resident full twins' ROW PREFIX (gate/up) and its
    /// own compact down-column block; card 1 holds suffix/compact copies.
    shared_down0: CudaSlice<u8>, // card0 [hidden, sff_h]
    shared_down1: CudaSlice<u8>,                // card1 [hidden, sff_h]
    shared_input_gate1: Option<CudaSlice<f32>>, // card1 [hidden]
    /// Row-stacked [gate_half; up_half] twins (proj-stack residency): card 0 stacks the
    /// ROW PREFIXES of the full mats (not contiguous in the resident full stack), card 1
    /// its suffix copies. Per-mat launches read row-offset views (0 / sff_h).
    shared_gu0_b16: CudaSlice<u8>,
    shared_gu1_b16: CudaSlice<u8>,
}

struct Tp2LayerW {
    attn_gate1: GateW,
    mlp_gate1: GateW,
    mixer0: MixerHalfW,
    mixer1: MixerHalfW,
    moe: MoeHalfW,
    ple1: Option<PleW>,
    /// This layer's resolved expert placement — the SAME object that chose which expert
    /// rows were gathered into `moe`'s card-1 bank. One source of truth for the upload
    /// and for the route split is what keeps a placement from being applied to one and
    /// not the other (the failure mode that would read as a model bug, not a config bug).
    place: LayerPlacement,
}

/// The TP2 shard: card-1 replicas + both cards' split halves + join plumbing.
pub struct Tp2Shard {
    layers: Vec<Tp2LayerW>,
    exit_gate1: GateW,
    lm_head1: CudaSlice<u8>, // card1 bf16 [vocab - vsplit, hidden]
    vsplit: usize,
    /// Join staging, TWO buffers per direction alternating by join parity. Two is
    /// provably enough: the overwrite of buffer (j+2 mod 2) is transitively ordered
    /// after the peer's read at join j (the peer's push at j+1 follows its add at j on
    /// its in-order stream, and our wait on that push precedes our overwrite).
    stage0: [CudaSlice<f32>; 2], // card0 staging (receives card1 partials)
    stage1: [CudaSlice<f32>; 2], // card1 staging (receives card0 partials)
    stage0_raw: [u64; 2],
    stage1_raw: [u64; 2],
    ev0: [cudarc::driver::CudaEvent; 2], // card0 push done, by join parity
    ev1: [cudarc::driver::CudaEvent; 2], // card1 push done, by join parity
}

enum MixerHalfState {
    Gdn {
        conv: CudaSlice<f32>,  // [pad, conv_dim_h]
        state: CudaSlice<f32>, // [nv_h, hv, hk]
    },
    Qsa {
        /// This card's KV half [cap, nkv_h*hd] — f32 or the kvq q8_0/q5_1 byte caches
        /// (format follows the single-card store; head halves are 32-block aligned at
        /// hd % 32 == 0, so quantized migration gathers BYTES verbatim).
        kv: QsaKvStore,
    },
}

struct Tp2LayerState {
    m0: MixerHalfState,
    m1: MixerHalfState,
    ple1: Option<PleState>,
}

struct Tp2State {
    ws1: StepPool,
    layers: Vec<Tp2LayerState>,
    graphs: Tp2Graphs,
    /// TP2-PREFILL join staging (chunk-sized [t*hidden] per direction, two buffers per
    /// direction by join parity — the decode stage buffers' proof carries over
    /// verbatim). Lazily sized at the first `forward_tp2` chunk; `raw` = the peer's
    /// UVA pointers baked for `launch_push`.
    pf_stage0: Option<[CudaSlice<f32>; 2]>, // on card0 (receives card1 partials)
    pf_stage1: Option<[CudaSlice<f32>; 2]>,
    pf_stage0_raw: [u64; 2],
    pf_stage1_raw: [u64; 2],
    pf_rows: usize,
}

/// Captured TP2 decode segments per card (the single-card StepGraphs pattern applied
/// per rank): `a[d][li]` = attn gate_read + GDN half + join push, `b[d][li]` = join add +
/// gate_write + mlp gate_read (+ card1 shared-half prestage), `exit[d]` = exit mixer +
/// lm_head half. GDN layers without PLE only; QSA/PLE layers, the router boundary,
/// the variable-shape MoE tail, and the MoE join stay eager. Event records/waits sit
/// BETWEEN segment launches (not capturable) — same choreography in warm and replay
/// modes. The first TP2 decode step runs fully eager to park every slot (allocations
/// inside a capture become graph mem nodes); captures are lazy on the second step.
#[derive(Default)]
struct Tp2Graphs {
    warm: bool,
    a: [Vec<Option<GraphEntry>>; 2],
    b: [Vec<Option<GraphEntry>>; 2],
    /// Count-gated MoE tail (routed half + shared add + join push) — fixed launch
    /// shapes via the pack blob, so the variable expert split still captures.
    c: [Vec<Option<GraphEntry>>; 2],
    /// MoE join add + gate_write.
    d: [Vec<Option<GraphEntry>>; 2],
    exit: [Option<GraphEntry>; 2],
}

/// Compact value-head order for card `d`: heads h with h % nk in [d*nk_h, (d+1)*nk_h),
/// ascending. With nv % nk == 0 this is exactly `(j / nk_h) * nk + (j % nk_h) + d*nk_h`,
/// and the compact system stays self-consistent with the kernels' kh = h % nk_h mapping.
fn tp2_gdn_head_map(d: usize, nk: usize, nv: usize) -> Vec<usize> {
    let nk_h = nk / 2;
    let nv_h = nv / 2;
    (0..nv_h)
        .map(|j| (j / nk_h) * nk + (j % nk_h) + d * nk_h)
        .collect()
}

/// Gather whole rows (row-major [rows, in_f]) into a compact copy.
fn gather_rows_host(src: &[f32], in_f: usize, rows: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * in_f);
    for &r in rows {
        out.extend_from_slice(&src[r * in_f..(r + 1) * in_f]);
    }
    out
}

/// Gather column blocks per row (row-major [nrows, ncols]) into a compact copy.
fn gather_cols_host(
    src: &[f32],
    nrows: usize,
    ncols: usize,
    blocks: &[(usize, usize)],
) -> Vec<f32> {
    let width: usize = blocks.iter().map(|&(_, l)| l).sum();
    let mut out = Vec::with_capacity(nrows * width);
    for r in 0..nrows {
        for &(start, len) in blocks {
            out.extend_from_slice(&src[r * ncols + start..r * ncols + start + len]);
        }
    }
    out
}

fn need_twin(e: &Engine, data: &[f32], in_f: usize, what: &str) -> Res<CudaSlice<u8>> {
    bf16_twin(e, data, in_f)?.ok_or_else(|| {
        format!("qwen4exp_gpu tp2: {what} has no exact bf16 twin (in_f {in_f})").into()
    })
}

/// Launch `q4e_push_f32`: UVA store of `n` f32 into the PEER address `dst_raw` on `e`'s
/// stream (the direct-join push).
fn launch_push(e: &Engine, src: &CudaSlice<f32>, dst_raw: u64, n: usize) -> Res<()> {
    let f = e.func("q4e_push_f32");
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let nl = n as i64;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(src).arg(&dst_raw).arg(&nl);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Enable bidirectional P2P + pool peer access between two engines (the
/// `configure_native_p2p` essentials for the qwen4_exp TP2 pair; pool access makes every
/// pooled allocation UVA-addressable from the peer, which is what `q4e_push_f32` needs).
pub fn tp2_enable_p2p(e0: &Engine, e1: &Engine) -> Res<()> {
    use cudarc::driver::sys;
    for (src, dst) in [(e0, e1), (e1, e0)] {
        let mut can = 0i32;
        unsafe {
            sys::cuDeviceCanAccessPeer(&mut can, src.ctx().cu_device(), dst.ctx().cu_device())
                .result()?;
        }
        if can == 0 {
            return Err(format!(
                "qwen4exp_gpu tp2: dev{} cannot access dev{} over P2P",
                src.ctx().ordinal(),
                dst.ctx().ordinal()
            )
            .into());
        }
        src.ctx().bind_to_thread()?;
        let rc = unsafe { sys::cuCtxEnablePeerAccess(dst.ctx().cu_ctx(), 0) };
        use cudarc::driver::sys::cudaError_enum as E;
        if rc != E::CUDA_SUCCESS && rc != E::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            return Err(format!("qwen4exp_gpu tp2: cuCtxEnablePeerAccess failed: {rc:?}").into());
        }
    }
    for (owner, accessor) in [(e0, e1), (e1, e0)] {
        let device = cudarc::driver::result::device::get(owner.ctx().ordinal() as i32)?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, device).result()?;
        }
        let desc = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: accessor.ctx().ordinal() as i32,
            },
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        let rc = unsafe { sys::cuMemPoolSetAccess(pool, &desc, 1) };
        if rc != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("qwen4exp_gpu tp2: cuMemPoolSetAccess failed: {rc:?}").into());
        }
    }
    Ok(())
}

/// Build the card-1 replica PLE weight set from the checkpoint's host weights (the
/// device parts of `PleW` with an EMPTY table — the 102 GB n-gram table stays host-
/// resident on the model and is passed to `ple_block` explicitly).
fn build_ple_replica(
    e: &Engine,
    weights: &ReferenceWeights,
    prefix: &str,
    ple_plan: &PleEmbeddingPlan,
    streams: usize,
    hidden: usize,
) -> Res<PleW> {
    let embed_dim = ple_plan.embed_dim as usize;
    let key_proj = expect(weights, &family_id(format!("{prefix}ple.key_proj.weight")))?;
    let conv_w = expect(weights, &family_id(format!("{prefix}ple.conv1d.weight")))?;
    let norm_slices = |name: &str| -> Res<Vec<CudaSlice<f32>>> {
        let t = expect(weights, &family_id(format!("{prefix}ple.{name}.weight")))?;
        split_rows(&t.data, streams, hidden, 1)
            .into_iter()
            .map(|v| e.htod(&v))
            .collect::<Result<_, _>>()
    };
    let ints = |name: &str| -> Res<Vec<i64>> {
        let t = expect(
            weights,
            &family_id(format!("{prefix}ple.ple_embedding.{name}")),
        )?;
        t.ints
            .clone()
            .ok_or_else(|| "qwen4exp_gpu: n-gram buffer must be I64".into())
    };
    Ok(PleW {
        plan: *ple_plan,
        key_proj: split_rows(&key_proj.data, streams, hidden, embed_dim)
            .into_iter()
            .map(|v| e.htod(&v))
            .collect::<Result<_, _>>()?,
        value_proj: upload(
            e,
            &expect(
                weights,
                &family_id(format!("{prefix}ple.value_proj.weight")),
            )?,
        )?,
        norm_key: norm_slices("norm_key")?,
        norm_query: norm_slices("norm_query")?,
        norm_conv: norm_slices("norm_conv")?,
        conv_w: split_rows(&conv_w.data, streams, hidden, ple_plan.conv_kernel as usize)
            .into_iter()
            .map(|v| e.htod(&v))
            .collect::<Result<_, _>>()?,
        multipliers: ints("layer_multipliers")?,
        sizes: ints("ngram_heads_vocab_sizes")?,
        offsets: ints("ngram_heads_offsets")?,
        table: NgramTable::F32(Vec::new()), // never gathered; the model's table is passed in
    })
}

/// Build one card's compact GDN half from host weights.
#[allow(clippy::too_many_arguments)]
fn build_gdn_half(
    e: &Engine,
    weights: &ReferenceWeights,
    index: u32,
    gdn: &GatedDeltaNetPlan,
    hidden: usize,
    d: usize,
) -> Res<GdnHalfW> {
    let (nk, nv) = (gdn.key_heads as usize, gdn.value_heads as usize);
    let (hk, hv) = (gdn.key_head_dim as usize, gdn.value_head_dim as usize);
    if nk % 2 != 0 || nv % nk != 0 {
        return Err(format!(
            "qwen4exp_gpu tp2: GDN layer {index} nk {nk} / nv {nv} does not split by key-head halves"
        )
        .into());
    }
    let (nk_h, nv_h) = (nk / 2, nv / 2);
    let head_map = tp2_gdn_head_map(d, nk, nv);
    let qkv = expect(weights, &layer_id(index, LayerTensor::GdnQkv))?;
    let z = expect(weights, &layer_id(index, LayerTensor::GdnGate))?;
    let beta = expect(weights, &layer_id(index, LayerTensor::GdnBeta))?;
    let alpha = expect(weights, &layer_id(index, LayerTensor::GdnAlpha))?;
    let out = expect(weights, &layer_id(index, LayerTensor::GdnOutput))?;
    let conv_w = expect(weights, &layer_id(index, LayerTensor::GdnConv1d))?;
    let a = expect(weights, &layer_id(index, LayerTensor::GdnA))?;
    let dt = expect(weights, &layer_id(index, LayerTensor::GdnDtBias))?;
    let norm = expect(weights, &layer_id(index, LayerTensor::GdnNorm))?;
    let kernel = gdn.conv_kernel as usize;
    // Row lists for the fused qkv/conv (q block, k block, v per compact head).
    let mut qkv_rows: Vec<usize> = Vec::with_capacity(2 * nk_h * hk + nv_h * hv);
    qkv_rows.extend(d * nk_h * hk..(d + 1) * nk_h * hk);
    qkv_rows.extend(nk * hk + d * nk_h * hk..nk * hk + (d + 1) * nk_h * hk);
    for &hm in &head_map {
        qkv_rows.extend(2 * nk * hk + hm * hv..2 * nk * hk + (hm + 1) * hv);
    }
    let mut z_rows: Vec<usize> = Vec::with_capacity(nv_h * hv);
    for &hm in &head_map {
        z_rows.extend(hm * hv..(hm + 1) * hv);
    }
    let out_blocks: Vec<(usize, usize)> = head_map.iter().map(|&hm| (hm * hv, hv)).collect();
    let qkv_c = gather_rows_host(&qkv.data, hidden, &qkv_rows);
    let z_c = gather_rows_host(&z.data, hidden, &z_rows);
    let beta_c = gather_rows_host(&beta.data, hidden, &head_map);
    let alpha_c = gather_rows_host(&alpha.data, hidden, &head_map);
    let out_c = gather_cols_host(&out.data, hidden, nv * hv, &out_blocks);
    let conv_c = gather_rows_host(&conv_w.data, kernel, &qkv_rows);
    let a_c: Vec<f32> = head_map.iter().map(|&hm| a.data[hm]).collect();
    let dt_c: Vec<f32> = head_map.iter().map(|&hm| dt.data[hm]).collect();
    Ok(GdnHalfW {
        nk_h,
        nv_h,
        hk,
        hv,
        kernel,
        gate_activation: gdn.gate_activation,
        proj_b16: need_stack_twin(
            e,
            &[&qkv_c, &z_c, &beta_c, &alpha_c],
            hidden,
            "tp2 gdn proj half",
        )?,
        out_b16: need_twin(e, &out_c, nv_h * hv, "tp2 gdn out half")?,
        conv_w: e.htod(&conv_c)?,
        a: e.htod(&a_c)?,
        dt: e.htod(&dt_c)?,
        norm: e.htod(&norm.data)?,
    })
}

/// Build one card's QSA half from host weights (query heads d*nh_h.., KV heads d*nkv_h..).
#[allow(clippy::too_many_arguments)]
fn build_qsa_half(
    e: &Engine,
    weights: &ReferenceWeights,
    index: u32,
    attn: &FullAttentionPlan,
    hidden: usize,
    d: usize,
) -> Res<QsaHalfW> {
    let nh = attn.query_heads as usize;
    let nkv = attn.kv_heads as usize;
    let hd = attn.key_head_dim as usize;
    if nh % 2 != 0 || nkv % 2 != 0 || nh % nkv != 0 {
        return Err(format!(
            "qwen4exp_gpu tp2: QSA layer {index} heads {nh}/{nkv} do not split in halves"
        )
        .into());
    }
    let (nh_h, nkv_h) = (nh / 2, nkv / 2);
    let wq = expect(weights, &layer_id(index, LayerTensor::Query))?;
    let wk = expect(weights, &layer_id(index, LayerTensor::Key))?;
    let wv = expect(weights, &layer_id(index, LayerTensor::Value))?;
    let wo = expect(weights, &layer_id(index, LayerTensor::AttentionOutput))?;
    // Fused [q|gate] per head: card d's heads are a contiguous row block.
    let q_rows: Vec<usize> = (d * nh_h * 2 * hd..(d + 1) * nh_h * 2 * hd).collect();
    let kv_rows: Vec<usize> = (d * nkv_h * hd..(d + 1) * nkv_h * hd).collect();
    let wq_c = gather_rows_host(&wq.data, hidden, &q_rows);
    let wk_c = gather_rows_host(&wk.data, hidden, &kv_rows);
    let wv_c = gather_rows_host(&wv.data, hidden, &kv_rows);
    let wo_c = gather_cols_host(&wo.data, hidden, nh * hd, &[(d * nh_h * hd, nh_h * hd)]);
    let opt_norm = |tensor: LayerTensor| -> Res<Option<CudaSlice<f32>>> {
        match weights.get(&layer_id(index, tensor)) {
            Some(t) => Ok(Some(e.htod(&t.data)?)),
            None => Ok(None),
        }
    };
    let scale = match attn.scale {
        memra_gguf::model_plan::AttentionScale::InverseSqrtKeyDim => 1.0 / (hd as f32).sqrt(),
        memra_gguf::model_plan::AttentionScale::Fixed(scale) => scale,
    };
    Ok(QsaHalfW {
        nh_h,
        nkv_h,
        hd,
        n_rot: attn.rope.dimensions as usize,
        rope_base: attn.rope.base,
        scale,
        proj_b16: need_stack_twin(e, &[&wq_c, &wk_c, &wv_c], hidden, "tp2 qsa proj half")?,
        wo_b16: need_twin(e, &wo_c, nh_h * hd, "tp2 qsa o half")?,
        q_norm: opt_norm(LayerTensor::QueryNorm)?,
        k_norm: opt_norm(LayerTensor::KeyNorm)?,
        // Device table on THIS half's card; the width check ran at single-card load.
        yarn: build_yarn(e, &attn.rope, None, index)?,
    })
}

/// Build the TP2 shard from a loaded checkpoint (host data), before the single-card
/// model consumes it. Card 0 gets its compact split copies on `e0`; card 1 gets its
/// replicas + halves on `e1`.
pub fn build_tp2_shard(e0: &Engine, e1: &Engine, ckpt: &LoadedCheckpoint) -> Res<Tp2Shard> {
    let plan = &ckpt.plan;
    let weights = &ckpt.weights;
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    if vocab % 2 != 0 {
        return Err("qwen4exp_gpu tp2: odd vocab".into());
    }
    let mixer_plan = plan
        .exit_mixer
        .ok_or("qwen4exp_gpu tp2: missing exit mixer")?;
    let streams = mixer_plan.streams as usize;
    let rank = mixer_plan.bottleneck_rank as usize;
    // Expert placement, read ONCE per shard build (MEMRA_Q4E_EP_MAP; unset = the even
    // split control arm). Refusals are load-time, before a single byte is uploaded.
    let plan_experts = plan
        .layers
        .iter()
        .find_map(|l| match &l.mlp {
            MlpPlan::Moe(m) => Some(m.expert_count as usize),
            _ => None,
        })
        .ok_or("qwen4exp_gpu tp2: no MoE layer in the plan")?;
    let placement = match Tp2Placement::from_env(plan_experts)? {
        Some(p) => p,
        None => Tp2Placement::even(plan_experts),
    };
    println!(
        "# tp2-placement\tstrategy={}\tentry_rank={}\texperts={plan_experts}\tsource={}",
        placement.strategy(),
        placement.entry_rank(),
        placement.source()
    );
    let mut layers = Vec::with_capacity(plan.layers.len());
    for layer in &plan.layers {
        let prefix = format!("trunk.layers.{}.", layer.index);
        let _g1 = e1.gpu.enter_main()?;
        let attn_gate1 = load_gate(
            e1,
            weights,
            &prefix,
            "attn_hyper_connection.",
            streams,
            hidden,
            rank,
            true,
        )?;
        let mlp_gate1 = load_gate(
            e1,
            weights,
            &prefix,
            "mlp_hyper_connection.",
            streams,
            hidden,
            rank,
            true,
        )?;
        let ple1 = match layer.ple.as_ref() {
            None => None,
            Some(ple_plan) => Some(build_ple_replica(
                e1, weights, &prefix, ple_plan, streams, hidden,
            )?),
        };
        drop(_g1);
        let (mixer0, mixer1) = match &layer.attention {
            AttentionPlan::GatedDeltaNet(gdn) => {
                let _g0 = e0.gpu.enter_main()?;
                let m0 = MixerHalfW::Gdn(build_gdn_half(e0, weights, layer.index, gdn, hidden, 0)?);
                drop(_g0);
                let _g1 = e1.gpu.enter_main()?;
                let m1 = MixerHalfW::Gdn(build_gdn_half(e1, weights, layer.index, gdn, hidden, 1)?);
                (m0, m1)
            }
            AttentionPlan::Full(attn) => {
                let _g0 = e0.gpu.enter_main()?;
                let m0 =
                    MixerHalfW::Qsa(build_qsa_half(e0, weights, layer.index, attn, hidden, 0)?);
                drop(_g0);
                let _g1 = e1.gpu.enter_main()?;
                let m1 =
                    MixerHalfW::Qsa(build_qsa_half(e1, weights, layer.index, attn, hidden, 1)?);
                (m0, m1)
            }
            other => {
                return Err(format!("qwen4exp_gpu tp2: unsupported mixer {other:?}").into());
            }
        };
        // MoE: card1 bank halves from the HOST bank sources; NVFP4 required.
        let MlpPlan::Moe(moe_plan) = &layer.mlp else {
            return Err("qwen4exp_gpu tp2: non-MoE layer".into());
        };
        let experts = moe_plan.expert_count as usize;
        let ff = moe_plan.expert_intermediate_size as usize;
        if experts % 2 != 0 {
            return Err("qwen4exp_gpu tp2: odd expert count".into());
        }
        // Streamed like every other bank consumer: this layer's source is read off the
        // mmap here and dropped at the end of the iteration. TP2 therefore reads the bank
        // bytes TWICE per load (once here for the card-1 gather, once in
        // `from_loaded_checkpoint` for card 0) — a load-time disk/page-cache cost, not a
        // numerics or steady-state one, and the price of not holding 72 GB of banks on a
        // host that has 180 GB total. Single-pass shard+model build is a named follow-up.
        let bank = ckpt.read_bank(layer.index)?;
        let bank = &bank;
        // The layer's expert placement, resolved ONCE here and carried on the shard so
        // the route split at decode/prefill cannot disagree with what was uploaded.
        let place = placement.layer(layer.index, experts)?;
        // Card 1's bank is a GATHER of the placed expert rows in local-slot order, not a
        // contiguous suffix slice. For the even control arm `card1` is exactly
        // `e_half..experts` ascending, so the gather concatenates the same bytes the old
        // slice handed over, in the same order — bit-identical by construction, which is
        // what makes "even split = control arm" a statement about bytes and not a hope.
        let upper =
            |src: &BankTensorSrc, out_f: usize, in_f: usize, what: &str| -> Res<Nvfp4Half> {
                let BankTensorSrc::Nvfp4 {
                    codes,
                    scales,
                    macros,
                    ..
                } = src
                else {
                    return Err(format!("qwen4exp_gpu tp2: {what} bank is not NVFP4").into());
                };
                let wbytes = out_f * in_f / 2;
                let sbytes = out_f * in_f / 16;
                let need_codes = place.card1.len() * wbytes;
                let need_scales = place.card1.len() * sbytes;
                if codes.len() < experts * wbytes || scales.len() < experts * sbytes {
                    return Err(format!(
                        "qwen4exp_gpu tp2: {what} bank is {} code / {} scale bytes, too \
                         small for {experts} experts x ({wbytes}, {sbytes})",
                        codes.len(),
                        scales.len()
                    )
                    .into());
                }
                let mut gcodes = Vec::with_capacity(need_codes);
                let mut gscales = Vec::with_capacity(need_scales);
                let mut gmacros = Vec::with_capacity(place.card1.len());
                for &eid in &place.card1 {
                    let e = eid as usize;
                    gcodes.extend_from_slice(&codes[e * wbytes..(e + 1) * wbytes]);
                    gscales.extend_from_slice(&scales[e * sbytes..(e + 1) * sbytes]);
                    gmacros.push(macros[e]);
                }
                Ok(Nvfp4Half {
                    codes: e1.htod_bytes(&gcodes)?,
                    scales: e1.htod_bytes(&gscales)?,
                    macros_dev: e1.htod(&gmacros)?,
                })
            };
        let shared = moe_plan
            .shared
            .as_ref()
            .ok_or("qwen4exp_gpu tp2: missing shared expert")?;
        let sff = shared.intermediate_size as usize;
        if sff % 2 != 0 {
            return Err("qwen4exp_gpu tp2: odd shared ff".into());
        }
        let sffh = sff / 2;
        let sh_gate = expect(weights, &layer_id(layer.index, LayerTensor::SharedMlpGate))?;
        let sh_up = expect(weights, &layer_id(layer.index, LayerTensor::SharedMlpUp))?;
        let sh_down = expect(weights, &layer_id(layer.index, LayerTensor::SharedMlpDown))?;
        let sh_ig = if shared.gated {
            Some(expect(
                weights,
                &layer_id(layer.index, LayerTensor::SharedMlpInputGate),
            )?)
        } else {
            None
        };
        let moe = {
            let _g1 = e1.gpu.enter_main()?;
            let gate1 = upper(&bank.gate, ff, hidden, "gate")?;
            let up1 = upper(&bank.up, ff, hidden, "up")?;
            let down1 = upper(&bank.down, hidden, ff, "down")?;
            let shared_gu1_b16 = need_stack_twin(
                e1,
                &[&sh_gate.data[sffh * hidden..], &sh_up.data[sffh * hidden..]],
                hidden,
                "tp2 shared gate/up (card1)",
            )?;
            let down1_c = gather_cols_host(&sh_down.data, hidden, sff, &[(sffh, sffh)]);
            let shared_down1 = need_twin(e1, &down1_c, sffh, "tp2 shared down (card1)")?;
            let shared_input_gate1 = match sh_ig.as_ref() {
                Some(t) => Some(e1.htod(&t.data)?),
                None => None,
            };
            drop(_g1);
            let _g0 = e0.gpu.enter_main()?;
            let down0_c = gather_cols_host(&sh_down.data, hidden, sff, &[(0, sffh)]);
            let shared_down0 = need_twin(e0, &down0_c, sffh, "tp2 shared down (card0)")?;
            let shared_gu0_b16 = need_stack_twin(
                e0,
                &[&sh_gate.data[..sffh * hidden], &sh_up.data[..sffh * hidden]],
                hidden,
                "tp2 shared gate/up (card0)",
            )?;
            MoeHalfW {
                gate1,
                up1,
                down1,
                shared_down0,
                shared_down1,
                shared_input_gate1,
                shared_gu0_b16,
                shared_gu1_b16,
            }
        };
        layers.push(Tp2LayerW {
            attn_gate1,
            mlp_gate1,
            mixer0,
            mixer1,
            moe,
            ple1,
            place,
        });
    }
    let _g1 = e1.gpu.enter_main()?;
    let exit_gate1 = load_gate(
        e1,
        weights,
        "trunk.hyper_connection_mixer.",
        "",
        streams,
        hidden,
        rank,
        false,
    )?;
    let vsplit = vocab / 2;
    let head = match weights.get(&TensorId::OutputProjection) {
        Some(t) => &t.data,
        None => &expect(weights, &TensorId::TokenEmbedding)?.data.clone(),
    };
    let lm_head1 = need_twin(
        e1,
        &head[vsplit * hidden..],
        hidden,
        "tp2 lm_head upper half",
    )?;
    let stage1 = [e1.zeros(hidden)?, e1.zeros(hidden)?];
    let ev1 = [e1.ctx().new_event(None)?, e1.ctx().new_event(None)?];
    let stage1_raw = {
        let s = e1.gpu.stream();
        [stage1[0].device_ptr(&s).0, stage1[1].device_ptr(&s).0]
    };
    drop(_g1);
    let _g0 = e0.gpu.enter_main()?;
    let stage0 = [e0.zeros(hidden)?, e0.zeros(hidden)?];
    let ev0 = [e0.ctx().new_event(None)?, e0.ctx().new_event(None)?];
    let stage0_raw = {
        let s = e0.gpu.stream();
        [stage0[0].device_ptr(&s).0, stage0[1].device_ptr(&s).0]
    };
    Ok(Tp2Shard {
        layers,
        exit_gate1,
        lm_head1,
        vsplit,
        stage0,
        stage1,
        stage0_raw,
        stage1_raw,
        ev0,
        ev1,
    })
}

impl Qwen4ExpGpu {
    /// Load a checkpoint dir for TP2: P2P is enabled, the shard is built from the host
    /// checkpoint data (before the single-card model consumes it), then the single-card
    /// model loads onto `e0` exactly as `load_from_dir_with`.
    pub fn load_from_dir_tp2(
        e0: &Engine,
        e1: &Engine,
        dir: &std::path::Path,
        opts: LoadOptions,
    ) -> Res<(Self, Tp2Shard)> {
        tp2_enable_p2p(e0, e1)?;
        let checkpoint = read_checkpoint_with(dir, opts)?;
        let shard = build_tp2_shard(e0, e1, &checkpoint)?;
        let model = Self::from_loaded_checkpoint(e0, checkpoint)?;
        Ok((model, shard))
    }

    /// One-time single-card -> TP2 half-state migration (host bounce; the state is
    /// TP2-latched afterwards). Card 0 keeps the host-side indexer raw-key cache and
    /// the single-card PLE history (replicated path); mixer device state splits.
    fn tp2_migrate(
        &self,
        e0: &Engine,
        e1: &Engine,
        _shard: &Tp2Shard,
        state: &mut Qwen4ExpState,
    ) -> Res<()> {
        let cap = state.capacity;
        let pos = state.pos;
        let mut tlayers = Vec::with_capacity(self.layers.len());
        for (layer, lstate) in self.layers.iter().zip(state.layers.iter_mut()) {
            let (m0, m1) = match (&layer.mixer, &mut lstate.mixer) {
                (
                    MixerW::Gdn(gdn),
                    MixerState::Gdn {
                        conv,
                        state: gstate,
                    },
                ) => {
                    let p = &gdn.plan;
                    let (nk, nv) = (p.key_heads as usize, p.value_heads as usize);
                    let (hk, hv) = (p.key_head_dim as usize, p.value_head_dim as usize);
                    let (nk_h, nv_h) = (nk / 2, nv / 2);
                    let pad = p.conv_kernel as usize - 1;
                    let conv_dim = 2 * nk * hk + nv * hv;
                    let conv_dim_h = 2 * nk_h * hk + nv_h * hv;
                    let state_host = {
                        let _g = e0.gpu.enter_main()?;
                        e0.dtoh(gstate)?
                    };
                    let conv_host = {
                        let _g = e0.gpu.enter_main()?;
                        e0.dtoh(conv)?
                    };
                    let mut halves = Vec::with_capacity(2);
                    for d in 0..2 {
                        let head_map = tp2_gdn_head_map(d, nk, nv);
                        let state_c = gather_rows_host(&state_host, hv * hk, &head_map);
                        let mut blocks: Vec<(usize, usize)> = vec![
                            (d * nk_h * hk, nk_h * hk),
                            (nk * hk + d * nk_h * hk, nk_h * hk),
                        ];
                        blocks.extend(head_map.iter().map(|&hm| (2 * nk * hk + hm * hv, hv)));
                        let conv_c = gather_cols_host(&conv_host, pad, conv_dim, &blocks);
                        let e = if d == 0 { e0 } else { e1 };
                        let _g = e.gpu.enter_main()?;
                        let state_dev = e.htod(&state_c)?;
                        let conv_dev = e.htod(&conv_c)?;
                        debug_assert_eq!(conv_c.len(), pad * conv_dim_h);
                        halves.push(MixerHalfState::Gdn {
                            conv: conv_dev,
                            state: state_dev,
                        });
                    }
                    let m1 = halves.pop().expect("two halves");
                    let m0 = halves.pop().expect("two halves");
                    (m0, m1)
                }
                (MixerW::Qsa(qsa), MixerState::Qsa { kv, .. }) => {
                    let nkv = qsa.attn.kv_heads as usize;
                    let hd = qsa.attn.key_head_dim as usize;
                    let nkv_h = nkv / 2;
                    let mut halves = Vec::with_capacity(2);
                    match &*kv {
                        QsaKvStore::F32 { k, v } => {
                            let (k_host, v_host) = {
                                let _g = e0.gpu.enter_main()?;
                                (
                                    e0.dtoh_view(&k.slice(0..pos * nkv * hd))?,
                                    e0.dtoh_view(&v.slice(0..pos * nkv * hd))?,
                                )
                            };
                            for d in 0..2 {
                                let block = [(d * nkv_h * hd, nkv_h * hd)];
                                let k_c = gather_cols_host(&k_host, pos, nkv * hd, &block);
                                let v_c = gather_cols_host(&v_host, pos, nkv * hd, &block);
                                let e = if d == 0 { e0 } else { e1 };
                                let _g = e.gpu.enter_main()?;
                                let mut k_dev = e.zeros(cap * nkv_h * hd)?;
                                let mut v_dev = e.zeros(cap * nkv_h * hd)?;
                                if pos > 0 {
                                    let mut kv_view = k_dev.slice_mut(0..pos * nkv_h * hd);
                                    e.gpu.stream().memcpy_htod(&k_c, &mut kv_view)?;
                                    let mut vv_view = v_dev.slice_mut(0..pos * nkv_h * hd);
                                    e.gpu.stream().memcpy_htod(&v_c, &mut vv_view)?;
                                }
                                halves.push(MixerHalfState::Qsa {
                                    kv: QsaKvStore::F32 { k: k_dev, v: v_dev },
                                });
                            }
                        }
                        QsaKvStore::Q8Q5 { k, v } => {
                            // Quantized halves: each head's hd elems are whole q8/q5
                            // 32-blocks (hd % 32 == 0 on real geometry), so the half
                            // rows gather BYTES verbatim — no dequant, no requant, the
                            // half caches are bit-slices of the single-card cache.
                            if hd % 32 != 0 {
                                return Err("qwen4exp_gpu tp2: quantized halves need \
                                            hd % 32 == 0 (byte-aligned head blocks)"
                                    .into());
                            }
                            let (krb, vrb) = (q8_row_bytes(nkv * hd), q5_row_bytes(nkv * hd));
                            let (krb_h, vrb_h) =
                                (q8_row_bytes(nkv_h * hd), q5_row_bytes(nkv_h * hd));
                            let (k_host, v_host) = {
                                let _g = e0.gpu.enter_main()?;
                                (
                                    e0.dtoh_u8_view(&k.slice(0..pos * krb))?,
                                    e0.dtoh_u8_view(&v.slice(0..pos * vrb))?,
                                )
                            };
                            for d in 0..2 {
                                let mut k_c = Vec::with_capacity(pos * krb_h);
                                let mut v_c = Vec::with_capacity(pos * vrb_h);
                                for r in 0..pos {
                                    let ko = r * krb + d * krb_h;
                                    k_c.extend_from_slice(&k_host[ko..ko + krb_h]);
                                    let vo = r * vrb + d * vrb_h;
                                    v_c.extend_from_slice(&v_host[vo..vo + vrb_h]);
                                }
                                let e = if d == 0 { e0 } else { e1 };
                                let _g = e.gpu.enter_main()?;
                                let mut k_dev = e.alloc_u8(cap * krb_h)?;
                                let mut v_dev = e.alloc_u8(cap * vrb_h)?;
                                if pos > 0 {
                                    let mut kv_view = k_dev.slice_mut(0..pos * krb_h);
                                    e.gpu.stream().memcpy_htod(&k_c, &mut kv_view)?;
                                    let mut vv_view = v_dev.slice_mut(0..pos * vrb_h);
                                    e.gpu.stream().memcpy_htod(&v_c, &mut vv_view)?;
                                }
                                halves.push(MixerHalfState::Qsa {
                                    kv: QsaKvStore::Q8Q5 { k: k_dev, v: v_dev },
                                });
                            }
                        }
                    }
                    // The single-card cache is DEAD after migration (a TP2-touched
                    // state refuses single-card forwards), and at long-context
                    // capacities it is the largest allocation on card 0 — stub it.
                    {
                        let _g = e0.gpu.enter_main()?;
                        *kv = match &*kv {
                            QsaKvStore::F32 { .. } => QsaKvStore::F32 {
                                k: e0.zeros(1)?,
                                v: e0.zeros(1)?,
                            },
                            QsaKvStore::Q8Q5 { .. } => QsaKvStore::Q8Q5 {
                                k: e0.alloc_u8(34)?,
                                v: e0.alloc_u8(24)?,
                            },
                        };
                    }
                    let m1 = halves.pop().expect("two halves");
                    let m0 = halves.pop().expect("two halves");
                    (m0, m1)
                }
                _ => return Err("qwen4exp_gpu tp2: layer/state mixer mismatch".into()),
            };
            // Card-1 PLE history replica (replicated path): copy card0's normed-conv rows.
            let ple1 = match lstate.ple.as_ref() {
                None => None,
                Some(ps) => {
                    let mut conv_hist = Vec::with_capacity(ps.conv_hist.len());
                    for h in &ps.conv_hist {
                        let host = {
                            let _g = e0.gpu.enter_main()?;
                            e0.dtoh(h)?
                        };
                        let _g = e1.gpu.enter_main()?;
                        conv_hist.push(e1.htod(&host)?);
                    }
                    Some(PleState {
                        conv_hist,
                        ngram_ids: Vec::new(),
                        ngram_history: Vec::new(),
                        ngram_last_eos: -1,
                    })
                }
            };
            tlayers.push(Tp2LayerState { m0, m1, ple1 });
        }
        state.tp2 = Some(Tp2State {
            ws1: StepPool::default(),
            layers: tlayers,
            graphs: Tp2Graphs::default(),
            pf_stage0: None,
            pf_stage1: None,
            pf_stage0_raw: [0; 2],
            pf_stage1_raw: [0; 2],
            pf_rows: 0,
        });
        Ok(())
    }

    /// Per-card GDN split half (t-generic — TP2 prefill runs chunk-sized t):
    /// projections, conv, scan, norm+gate, and the compact out-projection PARTIAL
    /// (joined by the driver). Mirrors `gdn_forward`.
    #[allow(clippy::too_many_arguments)]
    fn gdn_forward_half(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        eps: f32,
        h: &GdnHalfW,
        mixed: &CudaSlice<f32>,
        hstate: &mut MixerHalfState,
        t: usize,
    ) -> Res<CudaSlice<f32>> {
        let MixerHalfState::Gdn { conv, state } = hstate else {
            return Err("qwen4exp_gpu tp2: GDN half bound to non-GDN state".into());
        };
        let hidden = self.hidden;
        let (nk, nv, hk, hv) = (h.nk_h, h.nv_h, h.hk, h.hv);
        let kernel = h.kernel;
        let pad = kernel - 1;
        let conv_dim = 2 * nk * hk + nv * hv;
        let mut qkv = ws.take_f32(e, "gdn.qkv", t * conv_dim, 0)?;
        let mut z = ws.take_f32(e, "gdn.z", t * nv * hv, 0)?;
        let mut beta_raw = ws.take_f32(e, "gdn.beta", t * nv, 0)?;
        let mut alpha = ws.take_f32(e, "gdn.alpha", t * nv, 0)?;
        // Proj stack (round 4): the 4 half projections in ONE launch (bit-identical
        // rows; OFF arm = row-offset views of the same required stack). t == 1 only
        // (decode form); chunks run the per-mat row-offset launches.
        if t == 1 && proj_stack_on() {
            launch_qmatvec_bf16w_multi4(
                e,
                &h.proj_b16,
                mixed,
                &[
                    (&qkv, conv_dim),
                    (&z, nv * hv),
                    (&beta_raw, nv),
                    (&alpha, nv),
                ],
                hidden,
            )?;
        } else {
            launch_qmatvec_bf16w_off(e, &h.proj_b16, 0, mixed, &mut qkv, hidden, conv_dim, t)?;
            launch_qmatvec_bf16w_off(e, &h.proj_b16, conv_dim, mixed, &mut z, hidden, nv * hv, t)?;
            launch_qmatvec_bf16w_off(
                e,
                &h.proj_b16,
                conv_dim + nv * hv,
                mixed,
                &mut beta_raw,
                hidden,
                nv,
                t,
            )?;
            launch_qmatvec_bf16w_off(
                e,
                &h.proj_b16,
                conv_dim + nv * hv + nv,
                mixed,
                &mut alpha,
                hidden,
                nv,
                t,
            )?;
        }
        let mut g_log = ws.take_f32(e, "gdn.glog", t * nv, 0)?;
        e.gdn_glog_v(&alpha.slice(0..t * nv), &h.dt, &h.a, &mut g_log, nv, t)?;
        ws.put_f32("gdn.alpha", alpha);
        let mut conv_out = ws.take_f32(e, "gdn.conv_out", t * conv_dim, 0)?;
        launch_dwconv(
            e,
            &qkv,
            conv,
            &h.conv_w,
            &mut conv_out,
            t,
            pad,
            conv_dim,
            kernel,
            1,
            1,
        )?;
        let mut o = ws.take_f32(e, "gdn.o", t * nv * hv, 0)?;
        let scale = 1.0 / (hk as f32).sqrt();
        if t == 1 && gdn_step_on() && hk % 32 == 0 && hk <= 1024 {
            launch_gdn_scan_step(
                e, &conv_out, &g_log, &beta_raw, state, &mut o, nk, nv, hk, hv, scale, eps,
            )?;
        } else {
            launch_gdn_scan(
                e, &conv_out, &g_log, &beta_raw, state, &mut o, nk, nv, hk, hv, t, scale, eps,
            )?;
        }
        ws.put_f32("gdn.conv_out", conv_out);
        // conv history <- last `pad` raw qkv rows (zeros keep their place when t < pad).
        if t >= pad {
            e.copy_range_into(conv, 0, &qkv, (t - pad) * conv_dim, pad * conv_dim)?;
        } else {
            let keep = pad - t;
            let mut tmp = ws.take_f32(e, "gdn.tmp", keep * conv_dim, 0)?;
            e.copy_range_into(&mut tmp, 0, conv, t * conv_dim, keep * conv_dim)?;
            e.copy_range_into(conv, 0, &tmp, 0, keep * conv_dim)?;
            e.copy_range_into(conv, keep * conv_dim, &qkv, 0, t * conv_dim)?;
            ws.put_f32("gdn.tmp", tmp);
        }
        ws.put_f32("gdn.qkv", qkv);
        ws.put_f32("gdn.beta", beta_raw);
        ws.put_f32("gdn.glog", g_log);
        let mut gated = ws.take_f32(e, "gdn.gated", t * nv * hv, 0)?;
        match h.gate_activation {
            GdnGateActivation::Sigmoid if gdn_fuse_on() => {
                launch_rms_sigmul(e, &o, &h.norm, &z, &mut gated, hv, t * nv, eps)?;
            }
            GdnGateActivation::Sigmoid => {
                let mut normed = ws.take_f32(e, "gdn.normed", t * nv * hv, 0)?;
                e.rms_norm(&o, &h.norm, &mut normed, hv, t * nv, eps)?;
                let mut sg = ws.take_f32(e, "gdn.sg", t * nv * hv, 0)?;
                e.sigmoid(&z, &mut sg, t * nv * hv)?;
                e.mul(&normed, &sg, &mut gated, t * nv * hv)?;
                ws.put_f32("gdn.sg", sg);
                ws.put_f32("gdn.normed", normed);
            }
            GdnGateActivation::Silu => {
                let mut normed = ws.take_f32(e, "gdn.normed", t * nv * hv, 0)?;
                e.rms_norm(&o, &h.norm, &mut normed, hv, t * nv, eps)?;
                e.silu_mul(&z, &normed, &mut gated, t * nv * hv)?;
                ws.put_f32("gdn.normed", normed);
            }
        }
        let mut partial = ws.take_f32(e, "mixer.out", t * hidden, 0)?;
        launch_qmatvec_bf16w(
            e,
            &h.out_b16,
            &gated,
            &mut partial,
            nv * hv,
            hidden,
            t,
            1,
            0,
            0,
            nv * hv,
            0,
        )?;
        ws.put_f32("gdn.gated", gated);
        ws.put_f32("gdn.z", z);
        ws.put_f32("gdn.o", o);
        Ok(partial)
    }

    /// Per-card QSA split half UP TO the cache append (t-generic — TP2 prefill runs
    /// chunk-sized t); returns (q, gate) for the post-selection half
    /// (`qsa_half_attend`). The indexer selection is built once on card 0.
    #[allow(clippy::too_many_arguments)]
    fn qsa_half_proj(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        eps: f32,
        h: &QsaHalfW,
        mixed: &CudaSlice<f32>,
        hstate: &mut MixerHalfState,
        base_pos: usize,
        t: usize,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>)> {
        let MixerHalfState::Qsa { kv } = hstate else {
            return Err("qwen4exp_gpu tp2: QSA half bound to non-QSA state".into());
        };
        let hidden = self.hidden;
        let (nh, nkv, hd) = (h.nh_h, h.nkv_h, h.hd);
        let mut q_fused = ws.take_f32(e, "qsa.qf", t * 2 * nh * hd, 0)?;
        let mut k_new = ws.take_f32(e, "qsa.k", t * nkv * hd, 0)?;
        let mut v_new = ws.take_f32(e, "qsa.v", t * nkv * hd, 0)?;
        // Proj stack (round 4): wq/wk/wv halves in ONE launch (bit-identical rows; OFF
        // arm = row-offset views of the same required stack). t == 1 only (the multi4
        // kernel is a decode form); chunks run the per-mat row-offset launches.
        if t == 1 && proj_stack_on() {
            launch_qmatvec_bf16w_multi4(
                e,
                &h.proj_b16,
                mixed,
                &[
                    (&q_fused, 2 * nh * hd),
                    (&k_new, nkv * hd),
                    (&v_new, nkv * hd),
                ],
                hidden,
            )?;
        } else {
            launch_qmatvec_bf16w_off(
                e,
                &h.proj_b16,
                0,
                mixed,
                &mut q_fused,
                hidden,
                2 * nh * hd,
                t,
            )?;
            launch_qmatvec_bf16w_off(
                e,
                &h.proj_b16,
                2 * nh * hd,
                mixed,
                &mut k_new,
                hidden,
                nkv * hd,
                t,
            )?;
            launch_qmatvec_bf16w_off(
                e,
                &h.proj_b16,
                2 * nh * hd + nkv * hd,
                mixed,
                &mut v_new,
                hidden,
                nkv * hd,
                t,
            )?;
        }
        let mut q = ws.take_f32(e, "qsa.q", t * nh * hd, 0)?;
        let mut gate = ws.take_f32(e, "qsa.gate", t * nh * hd, 0)?;
        e.q_gate_split(&q_fused, &mut q, &mut gate, hd, nh, t)?;
        ws.put_f32("qsa.qf", q_fused);
        let mut q = if let Some(norm) = h.q_norm.as_ref() {
            let mut dst = ws.take_f32(e, "qsa.qn", t * nh * hd, 0)?;
            e.rms_norm(&q, norm, &mut dst, hd, t * nh, eps)?;
            ws.put_f32("qsa.q", q);
            dst
        } else {
            q
        };
        let mut k_new = if let Some(norm) = h.k_norm.as_ref() {
            let mut dst = ws.take_f32(e, "qsa.kn", t * nkv * hd, 0)?;
            e.rms_norm(&k_new, norm, &mut dst, hd, t * nkv, eps)?;
            ws.put_f32("qsa.k", k_new);
            dst
        } else {
            k_new
        };
        let positions: Vec<i32> = (0..t).map(|i| (base_pos + i) as i32).collect();
        let pos_dev = ws.take_i32(e, "qsa.pos", &positions, 0)?;
        if let Some(yarn) = h.yarn.as_ref() {
            e.rope_neox_ffm(
                &mut q,
                &pos_dev,
                hd,
                h.n_rot,
                nh,
                t,
                h.rope_base,
                1.0,
                &yarn.ff,
                yarn.mscale,
            )?;
            e.rope_neox_ffm(
                &mut k_new,
                &pos_dev,
                hd,
                h.n_rot,
                nkv,
                t,
                h.rope_base,
                1.0,
                &yarn.ff,
                yarn.mscale,
            )?;
        } else {
            e.rope_neox(&mut q, &pos_dev, hd, h.n_rot, nh, t, h.rope_base, 1.0)?;
            e.rope_neox(&mut k_new, &pos_dev, hd, h.n_rot, nkv, t, h.rope_base, 1.0)?;
        }
        ws.put_i32("qsa.pos", pos_dev);
        match kv {
            QsaKvStore::F32 { k, v } => {
                e.copy_range_into(k, base_pos * nkv * hd, &k_new, 0, t * nkv * hd)?;
                e.copy_range_into(v, base_pos * nkv * hd, &v_new, 0, t * nkv * hd)?;
            }
            QsaKvStore::Q8Q5 { k, v } => {
                launch_q4e_kv_append(e, &k_new, &v_new, k, v, base_pos, t, nkv * hd)?;
            }
        }
        ws.put_f32(
            if h.k_norm.is_some() {
                "qsa.kn"
            } else {
                "qsa.k"
            },
            k_new,
        );
        ws.put_f32("qsa.v", v_new);
        Ok((q, gate))
    }

    /// Post-selection QSA half: BLOCK-LIST SDPA over this card's KV half (bit-identical
    /// to the historical masked form on the same selection — the fixture-longatt /
    /// arm-0f pedigree — and the only form the quantized halves have), sigmoid gate,
    /// and the compact out-projection PARTIAL. t-generic.
    #[allow(clippy::too_many_arguments)]
    fn qsa_half_attend(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        h: &QsaHalfW,
        hstate: &MixerHalfState,
        q: CudaSlice<f32>,
        gate: CudaSlice<f32>,
        pos_dev: &CudaSlice<i32>,
        meta_dev: &CudaSlice<i32>,
        max_count: usize,
        t: usize,
        t_kv: usize,
    ) -> Res<CudaSlice<f32>> {
        let MixerHalfState::Qsa { kv } = hstate else {
            return Err("qwen4exp_gpu tp2: QSA half bound to non-QSA state".into());
        };
        let hidden = self.hidden;
        let (nh, nkv, hd) = (h.nh_h, h.nkv_h, h.hd);
        let mut attended = ws.take_f32(e, "qsa.att", t * nh * hd, 0)?;
        match kv {
            QsaKvStore::F32 { k, v } => {
                let k_view = k.slice(0..t_kv * nkv * hd);
                let v_view = v.slice(0..t_kv * nkv * hd);
                launch_sdpa_blocklist(
                    e,
                    &q,
                    &k_view,
                    &v_view,
                    &mut attended,
                    pos_dev,
                    meta_dev,
                    hd,
                    nh,
                    nkv,
                    t,
                    max_count,
                    h.scale,
                )?;
            }
            QsaKvStore::Q8Q5 { k, v } => {
                launch_q4e_sdpa_blocklist_q8q5(
                    e,
                    &q,
                    k,
                    v,
                    &mut attended,
                    pos_dev,
                    meta_dev,
                    hd,
                    nh,
                    nkv,
                    t,
                    max_count,
                    h.scale,
                )?;
            }
        }
        ws.put_f32(
            if h.q_norm.is_some() {
                "qsa.qn"
            } else {
                "qsa.q"
            },
            q,
        );
        let mut sg = ws.take_f32(e, "qsa.sg", t * nh * hd, 0)?;
        e.sigmoid(&gate, &mut sg, t * nh * hd)?;
        let mut gated = ws.take_f32(e, "qsa.gated", t * nh * hd, 0)?;
        e.mul(&attended, &sg, &mut gated, t * nh * hd)?;
        let mut partial = ws.take_f32(e, "mixer.out", t * hidden, 0)?;
        launch_qmatvec_bf16w(
            e,
            &h.wo_b16,
            &gated,
            &mut partial,
            nh * hd,
            hidden,
            t,
            1,
            0,
            0,
            nh * hd,
            0,
        )?;
        ws.put_f32("qsa.sg", sg);
        ws.put_f32("qsa.gated", gated);
        ws.put_f32("qsa.att", attended);
        ws.put_f32("qsa.gate", gate);
        Ok(partial)
    }

    /// The QSA indexer host twin factored for TP2 (runs on card 0's projection; the mask
    /// bytes feed BOTH cards' masked SDPA halves).
    // dead_code: bring-up scaffolding the in-flight qwen4exp lanes still call; not deleted in
    // the clippy-zero lane (bit-neutral by construction).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn qsa_indexer_mask(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        qsa: &QsaW,
        eps: f32,
        mixed: &CudaSlice<f32>,
        raw_keys: &mut IdxRawCache,
        pooled_keys: &mut Vec<f32>,
        base_pos: usize,
    ) -> Res<Vec<u8>> {
        let overlay = &qsa.overlay;
        let idx_dim = overlay.head_dim as usize;
        let qk_width = (overlay.query_heads as usize + overlay.kv_heads as usize) * idx_dim;
        let hidden = self.hidden;
        let mut idx_proj = ws.take_f32(e, "qsa.idxp", qk_width, 0)?;
        e.linear_device_into(mixed, &qsa.idx_proj, &mut idx_proj, 1, hidden, qk_width)?;
        let rows = e.dtoh_view(&idx_proj.slice(0..qk_width))?;
        ws.put_f32("qsa.idxp", idx_proj);
        raw_keys.append_rows_f32(
            &rows[overlay.query_heads as usize * idx_dim..qk_width],
            1,
            idx_dim,
        );
        indexer_mask_rows(
            overlay,
            qsa.attn.rope.base,
            qsa.yarn.as_ref().map(|y| (y.ff_host.as_slice(), y.mscale)),
            eps,
            &qsa.idx_q_norm,
            &qsa.idx_k_norm,
            &rows,
            raw_keys,
            pooled_keys,
            base_pos,
            1,
            base_pos + 1,
            0,
        )
    }

    /// Shared-expert half on one card: gate/up rows (card 0 = the resident full twins'
    /// row prefix; card 1 = its suffix copies), silu, compact down columns. Returns the
    /// down PARTIAL and the (replicated-deterministic) input-gate scalar buffer.
    #[allow(clippy::too_many_arguments)]
    fn tp2_shared_half(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        gu_b16: &CudaSlice<u8>,
        down_b16: &CudaSlice<u8>,
        input_gate: Option<&CudaSlice<f32>>,
        mixed: &CudaSlice<f32>,
        sffh: usize,
        t: usize,
    ) -> Res<(CudaSlice<f32>, Option<CudaSlice<f32>>)> {
        let hidden = self.hidden;
        let mut sh_gate = ws.take_f32(e, "moe.sh_gate", t * sffh, 0)?;
        let mut sh_up = ws.take_f32(e, "moe.sh_up", t * sffh, 0)?;
        // Proj stack (round 4): shared gate/up halves in ONE launch (bit-identical rows;
        // OFF arm = row-offset views of the same required stack). t == 1 only.
        if t == 1 && proj_stack_on() {
            launch_qmatvec_bf16w_multi4(
                e,
                gu_b16,
                mixed,
                &[(&sh_gate, sffh), (&sh_up, sffh)],
                hidden,
            )?;
        } else {
            launch_qmatvec_bf16w_off(e, gu_b16, 0, mixed, &mut sh_gate, hidden, sffh, t)?;
            launch_qmatvec_bf16w_off(e, gu_b16, sffh, mixed, &mut sh_up, hidden, sffh, t)?;
        }
        let mut act = ws.take_f32(e, "moe.sh_act", t * sffh, 0)?;
        e.silu_mul(&sh_gate, &sh_up, &mut act, t * sffh)?;
        let mut shared = ws.take_f32(e, "moe.sh_down", t * hidden, 0)?;
        launch_qmatvec_bf16w(
            e,
            down_b16,
            &act,
            &mut shared,
            sffh,
            hidden,
            t,
            1,
            0,
            0,
            sffh,
            0,
        )?;
        let g = match input_gate {
            Some(w) => {
                let mut g = ws.take_f32(e, "moe.g", t, 0)?;
                e.sigmoid_dot_rows_into(mixed, w, &mut g, hidden, t)?;
                Some(g)
            }
            None => None,
        };
        ws.put_f32("moe.sh_gate", sh_gate);
        ws.put_f32("moe.sh_up", sh_up);
        ws.put_f32("moe.sh_act", act);
        Ok((shared, g))
    }
}

impl Qwen4ExpGpu {
    /// One TP2 decode step (t == 1, eager issue — decode graphs stay off in TP2; the
    /// joins are the schedule). Prefill stays single-card; the first call migrates the
    /// state (one-way latch). Requires the bf16-trunk + fused-gate seams ON (replicated
    /// compute must be deterministic-kernel-only).
    pub fn decode_step_tp2(
        &self,
        e0: &Engine,
        e1: &Engine,
        shard: &Tp2Shard,
        token: u32,
        state: &mut Qwen4ExpState,
    ) -> Res<Vec<f32>> {
        if !trunk_bf16_on() || !hc_fused_gate_on() {
            return Err(
                "qwen4exp_gpu tp2: requires set_trunk_bf16(true) and set_hc_fused_gate(true) \
                 (replicated compute must run deterministic kernels)"
                    .into(),
            );
        }
        if state.pos + 1 > state.capacity {
            return Err("qwen4exp_gpu: state capacity exceeded".into());
        }
        if state.tp2.is_none() {
            self.tp2_migrate(e0, e1, shard, state)?;
            state.graphs = StepGraphs::default();
        }
        let hidden = self.hidden;
        let vocab = self.vocab;
        let vsplit = shard.vsplit;
        let base_pos = state.pos;
        let reserve = state.reserve;
        state.tokens.push(token);
        let Qwen4ExpState {
            ref tokens,
            ws: ref mut ws0,
            ref mut tp2,
            layers: ref mut lstates,
            ..
        } = *state;
        let Tp2State {
            ws1,
            layers: tlayers,
            graphs: tgraphs,
            ..
        } = tp2.as_mut().expect("migrated above");
        // Slot RESERVE unit: reserve-derived, NOT capacity — a long-context TP2 state
        // (1M rows) must not reserve capacity-sized plane slots (~10 GB each).
        let cap = reserve.max(1);

        // Entry: one embed row, H2D to both cards' plane slots (replicated planes).
        let token_us = token as usize;
        if token_us >= vocab {
            return Err(format!("qwen4exp_gpu: token {token_us} out of range").into());
        }
        let embedded = &self.embed_host[token_us * hidden..(token_us + 1) * hidden];
        let mut planes1: Vec<CudaSlice<f32>> = Vec::with_capacity(self.streams);
        let ptrs1 = {
            let _g = e1.gpu.enter_main()?;
            let embedded_dev = ws1.take_f32_h2d(e1, "entry.embed", embedded, cap * hidden)?;
            for s in 0..self.streams {
                let mut plane = ws1.take_f32(e1, PLANE_SLOTS[s], hidden, cap * hidden)?;
                e1.copy_into(&mut plane, 0, &embedded_dev, hidden)?;
                planes1.push(plane);
            }
            ws1.put_f32("entry.embed", embedded_dev);
            let ptr_vals: Vec<u64> = {
                let stream = e1.gpu.stream();
                planes1.iter().map(|p| p.device_ptr(&stream).0).collect()
            };
            ws1.take_u64_h2d(e1, "hc.ptrs", &ptr_vals, 0)?
        };
        let mut planes0: Vec<CudaSlice<f32>> = Vec::with_capacity(self.streams);
        let ptrs0 = {
            let _g = e0.gpu.enter_main()?;
            let embedded_dev = ws0.take_f32_h2d(e0, "entry.embed", embedded, cap * hidden)?;
            for s in 0..self.streams {
                let mut plane = ws0.take_f32(e0, PLANE_SLOTS[s], hidden, cap * hidden)?;
                e0.copy_into(&mut plane, 0, &embedded_dev, hidden)?;
                planes0.push(plane);
            }
            ws0.put_f32("entry.embed", embedded_dev);
            let ptr_vals: Vec<u64> = {
                let stream = e0.gpu.stream();
                planes0.iter().map(|p| p.device_ptr(&stream).0).collect()
            };
            ws0.take_u64_h2d(e0, "hc.ptrs", &ptr_vals, 0)?
        };

        // Segment-graph mode (the single-card StepGraphs pattern per rank): first TP2
        // step runs fully eager to park every slot; captures are lazy on the next step.
        let use_graphs = decode_graphs_on() && step_ws_on();
        let graphs_live = use_graphs && tgraphs.warm;
        if use_graphs && !tgraphs.warm {
            tgraphs.warm = true;
        }
        if graphs_live && tgraphs.a[0].len() != self.layers.len() {
            for d in 0..2 {
                tgraphs.a[d] = (0..self.layers.len()).map(|_| None).collect();
                tgraphs.b[d] = (0..self.layers.len()).map(|_| None).collect();
                tgraphs.c[d] = (0..self.layers.len()).map(|_| None).collect();
                tgraphs.d[d] = (0..self.layers.len()).map(|_| None).collect();
            }
        }
        for (li, layer) in self.layers.iter().enumerate() {
            let lstate = &mut lstates[li];
            let tw = &shard.layers[li];
            let ts = &mut tlayers[li];
            let eps_a = layer.eps_attn;
            let eps_m = layer.eps_mlp;
            let moe = &layer.moe;
            let ff = moe.plan.expert_intermediate_size as usize;
            let experts = moe.plan.expert_count as usize;
            let selected = moe.plan.experts_per_token as usize;
            let sff = moe
                .plan
                .shared
                .as_ref()
                .map(|s| s.intermediate_size as usize)
                .unwrap_or(0);
            let sffh = sff / 2;

            // ---- phase 1: attn gate + mixer half + join push (parity 0), per card ----
            match (&layer.mixer, &tw.mixer0, &tw.mixer1) {
                (MixerW::Gdn(_), MixerHalfW::Gdn(h0), MixerHalfW::Gdn(h1)) => {
                    {
                        let _g = e1.gpu.enter_main()?;
                        if let (Some(ple1), Some(ps1)) = (tw.ple1.as_ref(), ts.ple1.as_mut()) {
                            let table = &layer.ple.as_ref().expect("ple plan").table;
                            self.ple_block(
                                e1,
                                layer,
                                ple1,
                                table,
                                ps1,
                                &mut planes1,
                                tokens,
                                1,
                                false,
                                None,
                            )?;
                        }
                        if graphs_live && layer.ple.is_none() {
                            if tgraphs.a[1][li].is_none() {
                                tgraphs.a[1][li] =
                                    Some(e1.capture_graph_retained_nowarm(|eng| {
                                        self.tp2_gdn_seg_a(
                                            eng,
                                            ws1,
                                            &ptrs1,
                                            &tw.attn_gate1,
                                            h1,
                                            &mut ts.m1,
                                            &planes1,
                                            eps_a,
                                            shard.stage0_raw[0],
                                        )
                                    })?);
                            }
                            tgraphs.a[1][li].as_ref().unwrap().0.launch()?;
                        } else {
                            self.tp2_gdn_seg_a(
                                e1,
                                ws1,
                                &ptrs1,
                                &tw.attn_gate1,
                                h1,
                                &mut ts.m1,
                                &planes1,
                                eps_a,
                                shard.stage0_raw[0],
                            )?;
                        }
                        shard.ev1[0].record(&e1.gpu.stream())?;
                    }
                    {
                        let _g = e0.gpu.enter_main()?;
                        if let (Some(ple), Some(ps)) = (layer.ple.as_ref(), lstate.ple.as_mut()) {
                            self.ple_block(
                                e0,
                                layer,
                                ple,
                                &ple.table,
                                ps,
                                &mut planes0,
                                tokens,
                                1,
                                false,
                                None,
                            )?;
                        }
                        if graphs_live && layer.ple.is_none() {
                            if tgraphs.a[0][li].is_none() {
                                tgraphs.a[0][li] =
                                    Some(e0.capture_graph_retained_nowarm(|eng| {
                                        self.tp2_gdn_seg_a(
                                            eng,
                                            ws0,
                                            &ptrs0,
                                            &layer.attn_gate,
                                            h0,
                                            &mut ts.m0,
                                            &planes0,
                                            eps_a,
                                            shard.stage1_raw[0],
                                        )
                                    })?);
                            }
                            tgraphs.a[0][li].as_ref().unwrap().0.launch()?;
                        } else {
                            self.tp2_gdn_seg_a(
                                e0,
                                ws0,
                                &ptrs0,
                                &layer.attn_gate,
                                h0,
                                &mut ts.m0,
                                &planes0,
                                eps_a,
                                shard.stage1_raw[0],
                            )?;
                        }
                        shard.ev0[0].record(&e0.gpu.stream())?;
                    }
                }
                (MixerW::Qsa(qsa), MixerHalfW::Qsa(h0), MixerHalfW::Qsa(h1)) => {
                    // QSA stays eager: the indexer selection and the per-step t_kv
                    // launch shape are not capturable (single-card precedent).
                    let (q1, g1, inj1) = {
                        let _g = e1.gpu.enter_main()?;
                        let (mixed1, inj1) = self.gate_read(
                            e1,
                            ws1,
                            &ptrs1,
                            &tw.attn_gate1,
                            &planes1,
                            1,
                            eps_a,
                            false,
                        )?;
                        let (q1, g1) = self
                            .qsa_half_proj(e1, ws1, eps_a, h1, &mixed1, &mut ts.m1, base_pos, 1)?;
                        ws1.put_f32("hc.mixed", mixed1);
                        (q1, g1, inj1)
                    };
                    // The selection runs ONCE on card 0 (the single-card machinery:
                    // idxcache device raw cache, device scorer, audit twin) and its
                    // position lists feed BOTH cards' block-list halves — bit-identical
                    // to the historical masked form on the same selection.
                    let (sels, q0, g0, inj0) = {
                        let _g = e0.gpu.enter_main()?;
                        let (mixed0, inj0) = self.gate_read(
                            e0,
                            ws0,
                            &ptrs0,
                            &layer.attn_gate,
                            &planes0,
                            1,
                            eps_a,
                            false,
                        )?;
                        let (q0, g0) = self
                            .qsa_half_proj(e0, ws0, eps_a, h0, &mixed0, &mut ts.m0, base_pos, 1)?;
                        let MixerState::Qsa {
                            raw_keys,
                            pooled_keys,
                            pooled_dev,
                            pooled_dev_rows,
                            raw_dev,
                            raw_dev_rows,
                            idx_audit,
                            ..
                        } = &mut lstate.mixer
                        else {
                            return Err("qwen4exp_gpu tp2: QSA layer without raw-key cache".into());
                        };
                        let sels = self.qsa_update_select(
                            e0,
                            ws0,
                            qsa,
                            eps_a,
                            &mixed0,
                            raw_keys,
                            pooled_keys,
                            pooled_dev,
                            pooled_dev_rows,
                            raw_dev,
                            raw_dev_rows,
                            idx_audit.as_mut(),
                            base_pos,
                            1,
                            0,
                            false,
                        )?;
                        ws0.put_f32("hc.mixed", mixed0);
                        (sels, q0, g0, inj0)
                    };
                    let t_kv = base_pos + 1;
                    let block_size = qsa.overlay.block_size as usize;
                    let (pos_flat, meta, max_count) = rowsel_positions(&sels, block_size);
                    {
                        let _g = e1.gpu.enter_main()?;
                        let pos_dev = ws1.take_i32(e1, "qsa.selpos", &pos_flat, 0)?;
                        let meta_dev = ws1.take_i32(e1, "qsa.selmeta", &meta, 0)?;
                        let p1 = self.qsa_half_attend(
                            e1, ws1, h1, &ts.m1, q1, g1, &pos_dev, &meta_dev, max_count, 1, t_kv,
                        )?;
                        ws1.put_i32("qsa.selpos", pos_dev);
                        ws1.put_i32("qsa.selmeta", meta_dev);
                        launch_push(e1, &p1, shard.stage0_raw[0], hidden)?;
                        ws1.put_f32("mixer.out", p1);
                        put_inject(ws1, inj1);
                        shard.ev1[0].record(&e1.gpu.stream())?;
                    }
                    {
                        let _g = e0.gpu.enter_main()?;
                        let pos_dev = ws0.take_i32(e0, "qsa.selpos", &pos_flat, 0)?;
                        let meta_dev = ws0.take_i32(e0, "qsa.selmeta", &meta, 0)?;
                        let p0 = self.qsa_half_attend(
                            e0, ws0, h0, &ts.m0, q0, g0, &pos_dev, &meta_dev, max_count, 1, t_kv,
                        )?;
                        ws0.put_i32("qsa.selpos", pos_dev);
                        ws0.put_i32("qsa.selmeta", meta_dev);
                        launch_push(e0, &p0, shard.stage1_raw[0], hidden)?;
                        ws0.put_f32("mixer.out", p0);
                        put_inject(ws0, inj0);
                        shard.ev0[0].record(&e0.gpu.stream())?;
                    }
                }
                _ => return Err("qwen4exp_gpu tp2: mixer/shard shape mismatch".into()),
            }
            {
                let _g = e0.gpu.enter_main()?;
                e0.gpu.stream().wait(&shard.ev1[0])?;
            }
            {
                let _g = e1.gpu.enter_main()?;
                e1.gpu.stream().wait(&shard.ev0[0])?;
            }

            // ---- phase 2: join add + write + mlp gate (+ card1 shared prestage) ----
            {
                let _g = e1.gpu.enter_main()?;
                if graphs_live {
                    if tgraphs.b[1][li].is_none() {
                        tgraphs.b[1][li] = Some(e1.capture_graph_retained_nowarm(|eng| {
                            self.tp2_seg_b(
                                eng,
                                ws1,
                                &ptrs1,
                                &tw.mlp_gate1,
                                &mut planes1,
                                &shard.stage1[0],
                                false,
                                eps_m,
                                Some((
                                    &tw.moe.shared_gu1_b16,
                                    &tw.moe.shared_down1,
                                    tw.moe.shared_input_gate1.as_ref(),
                                    sffh,
                                )),
                            )
                        })?);
                    }
                    tgraphs.b[1][li].as_ref().unwrap().0.launch()?;
                } else {
                    self.tp2_seg_b(
                        e1,
                        ws1,
                        &ptrs1,
                        &tw.mlp_gate1,
                        &mut planes1,
                        &shard.stage1[0],
                        false,
                        eps_m,
                        Some((
                            &tw.moe.shared_gu1_b16,
                            &tw.moe.shared_down1,
                            tw.moe.shared_input_gate1.as_ref(),
                            sffh,
                        )),
                    )?;
                }
            }
            {
                let _g = e0.gpu.enter_main()?;
                if graphs_live {
                    if tgraphs.b[0][li].is_none() {
                        tgraphs.b[0][li] = Some(e0.capture_graph_retained_nowarm(|eng| {
                            self.tp2_seg_b(
                                eng,
                                ws0,
                                &ptrs0,
                                &layer.mlp_gate,
                                &mut planes0,
                                &shard.stage0[0],
                                true,
                                eps_m,
                                None,
                            )
                        })?);
                    }
                    tgraphs.b[0][li].as_ref().unwrap().0.launch()?;
                } else {
                    self.tp2_seg_b(
                        e0,
                        ws0,
                        &ptrs0,
                        &layer.mlp_gate,
                        &mut planes0,
                        &shard.stage0[0],
                        true,
                        eps_m,
                        None,
                    )?;
                }
            }

            // ---- phase 3: router host boundary + count-gated MoE tail (graphable via the
            // pack blob: fixed launch shapes, live slot count on device) + join (parity 1) ----
            let route = {
                let _g = e0.gpu.enter_main()?;
                let mixed0 = ws0.take_f32(e0, "hc.mixed", hidden, 0)?;
                let mut router_out = ws0.take_f32(e0, "moe.router", experts, 0)?;
                let none: Option<CudaSlice<u8>> = None;
                let rb = if router_bf16_on() {
                    &moe.router_b16
                } else {
                    &none
                };
                linear_trunk_into(
                    e0,
                    &moe.router,
                    rb,
                    &mixed0,
                    &mut router_out,
                    1,
                    hidden,
                    experts,
                )?;
                let logits = e0.dtoh_view(&router_out.slice(0..experts))?;
                ws0.put_f32("moe.router", router_out);
                ws0.put_f32("hc.mixed", mixed0);
                host_route_softmax_topk(&logits, selected)
            };
            // Split by PLACEMENT (even split when no map is loaded — then rank() is
            // `expert >= experts/2` and local() is `expert - experts/2`, i.e. exactly the
            // arithmetic this site used before the seam existed).
            let place = &tw.place;
            let mut sel0: Vec<i32> = Vec::with_capacity(selected);
            let mut w0: Vec<f32> = Vec::with_capacity(selected);
            let mut sel1: Vec<i32> = Vec::with_capacity(selected);
            let mut w1: Vec<f32> = Vec::with_capacity(selected);
            for &(expert, weight) in &route {
                if place.rank(expert) == 0 {
                    sel0.push(place.local(expert) as i32);
                    w0.push(weight);
                } else {
                    sel1.push(place.local(expert) as i32);
                    w1.push(weight);
                }
            }
            {
                // Route trace + per-rank engagement, in the decode shape (t == 1). The
                // trace rides the readback the host router twin already did.
                let r0: Vec<Vec<(usize, f32)>> = vec![
                    sel0.iter()
                        .zip(&w0)
                        .map(|(&s, &w)| (s as usize, w))
                        .collect(),
                ];
                let r1: Vec<Vec<(usize, f32)>> = vec![
                    sel1.iter()
                        .zip(&w1)
                        .map(|(&s, &w)| (s as usize, w))
                        .collect(),
                ];
                tp2_count_split(&r0, &r1);
                trace_moe_routes(layer.index, 1, std::slice::from_ref(&route));
            }
            match tp2_gate_red()? {
                Tp2GateRed::None => {}
                // Drop the peer's routed contribution entirely.
                Tp2GateRed::SkipPeerMoe => {
                    sel1.clear();
                    w1.clear();
                }
                // Send peer-owned experts to card 0's bank at their peer LOCAL slot: the
                // plausible off-by-remap bug — right magnitudes, wrong experts.
                Tp2GateRed::PeerLocalIds => {
                    sel0.extend(sel1.drain(..));
                    w0.extend(w1.drain(..));
                }
                Tp2GateRed::ReverseePeerWeights => w1.reverse(),
            }
            let max_sel = selected;
            {
                let _g = e1.gpu.enter_main()?;
                ws1.upsert_u8(e1, "moe.pack", &tp2_pack_bytes(&sel1, &w1, max_sel), 0)?;
                if graphs_live {
                    if tgraphs.c[1][li].is_none() {
                        tgraphs.c[1][li] = Some(e1.capture_graph_retained_nowarm(|eng| {
                            self.tp2_seg_c(
                                eng,
                                ws1,
                                (
                                    &tw.moe.gate1.codes,
                                    &tw.moe.gate1.scales,
                                    &tw.moe.gate1.macros_dev,
                                ),
                                (
                                    &tw.moe.up1.codes,
                                    &tw.moe.up1.scales,
                                    &tw.moe.up1.macros_dev,
                                ),
                                (
                                    &tw.moe.down1.codes,
                                    &tw.moe.down1.scales,
                                    &tw.moe.down1.macros_dev,
                                ),
                                ff,
                                max_sel,
                                None,
                                tw.moe.shared_input_gate1.is_some(),
                                shard.stage0_raw[1],
                            )
                        })?);
                    }
                    tgraphs.c[1][li].as_ref().unwrap().0.launch()?;
                } else {
                    self.tp2_seg_c(
                        e1,
                        ws1,
                        (
                            &tw.moe.gate1.codes,
                            &tw.moe.gate1.scales,
                            &tw.moe.gate1.macros_dev,
                        ),
                        (
                            &tw.moe.up1.codes,
                            &tw.moe.up1.scales,
                            &tw.moe.up1.macros_dev,
                        ),
                        (
                            &tw.moe.down1.codes,
                            &tw.moe.down1.scales,
                            &tw.moe.down1.macros_dev,
                        ),
                        ff,
                        max_sel,
                        None,
                        tw.moe.shared_input_gate1.is_some(),
                        shard.stage0_raw[1],
                    )?;
                }
                shard.ev1[1].record(&e1.gpu.stream())?;
            }
            {
                let _g = e0.gpu.enter_main()?;
                let (
                    BankHalf::Nvfp4 {
                        codes: gc,
                        scales: gs,
                        macros_dev: gm,
                        ..
                    },
                    BankHalf::Nvfp4 {
                        codes: uc,
                        scales: us,
                        macros_dev: um,
                        ..
                    },
                    BankHalf::Nvfp4 {
                        codes: dc,
                        scales: ds,
                        macros_dev: dm,
                        ..
                    },
                ) = (&moe.bank.gate, &moe.bank.up, &moe.bank.down)
                else {
                    return Err("qwen4exp_gpu tp2: card0 bank is not NVFP4".into());
                };
                ws0.upsert_u8(e0, "moe.pack", &tp2_pack_bytes(&sel0, &w0, max_sel), 0)?;
                if graphs_live {
                    if tgraphs.c[0][li].is_none() {
                        tgraphs.c[0][li] = Some(e0.capture_graph_retained_nowarm(|eng| {
                            self.tp2_seg_c(
                                eng,
                                ws0,
                                (gc, gs, gm),
                                (uc, us, um),
                                (dc, ds, dm),
                                ff,
                                max_sel,
                                Some((
                                    &tw.moe.shared_gu0_b16,
                                    &tw.moe.shared_down0,
                                    moe.shared_input_gate.as_ref(),
                                    sffh,
                                )),
                                false,
                                shard.stage1_raw[1],
                            )
                        })?);
                    }
                    tgraphs.c[0][li].as_ref().unwrap().0.launch()?;
                } else {
                    self.tp2_seg_c(
                        e0,
                        ws0,
                        (gc, gs, gm),
                        (uc, us, um),
                        (dc, ds, dm),
                        ff,
                        max_sel,
                        Some((
                            &tw.moe.shared_gu0_b16,
                            &tw.moe.shared_down0,
                            moe.shared_input_gate.as_ref(),
                            sffh,
                        )),
                        false,
                        shard.stage1_raw[1],
                    )?;
                }
                shard.ev0[1].record(&e0.gpu.stream())?;
            }
            {
                let _g = e1.gpu.enter_main()?;
                e1.gpu.stream().wait(&shard.ev0[1])?;
                if graphs_live {
                    if tgraphs.d[1][li].is_none() {
                        tgraphs.d[1][li] = Some(e1.capture_graph_retained_nowarm(|eng| {
                            self.tp2_seg_d(eng, ws1, &ptrs1, &mut planes1, &shard.stage1[1], false)
                        })?);
                    }
                    tgraphs.d[1][li].as_ref().unwrap().0.launch()?;
                } else {
                    self.tp2_seg_d(e1, ws1, &ptrs1, &mut planes1, &shard.stage1[1], false)?;
                }
            }
            {
                let _g = e0.gpu.enter_main()?;
                e0.gpu.stream().wait(&shard.ev1[1])?;
                if graphs_live {
                    if tgraphs.d[0][li].is_none() {
                        tgraphs.d[0][li] = Some(e0.capture_graph_retained_nowarm(|eng| {
                            self.tp2_seg_d(eng, ws0, &ptrs0, &mut planes0, &shard.stage0[1], true)
                        })?);
                    }
                    tgraphs.d[0][li].as_ref().unwrap().0.launch()?;
                } else {
                    self.tp2_seg_d(e0, ws0, &ptrs0, &mut planes0, &shard.stage0[1], true)?;
                }
            }
        }

        // Exit mixer (replicated) + vocab-split lm_head, per card (graphable).
        {
            let _g = e1.gpu.enter_main()?;
            if graphs_live {
                if tgraphs.exit[1].is_none() {
                    tgraphs.exit[1] = Some(e1.capture_graph_retained_nowarm(|eng| {
                        self.tp2_seg_exit(
                            eng,
                            ws1,
                            &ptrs1,
                            &shard.exit_gate1,
                            &planes1,
                            &shard.lm_head1,
                            vocab - vsplit,
                            1, // decode_step_tp2 is t == 1 by construction
                        )
                    })?);
                }
                tgraphs.exit[1].as_ref().unwrap().0.launch()?;
            } else {
                self.tp2_seg_exit(
                    e1,
                    ws1,
                    &ptrs1,
                    &shard.exit_gate1,
                    &planes1,
                    &shard.lm_head1,
                    vocab - vsplit,
                    1, // decode_step_tp2 is t == 1 by construction
                )?;
            }
        }
        {
            let _g = e0.gpu.enter_main()?;
            let head0 = self
                .output_b16
                .as_ref()
                .ok_or("qwen4exp_gpu tp2: lm_head has no bf16 twin")?;
            if graphs_live {
                if tgraphs.exit[0].is_none() {
                    tgraphs.exit[0] = Some(e0.capture_graph_retained_nowarm(|eng| {
                        self.tp2_seg_exit(
                            eng,
                            ws0,
                            &ptrs0,
                            &self.exit_mixer,
                            &planes0,
                            head0,
                            vsplit,
                            1, // decode_step_tp2 is t == 1 by construction
                        )
                    })?);
                }
                tgraphs.exit[0].as_ref().unwrap().0.launch()?;
            } else {
                self.tp2_seg_exit(
                    e0,
                    ws0,
                    &ptrs0,
                    &self.exit_mixer,
                    &planes0,
                    head0,
                    vsplit,
                    1, // decode_step_tp2 is t == 1 by construction
                )?;
            }
        }
        let mut out = vec![0.0f32; vocab];
        {
            let _g = e0.gpu.enter_main()?;
            let logits0 = ws0.peek_f32("logits")?;
            let host0 = e0.dtoh_view(&logits0.slice(0..vsplit))?;
            out[..vsplit].copy_from_slice(&host0);
        }
        {
            let _g = e1.gpu.enter_main()?;
            let logits1 = ws1.peek_f32("logits")?;
            let host1 = e1.dtoh_view(&logits1.slice(0..vocab - vsplit))?;
            out[vsplit..].copy_from_slice(&host1);
        }
        for (s, plane) in planes0.into_iter().enumerate() {
            ws0.put_f32(PLANE_SLOTS[s], plane);
        }
        for (s, plane) in planes1.into_iter().enumerate() {
            ws1.put_f32(PLANE_SLOTS[s], plane);
        }
        ws0.put_u64("hc.ptrs", ptrs0);
        ws1.put_u64("hc.ptrs", ptrs1);
        state.pos += 1;
        Ok(out)
    }
}

impl Qwen4ExpGpu {
    /// TP2-NATIVE long-context state (tp2-prefill lane): the per-card halves allocate
    /// DIRECTLY at `capacity` and the single-card KV allocates as a stub — a 1M-token
    /// state never materializes the single-card cache at all (the yarn cell's card-0
    /// blocker). The state is TP2-latched from birth: single-card forwards refuse it
    /// (`state.tp2.is_some()`), and `decode_step_tp2` skips the migration.
    pub fn alloc_state_tp2(
        &self,
        e0: &Engine,
        e1: &Engine,
        shard: &Tp2Shard,
        capacity: usize,
        reserve: usize,
    ) -> Res<Qwen4ExpState> {
        // The single-card side: stub KV, live idx caches (the TP2 indexer runs on
        // card 0 through the same machinery), PLE/GDN states on card 0 unused by the
        // TP2 route but kept tiny.
        let mut state = {
            // Stub the single-card KV by allocating under a 1-token capacity, then
            // restore the real capacity for the mask/meta bookkeeping.
            // The stub's reserve is 1, not `reserve`: `reserve.min(1).max(1)` was written
            // here and is the constant 1 for every usize (clippy::min_max, deny-by-default,
            // which is how it surfaced). Behaviour-identical simplification — the real
            // `reserve` is restored two lines down.
            let mut st = self.alloc_state_reserve(e0, 1, 1, None)?;
            st.capacity = capacity;
            st.reserve = reserve;
            st
        };
        let mut tlayers = Vec::with_capacity(self.layers.len());
        for (layer, tw) in self.layers.iter().zip(shard.layers.iter()) {
            let mk_half = |e: &Engine, hw: &MixerHalfW| -> Res<MixerHalfState> {
                let _g = e.gpu.enter_main()?;
                match hw {
                    MixerHalfW::Gdn(h) => {
                        let conv_dim = 2 * h.nk_h * h.hk + h.nv_h * h.hv;
                        let pad = h.kernel - 1;
                        Ok(MixerHalfState::Gdn {
                            conv: e.zeros(pad * conv_dim)?,
                            state: e.zeros(h.nv_h * h.hv * h.hk)?,
                        })
                    }
                    MixerHalfW::Qsa(h) => {
                        let kv_dim = h.nkv_h * h.hd;
                        let kv = if kv_quant_on() {
                            QsaKvStore::Q8Q5 {
                                k: e.alloc_u8(capacity * q8_row_bytes(kv_dim))?,
                                v: e.alloc_u8(capacity * q5_row_bytes(kv_dim))?,
                            }
                        } else {
                            QsaKvStore::F32 {
                                k: e.zeros(capacity * kv_dim)?,
                                v: e.zeros(capacity * kv_dim)?,
                            }
                        };
                        Ok(MixerHalfState::Qsa { kv })
                    }
                }
            };
            let m0 = mk_half(e0, &tw.mixer0)?;
            let m1 = mk_half(e1, &tw.mixer1)?;
            let ple1 = match layer.ple.as_ref() {
                None => None,
                Some(ple) => {
                    let pad = (ple.plan.conv_kernel as usize - 1) * ple.plan.max_ngram as usize;
                    let _g = e1.gpu.enter_main()?;
                    let mut conv_hist = Vec::with_capacity(self.streams);
                    for _ in 0..self.streams {
                        conv_hist.push(e1.zeros(pad * self.hidden)?);
                    }
                    Some(PleState {
                        conv_hist,
                        ngram_ids: Vec::new(),
                        ngram_history: Vec::new(),
                        ngram_last_eos: -1,
                    })
                }
            };
            tlayers.push(Tp2LayerState { m0, m1, ple1 });
        }
        state.tp2 = Some(Tp2State {
            ws1: StepPool::default(),
            layers: tlayers,
            graphs: Tp2Graphs::default(),
            pf_stage0: None,
            pf_stage1: None,
            pf_stage0_raw: [0; 2],
            pf_stage1_raw: [0; 2],
            pf_rows: 0,
        });
        Ok(state)
    }

    /// TP2 LONG-context chunked prefill: `prefill_extend`'s program on the TP2 route —
    /// KV/state fill happens SHARDED-LOCAL on each card (the yarn cell measured remote
    /// KV at 18x decode collapse; local halves are the 1M route). Returns the LAST
    /// row's logits [vocab].
    pub fn prefill_extend_tp2(
        &self,
        e0: &Engine,
        e1: &Engine,
        shard: &Tp2Shard,
        ids: &[u32],
        state: &mut Qwen4ExpState,
        chunk: usize,
    ) -> Res<Vec<f32>> {
        if ids.is_empty() || chunk == 0 {
            return Err("qwen4exp_gpu: prefill_extend_tp2 needs ids and a chunk size".into());
        }
        let mut last = Vec::new();
        for piece in ids.chunks(chunk) {
            let is_last =
                piece.as_ptr() as usize + piece.len() * 4 == ids.as_ptr() as usize + ids.len() * 4;
            let head = if is_last {
                HeadMode::LastRow
            } else {
                HeadMode::Skip
            };
            last = self.forward_tp2(e0, e1, shard, piece, state, head)?;
        }
        Ok(last)
    }

    /// One TP2 forward over `t` rows (eager; the TP2-prefill program). Replicated
    /// planes + gate reads on both cards, mixer/MoE halves with LOCAL KV/state, join
    /// adds in fixed rank order (the decode joins' determinism argument), the indexer
    /// selection ONCE on card 0 feeding both cards' block-list halves, and the MoE
    /// route split by expert half from the card-0 host route.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tp2(
        &self,
        e0: &Engine,
        e1: &Engine,
        shard: &Tp2Shard,
        ids: &[u32],
        state: &mut Qwen4ExpState,
        head: HeadMode,
    ) -> Res<Vec<f32>> {
        if !trunk_bf16_on() || !hc_fused_gate_on() {
            return Err(
                "qwen4exp_gpu tp2: requires set_trunk_bf16(true) and set_hc_fused_gate(true)"
                    .into(),
            );
        }
        let t = ids.len();
        if t == 0 {
            return Err("qwen4exp_gpu tp2: empty chunk".into());
        }
        if state.pos + t > state.capacity {
            return Err("qwen4exp_gpu: state capacity exceeded".into());
        }
        if state.tp2.is_none() {
            self.tp2_migrate(e0, e1, shard, state)?;
            state.graphs = StepGraphs::default();
        }
        let hidden = self.hidden;
        let vocab = self.vocab;
        let vsplit = shard.vsplit;
        let base_pos = state.pos;
        let reserve = state.reserve;
        state.tokens.extend_from_slice(ids);
        let Qwen4ExpState {
            ref tokens,
            ws: ref mut ws0,
            ref mut tp2,
            layers: ref mut lstates,
            ..
        } = *state;
        let tp2s = tp2.as_mut().expect("alloc'd or migrated above");
        // Prefill join staging: [t*hidden] x 2 per direction, grown to the largest
        // chunk seen (the two-buffer parity proof is the decode staging's, verbatim).
        if tp2s.pf_rows < t {
            {
                let _g = e1.gpu.enter_main()?;
                let s1 = [e1.zeros(t * hidden)?, e1.zeros(t * hidden)?];
                let s = e1.gpu.stream();
                tp2s.pf_stage1_raw = [s1[0].device_ptr(&s).0, s1[1].device_ptr(&s).0];
                tp2s.pf_stage1 = Some(s1);
            }
            {
                let _g = e0.gpu.enter_main()?;
                let s0 = [e0.zeros(t * hidden)?, e0.zeros(t * hidden)?];
                let s = e0.gpu.stream();
                tp2s.pf_stage0_raw = [s0[0].device_ptr(&s).0, s0[1].device_ptr(&s).0];
                tp2s.pf_stage0 = Some(s0);
            }
            tp2s.pf_rows = t;
        }
        let Tp2State {
            ws1,
            layers: tlayers,
            pf_stage0,
            pf_stage1,
            pf_stage0_raw,
            pf_stage1_raw,
            ..
        } = tp2s;
        let pf_stage0 = pf_stage0.as_ref().expect("sized above");
        let pf_stage1 = pf_stage1.as_ref().expect("sized above");
        let resv = reserve.max(t);

        // Entry: embed rows, H2D to BOTH cards' plane slots (replicated planes).
        let mut embedded = vec![0.0f32; t * hidden];
        for (row, &token) in ids.iter().enumerate() {
            let token = token as usize;
            if token >= vocab {
                return Err(format!("qwen4exp_gpu: token {token} out of range").into());
            }
            embedded[row * hidden..(row + 1) * hidden]
                .copy_from_slice(&self.embed_host[token * hidden..(token + 1) * hidden]);
        }
        let mut planes1: Vec<CudaSlice<f32>> = Vec::with_capacity(self.streams);
        let ptrs1 = {
            let _g = e1.gpu.enter_main()?;
            let embedded_dev = ws1.take_f32_h2d(e1, "entry.embed", &embedded, resv * hidden)?;
            for s in 0..self.streams {
                let mut plane = ws1.take_f32(e1, PLANE_SLOTS[s], t * hidden, resv * hidden)?;
                e1.copy_into(&mut plane, 0, &embedded_dev, t * hidden)?;
                planes1.push(plane);
            }
            ws1.put_f32("entry.embed", embedded_dev);
            let ptr_vals: Vec<u64> = {
                let stream = e1.gpu.stream();
                planes1.iter().map(|p| p.device_ptr(&stream).0).collect()
            };
            ws1.take_u64_h2d(e1, "hc.ptrs", &ptr_vals, 0)?
        };
        let mut planes0: Vec<CudaSlice<f32>> = Vec::with_capacity(self.streams);
        let ptrs0 = {
            let _g = e0.gpu.enter_main()?;
            let embedded_dev = ws0.take_f32_h2d(e0, "entry.embed", &embedded, resv * hidden)?;
            for s in 0..self.streams {
                let mut plane = ws0.take_f32(e0, PLANE_SLOTS[s], t * hidden, resv * hidden)?;
                e0.copy_into(&mut plane, 0, &embedded_dev, t * hidden)?;
                planes0.push(plane);
            }
            ws0.put_f32("entry.embed", embedded_dev);
            let ptr_vals: Vec<u64> = {
                let stream = e0.gpu.stream();
                planes0.iter().map(|p| p.device_ptr(&stream).0).collect()
            };
            ws0.take_u64_h2d(e0, "hc.ptrs", &ptr_vals, 0)?
        };

        for (li, layer) in self.layers.iter().enumerate() {
            let lstate = &mut lstates[li];
            let tw = &shard.layers[li];
            let ts = &mut tlayers[li];
            let eps_a = layer.eps_attn;
            let eps_m = layer.eps_mlp;
            let moe = &layer.moe;
            let ff = moe.plan.expert_intermediate_size as usize;
            let experts = moe.plan.expert_count as usize;
            let selected = moe.plan.experts_per_token as usize;
            let sff = moe
                .plan
                .shared
                .as_ref()
                .map(|s| s.intermediate_size as usize)
                .unwrap_or(0);
            let sffh = sff / 2;

            // ---- PLE (wide-stream add), replicated on both cards ----
            if let (Some(ple), Some(ps)) = (layer.ple.as_ref(), lstate.ple.as_mut()) {
                let _g = e0.gpu.enter_main()?;
                self.ple_block(
                    e0,
                    layer,
                    ple,
                    &ple.table,
                    ps,
                    &mut planes0,
                    tokens,
                    t,
                    false,
                    None,
                )?;
            }
            if let (Some(ple1), Some(ps1)) = (tw.ple1.as_ref(), ts.ple1.as_mut()) {
                let table = &layer.ple.as_ref().expect("ple plan").table;
                let _g = e1.gpu.enter_main()?;
                self.ple_block(
                    e1,
                    layer,
                    ple1,
                    table,
                    ps1,
                    &mut planes1,
                    tokens,
                    t,
                    false,
                    None,
                )?;
            }

            // ---- phase 1: attn gate + mixer halves + join push (parity 0) ----
            match (&layer.mixer, &tw.mixer0, &tw.mixer1) {
                (MixerW::Gdn(_), MixerHalfW::Gdn(h0), MixerHalfW::Gdn(h1)) => {
                    {
                        let _g = e1.gpu.enter_main()?;
                        let (mixed1, inj1) = self.gate_read(
                            e1,
                            ws1,
                            &ptrs1,
                            &tw.attn_gate1,
                            &planes1,
                            t,
                            eps_a,
                            false,
                        )?;
                        let p1 =
                            self.gdn_forward_half(e1, ws1, eps_a, h1, &mixed1, &mut ts.m1, t)?;
                        ws1.put_f32("hc.mixed", mixed1);
                        launch_push(e1, &p1, pf_stage0_raw[0], t * hidden)?;
                        ws1.put_f32("mixer.out", p1);
                        put_inject(ws1, inj1);
                        shard.ev1[0].record(&e1.gpu.stream())?;
                    }
                    {
                        let _g = e0.gpu.enter_main()?;
                        let (mixed0, inj0) = self.gate_read(
                            e0,
                            ws0,
                            &ptrs0,
                            &layer.attn_gate,
                            &planes0,
                            t,
                            eps_a,
                            false,
                        )?;
                        let p0 =
                            self.gdn_forward_half(e0, ws0, eps_a, h0, &mixed0, &mut ts.m0, t)?;
                        ws0.put_f32("hc.mixed", mixed0);
                        launch_push(e0, &p0, pf_stage1_raw[0], t * hidden)?;
                        ws0.put_f32("mixer.out", p0);
                        put_inject(ws0, inj0);
                        shard.ev0[0].record(&e0.gpu.stream())?;
                    }
                }
                (MixerW::Qsa(qsa), MixerHalfW::Qsa(h0), MixerHalfW::Qsa(h1)) => {
                    let (q1, g1, inj1) = {
                        let _g = e1.gpu.enter_main()?;
                        let (mixed1, inj1) = self.gate_read(
                            e1,
                            ws1,
                            &ptrs1,
                            &tw.attn_gate1,
                            &planes1,
                            t,
                            eps_a,
                            false,
                        )?;
                        let (q1, g1) = self
                            .qsa_half_proj(e1, ws1, eps_a, h1, &mixed1, &mut ts.m1, base_pos, t)?;
                        ws1.put_f32("hc.mixed", mixed1);
                        (q1, g1, inj1)
                    };
                    let (sels, q0, g0, inj0) = {
                        let _g = e0.gpu.enter_main()?;
                        let (mixed0, inj0) = self.gate_read(
                            e0,
                            ws0,
                            &ptrs0,
                            &layer.attn_gate,
                            &planes0,
                            t,
                            eps_a,
                            false,
                        )?;
                        let (q0, g0) = self
                            .qsa_half_proj(e0, ws0, eps_a, h0, &mixed0, &mut ts.m0, base_pos, t)?;
                        let MixerState::Qsa {
                            raw_keys,
                            pooled_keys,
                            pooled_dev,
                            pooled_dev_rows,
                            raw_dev,
                            raw_dev_rows,
                            idx_audit,
                            ..
                        } = &mut lstate.mixer
                        else {
                            return Err("qwen4exp_gpu tp2: QSA layer without raw-key cache".into());
                        };
                        let sels = self.qsa_update_select(
                            e0,
                            ws0,
                            qsa,
                            eps_a,
                            &mixed0,
                            raw_keys,
                            pooled_keys,
                            pooled_dev,
                            pooled_dev_rows,
                            raw_dev,
                            raw_dev_rows,
                            idx_audit.as_mut(),
                            base_pos,
                            t,
                            0,
                            false,
                        )?;
                        ws0.put_f32("hc.mixed", mixed0);
                        (sels, q0, g0, inj0)
                    };
                    let t_kv = base_pos + t;
                    let block_size = qsa.overlay.block_size as usize;
                    let (pos_flat, meta, max_count) = rowsel_positions(&sels, block_size);
                    {
                        let _g = e1.gpu.enter_main()?;
                        let pos_dev = ws1.take_i32(e1, "qsa.selpos", &pos_flat, 0)?;
                        let meta_dev = ws1.take_i32(e1, "qsa.selmeta", &meta, 0)?;
                        let p1 = self.qsa_half_attend(
                            e1, ws1, h1, &ts.m1, q1, g1, &pos_dev, &meta_dev, max_count, t, t_kv,
                        )?;
                        ws1.put_i32("qsa.selpos", pos_dev);
                        ws1.put_i32("qsa.selmeta", meta_dev);
                        launch_push(e1, &p1, pf_stage0_raw[0], t * hidden)?;
                        ws1.put_f32("mixer.out", p1);
                        put_inject(ws1, inj1);
                        shard.ev1[0].record(&e1.gpu.stream())?;
                    }
                    {
                        let _g = e0.gpu.enter_main()?;
                        let pos_dev = ws0.take_i32(e0, "qsa.selpos", &pos_flat, 0)?;
                        let meta_dev = ws0.take_i32(e0, "qsa.selmeta", &meta, 0)?;
                        let p0 = self.qsa_half_attend(
                            e0, ws0, h0, &ts.m0, q0, g0, &pos_dev, &meta_dev, max_count, t, t_kv,
                        )?;
                        ws0.put_i32("qsa.selpos", pos_dev);
                        ws0.put_i32("qsa.selmeta", meta_dev);
                        launch_push(e0, &p0, pf_stage1_raw[0], t * hidden)?;
                        ws0.put_f32("mixer.out", p0);
                        put_inject(ws0, inj0);
                        shard.ev0[0].record(&e0.gpu.stream())?;
                    }
                }
                _ => return Err("qwen4exp_gpu tp2: mixer/shard shape mismatch".into()),
            }
            {
                let _g = e0.gpu.enter_main()?;
                e0.gpu.stream().wait(&shard.ev1[0])?;
            }
            {
                let _g = e1.gpu.enter_main()?;
                e1.gpu.stream().wait(&shard.ev0[0])?;
            }

            // ---- phase 2: join add (fixed rank order) + gate_write + mlp gate_read ----
            let join_write = |e: &Engine,
                              ws: &mut StepPool,
                              ptrs: &CudaSlice<u64>,
                              planes: &mut [CudaSlice<f32>],
                              stage: &CudaSlice<f32>,
                              rank0: bool|
             -> Res<()> {
                let p = ws.take_f32(e, "mixer.out", t * hidden, 0)?;
                let mut out = ws.take_f32(e, "join.out", t * hidden, 0)?;
                if rank0 {
                    e.add(&p, stage, &mut out, t * hidden)?;
                } else {
                    e.add(stage, &p, &mut out, t * hidden)?;
                }
                let inj = take_inject(e, ws, self.streams, t)?;
                self.gate_write(e, planes, ptrs, &out, &inj, t)?;
                ws.put_f32("mixer.out", p);
                ws.put_f32("join.out", out);
                put_inject(ws, inj);
                Ok(())
            };
            {
                let _g = e1.gpu.enter_main()?;
                join_write(e1, ws1, &ptrs1, &mut planes1, &pf_stage1[0], false)?;
            }
            {
                let _g = e0.gpu.enter_main()?;
                join_write(e0, ws0, &ptrs0, &mut planes0, &pf_stage0[0], true)?;
            }

            // ---- phase 3: mlp gate + MoE halves + shared halves + join (parity 1) ----
            let mixed1 = {
                let _g = e1.gpu.enter_main()?;
                let (mixed1, injm1) =
                    self.gate_read(e1, ws1, &ptrs1, &tw.mlp_gate1, &planes1, t, eps_m, false)?;
                put_inject(ws1, injm1);
                mixed1
            };
            let mixed0 = {
                let _g = e0.gpu.enter_main()?;
                let (mixed0, injm0) =
                    self.gate_read(e0, ws0, &ptrs0, &layer.mlp_gate, &planes0, t, eps_m, false)?;
                put_inject(ws0, injm0);
                mixed0
            };
            // Route on card 0 (host twin — TP2 keeps host expert ids by construction),
            // split by expert half.
            let routes: Vec<Vec<(usize, f32)>> = {
                let _g = e0.gpu.enter_main()?;
                let mut router_out = ws0.take_f32(e0, "moe.router", t * experts, 0)?;
                let none: Option<CudaSlice<u8>> = None;
                let rb = if router_bf16_on() {
                    &moe.router_b16
                } else {
                    &none
                };
                linear_trunk_into(
                    e0,
                    &moe.router,
                    rb,
                    &mixed0,
                    &mut router_out,
                    t,
                    hidden,
                    experts,
                )?;
                let logits = e0.dtoh_view(&router_out.slice(0..t * experts))?;
                ws0.put_f32("moe.router", router_out);
                let mut routes = Vec::with_capacity(t);
                for token in 0..t {
                    routes.push(host_route_softmax_topk(
                        &logits[token * experts..(token + 1) * experts],
                        selected,
                    ));
                }
                routes
            };
            // Split by PLACEMENT (see the decode site); with no map loaded this is the
            // even split and reproduces the previous `eid < e_half` / `eid - e_half`
            // arithmetic exactly.
            let place = &tw.place;
            let split_half = |home: bool| -> Vec<Vec<(usize, f32)>> {
                routes
                    .iter()
                    .map(|r| {
                        r.iter()
                            .filter(|&&(eid, _)| (place.rank(eid) == 0) == home)
                            .map(|&(eid, w)| (place.local(eid), w))
                            .collect()
                    })
                    .collect()
            };
            let mut routes0 = split_half(true);
            let mut routes1 = split_half(false);
            // Per-rank engagement + the shared-format route trace, in the PREFILL shape
            // (one line per (layer, forward) carrying this chunk's t rows of picks).
            tp2_count_split(&routes0, &routes1);
            trace_moe_routes(layer.index, t, &routes);
            match tp2_gate_red()? {
                Tp2GateRed::None => {}
                Tp2GateRed::SkipPeerMoe => routes1.iter_mut().for_each(|r| r.clear()),
                Tp2GateRed::PeerLocalIds => {
                    for (r0, r1) in routes0.iter_mut().zip(routes1.iter_mut()) {
                        r0.append(r1);
                    }
                }
                Tp2GateRed::ReverseePeerWeights => {
                    for r in routes1.iter_mut() {
                        let n = r.len();
                        for i in 0..n / 2 {
                            let (a, b) = (r[i].1, r[n - 1 - i].1);
                            r[i].1 = b;
                            r[n - 1 - i].1 = a;
                        }
                    }
                }
            }
            let (routes0, routes1) = (routes0, routes1);
            // Card 1: routed half over the bank half (local ids) + shared suffix half.
            {
                let _g = e1.gpu.enter_main()?;
                let mut out1 = self.tp2_moe_rows(
                    e1,
                    ws1,
                    (
                        &tw.moe.gate1.codes,
                        &tw.moe.gate1.scales,
                        &tw.moe.gate1.macros_dev,
                    ),
                    (
                        &tw.moe.up1.codes,
                        &tw.moe.up1.scales,
                        &tw.moe.up1.macros_dev,
                    ),
                    (
                        &tw.moe.down1.codes,
                        &tw.moe.down1.scales,
                        &tw.moe.down1.macros_dev,
                    ),
                    &routes1,
                    &mixed1,
                    t,
                    ff,
                )?;
                let (sh, g) = self.tp2_shared_half(
                    e1,
                    ws1,
                    &tw.moe.shared_gu1_b16,
                    &tw.moe.shared_down1,
                    tw.moe.shared_input_gate1.as_ref(),
                    &mixed1,
                    sffh,
                    t,
                )?;
                match g.as_ref() {
                    Some(g) => e1.add_scaled_rows(&sh, g, &mut out1, hidden, t)?,
                    None => {
                        let mut summed = ws1.take_f32(e1, "moe.sum", t * hidden, 0)?;
                        e1.add(&out1, &sh, &mut summed, t * hidden)?;
                        ws1.put_f32("moe.out", out1);
                        out1 = summed;
                    }
                }
                ws1.put_f32("moe.sh_down", sh);
                if let Some(g) = g {
                    ws1.put_f32("moe.g", g);
                }
                launch_push(e1, &out1, pf_stage0_raw[1], t * hidden)?;
                ws1.put_f32("moe.out", out1);
                ws1.put_f32("hc.mixed", mixed1);
                shard.ev1[1].record(&e1.gpu.stream())?;
            }
            // Card 0: routed half over the FULL resident bank (absolute ids < E/2) +
            // shared prefix half.
            {
                let _g = e0.gpu.enter_main()?;
                let (
                    BankHalf::Nvfp4 {
                        codes: gc,
                        scales: gs,
                        macros_dev: gm,
                        ..
                    },
                    BankHalf::Nvfp4 {
                        codes: uc,
                        scales: us,
                        macros_dev: um,
                        ..
                    },
                    BankHalf::Nvfp4 {
                        codes: dc,
                        scales: ds,
                        macros_dev: dm,
                        ..
                    },
                ) = (&moe.bank.gate, &moe.bank.up, &moe.bank.down)
                else {
                    return Err("qwen4exp_gpu tp2: card0 bank is not NVFP4".into());
                };
                let mut out0 = self.tp2_moe_rows(
                    e0,
                    ws0,
                    (gc, gs, gm),
                    (uc, us, um),
                    (dc, ds, dm),
                    &routes0,
                    &mixed0,
                    t,
                    ff,
                )?;
                let (sh, g) = self.tp2_shared_half(
                    e0,
                    ws0,
                    &tw.moe.shared_gu0_b16,
                    &tw.moe.shared_down0,
                    moe.shared_input_gate.as_ref(),
                    &mixed0,
                    sffh,
                    t,
                )?;
                match g.as_ref() {
                    Some(g) => e0.add_scaled_rows(&sh, g, &mut out0, hidden, t)?,
                    None => {
                        let mut summed = ws0.take_f32(e0, "moe.sum", t * hidden, 0)?;
                        e0.add(&out0, &sh, &mut summed, t * hidden)?;
                        ws0.put_f32("moe.out", out0);
                        out0 = summed;
                    }
                }
                ws0.put_f32("moe.sh_down", sh);
                if let Some(g) = g {
                    ws0.put_f32("moe.g", g);
                }
                launch_push(e0, &out0, pf_stage1_raw[1], t * hidden)?;
                ws0.put_f32("moe.out", out0);
                ws0.put_f32("hc.mixed", mixed0);
                shard.ev0[1].record(&e0.gpu.stream())?;
            }
            {
                let _g = e1.gpu.enter_main()?;
                e1.gpu.stream().wait(&shard.ev0[1])?;
                let p = ws1.take_f32(e1, "moe.out", t * hidden, 0)?;
                let mut out = ws1.take_f32(e1, "join.out", t * hidden, 0)?;
                e1.add(&pf_stage1[1], &p, &mut out, t * hidden)?;
                let inj = take_inject(e1, ws1, self.streams, t)?;
                self.gate_write(e1, &mut planes1, &ptrs1, &out, &inj, t)?;
                ws1.put_f32("moe.out", p);
                ws1.put_f32("join.out", out);
                put_inject(ws1, inj);
            }
            {
                let _g = e0.gpu.enter_main()?;
                e0.gpu.stream().wait(&shard.ev1[1])?;
                let p = ws0.take_f32(e0, "moe.out", t * hidden, 0)?;
                let mut out = ws0.take_f32(e0, "join.out", t * hidden, 0)?;
                e0.add(&p, &pf_stage0[1], &mut out, t * hidden)?;
                let inj = take_inject(e0, ws0, self.streams, t)?;
                self.gate_write(e0, &mut planes0, &ptrs0, &out, &inj, t)?;
                ws0.put_f32("moe.out", p);
                ws0.put_f32("join.out", out);
                put_inject(ws0, inj);
            }
        }

        // Exit: Skip on interior chunks; LastRow copies each plane's final row into
        // t == 1 exit slots and runs the decode exit segment on them; All runs the exit
        // segment over ALL t rows straight off the planes.
        //
        // `All` used to fall through to the LastRow body, so a caller asking for every row
        // got exactly one and no error. That is the failure mode the loud-failure law is
        // about: the TP2 class gate's whole PRIME regime is "compare EVERY row of a full-head
        // forward", and it could not have done that — it only surfaced because the gate
        // length-checks single-card logits against TP2 logits before comparing
        // ("single-card produced 2483200 logits, TP2 248320"). Without that check the gate
        // would have compared one row and reported a t>=2 verdict.
        //
        // Cost note (why this stays an instrument, not a serving path): a [t, vocab] block is
        // t * 248320 * 4 bytes, so it is ~9.9 MB at the gate's 10-token probe and gigabytes at
        // a long-context chunk. Chunked prefill therefore still uses LastRow, exactly as the
        // single-card path does for the same reason.
        let head_rows = match head {
            HeadMode::All => t,
            _ => 1,
        };
        let mut out = vec![
            0.0f32;
            if head == HeadMode::Skip {
                0
            } else {
                head_rows * vocab
            }
        ];
        if head != HeadMode::Skip {
            {
                let _g = e1.gpu.enter_main()?;
                // All: the planes already hold every row, so the exit reads them directly
                // with the pointer array the trunk built. LastRow: copy each plane's final
                // row into the t == 1 exit slots (the decode-shaped exit).
                let mut exit_planes: Vec<CudaSlice<f32>> = Vec::with_capacity(self.streams);
                if head != HeadMode::All {
                    for (s, plane) in planes1.iter().enumerate() {
                        let mut row = ws1.take_f32(e1, EXIT_PLANE_SLOTS[s], hidden, hidden)?;
                        e1.copy_range_into(&mut row, 0, plane, (t - 1) * hidden, hidden)?;
                        exit_planes.push(row);
                    }
                }
                let use_planes: &[CudaSlice<f32>] = if head == HeadMode::All {
                    &planes1
                } else {
                    &exit_planes
                };
                let ptr_vals: Vec<u64> = {
                    let stream = e1.gpu.stream();
                    use_planes.iter().map(|p| p.device_ptr(&stream).0).collect()
                };
                let eptrs = ws1.take_u64_h2d(e1, "exit.ptrs", &ptr_vals, 0)?;
                self.tp2_seg_exit(
                    e1,
                    ws1,
                    &eptrs,
                    &shard.exit_gate1,
                    use_planes,
                    &shard.lm_head1,
                    vocab - vsplit,
                    head_rows,
                )?;
                ws1.put_u64("exit.ptrs", eptrs);
                for (s, p) in exit_planes.into_iter().enumerate() {
                    ws1.put_f32(EXIT_PLANE_SLOTS[s], p);
                }
                let logits1 = ws1.peek_f32("logits")?;
                let half1 = vocab - vsplit;
                let host1 = e1.dtoh_view(&logits1.slice(0..head_rows * half1))?;
                // This card owns the HIGH column half of every row, so a [rows, half1]
                // block scatters into [rows, vocab] one row at a time.
                for r in 0..head_rows {
                    out[r * vocab + vsplit..(r + 1) * vocab]
                        .copy_from_slice(&host1[r * half1..(r + 1) * half1]);
                }
            }
            {
                let _g = e0.gpu.enter_main()?;
                let head0 = self
                    .output_b16
                    .as_ref()
                    .ok_or("qwen4exp_gpu tp2: lm_head has no bf16 twin")?;
                let mut exit_planes: Vec<CudaSlice<f32>> = Vec::with_capacity(self.streams);
                if head != HeadMode::All {
                    for (s, plane) in planes0.iter().enumerate() {
                        let mut row = ws0.take_f32(e0, EXIT_PLANE_SLOTS[s], hidden, hidden)?;
                        e0.copy_range_into(&mut row, 0, plane, (t - 1) * hidden, hidden)?;
                        exit_planes.push(row);
                    }
                }
                let use_planes: &[CudaSlice<f32>] = if head == HeadMode::All {
                    &planes0
                } else {
                    &exit_planes
                };
                let ptr_vals: Vec<u64> = {
                    let stream = e0.gpu.stream();
                    use_planes.iter().map(|p| p.device_ptr(&stream).0).collect()
                };
                let eptrs = ws0.take_u64_h2d(e0, "exit.ptrs", &ptr_vals, 0)?;
                self.tp2_seg_exit(
                    e0,
                    ws0,
                    &eptrs,
                    &self.exit_mixer,
                    use_planes,
                    head0,
                    vsplit,
                    head_rows,
                )?;
                ws0.put_u64("exit.ptrs", eptrs);
                for (s, p) in exit_planes.into_iter().enumerate() {
                    ws0.put_f32(EXIT_PLANE_SLOTS[s], p);
                }
                let logits0 = ws0.peek_f32("logits")?;
                let host0 = e0.dtoh_view(&logits0.slice(0..head_rows * vsplit))?;
                // This card owns the LOW column half of every row.
                for r in 0..head_rows {
                    out[r * vocab..r * vocab + vsplit]
                        .copy_from_slice(&host0[r * vsplit..(r + 1) * vsplit]);
                }
            }
        } else {
            // Establish a host boundary per chunk so the chunk loop cannot run the
            // host arbitrarily far ahead of both devices.
            {
                let _g = e0.gpu.enter_main()?;
                e0.gpu.stream().synchronize()?;
            }
            {
                let _g = e1.gpu.enter_main()?;
                e1.gpu.stream().synchronize()?;
            }
        }
        for (s, plane) in planes0.into_iter().enumerate() {
            ws0.put_f32(PLANE_SLOTS[s], plane);
        }
        for (s, plane) in planes1.into_iter().enumerate() {
            ws1.put_f32(PLANE_SLOTS[s], plane);
        }
        ws0.put_u64("hc.ptrs", ptrs0);
        ws1.put_u64("hc.ptrs", ptrs1);
        state.pos += t;
        Ok(out)
    }

    /// Grouped routed-experts half at t rows (TP2 prefill): the single-card grouped
    /// prefill program (SLOT_CAP sub-batching, absolute-token maps, per-token
    /// slot-ordered combines) over THIS CARD's bank (card 0 = the full resident bank
    /// with absolute ids < E/2; card 1 = the half bank with local ids). Tokens with no
    /// experts on this card keep their zero rows (the join sums the halves).
    #[allow(clippy::too_many_arguments)]
    fn tp2_moe_rows(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        gate: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
        up: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
        down: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
        routes: &[Vec<(usize, f32)>],
        mixed: &CudaSlice<f32>,
        t: usize,
        ff: usize,
    ) -> Res<CudaSlice<f32>> {
        let hidden = self.hidden;
        if !(sel_gufuse_on() && hidden % 32 == 0 && ff % 4 == 0) {
            return Err(
                "qwen4exp_gpu tp2: prefill MoE needs the gufuse geometry (hidden%32, ff%4)".into(),
            );
        }
        let mut out = ws.take_f32(e, "moe.out", t * hidden, 0)?;
        {
            let mut view = out.slice_mut(0..t * hidden);
            e.memset_zeros_view(&mut view)?;
        }
        const SLOT_CAP: usize = 8192;
        let mut tok0 = 0usize;
        while tok0 < t {
            // Advance until the slot budget fills (routes are variable-length halves).
            let mut tok_n = 0usize;
            let mut slots = 0usize;
            while tok0 + tok_n < t {
                let n = routes[tok0 + tok_n].len();
                if tok_n > 0 && slots + n > SLOT_CAP {
                    break;
                }
                slots += n;
                tok_n += 1;
            }
            let batch = &routes[tok0..tok0 + tok_n];
            let mut sel_all: Vec<i32> = Vec::with_capacity(slots);
            let mut w_all: Vec<f32> = Vec::with_capacity(slots);
            let mut tok_all: Vec<i32> = Vec::with_capacity(slots);
            let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(tok_n);
            for (i, route) in batch.iter().enumerate() {
                ranges.push((sel_all.len(), route.len()));
                for &(eid, wgt) in route {
                    sel_all.push(eid as i32);
                    w_all.push(wgt);
                    tok_all.push((tok0 + i) as i32);
                }
            }
            let s_total = sel_all.len();
            if s_total > 0 {
                let sel = ws.take_i32(e, "moe.sel", &sel_all, 0)?;
                let w_dev = ws.take_f32_h2d(e, "moe.w", &w_all, 0)?;
                let tokm = ws.take_i32(e, "moe.tok", &tok_all, 0)?;
                let mut act = ws.take_f32(e, "moe.act", s_total * ff, 0)?;
                launch_nvfp4_sel_gu_silu(
                    e,
                    gate,
                    up,
                    Some(&sel),
                    0,
                    s_total,
                    mixed,
                    &mut act,
                    hidden,
                    ff,
                    Some((&tokm, hidden)),
                )?;
                let mut partial = ws.take_f32(e, "moe.partial", s_total * hidden, 0)?;
                launch_nvfp4_sel_matvec(
                    e,
                    down.0,
                    down.1,
                    down.2,
                    &sel,
                    &act,
                    &mut partial,
                    s_total,
                    ff,
                    hidden,
                    ff,
                )?;
                for (i, &(start, len)) in ranges.iter().enumerate() {
                    if len > 0 {
                        launch_axpy_rows_seq_at(
                            e,
                            &partial,
                            start,
                            &w_dev,
                            start,
                            &mut out,
                            tok0 + i,
                            hidden,
                            len,
                        )?;
                    }
                }
                ws.put_i32("moe.sel", sel);
                ws.put_i32("moe.tok", tokm);
                ws.put_f32("moe.w", w_dev);
                ws.put_f32("moe.act", act);
                ws.put_f32("moe.partial", partial);
            }
            tok0 += tok_n;
        }
        Ok(out)
    }

    /// TP2 segment A (GDN layers, graphable): attn gate_read + GDN half + join push;
    /// parks the partial in "mixer.out" and the inject scalars in their slots.
    #[allow(clippy::too_many_arguments)]
    fn tp2_gdn_seg_a(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        attn_gate: &GateW,
        h: &GdnHalfW,
        hstate: &mut MixerHalfState,
        planes: &[CudaSlice<f32>],
        eps: f32,
        push_raw: u64,
    ) -> Res<()> {
        let (mixed, inj) = self.gate_read(e, ws, ptrs, attn_gate, planes, 1, eps, false)?;
        let p = self.gdn_forward_half(e, ws, eps, h, &mixed, hstate, 1)?;
        ws.put_f32("hc.mixed", mixed);
        launch_push(e, &p, push_raw, self.hidden)?;
        ws.put_f32("mixer.out", p);
        put_inject(ws, inj);
        Ok(())
    }

    /// TP2 segment B (all layers, graphable): mixer join add (SAME rank order on both
    /// cards) + gate_write + mlp gate_read (+ optional card-1 shared-half prestage,
    /// parked in "tp2.sh"/"tp2.shg"); parks the mlp mixed in "hc.mixed" and the mlp
    /// inject in its slots.
    #[allow(clippy::too_many_arguments)]
    fn tp2_seg_b(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        mlp_gate: &GateW,
        planes: &mut [CudaSlice<f32>],
        stage_recv: &CudaSlice<f32>,
        rank0: bool,
        eps_m: f32,
        shared: Option<(
            &CudaSlice<u8>,
            &CudaSlice<u8>,
            Option<&CudaSlice<f32>>,
            usize,
        )>,
    ) -> Res<()> {
        let hidden = self.hidden;
        let p = ws.take_f32(e, "mixer.out", hidden, 0)?;
        let mut out = ws.take_f32(e, "join.out", hidden, 0)?;
        if rank0 {
            e.add(&p, stage_recv, &mut out, hidden)?;
        } else {
            e.add(stage_recv, &p, &mut out, hidden)?;
        }
        let inj = take_inject(e, ws, self.streams, 1)?;
        self.gate_write(e, planes, ptrs, &out, &inj, 1)?;
        ws.put_f32("mixer.out", p);
        ws.put_f32("join.out", out);
        put_inject(ws, inj);
        let (mixed, injm) = self.gate_read(e, ws, ptrs, mlp_gate, planes, 1, eps_m, false)?;
        if let Some((gu_b16, d_b16, ig, sffh)) = shared {
            let (sh, gg) = self.tp2_shared_half(e, ws, gu_b16, d_b16, ig, &mixed, sffh, 1)?;
            // Slot-cycle invariant: park under the SAME names tp2_shared_half takes
            // from ("moe.sh_down"/"moe.g"), or the next capture of this segment would
            // allocate inside the capture region (graph mem node).
            ws.put_f32("moe.sh_down", sh);
            if let Some(gg) = gg {
                ws.put_f32("moe.g", gg);
            }
        }
        ws.put_f32("hc.mixed", mixed);
        put_inject(ws, injm);
        Ok(())
    }

    /// TP2 exit segment (graphable): exit mixer read + this card's lm_head half into the
    /// parked "logits" slot.
    #[allow(clippy::too_many_arguments)]
    /// TP2 exit segment (mixer + this card's lm_head column half) over `rows` rows.
    ///
    /// `rows` used to be hardcoded to 1, which made `HeadMode::All` silently identical to
    /// `HeadMode::LastRow` in the TP2 forward — see the caller for why that was a defect
    /// and not merely a limitation. Both `gate_read_inner` and `launch_qmatvec_bf16w`
    /// already take a row count (the kernel's grid y-dim IS `t`, striding `x` by
    /// `x_tstride`), so this is a parameter that was never threaded, not new math: at
    /// `rows == 1` the launch arguments are byte-for-byte the ones this function used
    /// before, which is what makes the decode path a control rather than a hope.
    fn tp2_seg_exit(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        gate: &GateW,
        planes: &[CudaSlice<f32>],
        head_b16: &CudaSlice<u8>,
        out_f: usize,
        rows: usize,
    ) -> Res<()> {
        let x = self
            .gate_read_inner(e, ws, ptrs, gate, planes, rows, self.exit_eps, false, false)?
            .0;
        let mut logits = ws.take_f32(e, "logits", rows * out_f, rows * out_f)?;
        launch_qmatvec_bf16w(
            e,
            head_b16,
            &x,
            &mut logits,
            self.hidden,
            out_f,
            rows,
            1,
            0,
            0,
            self.hidden,
            0,
        )?;
        ws.put_f32("hc.mixed", x);
        ws.put_f32("logits", logits);
        Ok(())
    }
}

/// Launch the count-gated grouped sel matvec (`_v3c`, fixed grid over `max_sel` slots,
/// live count from the pack blob). TP2 graph segments only; geometry must admit the
/// 4-row kernel (the artifact does).
#[allow(clippy::too_many_arguments)]
fn launch_nvfp4_sel_matvec_pack(
    e: &Engine,
    codes: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    macros_dev: &CudaSlice<f32>,
    pack_raw: u64,
    max_sel: usize,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    in_f: usize,
    out_f: usize,
    x_stride: usize,
) -> Res<()> {
    if in_f % 32 != 0 || out_f % 4 != 0 {
        return Err(
            "qmatvec_nvfp4_modelopt_sel_f32_v3c: geometry needs in_f%32==0 && out_f%4==0".into(),
        );
    }
    let f = e.func("qmatvec_nvfp4_modelopt_sel_f32_v3c");
    let cfg = LaunchConfig {
        grid_dim: ((out_f / 4) as u32, max_sel as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inf, outf, ms) = (in_f as i32, out_f as i32, max_sel as i32);
    let xs = x_stride as i64;
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(codes)
        .arg(scales)
        .arg(macros_dev)
        .arg(&pack_raw)
        .arg(&ms)
        .arg(x)
        .arg(y)
        .arg(&inf)
        .arg(&outf)
        .arg(&xs);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

fn launch_axpy_rows_seq_pack(
    e: &Engine,
    x: &CudaSlice<f32>,
    pack_raw: u64,
    max_sel: usize,
    y: &mut CudaSlice<f32>,
    width: usize,
) -> Res<()> {
    let f = e.func("axpy_rows_seq_pack_f32");
    let cfg = LaunchConfig::for_num_elems(width as u32);
    let (ms, wi) = (max_sel as i32, width as i32);
    let stream = e.gpu.stream();
    let mut b = stream.launch_builder(&f);
    b.arg(x).arg(&pack_raw).arg(&ms).arg(y).arg(&wi);
    unsafe {
        b.launch(cfg)?;
    }
    Ok(())
}

/// Build the pack blob: [max_sel i32 sel padded][max_sel f32 w padded][i32 count].
fn tp2_pack_bytes(sel: &[i32], w: &[f32], max_sel: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((2 * max_sel + 1) * 4);
    for i in 0..max_sel {
        out.extend_from_slice(&sel.get(i).copied().unwrap_or(0).to_le_bytes());
    }
    for i in 0..max_sel {
        out.extend_from_slice(&w.get(i).copied().unwrap_or(0.0).to_le_bytes());
    }
    out.extend_from_slice(&(sel.len() as i32).to_le_bytes());
    out
}

impl Qwen4ExpGpu {
    /// TP2 segment C (graphable): count-gated routed half over the pack blob + shared
    /// add + join push. Card 1 takes its prestaged shared parts ("moe.sh_down"/"moe.g",
    /// parked by seg B); card 0 computes its shared half here. Parks the MoE partial in
    /// "moe.out".
    #[allow(clippy::too_many_arguments)]
    fn tp2_seg_c(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        gate: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
        up: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
        down: (&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<f32>),
        ff: usize,
        max_sel: usize,
        shared_compute: Option<(
            &CudaSlice<u8>,
            &CudaSlice<u8>,
            Option<&CudaSlice<f32>>,
            usize,
        )>,
        shared_gated: bool,
        push_raw: u64,
    ) -> Res<()> {
        let hidden = self.hidden;
        let pack_raw = {
            let pack = ws.peek_u8("moe.pack")?;
            let stream = e.gpu.stream();
            pack.device_ptr(&stream).0
        };
        let mixed = ws.take_f32(e, "hc.mixed", hidden, 0)?;
        let mut act = ws.take_f32(e, "moe.act", max_sel * ff, 0)?;
        // Fused gate+up+silu (round 4, count-gated pack mode): the capture bakes the
        // live arm; dead slots (>= live count) retire at the first instruction and the
        // count-gated down/axpy never read them. Bit-identical to the chain per slot.
        if sel_gufuse_on() && hidden % 32 == 0 && ff % 4 == 0 {
            launch_nvfp4_sel_gu_silu(
                e, gate, up, None, pack_raw, max_sel, &mixed, &mut act, hidden, ff, None,
            )?;
        } else {
            let mut yg = ws.take_f32(e, "moe.yg", max_sel * ff, 0)?;
            let mut yu = ws.take_f32(e, "moe.yu", max_sel * ff, 0)?;
            launch_nvfp4_sel_matvec_pack(
                e, gate.0, gate.1, gate.2, pack_raw, max_sel, &mixed, &mut yg, hidden, ff, 0,
            )?;
            launch_nvfp4_sel_matvec_pack(
                e, up.0, up.1, up.2, pack_raw, max_sel, &mixed, &mut yu, hidden, ff, 0,
            )?;
            e.silu_mul(&yg, &yu, &mut act, max_sel * ff)?;
            ws.put_f32("moe.yg", yg);
            ws.put_f32("moe.yu", yu);
        }
        let mut partial = ws.take_f32(e, "moe.partial", max_sel * hidden, 0)?;
        launch_nvfp4_sel_matvec_pack(
            e,
            down.0,
            down.1,
            down.2,
            pack_raw,
            max_sel,
            &act,
            &mut partial,
            ff,
            hidden,
            ff,
        )?;
        let mut r = ws.take_f32(e, "moe.out", hidden, 0)?;
        launch_axpy_rows_seq_pack(e, &partial, pack_raw, max_sel, &mut r, hidden)?;
        ws.put_f32("moe.act", act);
        ws.put_f32("moe.partial", partial);
        let (sh, g) = match shared_compute {
            Some((gu_b16, d_b16, ig, sffh)) => {
                self.tp2_shared_half(e, ws, gu_b16, d_b16, ig, &mixed, sffh, 1)?
            }
            None => {
                let sh = ws.take_f32(e, "moe.sh_down", hidden, 0)?;
                let g = if shared_gated {
                    Some(ws.take_f32(e, "moe.g", 1, 0)?)
                } else {
                    None
                };
                (sh, g)
            }
        };
        match g.as_ref() {
            Some(g) => e.add_scaled_rows(&sh, g, &mut r, hidden, 1)?,
            None => {
                let mut view = r.slice_mut(0..hidden);
                e.axpy_into(&sh, 1.0, &mut view, hidden)?;
            }
        }
        ws.put_f32("moe.sh_down", sh);
        if let Some(g) = g {
            ws.put_f32("moe.g", g);
        }
        launch_push(e, &r, push_raw, hidden)?;
        ws.put_f32("moe.out", r);
        ws.put_f32("hc.mixed", mixed);
        Ok(())
    }

    /// TP2 segment D (graphable): MoE join add (SAME rank order both cards) + gate_write.
    #[allow(clippy::too_many_arguments)]
    fn tp2_seg_d(
        &self,
        e: &Engine,
        ws: &mut StepPool,
        ptrs: &CudaSlice<u64>,
        planes: &mut [CudaSlice<f32>],
        stage_recv: &CudaSlice<f32>,
        rank0: bool,
    ) -> Res<()> {
        let hidden = self.hidden;
        let mp = ws.take_f32(e, "moe.out", hidden, 0)?;
        let mut mo = ws.take_f32(e, "join.out", hidden, 0)?;
        if rank0 {
            e.add(&mp, stage_recv, &mut mo, hidden)?;
        } else {
            e.add(stage_recv, &mp, &mut mo, hidden)?;
        }
        let injm = take_inject(e, ws, self.streams, 1)?;
        self.gate_write(e, planes, ptrs, &mo, &injm, 1)?;
        ws.put_f32("moe.out", mp);
        ws.put_f32("join.out", mo);
        put_inject(ws, injm);
        Ok(())
    }
}

#[cfg(test)]
mod sel_group_tests {
    use super::*;

    /// The seam lives in process-global atomics and `cargo test` runs these in parallel
    /// THREADS of one process, so every test that mutates it takes this lock. Without it the
    /// mutating tests race and the suite fails intermittently on whichever one loses.
    static SEAM: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The AUTO rule at the SERVING geometry, pinned as a test because it is the shape the
    /// seam ships and it was WRONG once: an earlier rule derived `rows` from `g` to hold
    /// rows-per-warp at 4, and the measured ladder showed rows-per-LANE is what pays
    /// (DOWNSEL.md section 4). A regression here is a silent shape change.
    #[test]
    fn auto_resolves_the_measured_serving_shapes() {
        // down: out_f = hidden 2560, in_f = expert ff 640 -> pairs 20 -> g 4 (largest power
        // of two dividing 20), rows 4 -> rows_per_warp 32, grid.x 80.
        assert_eq!(sel_group_resolve(SEL_GROUP_AUTO, 640, 2560), Some((4, 4)));
        // gate+up: out_f = ff 640, in_f = hidden 2560 -> pairs 80 -> g 16, rows 4 ->
        // rows_per_warp 8, grid.x 80.
        assert_eq!(sel_group_resolve(SEL_GROUP_AUTO, 2560, 640), Some((16, 4)));
    }

    #[test]
    fn off_and_odd_in_f_take_the_shipped_kernel() {
        assert_eq!(sel_group_resolve(SEL_GROUP_OFF, 640, 2560), None);
        // in_f % 32 != 0 is the v3 guard too; the group form must not claim it.
        assert_eq!(sel_group_resolve(SEL_GROUP_AUTO, 48, 2560), None);
    }

    /// AUTO must never hand back a shape the launcher cannot tile exactly: a ragged tile puts
    /// live and dead lanes in the same `__shfl_down_sync`. It steps `rows` down before giving
    /// up, and gives up rather than clamping.
    #[test]
    fn auto_backs_off_rows_then_refuses_rather_than_tiling_raggedly() {
        // pairs 2 -> g 2 -> 16 groups. out_f 32 admits rows 2 (rpw 32); rows 4 (rpw 64) does
        // not divide 32, so AUTO must step down instead of returning an untileable shape.
        assert_eq!(sel_group_resolve(SEL_GROUP_AUTO, 64, 32), Some((2, 2)));
        // out_f 24 divides by neither 64, 32 nor 16 (rows 4/2/1 at g=2) -> refuse.
        assert_eq!(sel_group_resolve(SEL_GROUP_AUTO, 64, 24), None);
        for &(in_f, out_f) in &[(640usize, 2560usize), (2560, 640), (32, 32), (64, 16)] {
            let (g, rows) = sel_group_resolve(SEL_GROUP_AUTO, in_f, out_f)
                .unwrap_or_else(|| panic!("auto refused {in_f}x{out_f}"));
            assert_eq!(
                out_f % ((32 / g) * rows),
                0,
                "{in_f}x{out_f} -> g{g} rows{rows}"
            );
        }
    }

    /// An explicit pin is honoured verbatim (the A/B ladder depends on it) but still refuses a
    /// geometry it cannot tile, so a mis-set cell falls back to the shipped kernel loudly
    /// rather than launching a ragged grid.
    #[test]
    fn explicit_pins_are_verbatim_and_still_tile_checked() {
        let _g = SEAM.lock().unwrap();
        assert!(set_sel_group("dn:8:1+gu:16:4"));
        assert_eq!(sel_group_resolve(sel_group_dn(), 640, 2560), Some((8, 1)));
        assert_eq!(sel_group_resolve(sel_group_gu(), 2560, 640), Some((16, 4)));
        // g=1 rows=4 -> rows_per_warp 128. Both serving widths are multiples of 128 and
        // tile fine (2560 = 20x128, 640 = 5x128); an out_f that is NOT must refuse.
        assert!(set_sel_group("dn:1:4"));
        assert_eq!(sel_group_resolve(sel_group_dn(), 2560, 640), Some((1, 4)));
        assert_eq!(sel_group_resolve(sel_group_dn(), 2560, 96), None);
        set_sel_group("off");
    }

    /// A malformed spec must APPLY NOTHING and report false. If it half-applied or reported
    /// true, a typo in a cell script would silently measure the wrong arm — the failure the
    /// seam grammar exists to make impossible.
    #[test]
    fn malformed_specs_apply_nothing_and_refuse() {
        let _g = SEAM.lock().unwrap();
        assert!(set_sel_group("dn:4:4+gu:16:4"));
        let before = sel_group_spec();
        for bad in [
            "dn:3:4",      // g not a power of two
            "dn:4:3",      // rows not in {1,2,4}
            "dn:64:4",     // g > 32
            "dn:4",        // no rows
            "xx:4:4",      // unknown family
            "dn:4:4+xx:1", // one good half, one bad -> still nothing applied
            "+",
        ] {
            assert!(!set_sel_group(bad), "{bad:?} was accepted");
            assert_eq!(
                sel_group_spec(),
                before,
                "{bad:?} mutated state while refusing"
            );
        }
        set_sel_group("off");
    }

    /// `seam_state` has to answer for this seam even though it is shape-valued: the shared
    /// `--ladder-ab-seam` harness restores the entry arm ONLY when it answers, and a `None`
    /// there leaves the ON arm armed for every number after the A/B block.
    #[test]
    fn seam_round_trips_through_the_boolean_harness() {
        let _g = SEAM.lock().unwrap();
        set_sel_group("off");
        assert_eq!(seam_state("selgroup"), Some(false));
        assert!(set_seam("selgroup", true, None));
        assert_eq!(seam_state("selgroup"), Some(true));
        assert_eq!(sel_group_spec(), "dn:auto+gu:auto");
        assert!(set_seam("selgroup", false, None));
        assert_eq!(seam_state("selgroup"), Some(false));
        assert_eq!(sel_group_spec(), "dn:off+gu:off");
        // One family armed is still "armed", or an A/B that moved only the down half would
        // restore to OFF and lose the entry state.
        assert!(set_sel_group("dn:4:4+gu:off"));
        assert_eq!(seam_state("selgroup"), Some(true));
        set_sel_group("off");
        // Listed name and dispatch arm agree (the drift `seam_names` cannot detect alone).
        assert!(seam_names().contains(&"selgroup"));
        assert!(seam_exists("selgroup"));
    }
}

// ============================================================ TP2/EP2 placement unit tests
//
// SCOPE, stated because this file is a GPU forward and these tests touch no GPU: every
// assertion below is over `Tp2Placement`/`LayerPlacement`, which are pure host logic: map
// parsing, the fail-closed refusal set, the bank-split arithmetic (`card1`/`local_of`/
// `rank_of`) and the even-split control-arm property. They run in plain `cargo test` on any
// machine, which is the point: the two-card BEHAVIOUR needs a box, but the two-card
// BOOKKEEPING is the part that silently moves expert weights under the router, and it had no
// coverage at all before this lane (`qwen4exp_gpu.rs` carried no test module).
//
// Lane: research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md.
#[cfg(test)]
mod tp2_placement_tests {
    use super::{LayerPlacement, Tp2Placement};

    /// One `memra-ep-map-v1` document over `experts` experts, `layers` = the (layer,
    /// assignment) rows given. Written through a temp file because `load` takes a path
    /// (the production door is `MEMRA_Q4E_EP_MAP=<path>`).
    fn write_map(name: &str, body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "memra-q4e-ep-map-{name}-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, body).expect("write map fixture");
        path
    }

    fn load(name: &str, body: &str, expert_count: usize) -> Result<Tp2Placement, String> {
        let path = write_map(name, body);
        let out = Tp2Placement::load(&path, expert_count).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        out
    }

    /// `{"format": ..., "ranks": 2, "entry_rank": 0, "expert_count": 4, <body>}`
    fn doc(body: &str) -> String {
        format!(
            "{{\"format\": \"memra-ep-map-v1\", \"strategy\": \"coactivation\", \
             \"ranks\": 2, \"entry_rank\": 0, \"expert_count\": 4, {body}}}"
        )
    }

    fn assert_refuses(name: &str, body: &str, expert_count: usize, clause: &str) {
        match load(name, body, expert_count) {
            Ok(_) => panic!("{name}: expected a refusal naming {clause:?}, but the map loaded"),
            Err(msg) => {
                assert!(
                    msg.contains(clause),
                    "{name}: refusal must name the broken clause {clause:?}, got: {msg}"
                );
                // Every refusal names the FILE too, or the placement lane cannot tell
                // which of several candidate maps it has to fix.
                assert!(
                    msg.contains("MEMRA_Q4E_EP_MAP"),
                    "{name}: refusal must name the flag/file, got: {msg}"
                );
            }
        }
    }

    // ---------------------------------------------------------------- the control arm

    /// The unset door is the even split, and the even split is the CONTROL ARM of the
    /// placement A/B. Its bit-identity claim rests on exactly three properties, all
    /// asserted here rather than argued: card 0 addresses its full resident bank by
    /// GLOBAL id (no remap), card 1's gather order is the ascending suffix (a contiguous
    /// copy of what the pre-seam code sliced), and `is_even` recognises it.
    #[test]
    fn even_split_is_the_contiguous_suffix_control_arm() {
        let p = Tp2Placement::even(512);
        assert_eq!(p.strategy(), "even");
        assert_eq!(p.entry_rank(), 0);
        assert!(p.source().contains("MEMRA_Q4E_EP_MAP unset"));

        let l = p.layer(0, 512).expect("even split resolves every layer");
        assert!(l.is_even(), "the built-in even split must report as even");
        assert_eq!(l.card1.len(), 256);
        // Ascending contiguous suffix.
        assert_eq!(l.card1, (256u32..512).collect::<Vec<_>>());
        for e in 0..256 {
            assert_eq!(l.rank(e), 0, "expert {e} belongs to card 0");
            assert_eq!(l.local(e), e, "card-0 local slot IS the global id");
        }
        for (slot, e) in (256..512).enumerate() {
            assert_eq!(l.rank(e), 1, "expert {e} belongs to card 1");
            assert_eq!(l.local(e), slot, "card-1 local slot is the gather position");
        }
        // Every MoE layer index resolves identically; the even split is layer-independent.
        let l47 = p.layer(47, 512).expect("layer 47");
        assert_eq!(l47.card1, l.card1);
    }

    /// A MEASURED map moves bytes, and the local-slot bookkeeping is what keeps the
    /// router and the bank agreeing. Non-contiguous ownership is the whole point of
    /// co-activation placement, so it is the case the arithmetic must get right.
    #[test]
    fn measured_map_resolves_ascending_gather_and_local_slots() {
        // 4 experts, card 1 owns {0, 3}, deliberately NOT a suffix.
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [1, 0, 0, 1]}]";
        let p = load("measured", &doc(body), 4).expect("balanced map loads");
        assert_eq!(p.strategy(), "coactivation");
        let l = p.layer(0, 4).expect("layer 0");

        assert!(
            !l.is_even(),
            "a non-suffix placement is not the control arm"
        );
        // ASCENDING is load-bearing: it makes the gather order a function of the map
        // alone, so no host set-iteration order can leak into device bytes.
        assert_eq!(l.card1, vec![0u32, 3]);
        assert_eq!((l.rank(0), l.rank(1), l.rank(2), l.rank(3)), (1, 0, 0, 1));
        // card 1: local slot = position in `card1`.
        assert_eq!(l.local(0), 0);
        assert_eq!(l.local(3), 1);
        // card 0: local slot = the global id, untouched.
        assert_eq!(l.local(1), 1);
        assert_eq!(l.local(2), 2);
    }

    /// `is_even` must not be fooled by a BALANCED-but-permuted map: it is the predicate a
    /// receipt uses to claim "this run was the control arm", so a false positive would let
    /// a measured-placement run be banked as its own control.
    #[test]
    fn is_even_rejects_a_balanced_permutation() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 1, 1, 0]}]";
        let p = load("perm", &doc(body), 4).expect("balanced map loads");
        let l = p.layer(0, 4).expect("layer 0");
        assert_eq!(l.card1, vec![1u32, 2]);
        assert!(!l.is_even());
    }

    /// A map whose assignment IS the even suffix must be recognised as the control arm,
    /// so the A/B harness can prove its two arms are the same program.
    #[test]
    fn an_explicit_even_map_matches_the_builtin_even_split() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 1]}]";
        let p = load("explicit-even", &doc(body), 4).expect("even map loads");
        let l = p.layer(0, 4).expect("layer 0");
        let builtin = Tp2Placement::even(4).layer(0, 4).expect("builtin");
        assert!(l.is_even());
        assert_eq!(l.card1, builtin.card1);
        for e in 0..4 {
            assert_eq!(l.rank(e), builtin.rank(e), "rank of expert {e}");
            assert_eq!(l.local(e), builtin.local(e), "local slot of expert {e}");
        }
    }

    // ---------------------------------------------------------------- the refusal set
    //
    // One test per contract clause. A half-applied placement moves expert weights under
    // the router and reads as a MODEL bug rather than a config bug, so each of these is a
    // load-time refusal by name, and each refusal has to name the clause it broke.

    #[test]
    fn refuses_a_foreign_format() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 1]}]";
        let text = format!(
            "{{\"format\": \"memra-ep-map-v2\", \"ranks\": 2, \"expert_count\": 4, {body}}}"
        );
        assert_refuses("format", &text, 4, "memra-ep-map-v1");
    }

    #[test]
    fn refuses_a_rank_count_that_is_not_two() {
        let text = "{\"format\": \"memra-ep-map-v1\", \"ranks\": 4, \"expert_count\": 4, \
                    \"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 1]}]}";
        assert_refuses("ranks", text, 4, "exactly two cards");
    }

    #[test]
    fn refuses_an_expert_count_that_is_not_the_plans() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 1]}]";
        assert_refuses("experts", &doc(body), 8, "expert_count=4");
    }

    #[test]
    fn refuses_an_entry_rank_outside_the_two_cards() {
        let text = "{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \"entry_rank\": 2, \
                    \"expert_count\": 4, \
                    \"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 1]}]}";
        assert_refuses("entry", text, 4, "entry_rank=2");
    }

    #[test]
    fn refuses_a_document_with_no_layers_array() {
        assert_refuses("nolayers", &doc("\"strategy2\": 0"), 4, "no `layers` array");
    }

    #[test]
    fn refuses_an_empty_layers_array() {
        assert_refuses(
            "emptylayers",
            &doc("\"layers\": []"),
            4,
            "`layers` is empty",
        );
    }

    #[test]
    fn refuses_an_assignment_of_the_wrong_length() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 1]}]";
        assert_refuses("shortassign", &doc(body), 4, "expected 4");
    }

    #[test]
    fn refuses_a_rank_id_outside_the_two_cards() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 7]}]";
        assert_refuses("badrank", &doc(body), 4, "expert 3");
    }

    /// The clause with the sharpest consequence: the card-1 bank halves are EQUAL-SIZE
    /// device allocations, so an unbalanced map is out-of-bounds rather than merely
    /// slower. The refusal must also point at the tool's rebalance knob.
    #[test]
    fn refuses_an_unbalanced_layer_and_names_the_rebalance_knob() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 1, 1, 1]}]";
        assert_refuses("unbalanced", &doc(body), 4, "card 1 owns 3 experts");
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 1, 1, 1]}]";
        assert_refuses("unbalanced2", &doc(body), 4, "--balance-tolerance");
    }

    /// A map that covers SOME MoE layers is not a placement. Falling the uncovered layers
    /// back to the even split would make the receipt a lie about which placement ran.
    #[test]
    fn refuses_a_layer_the_map_does_not_cover() {
        let body = "\"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 1]}]";
        let p = load("partial", &doc(body), 4).expect("map loads");
        assert!(p.layer(0, 4).is_ok(), "the covered layer resolves");
        let msg = p
            .layer(1, 4)
            .expect_err("an uncovered MoE layer must refuse")
            .to_string();
        assert!(msg.contains("does not cover MoE layer 1"), "got: {msg}");
        assert!(
            msg.contains("partly-applied map is not a placement"),
            "the refusal must say WHY it is fail-closed, got: {msg}"
        );
    }

    #[test]
    fn refuses_a_layer_whose_expert_count_disagrees_with_the_map() {
        let p = Tp2Placement::even(512);
        let msg = p
            .layer(0, 256)
            .expect_err("a layer geometry the map is not for must refuse")
            .to_string();
        assert!(msg.contains("map is for 512"), "got: {msg}");
    }

    #[test]
    fn refuses_an_unreadable_map_path() {
        let missing = std::env::temp_dir().join(format!(
            "memra-q4e-ep-map-absent-{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&missing);
        let msg = Tp2Placement::load(&missing, 4)
            .expect_err("an unreadable map must refuse at the load preflight")
            .to_string();
        assert!(msg.contains("MEMRA_Q4E_EP_MAP"), "got: {msg}");
    }

    /// An ODD routed bank has no equal halves, on either path.
    ///
    /// Scoped honestly, because the guard's first justification overclaimed and review caught
    /// it: production cannot reach this, since `build_tp2_shard` refuses `experts % 2 != 0`
    /// before it asks for a `LayerPlacement`. These are `pub` entry points and the refusal
    /// belongs on the contract it breaks.
    ///
    /// The loaded arm below is the one that closes a REAL hole, and it is deliberately the
    /// BALANCED odd map: `half = expert_count / 2` floors, so 2-of-5 on card 1 satisfies
    /// `on1 == half` and loaded clean before this check. An unbalanced odd map (3-of-5) would
    /// have been refused by the balance clause already, so testing only that would have made
    /// this arm nearly vacuous.
    #[test]
    fn refuses_an_odd_routed_bank_on_both_paths() {
        // built-in even split
        let msg = Tp2Placement::even(5)
            .layer(0, 5)
            .expect_err("an odd bank has no two-card placement")
            .to_string();
        assert!(msg.contains("ODD"), "got: {msg}");
        assert!(msg.contains("EQUAL-size"), "got: {msg}");
        // loaded map, BALANCED under the floored half (on1 == 5/2 == 2): this one passed the
        // balance clause before the geometry check existed.
        let balanced_odd = "{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \
                            \"expert_count\": 5, \
                            \"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 0, 1, 1]}]}";
        assert_refuses("odd-balanced", balanced_odd, 5, "ODD");
        // and the unbalanced odd map, which the balance clause would also have caught, so this
        // asserts the geometry clause wins the race and names the real problem.
        let unbalanced_odd = "{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \
                              \"expert_count\": 5, \
                              \"layers\": [{\"layer\": 0, \"assignment\": [0, 0, 1, 1, 1]}]}";
        assert_refuses("odd-unbalanced", unbalanced_odd, 5, "ODD");
    }

    // ---------------------------------------------------------------- bank-split arithmetic

    /// The invariant the card-1 bank upload depends on, asserted over EVERY expert of a
    /// serving-geometry bank: exactly half the ids land on card 1, `card1` is strictly
    /// ascending, and `local_of` is a bijection onto `0..half` for card 1 and the identity
    /// on card 0. A violation here is an out-of-bounds device read, not a slow placement.
    #[test]
    fn local_slots_are_a_bijection_at_the_serving_geometry() {
        let experts = 512usize;
        let half = experts / 2;
        // A deterministic non-contiguous, exactly-balanced placement: alternate ownership.
        let assignment: Vec<String> = (0..experts).map(|e| (e % 2).to_string()).collect();
        let body = format!(
            "\"layers\": [{{\"layer\": 0, \"assignment\": [{}]}}]",
            assignment.join(", ")
        );
        let text = format!(
            "{{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \"expert_count\": {experts}, \
             {body}}}"
        );
        let p = load("bijection", &text, experts).expect("balanced alternating map loads");
        let l: LayerPlacement = p.layer(0, experts).expect("layer 0");

        assert_eq!(l.card1.len(), half, "card 1 must own exactly half the bank");
        assert!(
            l.card1.windows(2).all(|w| w[0] < w[1]),
            "the gather order must be strictly ascending"
        );
        let mut seen = vec![false; half];
        for e in 0..experts {
            match l.rank(e) {
                0 => assert_eq!(l.local(e), e, "card-0 slot is the global id"),
                1 => {
                    let slot = l.local(e);
                    assert!(slot < half, "card-1 slot {slot} outside its half-bank");
                    assert!(!seen[slot], "card-1 slot {slot} claimed twice");
                    seen[slot] = true;
                }
                r => panic!("expert {e} has rank {r}"),
            }
        }
        assert!(seen.into_iter().all(|s| s), "card-1 slots must be dense");
        assert!(!l.is_even());
    }
}
