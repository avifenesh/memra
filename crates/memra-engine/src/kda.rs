//! Kimi Delta Attention (KDA) — the glm5_next (GLM-5.3-Flash) linear-attention mixer.
//!
//! Arithmetic contract: `memra_reference::kimi_delta_net`, pinned by
//! `kimi_delta_net_matches_hand_derived_three_token_recurrence`. Every step below cites the
//! reference stage it reproduces; the GPU-vs-reference gate is
//! `crates/memra-engine/tests/kda_fixture_gpu.rs`.
//!
//! Geometry (research/glm53-flash-bringup-20260827/CENSUS.md): 64 heads x 128, q/k/v all the
//! same width, short conv kernel 4, forget-gate lower bound -5.0. Symmetric widths and no GQA
//! repeat mean channel `c == h*head_dim + i` IS the (head, dim) pair, so every per-token tensor
//! stays token-major end to end — there is no analogue of GDN's qkv_to_gdn_repack scatter here.
//!
//! PREFILL DISPATCH — SEQUENTIAL SCAN, not the chunked UT transform (deliberate).
//! `memra_kda_scan_s128` runs prefill and decode alike, which is exactly the shipped
//! GDN arrangement next door: `gdn_scan_s128` IS the default prefill path and the chunked WY
//! kernels sit behind `MEMRA_GDN_CHUNKED`. One kernel for both also keeps the decode==verify
//! dispatch identity that cu/hybrid.cu's headers require. A chunked twin exists but is
//! SHELVED, ATTRIBUTED-NEGATIVE — it is not a pending tuning follow-up. It was built as L3
//! of the prefill-gap plan (`MEMRA_KDA_CHUNKED`, unmerged branch lane/glm5-kda-chunk-scan),
//! and the box prefill census then attributed the wall elsewhere: on a cold 4626-token prime
//! the whole kda family is 221.6 GPU ms of 6598 (3.4%, "confirms L3's ATTRIBUTED-NEGATIVE:
//! scan ~2.4%") while mla-prefill-attn owns 75.8% — receipts
//! `research/glm53-flash-bringup-20260827/launch-diet-20260830/WINDOW-20260830.md` §4 and
//! `box-receipts-20260830/census-analysis.txt`. No A/B is owed on the scan; a revival needs
//! a new attribution first. The algebra stays banked for that day: it is NOT a transcription
//! of the GDN K1-K5 chain — KDA's decay is per channel, so the chunk form needs a per-channel
//! cumulative log gate `Gcum[t][i]` with `k` scaled by `exp(-Gcum)` and `q` by `exp(+Gcum)`
//! (banked `chunk_kimi_delta_attention` in
//! research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py), where GDN gets away with
//! one scalar `G` per (token, head).
//!
//! CONV FUSION — fused WEIGHTS and a fused RING, per-plane launches. The checkpoint ships three
//! per-plane conv weights; they are concatenated once at load into one `[3*qkv, kernel]` f32
//! buffer, because the plan already declares the state carrier fused (`StatePlan::Recurrent`
//! `conv_width = 3*qkv`) and that makes a plane's weight offset and its ring offset the same
//! `plane*qkv` arithmetic. The three PROJECTIONS stay separate: they are independently
//! quantized tensors, and concatenating them would mean dequantizing to build one matmul.
//! Applying each plane's taps to its own plane is the fused grouped conv exactly (the reference
//! says so in-line), so nothing is approximated by the split.

use crate::Engine;
use crate::cache::{Cache, RecurLayer};
use crate::model::GpuTensor;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use memra_gguf::model_plan::KimiDeltaNetPlan;
use memra_gguf::source::TensorSource;
use std::sync::atomic::{AtomicU64, Ordering};

/// Engagement counter for the fused 6-way projection door (`MEMRA_KDA_FUSED_PROJ`), the
/// grouped-prefill `moe_grouped_prefill_dispatches` precedent: gates and box A/B arms count
/// dispatches at the arm's own call site instead of inferring engagement from a 200.
pub static KDA_FUSED6_DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub static KDA_FUSED6_E4M3_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Same door, BF16 operand arm (`qmatvec_kda6_bf16f32`, lane/glm5-decode-diet lever 3).
/// Counted separately so a box A/B on the serving recipe (MEMRA_BF16_MMV=1, where the q8 arm
/// refuses by design) can attribute engagement to the arm that actually ran.
pub static KDA_FUSED6_BF16_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Same door, W8-MIRROR arm (`qmatvec_kda6_q8f32_rp_v2`, lane/b200-gemv-hbm-20260902 round 3).
/// Counted separately for the same reason the bf16 arm is: a box A/B on the serving recipe must
/// be able to attribute engagement to the arm that actually ran.
pub static KDA_FUSED6_Q8RP_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// The only head width `memra_kda_scan_s128` is instantiated for, and the only one glm5_next
/// ships (`linear_attn_config.head_dim = 128`).
pub const KDA_HEAD_DIM: usize = 128;
/// The conv kernels hold their window in a fixed register array; wider kernels would silently
/// read past it, so the loader refuses them.
const KDA_MAX_CONV_KERNEL: usize = 8;
/// FLA l2norm epsilon. Fixed at 1e-6 and INSIDE the sqrt — independent of the layer's rms eps,
/// which is a different constant used by the output norm below.
const KDA_L2_EPS: f32 = 1e-6;

/// One loaded KDA mixer. Field names follow the reference's tensor roles, not the HF spellings.
pub struct KdaAttnLayer {
    pub plan: KimiDeltaNetPlan,
    /// q/k/v projections, `[qkv, hidden]` each.
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    /// Forget gate low-rank pair: `f_a [head_dim, hidden]`, `f_b [qkv, head_dim]`.
    pub f_a: GpuTensor,
    pub f_b: GpuTensor,
    /// Output gate low-rank pair, same shapes as the forget pair.
    pub g_a: GpuTensor,
    pub g_b: GpuTensor,
    /// Per-head beta projection, `[heads, hidden]`.
    pub b_proj: GpuTensor,
    /// Output projection, `[hidden, qkv]`.
    pub wo: GpuTensor,
    /// The three per-plane conv weights concatenated into `[3*qkv, kernel]` (see module header).
    pub conv: CudaSlice<f32>,
    /// `A_log [heads]`, `dt_bias [qkv]` (per CHANNEL, unlike GDN's per-head bias),
    /// `o_norm [head_dim]`.
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    pub o_norm: GpuTensor,
    /// glm5 TP-2 sidecar (`MEMRA_GLM5_TP`, lane/glm5-tp2). `Some` means THIS layer struct is
    /// the ROOT-RANK HEAD SHARD (heads/2) and the sidecar carries the peer shard + runtime.
    /// Every plain entry point REFUSES a sharded layer by name — only the TP walk
    /// (`glm5_tp::kda_tp_*`) may execute it. `None` everywhere else (zero cost, zero change).
    pub tp: Option<Box<crate::glm5_tp::Glm5TpKda>>,
}

impl KdaAttnLayer {
    pub fn heads(&self) -> usize {
        self.plan.num_heads as usize
    }
    pub fn head_dim(&self) -> usize {
        self.plan.head_dim as usize
    }
    pub fn qkv(&self) -> usize {
        self.heads() * self.head_dim()
    }
    pub fn conv_kernel(&self) -> usize {
        self.plan.conv_kernel as usize
    }
    /// Fused conv ring width, matching `StatePlan::Recurrent { conv_width }` for this layer.
    pub fn conv_width(&self) -> usize {
        3 * self.qkv()
    }
    /// Recurrent state elements, matching `StatePlan::Recurrent { state_width }`.
    pub fn state_width(&self) -> usize {
        self.heads() * self.head_dim() * self.head_dim()
    }

    /// Load block `il`'s KDA tensors. Names are the ggml-dialect contract names from
    /// `memra_gguf::tensor_contract::add_kda`; the safetensors source translates them.
    pub fn load(
        e: &Engine,
        src: &dyn TensorSource,
        il: u32,
        plan: &KimiDeltaNetPlan,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let heads = plan.num_heads as usize;
        let head_dim = plan.head_dim as usize;
        let kernel = plan.conv_kernel as usize;
        if head_dim != KDA_HEAD_DIM {
            return Err(format!(
                "blk.{il}: KDA head_dim {head_dim} is not the {KDA_HEAD_DIM} the scan kernel is \
                 instantiated for; a new memra_kda_scan_s<N> instantiation is required before \
                 this geometry can serve"
            )
            .into());
        }
        if heads == 0 {
            return Err(format!("blk.{il}: KDA num_heads must be positive").into());
        }
        if !(2..=KDA_MAX_CONV_KERNEL).contains(&kernel) {
            return Err(format!(
                "blk.{il}: KDA conv_kernel {kernel} outside the 2..={KDA_MAX_CONV_KERNEL} window \
                 the conv kernels hold in registers"
            )
            .into());
        }
        let p = |s: &str| format!("blk.{il}.{s}");
        let load = |name: String| GpuTensor::load_from_source(e, src, &name);

        let qkv = heads * head_dim;
        // Fuse the three per-plane conv weights into one [3*qkv, kernel] buffer (module header).
        // Each source tensor is [qkv, kernel] channel-major, so the planes concatenate as whole
        // row blocks and plane p lands at row p*qkv — the ring's own plane offset.
        let mut conv = e.zeros(3 * qkv * kernel)?;
        for (plane, name) in [
            "kda_q_conv1d.weight",
            "kda_k_conv1d.weight",
            "kda_v_conv1d.weight",
        ]
        .into_iter()
        .enumerate()
        {
            let w = load(p(name))?;
            let src_data = w.float_data();
            if src_data.len() != qkv * kernel {
                return Err(format!(
                    "blk.{il}.{name}: {} elements, contract requires {}",
                    src_data.len(),
                    qkv * kernel
                )
                .into());
            }
            e.copy_into(&mut conv, plane * qkv * kernel, src_data, qkv * kernel)?;
        }

        Ok(Self {
            plan: *plan,
            wq: load(p("kda_q.weight"))?,
            wk: load(p("kda_k.weight"))?,
            wv: load(p("kda_v.weight"))?,
            f_a: load(p("kda_f_a.weight"))?,
            f_b: load(p("kda_f_b.weight"))?,
            g_a: load(p("kda_g_a.weight"))?,
            g_b: load(p("kda_g_b.weight"))?,
            b_proj: load(p("kda_b.weight"))?,
            wo: load(p("kda_out.weight"))?,
            conv,
            a_log: load(p("kda_a_log"))?,
            dt_bias: load(p("kda_dt.bias"))?,
            o_norm: load(p("kda_o_norm.weight"))?,
            tp: None,
        })
    }
}

/// Which conv arm a call takes. `Prefill` reads the ring as a left pad and rolls it afterwards;
/// `Decode` fuses assemble+conv+roll for the single new row. The two produce bit-identical
/// values at T=1 (same ascending tap order over the same window) — the split exists so decode
/// and the spec verify keep one dispatch class, per the cu/hybrid.cu decode==verify law.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvArm {
    Prefill,
    Decode,
}

/// The scan-input buffers of one KDA step, STOLEN from the step instead of dropped
/// (lane/glm5-loop-port, port 3 — the module doc's named GdnStash/ReplaySSM diet): the
/// glm5 verify walk's rollback checkpoint keeps these ~160 KB of already-allocated
/// buffers per row per layer and retires the per-row 4 MiB recurrent-state clones
/// (~0.95 GiB transient at K=7). Replaying `kda_scan` over them from a pre-round state
/// snapshot rebuilds the post-row state EXACTLY: each replay is the ORIGINAL t=1 launch
/// re-issued — same kernel, same inputs, same shape — so the rebuilt state is
/// byte-identical to the clone it replaces by construction, not by a numeric argument.
pub struct KdaScanInputs {
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub g: CudaSlice<f32>,
    pub beta: CudaSlice<f32>,
}

/// The rollback stash of one BATCHED verify-rows KDA call (lane/glm5-verify-batch): the
/// per-layer t=K+1 twin of the per-row [`KdaScanInputs`] steal. Everything here is either
/// stolen from buffers the call allocated anyway (`raws`, `scan` — zero copies) or one
/// small clone per layer per round (`ring_snap`, `3*qkv*(kernel-1)` floats ~ 96 KiB).
///
/// Rollback to `keep` rows rebuilds both state planes EXACTLY:
///   * conv ring: restore `ring_snap`, then re-issue `kda_conv_ring_roll` per plane over
///     `raws` at T=keep — the roll is pure placement (no arithmetic), so the rebuilt ring
///     is the sequential chain's ring after row keep-1 byte-for-byte.
///   * ssm state: ONE `kda_scan` replay at T=keep from the caller's pre-round snapshot
///     over the batched `scan` inputs (the kernel walks rows 0..keep of the [t, ..]
///     buffers) — the in-kernel T-loop IS the chained t=1 program (register-resident
///     state, identical per-step order), held by the scan-chain bit-gate.
pub struct KdaRowsStash {
    /// The fused conv ring BEFORE this call's rolls (one clone per layer per round).
    pub ring_snap: CudaSlice<f32>,
    /// RAW (pre-conv) q/k/v projection rows `[t, qkv]`, stolen post-roll (plane order).
    pub raws: [CudaSlice<f32>; 3],
    /// Batched scan inputs `[t, ..]`, stolen post-scan.
    pub scan: KdaScanInputs,
    /// Row count of the call that filled this stash; rollback validates `keep` against it.
    pub rows: usize,
}

/// What a `kda_core` call is asked to leave behind for rollback — and, for `Rows`, which
/// matmul class the call rides (the decode-exact rows classes, `matmul_rows_exact`).
pub(crate) enum KdaStash<'a> {
    /// No rollback stash (prefill / plain decode).
    None,
    /// Per-row t=1 steal (loop-port 3, the per-row verify walk).
    Decode(&'a mut Option<KdaScanInputs>),
    /// BATCHED verify-rows steal (lane/glm5-verify-batch): scan inputs + raw conv rows +
    /// a pre-call ring snapshot; every matmul rides `matmul_rows_exact` so each row is
    /// bit-identical to the t=1 decode program per the decode-exact class contracts.
    Rows(&'a mut Option<KdaRowsStash>),
}

/// `MEMRA_KDA_STEP_TRACE=1` (gate-harness instrument, default OFF, never a serving flag): after
/// each sub-step of the KDA core, print the non-finite element count of every live buffer on one
/// line. memra#131 cell 9 placed the graph door's poison inside `kda_decode_cached` at layer 4
/// (finite input, finite state, all-NaN mixer output after a capture); this names the kernel.
fn kda_trace_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MEMRA_KDA_STEP_TRACE").as_deref() == Ok("1"))
}
fn kda_trace(e: &Engine, stage: &str, t: usize, bufs: &[(&str, &CudaSlice<f32>)]) {
    if !kda_trace_on() {
        return;
    }
    // A device-to-host copy is illegal inside an open CUDA graph capture (it invalidates the
    // capture); box cell 11 (memra#131) hit exactly that on the session's first decode step,
    // which the door captures. Print the stage with a note instead of counting.
    if crate::glm5_graph_capture_open() {
        eprintln!("[kda-step-trace] t={t} {stage}: (inside an open graph capture; not counted)");
        return;
    }
    let mut line = format!("[kda-step-trace] t={t} {stage}:");
    for (name, b) in bufs {
        let n = match e.dtoh(b) {
            Ok(v) => v.iter().filter(|x| !x.is_finite()).count(),
            Err(_) => usize::MAX,
        };
        line.push_str(&format!(" {name}={n}/{}", b.len()));
    }
    eprintln!("{line}");
}

/// The whole mixer, stage for stage against `memra_reference::kimi_delta_net`.
///
/// `ring` is the fused `[3*qkv, kernel-1]` conv state (zeroed = fresh prefill's zero left pad)
/// and is updated in place. `state_in`/`state_out` are the `[heads, 128, 128]` recurrent state
/// in the kernel's transposed `M[col][i]` layout; they MUST be distinct buffers.
#[allow(clippy::too_many_arguments)]
// allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
fn kda_core(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    ring: &mut CudaSlice<f32>,
    state_in: &CudaSlice<f32>,
    state_out: &mut CudaSlice<f32>,
    arm: ConvArm,
    stash: KdaStash<'_>,
    scan_clock: Option<&mut u64>,
    pre_q8: KdaPreQ8<'_>,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    // glm5 TP fail-closed choke point: every plain KDA entry (stateless, prime, decode,
    // stash — INCLUDING the batched verify-rows walk, `kda_verify_rows_cached`) funnels
    // through here. A TP-sharded layer holds heads/2 — running it on the plain path would
    // compute a silently-halved mixer, so it refuses by name instead.
    if la.tp.is_some() {
        return Err(format!(
            "KDA layer is glm5-TP-sharded (MEMRA_GLM5_TP): the plain mixer path is unwired \
             for a head shard — only the TP decode/prime walk may execute it (t={t}, arm \
             {})",
            if arm == ConvArm::Decode {
                "decode"
            } else {
                "prefill"
            }
        )
        .into());
    }
    // Verify-batch wo seam (lane/glm5-verify-batch): the rows arm routes the output
    // projection through the decode-exact classes, exactly like every projection inside
    // the core — the wo dispatch moved into this wrapper with the TP split, its routing
    // did not change.
    let rows_exact = matches!(stash, KdaStash::Rows(_));
    // MEMRA_KDA_ONORM_ZQ8: ask the core for the o_norm output's q8_1 pair when `wo` can take it.
    let want_pair = !rows_exact && kda_onorm_zq8_on() && e.mmvq_fast_eligible(&la.wo, t);
    let mut onorm_q8: KdaOnormQ8 = None;
    let gated = kda_core_gated(
        e,
        la,
        x,
        t,
        eps,
        ring,
        state_in,
        state_out,
        arm,
        stash,
        scan_clock,
        pre_q8,
        if want_pair { Some(&mut onorm_q8) } else { None },
    )?;
    if rows_exact {
        let y = e.matmul_rows_exact(&la.wo, &gated, t);
        // Door W: gated's last reader was the wo matmul above.
        e.vws_recycle(gated);
        y
    } else {
        if let Some((aq, ad)) = onorm_q8.as_ref()
            && let Some(y) = e.matmul_q8_fast(&la.wo, aq, ad, t)?
        {
            if KDA_ONORM_ZQ8_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                eprintln!(
                    "[kda-onorm-zq8] engaged: the KDA o_norm launch emits wo's q8_1 pair; the \
                     standalone quantize is gone (MEMRA_KDA_ONORM_ZQ8=1)"
                );
            }
            return Ok(y);
        }
        e.matmul(&la.wo, &gated, t)
    }
}

/// [`kda_core`] up to (and excluding) the output projection: returns the gated `[t, qkv]`
/// mixer output. Split out for the glm5 TP-2 seam, whose column-parallel `wo` runs over the
/// cross-rank GATHERED gated tensor rather than this shard's slice — the plain path is
/// `kda_core` above, byte-for-byte the pre-split body (the wo matmul and its rows-exact
/// routing moved, nothing else). This body is the CURRENT doored/batched core: it carries
/// the `MEMRA_KDA_FUSED_PROJ` door and the verify-batch rows arm; the TP decode/prime walk
/// calls it with `KdaStash::None`, the spec x TP verify walk (lane/glm5-composition) with
/// `KdaStash::Rows` per rank, and the TP load preflight refuses the fused-proj door by
/// name (unproven composition on head shards — see the FLAGS.md composition matrix).
#[allow(clippy::too_many_arguments)] // mirrors kda_core's own contract-shaped list
pub(crate) fn kda_core_gated(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    ring: &mut CudaSlice<f32>,
    state_in: &CudaSlice<f32>,
    state_out: &mut CudaSlice<f32>,
    arm: ConvArm,
    stash: KdaStash<'_>,
    mut scan_clock: Option<&mut u64>,
    pre_q8: KdaPreQ8<'_>,
    onorm_q8: Option<&mut KdaOnormQ8>,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let heads = la.heads();
    let head_dim = la.head_dim();
    let qkv = la.qkv();
    let kernel = la.conv_kernel();
    // The BATCHED verify-rows arm (lane/glm5-verify-batch): prefill conv dispatch (per-row
    // bit-identical to the decode arm — same ascending taps over the same window values,
    // held by the conv-arm bit-gate) + decode-exact matmul classes + the rows stash.
    let rows_exact = matches!(stash, KdaStash::Rows(_));
    if rows_exact && arm != ConvArm::Prefill {
        return Err("KDA rows stash requires the prefill conv arm".into());
    }
    if arm == ConvArm::Decode && t != 1 {
        return Err(format!("KDA decode arm requires t == 1, got {t}").into());
    }
    if ring.len() < la.conv_width() * (kernel - 1) {
        return Err(format!(
            "KDA conv ring holds {} floats, layer needs {}",
            ring.len(),
            la.conv_width() * (kernel - 1)
        )
        .into());
    }
    if state_in.len() < la.state_width() || state_out.len() < la.state_width() {
        return Err(format!(
            "KDA recurrent state holds {}/{} floats, layer needs {}",
            state_in.len(),
            state_out.len(),
            la.state_width()
        )
        .into());
    }

    // Stage 1 — the six projections that read x directly. f_b/g_b are chained off their own
    // down-projections below, exactly as the reference nests them.
    //
    // MEMRA_KDA_FUSED_PROJ=1 (default OFF): the six matvec calls collapse to one quantize +
    // one `qmatvec_kda6_q8f32_mmvq` launch — the program shape both vLLM and SGLang ship for
    // this trunk (ENGINE-SURVEY.md C1) and the step37 QKV_FUSED transfer (TRANSFER-MAP lever 1).
    // `kda_proj_fused6` refuses (returns None) on any operand/env shape where its bit-identity
    // claim would not hold, so the fall-through arm is always the unchanged program.
    let mut g6 = match e.kda_proj_fused6_pre(la, x, t, pre_q8)? {
        Some(outs) => outs,
        None if rows_exact => {
            // Verify-rows matmul class: per-weight decode-exact dispatch (the tcols /
            // batched-MMVQ / per-token-linear classes — each row bit-identical to the
            // t=1 program by the matmul_rows_exact contract).
            [&la.wq, &la.wk, &la.wv, &la.f_a, &la.g_a, &la.b_proj]
                .into_iter()
                .map(|w| e.matmul_rows_exact(w, x, t))
                .collect::<Result<Vec<_>, _>>()?
        }
        None => e.matmul_group(
            &[&la.wq, &la.wk, &la.wv, &la.f_a, &la.g_a, &la.b_proj],
            x,
            t,
        )?,
    };
    let beta_raw = g6.pop().unwrap(); // [T, heads]
    let gate_down = g6.pop().unwrap(); // [T, head_dim]
    let forget_down = g6.pop().unwrap(); // [T, head_dim]
    if kda_trace_on() {
        let mut v: Vec<(&str, &CudaSlice<f32>)> = vec![("x", x)];
        let names = ["g6_0", "g6_1", "g6_2", "g6_3", "g6_4", "g6_5"];
        for (k, b) in g6.iter().enumerate() {
            v.push((names[k.min(5)], b));
        }
        v.push(("forget_down", &forget_down));
        v.push(("gate_down", &gate_down));
        v.push(("beta_raw", &beta_raw));
        kda_trace(e, "proj", t, &v);
    }
    let v_raw = g6.pop().unwrap(); // [T, qkv]
    let k_raw = g6.pop().unwrap();
    let q_raw = g6.pop().unwrap();

    // Rows stash: snapshot the ring BEFORE the rolls mutate it (one ~96 KiB clone per
    // layer per round — the rollback's re-roll base). Door W: on the rows arm the snapshot
    // (and every scratch below) is a pooled draw — vws_uninit == alloc_uninit with the
    // door off, and the non-rows arms keep the plain allocs untouched.
    let ring_snap = match &stash {
        KdaStash::Rows(_) => {
            let mut snap = e.vws_uninit(ring.len())?;
            e.dtod_copy_into(ring, &mut snap, 0)?;
            Some(snap)
        }
        _ => None,
    };

    // Stage 2 — per-plane causal short conv + SiLU. Planes are ordered q, k, v in both the fused
    // weight buffer and the fused ring, which is the order the reference stores conv_state in.
    let mut q_conv = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    let mut k_conv = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    let mut v_conv = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    // MEMRA_KDA_CONV3 (lane/glm5-kda-conv3-20260904, default OFF): the decode arm's three
    // per-plane launches as ONE (plane = blockIdx.y), bit-identical per channel; the prefill arm
    // and the door-OFF decode keep the per-plane loop verbatim.
    if arm == ConvArm::Decode && kda_conv3_on() && kernel <= 9 {
        e.kda_conv_silu_decode3(
            [&q_raw, &k_raw, &v_raw],
            ring,
            &la.conv,
            [&mut q_conv, &mut k_conv, &mut v_conv],
            qkv,
            kernel,
        )?;
    } else {
        for (plane, (raw, out)) in [
            (&q_raw, &mut q_conv),
            (&k_raw, &mut k_conv),
            (&v_raw, &mut v_conv),
        ]
        .into_iter()
        .enumerate()
        {
            match arm {
                ConvArm::Prefill => {
                    e.kda_conv_silu(raw, &la.conv, ring, out, qkv, t, kernel, plane)?
                }
                ConvArm::Decode => {
                    e.kda_conv_silu_decode(raw, ring, &la.conv, out, qkv, kernel, plane)?
                }
            }
        }
    }
    // The prefill arm reads the OLD ring for every token, so the roll runs only after all three
    // planes have been convolved. The decode arm already rolled inside its fused kernel.
    if arm == ConvArm::Prefill {
        for (plane, raw) in [&q_raw, &k_raw, &v_raw].into_iter().enumerate() {
            e.kda_conv_ring_roll(raw, ring, qkv, t, kernel, plane)?;
        }
    }

    // Stage 3 — q/k L2 norm over head_dim (eps INSIDE the sqrt, fixed 1e-6). Rows of the
    // token-major layout are contiguous head_dim runs, so no repack is needed.
    let mut q_l2 = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    let mut k_l2 = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    // One launch for both norms (l2_norm2_f32; lane/launch-collapse-20260906): same per-row
    // body and block shape as the two l2_norm launches it replaces, byte-identical rows.
    e.l2_norm_pair(
        &q_conv,
        &mut q_l2,
        &k_conv,
        &mut k_l2,
        head_dim,
        t * heads,
        KDA_L2_EPS,
    )?;
    kda_trace(
        e,
        "conv+l2",
        t,
        &[
            ("q_conv", &q_conv),
            ("k_conv", &k_conv),
            ("v_conv", &v_conv),
            ("q_l2", &q_l2),
            ("k_l2", &k_l2),
        ],
    );
    // Door W: the convs' last readers were the l2 norms (the ring rolls read the raws).
    if rows_exact {
        e.vws_recycle(q_conv);
        e.vws_recycle(k_conv);
    }

    // Stage 4 — gates. forget: g = lower_bound * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias)),
    // emitted RAW (the scan applies expf). beta: per-head sigmoid of its own projection.
    let forget = if rows_exact {
        e.matmul_rows_exact(&la.f_b, &forget_down, t)?
    } else {
        matmul_lowrank(e, &la.f_b, &forget_down, t, "f_b")?
    };
    let mut g_log = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    let mut beta = if rows_exact {
        e.vws_uninit(t * heads)?
    } else {
        e.uninit(t * heads)?
    };
    // Forget gate and beta sigmoid in one launch (memra_kda_gate_beta_f32;
    // lane/launch-collapse-20260906): both bodies verbatim on their own tensors.
    e.kda_gate_beta(
        &forget,
        la.dt_bias.float_data(),
        la.a_log.float_data(),
        &mut g_log,
        &beta_raw,
        &mut beta,
        qkv,
        t,
        head_dim,
        la.plan.gate_lower_bound,
    )?;
    kda_trace(e, "gates", t, &[("forget", &forget), ("beta", &beta)]);
    // Door W: forget_down's last reader was the f_b matmul, forget's the gate kernel,
    // beta_raw's the sigmoid.
    if rows_exact {
        e.vws_recycle(forget_down);
        e.vws_recycle(forget);
        e.vws_recycle(beta_raw);
    }

    // Stage 5 — the delta-rule recurrence. `scale` carries the reference's head_dim^-0.5 query
    // scale: q feeds only the readout, never the state, so scaling the readout is exact.
    // At t > 1 the kernel walks the T steps IN-KERNEL over register-resident state — the
    // sequential chain preserved inside ONE launch (chained-t=1 identity by construction,
    // held by the scan-chain bit-gate). `scan_clock` is the trace-level-2 instrument: it
    // drains the stream around the launch so the sequential-class share lands in its own
    // bucket (shares, never walls).
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut core = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    let scan_t0 = scan_clock.as_ref().map(|_| {
        let _ = e.stream().synchronize();
        std::time::Instant::now()
    });
    e.kda_scan(
        &q_l2, &k_l2, &v_conv, &g_log, &beta, state_in, state_out, &mut core, heads, t, scale,
    )?;
    kda_trace(
        e,
        "scan",
        t,
        &[
            ("state_in", state_in),
            ("core", &core),
            ("state_out", state_out),
        ],
    );
    if let (Some(ns), Some(t0)) = (scan_clock.take(), scan_t0) {
        let _ = e.stream().synchronize();
        *ns += t0.elapsed().as_nanos() as u64;
    }

    // Stage 6 — sigmoid-gated RMSNorm over head_dim (layer rms eps here, NOT the l2 eps), then
    // the output projection.
    let gate = if rows_exact {
        e.matmul_rows_exact(&la.g_b, &gate_down, t)?
    } else {
        matmul_lowrank(e, &la.g_b, &gate_down, t, "g_b")?
    };
    let mut gated = if rows_exact {
        e.vws_uninit(t * qkv)?
    } else {
        e.uninit(t * qkv)?
    };
    match onorm_q8 {
        // MEMRA_KDA_ONORM_ZQ8: the fused norm+quantize twin hands `wo` its q8_1 pair.
        Some(slot) if !rows_exact && head_dim.is_multiple_of(32) => {
            let pair = e.kda_gated_rmsnorm_zq8(
                &core,
                la.o_norm.float_data(),
                &gate,
                &mut gated,
                head_dim,
                t * heads,
                eps,
            )?;
            *slot = Some(pair);
        }
        _ => e.kda_gated_rmsnorm(
            &core,
            la.o_norm.float_data(),
            &gate,
            &mut gated,
            head_dim,
            t * heads,
            eps,
        )?,
    }
    kda_trace(e, "gated_norm", t, &[("gate", &gate), ("gated", &gated)]);
    // Door W: gate_down's last reader was the g_b matmul; core's and gate's the
    // gated-rmsnorm above.
    if rows_exact {
        e.vws_recycle(gate_down);
        e.vws_recycle(gate);
        e.vws_recycle(core);
    }
    // Steal the scan/conv inputs for the caller's rollback stash: stage 5 has consumed
    // the scan inputs and the rolls were the raws' last readers — moving them out is
    // free (no copy, no launch; the buffers were allocated this call either way).
    match stash {
        KdaStash::None => {}
        KdaStash::Decode(s) => {
            *s = Some(KdaScanInputs {
                q: q_l2,
                k: k_l2,
                v: v_conv,
                g: g_log,
                beta,
            });
        }
        KdaStash::Rows(s) => {
            // Door W: the PREVIOUS round's stash dies here — its nine buffers restock
            // the pool instead of falling to nine async frees (per layer per round).
            if let Some(old) = s.take() {
                e.vws_recycle(old.ring_snap);
                for r in old.raws {
                    e.vws_recycle(r);
                }
                e.vws_recycle(old.scan.q);
                e.vws_recycle(old.scan.k);
                e.vws_recycle(old.scan.v);
                e.vws_recycle(old.scan.g);
                e.vws_recycle(old.scan.beta);
            }
            *s = Some(KdaRowsStash {
                ring_snap: ring_snap.expect("rows arm snapshotted the ring above"),
                raws: [q_raw, k_raw, v_raw],
                scan: KdaScanInputs {
                    q: q_l2,
                    k: k_l2,
                    v: v_conv,
                    g: g_log,
                    beta,
                },
                rows: t,
            });
        }
    }
    Ok(gated)
}

/// STATELESS prefill from a zero conv ring and a zero recurrent state — the arm the logits-only
/// forward paths take. Allocates and discards both state buffers.
pub fn kda_attn(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let mut ring = e.zeros(la.conv_width() * (la.conv_kernel() - 1))?;
    let state_in = e.zeros(la.state_width())?;
    let mut state_out = e.zeros(la.state_width())?;
    kda_core(
        e,
        la,
        x,
        t,
        eps,
        &mut ring,
        &state_in,
        &mut state_out,
        ConvArm::Prefill,
        KdaStash::None,
        None,
        None,
    )
}

/// STATEFUL prefill: carries the ring forward and advances the recurrent state from `state_in`
/// into `state_out`. Callers own the ping-pong; the two state buffers must be distinct.
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
pub fn kda_attn_prime(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    ring: &mut CudaSlice<f32>,
    state_in: &CudaSlice<f32>,
    state_out: &mut CudaSlice<f32>,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    kda_core(
        e,
        la,
        x,
        t,
        eps,
        ring,
        state_in,
        state_out,
        ConvArm::Prefill,
        KdaStash::None,
        None,
        None,
    )
}

/// T=1 decode step. Same math as a one-token prime; separate conv arm so the fused
/// assemble+conv+roll kernel keeps decode and the spec verify on one dispatch class.
pub fn kda_attn_decode(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    eps: f32,
    ring: &mut CudaSlice<f32>,
    state_in: &CudaSlice<f32>,
    state_out: &mut CudaSlice<f32>,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    kda_core(
        e,
        la,
        x,
        1,
        eps,
        ring,
        state_in,
        state_out,
        ConvArm::Decode,
        KdaStash::None,
        None,
        None,
    )
}

/// Stateful KDA against the shared recurrent-state carrier, in the eager GDN discipline: the
/// scan reads `ssm_state` and writes the spare `ssm_state_alt`, then the two OWNED resident
/// buffers swap in place. Stable pointers, no per-step alloc/free — the per-step scratch this
/// replaced churned the stream-ordered pool and made decode run-to-run nondeterministic
/// (crates/memra-kv `RecurLayer::ssm_state_alt`). NOT capture-safe: a captured graph bakes
/// capture-time pointers and never re-runs the host swap, which is why the capture loops refuse.
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
fn kda_cached(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    cache: &mut Cache,
    il: usize,
    arm: ConvArm,
    stash: KdaStash<'_>,
    scan_clock: Option<&mut u64>,
    pre_q8: KdaPreQ8<'_>,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let rl = cache.recur[il].as_mut().ok_or_else(|| {
        format!(
            "blk.{il}: KDA layer has no recurrent state — the cache allocator saw a \
                 non-Recurrent StatePlan for a KDA layer"
        )
    })?;
    let out = {
        let RecurLayer {
            conv_state,
            ssm_state,
            ssm_state_alt,
        } = rl;
        kda_core(
            e,
            la,
            x,
            t,
            eps,
            conv_state,
            ssm_state,
            ssm_state_alt,
            arm,
            stash,
            scan_clock,
            pre_q8,
        )?
    };
    std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
    Ok(out)
}

/// Stateful prefill of `t` tokens through the cache's KDA state for layer `il`.
pub fn kda_prime_cached(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    cache: &mut Cache,
    il: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    kda_cached(
        e,
        la,
        x,
        t,
        eps,
        cache,
        il,
        ConvArm::Prefill,
        KdaStash::None,
        None,
        None,
    )
}

/// One decode step through the cache's KDA state for layer `il`.
/// `MEMRA_KDA_CONV3` (lane/glm5-kda-conv3-20260904; default ON on sm_100a builds since 2026-09-04,
/// OFF elsewhere, `=0`/`=1` override): the T=1 KDA conv+SiLU runs its three planes in one launch. Read PER CALL. Why and receipts: the kernel header in
/// cu/kda.cu and docs/FLAGS.md.
pub(crate) fn kda_conv3_on() -> bool {
    kda_conv3_on_from(
        std::env::var("MEMRA_KDA_CONV3").ok().as_deref(),
        env!("MEMRA_BUILT_CUDA_ARCH"),
    )
}

/// The pure parse behind [`kda_conv3_on`]: `1` arms, `0` disarms, unset follows the BUILD ARCH
/// (ON for `100a`, OFF otherwise): the fused launch carries a 2x B200 receipt (+1.82% at c1,
/// darklanes research/glm5-b200-20260902/LANE.md, convab) and no SM120 one.
pub fn kda_conv3_on_from(v: Option<&str>, built_arch: &str) -> bool {
    match v.map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => built_arch == "100a",
    }
}

/// Engagement counter for `MEMRA_KDA_CONV3`; gates take a delta.
pub static KDA_CONV3_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Snapshot of [`KDA_CONV3_DISPATCHES`].
pub fn kda_conv3_dispatches() -> u64 {
    KDA_CONV3_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// A pre-quantized q8_1 view of the mixer input (`(aq, ad)` from `rms_norm_zq8_f32`), or
/// `None` for the launcher to quantize itself. Threaded from the walk to the fused
/// six-projection launcher (`MEMRA_GLM5_Q8_FUSE_ATTN`, lane/glm5-attn-norm-zq8-20260904).
pub type KdaPreQ8<'a> = Option<(&'a CudaSlice<i8>, &'a CudaSlice<f32>)>;

/// `MEMRA_KDA_ONORM_ZQ8=1` (lane/kda-onorm-zq8-20260905, default OFF pending its model-scale row):
/// the decode (t=1, non-rows) KDA core emits the o_norm output's q8_1 pair from the gated-norm
/// launch itself (`memra_kda_gated_rmsnorm_zq8_f32`) and hands it to the `wo` MMVQ through
/// `matmul_q8_fast`, dropping the standalone `quantize_q8_1` launch (34 per token on
/// GLM-5.3-Flash, in-graph). BIT-IDENTICAL: same norm bytes, same q8_1 arithmetic as
/// `quantize_q8_1` (gate `tests/kda_onorm_zq8_gpu.rs` + the decode-graph fixture arm). When `wo`
/// is not MMVQ-fast-eligible the pair is dropped and `matmul` runs unchanged. Read per call.
pub fn kda_onorm_zq8_on() -> bool {
    crate::b200_posture_door_from(
        std::env::var("MEMRA_KDA_ONORM_ZQ8").ok().as_deref(),
        env!("MEMRA_BUILT_CUDA_ARCH"),
    )
}

/// Launches of the fused o_norm+quantize kernel whose pair the `wo` MMVQ consumed.
pub static KDA_ONORM_ZQ8_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The o_norm q8_1 pair handed from [`kda_core_gated`] to the `wo` projection.
pub type KdaOnormQ8 = Option<(CudaSlice<i8>, CudaSlice<f32>)>;

/// `MEMRA_KDA_NARROW_Q8=1` (lane/kda-narrow-q8-20260905, default OFF pending its model-scale row):
/// the decode KDA core's low-rank `f_b` / `g_b` projections (128-wide inputs) ride
/// `qmatvec_q8_0_mmvq_f32in_narrow`, which quantizes the row inside the launch, instead of
/// `quantize_q8_1` + `qmatvec_q8_0_mmvq` (two launches per projection, 68 per token, in-graph).
/// BIT-IDENTICAL: quantize_q8_1's arithmetic then the mmvq body verbatim (gate
/// `tests/q8_narrow_f32in_gpu.rs`). Shapes or tensors that do not fit keep `matmul`. Read per call.
pub fn kda_narrow_q8_on() -> bool {
    std::env::var("MEMRA_KDA_NARROW_Q8").as_deref() == Ok("1")
}

/// Launches of the narrow in-kernel-quantize MMVQ on the f_b / g_b sites.
pub static KDA_NARROW_Q8_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `matmul` for a decode-arm low-rank projection, through the narrow in-kernel-quantize twin
/// when the door is on and the shape fits, else the unchanged `matmul`.
fn matmul_lowrank(
    e: &Engine,
    w: &crate::model::GpuTensor,
    x: &CudaSlice<f32>,
    t: usize,
    which: &str,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    if kda_narrow_q8_on()
        && let Some(y) = e.matmul_q8_narrow_f32in(w, x, t)?
    {
        if KDA_NARROW_Q8_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            eprintln!(
                "[kda-narrow-q8] engaged: the KDA {which} projection quantizes its {}-wide input \
                 inside the MMVQ launch (MEMRA_KDA_NARROW_Q8=1)",
                w.in_features()
            );
        }
        return Ok(y);
    }
    e.matmul(w, x, t)
}

/// [`kda_decode_cached`] with the mixer input's q8_1 view already emitted by the caller's norm
/// (`MEMRA_GLM5_Q8_FUSE_ATTN`): identical launches minus the fused launcher's own quantize.
pub fn kda_decode_cached_q8(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    pre_q8: KdaPreQ8<'_>,
    eps: f32,
    cache: &mut Cache,
    il: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    kda_cached(
        e,
        la,
        x,
        1,
        eps,
        cache,
        il,
        ConvArm::Decode,
        KdaStash::None,
        None,
        pre_q8,
    )
}

pub fn kda_decode_cached(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    eps: f32,
    cache: &mut Cache,
    il: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    kda_cached(
        e,
        la,
        x,
        1,
        eps,
        cache,
        il,
        ConvArm::Decode,
        KdaStash::None,
        None,
        None,
    )
}

/// [`kda_decode_cached`] with the step's scan inputs STOLEN for a rollback stash
/// (loop-port 3; doc on [`KdaScanInputs`]). Identical launches — the steal is a move of
/// buffers the step allocated either way.
pub fn kda_decode_cached_stash(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    eps: f32,
    cache: &mut Cache,
    il: usize,
) -> Result<(CudaSlice<f32>, KdaScanInputs), Box<dyn std::error::Error>> {
    let mut stash: Option<KdaScanInputs> = None;
    let out = kda_cached(
        e,
        la,
        x,
        1,
        eps,
        cache,
        il,
        ConvArm::Decode,
        KdaStash::Decode(&mut stash),
        None,
        None,
    )?;
    let stash = stash.ok_or("kda_core returned without filling the requested scan stash")?;
    Ok((out, stash))
}

/// THE BATCHED VERIFY-ROWS KDA CALL (lane/glm5-verify-batch): one t=K+1 `kda_core` pass
/// per layer per round, replacing t per-row [`kda_decode_cached_stash`] calls. Projections,
/// gates and norms batch m=t through the decode-exact matmul classes (`matmul_rows_exact`);
/// the conv takes the prefill dispatch (per-token bit-identical to the decode arm's taps);
/// the recurrence stays SEQUENTIAL inside one `memra_kda_scan_s128` launch (the in-kernel
/// T-loop over register-resident state == the chained t=1 program). Per-row bit-identity
/// vs the t=1 chain is held by the walk gates (`glm5_tparallel_verify_gpu`) and the
/// kernel bit-gates (`glm5_verify_batch_gpu`).
///
/// The caller owns the pre-round ssm snapshot (`Glm5VerifyCkpt::kda_ssm_snap`, cloned
/// BEFORE this call); the returned [`KdaRowsStash`] carries everything else rollback
/// needs. `scan_clock`: the trace-level-2 sequential-class bucket (ns accumulated around
/// the scan launch with stream drains — an instrument, never a serving mode).
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kda_cached call contract plus the trace clock
pub fn kda_verify_rows_cached(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    cache: &mut Cache,
    il: usize,
    scan_clock: Option<&mut u64>,
) -> Result<(CudaSlice<f32>, KdaRowsStash), Box<dyn std::error::Error>> {
    let mut stash: Option<KdaRowsStash> = None;
    let out = kda_cached(
        e,
        la,
        x,
        t,
        eps,
        cache,
        il,
        ConvArm::Prefill,
        KdaStash::Rows(&mut stash),
        scan_clock,
        None,
    )?;
    let stash = stash.ok_or("kda_core returned without filling the requested rows stash")?;
    Ok((out, stash))
}

/// Roll layer `il` back to "after row `keep-1`" from a BATCHED verify-rows round
/// (lane/glm5-verify-batch; the [`KdaRowsStash`] doc states the two-plane contract):
/// restore the pre-round conv ring and re-roll `keep` raw rows (pure placement), then
/// replay the scan ONCE at T=keep from the pre-round ssm snapshot over the batched
/// inputs. Full accept (`keep == rows`) never calls this — the resident state IS the
/// state after the last kept row.
pub fn kda_verify_rollback_rows(
    e: &Engine,
    la: &KdaAttnLayer,
    snap: &CudaSlice<f32>,
    stash: &KdaRowsStash,
    keep: usize,
    cache: &mut Cache,
    il: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let rl = cache.recur[il]
        .as_mut()
        .ok_or_else(|| format!("blk.{il}: KDA rows rollback on a layer with no recurrent state"))?;
    kda_verify_rollback_rows_on(e, la, snap, stash, keep, rl, il)
}

/// [`kda_verify_rollback_rows`] over a CALLER-OWNED state plane — the glm5 spec x TP seam
/// (lane/glm5-composition): under `MEMRA_GLM5_TP` each rank's shard-geometry conv ring +
/// ssm ping-pong lives in `cache.glm5_tp_recur[il][rank]` on that rank's engine, so the
/// rollback restores per rank through this entry with the rank's own `(engine, shard,
/// snapshot, stash)` tuple. The cache wrapper above delegates here — one body, byte-for-byte
/// the pre-refactor walk on the plain path.
pub fn kda_verify_rollback_rows_on(
    e: &Engine,
    la: &KdaAttnLayer,
    snap: &CudaSlice<f32>,
    stash: &KdaRowsStash,
    keep: usize,
    rl: &mut RecurLayer,
    il: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if keep == 0 || keep >= stash.rows {
        return Err(format!(
            "blk.{il}: KDA rows rollback keep={keep} outside 1..{} (full accept keeps the \
             resident state and never replays)",
            stash.rows
        )
        .into());
    }
    let qkv = la.qkv();
    let kernel = la.conv_kernel();
    let heads = la.heads();
    let scale = 1.0 / (la.head_dim() as f32).sqrt();
    // Conv ring: pre-round snapshot back, then re-roll the kept raw rows per plane. The
    // roll kernel reads every old slot into registers before any store, so T=keep < pad
    // mixes snapshot slots and kept rows exactly as the sequential chain's rolls did.
    e.copy_into(
        &mut rl.conv_state,
        0,
        &stash.ring_snap,
        stash.ring_snap.len(),
    )?;
    for (plane, raw) in stash.raws.iter().enumerate() {
        e.kda_conv_ring_roll(raw, &mut rl.conv_state, qkv, keep, kernel, plane)?;
    }
    // Recurrent state: ONE T=keep replay from the snapshot over the batched scan inputs
    // (the kernel walks rows 0..keep of the [t, ..] buffers); readout discarded. The
    // ping-pong ends with the rebuilt state under the `ssm_state` name, matching
    // `kda_cached`'s swap discipline.
    let mut o = e.uninit(keep * qkv)?;
    {
        let RecurLayer {
            ssm_state: _,
            ssm_state_alt,
            ..
        } = rl;
        e.kda_scan(
            &stash.scan.q,
            &stash.scan.k,
            &stash.scan.v,
            &stash.scan.g,
            &stash.scan.beta,
            snap,
            ssm_state_alt,
            &mut o,
            heads,
            keep,
            scale,
        )?;
    }
    std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
    Ok(())
}

/// Rebuild layer `il`'s recurrent state to "after row `inputs.len()-1`" by REPLAYING the
/// stashed scan inputs from the pre-round snapshot `snap` (loop-port 3, the module-doc
/// diet made concrete): each replay is the original t=1 `memra_kda_scan_s128` launch
/// re-issued over the very buffers that step consumed, so the rebuilt state is
/// byte-identical to the per-row clone it replaces BY CONSTRUCTION. The readout is
/// discarded; the conv ring is not touched (the walk still clones it per row — 288 KiB
/// against the 4 MiB ssm plane this retires). The ping-pong rides the resident pair and
/// ends with the rebuilt state under the `ssm_state` name, matching `kda_cached`'s own
/// swap discipline.
pub fn kda_scan_replay(
    e: &Engine,
    la: &KdaAttnLayer,
    snap: &CudaSlice<f32>,
    inputs: &[KdaScanInputs],
    cache: &mut Cache,
    il: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        return Err(format!(
            "blk.{il}: KDA replay needs at least one stashed row (rollback keep >= 1; a \
             restore TO the snapshot itself is a different contract)"
        )
        .into());
    }
    if la.tp.is_some() {
        return Err(format!(
            "blk.{il}: KDA scan replay (the PER-ROW rollback seam) is unwired for a \
             glm5-TP-sharded layer — the spec x TP composition requires the BATCHED \
             verify walk, whose rollback rides kda_verify_rollback_rows_on per rank"
        )
        .into());
    }
    let heads = la.heads();
    let scale = 1.0 / (la.head_dim() as f32).sqrt();
    let qkv = la.qkv();
    let rl = cache.recur[il]
        .as_mut()
        .ok_or_else(|| format!("blk.{il}: KDA replay on a layer with no recurrent state"))?;
    let mut o = e.uninit(qkv)?; // discarded readout scratch, reused across rows
    for (r, inp) in inputs.iter().enumerate() {
        {
            let RecurLayer {
                ssm_state,
                ssm_state_alt,
                ..
            } = rl;
            let state_in: &CudaSlice<f32> = if r == 0 { snap } else { ssm_state };
            e.kda_scan(
                &inp.q,
                &inp.k,
                &inp.v,
                &inp.g,
                &inp.beta,
                state_in,
                ssm_state_alt,
                &mut o,
                heads,
                1,
                scale,
            )?;
        }
        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
    }
    Ok(())
}

impl Engine {
    /// Per-plane causal short conv + SiLU over a T-token chunk (cu/kda.cu).
    #[allow(clippy::too_many_arguments)]
    pub fn kda_conv_silu(
        &self,
        x_tm: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        ring: &CudaSlice<f32>,
        y_tm: &mut CudaSlice<f32>,
        qkv: usize,
        t: usize,
        kernel: usize,
        plane: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("memra_kda_conv_silu_f32");
        let cfg = LaunchConfig {
            grid_dim: (qkv.div_ceil(256) as u32, t as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n, tt, k, p) = (qkv as i32, t as i32, kernel as i32, plane as i32);
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(x_tm)
            .arg(w)
            .arg(ring)
            .arg(&mut *y_tm)
            .arg(&n)
            .arg(&tt)
            .arg(&k)
            .arg(&p);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// Roll one plane of the fused conv ring forward over a T-token chunk (cu/kda.cu).
    pub fn kda_conv_ring_roll(
        &self,
        x_tm: &CudaSlice<f32>,
        ring: &mut CudaSlice<f32>,
        qkv: usize,
        t: usize,
        kernel: usize,
        plane: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("memra_kda_conv_ring_roll_f32");
        let cfg = LaunchConfig {
            grid_dim: (qkv.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n, tt, k, p) = (qkv as i32, t as i32, kernel as i32, plane as i32);
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(x_tm).arg(&mut *ring).arg(&n).arg(&tt).arg(&k).arg(&p);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// T=1 fused assemble + conv + SiLU + ring roll for one plane (cu/kda.cu).
    #[allow(clippy::too_many_arguments)]
    /// The three-plane form of [`Engine::kda_conv_silu_decode`] (door `MEMRA_KDA_CONV3`,
    /// lane/glm5-kda-conv3-20260904): one launch with `plane = blockIdx.y` in place of the three
    /// per-plane launches; per channel the same body, so outputs and the ring are bit-identical
    /// (gate `tests/kda_conv3_gpu.rs`). `kernel` (K) must be at most 9 (the `win[8]` window).
    #[allow(clippy::too_many_arguments)]
    pub fn kda_conv_silu_decode3(
        &self,
        x: [&CudaSlice<f32>; 3],
        ring: &mut CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: [&mut CudaSlice<f32>; 3],
        qkv: usize,
        kernel: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if kernel == 0 || kernel > 9 {
            return Err("kda_conv_silu_decode3: kernel width outside the 8-wide window".into());
        }
        if KDA_CONV3_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            eprintln!(
                "[kda-conv3] engaged: the three KDA conv+SiLU decode planes run as one launch \
                 (MEMRA_KDA_CONV3=1)"
            );
        }
        let f = self.func("memra_kda_conv_silu_decode3_f32");
        let cfg = LaunchConfig {
            grid_dim: (qkv.div_ceil(256) as u32, 3, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n, k) = (qkv as i32, kernel as i32);
        let [x0, x1, x2] = x;
        let [y0, y1, y2] = y;
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(x0)
            .arg(x1)
            .arg(x2)
            .arg(&mut *ring)
            .arg(w)
            .arg(&mut *y0)
            .arg(&mut *y1)
            .arg(&mut *y2)
            .arg(&n)
            .arg(&k);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kda_conv_silu_decode(
        &self,
        x_new: &CudaSlice<f32>,
        ring: &mut CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        qkv: usize,
        kernel: usize,
        plane: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("memra_kda_conv_silu_decode_f32");
        let cfg = LaunchConfig {
            grid_dim: (qkv.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n, k, p) = (qkv as i32, kernel as i32, plane as i32);
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(x_new)
            .arg(&mut *ring)
            .arg(w)
            .arg(&mut *y)
            .arg(&n)
            .arg(&k)
            .arg(&p);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// Per-channel forget gate, emitted as the RAW log-gate (cu/kda.cu).
    #[allow(clippy::too_many_arguments)]
    pub fn kda_gate(
        &self,
        forget: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        a_log: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        qkv: usize,
        t: usize,
        head_dim: usize,
        lower_bound: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("memra_kda_gate_f32");
        let cfg = LaunchConfig {
            grid_dim: (qkv.div_ceil(256) as u32, t as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n, tt, hd, lb) = (qkv as i32, t as i32, head_dim as i32, lower_bound);
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(forget)
            .arg(dt_bias)
            .arg(a_log)
            .arg(&mut *g)
            .arg(&n)
            .arg(&tt)
            .arg(&hd)
            .arg(&lb);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// The per-channel-decay delta-rule scan (cu/kda.cu). One warp per output column.
    /// [`Engine::kda_gate`] and `sigmoid(beta_raw)` in one launch (`memra_kda_gate_beta_f32`):
    /// the forget-gate body and sigmoid_f32's expression verbatim on their own tensors, same grid
    /// as `kda_gate`; both outputs byte-identical to the two launches. Gate
    /// `tests/kda_small_folds_gpu.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn kda_gate_beta(
        &self,
        forget: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        a_log: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        beta_raw: &CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
        qkv: usize,
        t: usize,
        head_dim: usize,
        lower_bound: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let heads = qkv / head_dim;
        if heads * head_dim != qkv
            || heads > qkv
            || beta_raw.len() < t * heads
            || beta.len() < t * heads
        {
            return Err(format!(
                "kda_gate_beta: qkv={qkv} head_dim={head_dim} beta rows {}",
                beta_raw.len()
            )
            .into());
        }
        let f = self.func("memra_kda_gate_beta_f32");
        let cfg = LaunchConfig {
            grid_dim: (qkv.div_ceil(256) as u32, t as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n, tt, hd, hh, lb) = (
            qkv as i32,
            t as i32,
            head_dim as i32,
            heads as i32,
            lower_bound,
        );
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(forget)
            .arg(dt_bias)
            .arg(a_log)
            .arg(&mut *g)
            .arg(beta_raw)
            .arg(&mut *beta)
            .arg(&n)
            .arg(&tt)
            .arg(&hd)
            .arg(&hh)
            .arg(&lb);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kda_scan(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state_in: &CudaSlice<f32>,
        state_out: &mut CudaSlice<f32>,
        o: &mut CudaSlice<f32>,
        heads: usize,
        t: usize,
        scale: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Four columns per block keeps one warp per column at 128 threads, the same shape
        // gdn_scan_s128 launches with.
        const COLS_PER_BLOCK: u32 = 4;
        let f = self.func("memra_kda_scan_s128");
        let cfg = LaunchConfig {
            grid_dim: (
                heads as u32,
                1,
                (KDA_HEAD_DIM as u32).div_ceil(COLS_PER_BLOCK),
            ),
            block_dim: (32, COLS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let (h, tt, s) = (heads as i32, t as i32, scale);
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(q)
            .arg(k)
            .arg(v)
            .arg(g)
            .arg(beta)
            .arg(state_in)
            .arg(&mut *state_out)
            .arg(&mut *o)
            .arg(&h)
            .arg(&tt)
            .arg(&s);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// Sigmoid-gated fp32 RMSNorm over head_dim (cu/kda.cu). GDN's `gated_rmsnorm` gates with
    /// SiLU; KDA's Glm5NextTextRMSNormGated hardcodes sigmoid.
    #[allow(clippy::too_many_arguments)]
    pub fn kda_gated_rmsnorm(
        &self,
        core: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        ncols: usize,
        nrows: usize,
        eps: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("memra_kda_gated_rmsnorm_f32");
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nc, ep) = (ncols as i32, eps);
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(core)
            .arg(w)
            .arg(gate)
            .arg(&mut *dst)
            .arg(&nc)
            .arg(&ep);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// [`Engine::kda_gated_rmsnorm`] emitting the q8_1 pair of `dst` beside it
    /// (`memra_kda_gated_rmsnorm_zq8_f32`): `dst` byte-identical to the plain kernel, the pair
    /// byte-identical to `quantize_q8_1(dst, t, heads*ncols)` (the `wo` MMVQ input). Returns
    /// `(q [nrows*ncols], d [nrows*ncols/32])`, which viewed per token is exactly the
    /// `[t, heads*ncols]` activation's q8_1 pair. Requires `ncols % 32 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn kda_gated_rmsnorm_zq8(
        &self,
        core: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        ncols: usize,
        nrows: usize,
        eps: f32,
    ) -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        if !ncols.is_multiple_of(32) {
            return Err(format!("kda_gated_rmsnorm_zq8 needs ncols % 32 == 0, got {ncols}").into());
        }
        let mut q = self.alloc_i8_uninit(nrows * ncols)?;
        let mut d = self.uninit(nrows * ncols / 32)?;
        let f = self.func("memra_kda_gated_rmsnorm_zq8_f32");
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nc, ep) = (ncols as i32, eps);
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(core)
            .arg(w)
            .arg(gate)
            .arg(&mut *dst)
            .arg(&mut q)
            .arg(&mut d)
            .arg(&nc)
            .arg(&ep);
        unsafe { b.launch(cfg)? };
        Ok((q, d))
    }

    /// The `MEMRA_KDA_FUSED_PROJ` door: run the KDA stage-1 six-projection group as ONE
    /// `quantize_q8_1` + ONE `qmatvec_kda6_q8f32_mmvq` launch, or return `None` and let the
    /// caller take the unchanged `matmul_group` arm.
    ///
    /// ENGAGEMENT IS DELIBERATELY NARROW — every condition below exists so the door's numeric
    /// claim stays exactly what the gate proves (`tests/kda_fused_proj_gpu.rs`):
    ///  * wq/wk/wv must be plain-layout Q8_0 (`rp: false`, no `rp4` mirror, `scale == 1.0`) —
    ///    the fused kernel's per-(token,row) body is `qmatvec_q8_0_mmvq` VERBATIM, so those
    ///    rows are BIT-IDENTICAL to the unfused MMVQ/batched arm; a repacked layout would ride
    ///    the `_rp` twins instead and the claim would be against the wrong kernel.
    ///  * f_a/g_a/b_proj must be f32 `Float` — their fused rows replace cuBLASLt with a
    ///    deterministic warp tree: a reduction-order class change (the step37 QKV_FUSED class),
    ///    measured and pinned in the gate.
    ///  * t in 1..=15 (the batch cap), and the env classes under which the UNFUSED arm rides
    ///    the MMVQ-class per-row program: `MEMRA_FAST!=0`, `mmvq_supports(Q8_0)`,
    ///    `MEMRA_NO_BATCHED` unset for t>=2, `MEMRA_B8!=0` for t>=5. Outside those envs the
    ///    unfused arm is a different kernel class (dp4a / Stage-A), so the door refuses rather
    ///    than weakening its identity claim.
    ///
    /// The flag is read PER CALL (the `MEMRA_MOE_FUSED_EPI` rollback-seam precedent), so both
    /// arms alternate inside one process. Output order matches `matmul_group`'s:
    /// `[q, k, v, forget_down, gate_down, beta_raw]`.
    pub fn kda_proj_fused6(
        &self,
        la: &KdaAttnLayer,
        x: &CudaSlice<f32>,
        t: usize,
    ) -> Result<Option<Vec<CudaSlice<f32>>>, Box<dyn std::error::Error>> {
        self.kda_proj_fused6_pre(la, x, t, None)
    }

    /// [`Engine::kda_proj_fused6`] with an optional pre-quantized activation for the W8 arm
    /// (`MEMRA_GLM5_Q8_FUSE_ATTN`); every other arm ignores it (they quantize per projection or
    /// run bf16) and stays the program it was.
    pub fn kda_proj_fused6_pre(
        &self,
        la: &KdaAttnLayer,
        x: &CudaSlice<f32>,
        t: usize,
        pre_q8: KdaPreQ8<'_>,
    ) -> Result<Option<Vec<CudaSlice<f32>>>, Box<dyn std::error::Error>> {
        if std::env::var("MEMRA_KDA_FUSED_PROJ").as_deref() != Ok("1") {
            return Ok(None);
        }
        // glm5 TP composition (2026-09-07): a head shard's six projections are ROW SUBSETS
        // of the full-width six (`shard_kda_layer` takes `shard_rows` on wq/wk/wv/b_proj and
        // replicates f_a/g_a; same in_f, same row_bytes), and this kernel's per-row dot is
        // the same program whichever rows it is handed, so the fused group engages on the
        // shards too. Until then the TP walk ran the six as five separate matvecs per layer
        // per rank (3 e4m3 + ~2.3 q8_0 launches, tptrace3). Receipt: the pair's tape with the
        // door on equals the tape with it off (tpwalk4 `symgraphPF`).
        if !(1..=15).contains(&t) {
            return Ok(None);
        }
        // E4M3 SIX-GROUP ARM (lane/glm5-b200-mint-consume, 2026-09-04) — the operand class the
        // GLM-5.3-Flash B200 hybrid mint actually ships. That mint quantizes ALL SIX KDA
        // projections to per-tensor e4m3, so under MEMRA_ST_E4M3 (default ON) every one of them
        // is QT_F8_E4M3-resident at 1.0 B/weight: cheaper than the bf16 serving recipe's 2.0 and
        // cheaper than the Q8_0 re-encode's 1.0625, with no lossy re-quant hop. The two arms
        // below cannot serve that shape — they require a FloatBf16 trio plus an f32 trio — so
        // without this arm the mint's cheapest operand would fall to SIX separate launches on
        // each of the 34 KDA layers, with six redundant broadcasts of the same activation.
        //
        // This is an operand arm of an EXISTING door, not a new one: it rides
        // MEMRA_KDA_FUSED_PROJ=1 exactly as the bf16 and q8rp arms do, and it additionally
        // declines wherever the unfused e4m3 program it claims bit-identity against would not
        // be the shipped one (MEMRA_FAST=0, no MMVQ support for the qtype, or the
        // MEMRA_E4M3_DUAL=0 rollback that restores per-tensor launches).
        //
        // m=1 ONLY. `qmatvec_e4m3_mmvq_fused6` pins the token index at 0, matching the
        // `e4m3_mmvq_row1` body it shares with the pair and triple. t>1 keeps the caller's arm.
        let e4m3 = |w: &GpuTensor| -> Option<(usize, usize, f32)> {
            match w {
                GpuTensor::Quant {
                    qtype: crate::QT_F8_E4M3,
                    row_bytes,
                    scale,
                    rp: false,
                    rp4: None,
                    blk: None,
                    ..
                } => Some((w.in_features(), *row_bytes, *scale)),
                _ => None,
            }
        };
        if let (Some(e_q), Some(e_k), Some(e_v), Some(e_fa), Some(e_ga), Some(e_b)) = (
            e4m3(&la.wq),
            e4m3(&la.wk),
            e4m3(&la.wv),
            e4m3(&la.f_a),
            e4m3(&la.g_a),
            e4m3(&la.b_proj),
        ) {
            let six = [e_q, e_k, e_v, e_fa, e_ga, e_b];
            let in_f = e_q.0;
            if t != 1
                || std::env::var("MEMRA_FAST").as_deref() == Ok("0")
                || !self.mmvq_supports(crate::QT_F8_E4M3)
                || !self.e4m3_dual_on()
                // Every range must share in_f and the q8_1 activation block, and an e4m3 row is
                // exactly in_f bytes — a row_bytes that disagrees means a padded or foreign
                // layout this kernel's single `row_bytes` cannot address.
                || six.iter().any(|&(i, rb, _)| i != in_f || rb != in_f)
                || !in_f.is_multiple_of(32)
                || x.len() < in_f
            {
                return Ok(None);
            }
            let dims = [
                la.wq.out_features(),
                la.wk.out_features(),
                la.wv.out_features(),
                la.f_a.out_features(),
                la.g_a.out_features(),
                la.b_proj.out_features(),
            ];
            let ws: [f32; 6] = std::array::from_fn(|i| six[i].2);
            fn e4m3_bytes(w: &GpuTensor) -> &CudaSlice<u8> {
                match w {
                    GpuTensor::Quant { bytes, .. } => bytes,
                    _ => unreachable!("e4m3() above only admits Quant"),
                }
            }
            let bytes = e4m3_bytes;
            let w = [
                bytes(&la.wq),
                bytes(&la.wk),
                bytes(&la.wv),
                bytes(&la.f_a),
                bytes(&la.g_a),
                bytes(&la.b_proj),
            ];
            // ONE activation quantize for all six ranges — the six-launch path pays this per
            // projection. When the walk already quantized `x` (`MEMRA_GLM5_Q8_FUSE_ATTN`:
            // rms_norm_zq8_f32's pair IS quantize_q8_1(x) byte for byte), consume that pair; the
            // e4m3 arm reads the same q8_1 activation as the W8 arm, so nothing changes but the
            // launch count (34 per token on GLM-5.3-Flash, in-graph). Gate:
            // tests/kda_fused6_e4m3_gpu.rs `pre_q8` arm (same bytes, and a red arm proving the
            // pair is what gets read).
            let owned;
            let (aq, ad): (&CudaSlice<i8>, &CudaSlice<f32>) = match pre_q8 {
                Some((q, d)) => (q, d),
                None => {
                    owned = self.quantize_q8_1(x, 1, in_f)?;
                    (&owned.0, &owned.1)
                }
            };
            let mut outs = [
                self.uninit(dims[0])?,
                self.uninit(dims[1])?,
                self.uninit(dims[2])?,
                self.uninit(dims[3])?,
                self.uninit(dims[4])?,
                self.uninit(dims[5])?,
            ];
            self.e4m3_fused6_into(w, aq, ad, in_f, dims, in_f, ws, &mut outs)?;
            if KDA_FUSED6_E4M3_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
                eprintln!(
                    "[kda-fused6] engaged arm=e4m3 in_f={in_f} out={dims:?} t={t} (one launch \
                     replaces the six per-tensor e4m3 projections and their six redundant \
                     activation quantizes; MEMRA_KDA_FUSED_PROJ=1 MEMRA_ST_E4M3=1)"
                );
            }
            return Ok(Some(outs.into_iter().collect()));
        }
        // The f32 trio is common to both operand arms. Any mismatch = refuse; the caller's
        // arm is the shipped program.
        let f32w = |w: &GpuTensor| -> Option<usize> {
            match w {
                GpuTensor::Float { .. } => Some(w.in_features()),
                _ => None,
            }
        };
        let (Some(in_fa), Some(in_ga), Some(in_b)) =
            (f32w(&la.f_a), f32w(&la.g_a), f32w(&la.b_proj))
        else {
            return Ok(None);
        };
        // BF16 operand arm (lever 3 of the decode diet): the serving recipe (MEMRA_BF16_MMV=1)
        // admits wq/wk/wv to raw bf16 residency, where the Q8_0 arm below never binds. Its
        // bit-identity bar is against `matvec_bf16_f32acc_x4_rows` (matmul's FloatBf16
        // decode-tier arm), so it refuses wherever that arm would not be the unfused program:
        // MEMRA_BF16_MMV off (the chunked cuBLASLt GEMM class), the step37 W8 mirror doors on
        // (matvec_bf16_rows_into reroutes through the q8 mirror when BOTH are set), or
        // MEMRA_GLM5_W8 on (2026-09-02, lane/b200-glm5-w8: the SAME reroute, independent
        // door — this fused kernel's bit-identity claim is against the unmirrored bf16
        // program, so it must decline whichever door moved that program's target).
        let bf16 = |w: &GpuTensor| -> Option<usize> {
            match w {
                GpuTensor::FloatBf16 { .. } => Some(w.in_features()),
                _ => None,
            }
        };
        if let (Some(in_q), Some(in_k), Some(in_v)) = (bf16(&la.wq), bf16(&la.wk), bf16(&la.wv)) {
            // MEMRA_B200_BF16_GEMV_LT (lane/b200-gemv-hbm-20260902) reroutes the SAME
            // unfused target (`matvec_bf16_f32acc_x4_rows`) to a cuBLASLt reference GEMV, so
            // this fused arm declines for exactly the reason it declines for the W8 mirrors:
            // its bit-identity bar is against the unmirrored, unrerouted bf16 program. With
            // the door on, the three bf16 projections fall to the unfused group and each one
            // takes the library GEMV, which is what the reference door is there to measure.
            // W8 POSTURE FUSION (lane/b200-gemv-hbm-20260902 round 3). Under MEMRA_GLM5_W8 the
            // six projections each reroute through `matvec_bf16_via_q8_mirror`, so this group
            // runs as SIX separate launches plus six redundant quantizes of the same `x` — and
            // the bf16 fused arm below cannot serve it, because its bit-identity bar is against
            // the unmirrored bf16 program. `qmatvec_kda6_q8f32_rp_v2` is the fused twin for
            // that posture: three mirrored ranges on the rp v2 body (bit-identical to
            // `qmatvec_q8_0_mmvq_rp` per row) and three f32 ranges on the same deterministic
            // warp tree the q8 arm of this door already ships and has pinned. Gated on
            // MEMRA_B200_GEMV_V2 so it carries its own receipt; without that door W8 still
            // declines to the unfused path exactly as before.
            if crate::glm5_w8_on() && !(crate::step_tp_w8_on() && crate::w8_hybrid_on()) {
                if !Self::bf16_mmv_on() || !crate::b200_gemv_v2_on() {
                    return Ok(None);
                }
                let in_f = in_q;
                if [in_k, in_v, in_fa, in_ga, in_b].iter().any(|&i| i != in_f)
                    || !in_f.is_multiple_of(128)
                    || x.len() < t * in_f
                    || Engine::q8_v2_smem_bytes(in_f) > 48 * 1024
                {
                    return Ok(None);
                }
                let dims = [
                    la.wq.out_features(),
                    la.wk.out_features(),
                    la.wv.out_features(),
                    la.f_a.out_features(),
                    la.g_a.out_features(),
                    la.b_proj.out_features(),
                ];
                let (
                    GpuTensor::FloatBf16 { data: bq, .. },
                    GpuTensor::FloatBf16 { data: bk, .. },
                    GpuTensor::FloatBf16 { data: bv, .. },
                ) = (&la.wq, &la.wk, &la.wv)
                else {
                    unreachable!("bf16() above only admits FloatBf16");
                };
                let (
                    GpuTensor::Float { data: wfa, .. },
                    GpuTensor::Float { data: wga, .. },
                    GpuTensor::Float { data: wb, .. },
                ) = (&la.f_a, &la.g_a, &la.b_proj)
                else {
                    unreachable!("f32w() above only admits Float");
                };
                let mut outs = [
                    self.uninit(t * dims[0])?,
                    self.uninit(t * dims[1])?,
                    self.uninit(t * dims[2])?,
                    self.uninit(t * dims[3])?,
                    self.uninit(t * dims[4])?,
                    self.uninit(t * dims[5])?,
                ];
                self.kda_proj_fused6_q8rp_raw_pre(
                    bq,
                    bk,
                    bv,
                    wfa,
                    wga,
                    wb,
                    x,
                    &mut outs,
                    in_f,
                    dims,
                    t,
                    crate::q8_row_ilp_on(),
                    pre_q8,
                )?;
                if KDA_FUSED6_Q8RP_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
                    eprintln!(
                        "[kda-fused6] engaged arm=q8rp_v2 in_f={in_f} out={dims:?} t={t} (one \
                         launch replaces the six W8-mirror projections and their six redundant \
                         activation quantizes; MEMRA_KDA_FUSED_PROJ=1 MEMRA_B200_GEMV_V2=1)"
                    );
                }
                return Ok(Some(outs.into_iter().collect()));
            }
            if !Self::bf16_mmv_on()
                || (crate::step_tp_w8_on() && crate::w8_hybrid_on())
                || crate::b200_bf16_gemv_lt_on()
            {
                return Ok(None);
            }
            let in_f = in_q;
            if [in_k, in_v, in_fa, in_ga, in_b].iter().any(|&i| i != in_f)
                || !in_f.is_multiple_of(128)
                || x.len() < t * in_f
            {
                return Ok(None);
            }
            let dims = [
                la.wq.out_features(),
                la.wk.out_features(),
                la.wv.out_features(),
                la.f_a.out_features(),
                la.g_a.out_features(),
                la.b_proj.out_features(),
            ];
            let (
                GpuTensor::FloatBf16 { data: bq, .. },
                GpuTensor::FloatBf16 { data: bk, .. },
                GpuTensor::FloatBf16 { data: bv, .. },
            ) = (&la.wq, &la.wk, &la.wv)
            else {
                unreachable!("bf16() above only admits FloatBf16");
            };
            let (
                GpuTensor::Float { data: wfa, .. },
                GpuTensor::Float { data: wga, .. },
                GpuTensor::Float { data: wb, .. },
            ) = (&la.f_a, &la.g_a, &la.b_proj)
            else {
                unreachable!("f32w() above only admits Float");
            };
            let mut outs = [
                self.uninit(t * dims[0])?,
                self.uninit(t * dims[1])?,
                self.uninit(t * dims[2])?,
                self.uninit(t * dims[3])?,
                self.uninit(t * dims[4])?,
                self.uninit(t * dims[5])?,
            ];
            self.kda_proj_fused6_bf16_raw(bq, bk, bv, wfa, wga, wb, x, &mut outs, in_f, dims, t)?;
            if KDA_FUSED6_BF16_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
                eprintln!(
                    "[kda-fused6] engaged arm=bf16 in_f={in_f} out={dims:?} t={t} (one launch \
                     replaces the six-projection group on the bf16-resident serving recipe; \
                     MEMRA_KDA_FUSED_PROJ=1)"
                );
            }
            return Ok(Some(outs.into_iter().collect()));
        }
        // Dispatch-class envs: the bit-identity bar is against the MMVQ-class per-row program.
        if std::env::var("MEMRA_FAST").as_deref() == Ok("0")
            || !self.mmvq_supports(crate::QT_Q8_0)
            || (t >= 2 && std::env::var("MEMRA_NO_BATCHED").is_ok())
            || (t >= 5 && !Self::b8_enabled())
        {
            return Ok(None);
        }
        // Q8_0 operand classes (the non-BF16_MMV shapes).
        let q8 = |w: &GpuTensor| -> Option<(usize, usize)> {
            match w {
                GpuTensor::Quant {
                    qtype: crate::QT_Q8_0,
                    row_bytes,
                    scale,
                    rp: false,
                    rp4: None,
                    ..
                } if *scale == 1.0 => Some((w.in_features(), *row_bytes)),
                _ => None,
            }
        };
        let (Some((in_q, rb_q)), Some((in_k, rb_k)), Some((in_v, rb_v))) =
            (q8(&la.wq), q8(&la.wk), q8(&la.wv))
        else {
            return Ok(None);
        };
        let in_f = in_q;
        if [in_k, in_v, in_fa, in_ga, in_b].iter().any(|&i| i != in_f)
            || rb_k != rb_q
            || rb_v != rb_q
            || !in_f.is_multiple_of(128)
            || x.len() < t * in_f
        {
            return Ok(None);
        }
        let dims = [
            la.wq.out_features(),
            la.wk.out_features(),
            la.wv.out_features(),
            la.f_a.out_features(),
            la.g_a.out_features(),
            la.b_proj.out_features(),
        ];
        let (
            GpuTensor::Quant { bytes: bq, .. },
            GpuTensor::Quant { bytes: bk, .. },
            GpuTensor::Quant { bytes: bv, .. },
        ) = (&la.wq, &la.wk, &la.wv)
        else {
            unreachable!("q8() above only admits Quant");
        };
        let (
            GpuTensor::Float { data: wfa, .. },
            GpuTensor::Float { data: wga, .. },
            GpuTensor::Float { data: wb, .. },
        ) = (&la.f_a, &la.g_a, &la.b_proj)
        else {
            unreachable!("f32w() above only admits Float");
        };

        let (aq, ad) = self.quantize_q8_1(x, t, in_f)?;
        let mut outs = [
            self.uninit(t * dims[0])?,
            self.uninit(t * dims[1])?,
            self.uninit(t * dims[2])?,
            self.uninit(t * dims[3])?,
            self.uninit(t * dims[4])?,
            self.uninit(t * dims[5])?,
        ];
        self.kda_proj_fused6_raw(
            bq, bk, bv, wfa, wga, wb, &aq, &ad, x, &mut outs, in_f, dims, t, rb_q,
        )?;

        // Engagement receipt: counted at the arm's own call site, announced once per boot
        // (the [bf16-mmv] RESIDENT lesson: engagement lines are receipts, never inferred).
        if KDA_FUSED6_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
            eprintln!(
                "[kda-fused6] engaged in_f={in_f} out={dims:?} t={t} (one launch replaces the \
                 six-projection group; MEMRA_KDA_FUSED_PROJ=1)"
            );
        }
        Ok(Some(outs.into_iter().collect()))
    }

    /// The raw fused-6 launch (`qmatvec_kda6_q8f32_mmvq`): three Q8_0 weights + three f32
    /// weights, one q8_1 activation pair + the raw f32 activation, six outputs, t token rows.
    /// Geometry-checked but POLICY-FREE: the gate's red arms drive mutations (transposed slice
    /// data, dropped ranges via `dims[i] = 0`) through this entry, so the mutation reaches the
    /// exact program the door serves.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn kda_proj_fused6_raw(
        &self,
        wq: &CudaSlice<u8>,
        wk: &CudaSlice<u8>,
        wv: &CudaSlice<u8>,
        wfa: &CudaSlice<f32>,
        wga: &CudaSlice<f32>,
        wb: &CudaSlice<f32>,
        aq: &CudaSlice<i8>,
        ad: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        outs: &mut [CudaSlice<f32>; 6],
        in_f: usize,
        dims: [usize; 6],
        t: usize,
        row_bytes: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        const ROWS_PER_BLOCK: usize = 4; // MEMRA_MMVQ_ROWS in qmatvec.cu
        if t == 0
            || !in_f.is_multiple_of(128)
            || x.len() < t * in_f
            || aq.len() < t * in_f
            || ad.len() < t * (in_f / 32)
        {
            return Err("kda_proj_fused6 geometry".into());
        }
        for (i, (w, want_rows)) in [(wq, dims[0]), (wk, dims[1]), (wv, dims[2])]
            .into_iter()
            .enumerate()
        {
            if w.len() < want_rows * row_bytes {
                return Err(format!(
                    "kda_proj_fused6: q8 weight {i} holds {} bytes, needs {}",
                    w.len(),
                    want_rows * row_bytes
                )
                .into());
            }
        }
        for (i, (w, want_rows)) in [(wfa, dims[3]), (wga, dims[4]), (wb, dims[5])]
            .into_iter()
            .enumerate()
        {
            if w.len() < want_rows * in_f {
                return Err(format!(
                    "kda_proj_fused6: f32 weight {} holds {} floats, needs {}",
                    i + 3,
                    w.len(),
                    want_rows * in_f
                )
                .into());
            }
        }
        for (i, (o, want)) in outs.iter().zip(dims).enumerate() {
            if o.len() < t * want {
                return Err(format!("kda_proj_fused6: output {i} too small").into());
            }
        }
        let blocks: usize = dims.iter().map(|d| d.div_ceil(ROWS_PER_BLOCK)).sum();
        let f = self.func("qmatvec_kda6_q8f32_mmvq");
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, t as u32, 1),
            block_dim: (32, ROWS_PER_BLOCK as u32, 1),
            shared_mem_bytes: 0,
        };
        let inf = in_f as i32;
        let d = dims.map(|v| v as i32);
        let (mi, rb) = (t as i32, row_bytes as i64);
        let [o0, o1, o2, o3, o4, o5] = outs;
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(wq)
            .arg(wk)
            .arg(wv)
            .arg(wfa)
            .arg(wga)
            .arg(wb)
            .arg(aq)
            .arg(ad)
            .arg(x)
            .arg(&mut *o0)
            .arg(&mut *o1)
            .arg(&mut *o2)
            .arg(&mut *o3)
            .arg(&mut *o4)
            .arg(&mut *o5)
            .arg(&inf)
            .arg(&d[0])
            .arg(&d[1])
            .arg(&d[2])
            .arg(&d[3])
            .arg(&d[4])
            .arg(&d[5])
            .arg(&mi)
            .arg(&rb);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// The raw BF16-arm fused-6 launch (`qmatvec_kda6_bf16f32`): three bf16-resident weights
    /// (raw checkpoint u16 bytes, the `admit=bf16_mmv` residency) + three f32 weights, one raw
    /// f32 activation, six outputs, t token rows. Block = `mmv_block()` — the SAME blockDim
    /// `matvec_bf16_rows_into` pins, because the bf16 body's shared-tree reduction shape (and
    /// therefore its bits) is a function of blockDim. Geometry-checked but POLICY-FREE: the
    /// gate's red arms drive mutations through this entry, exactly like the q8 raw above.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn kda_proj_fused6_bf16_raw(
        &self,
        wq: &CudaSlice<u8>,
        wk: &CudaSlice<u8>,
        wv: &CudaSlice<u8>,
        wfa: &CudaSlice<f32>,
        wga: &CudaSlice<f32>,
        wb: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        outs: &mut [CudaSlice<f32>; 6],
        in_f: usize,
        dims: [usize; 6],
        t: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.kda_proj_fused6_bf16_arm_raw(
            wq,
            wk,
            wv,
            wfa,
            wga,
            wb,
            x,
            outs,
            in_f,
            dims,
            t,
            crate::b200_gemv_v2_level(),
        )
    }

    /// The same launch with the arm chosen EXPLICITLY instead of from the memoized
    /// `MEMRA_B200_GEMV_V2` door, so a bench or gate can drive every arm inside one process
    /// (`b200_matvec_bench`, the `_arm_raw` precedent).
    ///
    /// `arm`: `0` = the shipped `qmatvec_kda6_bf16f32`; `1` = `_v2`, whose three BF16 ranges take
    /// the eight-rows-per-block walk (activation loaded once and reused across the rows, ten
    /// 16 B loads in flight before the first fma, one barrier chain per block) instead of
    /// `kda6_bf16_rows4`'s four sequential rows; `2` = `_v3`, the same walk with its weight
    /// tiles staged through shared memory by `cp.async` so the in-flight budget stops being
    /// register-bound. `2` falls back to `1` when v3's dynamic smem would exceed the 48 KB
    /// default cap. Per row the arithmetic is unchanged in every arm, so all three are
    /// BIT-IDENTICAL to each other and to `matvec_bf16_f32acc_x4_rows`.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn kda_proj_fused6_bf16_arm_raw(
        &self,
        wq: &CudaSlice<u8>,
        wk: &CudaSlice<u8>,
        wv: &CudaSlice<u8>,
        wfa: &CudaSlice<f32>,
        wga: &CudaSlice<f32>,
        wb: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        outs: &mut [CudaSlice<f32>; 6],
        in_f: usize,
        dims: [usize; 6],
        t: usize,
        arm: u8,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if t == 0 || !in_f.is_multiple_of(128) || x.len() < t * in_f {
            return Err("kda_proj_fused6_bf16 geometry".into());
        }
        for (i, (w, want_rows)) in [(wq, dims[0]), (wk, dims[1]), (wv, dims[2])]
            .into_iter()
            .enumerate()
        {
            if w.len() < want_rows * in_f * 2 {
                return Err(format!(
                    "kda_proj_fused6_bf16: bf16 weight {i} holds {} bytes, needs {}",
                    w.len(),
                    want_rows * in_f * 2
                )
                .into());
            }
        }
        for (i, (w, want_rows)) in [(wfa, dims[3]), (wga, dims[4]), (wb, dims[5])]
            .into_iter()
            .enumerate()
        {
            if w.len() < want_rows * in_f {
                return Err(format!(
                    "kda_proj_fused6_bf16: f32 weight {} holds {} floats, needs {}",
                    i + 3,
                    w.len(),
                    want_rows * in_f
                )
                .into());
            }
        }
        for (i, (o, want)) in outs.iter().zip(dims).enumerate() {
            if o.len() < t * want {
                return Err(format!("kda_proj_fused6_bf16: output {i} too small").into());
            }
        }
        // v3 declines to v2 when its staged tiles would not fit the 48 KB default dynamic
        // shared-memory cap (36 KB at the default mmv_block()=128, 72 KB at 256).
        let arm = if arm >= 2 && !crate::gemv_v3_fits() {
            1
        } else {
            arm
        };
        // Rows per block, and therefore the block partition of the six ranges: 4 for the
        // shipped kernel, `GEMV_V2_ROWS` for the v2/v3 twins. v2 takes the R-row reduction
        // window as DYNAMIC shared memory (R * blockDim.x floats); v3 takes that plus its
        // cp.async stage buffers.
        let nb = crate::mmv_block();
        let rpb = if arm >= 1 { crate::GEMV_V2_ROWS } else { 4 };
        let blocks: usize = dims.iter().map(|d| d.div_ceil(rpb)).sum();
        let f = self.func(match arm {
            0 => "qmatvec_kda6_bf16f32",
            1 => "qmatvec_kda6_bf16f32_v2",
            _ => "qmatvec_kda6_bf16f32_v3",
        });
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, t as u32, 1),
            block_dim: (nb, 1, 1),
            shared_mem_bytes: match arm {
                0 => 0,
                1 => (crate::GEMV_V2_ROWS as u32) * nb * 4,
                _ => crate::gemv_v3_smem_bytes(nb as usize) as u32,
            },
        };
        let inf = in_f as i32;
        let d = dims.map(|v| v as i32);
        let mi = t as i32;
        let [o0, o1, o2, o3, o4, o5] = outs;
        let stream = self.gpu.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(wq)
            .arg(wk)
            .arg(wv)
            .arg(wfa)
            .arg(wga)
            .arg(wb)
            .arg(x)
            .arg(&mut *o0)
            .arg(&mut *o1)
            .arg(&mut *o2)
            .arg(&mut *o3)
            .arg(&mut *o4)
            .arg(&mut *o5)
            .arg(&inf)
            .arg(&d[0])
            .arg(&d[1])
            .arg(&d[2])
            .arg(&d[3])
            .arg(&d[4])
            .arg(&d[5])
            .arg(&mi);
        unsafe { b.launch(cfg)? };
        Ok(())
    }
}

#[cfg(test)]
mod kda_conv3_default_tests {
    use super::kda_conv3_on_from;

    #[test]
    fn arch_keyed_default_with_explicit_override() {
        assert!(kda_conv3_on_from(None, "100a"));
        assert!(!kda_conv3_on_from(None, "120a"));
        assert!(kda_conv3_on_from(Some("1"), "120a"));
        assert!(!kda_conv3_on_from(Some("0"), "100a"));
    }
}
