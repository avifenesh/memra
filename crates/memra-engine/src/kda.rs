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
//! PREFILL DISPATCH — SEQUENTIAL SCAN, not the chunked UT transform (deliberate, this
//! increment). `memra_kda_scan_s128` runs prefill and decode alike, which is exactly the shipped
//! GDN arrangement next door: `gdn_scan_s128` IS the default prefill path and the chunked WY
//! kernels sit behind `MEMRA_GDN_CHUNKED`. One kernel for both also keeps the decode==verify
//! dispatch identity that cu/hybrid.cu's headers require. The chunked twin is a tuning-phase
//! follow-up and is NOT a transcription of the GDN K1-K5 chain: KDA's decay is per channel, so
//! the chunk algebra needs a per-channel cumulative log gate `Gcum[t][i]` with `k` scaled by
//! `exp(-Gcum)` and `q` by `exp(+Gcum)` (banked `chunk_kimi_delta_attention` in
//! research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py), where GDN gets away with one
//! scalar `G` per (token, head).
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
        })
    }
}

/// Which conv arm a call takes. `Prefill` reads the ring as a left pad and rolls it afterwards;
/// `Decode` fuses assemble+conv+roll for the single new row. The two produce bit-identical
/// values at T=1 (same ascending tap order over the same window) — the split exists so decode
/// and the spec verify keep one dispatch class, per the cu/hybrid.cu decode==verify law.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvArm {
    Prefill,
    Decode,
}

/// The whole mixer, stage for stage against `memra_reference::kimi_delta_net`.
///
/// `ring` is the fused `[3*qkv, kernel-1]` conv state (zeroed = fresh prefill's zero left pad)
/// and is updated in place. `state_in`/`state_out` are the `[heads, 128, 128]` recurrent state
/// in the kernel's transposed `M[col][i]` layout; they MUST be distinct buffers.
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
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let heads = la.heads();
    let head_dim = la.head_dim();
    let qkv = la.qkv();
    let kernel = la.conv_kernel();
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
    let mut g6 = e.matmul_group(
        &[&la.wq, &la.wk, &la.wv, &la.f_a, &la.g_a, &la.b_proj],
        x,
        t,
    )?;
    let beta_raw = g6.pop().unwrap(); // [T, heads]
    let gate_down = g6.pop().unwrap(); // [T, head_dim]
    let forget_down = g6.pop().unwrap(); // [T, head_dim]
    let v_raw = g6.pop().unwrap(); // [T, qkv]
    let k_raw = g6.pop().unwrap();
    let q_raw = g6.pop().unwrap();

    // Stage 2 — per-plane causal short conv + SiLU. Planes are ordered q, k, v in both the fused
    // weight buffer and the fused ring, which is the order the reference stores conv_state in.
    let mut q_conv = e.uninit(t * qkv)?;
    let mut k_conv = e.uninit(t * qkv)?;
    let mut v_conv = e.uninit(t * qkv)?;
    for (plane, (raw, out)) in [
        (&q_raw, &mut q_conv),
        (&k_raw, &mut k_conv),
        (&v_raw, &mut v_conv),
    ]
    .into_iter()
    .enumerate()
    {
        match arm {
            ConvArm::Prefill => e.kda_conv_silu(raw, &la.conv, ring, out, qkv, t, kernel, plane)?,
            ConvArm::Decode => {
                e.kda_conv_silu_decode(raw, ring, &la.conv, out, qkv, kernel, plane)?
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
    let mut q_l2 = e.uninit(t * qkv)?;
    let mut k_l2 = e.uninit(t * qkv)?;
    e.l2_norm(&q_conv, &mut q_l2, head_dim, t * heads, KDA_L2_EPS)?;
    e.l2_norm(&k_conv, &mut k_l2, head_dim, t * heads, KDA_L2_EPS)?;

    // Stage 4 — gates. forget: g = lower_bound * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias)),
    // emitted RAW (the scan applies expf). beta: per-head sigmoid of its own projection.
    let forget = e.matmul(&la.f_b, &forget_down, t)?;
    let mut g_log = e.uninit(t * qkv)?;
    e.kda_gate(
        &forget,
        la.dt_bias.float_data(),
        la.a_log.float_data(),
        &mut g_log,
        qkv,
        t,
        head_dim,
        la.plan.gate_lower_bound,
    )?;
    let mut beta = e.uninit(t * heads)?;
    e.sigmoid(&beta_raw, &mut beta, t * heads)?;

    // Stage 5 — the delta-rule recurrence. `scale` carries the reference's head_dim^-0.5 query
    // scale: q feeds only the readout, never the state, so scaling the readout is exact.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut core = e.uninit(t * qkv)?;
    e.kda_scan(
        &q_l2, &k_l2, &v_conv, &g_log, &beta, state_in, state_out, &mut core, heads, t, scale,
    )?;

    // Stage 6 — sigmoid-gated RMSNorm over head_dim (layer rms eps here, NOT the l2 eps), then
    // the output projection.
    let gate = e.matmul(&la.g_b, &gate_down, t)?;
    let mut gated = e.uninit(t * qkv)?;
    e.kda_gated_rmsnorm(
        &core,
        la.o_norm.float_data(),
        &gate,
        &mut gated,
        head_dim,
        t * heads,
        eps,
    )?;
    Ok(e.matmul(&la.wo, &gated, t)?)
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
    )
}

/// STATEFUL prefill: carries the ring forward and advances the recurrent state from `state_in`
/// into `state_out`. Callers own the ping-pong; the two state buffers must be distinct.
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
    kda_core(e, la, x, 1, eps, ring, state_in, state_out, ConvArm::Decode)
}

/// Stateful KDA against the shared recurrent-state carrier, in the eager GDN discipline: the
/// scan reads `ssm_state` and writes the spare `ssm_state_alt`, then the two OWNED resident
/// buffers swap in place. Stable pointers, no per-step alloc/free — the per-step scratch this
/// replaced churned the stream-ordered pool and made decode run-to-run nondeterministic
/// (crates/memra-kv `RecurLayer::ssm_state_alt`). NOT capture-safe: a captured graph bakes
/// capture-time pointers and never re-runs the host swap, which is why the capture loops refuse.
fn kda_cached(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    cache: &mut Cache,
    il: usize,
    arm: ConvArm,
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
        kda_core(e, la, x, t, eps, conv_state, ssm_state, ssm_state_alt, arm)?
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
    kda_cached(e, la, x, t, eps, cache, il, ConvArm::Prefill)
}

/// One decode step through the cache's KDA state for layer `il`.
pub fn kda_decode_cached(
    e: &Engine,
    la: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    eps: f32,
    cache: &mut Cache,
    il: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    kda_cached(e, la, x, 1, eps, cache, il, ConvArm::Decode)
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
}
