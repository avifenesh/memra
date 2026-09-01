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
//! PREFILL DISPATCH — the SEQUENTIAL SCAN by default, with the chunked per-channel-Gcum twin
//! behind `MEMRA_KDA_CHUNKED` (DEFAULT OFF — no throughput receipt yet; the flip condition is
//! the box A/B named in docs/FLAGS.md). `memra_kda_scan_s128` runs decode, the spec verify, and
//! flag-off prefill alike; the chunked form (cu/kda.cu `memra_kda_chunk_*`, derivation in its
//! header) engages ONLY from the Prefill conv arm at `t >= kda_chunk_min_t()`, so decode (t=1)
//! and the verify keep the decode==verify dispatch identity that cu/hybrid.cu's headers require,
//! byte-untouched. The chunked twin is NOT a transcription of the GDN K1-K5 chain: KDA's decay
//! is per channel, so the chunk algebra needs a per-channel cumulative log gate `Gcum[t][i]`
//! with `k` scaled by `exp(-Gcum)` and `q` by `exp(+Gcum)` (banked `chunk_kimi_delta_attention`
//! in research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py), where GDN gets away with
//! one scalar `G` per (token, head). Gate: `crates/memra-engine/tests/kda_chunked_gpu.rs`
//! (reference band + boundary crossings + red arms + decode byte-identity).
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
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
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
    // Dispatch is arm-keyed: the Decode arm calls the sequential scan DIRECTLY (decode==verify
    // dispatch identity, byte-untouched by the chunked seam); the Prefill arm goes through
    // `kda_scan_prefill`, which is the sequential scan unless MEMRA_KDA_CHUNKED engages.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut core = e.uninit(t * qkv)?;
    match arm {
        ConvArm::Decode => e.kda_scan(
            &q_l2, &k_l2, &v_conv, &g_log, &beta, state_in, state_out, &mut core, heads, t, scale,
        )?,
        ConvArm::Prefill => e.kda_scan_prefill(
            &q_l2, &k_l2, &v_conv, &g_log, &beta, state_in, state_out, &mut core, heads, t, scale,
        )?,
    }

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
    e.matmul(&la.wo, &gated, t)
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

    /// The MEMRA_KDA_CHUNKED seam. Read PER CALL, not latched: the fixture gate toggles both
    /// arms inside one process (the MEMRA_GDN_MMA / kernel-check precedent), and a prefill
    /// call at chunked widths amortizes the getenv over five kernel launches. DEFAULT OFF —
    /// the flag ships with correctness receipts only (rig is exactness-only by law); the flip
    /// condition is the interleaved x5 box A/B in the FLAGS.md row.
    pub fn kda_chunked_enabled() -> bool {
        std::env::var("MEMRA_KDA_CHUNKED").as_deref() == Ok("1")
    }

    /// Chunk size (MEMRA_KDA_CHUNK, default 64). Clamped to multiples of 32 in [32, 128]
    /// (the kernel row mappings require it). 64 balances the transient economics: A/P grow
    /// with C ([NC,H,C,C] = T*H*C floats each) while the Ssnap snapshot shrinks with C
    /// ([NC,H,D,D] = (T/C)*H*16384 floats); the box sweep owns the final number, as GDN's
    /// C=32 sweep did for its scalar-gate economics.
    pub fn kda_chunk_size() -> usize {
        std::env::var("MEMRA_KDA_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64usize)
            .clamp(32, 128)
            / 32
            * 32
    }

    /// Minimum prefill width for the chunked dispatch (MEMRA_KDA_CHUNK_MIN_T, default 256).
    /// DERIVED, not guessed (ARITHMETIC, box receipts cited): the chunked chain replaces one
    /// launch with five, costing ~4 x 6-8 us of launch+gap (the serving box family's measured
    /// launch class, research/glm53-flash-bringup-20260827/decode-attribution-receipts/
    /// ATTRIBUTION.txt and prefill-gap-20260829/PREFILL-GAP.md 1.6), while the sequential
    /// scan's per-token serial dependent chain (expf + two warp reductions + the state FMAs,
    /// ~200-400 cycles) costs ~0.1-0.2 us/token of wall — break-even near t ~ 150-300 — and
    /// the chunk algebra itself only wins once t spans several chunks (serial depth C + T/C
    /// against T). 256 = 4 chunks at the default C=64, above both bounds; spec-verify widths
    /// (K+1) and small primes stay sequential by construction. The box A/B knee sweep owns
    /// the final number; the env override exists for that sweep and for the gates.
    pub fn kda_chunk_min_t() -> usize {
        std::env::var("MEMRA_KDA_CHUNK_MIN_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    }

    /// PREFILL KDA scan dispatch: the chunked per-channel-Gcum form when enabled and `t` is
    /// in the batched-prefill regime, else the sequential scan. Prefill conv-arm callers only
    /// (`kda_core` routes the Decode arm to `kda_scan` directly).
    ///
    /// MEMRA_KDA_DIFF=1: numerical-oracle mode — runs BOTH forms on the same inputs, prints
    /// the per-call output/state error distribution, and keeps the SEQUENTIAL results so the
    /// run stays on the shipped path (the gdn_scan_diff stage-1 pattern; this is the band
    /// calibration instrument for the box A/B).
    #[allow(clippy::too_many_arguments)] // allow: mirrors the kda_scan kernel/FFI contract
    pub fn kda_scan_prefill(
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
        let engaged = Self::kda_chunked_enabled() && t >= Self::kda_chunk_min_t();
        // ENGAGEMENT RECEIPT (the moe-grouped-prefill pattern): the announce prints in BOTH
        // arms, once per process per flag value, so an A/B grep proves WHICH arm a boot ran
        // (health-200 proves a listener, not which server); engagement itself is the per-call
        // execute line below. Prove the path RAN before attributing a number to it.
        static KDA_ANNOUNCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        let bit = 1u8 << u8::from(Self::kda_chunked_enabled());
        if KDA_ANNOUNCED.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
            eprintln!(
                "[kda-chunked] flag={} t={t} min_t={} c={} (announce printed in both arms; \
                 engagement is the per-call execute line)",
                if Self::kda_chunked_enabled() {
                    "on"
                } else {
                    "off"
                },
                Self::kda_chunk_min_t(),
                Self::kda_chunk_size(),
            );
        }
        if std::env::var("MEMRA_KDA_DIFF").as_deref() == Ok("1") && t >= Self::kda_chunk_min_t() {
            return self.kda_scan_diff(q, k, v, g, beta, state_in, state_out, o, heads, t, scale);
        }
        if engaged {
            eprintln!(
                "[kda-chunked] execute t={t} nc={} c={}",
                t.div_ceil(Self::kda_chunk_size()),
                Self::kda_chunk_size(),
            );
            self.kda_scan_chunked(
                q,
                k,
                v,
                g,
                beta,
                state_in,
                state_out,
                o,
                heads,
                t,
                scale,
                Self::kda_chunk_size(),
            )
        } else {
            self.kda_scan(q, k, v, g, beta, state_in, state_out, o, heads, t, scale)
        }
    }

    /// Oracle arm of `kda_scan_prefill` (MEMRA_KDA_DIFF): both forms, error stats, sequential
    /// results kept.
    #[allow(clippy::too_many_arguments)] // allow: mirrors the kda_scan kernel/FFI contract
    fn kda_scan_diff(
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
        static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut o_c = self.uninit(o.len())?;
        let mut st_c = self.uninit(state_out.len())?;
        self.kda_scan_chunked(
            q,
            k,
            v,
            g,
            beta,
            state_in,
            &mut st_c,
            &mut o_c,
            heads,
            t,
            scale,
            Self::kda_chunk_size(),
        )?;
        self.kda_scan(q, k, v, g, beta, state_in, state_out, o, heads, t, scale)?;
        let (oh_s, oh_c) = (self.dtoh(o)?, self.dtoh(&o_c)?);
        let (sh_s, sh_c) = (self.dtoh(state_out)?, self.dtoh(&st_c)?);
        let stats = |a: &[f32], b: &[f32]| -> (f32, f32, f64) {
            let mut max_abs = 0f32;
            let mut max_rel = 0f32;
            let mut sum_rel = 0f64;
            for (x, y) in a.iter().zip(b) {
                let ad = (x - y).abs();
                let rel = ad / x.abs().max(y.abs()).max(1e-3);
                max_abs = max_abs.max(ad);
                max_rel = max_rel.max(rel);
                sum_rel += rel as f64;
            }
            (max_abs, max_rel, sum_rel / a.len() as f64)
        };
        let (o_ma, o_mr, o_mean) = stats(&oh_s, &oh_c);
        let (s_ma, s_mr, s_mean) = stats(&sh_s, &sh_c);
        println!(
            "[kda-diff call {call:3} T={t} C={}] out: max_abs={o_ma:.3e} max_rel={o_mr:.3e} mean_rel={o_mean:.3e} | \
             state: max_abs={s_ma:.3e} max_rel={s_mr:.3e} mean_rel={s_mean:.3e}",
            Self::kda_chunk_size()
        );
        Ok(())
    }

    /// The chunked per-channel-Gcum KDA prefill scan (cu/kda.cu `memra_kda_chunk_*`, math in
    /// that header). Same contract as `kda_scan` (layouts, distinct state buffers) but
    /// chunk-parallel: NOT bit-identical to the sequential scan (chunked FP accumulation
    /// order — the GDN A4 precedent); the reference band in tests/kda_chunked_gpu.rs is the
    /// accuracy authority. Prefill callers only.
    #[allow(clippy::too_many_arguments)] // allow: mirrors the kda_scan kernel/FFI contract
    pub fn kda_scan_chunked(
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
        c: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        const D: usize = KDA_HEAD_DIM;
        const NSPLIT: u32 = 4;
        if !(32..=128).contains(&c) || !c.is_multiple_of(32) {
            return Err(format!(
                "kda_scan_chunked: chunk size {c} outside the multiples of 32 in [32,128] the \
                 kernel row mappings require"
            )
            .into());
        }
        let h = heads;
        let qkv = h * D;
        let nc = t.div_ceil(c);
        let (hi, ti, ci) = (h as i32, t as i32, c as i32);
        let stream = self.gpu.stream();

        let mut gcum = self.uninit(t * qkv)?;
        {
            // K1: per-chunk per-channel cumulative log gate
            let f = self.func("memra_kda_chunk_cumgate_f32");
            let cfg = LaunchConfig {
                grid_dim: (nc as u32, h as u32, 1),
                block_dim: (D as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&f);
            b.arg(g).arg(&mut gcum).arg(&hi).arg(&ti).arg(&ci);
            unsafe { b.launch(cfg)? };
        }
        let mut a = self.uninit(nc * h * c * c)?;
        let mut p = self.uninit(nc * h * c * c)?;
        {
            // K2: pair matrices A (positive strictly-lower) and P (inclusive, upper zeroed)
            let f = self.func("memra_kda_chunk_attn_f32");
            let cfg = LaunchConfig {
                grid_dim: (nc as u32, h as u32, 1),
                block_dim: (32, 8, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&f);
            b.arg(q)
                .arg(k)
                .arg(&gcum)
                .arg(beta)
                .arg(&mut a)
                .arg(&mut p)
                .arg(&hi)
                .arg(&ti)
                .arg(&ci);
            unsafe { b.launch(cfg)? };
        }
        let mut u = self.uninit(nc * h * c * D)?;
        let mut w = self.uninit(nc * h * c * D)?;
        {
            // K3: both forward substitutions (U from v.beta, W from k.beta.exp(Gcum))
            let (name, pass_c) = match c {
                32 => ("memra_kda_chunk_solve32_f32", false),
                64 => ("memra_kda_chunk_solve64_f32", false),
                _ => ("memra_kda_chunk_solve_f32", true),
            };
            let f = self.func(name);
            let cfg = LaunchConfig {
                grid_dim: (nc as u32, h as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&f);
            b.arg(v)
                .arg(k)
                .arg(&a)
                .arg(&gcum)
                .arg(beta)
                .arg(&mut u)
                .arg(&mut w)
                .arg(&hi)
                .arg(&ti);
            if pass_c {
                b.arg(&ci);
            }
            unsafe { b.launch(cfg)? };
        }
        let mut y = self.uninit(nc * h * c * D)?;
        let mut ssnap = self.uninit(nc * h * D * D)?;
        {
            // K4: sequential inter-chunk state pass (blocks col-partition the state)
            let f = self.func("memra_kda_chunk_state_f32");
            let cfg = LaunchConfig {
                grid_dim: (h as u32, NSPLIT, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&f);
            b.arg(k)
                .arg(&gcum)
                .arg(&u)
                .arg(&w)
                .arg(&mut y)
                .arg(&mut ssnap)
                .arg(state_in)
                .arg(&mut *state_out)
                .arg(&hi)
                .arg(&ti)
                .arg(&ci);
            unsafe { b.launch(cfg)? };
        }
        {
            // K5: output assembly (chunk-parallel; writes o fully)
            let jt = c.div_ceil(32) as u32;
            let f = self.func("memra_kda_chunk_output_f32");
            let cfg = LaunchConfig {
                grid_dim: (nc as u32, h as u32, jt),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&f);
            b.arg(q)
                .arg(&gcum)
                .arg(&p)
                .arg(&y)
                .arg(&ssnap)
                .arg(&mut *o)
                .arg(&hi)
                .arg(&ti)
                .arg(&ci)
                .arg(&scale);
            unsafe { b.launch(cfg)? };
        }
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
