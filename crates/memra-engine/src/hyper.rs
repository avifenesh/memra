//! mHC — manifold-constrained hyper-connections, the `ResidualTopology::HyperConnections`
//! residual program (glm5_next / GLM-5.3-Flash, and the dsv4 class).
//!
//! ARITHMETIC CONTRACT. Truth is `memra_reference::execute`'s `execute_hyper_layer`, which is
//! itself built from `memra_gguf::dsv4_forward::{hc_expand, hc_pre, hc_post, hc_split_sinkhorn,
//! hc_head}`. Every stage below cites the reference stage it reproduces. The vendor module the
//! reference was derived from is
//! `research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py`.
//!
//! A trunk layer under this topology is NOT `x + attn; x += mlp`. Per site (attention, then
//! MLP), with the stream state `x [tokens, streams, hidden]`:
//!
//! ```text
//!   mixes[t, :]   = fn_w · x[t, :, :]                     (rows = (2+streams)*streams)
//!   mixes[t, :]  *= rsqrt(mean(x[t]^2) + eps)             (over the whole streams*hidden slab)
//!   pre/post/comb = sinkhorn(mixes[t, :], scale, base)    (per token, per site)
//!   y[t, :]       = Σ_c pre[t, c] · x[t, c, :]            (collapse streams -> 1)
//!   f             = branch(rms_norm(y))                   (the mixer or the FFN, unchanged)
//!   x'[t, k, :]   = post[t, k] · f[t, :] + Σ_j comb[t, j, k] · x[t, j, :]
//! ```
//!
//! SINKHORN IS PER TOKEN AND PER SITE, NOT A LOAD-TIME PRECOMPUTE. `mixes` is
//! `x @ fn_wᵀ` rescaled by the token's own RMS — an ACTIVATION, so the Sinkhorn normalization
//! that turns it into `comb` cannot be hoisted to load even though the weights are static
//! (`dsv4_forward.rs` `hc_pre`, the `matmul` + `rsq` block immediately before
//! `hc_split_sinkhorn`). It runs on device, once per (token, layer, site).
//!
//! MEMORY LAYOUT: TOKEN-MAJOR `[tokens, streams, hidden]`, element `(t, k, i)` at
//! `(t*streams + k)*hidden + i`. Forced, not chosen: it is the layout of `hc_expand` in the
//! reference and of every kernel in the `memra_dsv4_hc_*` family, and it makes one token's
//! `streams*hidden` slab contiguous — which is exactly the `[s, w]` operand the mixes GEMM and
//! `memra_dsv4_rowsq_scale` want. Streams-major would have cost a transpose at both ends of
//! every site. Any graph capture over these buffers sees one flat `t*streams*hidden` slab.
//!
//! KERNELS: no new math. `cu/dsv4_gpu.cu` already carries this exact program for the dsv4 GPU
//! fork (`crate::dsv4_gpu`) and is compiled unconditionally into this crate, so the site mixing
//! is `memra_dsv4_{rowsq_scale, hc_sinkhorn_m, hc_collapse, hc_post}` plus `hc_mean`/`hc_head_pre_m`
//! at the exit, and the mixes GEMM is `Engine::linear` (cuBLASLt f32 — the tiny
//! `[rows, streams*hidden]` operand is the wrong shape for the f64 island `dots` kernel the dsv4
//! decode path uses, and this is a serving trunk, not a byte-parity oracle). The one kernel that
//! did not exist, `memra_dsv4_hc_expand`, was added next to its inverse `memra_dsv4_hc_mean`.
//! The `dsv4_` prefix is that translation unit's namespace, not a model claim — the reference
//! reaches into `memra_gguf::dsv4_forward` for glm5_next in exactly the same way.
//!
//! NO ENV FLAG. The topology, its stream count, its epsilon, its Sinkhorn iteration count and
//! its collapse are read from the compiled `ModelPlan`. There is nothing here to switch.

use crate::Engine;
use crate::dsv4_ffi as k;
use crate::dsv4_ffi::ck;
use crate::model::GpuTensor;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use memra_gguf::model_plan::{HcCollapse, ModelPlan, ResidualTopology};
use memra_gguf::source::TensorSource;
use std::os::raw::c_void;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn sp(stream: &CudaStream) -> *mut c_void {
    stream.cu_stream() as *mut c_void
}

macro_rules! dpf {
    ($slice:expr, $stream:expr) => {{ $slice.device_ptr($stream).0 as *const f32 }};
}
macro_rules! dpm {
    ($slice:expr, $stream:expr) => {{ $slice.device_ptr_mut($stream).0 as *mut f32 }};
}

/// The trunk-wide hyper-connection topology, read off the plan at load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperTopology {
    pub streams: usize,
    pub epsilon: f32,
    pub sinkhorn_iterations: u32,
    pub collapse: HcCollapse,
}

impl HyperTopology {
    /// `(2 + streams) * streams` — pre gates, post gates, then the `streams x streams`
    /// combination block, in that row order (`hc_split_sinkhorn`).
    pub fn rows(&self) -> usize {
        (2 + self.streams) * self.streams
    }

    /// The plan's topology, or `None` for a serial/gemma trunk. Refuses a trunk whose layers
    /// disagree: the state carried between layers is one shape, so a per-layer stream count is
    /// not a thing this executor can mean. Mirrors `memra_reference`'s `hyper_topology`.
    pub fn from_plan(plan: &ModelPlan) -> Result<Option<Self>, String> {
        let mut found: Option<Self> = None;
        for layer in &plan.layers {
            let ResidualTopology::HyperConnections {
                streams,
                epsilon,
                sinkhorn_iterations,
                collapse,
            } = layer.residual
            else {
                if found.is_some() {
                    return Err(format!(
                        "layer {} declares a serial/gemma residual while an earlier trunk layer \
                         declares HyperConnections; the topology must be uniform across the trunk",
                        layer.index
                    ));
                }
                continue;
            };
            let this = Self {
                streams: streams as usize,
                epsilon,
                sinkhorn_iterations,
                collapse,
            };
            if streams == 0 || epsilon <= 0.0 || sinkhorn_iterations == 0 {
                return Err(format!(
                    "layer {}: HyperConnections need streams > 0, epsilon > 0 and \
                     sinkhorn_iterations > 0, got streams={streams} epsilon={epsilon} \
                     iterations={sinkhorn_iterations}",
                    layer.index
                ));
            }
            match found {
                None if layer.index != plan.layers[0].index => {
                    return Err(format!(
                        "layer {} declares HyperConnections but earlier trunk layers do not; the \
                         topology must be uniform across the trunk",
                        layer.index
                    ));
                }
                None => found = Some(this),
                Some(first) if first != this => {
                    return Err(format!(
                        "layer {} declares {this:?} but the trunk opened with {first:?}; the \
                         topology must be uniform across the trunk",
                        layer.index
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(found)
    }
}

/// One site's learned mixing parameters. `fn_w` is consumed as ROW-MAJOR `[rows,
/// streams*hidden]` — the layout `memra_reference::hyper_set` and `dsv4_forward::HcSet` read, and
/// the `[out_f, in_f]` operand `Engine::linear` wants. Only the element count is checked at load;
/// the checkpoint dialect's `ne` ordering is not consulted, so the two readers cannot fork.
pub struct HyperSite {
    pub fn_w: CudaSlice<f32>,
    pub base: CudaSlice<f32>,
    pub scale: CudaSlice<f32>,
}

/// The six per-layer hc tensors, present iff the plan declares HyperConnections for the trunk.
pub struct HyperLayer {
    pub attn: HyperSite,
    pub mlp: HyperSite,
}

/// Gated-head exit weights (`HcCollapse::GatedHead`, the dsv4 class). Absent under
/// `HcCollapse::Mean`, which has no learned head (`Glm5NextTextHyperHead` is an unweighted mean).
pub struct HyperHead {
    pub fn_w: CudaSlice<f32>,
    pub base: CudaSlice<f32>,
    pub scale: CudaSlice<f32>,
}

/// A loaded float tensor's device data, or a refusal naming the tensor. `GpuTensor::float_data`
/// panics on the quantized/bf16 variants; an hc parameter arriving in one of those is a
/// checkpoint the trunk cannot serve, and it must say which tensor and why.
fn float_data<'a>(name: &str, t: &'a GpuTensor, want: usize) -> Result<&'a CudaSlice<f32>, String> {
    let data = match t {
        GpuTensor::Float { data, .. } => data,
        GpuTensor::Quant { .. } => {
            return Err(format!(
                "{name}: hyper-connection parameters must be f32-resident, got a quantized \
                 tensor; re-mint this tensor unquantized (the whole hc program is an f32 island)"
            ));
        }
        GpuTensor::FloatBf16 { .. } => {
            return Err(format!(
                "{name}: hyper-connection parameters must be f32-resident, got a bf16-resident \
                 matmul weight"
            ));
        }
    };
    if data.len() != want {
        return Err(format!(
            "{name}: {} elements, the plan's HyperConnections require {want}",
            data.len()
        ));
    }
    Ok(data)
}

/// Load one site's trio, refusing loudly — by name — on the first absent tensor. There is no
/// serial fallback: a plan that declares HyperConnections and a checkpoint that does not carry
/// them describe two different functions, and guessing which one to compute is the failure this
/// refusal exists to prevent.
fn load_site(
    e: &Engine,
    src: &dyn TensorSource,
    il: u32,
    topology: &HyperTopology,
    hidden: usize,
    site: &str,
) -> Res<HyperSite> {
    let rows = topology.rows();
    let width = topology.streams * hidden;
    let mut out: Vec<CudaSlice<f32>> = Vec::with_capacity(3);
    for (suffix, want) in [
        ("fn", rows * width),
        ("base", rows),
        // Three gate scales — pre, post, combination — regardless of stream count
        // (`hc_split_sinkhorn` asserts `scale.len() == 3`).
        ("scale", 3),
    ] {
        // The ggml spellings `add_hyper_connections` (memra-gguf tensor_contract) emits.
        let name = format!("blk.{il}.{site}_{suffix}");
        if !src.has(&name) {
            return Err(format!(
                "{name} is absent, but the compiled ModelPlan declares \
                 ResidualTopology::HyperConnections{{ streams: {} }} for layer {il}. Refusing to \
                 load: a serial residual would compute a different model, silently.",
                topology.streams
            )
            .into());
        }
        let loaded = GpuTensor::load_from_source(e, src, &name)?;
        out.push(e.clone_dtod(float_data(&name, &loaded, want)?)?);
    }
    let mut out = out.into_iter();
    Ok(HyperSite {
        fn_w: out.next().expect("function"),
        base: out.next().expect("base"),
        scale: out.next().expect("scale"),
    })
}

impl HyperLayer {
    pub fn load(
        e: &Engine,
        src: &dyn TensorSource,
        il: u32,
        topology: &HyperTopology,
        hidden: usize,
    ) -> Res<Self> {
        Ok(Self {
            attn: load_site(e, src, il, topology, hidden, "hc_attn")?,
            mlp: load_site(e, src, il, topology, hidden, "hc_ffn")?,
        })
    }
}

impl HyperHead {
    /// `None` unless the collapse is gated. `hc_head`'s trio is shaped differently from a site's:
    /// `rows == streams` and one scale (`dsv4_forward::hc_head`).
    pub fn load(
        e: &Engine,
        src: &dyn TensorSource,
        topology: &HyperTopology,
        hidden: usize,
    ) -> Res<Option<Self>> {
        if topology.collapse != HcCollapse::GatedHead {
            return Ok(None);
        }
        let streams = topology.streams;
        let mut out: Vec<CudaSlice<f32>> = Vec::with_capacity(3);
        // The dsv4 checkpoint spellings (crate::dsv4_gpu's `hc_head_*` loads). The
        // TensorContract has no HyperHead rows — nothing in the GGUF/safetensors schema emits
        // them yet — so a gated-head trunk on THIS path refuses by name below until it does.
        for (name, want) in [
            ("hc_head_fn", streams * streams * hidden),
            ("hc_head_base", streams),
            ("hc_head_scale", 1),
        ] {
            if !src.has(name) {
                return Err(format!(
                    "{name} is absent, but the compiled ModelPlan declares \
                     HcCollapse::GatedHead. Refusing to load: collapsing with an unweighted mean \
                     instead would compute a different model, silently."
                )
                .into());
            }
            let loaded = GpuTensor::load_from_source(e, src, name)?;
            out.push(e.clone_dtod(float_data(name, &loaded, want)?)?);
        }
        let mut out = out.into_iter();
        Ok(Some(Self {
            fn_w: out.next().expect("function"),
            base: out.next().expect("base"),
            scale: out.next().expect("scale"),
        }))
    }
}

/// The per-token post gates and combination matrix a site's `hc_pre` produced, held for that
/// site's `hc_post`. `post` is `[tokens, streams]`, `comb` is `[tokens, streams, streams]`.
pub struct HcMix {
    pub post: CudaSlice<f32>,
    pub comb: CudaSlice<f32>,
}

/// Engagement counter for the fused pre-chain door's `=1` arm: incremented at the arm's own
/// call site, announced once per boot — the spec-engagement receipt the gate and any box A/B
/// arm must show ([bf16-mmv] RESIDENT lesson: engagement lines are receipts, never inferred).
pub static HC_FUSED_PRE_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Engagement counter for the fused pre-chain door's `=2` arm (lane/b200-sinkhorn-fusion-
/// 20260902 follow-up), same discipline as `HC_FUSED_PRE_DISPATCHES`.
pub static HC_FUSED_PRE_V2_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The three states of `MEMRA_HC_FUSED_PRE` (default OFF): the unfused three-kernel chain,
/// the `=1` fused kernel (`memra_dsv4_hc_pre_fused`, lane/glm5-decode-diet 2026-08-31), or
/// the `=2` fused kernel (`memra_dsv4_hc_pre_fused_v2`, lane/b200-sinkhorn-fusion-20260902 —
/// same stages, warp-scoped Sinkhorn sync). Any other value (unset, `0`, or unrecognized)
/// stays `Off`, the existing "read per call" rollback-seam contract.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HcFusedPreArm {
    Off,
    V1,
    V2,
}

/// `MEMRA_HC_FUSED_PRE` (default OFF, both `1` and `2` opt in): the three-kernel site
/// pre-chain (rowsq_scale + Sinkhorn + collapse) runs as ONE launch per site — bit-identical
/// to the unfused chain by construction in both arms (verbatim bodies, asserted bytewise in
/// `hc_fused_pre_gpu.rs` for `=1` and by `hc-fused-gate` for `=1` vs `=2`). Read PER CALL
/// (the `MEMRA_MOE_FUSED_EPI` rollback-seam precedent), so arms can alternate inside one
/// process and the flag is a live rollback seam.
fn hc_fused_pre_arm() -> HcFusedPreArm {
    hc_fused_pre_arm_from(
        std::env::var("MEMRA_HC_FUSED_PRE").ok().as_deref(),
        env!("MEMRA_BUILT_CUDA_ARCH"),
    )
}

/// The pure parse behind [`hc_fused_pre_arm`] (arch-keyed since 2026-09-04): `1` = V1, `2` = V2,
/// `0` = the unfused chain; UNSET follows the build arch: V2 on `100a` (the served posture on
/// the 2x B200 pair since 2026-09-02, receipts in darklanes research/glm5-b200-20260902/LANE.md
/// and the FLAGS row), the unfused chain on every other build until it has its own receipt.
pub fn hc_fused_pre_arm_from(v: Option<&str>, built_arch: &str) -> HcFusedPreArm {
    match v.map(str::trim) {
        Some("1") => HcFusedPreArm::V1,
        Some("2") => HcFusedPreArm::V2,
        Some("0") => HcFusedPreArm::Off,
        _ if built_arch == "100a" => HcFusedPreArm::V2,
        _ => HcFusedPreArm::Off,
    }
}

/// Model entry (`hc_expand`): `[tokens, hidden]` embeddings -> `[tokens, streams, hidden]`.
pub fn expand(
    e: &Engine,
    topology: &HyperTopology,
    embedded: &CudaSlice<f32>,
    t: usize,
    hidden: usize,
) -> Res<CudaSlice<f32>> {
    let streams = topology.streams;
    let mut out = e.uninit(t * streams * hidden)?;
    let stream = e.stream();
    unsafe {
        ck(
            "hc_expand",
            k::memra_dsv4_hc_expand(
                dpf!(embedded, &stream),
                dpm!(out, &stream),
                t as i32,
                streams as i32,
                hidden as i32,
                sp(&stream),
            ),
        )?;
    }
    Ok(out)
}

/// One site's pre-branch half (`hc_pre`): mixes GEMM, per-token RMS rescale, Sinkhorn, stream
/// collapse. Returns the branch input `[tokens, hidden]` and the gates its `post` half needs.
pub fn pre(
    e: &Engine,
    topology: &HyperTopology,
    site: &HyperSite,
    x: &CudaSlice<f32>,
    t: usize,
    hidden: usize,
) -> Res<(CudaSlice<f32>, HcMix)> {
    let width = topology.streams * hidden;
    let mixes = e.linear(x, &site.fn_w, t, width, topology.rows())?;
    pre_finish(e, topology, site, x, mixes, t, hidden)
}

/// `pre` with the DECODE-EXACT mixing GEMM: each token's mix coefficients come from the
/// SAME m=1 cuBLASLt program the serial T=1 decode step runs (`linear_t1_into` is `linear`
/// at m == 1 on a row view — same config, same weight pointer, same input bytes), instead
/// of one m=t call whose n-dependent reduction split changes every output bit (the lt_ndep
/// probe documented on `Engine::linear_decode_exact`). Everything after the GEMM is the
/// per-token kernel set `pre` already runs — block-per-token programs whose per-token bytes
/// do not depend on t. This is the entry the BATCHED hyper decode walk uses so that row b
/// of a B-row tick is bit-identical to that session's solo `decode_step_hyper` step.
pub fn pre_exact(
    e: &Engine,
    topology: &HyperTopology,
    site: &HyperSite,
    x: &CudaSlice<f32>,
    t: usize,
    hidden: usize,
) -> Res<(CudaSlice<f32>, HcMix)> {
    let rows = topology.rows();
    let width = topology.streams * hidden;
    let mut mixes = e.uninit(t * rows)?;
    for r in 0..t {
        let xr = x.slice(r * width..(r + 1) * width);
        let wv = site.fn_w.slice(0..site.fn_w.len());
        let mut yr = mixes.slice_mut(r * rows..(r + 1) * rows);
        hc_mixes_into(e, &xr, &wv, &mut yr, width, rows)
            .map_err(|err| format!("hc pre_exact row {r}: {err}"))?;
    }
    pre_finish(e, topology, site, x, mixes, t, hidden)
}

/// The per-token half `pre` and `pre_exact` share: RMS rescale of the mix coefficients,
/// Sinkhorn, stream collapse. Every kernel here is a block-per-token program (grid over t),
/// so per-token output bytes are invariant to t — the two entries differ ONLY in how the
/// mixes GEMM reduces.
fn pre_finish(
    e: &Engine,
    topology: &HyperTopology,
    site: &HyperSite,
    x: &CudaSlice<f32>,
    mut mixes: CudaSlice<f32>,
    t: usize,
    hidden: usize,
) -> Res<(CudaSlice<f32>, HcMix)> {
    let streams = topology.streams;
    let mut pre_gates = e.uninit(t * streams)?;
    let mut post = e.uninit(t * streams)?;
    let mut comb = e.uninit(t * streams * streams)?;
    let mut y = e.uninit(t * hidden)?;
    pre_finish_into(
        e,
        topology,
        site,
        x,
        &mut mixes,
        &mut pre_gates,
        &mut post,
        &mut comb,
        &mut y,
        t,
        hidden,
    )?;
    Ok((y, HcMix { post, comb }))
}

/// `pre_finish`'s kernel arms on caller-owned outputs — shared by the allocating entry above
/// and the persistent-workspace decode walk (`pre_t1_ws`), so the two cannot drift. Both arms
/// fully overwrite every output element, which is what makes workspace reuse byte-identical.
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI contract; the workspace caller passes disjoint field borrows
fn pre_finish_into(
    e: &Engine,
    topology: &HyperTopology,
    site: &HyperSite,
    x: &CudaSlice<f32>,
    mixes: &mut CudaSlice<f32>,
    pre_gates: &mut CudaSlice<f32>,
    post: &mut CudaSlice<f32>,
    comb: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    t: usize,
    hidden: usize,
) -> Res<()> {
    let streams = topology.streams;
    let rows = topology.rows();
    let width = streams * hidden;
    let eps = topology.epsilon;
    let stream = e.stream();

    // FUSED PRE-CHAIN DOOR (lane/glm5-decode-diet; `=2` arm lane/b200-sinkhorn-fusion-
    // 20260902). Engages at any t (block-per-token, per-token bytes t-invariant like the
    // unfused chain) whenever the stream count fits the kernel's static shared arrays;
    // every other shape falls through to the unchanged three-kernel program below. Both
    // kernels read the RAW mixes and apply the rowsq rescale internally, so the in-place
    // scale write below is subsumed (nothing reads the scaled mixes after this function
    // either way).
    let fused_arm = hc_fused_pre_arm();
    if fused_arm != HcFusedPreArm::Off && streams <= 8 {
        let (label, rc) = unsafe {
            match fused_arm {
                HcFusedPreArm::V1 => (
                    "hc_pre_fused",
                    k::memra_dsv4_hc_pre_fused(
                        dpf!(x, &stream),
                        dpf!(mixes, &stream),
                        dpf!(site.scale, &stream),
                        dpf!(site.base, &stream),
                        dpm!(pre_gates, &stream),
                        dpm!(post, &stream),
                        dpm!(comb, &stream),
                        dpm!(y, &stream),
                        t as i32,
                        streams as i32,
                        hidden as i32,
                        topology.sinkhorn_iterations as i32,
                        eps,
                        std::ptr::null_mut(),
                        sp(&stream),
                    ),
                ),
                // v3 is v2 with the width as a parameter and the register-Sinkhorn door on it.
                // It is selected when EITHER door is set: at width 128 v3 is bit-identical to v2
                // (same partition), so routing MEMRA_HC_PRE_SINK_REG=1 through v3 at the default
                // width changes only the Sinkhorn stage. Measured 2026-09-03: with the guard on
                // width alone, `MEMRA_HC_PRE_SINK_REG=1` at block 128 dispatched v2 and the door
                // was unreachable — the announce line said `kernel=hc_pre_fused_v2` and the arm
                // read 55.86 against a 55.94 baseline, a measurement of nothing.
                HcFusedPreArm::V2 if crate::hc_pre_block() != 128 || crate::hc_pre_sink_reg() => {
                    // MEMRA_HC_PRE_V4: the register schedule first; 40025 (shape does not fit)
                    // falls through to v3, any other non-zero rc is v4's error and is reported
                    // as such rather than masked by a v3 retry.
                    let v4 = if crate::hc_pre_v4_on() && !crate::hc_pre_split_collapse() {
                        let rc = k::memra_dsv4_hc_pre_v4(
                            dpf!(x, &stream),
                            dpf!(mixes, &stream),
                            dpf!(site.scale, &stream),
                            dpf!(site.base, &stream),
                            dpm!(pre_gates, &stream),
                            dpm!(post, &stream),
                            dpm!(comb, &stream),
                            dpm!(y, &stream),
                            t as i32,
                            streams as i32,
                            hidden as i32,
                            topology.sinkhorn_iterations as i32,
                            eps,
                            std::ptr::null_mut(),
                            crate::hc_pre_block() as i32,
                            sp(&stream),
                        );
                        if rc == 40025 {
                            None
                        } else {
                            if rc == 0 {
                                HC_PRE_V4_DISPATCHES
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            Some(("hc_pre_v4", rc))
                        }
                    } else {
                        None
                    };
                    match v4 {
                        Some(done) => done,
                        None => (
                            "hc_pre_fused_v3",
                            k::memra_dsv4_hc_pre_fused_v3(
                                dpf!(x, &stream),
                                dpf!(mixes, &stream),
                                dpf!(site.scale, &stream),
                                dpf!(site.base, &stream),
                                dpm!(pre_gates, &stream),
                                dpm!(post, &stream),
                                dpm!(comb, &stream),
                                dpm!(y, &stream),
                                t as i32,
                                streams as i32,
                                hidden as i32,
                                topology.sinkhorn_iterations as i32,
                                eps,
                                std::ptr::null_mut(),
                                crate::hc_pre_block() as i32,
                                crate::hc_pre_sink_reg() as i32,
                                crate::hc_pre_split_collapse() as i32,
                                sp(&stream),
                            ),
                        ),
                    }
                }
                HcFusedPreArm::V2 => (
                    "hc_pre_fused_v2",
                    k::memra_dsv4_hc_pre_fused_v2(
                        dpf!(x, &stream),
                        dpf!(mixes, &stream),
                        dpf!(site.scale, &stream),
                        dpf!(site.base, &stream),
                        dpm!(pre_gates, &stream),
                        dpm!(post, &stream),
                        dpm!(comb, &stream),
                        dpm!(y, &stream),
                        t as i32,
                        streams as i32,
                        hidden as i32,
                        topology.sinkhorn_iterations as i32,
                        eps,
                        std::ptr::null_mut(),
                        sp(&stream),
                    ),
                ),
                HcFusedPreArm::Off => unreachable!("guarded by the enclosing if"),
            }
        };
        ck(label, rc)?;
        // The announce says which KERNEL ran, not which arm was asked for: under
        // MEMRA_HC_PRE_BLOCK != 128 the V2 arm dispatches `_v3` with a wider block, and a
        // counter line reading `arm=2` while `_v3` executes is the kind of quiet mismatch a
        // later reader has to re-derive from nsys. The counter itself stays V2's (the arm is
        // still V2; the width is a property of that arm), and the width is printed.
        let block = crate::hc_pre_block();
        let (counter, tag) = match fused_arm {
            HcFusedPreArm::V1 => (&HC_FUSED_PRE_DISPATCHES, "1"),
            HcFusedPreArm::V2 => (&HC_FUSED_PRE_V2_DISPATCHES, "2"),
            HcFusedPreArm::Off => unreachable!("guarded by the enclosing if"),
        };
        if counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            let kern = if fused_arm == HcFusedPreArm::V2
                && HC_PRE_V4_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed) > 0
            {
                "hc_pre_v4"
            } else if fused_arm == HcFusedPreArm::V2 && (block != 128 || crate::hc_pre_sink_reg()) {
                "hc_pre_fused_v3"
            } else if fused_arm == HcFusedPreArm::V2 {
                "hc_pre_fused_v2"
            } else {
                "hc_pre_fused"
            };
            eprintln!(
                "[hc-fused-pre] engaged streams={streams} hidden={hidden} t={t} arm={tag} \
                 kernel={kern} block={block} sinkhorn={} (one launch replaces rowsq_scale + \
                 sinkhorn + collapse per site; MEMRA_HC_FUSED_PRE={tag}, MEMRA_HC_PRE_BLOCK={block})",
                if crate::hc_pre_sink_reg() {
                    "registers"
                } else {
                    "shared"
                }
            );
        }
        return Ok(());
    }
    unsafe {
        ck(
            "hc rowsq_scale",
            k::memra_dsv4_rowsq_scale(
                dpf!(x, &stream),
                dpm!(mixes, &stream),
                t as i32,
                width as i32,
                rows as i32,
                eps,
                sp(&stream),
            ),
        )?;
        ck(
            "hc_sinkhorn",
            k::memra_dsv4_hc_sinkhorn_m(
                dpf!(mixes, &stream),
                dpf!(site.scale, &stream),
                dpf!(site.base, &stream),
                dpm!(pre_gates, &stream),
                dpm!(post, &stream),
                dpm!(comb, &stream),
                t as i32,
                streams as i32,
                topology.sinkhorn_iterations as i32,
                eps,
                sp(&stream),
            ),
        )?;
        ck(
            "hc_collapse",
            k::memra_dsv4_hc_collapse(
                dpf!(x, &stream),
                dpf!(pre_gates, &stream),
                dpm!(y, &stream),
                t as i32,
                streams as i32,
                hidden as i32,
                sp(&stream),
            ),
        )?;
    }
    Ok(())
}

/// Persistent T=1 decode workspace for the hc glue (lane/glm5-decode-diet lever 2,
/// `MEMRA_HC_DECODE_WS`). One per engine (pp stage), pooled on the `Engine` like
/// `fa_part_pool`/`router_stage`: the launch-diet census measured 2,358
/// `cuMemAllocAsync+Free` calls/token (~2.5 ms of host time feeding the sync-serialized
/// drain cycles), and the hc glue chain — mixes, gates, comb, collapse y, the two norm
/// scratches and the two per-site post outputs — re-allocated all of it every token. Every
/// buffer here is FULLY OVERWRITTEN before any read on every step (GEMV beta=0, block-per-
/// token kernels, rms_norm, hc_post), which is what makes reuse byte-identical: the same
/// kernels read and write the same values, only the allocator calls disappear.
///
/// The stream-state ping-pong deliberately has ONE slot (`xb`): the walk swaps the owned
/// in-flight state `x` with `xb` after each site's `hc_post`, so the pair rotates without a
/// copy and the walk still returns an owned buffer to the caller (no signature churn at the
/// stage boundary — the ppN transport consumes it exactly as before).
pub struct HyperDecodeWs {
    pub mixes: CudaSlice<f32>,
    pub pre: CudaSlice<f32>,
    pub post: CudaSlice<f32>,
    pub comb: CudaSlice<f32>,
    pub y: CudaSlice<f32>,
    /// Attention-site rms_norm scratch (the walk's `h`).
    pub h: CudaSlice<f32>,
    /// MLP-site rms_norm scratch (the walk's `z`).
    pub z: CudaSlice<f32>,
    /// The `hc_post` output slot the walk ping-pongs with the in-flight stream state.
    pub xb: CudaSlice<f32>,
    streams: usize,
    hidden: usize,
}

impl HyperDecodeWs {
    pub fn new(e: &Engine, topology: &HyperTopology, hidden: usize) -> Res<Self> {
        let streams = topology.streams;
        Ok(Self {
            mixes: e.uninit(topology.rows())?,
            pre: e.uninit(streams)?,
            post: e.uninit(streams)?,
            comb: e.uninit(streams * streams)?,
            y: e.uninit(hidden)?,
            h: e.uninit(hidden)?,
            z: e.uninit(hidden)?,
            xb: e.uninit(streams * hidden)?,
            streams,
            hidden,
        })
    }

    /// A pooled workspace is only reusable for the same trunk geometry; anything else is
    /// rebuilt (one engine serves one loaded model in practice, this is a guard, not a path).
    pub fn matches(&self, topology: &HyperTopology, hidden: usize) -> bool {
        self.streams == topology.streams && self.hidden == hidden
    }
}

/// `pre` at T=1 into the workspace: the SAME m=1 mixes program the allocating entry runs
/// (`linear_t1_into` is `linear` at m == 1 — same cuBLASLt config, same weight pointer, same
/// input bytes; the `pre_exact` note), then the shared `pre_finish_into` arms. Byte-identical
/// to `pre(e, topology, site, x, 1, hidden)` with the outputs landing in `ws` instead of
/// fresh allocations.
pub fn pre_t1_ws(
    e: &Engine,
    topology: &HyperTopology,
    site: &HyperSite,
    x: &CudaSlice<f32>,
    ws: &mut HyperDecodeWs,
    hidden: usize,
) -> Res<()> {
    let rows = topology.rows();
    let width = topology.streams * hidden;
    {
        let xr = x.slice(0..width);
        let wv = site.fn_w.slice(0..site.fn_w.len());
        let mut yr = ws.mixes.slice_mut(0..rows);
        hc_mixes_into(e, &xr, &wv, &mut yr, width, rows)
            .map_err(|err| format!("hc pre_t1_ws mixes: {err}"))?;
    }
    let ws = &mut *ws;
    pre_finish_into(
        e,
        topology,
        site,
        x,
        &mut ws.mixes,
        &mut ws.pre,
        &mut ws.post,
        &mut ws.comb,
        &mut ws.y,
        1,
        hidden,
    )
}

/// `post` at T=1 into the workspace's `xb` slot (the caller swaps `xb` with its in-flight
/// state). Reads the gates `pre_t1_ws` left in `ws.post`/`ws.comb` — the same kernel, the
/// same operand bytes as the allocating `post`.
/// Which of the decode workspace's two norm scratches `pre_t1_ws_zq8` writes: the attention
/// site's `h` or the MLP site's `z`. Passed as a tag rather than a `&mut` so the function can
/// take the whole workspace by one mutable borrow and destructure it inside.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NormDst {
    H,
    Z,
}

pub static HC_MIXES_KERNEL_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The hc mixes projection at t=1: the native kernel under `MEMRA_HC_MIXES_KERNEL=1` where the
/// shape fits, cuBLASLt (`linear_t1_into`) otherwise and by default. One seam for every site.
fn hc_mixes_into(
    e: &Engine,
    x: &cudarc::driver::CudaView<'_, f32>,
    w: &cudarc::driver::CudaView<'_, f32>,
    y: &mut cudarc::driver::CudaViewMut<'_, f32>,
    in_f: usize,
    out_f: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if Engine::hc_mixes_kernel_on() && e.hc_mixes_gemv_into(x, w, y, in_f, out_f)? {
        if HC_MIXES_KERNEL_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            eprintln!(
                "[hc-mixes-kernel] engaged in_f={in_f} out_f={out_f} (native hc_mixes_gemv_f32 \
                 in place of cuBLASLt dot+reduce; MEMRA_HC_MIXES_KERNEL=1, numeric class)"
            );
        }
        return Ok(());
    }
    e.linear_t1_into(x, w, y, in_f, out_f)
}

pub static HC_PRE_ZQ8_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Sites served by the v4 register schedule (`MEMRA_HC_PRE_V4=1`).
pub static HC_PRE_V4_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn hc_pre_zq8_selfcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_HC_PRE_ZQ8").as_deref() == Ok("2"))
}

static HC_PRE_ZQ8_CHECK_SITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HC_PRE_ZQ8_CHECK_BAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The self-check body of `pre_t1_ws_zq8` (`MEMRA_HC_PRE_ZQ8=2`): fused into scratch, then the
/// two-launch program into the workspace (which is what the walk then consumes), then a host
/// compare of pre/post/comb/y/z/q/d. Prints `[hc-pre-zq8-check]` lines: one per site with a
/// mismatch (first eight differing words with both bit patterns), plus a running summary every
/// 256 sites.
#[allow(clippy::too_many_arguments)]
fn pre_t1_ws_zq8_selfcheck(
    e: &Engine,
    topology: &HyperTopology,
    site: &HyperSite,
    x: &CudaSlice<f32>,
    ws: &mut HyperDecodeWs,
    hidden: usize,
    norm_w: &CudaSlice<f32>,
    dst: NormDst,
    eps_norm: f32,
) -> Res<()> {
    let streams = topology.streams;
    let rows = topology.rows();
    let width = streams * hidden;
    let block = crate::hc_pre_block();
    let rms_bd = crate::rms_block() as usize;
    let stream = e.stream();
    {
        let xr = x.slice(0..width);
        let wv = site.fn_w.slice(0..site.fn_w.len());
        let mut yr = ws.mixes.slice_mut(0..rows);
        e.linear_t1_into(&xr, &wv, &mut yr, width, rows)
            .map_err(|err| format!("hc pre_t1_ws_zq8 selfcheck mixes: {err}"))?;
    }
    // fused program into scratch
    let mut s_pre = e.uninit(streams)?;
    let mut s_post = e.uninit(streams)?;
    let mut s_comb = e.uninit(streams * streams)?;
    let mut s_y = e.uninit(hidden)?;
    let mut s_z = e.uninit(hidden)?;
    let mut s_q = e.uninit_i8(hidden)?;
    let mut s_d = e.uninit(hidden / 32)?;
    unsafe {
        ck(
            "hc_pre_zq8 (selfcheck fused)",
            k::memra_dsv4_hc_pre_zq8(
                dpf!(x, &stream),
                dpf!(&ws.mixes, &stream),
                dpf!(site.scale, &stream),
                dpf!(site.base, &stream),
                dpm!(&mut s_pre, &stream),
                dpm!(&mut s_post, &stream),
                dpm!(&mut s_comb, &stream),
                dpm!(&mut s_y, &stream),
                1,
                streams as i32,
                hidden as i32,
                topology.sinkhorn_iterations as i32,
                topology.epsilon,
                std::ptr::null_mut(),
                block as i32,
                crate::hc_pre_sink_reg() as i32,
                dpf!(norm_w, &stream),
                dpm!(&mut s_z, &stream),
                s_q.device_ptr_mut(&stream).0 as *mut i8,
                s_d.device_ptr_mut(&stream).0 as *mut f32,
                rms_bd as i32,
                eps_norm,
                sp(&stream),
            ),
        )?;
    }
    // the two-launch program into the workspace, exactly as the walk runs it
    {
        let ws2 = &mut *ws;
        pre_finish_into(
            e,
            topology,
            site,
            x,
            &mut ws2.mixes,
            &mut ws2.pre,
            &mut ws2.post,
            &mut ws2.comb,
            &mut ws2.y,
            1,
            hidden,
        )?;
    }
    let (r_q, r_d) = {
        let zdst: &mut CudaSlice<f32> = match dst {
            NormDst::H => &mut ws.h,
            NormDst::Z => &mut ws.z,
        };
        e.rms_norm_zq8_f32(&ws.y, norm_w, zdst, hidden, 1, eps_norm)?
    };
    stream.synchronize()?;
    let ord = HC_PRE_ZQ8_CHECK_SITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let f = |a: &[f32], b: &[f32], name: &str| -> Vec<String> {
        a.iter()
            .zip(b)
            .enumerate()
            .filter(|(_, (p, q))| p.to_bits() != q.to_bits())
            .take(8)
            .map(|(i, (p, q))| {
                format!(
                    "{name}[{i}] fused={:#010x} two={:#010x}",
                    p.to_bits(),
                    q.to_bits()
                )
            })
            .collect()
    };
    let zref = match dst {
        NormDst::H => e.dtoh(&ws.h)?,
        NormDst::Z => e.dtoh(&ws.z)?,
    };
    let mut bad: Vec<String> = Vec::new();
    bad.extend(f(&e.dtoh(&s_pre)?, &e.dtoh(&ws.pre)?[..streams], "pre"));
    bad.extend(f(&e.dtoh(&s_post)?, &e.dtoh(&ws.post)?[..streams], "post"));
    bad.extend(f(
        &e.dtoh(&s_comb)?,
        &e.dtoh(&ws.comb)?[..streams * streams],
        "comb",
    ));
    bad.extend(f(&e.dtoh(&s_y)?, &e.dtoh(&ws.y)?[..hidden], "y"));
    bad.extend(f(&e.dtoh(&s_z)?, &zref[..hidden], "z"));
    bad.extend(f(&e.dtoh(&s_d)?, &e.dtoh(&r_d)?[..hidden / 32], "d"));
    let (sq, rq) = (e.dtoh_i8(&s_q)?, e.dtoh_i8(&r_q)?);
    let qbad: Vec<String> = sq
        .iter()
        .zip(rq.iter())
        .enumerate()
        .filter(|(_, (p, q))| p != q)
        .take(8)
        .map(|(i, (p, q))| format!("q[{i}] fused={p} two={q}"))
        .collect();
    bad.extend(qbad);
    if !bad.is_empty() {
        HC_PRE_ZQ8_CHECK_BAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[hc-pre-zq8-check] site #{ord} dst={dst:?} MISMATCH: {}",
            bad.join("; ")
        );
    }
    if ord.is_multiple_of(256) {
        eprintln!(
            "[hc-pre-zq8-check] {} sites compared, {} with a mismatch",
            ord + 1,
            HC_PRE_ZQ8_CHECK_BAD.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    Ok(())
}

/// `pre_t1_ws` and the `rms_norm_zq8` that consumes its `y`, as ONE launch (door
/// `MEMRA_HC_PRE_ZQ8`, lane/hcpre-zq8-fusion-20260905). Returns the q8_1 pair the walk would
/// otherwise get from `Engine::rms_norm_zq8_f32`, with `z` (the normed f32 row) written to
/// `ws.h` or `ws.z` per `dst`. Returns `None` -- without launching anything -- wherever the
/// fused kernel's preconditions do not hold, so the caller runs the two-launch program
/// unchanged: the fused pre arm must be V2 (the v3 kernel is what this body was generated
/// from), the norm's block must fit inside the pre-chain's, and the site must be within the
/// v3 kernel's warp-0 invariant.
#[allow(clippy::too_many_arguments)]
pub fn pre_t1_ws_zq8(
    e: &Engine,
    topology: &HyperTopology,
    site: &HyperSite,
    x: &CudaSlice<f32>,
    ws: &mut HyperDecodeWs,
    hidden: usize,
    norm_w: &CudaSlice<f32>,
    dst: NormDst,
    eps_norm: f32,
) -> Res<Option<(CudaSlice<i8>, CudaSlice<f32>)>> {
    let streams = topology.streams;
    let rows = topology.rows();
    let width = streams * hidden;
    let block = crate::hc_pre_block();
    let rms_bd = crate::rms_block() as usize;
    if hc_fused_pre_arm() != HcFusedPreArm::V2
        || streams > 8
        || rows > 32
        || rms_bd > block
        || !rms_bd.is_multiple_of(32)
        || !hidden.is_multiple_of(32)
    {
        return Ok(None);
    }
    // MEMRA_HC_PRE_ZQ8=2 (self-check, 2026-09-05): the served tape forks between the fused and
    // the two-launch program while the kernel gate is bitwise green on the served card. This arm
    // runs the fused kernel into SCRATCH, then returns None so the walk runs the real two-launch
    // program into the workspace, and compares all seven outputs on the host, printing the first
    // differing words per site with the site ordinal. Diagnostic only: dtoh per site.
    if hc_pre_zq8_selfcheck() {
        return pre_t1_ws_zq8_selfcheck(e, topology, site, x, ws, hidden, norm_w, dst, eps_norm)
            .map(|()| None);
    }
    let stream = e.stream();
    {
        let xr = x.slice(0..width);
        let wv = site.fn_w.slice(0..site.fn_w.len());
        let mut yr = ws.mixes.slice_mut(0..rows);
        hc_mixes_into(e, &xr, &wv, &mut yr, width, rows)
            .map_err(|err| format!("hc pre_t1_ws_zq8 mixes: {err}"))?;
    }
    let mut q = e.uninit_i8(hidden)?;
    let mut d = e.uninit(hidden / 32)?;
    let HyperDecodeWs {
        mixes,
        pre,
        post,
        comb,
        y,
        h,
        z,
        ..
    } = ws;
    let zdst: &mut CudaSlice<f32> = match dst {
        NormDst::H => h,
        NormDst::Z => z,
    };
    unsafe {
        ck(
            "hc_pre_zq8",
            k::memra_dsv4_hc_pre_zq8(
                dpf!(x, &stream),
                dpf!(mixes, &stream),
                dpf!(site.scale, &stream),
                dpf!(site.base, &stream),
                dpm!(pre, &stream),
                dpm!(post, &stream),
                dpm!(comb, &stream),
                dpm!(y, &stream),
                1,
                streams as i32,
                hidden as i32,
                topology.sinkhorn_iterations as i32,
                topology.epsilon,
                std::ptr::null_mut(),
                block as i32,
                crate::hc_pre_sink_reg() as i32,
                dpf!(norm_w, &stream),
                dpm!(zdst, &stream),
                q.device_ptr_mut(&stream).0 as *mut i8,
                d.device_ptr_mut(&stream).0 as *mut f32,
                rms_bd as i32,
                eps_norm,
                sp(&stream),
            ),
        )?;
    }
    if HC_PRE_ZQ8_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        eprintln!(
            "[hc-pre-zq8] engaged streams={streams} hidden={hidden} block={block} rms_bd={rms_bd} \
             (one launch replaces hc_pre_fused_v3 + rms_norm_zq8_f32_v2 per site; MEMRA_HC_PRE_ZQ8=1)"
        );
    }
    Ok(Some((q, d)))
}

pub fn post_t1_ws(
    e: &Engine,
    topology: &HyperTopology,
    f: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    ws: &mut HyperDecodeWs,
    hidden: usize,
) -> Res<()> {
    let stream = e.stream();
    let ws = &mut *ws;
    unsafe {
        ck(
            "hc_post",
            k::memra_dsv4_hc_post(
                dpf!(f, &stream),
                dpf!(residual, &stream),
                dpf!(ws.post, &stream),
                dpf!(ws.comb, &stream),
                dpm!(ws.xb, &stream),
                1,
                topology.streams as i32,
                hidden as i32,
                sp(&stream),
            ),
        )?;
    }
    Ok(())
}

/// One site's post-branch half (`hc_post`): `out[t, k, :] = post[t, k]·f[t, :] + Σ_j
/// comb[t, j, k]·residual[t, j, :]`. `residual` is the site's INPUT stream state, not the
/// layer's — the MLP site's residual is the attention site's output.
pub fn post(
    e: &Engine,
    topology: &HyperTopology,
    f: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    mix: &HcMix,
    t: usize,
    hidden: usize,
) -> Res<CudaSlice<f32>> {
    let streams = topology.streams;
    let mut out = e.uninit(t * streams * hidden)?;
    let stream = e.stream();
    unsafe {
        ck(
            "hc_post",
            k::memra_dsv4_hc_post(
                dpf!(f, &stream),
                dpf!(residual, &stream),
                dpf!(mix.post, &stream),
                dpf!(mix.comb, &stream),
                dpm!(out, &stream),
                t as i32,
                streams as i32,
                hidden as i32,
                sp(&stream),
            ),
        )?;
    }
    Ok(out)
}

/// UNWEIGHTED stream-mean contraction `[tokens, streams, hidden]` -> `[tokens, hidden]` —
/// the `hc_contract` the glm5 DFlash2 drafter's aux-hidden features are defined by (the
/// probe's capture seam: mean over the hc_mult stream blocks of the completed layer output,
/// == the SGLang glm5_next integration's pinned definition). Deliberately NOT keyed on
/// `topology.collapse`: the drafter contract is the mean by definition, whatever the trunk
/// exit does (for glm5_next the exit IS `Mean`, so this is also the collapse kernel).
pub fn contract_mean(
    e: &Engine,
    topology: &HyperTopology,
    x: &CudaSlice<f32>,
    t: usize,
    hidden: usize,
) -> Res<CudaSlice<f32>> {
    let streams = topology.streams;
    let stream = e.stream();
    let mut out = e.uninit(t * hidden)?;
    unsafe {
        ck(
            "hc_mean",
            k::memra_dsv4_hc_mean(
                dpf!(x, &stream),
                dpm!(out, &stream),
                t as i32,
                streams as i32,
                hidden as i32,
                sp(&stream),
            ),
        )?;
    }
    Ok(out)
}

/// Trunk exit: `[tokens, streams, hidden]` -> `[tokens, hidden]`, keyed on the plan's collapse.
/// `Mean` is glm5_next's unweighted mean (`Glm5NextTextHyperHead`); `GatedHead` is dsv4's
/// sigmoid-gated pre-only collapse (`dsv4_forward::hc_head`) and needs the head trio.
pub fn collapse(
    e: &Engine,
    topology: &HyperTopology,
    head: Option<&HyperHead>,
    x: &CudaSlice<f32>,
    t: usize,
    hidden: usize,
) -> Res<CudaSlice<f32>> {
    let streams = topology.streams;
    let stream = e.stream();
    let mut out = e.uninit(t * hidden)?;
    match topology.collapse {
        HcCollapse::Mean => unsafe {
            ck(
                "hc_mean",
                k::memra_dsv4_hc_mean(
                    dpf!(x, &stream),
                    dpm!(out, &stream),
                    t as i32,
                    streams as i32,
                    hidden as i32,
                    sp(&stream),
                ),
            )?;
        },
        HcCollapse::GatedHead => {
            let head = head.ok_or_else(|| {
                "HcCollapse::GatedHead reached the trunk exit with no head trio loaded".to_string()
            })?;
            let width = streams * hidden;
            let mut mixes = e.linear(x, &head.fn_w, t, width, streams)?;
            let mut gates = e.uninit(t * streams)?;
            unsafe {
                ck(
                    "hc_head rowsq_scale",
                    k::memra_dsv4_rowsq_scale(
                        dpf!(x, &stream),
                        dpm!(mixes, &stream),
                        t as i32,
                        width as i32,
                        streams as i32,
                        topology.epsilon,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "hc_head_pre",
                    k::memra_dsv4_hc_head_pre_m(
                        dpf!(mixes, &stream),
                        dpf!(head.scale, &stream),
                        dpf!(head.base, &stream),
                        dpm!(gates, &stream),
                        t as i32,
                        streams as i32,
                        topology.epsilon,
                        sp(&stream),
                    ),
                )?;
                ck(
                    "hc_head collapse",
                    k::memra_dsv4_hc_collapse(
                        dpf!(x, &stream),
                        dpf!(gates, &stream),
                        dpm!(out, &stream),
                        t as i32,
                        streams as i32,
                        hidden as i32,
                        sp(&stream),
                    ),
                )?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::model_plan::{
        ActivationPlan, AttentionPlan, DenseMlpPlan, DraftSourcePlan, KimiDeltaNetPlan, LayerPlan,
        MlpPlan, NormKind, NormPlan, StatePlan, WeightTransform,
    };

    fn norm() -> NormPlan {
        NormPlan {
            kind: NormKind::Rms,
            epsilon: 1e-5,
            weight_transform: WeightTransform::Identity,
        }
    }

    fn layer(index: u32, residual: ResidualTopology) -> LayerPlan {
        LayerPlan {
            index,
            pre_attention_norm: norm(),
            attention: AttentionPlan::KimiDeltaNet(KimiDeltaNetPlan {
                num_heads: 1,
                head_dim: 128,
                conv_kernel: 4,
                gate_lower_bound: -5.0,
            }),
            pre_mlp_norm: norm(),
            mlp: MlpPlan::Dense(DenseMlpPlan {
                intermediate_size: 16,
                activation: ActivationPlan::Silu,
            }),
            residual,
            state: StatePlan::Recurrent {
                conv_width: 384,
                conv_kernel: 4,
                state_width: 16384,
            },
            ple: None,
            sparse_overlay: None,
        }
    }

    fn plan(residuals: [ResidualTopology; 2]) -> ModelPlan {
        ModelPlan {
            arch: memra_gguf::config::Arch::Glm5Next,
            hidden_size: 8,
            vocab_size: 16,
            context_length: 32,
            embedding_scale: 1.0,
            vision: None,
            multimodal: None,
            layers: vec![layer(0, residuals[0]), layer(1, residuals[1])],
            output_norm: norm(),
            logits: Vec::new(),
            mtp_blocks: Vec::new(),
            drafter: None,
            exit_mixer: None,
            draft_source: DraftSourcePlan::Embedded,
            sampling_defaults: None,
            partition_boundaries: Vec::new(),
        }
    }

    fn hc(streams: u32) -> ResidualTopology {
        ResidualTopology::HyperConnections {
            streams,
            epsilon: 1e-6,
            sinkhorn_iterations: 20,
            collapse: HcCollapse::Mean,
        }
    }

    #[test]
    fn serial_trunk_has_no_topology() {
        let plan = plan([ResidualTopology::Serial, ResidualTopology::Serial]);
        assert!(HyperTopology::from_plan(&plan).unwrap().is_none());
    }

    #[test]
    fn uniform_trunk_yields_the_plans_constants() {
        let plan = plan([hc(4), hc(4)]);
        let topology = HyperTopology::from_plan(&plan).unwrap().unwrap();
        assert_eq!(topology.streams, 4);
        assert_eq!(topology.sinkhorn_iterations, 20);
        assert_eq!(topology.collapse, HcCollapse::Mean);
        // pre gates + post gates + the streams x streams combination block.
        assert_eq!(topology.rows(), 24);
    }

    #[test]
    fn a_mixed_trunk_is_refused_in_both_orders() {
        for residuals in [
            [hc(4), ResidualTopology::Serial],
            [ResidualTopology::Serial, hc(4)],
            [hc(4), hc(2)],
        ] {
            assert!(
                HyperTopology::from_plan(&plan(residuals)).is_err(),
                "a non-uniform trunk must be refused, not silently keyed off layer 0"
            );
        }
    }

    #[test]
    fn zero_iterations_are_refused() {
        let bad = ResidualTopology::HyperConnections {
            streams: 4,
            epsilon: 1e-6,
            sinkhorn_iterations: 0,
            collapse: HcCollapse::Mean,
        };
        assert!(HyperTopology::from_plan(&plan([bad, bad])).is_err());
    }
}
