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
    let streams = topology.streams;
    let rows = topology.rows();
    let width = streams * hidden;
    let eps = topology.epsilon;
    let stream = e.stream();

    let mut mixes = e.linear(x, &site.fn_w, t, width, rows)?;
    let mut pre_gates = e.uninit(t * streams)?;
    let mut post = e.uninit(t * streams)?;
    let mut comb = e.uninit(t * streams * streams)?;
    let mut y = e.uninit(t * hidden)?;
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
    Ok((y, HcMix { post, comb }))
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
