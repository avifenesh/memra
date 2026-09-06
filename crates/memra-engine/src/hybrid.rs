//! Qwen3.5/3.6 hybrid model: linear-attention (Gated DeltaNet) layers + periodic full-attention
//! layers + SwiGLU FFN. Loads weights, runs the forward, dual cache. Builds on the validated
//! conv1d + gdn_scan kernels (M2/M3) and the dense full-attn path (M0).

use crate::Engine;
use crate::model::{EmbedHost, GpuTensor, HostExps};
use cudarc::driver::CudaSlice;
use memra_gguf::config::{ModelConfig, SwigluClamp};
use memra_gguf::model_plan::{AttentionPlan, MlpPlan, TensorPresence};
use memra_gguf::source::{GgufSource, TensorSource};
use memra_gguf::{GgmlType, GgufFile};
use std::collections::HashMap;
use std::sync::Arc;

// Source-agnostic load helpers (GGUF or safetensors). The GGUF wrappers below keep `load()`
// byte-identical; only the source object differs.
fn load_t(
    e: &Engine,
    src: &dyn TensorSource,
    name: &str,
) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    GpuTensor::load_from_source(e, src, name)
}
fn load_opt(
    e: &Engine,
    src: &dyn TensorSource,
    name: &str,
) -> Result<Option<GpuTensor>, Box<dyn std::error::Error>> {
    GpuTensor::load_opt_from_source(e, src, name)
}

struct ResidencyBytes {
    experts: HashMap<usize, usize>,
    rest: usize,
    saw_experts: bool,
}

fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

fn residency_bytes_by_device<'a>(
    tensors: impl IntoIterator<Item = (&'a str, usize)>,
    layer_devices: &[usize],
    primary_device: usize,
) -> ResidencyBytes {
    let mut out = ResidencyBytes {
        experts: HashMap::new(),
        rest: 0,
        saw_experts: false,
    };
    for (name, bytes) in tensors {
        if name.starts_with("blk.") && name.contains("_exps.") {
            let device = block_index(name)
                .and_then(|il| layer_devices.get(il).copied())
                .unwrap_or(primary_device);
            *out.experts.entry(device).or_default() += bytes;
            out.saw_experts = true;
        } else {
            out.rest += bytes;
        }
    }
    out
}

/// Load-local resident-expert capacity decisions. PP stages on distinct devices are charged only
/// for their own layer slices; co-located stages share a device key and are charged together.
pub(crate) struct ResidentPlan {
    primary_device: usize,
    layer_devices: Vec<usize>,
    layer_counts: HashMap<usize, usize>,
    exact_expert_bytes: Option<HashMap<usize, usize>>,
    trunk_bytes: usize,
    decisions: HashMap<usize, bool>,
    pp: bool,
}

/// Model-load-local CUDA rank runtimes, keyed by their ordered device group.
///
/// Step layers keep their own checkpoint shards, but layers assigned to the same TP/EP group must
/// reuse one set of CUDA contexts, streams, and cuBLAS handles. Constructing a runtime per layer
/// multiplies context memory and makes multi-layer distributed serving impractical.
/// Which native expert artifact class the checkpoint census qualified. Every distributed expert
/// program keys on this: E4M3 = official FP8 (block-128 banks), Nvfp4 = official NVFP4 (packed
/// e2m1 + per-16 UE4M3 + per-expert macro). One checkpoint is exactly one class — mixing refuses
/// at census, never at decode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum StepExpertArtifact {
    #[default]
    E4m3,
    Nvfp4,
}

#[derive(Clone, Debug, Default)]
struct StepParallelLoadConfig {
    ep_specs: Vec<crate::tp::StepEpLayerSpec>,
    tp_specs: Vec<crate::tp::StepTpLayerSpec>,
    native_p2p: bool,
    ep_device_arithmetic: bool,
    f32_mirror: bool,
    bulk_p2p: bool,
    nvfp4_device_routes: bool,
    auto_parallel: bool,
    tp_attention_expert_overlap: bool,
    expert_artifact: StepExpertArtifact,
}

#[derive(Default)]
pub(crate) struct StepParallelRuntimeRegistry {
    config: StepParallelLoadConfig,
    runtimes: HashMap<(Vec<usize>, bool, bool, bool), Arc<crate::tp::TpE4m3HostBounce>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepExpertLayout {
    TensorParallel,
    ExpertParallel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StepExpertSelection {
    spec: crate::tp::StepEpLayerSpec,
    layout: StepExpertLayout,
    configured_by_tp: bool,
}

fn select_step_expert_layout_inner(
    layer: usize,
    ep_specs: &[crate::tp::StepEpLayerSpec],
    tp_specs: &[crate::tp::StepTpLayerSpec],
    allow_attention_ep_overlap: bool,
) -> Result<Option<StepExpertSelection>, String> {
    let ep = ep_specs.iter().find(|spec| spec.layer == layer);
    let tp = tp_specs.iter().find(|spec| spec.layer == layer);
    Ok(match (ep, tp) {
        (Some(spec), None) => Some(StepExpertSelection {
            spec: spec.clone(),
            layout: StepExpertLayout::ExpertParallel,
            configured_by_tp: false,
        }),
        (None, Some(spec)) => Some(StepExpertSelection {
            spec: spec.clone(),
            layout: if spec.devices.len() > 2 {
                StepExpertLayout::ExpertParallel
            } else {
                StepExpertLayout::TensorParallel
            },
            configured_by_tp: true,
        }),
        (None, None) => None,
        (Some(ep), Some(tp)) => {
            if !allow_attention_ep_overlap {
                return Err(format!(
                    "Step layer {layer} cannot enable MEMRA_STEP_EP and MEMRA_STEP_TP together"
                ));
            }
            if ep.devices.first() != tp.devices.first()
                || tp.devices.iter().any(|device| !ep.devices.contains(device))
            {
                return Err(format!(
                    "automatic TP-attention/EP overlap at layer {layer} requires the attention \
                     ranks {:?} to be an owner-first subset of expert ranks {:?}",
                    tp.devices, ep.devices
                ));
            }
            Some(StepExpertSelection {
                spec: ep.clone(),
                layout: StepExpertLayout::ExpertParallel,
                configured_by_tp: false,
            })
        }
    })
}

#[cfg(test)]
fn select_step_expert_layout(
    layer: usize,
    ep_specs: &[crate::tp::StepEpLayerSpec],
    tp_specs: &[crate::tp::StepTpLayerSpec],
) -> Result<Option<StepExpertSelection>, String> {
    select_step_expert_layout_inner(layer, ep_specs, tp_specs, false)
}

impl StepParallelRuntimeRegistry {
    fn with_config(config: StepParallelLoadConfig) -> Self {
        Self {
            config,
            runtimes: HashMap::new(),
        }
    }

    fn tp_spec(&self, layer: usize) -> Option<&crate::tp::StepTpLayerSpec> {
        self.config.tp_specs.iter().find(|spec| spec.layer == layer)
    }

    fn expert_selection(&self, layer: usize) -> Result<Option<StepExpertSelection>, String> {
        select_step_expert_layout_inner(
            layer,
            &self.config.ep_specs,
            &self.config.tp_specs,
            self.config.tp_attention_expert_overlap,
        )
    }

    fn runtime(
        &mut self,
        devices: &[usize],
        native_p2p: bool,
        ep_device_arithmetic: bool,
    ) -> Result<Arc<crate::tp::TpE4m3HostBounce>, Box<dyn std::error::Error>> {
        let bulk_p2p = self.config.bulk_p2p && native_p2p;
        let key = (devices.to_vec(), native_p2p, ep_device_arithmetic, bulk_p2p);
        if let Some(runtime) = self.runtimes.get(&key) {
            return Ok(Arc::clone(runtime));
        }
        let runtime = Arc::new(crate::tp::TpE4m3HostBounce::new_configured(
            devices,
            native_p2p,
            ep_device_arithmetic,
            bulk_p2p,
        )?);
        let names = runtime.device_names()?;
        if names
            .iter()
            .any(|name| !name.contains("RTX PRO 6000") || !name.contains("Blackwell"))
        {
            return Err(format!(
                "Step distributed execution is qualified only on RTX PRO 6000 Blackwell, \
                 got {names:?}"
            )
            .into());
        }
        self.runtimes.insert(key, Arc::clone(&runtime));
        Ok(runtime)
    }
}

impl ResidentPlan {
    fn from_layout(
        src: &dyn TensorSource,
        primary_device: usize,
        layer_devices: Vec<usize>,
        pp: bool,
    ) -> Self {
        let mut layer_counts = HashMap::new();
        for &device in &layer_devices {
            *layer_counts.entry(device).or_default() += 1;
        }
        let (exact_expert_bytes, trunk_bytes) = match src.gguf() {
            Some(g) => {
                let bytes = residency_bytes_by_device(
                    g.tensors
                        .iter()
                        .map(|t| (t.name.as_str(), t.n_bytes as usize)),
                    &layer_devices,
                    primary_device,
                );
                if bytes.saw_experts {
                    (Some(bytes.experts), bytes.rest)
                } else {
                    (None, 0)
                }
            }
            None => (None, 0),
        };
        Self {
            primary_device,
            layer_devices,
            layer_counts,
            exact_expert_bytes,
            trunk_bytes,
            decisions: HashMap::new(),
            pp,
        }
    }

    pub(crate) fn unsharded(e: &Engine, src: &dyn TensorSource, cfg: &ModelConfig) -> Self {
        let device = e.ctx().ordinal();
        Self::from_layout(src, device, vec![device; cfg.n_layer as usize], false)
    }

    pub(crate) fn pp(
        e: &Engine,
        src: &dyn TensorSource,
        cfg: &ModelConfig,
        n_trunk: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let primary = e.ctx().ordinal();
        let Some(_fence) = crate::pp::pp_cuts(n_trunk) else {
            return Ok(Self::unsharded(e, src, cfg));
        };
        let mut layer_devices = vec![primary; cfg.n_layer as usize];
        for (il, device) in layer_devices.iter_mut().take(n_trunk).enumerate() {
            *device = crate::pp::layer_engine(e, n_trunk, il)?.ctx().ordinal();
        }
        Ok(Self::from_layout(src, primary, layer_devices, true))
    }

    /// Distributed expert layers no longer consume the owning stage's local expert slab. Remove
    /// them from the fallback per-layer residency estimate so later local-only expert layers
    /// (for example an embedded MTP block) are judged on their own remaining footprint.
    fn exclude_distributed_expert_layers(&mut self, specs: impl IntoIterator<Item = usize>) {
        for layer in specs {
            let device = self
                .layer_devices
                .get(layer)
                .copied()
                .unwrap_or(self.primary_device);
            if let Some(count) = self.layer_counts.get_mut(&device) {
                *count = count.saturating_sub(1);
            }
        }
    }

    fn should_reside(&mut self, e: &Engine, il: usize, per_layer: usize) -> bool {
        let device = self
            .layer_devices
            .get(il)
            .copied()
            .unwrap_or(self.primary_device);
        debug_assert_eq!(e.ctx().ordinal(), device);
        if let Some(&decision) = self.decisions.get(&device) {
            return decision;
        }
        if std::env::var("MEMRA_MOE_RESIDENT").as_deref() == Ok("0") {
            self.decisions.insert(device, false);
            return false;
        }
        let (free, _total) = match e.ctx().mem_get_info() {
            Ok(v) => v,
            Err(_) => {
                self.decisions.insert(device, false);
                return false;
            }
        };
        let projected = self
            .exact_expert_bytes
            .as_ref()
            .map(|bytes| bytes.get(&device).copied().unwrap_or(0))
            .unwrap_or(per_layer * self.layer_counts.get(&device).copied().unwrap_or(1));
        let budget = std::env::var("MEMRA_MOE_RESIDENT_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|gb| (gb * 1e9) as usize)
            .unwrap_or_else(|| {
                let reserve = std::env::var("MEMRA_MOE_RESIDENT_HEADROOM_GB")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|gb| (gb * 1e9) as usize)
                    .unwrap_or(2_000_000_000);
                free.saturating_sub(self.trunk_bytes + reserve)
            });
        let ok = projected <= budget;
        eprintln!(
            "[moe] resident-experts decision ({}dev{}): experts {:.2}GB + trunk {:.2}GB vs free {:.2}GB (expert budget {:.2}GB) -> {}",
            if self.pp { "PP " } else { "" },
            device,
            projected as f64 / 1e9,
            self.trunk_bytes as f64 / 1e9,
            free as f64 / 1e9,
            budget as f64 / 1e9,
            if ok { "RESIDENT" } else { "SLRU cache" }
        );
        self.decisions.insert(device, ok);
        ok
    }
}

/// Load the mixer declared by one canonical layer. Shared by trunk and MTP loaders.
fn load_mixer_kind(
    e: &Engine,
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    il: u32,
    attention: &AttentionPlan,
    step_runtimes: &mut StepParallelRuntimeRegistry,
) -> Result<Mixer, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    Ok(match attention {
        AttentionPlan::Mla(mla) => Mixer::Mla(MlaAttnLayer::load(e, src, il, mla)?),
        AttentionPlan::Full(full)
        | AttentionPlan::SlidingWindow {
            attention: full, ..
        } => {
            Mixer::Full(FullAttnLayer {
                wq: load_t(e, src, &p("attn_q.weight"))?,
                wk: load_t(e, src, &p("attn_k.weight"))?,
                // gemma4 global layers ship NO v_proj (attention_k_eq_v): V = the K projection
                // output pre-rope (llama gemma4.cpp: `Vcur = wv ? mm(wv,cur) : Kcur`). Loading
                // wv := wk reproduces that exactly with zero forward changes; the gemma forward
                // adds the weightless V rms_norm (R7 part 2).
                wv: match load_opt(e, src, &p("attn_v.weight"))? {
                    Some(v) => v,
                    None => load_t(e, src, &p("attn_k.weight"))?,
                },
                wo: load_t(e, src, &p("attn_output.weight"))?,
                // AWQ artifacts only (memra#253); absent everywhere else.
                wo_pqs: load_opt(e, src, &p("attn_output.pre_quant_scale"))?,
                q_norm: match full.qk_norm {
                    TensorPresence::Absent => None,
                    TensorPresence::Optional => load_opt(e, src, &p("attn_q_norm.weight"))?,
                    TensorPresence::Required => Some(load_t(e, src, &p("attn_q_norm.weight"))?),
                },
                k_norm: match full.qk_norm {
                    TensorPresence::Absent => None,
                    TensorPresence::Optional => load_opt(e, src, &p("attn_k_norm.weight"))?,
                    TensorPresence::Required => Some(load_t(e, src, &p("attn_k_norm.weight"))?),
                },
                // step35: REQUIRED when the arch says so — a missing gate would silently drop the
                // per-head sigmoid and produce plausible-but-wrong logits, so this is load_t not
                // load_opt. Step-3.7-Flash ships it on all 45 blocks (width = that layer's n_head).
                attn_gate: if full.output_gate
                    == memra_gguf::config::AttentionGateKind::SeparateHead
                {
                    Some(load_t(e, src, &p("attn_gate.weight"))?)
                } else {
                    None
                },
                step_tp_qkv: build_step_tp_qkv(e, src, cfg, il as usize, step_runtimes)?,
            })
        }
        // glm5_next KDA (Kimi Delta Attention). Geometry refusals (head_dim, conv width) live
        // in KdaAttnLayer::load so an unsupported shape fails at load, never in a kernel.
        AttentionPlan::KimiDeltaNet(kda) => {
            Mixer::Kda(crate::kda::KdaAttnLayer::load(e, src, il, kda)?)
        }
        AttentionPlan::GatedDeltaNet(geometry) => Mixer::Linear(LinearAttnLayer {
            geometry: *geometry,
            wqkv: load_t(e, src, &p("attn_qkv.weight"))?,
            wqkv_gate: load_t(e, src, &p("attn_gate.weight"))?,
            ssm_beta: load_t(e, src, &p("ssm_beta.weight"))?,
            ssm_alpha: load_t(e, src, &p("ssm_alpha.weight"))?,
            ssm_a: load_t(e, src, &p("ssm_a"))?,
            ssm_dt: load_t(e, src, &p("ssm_dt.bias"))?,
            ssm_conv1d: load_t(e, src, &p("ssm_conv1d.weight"))?,
            ssm_norm: load_t(e, src, &p("ssm_norm.weight"))?,
            ssm_out: load_t(e, src, &p("ssm_out.weight"))?,
        }),
    })
}

/// Load the FFN (dense SwiGLU or routed MoE) for block `il`. Source-agnostic (GGUF or safetensors
/// via `TensorSource`); shared by the hybrid trunk/MTP loops AND the dense-attention MoE path (OLMoE).
/// Shared-expert tensors are OPTIONAL (`load_opt`): qwen35moe has them, OLMoE/vanilla-MoE do not.
/// When `spill` is `Some` (MEMRA_SPILL_DISK on) AND the source is the GGUF on disk, MoE experts load
/// through the per-expert tier split (`HostExps::load_tiered`: hottest pinned, rest mmap'd from disk);
/// otherwise experts take the all-host / gather path. Spill tiering is GGUF-only (needs the file mmap).
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
pub(crate) fn load_ffn(
    e: &Engine,
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    mlp: &MlpPlan,
    il: u32,
    spill: Option<(&GgufFile, &mut crate::spill::SpillCtx)>,
    resident: &mut ResidentPlan,
    step_runtimes: &mut StepParallelRuntimeRegistry,
) -> Result<Ffn, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    // ARTIFACT-DENSE OVERRIDE (restores the pre-plan nuance d143604b0a removed): Step3.7-flash
    // ships its MTP blocks (blk.45/46/47) with `ffn_gate/up/down.weight` and NO
    // `ffn_gate_inp`/`ffn_*_exps`, while the config carries the TRUNK's expert hparams — so a
    // plan-typed Moe block whose artifact ships neither stacked nor fused expert tensors but
    // does ship the dense projection loads DENSE, exactly as it did before the plan-driven
    // loader (the old load path keyed this on tensor presence, not hparams).
    let artifact_dense = matches!(mlp, MlpPlan::Moe(_))
        && !src.has(&p("ffn_gate_exps.weight"))
        && !src.has(&p("ffn_gate_up_exps.weight"))
        && src.has(&p("ffn_gate.weight"));
    Ok(if artifact_dense {
        Ffn::Dense {
            ffn_gate: GpuTensor::load_from_source(e, src, &p("ffn_gate.weight"))?,
            ffn_up: GpuTensor::load_from_source(e, src, &p("ffn_up.weight"))?,
            ffn_down: GpuTensor::load_from_source(e, src, &p("ffn_down.weight"))?,
            // AWQ artifacts only (memra#253); absent everywhere else.
            ffn_down_pqs: GpuTensor::load_opt_from_source(e, src, &p("ffn_down.pre_quant_scale"))?,
        }
    } else if let MlpPlan::Moe(moe) = mlp {
        let n_expert = moe.expert_count as usize;
        // Expert loader. `spill` carries an optional (GgufFile, SpillCtx) — only the GGUF on-disk
        // path can tier (it needs the file mmap); safetensors always gathers/stacks all-host.
        //  - spill Some -> per-expert tier split (hottest pinned, rest mmap'd from the GGUF).
        //  - GGUF 3D stacked name resolves -> load_stacked_from_source (all-host).
        //  - else (safetensors) -> gather N separate 2D expert tensors.
        let (gate_exps, up_exps, down_exps) = match spill {
            Some((g, ctx)) => (
                HostExps::load_tiered(e, g, &p("ffn_gate_exps.weight"), ctx)?,
                HostExps::load_tiered(e, g, &p("ffn_up_exps.weight"), ctx)?,
                HostExps::load_tiered(e, g, &p("ffn_down_exps.weight"), ctx)?,
            ),
            None => {
                let exps = |e: &Engine, n: &str| -> Result<HostExps, Box<dyn std::error::Error>> {
                    if src.has(n) {
                        HostExps::load_stacked_from_source(e, src, n)
                    } else {
                        HostExps::load_from_source(e, src, n, n_expert)
                    }
                };
                // gemma4: gate+up ship FUSED (ffn_gate_up_exps, gate rows first) — split at load.
                let fused = p("ffn_gate_up_exps.weight");
                if !src.has(&p("ffn_gate_exps.weight")) && src.has(&fused) {
                    let ff = moe.expert_intermediate_size as usize;
                    (
                        HostExps::load_stacked_split_from_source(e, src, &fused, 0, ff)?,
                        HostExps::load_stacked_split_from_source(e, src, &fused, ff, 2 * ff)?,
                        exps(e, &p("ffn_down_exps.weight"))?,
                    )
                } else {
                    (
                        exps(e, &p("ffn_gate_exps.weight"))?,
                        exps(e, &p("ffn_up_exps.weight"))?,
                        exps(e, &p("ffn_down_exps.weight"))?,
                    )
                }
            }
        };
        let (step_ep, step_tp) = build_step_distributed_exps(
            e,
            cfg,
            src,
            il as usize,
            &gate_exps,
            &up_exps,
            &down_exps,
            step_runtimes,
        )?;
        // FITS-VRAM RESIDENT EXPERTS: upload this layer's 3 expert slabs when the owning
        // device's budget (MEMRA_MOE_RESIDENT_GB override; default = free VRAM minus the file's
        // non-expert bytes minus a measured headroom reserve) covers the expert bytes assigned
        // to that device, summed exactly from the GGUF header. Decision is made once per device
        // (first MoE layer there). Failure to fit => None => the SLRU spill machinery.
        let dev_exps = if step_ep.is_some() || step_tp.is_some() {
            None
        } else {
            build_dev_exps(e, resident, il as usize, &gate_exps, &up_exps, &down_exps)?
        };
        // Device macro row [3*n_expert]: gate, up, down (ones when the artifact carries none).
        let mut macro_row = vec![1.0f32; 3 * n_expert];
        for (slot, exps) in [(0usize, &gate_exps), (1, &up_exps), (2, &down_exps)] {
            if let Some(ms) = exps.macros.as_ref() {
                macro_row[slot * n_expert..(slot + 1) * n_expert].copy_from_slice(ms);
            }
        }
        let has_macros = macro_row.iter().any(|&m| m != 1.0);
        let dev_macros = e.htod(&macro_row)?;
        // e_score_correction_bias (sigmoid routing): retain the host oracle row and upload a
        // zero-filled device row when absent so the token loop never allocates or transfers it.
        let exp_probs_b = src
            .find(&p("exp_probs_b.bias"))
            .map(|v| memra_gguf::dequant::dequantize(v.ggml_type, &v.bytes, n_expert));
        // A plan that DECLARES the selection bias may not fall back to zeros. The zero row is
        // for routers that have no bias at all; substituting it for a bias the plan declares
        // computes a different model, silently — noaux_tc selects on `sigmoid(logit) + bias`,
        // so a zero bias reduces selection to the raw top-k while every other stage still looks
        // right (glm5_next, 2026-08-28: the ggml->HF map had no `exp_probs_b.bias` row for this
        // arch, `find` answered None here, and the served model routed to the wrong experts on
        // all 42 MoE layers). Refuse by name instead: the checkpoint either carries the tensor
        // the plan declares or this is not that model.
        if exp_probs_b.is_none()
            && matches!(
                moe.router,
                memra_gguf::model_plan::RouterPlan::Sigmoid {
                    selection_bias: true,
                    ..
                } | memra_gguf::model_plan::RouterPlan::SqrtSoftplus {
                    selection_bias: true,
                    ..
                }
            )
        {
            return Err(format!(
                "layer {il}: {} is absent, but the compiled ModelPlan declares a router with a \
                 selection bias ({:?}). Refusing to load: a zero-filled bias would route to \
                 different experts than this model does, silently. Either the checkpoint does \
                 not carry the tensor, or this arch has no `exp_probs_b.bias` entry in \
                 hf_mapping's ggml->HF map",
                p("exp_probs_b.bias"),
                moe.router
            )
            .into());
        }
        let active_experts = src.active_experts(il).map(<[bool]>::to_vec);
        let route_bias = exp_probs_b.clone().unwrap_or_else(|| vec![0.0; n_expert]);
        let active_row: Vec<u8> = active_experts
            .as_ref()
            .map(|mask| mask.iter().map(|&is_active| u8::from(is_active)).collect())
            .unwrap_or_else(|| vec![1; n_expert]);
        let exp_probs_b_dev = e.htod(&route_bias)?;
        let active_experts_dev = e.htod_bytes(&active_row)?;
        let gate_shexp = load_opt(e, src, &p("ffn_gate_shexp.weight"))?;
        let up_shexp = load_opt(e, src, &p("ffn_up_shexp.weight"))?;
        let down_shexp = load_opt(e, src, &p("ffn_down_shexp.weight"))?;
        // Same law as the selection bias above: a plan that DECLARES an always-on shared expert
        // may not silently run without one. `load_opt` answering None is the correct behaviour
        // for the many MoE arches that have no shared expert at all (OLMoE, vanilla Mixtral) —
        // it is a defect only when the plan says the branch exists. glm5_next, 2026-08-28: the
        // ggml->HF map spelled it SINGULAR (`mlp.shared_expert.*`, qwen3moe) while this
        // checkpoint spells it PLURAL, so all three names resolved to absent tensors and the
        // shared branch was dropped from all 42 MoE layers with no diagnostic.
        if moe.shared.is_some()
            && (gate_shexp.is_none() || up_shexp.is_none() || down_shexp.is_none())
        {
            return Err(format!(
                "layer {il}: the compiled ModelPlan declares an always-on shared expert, but \
                 {}{}{} could not be resolved in the checkpoint. Refusing to load: dropping the \
                 shared branch computes a different model, silently. Either the checkpoint does \
                 not carry it, or this arch's shared-expert spelling is missing from \
                 hf_mapping's ggml->HF map",
                if gate_shexp.is_none() {
                    format!("{} ", p("ffn_gate_shexp.weight"))
                } else {
                    String::new()
                },
                if up_shexp.is_none() {
                    format!("{} ", p("ffn_up_shexp.weight"))
                } else {
                    String::new()
                },
                if down_shexp.is_none() {
                    p("ffn_down_shexp.weight")
                } else {
                    String::new()
                },
            )
            .into());
        }
        Ffn::Moe(MoeWeights {
            gate_inp: load_t(e, src, &p("ffn_gate_inp.weight"))?,
            gate_inp_shexp: load_opt(e, src, &p("ffn_gate_inp_shexp.weight"))?,
            exp_probs_b,
            exp_probs_b_dev,
            active_experts,
            active_experts_dev,
            gate_exps,
            up_exps,
            down_exps,
            gate_shexp,
            up_shexp,
            down_shexp,
            dev_exps,
            step_ep,
            step_tp,
            glm5_ep: None,
            glm5_tp_split: None,
            dev_macros,
            has_macros,
            w4a16_bf16_activations: matches!(
                src.expert_activation_precision(),
                memra_gguf::source::ExpertActivationPrecision::Bf16
            ),
        })
    } else {
        Ffn::Dense {
            ffn_gate: load_t(e, src, &p("ffn_gate.weight"))?,
            ffn_up: load_t(e, src, &p("ffn_up.weight"))?,
            ffn_down: load_t(e, src, &p("ffn_down.weight"))?,
            ffn_down_pqs: load_opt(e, src, &p("ffn_down.pre_quant_scale"))?,
        }
    })
}

fn host_e4m3_bank(
    exps: &HostExps,
) -> Result<crate::tp::E4m3ExpertBank<'_>, Box<dyn std::error::Error>> {
    if exps.qtype != crate::QT_F8_E4M3_BLK {
        return Err(format!(
            "Step EP requires native block-E4M3 expert banks, got qtype {}",
            exps.qtype
        )
        .into());
    }
    let scales = exps
        .fp8_blk
        .as_ref()
        .ok_or("Step EP native expert bank has no block-E4M3 scale plane")?;
    Ok(crate::tp::E4m3ExpertBank {
        codes: exps.bytes.as_bytes(),
        scales: &scales.scales,
        expert_count: exps.n_expert,
        out_features: exps.out_f,
        in_features: exps.in_f,
    })
}

fn validate_step_expert_specs(
    contract: &crate::parallel::ModelParallelContract,
    flag: &str,
    specs: &[crate::tp::StepEpLayerSpec],
    allow_dense_attention_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    for candidate in specs {
        if candidate.layer >= contract.trunk_layers {
            return Err(format!(
                "{flag} layer {} is outside Step trunk layers 0..{}",
                candidate.layer, contract.trunk_layers
            )
            .into());
        }
        if candidate.layer < contract.dense_prefix_layers {
            if allow_dense_attention_only {
                continue;
            }
            return Err(format!(
                "{flag} layer {} is outside Step routed-expert layers {}..{}",
                candidate.layer, contract.dense_prefix_layers, contract.trunk_layers
            )
            .into());
        }
    }
    Ok(())
}

fn validate_step_expert_activation_layout(
    cfg: &ModelConfig,
    flag: &str,
    selection: &StepExpertSelection,
) -> Result<(), Box<dyn std::error::Error>> {
    // step35's routed clamp (min(silu, limit) * clamp(up, +-limit)) is ELEMENTWISE, so the
    // column-sharded TP program preserves it exactly; the expert programs carry the limit
    // through StepTpExps::activation_limit (host oracle: step_expert_activation_host; device:
    // silu_mul_scaled_q8_1_sel_clamp). The historical whole-expert-ownership refusal predated
    // those clamp arms (2026-08-20 lift). E4M3 TP banks still have no clamp arm and refuse.
    let _ = (cfg, flag, selection);
    Ok(())
}

fn parse_auto_w4a16_bf16_mmv(value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(true),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_BF16_MMV={value:?} is invalid under MEMRA_PARALLEL=auto; expected 0 or 1"
        )),
    }
}

fn parse_auto_parallel_tp_attention(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_PARALLEL_TP_ATTENTION={value:?} is invalid; expected 0 or 1"
        )),
    }
}

fn auto_parallel_tp_attention_enabled() -> Result<bool, String> {
    parse_auto_parallel_tp_attention(std::env::var("MEMRA_PARALLEL_TP_ATTENTION").ok().as_deref())
}

fn parse_auto_parallel_tp_attention_ranks(value: Option<&str>) -> Result<Option<usize>, String> {
    match value {
        None => Ok(None),
        Some("2") => Ok(Some(2)),
        Some("3") => Ok(Some(3)),
        Some("4") => Ok(Some(4)),
        Some(value) => Err(format!(
            "MEMRA_PARALLEL_TP_ATTENTION_RANKS={value:?} is invalid; expected 2, 3, or 4"
        )),
    }
}

fn auto_parallel_tp_attention_ranks() -> Result<Option<usize>, String> {
    parse_auto_parallel_tp_attention_ranks(
        std::env::var("MEMRA_PARALLEL_TP_ATTENTION_RANKS")
            .ok()
            .as_deref(),
    )
}

/// Resolve one whole-model placement from the ModelPlan plus exact source census.
///
/// A selected pipeline is persisted into the existing process-level PP configuration before
/// `pp_cuts`, cache allocation, or weight placement reads it. Expert placement is passed directly
/// to the backend registry below. No architecture or layer list participates in this decision.
fn prepare_auto_parallel(
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    plan: &memra_gguf::model_plan::ModelPlan,
) -> Result<Option<crate::parallel::AutoParallelPlacement>, Box<dyn std::error::Error>> {
    let Some(devices) = crate::tp::auto_parallel_devices()? else {
        return Ok(None);
    };
    if std::env::var_os("MEMRA_PP_STAGES").is_some()
        || std::env::var_os("MEMRA_PP_DEVICES").is_some()
        || std::env::var_os("MEMRA_PP_SPLITS").is_some()
    {
        return Err(
            "MEMRA_PARALLEL=auto cannot be combined with MEMRA_PP_STAGES, MEMRA_PP_DEVICES, or \
             MEMRA_PP_SPLITS"
                .into(),
        );
    }
    let placement = crate::parallel::plan_auto_parallel(src, cfg, plan, &devices)?;
    let auto_w4a16_bf16 = placement.backend == crate::parallel::AutoParallelBackend::ExpertParallel
        && matches!(
            src.expert_activation_precision(),
            memra_gguf::source::ExpertActivationPrecision::Bf16
        );
    let bf16_nonexpert = if auto_w4a16_bf16 {
        let explicit = match std::env::var("MEMRA_BF16_MMV") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(format!("cannot read MEMRA_BF16_MMV: {error}").into()),
        };
        let enabled = parse_auto_w4a16_bf16_mmv(explicit.as_deref())?;
        if enabled && explicit.is_none() {
            // SAFETY: automatic placement is resolved before any model tensor loads or
            // `Engine::bf16_mmv_on()` reads the process-level numeric policy.
            unsafe {
                std::env::set_var("MEMRA_BF16_MMV", "1");
            }
        }
        match (enabled, explicit.is_some()) {
            (true, false) => "bf16-resident(auto)",
            (true, true) => "bf16-resident(explicit)",
            (false, true) => "f32-expanded(explicit-rollback)",
            (false, false) => unreachable!("unset auto W4A16 defaults BF16 residency on"),
        }
    } else {
        "placement-default"
    };
    if placement.backend == crate::parallel::AutoParallelBackend::Pipeline {
        let stages = placement.devices.len();
        let device_list = placement
            .devices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let splits = placement
            .pipeline_splits
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        // SAFETY: model loading owns this process-level policy before pp_cuts, transport, cache,
        // or weight placement reads any of these variables.
        unsafe {
            std::env::set_var("MEMRA_PP_STAGES", stages.to_string());
            std::env::set_var("MEMRA_PP_DEVICES", &device_list);
            std::env::set_var("MEMRA_PP_SPLITS", &splits);
        }
    }
    let family = if placement.routed_layers.is_empty() {
        "dense-transformer"
    } else {
        "routed-moe"
    };
    eprintln!(
        "[parallel-auto] family={family} variant={:?} devices={:?} placement={} \
         checkpoint_peak={:.2}GB ep_root={:.2}GB ep_peer={:.2}GB reserve={:.2}GB \
         capacity={:?} splits={:?} bf16_nonexpert={bf16_nonexpert} \
         wavefront=off(default) performance_claim=false",
        cfg.name,
        placement.devices,
        match placement.backend {
            crate::parallel::AutoParallelBackend::Pipeline => "pipeline",
            crate::parallel::AutoParallelBackend::ExpertParallel => "expert-parallel",
        },
        placement.checkpoint_peak_bytes as f64 / 1e9,
        placement.expert_root_bytes as f64 / 1e9,
        placement.expert_peer_bytes as f64 / 1e9,
        placement.reserve_bytes as f64 / 1e9,
        placement.device_capacity_bytes,
        placement.pipeline_splits,
    );
    Ok(Some(placement))
}

fn prepare_step_parallel_load(
    e: &Engine,
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    trunk_layers: usize,
    auto_placement: Option<&crate::parallel::AutoParallelPlacement>,
) -> Result<StepParallelLoadConfig, Box<dyn std::error::Error>> {
    let mut tp_specs = crate::tp::step_tp_layer_specs()?;
    let mut ep_specs = crate::tp::step_ep_layer_specs()?;
    let device_arithmetic = crate::tp::step_ep_device_arithmetic_enabled()?;
    let f32_mirror = crate::tp::step_tp_f32_mirror_enabled()?;
    let bulk_p2p = crate::tp::step_tp_bulk_p2p_enabled()?;
    let mut native_p2p = crate::tp::step_tp_native_p2p_enabled()?;
    let mut nvfp4_device_routes = crate::tp::step_nvfp4_dev_routes_enabled()?;
    let auto_tp_attention = auto_parallel_tp_attention_enabled()?;
    let requested_attention_ranks = auto_parallel_tp_attention_ranks()?;
    let mut auto_parallel = false;
    let mut tp_attention_expert_overlap = false;
    if requested_attention_ranks.is_some() && !auto_tp_attention {
        return Err(
            "MEMRA_PARALLEL_TP_ATTENTION_RANKS requires MEMRA_PARALLEL_TP_ATTENTION=1".into(),
        );
    }
    if auto_tp_attention && auto_placement.is_none() {
        return Err(
            "MEMRA_PARALLEL_TP_ATTENTION=1 requires MEMRA_PARALLEL=auto; explicit per-layer \
             recipes remain under MEMRA_STEP_TP"
                .into(),
        );
    }
    if let Some(placement) = auto_placement {
        if !tp_specs.is_empty() || !ep_specs.is_empty() {
            return Err(
                "MEMRA_PARALLEL=auto cannot be combined with MEMRA_STEP_EP or MEMRA_STEP_TP".into(),
            );
        }
        if placement.backend == crate::parallel::AutoParallelBackend::Pipeline {
            if auto_tp_attention {
                return Err(
                    "MEMRA_PARALLEL_TP_ATTENTION=1 requires automatic whole-expert EP; the \
                     selected checkpoint fits only the pipeline backend"
                        .into(),
                );
            }
            return Ok(StepParallelLoadConfig::default());
        }
        if auto_tp_attention {
            let contract = crate::parallel::ModelParallelContract::from_model(cfg)?;
            if !contract.tensor_attention_supported {
                return Err(format!(
                    "MEMRA_PARALLEL_TP_ATTENTION=1 cannot shard attention for {:?}: the \
                     compiled ModelPlan has no generic tensor-attention contract",
                    cfg.name
                )
                .into());
            }
            let attention_ranks = requested_attention_ranks.unwrap_or(placement.devices.len());
            if attention_ranks > placement.devices.len() {
                return Err(format!(
                    "MEMRA_PARALLEL_TP_ATTENTION_RANKS={attention_ranks} exceeds the automatic \
                     placement width {}",
                    placement.devices.len()
                )
                .into());
            }
            let attention_devices = placement.devices[..attention_ranks].to_vec();
            tp_specs = (0..trunk_layers)
                .map(|layer| crate::tp::StepTpLayerSpec {
                    layer,
                    devices: attention_devices.clone(),
                })
                .collect();
            if attention_ranks < placement.devices.len() {
                ep_specs = placement
                    .routed_layers
                    .iter()
                    .map(|&layer| crate::tp::StepEpLayerSpec {
                        layer,
                        devices: placement.devices.clone(),
                    })
                    .collect();
                tp_attention_expert_overlap = true;
            } else {
                ep_specs.clear();
            }
        } else {
            ep_specs = placement
                .routed_layers
                .iter()
                .map(|&layer| crate::tp::StepEpLayerSpec {
                    layer,
                    devices: placement.devices.clone(),
                })
                .collect();
        }
        auto_parallel = true;
        native_p2p = true;
        nvfp4_device_routes = matches!(
            src.expert_activation_precision(),
            memra_gguf::source::ExpertActivationPrecision::Bf16
        );
        eprintln!(
            "[parallel-auto-backend] devices={:?} routed_layers={} native_p2p=true \
             artifact_activation={:?} attention_layout={} attention_devices={:?} \
             expert_layout=expert-parallel expert_devices={:?} \
             backend={} performance_claim=false",
            placement.devices,
            placement.routed_layers.len(),
            src.expert_activation_precision(),
            if auto_tp_attention {
                "tensor-parallel"
            } else {
                "root-local"
            },
            tp_specs
                .first()
                .map(|spec| spec.devices.as_slice())
                .unwrap_or(&[]),
            ep_specs
                .first()
                .map(|spec| spec.devices.as_slice())
                .unwrap_or(placement.devices.as_slice()),
            if nvfp4_device_routes {
                "nvfp4-w4a16"
            } else {
                "artifact-selected-host-oracle"
            },
        );
    }
    if tp_specs.is_empty() {
        if auto_tp_attention {
            return Err("MEMRA_PARALLEL_TP_ATTENTION=1 produced no tensor-parallel layers".into());
        }
        if device_arithmetic || f32_mirror || bulk_p2p {
            return Err(
                "MEMRA_STEP_EP_DEVICE_ARITHMETIC=1, MEMRA_STEP_TP_F32_MIRROR=1, or \
                 MEMRA_STEP_TP_BULK_P2P=1 requires MEMRA_STEP_TP; device arithmetic and bulk \
                 transport also require MEMRA_STEP_TP_NATIVE_P2P=1"
                    .into(),
            );
        }
        if nvfp4_device_routes && ep_specs.is_empty() {
            return Err(
                "MEMRA_STEP_NVFP4_DEV_ROUTES=1 requires MEMRA_STEP_EP or MEMRA_STEP_TP".into(),
            );
        }
        if nvfp4_device_routes && !native_p2p {
            return Err("MEMRA_STEP_NVFP4_DEV_ROUTES=1 with explicit EP requires \
                 MEMRA_STEP_TP_NATIVE_P2P=1"
                .into());
        }
        // Pure-EP configs still need the artifact census: the EP bank build dispatches on it,
        // and defaulting to E4M3 refuses an NVFP4 checkpoint at load ("got qtype 7").
        let expert_artifact = if ep_specs.is_empty() {
            StepExpertArtifact::default()
        } else if nvfp4_device_routes
            && matches!(
                src.expert_activation_precision(),
                memra_gguf::source::ExpertActivationPrecision::Bf16
            )
        {
            // The physical checkpoint may store one tensor per expert or one stacked bank.
            // HostExps normalizes both to the canonical block_nvfp4 layout. Automatic and
            // explicit W4A16 device routes validate that normalized bank at layer load instead
            // of assuming one physical source packing here.
            StepExpertArtifact::Nvfp4
        } else {
            let contract = crate::parallel::ModelParallelContract::from_model(cfg)?;
            validate_step_expert_specs(&contract, "MEMRA_STEP_EP", &ep_specs, false)?;
            let layer_owners = (0..trunk_layers)
                .map(|layer| {
                    crate::pp::layer_engine(e, trunk_layers, layer)
                        .map(|engine| engine.ctx().ordinal())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut runtime_groups = Vec::<Vec<usize>>::new();
            for spec in &ep_specs {
                let owner = layer_owners[spec.layer];
                if !spec.devices.contains(&owner) {
                    return Err(format!(
                        "MEMRA_STEP_EP layer {} owning device {owner} is absent from {:?}",
                        spec.layer, spec.devices
                    )
                    .into());
                }
                if nvfp4_device_routes && spec.devices.first().copied() != Some(owner) {
                    return Err(format!(
                        "MEMRA_STEP_NVFP4_DEV_ROUTES=1 requires the owning device first; \
                         layer {} owner={owner} devices={:?}",
                        spec.layer, spec.devices
                    )
                    .into());
                }
                if !runtime_groups.contains(&spec.devices) {
                    runtime_groups.push(spec.devices.clone());
                }
            }
            for devices in &runtime_groups {
                let hardware = crate::parallel::detect_uniform_hardware(devices)?;
                if !contract.hardware_targets.contains(&hardware) {
                    return Err(format!(
                        "{} has no qualified {hardware:?} EP contract for devices {devices:?}",
                        contract.variant
                    )
                    .into());
                }
            }
            let artifact = match crate::parallel::validate_fp8_expert_checkpoint(src, &contract) {
                Ok(_) => StepExpertArtifact::E4m3,
                Err(fp8_error) => {
                    match crate::parallel::validate_nvfp4_expert_checkpoint(src, &contract) {
                        Ok(_) => StepExpertArtifact::Nvfp4,
                        Err(nvfp4_error) => {
                            return Err(format!(
                                "Step checkpoint qualifies as neither native expert artifact \
                                 class: [E4M3] {fp8_error} [NVFP4] {nvfp4_error}"
                            )
                            .into());
                        }
                    }
                }
            };
            if nvfp4_device_routes && artifact != StepExpertArtifact::Nvfp4 {
                return Err(
                    "MEMRA_STEP_NVFP4_DEV_ROUTES=1 requires a native ModelOpt NVFP4 expert \
                     artifact"
                        .into(),
                );
            }
            artifact
        };
        return Ok(StepParallelLoadConfig {
            ep_specs,
            native_p2p,
            nvfp4_device_routes,
            auto_parallel,
            expert_artifact,
            ..StepParallelLoadConfig::default()
        });
    }
    let contract = crate::parallel::ModelParallelContract::from_model(cfg)?;
    validate_step_expert_specs(&contract, "MEMRA_STEP_EP", &ep_specs, false)?;
    validate_step_expert_specs(&contract, "MEMRA_STEP_TP", &tp_specs, true)?;
    for spec in &tp_specs {
        let selection = select_step_expert_layout_inner(
            spec.layer,
            &ep_specs,
            &tp_specs,
            tp_attention_expert_overlap,
        )?
        .ok_or("Step TP expert selection disappeared during preflight")?;
        validate_step_expert_activation_layout(cfg, "MEMRA_STEP_TP", &selection)?;
    }

    let layer_owners = (0..trunk_layers)
        .map(|layer| {
            crate::pp::layer_engine(e, trunk_layers, layer).map(|engine| engine.ctx().ordinal())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plan = contract.preflight_step_tp_specs(
        tp_specs
            .iter()
            .map(|spec| (spec.layer, spec.devices.as_slice())),
        &layer_owners,
    )?;

    for devices in &plan.runtime_groups {
        let hardware = crate::parallel::detect_uniform_hardware(devices)?;
        if !contract.hardware_targets.contains(&hardware) {
            return Err(format!(
                "{} has no qualified {hardware:?} TP contract for devices {devices:?}",
                contract.variant
            )
            .into());
        }
    }

    if bulk_p2p && !native_p2p {
        return Err("MEMRA_STEP_TP_BULK_P2P=1 requires MEMRA_STEP_TP_NATIVE_P2P=1".into());
    }
    if device_arithmetic
        && (!ep_specs.is_empty()
            || !native_p2p
            || plan.expert_parallel_layers() == 0
            || plan.tensor_parallel_expert_layers() != 0)
    {
        return Err(
            "MEMRA_STEP_EP_DEVICE_ARITHMETIC=1 requires native-P2P TP4/TP8 \
             expert ownership for every selected routed-expert layer"
                .into(),
        );
    }
    // Census dispatch: one checkpoint is exactly one native expert artifact class. FP8 first
    // (the historical contract), NVFP4 as the fallback census; if neither qualifies, surface
    // BOTH refusals so the operator sees which contract each class failed.
    let (qualified_experts, expert_artifact) =
        match crate::parallel::validate_fp8_expert_checkpoint(src, &contract) {
            Ok(qualified) => (qualified, StepExpertArtifact::E4m3),
            Err(fp8_error) => {
                match crate::parallel::validate_nvfp4_expert_checkpoint(src, &contract) {
                    Ok(qualified) => (qualified, StepExpertArtifact::Nvfp4),
                    Err(nvfp4_error) => {
                        return Err(format!(
                            "Step checkpoint qualifies as neither native expert artifact class: \
                         [E4M3] {fp8_error} [NVFP4] {nvfp4_error}"
                        )
                        .into());
                    }
                }
            }
        };
    if expert_artifact == StepExpertArtifact::Nvfp4 {
        if device_arithmetic {
            return Err(
                "MEMRA_STEP_EP_DEVICE_ARITHMETIC=1 is qualified for the E4M3 expert artifact \
                 only; the NVFP4 expert program is host-canonical in this increment"
                    .into(),
            );
        }
        // f32_mirror is NOT refused here: it changes only the BF16 TP attention projections'
        // residency (load-time F32 expansion, same cuBLASLt values and shapes), which are the
        // same code path under both expert artifact classes. The per-call bf16_to_f32 expansion
        // it removes measured 595us/layer of QKV wall on the NVFP4 TP2 decode lane (2026-08-20).
        if bulk_p2p {
            return Err(
                "MEMRA_STEP_TP_BULK_P2P=1 is qualified for the E4M3 expert artifact only; the \
                 NVFP4 bank transport increment has not landed"
                    .into(),
            );
        }
    }

    if f32_mirror {
        eprintln!(
            "[step-tp-preflight] layers={} full_trunk={} runtime_groups={} \
             dense_attention_layers={} tensor_expert_layers={} expert_owner_layers={} \
             qualified_fp8_expert_projection_slices={} owner_first=true \
             hardware=rtx-pro-6000-blackwell \
             native_p2p={} bulk_p2p={} device_arithmetic={} bf16_residency=f32-mirror \
             weights_loaded=false performance_claim=false",
            plan.layers.len(),
            plan.full_trunk,
            plan.runtime_groups.len(),
            plan.dense_attention_layers(),
            plan.tensor_parallel_expert_layers(),
            plan.expert_parallel_layers(),
            qualified_experts,
            native_p2p,
            bulk_p2p,
            device_arithmetic,
        );
    } else {
        eprintln!(
            "[step-tp-preflight] layers={} full_trunk={} runtime_groups={} \
             dense_attention_layers={} tensor_expert_layers={} expert_owner_layers={} \
             qualified_fp8_expert_projection_slices={} owner_first=true \
             hardware=rtx-pro-6000-blackwell \
             native_p2p={} bulk_p2p={} device_arithmetic={} \
             weights_loaded=false performance_claim=false",
            plan.layers.len(),
            plan.full_trunk,
            plan.runtime_groups.len(),
            plan.dense_attention_layers(),
            plan.tensor_parallel_expert_layers(),
            plan.expert_parallel_layers(),
            qualified_experts,
            native_p2p,
            bulk_p2p,
            device_arithmetic,
        );
    }
    Ok(StepParallelLoadConfig {
        ep_specs,
        tp_specs,
        native_p2p,
        ep_device_arithmetic: device_arithmetic,
        f32_mirror,
        bulk_p2p,
        nvfp4_device_routes,
        auto_parallel,
        tp_attention_expert_overlap,
        expert_artifact,
    })
}

/// Resolve one routed projection's stacked NVFP4 native bank from the checkpoint source.
fn nvfp4_native_expert_bank<'a>(
    src: &'a dyn TensorSource,
    layer: usize,
    proj: &str,
) -> Result<memra_gguf::source::Nvfp4StackedNative<'a>, Box<dyn std::error::Error>> {
    let name = format!("blk.{layer}.ffn_{proj}_exps.weight");
    src.find_nvfp4_stacked_native(&name)
        .ok_or_else(|| format!("NVFP4 expert backend is missing native bank {name}").into())
}

/// Borrow a `Nvfp4StackedNative` as the TP program's bank view.
fn nvfp4_expert_bank_view<'a>(
    native: &'a memra_gguf::source::Nvfp4StackedNative<'a>,
) -> crate::tp::Nvfp4ExpertBank<'a> {
    crate::tp::Nvfp4ExpertBank {
        codes: native.codes,
        scales: native.scales,
        macros: &native.macros,
        expert_count: native.n_expert,
        out_features: native.out_f,
        in_features: native.in_f,
    }
}

#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
fn build_step_distributed_exps(
    e: &Engine,
    cfg: &ModelConfig,
    src: &dyn TensorSource,
    layer: usize,
    gate: &HostExps,
    up: &HostExps,
    down: &HostExps,
    step_runtimes: &mut StepParallelRuntimeRegistry,
) -> Result<(Option<StepEpExps>, Option<StepTpExps>), Box<dyn std::error::Error>> {
    let ep_device_arithmetic = step_runtimes.config.ep_device_arithmetic;
    if step_runtimes.config.ep_specs.is_empty() && step_runtimes.config.tp_specs.is_empty() {
        if ep_device_arithmetic {
            return Err(
                "MEMRA_STEP_EP_DEVICE_ARITHMETIC=1 requires MEMRA_STEP_TP and \
                 MEMRA_STEP_TP_NATIVE_P2P=1"
                    .into(),
            );
        }
        return Ok((None, None));
    }
    let contract = crate::parallel::ModelParallelContract::from_model(cfg)?;
    validate_step_expert_specs(
        &contract,
        "MEMRA_STEP_EP",
        &step_runtimes.config.ep_specs,
        false,
    )?;
    validate_step_expert_specs(
        &contract,
        "MEMRA_STEP_TP",
        &step_runtimes.config.tp_specs,
        true,
    )?;
    let Some(selection) = step_runtimes.expert_selection(layer)? else {
        return Ok((None, None));
    };
    validate_step_expert_activation_layout(
        cfg,
        if selection.configured_by_tp {
            "MEMRA_STEP_TP"
        } else {
            "MEMRA_STEP_EP"
        },
        &selection,
    )?;
    // MEMRA_STEP_EP/TP expert kernels encode step35's POST clamp end to end (upload banks,
    // grouped-decode projection, the `[step-ep-clamp] formula=min-silu-times-clamped-up`
    // receipt). glm5_next's PRE form has no arm here — refuse by name rather than route it
    // through a POST epilogue.
    let activation_limit = match cfg.clamp_exp_at(layer as u32) {
        None => None,
        Some(SwigluClamp::Post(l)) => Some(l),
        Some(SwigluClamp::Pre(_)) => {
            return Err(format!(
                "MEMRA_STEP_EP/TP layer {layer}: glm5_next PRE-clamped SwiGLU has no \
                 expert-parallel arm (the banks encode step35's post-clamp form)"
            )
            .into());
        }
    };
    let owner = e.ctx().ordinal();
    if !selection.spec.devices.contains(&owner) {
        let flag = if selection.configured_by_tp {
            "MEMRA_STEP_TP"
        } else {
            "MEMRA_STEP_EP"
        };
        return Err(format!(
            "{flag} layer {layer} owning PP device {owner} is absent from rank devices {:?}",
            selection.spec.devices
        )
        .into());
    }
    let expert_parallel = selection.layout == StepExpertLayout::ExpertParallel;
    if selection.configured_by_tp {
        contract.plan(crate::parallel::TopologyRequest {
            pipeline: 1,
            tensor: selection.spec.devices.len(),
            expert_parallel,
            available_devices: selection.spec.devices.len(),
            hardware: crate::parallel::HardwareTarget::RtxPro6000Blackwell,
        })?;
    }
    let native_p2p = selection.configured_by_tp && step_runtimes.config.native_p2p;
    if ep_device_arithmetic
        && (!selection.configured_by_tp
            || selection.layout != StepExpertLayout::ExpertParallel
            || !native_p2p)
    {
        return Err(
            "MEMRA_STEP_EP_DEVICE_ARITHMETIC=1 requires a MEMRA_STEP_TP TP4/TP8 \
             expert-owner layer and MEMRA_STEP_TP_NATIVE_P2P=1"
                .into(),
        );
    }
    let expert_artifact = step_runtimes.config.expert_artifact;
    match selection.layout {
        StepExpertLayout::ExpertParallel => {
            if expert_artifact == StepExpertArtifact::Nvfp4 {
                // TP4/TP8 plans use whole-expert ownership for routed MoE layers, so the
                // W4A16 device-routed EP program is valid there too. `configured_by_tp` names
                // the surrounding attention plan; it does not change the expert-bank layout.
                let w4a16_device_routes = step_runtimes.config.nvfp4_device_routes;
                if w4a16_device_routes
                    && !matches!(
                        src.expert_activation_precision(),
                        memra_gguf::source::ExpertActivationPrecision::Bf16
                    )
                {
                    return Err(
                        "explicit-EP MEMRA_STEP_NVFP4_DEV_ROUTES=1 requires an artifact that \
                         declares BF16 routed-expert activations; TP keeps its separately gated \
                         quantized-activation path"
                            .into(),
                    );
                }
                // Request the runtime with the immutable config's transport choice. The default
                // host-canonical program ignores native P2P, while the W4A16 decode door consumes
                // it for device input/output. Sharing the same runtime also avoids the measured
                // third-context flake when TP and EP coexist.
                let runtime = step_runtimes.runtime(
                    &selection.spec.devices,
                    step_runtimes.config.native_p2p,
                    false,
                )?;
                let experts = runtime.upload_expert_parallel_nvfp4_normalized(gate, up, down)?;
                let marker = if step_runtimes.config.auto_parallel {
                    "parallel-ep"
                } else {
                    "step-ep"
                };
                eprintln!(
                    "[{marker}] layer={layer} devices={:?} experts={} artifact=nvfp4 \
                     expert_layout=expert-parallel expert_transport={} \
                     macro_fold=post-kernel-once native_p2p={} w4a16_device_routes={} \
                     performance_claim=false",
                    selection.spec.devices,
                    contract.expert_count,
                    runtime.transport_label(),
                    runtime.native_p2p(),
                    w4a16_device_routes,
                );
                if let Some(limit) = activation_limit {
                    eprintln!(
                        "[step-ep-clamp] load layer={layer} routed_clamp={limit} \
                         formula=min-silu-times-clamped-up performance_claim=false"
                    );
                }
                return Ok((
                    Some(StepEpExps {
                        runtime,
                        experts: StepEpExpertBank::Nvfp4(experts),
                        devices: selection.spec.devices,
                        configured_by_tp: selection.configured_by_tp,
                        activation_limit,
                        nvfp4_device_routes: w4a16_device_routes,
                        grouped_decode: None,
                    }),
                    None,
                ));
            }
            let runtime =
                step_runtimes.runtime(&selection.spec.devices, native_p2p, ep_device_arithmetic)?;
            let experts = runtime.upload_expert_parallel(
                host_e4m3_bank(gate)?,
                host_e4m3_bank(up)?,
                host_e4m3_bank(down)?,
            )?;
            let grouped_decode = if ep_device_arithmetic {
                let tokens = 1;
                let selected = (0..contract.experts_per_token).collect::<Vec<_>>();
                let input = vec![0.0f32; contract.hidden_size];
                let route_weights = vec![1.0f32; contract.experts_per_token];
                let projection = runtime.prepare_step_grouped_expert_parallel_gate_with_capacity(
                    &experts,
                    &input,
                    tokens,
                    &selected,
                    activation_limit,
                    tokens,
                )?;
                let combine = runtime
                    .prepare_step_grouped_expert_parallel_combine(&projection, &route_weights)?;
                Some(std::sync::Mutex::new(StepEpGroupedDecode {
                    projection,
                    combine,
                }))
            } else {
                None
            };
            if selection.configured_by_tp {
                eprintln!(
                    "[step-tp-ep] layer={layer} devices={:?} experts={} tp={} \
                     attention_layout=tensor-parallel expert_layout=expert-parallel \
                     expert_transport={} tp_transport={} native_p2p={} \
                     activation={} accumulation={} output={} \
                     grouped_decode_prepared={} grouped_decode_capacity=1 \
                     performance_claim=false",
                    selection.spec.devices,
                    contract.expert_count,
                    selection.spec.devices.len(),
                    runtime.transport_label(),
                    runtime.transport_label(),
                    runtime.native_p2p(),
                    runtime.expert_activation_label(),
                    runtime.expert_accumulation_label(),
                    runtime.expert_output_label(),
                    grouped_decode.is_some(),
                );
            } else {
                eprintln!(
                    "[step-ep] layer={layer} devices={:?} experts={} \
                     expert_layout=expert-parallel expert_transport=host-bounce \
                     native_p2p=false performance_claim=false",
                    selection.spec.devices, contract.expert_count
                );
            }
            if let Some(limit) = activation_limit {
                eprintln!(
                    "[step-ep-clamp] load layer={layer} routed_clamp={limit} \
                     formula=min-silu-times-clamped-up performance_claim=false"
                );
            }
            Ok((
                Some(StepEpExps {
                    runtime,
                    experts: StepEpExpertBank::E4m3(experts),
                    devices: selection.spec.devices,
                    configured_by_tp: selection.configured_by_tp,
                    activation_limit,
                    nvfp4_device_routes: false,
                    grouped_decode,
                }),
                None,
            ))
        }
        StepExpertLayout::TensorParallel => {
            let runtime =
                step_runtimes.runtime(&selection.spec.devices, native_p2p, ep_device_arithmetic)?;
            if activation_limit.is_some() && expert_artifact == StepExpertArtifact::E4m3 {
                return Err(format!(
                    "layer {layer} uses the routed SwiGLU clamp and the E4M3 TP expert \
                     program has no clamp arm; select EP for this layer (the NVFP4 TP \
                     program carries the clamp)"
                )
                .into());
            }
            let experts = if expert_artifact == StepExpertArtifact::Nvfp4 {
                let gate_native = nvfp4_native_expert_bank(src, layer, "gate")?;
                let up_native = nvfp4_native_expert_bank(src, layer, "up")?;
                let down_native = nvfp4_native_expert_bank(src, layer, "down")?;
                StepTpExpertBank::Nvfp4(runtime.upload_tensor_parallel_nvfp4(
                    nvfp4_expert_bank_view(&gate_native),
                    nvfp4_expert_bank_view(&up_native),
                    nvfp4_expert_bank_view(&down_native),
                )?)
            } else {
                StepTpExpertBank::E4m3(runtime.upload_tensor_parallel(
                    host_e4m3_bank(gate)?,
                    host_e4m3_bank(up)?,
                    host_e4m3_bank(down)?,
                )?)
            };
            eprintln!(
                "[step-tp] layer={layer} devices={:?} experts={} tp={} artifact={} \
                 expert_layout=tensor-parallel transport={} native_p2p={} \
                 performance_claim=false",
                selection.spec.devices,
                contract.expert_count,
                selection.spec.devices.len(),
                match expert_artifact {
                    StepExpertArtifact::E4m3 => "e4m3",
                    StepExpertArtifact::Nvfp4 => "nvfp4",
                },
                runtime.transport_label(),
                runtime.native_p2p(),
            );
            if let Some(limit) = activation_limit {
                eprintln!(
                    "[step-tp-clamp] load layer={layer} routed_clamp={limit} \
                     formula=min-silu-times-clamped-up performance_claim=false"
                );
            }
            Ok((
                None,
                Some(StepTpExps {
                    runtime,
                    experts,
                    devices: selection.spec.devices,
                    activation_limit,
                }),
            ))
        }
    }
}

fn upload_step_bf16_column(
    runtime: &crate::tp::TpE4m3HostBounce,
    src: &dyn TensorSource,
    name: &str,
    expected_in: usize,
    expected_out: usize,
    f32_mirror: bool,
) -> Result<crate::tp::ResidentBf16ColumnParallel, Box<dyn std::error::Error>> {
    let tensor = src
        .find(name)
        .ok_or_else(|| format!("Step TP projection is missing {name}"))?;
    if tensor.ggml_type != GgmlType::BF16 {
        return Err(format!(
            "Step TP projection {name} must preserve checkpoint BF16 bytes, got {:?}",
            tensor.ggml_type
        )
        .into());
    }
    if tensor.ne.len() != 2 {
        return Err(format!(
            "Step TP projection {name} must be a 2-D matrix, got shape {:?}",
            tensor.ne
        )
        .into());
    }
    let matrix = crate::tp::Bf16Matrix {
        bytes: tensor.bytes.as_ref(),
        in_features: tensor.ne[0] as usize,
        out_features: tensor.ne[1] as usize,
    };
    matrix.validate()?;
    if matrix.in_features != expected_in || matrix.out_features != expected_out {
        return Err(format!(
            "Step TP projection {name} shape {}x{} != registered {expected_out}x{expected_in}",
            matrix.out_features, matrix.in_features
        )
        .into());
    }
    Ok(if f32_mirror {
        runtime.upload_step_bf16_column_parallel_f32_mirror(matrix)?
    } else {
        runtime.upload_step_bf16_column_parallel(matrix)?
    })
}

fn upload_step_bf16_row(
    runtime: &crate::tp::TpE4m3HostBounce,
    src: &dyn TensorSource,
    name: &str,
    expected_in: usize,
    expected_out: usize,
    f32_mirror: bool,
) -> Result<crate::tp::ResidentStepBf16RowParallel, Box<dyn std::error::Error>> {
    let tensor = src
        .find(name)
        .ok_or_else(|| format!("Step TP projection is missing {name}"))?;
    if tensor.ggml_type != GgmlType::BF16 {
        return Err(format!(
            "Step TP projection {name} must preserve checkpoint BF16 bytes, got {:?}",
            tensor.ggml_type
        )
        .into());
    }
    if tensor.ne.len() != 2 {
        return Err(format!(
            "Step TP projection {name} must be a 2-D matrix, got shape {:?}",
            tensor.ne
        )
        .into());
    }
    let matrix = crate::tp::Bf16Matrix {
        bytes: tensor.bytes.as_ref(),
        in_features: tensor.ne[0] as usize,
        out_features: tensor.ne[1] as usize,
    };
    matrix.validate()?;
    if matrix.in_features != expected_in || matrix.out_features != expected_out {
        return Err(format!(
            "Step TP projection {name} shape {}x{} != registered {expected_out}x{expected_in}",
            matrix.out_features, matrix.in_features
        )
        .into());
    }
    Ok(if f32_mirror {
        runtime.upload_step_bf16_row_parallel_f32_mirror(matrix)?
    } else {
        runtime.upload_step_bf16_row_parallel(matrix)?
    })
}

fn upload_step_tp_f32_copies(
    runtime: &crate::tp::TpE4m3HostBounce,
    src: &dyn TensorSource,
    name: &str,
    expected: usize,
) -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
    let tensor = src
        .find(name)
        .ok_or_else(|| format!("Step TP attention is missing {name}"))?;
    let values = memra_gguf::dequant::dequantize(
        tensor.ggml_type,
        &tensor.bytes,
        tensor.ne.iter().product::<u64>() as usize,
    );
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "Step TP attention {name} has {} finite values, expected {expected}",
            values.len()
        )
        .into());
    }
    let mut copies = Vec::with_capacity(runtime.devices().len());
    for rank in 0..runtime.devices().len() {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("Step TP attention has no engine for rank {rank}"))?;
        let _main = engine.gpu.enter_main()?;
        copies.push(engine.htod(&values)?);
    }
    Ok(copies)
}

/// Upload one [rows, cols] f32-expanded tensor as per-rank ROW shards (rank r holds rows
/// [r*rows/world, (r+1)*rows/world)). The v2 fused QKV+gate kernel consumes rank-local gate
/// weight rows so the per-layer gate matmul on the model engine (and its staging copies)
/// disappears under MEMRA_STEP_TP_QKV_FUSED.
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn upload_step_tp_f32_row_shards(
    runtime: &crate::tp::TpE4m3HostBounce,
    src: &dyn TensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
    let tensor = src
        .find(name)
        .ok_or_else(|| format!("Step TP attention is missing {name}"))?;
    let values = memra_gguf::dequant::dequantize(
        tensor.ggml_type,
        &tensor.bytes,
        tensor.ne.iter().product::<u64>() as usize,
    );
    let world = runtime.devices().len();
    if values.len() != rows * cols || rows % world != 0 || values.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "Step TP attention {name} has {} finite values, expected {rows}x{cols} \
             (rows divisible by world {world})",
            values.len()
        )
        .into());
    }
    let local_rows = rows / world;
    let mut shards = Vec::with_capacity(world);
    for rank in 0..world {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("Step TP attention has no engine for rank {rank}"))?;
        let _main = engine.gpu.enter_main()?;
        shards
            .push(engine.htod(&values[rank * local_rows * cols..(rank + 1) * local_rows * cols])?);
    }
    Ok(shards)
}

/// BF16 twin of `upload_step_tp_f32_row_shards`: raw checkpoint bytes, row shards per rank.
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn upload_step_tp_bf16_row_shards(
    runtime: &crate::tp::TpE4m3HostBounce,
    src: &dyn TensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<CudaSlice<u8>>, Box<dyn std::error::Error>> {
    let tensor = src
        .find(name)
        .ok_or_else(|| format!("Step TP attention is missing {name}"))?;
    if tensor.ggml_type != memra_gguf::GgmlType::BF16 || tensor.bytes.len() != rows * cols * 2 {
        return Err(format!(
            "Step TP attention {name} is not a bf16 [{rows}, {cols}] tensor ({} bytes, {:?})",
            tensor.bytes.len(),
            tensor.ggml_type
        )
        .into());
    }
    let world = runtime.devices().len();
    if rows % world != 0 {
        return Err(format!("{name} rows {rows} not divisible by world {world}").into());
    }
    let local = rows / world * cols * 2;
    let mut shards = Vec::with_capacity(world);
    for rank in 0..world {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("Step TP attention has no engine for rank {rank}"))?;
        let _main = engine.gpu.enter_main()?;
        shards.push(engine.htod_bytes(&tensor.bytes[rank * local..(rank + 1) * local])?);
    }
    Ok(shards)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepTpAttentionPlacement {
    RankLocalGlobal,
    RankLocalSwa,
    OwnerSwa,
    OwnerTransportFallback,
}

impl StepTpAttentionPlacement {
    fn resolve(native_p2p: bool, window: Option<u32>) -> Self {
        match (native_p2p, window.is_some()) {
            (true, true) => Self::RankLocalSwa,
            (false, true) => Self::OwnerSwa,
            (true, false) => Self::RankLocalGlobal,
            (false, false) => Self::OwnerTransportFallback,
        }
    }

    fn is_rank_local(self) -> bool {
        matches!(self, Self::RankLocalGlobal | Self::RankLocalSwa)
    }

    fn label(self) -> &'static str {
        match self {
            Self::RankLocalGlobal => "rank-local-global",
            Self::RankLocalSwa => "rank-local-swa-ring",
            Self::OwnerSwa => "owner-swa",
            Self::OwnerTransportFallback => "owner-transport-fallback",
        }
    }
}

fn build_step_tp_qkv(
    e: &Engine,
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    layer: usize,
    step_runtimes: &mut StepParallelRuntimeRegistry,
) -> Result<Option<StepTpQkv>, Box<dyn std::error::Error>> {
    let Some(spec) = step_runtimes.tp_spec(layer).cloned() else {
        return Ok(None);
    };
    let contract = crate::parallel::ModelParallelContract::from_model(cfg)?;
    if layer >= contract.trunk_layers {
        return Err(format!(
            "MEMRA_STEP_TP layer {layer} is outside Step trunk layers 0..{}",
            contract.trunk_layers
        )
        .into());
    }
    let owner = e.ctx().ordinal();
    if spec.devices.first().copied() != Some(owner) {
        return Err(format!(
            "MEMRA_STEP_TP layer {layer} owning PP device {owner} must be the first QKV rank, \
             got {:?}",
            spec.devices
        )
        .into());
    }
    let plan = contract.plan(crate::parallel::TopologyRequest {
        pipeline: 1,
        tensor: spec.devices.len(),
        expert_parallel: spec.devices.len() > 2,
        available_devices: spec.devices.len(),
        hardware: crate::parallel::HardwareTarget::RtxPro6000Blackwell,
    })?;
    for rank in 0..spec.devices.len() {
        plan.query_head_range(layer, rank).ok_or_else(|| {
            format!("Step TP layer {layer} has no query-head range for rank {rank}")
        })?;
        plan.kv_head_range(layer, rank)
            .ok_or_else(|| format!("Step TP layer {layer} has no KV-head range for rank {rank}"))?;
    }
    let native_p2p = step_runtimes.config.native_p2p;
    let ep_device_arithmetic = step_runtimes.config.ep_device_arithmetic;
    let f32_mirror = step_runtimes.config.f32_mirror;
    if ep_device_arithmetic && (!native_p2p || !matches!(spec.devices.len(), 4 | 8)) {
        return Err(
            "MEMRA_STEP_EP_DEVICE_ARITHMETIC=1 requires a MEMRA_STEP_TP TP4/TP8 \
             expert-owner layer and MEMRA_STEP_TP_NATIVE_P2P=1"
                .into(),
        );
    }
    let runtime = step_runtimes.runtime(&spec.devices, native_p2p, ep_device_arithmetic)?;
    let p = |suffix: &str| format!("blk.{layer}.{suffix}");
    let q = upload_step_bf16_column(
        &runtime,
        src,
        &p("attn_q.weight"),
        contract.hidden_size,
        contract.query_heads[layer] * contract.head_dim,
        f32_mirror,
    )?;
    let k = upload_step_bf16_column(
        &runtime,
        src,
        &p("attn_k.weight"),
        contract.hidden_size,
        contract.kv_heads[layer] * contract.head_dim,
        f32_mirror,
    )?;
    let v = upload_step_bf16_column(
        &runtime,
        src,
        &p("attn_v.weight"),
        contract.hidden_size,
        contract.kv_heads[layer] * contract.head_dim,
        f32_mirror,
    )?;
    let o = upload_step_bf16_row(
        &runtime,
        src,
        &p("attn_output.weight"),
        contract.query_heads[layer] * contract.head_dim,
        contract.hidden_size,
        f32_mirror,
    )?;
    let geometry = cfg.full_attention_geometry_at(layer as u32);
    let attention_placement =
        StepTpAttentionPlacement::resolve(runtime.native_p2p(), geometry.window);
    let attention = if attention_placement.is_rank_local() {
        // The v2 decode driver replicates the layer input on-device (evented, no host
        // round-trip), so it needs the same persistent replicated rows the FP8
        // device-arithmetic door uses. Configs with both doors off keep None and the v1
        // host-replicated arm, byte-stable with prior receipts.
        let decode_input = if ep_device_arithmetic || crate::tp::step_tp_decode_v2_enabled()? {
            Some(std::sync::Mutex::new(
                runtime.allocate_replicated_device_rows(1, contract.hidden_size)?,
            ))
        } else {
            None
        };
        // Gate row shards only load when the fused door will consume them: they duplicate
        // (rank-locally) a weight the owning-stage fallback also holds.
        let gate_fused =
            crate::tp::step_tp_qkv_fused_enabled()? && src.find(&p("attn_gate.weight")).is_some();
        let gate_shards = if gate_fused && f32_mirror {
            Some(upload_step_tp_f32_row_shards(
                &runtime,
                src,
                &p("attn_gate.weight"),
                contract.query_heads[layer],
                contract.hidden_size,
            )?)
        } else {
            None
        };
        let gate_shards_bf16 = if gate_fused && !f32_mirror {
            Some(upload_step_tp_bf16_row_shards(
                &runtime,
                src,
                &p("attn_gate.weight"),
                contract.query_heads[layer],
                contract.hidden_size,
            )?)
        } else {
            None
        };
        Some(StepTpAttention {
            q_norm: upload_step_tp_f32_copies(
                &runtime,
                src,
                &p("attn_q_norm.weight"),
                contract.head_dim,
            )?,
            k_norm: upload_step_tp_f32_copies(
                &runtime,
                src,
                &p("attn_k_norm.weight"),
                contract.head_dim,
            )?,
            decode_input,
            gate_shards,
            gate_shards_bf16,
        })
    } else {
        None
    };
    if f32_mirror {
        eprintln!(
            "[step-tp-qkv] load layer={layer} devices={:?} projections=qkv \
             qkv_tensor_parallel=true attention_local=true kv_local=true output_local=true \
             transport={} native_p2p={} bf16_residency=f32-mirror \
             output=root-readback performance_claim=false",
            spec.devices,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    } else {
        eprintln!(
            "[step-tp-qkv] load layer={layer} devices={:?} projections=qkv \
             qkv_tensor_parallel=true attention_local=true kv_local=true output_local=true \
             transport={} native_p2p={} output=root-readback performance_claim=false",
            spec.devices,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    }
    eprintln!(
        "[step-tp-attn-plan] load layer={layer} devices={:?} \
         qkv_tensor_parallel=true attention_tensor_parallel={} kv_cache_distributed={} \
         attention_scope={} transport={} native_p2p={} replicated_decode_input_prepared={} \
         performance_claim=false",
        spec.devices,
        attention_placement.is_rank_local(),
        attention_placement.is_rank_local(),
        attention_placement.label(),
        runtime.transport_label(),
        runtime.native_p2p(),
        attention
            .as_ref()
            .is_some_and(|attention| attention.decode_input.is_some()),
    );
    if f32_mirror {
        eprintln!(
            "[step-tp-o] load layer={layer} devices={:?} projection=o \
             o_tensor_parallel=true attention_local=true kv_local=true \
             transport={} native_p2p={} reduction=global-tp8-block-order \
             bf16_residency=f32-mirror output=root-readback performance_claim=false",
            spec.devices,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    } else {
        eprintln!(
            "[step-tp-o] load layer={layer} devices={:?} projection=o \
             o_tensor_parallel=true attention_local=true kv_local=true \
             transport={} native_p2p={} reduction=global-tp8-block-order \
             output=root-readback performance_claim=false",
            spec.devices,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    }
    Ok(Some(StepTpQkv {
        runtime,
        q,
        k,
        v,
        o,
        attention,
        devices: spec.devices,
        layer,
    }))
}

/// Decide + build the resident expert slabs for one layer. Budget check runs once per device,
/// RESIDENT-IF-FITS (2026-08-02, research/residency-cap-20260802/): the bank is resident when
/// its EXACT byte total (summed from the GGUF header — UD-quants make per-layer bytes
/// non-uniform, Ornith-35B blk.0 is +7% over the mean, so first-layer x n_layer misprojects)
/// plus the file's non-expert bytes plus a measured headroom reserve fits free VRAM. The old
/// default (0.80 x free vs first-layer x n_layer) reserved 20% of the card (4.8GB on 24GB)
/// and spilled the Ornith-35B bank that fits — a priced -33% decode / -54% prefill. Measured
/// need beside the weights at board shape is ~1.7GB (CUDA ctx + KV + workspace); reserve
/// default 2.0GB, machine-specific override `MEMRA_MOE_RESIDENT_HEADROOM_GB` (VRAM-budget
/// class). `MEMRA_MOE_RESIDENT_GB` stays the absolute expert-budget override;
/// MEMRA_MOE_RESIDENT=0 forces the SLRU path. Fits => every subsequent layer on that device
/// uploads too.
fn build_dev_exps(
    e: &Engine,
    resident: &mut ResidentPlan,
    il: usize,
    gate: &HostExps,
    up: &HostExps,
    down: &HostExps,
) -> Result<Option<crate::hybrid::DevExps>, Box<dyn std::error::Error>> {
    // The resident pointer-table kernels take one qtype/row stride per projection. Mixed-expert
    // layers stay on the metadata-aware staged/SLRU paths until those kernels group by layout.
    if !gate.is_uniform_layout() || !up.is_uniform_layout() || !down.is_uniform_layout() {
        return Ok(None);
    }
    let fp8_host = match (&gate.fp8_blk, &up.fp8_blk, &down.fp8_blk) {
        (None, None, None) => None,
        (Some(g), Some(u), Some(d)) => Some((g, u, d)),
        _ => {
            return Err("resident expert projections disagree on block-E4M3 scale carriage".into());
        }
    };
    let scale_bytes = fp8_host
        .map(|(g, u, d)| (g.scales.len() + u.scales.len() + d.scales.len()) * size_of::<f32>())
        .unwrap_or(0);
    let per_layer = gate.bytes.as_bytes().len()
        + up.bytes.as_bytes().len()
        + down.bytes.as_bytes().len()
        + scale_bytes;
    if gate.tiers.is_some() {
        return Ok(None); // tiered/spill loads keep the cache path
    }
    let fits = resident.should_reside(e, il, per_layer);
    if !fits {
        return Ok(None);
    }
    use cudarc::driver::DevicePtr;
    let n_expert = gate.n_expert;
    let (g, u) = (
        e.htod_bytes_padded(gate.bytes.as_bytes(), 8)?,
        e.htod_bytes_padded(up.bytes.as_bytes(), 8)?,
    );
    // 144B tail slack (2026-07-31, g26 prefill lever): the ragged-k expert MMA walks
    // whole 256-val superblocks — the LAST row's final partial superblock overreads up
    // to 144B past the slab (harmless bytes: the act's zero-padded k-range multiplies
    // every overread weight to zero; the slack only prevents the OOB fault).
    let d = e.htod_bytes_padded(down.bytes.as_bytes(), 144)?;
    // MEMRA_MOE_EXPERT_RP (memra#147): repack the three slabs on the device into the slot-major
    // per-row layout (QT_NVFP4_V2). Only whole NVFP4 slabs, only the plain (non-interleaved, non-FP8) provenance, and
    // only when the quant-plane rows stay 16B-aligned (in_f % 256 == 0 <=> nsb64 % 4 == 0).
    let rp_want = crate::moe_expert_rp_on()
        && fp8_host.is_none()
        && gate.qtype == crate::QT_NVFP4
        && up.qtype == crate::QT_NVFP4
        && down.qtype == crate::QT_NVFP4
        && gate.in_f.is_multiple_of(256)
        && up.in_f.is_multiple_of(256)
        && down.in_f.is_multiple_of(256)
        && gate.row_bytes == gate.in_f / 64 * 36
        && up.row_bytes == up.in_f / 64 * 36
        && down.row_bytes == down.in_f / 64 * 36;
    if crate::moe_expert_rp_on() && !rp_want {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[moe-rp] MEMRA_MOE_EXPERT_RP=1 REFUSED for layer {il}: needs plain NVFP4 gate/up/down \
                 slabs (qtypes {},{},{}), no block-FP8 scales, in_f % 256 == 0 \
                 (gate/up {}, down {}); this and any layer like it stay interleaved",
                gate.qtype, up.qtype, down.qtype, gate.in_f, down.in_f
            );
        }
    }
    let (g, u, d, rp) = if rp_want {
        let g2 = e.nvfp4_expert_split_repack(&g, n_expert, gate.out_f, gate.in_f / 64)?;
        let u2 = e.nvfp4_expert_split_repack(&u, n_expert, up.out_f, up.in_f / 64)?;
        let d2 = e.nvfp4_expert_split_repack(&d, n_expert, down.out_f, down.in_f / 64)?;
        e.stream().synchronize()?;
        drop((g, u, d));
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[moe-rp] resident expert slabs SLOT-MAJOR (QT_NVFP4_V2) from layer {il} (MEMRA_MOE_EXPERT_RP, memra#147): \
                 gate/up rows={} nsb64={}, down rows={} nsb64={}, n_expert={n_expert}; readers get QT_NVFP4_V2, \
                 unwired readers refuse by name",
                gate.out_f,
                gate.in_f / 64,
                down.out_f,
                down.in_f / 64
            );
        }
        (g2, u2, d2, true)
    } else {
        (g, u, d, false)
    };
    let fp8_blk = match fp8_host {
        Some((gate, up, down)) => {
            if e.fp8_blk_nan_count(&g)? != 0
                || e.fp8_blk_nan_count(&u)? != 0
                || e.fp8_blk_nan_count(&d)? != 0
            {
                return Err("native stacked block-E4M3 expert bank contains NaN codes".into());
            }
            Some(DevExpertFp8BlockScales {
                gate: DevExpertFp8ProjectionScales::upload(e, gate, n_expert)?,
                up: DevExpertFp8ProjectionScales::upload(e, up, n_expert)?,
                down: DevExpertFp8ProjectionScales::upload(e, down, n_expert)?,
            })
        }
        None => None,
    };
    let mut host = vec![0u64; 3 * n_expert];
    let (pg, pu, pd) = {
        let __s_e0 = e.stream();
        let (pg, _e0) = g.device_ptr(&__s_e0);
        let __s_e1 = e.stream();
        let (pu, _e1) = u.device_ptr(&__s_e1);
        let __s_e2 = e.stream();
        let (pd, _e2) = d.device_ptr(&__s_e2);
        (pg, pu, pd)
    };
    for ex in 0..n_expert {
        host[ex] = pg + (ex * gate.expert_stride) as u64;
        host[n_expert + ex] = pu + (ex * up.expert_stride) as u64;
        host[2 * n_expert + ex] = pd + (ex * down.expert_stride) as u64;
    }
    let ptr_row = e.htod_u64(&host)?;
    Ok(Some(crate::hybrid::DevExps {
        gate: g,
        up: u,
        down: d,
        ptr_row,
        rp,
        dev: e.ctx().ordinal(),
        fp8_blk,
    }))
}

impl FullAttnLayer {
    /// The per-head QK-norm weights for the norm kernels, or `None` when the family has none.
    /// Every fused kernel that takes these reads a null weight as "pass the row through", so a
    /// family without QK-norm keeps the SAME fused path rather than falling back to a slower one.
    pub fn q_norm_w(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.q_norm.as_ref().map(|t| t.float_data())
    }
    pub fn k_norm_w(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.k_norm.as_ref().map(|t| t.float_data())
    }
}

pub struct FullAttnLayer {
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
    /// AWQ per-input-channel scale for `wo` (o_proj), `None` outside AWQ artifacts (memra#253).
    /// o_proj's input is the attention output, which no preceding weight can absorb a scale
    /// into, so AWQ ships it explicitly and the forward must apply it.
    pub wo_pqs: Option<GpuTensor>,
    /// Per-head QK RMSNorm weights, `None` for families whose plan says
    /// `TensorPresence::Absent` (dense llama/mistral). Every forward path must branch on
    /// this rather than assume it: an all-ones RMSNorm is not the identity.
    pub q_norm: Option<GpuTensor>,
    pub k_norm: Option<GpuTensor>,
    /// step35-class SEPARATE head-wise attention gate: `blk.N.attn_gate.weight [n_embd, n_head_l]`
    /// where `n_head_l` is this layer's query-head count (64 full / 96 SWA on Step-3.7-Flash, so
    /// the width VARIES per layer). Produces one pre-sigmoid scalar per head from the
    /// post-attn_norm hidden state; the forward broadcasts sigmoid(gate) over head_dim and
    /// multiplies attn_out before wo (upstream `step35.cpp:267-285`).
    ///
    /// `None` for every other arch. Do NOT confuse with `LinearAttnLayer::wqkv_gate`, which reads
    /// the SAME tensor name on qwen35's SSM layers but is a different mechanism (a full-width
    /// z-gate, not a per-head scalar), nor with the qwen35 FUSED gate packed inside wq that
    /// `ModelConfig::attn_out_gate()` / `q_gate_split` handle.
    pub attn_gate: Option<GpuTensor>,
    /// Step-3.7 Q/K/V column and O row sharding. Qualified global-attention layers may also own
    /// rank-local QK normalization, RoPE, KV/cache, and attention; SWA layers retain the owning
    /// stage's windowed cache/attention path.
    pub step_tp_qkv: Option<StepTpQkv>,
}

pub struct StepTpQkv {
    pub runtime: Arc<crate::tp::TpE4m3HostBounce>,
    pub q: crate::tp::ResidentBf16ColumnParallel,
    pub k: crate::tp::ResidentBf16ColumnParallel,
    pub v: crate::tp::ResidentBf16ColumnParallel,
    pub o: crate::tp::ResidentStepBf16RowParallel,
    pub attention: Option<StepTpAttention>,
    pub devices: Vec<usize>,
    pub layer: usize,
}

pub struct StepTpAttention {
    pub q_norm: Vec<CudaSlice<f32>>,
    pub k_norm: Vec<CudaSlice<f32>>,
    pub decode_input: Option<std::sync::Mutex<crate::tp::ResidentReplicatedDeviceRows>>,
    /// Per-rank attn_gate row shards (rank-local heads x hidden, f32) — the fused QKV+gate
    /// kernel's fourth weight. None when the layer has no separate head gate.
    pub gate_shards: Option<Vec<CudaSlice<f32>>>,
    /// BF16 twin of `gate_shards` (raw checkpoint bytes) for the mirror-off fused kernels.
    pub gate_shards_bf16: Option<Vec<CudaSlice<u8>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepTpKvDeviceAdmission {
    pub device: usize,
    pub bytes: usize,
}

/// Latent-KV geometry for one MLA layer, resolved from its canonical attention plan. The KV
/// cache stores ONE `latent_dim`-wide row per token per layer: [rmsnorm(c_kv) | rope(k_pe)];
/// V is the first `kv_rank` elements of the SAME row (no V plane). All heads stream it (MQA).
#[derive(Clone, Copy, Debug)]
pub struct MlaGeom {
    pub n_head: usize,     // 64  — query heads; n_head_kv semantics = 1
    pub d_nope: usize,     // 192 — qk nope head dim (absorb GEMM K)
    pub d_rope: usize,     // 64  — decoupled rope width (q_pe / k_pe)
    pub d_v: usize,        // 256 — v head dim after wv_b decompression
    pub kv_rank: usize,    // 512 — latent rank (absorbed qk dim, AV accumulator width)
    pub latent_dim: usize, // 576 = kv_rank + d_rope — the cache row / K width
    pub scale: f32,        // 1/sqrt(d_nope + d_rope) = 1/16 — NOT 1/sqrt(latent_dim)
}

/// GLM-5.2 MLA attention block (DESIGN.md §3.1 mapping). INCREMENT 2: loader-only — the
/// projections + latent-cache geometry land on device; forward arms (prefill/decode/dc/graph)
/// are increment 4. The CPU oracle for those arms is `crate::mla` (naive ≡ absorbed, proven).
/// Geometry of one layer's DSA k-pool indexer, resolved from `SparseIndexPlan::Own`.
#[derive(Clone, Copy, Debug)]
pub struct MlaIndexerGeom {
    pub heads: usize,    // 32 — indexer heads (NOT the MLA query heads)
    pub head_dim: usize, // 128
    pub top_k: usize,    // 2048 — RAW TOKEN budget; the pool budget is top_k / pool
    pub pool: usize,     // 4 — consecutive cached tokens per candidate
    pub always_select_tail: bool,
}

impl MlaIndexerGeom {
    /// Candidate pools that fit the budget, given how many complete pools the cache holds.
    pub fn select_k(&self, n_pools: usize) -> usize {
        (self.top_k / self.pool).min(n_pools)
    }

    /// The `select_k_cap` a FIXED-GEOMETRY live launch must pass for a plane of `n_pools`
    /// capacity. It is `select_k(n_pools)`, the same clamped value [`Self::index_width`] sizes
    /// the row from, and that identity is the point: `memra_mla_kpool_select_live_f32` audits
    /// `width_cap >= select_k_cap * pool + pool - 1` and returns 40014 otherwise. Passing the
    /// UNCLAMPED `top_k / pool` instead makes the audit disagree with the width exactly when
    /// `capacity_tokens < top_k` (2026-09-06: a ctx=386 session against `top_k` 2048 failed
    /// capture and latched the whole stage eager, -8.71% on the served pair). Named so the
    /// invariant has one home and `live_select_k_cap_matches_index_width` can pin it.
    pub fn live_select_k_cap(&self, n_pools: usize) -> usize {
        self.select_k(n_pools)
    }

    /// Width of one query's index list: the expanded pool budget plus the maximum tail.
    pub fn index_width(&self, n_pools: usize) -> usize {
        self.select_k(n_pools) * self.pool
            + if self.always_select_tail {
                self.pool - 1
            } else {
                0
            }
    }

    /// Packed indexer state row: `[k_norm(wk(x)) | index_kpool_compress_gate(x)]`.
    pub fn state_width(&self) -> usize {
        2 * self.head_dim
    }
}

/// The DSA k-pool indexer of one MLA layer (glm5_next). Its projections are SEPARATE from the
/// attention path: the indexer scores pool-collapsed keys of its own and hands the MLA core a
/// gathered position list. Loading is ALL-OR-REFUSE — see `MlaAttnLayer::load`.
pub struct MlaIndexer {
    pub wq_b: GpuTensor,         // indexer.attn_q_b.weight  [Lq -> heads*head_dim]
    pub wk: GpuTensor,           // indexer.attn_k.weight    [H -> head_dim]
    pub k_norm_w: GpuTensor,     // indexer.k_norm.weight    [head_dim]  LayerNorm, not RMSNorm
    pub k_norm_b: GpuTensor,     // indexer.k_norm.bias      [head_dim]  — the bias is why
    pub weights_proj: GpuTensor, // indexer.proj.weight     [H -> heads]
    pub kpool_gate: GpuTensor,   // indexer.kpool_gate.weight [H -> head_dim]
    pub kpool_ape: GpuTensor,    // indexer.kpool_ape.weight  [pool][head_dim] row-major
    pub geom: MlaIndexerGeom,
}

pub struct MlaAttnLayer {
    pub wq_a: GpuTensor,      // attn_q_a.weight      [H -> Lq] (q down-projection)
    pub q_a_norm: GpuTensor,  // attn_q_a_norm.weight [Lq]
    pub wq_b: GpuTensor, // attn_q_b.weight      [Lq -> N*(nope+rope)] (q up, per head [nope|rope])
    pub wkv_a: GpuTensor, // attn_kv_a_mqa.weight [H -> Lkv+rope] (latent row producer)
    pub kv_a_norm: GpuTensor, // attn_kv_a_norm.weight [Lkv] (c_kv rms; k_pe is NOT normed)
    pub wk_b: GpuTensor, // attn_k_b.weight      [nope, Lkv, N] 3D — TRANSPOSED nope slice of
    //   kv_b (conversion split): the per-head absorb GEMM operand
    pub wv_b: GpuTensor, // attn_v_b.weight      [Lkv, V, N] 3D — the post-softmax decompress
    /// BF16 copies of `wk_b` / `wv_b` (door `MEMRA_MLA_ABSORB_BF16`), built at load ONLY when
    /// every f32 element of the resident plane is an exact widening of a BF16 value (the B200
    /// hybrid mint ships `kv_b_proj` in BF16); `None` otherwise, and the f32 path stays.
    pub wk_b16: Option<CudaSlice<u16>>,
    pub wv_b16: Option<CudaSlice<u16>>,
    pub wo: GpuTensor, // attn_output.weight   [N*V -> H]
    pub geom: MlaGeom,
    /// `Some` exactly when the layer's plan declares `SparseIndexPlan::Own { kpool: Some(..) }`.
    /// `None` means the layer attends DENSELY — correct only for a plan that asked for dense.
    pub index: Option<MlaIndexer>,
    /// glm5 TP sidecar (`MEMRA_GLM5_TP`, lane/glm5-tp2). `Some` means THIS struct is the
    /// ROOT-RANK HEAD SHARD (heads/ranks, replicated latent/indexer operands) and the
    /// sidecar carries the peer shards + runtime. Every plain entry refuses a sharded layer
    /// by name; only the TP walk may execute it. `None` everywhere else.
    pub tp: Option<Box<crate::glm5_tp::Glm5TpMla>>,
    /// True on EVERY rank's shard (root AND peers — the peers' `tp` is `None`, so this is
    /// the only marker they carry). Composition guard (lane/glm5-composition): doored
    /// kernels whose gates ran on the FULL-head geometry only (`MEMRA_MLA_TC_PREFILL`)
    /// decline a shard by this flag and fall through to their ungated-composition-free
    /// arms; the fixture gates cannot exercise those doors (kv_rank-stamped kernels), so
    /// the decline is fail-closed by construction until a real-artifact box gate lands.
    pub tp_shard: bool,
}

impl MlaAttnLayer {
    /// Load one MLA attention block to device. `attn_kv_b` (the unsplit tensor, when present)
    /// is intentionally NOT loaded — v1 runs absorbed-form everywhere; the MHA-prefill arm that
    /// would consume it is a later arc (DESIGN.md §3.1 "unused v1").
    ///
    /// NOTE (glm53-flash lane, 2026-08-28): wk_b/wv_b are 3D and ALWAYS f32-resident, on every
    /// checkpoint dtype. There is no quantized 3D layout in this engine — `row_bytes` is derived
    /// from `ne[1]`, which is the middle axis on a 3D tensor, so `GpuTensor::load_from_source`
    /// refuses a quantized 3D tensor by name rather than mis-striding it. Checkpoints that ship
    /// the fused `kv_b_proj` quantized are handled at the SOURCE: `TransformKind::MlaKeyUpSplit` /
    /// `MlaValueUpSplit` dequantize through `deq_f32` (BF16, F16, F32, F8-E4M3, modelopt NVFP4)
    /// and emit the F32 3D planes. The residency audit below is the load-time backstop.
    pub fn load(
        e: &Engine,
        src: &dyn TensorSource,
        il: u32,
        plan: &memra_gguf::model_plan::MlaAttentionPlan,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let memra_gguf::model_plan::MlaAttentionPlan::LatentKv {
            query_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_head_dim,
            rope_head_dim,
            value_head_dim,
            sparse_index,
            ..
        } = plan
        else {
            return Err(format!(
                "native MLA loader has no compressed-KV implementation for block {il}"
            )
            .into());
        };
        let d_nope = qk_head_dim
            .checked_sub(*rope_head_dim)
            .ok_or("MLA rope head width exceeds total QK head width")?;
        let p = |s: &str| format!("blk.{il}.{s}");
        let geom = MlaGeom {
            n_head: *query_heads as usize,
            d_nope: d_nope as usize,
            d_rope: *rope_head_dim as usize,
            d_v: *value_head_dim as usize,
            kv_rank: *kv_lora_rank as usize,
            latent_dim: (*kv_lora_rank + *rope_head_dim) as usize,
            scale: 1.0 / (*qk_head_dim as f32).sqrt(),
        };
        let wq_a = load_t(e, src, &p("attn_q_a.weight"))?;
        let wq_b = load_t(e, src, &p("attn_q_b.weight"))?;
        let wkv_a = load_t(e, src, &p("attn_kv_a_mqa.weight"))?;
        let wk_b = load_t(e, src, &p("attn_k_b.weight"))?;
        let wv_b = load_t(e, src, &p("attn_v_b.weight"))?;
        let wo = load_t(e, src, &p("attn_output.weight"))?;
        // RESIDENCY AUDIT, at load, by name. `mla_absorb_q` / `mla_decompress_v` take raw f32
        // slices: these two 3D operands have no quantized resident layout and never will while the
        // kernels are f32. The SOURCE is responsible for materializing them f32 whatever the
        // checkpoint ships — `TransformKind::MlaKeyUpSplit`/`MlaValueUpSplit` dequantize the fused
        // `kv_b_proj` through `deq_f32`, so BF16, F8-E4M3 and modelopt NVFP4 all land here Float.
        // `GpuTensor::load_from_source` already refuses a quantized 3D tensor outright (wrong
        // row_bytes); this catches the remaining shape — a quantized operand that satisfied that
        // guard — at load instead of in the forward path.
        for (w, tensor) in [(&wk_b, "attn_k_b"), (&wv_b, "attn_v_b")] {
            if !matches!(w, GpuTensor::Float { .. }) {
                return Err(format!(
                    "blk.{il}.{tensor}.weight is not f32-resident. The MLA conversion-split \
                     operands feed f32-only absorb/decompress kernels; the checkpoint source must \
                     dequantize them (TensorTransform::SplitMlaKv) rather than hand the engine a \
                     quantized plane"
                )
                .into());
            }
        }
        // shape audit at load (fail loudly, not as garbage activations later):
        let n_head = wq_b.out_features() / (geom.d_nope + geom.d_rope);
        assert_eq!(
            wq_b.out_features(),
            n_head * (geom.d_nope + geom.d_rope),
            "wq_b out {} not a multiple of qk_head_dim {}",
            wq_b.out_features(),
            geom.d_nope + geom.d_rope
        );
        assert_eq!(
            wq_a.in_features(),
            wkv_a.in_features(),
            "q_a/kv_a hidden mismatch"
        );
        assert_eq!(
            wq_b.in_features(),
            *q_lora_rank as usize,
            "wq_b in != q_lora_rank"
        );
        assert_eq!(
            n_head, geom.n_head,
            "MLA checkpoint head count != ModelPlan"
        );
        assert_eq!(
            wkv_a.out_features(),
            geom.latent_dim,
            "wkv_a out != kv_lora_rank + rope"
        );
        assert_eq!(
            wk_b.ne(),
            &[geom.d_nope as u64, geom.kv_rank as u64, n_head as u64],
            "attn_k_b must be the TRANSPOSED (nope, kv_rank, head) conversion split"
        );
        assert_eq!(
            wv_b.ne(),
            &[geom.kv_rank as u64, geom.d_v as u64, n_head as u64],
            "attn_v_b must be the (kv_rank, v, head) conversion split"
        );
        assert_eq!(
            wo.in_features(),
            n_head * geom.d_v,
            "wo in != n_head * v_head_dim"
        );
        let index = Self::load_indexer(e, src, il, sparse_index, *q_lora_rank)?;
        // BF16 copies of the absorb planes (door MEMRA_MLA_ABSORB_BF16): exact or absent.
        let bf16_copy = |w: &GpuTensor,
                         name: &str|
         -> Result<Option<CudaSlice<u16>>, Box<dyn std::error::Error>> {
            let GpuTensor::Float { data, .. } = w else {
                return Ok(None);
            };
            let host = e.dtoh(data)?;
            if host.iter().any(|v| v.to_bits() & 0xFFFF != 0) {
                eprintln!(
                    "[mla-absorb-bf16] {name}: resident plane is not an exact BF16 widening; the \
                     f32 path stays for this layer"
                );
                return Ok(None);
            }
            let bits: Vec<u16> = host.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            Ok(Some(e.htod_u16(&bits)?))
        };
        let wk_b16 = bf16_copy(&wk_b, "attn_k_b")?;
        let wv_b16 = bf16_copy(&wv_b, "attn_v_b")?;
        Ok(MlaAttnLayer {
            wq_a,
            q_a_norm: load_t(e, src, &p("attn_q_a_norm.weight"))?,
            wq_b,
            wkv_a,
            kv_a_norm: load_t(e, src, &p("attn_kv_a_norm.weight"))?,
            wk_b,
            wv_b,
            wk_b16,
            wv_b16,
            wo,
            geom,
            index,
            tp: None,
            tp_shard: false,
        })
    }

    /// Load the layer's DSA k-pool indexer, or refuse BY NAME.
    ///
    /// There is no fallback arm here on purpose. Below `index_topk` the indexer selects every
    /// visible position and dense attention is the same function; above it they diverge, and
    /// glm5_next's whole claim is a 1,048,576-token context. A layer whose plan declares the
    /// indexer and whose checkpoint is missing one of its tensors must stop the load, not serve
    /// dense attention that looks fluent and is wrong past 2048 tokens.
    ///
    /// `kpool: None` (the GLM-5.2 / dsv4 per-token indexer) returns `None`: that variant scores
    /// raw cache rows and has no implementation on this path — see the gap note in
    /// `HybridModel::mla_attn_core`.
    fn load_indexer(
        e: &Engine,
        src: &dyn TensorSource,
        il: u32,
        sparse_index: &memra_gguf::model_plan::SparseIndexPlan,
        q_lora_rank: u32,
    ) -> Result<Option<MlaIndexer>, Box<dyn std::error::Error>> {
        let memra_gguf::model_plan::SparseIndexPlan::Own {
            heads,
            head_dim,
            top_k,
            kpool: Some(kpool),
        } = sparse_index
        else {
            return Ok(None);
        };
        let geom = MlaIndexerGeom {
            heads: *heads as usize,
            head_dim: *head_dim as usize,
            top_k: *top_k as usize,
            pool: kpool.pool as usize,
            always_select_tail: kpool.always_select_tail,
        };
        if geom.heads == 0 || geom.head_dim == 0 || geom.pool == 0 || geom.top_k < geom.pool {
            return Err(format!(
                "blk.{il}: SparseIndexPlan::Own declares an unusable k-pool indexer \
                 (heads {}, head_dim {}, pool {}, top_k {}) — heads/head_dim/pool must be \
                 positive and top_k must admit at least one pool",
                geom.heads, geom.head_dim, geom.pool, geom.top_k
            )
            .into());
        }
        // Presence is checked BEFORE the load, not after: `GpuTensor::load_from_source` PANICS
        // on a missing tensor, and a panic mid-load leaves the caller nothing to report and no
        // way to name the constraint. This turns it into an error that says what is missing and
        // why the load must stop.
        let need = |suffix: &str| -> Result<GpuTensor, Box<dyn std::error::Error>> {
            let name = format!("blk.{il}.{suffix}");
            if !src.has(&name) {
                return Err(format!(
                    "blk.{il}: the layer's ModelPlan declares a DSA k-pool indexer but the \
                     checkpoint has no `{name}`. This layer MUST NOT fall back to dense \
                     attention: dense and indexed attention are the same function only below \
                     index_topk ({}), and glm5_next serves a 1,048,576-token context",
                    geom.top_k
                )
                .into());
            }
            load_t(e, src, &name).map_err(|source| -> Box<dyn std::error::Error> {
                format!("blk.{il}: DSA k-pool indexer tensor `{name}` failed to load: {source}")
                    .into()
            })
        };
        let wq_b = need("indexer.attn_q_b.weight")?;
        let wk = need("indexer.attn_k.weight")?;
        let k_norm_w = need("indexer.k_norm.weight")?;
        let k_norm_b = need("indexer.k_norm.bias")?;
        let weights_proj = need("indexer.proj.weight")?;
        let kpool_gate = need("indexer.kpool_gate.weight")?;
        let kpool_ape = need("indexer.kpool_ape.weight")?;
        // The three operands the kernels read through `float_data()` have no quantized resident
        // layout; audit at load rather than through that accessor's norm-flavoured panic.
        for (w, name) in [
            (&k_norm_w, "indexer.k_norm.weight"),
            (&k_norm_b, "indexer.k_norm.bias"),
            (&kpool_ape, "indexer.kpool_ape.weight"),
        ] {
            if !matches!(w, GpuTensor::Float { .. }) {
                return Err(format!(
                    "blk.{il}.{name} is not f32-resident. The indexer's LayerNorm affine and \
                     k-pool positional embedding feed f32-only kernels"
                )
                .into());
            }
        }
        assert_eq!(
            wq_b.in_features(),
            q_lora_rank as usize,
            "blk.{il}.indexer.attn_q_b in != q_lora_rank"
        );
        assert_eq!(
            wq_b.out_features(),
            geom.heads * geom.head_dim,
            "blk.{il}.indexer.attn_q_b out != index heads * head_dim"
        );
        assert_eq!(
            wk.out_features(),
            geom.head_dim,
            "blk.{il}.indexer.attn_k out != index head_dim"
        );
        assert_eq!(
            weights_proj.out_features(),
            geom.heads,
            "blk.{il}.indexer.proj out != index heads"
        );
        assert_eq!(
            kpool_gate.out_features(),
            geom.head_dim,
            "blk.{il}.indexer.kpool_gate out != index head_dim"
        );
        assert_eq!(
            kpool_ape.float_data().len(),
            geom.pool * geom.head_dim,
            "blk.{il}.indexer.kpool_ape must hold pool * head_dim elements"
        );
        Ok(Some(MlaIndexer {
            wq_b,
            wk,
            k_norm_w,
            k_norm_b,
            weights_proj,
            kpool_gate,
            kpool_ape,
            geom,
        }))
    }
}

/// Every `Mixer` match OUTSIDE the three wired MLA paths (stateless forward, stateful prime,
/// T=1 decode) routes here. Increment 4 landed the MLA kernel family and those three arms
/// (`cu/mla_attn.cu`, gated in `tests/mla_gpu_forward.rs` against the `crate::mla` CPU oracle);
/// the remaining paths — batched decode, speculative verify, the captured-graph core-split
/// prime, the TP/PP mirrors — each carry state and dispatch discipline no MLA parity gate has
/// covered, and a plausible-looking wrong answer is worse than a named stop. DESIGN.md puts
/// graph capture, the batched tick and the MTP/spec route in increment 7.
#[track_caller]
pub(crate) fn mla_path_unimplemented(path: &str) -> ! {
    panic!(
        "Mixer::Mla has no {path} arm — the MLA forward is wired for the stateless forward, \
         the stateful prime and T=1 decode only (cu/mla_attn.cu, increment 4); this path needs \
         its own parity gate before it may run \
         (research/mla-bringup-20260801/DESIGN.md §4, increment 7)"
    )
}

/// Every `Mixer` match OUTSIDE the three wired KDA paths (stateless forward, stateful prime,
/// T=1 decode) routes here. Those other paths — batched decode, speculative verify, the
/// captured-graph core-split prime, the TP/PP mirrors — each carry their own state and dispatch
/// discipline that a KDA layer has not been gated on, and a plausible-looking wrong answer is
/// worse than a named stop.
#[track_caller]
pub(crate) fn kda_path_unimplemented(path: &str) -> ! {
    panic!(
        "Mixer::Kda has no {path} arm — glm5_next KDA is wired for the stateless forward, the \
         stateful prime and T=1 decode only (crates/memra-engine/src/kda.rs); this path needs \
         its own parity gate before it may run"
    )
}

pub struct LinearAttnLayer {
    pub geometry: memra_gguf::model_plan::GatedDeltaNetPlan,
    pub wqkv: GpuTensor,       // [n_embd, conv_dim] -> qkv_mixed
    pub wqkv_gate: GpuTensor,  // [n_embd, value_dim] -> z
    pub ssm_beta: GpuTensor,   // [n_embd, num_v_heads]
    pub ssm_alpha: GpuTensor,  // [n_embd, num_v_heads]
    pub ssm_a: GpuTensor,      // [num_v_heads] (pre-negated -exp(A_log))
    pub ssm_dt: GpuTensor,     // [num_v_heads] bias
    pub ssm_conv1d: GpuTensor, // [d_conv, conv_dim]
    pub ssm_norm: GpuTensor,   // [head_v_dim]
    pub ssm_out: GpuTensor,    // [value_dim, n_embd]
}

#[allow(clippy::large_enum_variant)] // allow: variant size asymmetry is deliberate; these enums live in per-layer tables, not hot moves
pub enum Mixer {
    Full(FullAttnLayer),
    Linear(LinearAttnLayer),
    /// glm-dsa MLA block (loader-only in increment 2; forward = increment 4).
    Mla(MlaAttnLayer),
    /// glm5_next Kimi Delta Attention block (crate::kda).
    Kda(crate::kda::KdaAttnLayer),
}

/// MoE weights for one layer. Router + shared expert stay GPU-RESIDENT (tiny); the routed
/// experts stay HOST-RESIDENT (HostExps) and are staged per-token (EDGE-1).
///
/// The shared-expert fields are `Option`: qwen35moe carries a shared expert, but OLMoE (and most
/// vanilla MoE) have none (`shared_expert_intermediate_size` absent) — those layers `load_opt` the
/// shexp tensors to `None` (ST-MOE-PLAN §1.3, §3.2). When `None` the shared-expert branch is skipped.
pub struct MoeWeights {
    pub gate_inp: GpuTensor, // F32 [n_embd, n_expert] router  (GPU resident, Float)
    pub gate_inp_shexp: Option<GpuTensor>, // F32 [n_embd] 1-D shared gate dot (qwen35moe only)
    /// DeepSeek-V3/MiniMax-M3 `e_score_correction_bias` [n_expert]: added to the sigmoid scores
    /// for expert SELECTION only; the routing weights use the un-biased scores. The host row is
    /// the rollback oracle; the device row is zero-filled when the checkpoint carries no bias.
    pub exp_probs_b: Option<Vec<f32>>,
    pub exp_probs_b_dev: CudaSlice<f32>,
    /// Original-width router mask for physically pruned expert overlays. Inactive ids never enter
    /// top-k, so their absent weight files cannot be dispatched. The device row is all ones when
    /// no overlay mask exists.
    pub active_experts: Option<Vec<bool>>,
    pub active_experts_dev: CudaSlice<u8>,
    pub gate_exps: HostExps, // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub up_exps: HostExps,   // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub down_exps: HostExps, // [n_ff_exp, n_embd, n_expert] TRANSPOSED (HOST)
    pub gate_shexp: Option<GpuTensor>,
    pub up_shexp: Option<GpuTensor>,
    pub down_shexp: Option<GpuTensor>,
    /// FITS-VRAM RESIDENT EXPERTS (2026-07-06): when the WHOLE model's expert bytes fit the VRAM
    /// budget, each (proj) slab is uploaded once as a contiguous device buffer and the fused
    /// _dev kernels take base+ex*stride pointers — no SLRU, no dispatch, no residency checks
    /// (llama's full-offload regime; measured 169.55 vs memra's cache path 28.5 on the local 35B).
    /// None => the SLRU host-expert machinery (the spill regime, where it WINS vs llama's
    /// CPU-offload degradation). Decided at load in `load_ffn` (MEMRA_MOE_RESIDENT=0 forces off).
    pub dev_exps: Option<DevExps>,
    /// Step-only live EP correctness path. Routed experts are split across distinct rank-owned
    /// native E4M3 banks; router/shared-expert work remains on the owning PP stage. Host-bounce
    /// expert dispatch/combine is deterministic correctness evidence only.
    pub step_ep: Option<StepEpExps>,
    /// Step-only live TP correctness path. Every routed expert is tensor-sharded across the rank
    /// group when the checkpoint scale geometry permits it; TP4/TP8 use the `step_ep` ownership
    /// path instead. Router/shared-expert work remains on the owning PP stage.
    pub step_tp: Option<StepTpExps>,
    /// glm5-only EP-2 sidecar (`MEMRA_GLM5_TP`): whole-expert contiguous halves on the two
    /// rank devices; router/shared-expert/macros stay HERE unchanged. When `Some`, the MoE
    /// forward takes the EP dispatch/combine walk and every other arm is unreachable for
    /// this layer. `None` everywhere else (zero change).
    pub glm5_ep: Option<crate::glm5_tp::Glm5EpExps>,
    /// Expert TENSOR-parallel slabs (`MEMRA_GLM5_TP_EXPERT_SPLIT`, lane/tp-expert-split-20260906):
    /// every rank holds half of EVERY expert instead of all of half the experts, so routing luck
    /// cannot leave one card with 6 of the token's 8. Mutually exclusive with `glm5_ep`, which is
    /// the whole-expert arm; the loader arms exactly one.
    pub glm5_tp_split: Option<crate::glm5_tp::Glm5TpSplitExps>,
    /// Per-expert post-matmul macro-scales on DEVICE: [3*n_expert] f32 in (gate, up, down)
    /// order — all 1.0 unless the checkpoint carries compressed-tensors NVFP4 global scales
    /// (unsloth qwen3.6 class). The _dev gate_up epilogues multiply unconditionally (x*1.0f
    /// is bit-exact — zero change for macro-free artifacts); the down fold is one
    /// moe_w_scale_by_expert launch gated on `has_macros`.
    pub dev_macros: cudarc::driver::CudaSlice<f32>,
    pub has_macros: bool,
    /// ModelOpt W4A16 uses BF16 expert activations. This lives on the model/layer weights rather
    /// than in process-global state so a multi-model server can also host another NVFP4 program.
    pub w4a16_bf16_activations: bool,
}

/// Expert-parallel residency, one variant per qualified checkpoint artifact class.
#[allow(clippy::large_enum_variant)] // allow: variant size asymmetry is deliberate; these enums live in per-layer tables, not hot moves
pub enum StepEpExpertBank {
    E4m3(crate::tp::ResidentExpertParallel),
    Nvfp4(crate::tp::ResidentNvfp4ExpertParallel),
}

impl StepEpExpertBank {
    /// The E4M3 bank, for programs qualified on that artifact class only (grouped decode/prefill
    /// under device arithmetic). Reaching this with an NVFP4 bank is a wiring bug, not an
    /// operator error — those doors refuse at preflight for NVFP4.
    pub fn e4m3(&self) -> Result<&crate::tp::ResidentExpertParallel, String> {
        match self {
            Self::E4m3(bank) => Ok(bank),
            Self::Nvfp4(_) => Err(
                "Step grouped expert program reached an NVFP4 bank; this path is qualified \
                 for the E4M3 artifact only"
                    .to_string(),
            ),
        }
    }
}

pub struct StepEpExps {
    pub runtime: Arc<crate::tp::TpE4m3HostBounce>,
    pub experts: StepEpExpertBank,
    pub devices: Vec<usize>,
    pub configured_by_tp: bool,
    pub activation_limit: Option<f32>,
    /// Immutable load-time selection of the W4A16 device-resident NVFP4 EP decode program.
    pub nvfp4_device_routes: bool,
    /// Persistent one-token grouped projection/combine state for eager decode. Opt-in prefill
    /// uses the model-scoped executor instead of multiplying capacity workspaces per layer.
    pub grouped_decode: Option<std::sync::Mutex<StepEpGroupedDecode>>,
}

pub struct StepEpGroupedDecode {
    pub(crate) projection: crate::tp::PreparedStepGroupedExpertParallelGate,
    pub(crate) combine: crate::tp::PreparedPeerWeightedRouteCombine,
}

#[derive(Default)]
pub(crate) struct StepEpGroupedPrefill {
    pub(crate) state: Option<StepEpGroupedPrefillState>,
}

pub(crate) struct StepEpGroupedPrefillState {
    pub(crate) devices: Vec<usize>,
    pub(crate) grouped: StepEpGroupedDecode,
}

/// Tensor-parallel expert residency, one variant per qualified checkpoint artifact class.
#[allow(clippy::large_enum_variant)] // allow: variant size asymmetry is deliberate; these enums live in per-layer tables, not hot moves
pub enum StepTpExpertBank {
    E4m3(crate::tp::ResidentTensorParallel),
    Nvfp4(crate::tp::ResidentNvfp4TensorParallel),
}

pub struct StepTpExps {
    pub runtime: Arc<crate::tp::TpE4m3HostBounce>,
    pub experts: StepTpExpertBank,
    pub devices: Vec<usize>,
    /// step35 routed SwiGLU clamp for this layer (min(silu, limit) * clamp(up, +-limit)) —
    /// elementwise, so the column-sharded TP program preserves it exactly.
    pub activation_limit: Option<f32>,
}

impl MoeWeights {
    #[inline]
    pub fn has_uniform_expert_layout(&self) -> bool {
        self.gate_exps.is_uniform_layout()
            && self.up_exps.is_uniform_layout()
            && self.down_exps.is_uniform_layout()
    }

    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_experts
            .as_ref()
            .map(|mask| mask.iter().filter(|&&active| active).count())
            .unwrap_or(self.gate_exps.n_expert)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qmatvec_view(
        &self,
        e: &Engine,
        w: &CudaSlice<u8>,
        range: std::ops::Range<usize>,
        x: &cudarc::driver::CudaView<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        qtype: i32,
        row_bytes: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.w4a16_bf16_activations && qtype == crate::QT_NVFP4 {
            e.qmatvec_view_bf16_activation(w, range, x, m, in_f, out_f, qtype, row_bytes)
        } else {
            e.qmatvec_view(w, range, x, m, in_f, out_f, qtype, row_bytes)
        }
    }
}

/// Device-resident expert slabs for one layer (gate/up/down) + the prebuilt [3, n_expert]
/// pointer row the _dev kernels consume.
pub struct DevExps {
    pub gate: CudaSlice<u8>,
    pub up: CudaSlice<u8>,
    pub down: CudaSlice<u8>,
    /// [3*n_expert] u64 device row: gate ptrs, up ptrs, down ptrs (proj-major like layer_dev_row).
    pub ptr_row: CudaSlice<u64>,
    /// The CUDA device ordinal these slabs live on (the OWNING stage's device under the PP
    /// sharded loader — cx-503b sizes and `layer_engine` places per device). Consumers that
    /// dispatch from a DIFFERENT device must NOT dereference the slabs: an m=1 qmatvec over
    /// peer-read expert bytes is the measured 34-150x slow class (research/pp-prefill-20260807
    /// anatomy), strictly worse than SLRU staging. The sequential arm's slab-locality gate
    /// (lane/pp-leverb) keys on this field; the per-stage prime walker makes every layer's
    /// slab local by construction.
    pub dev: usize,
    /// The three slabs are slot-major per row (`QT_NVFP4_V2`; `MEMRA_MOE_EXPERT_RP`, memra#147):
    /// readers must be told so (`crate::rp_qt`) or refuse (`crate::moe_rp_refuse`).
    pub rp: bool,
    /// Native block-E4M3 expert scale slabs, projection-major. When present, the raw checkpoint
    /// code slabs above are the sole resident weight copy and each expert selects its contiguous
    /// scale-grid view.
    pub fp8_blk: Option<DevExpertFp8BlockScales>,
}

pub struct DevExpertFp8BlockScales {
    pub gate: DevExpertFp8ProjectionScales,
    pub up: DevExpertFp8ProjectionScales,
    pub down: DevExpertFp8ProjectionScales,
}

pub struct DevExpertFp8ProjectionScales {
    pub scales: CudaSlice<f32>,
    pub rows: usize,
    pub cols: usize,
    pub expert_stride: usize,
}

impl DevExpertFp8ProjectionScales {
    fn validate(
        host: &crate::model::HostExpertFp8BlockScales,
        n_expert: usize,
    ) -> Result<(), String> {
        if host.expert_stride == 0 {
            return Err("block-E4M3 expert scale stride must be nonzero".into());
        }
        if host.rows * host.cols != host.expert_stride {
            return Err(format!(
                "block-E4M3 expert scale stride mismatch: {}x{} != {}",
                host.rows, host.cols, host.expert_stride
            ));
        }
        let want = n_expert
            .checked_mul(host.expert_stride)
            .ok_or("block-E4M3 expert scale slab length overflow")?;
        if host.scales.len() != want {
            return Err(format!(
                "block-E4M3 scale slab length mismatch: got {}, want {n_expert}x{}={want}",
                host.scales.len(),
                host.expert_stride
            ));
        }
        Ok(())
    }

    fn upload(
        e: &Engine,
        host: &crate::model::HostExpertFp8BlockScales,
        n_expert: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::validate(host, n_expert)?;
        Ok(Self {
            scales: e.htod(&host.scales)?,
            rows: host.rows,
            cols: host.cols,
            expert_stride: host.expert_stride,
        })
    }
}

/// Per-layer FFN: dense SwiGLU (qwen35) or 256-expert MoE (qwen35moe).
#[allow(clippy::large_enum_variant)] // allow: variant size asymmetry is deliberate; these enums live in per-layer tables, not hot moves
pub enum Ffn {
    Dense {
        ffn_gate: GpuTensor,
        ffn_up: GpuTensor,
        ffn_down: GpuTensor,
        /// AWQ per-input-channel scale for `ffn_down` (memra#253). `Some` only for an
        /// AWQ-calibrated checkpoint; the input must be multiplied by it before the down
        /// projection. Adding this field is deliberate: it makes every destructuring site
        /// fail to compile until it decides what to do, so no path can skip the scale.
        ffn_down_pqs: Option<GpuTensor>,
    },
    Moe(MoeWeights),
}

pub struct HybridLayer {
    pub attn_norm: GpuTensor,
    pub post_attn_norm: GpuTensor, // "post_attention_norm" = PRE-FFN norm
    pub mixer: Mixer,
    pub ffn: Ffn,
    pub gemma4: Option<Gemma4LayerBits>,
    /// The layer's two hyper-connection sites (attention, MLP). `Some` iff the compiled plan
    /// declares `ResidualTopology::HyperConnections` — see `crate::hyper`. `None` means the
    /// serial residual, and the two states are never mixed: `HybridModel::hyper` decides which
    /// residual program a forward path runs, and the loader refuses a trunk that disagrees.
    pub hyper: Option<crate::hyper::HyperLayer>,
    /// The SYMMETRIC TP walk's per-peer copies of this layer's glue: the two norms and the two
    /// hyper-connection sites (lane/tp-symmetric-20260906). `tp_glue[i]` lives on peer rank
    /// `i + 1`. Empty on a plain load and on the root-orchestrated TP walk, which fans the root's
    /// activations out to peers every layer instead of letting each rank run its own glue.
    pub tp_glue: Vec<crate::glm5_tp::Glm5TpGlue>,
}

/// Gemma-4 per-layer extras (R8 wiring, HANDOVER "R8 VERIFIED WIRING"): the parallel shared
/// FFN branch, the four extra norms, the router prologue scale vector, per-expert output
/// scales, and the layer output scalar.
pub struct Gemma4LayerBits {
    pub ffn_norm: GpuTensor, // ffn pre-norm (dense: THE ffn norm; moe: shared branch)
    pub post_ffw_norm: GpuTensor, // combined post (before the attn_out residual)
    /// MoE-layer extras (None on the dense gemma4 variants — 31B/E4B): the parallel shared
    /// branch norms + tensors, the router prologue vector, per-expert output scales.
    pub moe_bits: Option<Gemma4MoeBits>,
    pub layer_scale: f32, // layer_output_scale [1]
    /// E4B extras (None on 26B/31B): the per-layer-embedding tail block + KV-share target.
    pub e4b: Option<Gemma4E4bLayer>,
}

/// gemma-4 E4B per-layer bits (see research/gemma4-bringup/e4b-arch-map.md):
/// tail block  cur += rms_norm(proj . (gelu(inp_gate . cur) * inp_pl[il]), post_norm)
/// and the KV-share map — layers il >= n_layer-shared_kv_layers have NO own k/v projections
/// and attend the cache of layer (n_layer-shared) - (swa ? 2 : 1) with their own Q.
pub struct Gemma4E4bLayer {
    pub inp_gate: GpuTensor,  // blk.N.inp_gate  [n_embd, n_epl]
    pub proj: GpuTensor,      // blk.N.proj      [n_epl, n_embd]
    pub post_norm: GpuTensor, // blk.N.post_norm [n_embd]
    /// wave-4b: wq|wk|wv concatenated along OUT (one Q4_0 matvec at t=1 instead of the
    /// fused3 3-subgrid launch). Built at the mirror hook from the GPU byte planes (rows
    /// are independent in Q4_0, so an out-dim concat is a byte concat); own-KV layers only.
    pub qkv_cat: Option<GpuTensor>,
    /// Some(target_layer) on KV-shared layers (wk/wv here are the TARGET layer's tensors,
    /// loaded for shape symmetry only — the forward must skip k/v compute + append and read
    /// the target's cache; TODO dedupe the duplicate weight upload ~63MB).
    pub kv_share: Option<u32>,
}

/// gemma-4 E4B model-level per-layer-embedding tensors (prologue inputs). The token table
/// stays HOST-side raw GGUF bytes at load (Q6_K [n_epl*n_layer, n_vocab], ~2.3GB VRAM when
/// uploaded — the forward arc decides resident-vs-gather placement).
pub struct Gemma4E4bModel {
    /// device copy of the per-layer token table, uploaded on first use (the 26B embd_gpu
    /// pattern — keeps the ~2.3GB off load-critical paths that never decode).
    pub tok_tbl_gpu: std::sync::OnceLock<CudaSlice<u8>>,
    pub tok_embd_bytes: Vec<u8>,
    pub tok_embd_qt: i32,
    pub tok_embd_row_bytes: usize,
    pub model_proj: GpuTensor, // per_layer_model_proj [n_embd, n_epl*n_layer] F16
    pub proj_norm: GpuTensor,  // per_layer_proj_norm [n_epl]
    pub n_epl: usize,
}

pub struct Gemma4MoeBits {
    pub post_ffw_norm_1: GpuTensor, // shared-branch post
    pub pre_ffw_norm_2: GpuTensor,  // moe-branch pre
    pub post_ffw_norm_2: GpuTensor, // moe-branch post
    pub shared_gate: GpuTensor,
    pub shared_up: GpuTensor,
    pub shared_down: GpuTensor,
    /// ffn_gate_inp.scale [n_embd] PRE-multiplied by 1/sqrt(n_embd) at load: the router
    /// prologue (weightless rms_norm x 1/sqrt(n_embd) x scale-vec) collapses to ONE rms_norm
    /// with this as the norm weight (x_hat * (v*s) vs llama's (x_hat*s)*v — one reassociation;
    /// the argmax gate arbitrates).
    pub router_scale_pre: CudaSlice<f32>,
    pub per_expert_scale: Vec<f32>, // ffn_down_exps.scale [n_expert] (host)
    pub per_expert_scale_d: CudaSlice<f32>, // device copy (router-weight fold kernel)
}

/// Qwen3.5 NextN/MTP head: a full transformer block (attn+FFN, same tensors as a trunk layer)
/// plus the MTP glue (enorm/hnorm/eh_proj that fold the next-token embedding into the trunk
/// hidden, and an optional shared_head_norm/head). Loaded from blk.{n_trunk}.* — the block the
/// trunk loop drops. Used for speculative decode (drafts 1 token per call). See research/mtp/MTP-PLAN.md.
/// MEMRA_MTP_HEAD_NVFP4=1: load a NextN block's own lm_head as NVFP4 instead of the BF16 the
/// step-3.7-flash checkpoint ships. Residency is the point — each untrimmed head is BF16
/// [128896, 4096] = 1.06 GB, so a 3-head chain spends 3.18 GB and does not fit beside a
/// 262144-token cache; NVFP4 takes the three to 0.89 GB. The repo's own draft-regime standard
/// already quantizes the draft head this way ("block Q4_K_M + head NVFP4 … NVFP4 head measured
/// zero acceptance cost", tools/make-trimmed-draft.sh), but that builder is a GGUF pipeline, so
/// safetensors families quantize here. Draft-head precision cannot change served output — verify
/// arbitrates every drafted token — so acceptance is the only quantity at risk.
fn load_mtp_head_maybe_nvfp4(
    e: &Engine,
    src: &dyn TensorSource,
    name: &str,
) -> Result<Option<GpuTensor>, Box<dyn std::error::Error>> {
    if !{
        static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        crate::step37_door(&ENV, "MEMRA_MTP_HEAD_NVFP4")
    } {
        return load_opt(e, src, name);
    }
    let Some(v) = src.find(name) else {
        return Ok(None);
    };
    if !matches!(v.ggml_type, GgmlType::BF16) || v.ne[0] % 64 != 0 {
        return load_opt(e, src, name);
    }
    let vals: Vec<f32> = v
        .bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    let blocks = memra_gguf::nvfp4_repack::f32_to_nvfp4(&vals);
    eprintln!(
        "[mtp-head] {name}: BF16 -> NVFP4 ({} MiB, was {} MiB)",
        blocks.len() >> 20,
        v.bytes.len() >> 20
    );
    Ok(Some(GpuTensor::from_quant_bytes(
        e,
        &blocks,
        GgmlType::NVFP4,
        v.ne[0],
        v.ne[1],
        1.0,
    )?))
}

/// Tensor name of the FIRST MTP block's OWN lm_head — the preferred source of FR-Spec trim
/// rows for families whose nextn blocks do not tie to the trunk head. step-3.7-flash ships a
/// DIFFERENT head matrix per nextn block, and gathering trunk rows there measured acceptance
/// 0/248 across K=1..8 while self-consistency still PASSED, so no exactness gate catches it.
/// First 8 hex of sha256 over a file's bytes — any drafter's boot-receipt identity pin
/// (streamed, so a 2.3 GB safetensors never lands in memory twice). `pub(crate)` because the
/// general draft-source seam (`dflash::load_drafter`) mints the pin for every family.
pub(crate) fn sha256_file_hex8(
    path: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    sha256_file_hex(path, 4)
}

/// First `n_bytes` bytes of sha256 over a file, hex-encoded (streamed). The identity pin
/// every draft-side artifact receipt prints: `hex8` for drafters, `hex16` for FR-Spec ranks
/// files (lane/frspec-dflash2-20260902, a ranks file is a per-tokenizer artifact, and a
/// wrong-model file loads silently unless its bytes are named in the engagement line).
pub(crate) fn sha256_file_hex(
    path: &std::path::Path,
    n_bytes: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let digest = hasher.finalize();
    Ok(digest
        .iter()
        .take(n_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// STRICT ranks `.txt` parse (lane/frspec-dflash2-20260902): one token id per line, rank
/// order. Blank lines are skipped; ANY other non-numeric line refuses, the lenient
/// `filter_map(parse().ok())` arm silently drops a corrupted or wrong-format line, and a
/// silently shorter ranks list is exactly the wrong-artifact class the DFlash2 slab must
/// never boot on. Duplicates refuse too (a duplicated id is one fewer distinct draftable
/// token and a sign the file was hand-edited). Pure: CPU-testable, red arms in
/// `frspec_ranks_tests`.
pub fn frspec_parse_ranks_txt_strict(text: &str, what: &str) -> Result<Vec<u32>, String> {
    let mut out: Vec<u32> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let id = line.parse::<u32>().map_err(|_| {
            format!(
                "{what}: line {} is not a token id ({line:?}); a ranks .txt is one integer id \
                 per line in rank order",
                lineno + 1
            )
        })?;
        out.push(id);
    }
    Ok(out)
}

/// Boot-time admission of a ranks list against the head it will index (lane/frspec-dflash2-
/// 20260902, owner order): non-empty, no duplicate id, every id < `n_vocab` (the head's row
/// count), and never MORE rows than the head has (a "trim" wider than the vocabulary is a
/// wrong-model file by construction). Refuses by name; the caller prints the file sha16 in
/// its engagement line so the refused or admitted bytes are identifiable. Pure.
pub fn frspec_validate_ranks(d2t: &[u32], n_vocab: usize, what: &str) -> Result<(), String> {
    if d2t.is_empty() {
        return Err(format!(
            "{what}: the ranks artifact yields an EMPTY id list"
        ));
    }
    if d2t.len() > n_vocab {
        return Err(format!(
            "{what}: {} ranks for a {n_vocab}-row head: a ranks list wider than the vocabulary \
             was minted for a different model",
            d2t.len()
        ));
    }
    if let Some(&bad) = d2t.iter().find(|&&t| t as usize >= n_vocab) {
        return Err(format!(
            "{what}: token id {bad} >= head rows {n_vocab}: the ranks artifact was minted for a \
             different vocabulary (wrong-model file refused at boot)"
        ));
    }
    let mut seen = vec![false; n_vocab];
    for &t in d2t {
        if seen[t as usize] {
            return Err(format!(
                "{what}: token id {t} appears more than once: a ranks list is a set of distinct \
                 ids in rank order"
            ));
        }
        seen[t as usize] = true;
    }
    Ok(())
}

/// The row gather every FR-Spec trim arm runs, as PURE host bytes: `rows[t*row_bytes..]` for
/// each ranked `t`, concatenated in rank order. Split out so the slab byte-identity claim
/// ("slab row r == head row d2t[r]") is a CPU-testable statement about this function, and
/// the GPU gate only has to prove the upload preserved it.
pub fn frspec_gather_rows(rows: &[u8], row_bytes: usize, d2t: &[u32]) -> Vec<u8> {
    let mut gathered = Vec::with_capacity(d2t.len() * row_bytes);
    for &t in d2t {
        let off = t as usize * row_bytes;
        gathered.extend_from_slice(&rows[off..off + row_bytes]);
    }
    gathered
}

#[cfg(test)]
mod frspec_ranks_tests {
    use super::{frspec_gather_rows, frspec_parse_ranks_txt_strict, frspec_validate_ranks};

    #[test]
    fn strict_parse_skips_blank_lines_and_refuses_anything_else() {
        let ok = frspec_parse_ranks_txt_strict("5\n\n 7 \n0\n", "t").unwrap();
        assert_eq!(ok, vec![5, 7, 0]);
        // Trailing newline / no trailing newline: same list.
        assert_eq!(
            frspec_parse_ranks_txt_strict("5\n7", "t").unwrap(),
            vec![5, 7]
        );
        // RED: a header row, a negative id, a float, a comment, every one refuses by line.
        for bad in ["id\n5\n", "5\n-1\n", "5\n7.0\n", "# ranks\n5\n", "5 7\n"] {
            let err = frspec_parse_ranks_txt_strict(bad, "t").unwrap_err();
            assert!(err.contains("is not a token id"), "{bad:?} -> {err}");
        }
        // Empty text parses to an empty list; the validator is what refuses it.
        assert!(frspec_parse_ranks_txt_strict("", "t").unwrap().is_empty());
    }

    #[test]
    fn validate_refuses_empty_oob_duplicate_and_wider_than_vocab() {
        assert!(frspec_validate_ranks(&[3, 1, 0], 4, "t").is_ok());
        // The full vocabulary as a permutation is admissible (a trim of width n_vocab).
        assert!(frspec_validate_ranks(&[3, 1, 0, 2], 4, "t").is_ok());
        let e = frspec_validate_ranks(&[], 4, "t").unwrap_err();
        assert!(e.contains("EMPTY"), "{e}");
        let e = frspec_validate_ranks(&[3, 4], 4, "t").unwrap_err();
        assert!(e.contains("token id 4 >= head rows 4"), "{e}");
        let e = frspec_validate_ranks(&[3, 1, 3], 4, "t").unwrap_err();
        assert!(e.contains("token id 3 appears more than once"), "{e}");
        let e = frspec_validate_ranks(&[0, 1, 2, 3, 0], 4, "t").unwrap_err();
        assert!(e.contains("5 ranks for a 4-row head"), "{e}");
    }

    #[test]
    fn gather_rows_is_the_rank_ordered_row_copy() {
        // 5 rows of 3 bytes: row t = [t, t+10, t+20].
        let rows: Vec<u8> = (0..5u8).flat_map(|t| [t, t + 10, t + 20]).collect();
        let g = frspec_gather_rows(&rows, 3, &[4, 0, 2]);
        assert_eq!(g, vec![4, 14, 24, 0, 10, 20, 2, 12, 22]);
        // RED: a permuted d2t must change the slab (the gather is order-preserving).
        assert_ne!(g, frspec_gather_rows(&rows, 3, &[0, 4, 2]));
        // Identity d2t reproduces the head byte for byte.
        assert_eq!(frspec_gather_rows(&rows, 3, &[0, 1, 2, 3, 4]), rows);
    }
}

pub(crate) fn frspec_trim_own_head_name(n_trunk: usize) -> String {
    format!("blk.{n_trunk}.nextn.shared_head_head.weight")
}

/// MEMRA_MTP_SKIP=1 stub draft head: the FR-Spec trimmed rows + d2t map WITHOUT the embedded
/// MTP/NextN block behind them. Exists so a dspark/DFlash2-drafted model can drop the block's
/// attention mixer + FFN + glue from VRAM while the DFlash2 round keeps its trimmed draft head
/// (dflash.rs consumes exactly `shared_head_head` + `d2t` + `d2t_from_target_head` from the MTP
/// struct, nothing else; verified 2026-08-30, mtp-skip lane). Deliberately NOT an `MtpHead`:
/// every MtpHead block tensor is non-optional, so a stub MtpHead would carry fake tensors
/// reachable by the MTP spec forward paths, and `mtp_spec_capable` keys on `model.mtp.is_some()`
/// and with the stub in its own field, `mtp = None` keeps the MTP spec arm off by construction.
/// Rows always come from the TARGET model's own output head (the loader refuses otherwise), so
/// this is semantically `d2t_from_target_head = true`.
pub struct DflashTrimHead {
    /// Trimmed rows of the trunk `output.weight` (or tied `token_embd.weight`), same gather
    /// (and optional MEMRA_FRSPEC_TRIM_NVFP4 requant) as the MtpHead trim path.
    pub head: GpuTensor,
    /// FR-Spec draft->target vocab map; `d2t[draft_idx]` = target token id of trimmed row.
    pub d2t: Vec<u32>,
    /// First 16 hex of sha256 over the ranks artifact's bytes, the identity the engagement
    /// line prints (`src=<sha16>`), so a wrong-model ranks file is nameable from the log.
    pub src_sha16: String,
}

/// Read a MEMRA_FRSPEC_TRIM d2t rank artifact (already `resolve_arg`-resolved): either the d2t
/// GGUF container or a plain `.txt` (one token id per line, rank order — frspec-owngen writes
/// both). Extracted verbatim from the trim arm of `load_from_source_impl` for the
/// MEMRA_MTP_SKIP stub path, which needs the same list without a loaded MtpHead.
/// The `.txt` arm is STRICT for every consumer (lane/frspec-dflash2-20260902, revuto finding
/// on the re-land): a non-blank non-numeric line refuses by name instead of being dropped,
/// so no trim arm can boot a silently shorter list. Every house writer emits exactly one
/// integer per line (`memra_gguf::d2t::write_d2t`), so a refusal here is a broken file.
fn frspec_read_d2t(path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(if path.ends_with(".txt") {
        let text = std::fs::read_to_string(path)?;
        frspec_parse_ranks_txt_strict(&text, &format!("MEMRA_FRSPEC_TRIM={path}"))?
    } else {
        let tg = GgufFile::open(path)?;
        let d2t_t = tg
            .find("d2t")
            .expect("MEMRA_FRSPEC_TRIM file has no d2t tensor");
        let d2t_bytes = tg.tensor_data(d2t_t);
        match d2t_t.ggml_type {
            GgmlType::I32 => d2t_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
                .collect(),
            GgmlType::I64 => d2t_bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
                .collect(),
            other => panic!("d2t must be I32/I64, got {other:?}"),
        }
    })
}

/// Gather the FR-Spec trimmed head rows from a full head view and upload them. A byte-level row
/// gather (quantized rows are independent — zero requant) unless `want_nvfp4_env` selects the
/// MEMRA_FRSPEC_TRIM_NVFP4 re-encode (BF16 heads with ne0 % 64 == 0 only, same eligibility as
/// the in-place trim arm). Returns the tensor plus `Some((nvfp4_bytes, gathered_bytes))` when
/// the NVFP4 re-encode ran (the caller's receipt line quotes both sizes). Extracted verbatim
/// from the trim arm of `load_from_source_impl` so the MEMRA_MTP_SKIP stub path shares one
/// gather program with the MtpHead trim.
#[allow(clippy::type_complexity)] // allow: one-shot composite return; naming it would hide the (tensor, nvfp4-size receipt) shape that matters at the call site
fn frspec_gather_trimmed_head(
    e: &Engine,
    v: &memra_gguf::source::TensorView<'_>,
    d2t: &[u32],
    want_nvfp4_env: bool,
    macro_scale: f32,
) -> Result<(GpuTensor, Option<(usize, usize)>), Box<dyn std::error::Error>> {
    let out_f = v.ne[1] as usize;
    let row_bytes = v.bytes.len() / out_f;
    assert!(
        d2t.iter().all(|&t| (t as usize) < out_f),
        "d2t token id >= lm_head rows {out_f}"
    );
    let gathered = frspec_gather_rows(&v.bytes, row_bytes, d2t);
    let want_nvfp4 =
        want_nvfp4_env && matches!(v.ggml_type, GgmlType::BF16) && v.ne[0].is_multiple_of(64);
    if want_nvfp4 {
        let in_f = v.ne[0] as usize;
        let vals: Vec<f32> = gathered
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        debug_assert_eq!(vals.len(), d2t.len() * in_f);
        let blocks = memra_gguf::nvfp4_repack::f32_to_nvfp4(&vals);
        let sizes = (blocks.len(), gathered.len());
        let trimmed = GpuTensor::from_quant_bytes(
            e,
            &blocks,
            GgmlType::NVFP4,
            v.ne[0],
            d2t.len() as u64,
            1.0,
        )?;
        Ok((trimmed, Some(sizes)))
    } else {
        let trimmed = match v.ggml_type {
            GgmlType::BF16 => GpuTensor::FloatBf16 {
                data: e.htod_bytes(&gathered)?,
                ne: vec![v.ne[0], d2t.len() as u64],
            },
            GgmlType::F32 => GpuTensor::Float {
                data: e.htod(
                    &gathered
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<f32>>(),
                )?,
                ne: vec![v.ne[0], d2t.len() as u64],
            },
            _ => GpuTensor::from_quant_bytes(
                e,
                &gathered,
                v.ggml_type,
                v.ne[0],
                d2t.len() as u64,
                macro_scale,
            )?,
        };
        Ok((trimmed, None))
    }
}

pub struct MtpHead {
    pub enorm: GpuTensor, // blk.N.nextn.enorm   — RMSNorm of the next-token embedding
    pub hnorm: GpuTensor, // blk.N.nextn.hnorm   — RMSNorm of the trunk hidden
    pub eh_proj: GpuTensor, // blk.N.nextn.eh_proj [2*n_embd, n_embd]: [e_norm; h_norm] -> n_embd
    pub attn_norm: GpuTensor, // blk.N.attn_norm
    pub post_attn_norm: GpuTensor, // blk.N.post_attention_norm (pre-FFN)
    pub mixer: Mixer,     // full-attn block (qwen35 MTP block is full-attn)
    pub ffn: Ffn,         // Dense or Moe, same loader as trunk
    pub shared_head_norm: Option<GpuTensor>, // blk.N.nextn.shared_head_norm (else reuse output_norm)
    pub shared_head_head: Option<GpuTensor>, // blk.N.nextn.shared_head      (else reuse output)
    /// FR-Spec draft->target vocab map: the draft lm_head is TRIMMED to the highest-frequency
    /// tokens (e.g. 32768 rows of the full 248320-row head); `d2t[draft_idx]` = the target vocab
    /// token id of trimmed row `draft_idx`. `None` for a full-vocab head (identity map). Host-side:
    /// the draft argmax already lands on host as one u32, so the map is a single Vec index.
    pub d2t: Option<Vec<u32>>,
    /// True only when `MEMRA_FRSPEC_TRIM` gathered these rows from this target model's own
    /// output head. An external MTP draft may also carry `d2t`, but its head is a different
    /// student artifact and must never be borrowed for DFlash2 target-head trimming.
    pub d2t_from_target_head: bool,
    /// DISTILLED-STUDENT geometry (None = the natural NextN block at trunk shape). A distilled
    /// draft (StudentSV) runs the same block structure at a narrower inner width with fewer
    /// heads, then up-projects back to n_embd (`out_up`) — the chain carrier and the head input
    /// stay at n_embd, so the trunk/verify interface is unchanged. Selected by the presence of
    /// `blk.N.nextn.out_up.weight` in a MEMRA_MTP_DRAFT file.
    pub geom: Option<DraftGeom>,
    /// step35: the DRAFT BLOCK's RESOLVED per-layer geometry (`None` for every arch whose
    /// geometry is uniform). Without it the head forward would use the trunk's max-derived
    /// scalars and compute wrong attention — and the failure mode is plausible-but-wrong drafts
    /// (tanked acceptance, correct output), exactly what the exactness gates cannot see.
    pub step35: Option<Step35MtpGeom>,
}

/// step35 MTP-block geometry, RESOLVED at load time from the file that actually carries the
/// block's own `Step35Config` arrays.
///
/// Why resolved and not "look it up per forward from the model's cfg": Step-3.7-Flash ships MTP
/// as a SEPARATE GGUF, and the two files disagree about which layers exist. The trunk artifact
/// declares `block_count=45` / `nextn_predict_layers=0`, so its per-layer arrays hold 45 entries
/// (0..=44) and `Step35Config::n_head(45)` falls off the end into the `.last()` fallback — index
/// 44, which is a FULL-attn layer at 64 heads. The draft file declares `block_count=48` /
/// `nextn=3` and its arrays' index 45 is the truth: SWA, 96 heads (matching that file's
/// `blk.45.attn_q.weight [4096, 12288]` = 96*128 and `blk.45.attn_gate.weight [4096, 96]`).
/// Receipt: `research/step37-bringup-20260802/raw/gguf-header-stepfun-mtp-q8-20260802.txt` plus
/// the tail dump in `research/step37-p2-20260806/raw/` — `head_count[43..48] = [96, 64, 96, 96,
/// 96]`, `sliding_window_pattern[43..48] = [True, False, True, True, True]`.
#[derive(Debug, Clone)]
pub struct Step35MtpGeom {
    /// Block index inside the file that carries it (45 for Step-3.7-Flash). Diagnostics only.
    pub il: u32,
    pub n_head: usize,    // 96 on Step-3.7-Flash's MTP block (SWA-type)
    pub n_head_kv: usize, // 8
    pub n_rot: usize,     // 128 (SWA keeps the unhalved rotary width)
    pub rope_base: f32,   // 1e4 (SWA base, not the trunk's 5e6 global)
    pub swa: bool,        // true
    pub window: usize,    // 512
    /// This block's `swiglu_clamp_shexp` limit. The MTP block's FFN is a DENSE SwiGLU, and
    /// upstream's one `build_ffn` serves both the dense MLP and the shared expert off the
    /// SHEXP array (llama-graph.cpp:1751) — so a dense MTP block keys off shexp, not exp.
    /// 0.0 (`None`) on Step-3.7-Flash's block 45; live (16.0) only on trunk layers 43-44.
    pub clamp_shexp: Option<f32>,
}

impl Step35MtpGeom {
    /// Resolve a tuned MTP attention geometry from the canonical block that owns it.
    pub fn from_plan(layer: &memra_gguf::model_plan::LayerPlan) -> Result<Self, String> {
        use memra_gguf::model_plan::{ActivationPlan, AttentionPlan};

        let (attention, window) = match &layer.attention {
            AttentionPlan::Full(attention) => (attention, None),
            AttentionPlan::SlidingWindow { attention, window } => (attention, Some(*window)),
            other => {
                return Err(format!(
                    "MTP block {} has unsupported tuned attention {other:?}",
                    layer.index
                ));
            }
        };
        if attention.output_gate != memra_gguf::config::AttentionGateKind::SeparateHead {
            return Err(format!(
                "MTP block {} does not declare a separate attention gate",
                layer.index
            ));
        }
        let activation = match &layer.mlp {
            MlpPlan::Dense(dense) => &dense.activation,
            MlpPlan::Moe(moe) => &moe.activation,
        };
        let clamp_shexp = match activation {
            ActivationPlan::SwiGluClamped { limit } if *limit > 0.0 => Some(*limit),
            _ => None,
        };
        Ok(Step35MtpGeom {
            il: layer.index,
            n_head: attention.query_heads as usize,
            n_head_kv: attention.kv_heads as usize,
            n_rot: attention.rope.dimensions as usize,
            rope_base: attention.rope.base,
            swa: window.is_some(),
            window: window.unwrap_or(0) as usize,
            clamp_shexp,
        })
    }
}

/// Draft-head geometry override for a distilled (narrower) student block.
pub struct DraftGeom {
    pub d_inner: usize, // block inner width (eh_proj out / attn / ffn), e.g. 2048
    pub n_head: usize,  // draft attention heads (head_dim = main head_dim)
    pub n_head_kv: usize,
    pub out_up: GpuTensor, // [d_inner -> n_embd]: carrier + head input up-projection
}

/// Which tensor is the DRAFT lm_head, for a standalone NextN/MTP draft GGUF whose block index is
/// `n`. Preference order is the artifact's, not ours — upstream step35.cpp:553 is
/// `layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output`.
///
/// Split out of `MtpHead::load_draft` purely so it is unit-testable: the loader needs a CUDA
/// device and a multi-GB file, while the failure this guards is invisible to every exactness gate
/// (a wrong head still produces CORRECT output — the verify arbitrates — it just accepts nothing).
/// `has` is the tensor-presence predicate (`src.has`).
pub fn draft_head_tensor(has: impl Fn(&str) -> bool, n: u32) -> String {
    let own = format!("blk.{n}.nextn.shared_head_head.weight");
    if has(&own) {
        return own;
    }
    // Legacy name kept as a probe so anything that ever matched it still does; no shipped
    // artifact or upstream mapping uses it (see the `load_draft` note).
    let legacy = format!("blk.{n}.nextn.shared_head.weight");
    if has(&legacy) {
        return legacy;
    }
    // FR-Spec / tied-head drafts: the file-level head IS the draft head.
    "output.weight".to_string()
}

impl MtpHead {
    /// Load an MTP/NextN head from a STANDALONE draft GGUF (MEMRA_MTP_DRAFT override). The draft
    /// file carries ONLY the NextN block (blk.N.nextn.* glue + attn/ffn) plus its own lm_head
    /// (`output.weight`) — which for an FR-Spec draft is TRIMMED to the top-frequency rows, with
    /// a `d2t` (i32/i64) tensor mapping trimmed-row index -> target vocab token id. Draft-token
    /// embedding still uses the MAIN model's token_embd (identical weights, saves VRAM), so the
    /// draft file's full-vocab token_embd copy is ignored.
    pub fn load_draft(
        e: &Engine,
        g: &GgufFile,
        main_cfg: &ModelConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let src = GgufSource(g);
        let dcfg = src.try_config().map_err(std::io::Error::other)?;
        let draft_plan = match memra_gguf::model_packs::for_config(&dcfg) {
            Some(pack) => pack.compile_plan(&dcfg)?,
            None => memra_gguf::model_plan::ModelPlan::compile(&dcfg)?,
        };
        let main_plan = match memra_gguf::model_packs::for_config(main_cfg) {
            Some(pack) => pack.compile_plan(main_cfg)?,
            None => memra_gguf::model_plan::ModelPlan::compile(main_cfg)?,
        };
        // NextN block index INSIDE THE DRAFT FILE (its block_count includes the trunk numbering).
        // Graceful error, not assert: the server's `+draft` attach path surfaces this to the
        // user (a gemma-assistant draft or any non-NextN GGUF lands here; a panic killed the
        // whole worker — serve-smoke find, 2026-07-30).
        if dcfg.nextn_predict_layers == 0 {
            return Err(format!(
                "draft GGUF has no nextn_predict_layers (arch {:?}) — not a NextN/MTP regime \
                 draft; gemma assistant drafters attach via MEMRA_DRAFT, not '+draft'",
                g.arch()
            )
            .into());
        }
        let n = dcfg.n_layer - dcfg.nextn_predict_layers;
        let draft_block = draft_plan
            .mtp_blocks
            .iter()
            .find(|block| block.layer.index == n)
            .ok_or_else(|| format!("draft ModelPlan has no MTP block {n}"))?;
        let p = |s: &str| format!("blk.{n}.{s}");

        // Distilled student (narrow block + out_up) vs natural NextN clone. The interface dims
        // (n_embd in/out, head_dim for the shared rope kernel) must match the main model; a
        // student may shrink the inner width and head counts.
        let student = src.has(&p("nextn.out_up.weight"));
        assert_eq!(dcfg.n_embd, main_cfg.n_embd, "draft n_embd != model n_embd");
        assert_eq!(
            dcfg.head_dim_k, main_cfg.head_dim_k,
            "draft head_dim != model head_dim"
        );
        // step35: geometry is PER-LAYER, so "same shape as the trunk" is the wrong question — the
        // draft block at il=45 is an SWA-type block (96 q heads, 128 rotary dims, rope base 1e4)
        // while the trunk's full-attn layers are 64/64/5e6. Resolve the block's geometry from the
        // DRAFT FILE's own arrays (the trunk artifact's arrays stop at index 44 — see
        // `Step35MtpGeom`'s note) and verify it against the block's real tensor shapes. The dims
        // that must still agree with the trunk are the INTERFACE ones (n_embd, head_dim, KV width).
        let main_sliding_gated = crate::plan_backend::decode_batch_program(&main_plan)
            == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe;
        let draft_sliding_gated = crate::plan_backend::decode_batch_program(&draft_plan)
            == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe;
        let step35 = match (main_sliding_gated, draft_sliding_gated) {
            (true, true) => {
                let g = Step35MtpGeom::from_plan(&draft_block.layer)?;
                // ne is inner-fastest: ne[0] = in_features, ne[1] = out_features for a [in, out] 2D.
                let out_f = |t: &str| -> Option<usize> {
                    src.find(&p(t))
                        .and_then(|v| v.ne.get(1).copied())
                        .map(|x| x as usize)
                };
                let hd = dcfg.head_dim_k as usize;
                let wq_out =
                    out_f("attn_q.weight").ok_or("step35 draft block has no attn_q.weight")?;
                assert_eq!(
                    wq_out,
                    g.n_head * hd,
                    "step35 draft blk.{n}: attn_q out {wq_out} != n_head({}) * head_dim({hd}) — \
                     the draft file's head_count array disagrees with its own tensors",
                    g.n_head
                );
                // The SEPARATE head-wise gate is [n_embd, n_head_l] — one scalar per head. Its
                // width is the second independent witness of this block's head count.
                let wg_out = out_f("attn_gate.weight")
                    .ok_or("step35 draft block has no attn_gate.weight (head-wise gate)")?;
                assert_eq!(
                    wg_out, g.n_head,
                    "step35 draft blk.{n}: attn_gate out {wg_out} != n_head({})",
                    g.n_head
                );
                // The draft attends its OWN scratch, but `MtpScratch::new` sizes those rows from
                // the TRUNK cfg's `n_head_kv` (for step35, the max over its per-layer array).
                // Compare against exactly that value, not a per-layer accessor.
                assert_eq!(
                    g.n_head_kv, main_cfg.n_head_kv as usize,
                    "step35 draft blk.{n} KV heads {} != trunk n_head_kv {} — the MTP scratch \
                     rows are sized from the trunk cfg, so a differing draft KV width would \
                     write past the row",
                    g.n_head_kv, main_cfg.n_head_kv
                );
                eprintln!(
                    "[mtp-draft] step35 MTP geometry blk.{n}: n_head={} n_head_kv={} n_rot={} \
                     rope_base={:.0} swa={} window={}",
                    g.n_head, g.n_head_kv, g.n_rot, g.rope_base, g.swa, g.window
                );
                Some(g)
            }
            (true, false) => {
                return Err(format!(
                    "MEMRA_MTP_DRAFT operations are incompatible with the model's \
                     sliding-gated-MoE program (draft arch {:?})",
                    g.arch()
                )
                .into());
            }
            (false, true) => {
                return Err(
                    "MEMRA_MTP_DRAFT requires sliding-gated-MoE operations but the model does not"
                        .into(),
                );
            }
            (false, false) => None,
        };
        if step35.is_none() && !student {
            // The head forward runs with the MAIN model's cfg — the draft block must be the
            // same shape or the forward is garbage.
            assert_eq!(dcfg.n_head, main_cfg.n_head, "draft n_head != model n_head");
            assert_eq!(
                dcfg.n_head_kv, main_cfg.n_head_kv,
                "draft n_head_kv != model n_head_kv"
            );
        }

        // Draft lm_head. PREFERENCE ORDER IS THE ARTIFACT'S, NOT OURS (upstream step35.cpp:553
        // `layer.nextn.shared_head_head ? ... : model.output`): a NextN block owns its OWN head,
        // and only a file that omits it falls back to the file-level `output.weight`.
        //
        // MEASURED ON THE SHIPPED ARTIFACT (Step3.7-flash-mtp-Q8_0.gguf, byte hashes in
        // research/step37-p2-20260806/raw/draft-head-tensor-hashes-20260807.txt): the file carries
        // BOTH, they are DIFFERENT matrices, and the three MTP blocks' heads differ from each
        // other too —
        //     output.weight                        sha 3eec5831…  <- the TRUNK lm_head, re-quantized
        //     blk.45.nextn.shared_head_head.weight sha c90b907b…  <- block 45's own head
        //     blk.46 …                             sha a22d2957…
        //     blk.47 …                             sha 4b21e137…
        // The tell: this file's top-level `output_norm.weight` is BYTE-IDENTICAL to the trunk
        // artifact's (both sha d7526f44…), i.e. the top level is a copy of the trunk's output
        // stack, present so the draft gguf stands alone. Reading it as the draft head projects
        // the MTP block's hidden through the TRUNK's head — coherent-looking drafts the verify
        // never accepts. Receipt: acceptance 0/248 across K=1..8 with self-consistency PASS
        // (raw/mtp-draft-20260806T212902Z.log) — the exact failure class run_spec.rs's
        // "acceptance == 0 with identical output" WARNING exists to catch.
        //
        // FR-Spec drafts (trimmed [n_embd, draft_vocab] + d2t) publish the trimmed head as the
        // file-level `output.weight` and carry no `nextn.shared_head_head`, so they keep the
        // fallback — hence preference, not replacement.
        // Name choice is factored into `draft_head_tensor` so it is testable WITHOUT a GPU or a
        // 3.5 GB artifact (this whole function needs both). Getting it wrong is invisible to
        // every exactness gate, so the choice itself is pinned by a unit test.
        let head_name = draft_head_tensor(|t| src.has(t), n);
        let head = load_t(e, &src, &head_name)?;
        let head_norm = match load_opt(e, &src, &p("nextn.shared_head_norm.weight"))? {
            Some(t) => Some(t),
            None => load_opt(e, &src, "output_norm.weight")?,
        };

        // d2t: draft-row -> target-token-id map (absolute ids, verified against the tokenizer).
        let d2t: Option<Vec<u32>> = g.find("d2t").map(|t| {
            let bytes = g.tensor_data(t);
            match t.ggml_type {
                GgmlType::I32 => bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
                    .collect(),
                GgmlType::I64 => bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
                    .collect(),
                other => panic!("d2t must be I32/I64, got {other:?}"),
            }
        });
        if let Some(map) = &d2t {
            assert_eq!(
                map.len(),
                head.out_features(),
                "d2t len {} != draft head rows {}",
                map.len(),
                head.out_features()
            );
            let n_vocab = main_cfg.n_vocab as u64;
            assert!(
                map.iter().all(|&t| (t as u64) < n_vocab),
                "d2t contains token id >= model n_vocab {n_vocab}"
            );
        }
        let eh_proj = load_t(e, &src, &p("nextn.eh_proj.weight"))?;
        // defensive load gates (review feedback): a malformed student gguf fails HERE with a
        // named assert, not later as garbage drafts. eh_proj consumes concat(e_norm, h_norm).
        assert_eq!(
            eh_proj.in_features(),
            2 * main_cfg.n_embd as usize,
            "eh_proj in dim != 2*n_embd"
        );
        let geom = if student {
            let out_up = load_t(e, &src, &p("nextn.out_up.weight"))?;
            let d_inner = eh_proj.out_features();
            assert_eq!(
                out_up.out_features(),
                main_cfg.n_embd as usize,
                "out_up out dim != n_embd"
            );
            assert_eq!(
                out_up.in_features(),
                d_inner,
                "out_up in dim != eh_proj out dim (d_inner)"
            );
            assert!(
                dcfg.n_head >= 1 && dcfg.n_head_kv >= 1 && dcfg.n_head % dcfg.n_head_kv == 0,
                "student head counts malformed ({}/{})",
                dcfg.n_head,
                dcfg.n_head_kv
            );
            Some(DraftGeom {
                d_inner,
                n_head: dcfg.n_head as usize,
                n_head_kv: dcfg.n_head_kv as usize,
                out_up,
            })
        } else {
            None
        };
        // Log the name WITHOUT the blk.{n}. prefix (already printed) so the line reads
        // `source=nextn.shared_head_head` vs `source=output.weight` — the one-glance receipt
        // that the head choice went the right way on this artifact.
        let blk_prefix = format!("blk.{n}.");
        let head_src = head_name.strip_prefix(&blk_prefix).unwrap_or(&head_name);
        eprintln!(
            "[mtp-draft] external draft head: blk.{n}, source={}, head_vocab={}{}{}",
            head_src,
            head.out_features(),
            if d2t.is_some() {
                " (trimmed, d2t map)"
            } else {
                " (full)"
            },
            match &geom {
                Some(g) => format!(
                    " (student d_inner={} heads={}/{})",
                    g.d_inner, g.n_head, g.n_head_kv
                ),
                None => String::new(),
            }
        );

        let mut resident = ResidentPlan::unsharded(e, &src, &dcfg);
        let mut step_runtimes = StepParallelRuntimeRegistry::default();
        Ok(MtpHead {
            enorm: load_t(e, &src, &p("nextn.enorm.weight"))?,
            hnorm: load_t(e, &src, &p("nextn.hnorm.weight"))?,
            eh_proj,
            attn_norm: load_t(e, &src, &p("attn_norm.weight"))?,
            post_attn_norm: load_opt(e, &src, &p("post_attention_norm.weight"))?
                .or(load_opt(e, &src, &p("ffn_norm.weight"))?)
                .expect("draft NextN block needs post_attention_norm or ffn_norm"),
            mixer: load_mixer_kind(
                e,
                &src,
                &dcfg,
                n,
                &draft_block.layer.attention,
                &mut step_runtimes,
            )?,
            ffn: load_ffn(
                e,
                &src,
                &dcfg,
                &draft_block.layer.mlp,
                n,
                None,
                &mut resident,
                &mut step_runtimes,
            )?,
            shared_head_norm: head_norm,
            shared_head_head: Some(head),
            d2t,
            d2t_from_target_head: false,
            geom,
            step35,
        })
    }
}

/// gemma4 model-level auxiliaries.
pub struct GemmaAux {
    /// rope_freqs.weight [hd_global/2] freq factors — global layers' RoPE (R9).
    /// Keep one copy on every PP device: global layers on either side of the cut read it.
    pub rope_freqs: Option<Vec<(usize, CudaSlice<f32>)>>,
    /// all-ones norm weight [512] (max head_dim) — the weightless rms_norms (R7 V-norm).
    /// Keep one copy on every PP device: every full-attention layer reads it.
    pub ones: Vec<(usize, CudaSlice<f32>)>,
    /// tokenizer suppress_tokens uploaded once (None when the model ships none) — masked to
    /// -inf on every logits row before argmax/sampling (12B QAT ships two control ids).
    pub suppress_d: Option<(CudaSlice<i32>, usize)>,
    /// E4B per-layer-embedding model tensors (None on 26B/31B).
    pub e4b: Option<Gemma4E4bModel>,
}

impl GemmaAux {
    pub fn rope_freqs(&self, e: &Engine) -> Option<&CudaSlice<f32>> {
        self.rope_freqs.as_ref().map(|copies| {
            let dev = e.ctx().ordinal();
            &copies
                .iter()
                .find(|(d, _)| *d == dev)
                .unwrap_or_else(|| panic!("gemma4 rope_freqs has no local copy for device {dev}"))
                .1
        })
    }

    pub fn ones(&self, e: &Engine) -> &CudaSlice<f32> {
        let dev = e.ctx().ordinal();
        &self
            .ones
            .iter()
            .find(|(d, _)| *d == dev)
            .unwrap_or_else(|| panic!("gemma4 ones has no local copy for device {dev}"))
            .1
    }
}

/// step35 model-level auxiliaries. Deliberately NOT folded into `GemmaAux`: every gemma4 path
/// does `gemma4_aux.as_ref().unwrap()` and would then also fire on a step35 model.
pub struct Step35Aux {
    /// `rope_freqs.weight [n_rot_full/2]` llama3-style freq factors. Upstream applies them to
    /// FULL-attention layers ONLY (`rope_factors = is_swa ? nullptr : get_rope_factors(...)`,
    /// step35.cpp:246) — the SWA layers pass a null factor pointer. Step-3.7-Flash ships [64] F32.
    /// Keep one copy on every PP device: this model-level tensor is read by full-attention
    /// layers on both sides of the cut, and a primary-only copy would be a mapped peer read.
    pub rope_freqs: Option<Vec<(usize, CudaSlice<f32>)>>,
}

impl Step35Aux {
    pub fn rope_freqs(&self, e: &Engine) -> Option<&CudaSlice<f32>> {
        self.rope_freqs.as_ref().map(|copies| {
            let dev = e.ctx().ordinal();
            &copies
                .iter()
                .find(|(d, _)| *d == dev)
                .unwrap_or_else(|| panic!("step35 rope_freqs has no local copy for device {dev}"))
                .1
        })
    }
}

pub struct HybridModel {
    pub cfg: ModelConfig,
    pub plan: memra_gguf::model_plan::ModelPlan,
    pub rewrite_qualifications: Option<memra_gguf::execution_manifest::RewriteQualifications>,
    pub embd: EmbedHost,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<HybridLayer>,
    pub mtp: Option<MtpHead>, // NextN spec-decode head (None if nextn_predict_layers == 0)
    /// Additional embedded NextN heads, in trained draft-step order. Standalone and trimmed
    /// drafts remain single-head and leave this empty.
    pub mtp_extra: Vec<MtpHead>,
    /// The FR-Spec trimmed draft head for a DFlash2 round that has NO trimmed MtpHead to read:
    /// the MEMRA_MTP_SKIP=1 stub (embedded MTP block skipped), or the glm5 DFlash2 slab
    /// (drafter loaded, NextN block never loaded, lane/frspec-dflash2-20260902). `None` when
    /// MEMRA_FRSPEC_TRIM is unset; never co-exists with a target-head-trimmed `mtp`.
    pub dflash_trim: Option<DflashTrimHead>,
    /// sha16 of the MEMRA_FRSPEC_TRIM ranks artifact whichever trim arm consumed it (MtpHead
    /// self-trim, MEMRA_MTP_SKIP stub, glm5 DFlash2 slab); `None` = no trim loaded. Printed
    /// as `src=<sha16>` in the trim engagement lines.
    pub frspec_src_sha16: Option<String>,
    /// Lazily-uploaded DEVICE copy of the raw embed table (spec/graph hot loops gather rows
    /// on-device instead of host-dequant + htod). ~0.5GB; uploaded once on first use.
    pub embd_gpu: std::sync::OnceLock<cudarc::driver::CudaSlice<u8>>,
    /// device copy of the drafter's d2t trim map (uploaded once; `MEMRA_GLM5_SPEC_DEV_IO`).
    pub d2t_gpu: std::sync::OnceLock<cudarc::driver::CudaSlice<u32>>,
    pub gemma4_aux: Option<GemmaAux>,
    /// Sliding-gated-MoE tuned-program auxiliaries, selected from canonical operations.
    pub step35_aux: Option<Step35Aux>,
    /// PRIME ACTIVATION SLABS (piecewise-graph foundation, 2026-07-26): the layer loop's
    /// seven trunk transients live in RESIDENT per-model buffers instead of per-call pool
    /// allocs — kills ~224 alloc/free API calls per prime AND freezes the Lt GEMM operand
    /// addresses (nvjet's alignment-variant kernels become run-to-run stable once their
    /// pointers stop moving). Sized on first prime to the largest T seen. The map lock covers
    /// lookup/grow only; each device owns a separate slab lock so PP stages on distinct
    /// devices can drive their host-synchronized layer walks concurrently.
    pub prime_slabs: std::sync::Mutex<
        std::collections::HashMap<
            usize,
            std::sync::Arc<std::sync::Mutex<crate::hybrid_forward::PrimeSlabs>>,
        >,
    >,
    /// Engine-bundle slice 3 + graphs-serve lane: the dspark verify-graph POOL —
    /// per-(segment, vt) linear-run graphs and per-(vt, rung, hi) full-verify graphs,
    /// persistent ACROSS generations AND across serve sessions (the captured bodies are
    /// cache-independent — state is addressed through per-round-refreshed pointer
    /// tables and ctx-owned slabs/staging, so a fresh Cache — a new generation or a
    /// DIFFERENT session's — only changes table contents; keys carry nothing
    /// session-scoped). Rebuilding per call re-captured ~80 graphs per prompt (measured
    /// 97.8 -> 79.1 tok/s on the e2e pack); on the serve surface the capture toll
    /// amortizes at K≈33 requests (DSF-ROUNDCOST §9). Locked for the duration of one
    /// generate call (bin arm) or one session burst (serve arm — the slab stash is live
    /// verify->commit inside each round); single-engine contract like the draft graphs.
    /// Size policy: `crate::spec::dspark_vg_cap`.
    pub(crate) dspark_vgraphs: std::sync::Mutex<Option<crate::spec::DsparkVerifyGraphs>>,
    /// One lazily-sized grouped routed-expert prefill executor shared by every Step layer.
    ///
    /// The executor owns no checkpoint weights; each call supplies the current layer's resident
    /// expert banks and clamp policy. Keeping it model-scoped avoids multiplying the large
    /// capacity workspaces by the routed layer count.
    pub(crate) step_grouped_prefill: std::sync::Mutex<StepEpGroupedPrefill>,
    /// Whole-token decode graph state (step TP graph increment B): the stitched parent per fa
    /// bucket plus the persistent token/pos/logits plumbing. None until the door builds it.
    pub(crate) step35_token_graph:
        std::sync::Mutex<Option<crate::hybrid_forward::Step35TokenGraphState>>,
    /// mHC residual topology (`crate::hyper`), `Some` iff the compiled plan declares
    /// `ResidualTopology::HyperConnections` for the trunk. Every forward path keys on this:
    /// the ones that implement the hc program branch to it, and the ones that do not refuse
    /// through `refuse_hyper` rather than run a serial residual on an hc model.
    pub hyper: Option<crate::hyper::HyperTopology>,
    /// Gated-head exit weights, `Some` only for `HcCollapse::GatedHead`. The glm5_next collapse
    /// is an unweighted `Mean` and has no learned head.
    pub hyper_head: Option<crate::hyper::HyperHead>,
    /// glm5 DFlash2 alternate draft source (lane/glm5-dflash-draft-src, 2026-08-30):
    /// `MEMRA_GLM5_DFLASH=<dir-or-hf-spec>` loads the pinned block-diffusion drafter on the
    /// HEAD engine. When set it is THE draft source for `Glm5SpecSession` — the native MTP
    /// head is neither required nor loaded for it (the q38 pattern: a full MoE trunk layer
    /// of VRAM back). Owner holds written approval from the DFlash2 owners (2026-08-30)
    /// for use beyond probe/eval.
    pub glm5_dflash: Option<crate::glm_spec::Glm5DflashDrafter>,
    /// Measured PER-SESSION draft-graph state high-water, in bytes
    /// (lane/step37-vram-admission-20260830). Since the multi-head chain capture each
    /// capturing session parks real device state — capture-retain keepers, q slots, the
    /// instantiated graphs' backing memory — that admission used to charge at ZERO. The
    /// engine records the effective-free delta across a session's capture block here
    /// (high-water, self-measured — generic-model law: no per-family constant), and
    /// admission charges it per spec-capable session. 0 until the first capture is
    /// observed (the boot calibration probe usually supplies it).
    pub(crate) draft_state_bytes: std::sync::atomic::AtomicUsize,
    /// Test helper: extra device ordinals participating in this model for structural tests.
    pub test_extra_devices: Vec<usize>,
}

impl HybridModel {
    pub fn install_rewrite_bundle(
        &mut self,
        bundle: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.rewrite_qualifications = Some(
            memra_gguf::execution_manifest::RewriteQualifications::load(bundle, &self.plan)
                .map_err(|error| format!("rewrite qualification: {error}"))?,
        );
        Ok(())
    }

    pub fn rewrite_allowed(&self, surface: memra_gguf::execution_manifest::RewriteSurface) -> bool {
        self.rewrite_qualifications
            .as_ref()
            .is_none_or(|qualifications| qualifications.allows(surface))
    }

    /// Return the set of unique CUDA device ordinals touched by this model's resident
    /// weights, pipeline stages, tensor-parallel attention ranks, and expert-parallel banks.
    pub fn devices(&self) -> Vec<usize> {
        let mut devs = std::collections::BTreeSet::new();
        devs.insert(self.output_norm.ordinal());
        devs.insert(self.output.ordinal());

        for layer in &self.layers {
            devs.insert(layer.attn_norm.ordinal());
            devs.insert(layer.post_attn_norm.ordinal());

            match &layer.mixer {
                Mixer::Full(full) => {
                    devs.insert(full.wq.ordinal());
                    devs.insert(full.wk.ordinal());
                    devs.insert(full.wv.ordinal());
                    devs.insert(full.wo.ordinal());
                    if let Some(tp) = &full.step_tp_qkv {
                        devs.extend(&tp.devices);
                        devs.extend(tp.runtime.devices());
                    }
                }
                Mixer::Linear(linear) => {
                    devs.insert(linear.wqkv.ordinal());
                    devs.insert(linear.ssm_out.ordinal());
                }
                Mixer::Mla(mla) => {
                    devs.insert(mla.wo.ordinal());
                }
                Mixer::Kda(kda) => {
                    devs.insert(kda.wo.ordinal());
                }
            }

            match &layer.ffn {
                Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    // memra#253: this site inspects or moves weights and runs no GEMM on an
                    // activation, so the AWQ activation-side scale plays no part in it.
                    ffn_down_pqs: _,
                } => {
                    devs.insert(ffn_gate.ordinal());
                    devs.insert(ffn_up.ordinal());
                    devs.insert(ffn_down.ordinal());
                }
                Ffn::Moe(moe) => {
                    devs.insert(moe.gate_inp.ordinal());
                    if let Some(step_ep) = &moe.step_ep {
                        devs.extend(&step_ep.devices);
                        devs.extend(step_ep.runtime.devices());
                    }
                    if let Some(step_tp) = &moe.step_tp {
                        devs.extend(step_tp.runtime.devices());
                    }
                    if let Some(glm5_ep) = &moe.glm5_ep {
                        devs.extend(glm5_ep.rt.devices());
                    }
                }
            }

            if let Some(gemma4) = &layer.gemma4 {
                devs.insert(gemma4.ffn_norm.ordinal());
                devs.insert(gemma4.post_ffw_norm.ordinal());
            }
        }

        if let Ok(guard) = self.step_grouped_prefill.lock()
            && let Some(state) = &guard.state
        {
            devs.extend(&state.devices);
        }

        devs.extend(&self.test_extra_devices);

        devs.into_iter().collect()
    }

    /// Whether this model spans multiple CUDA devices structurally (via pipeline parallelism,
    /// tensor parallelism, or expert parallelism).
    pub fn is_multi_device(&self) -> bool {
        self.devices().len() > 1
    }

    /// Record an observed per-session draft-graph state size (bytes) — high-water only
    /// (lane/step37-vram-admission-20260830). Called by the spec capture block with the
    /// effective-free delta it measured across a session's captures. Returns the new
    /// high-water when it moved (so the caller can log the flip once, not per burst).
    pub fn record_draft_state_bytes(&self, observed: usize) -> Option<usize> {
        use std::sync::atomic::Ordering;
        let prev = self
            .draft_state_bytes
            .fetch_max(observed, Ordering::Relaxed);
        (observed > prev).then_some(observed)
    }

    /// Per-session draft-graph state admission charge, in bytes: the measured high-water
    /// (see [`Self::record_draft_state_bytes`]), 0 until a capture has been observed.
    /// Admission adds this to the SESSION cost of every spec-capable admit — it is
    /// per-session state (each capturing session parks its own keepers/q-slots/graphs),
    /// unlike the shared transient floor.
    pub fn draft_session_admission_bytes(&self) -> usize {
        self.draft_state_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Device-local bytes that are not yet materialized for this cache's rank-local Step KV.
    ///
    /// The owning-stage shadow cache remains allocated as the rollback oracle. Native Step
    /// attention lazily adds one sharded sidecar on every TP rank, so admission must reserve
    /// these bytes until the sidecar exists and live CUDA memory accounting can see it.
    pub fn step_tp_unmaterialized_kv_bytes(
        &self,
        cache: Option<&crate::cache::Cache>,
        capacity: usize,
    ) -> Result<Vec<StepTpKvDeviceAdmission>, String> {
        if let Some(cache) = cache
            && cache.tp_kv.len() < self.layers.len()
        {
            return Err(format!(
                "Step TP admission cache has {} layers, model trunk has {}",
                cache.tp_kv.len(),
                self.layers.len()
            ));
        }

        let mut by_device: HashMap<usize, usize> = HashMap::new();
        for (layer, weights) in self.layers.iter().enumerate() {
            let Mixer::Full(attention) = &weights.mixer else {
                continue;
            };
            let Some(tp) = attention
                .step_tp_qkv
                .as_ref()
                .filter(|tp| tp.attention.is_some())
            else {
                continue;
            };
            if cache.is_some_and(|cache| cache.tp_kv[layer].is_some()) {
                continue;
            }
            let geometry = self.cfg.full_attention_geometry_at(layer as u32);
            let shape = crate::cache::tp_kv_rank_allocation_shape(
                geometry.n_head_kv as usize * geometry.head_dim_k as usize,
                geometry.n_head_kv as usize * geometry.head_dim_v as usize,
                tp.devices.len(),
            )?;
            let physical_rows = geometry
                .window
                .map(|window| crate::cache::swa_ring_rows(window as usize, capacity))
                .unwrap_or(capacity);
            let bytes = shape.allocation_bytes(physical_rows);
            for &device in &tp.devices {
                let total = by_device.entry(device).or_default();
                *total = total.saturating_add(bytes);
            }
        }

        let mut out: Vec<_> = by_device
            .into_iter()
            .map(|(device, bytes)| StepTpKvDeviceAdmission { device, bytes })
            .collect();
        out.sort_unstable_by_key(|charge| charge.device);
        Ok(out)
    }

    /// One engine backed by the default memory pool that owns Step TP allocations on `device`.
    pub fn step_tp_rank_engine(&self, device: usize) -> Option<&Engine> {
        self.layers.iter().find_map(|weights| {
            let Mixer::Full(attention) = &weights.mixer else {
                return None;
            };
            let tp = attention.step_tp_qkv.as_ref()?;
            let rank = tp
                .runtime
                .devices()
                .iter()
                .position(|&rank| rank == device)?;
            tp.runtime.rank_engine(rank)
        })
    }

    pub(crate) fn step_tp_runtime_for_layer(
        &self,
        layer: usize,
    ) -> Option<&crate::tp::TpE4m3HostBounce> {
        let Mixer::Full(attention) = &self.layers.get(layer)?.mixer else {
            return None;
        };
        let tp = attention.step_tp_qkv.as_ref()?;
        tp.attention.as_ref()?;
        Some(tp.runtime.as_ref())
    }

    pub fn decode_batch_program(&self) -> crate::plan_backend::DecodeBatchProgram {
        crate::plan_backend::decode_batch_program(&self.plan)
    }

    pub fn uses_gemma_program(&self) -> bool {
        self.decode_batch_program() == crate::plan_backend::DecodeBatchProgram::Gemma
    }

    pub fn uses_sliding_gated_moe_program(&self) -> bool {
        self.decode_batch_program() == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe
    }

    pub fn has_plan_operation(&self, operation: memra_gguf::model_plan::OperationKind) -> bool {
        self.plan.trunk_operations().contains(&operation)
    }

    /// Load a hybrid (qwen35) model from GGUF. Thin byte-identical wrapper over `load_from_source`.
    pub fn load(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source(e, &GgufSource(g))
    }

    /// Plain-generation loader. `run-gen` never calls the optional draft head, so avoid loading
    /// its weights and expert bank while preserving the model config and all trunk semantics.
    pub fn load_without_mtp(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source_impl(e, &GgufSource(g), false)
    }

    /// Load a hybrid model from any `TensorSource` (GGUF or a safetensors HF checkpoint). The whole
    /// loop speaks ggml names; the source maps them (and, for safetensors, applies the SSM value
    /// transforms via the owned-buffer seam). The forward graph is untouched.
    pub fn load_from_source(
        e: &Engine,
        src: &dyn TensorSource,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source_impl(e, src, true)
    }

    /// Source-backed twin of `load_without_mtp`, used by the safetensors/repack `run-gen` path.
    pub fn load_from_source_without_mtp(
        e: &Engine,
        src: &dyn TensorSource,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source_impl(e, src, false)
    }

    fn load_from_source_impl(
        e: &Engine,
        src: &dyn TensorSource,
        load_mtp: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = src.try_config().map_err(std::io::Error::other)?;
        let plan = match memra_gguf::model_packs::for_config(&cfg) {
            Some(pack) => pack.compile_plan(&cfg)?,
            None => memra_gguf::model_plan::ModelPlan::compile(&cfg)?,
        };
        let auto_parallel = prepare_auto_parallel(src, &cfg, &plan)?;
        let batch_program = crate::plan_backend::decode_batch_program(&plan);
        let gemma_program = batch_program == crate::plan_backend::DecodeBatchProgram::Gemma;
        let sliding_gated_moe_program =
            batch_program == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe;
        if matches!(
            src.expert_activation_precision(),
            memra_gguf::source::ExpertActivationPrecision::Bf16
        ) {
            eprintln!(
                "[w4a16] artifact contract accepted: expert_weights=nvfp4 \
                 expert_activations=bf16-rounded q8_expert_program=disabled"
            );
        }
        // OWNER FLIP 2026-08-27: the gated step37 serving doors (t-row walk, W8 q8 mirrors, SWA
        // ring, NVFP4 draft heads, prejoin/head-rows/weight-once verify fixes) default ON for
        // this family. Armed HERE — before any tensor upload, cache sizing, or mirror build reads
        // a door — and only for the SlidingGatedMoe program; every door keeps its =0 kill switch.
        if sliding_gated_moe_program {
            crate::arm_step37_serving_defaults();
        }
        // Refuse an architecture that declares no attention output-gate layout, BEFORE any
        // tensor is uploaded or split. The old permissive default answered "qwen3.5 FusedQ" for
        // anything it did not recognize, and `q_gate_split` then read 2x past the end of a wq
        // whose gate is a separate tensor. An undeclared arch is a load error now, not a guess.
        cfg.validate_attention_gate_layout()?;
        // The host-expf probe guards HOST-oracle correctness, not the device arm: the device
        // top-k path never calls host expf at serve time (vendored scalar, deterministic), so
        // the device default must not fail-close on a rig whose libm merely differs. Hard-fail
        // only when the =0 host-oracle arm — the one whose served bytes depend on host libm —
        // is selected; the default arm logs a WARN so replay/oracle tooling knows host-side
        // comparisons are unavailable on this host.
        if cfg.sigmoid_router().is_some() {
            let host_oracle = std::env::var("MEMRA_SIG_ROUTER").as_deref() == Ok("0");
            match crate::sigrouter_contract::verify_host_expf() {
                Ok(()) => {}
                Err(e) if host_oracle => return Err(e.into()),
                Err(e) => eprintln!(
                    "[sigrouter] WARN: host expf probe mismatch ({e}); device routing is \
                     unaffected, but host-oracle replay/comparison cells are invalid on this host"
                ),
            }
        }
        // SPEC-SERVING stream-k key, per model, set at LOAD so it governs the PRIME too
        // (2026-07-27; explicit MEMRA_MMQ_SK wins). The former per-process timing selector
        // made knife-edge prime shapes BIMODAL across independent boots and was removed
        // 2026-08-14. Big dense (n_embd >= 3500) still forces tiling under spec intent;
        // MoE/small models defer to the deterministic fail-closed TILE form unless
        // MEMRA_MMQ_SK_FORM pins a separately measured arm.
        // An earlier attempt set this in generate_spec_gemma — too late, the prime's
        // GEMMs had already selected their form.
        if std::env::var("MEMRA_DRAFT").is_ok() && std::env::var("MEMRA_MMQ_SK").is_err() {
            let force = if cfg.n_embd >= 3500 { 0i8 } else { -1i8 };
            crate::MMQ_SK_FORCE.store(force, std::sync::atomic::Ordering::Relaxed);
        }
        // B0 FIX (hoisted): cfg.n_layer == block_count INCLUDES the MTP/NextN block(s)
        // (41 for the 35B-MoE); the trunk is n_layer - nextn. Computed before any tensor
        // upload because the M2 sharded loader (crate::pp::layer_engine) places tensors
        // by the trunk stage map.
        let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
        // MEMRA_MTP_SKIP=1 (mtp-skip lane, 2026-08-30): skip loading the embedded MTP/NextN
        // block(s) entirely (attention mixer, full FFN, and nextn glue), reclaiming their VRAM
        // on dspark-drafted deployments where the MTP spec arm is disabled anyway and the only
        // live consumer of the block is the FR-Spec trimmed rows (which for tied-head families
        // come from the TRUNK output.weight, not from blk.N tensors; see the stub further
        // down). Parsed and REFUSED here, before any tensor upload: every refusal below is
        // answerable from env + host metadata alone, so a config that cannot be honored fails
        // in seconds instead of after the full trunk load. Strict values only, refuse-loud on
        // anything else (the mis-typed-seam law).
        let mtp_skip_requested = load_mtp
            && match std::env::var("MEMRA_MTP_SKIP").ok().as_deref() {
                None | Some("") | Some("0") => false,
                Some("1") => true,
                Some(other) => {
                    return Err(format!(
                        "MEMRA_MTP_SKIP={other:?}: expected 1 (skip the embedded MTP block) or \
                         0/unset (load it); refusing to guess"
                    )
                    .into());
                }
            };
        if mtp_skip_requested && std::env::var("MEMRA_MTP_DRAFT").is_ok_and(|p| !p.is_empty()) {
            return Err(
                "MEMRA_MTP_SKIP=1 together with MEMRA_MTP_DRAFT is contradictory: the skip \
                 removes the MTP head to reclaim VRAM while MEMRA_MTP_DRAFT attaches an \
                 external MTP head for MTP spec decode; unset one"
                    .into(),
            );
        }
        if mtp_skip_requested && cfg.nextn_predict_layers > 0 {
            // Loud skip receipt with the approximate weight bytes NOT loaded. For a GGUF source
            // the figure is the exact on-disk size of every blk.{n_trunk..} tensor (VRAM cost is
            // approximately that, plus per-tensor upload overhead); a non-GGUF source has no
            // cheap tensor enumeration, so the line still prints, without a byte figure.
            let prefixes: Vec<String> = (0..cfg.nextn_predict_layers)
                .map(|off| format!("blk.{}.", n_trunk as u32 + off))
                .collect();
            let skipped_bytes: Option<u64> = src.gguf().map(|g| {
                g.tensors
                    .iter()
                    .filter(|t| prefixes.iter().any(|p| t.name.starts_with(p.as_str())))
                    .map(|t| t.n_bytes)
                    .sum()
            });
            eprintln!(
                "[mtp-skip] MEMRA_MTP_SKIP=1: skipping {} embedded MTP/NextN block(s) \
                 blk.{}..=blk.{} ({}); MTP spec decode is unavailable for this model \
                 (dspark/DFlash2 drafting keeps its trimmed head via the MEMRA_FRSPEC_TRIM stub)",
                cfg.nextn_predict_layers,
                n_trunk,
                n_trunk as u32 + cfg.nextn_predict_layers - 1,
                match skipped_bytes {
                    Some(b) => format!("~{} MiB of weights not loaded", b >> 20),
                    None => "size unknown: non-GGUF source".to_string(),
                },
            );
        }
        // MEMRA_MTP_SKIP x MEMRA_FRSPEC_TRIM admission: validate NOW (env + metadata + a host
        // file read), build the stub AFTER the trunk loads (it needs the engine). The parsed
        // d2t rides through `mtp_skip_trim_d2t` so the artifact is read once.
        //
        // REFUSAL TEETH (not warnings: a drafting config that cannot be honored must not
        // boot; the silent-no-op is the defect class this flag was designed against):
        // - the artifact ships its OWN per-block lm_head (step35-class): the trim rows live in
        //   the very block being skipped, and substituting trunk rows is the wrong-head bug
        //   with the banked acceptance-0/248 receipt (`frspec_trim_own_head_name`). FATAL.
        // - the trim artifact yields an empty d2t list: a stub dflash would silently filter
        //   out. FATAL.
        // - no output.weight/token_embd.weight to gather from. FATAL.
        // A model with NO declared NextN block keeps the trim's (b) behavior below: nothing to
        // skip, no stub, no refusal (a global env must not kill a co-loaded plain model).
        let mtp_skip_trim_d2t: Option<(Vec<u32>, String)> = if mtp_skip_requested
            && cfg.nextn_predict_layers > 0
            && !crate::model::full_prec_enabled()
        {
            match std::env::var("MEMRA_FRSPEC_TRIM") {
                Ok(path) if !path.is_empty() => {
                    let path = memra_gguf::hf::resolve_arg(&path)
                        .map_err(|err| format!("MEMRA_FRSPEC_TRIM={path:?}: {err}"))?;
                    let own_head_name = frspec_trim_own_head_name(n_trunk);
                    if src.has(&own_head_name) {
                        return Err(format!(
                            "MEMRA_MTP_SKIP=1 with MEMRA_FRSPEC_TRIM: this artifact ships its \
                             own MTP-block lm_head ({own_head_name}), so the trimmed draft rows \
                             live in the block being skipped; gathering trunk rows instead is \
                             the wrong-head bug (acceptance 0/248 receipt, \
                             frspec_trim_own_head_name). Unset MEMRA_MTP_SKIP or \
                             MEMRA_FRSPEC_TRIM"
                        )
                        .into());
                    }
                    if !src.has("output.weight") && !src.has("token_embd.weight") {
                        return Err("MEMRA_MTP_SKIP=1 with MEMRA_FRSPEC_TRIM: model has no \
                             output.weight (or tied token_embd.weight) to gather trimmed draft \
                             rows from"
                            .into());
                    }
                    let d2t = frspec_read_d2t(&path)?;
                    if d2t.is_empty() {
                        return Err(format!(
                            "MEMRA_MTP_SKIP=1 with MEMRA_FRSPEC_TRIM={path}: the rank artifact \
                             yields an EMPTY d2t list, so no stub draft head can be built; fix \
                             the artifact or unset MEMRA_MTP_SKIP"
                        )
                        .into());
                    }
                    let sha16 = sha256_file_hex(std::path::Path::new(&path), 8)?;
                    Some((d2t, sha16))
                }
                _ => None,
            }
        } else {
            None
        };
        // The ranks artifact's identity, set by whichever trim arm consumed the file (the
        // MtpHead self-trim, the MEMRA_MTP_SKIP stub, or the glm5 DFlash2 slab below); the
        // session engagement line prints it as `src=<sha16>`.
        let mut frspec_src_sha16: Option<String> =
            mtp_skip_trim_d2t.as_ref().map(|(_, s)| s.clone());
        if let Some(fence) = crate::pp::pp_cuts(n_trunk) {
            let pipeline = crate::plan_backend::PIPELINE
                .trunk_capabilities(&plan)
                .pipeline;
            // Gemma retains its separately gated PP2 program. Every generic PP-N load must be
            // admitted by ModelPlan operations before the first shard is uploaded; a legacy env
            // door is not evidence that an arbitrary dense/stateful architecture is splittable.
            let qualified_gemma_pp2 = gemma_program && fence.len() == 3;
            if !pipeline.supported && !qualified_gemma_pp2 {
                return Err(format!(
                    "pipeline placement is unsupported for plan operations {:?}; blockers={:?}",
                    plan.trunk_operations(),
                    pipeline.blockers,
                )
                .into());
            }
            let illegal = illegal_pipeline_cuts(&fence, &plan.partition_boundaries);
            if !illegal.is_empty() {
                return Err(format!(
                    "pipeline placement cuts {illegal:?} split outside ModelPlan legal boundaries {:?}",
                    plan.partition_boundaries,
                )
                .into());
            }
        }
        crate::pp::init_model_transport(e, &cfg, n_trunk)?;
        let step_parallel =
            prepare_step_parallel_load(e, src, &cfg, n_trunk, auto_parallel.as_ref())?;
        // glm5 TP-2 door (MEMRA_GLM5_TP): structural preflight from the compiled plan, BEFORE
        // any TP CUDA state or shard exists. Illegal geometry, non-glm5 plans, and co-armed
        // parallel programs refuse here by name.
        let glm5_tp = if crate::glm5_tp::glm5_tp_armed() {
            use memra_gguf::model_plan::{AttentionPlan, MlpPlan};
            let moe = cfg.moe.as_ref().ok_or(
                "MEMRA_GLM5_TP requires a MoE model (glm5_next); this plan carries no MoE \
                 metadata",
            )?;
            let mut layer_class = Vec::with_capacity(n_trunk);
            let mut layer_is_moe = Vec::with_capacity(n_trunk);
            let (mut kda_heads, mut kda_head_dim, mut mla_heads) = (0usize, 0usize, 0usize);
            for (il, lp) in plan.layers.iter().take(n_trunk).enumerate() {
                match &lp.attention {
                    AttentionPlan::KimiDeltaNet(k) => {
                        layer_class.push(crate::glm5_tp::Glm5LayerClass::Kda);
                        kda_heads = k.num_heads as usize;
                        kda_head_dim = k.head_dim as usize;
                    }
                    AttentionPlan::Mla(memra_gguf::model_plan::MlaAttentionPlan::LatentKv {
                        query_heads,
                        ..
                    }) => {
                        layer_class.push(crate::glm5_tp::Glm5LayerClass::Mla);
                        mla_heads = *query_heads as usize;
                    }
                    other => {
                        return Err(format!(
                            "MEMRA_GLM5_TP requires a glm5_next-class plan (KDA/MLA mixers): \
                             trunk layer {il} declares {other:?}"
                        )
                        .into());
                    }
                }
                layer_is_moe.push(matches!(&lp.mlp, MlpPlan::Moe(_)));
            }
            let view = crate::glm5_tp::Glm5TpModelView {
                trunk_layers: n_trunk,
                layer_class,
                layer_is_moe,
                kda_heads,
                kda_head_dim,
                mla_heads,
                n_routed_experts: moe.expert_count as usize,
                top_k: moe.expert_used_count as usize,
            };
            crate::glm5_tp::prepare_glm5_tp_load(e, &view)?
        } else {
            // FAIL-CLOSED: a measured placement map on a glm5-class plan with the TP door
            // COLD would silently serve the even split while the operator believes the
            // map is live — exactly the trap LAW:coactivation-expert-placement's rollout
            // discipline forbids. Scoped to glm5-class plans (KDA mixers present) so a
            // co-loaded non-glm5 model never trips it (the MEMRA_FRSPEC_TRIM global-flag
            // lesson).
            let glm5_class = plan.layers.iter().take(n_trunk).any(|lp| {
                matches!(
                    lp.attention,
                    memra_gguf::model_plan::AttentionPlan::KimiDeltaNet(_)
                )
            });
            let ep_map_armed = crate::ep_map::ep_map_env()?;
            if let Some((flag, _)) = ep_map_armed
                && glm5_class
            {
                return Err(format!(
                    "{flag} is set but MEMRA_GLM5_TP is off: the map cannot \
                     engage, and a placement that silently reverts to the even split is \
                     refused by name (unset one of the two)"
                )
                .into());
            }
            // Same trap, same scope, for the EP dispatch-diet doors (lane/glm5-ep-diet):
            // an ENABLED diet flag on a glm5-class plan with the TP door cold would
            // silently run the plain walk while the operator believes the diet is live.
            // `=0` is a deliberate pin, not an arming, and never refuses.
            // The doors resolve through the general name + its glm5 alias, and report the
            // name the OPERATOR set — so this refusal's bytes are unchanged for every banked
            // script (which sets the alias) and correct for the general name.
            if glm5_class {
                for (armed, flag) in [crate::ep_diet_armed(), crate::ep_grouped_prime_armed()] {
                    if armed {
                        return Err(format!(
                            "{flag}=1 is set but MEMRA_GLM5_TP is off: the EP dispatch \
                             diet only exists inside the TP-2 EP walk and cannot engage \
                             (unset one of the two)"
                        )
                        .into());
                    }
                }
            }
            None
        };
        let embd = EmbedHost::from_source(src, "token_embd.weight");
        // M2 increment 2 (weight sharding): output_norm + lm head upload through the LAST
        // stage's engine — the stage that runs them (outside the pp door / MEMRA_PP_SHARD=0
        // this is the primary engine, byte-identical to the M1 loader).
        let e_head = crate::pp::layer_engine(e, n_trunk, n_trunk - 1)?;
        let output_norm = load_t(e_head, src, "output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent.
        let mut output = if src.has("output.weight") {
            load_t(e_head, src, "output.weight")?
        } else {
            load_t(e_head, src, "token_embd.weight")?
        };
        let mut resident = ResidentPlan::pp(e, src, &cfg, n_trunk)?;
        resident.exclude_distributed_expert_layers(
            step_parallel
                .ep_specs
                .iter()
                .map(|spec| spec.layer)
                .chain(step_parallel.tp_specs.iter().map(|spec| spec.layer)),
        );
        let mut step_runtimes = StepParallelRuntimeRegistry::with_config(step_parallel);

        // SPILLING-PLAN §2: build the tiered-spill context ONCE, before loading any experts, but
        // only for a MoE model with the disk tier forced on (`MEMRA_SPILL_DISK`). It probes free VRAM
        // + host RAM at runtime (never hardcoded) and opens one shared GGUF mmap; all expert tensors
        // draw down its single pinned-RAM budget (hottest pinned, the rest mmap'd from disk). When
        // unset/dense this stays `None` and the load takes the byte-identical all-host path.
        // Disk spill is GGUF-only (needs the on-disk file mmap); src.gguf() is None for safetensors.
        let gguf: Option<&GgufFile> = src.gguf();
        // The normalized config carries `moe` only for a positive expert bank. Keep the explicit
        // count check as a fail-closed guard against hand-built configs.
        let mut spill: Option<crate::spill::SpillCtx> = if cfg
            .moe
            .as_ref()
            .is_some_and(|m| m.expert_count > 0)
            && crate::spill::disk_tier_enabled()
            && gguf.is_some()
        {
            let budget = crate::spill::MemBudget::probe(e)?;
            #[allow(clippy::unnecessary_unwrap)]
            // allow: the Some-guard sits in a multi-clause regime gate; if-let would reshape the arm structure
            let ctx = crate::spill::SpillCtx::open(gguf.unwrap(), &budget)?;
            eprintln!(
                "[spill] disk tier ON: free_vram={} MiB  free_pinnable_ram={} MiB (MemAvailable*resolved_frac)",
                budget.free_vram >> 20,
                budget.free_pinnable_ram >> 20
            );
            Some(ctx)
        } else {
            None
        };

        // Running the MTP block as a trunk layer is wrong; iterate only the trunk layers
        // (n_trunk hoisted above). 9B (nextn=0): n_trunk = 32. 35B-MoE (nextn=1): 40.

        // mHC residual topology (crate::hyper). Derived from the compiled plan BEFORE any layer
        // is built, and uniform across the trunk by construction — the stream state is one shape
        // for the whole stack, so a per-layer disagreement is a load error, not a per-layer arm.
        let hyper = crate::hyper::HyperTopology::from_plan(&plan)?;
        let hyper_head = match hyper.as_ref() {
            Some(topology) => {
                crate::hyper::HyperHead::load(e_head, src, topology, cfg.n_embd as usize)?
            }
            None => None,
        };
        let mut layers = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk as u32 {
            let p = |s: &str| format!("blk.{il}.{s}");
            let layer_plan = plan
                .layers
                .get(il as usize)
                .ok_or_else(|| format!("ModelPlan has no trunk layer {il}"))?;
            // M2 weight sharding: this layer's tensors upload through the OWNING stage's
            // engine (shadowed `e`) — the bring-up remote peer-read placement dies here.
            // Door shut / MEMRA_PP_SHARD=0: `layer_engine` returns the primary (no change).
            let e = crate::pp::layer_engine(e, n_trunk, il as usize)?;
            // attn_norm always; post_attention_norm is the pre-FFN norm in qwen35
            layers.push(HybridLayer {
                attn_norm: load_t(e, src, &p("attn_norm.weight"))?,
                post_attn_norm: load_opt(e, src, &p("post_attention_norm.weight"))?
                    .or(load_opt(e, src, &p("ffn_norm.weight"))?)
                    .expect("need post_attention_norm or ffn_norm"),
                mixer: {
                    // E4B KV-shared layers ship NO attn_k/attn_v — load the SHARE TARGET's
                    // k/v tensors for shape symmetry (forward skips k/v compute there and
                    // reads the target layer's cache; see Gemma4E4bLayer::kv_share).
                    let g4_shared = cfg.gemma4.as_ref().map(|g| g.shared_kv_layers).unwrap_or(0);
                    let kv_from = n_trunk as u32 - g4_shared;
                    if g4_shared > 0
                        && il >= kv_from
                        && !src.has(&format!("blk.{il}.attn_k.weight"))
                    {
                        let g4 = cfg.gemma4.as_ref().unwrap();
                        let swa = g4.swa_pattern.get(il as usize).copied().unwrap_or(true);
                        let tgt = kv_from - if swa { 2 } else { 1 };
                        let tp = |s: &str| format!("blk.{tgt}.{s}");
                        Mixer::Full(FullAttnLayer {
                            wq: load_t(e, src, &p("attn_q.weight"))?,
                            wk: load_t(e, src, &tp("attn_k.weight"))?,
                            wv: load_t(e, src, &tp("attn_v.weight"))?,
                            wo: load_t(e, src, &p("attn_output.weight"))?,
                            wo_pqs: load_opt(e, src, &p("attn_output.pre_quant_scale"))?,
                            // gemma4 shared-KV layers: the family always ships QK-norm, so
                            // these stay required — the Option is for families that have none.
                            q_norm: Some(load_t(e, src, &p("attn_q_norm.weight"))?),
                            k_norm: Some(load_t(e, src, &tp("attn_k_norm.weight"))?),
                            attn_gate: None, // gemma4 has no separate head-wise gate
                            step_tp_qkv: None,
                        })
                    } else {
                        load_mixer_kind(
                            e,
                            src,
                            &cfg,
                            il,
                            &layer_plan.attention,
                            &mut step_runtimes,
                        )?
                    }
                },
                ffn: load_ffn(
                    e,
                    src,
                    &cfg,
                    &layer_plan.mlp,
                    il,
                    spill.as_mut().map(|c| (gguf.unwrap(), c)),
                    &mut resident,
                    &mut step_runtimes,
                )?,
                gemma4: if gemma_program {
                    let scalar = |n: &str| -> f32 {
                        let t = src.find(&p(n)).unwrap_or_else(|| panic!("missing {n}"));
                        memra_gguf::dequant::dequantize(t.ggml_type, &t.bytes, 1)[0]
                    };
                    let vecf = |n: &str| -> Vec<f32> {
                        let t = src.find(&p(n)).unwrap_or_else(|| panic!("missing {n}"));
                        memra_gguf::dequant::dequantize(
                            t.ggml_type,
                            &t.bytes,
                            t.ne.iter().product::<u64>() as usize,
                        )
                    };
                    let moe_bits = if src.find(&p("ffn_gate_inp.scale")).is_some() {
                        Some(crate::hybrid::Gemma4MoeBits {
                            post_ffw_norm_1: load_t(e, src, &p("post_ffw_norm_1.weight"))?,
                            pre_ffw_norm_2: load_t(e, src, &p("pre_ffw_norm_2.weight"))?,
                            post_ffw_norm_2: load_t(e, src, &p("post_ffw_norm_2.weight"))?,
                            shared_gate: load_t(e, src, &p("ffn_gate.weight"))?,
                            shared_up: load_t(e, src, &p("ffn_up.weight"))?,
                            shared_down: load_t(e, src, &p("ffn_down.weight"))?,
                            router_scale_pre: {
                                let inv = 1.0 / (cfg.n_embd as f32).sqrt();
                                let v: Vec<f32> =
                                    vecf("ffn_gate_inp.scale").iter().map(|x| x * inv).collect();
                                e.htod(&v)?
                            },
                            per_expert_scale: vecf("ffn_down_exps.scale"),
                            per_expert_scale_d: e.htod(&vecf("ffn_down_exps.scale"))?,
                        })
                    } else {
                        None
                    };
                    // E4B extras (tensor-presence: blk.N.inp_gate only exists on E4B)
                    let e4b = if src.has(&p("inp_gate.weight")) {
                        let g4 = cfg.gemma4.as_ref().unwrap();
                        let kv_from = n_trunk as u32 - g4.shared_kv_layers;
                        let kv_share = if g4.shared_kv_layers > 0 && il >= kv_from {
                            let swa = g4.swa_pattern.get(il as usize).copied().unwrap_or(true);
                            Some(kv_from - if swa { 2 } else { 1 })
                        } else {
                            None
                        };
                        Some(crate::hybrid::Gemma4E4bLayer {
                            inp_gate: load_t(e, src, &p("inp_gate.weight"))?,
                            proj: load_t(e, src, &p("proj.weight"))?,
                            post_norm: load_t(e, src, &p("post_norm.weight"))?,
                            kv_share,
                            qkv_cat: None, // built at the mirror hook (wave 4b)
                        })
                    } else {
                        None
                    };
                    Some(Gemma4LayerBits {
                        ffn_norm: load_t(e, src, &p("ffn_norm.weight"))?,
                        post_ffw_norm: load_t(e, src, &p("post_ffw_norm.weight"))?,
                        moe_bits,
                        layer_scale: scalar("layer_output_scale.weight"),
                        e4b,
                    })
                } else {
                    None
                },
                hyper: match hyper.as_ref() {
                    Some(topology) => Some(crate::hyper::HyperLayer::load(
                        e,
                        src,
                        il,
                        topology,
                        cfg.n_embd as usize,
                    )?),
                    None => None,
                },
                tp_glue: Vec::new(),
            });
            // glm5 TP-2 arming: shard the just-loaded layer in place. Transient VRAM is one
            // layer's full weights (the shards replace them before the next layer loads).
            if let Some(tp_plan) = &glm5_tp
                && tp_plan.layers.contains(&(il as usize))
            {
                let mut layer = layers.pop().expect("layer just pushed");
                layer.mixer = match layer.mixer {
                    Mixer::Kda(la) => {
                        Mixer::Kda(crate::glm5_tp::shard_kda_layer(e, &tp_plan.rt, la)?)
                    }
                    Mixer::Mla(la) => {
                        Mixer::Mla(crate::glm5_tp::shard_mla_layer(e, &tp_plan.rt, la)?)
                    }
                    _ => {
                        return Err(format!(
                            "MEMRA_GLM5_TP selected layer {il}, whose loaded mixer is not \
                             KDA/MLA — preflight and loader disagree (wiring bug)"
                        )
                        .into());
                    }
                };
                if crate::glm5_tp::glm5_tp_symmetric_on()
                    && let Some(hyper) = layer.hyper.as_ref()
                {
                    let moe = match &layer.ffn {
                        Ffn::Moe(m) => Some(m),
                        _ => None,
                    };
                    layer.tp_glue = crate::glm5_tp::replicate_layer_glue(
                        e,
                        &tp_plan.rt,
                        &layer.attn_norm,
                        &layer.post_attn_norm,
                        hyper,
                        moe,
                    )?;
                }
                if let Ffn::Moe(m) = &mut layer.ffn {
                    // The measured placement row for this layer, when MEMRA_EP_MAP (or
                    // its glm5 alias) armed one (validated at preflight: exact layer cover, so a
                    // missing row here is a wiring bug, never a silent even split).
                    let placement = match &tp_plan.ep_map {
                        Some(map) => Some(
                            map.layers
                                .get(&(il as usize))
                                .ok_or_else(|| {
                                    format!(
                                        "glm5-tp EP: preflight-validated map lost layer {il} \
                                         (wiring bug)"
                                    )
                                })?
                                .as_slice(),
                        ),
                        None => None,
                    };
                    crate::glm5_tp::arm_moe_ep(e, &tp_plan.rt, m, placement)?;
                }
                layers.push(layer);
            }
        }

        // Embedded artifacts may carry multiple trained NextN blocks. Preserve their declared
        // order; the speculative driver decides whether it can serve a chain. A missing first
        // block still means "external draft", while a hole inside a declared chain is malformed.
        let external_mtp_requested =
            load_mtp && std::env::var("MEMRA_MTP_DRAFT").is_ok_and(|path| !path.is_empty());
        let trim_mtp_requested = load_mtp
            && !crate::model::full_prec_enabled()
            && std::env::var("MEMRA_FRSPEC_TRIM").is_ok_and(|path| !path.is_empty());
        // The `trim_mtp_requested => 1` branch is gone, and both lanes wanted it gone:
        // (a) since the per-head trim (2026-08-27) every loaded head gathers its OWN block's
        //     trimmed rows, so a trim no longer costs the chain — MEMRA_MTP_HEADS is the only
        //     chain-width knob; and
        // (b) a model with no trained NextN block (nextn_predict_layers == 0) can never satisfy
        //     a trim request, and forcing head_count = 1 made a GLOBAL MEMRA_FRSPEC_TRIM fatal
        //     for every co-loaded plain model ("ModelPlan has no embedded MTP block") — e.g. an
        //     embedding model beside a spec'd chat model.
        // With the branch removed a headless model simply takes nextn_predict_layers = 0 and
        // loads plain, which is (b)'s fix by construction.
        let _ = trim_mtp_requested;
        // MEMRA_GLM5_MTP (default OFF): glm5_next's NextN block loads only when asked. The
        // artifact carries the full MTP layer (a MoE block the size of a trunk layer — 288
        // routed experts), and until 2026-08-30 the `nextn.*` glue names had no glm5_next
        // ggml->HF mapping row, so the head silently never loaded and nothing downstream
        // ever saw one. With the mapping fixed, loading it unconditionally would add a
        // trunk-layer's VRAM and load time to every glm5 serve with NOTHING consuming it
        // yet (the spec entry points refuse hc trunks; the MTP_SPEC capability manifest
        // reports unsupported for this plan, so the worker never routes to it). Default OFF
        // keeps prod byte-identical; the MTP draft gate and the verify arc opt in.
        let glm5_mtp_requested =
            !cfg.arch.is_glm5_next() || std::env::var("MEMRA_GLM5_MTP").as_deref() == Ok("1");
        // (MEMRA_MTP_SKIP was parsed and refusal-checked right after n_trunk, before any
        // tensor upload; here it only zeroes the embedded chain.)
        let embedded_head_count =
            if external_mtp_requested || !glm5_mtp_requested || mtp_skip_requested {
                0
            } else {
                cfg.nextn_predict_layers
            };
        if cfg.arch.is_glm5_next()
            && glm5_mtp_requested
            && !mtp_skip_requested
            && cfg.nextn_predict_layers > 0
        {
            eprintln!("[mtp-glm5] MEMRA_GLM5_MTP=1: loading the glm5_next NextN block");
        }
        // MEMRA_MTP_HEADS=N caps the embedded chain. It exists so the FR-Spec trim can be
        // measured HONESTLY: a trim forces the chain down to one head, so trimmed-vs-untrimmed
        // otherwise mixes the trim's effect with the loss of the chain. With this, the A/B is
        // 3-head untrimmed -> 1-head untrimmed -> 1-head trimmed and each step is attributable.
        let embedded_head_count = match std::env::var("MEMRA_MTP_HEADS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0)
        {
            Some(cap) if cap < embedded_head_count => {
                eprintln!(
                    "[mtp-chain] MEMRA_MTP_HEADS={cap}: capping the embedded chain from \
                     {embedded_head_count} heads (measurement knob)"
                );
                cap
            }
            _ => embedded_head_count,
        };
        let mut embedded_mtp = Vec::new();
        if load_mtp && embedded_head_count > 0 {
            for offset in 0..embedded_head_count {
                let n = n_trunk as u32 + offset;
                // M2 weight sharding: MTP/NextN blocks live past the trunk fence and
                // `layer_engine` maps them to the LAST stage — the stage that runs the
                // draft chain (glm_spec's head-engine contract) and holds the trunk lm
                // head the draft projects through. Door shut / MEMRA_PP_SHARD=0 /
                // devices unset: the primary, byte-identical to the previous load.
                let e = crate::pp::layer_engine(e, n_trunk, n as usize)?;
                let p = |s: &str| format!("blk.{n}.{s}");
                let mtp_plan = plan
                    .mtp_blocks
                    .iter()
                    .find(|block| block.layer.index == n)
                    .ok_or_else(|| format!("ModelPlan has no embedded MTP block {n}"))?;
                if !src.has(&p("nextn.eh_proj.weight")) {
                    if offset == 0 {
                        break;
                    }
                    return Err(format!(
                        "embedded MTP chain declares {} heads but blk.{n} has no \
                         nextn.eh_proj.weight",
                        cfg.nextn_predict_layers
                    )
                    .into());
                }
                embedded_mtp.push(MtpHead {
                    enorm: load_t(e, src, &p("nextn.enorm.weight"))?,
                    hnorm: load_t(e, src, &p("nextn.hnorm.weight"))?,
                    eh_proj: load_t(e, src, &p("nextn.eh_proj.weight"))?,
                    attn_norm: load_t(e, src, &p("attn_norm.weight"))?,
                    post_attn_norm: load_opt(e, src, &p("post_attention_norm.weight"))?
                        .or(load_opt(e, src, &p("ffn_norm.weight"))?)
                        .expect("MTP block needs post_attention_norm or ffn_norm"),
                    mixer: load_mixer_kind(
                        e,
                        src,
                        &cfg,
                        n,
                        &mtp_plan.layer.attention,
                        &mut step_runtimes,
                    )?,
                    ffn: load_ffn(
                        e,
                        src,
                        &cfg,
                        &mtp_plan.layer.mlp,
                        n,
                        spill.as_mut().map(|c| (gguf.unwrap(), c)),
                        &mut resident,
                        &mut step_runtimes,
                    )?,
                    shared_head_norm: load_opt(e, src, &p("nextn.shared_head_norm.weight"))?,
                    // `nextn.shared_head_head` is the name the convert script and upstream both
                    // use (LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD -> "blk.%d.nextn.shared_head_head");
                    // `nextn.shared_head` is a name no shipped artifact carries, so this arm was
                    // silently always-None and every embedded-MTP model fell back to the trunk
                    // `self.output` in `mtp_head_forward_dev` op 12. Harmless for qwen35-family
                    // heads that genuinely tie to the trunk head; wrong for any artifact that
                    // ships its own — which the StepFun step35 drafter does (see `load_draft`).
                    // Keep the old name as a fallback so nothing that did match still does.
                    shared_head_head: load_mtp_head_maybe_nvfp4(
                        e,
                        src,
                        &p("nextn.shared_head_head.weight"),
                    )?
                    .or(load_opt(e, src, &p("nextn.shared_head.weight"))?),
                    d2t: None,
                    d2t_from_target_head: false,
                    geom: None,
                    step35: if sliding_gated_moe_program {
                        Some(Step35MtpGeom::from_plan(&mtp_plan.layer)?)
                    } else {
                        None
                    },
                });
            }
        }
        let mut embedded_mtp = embedded_mtp.into_iter();
        let mut mtp = embedded_mtp.next();
        let mut mtp_extra: Vec<MtpHead> = embedded_mtp.collect();

        // MEMRA_MTP_DRAFT=<path.gguf>: REPLACE the MTP head with one loaded from a standalone
        // draft GGUF (e.g. an FR-Spec trimmed-vocab draft). Verify-based spec decode stays exact
        // regardless of the draft — a different draft only changes WHICH tokens get proposed.
        mtp = if load_mtp {
            match std::env::var("MEMRA_MTP_DRAFT") {
                Ok(path) if !path.is_empty() => {
                    eprintln!("[mtp-draft] loading external MTP draft: {path}");
                    let dg = GgufFile::open(&path)?;
                    mtp_extra.clear();
                    Some(MtpHead::load_draft(e, &dg, &cfg)?)
                }
                _ => mtp,
            }
        } else {
            None
        };

        // MEMRA_FRSPEC_TRIM=<frspec.gguf>: SELF-TRIMMED draft head. Reads ONLY the d2t ranked-token
        // list from the given file and gathers those rows from the MAIN model's own output.weight
        // bytes (quantized rows are independent — a byte-level row gather, zero requant). The MTP
        // block, norms, and head quant all stay main-model, so there is no cross-file quality
        // mismatch (the external Q4_K draft file measured -15pts acceptance vs the native block).
        // Draft lm_head reads drop vocab/32768-fold; verify stays full-vocab -> exactness unchanged.
        // FULL_PREC (MTP-heal ceiling): the self-trim gathers rows into `from_quant_bytes` (Quant
        // only) and, more to the point, the full-precision ceiling wants the model's NATURAL full
        // head — trimming the draft vocab is a speed lever, not part of the exactness measurement.
        // Disable trim under the flag (documented resolution, §item 2).
        let trim_env = if load_mtp {
            std::env::var("MEMRA_FRSPEC_TRIM")
        } else {
            Err(std::env::VarError::NotPresent)
        };
        if crate::model::full_prec_enabled()
            && trim_env.as_deref().map(|p| !p.is_empty()).unwrap_or(false)
        {
            eprintln!(
                "[frspec-trim] DISABLED under MEMRA_FULL_PREC — using the natural full MTP head"
            );
        }
        mtp = match (
            if crate::model::full_prec_enabled() {
                Err(std::env::VarError::NotPresent)
            } else {
                trim_env
            },
            mtp,
        ) {
            (Ok(path), Some(mut head)) if !path.is_empty() => {
                // The trimmed head is consumed by the draft chain on the LAST stage's
                // engine (same placement as the embedded block above); shadow `e` so
                // every gathered-row upload below lands there. Door shut: the primary.
                let e = crate::pp::layer_engine(e, n_trunk, n_trunk)?;
                // Match model and external-draft paths: a rank artifact may be an `hf:` spec
                // too. This keeps the q38 DFlash2 default copy-paste runnable without an
                // untracked sidecar path; `resolve_arg` narrows the repo to its one d2t GGUF.
                let path = memra_gguf::hf::resolve_arg(&path)
                    .map_err(|err| format!("MEMRA_FRSPEC_TRIM={path:?}: {err}"))?;
                // Two artifact forms: the d2t GGUF container, or a plain `.txt` (one token id
                // per line, rank order — frspec-owngen writes both). The text form keeps the
                // fully-safetensors serving path free of GGUF entirely.
                let d2t: Vec<u32> = frspec_read_d2t(&path)?;
                frspec_src_sha16 = Some(sha256_file_hex(std::path::Path::new(&path), 8)?);
                // WHICH HEAD DO THE ROWS COME FROM? For a tied-head family (qwen35) the MTP
                // block reuses the trunk's `output.weight`, so gathering trunk rows is exact.
                // The step-3.7-flash family does NOT: each nextn block ships its OWN lm_head,
                // and this repo already paid for reading the trunk head there — acceptance
                // 0/248 across K=1..8 with self-consistency PASS (the receipt lives at
                // `draft_head_tensor`, hybrid.rs). So prefer the FIRST MTP block's own head
                // whenever the artifact carries one, and fall back to the trunk head only for
                // the tied families that genuinely share it.
                let own_head_name = frspec_trim_own_head_name(n_trunk);
                let own_head = src.find(&own_head_name);
                let from_own_head = own_head.is_some();
                let v = own_head
                    .or_else(|| src.find("output.weight"))
                    .or_else(|| src.find("token_embd.weight"))
                    .expect("model has no output.weight for FR-Spec trim");
                // BOOT ADMISSION on this arm too (revuto finding on the re-land of
                // lane/frspec-dflash2-20260902): the same env var must refuse a wrong-model
                // file by name with its sha16 whichever arm consumes it, never reach the
                // gather's assert (a process abort) or boot a shorter list.
                frspec_validate_ranks(
                    &d2t,
                    v.ne[1] as usize,
                    &format!(
                        "MEMRA_FRSPEC_TRIM={path} (sha16={}) on {}",
                        frspec_src_sha16.as_deref().unwrap_or("unknown"),
                        if from_own_head {
                            own_head_name.as_str()
                        } else {
                            "main output.weight"
                        }
                    ),
                )?;
                // FLOAT HEADS ARE REAL: step-3.7-flash keeps both `lm_head.weight` and every
                // `nextn.*.shared_head.output.weight` in BF16 [128896, 4096] even though its
                // experts are NVFP4, and `from_quant_bytes` PANICS on BF16 ("unsupported
                // dtype"). A row gather is dtype-agnostic — rows are independent and nothing is
                // requantized — so the only thing that changes is which GpuTensor the rows land
                // in. The draft head matmul already has a FloatBf16 arm.
                // MEMRA_FRSPEC_TRIM_NVFP4=1: quantize the trimmed rows to NVFP4 instead of
                // keeping them BF16. This is the repo's own draft-regime standard — tools/
                // make-trimmed-draft.sh builds "block Q4_K_M + head NVFP4" and records "NVFP4
                // head measured zero acceptance cost" — but that builder is a GGUF pipeline and
                // this family is safetensors, so the quantization happens HERE instead.
                // `f32_to_nvfp4` already emits the internal block layout the decode dp4a path
                // consumes (QK=64, 36 B/block, 4 UE4M3 sub-scales + 32 interleaved code bytes),
                // so no kernel changes. Macro scale is 1.0: unlike a modelopt tensor there is no
                // sibling weight_scale_2 — the per-16 sub-block scales are self-contained.
                // Worth it for RESIDENCY: a trimmed head goes 0.27 GB (BF16) -> 0.076 GB, and the
                // full 3-head chain 3.18 -> 0.89 GB, which is what OOMs at the natural 262144
                // context. Draft-head precision cannot change served output (verify arbitrates),
                // so acceptance is the only thing to measure.
                let (trimmed, nvfp4_sizes) = frspec_gather_trimmed_head(
                    e,
                    &v,
                    &d2t,
                    std::env::var("MEMRA_FRSPEC_TRIM_NVFP4").as_deref() == Ok("1"),
                    /*nvfp4 macro-scale*/
                    match src.find("output.scale") {
                        Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                        None => 1.0,
                    },
                )?;
                match nvfp4_sizes {
                    Some((nvfp4_bytes, gathered_bytes)) => eprintln!(
                        "[frspec-trim] self-trimmed head: {} rows of {} re-quantized BF16 -> NVFP4 \
                         ({} MiB, was {} MiB)",
                        d2t.len(),
                        if from_own_head {
                            own_head_name.as_str()
                        } else {
                            "main output.weight"
                        },
                        nvfp4_bytes >> 20,
                        gathered_bytes >> 20,
                    ),
                    None => eprintln!(
                        "[frspec-trim] self-trimmed head: {} rows of {} ({:?})",
                        d2t.len(),
                        if from_own_head {
                            own_head_name.as_str()
                        } else {
                            "main output.weight"
                        },
                        v.ggml_type
                    ),
                }
                head.shared_head_head = Some(trimmed);
                head.d2t = Some(d2t);
                // The ids index the TARGET vocabulary either way (both heads are vocab-wide),
                // so downstream remapping is unchanged by which matrix supplied the rows.
                head.d2t_from_target_head = !from_own_head;
                Some(head)
            }
            (_, m) => m,
        };
        // MEMRA_MTP_SKIP=1 stub draft head. With the embedded block skipped, `mtp` is None and
        // the trim arm above no-ops, which would SILENTLY strip the dspark/DFlash2 trimmed
        // draft head from a production shape that carries MEMRA_FRSPEC_TRIM (the silent-no-op
        // defect class). So under skip+trim, build the trimmed rows anyway and park them in
        // `dflash_trim`: everything the DFlash2 round consumes (head rows + d2t; verified
        // against both dflash.rs borrow sites 2026-08-30) and nothing more. `mtp` stays None,
        // so `mtp_spec_capable` and every MTP forward path stay off by construction. The d2t
        // was read and every refusal executed BEFORE the trunk loaded (see the block after
        // n_trunk); rows come from the trunk head by construction, and the own-head artifact
        // shape already refused there.
        let mut dflash_trim: Option<DflashTrimHead> = match mtp_skip_trim_d2t {
            Some((d2t, src_sha16)) => {
                let v = src
                    .find("output.weight")
                    .or_else(|| src.find("token_embd.weight"))
                    .ok_or("model has no output.weight for FR-Spec trim")?;
                frspec_validate_ranks(
                    &d2t,
                    v.ne[1] as usize,
                    &format!("MEMRA_MTP_SKIP=1 with MEMRA_FRSPEC_TRIM (sha16={src_sha16})"),
                )?;
                let (head, nvfp4_sizes) = frspec_gather_trimmed_head(
                    e,
                    &v,
                    &d2t,
                    std::env::var("MEMRA_FRSPEC_TRIM_NVFP4").as_deref() == Ok("1"),
                    match src.find("output.scale") {
                        Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                        None => 1.0,
                    },
                )?;
                eprintln!(
                    "[mtp-skip] FR-Spec stub draft head built: {} rows of main output.weight \
                     ({}); DFlash2 trim serves without the embedded MTP block",
                    d2t.len(),
                    match nvfp4_sizes {
                        Some((nvfp4_bytes, gathered_bytes)) => format!(
                            "re-quantized BF16 -> NVFP4, {} MiB, was {} MiB",
                            nvfp4_bytes >> 20,
                            gathered_bytes >> 20
                        ),
                        None => format!("{:?}", v.ggml_type),
                    },
                );
                Some(DflashTrimHead {
                    head,
                    d2t,
                    src_sha16,
                })
            }
            None => None,
        };
        // PER-HEAD TRIM (2026-08-27). This used to `mtp_extra.clear()`, which silently collapsed
        // a MEMRA_MTP_HEADS=3 chain to ONE trimmed head recursed at offsets it was never trained
        // for — measured as the K=3 deep-slot collapse (0.734/0.330/0.053 trimmed vs
        // 0.731/0.538/0.282 untrimmed; bf16-head and no-W8 single-variable arms reproduced the
        // trimmed slots bit-for-bit, so it was never a numeric-door effect — it is the banked
        // "single +1 head recursed" signature). The d2t ranking is a token-frequency list and is
        // HEAD-INDEPENDENT (every downstream remap may keep reading head 0's d2t); only the
        // gathered ROWS are per-head, because this family ships a different lm_head per nextn
        // block. So: same d2t for every head, each extra head's rows gathered from its OWN
        // block's head. A block without its own head tensor ends the chain there — rows from
        // another block's head are exactly the wrong-head bug this row's receipt documents
        // (acceptance 0/248 with self-consistency still PASSING), never a fallback.
        if let Some(d2t) = mtp.as_ref().and_then(|head| head.d2t.clone()) {
            let want_nvfp4_env = std::env::var("MEMRA_FRSPEC_TRIM_NVFP4").as_deref() == Ok("1");
            let mut kept = 0usize;
            // Extra chain heads are trailing MTP blocks too — last-stage placement, same
            // as the first head's trim above.
            let e = crate::pp::layer_engine(e, n_trunk, n_trunk)?;
            for (i, head) in mtp_extra.iter_mut().enumerate() {
                let name = frspec_trim_own_head_name(n_trunk + 1 + i);
                let Some(v) = src.find(&name) else { break };
                let out_f = v.ne[1] as usize;
                let row_bytes = v.bytes.len() / out_f;
                if d2t.iter().any(|&t| (t as usize) >= out_f) {
                    break;
                }
                let mut gathered = Vec::with_capacity(d2t.len() * row_bytes);
                for &t in &d2t {
                    let off = t as usize * row_bytes;
                    gathered.extend_from_slice(&v.bytes[off..off + row_bytes]);
                }
                let want_nvfp4 =
                    want_nvfp4_env && matches!(v.ggml_type, GgmlType::BF16) && v.ne[0] % 64 == 0;
                let trimmed = if want_nvfp4 {
                    let vals: Vec<f32> = gathered
                        .chunks_exact(2)
                        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                        .collect();
                    let blocks = memra_gguf::nvfp4_repack::f32_to_nvfp4(&vals);
                    GpuTensor::from_quant_bytes(
                        e,
                        &blocks,
                        GgmlType::NVFP4,
                        v.ne[0],
                        d2t.len() as u64,
                        1.0,
                    )?
                } else {
                    match v.ggml_type {
                        GgmlType::BF16 => GpuTensor::FloatBf16 {
                            data: e.htod_bytes(&gathered)?,
                            ne: vec![v.ne[0], d2t.len() as u64],
                        },
                        GgmlType::F32 => GpuTensor::Float {
                            data: e.htod(
                                &gathered
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                                    .collect::<Vec<f32>>(),
                            )?,
                            ne: vec![v.ne[0], d2t.len() as u64],
                        },
                        _ => GpuTensor::from_quant_bytes(
                            e,
                            &gathered,
                            v.ggml_type,
                            v.ne[0],
                            d2t.len() as u64,
                            1.0,
                        )?,
                    }
                };
                head.shared_head_head = Some(trimmed);
                head.d2t = Some(d2t.clone());
                head.d2t_from_target_head = false;
                kept += 1;
            }
            let dropped = mtp_extra.len() - kept;
            mtp_extra.truncate(kept);
            eprintln!(
                "[frspec-trim] per-head trim: {kept} extra chain head(s) gathered from their own \
                 blocks{}",
                if dropped > 0 {
                    format!(" ({dropped} dropped: no own-head tensor)")
                } else {
                    String::new()
                }
            );
        }
        if !mtp_extra.is_empty() {
            if plan.draft_source != memra_gguf::model_plan::DraftSourcePlan::Embedded
                || plan.mtp_blocks.len() != 1 + mtp_extra.len()
                || plan
                    .mtp_blocks
                    .iter()
                    .any(|block| !matches!(block.layer.mlp, MlpPlan::Dense(_)))
                || mtp
                    .iter()
                    .chain(mtp_extra.iter())
                    .any(|head| !matches!(head.ffn, Ffn::Dense { .. }))
            {
                return Err(
                    "multi-head MTP requires embedded dense canonical blocks and matching loaded heads"
                        .into(),
                );
            }
            eprintln!(
                "[mtp-draft] embedded chain: heads={} blocks={}..={} scratch=per-head",
                1 + mtp_extra.len(),
                n_trunk,
                n_trunk + mtp_extra.len()
            );
        }

        // glm5 DFlash2 ALTERNATE DRAFT SOURCE (lane/glm5-dflash-draft-src, 2026-08-30;
        // owner holds written approval from the DFlash2 owners, 2026-08-30, for use beyond
        // probe/eval): MEMRA_GLM5_DFLASH=<dir-or-hf-spec> loads the pinned block-diffusion
        // drafter on the HEAD engine (where the trunk lm head it projects through lives —
        // the MTP-head placement law). The native MTP head is NOT needed and NOT loaded for
        // this source (the q38 pattern: layers.45 is a full MoE trunk layer of VRAM).
        // A set flag that cannot load is a LOUD boot failure, never a silent plain fallback.
        //
        // THE LOAD CONTRACT IS THE GENERAL SEAM (lane/glm5-extract2):
        // `dflash::load_drafter` holds every drafter<->target validation (DFlash2 family,
        // hidden == n_embd, taps inside the trunk, mask token inside the vocab) plus the
        // sha256 identity pin. All four are properties of the PAIR, not of glm5 — the next
        // spec family passes its own flag name and its own (n_trunk, n_embd, n_vocab). What
        // stays here is glm5's own: the family flag name and the `is_glm5_next()` route.
        // Error bytes unchanged (the general fn prefixes `{flag}={dir}`).
        let glm5_dflash = match std::env::var("MEMRA_GLM5_DFLASH") {
            Ok(spec) if !spec.is_empty() && cfg.arch.is_glm5_next() => {
                let dpath = memra_gguf::hf::resolve_arg(&spec)
                    .map_err(|err| format!("MEMRA_GLM5_DFLASH={spec:?}: {err}"))?;
                let de = crate::pp::layer_engine(e, n_trunk, n_trunk)?;
                Some(crate::dflash::load_drafter(
                    de,
                    std::path::Path::new(&dpath),
                    "MEMRA_GLM5_DFLASH",
                    n_trunk,
                    cfg.n_embd as usize,
                    output.out_features(),
                )?)
            }
            _ => None,
        };

        // glm5 DFlash2 DRAFT-HEAD RANK TRIM (lane/frspec-dflash2-20260902, owner order: "if the
        // masked path isn't wired, wire it"). The MtpHead self-trim above only lands when the
        // NextN block is loaded; the serving DFlash2 route boots WITHOUT it (the q38 VRAM
        // pattern), so a set MEMRA_FRSPEC_TRIM used to be a SILENT NO-OP there: the boot
        // receipt said `draft head FULL target vocab` and the drafter projected every round
        // through the full 154,880-row head. SAME CONTRACT, no new flag: the ranks file named
        // by MEMRA_FRSPEC_TRIM is gathered ONCE here into an `[n_ranks x d]` slab of the
        // trunk head's own rows (`frspec_gather_trimmed_head`, the one gather program every
        // trim arm shares; for glm5_next the trunk head is the draft head BY CONTRACT, the
        // NextN block ships no private lm_head) and parked in `dflash_trim`, which the
        // DFlash2 round consumes exactly as it consumes the MEMRA_MTP_SKIP stub: draft
        // logits over the slab, candidate ids remapped through d2t BEFORE the selector walk,
        // verify full-vocab and untouched. NUMERIC CLASS: the target's output distribution
        // and the greedy tape are unchanged by construction (a draft source can only move
        // acceptance, never output, module doc of glm_spec.rs); the slab's rows are byte-
        // identical to the head rows they were gathered from (`frspec_gather_rows`).
        // ADMISSION (owner order): the ranks file is parsed STRICTLY (.txt: every non-blank
        // line an integer, no duplicates) and validated against the head's row count; a
        // wrong-model file REFUSES the boot by name with its sha16, never loads silently.
        // Skipped when the MtpHead self-trim already carries target-head rows (the
        // MEMRA_GLM5_MTP=1 + trim shape: the round prefers that struct, no second slab) or
        // the MEMRA_MTP_SKIP stub already built one. Non-glm5 co-loaded models never reach
        // this arm (the drafter flag is glm5-scoped), so a global env cannot kill them.
        // DOOR, DEFAULT OFF (unset). MEASURED on the 2x B200 pair 2026-09-03 (glm53-flash-nvfp4
        // + DFlash2 b33c0347, sxc32768 ranks, vendor sampling, K=3 and K=5): neutral short
        // (within 2%), a consistent -10 to -22% decode loss at 42k in every pair; the door
        // stays the instrument for a workload-keyed mint, never a candidate default
        // (FLAGS.md row, darklanes research/glm5-b200-20260902/floor/raw/finalclean/).
        if cfg.arch.is_glm5_next()
            && glm5_dflash.is_some()
            && dflash_trim.is_none()
            && !crate::model::full_prec_enabled()
            && !mtp
                .as_ref()
                .is_some_and(|m| m.d2t_from_target_head && m.d2t.is_some())
            && let Ok(spec) = std::env::var("MEMRA_FRSPEC_TRIM")
            && !spec.is_empty()
        {
            let what = "MEMRA_FRSPEC_TRIM on the glm5 DFlash2 draft head";
            let path = memra_gguf::hf::resolve_arg(&spec)
                .map_err(|err| format!("{what}: {spec:?}: {err}"))?;
            let sha16 = sha256_file_hex(std::path::Path::new(&path), 8)?;
            let d2t: Vec<u32> = if path.ends_with(".txt") {
                let text = std::fs::read_to_string(&path)
                    .map_err(|err| format!("{what}: {path}: {err}"))?;
                frspec_parse_ranks_txt_strict(&text, &format!("{what} ({path}, sha16={sha16})"))?
            } else {
                frspec_read_d2t(&path)?
            };
            let n_vocab = output.out_features();
            frspec_validate_ranks(&d2t, n_vocab, &format!("{what} ({path}, sha16={sha16})"))?;
            let v = src
                .find("output.weight")
                .or_else(|| src.find("token_embd.weight"))
                .ok_or_else(|| {
                    format!("{what}: model has no output.weight (or tied token_embd.weight)")
                })?;
            if v.ne[1] as usize != n_vocab {
                return Err(format!(
                    "{what}: source head rows {} != loaded head rows {n_vocab}",
                    v.ne[1]
                )
                .into());
            }
            // The slab lives where the drafter and the trunk lm head live: the head engine.
            let de = crate::pp::layer_engine(e, n_trunk, n_trunk)?;
            let (head, nvfp4_sizes) = frspec_gather_trimmed_head(
                de,
                &v,
                &d2t,
                std::env::var("MEMRA_FRSPEC_TRIM_NVFP4").as_deref() == Ok("1"),
                match src.find("output.scale") {
                    Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                    None => 1.0,
                },
            )?;
            eprintln!(
                "[frspec-trim] glm5 DFlash2 draft-head slab: {} rows of {} gathered from main \
                 output.weight ({}) src={sha16} ({path})",
                d2t.len(),
                n_vocab,
                match nvfp4_sizes {
                    Some((nvfp4_bytes, gathered_bytes)) => format!(
                        "re-quantized BF16 -> NVFP4, {} MiB, was {} MiB",
                        nvfp4_bytes >> 20,
                        gathered_bytes >> 20
                    ),
                    None => format!(
                        "{:?}, {} MiB",
                        v.ggml_type,
                        (d2t.len() * (v.bytes.len() / n_vocab)) >> 20
                    ),
                },
            );
            frspec_src_sha16 = Some(sha16.clone());
            dflash_trim = Some(DflashTrimHead {
                head,
                d2t,
                src_sha16: sha16,
            });
        }

        // GLM5-SPEC BOOT RECEIPT (lane/glm5-spec-routing, 2026-08-30): the deploy gate greps
        // the server log for these lines (never-serve-greedy law: spec engagement must be
        // provable from the log, a 200 proves nothing). With MEMRA_GLM5_SPEC unset/0 the boot
        // log carries NO `[glm5-spec]` line at all — the receipt gate's red arm.
        // DRAFT-SOURCE SELECTION (lane/glm5-dflash-draft-src): a loaded DFlash2 drafter IS
        // the draft source (MEMRA_GLM5_DFLASH set = the operator asked for it by name);
        // the selection line is the receipt the source matrix gate asserts on.
        if cfg.arch.is_glm5_next() && crate::glm_spec::glm5_spec_on() {
            match (glm5_dflash.as_ref(), mtp.as_ref()) {
                (Some(dr), head) => {
                    // RANK-TRIMMED = the DFlash2 round WILL draft over a trimmed slab: the
                    // MtpHead self-trim (target-head rows) or the `dflash_trim` slab, in the
                    // round's own preference order (`glm5_dflash_trim`). `n_ranks` + the
                    // ranks file's sha16 make a wrong-model artifact nameable from the log.
                    let trim_note = match (
                        head.filter(|h| h.d2t_from_target_head)
                            .and_then(|h| h.d2t.as_ref())
                            .filter(|m| !m.is_empty()),
                        dflash_trim.as_ref(),
                    ) {
                        (Some(map), _) => format!(
                            "draft head RANK-TRIMMED n_ranks={} src={}",
                            map.len(),
                            frspec_src_sha16.as_deref().unwrap_or("unknown")
                        ),
                        (None, Some(slab)) => format!(
                            "draft head RANK-TRIMMED n_ranks={} src={}",
                            slab.d2t.len(),
                            slab.src_sha16
                        ),
                        (None, None) => "draft head FULL target vocab".to_string(),
                    };
                    eprintln!(
                        "[glm5-spec] serve route ARMED: draft source = dflash2 @ {}; {trim_note}; \
                         native MTP head {}",
                        dr.sha8,
                        if head.is_some() {
                            "ALSO loaded (idle for drafting — dflash2 wins by selection)"
                        } else {
                            "NOT loaded (the q38 pattern: a full MoE trunk layer of VRAM saved)"
                        }
                    );
                }
                (None, Some(head)) => {
                    match head.d2t.as_ref() {
                        Some(map) => eprintln!(
                            "[glm5-spec] serve route ARMED: MTP head loaded; draft head TRIMMED \
                             to {} rows (FR-Spec d2t engaged)",
                            map.len()
                        ),
                        None => eprintln!(
                            "[glm5-spec] serve route ARMED: MTP head loaded; draft head FULL \
                             target vocab (no FR-Spec trim)"
                        ),
                    }
                    eprintln!("[glm5-spec] draft source = native-mtp");
                }
                (None, None) => eprintln!(
                    "[glm5-spec] MEMRA_GLM5_SPEC=1 but no MTP head loaded \
                     (set MEMRA_GLM5_MTP=1 or MEMRA_GLM5_DFLASH=<drafter>) — route stays \
                     fail-closed, plain serving"
                ),
            }
        }

        if let Some(ctx) = spill.as_ref() {
            eprintln!(
                "[spill] experts placed: {} pinned (Tier 1), {} mmap'd from disk (Tier 2, {} MiB)",
                ctx.n_pinned,
                ctx.n_mmap,
                ctx.mmap_bytes >> 20
            );
        }

        // FA v4 GQA CAPACITY GUARD (2026-08-06, lane/122b-bringup): fa_v4_smem sizes its
        // per-warp Q arrays q_ints[8][64]/q_d[8][8] for gqa<=8 — every model before the
        // 122B-A10B (32 Q heads / 2 KV heads = gqa 16) fit. At gqa>8 the (32,gqa,1) block's
        // warps 8..15 write q_ints[wy] PAST the array into the k_ints/k_d K tile, corrupting
        // scores -> all-NaN decode logits (receipts: research/122b-bringup-20260806/, arm
        // battery: v4/deep MISMATCH+NaN, v3/v2/smem/reg/scalar all MATCH). The hd512 lane
        // already carries its own capacity guard at dispatch ("gqa <= 16 = fa_v4_smem_512's
        // q-array capacity"); hd256 v4 never got one. Key FA_V4_MAX_DEFAULT=0 at load so
        // EVERY v4 dispatch site (eager, rows-verify, dc, rows_dc, windowed, seqs) flips to
        // the v3 lane together — decode/verify stay kernel-family-identical (the parity law).
        // Explicit MEMRA_FA_V4_MAX env still wins (diagnostic seam). The real v4 gqa16
        // extension is a kernel change gated on its own battery + perf receipts (fix brief
        // in research/122b-bringup-20260806/VERDICT.md).
        if cfg.n_head_kv > 0 && cfg.n_head / cfg.n_head_kv > 8 {
            crate::FA_V4_MAX_DEFAULT.store(0, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[fa] v4 decode family disabled: gqa {} > fa_v4_smem capacity 8 (v3 lane serves)",
                cfg.n_head / cfg.n_head_kv
            );
        }

        if gemma_program {
            // gemma4 fa-vec crossover default (measured sweep 2026-07-10; env overrides).
            crate::FA_VEC_MIN_DEFAULT.store(1, std::sync::atomic::Ordering::Relaxed);
            // windowed split per gemma variant (2026-07-12 sweeps): MoE 26B = 32 (grid-limited
            // t=1 under the raw-e4m3 sV ceiling), dense 31B = 64 (37.13 vs 36.87 at 1.7k, N=2).
            let real_moe = plan
                .trunk_operations()
                .contains(&memra_gguf::model_plan::OperationKind::MoeMlp);
            crate::FA_SPW_DEFAULT.store(
                if real_moe { 32 } else { 64 },
                std::sync::atomic::Ordering::Relaxed,
            );
            // hd512 global split per variant (26B=16 landed 2026-07-11; 31B=32 swept 2026-07-12).
            crate::FA_SP512_DEFAULT.store(
                if real_moe { 16 } else { 32 },
                std::sync::atomic::Ordering::Relaxed,
            );
            // gemma4 router w8 RE-ARBITRATED 2026-08-01 (g26 decode dig): the 2026-07-31
            // knife-edge that stored false here was single-synthetic-prompt roulette — on 6
            // real prompts the w8 twin's gate outcome is IDENTICAL to the lone-warp form
            // (5 MATCH/5 MATCH; the one MISMATCH prompt fails both arms with the same
            // argmax pair, router-independent). w8 = +13% g26 decode (182->206 tok/s x3
            // interleaved, H100). Receipts: research/g26-decode-20260801/. gemma4 now rides
            // the global default (true); MEMRA_ROUTER_V2=0 is the rollback seam.
            // fused t=1 pair/triple mr1 per variant (2026-07-14 DRAM-duty arc: dense +1.1%
            // short / +0.6% depth on 31B; MoE 26B −1.2% — stays mr2).
            crate::FUSED_MR1_DEFAULT.store(!real_moe, std::sync::atomic::Ordering::Relaxed);
            // gemma4 rms_norm block 1024 (single-row 2816-col norms; battery-arbitrated per model).
            crate::RMS_BLOCK_DEFAULT.store(1024, std::sync::atomic::Ordering::Relaxed);
            // gemma4 fa split ladder (d1736 sweep; see fa_split_keys).
            crate::FA_SP_GEMMA.store(true, std::sync::atomic::Ordering::Relaxed);
            // depth fa: PARITY LAW (2026-07-10) — decode and verify share the rows_w/rows_dpl16
            // kernel symbols (decode t=1), so lane choice is freely tunable; v4 measured the
            // depth winner. Seams: MEMRA_FA_V4_MAX / MEMRA_FA_SMEM_TKV / MEMRA_GEMMA_ROWS_W.
        }
        // GLM-5.3-Flash on sm_100a builds: rms_norm block 1024 (B200 posture flip 2026-09-06;
        // +2.59% alone and inside every all4 receipt, darklanes
        // research/glm5-b200-mint-20260904/LANE.md). `MEMRA_RMS_BLOCK=<n>` still overrides;
        // other archs keep the per-model default above.
        if cfg.arch.is_glm5_next() && env!("MEMRA_BUILT_CUDA_ARCH") == "100a" {
            crate::RMS_BLOCK_DEFAULT.store(1024, std::sync::atomic::Ordering::Relaxed);
        }
        // gemma4: the dc serving loop + spec draft gather read the device embed table every
        // step — upload it AT LOAD (OnceLock init) so first-use cost never lands in a timed span.
        let force_embd_gpu = gemma_program;
        let gemma4_aux = if gemma_program {
            let rope_freqs = match src.find("rope_freqs.weight") {
                Some(t) => {
                    let host = memra_gguf::dequant::dequantize(
                        t.ggml_type,
                        &t.bytes,
                        t.ne.iter().product::<u64>() as usize,
                    );
                    let mut copies = Vec::new();
                    if let Some(fence) = crate::pp::pp_cuts(n_trunk) {
                        #[allow(clippy::needless_range_loop)]
                        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                        for s in 0..fence.len() - 1 {
                            let owner = crate::pp::layer_engine(e, n_trunk, fence[s])?;
                            let dev = owner.ctx().ordinal();
                            if copies.iter().all(|(d, _)| *d != dev) {
                                copies.push((dev, owner.htod(&host)?));
                            }
                        }
                    } else {
                        copies.push((e.ctx().ordinal(), e.htod(&host)?));
                    }
                    Some(copies)
                }
                // NATIVE SAFETENSORS (lane/gemma-vision): rope_freqs.weight is a GGUF-only
                // synthesized tensor — the official checkpoint ships none. Law verified
                // against the shipped GGUF bytes (research/gemma-vision-20260816): factors
                // are 1.0 for the first partial_rotary_factor fraction of the head_dim/2
                // pairs and ~1e30 beyond (frequency ÷ ~inf = unrotated tail = proportional
                // p-RoPE). Synthesize the same law from the HF partial factor (0.25 on the
                // 31B) so the global-layer forward reads identical freq-factors either way.
                None => {
                    let g4 = cfg.gemma4.as_ref().unwrap();
                    let n = (g4.rope_dims_global / 2) as usize;
                    let keep =
                        ((n as f32) * g4.partial_rotary_global.clamp(0.0, 1.0)).round() as usize;
                    let host: Vec<f32> = (0..n)
                        .map(|i| if i < keep { 1.0 } else { 1.0e30 })
                        .collect();
                    eprintln!(
                        "[gemma4] rope_freqs.weight synthesized ({n} factors, first {keep} \
                         rotate; source ships none — native checkpoint)"
                    );
                    let mut copies = Vec::new();
                    if let Some(fence) = crate::pp::pp_cuts(n_trunk) {
                        #[allow(clippy::needless_range_loop)]
                        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                        for s in 0..fence.len() - 1 {
                            let owner = crate::pp::layer_engine(e, n_trunk, fence[s])?;
                            let dev = owner.ctx().ordinal();
                            if copies.iter().all(|(d, _)| *d != dev) {
                                copies.push((dev, owner.htod(&host)?));
                            }
                        }
                    } else {
                        copies.push((e.ctx().ordinal(), e.htod(&host)?));
                    }
                    Some(copies)
                }
            };
            // E4B per-layer-embedding model tensors (tensor-presence gated).
            let e4b = match src.find("per_layer_token_embd.weight") {
                Some(t) => {
                    let n_epl = cfg
                        .gemma4
                        .as_ref()
                        .map(|g| g.n_embd_per_layer as usize)
                        .unwrap_or(0);
                    let row = t.ne[0] as usize; // n_epl * n_layer
                    let row_bytes = t.bytes.len() / (t.ne[1] as usize);
                    eprintln!(
                        "[gemma4-e4b] per-layer-embed model detected (n_epl={n_epl}, row {row}) — \
                               first-light forward (eager decode + prime); dc/graph/spec unwired \
                               (HANDOVER-E4B.md)"
                    );
                    Some(crate::hybrid::Gemma4E4bModel {
                        tok_tbl_gpu: std::sync::OnceLock::new(),
                        tok_embd_bytes: t.bytes.to_vec(),
                        tok_embd_qt: match t.ggml_type {
                            memra_gguf::GgmlType::Q6_K => crate::QT_Q6_K,
                            memra_gguf::GgmlType::Q8_0 => crate::QT_Q8_0,
                            other => panic!("e4b per-layer tok embd: unhandled dtype {other:?}"),
                        },
                        tok_embd_row_bytes: row_bytes,
                        model_proj: load_t(e, src, "per_layer_model_proj.weight")?,
                        proj_norm: load_t(e, src, "per_layer_proj_norm.weight")?,
                        n_epl,
                    })
                }
                None => None,
            };
            let suppress_d = {
                let sup = &cfg.gemma4.as_ref().unwrap().suppress_tokens;
                if sup.is_empty() {
                    None
                } else {
                    let ids: Vec<i32> = sup.iter().map(|&x| x as i32).collect();
                    eprintln!(
                        "[gemma4] suppress_tokens: {} ids masked at sampling",
                        ids.len()
                    );
                    Some((e.htod_i32(&ids)?, ids.len()))
                }
            };
            let ones_host = [1.0f32; 512];
            let mut ones = Vec::new();
            if let Some(fence) = crate::pp::pp_cuts(n_trunk) {
                #[allow(clippy::needless_range_loop)]
                // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                for s in 0..fence.len() - 1 {
                    let owner = crate::pp::layer_engine(e, n_trunk, fence[s])?;
                    let dev = owner.ctx().ordinal();
                    if ones.iter().all(|(d, _)| *d != dev) {
                        ones.push((dev, owner.htod(&ones_host)?));
                    }
                }
            } else {
                ones.push((e.ctx().ordinal(), e.htod(&ones_host)?));
            }
            Some(GemmaAux {
                rope_freqs,
                ones,
                suppress_d,
                e4b,
            })
        } else {
            None
        };
        // step35: rope_freqs.weight [n_rot_full/2] — FULL-attn layers only (SWA passes null).
        // Loaded by tensor presence, not required: the key is absent on a sibling without
        // llama3-style scaling, and `None` is the correct "no factors" signal for rope_neox2.
        let step35_aux = if sliding_gated_moe_program {
            let rope_freqs = match src.find("rope_freqs.weight") {
                Some(t) => {
                    let host = memra_gguf::dequant::dequantize(
                        t.ggml_type,
                        &t.bytes,
                        t.ne.iter().product::<u64>() as usize,
                    );
                    let mut copies = Vec::new();
                    if let Some(fence) = crate::pp::pp_cuts(n_trunk) {
                        #[allow(clippy::needless_range_loop)]
                        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                        for s in 0..fence.len() - 1 {
                            let owner = crate::pp::layer_engine(e, n_trunk, fence[s])?;
                            let dev = owner.ctx().ordinal();
                            if copies.iter().all(|(d, _)| *d != dev) {
                                copies.push((dev, owner.htod(&host)?));
                            }
                        }
                    } else {
                        copies.push((e.ctx().ordinal(), e.htod(&host)?));
                    }
                    Some(copies)
                }
                None => None,
            };
            Some(Step35Aux { rope_freqs })
        } else {
            None
        };
        let mut layers = layers;
        // Q8_0 SPLIT-PLANE DECODE MIRRORS (2026-07-26, the H100 lane): Q8_0-trunk models
        // (Qwen3.5-9B class) stream their whole weight mass through the 34B-stride GGUF
        // layout — ncu on H100 held Max Bandwidth at 41-46% (Mem Busy 66-76%) from sector
        // overfetch. Mirrors route the m<=16 mmvq/batched decode family to the aligned-16B
        // `_rp` twins (bit-identical). VRAM cost == the mirrored trunk (~model size), so
        // DEFAULT ON only on the Hopper lane (80GB); MEMRA_Q8RP=1/0 overrides either way.
        {
            let q8rp_on = match std::env::var("MEMRA_Q8RP").as_deref() {
                Ok("0") => false,
                Ok(_) => true,
                // Owner ruling (2026-08-16, gap-diagnosis arc): bit-identical + faster ships
                // default-ON wherever it costs nothing. The mirror is pure VRAM, so the unset
                // default is CAPACITY-KEYED: ON when free VRAM covers the mirror mass plus
                // serving headroom (the 96GB serving boxes; gemma4-31B NVFP4mix measured
                // 58.3->58.8 tok/s c1), OFF where it cannot (24GB rigs keep today's OFF).
                // Sharded trunks: `free` is engine-0's — the sharded rigs are the big-VRAM
                // class, so the conservative single-device read is acceptable.
                Err(_) => {
                    cfg!(memra_hopper_mma) || {
                        let q8b = |w: &crate::model::GpuTensor| -> usize {
                            match w {
                                crate::model::GpuTensor::Quant {
                                    bytes,
                                    qtype,
                                    row_bytes,
                                    ne,
                                    rp4: None,
                                    ..
                                } if *qtype == crate::QT_Q8_0
                                    && ne.len() == 2
                                    && (ne[0] as usize).is_multiple_of(32)
                                    && *row_bytes == (ne[0] as usize / 32) * 34 =>
                                {
                                    bytes.len()
                                }
                                _ => 0,
                            }
                        };
                        let mut need = q8b(&output);
                        for layer in layers.iter() {
                            match &layer.mixer {
                                Mixer::Full(fa) => {
                                    for w in [&fa.wq, &fa.wk, &fa.wv, &fa.wo] {
                                        need += q8b(w);
                                    }
                                }
                                Mixer::Linear(la) => {
                                    for w in [
                                        &la.wqkv,
                                        &la.wqkv_gate,
                                        &la.ssm_beta,
                                        &la.ssm_alpha,
                                        &la.ssm_out,
                                    ] {
                                        need += q8b(w);
                                    }
                                }
                                Mixer::Mla(_) => {}
                                Mixer::Kda(_) => {} // no q8 mirrors (same as MLA above)
                            }
                            if let Ffn::Dense {
                                ffn_gate,
                                ffn_up,
                                ffn_down,
                                // memra#253: this site inspects or moves weights and runs no GEMM on an
                                // activation, so the AWQ activation-side scale plays no part in it.
                                ffn_down_pqs: _,
                            } = &layer.ffn
                            {
                                for w in [ffn_gate, ffn_up, ffn_down] {
                                    need += q8b(w);
                                }
                            }
                        }
                        need > 0
                            && e.ctx()
                                .mem_get_info()
                                .map(|(free, _)| free >= need + (8usize << 30))
                                .unwrap_or(false)
                    }
                }
            };
            // K-quant split-plane mirrors (q4_K/q6_K, 2026-08-01 H100 coalescing fix) ride
            // the same trunk walk under their own seam (MEMRA_KQRP, default = hopper lane).
            // K-quant mirror capacity default (lane/gemma-q6kb, 2026-08-17): the H100
            // coalescing fix was Hopper-only by default, leaving the 96GB Blackwell
            // serving boxes on the misaligned-210B GGUF walk — the shipping trunk's
            // Q6_K ffn_down measured 862 GB/s base vs 1.15 TB/s through the mirror
            // (_b8_rp med 88->66us; c8 agg +4.6%). Same capacity pattern as Q8RP:
            // env keeps priority, unset admits iff free VRAM covers the admissible
            // q4_K/q6_K mirror mass + 8 GiB headroom; 24GB rigs refuse by construction.
            let kqrp_on = crate::Engine::kqrp_enabled() || {
                std::env::var("MEMRA_KQRP").is_err() && {
                    let kqb = |w: &crate::model::GpuTensor| -> usize {
                        match w {
                            crate::model::GpuTensor::Quant {
                                bytes,
                                qtype,
                                row_bytes,
                                ne,
                                rp4: None,
                                ..
                            } if ne.len() == 2 && (ne[0] as usize).is_multiple_of(256) => {
                                let sb = if *qtype == crate::QT_Q4_K {
                                    144
                                } else if *qtype == crate::QT_Q6_K {
                                    210
                                } else {
                                    return 0;
                                };
                                if *row_bytes == (ne[0] as usize / 256) * sb {
                                    bytes.len()
                                } else {
                                    0
                                }
                            }
                            _ => 0,
                        }
                    };
                    let mut need = kqb(&output);
                    for layer in layers.iter() {
                        if let Mixer::Full(fa) = &layer.mixer {
                            for w in [&fa.wq, &fa.wk, &fa.wv, &fa.wo] {
                                need += kqb(w);
                            }
                        }
                        if let Ffn::Dense {
                            ffn_gate,
                            ffn_up,
                            ffn_down,
                            // memra#253: this site inspects or moves weights and runs no GEMM on an
                            // activation, so the AWQ activation-side scale plays no part in it.
                            ffn_down_pqs: _,
                        } = &layer.ffn
                        {
                            for w in [ffn_gate, ffn_up, ffn_down] {
                                need += kqb(w);
                            }
                        }
                    }
                    need > 0
                        && e.ctx()
                            .mem_get_info()
                            .map(|(free, _)| free >= need + (8usize << 30))
                            .unwrap_or(false)
                }
            };
            if q8rp_on || kqrp_on {
                // f16 prefill mirrors, PER-MODEL argmax-gate arbitration (round 45): on the
                // qwen Q8_0 dense class the f16-prefill-vs-int8-decode gap (maxdiff ~0.67)
                // flips the run-gen argmax gate on real prompts (board-2048: 485 vs 332,
                // deterministic x5) — gate-violating defaults don't ship. gemma (Q4_0) and
                // the MoE hybrids hold MATCH on the same prompt and keep their mirrors.
                // MEMRA_PP_F16=1 forces (diagnostic seam); =0 still kills everywhere.
                let f16_model_ok = gemma_program
                    || plan
                        .trunk_operations()
                        .contains(&memra_gguf::model_plan::OperationKind::MoeMlp)
                    || std::env::var("MEMRA_PP_F16").as_deref() == Ok("1");
                let mut nmir = 0usize;
                // M2 weight sharding: mirrors are the DECODE weights on these paths — each
                // builds through its layer's OWNING stage engine (`e_ref` param), so the
                // mirror lands on the device that dereferences it.
                let mut mir = |e_ref: &crate::Engine,
                               w: &mut crate::model::GpuTensor|
                 -> Result<(), Box<dyn std::error::Error>> {
                    let before = matches!(w, crate::model::GpuTensor::Quant { rp4: Some(_), .. });
                    if q8rp_on {
                        e_ref.build_q8_rp4(w)?;
                    }
                    if kqrp_on {
                        e_ref.build_q4k_rp4(w)?;
                        e_ref.build_q6k_rp4(w)?;
                    }
                    // Q6_K mirrors are model-CLASS-agnostic (round 47): no MMQ arm exists for
                    // Q6_K — the fallback dequant-GEMM is ~10x the f16 lane (q27's prefill
                    // wall). The qwen-dense argmax-flip evidence (round 45) was the Q8_0
                    // mirror specifically; Q6_K admission is arbitrated by its own gate runs.
                    let q6k = matches!(w, crate::model::GpuTensor::Quant { qtype, .. }
                                       if *qtype == crate::QT_Q6_K);
                    if q8rp_on && crate::f16_ffi::pp_f16_enabled() && (f16_model_ok || q6k) {
                        e_ref.build_q8_f16(w)?;
                    }
                    if !before && matches!(w, crate::model::GpuTensor::Quant { rp4: Some(_), .. }) {
                        nmir += 1;
                    }
                    Ok(())
                };
                for (il, layer) in layers.iter_mut().enumerate() {
                    let el = crate::pp::layer_engine(e, n_trunk, il)?;
                    match &mut layer.mixer {
                        Mixer::Full(fa) => {
                            for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] {
                                mir(el, w)?;
                            }
                        }
                        Mixer::Linear(la) => {
                            for w in [
                                &mut la.wqkv,
                                &mut la.wqkv_gate,
                                &mut la.ssm_beta,
                                &mut la.ssm_alpha,
                                &mut la.ssm_out,
                            ] {
                                mir(el, w)?;
                            }
                        }
                        // MLA: no decode mirrors in increment 2 (its kernels arrive in inc 4;
                        // mirror admission is arbitrated there with measurements).
                        Mixer::Mla(_) => {}
                        Mixer::Kda(_) => {} // no q8 mirrors (same as MLA above)
                    }
                    if let Ffn::Dense {
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        // memra#253: this site inspects or moves weights and runs no GEMM on an
                        // activation, so the AWQ activation-side scale plays no part in it.
                        ffn_down_pqs: _,
                    } = &mut layer.ffn
                    {
                        for w in [ffn_gate, ffn_up, ffn_down] {
                            mir(el, w)?;
                        }
                    }
                }
                mir(e_head, &mut output)?;
                if nmir > 0 {
                    eprintln!("[q8rp] split-plane decode mirrors built: {nmir} tensors");
                }
                // Q4_K f16 prefill mirrors (round 49): Q4_K joins the q6k carve-out —
                // model-class-agnostic admission, arbitrated by per-model argmax gates
                // (the round-45 flip evidence was the Q8_0 mirror on qwen-dense; the q27
                // Q4_K bulk rides mul_mat_q_q45k int8-MMA, which the Lt f16 lane beats at
                // large m — campaign-A precedent). SECOND pass over the trunk so the shared
                // MEMRA_PP_F16_BUDGET_MB keeps FULL Q6_K coverage as its floor: Q6_K mirrors
                // replace a ~10x dequant-GEMM (no MMQ arm exists), Q4_K mirrors upgrade a
                // working int8-MMA arm — a joint walk would evict late-layer Q6_K mirrors
                // for the weaker lever. Layer-order prefix within the Q4_K class.
                // Round 49b: Q5_K (q27's 48 ssm_out — the last mul_mat_q_q45k class) rides
                // a THIRD pass strictly after all Q4_K, so the default-budget composition
                // (and its banked gates) stays byte-identical: the 32GB default is exhausted
                // by the Q4_K pass; Q5_K mirrors only light up under a raised
                // MEMRA_PP_F16_BUDGET_MB (machine-specific config).
                if q8rp_on && crate::f16_ffi::pp_f16_enabled() {
                    for (want, tag) in [(crate::QT_Q4_K, "q4kf16"), (crate::QT_Q5_K, "q5kf16")] {
                        let (mut n4, mut b4) = (0usize, 0usize);
                        let mut mirk =
                            |e_ref: &crate::Engine,
                             w: &mut crate::model::GpuTensor|
                             -> Result<(), Box<dyn std::error::Error>> {
                                if matches!(w, crate::model::GpuTensor::Quant { qtype, f16: None, .. }
                                        if *qtype == want)
                                {
                                    e_ref.build_q8_f16(w)?;
                                    if let crate::model::GpuTensor::Quant { f16: Some(m), .. } = w {
                                        n4 += 1;
                                        b4 += m.len();
                                    }
                                }
                                Ok(())
                            };
                        for (il, layer) in layers.iter_mut().enumerate() {
                            let el = crate::pp::layer_engine(e, n_trunk, il)?;
                            match &mut layer.mixer {
                                Mixer::Full(fa) => {
                                    for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] {
                                        mirk(el, w)?;
                                    }
                                }
                                Mixer::Linear(la) => {
                                    for w in [
                                        &mut la.wqkv,
                                        &mut la.wqkv_gate,
                                        &mut la.ssm_beta,
                                        &mut la.ssm_alpha,
                                        &mut la.ssm_out,
                                    ] {
                                        mirk(el, w)?;
                                    }
                                }
                                Mixer::Mla(_) => {} // no mirrors in increment 2 (see above)
                                Mixer::Kda(_) => {} // no q8 mirrors (same as MLA above)
                            }
                            if let Ffn::Dense {
                                ffn_gate,
                                ffn_up,
                                ffn_down,
                                // memra#253: this site inspects or moves weights and runs no GEMM on an
                                // activation, so the AWQ activation-side scale plays no part in it.
                                ffn_down_pqs: _,
                            } = &mut layer.ffn
                            {
                                for w in [ffn_gate, ffn_up, ffn_down] {
                                    mirk(el, w)?;
                                }
                            }
                        }
                        mirk(e_head, &mut output)?;
                        if n4 > 0 {
                            eprintln!(
                                "[{tag}] prefill fp16 mirrors built: {n4} tensors \
                                       ({} MB)",
                                b4 >> 20
                            );
                        }
                    }
                }
            }
        }
        // Q4_0 SPLIT-PLANE DECODE MIRRORS (2026-07-10, MEMRA_Q4RP seam): gemma-4 MoE-class trunk
        // (26B — attn wq/wk/wv/wo + the parallel shared FFN triple). The 18B GGUF block stride
        // costs ~25-35% decode bandwidth in sector overfetch (rp_q4_probe: m=1 1.34x, m=3 1.17x,
        // bitwise); the mirror (~0.7GB for the 26B) fixes the m<=8 mmvq/batched/fused family.
        // Dense 31B is NOT mirrored (its 15GB trunk mirror does not fit 24GB — the full layout
        // swap is the follow-up arc); raw bytes stay for prefill/gemm/Stage-A either way.
        if gemma_program && crate::Engine::q4rp_enabled() {
            let mut nmir = 0usize;
            for (il, layer) in layers.iter_mut().enumerate() {
                // M2 weight sharding: mirrors/concats build through the owning stage engine.
                let e = crate::pp::layer_engine(e, n_trunk, il)?;
                // 26B MoE-class trunk (moe_bits) OR the E4B dense trunk (e4b bits). E4B mirror
                // arithmetic: attn ~7.5MB/layer (shared layers skip wk/wv via build's no-op on
                // duplicate mirrors is NOT automatic — they alias the target's tensors as
                // separate GpuTensors, so their mirrors double ~1.5MB/shared-layer; acceptable)
                // + dense ffn 3 x 2560x10240 Q4_0 ~44MB + inp_gate/proj ~0.75MB => ~2.2GB for
                // the 5.2GB model; 24GB card holds model+mirror+KV with >14GB headroom.
                // Dense 31B stays unmirrored (15GB mirror does not fit) — its arm is the
                // layout-swap follow-up.
                let is_moe26 = layer.gemma4.as_ref().is_some_and(|g| g.moe_bits.is_some());
                let is_e4b = layer.gemma4.as_ref().is_some_and(|g| g.e4b.is_some());
                if !(is_moe26 || is_e4b) {
                    continue;
                }
                if let Mixer::Full(fa) = &mut layer.mixer {
                    for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] {
                        e.build_q4_rp4(w)?;
                        nmir += 1;
                    }
                }
                if is_e4b {
                    // wave-4b: own-KV layers get the wq|wk|wv OUT-concat (one matvec at t=1).
                    let own_kv = layer
                        .gemma4
                        .as_ref()
                        .unwrap()
                        .e4b
                        .as_ref()
                        .is_some_and(|e4| e4.kv_share.is_none());
                    if own_kv
                        && let Mixer::Full(fa) = &layer.mixer
                        && let Some(mut cat) = e.build_q4_out_concat3(&fa.wq, &fa.wk, &fa.wv)?
                    {
                        e.build_q4_rp4(&mut cat)?;
                        nmir += 1;
                        layer.gemma4.as_mut().unwrap().e4b.as_mut().unwrap().qkv_cat = Some(cat);
                    }
                    if let Ffn::Dense {
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        // memra#253: this site inspects or moves weights and runs no GEMM on an
                        // activation, so the AWQ activation-side scale plays no part in it.
                        ffn_down_pqs: _,
                    } = &mut layer.ffn
                    {
                        for w in [ffn_gate, ffn_up, ffn_down] {
                            e.build_q4_rp4(w)?;
                            nmir += 1;
                        }
                    }
                    let e4 = layer.gemma4.as_mut().unwrap().e4b.as_mut().unwrap();
                    for w in [&mut e4.inp_gate, &mut e4.proj] {
                        e.build_q4_rp4(w)?;
                        nmir += 1;
                    }
                }
                if let Some(mb) = layer.gemma4.as_mut().unwrap().moe_bits.as_mut() {
                    for w in [&mut mb.shared_gate, &mut mb.shared_up, &mut mb.shared_down] {
                        e.build_q4_rp4(w)?;
                        nmir += 1;
                    }
                }
            }
            if nmir > 0 {
                eprintln!("[q4rp] split-plane decode mirrors built: {nmir} trunk tensors");
            }
            // DENSE gemma (31B / E4B trunks): the trunk is too big to MIRROR on 24GB, so the
            // split layout replaces the GGUF bytes IN PLACE (zero steady-state VRAM; the 31B
            // profile put 76% of decode on the non-rp q4_0 matvecs). Every consumer routes
            // off the tensor's rp flag: mmvq/batched `_rp` twins + qmatvec_gemm_q4_0_rp
            // prefill. The Stage-A f32 oracle reads GGUF layout, so the swap is gated on the
            // fast path being active (MEMRA_FAST=0 keeps GGUF bytes end to end — exact oracle).
            let fast_on = std::env::var("MEMRA_FAST").as_deref() != Ok("0");
            if fast_on {
                let mut nswap = 0usize;
                let mut nf16 = 0usize;
                // f16 prefill mirrors (campaign A, 2026-07-31): built from the GGUF Q4_0
                // bytes BEFORE the in-place rp swap destroys that layout. Same Lt lane and
                // budget env as the qwen Q8_0 mirrors (MEMRA_PP_F16 / MEMRA_PP_F16_BUDGET_MB;
                // Hopper default ON, sm_120a default OFF — the 24GB card can't carry them).
                // Per-model (battery-keyed, 2026-07-31, REAL-prompt gates — the fox-repeat
                // family is layout-lottery degenerate and was retired from campaign gates):
                // 12B pp1736 8.3k -> 17.1k MATCH; 31B pp1736 4.8k -> 7.6k MATCH but ONLY
                // with the full-trunk mirror (420 tensors ~53GB — set
                // MEMRA_PP_F16_BUDGET_MB=57344 on 80GB boxes; the default 32GB partial
                // mirror measured FLAT there). MEMRA_Q4F16=1|0 forces either way.
                let q4f16_model_ok = matches!(cfg.n_embd, 3840 | 5376); // 12B | 31B geometry
                // Capacity-keyed default (zoo-fusion arc, 2026-08-17): with MEMRA_PP_F16
                // unset, admit the mirrors iff free VRAM covers the admissible f16 mass +
                // 8GiB serving headroom. The 31B downQ6K trunk's Q6_K ffn_down otherwise
                // rides the 3.46ms/call dequant-GEMM prefill wall (30% of c8 GPU time,
                // measured c8 agg +37% / ttft -70% with mirrors). Env keeps priority both
                // ways; 24GB rigs refuse by construction. Mirror mass = every 2D tensor
                // build_q8_f16 admits (Q8_0/Q4_0/Q6_K/Q4_K/Q5_K) in this walk.
                if let Ok(v) = std::env::var("MEMRA_Q4F16")
                    && v != "0"
                    && v != "1"
                {
                    return Err(format!(
                        "MEMRA_Q4F16={v} is not 0 or 1 — this env selects the prefill \
                             ARITHMETIC (fp16 mirrors vs int8 MMQ) and must never be guessed"
                    )
                    .into());
                }
                let f16_need = {
                    let f16b = |w: &crate::model::GpuTensor| -> usize {
                        match w {
                            crate::model::GpuTensor::Quant {
                                qtype,
                                ne,
                                f16: None,
                                ..
                            } if ne.len() == 2
                                && matches!(
                                    *qtype,
                                    crate::QT_Q8_0
                                        | crate::QT_Q4_0
                                        | crate::QT_Q6_K
                                        | crate::QT_Q4_K
                                        | crate::QT_Q5_K
                                ) =>
                            {
                                (ne[0] as usize) * (ne[1] as usize) * 2
                            }
                            _ => 0,
                        }
                    };
                    let mut need = 0usize;
                    for layer in layers.iter() {
                        if layer.gemma4.as_ref().is_none_or(|g| g.moe_bits.is_some()) {
                            continue;
                        }
                        if let Mixer::Full(fa) = &layer.mixer {
                            for w in [&fa.wq, &fa.wk, &fa.wv, &fa.wo] {
                                need += f16b(w);
                            }
                        }
                        if let Ffn::Dense {
                            ffn_gate,
                            ffn_up,
                            ffn_down,
                            // memra#253: this site inspects or moves weights and runs no GEMM on an
                            // activation, so the AWQ activation-side scale plays no part in it.
                            ffn_down_pqs: _,
                        } = &layer.ffn
                        {
                            for w in [ffn_gate, ffn_up, ffn_down] {
                                need += f16b(w);
                            }
                        }
                    }
                    need
                };
                let f16_free = e.ctx().mem_get_info().map(|(free, _)| free).unwrap_or(0);
                let f16_auto = q4f16_model_ok
                    && std::env::var("MEMRA_Q4F16").is_err()
                    && crate::f16_ffi::pp_f16_capacity_ok(f16_free, f16_need);
                // FOOTGUN FIX (lane/gemma-restore-exactness-20260819): the Ok("1") arm used to
                // be `pp_f16_enabled()`, which is FALSE unless MEMRA_PP_F16 is also set — so
                // MEMRA_Q4F16=1 silently disabled the mirrors it names. Measured on box2: =1
                // and =0 both produced the mirror-OFF greedy bytes (f985eb6a) while unset
                // produced the mirror-ON bytes (d966836a). Explicit =1 now means ON.
                let (f16_on, f16_why) = match std::env::var("MEMRA_Q4F16").as_deref() {
                    Ok("1") => (true, "env MEMRA_Q4F16=1"),
                    Ok("0") => (false, "env MEMRA_Q4F16=0"),
                    _ if crate::f16_ffi::pp_f16_enabled() && q4f16_model_ok => {
                        (true, "env MEMRA_PP_F16")
                    }
                    _ if f16_auto => (true, "capacity-keyed auto (UNPINNED)"),
                    _ if !q4f16_model_ok => (false, "model geometry not eligible"),
                    _ => (false, "capacity-keyed auto REFUSED (UNPINNED)"),
                };
                // The prefill program is a NUMERIC choice, not a perf knob: greedy output
                // bytes differ between the fp16-mirror and int8-MMQ prefill arms (measured,
                // research/gemma-load-cache-20260819/EXACTNESS.md — cold sha d966836a with
                // mirrors vs f985eb6a without, deterministic x2 each). It is therefore stated
                // unconditionally at boot, including the threshold it was decided against, so
                // a serving box's log records which arithmetic it is actually running.
                eprintln!(
                    "[q4f16] prefill program = {} (reason: {}); free {} MiB, mirror mass {} MiB, \
                     capacity threshold {} MiB (mass + 8192 headroom) — SELECTS PREFILL ARITHMETIC",
                    if f16_on {
                        "FP16 MIRRORS"
                    } else {
                        "INT8 MMQ (no f16 mirrors)"
                    },
                    f16_why,
                    f16_free >> 20,
                    f16_need >> 20,
                    (f16_need + (8usize << 30)) >> 20,
                );
                for (il, layer) in layers.iter_mut().enumerate() {
                    // M2 weight sharding: swap/mirror through the owning stage engine.
                    let e = crate::pp::layer_engine(e, n_trunk, il)?;
                    let dense_gemma = layer.gemma4.as_ref().is_some_and(|g| g.moe_bits.is_none());
                    if !dense_gemma {
                        continue;
                    }
                    if let Mixer::Full(fa) = &mut layer.mixer {
                        for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] {
                            if f16_on {
                                e.build_q8_f16(w)?;
                                if matches!(w, crate::model::GpuTensor::Quant { f16: Some(_), .. })
                                {
                                    nf16 += 1;
                                }
                            }
                            if e.build_q4_rp_swap(w)? {
                                nswap += 1;
                            }
                        }
                    }
                    if let Ffn::Dense {
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        // memra#253: this site inspects or moves weights and runs no GEMM on an
                        // activation, so the AWQ activation-side scale plays no part in it.
                        ffn_down_pqs: _,
                    } = &mut layer.ffn
                    {
                        for w in [ffn_gate, ffn_up, ffn_down] {
                            if f16_on {
                                e.build_q8_f16(w)?;
                                if matches!(w, crate::model::GpuTensor::Quant { f16: Some(_), .. })
                                {
                                    nf16 += 1;
                                }
                            }
                            if e.build_q4_rp_swap(w)? {
                                nswap += 1;
                            }
                        }
                    }
                }
                if nswap > 0 {
                    eprintln!("[q4rp] split-plane IN-PLACE swap: {nswap} dense trunk tensors");
                }
                if nf16 > 0 {
                    eprintln!("[q4f16] prefill fp16 mirrors built: {nf16} dense trunk tensors");
                }
            }
        }
        let model = HybridModel {
            cfg,
            plan,
            rewrite_qualifications: None,
            embd,
            output_norm,
            output,
            layers,
            mtp,
            mtp_extra,
            dflash_trim,
            frspec_src_sha16,
            embd_gpu: std::sync::OnceLock::new(),
            d2t_gpu: std::sync::OnceLock::new(),
            gemma4_aux,
            step35_aux,
            prime_slabs: std::sync::Mutex::new(std::collections::HashMap::new()),
            dspark_vgraphs: std::sync::Mutex::new(None),
            step_grouped_prefill: std::sync::Mutex::new(StepEpGroupedPrefill::default()),
            step35_token_graph: std::sync::Mutex::new(None),
            hyper,
            hyper_head,
            glm5_dflash,
            draft_state_bytes: std::sync::atomic::AtomicUsize::new(0),
            test_extra_devices: Vec::new(),
        };
        e.configure_moe_cache_layout(model.moe_cache_block_sizes());
        if force_embd_gpu {
            let _ = model
                .embd_gpu
                .get_or_init(|| e.upload_u8(&model.embd.raw).expect("embed table upload"));
        }
        // M2 LOAD BARRIER (pp door open at load): uploads + mirror builds above ran on
        // the loading engines' worker streams; the first decode consumer runs on OTHER
        // streams with no event between them. Synchronize every stage context once so
        // no consumer can ever read a half-built tensor (the 2026-08-02 split5 ref=0.0
        // head-mirror find). No-op with the door shut.
        crate::pp::sync_stages_after_load(e, n_trunk)?;
        Ok(model)
    }

    /// Force the device embed table resident, FALLIBLY (F5 right-size ladder,
    /// 2026-08-05). The lazy `embd_gpu.get_or_init(.. expect ..)` sites panic the
    /// GPU worker on OOM; on a VRAM-tight rig a right-sized spec session that
    /// "fits" can leave too little for this ~hundreds-of-MB upload and die on its
    /// first prefill (observed: research/specpool-20260804/server-ladder-miss.log).
    /// The server calls this after each ladder landing so the biggest lazy
    /// transient surfaces as a catchable Err (shrink further / fall back) instead
    /// of a panic. No-op when the table is already resident.
    pub fn ensure_embed_resident(&self, e: &Engine) -> Result<(), Box<dyn std::error::Error>> {
        if self.embd_gpu.get().is_none() {
            let buf = e.upload_u8(&self.embd.raw)?;
            let _ = self.embd_gpu.set(buf); // racing set = already resident; fine
        }
        Ok(())
    }

    pub fn embed(
        &self,
        e: &Engine,
        tokens: &[u32],
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        // DEVICE embed gather (round 30; the gemma4 machinery adopted for every model):
        // resident quantized table + gather kernel — replaces the CPU row gather + 31MB
        // pageable HtoD (2.2ms at T=2048, the lane's largest host stall). Same d*q
        // dequant math as the CPU gather; the greedy-stream A/B arbitrated it.
        let tbl = self
            .embd_gpu
            .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload"));
        let tok_d = e.htod_u32_v(tokens)?;
        let (qt, rb) = self.embd.qt_and_row_bytes(n_embd);
        e.embed_gather_device_td(tbl, &tok_d, tokens.len(), n_embd, qt, rb)
    }
}

fn illegal_pipeline_cuts(fence: &[usize], legal_boundaries: &[usize]) -> Vec<usize> {
    fence
        .get(1..fence.len().saturating_sub(1))
        .unwrap_or_default()
        .iter()
        .copied()
        .filter(|cut| !legal_boundaries.contains(cut))
        .collect()
}

#[cfg(test)]
mod pipeline_cut_tests {
    use super::illegal_pipeline_cuts;

    #[test]
    fn manual_pipeline_cuts_cannot_bypass_model_plan_boundaries() {
        assert!(illegal_pipeline_cuts(&[0, 8, 16, 24], &[8, 16]).is_empty());
        assert_eq!(illegal_pipeline_cuts(&[0, 7, 16, 24], &[8, 16]), vec![7]);
        assert_eq!(
            illegal_pipeline_cuts(&[0, 7, 15, 24], &[8, 16]),
            vec![7, 15]
        );
    }
}

#[cfg(test)]
mod auto_parallel_policy_tests {
    use super::{
        parse_auto_parallel_tp_attention, parse_auto_parallel_tp_attention_ranks,
        parse_auto_w4a16_bf16_mmv,
    };

    #[test]
    fn automatic_w4a16_bf16_residency_defaults_on_with_explicit_rollback() {
        assert!(parse_auto_w4a16_bf16_mmv(None).unwrap());
        assert!(!parse_auto_w4a16_bf16_mmv(Some("0")).unwrap());
        assert!(parse_auto_w4a16_bf16_mmv(Some("1")).unwrap());
        assert!(parse_auto_w4a16_bf16_mmv(Some("true")).is_err());
        assert!(parse_auto_w4a16_bf16_mmv(Some("")).is_err());
    }

    #[test]
    fn automatic_tp_attention_is_strict_and_defaults_off() {
        assert!(!parse_auto_parallel_tp_attention(None).unwrap());
        assert!(!parse_auto_parallel_tp_attention(Some("")).unwrap());
        assert!(!parse_auto_parallel_tp_attention(Some("0")).unwrap());
        assert!(parse_auto_parallel_tp_attention(Some("1")).unwrap());
        assert!(parse_auto_parallel_tp_attention(Some("true")).is_err());
        assert!(parse_auto_parallel_tp_attention(Some("2")).is_err());
    }

    #[test]
    fn automatic_tp_attention_rank_count_is_explicit_and_bounded() {
        assert_eq!(parse_auto_parallel_tp_attention_ranks(None).unwrap(), None);
        assert_eq!(
            parse_auto_parallel_tp_attention_ranks(Some("2")).unwrap(),
            Some(2)
        );
        assert_eq!(
            parse_auto_parallel_tp_attention_ranks(Some("3")).unwrap(),
            Some(3)
        );
        assert_eq!(
            parse_auto_parallel_tp_attention_ranks(Some("4")).unwrap(),
            Some(4)
        );
        for bad in ["", "0", "1", "5", "all"] {
            assert!(parse_auto_parallel_tp_attention_ranks(Some(bad)).is_err());
        }
    }
}

#[cfg(test)]
mod step_expert_selection_tests {
    use super::{
        StepExpertArtifact, StepExpertLayout, StepParallelLoadConfig, StepParallelRuntimeRegistry,
        StepTpAttentionPlacement, select_step_expert_layout, select_step_expert_layout_inner,
    };
    use crate::tp::StepEpLayerSpec;

    fn spec(layer: usize, ranks: usize) -> StepEpLayerSpec {
        StepEpLayerSpec {
            layer,
            devices: (0..ranks).collect(),
        }
    }

    #[test]
    fn tp2_keeps_projection_sharded_experts() {
        let selection = select_step_expert_layout(24, &[], &[spec(24, 2)])
            .unwrap()
            .unwrap();
        assert_eq!(selection.layout, StepExpertLayout::TensorParallel);
        assert!(selection.configured_by_tp);
    }

    #[test]
    fn tp4_and_tp8_use_expert_ownership_without_a_second_flag() {
        for ranks in [4, 8] {
            let selection = select_step_expert_layout(24, &[], &[spec(24, ranks)])
                .unwrap()
                .unwrap();
            assert_eq!(selection.layout, StepExpertLayout::ExpertParallel);
            assert!(selection.configured_by_tp);
            assert_eq!(selection.spec.devices.len(), ranks);
        }
    }

    #[test]
    fn explicit_ep_remains_expert_parallel() {
        let selection = select_step_expert_layout(24, &[spec(24, 2)], &[])
            .unwrap()
            .unwrap();
        assert_eq!(selection.layout, StepExpertLayout::ExpertParallel);
        assert!(!selection.configured_by_tp);
    }

    #[test]
    fn conflicting_ep_and_tp_assignments_fail_closed() {
        let error = select_step_expert_layout(24, &[spec(24, 4)], &[spec(24, 4)]).unwrap_err();
        assert!(error.contains("cannot enable MEMRA_STEP_EP and MEMRA_STEP_TP together"));
    }

    #[test]
    fn automatic_tp2_attention_can_overlap_ep4_expert_ownership() {
        let selection = select_step_expert_layout_inner(24, &[spec(24, 4)], &[spec(24, 2)], true)
            .unwrap()
            .unwrap();
        assert_eq!(selection.layout, StepExpertLayout::ExpertParallel);
        assert!(!selection.configured_by_tp);
        assert_eq!(selection.spec.devices, vec![0, 1, 2, 3]);

        let error =
            select_step_expert_layout_inner(24, &[spec(24, 4)], &[spec(24, 2)], false).unwrap_err();
        assert!(error.contains("cannot enable MEMRA_STEP_EP and MEMRA_STEP_TP together"));
    }

    #[test]
    fn runtime_registry_owns_one_immutable_load_snapshot() {
        let mut source_specs = vec![spec(24, 8)];
        let registry = StepParallelRuntimeRegistry::with_config(StepParallelLoadConfig {
            ep_specs: Vec::new(),
            tp_specs: source_specs.clone(),
            native_p2p: true,
            ep_device_arithmetic: true,
            f32_mirror: true,
            bulk_p2p: true,
            nvfp4_device_routes: true,
            auto_parallel: true,
            tp_attention_expert_overlap: false,
            expert_artifact: StepExpertArtifact::default(),
        });
        source_specs[0].devices.clear();

        let stored = registry.tp_spec(24).unwrap();
        assert_eq!(stored.devices, (0..8).collect::<Vec<_>>());
        assert!(registry.config.native_p2p);
        assert!(registry.config.ep_device_arithmetic);
        assert!(registry.config.f32_mirror);
        assert!(registry.config.bulk_p2p);
        assert!(registry.config.nvfp4_device_routes);
        assert!(registry.config.auto_parallel);
        assert_eq!(
            registry.expert_selection(24).unwrap().unwrap().layout,
            StepExpertLayout::ExpertParallel
        );

        let standalone = StepParallelRuntimeRegistry::default();
        assert!(standalone.tp_spec(24).is_none());
        assert!(!standalone.config.native_p2p);
        assert!(!standalone.config.ep_device_arithmetic);
        assert!(!standalone.config.f32_mirror);
        assert!(!standalone.config.bulk_p2p);
    }

    #[test]
    fn rank_local_attention_uses_bounded_swa_rings_only_with_native_p2p() {
        assert_eq!(
            StepTpAttentionPlacement::resolve(true, None),
            StepTpAttentionPlacement::RankLocalGlobal
        );
        assert_eq!(
            StepTpAttentionPlacement::resolve(true, Some(512)),
            StepTpAttentionPlacement::RankLocalSwa
        );
        assert_eq!(
            StepTpAttentionPlacement::resolve(false, None),
            StepTpAttentionPlacement::OwnerTransportFallback
        );
        assert_eq!(
            StepTpAttentionPlacement::resolve(false, Some(512)),
            StepTpAttentionPlacement::OwnerSwa
        );
    }
}

#[cfg(test)]
mod residency_tests {
    use super::{DevExpertFp8ProjectionScales, ResidentPlan, residency_bytes_by_device};
    use crate::model::HostExpertFp8BlockScales;
    use std::collections::HashMap;

    #[test]
    fn pp_residency_counts_only_each_devices_expert_slice() {
        let tensors = [
            ("blk.0.ffn_gate_exps.weight", 10usize),
            ("blk.0.ffn_up_exps.weight", 20),
            ("blk.1.ffn_down_exps.weight", 30),
            ("blk.2.ffn_gate_exps.weight", 40),
            ("blk.3.ffn_up_exps.weight", 50),
            ("blk.0.attn_q.weight", 7),
            ("output.weight", 11),
        ];
        let bytes = residency_bytes_by_device(tensors, &[0, 0, 1, 1], 0);
        assert_eq!(bytes.experts.get(&0), Some(&60));
        assert_eq!(bytes.experts.get(&1), Some(&90));
        assert_eq!(bytes.rest, 18);
        assert!(bytes.saw_experts);
    }

    #[test]
    fn pp_residency_combines_stages_that_share_one_device() {
        let tensors = [
            ("blk.0.ffn_gate_exps.weight", 10usize),
            ("blk.1.ffn_gate_exps.weight", 20),
            ("blk.2.ffn_gate_exps.weight", 30),
            ("blk.3.ffn_gate_exps.weight", 40),
        ];
        let bytes = residency_bytes_by_device(tensors, &[0, 0, 0, 0], 0);
        assert_eq!(bytes.experts.get(&0), Some(&100));
        assert_eq!(bytes.experts.len(), 1);
    }

    #[test]
    fn distributed_trunk_layers_do_not_poison_local_mtp_residency_estimates() {
        let mut plan = ResidentPlan {
            primary_device: 0,
            layer_devices: vec![0; 81],
            layer_counts: HashMap::from([(0, 81)]),
            exact_expert_bytes: None,
            trunk_bytes: 0,
            decisions: HashMap::new(),
            pp: false,
        };
        plan.exclude_distributed_expert_layers(1..80);
        assert_eq!(plan.layer_counts.get(&0), Some(&2));
    }

    #[test]
    fn resident_fp8_scale_slab_must_match_every_expert() {
        let valid = HostExpertFp8BlockScales {
            scales: vec![1.0; 12],
            rows: 2,
            cols: 3,
            expert_stride: 6,
        };
        DevExpertFp8ProjectionScales::validate(&valid, 2).unwrap();

        let short = HostExpertFp8BlockScales {
            scales: vec![1.0; 11],
            ..valid
        };
        assert_eq!(
            DevExpertFp8ProjectionScales::validate(&short, 2).unwrap_err(),
            "block-E4M3 scale slab length mismatch: got 11, want 2x6=12"
        );
    }

    #[test]
    fn resident_fp8_scale_stride_must_match_its_grid() {
        let invalid = HostExpertFp8BlockScales {
            scales: vec![1.0; 8],
            rows: 2,
            cols: 2,
            expert_stride: 0,
        };
        assert_eq!(
            DevExpertFp8ProjectionScales::validate(&invalid, 2).unwrap_err(),
            "block-E4M3 expert scale stride must be nonzero"
        );
    }
}

#[cfg(test)]
mod draft_head_tests {
    use super::{draft_head_tensor, frspec_trim_own_head_name};

    /// Names present in the real Step-3.7-Flash MTP drafter (Step3.7-flash-mtp-Q8_0.gguf), as
    /// enumerated by the on-disk byte probe in
    /// research/step37-p2-20260806/raw/draft-head-tensor-hashes-20260807.txt.
    /// Both candidate heads exist in that file with IDENTICAL [4096, 128896] Q8_0 shape, so no
    /// shape or dtype check can distinguish them — only the sha256 of the payload could, and it
    /// showed them to be different matrices (blk.45 head c90b907b… vs output.weight 3eec5831…).
    const STEP37_DRAFTER: &[&str] = &[
        "output.weight",
        "output_norm.weight",
        "token_embd.weight",
        "blk.45.nextn.shared_head_norm.weight",
        "blk.45.nextn.shared_head_head.weight",
        "blk.46.nextn.shared_head_head.weight",
        "blk.47.nextn.shared_head_head.weight",
    ];

    fn present(names: &'static [&'static str]) -> impl Fn(&str) -> bool {
        move |t: &str| names.contains(&t)
    }

    /// THE REGRESSION. Reading `output.weight` off this drafter cost acceptance 0/248 across
    /// K=1..8 with self-consistency PASS at every K — correct output, dead speculation, no gate
    /// red (raw/mtp-draft-20260806T212902Z.log). The drafter's top-level output stack is a
    /// re-quantized COPY OF THE TRUNK'S (its output_norm is byte-identical to the trunk's,
    /// d7526f44…), so it is the standalone-decode head, not the MTP head. Preferring
    /// blk.45.nextn.shared_head_head took K=1 to 14/18 = 77.8%
    /// (raw/mtp-draft-PASS-20260806T215132Z.log).
    #[test]
    fn step37_drafter_prefers_the_blocks_own_nextn_head_over_file_level_output() {
        assert_eq!(
            draft_head_tensor(present(STEP37_DRAFTER), 45),
            "blk.45.nextn.shared_head_head.weight"
        );
    }

    /// Each NextN block owns a DIFFERENT head (c90b907b / a22d2957 / 4b21e137 — a shared head
    /// would have collided), so the name must be built from the block index, never hardcoded.
    /// This is what multi-block chaining (45->46->47) will index when it lands.
    #[test]
    fn each_nextn_block_selects_its_own_head() {
        for n in 45..=47u32 {
            assert_eq!(
                draft_head_tensor(present(STEP37_DRAFTER), n),
                format!("blk.{n}.nextn.shared_head_head.weight")
            );
        }
    }

    /// FR-Spec / tied-head drafts publish the (possibly vocab-trimmed) head as the file-level
    /// `output.weight` and ship no nextn head. They must keep working — hence preference, not
    /// replacement.
    #[test]
    fn draft_without_a_nextn_head_falls_back_to_file_level_output() {
        let fr_spec: &[&str] = &["output.weight", "output_norm.weight", "d2t.weight"];
        assert_eq!(draft_head_tensor(present(fr_spec), 45), "output.weight");
    }

    /// The legacy `nextn.shared_head` probe sits between the two: no shipped artifact and no
    /// upstream mapping uses it (upstream is LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD ->
    /// "blk.%d.nextn.shared_head_head"), but anything that ever matched it still must, and it
    /// must never win over the real name.
    #[test]
    fn legacy_shared_head_is_probed_but_loses_to_shared_head_head() {
        let legacy_only: &[&str] = &["output.weight", "blk.45.nextn.shared_head.weight"];
        assert_eq!(
            draft_head_tensor(present(legacy_only), 45),
            "blk.45.nextn.shared_head.weight"
        );

        let both: &[&str] = &[
            "output.weight",
            "blk.45.nextn.shared_head.weight",
            "blk.45.nextn.shared_head_head.weight",
        ];
        assert_eq!(
            draft_head_tensor(present(both), 45),
            "blk.45.nextn.shared_head_head.weight"
        );
    }

    /// A drafter whose nextn head belongs to a DIFFERENT block must not be borrowed: asking for
    /// block 45 in a file that only carries 46/47 falls back rather than silently mismatching
    /// the geometry the trunk verified against.
    #[test]
    fn a_different_blocks_nextn_head_is_never_borrowed() {
        let wrong_block: &[&str] = &[
            "output.weight",
            "blk.46.nextn.shared_head_head.weight",
            "blk.47.nextn.shared_head_head.weight",
        ];
        assert_eq!(draft_head_tensor(present(wrong_block), 45), "output.weight");
    }

    /// The FR-Spec trim must gather from the nextn block's OWN head on step-3.7-flash. Reading
    /// the trunk head there is the 0/248-acceptance defect that self-consistency does not
    /// catch, so the name this helper builds is pinned rather than left to a format! call
    /// sitting inline in a 400-line loader arm.
    #[test]
    fn frspec_trim_prefers_the_nextn_blocks_own_head_name() {
        assert_eq!(
            frspec_trim_own_head_name(45),
            "blk.45.nextn.shared_head_head.weight"
        );
        // Same shape the loader's own draft-head preference uses, so the two cannot drift.
        assert_eq!(
            frspec_trim_own_head_name(45),
            format!("blk.{}.nextn.shared_head_head.weight", 45)
        );
        assert_eq!(
            frspec_trim_own_head_name(40),
            "blk.40.nextn.shared_head_head.weight"
        );
    }
}

#[cfg(test)]
mod mla_indexer_geom_tests {
    use super::MlaIndexerGeom;

    /// The live selector's audit is `width_cap >= select_k_cap * pool + pool - 1`, so the cap a
    /// live launch passes must be the one `index_width` sized the row from. Swept across
    /// capacities either side of the pool budget, including the served glm5 shape that broke
    /// (top_k 2048, pool 4, a ctx=386 session = 96 capacity pools).
    #[test]
    fn live_select_k_cap_matches_index_width() {
        let g = MlaIndexerGeom {
            heads: 32,
            head_dim: 128,
            top_k: 2048,
            pool: 4,
            always_select_tail: true,
        };
        for pools in [1usize, 96, 511, 512, 513, 4096, 262_144] {
            let width = g.index_width(pools);
            let cap = g.live_select_k_cap(pools);
            assert!(
                width >= cap * g.pool + g.pool - 1,
                "live audit fails at {pools} pools: width {width} < cap {cap} * pool {} + {}",
                g.pool,
                g.pool - 1
            );
        }
    }

    /// RED ARM: the value the call site used to pass. Below the pool budget the unclamped
    /// `top_k / pool` is strictly larger than the width can support, which is the 40014 the
    /// served boot hit; above it the two agree, which is why every gate and every long session
    /// passed and this reached the pair.
    #[test]
    fn unclamped_cap_is_what_the_audit_rejects() {
        let g = MlaIndexerGeom {
            heads: 32,
            head_dim: 128,
            top_k: 2048,
            pool: 4,
            always_select_tail: true,
        };
        let unclamped = g.top_k / g.pool;
        let short = 96; // ctx 386 / pool 4, the boot that failed
        assert!(
            g.index_width(short) < unclamped * g.pool + g.pool - 1,
            "the short-capacity case must be the one the audit rejects"
        );
        let long = 262_144;
        assert_eq!(
            g.live_select_k_cap(long),
            unclamped,
            "at capacity the clamped and unclamped caps agree, which is why this hid"
        );
    }
}
